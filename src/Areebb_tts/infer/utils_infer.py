# A unified script for inference process
# Make adjustments inside functions, and consider both gradio and cli scripts if need to change func output format
import os
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor


os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # for MPS device compatibility
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../../third_party/BigVGAN/")

import re

import numpy as np
import torch
import torchaudio
import tqdm

from Areebb_tts.model.utils import text_list_formatter


device = (
    "cuda"
    if torch.cuda.is_available()
    else "xpu"
    if torch.xpu.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# -----------------------------------------

target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024
mel_spec_type = "vocos"
target_rms = 0.1
cross_fade_duration = 0.15
ode_method = "euler"
nfe_step = 32  # 16, 32
cfg_strength = 2.0
sway_sampling_coef = -1.0
speed = 1.0
fix_duration = None

# -----------------------------------------


# chunk text into smaller pieces

# Latin + Arabic sentence punctuation (ASCII comma alone is not enough for Arabic LLM output).
_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[;:,.!?\u060c\u061b\u061f])\s+|(?<=[；：，。！？])"
)
def _chunk_has_letter_or_digit(t: str) -> bool:
    """True if chunk contains a real letter or digit (Arabic comma U+060C is Po, not Lo)."""
    for ch in t:
        if ch.isspace():
            continue
        cat = unicodedata.category(ch)
        if cat.startswith("L") or cat == "Nd":
            return True
    return False


def _merge_letterless_prefix_chunks(chunks: list[str]) -> list[str]:
    """Glue punctuation-only fragments onto the following chunk so they are not synthesized alone."""
    out: list[str] = []
    pending = ""
    for c in chunks:
        t = c.strip()
        if not t:
            continue
        short = len(t.encode("utf-8")) < 48
        if short and not _chunk_has_letter_or_digit(t):
            pending = f"{pending} {t}".strip() if pending else t
            continue
        if pending:
            out.append(f"{pending} {t}".strip())
            pending = ""
        else:
            out.append(c.strip())
    if pending:
        if out:
            out[-1] = f"{out[-1]} {pending}".strip()
        else:
            out.append(pending)
    return [x for x in out if x.strip()]


def _split_utf8_byte_budget(text: str, max_chars: int) -> list[str]:
    """Split so every piece has UTF-8 byte length <= max_chars (character-safe)."""
    if max_chars < 8:
        max_chars = 8
    out: list[str] = []
    cur = ""
    for ch in text:
        cand = cur + ch
        if len(cand.encode("utf-8")) <= max_chars:
            cur = cand
        else:
            if cur:
                out.append(cur)
            cur = ch
    if cur:
        out.append(cur)
    return out


def _split_oversized_chunk(text: str, max_chars: int) -> list[str]:
    """Prefer word boundaries; fall back to byte-budget character walk for long tokens."""
    text = text.strip()
    if not text or len(text.encode("utf-8")) <= max_chars:
        return [text] if text else []

    parts: list[str] = []
    buf = ""
    for word in text.split():
        trial = f"{buf} {word}".strip() if buf else word
        if len(trial.encode("utf-8")) <= max_chars:
            buf = trial
        else:
            if buf:
                parts.append(buf)
            wbytes = len(word.encode("utf-8"))
            if wbytes <= max_chars:
                buf = word
            else:
                parts.extend(_split_utf8_byte_budget(word, max_chars))
                buf = ""
    if buf:
        parts.append(buf)
    return parts


def chunk_text(text, max_chars=135):
    """
    Splits the input text into chunks, each with a maximum number of characters.

    Args:
        text (str): The text to be split.
        max_chars (int): The maximum number of characters per chunk.

    Returns:
        List[str]: A list of text chunks.
    """
    if max_chars < 8:
        max_chars = 8

    chunks = []
    current_chunk = ""
    sentences = _SENTENCE_SPLIT_RE.split(text)

    for sentence in sentences:
        if len(current_chunk.encode("utf-8")) + len(sentence.encode("utf-8")) <= max_chars:
            current_chunk += sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    # Long Arabic (or unpunctuated) paragraphs used to stay as one batch and get truncated in the model.
    expanded: list[str] = []
    for c in chunks:
        if not c:
            continue
        if len(c.encode("utf-8")) <= max_chars:
            expanded.append(c)
        else:
            expanded.extend(_split_oversized_chunk(c, max_chars))
    expanded = [x for x in expanded if x.strip()]
    return _merge_letterless_prefix_chunks(expanded)


# infer process: chunk text -> infer batches [i.e. infer_batch_process()]


def infer_process(
    ref_audio,
    ref_text,
    gen_text,
    model_obj,
    vocoder,
    mel_spec_type=mel_spec_type,
    show_info=print,
    progress=tqdm,
    target_rms=target_rms,
    cross_fade_duration=cross_fade_duration,
    nfe_step=nfe_step,
    cfg_strength=cfg_strength,
    sway_sampling_coef=sway_sampling_coef,
    speed=speed,
    fix_duration=fix_duration,
    device=device,
    dialect_id=None,
):
    # Split the input text into batches
    audio, sr = torchaudio.load(ref_audio)
    ref_dur = float(audio.shape[-1]) / float(sr)
    # ~22s total window heuristic; keep budget positive for long reference clips.
    gen_budget_sec = max(0.5, 22.0 - ref_dur)
    ref_text_bytes = max(len(ref_text.encode("utf-8")), 1)
    max_chars = int(ref_text_bytes / max(ref_dur, 1e-3) * gen_budget_sec * speed)
    max_chars = max(40, max_chars)
    gen_text_batches = chunk_text(gen_text, max_chars=max_chars)
    for i, gen_text in enumerate(gen_text_batches):
        print(f"gen_text {i}", gen_text)
    print("\n")

    show_info(f"Generating audio in {len(gen_text_batches)} batches...")
    return next(
        infer_batch_process(
            (audio, sr),
            ref_text,
            gen_text_batches,
            model_obj,
            vocoder,
            mel_spec_type=mel_spec_type,
            progress=progress,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
            device=device,
            dialect_id=dialect_id,
        )
    )


# infer batches


def infer_batch_process(
    ref_audio,
    ref_text,
    gen_text_batches,
    model_obj,
    vocoder,
    mel_spec_type="vocos",
    progress=tqdm,
    target_rms=0.1,
    cross_fade_duration=0.15,
    nfe_step=32,
    cfg_strength=2.0,
    sway_sampling_coef=-1,
    speed=1,
    fix_duration=None,
    device=None,
    streaming=False,
    chunk_size=2048,
    dialect_id=None,
):
    audio, sr = ref_audio
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < target_rms:
        audio = audio * target_rms / rms
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)
    audio = audio.to(device)

    generated_waves = []
    spectrograms = []

    if len(ref_text[-1].encode("utf-8")) == 1:
        ref_text = ref_text + " "

    def process_batch(gen_text):
        local_speed = speed
        if len(gen_text.encode("utf-8")) < 10:
            local_speed = 0.3

        # Prepare the text
        text_list = [ref_text + gen_text]
        final_text_list = text_list_formatter(text_list, dialect_id=dialect_id)

        ref_audio_len = audio.shape[-1] // hop_length
        if fix_duration is not None:
            duration = int(fix_duration * target_sample_rate / hop_length)
        else:
            # Calculate duration
            ref_text_len = len(ref_text.encode("utf-8"))
            gen_text_len = len(gen_text.encode("utf-8"))
            duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / local_speed)

        # inference
        with torch.inference_mode():
            generated, _ = model_obj.sample(
                cond=audio,
                text=final_text_list,
                duration=duration,
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
            )
            del _

            generated = generated.to(torch.float32)  # generated mel spectrogram
            generated = generated[:, ref_audio_len:, :]
            generated = generated.permute(0, 2, 1)
            if mel_spec_type == "vocos":
                generated_wave = vocoder.decode(generated)
            elif mel_spec_type == "bigvgan":
                generated_wave = vocoder(generated)
            if rms < target_rms:
                generated_wave = generated_wave * rms / target_rms

            # wav -> numpy
            generated_wave = generated_wave.squeeze().cpu().numpy()

            if streaming:
                for j in range(0, len(generated_wave), chunk_size):
                    yield generated_wave[j : j + chunk_size], target_sample_rate
            else:
                generated_cpu = generated[0].cpu().numpy()
                del generated
                yield generated_wave, generated_cpu

    if streaming:
        for gen_text in progress.tqdm(gen_text_batches) if progress is not None else gen_text_batches:
            for chunk in process_batch(gen_text):
                yield chunk
    else:
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_batch, gen_text) for gen_text in gen_text_batches]
            for future in progress.tqdm(futures) if progress is not None else futures:
                result = future.result()
                if result:
                    generated_wave, generated_mel_spec = next(result)
                    generated_waves.append(generated_wave)
                    spectrograms.append(generated_mel_spec)

        if generated_waves:
            if cross_fade_duration <= 0:
                # Simply concatenate
                final_wave = np.concatenate(generated_waves)
            else:
                # Combine all generated waves with cross-fading
                final_wave = generated_waves[0]
                for i in range(1, len(generated_waves)):
                    prev_wave = final_wave
                    next_wave = generated_waves[i]

                    # Calculate cross-fade samples, ensuring it does not exceed wave lengths
                    cross_fade_samples = int(cross_fade_duration * target_sample_rate)
                    cross_fade_samples = min(cross_fade_samples, len(prev_wave), len(next_wave))

                    if cross_fade_samples <= 0:
                        # No overlap possible, concatenate
                        final_wave = np.concatenate([prev_wave, next_wave])
                        continue

                    # Short prev (e.g. punctuation-only chunk) makes prev_wave[:-cross] empty or tiny and
                    # masks the start of next_wave inside the overlap window.
                    if len(prev_wave) <= cross_fade_samples or len(next_wave) <= cross_fade_samples:
                        final_wave = np.concatenate([prev_wave, next_wave])
                        continue

                    # Overlapping parts
                    prev_overlap = prev_wave[-cross_fade_samples:]
                    next_overlap = next_wave[:cross_fade_samples]

                    # Fade out and fade in
                    fade_out = np.linspace(1, 0, cross_fade_samples)
                    fade_in = np.linspace(0, 1, cross_fade_samples)

                    # Cross-faded overlap
                    cross_faded_overlap = prev_overlap * fade_out + next_overlap * fade_in

                    # Combine
                    new_wave = np.concatenate(
                        [prev_wave[:-cross_fade_samples], cross_faded_overlap, next_wave[cross_fade_samples:]]
                    )

                    final_wave = new_wave

            # Create a combined spectrogram
            combined_spectrogram = np.concatenate(spectrograms, axis=1)

            yield final_wave, target_sample_rate, combined_spectrogram

        else:
            yield None, target_sample_rate, None

