import argparse
import logging
import random
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from cached_path import cached_path
from f5_tts.infer.utils_infer import load_model, load_vocoder, preprocess_ref_audio_text
from f5_tts.model import DiT

LOGGER = logging.getLogger("habibi_interactive_cli")

# Project-aligned defaults (same baseline used by app.py / utils_infer).
DEFAULT_CFG_STRENGTH = 2.0
DEFAULT_NFE_STEP = 32
DEFAULT_SPEED = 1.0
DEFAULT_TARGET_RMS = 0.1
DEFAULT_CROSS_FADE = 0.15
DEFAULT_SWAY = -1.0

DEFAULT_MODEL_REPO = "SWivid/Habibi-TTS"
UAE_DIALECT = "UAE"
DEFAULT_MODEL_STEP = 100000
DEFAULT_OUTPUT_FILE = "clean.wav"
DEFAULT_REF_AUDIO = "assets/UAE.wav"
DEFAULT_REF_TEXT = "قمنا نشتريها بشكل متكرر أو لما نلقى ستايل يعجبنا."
TARGET_SAMPLE_RATE = 24000
HOP_LENGTH = 256

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[;:,.!?\u060c\u061b\u061f])\s+|(?<=[؛:،.!?])\s*")


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean interactive Habibi-TTS inference CLI.")
    parser.add_argument("--ref-audio", type=Path, default=Path(DEFAULT_REF_AUDIO), help="Path to reference voice WAV/MP3 file.")
    parser.add_argument("--ref-text", type=str, default=DEFAULT_REF_TEXT, help="Transcript of reference audio.")
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT_FILE), help="Output WAV path.")
    parser.add_argument("--model-repo", type=str, default=DEFAULT_MODEL_REPO, help="HF repo name for checkpoints.")
    parser.add_argument("--model-step", type=int, default=DEFAULT_MODEL_STEP, help="Checkpoint step.")
    parser.add_argument("--seed", type=int, default=-1, help=">=0 for deterministic output. -1 for random.")
    parser.add_argument("--target-rms", type=float, default=DEFAULT_TARGET_RMS)
    parser.add_argument("--cross-fade-duration", type=float, default=DEFAULT_CROSS_FADE)
    parser.add_argument("--nfe-step", type=int, default=DEFAULT_NFE_STEP)
    parser.add_argument("--cfg-strength", type=float, default=DEFAULT_CFG_STRENGTH)
    parser.add_argument("--sway-sampling-coef", type=float, default=DEFAULT_SWAY)
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED)
    parser.add_argument("--fix-duration", type=float, default=None)
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.ref_audio.is_file():
        raise FileNotFoundError(f"Reference audio not found: {args.ref_audio}")
    if not args.ref_text.strip():
        raise ValueError("ref-text cannot be empty.")
    if args.speed <= 0:
        raise ValueError("speed must be > 0.")
    if args.nfe_step <= 0:
        raise ValueError("nfe-step must be > 0.")
    if args.target_rms <= 0:
        raise ValueError("target-rms must be > 0.")


def set_seed(seed: int) -> None:
    if seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _chunk_has_letter_or_digit(text: str) -> bool:
    for ch in text:
        if ch.isalpha() or ch.isdigit():
            return True
    return False


def _merge_letterless_prefix_chunks(chunks: list[str]) -> list[str]:
    out: list[str] = []
    pending = ""
    for chunk in chunks:
        token = chunk.strip()
        if not token:
            continue
        short = len(token.encode("utf-8")) < 48
        if short and not _chunk_has_letter_or_digit(token):
            pending = f"{pending} {token}".strip() if pending else token
            continue
        if pending:
            out.append(f"{pending} {token}".strip())
            pending = ""
        else:
            out.append(token)
    if pending:
        if out:
            out[-1] = f"{out[-1]} {pending}".strip()
        else:
            out.append(pending)
    return [x for x in out if x.strip()]


def resolve_paths(model_repo: str, model_step: int) -> tuple[str, str]:
    model_path = str(cached_path(f"hf://{model_repo}/Specialized/{UAE_DIALECT}/model_{model_step}.safetensors"))
    vocab_path = str(cached_path(f"hf://{model_repo}/Specialized/{UAE_DIALECT}/vocab.txt"))
    return model_path, vocab_path


def load_runtime(model_path: str, vocab_path: str, device: str) -> tuple[torch.nn.Module, torch.nn.Module]:
    model = load_model(
        DiT,
        dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
        model_path,
        vocab_file=vocab_path,
        device=device,
    )
    vocoder = load_vocoder(device=device)
    return model, vocoder


def chunk_text(text: str, max_chars: int = 135) -> list[str]:
    if max_chars < 12:
        max_chars = 12
    chunks: list[str] = []
    current = ""
    for sentence in _SENTENCE_SPLIT_RE.split(text.strip()):
        if not sentence:
            continue
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate.encode("utf-8")) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(sentence.encode("utf-8")) <= max_chars:
                current = sentence
            else:
                word_buffer = ""
                for word in sentence.split():
                    trial = f"{word_buffer} {word}".strip() if word_buffer else word
                    if len(trial.encode("utf-8")) <= max_chars:
                        word_buffer = trial
                    else:
                        if word_buffer:
                            chunks.append(word_buffer)
                        word_buffer = word
                current = word_buffer
    if current:
        chunks.append(current)
    return _merge_letterless_prefix_chunks(chunks or [text.strip()])


def _cross_fade_waves(waves: list[np.ndarray], cross_fade_duration: float) -> np.ndarray:
    if not waves:
        return np.array([], dtype=np.float32)
    if cross_fade_duration <= 0:
        return np.concatenate(waves)
    final_wave = waves[0]
    for i in range(1, len(waves)):
        prev_wave = final_wave
        next_wave = waves[i]
        overlap = int(cross_fade_duration * TARGET_SAMPLE_RATE)
        overlap = min(overlap, len(prev_wave), len(next_wave))
        if overlap <= 0:
            final_wave = np.concatenate([prev_wave, next_wave])
            continue
        if len(prev_wave) <= overlap or len(next_wave) <= overlap:
            final_wave = np.concatenate([prev_wave, next_wave])
            continue
        fade_out = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
        mixed = prev_wave[-overlap:] * fade_out + next_wave[:overlap] * fade_in
        final_wave = np.concatenate([prev_wave[:-overlap], mixed, next_wave[overlap:]])
    return final_wave


def infer_process(
    ref_audio_path: str,
    ref_text: str,
    gen_text: str,
    model: torch.nn.Module,
    vocoder: torch.nn.Module,
    target_rms: float,
    cross_fade_duration: float,
    nfe_step: int,
    cfg_strength: float,
    sway_sampling_coef: float,
    speed: float,
    fix_duration: float | None,
    device: str,
) -> tuple[np.ndarray | None, int, None]:
    audio, sr = torchaudio.load(ref_audio_path)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    audio = audio.float()
    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < target_rms:
        audio = audio * (target_rms / max(rms, torch.tensor(1e-8)))

    if sr != TARGET_SAMPLE_RATE:
        audio = torchaudio.transforms.Resample(sr, TARGET_SAMPLE_RATE)(audio)
    audio = audio.to(device)

    ref_dur = float(audio.shape[-1]) / float(TARGET_SAMPLE_RATE)
    ref_text_bytes = max(len(ref_text.encode("utf-8")), 1)
    budget_sec = max(0.5, 22.0 - ref_dur)
    max_chars = max(40, int(ref_text_bytes / max(ref_dur, 1e-3) * budget_sec * speed))
    gen_text_batches = chunk_text(gen_text, max_chars=max_chars)

    if not ref_text:
        return None, TARGET_SAMPLE_RATE, None
    if len(ref_text[-1].encode("utf-8")) == 1:
        ref_text = f"{ref_text} "

    ref_audio_len = audio.shape[-1] // HOP_LENGTH
    generated_waves: list[np.ndarray] = []

    for chunk in gen_text_batches:
        local_speed = speed if len(chunk.encode("utf-8")) >= 10 else 0.3
        if fix_duration is not None:
            duration = int(fix_duration * TARGET_SAMPLE_RATE / HOP_LENGTH)
        else:
            duration = ref_audio_len + int(ref_audio_len / ref_text_bytes * len(chunk.encode("utf-8")) / local_speed)

        prompt = [ref_text + chunk]

        with torch.inference_mode():
            generated, _ = model.sample(
                cond=audio,
                text=prompt,
                duration=duration,
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
            )
            generated = generated.to(torch.float32)[:, ref_audio_len:, :].permute(0, 2, 1)
            generated_wave = vocoder.decode(generated).squeeze().cpu().numpy().astype(np.float32)
            if rms < target_rms:
                generated_wave = generated_wave * float(rms / target_rms)
            generated_waves.append(generated_wave)

    if not generated_waves:
        return None, TARGET_SAMPLE_RATE, None

    return _cross_fade_waves(generated_waves, cross_fade_duration), TARGET_SAMPLE_RATE, None


def synthesize_once(
    text: str,
    ref_audio: str,
    ref_text: str,
    model: torch.nn.Module,
    vocoder: torch.nn.Module,
    args: argparse.Namespace,
    device: str,
) -> tuple[np.ndarray | None, int]:
    wave, sample_rate, _ = infer_process(
        ref_audio,
        ref_text,
        text,
        model,
        vocoder,
        target_rms=args.target_rms,
        cross_fade_duration=args.cross_fade_duration,
        nfe_step=args.nfe_step,
        cfg_strength=args.cfg_strength,
        sway_sampling_coef=args.sway_sampling_coef,
        speed=args.speed,
        fix_duration=args.fix_duration,
        device=device,
    )
    return wave, sample_rate


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    try:
        validate_args(args)
    except Exception as exc:
        LOGGER.error("Invalid arguments: %s", exc)
        return 2

    device = get_device()
    LOGGER.info("Device: %s", device)

    set_seed(args.seed)

    try:
        model_path, vocab_path = resolve_paths(args.model_repo, args.model_step)
        model, vocoder = load_runtime(model_path, vocab_path, device)
        clean_ref_audio, clean_ref_text = preprocess_ref_audio_text(str(args.ref_audio), args.ref_text)
    except Exception as exc:
        LOGGER.exception("Failed during model/runtime preparation: %s", exc)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Interactive mode started. Type 'exit' or 'quit' to stop.")

    while True:
        try:
            user_text = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            LOGGER.info("Stopped by user.")
            return 0

        if user_text.lower() in {"exit", "quit"}:
            LOGGER.info("Session ended.")
            return 0

        if not user_text:
            continue

        try:
            audio, sr = synthesize_once(
                text=user_text,
                ref_audio=clean_ref_audio,
                ref_text=clean_ref_text,
                model=model,
                vocoder=vocoder,
                args=args,
                device=device,
            )
            if audio is None:
                LOGGER.warning("No audio generated for current input.")
                continue
            sf.write(str(args.output), audio, sr)
            LOGGER.info("Saved: %s", args.output)
        except Exception as exc:
            LOGGER.exception("Synthesis failed: %s", exc)


if __name__ == "__main__":
    raise SystemExit(main())
