import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
from tqdm import tqdm

from Areebb_tts.eval.utils_eval import normalize_arabic_text, word_error_rate


device = (
    "cuda"
    if torch.cuda.is_available()
    else "xpu"
    if torch.xpu.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

lang_map = {
    "MSA": "arb_Arab",
    "SAU-Najdi": "ars_Arab",
    "SAU-Hijazi": "acw_Arab",
    "SAU-Gulf": "afb_Arab",
    "UAE": "afb_Arab",
    "ALG": "arq_Arab",
    "IRQ": "ayp_Arab",
    "EGY": "arz_Arab",
    "MAR": "ary_Arab",
    # "OMN": "acx_Arab",  # not yet supported by omniASR
    "TUN": "aeb_Arab",
    "LEV": "apc_Arab",
    "SDN": "apd_Arab",
    "LBY": "ayl_Arab",
}


def calculate_wer(pipeline, wav_dir, dialect, batch_size):
    benchmark = load_dataset("SWivid/Areebb_tts", dialect, split="test")

    audio_path_batch = []
    gt_text_batch = []
    lang_batch = []

    wer_objs = []
    for b in tqdm(benchmark):
        audio_path = b["audio"]["path"]
        gt_text = b["text"]

        if "11Labs_3a" in wav_dir:
            print("Note that omniASR only accept .wav files")
        audio_path_batch.append(f"{wav_dir}/{audio_path}")
        gt_text_batch.append(gt_text)
        lang_batch.append(lang_map.get(b["dialect"], "arb_Arab"))
        if len(audio_path_batch) < batch_size:
            continue

        transcriptions = pipeline.transcribe(
            audio_path_batch,
            lang=lang_batch,
            batch_size=batch_size,
        )

        for i in range(len(audio_path_batch)):
            gt_text_norm = normalize_arabic_text(gt_text_batch[i])
            pr_text_norm = normalize_arabic_text(transcriptions[i])
            wer_obj = {
                "audio_path": audio_path_batch[i],
                "gt_text_norm": gt_text_norm,
                "pr_text_norm": pr_text_norm,
            }
            wer = word_error_rate([pr_text_norm], [gt_text_norm])
            wer_obj["wer"] = wer
            wer_objs.append(wer_obj)

        audio_path_batch = []
        gt_text_batch = []
        lang_batch = []

    # Process remaining files in the last batch
    if len(audio_path_batch) > 0:
        transcriptions = pipeline.transcribe(
            audio_path_batch,
            lang=lang_batch,
            batch_size=len(audio_path_batch),
        )
        for i in range(len(audio_path_batch)):
            gt_text_norm = normalize_arabic_text(gt_text_batch[i])
            pr_text_norm = normalize_arabic_text(transcriptions[i])
            wer_obj = {
                "audio_path": audio_path_batch[i],
                "gt_text_norm": gt_text_norm,
                "pr_text_norm": pr_text_norm,
            }
            wer = word_error_rate([pr_text_norm], [gt_text_norm])
            wer_obj["wer"] = wer
            wer_objs.append(wer_obj)

    return wer_objs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--wav-dir", type=str, required=True)
    parser.add_argument("-d", "--dialect", type=str, required=True, help="MSA | SAU | UAE | ALG | IRQ | EGY | MAR")
    parser.add_argument("-b", "--batch-size", type=int, default=64)
    args = parser.parse_args()

    pipeline = ASRInferencePipeline(model_card="omniASR_LLM_7B")  # "omniASR_LLM_7B_v2"
    wer_objs = calculate_wer(pipeline, args.wav_dir, args.dialect, args.batch_size)

    wer_result_path = Path(args.wav_dir) / "_wer_o_results.jsonl"
    pr_text_norms = []
    gt_text_norms = []
    with open(wer_result_path, "w", encoding="utf-8") as f:
        for wer_obj in wer_objs:
            pr_text_norms.append(wer_obj["pr_text_norm"])
            gt_text_norms.append(wer_obj["gt_text_norm"])
            f.write(json.dumps(wer_obj, ensure_ascii=False) + "\n")
        f.write(f"\nGlobal WER-O: {word_error_rate(pr_text_norms, gt_text_norms)}\n")

    print(f"Global WER-O: {word_error_rate(pr_text_norms, gt_text_norms)}")
    print(f"Single WER-O results saved to {wer_result_path}")

