# Example template for benchmark use

import argparse
from collections import defaultdict
from pprint import pprint

import torch
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dialect", type=str, required=True, help="MSA | SAU | UAE | ALG | IRQ | EGY | MAR")
    args = parser.parse_args()

    # pull benchmark dataset
    benchmark = load_dataset("SWivid/Areebb_tts", args.dialect, split="test")

    spk_id_dict = defaultdict(list)
    for obj in benchmark:
        spk_id_dict[obj["speaker_id"]].append(obj)

    # The current entry uses the previous entry with the same speaker_id as prompt
    # The last entry of a speaker_id is the prompt of the first
    benchmark = []
    for spk_id in spk_id_dict:
        for i in range(len(spk_id_dict[spk_id])):
            ref = spk_id_dict[spk_id][i - 1]
            gen = spk_id_dict[spk_id][i]
            benchmark.append(
                {
                    "speaker_id": spk_id,
                    "ref_audio_path": ref["audio"]["path"],
                    "ref_audio_array": ref["audio"]["array"],
                    "ref_audio_sampling_rate": ref["audio"]["sampling_rate"],
                    "ref_audio_duration": ref["duration"],
                    "ref_text": ref["text"],
                    "ref_dialect": ref["dialect"],
                    "gen_audio_path": gen["audio"]["path"],
                    "gen_audio_array": gen["audio"]["array"],
                    "gen_audio_sampling_rate": gen["audio"]["sampling_rate"],
                    "gen_audio_duration": gen["duration"],
                    "gen_text": gen["text"],
                    "gen_dialect": gen["dialect"],
                }
            )

    for b in benchmark:
        pprint(b)
        ref_audio_tensor = torch.from_numpy(b["ref_audio_array"]).float().unsqueeze(0)  # noqa
        break
        # Custom model inferecence ...


if __name__ == "__main__":
    main()

