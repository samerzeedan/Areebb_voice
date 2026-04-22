import argparse
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from importlib.resources import files
from io import BytesIO

from datasets import load_dataset
from elevenlabs.client import ElevenLabs
from elevenlabs.core.api_error import ApiError
from tqdm import tqdm


rel_path = str(files("Areebb_tts").joinpath("../../"))

voice_id_map = {
    "MSA": "JjTirzdD7T3GMLkwdd3a",
    "SAU-Najdi": f"{rel_path}/src/Areebb_tts/assets/Najdi.wav",
    "SAU-Hijazi": f"{rel_path}/src/Areebb_tts/assets/Hijazi.wav",
    "SAU-Gulf": f"{rel_path}/src/Areebb_tts/assets/Gulf.wav",
    "UAE": f"{rel_path}/src/Areebb_tts/assets/UAE.wav",
    "ALG": f"{rel_path}/src/Areebb_tts/assets/ALG.wav",
    "IRQ": f"{rel_path}/src/Areebb_tts/assets/IRQ.wav",
    "EGY": "IES4nrmZdUBHByLBde0P",
    "MAR": "OfGMGmhShO8iL9jCkXy8",
}


def single_api_call(b, output_dir, api_key, language_code, seed):
    gen_text, gen_path, gen_dialect = b

    output_path = f"{output_dir}/{os.path.splitext(gen_path)[0]}.wav"  # mp3
    if os.path.exists(output_path):
        print(f"{output_path} already exist, skip...")
        return 1

    client = ElevenLabs(api_key=api_key)

    voice_id_entry = voice_id_map.get(gen_dialect, "JjTirzdD7T3GMLkwdd3a")
    if "Areebb_tts" in voice_id_entry:
        voice = client.voices.ivc.create(
            name=os.path.splitext(voice_id_entry.split("/")[-1])[0],
            description=voice_id_entry,
            files=[BytesIO(open(voice_id_entry, "rb").read())],
        )
        voice_id = voice.voice_id
        print(f"Voice ID: {voice_id} with uploaded {voice_id_entry}")

        voice_id_map[gen_dialect] = voice_id
    else:
        voice_id = voice_id_entry

    try:
        audio = client.text_to_speech.convert(
            text=gen_text,
            voice_id=voice_id,
            model_id="eleven_v3",
            output_format="wav_24000",  # "mp3_44100_128"
            language_code=language_code,
            seed=seed,
        )
        with open(output_path, "wb") as f:
            for chunk in audio:
                f.write(chunk)
    except ApiError as e:
        os.remove(output_path)  # delete failed broken file
        if e.status_code == 401:  # and e.body["detail"]["status"] == "quota_exceeded":
            print(f"Quota of {api_key} run out, {e.body['detail']['message']}")
        else:
            raise NotImplementedError(f"[Code 1_infer_11labs.py] Not handled ApiError: {e}")
    finally:
        pass

    assert os.path.exists(output_path), "[Code 1_infer_11labs.py] API keys expired, please check."

    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--api-key", type=str, required=True, help="ElevenLabs account API key")
    parser.add_argument("-d", "--dialect", type=str, required=True, help="MSA | SAU | UAE | ALG | IRQ | EGY | MAR")
    parser.add_argument("-c", "--concurrency", default=5, type=int)  # Creator subscription max concurrency is 5
    parser.add_argument("-l", "--language-code", default="ar", type=str)
    parser.add_argument("-s", "--seed", default=0, type=int)
    args = parser.parse_args()
    dialect = args.dialect

    # pull benchmark dataset
    benchmark = load_dataset("SWivid/Areebb_tts", dialect, split="test")

    spk_id_dict = defaultdict(list)
    for obj in benchmark:
        spk_id_dict[obj["speaker_id"]].append(obj)

    # The current entry uses the previous entry with the same speaker_id as prompt
    # The last entry of a speaker_id is the prompt of the first
    benchmark = []
    for spk_id in spk_id_dict:
        for i in range(len(spk_id_dict[spk_id])):
            benchmark.append(
                [
                    spk_id_dict[spk_id][i]["text"],  # gen_text
                    spk_id_dict[spk_id][i]["audio"]["path"],  # gen_path
                    spk_id_dict[spk_id][i]["dialect"],  # gen_dialect
                ]
            )

    output_dir = f"{rel_path}/results/11Labs_3a/{dialect}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    executor = ProcessPoolExecutor(max_workers=args.concurrency)
    futures = []
    for b in benchmark:
        futures.append(executor.submit(single_api_call, b, output_dir, args.api_key, args.language_code, args.seed))
    for futures in tqdm(futures, total=len(futures)):
        _ = futures.result()
    executor.shutdown()


if __name__ == "__main__":
    main()

    # client.voices.delete(voice_id="")  # delete uploaded voice if wish

