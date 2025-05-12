# inference/evaluation/gen_real_examples.py
import json
import random
import os

TRACK_TEXT_PATH = "/mnt/block/processed/track_texts.json"
OUTPUT_PATH = "evaluation/online_evaluation/example_tracks.json"


def generate_example_samples(n=20):
    with open(TRACK_TEXT_PATH, "r", encoding="utf-8") as f:
        track_map = json.load(f)

    samples = [{"text": t} for t in random.sample(list(track_map.values()), k=min(n, len(track_map)))]
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
        json.dump(samples, out_f, indent=2, ensure_ascii=False)
    print(f"✅ Saved {len(samples)} track samples to {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_example_samples(n=20)