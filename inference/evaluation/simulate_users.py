# inference/evaluation/simulate_users.py
import os
import json
import time
import random
import requests

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8080/predict")
EXAMPLE_PATH = "inference/evaluation/example_tracks.json"

with open(EXAMPLE_PATH, "r", encoding="utf-8") as f:
    EXAMPLE_TRACKS = json.load(f)

def simulate_request(track_texts):
    payload = {"tracks": track_texts}
    response = requests.post(FASTAPI_URL, json=payload)
    try:
        res_json = response.json()
        print(f"Response: {res_json['request_id']} => {res_json['predictions'][:3]}...")
    except Exception as e:
        print(f"Failed to decode response: {e}")

def simulate_users(rounds=5, delay=5):
    print(f"🚀 Sending {rounds} simulated requests to {FASTAPI_URL}...")
    for i in range(rounds):
        sampled_tracks = random.sample(EXAMPLE_TRACKS, k=random.randint(1, 3))
        simulate_request(sampled_tracks)
        time.sleep(delay)

if __name__ == "__main__":
    simulate_users(rounds=20, delay=2)