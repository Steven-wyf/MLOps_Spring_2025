"""
    This script is used to process the data from the json file and convert them to tensor
"""

import json
import os
import sys
import csv
from collections import defaultdict

def process_data(path, output_dir, quick=False):
    filenames = sorted([f for f in os.listdir(path) if f.startswith("mpd.slice.") and f.endswith(".json")])
    flat_rows = []
    track_metadata = defaultdict(str)
    seen_tracks = set()

    for i, filename in enumerate(filenames):
        if quick and i >= 2:  # just process 2 files for fast dev
            break
        print(f"Processing {filename}")
        fullpath = os.path.join(path, filename)
        with open(fullpath, 'r', encoding='utf-8') as f:
            mpd_slice = json.load(f)

        for playlist in mpd_slice['playlists']:
            pid = playlist['pid']
            for track in playlist['tracks']:
                tid = track["track_uri"]
                flat_rows.append((pid, tid))

                if tid not in seen_tracks:
                    text = f"{track['track_name']} by {track['artist_name']} from {track['album_name']}"
                    track_metadata[tid] = text
                    seen_tracks.add(tid)

    # Save CSV for LightGCN
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "playlist_track_pairs.csv")
    with open(csv_path, "w", newline='', encoding='utf-8') as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["playlist_id", "track_uri"])
        writer.writerows(flat_rows)
    print(f"Saved {len(flat_rows)} entries to {csv_path}")

    # Save JSON for BERT
    json_path = os.path.join(output_dir, "track_texts.json")
    with open(json_path, "w", encoding='utf-8') as f_json:
        json.dump(track_metadata, f_json, indent=2, ensure_ascii=False)
    print(f"Saved {len(track_metadata)} unique tracks to {json_path}")

if __name__ == "__main__":
    quick = False
    path = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "--quick":
        quick = True
    process_data(path, "../processed", quick)
