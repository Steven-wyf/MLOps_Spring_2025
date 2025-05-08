import torch
import torch.nn.functional as F
import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from llara import Classifier  # your model class
from sklearn.model_selection import train_test_split

# ========== Config ==========
PROJECTED_EMB = "projected_lightgcn.npz"
PLAYLIST_FILE = "playlist_track.csv"
MODEL_PATH = "llara_classifier.pt"
TOP_K = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== Load Embeddings ==========
emb_data = np.load(PROJECTED_EMB)
all_track_ids = list(emb_data.files)
track_id_to_idx = {tid: i for i, tid in enumerate(all_track_ids)}
idx_to_track_id = {i: tid for tid, i in track_id_to_idx.items()}
X_emb = np.stack([emb_data[tid] for tid in all_track_ids])

# ========== Load Playlist Data ==========
df = pd.read_csv(PLAYLIST_FILE).dropna()
df = df[df['track_uri'].isin(track_id_to_idx)]

playlists = df.groupby('playlist_id')['track_uri'].apply(list)
samples = []
for plist in playlists:
    if len(plist) < 2:
        continue
    for i in range(1, len(plist)):
        context = plist[i - 1]
        target = plist[i]
        if context in track_id_to_idx and target in track_id_to_idx:
            samples.append((track_id_to_idx[context], track_id_to_idx[target]))

_, val_samples = train_test_split(samples, test_size=0.1, random_state=42)

# ========== Load Model ==========
model = Classifier(embed_dim=768, num_classes=len(all_track_ids)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ========== Evaluation ==========
correct = 0
total = 0

for input_idx, target_idx in random.sample(val_samples, k=500):  # sample if large
    input_vec = torch.tensor(X_emb[input_idx], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    logits = model(input_vec)
    topk = torch.topk(logits, k=TOP_K, dim=1).indices[0].cpu().numpy()
    
    if target_idx in topk:
        correct += 1
    total += 1

recall_at_k = correct / total
print(f"Recall@{TOP_K}: {recall_at_k:.4f}")
