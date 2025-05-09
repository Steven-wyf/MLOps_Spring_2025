import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import mlflow
import random
import pandas as pd

# ====================
# CONFIG
# ====================
DATA_DIR = os.environ.get("PLAYLIST_DATA_DIR", "./data")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./outputs")
PROJECTED_EMB = os.path.join(OUTPUT_DIR, "projected_lightgcn.npz")
PLAYLIST_FILE = os.path.join(DATA_DIR, "playlist_track_list.csv")  # Format: playlist_id,track_uri
MODEL_PATH = os.path.join(OUTPUT_DIR, "llara_classifier.pt")
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:8000")
EXPERIMENT_NAME = "llara-classifier"
EMBED_DIM = 768
EPOCHS = 50
BATCH_SIZE = 128
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configure MLflow
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
os.environ["AWS_ACCESS_KEY_ID"] = os.environ.get("AWS_ACCESS_KEY_ID", "minio")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ.get("AWS_SECRET_ACCESS_KEY", "minio123")

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# ====================
# LOAD EMBEDDINGS
# ====================
print(f"Loading embeddings from {PROJECTED_EMB}")
emb_data = np.load(PROJECTED_EMB)
all_track_ids = list(emb_data.files)
track_id_to_idx = {tid: i for i, tid in enumerate(all_track_ids)}
X_emb = np.stack([emb_data[tid] for tid in all_track_ids])

# ====================
# LOAD PLAYLIST DATA
# ====================
print(f"Loading playlist data from {PLAYLIST_FILE}")
df = pd.read_csv(PLAYLIST_FILE).dropna()
df = df[df['track_uri'].isin(track_id_to_idx)]
print(f"Loaded {len(df)} valid playlist-track pairs")

# Group playlists
playlists = df.groupby('playlist_id')['track_uri'].apply(list)
print(f"Working with {len(playlists)} playlists")

# Create (context → target) training pairs
samples = []
for plist in playlists:
    if len(plist) < 2:
        continue
    for i in range(1, len(plist)):
        context = plist[i - 1]
        target = plist[i]
        if context in track_id_to_idx and target in track_id_to_idx:
            samples.append((track_id_to_idx[context], track_id_to_idx[target]))

print(f"Created {len(samples)} training samples")

# ====================
# SPLIT DATA
# ====================
train_samples, val_samples = train_test_split(samples, test_size=0.1, random_state=42)
print(f"Training set: {len(train_samples)}, Validation set: {len(val_samples)}")

def get_batch(samples, batch_size):
    batch = random.sample(samples, min(batch_size, len(samples)))
    x = torch.tensor([X_emb[i] for i, _ in batch], dtype=torch.float32)
    y = torch.tensor([t for _, t in batch], dtype=torch.long)
    return x.to(DEVICE), y.to(DEVICE)

# ====================
# MODEL
# ====================
class Classifier(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

model = Classifier(EMBED_DIM, len(all_track_ids)).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

best_val_loss = float("inf")
with mlflow.start_run():
    mlflow.log_params({
        "epochs": EPOCHS,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "embedding_dim": EMBED_DIM,
        "num_tracks": len(all_track_ids),
        "device": str(DEVICE)
    })

    for epoch in range(EPOCHS):
        model.train()
        x_batch, y_batch = get_batch(train_samples, BATCH_SIZE)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mlflow.log_metric("train_loss", loss.item(), step=epoch)

        # validation
        model.eval()
        with torch.no_grad():
            val_x, val_y = get_batch(val_samples, BATCH_SIZE)
            val_logits = model(val_x)
            val_loss = criterion(val_logits, val_y)
            mlflow.log_metric("val_loss", val_loss.item(), step=epoch)

        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

        # save best model
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            torch.save(model.state_dict(), MODEL_PATH)
            mlflow.log_artifact(MODEL_PATH)

print(f"Training complete. Best model saved to {MODEL_PATH}")
