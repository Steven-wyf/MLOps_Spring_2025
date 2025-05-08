import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import mlflow
import random

# ====================
# CONFIG
# ====================
PROJECTED_EMB = "projected_lightgcn.npz"
PLAYLIST_FILE = "playlist_track.csv"  # Format: playlist_id,track_uri
MODEL_PATH = "llara_classifier.pt"
MLFLOW_URI = "http://<your-node-ip>:8000"
EXPERIMENT_NAME = "llara-classifier"
EMBED_DIM = 768
EPOCHS = 50
BATCH_SIZE = 128
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================
# LOAD EMBEDDINGS
# ====================
emb_data = np.load(PROJECTED_EMB)
all_track_ids = list(emb_data.files)
track_id_to_idx = {tid: i for i, tid in enumerate(all_track_ids)}
X_emb = np.stack([emb_data[tid] for tid in all_track_ids])

# ====================
# LOAD PLAYLIST DATA
# ====================
import pandas as pd
df = pd.read_csv(PLAYLIST_FILE).dropna()
df = df[df['track_uri'].isin(track_id_to_idx)]

# Group playlists
playlists = df.groupby('playlist_id')['track_uri'].apply(list)

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

# ====================
# SPLIT DATA
# ====================
train_samples, val_samples = train_test_split(samples, test_size=0.1, random_state=42)

def get_batch(samples, batch_size):
    batch = random.sample(samples, batch_size)
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

# ====================
# MLFLOW
# ====================
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

best_val_loss = float("inf")
with mlflow.start_run():
    mlflow.log_params({
        "epochs": EPOCHS,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "embedding_dim": EMBED_DIM,
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
