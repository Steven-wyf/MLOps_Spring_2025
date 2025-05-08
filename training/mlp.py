import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import os

# ============== CONFIG ==============
BERT_NPZ = "./bert_track_embeddings.npz"
LIGHTGCN_NPZ = "./lightgcn_embeddings.npz"
MODEL_PATH = "mlp_projector.pt"
MLFLOW_URI = "http://<your-node-ip>:8000"
EXPERIMENT_NAME = "mlp-projector"
EPOCHS = 100
LR = 1e-3
HIDDEN_DIM = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============== LOAD EMBEDDINGS ==============
bert = np.load(BERT_NPZ)
lightgcn = np.load(LIGHTGCN_NPZ)

# Find intersecting track IDs
track_ids = list(set(bert.files).intersection(set(lightgcn.files)))
print(f"Training on {len(track_ids)} overlapping tracks")

X = np.stack([lightgcn[k] for k in track_ids])
Y = np.stack([bert[k] for k in track_ids])

X = torch.tensor(X, dtype=torch.float32).to(DEVICE)
Y = torch.tensor(Y, dtype=torch.float32).to(DEVICE)

# ============== DEFINE MODEL ==============
class Projector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

model = Projector(X.shape[1], HIDDEN_DIM, Y.shape[1]).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# ============== MLflow SETUP ==============
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run():
    mlflow.log_params({
        "epochs": EPOCHS,
        "lr": LR,
        "hidden_dim": HIDDEN_DIM,
        "input_dim": X.shape[1],
        "output_dim": Y.shape[1]
    })

    # ============== TRAIN LOOP ==============
    for epoch in range(EPOCHS):
        model.train()
        pred = model(X)
        loss = loss_fn(pred, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mlflow.log_metric("loss", loss.item(), step=epoch)
        print(f"Epoch {epoch+1}/{EPOCHS} Loss: {loss.item():.6f}")

    # ============== SAVE MODEL ==============
    torch.save(model.state_dict(), MODEL_PATH)
    mlflow.log_artifact(MODEL_PATH)

    # Optional: save projected embeddings
    projected = model(X).detach().cpu().numpy()
    np.savez("projected_lightgcn.npz", **{track_ids[i]: projected[i] for i in range(len(track_ids))})
    mlflow.log_artifact("projected_lightgcn.npz")

print("MLP projector training complete.")
