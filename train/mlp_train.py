import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import mlflow
import os
import logging
import tempfile
from typing import Tuple, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://129.114.25.37:8000/")
MLFLOW_S3_ENDPOINT_URL = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://129.114.25.37:9000")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "your-acccess-key")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "hrwbqzUS85G253yKi43T")
EXPERIMENT_NAME = "mlp-projector"

# Configure MLflow
os.environ["MLFLOW_S3_ENDPOINT_URL"] = MLFLOW_S3_ENDPOINT_URL
os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

def load_embeddings_from_mlflow() -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, int]]:
    bert_run_id = mlflow.search_runs(["bert-track-embeddings"], "status = 'FINISHED'", ["start_time DESC"]).iloc[0].run_id
    mf_run_id = mlflow.search_runs(["matrix-factorization"], "status = 'FINISHED'", ["start_time DESC"]).iloc[0].run_id

    with tempfile.TemporaryDirectory() as tmp_dir:
        bert_dir = mlflow.artifacts.download_artifacts(run_id=bert_run_id, artifact_path="embeddings", dst_path=tmp_dir)
        mf_dir = mlflow.artifacts.download_artifacts(run_id=mf_run_id, artifact_path="embeddings", dst_path=tmp_dir)

        bert_embeddings = {}
        for chunk in os.listdir(bert_dir):
            chunk_path = os.path.join(bert_dir, chunk)
            if os.path.isdir(chunk_path):
                for file in os.listdir(chunk_path):
                    if file.endswith(".npz"):
                        data = np.load(os.path.join(chunk_path, file), allow_pickle=True)
                        bert_embeddings.update({k: data[k] for k in data.files})

        mf_embeddings = {}
        uri_to_idx = {}
        for chunk in os.listdir(mf_dir):
            chunk_path = os.path.join(mf_dir, chunk)
            if os.path.isdir(chunk_path):
                for file in os.listdir(chunk_path):
                    if file.endswith(".npz"):
                        data = np.load(os.path.join(chunk_path, file), allow_pickle=True)
                        mf_embeddings.update({k: data[k] for k in data.files})

                        if 'item_mapping' in data:
                            index_to_uri = data['item_mapping'].item()
                            uri_to_idx.update({str(v): int(k) for k, v in index_to_uri.items()})

        return bert_embeddings, mf_embeddings['item_embeddings'], uri_to_idx

# Load input data
bert_embeddings, item_embeddings, uri_to_idx = load_embeddings_from_mlflow()

common_uris = list(set(bert_embeddings.keys()) & set(uri_to_idx.keys()))
if not common_uris:
    raise ValueError("No common track URIs found between BERT and MF embeddings")

X = np.array([item_embeddings[uri_to_idx[uri]] for uri in common_uris], dtype=np.float32)
Y = np.array([bert_embeddings[uri] for uri in common_uris], dtype=np.float32)

X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

class MLPProjector(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

with mlflow.start_run() as run:
    mlflow.log_params({
        "input_dim": X_tensor.shape[1],
        "output_dim": Y_tensor.shape[1],
        "hidden_dim": 256,
        "learning_rate": 1e-3,
        "epochs": 20,
        "num_samples": len(common_uris)
    })

    model = MLPProjector(X_tensor.shape[1], Y_tensor.shape[1]).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    X_tensor = X_tensor.to(model.net[0].weight.device)
    Y_tensor = Y_tensor.to(model.net[0].weight.device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    for epoch in tqdm(range(20), desc="Training MLP Projector"):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, Y_tensor)
        loss.backward()
        optimizer.step()
        mlflow.log_metric("loss", loss.item(), step=epoch)

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, "mlp_projector.pt")
        torch.save({
            "model_state": model.state_dict(),
            "input_dim": X_tensor.shape[1],
            "output_dim": Y_tensor.shape[1],
            "hidden_dim": 256,
            "mlflow_run_id": run.info.run_id
        }, model_path)
        mlflow.log_artifact(model_path, "models")

        projected = model(torch.tensor(item_embeddings, dtype=torch.float32).to(model.net[0].weight.device)).cpu().detach().numpy()
        out_path = os.path.join(tmp_dir, "projected_embeddings.npz")
        np.savez(out_path, embeddings=projected, track_uris=list(uri_to_idx.keys()))
        mlflow.log_artifact(out_path, "embeddings")
