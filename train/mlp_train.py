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

def get_latest_run_id(experiment_name: str) -> str:
    """Get the latest successful run ID for an experiment."""
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string="status = 'FINISHED'",
        order_by=["start_time DESC"]
    )
    if runs.empty:
        raise ValueError(f"No successful runs found for experiment {experiment_name}")
    return runs.iloc[0].run_id

def load_embeddings_from_mlflow() -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, int]]:
    """Load BERT and MF embeddings from MLflow."""
    try:
        # Get latest run IDs
        bert_run_id = get_latest_run_id("bert-track-embeddings")
        mf_run_id = get_latest_run_id("matrix-factorization")
        
        # Download embeddings
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Download BERT embeddings
            bert_embeddings = {}
            chunk_dir = mlflow.artifacts.download_artifacts(
                run_id=bert_run_id,
                artifact_path="embeddings",
                dst_path=tmp_dir
            )
            
            for chunk in os.listdir(chunk_dir):
                chunk_path = os.path.join(chunk_dir, chunk)
                if os.path.isdir(chunk_path):
                    for file in os.listdir(chunk_path):
                        if file.endswith(".npz"):
                            data = np.load(os.path.join(chunk_path, file), allow_pickle=True)
                            bert_embeddings.update({k: data[k] for k in data.files})
            
            # Download MF embeddings
            mf_dir = mlflow.artifacts.download_artifacts(
                run_id=mf_run_id,
                artifact_path="embeddings",
                dst_path=tmp_dir
            )
            
            mf_embeddings = {}
            for chunk in os.listdir(mf_dir):
                chunk_path = os.path.join(mf_dir, chunk)
                if os.path.isdir(chunk_path):
                    for file in os.listdir(chunk_path):
                        if file.endswith(".npz"):
                            data = np.load(os.path.join(chunk_path, file), allow_pickle=True)
                            mf_embeddings.update({k: data[k] for k in data.files})
            
            # Load track mapping
            mapping_path = mlflow.artifacts.download_artifacts(
                run_id=bert_run_id,
                artifact_path="mappings/track_uri_mapping.npz",
                dst_path=tmp_dir
            )
            mapping_data = np.load(mapping_path)
            track_uris = mapping_data['track_uris']
            track_ids = mapping_data['track_ids']
            uri_to_idx = {str(uri): int(idx) for uri, idx in zip(track_uris, track_ids)}
            
            return bert_embeddings, mf_embeddings['item_embeddings'], uri_to_idx
    
    except Exception as e:
        logger.error(f"Error loading embeddings: {str(e)}")
        raise

# Load input data
bert_embeddings, item_embeddings, uri_to_idx = load_embeddings_from_mlflow()

# Print sample embeddings
logger.info(f"Sample BERT keys: {list(bert_embeddings.keys())[:3]}")
logger.info(f"Sample MF URIs: {list(uri_to_idx.keys())[:3]}")

# Align track IDs in both
bert_uris = set(bert_embeddings.keys())
mf_uris = set(uri_to_idx.keys())
common_uris = list(bert_uris & mf_uris)
common_uris = sorted(common_uris)  # ensure order

if not common_uris:
    raise ValueError("No common URIs found between BERT and MF embeddings!")

# Convert URIs to indices for MF embeddings
common_indices = np.array([uri_to_idx[uri] for uri in common_uris], dtype=np.int64)

# Use MF embeddings as input (X) and BERT embeddings as target (Y)
X = item_embeddings[common_indices]
Y = np.array([bert_embeddings[uri] for uri in common_uris])

# Skip tracks with invalid embeddings
valid_indices = []
for i in range(len(X)):
    if not np.isnan(X[i]).any() and not np.isnan(Y[i]).any():
        valid_indices.append(i)

if not valid_indices:
    raise ValueError("No valid embeddings found after filtering!")

X = X[valid_indices]
Y = Y[valid_indices]
common_uris = [common_uris[i] for i in valid_indices]

# Convert to tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

# === Define MLP Projector ===
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

# === Train MLP ===
with mlflow.start_run() as run:
    # Log parameters
    mlflow.log_params({
        "input_dim": X_tensor.shape[1],  # MF embedding dimension
        "output_dim": Y_tensor.shape[1],  # BERT embedding dimension
        "hidden_dim": 256,
        "learning_rate": 1e-3,
        "epochs": 20,
        "num_samples": len(common_uris)
    })
    
    input_dim = X_tensor.shape[1]
    output_dim = Y_tensor.shape[1]
    model = MLPProjector(input_dim, output_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    X_tensor = X_tensor.to(device)
    Y_tensor = Y_tensor.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 20

    model.train()
    for epoch in tqdm(range(epochs), desc="Training MLP Projector"):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, Y_tensor)
        loss.backward()
        optimizer.step()

        # Log metrics
        mlflow.log_metric("loss", loss.item(), step=epoch)
        logger.info(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

    # Save model to MinIO through MLflow
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, "mlp_projector.pt")
        torch.save({
            "model_state": model.state_dict(),
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dim": 256,
            "mlflow_run_id": run.info.run_id
        }, model_path)
        mlflow.log_artifact(model_path, "models")
        logger.info(f"Model saved to MinIO through MLflow run {run.info.run_id}")
        
        # Generate projected embeddings for all tracks
        logger.info("Generating projected embeddings...")
        model.eval()
        with torch.no_grad():
            # Get all MF embeddings
            all_mf_embeddings = item_embeddings
            all_mf_embeddings = torch.tensor(all_mf_embeddings, dtype=torch.float32).to(device)
            
            # Project embeddings
            projected_embeddings = model(all_mf_embeddings).cpu().numpy()
            
            # Save projected embeddings
            projected_path = os.path.join(tmp_dir, "projected_embeddings.npz")
            np.savez(
                projected_path,
                embeddings=projected_embeddings,
                track_uris=list(map(str, common_uris))
            )
            mlflow.log_artifact(projected_path, "embeddings")
            logger.info(f"Projected embeddings saved to MinIO through MLflow run {run.info.run_id}")

logger.info("Training completed successfully!")
