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

def load_embeddings_from_mlflow() -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Load BERT and MF embeddings from MLflow."""
    try:
        # Get latest run IDs
        bert_run_id = get_latest_run_id("bert-track-embeddings")
        mf_run_id = get_latest_run_id("matrix-factorization")
        
        logger.info(f"Loading embeddings from BERT run {bert_run_id} and MF run {mf_run_id}")
        
        # Download embeddings
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Download BERT embeddings (now in chunks)
            bert_embeddings = {}
            chunk_idx = 0
            while True:
                try:
                    # 下载整个 embeddings 目录
                    chunk_dir = mlflow.artifacts.download_artifacts(
                        run_id=bert_run_id,
                        artifact_path="embeddings",
                        dst_path=tmp_dir
                    )
                    
                    # 查找所有 .npz 文件
                    npz_files = [f for f in os.listdir(chunk_dir) if f.endswith('.npz')]
                    if not npz_files:
                        if chunk_idx == 0:
                            raise Exception("No BERT embeddings found")
                        break
                    
                    # 加载每个 .npz 文件
                    for npz_file in npz_files:
                        npz_path = os.path.join(chunk_dir, npz_file)
                        chunk_data = np.load(npz_path)
                        bert_embeddings.update({k: chunk_data[k] for k in chunk_data.files})
                        logger.info(f"Loaded BERT chunk from {npz_file}")
                    
                    break  # 所有文件都已加载
                    
                except Exception as e:
                    if chunk_idx == 0:
                        raise Exception(f"No BERT embeddings found: {str(e)}")
                    break
            
            # Download MF embeddings (now in chunks)
            mf_embeddings = {}
            chunk_idx = 0
            while True:
                try:
                    # 下载整个 embeddings 目录
                    chunk_dir = mlflow.artifacts.download_artifacts(
                        run_id=mf_run_id,
                        artifact_path="embeddings",
                        dst_path=tmp_dir
                    )
                    
                    # 查找所有 .npz 文件
                    npz_files = [f for f in os.listdir(chunk_dir) if f.endswith('.npz')]
                    if not npz_files:
                        if chunk_idx == 0:
                            raise Exception("No MF embeddings found")
                        break
                    
                    # 加载每个 .npz 文件
                    for npz_file in npz_files:
                        npz_path = os.path.join(chunk_dir, npz_file)
                        chunk_data = np.load(npz_path)
                        mf_embeddings.update({k: chunk_data[k] for k in chunk_data.files})
                        logger.info(f"Loaded MF chunk from {npz_file}")
                    
                    break  # 所有文件都已加载
                    
                except Exception as e:
                    if chunk_idx == 0:
                        raise Exception(f"No MF embeddings found: {str(e)}")
                    break
        
        return bert_embeddings, mf_embeddings
    
    except Exception as e:
        logger.error(f"Error loading embeddings: {str(e)}")
        raise

# === Load input data ===
logger.info("Loading embeddings from MLflow...")
bert_embeddings, mf_embeddings = load_embeddings_from_mlflow()

# Load track embeddings
item_embeddings = mf_embeddings['item_embeddings']  # matrix [num_items, dim]

# Align track IDs in both
common_ids = list(set(bert_embeddings.keys()) & set(map(str, range(item_embeddings.shape[0]))))
common_ids = sorted(common_ids, key=int)  # ensure order

# Use MF embeddings as input (X) and BERT embeddings as target (Y)
X = item_embeddings[np.array(list(map(int, common_ids)))]
Y = np.array([bert_embeddings[tid] for tid in common_ids])

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
        "num_samples": len(common_ids)
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
                track_ids=list(map(str, range(len(all_mf_embeddings))))
            )
            mlflow.log_artifact(projected_path, "embeddings")
            logger.info(f"Projected embeddings saved to MinIO through MLflow run {run.info.run_id}")

logger.info("Training completed successfully!")
