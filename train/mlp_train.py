import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import mlflow
import os
import logging
import tempfile
from typing import Tuple, Dict, Any

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

def load_track_mapping() -> Tuple[Dict[str, int], Dict[int, str]]:
    """Load the track URI to integer mapping from BERT encoding."""
    # Get the latest BERT run ID
    bert_run_id = get_latest_run_id("bert-track-embeddings")
    logger.info(f"Loading track mapping from BERT run {bert_run_id}")
    
    # Download and load mapping info
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Load mapping info
        info_path = mlflow.artifacts.download_artifacts(
            run_id=bert_run_id,
            artifact_path="mappings/mapping_info.npz",
            dst_path=tmp_dir
        )
        info_data = np.load(info_path)
        total_tracks = int(info_data['total_tracks'])
        num_chunks = int(info_data['num_chunks'])
        chunk_size = int(info_data['chunk_size'])
        
        logger.info(f"Loading {total_tracks} tracks from {num_chunks} chunks")
        
        # Initialize mapping dictionaries
        uri_to_idx = {}
        idx_to_uri = {}
        
        # Load each chunk
        for i in range(num_chunks):
            logger.info(f"Loading chunk {i+1}/{num_chunks}...")
            chunk_dir = mlflow.artifacts.download_artifacts(
                run_id=bert_run_id,
                artifact_path=f"mappings/chunk_{i}",
                dst_path=tmp_dir
            )
            
            # Find the .npz file in the chunk directory
            chunk_files = [f for f in os.listdir(chunk_dir) if f.endswith('.npz')]
            if not chunk_files:
                raise FileNotFoundError(f"No .npz file found in chunk directory: {chunk_dir}")
            
            chunk_path = os.path.join(chunk_dir, chunk_files[0])
            chunk_data = np.load(chunk_path)
            
            # Get chunk data
            chunk_uris = chunk_data['track_uris']
            chunk_ids = chunk_data['track_ids']
            
            # Update mapping dictionaries
            for uri, idx in zip(chunk_uris, chunk_ids):
                uri_str = str(uri)
                idx_int = int(idx)
                uri_to_idx[uri_str] = idx_int
                idx_to_uri[idx_int] = uri_str
            
            logger.info(f"Loaded {len(chunk_uris)} tracks from chunk {i+1}")
        
        logger.info(f"Successfully loaded {len(uri_to_idx)} tracks")
        return uri_to_idx, idx_to_uri

def load_embeddings_from_mlflow() -> Dict[str, Any]:
    """Load embeddings from Matrix Factorization model"""
    try:
        # Get latest Matrix Factorization run ID
        mf_run_id = get_latest_run_id("matrix-factorization")
        logger.info(f"Loading embeddings from Matrix Factorization run {mf_run_id}")
        
        # Download embeddings
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Download embeddings
            embeddings_path = mlflow.artifacts.download_artifacts(
                run_id=mf_run_id,
                artifact_path="embeddings/item_embeddings.npz",
                dst_path=tmp_dir
            )
            
            # Load embeddings
            data = np.load(embeddings_path)
            embeddings = data['item_embeddings']
            track_ids = data['track_ids']  # 这些是 BERT 的索引
            
            logger.info(f"Successfully loaded {len(track_ids)} embeddings")
            
            return {
                'embeddings': embeddings,
                'track_ids': track_ids
            }
    
    except Exception as e:
        logger.error(f"Error loading embeddings: {str(e)}")
        raise

class MLPProjector(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 输出维度为1
        )

    def forward(self, x):
        return self.net(x)

def main():
    # Load track mapping from BERT
    uri_to_idx, idx_to_uri = load_track_mapping()
    
    # Load embeddings from Matrix Factorization
    emb_data = load_embeddings_from_mlflow()
    
    # Create training data
    X = emb_data['embeddings']
    # 直接使用 track_ids，因为它们已经是 BERT 的索引
    y = emb_data['track_ids']
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Convert to tensors and move to device
    X_tensor = torch.FloatTensor(X).to(device)
    Y_tensor = torch.FloatTensor(y).unsqueeze(1).to(device)  # 添加一个维度，使其形状为 [batch_size, 1]
    
    # Initialize model
    input_dim = X.shape[1]
    model = MLPProjector(input_dim)
    model.to(device)
    
    # Training loop
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 20
    
    # Train model
    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_params({
            "input_dim": input_dim,
            "hidden_dim": 256,
            "learning_rate": 1e-3,
            "epochs": 20,
            "num_samples": len(X)
        })
        
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
        
        # Save model
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "mlp_projector.pt")
            torch.save({
                "model_state": model.state_dict(),
                "input_dim": input_dim,
                "hidden_dim": 256,
                "mlflow_run_id": run.info.run_id
            }, model_path)
            mlflow.log_artifact(model_path, "models")
            logger.info(f"Model saved to MinIO through MLflow run {run.info.run_id}")
            
            # Save projected embeddings
            logger.info("Generating and saving projected embeddings...")
            model.eval()
            with torch.no_grad():
                # 确保在 CPU 上保存
                projected_embeddings = model(X_tensor).cpu().numpy()
                
                # Save in chunks
                chunk_size = 10000
                num_chunks = (len(projected_embeddings) + chunk_size - 1) // chunk_size
                
                for i in range(num_chunks):
                    start_idx = i * chunk_size
                    end_idx = min((i + 1) * chunk_size, len(projected_embeddings))
                    
                    chunk_path = os.path.join(tmp_dir, f"chunk_{i}.npz")
                    np.savez(
                        chunk_path,
                        embeddings=projected_embeddings[start_idx:end_idx],
                        track_ids=emb_data['track_ids'][start_idx:end_idx]
                    )
                    mlflow.log_artifact(chunk_path, f"embeddings/chunk_{i}")
                
                logger.info(f"Projected embeddings saved in {num_chunks} chunks")

if __name__ == "__main__":
    main()
