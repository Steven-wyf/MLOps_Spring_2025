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
    """Load the latest MLP projected embeddings from MLflow"""
    try:
        # Get latest MLP run ID
        mlp_run_id = get_latest_run_id("mlp-projector")
        logger.info(f"Loading embeddings from MLP run {mlp_run_id}")
        
        # Download embeddings
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Download projected embeddings
            chunk_dir = mlflow.artifacts.download_artifacts(
                run_id=mlp_run_id,
                artifact_path="embeddings",
                dst_path=tmp_dir
            )
            
            # 遍历每个 chunk 目录
            found_files = False
            embeddings = None
            track_ids = None
            
            for chunk_name in os.listdir(chunk_dir):
                if chunk_name.startswith('chunk_'):
                    chunk_path = os.path.join(chunk_dir, chunk_name)
                    if os.path.isdir(chunk_path):
                        # 在 chunk 目录中查找 .npz 文件
                        for file_name in os.listdir(chunk_path):
                            if file_name.endswith('.npz'):
                                found_files = True
                                file_path = os.path.join(chunk_path, file_name)
                                try:
                                    chunk_data = np.load(file_path, allow_pickle=True)
                                    if embeddings is None:
                                        embeddings = chunk_data['embeddings']
                                        track_ids = chunk_data['track_ids']
                                    else:
                                        embeddings = np.concatenate([embeddings, chunk_data['embeddings']])
                                        track_ids = np.concatenate([track_ids, chunk_data['track_ids']])
                                except Exception as e:
                                    logger.error(f"Error loading file {file_name}: {str(e)}")
            
            if not found_files:
                raise Exception("No .npz files found in embeddings directory!")
            
            if embeddings is None:
                raise Exception("No embeddings loaded!")
            
            return {
                'embeddings': embeddings,
                'track_ids': track_ids
            }
    
    except Exception as e:
        logger.error(f"Error loading embeddings: {str(e)}")
        raise

def main():
    # Load track mapping from BERT
    uri_to_idx, idx_to_uri = load_track_mapping()
    
    # Load embeddings from MLP
    emb_data = load_embeddings_from_mlflow()
    
    # Create training data
    X = emb_data['embeddings']
    y = np.array([uri_to_idx[track_id] for track_id in emb_data['track_ids']])
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    Y_tensor = torch.FloatTensor(y)
    
    # Initialize model
    input_dim = X.shape[1]
    output_dim = Y_tensor.shape[1]
    model = MLPProjector(input_dim, output_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
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
            "output_dim": output_dim,
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
                "output_dim": output_dim,
                "hidden_dim": 256,
                "mlflow_run_id": run.info.run_id
            }, model_path)
            mlflow.log_artifact(model_path, "models")
            logger.info(f"Model saved to MinIO through MLflow run {run.info.run_id}")

if __name__ == "__main__":
    main()
