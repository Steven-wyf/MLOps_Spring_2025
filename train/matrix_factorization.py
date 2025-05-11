import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, average_precision_score
import mlflow
import os
import logging
import tempfile
from tqdm import tqdm
import boto3
import io
from typing import Tuple, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://129.114.25.37:8000/")
MLFLOW_S3_ENDPOINT_URL = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://129.114.25.37:9000")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "your-acccess-key")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "hrwbqzUS85G253yKi43T")
EXPERIMENT_NAME = "matrix-factorization"

# Configure MLflow
os.environ["MLFLOW_S3_ENDPOINT_URL"] = MLFLOW_S3_ENDPOINT_URL
os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

def save_to_minio(data, filename, bucket='mlflow-artifacts'):
    """Save data directly to MinIO using boto3"""
    s3_client = boto3.client('s3',
        endpoint_url=MLFLOW_S3_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )
    
    # 将数据保存到内存中的文件
    buffer = io.BytesIO()
    if isinstance(data, dict) and 'model_state' in data:
        # 如果是模型，使用 torch.save
        torch.save(data, buffer)
    else:
        # 如果是 numpy 数组，使用 np.savez
        np.savez(buffer, **data)
    buffer.seek(0)
    
    # 上传到 MinIO
    s3_client.upload_fileobj(
        buffer,
        bucket,
        filename
    )
    logger.info(f"Saved {filename} to MinIO bucket {bucket}")

def save_embeddings_to_minio(embeddings_dict, run_id):
    """Save embeddings to MinIO using MLflow"""
    print("Starting to save embeddings...")
    
    # 将 embeddings 分成较小的块
    chunk_size = 1000  # 每个文件保存1000个embeddings
    items = list(embeddings_dict.items())
    chunks = [dict(items[i:i + chunk_size]) for i in range(0, len(items), chunk_size)]
    
    print(f"Saving {len(chunks)} chunks of embeddings...")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, chunk in enumerate(chunks):
            print(f"Saving chunk {i+1}/{len(chunks)}...")
            # 保存每个块到临时文件
            temp_npz = os.path.join(tmp_dir, f"mf_embeddings_chunk_{i}.npz")
            np.savez(temp_npz, **chunk)
            print(f"Saved chunk to {temp_npz}")
            
            # 上传到 MinIO
            print(f"Uploading chunk {i+1} to MinIO...")
            try:
                mlflow.log_artifact(temp_npz, f"embeddings/chunk_{i}")
                print(f"Successfully uploaded chunk {i+1}")
            except Exception as e:
                print(f"Error uploading chunk {i+1}: {str(e)}")
                # 如果上传失败，保存到本地
                backup_path = f"mf_embeddings_chunk_{i}_backup.npz"
                np.savez(backup_path, **chunk)
                print(f"Saved backup to {backup_path}")
        
        print(f"All chunks processed. Total chunks: {len(chunks)}")

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
    # Get the latest BERT run
    bert_run_id = get_latest_run_id("bert-track-embeddings")
    
    # Download the mapping
    with tempfile.TemporaryDirectory() as tmp_dir:
        mapping_path = mlflow.artifacts.download_artifacts(
            run_id=bert_run_id,
            artifact_path="mappings/track_uri_mapping.npz",  # 确保与 BERT 保存的路径一致
            dst_path=tmp_dir
        )
        mapping_data = np.load(mapping_path)
        track_uris = mapping_data['track_uris']
        track_ids = mapping_data['track_ids']
        
        # Create mapping dictionaries
        uri_to_idx = {str(uri): int(idx) for uri, idx in zip(track_uris, track_ids)}
        idx_to_uri = {int(idx): str(uri) for uri, idx in zip(track_uris, track_ids)}
        
        logger.info(f"Loaded track mapping with {len(uri_to_idx)} tracks")
        logger.info(f"Sample track URIs: {list(uri_to_idx.keys())[:3]}")
        logger.info(f"Sample track IDs: {list(uri_to_idx.values())[:3]}")
        
        return uri_to_idx, idx_to_uri

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(MatrixFactorization, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
    
    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        return torch.sum(user_emb * item_emb, dim=1)

def main():
    # Load track mapping from BERT
    uri_to_idx, idx_to_uri = load_track_mapping()
    
    # Load and preprocess data
    DATA_DIR = os.path.expanduser(os.environ.get("PLAYLIST_DATA_DIR", "~/processed_data"))
    df = pd.read_csv(os.path.join(DATA_DIR, "playlist_track_pairs.csv"))
    
    # Use BERT's track mapping
    df['item_id'] = df['track_uri'].map(uri_to_idx)
    df = df.dropna(subset=['item_id'])  # Remove tracks not in BERT mapping
    
    # Encode users
    user_encoder = LabelEncoder()
    df['user_id'] = user_encoder.fit_transform(df['playlist_id'])
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Model parameters
    num_users = len(user_encoder.classes_)
    num_items = len(uri_to_idx)  # Use BERT's number of items
    embedding_dim = 32
    learning_rate = 0.001
    epochs = 10
    batch_size = 256
    
    # Initialize model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model = MatrixFactorization(num_users, num_items, embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Training loop with MLflow
    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_params({
            "embedding_dim": embedding_dim,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "num_users": num_users,
            "num_items": num_items
        })
        
        model.train()
        for epoch in tqdm(range(epochs), desc="Training Matrix Factorization"):
            total_loss = 0
            num_batches = 0
            
            # Process in batches
            for i in range(0, len(train_df), batch_size):
                batch = train_df.iloc[i:i+batch_size]
                
                # Prepare batch data
                user_ids = torch.tensor(batch['user_id'].values, dtype=torch.long).to(device)
                item_ids = torch.tensor(batch['item_id'].values, dtype=torch.long).to(device)
                ratings = torch.ones(len(batch), dtype=torch.float).to(device)  # Implicit feedback
                
                # Forward pass
                predictions = model(user_ids, item_ids)
                loss = criterion(predictions, ratings)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            # Log metrics
            avg_loss = total_loss / num_batches
            mlflow.log_metric("loss", avg_loss, step=epoch)
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Evaluation
        logger.info("Starting evaluation...")
        model.eval()
        
        # Prepare evaluation data
        user_ids = torch.tensor(test_df["user_id"].values, dtype=torch.long).to(device)
        item_ids = torch.tensor(test_df["item_id"].values, dtype=torch.long).to(device)
        labels = torch.ones(len(test_df), dtype=torch.float32).to(device)
        
        # Batch evaluation
        eval_batch_size = 1024
        all_preds = []
        all_labels = []
        
        for i in tqdm(range(0, len(test_df), eval_batch_size), desc="Evaluating"):
            batch_user_ids = user_ids[i:i+eval_batch_size]
            batch_item_ids = item_ids[i:i+eval_batch_size]
            batch_labels = labels[i:i+eval_batch_size]
            
            with torch.no_grad():
                preds = model(batch_user_ids, batch_item_ids)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch_labels.cpu().numpy())
        
        # Calculate metrics
        preds_np = np.concatenate(all_preds)
        labels_np = np.concatenate(all_labels)
        rmse = np.sqrt(mean_squared_error(labels_np, preds_np))
        avg_prec = average_precision_score(labels_np, preds_np)
        
        # Log evaluation metrics
        mlflow.log_metrics({
            "test_rmse": rmse,
            "test_avg_precision": avg_prec
        })
        
        logger.info(f"Test RMSE: {rmse:.4f}")
        logger.info(f"Test Average Precision: {avg_prec:.4f}")
        
        # Save model and embeddings
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save model
            model_path = os.path.join(tmp_dir, "mf_model.pt")
            torch.save({
                "model_state": model.state_dict(),
                "num_users": num_users,
                "num_items": num_items,
                "embedding_dim": embedding_dim
            }, model_path)
            mlflow.log_artifact(model_path, "models")
            
            # Save embeddings with BERT's track mapping
            embeddings_path = os.path.join(tmp_dir, "item_embeddings.npz")
            np.savez(
                embeddings_path,
                item_embeddings=model.item_embedding.weight.detach().cpu().numpy(),
                track_uris=list(uri_to_idx.keys()),
                track_ids=list(uri_to_idx.values())
            )
            mlflow.log_artifact(embeddings_path, "embeddings")
            
            logger.info(f"Saved model and embeddings to MLflow run {run.info.run_id}")

if __name__ == "__main__":
    main() 