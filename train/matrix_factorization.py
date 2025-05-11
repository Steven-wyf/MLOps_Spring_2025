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

# Load and preprocess data
DATA_DIR = os.path.expanduser(os.environ.get("PLAYLIST_DATA_DIR", "~/processed_data"))
df = pd.read_csv(os.path.join(DATA_DIR, "playlist_track_pairs.csv"))
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

df['user_id'] = user_encoder.fit_transform(df['playlist_id'])
df['item_id'] = item_encoder.fit_transform(df['track_uri'])

# Split data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Model parameters
num_users = len(user_encoder.classes_)
num_items = len(item_encoder.classes_)
embedding_dim = 32
learning_rate = 0.001
epochs = 10
batch_size = 256

# Initialize model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
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
    
    # Use only positive interactions for evaluation
    eval_df = test_df.copy()
    
    # Prepare evaluation data
    user_ids = torch.tensor(eval_df["user_id"].values, dtype=torch.long).to(device)
    item_ids = torch.tensor(eval_df["item_id"].values, dtype=torch.long).to(device)
    labels = torch.ones(len(eval_df), dtype=torch.float32).to(device)  # All positive interactions

    # Batch evaluation
    eval_batch_size = 1024  # Can be adjusted based on available memory
    all_preds = []
    all_labels = []
    
    for i in tqdm(range(0, len(eval_df), eval_batch_size), desc="Evaluating"):
        batch_user_ids = user_ids[i:i+eval_batch_size]
        batch_item_ids = item_ids[i:i+eval_batch_size]
        batch_labels = labels[i:i+eval_batch_size]

        # Get predictions
        with torch.no_grad():
            preds = model(batch_user_ids, batch_item_ids)
        
        # Store predictions and labels
        all_preds.append(preds.cpu().numpy())
        all_labels.append(batch_labels.cpu().numpy())
    
    # Concatenate all batches
    preds_np = np.concatenate(all_preds)
    labels_np = np.concatenate(all_labels)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(labels_np, preds_np))
    avg_prec = average_precision_score(labels_np, preds_np)

    # Log evaluation metrics
    mlflow.log_metrics({
        "test_rmse": rmse,
        "test_avg_precision": avg_prec
    })

    # Print evaluation results
    logger.info(f"Test RMSE: {rmse:.4f}")
    logger.info(f"Test Average Precision: {avg_prec:.4f}")
    
    # Save model to MinIO through MLflow
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
        logger.info(f"Model saved to MinIO through MLflow run {run.info.run_id}")
        
        # Save embeddings in chunks
        logger.info("Saving embeddings in chunks...")
        embeddings_dict = {
            'user_embeddings': model.user_embedding.weight.cpu().numpy(),
            'item_embeddings': model.item_embedding.weight.cpu().numpy(),
            'user_mapping': dict(enumerate(user_encoder.classes_)),
            'item_mapping': dict(enumerate(item_encoder.classes_))
        }
        save_embeddings_to_minio(embeddings_dict, run.info.run_id)
        
        # Save model info
        model_info_path = os.path.join(tmp_dir, "mf_model_info.pt")
        torch.save({
            "num_users": num_users,
            "num_items": num_items,
            "embedding_dim": embedding_dim,
            "mlflow_run_id": run.info.run_id,
            "test_rmse": float(rmse),
            "test_avg_precision": float(avg_prec)
        }, model_info_path)
        mlflow.log_artifact(model_info_path, "models")

logger.info("Training and evaluation completed successfully!") 