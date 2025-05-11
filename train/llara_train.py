#!/usr/bin/env python
# llara_train.py - Playlist continuation prediction model
import os
import mlflow
import torch
import numpy as np
import pandas as pd
import json
import logging
import time
import tempfile
from tqdm import tqdm
from datetime import datetime
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, ndcg_score
from typing import Dict, List, Tuple, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://129.114.25.37:8000/")
MLFLOW_S3_ENDPOINT_URL = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://129.114.25.37:9000")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "your-acccess-key")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "hrwbqzUS85G253yKi43T")
EXPERIMENT_NAME = "llara-classifier"
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/mnt/object/outputs")

# Model parameters
HIDDEN_DIM = 512
DROPOUT = 0.3
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
EPOCHS = 50
BATCH_SIZE = 128
EVAL_TOP_K = [5, 10, 20]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
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

def load_playlist_data() -> pd.DataFrame:
    """Load and process playlist data from local file"""
    DATA_DIR = os.path.expanduser(os.environ.get("PLAYLIST_DATA_DIR", "~/processed_data"))
    data_path = os.path.join(DATA_DIR, "playlist_track_pairs.csv")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Playlist data not found at: {data_path}")
    
    return pd.read_csv(data_path).dropna()

def create_training_pairs(df: pd.DataFrame, track_to_idx: Dict[str, int]) -> List[Tuple[int, int]]:
    """Create training pairs from playlist data"""
    playlists = df.groupby('playlist_id')['track_uri'].apply(list)
    samples = []
    
    for plist in playlists:
        if len(plist) < 2:
            continue
        for i in range(1, len(plist)):
            context = plist[i - 1]
            target = plist[i]
            if context in track_to_idx and target in track_to_idx:
                samples.append((track_to_idx[context], track_to_idx[target]))
    
    return samples

class LlaRAClassifier(nn.Module):
    """Linear layer with Regularization and Activation (LlaRA) for playlist continuation"""
    def __init__(self, input_dim, num_classes, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)

def evaluate_model(model, test_loader, device, num_classes, top_k_values):
    """Evaluate model with various metrics"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Basic classification metrics
    metrics = {
        "accuracy": accuracy_score(all_targets, all_preds),
        "precision": precision_recall_fscore_support(all_targets, all_preds, average='macro')[0],
        "recall": precision_recall_fscore_support(all_targets, all_preds, average='macro')[1],
        "f1": precision_recall_fscore_support(all_targets, all_preds, average='macro')[2]
    }
    
    # Ranking metrics
    for k in top_k_values:
        hits = 0
        total = 0
        ndcg_values = []
        
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            
            # Get top-k predictions
            _, top_k_preds = torch.topk(outputs, k, dim=1)
            
            # Calculate hits
            for i in range(len(targets)):
                if targets[i] in top_k_preds[i]:
                    hits += 1
                total += 1
                
                # Calculate NDCG
                target_one_hot = torch.zeros(num_classes, device=device)
                target_one_hot[targets[i]] = 1
                try:
                    ndcg = ndcg_score(
                        target_one_hot.cpu().numpy().reshape(1, -1),
                        outputs[i].cpu().numpy().reshape(1, -1),
                        k=k
                    )
                    ndcg_values.append(ndcg)
                except:
                    pass
        
        metrics[f"top_{k}_accuracy"] = hits / total if total > 0 else 0
        metrics[f"ndcg@{k}"] = np.mean(ndcg_values) if ndcg_values else 0
    
    return metrics

def train_model(model, train_loader, val_loader, test_loader, device, num_classes):
    """Train the LLaRA model"""
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    best_model_state = None
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in tqdm(range(EPOCHS), desc="Training LLaRA"):
        # Training
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        val_acc = accuracy_score(val_targets, val_preds)
        val_accuracies.append(val_acc)
        
        # Log metrics
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
    
    # Load best model for evaluation
    model.load_state_dict(best_model_state)
    
    # Final evaluation
    metrics = evaluate_model(model, test_loader, device, num_classes, EVAL_TOP_K)
    
    return model, train_losses, val_losses, val_accuracies, metrics

def main():
    # Load data
    logger.info("Loading embeddings and playlist data...")
    # 1. 加载 BERT 的 track mapping
    uri_to_idx, idx_to_uri = load_track_mapping()
    # 2. 加载 MLP 的 embeddings
    emb_data = load_embeddings_from_mlflow()
    # 3. 加载播放列表数据
    df = load_playlist_data()
    
    # Create training pairs using BERT's track mapping
    logger.info("Creating training pairs...")
    samples = create_training_pairs(df, uri_to_idx)
    
    # Prepare data
    # 使用 BERT 的 track mapping 来获取 embeddings
    X = emb_data['embeddings'][[uri_to_idx[i] for i, _ in samples]]
    y = np.array([uri_to_idx[j] for _, j in samples])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Create data loaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    input_dim = X.shape[1]
    num_classes = len(uri_to_idx)  # 使用 BERT 的 track 数量
    model = LlaRAClassifier(input_dim, num_classes, HIDDEN_DIM, DROPOUT).to(DEVICE)
    
    # Train model
    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_params({
            "input_dim": input_dim,
            "num_classes": num_classes,
            "hidden_dim": HIDDEN_DIM,
            "dropout": DROPOUT,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "num_samples": len(samples)
        })
        
        # Train and evaluate
        best_model_state, metrics = train_model(model, train_loader, val_loader, test_loader, DEVICE, num_classes)
        
        # Log metrics
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)
        
        # Save model
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "llara_model.pt")
            torch.save({
                "model_state": best_model_state,
                "input_dim": input_dim,
                "num_classes": num_classes,
                "hidden_dim": HIDDEN_DIM,
                "dropout": DROPOUT,
                "mlflow_run_id": run.info.run_id,
                "track_to_idx": uri_to_idx,  # 保存 BERT 的 track mapping
                "idx_to_track": idx_to_uri   # 保存反向映射
            }, model_path)
            mlflow.log_artifact(model_path, "models")
            logger.info(f"Model saved to MinIO through MLflow run {run.info.run_id}")

if __name__ == "__main__":
    main() 