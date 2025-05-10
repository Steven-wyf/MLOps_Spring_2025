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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://129.114.25.37:8000/")
MLFLOW_S3_ENDPOINT_URL = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://129.114.25.37:9000")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "admin")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "XCqPacaUHUur82cNZI1R")
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

# Configure MLflow
os.environ["MLFLOW_S3_ENDPOINT_URL"] = MLFLOW_S3_ENDPOINT_URL
os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

def load_latest_embeddings():
    """Load the latest MLP projected embeddings from MLflow"""
    # Get the latest MLP run
    mlp_runs = mlflow.search_runs(
        experiment_names=["mlp-projector"],
        filter_string="status = 'FINISHED'",
        order_by=["start_time DESC"]
    )
    if mlp_runs.empty:
        raise ValueError("No MLP runs found")
    mlp_run_id = mlp_runs.iloc[0].run_id
    
    # Download embeddings
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Download projected embeddings
        emb_path = mlflow.artifacts.download_artifacts(
            run_id=mlp_run_id,
            artifact_path="embeddings/projected_embeddings.npz",
            dst_path=tmp_dir
        )
        emb_data = np.load(emb_path)
    
    return emb_data

def load_playlist_data():
    """Load and process playlist data"""
    playlist_file = os.path.join(OUTPUT_DIR, "playlist_track_list.csv")
    logger.info(f"Loading playlist data from {playlist_file}")
    
    df = pd.read_csv(playlist_file).dropna()
    return df

def create_training_pairs(df, track_to_idx):
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
    """Main training function"""
    logger.info("Starting LLaRA training")
    logger.info(f"Using device: {DEVICE}")
    
    # Load data
    emb_data = load_latest_embeddings()
    track_ids = list(emb_data.files)
    track_to_idx = {tid: i for i, tid in enumerate(track_ids)}
    
    df = load_playlist_data()
    samples = create_training_pairs(df, track_to_idx)
    
    # Split data
    train_samples, temp_samples = train_test_split(samples, test_size=0.2, random_state=42)
    val_samples, test_samples = train_test_split(temp_samples, test_size=0.5, random_state=42)
    
    # Create datasets
    X_train = np.array([emb_data[track_ids[i]] for i, _ in train_samples])
    y_train = np.array([j for _, j in train_samples])
    
    X_val = np.array([emb_data[track_ids[i]] for i, _ in val_samples])
    y_val = np.array([j for _, j in val_samples])
    
    X_test = np.array([emb_data[track_ids[i]] for i, _ in test_samples])
    y_test = np.array([j for _, j in test_samples])
    
    # Create dataloaders
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    model = LlaRAClassifier(
        input_dim=X_train.shape[1],
        num_classes=len(track_ids),
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT
    ).to(DEVICE)
    
    # Start MLflow run
    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_params({
            "hidden_dim": HIDDEN_DIM,
            "dropout": DROPOUT,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "input_dim": X_train.shape[1],
            "num_classes": len(track_ids),
            "num_train_samples": len(X_train),
            "num_val_samples": len(X_val),
            "num_test_samples": len(X_test)
        })
        
        # Train model
        model, train_losses, val_losses, val_accuracies, metrics = train_model(
            model, train_loader, val_loader, test_loader, DEVICE, len(track_ids)
        )
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Save model to MinIO through MLflow
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "llara_classifier.pt")
            torch.save({
                "model_state": model.state_dict(),
                "input_dim": X_train.shape[1],
                "num_classes": len(track_ids),
                "hidden_dim": HIDDEN_DIM,
                "dropout": DROPOUT
            }, model_path)
            mlflow.log_artifact(model_path, "models")
            
            # Save track ID mapping
            track_mapping = {idx: track_id for track_id, idx in track_to_idx.items()}
            mapping_path = os.path.join(tmp_dir, "track_id_mapping.json")
            with open(mapping_path, "w") as f:
                json.dump(track_mapping, f)
            mlflow.log_artifact(mapping_path, "mappings")
        
        logger.info("Training completed successfully!")
        logger.info(f"Test accuracy: {metrics['accuracy']:.4f}")
        for k in EVAL_TOP_K:
            logger.info(f"Top-{k} accuracy: {metrics[f'top_{k}_accuracy']:.4f}")

if __name__ == "__main__":
    try:
        main()
        exit(0)
    except Exception as e:
        logger.exception(f"LLaRA training failed: {str(e)}")
        exit(1) 