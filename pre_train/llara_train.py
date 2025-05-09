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
from tqdm import tqdm
from datetime import datetime
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Configure logging
LOG_DIR = os.environ.get("LOG_DIR", "./logs")
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{LOG_DIR}/llara_train_{datetime.now().strftime('%Y%m%d%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---- CONFIG SECTION ---- #
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:8000")
MLFLOW_S3_ENDPOINT_URL = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin")
EXPERIMENT_NAME = "llara-playlist-continuation"
RUN_ID = os.environ.get("RUN_ID", datetime.now().strftime('%Y%m%d%H%M%S'))

# Data paths
DATA_DIR = os.environ.get("PLAYLIST_DATA_DIR", "/mnt/block")
PLAYLIST_DATA_PATH = os.path.join(DATA_DIR, "playlist_data.csv")  # CSV with user-track interactions
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
PROJECTED_EMB_PATH = os.environ.get("PROJECTED_EMB", os.path.join(OUTPUT_DIR, "projected_lightgcn.npz"))
MODEL_PATH = os.path.join(OUTPUT_DIR, "llara_classifier.pt")
METRICS_PATH = os.path.join(OUTPUT_DIR, "llara_metrics.json")

# Check for previous model info
MLP_MODEL_INFO = os.path.join(OUTPUT_DIR, "mlp_model_info.json")

# Model parameters
HIDDEN_DIM = 512
DROPOUT = 0.3
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
EPOCHS = 50
BATCH_SIZE = 128
TEST_SIZE = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ------------------------ #

def configure_mlflow():
    """Set up MLflow tracking and artifact storage"""
    # Set environment variables for MinIO
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = MLFLOW_S3_ENDPOINT_URL
    os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
    os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY

    # Configure MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    logger.info(f"MLflow S3 endpoint URL: {MLFLOW_S3_ENDPOINT_URL}")
    logger.info(f"Experiment name: {EXPERIMENT_NAME}")

def load_data():
    """Load projected embeddings and playlist data for training"""
    # Load projected embeddings
    logger.info(f"Loading projected embeddings from {PROJECTED_EMB_PATH}")
    embedding_data = np.load(PROJECTED_EMB_PATH)
    track_ids = list(embedding_data.files)
    
    # Create track ID to index mapping
    track_to_idx = {track_id: i for i, track_id in enumerate(track_ids)}
    
    # Load playlist data
    logger.info(f"Loading playlist data from {PLAYLIST_DATA_PATH}")
    playlist_df = pd.read_csv(PLAYLIST_DATA_PATH)
    
    # Process playlist data to create training pairs
    logger.info("Creating playlist continuation training pairs")
    playlist_groups = playlist_df.groupby('playlist_id')
    
    # Prepare data for training
    input_tracks = []
    target_tracks = []
    
    for playlist_id, tracks in tqdm(playlist_groups, desc="Processing playlists"):
        track_list = tracks['track_uri'].tolist()
        
        if len(track_list) < 2:
            continue  # Skip playlists with only one track
        
        # Create training pairs: each track predicts the next track in the playlist
        for i in range(len(track_list) - 1):
            current_track = track_list[i]
            next_track = track_list[i + 1]
            
            # Check if both tracks have embeddings
            if current_track in track_to_idx and next_track in track_to_idx:
                input_tracks.append(current_track)
                target_tracks.append(next_track)
    
    logger.info(f"Created {len(input_tracks)} training pairs")
    
    # Convert to embeddings and target indices
    X = np.array([embedding_data[track] for track in input_tracks])
    y = np.array([track_to_idx[track] for track in target_tracks])
    
    return X, y, track_to_idx, track_ids

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

def train_llara(X, y):
    """Train LlaRA model for playlist continuation"""
    # Split into train, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42  # 0.25 * 0.8 = 0.2 of total
    )
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )
    
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    input_dim = X.shape[1]
    num_classes = len(np.unique(y))
    
    model = LlaRAClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT
    ).to(DEVICE)
    
    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Train model
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_acc = 0.0
    best_model_state = None
    
    logger.info("Starting LlaRA training...")
    progress_bar = tqdm(range(EPOCHS), desc="Training LlaRA")
    
    for epoch in progress_bar:
        # Training phase
        model.train()
        train_loss = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Calculate accuracy
        val_acc = accuracy_score(val_targets, val_preds)
        val_accuracies.append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
        
        progress_bar.set_postfix({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc
        })
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Evaluate on test set
    model.eval()
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    test_acc = accuracy_score(test_targets, test_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_targets, test_preds, average='macro'
    )
    
    # Calculate Top-K accuracy
    top_k_accs = []
    k_values = [5, 10, 20]
    
    with torch.no_grad():
        for k in k_values:
            top_k_correct = 0
            total = 0
            
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                
                # Get top-k predictions
                _, top_k_preds = torch.topk(outputs, k, dim=1)
                
                # Check if target is in top-k
                for i, target in enumerate(targets):
                    if target in top_k_preds[i]:
                        top_k_correct += 1
                
                total += targets.size(0)
            
            top_k_acc = top_k_correct / total
            top_k_accs.append(top_k_acc)
    
    metrics = {
        "accuracy": test_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "top_5_accuracy": top_k_accs[0],
        "top_10_accuracy": top_k_accs[1],
        "top_20_accuracy": top_k_accs[2],
        "best_val_accuracy": best_val_acc
    }
    
    return model, train_losses, val_losses, val_accuracies, metrics

def main():
    """Main LlaRA training function"""
    start_time = time.time()
    logger.info(f"Starting LlaRA training - Run ID: {RUN_ID}")
    logger.info(f"Using device: {DEVICE}")
    
    # Configure MLflow
    configure_mlflow()
    
    # Load data
    X, y, track_to_idx, track_ids = load_data()
    
    # Check for previous model info
    mlp_run_id = None
    bert_run_id = None
    lightgcn_run_id = None
    
    if os.path.exists(MLP_MODEL_INFO):
        with open(MLP_MODEL_INFO, 'r') as f:
            mlp_info = json.load(f)
            mlp_run_id = mlp_info.get('mlp_run_id')
            bert_run_id = mlp_info.get('bert_run_id')
            lightgcn_run_id = mlp_info.get('lightgcn_run_id')
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"llara-classifier-{RUN_ID}") as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run ID: {run_id}")
        
        # Log parameters
        params = {
            "hidden_dim": HIDDEN_DIM,
            "dropout": DROPOUT,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "test_size": TEST_SIZE,
            "embedding_dim": X.shape[1],
            "num_classes": len(np.unique(y)),
            "num_training_pairs": len(X),
            "device": str(DEVICE)
        }
        
        if mlp_run_id:
            params["mlp_run_id"] = mlp_run_id
        
        if bert_run_id:
            params["bert_run_id"] = bert_run_id
        
        if lightgcn_run_id:
            params["lightgcn_run_id"] = lightgcn_run_id
            
        mlflow.log_params(params)
        
        # Train model
        model, train_losses, val_losses, val_accuracies, metrics = train_llara(X, y)
        
        # Save model
        torch.save(model.state_dict(), MODEL_PATH)
        
        # Save metrics
        with open(METRICS_PATH, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Log artifacts
        mlflow.log_artifact(MODEL_PATH, "model")
        mlflow.log_artifact(METRICS_PATH, "metrics")
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        for i, (train_loss, val_loss, val_acc) in enumerate(zip(train_losses, val_losses, val_accuracies)):
            mlflow.log_metric("train_loss", train_loss, step=i)
            mlflow.log_metric("val_loss", val_loss, step=i)
            mlflow.log_metric("val_accuracy", val_acc, step=i)
        
        # Log completion time
        end_time = time.time()
        duration = end_time - start_time
        mlflow.log_metric("duration_seconds", duration)
        
        logger.info(f"LlaRA training completed in {duration:.2f} seconds")
        logger.info(f"Test accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Top-5 accuracy: {metrics['top_5_accuracy']:.4f}")
        logger.info(f"Top-10 accuracy: {metrics['top_10_accuracy']:.4f}")
        
        # Create a track ID lookup file for inference
        track_lookup = {idx: track_id for track_id, idx in track_to_idx.items()}
        with open(os.path.join(OUTPUT_DIR, "track_id_mapping.json"), "w") as f:
            json.dump(track_lookup, f)
        
        mlflow.log_artifact(os.path.join(OUTPUT_DIR, "track_id_mapping.json"), "mappings")
        
        # Write model info to file
        model_info = {
            "llara_run_id": run_id,
            "model_file": MODEL_PATH,
            "metrics_file": METRICS_PATH,
            "input_dim": X.shape[1],
            "hidden_dim": HIDDEN_DIM,
            "num_classes": len(np.unique(y)),
            "num_tracks": len(track_ids),
            "accuracy": metrics["accuracy"],
            "top_10_accuracy": metrics["top_10_accuracy"]
        }
        
        if mlp_run_id:
            model_info["mlp_run_id"] = mlp_run_id
        
        if bert_run_id:
            model_info["bert_run_id"] = bert_run_id
        
        if lightgcn_run_id:
            model_info["lightgcn_run_id"] = lightgcn_run_id
        
        with open(os.path.join(OUTPUT_DIR, "llara_model_info.json"), "w") as f:
            json.dump(model_info, f, indent=2)
        
        # Create a model version file
        model_version = datetime.now().strftime('%Y%m%d%H%M%S')
        with open(os.path.join(OUTPUT_DIR, "model_version.txt"), "w") as f:
            f.write(model_version)
        
        mlflow.log_artifact(os.path.join(OUTPUT_DIR, "model_version.txt"), "metadata")

if __name__ == "__main__":
    try:
        main()
        exit(0)
    except Exception as e:
        logger.exception(f"LlaRA training failed: {str(e)}")
        exit(1) 