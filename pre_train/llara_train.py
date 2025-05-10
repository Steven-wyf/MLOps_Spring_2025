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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, ndcg_score

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
LIGHTGCN_MODEL_INFO = os.path.join(OUTPUT_DIR, "lightgcn_model_info.json")

# Model parameters
HIDDEN_DIM = 512
DROPOUT = 0.3
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
EPOCHS = 50
BATCH_SIZE = 128
# Data split ratios - same as LightGCN
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
# Evaluation parameters
EVAL_TOP_K = [5, 10, 20]
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
    """Load and prepare data for LlaRA training with consistent dataset splits"""
    # STEP 1: Load pre-computed embeddings from MLP projection
    logger.info(f"Loading projected embeddings from {PROJECTED_EMB_PATH}")
    embedding_data = np.load(PROJECTED_EMB_PATH)
    track_ids = list(embedding_data.files)
    
    # Create a mapping from track URI to numeric index
    track_to_idx = {track_id: i for i, track_id in enumerate(track_ids)}
    
    # STEP 2: Load playlist data
    logger.info(f"Loading playlist data from {PLAYLIST_DATA_PATH}")
    playlist_df = pd.read_csv(PLAYLIST_DATA_PATH)
    
    # STEP 3: Use LightGCN splits or create our own
    # Try to load existing splits from the LightGCN model for consistency
    split_info = get_lightgcn_splits()
    
    # Group playlists for processing
    logger.info("Creating playlist continuation training pairs")
    playlist_groups = playlist_df.groupby('playlist_id')
    all_playlists = list(playlist_groups.groups.keys())
    
    # STEP 4: Get train/val/test playlist assignments
    if split_info and 'train_playlists' in split_info:
        # Use existing split from LightGCN
        train_playlists = split_info['train_playlists']
        val_playlists = split_info['val_playlists']
        test_playlists = split_info['test_playlists']
        logger.info("Using playlist splits from LightGCN")
    else:
        # Create new split with same 8:1:1 ratio
        logger.info("Creating new playlist split with 8:1:1 ratio")
        train_playlists, val_playlists, test_playlists = create_new_split(all_playlists)
    
    # Log split statistics
    logger.info(f"Data split: {len(train_playlists)} train, {len(val_playlists)} validation, {len(test_playlists)} test playlists")
    
    # STEP 5: Create training pairs for each split
    train_inputs, train_targets = [], []
    val_inputs, val_targets = [], []
    test_inputs, test_targets = [], []
    
    # Process each playlist to create input-output pairs
    for playlist_id, tracks in tqdm(playlist_groups, desc="Processing playlists"):
        # Get ordered tracks in this playlist
        track_list = tracks['track_uri'].tolist()
        
        # Skip playlists with only one track (need at least 2 for sequence prediction)
        if len(track_list) < 2:
            continue
        
        # For each track in playlist, predict the next track
        for i in range(len(track_list) - 1):
            current_track = track_list[i]      # Input: current track
            next_track = track_list[i + 1]     # Target: next track
            
            # Skip if either track doesn't have an embedding
            if current_track not in track_to_idx or next_track not in track_to_idx:
                continue
                
            # Add to appropriate split based on playlist ID
            if playlist_id in train_playlists:
                train_inputs.append(current_track)
                train_targets.append(next_track)
            elif playlist_id in val_playlists:
                val_inputs.append(current_track)
                val_targets.append(next_track)
            elif playlist_id in test_playlists:
                test_inputs.append(current_track)
                test_targets.append(next_track)
    
    # STEP 6: Convert track IDs to embeddings and indices
    # Log data statistics
    logger.info(f"Created {len(train_inputs)} training, {len(val_inputs)} validation, and {len(test_inputs)} test pairs")
    
    # Convert input tracks to their embeddings
    X_train = np.array([embedding_data[track] for track in train_inputs])
    X_val = np.array([embedding_data[track] for track in val_inputs])
    X_test = np.array([embedding_data[track] for track in test_inputs])
    
    # Convert target tracks to their indices
    y_train = np.array([track_to_idx[track] for track in train_targets])
    y_val = np.array([track_to_idx[track] for track in val_targets])
    y_test = np.array([track_to_idx[track] for track in test_targets])
    
    return X_train, X_val, X_test, y_train, y_val, y_test, track_to_idx, track_ids

def get_lightgcn_splits():
    """Attempt to load playlist splits from LightGCN for consistency"""
    try:
        if os.path.exists(LIGHTGCN_MODEL_INFO):
            with open(LIGHTGCN_MODEL_INFO, 'r') as f:
                lightgcn_info = json.load(f)
                
                # If split file exists, load it
                if 'split_info_file' in lightgcn_info and os.path.exists(lightgcn_info['split_info_file']):
                    with open(lightgcn_info['split_info_file'], 'r') as sf:
                        return json.load(sf)
    except Exception as e:
        logger.warning(f"Could not load LightGCN split info: {str(e)}")
    
    return None

def create_new_split(playlists):
    """Create a new 8:1:1 train/val/test split of playlists"""
    # Shuffle playlists randomly
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(playlists)
    
    # Calculate sizes for each split
    train_size = int(len(playlists) * TRAIN_RATIO)
    val_size = int(len(playlists) * VAL_RATIO)
    
    # Split the playlists
    train_playlists = playlists[:train_size]
    val_playlists = playlists[train_size:train_size+val_size]
    test_playlists = playlists[train_size+val_size:]
    
    # Save split info for future reference
    split_info = {
        'train_playlists': train_playlists,
        'val_playlists': val_playlists,
        'test_playlists': test_playlists
    }
    
    # Save to file
    split_info_file = os.path.join(OUTPUT_DIR, "playlist_split_info.json")
    with open(split_info_file, 'w') as f:
        json.dump(split_info, f)
    logger.info(f"Saved new playlist split information to {split_info_file}")
    
    return train_playlists, val_playlists, test_playlists

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

def train_llara(X_train, X_val, X_test, y_train, y_val, y_test):
    """Train LlaRA model for playlist continuation with pre-split data"""
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
    input_dim = X_train.shape[1]
    all_targets = np.concatenate([y_train, y_val, y_test])
    num_classes = len(np.unique(all_targets))
    
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
    
    # Training tracking variables
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_acc = 0.0
    best_model_state = None
    
    logger.info("Starting LlaRA training...")
    progress_bar = tqdm(range(EPOCHS), desc="Training LlaRA")
    
    # ---- TRAINING LOOP ---- #
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
    
    # Load best model for evaluation
    model.load_state_dict(best_model_state)
    
    # ---- EVALUATION PHASE ---- #
    logger.info("Evaluating model on test set...")
    model.eval()
    
    # Calculate standard classification metrics
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(targets.cpu().numpy())
    
    # Basic classification metrics
    acc = accuracy_score(test_targets, test_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_targets, test_preds, average='macro'
    )
    
    # Store results
    results = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "best_val_accuracy": best_val_acc
    }
    
    # ---- CALCULATE RANKING METRICS (Precision@k, Recall@k, NDCG@k) ---- #
    logger.info("Calculating ranking metrics...")
    
    with torch.no_grad():
        # For each k value (5, 10, 20)
        for k in EVAL_TOP_K:
            # Initialize metric trackers
            precision_values = []  # Precision@k values
            recall_values = []     # Recall@k values
            ndcg_values = []       # NDCG@k values
            hits = 0               # Count of correct predictions in top-k
            total = 0              # Total predictions made
            
            # Process each batch in test set
            for inputs, targets in test_loader:
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)
                batch_size = inputs.size(0)
                
                # Get model predictions (logits)
                logits = model(inputs)
                
                # Get top-k predictions for each sample
                _, top_k_preds = torch.topk(logits, k, dim=1)
                
                # Evaluate each prediction in the batch
                for i in range(batch_size):
                    target = targets[i].item()
                    pred_logits = logits[i].cpu().numpy()
                    
                    # Create ground truth one-hot vector
                    # 1 = correct class, 0 = incorrect class
                    ground_truth = np.zeros(num_classes)
                    ground_truth[target] = 1
                    
                    # Check if correct class is in top-k predictions
                    if target in top_k_preds[i]:
                        hits += 1
                    
                    # Get top-k predicted classes
                    top_k_indices = np.argsort(-pred_logits)[:k]
                    
                    # For a recommendation task, calculate:
                    # 1. Precision@k = relevant items retrieved / k
                    # 2. Recall@k = relevant items retrieved / total relevant
                    # 3. NDCG@k = normalized discounted cumulative gain
                    
                    # Since we have a single relevant item (the true class),
                    # precision@k = 1/k if correct, 0 if incorrect
                    # recall@k = 1 if correct, 0 if incorrect
                    
                    if ground_truth[top_k_indices].sum() > 0:
                        # There's at least one relevant item in the top-k
                        precision_values.append(ground_truth[top_k_indices].sum() / k)
                        recall_values.append(1.0)  # We found the item
                        
                        # Calculate NDCG
                        try:
                            ndcg = ndcg_score(
                                ground_truth.reshape(1, -1),
                                pred_logits.reshape(1, -1),
                                k=k
                            )
                            ndcg_values.append(ndcg)
                        except:
                            pass
                    else:
                        # No relevant items in top-k
                        precision_values.append(0.0)
                        recall_values.append(0.0)
                
                total += batch_size
            
            # Calculate average metrics
            hit_rate = hits / total if total > 0 else 0
            results[f"top_{k}_accuracy"] = hit_rate
            
            # Calculate average precision, recall, and NDCG at k
            if precision_values:
                results[f"precision@{k}"] = np.mean(precision_values)
            else:
                results[f"precision@{k}"] = 0.0
                
            if recall_values:
                results[f"recall@{k}"] = np.mean(recall_values)
            else:
                results[f"recall@{k}"] = 0.0
                
            if ndcg_values:
                results[f"ndcg@{k}"] = np.mean(ndcg_values)
            else:
                results[f"ndcg@{k}"] = 0.0
    
    return model, train_losses, val_losses, val_accuracies, results

def main():
    """Main LlaRA training function"""
    start_time = time.time()
    logger.info(f"Starting LlaRA training - Run ID: {RUN_ID}")
    logger.info(f"Using device: {DEVICE}")
    
    # Configure MLflow
    configure_mlflow()
    
    # Load pre-split data
    X_train, X_val, X_test, y_train, y_val, y_test, track_to_idx, track_ids = load_data()
    
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
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "test_ratio": TEST_RATIO,
            "embedding_dim": X_train.shape[1],
            "num_train_samples": len(X_train),
            "num_val_samples": len(X_val),
            "num_test_samples": len(X_test),
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
        model, train_losses, val_losses, val_accuracies, metrics = train_llara(X_train, X_val, X_test, y_train, y_val, y_test)
        
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
        
        # Log top-k accuracy metrics 
        for k in EVAL_TOP_K:
            logger.info(f"Top-{k} accuracy: {metrics[f'top_{k}_accuracy']:.4f}")
        
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
            "input_dim": X_train.shape[1],
            "hidden_dim": HIDDEN_DIM,
            "num_classes": len(np.unique(np.concatenate([y_train, y_val, y_test]))),
            "num_tracks": len(track_ids),
            "accuracy": metrics["accuracy"],
            "evaluation": {
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"]
            }
        }
        
        # Add all top-k metrics to evaluation
        for k in EVAL_TOP_K:
            model_info["evaluation"][f"top_{k}_accuracy"] = metrics[f"top_{k}_accuracy"]
            model_info["evaluation"][f"precision@{k}"] = metrics[f"precision@{k}"]
            model_info["evaluation"][f"recall@{k}"] = metrics[f"recall@{k}"]
            model_info["evaluation"][f"ndcg@{k}"] = metrics[f"ndcg@{k}"]
        
        if mlp_run_id:
            model_info["mlp_run_id"] = mlp_run_id
        
        if bert_run_id:
            model_info["bert_run_id"] = bert_run_id
        
        if lightgcn_run_id:
            model_info["lightgcn_run_id"] = lightgcn_run_id
        
        # For consistency with LightGCN
        model_info["split_info_file"] = os.path.join(OUTPUT_DIR, "playlist_split_info.json")
        
        with open(os.path.join(OUTPUT_DIR, "llara_model_info.json"), "w") as f:
            json.dump(model_info, f, indent=2)
        
        # Log enhanced evaluation metrics
        logger.info("Evaluation metrics:")
        for k in EVAL_TOP_K:
            logger.info(f"Precision@{k}: {metrics[f'precision@{k}']:.4f}")
            logger.info(f"Recall@{k}: {metrics[f'recall@{k}']:.4f}")
            logger.info(f"NDCG@{k}: {metrics[f'ndcg@{k}']:.4f}")
        
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