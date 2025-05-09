#!/usr/bin/env python
# mlp_train.py - MLP projector between embedding spaces
import os
import mlflow
import torch
import numpy as np
import json
import logging
import time
from tqdm import tqdm
from datetime import datetime
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Configure logging
LOG_DIR = os.environ.get("LOG_DIR", "./logs")
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{LOG_DIR}/mlp_train_{datetime.now().strftime('%Y%m%d%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---- CONFIG SECTION ---- #
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:8000")
MLFLOW_S3_ENDPOINT_URL = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin")
EXPERIMENT_NAME = "mlp-projector"
RUN_ID = os.environ.get("RUN_ID", datetime.now().strftime('%Y%m%d%H%M%S'))

# Data paths
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
BERT_NPZ_PATH = os.environ.get("BERT_NPZ", os.path.join(OUTPUT_DIR, "bert_track_embeddings.npz"))
LIGHTGCN_NPZ_PATH = os.environ.get("LIGHTGCN_NPZ", os.path.join(OUTPUT_DIR, "lightgcn_embeddings.npz"))
OUTPUT_NPZ_PATH = os.path.join(OUTPUT_DIR, "projected_lightgcn.npz")
MODEL_PATH = os.path.join(OUTPUT_DIR, "mlp_projector.pt")

# Check for previous model info
BERT_MODEL_INFO = os.path.join(OUTPUT_DIR, "bert_model_info.json")
LIGHTGCN_MODEL_INFO = os.path.join(OUTPUT_DIR, "lightgcn_model_info.json")

# Model parameters
HIDDEN_LAYERS = [512, 256]
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
EPOCHS = 100
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

def load_embeddings():
    """Load BERT and LightGCN embeddings and align them"""
    logger.info(f"Loading BERT embeddings from {BERT_NPZ_PATH}")
    bert_data = np.load(BERT_NPZ_PATH)
    bert_track_ids = list(bert_data.files)
    
    logger.info(f"Loading LightGCN embeddings from {LIGHTGCN_NPZ_PATH}")
    lightgcn_data = np.load(LIGHTGCN_NPZ_PATH)
    lightgcn_track_ids = list(lightgcn_data.files)
    
    # Find common track IDs
    common_track_ids = sorted(set(bert_track_ids) & set(lightgcn_track_ids))
    logger.info(f"Found {len(common_track_ids)} tracks in both embedding spaces")
    
    if len(common_track_ids) == 0:
        raise ValueError("No common tracks found between BERT and LightGCN embeddings")
    
    # Create aligned embedding matrices
    bert_embs = []
    lightgcn_embs = []
    
    for track_id in common_track_ids:
        bert_embs.append(bert_data[track_id])
        lightgcn_embs.append(lightgcn_data[track_id])
    
    bert_embs = np.vstack(bert_embs)
    lightgcn_embs = np.vstack(lightgcn_embs)
    
    # Get embedding dimensions
    bert_dim = bert_embs.shape[1]
    lightgcn_dim = lightgcn_embs.shape[1]
    
    logger.info(f"BERT embedding dimension: {bert_dim}")
    logger.info(f"LightGCN embedding dimension: {lightgcn_dim}")
    
    return bert_embs, lightgcn_embs, common_track_ids, bert_dim, lightgcn_dim

class MLPProjector(nn.Module):
    """MLP model to project from BERT to LightGCN embedding space"""
    def __init__(self, input_dim, output_dim, hidden_dims):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def train_mlp(bert_embs, lightgcn_embs):
    """Train MLP model to project BERT embeddings to LightGCN space"""
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        bert_embs, lightgcn_embs, test_size=TEST_SIZE, random_state=42
    )
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    model = MLPProjector(
        input_dim=bert_embs.shape[1],
        output_dim=lightgcn_embs.shape[1],
        hidden_dims=HIDDEN_LAYERS
    ).to(DEVICE)
    
    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    criterion = nn.MSELoss()
    
    # Train model
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    
    logger.info("Starting MLP training...")
    progress_bar = tqdm(range(EPOCHS), desc="Training MLP")
    
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
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
        
        progress_bar.set_postfix({
            "train_loss": train_loss,
            "val_loss": val_loss
        })
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses, best_val_loss

def main():
    """Main MLP training function"""
    start_time = time.time()
    logger.info(f"Starting MLP projector training - Run ID: {RUN_ID}")
    logger.info(f"Using device: {DEVICE}")
    
    # Configure MLflow
    configure_mlflow()
    
    # Load embeddings
    bert_embs, lightgcn_embs, common_track_ids, bert_dim, lightgcn_dim = load_embeddings()
    
    # Check for previous model info
    bert_run_id = None
    lightgcn_run_id = None
    
    if os.path.exists(BERT_MODEL_INFO):
        with open(BERT_MODEL_INFO, 'r') as f:
            bert_info = json.load(f)
            bert_run_id = bert_info.get('bert_run_id')
    
    if os.path.exists(LIGHTGCN_MODEL_INFO):
        with open(LIGHTGCN_MODEL_INFO, 'r') as f:
            lightgcn_info = json.load(f)
            lightgcn_run_id = lightgcn_info.get('lightgcn_run_id')
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"mlp-projector-{RUN_ID}") as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run ID: {run_id}")
        
        # Log parameters
        params = {
            "hidden_layers": HIDDEN_LAYERS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "test_size": TEST_SIZE,
            "bert_dim": bert_dim,
            "lightgcn_dim": lightgcn_dim,
            "num_common_tracks": len(common_track_ids),
            "device": str(DEVICE)
        }
        
        if bert_run_id:
            params["bert_run_id"] = bert_run_id
        
        if lightgcn_run_id:
            params["lightgcn_run_id"] = lightgcn_run_id
            
        mlflow.log_params(params)
        
        # Train model
        model, train_losses, val_losses, best_val_loss = train_mlp(bert_embs, lightgcn_embs)
        
        # Project all BERT embeddings to LightGCN space
        logger.info("Projecting all BERT embeddings to LightGCN space...")
        bert_data = np.load(BERT_NPZ_PATH)
        projected_embeddings = {}
        
        # Process in batches
        batch_size = 128
        all_track_ids = list(bert_data.files)
        
        for i in range(0, len(all_track_ids), batch_size):
            batch_track_ids = all_track_ids[i:i+batch_size]
            
            # Get batch embeddings
            batch_embs = np.array([bert_data[tid] for tid in batch_track_ids])
            
            # Project embeddings
            with torch.no_grad():
                batch_tensor = torch.tensor(batch_embs, dtype=torch.float32).to(DEVICE)
                projected = model(batch_tensor).cpu().numpy()
            
            # Store projected embeddings
            for j, tid in enumerate(batch_track_ids):
                projected_embeddings[tid] = projected[j]
        
        # Save projected embeddings
        logger.info(f"Saving {len(projected_embeddings)} projected embeddings to {OUTPUT_NPZ_PATH}")
        np.savez(OUTPUT_NPZ_PATH, **projected_embeddings)
        
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'hidden_layers': HIDDEN_LAYERS,
            'input_dim': bert_dim,
            'output_dim': lightgcn_dim
        }, MODEL_PATH)
        
        # Log artifacts
        mlflow.log_artifact(OUTPUT_NPZ_PATH, "embeddings")
        mlflow.log_artifact(MODEL_PATH, "model")
        
        # Log metrics
        mlflow.log_metric("best_val_loss", best_val_loss)
        
        for i, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
            mlflow.log_metric("train_loss", train_loss, step=i)
            mlflow.log_metric("val_loss", val_loss, step=i)
        
        # Log completion time
        end_time = time.time()
        duration = end_time - start_time
        mlflow.log_metric("duration_seconds", duration)
        
        logger.info(f"MLP projection completed in {duration:.2f} seconds")
        
        # Write model info to file
        model_info = {
            "mlp_run_id": run_id,
            "embedding_file": OUTPUT_NPZ_PATH,
            "model_file": MODEL_PATH,
            "input_dim": bert_dim,
            "output_dim": lightgcn_dim,
            "hidden_layers": HIDDEN_LAYERS,
            "num_tracks": len(projected_embeddings)
        }
        
        if bert_run_id:
            model_info["bert_run_id"] = bert_run_id
        
        if lightgcn_run_id:
            model_info["lightgcn_run_id"] = lightgcn_run_id
        
        with open(os.path.join(OUTPUT_DIR, "mlp_model_info.json"), "w") as f:
            json.dump(model_info, f, indent=2)

if __name__ == "__main__":
    try:
        main()
        exit(0)
    except Exception as e:
        logger.exception(f"MLP training failed: {str(e)}")
        exit(1) 