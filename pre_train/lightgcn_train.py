#!/usr/bin/env python
# lightgcn_train.py - User-track graph neural network
import os
import mlflow
import torch
import numpy as np
import pandas as pd
import json
import logging
import time
import networkx as nx
from tqdm import tqdm
from datetime import datetime
from torch_geometric.nn import LightGCN
from torch_geometric.data import Data
import scipy.sparse as sp

# Configure logging
LOG_DIR = os.environ.get("LOG_DIR", "./logs")
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{LOG_DIR}/lightgcn_train_{datetime.now().strftime('%Y%m%d%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---- CONFIG SECTION ---- #
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:8000")
MLFLOW_S3_ENDPOINT_URL = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin")
EXPERIMENT_NAME = "lightgcn-user-track"
RUN_ID = os.environ.get("RUN_ID", datetime.now().strftime('%Y%m%d%H%M%S'))

# Data paths
DATA_DIR = os.environ.get("PLAYLIST_DATA_DIR", "/mnt/block")
PLAYLIST_DATA_PATH = os.path.join(DATA_DIR, "playlist_data.csv")  # Assuming CSV with user-track interactions
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_NPZ_PATH = os.path.join(OUTPUT_DIR, "lightgcn_embeddings.npz")
MODEL_PATH = os.path.join(OUTPUT_DIR, "lightgcn_model.pt")

# Check for BERT output
BERT_MODEL_INFO = os.path.join(OUTPUT_DIR, "bert_model_info.json")

# Model parameters
EMBEDDING_DIM = 128
NUM_LAYERS = 3
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
EPOCHS = 100
BATCH_SIZE = 1024
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

def load_playlist_data():
    """Load playlist interaction data and create user-track graph"""
    logger.info(f"Loading playlist data from {PLAYLIST_DATA_PATH}")
    
    try:
        # Load playlist data (CSV with columns: playlist_id, track_uri)
        playlist_df = pd.read_csv(PLAYLIST_DATA_PATH)
        logger.info(f"Loaded {len(playlist_df)} playlist-track interactions")
        
        # Create user and track ID mappings
        unique_playlists = playlist_df['playlist_id'].unique()
        unique_tracks = playlist_df['track_uri'].unique()
        
        playlist_to_idx = {playlist: i for i, playlist in enumerate(unique_playlists)}
        track_to_idx = {track: i + len(unique_playlists) for i, track in enumerate(unique_tracks)}
        
        # Create reverse mappings
        idx_to_playlist = {i: playlist for playlist, i in playlist_to_idx.items()}
        idx_to_track = {i: track for track, i in track_to_idx.items()}
        
        # Create edge indices (bipartite graph between playlists and tracks)
        edge_index = []
        
        for _, row in tqdm(playlist_df.iterrows(), total=len(playlist_df), desc="Creating graph"):
            playlist_idx = playlist_to_idx[row['playlist_id']]
            track_idx = track_to_idx[row['track_uri']]
            
            # Bidirectional edges
            edge_index.append([playlist_idx, track_idx])
            edge_index.append([track_idx, playlist_idx])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Create graph data object
        data = Data(
            edge_index=edge_index,
            num_nodes=len(playlist_to_idx) + len(track_to_idx)
        )
        
        return data, playlist_to_idx, track_to_idx, idx_to_playlist, idx_to_track
    
    except Exception as e:
        logger.error(f"Error loading playlist data: {str(e)}")
        raise

def train_lightgcn(data, num_users, num_items):
    """Train LightGCN model on playlist-track interactions"""
    # Initialize model
    model = LightGCN(
        num_nodes=num_users + num_items,
        embedding_dim=EMBEDDING_DIM,
        num_layers=NUM_LAYERS
    ).to(DEVICE)
    
    # Move data to device
    data = data.to(DEVICE)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # Train model
    model.train()
    progress_bar = tqdm(range(EPOCHS), desc="Training LightGCN")
    
    losses = []
    for epoch in progress_bar:
        # Forward pass
        embeddings = model(data.edge_index)
        
        # Calculate BPR loss (Bayesian Personalized Ranking)
        loss = calculate_bpr_loss(embeddings, data.edge_index, num_users, num_items)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        progress_bar.set_postfix({"loss": loss.item()})
    
    return model, losses

def calculate_bpr_loss(embeddings, edge_index, num_users, num_items):
    """Calculate Bayesian Personalized Ranking loss for LightGCN"""
    # Sample negative edges
    pos_edge_index = edge_index
    
    # Take only user->item edges (first half of edges)
    user_indices = pos_edge_index[0, :num_users]
    item_indices = pos_edge_index[1, :num_users]
    
    # Sample negative items for each user
    neg_item_indices = torch.randint(
        num_users, num_users + num_items, 
        (len(user_indices),), 
        device=DEVICE
    )
    
    # Calculate scores for positive and negative interactions
    pos_scores = (embeddings[user_indices] * embeddings[item_indices]).sum(dim=1)
    neg_scores = (embeddings[user_indices] * embeddings[neg_item_indices]).sum(dim=1)
    
    # BPR loss
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
    
    return loss

def main():
    """Main LightGCN training function"""
    start_time = time.time()
    logger.info(f"Starting LightGCN training - Run ID: {RUN_ID}")
    logger.info(f"Using device: {DEVICE}")
    
    # Configure MLflow
    configure_mlflow()
    
    # Load playlist data
    data, playlist_to_idx, track_to_idx, idx_to_playlist, idx_to_track = load_playlist_data()
    num_playlists = len(playlist_to_idx)
    num_tracks = len(track_to_idx)
    
    logger.info(f"Created graph with {num_playlists} playlists and {num_tracks} tracks")
    logger.info(f"Total nodes: {num_playlists + num_tracks}, Edges: {data.edge_index.size(1)}")
    
    # Check if BERT info is available
    bert_run_id = None
    if os.path.exists(BERT_MODEL_INFO):
        with open(BERT_MODEL_INFO, 'r') as f:
            bert_info = json.load(f)
            bert_run_id = bert_info.get('bert_run_id')
            logger.info(f"Found BERT model info, run ID: {bert_run_id}")
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"lightgcn-embeddings-{RUN_ID}") as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run ID: {run_id}")
        
        # Log parameters
        params = {
            "embedding_dim": EMBEDDING_DIM,
            "num_layers": NUM_LAYERS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "num_playlists": num_playlists,
            "num_tracks": num_tracks,
            "device": str(DEVICE)
        }
        
        if bert_run_id:
            params["bert_run_id"] = bert_run_id
            
        mlflow.log_params(params)
        
        # Train model
        logger.info("Training LightGCN model...")
        model, losses = train_lightgcn(data, num_playlists, num_tracks)
        
        # Extract track embeddings
        model.eval()
        with torch.no_grad():
            all_embeddings = model(data.edge_index)
            track_embeddings = all_embeddings[num_playlists:].cpu().numpy()
        
        logger.info(f"Extracted embeddings for {num_tracks} tracks")
        
        # Save track embeddings to NPZ file
        track_embeddings_dict = {}
        for track_id, idx in track_to_idx.items():
            # Adjust index to account for playlist indices
            adj_idx = idx - num_playlists
            if adj_idx >= 0 and adj_idx < len(track_embeddings):
                track_embeddings_dict[track_id] = track_embeddings[adj_idx]
        
        logger.info(f"Saving {len(track_embeddings_dict)} track embeddings to {OUTPUT_NPZ_PATH}")
        np.savez(OUTPUT_NPZ_PATH, **track_embeddings_dict)
        
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'playlist_to_idx': playlist_to_idx,
            'track_to_idx': track_to_idx,
            'embedding_dim': EMBEDDING_DIM,
            'num_layers': NUM_LAYERS
        }, MODEL_PATH)
        
        # Log artifacts
        mlflow.log_artifact(OUTPUT_NPZ_PATH, "embeddings")
        mlflow.log_artifact(MODEL_PATH, "model")
        
        # Log metrics
        train_losses = np.array(losses)
        mlflow.log_metric("final_loss", losses[-1])
        mlflow.log_metric("min_loss", np.min(train_losses))
        
        for i, loss in enumerate(losses):
            mlflow.log_metric("train_loss", loss, step=i)
        
        # Log completion time
        end_time = time.time()
        duration = end_time - start_time
        mlflow.log_metric("duration_seconds", duration)
        
        logger.info(f"LightGCN training completed in {duration:.2f} seconds")
        
        # Write model info for next steps
        model_info = {
            "lightgcn_run_id": run_id,
            "embedding_file": OUTPUT_NPZ_PATH,
            "model_file": MODEL_PATH,
            "embedding_dim": EMBEDDING_DIM,
            "num_tracks": num_tracks,
            "num_playlists": num_playlists
        }
        
        if bert_run_id:
            model_info["bert_run_id"] = bert_run_id
            
        with open(os.path.join(OUTPUT_DIR, "lightgcn_model_info.json"), "w") as f:
            json.dump(model_info, f, indent=2)

if __name__ == "__main__":
    try:
        main()
        exit(0)
    except Exception as e:
        logger.exception(f"LightGCN training failed: {str(e)}")
        exit(1) 