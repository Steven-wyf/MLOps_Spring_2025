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
import gc
import psutil

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
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "128"))
NUM_LAYERS = int(os.environ.get("NUM_LAYERS", "3"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.001"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-5"))
EPOCHS = int(os.environ.get("EPOCHS", "100"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1024"))
# Data loading in chunks to reduce memory usage
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "100000"))
# Memory management
MEMORY_LIMIT_PERCENT = float(os.environ.get("MEMORY_LIMIT_PERCENT", "85.0"))
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

def check_memory_usage():
    """Check current memory usage and force garbage collection if necessary"""
    memory_info = psutil.virtual_memory()
    memory_percent = memory_info.percent
    
    if memory_percent > MEMORY_LIMIT_PERCENT:
        logger.warning(f"Memory usage high ({memory_percent:.1f}%). Triggering garbage collection.")
        gc.collect()
        torch.cuda.empty_cache()
        memory_info = psutil.virtual_memory()
        logger.info(f"Memory usage after GC: {memory_info.percent:.1f}%")
        
    return memory_info.percent

def load_playlist_data_in_chunks():
    """Load playlist interaction data efficiently in chunks"""
    logger.info(f"Loading playlist data from {PLAYLIST_DATA_PATH}")
    
    try:
        # Get total number of rows to determine chunks
        total_rows = sum(1 for _ in open(PLAYLIST_DATA_PATH, 'r')) - 1  # Subtract header
        logger.info(f"Total interactions to process: {total_rows}")
        
        # Load data in chunks
        chunk_indices = list(range(0, total_rows, CHUNK_SIZE))
        
        # Create dictionaries to map playlist and track IDs to indices
        playlist_to_idx = {}
        track_to_idx = {}
        next_playlist_idx = 0
        next_track_idx = 0
        
        # Lists to store edge indices
        edge_list = []
        
        # Process data in chunks
        for i, chunk_start in enumerate(chunk_indices):
            # Determine rows to skip and number of rows to read
            skip_rows = chunk_start + 1 if i > 0 else 1  # Skip header for first chunk
            nrows = min(CHUNK_SIZE, total_rows - chunk_start)
            
            logger.info(f"Processing chunk {i+1}/{len(chunk_indices)} (rows {chunk_start+1}-{chunk_start+nrows})")
            
            # Read chunk
            chunk_df = pd.read_csv(
                PLAYLIST_DATA_PATH, 
                skiprows=skip_rows, 
                nrows=nrows,
                header=None if i > 0 else 0,
                names=['playlist_id', 'track_uri'] if i > 0 else None
            )
            
            # Process interactions in chunk
            for _, row in tqdm(chunk_df.iterrows(), total=len(chunk_df), desc=f"Processing chunk {i+1}"):
                playlist_id = row['playlist_id']
                track_uri = row['track_uri']
                
                # Get or assign indices
                if playlist_id not in playlist_to_idx:
                    playlist_to_idx[playlist_id] = next_playlist_idx
                    next_playlist_idx += 1
                
                if track_uri not in track_to_idx:
                    track_to_idx[track_uri] = next_track_idx
                    next_track_idx += 1
                
                playlist_idx = playlist_to_idx[playlist_id]
                track_idx = track_to_idx[track_uri]
                
                # Store edge (bidirectional)
                edge_list.append([playlist_idx, track_idx + next_playlist_idx])
                edge_list.append([track_idx + next_playlist_idx, playlist_idx])
            
            # Check memory and clean up
            check_memory_usage()
            
            # Save intermediate edge list if memory usage is high
            if psutil.virtual_memory().percent > MEMORY_LIMIT_PERCENT * 0.8:
                logger.info(f"Intermediate save of edge list (length: {len(edge_list)})")
                temp_edge_file = os.path.join(OUTPUT_DIR, f"lightgcn_edges_temp_{i}.npy")
                np.save(temp_edge_file, np.array(edge_list))
                edge_list = []
                gc.collect()
        
        # Create reverse mappings
        idx_to_playlist = {i: playlist for playlist, i in playlist_to_idx.items()}
        idx_to_track = {i: track for track, i in track_to_idx.items()}
        
        # Load any saved edge lists
        temp_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("lightgcn_edges_temp_") and f.endswith(".npy")]
        if temp_files:
            logger.info(f"Loading {len(temp_files)} saved edge lists")
            for temp_file in temp_files:
                temp_edges = np.load(os.path.join(OUTPUT_DIR, temp_file))
                edge_list.extend(temp_edges.tolist())
                os.remove(os.path.join(OUTPUT_DIR, temp_file))
        
        # Create edge index tensor
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        # Create graph data object with sparse representation
        data = Data(
            edge_index=edge_index,
            num_nodes=len(playlist_to_idx) + len(track_to_idx)
        )
        
        num_playlists = len(playlist_to_idx)
        num_tracks = len(track_to_idx)
        
        logger.info(f"Created graph with {num_playlists} playlists and {num_tracks} tracks")
        logger.info(f"Total nodes: {num_playlists + num_tracks}, Edges: {edge_index.size(1)}")
        
        # Free memory
        del edge_list
        gc.collect()
        
        return data, playlist_to_idx, track_to_idx, idx_to_playlist, idx_to_track
    
    except Exception as e:
        logger.error(f"Error loading playlist data: {str(e)}")
        raise

def train_lightgcn(data, num_users, num_items):
    """Train LightGCN model on playlist-track interactions with memory optimization"""
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
        
        # Check memory periodically and clean up if needed
        if epoch % 10 == 0:
            check_memory_usage()
    
    return model, losses

def calculate_bpr_loss(embeddings, edge_index, num_users, num_items):
    """Calculate Bayesian Personalized Ranking loss for LightGCN with memory optimization"""
    # Sample negative edges
    pos_edge_index = edge_index
    
    # Take only user->item edges from the first half of edges
    # Use sampling to reduce memory usage
    num_edges = min(num_users * 10, pos_edge_index.size(1) // 2)  # Limit sample size
    indices = torch.randperm(pos_edge_index.size(1) // 2)[:num_edges].to(DEVICE)
    
    user_indices = pos_edge_index[0, indices]
    item_indices = pos_edge_index[1, indices]
    
    # Filter valid user->item edges
    mask = (user_indices < num_users) & (item_indices >= num_users)
    user_indices = user_indices[mask]
    item_indices = item_indices[mask]
    
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

def extract_track_embeddings_in_batches(model, data, track_to_idx, num_playlists):
    """Extract track embeddings in batches to reduce memory usage"""
    model.eval()
    
    # Get all track IDs and their adjusted indices
    track_ids = list(track_to_idx.keys())
    track_indices = [track_to_idx[tid] - num_playlists for tid in track_ids]
    
    # Process in batches
    track_embeddings_dict = {}
    batch_size = 1000
    
    for i in tqdm(range(0, len(track_ids), batch_size), desc="Extracting embeddings"):
        batch_track_ids = track_ids[i:i+batch_size]
        
        # Generate embeddings for all nodes
        with torch.no_grad():
            all_embeddings = model(data.edge_index)
        
        # Extract embeddings for current batch
        for j, track_id in enumerate(batch_track_ids):
            idx = track_to_idx[track_id]
            if idx >= num_playlists:
                track_embeddings_dict[track_id] = all_embeddings[idx].cpu().numpy()
        
        # Clear GPU memory
        del all_embeddings
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        check_memory_usage()
    
    return track_embeddings_dict

def main():
    """Main LightGCN training function"""
    start_time = time.time()
    logger.info(f"Starting LightGCN training - Run ID: {RUN_ID}")
    logger.info(f"Using device: {DEVICE}")
    logger.info(f"Memory usage before starting: {check_memory_usage():.1f}%")
    logger.info(f"Batch size: {BATCH_SIZE}, Chunk size: {CHUNK_SIZE}")
    
    # Configure MLflow
    configure_mlflow()
    
    # Load playlist data in chunks
    data, playlist_to_idx, track_to_idx, idx_to_playlist, idx_to_track = load_playlist_data_in_chunks()
    num_playlists = len(playlist_to_idx)
    num_tracks = len(track_to_idx)
    
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
            "chunk_size": CHUNK_SIZE,
            "memory_limit_percent": MEMORY_LIMIT_PERCENT,
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
        
        # Extract track embeddings in batches
        logger.info("Extracting track embeddings...")
        track_embeddings_dict = extract_track_embeddings_in_batches(model, data, track_to_idx, num_playlists)
        
        logger.info(f"Saving {len(track_embeddings_dict)} track embeddings to {OUTPUT_NPZ_PATH}")
        np.savez(OUTPUT_NPZ_PATH, **track_embeddings_dict)
        
        # Save model
        save_dict = {
            'model_state_dict': model.state_dict(),
            'embedding_dim': EMBEDDING_DIM,
            'num_layers': NUM_LAYERS
        }
        
        # Save mappings in separate files to reduce memory usage
        playlist_map_path = os.path.join(OUTPUT_DIR, "lightgcn_playlist_map.json")
        track_map_path = os.path.join(OUTPUT_DIR, "lightgcn_track_map.json")
        
        with open(playlist_map_path, 'w') as f:
            json.dump(playlist_to_idx, f)
        
        with open(track_map_path, 'w') as f:
            json.dump(track_to_idx, f)
        
        save_dict['playlist_map_path'] = playlist_map_path
        save_dict['track_map_path'] = track_map_path
        
        logger.info(f"Saving model to {MODEL_PATH}")
        torch.save(save_dict, MODEL_PATH)
        
        # Log artifacts
        mlflow.log_artifact(OUTPUT_NPZ_PATH, "embeddings")
        mlflow.log_artifact(MODEL_PATH, "model")
        mlflow.log_artifact(playlist_map_path, "mappings")
        mlflow.log_artifact(track_map_path, "mappings")
        
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
        logger.info(f"Final memory usage: {check_memory_usage():.1f}%")
        
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