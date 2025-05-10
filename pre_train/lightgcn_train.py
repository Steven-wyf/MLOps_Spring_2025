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
from sklearn.metrics import ndcg_score, precision_score, recall_score, average_precision_score

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
EVAL_RESULTS_PATH = os.path.join(OUTPUT_DIR, "lightgcn_eval_results.json")

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
# Evaluation parameters
EVAL_TOP_K = [5, 10, 20]
NEG_SAMPLES = 100
# Dataset splits - 8:1:1 for train:val:test
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
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
    """Load playlist interaction data efficiently in chunks and create train/val/test splits"""
    logger.info(f"Loading playlist data from {PLAYLIST_DATA_PATH}")
    
    try:
        # Get total number of rows to determine chunks
        total_rows = sum(1 for _ in open(PLAYLIST_DATA_PATH, 'r')) - 1  # Subtract header
        logger.info(f"Total interactions to process: {total_rows}")
        
        # Load data in chunks to get playlist-track interactions
        chunk_indices = list(range(0, total_rows, CHUNK_SIZE))
        
        # Track all playlist-track interactions
        all_interactions = []
        
        # Process data in chunks to extract all interactions
        for i, chunk_start in enumerate(chunk_indices):
            # Determine rows to skip and number of rows to read
            skip_rows = chunk_start + 1 if i > 0 else 1  # Skip header for first chunk
            nrows = min(CHUNK_SIZE, total_rows - chunk_start)
            
            logger.info(f"Loading chunk {i+1}/{len(chunk_indices)} (rows {chunk_start+1}-{chunk_start+nrows})")
            
            # Read chunk
            chunk_df = pd.read_csv(
                PLAYLIST_DATA_PATH, 
                skiprows=skip_rows, 
                nrows=nrows,
                header=None if i > 0 else 0,
                names=['playlist_id', 'track_uri'] if i > 0 else None
            )
            
            # Add to interactions
            for _, row in chunk_df.iterrows():
                all_interactions.append((row['playlist_id'], row['track_uri']))
            
            # Check memory and clean up
            check_memory_usage()
            
            # Save interactions to disk if memory usage is high
            if psutil.virtual_memory().percent > MEMORY_LIMIT_PERCENT * 0.8:
                logger.info(f"Intermediate save of interactions (length: {len(all_interactions)})")
                temp_file = os.path.join(OUTPUT_DIR, f"lightgcn_interactions_temp_{i}.pkl")
                pd.DataFrame(all_interactions, columns=['playlist_id', 'track_uri']).to_pickle(temp_file)
                all_interactions = []
                gc.collect()
        
        # Load any saved interactions
        temp_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("lightgcn_interactions_temp_") and f.endswith(".pkl")]
        if temp_files:
            logger.info(f"Loading {len(temp_files)} saved interaction files")
            for temp_file in temp_files:
                temp_interactions = pd.read_pickle(os.path.join(OUTPUT_DIR, temp_file))
                all_interactions.extend(temp_interactions.values.tolist())
                os.remove(os.path.join(OUTPUT_DIR, temp_file))
        
        # Create a DataFrame with all interactions
        interactions_df = pd.DataFrame(all_interactions, columns=['playlist_id', 'track_uri'])
        
        # Group by playlist to handle playlist-level splitting
        playlist_groups = interactions_df.groupby('playlist_id')
        all_playlists = list(playlist_groups.groups.keys())
        
        # Shuffle playlists for random splitting
        np.random.seed(42)
        np.random.shuffle(all_playlists)
        
        # Determine split indices
        train_size = int(len(all_playlists) * TRAIN_RATIO)
        val_size = int(len(all_playlists) * VAL_RATIO)
        
        train_playlists = all_playlists[:train_size]
        val_playlists = all_playlists[train_size:train_size+val_size]
        test_playlists = all_playlists[train_size+val_size:]
        
        logger.info(f"Split data: Train: {len(train_playlists)} playlists, Val: {len(val_playlists)} playlists, Test: {len(test_playlists)} playlists")
        
        # Create dataframes for each split
        train_interactions = interactions_df[interactions_df['playlist_id'].isin(train_playlists)]
        val_interactions = interactions_df[interactions_df['playlist_id'].isin(val_playlists)]
        test_interactions = interactions_df[interactions_df['playlist_id'].isin(test_playlists)]
        
        logger.info(f"Training interactions: {len(train_interactions)}")
        logger.info(f"Validation interactions: {len(val_interactions)}")
        logger.info(f"Test interactions: {len(test_interactions)}")
        
        # Create ID mappings
        unique_playlists = interactions_df['playlist_id'].unique()
        unique_tracks = interactions_df['track_uri'].unique()
        
        playlist_to_idx = {playlist: i for i, playlist in enumerate(unique_playlists)}
        track_to_idx = {track: i + len(unique_playlists) for i, track in enumerate(unique_tracks)}
        
        # Create reverse mappings
        idx_to_playlist = {i: playlist for playlist, i in playlist_to_idx.items()}
        idx_to_track = {i: track for track, i in track_to_idx.items()}
        
        # Process each split to create edge indices
        train_edge_index = create_edge_index(train_interactions, playlist_to_idx, track_to_idx)
        val_edge_index = create_edge_index(val_interactions, playlist_to_idx, track_to_idx)
        test_edge_index = create_edge_index(test_interactions, playlist_to_idx, track_to_idx)
        
        # Create graph data objects
        train_data = Data(
            edge_index=train_edge_index,
            num_nodes=len(playlist_to_idx) + len(track_to_idx)
        )
        
        val_data = Data(
            edge_index=val_edge_index,
            num_nodes=len(playlist_to_idx) + len(track_to_idx)
        )
        
        test_data = Data(
            edge_index=test_edge_index,
            num_nodes=len(playlist_to_idx) + len(track_to_idx)
        )
        
        num_playlists = len(playlist_to_idx)
        num_tracks = len(track_to_idx)
        
        logger.info(f"Created graph with {num_playlists} playlists and {num_tracks} tracks")
        logger.info(f"Train edges: {train_edge_index.size(1)}, Val edges: {val_edge_index.size(1)}, Test edges: {test_edge_index.size(1)}")
        
        # Free memory
        del interactions_df, all_interactions
        gc.collect()
        
        return train_data, val_data, test_data, playlist_to_idx, track_to_idx, idx_to_playlist, idx_to_track
    
    except Exception as e:
        logger.error(f"Error loading and splitting playlist data: {str(e)}")
        raise

def create_edge_index(interactions_df, playlist_to_idx, track_to_idx):
    """Create edge index from interactions dataframe"""
    edge_list = []
    
    for _, row in interactions_df.iterrows():
        playlist_idx = playlist_to_idx[row['playlist_id']]
        track_idx = track_to_idx[row['track_uri']]
        
        # Add bidirectional edges
        edge_list.append([playlist_idx, track_idx])
        edge_list.append([track_idx, playlist_idx])
    
    return torch.tensor(edge_list, dtype=torch.long).t().contiguous()

def train_lightgcn(train_data, val_data, playlist_to_idx, track_to_idx):
    """Train LightGCN model on playlist-track interactions with validation"""
    num_playlists = len(playlist_to_idx)
    num_tracks = len(track_to_idx)
    num_nodes = num_playlists + num_tracks
    
    # Initialize model
    model = LightGCN(
        num_nodes=num_nodes,
        embedding_dim=EMBEDDING_DIM,
        num_layers=NUM_LAYERS
    ).to(DEVICE)
    
    # Move data to device
    train_data = train_data.to(DEVICE)
    val_data = val_data.to(DEVICE)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # Train model
    model.train()
    progress_bar = tqdm(range(EPOCHS), desc="Training LightGCN")
    
    best_val_loss = float('inf')
    best_model_state = None
    train_losses = []
    val_losses = []
    
    for epoch in progress_bar:
        # Forward pass
        embeddings = model(train_data.edge_index)
        
        # Calculate BPR loss (Bayesian Personalized Ranking)
        train_loss = calculate_bpr_loss(embeddings, train_data.edge_index, num_playlists, num_tracks)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        train_losses.append(train_loss.item())
        
        # Validate
        model.eval()
        with torch.no_grad():
            val_embeddings = model(val_data.edge_index)
            val_loss = calculate_bpr_loss(val_embeddings, val_data.edge_index, num_playlists, num_tracks)
            val_losses.append(val_loss.item())
        
        # Save best model
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_model_state = model.state_dict().copy()
        
        # Back to train mode
        model.train()
        
        progress_bar.set_postfix({
            "train_loss": train_loss.item(), 
            "val_loss": val_loss.item()
        })
        
        # Check memory periodically and clean up if needed
        if epoch % 10 == 0:
            check_memory_usage()
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses, best_val_loss

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

def evaluate_model(model, test_data, playlist_to_idx, track_to_idx):
    """Evaluate the model using various metrics"""
    model.eval()
    num_playlists = len(playlist_to_idx)
    num_tracks = len(track_to_idx)
    
    logger.info("Evaluating model on test set")
    
    # Group test edges by playlist
    test_edges = test_data.edge_index.cpu().numpy()
    playlist_to_tracks = {}
    
    # Only consider playlist->track edges
    for i in range(test_edges.shape[1]):
        src, dst = test_edges[0, i], test_edges[1, i]
        if src < num_playlists and dst >= num_playlists:
            playlist = src
            track = dst
            if playlist not in playlist_to_tracks:
                playlist_to_tracks[playlist] = []
            playlist_to_tracks[playlist].append(track)
    
    # Compute embeddings for all nodes
    with torch.no_grad():
        embeddings = model(test_data.edge_index.to(DEVICE)).cpu().numpy()
    
    # Setup metrics
    precision_k = {k: [] for k in EVAL_TOP_K}
    recall_k = {k: [] for k in EVAL_TOP_K}
    ndcg_k = {k: [] for k in EVAL_TOP_K}
    
    # Calculate metrics with sampling to reduce memory usage
    max_playlists_to_evaluate = min(5000, len(playlist_to_tracks))
    sampled_playlists = np.random.choice(list(playlist_to_tracks.keys()), max_playlists_to_evaluate, replace=False)
    
    for i, playlist_idx in enumerate(tqdm(sampled_playlists, desc="Evaluating")):
        # Skip playlists with too few tracks
        if len(playlist_to_tracks[playlist_idx]) < 2:
            continue
        
        # Get playlist embedding
        playlist_emb = embeddings[playlist_idx]
        
        # Calculate scores for all tracks
        track_scores = np.dot(embeddings[num_playlists:], playlist_emb)
        
        # Get ground truth tracks for this playlist
        true_tracks = np.array(playlist_to_tracks[playlist_idx]) - num_playlists
        
        # Get top-k predictions
        for k in EVAL_TOP_K:
            top_k_tracks = np.argsort(-track_scores)[:k]
            
            # Calculate precision@k
            relevant_tracks_in_k = np.isin(top_k_tracks, true_tracks).sum()
            precision_k[k].append(relevant_tracks_in_k / k)
            
            # Calculate recall@k
            recall_k[k].append(relevant_tracks_in_k / len(true_tracks))
            
            # Calculate NDCG@k
            # Create binary relevance vector
            y_true = np.zeros(len(track_scores))
            y_true[true_tracks] = 1
            
            # Select top k predictions
            top_k_indices = np.argsort(-track_scores)[:k]
            y_score = np.zeros(len(track_scores))
            y_score[top_k_indices] = 1
            
            # Calculate NDCG
            try:
                ndcg = ndcg_score(y_true.reshape(1, -1), y_score.reshape(1, -1), k=k)
                ndcg_k[k].append(ndcg)
            except:
                # Skip if no relevant tracks
                pass
        
        # Clean up memory periodically
        if i % 100 == 0:
            check_memory_usage()
    
    # Calculate average metrics
    results = {}
    for k in EVAL_TOP_K:
        results[f'precision@{k}'] = np.mean(precision_k[k])
        results[f'recall@{k}'] = np.mean(recall_k[k])
        results[f'ndcg@{k}'] = np.mean(ndcg_k[k])
    
    return results

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
    
    # Load playlist data in chunks with train/val/test split
    train_data, val_data, test_data, playlist_to_idx, track_to_idx, idx_to_playlist, idx_to_track = load_playlist_data_in_chunks()
    num_playlists = len(playlist_to_idx)
    num_tracks = len(track_to_idx)
    
    # Save split information for consistency across models
    split_info = {
        'train_playlists': list(set([idx_to_playlist[idx.item()] for idx in train_data.edge_index[0] if idx.item() < num_playlists])),
        'val_playlists': list(set([idx_to_playlist[idx.item()] for idx in val_data.edge_index[0] if idx.item() < num_playlists])),
        'test_playlists': list(set([idx_to_playlist[idx.item()] for idx in test_data.edge_index[0] if idx.item() < num_playlists]))
    }
    split_info_file = os.path.join(OUTPUT_DIR, "playlist_split_info.json")
    with open(split_info_file, 'w') as f:
        json.dump(split_info, f)
    logger.info(f"Saved playlist split information to {split_info_file}")
    
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
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "test_ratio": TEST_RATIO,
            "eval_top_k": EVAL_TOP_K,
            "device": str(DEVICE)
        }
        
        if bert_run_id:
            params["bert_run_id"] = bert_run_id
            
        mlflow.log_params(params)
        
        # Train model
        logger.info("Training LightGCN model...")
        model, train_losses, val_losses, best_val_loss = train_lightgcn(train_data, val_data, playlist_to_idx, track_to_idx)
        
        # Evaluate model
        logger.info("Evaluating model on test set...")
        eval_results = evaluate_model(model, test_data, playlist_to_idx, track_to_idx)
        
        # Save evaluation results
        with open(EVAL_RESULTS_PATH, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        # Extract track embeddings in batches
        logger.info("Extracting track embeddings...")
        track_embeddings_dict = extract_track_embeddings_in_batches(model, train_data, track_to_idx, num_playlists)
        
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
        mlflow.log_artifact(EVAL_RESULTS_PATH, "evaluation")
        
        # Log metrics
        # 1. Training metrics
        for i, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
            mlflow.log_metric("train_loss", train_loss, step=i)
            mlflow.log_metric("val_loss", val_loss, step=i)
        
        mlflow.log_metric("final_train_loss", train_losses[-1])
        mlflow.log_metric("final_val_loss", val_losses[-1])
        mlflow.log_metric("best_val_loss", best_val_loss)
        
        # 2. Evaluation metrics
        for metric_name, value in eval_results.items():
            mlflow.log_metric(metric_name, value)
        
        # Log completion time
        end_time = time.time()
        duration = end_time - start_time
        mlflow.log_metric("duration_seconds", duration)
        
        logger.info(f"LightGCN training completed in {duration:.2f} seconds")
        logger.info(f"Final memory usage: {check_memory_usage():.1f}%")
        
        # Log evaluation results
        for metric, value in eval_results.items():
            logger.info(f"{metric}: {value:.4f}")
        
        # Write model info for next steps
        model_info = {
            "lightgcn_run_id": run_id,
            "embedding_file": OUTPUT_NPZ_PATH,
            "model_file": MODEL_PATH,
            "embedding_dim": EMBEDDING_DIM,
            "num_tracks": num_tracks,
            "num_playlists": num_playlists,
            "evaluation": eval_results,
            "split_info_file": split_info_file
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