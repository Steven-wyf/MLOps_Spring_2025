#!/usr/bin/env python
# bert_encoding.py - BERT track text encoding
import os
import mlflow
import torch
import numpy as np
import json
import tempfile
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- CONFIG SECTION ---- #
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://129.114.25.37:8000/")
MLFLOW_S3_ENDPOINT_URL = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://129.114.25.37:9000")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "your-acccess-key")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "hrwbqzUS85G253yKi43T")
EXPERIMENT_NAME = "bert-track-embeddings"

# Data paths
DATA_DIR = os.path.expanduser(os.environ.get("PLAYLIST_DATA_DIR", "~/processed_data"))
TRACK_JSON_PATH = os.path.join(DATA_DIR, "track_texts.json")  # JSON file with track information

# Model parameters
MAX_LEN = 128
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ------------------------ #

# Set up MLflow
os.environ["MLFLOW_S3_ENDPOINT_URL"] = MLFLOW_S3_ENDPOINT_URL
os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

def encode_tracks(tracks):
    """Encode track URIs to integers using LabelEncoder."""
    le = LabelEncoder()
    track_ids = le.fit_transform(tracks)
    
    # Save the label encoder mapping
    with tempfile.TemporaryDirectory() as tmp_dir:
        mapping_path = os.path.join(tmp_dir, "track_uri_mapping.npz")
        np.savez(
            mapping_path,
            track_uris=le.classes_,
            track_ids=np.arange(len(le.classes_))
        )
        mlflow.log_artifact(mapping_path, "mappings")
        logger.info(f"Saved track URI mapping with {len(le.classes_)} tracks")
    
    return track_ids, le

def process_tracks(tracks, embeddings):
    """Process tracks and save embeddings with encoded IDs."""
    # Encode track URIs
    track_ids, le = encode_tracks(tracks)
    
    # Save embeddings with encoded IDs
    with tempfile.TemporaryDirectory() as tmp_dir:
        embeddings_path = os.path.join(tmp_dir, "track_embeddings.npz")
        np.savez(
            embeddings_path,
            embeddings=embeddings,
            track_ids=track_ids
        )
        mlflow.log_artifact(embeddings_path, "embeddings")
        logger.info(f"Saved embeddings for {len(track_ids)} tracks")
    
    return embeddings, track_ids

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
            temp_npz = os.path.join(tmp_dir, f"bert_track_embeddings_chunk_{i}.npz")
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
                backup_path = f"bert_track_embeddings_chunk_{i}_backup.npz"
                np.savez(backup_path, **chunk)
                print(f"Saved backup to {backup_path}")
        
        print(f"All chunks processed. Total chunks: {len(chunks)}")

def main():
    # Load track data
    print(f"Loading track data from: {TRACK_JSON_PATH}")
    if not os.path.exists(TRACK_JSON_PATH):
        raise FileNotFoundError(f"Track data file not found at: {TRACK_JSON_PATH}")
        
    with open(TRACK_JSON_PATH, 'r') as f:
        track_data = json.load(f)
    
    # Initialize BERT model and tokenizer
    print("Initializing BERT model...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    model.to(DEVICE)
    model.eval()
    
    # Process tracks in batches
    embeddings = {}
    batch_size = BATCH_SIZE
    track_ids = list(track_data.keys())
    
    print("Starting MLflow run...")
    with mlflow.start_run() as run:
        for i in tqdm(range(0, len(track_ids), batch_size)):
            batch_ids = track_ids[i:i + batch_size]
            batch_texts = [track_data[tid] for tid in batch_ids]
            
            # Tokenize
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            # Store embeddings
            for tid, emb in zip(batch_ids, batch_embeddings):
                embeddings[tid] = emb
        
        print("Processing complete. Starting to save embeddings...")
        # Save embeddings to MinIO
        save_embeddings_to_minio(embeddings, run.info.run_id)
        
        print("Logging parameters...")
        # Log parameters
        mlflow.log_params({
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LEN,
            "num_tracks": len(embeddings),
            "embedding_dim": batch_embeddings.shape[1]
        })
        print("All operations completed successfully!")

if __name__ == "__main__":
    main()
