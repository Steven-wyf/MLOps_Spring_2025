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

# ---- CONFIG SECTION ---- #
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://129.114.25.37:8000/")
MLFLOW_S3_ENDPOINT_URL = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://129.114.25.37:9000")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "your-acccess-key")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "hrwbqzUS85G253yKi43T")
EXPERIMENT_NAME = "bert-track-embeddings"

# Data paths
DATA_DIR = os.environ.get("PLAYLIST_DATA_DIR", "~/processed_data")
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

def save_embeddings_to_minio(embeddings_dict, run_id):
    """Save embeddings to MinIO using MLflow"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save embeddings to temporary file
        temp_npz = os.path.join(tmp_dir, "bert_track_embeddings.npz")
        np.savez(temp_npz, **embeddings_dict)
        
        # Log to MLflow (which will save to MinIO)
        mlflow.log_artifact(temp_npz, "embeddings")
        print(f"Embeddings saved to MinIO through MLflow run {run_id}")

def main():
    # Load track data
    with open(TRACK_JSON_PATH, 'r') as f:
        track_data = json.load(f)
    
    # Initialize BERT model and tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    model.to(DEVICE)
    model.eval()
    
    # Process tracks in batches
    embeddings = {}
    batch_size = BATCH_SIZE
    track_ids = list(track_data.keys())
    
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
        
        # Save embeddings to MinIO
        save_embeddings_to_minio(embeddings, run.info.run_id)
        
        # Log parameters
        mlflow.log_params({
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LEN,
            "num_tracks": len(embeddings),
            "embedding_dim": batch_embeddings.shape[1]
        })

if __name__ == "__main__":
    main()
