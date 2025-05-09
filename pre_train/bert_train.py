#!/usr/bin/env python
# bert_train.py - Text embedding generator
import os
import mlflow
import torch
import numpy as np
import pandas as pd
import json
import logging
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm
import time
from datetime import datetime
import gc
import psutil

# Configure logging
LOG_DIR = os.environ.get("LOG_DIR", "./logs")
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{LOG_DIR}/bert_train_{datetime.now().strftime('%Y%m%d%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---- CONFIG SECTION ---- #
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:8000")
MLFLOW_S3_ENDPOINT_URL = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin")
EXPERIMENT_NAME = "bert-track-embeddings"
RUN_ID = os.environ.get("RUN_ID", datetime.now().strftime('%Y%m%d%H%M%S'))

# Data paths
DATA_DIR = os.environ.get("PLAYLIST_DATA_DIR", "/mnt/block")
TRACK_JSON_PATH = os.path.join(DATA_DIR, "track_text.json")  # JSON file with track information
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_NPZ_PATH = os.path.join(OUTPUT_DIR, "bert_track_embeddings.npz")
MODEL_PATH = os.path.join(OUTPUT_DIR, "bert_encoder_model.pt")

# Model parameters
MAX_LEN = 128
# Smaller batch size to reduce memory usage
BATCH_SIZE = int(os.environ.get("BERT_BATCH_SIZE", "16"))
# Process data in chunks to avoid loading everything into memory
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
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

def process_in_chunks(track_data):
    """Process track data in chunks to minimize memory usage"""
    track_uris = list(track_data.keys())
    total_tracks = len(track_uris)
    chunk_indices = list(range(0, total_tracks, CHUNK_SIZE))
    
    all_embeddings = {}
    
    for i, start_idx in enumerate(chunk_indices):
        end_idx = min(start_idx + CHUNK_SIZE, total_tracks)
        chunk_uris = track_uris[start_idx:end_idx]
        
        logger.info(f"Processing chunk {i+1}/{len(chunk_indices)} ({start_idx}:{end_idx}, {len(chunk_uris)} tracks)")
        
        # Process chunk
        chunk_embeddings = process_track_chunk(chunk_uris, track_data)
        
        # Add to overall embeddings
        all_embeddings.update(chunk_embeddings)
        
        # Save intermediate results periodically
        if (i + 1) % 5 == 0 or (i + 1) == len(chunk_indices):
            logger.info(f"Saving intermediate embeddings ({len(all_embeddings)} tracks so far)")
            tmp_path = os.path.join(OUTPUT_DIR, f"bert_embeddings_temp_{i+1}.npz")
            np.savez(tmp_path, **all_embeddings)
        
        # Check memory usage and clean up if necessary
        check_memory_usage()
        
    return all_embeddings

def process_track_chunk(chunk_uris, track_data):
    """Process a single chunk of track data to generate embeddings"""
    chunk_texts = []
    
    for track_uri in chunk_uris:
        track_info = track_data[track_uri]
        # Assuming the JSON has text field, otherwise construct from available fields
        if 'text' in track_info:
            chunk_texts.append(track_info['text'])
        else:
            # Construct text from available fields (modify based on actual JSON structure)
            track_name = track_info.get('track_name', 'Unknown Track')
            artist_name = track_info.get('artist_name', 'Unknown Artist')
            genre = track_info.get('genre', 'Unknown Genre')
            chunk_texts.append(f"Track: {track_name} Artist: {artist_name} Genre: {genre}")
    
    # Generate embeddings for chunk
    chunk_embeddings = generate_embeddings(chunk_uris, chunk_texts)
    
    return chunk_embeddings

def generate_embeddings(track_uris, track_texts):
    """Generate embeddings for a batch of track texts"""
    # Load model and tokenizer (could optimize to only load once)
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(DEVICE)
    
    embeddings = {}
    
    # Process in batches
    for i in tqdm(range(0, len(track_uris), BATCH_SIZE), desc="Generating embeddings"):
        batch_texts = track_texts[i:i+BATCH_SIZE]
        batch_uris = track_uris[i:i+BATCH_SIZE]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        ).to(DEVICE)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            
            # Use CLS token embeddings (first token of last layer)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            # Store embeddings
            for j, uri in enumerate(batch_uris):
                if j < len(cls_embeddings):  # Ensure index is valid
                    embeddings[uri] = cls_embeddings[j]
        
        # Clean up to free memory
        del inputs, outputs, cls_embeddings
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Clear model and tokenizer from memory
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return embeddings

def main():
    """Main BERT training and embedding generation function"""
    start_time = time.time()
    logger.info(f"Starting BERT training - Run ID: {RUN_ID}")
    logger.info(f"Using device: {DEVICE}")
    logger.info(f"Memory usage before starting: {check_memory_usage():.1f}%")
    logger.info(f"Batch size: {BATCH_SIZE}, Chunk size: {CHUNK_SIZE}")
    
    # Configure MLflow
    configure_mlflow()
    
    # Load track metadata from JSON
    logger.info(f"Loading track metadata from {TRACK_JSON_PATH}...")
    try:
        with open(TRACK_JSON_PATH, 'r') as f:
            track_data = json.load(f)
        logger.info(f"Loaded metadata for {len(track_data)} tracks")
    except Exception as e:
        logger.error(f"Error loading track metadata: {str(e)}")
        raise
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"bert-embeddings-{RUN_ID}") as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run ID: {run_id}")
        
        # Log parameters
        params = {
            "model": "distilbert-base-uncased",
            "max_length": MAX_LEN,
            "batch_size": BATCH_SIZE,
            "chunk_size": CHUNK_SIZE,
            "memory_limit_percent": MEMORY_LIMIT_PERCENT,
            "num_tracks": len(track_data),
            "embedding_dim": 768,  # DistilBERT dim
            "device": str(DEVICE)
        }
        mlflow.log_params(params)
        
        # Generate embeddings chunk by chunk
        logger.info("Generating track embeddings...")
        embeddings = process_in_chunks(track_data)
        
        # Save embeddings to NPZ file
        logger.info(f"Saving {len(embeddings)} track embeddings to {OUTPUT_NPZ_PATH}")
        np.savez(OUTPUT_NPZ_PATH, **embeddings)
        
        # Log artifacts
        mlflow.log_artifact(OUTPUT_NPZ_PATH, "embeddings")
        
        # Save tokenizer and model
        # Load again as we've deleted them to save memory
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        torch.save({
            'model_state_dict': model.state_dict(),
            'tokenizer_config': tokenizer.init_kwargs,
        }, MODEL_PATH)
        
        mlflow.log_artifact(MODEL_PATH, "model")
        
        # Log metric for embedding coverage
        mlflow.log_metric("embedding_coverage_percent", 
                          100.0 * len(embeddings) / len(track_data))
        
        # Log completion time
        end_time = time.time()
        duration = end_time - start_time
        mlflow.log_metric("duration_seconds", duration)
        
        logger.info(f"BERT embeddings generation completed in {duration:.2f} seconds")
        logger.info(f"Generated embeddings for {len(embeddings)} tracks")
        logger.info(f"MLflow run ID: {run_id}")
        logger.info(f"Final memory usage: {check_memory_usage():.1f}%")
        
        # Write model info to a file for next steps
        model_info = {
            "bert_run_id": run_id,
            "embedding_file": OUTPUT_NPZ_PATH,
            "model_file": MODEL_PATH,
            "embedding_dim": 768,
            "num_tracks": len(embeddings)
        }
        
        with open(os.path.join(OUTPUT_DIR, "bert_model_info.json"), "w") as f:
            json.dump(model_info, f, indent=2)
        
        # Clean temporary files
        for temp_file in os.listdir(OUTPUT_DIR):
            if temp_file.startswith("bert_embeddings_temp_") and temp_file.endswith(".npz"):
                try:
                    os.remove(os.path.join(OUTPUT_DIR, temp_file))
                    logger.info(f"Removed temporary file: {temp_file}")
                except Exception as e:
                    logger.warning(f"Could not remove temp file {temp_file}: {str(e)}")

if __name__ == "__main__":
    try:
        main()
        exit(0)
    except Exception as e:
        logger.exception(f"BERT training failed: {str(e)}")
        exit(1) 