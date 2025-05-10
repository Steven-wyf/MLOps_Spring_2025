#!/usr/bin/env python
# bert_encoding.py - BERT track text encoding
import os
import mlflow
import torch
import numpy as np
import json
import gc
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

# ---- CONFIG SECTION ---- #
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://129.114.25.37:8000/")
EXPERIMENT_NAME = "bert-track-encoding"

# Data paths
DATA_DIR = os.environ.get("PLAYLIST_DATA_DIR", "/mnt/object")
TRACK_TEXT_PATH = os.path.join(DATA_DIR, "processed_data/track_texts.json")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/mnt/object/outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_NPZ_PATH = os.path.join(OUTPUT_DIR, "bert_track_embeddings.npz")

# Model parameters
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ------------------------ #

def cleanup_memory():
    """Clean up both RAM and GPU memory"""
    # Clear RAM
    gc.collect()
    
    # Clear GPU memory if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def main():
    """Main function for BERT track encoding"""
    print(f"Using device: {DEVICE}")
    
    # Configure MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Load track text data
    print(f"Loading track text data from {TRACK_TEXT_PATH}")
    with open(TRACK_TEXT_PATH, "r") as f:
        track_map = json.load(f)
    print(f"Loaded {len(track_map)} tracks")
    
    track_ids = list(track_map.keys())
    descriptions = list(track_map.values())
    
    # Initialize BERT model and tokenizer
    print("Initializing BERT model and tokenizer")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(DEVICE)
    model.eval()
    
    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            "model": "bert-base-uncased",
            "batch_size": BATCH_SIZE,
            "num_tracks": len(track_ids),
            "device": str(DEVICE)
        })
        
        # Generate embeddings
        print("Generating BERT embeddings")
        all_embeddings = []
        all_track_ids = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(descriptions), BATCH_SIZE), desc="Encoding BERT embeddings"):
                batch = descriptions[i:i+BATCH_SIZE]
                batch_ids = track_ids[i:i+BATCH_SIZE]
                
                encodings = tokenizer(batch, truncation=True, padding='longest', return_tensors='pt')
                encodings = {k: v.to(device) for k, v in encodings.items()}
                
                outputs = model(**encodings)
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu()
                
                all_embeddings.extend(cls_embeddings)
                all_track_ids.extend(batch_ids)
                
                # Clean up memory after each batch
                del encodings, outputs, cls_embeddings
                cleanup_memory()
        
        # Create embedding dictionary
        embedding_dict = {tid: emb.numpy() for tid, emb in zip(all_track_ids, all_embeddings)}
        
        # Save embeddings to NPZ file
        print(f"Saving {len(embedding_dict)} track embeddings to {OUTPUT_NPZ_PATH}")
        np.savez(OUTPUT_NPZ_PATH, **embedding_dict)
        
        # Log artifact
        mlflow.log_artifact(OUTPUT_NPZ_PATH)
        
        # Final cleanup
        del all_embeddings, all_track_ids, embedding_dict
        cleanup_memory()
        
        print("BERT encoding completed")

if __name__ == "__main__":
    main()
