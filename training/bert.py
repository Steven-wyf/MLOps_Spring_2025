# bert.py
import os
import mlflow
import torch
import numpy as np
import pandas as pd
import json
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm

# ---- CONFIG SECTION ---- #
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://129.114.25.37:8000/")
MLFLOW_S3_ENDPOINT_URL = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://129.114.25.37:9000")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "admin")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "XCqPacaUHUur82cNZI1R")
EXPERIMENT_NAME = "bert-track-embeddings"

# Data paths
DATA_DIR = os.environ.get("PLAYLIST_DATA_DIR", "/mnt/block/processed")
TRACK_JSON_PATH = os.path.join(DATA_DIR, "track_texts.json")  # JSON file with track information
OUTPUT_NPZ_PATH = "/mnt/block/outputs/bert_track_embeddings.npz"
MODEL_PATH = "/mnt/block/models/bert_encoder_model.pt"

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

def main():
    print(f"Using device: {DEVICE}")
    
    # Load track metadata from JSON
    print(f"Loading track metadata from {TRACK_JSON_PATH}...")
    try:
        with open(TRACK_JSON_PATH, 'r') as f:
            track_data = json.load(f)
        print(f"Loaded metadata for {len(track_data)} tracks")
    except Exception as e:
        print(f"Error loading track metadata: {e}")
        exit(1)
    
    # Prepare track data
    track_uris = []
    track_texts = []
    
    for track_uri, track_info in track_data.items():
        track_uris.append(track_uri)
        # Assuming the JSON has text field, otherwise construct from available fields
        if 'text' in track_info:
            track_texts.append(track_info['text'])
        else:
            # Construct text from available fields (modify based on actual JSON structure)
            track_name = track_info.get('track_name', 'Unknown Track')
            artist_name = track_info.get('artist_name', 'Unknown Artist')
            genre = track_info.get('genre', 'Unknown Genre')
            track_texts.append(f"Track: {track_name} Artist: {artist_name} Genre: {genre}")
    
    # Load model and tokenizer
    print("Loading DistilBERT model and tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(DEVICE)
    
    with mlflow.start_run():
        mlflow.log_params({
            "model": "distilbert-base-uncased",
            "max_length": MAX_LEN,
            "batch_size": BATCH_SIZE,
            "num_tracks": len(track_uris)
        })
        
        # Generate embeddings
        print("Generating track embeddings...")
        embeddings = {}
        
        for i in tqdm(range(0, len(track_uris), BATCH_SIZE)):
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
                for j, track_uri in enumerate(track_uris):
                    embeddings[track_uri] = cls_embeddings[j]
        
        # Save embeddings to NPZ file
        print(f"Saving {len(embeddings)} track embeddings to {OUTPUT_NPZ_PATH}")
        np.savez(OUTPUT_NPZ_PATH, **embeddings)
        
        # Log artifacts
        mlflow.log_artifact(OUTPUT_NPZ_PATH)
        
        # Save tokenizer and model
        torch.save({
            'model_state_dict': model.state_dict(),
            'tokenizer': tokenizer,
        }, MODEL_PATH)
        mlflow.log_artifact(MODEL_PATH)
        
        print("BERT track embeddings generation complete.")

if __name__ == "__main__":
    main()
