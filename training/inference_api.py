import os
import numpy as np
import torch
import json
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables and constants
MODEL_DIR = os.environ.get("MODEL_DIR", "./outputs")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
TOP_K = int(os.environ.get("TOP_K", "10"))

# Initialize FastAPI
app = FastAPI(
    title="Playlist Continuation API",
    description="API for recommending tracks to continue a playlist",
    version="1.0.0"
)

# Data models
class Playlist(BaseModel):
    playlist_id: Optional[str] = None
    tracks: List[str]
    num_recommendations: int = 5

class RecommendationResponse(BaseModel):
    recommended_tracks: List[str]
    model_version: str

# Global variables to store models and data
bert_embeddings = None
lightgcn_embeddings = None
projected_embeddings = None
model_llara = None
track_id_to_idx = None
idx_to_track_id = None
is_loaded = False
model_version = "1.0.0"

# Model loading function
@app.on_event("startup")
async def load_models():
    global bert_embeddings, lightgcn_embeddings, projected_embeddings, model_llara
    global track_id_to_idx, idx_to_track_id, is_loaded
    
    try:
        logger.info(f"Loading models from {MODEL_DIR}")
        
        # 1. Load BERT embeddings
        bert_path = os.path.join(MODEL_DIR, "bert_track_embeddings.npz")
        if os.path.exists(bert_path):
            bert_embeddings = np.load(bert_path)
            logger.info(f"Loaded BERT embeddings for {len(bert_embeddings.files)} tracks")
        else:
            logger.warning(f"BERT embeddings not found at {bert_path}")
        
        # 2. Load LightGCN embeddings
        lightgcn_path = os.path.join(MODEL_DIR, "lightgcn_embeddings.npz")
        if os.path.exists(lightgcn_path):
            lightgcn_embeddings = np.load(lightgcn_path)
            logger.info(f"Loaded LightGCN embeddings for {len(lightgcn_embeddings.files)} tracks")
        else:
            logger.warning(f"LightGCN embeddings not found at {lightgcn_path}")
        
        # 3. Load projected embeddings
        projected_path = os.path.join(MODEL_DIR, "projected_lightgcn.npz")
        if os.path.exists(projected_path):
            projected_embeddings = np.load(projected_path)
            logger.info(f"Loaded projected embeddings for {len(projected_embeddings.files)} tracks")
            
            # Create track ID mappings
            all_track_ids = list(projected_embeddings.files)
            track_id_to_idx = {tid: i for i, tid in enumerate(all_track_ids)}
            idx_to_track_id = {i: tid for tid, i in track_id_to_idx.items()}
        else:
            logger.warning(f"Projected embeddings not found at {projected_path}")
        
        # 4. Load LlaRA model
        llara_path = os.path.join(MODEL_DIR, "llara_classifier.pt")
        if os.path.exists(llara_path) and projected_embeddings is not None:
            embed_dim = next(iter(projected_embeddings.values())).shape[0]
            num_classes = len(projected_embeddings.files)
            
            # Define LlaRA model class (same as in training)
            class Classifier(torch.nn.Module):
                def __init__(self, embed_dim, num_classes):
                    super().__init__()
                    self.fc = torch.nn.Linear(embed_dim, num_classes)
                
                def forward(self, x):
                    return self.fc(x)
            
            model_llara = Classifier(embed_dim, num_classes).to(DEVICE)
            model_llara.load_state_dict(torch.load(llara_path, map_location=DEVICE))
            model_llara.eval()
            logger.info(f"Loaded LlaRA model from {llara_path}")
        else:
            logger.warning(f"LlaRA model not found at {llara_path} or projected embeddings not available")
        
        is_loaded = True
        logger.info("All available models loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

# Helper functions
def get_track_embedding(track_id):
    if track_id in projected_embeddings:
        return projected_embeddings[track_id]
    return None

def get_recommendations(track_ids, num_recommendations=5):
    if not is_loaded or model_llara is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    
    if not track_ids:
        raise HTTPException(status_code=400, detail="No tracks provided")
    
    # Get the last track in the playlist
    last_track = track_ids[-1]
    
    if last_track not in track_id_to_idx:
        raise HTTPException(status_code=404, detail=f"Track {last_track} not found in the database")
    
    # Get the embedding for the last track
    track_embedding = get_track_embedding(last_track)
    if track_embedding is None:
        raise HTTPException(status_code=404, detail=f"Embedding for track {last_track} not found")
    
    # Convert to tensor
    track_tensor = torch.tensor(track_embedding, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    # Get recommendations
    with torch.no_grad():
        logits = model_llara(track_tensor)
        probs = torch.softmax(logits, dim=1)
        
        # Get top K track indices
        top_tracks = torch.topk(probs[0], k=num_recommendations + len(track_ids)).indices.cpu().numpy()
        
        # Filter out tracks that are already in the playlist
        existing_track_idxs = [track_id_to_idx[tid] for tid in track_ids if tid in track_id_to_idx]
        recommendations = [idx_to_track_id[idx.item()] for idx in top_tracks 
                         if idx.item() not in existing_track_idxs][:num_recommendations]
        
    return recommendations

# API endpoints
@app.get("/")
async def root():
    return {"message": "Playlist Continuation API", "status": "active", "models_loaded": is_loaded}

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_tracks(playlist: Playlist, background_tasks: BackgroundTasks):
    try:
        recommendations = get_recommendations(
            playlist.tracks, 
            num_recommendations=playlist.num_recommendations
        )
        
        return {
            "recommended_tracks": recommendations,
            "model_version": model_version
        }
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": is_loaded,
        "device": str(DEVICE),
        "available_models": {
            "bert": bert_embeddings is not None,
            "lightgcn": lightgcn_embeddings is not None,
            "projected": projected_embeddings is not None,
            "llara": model_llara is not None
        }
    }

if __name__ == "__main__":
    # When running directly, use this port
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("inference_api:app", host="0.0.0.0", port=port, reload=False) 