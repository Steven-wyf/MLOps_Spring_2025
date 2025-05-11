# Standard library imports
import json
import os
import sys
import logging
import tempfile
from datetime import datetime
from typing import Dict, Any, List

# Third-party imports
import boto3
import numpy as np
import torch
import mlflow
from tqdm import tqdm
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import DistilBertTokenizer

# Local imports
sys.path.append('/app')
from data_processing.data_preprocess import process_data

app = FastAPI(title="Music Recommendation")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize MLflow client
mlflow.set_tracking_uri("http://0.0.0.0:8000")
client = mlflow.tracking.MlflowClient()

# Initialize MinIO client
s3_client = boto3.client('s3',
    endpoint_url='http://minio:9000',
    aws_access_key_id='admin',
    aws_secret_access_key='hrwbqzUS85G253yKi43T'
)

# Initialize BERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLPProjector(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)

class LLARAModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_heads=4):
        super().__init__()
        self.self_attention = torch.nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # Self-attention
        attn_output, _ = self.self_attention(x, x, x)
        # FFN
        output = self.ffn(attn_output)
        return output

def get_latest_run_id(experiment_name: str) -> str:
    """Get the latest successful run ID for an experiment"""
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string="status = 'FINISHED'",
        order_by=["start_time DESC"]
    )
    if runs.empty:
        raise ValueError(f"No successful runs found for experiment {experiment_name}")
    return runs.iloc[0].run_id

def load_mlp_model(run_id: str) -> MLPProjector:
    """Load MLP model from MLflow"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="models/mlp_projector.pt",
            dst_path=tmp_dir
        )
        
        checkpoint = torch.load(model_path)
        model = MLPProjector(
            input_dim=checkpoint['input_dim'],
            hidden_dim=checkpoint['hidden_dim']
        )
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        return model

def load_llara_model(run_id: str) -> LLARAModel:
    """Load LLARA model from MLflow"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="models/llara_model.pt",
            dst_path=tmp_dir
        )
        
        checkpoint = torch.load(model_path)
        model = LLARAModel(
            input_dim=checkpoint['input_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            num_heads=checkpoint['num_heads']
        )
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        return model

def process_with_mlp(model: MLPProjector, 
                    embeddings: np.ndarray, 
                    batch_size: int = 1024,
                    device: str = 'cuda') -> np.ndarray:
    """Process embeddings through MLP model"""
    model = model.to(device)
    predictions = []
    
    for i in tqdm(range(0, len(embeddings), batch_size), desc="MLP Processing"):
        batch = embeddings[i:i+batch_size]
        with torch.no_grad():
            batch_tensor = torch.FloatTensor(batch).to(device)
            batch_preds = model(batch_tensor)
            predictions.append(batch_preds.cpu().numpy())
    
    return np.concatenate(predictions)

def process_with_llara(model: LLARAModel,
                      mlp_outputs: np.ndarray,
                      sequence_length: int = 10,
                      batch_size: int = 32,
                      device: str = 'cuda') -> np.ndarray:
    """Process MLP outputs through LLARA model"""
    model = model.to(device)
    final_predictions = []
    
    # Reshape data into sequences
    num_sequences = len(mlp_outputs) // sequence_length
    mlp_outputs = mlp_outputs[:num_sequences * sequence_length]
    sequences = mlp_outputs.reshape(-1, sequence_length, 1)
    
    for i in tqdm(range(0, len(sequences), batch_size), desc="LLARA Processing"):
        batch_sequences = sequences[i:i+batch_size]
        with torch.no_grad():
            batch_tensor = torch.FloatTensor(batch_sequences).to(device)
            batch_preds = model(batch_tensor)
            final_predictions.append(batch_preds.cpu().numpy())
    
    return np.concatenate(final_predictions)

def main():
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:5000")
    
    try:
        # Get latest run IDs
        mf_run_id = get_latest_run_id("matrix-factorization")
        mlp_run_id = get_latest_run_id("mlp-projector")
        llara_run_id = get_latest_run_id("llara-model")
        
        logger.info("Loading MLP model...")
        mlp_model = load_mlp_model(mlp_run_id)
        
        logger.info("Loading LLARA model...")
        llara_model = load_llara_model(llara_run_id)
        
        # Load MF embeddings
        logger.info("Loading MF embeddings...")
        with tempfile.TemporaryDirectory() as tmp_dir:
            emb_path = mlflow.artifacts.download_artifacts(
                run_id=mf_run_id,
                artifact_path="embeddings/item_embeddings.npz",
                dst_path=tmp_dir
            )
            emb_data = np.load(emb_path)
            mf_embeddings = emb_data['item_embeddings']
            track_ids = emb_data['track_ids']
        
        # Process through MLP
        logger.info("Processing through MLP...")
        mlp_outputs = process_with_mlp(mlp_model, mf_embeddings)
        
        # Process through LLARA
        logger.info("Processing through LLARA...")
        final_predictions = process_with_llara(llara_model, mlp_outputs)
        
        # Save results
        logger.info("Saving results...")
        with tempfile.TemporaryDirectory() as tmp_dir:
            result_path = os.path.join(tmp_dir, "final_predictions.npz")
            np.savez(
                result_path,
                predictions=final_predictions,
                track_ids=track_ids
            )
            # Log to MLflow
            with mlflow.start_run() as run:
                mlflow.log_artifact(result_path, "predictions")
                logger.info(f"Results saved to MLflow run: {run.info.run_id}")
                
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()