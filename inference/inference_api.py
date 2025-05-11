from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import json
import os
import mlflow
import boto3
from datetime import datetime
import sys
import torch
from transformers import DistilBertTokenizer
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

def get_latest_model_run():
    """Get the latest model run ID from MLflow"""
    experiments = client.search_experiments()
    latest_run = None
    latest_run_id = None
    
    for exp in experiments:
        runs = client.search_runs(exp.experiment_id)
        for run in runs:
            if latest_run is None or run.info.run_id > latest_run_id:
                latest_run = run
                latest_run_id = run.info.run_id
    
    return latest_run

def load_models(run_id):
    """Load all models from the latest run"""
    models = {}
    # Load BERT model
    models['bert'] = mlflow.pytorch.load_model(f"runs:/{run_id}/models/bert_model")
    # Load Matrix Factorization model
    models['mf'] = mlflow.sklearn.load_model(f"runs:/{run_id}/models/mf_model")
    # Load MLP model
    models['mlp'] = mlflow.pytorch.load_model(f"runs:/{run_id}/models/mlp_projector")
    # Load LLARA model
    models['llara'] = mlflow.pytorch.load_model(f"runs:/{run_id}/models/llara_model")
    return models

@app.get("/")
async def read_root(request: Request):
    """Serve the main page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(request: Request):
    """Handle prediction requests"""
    try:
        # Get JSON data from request
        data = await request.json()
        
        # Generate unique ID for this request
        request_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save input JSON
        input_path = f"/mnt/object/inference/input_{request_id}.json"
        with open(input_path, 'w') as f:
            json.dump(data, f)
        
        # Process data
        processed_files = process_data(input_path)
        
        # Get latest model run
        latest_run = get_latest_model_run()
        if not latest_run:
            raise HTTPException(status_code=500, detail="No models found in MLflow")
        
        # Load models
        models = load_models(latest_run.info.run_id)
        
        # Run inference pipeline
        # 1. BERT encoding
        # Tokenize input text
        inputs = tokenizer(processed_files[0], padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            bert_output = models['bert'](**inputs).last_hidden_state[:, 0, :].cpu().numpy()
        
        # 2. Matrix Factorization
        # Convert BERT output to user-item format
        mf_input = torch.tensor(bert_output, dtype=torch.float32)
        mf_output = models['mf'].predict(mf_input)
        
        # 3. MLP
        # Convert MF output to tensor
        mlp_input = torch.tensor(mf_output, dtype=torch.float32)
        mlp_output = models['mlp'](mlp_input)
        
        # 4. LLARA
        # Convert MLP output to tensor
        llara_input = torch.tensor(mlp_output, dtype=torch.float32)
        final_output = models['llara'](llara_input)
        
        # Convert output to original format
        output_data = {
            "request_id": request_id,
            "predictions": final_output.tolist()
        }
        
        # Save output JSON
        output_path = f"/mnt/object/inference/output_{request_id}.json"
        with open(output_path, 'w') as f:
            json.dump(output_data, f)
        
        return JSONResponse(content=output_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 