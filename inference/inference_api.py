from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import os
import json
import mlflow
import boto3
from datetime import datetime
import sys
import torch
import numpy as np 
from transformers import DistilBertTokenizer

sys.path.append('/app')
from data_processing.data_preprocess import process_data

from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput

app = FastAPI(title="Music Recommendation")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Flag to switch Triton on/off via environment variable
USE_TRITON = os.getenv("USE_TRITON", "true").lower() == "true"

# Initialize Triton client if enabled
if USE_TRITON:
    TRITON_URL = os.getenv("TRITON_SERVER_URL", "http://triton:8000")
    triton_client = InferenceServerClient(url=TRITON_URL)

# Initialize MLflow client
mlflow.set_tracking_uri("http://0.0.0.0:8000")
client = mlflow.tracking.MlflowClient()

# Initialize MinIO client
s3_client = boto3.client(
    's3',
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
    models['bert']  = mlflow.pytorch.load_model(f"runs:/{run_id}/models/bert_model")
    models['mf']    = mlflow.sklearn.load_model(f"runs:/{run_id}/models/mf_model")
    models['mlp']   = mlflow.pytorch.load_model(f"runs:/{run_id}/models/mlp_projector")
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
        # 1. Receive and save raw input JSON
        data = await request.json()
        request_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_path = f"/mnt/object/inference/input_{request_id}.json"
        with open(input_path, 'w') as f:
            json.dump(data, f)

        # 2. Preprocess input data
        processed_dir = "/mnt/object/processed_data"
        process_data(input_path, processed_dir, quick=False)

        # 3. Fetch latest MLflow run and load models
        latest_run = get_latest_model_run()
        if not latest_run:
            raise HTTPException(status_code=500, detail="No models found in MLflow")
        models = load_models(latest_run.info.run_id)

        # 4. Run original inference pipeline

        # 4.1 Load processed track metadata
        json_path = os.path.join(processed_dir, "track_texts.json") 
        with open(json_path, "r", encoding="utf-8") as f:
            track_metadata = json.load(f)
        print(f"Loaded {len(track_metadata)} tracks from {json_path}")

        # 4.2 BERT encoding
        texts = [item["text"] for item in track_metadata]
        inputs = tokenizer(texts, padding=True, truncation=True,
                           max_length=128, return_tensors="pt")
        with torch.no_grad():
            bert_output = models['bert'](**inputs).last_hidden_state[:, 0, :].cpu().numpy()
        print(f"Encoded {len(texts)} tracks with BERT")

        # 4.3 Matrix Factorization
        print("starting matrix factorization...")
        mf_input = torch.tensor(bert_output, dtype=torch.float32)
        mf_output = models['mf'].predict(mf_input)
        print(f"Finished MF on {mf_input.shape[0]} tracks")

        # 4.4 MLP
        print("starting mlp...")
        mlp_input = torch.tensor(mf_output, dtype=torch.float32)
        mlp_output = models['mlp'](mlp_input)
        print(f"Finished MLP on {mlp_input.shape[0]} tracks")

        # 4.5 LLARA
        print("starting llara...")
        llara_input  = torch.tensor(mlp_output, dtype=torch.float32)
        final_output = models['llara'](llara_input)
        print(f"Finished LLARA on {llara_input.shape[0]} tracks")

        # 5. Branch: use Triton if enabled
        if USE_TRITON:
            # convert to numpy
            output_np = final_output.cpu().numpy() 
            # prepare Triton input
            inp = InferInput("input_embeddings", output_np.shape, "FP32")
            inp.set_data_from_numpy(output_np.astype(np.float32))
            # specify Triton output
            out = InferRequestedOutput("scores", binary_data=False)
            # perform Triton inference
            triton_res = triton_client.infer(
                model_name="music_rec",
                inputs=[inp],
                outputs=[out]
            )
            # extract Triton predictions
            predictions = triton_res.as_numpy("scores")[0].tolist()
        else:
            # bypass Triton, return raw output
            predictions = final_output.cpu().tolist() 

        # 6. Save and return output JSON
        output_data = {
            "request_id": request_id,
            "predictions": predictions
        }
        output_path = f"/mnt/object/inference/output_{request_id}.json"
        with open(output_path, 'w') as f:
            json.dump(output_data, f)
        print(f"Saved output to {output_path}")

        return JSONResponse(content=output_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
