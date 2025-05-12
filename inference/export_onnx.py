#!/usr/bin/env python3
# inference/export_onnx.py

import os
import sys
import torch
import mlflow

# 1) Ensure we can import your model definition from train/llara_train.py
proj_root = os.path.abspath(os.path.join(__file__, "..", ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)
from train.llara_train import get_latest_run_id, LlaRAClassifier

# 2) Configure MLflow to point at your remote Tracking Server and MinIO
os.environ["MLFLOW_TRACKING_URI"]    = "http://129.114.25.37:8000"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://129.114.25.37:9000"
os.environ["AWS_ACCESS_KEY_ID"]      = "admin"
os.environ["AWS_SECRET_ACCESS_KEY"]  = "hrwbqzUS85G253yKi43T"

# 3) Retrieve the latest successful run of the llara-classifier experiment
run_id = get_latest_run_id("llara-classifier")
print(f"Exporting ONNX from MLflow run {run_id}")

# 4) Load the logged PyTorch model via MLflow’s PyTorch flavor
#    Note: `models/llara_model` matches the artifact path used in mlflow.log_artifact(...)
model_uri = f"runs:/{run_id}/models/llara_model"
model = mlflow.pytorch.load_model(model_uri)
model.eval()
print("Model loaded from MLflow")

# 5) Determine the input dimension by inspecting the first Linear layer
#    (assumes your LlaRAClassifier has at least one nn.Linear module)
import torch.nn as nn
input_dim = None
for module in model.modules():
    if isinstance(module, nn.Linear):
        input_dim = module.in_features
        break
if input_dim is None:
    raise RuntimeError("Could not infer input_dim from model architecture")
print(f"Inferred input_dim = {input_dim}")

# 6) Export the model to ONNX format
output_dir = "models/music_rec/1"
os.makedirs(output_dir, exist_ok=True)
onnx_path = os.path.join(output_dir, "model.onnx")

# Create a dummy tensor with the correct input shape
dummy_input = torch.randn(1, input_dim)

# Perform the export
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["input_embeddings"],
    output_names=["scores"],
    opset_version=13,
    dynamic_axes={
        "input_embeddings": {0: "batch"},
        "scores":           {0: "batch"},
    },
)
print(f"ONNX model saved to {onnx_path}")

