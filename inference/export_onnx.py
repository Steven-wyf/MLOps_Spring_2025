#!/usr/bin/env python
# inference/export_onnx.py

import os
import mlflow
import torch
import tempfile
from llara_train import LlaRAClassifier, get_latest_run_id

# 1) Configure MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://129.114.25.37:8000/")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# 2) Find the latest run for the LLaRA experiment
run_id = get_latest_run_id("llara-classifier")
print(f"Exporting ONNX from MLflow run {run_id}")

# 3) Download exactly the llara_model.pt artifact
with tempfile.TemporaryDirectory() as tmp:
    pt_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="models/llara_model.pt",
        dst_path=tmp
    )

    # 4) Load checkpoint
    ckpt = torch.load(pt_path, map_location="cpu")
    input_dim   = ckpt["input_dim"]
    num_classes = ckpt["num_classes"]
    hidden_dim  = ckpt.get("hidden_dim", 512)
    dropout     = ckpt.get("dropout", 0.3)
    state_dict  = ckpt["model_state"]

    # 5) Reconstruct model and load weights
    model = LlaRAClassifier(input_dim, num_classes, hidden_dim, dropout)
    model.load_state_dict(state_dict)
    model.eval()

    # 6) Export to ONNX
    onnx_dir = os.path.abspath("models/music_rec/1")
    os.makedirs(onnx_dir, exist_ok=True)
    onnx_path = os.path.join(onnx_dir, "model.onnx")

    dummy = torch.randn(1, input_dim)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input_embeddings"],
        output_names=["scores"],
        opset_version=13,
        dynamic_axes={
            "input_embeddings": {0: "batch"},
            "scores":           {0: "batch"}
        }
    )

    print(f"ONNX model successfully saved to {onnx_path}")
