#!/usr/bin/env python
# inference/export_onnx.py

import os

# 指定 MLflow artifact store（MinIO）的 endpoint 和凭证
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://129.114.25.37:9000"
os.environ["AWS_ACCESS_KEY_ID"]      = "admin"
os.environ["AWS_SECRET_ACCESS_KEY"]  = "hrwbqzUS85G253yKi43T"
import mlflow
import torch
import tempfile
import sys

proj_root = os.path.abspath(os.path.join(__file__, "..", ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from train.llara_train import LlaRAClassifier, get_latest_run_id
# 1) Configure MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://129.114.25.37:8000/")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# 2) Find the latest run for the LLaRA experiment
run_id = get_latest_run_id("llara-classifier")
print(f"Exporting ONNX from MLflow run {run_id}")

with tempfile.TemporaryDirectory() as tmp:
    # 只指定到 models 目录
    local_models_dir = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="models",
        dst_path=tmp
    )
    # local_models_dir = /tmp/tmpXXXX/models
    print("Downloaded to:", local_models_dir)
    print("Contents:", os.listdir(local_models_dir))
    
    # 构造 .pt 文件的完整路径
    pt_path = os.path.join(local_models_dir, "llara_model.pt")
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"{pt_path} not found after download")

    # 加载 checkpoint
    ckpt = torch.load(pt_path, map_location="cpu")
    input_dim   = ckpt["input_dim"]
    num_classes = ckpt["num_classes"]
    hidden_dim  = ckpt.get("hidden_dim", 512)
    dropout     = ckpt.get("dropout", 0.3)
    state_dict  = ckpt["model_state"]

    # 重建模型并 load
    model = LlaRAClassifier(input_dim, num_classes, hidden_dim, dropout)
    model.load_state_dict(state_dict)
    model.eval()

    # 导出 ONNX
    onnx_dir = os.path.abspath("models/music_rec/1")
    os.makedirs(onnx_dir, exist_ok=True)
    onnx_path = os.path.join(onnx_dir, "model.onnx")

    dummy = torch.randn(1, input_dim)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input_embeddings"],
        output_names=["scores"],
        opset_version=13,
        dynamic_axes={"input_embeddings": {0: "batch"}, "scores": {0: "batch"}}
    )
    print(f"✅ ONNX model saved to {onnx_path}")
