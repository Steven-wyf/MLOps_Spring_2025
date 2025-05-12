#!/usr/bin/env python3
# inference/export_onnx.py

import os, sys, torch, mlflow
from mlflow.tracking import MlflowClient

# —— 如果你需要走 MinIO，也把 env 放最前 —— 
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://129.114.25.37:9000"
os.environ["AWS_ACCESS_KEY_ID"]      = "admin"
os.environ["AWS_SECRET_ACCESS_KEY"]  = "hrwbqzUS85G253yKi43T"

# 确保能 import train.llara_train
proj_root = os.path.abspath(os.path.join(__file__, "..", ".."))
sys.path.insert(0, proj_root)
from train.llara_train import LlaRAClassifier, get_latest_run_id

# 1) 找到最新 Run
mlflow.set_tracking_uri("http://129.114.25.37:8000/")
run_id = get_latest_run_id("llara-classifier")
print(f"Loading model via MLflow PyTorch flavor from run {run_id}")

# 2) 用 mlflow.pytorch.load_model 直接加载
model_uri = f"runs:/{run_id}/models/llara_model"
model = mlflow.pytorch.load_model(model_uri)  # 返回一个 torch.nn.Module
model.eval()

# 如果你想确认类型：
print("Loaded model:", model)

# 3) 导出 ONNX
#    需要知道 input_dim，取自 model.net[0].in_features
#    下面假设你的 classifier 用 Sequential，第一个是 Linear
first_lin = next(m for m in model.modules() if isinstance(m, torch.nn.Linear))
input_dim = first_lin.in_features

onnx_dir = os.path.abspath("models/music_rec/1")
os.makedirs(onnx_dir, exist_ok=True)
onnx_path = os.path.join(onnx_dir, "model.onnx")

dummy = torch.randn(1, input_dim)
torch.onnx.export(
    model, dummy, onnx_path,
    input_names=["input_embeddings"],
    output_names=["scores"],
    opset_version=13,
    dynamic_axes={"input_embeddings":{0:"batch"}, "scores":{0:"batch"}}
)
print(f"✅ ONNX model saved to {onnx_path}")
