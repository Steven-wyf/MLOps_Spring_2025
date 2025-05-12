#!/usr/bin/env python3
# inference/export_onnx.py

import os
import sys
import torch
import mlflow
from mlflow.tracking import MlflowClient

# —— 如果你的 MLflow Server 配置了 Artifact HTTP 代理，确保先设 URI —— 
mlflow.set_tracking_uri("http://129.114.25.37:8000/")

# —— 为了能 import 你们的训练脚本，插入项目根到 sys.path —— 
proj_root = os.path.abspath(os.path.join(__file__, "..", ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)
from train.llara_train import get_latest_run_id

# 1) 找到最新 Run
run_id = get_latest_run_id("llara-classifier")
print(f"🔍 Loading LLaRA model from MLflow run {run_id}")

# 2) 用 PyTorch Flavor 接口加载模型  
#    model_uri 的格式是 "runs:/<run_id>/<artifact_path_without_suffix>"
#    训练时你是 mlflow.log_artifact(model_path, "models"),
#    那么这里就是 "models/llara_model"
model_uri = f"runs:/{run_id}/models/llara_model"
model = mlflow.pytorch.load_model(model_uri)
model.eval()
print("✅ Model loaded:", model)

# 3) 从模型结构里自动推断 input_dim（取第一个 Linear 层的输入维度）
first_linear = next(m for m in model.modules() if isinstance(m, torch.nn.Linear))
input_dim = first_linear.in_features
print(f"ℹ️  Detected input_dim = {input_dim}")

# 4) 导出成 ONNX
out_dir = os.path.abspath("models/music_rec/1")
os.makedirs(out_dir, exist_ok=True)
onnx_path = os.path.join(out_dir, "model.onnx")

# 构造一个 dummy 输入
dummy_input = torch.randn(1, input_dim)
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
print(f"✅ ONNX model saved to {onnx_path}")
