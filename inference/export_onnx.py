#!/usr/bin/env python3
# inference/export_onnx.py

import os, sys, torch, mlflow

# —— 让脚本能 import 到 train/llara_train.py —— 
proj_root = os.path.abspath(os.path.join(__file__, "..", ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from train.llara_train import LlaRAClassifier, get_latest_run_id

# 1) 先设置 Tracking URI （只针对 HTTP 下载）
mlflow.set_tracking_uri("http://129.114.25.37:8000/")

# 2) 找到最新的 run
run_id = get_latest_run_id("llara-classifier")
print(f"🔍 Loading model from MLflow run {run_id}")

# 3) 用 MLflow 的 PyTorch flavor 接口直接加载模型
#    这里的 "models/llara_model" 对应你 training 脚本里 mlflow.log_artifact(model_path, "models")
model_uri = f"runs:/{run_id}/models/llara_model"
model = mlflow.pytorch.load_model(model_uri)  # 返回 torch.nn.Module
model.eval()

print("✅ Model loaded:", model)

# 4) 导出 ONNX
#    自动从模型结构里找到输入维度
#    假定第一个 Linear 层是 input_dim
first_lin = next(m for m in model.modules() if isinstance(m, torch.nn.Linear))
input_dim = first_lin.in_features

out_dir = os.path.abspath("models/music_rec/1")
os.makedirs(out_dir, exist_ok=True)
onnx_path = os.path.join(out_dir, "model.onnx")

dummy = torch.randn(1, input_dim)
torch.onnx.export(
    model, dummy, onnx_path,
    input_names=["input_embeddings"],
    output_names=["scores"],
    opset_version=13,
    dynamic_axes={
        "input_embeddings": {0: "batch"},
        "scores":           {0: "batch"},
    }
)
print(f"✅ ONNX model saved to {onnx_path}")
