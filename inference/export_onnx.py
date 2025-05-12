#!/usr/bin/env python3
# inference/export_onnx_direct.py

import os
import sys
import boto3
import torch
import tempfile

# —— 让脚本能够 import train/llara_train.py —— 
proj_root = os.path.abspath(os.path.join(__file__, "..", ".."))
sys.path.insert(0, proj_root)
from train.llara_train import LlaRAClassifier  # 只需模型类

# 1) MinIO S3 配置（和训练时一致）
MINIO_ENDPOINT = "http://129.114.25.37:9000"
AWS_ACCESS_KEY_ID = "admin"
AWS_SECRET_ACCESS_KEY = "hrwbqzUS85G253yKi43T"
BUCKET = "mlflow-artifacts"

# 2) 运行 ID & 存储路径
RUN_ID = "a497978bf50a43fe89e780f9e0591479"  # 你的 Run ID
# 从你给的路径看，模型对象 key 是：
KEY = f"4/{RUN_ID}/artifacts/models/llara_model.pt"

# 3) 在临时目录下载 .pt
with tempfile.TemporaryDirectory() as tmp:
    local_pt = os.path.join(tmp, "llara_model.pt")

    s3 = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
    print(f"⬇️  Downloading s3://{BUCKET}/{KEY} → {local_pt}")
    s3.download_file(BUCKET, KEY, local_pt)
    print("✅ Download complete")

    # 4) Load checkpoint
    ckpt = torch.load(local_pt, map_location="cpu")
    in_dim   = ckpt["input_dim"]
    num_cls  = ckpt["num_classes"]
    hid_dim  = ckpt.get("hidden_dim", 512)
    drop     = ckpt.get("dropout", 0.3)
    state    = ckpt["model_state"]

    # 5) Rebuild & load model
    model = LlaRAClassifier(in_dim, num_cls, hid_dim, drop)
    model.load_state_dict(state)
    model.eval()

    # 6) Export ONNX
    out_dir = os.path.abspath("models/music_rec/1")
    os.makedirs(out_dir, exist_ok=True)
    onnx_path = os.path.join(out_dir, "model.onnx")

    dummy = torch.randn(1, in_dim)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input_embeddings"],
        output_names=["scores"],
        opset_version=13,
        dynamic_axes={"input_embeddings": {0: "batch"}, "scores": {0: "batch"}},
    )
    print(f"✅ ONNX model saved to {onnx_path}")
