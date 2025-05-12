#!/usr/bin/env python3
# inference/export_onnx_local.py

import os, sys, torch

# 确保能 import 到你的模型定义
proj_root = os.path.abspath(os.path.join(__file__, "..", ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)
from train.llara_train import LlaRAClassifier

pt_path = "models/music_rec/1/llara_model.pt"
if not os.path.exists(pt_path):
    raise FileNotFoundError(pt_path)

# 1) load checkpoint
ckpt = torch.load(pt_path, map_location="cpu")
in_dim   = ckpt["input_dim"]
num_cls  = ckpt["num_classes"]
hid_dim  = ckpt.get("hidden_dim", 512)
drop     = ckpt.get("dropout", 0.3)
state    = ckpt["model_state"]

# 2) rebuild model
model = LlaRAClassifier(in_dim, num_cls, hid_dim, drop)
model.load_state_dict(state)
model.eval()

# 3) export ONNX
out_dir = os.path.dirname(pt_path)
onnx_path = os.path.join(out_dir, "model.onnx")
dummy = torch.randn(1, in_dim)
torch.onnx.export(
    model, dummy, onnx_path,
    input_names=["input_embeddings"],
    output_names=["scores"],
    opset_version=13,
    dynamic_axes={"input_embeddings":{0:"batch"}, "scores":{0:"batch"}},
)
print(f"✅ ONNX saved to {onnx_path}")
