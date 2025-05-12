import os
import time
import numpy as np
import torch
import onnxruntime as ort
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from onnxruntime.quantization import (
    quantize_dynamic,
    quantize_static,
    CalibrationDataReader,
    QuantType
)

# 1) Load test data from the numpy files
X = np.load("../outputs/llara/X_test.npy")
y = np.load("../outputs/llara/y_test.npy")
test_loader = DataLoader(
    TensorDataset(torch.from_numpy(X).float(),
                  torch.from_numpy(y).long()),
    batch_size=128,
    shuffle=False
)

# 2) Reload the fine-tuned LLaRA+LoRA model
ckpt = torch.load("../outputs/models/llara_model.pt", map_location="cpu")
state_dict = ckpt["model_state"]
input_dim   = ckpt["input_dim"]
num_classes = ckpt["num_classes"]

class LlaRAClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x):
        return self.net(x)

model = LlaRAClassifier(input_dim, num_classes)
model.load_state_dict(state_dict)
model.eval()

# 3) Benchmark PyTorch model on CPU
def benchmark_pt(model, loader, model_path):
    # Report model file size
    print(f"[PyTorch] Model size: {os.path.getsize(model_path)/1e6:.2f} MB")

    # Compute test accuracy
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    print(f"[PyTorch] Accuracy: {correct/total*100:.2f}%")

    # Measure single-sample latency
    single = next(iter(loader))[0][:1]
    with torch.no_grad():
        model(single)  # warm-up
    latencies = []
    for _ in range(100):
        t0 = time.time()
        with torch.no_grad():
            model(single)
        latencies.append((time.time() - t0) * 1000)
    print(f"[PyTorch] Latency (ms) median/95th: "
          f"{np.percentile(latencies,50):.2f}/{np.percentile(latencies,95):.2f}")

    # Measure batch throughput
    batch = next(iter(loader))[0]
    with torch.no_grad():
        model(batch)  # warm-up
    durations = []
    for _ in range(50):
        t0 = time.time()
        with torch.no_grad():
            model(batch)
        durations.append(time.time() - t0)
    fps = batch.size(0) * 50 / sum(durations)
    print(f"[PyTorch] Throughput (FPS): {fps:.1f}")

benchmark_pt(model, test_loader, "../outputs/models/llara_model.pt")

# 4) Export to ONNX and benchmark
dummy = torch.randn(1, input_dim)
torch.onnx.export(
    model, dummy, "../outputs/models/llara.onnx",
    export_params=True, opset_version=14,
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input":{0:"batch"}, "output":{0:"batch"}}
)
print("Exported ONNX to outputs/models/llara.onnx")

def benchmark_onnx(path, providers, loader):
    print(f"--- {providers[0]} ---")
    # Report ONNX model size
    print(f"Model size: {os.path.getsize(path)/1e6:.2f} MB")

    sess = ort.InferenceSession(path, providers=providers)
    inp_name = sess.get_inputs()[0].name

    # Compute test accuracy
    correct = total = 0
    for x, y in loader:
        out = sess.run(None, {inp_name: x.numpy()})[0]
        preds = out.argmax(axis=1)
        correct += (preds == y.numpy()).sum()
        total += y.size(0)
    print(f"Accuracy: {correct/total*100:.2f}%")

    # Measure single-sample latency
    single = next(iter(loader))[0][:1].numpy()
    sess.run(None, {inp_name: single})  # warm-up
    latencies = []
    for _ in range(100):
        t0 = time.time()
        sess.run(None, {inp_name: single})
        latencies.append((time.time() - t0) * 1000)
    print(f"Latency (ms) median/95th: "
          f"{np.percentile(latencies,50):.2f}/{np.percentile(latencies,95):.2f}")

    # Measure batch throughput
    batch = next(iter(loader))[0].numpy()
    sess.run(None, {inp_name: batch})  # warm-up
    durations = []
    for _ in range(50):
        t0 = time.time()
        sess.run(None, {inp_name: batch})
        durations.append(time.time() - t0)
    fps = batch.shape[0] * 50 / sum(durations)
    print(f"Throughput (FPS): {fps:.1f}")

benchmark_onnx("../outputs/models/llara.onnx", ["CPUExecutionProvider"], test_loader)

# 5) Apply graph optimizations (operator fusion)
opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
opts.optimized_model_filepath = "../outputs/models/llara_opt.onnx"
ort.InferenceSession(
    "../outputs/models/llara.onnx",
    sess_options=opts,
    providers=["CPUExecutionProvider"]
)
print("Graph-optimized ONNX → outputs/models/llara_opt.onnx")
benchmark_onnx("../outputs/models/llara_opt.onnx", ["CPUExecutionProvider"], test_loader)

# 6) Dynamic quantization
quantize_dynamic(
    "../outputs/models/llara.onnx",
    "../outputs/models/llara_quant_dyn.onnx",
    weight_type=QuantType.QInt8
)
print("Dynamic-quant ONNX → outputs/models/llara_quant_dyn.onnx")
benchmark_onnx("../outputs/models/llara_quant_dyn.onnx", ["CPUExecutionProvider"], test_loader)

# 7) Static quantization
class CalibReader(CalibrationDataReader):
    def __init__(self, loader):
        self.iterator = iter(loader)
    def get_next(self):
        try:
            batch, _ = next(self.iterator)
            return {"input": batch.numpy()}
        except StopIteration:
            return None

quantize_static(
    "../outputs/models/llara.onnx",
    "../outputs/models/llara_quant_stat.onnx",
    calib_data_reader=CalibReader(test_loader),
    weight_type=QuantType.QInt8,
    optimize_model=True
)
print("Static-quant ONNX → outputs/models/llara_quant_stat.onnx")
benchmark_onnx("../outputs/models/llara_quant_stat.onnx", ["CPUExecutionProvider"], test_loader)

# 8) Compare hardware-specific execution providers
for provider in ["CUDAExecutionProvider", "TensorrtExecutionProvider", "OpenVINOExecutionProvider"]:
    try:
        benchmark_onnx("../outputs/models/llara.onnx", [provider], test_loader)
    except Exception as e:
        print(f"{provider} not available: {e}")
