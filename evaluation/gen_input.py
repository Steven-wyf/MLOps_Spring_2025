# gen_input.py
import json
import numpy as np

# Generate a dummy 128-dimensional embedding vector and write to input.json

# 1) Create a random vector of length 128
emb = np.random.rand(128).tolist()

# 2) Build the payload
payload = {"embeddings": emb}

# 3) Write to input.json
with open("input.json", "w") as f:
    json.dump(payload, f)

print("Generated input.json with length:", len(emb))
