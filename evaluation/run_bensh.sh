#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# --- Triton queue vs compute latency tests ---
echo ">>> Triton Perf Analyzer Tests"
for c in 1 4 8 16; do
  echo "-- concurrency = $c --"
  docker exec triton perf_analyzer \
    -u localhost:8005 \
    -m music_rec \
    --input-data /evaluation/input.json \
    -b 1 \
    --concurrency-range $c \
    --percentiles=50,95,99
  echo
done

# --- FastAPI throughput tests via wrk ---
echo ">>> FastAPI wrk Benchmark"
API="http://localhost:8080/predict"
for c in 1 4 8 16; do
  echo "-- concurrency = $c --"
  wrk -t$c -c$c -d20s -s post.lua $API
  echo
done

# --- GPU utilization snapshot ---
echo ">>> GPU Utilization (nvidia-smi)"
docker exec triton nvidia-smi
