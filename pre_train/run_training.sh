#!/bin/bash
# run_training.sh - Execute the entire training pipeline

# Exit script if any command fails
set -e

# Define environment variables (can be overridden by passing them before script execution)
export MLFLOW_URI=${MLFLOW_URI:-"http://localhost:8000"}
export MINIO_URI=${MINIO_URI:-"http://localhost:9000"}
export MINIO_ACCESS_KEY=${MINIO_ACCESS_KEY:-"minioadmin"}
export MINIO_SECRET_KEY=${MINIO_SECRET_KEY:-"minioadmin"}
export BLOCK_STORAGE_MOUNT=${BLOCK_STORAGE_MOUNT:-"/mnt/object"}
export RUN_ID=$(date +%Y%m%d%H%M%S)

# Memory optimization parameters
export BERT_BATCH_SIZE=${BERT_BATCH_SIZE:-"16"}
export CHUNK_SIZE=${CHUNK_SIZE:-"50000"}
export MEMORY_LIMIT_PERCENT=${MEMORY_LIMIT_PERCENT:-"85.0"}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"max_split_size_mb:128"}

# Print environment
echo "=== Training Environment ==="
echo "MLflow URI:          $MLFLOW_URI"
echo "MinIO URI:           $MINIO_URI"
echo "Block Storage Mount: $BLOCK_STORAGE_MOUNT"
echo "Run ID:              $RUN_ID"
echo "=== Memory Settings ==="
echo "BERT Batch Size:     $BERT_BATCH_SIZE"
echo "Chunk Size:          $CHUNK_SIZE"
echo "Memory Limit:        $MEMORY_LIMIT_PERCENT%"
echo "==========================="

# Create required directories
mkdir -p ./outputs ./logs
echo "Created output and log directories"

# Check if block storage is mounted
if [ ! -d "$BLOCK_STORAGE_MOUNT" ]; then
    echo "ERROR: Block storage not mounted at $BLOCK_STORAGE_MOUNT"
    echo "Run ./mount_block_storage.sh first or set BLOCK_STORAGE_MOUNT correctly"
    exit 1
fi

echo "Block storage found at $BLOCK_STORAGE_MOUNT"

# Check available memory and adjust parameters if needed
TOTAL_MEMORY_MB=$(free -m | grep "Mem:" | awk '{print $2}')
echo "Total system memory: ${TOTAL_MEMORY_MB}MB"

if [ $TOTAL_MEMORY_MB -lt 8000 ]; then
    # Low memory system (less than 8GB)
    echo "Low memory system detected, reducing batch and chunk sizes"
    export BERT_BATCH_SIZE=8
    export CHUNK_SIZE=10000
    export MEMORY_LIMIT_PERCENT=75.0
    echo "Adjusted settings: BERT_BATCH_SIZE=$BERT_BATCH_SIZE, CHUNK_SIZE=$CHUNK_SIZE"
elif [ $TOTAL_MEMORY_MB -lt 16000 ]; then
    # Medium memory system (8-16GB)
    echo "Medium memory system detected, using moderate batch and chunk sizes"
    export BERT_BATCH_SIZE=12
    export CHUNK_SIZE=20000
    echo "Adjusted settings: BERT_BATCH_SIZE=$BERT_BATCH_SIZE, CHUNK_SIZE=$CHUNK_SIZE"
fi

# Check if we should use Docker Compose or direct Python execution
if [ "$USE_DOCKER" = "true" ]; then
    echo "=== Starting training pipeline with Docker Compose ==="
    
    # Make script executable
    chmod +x mount_block_storage.sh
    
    # Check if Docker and Docker Compose are installed
    if ! command -v docker &> /dev/null || ! command -v docker-compose &> /dev/null; then
        echo "ERROR: Docker and/or Docker Compose not installed"
        exit 1
    fi
    
    # Export memory settings to Docker environment
    docker-compose up --build \
        -e BERT_BATCH_SIZE=$BERT_BATCH_SIZE \
        -e CHUNK_SIZE=$CHUNK_SIZE \
        -e MEMORY_LIMIT_PERCENT=$MEMORY_LIMIT_PERCENT \
        -e PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF
    
    # Check if training completed successfully
    if [ $? -eq 0 ]; then
        echo "=== Training pipeline completed successfully ==="
    else
        echo "=== ERROR: Training pipeline failed ==="
        exit 1
    fi
else
    echo "=== Starting training pipeline with Python ==="
    
    # Check if required Python packages are installed
    pip install -r requirements.txt
    
    # Run each training script sequentially
    echo "=== Step 1/4: BERT Training ==="
    python bert_train.py
    if [ $? -ne 0 ]; then
        echo "BERT training failed"
        exit 1
    fi
    
    # Clear memory between steps
    echo "Clearing memory cache..."
    sync && echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
    
    echo "=== Step 2/4: LightGCN Training ==="
    python lightgcn_train.py
    if [ $? -ne 0 ]; then
        echo "LightGCN training failed"
        exit 1
    fi
    
    # Clear memory between steps
    echo "Clearing memory cache..."
    sync && echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
    
    echo "=== Step 3/4: MLP Training ==="
    python mlp_train.py
    if [ $? -ne 0 ]; then
        echo "MLP training failed"
        exit 1
    fi
    
    # Clear memory between steps
    echo "Clearing memory cache..."
    sync && echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
    
    echo "=== Step 4/4: LlaRA Training ==="
    python llara_train.py
    if [ $? -ne 0 ]; then
        echo "LlaRA training failed"
        exit 1
    fi
    
    echo "=== Training pipeline completed successfully ==="
fi

# Create a model version file
MODEL_VERSION=$(date +%Y%m%d%H%M%S)
echo "$MODEL_VERSION" > ./outputs/model_version.txt
echo "Model version: $MODEL_VERSION"

# Clean up temporary files
echo "Cleaning up temporary files..."
find ./outputs -name "*_temp_*" -type f -delete

echo "=== Training artifacts ==="
ls -la ./outputs/
echo "=========================="

echo "=== Training logs ==="
ls -la ./logs/
echo "=====================" 