#!/bin/bash
# run_training.sh - Execute the entire training pipeline

# Exit script if any command fails
set -e

# Define environment variables (can be overridden by passing them before script execution)
export MLFLOW_URI=${MLFLOW_URI:-"http://localhost:8000"}
export MINIO_URI=${MINIO_URI:-"http://localhost:9000"}
export MINIO_ACCESS_KEY=${MINIO_ACCESS_KEY:-"minioadmin"}
export MINIO_SECRET_KEY=${MINIO_SECRET_KEY:-"minioadmin"}
export BLOCK_STORAGE_MOUNT=${BLOCK_STORAGE_MOUNT:-"/mnt/block"}
export RUN_ID=$(date +%Y%m%d%H%M%S)

# Print environment
echo "=== Training Environment ==="
echo "MLflow URI:          $MLFLOW_URI"
echo "MinIO URI:           $MINIO_URI"
echo "Block Storage Mount: $BLOCK_STORAGE_MOUNT"
echo "Run ID:              $RUN_ID"
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
    
    # Run the training pipeline using Docker Compose
    docker-compose up --build
    
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
    
    echo "=== Step 2/4: LightGCN Training ==="
    python lightgcn_train.py
    if [ $? -ne 0 ]; then
        echo "LightGCN training failed"
        exit 1
    fi
    
    echo "=== Step 3/4: MLP Training ==="
    python mlp_train.py
    if [ $? -ne 0 ]; then
        echo "MLP training failed"
        exit 1
    fi
    
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

echo "=== Training artifacts ==="
ls -la ./outputs/
echo "=========================="

echo "=== Training logs ==="
ls -la ./logs/
echo "=====================" 