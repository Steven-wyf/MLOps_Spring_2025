# Use standard PyTorch image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY inference/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user
RUN useradd -m -s /bin/bash mlops
RUN chown -R mlops:mlops /app

# Set volume mount points (read-only for models, read-write for inference data)
VOLUME ["/mnt/object"]

# Copy all necessary files
COPY inference/inference_api.py .
COPY data_processing/data_preprocess.py ./data_processing/
COPY inference/templates ./templates/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV MLFLOW_S3_ENDPOINT_URL=http://minio:9000
ENV AWS_ACCESS_KEY_ID=admin
ENV AWS_SECRET_ACCESS_KEY=hrwbqzUS85G253yKi43T

# Switch to non-root user
USER mlops

# Default command with host set to 0.0.0.0 to allow external access
CMD ["uvicorn", "inference_api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"] 