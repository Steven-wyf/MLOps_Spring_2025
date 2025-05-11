# Use PyTorch with ROCm support for AMD GPUs
FROM pytorch/pytorch:2.0.1-rocm5.6-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY train/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user
RUN useradd -m -s /bin/bash mlops
RUN chown -R mlops:mlops /app

# Set volume mount points (read-only)
VOLUME ["/mnt/object"]

# Copy training scripts
COPY train/*.py ./

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV HSA_OVERRIDE_GFX_VERSION=10.3.0
ENV HIP_VISIBLE_DEVICES=0
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV MLFLOW_S3_ENDPOINT_URL=http://minio:9000
ENV AWS_ACCESS_KEY_ID=admin
ENV AWS_SECRET_ACCESS_KEY=hrwbqzUS85G253yKi43T

# Switch to non-root user
USER mlops

# Default command
CMD ["python"] 