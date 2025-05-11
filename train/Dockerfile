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
COPY train/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user
RUN useradd -m -s /bin/bash mlops
RUN chown -R mlops:mlops /app

# Set volume mount points
VOLUME ["/mnt/object"]

# Copy training scripts
COPY train/*.py ./
COPY data_processing/data_preprocess.py ./data_processing/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV MLFLOW_S3_ENDPOINT_URL=http://129.114.25.37:9000
ENV AWS_ACCESS_KEY_ID=admin
ENV AWS_SECRET_ACCESS_KEY=hrwbqzUS85G253yKi43T
ENV AWS_DEFAULT_REGION=us-east-1

# Switch to non-root user
USER mlops

# Default command
CMD ["python", "bert_encoding.py"] 