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
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create output and log directories
RUN mkdir -p /mnt/block/outputs /mnt/block/logs

# Set volume mount points
VOLUME ["/mnt/block/outputs", "/mnt/block/logs"]

# Copy training scripts
COPY *.py ./

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HSA_OVERRIDE_GFX_VERSION=10.3.0
ENV HIP_VISIBLE_DEVICES=0

# Default command
CMD ["python", "llara_train.py"] 