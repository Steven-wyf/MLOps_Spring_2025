FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directories
RUN mkdir -p /app/outputs /app/logs

# Set volume mountpoints
VOLUME ["/app/outputs", "/app/logs"]

# Copy training scripts
COPY *.py .

# Default command - will be overridden in docker-compose
CMD ["python", "bert_train.py"] 