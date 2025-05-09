FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirement.txt .
RUN pip install --no-cache-dir -r requirement.txt

# Copy training scripts
COPY *.py .

# Create output directory
RUN mkdir -p /app/outputs /app/data

# Set working directory as volume
VOLUME ["/app/data", "/app/outputs"]

# Default command runs the specified script (this will be overridden in docker-compose)
CMD ["python", "bert.py"]
