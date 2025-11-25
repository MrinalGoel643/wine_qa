# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY config.yaml .
COPY train.py .
COPY download_from_azure.py .
COPY preprocess_wine_data.py .

# Create directories for models and data
RUN mkdir -p models data mlruns mlartifacts

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command: train the model
CMD ["python", "train.py"]