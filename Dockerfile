# Use CUDA base image if GPU is needed, otherwise use Python base
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY src/ /app/src/

# Create necessary directories
RUN mkdir -p /app/temp /app/output

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "src/main/app.py"] 