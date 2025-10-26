# Triton Inference Server with NLP Models
FROM nvcr.io/nvidia/tritonserver:24.08-py3

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libicu-dev \
    pkg-config \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements
COPY requirements.txt /workspace/

# Install Python dependencies
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Note: ML models (spaCy, NLTK, transformers) are mounted from host via docker-compose
# This significantly reduces image size and build time
# Run 'make download-models' on host before starting containers

# Create directories for model repository and model cache
RUN mkdir -p /models /models_cache

# Copy model repository
COPY model_repository /models

# Create Python environment packages for each model
RUN cd /models && \
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')") && \
    for model_dir in preprocessing data_type_detector transliteration translation ner postprocessing; do \
        if [ -d "$model_dir/1" ]; then \
            echo "Creating environment for $model_dir (Python $PYTHON_VERSION)"; \
            cd /models/$model_dir/1 && \
            tar -czf ../env.tar.gz -C /usr/local/lib/python${PYTHON_VERSION}/dist-packages . || true; \
        fi \
    done

# Expose Triton's gRPC, HTTP, and metrics ports
EXPOSE 8000 8001 8002

# Set environment variables
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PYTHONPATH=/workspace:$PYTHONPATH

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/v2/health/ready || exit 1

# Start Triton Server
CMD ["tritonserver", "--model-repository=/models", "--strict-model-config=false", "--log-verbose=1"]
