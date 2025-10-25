# Triton Inference Server with NLP Models
FROM nvcr.io/nvidia/tritonserver:23.06-py3

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

# Download spaCy models
RUN python3 -m spacy download en_core_web_sm && \
    python3 -m spacy download en_core_web_md

# Download NLTK data
RUN python3 -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('maxent_ne_chunker'); nltk.download('words')"

# Create directories for model repository
RUN mkdir -p /models

# Copy model repository
COPY model_repository /models

# Create Python environment packages for each model
RUN cd /models && \
    for model_dir in preprocessing data_type_detector transliteration translation ner postprocessing; do \
        if [ -d "$model_dir/1" ]; then \
            echo "Creating environment for $model_dir"; \
            cd /models/$model_dir/1 && \
            tar -czf ../env.tar.gz -C /usr/local/lib/python3.10/dist-packages . || true; \
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
