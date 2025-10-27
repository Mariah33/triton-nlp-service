# Multi-stage build using UV container for dependency management
# Stage 1: Build dependencies with UV
FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim AS builder

# Set working directory
WORKDIR /app

# Enable bytecode compilation for faster startup
ENV UV_COMPILE_BYTECODE=1

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies into /app/.venv using lock file
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Copy the rest of the application
COPY . .

# Install the project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Stage 2: Triton Inference Server runtime
FROM nvcr.io/nvidia/tritonserver:24.08-py3

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libicu-dev \
    pkg-config \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy virtual environment from builder
COPY --from=builder /app/.venv /workspace/.venv

# Set PATH to use the virtual environment
ENV PATH="/workspace/.venv/bin:$PATH"
ENV PYTHONPATH="/workspace:$PYTHONPATH"

# Download spaCy models (Python packages - must be in container)
RUN python3 -m spacy download en_core_web_sm && \
    python3 -m spacy download en_core_web_md

# Note: NLTK data and transformer models are mounted from host via docker-compose
# This reduces image size while keeping spaCy models (Python packages) in the container
# Run 'make download-models' on host to prepare NLTK and transformer models

# Create directories for model repository and model cache
RUN mkdir -p /models /models_cache

# Copy model repository
COPY model_repository /models

# Copy application code
COPY . /workspace/

# Expose Triton's gRPC, HTTP, and metrics ports
EXPOSE 8000 8001 8002

# Set additional environment variables
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/v2/health/ready || exit 1

# Start Triton Server
CMD ["tritonserver", "--model-repository=/models", "--strict-model-config=false", "--log-verbose=1"]
