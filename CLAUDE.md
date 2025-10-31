# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Triton NLP Service is a comprehensive NLP pipeline built on NVIDIA Triton Inference Server. It provides transliteration, translation, NER, and data type detection (including PII) through a unified ensemble model architecture.

**Package Manager**: `uv` (fast Python package installer)
**Linter/Formatter**: `ruff`
**Testing**: `pytest`
**Python Version**: 3.10-3.11
**Supported Platforms**: Linux (x86_64, ARM64), macOS (Apple Silicon, Intel)

**Lock File**: `uv.lock` provides reproducible builds across all platforms. The lock file includes wheels for both Linux (Docker deployment) and macOS (local development).

## Development Commands

### Environment Setup
```bash
# Install uv if not present
make install-uv

# Create virtual environment
make create-env
source .venv/bin/activate

# macOS: Install ICU library first (required for PyICU)
brew install icu4c pkg-config
export PKG_CONFIG_PATH="/opt/homebrew/opt/icu4c@77/lib/pkgconfig:$PKG_CONFIG_PATH"
export PATH="/opt/homebrew/opt/icu4c@77/bin:$PATH"

# Install dependencies
make install-dev  # Development dependencies + pre-commit hooks
make install      # Production only

# Download NLTK data and transformer models (spaCy models are in Docker image)
make download-models
```

### Dependency Management
```bash
# Update lock file after changing pyproject.toml
make lock

# Update all dependencies to latest compatible versions
make update-deps

# Install from lock file (reproducible)
make install       # Production dependencies
make install-dev   # Development dependencies
make install-all   # All optional dependencies
```

**Note**: The `uv.lock` file ensures reproducible builds across Linux and macOS. All installations use `uv sync --frozen` to match the lock file exactly.

### Running Tests
```bash
make test              # Run all tests (unit + integration)
make test-unit         # Unit tests only
make test-integration  # Integration tests only
make test-coverage     # Tests with coverage report

# Run specific test file
make test-specific TEST=tests/unit/test_data_type_detector.py

# Or use pytest directly
pytest tests/unit/test_data_type_detector.py -v
```

### Code Quality
```bash
make lint          # Check code with ruff
make lint-fix      # Auto-fix lint issues
make format        # Format code with ruff
make check-format  # Check formatting without changes
make type-check    # Run mypy type checking
make security-check # Run bandit security checks
make check         # Run all checks (lint + type + security)
```

### Running the Service
```bash
# With Docker (recommended)
make run-docker    # Builds and starts all services with docker-compose

# Stop services
make stop-docker

# View logs
make logs-docker

# Local (requires Triton installation)
make run-local     # Runs tritonserver directly

# FastAPI server (client wrapper)
make run-fastapi   # Starts FastAPI on port 8080
```

### Service Endpoints
- **HTTP**: http://localhost:8000 (Triton)
- **gRPC**: localhost:8001 (Triton)
- **Metrics**: http://localhost:8002 (Triton)
- **FastAPI**: http://localhost:8080 (Client wrapper)

## Architecture

### Triton Ensemble Pipeline

The service uses Triton's ensemble model for orchestrating multiple NLP models in a single request:

```
Client Request → ensemble_nlp
    ↓
1. preprocessing (text normalization)
    ↓
2. Parallel execution of 4 models:
   - data_type_detector (regex-based, fast)
   - data_type_detector_ml (ML-based, accurate)
   - ner (Named Entity Recognition)
   - transliteration (script conversion)
   - translation (language translation)
    ↓
3. postprocessing (result aggregation)
    ↓
JSON Response
```

### Model Repository Structure

Each model in `model_repository/` follows Triton's standard layout:
```
model_repository/
├── <model_name>/
│   ├── config.pbtxt      # Triton model configuration
│   └── 1/                # Version directory
│       └── model.py      # Python backend implementation
```

**Key Models:**
- `ensemble_nlp`: Orchestrates the entire pipeline (see `model_repository/ensemble_nlp/config.pbtxt` for flow)
- `preprocessing`: Text normalization, lowercasing, whitespace handling
- `data_type_detector`: Regex-based PII/data type detection (fast, ~1ms)
- `data_type_detector_ml`: ML-based detection using Presidio + transformers (accurate, ~50ms)
- `ner`: Named entity recognition using spaCy
- `transliteration`: Script conversion (e.g., Devanagari → Latin)
- `translation`: Language translation
- `postprocessing`: Aggregates results from all models into JSON

### Ensemble Configuration

The ensemble is defined in `model_repository/ensemble_nlp/config.pbtxt`. Key points:
- **Inputs**: `text`, optional `services`, `source_language`, `target_language`
- **Outputs**: Single JSON `result` containing all model outputs
- **Scheduling**: Steps 2-5 run in parallel, then postprocessing aggregates
- **Batch Size**: Max 16 requests

### Data Type Detection: Two Approaches

This service provides **both** regex and ML-based detection (see `docs/regex_vs_ml_comparison.md`):

1. **Regex-based** (`data_type_detector`): Fast (<1ms), deterministic, good for well-formatted data
2. **ML-based** (`data_type_detector_ml`): Accurate (90-95%), context-aware, handles obfuscated/partial data

**When to use which:**
- Use regex for high-throughput, simple formats, edge devices
- Use ML for critical PII detection, user-generated content, compliance requirements

## Client Usage

### Python Client (`client/triton_client.py`)
```python
from client.triton_client import TritonNLPClient

# Initialize client
client = TritonNLPClient(url="localhost:8001", protocol="grpc")

# Process text through ensemble
result = client.process_text("Your text here")

# Check specific models
client.check_models()  # Validates all models are loaded
```

### FastAPI Server (`client/fastapi_server.py`)
Provides REST API wrapper around Triton gRPC client. Run with:
```bash
make run-fastapi
# Or: cd client && uvicorn fastapi_server:app --reload
```

## Testing Strategy

- **Unit Tests** (`tests/unit/`): Test individual model logic without Triton server
- **Integration Tests** (`tests/integration/`): Test full pipeline with running Triton server
- **Fixtures** (`tests/conftest.py`): Shared test fixtures

**Test Requirements:**
- Integration tests require Triton server running (`make run-docker`)
- Unit tests can run standalone

## Important Configuration Files

- `pyproject.toml`: All project configuration (dependencies, tool configs, build settings)
- `Makefile`: Development commands and workflows
- `docker-compose.yml`: Multi-service setup (Triton + Redis + Prometheus + Grafana)
- `.pre-commit-config.yaml`: Pre-commit hooks for code quality

## Code Style

**Ruff Configuration** (`pyproject.toml`):
- Line length: 150 characters
- Docstring style: Google format
- Extensive linting rules enabled (see `tool.ruff.select` in pyproject.toml)
- Auto-fix enabled for most issues

**Type Checking**:
- MyPy configured but permissive (`disallow_untyped_defs = false`)
- Type hints encouraged but not required
- Ignore missing imports for third-party libraries

## Key Dependencies

**Core NLP:**
- `torch`, `transformers`, `sentence-transformers`: Deep learning
- `spacy`, `nltk`: Traditional NLP
- `presidio-analyzer`, `presidio-anonymizer`: PII detection

**Triton:**
- `tritonclient[all]`: Python client for Triton server
- Models use `triton_python_backend_utils` (imported in model.py files)

**Data Type Detection:**
- `phonenumbers`, `email-validator`, `python-stdnum`: Validators
- `scrubadub`: Additional PII scrubbing

**Translation/Transliteration:**
- `indic-nlp-library`, `ai4bharat-transliteration`: Indic language support
- `sentencepiece`, `sacremoses`: Tokenization

## Model Implementation Pattern

All Python backend models follow this structure:
```python
class TritonPythonModel:
    def initialize(self, args):
        """Called once on model load. Parse config, load resources."""
        self.model_config = json.loads(args['model_config'])
        # Initialize model-specific resources

    def execute(self, requests):
        """Process batch of inference requests."""
        responses = []
        for request in requests:
            # Get inputs
            input_tensor = pb_utils.get_input_tensor_by_name(request, "input_name")
            data = input_tensor.as_numpy()

            # Process
            result = self._process(data)

            # Create output
            output_tensor = pb_utils.Tensor("output_name", result)
            responses.append(pb_utils.InferenceResponse([output_tensor]))
        return responses

    def finalize(self):
        """Called on model unload. Cleanup resources."""
        pass
```

## Common Development Tasks

### Adding a New Model
1. Create directory: `model_repository/<model_name>/1/`
2. Add `config.pbtxt` with input/output specs
3. Implement `model.py` with `TritonPythonModel` class
4. Update ensemble config if part of pipeline
5. Validate: `make validate-models`

### Updating Dependencies
```bash
make update-deps    # Update all dependencies
make freeze-deps    # Freeze current versions
```

### Debugging Models
- Check model status: Use `client.check_models()` or Triton health endpoint
- View logs: `make logs-docker` or check container logs
- Validate configs: `make validate-models`
- Test individual models before ensemble integration

### Performance Profiling
```bash
make profile        # Profile with cProfile
make memory-check   # Check memory usage
make benchmark      # Run performance benchmarks
```

## Deployment

**Docker:**
- Base image: NVIDIA Triton Inference Server
- GPU support: Configure in `docker-compose.yml` (CUDA devices)
- Health checks: Built-in at `/v2/health/ready`

**Kubernetes:**
```bash
make k8s-deploy     # Deploy to K8s
make k8s-delete     # Remove from K8s
make k8s-logs       # View logs
```

**Monitoring:**
```bash
make monitor-start  # Start Prometheus + Grafana
# Access: Prometheus (9090), Grafana (3000, admin/admin)
```

## Important Notes

- **Not a Git Repository**: This directory is not currently a git repository. Initialize if needed.
- **GPU Optional**: Service works without GPU but ML models are significantly faster with CUDA
- **Model Downloads**: Run `make download-models` to download NLTK data and transformers (~500MB). spaCy models are downloaded during Docker build.
- **Memory**: ML-based detection requires ~2-4GB RAM for models
- **Python Backend**: All models use Triton Python backend (not TensorRT, ONNX)

## Troubleshooting

### `make install` fails with PyICU error
**Error**: `RuntimeError: Please install pkg-config on your system or set the ICU_VERSION environment variable`

**Solution** (macOS):
```bash
# Install ICU library
brew install icu4c pkg-config

# Set environment variables (adjust version if needed)
export PKG_CONFIG_PATH="/opt/homebrew/opt/icu4c@77/lib/pkgconfig:$PKG_CONFIG_PATH"
export PATH="/opt/homebrew/opt/icu4c@77/bin:$PATH"

# Retry installation
make install
```

**Solution** (Linux):
```bash
sudo apt-get install libicu-dev pkg-config
```

### pyproject.toml has invalid dependency "artillery"
This is a known issue - `artillery` is a JavaScript tool incorrectly listed in the `perf` dependencies. It can be safely ignored or removed from `pyproject.toml`.

### Integration tests fail: "Connection refused"
Ensure Triton server is running:
```bash
make run-docker
# Wait ~30 seconds for models to load
docker-compose ps  # Verify triton-nlp-server is healthy
```