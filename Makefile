# Makefile for Triton NLP Service
# Package manager: uv
# Formatter/Linter: ruff
# Testing: pytest

# Variables
PYTHON := python3
UV := uv
RUFF := ruff
PYTEST := pytest
PROJECT_NAME := triton-nlp-service
PYTHON_VERSION := 3.10

# Directories
SRC_DIR := src
TEST_DIR := tests
MODEL_DIR := model_repository
CLIENT_DIR := client
DOCS_DIR := docs

# Docker variables
DOCKER_IMAGE := triton-nlp-service
DOCKER_TAG := latest


# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Default target
.DEFAULT_GOAL := help

# Phony targets
.PHONY: help install install-dev lint format test test-unit test-integration test-coverage \
        run-local run-docker build-docker push-docker clean clean-cache clean-docker \
        setup-pre-commit check security-check type-check docs serve-docs \
        validate-models benchmark create-env

# Help target
help: ## Show this help message
	@echo "$(BLUE)Triton NLP Service - Makefile Commands$(NC)"
	@echo ""
	@echo "$(GREEN)Available targets:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Usage:$(NC)"
	@echo "  make install        - Install production dependencies"
	@echo "  make test          - Run all tests"
	@echo "  make run-local     - Run service locally"

# Environment setup
create-env: ## Create virtual environment with uv
	@echo "$(GREEN)Creating virtual environment with uv...$(NC)"
	$(UV) venv --python $(PYTHON_VERSION)
	@echo "$(GREEN)Virtual environment created. Activate with: source .venv/bin/activate$(NC)"

# Installation targets
install: ## Install production dependencies with uv
	@echo "$(GREEN)Installing production dependencies with uv...$(NC)"
	$(UV) pip install -e .
	@echo "$(GREEN)Installing spaCy models...$(NC)"
	$(PYTHON) -m spacy download en_core_web_sm
	$(PYTHON) -m spacy download en_core_web_md
	@echo "$(GREEN)Downloading NLTK data...$(NC)"
	$(PYTHON) -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('maxent_ne_chunker'); nltk.download('words')"
	@echo "$(GREEN)Production dependencies installed successfully!$(NC)"

install-dev: install ## Install development dependencies with uv
	@echo "$(GREEN)Installing development dependencies with uv...$(NC)"
	$(UV) pip install -e ".[dev]"
	@echo "$(GREEN)Setting up pre-commit hooks...$(NC)"
	pre-commit install
	@echo "$(GREEN)Development environment ready!$(NC)"

install-all: ## Install all optional dependencies
	@echo "$(GREEN)Installing all dependencies with uv...$(NC)"
	$(UV) pip install -e ".[all]"
	@echo "$(GREEN)All dependencies installed!$(NC)"

install-docs: ## Install documentation dependencies
	@echo "$(GREEN)Installing documentation dependencies...$(NC)"
	$(UV) pip install -e ".[docs]"
	@echo "$(GREEN)Documentation dependencies installed!$(NC)"

install-monitoring: ## Install monitoring dependencies
	@echo "$(GREEN)Installing monitoring dependencies...$(NC)"
	$(UV) pip install -e ".[monitoring]"
	@echo "$(GREEN)Monitoring dependencies installed!$(NC)"

install-uv: ## Install uv package manager if not present
	@command -v uv >/dev/null 2>&1 || { \
		echo "$(YELLOW)Installing uv package manager...$(NC)"; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		echo "$(GREEN)uv installed successfully!$(NC)"; \
	}

# Linting and formatting
lint: ## Run ruff linter on all Python files
	@echo "$(GREEN)Running ruff linter...$(NC)"
	$(RUFF) check $(SRC_DIR) $(MODEL_DIR) $(CLIENT_DIR) $(TEST_DIR) --show-fixes
	@echo "$(GREEN)Checking import sorting...$(NC)"
	$(RUFF) check --select I $(SRC_DIR) $(MODEL_DIR) $(CLIENT_DIR) $(TEST_DIR)
	@echo "$(GREEN)Linting complete!$(NC)"

lint-fix: ## Run ruff linter and automatically fix issues
	@echo "$(GREEN)Running ruff linter with auto-fix...$(NC)"
	$(RUFF) check --fix $(SRC_DIR) $(MODEL_DIR) $(CLIENT_DIR) $(TEST_DIR)
	@echo "$(GREEN)Auto-fix complete!$(NC)"

format: ## Format code with ruff
	@echo "$(GREEN)Formatting code with ruff...$(NC)"
	$(RUFF) format $(SRC_DIR) $(MODEL_DIR) $(CLIENT_DIR) $(TEST_DIR)
	@echo "$(GREEN)Sorting imports...$(NC)"
	$(RUFF) check --select I --fix $(SRC_DIR) $(MODEL_DIR) $(CLIENT_DIR) $(TEST_DIR)
	@echo "$(GREEN)Formatting complete!$(NC)"

check-format: ## Check if code is formatted correctly (no changes)
	@echo "$(GREEN)Checking code format...$(NC)"
	$(RUFF) format --check $(SRC_DIR) $(MODEL_DIR) $(CLIENT_DIR) $(TEST_DIR)
	@echo "$(GREEN)Format check complete!$(NC)"

# Testing
test: test-unit test-integration ## Run all tests

test-unit: ## Run unit tests with pytest
	@echo "$(GREEN)Running unit tests with pytest...$(NC)"
	$(PYTEST) $(TEST_DIR)/unit \
		--verbose \
		--color=yes \
		--tb=short \
		--disable-warnings
	@echo "$(GREEN)Unit tests complete!$(NC)"

test-integration: ## Run integration tests
	@echo "$(GREEN)Running integration tests...$(NC)"
	$(PYTEST) $(TEST_DIR)/integration \
		--verbose \
		--color=yes \
		--tb=short \
		--disable-warnings
	@echo "$(GREEN)Integration tests complete!$(NC)"

test-coverage: ## Run tests with coverage report
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	$(PYTEST) $(TEST_DIR) \
		--cov=$(SRC_DIR) \
		--cov=$(MODEL_DIR) \
		--cov-report=html \
		--cov-report=term-missing \
		--cov-report=xml \
		--verbose
	@echo "$(GREEN)Coverage report generated in htmlcov/index.html$(NC)"

test-specific: ## Run specific test file (use TEST=path/to/test.py)
	@echo "$(GREEN)Running specific test: $(TEST)$(NC)"
	$(PYTEST) $(TEST) --verbose --color=yes --tb=short

test-watch: ## Run tests in watch mode (auto-rerun on changes)
	@echo "$(GREEN)Starting test watcher...$(NC)"
	$(PYTEST)-watch $(TEST_DIR) --verbose --color=yes

# Code quality checks
type-check: ## Run type checking with mypy
	@echo "$(GREEN)Running type checking with mypy...$(NC)"
	mypy $(SRC_DIR) $(MODEL_DIR) $(CLIENT_DIR) --ignore-missing-imports
	@echo "$(GREEN)Type checking complete!$(NC)"

security-check: ## Run security vulnerability checks
	@echo "$(GREEN)Running security checks...$(NC)"
	bandit -r $(SRC_DIR) $(MODEL_DIR) $(CLIENT_DIR) -ll
	safety check --json
	@echo "$(GREEN)Security checks complete!$(NC)"

check: lint type-check security-check ## Run all code quality checks

# Running locally
run-local: ## Run Triton server locally
	@echo "$(GREEN)Starting Triton server locally...$(NC)"
	@echo "$(YELLOW)Checking if Triton is installed...$(NC)"
	@command -v tritonserver >/dev/null 2>&1 || { \
		echo "$(RED)Triton server not found. Please run: make run-docker$(NC)"; \
		exit 1; \
	}
	tritonserver --model-repository=$(MODEL_DIR) \
		--backend-config=python,shm-default-byte-size=268435456 \
		--http-port=8000 \
		--grpc-port=8001 \
		--metrics-port=8002 \
		--log-verbose=1

run-fastapi: ## Run FastAPI server
	@echo "$(GREEN)Starting FastAPI server...$(NC)"
	cd $(CLIENT_DIR) && \
	$(UV) run uvicorn fastapi_server:app \
		--host 0.0.0.0 \
		--port 8080 \
		--reload \
		--log-level info

run-client-test: ## Run client test suite
	@echo "$(GREEN)Running client tests...$(NC)"
	cd $(CLIENT_DIR) && \
	$(PYTHON) triton_client.py --test

# Docker operations
run-docker: build-docker ## Run service with Docker Compose
	@echo "$(GREEN)Starting services with Docker Compose...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)Waiting for services to be ready...$(NC)"
	@sleep 10
	@docker-compose ps
	@echo "$(GREEN)Services are running!$(NC)"
	@echo "  HTTP: http://localhost:8000"
	@echo "  gRPC: localhost:8001"
	@echo "  Metrics: http://localhost:8002"
	@echo "  FastAPI: http://localhost:8080"

build-docker: ## Build Docker image
	@echo "$(GREEN)Building Docker image...$(NC)"
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .
	@echo "$(GREEN)Docker image built: $(DOCKER_IMAGE):$(DOCKER_TAG)$(NC)"

push-docker: ## Push Docker image to registry
	@echo "$(GREEN)Pushing Docker image to registry...$(NC)"
	docker tag $(DOCKER_IMAGE):$(DOCKER_TAG) $(DOCKER_REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG)
	docker push $(DOCKER_REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG)
	@echo "$(GREEN)Image pushed to $(DOCKER_REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG)$(NC)"

stop-docker: ## Stop Docker Compose services
	@echo "$(YELLOW)Stopping Docker services...$(NC)"
	docker-compose down
	@echo "$(GREEN)Services stopped$(NC)"

logs-docker: ## Show Docker Compose logs
	docker-compose logs -f --tail=100

# Model validation
validate-models: ## Validate Triton model configurations
	@echo "$(GREEN)Validating model configurations...$(NC)"
	@for model in $(MODEL_DIR)/*/; do \
		model_name=$$(basename $$model); \
		echo "$(BLUE)Checking $$model_name...$(NC)"; \
		if [ -f "$$model/config.pbtxt" ]; then \
			echo "  ✓ config.pbtxt found"; \
		else \
			echo "  $(RED)✗ config.pbtxt missing$(NC)"; \
		fi; \
		if [ -d "$$model/1" ]; then \
			echo "  ✓ version 1 found"; \
		else \
			echo "  $(RED)✗ version 1 missing$(NC)"; \
		fi; \
	done
	@echo "$(GREEN)Model validation complete!$(NC)"

# Benchmarking
benchmark: ## Run performance benchmarks
	@echo "$(GREEN)Running performance benchmarks...$(NC)"
	$(PYTHON) $(TEST_DIR)/benchmark/run_benchmarks.py
	@echo "$(GREEN)Benchmark results saved to benchmark_results.json$(NC)"

benchmark-models: ## Benchmark individual models
	@echo "$(GREEN)Benchmarking individual models...$(NC)"
	perf_analyzer -m data_type_detector_ml \
		-u localhost:8001 \
		--concurrency-range 1:10:2 \
		--measurement-interval 5000
	@echo "$(GREEN)Model benchmarking complete!$(NC)"

# Documentation
docs: ## Build documentation
	@echo "$(GREEN)Building documentation...$(NC)"
	cd $(DOCS_DIR) && mkdocs build
	@echo "$(GREEN)Documentation built in $(DOCS_DIR)/site/$(NC)"

serve-docs: ## Serve documentation locally
	@echo "$(GREEN)Serving documentation at http://localhost:8088$(NC)"
	cd $(DOCS_DIR) && mkdocs serve -p 8088

# Cleaning
clean: clean-cache clean-docker ## Clean all generated files and caches
	@echo "$(GREEN)Cleaning project...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	rm -f .coverage
	rm -f coverage.xml
	rm -rf .venv
	@echo "$(GREEN)Project cleaned!$(NC)"

clean-cache: ## Clean Python and model caches
	@echo "$(YELLOW)Cleaning caches...$(NC)"
	rm -rf ~/.cache/torch
	rm -rf ~/.cache/huggingface
	rm -rf ~/.cache/triton
	@echo "$(GREEN)Caches cleaned!$(NC)"

clean-docker: ## Clean Docker resources
	@echo "$(YELLOW)Cleaning Docker resources...$(NC)"
	docker-compose down -v --remove-orphans 2>/dev/null || true
	docker system prune -f
	@echo "$(GREEN)Docker resources cleaned!$(NC)"

# Development helpers
setup-pre-commit: ## Setup pre-commit hooks
	@echo "$(GREEN)Setting up pre-commit hooks...$(NC)"
	pre-commit install
	pre-commit install --hook-type commit-msg
	@echo "$(GREEN)Pre-commit hooks installed!$(NC)"

update-deps: ## Update all dependencies to latest versions
	@echo "$(GREEN)Updating dependencies...$(NC)"
	$(UV) pip compile requirements.in -o requirements.txt --upgrade
	$(UV) pip compile requirements-dev.in -o requirements-dev.txt --upgrade
	@echo "$(GREEN)Dependencies updated!$(NC)"

freeze-deps: ## Freeze current dependencies
	@echo "$(GREEN)Freezing dependencies...$(NC)"
	$(UV) pip freeze > requirements-frozen.txt
	@echo "$(GREEN)Dependencies frozen to requirements-frozen.txt$(NC)"

# CI/CD helpers
ci-setup: install-uv install-dev ## Setup CI environment

ci-test: lint type-check test-coverage ## Run CI test suite

ci-build: build-docker ## Build for CI

# Kubernetes operations
k8s-deploy: ## Deploy to Kubernetes
	@echo "$(GREEN)Deploying to Kubernetes...$(NC)"
	kubectl apply -f deployment/k8s-deployment.yaml
	@echo "$(GREEN)Deployment complete!$(NC)"

k8s-delete: ## Delete from Kubernetes
	@echo "$(YELLOW)Deleting from Kubernetes...$(NC)"
	kubectl delete -f deployment/k8s-deployment.yaml
	@echo "$(GREEN)Deletion complete!$(NC)"

k8s-logs: ## Show Kubernetes logs
	kubectl logs -l app=triton-nlp -f --tail=100

# Monitoring
monitor-start: ## Start monitoring stack (Prometheus + Grafana)
	@echo "$(GREEN)Starting monitoring stack...$(NC)"
	docker-compose -f docker-compose.monitoring.yml up -d
	@echo "$(GREEN)Monitoring available at:$(NC)"
	@echo "  Prometheus: http://localhost:9090"
	@echo "  Grafana: http://localhost:3000 (admin/admin)"

monitor-stop: ## Stop monitoring stack
	@echo "$(YELLOW)Stopping monitoring stack...$(NC)"
	docker-compose -f docker-compose.monitoring.yml down

# Utility targets
shell: ## Open shell in Docker container
	docker-compose exec triton-nlp bash

gpu-check: ## Check GPU availability
	@echo "$(GREEN)Checking GPU availability...$(NC)"
	@python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}')" || echo "$(RED)PyTorch not installed$(NC)"
	@nvidia-smi 2>/dev/null || echo "$(YELLOW)nvidia-smi not available$(NC)"

version: ## Show version information
	@echo "$(BLUE)Triton NLP Service Version Information$(NC)"
	@echo "Python: $$(python --version)"
	@echo "uv: $$(uv --version 2>/dev/null || echo 'not installed')"
	@echo "ruff: $$(ruff --version 2>/dev/null || echo 'not installed')"
	@echo "pytest: $$(pytest --version 2>/dev/null || echo 'not installed')"
	@echo "Docker: $$(docker --version 2>/dev/null || echo 'not installed')"
	@echo "Docker Compose: $$(docker-compose --version 2>/dev/null || echo 'not installed')"

# Advanced targets
profile: ## Profile the application
	@echo "$(GREEN)Starting profiling...$(NC)"
	$(PYTHON) -m cProfile -o profile.stats $(CLIENT_DIR)/triton_client.py --test
	$(PYTHON) -m pstats profile.stats
	@echo "$(GREEN)Profiling complete!$(NC)"

memory-check: ## Check memory usage
	@echo "$(GREEN)Checking memory usage...$(NC)"
	$(PYTHON) -m memory_profiler $(CLIENT_DIR)/triton_client.py --test
	@echo "$(GREEN)Memory check complete!$(NC)"

optimize-models: ## Optimize ML models for inference
	@echo "$(GREEN)Optimizing models...$(NC)"
	$(PYTHON) scripts/optimize_models.py
	@echo "$(GREEN)Model optimization complete!$(NC)"
