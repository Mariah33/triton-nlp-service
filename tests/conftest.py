"""Pytest configuration and fixtures."""

import json
import os
from pathlib import Path
import sys

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "model_repository"))
sys.path.insert(0, str(project_root / "client"))


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU")
    config.addinivalue_line("markers", "model: Model-specific tests")
    config.addinivalue_line("markers", "benchmark: Performance benchmarks")


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption("--run-slow", action="store_true", default=False, help="Run slow tests")
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests",
    )
    parser.addoption("--run-gpu", action="store_true", default=False, help="Run GPU tests")
    parser.addoption("--benchmark", action="store_true", default=False, help="Run benchmarks")


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers."""
    skip_slow = pytest.mark.skip(reason="Need --run-slow option to run")
    skip_integration = pytest.mark.skip(reason="Need --run-integration option to run")
    skip_gpu = pytest.mark.skip(reason="Need --run-gpu option to run")
    skip_benchmark = pytest.mark.skip(reason="Need --benchmark option to run")

    for item in items:
        if "slow" in item.keywords and not config.getoption("--run-slow"):
            item.add_marker(skip_slow)
        if "integration" in item.keywords and not config.getoption("--run-integration"):
            item.add_marker(skip_integration)
        if "gpu" in item.keywords and not config.getoption("--run-gpu"):
            item.add_marker(skip_gpu)
        if "benchmark" in item.keywords and not config.getoption("--benchmark"):
            item.add_marker(skip_benchmark)


@pytest.fixture(scope="session")
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def sample_texts():
    """Sample texts for testing."""
    return {
        "email": "john.doe@example.com",
        "phone": "+1-555-123-4567",
        "ssn": "123-45-6789",
        "credit_card": "4532-0151-1283-0366",
        "passport": "GB12345678",
        "iban": "GB82 WEST 1234 5698 7654 32",
        "address": "123 Main Street, New York, NY 10001",
        "hindi": "नमस्ते दुनिया",
        "arabic": "مرحبا بالعالم",
        "chinese": "你好世界",
        "mixed": "Contact John Smith at john@example.com or 555-1234",
    }


@pytest.fixture(scope="session")
def mock_triton_response():
    """Mock Triton server response."""
    return {
        "result": json.dumps(
            {
                "original_text": "test",
                "results": {
                    "data_type_detection": {
                        "detected": True,
                        "primary_type": "text",
                        "confidence": 0.9,
                    },
                    "named_entities": {"entities": [], "count": 0},
                },
                "summary": {"key_findings": []},
            }
        )
    }


@pytest.fixture
def mock_ml_models():
    """Mock ML models for testing."""
    from unittest.mock import MagicMock

    return {
        "pii_model": MagicMock(),
        "tokenizer": MagicMock(),
        "zero_shot": MagicMock(),
        "sentence_encoder": MagicMock(),
        "ner_model": MagicMock(),
    }


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment variables before each test."""
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)


# Performance tracking
@pytest.fixture
def track_performance(request):
    """Track test performance."""
    import time

    start_time = time.time()
    yield
    time.time() - start_time


# Test data generators
@pytest.fixture
def generate_test_emails():
    """Generate test email addresses."""
    from faker import Faker

    fake = Faker()
    return [fake.email() for _ in range(10)]


@pytest.fixture
def generate_test_phones():
    """Generate test phone numbers."""
    from faker import Faker

    fake = Faker()
    return [fake.phone_number() for _ in range(10)]


@pytest.fixture
def generate_test_addresses():
    """Generate test addresses."""
    from faker import Faker

    fake = Faker()
    return [fake.address() for _ in range(10)]


# Mock server fixtures
@pytest.fixture
def mock_triton_server():
    """Mock Triton server for testing."""
    from unittest.mock import MagicMock

    server = MagicMock()
    server.is_server_live.return_value = True
    server.is_server_ready.return_value = True
    server.is_model_ready.return_value = True
    return server


@pytest.fixture
def mock_fastapi_client():
    """Mock FastAPI client."""
    from unittest.mock import MagicMock

    client = MagicMock()
    client.post.return_value.status_code = 200
    client.post.return_value.json.return_value = {"status": "success"}
    return client
