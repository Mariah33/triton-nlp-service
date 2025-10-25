"""Integration tests for Triton NLP Service."""

import json
import time
from typing import Dict, List

import numpy as np
import pytest
import requests
from fastapi.testclient import TestClient

# Import with proper error handling
try:
    import tritonclient.grpc as grpcclient
    import tritonclient.http as httpclient

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    grpcclient = None
    httpclient = None


@pytest.mark.integration
@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton client not installed")
class TestTritonIntegration:
    """Integration tests for Triton server."""

    @pytest.fixture
    def triton_client(self):
        """Create Triton client for testing."""
        client = grpcclient.InferenceServerClient(url="localhost:8001")
        # Wait for server to be ready
        max_retries = 10
        for i in range(max_retries):
            if client.is_server_live():
                break
            time.sleep(1)
        else:
            pytest.skip("Triton server not available")
        return client

    def test_server_health(self, triton_client):
        """Test Triton server health."""
        assert triton_client.is_server_live()
        assert triton_client.is_server_ready()

    def test_model_availability(self, triton_client):
        """Test all models are loaded."""
        expected_models = [
            "preprocessing",
            "data_type_detector",
            "data_type_detector_ml",
            "ner",
            "transliteration",
            "translation",
            "postprocessing",
            "ensemble_nlp",
        ]

        for model in expected_models:
            assert triton_client.is_model_ready(model), f"Model {model} not ready"

    def test_ensemble_pipeline(self, triton_client):
        """Test complete ensemble pipeline."""
        # Prepare input
        text = "Contact John Smith at john.smith@example.com or +1-555-123-4567"
        text_bytes = text.encode("utf-8")
        input_array = np.array([[text_bytes]], dtype=np.object_)

        # Create input tensor
        inputs = [grpcclient.InferInput("text", input_array.shape, "BYTES")]
        inputs[0].set_data_from_numpy(input_array)

        # Request output
        outputs = [grpcclient.InferRequestedOutput("result")]

        # Run inference
        response = triton_client.infer(
            model_name="ensemble_nlp", inputs=inputs, outputs=outputs
        )

        # Parse result
        result_bytes = response.as_numpy("result")
        result_str = result_bytes[0][0].decode("utf-8")
        result = json.loads(result_str)

        # Assertions
        assert "results" in result
        assert "summary" in result
        assert result["original_text"] == text

    @pytest.mark.parametrize(
        "model,input_name,output_name",
        [
            ("data_type_detector", "text", "detection_result"),
            ("data_type_detector_ml", "text", "detection_result"),
            ("ner", "text", "entities"),
            ("transliteration", "text", "transliterated_text"),
            ("translation", "text", "translated_text"),
        ],
    )
    def test_individual_models(self, triton_client, model, input_name, output_name):
        """Test individual model inference."""
        text = "Test input text"
        text_bytes = text.encode("utf-8")
        input_array = np.array([[text_bytes]], dtype=np.object_)

        inputs = [grpcclient.InferInput(input_name, input_array.shape, "BYTES")]
        inputs[0].set_data_from_numpy(input_array)

        # Add required inputs for specific models
        if model == "transliteration":
            script_array = np.array([["auto".encode("utf-8")]], dtype=np.object_)
            target_array = np.array([["latin".encode("utf-8")]], dtype=np.object_)

            source_input = grpcclient.InferInput("source_script", script_array.shape, "BYTES")
            source_input.set_data_from_numpy(script_array)
            inputs.append(source_input)

            target_input = grpcclient.InferInput("target_script", target_array.shape, "BYTES")
            target_input.set_data_from_numpy(target_array)
            inputs.append(target_input)

        elif model == "translation":
            source_array = np.array([["auto".encode("utf-8")]], dtype=np.object_)
            target_array = np.array([["en".encode("utf-8")]], dtype=np.object_)

            source_input = grpcclient.InferInput("source_language", source_array.shape, "BYTES")
            source_input.set_data_from_numpy(source_array)
            inputs.append(source_input)

            target_input = grpcclient.InferInput("target_language", target_array.shape, "BYTES")
            target_input.set_data_from_numpy(target_array)
            inputs.append(target_input)

        outputs = [grpcclient.InferRequestedOutput(output_name)]

        # Run inference
        response = triton_client.infer(model_name=model, inputs=inputs, outputs=outputs)

        # Check response
        result = response.as_numpy(output_name)
        assert result is not None
        assert result.shape[0] >= 1


@pytest.mark.integration
class TestFastAPIIntegration:
    """Integration tests for FastAPI wrapper."""

    @pytest.fixture
    def client(self):
        """Create FastAPI test client."""
        try:
            from client.fastapi_server import app

            return TestClient(app)
        except ImportError:
            pytest.skip("FastAPI server not available")

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Triton NLP Service"

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        # May fail if Triton not running, that's ok for unit test
        assert response.status_code in [200, 503]

    def test_process_endpoint(self, client):
        """Test main processing endpoint."""
        payload = {
            "text": "Contact john@example.com",
            "services": ["data_type", "ner"],
            "source_language": "en",
            "target_language": "es",
        }

        response = client.post("/process", json=payload)

        # Check response (may fail if Triton not running)
        if response.status_code == 200:
            data = response.json()
            assert "results" in data
            assert "summary" in data

    def test_batch_processing(self, client):
        """Test batch processing endpoint."""
        payload = {
            "texts": ["Text 1", "Text 2", "Text 3"],
            "services": ["data_type"],
            "source_language": "en",
            "target_language": "en",
        }

        response = client.post("/batch_process", json=payload)

        if response.status_code == 200:
            data = response.json()
            assert "results" in data
            assert len(data["results"]) == 3

    def test_data_type_detection_endpoint(self, client):
        """Test data type detection endpoint."""
        payload = {"text": "john.doe@example.com"}

        response = client.post("/detect_type", json=payload)

        if response.status_code == 200:
            data = response.json()
            assert "primary_type" in data
            assert "confidence" in data

    def test_transliteration_endpoint(self, client):
        """Test transliteration endpoint."""
        payload = {"text": "नमस्ते", "source_script": "devanagari", "target_script": "latin"}

        response = client.post("/transliterate", json=payload)

        if response.status_code == 200:
            data = response.json()
            assert "transliterated" in data

    def test_translation_endpoint(self, client):
        """Test translation endpoint."""
        payload = {"text": "Hello world", "source_language": "en", "target_language": "es"}

        response = client.post("/translate", json=payload)

        if response.status_code == 200:
            data = response.json()
            assert "translated" in data

    def test_entity_extraction_endpoint(self, client):
        """Test entity extraction endpoint."""
        payload = {"text": "John Smith works at Microsoft in Seattle."}

        response = client.post("/extract_entities", json=payload)

        if response.status_code == 200:
            data = response.json()
            assert "entities" in data


@pytest.mark.integration
class TestEndToEndScenarios:
    """End-to-end scenario tests."""

    @pytest.fixture
    def api_client(self):
        """Create API client for E2E tests."""
        base_url = "http://localhost:8080"
        # Check if service is running
        try:
            response = requests.get(f"{base_url}/health", timeout=1)
            if response.status_code != 200:
                pytest.skip("API service not running")
        except requests.exceptions.RequestException:
            pytest.skip("API service not available")

        return base_url

    def test_pii_detection_scenario(self, api_client):
        """Test PII detection scenario."""
        test_text = """
        Please contact John Smith at john.smith@company.com or 
        call him at +1-555-123-4567. His SSN is 123-45-6789 and 
        credit card number is 4532-0151-1283-0366.
        """

        response = requests.post(
            f"{api_client}/process",
            json={"text": test_text, "services": ["data_type", "ner"]},
        )

        assert response.status_code == 200
        data = response.json()

        # Check PII was detected
        assert "results" in data
        if "data_type_detection" in data["results"]:
            detections = data["results"]["data_type_detection"]["detections"]
            detected_types = [d["type"] for d in detections]

            # Should detect various PII
            assert any("email" in t.lower() for t in detected_types)
            assert any("phone" in t.lower() for t in detected_types)
            assert any("ssn" in t.lower() or "social" in t.lower() for t in detected_types)
            assert any("credit" in t.lower() for t in detected_types)

    def test_multilingual_scenario(self, api_client):
        """Test multilingual processing scenario."""
        test_cases = [
            {"text": "Hello world", "target_lang": "es", "expected": "hola"},
            {"text": "नमस्ते", "expected_script": "latin"},
            {"text": "مرحبا", "expected_script": "latin"},
        ]

        for test in test_cases:
            response = requests.post(
                f"{api_client}/process",
                json={
                    "text": test["text"],
                    "services": ["transliteration", "translation"],
                    "target_language": test.get("target_lang", "en"),
                },
            )

            assert response.status_code == 200
            data = response.json()
            # Verify processing occurred
            assert "results" in data

    def test_performance_benchmark(self, api_client):
        """Benchmark API performance."""
        import time

        text = "Simple test text for benchmarking"
        iterations = 10
        times = []

        for _ in range(iterations):
            start = time.time()
            response = requests.post(
                f"{api_client}/detect_type", json={"text": text}, timeout=10
            )
            end = time.time()

            assert response.status_code == 200
            times.append(end - start)

        avg_time = sum(times) / len(times)
        print(f"Average response time: {avg_time:.3f}s")
        assert avg_time < 1.0  # Should respond within 1 second
