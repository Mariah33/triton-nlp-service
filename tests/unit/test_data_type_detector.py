"""Unit tests for data type detector ML model.."""

import json
from unittest.mock import MagicMock, patch

import pytest
import torch
import sys

# Mock triton_python_backend_utils for testing
mock_pb_utils = MagicMock()


class TestDataTypeDetectorML:
    """Test suite for ML-based data type detection.."""

    @pytest.fixture
    def model_config(self):
        """Model configuration fixture.."""
        return {
            "name": "data_type_detector_ml",
            "backend": "python",
            "max_batch_size": 32,
        }

    @pytest.fixture
    def mock_model(self, model_config):
        """Create a mock model instance.."""
        with patch.dict("sys.modules", {"triton_python_backend_utils": mock_pb_utils}):
            # Import after patching
            sys.path.insert(0, "model_repository/data_type_detector_ml/1")
            from model import TritonPythonModel

            model = TritonPythonModel()
            model.model_config = model_config

            # Mock ML models
            model.pii_model = MagicMock()
            model.pii_tokenizer = MagicMock()
            model.zero_shot_classifier = MagicMock()
            model.sentence_encoder = MagicMock()
            model.token_classifier = MagicMock()
            model.presidio_analyzer = MagicMock()

            return model

    @pytest.mark.unit
    def test_email_detection(self, mock_model):
        """Test email address detection.."""
        test_cases = [
            ("john.doe@example.com", True, 0.9),
            ("admin@company.org", True, 0.9),
            ("not_an_email", False, 0.0),
            ("john@", False, 0.0),
        ]

        for text, should_detect, min_confidence in test_cases:
            result = mock_model._detect_data_types_ml(text)

            if should_detect:
                assert result["primary_type"] in ["email", "email address"]
                assert result["confidence"] >= min_confidence
            else:
                assert result["primary_type"] != "email"

    @pytest.mark.unit
    def test_phone_number_detection(self, mock_model):
        """Test phone number detection.."""
        test_cases = [
            ("+1-555-123-4567", True),
            ("(555) 123-4567", True),
            ("555.123.4567", True),
            ("12345", False),
        ]

        for text, should_detect in test_cases:
            result = mock_model._detect_phone_with_library(text)

            if should_detect:
                assert result is not None
                assert result["type"] == "phone_number"
            else:
                assert result is None

    @pytest.mark.unit
    def test_credit_card_detection(self, mock_model):
        """Test credit card number detection.."""
        # Valid test credit card numbers (Luhn valid)
        valid_cards = [
            "4532015112830366",  # Visa
            "5425233430109903",  # Mastercard
            "374245455400126",  # Amex
        ]

        for card in valid_cards:
            assert mock_model._is_valid_credit_card(card) is True

        # Invalid cards
        invalid_cards = ["1234567890123456", "0000000000000000", "123"]

        for card in invalid_cards:
            assert mock_model._is_valid_credit_card(card) is False

    @pytest.mark.unit
    def test_ssn_masking(self, mock_model):
        """Test SSN masking for privacy.."""
        ssn = "123-45-6789"
        masked = mock_model._mask_sensitive_data(ssn, "ssn")
        assert masked == "***-**-6789"

    @pytest.mark.unit
    def test_credit_card_masking(self, mock_model):
        """Test credit card masking.."""
        card = "4532-0151-1283-0366"
        masked = mock_model._mask_sensitive_data(card, "credit_card")
        assert masked == "****-****-****-0366"

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "text,expected_category",
        [
            ("email@example.com", "contact"),
            ("4532015112830366", "financial"),
            ("123-45-6789", "government_id"),
            ("GB12345678", "document"),
            ("192.168.1.1", "technical"),
            ("John Smith", "personal"),
        ],
    )
    def test_category_classification(self, mock_model, text, expected_category):
        """Test category classification for different data types.."""
        # This would test the _get_category method
        # Implementation depends on model structure

    @pytest.mark.unit
    def test_deduplication(self, mock_model):
        """Test deduplication of detection results.."""
        detections = [
            {"type": "email", "confidence": 0.8, "value": "test@test.com"},
            {"type": "email", "confidence": 0.9, "value": "test@test.com"},
            {"type": "phone", "confidence": 0.7, "value": "555-1234"},
        ]

        deduplicated = mock_model._deduplicate_detections(detections)

        assert len(deduplicated) == 2  # Only 2 unique types
        assert deduplicated[0]["confidence"] == 0.9  # Highest confidence kept

    @pytest.mark.integration
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_gpu_inference(self, mock_model):
        """Test GPU inference if available.."""
        mock_model.device = torch.device("cuda")
        text = "Contact john.doe@example.com or call 555-1234"

        # Mock tensor operations
        with patch.object(mock_model, "sentence_encoder") as mock_encoder:
            mock_encoder.encode.return_value = torch.randn(1, 384).to("cuda")
            result = mock_model._detect_data_types_ml(text)

            assert result is not None
            assert "detections" in result

    @pytest.mark.unit
    def test_zero_shot_classification(self, mock_model):
        """Test zero-shot classification.."""
        mock_model.zero_shot_classifier.return_value = {
            "labels": ["email address", "phone number", "general text"],
            "scores": [0.9, 0.05, 0.05],
        }

        text = "john.doe@example.com"
        result = mock_model._detect_data_types_ml(text)

        assert result["primary_type"] == "email address"
        assert result["confidence"] >= 0.9

    @pytest.mark.unit
    def test_presidio_integration(self, mock_model):
        """Test Presidio analyzer integration.."""
        from unittest.mock import MagicMock

        # Mock Presidio result
        mock_result = MagicMock()
        mock_result.entity_type = "PHONE_NUMBER"
        mock_result.score = 0.95
        mock_result.start = 0
        mock_result.end = 12

        mock_model.presidio_analyzer.analyze.return_value = [mock_result]

        text = "555-123-4567"
        result = mock_model._detect_data_types_ml(text)

        assert len(result["detections"]) > 0
        assert any(d["type"] == "phone number" for d in result["detections"])

    @pytest.mark.benchmark
    def test_performance_benchmark(self, mock_model, benchmark):
        """Benchmark detection performance.."""
        text = "Contact john.doe@example.com or call 555-1234"

        # Benchmark the detection
        result = benchmark(mock_model._detect_data_types_ml, text)
        assert result is not None


class TestDataTypeDetectorRegex:
    """Test suite for regex-based data type detection.."""

    @pytest.fixture
    def regex_model(self):
        """Create regex-based model for comparison.."""
        with patch.dict("sys.modules", {"triton_python_backend_utils": mock_pb_utils}):
            from model_repository.data_type_detector.model import TritonPythonModel

            model = TritonPythonModel()
            model.initialize({"model_config": json.dumps({"name": "data_type_detector"})})
            return model

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "text,expected_type",
        [
            ("john@example.com", "email"),
            ("192.168.1.1", "ipv4"),
            ("https://example.com", "url"),
            ("2023-12-25", "date_iso"),
            ("GB12345678", "passport"),
        ],
    )
    def test_pattern_matching(self, regex_model, text, expected_type):
        """Test basic pattern matching.."""
        result = regex_model._detect_data_types(text)
        assert result["primary_type"] == expected_type

    @pytest.mark.unit
    def test_regex_vs_ml_comparison(self, mock_model, regex_model):
        """Compare regex and ML detection accuracy.."""
        test_cases = [
            "john dot doe at gmail dot com",  # Obfuscated email
            "call me at five five five 1234",  # Natural language phone
            "last 4 of SSN: 1234",  # Partial SSN
        ]

        for text in test_cases:
            mock_model._detect_data_types_ml(text)
            regex_model._detect_data_types(text)

            # ML should generally have higher confidence for these cases
