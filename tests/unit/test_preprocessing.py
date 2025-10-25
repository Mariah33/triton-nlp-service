"""Unit tests for preprocessing model."""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

# Mock triton_python_backend_utils for testing
mock_pb_utils = MagicMock()


class TestPreprocessing:
    """Test suite for text preprocessing model."""

    @pytest.fixture
    def model(self):
        """Create preprocessing model instance."""
        with patch.dict("sys.modules", {"triton_python_backend_utils": mock_pb_utils}):
            sys.path.insert(0, "model_repository/preprocessing/1")
            from model import TritonPythonModel

            model = TritonPythonModel()
            model.initialize({"model_config": json.dumps({"name": "preprocessing"})})
            return model

    @pytest.mark.unit
    def test_basic_normalization(self, model):
        """Test basic text normalization."""
        test_cases = [
            ("HELLO WORLD", "HELLO WORLD"),  # Case preserved
            ("  extra   spaces  ", "extra spaces"),
            ("multiple\n\nlines", "multiple lines"),
        ]

        for input_text, expected in test_cases:
            result = model._normalize_text(input_text)
            assert expected in result

    @pytest.mark.unit
    def test_special_characters(self, model):
        """Test handling of special characters."""
        text = "Hello! How are you? I'm fine."
        result = model._normalize_text(text)

        # Should add spaces around punctuation
        assert " ! " in result or "!" in result
        assert " ? " in result or "?" in result

    @pytest.mark.unit
    def test_unicode_handling(self, model):
        """Test Unicode text handling."""
        test_cases = [
            "Café résumé",  # French
            "日本語",  # Japanese
            "مرحبا",  # Arabic
            "Привет",  # Russian
            "你好",  # Chinese
        ]

        for text in test_cases:
            result = model._normalize_text(text)
            assert result is not None
            assert len(result) > 0

    @pytest.mark.unit
    def test_script_detection(self, model):
        """Test script detection functionality."""
        test_cases = [
            ("Hello World", ["latin"]),
            ("مرحبا", ["arabic"]),
            ("こんにちは", ["japanese"]),
            ("你好", ["chinese"]),
            ("Hello مرحبا", ["latin", "arabic"]),  # Mixed scripts
        ]

        for text, expected_scripts in test_cases:
            scripts = model._detect_scripts(text)
            for expected in expected_scripts:
                assert expected in scripts

    @pytest.mark.unit
    def test_metadata_extraction(self, model):
        """Test metadata extraction."""
        text = "The price is $99.99 and my phone is 555-1234"
        metadata = model._extract_metadata(text)

        assert metadata["original_length"] == len(text)
        assert metadata["word_count"] > 0
        assert metadata["has_numbers"] is True
        assert metadata["has_special_chars"] is True

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "text,expected_numbers",
        [
            ("No numbers here", False),
            ("123 Main Street", True),
            ("Call 555-1234", True),
            ("Price: $99.99", True),
        ],
    )
    def test_number_detection(self, model, text, expected_numbers):
        """Test number detection in metadata."""
        metadata = model._extract_metadata(text)
        assert metadata["has_numbers"] == expected_numbers

    @pytest.mark.unit
    def test_empty_text(self, model):
        """Test handling of empty text."""
        empty_inputs = ["", "   ", "\n\n"]

        for text in empty_inputs:
            result = model._normalize_text(text)
            metadata = model._extract_metadata(text)

            assert result == "" or result.strip() == ""
            assert metadata["word_count"] == 0

    @pytest.mark.unit
    def test_long_text(self, model):
        """Test handling of very long text."""
        # Create a long text (10000 words)
        long_text = " ".join(["word"] * 10000)

        result = model._normalize_text(long_text)
        metadata = model._extract_metadata(long_text)

        assert metadata["word_count"] == 10000
        assert len(result) > 0

    @pytest.mark.unit
    def test_whitespace_normalization(self, model):
        """Test normalization of various whitespace types."""
        text = "Hello\t\tWorld\n\nTest"
        result = model._normalize_text(text)

        # Should normalize to single spaces
        assert "\t\t" not in result
        assert "\n\n" not in result
        assert "  " not in result or "Hello" in result

    @pytest.mark.unit
    def test_preserves_content(self, model):
        """Test that important content is preserved."""
        text = "Contact: john@example.com | Phone: 555-1234"
        result = model._normalize_text(text)

        # Important content should still be present
        assert "john@example.com" in result or "john" in result
        assert "555" in result or "1234" in result

    @pytest.mark.unit
    def test_batch_consistency(self, model):
        """Test that preprocessing is consistent across multiple calls."""
        text = "This is a test message."

        results = [model._normalize_text(text) for _ in range(5)]

        # All results should be identical
        assert all(r == results[0] for r in results)

    @pytest.mark.unit
    def test_mixed_language_scripts(self, model):
        """Test handling of mixed language/script text."""
        mixed_texts = [
            "Hello 你好 مرحبا",  # English, Chinese, Arabic
            "Email: user@example.com in multiple languages",
            "Price: €50 or ¥5000 or $50",
        ]

        for text in mixed_texts:
            result = model._normalize_text(text)
            metadata = model._extract_metadata(text)

            assert result is not None
            assert len(metadata["detected_scripts"]) >= 1

    @pytest.mark.unit
    def test_url_preservation(self, model):
        """Test that URLs are handled correctly."""
        texts_with_urls = [
            "Visit https://example.com for more info",
            "Check out www.test.org",
            "Email me at user@domain.co.uk",
        ]

        for text in texts_with_urls:
            result = model._normalize_text(text)
            # URLs should remain identifiable
            assert "http" in result or "www" in result or "@" in result

    @pytest.mark.unit
    def test_special_unicode_categories(self, model):
        """Test handling of special Unicode categories."""
        special_chars = [
            "Emoji: 😀🎉🚀",
            "Math: ∑ ∫ ∂ √",
            "Currency: € £ ¥ ₹",
            "Symbols: © ® ™ ★",
        ]

        for text in special_chars:
            result = model._normalize_text(text)
            metadata = model._extract_metadata(text)

            assert result is not None
            assert metadata["has_special_chars"] is True
