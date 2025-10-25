"""Unit tests for translation model."""

import json
from unittest.mock import MagicMock, patch

import pytest
import sys

# Mock triton_python_backend_utils for testing
mock_pb_utils = MagicMock()


class TestTranslation:
    """Test suite for language translation."""

    @pytest.fixture
    def model(self):
        """Create translation model instance."""
        with patch.dict("sys.modules", {"triton_python_backend_utils": mock_pb_utils, "torch": MagicMock()}):
            import sys

            sys.path.insert(0, "model_repository/translation/1")
            from model import TritonPythonModel

            model = TritonPythonModel()
            model.initialize({"model_config": json.dumps({"name": "translation"})})
            return model

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "text,expected_translation",
        [
            ("hello", "hola"),
            ("world", "mundo"),
            ("thank you", "gracias"),
            ("goodbye", "adiós"),
            ("yes", "sí"),
            ("no", "no"),
        ],
    )
    def test_english_to_spanish_translation(self, model, text, expected_translation):
        """Test English to Spanish translation."""
        result = model._translate(text, "en", "es")

        assert result["source_language"] == "en"
        assert result["target_language"] == "es"
        assert expected_translation in result["translated"].lower()
        assert result["confidence"] > 0.0

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "text,expected_translation",
        [
            ("hello", "bonjour"),
            ("world", "monde"),
            ("thank you", "merci"),
            ("goodbye", "au revoir"),
            ("yes", "oui"),
            ("no", "non"),
        ],
    )
    def test_english_to_french_translation(self, model, text, expected_translation):
        """Test English to French translation."""
        result = model._translate(text, "en", "fr")

        assert result["source_language"] == "en"
        assert result["target_language"] == "fr"
        assert expected_translation in result["translated"].lower()
        assert result["confidence"] > 0.0

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "text,expected_translation",
        [
            ("hello", "hallo"),
            ("world", "welt"),
            ("thank you", "danke"),
            ("goodbye", "auf wiedersehen"),
            ("yes", "ja"),
            ("no", "nein"),
        ],
    )
    def test_english_to_german_translation(self, model, text, expected_translation):
        """Test English to German translation."""
        result = model._translate(text, "en", "de")

        assert result["source_language"] == "en"
        assert result["target_language"] == "de"
        assert expected_translation in result["translated"].lower()
        assert result["confidence"] > 0.0

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "text,expected_contains",
        [
            ("hello", "नमस्ते"),
            ("world", "विश्व"),
            ("thank you", "धन्यवाद"),
            ("yes", "हाँ"),
            ("no", "नहीं"),
        ],
    )
    def test_english_to_hindi_translation(self, model, text, expected_contains):
        """Test English to Hindi translation."""
        result = model._translate(text, "en", "hi")

        assert result["source_language"] == "en"
        assert result["target_language"] == "hi"
        assert expected_contains in result["translated"]
        assert result["confidence"] > 0.0

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "text,expected_contains",
        [
            ("hello", "مرحبا"),
            ("world", "عالم"),
            ("thank you", "شكرا"),
            ("yes", "نعم"),
            ("no", "لا"),
        ],
    )
    def test_english_to_arabic_translation(self, model, text, expected_contains):
        """Test English to Arabic translation."""
        result = model._translate(text, "en", "ar")

        assert result["source_language"] == "en"
        assert result["target_language"] == "ar"
        assert expected_contains in result["translated"]
        assert result["confidence"] > 0.0

    @pytest.mark.unit
    def test_same_language_no_translation(self, model):
        """Test that same source and target languages return original text."""
        text = "Hello world"
        result = model._translate(text, "en", "en")

        assert result["source_language"] == "en"
        assert result["target_language"] == "en"
        assert result["translated"] == text
        assert result["confidence"] == 1.0
        assert result["method"] == "no_translation_needed"

    @pytest.mark.unit
    def test_auto_detect_english(self, model):
        """Test auto-detection of English text."""
        text = "The quick brown fox jumps"
        detected = model._detect_language(text)

        assert detected == "en"

    @pytest.mark.unit
    def test_auto_detect_french(self, model):
        """Test auto-detection of French text."""
        text = "Le chat est noir"
        detected = model._detect_language(text)

        assert detected == "fr"

    @pytest.mark.unit
    def test_auto_detect_spanish(self, model):
        """Test auto-detection of Spanish text."""
        text = "El perro es grande"
        detected = model._detect_language(text)

        assert detected == "es"

    @pytest.mark.unit
    def test_auto_detect_german(self, model):
        """Test auto-detection of German text."""
        text = "Der Hund ist groß"
        detected = model._detect_language(text)

        assert detected == "de"

    @pytest.mark.unit
    def test_auto_detect_russian(self, model):
        """Test auto-detection of Russian text."""
        text = "Привет мир"
        detected = model._detect_language(text)

        assert detected == "ru"

    @pytest.mark.unit
    def test_auto_detect_arabic(self, model):
        """Test auto-detection of Arabic text."""
        text = "مرحبا بالعالم"
        detected = model._detect_language(text)

        assert detected == "ar"

    @pytest.mark.unit
    def test_auto_detect_hindi(self, model):
        """Test auto-detection of Hindi text."""
        text = "नमस्ते दुनिया"
        detected = model._detect_language(text)

        assert detected == "hi"

    @pytest.mark.unit
    def test_auto_detect_chinese(self, model):
        """Test auto-detection of Chinese text."""
        text = "你好世界"
        detected = model._detect_language(text)

        assert detected == "zh"

    @pytest.mark.unit
    def test_auto_detect_japanese(self, model):
        """Test auto-detection of Japanese text."""
        text = "こんにちは世界"
        detected = model._detect_language(text)

        assert detected == "ja"

    @pytest.mark.unit
    def test_translation_with_auto_detect(self, model):
        """Test translation with automatic language detection."""
        text = "hello"
        result = model._translate(text, "auto", "es")

        # Should auto-detect as English
        assert result["source_language"] == "en"
        assert result["target_language"] == "es"
        assert "hola" in result["translated"].lower()

    @pytest.mark.unit
    def test_unsupported_language_pair(self, model):
        """Test handling of unsupported language pairs."""
        text = "hello"
        result = model._translate(text, "en", "pt")  # English to Portuguese not in simple_translations

        # Should still attempt translation or return info
        assert result["source_language"] == "en"
        assert result["target_language"] == "pt"
        # Confidence might be low or method might be unsupported
        assert result["method"] in ["unsupported", "unsupported_pair", "neural_mt"]

    @pytest.mark.unit
    def test_empty_text_translation(self, model):
        """Test handling of empty text."""
        result = model._translate("", "en", "es")

        assert result["original"] == ""
        # Empty text should be handled gracefully
        assert "translated" in result

    @pytest.mark.unit
    def test_word_by_word_translation(self, model):
        """Test word-by-word translation for phrase."""
        text = "hello world"
        result = model._translate(text, "en", "es")

        # Should translate both words
        translated = result["translated"].lower()
        # Either as phrase or word-by-word
        assert "hola" in translated or "mundo" in translated

    @pytest.mark.unit
    def test_confidence_scores_validity(self, model):
        """Test that confidence scores are in valid range."""
        test_cases = [
            ("hello", "en", "es"),
            ("hello", "en", "fr"),
            ("hello", "en", "de"),
            ("hello", "en", "hi"),
            ("hello", "en", "ar"),
        ]

        for text, source, target in test_cases:
            result = model._translate(text, source, target)

            assert 0.0 <= result["confidence"] <= 1.0
            assert isinstance(result["confidence"], (int, float))

    @pytest.mark.unit
    def test_result_structure(self, model):
        """Test that result has expected structure."""
        text = "hello"
        result = model._translate(text, "en", "es")

        # Check all expected fields
        assert "original" in result
        assert "translated" in result
        assert "source_language" in result
        assert "target_language" in result
        assert "confidence" in result
        assert "method" in result
        assert "alternative_translations" in result

        # Check types
        assert isinstance(result["original"], str)
        assert isinstance(result["translated"], str)
        assert isinstance(result["alternative_translations"], list)

    @pytest.mark.unit
    def test_alternative_translations_english_spanish(self, model):
        """Test alternative translations for English to Spanish."""
        text = "hello"
        alternatives = model._get_alternatives(text, "en-es")

        # Should return list of alternatives
        assert isinstance(alternatives, list)
        # For "hello", should include alternatives like "hola", "buenos días"
        if alternatives:
            assert any("hola" in alt.lower() for alt in alternatives)

    @pytest.mark.unit
    def test_alternative_translations_english_french(self, model):
        """Test alternative translations for English to French."""
        text = "hello"
        alternatives = model._get_alternatives(text, "en-fr")

        assert isinstance(alternatives, list)
        # For "hello", should include alternatives like "bonjour", "salut"
        if alternatives:
            assert any("bonjour" in alt.lower() for alt in alternatives)

    @pytest.mark.unit
    def test_case_insensitive_translation(self, model):
        """Test that translation works regardless of case."""
        test_cases = [
            ("HELLO", "en", "es"),
            ("Hello", "en", "es"),
            ("hello", "en", "es"),
        ]

        for text, source, target in test_cases:
            result = model._translate(text, source, target)

            # Should translate regardless of case
            assert "hola" in result["translated"].lower()

    @pytest.mark.unit
    def test_punctuation_preservation(self, model):
        """Test that punctuation is handled correctly."""
        text = "hello!"
        result = model._translate(text, "en", "es")

        # Translation should occur for the word
        assert "hola" in result["translated"].lower() or "hello" in result["translated"].lower()

    @pytest.mark.unit
    def test_numbers_preservation(self, model):
        """Test that numbers are preserved in translation."""
        text = "I have 5 apples"
        result = model._translate(text, "en", "es")

        # Numbers should be preserved
        assert "5" in result["translated"]

    @pytest.mark.unit
    def test_mixed_case_phrase(self, model):
        """Test translation of mixed case phrases."""
        text = "Thank You"
        result = model._translate(text, "en", "es")

        # Should translate despite mixed case
        assert result["translated"] is not None
        assert len(result["translated"]) > 0

    @pytest.mark.unit
    def test_unknown_word_handling(self, model):
        """Test handling of unknown words in translation."""
        text = "xyzabc"  # Non-existent word
        result = model._translate(text, "en", "es")

        # Should handle gracefully
        assert result["translated"] is not None
        # Unknown words typically remain unchanged
        assert "xyzabc" in result["translated"] or "[Translation" in result["translated"]

    @pytest.mark.unit
    def test_method_field_accuracy(self, model):
        """Test that method field accurately reflects translation approach."""
        # Dictionary translation
        result1 = model._translate("hello", "en", "es")
        assert result1["method"] in ["dictionary", "neural_mt", "unsupported"]

        # No translation needed
        result2 = model._translate("hello", "en", "en")
        assert result2["method"] == "no_translation_needed"

    @pytest.mark.unit
    def test_original_text_preservation(self, model):
        """Test that original text is preserved in result."""
        text = "hello world"
        result = model._translate(text, "en", "es")

        # Original should match input exactly
        assert result["original"] == text

    @pytest.mark.unit
    def test_multiple_words_translation(self, model):
        """Test translation of multiple words."""
        text = "hello world thank you"
        result = model._translate(text, "en", "es")

        # Should attempt to translate all words
        assert result["translated"] is not None
        assert len(result["translated"]) > 0

    @pytest.mark.unit
    def test_special_characters_handling(self, model):
        """Test handling of special characters."""
        text = "hello @world #test"
        result = model._translate(text, "en", "es")

        # Special characters should be handled
        assert result["translated"] is not None

    @pytest.mark.unit
    def test_long_text_translation(self, model):
        """Test translation of longer text."""
        text = " ".join(["hello"] * 10)
        result = model._translate(text, "en", "es")

        assert result["translated"] is not None
        assert len(result["translated"]) > 0

    @pytest.mark.unit
    def test_unicode_text_translation(self, model):
        """Test translation with Unicode characters."""
        # English to Hindi already uses Unicode
        text = "hello"
        result = model._translate(text, "en", "hi")

        # Should handle Unicode properly
        assert result["translated"] is not None
        # Hindi uses Devanagari script
        assert any(ord(c) > 127 for c in result["translated"])

    @pytest.mark.unit
    def test_default_language_fallback(self, model):
        """Test that unknown text defaults to English."""
        text = "xyzabc123"  # Ambiguous text
        detected = model._detect_language(text)

        # Should default to English when uncertain
        assert detected == "en"

    @pytest.mark.unit
    def test_language_pairs_existence(self, model):
        """Test that language pairs are properly defined."""
        # Check that language_pairs dictionary exists
        assert hasattr(model, "language_pairs")
        assert isinstance(model.language_pairs, dict)

        # Check for expected pairs
        expected_pairs = ["en-es", "en-fr", "en-de", "en-hi", "en-ar"]
        for pair in expected_pairs:
            assert pair in model.language_pairs

    @pytest.mark.unit
    def test_simple_translations_existence(self, model):
        """Test that simple translations dictionary exists."""
        assert hasattr(model, "simple_translations")
        assert isinstance(model.simple_translations, dict)

        # Check for expected language pairs
        expected_pairs = ["en-es", "en-fr", "en-de", "en-hi", "en-ar"]
        for pair in expected_pairs:
            assert pair in model.simple_translations
            assert isinstance(model.simple_translations[pair], dict)

    @pytest.mark.unit
    def test_whitespace_handling(self, model):
        """Test handling of extra whitespace."""
        text = "hello  world"  # Extra space
        result = model._translate(text, "en", "es")

        # Should handle extra whitespace
        assert result["translated"] is not None

    @pytest.mark.unit
    def test_supported_language_pair_detection(self, model):
        """Test detection of supported language pairs."""
        # Supported pair
        result1 = model._translate("hello", "en", "es")
        assert result1["method"] in ["dictionary", "neural_mt"]
        assert result1["confidence"] > 0.0

        # Potentially unsupported pair
        result2 = model._translate("hello", "ja", "ko")
        # Method might be unsupported
        assert result2["method"] in ["unsupported", "unsupported_pair", "neural_mt"]

    @pytest.mark.unit
    def test_rtl_language_support(self, model):
        """Test support for right-to-left languages like Arabic."""
        text = "hello"
        result = model._translate(text, "en", "ar")

        # Arabic is RTL
        assert result["target_language"] == "ar"
        # Should contain Arabic characters
        arabic_chars = set("مرحباعالمشكروداعنعملافضآسفكيفحل")
        assert any(c in arabic_chars for c in result["translated"])

    @pytest.mark.benchmark
    def test_translation_performance(self, model, benchmark):
        """Benchmark translation performance."""
        text = "hello world"

        result = benchmark(model._translate, text, "en", "es")

        assert result["translated"] is not None
        assert result["confidence"] >= 0.0
