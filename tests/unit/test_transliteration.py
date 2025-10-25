"""Unit tests for transliteration model."""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

# Mock triton_python_backend_utils for testing
mock_pb_utils = MagicMock()


class TestTransliteration:
    """Test suite for script transliteration."""

    @pytest.fixture
    def model(self):
        """Create transliteration model instance."""
        with patch.dict("sys.modules", {"triton_python_backend_utils": mock_pb_utils}):
            sys.path.insert(0, "model_repository/transliteration/1")
            from model import TritonPythonModel

            model = TritonPythonModel()
            model.initialize({"model_config": json.dumps({"name": "transliteration"})})
            return model

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "text,script",
        [
            ("नमस्ते", "devanagari"),  # Hindi "Namaste"
            ("स्वागत", "devanagari"),  # Hindi "Welcome"
            ("भारत", "devanagari"),  # Hindi "India"
        ],
    )
    def test_devanagari_to_latin(self, model, text, script):
        """Test Devanagari to Latin transliteration."""
        result = model._transliterate(text, script, "latin")

        assert result["source_script"] == "devanagari"
        assert result["target_script"] == "latin"
        assert result["transliterated"] != text  # Should be different from original
        assert result["confidence"] > 0.7
        assert result["method"] == "rule_based"

        # Transliterated text should be in Latin script
        assert all(ord(c) < 128 or c.isspace() for c in result["transliterated"])

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "text,expected_contains",
        [
            ("مرحبا", "m"),  # Arabic "Hello" contains 'm'
            ("السلام", "s"),  # Arabic "Peace" contains 's'
            ("شكرا", "sh"),  # Arabic "Thanks" contains 'sh'
        ],
    )
    def test_arabic_to_latin(self, model, text, expected_contains):
        """Test Arabic to Latin transliteration."""
        result = model._transliterate(text, "arabic", "latin")

        assert result["source_script"] == "arabic"
        assert result["target_script"] == "latin"
        assert result["confidence"] >= 0.7
        assert result["method"] == "rule_based"

        # Check that expected characters appear in transliteration
        assert expected_contains.lower() in result["transliterated"].lower()

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "text,expected_contains",
        [
            ("Привет", "Privet"),  # Russian "Hello"
            ("Москва", "Moskva"),  # Russian "Moscow"
            ("Спасибо", "Spasibo"),  # Russian "Thank you"
        ],
    )
    def test_cyrillic_to_latin(self, model, text, expected_contains):
        """Test Cyrillic to Latin transliteration."""
        result = model._transliterate(text, "cyrillic", "latin")

        assert result["source_script"] == "cyrillic"
        assert result["target_script"] == "latin"
        assert result["confidence"] >= 0.85
        assert result["method"] == "rule_based"

        # Check expected transliteration
        assert expected_contains.lower() in result["transliterated"].lower()

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "text,expected_contains",
        [
            ("Αθήνα", "Ath"),  # Greek "Athens"
            ("Ελλάδα", "Ell"),  # Greek "Greece"
            ("Φιλοσοφία", "Ph"),  # Greek "Philosophy"
        ],
    )
    def test_greek_to_latin(self, model, text, expected_contains):
        """Test Greek to Latin transliteration."""
        result = model._transliterate(text, "greek", "latin")

        assert result["source_script"] == "greek"
        assert result["target_script"] == "latin"
        assert result["confidence"] >= 0.8
        assert result["method"] == "rule_based"

        # Check expected characters appear
        assert expected_contains in result["transliterated"]

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "text,expected_script",
        [
            ("Hello World", "latin"),
            ("नमस्ते", "devanagari"),
            ("مرحبا", "arabic"),
            ("Привет", "cyrillic"),
            ("Ελλάδα", "greek"),
            ("你好", "chinese"),
            ("こんにちは", "japanese"),
        ],
    )
    def test_script_detection(self, model, text, expected_script):
        """Test automatic script detection."""
        detected = model._detect_script(text)

        assert detected == expected_script

    @pytest.mark.unit
    def test_auto_detect_devanagari(self, model):
        """Test auto-detection with Devanagari text."""
        text = "नमस्ते दुनिया"
        result = model._transliterate(text, "auto", "latin")

        assert result["source_script"] == "devanagari"
        assert result["target_script"] == "latin"
        assert result["confidence"] > 0.7

    @pytest.mark.unit
    def test_auto_detect_arabic(self, model):
        """Test auto-detection with Arabic text."""
        text = "مرحبا بالعالم"
        result = model._transliterate(text, "auto", "latin")

        assert result["source_script"] == "arabic"
        assert result["target_script"] == "latin"
        assert result["confidence"] > 0.7

    @pytest.mark.unit
    def test_auto_detect_cyrillic(self, model):
        """Test auto-detection with Cyrillic text."""
        text = "Привет мир"
        result = model._transliterate(text, "auto", "latin")

        assert result["source_script"] == "cyrillic"
        assert result["target_script"] == "latin"
        assert result["confidence"] >= 0.8

    @pytest.mark.unit
    def test_same_script_no_conversion(self, model):
        """Test that same source and target scripts return original text."""
        text = "Hello World"
        result = model._transliterate(text, "latin", "latin")

        assert result["source_script"] == "latin"
        assert result["target_script"] == "latin"
        assert result["transliterated"] == text
        assert result["confidence"] == 1.0

    @pytest.mark.unit
    def test_unsupported_script_pair(self, model):
        """Test handling of unsupported script pairs."""
        text = "Hello"
        result = model._transliterate(text, "latin", "devanagari")

        # Should return original text for unsupported pairs
        assert result["confidence"] == 0.0
        assert result["method"] == "unsupported"

    @pytest.mark.unit
    def test_empty_text(self, model):
        """Test handling of empty text."""
        result = model._transliterate("", "devanagari", "latin")

        assert result["original"] == ""
        assert result["transliterated"] == ""

    @pytest.mark.unit
    def test_mixed_script_text(self, model):
        """Test transliteration with mixed scripts."""
        text = "Hello नमस्ते"  # English + Hindi
        result = model._transliterate(text, "auto", "latin")

        # Should detect dominant script (likely devanagari) and transliterate
        assert result["transliterated"] is not None
        # English portion should remain unchanged
        assert "Hello" in result["transliterated"]

    @pytest.mark.unit
    def test_numbers_preservation(self, model):
        """Test that numbers are preserved during transliteration."""
        # Devanagari numbers
        text = "०१२३४५६७८९"
        result = model._transliterate(text, "devanagari", "latin")

        # Should transliterate to Latin digits
        assert result["transliterated"] == "0123456789"

    @pytest.mark.unit
    def test_arabic_numbers_preservation(self, model):
        """Test Arabic-Indic digits transliteration."""
        text = "٠١٢٣٤٥٦٧٨٩"
        result = model._transliterate(text, "arabic", "latin")

        # Should transliterate to Latin digits
        assert result["transliterated"] == "0123456789"

    @pytest.mark.unit
    def test_whitespace_preservation(self, model):
        """Test that whitespace is preserved."""
        text = "नमस्ते   दुनिया"  # Multiple spaces
        result = model._transliterate(text, "devanagari", "latin")

        # Whitespace should be preserved
        assert "   " in result["transliterated"]

    @pytest.mark.unit
    def test_special_characters_preservation(self, model):
        """Test that special characters are preserved."""
        text = "नमस्ते! दुनिया?"
        result = model._transliterate(text, "devanagari", "latin")

        # Special characters should remain
        assert "!" in result["transliterated"]
        assert "?" in result["transliterated"]

    @pytest.mark.unit
    def test_result_structure(self, model):
        """Test that result has expected structure."""
        text = "नमस्ते"
        result = model._transliterate(text, "devanagari", "latin")

        # Check all expected fields
        assert "original" in result
        assert "transliterated" in result
        assert "source_script" in result
        assert "target_script" in result
        assert "confidence" in result
        assert "method" in result

        # Check types
        assert isinstance(result["original"], str)
        assert isinstance(result["transliterated"], str)
        assert isinstance(result["confidence"], (int, float))
        assert 0.0 <= result["confidence"] <= 1.0

    @pytest.mark.unit
    def test_confidence_scores_validity(self, model):
        """Test that confidence scores are in valid range."""
        test_cases = [
            ("नमस्ते", "devanagari", "latin"),
            ("مرحبا", "arabic", "latin"),
            ("Привет", "cyrillic", "latin"),
            ("Ελλάδα", "greek", "latin"),
        ]

        for text, source, target in test_cases:
            result = model._transliterate(text, source, target)

            assert 0.0 <= result["confidence"] <= 1.0
            assert isinstance(result["confidence"], (int, float))

    @pytest.mark.unit
    def test_character_mapping_accuracy(self, model):
        """Test accuracy of character mappings."""
        # Test specific character mappings
        test_mappings = [
            ("क", "devanagari", "k"),  # Devanagari 'ka'
            ("ख", "devanagari", "kh"),  # Devanagari 'kha'
            ("ا", "arabic", "a"),  # Arabic 'alif'  # noqa: RUF001
            ("А", "cyrillic", "A"),  # Cyrillic 'A'  # noqa: RUF001
            ("Α", "greek", "A"),  # Greek 'Alpha'  # noqa: RUF001
        ]

        for char, script, expected in test_mappings:
            result = model._transliterate(char, script, "latin")
            assert expected in result["transliterated"], f"Failed for {char}: expected '{expected}' in '{result['transliterated']}'"

    @pytest.mark.unit
    def test_long_text_transliteration(self, model):
        """Test transliteration of longer text."""
        # Create longer text
        text = "नमस्ते " * 10  # Repeat 10 times
        result = model._transliterate(text, "devanagari", "latin")

        assert result["transliterated"] is not None
        assert len(result["transliterated"]) > 0
        # Latin text should generally be longer or same length
        assert len(result["transliterated"]) >= len(text.replace(" ", ""))

    @pytest.mark.unit
    def test_case_preservation_cyrillic(self, model):
        """Test that uppercase/lowercase is preserved for Cyrillic."""
        # Cyrillic has uppercase and lowercase
        text = "АБВ абв"  # Uppercase and lowercase
        result = model._transliterate(text, "cyrillic", "latin")

        # Result should have both upper and lowercase
        assert any(c.isupper() for c in result["transliterated"])
        assert any(c.islower() for c in result["transliterated"])

    @pytest.mark.unit
    def test_case_preservation_greek(self, model):
        """Test that uppercase/lowercase is preserved for Greek."""
        text = "ΑΒΓ αβγ"  # Uppercase and lowercase
        result = model._transliterate(text, "greek", "latin")

        # Result should have both upper and lowercase
        assert any(c.isupper() for c in result["transliterated"])
        assert any(c.islower() for c in result["transliterated"])

    @pytest.mark.unit
    def test_unknown_characters_preservation(self, model):
        """Test that unknown characters are preserved."""
        # Text with emoji or other Unicode characters
        text = "नमस्ते 😊"
        result = model._transliterate(text, "devanagari", "latin")

        # Emoji should be preserved
        assert "😊" in result["transliterated"]

    @pytest.mark.unit
    def test_multiple_words_devanagari(self, model):
        """Test transliteration of multiple Devanagari words."""
        text = "नमस्ते दुनिया कैसे हैं"
        result = model._transliterate(text, "devanagari", "latin")

        assert result["confidence"] > 0.7
        # Should have spaces between words
        assert " " in result["transliterated"]
        # Result should be longer than just spaces
        assert len(result["transliterated"].strip()) > 0

    @pytest.mark.unit
    def test_punctuation_preservation(self, model):
        """Test preservation of various punctuation marks."""
        text = "नमस्ते, दुनिया! कैसे हैं?"
        result = model._transliterate(text, "devanagari", "latin")

        # All punctuation should be preserved
        assert "," in result["transliterated"]
        assert "!" in result["transliterated"]
        assert "?" in result["transliterated"]

    @pytest.mark.unit
    def test_script_detection_mixed_content(self, model):
        """Test script detection with mixed content."""
        # Text that is mostly Devanagari with some Latin
        text = "नमस्ते world"
        detected = model._detect_script(text)

        # Should detect the dominant script
        assert detected in ["devanagari", "latin"]

    @pytest.mark.unit
    def test_original_text_preservation(self, model):
        """Test that original text is preserved in result."""
        text = "नमस्ते दुनिया"
        result = model._transliterate(text, "devanagari", "latin")

        # Original should match input exactly
        assert result["original"] == text

    @pytest.mark.unit
    def test_method_field_accuracy(self, model):
        """Test that method field correctly indicates the approach used."""
        # Supported pair
        result1 = model._transliterate("नमस्ते", "devanagari", "latin")
        assert result1["method"] == "rule_based"

        # Unsupported pair
        result2 = model._transliterate("Hello", "latin", "devanagari")
        assert result2["method"] == "unsupported"

        # Same script
        result3 = model._transliterate("Hello", "latin", "latin")
        assert result3["method"] == "rule_based"

    @pytest.mark.unit
    def test_multi_character_mappings(self, model):
        """Test that multi-character mappings work correctly."""
        # Test Cyrillic characters that map to multiple Latin characters
        test_cases = [
            ("Ё", "cyrillic", "Yo"),
            ("Ж", "cyrillic", "Zh"),
            ("Ч", "cyrillic", "Ch"),
            ("Ш", "cyrillic", "Sh"),
            ("Щ", "cyrillic", "Shch"),
            ("Ю", "cyrillic", "Yu"),
            ("Я", "cyrillic", "Ya"),
        ]

        for char, script, expected in test_cases:
            result = model._transliterate(char, script, "latin")
            assert expected.lower() in result["transliterated"].lower(), f"Failed for {char}: expected '{expected}' in '{result['transliterated']}'"

    @pytest.mark.benchmark
    def test_transliteration_performance(self, model, benchmark):
        """Benchmark transliteration performance."""
        text = "नमस्ते दुनिया कैसे हैं आप"

        result = benchmark(model._transliterate, text, "devanagari", "latin")

        assert result["transliterated"] is not None
        assert result["confidence"] > 0.7
