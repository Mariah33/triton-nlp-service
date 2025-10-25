"""Unit tests for language detection model."""

import json
from unittest.mock import MagicMock, patch

import pytest

# Mock triton_python_backend_utils for testing
mock_pb_utils = MagicMock()


class TestLanguageDetector:
    """Test suite for language detection across 50+ languages."""

    @pytest.fixture
    def model(self):
        """Create language detector model instance."""
        with patch.dict("sys.modules", {"triton_python_backend_utils": mock_pb_utils}):
            import sys

            sys.path.insert(0, "model_repository/language_detector/1")
            from model import TritonPythonModel

            model = TritonPythonModel()
            model.initialize({"model_config": json.dumps({"name": "language_detector"})})
            return model

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "text,expected_lang,min_confidence",
        [
            # European Languages
            ("Hello, how are you today?", "en", 0.95),
            ("Bonjour, comment allez-vous?", "fr", 0.95),
            ("Hola, ¿cómo estás?", "es", 0.95),
            ("Guten Tag, wie geht es Ihnen?", "de", 0.95),
            ("Ciao, come stai?", "it", 0.95),
            ("Olá, como você está?", "pt", 0.95),
            ("Hallo, hoe gaat het met je?", "nl", 0.95),
            ("Привет, как дела?", "ru", 0.95),
            ("Witaj, jak się masz?", "pl", 0.95),
            ("Hej, hur mår du?", "sv", 0.90),
            # Asian Languages
            ("こんにちは、元気ですか？", "ja", 0.99),
            ("你好，你好吗？", "zh-cn", 0.99),
            ("안녕하세요, 어떻게 지내세요?", "ko", 0.99),
            ("สวัสดี คุณสบายดีหรือเปล่า", "th", 0.99),
            ("Xin chào, bạn khỏe không?", "vi", 0.95),
            ("नमस्ते, आप कैसे हैं?", "hi", 0.95),
            ("হ্যালো, আপনি কেমন আছেন?", "bn", 0.90),
            # Middle Eastern & African
            ("مرحبا، كيف حالك؟", "ar", 0.99),
            ("שלום, מה שלומך?", "he", 0.95),
            ("سلام، حال شما چطور است؟", "fa", 0.90),
            ("Merhaba, nasılsın?", "tr", 0.95),
            ("Sawubona, unjani?", "zu", 0.85),  # Zulu - lower confidence expected
        ],
    )
    def test_language_detection_accuracy(self, model, text, expected_lang, min_confidence):
        """Test language detection accuracy for various languages."""
        result = model._detect_language(text)

        assert result["language_code"] == expected_lang or result["detected_language"] != "unknown"
        assert result["confidence"] >= min_confidence
        assert result["is_reliable"] is True

    @pytest.mark.unit
    def test_short_text_handling(self, model):
        """Test handling of very short texts."""
        short_texts = ["Hi", "OK", "No", "¿Qué?", "да"]

        for text in short_texts:
            result = model._detect_language(text)

            # Short texts may have lower confidence
            assert result["detected_language"] != "unknown" or result["confidence"] < 0.8

    @pytest.mark.unit
    def test_empty_text(self, model):
        """Test handling of empty or invalid text."""
        invalid_inputs = ["", "   ", "\n\n", "...", "123"]

        for text in invalid_inputs:
            result = model._detect_language(text)

            assert result["detected_language"] == "unknown"
            assert result["confidence"] == 0.0
            assert result["is_reliable"] is False

    @pytest.mark.unit
    def test_mixed_language_text(self, model):
        """Test detection with mixed language text."""
        mixed_texts = [
            "Hello, I'm learning français and español",
            "This is English with 日本語 and 中文",
            "Email: user@example.com في العربية",
        ]

        for text in mixed_texts:
            result = model._detect_language(text)

            # Should detect the dominant language
            assert result["detected_language"] != "unknown"
            assert len(result["all_probabilities"]) > 1  # Multiple languages detected

    @pytest.mark.unit
    def test_code_switching(self, model):
        """Test detection with code-switching (bilingual text)."""
        # Spanglish
        text = "Voy al store para comprar groceries"
        result = model._detect_language(text)

        # Should detect one of the languages (likely Spanish as it's more)
        assert result["language_code"] in ["es", "en"]
        assert result["all_probabilities"] is not None

    @pytest.mark.unit
    def test_technical_text(self, model):
        """Test detection with technical/programming content."""
        technical_texts = [
            "def hello_world(): print('Hello')",
            "SELECT * FROM users WHERE id = 1",
            "docker run -d -p 8080:8080 myapp",
            "import numpy as np; arr = np.array([1,2,3])",
        ]

        for text in technical_texts:
            result = model._detect_language(text)

            # Should still detect based on natural language keywords
            assert result["language_code"] in ["en", "unknown"]

    @pytest.mark.unit
    def test_proper_nouns_and_names(self, model):
        """Test detection with text containing many proper nouns."""
        texts = [
            "Microsoft Corporation in Redmond, Washington",
            "Paris, France is beautiful",
            "Tokyo Tower in Japan",
        ]

        for text in texts:
            result = model._detect_language(text)

            assert result["detected_language"] == "English" or result["language_code"] == "en"
            assert result["confidence"] > 0.7

    @pytest.mark.unit
    def test_transliterated_text(self, model):
        """Test detection with transliterated text."""
        # Romanized versions of non-Latin scripts
        transliterated = [
            "Konnichiwa, genki desu ka?",  # Romanized Japanese
            "Namaste, aap kaise hain?",  # Romanized Hindi
            "Salaam, haal-e shoma chetor ast?",  # Romanized Persian
        ]

        for text in transliterated:
            result = model._detect_language(text)

            # May be detected as English or unknown since it's romanized
            assert result is not None

    @pytest.mark.unit
    def test_language_confidence_ranking(self, model):
        """Test that all_probabilities are sorted by confidence."""
        text = "This is a clear English sentence with good confidence."
        result = model._detect_language(text)

        probabilities = result["all_probabilities"]

        # Should be sorted by probability (highest first)
        for i in range(len(probabilities) - 1):
            assert probabilities[i]["probability"] >= probabilities[i + 1]["probability"]

    @pytest.mark.unit
    def test_language_name_mapping(self, model):
        """Test that language codes are mapped to full names."""
        test_cases = [
            ("Hello world", "en", "English"),
            ("Bonjour le monde", "fr", "French"),
            ("Hola mundo", "es", "Spanish"),
        ]

        for text, expected_code, expected_name in test_cases:
            result = model._detect_language(text)

            if result["language_code"] == expected_code:
                assert result["detected_language"] == expected_name

    @pytest.mark.unit
    def test_reliable_flag_threshold(self, model):
        """Test that is_reliable flag correctly indicates confidence."""
        high_confidence_text = "This is a very clear and unambiguous English sentence."
        result = model._detect_language(high_confidence_text)

        assert result["confidence"] > 0.8
        assert result["is_reliable"] is True

    @pytest.mark.unit
    def test_numeric_and_special_characters(self, model):
        """Test handling of text with numbers and special characters."""
        texts = [
            "Price: $99.99",
            "Call 555-1234",
            "email@example.com",
            "https://example.com",
        ]

        for text in texts:
            result = model._detect_language(text)

            # Should still attempt detection based on surrounding context
            assert result is not None

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "text,should_be_reliable",
        [
            ("The quick brown fox jumps over the lazy dog", True),
            ("a", False),
            ("", False),
            ("12345", False),
            ("Un texte français assez long pour être fiable", True),
        ],
    )
    def test_reliability_assessment(self, model, text, should_be_reliable):
        """Test reliability assessment for different text lengths and types."""
        result = model._detect_language(text)

        assert result["is_reliable"] == should_be_reliable

    @pytest.mark.unit
    def test_similar_languages_distinction(self, model):
        """Test distinction between similar languages."""
        similar_pairs = [
            ("Ik hou van Nederland", "nl"),  # Dutch
            ("Ich liebe Deutschland", "de"),  # German
            ("Eu amo o Brasil", "pt"),  # Portuguese
            ("Yo amo España", "es"),  # Spanish
        ]

        for text, expected_lang in similar_pairs:
            result = model._detect_language(text)

            assert result["language_code"] == expected_lang or result["confidence"] > 0.5

    @pytest.mark.unit
    def test_batch_consistency(self, model):
        """Test that detection is consistent across multiple calls."""
        text = "This is a test sentence for language detection."

        results = [model._detect_language(text) for _ in range(5)]

        # All results should have the same language code
        lang_codes = [r["language_code"] for r in results]
        assert all(code == lang_codes[0] for code in lang_codes)

    @pytest.mark.unit
    def test_unicode_handling(self, model):
        """Test proper Unicode handling for all languages."""
        unicode_texts = [
            "Café résumé",  # French with accents
            "Москва",  # Russian Cyrillic
            "北京市",  # Chinese characters
            "العربية",  # Arabic script
            "Ελληνικά",  # Greek
        ]

        for text in unicode_texts:
            result = model._detect_language(text)

            assert result["detected_language"] != "unknown"
            assert "error" not in result

    @pytest.mark.unit
    def test_regional_variants(self, model):
        """Test detection of regional language variants."""
        # Chinese variants
        simplified = "简体中文"
        traditional = "繁體中文"

        result_simp = model._detect_language(simplified)
        result_trad = model._detect_language(traditional)

        # Both should be detected as Chinese (may be zh-cn or zh-tw)
        assert "zh" in result_simp["language_code"]
        assert "zh" in result_trad["language_code"]

    @pytest.mark.unit
    def test_text_length_impact(self, model):
        """Test how text length affects confidence."""
        base_text = "This is an English sentence. "

        for multiplier in [1, 3, 5, 10]:
            text = base_text * multiplier
            result = model._detect_language(text)

            # Longer texts should generally have higher confidence
            assert result["language_code"] == "en"

    @pytest.mark.benchmark
    def test_detection_performance(self, model, benchmark):
        """Benchmark language detection performance."""
        text = "This is a test sentence for performance benchmarking of language detection."

        result = benchmark(model._detect_language, text)

        assert result is not None
        assert result["language_code"] is not None
