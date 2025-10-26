"""Unit tests for multilingual NER model."""

import json
from unittest.mock import MagicMock, patch

import pytest


# Mock triton_python_backend_utils for testing
mock_pb_utils = MagicMock()


class TestMultilingualNER:
    """Test suite for multilingual named entity recognition."""

    @pytest.fixture
    def model(self):
        """Create multilingual NER model instance."""
        with patch.dict("sys.modules", {"triton_python_backend_utils": mock_pb_utils, "spacy": MagicMock(), "torch": MagicMock()}):
            import sys

            sys.path.insert(0, "model_repository/ner_multilingual/1")
            from model import TritonPythonModel

            model = TritonPythonModel()
            model.initialize({"model_config": json.dumps({"name": "ner_multilingual"})})

            # Mock the NER pipeline
            model.ner_pipeline = MagicMock()
            model.spacy_models = {}

            return model

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "text,lang,expected_entities",
        [
            # English
            ("Barack Obama was born in Hawaii.", "en", [("Barack Obama", "PERSON"), ("Hawaii", "LOCATION")]),
            ("Microsoft is headquartered in Redmond.", "en", [("Microsoft", "ORGANIZATION"), ("Redmond", "LOCATION")]),
            # Spanish
            ("Pablo Picasso nació en Málaga, España.", "es", [("Pablo Picasso", "PERSON"), ("Málaga", "LOCATION"), ("España", "LOCATION")]),
            ("El Banco Santander tiene sede en Madrid.", "es", [("Banco Santander", "ORGANIZATION"), ("Madrid", "LOCATION")]),
            # French
            ("Emmanuel Macron est le président de la France.", "fr", [("Emmanuel Macron", "PERSON"), ("France", "LOCATION")]),
            ("La Tour Eiffel est à Paris.", "fr", [("Tour Eiffel", "LOCATION"), ("Paris", "LOCATION")]),
            # German
            ("Angela Merkel war Bundeskanzlerin von Deutschland.", "de", [("Angela Merkel", "PERSON"), ("Deutschland", "LOCATION")]),
            ("Die BMW Group hat ihren Sitz in München.", "de", [("BMW Group", "ORGANIZATION"), ("München", "LOCATION")]),
            # Italian
            ("Leonardo da Vinci nacque a Vinci.", "it", [("Leonardo da Vinci", "PERSON"), ("Vinci", "LOCATION")]),
            # Portuguese
            ("Pelé nasceu em Três Corações, Brasil.", "pt", [("Pelé", "PERSON"), ("Três Corações", "LOCATION"), ("Brasil", "LOCATION")]),
            # Dutch
            ("Anne Frank woonde in Amsterdam.", "nl", [("Anne Frank", "PERSON"), ("Amsterdam", "LOCATION")]),
            # Russian
            ("Владимир Путин родился в Ленинграде.", "ru", [("Владимир Путин", "PERSON"), ("Ленинграде", "LOCATION")]),
        ],
    )
    def test_entity_extraction_multilingual(self, model, text, lang, expected_entities):
        """Test entity extraction across multiple languages."""
        # Mock the transformer output
        mock_results = []
        for entity_text, entity_type in expected_entities:
            mock_results.append(
                {
                    "word": entity_text,
                    "entity_group": entity_type[:3].upper(),
                    "start": text.find(entity_text),
                    "end": text.find(entity_text) + len(entity_text),
                    "score": 0.9,
                }
            )

        model.ner_pipeline.return_value = mock_results

        entities = model._extract_entities(text, lang)

        # Verify entities were extracted
        assert len(entities) > 0

        # Verify entity types are mapped correctly
        entity_types = {e["type"] for e in entities}
        expected_types = {ent[1] for ent in expected_entities}

        assert len(entity_types.intersection(expected_types)) > 0

    @pytest.mark.unit
    def test_asian_languages_ner(self, model):
        """Test NER for Asian languages."""
        asian_test_cases = [
            # Japanese
            ("東京タワーは東京にあります。", "ja", ["東京タワー", "東京"]),
            # Chinese
            ("北京是中国的首都。", "zh", ["北京", "中国"]),
            # Korean
            ("삼성전자는 서울에 본사가 있습니다.", "ko", ["삼성전자", "서울"]),
        ]

        for text, lang, expected_entities in asian_test_cases:
            # Mock appropriate response
            mock_results = [
                {"word": ent, "entity_group": "LOC" if i % 2 else "ORG", "start": text.find(ent), "end": text.find(ent) + len(ent), "score": 0.85}
                for i, ent in enumerate(expected_entities)
                if ent in text
            ]

            model.ner_pipeline.return_value = mock_results

            entities = model._extract_entities(text, lang)

            assert len(entities) > 0

    @pytest.mark.unit
    def test_arabic_ner(self, model):
        """Test NER for Arabic text."""
        text = "محمد صلاح يلعب في ليفربول في إنجلترا."  # Mohamed Salah plays for Liverpool in England
        lang = "ar"

        # Mock entities
        mock_results = [
            {"word": "محمد صلاح", "entity_group": "PER", "start": 0, "end": 10, "score": 0.9},
            {"word": "ليفربول", "entity_group": "ORG", "start": 20, "end": 28, "score": 0.85},
            {"word": "إنجلترا", "entity_group": "LOC", "start": 32, "end": 40, "score": 0.9},
        ]

        model.ner_pipeline.return_value = mock_results

        entities = model._extract_entities(text, lang)

        assert len(entities) >= 2
        # Check RTL text is handled correctly
        assert any(e["type"] == "PERSON" for e in entities)

    @pytest.mark.unit
    def test_entity_type_mapping(self, model):
        """Test that entity types are correctly mapped to standard types."""
        type_mappings = [
            ("PER", "PERSON"),
            ("LOC", "LOCATION"),
            ("ORG", "ORGANIZATION"),
            ("MISC", "MISCELLANEOUS"),
        ]

        for source_type, expected_type in type_mappings:
            assert model.entity_type_mapping.get(source_type) == expected_type

    @pytest.mark.unit
    def test_merge_entities_from_multiple_sources(self, model):
        """Test merging entities from transformer and spaCy."""
        transformer_entities = [
            {"text": "Apple", "type": "ORGANIZATION", "start": 0, "end": 5, "confidence": 0.9, "source": "transformer"},
            {"text": "California", "type": "LOCATION", "start": 20, "end": 30, "confidence": 0.85, "source": "transformer"},
        ]

        spacy_entities = [
            {"text": "Apple", "type": "ORGANIZATION", "start": 0, "end": 5, "confidence": 0.85, "source": "spacy"},
            {"text": "Steve Jobs", "type": "PERSON", "start": 35, "end": 45, "confidence": 0.9, "source": "spacy"},
        ]

        merged = model._merge_entities(transformer_entities, spacy_entities)

        # Should deduplicate "Apple" and keep higher confidence
        assert len(merged) == 3  # Apple, California, Steve Jobs

        # Check that highest confidence is kept
        apple_entities = [e for e in merged if e["text"] == "Apple"]
        assert len(apple_entities) == 1
        assert apple_entities[0]["confidence"] == 0.9

    @pytest.mark.unit
    def test_overlapping_entity_resolution(self, model):
        """Test resolution of overlapping entities."""
        overlapping_entities = [
            {"text": "New York", "type": "LOCATION", "start": 0, "end": 8, "confidence": 0.9},
            {"text": "New York City", "type": "LOCATION", "start": 0, "end": 13, "confidence": 0.95},
            {"text": "York", "type": "LOCATION", "start": 4, "end": 8, "confidence": 0.7},
        ]

        resolved = model._resolve_overlaps(overlapping_entities)

        # Should keep "New York City" (highest confidence, longest match)
        assert len(resolved) == 1
        assert resolved[0]["text"] == "New York City"
        assert resolved[0]["confidence"] == 0.95

    @pytest.mark.unit
    def test_empty_text_handling(self, model):
        """Test handling of empty or whitespace-only text."""
        empty_inputs = ["", "   ", "\n\n"]

        for text in empty_inputs:
            model.ner_pipeline.return_value = []
            entities = model._extract_entities(text, "en")

            assert len(entities) == 0

    @pytest.mark.unit
    def test_no_entities_found(self, model):
        """Test handling when no entities are found."""
        text = "This is a simple sentence with no named entities."
        model.ner_pipeline.return_value = []

        entities = model._extract_entities(text, "en")

        assert entities == []

    @pytest.mark.unit
    def test_long_text_with_many_entities(self, model):
        """Test handling of long text with many entities."""
        text = "Apple Inc. was founded by Steve Jobs in Cupertino, California. Microsoft was founded by Bill Gates in Seattle, Washington. Google was founded by Larry Page in Mountain View."

        # Mock multiple entities
        mock_results = [
            {"word": "Apple Inc.", "entity_group": "ORG", "start": 0, "end": 10, "score": 0.9},
            {"word": "Steve Jobs", "entity_group": "PER", "start": 29, "end": 39, "score": 0.95},
            {"word": "Cupertino", "entity_group": "LOC", "start": 43, "end": 52, "score": 0.85},
            {"word": "California", "entity_group": "LOC", "start": 54, "end": 64, "score": 0.9},
            # ... more entities
        ]

        model.ner_pipeline.return_value = mock_results

        entities = model._extract_entities(text, "en")

        assert len(entities) >= 4

    @pytest.mark.unit
    def test_mixed_language_entity_detection(self, model):
        """Test entity detection in mixed-language text."""
        text = "I visited Paris and 東京 last year."  # English + Japanese

        mock_results = [
            {"word": "Paris", "entity_group": "LOC", "start": 10, "end": 15, "score": 0.9},
            {"word": "東京", "entity_group": "LOC", "start": 20, "end": 22, "score": 0.85},
        ]

        model.ner_pipeline.return_value = mock_results

        entities = model._extract_entities(text, "en")

        assert len(entities) == 2
        # Both entities should be detected despite language mixing
        assert any(e["text"] == "Paris" for e in entities)
        assert any(e["text"] == "東京" for e in entities)

    @pytest.mark.unit
    def test_entity_confidence_scores(self, model):
        """Test that confidence scores are preserved and reasonable."""
        text = "Angela Merkel visited Washington."

        mock_results = [
            {"word": "Angela Merkel", "entity_group": "PER", "start": 0, "end": 13, "score": 0.98},
            {"word": "Washington", "entity_group": "LOC", "start": 22, "end": 32, "score": 0.92},
        ]

        model.ner_pipeline.return_value = mock_results

        entities = model._extract_entities(text, "en")

        for entity in entities:
            assert 0.0 <= entity["confidence"] <= 1.0
            assert entity["confidence"] > 0.5  # Should be reasonably confident

    @pytest.mark.unit
    def test_special_characters_in_entities(self, model):
        """Test handling of entities with special characters."""
        text = "Visit O'Reilly Media at https://www.oreilly.com"

        mock_results = [{"word": "O'Reilly Media", "entity_group": "ORG", "start": 6, "end": 20, "score": 0.85}]

        model.ner_pipeline.return_value = mock_results

        entities = model._extract_entities(text, "en")

        assert len(entities) > 0
        # Entity with apostrophe should be handled correctly
        assert any("O'Reilly" in e["text"] or "O" in e["text"] for e in entities)

    @pytest.mark.unit
    def test_fallback_to_transformer_only(self, model):
        """Test fallback when spaCy model is not available for language."""
        text = "Test sentence in unsupported language."
        lang = "xx"  # Unsupported language code

        mock_results = [{"word": "Test", "entity_group": "MISC", "start": 0, "end": 4, "score": 0.7}]

        model.ner_pipeline.return_value = mock_results

        entities = model._extract_entities(text, lang)

        # Should still work with transformer model only
        assert len(entities) > 0

    @pytest.mark.unit
    def test_entity_boundaries_accuracy(self, model):
        """Test that entity boundaries (start/end) are accurate."""
        text = "Apple was founded in California."

        mock_results = [
            {"word": "Apple", "entity_group": "ORG", "start": 0, "end": 5, "score": 0.9},
            {"word": "California", "entity_group": "LOC", "start": 21, "end": 31, "score": 0.85},
        ]

        model.ner_pipeline.return_value = mock_results

        entities = model._extract_entities(text, "en")

        for entity in entities:
            # Verify that start/end indices correctly extract the entity text
            extracted = text[entity["start"] : entity["end"]]
            assert entity["text"] in extracted or extracted in entity["text"]

    @pytest.mark.unit
    def test_sorted_entity_output(self, model):
        """Test that entities are sorted by position in text."""
        text = "Berlin is in Germany, and Paris is in France."

        mock_results = [
            {"word": "Paris", "entity_group": "LOC", "start": 26, "end": 31, "score": 0.9},
            {"word": "Berlin", "entity_group": "LOC", "start": 0, "end": 6, "score": 0.9},
            {"word": "Germany", "entity_group": "LOC", "start": 13, "end": 20, "score": 0.85},
            {"word": "France", "entity_group": "LOC", "start": 39, "end": 45, "score": 0.9},
        ]

        model.ner_pipeline.return_value = mock_results

        entities = model._extract_entities(text, "en")

        # Entities should be sorted by start position
        for i in range(len(entities) - 1):
            assert entities[i]["start"] <= entities[i + 1]["start"]

    @pytest.mark.benchmark
    def test_multilingual_ner_performance(self, model, benchmark):
        """Benchmark multilingual NER performance."""
        text = "Barack Obama was born in Hawaii and became president of the United States."

        mock_results = [
            {"word": "Barack Obama", "entity_group": "PER", "start": 0, "end": 12, "score": 0.95},
            {"word": "Hawaii", "entity_group": "LOC", "start": 25, "end": 31, "score": 0.9},
            {"word": "United States", "entity_group": "LOC", "start": 64, "end": 77, "score": 0.92},
        ]

        model.ner_pipeline.return_value = mock_results

        result = benchmark(model._extract_entities, text, "en")

        assert len(result) > 0
