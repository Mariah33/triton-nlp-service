"""Unit tests for postprocessing model."""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

# Mock triton_python_backend_utils for testing
mock_pb_utils = MagicMock()


class TestPostprocessing:
    """Test suite for postprocessing and result aggregation."""

    @pytest.fixture
    def model(self):
        """Create postprocessing model instance."""
        with patch.dict("sys.modules", {"triton_python_backend_utils": mock_pb_utils}):
            sys.path.insert(0, "model_repository/postprocessing/1")
            from model import TritonPythonModel

            model = TritonPythonModel()
            model.initialize({"model_config": json.dumps({"name": "postprocessing"})})
            return model

    @pytest.mark.unit
    def test_format_data_type_result_with_detections(self, model):
        """Test formatting of data type detection results."""
        data_type_dict = {
            "text": "john@example.com",
            "detections": [
                {"type": "email", "confidence": 0.95, "value": "john@example.com", "category": "contact"},
                {"type": "text", "confidence": 0.5, "value": "john@example.com", "category": "general"},
            ],
            "primary_type": "email",
            "confidence": 0.95,
        }

        result = model._format_data_type_result(data_type_dict)

        assert result["detected"] is True
        assert result["primary_type"] == "email"
        assert result["confidence"] == 0.95
        assert len(result["detections"]) == 2

        # Check first detection
        assert result["detections"][0]["type"] == "email"
        assert result["detections"][0]["confidence"] == 0.95
        assert result["detections"][0]["category"] == "contact"

    @pytest.mark.unit
    def test_format_data_type_result_empty(self, model):
        """Test formatting of empty data type results."""
        result = model._format_data_type_result({})

        assert result["detected"] is False
        assert result["type"] == "unknown"

    @pytest.mark.unit
    def test_format_data_type_result_value_truncation(self, model):
        """Test that long values are truncated."""
        long_value = "a" * 100  # 100 character string
        data_type_dict = {
            "detections": [{"type": "text", "confidence": 0.8, "value": long_value, "category": "general"}],
            "primary_type": "text",
            "confidence": 0.8,
        }

        result = model._format_data_type_result(data_type_dict)

        # Value should be truncated to 50 chars + "..."
        assert len(result["detections"][0]["value"]) == 53  # 50 + "..."
        assert result["detections"][0]["value"].endswith("...")

    @pytest.mark.unit
    def test_format_ner_result_with_entities(self, model):
        """Test formatting of NER results with entities."""
        ner_dict = {
            "text": "Apple Inc. is in California",
            "entities": [
                {"text": "Apple Inc.", "type": "ORGANIZATION", "start": 0, "end": 10, "confidence": 0.9},
                {"text": "California", "type": "LOCATION", "start": 17, "end": 27, "confidence": 0.85},
            ],
            "entity_count": 2,
            "entity_types": ["ORGANIZATION", "LOCATION"],
        }

        result = model._format_ner_result(ner_dict)

        assert result["count"] == 2
        assert len(result["types_found"]) == 2
        assert "ORGANIZATION" in result["types_found"]
        assert "LOCATION" in result["types_found"]

        # Check entities are included
        assert len(result["entities"]) == 2

    @pytest.mark.unit
    def test_format_ner_result_empty(self, model):
        """Test formatting of empty NER results."""
        result = model._format_ner_result({})

        assert result["entities"] == []
        assert result["count"] == 0

    @pytest.mark.unit
    def test_format_ner_result_grouping_by_type(self, model):
        """Test that entities are grouped by type."""
        ner_dict = {
            "entities": [
                {"text": "John Smith", "type": "PERSON", "start": 0, "end": 10, "confidence": 0.9},
                {"text": "Jane Doe", "type": "PERSON", "start": 15, "end": 23, "confidence": 0.85},
                {"text": "New York", "type": "LOCATION", "start": 30, "end": 38, "confidence": 0.9},
            ],
            "entity_count": 3,
            "entity_types": ["PERSON", "LOCATION"],
        }

        result = model._format_ner_result(ner_dict)

        # Check grouping
        assert "entities_by_type" in result
        assert "PERSON" in result["entities_by_type"]
        assert "LOCATION" in result["entities_by_type"]

        # Check PERSON group has 2 entities
        assert len(result["entities_by_type"]["PERSON"]) == 2
        assert len(result["entities_by_type"]["LOCATION"]) == 1

    @pytest.mark.unit
    def test_format_ner_result_entity_positions(self, model):
        """Test that entity positions are formatted correctly."""
        ner_dict = {
            "entities": [{"text": "Apple", "type": "ORGANIZATION", "start": 0, "end": 5, "confidence": 0.9}],
            "entity_count": 1,
            "entity_types": ["ORGANIZATION"],
        }

        result = model._format_ner_result(ner_dict)

        # Check position format
        entity = result["entities_by_type"]["ORGANIZATION"][0]
        assert entity["position"] == [0, 5]
        assert entity["text"] == "Apple"
        assert entity["confidence"] == 0.9

    @pytest.mark.unit
    def test_format_transliteration_result_success(self, model):
        """Test formatting of successful transliteration."""
        transliteration_dict = {
            "original": "नमस्ते",
            "transliterated": "namaste",
            "source_script": "devanagari",
            "target_script": "latin",
            "confidence": 0.85,
            "method": "rule_based",
        }

        result = model._format_transliteration_result(transliteration_dict)

        assert result["success"] is True
        assert result["original"] == "नमस्ते"
        assert result["transliterated"] == "namaste"
        assert result["source_script"] == "devanagari"
        assert result["target_script"] == "latin"
        assert result["confidence"] == 0.85
        assert result["method"] == "rule_based"

    @pytest.mark.unit
    def test_format_transliteration_result_empty(self, model):
        """Test formatting of empty transliteration results."""
        result = model._format_transliteration_result({})

        assert result["success"] is False

    @pytest.mark.unit
    def test_format_transliteration_result_failure(self, model):
        """Test formatting of failed transliteration (confidence 0)."""
        transliteration_dict = {"original": "text", "transliterated": "text", "confidence": 0.0, "method": "unsupported"}

        result = model._format_transliteration_result(transliteration_dict)

        assert result["success"] is False  # Confidence 0 means not successful

    @pytest.mark.unit
    def test_format_translation_result_success(self, model):
        """Test formatting of successful translation."""
        translation_dict = {
            "original": "hello",
            "translated": "hola",
            "source_language": "en",
            "target_language": "es",
            "confidence": 0.9,
            "method": "neural_mt",
            "alternative_translations": ["hola", "buenos días"],
        }

        result = model._format_translation_result(translation_dict)

        assert result["success"] is True
        assert result["original"] == "hello"
        assert result["translated"] == "hola"
        assert result["source_language"] == "en"
        assert result["target_language"] == "es"
        assert result["confidence"] == 0.9
        assert result["method"] == "neural_mt"
        assert len(result["alternatives"]) == 2

    @pytest.mark.unit
    def test_format_translation_result_empty(self, model):
        """Test formatting of empty translation results."""
        result = model._format_translation_result({})

        assert result["success"] is False

    @pytest.mark.unit
    def test_format_translation_result_no_alternatives(self, model):
        """Test translation formatting when no alternatives provided."""
        translation_dict = {
            "original": "hello",
            "translated": "hola",
            "source_language": "en",
            "target_language": "es",
            "confidence": 0.8,
            "method": "dictionary",
        }

        result = model._format_translation_result(translation_dict)

        assert result["success"] is True
        assert result["alternatives"] == []  # Default to empty list

    @pytest.mark.unit
    def test_generate_summary_all_services(self, model):
        """Test summary generation with all services."""
        result_dict = {
            "original_text": "john@example.com works at Apple in California",
            "preprocessed_text": "john@example.com works at Apple in California",
            "metadata": {},
            "results": {
                "data_type_detection": {"detected": True, "primary_type": "email", "confidence": 0.95, "detections": []},
                "named_entities": {"count": 2, "types_found": ["ORGANIZATION", "LOCATION"], "entities": []},
                "transliteration": {"success": True, "source_script": "latin", "target_script": "latin", "confidence": 1.0},
                "translation": {"success": True, "source_language": "en", "target_language": "es", "confidence": 0.85},
            },
        }

        summary = model._generate_summary(result_dict)

        assert "text_length" in summary
        assert summary["text_length"] == len(result_dict["original_text"])

        assert "services_applied" in summary
        assert len(summary["services_applied"]) == 4

        assert "key_findings" in summary
        assert len(summary["key_findings"]) >= 3  # Should have findings for each service

    @pytest.mark.unit
    def test_generate_summary_data_type_finding(self, model):
        """Test that data type detection appears in summary."""
        result_dict = {
            "original_text": "john@example.com",
            "results": {"data_type_detection": {"detected": True, "primary_type": "email", "confidence": 0.95}},
        }

        summary = model._generate_summary(result_dict)

        # Should have a finding about email detection
        findings_text = " ".join(summary["key_findings"])
        assert "email" in findings_text.lower()
        assert "0.95" in findings_text

    @pytest.mark.unit
    def test_generate_summary_ner_finding(self, model):
        """Test that NER results appear in summary."""
        result_dict = {
            "original_text": "Apple is in California",
            "results": {"named_entities": {"count": 2, "types_found": ["ORGANIZATION", "LOCATION"], "entities": []}},
        }

        summary = model._generate_summary(result_dict)

        # Should have a finding about entities
        findings_text = " ".join(summary["key_findings"])
        assert "2" in findings_text or "entities" in findings_text.lower()

    @pytest.mark.unit
    def test_generate_summary_transliteration_finding(self, model):
        """Test that transliteration appears in summary."""
        result_dict = {
            "original_text": "नमस्ते",
            "results": {"transliteration": {"success": True, "source_script": "devanagari", "target_script": "latin", "confidence": 0.85}},
        }

        summary = model._generate_summary(result_dict)

        # Should have a finding about transliteration
        findings_text = " ".join(summary["key_findings"])
        assert "devanagari" in findings_text.lower() or "latin" in findings_text.lower()

    @pytest.mark.unit
    def test_generate_summary_translation_finding(self, model):
        """Test that translation appears in summary."""
        result_dict = {
            "original_text": "hello",
            "results": {"translation": {"success": True, "source_language": "en", "target_language": "es", "confidence": 0.9}},
        }

        summary = model._generate_summary(result_dict)

        # Should have a finding about translation
        findings_text = " ".join(summary["key_findings"])
        assert "en" in findings_text.lower() or "es" in findings_text.lower() or "translated" in findings_text.lower()

    @pytest.mark.unit
    def test_generate_summary_no_findings(self, model):
        """Test summary generation when no results detected."""
        result_dict = {
            "original_text": "test",
            "results": {"data_type_detection": {"detected": False}, "named_entities": {"count": 0, "entities": []}},
        }

        summary = model._generate_summary(result_dict)

        # Should still have structure
        assert "key_findings" in summary
        # Findings might be empty or minimal
        assert isinstance(summary["key_findings"], list)

    @pytest.mark.unit
    def test_generate_summary_text_length(self, model):
        """Test that text length is calculated correctly."""
        test_cases = [
            ("short", 5),
            ("a longer text string", 21),
            ("", 0),
        ]

        for text, expected_length in test_cases:
            result_dict = {"original_text": text, "results": {}}
            summary = model._generate_summary(result_dict)

            assert summary["text_length"] == expected_length

    @pytest.mark.unit
    def test_format_data_type_result_optional_fields(self, model):
        """Test handling of optional fields in data type detection."""
        data_type_dict = {
            "detections": [
                {
                    "type": "phone_number",
                    "confidence": 0.9,
                    "value": "555-1234",
                    "category": "contact",
                    # No subtype field
                }
            ],
            "primary_type": "phone_number",
            "confidence": 0.9,
        }

        result = model._format_data_type_result(data_type_dict)

        # Should handle missing subtype gracefully
        assert result["detections"][0]["subtype"] is None

    @pytest.mark.unit
    def test_format_ner_result_unknown_entity_type(self, model):
        """Test handling of entities with unknown types."""
        ner_dict = {
            "entities": [
                {"text": "Something", "start": 0, "end": 9, "confidence": 0.5}
                # No type field
            ],
            "entity_count": 1,
            "entity_types": [],
        }

        result = model._format_ner_result(ner_dict)

        # Should handle missing type as UNKNOWN
        assert "UNKNOWN" in result["entities_by_type"]
        assert len(result["entities_by_type"]["UNKNOWN"]) == 1

    @pytest.mark.unit
    def test_format_transliteration_result_default_values(self, model):
        """Test default values for missing transliteration fields."""
        transliteration_dict = {
            "confidence": 0.8
            # Other fields missing
        }

        result = model._format_transliteration_result(transliteration_dict)

        # Should provide defaults
        assert result["original"] == ""
        assert result["transliterated"] == ""
        assert result["source_script"] == "unknown"
        assert result["target_script"] == "latin"

    @pytest.mark.unit
    def test_format_translation_result_default_values(self, model):
        """Test default values for missing translation fields."""
        translation_dict = {
            "confidence": 0.7
            # Other fields missing
        }

        result = model._format_translation_result(translation_dict)

        # Should provide defaults
        assert result["original"] == ""
        assert result["translated"] == ""
        assert result["source_language"] == "unknown"
        assert result["target_language"] == "unknown"

    @pytest.mark.unit
    def test_summary_services_applied_list(self, model):
        """Test that services_applied correctly lists all applied services."""
        result_dict = {
            "original_text": "test",
            "results": {"data_type_detection": {}, "named_entities": {}, "translation": {}},
        }

        summary = model._generate_summary(result_dict)

        assert "data_type_detection" in summary["services_applied"]
        assert "named_entities" in summary["services_applied"]
        assert "translation" in summary["services_applied"]
        assert len(summary["services_applied"]) == 3

    @pytest.mark.unit
    def test_multiple_entities_same_type(self, model):
        """Test handling of multiple entities of the same type."""
        ner_dict = {
            "entities": [
                {"text": "John", "type": "PERSON", "start": 0, "end": 4, "confidence": 0.9},
                {"text": "Jane", "type": "PERSON", "start": 10, "end": 14, "confidence": 0.85},
                {"text": "Bob", "type": "PERSON", "start": 20, "end": 23, "confidence": 0.8},
            ],
            "entity_count": 3,
            "entity_types": ["PERSON"],
        }

        result = model._format_ner_result(ner_dict)

        # All three should be in PERSON group
        assert len(result["entities_by_type"]["PERSON"]) == 3

    @pytest.mark.unit
    def test_confidence_score_formatting(self, model):
        """Test that confidence scores are preserved correctly."""
        test_confidence = 0.87654321

        data_type_dict = {"primary_type": "email", "confidence": test_confidence, "detections": []}

        result = model._format_data_type_result(data_type_dict)

        # Confidence should be preserved (not rounded in formatting)
        assert result["confidence"] == test_confidence

    @pytest.mark.unit
    def test_empty_entity_list(self, model):
        """Test NER formatting with empty entity list."""
        ner_dict = {"entities": [], "entity_count": 0, "entity_types": []}

        result = model._format_ner_result(ner_dict)

        assert result["count"] == 0
        assert result["entities"] == []
        assert result["entities_by_type"] == {}

    @pytest.mark.unit
    def test_summary_key_findings_order(self, model):
        """Test that key findings are in a consistent order."""
        result_dict = {
            "original_text": "test",
            "results": {
                "data_type_detection": {"detected": True, "primary_type": "text", "confidence": 0.8},
                "named_entities": {"count": 1, "types_found": ["PERSON"], "entities": []},
                "transliteration": {"success": True, "source_script": "latin", "target_script": "latin", "confidence": 1.0},
                "translation": {"success": True, "source_language": "en", "target_language": "es", "confidence": 0.9},
            },
        }

        summary = model._generate_summary(result_dict)

        # Should have findings in order: data_type, ner, transliteration, translation
        findings = summary["key_findings"]
        assert len(findings) == 4

        # Check that data type is first
        assert "text" in findings[0].lower() or "detected" in findings[0].lower()

    @pytest.mark.unit
    def test_unicode_text_handling(self, model):
        """Test handling of Unicode text in results."""
        result_dict = {
            "original_text": "नमस्ते 你好 مرحبا",
            "results": {},
        }

        summary = model._generate_summary(result_dict)

        # Should handle Unicode length correctly
        assert summary["text_length"] == len("नमस्ते 你好 مरحبا")

    @pytest.mark.unit
    def test_special_characters_in_findings(self, model):
        """Test that special characters in findings are handled."""
        result_dict = {
            "original_text": "test@example.com",
            "results": {"data_type_detection": {"detected": True, "primary_type": "email", "confidence": 0.95}},
        }

        summary = model._generate_summary(result_dict)

        # Findings should be valid strings
        for finding in summary["key_findings"]:
            assert isinstance(finding, str)
            assert len(finding) > 0
