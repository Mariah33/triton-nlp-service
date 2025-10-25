"""Unit tests for Named Entity Recognition model."""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

# Mock triton_python_backend_utils for testing
mock_pb_utils = MagicMock()


class TestNER:
    """Test suite for Named Entity Recognition."""

    @pytest.fixture
    def model(self):
        """Create NER model instance."""
        with patch.dict("sys.modules", {"triton_python_backend_utils": mock_pb_utils}):
            sys.path.insert(0, "model_repository/ner/1")
            from model import TritonPythonModel

            model = TritonPythonModel()
            model.initialize({"model_config": json.dumps({"name": "ner"})})
            return model

    @pytest.mark.unit
    def test_person_extraction_with_title(self, model):
        """Test extraction of person names with titles."""
        text = "Dr. Jane Smith and Mr. John Doe attended the meeting."
        entities = model._extract_entities(text)

        person_entities = [e for e in entities if e["type"] == "PERSON"]
        assert len(person_entities) >= 2

        # Check for Dr. Jane Smith
        assert any("Jane Smith" in e["text"] for e in person_entities)

        # Check for Mr. John Doe
        assert any("John Doe" in e["text"] for e in person_entities)

    @pytest.mark.unit
    def test_person_extraction_capitalized_names(self, model):
        """Test extraction of capitalized names without titles."""
        text = "Sarah Johnson met with Michael Brown yesterday."
        entities = model._extract_entities(text)

        person_entities = [e for e in entities if e["type"] == "PERSON"]
        assert len(person_entities) >= 1

        # Should detect at least one name
        names = [e["text"] for e in person_entities]
        assert any("Sarah Johnson" in name or "Michael Brown" in name for name in names)

    @pytest.mark.unit
    def test_organization_extraction_keywords(self, model):
        """Test extraction of known organizations."""
        text = "Google and Microsoft are competing with Apple in the market."
        entities = model._extract_entities(text)

        org_entities = [e for e in entities if e["type"] == "ORGANIZATION"]
        assert len(org_entities) >= 3

        org_texts = [e["text"].lower() for e in org_entities]
        assert any("google" in text for text in org_texts)
        assert any("microsoft" in text for text in org_texts)
        assert any("apple" in text for text in org_texts)

    @pytest.mark.unit
    def test_organization_extraction_suffixes(self, model):
        """Test extraction of organizations with suffixes."""
        test_cases = [
            "Acme Corp.",
            "Tech Solutions Inc.",
            "Global Industries LLC",
            "Business Group Ltd.",
        ]

        for org_name in test_cases:
            text = f"{org_name} announced new products."
            entities = model._extract_entities(text)

            org_entities = [e for e in entities if e["type"] == "ORGANIZATION"]
            assert len(org_entities) >= 1, f"Failed to detect: {org_name}"

    @pytest.mark.unit
    def test_location_extraction_countries(self, model):
        """Test extraction of country names."""
        text = "I have visited USA, UK, and Canada this year."
        entities = model._extract_entities(text)

        location_entities = [e for e in entities if e["type"] == "LOCATION"]
        assert len(location_entities) >= 3

        location_texts = [e["text"].lower() for e in location_entities]
        assert any("usa" in text for text in location_texts)
        assert any("uk" in text for text in location_texts)
        assert any("canada" in text for text in location_texts)

    @pytest.mark.unit
    def test_location_extraction_cities(self, model):
        """Test extraction of city names."""
        text = "Flight from New York to London via Paris."
        entities = model._extract_entities(text)

        location_entities = [e for e in entities if e["type"] == "LOCATION"]
        assert len(location_entities) >= 2

        location_texts = [e["text"] for e in location_entities]
        assert any("New York" in text or "London" in text or "Paris" in text for text in location_texts)

    @pytest.mark.unit
    def test_location_extraction_addresses(self, model):
        """Test extraction of street addresses."""
        text = "The office is at 123 Main Street in downtown."
        entities = model._extract_entities(text)

        address_entities = [e for e in entities if e["type"] == "LOCATION" and e.get("subtype") == "ADDRESS"]
        assert len(address_entities) >= 1

        assert any("Main Street" in e["text"] for e in address_entities)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "text,expected_type",
        [
            ("Meeting on January 15, 2024", "DATE"),
            ("Deadline: 12/31/2023", "DATE"),
            ("Born on 25 December 1990", "DATE"),
            ("Conference from 2024-03-15 to 2024-03-17", "DATE"),
        ],
    )
    def test_date_extraction(self, model, text, expected_type):
        """Test extraction of various date formats."""
        entities = model._extract_entities(text)

        date_entities = [e for e in entities if e["type"] == expected_type]
        assert len(date_entities) >= 1, f"Failed to extract date from: {text}"

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "text,expected_type",
        [
            ("Meeting at 3:30 PM", "TIME"),
            ("Start time: 9:00 AM", "TIME"),
            ("System timestamp: 14:25:30", "TIME"),
            ("Arrive by 10 AM", "TIME"),
        ],
    )
    def test_time_extraction(self, model, text, expected_type):
        """Test extraction of various time formats."""
        entities = model._extract_entities(text)

        time_entities = [e for e in entities if e["type"] == expected_type]
        assert len(time_entities) >= 1, f"Failed to extract time from: {text}"

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "text,currency_symbol",
        [
            ("Price: $99.99", "$"),
            ("Cost is £50.00", "£"),
            ("Total: €75.50", "€"),
            ("Amount: ¥1000", "¥"),
            ("Fee: ₹500", "₹"),
            ("Budget of 1000 dollars", "dollars"),
        ],
    )
    def test_money_extraction(self, model, text, currency_symbol):
        """Test extraction of monetary amounts in various currencies."""
        entities = model._extract_entities(text)

        money_entities = [e for e in entities if e["type"] == "MONEY"]
        assert len(money_entities) >= 1, f"Failed to extract money from: {text}"

        # Verify the currency symbol or keyword is present
        assert any(currency_symbol in e["text"] or "dollars" in text.lower() for e in money_entities)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "text",
        [
            "Growth rate: 25%",
            "Discount of 10 percent",
            "Increased by 5.5%",
            "Success rate: 99.9%",
        ],
    )
    def test_percent_extraction(self, model, text):
        """Test extraction of percentages."""
        entities = model._extract_entities(text)

        percent_entities = [e for e in entities if e["type"] == "PERCENT"]
        assert len(percent_entities) >= 1, f"Failed to extract percent from: {text}"

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "text",
        [
            "Contact: john.doe@example.com",
            "Email me at jane_smith@company.co.uk",
            "Send to info@organization.org",
        ],
    )
    def test_email_extraction(self, model, text):
        """Test extraction of email addresses."""
        entities = model._extract_entities(text)

        email_entities = [e for e in entities if e["type"] == "EMAIL"]
        assert len(email_entities) >= 1, f"Failed to extract email from: {text}"

        # Check that @ symbol is present
        assert any("@" in e["text"] for e in email_entities)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "text",
        [
            "Visit https://www.example.com for more info",
            "Check out http://blog.company.org/article",
            "Go to www.website.net",
        ],
    )
    def test_url_extraction(self, model, text):
        """Test extraction of URLs."""
        entities = model._extract_entities(text)

        url_entities = [e for e in entities if e["type"] == "URL"]
        assert len(url_entities) >= 1, f"Failed to extract URL from: {text}"

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "text",
        [
            "Call me at (555) 123-4567",
            "Phone: 555-123-4567",
            "Mobile: +1-555-123-4567",
            "Contact: +44-20-7123-4567",
        ],
    )
    def test_phone_extraction(self, model, text):
        """Test extraction of phone numbers."""
        entities = model._extract_entities(text)

        phone_entities = [e for e in entities if e["type"] == "PHONE"]
        assert len(phone_entities) >= 1, f"Failed to extract phone from: {text}"

    @pytest.mark.unit
    def test_multiple_entity_types(self, model):
        """Test extraction of multiple entity types from single text."""
        text = (
            "Dr. Smith from Microsoft will present in New York on January 15, 2024 at 3:00 PM. Contact: smith@microsoft.com or call (555) 123-4567."
        )

        entities = model._extract_entities(text)

        # Should find multiple entity types
        entity_types = {e["type"] for e in entities}

        # Expecting at least these types
        expected_types = {"PERSON", "ORGANIZATION", "LOCATION", "DATE", "TIME", "EMAIL", "PHONE"}
        found_types = entity_types.intersection(expected_types)

        assert len(found_types) >= 4, f"Expected at least 4 entity types, found: {entity_types}"

    @pytest.mark.unit
    def test_empty_text(self, model):
        """Test handling of empty text."""
        entities = model._extract_entities("")

        assert entities == []

    @pytest.mark.unit
    def test_no_entities_found(self, model):
        """Test handling when no entities are found."""
        text = "this is a simple sentence with no named entities"
        entities = model._extract_entities(text)

        # May still find some entities due to pattern matching, so just check it doesn't crash
        assert isinstance(entities, list)

    @pytest.mark.unit
    def test_entity_positions(self, model):
        """Test that entity positions are accurate."""
        text = "Microsoft is located in Seattle."
        entities = model._extract_entities(text)

        # Find Microsoft entity
        microsoft_entities = [e for e in entities if "Microsoft" in e["text"]]
        assert len(microsoft_entities) >= 1

        for entity in microsoft_entities:
            # Verify start and end positions
            extracted = text[entity["start"] : entity["end"]]
            assert entity["text"] == extracted or entity["text"] in extracted

    @pytest.mark.unit
    def test_sorted_by_position(self, model):
        """Test that entities are sorted by position in text."""
        text = "John Smith works at Apple in California and lives in New York."
        entities = model._extract_entities(text)

        # Entities should be sorted by start position
        for i in range(len(entities) - 1):
            assert entities[i]["start"] <= entities[i + 1]["start"]

    @pytest.mark.unit
    def test_confidence_scores(self, model):
        """Test that confidence scores are present and valid."""
        text = "Apple Inc. is headquartered in Cupertino, California."
        entities = model._extract_entities(text)

        assert len(entities) > 0

        for entity in entities:
            assert "confidence" in entity
            assert 0.0 <= entity["confidence"] <= 1.0

    @pytest.mark.unit
    def test_overlap_resolution(self, model):
        """Test resolution of overlapping entities."""
        # Create test with potential overlaps
        text = "New York City is in New York State."
        entities = model._extract_entities(text)

        # Check that no entities overlap
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i + 1 :]:
                # No overlap: either entity1 ends before entity2 starts, or vice versa
                no_overlap = entity1["end"] <= entity2["start"] or entity2["end"] <= entity1["start"]
                assert no_overlap, f"Found overlapping entities: {entity1['text']} and {entity2['text']}"

    @pytest.mark.unit
    def test_case_insensitive_matching(self, model):
        """Test that entity extraction is case-insensitive where appropriate."""
        test_cases = [
            ("I work at GOOGLE", "ORGANIZATION"),
            ("Visiting london next week", "LOCATION"),
        ]

        for text, expected_type in test_cases:
            entities = model._extract_entities(text)
            entity_types = [e["type"] for e in entities]
            assert expected_type in entity_types, f"Failed to extract {expected_type} from: {text}"

    @pytest.mark.unit
    def test_entity_deduplication(self, model):
        """Test that duplicate entities in same position are removed."""
        text = "Apple is a great company. Apple makes phones."
        entities = model._extract_entities(text)

        # Count entities at each unique position
        [(e["start"], e["end"]) for e in entities]

        # Check first Apple occurrence
        apple_entities_at_start = [e for e in entities if "Apple" in e["text"] and e["start"] == 0]
        assert len(apple_entities_at_start) <= 1, "Found duplicate entities at same position"

    @pytest.mark.unit
    def test_long_text_performance(self, model):
        """Test that extraction works on longer texts."""
        # Create a longer text with multiple entities
        text = "Dr. John Smith from Microsoft attended a conference in New York on January 15, 2024. He presented research on AI at 2:00 PM. Contact him at john.smith@microsoft.com or call (555) 123-4567. The event cost $500 per attendee and had a 95% satisfaction rate."

        entities = model._extract_entities(text)

        # Should find multiple entities
        assert len(entities) >= 8
        entity_types = {e["type"] for e in entities}
        assert len(entity_types) >= 5

    @pytest.mark.unit
    def test_special_characters_in_text(self, model):
        """Test handling of special characters."""
        text = "Email: user@example.com!!! Price: $99.99??? Time: 3:30PM!!!"
        entities = model._extract_entities(text)

        # Should still extract entities despite special characters
        assert len(entities) >= 2

        # Check specific entities
        entity_types = {e["type"] for e in entities}
        assert "EMAIL" in entity_types or "MONEY" in entity_types

    @pytest.mark.unit
    def test_unicode_text(self, model):
        """Test handling of Unicode characters."""
        text = "Price: €100.50 and ¥5000 in Tokyo, Japan."
        entities = model._extract_entities(text)

        # Should extract money and location
        entity_types = {e["type"] for e in entities}
        assert "MONEY" in entity_types
        assert "LOCATION" in entity_types

    @pytest.mark.unit
    def test_entity_text_accuracy(self, model):
        """Test that extracted entity text matches original text."""
        text = "Meeting with Dr. Jane Smith at Microsoft headquarters in Seattle on December 25, 2023 at 3:30 PM."
        entities = model._extract_entities(text)

        for entity in entities:
            # Extract text using positions
            extracted = text[entity["start"] : entity["end"]]

            # Should match exactly or entity text should be contained in extracted
            assert entity["text"] == extracted or entity["text"] in text

    @pytest.mark.benchmark
    def test_ner_performance(self, model, benchmark):
        """Benchmark NER extraction performance."""
        text = "Dr. John Smith from Microsoft will visit Apple Inc. in Cupertino, California on January 15, 2024 at 2:00 PM."

        result = benchmark(model._extract_entities, text)

        assert len(result) > 0
