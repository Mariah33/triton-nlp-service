"""Named Entity Recognition Model.

Identifies entities like persons, organizations, locations, dates, etc.
"""

import json
import re
from typing import Any

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """NER model for entity extraction.."""

    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])

        # Entity patterns for rule-based NER
        # In production, use spaCy, Hugging Face, or Flair models
        self.entity_patterns = {
            "PERSON": {
                "titles": ["Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sir", "Lady"],
                "common_first_names": [
                    "John",
                    "Jane",
                    "Michael",
                    "Sarah",
                    "David",
                    "Emma",
                    "James",
                    "Mary",
                ],
                "common_last_names": [
                    "Smith",
                    "Johnson",
                    "Brown",
                    "Jones",
                    "Miller",
                    "Davis",
                    "Garcia",
                    "Wilson",
                ],
            },
            "ORGANIZATION": {
                "suffixes": [
                    "Inc.",
                    "Corp.",
                    "LLC",
                    "Ltd.",
                    "Co.",
                    "Company",
                    "Corporation",
                    "Group",
                    "Industries",
                ],
                "keywords": [
                    "Google",
                    "Microsoft",
                    "Apple",
                    "Amazon",
                    "Facebook",
                    "Tesla",
                    "IBM",
                    "Oracle",
                ],
            },
            "LOCATION": {
                "countries": [
                    "United States",
                    "USA",
                    "UK",
                    "United Kingdom",
                    "Canada",
                    "Australia",
                    "Germany",
                    "France",
                    "India",
                    "China",
                    "Japan",
                ],
                "cities": [
                    "New York",
                    "London",
                    "Paris",
                    "Tokyo",
                    "Sydney",
                    "Mumbai",
                    "Berlin",
                    "Toronto",
                    "Singapore",
                    "Dubai",
                ],
                "keywords": [
                    "Street",
                    "Avenue",
                    "Road",
                    "Boulevard",
                    "Lane",
                    "Drive",
                    "Court",
                    "Plaza",
                ],
            },
            "DATE": {
                "months": [
                    "January",
                    "February",
                    "March",
                    "April",
                    "May",
                    "June",
                    "July",
                    "August",
                    "September",
                    "October",
                    "November",
                    "December",
                    "Jan",
                    "Feb",
                    "Mar",
                    "Apr",
                    "May",
                    "Jun",
                    "Jul",
                    "Aug",
                    "Sep",
                    "Oct",
                    "Nov",
                    "Dec",
                ],
                "days": [
                    "Monday",
                    "Tuesday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                    "Saturday",
                    "Sunday",
                    "Mon",
                    "Tue",
                    "Wed",
                    "Thu",
                    "Fri",
                    "Sat",
                    "Sun",
                ],
                "patterns": [
                    r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}",  # MM/DD/YYYY or DD-MM-YYYY
                    r"\d{4}[/-]\d{1,2}[/-]\d{1,2}",  # YYYY-MM-DD
                    r"\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{2,4}",
                    r"(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{2,4}",
                ],
            },
            "TIME": {
                "patterns": [
                    r"\d{1,2}:\d{2}\s*(AM|PM|am|pm)?",  # 12:30 PM
                    r"\d{1,2}\s*(AM|PM|am|pm)",  # 3 PM
                    r"\d{1,2}:\d{2}:\d{2}",  # 14:30:45
                ]
            },
            "MONEY": {
                "patterns": [
                    r"\$[\d,]+\.?\d*",  # $100.50
                    r"[\d,]+\.?\d*\s*(dollars?|USD|EUR|GBP|pounds?|euros?)",
                    r"£[\d,]+\.?\d*",  # £100.50
                    r"€[\d,]+\.?\d*",  # €100.50
                    r"¥[\d,]+\.?\d*",  # ¥100.50
                    r"₹[\d,]+\.?\d*",  # ₹100.50
                ]
            },
            "PERCENT": {
                "patterns": [
                    r"\d+\.?\d*\s*%",  # 50%
                    r"\d+\.?\d*\s+percent",  # 50 percent
                ]
            },
            "EMAIL": {"patterns": [r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"]},
            "URL": {
                "patterns": [
                    r"https?://(?:[-\w.])+(?::\d+)?(?:[/\w\-._~:/?#[\]@!$&\'()*+,;=.]+)?",
                    r"www\.[A-Za-z0-9\-._~:/?#[\]@!$&\'()*+,;=.]+\.[A-Za-z]{2,}",
                ]
            },
            "PHONE": {
                "patterns": [
                    r"\+?\d{1,4}[\s-]?\(?\d{1,4}\)?[\s-]?\d{1,4}[\s-]?\d{1,4}",
                    r"\(\d{3}\)\s*\d{3}-\d{4}",  # (123) 456-7890
                    r"\d{3}-\d{3}-\d{4}",  # 123-456-7890
                ]
            },
        }

    def execute(self, requests):
        responses = []

        for request in requests:
            # Get input tensor
            text_tensor = pb_utils.get_input_tensor_by_name(request, "text")
            texts = text_tensor.as_numpy().tolist()

            ner_results = []

            for text_bytes in texts:
                text = text_bytes.decode("utf-8") if isinstance(text_bytes, bytes) else str(text_bytes)

                # Perform NER
                entities = self._extract_entities(text)
                result = {
                    "text": text,
                    "entities": entities,
                    "entity_count": len(entities),
                    "entity_types": list({e["type"] for e in entities}),
                }
                ner_results.append(json.dumps(result))

            # Create output tensor
            out_tensor = pb_utils.Tensor("entities", np.array(ner_results, dtype=np.object_))

            # Create response
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(inference_response)

        return responses

    def _extract_entities(self, text: str) -> list[dict[str, Any]]:
        """Extract named entities from text.."""
        entities = []

        # Extract pattern-based entities first
        for entity_type in ["DATE", "TIME", "MONEY", "PERCENT", "EMAIL", "URL", "PHONE"]:
            if entity_type in self.entity_patterns and "patterns" in self.entity_patterns[entity_type]:
                for pattern in self.entity_patterns[entity_type]["patterns"]:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in matches:
                        entities.append(
                            {
                                "text": match.group(),
                                "type": entity_type,
                                "start": match.start(),
                                "end": match.end(),
                                "confidence": 0.85,
                            }
                        )

        # Extract PERSON entities
        entities.extend(self._extract_persons(text))

        # Extract ORGANIZATION entities
        entities.extend(self._extract_organizations(text))

        # Extract LOCATION entities
        entities.extend(self._extract_locations(text))

        # Remove duplicates and overlapping entities
        entities = self._resolve_overlaps(entities)

        # Sort by position
        entities.sort(key=lambda x: x["start"])

        return entities

    def _extract_persons(self, text: str) -> list[dict[str, Any]]:
        """Extract person names.."""
        entities = []
        words = text.split()

        # Look for titles followed by names
        for i, word in enumerate(words):
            if word in self.entity_patterns["PERSON"]["titles"] and i + 1 < len(words):
                # Check next 1-2 words as potential name
                name_parts = []
                for j in range(1, min(3, len(words) - i)):
                    next_word = words[i + j]
                    if next_word[0].isupper():
                        name_parts.append(next_word)
                    else:
                        break

                if name_parts:
                    full_name = f"{word} {' '.join(name_parts)}"
                    start_idx = text.find(full_name)
                    if start_idx != -1:
                        entities.append(
                            {
                                "text": full_name,
                                "type": "PERSON",
                                "start": start_idx,
                                "end": start_idx + len(full_name),
                                "confidence": 0.8,
                            }
                        )

        # Look for capitalized word sequences that might be names
        capitalized_pattern = r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b"
        matches = re.finditer(capitalized_pattern, text)
        for match in matches:
            potential_name = match.group()
            words_in_name = potential_name.split()

            # Check if any word is a known first or last name
            is_likely_name = any(
                word in self.entity_patterns["PERSON"]["common_first_names"] or word in self.entity_patterns["PERSON"]["common_last_names"]
                for word in words_in_name
            )

            if is_likely_name or len(words_in_name) == 2:  # Two capitalized words often indicate names
                entities.append(
                    {
                        "text": potential_name,
                        "type": "PERSON",
                        "start": match.start(),
                        "end": match.end(),
                        "confidence": 0.7 if is_likely_name else 0.6,
                    }
                )

        return entities

    def _extract_organizations(self, text: str) -> list[dict[str, Any]]:
        """Extract organization names.."""
        entities = []

        # Look for known organizations
        for org in self.entity_patterns["ORGANIZATION"]["keywords"]:
            pattern = r"\b" + re.escape(org) + r"\b"
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append(
                    {
                        "text": match.group(),
                        "type": "ORGANIZATION",
                        "start": match.start(),
                        "end": match.end(),
                        "confidence": 0.9,
                    }
                )

        # Look for organization suffixes
        for suffix in self.entity_patterns["ORGANIZATION"]["suffixes"]:
            pattern = r"\b[A-Z][A-Za-z\s&]+\s+" + re.escape(suffix)
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append(
                    {
                        "text": match.group(),
                        "type": "ORGANIZATION",
                        "start": match.start(),
                        "end": match.end(),
                        "confidence": 0.75,
                    }
                )

        return entities

    def _extract_locations(self, text: str) -> list[dict[str, Any]]:
        """Extract location names.."""
        entities = []

        # Look for known locations
        all_locations = self.entity_patterns["LOCATION"]["countries"] + self.entity_patterns["LOCATION"]["cities"]

        for location in all_locations:
            pattern = r"\b" + re.escape(location) + r"\b"
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append(
                    {
                        "text": match.group(),
                        "type": "LOCATION",
                        "start": match.start(),
                        "end": match.end(),
                        "confidence": 0.85,
                    }
                )

        # Look for address patterns
        for keyword in self.entity_patterns["LOCATION"]["keywords"]:
            pattern = r"\d+\s+[A-Z][A-Za-z\s]+\s+" + keyword
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append(
                    {
                        "text": match.group(),
                        "type": "LOCATION",
                        "subtype": "ADDRESS",
                        "start": match.start(),
                        "end": match.end(),
                        "confidence": 0.7,
                    }
                )

        return entities

    def _resolve_overlaps(self, entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove overlapping entities, keeping the one with higher confidence.."""
        if not entities:
            return entities

        # Sort by start position and confidence
        entities.sort(key=lambda x: (x["start"], -x["confidence"]))

        resolved = []
        for entity in entities:
            # Check if this entity overlaps with any already resolved entity
            overlaps = False
            for resolved_entity in resolved:
                if entity["start"] < resolved_entity["end"] and entity["end"] > resolved_entity["start"]:
                    overlaps = True
                    break

            if not overlaps:
                resolved.append(entity)

        return resolved

    def finalize(self):
        pass
