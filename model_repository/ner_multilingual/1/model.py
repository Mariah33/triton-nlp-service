"""Multilingual Named Entity Recognition Model.

Supports NER across 100+ languages using transformer-based models
Uses XLM-RoBERTa or language-specific spaCy models for accurate entity detection
"""

import json
from typing import Any

import numpy as np
import spacy
import torch
import triton_python_backend_utils as pb_utils
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline


class TritonPythonModel:
    """Multilingual NER model using transformers and spaCy."""

    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize multilingual transformer model
        try:
            model_name = "Davlan/xlm-roberta-base-ner-hrl"  # Multilingual NER model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(model_name).to(self.device)
            self.ner_pipeline = pipeline(
                "ner", model=self.model, tokenizer=self.tokenizer, aggregation_strategy="simple", device=0 if self.device == "cuda" else -1
            )
        except Exception as e:
            print(f"Warning: Could not load transformer model: {e}")
            self.ner_pipeline = None

        # Load language-specific spaCy models (lazy loading)
        self.spacy_models = {}
        self.supported_spacy_languages = {
            "en": "en_core_web_sm",
            "de": "de_core_news_sm",
            "es": "es_core_news_sm",
            "fr": "fr_core_news_sm",
            "it": "it_core_news_sm",
            "pt": "pt_core_news_sm",
            "nl": "nl_core_news_sm",
            "el": "el_core_news_sm",
            "zh": "zh_core_web_sm",
            "ja": "ja_core_news_sm",
            "ru": "ru_core_news_sm",
            "pl": "pl_core_news_sm",
            "ro": "ro_core_news_sm",
            "da": "da_core_news_sm",
            "fi": "fi_core_news_sm",
            "sv": "sv_core_news_sm",
            "nb": "nb_core_news_sm",
            "lt": "lt_core_news_sm",
            "mk": "mk_core_news_sm",
            "ca": "ca_core_news_sm",
            "hr": "hr_core_news_sm",
            "uk": "uk_core_news_sm",
        }

        # Entity type mapping from different models to standard types
        self.entity_type_mapping = {
            "PER": "PERSON",
            "LOC": "LOCATION",
            "ORG": "ORGANIZATION",
            "MISC": "MISCELLANEOUS",
            "DATE": "DATE",
            "TIME": "TIME",
            "MONEY": "MONEY",
            "PERCENT": "PERCENT",
        }

    def execute(self, requests):
        responses = []

        for request in requests:
            # Get input tensors
            text_tensor = pb_utils.get_input_tensor_by_name(request, "text")
            texts = text_tensor.as_numpy().tolist()

            # Get optional language code
            lang_tensor = pb_utils.get_input_tensor_by_name(request, "language_code")
            language_codes = lang_tensor.as_numpy().tolist() if lang_tensor else [None] * len(texts)

            ner_results = []

            for text_bytes, lang_bytes in zip(texts, language_codes):
                text = text_bytes.decode("utf-8") if isinstance(text_bytes, bytes) else str(text_bytes)

                if lang_bytes is not None:
                    lang_code = lang_bytes.decode("utf-8") if isinstance(lang_bytes, bytes) else str(lang_bytes)
                else:
                    lang_code = "en"  # Default to English

                # Perform multilingual NER
                entities = self._extract_entities(text, lang_code)
                result = {
                    "text": text,
                    "language": lang_code,
                    "entities": entities,
                    "entity_count": len(entities),
                    "entity_types": list(set([e["type"] for e in entities])),
                }
                ner_results.append(json.dumps(result))

            # Create output tensor
            out_tensor = pb_utils.Tensor("entities", np.array(ner_results, dtype=np.object_))

            # Create response
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(inference_response)

        return responses

    def _extract_entities(self, text: str, language_code: str = "en") -> list[dict[str, Any]]:
        """Extract entities using the best available model for the language."""
        entities = []

        # Try transformer-based multilingual NER first (works for all languages)
        if self.ner_pipeline:
            try:
                transformer_entities = self._extract_with_transformer(text)
                entities.extend(transformer_entities)
            except Exception as e:
                print(f"Transformer NER failed: {e}")

        # Try language-specific spaCy model if available
        if language_code in self.supported_spacy_languages:
            try:
                spacy_entities = self._extract_with_spacy(text, language_code)
                # Merge with transformer entities, removing duplicates
                entities = self._merge_entities(entities, spacy_entities)
            except Exception as e:
                print(f"spaCy NER failed for {language_code}: {e}")

        # Remove duplicates and overlaps
        entities = self._resolve_overlaps(entities)

        # Sort by position
        entities.sort(key=lambda x: x["start"])

        return entities

    def _extract_with_transformer(self, text: str) -> list[dict[str, Any]]:
        """Extract entities using transformer model (multilingual)."""
        if not self.ner_pipeline:
            return []

        entities = []
        results = self.ner_pipeline(text)

        for result in results:
            entity_type = result["entity_group"]
            # Map to standard entity types
            standard_type = self.entity_type_mapping.get(entity_type, entity_type)

            entities.append(
                {
                    "text": result["word"],
                    "type": standard_type,
                    "start": result["start"],
                    "end": result["end"],
                    "confidence": result["score"],
                    "source": "transformer",
                }
            )

        return entities

    def _extract_with_spacy(self, text: str, language_code: str) -> list[dict[str, Any]]:
        """Extract entities using language-specific spaCy model."""
        # Lazy load the spaCy model
        if language_code not in self.spacy_models:
            model_name = self.supported_spacy_languages[language_code]
            try:
                self.spacy_models[language_code] = spacy.load(model_name)
            except OSError:
                # Model not installed, skip
                return []

        nlp = self.spacy_models[language_code]
        doc = nlp(text)

        entities = []
        for ent in doc.ents:
            # Map to standard entity types
            standard_type = self.entity_type_mapping.get(ent.label_, ent.label_)

            entities.append(
                {
                    "text": ent.text,
                    "type": standard_type,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "confidence": 0.85,
                    "source": "spacy",
                    "original_label": ent.label_,
                }
            )

        return entities

    def _merge_entities(self, entities1: list[dict[str, Any]], entities2: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merge two lists of entities, removing duplicates and keeping higher confidence ones."""
        # Create a map of positions to entities
        merged = []
        seen_positions = set()

        # Sort both lists by confidence
        all_entities = sorted(entities1 + entities2, key=lambda x: x["confidence"], reverse=True)

        for entity in all_entities:
            pos_key = (entity["start"], entity["end"])
            if pos_key not in seen_positions:
                merged.append(entity)
                seen_positions.add(pos_key)

        return merged

    def _resolve_overlaps(self, entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove overlapping entities, keeping the one with higher confidence."""
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
        """Clean up resources."""
        # Unload spaCy models
        for model in self.spacy_models.values():
            del model
        self.spacy_models.clear()

        # Unload transformer model
        if self.ner_pipeline:
            del self.ner_pipeline
            del self.model
            del self.tokenizer

        # Clear CUDA cache if using GPU
        if self.device == "cuda":
            torch.cuda.empty_cache()
