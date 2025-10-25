"""
ML-based Data Type Detection Model using Transformers and Pre-trained Models
This version uses actual machine learning models for robust detection
"""

from datetime import datetime
import json
import re
from typing import Any, Dict, List

import numpy as np
import phonenumbers
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """ML-based data type detection model"""

    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize transformer models for PII detection
        print("Loading PII detection models...")

        # Microsoft's DeBERTa model for PII detection
        # This model can detect various PII types
        try:
            self.pii_tokenizer = AutoTokenizer.from_pretrained("lakshyakh93/deberta_finetuned_pii")
            self.pii_model = AutoModelForSequenceClassification.from_pretrained(
                "lakshyakh93/deberta_finetuned_pii"
            ).to(self.device)
            self.pii_model.eval()
            print("Loaded DeBERTa PII detection model")
        except:
            print("Could not load DeBERTa PII model, using fallback")
            self.pii_model = None

        # Google's Universal Sentence Encoder for similarity-based detection
        # This helps identify data types by semantic similarity
        try:
            from sentence_transformers import SentenceTransformer

            self.sentence_encoder = SentenceTransformer("all-MiniLM-L6-v2")
            self.sentence_encoder.to(self.device)
            print("Loaded sentence encoder for similarity matching")

            # Pre-compute embeddings for known data type examples
            self._initialize_reference_embeddings()
        except:
            print("Could not load sentence transformer")
            self.sentence_encoder = None

        # Hugging Face pipeline for token classification (NER-style)
        # This can detect entities that correspond to data types
        try:
            self.token_classifier = pipeline(
                "token-classification",
                model="dslim/bert-base-NER",
                device=0 if torch.cuda.is_available() else -1,
            )
            print("Loaded BERT NER for entity-based data type detection")
        except:
            print("Could not load BERT NER model")
            self.token_classifier = None

        # Load specialized models for specific data types
        self._initialize_specialized_models()

        # Initialize zero-shot classification for general data types
        try:
            from transformers import pipeline

            self.zero_shot_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if torch.cuda.is_available() else -1,
            )
            print("Loaded zero-shot classifier for data type detection")
        except:
            print("Could not load zero-shot classifier")
            self.zero_shot_classifier = None

        # Define data type labels for zero-shot classification
        self.data_type_labels = [
            "email address",
            "phone number",
            "credit card number",
            "social security number",
            "passport number",
            "driver license",
            "bank account number",
            "postal address",
            "person name",
            "date of birth",
            "medical record number",
            "vehicle identification number",
            "IP address",
            "URL or website",
            "username or user ID",
            "password or secret",
            "API key or token",
            "cryptocurrency address",
            "IBAN number",
            "tax identification number",
            "employee ID",
            "student ID",
            "insurance policy number",
            "tracking number",
            "invoice number",
            "order number",
            "reference number",
            "general text",
            "numeric value",
            "alphanumeric code",
        ]

    def _initialize_reference_embeddings(self):
        """Initialize reference embeddings for similarity-based detection"""
        if not self.sentence_encoder:
            return

        # Reference examples for each data type
        self.reference_examples = {
            "email": ["john.doe@example.com", "admin@company.org", "support@service.net"],
            "phone": ["+1-555-123-4567", "(555) 123-4567", "555.123.4567"],
            "ssn": ["123-45-6789", "987-65-4321"],
            "credit_card": ["4532-1234-5678-9012", "5412 7512 3412 3456", "4916338506082832"],
            "passport": ["GB1234567", "US12345678", "A12345678"],
            "address": [
                "123 Main Street, New York, NY 10001",
                "456 Oak Avenue, Los Angeles, CA 90001",
                "789 Park Lane, Chicago, IL 60601",
            ],
            "iban": [
                "GB82 WEST 1234 5698 7654 32",
                "DE89 3704 0044 0532 0130 00",
                "FR14 2004 1010 0505 0001 3M02 606",
            ],
            "url": ["https://www.example.com", "http://subdomain.site.org/path", "www.company.net"],
            "ip": ["192.168.1.1", "10.0.0.1", "2001:0db8:85a3:0000:0000:8a2e:0370:7334"],
            "bitcoin": ["1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa", "3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy"],
            "ethereum": [
                "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
                "0x5aAeb6053f3E94C9b9A09f33669435E7Ef1BeAed",
            ],
            "date": ["2023-12-25", "01/15/2024", "December 25, 2023"],
            "vin": ["1HGCM82633A123456", "WBAVD13516KX00000"],
        }

        # Compute embeddings for reference examples
        self.reference_embeddings = {}
        for dtype, examples in self.reference_examples.items():
            embeddings = self.sentence_encoder.encode(
                examples, convert_to_tensor=True, device=self.device
            )
            self.reference_embeddings[dtype] = embeddings.mean(dim=0)  # Average embedding

    def _initialize_specialized_models(self):
        """Initialize specialized models for specific data types"""

        # Credit card detection using specialized model
        try:
            from transformers import pipeline

            self.credit_card_detector = pipeline(
                "text-classification",
                model="philomath-1209/credit-card-detection",
                device=0 if torch.cuda.is_available() else -1,
            )
            print("Loaded specialized credit card detector")
        except:
            self.credit_card_detector = None

        # PII detection using Microsoft Presidio (if available)
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_analyzer.nlp_engine import NlpEngineProvider

            # Create NLP engine provider
            provider = NlpEngineProvider(
                nlp_configuration={
                    "nlp_engine_name": "spacy",
                    "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
                }
            )

            # Create analyzer engine
            self.presidio_analyzer = AnalyzerEngine(
                nlp_engine=provider.create_engine(), supported_languages=["en"]
            )
            print("Loaded Microsoft Presidio for PII detection")
        except:
            print("Presidio not available, using alternative methods")
            self.presidio_analyzer = None

        # Email validation using specialized model
        try:
            from email_validator import EmailNotValidError, validate_email

            self.email_validator = validate_email
            print("Loaded email validator")
        except:
            self.email_validator = None

    def execute(self, requests):
        responses = []

        for request in requests:
            # Get input tensor
            in_0 = pb_utils.get_input_tensor_by_name(request, "text")
            texts = in_0.as_numpy().tolist()

            detection_results = []

            for text_bytes in texts:
                if isinstance(text_bytes, bytes):
                    text = text_bytes.decode("utf-8")
                else:
                    text = str(text_bytes)

                # Detect data types using ML models
                result = self._detect_data_types_ml(text)
                detection_results.append(json.dumps(result))

            # Create output tensor
            out_tensor = pb_utils.Tensor(
                "detection_result", np.array(detection_results, dtype=np.object_)
            )

            # Create response
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(inference_response)

        return responses

    def _detect_data_types_ml(self, text: str) -> Dict[str, Any]:
        """Detect data types using ML models"""
        text = text.strip()
        detections = []

        # 1. Use Presidio for comprehensive PII detection if available
        if self.presidio_analyzer:
            try:
                presidio_results = self.presidio_analyzer.analyze(text=text, language="en")
                for result in presidio_results:
                    detections.append(
                        {
                            "type": result.entity_type.lower().replace("_", " "),
                            "confidence": result.score,
                            "value": text[result.start : result.end],
                            "start": result.start,
                            "end": result.end,
                            "method": "presidio",
                            "category": self._get_category(result.entity_type),
                        }
                    )
            except Exception as e:
                print(f"Presidio error: {e}")

        # 2. Use zero-shot classification for general data type detection
        if self.zero_shot_classifier and not detections:
            try:
                zs_result = self.zero_shot_classifier(
                    text, candidate_labels=self.data_type_labels, multi_label=False
                )

                # Get top predictions
                for label, score in zip(zs_result["labels"][:3], zs_result["scores"][:3]):
                    if score > 0.5:  # Confidence threshold
                        detections.append(
                            {
                                "type": label,
                                "confidence": float(score),
                                "value": text,
                                "method": "zero_shot",
                                "category": self._get_category_from_label(label),
                            }
                        )
            except Exception as e:
                print(f"Zero-shot error: {e}")

        # 3. Use DeBERTa PII model if available
        if self.pii_model and self.pii_tokenizer:
            try:
                inputs = self.pii_tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=512
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.pii_model(**inputs)
                    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

                    # Get predicted class and confidence
                    confidence, predicted_class = torch.max(predictions, dim=-1)

                    if confidence.item() > 0.7:  # Confidence threshold
                        # Map class to PII type (model-specific)
                        pii_types = ["non_pii", "email", "phone", "ssn", "credit_card", "address"]
                        if predicted_class.item() < len(pii_types):
                            pii_type = pii_types[predicted_class.item()]
                            if pii_type != "non_pii":
                                detections.append(
                                    {
                                        "type": pii_type,
                                        "confidence": confidence.item(),
                                        "value": text,
                                        "method": "deberta_pii",
                                        "category": "pii",
                                    }
                                )
            except Exception as e:
                print(f"DeBERTa error: {e}")

        # 4. Use similarity matching with sentence embeddings
        if self.sentence_encoder and hasattr(self, "reference_embeddings"):
            try:
                text_embedding = self.sentence_encoder.encode(
                    text, convert_to_tensor=True, device=self.device
                )

                # Calculate similarities with reference embeddings
                similarities = {}
                for dtype, ref_embedding in self.reference_embeddings.items():
                    similarity = torch.nn.functional.cosine_similarity(
                        text_embedding.unsqueeze(0), ref_embedding.unsqueeze(0)
                    ).item()
                    similarities[dtype] = similarity

                # Get best match if similarity is high enough
                best_match = max(similarities, key=similarities.get)
                best_score = similarities[best_match]

                if best_score > 0.7:  # Similarity threshold
                    detections.append(
                        {
                            "type": best_match,
                            "confidence": best_score,
                            "value": text,
                            "method": "similarity",
                            "category": self._get_category(best_match),
                        }
                    )
            except Exception as e:
                print(f"Similarity matching error: {e}")

        # 5. Use token classification (NER) for entity-based detection
        if self.token_classifier:
            try:
                ner_results = self.token_classifier(text)
                for entity in ner_results:
                    # Map NER labels to data types
                    entity_type = entity["entity"].replace("B-", "").replace("I-", "")
                    if entity_type in ["PER", "PERSON"]:
                        dtype = "person_name"
                    elif entity_type in ["LOC", "LOCATION"]:
                        dtype = "address"
                    elif entity_type in ["ORG", "ORGANIZATION"]:
                        dtype = "organization"
                    else:
                        continue

                    detections.append(
                        {
                            "type": dtype,
                            "confidence": entity["score"],
                            "value": entity["word"],
                            "start": entity["start"],
                            "end": entity["end"],
                            "method": "ner",
                            "category": "entity",
                        }
                    )
            except Exception as e:
                print(f"NER error: {e}")

        # 6. Try phone number detection with phonenumbers library
        try:
            phone_result = self._detect_phone_with_library(text)
            if phone_result:
                detections.append(phone_result)
        except:
            pass

        # 7. Use specialized credit card detector if available
        if self.credit_card_detector:
            try:
                cc_result = self.credit_card_detector(text)
                if cc_result and cc_result[0]["label"] == "CREDIT_CARD":
                    detections.append(
                        {
                            "type": "credit_card",
                            "confidence": cc_result[0]["score"],
                            "value": self._mask_sensitive_data(text, "credit_card"),
                            "method": "specialized_model",
                            "category": "financial",
                        }
                    )
            except:
                pass

        # Remove duplicates and keep highest confidence for each type
        detections = self._deduplicate_detections(detections)

        # If no ML detections, fall back to pattern matching with lower confidence
        if not detections:
            pattern_result = self._fallback_pattern_detection(text)
            if pattern_result:
                detections.append(pattern_result)

        # Prepare final result
        return {
            "text": text,
            "detections": detections,
            "primary_type": detections[0]["type"] if detections else "unknown",
            "confidence": detections[0]["confidence"] if detections else 0.0,
            "ml_models_used": self._get_available_models(),
        }

    def _detect_phone_with_library(self, text: str) -> Dict[str, Any]:
        """Detect phone numbers using phonenumbers library"""
        try:
            # Try to parse with country code
            if text.startswith("+"):
                parsed = phonenumbers.parse(text, None)
            else:
                # Try common formats for different countries
                for country in ["US", "GB", "CA", "AU", "IN", "DE", "FR"]:
                    try:
                        parsed = phonenumbers.parse(text, country)
                        if phonenumbers.is_valid_number(parsed):
                            return {
                                "type": "phone_number",
                                "country": country,
                                "international": phonenumbers.format_number(
                                    parsed, phonenumbers.PhoneNumberFormat.INTERNATIONAL
                                ),
                                "confidence": 0.95,
                                "value": text,
                                "method": "phonenumbers_library",
                                "category": "contact",
                            }
                    except:
                        continue

            if phonenumbers.is_valid_number(parsed):
                return {
                    "type": "phone_number",
                    "international": phonenumbers.format_number(
                        parsed, phonenumbers.PhoneNumberFormat.INTERNATIONAL
                    ),
                    "confidence": 0.95,
                    "value": text,
                    "method": "phonenumbers_library",
                    "category": "contact",
                }
        except:
            pass
        return None

    def _get_category(self, entity_type: str) -> str:
        """Map entity type to category"""
        categories = {
            "email": "contact",
            "phone": "contact",
            "phone_number": "contact",
            "credit_card": "financial",
            "credit_card_number": "financial",
            "iban": "financial",
            "bank_account": "financial",
            "ssn": "government_id",
            "social_security_number": "government_id",
            "passport": "document",
            "passport_number": "document",
            "driver_license": "document",
            "address": "location",
            "location": "location",
            "person": "personal",
            "person_name": "personal",
            "date_of_birth": "personal",
            "medical_record_number": "medical",
            "ip_address": "technical",
            "url": "technical",
            "bitcoin": "cryptocurrency",
            "ethereum": "cryptocurrency",
            "vin": "vehicle",
        }

        entity_lower = entity_type.lower()
        for key, category in categories.items():
            if key in entity_lower:
                return category
        return "other"

    def _get_category_from_label(self, label: str) -> str:
        """Get category from zero-shot label"""
        return self._get_category(label)

    def _mask_sensitive_data(self, text: str, dtype: str) -> str:
        """Mask sensitive parts of detected data"""
        if dtype == "credit_card":
            # Keep only last 4 digits
            clean = re.sub(r"\D", "", text)
            if len(clean) >= 12:
                return f"****-****-****-{clean[-4:]}"
        elif dtype == "ssn":
            # Mask first 5 digits
            clean = re.sub(r"\D", "", text)
            if len(clean) == 9:
                return f"***-**-{clean[-4:]}"
        return text

    def _deduplicate_detections(self, detections: List[Dict]) -> List[Dict]:
        """Remove duplicate detections, keeping highest confidence"""
        if not detections:
            return detections

        # Group by type
        by_type = {}
        for det in detections:
            dtype = det["type"]
            if dtype not in by_type or det["confidence"] > by_type[dtype]["confidence"]:
                by_type[dtype] = det

        # Return sorted by confidence
        return sorted(by_type.values(), key=lambda x: x["confidence"], reverse=True)

    def _fallback_pattern_detection(self, text: str) -> Dict[str, Any]:
        """Basic pattern matching as fallback"""
        # Simple patterns for common data types
        patterns = {
            "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            "url": r"^https?://.*",
            "ipv4": r"^(?:\d{1,3}\.){3}\d{1,3}$",
            "date": r"^\d{4}-\d{2}-\d{2}$",
            "uuid": r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        }

        for dtype, pattern in patterns.items():
            if re.match(pattern, text, re.IGNORECASE):
                return {
                    "type": dtype,
                    "confidence": 0.5,  # Lower confidence for pattern matching
                    "value": text,
                    "method": "pattern_fallback",
                    "category": self._get_category(dtype),
                }

        # Check if numeric
        if text.replace(".", "").replace(",", "").replace("-", "").isdigit():
            return {
                "type": "number",
                "confidence": 0.6,
                "value": text,
                "method": "pattern_fallback",
                "category": "numeric",
            }

        return None

    def _get_available_models(self) -> List[str]:
        """Return list of available ML models"""
        models = []
        if self.presidio_analyzer:
            models.append("presidio")
        if self.zero_shot_classifier:
            models.append("zero_shot_bart")
        if self.pii_model:
            models.append("deberta_pii")
        if self.sentence_encoder:
            models.append("sentence_similarity")
        if self.token_classifier:
            models.append("bert_ner")
        if self.credit_card_detector:
            models.append("credit_card_specialized")
        return models

    def finalize(self):
        """Clean up models"""
        self.pii_model = None
        self.sentence_encoder = None
        self.token_classifier = None
        self.zero_shot_classifier = None
        self.credit_card_detector = None
        self.presidio_analyzer = None
