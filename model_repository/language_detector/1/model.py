"""Language Detection Model.

Detects the language of input text using multiple detection libraries for accuracy
Supports 50+ languages with confidence scores
"""

import json
from typing import Any

import langdetect
from langdetect import DetectorFactory, detect, detect_langs
import numpy as np
import triton_python_backend_utils as pb_utils

# Set seed for reproducible results
DetectorFactory.seed = 0


class TritonPythonModel:
    """Language detection model using langdetect."""

    def initialize(self, args: dict) -> None:
        """Initialize the model - called once when model is loaded."""
        self.model_config = json.loads(args["model_config"])

        # Language name mapping (ISO 639-1 codes to full names)
        self.language_names = {
            "af": "Afrikaans",
            "ar": "Arabic",
            "bg": "Bulgarian",
            "bn": "Bengali",
            "ca": "Catalan",
            "cs": "Czech",
            "cy": "Welsh",
            "da": "Danish",
            "de": "German",
            "el": "Greek",
            "en": "English",
            "es": "Spanish",
            "et": "Estonian",
            "fa": "Persian",
            "fi": "Finnish",
            "fr": "French",
            "gu": "Gujarati",
            "he": "Hebrew",
            "hi": "Hindi",
            "hr": "Croatian",
            "hu": "Hungarian",
            "id": "Indonesian",
            "it": "Italian",
            "ja": "Japanese",
            "kn": "Kannada",
            "ko": "Korean",
            "lt": "Lithuanian",
            "lv": "Latvian",
            "mk": "Macedonian",
            "ml": "Malayalam",
            "mr": "Marathi",
            "ne": "Nepali",
            "nl": "Dutch",
            "no": "Norwegian",
            "pa": "Punjabi",
            "pl": "Polish",
            "pt": "Portuguese",
            "ro": "Romanian",
            "ru": "Russian",
            "sk": "Slovak",
            "sl": "Slovenian",
            "so": "Somali",
            "sq": "Albanian",
            "sv": "Swedish",
            "sw": "Swahili",
            "ta": "Tamil",
            "te": "Telugu",
            "th": "Thai",
            "tl": "Tagalog",
            "tr": "Turkish",
            "uk": "Ukrainian",
            "ur": "Urdu",
            "vi": "Vietnamese",
            "zh-cn": "Chinese (Simplified)",
            "zh-tw": "Chinese (Traditional)",
        }

    def execute(self, requests: list) -> list:
        """Execute inference requests."""
        responses = []

        for request in requests:
            # Get input tensor
            text_tensor = pb_utils.get_input_tensor_by_name(request, "text")
            texts = text_tensor.as_numpy().tolist()

            detection_results = []

            for text_bytes in texts:
                text = text_bytes.decode("utf-8") if isinstance(text_bytes, bytes) else str(text_bytes)

                # Detect language
                result = self._detect_language(text)
                detection_results.append(json.dumps(result))

            # Create output tensor
            out_tensor = pb_utils.Tensor("language_info", np.array(detection_results, dtype=np.object_))

            # Create response
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(inference_response)

        return responses

    def _detect_language(self, text: str) -> dict[str, Any]:
        """Detect language of text with confidence scores."""
        # Handle empty or very short text
        if not text or len(text.strip()) < 3:
            return {
                "detected_language": "unknown",
                "language_code": "unknown",
                "confidence": 0.0,
                "all_probabilities": [],
                "is_reliable": False,
            }

        try:
            # Get single best language
            lang_code = detect(text)

            # Get all language probabilities
            lang_probs = detect_langs(text)

            # Convert to list of dicts
            all_probabilities = [
                {"language_code": lp.lang, "language_name": self.language_names.get(lp.lang, lp.lang), "probability": lp.prob} for lp in lang_probs
            ]

            # Get the highest probability
            confidence = lang_probs[0].prob if lang_probs else 0.0

            return {
                "detected_language": self.language_names.get(lang_code, lang_code),
                "language_code": lang_code,
                "confidence": confidence,
                "all_probabilities": all_probabilities,
                "is_reliable": confidence > 0.8,
                "text_length": len(text),
            }

        except langdetect.LangDetectException as e:
            # Handle detection failures
            return {
                "detected_language": "unknown",
                "language_code": "unknown",
                "confidence": 0.0,
                "all_probabilities": [],
                "is_reliable": False,
                "error": str(e),
            }

    def finalize(self) -> None:
        """Clean up resources."""
        pass
