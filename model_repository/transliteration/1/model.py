"""Transliteration Model using AI4Bharat IndicXlit.

Converts text from one script to another (e.g., Devanagari to Latin)
"""

import json

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Transliteration model for script conversion.."""

    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])

        # Initialize transliterators for various language pairs
        self.transliterators = {}

        # Setup transliteration pairs
        self.language_pairs = {
            "hi_to_en": ("hi", "en"),  # Hindi to English
            "ta_to_en": ("ta", "en"),  # Tamil to English
            "te_to_en": ("te", "en"),  # Telugu to English
            "bn_to_en": ("bn", "en"),  # Bengali to English
            "gu_to_en": ("gu", "en"),  # Gujarati to English
            "mr_to_en": ("mr", "en"),  # Marathi to English
            "pa_to_en": ("pa", "en"),  # Punjabi to English
            "ur_to_en": ("ur", "en"),  # Urdu to English
            "ar_to_en": ("ar", "en"),  # Arabic to English
            "ru_to_en": ("ru", "en"),  # Russian to English
            "zh_to_pinyin": ("zh", "pinyin"),  # Chinese to Pinyin
            "ja_to_romaji": ("ja", "romaji"),  # Japanese to Romaji
        }

        # Simple rule-based transliteration mappings
        # For production, you would load actual ML models here
        self.transliteration_maps = {
            "devanagari_to_latin": {
                "अ": "a",
                "आ": "aa",
                "इ": "i",
                "ई": "ee",
                "उ": "u",
                "ऊ": "oo",
                "ए": "e",
                "ऐ": "ai",
                "ओ": "o",
                "औ": "au",
                "क": "k",
                "ख": "kh",
                "ग": "g",
                "घ": "gh",
                "ङ": "ng",
                "च": "ch",
                "छ": "chh",
                "ज": "j",
                "झ": "jh",
                "ञ": "ny",
                "ट": "t",
                "ठ": "th",
                "ड": "d",
                "ढ": "dh",
                "ण": "n",
                "त": "t",
                "थ": "th",
                "द": "d",
                "ध": "dh",
                "न": "n",
                "प": "p",
                "फ": "ph",
                "ब": "b",
                "भ": "bh",
                "म": "m",
                "य": "y",
                "र": "r",
                "ल": "l",
                "व": "v",
                "श": "sh",
                "ष": "sh",
                "स": "s",
                "ह": "h",
                "ा": "a",
                "ि": "i",
                "ी": "ee",
                "ु": "u",
                "ू": "oo",
                "े": "e",
                "ै": "ai",
                "ो": "o",
                "ौ": "au",
                "ं": "n",
                "ः": "h",
                "्": "",
                "०": "0",
                "१": "1",
                "२": "2",
                "३": "3",
                "४": "4",
                "५": "5",
                "६": "6",
                "७": "7",
                "८": "8",
                "९": "9",
            },
            "arabic_to_latin": {
                "ا": "a",
                "ب": "b",
                "ت": "t",
                "ث": "th",
                "ج": "j",
                "ح": "h",
                "خ": "kh",
                "د": "d",
                "ذ": "dh",
                "ر": "r",
                "ز": "z",
                "س": "s",
                "ش": "sh",
                "ص": "s",
                "ض": "d",
                "ط": "t",
                "ظ": "z",
                "ع": "a",
                "غ": "gh",
                "ف": "f",
                "ق": "q",
                "ك": "k",
                "ل": "l",
                "م": "m",
                "ن": "n",
                "ه": "h",
                "و": "w",
                "ي": "y",
                "٠": "0",
                "١": "1",
                "٢": "2",
                "٣": "3",
                "٤": "4",
                "٥": "5",
                "٦": "6",
                "٧": "7",
                "٨": "8",
                "٩": "9",
            },
            "cyrillic_to_latin": {
                "А": "A",
                "Б": "B",
                "В": "V",
                "Г": "G",
                "Д": "D",
                "Е": "E",
                "Ё": "Yo",
                "Ж": "Zh",
                "З": "Z",
                "И": "I",
                "Й": "Y",
                "К": "K",
                "Л": "L",
                "М": "M",
                "Н": "N",
                "О": "O",
                "П": "P",
                "Р": "R",
                "С": "S",
                "Т": "T",
                "У": "U",
                "Ф": "F",
                "Х": "Kh",
                "Ц": "Ts",
                "Ч": "Ch",
                "Ш": "Sh",
                "Щ": "Shch",
                "Ъ": "",
                "Ы": "Y",
                "Ь": "",
                "Э": "E",
                "Ю": "Yu",
                "Я": "Ya",
                "а": "a",
                "б": "b",
                "в": "v",
                "г": "g",
                "д": "d",
                "е": "e",
                "ё": "yo",
                "ж": "zh",
                "з": "z",
                "и": "i",
                "й": "y",
                "к": "k",
                "л": "l",
                "м": "m",
                "н": "n",
                "о": "o",
                "п": "p",
                "р": "r",
                "с": "s",
                "т": "t",
                "у": "u",
                "ф": "f",
                "х": "kh",
                "ц": "ts",
                "ч": "ch",
                "ш": "sh",
                "щ": "shch",
                "ъ": "",
                "ы": "y",
                "ь": "",
                "э": "e",
                "ю": "yu",
                "я": "ya",
            },
            "greek_to_latin": {
                "Α": "A",
                "Β": "B",
                "Γ": "G",
                "Δ": "D",
                "Ε": "E",
                "Ζ": "Z",
                "Η": "H",
                "Θ": "Th",
                "Ι": "I",
                "Κ": "K",
                "Λ": "L",
                "Μ": "M",
                "Ν": "N",
                "Ξ": "X",
                "Ο": "O",
                "Π": "P",
                "Ρ": "R",
                "Σ": "S",
                "Τ": "T",
                "Υ": "Y",
                "Φ": "Ph",
                "Χ": "Ch",
                "Ψ": "Ps",
                "Ω": "O",
                "α": "a",
                "β": "b",
                "γ": "g",
                "δ": "d",
                "ε": "e",
                "ζ": "z",
                "η": "h",
                "θ": "th",
                "ι": "i",
                "κ": "k",
                "λ": "l",
                "μ": "m",
                "ν": "n",
                "ξ": "x",
                "ο": "o",
                "π": "p",
                "ρ": "r",
                "σ": "s",
                "τ": "t",
                "υ": "y",
                "φ": "ph",
                "χ": "ch",
                "ψ": "ps",
                "ω": "o",
            },
        }

    def execute(self, requests):
        responses = []

        for request in requests:
            # Get input tensors
            text_tensor = pb_utils.get_input_tensor_by_name(request, "text")
            source_lang_tensor = pb_utils.get_input_tensor_by_name(request, "source_script")
            target_lang_tensor = pb_utils.get_input_tensor_by_name(request, "target_script")

            texts = text_tensor.as_numpy().tolist()
            source_scripts = source_lang_tensor.as_numpy().tolist() if source_lang_tensor else ["auto"] * len(texts)
            target_scripts = target_lang_tensor.as_numpy().tolist() if target_lang_tensor else ["latin"] * len(texts)

            transliterated_results = []

            for text_bytes, source, target in zip(texts, source_scripts, target_scripts):
                if isinstance(text_bytes, bytes):
                    text = text_bytes.decode("utf-8")
                else:
                    text = str(text_bytes)

                if isinstance(source, bytes):
                    source = source.decode("utf-8")
                if isinstance(target, bytes):
                    target = target.decode("utf-8")

                # Perform transliteration
                result = self._transliterate(text, source, target)
                transliterated_results.append(json.dumps(result))

            # Create output tensor
            out_tensor = pb_utils.Tensor("transliterated_text", np.array(transliterated_results, dtype=np.object_))

            # Create response
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(inference_response)

        return responses

    def _transliterate(self, text: str, source_script: str, target_script: str) -> Dict:
        """Perform transliteration.."""

        # Auto-detect source script if not specified
        if source_script == "auto":
            source_script = self._detect_script(text)

        # Select appropriate transliteration map
        map_key = f"{source_script}_to_{target_script}"

        transliterated = text
        confidence = 0.0

        # Apply transliteration based on the script pair
        if source_script == "devanagari" and target_script == "latin":
            transliterated = self._apply_mapping(text, self.transliteration_maps["devanagari_to_latin"])
            confidence = 0.85
        elif source_script == "arabic" and target_script == "latin":
            transliterated = self._apply_mapping(text, self.transliteration_maps["arabic_to_latin"])
            confidence = 0.80
        elif source_script == "cyrillic" and target_script == "latin":
            transliterated = self._apply_mapping(text, self.transliteration_maps["cyrillic_to_latin"])
            confidence = 0.90
        elif source_script == "greek" and target_script == "latin":
            transliterated = self._apply_mapping(text, self.transliteration_maps["greek_to_latin"])
            confidence = 0.85
        elif source_script == target_script:
            # Same script, no transliteration needed
            confidence = 1.0
        else:
            # Unsupported pair, return original
            confidence = 0.0

        return {
            "original": text,
            "transliterated": transliterated,
            "source_script": source_script,
            "target_script": target_script,
            "confidence": confidence,
            "method": "rule_based" if confidence > 0 else "unsupported",
        }

    def _detect_script(self, text: str) -> str:
        """Detect the script of the text.."""

        scripts_count = {
            "latin": 0,
            "devanagari": 0,
            "arabic": 0,
            "cyrillic": 0,
            "greek": 0,
            "chinese": 0,
            "japanese": 0,
        }

        for char in text:
            code = ord(char)
            if 0x0041 <= code <= 0x024F:
                scripts_count["latin"] += 1
            elif 0x0900 <= code <= 0x097F:
                scripts_count["devanagari"] += 1
            elif 0x0600 <= code <= 0x06FF:
                scripts_count["arabic"] += 1
            elif 0x0400 <= code <= 0x04FF:
                scripts_count["cyrillic"] += 1
            elif 0x0370 <= code <= 0x03FF or 0x1F00 <= code <= 0x1FFF:
                scripts_count["greek"] += 1
            elif 0x4E00 <= code <= 0x9FFF:
                scripts_count["chinese"] += 1
            elif 0x3040 <= code <= 0x309F or 0x30A0 <= code <= 0x30FF:
                scripts_count["japanese"] += 1

        # Return the script with the most characters
        return max(scripts_count, key=scripts_count.get) if any(scripts_count.values()) else "unknown"

    def _apply_mapping(self, text: str, mapping: dict[str, str]) -> str:
        """Apply character mapping for transliteration.."""

        result = []
        i = 0
        while i < len(text):
            # Try multi-character mappings first (for conjuncts)
            matched = False
            for length in [3, 2, 1]:
                if i + length <= len(text):
                    substr = text[i : i + length]
                    if substr in mapping:
                        result.append(mapping[substr])
                        i += length
                        matched = True
                        break

            if not matched:
                # Keep the original character if no mapping found
                result.append(text[i])
                i += 1

        return "".join(result)

    def finalize(self):
        pass
