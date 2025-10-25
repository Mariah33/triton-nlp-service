"""Preprocessing model for text normalization."""

import json
import re
from typing import Any

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Preprocessing model for text normalization and preparation.."""

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.."""
        self.model_config = model_config = json.loads(args["model_config"])

        # Get output configurations
        output0_config = pb_utils.get_output_config_by_name(model_config, "preprocessed_text")
        output1_config = pb_utils.get_output_config_by_name(model_config, "text_metadata")

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])
        self.output1_dtype = pb_utils.triton_string_to_numpy(output1_config["data_type"])

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`.

        function receives a list of pb_utils.InferenceRequest as the only
        argument.
        """
        responses = []

        for request in requests:
            # Get input tensor
            in_0 = pb_utils.get_input_tensor_by_name(request, "raw_text")
            raw_texts = in_0.as_numpy().tolist()

            preprocessed_texts = []
            metadata_list = []

            for text_bytes in raw_texts:
                text = text_bytes.decode("utf-8") if isinstance(text_bytes, bytes) else str(text_bytes)

                # Normalize text
                processed_text = self._normalize_text(text)
                metadata = self._extract_metadata(text)

                preprocessed_texts.append(processed_text)
                metadata_list.append(json.dumps(metadata))

            # Create output tensors
            out_tensor_0 = pb_utils.Tensor("preprocessed_text", np.array(preprocessed_texts, dtype=self.output0_dtype))

            out_tensor_1 = pb_utils.Tensor("text_metadata", np.array(metadata_list, dtype=self.output1_dtype))

            # Create response
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0, out_tensor_1])
            responses.append(inference_response)

        return responses

    def _normalize_text(self, text: str) -> str:
        """Normalize text for processing.."""
        # Remove excessive whitespace
        text = " ".join(text.split())

        # Normalize unicode characters
        text = text.strip()

        # Handle special characters for better model processing
        text = re.sub(r"([.!?,;:])", r" \1 ", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _extract_metadata(self, text: str) -> dict[str, Any]:
        """Extract basic metadata from text.."""
        return {
            "original_length": len(text),
            "word_count": len(text.split()),
            "has_numbers": bool(re.search(r"\d", text)),
            "has_special_chars": bool(re.search(r"[^a-zA-Z0-9\s]", text)),
            "detected_scripts": self._detect_scripts(text),
        }

    def _detect_scripts(self, text: str) -> list[str]:
        """Detect writing scripts in text.."""
        scripts = []

        # Check for various scripts
        if re.search(r"[a-zA-Z]", text):
            scripts.append("latin")
        if re.search(r"[\u0600-\u06FF]", text):
            scripts.append("arabic")
        if re.search(r"[\u0900-\u097F]", text):
            scripts.append("devanagari")
        if re.search(r"[\u4E00-\u9FFF]", text):
            scripts.append("chinese")
        if re.search(r"[\u0400-\u04FF]", text):
            scripts.append("cyrillic")
        if re.search(r"[\u1F00-\u1FFF]", text):
            scripts.append("greek")

        return scripts if scripts else ["unknown"]

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.."""
