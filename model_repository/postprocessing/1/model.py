"""Postprocessing Model.

Aggregates results from all NLP models and formats the final response
"""

import json

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Postprocessing model to aggregate and format results."""

    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])

    def execute(self, requests):
        responses = []

        for request in requests:
            # Get all input tensors
            original_text = self._get_string_tensor(request, "original_text")
            preprocessed_text = self._get_string_tensor(request, "preprocessed_text")
            metadata = self._get_string_tensor(request, "metadata")
            data_type_result = self._get_string_tensor(request, "data_type_result")
            ner_result = self._get_string_tensor(request, "ner_result")
            transliteration_result = self._get_string_tensor(request, "transliteration_result")
            translation_result = self._get_string_tensor(request, "translation_result")
            requested_services = self._get_string_tensor(request, "requested_services", optional=True)

            final_results = []

            # Process each batch item
            batch_size = len(original_text)
            for i in range(batch_size):
                # Parse individual results
                metadata_dict = json.loads(metadata[i]) if metadata[i] else {}
                data_type_dict = json.loads(data_type_result[i]) if data_type_result[i] else {}
                ner_dict = json.loads(ner_result[i]) if ner_result[i] else {}
                transliteration_dict = json.loads(transliteration_result[i]) if transliteration_result[i] else {}
                translation_dict = json.loads(translation_result[i]) if translation_result[i] else {}

                # Parse requested services
                services = []
                if requested_services and i < len(requested_services):
                    services_str = requested_services[i]
                    if services_str:
                        services = services_str.split(",") if isinstance(services_str, str) else []

                # If no services specified, include all
                if not services:
                    services = ["data_type", "ner", "transliteration", "translation"]

                # Build final result
                result = {
                    "original_text": original_text[i],
                    "preprocessed_text": preprocessed_text[i],
                    "metadata": metadata_dict,
                    "results": {},
                }

                # Add requested service results
                if "data_type" in services:
                    result["results"]["data_type_detection"] = self._format_data_type_result(data_type_dict)

                if "ner" in services:
                    result["results"]["named_entities"] = self._format_ner_result(ner_dict)

                if "transliteration" in services:
                    result["results"]["transliteration"] = self._format_transliteration_result(transliteration_dict)

                if "translation" in services:
                    result["results"]["translation"] = self._format_translation_result(translation_dict)

                # Add summary
                result["summary"] = self._generate_summary(result)

                final_results.append(json.dumps(result))

            # Create output tensor
            out_tensor = pb_utils.Tensor("final_result", np.array(final_results, dtype=np.object_))

            # Create response
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(inference_response)

        return responses

    def _get_string_tensor(self, request, name: str, optional: bool = False) -> list[str]:
        """Helper to get string tensor values."""
        tensor = pb_utils.get_input_tensor_by_name(request, name)
        if tensor is None:
            if optional:
                return []
            msg = f"Required tensor '{name}' not found"
            raise ValueError(msg)

        values = tensor.as_numpy().tolist()
        return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values]

    def _format_data_type_result(self, data_type_dict: dict) -> dict:
        """Format data type detection results."""
        if not data_type_dict:
            return {"detected": False, "type": "unknown"}

        formatted = {
            "detected": True,
            "primary_type": data_type_dict.get("primary_type", "unknown"),
            "confidence": data_type_dict.get("confidence", 0.0),
            "detections": [],
        }

        for detection in data_type_dict.get("detections", []):
            formatted["detections"].append(
                {
                    "type": detection.get("type"),
                    "subtype": detection.get("subtype", None),
                    "category": detection.get("category"),
                    "confidence": detection.get("confidence"),
                    "value": detection.get("value", "")[:50] + ".." if len(detection.get("value", "")) > 50 else detection.get("value", ""),
                }
            )

        return formatted

    def _format_ner_result(self, ner_dict: dict) -> dict:
        """Format NER results."""
        if not ner_dict:
            return {"entities": [], "count": 0}

        formatted = {
            "entities": [],
            "count": ner_dict.get("entity_count", 0),
            "types_found": ner_dict.get("entity_types", []),
        }

        # Group entities by type
        entities_by_type = {}
        for entity in ner_dict.get("entities", []):
            entity_type = entity.get("type", "UNKNOWN")
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append(
                {
                    "text": entity.get("text"),
                    "confidence": entity.get("confidence", 0.0),
                    "position": [entity.get("start", 0), entity.get("end", 0)],
                }
            )

        formatted["entities_by_type"] = entities_by_type
        formatted["entities"] = ner_dict.get("entities", [])

        return formatted

    def _format_transliteration_result(self, transliteration_dict: dict) -> dict:
        """Format transliteration results."""
        if not transliteration_dict:
            return {"success": False}

        return {
            "success": transliteration_dict.get("confidence", 0) > 0,
            "original": transliteration_dict.get("original", ""),
            "transliterated": transliteration_dict.get("transliterated", ""),
            "source_script": transliteration_dict.get("source_script", "unknown"),
            "target_script": transliteration_dict.get("target_script", "latin"),
            "confidence": transliteration_dict.get("confidence", 0.0),
            "method": transliteration_dict.get("method", "unknown"),
        }

    def _format_translation_result(self, translation_dict: dict) -> dict:
        """Format translation results."""
        if not translation_dict:
            return {"success": False}

        return {
            "success": translation_dict.get("confidence", 0) > 0,
            "original": translation_dict.get("original", ""),
            "translated": translation_dict.get("translated", ""),
            "source_language": translation_dict.get("source_language", "unknown"),
            "target_language": translation_dict.get("target_language", "unknown"),
            "confidence": translation_dict.get("confidence", 0.0),
            "method": translation_dict.get("method", "unknown"),
            "alternatives": translation_dict.get("alternative_translations", []),
        }

    def _generate_summary(self, result: dict) -> dict:
        """Generate a summary of all results."""
        summary = {
            "text_length": len(result["original_text"]),
            "services_applied": list(result["results"].keys()),
            "key_findings": [],
        }

        # Summarize data type detection
        if "data_type_detection" in result["results"]:
            dt_result = result["results"]["data_type_detection"]
            if dt_result["detected"]:
                summary["key_findings"].append(f"Detected as {dt_result['primary_type']} with {dt_result['confidence']:.2f} confidence")

        # Summarize NER
        if "named_entities" in result["results"]:
            ner_result = result["results"]["named_entities"]
            if ner_result["count"] > 0:
                summary["key_findings"].append(f"Found {ner_result['count']} named entities of types: {', '.join(ner_result['types_found'])}")

        # Summarize transliteration
        if "transliteration" in result["results"]:
            trans_result = result["results"]["transliteration"]
            if trans_result["success"]:
                summary["key_findings"].append(f"Transliterated from {trans_result['source_script']} to {trans_result['target_script']}")

        # Summarize translation
        if "translation" in result["results"]:
            trans_result = result["results"]["translation"]
            if trans_result["success"]:
                summary["key_findings"].append(f"Translated from {trans_result['source_language']} to {trans_result['target_language']}")

        return summary

    def finalize(self):
        pass
