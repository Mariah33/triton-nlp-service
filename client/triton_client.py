"""Triton Client for NLP Service.

Tests all functionality: transliteration, translation, NER, and data type detection
"""

import argparse
import json
import logging

import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TritonNLPClient:
    """Client for Triton NLP Service with comprehensive NLP capabilities."""

    def __init__(self, url: str = "localhost:8001", protocol: str = "grpc") -> None:
        """Initialize Triton client.

        Args:
            url: Triton server URL
            protocol: 'grpc' or 'http'
        """
        self.protocol = protocol
        if protocol == "grpc":
            self.client = grpcclient.InferenceServerClient(url=url)
        else:
            self.client = httpclient.InferenceServerClient(url=url)

        # Check if server is live
        if not self.client.is_server_live():
            msg = f"Triton server at {url} is not live"
            raise ConnectionError(msg)

        logger.info(f"Connected to Triton server at {url}")

        # Check model status
        self.check_models()

    def check_models(self) -> None:
        """Check if all required models are loaded."""
        required_models = [
            "preprocessing",
            "data_type_detector",
            "ner",
            "transliteration",
            "translation",
            "postprocessing",
            "ensemble_nlp",
        ]

        for model in required_models:
            if self.client.is_model_ready(model):
                logger.info(f"✓ Model '{model}' is ready")
            else:
                logger.warning(f"✗ Model '{model}' is not ready")

    def process_text(
        self,
        text: str,
        services: list[str] | None = None,
        source_language: str = "auto",
        target_language: str = "en",
    ) -> dict:
        """Process text through the ensemble NLP pipeline.

        Args:
            text: Input text to process
            services: List of services to apply ['data_type', 'ner', 'transliteration', 'translation']
            source_language: Source language code
            target_language: Target language code

        Returns:
            Dictionary with processing results
        """
        # Prepare inputs
        text_input = self._prepare_string_input("text", [text])

        inputs = [text_input]

        # Add optional inputs
        if services:
            services_str = ",".join(services)
            services_input = self._prepare_string_input("services", [services_str])
            inputs.append(services_input)

        if source_language:
            source_input = self._prepare_string_input("source_language", [source_language])
            inputs.append(source_input)

        if target_language:
            target_input = self._prepare_string_input("target_language", [target_language])
            inputs.append(target_input)

        # Prepare outputs
        outputs = self._prepare_outputs(["result"])

        # Run inference
        response = self.client.infer(model_name="ensemble_nlp", inputs=inputs, outputs=outputs)

        # Parse response
        result = self._parse_string_output(response, "result")[0]
        return json.loads(result)

    def detect_data_type(self, text: str) -> dict:
        """Detect data type of text.

        Args:
            text: Input text

        Returns:
            Data type detection results
        """
        # Prepare inputs
        inputs = [self._prepare_string_input("text", [text])]
        outputs = self._prepare_outputs(["detection_result"])

        # Run inference
        response = self.client.infer(model_name="data_type_detector", inputs=inputs, outputs=outputs)

        # Parse response
        result = self._parse_string_output(response, "detection_result")[0]
        return json.loads(result)

    def extract_entities(self, text: str) -> dict:
        """Extract named entities from text.

        Args:
            text: Input text

        Returns:
            Named entity recognition results
        """
        # Prepare inputs
        inputs = [self._prepare_string_input("text", [text])]
        outputs = self._prepare_outputs(["entities"])

        # Run inference
        response = self.client.infer(model_name="ner", inputs=inputs, outputs=outputs)

        # Parse response
        result = self._parse_string_output(response, "entities")[0]
        return json.loads(result)

    def transliterate(self, text: str, source_script: str = "auto", target_script: str = "latin") -> dict:
        """Transliterate text between scripts.

        Args:
            text: Input text
            source_script: Source script (auto-detect if not specified)
            target_script: Target script

        Returns:
            Transliteration results
        """
        # Prepare inputs
        inputs = [
            self._prepare_string_input("text", [text]),
            self._prepare_string_input("source_script", [source_script]),
            self._prepare_string_input("target_script", [target_script]),
        ]
        outputs = self._prepare_outputs(["transliterated_text"])

        # Run inference
        response = self.client.infer(model_name="transliteration", inputs=inputs, outputs=outputs)

        # Parse response
        result = self._parse_string_output(response, "transliterated_text")[0]
        return json.loads(result)

    def translate(self, text: str, source_language: str = "auto", target_language: str = "en") -> dict:
        """Translate text between languages.

        Args:
            text: Input text
            source_language: Source language code
            target_language: Target language code

        Returns:
            Translation results
        """
        # Prepare inputs
        inputs = [
            self._prepare_string_input("text", [text]),
            self._prepare_string_input("source_language", [source_language]),
            self._prepare_string_input("target_language", [target_language]),
        ]
        outputs = self._prepare_outputs(["translated_text"])

        # Run inference
        response = self.client.infer(model_name="translation", inputs=inputs, outputs=outputs)

        # Parse response
        result = self._parse_string_output(response, "translated_text")[0]
        return json.loads(result)

    def _prepare_string_input(self, name: str, values: list[str]) -> grpcclient.InferInput | httpclient.InferInput:
        """Prepare string input tensor."""
        values_bytes = [v.encode("utf-8") for v in values]
        values_np = np.array(values_bytes, dtype=np.object_)
        values_np = values_np.reshape((len(values), 1))

        if self.protocol == "grpc":
            input_tensor = grpcclient.InferInput(name, values_np.shape, "BYTES")
        else:
            input_tensor = httpclient.InferInput(name, values_np.shape, "BYTES")

        input_tensor.set_data_from_numpy(values_np)
        return input_tensor

    def _prepare_outputs(self, names: list[str]) -> list[grpcclient.InferRequestedOutput] | list[httpclient.InferRequestedOutput]:
        """Prepare output tensors."""
        outputs = []
        for name in names:
            if self.protocol == "grpc":
                outputs.append(grpcclient.InferRequestedOutput(name))
            else:
                outputs.append(httpclient.InferRequestedOutput(name))
        return outputs

    def _parse_string_output(self, response: grpcclient.InferResult | httpclient.InferResult, name: str) -> list[str]:
        """Parse string output from response."""
        output = response.as_numpy(name)
        return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in output.flatten()]


def run_tests() -> None:
    """Run comprehensive tests of all NLP services."""
    client = TritonNLPClient()

    logger.info("\n" + "="*60)
    logger.info("TESTING NLP SERVICES")
    logger.info("="*60)

    # Test cases
    test_cases = [
        {"text": "john.smith@example.com", "description": "Email detection"},
        {"text": "+1-555-123-4567", "description": "Phone number detection"},
        {"text": "GB12345678", "description": "British passport detection"},
        {"text": "4532-1234-5678-9012", "description": "Credit card detection"},
        {
            "text": "Apple Inc. was founded by Steve Jobs in Cupertino on April 1, 1976.",
            "description": "Named entity recognition",
        },
        {"text": "नमस्ते दुनिया", "description": "Hindi transliteration"},
        {"text": "مرحبا بالعالم", "description": "Arabic transliteration"},
        {
            "text": "Hello world",
            "description": "English to Spanish translation",
            "target_language": "es",
        },
        {
            "text": "The meeting is scheduled for Monday, December 25, 2023 at 3:30 PM",
            "description": "Date and time extraction",
        },
        {
            "text": "Contact John Smith at +44-20-7123-4567 or email john.smith@company.co.uk",
            "description": "Multiple entity types",
        },
    ]

    for i, test in enumerate(test_cases, 1):
        logger.info(f"\nTest {i}: {test['description']}")
        logger.info(f"Input: {test['text']}")
        logger.info("-" * 40)

        try:
            # Test individual services

            # 1. Data Type Detection
            logger.info("\n1. Data Type Detection:")
            dt_result = client.detect_data_type(test["text"])
            logger.info(f"   Type: {dt_result.get('primary_type', 'unknown')}")
            logger.info(f"   Confidence: {dt_result.get('confidence', 0):.2f}")
            if "detections" in dt_result:
                for detection in dt_result["detections"][:3]:  # Show first 3
                    logger.info(f"   - {detection.get('type')}: {detection.get('confidence', 0):.2f}")

            # 2. Named Entity Recognition
            logger.info("\n2. Named Entity Recognition:")
            ner_result = client.extract_entities(test["text"])
            logger.info(f"   Found {ner_result.get('entity_count', 0)} entities")
            if "entities" in ner_result:
                for entity in ner_result["entities"][:5]:  # Show first 5
                    logger.info(f"   - {entity.get('type')}: {entity.get('text')}")

            # 3. Transliteration (if non-Latin text)
            if not all(ord(c) < 128 for c in test["text"] if c.isalpha()):
                logger.info("\n3. Transliteration:")
                trans_result = client.transliterate(test["text"])
                logger.info(f"   Script: {trans_result.get('source_script')} → {trans_result.get('target_script')}")
                logger.info(f"   Result: {trans_result.get('transliterated', 'N/A')}")

            # 4. Translation (if specified)
            if "target_language" in test:
                logger.info("\n4. Translation:")
                translate_result = client.translate(
                    test["text"],
                    source_language="auto",
                    target_language=test["target_language"],
                )
                logger.info(f"   Languages: {translate_result.get('source_language')} → {translate_result.get('target_language')}")
                logger.info(f"   Result: {translate_result.get('translated', 'N/A')}")

            # 5. Test ensemble with all services
            logger.info("\n5. Ensemble (All Services):")
            ensemble_result = client.process_text(
                test["text"],
                services=["data_type", "ner", "transliteration", "translation"],
                target_language=test.get("target_language", "en"),
            )
            logger.info(f"   Summary: {len(ensemble_result.get('summary', {}).get('key_findings', []))} key findings")
            for finding in ensemble_result.get("summary", {}).get("key_findings", []):
                logger.info(f"   - {finding}")

        except Exception as e:
            logger.error(f"   ERROR: {e!s}")

        logger.info("-" * 40)

    logger.info("\n" + "="*60)
    logger.info("TESTING COMPLETE")
    logger.info("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton NLP Service Client")
    parser.add_argument("--server", default="localhost:8001", help="Triton server URL")
    parser.add_argument("--protocol", default="grpc", choices=["grpc", "http"], help="Protocol to use")
    parser.add_argument("--test", action="store_true", help="Run test suite")
    parser.add_argument("--text", type=str, help="Text to process")
    parser.add_argument(
        "--services",
        nargs="+",
        help="Services to apply",
        choices=["data_type", "ner", "transliteration", "translation"],
    )
    parser.add_argument("--source-lang", default="auto", help="Source language")
    parser.add_argument("--target-lang", default="en", help="Target language")

    args = parser.parse_args()

    if args.test:
        run_tests()
    elif args.text:
        client = TritonNLPClient(url=args.server, protocol=args.protocol)
        result = client.process_text(
            args.text,
            services=args.services,
            source_language=args.source_lang,
            target_language=args.target_lang,
        )
        logger.info(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        logger.info("Please specify --test or --text")
