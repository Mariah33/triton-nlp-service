"""FastAPI wrapper for Triton NLP Service.

Provides a user-friendly REST API interface
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
from pydantic import BaseModel, Field
import tritonclient.grpc as grpcclient
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Triton NLP Service API",
    description="REST API for NLP services including transliteration, translation, NER, and data type detection",
    version="1.0.0",
)

# Thread pool for async operations
executor = ThreadPoolExecutor(max_workers=10)

# Global Triton client
triton_client = None


class TextRequest(BaseModel):
    text: str = Field(..., description="Input text to process")
    services: list[str] | None = Field(
        default=None,
        description="List of services to apply: data_type, ner, transliteration, translation",
    )
    source_language: str | None = Field(default="auto", description="Source language code")
    target_language: str | None = Field(default="en", description="Target language code")


class BatchTextRequest(BaseModel):
    texts: list[str] = Field(..., description="List of texts to process")
    services: list[str] | None = Field(default=None)
    source_language: str | None = Field(default="auto")
    target_language: str | None = Field(default="en")


class DataTypeRequest(BaseModel):
    text: str = Field(..., description="Text to analyze for data type")


class TransliterationRequest(BaseModel):
    text: str = Field(..., description="Text to transliterate")
    source_script: str | None = Field(default="auto", description="Source script")
    target_script: str | None = Field(default="latin", description="Target script")


class TranslationRequest(BaseModel):
    text: str = Field(..., description="Text to translate")
    source_language: str | None = Field(default="auto", description="Source language")
    target_language: str | None = Field(default="en", description="Target language")


class NERRequest(BaseModel):
    text: str = Field(..., description="Text for entity extraction")


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize Triton client on startup.."""
    global triton_client
    try:
        triton_client = grpcclient.InferenceServerClient(url="localhost:8001")
        if not triton_client.is_server_live():
            msg = "Triton server is not live"
            raise ConnectionError(msg)
        logger.info("Connected to Triton server")
    except Exception as e:
        logger.exception(f"Failed to connect to Triton server: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint with API information.."""
    return {
        "service": "Triton NLP Service",
        "version": "1.0.0",
        "endpoints": [
            "/process",
            "/batch_process",
            "/detect_type",
            "/transliterate",
            "/translate",
            "/extract_entities",
            "/health",
            "/models",
        ],
    }


@app.get("/health")
async def health_check():
    """Health check endpoint.."""
    if triton_client and triton_client.is_server_live():
        return {"status": "healthy", "triton": "connected"}
    raise HTTPException(status_code=503, detail="Triton server not available")


@app.get("/models")
async def list_models():
    """List available models and their status.."""
    models = [
        "preprocessing",
        "data_type_detector",
        "ner",
        "transliteration",
        "translation",
        "postprocessing",
        "ensemble_nlp",
    ]

    model_status = {}
    for model in models:
        try:
            ready = triton_client.is_model_ready(model)
            model_status[model] = "ready" if ready else "not ready"
        except:
            model_status[model] = "error"

    return {"models": model_status}


@app.post("/process")
async def process_text(request: TextRequest):
    """Process text through the NLP pipeline.."""
    try:
        result = await run_in_executor(
            process_with_triton,
            request.text,
            request.services,
            request.source_language,
            request.target_language,
        )
        return JSONResponse(content=result)
    except Exception as e:
        logger.exception(f"Error processing text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_process")
async def batch_process_text(request: BatchTextRequest):
    """Process multiple texts in batch.."""
    try:
        tasks = []
        for text in request.texts:
            task = run_in_executor(
                process_with_triton,
                text,
                request.services,
                request.source_language,
                request.target_language,
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        return JSONResponse(content={"results": results})
    except Exception as e:
        logger.exception(f"Error in batch processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect_type")
async def detect_data_type(request: DataTypeRequest):
    """Detect data type of text.."""
    try:
        result = await run_in_executor(detect_type_with_triton, request.text)
        return JSONResponse(content=result)
    except Exception as e:
        logger.exception(f"Error detecting data type: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transliterate")
async def transliterate_text(request: TransliterationRequest):
    """Transliterate text between scripts.."""
    try:
        result = await run_in_executor(transliterate_with_triton, request.text, request.source_script, request.target_script)
        return JSONResponse(content=result)
    except Exception as e:
        logger.exception(f"Error transliterating text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/translate")
async def translate_text(request: TranslationRequest):
    """Translate text between languages.."""
    try:
        result = await run_in_executor(translate_with_triton, request.text, request.source_language, request.target_language)
        return JSONResponse(content=result)
    except Exception as e:
        logger.exception(f"Error translating text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract_entities")
async def extract_entities(request: NERRequest):
    """Extract named entities from text.."""
    try:
        result = await run_in_executor(extract_entities_with_triton, request.text)
        return JSONResponse(content=result)
    except Exception as e:
        logger.exception(f"Error extracting entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def run_in_executor(func, *args):
    """Run blocking function in executor.."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, func, *args)


def process_with_triton(text: str, services: list[str], source_lang: str, target_lang: str) -> dict:
    """Process text using Triton ensemble model.."""
    # Prepare inputs
    text_input = prepare_string_input("text", [text])
    inputs = [text_input]

    if services:
        services_str = ",".join(services)
        services_input = prepare_string_input("services", [services_str])
        inputs.append(services_input)

    if source_lang:
        source_input = prepare_string_input("source_language", [source_lang])
        inputs.append(source_input)

    if target_lang:
        target_input = prepare_string_input("target_language", [target_lang])
        inputs.append(target_input)

    # Prepare outputs
    outputs = [grpcclient.InferRequestedOutput("result")]

    # Run inference
    response = triton_client.infer(model_name="ensemble_nlp", inputs=inputs, outputs=outputs)

    # Parse response
    result = parse_string_output(response, "result")[0]
    return json.loads(result)


def detect_type_with_triton(text: str) -> dict:
    """Detect data type using Triton.."""
    inputs = [prepare_string_input("text", [text])]
    outputs = [grpcclient.InferRequestedOutput("detection_result")]

    response = triton_client.infer(model_name="data_type_detector", inputs=inputs, outputs=outputs)

    result = parse_string_output(response, "detection_result")[0]
    return json.loads(result)


def transliterate_with_triton(text: str, source_script: str, target_script: str) -> dict:
    """Transliterate text using Triton.."""
    inputs = [
        prepare_string_input("text", [text]),
        prepare_string_input("source_script", [source_script]),
        prepare_string_input("target_script", [target_script]),
    ]
    outputs = [grpcclient.InferRequestedOutput("transliterated_text")]

    response = triton_client.infer(model_name="transliteration", inputs=inputs, outputs=outputs)

    result = parse_string_output(response, "transliterated_text")[0]
    return json.loads(result)


def translate_with_triton(text: str, source_lang: str, target_lang: str) -> dict:
    """Translate text using Triton.."""
    inputs = [
        prepare_string_input("text", [text]),
        prepare_string_input("source_language", [source_lang]),
        prepare_string_input("target_language", [target_lang]),
    ]
    outputs = [grpcclient.InferRequestedOutput("translated_text")]

    response = triton_client.infer(model_name="translation", inputs=inputs, outputs=outputs)

    result = parse_string_output(response, "translated_text")[0]
    return json.loads(result)


def extract_entities_with_triton(text: str) -> dict:
    """Extract entities using Triton.."""
    inputs = [prepare_string_input("text", [text])]
    outputs = [grpcclient.InferRequestedOutput("entities")]

    response = triton_client.infer(model_name="ner", inputs=inputs, outputs=outputs)

    result = parse_string_output(response, "entities")[0]
    return json.loads(result)


def prepare_string_input(name: str, values: list[str]):
    """Prepare string input tensor for Triton.."""
    values_bytes = [v.encode("utf-8") for v in values]
    values_np = np.array(values_bytes, dtype=np.object_)
    values_np = values_np.reshape((len(values), 1))

    input_tensor = grpcclient.InferInput(name, values_np.shape, "BYTES")
    input_tensor.set_data_from_numpy(values_np)
    return input_tensor


def parse_string_output(response, name: str) -> list[str]:
    """Parse string output from Triton response.."""
    output = response.as_numpy(name)
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in output.flatten()]


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
