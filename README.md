# Triton NLP Service

A comprehensive NLP service using NVIDIA Triton Inference Server providing:
- Transliteration (script conversion)
- Translation (language translation)
- Named Entity Recognition (NER)
- Data Type Detection (phone numbers, passports, emails, etc.)

## Architecture

```
Client Request
    ↓
Triton Ensemble Pipeline
    ↓
┌──────────────────────────────────────┐
│  Preprocessing (text normalization)  │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│  Parallel Model Execution:           │
│  - Transliteration Model             │
│  - Translation Model                 │
│  - NER Model                         │
│  - Data Type Detection Model         │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│  Postprocessing (result aggregation) │
└──────────────────────────────────────┘
    ↓
Response
```

## Quick Start

1. Build the Docker image:
```bash
docker build -t triton-nlp-service:latest .
```

2. Start Triton Server:
```bash
docker-compose up -d
```

3. Test the service:
```bash
python client/test_client.py
```

## Project Structure

```
triton-nlp-service/
├── model_repository/           # Triton model repository
│   ├── ensemble_nlp/         # Ensemble model
│   ├── preprocessing/        # Text preprocessing
│   ├── transliteration/      # Transliteration model
│   ├── translation/          # Translation model
│   ├── ner/                  # NER model
│   ├── data_type_detector/   # Data type detection
│   └── postprocessing/        # Result aggregation
├── models/                    # Model preparation scripts
├── client/                    # Client examples
├── deployment/                # Kubernetes/Helm configs
└── tests/                     # Test suite
```
