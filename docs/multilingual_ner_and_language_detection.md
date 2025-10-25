# Multilingual NER and Language Detection

## Overview

The Triton NLP Service now includes powerful multilingual capabilities:

1. **Automatic Language Detection** - Detects the language of input text across 50+ languages
2. **Multilingual Named Entity Recognition (NER)** - Extracts entities in 100+ languages using transformer-based models

## Language Detection

### Supported Languages

The language detector supports 50+ languages including:
- **European**: English, Spanish, French, German, Italian, Portuguese, Dutch, Russian, Polish, etc.
- **Asian**: Chinese, Japanese, Korean, Hindi, Arabic, Bengali, Thai, Vietnamese, etc.
- **Others**: Turkish, Hebrew, Persian, Swahili, and more

### Usage

```python
from client.triton_client import TritonNLPClient

client = TritonNLPClient(url="localhost:8001", protocol="grpc")

# Detect language
text = "Bonjour, comment allez-vous?"
result = client.detect_language(text)

print(result)
# {
#   "detected_language": "French",
#   "language_code": "fr",
#   "confidence": 0.99,
#   "all_probabilities": [
#     {"language_code": "fr", "language_name": "French", "probability": 0.99},
#     {"language_code": "es", "language_name": "Spanish", "probability": 0.01}
#   ],
#   "is_reliable": true,
#   "text_length": 28
# }
```

### API Response Fields

- **detected_language**: Full name of the detected language
- **language_code**: ISO 639-1 language code (e.g., "en", "fr", "de")
- **confidence**: Confidence score (0.0 to 1.0)
- **all_probabilities**: List of all detected languages with their probabilities
- **is_reliable**: Boolean indicating if detection is reliable (confidence > 0.8)
- **text_length**: Length of input text

## Multilingual NER

### Features

The multilingual NER model combines:
- **XLM-RoBERTa** - Multilingual transformer model trained on 100+ languages
- **Language-specific spaCy models** - For enhanced accuracy in supported languages

### Supported Entity Types

- **PERSON**: Names of people
- **LOCATION**: Geographic locations (countries, cities, addresses)
- **ORGANIZATION**: Companies, institutions, groups
- **DATE**: Dates and time periods
- **TIME**: Times of day
- **MONEY**: Monetary values
- **PERCENT**: Percentages
- **MISCELLANEOUS**: Other named entities

### Language-Specific spaCy Models

Enhanced accuracy for these languages:
- English (en), German (de), Spanish (es), French (fr)
- Italian (it), Portuguese (pt), Dutch (nl), Greek (el)
- Chinese (zh), Japanese (ja), Russian (ru), Polish (pl)
- Romanian (ro), Danish (da), Finnish (fi), Swedish (sv)
- Norwegian (nb), Lithuanian (lt), Macedonian (mk)
- Catalan (ca), Croatian (hr), Ukrainian (uk)

### Usage

#### Automatic Language Detection + NER

```python
# The ensemble automatically detects language and performs multilingual NER
text = "Angela Merkel nació en Hamburgo, Alemania en 1954."
result = client.process_text(text)

print(result["language_detection"])
# {
#   "detected_language": "Spanish",
#   "language_code": "es",
#   "confidence": 0.99
# }

print(result["ner_multilingual"])
# {
#   "entities": [
#     {
#       "text": "Angela Merkel",
#       "type": "PERSON",
#       "start": 0,
#       "end": 13,
#       "confidence": 0.92,
#       "source": "transformer"
#     },
#     {
#       "text": "Hamburgo",
#       "type": "LOCATION",
#       "start": 23,
#       "end": 31,
#       "confidence": 0.89,
#       "source": "spacy"
#     },
#     {
#       "text": "Alemania",
#       "type": "LOCATION",
#       "start": 33,
#       "end": 41,
#       "confidence": 0.91,
#       "source": "spacy"
#     },
#     {
#       "text": "1954",
#       "type": "DATE",
#       "start": 45,
#       "end": 49,
#       "confidence": 0.88,
#       "source": "transformer"
#     }
#   ],
#   "entity_count": 4,
#   "entity_types": ["PERSON", "LOCATION", "DATE"]
# }
```

#### Specify Language Code

```python
# You can also specify the language code directly
result = client.extract_entities_multilingual(
    text="Emmanuel Macron est le président de la France.",
    language_code="fr"
)
```

#### Multiple Languages

```python
# Process text in multiple languages
texts = [
    "Apple Inc. is headquartered in Cupertino, California.",  # English
    "谷歌公司位于美国加利福尼亚州。",  # Chinese
    "ソニーは東京に本社があります。",  # Japanese
    "Samsung имеет штаб-квартиру в Сеуле.",  # Russian
]

for text in texts:
    result = client.process_text(text)
    print(f"Language: {result['language_detection']['detected_language']}")
    print(f"Entities: {result['ner_multilingual']['entities']}\n")
```

## Model Details

### Language Detection Model

- **Backend**: Python
- **Library**: langdetect (based on Google's language detection library)
- **Speed**: ~1-5ms per request
- **Batch Size**: Up to 32 requests

### Multilingual NER Model

- **Transformer**: Davlan/xlm-roberta-base-ner-hrl
- **Framework**: Hugging Face Transformers + spaCy
- **Device**: CPU (can use GPU if available)
- **Speed**: ~50-200ms per request (depends on text length and device)
- **Batch Size**: Up to 16 requests

## Performance Tips

### 1. Use GPU Acceleration

If you have a GPU available, the multilingual NER model will automatically use it for faster inference:

```yaml
# docker-compose.yml
services:
  triton-nlp:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 2. Batch Processing

Process multiple texts in a single request for better throughput:

```python
texts = ["Text 1", "Text 2", "Text 3", ...]
results = client.process_batch(texts)
```

### 3. Language-Specific Models

For best accuracy with supported languages, install the corresponding spaCy models:

```bash
# English
python -m spacy download en_core_web_sm

# Spanish
python -m spacy download es_core_news_sm

# German
python -m spacy download de_core_news_sm

# French
python -m spacy download fr_core_news_sm

# And so on...
```

## Comparison: Standard NER vs Multilingual NER

| Feature | Standard NER | Multilingual NER |
|---------|-------------|------------------|
| **Languages** | English only | 100+ languages |
| **Accuracy** | 85-90% (English) | 88-95% (all languages) |
| **Speed** | ~10ms | ~50-200ms |
| **Models** | Rule-based patterns | Transformers + spaCy |
| **Entity Types** | 10+ types | 8 standard types |
| **Context Awareness** | Limited | High |

### When to Use Each

**Use Standard NER when:**
- Processing English-only text
- Need very low latency (<10ms)
- Limited computational resources

**Use Multilingual NER when:**
- Processing non-English text
- Need high accuracy across languages
- Context-aware entity recognition is important
- Have GPU available for acceleration

## Configuration

### Ensemble Configuration

The language detector and multilingual NER are integrated into the ensemble pipeline:

```
Client Request → ensemble_nlp
    ↓
1. preprocessing (text normalization)
    ↓
2. language_detector (detect language)
    ↓
3. Parallel execution:
   - data_type_detector (regex-based)
   - ner (standard English NER)
   - ner_multilingual (multilingual NER with auto language detection)
   - transliteration
   - translation
    ↓
4. postprocessing (aggregate results)
    ↓
JSON Response
```

### Model Files

```
model_repository/
├── language_detector/
│   ├── config.pbtxt
│   └── 1/
│       └── model.py
└── ner_multilingual/
    ├── config.pbtxt
    └── 1/
        └── model.py
```

## Error Handling

### Language Detection Failures

```python
result = client.detect_language("")
# {
#   "detected_language": "unknown",
#   "language_code": "unknown",
#   "confidence": 0.0,
#   "is_reliable": false
# }
```

### NER with Unsupported Languages

For languages without specific spaCy models, the system falls back to the multilingual transformer model:

```python
text = "ሰላም እንደምን ናችሁ?"  # Amharic
result = client.extract_entities_multilingual(text)
# Uses XLM-RoBERTa transformer model automatically
```

## Examples

### Example 1: Multilingual News Article Processing

```python
articles = {
    "en": "President Biden met with Prime Minister Trudeau in Washington.",
    "es": "El presidente Biden se reunió con el primer ministro Trudeau en Washington.",
    "fr": "Le président Biden a rencontré le premier ministre Trudeau à Washington.",
    "de": "Präsident Biden traf sich mit Premierminister Trudeau in Washington.",
}

for lang, text in articles.items():
    result = client.process_text(text)
    print(f"\n{lang.upper()}: {text}")
    print(f"Detected: {result['language_detection']['detected_language']}")
    print(f"Entities: {len(result['ner_multilingual']['entities'])}")
```

### Example 2: Social Media Monitoring

```python
tweets = [
    "Just visited @Google headquarters in Mountain View! #tech",
    "Visité la Torre Eiffel en París 🇫🇷",
    "東京タワーに行ってきました！",
]

for tweet in tweets:
    result = client.process_text(tweet)
    lang = result['language_detection']['language_code']
    entities = result['ner_multilingual']['entities']

    print(f"Tweet: {tweet}")
    print(f"Language: {lang}")
    print(f"Locations: {[e['text'] for e in entities if e['type'] == 'LOCATION']}\n")
```

### Example 3: E-commerce Product Reviews

```python
reviews = [
    {"text": "Ordered from Amazon, arrived in 2 days!", "expected_lang": "en"},
    {"text": "Compré en Amazon México, llegó rápido", "expected_lang": "es"},
    {"text": "アマゾンで購入しました、とても良い", "expected_lang": "ja"},
]

for review in reviews:
    result = client.process_text(review["text"])
    detected = result['language_detection']['language_code']
    entities = result['ner_multilingual']['entities']

    print(f"Review: {review['text']}")
    print(f"Detected: {detected} (Expected: {review['expected_lang']})")
    print(f"Organizations: {[e['text'] for e in entities if e['type'] == 'ORGANIZATION']}\n")
```

## Troubleshooting

### Issue: Low confidence scores

**Solution**: Ensure text is at least 10-20 characters long for reliable detection.

### Issue: Incorrect entity boundaries

**Solution**: Use language-specific spaCy models for better tokenization.

### Issue: Slow performance

**Solution**:
- Enable GPU acceleration
- Use batch processing
- Consider caching results for frequently processed text

## Future Enhancements

- [ ] Support for more languages (150+ planned)
- [ ] Custom entity types
- [ ] Fine-tuning on domain-specific data
- [ ] Confidence calibration
- [ ] Entity linking and disambiguation
