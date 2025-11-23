# EntityExtractor.__init__() - Function Specification

**Level:** 3 | **Component:** EntityExtractor | **Module:** zapomni_core
**Author:** Goncharenko Anton aka alienxs2 | **Status:** Draft | **V:** 1.0

## Signature
```python
def __init__(
    self,
    spacy_model: str = "en_core_web_sm",
    ollama_host: str = "http://localhost:11434",
    extraction_model: str = "llama3"
) -> None:
    """Initialize entity extractor with SpaCy and Ollama."""
```

## Parameters
- **spacy_model**: SpaCy model name (default: "en_core_web_sm")
- **ollama_host**: Ollama API URL (default: "http://localhost:11434")
- **extraction_model**: Ollama model for extraction (default: "llama3")

## Edge Cases
1. Invalid spacy_model → NotImplementedError in Phase 1
2. Empty spacy_model → ValueError
3. Invalid ollama_host URL → ValueError
4. Empty extraction_model → ValueError
5. SpaCy model not installed → Warning logged, Phase 2 feature

## Tests (10)
1. test_init_defaults, 2. test_init_custom_params, 3. test_init_empty_spacy_model_raises,
4. test_init_invalid_ollama_url_raises, 5. test_init_empty_extraction_model_raises,
6. test_init_phase1_not_implemented, 7. test_init_stores_params,
8. test_init_creates_logger, 9. test_init_lazy_spacy_load,
10. test_init_lazy_ollama_connection
