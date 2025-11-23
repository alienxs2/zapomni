# OllamaEmbedder.__init__() - Function Specification

**Level:** 3 | **Component:** OllamaEmbedder | **Module:** zapomni_core
**Author:** Goncharenko Anton aka alienxs2 | **Status:** Draft | **Version:** 1.0

## Function Signature

```python
def __init__(
    self,
    host: str = "http://localhost:11434",
    model: str = "nomic-embed-text",
    timeout: int = 30,
    batch_size: int = 32
) -> None:
    """Initialize OllamaEmbedder with API configuration."""
```

## Parameters

- **host**: Ollama API URL (default: "http://localhost:11434")
  - Validation: Must be valid HTTP/HTTPS URL
- **model**: Embedding model name (default: "nomic-embed-text")
  - Validation: Non-empty string
- **timeout**: Request timeout in seconds (default: 30)
  - Validation: Must be > 0
- **batch_size**: Embedding batch size (default: 32)
  - Validation: 1 <= batch_size <= 128

## Edge Cases

1. Invalid URL → ValueError
2. Empty model name → ValueError
3. timeout <= 0 → ValueError
4. batch_size = 0 → ValueError
5. batch_size > 128 → ValueError

## Tests (10)

1. test_init_defaults_success
2. test_init_custom_params_success
3. test_init_invalid_url_raises
4. test_init_empty_model_raises
5. test_init_zero_timeout_raises
6. test_init_negative_timeout_raises
7. test_init_zero_batch_size_raises
8. test_init_batch_size_too_large_raises
9. test_init_creates_http_client
10. test_init_stores_all_params

**License:** MIT
