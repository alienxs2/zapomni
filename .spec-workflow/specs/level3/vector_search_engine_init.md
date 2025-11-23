# VectorSearchEngine.__init__() - Function Specification

**Level:** 3 | **Component:** VectorSearchEngine | **Module:** zapomni_core
**Author:** Goncharenko Anton aka alienxs2 | **Status:** Draft | **V:** 1.0

## Signature
```python
def __init__(self, db_client: FalkorDBClient, similarity_threshold: float = 0.5) -> None:
    """Initialize vector search engine."""
```

## Parameters
- **db_client**: FalkorDB client (required)
- **similarity_threshold**: Minimum similarity score (default: 0.5, range: 0.0-1.0)

## Edge Cases
1. db_client = None → ValueError
2. similarity_threshold = -0.1 → ValueError
3. similarity_threshold = 1.1 → ValueError
4. similarity_threshold = 0.0 → Valid (matches everything)
5. similarity_threshold = 1.0 → Valid (only exact matches)

## Tests (10)
1. test_init_success, 2. test_init_missing_db_raises, 3. test_init_wrong_type_db_raises,
4. test_init_negative_threshold_raises, 5. test_init_threshold_too_large_raises,
6. test_init_zero_threshold_valid, 7. test_init_one_threshold_valid,
8. test_init_stores_db_client, 9. test_init_stores_threshold,
10. test_init_creates_logger
