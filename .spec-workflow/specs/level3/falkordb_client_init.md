# FalkorDBClient.__init__() - Function Specification

**Level:** 3 | **Component:** FalkorDBClient | **Module:** zapomni_db
**Author:** Goncharenko Anton aka alienxs2 | **Status:** Draft | **V:** 1.0

## Signature
```python
def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0) -> None:
    """Initialize FalkorDB client connection."""
```

## Parameters
- **host**: Redis host (default: "localhost")
- **port**: Redis port (default: 6379, range: 1-65535)
- **db**: Database index (default: 0, range: 0-15)

## Edge Cases
1. port = 0 → ValueError
2. port = 65536 → ValueError
3. db = -1 → ValueError
4. db = 16 → ValueError
5. host = "" → ValueError

## Tests (10)
1. test_init_defaults, 2. test_init_custom_params, 3. test_init_zero_port_raises,
4. test_init_port_too_large_raises, 5. test_init_negative_db_raises,
6. test_init_db_too_large_raises, 7. test_init_empty_host_raises,
8. test_init_creates_connection_pool, 9. test_init_lazy_connection,
10. test_init_stores_params
