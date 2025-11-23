# ConfigurationManager.load_config() - Function Specification

**Level:** 3 (Function)
**Component:** ConfigurationManager (ZapomniSettings)
**Module:** shared (cross-cutting)
**Author:** Goncharenko Anton aka alienxs2
**Status:** Draft
**Version:** 1.0
**Created:** 2025-11-23

## Function Signature

```python
def __init__(self, **values: Any) -> None:
    """Initialize configuration from env, .env file, and defaults."""
```

**Note:** ConfigurationManager uses Pydantic BaseSettings which provides automatic `__init__`. This spec covers the initialization behavior.

## Purpose

Load and validate configuration from multiple sources (environment variables, .env file, defaults) with priority: system env > .env file > defaults.

## Parameters

### **values: Any
- **Purpose:** Optional keyword arguments to override config (primarily for testing)
- **Example:** `ZapomniSettings(falkordb_host="testdb.com")`

## Returns

None (constructor)

## Raises

- `ValidationError`: If any config parameter fails validation

## Algorithm

```
1. Load environment variables from system
2. Parse .env file (if present)
3. For each field:
   a. Get from env var (priority 1)
   b. Get from .env file (priority 2)
   c. Use default (priority 3)
   d. Convert to target type
   e. Run field validators (ge, le)
   f. Run custom @validator methods
4. Create directories (data_dir, temp_dir)
5. Return initialized instance
```

## Edge Cases

1. **Missing .env file** → Use system env + defaults (not an error)
2. **Invalid env var type** → ValidationError with clear message
3. **Directory creation fails** → OSError
4. **Password empty** → None (valid, no auth)
5. **Invalid log level** → ValidationError
6. **Port out of range** → ValidationError

## Test Scenarios (10+)

1. test_init_default_configuration
2. test_init_environment_override
3. test_init_dotenv_loading
4. test_init_priority_order
5. test_init_invalid_port_raises
6. test_init_invalid_log_level_raises
7. test_init_directory_creation
8. test_init_directory_creation_failure
9. test_init_chunk_overlap_validation
10. test_init_vector_dimensions_warning
11. test_init_custom_values

## Performance

- Target: < 100ms initialization
- Actual: ~10-20ms

**Status:** Draft v1.0 | **Author:** Goncharenko Anton aka alienxs2 | **License:** MIT
