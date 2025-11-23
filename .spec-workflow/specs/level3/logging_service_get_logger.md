# LoggingService.get_logger() - Function Specification

**Level:** 3 (Function)
**Component:** LoggingService
**Module:** shared (cross-cutting)
**Author:** Goncharenko Anton aka alienxs2
**Status:** Draft
**Version:** 1.0
**Created:** 2025-11-23

## Function Signature

```python
@classmethod
def get_logger(cls, name: str) -> structlog.BoundLogger:
    """Get a module/component-specific logger."""
```

## Purpose

Return cached logger if exists, otherwise create new logger with automatic context enrichment.

## Parameters

### name: str
- **Purpose:** Logger name (typically module path)
- **Format:** `"zapomni.mcp.tools.add_memory"`
- **Convention:** Python module path
- **Example:** `__name__` in module

## Returns

- **Type:** `structlog.BoundLogger`
- **Features:**
  - Automatic timestamp enrichment
  - Module name bound to context
  - JSON output formatting
  - Thread-safe operation

## Raises

- `RuntimeError`: If logging not configured yet

## Algorithm

```
1. Check if logging configured:
   IF NOT cls._configured:
       RAISE RuntimeError("configure_logging() must be called first")

2. Check cache:
   IF name in cls._loggers:
       RETURN cls._loggers[name]

3. Create new logger:
   logger = structlog.get_logger(name)
   logger = logger.bind(module=name)

4. Cache logger:
   cls._loggers[name] = logger

5. Return logger
```

## Edge Cases

1. **Not configured** → RuntimeError
2. **Empty name** → Creates logger with empty name (valid)
3. **Duplicate calls** → Return cached (fast)
4. **Concurrent calls** → Thread-safe (class-level lock not needed due to GIL)

## Test Scenarios (10)

1. test_get_logger_success
2. test_get_logger_not_configured_raises
3. test_get_logger_caching
4. test_get_logger_different_names
5. test_get_logger_with_module_name
6. test_get_logger_context_enrichment
7. test_get_logger_json_output
8. test_get_logger_thread_safe
9. test_get_logger_empty_name
10. test_get_logger_performance

## Performance

- **First call:** < 5ms (create logger)
- **Cached calls:** < 1ms (dict lookup)

## Usage Example

```python
# In a module
logger = LoggingService.get_logger(__name__)

logger.info("operation_started", correlation_id="uuid", input_size=1024)
```

**Status:** Draft v1.0 | **Author:** Goncharenko Anton aka alienxs2 | **License:** MIT
