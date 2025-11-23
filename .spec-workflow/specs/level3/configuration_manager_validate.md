# ConfigurationManager.validate() - Function Specification

**Level:** 3 | **Component:** ConfigurationManager | **Module:** zapomni_core
**Author:** Goncharenko Anton aka alienxs2 | **Status:** Draft | **V:** 1.0

## Signature
```python
def validate(self) -> List[str]:
    """
    Validate loaded configuration for errors and warnings.
    
    Returns:
        List of validation error messages (empty if valid)
        
    Example:
        >>> config = ConfigurationManager()
        >>> errors = config.validate()
        >>> if errors:
        ...     print(f"Config invalid: {errors}")
        >>> else:
        ...     print("Config valid")
    """
```

## Purpose

Validate configuration after loading:
- Required fields present
- Values in valid ranges
- URLs well-formed
- File paths exist
- Dependencies available

## Edge Cases
1. All valid → Returns []
2. Missing required field → Returns ["Missing required: OLLAMA_HOST"]
3. Invalid URL → Returns ["Invalid URL: OLLAMA_HOST"]
4. Port out of range → Returns ["Port must be 1-65535: REDIS_PORT"]
5. Multiple errors → Returns list of all errors

## Algorithm
```
1. Check required env vars set (OLLAMA_HOST, REDIS_HOST)
2. Validate URLs well-formed
3. Validate ports in range 1-65535
4. Validate paths exist (if specified)
5. Validate numeric ranges (chunk_size, etc.)
6. Collect all errors
7. Return error list (empty if valid)
```

## Tests (10)
1. test_validate_success
2. test_validate_missing_ollama_host
3. test_validate_invalid_url
4. test_validate_port_out_of_range
5. test_validate_invalid_chunk_size
6. test_validate_multiple_errors
7. test_validate_path_not_exists
8. test_validate_negative_values
9. test_validate_empty_strings
10. test_validate_all_fields
