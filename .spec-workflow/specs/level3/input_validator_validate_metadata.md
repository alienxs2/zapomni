# InputValidator.validate_metadata() - Function Specification

**Level:** 3 | **Component:** InputValidator | **Module:** zapomni_core
**Author:** Goncharenko Anton aka alienxs2 | **Status:** Draft | **V:** 1.0

## Signature
```python
@staticmethod
def validate_metadata(metadata: Optional[Dict[str, Any]]) -> None:
    """
    Validate metadata structure and content.
    
    Args:
        metadata: Metadata dict to validate (can be None)
        
    Raises:
        ValidationError: If metadata has reserved keys
        ValidationError: If metadata has non-string keys
        ValidationError: If metadata has non-JSON-serializable values
        
    Returns:
        None (raises on invalid, returns on valid)
    """
```

## Purpose

Static method that validates metadata dict:
- All keys are strings
- No reserved keys ("memory_id", "timestamp", "chunks")
- All values are JSON-serializable (str, int, float, bool, list, dict, None)

## Edge Cases
1. metadata = None → Valid (no validation)
2. metadata = {} → Valid (empty dict)
3. metadata = {"memory_id": "x"} → ValidationError("Reserved key: memory_id")
4. metadata = {123: "x"} → ValidationError("Keys must be strings")
5. metadata = {"key": lambda: None} → ValidationError("Value not JSON-serializable")
6. metadata = {"tags": ["a", "b"]} → Valid
7. metadata = {"nested": {"ok": True}} → Valid

## Algorithm
```
1. If metadata is None, return immediately
2. Check all keys are strings
3. Check no reserved keys ("memory_id", "timestamp", "chunks")
4. For each value, attempt json.dumps() to verify JSON-serializable
5. If any check fails, raise ValidationError
```

## Tests (10)
1. test_validate_none_success
2. test_validate_empty_dict_success
3. test_validate_reserved_key_memory_id_raises
4. test_validate_reserved_key_timestamp_raises
5. test_validate_reserved_key_chunks_raises
6. test_validate_non_string_key_raises
7. test_validate_non_serializable_value_raises
8. test_validate_valid_metadata_success
9. test_validate_nested_dict_success
10. test_validate_list_values_success
