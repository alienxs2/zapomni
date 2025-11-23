# AddMemoryTool._validate_arguments() - Function Specification

**Level:** 3 | **Component:** AddMemoryTool | **Module:** zapomni_mcp
**Author:** Goncharenko Anton aka alienxs2 | **Status:** Draft | **V:** 1.0

## Signature
```python
def _validate_arguments(self, arguments: Dict[str, Any]) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Validate and extract arguments for add_memory tool.
    
    Args:
        arguments: Raw arguments dict from MCP client
        
    Returns:
        Tuple of (text: str, metadata: Optional[Dict])
        
    Raises:
        ValidationError: If text missing, empty, or invalid type
        ValidationError: If metadata invalid type or structure
    """
```

## Purpose

Private helper that validates MCP tool arguments before passing to core engine.
Ensures contract compliance: text is non-empty string, metadata is valid dict.

## Edge Cases
1. arguments = {} → ValidationError("Missing required field: text")
2. arguments = {"text": ""} → ValidationError("text cannot be empty")
3. arguments = {"text": 123} → ValidationError("text must be string")
4. arguments = {"text": "ok", "metadata": "bad"} → ValidationError("metadata must be dict")
5. arguments = {"text": "  "} → ValidationError("text cannot be empty") (after strip)
6. arguments = {"text": "ok"} → Returns ("ok", None)
7. arguments = {"text": "ok", "metadata": {}} → Returns ("ok", {})

## Algorithm
```
1. Check "text" key exists in arguments
2. Extract text value
3. Validate text is string type
4. Validate text is non-empty (after strip)
5. Extract metadata (optional)
6. If metadata provided, validate is dict
7. Return (text, metadata)
```

## Tests (10)
1. test_validate_missing_text_raises
2. test_validate_empty_text_raises
3. test_validate_whitespace_text_raises
4. test_validate_text_wrong_type_raises
5. test_validate_metadata_wrong_type_raises
6. test_validate_text_only_success
7. test_validate_text_and_metadata_success
8. test_validate_empty_metadata_success
9. test_validate_strips_whitespace
10. test_validate_returns_tuple
