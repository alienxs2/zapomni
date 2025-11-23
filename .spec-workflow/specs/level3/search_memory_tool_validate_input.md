# SearchMemoryTool._validate_input() - Function Specification

**Level:** 3 | **Component:** SearchMemoryTool | **Module:** zapomni_mcp
**Author:** Goncharenko Anton aka alienxs2 | **Status:** Draft | **V:** 1.0

## Signature
```python
def _validate_input(
    self,
    arguments: Dict[str, Any]
) -> Tuple[str, int, Optional[Dict[str, Any]], str]:
    """
    Validate and extract search arguments.
    
    Args:
        arguments: Raw arguments from MCP client
        
    Returns:
        Tuple of (query: str, limit: int, filters: Optional[Dict], search_mode: str)
        
    Raises:
        ValidationError: If query missing/empty/invalid
        ValidationError: If limit out of range [1, 100]
        ValidationError: If filters invalid structure
        ValidationError: If search_mode invalid
    """
```

## Purpose

Validate search tool arguments: query (required), limit (1-100), filters (optional dict), search_mode (default "vector").

## Edge Cases
1. Missing query → ValidationError
2. Empty query "" → ValidationError
3. limit = 0 → ValidationError("limit must be >= 1")
4. limit = 101 → ValidationError("limit must be <= 100")
5. limit missing → Uses default 10
6. search_mode = "invalid" → ValidationError
7. search_mode missing → Uses default "vector"
8. filters = "bad" → ValidationError("filters must be dict")

## Algorithm
```
1. Validate query exists and non-empty string
2. Extract limit (default 10), validate 1-100
3. Extract filters (optional), validate dict if provided
4. Extract search_mode (default "vector"), validate in allowed modes
5. Return tuple
```

## Tests (10)
1. test_validate_missing_query_raises
2. test_validate_empty_query_raises
3. test_validate_zero_limit_raises
4. test_validate_limit_too_large_raises
5. test_validate_invalid_search_mode_raises
6. test_validate_filters_wrong_type_raises
7. test_validate_minimal_args_success
8. test_validate_all_args_success
9. test_validate_defaults_applied
10. test_validate_returns_tuple
