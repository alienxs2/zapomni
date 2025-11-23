# GetStatsTool.execute() - Function Specification

**Level:** 3 (Function)
**Component:** GetStatsTool
**Module:** zapomni_mcp
**Author:** Goncharenko Anton aka alienxs2
**Status:** Draft
**Version:** 1.0
**Created:** 2025-11-23

---

## Function Signature

```python
async def execute(
    self,
    arguments: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute get_stats tool and return system statistics.

    This method retrieves statistics from the core engine and formats
    them into an MCP-compliant response. Since this tool requires no
    parameters, the arguments dict is expected to be empty (but is
    accepted for consistency with the MCP tool interface).

    Args:
        arguments: Dictionary of arguments (should be empty {}).
                  Any provided arguments are ignored as per tool spec.

    Returns:
        Dictionary in MCP response format:
        {
            "content": [
                {
                    "type": "text",
                    "text": "Formatted statistics string"
                }
            ],
            "isError": False
        }

    Raises:
        No exceptions are raised. All errors are caught and returned
        as MCP error responses with isError=True.

    Example:
        ```python
        tool = GetStatsTool(core=engine)

        # Execute with empty arguments
        result = await tool.execute({})

        # Success result:
        {
            "content": [{
                "type": "text",
                "text": "Memory System Statistics:\\n"
                        "Total Memories: 1,234\\n"
                        "Total Chunks: 5,678\\n"
                        "Database Size: 45.6 MB\\n"
                        "Graph Name: zapomni_memory\\n"
                        "Cache Hit Rate: 65.3%\\n"
                        "Avg Query Latency: 23.4 ms"
            }],
            "isError": False
        }
        ```
    """
```

## Purpose & Context

### What It Does

Executes the get_stats MCP tool by:
1. Validating that arguments is a dictionary (type check only)
2. Calling `self.core.get_stats()` to retrieve system statistics
3. Formatting raw statistics into human-readable text via `_format_response()`
4. Returning MCP-compliant response structure
5. Catching and handling all errors, returning error responses instead of raising

### Why It Exists

Required by MCP protocol specification. All tools must implement an `execute()` method that accepts arguments and returns a response dictionary. This is the entry point called when Claude Desktop invokes the get_stats tool.

### When To Use

Called automatically by MCP server when it receives a tools/call request for the "get_stats" tool from the client (Claude Desktop).

### When NOT To Use

- Do not call directly from application code
- Do not use for internal statistics gathering (use `core.get_stats()` directly)

---

## Parameters (Detailed)

### arguments: Dict[str, Any]

**Type:** `Dict[str, Any]`

**Purpose:** Contains tool arguments from MCP client (expected to be empty for this tool)

**Constraints:**
- MUST be a dictionary (type validation performed)
- Content is ignored (no parameters are used)
- Empty dict `{}` is the expected input
- Extra keys are silently ignored

**Validation:**
```python
if not isinstance(arguments, dict):
    return {
        "content": [{
            "type": "text",
            "text": "Error: Arguments must be a dictionary (expected empty {})"
        }],
        "isError": True
    }
```

**Examples:**
- Valid: `{}`
- Valid: `{"ignored": "value"}` (extra keys ignored)
- Invalid: `None` → Type error response
- Invalid: `[]` → Type error response
- Invalid: `"string"` → Type error response

---

## Return Value

**Type:** `Dict[str, Any]`

**Structure:**
```python
{
    "content": [
        {
            "type": "text",
            "text": str  # Formatted statistics or error message
        }
    ],
    "isError": bool  # False on success, True on error
}
```

**Success Case:**
```python
{
    "content": [{
        "type": "text",
        "text": (
            "Memory System Statistics:\n"
            "Total Memories: 1,234\n"
            "Total Chunks: 5,678\n"
            "Database Size: 45.67 MB\n"
            "Graph Name: zapomni_memory\n"
            "Cache Hit Rate: 65.3%\n"
            "Avg Query Latency: 23.4 ms"
        )
    }],
    "isError": False
}
```

**Error Case:**
```python
{
    "content": [{
        "type": "text",
        "text": "Error: Failed to retrieve statistics - Database connection lost"
    }],
    "isError": True
}
```

---

## Exceptions

**No exceptions are raised by this function.** All errors are caught and converted to error responses.

### Caught Exceptions

#### CoreError (from zapomni_core.exceptions)

**When Caught:**
- Core engine fails to retrieve statistics
- Database query fails
- Processing error in core layer

**Handling:**
```python
except CoreError as e:
    logger.error("get_stats_core_error", error=str(e), error_type=type(e).__name__)
    return {
        "content": [{
            "type": "text",
            "text": f"Error: Failed to retrieve statistics - {str(e)}"
        }],
        "isError": True
    }
```

**Example Error Message:**
```
"Error: Failed to retrieve statistics - FalkorDB connection timeout"
```

#### Exception (catch-all)

**When Caught:**
- Unexpected errors (bugs, unforeseen conditions)
- Type errors from malformed responses
- Any other exception not caught by CoreError

**Handling:**
```python
except Exception as e:
    logger.error("get_stats_unexpected_error", error=str(e), error_type=type(e).__name__)
    return {
        "content": [{
            "type": "text",
            "text": "Error: An unexpected error occurred while retrieving statistics"
        }],
        "isError": True
    }
```

**Example Error Message:**
```
"Error: An unexpected error occurred while retrieving statistics"
```

---

## Algorithm (Pseudocode)

```
FUNCTION execute(self, arguments):
    # Step 1: Log request
    log_info("get_stats_requested", arguments=arguments)

    # Step 2: Validate arguments type
    IF NOT isinstance(arguments, dict):
        log_warning("get_stats_invalid_arguments_type", type=type(arguments))
        RETURN error_response("Arguments must be a dictionary (expected empty {})")

    # Step 3: Retrieve statistics from core engine
    TRY:
        stats = AWAIT self.core.get_stats()
        log_debug("stats_retrieved", stats=stats)

    # Step 4: Handle core errors
    CATCH CoreError as e:
        log_error("get_stats_core_error", error=str(e), error_type=type(e))
        RETURN error_response(f"Failed to retrieve statistics - {e}")

    # Step 5: Handle unexpected errors
    CATCH Exception as e:
        log_error("get_stats_unexpected_error", error=str(e), error_type=type(e))
        RETURN error_response("An unexpected error occurred while retrieving statistics")

    # Step 6: Format response
    response = self._format_response(stats)
    log_info("get_stats_success", stats_keys=list(stats.keys()))

    # Step 7: Return formatted response
    RETURN response
END FUNCTION
```

---

## Preconditions

- ✅ GetStatsTool instance initialized via `__init__(core)`
- ✅ `self.core` is a valid MemoryEngine instance
- ✅ Core engine is connected to database (optional - errors handled gracefully if not)

---

## Postconditions

**On Success:**
- ✅ Statistics retrieved from core engine
- ✅ Response logged with stats keys
- ✅ MCP response returned with `isError: False`
- ✅ No state changes (read-only operation)

**On Error:**
- ✅ Error logged with details (error type, message)
- ✅ MCP error response returned with `isError: True`
- ✅ No exceptions propagated to caller
- ✅ No state changes

---

## Edge Cases & Handling

### Edge Case 1: Empty Arguments (Normal Case)

**Scenario:** Client sends `arguments: {}`

**Expected Behavior:**
```python
result = await tool.execute({})
assert result["isError"] is False
assert "Total Memories:" in result["content"][0]["text"]
```

**Test Scenario:**
```python
async def test_execute_empty_arguments():
    tool = GetStatsTool(core=mock_engine)
    result = await tool.execute({})
    assert result["isError"] is False
```

---

### Edge Case 2: Arguments with Extra Keys

**Scenario:** Client sends `arguments: {"foo": "bar"}`

**Expected Behavior:**
Extra keys are ignored, statistics are retrieved normally.

**Test Scenario:**
```python
async def test_execute_ignores_extra_arguments():
    tool = GetStatsTool(core=mock_engine)
    result = await tool.execute({"foo": "bar", "baz": 123})
    assert result["isError"] is False  # Extra keys ignored
```

---

### Edge Case 3: Arguments Not a Dictionary

**Scenario:** Client sends `arguments: "string"` or `arguments: None`

**Expected Behavior:**
```python
{
    "content": [{
        "type": "text",
        "text": "Error: Arguments must be a dictionary (expected empty {})"
    }],
    "isError": True
}
```

**Test Scenario:**
```python
async def test_execute_invalid_arguments_type():
    tool = GetStatsTool(core=mock_engine)

    # Test with None
    result = await tool.execute(None)
    assert result["isError"] is True
    assert "must be a dictionary" in result["content"][0]["text"]

    # Test with string
    result = await tool.execute("not a dict")
    assert result["isError"] is True
```

---

### Edge Case 4: Core Engine Raises CoreError

**Scenario:** `core.get_stats()` raises `DatabaseError("Connection lost")`

**Expected Behavior:**
```python
{
    "content": [{
        "type": "text",
        "text": "Error: Failed to retrieve statistics - Connection lost"
    }],
    "isError": True
}
```

**Test Scenario:**
```python
async def test_execute_core_error_database():
    mock_engine = Mock()
    mock_engine.get_stats = AsyncMock(side_effect=DatabaseError("Connection lost"))

    tool = GetStatsTool(core=mock_engine)
    result = await tool.execute({})

    assert result["isError"] is True
    assert "Connection lost" in result["content"][0]["text"]
```

---

### Edge Case 5: Core Engine Raises Unexpected Exception

**Scenario:** `core.get_stats()` raises `KeyError("unexpected")`

**Expected Behavior:**
```python
{
    "content": [{
        "type": "text",
        "text": "Error: An unexpected error occurred while retrieving statistics"
    }],
    "isError": True
}
```

**Test Scenario:**
```python
async def test_execute_unexpected_error():
    mock_engine = Mock()
    mock_engine.get_stats = AsyncMock(side_effect=KeyError("unexpected"))

    tool = GetStatsTool(core=mock_engine)
    result = await tool.execute({})

    assert result["isError"] is True
    assert "unexpected error" in result["content"][0]["text"]
    # Generic message - no internal details leaked
```

---

### Edge Case 6: Stats Dict Missing Optional Keys

**Scenario:** `core.get_stats()` returns dict without optional keys (cache_hit_rate, etc.)

**Expected Behavior:**
`_format_response()` handles missing keys gracefully, omitting optional fields.

**Test Scenario:**
```python
async def test_execute_minimal_stats():
    mock_engine = Mock()
    mock_engine.get_stats = AsyncMock(return_value={
        "total_memories": 10,
        "total_chunks": 50,
        "database_size_mb": 1.5,
        "graph_name": "test_graph"
        # Optional keys omitted
    })

    tool = GetStatsTool(core=mock_engine)
    result = await tool.execute({})

    assert result["isError"] is False
    text = result["content"][0]["text"]
    assert "Total Memories: 10" in text
    assert "Cache Hit Rate" not in text  # Optional, not present
```

---

## Test Scenarios (Complete List)

### Happy Path Tests

1. **test_execute_success_minimal**
   - Input: `{}` (empty arguments)
   - Mock: `core.get_stats()` returns minimal stats
   - Expected: `isError=False`, stats formatted correctly

2. **test_execute_success_full_stats**
   - Input: `{}`
   - Mock: `core.get_stats()` returns all stats (including optional)
   - Expected: All fields formatted in response text

3. **test_execute_ignores_extra_arguments**
   - Input: `{"foo": "bar"}`
   - Expected: Extra keys ignored, stats retrieved normally

### Error Tests

4. **test_execute_invalid_arguments_type_none**
   - Input: `None`
   - Expected: `isError=True`, message about dict requirement

5. **test_execute_invalid_arguments_type_string**
   - Input: `"string"`
   - Expected: `isError=True`, message about dict requirement

6. **test_execute_invalid_arguments_type_list**
   - Input: `[]`
   - Expected: `isError=True`, message about dict requirement

7. **test_execute_core_error_database**
   - Mock: `core.get_stats()` raises `DatabaseError`
   - Expected: `isError=True`, error message includes exception text

8. **test_execute_core_error_processing**
   - Mock: `core.get_stats()` raises `ProcessingError`
   - Expected: `isError=True`, error message includes exception text

9. **test_execute_unexpected_error**
   - Mock: `core.get_stats()` raises `KeyError`
   - Expected: `isError=True`, generic error message (no internals)

10. **test_execute_unexpected_error_attribute**
    - Mock: `core.get_stats()` raises `AttributeError`
    - Expected: `isError=True`, generic error message

### Integration/Dependency Tests

11. **test_execute_calls_format_response**
    - Verify: `_format_response()` is called with stats dict
    - Mock: Spy on `_format_response()`
    - Expected: Called once with correct stats

12. **test_execute_logging**
    - Verify: Logs at appropriate levels (info, debug, error)
    - Mock: Capture log calls
    - Expected: Logs "get_stats_requested", "stats_retrieved", "get_stats_success"

---

## Performance Requirements

**Latency Targets:**
- P50: < 50ms
- P95: < 100ms
- P99: < 200ms
- Maximum: < 500ms

**Throughput:**
- Should handle concurrent calls (thread-safe via async)
- No rate limiting needed (read-only operation)

**Resource Usage:**
- Memory: O(1) - only stores stats dict temporarily
- CPU: Minimal (formatting is simple string operations)

---

## Security Considerations

### Input Validation

- ✅ Type check on `arguments` parameter
- ✅ No execution of user-provided code
- ✅ No file system access
- ✅ No network calls (except to core engine)

### Data Protection

- ✅ No sensitive data in statistics output
- ✅ Generic error messages (don't leak internals)
- ✅ Logging sanitized (no passwords, tokens)

### Error Messages

- ✅ Success: Informative statistics
- ✅ Validation errors: Clear message about expected input
- ✅ Core errors: Include exception message (safe - no secrets)
- ✅ Unexpected errors: Generic message (don't leak internals)

---

## Related Functions

**Calls:**
- `self.core.get_stats()` - Retrieves statistics from core engine
- `self._format_response(stats)` - Formats stats into MCP response
- `logger.info()`, `logger.debug()`, `logger.error()` - Logging

**Called By:**
- MCP server `call_tool()` handler when tool name is "get_stats"

---

## Implementation Notes

### Libraries Used

- `structlog` - Structured logging
- `typing` - Type hints

### Known Limitations

- Statistics are not cached (always fetched fresh from core)
- No pagination (all stats returned in single response)
- No filtering (always returns all available stats)

### Future Enhancements

- Add caching with configurable TTL (e.g., cache for 30 seconds)
- Support filtering statistics (e.g., only database stats)
- Add historical statistics (trends over time)

---

## References

- Component spec: [get_stats_tool_component.md](../level2/get_stats_tool_component.md)
- Module spec: [zapomni_mcp_module.md](../level1/zapomni_mcp_module.md)
- MCP Specification: https://spec.modelcontextprotocol.io/

---

**Document Status:** Draft v1.0
**Created:** 2025-11-23
**Author:** Goncharenko Anton aka alienxs2
**License:** MIT License
**Ready for Implementation:** Yes ✅
