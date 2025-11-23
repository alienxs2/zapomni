# AddMemoryTool.execute() - Function Specification

**Level:** 3 (Function)
**Component:** AddMemoryTool
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
    Execute add_memory tool with provided arguments from MCP client.

    This is the main entry point called by the MCP server when a client
    invokes the add_memory tool. It validates inputs, processes the memory
    through the core engine, and returns a formatted MCP response.

    The method implements a fail-safe design: all exceptions are caught
    and converted to MCP error responses. No exceptions propagate to caller.

    Args:
        arguments: Dictionary containing tool parameters from MCP client
            - text (str, required): Memory content to store (1 to 10,000,000 chars)
            - metadata (dict, optional): Additional metadata (JSON-serializable)

    Returns:
        MCP-formatted response dictionary conforming to MCP protocol:

        Success response:
        {
            "content": [
                {
                    "type": "text",
                    "text": "Memory stored successfully.\n"
                            "ID: {memory_id}\n"
                            "Chunks created: {chunk_count}\n"
                            "Preview: {text_preview}"
                }
            ],
            "isError": False
        }

        Error response:
        {
            "content": [
                {
                    "type": "text",
                    "text": "Error: {error_message}"
                }
            ],
            "isError": True
        }

    Raises:
        Never raises exceptions. All errors are caught and returned as
        MCP error responses with isError=True.

    Example:
        >>> tool = AddMemoryTool(core_engine=engine)
        >>> result = await tool.execute({
        ...     "text": "Python is a high-level programming language",
        ...     "metadata": {"source": "user", "tags": ["programming"]}
        ... })
        >>> print(result["isError"])
        False
        >>> print("ID:" in result["content"][0]["text"])
        True

    Thread Safety:
        Not thread-safe. Must be called in async/await context only.
        MCP server ensures sequential execution (stdio transport limitation).

    Performance:
        - Validation overhead: < 5ms for typical inputs
        - Core delegation: O(1) function call (actual processing in core)
        - Response formatting: < 5ms
        - Total MCP layer overhead: < 10ms
    """
```

---

## Purpose & Context

### What It Does

The `execute()` method is the primary entry point for the `add_memory` MCP tool. It performs a three-step process:

1. **Validates** input arguments using Pydantic models
2. **Delegates** memory processing to `MemoryEngine.add_memory()`
3. **Formats** the result into MCP protocol response format

This method acts as a pure adapter between the MCP protocol layer and the Zapomni core engine, containing zero business logic.

### Why It Exists

**MCP Protocol Requirement:**
- All MCP tools must implement an `execute()` method
- This is the standard interface called by MCP servers
- Signature is dictated by MCP SDK protocol

**Separation of Concerns:**
- MCP layer handles protocol concerns (validation, formatting)
- Core engine handles business logic (processing, storage)
- Clear boundary enables independent testing and evolution

### When To Use

**Called Automatically By:**
- `MCPServer.handle_tool_call()` when routing `add_memory` requests
- MCP SDK when client invokes tool via stdio transport

**Not Called Directly By:**
- Application code (use `MemoryEngine.add_memory()` instead)
- Tests (test via tool interface, not direct calls)

### When NOT To Use

**Don't use this if you want to:**
- Add memories programmatically → use `MemoryEngine.add_memory()` directly
- Batch process memories → use core engine's batch methods
- Skip validation → this method always validates

---

## Parameters (Detailed)

### arguments: Dict[str, Any]

**Type:** `Dict[str, Any]`

**Purpose:**
Container for tool arguments from MCP client. Conforms to `input_schema` defined in `AddMemoryTool` class.

**Structure:**
```python
{
    "text": str,              # Required: memory content
    "metadata": dict | None   # Optional: additional metadata
}
```

**Constraints:**

1. **Type Constraint:**
   - Must be a Python `dict`
   - Keys must be strings
   - Values can be any JSON-serializable type

2. **Required Keys:**
   - `"text"`: MUST be present

3. **text Field Constraints:**
   - Type: `str`
   - Minimum length: 1 character (after stripping whitespace)
   - Maximum length: 10,000,000 characters
   - Encoding: Valid UTF-8
   - Cannot be only whitespace

4. **metadata Field Constraints (if provided):**
   - Type: `dict[str, Any]`
   - Must be JSON-serializable
   - Can contain nested structures
   - Special fields recognized:
     - `source`: `str` - Origin of memory
     - `tags`: `list[str]` - Classification tags
     - `timestamp`: `str` - ISO 8601 timestamp
     - `language`: `str` - Programming language (if code)
   - Additional properties allowed (passed through to core)

**Validation Logic:**
```python
# Step 1: Validate using Pydantic AddMemoryRequest model
try:
    request = AddMemoryRequest(**arguments)
except ValidationError as e:
    # Pydantic validation failed
    raise ValidationError(f"Invalid arguments: {e}")

# Step 2: Extract and sanitize
text = request.text.strip()

# Step 3: Check post-strip emptiness
if not text:
    raise ValidationError("text cannot be empty or whitespace-only")

# Step 4: Extract metadata (default to empty dict)
metadata = request.metadata or {}
```

**Examples:**

**Valid - Minimal:**
```python
{
    "text": "Python is a programming language"
}
```

**Valid - With Metadata:**
```python
{
    "text": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "metadata": {
        "source": "code_file",
        "language": "python",
        "tags": ["algorithms", "recursion"],
        "timestamp": "2025-11-23T10:30:00Z"
    }
}
```

**Valid - Custom Metadata:**
```python
{
    "text": "Meeting notes from standup",
    "metadata": {
        "source": "user",
        "custom_field": "custom_value",
        "project_id": "proj-123"
    }
}
```

**Invalid - Missing text:**
```python
{
    "metadata": {"source": "user"}
}
# Raises: ValidationError("field required: text")
```

**Invalid - Empty text:**
```python
{
    "text": ""
}
# Raises: ValidationError("text must have at least 1 character")
```

**Invalid - Whitespace-only text:**
```python
{
    "text": "   \n\t  "
}
# After strip() becomes empty
# Raises: ValidationError("text cannot be empty or whitespace-only")
```

**Invalid - Text too long:**
```python
{
    "text": "x" * 10_000_001
}
# Raises: ValidationError("text exceeds maximum length of 10,000,000 characters")
```

**Invalid - Wrong text type:**
```python
{
    "text": 12345
}
# Raises: ValidationError("text must be a string")
```

**Invalid - Wrong metadata type:**
```python
{
    "text": "valid text",
    "metadata": "invalid"
}
# Raises: ValidationError("metadata must be an object/dict")
```

**Invalid - Non-JSON-serializable metadata:**
```python
{
    "text": "valid text",
    "metadata": {"function": lambda x: x}
}
# Raises: ValidationError("metadata must be JSON-serializable")
```

---

## Return Value

**Type:** `Dict[str, Any]`

**Purpose:**
MCP protocol-compliant response that communicates operation result to client.

### Success Response Structure

```python
{
    "content": [
        {
            "type": "text",
            "text": str  # Human-readable success message
        }
    ],
    "isError": False
}
```

**Success Message Format:**
```
Memory stored successfully.
ID: {memory_id}
Chunks created: {chunk_count}
Preview: {text_preview}
```

**Fields in Success Message:**
- `memory_id`: UUID string (e.g., `"550e8400-e29b-41d4-a716-446655440000"`)
- `chunk_count`: Integer (e.g., `3`)
- `text_preview`: First 100 characters of stored text

**Example Success Response:**
```python
{
    "content": [
        {
            "type": "text",
            "text": "Memory stored successfully.\n"
                    "ID: 550e8400-e29b-41d4-a716-446655440000\n"
                    "Chunks created: 3\n"
                    "Preview: Python is a high-level programming language designed for readability and simplicity. It sup..."
        }
    ],
    "isError": False
}
```

### Error Response Structure

```python
{
    "content": [
        {
            "type": "text",
            "text": str  # Human-readable error message
        }
    ],
    "isError": True
}
```

**Error Message Categories:**

1. **Validation Errors** (user-facing, detailed):
   ```
   Error: {detailed_validation_message}
   ```
   - Example: `"Error: text cannot be empty or whitespace-only"`
   - Example: `"Error: field required: text"`

2. **Database Errors** (generic, with retry suggestion):
   ```
   Error: Database temporarily unavailable. Please retry in a few seconds.
   ```

3. **Processing Errors** (generic, security-safe):
   ```
   Error: An internal error occurred while processing your memory.
   ```

**Example Error Response (Validation):**
```python
{
    "content": [
        {
            "type": "text",
            "text": "Error: text cannot be empty or whitespace-only"
        }
    ],
    "isError": True
}
```

**Example Error Response (Database):**
```python
{
    "content": [
        {
            "type": "text",
            "text": "Error: Database temporarily unavailable. Please retry in a few seconds."
        }
    ],
    "isError": True
}
```

**Security Considerations:**
- Never expose internal paths, stack traces, or database details
- ValidationError messages are safe (user input validation)
- All other errors use generic messages
- Detailed errors logged to stderr, not returned to client

---

## Exceptions

### Never Raises

**Critical Design Decision:**

This method **NEVER** raises exceptions. All errors are caught and converted to MCP error responses.

**Rationale:**
1. MCP protocol expects all responses in standard format
2. Exceptions would crash MCP server stdio loop
3. Clients expect predictable response structure
4. Error details still captured in structured logs

### Exception Handling Strategy

```python
try:
    # Validation
    text, metadata = self._validate_arguments(arguments)

    # Processing
    result = await self.core_engine.add_memory(text, metadata)

    # Format success
    return self._format_success(result)

except ValidationError as e:
    # User input errors - safe to expose details
    self.logger.warning("validation_error", error=str(e))
    return self._format_error(e)

except Exception as e:
    # Any other error - log with traceback, return generic message
    self.logger.error(
        "processing_error",
        error_type=type(e).__name__,
        error=str(e),
        exc_info=True
    )
    return self._format_error(e)
```

### Exceptions Caught

**1. ValidationError (from Pydantic)**

**When Raised:**
- `arguments` missing required `"text"` key
- `text` is empty string
- `text` exceeds maximum length
- `text` is wrong type (not string)
- `metadata` is wrong type (not dict)
- `metadata` has invalid structure

**Handling:**
- Log as warning (expected user error)
- Extract detailed validation message
- Return MCP error response with full details
- Client should fix input and retry

**Example:**
```python
ValidationError("field required: text")
→ {"content": [{"type": "text", "text": "Error: field required: text"}], "isError": true}
```

**2. DatabaseError (from core engine)**

**When Raised:**
- Database connection lost
- Query execution failed
- Transaction rollback
- Connection pool exhausted

**Handling:**
- Log as error with full context
- Return generic "Database temporarily unavailable" message
- Suggest retry with backoff
- No internal details exposed

**Example:**
```python
DatabaseError("Connection to FalkorDB failed")
→ {"content": [{"type": "text", "text": "Error: Database temporarily unavailable. Please retry in a few seconds."}], "isError": true}
```

**3. ProcessingError (from core engine)**

**When Raised:**
- Embedding generation failed
- Chunking failed
- Graph storage failed
- Invalid memory format

**Handling:**
- Log as error with full traceback
- Return generic "internal error" message
- No retry suggestion (may not help)
- No internal details exposed

**Example:**
```python
ProcessingError("Embedding model returned invalid shape")
→ {"content": [{"type": "text", "text": "Error: An internal error occurred while processing your memory."}], "isError": true}
```

**4. Any Other Exception**

**When Raised:**
- Unexpected errors
- Programming errors (bugs)
- System errors (OOM, etc.)

**Handling:**
- Log as error with full traceback
- Return generic "internal error" message
- Alert monitoring systems
- Investigate immediately

**Example:**
```python
AttributeError("'NoneType' object has no attribute 'process'")
→ {"content": [{"type": "text", "text": "Error: An internal error occurred while processing your memory."}], "isError": true}
```

---

## Algorithm (Pseudocode)

```
FUNCTION execute(self, arguments: dict) -> dict:
    # Step 1: Initialize request tracking
    request_id = generate_request_id(arguments)
    logger = self.logger.bind(request_id=request_id)
    logger.info("execute_started")

    # Step 2: Execute with comprehensive error handling
    TRY:
        # Step 2.1: Validate input arguments
        logger.info("validating_arguments")

        TRY:
            text, metadata = self._validate_arguments(arguments)
        CATCH ValidationError as validation_error:
            # Input validation failed - expected user error
            logger.warning("validation_error", error=str(validation_error))
            RETURN self._format_error(validation_error)

        # Step 2.2: Log validated inputs
        logger.info(
            "processing_memory",
            text_length=len(text),
            has_metadata=bool(metadata)
        )

        # Step 2.3: Delegate to core engine
        TRY:
            result = AWAIT self.core_engine.add_memory(
                text=text,
                metadata=metadata or {}
            )
        CATCH DatabaseError as db_error:
            # Database unavailable - transient error
            logger.error(
                "database_error",
                error=str(db_error),
                exc_info=True
            )
            RETURN self._format_error(db_error)
        CATCH ProcessingError as proc_error:
            # Core processing failed - may be bug or bad data
            logger.error(
                "processing_error",
                error=str(proc_error),
                exc_info=True
            )
            RETURN self._format_error(proc_error)

        # Step 2.4: Log success
        logger.info(
            "memory_added_successfully",
            memory_id=result.memory_id,
            chunks=result.chunks_created
        )

        # Step 2.5: Format and return success response
        RETURN self._format_success(result)

    CATCH Exception as unexpected_error:
        # Unexpected error - programming bug or system issue
        logger.error(
            "unexpected_error",
            error_type=type(unexpected_error).__name__,
            error=str(unexpected_error),
            exc_info=True
        )
        RETURN self._format_error(unexpected_error)

END FUNCTION
```

**Step-by-Step Breakdown:**

1. **Request ID Generation** (1-2ms)
   - Create unique ID for request correlation
   - Bind to logger for context

2. **Argument Validation** (2-5ms)
   - Call `_validate_arguments()`
   - Pydantic validates structure
   - Strip whitespace from text
   - Extract metadata

3. **Core Delegation** (variable, depends on text size)
   - Call `core_engine.add_memory()`
   - Wait for processing completion
   - Receive MemoryResult

4. **Response Formatting** (< 1ms)
   - Call `_format_success()` or `_format_error()`
   - Build MCP content block
   - Return formatted response

**Total Execution Time:**
- Best case: ~10ms (small text, no errors)
- Typical case: ~50-200ms (including core processing)
- Worst case: Variable (depends on core engine timeout)

---

## Preconditions

### Required State

✅ **Tool Initialization:**
- `AddMemoryTool.__init__()` must have been called
- `self.core_engine` must be set to valid MemoryEngine instance
- `self.logger` must be initialized with tool context

✅ **Core Engine Readiness:**
- `core_engine` must be initialized (`await core_engine.initialize()` called)
- Database connection must be established
- Embedding model must be loaded (or lazy-loaded)

✅ **Arguments Format:**
- `arguments` parameter must be a Python `dict`
- Dictionary must be JSON-deserializable (no circular refs, etc.)

### Not Required (Handled Internally)

❌ **Pre-validation:**
- No need to validate arguments before calling (done internally)

❌ **Error Handling:**
- No need for caller to wrap in try/except (never raises)

❌ **Response Validation:**
- Return value always conforms to MCP schema

---

## Postconditions

### On Success (isError=False)

✅ **Memory Stored:**
- Text content persisted in FalkorDB
- Embeddings generated and stored
- Chunks created and linked
- Metadata attached to memory node

✅ **Response Valid:**
- Return dict conforms to MCP response schema
- `isError` field is `False`
- `content` array has exactly 1 element
- `content[0]["type"]` is `"text"`
- `content[0]["text"]` contains success message with memory_id

✅ **Logging Complete:**
- Request logged with `request_id`
- Success logged with `memory_id` and `chunks_created`

✅ **State Unchanged:**
- No state changes in `AddMemoryTool` instance
- Tool ready for next request

### On Error (isError=True)

✅ **No Memory Stored:**
- No partial writes to database (atomic operation)
- No orphaned chunks or embeddings
- Clean rollback if transaction failed

✅ **Response Valid:**
- Return dict conforms to MCP response schema
- `isError` field is `True`
- `content` array has exactly 1 element
- `content[0]["type"]` is `"text"`
- `content[0]["text"]` contains error message

✅ **Logging Complete:**
- Error logged with error type and message
- Stack trace logged (if unexpected error)
- Warning logged (if validation error)

✅ **State Unchanged:**
- No state changes in `AddMemoryTool` instance
- Tool ready for next request

---

## Edge Cases & Handling

### Edge Case 1: Empty Text (After Stripping)

**Scenario:**
User provides text that is only whitespace: `{"text": "   \n\t  "}`

**Input:**
```python
arguments = {"text": "   \n\t  "}
```

**Processing:**
1. Pydantic validates: `minLength: 1` check passes (3 spaces)
2. `_validate_arguments()` calls `request.text.strip()`
3. After strip: `""` (empty string)
4. Check: `if not text:` → True
5. Raise `ValidationError("text cannot be empty or whitespace-only")`

**Expected Behavior:**
```python
# Method returns (does not raise):
{
    "content": [
        {
            "type": "text",
            "text": "Error: text cannot be empty or whitespace-only"
        }
    ],
    "isError": True
}
```

**Test Scenario:**
```python
def test_execute_empty_text_after_strip():
    """Test text that becomes empty after whitespace stripping."""
    tool = AddMemoryTool(core_engine=mock_engine)

    result = await tool.execute({"text": "   \n\t  "})

    assert result["isError"] is True
    assert "empty" in result["content"][0]["text"].lower()
    assert "whitespace" in result["content"][0]["text"].lower()
```

---

### Edge Case 2: Text Exactly at Maximum Length

**Scenario:**
User provides text with exactly 10,000,000 characters (boundary case).

**Input:**
```python
arguments = {"text": "x" * 10_000_000}
```

**Processing:**
1. Pydantic validates: `maxLength: 10_000_000` → PASS (equal is valid)
2. Strip whitespace (no change)
3. Extract text and metadata
4. Call `core_engine.add_memory()`
5. Process successfully

**Expected Behavior:**
Success response (edge is within valid range).

**Test Scenario:**
```python
def test_execute_text_at_max_length():
    """Test text at exact maximum length boundary."""
    tool = AddMemoryTool(core_engine=mock_engine)
    mock_engine.add_memory = AsyncMock(return_value=MemoryResult(
        memory_id="test-id",
        chunks_created=1000,  # Many chunks for large text
        text_preview="x" * 100
    ))

    max_text = "x" * 10_000_000
    result = await tool.execute({"text": max_text})

    assert result["isError"] is False
    assert "test-id" in result["content"][0]["text"]
    mock_engine.add_memory.assert_called_once_with(
        text=max_text,
        metadata={}
    )
```

---

### Edge Case 3: Text Exceeds Maximum Length

**Scenario:**
User provides text with 10,000,001 characters (one over limit).

**Input:**
```python
arguments = {"text": "x" * 10_000_001}
```

**Processing:**
1. Pydantic validates: `maxLength: 10_000_000` → FAIL
2. Raise `ValidationError("text exceeds maximum length")`

**Expected Behavior:**
```python
{
    "content": [
        {
            "type": "text",
            "text": "Error: text exceeds maximum length of 10,000,000 characters"
        }
    ],
    "isError": True
}
```

**Test Scenario:**
```python
def test_execute_text_exceeds_max_length():
    """Test text exceeding maximum length limit."""
    tool = AddMemoryTool(core_engine=mock_engine)

    too_long = "x" * 10_000_001
    result = await tool.execute({"text": too_long})

    assert result["isError"] is True
    assert "exceeds" in result["content"][0]["text"].lower()
    assert "maximum" in result["content"][0]["text"].lower()
```

---

### Edge Case 4: Missing Required "text" Key

**Scenario:**
Arguments dict missing required `"text"` field.

**Input:**
```python
arguments = {"metadata": {"source": "user"}}
```

**Processing:**
1. Pydantic validates: `required: ["text"]` → FAIL
2. Raise `ValidationError("field required: text")`

**Expected Behavior:**
```python
{
    "content": [
        {
            "type": "text",
            "text": "Error: field required: text"
        }
    ],
    "isError": True
}
```

**Test Scenario:**
```python
def test_execute_missing_text_field():
    """Test arguments missing required text field."""
    tool = AddMemoryTool(core_engine=mock_engine)

    result = await tool.execute({"metadata": {"source": "test"}})

    assert result["isError"] is True
    assert "required" in result["content"][0]["text"].lower()
    assert "text" in result["content"][0]["text"].lower()
```

---

### Edge Case 5: Invalid Metadata Type

**Scenario:**
Metadata provided as string instead of dict.

**Input:**
```python
arguments = {
    "text": "valid text",
    "metadata": "invalid string"
}
```

**Processing:**
1. Pydantic validates: `metadata` type must be `dict` → FAIL
2. Raise `ValidationError("metadata must be an object")`

**Expected Behavior:**
```python
{
    "content": [
        {
            "type": "text",
            "text": "Error: metadata must be an object/dict, not string"
        }
    ],
    "isError": True
}
```

**Test Scenario:**
```python
def test_execute_invalid_metadata_type():
    """Test metadata as wrong type (string instead of dict)."""
    tool = AddMemoryTool(core_engine=mock_engine)

    result = await tool.execute({
        "text": "valid",
        "metadata": "invalid"
    })

    assert result["isError"] is True
    assert "metadata" in result["content"][0]["text"].lower()
    assert "object" in result["content"][0]["text"].lower() or \
           "dict" in result["content"][0]["text"].lower()
```

---

### Edge Case 6: Database Connection Lost

**Scenario:**
Core engine raises `DatabaseError` during `add_memory()` call.

**Input:**
```python
arguments = {"text": "valid memory text"}
```

**Mock Behavior:**
```python
mock_engine.add_memory = AsyncMock(
    side_effect=DatabaseError("Connection to FalkorDB lost")
)
```

**Processing:**
1. Validation succeeds
2. Call `core_engine.add_memory()`
3. DatabaseError raised
4. Caught in except block
5. Logged as error
6. Formatted with generic message

**Expected Behavior:**
```python
{
    "content": [
        {
            "type": "text",
            "text": "Error: Database temporarily unavailable. Please retry in a few seconds."
        }
    ],
    "isError": True
}
```

**Test Scenario:**
```python
def test_execute_database_error():
    """Test handling of database connection failure."""
    tool = AddMemoryTool(core_engine=mock_engine)
    mock_engine.add_memory = AsyncMock(
        side_effect=DatabaseError("Connection lost")
    )

    result = await tool.execute({"text": "test"})

    assert result["isError"] is True
    assert "database" in result["content"][0]["text"].lower()
    assert "unavailable" in result["content"][0]["text"].lower()
    assert "retry" in result["content"][0]["text"].lower()
    # Ensure internal details NOT exposed
    assert "Connection lost" not in result["content"][0]["text"]
```

---

### Edge Case 7: Core Processing Failure

**Scenario:**
Core engine raises `ProcessingError` during memory processing.

**Input:**
```python
arguments = {"text": "valid memory text"}
```

**Mock Behavior:**
```python
mock_engine.add_memory = AsyncMock(
    side_effect=ProcessingError("Embedding generation failed")
)
```

**Processing:**
1. Validation succeeds
2. Call `core_engine.add_memory()`
3. ProcessingError raised
4. Caught in except block
5. Logged as error with traceback
6. Formatted with generic message

**Expected Behavior:**
```python
{
    "content": [
        {
            "type": "text",
            "text": "Error: An internal error occurred while processing your memory."
        }
    ],
    "isError": True
}
```

**Test Scenario:**
```python
def test_execute_processing_error():
    """Test handling of core engine processing failure."""
    tool = AddMemoryTool(core_engine=mock_engine)
    mock_engine.add_memory = AsyncMock(
        side_effect=ProcessingError("Embedding failed")
    )

    result = await tool.execute({"text": "test"})

    assert result["isError"] is True
    assert "internal error" in result["content"][0]["text"].lower()
    # Ensure internal details NOT exposed
    assert "Embedding" not in result["content"][0]["text"]
```

---

### Edge Case 8: Unexpected Exception

**Scenario:**
Core engine raises unexpected exception (e.g., `AttributeError`).

**Input:**
```python
arguments = {"text": "valid memory text"}
```

**Mock Behavior:**
```python
mock_engine.add_memory = AsyncMock(
    side_effect=AttributeError("'NoneType' object has no attribute 'embed'")
)
```

**Processing:**
1. Validation succeeds
2. Call `core_engine.add_memory()`
3. AttributeError raised (unexpected)
4. Caught by generic `except Exception`
5. Logged as error with full traceback
6. Formatted with generic message

**Expected Behavior:**
```python
{
    "content": [
        {
            "type": "text",
            "text": "Error: An internal error occurred while processing your memory."
        }
    ],
    "isError": True
}
```

**Test Scenario:**
```python
def test_execute_unexpected_error():
    """Test handling of unexpected exceptions."""
    tool = AddMemoryTool(core_engine=mock_engine)
    mock_engine.add_memory = AsyncMock(
        side_effect=AttributeError("Unexpected bug")
    )

    result = await tool.execute({"text": "test"})

    assert result["isError"] is True
    assert "internal error" in result["content"][0]["text"].lower()
    # Ensure internal details NOT exposed
    assert "AttributeError" not in result["content"][0]["text"]
    assert "Unexpected bug" not in result["content"][0]["text"]
```

---

### Edge Case 9: Metadata with Non-JSON-Serializable Values

**Scenario:**
Metadata contains values that can't be serialized to JSON (e.g., functions).

**Input:**
```python
arguments = {
    "text": "valid text",
    "metadata": {
        "callback": lambda x: x,
        "valid_field": "valid_value"
    }
}
```

**Processing:**
1. Pydantic validation attempts to validate metadata
2. Detects non-JSON-serializable content
3. Raises `ValidationError`

**Expected Behavior:**
```python
{
    "content": [
        {
            "type": "text",
            "text": "Error: metadata must be JSON-serializable (no functions, classes, etc.)"
        }
    ],
    "isError": True
}
```

**Test Scenario:**
```python
def test_execute_metadata_not_json_serializable():
    """Test metadata with non-JSON-serializable values."""
    tool = AddMemoryTool(core_engine=mock_engine)

    result = await tool.execute({
        "text": "test",
        "metadata": {"func": lambda: None}
    })

    assert result["isError"] is True
    assert "json" in result["content"][0]["text"].lower() or \
           "serializable" in result["content"][0]["text"].lower()
```

---

### Edge Case 10: Very Long Metadata

**Scenario:**
Metadata dict is very large (e.g., 10,000 nested fields).

**Input:**
```python
arguments = {
    "text": "valid text",
    "metadata": {f"field_{i}": f"value_{i}" for i in range(10000)}
}
```

**Processing:**
1. Pydantic validates structure (may be slow for huge dicts)
2. If valid JSON-serializable, validation passes
3. Passed to core engine
4. Core engine may impose size limits (not MCP layer's responsibility)

**Expected Behavior:**
Success (MCP layer doesn't impose metadata size limits).

**Test Scenario:**
```python
def test_execute_large_metadata():
    """Test with very large metadata dict."""
    tool = AddMemoryTool(core_engine=mock_engine)
    mock_engine.add_memory = AsyncMock(return_value=MemoryResult(
        memory_id="test-id",
        chunks_created=1,
        text_preview="test"
    ))

    large_metadata = {f"field_{i}": f"value_{i}" for i in range(10000)}
    result = await tool.execute({
        "text": "test",
        "metadata": large_metadata
    })

    assert result["isError"] is False
    mock_engine.add_memory.assert_called_once()
```

---

## Test Scenarios (Complete List)

### Happy Path Tests

**1. test_execute_success_minimal**
- **Input:** Valid text, no metadata
- **Expected:** `isError=False`, response contains `memory_id`, `chunks_created`, `text_preview`
- **Verifies:** Basic success case with minimal arguments

**2. test_execute_success_with_metadata**
- **Input:** Valid text + complete metadata dict
- **Expected:** `isError=False`, metadata passed to core engine
- **Verifies:** Metadata handling in success case

**3. test_execute_success_boundary_text_length**
- **Input:** Text with exactly 10,000,000 characters
- **Expected:** `isError=False`, processes successfully
- **Verifies:** Maximum length boundary case

**4. test_execute_success_with_unicode**
- **Input:** Text with Unicode characters (emoji, Chinese, etc.)
- **Expected:** `isError=False`, Unicode preserved
- **Verifies:** UTF-8 encoding support

**5. test_execute_success_with_code**
- **Input:** Python code in text field, language metadata
- **Expected:** `isError=False`, code stored correctly
- **Verifies:** Code memory use case

---

### Validation Error Tests

**6. test_execute_error_empty_text_after_strip**
- **Input:** Text is only whitespace
- **Expected:** `isError=True`, error mentions "empty" and "whitespace"
- **Verifies:** Edge case 1

**7. test_execute_error_text_too_long**
- **Input:** Text with 10,000,001 characters
- **Expected:** `isError=True`, error mentions "exceeds maximum"
- **Verifies:** Edge case 3

**8. test_execute_error_missing_text_field**
- **Input:** Arguments missing "text" key
- **Expected:** `isError=True`, error mentions "required"
- **Verifies:** Edge case 4

**9. test_execute_error_text_wrong_type**
- **Input:** text is integer instead of string
- **Expected:** `isError=True`, error mentions type mismatch
- **Verifies:** Type validation

**10. test_execute_error_metadata_wrong_type**
- **Input:** metadata is string instead of dict
- **Expected:** `isError=True`, error mentions "object/dict"
- **Verifies:** Edge case 5

**11. test_execute_error_metadata_not_json_serializable**
- **Input:** metadata contains function/lambda
- **Expected:** `isError=True`, error mentions "JSON-serializable"
- **Verifies:** Edge case 9

---

### Core Engine Error Tests

**12. test_execute_error_database_failure**
- **Mock:** `core_engine.add_memory()` raises `DatabaseError`
- **Expected:** `isError=True`, generic database message, no internal details
- **Verifies:** Edge case 6

**13. test_execute_error_processing_failure**
- **Mock:** `core_engine.add_memory()` raises `ProcessingError`
- **Expected:** `isError=True`, generic internal error, no details
- **Verifies:** Edge case 7

**14. test_execute_error_unexpected_exception**
- **Mock:** `core_engine.add_memory()` raises `AttributeError`
- **Expected:** `isError=True`, generic internal error, no details
- **Verifies:** Edge case 8

---

### Logging Tests

**15. test_execute_logs_request_start**
- **Verifies:** Request start logged with `request_id`

**16. test_execute_logs_validation_error**
- **Verifies:** Validation errors logged as warnings

**17. test_execute_logs_processing_error**
- **Verifies:** Processing errors logged with traceback

**18. test_execute_logs_success**
- **Verifies:** Success logged with `memory_id` and `chunks_created`

---

### Response Format Tests

**19. test_execute_response_format_success**
- **Verifies:** Success response conforms to MCP schema

**20. test_execute_response_format_error**
- **Verifies:** Error response conforms to MCP schema

**21. test_execute_response_has_single_content_block**
- **Verifies:** `content` array has exactly 1 element

**22. test_execute_response_content_is_text_type**
- **Verifies:** `content[0]["type"]` is always `"text"`

---

### Integration Tests

**23. test_execute_calls_validate_arguments**
- **Verifies:** `_validate_arguments()` method called

**24. test_execute_calls_core_engine_add_memory**
- **Verifies:** `core_engine.add_memory()` called with correct arguments

**25. test_execute_calls_format_success_on_success**
- **Verifies:** `_format_success()` called with result

**26. test_execute_calls_format_error_on_error**
- **Verifies:** `_format_error()` called with exception

---

### Performance Tests

**27. test_execute_performance_small_text**
- **Input:** 1KB text
- **Expected:** Total time < 50ms (excluding core processing)
- **Verifies:** MCP layer overhead minimal

**28. test_execute_performance_large_text**
- **Input:** 10MB text
- **Expected:** Validation completes in < 200ms
- **Verifies:** Validation scales reasonably

---

### Security Tests

**29. test_execute_error_message_sanitization**
- **Verifies:** Internal errors don't leak sensitive info

**30. test_execute_no_stack_trace_in_response**
- **Verifies:** Exception stack traces not in response text

---

## Performance Requirements

### Latency Targets

**MCP Layer Overhead Only:**
- Validation: < 5ms for typical inputs (< 10KB)
- Validation: < 50ms for large inputs (1-10MB)
- Response formatting: < 1ms
- **Total MCP overhead: < 10ms**

**End-to-End (including core):**
- Small text (< 1KB): < 100ms total
- Medium text (1-100KB): < 500ms total
- Large text (1-10MB): < 5s total

### Throughput

**Concurrent Requests:**
- MCP stdio transport: Sequential only (single-threaded)
- If HTTP transport added: Support 100+ req/sec

**Resource Usage:**
- Memory: O(n) where n = text size (2x during validation)
- CPU: O(n) for validation (Pydantic overhead)

### Optimization Notes

**Already Optimal:**
- Validation is unavoidable (security requirement)
- Pydantic is industry-standard fast validator
- Response formatting is O(1)

**Potential Improvements:**
- Stream validation for very large texts (future)
- Pre-compile validation schemas (Pydantic v2 does this)

---

## Security Considerations

### Input Validation

✅ **All inputs validated:**
- Type checking via Pydantic
- Length limits enforced
- UTF-8 encoding verified
- Whitespace sanitized

✅ **Injection prevention:**
- No direct eval() or exec()
- No string formatting of user input into queries
- All DB operations via parameterized queries (in core)

### Error Message Safety

✅ **Safe to expose:**
- Pydantic validation errors (user input issues)
- Generic database unavailable messages

❌ **Never expose:**
- Internal exception messages (e.g., `"NoneType has no attribute..."`)
- Database connection strings or paths
- Stack traces
- File paths
- Configuration details

### Data Protection

**Sensitive Data Handling:**
- Text content may contain PII → logged with care
- Metadata may contain sensitive fields → logged selectively
- Memory IDs are UUIDs → safe to expose

**Logging Guidelines:**
- Log validation errors: Full message (user input)
- Log processing errors: Type + generic message only
- Log unexpected errors: Full traceback to stderr, generic to response

---

## Related Functions

### Calls

**1. `_validate_arguments(arguments: dict) -> Tuple[str, dict]`**
- **Purpose:** Extract and validate text and metadata from arguments
- **When:** First step of execute()
- **Returns:** Tuple of (validated_text, metadata_dict)

**2. `core_engine.add_memory(text: str, metadata: dict) -> MemoryResult`**
- **Purpose:** Process and store memory in core system
- **When:** After successful validation
- **Returns:** MemoryResult with memory_id, chunks_created, text_preview

**3. `_format_success(result: MemoryResult) -> dict`**
- **Purpose:** Convert MemoryResult to MCP success response
- **When:** After successful core processing
- **Returns:** MCP response dict with isError=False

**4. `_format_error(error: Exception) -> dict`**
- **Purpose:** Convert exception to MCP error response
- **When:** When any error occurs
- **Returns:** MCP response dict with isError=True

### Called By

**1. `MCPServer.handle_tool_call(tool_name: str, arguments: dict)`**
- **Purpose:** MCP server routing tool requests
- **When:** Client invokes add_memory tool
- **How:** `await tool.execute(arguments)`

---

## Implementation Notes

### Dependencies

**External Libraries:**
- `pydantic` (>=2.5.0) - Input validation
- `structlog` (>=23.2.0) - Structured logging
- `typing` (stdlib) - Type annotations

**Internal Dependencies:**
- `zapomni_core.MemoryEngine` - Core processing
- `zapomni_core.models.MemoryResult` - Result model
- `zapomni_mcp.schemas.requests.AddMemoryRequest` - Request validation model
- `zapomni_mcp.schemas.responses.AddMemoryResponse` - Response model

### Known Limitations

**1. Synchronous Validation:**
- Pydantic validation is CPU-bound
- For very large texts (10MB), validation may take 50-100ms
- Can't be parallelized (single request processing)

**2. Memory Usage:**
- Pydantic creates copy of data during validation
- Peak memory: ~2x text size during validation
- Acceptable for 10MB limit

**3. Error Granularity:**
- Some errors use generic messages for security
- May make debugging harder for users
- Detailed errors in stderr logs for operators

### Future Enhancements

**1. Streaming Validation:**
- For texts > 10MB, consider streaming validation
- Validate in chunks, process incrementally
- Requires core engine streaming support

**2. Async Logging:**
- Current logging is synchronous (minor overhead)
- Could use async logging for high-throughput scenarios

**3. Response Caching:**
- Not applicable (each memory is unique)
- But could cache validation schemas

**4. Metrics Collection:**
- Add Prometheus metrics for observability
- Track: request rate, error rate, latency percentiles

---

## References

### Component Spec
- [AddMemoryTool Component Specification](../level2/add_memory_tool_component.md)

### Module Spec
- [Zapomni MCP Module Specification](../level1/zapomni_mcp_module.md)

### Related Function Specs
- `AddMemoryTool._validate_arguments()` (Level 3) - To be created
- `AddMemoryTool._format_success()` (Level 3) - To be created
- `AddMemoryTool._format_error()` (Level 3) - To be created

### External Documentation
- [MCP Specification - Tools](https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/tools/)
- [Pydantic Documentation](https://docs.pydantic.dev/latest/)
- [Structlog Documentation](https://www.structlog.org/)

---

## Document Status

**Version:** 1.0
**Created:** 2025-11-23
**Author:** Goncharenko Anton aka alienxs2
**Copyright:** Copyright (c) 2025 Goncharenko Anton aka alienxs2
**License:** MIT License
**Status:** Draft - Ready for Review

**Next Steps:**
1. Review against function specification template
2. Verify alignment with component spec
3. Create test implementation from scenarios
4. Proceed to implementation

---

**Estimated Implementation Effort:** 1-2 hours (execute method only)
**Lines of Code (Estimated):** ~40 lines
**Test Coverage Target:** 95%+ (30 test scenarios defined)
**Test File:** `tests/unit/mcp/tools/test_add_memory_tool_execute.py`
