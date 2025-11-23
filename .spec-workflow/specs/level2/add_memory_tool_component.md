# AddMemoryTool - Component Specification

**Level:** 2 (Component)
**Module:** zapomni_mcp
**Author:** Goncharenko Anton aka alienxs2
**Status:** Draft
**Version:** 1.0
**Created:** 2025-11-23

---

## Overview

### Purpose

The AddMemoryTool component implements the MCP tool interface for adding memories (text or code) to the Zapomni knowledge graph. It serves as the primary entry point for storing new information in the system.

**Core Responsibilities:**
- Validate incoming `add_memory` tool requests from MCP clients
- Extract and sanitize text content and optional metadata
- Delegate memory processing to `zapomni_core.MemoryEngine`
- Format processing results as MCP-compliant responses
- Handle and report errors in MCP protocol format

**Design Philosophy:** Pure adapter - validates inputs, delegates processing, formats outputs. Contains zero business logic.

### Responsibilities

1. **Input Validation**
   - Validate `text` parameter is present, non-empty, within size limits
   - Validate `metadata` parameter structure (if provided)
   - Sanitize inputs to prevent injection attacks
   - Provide clear, actionable error messages

2. **Core Engine Delegation**
   - Call `MemoryEngine.add_memory(text, metadata)`
   - Pass validated inputs without transformation
   - Handle core engine exceptions gracefully

3. **Response Formatting**
   - Convert core results to MCP response format
   - Include memory ID, chunk count, preview
   - Format errors as MCP error responses
   - Ensure JSON serialization compatibility

4. **Error Handling**
   - Catch validation errors → 400-level MCP errors
   - Catch processing errors → 500-level MCP errors
   - Catch database errors → 503-level MCP errors
   - Log all errors to stderr with context

### Position in Module

**Component Location:** `zapomni_mcp.tools.add_memory`

**Relationship to Module:**
```
zapomni_mcp/
├── server.py                 # MCPServer (orchestrator)
│   └── registers → AddMemoryTool
├── tools/
│   ├── add_memory.py         # ← THIS COMPONENT
│   ├── search_memory.py      # Sibling: SearchMemoryTool
│   └── get_stats.py          # Sibling: GetStatsTool
└── schemas/
    ├── requests.py           # AddMemoryRequest (used here)
    └── responses.py          # AddMemoryResponse (used here)
```

**Lifecycle:**
1. MCPServer instantiates AddMemoryTool during initialization
2. MCPServer registers tool with MCP SDK
3. MCPServer routes `add_memory` requests to this tool's `execute()` method
4. Tool lives for entire server lifetime (no per-request instantiation)

---

## Class Definition

### Class Diagram

```
┌─────────────────────────────────────────────┐
│           AddMemoryTool                     │
├─────────────────────────────────────────────┤
│ - name: str = "add_memory"                  │
│ - description: str                          │
│ - input_schema: dict[str, Any]              │
│ - core_engine: MemoryEngine                 │
│ - logger: BoundLogger                       │
├─────────────────────────────────────────────┤
│ + __init__(core_engine: MemoryEngine)       │
│ + execute(arguments: dict) -> dict          │
│ - _validate_arguments(args: dict) -> tuple  │
│ - _format_success(result: MemoryResult)     │
│ - _format_error(error: Exception) -> dict   │
└─────────────────────────────────────────────┘
            │
            │ implements
            ↓
┌─────────────────────────────────────────────┐
│         MCPTool (Protocol)                  │
├─────────────────────────────────────────────┤
│ + name: str                                 │
│ + description: str                          │
│ + input_schema: dict[str, Any]              │
│ + execute(arguments: dict) -> dict          │
└─────────────────────────────────────────────┘
```

### Full Class Signature

```python
from typing import Any, Dict, Tuple
import structlog
from pydantic import ValidationError

from zapomni_core import MemoryEngine
from zapomni_core.models import MemoryResult
from zapomni_mcp.schemas.requests import AddMemoryRequest
from zapomni_mcp.schemas.responses import AddMemoryResponse


class AddMemoryTool:
    """
    MCP tool for adding memories to the Zapomni knowledge graph.

    This tool provides the primary interface for storing text or code snippets
    in the memory system. It validates inputs, delegates processing to the core
    engine, and returns MCP-formatted responses.

    Attributes:
        name: Tool identifier ("add_memory") used by MCP clients
        description: Human-readable description shown in tool listings
        input_schema: JSON Schema defining valid input arguments
        core_engine: Reference to MemoryEngine for processing
        logger: Structured logger instance with context binding

    Example:
        >>> from zapomni_core import MemoryEngine
        >>> engine = MemoryEngine(config)
        >>> tool = AddMemoryTool(core_engine=engine)
        >>> result = await tool.execute({
        ...     "text": "Python is a high-level programming language",
        ...     "metadata": {"source": "user", "tags": ["programming"]}
        ... })
        >>> print(result)
        {
            "content": [{
                "type": "text",
                "text": "Memory stored successfully. ID: 550e8400-..."
            }],
            "isError": false
        }

    Thread Safety:
        This class is NOT thread-safe. It should be used in async/await context
        only. The core_engine handles its own thread safety.

    Performance:
        - Validation overhead: < 5ms for typical inputs
        - Core delegation: O(1) function call
        - Response formatting: < 5ms
        - Total MCP layer overhead: < 10ms
    """

    # Class constants (MCP tool metadata)
    name: str = "add_memory"
    description: str = (
        "Add a memory (text or code) to the knowledge graph. "
        "The memory will be processed, chunked, embedded, and stored for later retrieval."
    )

    # JSON Schema for MCP input validation
    input_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": (
                    "Text content to remember. Can be natural language, code, "
                    "documentation, or any UTF-8 text. Maximum 10MB."
                ),
                "minLength": 1,
                "maxLength": 10_000_000
            },
            "metadata": {
                "type": "object",
                "description": (
                    "Optional metadata to attach to this memory. "
                    "Useful for filtering and organization."
                ),
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Source of the memory (e.g., 'user', 'api', 'file')"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for categorization"
                    },
                    "timestamp": {
                        "type": "string",
                        "format": "date-time",
                        "description": "Optional timestamp (ISO 8601 format)"
                    },
                    "language": {
                        "type": "string",
                        "description": "Programming language if text is code (e.g., 'python')"
                    }
                },
                "additionalProperties": true  # Allow custom metadata fields
            }
        },
        "required": ["text"],
        "additionalProperties": false
    }

    def __init__(self, core_engine: MemoryEngine) -> None:
        """
        Initialize AddMemoryTool with core engine reference.

        Args:
            core_engine: MemoryEngine instance for processing memories.
                Must be initialized and connected to database.

        Raises:
            TypeError: If core_engine is not a MemoryEngine instance
            ValueError: If core_engine is not initialized

        Example:
            >>> engine = MemoryEngine(config)
            >>> await engine.initialize()
            >>> tool = AddMemoryTool(core_engine=engine)
        """
        if not isinstance(core_engine, MemoryEngine):
            raise TypeError(
                f"core_engine must be MemoryEngine instance, got {type(core_engine)}"
            )

        self.core_engine = core_engine
        self.logger = structlog.get_logger().bind(tool=self.name)

        self.logger.info("add_memory_tool_initialized")

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute add_memory tool with provided arguments.

        This is the main entry point called by the MCP server when a client
        invokes the add_memory tool. It validates inputs, processes the memory
        through the core engine, and returns a formatted MCP response.

        Args:
            arguments: Dictionary containing:
                - text (str, required): Memory content to store
                - metadata (dict, optional): Additional metadata

        Returns:
            MCP-formatted response dictionary:
            {
                "content": [
                    {
                        "type": "text",
                        "text": "Memory stored successfully. ID: {memory_id}\\n"
                                "Chunks created: {chunk_count}\\n"
                                "Preview: {text_preview}"
                    }
                ],
                "isError": false
            }

            On error:
            {
                "content": [
                    {
                        "type": "text",
                        "text": "Error: {error_message}"
                    }
                ],
                "isError": true
            }

        Raises:
            This method never raises exceptions. All errors are caught and
            returned as MCP error responses.

        Example:
            >>> result = await tool.execute({
            ...     "text": "Claude is an AI assistant",
            ...     "metadata": {"source": "user"}
            ... })
            >>> print(result["isError"])
            False
        """
        request_id = id(arguments)  # Simple request ID for logging
        self.logger = self.logger.bind(request_id=request_id)

        try:
            # Step 1: Validate and extract arguments
            self.logger.info("validating_arguments")
            text, metadata = self._validate_arguments(arguments)

            # Step 2: Process memory via core engine
            self.logger.info(
                "processing_memory",
                text_length=len(text),
                has_metadata=bool(metadata)
            )
            result = await self.core_engine.add_memory(
                text=text,
                metadata=metadata or {}
            )

            # Step 3: Format success response
            self.logger.info(
                "memory_added_successfully",
                memory_id=result.memory_id,
                chunks=result.chunks_created
            )
            return self._format_success(result)

        except ValidationError as e:
            # Input validation failed
            self.logger.warning("validation_error", error=str(e))
            return self._format_error(e)

        except Exception as e:
            # Core engine or unexpected error
            self.logger.error(
                "processing_error",
                error_type=type(e).__name__,
                error=str(e),
                exc_info=True
            )
            return self._format_error(e)

    def _validate_arguments(
        self,
        arguments: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Validate and extract arguments from MCP request.

        Args:
            arguments: Raw arguments dictionary from MCP client

        Returns:
            Tuple of (text, metadata) where:
            - text: Validated and sanitized text content
            - metadata: Validated metadata dict (or empty dict)

        Raises:
            ValidationError: If arguments don't match schema

        Implementation Notes:
            - Uses Pydantic model for validation (AddMemoryRequest)
            - Strips whitespace from text
            - Ensures metadata is JSON-serializable
        """
        # Validate using Pydantic model
        request = AddMemoryRequest(**arguments)

        # Extract and sanitize
        text = request.text.strip()
        metadata = request.metadata or {}

        return text, metadata

    def _format_success(self, result: MemoryResult) -> Dict[str, Any]:
        """
        Format successful memory addition as MCP response.

        Args:
            result: MemoryResult from core engine containing:
                - memory_id: UUID of stored memory
                - chunks_created: Number of chunks generated
                - text_preview: First 100 chars of text

        Returns:
            MCP response dictionary with content and isError=false

        Example Output:
            {
                "content": [{
                    "type": "text",
                    "text": "Memory stored successfully. ID: 550e8400-e29b-41d4-a716-446655440000\\nChunks created: 3\\nPreview: Python is a high-level..."
                }],
                "isError": false
            }
        """
        # Create Pydantic response model
        response = AddMemoryResponse(
            status="success",
            memory_id=result.memory_id,
            chunks_created=result.chunks_created,
            text_preview=result.text_preview,
            error=None
        )

        # Format as MCP content block
        message = (
            f"Memory stored successfully.\n"
            f"ID: {response.memory_id}\n"
            f"Chunks created: {response.chunks_created}\n"
            f"Preview: {response.text_preview}"
        )

        return {
            "content": [
                {
                    "type": "text",
                    "text": message
                }
            ],
            "isError": False
        }

    def _format_error(self, error: Exception) -> Dict[str, Any]:
        """
        Format error as MCP error response.

        Args:
            error: Exception that occurred during processing

        Returns:
            MCP error response dictionary with content and isError=true

        Error Message Guidelines:
            - ValidationError: Include field name and constraint
            - ProcessingError: Generic message (no internal details)
            - DatabaseError: Suggest retry
            - Unknown errors: Generic "internal error" message

        Example Output:
            {
                "content": [{
                    "type": "text",
                    "text": "Error: text cannot be empty or whitespace-only"
                }],
                "isError": true
            }
        """
        # Determine error message based on exception type
        if isinstance(error, ValidationError):
            # Pydantic validation error - safe to expose
            error_msg = str(error)
        elif hasattr(error, '__class__') and 'Database' in error.__class__.__name__:
            # Database error - suggest retry
            error_msg = "Database temporarily unavailable. Please retry in a few seconds."
        else:
            # Unknown error - generic message for security
            error_msg = "An internal error occurred while processing your memory."

        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Error: {error_msg}"
                }
            ],
            "isError": True
        }
```

---

## Dependencies

### Component Dependencies

**Internal (Zapomni):**

1. **zapomni_core.MemoryEngine**
   - **Purpose:** Core processing engine for memory operations
   - **Used For:** `add_memory(text, metadata)` method
   - **Dependency Type:** Constructor injection
   - **Interface:**
     ```python
     class MemoryEngine:
         async def add_memory(
             text: str,
             metadata: Dict[str, Any]
         ) -> MemoryResult
     ```

2. **zapomni_core.models.MemoryResult**
   - **Purpose:** Data model for memory processing results
   - **Used For:** Return type from `add_memory()`
   - **Attributes:**
     - `memory_id: str` - UUID of stored memory
     - `chunks_created: int` - Number of chunks generated
     - `text_preview: str` - First 100 chars

3. **zapomni_mcp.schemas.requests.AddMemoryRequest**
   - **Purpose:** Pydantic model for request validation
   - **Used For:** Validating `arguments` dictionary
   - **Fields:** `text: str`, `metadata: Optional[Dict]`

4. **zapomni_mcp.schemas.responses.AddMemoryResponse**
   - **Purpose:** Pydantic model for response structure
   - **Used For:** Validating response before formatting
   - **Fields:** `status`, `memory_id`, `chunks_created`, `text_preview`, `error`

### External Libraries

1. **structlog**
   - **Purpose:** Structured logging with context binding
   - **Used For:** Logging operations and errors
   - **Version:** `>=23.2.0`

2. **pydantic**
   - **Purpose:** Data validation and parsing
   - **Used For:** Request/response validation
   - **Version:** `>=2.5.0`

3. **typing (stdlib)**
   - **Purpose:** Type hints for static analysis
   - **Used For:** All type annotations

### Dependency Injection

**Pattern:** Constructor injection

**Injected Dependencies:**
- `core_engine: MemoryEngine` - Injected via `__init__`

**Rationale:**
- Enables easy testing (mock MemoryEngine)
- Clear dependency graph
- No hidden dependencies
- Follows Dependency Inversion Principle

**Example:**
```python
# Production
engine = MemoryEngine(config)
tool = AddMemoryTool(core_engine=engine)

# Testing
mock_engine = Mock(spec=MemoryEngine)
tool = AddMemoryTool(core_engine=mock_engine)
```

---

## State Management

### Attributes

**Immutable (Class Constants):**
- `name: str = "add_memory"` - Tool identifier, never changes
- `description: str` - Tool description, never changes
- `input_schema: dict` - JSON Schema, never changes

**Mutable (Instance Attributes):**
- `core_engine: MemoryEngine` - Reference to core engine, set once in `__init__`, never modified
- `logger: BoundLogger` - Logger with tool context, set once in `__init__`, rebound per request

**Lifetime:**
- Tool instance lives for entire server lifetime
- No per-request state (all state in function arguments)
- Logger binding is request-scoped (using `.bind()`)

### State Transitions

```
[Not Initialized]
    ↓ __init__(core_engine)
[Initialized - Idle]
    ↓ execute(arguments) called
[Validating Arguments]
    ↓ validation success
[Processing Memory]
    ↓ core_engine.add_memory() returns
[Formatting Response]
    ↓ return formatted result
[Initialized - Idle]  (ready for next request)
```

**Key Points:**
- No persistent state between requests
- Each `execute()` call is independent
- Errors reset state to Idle (no corruption)

### Thread Safety

**Is this component thread-safe?** No, by design.

**Concurrency Constraints:**
- Must be used in async/await context only
- MCP server processes requests sequentially (stdio limitation)
- No concurrent `execute()` calls possible
- `core_engine` is responsible for its own thread safety

**Synchronization Mechanisms:** None needed (single-threaded execution)

**Future Considerations:**
- If HTTP transport added (concurrent requests), would need:
  - Read-only attributes (already the case)
  - No shared mutable state (already the case)
  - Thread-safe logger (structlog is thread-safe)
- **Conclusion:** Already prepared for multi-threaded use if needed

---

## Public Methods (Detailed)

### Method 1: `__init__`

**Signature:**
```python
def __init__(self, core_engine: MemoryEngine) -> None
```

**Purpose:** Initialize AddMemoryTool with core engine dependency

**Parameters:**

- `core_engine`: MemoryEngine
  - **Description:** Initialized MemoryEngine instance for processing
  - **Constraints:**
    - Must be instance of `MemoryEngine` (type check enforced)
    - Must be initialized (connection to database established)
  - **Example:** `engine = MemoryEngine(config); await engine.initialize()`

**Returns:** None (constructor)

**Raises:**
- `TypeError`: If `core_engine` is not a `MemoryEngine` instance
- `ValueError`: If `core_engine` is not initialized (future: add check)

**Preconditions:**
- `core_engine` must be a valid MemoryEngine instance
- MemoryEngine should be initialized and connected

**Postconditions:**
- `self.core_engine` is set to provided engine
- `self.logger` is initialized with tool context
- Tool is ready to process requests

**Algorithm Outline:**
```
1. Validate core_engine type (isinstance check)
2. If invalid type → raise TypeError
3. Store reference to core_engine
4. Initialize logger with tool context binding
5. Log initialization success
```

**Edge Cases:**
1. `core_engine` is None → TypeError
2. `core_engine` is wrong type → TypeError
3. `core_engine` not initialized → Should fail on first execute() call

**Related Methods:**
- Called by: `MCPServer` during server initialization
- Calls: `structlog.get_logger()`

---

### Method 2: `execute` (Main Entry Point)

**Signature:**
```python
async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]
```

**Purpose:** Execute add_memory tool with MCP arguments

**Parameters:**

- `arguments`: Dict[str, Any]
  - **Description:** Dictionary of tool arguments from MCP client
  - **Required Keys:**
    - `"text"`: str - Memory content (1 to 10,000,000 chars)
  - **Optional Keys:**
    - `"metadata"`: dict - Additional metadata (any JSON-serializable data)
  - **Constraints:**
    - Must be valid Python dict
    - `text` must be non-empty after stripping whitespace
    - `text` must be valid UTF-8
    - `metadata` must be JSON-serializable (if provided)
  - **Example:**
    ```python
    {
        "text": "Python is a programming language",
        "metadata": {
            "source": "user",
            "tags": ["programming", "python"],
            "language": "python"
        }
    }
    ```

**Returns:** Dict[str, Any]

**Success Response Structure:**
```python
{
    "content": [
        {
            "type": "text",
            "text": "Memory stored successfully.\n"
                    "ID: 550e8400-e29b-41d4-a716-446655440000\n"
                    "Chunks created: 3\n"
                    "Preview: Python is a programming..."
        }
    ],
    "isError": False
}
```

**Error Response Structure:**
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

**Raises:** Never raises exceptions (all caught and returned as error responses)

**Preconditions:**
- Tool must be initialized (`__init__` called)
- `core_engine` must be connected and ready
- `arguments` should be a dictionary (validated internally)

**Postconditions:**
- If success: Memory stored in database, response contains memory_id
- If error: No memory stored, error message returned
- All operations logged to stderr
- No state changes in tool instance

**Algorithm Outline:**
```
1. Generate request_id for logging
2. Bind request_id to logger context
3. TRY:
     4. Validate arguments using _validate_arguments()
     5. Extract text and metadata
     6. Call core_engine.add_memory(text, metadata)
     7. Wait for MemoryResult
     8. Format success response using _format_success()
     9. Return formatted response
   EXCEPT ValidationError:
     10. Log warning
     11. Format error using _format_error()
     12. Return error response
   EXCEPT ANY Exception:
     13. Log error with full traceback
     14. Format error using _format_error()
     15. Return error response
```

**Edge Cases:**

1. **Empty text (after stripping)**
   - **Input:** `{"text": "   "}`
   - **Behavior:** ValidationError raised
   - **Output:** `{"content": [{"type": "text", "text": "Error: text cannot be empty..."}], "isError": true}`

2. **Text too long (> 10MB)**
   - **Input:** `{"text": "x" * 10_000_001}`
   - **Behavior:** ValidationError raised
   - **Output:** `{"content": [{"type": "text", "text": "Error: text exceeds maximum length"}], "isError": true}`

3. **Missing text parameter**
   - **Input:** `{"metadata": {"source": "user"}}`
   - **Behavior:** ValidationError raised
   - **Output:** `{"content": [{"type": "text", "text": "Error: field required"}], "isError": true}`

4. **Invalid metadata type**
   - **Input:** `{"text": "valid", "metadata": "invalid"}`
   - **Behavior:** ValidationError raised
   - **Output:** Error explaining metadata must be object

5. **Database connection lost**
   - **Input:** Valid request, but DB disconnected
   - **Behavior:** MemoryEngine raises DatabaseError
   - **Output:** `{"content": [{"type": "text", "text": "Error: Database temporarily unavailable..."}], "isError": true}`

6. **Core engine processing failure**
   - **Input:** Valid request, but embedding generation fails
   - **Behavior:** MemoryEngine raises ProcessingError
   - **Output:** `{"content": [{"type": "text", "text": "Error: An internal error occurred..."}], "isError": true}`

**Related Methods:**
- Calls: `_validate_arguments()`, `core_engine.add_memory()`, `_format_success()`, `_format_error()`
- Called by: `MCPServer.handle_tool_call()`

---

### Method 3: `_validate_arguments` (Private)

**Signature:**
```python
def _validate_arguments(
    self,
    arguments: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]
```

**Purpose:** Validate and extract text and metadata from arguments

**Parameters:**

- `arguments`: Dict[str, Any]
  - **Description:** Raw arguments from MCP request
  - **Expected Structure:** See `execute()` method
  - **Validation:** Performed using Pydantic AddMemoryRequest model

**Returns:** Tuple[str, Dict[str, Any]]
- **Index 0:** Validated and sanitized text (whitespace stripped)
- **Index 1:** Validated metadata dictionary (or empty dict if not provided)

**Raises:**
- `ValidationError`: If arguments don't match schema

**Preconditions:**
- `arguments` is a dictionary

**Postconditions:**
- Returned text is non-empty, within size limits
- Returned metadata is JSON-serializable

**Algorithm Outline:**
```
1. Create AddMemoryRequest instance from arguments (Pydantic validation)
2. Extract text field from validated request
3. Strip whitespace from text
4. Extract metadata field (default to empty dict if None)
5. Return tuple (text, metadata)
```

**Edge Cases:**
- All edge cases handled by Pydantic validation
- Raises `ValidationError` with detailed error messages

**Related Methods:**
- Called by: `execute()`

---

### Method 4: `_format_success` (Private)

**Signature:**
```python
def _format_success(self, result: MemoryResult) -> Dict[str, Any]
```

**Purpose:** Convert MemoryResult to MCP success response

**Parameters:**

- `result`: MemoryResult
  - **Description:** Result object from core engine
  - **Required Fields:**
    - `memory_id: str` - UUID of stored memory
    - `chunks_created: int` - Number of chunks generated
    - `text_preview: str` - First 100 chars of text

**Returns:** MCP-formatted success response dictionary

**Raises:** Never (assumes result is valid)

**Preconditions:**
- `result` is a valid MemoryResult instance

**Postconditions:**
- Returned dict conforms to MCP response schema
- `isError` is always `False`

**Algorithm Outline:**
```
1. Create AddMemoryResponse Pydantic model from result
2. Build human-readable message string
3. Format as MCP content block
4. Return dictionary with content and isError=false
```

**Related Methods:**
- Called by: `execute()`

---

### Method 5: `_format_error` (Private)

**Signature:**
```python
def _format_error(self, error: Exception) -> Dict[str, Any]
```

**Purpose:** Convert exception to MCP error response

**Parameters:**

- `error`: Exception
  - **Description:** Exception that occurred during processing
  - **Types Handled:**
    - `ValidationError` - User input errors (safe to expose)
    - `DatabaseError` - DB connection errors (suggest retry)
    - Other exceptions - Generic error message

**Returns:** MCP-formatted error response dictionary

**Raises:** Never (error handler itself doesn't raise)

**Preconditions:**
- `error` is an Exception instance

**Postconditions:**
- Returned dict conforms to MCP response schema
- `isError` is always `True`
- Error message is safe to expose to client

**Algorithm Outline:**
```
1. Check exception type
2. IF ValidationError:
     3. Extract detailed validation error message
   ELIF DatabaseError:
     4. Use generic "database unavailable" message
   ELSE:
     5. Use generic "internal error" message
6. Format as MCP content block
7. Return dictionary with content and isError=true
```

**Security Considerations:**
- Never expose internal paths, stack traces, or DB details
- ValidationError messages are safe (user input validation)
- All other errors use generic messages

**Related Methods:**
- Called by: `execute()`

---

## Error Handling

### Exceptions Defined

**No custom exceptions defined in this component.**

**Rationale:** Component uses exceptions from dependencies:
- `pydantic.ValidationError` - Input validation failures
- `zapomni_core.ProcessingError` - Core processing failures
- `zapomni_core.DatabaseError` - Database operation failures

### Error Recovery

**Strategy:** Fail fast, return clear error, no retries

**Validation Errors (ValidationError):**
- **When:** Invalid input arguments
- **Recovery:** None - return error to client immediately
- **Retry:** Client should fix input and retry
- **Example:** "text cannot be empty"

**Processing Errors (ProcessingError):**
- **When:** Core engine fails (embedding generation, chunking)
- **Recovery:** None at MCP layer (core should handle retries)
- **Retry:** Client can retry request (may succeed on retry)
- **Example:** "An internal error occurred"

**Database Errors (DatabaseError):**
- **When:** Database connection lost or query fails
- **Recovery:** None at MCP layer
- **Retry:** Client should retry with backoff
- **Example:** "Database temporarily unavailable"

**Unknown Errors (Exception):**
- **When:** Unexpected exceptions
- **Recovery:** None - log and return generic error
- **Retry:** Unknown if retry will help
- **Example:** "An internal error occurred"

### Error Propagation

**All exceptions caught at top level:**
- `execute()` method has try/except wrapping entire flow
- No exceptions propagate to MCP server
- All errors returned as MCP error responses
- All errors logged to stderr with context

**Error Logging:**
```python
# Validation errors - warning level
logger.warning("validation_error", error=str(e))

# Processing/database errors - error level
logger.error("processing_error", error_type=..., error=..., exc_info=True)
```

---

## Usage Examples

### Basic Usage

```python
from zapomni_core import MemoryEngine
from zapomni_mcp.tools.add_memory import AddMemoryTool

# Initialize core engine
config = load_config()
engine = MemoryEngine(config)
await engine.initialize()

# Create tool
tool = AddMemoryTool(core_engine=engine)

# Execute with valid input
result = await tool.execute({
    "text": "Python is a high-level programming language",
    "metadata": {
        "source": "user",
        "tags": ["programming", "python"]
    }
})

# Check result
if result["isError"]:
    print(f"Error: {result['content'][0]['text']}")
else:
    print(f"Success: {result['content'][0]['text']}")
    # Output:
    # Success: Memory stored successfully.
    # ID: 550e8400-e29b-41d4-a716-446655440000
    # Chunks created: 3
    # Preview: Python is a high-level programming...
```

### Minimal Valid Request

```python
# Simplest possible request (only required fields)
result = await tool.execute({
    "text": "Remember this"
})

# metadata defaults to empty dict
# Still creates memory successfully
```

### Code Memory Example

```python
# Storing code with language metadata
result = await tool.execute({
    "text": """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
""",
    "metadata": {
        "language": "python",
        "source": "code_file",
        "tags": ["algorithms", "recursion"]
    }
})
```

### Error Handling Example

```python
# Handle validation errors
result = await tool.execute({
    "text": "   ",  # Invalid: whitespace only
    "metadata": {"source": "test"}
})

assert result["isError"] is True
assert "empty" in result["content"][0]["text"].lower()
```

### Advanced Usage (with logging)

```python
import structlog

logger = structlog.get_logger()

# Process multiple memories
memories = [
    "First memory",
    "Second memory",
    "Third memory"
]

for i, memory_text in enumerate(memories):
    logger.info("processing_memory", index=i)

    result = await tool.execute({
        "text": memory_text,
        "metadata": {"batch_id": "batch_001", "index": i}
    })

    if result["isError"]:
        logger.error("batch_memory_failed", index=i, error=result["content"][0]["text"])
    else:
        logger.info("batch_memory_success", index=i)
```

---

## Testing Approach

### Unit Tests Required

**Test File:** `tests/unit/mcp/tools/test_add_memory_tool.py`

**Test Scenarios:**

1. **test_init_success**
   - Verify tool initializes with valid MemoryEngine
   - Check logger is bound with tool context
   - Assert name, description, input_schema are set

2. **test_init_invalid_core_engine_type**
   - Pass non-MemoryEngine object
   - Expect TypeError with clear message

3. **test_execute_success_minimal**
   - Valid text, no metadata
   - Mock core_engine.add_memory to return success
   - Verify MCP response format
   - Assert isError is False

4. **test_execute_success_with_metadata**
   - Valid text + metadata
   - Verify metadata passed to core engine
   - Check response includes memory_id, chunks, preview

5. **test_execute_validation_error_empty_text**
   - Empty text: `{"text": ""}`
   - Expect ValidationError caught
   - Verify error response with isError=True

6. **test_execute_validation_error_whitespace_only**
   - Whitespace text: `{"text": "   "}`
   - Expect ValidationError
   - Check error message clarity

7. **test_execute_validation_error_missing_text**
   - Arguments missing text: `{"metadata": {}}`
   - Expect ValidationError
   - Verify error mentions "required"

8. **test_execute_validation_error_text_too_long**
   - Text exceeding 10MB
   - Expect ValidationError
   - Check error message mentions length

9. **test_execute_validation_error_invalid_metadata_type**
   - metadata is string instead of dict
   - Expect ValidationError
   - Verify error explains type mismatch

10. **test_execute_processing_error**
    - Mock core_engine.add_memory to raise ProcessingError
    - Verify error caught and formatted
    - Check generic error message (no internal details)

11. **test_execute_database_error**
    - Mock core_engine.add_memory to raise DatabaseError
    - Verify error message suggests retry

12. **test_execute_unexpected_error**
    - Mock core_engine.add_memory to raise arbitrary exception
    - Verify generic error response
    - Check error logged with traceback

13. **test_validate_arguments_success**
    - Direct test of _validate_arguments
    - Valid input → returns (text, metadata)

14. **test_format_success**
    - Direct test of _format_success
    - MemoryResult → MCP response
    - Verify content structure

15. **test_format_error_validation_error**
    - ValidationError → detailed error message

16. **test_format_error_database_error**
    - DatabaseError → retry suggestion

17. **test_format_error_generic**
    - Generic Exception → generic message

### Mocking Strategy

**Mock Objects:**

1. **Mock MemoryEngine:**
   ```python
   from unittest.mock import AsyncMock, Mock

   @pytest.fixture
   def mock_core_engine():
       engine = Mock(spec=MemoryEngine)
       engine.add_memory = AsyncMock()
       return engine
   ```

2. **Mock MemoryResult:**
   ```python
   from zapomni_core.models import MemoryResult

   @pytest.fixture
   def mock_memory_result():
       return MemoryResult(
           memory_id="550e8400-e29b-41d4-a716-446655440000",
           chunks_created=3,
           text_preview="Sample text preview..."
       )
   ```

**Example Test:**
```python
import pytest
from unittest.mock import AsyncMock, Mock
from zapomni_mcp.tools.add_memory import AddMemoryTool
from zapomni_core.models import MemoryResult

@pytest.mark.asyncio
async def test_execute_success_minimal():
    """Test successful memory addition with minimal input."""
    # Setup
    mock_engine = Mock()
    mock_engine.add_memory = AsyncMock(return_value=MemoryResult(
        memory_id="test-id-123",
        chunks_created=2,
        text_preview="Test preview"
    ))

    tool = AddMemoryTool(core_engine=mock_engine)

    # Execute
    result = await tool.execute({
        "text": "Test memory"
    })

    # Verify
    assert result["isError"] is False
    assert "test-id-123" in result["content"][0]["text"]
    assert "Chunks created: 2" in result["content"][0]["text"]

    # Verify core was called correctly
    mock_engine.add_memory.assert_called_once_with(
        text="Test memory",
        metadata={}
    )
```

### Integration Tests

**Test File:** `tests/integration/mcp/test_add_memory_integration.py`

**Scope:** Test AddMemoryTool with real MemoryEngine, mock database

**Scenarios:**

1. **test_add_memory_end_to_end**
   - Real MemoryEngine instance
   - Mock FalkorDB client
   - Verify entire flow: validate → process → store → respond

2. **test_add_memory_with_real_embeddings**
   - Real embedding generation
   - Mock database storage
   - Verify embeddings created correctly

3. **test_error_propagation_from_core**
   - Real MemoryEngine raises error
   - Verify error propagates correctly through tool

**Example:**
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_add_memory_end_to_end(mock_db_client):
    """Integration test with real core engine."""
    # Setup real core with mocked DB
    config = TestConfig()
    engine = MemoryEngine(config, db_client=mock_db_client)
    await engine.initialize()

    tool = AddMemoryTool(core_engine=engine)

    # Execute
    result = await tool.execute({
        "text": "Integration test memory",
        "metadata": {"test": True}
    })

    # Verify
    assert result["isError"] is False
    assert mock_db_client.store_memory.called
```

---

## Performance Considerations

### Time Complexity

**execute() method:**
- Validation: O(n) where n = text length (Pydantic validation)
- Core delegation: O(1) function call
- Response formatting: O(1) string formatting
- **Total:** O(n) dominated by validation

**Breakdown:**
- `_validate_arguments()`: O(n) - text length validation
- `core_engine.add_memory()`: O(m) - core processing time (not counted here)
- `_format_success()`: O(1) - fixed size response
- `_format_error()`: O(1) - fixed size error message

**Optimization:** Already optimal for MCP layer responsibilities

### Space Complexity

**Memory Usage:**
- Input arguments: O(n) where n = text size (up to 10MB)
- Pydantic model: O(n) duplicate during validation
- Response dict: O(1) small fixed size
- Logger context: O(1) few KB

**Peak Memory:** ~2x text size during validation (Pydantic creates copy)

**Optimization Opportunities:**
- Use streaming validation for very large texts (future)
- Current approach acceptable for 10MB limit

### Optimization Opportunities

**Current Performance (Estimated):**
- Validation: 2-5ms for typical 10KB text
- Core call overhead: < 1ms
- Response formatting: < 1ms
- **Total MCP overhead: < 10ms**

**Potential Optimizations:**

1. **Lazy Validation (if needed):**
   - Currently validates entire text eagerly
   - Could validate in chunks for very large texts
   - **Trade-off:** Complexity vs. marginal speed gain

2. **Response Caching (not applicable):**
   - Each memory is unique (no cache hits)
   - Not beneficial for this use case

3. **Pre-compiled Regex (if added):**
   - Currently no regex validation
   - Could add for sanitization if needed

**Conclusion:** Current implementation is already optimal for this layer. Core processing time dominates (embeddings, DB), making MCP layer overhead negligible.

---

## References

### Module Spec
- **zapomni_mcp_module.md** (Level 1) - Parent module specification

### Related Component Specs (Level 2)
- **search_memory_tool_component.md** - Sibling component for search
- **get_stats_tool_component.md** - Sibling component for statistics

### Core Dependencies
- **zapomni_core_module.md** (Level 1) - MemoryEngine interface
- **data_flow_architecture.md** (Level 1) - Memory processing flow

### External Docs
- **MCP Specification:** https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/tools/
- **Pydantic Docs:** https://docs.pydantic.dev/latest/
- **structlog Docs:** https://www.structlog.org/

---

## Document Status

**Version:** 1.0
**Created:** 2025-11-23
**Author:** Goncharenko Anton aka alienxs2
**Copyright:** Copyright (c) 2025 Goncharenko Anton aka alienxs2
**License:** MIT License
**Status:** Draft - Ready for Review

**Next Steps:**
1. Review against component specification template
2. Verify alignment with parent module spec
3. Proceed to Level 3 (Function specs for each method)

---

**Estimated Implementation Effort:** 4-6 hours
**Lines of Code (Estimated):** ~200 lines (excluding tests)
**Test Coverage Target:** 95%+
