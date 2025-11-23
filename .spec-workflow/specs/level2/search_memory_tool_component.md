# SearchMemoryTool - Component Specification

**Level:** 2 (Component)
**Module:** zapomni_mcp
**Author:** Goncharenko Anton aka alienxs2
**Status:** Draft
**Version:** 1.0
**Created:** 2025-11-23

---

## Overview

### Purpose

The `SearchMemoryTool` implements the MCP tool interface for searching stored memories in the Zapomni knowledge graph. It provides natural language semantic search capabilities to Claude Desktop and other MCP clients, enabling intelligent retrieval of relevant information from previously stored memories.

This component serves as a **thin adapter** between the MCP protocol and the core search engine, handling request validation, response formatting, and error marshalling while delegating all search logic to `zapomni_core.MemoryEngine`.

### Responsibilities

1. **Input Validation**: Validate search query, limit, and filter parameters against strict constraints
2. **Request Delegation**: Delegate validated search requests to `zapomni_core.MemoryEngine.search_memory()`
3. **Response Formatting**: Transform core search results into MCP-compliant response format
4. **Error Handling**: Catch core exceptions and format as user-friendly MCP error messages
5. **MCP Tool Registration**: Provide tool metadata (name, description, input schema) for MCP server

**Design Philosophy**: Pure delegation pattern with no business logic - all search intelligence lives in the core layer.

### Position in Module

```
zapomni_mcp/
├── server.py                    # Main MCP server
├── tools/
│   ├── __init__.py              # Tool registry
│   ├── add_memory.py            # AddMemoryTool component
│   ├── search_memory.py         # THIS COMPONENT ←
│   └── get_stats.py             # GetStatsTool component
└── schemas/
    ├── requests.py              # Request models (used here)
    └── responses.py             # Response models (used here)
```

**Relationship to Other Components**:
- **MCP Server** registers this tool and routes `search_memory` calls to it
- **SearchMemoryRequest** (schema) validates incoming arguments
- **SearchMemoryResponse** (schema) formats outgoing results
- **MemoryEngine** (core) executes actual search operations

---

## Class Definition

### Class Diagram

```
┌─────────────────────────────────────────┐
│         SearchMemoryTool                │
├─────────────────────────────────────────┤
│ + name: str = "search_memory"           │
│ + description: str                      │
│ + input_schema: dict[str, Any]          │
│ - core: MemoryEngine                    │
│ - logger: BoundLogger                   │
├─────────────────────────────────────────┤
│ + __init__(core: MemoryEngine)          │
│ + execute(arguments: dict) -> dict      │
│ - _validate_input(arguments) -> Request │
│ - _format_response(results) -> dict     │
│ - _format_error(error: Exception) -> dict│
└─────────────────────────────────────────┘
         │
         │ uses
         ↓
┌─────────────────────────────────────────┐
│       MemoryEngine (Protocol)           │
│  ┌───────────────────────────────────┐  │
│  │ search_memory(request) -> Response│  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

### Full Class Signature

```python
# File: zapomni_mcp/tools/search_memory.py

from typing import Dict, Any, List
from structlog import get_logger
from pydantic import ValidationError

from zapomni_core.interfaces import MemoryEngine, SearchRequest, SearchResponse
from zapomni_core.exceptions import SearchError, CoreError


class SearchMemoryTool:
    """
    MCP tool for searching memories in the knowledge graph.

    This tool provides natural language semantic search over stored memories,
    returning ranked results based on vector similarity (and optionally BM25
    keyword matching in Phase 2).

    The tool is a thin adapter: it validates inputs, delegates to the core
    search engine, and formats responses. All search intelligence (embeddings,
    ranking, filtering) is handled by zapomni_core.

    Attributes:
        name: MCP tool name ("search_memory")
        description: Human-readable tool description for Claude
        input_schema: JSON Schema defining valid tool arguments
        core: Core memory engine for executing searches
        logger: Structured logger for debugging

    Example:
        >>> from zapomni_core import MemoryProcessor
        >>> from zapomni_db import FalkorDBClient
        >>>
        >>> core = MemoryProcessor(db=FalkorDBClient())
        >>> tool = SearchMemoryTool(core=core)
        >>>
        >>> # Simulate MCP tool call
        >>> result = await tool.execute({
        ...     "query": "What is Python?",
        ...     "limit": 5
        ... })
        >>>
        >>> print(result["content"][0]["text"])
        # "Found 5 results:\n1. Python is a programming language..."
    """

    # MCP Tool Metadata (class attributes)
    name: str = "search_memory"

    description: str = (
        "Search your personal memory graph for information. "
        "Performs semantic search to find relevant memories based on meaning, "
        "not just keyword matching. Returns ranked results with similarity scores. "
        "Use this when you need to recall previously stored information."
    )

    input_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Natural language search query (e.g., 'information about Python', "
                    "'my notes on machine learning', 'what did I learn about Docker?')"
                ),
                "minLength": 1,
                "maxLength": 1000
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return (default: 10, max: 100)",
                "default": 10,
                "minimum": 1,
                "maximum": 100
            },
            "filters": {
                "type": "object",
                "description": (
                    "Optional metadata filters to narrow results. "
                    "Supported keys: 'tags' (list), 'source' (string), "
                    "'date_from' (ISO date), 'date_to' (ISO date)"
                ),
                "properties": {
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Only return memories with these tags"
                    },
                    "source": {
                        "type": "string",
                        "description": "Only return memories from this source"
                    },
                    "date_from": {
                        "type": "string",
                        "format": "date",
                        "description": "Only memories created after this date (YYYY-MM-DD)"
                    },
                    "date_to": {
                        "type": "string",
                        "format": "date",
                        "description": "Only memories created before this date (YYYY-MM-DD)"
                    }
                },
                "additionalProperties": False
            }
        },
        "required": ["query"]
    }

    def __init__(self, core: MemoryEngine) -> None:
        """
        Initialize SearchMemoryTool with core memory engine.

        Args:
            core: Core memory engine implementing MemoryEngine protocol.
                  Must provide search_memory() method.

        Raises:
            TypeError: If core does not implement MemoryEngine protocol
                       (checked at runtime via duck typing)
        """
        self.core = core
        self.logger = get_logger(__name__).bind(tool="search_memory")

        # Validate core implements required protocol methods
        if not hasattr(core, 'search_memory') or not callable(core.search_memory):
            raise TypeError(
                "core must implement MemoryEngine protocol with search_memory() method"
            )

        self.logger.info("search_memory_tool_initialized")

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute search_memory tool with provided arguments.

        This is the main entry point called by the MCP server when a client
        invokes the search_memory tool. It orchestrates the complete workflow:
        validate → delegate → format → return.

        Workflow:
        1. Validate arguments using Pydantic (SearchRequest model)
        2. Delegate search to core.search_memory()
        3. Format results as MCP response
        4. Return success response with results

        Args:
            arguments: Dictionary from MCP client with keys:
                - query (str, required): Search query text
                - limit (int, optional): Max results (default: 10)
                - filters (dict, optional): Metadata filters

        Returns:
            MCP-formatted response dictionary:
            {
                "content": [
                    {
                        "type": "text",
                        "text": "Found N results:\n1. [text excerpt]..."
                    }
                ],
                "isError": false
            }

        Raises:
            No exceptions raised - all errors caught and returned as MCP error responses

        Performance Target:
            - Validation: < 5ms
            - Core delegation: < 200ms (P50), < 500ms (P95)
            - Formatting: < 10ms
            - Total: < 220ms (P50), < 520ms (P95)

        Error Handling:
            - ValidationError → MCP error with field details
            - SearchError → MCP error with user-friendly message
            - CoreError → MCP error with generic message
            - Unexpected exceptions → MCP error, logged with traceback
        """
        self.logger.debug("search_memory_execute_start", arguments=arguments)

        try:
            # Step 1: Validate input
            request = self._validate_input(arguments)
            self.logger.info(
                "search_request_validated",
                query_length=len(request.query),
                limit=request.limit,
                has_filters=request.filters is not None
            )

            # Step 2: Delegate to core engine
            response = await self.core.search_memory(request)
            self.logger.info(
                "search_completed",
                result_count=response.count,
                query=request.query[:50]  # Log first 50 chars only
            )

            # Step 3: Format response
            formatted = self._format_response(response)

            return formatted

        except ValidationError as e:
            self.logger.warning("validation_error", errors=e.errors())
            return self._format_error(e)

        except SearchError as e:
            self.logger.error("search_error", error=str(e))
            return self._format_error(e)

        except CoreError as e:
            self.logger.error("core_error", error=str(e))
            return self._format_error(e)

        except Exception as e:
            self.logger.exception("unexpected_error", error=str(e))
            return self._format_error(e)

    def _validate_input(self, arguments: Dict[str, Any]) -> SearchRequest:
        """
        Validate and parse tool arguments into SearchRequest model.

        Uses Pydantic validation to ensure all constraints are met:
        - query: non-empty, max 1000 chars
        - limit: 1-100 range, default 10
        - filters: valid structure if provided

        Args:
            arguments: Raw arguments dictionary from MCP client

        Returns:
            SearchRequest: Validated, immutable request model

        Raises:
            ValidationError: If arguments invalid (caught by execute())

        Example Valid Input:
            {
                "query": "Python programming",
                "limit": 10,
                "filters": {"tags": ["tech"], "date_from": "2025-01-01"}
            }

        Example Invalid Input:
            {"query": ""} → ValidationError: query cannot be empty
            {"query": "x", "limit": 0} → ValidationError: limit must be >= 1
            {"query": "x", "limit": 200} → ValidationError: limit must be <= 100
        """
        # Pydantic will validate constraints and raise ValidationError if invalid
        request = SearchRequest(**arguments)
        return request

    def _format_response(self, response: SearchResponse) -> Dict[str, Any]:
        """
        Format SearchResponse from core into MCP response format.

        Converts core search results into a human-readable text response
        that Claude can present to the user. Includes result count,
        similarity scores, and text excerpts.

        Args:
            response: SearchResponse from core.search_memory()

        Returns:
            MCP-formatted response dictionary with content blocks

        Response Format (Success):
            {
                "content": [
                    {
                        "type": "text",
                        "text": "Found 3 results:\n\n1. [Score: 0.89] Python is...\n\n2. ..."
                    }
                ],
                "isError": false
            }

        Response Format (No Results):
            {
                "content": [
                    {
                        "type": "text",
                        "text": "No results found for query: 'obscure topic'"
                    }
                ],
                "isError": false
            }

        Example:
            >>> response = SearchResponse(
            ...     count=2,
            ...     results=[
            ...         SearchResult(
            ...             memory_id="550e...",
            ...             text="Python is a language",
            ...             similarity_score=0.92,
            ...             metadata={"tags": ["tech"]}
            ...         )
            ...     ]
            ... )
            >>> formatted = tool._format_response(response)
            >>> print(formatted["content"][0]["text"])
            "Found 2 results:\n\n1. [Score: 0.92] Python is a language\n..."
        """
        if response.count == 0:
            # No results found
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "No results found matching your query."
                    }
                ],
                "isError": False
            }

        # Build formatted text response
        lines = [f"Found {response.count} results:\n"]

        for i, result in enumerate(response.results, start=1):
            # Format: "N. [Score: 0.XX] text excerpt"
            score = f"{result.similarity_score:.2f}"
            text_preview = result.text[:200]  # First 200 chars
            if len(result.text) > 200:
                text_preview += "..."

            # Include tags if present
            tags_str = ""
            if result.metadata.get("tags"):
                tags_str = f" [Tags: {', '.join(result.metadata['tags'])}]"

            lines.append(
                f"\n{i}. [Score: {score}]{tags_str}\n{text_preview}\n"
            )

        formatted_text = "".join(lines)

        return {
            "content": [
                {
                    "type": "text",
                    "text": formatted_text
                }
            ],
            "isError": False
        }

    def _format_error(self, error: Exception) -> Dict[str, Any]:
        """
        Format exception into MCP error response.

        Converts various exception types into user-friendly error messages
        while preserving error details for debugging (via logs).

        Args:
            error: Exception raised during execution

        Returns:
            MCP-formatted error response

        Error Response Format:
            {
                "content": [
                    {
                        "type": "text",
                        "text": "Error: <user-friendly message>"
                    }
                ],
                "isError": true
            }

        Error Message Mapping:
            - ValidationError → "Invalid input: <field errors>"
            - SearchError → "Search failed: <error message>"
            - CoreError → "Processing error: <error message>"
            - Other → "An unexpected error occurred. Check logs."

        Security Note:
            Never leak internal paths, credentials, or stack traces in error
            messages. Detailed errors only logged to stderr.

        Example:
            >>> error = ValidationError([{"loc": ("query",), "msg": "field required"}])
            >>> formatted = tool._format_error(error)
            >>> print(formatted["content"][0]["text"])
            "Error: Invalid input - query: field required"
        """
        if isinstance(error, ValidationError):
            # Extract field-level errors
            error_msgs = []
            for err in error.errors():
                field = ".".join(str(loc) for loc in err["loc"])
                message = err["msg"]
                error_msgs.append(f"{field}: {message}")

            error_text = f"Invalid input - {'; '.join(error_msgs)}"

        elif isinstance(error, SearchError):
            error_text = f"Search failed: {str(error)}"

        elif isinstance(error, CoreError):
            error_text = f"Processing error: {str(error)}"

        else:
            # Unexpected error - generic message, details in logs
            error_text = "An unexpected error occurred. Please check logs for details."

        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Error: {error_text}"
                }
            ],
            "isError": True
        }
```

---

## Dependencies

### Component Dependencies

**Internal (zapomni_core):**
- `MemoryEngine` (Protocol) - Interface for core memory operations
  - Used for: Delegating search requests
  - Method: `search_memory(request: SearchRequest) -> SearchResponse`

- `SearchRequest` (Pydantic model) - Request validation
  - Fields: query, limit, filters
  - Validates: query length, limit range, filter structure

- `SearchResponse` (Pydantic model) - Response structure
  - Fields: count, results (List[SearchResult])

- `SearchResult` (Pydantic model) - Individual result
  - Fields: memory_id, text, similarity_score, metadata

**Internal (zapomni_core.exceptions):**
- `CoreError` - Base exception for core layer
- `SearchError` - Search-specific exceptions
- `ValidationError` - From Pydantic (re-exported)

**External Libraries:**
- `structlog` - Structured logging
  - Used for: Logging search operations, errors
  - Why: JSON logs for observability

- `pydantic` - Data validation
  - Used for: Request/response validation
  - Why: Type-safe, automatic validation

### Dependency Injection

**Pattern**: Constructor injection via `__init__(core: MemoryEngine)`

**Why Constructor Injection?**
- Explicit dependencies (no hidden globals)
- Easy to mock for testing
- Type-checked by mypy/pyright
- Follows "dependency inversion principle"

**Example Usage**:
```python
# In server.py
from zapomni_core import MemoryProcessor
from zapomni_db import FalkorDBClient

# Create dependencies
db = FalkorDBClient(host="localhost")
core = MemoryProcessor(db=db)

# Inject into tool
search_tool = SearchMemoryTool(core=core)

# Register with MCP server
server.register_tool(search_tool)
```

---

## State Management

### Attributes

**Instance Attributes:**

1. `self.core: MemoryEngine`
   - **Type**: Protocol (structural interface)
   - **Purpose**: Core memory engine for executing searches
   - **Lifetime**: Entire tool lifetime (set in __init__, never changes)
   - **Thread-Safety**: Read-only after initialization (safe)

2. `self.logger: BoundLogger`
   - **Type**: structlog.BoundLogger
   - **Purpose**: Structured logging with context binding
   - **Lifetime**: Entire tool lifetime
   - **Thread-Safety**: structlog is thread-safe

**Class Attributes:**

1. `name: str = "search_memory"`
   - **Type**: str (immutable)
   - **Purpose**: MCP tool identifier
   - **Shared**: All instances (but never modified)

2. `description: str`
   - **Type**: str (immutable)
   - **Purpose**: Tool description for Claude

3. `input_schema: Dict[str, Any]`
   - **Type**: dict (frozen at class level)
   - **Purpose**: JSON Schema for validation

### State Transitions

**Stateless Component**: SearchMemoryTool maintains NO mutable state.

```
Initialization:
    __init__(core) → Tool Ready
        ↓
    [Tool remains in "Ready" state forever]
        ↓
    execute(args) → Transient execution (no state change)
        ↓
    [Returns to "Ready" state]
```

**No State Mutations**:
- Each `execute()` call is independent
- No request history stored
- No caching (handled by core layer)
- No side effects (besides logging)

### Thread Safety

**Status**: ✅ **Thread-safe**

**Rationale**:
- No mutable state (only read-only attributes)
- Logging is thread-safe (structlog)
- Each `execute()` call uses local variables only
- Core engine responsible for its own thread-safety

**Concurrency Limitations**:
- MCP stdio transport is inherently sequential
- No concurrent requests possible in current architecture
- Thread-safety provided for future HTTP transport

---

## Public Methods (Detailed)

### Method 1: `__init__`

**Signature:**
```python
def __init__(self, core: MemoryEngine) -> None
```

**Purpose**: Initialize SearchMemoryTool with core memory engine dependency.

**Parameters:**
- `core: MemoryEngine`
  - **Description**: Core memory processing engine implementing MemoryEngine protocol
  - **Constraints**: Must have callable `search_memory()` method
  - **Example**: `MemoryProcessor(db=FalkorDBClient())`
  - **Validation**: Runtime check for `search_memory` method existence

**Returns:**
- Type: `None`
- Side Effects:
  - Sets `self.core` attribute
  - Sets `self.logger` attribute (bound logger with tool context)
  - Logs "search_memory_tool_initialized" event

**Raises:**
- `TypeError`: When core does not implement MemoryEngine protocol
  - Message: "core must implement MemoryEngine protocol with search_memory() method"
  - When: If `hasattr(core, 'search_memory')` is False or not callable

**Preconditions:**
- `core` is a valid object (not None)
- `core` implements MemoryEngine protocol methods

**Postconditions:**
- Tool is in "Ready" state
- `self.core` references provided core engine
- `self.logger` is configured for structured logging

**Example Usage:**
```python
from zapomni_core import MemoryProcessor
from zapomni_db import FalkorDBClient

# Valid initialization
core = MemoryProcessor(db=FalkorDBClient())
tool = SearchMemoryTool(core=core)

# Invalid initialization (TypeError)
tool = SearchMemoryTool(core=None)  # Raises TypeError
```

---

### Method 2: `execute`

**Signature:**
```python
async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]
```

**Purpose**: Execute search operation with MCP client arguments.

**Parameters:**
- `arguments: Dict[str, Any]`
  - **Description**: MCP tool arguments from client
  - **Structure**:
    ```python
    {
        "query": str,              # Required
        "limit": int,              # Optional (default: 10)
        "filters": {               # Optional
            "tags": List[str],     # Optional
            "source": str,         # Optional
            "date_from": str,      # Optional (YYYY-MM-DD)
            "date_to": str         # Optional (YYYY-MM-DD)
        }
    }
    ```
  - **Constraints**:
    - `query`: 1-1000 characters, non-empty
    - `limit`: 1-100 range
    - `filters`: Valid filter keys only
  - **Example**:
    ```python
    {
        "query": "Python programming",
        "limit": 5,
        "filters": {"tags": ["tech", "tutorial"]}
    }
    ```

**Returns:**
- **Type**: `Dict[str, Any]`
- **Success Structure**:
  ```python
  {
      "content": [
          {
              "type": "text",
              "text": "Found N results:\n\n1. [Score: X.XX] text..."
          }
      ],
      "isError": False
  }
  ```
- **Error Structure**:
  ```python
  {
      "content": [
          {
              "type": "text",
              "text": "Error: <message>"
          }
      ],
      "isError": True
  }
  ```

**Raises:**
- **No exceptions raised** - all errors caught and returned as MCP error responses

**Error Handling (Internal):**
1. `ValidationError` → Formatted as input validation error
2. `SearchError` → Formatted as search failure error
3. `CoreError` → Formatted as processing error
4. `Exception` → Formatted as unexpected error (logged with traceback)

**Preconditions:**
- Tool initialized with valid core engine
- `arguments` is a dictionary (MCP guarantees this)

**Postconditions:**
- Returns valid MCP response (success or error)
- Logs operation details to stderr
- No state changes in tool instance

**Algorithm Outline:**
```
1. Log execution start with arguments
2. TRY:
   a. Validate arguments → SearchRequest (may raise ValidationError)
   b. Log validation success
   c. Delegate to core.search_memory(request) → SearchResponse
   d. Log search completion with result count
   e. Format response as MCP success message
   f. Return formatted response
3. CATCH ValidationError:
   a. Log warning with error details
   b. Format as MCP error response
   c. Return error response
4. CATCH SearchError:
   a. Log error with message
   b. Format as MCP error response
   c. Return error response
5. CATCH CoreError:
   a. Log error with message
   b. Format as MCP error response
   c. Return error response
6. CATCH Exception:
   a. Log exception with full traceback
   b. Format as generic MCP error
   c. Return error response
```

**Edge Cases:**

1. **Empty Query**:
   - Input: `{"query": ""}`
   - Behavior: ValidationError raised
   - Response: `"Error: Invalid input - query: ensure this value has at least 1 character"`

2. **Query Too Long**:
   - Input: `{"query": "x" * 1001}`
   - Behavior: ValidationError raised
   - Response: `"Error: Invalid input - query: ensure this value has at most 1000 characters"`

3. **Invalid Limit**:
   - Input: `{"query": "test", "limit": 0}`
   - Behavior: ValidationError raised
   - Response: `"Error: Invalid input - limit: ensure this value is greater than or equal to 1"`

4. **Limit Exceeds Maximum**:
   - Input: `{"query": "test", "limit": 200}`
   - Behavior: ValidationError raised
   - Response: `"Error: Invalid input - limit: ensure this value is less than or equal to 100"`

5. **No Results Found**:
   - Input: `{"query": "nonexistent topic"}`
   - Behavior: Success (count=0)
   - Response: `"No results found matching your query."`

6. **Invalid Filter Keys**:
   - Input: `{"query": "test", "filters": {"invalid_key": "value"}}`
   - Behavior: ValidationError raised
   - Response: `"Error: Invalid input - filters: extra fields not permitted"`

7. **Malformed Date Filter**:
   - Input: `{"query": "test", "filters": {"date_from": "not-a-date"}}`
   - Behavior: ValidationError raised
   - Response: `"Error: Invalid input - filters.date_from: invalid date format"`

8. **Core Engine Unavailable**:
   - Input: Valid query
   - Behavior: CoreError raised by core layer
   - Response: `"Error: Processing error: <core error message>"`

**Related Methods:**
- **Calls**: `_validate_input()`, `_format_response()`, `_format_error()`
- **Called by**: MCP server when client invokes `search_memory` tool

---

### Method 3: `_validate_input` (Private)

**Signature:**
```python
def _validate_input(self, arguments: Dict[str, Any]) -> SearchRequest
```

**Purpose**: Validate and parse raw MCP arguments into typed SearchRequest model.

**Parameters:**
- `arguments: Dict[str, Any]`
  - Raw dictionary from MCP client
  - May contain invalid or missing fields

**Returns:**
- **Type**: `SearchRequest`
- **Properties**: Immutable Pydantic model with validated fields

**Raises:**
- `ValidationError`: When arguments fail validation
  - Pydantic provides detailed field-level errors
  - Example: `[{"loc": ("query",), "msg": "field required", "type": "value_error.missing"}]`

**Validation Rules:**
1. **query** (required):
   - Must be present
   - Must be string
   - Length: 1-1000 characters
   - Cannot be whitespace-only

2. **limit** (optional):
   - Must be integer if provided
   - Range: 1-100
   - Default: 10

3. **filters** (optional):
   - Must be dict if provided
   - Allowed keys: tags, source, date_from, date_to
   - No additional keys permitted
   - Each filter value must match expected type

**Example Usage:**
```python
# Valid
request = tool._validate_input({"query": "Python"})
assert request.query == "Python"
assert request.limit == 10  # Default

# Valid with filters
request = tool._validate_input({
    "query": "Python",
    "limit": 20,
    "filters": {"tags": ["tech"], "source": "docs"}
})

# Invalid - raises ValidationError
request = tool._validate_input({"limit": 5})  # Missing query
request = tool._validate_input({"query": ""})  # Empty query
```

---

### Method 4: `_format_response` (Private)

**Signature:**
```python
def _format_response(self, response: SearchResponse) -> Dict[str, Any]
```

**Purpose**: Convert core SearchResponse into MCP-formatted text response.

**Parameters:**
- `response: SearchResponse`
  - From `core.search_memory()`
  - Contains: count, results (List[SearchResult])

**Returns:**
- **Type**: `Dict[str, Any]`
- **Structure**: MCP content block with formatted text

**Response Formatting:**

**Case 1: Results Found**
```
Found N results:

1. [Score: 0.92] [Tags: tech, python]
Python is a high-level programming language...

2. [Score: 0.85] [Tags: tutorial]
Learn Python basics with this comprehensive...

3. ...
```

**Case 2: No Results**
```
No results found matching your query.
```

**Formatting Rules:**
- Each result numbered (1, 2, 3, ...)
- Similarity score rounded to 2 decimals
- Tags displayed if present in metadata
- Text preview: first 200 characters (+ "..." if truncated)
- Blank line between results

**Example:**
```python
response = SearchResponse(
    count=2,
    results=[
        SearchResult(
            memory_id="uuid-1",
            text="Python is great for data science" * 10,  # Long text
            similarity_score=0.9234,
            metadata={"tags": ["python", "data"]}
        ),
        SearchResult(
            memory_id="uuid-2",
            text="Short text",
            similarity_score=0.8567,
            metadata={}
        )
    ]
)

formatted = tool._format_response(response)
print(formatted["content"][0]["text"])
# Output:
# Found 2 results:
#
# 1. [Score: 0.92] [Tags: python, data]
# Python is great for data sciencePython is great for data science...
#
# 2. [Score: 0.86]
# Short text
```

---

### Method 5: `_format_error` (Private)

**Signature:**
```python
def _format_error(self, error: Exception) -> Dict[str, Any]
```

**Purpose**: Convert exception into user-friendly MCP error response.

**Parameters:**
- `error: Exception`
  - Any exception caught during execution
  - Can be ValidationError, SearchError, CoreError, or unexpected

**Returns:**
- **Type**: `Dict[str, Any]`
- **Structure**: MCP error response with `isError: True`

**Error Message Mapping:**

| Exception Type | Error Prefix | Example Message |
|----------------|--------------|-----------------|
| `ValidationError` | "Invalid input" | "Invalid input - query: field required" |
| `SearchError` | "Search failed" | "Search failed: Vector index not found" |
| `CoreError` | "Processing error" | "Processing error: Embedding generation failed" |
| `Exception` | "An unexpected error occurred" | "An unexpected error occurred. Please check logs for details." |

**Security Considerations:**
- Never leak internal file paths
- Never leak database credentials
- Never leak stack traces to user
- Detailed errors logged to stderr only

**Example:**
```python
# ValidationError
error = ValidationError([{"loc": ("query",), "msg": "field required"}])
formatted = tool._format_error(error)
assert formatted["isError"] is True
assert "Invalid input" in formatted["content"][0]["text"]

# SearchError
error = SearchError("No vector index found")
formatted = tool._format_error(error)
assert "Search failed" in formatted["content"][0]["text"]

# Unexpected error
error = RuntimeError("Internal crash")
formatted = tool._format_error(error)
assert "unexpected error" in formatted["content"][0]["text"]
# Note: Full traceback only in logs, not in response
```

---

## Error Handling

### Exception Hierarchy

**Zapomni Core Exceptions**:
```python
CoreError (base)
    ├── ValidationError (input validation)
    ├── SearchError (search operations)
    ├── EmbeddingError (embedding generation)
    └── DatabaseError (storage operations)
```

**Tool's Error Handling Strategy**:
- **Catch all exceptions** at `execute()` level
- **Log detailed errors** to stderr (structlog)
- **Return MCP error responses** (never raise to MCP server)
- **Sanitize error messages** (no sensitive data)

### Error Recovery

**No Automatic Retries**:
- SearchMemoryTool does NOT retry failed operations
- Retries are client's responsibility
- Rationale: MCP tools should be deterministic and fast

**Fallback Behavior**:
- No fallback - return error immediately
- Core layer may have fallbacks (e.g., sentence-transformers if Ollama fails)
- Tool layer is transparent - just reports core errors

**Error Propagation**:
- Core exceptions caught and formatted
- MCP server receives success response with `isError: True`
- MCP protocol does not support exceptions (JSON-RPC errors only)

### Logging Strategy

**Log Levels**:
- **DEBUG**: Execution start, arguments
- **INFO**: Validation success, search completion, result count
- **WARNING**: Validation errors (user fault)
- **ERROR**: Search errors, core errors (system fault)
- **EXCEPTION**: Unexpected errors with full traceback

**Log Context**:
- All logs include `tool="search_memory"` binding
- Request-specific logs include query (first 50 chars), limit, filter flags
- Result logs include count, not full results (avoid large logs)

**Example Logs**:
```json
{"timestamp": "2025-11-23T10:30:00Z", "level": "info", "event": "search_memory_tool_initialized", "tool": "search_memory"}
{"timestamp": "2025-11-23T10:30:01Z", "level": "debug", "event": "search_memory_execute_start", "tool": "search_memory", "arguments": {"query": "Python", "limit": 10}}
{"timestamp": "2025-11-23T10:30:01Z", "level": "info", "event": "search_request_validated", "tool": "search_memory", "query_length": 6, "limit": 10, "has_filters": false}
{"timestamp": "2025-11-23T10:30:01Z", "level": "info", "event": "search_completed", "tool": "search_memory", "result_count": 5, "query": "Python"}
{"timestamp": "2025-11-23T10:30:02Z", "level": "error", "event": "search_error", "tool": "search_memory", "error": "Vector index not built"}
```

---

## Usage Examples

### Basic Usage

```python
from zapomni_mcp.tools.search_memory import SearchMemoryTool
from zapomni_core import MemoryProcessor
from zapomni_db import FalkorDBClient

# Initialize dependencies
db = FalkorDBClient(host="localhost", port=6379)
core = MemoryProcessor(db=db, ollama_host="http://localhost:11434")

# Create tool
search_tool = SearchMemoryTool(core=core)

# Simulate MCP tool call
result = await search_tool.execute({
    "query": "What is Python?"
})

print(result["content"][0]["text"])
# Output:
# Found 3 results:
#
# 1. [Score: 0.92] Python is a high-level programming language...
# 2. [Score: 0.88] Python was created by Guido van Rossum...
# 3. [Score: 0.85] Python is widely used for web development...
```

### Advanced Usage with Filters

```python
# Search with metadata filters
result = await search_tool.execute({
    "query": "machine learning tutorials",
    "limit": 20,
    "filters": {
        "tags": ["ML", "tutorial"],
        "source": "documentation",
        "date_from": "2025-01-01",
        "date_to": "2025-12-31"
    }
})

if result["isError"]:
    print(f"Search failed: {result['content'][0]['text']}")
else:
    print(f"Found {len(result['results'])} results")
```

### Error Handling Example

```python
# Invalid input
result = await search_tool.execute({
    "query": "",  # Empty query
    "limit": 200  # Exceeds max
})

assert result["isError"] is True
print(result["content"][0]["text"])
# Output:
# Error: Invalid input - query: ensure this value has at least 1 character; limit: ensure this value is less than or equal to 100
```

### Integration with MCP Server

```python
from mcp.server import Server
from mcp.server.stdio import stdio_server

# Initialize MCP server
server = Server("zapomni")

# Create and register tool
search_tool = SearchMemoryTool(core=core)
server.register_tool(search_tool)

# Start server (blocks until EOF)
async with stdio_server() as (read_stream, write_stream):
    await server.run(read_stream, write_stream)
```

---

## Testing Approach

### Unit Tests Required

**Test Categories**:
1. Initialization tests
2. Input validation tests
3. Response formatting tests
4. Error formatting tests
5. Integration tests (with mock core)

**Specific Test Cases**:

1. **test_init_success**
   - Valid core → Tool initialized
   - Logger configured
   - Core reference stored

2. **test_init_invalid_core**
   - Core without search_memory() → TypeError
   - None core → TypeError

3. **test_execute_success_default_limit**
   - Valid query, no limit → Uses default 10
   - Core returns results → Formatted correctly

4. **test_execute_success_custom_limit**
   - Valid query, limit=20 → Passes to core
   - Verify core.search_memory called with limit=20

5. **test_execute_success_with_filters**
   - Query + filters → Passed to core
   - Verify filters in SearchRequest

6. **test_execute_empty_query_raises**
   - Empty query → ValidationError
   - Response has isError=True
   - Message mentions "query"

7. **test_execute_query_too_long_raises**
   - 1001 char query → ValidationError
   - Response has isError=True

8. **test_execute_limit_zero_raises**
   - limit=0 → ValidationError

9. **test_execute_limit_negative_raises**
   - limit=-5 → ValidationError

10. **test_execute_limit_exceeds_max_raises**
    - limit=200 → ValidationError

11. **test_execute_invalid_filter_keys_raises**
    - filters={"unknown": "value"} → ValidationError

12. **test_execute_core_search_error**
    - Core raises SearchError → MCP error response
    - isError=True
    - Message contains "Search failed"

13. **test_execute_core_generic_error**
    - Core raises CoreError → MCP error response

14. **test_execute_unexpected_exception**
    - Core raises RuntimeError → MCP error response
    - Message: "unexpected error"
    - Exception logged

15. **test_format_response_with_results**
    - Response with 3 results → Formatted text
    - Verify numbered list
    - Verify scores formatted

16. **test_format_response_no_results**
    - Response with count=0 → "No results found"

17. **test_format_response_text_truncation**
    - Result with 500 char text → Truncated to 200 + "..."

18. **test_format_response_includes_tags**
    - Result with tags → Tags in output

19. **test_format_error_validation_error**
    - ValidationError → "Invalid input" message

20. **test_format_error_search_error**
    - SearchError → "Search failed" message

21. **test_format_error_unexpected_error**
    - Generic Exception → "unexpected error" message

### Mocking Strategy

**Mock `zapomni_core.MemoryEngine`**:
```python
from unittest.mock import AsyncMock, MagicMock

@pytest.fixture
def mock_core():
    """Mock MemoryEngine for testing."""
    core = MagicMock()
    core.search_memory = AsyncMock()
    return core

@pytest.mark.asyncio
async def test_execute_success(mock_core):
    """Test successful search execution."""
    # Setup mock
    mock_core.search_memory.return_value = SearchResponse(
        count=2,
        results=[
            SearchResult(
                memory_id="uuid-1",
                text="Python is great",
                similarity_score=0.92,
                metadata={"tags": ["python"]}
            )
        ]
    )

    # Create tool with mock
    tool = SearchMemoryTool(core=mock_core)

    # Execute
    result = await tool.execute({"query": "Python"})

    # Assertions
    assert result["isError"] is False
    assert "Found 2 results" in result["content"][0]["text"]
    mock_core.search_memory.assert_called_once()
```

**Mock Pydantic Validation**:
- Use real Pydantic models (don't mock validation)
- Test edge cases with invalid data
- Verify ValidationError structure

### Integration Tests

**Test with Real Core (Mock DB)**:
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_search_integration(mock_db):
    """Integration test: Tool → Core → Mock DB."""
    # Setup real core with mock DB
    core = MemoryProcessor(db=mock_db)

    # Setup mock DB responses
    mock_db.search_vectors.return_value = [
        {"id": "uuid-1", "text": "Python", "score": 0.92}
    ]

    # Create tool
    tool = SearchMemoryTool(core=core)

    # Execute
    result = await tool.execute({"query": "Python"})

    # Verify
    assert result["isError"] is False
    mock_db.search_vectors.assert_called_once()
```

**Coverage Target**: 95%+ for SearchMemoryTool code

---

## Performance Considerations

### Time Complexity

**SearchMemoryTool Operations**:
- `__init__()`: O(1) - simple attribute assignment
- `_validate_input()`: O(n) where n = number of filter keys (typically small, n ≤ 4)
- `_format_response()`: O(m) where m = number of results (max 100)
- `_format_error()`: O(k) where k = number of validation errors (typically k ≤ 5)
- `execute()`: O(n + m) (validation + formatting)

**Core Search Time Complexity** (not in this component):
- Vector search: O(log N) with HNSW index (N = total chunks)
- BM25 search: O(N) (linear scan)
- Hybrid search: O(log N + N) (combines both)

### Space Complexity

**SearchMemoryTool Memory Usage**:
- Instance overhead: ~500 bytes (core ref, logger, class attrs)
- Per-request overhead: ~1-5 KB (arguments, request/response models)
- Response formatting: O(m * k) where m = results, k = avg text length
  - Max: 100 results * 200 chars/result = 20 KB

**Total Memory**: < 30 KB per request (negligible)

### Optimization Opportunities

**Current Performance**:
- Tool layer adds ~20ms overhead (P50)
- Acceptable for Phase 1 (stdio is sequential anyway)

**Potential Optimizations (Phase 2)**:

1. **Response Streaming**:
   - Stream large result sets (>50 results)
   - Reduce memory footprint
   - Improve perceived latency

2. **Caching Input Schemas**:
   - Cache Pydantic JSON schema generation
   - Saves ~1ms per request

3. **Lazy Formatting**:
   - Format results on-demand (if MCP supports pagination)
   - Current: format all results immediately

**Trade-offs**:
- Streaming adds complexity (not justified for Phase 1)
- Caching adds state (violates stateless design)
- Lazy formatting requires protocol changes

**Decision**: Keep current simple implementation. Optimize only if profiling shows tool layer is bottleneck (unlikely).

---

## References

### Module Spec
- **zapomni_mcp_module.md** (Level 1) - Parent module specification
- Section: "Public API > MCPTool Protocol" - Interface requirements
- Section: "Data Models > Request/Response Models" - Schema definitions

### Related Specs
- **zapomni_core_module.md** (Level 1) - Core search engine specification
  - Section: "Public API > MemoryProcessor.search_memory()" - Core method signature
  - Section: "Data Models > SearchResult" - Result structure

- **cross_module_interfaces.md** (Level 1) - Interface contracts
  - Section: "MCP → Core: MemoryEngine Protocol" - Protocol definition
  - Section: "Error Handling Strategy" - Exception handling patterns

### External Docs
- **MCP Specification**: https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/tools/
  - Tool registration format
  - Input schema requirements
  - Response format

- **Pydantic Documentation**: https://docs.pydantic.dev/latest/
  - Validation patterns
  - JSON schema generation

- **structlog Documentation**: https://www.structlog.org/
  - Structured logging best practices
  - Context binding

---

## Document Status

**Version:** 1.0
**Created:** 2025-11-23
**Author:** Goncharenko Anton aka alienxs2
**Status:** Draft - Ready for Review

**Verification Checklist**:
- ✅ All public methods have full signatures
- ✅ All parameters typed and documented
- ✅ All returns typed and documented
- ✅ All exceptions documented
- ✅ Edge cases identified (7+)
- ✅ Test scenarios defined (20+)
- ✅ Examples provided
- ✅ Dependencies listed
- ✅ Performance targets specified
- ✅ Thread-safety analyzed

**Next Steps**:
1. Multi-agent verification
2. Reconciliation with steering documents
3. Approval
4. Proceed to Level 3 (Function specs for each method)

---

**Copyright:** Copyright (c) 2025 Goncharenko Anton aka alienxs2
**License:** MIT License
**Document Length:** ~1400 lines
**Estimated Reading Time:** 40-50 minutes
**Target Audience:** Developers implementing SearchMemoryTool component
