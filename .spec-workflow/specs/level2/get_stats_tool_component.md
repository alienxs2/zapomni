# GetStatsTool - Component Specification

**Level:** 2 (Component)
**Module:** zapomni_mcp
**Author:** Goncharenko Anton aka alienxs2
**Status:** Draft
**Version:** 1.0
**Created:** 2025-11-23

---

## Overview

### Purpose

The **GetStatsTool** component implements an MCP tool that retrieves and reports system statistics about the Zapomni memory system. It provides insights into memory storage, chunk counts, database size, and performance metrics without requiring any input parameters.

**Core Function**: Exposes system statistics to MCP clients (Claude Desktop) in a standardized format.

### Responsibilities

1. **Tool Definition**: Define MCP tool schema with no required parameters
2. **Stats Retrieval**: Delegate to `MemoryEngine.get_stats()` to fetch system statistics
3. **Response Formatting**: Transform raw statistics into MCP-compliant response format
4. **Error Handling**: Handle potential errors from core layer gracefully
5. **Logging**: Log stats requests for audit and monitoring

### Position in Module

```
zapomni_mcp/
├── server.py                    # Main MCP server
├── tools/
│   ├── __init__.py              # Tool registry
│   ├── add_memory.py            # AddMemoryTool
│   ├── search_memory.py         # SearchMemoryTool
│   └── get_stats.py             # ← THIS COMPONENT
└── schemas/
    └── responses.py             # Response models
```

**Integration Points**:
- **Registers with**: MCPServer tool registry
- **Calls**: `zapomni_core.MemoryEngine.get_stats()`
- **Uses**: Response formatting utilities from MCP layer

---

## Class Definition

### Class Diagram

```
┌─────────────────────────────────────┐
│         GetStatsTool                │
├─────────────────────────────────────┤
│ - name: str = "get_stats"           │
│ - description: str                  │
│ - input_schema: dict[str, Any]      │
│ - core: MemoryEngine                │
├─────────────────────────────────────┤
│ + __init__(core: MemoryEngine)      │
│ + execute(arguments: dict) -> dict  │
│ - _format_response(stats) -> dict   │
└─────────────────────────────────────┘
```

### Full Class Signature

```python
# File: zapomni_mcp/tools/get_stats.py

from typing import Any, Dict, Protocol
from zapomni_core.interfaces import MemoryEngine
from zapomni_core.exceptions import CoreError
import structlog

logger = structlog.get_logger(__name__)


class GetStatsTool:
    """
    MCP tool for retrieving memory system statistics.

    This tool provides insights into the current state of the Zapomni
    memory system including total memories, chunks, database size,
    and optional performance metrics.

    Attributes:
        name: Tool name exposed to MCP clients ("get_stats")
        description: Human-readable description of tool functionality
        input_schema: JSON schema for tool inputs (empty - no params required)
        core: MemoryEngine instance for retrieving statistics

    Example:
        ```python
        from zapomni_core.engine import ZapomniEngine

        # Initialize core engine
        engine = ZapomniEngine(storage=storage, chunker=chunker, embedder=embedder)

        # Create tool
        tool = GetStatsTool(core=engine)

        # Execute (no arguments needed)
        result = await tool.execute({})

        # Returns:
        # {
        #     "content": [{
        #         "type": "text",
        #         "text": "Memory System Statistics:\\n..."
        #     }],
        #     "isError": False
        # }
        ```

    Thread Safety:
        This tool is thread-safe as it only reads data and maintains no
        mutable state beyond the injected core engine reference.
    """

    # Class constants
    name: str = "get_stats"
    description: str = (
        "Get statistics about the memory system including total memories, "
        "chunks, database size, and performance metrics."
    )

    # JSON Schema for MCP tool registration (no parameters required)
    input_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {},  # No properties - tool takes no arguments
        "required": [],    # No required fields
        "additionalProperties": False
    }

    def __init__(self, core: MemoryEngine) -> None:
        """
        Initialize GetStatsTool with core memory engine.

        Args:
            core: MemoryEngine instance that implements get_stats() method.
                  Must satisfy the MemoryEngine protocol from zapomni_core.

        Raises:
            TypeError: If core does not implement MemoryEngine protocol
                      (caught at type-checking time with mypy).

        Example:
            ```python
            engine = ZapomniEngine(storage=db_client, ...)
            tool = GetStatsTool(core=engine)
            ```

        Notes:
            - The core engine is stored as a reference, not copied.
            - No validation is performed at runtime (rely on type hints).
            - The engine should already be initialized and connected.
        """
        self.core: MemoryEngine = core
        logger.info("get_stats_tool_initialized")

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
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

            On success, text contains formatted statistics:
            - Total memories count
            - Total chunks count
            - Database size in MB
            - Graph name
            - Cache hit rate (if available)
            - Average query latency (if available)

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

            # Error result:
            {
                "content": [{
                    "type": "text",
                    "text": "Error: Failed to retrieve statistics"
                }],
                "isError": True
            }
            ```

        Performance:
            - Expected latency: < 100ms
            - No side effects (read-only operation)
            - No caching (always fetch fresh stats)

        Notes:
            - Arguments are validated to be a dict but content is ignored
            - Statistics are formatted as human-readable text
            - Missing optional metrics are omitted from output
            - Errors are logged and returned as MCP error responses
        """
        logger.info("get_stats_requested", arguments=arguments)

        try:
            # 1. Validate arguments (should be empty dict, but accept any)
            if not isinstance(arguments, dict):
                logger.warning(
                    "get_stats_invalid_arguments_type",
                    type=type(arguments).__name__
                )
                return {
                    "content": [{
                        "type": "text",
                        "text": "Error: Arguments must be a dictionary (expected empty {})"
                    }],
                    "isError": True
                }

            # 2. Retrieve statistics from core engine
            stats = await self.core.get_stats()
            logger.debug("stats_retrieved", stats=stats)

            # 3. Format response
            response = self._format_response(stats)
            logger.info("get_stats_success", stats_keys=list(stats.keys()))

            return response

        except CoreError as e:
            # Core layer error (processing, database, etc.)
            logger.error(
                "get_stats_core_error",
                error=str(e),
                error_type=type(e).__name__
            )
            return {
                "content": [{
                    "type": "text",
                    "text": f"Error: Failed to retrieve statistics - {str(e)}"
                }],
                "isError": True
            }

        except Exception as e:
            # Unexpected error
            logger.error(
                "get_stats_unexpected_error",
                error=str(e),
                error_type=type(e).__name__
            )
            return {
                "content": [{
                    "type": "text",
                    "text": "Error: An unexpected error occurred while retrieving statistics"
                }],
                "isError": True
            }

    def _format_response(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format raw statistics into MCP response structure.

        Transforms the statistics dictionary from core engine into a
        human-readable formatted text response suitable for display
        to users in Claude Desktop.

        Args:
            stats: Statistics dictionary from MemoryEngine.get_stats()
                  Expected keys:
                  - total_memories: int (required)
                  - total_chunks: int (required)
                  - database_size_mb: float (required)
                  - graph_name: str (required)
                  - cache_hit_rate: float (optional, 0.0-1.0)
                  - avg_query_latency_ms: float (optional)

        Returns:
            MCP response dictionary with formatted text content:
            {
                "content": [{"type": "text", "text": "..."}],
                "isError": False
            }

        Example:
            ```python
            stats = {
                "total_memories": 1234,
                "total_chunks": 5678,
                "database_size_mb": 45.67,
                "graph_name": "zapomni_memory",
                "cache_hit_rate": 0.653,
                "avg_query_latency_ms": 23.4
            }

            response = self._format_response(stats)

            # Returns:
            {
                "content": [{
                    "type": "text",
                    "text": "Memory System Statistics:\\n"
                            "Total Memories: 1,234\\n"
                            "Total Chunks: 5,678\\n"
                            "Database Size: 45.67 MB\\n"
                            "Graph Name: zapomni_memory\\n"
                            "Cache Hit Rate: 65.3%\\n"
                            "Avg Query Latency: 23.4 ms"
                }],
                "isError": False
            }
            ```

        Notes:
            - Numbers are formatted with thousand separators (1,234)
            - Percentages are formatted with one decimal place (65.3%)
            - Optional metrics are omitted if None or not present
            - Database size shows 2 decimal places
            - Latency shows 1 decimal place
        """
        # Build formatted text output
        lines = ["Memory System Statistics:"]

        # Required fields (always present)
        lines.append(f"Total Memories: {stats.get('total_memories', 0):,}")
        lines.append(f"Total Chunks: {stats.get('total_chunks', 0):,}")
        lines.append(f"Database Size: {stats.get('database_size_mb', 0.0):.2f} MB")
        lines.append(f"Graph Name: {stats.get('graph_name', 'unknown')}")

        # Optional fields (only if present and not None)
        if 'total_entities' in stats and stats['total_entities'] is not None:
            lines.append(f"Total Entities: {stats['total_entities']:,}")

        if 'cache_hit_rate' in stats and stats['cache_hit_rate'] is not None:
            hit_rate_pct = stats['cache_hit_rate'] * 100
            lines.append(f"Cache Hit Rate: {hit_rate_pct:.1f}%")

        if 'avg_query_latency_ms' in stats and stats['avg_query_latency_ms'] is not None:
            lines.append(f"Avg Query Latency: {stats['avg_query_latency_ms']:.1f} ms")

        # Join into single text block
        formatted_text = "\n".join(lines)

        return {
            "content": [{
                "type": "text",
                "text": formatted_text
            }],
            "isError": False
        }
```

---

## Dependencies

### Component Dependencies

**Internal (Zapomni)**:
- `zapomni_core.interfaces.MemoryEngine` (Protocol)
  - **Purpose**: Interface to core memory engine
  - **Used for**: Calling `get_stats()` method
  - **Type**: Protocol (structural typing)

- `zapomni_core.exceptions.CoreError` (Exception class)
  - **Purpose**: Catch errors from core layer
  - **Used for**: Error handling and logging

### External Libraries

- `structlog>=23.2.0`
  - **Purpose**: Structured logging to stderr
  - **Used for**: Logging stats requests, successes, errors
  - **Why**: JSON-formatted logs for observability

- `typing` (stdlib)
  - **Purpose**: Type hints (Any, Dict, Protocol)
  - **Used for**: Type safety and IDE support

### Dependency Injection

```python
# Dependency injection happens at tool initialization

# In server.py:
from zapomni_core.engine import ZapomniEngine
from zapomni_mcp.tools.get_stats import GetStatsTool

# Create engine
engine = ZapomniEngine(storage=db_client, ...)

# Inject into tool
get_stats_tool = GetStatsTool(core=engine)

# Register with MCP server
server.register_tool(get_stats_tool)
```

**Benefits**:
- Easy to test (inject mock engine)
- Loose coupling (depends on protocol, not concrete class)
- Configuration external to component

---

## State Management

### Attributes

**Instance Attributes**:

1. `core: MemoryEngine`
   - **Type**: MemoryEngine protocol
   - **Purpose**: Reference to core memory engine for retrieving stats
   - **Lifetime**: Same as tool instance (entire MCP server session)
   - **Mutability**: Immutable reference (object itself may change state)

**Class Attributes**:

1. `name: str = "get_stats"`
   - **Type**: str (constant)
   - **Purpose**: Tool identifier for MCP registration
   - **Lifetime**: Static (class-level)

2. `description: str = "..."`
   - **Type**: str (constant)
   - **Purpose**: Human-readable tool description
   - **Lifetime**: Static (class-level)

3. `input_schema: dict[str, Any] = {...}`
   - **Type**: dict (constant)
   - **Purpose**: JSON Schema for MCP tool arguments
   - **Lifetime**: Static (class-level)

### State Transitions

This component is **stateless** (no state machine):

```
Initialized
    ↓
Ready (idle)
    ↓
[execute() called]
    ↓
Fetching stats from core
    ↓
Formatting response
    ↓
Ready (idle)
```

**No state is retained between calls** - each `execute()` is independent.

### Thread Safety

**Thread-Safe**: Yes

**Rationale**:
- No mutable state (only read-only reference to core)
- No shared data structures
- Each `execute()` call is independent
- Core engine handles its own thread safety

**Concurrency Model**:
- Async/await (cooperative multitasking)
- No locks needed
- Safe for concurrent calls (if MCP server supports it)

---

## Public Methods (Detailed)

### Method 1: `__init__`

**Signature**:
```python
def __init__(self, core: MemoryEngine) -> None
```

**Purpose**: Initialize GetStatsTool with dependency-injected core engine.

**Parameters**:
- `core`: MemoryEngine
  - **Description**: Core memory engine instance
  - **Constraints**: Must implement MemoryEngine protocol (has `get_stats()` method)
  - **Example**: `ZapomniEngine(storage=db, chunker=chunker, embedder=embedder)`

**Returns**: None (constructor)

**Raises**: None (type errors caught at type-checking time)

**Preconditions**:
- Core engine must be initialized
- Core engine should be connected to database

**Postconditions**:
- `self.core` set to provided engine
- Tool is ready to execute

**Algorithm Outline**:
```
1. Store core engine reference in self.core
2. Log tool initialization
3. Return (no further setup needed)
```

**Edge Cases**:
1. **Core is None**: Type checker prevents this (non-optional parameter)
2. **Core not implementing protocol**: Type checker catches this
3. **Core not initialized**: Will fail at execute() time with CoreError

**Related Methods**:
- Called by: MCP server during tool registration
- Calls: `structlog.get_logger()` for logging

---

### Method 2: `execute`

**Signature**:
```python
async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]
```

**Purpose**: Retrieve system statistics and return formatted MCP response.

**Parameters**:
- `arguments`: Dict[str, Any]
  - **Description**: MCP tool arguments (expected to be empty {})
  - **Constraints**: Must be a dict, content is ignored
  - **Example**: `{}` (empty dict)

**Returns**:
- Type: `Dict[str, Any]`
- Structure: MCP response format
  ```python
  {
      "content": [{"type": "text", "text": "Statistics..."}],
      "isError": False  # or True on error
  }
  ```

**Raises**: None (all exceptions caught and converted to error responses)

**Preconditions**:
- Tool initialized (`__init__` called)
- Core engine connected and healthy

**Postconditions**:
- Statistics retrieved from core (if successful)
- Response logged
- MCP response returned

**Algorithm Outline**:
```
1. Log request
2. Validate arguments is dict (else return error)
3. Call self.core.get_stats()
4. If CoreError: log and return error response
5. If unexpected error: log and return generic error
6. Format stats via _format_response()
7. Log success
8. Return formatted response
```

**Edge Cases**:

1. **Arguments not a dict**:
   - Return error response: "Arguments must be a dictionary"
   - Log warning

2. **Core.get_stats() raises CoreError**:
   - Catch exception
   - Log error with details
   - Return error response with exception message

3. **Core.get_stats() raises unexpected exception**:
   - Catch exception
   - Log error with type and message
   - Return generic error (don't leak internals)

4. **Stats dict missing required keys**:
   - _format_response() uses `.get()` with defaults
   - Shows "0" or "unknown" for missing values

5. **Stats dict has extra keys**:
   - Ignored by _format_response()
   - Only known keys are formatted

**Related Methods**:
- Calls: `self.core.get_stats()` (core engine)
- Calls: `self._format_response(stats)` (private method)
- Called by: MCP server when tool is invoked

---

### Method 3: `_format_response` (Private)

**Signature**:
```python
def _format_response(self, stats: Dict[str, Any]) -> Dict[str, Any]
```

**Purpose**: Format raw statistics dict into human-readable MCP response.

**Parameters**:
- `stats`: Dict[str, Any]
  - **Description**: Statistics from core engine
  - **Expected keys**:
    - `total_memories`: int (required)
    - `total_chunks`: int (required)
    - `database_size_mb`: float (required)
    - `graph_name`: str (required)
    - `total_entities`: int (optional)
    - `cache_hit_rate`: float (optional, 0.0-1.0)
    - `avg_query_latency_ms`: float (optional)
  - **Example**:
    ```python
    {
        "total_memories": 1234,
        "total_chunks": 5678,
        "database_size_mb": 45.67,
        "graph_name": "zapomni_memory",
        "cache_hit_rate": 0.653,
        "avg_query_latency_ms": 23.4
    }
    ```

**Returns**:
- Type: `Dict[str, Any]`
- Structure: MCP response format
  ```python
  {
      "content": [{"type": "text", "text": "formatted statistics"}],
      "isError": False
  }
  ```

**Raises**: None

**Algorithm Outline**:
```
1. Create list of output lines
2. Add "Memory System Statistics:" header
3. Add required fields with formatting:
   - total_memories with thousand separators
   - total_chunks with thousand separators
   - database_size_mb with 2 decimal places
   - graph_name as-is
4. Add optional fields if present and not None:
   - total_entities with thousand separators
   - cache_hit_rate as percentage (1 decimal)
   - avg_query_latency_ms with 1 decimal
5. Join lines with newline
6. Wrap in MCP response structure
7. Return response dict
```

**Edge Cases**:

1. **Missing required key**:
   - Use `.get(key, default)` to provide fallback
   - total_memories → 0
   - total_chunks → 0
   - database_size_mb → 0.0
   - graph_name → "unknown"

2. **Optional key is None**:
   - Skip adding that line
   - Only show if key exists AND value is not None

3. **Optional key is 0**:
   - Show it (0% hit rate is valid information)

4. **Negative values**:
   - No validation (trust core layer)
   - Format as-is (may look weird but indicates bug)

5. **Very large numbers**:
   - Thousand separators handle readability (1,234,567,890)

**Related Methods**:
- Called by: `execute()` after successful stats retrieval
- Calls: None (pure formatting)

---

## Error Handling

### Exceptions Defined

This component defines **no custom exceptions**. It catches and handles exceptions from dependencies:

**Caught Exceptions**:

1. `CoreError` (from `zapomni_core.exceptions`)
   - **When**: Core engine fails to retrieve statistics
   - **Examples**: ProcessingError, DatabaseError, SearchError
   - **Handling**: Log error, return MCP error response with message

2. `Exception` (catch-all)
   - **When**: Unexpected error (bug or unforeseen condition)
   - **Handling**: Log error with type, return generic error message

### Error Recovery

**No Retry Logic**:
- Stats retrieval is lightweight (< 100ms expected)
- If it fails, likely due to database issue
- Retrying won't help (would just delay error response)
- Client can retry entire tool call if needed

**Fallback Behavior**: None
- Cannot provide stats if core fails
- Return error response immediately

**Error Propagation**:
- **Do NOT propagate exceptions** from execute()
- All exceptions caught and converted to MCP error responses
- This prevents MCP server crashes

### Logging Strategy

**Log Levels**:

1. **INFO**: Normal operations
   - Tool initialization
   - Stats request received
   - Stats retrieved successfully

2. **DEBUG**: Detailed information
   - Raw stats dict content (for debugging)

3. **WARNING**: Unusual but non-fatal
   - Arguments not a dict (edge case)

4. **ERROR**: Failures
   - CoreError during get_stats()
   - Unexpected exceptions

**Log Fields** (structured logging):
```python
logger.info("get_stats_success", stats_keys=["total_memories", "total_chunks", ...])
logger.error("get_stats_core_error", error="DB connection failed", error_type="DatabaseError")
```

---

## Usage Examples

### Basic Usage

```python
# File: zapomni_mcp/server.py

from zapomni_core.engine import ZapomniEngine
from zapomni_db.falkordb_client import FalkorDBClient
from zapomni_mcp.tools.get_stats import GetStatsTool

# Initialize dependencies
db_client = FalkorDBClient(host="localhost", port=6379, graph_name="zapomni_memory")
await db_client.connect()

engine = ZapomniEngine(
    storage=db_client,
    chunker=chunker,
    embedder=embedder
)

# Create tool
get_stats_tool = GetStatsTool(core=engine)

# Execute (no arguments needed)
result = await get_stats_tool.execute({})

# Result:
# {
#     "content": [{
#         "type": "text",
#         "text": "Memory System Statistics:\n"
#                 "Total Memories: 1,234\n"
#                 "Total Chunks: 5,678\n"
#                 "Database Size: 45.67 MB\n"
#                 "Graph Name: zapomni_memory\n"
#                 "Cache Hit Rate: 65.3%\n"
#                 "Avg Query Latency: 23.4 ms"
#     }],
#     "isError": False
# }
```

### Advanced Usage (MCP Server Integration)

```python
# File: zapomni_mcp/server.py

from mcp.server import Server
from mcp.server.stdio import stdio_server
from zapomni_mcp.tools.get_stats import GetStatsTool

async def main():
    """Run MCP server with get_stats tool."""

    # Initialize server
    server = Server("zapomni-mcp")

    # Create engine and tools
    engine = await create_engine()
    get_stats_tool = GetStatsTool(core=engine)

    # Register tool
    @server.list_tools()
    async def list_tools():
        return [
            {
                "name": get_stats_tool.name,
                "description": get_stats_tool.description,
                "inputSchema": get_stats_tool.input_schema
            }
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        if name == "get_stats":
            return await get_stats_tool.execute(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")

    # Run server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### Error Handling Example

```python
# Simulate error scenario

from zapomni_mcp.tools.get_stats import GetStatsTool
from zapomni_core.exceptions import DatabaseError

# Mock core that raises error
class MockEngineWithError:
    async def get_stats(self):
        raise DatabaseError("Connection to FalkorDB lost")

# Create tool with mock
tool = GetStatsTool(core=MockEngineWithError())

# Execute
result = await tool.execute({})

# Result:
# {
#     "content": [{
#         "type": "text",
#         "text": "Error: Failed to retrieve statistics - Connection to FalkorDB lost"
#     }],
#     "isError": True
# }
```

---

## Testing Approach

### Unit Tests Required

**Test File**: `tests/unit/tools/test_get_stats_tool.py`

**Test Cases**:

1. **test_init_success()**
   - Initialize tool with valid core engine
   - Assert `self.core` is set correctly
   - Assert class attributes are correct

2. **test_execute_success_minimal_stats()**
   - Mock core.get_stats() to return minimal stats (required fields only)
   - Execute tool
   - Assert response.isError == False
   - Assert text contains expected values

3. **test_execute_success_full_stats()**
   - Mock core.get_stats() to return all stats (including optional fields)
   - Execute tool
   - Assert response contains all formatted fields

4. **test_execute_empty_arguments()**
   - Call execute({})
   - Should succeed (no parameters required)

5. **test_execute_ignores_extra_arguments()**
   - Call execute({"foo": "bar"})
   - Should succeed (extra arguments ignored)

6. **test_execute_invalid_arguments_type()**
   - Call execute("not a dict")
   - Assert isError == True
   - Assert error message mentions "dictionary"

7. **test_execute_core_error_database()**
   - Mock core.get_stats() to raise DatabaseError
   - Execute tool
   - Assert isError == True
   - Assert error message contains exception text

8. **test_execute_unexpected_error()**
   - Mock core.get_stats() to raise KeyError (unexpected)
   - Execute tool
   - Assert isError == True
   - Assert generic error message (no internal details leaked)

9. **test_format_response_required_fields_only()**
   - Call _format_response() with minimal stats
   - Assert formatted text contains required fields
   - Assert optional fields are absent

10. **test_format_response_all_fields()**
    - Call _format_response() with all fields
    - Assert all fields formatted correctly

11. **test_format_response_formatting()**
    - Verify thousand separators (1234 → "1,234")
    - Verify percentage formatting (0.653 → "65.3%")
    - Verify decimal places (45.678 → "45.68 MB")

12. **test_format_response_missing_required_key()**
    - Call _format_response({})
    - Should use defaults (0, "unknown")
    - Should not crash

### Mocking Strategy

**Mock MemoryEngine**:
```python
from unittest.mock import AsyncMock, Mock

@pytest.fixture
def mock_engine():
    """Mock MemoryEngine for testing."""
    engine = Mock()
    engine.get_stats = AsyncMock(return_value={
        "total_memories": 1234,
        "total_chunks": 5678,
        "database_size_mb": 45.67,
        "graph_name": "zapomni_memory",
        "cache_hit_rate": 0.653,
        "avg_query_latency_ms": 23.4
    })
    return engine

@pytest.mark.asyncio
async def test_execute_success(mock_engine):
    """Test successful stats retrieval."""
    tool = GetStatsTool(core=mock_engine)
    result = await tool.execute({})

    assert result["isError"] is False
    assert "1,234" in result["content"][0]["text"]
    mock_engine.get_stats.assert_called_once()
```

**Mock Exceptions**:
```python
from zapomni_core.exceptions import DatabaseError

@pytest.mark.asyncio
async def test_execute_database_error(mock_engine):
    """Test handling of database error."""
    mock_engine.get_stats = AsyncMock(side_effect=DatabaseError("Connection lost"))

    tool = GetStatsTool(core=mock_engine)
    result = await tool.execute({})

    assert result["isError"] is True
    assert "Connection lost" in result["content"][0]["text"]
```

### Integration Tests

**Test File**: `tests/integration/test_get_stats_integration.py`

**Test Cases**:

1. **test_get_stats_with_real_engine()**
   - Use real ZapomniEngine with mock database
   - Add some test memories first
   - Call get_stats tool
   - Verify counts are correct

2. **test_get_stats_empty_database()**
   - Fresh database with no data
   - All counts should be 0
   - Should not crash

3. **test_get_stats_after_add_memory()**
   - Add memory via AddMemoryTool
   - Call get_stats
   - Verify counts incremented

**Integration Test Environment**:
- Real `zapomni_core.ZapomniEngine`
- Mock `zapomni_db.FalkorDBClient` (or test database)
- Real `GetStatsTool`

---

## Performance Considerations

### Time Complexity

**execute() method**:
- **O(1)**: Constant time (single get_stats() call)
- No loops or recursive operations

**_format_response() method**:
- **O(n)**: Linear in number of stats fields (typically ~7 fields)
- String concatenation via list (efficient)

**Overall**: O(1) - dominated by database query time in core layer

### Space Complexity

**Memory Usage**:
- **O(1)**: Constant space
- No large data structures
- Stats dict is small (~7 key-value pairs)
- Formatted string is ~200 characters max

**No Memory Leaks**:
- No persistent state
- Each execute() creates fresh response dict
- Garbage collected after response sent

### Optimization Opportunities

**Current Performance**: Already optimal for this use case

**Potential Optimizations** (not needed now):
1. **Cache stats for N seconds**: Reduce database load if called frequently
   - Trade-off: Stale data
   - Not implemented (stats should be fresh)

2. **Lazy formatting**: Format only if needed
   - Trade-off: More complex code
   - Not worth it (formatting is trivial)

**Performance Targets**:
- **Execute latency**: < 100ms (P95)
- **Memory overhead**: < 1MB
- **CPU usage**: < 1% during execution

---

## References

### Module Spec
- [zapomni_mcp_module.md](../level1/zapomni_mcp_module.md) - Parent module specification

### Related Components
- [add_memory_tool_component.md](./add_memory_tool_component.md) - Similar tool structure
- [search_memory_tool_component.md](./search_memory_tool_component.md) - Similar tool structure

### Interface Contracts
- [cross_module_interfaces.md](../level1/cross_module_interfaces.md) - MemoryEngine protocol definition

### External Docs
- MCP Specification: https://spec.modelcontextprotocol.io/
- Pydantic Documentation: https://docs.pydantic.dev/
- structlog Documentation: https://www.structlog.org/

---

## Document Status

**Version:** 1.0
**Created:** 2025-11-23
**Author:** Goncharenko Anton aka alienxs2
**Copyright:** Copyright (c) 2025 Goncharenko Anton aka alienxs2
**License:** MIT License
**Status:** Draft - Ready for Verification

**Next Steps**:
1. Verification review
2. Approval
3. Implementation (write code from this spec)
4. Write tests based on test scenarios
5. Proceed to Level 3 (Function specs for execute, _format_response)

---

**Estimated Implementation Time**: 2-3 hours (including tests)
**Lines of Code**: ~150 (tool) + ~200 (tests) = ~350 total
**Dependencies**: zapomni_core (MemoryEngine protocol)
