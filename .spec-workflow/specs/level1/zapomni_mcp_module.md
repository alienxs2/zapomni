# Zapomni MCP Module - Module Specification

**Level:** 1 (Module)
**Author:** Goncharenko Anton aka alienxs2
**Status:** Draft
**Version:** 1.0
**Created:** 2025-11-23

---

## Overview

### Purpose

The Zapomni MCP Module provides a **thin MCP protocol adapter** that exposes the Zapomni memory system as standardized MCP tools via stdio transport. This module serves as the integration layer between Claude Desktop (and other MCP clients) and the Zapomni core processing engine.

**Key Functions:**
- Implements MCP protocol specification (stdio transport)
- Exposes 3 core MCP tools: `add_memory`, `search_memory`, `get_stats`
- Validates incoming requests and formats responses
- Handles MCP-specific error marshalling
- Delegates all business logic to `zapomni_core`

**Design Philosophy:** Pure adapter pattern - zero business logic, maximum simplicity, clean delegation.

### Scope

**Included in This Module:**
- MCP server implementation (stdio transport only)
- Tool definitions and registration
- Request/response schema validation (Pydantic models)
- MCP-compliant error formatting
- Logging setup (stderr only, JSON structured)
- Configuration management (environment variables)

**Explicitly NOT Included (Delegated):**
- Memory processing logic → `zapomni_core.processors`
- Embedding generation → `zapomni_core.embeddings`
- Database operations → `zapomni_db.falkordb`
- Search algorithms → `zapomni_core.search`
- Entity extraction → `zapomni_core.extractors`

**Rationale:** Strict separation of concerns enables testing MCP protocol compliance independently from business logic implementation.

### Position in Architecture

**System Context:**
```
┌─────────────────┐
│   MCP Client    │
│ (Claude Desktop)│
└────────┬────────┘
         │ stdio (JSON-RPC 2.0)
         ↓
┌─────────────────┐
│  zapomni_mcp    │ ← THIS MODULE
│  (MCP Server)   │
└────────┬────────┘
         │ Python API
         ↓
┌─────────────────┐
│  zapomni_core   │
│ (Business Logic)│
└────────┬────────┘
         │ Python API
         ↓
┌─────────────────┐
│   zapomni_db    │
│ (Data Layer)    │
└─────────────────┘
```

**Layer Responsibilities:**
- **zapomni_mcp:** Protocol adapter (this module)
- **zapomni_core:** Processing engine
- **zapomni_db:** Storage abstraction

**Communication Flow:**
1. Claude Desktop sends MCP request via stdin
2. zapomni_mcp validates and parses request
3. zapomni_mcp calls appropriate zapomni_core method
4. zapomni_core processes and returns result
5. zapomni_mcp formats response as MCP JSON
6. zapomni_mcp writes response to stdout

---

## Architecture

### High-Level Module Diagram

```
zapomni_mcp/
│
├── server.py                  # Main entry point, stdio loop
│       ↓
├── tools/                     # MCP tool implementations
│   ├── __init__.py            # Tool registry
│   ├── add_memory.py          # add_memory tool
│   ├── search_memory.py       # search_memory tool
│   └── get_stats.py           # get_stats tool
│       ↓
├── schemas/                   # Request/response models
│   ├── __init__.py
│   ├── requests.py            # Pydantic request models
│   └── responses.py           # Pydantic response models
│       ↓
├── config.py                  # Configuration (env vars)
│       ↓
└── logging.py                 # Structured logging setup
```

### Key Responsibilities

1. **MCP Protocol Compliance**
   - Implement stdio transport (JSON-RPC 2.0 over stdin/stdout)
   - Adhere to MCP specification (tool registration, execution)
   - Handle protocol-level errors gracefully

2. **Tool Registration & Routing**
   - Register all available tools with MCP server
   - Route incoming tool calls to correct handlers
   - Support tool discovery (list_tools)

3. **Input Validation**
   - Validate all incoming arguments against schemas
   - Sanitize inputs before passing to core
   - Return clear validation errors to client

4. **Response Formatting**
   - Convert core results to MCP response format
   - Ensure JSON serialization correctness
   - Include proper error metadata

5. **Error Handling**
   - Catch exceptions from core layer
   - Format as MCP-compliant error responses
   - Log errors to stderr (never stdout)
   - Never leak sensitive information in errors

6. **Configuration Management**
   - Load environment variables
   - Provide defaults for all settings
   - Validate configuration on startup

---

## Public API

### Module Entry Point

**Command-line invocation:**
```bash
python -m zapomni_mcp.server
```

**Expected behavior:**
- Starts MCP server
- Listens on stdin for requests
- Writes responses to stdout
- Logs to stderr
- Runs until EOF or SIGINT

**Server Initialization Pattern (Dependency Injection):**

```python
# File: zapomni_mcp/server.py

import asyncio
from zapomni_core import ZapomniCore
from zapomni_db import FalkorDBClient
from zapomni_mcp import MCPServer
from zapomni_mcp.tools import AddMemoryTool, SearchMemoryTool, GetStatsTool

async def main():
    """
    Main entry point with proper dependency injection.

    CRITICAL: Initialize all dependencies ONCE at startup, then pass
    to tools via server registration. This prevents recreating DB
    connections on every request (100-200ms overhead).
    """

    # 1. Initialize database client ONCE
    db_client = FalkorDBClient(
        host=config.FALKORDB_HOST,
        port=config.FALKORDB_PORT,
        graph_name=config.GRAPH_NAME
    )

    # 2. Initialize core engine ONCE with DB client
    core_engine = ZapomniCore(
        db=db_client,
        ollama_host=config.OLLAMA_HOST,
        embedding_model=config.EMBEDDING_MODEL
    )

    # 3. Initialize MCP server with core engine
    server = MCPServer(core_engine=core_engine)

    # 4. Register tools with access to shared core_engine
    #    Tools receive core_engine via closure (no re-initialization)
    server.register_tool(AddMemoryTool(engine=core_engine))
    server.register_tool(SearchMemoryTool(engine=core_engine))
    server.register_tool(GetStatsTool(engine=core_engine))

    # 5. Run server (blocks until shutdown)
    await server.run()

if __name__ == "__main__":
    asyncio.run(main())
```

**Anti-Pattern (DO NOT DO THIS):**

```python
# WRONG: Creating new instances inside tool handler
async def add_memory_tool(text: str, metadata: dict) -> dict:
    # ❌ BAD: Creates new DB connection on EVERY call (100-200ms overhead)
    db = FalkorDBClient(...)
    processor = DocumentProcessor(db=db)
    result = await processor.add_memory(text, metadata)
    return result
```

**Correct Pattern:**

```python
# CORRECT: Use dependency injection via constructor
class AddMemoryTool:
    def __init__(self, engine: ZapomniCore):
        # ✅ GOOD: Engine initialized once, reused for all calls
        self.engine = engine

    async def execute(self, arguments: dict) -> dict:
        # Uses shared self.engine (no re-initialization)
        result = await self.engine.add_memory(...)
        return format_mcp_response(result)
```

### Interfaces

#### 1. MCPTool Protocol

```python
from typing import Protocol, Any

class MCPTool(Protocol):
    """Interface that all MCP tools must implement.

    This is a structural typing protocol (duck typing) rather than
    a concrete base class. Any class implementing these attributes
    and methods is compatible.
    """

    name: str
    """Tool name as exposed to MCP clients (e.g., 'add_memory')."""

    description: str
    """Human-readable description of what the tool does."""

    input_schema: dict[str, Any]
    """JSON Schema defining valid input arguments."""

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute tool with provided arguments.

        Args:
            arguments: Dictionary of arguments matching input_schema

        Returns:
            Dictionary in MCP response format (REQUIRED):
            {
                "content": [
                    {"type": "text", "text": "result message"}
                ],
                "isError": false
            }

            Note: ALL tool implementations MUST wrap their responses in this format.
            Do NOT return raw AddMemoryResponse, SearchMemoryResponse, or GetStatsResponse.

        Raises:
            ValidationError: If arguments invalid
            ProcessingError: If core processing fails
            DatabaseError: If database operation fails
        """
        ...
```

#### 2. MCPServer Class

```python
from mcp.server import Server
from mcp.server.stdio import stdio_server

class MCPServer:
    """Main MCP server implementing stdio transport.

    This is a wrapper around the official MCP Python SDK's Server class,
    providing Zapomni-specific initialization and tool registration.

    Attributes:
        server: mcp.server.Server instance
        core_engine: zapomni_core.ZapomniCore instance
        tools: List of registered MCPTool instances
        config: Configuration settings

    Example:
        >>> from zapomni_core import ZapomniCore
        >>> core = ZapomniCore(config)
        >>> server = MCPServer(core_engine=core)
        >>> await server.run()  # Blocks until EOF or SIGINT
    """

    def __init__(
        self,
        core_engine: 'ZapomniCore',
        config: Optional['Settings'] = None
    ) -> None:
        """Initialize MCP server with core engine.

        CRITICAL: Dependencies (core_engine) MUST be initialized ONCE
        at server startup and passed to MCPServer constructor.
        DO NOT create new instances inside tool handlers.

        Args:
            core_engine: Zapomni core processing engine (initialized once)
            config: Configuration settings (uses defaults if None)

        Raises:
            ConfigurationError: If required config missing
        """

    def register_tool(self, tool: MCPTool) -> None:
        """Register a tool with the MCP server.

        Args:
            tool: Tool instance implementing MCPTool protocol

        Raises:
            ValueError: If tool.name already registered
        """

    async def run(self) -> None:
        """Start server and process requests from stdin.

        Blocks until:
        - stdin receives EOF (normal shutdown)
        - SIGINT/SIGTERM received (graceful shutdown)
        - Unrecoverable error (crash)

        Returns:
            None (runs indefinitely until stopped)

        Raises:
            RuntimeError: If server already running
        """
```

### Data Models

#### Request Models (Pydantic)

```python
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional, List

class AddMemoryRequest(BaseModel):
    """Request schema for add_memory tool."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=10_000_000,  # 10MB
        description="Text content to remember"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata (tags, source, date, etc.)"
    )

    @validator("text")
    def validate_text_not_whitespace(cls, v: str) -> str:
        """Ensure text is not just whitespace."""
        if not v.strip():
            raise ValueError("text cannot be empty or whitespace-only")
        return v.strip()

class SearchMemoryRequest(BaseModel):
    """Request schema for search_memory tool."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Natural language search query"
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results to return"
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional filters (date_from, date_to, tags, source)"
    )

    @validator("query")
    def validate_query_not_whitespace(cls, v: str) -> str:
        """Ensure query is not just whitespace."""
        if not v.strip():
            raise ValueError("query cannot be empty or whitespace-only")
        return v.strip()

class GetStatsRequest(BaseModel):
    """Request schema for get_stats tool.

    This tool requires no parameters.
    """
    pass  # No parameters needed
```

#### Response Models (Pydantic)

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class AddMemoryResponse(BaseModel):
    """Response schema for add_memory tool."""

    status: str = Field(..., description="'success' or 'error'")
    memory_id: str = Field(..., description="UUID of stored memory")
    chunks_created: int = Field(..., ge=0, description="Number of chunks generated")
    text_preview: str = Field(..., description="First 100 chars of text")
    error: Optional[str] = Field(default=None, description="Error message if status='error'")

class SearchResult(BaseModel):
    """Individual search result."""

    memory_id: str = Field(..., description="UUID of memory")
    text: str = Field(..., description="Chunk text")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity")
    tags: List[str] = Field(default_factory=list, description="Memory tags")
    source: str = Field(default="", description="Source identifier")
    timestamp: Optional[datetime] = Field(default=None, description="When memory was created")

class SearchMemoryResponse(BaseModel):
    """Response schema for search_memory tool."""

    status: str = Field(..., description="'success' or 'error'")
    count: int = Field(..., ge=0, description="Number of results returned")
    results: List[SearchResult] = Field(..., description="Search results")
    error: Optional[str] = Field(default=None, description="Error message if status='error'")

class Statistics(BaseModel):
    """Statistics about memory system."""

    total_memories: int = Field(..., ge=0, description="Total memories stored")
    total_chunks: int = Field(..., ge=0, description="Total chunks")
    database_size_mb: float = Field(..., ge=0.0, description="Database size in MB")
    graph_name: str = Field(..., description="FalkorDB graph name")
    cache_hit_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    avg_query_latency_ms: Optional[float] = Field(default=None, ge=0.0)

class GetStatsResponse(BaseModel):
    """Response schema for get_stats tool."""

    status: str = Field(..., description="'success' or 'error'")
    statistics: Statistics = Field(..., description="System statistics")
    error: Optional[str] = Field(default=None, description="Error message if status='error'")
```

---

## Dependencies

### External Dependencies

**MCP Protocol:**
- `mcp>=0.1.0` (Official MCP Python SDK)
  - **Purpose:** MCP protocol implementation, stdio transport
  - **Why:** Official SDK ensures protocol compliance
  - **Alternative:** None (required for MCP)

**Validation & Data Models:**
- `pydantic>=2.5.0` (Data validation)
  - **Purpose:** Request/response validation, type safety
  - **Why:** Industry-standard validation, excellent error messages
  - **Alternative:** `marshmallow` (less type-safe)

- `pydantic-settings>=2.1.0` (Configuration)
  - **Purpose:** Environment variable loading
  - **Why:** Integrates with Pydantic models
  - **Alternative:** `python-dotenv` + manual parsing

**Logging:**
- `structlog>=23.2.0` (Structured logging)
  - **Purpose:** JSON-formatted logs to stderr
  - **Why:** Machine-readable logs, context propagation
  - **Alternative:** Standard `logging` (less structured)

**Utilities:**
- `python-dotenv>=1.0.0` (Environment file loading)
  - **Purpose:** Load `.env` file during development
  - **Why:** Simplifies local development
  - **Alternative:** Manual environment variable setting

### Internal Dependencies

**zapomni_core:**
- `zapomni_core.ZapomniCore` (Main processing engine)
  - **Purpose:** All business logic (chunking, embedding, storage)
  - **Interface:** `add(text, metadata)`, `search(query, limit, filters)`, `get_stats()`

**zapomni_db:**
- Indirectly via `zapomni_core` (no direct dependency)
  - **Rationale:** MCP layer should not know about database implementation

### Dependency Rationale

**Why Thin Dependencies?**
- MCP module should be lightweight
- Easy to swap transport layer in future (stdio → HTTP)
- Minimal surface area for security vulnerabilities

**Why No Direct Database Dependency?**
- Enforces clean architecture (adapter pattern)
- MCP layer testable without database
- Core layer handles all data access

**Why Pydantic Over Alternatives?**
- Type hints provide IDE autocomplete
- Automatic JSON schema generation (for MCP input_schema)
- Excellent validation error messages for users
- Compatible with FastAPI if we add HTTP transport later

---

## Data Flow

### Input

**Source:** stdin (JSON-RPC 2.0 messages from MCP client)

**Format:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "add_memory",
    "arguments": {
      "text": "Sample text",
      "metadata": {"source": "user"}
    }
  }
}
```

**Validation:**
1. JSON deserialization (catch malformed JSON)
2. JSON-RPC structure validation (method, params)
3. Tool name validation (is tool registered?)
4. Arguments validation (against Pydantic schema)

**Constraints:**
- Maximum request size: 100MB (enforced by MCP SDK)
- Maximum text size: 10MB (enforced by AddMemoryRequest)
- UTF-8 encoding required

### Processing

**Flow for add_memory:**
```
1. Parse stdin → ToolRequest
2. Validate arguments → AddMemoryRequest (Pydantic)
3. Call core_engine.add(text, metadata)
4. Receive result → AddMemoryResponse (from core)
5. Wrap in MCP response format:
   {
       "content": [{
           "type": "text",
           "text": f"Memory stored. ID: {result.memory_id}, Chunks: {result.chunks_created}"
       }],
       "isError": false
   }
6. Write to stdout
```

**CRITICAL**: Step 5 is mandatory. Tools MUST NOT return AddMemoryResponse directly.

**Flow for search_memory:**
```
1. Parse stdin → ToolRequest
2. Validate arguments → SearchMemoryRequest (Pydantic)
3. Call core_engine.search(query, limit, filters)
4. Receive result → SearchMemoryResponse (from core)
5. Wrap in MCP response format:
   {
       "content": [{
           "type": "text",
           "text": f"Found {result.count} results:\n{formatted_results}"
       }],
       "isError": false
   }
6. Write to stdout
```

**CRITICAL**: Step 5 is mandatory. Format search results as readable text within MCP envelope.

**Flow for get_stats:**
```
1. Parse stdin → ToolRequest
2. No argument validation (no params)
3. Call core_engine.get_stats()
4. Receive result → GetStatsResponse (from core)
5. Wrap in MCP response format:
   {
       "content": [{
           "type": "text",
           "text": f"Stats: {result.total_memories} memories, {result.total_chunks} chunks, ..."
       }],
       "isError": false
   }
6. Write to stdout
```

**CRITICAL**: Step 5 is mandatory. Format statistics as readable text within MCP envelope.

**Error Handling:**
- ValidationError → MCP error response (400-level)
- ProcessingError → MCP error response (500-level)
- DatabaseError → MCP error response (503-level)
- All errors logged to stderr

### Output

**Destination:** stdout (JSON-RPC 2.0 responses)

**Success Format:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Memory stored successfully. ID: 550e8400-..."
      }
    ],
    "isError": false
  }
}
```

**Error Format:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32600,
    "message": "Invalid request: text cannot be empty",
    "data": {
      "validation_errors": [
        {"field": "text", "error": "must not be empty"}
      ]
    }
  }
}
```

**Guarantees:**
- All responses are valid JSON
- All responses match MCP specification
- stdout used ONLY for MCP messages
- stderr used ONLY for logs
- No partial writes (atomic stdout writes)

---

## Design Decisions

### Decision 1: Stdio Transport Only

**Context:** MCP supports both stdio and HTTP transports. We need to choose initial implementation.

**Options Considered:**
- **Option A:** Stdio only
  - Pros: Simple, secure (process isolation), required by Claude Desktop
  - Cons: Cannot expose HTTP endpoint for web clients

- **Option B:** HTTP only
  - Pros: Web-accessible, familiar REST patterns
  - Cons: Not compatible with Claude Desktop, requires authentication layer

- **Option C:** Both stdio and HTTP
  - Pros: Maximum flexibility
  - Cons: More code to maintain, testing complexity

**Chosen:** Option A (stdio only)

**Rationale:**
- Claude Desktop requires stdio (primary use case)
- Process isolation provides security by default
- Simpler implementation → faster MVP
- Can add HTTP transport later if needed (separate module)
- Follows MCP reference implementations

**Future Path:** Create `zapomni_http` module if HTTP access needed.

### Decision 2: Thin Adapter Pattern

**Context:** Should MCP layer contain any business logic or purely delegate?

**Options Considered:**
- **Option A:** Thin adapter (pure delegation)
  - Pros: Easy to test, clear separation, swappable core
  - Cons: More indirection, extra function calls

- **Option B:** Business logic in MCP layer
  - Pros: Fewer layers, potentially faster
  - Cons: Tight coupling, hard to test, violates SRP

**Chosen:** Option A (thin adapter)

**Rationale:**
- Testability: Can mock core easily
- Flexibility: Can swap MCP SDK or core implementation
- Clarity: Clear boundaries between protocol and logic
- Maintainability: Changes to business logic don't affect protocol
- Performance impact negligible (async I/O dominates)

**Implementation:** MCP tools are ~20 lines each (validate → delegate → format).

### Decision 3: Pydantic for Validation

**Context:** Need to validate incoming arguments and generate JSON schemas.

**Options Considered:**
- **Option A:** Pydantic
  - Pros: Type-safe, auto JSON schema, great errors
  - Cons: Additional dependency

- **Option B:** Manual validation
  - Pros: No dependency, full control
  - Cons: Error-prone, verbose, no type safety

- **Option C:** JSON Schema + jsonschema library
  - Pros: Standard JSON Schema
  - Cons: No type hints, less ergonomic

**Chosen:** Option A (Pydantic)

**Rationale:**
- Type hints provide IDE support during development
- Automatic JSON schema generation for MCP input_schema
- Clear, user-friendly validation errors
- Industry-standard (used by FastAPI, etc.)
- Small dependency cost justified by productivity gain

**Example Benefit:**
```python
# Pydantic (chosen)
class AddMemoryRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10_000_000)
# Automatic validation, JSON schema, type hints

# vs Manual (rejected)
def validate_add_memory(args: dict):
    if "text" not in args:
        raise ValueError("Missing text")
    if not isinstance(args["text"], str):
        raise ValueError("text must be string")
    if len(args["text"]) > 10_000_000:
        raise ValueError("text too long")
    # ... lots more boilerplate
```

### Decision 4: Structured Logging (structlog)

**Context:** Need to log operations for debugging without polluting stdout.

**Options Considered:**
- **Option A:** Standard logging module
  - Pros: Built-in, no dependency
  - Cons: Unstructured text, hard to parse

- **Option B:** structlog
  - Pros: JSON output, context propagation, machine-readable
  - Cons: Additional dependency

- **Option C:** No logging
  - Pros: Simplest
  - Cons: Debugging nightmares, no observability

**Chosen:** Option B (structlog)

**Rationale:**
- JSON logs easy to parse with `jq` or log aggregators
- Context binding (add request_id to all logs)
- Clear separation: stdout = MCP, stderr = logs
- Small dependency with large observability benefit
- Production-ready from day one

**Log Format:**
```json
{"timestamp": "2025-11-23T10:30:00Z", "level": "info", "event": "memory_added", "memory_id": "550e...", "chunks": 3}
```

### Decision 5: Zero Configuration Required

**Context:** What should happen if user doesn't provide config?

**Options Considered:**
- **Option A:** Fail if config missing
  - Pros: Explicit, forces user to think
  - Cons: Poor UX, breaks "just works" principle

- **Option B:** Sensible defaults for everything
  - Pros: Works out-of-box, great UX
  - Cons: May use wrong defaults silently

- **Option C:** Defaults with warnings
  - Pros: Works + informs user
  - Cons: Warning noise

**Chosen:** Option B (sensible defaults)

**Rationale:**
- Aligns with "local-first" philosophy (should just work)
- Defaults match most common setup (localhost FalkorDB, Ollama)
- Advanced users can override via environment variables
- Fail fast on actual connection errors (DB unreachable)
- Log effective configuration on startup for transparency

**Defaults:**
```bash
FALKORDB_HOST=localhost
FALKORDB_PORT=6379
OLLAMA_HOST=http://localhost:11434
LOG_LEVEL=INFO
```

---

## Non-Functional Requirements

### Performance

**Latency Targets:**
- Request parsing: < 10ms
- Validation: < 5ms
- Core delegation: < 1ms (just function call)
- Response formatting: < 5ms
- **Total MCP overhead: < 20ms** (excluding core processing time)

**Rationale:** MCP layer should add minimal latency. Core processing (embeddings, DB) dominates total time.

**Throughput:**
- Sequential processing (stdio is inherently sequential)
- No throughput target (limited by core layer)

**Resource Usage:**
- Memory: < 50MB baseline (before core initialization)
- CPU: < 5% when idle
- No memory leaks (proper cleanup of async resources)

**Monitoring:**
- Log request count, average latency, error rate
- Expose via get_stats if useful

### Scalability

**Current Scope:** Single-process, single-user
- Stdio transport inherently single-process
- No concurrency within MCP layer (requests processed sequentially)

**Scaling Limitations:**
- One MCP server per Claude Desktop instance
- Cannot handle concurrent requests (stdio limitation)

**Scaling Strategy (Future):**
- Horizontal: Multiple users → multiple processes (Claude handles this)
- Vertical: Faster hardware benefits core layer more than MCP layer

**Bottlenecks:**
- Core processing (embeddings, DB) NOT MCP layer
- MCP layer designed to never be bottleneck

### Security

**Threat Model:**
- **Trusted client:** Claude Desktop is trusted
- **Untrusted input:** User-provided text and metadata
- **Local-only:** No network exposure (stdio)

**Security Measures:**

1. **Input Validation:**
   - All inputs validated against strict schemas
   - Max input sizes enforced (10MB text, 1KB query)
   - No code injection possible (no eval, exec)
   - UTF-8 encoding enforced

2. **Error Sanitization:**
   - Never leak file paths in error messages
   - Never leak database connection strings
   - Generic errors for internal failures
   - Detailed errors only for validation failures

3. **Process Isolation:**
   - Stdio provides natural isolation
   - No shared memory with other processes
   - Each Claude instance = separate process

4. **Logging Security:**
   - Never log sensitive data (full text, metadata)
   - Log hashes/IDs only
   - Logs to stderr (not publicly accessible)

**Non-Requirements:**
- No authentication (process isolation sufficient)
- No authorization (single-user system)
- No encryption (local communication)

### Reliability

**Error Handling Strategy:**

1. **Validation Errors (User Fault):**
   - Return clear error message
   - Include field name and constraint violated
   - HTTP-like status codes (400-level)
   - Example: "text cannot be empty"

2. **Processing Errors (Core Fault):**
   - Return generic error message
   - Log detailed error to stderr
   - HTTP-like status codes (500-level)
   - Example: "Processing failed. Check logs."

3. **Database Errors (Infrastructure Fault):**
   - Return service unavailable message
   - Log error with retry suggestion
   - HTTP-like status codes (503-level)
   - Example: "Database unavailable. Retry in 5 seconds."

**Recovery Strategies:**

- **No automatic retries:** MCP layer does NOT retry (client's responsibility)
- **Fail fast:** Report errors immediately
- **Graceful degradation:** Not applicable (all operations critical)

**Fail-Safe Mechanisms:**

- **Startup checks:** Validate config, test core connectivity
- **Shutdown hooks:** Clean up resources on SIGINT/SIGTERM
- **Exception handling:** Top-level handler prevents crashes

**Reliability Guarantees:**

- ✅ Never crash on bad input (validation catches it)
- ✅ Never hang (no infinite loops, timeouts in core)
- ✅ Never corrupt state (stateless MCP layer)
- ❌ No at-least-once delivery (stdio is best-effort)
- ❌ No idempotency (add_memory creates new memory each time)

**Monitoring:**
- Log error rate (% requests failing)
- Alert if error rate > 5%
- Track error types (validation vs. processing vs. database)

---

## Testing Strategy

### Unit Testing

**Scope:** Test MCP layer in isolation (mock core and MCP SDK)

**What to Test:**

1. **Request Validation:**
   - Valid requests pass validation
   - Invalid requests fail with clear errors
   - Edge cases (empty, too long, wrong type)

2. **Response Formatting:**
   - Core results formatted correctly
   - Errors formatted as MCP errors
   - JSON serialization works

3. **Tool Registration:**
   - All tools registered
   - Duplicate names rejected
   - Tool discovery works

4. **Configuration:**
   - Defaults applied correctly
   - Environment variables override defaults
   - Invalid config fails startup

**Mocking Strategy:**
- Mock `zapomni_core.ZapomniCore` with test doubles
- Mock `mcp.server.Server` if needed (use real SDK when possible)
- Use `pytest-asyncio` for async tests

**Example Test:**
```python
@pytest.mark.asyncio
async def test_add_memory_success(mock_core):
    """Test add_memory with valid input."""
    mock_core.add.return_value = MemoryResult(
        memory_id="550e8400-e29b-41d4-a716-446655440000",
        chunks_created=3
    )

    tool = AddMemoryTool(core=mock_core)
    result = await tool.execute({
        "text": "Sample text",
        "metadata": {"source": "test"}
    })

    assert result["content"][0]["text"].startswith("Memory stored")
    assert result["isError"] is False
    mock_core.add.assert_called_once()
```

**Coverage Target:** 90%+ for MCP layer code

### Integration Testing

**Scope:** Test MCP layer with real core (but mock database)

**Integration Points to Test:**

1. **MCP → Core:**
   - add_memory calls core.add() correctly
   - search_memory calls core.search() correctly
   - get_stats calls core.get_stats() correctly

2. **Error Propagation:**
   - Core ValidationError → MCP error response
   - Core ProcessingError → MCP error response
   - Core DatabaseError → MCP error response

3. **End-to-End Flow:**
   - stdin → parse → validate → core → format → stdout
   - Verify stdout contains valid JSON
   - Verify stderr contains logs

**Test Environment:**
- Real `zapomni_core` instance
- Mock `zapomni_db.FalkorDBClient`
- Use `pytest` fixtures for setup/teardown

**Example Test:**
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_add_memory_integration(core_engine, mock_db):
    """Integration test: MCP layer → Core → Mock DB."""
    server = MCPServer(core_engine=core_engine)

    # Simulate MCP request
    request = {
        "method": "tools/call",
        "params": {
            "name": "add_memory",
            "arguments": {"text": "Integration test"}
        }
    }

    response = await server.handle_request(request)

    assert response["result"]["isError"] is False
    assert "memory_id" in response["result"]["content"][0]["text"]
    mock_db.add_memory.assert_called_once()
```

**Coverage Target:** All critical paths covered

### E2E Testing (Optional)

**Scope:** Test with real MCP client if available

**Approach:**
- Use MCP client simulator or actual Claude Desktop
- Send real stdin messages
- Verify stdout responses
- Not automated (manual testing initially)

**Future:** Automated E2E tests when MCP testing tools available

---

## Future Considerations

### Potential Enhancements

1. **HTTP Transport:**
   - Add `zapomni_mcp_http` module for web access
   - Reuse same tool implementations
   - Add authentication layer

2. **Additional Tools (Phase 2):**
   - `build_graph` (entity extraction)
   - `get_related` (graph traversal)
   - `graph_status` (background task status)

3. **Streaming Responses:**
   - MCP supports streaming (for large results)
   - Implement if search results become huge

4. **Performance Monitoring:**
   - Expose Prometheus metrics
   - Integrate with observability stack

5. **Rate Limiting:**
   - Protect against abusive clients
   - Not needed for single-user Claude Desktop

### Known Limitations

1. **Sequential Processing:**
   - Stdio is inherently sequential
   - Cannot process concurrent requests
   - **Mitigation:** Core layer should be async-ready for future HTTP transport

2. **No Idempotency:**
   - add_memory creates new memory each time
   - No deduplication at MCP layer
   - **Mitigation:** Consider deduplication in core layer (future)

3. **No Authentication:**
   - Relies on process isolation
   - Not suitable for multi-user scenarios
   - **Mitigation:** HTTP transport can add auth (future)

4. **Limited Error Context:**
   - MCP error format is simple
   - Cannot include rich debugging info
   - **Mitigation:** Use logs for detailed debugging

### Evolution Path

**Phase 1 (Current):** 3 basic tools, stdio only
**Phase 2:** Add knowledge graph tools (build_graph, get_related)
**Phase 3:** Add code indexing tools (index_codebase)
**Phase 4:** Optional HTTP transport module

**Migration Strategy:** Maintain backward compatibility in MCP protocol. Core API can evolve independently.

---

## References

### Internal Documents

- **product.md:** Section "Core Features by Phase" → defines 3 core tools
- **tech.md:** Section "MCP Server Architecture" → stdio transport decision
- **structure.md:** Section "zapomni_mcp/" → module organization

### External Specifications

- **MCP Protocol:** https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/architecture/
- **JSON-RPC 2.0:** https://www.jsonrpc.org/specification
- **MCP Python SDK:** https://github.com/anthropics/anthropic-mcp-python

### Related Specs

- **zapomni_core_module.md** (Level 1) - Core processing engine
- **zapomni_db_module.md** (Level 1) - Database layer
- **cross_module_interfaces.md** (Level 1) - Interface contracts

---

## Document Status

**Version:** 1.0
**Created:** 2025-11-23
**Author:** Goncharenko Anton aka alienxs2
**Copyright:** Copyright (c) 2025 Goncharenko Anton aka alienxs2
**License:** MIT License
**Status:** Draft - Ready for Verification

**Next Steps:**
1. Multi-agent verification (5 agents)
2. Synthesis and reconciliation
3. User approval
4. Proceed to Level 2 (Component specs)

---

**Document Length:** ~1000 lines
**Estimated Reading Time:** 30-40 minutes
**Target Audience:** Developers implementing zapomni_mcp module
