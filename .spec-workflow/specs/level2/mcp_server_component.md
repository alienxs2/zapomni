# MCPServer - Component Specification

**Level:** 2 (Component)
**Module:** zapomni_mcp
**Author:** Goncharenko Anton aka alienxs2
**Status:** Draft
**Version:** 1.0
**Created:** 2025-11-23

---

## Overview

### Purpose

The MCPServer component is the main server class that implements the Model Context Protocol (MCP) specification for stdio communication. It manages the complete lifecycle of MCP tool interactions, from initialization and tool registration to request handling and graceful shutdown.

MCPServer acts as the coordination layer between MCP clients (Claude Desktop, Cursor, Cline) and the Zapomni core memory engine, ensuring protocol compliance, proper error handling, and efficient request routing.

### Responsibilities

1. **Server Lifecycle Management**
   - Initialize server with ZapomniCore engine
   - Configure MCP SDK components
   - Manage server state (running, stopped)
   - Handle graceful shutdown on signals

2. **Tool Registry Management**
   - Register all MCP tools with the server
   - Validate tool definitions (name, schema, handler)
   - Prevent duplicate tool registrations
   - Provide tool discovery capabilities

3. **Request Routing & Execution**
   - Parse incoming JSON-RPC 2.0 messages from stdin
   - Route tool calls to appropriate handlers
   - Execute tool logic with proper error handling
   - Format responses according to MCP specification

4. **Error Handling & Logging**
   - Catch and format exceptions as MCP errors
   - Log all operations to stderr (structured logging)
   - Never leak sensitive information in errors
   - Provide clear error messages to clients

5. **State Management**
   - Track registered tools (name → handler mapping)
   - Monitor server running state
   - Maintain request/response statistics
   - Coordinate with background task manager

### Position in Module

MCPServer is the **entry point and orchestrator** for the zapomni_mcp module:

```
zapomni_mcp/
│
├── server.py ← MCPServer (THIS COMPONENT)
│   ↓
├── tools/
│   ├── add_memory.py
│   ├── search_memory.py
│   └── get_stats.py
│   ↓
├── schemas/
│   ├── requests.py
│   └── responses.py
```

**Relationship to Other Components**:
- **Upstream**: Receives requests from MCP clients via stdio
- **Downstream**: Delegates to tool implementations (AddMemoryTool, SearchMemoryTool, etc.)
- **Core Dependency**: Uses ZapomniCore for all business logic

---

## Class Definition

### Class Diagram

```
┌─────────────────────────────────────────────┐
│              MCPServer                      │
├─────────────────────────────────────────────┤
│ - _server: mcp.server.Server                │
│ - _core_engine: ZapomniCore                 │
│ - _tools: Dict[str, MCPTool]                │
│ - _config: Settings                         │
│ - _running: bool                            │
│ - _logger: structlog.BoundLogger            │
│ - _request_count: int                       │
│ - _error_count: int                         │
├─────────────────────────────────────────────┤
│ + __init__(core_engine, config)             │
│ + register_tool(tool: MCPTool) -> None      │
│ + register_all_tools() -> None              │
│ + run() -> None                             │
│ + shutdown() -> None                        │
│ + get_stats() -> ServerStats                │
│ - _setup_signal_handlers() -> None          │
│ - _handle_request(request) -> Response      │
│ - _format_error(exception) -> ErrorResponse │
│ - _validate_tool(tool) -> None              │
└─────────────────────────────────────────────┘
```

### Full Class Signature

```python
from typing import Optional, Dict, Any
from dataclasses import dataclass
import asyncio
import signal
import sys
import structlog
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from zapomni_core import ZapomniCore
from zapomni_mcp.config import Settings
from zapomni_mcp.schemas.requests import ToolRequest
from zapomni_mcp.schemas.responses import ToolResponse, ErrorResponse
from zapomni_mcp.tools import MCPTool


@dataclass
class ServerStats:
    """Statistics about server operation."""
    total_requests: int
    total_errors: int
    registered_tools: int
    uptime_seconds: float
    running: bool


class MCPServer:
    """
    Main MCP protocol server implementing stdio transport.

    This is the central component of the zapomni_mcp module, responsible for:
    - Managing the MCP server lifecycle (start, run, stop)
    - Registering and routing MCP tools
    - Handling requests from MCP clients
    - Delegating business logic to ZapomniCore
    - Logging operations and errors

    The server follows the MCP specification (https://spec.modelcontextprotocol.io/)
    and uses stdio transport for communication with clients.

    Attributes:
        _server: Internal mcp.server.Server instance
        _core_engine: ZapomniCore processing engine for business logic
        _tools: Registry of registered tools (name -> MCPTool instance)
        _config: Server configuration settings
        _running: Flag indicating if server is currently running
        _logger: Structured logger for stderr output
        _request_count: Total number of requests processed
        _error_count: Total number of errors encountered
        _start_time: Server start timestamp for uptime calculation

    Thread Safety:
        MCPServer is NOT thread-safe. Stdio transport is inherently sequential,
        so only one request is processed at a time.

    Example:
        ```python
        from zapomni_core import ZapomniCore
        from zapomni_mcp.server import MCPServer
        from zapomni_mcp.config import Settings

        # Initialize core engine
        core = ZapomniCore(config=core_config)

        # Create and configure MCP server
        config = Settings(log_level="INFO")
        server = MCPServer(core_engine=core, config=config)

        # Register tools and start
        server.register_all_tools()
        await server.run()  # Blocks until shutdown
        ```

    Raises:
        ConfigurationError: If required configuration is missing or invalid
        RuntimeError: If server is already running when run() is called
    """

    def __init__(
        self,
        core_engine: ZapomniCore,
        config: Optional[Settings] = None
    ) -> None:
        """
        Initialize MCP server with core processing engine.

        Sets up the MCP server infrastructure, including:
        - MCP SDK server instance
        - Tool registry (empty initially)
        - Structured logging to stderr
        - Configuration with defaults
        - Signal handlers for graceful shutdown

        Args:
            core_engine: ZapomniCore instance for business logic processing.
                Must be fully initialized and ready to use.

            config: Optional server configuration settings.
                If None, uses defaults from Settings class:
                - log_level: "INFO"
                - server_name: "zapomni-memory"
                - max_concurrent_tasks: 4

        Raises:
            ConfigurationError: If core_engine is None or invalid config provided
            ValueError: If config contains invalid values (e.g., negative timeouts)

        Side Effects:
            - Configures structlog for stderr logging
            - Registers SIGINT and SIGTERM signal handlers
            - Initializes internal MCP server instance

        Preconditions:
            - core_engine must be initialized with valid FalkorDB connection
            - core_engine must be initialized with valid Ollama connection

        Postconditions:
            - Server is initialized but NOT running (call run() to start)
            - Tool registry is empty (call register_all_tools() to populate)
            - Logger is configured and ready
            - Signal handlers installed

        Example:
            ```python
            core = ZapomniCore(
                falkordb_config={"host": "localhost", "port": 6379},
                ollama_config={"host": "http://localhost:11434"}
            )

            server = MCPServer(core_engine=core)
            # Server ready but not running yet
            ```
        """

    def register_tool(self, tool: MCPTool) -> None:
        """
        Register a single MCP tool with the server.

        Adds the tool to the internal registry and configures the MCP SDK
        to recognize and route requests to this tool.

        Tools must implement the MCPTool protocol:
        - name: str (unique identifier)
        - description: str (human-readable purpose)
        - input_schema: dict (JSON Schema for arguments)
        - execute(arguments: dict) -> dict (async handler)

        Args:
            tool: Tool instance implementing MCPTool protocol.
                Must have unique name not already registered.

        Raises:
            ValueError: If tool.name is empty or already registered
            TypeError: If tool doesn't implement MCPTool protocol
            ValidationError: If tool.input_schema is invalid JSON Schema

        Side Effects:
            - Adds tool to _tools registry
            - Registers tool with internal MCP server
            - Logs tool registration

        Preconditions:
            - tool.name must be unique (not in _tools)
            - tool must implement all MCPTool protocol methods

        Postconditions:
            - Tool is registered and callable via MCP
            - Tool appears in list_tools() response

        Example:
            ```python
            from zapomni_mcp.tools import AddMemoryTool

            add_tool = AddMemoryTool(core=core_engine)
            server.register_tool(add_tool)
            # Tool now available to MCP clients
            ```
        """

    def register_all_tools(self) -> None:
        """
        Register all standard Zapomni MCP tools.

        This is a convenience method that registers all core tools:
        - add_memory: Store new information in memory
        - search_memory: Retrieve relevant information
        - get_stats: Query system statistics

        Calls register_tool() for each tool implementation found in
        zapomni_mcp.tools module.

        Raises:
            ValueError: If any tool name conflicts
            ImportError: If tool modules cannot be imported

        Side Effects:
            - Registers 3+ tools (depending on phase)
            - Logs each tool registration
            - Initializes tool instances with core_engine

        Preconditions:
            - core_engine must be initialized
            - Tool modules must be importable

        Postconditions:
            - All standard tools registered and available
            - Server ready to process MCP requests

        Example:
            ```python
            server = MCPServer(core_engine=core)
            server.register_all_tools()
            # All tools now available
            # Can now call: server.run()
            ```

        Note:
            This method is typically called once during server initialization,
            before calling run(). Custom tools should be registered separately
            using register_tool().
        """

    async def run(self) -> None:
        """
        Start the MCP server and process requests from stdin.

        This is the main server loop that:
        1. Sets server state to running
        2. Starts stdio server (blocking operation)
        3. Processes incoming JSON-RPC 2.0 messages
        4. Routes tool calls to registered handlers
        5. Sends responses to stdout
        6. Runs until EOF on stdin or shutdown signal

        The server uses stdio transport as specified by MCP:
        - Reads from stdin (standard input)
        - Writes to stdout (standard output)
        - Logs to stderr (standard error)

        Blocking Behavior:
            This method blocks until one of:
            - stdin receives EOF (pipe closed)
            - SIGINT received (Ctrl+C)
            - SIGTERM received (kill command)
            - shutdown() is called from another context

        Raises:
            RuntimeError: If server is already running
            ConnectionError: If stdin/stdout unavailable
            MCPProtocolError: If invalid MCP messages received

        Side Effects:
            - Sets _running = True
            - Increments _request_count for each request
            - Increments _error_count for each error
            - Writes JSON-RPC responses to stdout
            - Logs operations to stderr

        Preconditions:
            - At least one tool registered (via register_tool/register_all_tools)
            - core_engine is operational (DB + Ollama accessible)
            - stdin/stdout are available and unbuffered

        Postconditions:
            - Server stops processing requests
            - _running = False
            - Resources cleaned up
            - Final statistics logged

        Example:
            ```python
            server = MCPServer(core_engine=core)
            server.register_all_tools()

            # Start server (blocks)
            await server.run()
            # Server runs until shutdown

            # This line only reached after shutdown
            print("Server stopped")
            ```

        Signal Handling:
            - SIGINT (Ctrl+C): Graceful shutdown
            - SIGTERM (kill): Graceful shutdown
            - Both signals trigger shutdown() internally

        Performance:
            - Target latency: < 20ms overhead per request
            - Throughput: Limited by stdio (sequential processing)
        """

    def shutdown(self) -> None:
        """
        Gracefully shut down the MCP server.

        Performs cleanup operations and stops the server loop:
        1. Sets running flag to False
        2. Closes stdin to stop main loop
        3. Flushes stdout (ensure all responses sent)
        4. Logs shutdown statistics
        5. Releases resources

        This method is called automatically by signal handlers
        (SIGINT, SIGTERM) but can also be called programmatically.

        Raises:
            No exceptions raised (best-effort cleanup)

        Side Effects:
            - Sets _running = False
            - Closes stdin stream
            - Flushes stdout stream
            - Logs final statistics (requests, errors, uptime)
            - Stops background task manager (if active)

        Preconditions:
            - None (safe to call even if not running)

        Postconditions:
            - Server no longer processes requests
            - All resources released
            - Statistics logged to stderr

        Thread Safety:
            This method IS thread-safe and can be called from signal
            handlers or other threads.

        Example:
            ```python
            # Manual shutdown
            server.shutdown()

            # Automatic via signal (typical)
            # User presses Ctrl+C → SIGINT → shutdown() called
            ```

        Statistics Logged:
            - Total requests processed
            - Total errors encountered
            - Server uptime in seconds
            - Average request latency
            - Error rate percentage
        """

    def get_stats(self) -> ServerStats:
        """
        Get current server statistics.

        Returns operational metrics about the server's performance
        and state. Useful for monitoring, debugging, and health checks.

        Returns:
            ServerStats object containing:
                - total_requests: Number of requests processed since start
                - total_errors: Number of errors encountered
                - registered_tools: Count of registered tools
                - uptime_seconds: Time since server started
                - running: Current server state (True/False)

        Raises:
            No exceptions raised

        Side Effects:
            None (pure read operation)

        Time Complexity: O(1)

        Example:
            ```python
            stats = server.get_stats()
            print(f"Requests: {stats.total_requests}")
            print(f"Errors: {stats.total_errors}")
            print(f"Error rate: {stats.total_errors / stats.total_requests:.2%}")
            print(f"Uptime: {stats.uptime_seconds:.1f}s")
            ```

        Use Cases:
            - Health check endpoints (if HTTP transport added)
            - Monitoring dashboards
            - Performance profiling
            - Debugging issues
        """

    # Private methods (implementation details)

    def _setup_signal_handlers(self) -> None:
        """
        Install signal handlers for graceful shutdown.

        Registers handlers for:
        - SIGINT (interrupt signal, e.g., Ctrl+C)
        - SIGTERM (termination signal, e.g., kill)

        Both signals trigger shutdown() for graceful cleanup.

        Internal implementation detail, not part of public API.
        """

    async def _handle_request(self, request: ToolRequest) -> ToolResponse:
        """
        Handle a single MCP tool request.

        Internal request processing pipeline:
        1. Parse and validate request
        2. Look up tool by name
        3. Execute tool with arguments
        4. Format response
        5. Update statistics

        Internal implementation detail, not part of public API.
        """

    def _format_error(self, exception: Exception) -> ErrorResponse:
        """
        Format an exception as MCP-compliant error response.

        Converts Python exceptions to JSON-RPC 2.0 error format
        with appropriate error codes and sanitized messages.

        Internal implementation detail, not part of public API.
        """

    def _validate_tool(self, tool: MCPTool) -> None:
        """
        Validate that a tool implements MCPTool protocol correctly.

        Checks:
        - Has name attribute (non-empty string)
        - Has description attribute (string)
        - Has input_schema attribute (valid JSON Schema)
        - Has execute method (callable, async)

        Internal implementation detail, not part of public API.
        """
```

---

## Dependencies

### Component Dependencies

**Internal (zapomni_mcp module)**:
- `MCPTool` protocol from `zapomni_mcp.tools.__init__`
  - Purpose: Interface definition for all tools
  - Usage: Type checking and validation

- `AddMemoryTool` from `zapomni_mcp.tools.add_memory`
  - Purpose: Implements add_memory MCP tool
  - Usage: Registered in register_all_tools()

- `SearchMemoryTool` from `zapomni_mcp.tools.search_memory`
  - Purpose: Implements search_memory MCP tool
  - Usage: Registered in register_all_tools()

- `GetStatsTool` from `zapomni_mcp.tools.get_stats`
  - Purpose: Implements get_stats MCP tool
  - Usage: Registered in register_all_tools()

- `Settings` from `zapomni_mcp.config`
  - Purpose: Configuration management
  - Usage: Server configuration (log level, timeouts, etc.)

- `ToolRequest`, `ToolResponse`, `ErrorResponse` from `zapomni_mcp.schemas`
  - Purpose: Request/response validation
  - Usage: Type validation and MCP compliance

**External (zapomni_core module)**:
- `ZapomniCore` from `zapomni_core`
  - Purpose: Core business logic engine
  - Interface:
    - `add_memory(text, metadata) -> MemoryResult`
    - `search_memory(query, limit, filters) -> List[SearchResult]`
    - `get_stats() -> Statistics`
  - Usage: All business logic delegated to core

**External Libraries**:
- `mcp` (Official MCP Python SDK)
  - `mcp.server.Server` - Core server implementation
  - `mcp.server.stdio.stdio_server` - Stdio transport helper
  - `mcp.types` - MCP type definitions (Tool, TextContent, etc.)
  - Purpose: MCP protocol compliance

- `structlog` (Structured logging)
  - Purpose: JSON-formatted logging to stderr
  - Configuration: Pre-configured with timestamp, level, context

- `pydantic` (Data validation)
  - Used indirectly via schemas module
  - Purpose: Request/response validation

- `asyncio` (Python standard library)
  - Purpose: Async request handling
  - Usage: Event loop for stdio server

- `signal` (Python standard library)
  - Purpose: Signal handler registration
  - Usage: SIGINT and SIGTERM handlers

### Dependency Injection

MCPServer uses **constructor injection** for its main dependency:

```python
# Core engine injected at construction
server = MCPServer(core_engine=core)
```

**Rationale**:
- Clear dependencies (explicit in constructor)
- Easy to test (can inject mock core)
- No hidden dependencies or global state

**Configuration** is also injected but optional:

```python
# With custom config
server = MCPServer(core_engine=core, config=settings)

# With defaults
server = MCPServer(core_engine=core)  # Uses Settings()
```

---

## State Management

### Attributes

**`_server: mcp.server.Server`**
- Type: Internal MCP SDK server instance
- Purpose: Handles MCP protocol details
- Lifetime: Created in __init__, persists until shutdown
- Thread Safety: Single-threaded (stdio is sequential)

**`_core_engine: ZapomniCore`**
- Type: Business logic engine
- Purpose: Processes memory operations
- Lifetime: Injected at construction, referenced throughout
- Thread Safety: Assumed thread-safe by design

**`_tools: Dict[str, MCPTool]`**
- Type: Tool registry (name → handler mapping)
- Purpose: Quick lookup of tools by name
- Lifetime: Initialized empty, populated during registration
- Modifications: Only during registration (before run())
- Thread Safety: Read-only after run() starts

**`_config: Settings`**
- Type: Server configuration
- Purpose: Stores settings (log level, timeouts, etc.)
- Lifetime: Set at construction, immutable afterward
- Thread Safety: Immutable, safe to read

**`_running: bool`**
- Type: Server state flag
- Purpose: Indicates if server is currently running
- Lifetime: False initially, True during run(), False after shutdown
- Thread Safety: Modified by shutdown() (can be from signal handler)

**`_logger: structlog.BoundLogger`**
- Type: Structured logger
- Purpose: Logs operations to stderr
- Lifetime: Created in __init__, used throughout
- Thread Safety: structlog is thread-safe

**`_request_count: int`**
- Type: Request counter
- Purpose: Track total requests processed
- Lifetime: 0 initially, incremented for each request
- Thread Safety: Sequential access (stdio is single-threaded)

**`_error_count: int`**
- Type: Error counter
- Purpose: Track total errors encountered
- Lifetime: 0 initially, incremented for each error
- Thread Safety: Sequential access

**`_start_time: float`**
- Type: Timestamp (from time.time())
- Purpose: Calculate uptime
- Lifetime: Set in run(), used in get_stats()
- Thread Safety: Read-only after set

### State Transitions

```
┌─────────────────┐
│  INITIALIZED    │ ← __init__() called
│  _running=False │
└────────┬────────┘
         │ register_all_tools()
         │
         ▼
┌─────────────────┐
│ TOOLS_REGISTERED│
│  _running=False │
└────────┬────────┘
         │ run()
         │
         ▼
┌─────────────────┐
│    RUNNING      │ ◄─────────┐
│  _running=True  │            │
└────────┬────────┘            │
         │                     │
         ├──► Process Request ─┤
         │                     │
         │ shutdown() or EOF or Signal
         │
         ▼
┌─────────────────┐
│   STOPPED       │
│  _running=False │
└─────────────────┘
```

**Transitions**:
1. **INITIALIZED → TOOLS_REGISTERED**: register_all_tools() or manual register_tool() calls
2. **TOOLS_REGISTERED → RUNNING**: run() starts stdio server
3. **RUNNING → RUNNING**: Each request processed (loop)
4. **RUNNING → STOPPED**: shutdown() or stdin EOF or signal

**Invalid Transitions**:
- RUNNING → RUNNING: Calling run() again raises RuntimeError
- STOPPED → RUNNING: Cannot restart (create new instance)

### Thread Safety

**MCPServer is NOT thread-safe**, but this is intentional:

**Rationale**:
- Stdio transport is inherently sequential (one request at a time)
- No concurrent requests possible from single stdin stream
- Simplifies implementation (no locking needed)

**Exception**: Signal handlers (SIGINT, SIGTERM) may call shutdown() from different thread:
- `_running` flag uses atomic bool operations (safe)
- shutdown() is designed to be signal-safe

**Future Consideration**: If HTTP transport is added (Phase 5+), that component would need thread safety (async handlers, locks for shared state).

---

## Public Methods (Detailed)

### Method 1: `__init__`

**Signature:**
```python
def __init__(
    self,
    core_engine: ZapomniCore,
    config: Optional[Settings] = None
) -> None
```

**Purpose**: Initialize the MCP server with core processing engine and configuration

**Parameters:**

**`core_engine: ZapomniCore`**
- Description: The Zapomni core processing engine that handles all business logic
- Constraints:
  - Must not be None
  - Must be fully initialized with valid connections:
    - FalkorDB connection established and tested
    - Ollama connection established and tested
  - Must implement required interface:
    - `add_memory(text: str, metadata: dict) -> MemoryResult`
    - `search_memory(query: str, limit: int, filters: dict) -> List[SearchResult]`
    - `get_stats() -> Statistics`
- Example:
  ```python
  from zapomni_core import ZapomniCore, CoreConfig

  config = CoreConfig(
      falkordb_host="localhost",
      falkordb_port=6379,
      ollama_host="http://localhost:11434"
  )
  core = ZapomniCore(config=config)
  ```

**`config: Optional[Settings]`**
- Description: Optional server configuration settings
- Default: `None` (creates Settings() with defaults)
- Structure:
  ```python
  @dataclass
  class Settings:
      server_name: str = "zapomni-memory"
      version: str = "0.1.0"
      log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
      max_concurrent_tasks: int = 4
      request_timeout_seconds: int = 300  # 5 minutes
  ```
- Validation:
  - log_level must be valid Python logging level
  - max_concurrent_tasks must be > 0
  - request_timeout_seconds must be > 0
- Example:
  ```python
  from zapomni_mcp.config import Settings

  config = Settings(
      log_level="DEBUG",
      max_concurrent_tasks=8
  )
  ```

**Returns**: None (constructor)

**Raises:**

**`ConfigurationError`**
- When: core_engine is None
- Message: "core_engine cannot be None"
- Example:
  ```python
  server = MCPServer(core_engine=None)  # Raises ConfigurationError
  ```

**`ValidationError`**
- When: config has invalid values
- Message: Specific field and constraint violated
- Example:
  ```python
  config = Settings(log_level="INVALID")  # Raises ValidationError
  ```

**Preconditions:**
- ✅ core_engine is initialized and operational
- ✅ FalkorDB is accessible (can connect)
- ✅ Ollama is accessible (can generate embeddings)
- ✅ (Optional) config has valid values

**Postconditions:**
- ✅ _server instance created (mcp.server.Server)
- ✅ _core_engine reference stored
- ✅ _tools registry initialized (empty dict)
- ✅ _config set (provided or defaults)
- ✅ _running = False
- ✅ _logger configured for stderr
- ✅ Signal handlers installed (SIGINT, SIGTERM)
- ✅ _request_count = 0
- ✅ _error_count = 0

**Algorithm Outline:**
```
1. Validate core_engine is not None
2. If config is None:
     config = Settings()  # Use defaults
3. Validate config values
4. Initialize structlog logger with config.log_level
5. Create mcp.server.Server instance
6. Store core_engine reference
7. Initialize _tools = {}
8. Set _running = False
9. Set _request_count = 0, _error_count = 0
10. Install signal handlers (SIGINT, SIGTERM → shutdown)
11. Log initialization: "MCPServer initialized with X tools"
```

**Edge Cases:**

**Edge Case 1: core_engine is None**
- Scenario: `MCPServer(core_engine=None)`
- Expected Behavior:
  ```python
  raise ConfigurationError("core_engine cannot be None")
  ```
- Test:
  ```python
  def test_init_none_core_raises():
      with pytest.raises(ConfigurationError, match="core_engine cannot be None"):
          MCPServer(core_engine=None)
  ```

**Edge Case 2: core_engine not initialized (DB unavailable)**
- Scenario: core_engine has no DB connection
- Expected Behavior:
  - __init__ succeeds (lazy validation)
  - Error occurs later when run() tries to use core
- Rationale: Allow server to start even if DB temporarily unavailable

**Edge Case 3: config with invalid log level**
- Scenario: `Settings(log_level="TRACE")`
- Expected Behavior:
  ```python
  raise ValidationError("log_level must be DEBUG, INFO, WARNING, or ERROR")
  ```
- Test:
  ```python
  def test_init_invalid_log_level():
      with pytest.raises(ValidationError):
          config = Settings(log_level="TRACE")
          MCPServer(core_engine=mock_core, config=config)
  ```

**Edge Case 4: config with negative timeout**
- Scenario: `Settings(request_timeout_seconds=-1)`
- Expected Behavior:
  ```python
  raise ValidationError("request_timeout_seconds must be positive")
  ```

**Related Methods:**
- Calls: `_setup_signal_handlers()`
- Called by: User code, typically in main() or startup script

---

### Method 2: `register_tool`

**Signature:**
```python
def register_tool(self, tool: MCPTool) -> None
```

**Purpose**: Register a single MCP tool with the server for client invocation

**Parameters:**

**`tool: MCPTool`**
- Description: Tool instance implementing MCPTool protocol
- Type: Any class implementing MCPTool protocol:
  ```python
  class MCPTool(Protocol):
      name: str
      description: str
      input_schema: dict[str, Any]
      async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]: ...
  ```
- Constraints:
  - `name` must be non-empty string
  - `name` must be unique (not already registered)
  - `name` must match pattern: `^[a-z_][a-z0-9_]*$` (lowercase, underscores)
  - `description` must be non-empty string
  - `input_schema` must be valid JSON Schema (checked by jsonschema)
  - `execute` must be async callable
- Example:
  ```python
  from zapomni_mcp.tools import AddMemoryTool

  tool = AddMemoryTool(core=core_engine)
  # tool.name = "add_memory"
  # tool.description = "Store new information in memory"
  # tool.input_schema = {...}  # JSON Schema
  # tool.execute = async method
  ```

**Returns**: None

**Raises:**

**`ValueError`**
- When: tool.name is empty, invalid format, or already registered
- Messages:
  - "Tool name cannot be empty"
  - "Tool name 'XYZ' already registered"
  - "Tool name 'XYZ' invalid format (must be lowercase with underscores)"
- Example:
  ```python
  tool1 = AddMemoryTool(core=core)
  server.register_tool(tool1)
  server.register_tool(tool1)  # Raises ValueError: already registered
  ```

**`TypeError`**
- When: tool doesn't implement MCPTool protocol
- Message: "Tool must implement MCPTool protocol (missing: name, execute, ...)"
- Example:
  ```python
  class BadTool:
      pass  # Missing required attributes

  server.register_tool(BadTool())  # Raises TypeError
  ```

**`ValidationError`**
- When: tool.input_schema is not valid JSON Schema
- Message: "Tool input_schema invalid: <specific error>"
- Example:
  ```python
  class BadTool:
      name = "bad"
      description = "test"
      input_schema = {"type": "invalid"}  # Invalid JSON Schema
      async def execute(self, args): pass

  server.register_tool(BadTool())  # Raises ValidationError
  ```

**Preconditions:**
- ✅ Server is initialized (__init__ called)
- ✅ Server is NOT running (call before run())
- ✅ tool implements MCPTool protocol fully

**Postconditions:**
- ✅ tool added to _tools[tool.name]
- ✅ tool registered with internal _server
- ✅ tool callable via MCP requests
- ✅ Registration logged to stderr

**Algorithm Outline:**
```
1. Validate tool is not None
2. Validate tool implements MCPTool protocol:
     - Has 'name' attribute (str)
     - Has 'description' attribute (str)
     - Has 'input_schema' attribute (dict)
     - Has 'execute' method (async callable)
3. Validate tool.name:
     - Not empty
     - Matches pattern ^[a-z_][a-z0-9_]*$
     - Not in _tools (no duplicates)
4. Validate tool.input_schema is valid JSON Schema
5. Add to registry: _tools[tool.name] = tool
6. Register with MCP SDK: _server.add_tool(...)
7. Log: "Registered tool: {tool.name}"
```

**Edge Cases:**

**Edge Case 1: Duplicate tool name**
- Scenario:
  ```python
  tool1 = AddMemoryTool(core=core)
  tool2 = AddMemoryTool(core=core)
  server.register_tool(tool1)
  server.register_tool(tool2)  # Same name
  ```
- Expected Behavior:
  ```python
  raise ValueError("Tool name 'add_memory' already registered")
  ```
- Test:
  ```python
  def test_register_tool_duplicate_raises():
      tool1 = AddMemoryTool(core=mock_core)
      tool2 = AddMemoryTool(core=mock_core)
      server.register_tool(tool1)
      with pytest.raises(ValueError, match="already registered"):
          server.register_tool(tool2)
  ```

**Edge Case 2: Invalid tool name format**
- Scenario:
  ```python
  tool.name = "Add-Memory"  # Uppercase and dash
  server.register_tool(tool)
  ```
- Expected Behavior:
  ```python
  raise ValueError("Tool name 'Add-Memory' invalid format")
  ```

**Edge Case 3: Tool with missing execute method**
- Scenario:
  ```python
  class BadTool:
      name = "bad"
      description = "test"
      input_schema = {}
      # Missing execute method

  server.register_tool(BadTool())
  ```
- Expected Behavior:
  ```python
  raise TypeError("Tool must implement execute method")
  ```

**Edge Case 4: Register tool after server started**
- Scenario:
  ```python
  await server.run()  # Server running
  # In another thread/context:
  server.register_tool(new_tool)
  ```
- Expected Behavior:
  - Current: Tool registered but not available (race condition)
  - Better: Raise RuntimeError("Cannot register tools while server running")
- Recommendation: Add _running check

**Related Methods:**
- Calls: `_validate_tool(tool)`
- Called by: `register_all_tools()`, user code

---

### Method 3: `register_all_tools`

**Signature:**
```python
def register_all_tools(self) -> None
```

**Purpose**: Register all standard Zapomni MCP tools in one operation

**Parameters**: None

**Returns**: None

**Raises:**

**`ValueError`**: If any tool registration fails (name conflict, invalid schema)
**`ImportError`**: If tool modules cannot be imported

**Preconditions:**
- ✅ Server initialized
- ✅ core_engine operational

**Postconditions:**
- ✅ 3+ tools registered (Phase 1: add_memory, search_memory, get_stats)
- ✅ Future phases add more tools (build_graph, get_related, etc.)
- ✅ All registrations logged

**Algorithm Outline:**
```
1. Import all tool classes from zapomni_mcp.tools:
     - AddMemoryTool
     - SearchMemoryTool
     - GetStatsTool
     (Phase 2+: BuildGraphTool, GetRelatedTool, etc.)

2. For each tool class:
     a. Instantiate: tool = ToolClass(core=_core_engine)
     b. Call: register_tool(tool)
     c. Handle errors (log and re-raise)

3. Log: "Registered {count} tools successfully"
```

**Edge Cases:**

**Edge Case 1: Tool import fails**
- Scenario: One tool module has syntax error
- Expected Behavior:
  ```python
  raise ImportError("Failed to import AddMemoryTool: <error>")
  ```
- Consequence: Server cannot start
- Mitigation: Validate all tool modules at build/test time

**Edge Case 2: Tool instantiation fails**
- Scenario: Tool constructor raises exception
- Expected Behavior:
  - Log error: "Failed to instantiate AddMemoryTool: <error>"
  - Re-raise exception
  - Server startup fails
- Example:
  ```python
  class AddMemoryTool:
      def __init__(self, core):
          if core is None:
              raise ValueError("Core required")
  ```

**Edge Case 3: Partial registration failure**
- Scenario: Tool 1 and 2 register OK, Tool 3 fails
- Current Behavior:
  - Tools 1-2 registered
  - Tool 3 raises exception
  - _tools contains 1-2 (partial state)
- Better Behavior:
  - Rollback: Clear _tools on any failure
  - All-or-nothing registration
- Recommendation: Add transaction-like behavior

**Related Methods:**
- Calls: `register_tool()` for each tool
- Called by: User code, typically in main() before run()

---

### Method 4: `run`

**Signature:**
```python
async def run(self) -> None
```

**Purpose**: Start the MCP server main loop and process requests until shutdown

**Parameters**: None

**Returns**: None (blocks until shutdown)

**Raises:**

**`RuntimeError`**
- When: Server is already running (_running = True)
- Message: "Server is already running"
- Example:
  ```python
  await server.run()  # First call OK
  await server.run()  # Second call raises RuntimeError
  ```

**`ConnectionError`**
- When: stdin or stdout unavailable
- Message: "stdin or stdout not available"
- Example: Running in environment without stdio

**Preconditions:**
- ✅ At least one tool registered
- ✅ core_engine is operational
- ✅ Server not already running

**Postconditions (on shutdown):**
- ✅ _running = False
- ✅ All requests processed
- ✅ Resources cleaned up
- ✅ Statistics logged

**Algorithm Outline:**
```
1. Check _running flag:
     If True: raise RuntimeError("Server already running")

2. Set _running = True

3. Set _start_time = time.time()

4. Log: "Starting MCP server with {len(_tools)} tools"

5. Try:
     a. Start stdio_server from MCP SDK:
          await stdio_server(
              read_stream=sys.stdin.buffer,
              write_stream=sys.stdout.buffer
          )
     b. For each incoming request:
          - Parse JSON-RPC 2.0 message
          - Extract tool name and arguments
          - Call _handle_request(request)
          - Send response to stdout
     c. Loop until stdin EOF or shutdown

6. Except Exception as e:
     Log error: "Server error: {e}"
     Raise

7. Finally:
     Call shutdown()
```

**Blocking Behavior:**
- Blocks until one of:
  - stdin receives EOF (normal)
  - SIGINT received (Ctrl+C)
  - SIGTERM received (kill)
  - Unhandled exception

**Edge Cases:**

**Edge Case 1: No tools registered**
- Scenario:
  ```python
  server = MCPServer(core=core)
  # Forgot to call register_all_tools()
  await server.run()
  ```
- Expected Behavior:
  - Option A: Raise RuntimeError("No tools registered")
  - Option B: Run but log warning, return error for all requests
- Recommendation: Option A (fail fast)

**Edge Case 2: stdin closes immediately**
- Scenario: Pipe closed before any requests
- Expected Behavior:
  - Log: "stdin closed, shutting down"
  - Clean shutdown
  - No errors
- Test:
  ```python
  async def test_run_stdin_eof():
      # Simulate stdin EOF
      with patch('sys.stdin.buffer', MagicMock()):
          await server.run()
      # Should exit cleanly
  ```

**Edge Case 3: Unhandled exception during request**
- Scenario: Tool raises unexpected exception
- Expected Behavior:
  - Catch exception in _handle_request
  - Format as MCP error response
  - Send to client
  - Log error
  - Continue processing (don't crash server)

**Edge Case 4: Calling run() twice**
- Scenario:
  ```python
  task1 = asyncio.create_task(server.run())
  task2 = asyncio.create_task(server.run())  # Concurrent
  ```
- Expected Behavior:
  - First call: OK, server runs
  - Second call: RuntimeError("Server already running")

**Performance:**
- Target overhead: < 20ms per request
- Actual time dominated by:
  - Tool execution (embedding generation, DB queries)
  - Not by MCP protocol overhead

**Related Methods:**
- Calls: `_handle_request()` for each request
- Calls: `shutdown()` on exit
- Called by: User code (main loop, async entry point)

---

### Method 5: `shutdown`

**Signature:**
```python
def shutdown(self) -> None
```

**Purpose**: Gracefully stop the server and clean up resources

**Parameters**: None

**Returns**: None

**Raises**: None (best-effort cleanup, no exceptions)

**Preconditions**: None (safe to call multiple times)

**Postconditions:**
- ✅ _running = False
- ✅ stdin closed
- ✅ stdout flushed
- ✅ Final statistics logged

**Algorithm Outline:**
```
1. If not _running:
     Log: "Shutdown called but server not running"
     Return (idempotent)

2. Set _running = False

3. Log: "Shutting down MCP server..."

4. Try:
     a. Close stdin (triggers stdio_server exit)
     b. Flush stdout (ensure all responses sent)
     c. Stop background task manager (if running)

5. Calculate uptime = time.time() - _start_time

6. Log final statistics:
     - Total requests: _request_count
     - Total errors: _error_count
     - Error rate: _error_count / _request_count (if > 0)
     - Uptime: {uptime:.1f}s

7. Log: "Server shutdown complete"
```

**Edge Cases:**

**Edge Case 1: Multiple shutdown calls**
- Scenario:
  ```python
  server.shutdown()
  server.shutdown()  # Second call
  ```
- Expected Behavior:
  - First call: Full cleanup
  - Second call: Log warning, return immediately (idempotent)

**Edge Case 2: Shutdown during request processing**
- Scenario: Signal received while handling request
- Expected Behavior:
  - Request allowed to complete (best effort)
  - Response sent
  - Then shutdown proceeds

**Edge Case 3: Shutdown with pending background tasks**
- Scenario: graph building task running when shutdown called
- Expected Behavior:
  - Log: "Cancelling X background tasks"
  - Cancel tasks gracefully
  - Wait for cancellation (timeout 5s)
  - Then proceed with shutdown

**Thread Safety:**
- **IS thread-safe** (can be called from signal handler)
- Uses atomic _running flag
- No locks needed (stdio is single-threaded)

**Related Methods:**
- Called by: `run()` (in finally block), signal handlers, user code

---

### Method 6: `get_stats`

**Signature:**
```python
def get_stats(self) -> ServerStats
```

**Purpose**: Return current server operational statistics

**Parameters**: None

**Returns:**
```python
@dataclass
class ServerStats:
    total_requests: int      # Requests processed since start
    total_errors: int        # Errors encountered
    registered_tools: int    # Number of tools registered
    uptime_seconds: float    # Time since server started
    running: bool            # Current server state
```

**Raises**: None

**Preconditions**: None

**Postconditions**: None (read-only operation)

**Algorithm Outline:**
```
1. Calculate uptime:
     if _running:
         uptime = time.time() - _start_time
     else:
         uptime = 0.0

2. Create ServerStats:
     return ServerStats(
         total_requests=_request_count,
         total_errors=_error_count,
         registered_tools=len(_tools),
         uptime_seconds=uptime,
         running=_running
     )
```

**Edge Cases:**

**Edge Case 1: Called before server started**
- Scenario:
  ```python
  server = MCPServer(core=core)
  stats = server.get_stats()
  ```
- Expected Behavior:
  ```python
  ServerStats(
      total_requests=0,
      total_errors=0,
      registered_tools=0,  # Or however many registered
      uptime_seconds=0.0,
      running=False
  )
  ```

**Edge Case 2: Called after shutdown**
- Scenario:
  ```python
  await server.run()
  # Server shuts down
  stats = server.get_stats()
  ```
- Expected Behavior:
  ```python
  ServerStats(
      total_requests=42,  # Actual count
      total_errors=3,
      registered_tools=3,
      uptime_seconds=0.0,  # Not running
      running=False
  )
  ```

**Time Complexity**: O(1) - all values are pre-computed

**Use Cases:**
- Health checks
- Monitoring dashboards
- Debugging
- Performance analysis

**Related Methods:** None (standalone query)

---

## Error Handling

### Exceptions Defined

**`MCPServerError` (Base Exception)**
```python
class MCPServerError(Exception):
    """Base exception for all MCP server errors."""
    pass
```

**`ConfigurationError` (Inherits MCPServerError)**
```python
class ConfigurationError(MCPServerError):
    """Raised when server configuration is invalid."""
    pass
```
- When Raised:
  - core_engine is None
  - config has invalid values
  - Required environment variables missing

**`ToolRegistrationError` (Inherits MCPServerError)**
```python
class ToolRegistrationError(MCPServerError):
    """Raised when tool registration fails."""
    pass
```
- When Raised:
  - Duplicate tool name
  - Tool doesn't implement MCPTool protocol
  - Invalid input_schema

**`RequestHandlingError` (Inherits MCPServerError)**
```python
class RequestHandlingError(MCPServerError):
    """Raised when request processing fails."""
    pass
```
- When Raised:
  - Invalid JSON-RPC message
  - Unknown tool name
  - Tool execution exception

### Error Recovery

**Retry Strategy:**
- **No automatic retries** at server level
- MCP client responsible for retries
- Tools may implement retry logic internally

**Fallback Behavior:**
- Server never crashes on single request error
- Each error logged, response sent, server continues
- Only fatal errors (stdin closed, SIGKILL) stop server

**Error Propagation:**
- Tool exceptions → Caught in _handle_request
- Formatted as MCP error response
- Sent to client with error code
- Server continues processing

**Error Codes (JSON-RPC 2.0):**
```python
ERROR_CODES = {
    -32700: "Parse error",        # Invalid JSON
    -32600: "Invalid request",    # Not valid JSON-RPC
    -32601: "Method not found",   # Tool not registered
    -32602: "Invalid params",     # Validation failed
    -32603: "Internal error",     # Unexpected exception
    -32000: "Server error",       # Custom server errors
}
```

### Example Error Handling

**Client Request with Unknown Tool:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "unknown_tool",
    "arguments": {}
  }
}
```

**Server Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32601,
    "message": "Tool not found: unknown_tool",
    "data": {
      "available_tools": ["add_memory", "search_memory", "get_stats"]
    }
  }
}
```

---

## Usage Examples

### Basic Usage

```python
import asyncio
from zapomni_core import ZapomniCore, CoreConfig
from zapomni_mcp.server import MCPServer
from zapomni_mcp.config import Settings

async def main():
    # Initialize core engine
    core_config = CoreConfig(
        falkordb_host="localhost",
        falkordb_port=6379,
        ollama_host="http://localhost:11434"
    )
    core = ZapomniCore(config=core_config)

    # Create MCP server
    server_config = Settings(log_level="INFO")
    server = MCPServer(core_engine=core, config=server_config)

    # Register tools
    server.register_all_tools()

    # Start server (blocks until shutdown)
    await server.run()

if __name__ == "__main__":
    asyncio.run(main())
```

### Advanced Usage (Custom Tool)

```python
from zapomni_mcp.server import MCPServer
from zapomni_mcp.tools import MCPTool

class CustomTool:
    """Custom MCP tool example."""

    name = "custom_operation"
    description = "Perform a custom operation"
    input_schema = {
        "type": "object",
        "properties": {
            "data": {"type": "string"}
        },
        "required": ["data"]
    }

    def __init__(self, core: ZapomniCore):
        self.core = core

    async def execute(self, arguments: dict) -> dict:
        """Execute custom logic."""
        data = arguments["data"]
        # Custom processing
        result = f"Processed: {data}"

        return {
            "content": [
                {"type": "text", "text": result}
            ],
            "isError": False
        }

# Usage
server = MCPServer(core_engine=core)
server.register_all_tools()  # Standard tools
server.register_tool(CustomTool(core=core))  # Custom tool
await server.run()
```

### Monitoring Example

```python
import asyncio
from zapomni_mcp.server import MCPServer

async def monitor_server(server: MCPServer):
    """Monitor server statistics."""
    while True:
        await asyncio.sleep(60)  # Every minute

        stats = server.get_stats()
        print(f"Server Stats:")
        print(f"  Requests: {stats.total_requests}")
        print(f"  Errors: {stats.total_errors}")
        print(f"  Error Rate: {stats.total_errors / stats.total_requests:.2%}")
        print(f"  Uptime: {stats.uptime_seconds:.1f}s")
        print(f"  Running: {stats.running}")

# Start monitoring task
asyncio.create_task(monitor_server(server))
await server.run()
```

---

## Testing Approach

### Unit Tests Required

**Initialization Tests:**
1. `test_init_success()` - Normal initialization with valid core
2. `test_init_with_config()` - Custom config provided
3. `test_init_defaults()` - No config, uses defaults
4. `test_init_none_core_raises()` - None core raises ConfigurationError
5. `test_init_invalid_config_raises()` - Invalid config raises ValidationError

**Tool Registration Tests:**
6. `test_register_tool_success()` - Register valid tool
7. `test_register_tool_duplicate_raises()` - Duplicate name raises ValueError
8. `test_register_tool_invalid_protocol_raises()` - Missing attributes raises TypeError
9. `test_register_tool_invalid_schema_raises()` - Bad input_schema raises ValidationError
10. `test_register_all_tools_success()` - All standard tools registered
11. `test_register_all_tools_count()` - Correct number of tools

**Server Lifecycle Tests:**
12. `test_run_starts_server()` - run() starts successfully
13. `test_run_already_running_raises()` - Second run() raises RuntimeError
14. `test_run_stdin_eof_exits()` - stdin EOF triggers clean shutdown
15. `test_shutdown_stops_server()` - shutdown() sets _running = False
16. `test_shutdown_idempotent()` - Multiple shutdown() calls safe

**Request Handling Tests:**
17. `test_handle_valid_request()` - Valid request processed
18. `test_handle_unknown_tool_error()` - Unknown tool returns error
19. `test_handle_invalid_arguments_error()` - Bad args return validation error
20. `test_handle_tool_exception_error()` - Tool exception caught and formatted

**Statistics Tests:**
21. `test_get_stats_before_start()` - Stats before run()
22. `test_get_stats_after_requests()` - Stats reflect requests
23. `test_get_stats_error_count()` - Errors counted correctly
24. `test_get_stats_uptime()` - Uptime calculated correctly

**Signal Handling Tests:**
25. `test_sigint_triggers_shutdown()` - SIGINT calls shutdown()
26. `test_sigterm_triggers_shutdown()` - SIGTERM calls shutdown()

### Mocking Strategy

**Mock ZapomniCore:**
```python
@pytest.fixture
def mock_core():
    """Mock ZapomniCore for testing."""
    core = MagicMock(spec=ZapomniCore)
    core.add_memory = AsyncMock(return_value=MemoryResult(
        memory_id="test-id",
        chunks_created=1
    ))
    core.search_memory = AsyncMock(return_value=[])
    core.get_stats = AsyncMock(return_value=Statistics(
        total_memories=0,
        total_chunks=0
    ))
    return core
```

**Mock Tools:**
```python
@pytest.fixture
def mock_tool():
    """Mock MCPTool for testing."""
    tool = MagicMock(spec=MCPTool)
    tool.name = "test_tool"
    tool.description = "Test tool"
    tool.input_schema = {"type": "object"}
    tool.execute = AsyncMock(return_value={
        "content": [{"type": "text", "text": "Success"}],
        "isError": False
    })
    return tool
```

**Mock stdin/stdout:**
```python
@pytest.fixture
def mock_stdio():
    """Mock stdin/stdout for testing."""
    stdin = MagicMock()
    stdout = MagicMock()
    with patch('sys.stdin.buffer', stdin), \
         patch('sys.stdout.buffer', stdout):
        yield stdin, stdout
```

### Integration Tests

**Test with Real MCP SDK:**
```python
@pytest.mark.integration
async def test_full_request_cycle(mock_core):
    """Integration test: full request processing."""
    server = MCPServer(core_engine=mock_core)
    server.register_all_tools()

    # Simulate MCP request
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "get_stats",
            "arguments": {}
        }
    }

    # Process request (internal method test)
    response = await server._handle_request(request)

    assert response["id"] == 1
    assert "result" in response
    assert response["result"]["content"]
```

---

## Performance Considerations

### Time Complexity

- `__init__`: O(1) - constant initialization
- `register_tool`: O(1) - dict insertion
- `register_all_tools`: O(n) where n = number of tools
- `run`: O(m) where m = number of requests (blocks indefinitely)
- `shutdown`: O(1) - cleanup operations
- `get_stats`: O(1) - attribute access

### Space Complexity

- Server overhead: ~10-20MB (MCP SDK + logging + tool registry)
- Per-tool overhead: ~1-2KB (tool metadata)
- Per-request overhead: ~5-10KB (parsing, logging)

### Optimization Opportunities

**Current Performance:**
- MCP overhead: < 20ms per request (target met)
- Bottleneck: Tool execution (embeddings, DB queries), NOT server

**Future Optimizations:**
1. **Request caching**: Cache identical requests (semantic similarity)
2. **Tool preloading**: Lazy-load tools only when first called
3. **Async batching**: If HTTP transport added, batch multiple requests
4. **Connection pooling**: Reuse DB connections across requests

**Trade-offs:**
- **Simplicity vs Speed**: Current design favors simplicity (stdio is inherently simple)
- **Memory vs Latency**: Could cache more aggressively but increases memory usage
- **Flexibility vs Performance**: Generic tool interface vs hardcoded fast paths

---

## References

### Module Spec
- [zapomni_mcp_module.md](/home/dev/zapomni/.spec-workflow/specs/level1/zapomni_mcp_module.md) - Parent module specification

### Related Components
- AddMemoryTool (level 2 component spec - to be created)
- SearchMemoryTool (level 2 component spec - to be created)
- GetStatsTool (level 2 component spec - to be created)

### External Documentation
- [MCP Specification](https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/architecture/)
- [JSON-RPC 2.0](https://www.jsonrpc.org/specification)
- [MCP Python SDK](https://github.com/anthropics/anthropic-mcp-python)
- [asyncio Documentation](https://docs.python.org/3/library/asyncio.html)
- [structlog Documentation](https://www.structlog.org/)

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
4. Proceed to Level 3 (Function specs for each public method)

---

**Document Length:** ~1200 lines
**Estimated Reading Time:** 40-50 minutes
**Target Audience:** Developers implementing MCPServer class
