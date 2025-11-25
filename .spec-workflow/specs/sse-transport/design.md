# Design Document - SSE Transport for Concurrent Connections

**Spec Name:** sse-transport
**Status:** Design Phase
**Created:** 2025-11-25
**Author:** Specification Writer Agent

---

## 1. System Architecture

### 1.1 High-Level Architecture

```
                                   +-----------------------------------------+
                                   |          Zapomni MCP Server             |
                                   |                                         |
   +----------------+              |  +----------------------------------+   |
   |  Claude        | GET /sse    |  |        Starlette App              |   |
   |  Desktop       |------------>|  |  +----------------------------+   |   |
   |  (Client 1)    |<------------|  |  |  handle_sse()              |   |   |
   +----------------+  SSE Stream |  |  |  - Create SseServerTransport|   |   |
                                   |  |  |  - Track session           |   |   |
   +----------------+              |  |  |  - Run mcp_server.run()    |   |   |
   |  VS Code       | GET /sse    |  |  +----------------------------+   |   |
   |  Extension     |------------>|  |                                   |   |
   |  (Client 2)    |<------------|  |  +----------------------------+   |   |
   +----------------+  SSE Stream |  |  |  handle_messages()         |   |   |
                                   |  |  |  - Validate session_id     |   |   |
   +----------------+              |  |  |  - Forward to transport    |   |   |
   |  CLI Tool      | POST        |  |  |  - Return 202 Accepted     |   |   |
   |  (Client 3)    |------------>|  |  +----------------------------+   |   |
   +----------------+  /messages/ |  |                                   |   |
                                   |  +----------------------------------+   |
                                   |                  |                      |
                                   |                  v                      |
                                   |  +----------------------------------+   |
                                   |  |      MCPServer (Singleton)       |   |
                                   |  |  - _server: mcp.server.Server    |   |
                                   |  |  - _tools: Dict[str, MCPTool]    |   |
                                   |  |  - Sessions share this instance  |   |
                                   |  +----------------------------------+   |
                                   |                  |                      |
                                   |                  v                      |
                                   |  +----------------------------------+   |
                                   |  |      MemoryProcessor             |   |
                                   |  |  - db_client (FalkorDB)          |   |
                                   |  |  - chunker (SemanticChunker)     |   |
                                   |  |  - embedder (OllamaEmbedder)     |   |
                                   |  |  - extractor (EntityExtractor)   |   |
                                   |  +----------------------------------+   |
                                   |                  |                      |
                                   |                  v                      |
                                   |  +----------------------------------+   |
                                   |  |      FalkorDBClient              |   |
                                   |  |  - Connection Pool (20 default)  |   |
                                   |  |  - Thread-safe operations        |   |
                                   |  +----------------------------------+   |
                                   +-----------------------------------------+
```

### 1.2 Transport Layer Architecture

```
+------------------------------------------------------------------+
|                    Transport Layer (SSE)                         |
+------------------------------------------------------------------+
|                                                                  |
|   +-----------------+     +-----------------+     +------------+ |
|   |   SSE Endpoint  |     | Messages        |     |   CORS     | |
|   |   GET /sse      |     | POST /messages/ |     | Middleware | |
|   +-----------------+     +-----------------+     +------------+ |
|           |                       |                      |       |
|           v                       v                      v       |
|   +---------------------------------------------------+         |
|   |              Session Manager                       |         |
|   |   sessions: Dict[str, SseServerTransport]          |         |
|   |   - create_session(session_id, transport)          |         |
|   |   - get_session(session_id) -> transport           |         |
|   |   - delete_session(session_id)                     |         |
|   |   - cleanup_stale_sessions()                       |         |
|   +---------------------------------------------------+         |
|                                                                  |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                    MCP Protocol Layer                            |
+------------------------------------------------------------------+
|                                                                  |
|   +----------------------------+   +-------------------------+   |
|   |   SseServerTransport       |   |   mcp.server.Server     |   |
|   |   (per-session)            |   |   (singleton)           |   |
|   |   - connect_sse()          |   |   - list_tools()        |   |
|   |   - handle_post_message()  |   |   - call_tool()         |   |
|   |   - Memory streams         |   |   - run()               |   |
|   +----------------------------+   +-------------------------+   |
|                                                                  |
+------------------------------------------------------------------+
```

### 1.3 Concurrency Model

```
+------------------------------------------------------------------+
|                   Single AsyncIO Event Loop                      |
+------------------------------------------------------------------+
|                                                                  |
|   +-------------+  +-------------+  +-------------+              |
|   | Session 1   |  | Session 2   |  | Session 3   |              |
|   | (Client A)  |  | (Client B)  |  | (Client C)  |              |
|   +------+------+  +------+------+  +------+------+              |
|          |                |                |                     |
|          v                v                v                     |
|   +---------------------------------------------------+         |
|   |         MCPServer (Shared Instance)               |         |
|   |         - Tool handlers (async)                   |         |
|   |         - Concurrent request processing           |         |
|   +---------------------------------------------------+         |
|                           |                                     |
|          +----------------+----------------+                     |
|          |                |                |                     |
|          v                v                v                     |
|   +-----------+    +-----------+    +-----------+               |
|   | I/O-bound |    | I/O-bound |    | CPU-bound |               |
|   | (DB query)|    | (Embed)   |    | (SpaCy)   |               |
|   |  async    |    |  async    |    | Executor  |               |
|   +-----------+    +-----------+    +-----------+               |
|                                            |                     |
|                                            v                     |
|                                    +---------------+            |
|                                    | ThreadPool    |            |
|                                    | (5 workers)   |            |
|                                    +---------------+            |
+------------------------------------------------------------------+
```

---

## 2. Component Design

### 2.1 SSE Transport Module

**File:** `src/zapomni_mcp/sse_transport.py`

```python
"""
SSE Transport implementation for Zapomni MCP Server.

Provides HTTP endpoints for SSE-based MCP communication:
- GET /sse: Establish SSE connection
- POST /messages/{session_id}: Receive JSON-RPC messages
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
from uuid import uuid4
import asyncio

from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.requests import Request
from starlette.responses import Response
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from mcp.server.sse import SseServerTransport

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class SSEConfig:
    """Configuration for SSE transport."""
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    heartbeat_interval: int = 30  # seconds
    max_connection_lifetime: int = 3600  # 1 hour


class SessionManager:
    """Manages SSE session lifecycle."""

    def __init__(self):
        self._sessions: Dict[str, SseServerTransport] = {}
        self._session_metadata: Dict[str, dict] = {}
        self._lock = asyncio.Lock()

    async def create_session(self, transport: SseServerTransport) -> str:
        """Create new session with unique ID."""
        session_id = str(uuid4())
        async with self._lock:
            self._sessions[session_id] = transport
            self._session_metadata[session_id] = {
                "created_at": asyncio.get_event_loop().time(),
                "last_activity": asyncio.get_event_loop().time(),
            }
        return session_id

    def get_session(self, session_id: str) -> Optional[SseServerTransport]:
        """Get session by ID."""
        return self._sessions.get(session_id)

    async def delete_session(self, session_id: str) -> None:
        """Remove session and cleanup resources."""
        async with self._lock:
            self._sessions.pop(session_id, None)
            self._session_metadata.pop(session_id, None)

    @property
    def active_session_count(self) -> int:
        """Get number of active sessions."""
        return len(self._sessions)


def create_sse_app(mcp_server, config: SSEConfig) -> Starlette:
    """
    Create Starlette application with SSE routes.

    Args:
        mcp_server: MCPServer instance (singleton)
        config: SSE configuration

    Returns:
        Configured Starlette application
    """
    session_manager = SessionManager()

    async def handle_sse(request: Request) -> Response:
        """Handle SSE connection establishment."""
        # Implementation details in tasks
        pass

    async def handle_messages(request: Request) -> Response:
        """Handle POST messages to session."""
        session_id = request.path_params["session_id"]
        transport = session_manager.get_session(session_id)
        if not transport:
            return Response(
                content='{"error": "Session not found"}',
                status_code=404,
                media_type="application/json"
            )
        body = await request.body()
        # Forward to transport for processing
        await transport.handle_post_message(scope=request.scope, body=body)
        return Response(
            content='{"status": "accepted"}',
            status_code=202,
            media_type="application/json"
        )

    # CORS middleware configuration
    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=config.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
            expose_headers=["*"],
        )
    ]

    routes = [
        Route("/sse", endpoint=handle_sse, methods=["GET"]),
        Route("/messages/{session_id}", endpoint=handle_messages, methods=["POST"]),
    ]

    return Starlette(routes=routes, middleware=middleware)
```

### 2.2 MCPServer SSE Extension

**File:** `src/zapomni_mcp/server.py` (additions)

```python
# New method to add to MCPServer class

async def run_sse(
    self,
    host: str = "127.0.0.1",
    port: int = 8000,
    cors_origins: list[str] = None,
) -> None:
    """
    Start the MCP server with SSE transport.

    This enables multiple concurrent client connections via HTTP.

    Args:
        host: Bind address (default: 127.0.0.1 for local only)
        port: HTTP port (default: 8000)
        cors_origins: Allowed CORS origins (default: ["*"])

    Raises:
        RuntimeError: If server is already running
        OSError: If port is already in use
    """
    import uvicorn
    from zapomni_mcp.sse_transport import create_sse_app, SSEConfig

    if self._running:
        raise RuntimeError("Server is already running")

    config = SSEConfig(
        host=host,
        port=port,
        cors_origins=cors_origins or ["*"],
    )

    app = create_sse_app(mcp_server=self, config=config)

    self._running = True
    self._start_time = time.time()

    self._logger.info(
        "Starting SSE server",
        host=host,
        port=port,
        tool_count=len(self._tools),
    )

    uvicorn_config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
    )
    server = uvicorn.Server(uvicorn_config)

    try:
        await server.serve()
    finally:
        self.shutdown()
```

### 2.3 EntityExtractor Async Wrapper

**File:** `src/zapomni_core/extractors/entity_extractor.py` (modifications)

```python
# Additions to EntityExtractor class

from concurrent.futures import ThreadPoolExecutor

class EntityExtractor:
    """Hybrid entity extraction with async support."""

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        confidence_threshold: float = 0.7,
        llm_client: Optional["OllamaLLMClient"] = None,
        executor_workers: int = 5,  # NEW parameter
    ):
        # ... existing init code ...

        # NEW: Thread pool for CPU-bound operations
        self._executor = ThreadPoolExecutor(
            max_workers=executor_workers,
            thread_name_prefix="entity_extractor_",
        )

    async def extract_entities(self, text: str) -> List[Entity]:
        """
        Async entity extraction with executor for CPU-bound work.

        Args:
            text: Input text for entity extraction

        Returns:
            List of extracted Entity objects
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._extract_entities_sync,
            text,
        )

    def _extract_entities_sync(self, text: str) -> List[Entity]:
        """
        Synchronous entity extraction (runs in executor).

        This is the original extract_entities() logic, renamed.
        """
        # ... existing extraction logic ...
        pass

    def close(self) -> None:
        """Cleanup resources including thread pool."""
        self._executor.shutdown(wait=True)
```

### 2.4 Configuration Extensions

**File:** `src/zapomni_core/config.py` (additions)

```python
class ZapomniSettings(BaseSettings):
    # ... existing fields ...

    # ========================================
    # SSE TRANSPORT CONFIGURATION
    # ========================================

    sse_host: str = Field(
        default="127.0.0.1",
        description="SSE server bind address"
    )

    sse_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="SSE server port"
    )

    sse_cors_origins: str = Field(
        default="*",
        description="Comma-separated CORS origins (* for all)"
    )

    sse_heartbeat_interval: int = Field(
        default=30,
        ge=5,
        le=300,
        description="SSE heartbeat interval in seconds"
    )

    sse_max_connection_lifetime: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="Maximum SSE connection lifetime in seconds"
    )

    # ========================================
    # EXECUTOR CONFIGURATION
    # ========================================

    entity_extractor_workers: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Thread pool workers for entity extraction"
    )
```

**File:** `src/zapomni_mcp/config.py` (additions)

```python
@dataclass
class Settings:
    # ... existing fields ...

    # SSE transport settings
    sse_host: str = "127.0.0.1"
    sse_port: int = 8000
    cors_origins: str = "*"
```

---

## 3. API Contracts

### 3.1 SSE Endpoint

**Endpoint:** `GET /sse`

**Description:** Establish SSE connection for MCP communication.

**Request:**
```http
GET /sse HTTP/1.1
Host: localhost:8000
Accept: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
```

**Response:**
```http
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
X-Accel-Buffering: no

event: endpoint
data: /messages/a1b2c3d4-e5f6-7890-abcd-ef1234567890

event: message
data: {"jsonrpc":"2.0","id":1,"result":{"tools":[...]}}
```

**Error Responses:**
| Status | Reason |
|--------|--------|
| 429 | Too many connections |
| 503 | Server overloaded |

### 3.2 Messages Endpoint

**Endpoint:** `POST /messages/{session_id}`

**Description:** Send JSON-RPC message to MCP server via session.

**Request:**
```http
POST /messages/a1b2c3d4-e5f6-7890-abcd-ef1234567890 HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "jsonrpc": "2.0",
  "method": "tools/list",
  "id": 1
}
```

**Response:**
```http
HTTP/1.1 202 Accepted
Content-Type: application/json

{"status": "accepted"}
```

**Error Responses:**
| Status | Reason | Body |
|--------|--------|------|
| 400 | Invalid JSON-RPC | `{"error": "Invalid JSON-RPC format"}` |
| 404 | Session not found | `{"error": "Session not found"}` |
| 500 | Internal error | `{"error": "Internal server error"}` |

### 3.3 Health Endpoint (Required)

**Endpoint:** `GET /health`

**Description:** Health check for monitoring and service verification.

**Request:**
```http
GET /health HTTP/1.1
Host: localhost:8000
```

**Response:**
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "status": "healthy",
  "active_sessions": 5,
  "uptime_seconds": 3600,
  "version": "0.1.0"
}
```

**Error Responses:**
| Status | Reason |
|--------|--------|
| 503 | Server unhealthy (database unavailable, etc.) |

---

## 4. Sequence Diagrams

### 4.1 SSE Connection Establishment

```
Client                    Starlette App              SessionManager           MCPServer
   |                           |                           |                      |
   |----GET /sse-------------->|                           |                      |
   |                           |                           |                      |
   |                           |--create_session()-------->|                      |
   |                           |                           |                      |
   |                           |<--session_id--------------|                      |
   |                           |                           |                      |
   |                           |--SseServerTransport-------|--------------------->|
   |                           |   .connect_sse()          |                      |
   |                           |                           |                      |
   |<---event: endpoint--------|                           |                      |
   |    data: /messages/{id}   |                           |                      |
   |                           |                           |                      |
   |                           |---------------------------|---mcp_server.run()-->|
   |                           |                           |   (streams)          |
   |                           |                           |                      |
   |<---event: message---------|<--------------------------|<--list_tools()-------|
   |    data: {tools:[...]}    |                           |                      |
   |                           |                           |                      |
   |          [Connection open for server-to-client messages]                    |
   |                           |                           |                      |
```

### 4.2 Tool Execution Flow

```
Client                    Starlette App              SseServerTransport        MCPServer
   |                           |                           |                      |
   |--POST /messages/{id}----->|                           |                      |
   |  {"method":"tools/call",  |                           |                      |
   |   "params":{"name":"..."}}|                           |                      |
   |                           |                           |                      |
   |                           |--get_session(id)--------->|                      |
   |                           |                           |                      |
   |                           |<--transport---------------|                      |
   |                           |                           |                      |
   |                           |--handle_post_message()--->|                      |
   |                           |                           |                      |
   |<--202 Accepted------------|                           |                      |
   |                           |                           |                      |
   |                           |                           |--call_tool()-------->|
   |                           |                           |                      |
   |                           |                           |                      |--[Tool Logic]
   |                           |                           |                      |  EntityExtractor
   |                           |                           |                      |  FalkorDB
   |                           |                           |                      |
   |                           |                           |<--result-------------|
   |                           |                           |                      |
   |<---event: message---------|<--------------------------|                      |
   |    data: {result:{...}}   |                           |                      |
   |                           |                           |                      |
```

### 4.3 Concurrent Request Handling

```
Client A                  Client B                  Event Loop               ThreadPool
   |                         |                           |                      |
   |--build_graph----------->|                           |                      |
   |                         |                           |                      |
   |                         |                           |--run_in_executor()-->|
   |                         |                           |   (SpaCy NLP)        |
   |                         |                           |                      |--[CPU Work]
   |                         |                           |                      |
   |                         |--get_stats--------------->|                      |
   |                         |                           |                      |
   |                         |<--result (immediate)------|                      |
   |                         |                           |                      |
   |                         |  [Event loop unblocked]   |                      |
   |                         |                           |                      |
   |                         |                           |<--result-------------|
   |                         |                           |                      |
   |<--result----------------|                           |                      |
   |                         |                           |                      |
```

### 4.4 Session Cleanup Flow

```
Client                    Starlette App              SessionManager           MCPServer
   |                           |                           |                      |
   |          [Connection closes / timeout / error]        |                      |
   |                           |                           |                      |
   |----X Connection Lost X--->|                           |                      |
   |                           |                           |                      |
   |                           |  finally: block triggers  |                      |
   |                           |                           |                      |
   |                           |--delete_session(id)------>|                      |
   |                           |                           |                      |
   |                           |                           |--[Remove from dict]  |
   |                           |                           |                      |
   |                           |  Log: "Session closed"    |                      |
   |                           |                           |                      |
```

---

## 5. Data Models

### 5.1 Session State

```python
@dataclass
class SessionState:
    """State for a single SSE session."""
    session_id: str
    transport: SseServerTransport
    created_at: float  # monotonic time
    last_activity: float
    client_ip: str
    request_count: int = 0
    error_count: int = 0
```

### 5.2 SSE Metrics

```python
@dataclass
class SSEMetrics:
    """Metrics for SSE transport."""
    active_connections: int
    total_connections: int
    total_messages: int
    total_errors: int
    avg_session_duration: float
    uptime_seconds: float
```

### 5.3 Configuration Model

```python
@dataclass
class SSEConfig:
    """Configuration for SSE transport."""
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    heartbeat_interval: int = 30
    max_connection_lifetime: int = 3600
    max_concurrent_connections: int = 100
```

---

## 6. Error Handling Strategy

### 6.1 Error Categories

| Category | Examples | Handling |
|----------|----------|----------|
| Connection Errors | Client disconnect, timeout | Log, cleanup session, continue |
| Protocol Errors | Invalid JSON-RPC, unknown method | Return 400, log warning |
| Session Errors | Invalid session ID, expired session | Return 404, log info |
| Tool Errors | DB error, extraction error | Return error via SSE, log error |
| System Errors | Out of memory, port in use | Log critical, graceful shutdown |

### 6.2 Error Response Format

```python
# SSE error event
event: error
data: {"code": -32603, "message": "Internal error", "data": {"details": "..."}}

# HTTP error response
{
    "error": {
        "code": -32600,
        "message": "Invalid Request",
        "data": {"field": "session_id", "reason": "not found"}
    }
}
```

### 6.3 Retry Logic

```python
class RetryConfig:
    """Retry configuration for database operations."""
    max_retries: int = 3
    initial_delay: float = 0.1
    max_delay: float = 2.0
    exponential_base: float = 2.0

async def with_retry(operation, config: RetryConfig):
    """Execute operation with exponential backoff retry."""
    delay = config.initial_delay
    for attempt in range(config.max_retries):
        try:
            return await operation()
        except (ConnectionError, TimeoutError) as e:
            if attempt == config.max_retries - 1:
                raise
            await asyncio.sleep(delay)
            delay = min(delay * config.exponential_base, config.max_delay)
```

---

## 7. Security Considerations

### 7.1 DNS Rebinding Protection

```python
from mcp.server.transport_security import TransportSecuritySettings

security_settings = TransportSecuritySettings(
    allowed_hosts=["localhost", "127.0.0.1"],
    validate_origin=True,
)

sse_transport = SseServerTransport(
    endpoint="/messages/",
    security_settings=security_settings,
)
```

### 7.2 Session Security

```python
import secrets

def generate_session_id() -> str:
    """Generate cryptographically secure session ID."""
    return secrets.token_urlsafe(32)  # 256-bit entropy
```

### 7.3 Local Deployment Security Note

Authentication and rate limiting are explicitly out of scope for this implementation.
The server is designed for local deployment only, binding to 127.0.0.1 by default.
For production deployments with remote access, additional security measures would be required.

---

## 8. Configuration Schema

### 8.1 Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ZAPOMNI_TRANSPORT` | string | `sse` | Transport protocol (sse, stdio) |
| `ZAPOMNI_SSE_HOST` | string | `127.0.0.1` | SSE server bind address |
| `ZAPOMNI_SSE_PORT` | int | `8000` | SSE server port |
| `ZAPOMNI_SSE_CORS_ORIGINS` | string | `*` | Comma-separated CORS origins |
| `FALKORDB_POOL_SIZE` | int | `20` | Database connection pool size |
| `ENTITY_EXTRACTOR_WORKERS` | int | `5` | Thread pool workers |

### 8.2 CLI Arguments

```
usage: python -m zapomni_mcp [-h] [--transport {stdio,sse}]
                             [--host HOST] [--port PORT]

Zapomni MCP Server

optional arguments:
  -h, --help            show this help message and exit
  --transport {stdio,sse}
                        Transport protocol (default: sse)
  --host HOST           SSE server host (default: 127.0.0.1)
  --port PORT           SSE server port (default: 8000)
```

### 8.3 Configuration Priority

1. CLI arguments (highest)
2. Environment variables
3. .env file
4. Default values (lowest)

---

## 9. Testing Strategy

### 9.1 Unit Tests

| Component | Test Focus |
|-----------|-----------|
| SessionManager | Session CRUD, cleanup, concurrency |
| SSEConfig | Validation, defaults, parsing |
| handle_sse | SSE event format, error handling |
| handle_messages | Session lookup, error responses |

### 9.2 Integration Tests

| Scenario | Verification |
|----------|-------------|
| Single client lifecycle | Connect, call tools, disconnect |
| Concurrent clients | Multiple connections, parallel requests |
| Tool execution | All tools work via SSE |
| Error propagation | Errors returned correctly via SSE |

### 9.3 Load Tests

| Metric | Target | Tool |
|--------|--------|------|
| 50 concurrent clients | Stable for 30 min | locust |
| Memory usage | < 500MB | psutil |
| Latency P95 | < 500ms | locust |
| Error rate | < 0.1% | locust |

---

## 10. File Structure

```
src/zapomni_mcp/
    __init__.py
    __main__.py           # MODIFY: Add --transport argument
    server.py             # MODIFY: Add run_sse() method
    config.py             # MODIFY: Add SSE settings
    sse_transport.py      # NEW: SSE transport implementation
    session_manager.py    # NEW: Session lifecycle management

src/zapomni_core/
    config.py             # MODIFY: Add SSE settings
    extractors/
        entity_extractor.py  # MODIFY: Add async wrapper

tests/
    integration/
        test_sse_transport.py  # NEW: SSE integration tests
    load/
        locustfile.py         # NEW: Load testing

pyproject.toml            # MODIFY: Add dependencies
```

---

**End of Design Document**
