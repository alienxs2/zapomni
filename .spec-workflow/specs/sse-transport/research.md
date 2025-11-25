# SSE Transport Research for Zapomni MCP Server

**Research Date:** 2025-11-25
**Investigator:** Research Agent
**Project:** Zapomni MCP Server Migration to SSE Transport

---

## Executive Summary

### Key Findings

1. **MCP Protocol Evolution**: As of March 2025, MCP has deprecated SSE in favor of Streamable HTTP as the primary transport for remote servers. However, SSE remains widely supported during the transition period.

2. **Dual Transport Strategy**: For production-ready MCP servers in 2025, implementing **both SSE and Streamable HTTP** is recommended to ensure compatibility with the widest range of clients during the transition period.

3. **Concurrency Model**: SSE transport enables true concurrent client connections through asyncio event loops, session management via UUID-based session tracking, and non-blocking I/O operations - addressing all current stdio transport limitations.

4. **Architecture Pattern**: The MCP SDK provides `SseServerTransport` which requires two endpoints:
   - `GET /sse` - Establishes SSE connection and provides session endpoint
   - `POST /messages/` - Receives client messages linked to SSE session

5. **Critical Considerations**:
   - FalkorDB connection pool thread safety needs verification for concurrent access
   - Memory management for long-lived connections requires proper cleanup
   - CORS configuration essential for local development
   - Heartbeat/keep-alive mechanisms prevent connection timeouts

---

## MCP SSE Protocol

### Protocol Specification

The MCP SSE transport implements a bidirectional communication pattern using Server-Sent Events for server-to-client messages and HTTP POST for client-to-server messages.

#### Connection Flow

1. **Client Initiates Connection**
   - Client sends `GET /sse` request
   - Server creates unique session ID (UUID)
   - Server sends `endpoint` event via SSE containing POST URL with session ID
   - SSE connection remains open for server-to-client messages

2. **Client Sends Messages**
   - Client POSTs JSON-RPC messages to `/messages/?session_id={uuid}`
   - Server validates session ID
   - Server processes message and sends response via SSE
   - Server returns `202 Accepted` for POST request

3. **Connection Lifecycle**
   - Server maintains session state in memory: `_read_stream_writers: dict[UUID, MemoryObjectSendStream]`
   - Client disconnection detected via SSE stream closure
   - Server cleans up session resources on disconnect

#### Message Format

**SSE Events:**
```
event: endpoint
data: /messages/?session_id=abc123

event: message
data: {"jsonrpc":"2.0","id":1,"result":{...}}
```

**POST Messages:**
```json
{
  "jsonrpc": "2.0",
  "method": "tools/list",
  "id": 1
}
```

### MCP SDK Implementation (from `mcp.server.sse`)

The official MCP Python SDK provides `SseServerTransport` with the following characteristics:

```python
class SseServerTransport:
    def __init__(self, endpoint: str, security_settings: TransportSecuritySettings | None = None):
        # endpoint: relative path for POST messages (e.g., "/messages/")
        # security_settings: DNS rebinding protection

    @asynccontextmanager
    async def connect_sse(self, scope: Scope, receive: Receive, send: Send):
        # Creates bidirectional memory streams
        # Returns (read_stream, write_stream) tuple
        # Handles session creation and SSE event streaming

    async def handle_post_message(self, scope: Scope, receive: Receive, send: Send):
        # Validates session_id query parameter
        # Parses JSON-RPC message
        # Forwards to appropriate session's read stream
```

**Key Features:**
- Automatic session management with UUID tracking
- Memory-based message queues using `anyio.create_memory_object_stream()`
- Built-in DNS rebinding protection via `TransportSecurityMiddleware`
- Automatic client disconnect detection
- Graceful cleanup on connection closure

### Transport Security

The SDK includes `TransportSecuritySettings` for DNS rebinding protection:
- Validates `Host` header matches expected values
- Checks `Origin` and `Referer` headers for POST requests
- Returns 403 Forbidden for suspicious requests
- Essential for production deployments

---

## Implementation Patterns

### Recommended Architecture: Singleton Server Pattern

Based on [yigitkonur/example-mcp-server-sse](https://github.com/yigitkonur/example-mcp-server-sse):

```python
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.responses import Response
import uvicorn

# Create single MCP server instance (shared across all sessions)
mcp_server = Server("zapomni")

# Session management
sse_transports: dict[str, SseServerTransport] = {}

@mcp_server.list_tools()
async def list_tools():
    # Business logic here (shared by all clients)
    return [...]

@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict):
    # Tool execution (handles concurrent requests)
    return await execute_tool(name, arguments)

async def handle_sse(request):
    # Create new SSE transport for this connection
    session_id = str(uuid4())
    sse = SseServerTransport(f"/messages/{session_id}")
    sse_transports[session_id] = sse

    async with sse.connect_sse(
        request.scope, request.receive, request._send
    ) as streams:
        try:
            await mcp_server.run(
                streams[0], streams[1],
                mcp_server.create_initialization_options()
            )
        finally:
            # Cleanup on disconnect
            del sse_transports[session_id]

    return Response()

# Starlette routes
routes = [
    Route("/sse", endpoint=handle_sse, methods=["GET"]),
    Mount("/messages/", app=lambda scope, receive, send:
          sse_transports[scope['path'].split('/')[-1]].handle_post_message(scope, receive, send)
    ),
]

app = Starlette(routes=routes)
uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Key Principles:**
1. **Single Server Instance**: One `mcp.server.Server` handles all business logic
2. **Per-Session Transports**: Each client gets a lightweight `SseServerTransport` instance
3. **Session Tracking**: Dictionary maps session ID to transport instance
4. **Resource Cleanup**: Remove session from dict on disconnect

### Integration with Zapomni Architecture

**Current Zapomni Structure:**
```
zapomni_mcp/__main__.py
  ├─ Creates MemoryProcessor (core_engine)
  ├─ Creates MCPServer(core_engine)
  └─ Calls server.run() → stdio_server()
```

**Proposed SSE Structure:**
```
zapomni_mcp/__main__.py
  ├─ Creates MemoryProcessor (core_engine) - UNCHANGED
  ├─ Creates MCPServer(core_engine) - UNCHANGED
  └─ NEW: Creates Starlette app with SSE routes
       └─ Each SSE connection → mcp_server.run(streams)
```

**Migration Steps:**
1. Keep existing `MCPServer` class unchanged (maintains stdio compatibility)
2. Add new `run_sse()` method to `MCPServer` class
3. Create Starlette app factory function
4. Add command-line flag for transport selection: `--transport [stdio|sse]`

### Example Code for Zapomni

```python
# zapomni_mcp/server.py additions

async def run_sse(self, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the server using SSE transport."""
    import uvicorn
    from starlette.applications import Starlette
    from starlette.routing import Route, Mount
    from starlette.responses import Response
    from mcp.server.sse import SseServerTransport
    from uuid import uuid4

    # Session tracking
    sessions: dict[str, SseServerTransport] = {}

    async def handle_sse(request):
        session_id = str(uuid4())
        sse = SseServerTransport(f"/messages/{session_id}")
        sessions[session_id] = sse

        self._logger.info("SSE connection established", session_id=session_id)

        async with sse.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            try:
                await self._server.run(
                    streams[0], streams[1],
                    self._server.create_initialization_options()
                )
            finally:
                del sessions[session_id]
                self._logger.info("SSE connection closed", session_id=session_id)

        return Response()

    async def handle_messages(scope, receive, send):
        # Extract session_id from path: /messages/{session_id}
        path_parts = scope['path'].strip('/').split('/')
        if len(path_parts) >= 2:
            session_id = path_parts[1]
            if session_id in sessions:
                await sessions[session_id].handle_post_message(scope, receive, send)
                return

        # Session not found
        from starlette.responses import Response
        response = Response("Session not found", status_code=404)
        await response(scope, receive, send)

    # CORS middleware for local development
    from starlette.middleware import Middleware
    from starlette.middleware.cors import CORSMiddleware

    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:*", "http://127.0.0.1:*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    ]

    routes = [
        Route("/sse", endpoint=handle_sse, methods=["GET"]),
        Mount("/messages/", app=handle_messages),
    ]

    app = Starlette(routes=routes, middleware=middleware)

    self._logger.info("Starting SSE server", host=host, port=port)
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()
```

---

## Concurrency Model

### How SSE Enables True Concurrency

**Stdio Transport (Current):**
- Single stdin/stdout stream
- Sequential request processing
- One client connection only
- Blocking I/O operations

**SSE Transport (Proposed):**
- Multiple concurrent client connections
- Parallel request processing via asyncio event loop
- Non-blocking I/O through `anyio` memory streams
- Session isolation via separate memory queues

### AsyncIO Event Loop Architecture

```
┌─────────────────────────────────────────────┐
│         Single AsyncIO Event Loop           │
│                                             │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐    │
│  │Client 1 │  │Client 2 │  │Client 3 │    │
│  │Session  │  │Session  │  │Session  │    │
│  └────┬────┘  └────┬────┘  └────┬────┘    │
│       │            │            │          │
│       ▼            ▼            ▼          │
│  ┌─────────────────────────────────────┐  │
│  │   MCP Server (Singleton)            │  │
│  │   - Tool handlers                    │  │
│  │   - Shared MemoryProcessor          │  │
│  │   - Concurrent request processing   │  │
│  └─────────────────────────────────────┘  │
│       │            │            │          │
│       ▼            ▼            ▼          │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  │
│  │ Queue 1 │  │ Queue 2 │  │ Queue 3 │  │
│  └────┬────┘  └────┬────┘  └────┬────┘  │
│       │            │            │          │
└───────┼────────────┼────────────┼──────────┘
        ▼            ▼            ▼
    SSE Stream   SSE Stream   SSE Stream
```

**Key Points:**
1. Single event loop handles all connections efficiently
2. No thread context switching overhead
3. Memory queues provide session isolation
4. Cooperative multitasking via `async`/`await`

### Handling CPU-Bound Operations

**Problem:** CPU-intensive operations (e.g., spaCy NLP processing in `EntityExtractor`) block the event loop, preventing concurrent request processing.

**Solution:** Use `loop.run_in_executor()` with `ThreadPoolExecutor`:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Create thread pool (typically 5x CPU count for I/O-bound tasks)
executor = ThreadPoolExecutor(max_workers=10)

async def handle_cpu_bound_tool(arguments: dict):
    """Example: Entity extraction (CPU-bound)"""
    loop = asyncio.get_event_loop()

    # Run blocking operation in thread pool
    result = await loop.run_in_executor(
        executor,
        sync_entity_extraction,  # Blocking function
        arguments['text']
    )

    return result

def sync_entity_extraction(text: str):
    """CPU-bound synchronous function"""
    # This runs in a separate thread, doesn't block event loop
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
```

**Recommendations for Zapomni:**
- Identify CPU-bound operations:
  - `EntityExtractor.extract()` (spaCy NLP)
  - Large text chunking in `SemanticChunker`
  - Graph traversal in `GraphBuilder`
- Wrap in `run_in_executor()` for concurrency
- Use `asyncio.gather()` for parallel processing of multiple requests

### Database Connection Pooling

**FalkorDB Connection Pool Considerations:**

From `/home/dev/zapomni/src/zapomni_db/falkordb_client.py`:
```python
class FalkorDBClient:
    DEFAULT_POOL_SIZE = 10

    def __init__(self, ..., pool_size: int = DEFAULT_POOL_SIZE):
        self.pool_size = pool_size
        self._pool = None  # Connection pool
```

**Thread Safety Verification Needed:**
- FalkorDB uses Redis protocol → likely uses `redis-py` connection pool
- Redis connection pools are generally thread-safe
- **Action Required**: Verify FalkorDB Python client thread safety in documentation

**Best Practices:**
1. **Connection Pool Size**: Set to expected concurrent clients + buffer
   - Example: 50 concurrent clients → pool_size=60
2. **Connection Timeout**: Configure reasonable timeout for busy periods
3. **Connection Validation**: Implement health checks for stale connections
4. **Error Handling**: Retry logic for connection pool exhaustion

**Recommended Configuration for SSE:**
```python
db_client = FalkorDBClient(
    host=settings.falkordb_host,
    port=settings.falkordb_port,
    pool_size=50,  # Increased from default 10
    max_retries=3,
)
```

### Rate Limiting (Optional)

Prevent resource exhaustion from too many concurrent connections:

```python
from asyncio import Semaphore

# Limit concurrent SSE connections
connection_semaphore = Semaphore(100)  # Max 100 concurrent clients

async def handle_sse(request):
    async with connection_semaphore:
        # SSE connection logic here
        session_id = str(uuid4())
        # ... rest of implementation
```

---

## Security Considerations

### CORS Configuration for Local Development

**Purpose:** Allow MCP clients (Claude Desktop, IDEs) running on different ports to connect to SSE server.

**Starlette/FastAPI Configuration:**

```python
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:*",       # Any localhost port
            "http://127.0.0.1:*",      # Any 127.0.0.1 port
            "https://claude.ai",        # Claude web app
        ],
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
]

app = Starlette(routes=routes, middleware=middleware)
```

**Development vs Production:**
- **Development**: Permissive CORS (`allow_origins=["*"]`)
- **Production**: Explicit origins only
- **Environment Variable**: Control via `ZAPOMNI_CORS_ORIGINS`

### DNS Rebinding Protection

The MCP SDK includes built-in protection via `TransportSecuritySettings`:

```python
from mcp.server.transport_security import TransportSecuritySettings

security_settings = TransportSecuritySettings(
    allowed_hosts=["localhost", "127.0.0.1"],
    validate_origin=True,
)

sse = SseServerTransport("/messages/", security_settings=security_settings)
```

**Validation:**
- Checks `Host` header against allowed list
- Validates `Origin` and `Referer` for POST requests
- Returns 403 Forbidden for mismatches

### Authentication (Future Consideration)

MCP SSE transport supports OAuth 2.0 authentication:

```python
from mcp.server.auth.settings import AuthSettings
from mcp.server.auth.provider import OAuthAuthorizationServerProvider

auth_settings = AuthSettings(
    issuer_url="https://auth.example.com",
    required_scopes=["mcp:read", "mcp:write"],
)

# Note: For Zapomni, authentication is likely not needed for local deployment
# Include for completeness if future remote access is required
```

### Rate Limiting

**Prevent Abuse:**
```python
from collections import defaultdict
from time import time

class RateLimiter:
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests = defaultdict(list)

    async def check_rate_limit(self, client_id: str) -> bool:
        now = time()
        # Remove old requests outside window
        self.requests[client_id] = [
            t for t in self.requests[client_id]
            if now - t < self.window
        ]

        if len(self.requests[client_id]) >= self.max_requests:
            return False  # Rate limit exceeded

        self.requests[client_id].append(now)
        return True

# Usage in handle_sse
rate_limiter = RateLimiter(max_requests=100, window_seconds=60)

async def handle_sse(request):
    client_id = request.client.host
    if not await rate_limiter.check_rate_limit(client_id):
        return Response("Rate limit exceeded", status_code=429)
    # ... proceed with SSE connection
```

---

## Reference Implementations

### Official MCP SDK Examples

1. **[mcp.server.sse Module](https://github.com/modelcontextprotocol/python-sdk)**
   - Source: `/home/dev/zapomni/.venv/lib/python3.12/site-packages/mcp/server/sse.py`
   - Provides `SseServerTransport` class
   - Example usage in docstring (lines 7-31)
   - Production-ready implementation

2. **[FastMCP Server (Starlette Integration)](https://github.com/modelcontextprotocol/python-sdk)**
   - Source: `/home/dev/zapomni/.venv/lib/python3.12/site-packages/mcp/server/fastmcp/server.py`
   - Lines 738-924: `run_sse_async()` and `sse_app()` methods
   - Shows Starlette route configuration
   - Includes auth middleware integration

### Community Examples

1. **[yigitkonur/example-mcp-server-sse](https://github.com/yigitkonur/example-mcp-server-sse)**
   - Demonstrates Singleton Server Pattern
   - Clean session management with in-memory map
   - Educational examples for different protocols
   - **Highly Recommended** for Zapomni implementation reference

2. **[sidharthrajaram/mcp-sse](https://github.com/sidharthrajaram/mcp-sse)**
   - Shows decoupled client-server architecture
   - Demonstrates how clients can connect/disconnect dynamically
   - Good for understanding lifecycle management

3. **[bh-rat/mcp-db](https://github.com/bh-rat/mcp-db)**
   - Advanced: Persistent session storage in Redis
   - Enables distributed MCP servers behind load balancer
   - Cross-node session recovery
   - **Future Enhancement** for Zapomni scalability

### Starlette/FastAPI SSE Resources

1. **[sse-starlette Library](https://github.com/sysid/sse-starlette)**
   - Production-ready SSE implementation
   - Used by MCP SDK under the hood
   - Features: connection management, heartbeats, graceful shutdown
   - Thread-safe, multi-loop compatible

2. **[Building SSE MCP Server with FastAPI](https://www.ragie.ai/blog/building-a-server-sent-events-sse-mcp-server-with-fastapi)**
   - Tutorial walkthrough
   - Shows FastAPI integration patterns
   - Includes CORS configuration examples

---

## Risks and Mitigations

### Risk 1: Memory Leaks from Long-Lived Connections

**Description:**
Each SSE connection holds resources (memory streams, session state) for extended periods. Without proper cleanup, memory usage grows unbounded.

**Mitigation:**
1. **Automatic Cleanup**: Use `async with` context managers to ensure cleanup
   ```python
   async with sse.connect_sse(...) as streams:
       try:
           await server.run(streams[0], streams[1], ...)
       finally:
           # Cleanup guaranteed by context manager
           del sessions[session_id]
   ```

2. **Connection Heartbeats**: Send periodic keep-alive to detect stale connections
   ```python
   async def heartbeat_task():
       while True:
           await asyncio.sleep(30)  # 30 second heartbeat
           try:
               await write_stream.send({"type": "ping"})
           except Exception:
               # Connection dead, cleanup
               break
   ```

3. **Connection Timeout**: Implement maximum connection lifetime
   ```python
   MAX_CONNECTION_DURATION = 3600  # 1 hour
   start_time = time.time()

   while time.time() - start_time < MAX_CONNECTION_DURATION:
       # Process messages
       pass
   ```

4. **Memory Monitoring**: Log memory usage periodically
   ```python
   import psutil

   process = psutil.Process()
   self._logger.info(
       "Memory usage",
       memory_mb=process.memory_info().rss / 1024 / 1024,
       active_sessions=len(sessions)
   )
   ```

**Testing:**
- Load test with 50+ concurrent clients for 30 minutes
- Monitor memory with `htop` or Prometheus
- Verify cleanup on disconnect (check session dict size)

### Risk 2: Database Connection Pool Exhaustion

**Description:**
Multiple concurrent clients may exhaust FalkorDB connection pool, causing request failures.

**Mitigation:**
1. **Increase Pool Size**: Based on expected concurrent clients
   ```python
   pool_size = max(50, expected_concurrent_clients * 1.2)
   ```

2. **Connection Validation**: Implement health checks
   ```python
   @retry(max_attempts=3, backoff=exponential)
   async def execute_query_with_retry(query: str):
       try:
           return await db_client.query(query)
       except ConnectionError:
           # Pool exhausted, wait and retry
           await asyncio.sleep(1)
           raise
   ```

3. **Circuit Breaker Pattern**: Fail fast when pool consistently unavailable
   ```python
   from pybreaker import CircuitBreaker

   db_breaker = CircuitBreaker(fail_max=5, timeout_duration=30)

   @db_breaker
   async def query_database(query: str):
       return await db_client.query(query)
   ```

4. **Pool Size Monitoring**: Track connection usage
   ```python
   self._logger.info(
       "Connection pool status",
       active_connections=db_client._pool.active_count,
       available_connections=db_client._pool.available_count
   )
   ```

**Testing:**
- Simulate 100 concurrent clients making database queries
- Verify graceful degradation (retry logic works)
- Check error logs for pool exhaustion warnings

### Risk 3: EventLoop Blocking by CPU-Bound Operations

**Description:**
Synchronous CPU-bound operations (spaCy NLP, large text processing) block the asyncio event loop, preventing concurrent request handling.

**Mitigation:**
1. **ThreadPoolExecutor for CPU-Bound Tasks**:
   ```python
   executor = ThreadPoolExecutor(max_workers=10)

   async def extract_entities(text: str):
       loop = asyncio.get_event_loop()
       result = await loop.run_in_executor(
           executor,
           entity_extractor.extract_sync,  # Blocking call
           text
       )
       return result
   ```

2. **Async-Native Libraries**: Use async-compatible alternatives where possible
   - Consider `aiohttp` for HTTP requests (already used: `OllamaEmbedder`)
   - Evaluate async NLP libraries (limited options for spaCy replacement)

3. **Operation Timeout**: Prevent runaway operations
   ```python
   try:
       result = await asyncio.wait_for(
           extract_entities(text),
           timeout=30.0  # 30 second timeout
       )
   except asyncio.TimeoutError:
       self._logger.error("Entity extraction timeout")
       raise
   ```

4. **Monitor Event Loop Lag**: Detect blocking operations
   ```python
   import asyncio

   async def monitor_event_loop():
       while True:
           start = asyncio.get_event_loop().time()
           await asyncio.sleep(0.1)
           lag = asyncio.get_event_loop().time() - start - 0.1
           if lag > 0.05:  # 50ms lag threshold
               self._logger.warning("Event loop lag detected", lag_seconds=lag)
   ```

**Testing:**
- Load test with 10 concurrent `build_graph` requests (CPU-intensive)
- Verify response times remain reasonable for other clients
- Check CPU usage distribution across threads

### Risk 4: Session Hijacking / Unauthorized Access

**Description:**
Session IDs transmitted in URLs could be intercepted or guessed, allowing unauthorized access.

**Mitigation:**
1. **HTTPS Only**: Enforce TLS for production deployments
   ```python
   if not request.url.scheme == "https":
       return Response("HTTPS required", status_code=403)
   ```

2. **Secure Session IDs**: Use cryptographically secure UUIDs
   ```python
   import secrets
   session_id = secrets.token_urlsafe(32)  # 256-bit secure token
   ```

3. **Session Expiration**: Implement time-based expiration
   ```python
   session_expiry = {}

   async def create_session():
       session_id = str(uuid4())
       session_expiry[session_id] = time.time() + 3600  # 1 hour
       return session_id

   async def validate_session(session_id: str) -> bool:
       if session_id not in sessions:
           return False
       if time.time() > session_expiry[session_id]:
           del sessions[session_id]
           del session_expiry[session_id]
           return False
       return True
   ```

4. **Origin Validation**: Use built-in DNS rebinding protection
   ```python
   security_settings = TransportSecuritySettings(
       allowed_hosts=["localhost", "127.0.0.1"],
       validate_origin=True,
   )
   ```

**Note:** For Zapomni's local deployment use case, these risks are minimal but should be addressed if remote access is added.

### Risk 5: Client Reconnection Storm

**Description:**
Many clients disconnecting and reconnecting simultaneously (e.g., network interruption) could overwhelm the server.

**Mitigation:**
1. **Connection Rate Limiting**: Limit new connections per time window
   ```python
   connection_limiter = RateLimiter(max_requests=10, window_seconds=1)

   async def handle_sse(request):
       client_ip = request.client.host
       if not await connection_limiter.check_rate_limit(client_ip):
           await asyncio.sleep(1)  # Backoff
           return Response("Too many connections", status_code=429)
   ```

2. **Exponential Backoff**: Guide clients to retry with increasing delays
   ```python
   retry_after = min(2 ** attempt, 60)  # Max 60 seconds
   return Response(
       "Server busy",
       status_code=503,
       headers={"Retry-After": str(retry_after)}
   )
   ```

3. **Connection Queue**: Buffer incoming connections
   ```python
   connection_queue = asyncio.Queue(maxsize=20)

   async def connection_handler():
       while True:
           request = await connection_queue.get()
           await handle_sse(request)
   ```

**Testing:**
- Simulate 50 clients disconnecting and reconnecting within 1 second
- Verify server remains responsive to existing connections
- Check for graceful backoff behavior

---

## Recommendations

### Top 3 Recommendations for Zapomni

#### 1. Implement Dual Transport Support (SSE + Stdio)

**Rationale:**
Maintain backward compatibility with stdio while enabling SSE for concurrent access. This provides flexibility during development and testing.

**Implementation:**
```python
# zapomni_mcp/__main__.py

async def main():
    # ... existing initialization code ...

    # Parse transport from environment variable or CLI arg
    transport = os.getenv("ZAPOMNI_TRANSPORT", "stdio")  # Default: stdio

    if transport == "sse":
        await server.run_sse(
            host=os.getenv("ZAPOMNI_HOST", "127.0.0.1"),
            port=int(os.getenv("ZAPOMNI_PORT", "8000"))
        )
    elif transport == "stdio":
        await server.run()  # Existing stdio implementation
    else:
        raise ValueError(f"Unknown transport: {transport}")
```

**Benefits:**
- Zero breaking changes to existing stdio users
- Easy A/B testing between transports
- Smooth migration path
- Stdio remains useful for development/debugging

**Effort:** Medium (2-3 days)

#### 2. Wrap CPU-Bound Operations with ThreadPoolExecutor

**Rationale:**
Prevent EntityExtractor (spaCy) and other CPU-intensive operations from blocking the asyncio event loop, ensuring concurrent request processing works correctly.

**Implementation:**
```python
# zapomni_core/entity_extraction/entity_extractor.py

import asyncio
from concurrent.futures import ThreadPoolExecutor

class EntityExtractor:
    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=5)
        # ... existing init code ...

    async def extract(self, text: str) -> List[Entity]:
        """Async wrapper for entity extraction."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._extract_sync,  # Existing sync method
            text
        )

    def _extract_sync(self, text: str) -> List[Entity]:
        """Renamed existing extract() method."""
        # ... existing spaCy logic ...
        doc = self.nlp(text)
        return [Entity(...) for ent in doc.ents]
```

**Areas to Update:**
- `EntityExtractor.extract()` - spaCy NLP processing
- `SemanticChunker.chunk()` - Large text chunking (if synchronous)
- `GraphBuilder` - Complex graph operations

**Benefits:**
- True concurrent request handling
- Improved throughput under load
- Better resource utilization (CPU + I/O parallelism)

**Effort:** Medium (3-5 days including testing)

#### 3. Configure FalkorDB Connection Pool for Concurrency

**Rationale:**
Default pool size (10) insufficient for concurrent SSE clients. Increase and validate thread safety.

**Implementation:**
```python
# zapomni_mcp/__main__.py

# STAGE 2: Initialize database client with SSE-appropriate settings
logger.info("Initializing FalkorDB client for SSE transport")
db_client = FalkorDBClient(
    host=settings.falkordb_host,
    port=settings.falkordb_port,
    graph_name=settings.graph_name,
    password=settings.falkordb_password.get_secret_value() if settings.falkordb_password else None,
    pool_size=50,  # INCREASED from default 10 for concurrent SSE clients
    max_retries=3,
)
logger.info(
    "FalkorDB client initialized for SSE",
    pool_size=50,
    expected_concurrent_clients=30,
)
```

**Configuration Guidelines:**
- **Development**: pool_size=20 (sufficient for testing)
- **Production**: pool_size=50+ (depends on expected concurrent clients)
- **Formula**: `pool_size = expected_concurrent_clients * 1.5`

**Additional Steps:**
1. **Verify Thread Safety**: Check FalkorDB Python client documentation
2. **Add Pool Monitoring**: Log active/available connections
3. **Implement Retry Logic**: Handle pool exhaustion gracefully

**Benefits:**
- Prevents connection pool exhaustion errors
- Improved reliability under load
- Better user experience (no failed requests)

**Effort:** Low (1 day)

### Secondary Recommendations

#### 4. Add Connection Monitoring and Metrics

Implement observability for SSE connections:
```python
# Track metrics
active_connections = 0
total_connections = 0
failed_connections = 0

async def handle_sse(request):
    global active_connections, total_connections
    active_connections += 1
    total_connections += 1

    self._logger.info(
        "SSE connection metrics",
        active=active_connections,
        total=total_connections
    )

    try:
        # ... SSE logic ...
    finally:
        active_connections -= 1
```

#### 5. Implement Graceful Shutdown

Ensure in-progress requests complete before server shutdown:
```python
import signal

async def shutdown_handler():
    logger.info("Shutdown signal received, closing connections...")

    # Close all SSE sessions gracefully
    for session_id, sse in list(sessions.items()):
        await sse.close()

    # Wait for in-progress requests (max 30 seconds)
    await asyncio.sleep(30)

# Register signal handlers
loop = asyncio.get_event_loop()
for sig in (signal.SIGTERM, signal.SIGINT):
    loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown_handler()))
```

#### 6. Add CORS Configuration via Environment Variables

```python
# zapomni_core/config.py

class ZapomniSettings(BaseSettings):
    # ... existing settings ...

    # SSE Transport settings
    sse_host: str = "127.0.0.1"
    sse_port: int = 8000
    cors_origins: List[str] = ["http://localhost:*", "http://127.0.0.1:*"]
    max_concurrent_connections: int = 100
```

---

## Implementation Timeline

### Phase 1: Foundation (Week 1)
- [ ] Add `run_sse()` method to `MCPServer` class
- [ ] Create Starlette app factory with SSE routes
- [ ] Implement basic session management
- [ ] Add transport selection (stdio vs SSE)

### Phase 2: Concurrency (Week 2)
- [ ] Wrap `EntityExtractor.extract()` with ThreadPoolExecutor
- [ ] Update other CPU-bound operations
- [ ] Increase FalkorDB pool size
- [ ] Add connection monitoring

### Phase 3: Polish (Week 3)
- [ ] Implement CORS configuration
- [ ] Add graceful shutdown handling
- [ ] Add heartbeat/keep-alive mechanism
- [ ] Load testing and optimization

### Phase 4: Documentation (Week 4)
- [ ] Update README with SSE setup instructions
- [ ] Document environment variables
- [ ] Create migration guide from stdio
- [ ] Add troubleshooting section

---

## Testing Strategy

### Unit Tests
```python
# tests/unit/test_sse_server.py

import pytest
from zapomni_mcp.server import MCPServer

@pytest.mark.asyncio
async def test_sse_session_creation():
    """Test SSE session creation and cleanup."""
    server = MCPServer(mock_core_engine)
    # ... test session lifecycle

@pytest.mark.asyncio
async def test_concurrent_requests():
    """Test handling of concurrent tool calls."""
    # Simulate 10 concurrent clients
    # Verify all requests complete successfully
```

### Integration Tests
```python
# tests/integration/test_sse_transport.py

@pytest.mark.asyncio
async def test_sse_connection_lifecycle():
    """Test full SSE connection lifecycle."""
    async with httpx.AsyncClient() as client:
        # Connect to /sse
        # Send tool call via POST /messages/
        # Receive response via SSE
        # Disconnect and verify cleanup

@pytest.mark.asyncio
async def test_multiple_concurrent_clients():
    """Test 50 concurrent clients."""
    # Create 50 SSE connections
    # Each sends 10 tool calls
    # Verify all complete without errors
```

### Load Tests
```bash
# Use locust or similar tool
# tests/load/locustfile.py

from locust import HttpUser, task, between

class MCPUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def connect_sse(self):
        # Connect to /sse endpoint
        # Send tool calls
        # Measure response times
```

**Load Test Targets:**
- 50 concurrent clients for 30 minutes
- Memory usage < 500MB
- Average response time < 500ms
- Zero connection errors

---

## References

### Documentation
- [MCP Transports Specification](https://modelcontextprotocol.io/specification/2025-06-18/basic/transports)
- [Why MCP Deprecated SSE](https://blog.fka.dev/blog/2025-06-06-why-mcp-deprecated-sse-and-go-with-streamable-http/)
- [Python AsyncIO Best Practices](https://docs.python.org/3/library/asyncio-dev.html)
- [FastAPI CORS Guide](https://fastapi.tiangolo.com/tutorial/cors/)
- [sse-starlette Documentation](https://github.com/sysid/sse-starlette)

### Libraries
- `mcp` - Model Context Protocol SDK
- `starlette` - ASGI web framework
- `uvicorn` - ASGI server
- `sse-starlette` - SSE implementation
- `anyio` - Async I/O abstraction

### Code Examples
- [yigitkonur/example-mcp-server-sse](https://github.com/yigitkonur/example-mcp-server-sse) - Singleton pattern
- [sidharthrajaram/mcp-sse](https://github.com/sidharthrajaram/mcp-sse) - Working SSE pattern
- [bh-rat/mcp-db](https://github.com/bh-rat/mcp-db) - Distributed session management

---

## Conclusion

The migration from stdio to SSE transport is **feasible and recommended** for Zapomni. The MCP SDK provides robust SSE support through `SseServerTransport`, and the architecture can be implemented with minimal changes to existing code.

**Key Success Factors:**
1. Maintain dual transport support (stdio + SSE) for flexibility
2. Address CPU-bound operations with ThreadPoolExecutor
3. Configure FalkorDB connection pool appropriately
4. Implement proper monitoring and observability

**Expected Benefits:**
- Support for multiple concurrent clients
- Non-blocking parallel request processing
- Better resource utilization (CPU + I/O)
- Foundation for future scalability (load balancing, distributed servers)

**Risk Level:** Low-Medium
**Effort Estimate:** 3-4 weeks
**Recommended Start:** Immediate (Phase 1 implementation)

---

**Research Complete:** 2025-11-25
**Next Steps:** Review findings with team, create implementation tasks, begin Phase 1 development
