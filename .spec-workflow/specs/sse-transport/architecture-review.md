# Architecture Review: SSE Transport Integration

**Review Date:** 2025-11-25
**Reviewer:** Architecture Agent
**Project:** Zapomni MCP Server Migration to SSE Transport

---

## Executive Summary

### Key Architectural Decisions

1. **Singleton Server Pattern**: Use single `mcp.server.Server` instance with per-session `SseServerTransport` instances. This maximizes resource sharing while maintaining session isolation.

2. **Dual Transport Support**: Add `run_sse()` method to existing `MCPServer` class while preserving `run()` for stdio. Transport selection via `--transport` CLI flag (default: sse).

3. **ThreadPoolExecutor for CPU-Bound Operations**: Wrap SpaCy entity extraction (`EntityExtractor.extract_entities()`) and GraphBuilder operations in `asyncio.run_in_executor()` to prevent event loop blocking.

4. **Connection Pool Scaling**: Increase FalkorDB pool from default 10 to 50+ connections for concurrent SSE clients. Current pool_size is configurable via `ZapomniSettings.falkordb_pool_size`.

5. **Starlette ASGI Application**: Use Starlette for routing with uvicorn as ASGI server. This provides the minimal HTTP layer needed without FastAPI overhead.

---

## Current Architecture Analysis

### Server Structure (`src/zapomni_mcp/server.py`)

The current `MCPServer` class is well-designed for extension:

```
MCPServer
├── __init__(core_engine, config)      # Initialization with MemoryProcessor
├── _server: mcp.server.Server         # Internal MCP server instance
├── _tools: Dict[str, MCPTool]         # Tool registry
├── register_tool(tool)                # Single tool registration
├── register_all_tools(memory_processor)  # All standard tools
├── run()                              # CURRENT: stdio transport only
├── shutdown()                         # Graceful shutdown
└── get_stats()                        # Server statistics
```

**Current `run()` Implementation (Lines 276-359):**
- Registers `@self._server.call_tool()` and `@self._server.list_tools()` handlers
- Uses `async with stdio_server() as streams:` for transport
- Calls `self._server.run(read_stream, write_stream, init_options)`

**Key Observation**: The MCP server instance `self._server` is already separated from transport logic. This makes SSE integration straightforward - we just need to provide different streams.

### Tool Registration Pattern

Tools are registered via `register_all_tools()` in `MCPServer`:

```python
# Lines 241-268: All tools instantiated with MemoryProcessor
tools = [
    AddMemoryTool(memory_processor=memory_processor),
    SearchMemoryTool(memory_processor=memory_processor),
    GetStatsTool(memory_processor=memory_processor),
    BuildGraphTool(memory_processor=memory_processor),
    GetRelatedTool(memory_processor=memory_processor),
    GraphStatusTool(memory_processor=memory_processor),
    ExportGraphTool(memory_processor=memory_processor),
    DeleteMemoryTool(memory_processor=memory_processor),
    ClearAllTool(memory_processor=memory_processor),
]
```

**Implication for SSE**: Tools are stateless - they operate on shared `MemoryProcessor`. This is thread-safe since `MemoryProcessor` uses async patterns and database client has connection pooling.

### Async Patterns

**Current async flow:**
1. `__main__.py`: `asyncio.run(main())` creates single event loop
2. All tool `execute()` methods are `async`
3. `MemoryProcessor` methods are `async`
4. `FalkorDBClient` uses `await asyncio.to_thread()` for blocking FalkorDB calls

**Good Existing Patterns:**
- Line 332 in `falkordb_client.py`: `result = await asyncio.to_thread(self.graph.query, cypher, parameters)`
- Line 997: Same pattern for `_execute_cypher()`

**Problematic Patterns:**
- `EntityExtractor.extract_entities()` is **synchronous** (line 186)
- `EntityExtractor._spacy_extract()` is **synchronous** (line 430) - calls `doc = self.spacy_nlp(text)`
- These block the event loop during NLP processing

### Entry Point (`src/zapomni_mcp/__main__.py`)

**Current initialization sequence:**
```
1. Configure logging
2. Load ZapomniSettings from environment
3. Initialize FalkorDBClient
4. Initialize SemanticChunker
5. Initialize OllamaEmbedder
6. Initialize MemoryProcessor (lazy loads SpaCy)
7. Create MCPServer
8. Register all tools
9. await server.run()  # <-- stdio transport
```

This sequence needs minimal changes for SSE - only step 9 changes based on transport flag.

---

## Proposed Changes

### Files to Modify

| File | Changes Required | Impact |
|------|------------------|--------|
| `src/zapomni_mcp/server.py` | Add `run_sse()` method, create Starlette app factory | High - core transport change |
| `src/zapomni_mcp/__main__.py` | Add CLI argument parsing, transport selection logic | Medium - entry point changes |
| `src/zapomni_mcp/config.py` | Add SSE-specific settings (host, port, cors_origins) | Low - config extension |
| `src/zapomni_core/config.py` | Add `sse_*` and `falkordb_pool_size` settings | Low - config extension |
| `src/zapomni_core/extractors/entity_extractor.py` | Make `extract_entities()` async with executor | High - CPU-bound fix |
| `src/zapomni_core/graph/graph_builder.py` | Ensure async patterns, may need executor wrapping | Medium - consistency |
| `src/zapomni_db/falkordb_client.py` | Verify thread-safety, add pool monitoring | Low - validation |
| `pyproject.toml` | Add `uvicorn`, `starlette` dependencies | Low - dependencies |

### New Files to Create

| File | Purpose |
|------|---------|
| `src/zapomni_mcp/sse_transport.py` | SSE transport implementation, Starlette routes, session management |
| `src/zapomni_mcp/session_manager.py` | Session lifecycle management (create, track, cleanup) |
| `tests/integration/test_sse_transport.py` | Integration tests for SSE endpoints |
| `tests/load/locustfile.py` | Load testing for concurrent SSE clients |

---

## CPU-Bound Operations Analysis

### Critical Blocking Operations

| Location | Operation | Blocking Duration | Fix Strategy |
|----------|-----------|-------------------|--------------|
| `EntityExtractor.extract_entities()` | SpaCy NLP processing | 10-100ms per document | Wrap in `run_in_executor()` |
| `EntityExtractor._spacy_extract()` | `self.spacy_nlp(text)` | 5-50ms | Called by extract_entities() |
| `EntityExtractor._llm_refine()` | LLM API call | 1-2s (but already uses executor) | Already handled |
| `SemanticChunker.chunk_text()` | Text chunking | <5ms typically | Low priority |
| `GraphBuilder.build_graph()` | Entity iteration | Depends on entity count | May need batching |

### Recommended Fix: EntityExtractor

**Current (Synchronous):**
```python
# src/zapomni_core/extractors/entity_extractor.py:186
def extract_entities(self, text: str) -> List[Entity]:
    # Validation...
    entities = self._spacy_extract(text)  # BLOCKING
    # ...
```

**Proposed (Async with Executor):**
```python
from concurrent.futures import ThreadPoolExecutor
import asyncio

class EntityExtractor:
    def __init__(self, ...):
        # ...
        self._executor = ThreadPoolExecutor(max_workers=5)

    async def extract_entities(self, text: str) -> List[Entity]:
        """Async entity extraction with executor."""
        loop = asyncio.get_event_loop()
        entities = await loop.run_in_executor(
            self._executor,
            self._extract_entities_sync,
            text
        )
        return entities

    def _extract_entities_sync(self, text: str) -> List[Entity]:
        """Synchronous extraction (runs in executor)."""
        # Original extract_entities() logic here
        # ...
```

### Impact on Dependent Code

Files that call `extract_entities()`:
1. `src/zapomni_mcp/tools/build_graph.py` (line 328) - Already async, needs `await`
2. `src/zapomni_core/graph/graph_builder.py` (line 268) - Already async, needs `await`
3. Tests - Need async test fixtures

---

## Database Connection Pooling

### Current State

**`src/zapomni_db/falkordb_client.py`:**
```python
class FalkorDBClient:
    DEFAULT_POOL_SIZE = 10  # Line 43

    def __init__(self, ..., pool_size: int = DEFAULT_POOL_SIZE):
        # ...
        self.pool_size = pool_size  # Stored but not well-utilized
```

**`src/zapomni_core/config.py`:**
```python
falkordb_pool_size: int = Field(
    default=10,
    ge=1,
    le=100,
    description="FalkorDB connection pool size"
)
```

### Current Pool Usage

Looking at `FalkorDBClient._init_connection()`:
```python
def _init_connection(self):
    self._pool = FalkorDB(
        host=self.host,
        port=self.port,
        password=self.password
    )
    self.graph = self._pool.select_graph(self.graph_name)
```

**Issue**: FalkorDB Python client doesn't explicitly expose connection pooling configuration in the constructor. The underlying Redis connection pool is managed internally.

### Recommended Changes

1. **Increase Default Pool Size in Config:**
```python
# src/zapomni_core/config.py
falkordb_pool_size: int = Field(
    default=50,  # Changed from 10
    ge=1,
    le=200,      # Increased max
    description="FalkorDB connection pool size for SSE concurrency"
)
```

2. **Configure Redis Connection Pool Explicitly:**
```python
# src/zapomni_db/falkordb_client.py
from redis import ConnectionPool

def _init_connection(self):
    # Create explicit connection pool for better control
    pool = ConnectionPool(
        host=self.host,
        port=self.port,
        password=self.password,
        max_connections=self.pool_size,
    )
    self._pool = FalkorDB.from_pool(pool)  # Or equivalent
    self.graph = self._pool.select_graph(self.graph_name)
```

3. **Add Pool Monitoring:**
```python
async def get_pool_stats(self) -> Dict[str, Any]:
    """Get connection pool statistics."""
    return {
        "pool_size": self.pool_size,
        "active_connections": self._pool._pool.num_connections if self._pool else 0,
        # Additional metrics...
    }
```

### Thread Safety Verification

FalkorDB uses the Redis protocol. The `redis-py` library's connection pool is thread-safe. Key points:

1. Each `FalkorDBClient` operation:
   - Acquires connection from pool
   - Executes query
   - Returns connection to pool

2. Current async pattern `asyncio.to_thread()` ensures:
   - Blocking Redis calls run in thread pool
   - Event loop remains responsive
   - Connection pool handles thread contention

**Verification Needed**: Run concurrent tests to confirm no race conditions with pool size 50+.

---

## Integration Points

### SSE Transport Integration Diagram

```
                                   ┌─────────────────────────────────────┐
                                   │         Starlette App               │
                                   │  ┌─────────────────────────────┐   │
     Client 1 ──GET /sse──────────>│  │  handle_sse()               │   │
                                   │  │  ├─ Create SseServerTransport│   │
                                   │  │  ├─ sessions[id] = transport │   │
                                   │  │  └─ mcp_server.run(streams)  │   │
                                   │  └─────────────────────────────┘   │
     Client 1 ──POST /messages/id─>│  ┌─────────────────────────────┐   │
                                   │  │  handle_messages()          │   │
                                   │  │  └─ sessions[id].handle_post │   │
                                   │  └─────────────────────────────┘   │
                                   └──────────────┬──────────────────────┘
                                                  │
                                                  ▼
                                   ┌─────────────────────────────────────┐
                                   │         MCPServer (Singleton)       │
                                   │  ├─ _server: mcp.server.Server      │
                                   │  ├─ _tools: Dict[str, MCPTool]      │
                                   │  │   ├─ add_memory                  │
                                   │  │   ├─ search_memory               │
                                   │  │   ├─ build_graph  ───────────────┼──> EntityExtractor (w/ executor)
                                   │  │   └─ ...                         │
                                   │  └─ core_engine: MemoryProcessor    │
                                   └──────────────┬──────────────────────┘
                                                  │
                                                  ▼
                                   ┌─────────────────────────────────────┐
                                   │         FalkorDBClient              │
                                   │  └─ Connection Pool (50+ conns)     │
                                   └─────────────────────────────────────┘
```

### Session Lifecycle

```
1. Client connects: GET /sse
   ├─ Generate UUID session_id
   ├─ Create SseServerTransport(endpoint=f"/messages/{session_id}")
   ├─ Store: sessions[session_id] = transport
   ├─ Send SSE event: endpoint=/messages/{session_id}
   └─ Keep connection open for SSE stream

2. Client sends message: POST /messages/{session_id}
   ├─ Extract session_id from path
   ├─ Lookup transport: sessions[session_id]
   ├─ Forward to transport.handle_post_message()
   └─ Return 202 Accepted

3. Client disconnects
   ├─ SSE stream closes (detected by context manager)
   ├─ Cleanup: del sessions[session_id]
   └─ Log disconnection
```

---

## Breaking Changes

### API Changes (None Expected)

The MCP protocol interface remains unchanged:
- Same tool definitions
- Same JSON-RPC message format
- Same request/response structure

### Behavioral Changes

| Change | Impact | Mitigation |
|--------|--------|------------|
| Concurrent request processing | Multiple clients can execute tools simultaneously | Ensure all tools are stateless (already true) |
| EntityExtractor becomes async | Callers must use `await` | Update all call sites (2 files) |
| Connection pool exhaustion possible | High concurrency may exhaust pool | Increase pool size, add monitoring |
| Memory usage per connection | Each SSE connection holds resources | Implement connection limits, cleanup |

### Configuration Changes

New environment variables (with defaults for backward compatibility):
```bash
ZAPOMNI_TRANSPORT=sse          # New, default: sse
ZAPOMNI_SSE_HOST=127.0.0.1     # New, default: 127.0.0.1
ZAPOMNI_SSE_PORT=8000          # New, default: 8000
ZAPOMNI_CORS_ORIGINS=*         # New, default: * (permissive for dev)
FALKORDB_POOL_SIZE=50          # Existing, new default: 50
```

---

## Backward Compatibility

### Stdio Support Preservation

**Strategy**: Add new `run_sse()` method without modifying existing `run()`:

```python
class MCPServer:
    async def run(self) -> None:
        """Start server with stdio transport (original behavior)."""
        # ... existing implementation unchanged ...

    async def run_sse(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        """Start server with SSE transport (new)."""
        # ... SSE implementation ...
```

**Entry Point (`__main__.py`):**
```python
import argparse

async def main():
    # ... initialization ...

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="sse",
        help="Transport protocol (default: sse)"
    )
    args = parser.parse_args()

    if args.transport == "sse":
        await server.run_sse(
            host=os.getenv("ZAPOMNI_SSE_HOST", "127.0.0.1"),
            port=int(os.getenv("ZAPOMNI_SSE_PORT", "8000"))
        )
    else:
        await server.run()  # Original stdio
```

### Claude Desktop Compatibility

Claude Desktop can use either transport:
- **Stdio**: `python -m zapomni_mcp --transport stdio`
- **SSE**: Configure Claude Desktop to connect to `http://localhost:8000/sse`

---

## Architectural Concerns

### Risk 1: Event Loop Blocking (CRITICAL)

**Problem**: SpaCy NLP in `EntityExtractor.extract_entities()` is CPU-bound and synchronous. Under SSE with concurrent clients, this blocks the event loop and delays ALL other requests.

**Severity**: High - Will cause noticeable latency spikes

**Mitigation**:
1. Wrap `extract_entities()` in `run_in_executor()` (mandatory)
2. Set executor pool size to `5-10` (balance CPU utilization)
3. Add timeout for entity extraction operations

### Risk 2: Memory Leaks from Long-Lived Connections

**Problem**: SSE connections remain open indefinitely. Session state accumulates.

**Severity**: Medium - Gradual memory growth

**Mitigation**:
1. Use `async with` context manager for automatic cleanup
2. Implement connection heartbeats (30s interval)
3. Add connection timeout (1 hour max)
4. Monitor active sessions count

### Risk 3: Connection Pool Exhaustion

**Problem**: 50+ concurrent clients may exhaust FalkorDB connections if operations are slow.

**Severity**: Medium - Causes request failures

**Mitigation**:
1. Increase pool size to 50+ (from default 10)
2. Add pool monitoring metrics
3. Implement retry with exponential backoff
4. Add circuit breaker for database operations

### Risk 4: Session Hijacking (Low for Local Use)

**Problem**: Session IDs in URLs could be intercepted.

**Severity**: Low (local deployment)

**Mitigation**:
1. Use cryptographically secure session IDs (`secrets.token_urlsafe(32)`)
2. Enable DNS rebinding protection via `TransportSecuritySettings`
3. For remote access: require HTTPS

### Risk 5: No Rate Limiting

**Problem**: Single client could overwhelm server with requests.

**Severity**: Low (local deployment)

**Mitigation**:
1. Add connection rate limiting (10 new connections/second)
2. Add request rate limiting per session (100 requests/minute)
3. Implement graceful backoff (429 Too Many Requests)

---

## Recommended Implementation Order

### Phase 1: Foundation (Week 1)

1. **Add dependencies to `pyproject.toml`:**
   - `uvicorn[standard]>=0.30.0`
   - `starlette>=0.38.0`

2. **Create `src/zapomni_mcp/sse_transport.py`:**
   - Starlette app factory
   - `/sse` endpoint handler
   - `/messages/{session_id}` endpoint handler
   - Session tracking dict

3. **Add `run_sse()` to `MCPServer`:**
   - Create Starlette app
   - Configure CORS middleware
   - Start uvicorn server

4. **Update `__main__.py`:**
   - Add argparse for `--transport` flag
   - Route to appropriate `run()` or `run_sse()`

5. **Add SSE settings to config:**
   - `sse_host`, `sse_port`, `cors_origins`

### Phase 2: Concurrency Fixes (Week 2)

6. **Make `EntityExtractor.extract_entities()` async:**
   - Add `ThreadPoolExecutor` to class
   - Create async wrapper method
   - Keep sync version as `_extract_entities_sync()`

7. **Update call sites:**
   - `build_graph.py`: Add `await`
   - `graph_builder.py`: Add `await`
   - Update tests

8. **Increase FalkorDB pool size:**
   - Change default from 10 to 50
   - Add environment variable override

9. **Add pool monitoring:**
   - Connection count metrics
   - Pool exhaustion warnings

### Phase 3: Polish (Week 3)

10. **Implement connection lifecycle:**
    - Connection heartbeats
    - Connection timeout
    - Graceful shutdown

11. **Add CORS configuration:**
    - Environment variable for origins
    - Production vs development modes

12. **Add metrics and logging:**
    - Active connections count
    - Request latency histograms
    - Error rates

### Phase 4: Testing (Week 4)

13. **Unit tests:**
    - SSE endpoint handlers
    - Session management
    - Transport selection

14. **Integration tests:**
    - Full SSE connection lifecycle
    - Concurrent client handling
    - Error scenarios

15. **Load tests:**
    - 50 concurrent clients
    - Memory usage monitoring
    - Latency percentiles

---

## Appendix: Key File Locations

| Component | File Path |
|-----------|-----------|
| MCP Server | `/home/dev/zapomni/src/zapomni_mcp/server.py` |
| Entry Point | `/home/dev/zapomni/src/zapomni_mcp/__main__.py` |
| MCP Config | `/home/dev/zapomni/src/zapomni_mcp/config.py` |
| Core Config | `/home/dev/zapomni/src/zapomni_core/config.py` |
| Entity Extractor | `/home/dev/zapomni/src/zapomni_core/extractors/entity_extractor.py` |
| Graph Builder | `/home/dev/zapomni/src/zapomni_core/graph/graph_builder.py` |
| FalkorDB Client | `/home/dev/zapomni/src/zapomni_db/falkordb_client.py` |
| Memory Processor | `/home/dev/zapomni/src/zapomni_core/memory_processor.py` |
| Build Graph Tool | `/home/dev/zapomni/src/zapomni_mcp/tools/build_graph.py` |
| Research Document | `/home/dev/zapomni/.spec-workflow/specs/sse-transport/research.md` |

---

**Review Complete:** 2025-11-25
**Next Steps:** Review with team, create implementation tasks in tasks.md, begin Phase 1
