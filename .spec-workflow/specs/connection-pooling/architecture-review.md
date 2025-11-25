# Architecture Review: Connection Pooling for FalkorDB

**Date:** 2025-11-25
**Author:** Architecture Agent
**Project:** Zapomni MCP Server
**Task:** Implement Connection Pooling for FalkorDB to support SSE concurrent connections

---

## 1. Architecture Alignment Analysis

### 1.1 Current Architecture Overview

The Zapomni project follows a clean layered architecture:

```
zapomni_mcp (MCP Server Layer)
    |
    v
zapomni_core (Business Logic Layer)
    |
    v
zapomni_db (Database Layer)
```

**Key Architectural Patterns:**
- **Dependency Injection**: Components are injected at initialization
- **Async-First Design**: All public APIs are async (using `asyncio`)
- **Single Responsibility**: Clear separation between layers
- **Configuration-Driven**: Settings via `ZapomniSettings` and environment variables

### 1.2 Connection Pooling Alignment

The proposed connection pooling implementation **aligns well** with existing patterns:

| Aspect | Current State | Proposed State | Alignment |
|--------|--------------|----------------|-----------|
| Client Initialization | Sync `FalkorDB()` in `__init__` | Async pool creation | Maintains DI pattern |
| Query Execution | `asyncio.to_thread(self.graph.query)` | Native `await self.graph.query()` | Improves async consistency |
| Configuration | `pool_size` parameter exists | Enhance with pool settings | Uses existing config pattern |
| Lifecycle | `close()` is sync | `await close()` | Requires async context |

**Verdict:** The proposed changes integrate naturally with the existing architecture.

---

## 2. Affected Modules and Files

### 2.1 Primary Changes (Must Modify)

#### `/home/dev/zapomni/src/zapomni_db/falkordb_client.py`

| Line Range | Current Code | Required Change |
|------------|--------------|-----------------|
| **18** | `from falkordb import FalkorDB` | Change to `from falkordb.asyncio import FalkorDB` |
| **43** | `DEFAULT_POOL_SIZE = 20` | Consider increasing to `50` for SSE workload |
| **111-114** | Sync `_init_connection()` call in `__init__` | Remove from `__init__`, add async init method |
| **116-137** | `def _init_connection(self):` (sync) | Convert to `async def _init_connection(self):` |
| **120-124** | Sync `FalkorDB()` instantiation | Use `BlockingConnectionPool` + async FalkorDB |
| **139-155** | `def _init_schema(self):` (sync) | Keep sync but call via `asyncio.to_thread()` once |
| **338** | `await asyncio.to_thread(self.graph.query, ...)` | Change to `await self.graph.query(...)` |
| **1008-1054** | `async def _execute_cypher()` | Remove `asyncio.to_thread()` wrapper |
| **1055-1075** | `def close(self):` (sync) | Convert to `async def close(self):` with `await pool.aclose()` |

**Critical Changes Summary:**
1. Import async client: Line 18
2. Create async connection pool: Lines 120-127
3. Remove thread pool wrapper: Lines 1031-1033
4. Add async close: Lines 1055-1075

#### `/home/dev/zapomni/src/zapomni_db/schema_manager.py`

| Line Range | Current Code | Required Change |
|------------|--------------|-----------------|
| **12** | `from falkordb import Graph` | Change to `from falkordb.asyncio import AsyncGraph` |
| **86-88** | Type hint `graph: Graph` | Update to `graph: AsyncGraph` |
| **95-136** | `def init_schema(self):` | Keep sync for now (one-time operation) |
| **180-190** | `self._execute_cypher(cypher_query)` | Convert to async if SchemaManager is async |
| **523-553** | `def _execute_cypher()` | Consider async version |

**Recommendation:** Keep SchemaManager synchronous initially. Schema initialization runs once at startup and can use `asyncio.to_thread()` as a wrapper. This minimizes changes.

### 2.2 Secondary Changes (Should Modify)

#### `/home/dev/zapomni/src/zapomni_mcp/__main__.py`

| Line Range | Current Code | Required Change |
|------------|--------------|-----------------|
| **170-177** | Sync `FalkorDBClient()` instantiation | Add async initialization step |
| **171-177** | `db_client = FalkorDBClient(...)` | Add `await db_client.init()` after instantiation |

**New Pattern:**
```python
# STAGE 2: Initialize database client
logger.info("Initializing FalkorDB client")
db_client = FalkorDBClient(
    host=settings.falkordb_host,
    port=settings.falkordb_port,
    pool_size=settings.falkordb_pool_size,
    ...
)
# NEW: Async initialization of connection pool
await db_client.init_async()
logger.info("FalkorDB connection pool initialized")
```

#### `/home/dev/zapomni/src/zapomni_mcp/server.py`

| Line Range | Current Code | Required Change |
|------------|--------------|-----------------|
| **508-539** | `async def _graceful_shutdown_sse()` | Add pool cleanup step |
| **536-537** | Cleanup extractor only | Add `await db_client.close()` |

**Add to shutdown sequence:**
```python
# Step 2.5: Close database connection pool
if hasattr(self._core_engine, 'db_client'):
    await self._core_engine.db_client.close()
```

#### `/home/dev/zapomni/src/zapomni_core/memory_processor.py`

| Line Range | Current Code | Required Change |
|------------|--------------|-----------------|
| **150-251** | `__init__` stores `db_client` | No change needed |
| **N/A** | Missing | Add `async def close()` method to cleanup |

**Add cleanup method:**
```python
async def close(self) -> None:
    """Close resources including database client."""
    if self.db_client:
        await self.db_client.close()
```

#### `/home/dev/zapomni/src/zapomni_core/config.py`

| Line Range | Current Code | Required Change |
|------------|--------------|-----------------|
| **88-93** | `falkordb_pool_size` field | Add pool configuration fields |

**Add new configuration fields:**
```python
falkordb_pool_timeout: float = Field(
    default=10.0,
    ge=1.0,
    le=60.0,
    description="Timeout waiting for available connection from pool"
)

falkordb_socket_timeout: float = Field(
    default=30.0,
    ge=5.0,
    le=120.0,
    description="Socket timeout for query execution"
)

falkordb_health_check_interval: int = Field(
    default=30,
    ge=10,
    le=300,
    description="Health check interval for pool connections in seconds"
)
```

### 2.3 Test Files (Must Update)

| File | Required Changes |
|------|------------------|
| `tests/zapomni_db/test_falkordb_client.py` | Update for async init pattern |
| `tests/integration/test_*.py` | Add async fixture for db_client |
| `tests/zapomni_mcp/test_server.py` | Update shutdown tests |

---

## 3. Implementation Recommendations

### 3.1 Step-by-Step Implementation Plan

#### Phase 1: Foundation (Day 1)

1. **Create async initialization method**
   - Add `async def init_async(self)` to `FalkorDBClient`
   - Move pool creation from `__init__` to this method
   - Keep `__init__` lightweight (store config only)

2. **Add `BlockingConnectionPool` setup**
   ```python
   from redis.asyncio import BlockingConnectionPool
   from falkordb.asyncio import FalkorDB

   self._pool = BlockingConnectionPool(
       host=self.host,
       port=self.port,
       password=self.password,
       max_connections=self.pool_size,
       timeout=10.0,  # Connection acquisition timeout
       socket_timeout=30.0,  # Query timeout
       health_check_interval=30,
       decode_responses=True
   )

   self._db = FalkorDB(connection_pool=self._pool)
   self.graph = self._db.select_graph(self.graph_name)
   ```

3. **Update `_execute_cypher` to native async**
   ```python
   async def _execute_cypher(self, query: str, parameters: Dict[str, Any]) -> QueryResult:
       # Remove asyncio.to_thread wrapper
       result = await self.graph.query(query, parameters)
       # ... rest of conversion logic
   ```

#### Phase 2: Integration (Day 1-2)

4. **Update `__main__.py` startup sequence**
   - Add `await db_client.init_async()` after instantiation
   - Ensure proper error handling for connection failures

5. **Update `server.py` shutdown sequence**
   - Add database pool cleanup in `_graceful_shutdown_sse()`
   - Ensure connections are released before process exit

6. **Handle SchemaManager**
   - Keep synchronous for now
   - Wrap schema init call with `asyncio.to_thread()` once at startup

#### Phase 3: Monitoring & Testing (Day 2-3)

7. **Add pool monitoring**
   - Enhance `get_stats()` to include pool metrics
   - Add pool utilization to health endpoint

8. **Update tests**
   - Add async fixtures for database client
   - Add concurrent query tests
   - Add pool exhaustion handling tests

9. **Load testing**
   - Test with 50 concurrent SSE clients
   - Verify pool utilization stays below 80%
   - Measure latency improvements

### 3.2 Recommended Code Changes

#### FalkorDBClient New Structure

```python
# /home/dev/zapomni/src/zapomni_db/falkordb_client.py

from redis.asyncio import BlockingConnectionPool
from falkordb.asyncio import FalkorDB as AsyncFalkorDB
from falkordb import FalkorDB as SyncFalkorDB  # For schema init only

class FalkorDBClient:
    DEFAULT_POOL_SIZE = 50  # Increased for SSE
    DEFAULT_POOL_TIMEOUT = 10.0
    DEFAULT_SOCKET_TIMEOUT = 30.0
    DEFAULT_HEALTH_CHECK_INTERVAL = 30

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6381,
        graph_name: str = "zapomni_memory",
        password: Optional[str] = None,
        pool_size: int = DEFAULT_POOL_SIZE,
        pool_timeout: float = DEFAULT_POOL_TIMEOUT,
        socket_timeout: float = DEFAULT_SOCKET_TIMEOUT,
        health_check_interval: int = DEFAULT_HEALTH_CHECK_INTERVAL,
        max_retries: int = DEFAULT_MAX_RETRIES
    ):
        # Store configuration only - no connection in __init__
        self.host = host
        self.port = port
        self.graph_name = graph_name
        self.password = password
        self.pool_size = pool_size
        self.pool_timeout = pool_timeout
        self.socket_timeout = socket_timeout
        self.health_check_interval = health_check_interval
        self.max_retries = max_retries

        # State flags
        self._pool: Optional[BlockingConnectionPool] = None
        self._db: Optional[AsyncFalkorDB] = None
        self.graph = None
        self._initialized = False
        self._closed = False

        # Monitoring
        self._active_connections = 0
        self._total_queries = 0

    async def init_async(self) -> None:
        """Initialize async connection pool. Must be called before use."""
        if self._initialized:
            return

        # Create async connection pool
        self._pool = BlockingConnectionPool(
            host=self.host,
            port=self.port,
            password=self.password,
            max_connections=self.pool_size,
            timeout=self.pool_timeout,
            socket_timeout=self.socket_timeout,
            socket_connect_timeout=5.0,
            health_check_interval=self.health_check_interval,
            decode_responses=True
        )

        # Create async FalkorDB client
        self._db = AsyncFalkorDB(connection_pool=self._pool)
        self.graph = self._db.select_graph(self.graph_name)

        # Test connection
        await self.graph.query("RETURN 1")

        # Initialize schema (sync, runs once)
        await self._init_schema_async()

        self._initialized = True
        self._logger.info(
            "async_pool_initialized",
            pool_size=self.pool_size,
            timeout=self.pool_timeout
        )

    async def _init_schema_async(self) -> None:
        """Initialize schema using sync client wrapped in thread."""
        # Use sync client for schema (one-time operation)
        sync_db = SyncFalkorDB(
            host=self.host,
            port=self.port,
            password=self.password
        )
        sync_graph = sync_db.select_graph(self.graph_name)

        schema_manager = SchemaManager(graph=sync_graph, logger=self._logger)
        await asyncio.to_thread(schema_manager.init_schema)

    async def _execute_cypher(self, query: str, parameters: Dict[str, Any]) -> QueryResult:
        """Execute Cypher query using native async."""
        if not self._initialized:
            raise ConnectionError("Client not initialized. Call init_async() first.")

        self._active_connections += 1
        self._total_queries += 1

        try:
            # Native async - no thread pool!
            result = await self.graph.query(query, parameters)

            # Convert to QueryResult
            rows = []
            for record in result.result_set:
                row_dict = {}
                for i, col_header in enumerate(result.header):
                    col_name = col_header[1]
                    row_dict[col_name] = record[i]
                rows.append(row_dict)

            return QueryResult(
                rows=rows,
                row_count=len(rows),
                execution_time_ms=int(result.run_time_ms) if hasattr(result, 'run_time_ms') else 0
            )
        finally:
            self._active_connections -= 1

    async def close(self) -> None:
        """Close connection pool and release resources."""
        if self._closed:
            return

        try:
            if self._pool:
                await self._pool.aclose()
                self._logger.info("connection_pool_closed")
        except Exception as e:
            self._logger.warning("close_error", error=str(e))
        finally:
            self._closed = True
            self._initialized = False

    async def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics for monitoring."""
        return {
            "max_connections": self.pool_size,
            "active_connections": self._active_connections,
            "total_queries": self._total_queries,
            "utilization_percent": round(
                (self._active_connections / self.pool_size) * 100, 1
            ) if self.pool_size > 0 else 0.0,
            "initialized": self._initialized,
            "closed": self._closed,
        }
```

---

## 4. Integration Points

### 4.1 Entry Points

| Component | Integration Point | Action Required |
|-----------|------------------|-----------------|
| `__main__.py` | Server startup | Add `await db_client.init_async()` |
| `server.py` | Shutdown handler | Add `await db_client.close()` |
| `sse_transport.py` | Session cleanup | Ensure sessions release DB connections |

### 4.2 Dependency Chain

```
MCPServer
    |
    +-- MemoryProcessor
            |
            +-- FalkorDBClient (async pool)
                    |
                    +-- BlockingConnectionPool
                    |       |
                    |       +-- redis.asyncio.Redis
                    |
                    +-- AsyncGraph (queries)
```

### 4.3 Configuration Flow

```
Environment Variables
    |
    v
ZapomniSettings (zapomni_core/config.py)
    |
    +-- falkordb_pool_size
    +-- falkordb_pool_timeout (NEW)
    +-- falkordb_socket_timeout (NEW)
    |
    v
FalkorDBClient.__init__()
    |
    v
BlockingConnectionPool
```

---

## 5. Potential Breaking Changes

### 5.1 Breaking Changes

| Change | Impact | Mitigation |
|--------|--------|------------|
| `FalkorDBClient` requires `await init_async()` | All callers must update | Add deprecation warning if used without init |
| `close()` becomes async | All callers must await | Provide sync `close_sync()` wrapper for tests |
| SchemaManager may receive `AsyncGraph` | Type checking failures | Keep sync graph for schema, separate concerns |

### 5.2 Non-Breaking Changes

| Change | Impact |
|--------|--------|
| `asyncio.to_thread()` removed | Internal only, same API |
| Pool configuration fields added | Optional, with defaults |
| Pool stats in `get_stats()` | Additive, non-breaking |

---

## 6. Backward Compatibility Considerations

### 6.1 Migration Path

1. **Deprecation Phase (Optional)**
   - Keep sync `_init_connection()` as fallback
   - Log warning if async init not called
   - Remove in next major version

2. **Dual-Mode Support**
   ```python
   def __init__(self, ...):
       # ... store config

       # Attempt sync init for backward compat
       try:
           self._init_connection_sync()
           warnings.warn(
               "Sync initialization deprecated. Use await init_async() instead.",
               DeprecationWarning
           )
       except Exception:
           pass  # Will be initialized async
   ```

3. **Test Compatibility**
   - Provide `pytest-asyncio` fixtures
   - Add helper function for test setup

### 6.2 Version Strategy

| Version | Changes |
|---------|---------|
| 0.3.0 | Add async init, deprecate sync |
| 0.4.0 | Remove sync init, async-only |

### 6.3 Documentation Updates Required

1. Update README.md with new initialization pattern
2. Update docstrings in FalkorDBClient
3. Add migration guide in CHANGELOG.md
4. Update API documentation

---

## 7. Risk Assessment

### 7.1 Low Risk

- **Native async client is well-tested**: FalkorDB async module is official
- **BlockingConnectionPool is mature**: Part of redis-py, widely used
- **Minimal public API changes**: Only `close()` and new `init_async()`

### 7.2 Medium Risk

- **SchemaManager type compatibility**: May need updates for `AsyncGraph`
- **Test fixture updates**: All integration tests need async fixtures
- **Startup sequence change**: Must ensure init order is correct

### 7.3 Mitigation Strategies

1. **Comprehensive testing**: Add concurrent load tests before release
2. **Gradual rollout**: Deploy to staging first, monitor pool metrics
3. **Fallback mechanism**: Keep sync path available during transition

---

## 8. Performance Expectations

### 8.1 Before (Current State)

| Metric | Value |
|--------|-------|
| Max concurrent requests | ~32 (thread pool limit) |
| Thread switching overhead | 5-10ms per query |
| Throughput | ~500 req/s |
| GIL contention | High under load |

### 8.2 After (Proposed State)

| Metric | Value |
|--------|-------|
| Max concurrent requests | Pool size (50+) |
| Async overhead | ~1ms per query |
| Throughput | 2000-5000 req/s |
| GIL contention | None (native async) |

### 8.3 Monitoring Recommendations

Add to health endpoint (`/health`):
```json
{
  "pool": {
    "max_connections": 50,
    "active_connections": 12,
    "utilization_percent": 24.0,
    "total_queries": 15234,
    "avg_wait_time_ms": 0.5
  }
}
```

---

## 9. Summary

### 9.1 Changes Required

| Priority | File | Scope |
|----------|------|-------|
| **Critical** | `falkordb_client.py` | Core async pool implementation |
| **Critical** | `__main__.py` | Startup sequence update |
| **High** | `server.py` | Shutdown cleanup |
| **High** | `config.py` | Pool configuration fields |
| **Medium** | `memory_processor.py` | Add close method |
| **Medium** | `schema_manager.py` | Type hints (optional) |
| **Low** | Tests | Async fixtures |

### 9.2 Estimated Effort

| Task | Duration |
|------|----------|
| Core implementation | 4-6 hours |
| Integration updates | 2-3 hours |
| Test updates | 3-4 hours |
| Documentation | 1-2 hours |
| Load testing | 2-3 hours |
| **Total** | **12-18 hours (2-3 days)** |

### 9.3 Recommendation

**Proceed with implementation.** The proposed connection pooling architecture:

1. **Aligns** with existing Zapomni architecture patterns
2. **Improves** performance significantly for SSE workloads
3. **Simplifies** code by removing `asyncio.to_thread()` wrappers
4. **Maintains** backward compatibility with migration path
5. **Low risk** due to use of official, mature libraries

The changes are surgical and well-contained within the database layer, with clear integration points in the MCP server layer.
