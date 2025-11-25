# Requirements Document - Connection Pooling for FalkorDB

**Spec Name:** connection-pooling
**Status:** Requirements Phase
**Created:** 2025-11-25
**Dependencies:** SSE Transport (complete), Phase 3 MCP Tools (complete)

---

## 1. Executive Summary

### 1.1 Problem Statement

The current Zapomni implementation uses a **synchronous FalkorDB client** with `asyncio.to_thread()` to run queries in a thread pool. With the recent SSE transport implementation enabling multiple concurrent clients, this approach has critical limitations:

1. **Thread Pool Bottleneck**: Multiple SSE clients sharing `asyncio.to_thread()` calls compete for limited thread pool slots (default: 32 threads)
2. **GIL Contention**: Synchronous Redis operations under high concurrency cause Python GIL contention
3. **No Connection Reuse Optimization**: Current implementation doesn't leverage Redis's native async connection pooling
4. **Limited Scalability**: Thread-based approach caps throughput at ~500 req/s regardless of I/O capacity

### 1.2 Solution Overview

Migrate to **native async FalkorDB client** (`falkordb.asyncio.FalkorDB`) with `redis.asyncio.BlockingConnectionPool` for proper connection management:

- **True Async I/O**: Native `await self.graph.query()` without thread pool overhead
- **Connection Pooling**: Configurable pool with blocking semantics for high concurrency
- **Lifecycle Management**: Proper async initialization and graceful shutdown
- **Retry Logic**: Exponential backoff for transient failures

### 1.3 Key Deliverables

| Deliverable | Description |
|------------|-------------|
| Async FalkorDB Client | Native async client with `BlockingConnectionPool` |
| Pool Configuration | Environment variables for pool tuning (min/max/timeout) |
| Async Initialization | `init_async()` method for proper startup sequence |
| Graceful Shutdown | `close()` method with `await pool.aclose()` |
| Retry Logic | Exponential backoff for `ConnectionError` and `BusyLoadingError` |
| Pool Monitoring | Utilization metrics and health endpoint integration |

---

## 2. User Stories

### US-1: Concurrent Database Access

```
AS an SSE client user
I WANT multiple concurrent database queries to execute in parallel
SO THAT heavy queries from one client don't block others

GIVEN 10 SSE clients are connected simultaneously
WHEN each client executes a database query
THEN all queries execute concurrently without blocking each other
AND total execution time is roughly equal to the longest single query (not sum)
```

**Acceptance Criteria:**
- [ ] 10 concurrent queries complete in ~1x single query time (not 10x)
- [ ] No `asyncio.to_thread()` calls in query execution path
- [ ] Connection pool utilization visible in health endpoint
- [ ] No thread pool exhaustion warnings under normal load

### US-2: Connection Pool Configuration

```
AS a system administrator
I WANT to configure database connection pool via environment variables
SO THAT I can tune performance for my deployment's workload

GIVEN I set DB_POOL_MAX_SIZE=50 environment variable
WHEN the server starts
THEN the connection pool is configured with 50 max connections
AND this is reflected in the server startup logs
```

**Acceptance Criteria:**
- [ ] `DB_POOL_MIN_SIZE` environment variable (default: 5)
- [ ] `DB_POOL_MAX_SIZE` environment variable (default: 20)
- [ ] `DB_POOL_TIMEOUT` environment variable (default: 10s)
- [ ] Configuration logged at startup
- [ ] Invalid values result in clear error messages

### US-3: Graceful Server Lifecycle

```
AS a system administrator
I WANT database connections to initialize on startup and close on shutdown
SO THAT resources are properly managed and no connections leak

GIVEN the MCP server is starting up
WHEN the lifespan startup event fires
THEN the connection pool is initialized with configured size
AND connection count is logged

GIVEN the MCP server is shutting down
WHEN the lifespan shutdown event fires
THEN all database connections are gracefully closed
AND no connection errors appear in logs
```

**Acceptance Criteria:**
- [ ] Pool initialized during server startup (lifespan event)
- [ ] Startup logs show "connection pool initialized" with pool size
- [ ] Shutdown logs show "connection pool closed"
- [ ] No connection errors during graceful shutdown
- [ ] Pending queries complete before pool closes (with timeout)

### US-4: Transient Error Recovery

```
AS a user executing database queries
I WANT transient connection errors to be automatically retried
SO THAT temporary network issues don't cause request failures

GIVEN a temporary network glitch occurs
WHEN a database query fails with ConnectionError
THEN the system retries with exponential backoff
AND the query eventually succeeds without user intervention
```

**Acceptance Criteria:**
- [ ] `ConnectionError` triggers automatic retry
- [ ] `BusyLoadingError` triggers automatic retry
- [ ] Exponential backoff: 0.1s, 0.2s, 0.4s, 0.8s (max 3 retries)
- [ ] Retry attempts logged with warning level
- [ ] Permanent failures after max retries return error to caller

---

## 3. Functional Requirements

### FR-001: Async FalkorDB Client Migration

**Description:** Migrate from sync `falkordb.FalkorDB` to async `falkordb.asyncio.FalkorDB`.

**Requirements:**
- FR-001.1: Import MUST change from `from falkordb import FalkorDB` to `from falkordb.asyncio import FalkorDB`
- FR-001.2: Client MUST use `redis.asyncio.BlockingConnectionPool` for connection management
- FR-001.3: All `asyncio.to_thread()` calls in `_execute_cypher()` MUST be removed
- FR-001.4: Query execution MUST use native `await self.graph.query()`
- FR-001.5: Existing sync methods MUST be converted to async

**Current Code (to be changed):**
```python
# File: src/zapomni_db/falkordb_client.py, Line 18
from falkordb import FalkorDB

# Line 338 (approx)
result = await asyncio.to_thread(self.graph.query, cypher, parameters)
```

### FR-002: Connection Pool Configuration

**Description:** Implement configurable connection pool with environment variables.

**Requirements:**
- FR-002.1: Pool MUST support `DB_POOL_MIN_SIZE` environment variable (default: 5)
- FR-002.2: Pool MUST support `DB_POOL_MAX_SIZE` environment variable (default: 20)
- FR-002.3: Pool MUST support `DB_POOL_TIMEOUT` environment variable (default: 10.0 seconds)
- FR-002.4: Pool MUST support `DB_SOCKET_TIMEOUT` environment variable (default: 30.0 seconds)
- FR-002.5: Pool MUST support `DB_HEALTH_CHECK_INTERVAL` environment variable (default: 30 seconds)
- FR-002.6: Configuration MUST be validated at startup (min <= max, positive values)
- FR-002.7: Invalid configuration MUST result in clear error message and server exit

**Pool Configuration:**
```python
pool = BlockingConnectionPool(
    host=self.host,
    port=self.port,
    password=self.password,
    max_connections=self.pool_max_size,  # DB_POOL_MAX_SIZE
    timeout=self.pool_timeout,  # DB_POOL_TIMEOUT - wait for available connection
    socket_timeout=self.socket_timeout,  # DB_SOCKET_TIMEOUT - query execution timeout
    socket_connect_timeout=5.0,  # Fixed - initial connection timeout
    health_check_interval=self.health_check_interval,  # DB_HEALTH_CHECK_INTERVAL
    decode_responses=True
)
```

### FR-003: Async Initialization Pattern

**Description:** Implement async initialization for connection pool.

**Requirements:**
- FR-003.1: `__init__()` MUST only store configuration (no network I/O)
- FR-003.2: Client MUST provide `async def init_async()` method for pool initialization
- FR-003.3: `init_async()` MUST create `BlockingConnectionPool`
- FR-003.4: `init_async()` MUST create async `FalkorDB` client with pool
- FR-003.5: `init_async()` MUST test connection with `RETURN 1` query
- FR-003.6: `init_async()` MUST initialize schema (via SchemaManager)
- FR-003.7: `init_async()` MUST set `_initialized = True` flag
- FR-003.8: Query methods MUST raise `ConnectionError` if called before `init_async()`

**Initialization Pattern:**
```python
class FalkorDBClient:
    def __init__(self, host, port, ...):
        # Store config only - no network I/O
        self.host = host
        self.port = port
        self._pool = None
        self._initialized = False

    async def init_async(self) -> None:
        """Initialize async connection pool. Must be called before use."""
        if self._initialized:
            return

        # Create pool and client
        self._pool = BlockingConnectionPool(...)
        self._db = AsyncFalkorDB(connection_pool=self._pool)
        self.graph = self._db.select_graph(self.graph_name)

        # Test connection
        await self.graph.query("RETURN 1")

        # Initialize schema
        await self._init_schema_async()

        self._initialized = True
```

### FR-004: Lifecycle Management

**Description:** Implement proper lifecycle management for connection pool.

**Requirements:**
- FR-004.1: Client MUST provide `async def close()` method
- FR-004.2: `close()` MUST call `await self._pool.aclose()`
- FR-004.3: `close()` MUST set `_closed = True` flag
- FR-004.4: `close()` MUST set `_initialized = False` flag
- FR-004.5: `close()` MUST log connection pool closure
- FR-004.6: `close()` MUST be idempotent (safe to call multiple times)
- FR-004.7: Pending queries SHOULD complete before pool closes (with 10s timeout)

### FR-005: Retry Logic with Exponential Backoff

**Description:** Implement retry logic for transient database errors.

**Requirements:**
- FR-005.1: Retry MUST trigger on `ConnectionError` exceptions
- FR-005.2: Retry MUST trigger on `redis.BusyLoadingError` exceptions
- FR-005.3: Retry MUST NOT trigger on `QueryError` (invalid Cypher)
- FR-005.4: Retry MUST use exponential backoff: initial=0.1s, multiplier=2.0, max=2.0s
- FR-005.5: Maximum retry attempts MUST be 3 (configurable)
- FR-005.6: Each retry attempt MUST be logged at WARNING level
- FR-005.7: Final failure MUST be logged at ERROR level and re-raise exception

**Backoff Sequence:**
```
Attempt 1: Execute immediately
Attempt 2: Wait 0.1s, then execute
Attempt 3: Wait 0.2s, then execute
Attempt 4: Wait 0.4s, then execute (if max_retries=4)
```

### FR-006: Pool Monitoring and Metrics

**Description:** Implement monitoring for connection pool health.

**Requirements:**
- FR-006.1: Client MUST provide `async def get_pool_stats()` method
- FR-006.2: Stats MUST include: `max_connections`, `active_connections`, `total_queries`
- FR-006.3: Stats MUST include: `utilization_percent` (active/max * 100)
- FR-006.4: Warning MUST be logged when utilization exceeds 80%
- FR-006.5: Pool stats MUST be available via health endpoint (`/health`)
- FR-006.6: Stats MUST include: `initialized`, `closed` flags

**Stats Response:**
```json
{
  "max_connections": 20,
  "active_connections": 12,
  "total_queries": 15234,
  "utilization_percent": 60.0,
  "initialized": true,
  "closed": false
}
```

### FR-007: Server Integration

**Description:** Integrate connection pool with MCP server lifecycle.

**Requirements:**
- FR-007.1: Pool initialization MUST occur during server startup (lifespan event)
- FR-007.2: Pool closure MUST occur during server shutdown (lifespan event)
- FR-007.3: `__main__.py` MUST call `await db_client.init_async()` after instantiation
- FR-007.4: `server.py` shutdown MUST call `await db_client.close()`
- FR-007.5: Pool stats MUST be included in `/health` endpoint response

---

## 4. Non-Functional Requirements

### NFR-001: Performance

| Metric | Requirement | Notes |
|--------|------------|-------|
| Query latency overhead | < 5ms | Async vs sync overhead |
| Concurrent query capacity | 50+ simultaneous | Limited by pool size |
| Throughput | > 2000 req/s | Under optimal conditions |
| Connection acquisition | < 100ms P95 | When pool has available connections |
| Memory per connection | < 60KB | ~50KB Redis + ~10KB FalkorDB |

### NFR-002: Reliability

| Metric | Requirement |
|--------|------------|
| Retry success rate | > 95% for transient errors |
| Pool recovery time | < 5s after connection loss |
| Zero connection leaks | Verified via monitoring |
| Graceful degradation | Queue requests when pool exhausted |

### NFR-003: Scalability

| Dimension | Requirement |
|-----------|------------|
| Connection pool range | 5-200 connections |
| Concurrent SSE clients | 100+ with pool_size=50 |
| Linear memory scaling | ~60KB * pool_size |

### NFR-004: Observability

| Metric | Requirement |
|--------|------------|
| Pool utilization | Visible in `/health` endpoint |
| Connection errors | Logged with context |
| Retry attempts | Logged with backoff timing |
| Startup/shutdown | Logged with pool configuration |

---

## 5. Acceptance Criteria

### AC-1: Async Client Migration

- [ ] Import changed to `from falkordb.asyncio import FalkorDB`
- [ ] `BlockingConnectionPool` used for connection management
- [ ] All `asyncio.to_thread()` removed from query path
- [ ] Native `await self.graph.query()` used
- [ ] All existing tests pass with async client

### AC-2: Configuration

- [ ] `DB_POOL_MIN_SIZE` environment variable works (default: 5)
- [ ] `DB_POOL_MAX_SIZE` environment variable works (default: 20)
- [ ] `DB_POOL_TIMEOUT` environment variable works (default: 10s)
- [ ] Configuration logged at startup
- [ ] Invalid values rejected with clear error

### AC-3: Lifecycle Management

- [ ] Pool initialized during server startup
- [ ] Pool closed during server shutdown
- [ ] No connection errors in graceful shutdown
- [ ] Pending queries complete before close
- [ ] Repeated `close()` calls are safe

### AC-4: Retry Logic

- [ ] `ConnectionError` triggers retry with backoff
- [ ] `BusyLoadingError` triggers retry with backoff
- [ ] Max 3 retry attempts (configurable)
- [ ] Retry attempts logged at WARNING level
- [ ] Final failure logged at ERROR level

### AC-5: Performance Under Load

- [ ] 10 concurrent queries execute in parallel (not serialized)
- [ ] Pool utilization visible in `/health` endpoint
- [ ] Warning logged when utilization > 80%
- [ ] No thread pool warnings under normal load
- [ ] Throughput improvement measurable vs current implementation

---

## 6. Constraints & Assumptions

### 6.1 Technical Constraints

| Constraint | Description |
|-----------|-------------|
| TC-1 | Must use `falkordb.asyncio.FalkorDB` (official async client) |
| TC-2 | Must use `redis.asyncio.BlockingConnectionPool` (not standard `ConnectionPool`) |
| TC-3 | Schema initialization may remain sync (one-time operation) |
| TC-4 | Python 3.11+ required for full async support |
| TC-5 | FalkorDB SDK version >= 1.2.0 required |

### 6.2 Design Constraints

| Constraint | Description |
|-----------|-------------|
| DC-1 | Single pool shared across all sessions |
| DC-2 | Pool configuration immutable after initialization |
| DC-3 | Backward compatibility with existing query interfaces |
| DC-4 | No changes to Cypher query syntax |

### 6.3 Assumptions

| Assumption | Description |
|-----------|-------------|
| A-1 | FalkorDB async client is stable and production-ready |
| A-2 | `BlockingConnectionPool` handles connection recovery automatically |
| A-3 | Default pool size (20) sufficient for initial deployment |
| A-4 | Single FalkorDB server (no clustering) |

---

## 7. Out of Scope

| Item | Reason | Future Phase |
|------|--------|--------------|
| OS-1: Connection clustering | Single-server deployment sufficient | Phase 6 |
| OS-2: Read replicas | Not needed for current scale | Phase 6 |
| OS-3: Connection encryption (TLS) | Local deployment uses unencrypted | Phase 5 |
| OS-4: Query caching layer | Semantic cache sufficient | N/A |
| OS-5: Connection pool auto-scaling | Fixed pool size sufficient | Phase 6 |

---

## 8. Dependencies

### 8.1 Internal Dependencies

| Dependency | Status | Notes |
|-----------|--------|-------|
| SSE Transport | Complete | Enables concurrent clients |
| FalkorDBClient | Exists | Target for refactoring |
| SchemaManager | Exists | May need async wrapper |
| ZapomniSettings | Exists | Add pool config fields |

### 8.2 External Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| falkordb | >= 1.2.0 | Async client support |
| redis | >= 5.0.0 | BlockingConnectionPool |

---

## 9. Risks & Mitigations

### R-1: Schema Manager Compatibility (MEDIUM)

**Risk:** Current `SchemaManager` uses sync operations that may not work with async graph.

**Impact:** Medium - Schema initialization could fail.

**Mitigation:**
1. Keep separate sync FalkorDB instance for schema initialization (one-time operation)
2. Wrap schema init in `asyncio.to_thread()` as fallback
3. Test schema initialization with async client

### R-2: Connection Pool Exhaustion (MEDIUM)

**Risk:** 50 SSE clients x 3 queries = 150 concurrent requests could exhaust pool.

**Impact:** Medium - Requests queue or timeout.

**Mitigation:**
1. Use `BlockingConnectionPool` (queues instead of errors)
2. Set appropriate timeout (10s) to prevent indefinite waits
3. Monitor utilization and alert at 80%
4. Document pool sizing guidelines

### R-3: Backward Compatibility (LOW)

**Risk:** Existing code may assume sync behavior.

**Impact:** Low - Well-defined interface.

**Mitigation:**
1. Keep same public method signatures
2. Update all call sites to use await
3. Add deprecation warnings for sync fallbacks
4. Comprehensive test coverage

### R-4: Query Timeout Handling (LOW)

**Risk:** Long-running queries may block connections.

**Impact:** Low - Affects pool utilization.

**Mitigation:**
1. Set `socket_timeout=30.0` for query execution
2. Use FalkorDB query timeout parameter
3. Monitor query duration metrics

---

## 10. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Query throughput | > 2000 req/s | Load testing |
| Concurrent query support | 50+ simultaneous | Load testing |
| Latency P95 | < 100ms | Load testing |
| Pool utilization at peak | < 80% | Monitoring |
| Retry success rate | > 95% | Error tracking |
| Zero connection leaks | 0 leaked connections | Long-running tests |
| Test coverage | > 80% | Coverage reports |

---

## 11. Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DB_POOL_MIN_SIZE` | int | 5 | Minimum connections to maintain |
| `DB_POOL_MAX_SIZE` | int | 20 | Maximum connections allowed |
| `DB_POOL_TIMEOUT` | float | 10.0 | Seconds to wait for available connection |
| `DB_SOCKET_TIMEOUT` | float | 30.0 | Query execution timeout in seconds |
| `DB_HEALTH_CHECK_INTERVAL` | int | 30 | Seconds between connection health checks |

---

**End of Requirements Document**
