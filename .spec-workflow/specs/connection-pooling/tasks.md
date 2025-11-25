# Implementation Tasks - Connection Pooling for FalkorDB

**Spec Name:** connection-pooling
**Status:** Implementation Phase
**Created:** 2025-11-25
**Estimated Duration:** 2-3 days

---

## Overview

Total Tasks: **28** (including subtasks)
Estimated Effort: **~18 hours**

### Task Summary by Phase

| Phase | Description | Tasks | Priority | Est. Hours |
|-------|-------------|-------|----------|------------|
| 1 | Configuration | 5 | Critical | 2h |
| 2 | Async Client Migration | 8 | Critical | 6h |
| 3 | Retry Logic | 4 | High | 2h |
| 4 | Server Integration | 4 | High | 3h |
| 5 | Monitoring | 3 | Medium | 2h |
| 6 | Testing | 4 | High | 3h |

---

## Phase 1: Configuration (2 hours)

### 1.1 Pool Configuration Classes

- [ ] **1.1.1** Create PoolConfig dataclass
  - Files: `src/zapomni_db/pool_config.py` (NEW)
  - Implement:
    - `min_size`, `max_size`, `timeout`, `socket_timeout`, `health_check_interval` fields
    - Validation in `__post_init__()` (min <= max, positive values)
    - `from_env()` classmethod to load from environment variables
  - Est: 30 min

- [ ] **1.1.2** Create RetryConfig dataclass
  - Files: `src/zapomni_db/pool_config.py`
  - Implement:
    - `max_retries`, `initial_delay`, `max_delay`, `exponential_base` fields
    - Validation for positive values
    - Default values per spec (max_retries=3, initial_delay=0.1s)
  - Est: 15 min

### 1.2 Settings Integration

- [ ] **1.2.1** Add pool configuration to ZapomniSettings
  - Files: `src/zapomni_core/config.py`
  - Add fields:
    - `db_pool_min_size: int = 5`
    - `db_pool_max_size: int = 20`
    - `db_pool_timeout: float = 10.0`
    - `db_socket_timeout: float = 30.0`
    - `db_health_check_interval: int = 30`
  - Add retry fields:
    - `db_max_retries: int = 3`
    - `db_retry_initial_delay: float = 0.1`
    - `db_retry_max_delay: float = 2.0`
  - Est: 30 min

- [ ] **1.2.2** Add environment variable documentation
  - Files: `src/zapomni_core/config.py`
  - Add Field descriptions for each pool/retry setting
  - Document env var names in docstring
  - Est: 15 min

- [ ] **1.2.3** Update existing falkordb_pool_size to use new default
  - Files: `src/zapomni_core/config.py`
  - Ensure `falkordb_pool_size` default is 20 (already done in SSE transport)
  - Add deprecation note if needed (use db_pool_max_size instead)
  - Est: 15 min

---

## Phase 2: Async Client Migration (6 hours)

### 2.1 Import Changes

- [ ] **2.1.1** Update imports in FalkorDBClient
  - Files: `src/zapomni_db/falkordb_client.py`
  - Change line 18: `from falkordb import FalkorDB` to:
    ```python
    from redis.asyncio import BlockingConnectionPool
    from falkordb.asyncio import FalkorDB as AsyncFalkorDB
    from falkordb import FalkorDB as SyncFalkorDB  # For schema init
    ```
  - Import `PoolConfig`, `RetryConfig` from `pool_config.py`
  - Est: 15 min

### 2.2 Constructor Refactoring

- [ ] **2.2.1** Refactor `__init__` to store config only (no network I/O)
  - Files: `src/zapomni_db/falkordb_client.py`
  - Lines: 47-114
  - Changes:
    - Add `pool_config: Optional[PoolConfig]` parameter
    - Add `retry_config: Optional[RetryConfig]` parameter
    - Remove call to `self._init_connection()` from `__init__`
    - Initialize state variables: `self._pool = None`, `self._db = None`
    - Keep validation of host, port, db parameters
  - Est: 30 min

- [ ] **2.2.2** Add state tracking attributes
  - Files: `src/zapomni_db/falkordb_client.py`
  - Add to `__init__`:
    ```python
    self._pool: Optional[BlockingConnectionPool] = None
    self._db: Optional[AsyncFalkorDB] = None
    self._initialized = False
    self._closed = False
    self._total_retries = 0
    ```
  - Est: 15 min

### 2.3 Async Initialization

- [ ] **2.3.1** Implement `init_async()` method
  - Files: `src/zapomni_db/falkordb_client.py`
  - Insert after `__init__` (around line 115)
  - Implement:
    ```python
    async def init_async(self) -> None:
        """Initialize async connection pool. Must be called before use."""
        if self._initialized:
            return
        if self._closed:
            raise ConnectionError("Client has been closed")

        # Create BlockingConnectionPool
        self._pool = BlockingConnectionPool(
            host=self.host,
            port=self.port,
            password=self.password,
            max_connections=self.pool_config.max_size,
            timeout=self.pool_config.timeout,
            socket_timeout=self.pool_config.socket_timeout,
            socket_connect_timeout=5.0,
            health_check_interval=self.pool_config.health_check_interval,
            decode_responses=True,
        )

        # Create async FalkorDB client
        self._db = AsyncFalkorDB(connection_pool=self._pool)
        self.graph = self._db.select_graph(self.graph_name)

        # Test connection
        await self.graph.query("RETURN 1")

        # Initialize schema
        await self._init_schema_async()

        self._initialized = True
        self._logger.info("connection_pool_initialized", ...)
    ```
  - Est: 45 min

- [ ] **2.3.2** Implement `_init_schema_async()` method
  - Files: `src/zapomni_db/falkordb_client.py`
  - Insert after `init_async()`
  - Implement:
    - Use sync FalkorDB for schema (one-time operation)
    - Wrap SchemaManager.init_schema() in `asyncio.to_thread()`
    ```python
    async def _init_schema_async(self) -> None:
        """Initialize schema using sync client."""
        sync_db = SyncFalkorDB(host=self.host, port=self.port, password=self.password)
        sync_graph = sync_db.select_graph(self.graph_name)
        self._schema_manager = SchemaManager(graph=sync_graph, logger=self._logger)
        await asyncio.to_thread(self._schema_manager.init_schema)
    ```
  - Est: 30 min

### 2.4 Query Execution Migration

- [ ] **2.4.1** Update `_execute_cypher()` to native async
  - Files: `src/zapomni_db/falkordb_client.py`
  - Location: Around lines 1008-1054
  - Changes:
    - Remove `await asyncio.to_thread(self.graph.query, ...)` wrapper
    - Replace with: `result = await self._execute_with_retry(query, parameters)`
    - Add initialization check at start:
      ```python
      if not self._initialized:
          raise ConnectionError("Client not initialized. Call init_async() first.")
      ```
    - Track `_active_connections` before/after query
  - Est: 45 min

- [ ] **2.4.2** Remove old `_init_connection()` method
  - Files: `src/zapomni_db/falkordb_client.py`
  - Lines: 116-137
  - Action: Delete sync `_init_connection()` method (replaced by `init_async()`)
  - Keep `_init_schema()` as fallback or remove if fully async
  - Est: 15 min

### 2.5 Close Method

- [ ] **2.5.1** Convert `close()` to async and implement pool cleanup
  - Files: `src/zapomni_db/falkordb_client.py`
  - Location: Around lines 1055-1075
  - Change: `def close(self)` to `async def close(self)`
  - Implement:
    ```python
    async def close(self) -> None:
        """Close connection pool (idempotent)."""
        if self._closed:
            return

        self._logger.info("closing_connection_pool")

        # Wait for pending queries (up to 10s)
        if self._active_connections > 0:
            for _ in range(100):
                if self._active_connections == 0:
                    break
                await asyncio.sleep(0.1)

        # Close pool
        if self._pool:
            await self._pool.aclose()

        self._closed = True
        self._initialized = False
        self._pool = None
        self._db = None
        self.graph = None
    ```
  - Est: 30 min

---

## Phase 3: Retry Logic (2 hours)

### 3.1 Retry Implementation

- [ ] **3.1.1** Implement `_execute_with_retry()` method
  - Files: `src/zapomni_db/falkordb_client.py`
  - Insert before `_execute_cypher()`
  - Implement:
    ```python
    async def _execute_with_retry(
        self,
        query: str,
        parameters: Dict[str, Any],
    ) -> QueryResult:
        """Execute query with exponential backoff retry."""
        import redis

        delay = self.retry_config.initial_delay

        for attempt in range(self.retry_config.max_retries + 1):
            try:
                result = await self.graph.query(query, parameters)
                return self._convert_result(result)

            except (ConnectionError, redis.BusyLoadingError, OSError) as e:
                if attempt < self.retry_config.max_retries:
                    self._total_retries += 1
                    self._logger.warning(
                        "query_retry",
                        attempt=attempt + 1,
                        max_retries=self.retry_config.max_retries,
                        delay=delay,
                        error=str(e),
                    )
                    await asyncio.sleep(delay)
                    delay = min(
                        delay * self.retry_config.exponential_base,
                        self.retry_config.max_delay,
                    )
                else:
                    self._logger.error(
                        "query_failed_after_retries",
                        attempts=attempt + 1,
                        error=str(e),
                    )
                    raise

            except Exception as e:
                raise QueryError(f"Query failed: {e}")
    ```
  - Est: 45 min

- [ ] **3.1.2** Implement `_convert_result()` helper method
  - Files: `src/zapomni_db/falkordb_client.py`
  - Extract result conversion logic from `_execute_cypher()`:
    ```python
    def _convert_result(self, result) -> QueryResult:
        """Convert FalkorDB result to QueryResult."""
        rows = []
        if result.result_set:
            for record in result.result_set:
                row_dict = {}
                if result.header:
                    for i, col_header in enumerate(result.header):
                        col_name = col_header[1] if len(col_header) > 1 else f"col_{i}"
                        row_dict[col_name] = record[i] if i < len(record) else None
                rows.append(row_dict)

        return QueryResult(
            rows=rows,
            row_count=len(rows),
            execution_time_ms=int(result.run_time_ms) if hasattr(result, 'run_time_ms') else 0,
        )
    ```
  - Est: 30 min

### 3.2 Pool Utilization Check

- [ ] **3.2.1** Implement `_check_pool_utilization()` method
  - Files: `src/zapomni_db/falkordb_client.py`
  - Implement:
    ```python
    async def _check_pool_utilization(self) -> None:
        """Log warning if pool utilization exceeds threshold."""
        if self.pool_config.max_size > 0:
            utilization = (self._active_connections / self.pool_config.max_size) * 100
            if utilization > 80 and not self._utilization_warning_logged:
                self._logger.warning(
                    "high_pool_utilization",
                    utilization_percent=round(utilization, 1),
                    active=self._active_connections,
                    max=self.pool_config.max_size,
                )
                self._utilization_warning_logged = True
            elif utilization <= 60:
                self._utilization_warning_logged = False
    ```
  - Est: 15 min

- [ ] **3.2.2** Add utilization check call in `_execute_cypher()`
  - Files: `src/zapomni_db/falkordb_client.py`
  - Add at start of `_execute_cypher()`:
    ```python
    await self._check_pool_utilization()
    ```
  - Est: 5 min

---

## Phase 4: Server Integration (3 hours)

### 4.1 Startup Integration

- [ ] **4.1.1** Update `__main__.py` to call `init_async()`
  - Files: `src/zapomni_mcp/__main__.py`
  - Location: After FalkorDBClient instantiation (around line 170-177)
  - Changes:
    ```python
    # Create pool config from settings
    from zapomni_db.pool_config import PoolConfig, RetryConfig

    pool_config = PoolConfig(
        min_size=settings.db_pool_min_size,
        max_size=settings.db_pool_max_size,
        timeout=settings.db_pool_timeout,
        socket_timeout=settings.db_socket_timeout,
        health_check_interval=settings.db_health_check_interval,
    )

    retry_config = RetryConfig(
        max_retries=settings.db_max_retries,
        initial_delay=settings.db_retry_initial_delay,
        max_delay=settings.db_retry_max_delay,
    )

    db_client = FalkorDBClient(
        host=settings.falkordb_host,
        port=settings.falkordb_port,
        graph_name=settings.graph_name,
        password=settings.falkordb_password.get_secret_value() if settings.falkordb_password else None,
        pool_config=pool_config,
        retry_config=retry_config,
    )

    # NEW: Async initialization
    await db_client.init_async()
    logger.info("FalkorDB connection pool initialized")
    ```
  - Est: 45 min

### 4.2 Shutdown Integration

- [ ] **4.2.1** Update `server.py` shutdown to close pool
  - Files: `src/zapomni_mcp/server.py`
  - Location: `_graceful_shutdown_sse()` method (around line 536)
  - Add database cleanup step:
    ```python
    # Step 2.5: Close database connection pool
    if hasattr(self, '_core_engine') and hasattr(self._core_engine, 'db_client'):
        db_client = self._core_engine.db_client
        if db_client and not db_client._closed:
            self._logger.info("closing_database_pool")
            await db_client.close()
            self._logger.info("database_pool_closed")
    ```
  - Est: 30 min

- [ ] **4.2.2** Store db_client reference for shutdown access
  - Files: `src/zapomni_mcp/__main__.py`, `src/zapomni_mcp/server.py`
  - Ensure MCPServer has access to db_client for shutdown
  - Options:
    - Pass db_client to MCPServer constructor
    - Store on MemoryProcessor and access via core_engine
  - Est: 30 min

### 4.3 Health Endpoint Integration

- [ ] **4.3.1** Add pool stats to health endpoint response
  - Files: `src/zapomni_mcp/sse_transport.py`
  - Location: `handle_health()` function
  - Changes:
    ```python
    async def handle_health(request: Request) -> Response:
        # ... existing health info ...

        # Add database pool stats
        if hasattr(request.app.state, 'db_client'):
            db_client = request.app.state.db_client
            pool_stats = await db_client.get_pool_stats()
            health_info["database"] = {
                "pool": {
                    "max_connections": pool_stats["max_connections"],
                    "active_connections": pool_stats["active_connections"],
                    "utilization_percent": pool_stats["utilization_percent"],
                    "initialized": pool_stats["initialized"],
                }
            }

        return JSONResponse(health_info)
    ```
  - Est: 30 min

---

## Phase 5: Monitoring (2 hours)

### 5.1 Pool Statistics

- [ ] **5.1.1** Implement `get_pool_stats()` method
  - Files: `src/zapomni_db/falkordb_client.py`
  - Implement:
    ```python
    async def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics for monitoring."""
        utilization = 0.0
        if self.pool_config.max_size > 0:
            utilization = (self._active_connections / self.pool_config.max_size) * 100

        return {
            "max_connections": self.pool_config.max_size,
            "active_connections": self._active_connections,
            "total_queries": self._total_queries,
            "total_retries": self._total_retries,
            "utilization_percent": round(utilization, 1),
            "initialized": self._initialized,
            "closed": self._closed,
        }
    ```
  - Est: 30 min

### 5.2 Logging Enhancements

- [ ] **5.2.1** Add structured logging for pool events
  - Files: `src/zapomni_db/falkordb_client.py`
  - Add logging at key points:
    - `connection_pool_initialized` - INFO - pool_max_size, timeout
    - `high_pool_utilization` - WARNING - utilization_percent, active, max
    - `query_retry` - WARNING - attempt, delay, error
    - `query_failed_after_retries` - ERROR - attempts, error
    - `connection_pool_closed` - INFO
  - Est: 30 min

- [ ] **5.2.2** Add pool stats to existing `get_stats()` method
  - Files: `src/zapomni_db/falkordb_client.py`
  - Location: `get_stats()` method
  - Add pool stats to response:
    ```python
    stats["pool"] = {
        "max_connections": self.pool_config.max_size,
        "active_connections": self._active_connections,
        "utilization_percent": ...,
    }
    ```
  - Est: 15 min

---

## Phase 6: Testing (3 hours)

### 6.1 Unit Tests

- [ ] **6.1.1** Create unit tests for PoolConfig and RetryConfig
  - Files: `tests/unit/test_pool_config.py` (NEW)
  - Test:
    - PoolConfig validation (min <= max, positive values)
    - PoolConfig.from_env() loads environment variables
    - RetryConfig validation
    - Default values
  - Est: 45 min

- [ ] **6.1.2** Create unit tests for retry logic
  - Files: `tests/unit/test_retry_logic.py` (NEW)
  - Test:
    - Retry on ConnectionError
    - Retry on BusyLoadingError
    - No retry on QueryError
    - Exponential backoff timing
    - Max retries respected
  - Est: 45 min

### 6.2 Integration Tests

- [ ] **6.2.1** Create integration tests for connection pool
  - Files: `tests/integration/test_connection_pool.py` (NEW)
  - Test:
    - Client lifecycle: `__init__` -> `init_async()` -> queries -> `close()`
    - Concurrent queries execute in parallel
    - Pool stats reflect actual state
    - Repeated close() calls are safe
  - Est: 1 hour

- [ ] **6.2.2** Create load test for concurrent queries
  - Files: `tests/integration/test_pool_load.py` (NEW)
  - Test:
    - 10 concurrent queries complete successfully
    - 50 concurrent queries with pool_size=20 queues correctly
    - Pool utilization warning logged at > 80%
    - Throughput improvement vs baseline
  - Est: 45 min

---

## Task Dependencies

```
Phase 1 (Config)
├── 1.1.1 (PoolConfig) ──────────────────────────────────────────────┐
├── 1.1.2 (RetryConfig) ─────────────────────────────────────────────┤
└── 1.2.1 (Settings) ────────────────────────────────────────────────┤
                                                                     │
Phase 2 (Async Migration)                                            │
├── 2.1.1 (Imports) ─────────────────────────────────────────────────┤
├── 2.2.1 (__init__ refactor) ───────────────────────────────────────┤
├── 2.3.1 (init_async) ──────────────────────────────────────────────┤
├── 2.3.2 (_init_schema_async) ──────────────────────────────────────┤
├── 2.4.1 (_execute_cypher) ─────────────────────────────────────────┤
└── 2.5.1 (close) ───────────────────────────────────────────────────┤
                                                                     │
Phase 3 (Retry Logic)                                                │
├── 3.1.1 (_execute_with_retry) ─────────────────────────────────────┤
└── 3.2.1 (_check_pool_utilization) ─────────────────────────────────┤
                                                                     │
Phase 4 (Integration)                                                │
├── 4.1.1 (__main__.py init) ────────────────────────────────────────┤
├── 4.2.1 (server.py shutdown) ──────────────────────────────────────┤
└── 4.3.1 (health endpoint) ─────────────────────────────────────────┤
                                                                     │
Phase 5 (Monitoring)                                                 │
├── 5.1.1 (get_pool_stats) ──────────────────────────────────────────┤
└── 5.2.1 (logging) ─────────────────────────────────────────────────┤
                                                                     │
Phase 6 (Testing)                                                    │
├── 6.1.1 (unit tests) ──────────────────────────────────────────────┤
├── 6.1.2 (retry tests) ─────────────────────────────────────────────┤
├── 6.2.1 (integration tests) ───────────────────────────────────────┤
└── 6.2.2 (load tests) ──────────────────────────────────────────────┘
```

---

## Acceptance Checklist

### Phase 1 Complete When:
- [ ] `PoolConfig` class created with validation
- [ ] `RetryConfig` class created with defaults
- [ ] Environment variables documented and loaded
- [ ] Settings fields added to ZapomniSettings

### Phase 2 Complete When:
- [ ] Import changed to `falkordb.asyncio.FalkorDB`
- [ ] `BlockingConnectionPool` used for connections
- [ ] `__init__` does NOT create network connections
- [ ] `init_async()` creates pool and tests connection
- [ ] `_execute_cypher()` uses native `await graph.query()`
- [ ] `close()` is async and calls `pool.aclose()`

### Phase 3 Complete When:
- [ ] `ConnectionError` triggers retry with backoff
- [ ] `BusyLoadingError` triggers retry with backoff
- [ ] Max 3 retries (configurable)
- [ ] Retry attempts logged at WARNING level
- [ ] Pool utilization warning at > 80%

### Phase 4 Complete When:
- [ ] `__main__.py` calls `await db_client.init_async()`
- [ ] `server.py` calls `await db_client.close()` on shutdown
- [ ] `/health` endpoint includes pool stats
- [ ] Startup logs show "connection_pool_initialized"
- [ ] Shutdown logs show "connection_pool_closed"

### Phase 5 Complete When:
- [ ] `get_pool_stats()` returns monitoring metrics
- [ ] Structured logging for all pool events
- [ ] `get_stats()` includes pool information

### Phase 6 Complete When:
- [ ] Unit tests pass for PoolConfig, RetryConfig
- [ ] Unit tests pass for retry logic
- [ ] Integration tests verify full lifecycle
- [ ] Load tests show concurrent query support

---

## Critical Path

The critical path for implementation:

```
1.1.1 (PoolConfig) -> 2.1.1 (Imports) -> 2.3.1 (init_async) ->
2.4.1 (_execute_cypher) -> 3.1.1 (retry) -> 4.1.1 (__main__.py) ->
6.2.1 (integration tests)
```

**Estimated Critical Path Duration:** 8-10 hours

---

## Notes

1. **Backward Compatibility**: Existing query method signatures unchanged
2. **Schema Init**: Keep sync for one-time schema initialization
3. **Testing**: Run existing tests after Phase 2 to verify no regressions
4. **Monitoring**: Pool stats critical for production observability
5. **Documentation**: Update README after completion

---

## Risk Mitigation

| Risk | Mitigation | Task |
|------|------------|------|
| Schema manager incompatibility | Use sync client for schema only | 2.3.2 |
| Existing tests fail | Run tests after each phase | All |
| Pool exhaustion | Use BlockingConnectionPool (queues) | 2.3.1 |
| Startup order issues | Clear init_async() requirement | 4.1.1 |

---

**End of Tasks Document**
