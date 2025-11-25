# Design Document - Connection Pooling for FalkorDB

**Spec Name:** connection-pooling
**Status:** Design Phase
**Created:** 2025-11-25
**Author:** Specification Writer Agent

---

## 1. System Architecture

### 1.1 Current Architecture (Before)

```
+------------------------------------------------------------------+
|                     Current FalkorDBClient                        |
+------------------------------------------------------------------+
|                                                                   |
|   SSE Client 1 ──┐                                                |
|                  │                                                |
|   SSE Client 2 ──┼──> asyncio.to_thread() ──> Sync FalkorDB      |
|                  │         (Thread Pool)      (Single Connection) |
|   SSE Client N ──┘                                                |
|                                                                   |
|   Problems:                                                       |
|   - Thread pool bottleneck (32 threads default)                   |
|   - GIL contention under load                                     |
|   - No connection pooling optimization                            |
|   - Max ~500 req/s throughput                                     |
|                                                                   |
+------------------------------------------------------------------+
```

### 1.2 Proposed Architecture (After)

```
+------------------------------------------------------------------+
|                   Async FalkorDBClient with Pool                  |
+------------------------------------------------------------------+
|                                                                   |
|   SSE Client 1 ──┐                                                |
|                  │                                                |
|   SSE Client 2 ──┼──> Event Loop ──> BlockingConnectionPool      |
|                  │    (Native Async)   (20 connections default)   |
|   SSE Client N ──┘                                                |
|                                                                   |
|                            │                                      |
|                            v                                      |
|                    ┌───────────────┐                              |
|                    │ AsyncFalkorDB │                              |
|                    │  .select_graph()                             |
|                    │  .query()     │                              |
|                    └───────────────┘                              |
|                                                                   |
|   Benefits:                                                       |
|   - True async I/O (no thread pool)                               |
|   - Connection reuse via pool                                     |
|   - No GIL contention                                             |
|   - 2000-5000 req/s throughput                                    |
|                                                                   |
+------------------------------------------------------------------+
```

### 1.3 Component Interaction

```
+------------------------------------------------------------------+
|                      MCP Server Startup                           |
+------------------------------------------------------------------+
|                                                                   |
|   __main__.py                                                     |
|       │                                                           |
|       │ 1. Create FalkorDBClient (config only)                    |
|       v                                                           |
|   FalkorDBClient.__init__()                                       |
|       - Store host, port, pool_size                               |
|       - NO network I/O                                            |
|       │                                                           |
|       │ 2. Initialize async pool                                  |
|       v                                                           |
|   await db_client.init_async()                                    |
|       │                                                           |
|       ├──> Create BlockingConnectionPool                          |
|       │        max_connections=20                                 |
|       │        timeout=10.0                                       |
|       │                                                           |
|       ├──> Create AsyncFalkorDB(connection_pool=pool)             |
|       │                                                           |
|       ├──> self.graph = db.select_graph("zapomni_memory")         |
|       │                                                           |
|       ├──> await self.graph.query("RETURN 1")  # Test connection  |
|       │                                                           |
|       └──> await self._init_schema_async()                        |
|                                                                   |
|   3. Server running - queries use pool                            |
|       │                                                           |
|       v                                                           |
|   await self.graph.query(cypher, params)  # Native async          |
|                                                                   |
|   4. Server shutdown                                              |
|       │                                                           |
|       v                                                           |
|   await db_client.close()                                         |
|       └──> await self._pool.aclose()                              |
|                                                                   |
+------------------------------------------------------------------+
```

### 1.4 Connection Pool Lifecycle

```
                    ┌─────────────────────────────────────┐
                    │         Pool State Machine          │
                    └─────────────────────────────────────┘

     ┌──────────┐     init_async()     ┌─────────────┐
     │  CREATED │ ──────────────────>  │ INITIALIZED │
     └──────────┘                      └─────────────┘
          │                                   │
          │ (queries fail)                    │ (queries succeed)
          │                                   │
          v                                   v
    ConnectionError              ┌─────────────────────────┐
                                 │     ACTIVE (Running)    │
                                 │   - Pool acquiring      │
                                 │   - Connections in use  │
                                 │   - Health checks       │
                                 └─────────────────────────┘
                                              │
                                              │ close()
                                              v
                                      ┌──────────┐
                                      │  CLOSED  │
                                      └──────────┘
                                              │
                                              │ (queries fail)
                                              v
                                       ConnectionError
```

---

## 2. Component Design

### 2.1 FalkorDBClient Refactored

**File:** `src/zapomni_db/falkordb_client.py`

```python
"""
FalkorDBClient - Async database client for FalkorDB with connection pooling.

Migration from sync to async client for SSE concurrent connections.
"""

import asyncio
import structlog
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

from redis.asyncio import BlockingConnectionPool
from falkordb.asyncio import FalkorDB as AsyncFalkorDB
from falkordb import FalkorDB as SyncFalkorDB  # For schema init only

from zapomni_db.models import Memory, Chunk, SearchResult, QueryResult
from zapomni_db.exceptions import (
    ValidationError,
    DatabaseError,
    ConnectionError,
    QueryError,
)
from zapomni_db.schema_manager import SchemaManager

logger = structlog.get_logger(__name__)


class PoolConfig:
    """Connection pool configuration."""

    def __init__(
        self,
        min_size: int = 5,
        max_size: int = 20,
        timeout: float = 10.0,
        socket_timeout: float = 30.0,
        socket_connect_timeout: float = 5.0,
        health_check_interval: int = 30,
    ):
        # Validate
        if min_size < 1:
            raise ValidationError(f"min_size must be >= 1, got {min_size}")
        if max_size < min_size:
            raise ValidationError(f"max_size ({max_size}) must be >= min_size ({min_size})")
        if max_size > 200:
            raise ValidationError(f"max_size must be <= 200, got {max_size}")
        if timeout <= 0:
            raise ValidationError(f"timeout must be > 0, got {timeout}")

        self.min_size = min_size
        self.max_size = max_size
        self.timeout = timeout
        self.socket_timeout = socket_timeout
        self.socket_connect_timeout = socket_connect_timeout
        self.health_check_interval = health_check_interval

    @classmethod
    def from_env(cls) -> "PoolConfig":
        """Load pool configuration from environment variables."""
        import os
        return cls(
            min_size=int(os.getenv("DB_POOL_MIN_SIZE", "5")),
            max_size=int(os.getenv("DB_POOL_MAX_SIZE", "20")),
            timeout=float(os.getenv("DB_POOL_TIMEOUT", "10.0")),
            socket_timeout=float(os.getenv("DB_SOCKET_TIMEOUT", "30.0")),
            health_check_interval=int(os.getenv("DB_HEALTH_CHECK_INTERVAL", "30")),
        )


class RetryConfig:
    """Retry configuration for transient errors."""

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 0.1,
        max_delay: float = 2.0,
        exponential_base: float = 2.0,
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base


class FalkorDBClient:
    """
    Async FalkorDB client with connection pooling.

    Uses native async client for SSE concurrent connections.
    """

    DEFAULT_POOL_SIZE = 20
    DEFAULT_MAX_RETRIES = 3

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6381,
        db: int = 0,
        graph_name: str = "zapomni_memory",
        password: Optional[str] = None,
        pool_config: Optional[PoolConfig] = None,
        retry_config: Optional[RetryConfig] = None,
    ):
        """
        Initialize FalkorDB client configuration.

        Note: Does NOT connect to database. Call init_async() to establish connection.

        Args:
            host: FalkorDB host address
            port: FalkorDB port
            db: Database index (0-15)
            graph_name: Name of graph to use
            password: Optional Redis password
            pool_config: Connection pool configuration
            retry_config: Retry configuration for transient errors
        """
        # Validate parameters
        if not host or not host.strip():
            raise ValidationError("host cannot be empty")
        if port <= 0 or port > 65535:
            raise ValidationError(f"port must be in range 1-65535, got {port}")
        if db < 0 or db > 15:
            raise ValidationError(f"db must be in range 0-15, got {db}")

        # Store configuration
        self.host = host
        self.port = port
        self.db = db
        self.graph_name = graph_name
        self.password = password
        self.pool_config = pool_config or PoolConfig()
        self.retry_config = retry_config or RetryConfig()

        # State - uninitialized
        self._pool: Optional[BlockingConnectionPool] = None
        self._db: Optional[AsyncFalkorDB] = None
        self.graph = None
        self._schema_manager = None
        self._initialized = False
        self._closed = False

        # Monitoring
        self._active_connections = 0
        self._total_queries = 0
        self._total_retries = 0
        self._utilization_warning_logged = False

        self._logger = logger.bind(
            host=host,
            port=port,
            graph=graph_name,
            pool_max=self.pool_config.max_size,
        )

    async def init_async(self) -> None:
        """
        Initialize async connection pool.

        Must be called before any database operations.

        Raises:
            ConnectionError: If connection cannot be established
        """
        if self._initialized:
            self._logger.debug("already_initialized")
            return

        if self._closed:
            raise ConnectionError("Client has been closed. Create new instance.")

        try:
            # Create async connection pool
            self._pool = BlockingConnectionPool(
                host=self.host,
                port=self.port,
                password=self.password,
                max_connections=self.pool_config.max_size,
                timeout=self.pool_config.timeout,
                socket_timeout=self.pool_config.socket_timeout,
                socket_connect_timeout=self.pool_config.socket_connect_timeout,
                health_check_interval=self.pool_config.health_check_interval,
                decode_responses=True,
            )

            # Create async FalkorDB client
            self._db = AsyncFalkorDB(connection_pool=self._pool)
            self.graph = self._db.select_graph(self.graph_name)

            # Test connection
            await self.graph.query("RETURN 1")

            # Initialize schema (sync operation wrapped)
            await self._init_schema_async()

            self._initialized = True

            self._logger.info(
                "connection_pool_initialized",
                pool_max_size=self.pool_config.max_size,
                pool_timeout=self.pool_config.timeout,
                socket_timeout=self.pool_config.socket_timeout,
            )

        except Exception as e:
            self._logger.error("init_failed", error=str(e))
            # Cleanup partial init
            if self._pool:
                try:
                    await self._pool.aclose()
                except Exception:
                    pass
                self._pool = None
            raise ConnectionError(f"Failed to initialize FalkorDB client: {e}")

    async def _init_schema_async(self) -> None:
        """Initialize schema using sync client (one-time operation)."""
        # Use sync client for schema initialization
        # Schema init is one-time, so asyncio.to_thread is acceptable here
        sync_db = SyncFalkorDB(
            host=self.host,
            port=self.port,
            password=self.password,
        )
        sync_graph = sync_db.select_graph(self.graph_name)

        self._schema_manager = SchemaManager(
            graph=sync_graph,
            logger=self._logger,
        )

        # Run schema init in thread pool (one-time operation)
        await asyncio.to_thread(self._schema_manager.init_schema)

        self._logger.info("schema_initialized")

    async def _execute_cypher(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> QueryResult:
        """
        Execute Cypher query with retry logic.

        Args:
            query: Cypher query string
            parameters: Query parameters

        Returns:
            QueryResult with rows and metadata

        Raises:
            ConnectionError: If not initialized or connection fails
            QueryError: If query is invalid
        """
        if not self._initialized:
            raise ConnectionError("Client not initialized. Call init_async() first.")

        if self._closed:
            raise ConnectionError("Client has been closed.")

        # Check pool utilization
        await self._check_pool_utilization()

        # Track active connection
        self._active_connections += 1
        self._total_queries += 1

        try:
            return await self._execute_with_retry(query, parameters or {})
        finally:
            self._active_connections -= 1

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
                # Native async query execution
                result = await self.graph.query(query, parameters)

                # Convert to QueryResult
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

            except (ConnectionError, redis.BusyLoadingError, OSError) as e:
                # Retryable errors
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
                # Non-retryable errors (query syntax, etc.)
                self._logger.error("query_error", error=str(e), query=query[:100])
                raise QueryError(f"Query failed: {e}")

    async def _check_pool_utilization(self) -> None:
        """Check and log pool utilization warnings."""
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

    async def close(self) -> None:
        """
        Close connection pool and release resources.

        Safe to call multiple times (idempotent).
        """
        if self._closed:
            self._logger.debug("already_closed")
            return

        self._logger.info("closing_connection_pool")

        try:
            # Wait for pending queries (with timeout)
            if self._active_connections > 0:
                self._logger.info(
                    "waiting_for_pending_queries",
                    active=self._active_connections,
                )
                # Wait up to 10 seconds for pending queries
                for _ in range(100):  # 100 * 0.1s = 10s
                    if self._active_connections == 0:
                        break
                    await asyncio.sleep(0.1)

            # Close pool
            if self._pool:
                await self._pool.aclose()
                self._logger.info("connection_pool_closed")

        except Exception as e:
            self._logger.warning("close_error", error=str(e))

        finally:
            self._closed = True
            self._initialized = False
            self._pool = None
            self._db = None
            self.graph = None

    # ... existing query methods (add_memory, search_memory, etc.) remain async ...
    # They will call self._execute_cypher() which is now native async
```

### 2.2 Configuration Extensions

**File:** `src/zapomni_core/config.py` (additions)

```python
class ZapomniSettings(BaseSettings):
    # ... existing fields ...

    # ========================================
    # DATABASE POOL CONFIGURATION
    # ========================================

    db_pool_min_size: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Minimum database connections to maintain in pool"
    )

    db_pool_max_size: int = Field(
        default=20,
        ge=1,
        le=200,
        description="Maximum database connections allowed in pool"
    )

    db_pool_timeout: float = Field(
        default=10.0,
        ge=1.0,
        le=60.0,
        description="Seconds to wait for available connection from pool"
    )

    db_socket_timeout: float = Field(
        default=30.0,
        ge=5.0,
        le=120.0,
        description="Socket timeout for query execution in seconds"
    )

    db_health_check_interval: int = Field(
        default=30,
        ge=10,
        le=300,
        description="Health check interval for pool connections in seconds"
    )

    # ========================================
    # RETRY CONFIGURATION
    # ========================================

    db_max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts for transient database errors"
    )

    db_retry_initial_delay: float = Field(
        default=0.1,
        ge=0.01,
        le=1.0,
        description="Initial delay for retry backoff in seconds"
    )

    db_retry_max_delay: float = Field(
        default=2.0,
        ge=0.1,
        le=30.0,
        description="Maximum delay for retry backoff in seconds"
    )
```

### 2.3 Server Integration

**File:** `src/zapomni_mcp/__main__.py` (modifications)

```python
async def main():
    # ... existing initialization ...

    # STAGE 2: Initialize database client
    logger.info("Initializing FalkorDB client")

    # Create pool config from environment
    pool_config = PoolConfig.from_env()

    db_client = FalkorDBClient(
        host=settings.falkordb_host,
        port=settings.falkordb_port,
        graph_name=settings.graph_name,
        password=settings.falkordb_password.get_secret_value() if settings.falkordb_password else None,
        pool_config=pool_config,
    )

    # NEW: Async initialization of connection pool
    await db_client.init_async()
    logger.info("FalkorDB connection pool initialized")

    # ... rest of initialization ...
```

**File:** `src/zapomni_mcp/server.py` (modifications)

```python
class MCPServer:
    async def _graceful_shutdown_sse(self) -> None:
        """Enhanced shutdown with database cleanup."""
        # ... existing shutdown steps ...

        # NEW: Close database connection pool
        if hasattr(self, '_db_client') and self._db_client:
            self._logger.info("closing_database_pool")
            await self._db_client.close()
            self._logger.info("database_pool_closed")

        # ... rest of shutdown ...
```

---

## 3. API Contracts

### 3.1 FalkorDBClient Public Interface

```python
class FalkorDBClient:
    """Async FalkorDB client with connection pooling."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6381,
        db: int = 0,
        graph_name: str = "zapomni_memory",
        password: Optional[str] = None,
        pool_config: Optional[PoolConfig] = None,
        retry_config: Optional[RetryConfig] = None,
    ) -> None:
        """Initialize client configuration (no network I/O)."""

    async def init_async(self) -> None:
        """Initialize connection pool. Must be called before use."""

    async def close(self) -> None:
        """Close connection pool (idempotent)."""

    async def get_pool_stats(self) -> Dict[str, Any]:
        """Get pool statistics for monitoring."""

    # Existing query methods (unchanged signatures):
    async def add_memory(self, memory: Memory) -> str: ...
    async def search_memory(self, query: str, ...) -> List[SearchResult]: ...
    async def get_stats(self) -> Dict[str, Any]: ...
    # ... other existing methods ...
```

### 3.2 Pool Stats Response

```python
@dataclass
class PoolStats:
    """Connection pool statistics."""
    max_connections: int  # Pool max size
    active_connections: int  # Currently in use
    total_queries: int  # Total queries executed
    total_retries: int  # Total retry attempts
    utilization_percent: float  # active/max * 100
    initialized: bool  # Pool is ready
    closed: bool  # Pool is closed
```

**JSON Response:**
```json
{
  "max_connections": 20,
  "active_connections": 12,
  "total_queries": 15234,
  "total_retries": 42,
  "utilization_percent": 60.0,
  "initialized": true,
  "closed": false
}
```

### 3.3 Health Endpoint Integration

**Endpoint:** `GET /health`

**Response with Pool Stats:**
```json
{
  "status": "healthy",
  "active_sessions": 5,
  "uptime_seconds": 3600,
  "version": "0.2.0",
  "database": {
    "pool": {
      "max_connections": 20,
      "active_connections": 12,
      "utilization_percent": 60.0,
      "initialized": true
    }
  }
}
```

---

## 4. Sequence Diagrams

### 4.1 Client Initialization

```
__main__.py              FalkorDBClient           BlockingConnectionPool      FalkorDB
     |                         |                           |                      |
     |--FalkorDBClient()------>|                           |                      |
     |                         |                           |                      |
     |                         |  (store config only)      |                      |
     |                         |  NO network I/O           |                      |
     |                         |                           |                      |
     |<--client instance-------|                           |                      |
     |                         |                           |                      |
     |--await init_async()---->|                           |                      |
     |                         |                           |                      |
     |                         |--BlockingConnectionPool-->|                      |
     |                         |   (host, port, max_conn)  |                      |
     |                         |                           |                      |
     |                         |<--pool instance-----------|                      |
     |                         |                           |                      |
     |                         |--AsyncFalkorDB(pool)----->|----------------->    |
     |                         |                           |                      |
     |                         |<--db instance-------------|<-----------------    |
     |                         |                           |                      |
     |                         |--select_graph()---------->|----------------->    |
     |                         |                           |                      |
     |                         |<--graph instance----------|<-----------------    |
     |                         |                           |                      |
     |                         |--query("RETURN 1")------->|----------------->    |
     |                         |   (test connection)       |                      |
     |                         |                           |                      |
     |                         |<--result------------------|<-----------------    |
     |                         |                           |                      |
     |                         |--_init_schema_async()---->|                      |
     |                         |                           |                      |
     |                         |  _initialized = True      |                      |
     |                         |                           |                      |
     |<--None------------------|                           |                      |
     |                         |                           |                      |
     |  (client ready)         |                           |                      |
```

### 4.2 Query Execution with Retry

```
Tool Handler              FalkorDBClient           BlockingConnectionPool      FalkorDB
     |                         |                           |                      |
     |--await _execute_cypher->|                           |                      |
     |   (cypher, params)      |                           |                      |
     |                         |                           |                      |
     |                         |--_check_pool_utilization->|                      |
     |                         |                           |                      |
     |                         |  _active_connections++    |                      |
     |                         |                           |                      |
     |                         |--graph.query()----------->|----------------->    |
     |                         |                           |                      |
     |                         |                           |  [ConnectionError]   |
     |                         |<--ConnectionError---------|<-----------------    |
     |                         |                           |                      |
     |                         |  (retry attempt 1)        |                      |
     |                         |--asyncio.sleep(0.1s)----->|                      |
     |                         |                           |                      |
     |                         |--graph.query()----------->|----------------->    |
     |                         |                           |                      |
     |                         |                           |  [Success]           |
     |                         |<--result------------------|<-----------------    |
     |                         |                           |                      |
     |                         |  _active_connections--    |                      |
     |                         |                           |                      |
     |<--QueryResult-----------|                           |                      |
```

### 4.3 Graceful Shutdown

```
MCPServer              FalkorDBClient           BlockingConnectionPool
     |                         |                           |
     |  (SIGTERM received)     |                           |
     |                         |                           |
     |--await close()--------->|                           |
     |                         |                           |
     |                         |  if _active_connections > 0:
     |                         |    wait up to 10s         |
     |                         |                           |
     |                         |--await _pool.aclose()---->|
     |                         |                           |
     |                         |                           |  [close all conns]
     |                         |                           |
     |                         |<--None--------------------|
     |                         |                           |
     |                         |  _closed = True           |
     |                         |  _initialized = False     |
     |                         |                           |
     |<--None------------------|                           |
     |                         |                           |
     |  (shutdown complete)    |                           |
```

### 4.4 Concurrent Queries (Pool in Action)

```
Client A                 Client B                 Pool                    FalkorDB
     |                         |                    |                         |
     |--query(add_memory)----->|                    |                         |
     |                         |                    |                         |
     |                         |--query(search)---->|                         |
     |                         |                    |                         |
     |                         |                    |--acquire conn 1-------->|
     |                         |                    |                         |
     |                         |                    |--acquire conn 2-------->|
     |                         |                    |                         |
     |                         |                    |  [queries run parallel] |
     |                         |                    |                         |
     |                         |                    |<--result 2--------------|
     |                         |<--result----------|                         |
     |                         |                    |                         |
     |                         |                    |<--result 1--------------|
     |<--result----------------|                    |                         |
     |                         |                    |                         |
     |                         |                    |  [connections returned] |
```

---

## 5. Data Models

### 5.1 PoolConfig

```python
@dataclass
class PoolConfig:
    """Connection pool configuration."""
    min_size: int = 5  # Minimum connections
    max_size: int = 20  # Maximum connections
    timeout: float = 10.0  # Wait for available connection
    socket_timeout: float = 30.0  # Query execution timeout
    socket_connect_timeout: float = 5.0  # Initial connection
    health_check_interval: int = 30  # Seconds between checks

    @classmethod
    def from_env(cls) -> "PoolConfig":
        """Load from environment variables."""
```

### 5.2 RetryConfig

```python
@dataclass
class RetryConfig:
    """Retry configuration for transient errors."""
    max_retries: int = 3  # Maximum attempts
    initial_delay: float = 0.1  # First retry delay
    max_delay: float = 2.0  # Maximum delay cap
    exponential_base: float = 2.0  # Backoff multiplier
```

### 5.3 PoolStats

```python
@dataclass
class PoolStats:
    """Connection pool statistics."""
    max_connections: int
    active_connections: int
    total_queries: int
    total_retries: int
    utilization_percent: float
    initialized: bool
    closed: bool
```

---

## 6. Error Handling Strategy

### 6.1 Error Categories

| Category | Examples | Handling | Retry |
|----------|----------|----------|-------|
| Connection Errors | Network timeout, Connection refused | Retry with backoff | Yes |
| Busy Errors | BusyLoadingError (Redis loading) | Retry with backoff | Yes |
| Query Errors | Invalid Cypher syntax | Return error, no retry | No |
| Validation Errors | Invalid parameters | Return error, no retry | No |
| Pool Exhaustion | No available connections | Wait (BlockingPool) | N/A |

### 6.2 Retry Logic Implementation

```python
async def _execute_with_retry(self, query: str, parameters: dict) -> QueryResult:
    """Execute query with exponential backoff retry."""
    delay = self.retry_config.initial_delay

    for attempt in range(self.retry_config.max_retries + 1):
        try:
            result = await self.graph.query(query, parameters)
            return self._convert_result(result)

        except (ConnectionError, redis.BusyLoadingError, OSError) as e:
            if attempt < self.retry_config.max_retries:
                self._logger.warning(
                    "query_retry",
                    attempt=attempt + 1,
                    delay=delay,
                    error=str(e),
                )
                await asyncio.sleep(delay)
                delay = min(
                    delay * self.retry_config.exponential_base,
                    self.retry_config.max_delay,
                )
            else:
                raise

        except Exception as e:
            # Non-retryable
            raise QueryError(f"Query failed: {e}")
```

### 6.3 Backoff Sequence

| Attempt | Delay Before | Total Wait |
|---------|--------------|------------|
| 1 | 0s (immediate) | 0s |
| 2 | 0.1s | 0.1s |
| 3 | 0.2s | 0.3s |
| 4 | 0.4s | 0.7s |
| Fail | - | 0.7s total |

---

## 7. Monitoring & Observability

### 7.1 Logging Events

| Event | Level | Fields |
|-------|-------|--------|
| `connection_pool_initialized` | INFO | pool_max_size, pool_timeout |
| `high_pool_utilization` | WARNING | utilization_percent, active, max |
| `query_retry` | WARNING | attempt, delay, error |
| `query_failed_after_retries` | ERROR | attempts, error |
| `connection_pool_closed` | INFO | - |

### 7.2 Health Endpoint Integration

```python
# In sse_transport.py handle_health()
async def handle_health(request: Request) -> Response:
    """Health check with pool stats."""
    # ... existing health info ...

    # Add pool stats if available
    if hasattr(request.app.state, 'db_client'):
        pool_stats = await request.app.state.db_client.get_pool_stats()
        health_info["database"] = {"pool": pool_stats}

    return JSONResponse(health_info)
```

### 7.3 Metrics to Monitor

| Metric | Alert Threshold | Description |
|--------|----------------|-------------|
| `pool_utilization_percent` | > 80% | Pool nearing capacity |
| `total_retries` | > 10/min | Network instability |
| `active_connections` | = max_connections | Pool exhausted |
| `query_latency_p95` | > 500ms | Slow queries |

---

## 8. Testing Strategy

### 8.1 Unit Tests

| Component | Test Focus |
|-----------|-----------|
| PoolConfig | Validation, from_env loading |
| RetryConfig | Validation, defaults |
| FalkorDBClient | Init, close, get_pool_stats |
| _execute_with_retry | Retry logic, backoff timing |

### 8.2 Integration Tests

| Scenario | Verification |
|----------|-------------|
| Client lifecycle | init_async -> queries -> close |
| Concurrent queries | 10 parallel queries succeed |
| Retry behavior | Simulated connection errors retried |
| Pool stats | Stats reflect actual state |

### 8.3 Load Tests

| Metric | Target |
|--------|--------|
| 50 concurrent queries | < 5s total |
| Throughput | > 1000 queries/s |
| Memory stability | No growth over 30 min |
| Error rate | < 0.1% |

---

## 9. File Structure

```
src/zapomni_db/
    falkordb_client.py      # MODIFY: Async client with pool
    pool_config.py          # NEW: Pool configuration
    retry_config.py         # NEW: Retry configuration
    exceptions.py           # MODIFY: Add retry-related exceptions

src/zapomni_core/
    config.py               # MODIFY: Add pool settings

src/zapomni_mcp/
    __main__.py             # MODIFY: Add await init_async()
    server.py               # MODIFY: Add pool close in shutdown
    sse_transport.py        # MODIFY: Add pool stats to /health

tests/
    unit/
        test_pool_config.py    # NEW: Pool config tests
        test_retry_logic.py    # NEW: Retry logic tests
    integration/
        test_connection_pool.py # NEW: Pool integration tests
```

---

## 10. Migration Strategy

### 10.1 Phase 1: Foundation

1. Add `PoolConfig` and `RetryConfig` classes
2. Add configuration fields to `ZapomniSettings`
3. Update `FalkorDBClient.__init__` (config only)

### 10.2 Phase 2: Async Implementation

4. Add `init_async()` method
5. Add `_execute_with_retry()` method
6. Update `_execute_cypher()` to native async
7. Add `close()` method

### 10.3 Phase 3: Integration

8. Update `__main__.py` to call `init_async()`
9. Update `server.py` shutdown to call `close()`
10. Add pool stats to health endpoint

### 10.4 Phase 4: Testing & Validation

11. Add unit tests
12. Add integration tests
13. Run load tests
14. Update documentation

---

**End of Design Document**
