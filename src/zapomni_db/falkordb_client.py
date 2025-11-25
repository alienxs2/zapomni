"""
FalkorDBClient - Async database client for FalkorDB with connection pooling.

Implements native async FalkorDB client with BlockingConnectionPool for
high-concurrency SSE transport support.

Migration from sync to async client for concurrent connections.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

import asyncio
import json
import math
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import redis
import structlog
from falkordb import FalkorDB as SyncFalkorDB  # For schema init only
from falkordb.asyncio import FalkorDB as AsyncFalkorDB
from redis.asyncio import BlockingConnectionPool

from zapomni_db.cypher_query_builder import CypherQueryBuilder
from zapomni_db.exceptions import (
    ConnectionError,
    DatabaseError,
    QueryError,
    TransactionError,
    ValidationError,
)
from zapomni_db.models import (
    DEFAULT_WORKSPACE_ID,
    Chunk,
    Entity,
    Memory,
    QueryResult,
    Relationship,
    SearchResult,
    Workspace,
    WorkspaceStats,
)
from zapomni_db.pool_config import PoolConfig, RetryConfig
from zapomni_db.schema_manager import SchemaManager

logger = structlog.get_logger(__name__)


class FalkorDBClient:
    """
    Async FalkorDB client with connection pooling.

    Uses native async client for SSE concurrent connections.
    Provides high-level interface for storing and querying memories,
    performing vector similarity search, and executing graph queries.

    BREAKING CHANGE: close() is now async. Use `await db_client.close()`.
    """

    DEFAULT_POOL_SIZE = 20
    DEFAULT_VECTOR_DIMENSION = 768
    DEFAULT_MAX_RETRIES = 3

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6381,
        db: int = 0,
        graph_name: str = "zapomni_memory",
        password: Optional[str] = None,
        pool_size: int = DEFAULT_POOL_SIZE,
        max_retries: int = DEFAULT_MAX_RETRIES,
        pool_config: Optional[PoolConfig] = None,
        retry_config: Optional[RetryConfig] = None,
    ):
        """
        Initialize FalkorDB client configuration.

        Note: Does NOT connect to database. Call init_async() to establish connection.

        Args:
            host: FalkorDB host address
            port: FalkorDB port (Redis protocol)
            db: Database index (0-15)
            graph_name: Name of graph to use
            password: Optional Redis password
            pool_size: Connection pool size (deprecated, use pool_config)
            max_retries: Maximum retry attempts (deprecated, use retry_config)
            pool_config: Connection pool configuration
            retry_config: Retry configuration for transient errors

        Raises:
            ValidationError: If parameters are invalid
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

        # Use new config classes, with backwards compat for pool_size/max_retries
        self.pool_config = pool_config or PoolConfig(max_size=pool_size)
        self.retry_config = retry_config or RetryConfig(max_retries=max_retries)

        # Legacy attributes for backwards compatibility
        self.pool_size = self.pool_config.max_size
        self.max_retries = self.retry_config.max_retries

        # Initialize state - NOT connected yet
        self._pool: Optional[BlockingConnectionPool] = None
        self._db: Optional[AsyncFalkorDB] = None
        self.graph: Any = None
        self._schema_manager: Optional[SchemaManager] = None
        self._initialized = False
        self._schema_ready = False
        self._closed = False

        # Structured logger with context
        self._logger = logger.bind(
            host=host,
            port=port,
            graph=graph_name,
            pool_max=self.pool_config.max_size,
        )

        # Pool monitoring state
        self._active_connections = 0
        self._total_queries = 0
        self._total_retries = 0
        self._pool_wait_count = 0
        self._utilization_warning_logged = False

        self._logger.info(
            "client_configured",
            pool_max_size=self.pool_config.max_size,
            pool_timeout=self.pool_config.timeout,
            max_retries=self.retry_config.max_retries,
        )

    async def init_async(self) -> None:
        """
        Initialize async connection pool.

        Must be called before any database operations.
        Creates BlockingConnectionPool for high-concurrency SSE support.

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
                timeout=int(self.pool_config.timeout),  # BlockingConnectionPool expects int
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

            # Initialize schema (sync operation wrapped in thread)
            await self._init_schema_async()

            self._initialized = True
            self._schema_ready = True

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

    def _init_connection(self) -> None:
        """
        Legacy sync initialization method.

        Deprecated: Use init_async() instead.
        This method is kept for backwards compatibility but should not be used
        for new code. It initializes a sync connection which doesn't support
        the async connection pool.
        """
        try:
            # Create sync FalkorDB connection (legacy)
            sync_db = SyncFalkorDB(
                host=self.host,
                port=self.port,
                password=self.password,
            )

            # Select graph
            self.graph = sync_db.select_graph(self.graph_name)

            self._initialized = True

            # Initialize schema
            self._init_schema()

            self._schema_ready = True
        except Exception as e:
            self._logger.error("connection_failed", error=str(e))
            raise ConnectionError(f"Failed to connect to FalkorDB: {e}")

    def _init_schema(self) -> None:
        """Initialize graph schema using SchemaManager (legacy sync method)."""
        try:
            self._schema_manager = SchemaManager(
                graph=self.graph,
                logger=self._logger,
            )
            self._schema_manager.init_schema()
            self._logger.info("schema_initialized_via_manager")
        except Exception as e:
            self._logger.error("schema_initialization_failed", error=str(e))
            raise

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

    def _convert_result(self, result: Any) -> QueryResult:
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
            execution_time_ms=int(result.run_time_ms) if hasattr(result, "run_time_ms") else 0,
        )

    async def _execute_with_retry(
        self,
        query: str,
        parameters: Dict[str, Any],
    ) -> QueryResult:
        """
        Execute query with exponential backoff retry.

        Retries on:
        - ConnectionError
        - redis.BusyLoadingError
        - OSError (network issues)

        Does NOT retry on:
        - QueryError (invalid Cypher syntax)
        - Other exceptions

        Args:
            query: Cypher query string
            parameters: Query parameters

        Returns:
            QueryResult with rows and metadata

        Raises:
            ConnectionError: After max retries exhausted
            QueryError: For invalid query syntax
        """
        delay = self.retry_config.initial_delay

        for attempt in range(self.retry_config.max_retries + 1):
            try:
                # Native async query execution
                result = await self.graph.query(query, parameters)
                return self._convert_result(result)

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
                    raise ConnectionError(f"Query failed after {attempt + 1} attempts: {e}")

            except Exception as e:
                # Non-retryable errors (query syntax, etc.)
                self._logger.error("query_error", error=str(e), query=query[:100])
                raise QueryError(f"Query failed: {e}")

        # Should not reach here, but just in case
        raise ConnectionError("Query failed: max retries exhausted")

    async def _execute_cypher(
        self,
        query: str,
        parameters: Dict[str, Any],
    ) -> QueryResult:
        """
        Execute Cypher query with pool monitoring and retry logic.

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

    async def get_pool_stats(self) -> Dict[str, Any]:
        """
        Get connection pool statistics for monitoring.

        Returns:
            Dict containing:
                - max_connections: Pool max size
                - active_connections: Currently in use
                - total_queries: Total queries executed
                - total_retries: Total retry attempts
                - utilization_percent: active/max * 100
                - initialized: Pool is ready
                - closed: Pool is closed
        """
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

        BREAKING CHANGE: This method is now async.
        Use `await db_client.close()` instead of `db_client.close()`.

        Safe to call multiple times (idempotent).
        Waits for pending queries to complete (up to 10 seconds).
        """
        if self._closed:
            self._logger.debug("already_closed")
            return

        self._logger.info("closing_connection_pool")

        try:
            # Wait for pending queries (up to 10s)
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
            if self._pool and hasattr(self._pool, "aclose"):
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

    # ========================================
    # MEMORY OPERATIONS
    # ========================================

    async def add_memory(
        self,
        memory: Memory,
        workspace_id: Optional[str] = None,
    ) -> str:
        """
        Store a complete memory with chunks and embeddings in graph database.

        Args:
            memory: Memory object containing text, chunks, embeddings, metadata
            workspace_id: Workspace ID for data isolation. Defaults to "default".

        Returns:
            memory_id: UUID string identifying the stored memory

        Raises:
            ValidationError: If input validation fails
            DatabaseError: If database operations fail
            TransactionError: If transaction state is invalid
        """
        # Use workspace_id from parameter, memory object, or default
        effective_workspace_id = workspace_id or memory.workspace_id or DEFAULT_WORKSPACE_ID
        # STEP 1: INPUT VALIDATION

        # Validate text
        if not memory.text or not memory.text.strip():
            raise ValidationError("Validation failed for memory: text cannot be empty")

        if len(memory.text) > 1_000_000:
            raise ValidationError(
                "Validation failed for memory: text exceeds max length (1,000,000)"
            )

        # Validate chunks
        if not memory.chunks:
            raise ValidationError("Validation failed for memory: chunks list cannot be empty")

        if len(memory.chunks) > 100:
            raise ValidationError("Validation failed for memory: too many chunks (max 100)")

        # Validate embeddings
        if not memory.embeddings:
            raise ValidationError("Validation failed for memory: embeddings list cannot be empty")

        # Validate chunks/embeddings count match
        if len(memory.chunks) != len(memory.embeddings):
            raise ValidationError(
                f"Validation failed for memory: chunks and embeddings count mismatch "
                f"({len(memory.chunks)} chunks, {len(memory.embeddings)} embeddings)"
            )

        # Validate each chunk and embedding
        for i, (chunk, embedding) in enumerate(zip(memory.chunks, memory.embeddings)):
            # Check chunk index is sequential
            if chunk.index != i:
                raise ValidationError(
                    f"Validation failed for memory: chunk index mismatch: expected {i}, got {chunk.index}"
                )

            # Check chunk text not empty
            if not chunk.text or not chunk.text.strip():
                raise ValidationError(
                    f"Validation failed for memory: chunk {i} text cannot be empty"
                )

            # Check embedding dimension
            if len(embedding) != 768:
                raise ValidationError(
                    f"Validation failed for memory: embedding {i} dimension must be 768, got {len(embedding)}"
                )

            # Check embedding values are numeric
            for j, value in enumerate(embedding):
                if not isinstance(value, (int, float)):
                    raise ValidationError(
                        "Validation failed for memory: embedding {i} contains non-numeric values"
                    )
                if not math.isfinite(value):
                    raise ValidationError(
                        "Validation failed for memory: embedding {i} contains NaN or Inf values"
                    )

        # Validate metadata
        try:
            serialized_metadata = json.dumps(memory.metadata)
            if len(serialized_metadata) > 100_000:
                raise ValidationError(
                    "Validation failed for memory: metadata exceeds max size (100 KB)"
                )
        except (TypeError, ValueError) as e:
            raise ValidationError(
                f"Validation failed for memory: metadata not JSON-serializable: {e}"
            )

        # STEP 2: GENERATE UUIDS
        memory_id = str(uuid.uuid4())
        chunk_ids = [str(uuid.uuid4()) for _ in range(len(memory.chunks))]

        # STEP 3: EXECUTE WITH RETRY
        retry_count = 0
        while retry_count <= self.retry_config.max_retries:
            try:
                await self._execute_transaction(
                    memory, memory_id, chunk_ids, effective_workspace_id
                )

                self._logger.info(
                    "memory_added",
                    memory_id=memory_id,
                    num_chunks=len(memory.chunks),
                    workspace_id=effective_workspace_id,
                )

                return memory_id

            except (ConnectionError, TimeoutError) as e:
                retry_count += 1
                if retry_count > self.retry_config.max_retries:
                    raise DatabaseError(
                        f"Database error during add_memory: connection failed after {self.retry_config.max_retries} retries: {e}"
                    )

                backoff_seconds = 2 ** (retry_count - 1)
                self._logger.warning(
                    "retry_attempt",
                    attempt=retry_count,
                    delay=backoff_seconds,
                )
                await asyncio.sleep(backoff_seconds)

            except Exception as e:
                self._logger.error("add_memory_error", error=str(e))
                raise DatabaseError(f"Unexpected database error during add_memory: {e}")

        # Should not reach here
        raise DatabaseError("add_memory failed: max retries exhausted")

    async def _execute_transaction(
        self,
        memory: Memory,
        memory_id: str,
        chunk_ids: List[str],
        workspace_id: str = DEFAULT_WORKSPACE_ID,
    ) -> str:
        """Execute transaction to store memory."""
        if not self._initialized:
            raise ConnectionError("Not initialized")

        # Build Cypher query for atomic transaction
        cypher = """
        // Create Memory node with workspace_id
        CREATE (m:Memory {
            id: $memory_id,
            text: $text,
            tags: $tags,
            source: $source,
            metadata: $metadata,
            workspace_id: $workspace_id,
            created_at: $timestamp
        })

        // Create Chunk nodes with embeddings and workspace_id
        WITH m
        UNWIND $chunks_data AS chunk_data
        CREATE (c:Chunk {
            id: chunk_data.id,
            text: chunk_data.text,
            index: chunk_data.index,
            workspace_id: $workspace_id,
            embedding: vecf32(chunk_data.embedding)
        })
        CREATE (m)-[:HAS_CHUNK {index: chunk_data.index}]->(c)

        RETURN m.id AS memory_id
        """

        # Prepare parameters
        chunks_data = [
            {
                "id": chunk_ids[i],
                "text": chunk.text,
                "index": chunk.index,
                "embedding": embedding,
            }
            for i, (chunk, embedding) in enumerate(zip(memory.chunks, memory.embeddings))
        ]

        parameters = {
            "memory_id": memory_id,
            "text": memory.text,
            "tags": memory.metadata.get("tags", []),
            "source": memory.metadata.get("source", ""),
            "metadata": json.dumps(memory.metadata),
            "workspace_id": workspace_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "chunks_data": chunks_data,
        }

        # Execute query with retry logic
        result = await self._execute_cypher(cypher, parameters)

        # Debug logging
        self._logger.debug(
            "transaction_result",
            row_count=result.row_count,
        )

        # Verify result
        if result.row_count > 0:
            return memory_id
        else:
            raise DatabaseError("Memory creation failed: no records returned")

    async def vector_search(
        self,
        embedding: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        workspace_id: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Perform HNSW vector similarity search on chunk embeddings.

        Args:
            embedding: Query embedding vector (must be 768-dimensional)
            limit: Maximum number of results to return (1-1000)
            filters: Optional metadata filters
            workspace_id: Workspace ID for data isolation. Defaults to "default".

        Returns:
            List of SearchResult objects sorted by similarity (descending)

        Raises:
            ValidationError: If input validation fails
            QueryError: If database query fails
            DatabaseError: If database operation fails after retries
        """
        # Determine effective workspace_id
        effective_workspace_id = workspace_id or DEFAULT_WORKSPACE_ID
        # STEP 1: INPUT VALIDATION

        # Validate embedding dimension
        if len(embedding) != 768:
            raise ValidationError(
                f"Validation failed: embedding dimension must be 768, got {len(embedding)}"
            )

        # Validate embedding values
        if not all(math.isfinite(x) for x in embedding):
            raise ValidationError("Validation failed: embedding contains NaN or Inf values")

        # Validate limit
        if not isinstance(limit, int):
            raise ValidationError(
                f"Validation failed: limit must be int, got {type(limit).__name__}"
            )

        if limit < 1:
            raise ValidationError(f"Validation failed: limit must be >= 1, got {limit}")

        if limit > 1000:
            raise ValidationError(f"Validation failed: limit cannot exceed 1000, got {limit}")

        # Validate filters
        if filters is not None:
            if not isinstance(filters, dict):
                raise ValidationError("Validation failed: filters must be dict or None")

            # Validate tags
            if "tags" in filters:
                tags = filters["tags"]
                if not isinstance(tags, list) or len(tags) == 0:
                    raise ValidationError("Validation failed: filters['tags'] cannot be empty list")

            # Validate source
            if "source" in filters:
                source = filters["source"]
                if not isinstance(source, str) or not source:
                    raise ValidationError(
                        "Validation failed: filters['source'] must be non-empty string"
                    )

            # Validate date formats
            if "date_from" in filters:
                try:
                    datetime.fromisoformat(filters["date_from"].replace("Z", "+00:00"))
                except ValueError as e:
                    raise ValidationError(
                        f"Validation failed: filters['date_from'] invalid ISO 8601 format: {e}"
                    )

            if "date_to" in filters:
                try:
                    datetime.fromisoformat(filters["date_to"].replace("Z", "+00:00"))
                except ValueError as e:
                    raise ValidationError(
                        f"Validation failed: filters['date_to'] invalid ISO 8601 format: {e}"
                    )

            # Validate min_similarity
            if "min_similarity" in filters:
                min_sim = filters["min_similarity"]
                if not isinstance(min_sim, (int, float)):
                    raise ValidationError(
                        "Validation failed: filters['min_similarity'] must be numeric"
                    )
                if not (0.0 <= min_sim <= 1.0):
                    raise ValidationError(
                        f"Validation failed: filters['min_similarity'] must be in [0.0, 1.0], got {min_sim}"
                    )

        # STEP 2: BUILD QUERY USING CypherQueryBuilder
        query_builder = CypherQueryBuilder()
        min_similarity = filters.get("min_similarity", 0.5) if filters else 0.5
        cypher, params = query_builder.build_vector_search_query(
            embedding=embedding,
            limit=limit,
            filters=filters,
            min_similarity=min_similarity,
            workspace_id=effective_workspace_id,
        )

        # STEP 3: EXECUTE SEARCH
        try:
            result = await self._execute_cypher(cypher, params)
            return self._parse_search_results(result)
        except Exception as e:
            self._logger.error("vector_search_error", error=str(e))
            raise DatabaseError(f"Vector search failed: {e}")

    def _parse_search_results(self, query_result: QueryResult) -> List[SearchResult]:
        """Parse database results into SearchResult objects."""
        results = []

        for row in query_result.rows:
            try:
                timestamp_str = row.get("timestamp", datetime.now(timezone.utc).isoformat())
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))

                # Parse tags from JSON string if needed
                tags = row.get("tags", [])
                if isinstance(tags, str):
                    tags = json.loads(tags)

                search_result = SearchResult(
                    memory_id=row["memory_id"],
                    content=row["text"],
                    relevance_score=float(row["similarity_score"]),
                    chunk_id=row.get("chunk_id"),
                    text=row["text"],
                    similarity_score=float(row["similarity_score"]),
                    tags=tags,
                    source=row.get("source", ""),
                    timestamp=timestamp,
                    chunk_index=int(row.get("chunk_index", 0)),
                )
                results.append(search_result)
            except (KeyError, ValueError, TypeError) as e:
                self._logger.warning("malformed_search_row", row=row, error=str(e))
                continue

        results.sort(key=lambda r: r.similarity_score or 0.0, reverse=True)
        return results

    async def get_stats(self) -> Dict[str, Any]:
        """
        Retrieve comprehensive statistics about the knowledge graph.

        Returns:
            Dict containing node counts, relationship counts, storage metrics,
            index statistics, health indicators, and pool statistics.

        Raises:
            DatabaseError: If database queries fail
            ConnectionError: If database connection is lost
        """
        stats: Dict[str, Any] = {
            "nodes": {},
            "relationships": {},
            "storage": {},
            "indexes": {},
            "health": {},
        }

        start_time = time.time()

        try:
            # Count nodes by type
            node_results = await self._execute_cypher(
                "MATCH (n) RETURN labels(n)[0] AS node_type, count(n) AS count", {}
            )

            stats["nodes"]["total"] = 0
            stats["nodes"]["memory"] = 0
            stats["nodes"]["chunk"] = 0
            stats["nodes"]["entity"] = 0
            stats["nodes"]["document"] = 0

            for row in node_results.rows:
                node_type = row["node_type"].lower() if row.get("node_type") else "unknown"
                count = row["count"]
                stats["nodes"][node_type] = count
                stats["nodes"]["total"] += count

            # Count relationships by type
            rel_results = await self._execute_cypher(
                "MATCH ()-[r]->() RETURN type(r) AS rel_type, count(r) AS count", {}
            )

            stats["relationships"]["total"] = 0
            stats["relationships"]["has_chunk"] = 0
            stats["relationships"]["mentions"] = 0
            stats["relationships"]["related_to"] = 0

            for row in rel_results.rows:
                rel_type = (
                    row["rel_type"].lower().replace("_", "_") if row.get("rel_type") else "unknown"
                )
                count = row["count"]
                stats["relationships"][rel_type] = count
                stats["relationships"]["total"] += count

            # Calculate storage metrics
            stats["storage"]["total_memories"] = stats["nodes"]["memory"]
            stats["storage"]["total_chunks"] = stats["nodes"]["chunk"]
            stats["storage"]["total_entities"] = stats["nodes"]["entity"]

            if stats["nodes"]["memory"] > 0:
                stats["storage"]["avg_chunks_per_memory"] = (
                    stats["nodes"]["chunk"] / stats["nodes"]["memory"]
                )
            else:
                stats["storage"]["avg_chunks_per_memory"] = 0.0

            # Index stats
            try:
                index_results = await self._execute_cypher(
                    "CALL db.indexes() YIELD name, type WHERE name = 'chunk_embedding_idx' RETURN name, type",
                    {},
                )

                if index_results.rows:
                    stats["indexes"]["vector_index_name"] = index_results.rows[0]["name"]
                    stats["indexes"]["vector_index_size"] = stats["nodes"]["chunk"]
                else:
                    stats["indexes"]["vector_index_name"] = "chunk_embedding_idx"
                    stats["indexes"]["vector_index_size"] = 0
            except Exception as e:
                # db.indexes() may not be available in all FalkorDB versions
                self._logger.debug("index_query_failed", error=str(e))
                stats["indexes"]["vector_index_name"] = "chunk_embedding_idx"
                stats["indexes"]["vector_index_size"] = stats["nodes"]["chunk"]

            # Health metrics
            stats["health"]["connected"] = True
            stats["health"]["graph_name"] = self.graph_name

            end_time = time.time()
            total_latency_ms = (end_time - start_time) * 1000
            num_queries = 3
            stats["health"]["query_latency_ms"] = round(total_latency_ms / num_queries, 2)

            # Add backward compatible top-level fields
            stats["total_memories"] = stats["storage"]["total_memories"]
            stats["total_chunks"] = stats["storage"]["total_chunks"]
            stats["database_size_mb"] = 0.0  # Not available in current implementation
            stats["avg_query_latency_ms"] = stats["health"]["query_latency_ms"]

            # Add pool monitoring stats
            pool_stats = await self.get_pool_stats()
            stats["pool"] = {
                "size": self.pool_config.max_size,
                "active_connections": pool_stats["active_connections"],
                "total_queries": pool_stats["total_queries"],
                "total_retries": pool_stats["total_retries"],
                "utilization_percent": pool_stats["utilization_percent"],
                "initialized": pool_stats["initialized"],
                "closed": pool_stats["closed"],
            }

            self._logger.info(
                "stats_retrieved",
                total_nodes=stats["nodes"]["total"],
                pool_active=self._active_connections,
            )

            return stats

        except Exception as e:
            self._logger.error("stats_error", error=str(e))
            raise DatabaseError(f"Failed to retrieve graph statistics: {e}")

    # ========================================
    # ENTITY OPERATIONS
    # ========================================

    async def add_entity(self, entity: Entity) -> str:
        """
        Add an entity node to the knowledge graph.

        Args:
            entity: Entity object with name, type, description, confidence

        Returns:
            entity_id: UUID string identifying the entity

        Raises:
            ValidationError: If entity validation fails
            DatabaseError: If database write fails after retries
        """
        entity_id = str(uuid.uuid4())

        cypher = """
        MERGE (e:Entity {name: $name})
        SET e.id = $entity_id,
            e.type = $type,
            e.description = $description,
            e.confidence = $confidence,
            e.updated_at = $timestamp
        RETURN e.id AS entity_id
        """

        parameters = {
            "entity_id": entity_id,
            "name": entity.name,
            "type": entity.type,
            "description": entity.description or "",
            "confidence": entity.confidence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        try:
            result = await self._execute_cypher(cypher, parameters)

            if result.row_count > 0:
                self._logger.info("entity_added", entity_id=entity_id, name=entity.name)
                return entity_id
            else:
                raise DatabaseError("Entity creation failed: no records returned")

        except Exception as e:
            self._logger.error("add_entity_error", error=str(e))
            raise DatabaseError(f"Failed to add entity: {e}")

    async def add_relationship(
        self,
        from_entity_id: str,
        to_entity_id: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add a relationship edge between two entities.

        Args:
            from_entity_id: Source entity UUID
            to_entity_id: Target entity UUID
            relationship_type: Relationship type (MENTIONS, RELATED_TO, USES, etc.)
            properties: Optional edge properties (strength, confidence, context)

        Returns:
            relationship_id: UUID string identifying the relationship

        Raises:
            ValidationError: If entity IDs are invalid
            DatabaseError: If entities not found or write fails
        """
        # Validate UUIDs
        try:
            uuid.UUID(from_entity_id)
            uuid.UUID(to_entity_id)
        except ValueError as e:
            raise ValidationError(f"Invalid entity UUID: {e}")

        # Extract properties
        if properties is None:
            properties = {}

        strength = properties.get("strength", 1.0)
        confidence = properties.get("confidence", 1.0)
        context = properties.get("context", "")

        # Validate strength and confidence
        if not (0.0 <= strength <= 1.0):
            raise ValidationError(f"strength must be in [0.0, 1.0], got {strength}")
        if not (0.0 <= confidence <= 1.0):
            raise ValidationError(f"confidence must be in [0.0, 1.0], got {confidence}")

        relationship_id = str(uuid.uuid4())

        cypher = f"""
        MATCH (from:Entity {{id: $from_id}})
        MATCH (to:Entity {{id: $to_id}})
        CREATE (from)-[r:{relationship_type} {{
            id: $rel_id,
            strength: $strength,
            confidence: $confidence,
            context: $context,
            created_at: $timestamp
        }}]->(to)
        RETURN r.id AS relationship_id
        """

        parameters = {
            "from_id": from_entity_id,
            "to_id": to_entity_id,
            "rel_id": relationship_id,
            "strength": strength,
            "confidence": confidence,
            "context": context,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        try:
            result = await self._execute_cypher(cypher, parameters)

            if result.row_count > 0:
                self._logger.info(
                    "relationship_added",
                    relationship_id=relationship_id,
                    type=relationship_type,
                )
                return relationship_id
            else:
                raise DatabaseError(
                    f"Relationship creation failed: entities not found "
                    f"(from: {from_entity_id}, to: {to_entity_id})"
                )

        except Exception as e:
            self._logger.error("add_relationship_error", error=str(e))
            raise DatabaseError(f"Failed to add relationship: {e}")

    async def get_related_entities(
        self,
        entity_id: str,
        depth: int = 1,
        limit: int = 20,
    ) -> List[Entity]:
        """
        Get entities related to a given entity via graph traversal.

        Args:
            entity_id: Starting entity UUID
            depth: Traversal depth (1-5 hops)
            limit: Maximum number of related entities to return (max: 100)

        Returns:
            List of Entity objects sorted by relationship strength

        Raises:
            ValidationError: If parameters are invalid
            DatabaseError: If query fails
        """
        # Validate entity UUID
        try:
            uuid.UUID(entity_id)
        except ValueError as e:
            raise ValidationError(f"Invalid entity UUID: {e}")

        # Validate depth
        if not isinstance(depth, int) or depth < 1 or depth > 5:
            raise ValidationError(f"depth must be in [1, 5], got {depth}")

        # Validate limit
        if not isinstance(limit, int) or limit < 1 or limit > 100:
            raise ValidationError(f"limit must be in [1, 100], got {limit}")

        cypher = f"""
        MATCH (start:Entity {{id: $entity_id}})
        MATCH path = (start)-[*1..{depth}]-(related:Entity)
        WHERE related.id <> $entity_id
        WITH DISTINCT related,
             [r IN relationships(path) | r.strength] AS strengths
        WITH related,
             reduce(s = 0.0, strength IN strengths | s + strength) / size(strengths) AS avg_strength
        RETURN related.id AS id,
               related.name AS name,
               related.type AS type,
               related.description AS description,
               related.confidence AS confidence,
               avg_strength
        ORDER BY avg_strength DESC
        LIMIT {limit}
        """

        parameters = {"entity_id": entity_id}

        try:
            result = await self._execute_cypher(cypher, parameters)

            entities = []
            for row in result.rows:
                entity = Entity(
                    name=row["name"],
                    type=row["type"],
                    description=row.get("description", ""),
                    confidence=float(row.get("confidence", 1.0)),
                )
                entities.append(entity)

            self._logger.info(
                "related_entities_retrieved",
                entity_id=entity_id,
                count=len(entities),
            )

            return entities

        except Exception as e:
            self._logger.error("get_related_entities_error", error=str(e))
            raise DatabaseError(f"Failed to retrieve related entities: {e}")

    async def graph_query(
        self,
        cypher: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> QueryResult:
        """
        Execute a Cypher query on the graph.

        Args:
            cypher: Cypher query string (parameterized queries recommended)
            parameters: Optional query parameters for parameterized queries

        Returns:
            QueryResult object with rows, row_count, execution_time_ms

        Raises:
            ValidationError: If query is invalid
            DatabaseError: If query execution fails
        """
        # Validate cypher
        if not cypher or not cypher.strip():
            raise ValidationError("cypher query cannot be empty")

        if len(cypher) > 100_000:
            raise ValidationError("cypher query exceeds max length (100,000 chars)")

        # Validate parameters are JSON-serializable
        if parameters:
            try:
                json.dumps(parameters)
            except (TypeError, ValueError) as e:
                raise ValidationError(f"parameters must be JSON-serializable: {e}")

        try:
            result = await self._execute_cypher(cypher, parameters or {})

            self._logger.info(
                "graph_query_executed",
                row_count=result.row_count,
                execution_time_ms=result.execution_time_ms,
            )

            return result

        except Exception as e:
            self._logger.error("graph_query_error", error=str(e))
            raise DatabaseError(f"Graph query failed: {e}")

    async def clear_all(self, workspace_id: Optional[str] = None) -> None:
        """
        Clear all data from the graph (DELETE all nodes/edges).

        WARNING: This is destructive and irreversible.

        Args:
            workspace_id: If provided, only clear data in this workspace.
                          If None, clears ALL data (dangerous!).

        Raises:
            DatabaseError: If clear operation fails
        """
        if workspace_id:
            # Clear only data in the specified workspace
            cypher = """
            MATCH (n)
            WHERE n.workspace_id = $workspace_id
            DETACH DELETE n
            """
            parameters = {"workspace_id": workspace_id}
        else:
            # Clear all data
            cypher = "MATCH (n) DETACH DELETE n"
            parameters = {}

        try:
            result = await self._execute_cypher(cypher, parameters)
            self._logger.warning(
                "graph_cleared",
                nodes_deleted=result.row_count,
                workspace_id=workspace_id,
            )

        except Exception as e:
            self._logger.error("clear_all_error", error=str(e))
            raise DatabaseError(f"Failed to clear graph: {e}")

    # ========================================
    # WORKSPACE OPERATIONS
    # ========================================

    async def create_workspace(self, workspace: Workspace) -> str:
        """
        Create a new workspace in the knowledge graph.

        Args:
            workspace: Workspace object with id, name, description

        Returns:
            workspace_id: The workspace ID

        Raises:
            ValidationError: If workspace validation fails
            DatabaseError: If workspace already exists or write fails
        """
        # Validate workspace id
        if not workspace.id or not workspace.id.strip():
            raise ValidationError("Workspace ID cannot be empty")

        # Check for valid characters (alphanumeric, hyphen, underscore)
        if not re.match(r"^[a-zA-Z0-9_-]+$", workspace.id):
            raise ValidationError(
                "Workspace ID must contain only alphanumeric characters, hyphens, and underscores"
            )

        # Check if workspace already exists
        existing = await self.get_workspace(workspace.id)
        if existing:
            raise DatabaseError(f"Workspace '{workspace.id}' already exists")

        cypher = """
        CREATE (w:Workspace {
            id: $workspace_id,
            name: $name,
            description: $description,
            created_at: $created_at,
            metadata: $metadata
        })
        RETURN w.id AS workspace_id
        """

        parameters = {
            "workspace_id": workspace.id,
            "name": workspace.name,
            "description": workspace.description or "",
            "created_at": workspace.created_at.isoformat(),
            "metadata": json.dumps(workspace.metadata or {}),
        }

        try:
            result = await self._execute_cypher(cypher, parameters)

            if result.row_count > 0:
                self._logger.info(
                    "workspace_created",
                    workspace_id=workspace.id,
                    name=workspace.name,
                )
                return workspace.id
            else:
                raise DatabaseError("Workspace creation failed: no records returned")

        except Exception as e:
            self._logger.error("create_workspace_error", error=str(e))
            raise DatabaseError(f"Failed to create workspace: {e}")

    async def get_workspace(self, workspace_id: str) -> Optional[Workspace]:
        """
        Get a workspace by ID.

        Args:
            workspace_id: Workspace ID to retrieve

        Returns:
            Workspace object if found, None otherwise

        Raises:
            DatabaseError: If query fails
        """
        cypher = """
        MATCH (w:Workspace {id: $workspace_id})
        RETURN w.id AS id,
               w.name AS name,
               w.description AS description,
               w.created_at AS created_at,
               w.updated_at AS updated_at,
               w.metadata AS metadata
        """

        parameters = {"workspace_id": workspace_id}

        try:
            result = await self._execute_cypher(cypher, parameters)

            if result.row_count > 0:
                row = result.rows[0]
                metadata = row.get("metadata", "{}")
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)

                created_at = row.get("created_at")
                if isinstance(created_at, str):
                    created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))

                updated_at = row.get("updated_at")
                if isinstance(updated_at, str):
                    updated_at = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))

                return Workspace(
                    id=row["id"],
                    name=row["name"],
                    description=row.get("description", ""),
                    created_at=created_at,
                    updated_at=updated_at,
                    metadata=metadata,
                )
            return None

        except Exception as e:
            self._logger.error("get_workspace_error", error=str(e))
            raise DatabaseError(f"Failed to get workspace: {e}")

    async def list_workspaces(self) -> List[Workspace]:
        """
        List all workspaces.

        Returns:
            List of Workspace objects

        Raises:
            DatabaseError: If query fails
        """
        cypher = """
        MATCH (w:Workspace)
        RETURN w.id AS id,
               w.name AS name,
               w.description AS description,
               w.created_at AS created_at,
               w.updated_at AS updated_at,
               w.metadata AS metadata
        ORDER BY w.created_at ASC
        """

        try:
            result = await self._execute_cypher(cypher, {})

            workspaces = []
            for row in result.rows:
                metadata = row.get("metadata", "{}")
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)

                created_at = row.get("created_at")
                if isinstance(created_at, str):
                    created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))

                updated_at = row.get("updated_at")
                if isinstance(updated_at, str):
                    updated_at = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))

                workspaces.append(
                    Workspace(
                        id=row["id"],
                        name=row["name"],
                        description=row.get("description", ""),
                        created_at=created_at,
                        updated_at=updated_at,
                        metadata=metadata,
                    )
                )

            return workspaces

        except Exception as e:
            self._logger.error("list_workspaces_error", error=str(e))
            raise DatabaseError(f"Failed to list workspaces: {e}")

    async def delete_workspace(
        self,
        workspace_id: str,
        confirm: bool = False,
    ) -> bool:
        """
        Delete a workspace and optionally all its data.

        IMPORTANT: According to validation report, we must validate workspace exists
        before deleting to prevent security issues.

        Args:
            workspace_id: Workspace ID to delete
            confirm: Must be True to confirm deletion

        Returns:
            True if deleted, False if workspace not found

        Raises:
            ValidationError: If workspace_id is the default workspace
            ValidationError: If confirm is not True
            DatabaseError: If delete operation fails
        """
        # Security: Cannot delete default workspace
        if workspace_id == DEFAULT_WORKSPACE_ID:
            raise ValidationError("Cannot delete the default workspace")

        # Safety: Require explicit confirmation
        if not confirm:
            raise ValidationError("Deletion requires explicit confirmation (confirm=True)")

        # Security: Verify workspace exists before deleting
        existing = await self.get_workspace(workspace_id)
        if not existing:
            self._logger.warning(
                "workspace_not_found_for_delete",
                workspace_id=workspace_id,
            )
            return False

        # Delete all data in the workspace first
        await self.clear_all(workspace_id=workspace_id)

        # Then delete the workspace node itself
        cypher = """
        MATCH (w:Workspace {id: $workspace_id})
        DELETE w
        RETURN count(w) AS deleted_count
        """

        parameters = {"workspace_id": workspace_id}

        try:
            result = await self._execute_cypher(cypher, parameters)

            if result.row_count > 0 and result.rows[0]["deleted_count"] > 0:
                self._logger.info("workspace_deleted", workspace_id=workspace_id)
                return True
            else:
                self._logger.warning("workspace_not_found", workspace_id=workspace_id)
                return False

        except Exception as e:
            self._logger.error("delete_workspace_error", error=str(e))
            raise DatabaseError(f"Failed to delete workspace: {e}")

    async def get_workspace_stats(
        self,
        workspace_id: Optional[str] = None,
    ) -> WorkspaceStats:
        """
        Get statistics for a specific workspace.

        Args:
            workspace_id: Workspace ID. Defaults to "default".

        Returns:
            WorkspaceStats object with counts

        Raises:
            DatabaseError: If query fails
        """
        effective_workspace_id = workspace_id or DEFAULT_WORKSPACE_ID

        cypher = """
        MATCH (m:Memory {workspace_id: $workspace_id})
        WITH count(m) AS total_memories
        MATCH (c:Chunk {workspace_id: $workspace_id})
        WITH total_memories, count(c) AS total_chunks
        OPTIONAL MATCH (e:Entity {workspace_id: $workspace_id})
        WITH total_memories, total_chunks, count(e) AS total_entities
        OPTIONAL MATCH (:Entity {workspace_id: $workspace_id})-[r]->(:Entity {workspace_id: $workspace_id})
        WITH total_memories, total_chunks, total_entities, count(r) AS total_relationships
        RETURN total_memories, total_chunks, total_entities, total_relationships
        """

        parameters = {"workspace_id": effective_workspace_id}

        try:
            result = await self._execute_cypher(cypher, parameters)

            if result.row_count > 0:
                row = result.rows[0]
                return WorkspaceStats(
                    workspace_id=effective_workspace_id,
                    total_memories=row.get("total_memories", 0) or 0,
                    total_chunks=row.get("total_chunks", 0) or 0,
                    total_entities=row.get("total_entities", 0) or 0,
                    total_relationships=row.get("total_relationships", 0) or 0,
                )
            else:
                return WorkspaceStats(
                    workspace_id=effective_workspace_id,
                    total_memories=0,
                    total_chunks=0,
                    total_entities=0,
                    total_relationships=0,
                )

        except Exception as e:
            self._logger.error("get_workspace_stats_error", error=str(e))
            raise DatabaseError(f"Failed to get workspace stats: {e}")

    async def delete_memory(
        self,
        memory_id: str,
        workspace_id: Optional[str] = None,
    ) -> bool:
        """
        Delete a memory and its associated chunks.

        IMPORTANT: According to validation report, we must validate workspace
        to prevent cross-workspace deletion.

        Args:
            memory_id: Memory UUID to delete
            workspace_id: Workspace ID for validation. If provided, memory must
                          belong to this workspace.

        Returns:
            True if deleted, False if memory not found

        Raises:
            ValidationError: If memory_id is invalid UUID
            ValidationError: If memory exists but in different workspace
            DatabaseError: If delete operation fails
        """
        # Validate UUID
        try:
            uuid.UUID(memory_id)
        except ValueError as e:
            raise ValidationError(f"Invalid memory UUID: {e}")

        # If workspace_id provided, validate memory belongs to it
        if workspace_id:
            check_cypher = """
            MATCH (m:Memory {id: $memory_id})
            RETURN m.workspace_id AS workspace_id
            """
            check_params = {"memory_id": memory_id}

            try:
                check_result = await self._execute_cypher(check_cypher, check_params)
                if check_result.row_count > 0:
                    actual_workspace = check_result.rows[0].get("workspace_id")
                    if actual_workspace and actual_workspace != workspace_id:
                        raise ValidationError(
                            f"Memory belongs to workspace '{actual_workspace}', "
                            f"not '{workspace_id}'"
                        )
            except ValidationError:
                raise
            except Exception:
                pass  # Memory might not exist, deletion will return False

        cypher = """
        MATCH (m:Memory {id: $memory_id})
        OPTIONAL MATCH (m)-[:HAS_CHUNK]->(c:Chunk)
        DETACH DELETE m, c
        RETURN count(m) AS deleted_count
        """

        parameters = {"memory_id": memory_id}

        try:
            result = await self._execute_cypher(cypher, parameters)

            if result.row_count > 0 and result.rows[0]["deleted_count"] > 0:
                self._logger.info(
                    "memory_deleted",
                    memory_id=memory_id,
                    workspace_id=workspace_id,
                )
                return True
            else:
                self._logger.warning("memory_not_found", memory_id=memory_id)
                return False

        except Exception as e:
            self._logger.error("delete_memory_error", error=str(e))
            raise DatabaseError(f"Failed to delete memory: {e}")
