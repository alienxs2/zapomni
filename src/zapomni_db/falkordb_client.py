"""
FalkorDBClient - Main database client for FalkorDB unified vector + graph database.

Author: Goncharenko Anton aka alienxs2
TDD Implementation: Code written to pass tests from specifications.
"""

import uuid
import json
import time
import math
import asyncio
import structlog
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from falkordb import FalkorDB

from zapomni_db.models import Memory, Chunk, SearchResult, QueryResult, Entity, Relationship
from zapomni_db.exceptions import (
    ValidationError,
    DatabaseError,
    ConnectionError,
    QueryError,
    TransactionError
)
from zapomni_db.schema_manager import SchemaManager
from zapomni_db.cypher_query_builder import CypherQueryBuilder


logger = structlog.get_logger(__name__)


class FalkorDBClient:
    """
    Main client for FalkorDB unified vector + graph database.

    Provides high-level interface for storing and querying memories,
    performing vector similarity search, and executing graph queries.
    """

    DEFAULT_POOL_SIZE = 10
    DEFAULT_VECTOR_DIMENSION = 768
    DEFAULT_MAX_RETRIES = 3

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6381,  # FalkorDB default port
        db: int = 0,
        graph_name: str = "zapomni_memory",
        password: Optional[str] = None,
        pool_size: int = DEFAULT_POOL_SIZE,
        max_retries: int = DEFAULT_MAX_RETRIES
    ):
        """
        Initialize FalkorDB client with connection pool.

        Args:
            host: FalkorDB host address
            port: FalkorDB port (Redis protocol)
            db: Database index (0-15)
            graph_name: Name of graph to use
            password: Optional Redis password
            pool_size: Connection pool size
            max_retries: Maximum retry attempts

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
        self.pool_size = pool_size
        self.max_retries = max_retries

        # Initialize state
        self._pool = None
        self.graph = None
        self._schema_manager = None
        self._initialized = False
        self._schema_ready = False
        self._closed = False
        self._logger = logger.bind(
            host=host,
            port=port,
            graph=graph_name
        )

        # Lazy connection - will connect on first operation
        try:
            self._init_connection()
        except Exception as e:
            self._logger.warning("lazy_connection_deferred", error=str(e))

    def _init_connection(self):
        """Initialize connection pool (lazy)."""
        try:
            # Create FalkorDB connection
            self._pool = FalkorDB(
                host=self.host,
                port=self.port,
                password=self.password
            )

            # Select graph
            self.graph = self._pool.select_graph(self.graph_name)

            self._initialized = True

            # Initialize schema (vector indexes, constraints)
            self._init_schema()

            self._schema_ready = True
        except Exception as e:
            self._logger.error("connection_failed", error=str(e))
            raise ConnectionError(f"Failed to connect to FalkorDB: {e}")

    def _init_schema(self):
        """Initialize graph schema using SchemaManager."""
        try:
            # Create SchemaManager instance with the graph
            self._schema_manager = SchemaManager(
                graph=self.graph,
                logger=self._logger
            )

            # Initialize schema (idempotent - safe to run multiple times)
            self._schema_manager.init_schema()

            self._logger.info("schema_initialized_via_manager")

        except Exception as e:
            self._logger.error("schema_initialization_failed", error=str(e))
            raise

    async def add_memory(self, memory: Memory) -> str:
        """
        Store a complete memory with chunks and embeddings in graph database.

        Args:
            memory: Memory object containing text, chunks, embeddings, metadata

        Returns:
            memory_id: UUID string identifying the stored memory

        Raises:
            ValidationError: If input validation fails
            DatabaseError: If database operations fail
            TransactionError: If transaction state is invalid
        """
        # STEP 1: INPUT VALIDATION

        # Validate text
        if not memory.text or not memory.text.strip():
            raise ValidationError("Validation failed for memory: text cannot be empty")

        if len(memory.text) > 1_000_000:
            raise ValidationError(
                f"Validation failed for memory: text exceeds max length (1,000,000)"
            )

        # Validate chunks
        if not memory.chunks:
            raise ValidationError("Validation failed for memory: chunks list cannot be empty")

        if len(memory.chunks) > 100:
            raise ValidationError(
                f"Validation failed for memory: too many chunks (max 100)"
            )

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
                        f"Validation failed for memory: embedding {i} contains non-numeric values"
                    )
                if not math.isfinite(value):
                    raise ValidationError(
                        f"Validation failed for memory: embedding {i} contains NaN or Inf values"
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
        while retry_count <= self.max_retries:
            try:
                # Mock transaction execution (real implementation would use FalkorDB)
                await self._execute_transaction(memory, memory_id, chunk_ids)

                self._logger.info(
                    "memory_added",
                    memory_id=memory_id,
                    num_chunks=len(memory.chunks)
                )

                return memory_id

            except (ConnectionError, TimeoutError) as e:
                retry_count += 1
                if retry_count > self.max_retries:
                    raise DatabaseError(
                        f"Database error during add_memory: connection failed after {self.max_retries} retries: {e}"
                    )

                backoff_seconds = 2 ** (retry_count - 1)
                self._logger.warning(
                    f"retry_attempt",
                    attempt=retry_count,
                    delay=backoff_seconds
                )
                await asyncio.sleep(backoff_seconds)

            except Exception as e:
                self._logger.error("add_memory_error", error=str(e))
                raise DatabaseError(f"Unexpected database error during add_memory: {e}")

    async def _execute_transaction(self, memory: Memory, memory_id: str, chunk_ids: List[str]):
        """Execute transaction to store memory."""
        if not self._initialized:
            raise ConnectionError("Not initialized")

        # Build Cypher query for atomic transaction
        cypher = """
        // Create Memory node
        CREATE (m:Memory {
            id: $memory_id,
            text: $text,
            tags: $tags,
            source: $source,
            metadata: $metadata,
            created_at: $timestamp
        })

        // Create Chunk nodes with embeddings
        WITH m
        UNWIND $chunks_data AS chunk_data
        CREATE (c:Chunk {
            id: chunk_data.id,
            text: chunk_data.text,
            index: chunk_data.index,
            embedding: vecf32(chunk_data.embedding)
        })
        CREATE (m)-[:HAS_CHUNK {index: chunk_data.index}]->(c)

        RETURN m.id AS memory_id
        """

        # Prepare parameters
        import json
        chunks_data = [
            {
                "id": chunk_ids[i],
                "text": chunk.text,
                "index": chunk.index,
                "embedding": embedding
            }
            for i, (chunk, embedding) in enumerate(zip(memory.chunks, memory.embeddings))
        ]

        parameters = {
            "memory_id": memory_id,
            "text": memory.text,
            "tags": memory.metadata.get("tags", []),
            "source": memory.metadata.get("source", ""),
            "metadata": json.dumps(memory.metadata),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "chunks_data": chunks_data
        }

        # Execute query in thread pool (FalkorDB is sync)
        result = await asyncio.to_thread(self.graph.query, cypher, parameters)

        # Debug logging
        self._logger.debug(
            "transaction_result",
            result_set_len=len(result.result_set) if result else 0,
            nodes_created=result.nodes_created if result else 0,
            rels_created=result.relationships_created if result else 0
        )

        # Verify result (result_set should contain at least one row with memory_id)
        if result and len(result.result_set) > 0:
            # Success - memory and chunks created
            return memory_id
        else:
            raise DatabaseError(f"Memory creation failed: no records returned (result={result}, result_set={result.result_set if result else None})")

    async def vector_search(
        self,
        embedding: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Perform HNSW vector similarity search on chunk embeddings.

        Args:
            embedding: Query embedding vector (must be 768-dimensional)
            limit: Maximum number of results to return (1-1000)
            filters: Optional metadata filters

        Returns:
            List of SearchResult objects sorted by similarity (descending)

        Raises:
            ValidationError: If input validation fails
            QueryError: If database query fails
            DatabaseError: If database operation fails after retries
        """
        # STEP 1: INPUT VALIDATION

        # Validate embedding dimension
        if len(embedding) != 768:
            raise ValidationError(
                f"Validation failed: embedding dimension must be 768, got {len(embedding)}"
            )

        # Validate embedding values
        if not all(math.isfinite(x) for x in embedding):
            raise ValidationError(
                "Validation failed: embedding contains NaN or Inf values"
            )

        # Validate limit
        if not isinstance(limit, int):
            raise ValidationError(
                f"Validation failed: limit must be int, got {type(limit).__name__}"
            )

        if limit < 1:
            raise ValidationError(
                f"Validation failed: limit must be >= 1, got {limit}"
            )

        if limit > 1000:
            raise ValidationError(
                f"Validation failed: limit cannot exceed 1000, got {limit}"
            )

        # Validate filters
        if filters is not None:
            if not isinstance(filters, dict):
                raise ValidationError("Validation failed: filters must be dict or None")

            # Validate tags
            if "tags" in filters:
                tags = filters["tags"]
                if not isinstance(tags, list) or len(tags) == 0:
                    raise ValidationError(
                        "Validation failed: filters['tags'] cannot be empty list"
                    )

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
                    datetime.fromisoformat(filters["date_from"].replace('Z', '+00:00'))
                except ValueError as e:
                    raise ValidationError(
                        f"Validation failed: filters['date_from'] invalid ISO 8601 format: {e}"
                    )

            if "date_to" in filters:
                try:
                    datetime.fromisoformat(filters["date_to"].replace('Z', '+00:00'))
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
            min_similarity=min_similarity
        )

        # STEP 3: EXECUTE SEARCH WITH RETRY
        retry_count = 0
        while retry_count <= self.max_retries:
            try:
                result = await self._execute_cypher(cypher, params)
                return self._parse_search_results(result)

            except (ConnectionError, TimeoutError) as e:
                retry_count += 1
                if retry_count > self.max_retries:
                    raise DatabaseError(
                        f"Database operation failed after {self.max_retries} retries: {e}"
                    )

                backoff_seconds = 2 ** (retry_count - 1)
                await asyncio.sleep(backoff_seconds)

            except Exception as e:
                self._logger.error("vector_search_error", error=str(e))
                raise DatabaseError(f"Unexpected database error: {e}")

    def _parse_search_results(self, query_result: QueryResult) -> List[SearchResult]:
        """Parse database results into SearchResult objects."""
        results = []

        for row in query_result.rows:
            try:
                timestamp_str = row.get("timestamp", datetime.now(timezone.utc).isoformat())
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))

                # Parse tags from JSON string if needed
                tags = row.get("tags", [])
                if isinstance(tags, str):
                    import json
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
                    chunk_index=int(row.get("chunk_index", 0))
                )
                results.append(search_result)
            except (KeyError, ValueError, TypeError) as e:
                self._logger.warning("malformed_search_row", row=row, error=str(e))
                continue

        results.sort(key=lambda r: r.similarity_score, reverse=True)
        return results

    async def get_stats(self) -> Dict[str, Any]:
        """
        Retrieve comprehensive statistics about the knowledge graph.

        Returns:
            Dict containing node counts, relationship counts, storage metrics,
            index statistics, and health indicators.

        Raises:
            DatabaseError: If database queries fail
            ConnectionError: If database connection is lost
        """
        stats = {
            "nodes": {},
            "relationships": {},
            "storage": {},
            "indexes": {},
            "health": {}
        }

        start_time = time.time()

        try:
            # Count nodes by type
            node_results = await self._execute_cypher("MATCH (n) RETURN labels(n)[0] AS node_type, count(n) AS count", {})

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
            rel_results = await self._execute_cypher("MATCH ()-[r]->() RETURN type(r) AS rel_type, count(r) AS count", {})

            stats["relationships"]["total"] = 0
            stats["relationships"]["has_chunk"] = 0
            stats["relationships"]["mentions"] = 0
            stats["relationships"]["related_to"] = 0

            for row in rel_results.rows:
                rel_type = row["rel_type"].lower().replace("_", "_") if row.get("rel_type") else "unknown"
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
                    {}
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

            self._logger.info(
                "stats_retrieved",
                total_nodes=stats["nodes"]["total"]
            )

            return stats

        except Exception as e:
            self._logger.error("stats_error", error=str(e))
            raise DatabaseError(f"Failed to retrieve graph statistics: {e}")

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
        # Validation happens via Pydantic model
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
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        retry_count = 0
        while retry_count <= self.max_retries:
            try:
                result = await self._execute_cypher(cypher, parameters)

                if result.row_count > 0:
                    self._logger.info("entity_added", entity_id=entity_id, name=entity.name)
                    return entity_id
                else:
                    raise DatabaseError(f"Entity creation failed: no records returned")

            except (ConnectionError, TimeoutError) as e:
                retry_count += 1
                if retry_count > self.max_retries:
                    raise DatabaseError(
                        f"Database error during add_entity: connection failed after {self.max_retries} retries: {e}"
                    )
                backoff_seconds = 2 ** (retry_count - 1)
                await asyncio.sleep(backoff_seconds)

            except Exception as e:
                self._logger.error("add_entity_error", error=str(e))
                raise DatabaseError(f"Unexpected database error during add_entity: {e}")

    async def add_relationship(
        self,
        from_entity_id: str,
        to_entity_id: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None
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

        # Note: Cypher doesn't support parameterized relationship types in this way
        # We need to use string interpolation for the relationship type
        # Also: FalkorDB doesn't support datetime() function - use ISO format string instead
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
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        retry_count = 0
        while retry_count <= self.max_retries:
            try:
                result = await self._execute_cypher(cypher, parameters)

                if result.row_count > 0:
                    self._logger.info(
                        "relationship_added",
                        relationship_id=relationship_id,
                        type=relationship_type
                    )
                    return relationship_id
                else:
                    raise DatabaseError(
                        f"Relationship creation failed: entities not found "
                        f"(from: {from_entity_id}, to: {to_entity_id})"
                    )

            except (ConnectionError, TimeoutError) as e:
                retry_count += 1
                if retry_count > self.max_retries:
                    raise DatabaseError(
                        f"Database error during add_relationship: connection failed after {self.max_retries} retries: {e}"
                    )
                backoff_seconds = 2 ** (retry_count - 1)
                await asyncio.sleep(backoff_seconds)

            except Exception as e:
                self._logger.error("add_relationship_error", error=str(e))
                raise DatabaseError(f"Unexpected database error during add_relationship: {e}")

    async def get_related_entities(
        self,
        entity_id: str,
        depth: int = 1,
        limit: int = 20
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
                    confidence=float(row.get("confidence", 1.0))
                )
                entities.append(entity)

            self._logger.info(
                "related_entities_retrieved",
                entity_id=entity_id,
                count=len(entities)
            )

            return entities

        except Exception as e:
            self._logger.error("get_related_entities_error", error=str(e))
            raise DatabaseError(f"Failed to retrieve related entities: {e}")

    async def graph_query(
        self,
        cypher: str,
        parameters: Optional[Dict[str, Any]] = None
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
            raise ValidationError(f"cypher query exceeds max length (100,000 chars)")

        # Validate parameters are JSON-serializable
        if parameters:
            try:
                json.dumps(parameters)
            except (TypeError, ValueError) as e:
                raise ValidationError(f"parameters must be JSON-serializable: {e}")

        # Execute query with retry
        retry_count = 0
        while retry_count <= self.max_retries:
            try:
                result = await self._execute_cypher(cypher, parameters or {})

                self._logger.info(
                    "graph_query_executed",
                    row_count=result.row_count,
                    execution_time_ms=result.execution_time_ms
                )

                return result

            except (ConnectionError, TimeoutError) as e:
                retry_count += 1
                if retry_count > self.max_retries:
                    raise DatabaseError(
                        f"Database error during graph_query: connection failed after {self.max_retries} retries: {e}"
                    )
                backoff_seconds = 2 ** (retry_count - 1)
                await asyncio.sleep(backoff_seconds)

            except Exception as e:
                self._logger.error("graph_query_error", error=str(e))
                raise DatabaseError(f"Unexpected database error during graph_query: {e}")

    async def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory and its associated chunks.

        Args:
            memory_id: Memory UUID to delete

        Returns:
            True if deleted, False if memory not found

        Raises:
            ValidationError: If memory_id is invalid UUID
            DatabaseError: If delete operation fails
        """
        # Validate UUID
        try:
            uuid.UUID(memory_id)
        except ValueError as e:
            raise ValidationError(f"Invalid memory UUID: {e}")

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
                self._logger.info("memory_deleted", memory_id=memory_id)
                return True
            else:
                self._logger.warning("memory_not_found", memory_id=memory_id)
                return False

        except Exception as e:
            self._logger.error("delete_memory_error", error=str(e))
            raise DatabaseError(f"Failed to delete memory: {e}")

    async def clear_all(self) -> None:
        """
        Clear all data from the graph (DELETE all nodes/edges).

        WARNING: This is destructive and irreversible.

        Raises:
            DatabaseError: If clear operation fails
        """
        cypher = "MATCH (n) DETACH DELETE n"

        try:
            result = await self._execute_cypher(cypher, {})
            self._logger.warning("graph_cleared", nodes_deleted=result.row_count)

        except Exception as e:
            self._logger.error("clear_all_error", error=str(e))
            raise DatabaseError(f"Failed to clear graph: {e}")

    async def _execute_cypher(self, query: str, parameters: Dict[str, Any]) -> QueryResult:
        """Execute Cypher query using FalkorDB."""
        if not self._initialized:
            raise ConnectionError("Not initialized")

        # Execute query in thread pool (FalkorDB is sync)
        result = await asyncio.to_thread(self.graph.query, query, parameters)

        # Convert FalkorDB result to our QueryResult format
        rows = []
        for record in result.result_set:
            # Convert result record to dict
            row_dict = {}
            for i, col_header in enumerate(result.header):
                # col_header is [type_id, column_name]
                col_name = col_header[1]
                row_dict[col_name] = record[i]
            rows.append(row_dict)

        return QueryResult(
            rows=rows,
            row_count=len(rows),
            execution_time_ms=int(result.run_time_ms) if hasattr(result, 'run_time_ms') else 0
        )

    def close(self) -> None:
        """
        Close database connection and release resources.

        Idempotent - can be called multiple times safely.
        """
        if self._closed:
            return

        try:
            if self._pool:
                # FalkorDB doesn't have close() method, it uses Redis connection
                # Just mark as closed
                self._logger.info("connection_closed")

            self._closed = True
            self._initialized = False

        except Exception as e:
            self._logger.warning("close_error", error=str(e))
