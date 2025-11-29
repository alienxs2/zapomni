"""
SchemaManager - Manages database schema for Zapomni FalkorDB graph.

This module provides idempotent schema initialization, vector index creation,
graph schema definition, and future migration support.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from typing import Any, Dict, List, Optional

import structlog
from falkordb import Graph
from structlog.stdlib import BoundLogger

from zapomni_db.exceptions import DatabaseError, QuerySyntaxError


class SchemaManager:
    """
    Manages database schema for Zapomni FalkorDB graph.

    Provides idempotent schema initialization, vector index creation,
    graph schema definition, and future migration support. All operations
    are safe to run multiple times without errors.

    Attributes:
        graph: FalkorDB Graph instance for executing schema operations
        logger: Structured logger for schema operations
        schema_version: Current schema version (e.g., "1.0.0")
        initialized: Flag indicating if schema has been initialized

    Example:
        ```python
        from falkordb import FalkorDB

        db = FalkorDB(host="localhost", port=6381)
        graph = db.select_graph("zapomni_memory")

        schema_manager = SchemaManager(graph=graph)
        schema_manager.init_schema()  # Idempotent - safe to run multiple times

        # Verify schema is ready
        status = schema_manager.verify_schema()
        print(f"Schema version: {status['version']}")
        print(f"Indexes: {status['indexes']}")
        ```
    """

    # Class constants
    SCHEMA_VERSION: str = "1.0.0"
    VECTOR_DIMENSION: int = 768  # nomic-embed-text dimension
    SIMILARITY_FUNCTION: str = "cosine"

    # Node labels
    NODE_MEMORY: str = "Memory"
    NODE_CHUNK: str = "Chunk"
    NODE_ENTITY: str = "Entity"
    NODE_DOCUMENT: str = "Document"

    # Edge labels
    EDGE_HAS_CHUNK: str = "HAS_CHUNK"
    EDGE_MENTIONS: str = "MENTIONS"
    EDGE_RELATED_TO: str = "RELATED_TO"
    EDGE_CALLS: str = "CALLS"

    # Index names
    INDEX_VECTOR: str = "chunk_embedding_idx"
    INDEX_MEMORY_ID: str = "memory_id_idx"
    INDEX_ENTITY_NAME: str = "entity_name_idx"
    INDEX_TIMESTAMP: str = "timestamp_idx"
    INDEX_MEMORY_STALE: str = "memory_stale_idx"
    INDEX_MEMORY_FILE_PATH: str = "memory_file_path_idx"
    INDEX_MEMORY_QUALIFIED_NAME: str = "memory_qualified_name_idx"

    def __init__(self, graph: Graph, logger: Optional[BoundLogger] = None) -> None:
        """
        Initialize SchemaManager.

        Args:
            graph: FalkorDB Graph instance for schema operations
            logger: Optional structured logger (defaults to new logger)

        Raises:
            TypeError: If graph is not a FalkorDB Graph instance
        """
        if not isinstance(graph, Graph):
            raise TypeError("graph must be a FalkorDB Graph instance")

        self.graph = graph
        self.logger = logger or structlog.get_logger(__name__)
        self.schema_version = self.SCHEMA_VERSION
        self.initialized = False

    def init_schema(self) -> None:
        """
        Initialize complete database schema (idempotent).

        Creates all necessary indexes and schema elements in correct order:
        1. Vector indexes (for embeddings)
        2. Graph schema (node/edge labels)
        3. Property indexes (for fast lookups)

        This method is idempotent - safe to call multiple times.
        Existing indexes are skipped, new ones are created.

        Raises:
            DatabaseError: If schema initialization fails
            QuerySyntaxError: If Cypher queries are malformed
        """
        # Skip if already initialized
        if self.initialized:
            self.logger.debug("Schema already initialized, skipping")
            return

        self.logger.info("Initializing database schema", version=self.schema_version)

        try:
            # Step 1: Create vector index
            self.create_vector_index()

            # Step 2: Create graph schema (node/edge labels)
            self.create_graph_schema()

            # Step 3: Create property indexes
            self.create_property_indexes()

            # Mark as initialized
            self.initialized = True

            self.logger.info("Database schema initialized successfully")

        except (DatabaseError, QuerySyntaxError) as e:
            self.logger.error("Schema initialization failed", error=str(e))
            raise

    def create_vector_index(self) -> None:
        """
        Create HNSW vector index for chunk embeddings (idempotent).

        Creates a vector index on Chunk nodes for fast approximate
        nearest neighbor search using HNSW algorithm.

        Index configuration:
        - Node label: Chunk
        - Property: embedding
        - Dimension: 768 (nomic-embed-text)
        - Similarity: cosine
        - M: 16 (HNSW graph degree)
        - EF_construction: 200 (build quality)

        Raises:
            DatabaseError: If index creation fails
            QuerySyntaxError: If Cypher syntax invalid
        """
        self.logger.debug("Creating vector index for chunk embeddings")

        # Check if index already exists (idempotent)
        if self._index_exists(self.INDEX_VECTOR):
            self.logger.debug("Vector index already exists, skipping", index_name=self.INDEX_VECTOR)
            return

        # Build CREATE VECTOR INDEX Cypher query
        # Note: FalkorDB does not support named vector indexes in CREATE syntax
        # Index name is auto-generated from label:property
        cypher_query = f"""
            CREATE VECTOR INDEX FOR (c:{self.NODE_CHUNK})
            ON (c.embedding)
            OPTIONS {{
                dimension: {self.VECTOR_DIMENSION},
                similarityFunction: '{self.SIMILARITY_FUNCTION}',
                M: 16,
                efConstruction: 200
            }}
        """

        try:
            self._execute_cypher(cypher_query)
        except Exception as e:
            # "already indexed" is expected for idempotent operation
            if "already indexed" in str(e).lower():
                self.logger.debug(
                    "Vector index already exists (detected via error)", index_name=self.INDEX_VECTOR
                )
                return
            raise

        self.logger.info(
            "Vector index created successfully",
            index_name=self.INDEX_VECTOR,
            dimension=self.VECTOR_DIMENSION,
        )

    def create_graph_schema(self) -> None:
        """
        Create graph schema (node and edge labels) (idempotent).

        Defines node types and edge types used in knowledge graph:

        Node Labels:
        - Memory: Main memory/document node
        - Chunk: Text chunk with embedding
        - Entity: Named entity (person, org, concept, etc.)
        - Document: Source document metadata

        Edge Labels:
        - HAS_CHUNK: Memory → Chunk (one-to-many)
        - MENTIONS: Chunk → Entity (many-to-many)
        - RELATED_TO: Entity → Entity (semantic relationships)

        Note: FalkorDB creates labels implicitly on first use,
        so this method mainly documents the schema structure.

        Raises:
            DatabaseError: If schema creation fails
        """
        self.logger.debug("Defining graph schema (node/edge labels)")

        # Document node labels
        node_labels = [self.NODE_MEMORY, self.NODE_CHUNK, self.NODE_ENTITY, self.NODE_DOCUMENT]

        # Document edge labels
        edge_labels = [
            self.EDGE_HAS_CHUNK,
            self.EDGE_MENTIONS,
            self.EDGE_RELATED_TO,
            self.EDGE_CALLS,
        ]

        self.logger.info("Graph schema defined", node_labels=node_labels, edge_labels=edge_labels)

    def create_property_indexes(self) -> None:
        """
        Create property indexes for fast lookups (idempotent).

        Creates indexes on frequently queried properties:

        Indexes created:
        1. Memory.id - UUID lookups (exact match)
        2. Entity.name - Entity name lookups (exact match)
        3. Memory.timestamp - Date range queries
        4. Chunk.memory_id - Chunk-to-memory lookups

        Raises:
            DatabaseError: If index creation fails
            QuerySyntaxError: If Cypher syntax invalid
        """
        self.logger.debug("Creating property indexes")

        # Define indexes to create
        indexes = [
            (self.INDEX_MEMORY_ID, self.NODE_MEMORY, "id"),
            (self.INDEX_ENTITY_NAME, self.NODE_ENTITY, "name"),
            (self.INDEX_TIMESTAMP, self.NODE_MEMORY, "timestamp"),
            ("chunk_memory_id_idx", self.NODE_CHUNK, "memory_id"),
            # Garbage collection indexes
            (self.INDEX_MEMORY_STALE, self.NODE_MEMORY, "stale"),
            (self.INDEX_MEMORY_FILE_PATH, self.NODE_MEMORY, "file_path"),
            # Index for call graph queries
            (self.INDEX_MEMORY_QUALIFIED_NAME, self.NODE_MEMORY, "qualified_name"),
        ]

        created_count = 0

        for index_name, node_label, property_name in indexes:
            # Check if index already exists (idempotent)
            if self._index_exists(index_name):
                self.logger.debug("Property index already exists, skipping", index_name=index_name)
                continue

            # Create property index
            # Note: FalkorDB does not support named indexes in CREATE syntax
            cypher_query = f"""
                CREATE INDEX FOR (n:{node_label})
                ON (n.{property_name})
            """

            try:
                self._execute_cypher(cypher_query)
                created_count += 1
            except Exception as e:
                # "already indexed" is expected for idempotent operation
                if "already indexed" in str(e).lower():
                    self.logger.debug(
                        "Property index already exists (detected via error)", index_name=index_name
                    )
                    continue
                raise

        self.logger.info(
            "Property indexes created", created_count=created_count, total_indexes=len(indexes)
        )

    def verify_schema(self) -> Dict[str, Any]:
        """
        Verify schema consistency and return status.

        Checks that all required indexes exist and are configured correctly.
        Returns detailed status for debugging and monitoring.

        Returns:
            Dictionary with keys:
                - version: str (schema version, e.g., "1.0.0")
                - initialized: bool (true if schema ready)
                - indexes: Dict[str, Dict] (index name -> config)
                    - vector_index: {exists: bool, dimension: int, similarity: str}
                    - property_indexes: {name: {exists: bool, property: str}}
                - node_labels: List[str] (defined node labels)
                - edge_labels: List[str] (defined edge labels)
                - issues: List[str] (problems found, empty if OK)

        Raises:
            DatabaseError: If verification queries fail
        """
        self.logger.debug("Verifying schema consistency")

        status: Dict[str, Any] = {
            "version": self.schema_version,
            "initialized": False,
            "indexes": {"vector_index": {}, "property_indexes": {}},
            "node_labels": [],
            "edge_labels": [],
            "issues": [],
        }

        try:
            # Query all indexes
            existing_indexes: Dict[str, List[Any]] = {}
            try:
                result = self.graph.query("SHOW INDEXES")
                if result.result_set:
                    for row in result.result_set:
                        index_name = row[0]
                        existing_indexes[index_name] = row
            except Exception:
                # FalkorDB may not support SHOW INDEXES, assume no indexes exist
                self.logger.debug("Could not query indexes, assuming none exist")

            # Check vector index
            if self.INDEX_VECTOR in existing_indexes:
                status["indexes"]["vector_index"] = {
                    "exists": True,
                    "name": self.INDEX_VECTOR,
                    "dimension": self.VECTOR_DIMENSION,
                    "similarity": self.SIMILARITY_FUNCTION,
                }
            else:
                status["indexes"]["vector_index"] = {
                    "exists": False,
                    "name": self.INDEX_VECTOR,
                    "dimension": 0,
                    "similarity": "",
                }
                status["issues"].append("Vector index not found")

            # Check property indexes
            required_prop_indexes = [
                (self.INDEX_MEMORY_ID, "id"),
                (self.INDEX_ENTITY_NAME, "name"),
                (self.INDEX_TIMESTAMP, "timestamp"),
            ]

            for index_name, property_name in required_prop_indexes:
                if index_name in existing_indexes:
                    status["indexes"]["property_indexes"][index_name] = {
                        "exists": True,
                        "property": property_name,
                    }
                else:
                    status["indexes"]["property_indexes"][index_name] = {
                        "exists": False,
                        "property": property_name,
                    }
                    status["issues"].append(f"Property index {index_name} not found")

            # Set node and edge labels
            status["node_labels"] = [
                self.NODE_MEMORY,
                self.NODE_CHUNK,
                self.NODE_ENTITY,
                self.NODE_DOCUMENT,
            ]
            status["edge_labels"] = [
                self.EDGE_HAS_CHUNK,
                self.EDGE_MENTIONS,
                self.EDGE_RELATED_TO,
                self.EDGE_CALLS,
            ]

            # Set initialized flag
            status["initialized"] = len(status["issues"]) == 0

            self.logger.info(
                "Schema verification complete",
                initialized=status["initialized"],
                issues_count=len(status["issues"]),
            )

            return status

        except Exception as e:
            self.logger.error("Schema verification failed", error=str(e))
            raise DatabaseError(f"Failed to verify schema: {str(e)}") from e

    def migrate(self, from_version: str, to_version: str) -> None:
        """
        Migrate schema from one version to another (future).

        FUTURE FEATURE - Not implemented in MVP.

        Will support safe schema migrations:
        - Add new indexes
        - Add new node/edge labels
        - Modify properties (backward-compatible)
        - Data transformations

        Args:
            from_version: Current schema version (e.g., "1.0.0")
            to_version: Target schema version (e.g., "1.1.0")

        Raises:
            NotImplementedError: Always (not implemented yet)
        """
        raise NotImplementedError("Schema migration not implemented yet")

    def drop_all(self) -> None:
        """
        Drop all schema and data (DANGEROUS - testing only).

        WARNING: This is destructive and irreversible.
        Deletes ALL nodes, edges, and indexes.

        Use cases:
        - Integration tests (clean slate)
        - Development reset
        - Emergency data wipe

        DO NOT use in production!

        Raises:
            DatabaseError: If drop operation fails
        """
        self.logger.warning("Dropping all data and schema (irreversible)")

        try:
            # Delete all nodes and edges
            self._execute_cypher("MATCH (n) DETACH DELETE n")

            # Drop all indexes (query for existing indexes first)
            try:
                result = self.graph.query("SHOW INDEXES")
                if result.result_set:
                    for row in result.result_set:
                        index_name = row[0]
                        try:
                            self._execute_cypher(f"DROP INDEX {index_name}")
                        except Exception as e:
                            self.logger.warning(
                                "Failed to drop index", index_name=index_name, error=str(e)
                            )
            except Exception as e:
                # FalkorDB may not support SHOW INDEXES, skip index dropping
                self.logger.debug("Could not query indexes for dropping", error=str(e))

            # Reset initialized flag
            self.initialized = False

            self.logger.info("All data and schema dropped")

        except Exception as e:
            self.logger.error("Failed to drop schema", error=str(e))
            raise DatabaseError(f"Failed to drop schema: {str(e)}") from e

    # Private helper methods

    def _index_exists(self, index_name: str) -> bool:
        """
        Check if an index exists (private helper).

        Args:
            index_name: Name of index to check

        Returns:
            True if index exists, False otherwise

        Raises:
            DatabaseError: If query fails
        """
        try:
            result = self.graph.query("SHOW INDEXES")
            if result.result_set:
                for row in result.result_set:
                    if row[0] == index_name:
                        return True
            return False

        except Exception as e:
            # FalkorDB may not support SHOW INDEXES, return False to allow creation to proceed
            self.logger.debug(
                "Index check failed, assuming index does not exist",
                error=str(e),
                index_name=index_name,
            )
            return False

    def _execute_cypher(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> None:
        """
        Execute Cypher query with error handling (private helper).

        Args:
            query: Cypher query string
            parameters: Optional query parameters

        Raises:
            DatabaseError: If query execution fails
            QuerySyntaxError: If Cypher syntax invalid
        """
        try:
            if parameters:
                self.graph.query(query, parameters)
            else:
                self.graph.query(query)

        except Exception as e:
            error_msg = str(e).lower()

            # Check if it's a syntax error
            if "syntax" in error_msg or "parse" in error_msg:
                raise QuerySyntaxError(f"Invalid Cypher syntax: {str(e)}") from e

            # Otherwise, generic database error
            raise DatabaseError(f"Query execution failed: {str(e)}") from e
