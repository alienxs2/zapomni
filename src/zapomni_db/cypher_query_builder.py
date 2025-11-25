"""
CypherQueryBuilder - Type-safe parameterized Cypher query generation for FalkorDB.

Provides SQL/Cypher-injection safe query building with parameter validation.
All queries return (cypher_string, parameters_dict) tuples for safe execution.

Author: Goncharenko Anton aka alienxs2
TDD Implementation: Code written to pass tests from specifications.
"""

import uuid
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from zapomni_db.exceptions import ValidationError


class CypherQueryBuilder:
    """
    Type-safe Cypher query builder for FalkorDB operations.

    Generates parameterized Cypher queries to prevent injection attacks and
    ensure consistent query patterns across the Zapomni memory system.

    All queries return a tuple of (cypher_string, parameters_dict) where
    parameters are passed separately to FalkorDB for safe execution.

    Attributes:
        VECTOR_INDEX_NAME: Name of the vector index for embeddings
        VECTOR_DIMENSION: Expected embedding vector dimension (768 for nomic-embed-text)
        DEFAULT_SIMILARITY_FUNCTION: Similarity function for vector search (cosine)

    Security Notes:
        - NEVER concatenate user input into Cypher strings
        - ALWAYS use parameterized queries ($param_name)
        - All UUIDs validated before use
        - All embeddings validated for correct dimensions
        - Filter values sanitized and parameterized
    """

    # Class constants
    VECTOR_INDEX_NAME: str = "chunk_embedding_idx"
    VECTOR_DIMENSION: int = 768  # nomic-embed-text dimension
    DEFAULT_SIMILARITY_FUNCTION: str = "cosine"

    def __init__(self) -> None:
        """
        Initialize CypherQueryBuilder.

        No configuration required - all settings are class constants.
        This is a stateless builder that can be instantiated once and reused.
        """
        pass

    def build_add_memory_query(
        self,
        memory: "Memory"
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate Cypher query to insert a Memory with its Chunks and embeddings.

        Creates a transaction-safe query that:
        1. Creates a Memory node with metadata
        2. Creates Chunk nodes with embeddings as vector properties
        3. Creates HAS_CHUNK relationships from Memory to each Chunk

        Args:
            memory: Memory object containing text, chunks, embeddings, metadata

        Returns:
            Tuple of (cypher_string, parameters_dict)
            - cypher_string: Parameterized Cypher query
            - parameters_dict: Parameters to pass to graph.query()

        Raises:
            ValidationError: If memory structure is invalid
            ValidationError: If embeddings have wrong dimension
            ValidationError: If chunks count != embeddings count

        Example:
            ```python
            builder = CypherQueryBuilder()
            memory = Memory(
                text="Python is great",
                chunks=[Chunk(text="Python is great", index=0)],
                embeddings=[[0.1] * 768],
                metadata={"source": "user", "tags": ["python"]}
            )
            cypher, params = builder.build_add_memory_query(memory)
            # Returns:
            # cypher = "CREATE (m:Memory {id: $memory_id, ...}) ..."
            # params = {"memory_id": "...", "text": "Python is great", ...}
            ```
        """
        # STEP 1: Validate memory structure
        if not memory.chunks:
            raise ValidationError("Memory chunks cannot be empty")

        if len(memory.chunks) != len(memory.embeddings):
            raise ValidationError(
                f"Chunks/embeddings mismatch: {len(memory.chunks)} chunks, "
                f"{len(memory.embeddings)} embeddings"
            )

        if not memory.text or not memory.text.strip():
            raise ValidationError("Memory text cannot be empty")

        # STEP 2: Validate embeddings
        for i, embedding in enumerate(memory.embeddings):
            self._validate_embedding(embedding)

        # STEP 3: Generate UUIDs
        memory_id = str(uuid.uuid4())
        chunk_ids = [str(uuid.uuid4()) for _ in range(len(memory.chunks))]

        # STEP 4: Build parameters
        created_at = datetime.now().isoformat()

        # Build chunks data array with embeddings
        chunks_data = []
        for i, (chunk, embedding) in enumerate(zip(memory.chunks, memory.embeddings)):
            chunks_data.append({
                "id": chunk_ids[i],
                "text": chunk.text,
                "index": chunk.index,
                "embedding": embedding
            })

        parameters = {
            "memory_id": memory_id,
            "text": memory.text,
            "source": memory.metadata.get("source", ""),
            "tags": memory.metadata.get("tags", []),
            "created_at": created_at,
            "chunks": chunks_data
        }

        # STEP 5: Build Cypher query
        cypher = """
        CREATE (m:Memory {
            id: $memory_id,
            text: $text,
            source: $source,
            tags: $tags,
            created_at: $created_at
        })
        WITH m
        UNWIND $chunks AS chunk_data
        CREATE (c:Chunk {
            id: chunk_data.id,
            text: chunk_data.text,
            index: chunk_data.index,
            embedding: vecf32(chunk_data.embedding)
        })
        CREATE (m)-[:HAS_CHUNK {index: chunk_data.index}]->(c)
        RETURN m.id
        """

        return (cypher, parameters)

    def build_vector_search_query(
        self,
        embedding: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        min_similarity: float = 0.5
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate Cypher query for vector similarity search using HNSW index.

        Uses FalkorDB's vector search function with parameterized filters
        to find semantically similar chunks based on embedding cosine similarity.

        Args:
            embedding: Query embedding vector (768-dimensional)
            limit: Maximum number of results to return (1-1000)
            filters: Optional metadata filters
                - tags: List[str] - Match any of these tags
                - source: str - Match exact source
                - date_from: datetime - Memories created after this date
                - date_to: datetime - Memories created before this date
            min_similarity: Minimum similarity threshold (0.0-1.0)

        Returns:
            Tuple of (cypher_string, parameters_dict)

        Raises:
            ValidationError: If embedding dimension != 768
            ValidationError: If limit < 1 or > 1000
            ValidationError: If min_similarity not in [0.0, 1.0]
            ValidationError: If filters have invalid structure

        Example:
            ```python
            cypher, params = builder.build_vector_search_query(
                embedding=[0.1] * 768,
                limit=10,
                filters={"tags": ["python", "coding"]},
                min_similarity=0.7
            )
            # Returns parameterized query with CALL db.idx.vector.queryNodes
            ```
        """
        # STEP 1: Validate inputs
        self._validate_embedding(embedding)

        if not isinstance(limit, int) or limit < 1 or limit > 1000:
            raise ValidationError(
                f"limit must be int in range [1, 1000], got {limit}"
            )

        if not isinstance(min_similarity, (int, float)) or not (0.0 <= min_similarity <= 1.0):
            raise ValidationError(
                f"min_similarity must be in [0.0, 1.0], got {min_similarity}"
            )

        # STEP 2: Build parameters with base query parameters
        parameters = {
            "query_embedding": embedding,
            "limit": limit,
            "min_similarity": min_similarity
        }

        # STEP 3: Build filter clause and merge parameters
        filter_clause, filter_params = self._build_filter_clause(filters)
        parameters.update(filter_params)

        # STEP 4: Build Cypher query
        # FalkorDB queryNodes signature: (label, attribute, k, query_vector)
        cypher = f"""
        CALL db.idx.vector.queryNodes(
            'Chunk',
            'embedding',
            $limit,
            vecf32($query_embedding)
        ) YIELD node AS c, score
        MATCH (m:Memory)-[:HAS_CHUNK]->(c)
        WHERE score >= $min_similarity
        {filter_clause}
        RETURN m.id AS memory_id,
               c.text AS text,
               score AS similarity_score,
               m.tags AS tags,
               m.source AS source,
               m.created_at AS timestamp
        ORDER BY score DESC
        """

        return (cypher, parameters)

    def build_graph_traversal_query(
        self,
        entity_id: str,
        depth: int = 1,
        limit: int = 20
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate Cypher query for graph traversal to find related entities.

        Performs variable-length pattern matching to find entities connected
        to the starting entity via relationships at specified depth.

        Args:
            entity_id: Starting entity UUID
            depth: Traversal depth (1-3 hops recommended, max 5)
            limit: Maximum number of related entities to return

        Returns:
            Tuple of (cypher_string, parameters_dict)

        Raises:
            ValidationError: If entity_id is not valid UUID
            ValidationError: If depth < 1 or > 5
            ValidationError: If limit < 1 or > 100

        Example:
            ```python
            cypher, params = builder.build_graph_traversal_query(
                entity_id="550e8400-e29b-41d4-a716-446655440000",
                depth=2,
                limit=20
            )
            # Returns:
            # MATCH (start:Entity {id: $entity_id})
            # MATCH (start)-[*1..2]-(related:Entity)
            # RETURN DISTINCT related ...
            ```

        Notes:
            - Depth 1: Direct neighbors only
            - Depth 2: Neighbors and neighbors-of-neighbors
            - Depth 3+: Multi-hop traversal (can be expensive)
            - Results sorted by relationship strength (if available)
        """
        # STEP 1: Validate inputs
        self._validate_uuid(entity_id)

        if not isinstance(depth, int) or depth < 1 or depth > 5:
            raise ValidationError(
                f"depth must be int in range [1, 5], got {depth}"
            )

        if not isinstance(limit, int) or limit < 1 or limit > 100:
            raise ValidationError(
                f"limit must be int in range [1, 100], got {limit}"
            )

        # STEP 2: Build parameters
        parameters = {
            "entity_id": entity_id,
            "limit": limit
        }

        # STEP 3: Build Cypher query with variable-length pattern
        pattern = f"[rels*1..{depth}]"

        cypher = f"""
        MATCH (start:Entity {{id: $entity_id}})
        MATCH (start)-{pattern}-(related:Entity)
        WHERE related.id <> $entity_id
        WITH DISTINCT related,
             reduce(strength = 1.0, r IN rels | strength * coalesce(r.strength, 1.0)) AS path_strength
        RETURN related.id AS entity_id,
               related.name AS name,
               related.type AS type,
               related.description AS description,
               path_strength
        ORDER BY path_strength DESC
        LIMIT $limit
        """

        return (cypher, parameters)

    def build_stats_query(self) -> Tuple[str, Dict[str, Any]]:
        """
        Generate Cypher query to retrieve database statistics.

        Returns counts of nodes by type, relationships, and graph metadata.

        Returns:
            Tuple of (cypher_string, parameters_dict)
            - parameters_dict will be empty (no parameters needed)

        Example:
            ```python
            cypher, params = builder.build_stats_query()
            result = graph.query(cypher, params)
            # Returns statistics about graph size, node counts, etc.
            ```

        Query Returns:
            - total_memories: Count of Memory nodes
            - total_chunks: Count of Chunk nodes
            - total_entities: Count of Entity nodes
            - total_relationships: Count of all relationships
        """
        cypher = """
        MATCH (m:Memory)
        WITH count(m) AS total_memories
        MATCH (c:Chunk)
        WITH total_memories, count(c) AS total_chunks
        MATCH (e:Entity)
        WITH total_memories, total_chunks, count(e) AS total_entities
        MATCH ()-[r]->()
        WITH total_memories, total_chunks, total_entities, count(r) AS total_relationships
        RETURN total_memories, total_chunks, total_entities, total_relationships
        """

        return (cypher, {})

    def build_delete_memory_query(
        self,
        memory_id: str
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate Cypher query to delete a Memory and its associated Chunks.

        Cascades delete to all Chunk nodes connected via HAS_CHUNK relationships.

        Args:
            memory_id: Memory UUID to delete

        Returns:
            Tuple of (cypher_string, parameters_dict)

        Raises:
            ValidationError: If memory_id is not valid UUID

        Example:
            ```python
            cypher, params = builder.build_delete_memory_query(
                memory_id="550e8400-e29b-41d4-a716-446655440000"
            )
            # Returns:
            # MATCH (m:Memory {id: $memory_id})
            # OPTIONAL MATCH (m)-[:HAS_CHUNK]->(c:Chunk)
            # DETACH DELETE m, c
            ```
        """
        # STEP 1: Validate UUID
        self._validate_uuid(memory_id)

        # STEP 2: Build parameters
        parameters = {
            "memory_id": memory_id
        }

        # STEP 3: Build Cypher query
        cypher = """
        MATCH (m:Memory {id: $memory_id})
        OPTIONAL MATCH (m)-[:HAS_CHUNK]->(c:Chunk)
        DETACH DELETE m, c
        RETURN count(m) AS deleted_count
        """

        return (cypher, parameters)

    def build_add_entity_query(
        self,
        entity: "Entity"
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate Cypher query to add an Entity node to the knowledge graph.

        Args:
            entity: Entity object (name, type, description, confidence)

        Returns:
            Tuple of (cypher_string, parameters_dict)

        Raises:
            ValidationError: If entity name is empty
            ValidationError: If entity type is empty
            ValidationError: If confidence not in [0.0, 1.0]

        Example:
            ```python
            entity = Entity(
                name="Python",
                type="TECHNOLOGY",
                description="Programming language",
                confidence=0.95
            )
            cypher, params = builder.build_add_entity_query(entity)
            # Returns:
            # CREATE (e:Entity {id: $entity_id, name: $name, ...})
            # RETURN e.id
            ```
        """
        # STEP 1: Validate entity fields
        if not entity.name or not entity.name.strip():
            raise ValidationError("Entity name cannot be empty")

        if not entity.type or not entity.type.strip():
            raise ValidationError("Entity type cannot be empty")

        if not isinstance(entity.confidence, (int, float)) or not (0.0 <= entity.confidence <= 1.0):
            raise ValidationError(
                f"Entity confidence must be in [0.0, 1.0], got {entity.confidence}"
            )

        # STEP 2: Generate UUID
        entity_id = str(uuid.uuid4())

        # STEP 3: Build parameters
        created_at = datetime.now().isoformat()
        parameters = {
            "entity_id": entity_id,
            "name": entity.name,
            "type": entity.type,
            "description": entity.description or "",
            "confidence": entity.confidence,
            "created_at": created_at
        }

        # STEP 4: Build Cypher query
        cypher = """
        CREATE (e:Entity {
            id: $entity_id,
            name: $name,
            type: $type,
            description: $description,
            confidence: $confidence,
            created_at: $created_at
        })
        RETURN e.id
        """

        return (cypher, parameters)

    def build_add_relationship_query(
        self,
        from_entity_id: str,
        to_entity_id: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate Cypher query to add a relationship between two entities.

        Args:
            from_entity_id: Source entity UUID
            to_entity_id: Target entity UUID
            relationship_type: Relationship type (e.g., "MENTIONS", "RELATED_TO")
                Must be uppercase alphanumeric with underscores
            properties: Optional edge properties (strength, confidence, etc.)

        Returns:
            Tuple of (cypher_string, parameters_dict)

        Raises:
            ValidationError: If entity IDs are not valid UUIDs
            ValidationError: If relationship_type is empty or has invalid format
            ValidationError: If properties are not JSON-serializable

        Example:
            ```python
            cypher, params = builder.build_add_relationship_query(
                from_entity_id="550e8400-e29b-41d4-a716-446655440000",
                to_entity_id="6ba7b810-9dad-11d1-80b4-00c04fd430c8",
                relationship_type="MENTIONS",
                properties={"strength": 0.8, "confidence": 0.9}
            )
            # Returns:
            # MATCH (from:Entity {id: $from_id})
            # MATCH (to:Entity {id: $to_id})
            # CREATE (from)-[r:MENTIONS {properties}]->(to)
            # RETURN r
            ```
        """
        # STEP 1: Validate entity UUIDs
        self._validate_uuid(from_entity_id)
        self._validate_uuid(to_entity_id)

        # STEP 2: Validate relationship type
        if not relationship_type or not relationship_type.strip():
            raise ValidationError("relationship_type cannot be empty")

        # Check format: uppercase alphanumeric with underscores
        if not re.match(r"^[A-Z_][A-Z0-9_]*$", relationship_type):
            raise ValidationError(
                f"relationship_type must match pattern [A-Z_][A-Z0-9_]*, got '{relationship_type}'"
            )

        # STEP 3: Build parameters
        relationship_id = str(uuid.uuid4())
        created_at = datetime.now().isoformat()

        parameters = {
            "from_id": from_entity_id,
            "to_id": to_entity_id,
            "rel_id": relationship_id,
            "created_at": created_at
        }

        # Extract properties with defaults
        if properties:
            strength = properties.get("strength", 1.0)
            confidence = properties.get("confidence", 1.0)
            context = properties.get("context", "")

            # Validate strength and confidence
            if not isinstance(strength, (int, float)) or not (0.0 <= strength <= 1.0):
                raise ValidationError(
                    f"properties['strength'] must be in [0.0, 1.0], got {strength}"
                )
            if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
                raise ValidationError(
                    f"properties['confidence'] must be in [0.0, 1.0], got {confidence}"
                )

            parameters["strength"] = strength
            parameters["confidence"] = confidence
            parameters["context"] = context
        else:
            parameters["strength"] = 1.0
            parameters["confidence"] = 1.0
            parameters["context"] = ""

        # STEP 4: Build Cypher query
        # Note: Relationship type must be hardcoded, not parameterized in Cypher
        cypher = f"""
        MATCH (from:Entity {{id: $from_id}})
        MATCH (to:Entity {{id: $to_id}})
        CREATE (from)-[r:{relationship_type} {{
            id: $rel_id,
            strength: $strength,
            confidence: $confidence,
            context: $context,
            created_at: $created_at
        }}]->(to)
        RETURN r.id
        """

        return (cypher, parameters)

    # Private helper methods

    def _validate_uuid(self, uuid_str: str) -> None:
        """
        Validate that string is a valid UUID4.

        Args:
            uuid_str: String to validate

        Raises:
            ValidationError: If uuid_str is not valid UUID4 format
        """
        try:
            uuid.UUID(uuid_str)
        except (ValueError, TypeError, AttributeError):
            raise ValidationError(f"Invalid UUID format: '{uuid_str}'")

    def _validate_embedding(self, embedding: List[float]) -> None:
        """
        Validate embedding vector dimensions and format.

        Args:
            embedding: Embedding vector to validate

        Raises:
            ValidationError: If embedding is not 768-dimensional
            ValidationError: If embedding contains non-numeric values
            ValidationError: If embedding is empty
        """
        if not embedding:
            raise ValidationError("Embedding cannot be empty")

        if len(embedding) != self.VECTOR_DIMENSION:
            raise ValidationError(
                f"Embedding dimension mismatch: expected {self.VECTOR_DIMENSION}, "
                f"got {len(embedding)}"
            )

        if not all(isinstance(x, (int, float)) for x in embedding):
            raise ValidationError("Embedding must contain only numeric values")

    def _build_filter_clause(
        self,
        filters: Optional[Dict[str, Any]]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Build parameterized WHERE clause from metadata filters.

        Args:
            filters: Dictionary of filter criteria

        Returns:
            Tuple of (where_clause_string, filter_parameters_dict)

        Example:
            Input: {"tags": ["python", "coding"], "source": "user"}
            Output: (
                "AND ($tag_0 IN m.tags OR $tag_1 IN m.tags) AND m.source = $source",
                {"tag_0": "python", "tag_1": "coding", "source": "user"}
            )

        Notes:
            - Empty filters return ("", {})
            - All values parameterized (no injection risk)
            - Tag filters use OR logic (match any tag)
            - Other filters use AND logic
        """
        if not filters:
            return ("", {})

        clause_parts = []
        params = {}

        # Handle tags filter (OR logic - match ANY tag)
        if "tags" in filters and filters["tags"]:
            tags = filters["tags"]
            if isinstance(tags, list) and len(tags) > 0:
                tag_conditions = []
                for i, tag in enumerate(tags):
                    param_name = f"tag_{i}"
                    params[param_name] = tag
                    tag_conditions.append(f"${param_name} IN m.tags")
                clause_parts.append(f"({' OR '.join(tag_conditions)})")

        # Handle source filter (exact match)
        if "source" in filters and filters["source"]:
            source = filters["source"]
            if isinstance(source, str):
                params["source"] = source
                clause_parts.append("m.source = $source")

        # Handle date_from filter
        if "date_from" in filters and filters["date_from"]:
            date_from = filters["date_from"]
            # Convert datetime to ISO string if needed
            if isinstance(date_from, datetime):
                date_from = date_from.isoformat()
            params["date_from"] = date_from
            clause_parts.append("m.created_at >= $date_from")

        # Handle date_to filter
        if "date_to" in filters and filters["date_to"]:
            date_to = filters["date_to"]
            # Convert datetime to ISO string if needed
            if isinstance(date_to, datetime):
                date_to = date_to.isoformat()
            params["date_to"] = date_to
            clause_parts.append("m.created_at <= $date_to")

        # Build final WHERE clause
        if clause_parts:
            where_clause = "AND " + " AND ".join(clause_parts)
            return (where_clause, params)

        return ("", params)
