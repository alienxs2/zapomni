"""
CypherQueryBuilder - Type-safe parameterized Cypher query generation for FalkorDB.

Provides SQL/Cypher-injection safe query building with parameter validation.
All queries return (cypher_string, parameters_dict) tuples for safe execution.

Author: Goncharenko Anton aka alienxs2
TDD Implementation: Code written to pass tests from specifications.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple

from zapomni_db.exceptions import ValidationError
from zapomni_db.models import DEFAULT_WORKSPACE_ID

if TYPE_CHECKING:
    from zapomni_db.models import Entity, Memory


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

    def build_add_memory_query(self, memory: "Memory") -> Tuple[str, Dict[str, Any]]:
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
            chunks_data.append(
                {
                    "id": chunk_ids[i],
                    "text": chunk.text,
                    "index": chunk.index,
                    "embedding": embedding,
                }
            )

        parameters = {
            "memory_id": memory_id,
            "text": memory.text,
            "source": memory.metadata.get("source", ""),
            "tags": memory.metadata.get("tags", []),
            "created_at": created_at,
            "chunks": chunks_data,
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
        min_similarity: float = 0.5,
        workspace_id: Optional[str] = None,
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
            workspace_id: Workspace ID for data isolation. Defaults to "default".

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
                min_similarity=0.7,
                workspace_id="my-workspace"
            )
            # Returns parameterized query with CALL db.idx.vector.queryNodes
            ```
        """
        # STEP 1: Validate inputs
        self._validate_embedding(embedding)

        if not isinstance(limit, int) or limit < 1 or limit > 1000:
            raise ValidationError(f"limit must be int in range [1, 1000], got {limit}")

        if not isinstance(min_similarity, (int, float)) or not (0.0 <= min_similarity <= 1.0):
            raise ValidationError(f"min_similarity must be in [0.0, 1.0], got {min_similarity}")

        # Determine effective workspace_id
        effective_workspace_id = workspace_id or DEFAULT_WORKSPACE_ID

        # STEP 2: Build parameters with base query parameters
        parameters = {
            "query_embedding": embedding,
            "limit": limit,
            "min_similarity": min_similarity,
            "workspace_id": effective_workspace_id,
        }

        # STEP 3: Build filter clause and merge parameters
        filter_clause, filter_params = self._build_filter_clause(filters)
        parameters.update(filter_params)

        # STEP 4: Build Cypher query
        # FalkorDB queryNodes signature: (label, attribute, k, query_vector)
        # Note: FalkorDB returns cosine DISTANCE (0=identical, 2=opposite)
        # Convert min_similarity to max_distance: max_distance = 1 - min_similarity
        # IMPORTANT: workspace_id filter must come AFTER YIELD clause (in-filtering pattern)
        # Bi-temporal: Only return current versions (is_current = true)
        cypher = f"""
        CALL db.idx.vector.queryNodes(
            'Chunk',
            'embedding',
            $limit,
            vecf32($query_embedding)
        ) YIELD node AS c, score
        WHERE c.workspace_id = $workspace_id
        MATCH (m:Memory)-[:HAS_CHUNK]->(c)
        WHERE score <= (1.0 - $min_similarity)
        AND m.workspace_id = $workspace_id
        AND m.is_current = true
        {filter_clause}
        RETURN m.id AS memory_id,
               c.id AS chunk_id,
               c.text AS text,
               (1.0 - score) AS similarity_score,
               m.tags AS tags,
               m.source AS source,
               m.created_at AS timestamp,
               c.index AS chunk_index,
               m.workspace_id AS workspace_id
        ORDER BY score ASC
        """

        return (cypher, parameters)

    def build_graph_traversal_query(
        self, entity_id: str, depth: int = 1, limit: int = 20
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
            raise ValidationError(f"depth must be int in range [1, 5], got {depth}")

        if not isinstance(limit, int) or limit < 1 or limit > 100:
            raise ValidationError(f"limit must be int in range [1, 100], got {limit}")

        # STEP 2: Build parameters
        parameters = {"entity_id": entity_id, "limit": limit}

        # STEP 3: Build Cypher query with variable-length pattern
        pattern = f"[rels*1..{depth}]"

        cypher = f"""
        MATCH (start:Entity {{id: $entity_id}})
        MATCH (start)-{pattern}-(related:Entity)
        WHERE related.id <> $entity_id
        WITH DISTINCT related,
             reduce(strength = 1.0, r IN rels |
                    strength * coalesce(r.strength, 1.0)) AS path_strength
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

    def build_delete_memory_query(self, memory_id: str) -> Tuple[str, Dict[str, Any]]:
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
        parameters = {"memory_id": memory_id}

        # STEP 3: Build Cypher query
        cypher = """
        MATCH (m:Memory {id: $memory_id})
        OPTIONAL MATCH (m)-[:HAS_CHUNK]->(c:Chunk)
        DETACH DELETE m, c
        RETURN count(m) AS deleted_count
        """

        return (cypher, parameters)

    def build_add_entity_query(self, entity: "Entity") -> Tuple[str, Dict[str, Any]]:
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
            "created_at": created_at,
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
        properties: Optional[Dict[str, Any]] = None,
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

        parameters: Dict[str, Any] = {
            "from_id": from_entity_id,
            "to_id": to_entity_id,
            "rel_id": relationship_id,
            "created_at": created_at,
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

    def _build_filter_clause(self, filters: Optional[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
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

    # ========================================
    # CALL GRAPH QUERIES
    # ========================================

    def build_create_calls_relationship(
        self,
        caller_qualified_name: str,
        callee_qualified_name: str,
        call_line: int,
        call_type: str,
        arguments_count: int,
        workspace_id: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Create CALLS relationship between two Memory nodes.

        Matches caller and callee Memory nodes by their qualified_name property
        and creates a CALLS edge with call site metadata.

        Args:
            caller_qualified_name: Qualified name of the calling function/method
            callee_qualified_name: Qualified name of the called function/method
            call_line: Line number where the call occurs
            call_type: Type of call (function, method, constructor, etc.)
            arguments_count: Number of arguments in the call
            workspace_id: Workspace ID for data isolation

        Returns:
            Tuple of (cypher_string, parameters_dict)

        Raises:
            ValidationError: If qualified names are empty
            ValidationError: If call_line < 0
            ValidationError: If workspace_id is empty

        Example:
            ```python
            cypher, params = builder.build_create_calls_relationship(
                caller_qualified_name="module.MyClass.process",
                callee_qualified_name="module.helper_func",
                call_line=42,
                call_type="function",
                arguments_count=2,
                workspace_id="default"
            )
            ```
        """
        # STEP 1: Validate inputs
        if not caller_qualified_name or not caller_qualified_name.strip():
            raise ValidationError("caller_qualified_name cannot be empty")

        if not callee_qualified_name or not callee_qualified_name.strip():
            raise ValidationError("callee_qualified_name cannot be empty")

        if call_line < 0:
            raise ValidationError(f"call_line must be >= 0, got {call_line}")

        if not workspace_id or not workspace_id.strip():
            raise ValidationError("workspace_id cannot be empty")

        # STEP 2: Build parameters
        relationship_id = str(uuid.uuid4())
        parameters = {
            "caller_qualified_name": caller_qualified_name,
            "callee_qualified_name": callee_qualified_name,
            "call_line": call_line,
            "call_type": call_type,
            "arguments_count": arguments_count,
            "workspace_id": workspace_id,
            "rel_id": relationship_id,
            "created_at": datetime.now().isoformat(),
        }

        # STEP 3: Build Cypher query
        # Match Memory nodes by qualified_name within the workspace
        # Use MERGE to avoid duplicate relationships for the same call site
        cypher = """
        MATCH (caller:Memory {qualified_name: $caller_qualified_name, workspace_id: $workspace_id})
        MATCH (callee:Memory {qualified_name: $callee_qualified_name, workspace_id: $workspace_id})
        MERGE (caller)-[r:CALLS {call_line: $call_line}]->(callee)
        ON CREATE SET
            r.id = $rel_id,
            r.call_type = $call_type,
            r.arguments_count = $arguments_count,
            r.created_at = $created_at
        ON MATCH SET
            r.call_type = $call_type,
            r.arguments_count = $arguments_count
        RETURN r.id AS relationship_id
        """

        return (cypher, parameters)

    def build_get_callers_query(
        self,
        qualified_name: str,
        workspace_id: str,
        limit: int = 50,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Find all functions that call the given function.

        Returns Memory nodes that have a CALLS relationship pointing
        to the target function, sorted by call frequency.

        Args:
            qualified_name: Qualified name of the target function
            workspace_id: Workspace ID for data isolation
            limit: Maximum number of results (1-100)

        Returns:
            Tuple of (cypher_string, parameters_dict)

        Raises:
            ValidationError: If qualified_name is empty
            ValidationError: If workspace_id is empty
            ValidationError: If limit is out of range

        Example:
            ```python
            cypher, params = builder.build_get_callers_query(
                qualified_name="module.helper_func",
                workspace_id="default",
                limit=20
            )
            # Returns all functions that call helper_func
            ```
        """
        # STEP 1: Validate inputs
        if not qualified_name or not qualified_name.strip():
            raise ValidationError("qualified_name cannot be empty")

        if not workspace_id or not workspace_id.strip():
            raise ValidationError("workspace_id cannot be empty")

        if not isinstance(limit, int) or limit < 1 or limit > 100:
            raise ValidationError(f"limit must be int in range [1, 100], got {limit}")

        # STEP 2: Build parameters
        parameters = {
            "qualified_name": qualified_name,
            "workspace_id": workspace_id,
            "limit": limit,
        }

        # STEP 3: Build Cypher query
        cypher = """
        MATCH (caller:Memory)-[r:CALLS]->(callee:Memory {
            qualified_name: $qualified_name, workspace_id: $workspace_id})
        WHERE caller.workspace_id = $workspace_id
        RETURN caller.qualified_name AS caller_qualified_name,
               caller.id AS caller_id,
               caller.file_path AS caller_file_path,
               r.call_line AS call_line,
               r.call_type AS call_type,
               r.arguments_count AS arguments_count,
               count(r) AS call_count
        ORDER BY call_count DESC
        LIMIT $limit
        """

        return (cypher, parameters)

    def build_get_callees_query(
        self,
        qualified_name: str,
        workspace_id: str,
        limit: int = 50,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Find all functions called by the given function.

        Returns Memory nodes that the target function calls via
        CALLS relationships, sorted by call frequency.

        Args:
            qualified_name: Qualified name of the calling function
            workspace_id: Workspace ID for data isolation
            limit: Maximum number of results (1-100)

        Returns:
            Tuple of (cypher_string, parameters_dict)

        Raises:
            ValidationError: If qualified_name is empty
            ValidationError: If workspace_id is empty
            ValidationError: If limit is out of range

        Example:
            ```python
            cypher, params = builder.build_get_callees_query(
                qualified_name="module.MyClass.process",
                workspace_id="default",
                limit=20
            )
            # Returns all functions that process() calls
            ```
        """
        # STEP 1: Validate inputs
        if not qualified_name or not qualified_name.strip():
            raise ValidationError("qualified_name cannot be empty")

        if not workspace_id or not workspace_id.strip():
            raise ValidationError("workspace_id cannot be empty")

        if not isinstance(limit, int) or limit < 1 or limit > 100:
            raise ValidationError(f"limit must be int in range [1, 100], got {limit}")

        # STEP 2: Build parameters
        parameters = {
            "qualified_name": qualified_name,
            "workspace_id": workspace_id,
            "limit": limit,
        }

        # STEP 3: Build Cypher query
        cypher = """
        MATCH (caller:Memory {
            qualified_name: $qualified_name, workspace_id: $workspace_id}
        )-[r:CALLS]->(callee:Memory)
        WHERE callee.workspace_id = $workspace_id
        RETURN callee.qualified_name AS callee_qualified_name,
               callee.id AS callee_id,
               callee.file_path AS callee_file_path,
               r.call_line AS call_line,
               r.call_type AS call_type,
               r.arguments_count AS arguments_count,
               count(r) AS call_count
        ORDER BY call_count DESC
        LIMIT $limit
        """

        return (cypher, parameters)

    # ========================================
    # BI-TEMPORAL QUERIES (Issue #27)
    # ========================================

    def _build_temporal_filter_clause(
        self,
        time_type: str = "current",
        as_of_valid: Optional[str] = None,
        as_of_transaction: Optional[str] = None,
        node_alias: str = "m",
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Build temporal WHERE clause for bi-temporal queries.

        Generates parameterized temporal filter conditions based on the requested
        time type. Supports current state, valid time, transaction time, or both.

        Args:
            time_type: Type of temporal query
                - "current": Filter by is_current = true (default)
                - "valid": Filter by valid time dimension
                - "transaction": Filter by transaction time dimension
                - "both": Filter by both valid and transaction time
            as_of_valid: Point in valid time (ISO 8601 timestamp)
                Required when time_type is "valid" or "both"
            as_of_transaction: Point in transaction time (ISO 8601 timestamp)
                Required when time_type is "transaction" or "both"
            node_alias: Cypher node alias (default "m" for Memory)

        Returns:
            Tuple of (clause_string, parameters_dict)
            - clause_string: WHERE clause fragment starting with "AND"
            - parameters_dict: Parameters for the clause

        Example:
            ```python
            # Current state filter
            clause, params = builder._build_temporal_filter_clause(
                time_type="current"
            )
            # Returns: ("AND m.is_current = true", {})

            # Valid time filter
            clause, params = builder._build_temporal_filter_clause(
                time_type="valid",
                as_of_valid="2025-11-15T00:00:00Z"
            )
            # Returns: (
            #     "AND m.valid_from <= $as_of_valid AND (m.valid_to IS NULL OR m.valid_to > $as_of_valid)",
            #     {"as_of_valid": "2025-11-15T00:00:00Z"}
            # )
            ```

        Notes:
            - NULL handling: valid_to/transaction_to NULL means "still valid/current"
            - Time comparisons use <= for start and > for end (half-open interval)
            - For "both", valid and transaction filters are combined with AND
        """
        params: Dict[str, Any] = {}
        alias = node_alias

        if time_type == "current":
            # Optimized current state query using indexed is_current field
            return (f"AND {alias}.is_current = true", params)

        if time_type == "valid":
            if not as_of_valid:
                raise ValidationError("as_of_valid is required for time_type='valid'")
            params["as_of_valid"] = as_of_valid
            clause = (
                f"AND {alias}.valid_from <= $as_of_valid "
                f"AND ({alias}.valid_to IS NULL OR {alias}.valid_to > $as_of_valid)"
            )
            return (clause, params)

        if time_type == "transaction":
            if not as_of_transaction:
                raise ValidationError("as_of_transaction is required for time_type='transaction'")
            params["as_of_transaction"] = as_of_transaction
            clause = (
                f"AND {alias}.created_at <= $as_of_transaction "
                f"AND ({alias}.transaction_to IS NULL OR {alias}.transaction_to > $as_of_transaction)"
            )
            return (clause, params)

        if time_type == "both":
            if not as_of_valid:
                raise ValidationError("as_of_valid is required for time_type='both'")
            if not as_of_transaction:
                raise ValidationError("as_of_transaction is required for time_type='both'")
            params["as_of_valid"] = as_of_valid
            params["as_of_transaction"] = as_of_transaction
            clause = (
                f"AND {alias}.valid_from <= $as_of_valid "
                f"AND ({alias}.valid_to IS NULL OR {alias}.valid_to > $as_of_valid) "
                f"AND {alias}.created_at <= $as_of_transaction "
                f"AND ({alias}.transaction_to IS NULL OR {alias}.transaction_to > $as_of_transaction)"
            )
            return (clause, params)

        raise ValidationError(
            f"Invalid time_type: '{time_type}'. "
            "Must be one of: 'current', 'valid', 'transaction', 'both'"
        )

    def build_point_in_time_query(
        self,
        workspace_id: str,
        file_path: str,
        as_of: str,
        time_type: Literal["valid", "transaction", "both"] = "valid",
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Build Cypher query for point-in-time lookup.

        Retrieves the memory version that was valid/known at the specified
        point in time. Returns the single most recent version matching
        the temporal criteria.

        Args:
            workspace_id: Workspace ID for data isolation
            file_path: File path to query
            as_of: ISO 8601 timestamp for the point in time
            time_type: Which time dimension to query
                - "valid": What was true in reality at as_of
                - "transaction": What we knew in the database at as_of
                - "both": Bi-temporal query (requires as_of for both dimensions)

        Returns:
            Tuple of (cypher_string, parameters_dict)

        Raises:
            ValidationError: If workspace_id is empty
            ValidationError: If file_path is empty
            ValidationError: If as_of is empty
            ValidationError: If time_type is invalid

        Example:
            ```python
            # What was the state of main.py on November 15th?
            cypher, params = builder.build_point_in_time_query(
                workspace_id="default",
                file_path="/project/src/main.py",
                as_of="2025-11-15T00:00:00Z",
                time_type="valid"
            )
            ```

        Query Pattern:
            ```cypher
            MATCH (m:Memory)
            WHERE m.workspace_id = $workspace_id
              AND m.file_path = $file_path
              AND m.valid_from <= $as_of_valid
              AND (m.valid_to IS NULL OR m.valid_to > $as_of_valid)
            ORDER BY m.valid_from DESC
            LIMIT 1
            ```
        """
        # STEP 1: Validate inputs
        if not workspace_id or not workspace_id.strip():
            raise ValidationError("workspace_id cannot be empty")

        if not file_path or not file_path.strip():
            raise ValidationError("file_path cannot be empty")

        if not as_of or not as_of.strip():
            raise ValidationError("as_of cannot be empty")

        if time_type not in ("valid", "transaction", "both"):
            raise ValidationError(
                f"Invalid time_type: '{time_type}'. "
                "Must be one of: 'valid', 'transaction', 'both'"
            )

        # STEP 2: Build parameters
        parameters: Dict[str, Any] = {
            "workspace_id": workspace_id,
            "file_path": file_path,
        }

        # STEP 3: Build temporal filter
        if time_type == "both":
            temporal_clause, temporal_params = self._build_temporal_filter_clause(
                time_type="both",
                as_of_valid=as_of,
                as_of_transaction=as_of,
            )
        elif time_type == "valid":
            temporal_clause, temporal_params = self._build_temporal_filter_clause(
                time_type="valid",
                as_of_valid=as_of,
            )
        else:  # transaction
            temporal_clause, temporal_params = self._build_temporal_filter_clause(
                time_type="transaction",
                as_of_transaction=as_of,
            )
        parameters.update(temporal_params)

        # STEP 4: Determine ORDER BY field based on time_type
        order_field = "m.valid_from" if time_type in ("valid", "both") else "m.created_at"

        # STEP 5: Build Cypher query
        cypher = f"""
        MATCH (m:Memory)
        WHERE m.workspace_id = $workspace_id
        AND m.file_path = $file_path
        {temporal_clause}
        RETURN m.id AS memory_id,
               m.text AS text,
               m.file_path AS file_path,
               m.qualified_name AS qualified_name,
               m.version AS version,
               m.valid_from AS valid_from,
               m.valid_to AS valid_to,
               m.created_at AS created_at,
               m.transaction_to AS transaction_to,
               m.is_current AS is_current,
               m.source AS source,
               m.tags AS tags,
               m.workspace_id AS workspace_id
        ORDER BY {order_field} DESC
        LIMIT 1
        """

        return (cypher, parameters)

    def build_history_query(
        self,
        workspace_id: str,
        file_path: Optional[str] = None,
        entity_id: Optional[str] = None,
        limit: int = 50,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Build Cypher query for version history.

        Retrieves all versions of a memory, ordered by valid_from descending.
        At least one of file_path or entity_id must be provided.

        Args:
            workspace_id: Workspace ID for data isolation
            file_path: File path to get history for (optional if entity_id provided)
            entity_id: Memory ID to get history for (optional if file_path provided)
            limit: Maximum number of versions to return (1-1000, default 50)

        Returns:
            Tuple of (cypher_string, parameters_dict)

        Raises:
            ValidationError: If workspace_id is empty
            ValidationError: If both file_path and entity_id are None
            ValidationError: If limit is out of range
            ValidationError: If entity_id is not a valid UUID

        Example:
            ```python
            # Get history by file path
            cypher, params = builder.build_history_query(
                workspace_id="default",
                file_path="/project/src/main.py",
                limit=100
            )

            # Get history by entity ID
            cypher, params = builder.build_history_query(
                workspace_id="default",
                entity_id="550e8400-e29b-41d4-a716-446655440000"
            )
            ```

        Query Returns:
            - memory_id: UUID of the version
            - version: Version number (1, 2, 3...)
            - valid_from: When version became valid
            - valid_to: When version was superseded (NULL if current)
            - created_at: When recorded in database
            - transaction_to: When superseded in database (NULL if current)
            - is_current: Boolean flag for current version
        """
        # STEP 1: Validate inputs
        if not workspace_id or not workspace_id.strip():
            raise ValidationError("workspace_id cannot be empty")

        if file_path is None and entity_id is None:
            raise ValidationError("Either file_path or entity_id must be provided")

        if not isinstance(limit, int) or limit < 1 or limit > 1000:
            raise ValidationError(f"limit must be int in range [1, 1000], got {limit}")

        if entity_id is not None:
            self._validate_uuid(entity_id)

        # STEP 2: Build parameters
        parameters: Dict[str, Any] = {
            "workspace_id": workspace_id,
            "limit": limit,
        }

        # STEP 3: Build filter condition
        if file_path is not None:
            parameters["file_path"] = file_path
            filter_condition = "AND m.file_path = $file_path"
        else:
            parameters["entity_id"] = entity_id
            filter_condition = "AND m.id = $entity_id"

        # STEP 4: Build Cypher query
        cypher = f"""
        MATCH (m:Memory)
        WHERE m.workspace_id = $workspace_id
        {filter_condition}
        RETURN m.id AS memory_id,
               m.version AS version,
               m.valid_from AS valid_from,
               m.valid_to AS valid_to,
               m.created_at AS created_at,
               m.transaction_to AS transaction_to,
               m.is_current AS is_current,
               m.file_path AS file_path,
               m.qualified_name AS qualified_name,
               m.text AS text,
               m.source AS source
        ORDER BY m.valid_from DESC
        LIMIT $limit
        """

        return (cypher, parameters)

    def build_changes_query(
        self,
        workspace_id: str,
        since: str,
        until: Optional[str] = None,
        change_type: Optional[str] = None,
        path_pattern: Optional[str] = None,
        limit: int = 100,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Build Cypher query for changes in a time range.

        Retrieves memories created/modified within the specified time range,
        with optional filtering by change type and path pattern.

        Args:
            workspace_id: Workspace ID for data isolation
            since: ISO 8601 timestamp for range start (inclusive)
            until: ISO 8601 timestamp for range end (inclusive, optional)
            change_type: Filter by change type (optional)
                - "created": Only version=1 records (new files)
                - "modified": Only version>1 records (updates)
                - "deleted": Only records where transaction_to is set
            path_pattern: Cypher STARTS WITH pattern for file_path (optional)
            limit: Maximum number of results (1-1000, default 100)

        Returns:
            Tuple of (cypher_string, parameters_dict)

        Raises:
            ValidationError: If workspace_id is empty
            ValidationError: If since is empty
            ValidationError: If change_type is invalid
            ValidationError: If limit is out of range

        Example:
            ```python
            # Get all changes in the last day
            cypher, params = builder.build_changes_query(
                workspace_id="default",
                since="2025-12-04T00:00:00Z",
                until="2025-12-05T00:00:00Z",
                limit=50
            )

            # Get only new files in src/
            cypher, params = builder.build_changes_query(
                workspace_id="default",
                since="2025-12-01T00:00:00Z",
                change_type="created",
                path_pattern="/project/src/"
            )
            ```

        Query Returns:
            - memory_id: UUID of the memory
            - file_path: File path
            - version: Version number
            - created_at: When recorded in database
            - change_type: "created", "modified", or "deleted"
            - qualified_name: Fully qualified name (if available)
        """
        # STEP 1: Validate inputs
        if not workspace_id or not workspace_id.strip():
            raise ValidationError("workspace_id cannot be empty")

        if not since or not since.strip():
            raise ValidationError("since cannot be empty")

        if change_type is not None and change_type not in ("created", "modified", "deleted"):
            raise ValidationError(
                f"Invalid change_type: '{change_type}'. "
                "Must be one of: 'created', 'modified', 'deleted'"
            )

        if not isinstance(limit, int) or limit < 1 or limit > 1000:
            raise ValidationError(f"limit must be int in range [1, 1000], got {limit}")

        # STEP 2: Build parameters
        parameters: Dict[str, Any] = {
            "workspace_id": workspace_id,
            "since": since,
            "limit": limit,
        }

        # STEP 3: Build optional filters
        optional_filters = []

        if until is not None:
            parameters["until"] = until
            optional_filters.append("AND m.created_at <= $until")

        if path_pattern is not None:
            parameters["path_pattern"] = path_pattern
            optional_filters.append("AND m.file_path STARTS WITH $path_pattern")

        # STEP 4: Build change_type filter
        change_type_filter = ""
        if change_type == "created":
            change_type_filter = "AND m.version = 1"
        elif change_type == "modified":
            change_type_filter = "AND m.version > 1"
        elif change_type == "deleted":
            change_type_filter = "AND m.transaction_to IS NOT NULL"

        optional_clause = " ".join(optional_filters)

        # STEP 5: Build Cypher query
        cypher = f"""
        MATCH (m:Memory)
        WHERE m.workspace_id = $workspace_id
        AND m.created_at >= $since
        {optional_clause}
        {change_type_filter}
        RETURN m.id AS memory_id,
               m.file_path AS file_path,
               m.version AS version,
               m.created_at AS created_at,
               m.qualified_name AS qualified_name,
               m.valid_from AS valid_from,
               m.valid_to AS valid_to,
               m.transaction_to AS transaction_to,
               CASE
                 WHEN m.version = 1 THEN 'created'
                 WHEN m.transaction_to IS NOT NULL THEN 'deleted'
                 ELSE 'modified'
               END AS change_type
        ORDER BY m.created_at DESC
        LIMIT $limit
        """

        return (cypher, parameters)

    def build_close_version_query(
        self,
        memory_id: str,
        valid_to: str,
        transaction_to: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Build Cypher query to close a version (mark as superseded).

        Updates a memory version to mark it as no longer current. This is
        called when creating a new version of the same entity.

        Args:
            memory_id: UUID of the memory version to close
            valid_to: ISO 8601 timestamp when version ceased to be valid
            transaction_to: ISO 8601 timestamp when version was superseded

        Returns:
            Tuple of (cypher_string, parameters_dict)

        Raises:
            ValidationError: If memory_id is not valid UUID
            ValidationError: If valid_to is empty
            ValidationError: If transaction_to is empty

        Example:
            ```python
            # Close previous version when creating new one
            cypher, params = builder.build_close_version_query(
                memory_id="550e8400-e29b-41d4-a716-446655440000",
                valid_to="2025-12-05T10:30:00Z",
                transaction_to="2025-12-05T10:30:00Z"
            )
            ```

        Query Effects:
            - Sets valid_to to mark when content became invalid
            - Sets transaction_to to mark when superseded in database
            - Sets is_current = false for optimized queries

        Notes:
            - This should be called in the same transaction as creating
              the new version to maintain consistency
            - The valid_to/transaction_to timestamps should typically
              match the valid_from/created_at of the new version
        """
        # STEP 1: Validate inputs
        self._validate_uuid(memory_id)

        if not valid_to or not valid_to.strip():
            raise ValidationError("valid_to cannot be empty")

        if not transaction_to or not transaction_to.strip():
            raise ValidationError("transaction_to cannot be empty")

        # STEP 2: Build parameters
        parameters: Dict[str, Any] = {
            "memory_id": memory_id,
            "valid_to": valid_to,
            "transaction_to": transaction_to,
        }

        # STEP 3: Build Cypher query
        cypher = """
        MATCH (m:Memory {id: $memory_id})
        SET m.valid_to = $valid_to,
            m.transaction_to = $transaction_to,
            m.is_current = false
        RETURN m.id AS memory_id,
               m.is_current AS is_current,
               m.valid_to AS valid_to,
               m.transaction_to AS transaction_to
        """

        return (cypher, parameters)
