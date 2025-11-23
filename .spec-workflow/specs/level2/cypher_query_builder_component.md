# CypherQueryBuilder - Component Specification

**Level:** 2 (Component)
**Module:** zapomni_db
**Author:** Goncharenko Anton aka alienxs2
**Status:** Draft
**Version:** 1.0
**Date:** 2025-11-23

## Overview

### Purpose

The `CypherQueryBuilder` component provides type-safe, parameterized Cypher query generation for FalkorDB operations. It acts as a secure abstraction layer between business logic and raw Cypher queries, preventing SQL/Cypher injection vulnerabilities and ensuring consistent query patterns across the Zapomni memory system.

### Responsibilities

1. **Type-Safe Query Construction** - Build Cypher queries using parameterized templates (no string concatenation)
2. **Injection Prevention** - All user inputs passed as parameters, never embedded in query strings
3. **Query Template Management** - Provide reusable templates for common operations (add memory, vector search, graph traversal)
4. **Parameter Validation** - Validate query parameters before query generation
5. **Query Optimization** - Generate optimized Cypher patterns (indexed lookups, efficient MATCH patterns)

### Position in Module

`CypherQueryBuilder` is a core utility component within the `zapomni_db` module. It is used by `FalkorDBClient` to generate all Cypher queries before execution.

```
┌─────────────────────────────────────┐
│     FalkorDBClient                  │
│  (uses CypherQueryBuilder)          │
└───────────┬─────────────────────────┘
            │
            ↓ calls build_* methods
┌─────────────────────────────────────┐
│   CypherQueryBuilder                │  ← THIS COMPONENT
│   - build_add_memory_query()        │
│   - build_vector_search_query()     │
│   - build_graph_traversal_query()   │
│   - build_stats_query()             │
└───────────┬─────────────────────────┘
            │
            ↓ returns (cypher, params)
┌─────────────────────────────────────┐
│    FalkorDB Graph.query()           │
│    (executes parameterized query)   │
└─────────────────────────────────────┘
```

## Class Definition

### Class Diagram

```
┌──────────────────────────────────────────────┐
│         CypherQueryBuilder                   │
├──────────────────────────────────────────────┤
│ - VECTOR_INDEX_NAME: str                     │
│ - VECTOR_DIMENSION: int                      │
│ - DEFAULT_SIMILARITY_FUNCTION: str           │
├──────────────────────────────────────────────┤
│ + build_add_memory_query(memory)             │
│ + build_vector_search_query(...)             │
│ + build_graph_traversal_query(...)           │
│ + build_stats_query()                        │
│ + build_delete_memory_query(memory_id)       │
│ + build_add_entity_query(entity)             │
│ + build_add_relationship_query(...)          │
│ - _validate_uuid(uuid_str)                   │
│ - _validate_embedding(embedding)             │
│ - _build_filter_clause(filters)              │
└──────────────────────────────────────────────┘
```

### Full Class Signature

```python
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import uuid
import re

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

    Example:
        ```python
        builder = CypherQueryBuilder()

        # Generate add memory query
        cypher, params = builder.build_add_memory_query(memory)
        result = graph.query(cypher, params)

        # Generate vector search query
        cypher, params = builder.build_vector_search_query(
            embedding=[0.1] * 768,
            limit=10,
            filters={"tags": ["python"]}
        )
        results = graph.query(cypher, params)
        ```

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
            # CREATE (from)-[r:MENTIONS $properties]->(to)
            # RETURN r
            ```
        """

    # Private helper methods

    def _validate_uuid(self, uuid_str: str) -> None:
        """
        Validate that string is a valid UUID4.

        Args:
            uuid_str: String to validate

        Raises:
            ValidationError: If uuid_str is not valid UUID4 format
        """

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
                "WHERE $tag1 IN m.tags OR $tag2 IN m.tags AND m.source = $source",
                {"tag1": "python", "tag2": "coding", "source": "user"}
            )

        Notes:
            - Empty filters return ("", {})
            - All values parameterized (no injection risk)
            - Tag filters use OR logic (match any tag)
            - Other filters use AND logic
        """
```

## Dependencies

### Component Dependencies

- **FalkorDBClient** (sibling component in zapomni_db)
  - For: Query execution after building
  - Usage: FalkorDBClient calls CypherQueryBuilder methods

- **Data Models** (from zapomni_db.models)
  - For: Type hints and validation
  - Usage: Memory, Entity, Chunk, SearchResult types

### External Libraries

- **uuid** (Python stdlib)
  - For: UUID validation and generation
  - Usage: `uuid.UUID()` to validate UUID strings

- **re** (Python stdlib)
  - For: Regex pattern matching
  - Usage: Validate relationship type format

- **typing** (Python stdlib)
  - For: Type hints (List, Dict, Tuple, Optional)
  - Usage: Function signatures

### Dependency Injection

CypherQueryBuilder is stateless and requires no dependencies injected.
Instantiated directly by FalkorDBClient:

```python
class FalkorDBClient:
    def __init__(self, ...):
        self.query_builder = CypherQueryBuilder()
```

## State Management

### Attributes

CypherQueryBuilder is **stateless** - no instance attributes.

All configuration is class-level constants:
- `VECTOR_INDEX_NAME`: str (constant)
- `VECTOR_DIMENSION`: int (constant)
- `DEFAULT_SIMILARITY_FUNCTION`: str (constant)

### State Transitions

Not applicable - no state to transition.

Each method call is independent and produces a deterministic output
given the same inputs (pure functions).

### Thread Safety

**Thread-safe**: Yes

- No mutable state (stateless)
- No shared resources
- Pure functions (deterministic, no side effects)
- Can be called concurrently from multiple threads

Safe usage pattern:
```python
# Single shared instance across threads
query_builder = CypherQueryBuilder()

# Thread 1
cypher1, params1 = query_builder.build_vector_search_query(...)

# Thread 2 (concurrent)
cypher2, params2 = query_builder.build_add_memory_query(...)
```

## Public Methods (Detailed)

### Method 1: `build_add_memory_query`

**Signature:**
```python
def build_add_memory_query(
    self,
    memory: Memory
) -> Tuple[str, Dict[str, Any]]
```

**Purpose:** Generate parameterized Cypher query to insert a Memory node with its Chunks and embeddings in a single transaction-safe query.

**Parameters:**

- `memory`: Memory
  - Description: Memory object containing full text, chunks, embeddings, and metadata
  - Constraints:
    - `memory.chunks` must not be empty
    - `len(memory.chunks) == len(memory.embeddings)`
    - Each embedding must be 768-dimensional
    - `memory.text` must not be empty
  - Example:
    ```python
    Memory(
        text="Python is a programming language",
        chunks=[
            Chunk(text="Python is a programming", index=0),
            Chunk(text="programming language", index=1)
        ],
        embeddings=[
            [0.1, 0.2, ...],  # 768 dimensions
            [0.3, 0.4, ...]   # 768 dimensions
        ],
        metadata={
            "source": "user_input",
            "tags": ["python", "programming"],
            "timestamp": "2025-11-23T12:00:00Z"
        }
    )
    ```

**Returns:**
- Type: `Tuple[str, Dict[str, Any]]`
- First element (str): Parameterized Cypher query string
- Second element (dict): Parameter dictionary for safe execution
- Example:
  ```python
  (
      """
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
          embedding: chunk_data.embedding
      })
      CREATE (m)-[:HAS_CHUNK {index: chunk_data.index}]->(c)
      RETURN m.id
      """,
      {
          "memory_id": "550e8400-e29b-41d4-a716-446655440000",
          "text": "Python is a programming language",
          "source": "user_input",
          "tags": ["python", "programming"],
          "created_at": "2025-11-23T12:00:00Z",
          "chunks": [
              {
                  "id": "...",
                  "text": "Python is a programming",
                  "index": 0,
                  "embedding": [0.1, 0.2, ...]
              },
              ...
          ]
      }
  )
  ```

**Raises:**
- `ValidationError`: If `memory.chunks` is empty
- `ValidationError`: If chunks count != embeddings count
- `ValidationError`: If any embedding has wrong dimension (!= 768)
- `ValidationError`: If `memory.text` is empty

**Preconditions:**
- Memory object is valid and passes Pydantic validation
- All embeddings are 768-dimensional

**Postconditions:**
- Returns valid parameterized Cypher query
- No user input embedded in query string (all parameterized)
- Query is idempotent (can be executed multiple times safely)

**Algorithm Outline:**
```
1. Generate unique memory_id (UUID4)
2. Validate memory structure:
   - Check chunks not empty
   - Check len(chunks) == len(embeddings)
   - Validate each embedding dimension
3. Extract metadata (source, tags, timestamp)
4. Build chunks parameter list:
   - For each chunk:
     - Generate chunk_id (UUID4)
     - Extract text, index
     - Pair with corresponding embedding
5. Build Cypher query template:
   - CREATE Memory node with $parameters
   - UNWIND $chunks (batch insert)
   - CREATE Chunk nodes with embeddings
   - CREATE HAS_CHUNK relationships
6. Return (cypher_string, params_dict)
```

**Edge Cases:**

1. Single chunk memory:
   - Handle normally (UNWIND works with single-element list)

2. Large number of chunks (> 100):
   - Consider batch size limit (FalkorDB max query size)
   - Log warning if > 100 chunks

3. Empty metadata:
   - Store empty dict, not null

4. Non-ASCII text:
   - UTF-8 encoded properly (handled by Pydantic)

**Related Methods:**
- Calls: `_validate_embedding()` for each embedding
- Called by: `FalkorDBClient.add_memory()`

---

### Method 2: `build_vector_search_query`

**Signature:**
```python
def build_vector_search_query(
    self,
    embedding: List[float],
    limit: int = 10,
    filters: Optional[Dict[str, Any]] = None,
    min_similarity: float = 0.5
) -> Tuple[str, Dict[str, Any]]
```

**Purpose:** Generate parameterized Cypher query for vector similarity search using FalkorDB's HNSW index with optional metadata filtering.

**Parameters:**

- `embedding`: List[float]
  - Description: Query embedding vector (768-dimensional)
  - Constraints: Must be exactly 768 dimensions
  - Example: `[0.1, 0.2, 0.3, ..., 0.768]` (768 floats)

- `limit`: int (default: 10)
  - Description: Maximum number of results to return
  - Constraints: Must be in range [1, 1000]
  - Example: `10` (return top 10 most similar)

- `filters`: Optional[Dict[str, Any]] (default: None)
  - Description: Metadata filters to apply
  - Structure:
    ```python
    {
        "tags": ["python", "coding"],  # Match ANY of these tags
        "source": "user_input",        # Exact match
        "date_from": datetime(...),    # Created after
        "date_to": datetime(...)        # Created before
    }
    ```
  - Constraints: All values must be JSON-serializable
  - Example: `{"tags": ["python"], "source": "user"}`

- `min_similarity`: float (default: 0.5)
  - Description: Minimum cosine similarity threshold
  - Constraints: Must be in range [0.0, 1.0]
  - Example: `0.7` (only return results with >= 70% similarity)

**Returns:**
- Type: `Tuple[str, Dict[str, Any]]`
- Example:
  ```python
  (
      """
      CALL db.idx.vector.queryNodes(
          'chunk_embedding_idx',
          $limit,
          $query_embedding
      ) YIELD node AS c, score
      MATCH (m:Memory)-[:HAS_CHUNK]->(c)
      WHERE score >= $min_similarity
      AND ($tag0 IN m.tags OR $tag1 IN m.tags)
      AND m.source = $source
      RETURN m.id AS memory_id,
             c.text AS text,
             score AS similarity_score,
             m.tags AS tags,
             m.source AS source,
             m.created_at AS timestamp
      ORDER BY score DESC
      """,
      {
          "query_embedding": [0.1, 0.2, ...],
          "limit": 10,
          "min_similarity": 0.7,
          "tag0": "python",
          "tag1": "coding",
          "source": "user"
      }
  )
  ```

**Raises:**
- `ValidationError`: If embedding dimension != 768
- `ValidationError`: If limit < 1 or > 1000
- `ValidationError`: If min_similarity not in [0.0, 1.0]
- `ValidationError`: If filters contain invalid keys

**Preconditions:**
- Vector index exists (created during FalkorDBClient initialization)
- Embedding is normalized (for cosine similarity)

**Postconditions:**
- Returns valid parameterized query
- Results will be sorted by similarity (descending)
- All filters applied via parameters (injection-safe)

**Algorithm Outline:**
```
1. Validate embedding dimension (must be 768)
2. Validate limit (1-1000 range)
3. Validate min_similarity (0.0-1.0 range)
4. Build base vector search query:
   - CALL db.idx.vector.queryNodes()
   - WITH node, score
5. Build filter clause:
   - If filters provided:
     - Call _build_filter_clause(filters)
     - Add WHERE conditions
6. Build RETURN clause:
   - Select memory_id, text, score, tags, source, timestamp
   - ORDER BY score DESC
7. Merge parameters:
   - query_embedding, limit, min_similarity
   - Plus filter parameters
8. Return (cypher_string, params_dict)
```

**Edge Cases:**

1. No results above min_similarity:
   - Return empty list (handled by caller)

2. Limit larger than available chunks:
   - Return all available chunks (Cypher handles gracefully)

3. Multiple tags filter (OR logic):
   - Generate multiple tag parameters (tag0, tag1, ...)
   - Build WHERE clause: `$tag0 IN m.tags OR $tag1 IN m.tags`

4. No filters (filters=None):
   - Skip WHERE clause entirely (search all)

**Related Methods:**
- Calls: `_validate_embedding()`, `_build_filter_clause()`
- Called by: `FalkorDBClient.vector_search()`

---

### Method 3: `build_graph_traversal_query`

**Signature:**
```python
def build_graph_traversal_query(
    self,
    entity_id: str,
    depth: int = 1,
    limit: int = 20
) -> Tuple[str, Dict[str, Any]]
```

**Purpose:** Generate parameterized Cypher query for graph traversal to find entities related to a starting entity at specified depth.

**Parameters:**

- `entity_id`: str
  - Description: Starting entity UUID
  - Constraints: Must be valid UUID4 format
  - Example: `"550e8400-e29b-41d4-a716-446655440000"`

- `depth`: int (default: 1)
  - Description: Traversal depth (number of relationship hops)
  - Constraints: Must be in range [1, 5]
  - Example: `2` (neighbors and neighbors-of-neighbors)

- `limit`: int (default: 20)
  - Description: Maximum number of related entities to return
  - Constraints: Must be in range [1, 100]
  - Example: `20`

**Returns:**
- Type: `Tuple[str, Dict[str, Any]]`
- Example:
  ```python
  (
      """
      MATCH (start:Entity {id: $entity_id})
      MATCH (start)-[rels*1..2]-(related:Entity)
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
      """,
      {
          "entity_id": "550e8400-e29b-41d4-a716-446655440000",
          "limit": 20
      }
  )
  ```

**Raises:**
- `ValidationError`: If entity_id is not valid UUID
- `ValidationError`: If depth < 1 or > 5
- `ValidationError`: If limit < 1 or > 100

**Preconditions:**
- Entity with entity_id exists in graph (caller should check)
- Graph has relationships to traverse

**Postconditions:**
- Returns related entities sorted by relationship strength
- Excludes starting entity from results
- Results are DISTINCT (no duplicates)

**Algorithm Outline:**
```
1. Validate entity_id (UUID format)
2. Validate depth (1-5 range)
3. Validate limit (1-100 range)
4. Build variable-length pattern:
   - If depth == 1: -[r]-
   - If depth > 1: -[rels*1..{depth}]-
5. Build MATCH clause:
   - MATCH (start:Entity {id: $entity_id})
   - MATCH (start)-[pattern]-(related:Entity)
6. Build WHERE clause:
   - Exclude starting entity (related.id <> $entity_id)
7. Calculate path strength:
   - Use reduce() to multiply relationship strengths
8. Build RETURN clause:
   - Select entity properties, path_strength
   - ORDER BY path_strength DESC
   - LIMIT $limit
9. Return (cypher_string, params_dict)
```

**Edge Cases:**

1. No related entities:
   - Return empty result set

2. Depth = 1 (direct neighbors only):
   - Optimize pattern: `-[r]-` instead of `-[rels*1..1]-`

3. Circular relationships:
   - DISTINCT prevents duplicate entities
   - WHERE clause excludes starting entity

4. Unweighted relationships (no strength property):
   - Use coalesce(r.strength, 1.0) to default to 1.0

**Related Methods:**
- Calls: `_validate_uuid()`
- Called by: `FalkorDBClient.get_related_entities()`

---

### Method 4: `build_stats_query`

**Signature:**
```python
def build_stats_query(self) -> Tuple[str, Dict[str, Any]]
```

**Purpose:** Generate Cypher query to retrieve database statistics (node counts by type, relationship counts).

**Parameters:** None

**Returns:**
- Type: `Tuple[str, Dict[str, Any]]`
- Example:
  ```python
  (
      """
      MATCH (m:Memory)
      WITH count(m) AS total_memories
      MATCH (c:Chunk)
      WITH total_memories, count(c) AS total_chunks
      MATCH (e:Entity)
      WITH total_memories, total_chunks, count(e) AS total_entities
      MATCH ()-[r]->()
      WITH total_memories, total_chunks, total_entities, count(r) AS total_relationships
      RETURN total_memories, total_chunks, total_entities, total_relationships
      """,
      {}  # No parameters needed
  )
  ```

**Raises:** None

**Preconditions:** None

**Postconditions:**
- Returns counts for all node types
- Returns total relationship count

**Algorithm Outline:**
```
1. Build query with multiple MATCH clauses:
   - Count Memory nodes
   - Count Chunk nodes
   - Count Entity nodes
   - Count all relationships
2. Use WITH to chain counts
3. RETURN all counts in single row
4. Return (cypher_string, empty_params_dict)
```

**Edge Cases:**

1. Empty graph:
   - All counts return 0

2. Missing node types:
   - Count returns 0 for missing types

**Related Methods:**
- Called by: `FalkorDBClient.get_stats()`

---

### Method 5: `_validate_uuid`

**Signature:**
```python
def _validate_uuid(self, uuid_str: str) -> None
```

**Purpose:** Validate that a string is a valid UUID4 format to prevent injection.

**Parameters:**
- `uuid_str`: str - String to validate

**Raises:**
- `ValidationError`: If uuid_str is not valid UUID4 format

**Algorithm:**
```python
try:
    uuid.UUID(uuid_str, version=4)
except ValueError:
    raise ValidationError(f"Invalid UUID: {uuid_str}")
```

---

### Method 6: `_validate_embedding`

**Signature:**
```python
def _validate_embedding(self, embedding: List[float]) -> None
```

**Purpose:** Validate embedding vector dimensions and format.

**Parameters:**
- `embedding`: List[float] - Embedding vector to validate

**Raises:**
- `ValidationError`: If embedding is not 768-dimensional
- `ValidationError`: If embedding contains non-numeric values
- `ValidationError`: If embedding is empty

**Algorithm:**
```python
if not embedding:
    raise ValidationError("Embedding cannot be empty")
if len(embedding) != self.VECTOR_DIMENSION:
    raise ValidationError(
        f"Embedding dimension mismatch: expected {self.VECTOR_DIMENSION}, "
        f"got {len(embedding)}"
    )
if not all(isinstance(x, (int, float)) for x in embedding):
    raise ValidationError("Embedding must contain only numeric values")
```

---

### Method 7: `_build_filter_clause`

**Signature:**
```python
def _build_filter_clause(
    self,
    filters: Optional[Dict[str, Any]]
) -> Tuple[str, Dict[str, Any]]
```

**Purpose:** Build parameterized WHERE clause from metadata filters.

**Parameters:**
- `filters`: Optional[Dict[str, Any]] - Filter criteria

**Returns:**
- Tuple[str, Dict[str, Any]] - (where_clause, filter_parameters)

**Algorithm:**
```
1. If filters is None or empty:
   - Return ("", {})
2. Initialize clause_parts = []
3. Initialize params = {}
4. For each filter key:
   - If key == "tags":
     - For each tag:
       - Add "$tagN IN m.tags" to clause_parts
       - Add {"tagN": tag_value} to params
     - Join tag clauses with OR
   - If key == "source":
     - Add "m.source = $source" to clause_parts
     - Add {"source": value} to params
   - If key == "date_from":
     - Add "m.created_at >= $date_from" to clause_parts
   - If key == "date_to":
     - Add "m.created_at <= $date_to" to clause_parts
5. Join clause_parts with AND
6. Return ("WHERE " + joined_clauses, params)
```

## Error Handling

### Exceptions Defined

```python
class ValidationError(Exception):
    """Raised when query parameter validation fails."""
    pass
```

### Error Recovery

**No automatic retry** - validation errors are permanent (caller must fix input).

**Error propagation:**
- All ValidationError exceptions bubble up to FalkorDBClient
- FalkorDBClient catches and re-raises with additional context

**Error messages:**
- Include specific validation failure reason
- Include parameter name and expected format
- Do NOT include sensitive data

## Usage Examples

### Basic Usage

```python
from zapomni_db.cypher_query_builder import CypherQueryBuilder
from zapomni_db.models import Memory, Chunk

# Initialize builder (stateless, can reuse)
builder = CypherQueryBuilder()

# Example 1: Add memory query
memory = Memory(
    text="Python is a high-level programming language",
    chunks=[
        Chunk(text="Python is a high-level", index=0),
        Chunk(text="high-level programming language", index=1)
    ],
    embeddings=[
        [0.1] * 768,
        [0.2] * 768
    ],
    metadata={"source": "user", "tags": ["python"]}
)

cypher, params = builder.build_add_memory_query(memory)
print(f"Query: {cypher}")
print(f"Params: {params.keys()}")
# Execute with FalkorDB:
# result = graph.query(cypher, params)

# Example 2: Vector search query
query_embedding = [0.15] * 768
cypher, params = builder.build_vector_search_query(
    embedding=query_embedding,
    limit=10,
    filters={"tags": ["python", "coding"]},
    min_similarity=0.7
)
# result = graph.query(cypher, params)

# Example 3: Graph traversal query
cypher, params = builder.build_graph_traversal_query(
    entity_id="550e8400-e29b-41d4-a716-446655440000",
    depth=2,
    limit=20
)
# result = graph.query(cypher, params)

# Example 4: Stats query
cypher, params = builder.build_stats_query()
# result = graph.query(cypher, params)
```

### Advanced Usage

```python
# Complex filtering
cypher, params = builder.build_vector_search_query(
    embedding=[0.1] * 768,
    limit=50,
    filters={
        "tags": ["python", "machine-learning", "ai"],
        "source": "academic_papers",
        "date_from": datetime(2024, 1, 1),
        "date_to": datetime(2025, 11, 23)
    },
    min_similarity=0.8
)

# Deep graph traversal
cypher, params = builder.build_graph_traversal_query(
    entity_id="...",
    depth=3,  # 3 hops
    limit=50
)

# Batch operations (use same builder instance)
builder = CypherQueryBuilder()
for memory in memory_batch:
    cypher, params = builder.build_add_memory_query(memory)
    graph.query(cypher, params)  # Execute each
```

## Testing Approach

### Unit Tests Required

1. **test_build_add_memory_query_success()**
   - Valid memory with single chunk
   - Verify query structure
   - Verify parameter names

2. **test_build_add_memory_query_multiple_chunks()**
   - Valid memory with 5 chunks
   - Verify UNWIND logic
   - Verify chunk ordering

3. **test_build_add_memory_query_invalid_embedding_dimension()**
   - Memory with 512-dim embeddings (wrong!)
   - Expect ValidationError

4. **test_build_add_memory_query_mismatched_chunks_embeddings()**
   - 3 chunks, 2 embeddings
   - Expect ValidationError

5. **test_build_vector_search_query_success()**
   - Valid embedding, no filters
   - Verify CALL db.idx.vector.queryNodes

6. **test_build_vector_search_query_with_filters()**
   - With tags, source, date filters
   - Verify WHERE clause parameterization

7. **test_build_vector_search_query_invalid_limit()**
   - limit = 0, limit = 1001
   - Expect ValidationError

8. **test_build_graph_traversal_query_depth_1()**
   - Direct neighbors only
   - Verify pattern: -[r]-

9. **test_build_graph_traversal_query_depth_3()**
   - 3-hop traversal
   - Verify pattern: -[rels*1..3]-

10. **test_build_graph_traversal_query_invalid_uuid()**
    - entity_id = "not-a-uuid"
    - Expect ValidationError

11. **test_build_stats_query()**
    - No parameters
    - Verify query structure

12. **test_validate_uuid_valid()**
    - Valid UUID4 string
    - No exception

13. **test_validate_uuid_invalid()**
    - Invalid UUID string
    - Expect ValidationError

14. **test_validate_embedding_valid()**
    - 768-dimensional embedding
    - No exception

15. **test_validate_embedding_wrong_dimension()**
    - 512-dimensional embedding
    - Expect ValidationError

16. **test_build_filter_clause_empty()**
    - filters = None
    - Return ("", {})

17. **test_build_filter_clause_tags_only()**
    - filters = {"tags": ["python"]}
    - Verify OR logic

18. **test_build_filter_clause_multiple_filters()**
    - tags + source + dates
    - Verify AND logic

### Mocking Strategy

- Mock Memory, Chunk, Entity objects (use Pydantic validation)
- No external dependencies to mock (stateless class)
- Mock FalkorDB graph.query() for integration tests (Level 3)

### Integration Tests

**Deferred to FalkorDBClient integration tests:**
- Test actual query execution with FalkorDB
- Verify query results match expectations
- Test transaction behavior

## Performance Considerations

### Time Complexity

- `build_add_memory_query()`: O(n) where n = number of chunks
- `build_vector_search_query()`: O(1) (constant query generation)
- `build_graph_traversal_query()`: O(1)
- `build_stats_query()`: O(1)
- `_validate_embedding()`: O(n) where n = embedding dimension (768)
- `_build_filter_clause()`: O(k) where k = number of filters

### Space Complexity

- Memory usage: O(n) where n = number of chunks (query string size)
- No caching or persistent state
- Temporary query string allocation

### Optimization Opportunities

1. **Query template caching** (future):
   - Pre-compile common query templates
   - Reduce string formatting overhead

2. **Batch query generation** (future):
   - Single query for multiple memories
   - Reduce transaction overhead

**Trade-offs:**
- Current approach prioritizes simplicity and safety
- Performance is sufficient for expected workload (< 1ms per query generation)

## References

- **Module spec:** zapomni_db_module.md (parent spec)
- **Related components:** FalkorDBClient (sibling component)
- **External docs:**
  - FalkorDB Cypher documentation: https://docs.falkordb.com/cypher.html
  - FalkorDB Vector Index: https://docs.falkordb.com/indexing.html#vector-index
  - Cypher Query Language: https://neo4j.com/docs/cypher-manual/current/
  - Python UUID module: https://docs.python.org/3/library/uuid.html

---

**Document Status:** Draft v1.0
**Created:** 2025-11-23
**Author:** Goncharenko Anton aka alienxs2
**License:** MIT
**Ready for Verification:** Yes
