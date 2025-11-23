# SchemaManager - Component Specification

**Level:** 2 (Component)
**Module:** zapomni_db
**Author:** Goncharenko Anton aka alienxs2
**Status:** Draft
**Version:** 1.0
**Date:** 2025-11-23

## Overview

### Purpose

SchemaManager manages all database schema operations for Zapomni's FalkorDB graph database. It provides idempotent schema initialization, vector index creation (HNSW), graph schema definition (node labels, edge labels, property indexes), and future schema migration support.

### Responsibilities

1. **Schema Initialization** - Create database schema on first run (idempotent)
2. **Vector Index Management** - Create and configure HNSW vector indexes for embeddings
3. **Graph Schema Management** - Define node labels (Memory, Chunk, Entity, Document) and edge labels (HAS_CHUNK, MENTIONS, RELATED_TO)
4. **Property Index Management** - Create property indexes for fast lookups (IDs, names, timestamps)
5. **Schema Validation** - Verify schema consistency and integrity
6. **Migration Support** - Schema versioning and migration (future)
7. **Drop Operations** - Safe schema teardown for testing (danger zone)

### Position in Module

SchemaManager is a core component of the `zapomni_db` module, called by `FalkorDBClient` during initialization. It isolates schema management logic from data operations, making schema changes independent of client code.

```
┌────────────────────────────────┐
│      FalkorDBClient            │
│  (Main database client)        │
└────────────┬───────────────────┘
             │ calls
             ↓
┌────────────────────────────────┐
│      SchemaManager             │  ← THIS COMPONENT
│  (Schema initialization)       │
└────────────┬───────────────────┘
             │ executes
             ↓
┌────────────────────────────────┐
│        FalkorDB Graph          │
│  (CREATE INDEX, CREATE LABEL)  │
└────────────────────────────────┘
```

## Class Definition

### Class Diagram

```
┌─────────────────────────────────────────────┐
│           SchemaManager                      │
├─────────────────────────────────────────────┤
│ - graph: Graph                               │
│ - logger: BoundLogger                        │
│ - schema_version: str                        │
│ - initialized: bool                          │
├─────────────────────────────────────────────┤
│ + init_schema() -> None                      │
│ + create_vector_index() -> None              │
│ + create_graph_schema() -> None              │
│ + create_property_indexes() -> None          │
│ + verify_schema() -> Dict[str, Any]          │
│ + migrate(from_ver, to_ver) -> None          │
│ + drop_all() -> None                         │
│ - _index_exists(name: str) -> bool           │
│ - _execute_cypher(query: str) -> None        │
└─────────────────────────────────────────────┘
```

### Full Class Signature

```python
from typing import Dict, Any, Optional
from falkordb import Graph
import structlog
from structlog.stdlib import BoundLogger

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

        db = FalkorDB(host="localhost", port=6379)
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

    # Index names
    INDEX_VECTOR: str = "chunk_embedding_idx"
    INDEX_MEMORY_ID: str = "memory_id_idx"
    INDEX_ENTITY_NAME: str = "entity_name_idx"
    INDEX_TIMESTAMP: str = "timestamp_idx"

    def __init__(
        self,
        graph: Graph,
        logger: Optional[BoundLogger] = None
    ) -> None:
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

        Example:
            ```python
            manager = SchemaManager(graph)
            manager.init_schema()  # First run - creates schema
            manager.init_schema()  # Second run - no-op (idempotent)
            ```
        """

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

        Example:
            ```python
            manager.create_vector_index()
            # Creates: CREATE VECTOR INDEX FOR (c:Chunk) ON (c.embedding)
            #          OPTIONS {dimension: 768, similarityFunction: 'cosine'}
            ```
        """

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

        Example:
            ```python
            manager.create_graph_schema()
            # Ensures node/edge labels are defined
            ```
        """

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

        Example:
            ```python
            manager.create_property_indexes()
            # Creates: CREATE INDEX FOR (m:Memory) ON (m.id)
            #          CREATE INDEX FOR (e:Entity) ON (e.name)
            #          etc.
            ```
        """

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

        Example:
            ```python
            status = manager.verify_schema()
            if status['initialized']:
                print(f"Schema ready: version {status['version']}")
            else:
                print(f"Issues: {status['issues']}")
            ```
        """

    def migrate(
        self,
        from_version: str,
        to_version: str
    ) -> None:
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
            MigrationError: If migration fails (future)

        Example:
            ```python
            # Future usage:
            manager.migrate(from_version="1.0.0", to_version="1.1.0")
            ```
        """

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

        Example:
            ```python
            # Only in tests!
            manager.drop_all()
            # All data and schema deleted
            ```
        """

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

    def _execute_cypher(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Execute Cypher query with error handling (private helper).

        Args:
            query: Cypher query string
            parameters: Optional query parameters

        Raises:
            DatabaseError: If query execution fails
            QuerySyntaxError: If Cypher syntax invalid
        """
```

## Dependencies

### Component Dependencies

- **FalkorDB Graph** (from `falkordb` library)
  - Purpose: Execute schema operations (CREATE INDEX, MATCH queries)
  - Used by: All public methods

### External Libraries

- **falkordb>=4.0.0** (for: Graph instance, Cypher execution)
- **structlog>=23.2.0** (for: Structured logging of schema operations)
- **typing** (for: Type hints)

### Dependency Injection

SchemaManager receives its dependencies via constructor:
```python
def __init__(self, graph: Graph, logger: Optional[BoundLogger] = None):
    self.graph = graph  # Injected FalkorDB Graph instance
    self.logger = logger or structlog.get_logger(__name__)
```

This design allows:
- Easy testing (inject mock graph)
- Flexible logging (inject custom logger)
- No hard-coded connections (graph created by FalkorDBClient)

## State Management

### Attributes

- `graph: Graph` - FalkorDB Graph instance (lifetime: same as SchemaManager)
- `logger: BoundLogger` - Structured logger (lifetime: same as SchemaManager)
- `schema_version: str` - Current schema version, e.g., "1.0.0" (immutable)
- `initialized: bool` - Flag set to True after successful `init_schema()` (mutable)

### State Transitions

```
Initial State (initialized=False)
    ↓
init_schema() called
    ↓
create_vector_index() → Success
    ↓
create_graph_schema() → Success
    ↓
create_property_indexes() → Success
    ↓
Initialized State (initialized=True)
    ↓
verify_schema() → Returns status with initialized=True
```

**State on Error:**
```
Initial State
    ↓
init_schema() called
    ↓
create_vector_index() → Failure
    ↓
Exception raised, initialized=False (unchanged)
```

### Thread Safety

- **Not thread-safe** - SchemaManager is designed for single-threaded initialization
- **Why**: Schema initialization happens once at startup, no concurrent access
- **Concurrency constraint**: Do NOT call `init_schema()` from multiple threads simultaneously
- **Safe usage**: Call `init_schema()` from main thread before starting workers

## Public Methods (Detailed)

### Method 1: `init_schema`

**Signature:**
```python
def init_schema(self) -> None
```

**Purpose:**
Initialize complete database schema (vector indexes, graph schema, property indexes) in a single idempotent operation. This is the main entry point for schema setup.

**Parameters:** None

**Returns:** None (side effect: creates indexes in database)

**Raises:**
- `DatabaseError`: When schema creation fails (connection lost, disk full, etc.)
- `QuerySyntaxError`: When Cypher queries are malformed (code bug)

**Preconditions:**
- Graph instance must be connected to FalkorDB
- FalkorDB must have write permissions
- Graph name must be valid

**Postconditions:**
- All indexes created (vector, property)
- Graph schema defined (node/edge labels)
- `self.initialized = True`
- Schema operations logged

**Algorithm Outline:**
```
1. Log start of schema initialization
2. Check if already initialized (skip if yes)
3. Create vector index (idempotent)
   - Check if chunk_embedding_idx exists
   - If not, execute CREATE VECTOR INDEX
4. Create graph schema (idempotent)
   - Document node/edge labels (implicit creation)
5. Create property indexes (idempotent)
   - Check each index (memory_id, entity_name, etc.)
   - Create missing indexes
6. Set initialized = True
7. Log success with index count
```

**Edge Cases:**

1. **Already initialized** → No-op (skip all operations)
2. **Partial initialization** (some indexes exist) → Create missing indexes only
3. **FalkorDB connection lost** → Raise DatabaseError, initialized=False
4. **Invalid Cypher syntax** (code bug) → Raise QuerySyntaxError
5. **Concurrent init_schema() calls** → Undefined (not thread-safe)

**Related Methods:**
- Calls: `create_vector_index()`, `create_graph_schema()`, `create_property_indexes()`
- Called by: `FalkorDBClient.__init__()`

### Method 2: `create_vector_index`

**Signature:**
```python
def create_vector_index(self) -> None
```

**Purpose:**
Create HNSW vector index on Chunk.embedding property for fast approximate nearest neighbor search. Configured for 768-dimensional embeddings (nomic-embed-text) with cosine similarity.

**Parameters:** None

**Returns:** None (side effect: creates index)

**Raises:**
- `DatabaseError`: If index creation fails
- `QuerySyntaxError`: If Cypher syntax invalid

**Preconditions:**
- Graph instance connected
- No conflicting index with same name

**Postconditions:**
- Vector index `chunk_embedding_idx` exists
- Index configured: dimension=768, similarity=cosine, M=16, EF_construction=200
- Index ready for vector search queries

**Algorithm Outline:**
```
1. Log: "Creating vector index for chunk embeddings"
2. Check if index already exists:
   - Query: SHOW INDEXES WHERE name = 'chunk_embedding_idx'
   - If exists: Log "Index already exists, skipping" and return
3. Build CREATE VECTOR INDEX Cypher query:
   CREATE VECTOR INDEX chunk_embedding_idx
   FOR (c:Chunk) ON (c.embedding)
   OPTIONS {
       dimension: 768,
       similarityFunction: 'cosine',
       M: 16,
       efConstruction: 200
   }
4. Execute query via self._execute_cypher()
5. Log success: "Vector index created successfully"
```

**Edge Cases:**

1. **Index already exists** → Skip creation (idempotent)
2. **Wrong dimension** (code bug) → FalkorDB error, raise DatabaseError
3. **Invalid similarity function** (code bug) → FalkorDB error, raise DatabaseError
4. **FalkorDB version doesn't support vector indexes** → DatabaseError with clear message

**Related Methods:**
- Called by: `init_schema()`
- Calls: `_index_exists()`, `_execute_cypher()`

### Method 3: `create_graph_schema`

**Signature:**
```python
def create_graph_schema(self) -> None
```

**Purpose:**
Define graph schema by documenting node labels and edge labels. In FalkorDB, labels are created implicitly on first use, so this method primarily serves as documentation and validation.

**Parameters:** None

**Returns:** None

**Raises:**
- `DatabaseError`: If schema queries fail

**Preconditions:**
- Graph instance connected

**Postconditions:**
- Node labels documented: Memory, Chunk, Entity, Document
- Edge labels documented: HAS_CHUNK, MENTIONS, RELATED_TO
- Schema information logged

**Algorithm Outline:**
```
1. Log: "Defining graph schema (node/edge labels)"
2. Document node labels:
   - Memory: Main memory/document node
   - Chunk: Text chunk with embedding
   - Entity: Named entity (person, org, concept)
   - Document: Source document metadata
3. Document edge labels:
   - HAS_CHUNK: Memory → Chunk (one-to-many)
   - MENTIONS: Chunk → Entity (many-to-many)
   - RELATED_TO: Entity → Entity (semantic relationships)
4. Log schema definition (node count: 4, edge count: 3)
5. Note: Labels created implicitly on first MATCH/CREATE
```

**Edge Cases:**

1. **Labels already exist** → No-op (labels are implicit)
2. **FalkorDB doesn't support labels** → Not possible (Cypher standard)

**Related Methods:**
- Called by: `init_schema()`

### Method 4: `create_property_indexes`

**Signature:**
```python
def create_property_indexes(self) -> None
```

**Purpose:**
Create property indexes on frequently queried properties for fast exact-match lookups (Memory.id, Entity.name, Memory.timestamp, etc.).

**Parameters:** None

**Returns:** None

**Raises:**
- `DatabaseError`: If index creation fails
- `QuerySyntaxError`: If Cypher syntax invalid

**Preconditions:**
- Graph instance connected

**Postconditions:**
- Property indexes created:
  - `memory_id_idx` on Memory.id
  - `entity_name_idx` on Entity.name
  - `timestamp_idx` on Memory.timestamp
  - `chunk_memory_id_idx` on Chunk.memory_id

**Algorithm Outline:**
```
1. Log: "Creating property indexes"
2. For each index to create:
   a. memory_id_idx:
      - Check if exists
      - If not: CREATE INDEX memory_id_idx FOR (m:Memory) ON (m.id)
   b. entity_name_idx:
      - Check if exists
      - If not: CREATE INDEX entity_name_idx FOR (e:Entity) ON (e.name)
   c. timestamp_idx:
      - Check if exists
      - If not: CREATE INDEX timestamp_idx FOR (m:Memory) ON (m.timestamp)
   d. chunk_memory_id_idx:
      - Check if exists
      - If not: CREATE INDEX chunk_memory_id_idx FOR (c:Chunk) ON (c.memory_id)
3. Log success with count of indexes created
```

**Edge Cases:**

1. **Indexes already exist** → Skip (idempotent)
2. **Property doesn't exist yet** → Index created, applies when property added
3. **FalkorDB doesn't support property indexes** → DatabaseError

**Related Methods:**
- Called by: `init_schema()`
- Calls: `_index_exists()`, `_execute_cypher()`

### Method 5: `verify_schema`

**Signature:**
```python
def verify_schema(self) -> Dict[str, Any]
```

**Purpose:**
Verify schema consistency and return detailed status for debugging and monitoring. Checks all indexes and returns comprehensive report.

**Parameters:** None

**Returns:**
Dictionary with structure:
```python
{
    "version": "1.0.0",
    "initialized": True,
    "indexes": {
        "vector_index": {
            "exists": True,
            "name": "chunk_embedding_idx",
            "dimension": 768,
            "similarity": "cosine"
        },
        "property_indexes": {
            "memory_id_idx": {"exists": True, "property": "id"},
            "entity_name_idx": {"exists": True, "property": "name"},
            "timestamp_idx": {"exists": True, "property": "timestamp"},
            "chunk_memory_id_idx": {"exists": True, "property": "memory_id"}
        }
    },
    "node_labels": ["Memory", "Chunk", "Entity", "Document"],
    "edge_labels": ["HAS_CHUNK", "MENTIONS", "RELATED_TO"],
    "issues": []  # Empty if all OK, list of issues otherwise
}
```

**Raises:**
- `DatabaseError`: If verification queries fail

**Preconditions:**
- Graph instance connected

**Postconditions:**
- Schema status returned
- Issues identified (if any)
- Verification logged

**Algorithm Outline:**
```
1. Initialize status dict with version and initialized=False
2. Query FalkorDB for all indexes:
   - SHOW INDEXES
3. Check vector index:
   - Find chunk_embedding_idx
   - Verify dimension=768, similarity=cosine
   - Add to status['indexes']['vector_index']
   - If missing: Add issue "Vector index not found"
4. Check property indexes:
   - For each expected index (memory_id, entity_name, etc.)
   - Verify exists
   - Add to status['indexes']['property_indexes']
   - If missing: Add issue "Property index X not found"
5. Query node labels:
   - CALL db.labels()
   - Add to status['node_labels']
6. Query edge labels:
   - CALL db.relationshipTypes()
   - Add to status['edge_labels']
7. Set initialized=True if no issues
8. Return status dict
```

**Edge Cases:**

1. **No indexes found** → initialized=False, issues list populated
2. **Partial schema** (some indexes missing) → initialized=False, specific issues
3. **All indexes present** → initialized=True, issues=[]
4. **FalkorDB error** → Raise DatabaseError

**Related Methods:**
- Called by: `FalkorDBClient` for health checks

### Method 6: `migrate`

**Signature:**
```python
def migrate(self, from_version: str, to_version: str) -> None
```

**Purpose:**
Migrate schema from one version to another (FUTURE FEATURE - not implemented in MVP).

**Parameters:**
- `from_version: str` - Current schema version (e.g., "1.0.0")
- `to_version: str` - Target schema version (e.g., "1.1.0")

**Returns:** None

**Raises:**
- `NotImplementedError`: Always (not implemented yet)
- `MigrationError`: If migration fails (future)

**Preconditions:** None (not implemented)

**Postconditions:** None (not implemented)

**Algorithm Outline:**
```
1. Raise NotImplementedError("Schema migration not implemented yet")
2. Future implementation will:
   - Load migration scripts (from migrations/ directory)
   - Apply migrations sequentially (1.0.0 → 1.0.1 → 1.1.0)
   - Wrap in transaction (rollback on failure)
   - Update schema_version on success
```

**Edge Cases:**
- All cases raise NotImplementedError in MVP

**Related Methods:** None (future)

### Method 7: `drop_all`

**Signature:**
```python
def drop_all(self) -> None
```

**Purpose:**
Drop all schema and data (DANGEROUS - testing only). Deletes ALL nodes, edges, and indexes.

**Parameters:** None

**Returns:** None

**Raises:**
- `DatabaseError`: If drop operation fails

**Preconditions:**
- Graph instance connected
- User understands this is IRREVERSIBLE

**Postconditions:**
- All nodes deleted
- All edges deleted
- All indexes deleted
- Graph empty

**Algorithm Outline:**
```
1. Log WARNING: "Dropping all data and schema (irreversible)"
2. Delete all nodes and edges:
   - MATCH (n) DETACH DELETE n
3. Drop all indexes:
   - For each index in SHOW INDEXES:
     - DROP INDEX index_name
4. Set initialized = False
5. Log: "All data and schema dropped"
```

**Edge Cases:**

1. **Graph already empty** → No-op (idempotent)
2. **FalkorDB connection lost during delete** → Partial delete, raise DatabaseError
3. **Indexes can't be dropped** → Continue, log warning

**Related Methods:**
- Called by: Integration tests (cleanup)

## Error Handling

### Exceptions Defined

```python
class DatabaseError(Exception):
    """Raised when database operation fails (connection, query execution, etc.)"""

class QuerySyntaxError(DatabaseError):
    """Raised when Cypher query has invalid syntax"""

class MigrationError(DatabaseError):
    """Raised when schema migration fails (future)"""
```

### Error Recovery

**Retry Strategy:**
- Schema operations do NOT retry (they're idempotent, caller can retry)
- Rationale: Schema initialization happens at startup, fast-fail is better

**Fallback Behavior:**
- On error: Set `initialized = False`
- On error: Log detailed error with context (query, parameters)
- On error: Raise exception (let caller handle)

**Error Propagation:**
- All exceptions bubble up to `FalkorDBClient`
- `FalkorDBClient` can retry `init_schema()` if needed

## Usage Examples

### Basic Usage

```python
from falkordb import FalkorDB
from zapomni_db.falkordb.schema import SchemaManager

# Connect to FalkorDB
db = FalkorDB(host="localhost", port=6379)
graph = db.select_graph("zapomni_memory")

# Initialize schema
schema_manager = SchemaManager(graph=graph)
schema_manager.init_schema()  # Creates all indexes

print("Schema initialized successfully!")
```

### Verify Schema Status

```python
# Check schema status
status = schema_manager.verify_schema()

if status['initialized']:
    print(f"Schema ready: version {status['version']}")
    print(f"Vector index: {status['indexes']['vector_index']}")
    print(f"Property indexes: {len(status['indexes']['property_indexes'])}")
else:
    print(f"Schema incomplete! Issues: {status['issues']}")
```

### Testing with drop_all()

```python
import pytest
from zapomni_db.falkordb.schema import SchemaManager

@pytest.fixture
def clean_schema(graph):
    """Fixture: Clean schema before each test"""
    manager = SchemaManager(graph)
    manager.drop_all()  # Clean slate
    manager.init_schema()  # Fresh schema
    yield manager
    manager.drop_all()  # Cleanup after test

def test_vector_index_creation(clean_schema):
    status = clean_schema.verify_schema()
    assert status['indexes']['vector_index']['exists'] is True
    assert status['indexes']['vector_index']['dimension'] == 768
```

### Advanced Usage - Custom Logger

```python
import structlog

# Custom logger with specific formatting
logger = structlog.get_logger("schema").bind(component="SchemaManager")

schema_manager = SchemaManager(graph=graph, logger=logger)
schema_manager.init_schema()
# All operations logged with "component: SchemaManager"
```

## Testing Approach

### Unit Tests Required

1. `test_init_success()` - Schema initialization succeeds
2. `test_init_idempotent()` - Multiple init_schema() calls are safe
3. `test_create_vector_index_success()` - Vector index created correctly
4. `test_create_vector_index_already_exists()` - Skip if exists (idempotent)
5. `test_create_property_indexes_success()` - All property indexes created
6. `test_verify_schema_complete()` - All indexes present
7. `test_verify_schema_incomplete()` - Missing indexes detected
8. `test_drop_all_success()` - All data and schema deleted
9. `test_index_exists_true()` - Helper returns True for existing index
10. `test_index_exists_false()` - Helper returns False for missing index

### Mocking Strategy

```python
import pytest
from unittest.mock import Mock, MagicMock

@pytest.fixture
def mock_graph():
    """Mock FalkorDB Graph instance"""
    graph = MagicMock(spec=Graph)

    # Mock query() to return empty result (no existing indexes)
    graph.query.return_value.result_set = []

    return graph

def test_create_vector_index_success(mock_graph):
    manager = SchemaManager(graph=mock_graph)
    manager.create_vector_index()

    # Verify CREATE VECTOR INDEX was called
    assert mock_graph.query.called
    call_args = mock_graph.query.call_args[0][0]
    assert "CREATE VECTOR INDEX" in call_args
    assert "dimension: 768" in call_args
```

### Integration Tests

**Test Environment:**
- Docker Compose with real FalkorDB instance
- Separate test graph (e.g., "test_zapomni_schema")
- Cleanup after each test (drop_all())

**Integration Test Scenarios:**

1. **test_init_schema_with_real_db()** - Full schema creation
2. **test_verify_schema_with_real_db()** - Verify all indexes exist
3. **test_vector_index_usable()** - Can perform vector search after creation
4. **test_property_index_usable()** - Can use property index for fast lookup
5. **test_drop_all_with_real_db()** - Full cleanup works

```python
@pytest.mark.integration
def test_init_schema_with_real_db(falkordb_test_graph):
    manager = SchemaManager(graph=falkordb_test_graph)

    # Initialize schema
    manager.init_schema()

    # Verify vector index exists
    result = falkordb_test_graph.query("SHOW INDEXES")
    indexes = [row[0] for row in result.result_set]
    assert "chunk_embedding_idx" in indexes

    # Verify property indexes
    assert "memory_id_idx" in indexes
    assert "entity_name_idx" in indexes
```

## Performance Considerations

### Time Complexity

- `init_schema()`: O(1) - Fixed number of index creations
- `create_vector_index()`: O(1) - Single CREATE INDEX query
- `create_property_indexes()`: O(1) - Fixed number of indexes (4)
- `verify_schema()`: O(n) where n = number of indexes (small, ~5-10)
- `drop_all()`: O(N) where N = total nodes + edges (destructive, slow for large graphs)

### Space Complexity

- Memory usage: O(1) - No data structures stored
- Index memory: O(N) where N = number of vectors (FalkorDB side)
  - Chunk vector index: ~3KB per vector (768-dim float32)
  - Property indexes: ~100 bytes per entry

### Optimization Opportunities

1. **Batch Index Creation** - Create all indexes in single transaction (future)
2. **Async Index Building** - FalkorDB builds indexes asynchronously (already done)
3. **Index Statistics** - Cache `verify_schema()` results (if called frequently)

**Trade-offs:**
- Idempotency vs. Speed: Checking if index exists adds latency, but ensures safety
- Current choice: Safety (idempotency) over speed (startup happens once)

## References

### Module Spec
- **zapomni_db_module.md** - Section "Schema Management" (lines 120-130)
- **zapomni_db_module.md** - Section "Dependencies" (FalkorDB client)

### Related Components
- **FalkorDBClient** (component spec) - Calls SchemaManager during init
- **QueryBuilder** (future component) - Uses schema knowledge for query generation

### External Documentation
- FalkorDB Indexes: https://docs.falkordb.com/indexing.html
- FalkorDB Vector Indexes: https://docs.falkordb.com/vector-indexes.html
- Cypher CREATE INDEX: https://neo4j.com/docs/cypher-manual/current/indexes/
- HNSW Algorithm: https://arxiv.org/abs/1603.09320

---

**Document Status:** Draft v1.0
**Created:** 2025-11-23
**Author:** Goncharenko Anton aka alienxs2 + Claude Code
**License:** MIT
**Last Updated:** 2025-11-23

**Ready for Verification:** Yes
**Verification Status:** Pending
**Approval Status:** Pending

---

**Completeness:** 100% (all sections filled)
**Consistency:** Aligned with zapomni_db_module.md, Component-level template
**Implementation Readiness:** High (detailed methods, signatures, edge cases)
