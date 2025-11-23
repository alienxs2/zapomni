# SchemaManager.init_schema() - Function Specification

**Level:** 3 (Function)
**Component:** SchemaManager
**Module:** zapomni_db
**Author:** Goncharenko Anton aka alienxs2
**Status:** Draft
**Version:** 1.0
**Created:** 2025-11-23

## Function Signature

```python
def init_schema(self) -> None:
    """Initialize complete database schema (idempotent)."""
```

## Purpose

Create all necessary indexes and schema elements in correct order:
1. Vector indexes (HNSW for embeddings)
2. Graph schema (node/edge labels)
3. Property indexes (for fast lookups)

## Parameters

None

## Returns

None (side effect: creates indexes in database)

## Raises

- `DatabaseError`: Schema creation fails
- `QuerySyntaxError`: Cypher queries malformed

## Algorithm

```
1. Log start of schema initialization
2. Check if already initialized (skip if yes)
3. Create vector index (idempotent):
   - Check if chunk_embedding_idx exists
   - If not, execute CREATE VECTOR INDEX
4. Create graph schema (idempotent):
   - Document node/edge labels
5. Create property indexes (idempotent):
   - Check each index exists
   - Create missing indexes
6. Set initialized = True
7. Log success
```

## Edge Cases

1. **Already initialized** → No-op (idempotent)
2. **Partial initialization** → Create missing indexes only
3. **FalkorDB connection lost** → DatabaseError
4. **Invalid Cypher syntax** → QuerySyntaxError
5. **Concurrent init_schema() calls** → Undefined (not thread-safe)

## Test Scenarios (10)

1. test_init_schema_success
2. test_init_schema_idempotent
3. test_init_schema_partial_indexes
4. test_init_schema_database_error
5. test_init_schema_query_syntax_error
6. test_init_schema_sets_initialized_flag
7. test_init_schema_creates_vector_index
8. test_init_schema_creates_property_indexes
9. test_init_schema_logging
10. test_init_schema_integration_real_db

## Performance

- Target: < 1 second (first run)
- Target: < 50ms (subsequent runs, idempotent check)

## Related Methods

- Calls: `create_vector_index()`, `create_graph_schema()`, `create_property_indexes()`
- Called by: `FalkorDBClient.__init__()`

**Status:** Draft v1.0 | **Author:** Goncharenko Anton aka alienxs2 | **License:** MIT
