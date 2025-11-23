# CypherQueryBuilder.build_vector_search_query() - Function Specification

**Level:** 3 (Function)
**Component:** CypherQueryBuilder
**Module:** zapomni_db
**Author:** Goncharenko Anton aka alienxs2
**Status:** Draft
**Version:** 1.0
**Created:** 2025-11-23

---

## Function Signature

```python
def build_vector_search_query(
    self,
    embedding: List[float],
    limit: int = 10,
    filters: dict = None
) -> tuple[str, dict]:
    """
    Build parameterized Cypher query for vector similarity search.

    Generates secure, injection-proof Cypher query using FalkorDB's HNSW
    vector index for efficient semantic search. Returns query string and
    parameters for safe execution.

    Args:
        embedding: 768-dimensional query embedding vector
        limit: Maximum results to return (1-1000, default: 10)
        filters: Optional metadata filters {"tags": ["python"], "source": "user"}

    Returns:
        tuple[str, dict]: (cypher_query, parameters)
            - cypher_query: Parameterized Cypher with $placeholders
            - parameters: Dict of parameter values for safe execution

    Raises:
        ValidationError: If embedding invalid (wrong dimension, non-numeric)
        ValueError: If limit out of range

    Example:
        >>> builder = CypherQueryBuilder()
        >>> emb = [0.1] * 768
        >>> cypher, params = builder.build_vector_search_query(emb, limit=5)
        >>> print(cypher[:50])
        'CALL db.idx.vector.queryNodes('chunk_embedding_i...'
        >>> print(params["limit"])
        5
    """
```

## Algorithm

```
FUNCTION build_vector_search_query(embedding, limit, filters):
    # Validate embedding
    IF len(embedding) != 768:
        RAISE ValidationError("Embedding must be 768-dimensional")

    # Validate limit
    IF limit < 1 OR limit > 1000:
        RAISE ValueError("Limit must be 1-1000")

    # Build base query
    cypher = """
        CALL db.idx.vector.queryNodes(
            'chunk_embedding_idx',
            $k,
            $query_embedding
        ) YIELD node AS chunk, score
        MATCH (m:Memory)-[:HAS_CHUNK]->(chunk)
    """

    # Add filters if provided
    IF filters:
        cypher += " WHERE "
        conditions = []
        FOR key, value IN filters:
            conditions.APPEND(f"m.metadata.{key} = ${key}")
        cypher += " AND ".join(conditions)

    # Add ordering and limit
    cypher += " RETURN m, chunk, score ORDER BY score DESC LIMIT $limit"

    # Build parameters
    params = {
        "query_embedding": embedding,
        "k": limit * 2,  # Query more, filter to limit
        "limit": limit
    }

    # Add filter params
    IF filters:
        params.UPDATE(filters)

    RETURN (cypher, params)
END
```

## Edge Cases

1. **Empty filters:** Query without WHERE clause
2. **Limit = 1:** Minimal query
3. **Limit = 1000:** Maximum allowed
4. **Complex filters:** Multiple metadata fields
5. **Vector dimension mismatch:** Validation error

## Test Scenarios

1. test_build_vector_search_basic
2. test_build_vector_search_with_filters
3. test_build_vector_search_max_limit
4. test_build_vector_search_invalid_embedding_dimension
5. test_build_vector_search_limit_out_of_range
6. test_build_vector_search_parameterization (injection prevention)

## Performance

- Query generation: < 1ms
- Query execution: < 50ms (small graph), < 200ms (large graph)

---

**Estimated Implementation:** 1 hour | **LoC:** ~40 | **Test File:** `test_cypher_query_builder_build_vector_search.py`
