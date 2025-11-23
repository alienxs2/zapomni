# SchemaManager.create_vector_index() - Function Specification

**Level:** 3 | **Component:** SchemaManager | **Module:** zapomni_db
**Author:** Goncharenko Anton aka alienxs2 | **Status:** Draft | **V:** 1.0

## Signature
```python
async def create_vector_index(
    self,
    index_name: str,
    dimension: int = 768,
    metric: str = "cosine"
) -> None:
    """
    Create HNSW vector index in FalkorDB.
    
    Args:
        index_name: Name for the index (e.g., "chunk_embeddings")
        dimension: Vector dimension (default: 768 for nomic-embed-text)
        metric: Distance metric - "cosine", "euclidean", or "dot_product" (default: "cosine")
        
    Raises:
        ValueError: If dimension < 1 or > 4096
        ValueError: If metric not in allowed values
        DatabaseError: If index creation fails
    """
```

## Purpose

Create HNSW (Hierarchical Navigable Small World) vector index in FalkorDB for efficient similarity search.

## Edge Cases
1. Index already exists → Skip (idempotent) or drop and recreate
2. dimension = 0 → ValueError
3. dimension = 4097 → ValueError
4. metric = "invalid" → ValueError
5. DB connection lost → DatabaseError

## Algorithm
```
1. Validate dimension: 1 <= value <= 4096
2. Validate metric in ["cosine", "euclidean", "dot_product"]
3. Check if index exists (query schema)
4. If exists, skip or drop
5. Execute CREATE INDEX Cypher query
6. Verify index created
7. Log success
```

## Tests (10)
1. test_create_index_success
2. test_create_index_already_exists
3. test_create_index_zero_dimension_raises
4. test_create_index_dimension_too_large_raises
5. test_create_index_invalid_metric_raises
6. test_create_index_euclidean_metric
7. test_create_index_dot_product_metric
8. test_create_index_custom_dimension
9. test_create_index_db_error_raises
10. test_create_index_verifies_creation
