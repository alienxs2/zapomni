# SemanticCache.set_embedding() - Function Specification

**Level:** 3 (Function)
**Component:** SemanticCache
**Module:** zapomni_core
**Author:** Goncharenko Anton aka alienxs2
**Status:** Draft
**Version:** 1.0
**Created:** 2025-11-23

## Function Signature

```python
def set_embedding(self, text: str, embedding: List[float]) -> None:
    """Cache embedding for text."""
```

## Purpose

Store embedding in Redis with TTL and LRU tracking to speed up future retrievals.

## Parameters

### text: str
- **Constraints:** Non-empty after normalization
- **Example:** `"Python is great"`

### embedding: List[float]
- **Constraints:**
  - Length: 768 or 384 (standard dimensions)
  - All elements finite (no NaN/Inf)
- **Example:** `[0.1, 0.2, ..., 0.768]`

## Returns

None (void method)

## Raises

- `ValueError`: If text empty or embedding invalid
- `RedisError`: If Redis operation fails

## Algorithm

```
1. Validate text (non-empty)
2. Validate embedding:
   - Length in [384, 768]
   - All elements finite
3. Normalize text
4. Compute hash
5. Serialize embedding: json.dumps(embedding)
6. Redis pipeline (atomic):
   - SET zapomni:cache:embedding:{hash} {json} EX {ttl}
   - ZADD zapomni:cache:lru {timestamp} {hash}
   - IF count > max_entries:
       ZREMRANGEBYRANK (evict oldest)
7. Execute pipeline
```

## Edge Cases

1. **Empty text** → ValueError
2. **Wrong dimension** (512) → ValueError
3. **Contains NaN** → ValueError
4. **Cache at max capacity** → Evict oldest, insert new
5. **Redis OOM** → Log error, raise RedisError

## Test Scenarios (10)

1. test_set_embedding_success
2. test_set_embedding_empty_text_raises
3. test_set_embedding_wrong_dimension_raises
4. test_set_embedding_nan_raises
5. test_set_embedding_inf_raises
6. test_set_embedding_lru_eviction
7. test_set_embedding_redis_oom
8. test_set_embedding_ttl_set
9. test_set_embedding_idempotent
10. test_set_embedding_performance

## Performance

- Target: < 10ms (P95)

**Status:** Draft v1.0 | **Author:** Goncharenko Anton aka alienxs2 | **License:** MIT
