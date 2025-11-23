# SemanticCache.get_embedding() - Function Specification

**Level:** 3 (Function)
**Component:** SemanticCache
**Module:** zapomni_core
**Author:** Goncharenko Anton aka alienxs2
**Status:** Draft
**Version:** 1.0
**Created:** 2025-11-23

## Function Signature

```python
def get_embedding(self, text: str) -> Optional[List[float]]:
    """Retrieve cached embedding for text."""
```

## Purpose

Check Redis cache for pre-computed embedding to avoid regenerating via Ollama.

## Parameters

### text: str
- **Constraints:** Non-empty after normalization
- **Processing:** Strip whitespace, lowercase
- **Examples:**
  - Valid: `"Python is great"`
  - Invalid: `""`, `"   "`

## Returns

- **HIT:** `[0.123, 0.456, ..., 0.789]` (768-dim list)
- **MISS:** `None`

## Raises

- `ValueError`: If text empty after normalization
- `RedisError`: Caught and returns None (graceful degradation)

## Algorithm

```
1. Validate text (non-empty)
2. Normalize: text.strip().lower()
3. Compute hash: SHA256(normalized_text)
4. Build key: "zapomni:cache:embedding:{hash}"
5. Query Redis: GET key
6. IF found:
     Deserialize JSON → List[float]
     Increment hit_count
     RETURN embedding
7. ELSE:
     Increment miss_count
     RETURN None
```

## Edge Cases

1. **Empty text** → ValueError
2. **Whitespace only** → ValueError (after normalize)
3. **Very long text** → Hash successfully, query Redis
4. **Redis unavailable** → Log error, return None
5. **Expired entry** → Redis returns None (auto-deleted)
6. **Special characters** → Normalize and hash correctly

## Test Scenarios (10+)

1. test_get_embedding_hit
2. test_get_embedding_miss
3. test_get_embedding_empty_raises
4. test_get_embedding_whitespace_raises
5. test_get_embedding_special_chars
6. test_get_embedding_large_text
7. test_get_embedding_redis_error_returns_none
8. test_get_embedding_expired_returns_none
9. test_get_embedding_increments_hit_count
10. test_get_embedding_increments_miss_count
11. test_get_embedding_latency_hit
12. test_get_embedding_latency_miss

## Performance

- **Cache HIT:** < 5ms (P95)
- **Cache MISS:** < 2ms (P95)

**Status:** Draft v1.0 | **Author:** Goncharenko Anton aka alienxs2 | **License:** MIT
