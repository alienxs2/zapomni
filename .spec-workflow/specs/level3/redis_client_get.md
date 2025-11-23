# RedisClient.get() - Function Specification

**Level:** 3 (Function)
**Component:** RedisClient
**Module:** zapomni_db
**Author:** Goncharenko Anton aka alienxs2
**Status:** Draft
**Version:** 1.0
**Created:** 2025-11-23

---

## Function Signature

```python
async def get(
    self,
    key: str
) -> Optional[Any]:
    """
    Retrieve value from Redis cache by key.

    Args:
        key: Cache key (e.g., "embedding:text_hash_abc123")
            - Must be non-empty string
            - Max length: 512 characters

    Returns:
        Optional[Any]: Cached value (deserialized from JSON) or None if:
            - Key doesn't exist
            - Key expired (TTL elapsed)
            - Deserialization failed

    Raises:
        ConnectionError: If Redis unavailable
        ValueError: If key invalid (empty, too long)

    Example:
        >>> cache = RedisClient()
        >>> embedding = await cache.get("embedding:abc123")
        >>> if embedding:
        ...     print(f"Cache hit! {len(embedding)} dims")
        ... else:
        ...     print("Cache miss")
    """
```

## Algorithm

```
ASYNC FUNCTION get(key: str) -> Optional[Any]:
    # Validate key
    IF NOT key OR len(key) > 512:
        RAISE ValueError("Invalid key")

    TRY:
        # Get from Redis
        data = AWAIT self.client.get(key)

        IF data IS None:
            self._logger.debug("cache_miss", key=key)
            RETURN None

        # Deserialize JSON
        value = json.loads(data)
        self._logger.debug("cache_hit", key=key)
        RETURN value

    CATCH redis.ConnectionError:
        RAISE ConnectionError("Redis unavailable")
    CATCH json.JSONDecodeError:
        # Corrupted data - delete key
        AWAIT self.client.delete(key)
        RETURN None
END
```

## Edge Cases

1. **Key doesn't exist:** Return None (not error)
2. **Key expired:** Return None
3. **Corrupted data:** Delete key, return None
4. **Redis down:** Raise ConnectionError
5. **Empty key:** ValidationError
6. **Very long key:** ValidationError

## Test Scenarios

1. test_get_existing_key
2. test_get_nonexistent_key
3. test_get_expired_key
4. test_get_corrupted_data
5. test_get_empty_key_raises
6. test_get_redis_down_raises
7. test_get_deserializes_json

## Performance

- Latency: < 2ms (local Redis), < 10ms (network Redis)
- Throughput: 10,000+ ops/sec

---

**Estimated Implementation:** 30 min | **LoC:** ~25 | **Test File:** `test_redis_client_get.py`
