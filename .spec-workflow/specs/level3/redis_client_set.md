# RedisClient.set() - Function Specification

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
async def set(
    self,
    key: str,
    value: Any,
    ttl: int = None
) -> bool:
    """
    Store value in Redis cache with optional TTL.

    Args:
        key: Cache key (non-empty, max 512 chars)
        value: Value to cache (must be JSON-serializable)
        ttl: Time-to-live in seconds (default: self.ttl_seconds = 86400)
            - If None: Use default TTL
            - If 0: No expiration
            - If > 0: Expire after N seconds

    Returns:
        bool: True if stored successfully, False otherwise

    Raises:
        ValueError: If key invalid or value not JSON-serializable
        ConnectionError: If Redis unavailable

    Example:
        >>> cache = RedisClient()
        >>> embedding = [0.1, 0.2, 0.3]
        >>> success = await cache.set("emb:abc", embedding, ttl=3600)
        >>> assert success is True
    """
```

## Algorithm

```
ASYNC FUNCTION set(key: str, value: Any, ttl: int = None) -> bool:
    # Validate key
    IF NOT key OR len(key) > 512:
        RAISE ValueError("Invalid key")

    # Determine TTL
    ttl_to_use = ttl IF ttl IS NOT None ELSE self.ttl_seconds

    TRY:
        # Serialize to JSON
        serialized = json.dumps(value)

        # Store in Redis
        IF ttl_to_use > 0:
            result = AWAIT self.client.setex(key, ttl_to_use, serialized)
        ELSE:
            result = AWAIT self.client.set(key, serialized)

        self._logger.debug("cache_set", key=key, ttl=ttl_to_use)
        RETURN result IS True

    CATCH TypeError:
        RAISE ValueError("Value not JSON-serializable")
    CATCH redis.ConnectionError:
        RAISE ConnectionError("Redis unavailable")
END
```

## Edge Cases

1. **TTL = None:** Use default (86400s)
2. **TTL = 0:** No expiration
3. **TTL = -1:** Invalid (raise ValueError)
4. **Non-serializable value:** Raise ValueError
5. **Redis down:** Raise ConnectionError
6. **Overwrite existing key:** Succeeds

## Test Scenarios

1. test_set_with_default_ttl
2. test_set_with_custom_ttl
3. test_set_no_expiration (ttl=0)
4. test_set_non_serializable_value_raises
5. test_set_empty_key_raises
6. test_set_redis_down_raises
7. test_set_overwrites_existing_key
8. test_set_then_get_roundtrip

## Performance

- Latency: < 2ms (local), < 10ms (network)
- Throughput: 10,000+ ops/sec

---

**Estimated Implementation:** 30 min | **LoC:** ~30 | **Test File:** `test_redis_client_set.py`
