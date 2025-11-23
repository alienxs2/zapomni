# RedisClient - Component Specification

**Level:** 2 (Component)
**Module:** zapomni_db
**Author:** Goncharenko Anton aka alienxs2
**Status:** Draft
**Version:** 1.0
**Date:** 2025-11-23

## Overview

### Purpose

RedisClient is a lightweight wrapper for Redis operations used by the semantic caching layer in Zapomni. It provides a simplified interface for storing and retrieving embeddings with TTL (Time-To-Live) support, enabling fast cache lookups to reduce Ollama API calls.

### Responsibilities

1. Establish and maintain Redis connection with connection pooling
2. Provide get/set/delete operations for embedding cache
3. Implement TTL-based cache expiration (default: 24 hours)
4. Support pattern-based key scanning for cache management
5. Provide cache statistics (memory usage, key counts, hit rate)
6. Handle connection failures with graceful degradation
7. Serialize/deserialize complex objects (embeddings as JSON)

### Position in Module

RedisClient is part of the `zapomni_db.redis_cache` sub-package within the `zapomni_db` module. It is used by the SemanticCache component in `zapomni_core` to cache embeddings and reduce latency.

```
zapomni_db/
├── falkordb/
│   └── client.py (FalkorDBClient)
└── redis_cache/
    └── cache_client.py (RedisClient) ← THIS COMPONENT
```

## Class Definition

### Class Diagram

```
┌─────────────────────────────────────┐
│          RedisClient                │
├─────────────────────────────────────┤
│ - host: str                         │
│ - port: int                         │
│ - db: int                           │
│ - ttl_seconds: int                  │
│ - password: Optional[str]           │
│ - client: redis.Redis               │
│ - pool: redis.ConnectionPool        │
├─────────────────────────────────────┤
│ + __init__(host, port, db, ...)    │
│ + get(key) -> Optional[Any]         │
│ + set(key, value, ttl) -> bool      │
│ + delete(key) -> bool               │
│ + scan(pattern) -> List[str]        │
│ + info() -> Dict[str, Any]          │
│ + close() -> None                   │
│ - _serialize(value) -> str          │
│ - _deserialize(data) -> Any         │
└─────────────────────────────────────┘
```

### Full Class Signature

```python
from typing import Optional, Dict, Any, List
import redis
import json
import structlog

logger = structlog.get_logger(__name__)

class RedisClient:
    """
    Lightweight wrapper for Redis operations used by semantic caching.

    Provides a simplified interface for storing embeddings and other
    cache data with TTL support. Handles connection pooling, serialization,
    and error recovery.

    Attributes:
        host: Redis server host address
        port: Redis server port
        db: Redis database number (0-15)
        ttl_seconds: Default TTL for cache entries (seconds)
        password: Optional Redis password for authentication
        client: Redis client instance
        pool: Redis connection pool

    Example:
        ```python
        # Initialize client
        cache = RedisClient(
            host="localhost",
            port=6380,
            db=0,
            ttl_seconds=86400  # 24 hours
        )

        # Cache an embedding
        embedding = [0.1, 0.2, 0.3, ...]
        cache.set("text_hash_abc123", embedding, ttl=3600)

        # Retrieve cached embedding
        cached_embedding = cache.get("text_hash_abc123")
        if cached_embedding:
            print("Cache hit!")

        # Get cache stats
        stats = cache.info()
        print(f"Memory usage: {stats['used_memory_mb']} MB")

        # Close connection
        cache.close()
        ```
    """

    # Class constants
    DEFAULT_TTL_SECONDS: int = 86400  # 24 hours
    DEFAULT_DB: int = 0
    MAX_RETRIES: int = 3
    CONNECTION_TIMEOUT: int = 5

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6380,
        db: int = 0,
        ttl_seconds: int = 86400,
        password: Optional[str] = None,
        max_connections: int = 10,
        decode_responses: bool = False
    ) -> None:
        """
        Initialize Redis client with connection pooling.

        Args:
            host: Redis server host address (default: "localhost")
            port: Redis server port (default: 6380)
            db: Redis database number 0-15 (default: 0)
            ttl_seconds: Default TTL for cache entries in seconds (default: 86400 = 24h)
            password: Optional Redis password for authentication
            max_connections: Maximum connections in pool (default: 10)
            decode_responses: Whether to decode byte responses to strings (default: False)

        Raises:
            ConnectionError: If cannot connect to Redis server
            ValueError: If db is not in range 0-15

        Example:
            ```python
            # Basic initialization
            cache = RedisClient()

            # With authentication
            cache = RedisClient(
                host="redis.example.com",
                port=6379,
                password="secret123"
            )

            # Custom TTL (1 hour)
            cache = RedisClient(ttl_seconds=3600)
            ```
        """

    def get(
        self,
        key: str
    ) -> Optional[Any]:
        """
        Get value from cache by key.

        Retrieves cached value, deserializes from JSON, and returns.
        Returns None if key doesn't exist or has expired.

        Args:
            key: Cache key (string identifier)

        Returns:
            Cached value (deserialized from JSON) if found, None otherwise

        Raises:
            ValueError: If key is empty or invalid
            DeserializationError: If cached data is corrupted

        Example:
            ```python
            # Get cached embedding
            embedding = cache.get("embedding_abc123")
            if embedding is not None:
                print(f"Found cached embedding with {len(embedding)} dimensions")
            else:
                print("Cache miss - need to generate embedding")

            # Get cached metadata
            metadata = cache.get("metadata_xyz789")
            ```
        """

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in cache with TTL.

        Serializes value to JSON and stores in Redis with expiration time.
        Uses default TTL if not specified.

        Args:
            key: Cache key (string identifier)
            value: Value to cache (must be JSON-serializable)
            ttl: Time-to-live in seconds (optional, uses default if None)

        Returns:
            True if successfully stored, False otherwise

        Raises:
            ValueError: If key is empty or ttl is negative
            SerializationError: If value cannot be serialized to JSON
            CacheError: If Redis write operation fails

        Example:
            ```python
            # Cache embedding with default TTL (24h)
            embedding = [0.1] * 768
            success = cache.set("embedding_abc123", embedding)

            # Cache with custom TTL (1 hour)
            cache.set("temp_data", {"foo": "bar"}, ttl=3600)

            # Cache complex object
            cache.set("metadata", {
                "tags": ["python", "redis"],
                "timestamp": "2025-11-23",
                "confidence": 0.95
            })
            ```
        """

    def delete(
        self,
        key: str
    ) -> bool:
        """
        Delete key from cache.

        Removes the specified key and its value from Redis.

        Args:
            key: Cache key to delete

        Returns:
            True if key was deleted, False if key didn't exist

        Raises:
            ValueError: If key is empty
            CacheError: If Redis delete operation fails

        Example:
            ```python
            # Delete single key
            deleted = cache.delete("embedding_abc123")
            if deleted:
                print("Cache entry removed")

            # Delete multiple keys
            for key in ["key1", "key2", "key3"]:
                cache.delete(key)
            ```
        """

    def scan(
        self,
        pattern: str = "*",
        count: int = 100
    ) -> List[str]:
        """
        Scan cache keys by pattern.

        Iterates through keys matching the given pattern using SCAN
        (cursor-based iteration, not blocking).

        Args:
            pattern: Key pattern with wildcards (* and ?) (default: "*" = all keys)
            count: Hint for number of keys per iteration (default: 100)

        Returns:
            List of matching key names (strings)

        Raises:
            ValueError: If pattern is empty
            CacheError: If Redis scan operation fails

        Example:
            ```python
            # Get all embedding keys
            embedding_keys = cache.scan("embedding_*")
            print(f"Found {len(embedding_keys)} cached embeddings")

            # Get keys for specific text hash
            keys = cache.scan("text_hash_abc*")

            # Get all keys
            all_keys = cache.scan()

            # Delete all embeddings
            for key in cache.scan("embedding_*"):
                cache.delete(key)
            ```
        """

    def info(self) -> Dict[str, Any]:
        """
        Get Redis cache statistics and information.

        Retrieves memory usage, key counts, and other useful metrics
        from Redis INFO command.

        Returns:
            Dictionary with keys:
                - used_memory_mb: Memory used by Redis (float, MB)
                - used_memory_human: Human-readable memory (str, e.g., "15.2M")
                - total_keys: Total number of keys in database (int)
                - db_name: Database name (str, e.g., "db0")
                - maxmemory_mb: Max memory limit (float, MB, 0 = unlimited)
                - eviction_policy: Eviction policy (str, e.g., "noeviction")

        Raises:
            CacheError: If Redis INFO command fails

        Example:
            ```python
            # Get cache statistics
            stats = cache.info()
            print(f"Memory usage: {stats['used_memory_mb']:.2f} MB")
            print(f"Total keys: {stats['total_keys']}")
            print(f"Eviction policy: {stats['eviction_policy']}")

            # Check memory usage threshold
            if stats['used_memory_mb'] > 500:
                print("WARNING: Cache using > 500 MB")
            ```
        """

    def close(self) -> None:
        """
        Close Redis connection and release resources.

        Closes the connection pool and releases all connections.
        Should be called when done using the cache.

        Raises:
            CacheError: If connection close fails

        Example:
            ```python
            cache = RedisClient()
            try:
                # Use cache
                cache.set("key", "value")
            finally:
                # Always close
                cache.close()

            # Or use context manager (future)
            async with RedisClient() as cache:
                cache.set("key", "value")
            ```
        """

    def _serialize(self, value: Any) -> str:
        """
        Serialize value to JSON string.

        Internal method for converting Python objects to JSON for storage.

        Args:
            value: Python object to serialize

        Returns:
            JSON string representation

        Raises:
            SerializationError: If value cannot be serialized

        Example:
            ```python
            # Serialize embedding (list of floats)
            json_str = self._serialize([0.1, 0.2, 0.3])
            # Returns: "[0.1, 0.2, 0.3]"

            # Serialize metadata (dict)
            json_str = self._serialize({"tag": "python"})
            # Returns: '{"tag": "python"}'
            ```
        """

    def _deserialize(self, data: str) -> Any:
        """
        Deserialize JSON string to Python object.

        Internal method for converting JSON strings back to Python objects.

        Args:
            data: JSON string to deserialize

        Returns:
            Deserialized Python object

        Raises:
            DeserializationError: If JSON is invalid or corrupted

        Example:
            ```python
            # Deserialize embedding
            embedding = self._deserialize("[0.1, 0.2, 0.3]")
            # Returns: [0.1, 0.2, 0.3]

            # Deserialize metadata
            metadata = self._deserialize('{"tag": "python"}')
            # Returns: {"tag": "python"}
            ```
        """
```

## Dependencies

### Component Dependencies

- **None** - RedisClient is a standalone component with no internal component dependencies

### External Libraries

- **redis>=5.0.0** (for Redis client and connection pooling)
  - High-performance async-ready Redis client
  - Built-in connection pooling
  - SCAN cursor support

- **structlog>=23.2.0** (for structured logging)
  - Consistent logging across Zapomni
  - JSON-formatted logs

### Dependency Injection

RedisClient is instantiated directly by SemanticCache in `zapomni_core`. Configuration is provided via environment variables:

```python
# In zapomni_core/semantic_cache.py
from zapomni_db.redis_cache import RedisClient
import os

redis_client = RedisClient(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", "6380")),
    password=os.getenv("REDIS_PASSWORD"),
    ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "86400"))
)
```

## State Management

### Attributes

- `host: str` - Redis server host (immutable after init, lifetime: instance)
- `port: int` - Redis server port (immutable after init, lifetime: instance)
- `db: int` - Redis database number (immutable after init, lifetime: instance)
- `ttl_seconds: int` - Default TTL for cache entries (immutable after init, lifetime: instance)
- `password: Optional[str]` - Redis password (immutable after init, lifetime: instance)
- `client: redis.Redis` - Redis client instance (mutable, lifetime: instance, closed on close())
- `pool: redis.ConnectionPool` - Connection pool (mutable, lifetime: instance, closed on close())

### State Transitions

```
Uninitialized
    ↓
__init__() → Connected
    ↓
get/set/delete/scan/info → Operating (connection active)
    ↓
ConnectionError → Disconnected (retry or fail)
    ↓
Retry → Connected
    ↓
close() → Closed (final state)
```

### Thread Safety

- **Is this component thread-safe?** Yes, with caveats
- **Concurrency constraints:**
  - redis-py client is thread-safe with connection pooling
  - Multiple threads can call get/set/delete concurrently
  - Connection pool manages thread safety internally
  - Max concurrent operations limited by pool size (default: 10)
- **Synchronization mechanisms:**
  - redis-py uses internal locks for connection pool
  - No additional synchronization needed in RedisClient

## Public Methods (Detailed)

### Method 1: `__init__`

**Signature:**
```python
def __init__(
    self,
    host: str = "localhost",
    port: int = 6380,
    db: int = 0,
    ttl_seconds: int = 86400,
    password: Optional[str] = None,
    max_connections: int = 10,
    decode_responses: bool = False
) -> None
```

**Purpose:** Initialize Redis connection with pooling and validate configuration

**Parameters:**

- `host`: str
  - Description: Redis server hostname or IP address
  - Constraints: Non-empty string
  - Example: "localhost", "redis.example.com", "10.0.1.5"

- `port`: int
  - Description: Redis server port
  - Constraints: Valid port number (1-65535)
  - Example: 6380, 6379

- `db`: int
  - Description: Redis database number
  - Constraints: Integer 0-15 (Redis supports 16 databases)
  - Example: 0 (default), 1, 2

- `ttl_seconds`: int
  - Description: Default TTL for cache entries in seconds
  - Constraints: Positive integer
  - Example: 86400 (24h), 3600 (1h), 604800 (1 week)

- `password`: Optional[str]
  - Description: Redis password for authentication
  - Constraints: None if not required, string if required
  - Example: None, "secret123"

- `max_connections`: int
  - Description: Maximum connections in connection pool
  - Constraints: Positive integer (typically 5-50)
  - Example: 10 (default), 20

- `decode_responses`: bool
  - Description: Whether to decode byte responses to strings
  - Constraints: Boolean
  - Example: False (default, work with bytes), True (decode to str)

**Returns:**
- Type: `None`
- Side effect: Initializes `self.client` and `self.pool`

**Raises:**
- `ConnectionError`: When cannot connect to Redis server
- `ValueError`: When db not in range 0-15 or port invalid

**Preconditions:**
- Redis server must be running and accessible
- Network connectivity to Redis host

**Postconditions:**
- `self.client` is initialized and connected
- `self.pool` is created with max_connections
- Connection tested with PING command

**Algorithm Outline:**
```
1. Validate parameters (db in 0-15, port valid, ttl_seconds > 0)
2. Store configuration attributes (host, port, db, ttl_seconds, password)
3. Create ConnectionPool with max_connections
4. Create Redis client from pool
5. Test connection with PING command
6. Log successful connection
7. If connection fails, raise ConnectionError with details
```

**Edge Cases:**
1. Redis server not running → ConnectionError
2. Invalid password → AuthenticationError (subclass of ConnectionError)
3. db > 15 → ValueError
4. port < 1 or port > 65535 → ValueError
5. Network timeout → TimeoutError wrapped as ConnectionError

**Related Methods:**
- Calls: `redis.ConnectionPool()`, `redis.Redis()`, `client.ping()`
- Called by: SemanticCache initialization in `zapomni_core`

### Method 2: `get`

**Signature:**
```python
def get(self, key: str) -> Optional[Any]
```

**Purpose:** Retrieve cached value by key with automatic deserialization

**Parameters:**

- `key`: str
  - Description: Cache key identifier
  - Constraints: Non-empty string
  - Example: "embedding_abc123", "text_hash_xyz"

**Returns:**
- Type: `Optional[Any]`
- Success: Deserialized Python object (list, dict, etc.)
- Cache miss: None
- Expired: None

**Raises:**
- `ValueError`: If key is empty string
- `DeserializationError`: If cached data is corrupted JSON
- `CacheError`: If Redis GET operation fails

**Preconditions:**
- Client must be connected (initialized)
- Key must be valid string

**Postconditions:**
- No state change in cache
- Returns cached value or None
- Logs cache hit/miss

**Algorithm Outline:**
```
1. Validate key is non-empty
2. Call Redis GET command
3. If result is None → cache miss, return None
4. If result is bytes/string → deserialize from JSON
5. Return deserialized value
6. On error → log and raise appropriate exception
```

**Edge Cases:**
1. Key doesn't exist → return None (not an error)
2. Key expired → return None (TTL elapsed)
3. Corrupted JSON data → raise DeserializationError
4. Connection lost → raise CacheError, log, attempt reconnect
5. Empty key "" → raise ValueError

**Related Methods:**
- Calls: `client.get()`, `self._deserialize()`
- Called by: SemanticCache.get_embedding()

### Method 3: `set`

**Signature:**
```python
def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool
```

**Purpose:** Store value in cache with TTL and automatic serialization

**Parameters:**

- `key`: str
  - Description: Cache key identifier
  - Constraints: Non-empty string
  - Example: "embedding_abc123"

- `value`: Any
  - Description: Value to cache (must be JSON-serializable)
  - Constraints: Serializable to JSON (list, dict, str, int, float, bool, None)
  - Example: [0.1, 0.2, ...], {"tag": "python"}

- `ttl`: Optional[int]
  - Description: Time-to-live in seconds (overrides default)
  - Constraints: Positive integer if provided, None to use default
  - Example: None (use default 24h), 3600 (1h)

**Returns:**
- Type: `bool`
- Success: True (value stored)
- Failure: False (could not store)

**Raises:**
- `ValueError`: If key is empty or ttl is negative
- `SerializationError`: If value cannot be serialized to JSON
- `CacheError`: If Redis SET operation fails

**Preconditions:**
- Client connected
- Value is JSON-serializable

**Postconditions:**
- Value stored in Redis with TTL
- Key expires after TTL seconds
- Logs cache write

**Algorithm Outline:**
```
1. Validate key is non-empty
2. Determine TTL (use provided or default)
3. Validate TTL is positive
4. Serialize value to JSON string
5. Call Redis SET with EX (expiration)
6. Return True if SET successful
7. On error → log and raise
```

**Edge Cases:**
1. Empty key → ValueError
2. Negative TTL → ValueError
3. Non-serializable value (e.g., object with circular refs) → SerializationError
4. Redis out of memory → CacheError (eviction policy dependent)
5. Connection lost → CacheError, retry

**Related Methods:**
- Calls: `self._serialize()`, `client.set()`
- Called by: SemanticCache.set_embedding()

### Method 4: `delete`

**Signature:**
```python
def delete(self, key: str) -> bool
```

**Purpose:** Remove key from cache

**Parameters:**

- `key`: str
  - Description: Cache key to delete
  - Constraints: Non-empty string
  - Example: "embedding_abc123"

**Returns:**
- Type: `bool`
- True: Key existed and was deleted
- False: Key didn't exist

**Raises:**
- `ValueError`: If key is empty
- `CacheError`: If Redis DEL operation fails

**Preconditions:**
- Client connected

**Postconditions:**
- Key removed from Redis (if existed)
- Memory freed
- Logs deletion

**Algorithm Outline:**
```
1. Validate key is non-empty
2. Call Redis DEL command
3. Check return value (number of keys deleted)
4. Return True if count > 0, False otherwise
5. On error → log and raise
```

**Edge Cases:**
1. Empty key → ValueError
2. Key doesn't exist → return False (not an error)
3. Connection lost → CacheError

**Related Methods:**
- Calls: `client.delete()`
- Called by: SemanticCache.invalidate(), cache cleanup

### Method 5: `scan`

**Signature:**
```python
def scan(self, pattern: str = "*", count: int = 100) -> List[str]
```

**Purpose:** Scan and return keys matching pattern using cursor-based iteration

**Parameters:**

- `pattern`: str
  - Description: Key pattern with wildcards (* = any chars, ? = single char)
  - Constraints: Non-empty string
  - Example: "*", "embedding_*", "text_hash_abc*"

- `count`: int
  - Description: Hint for number of keys per iteration
  - Constraints: Positive integer
  - Example: 100 (default), 1000

**Returns:**
- Type: `List[str]`
- List of key names matching pattern

**Raises:**
- `ValueError`: If pattern is empty
- `CacheError`: If Redis SCAN operation fails

**Preconditions:**
- Client connected

**Postconditions:**
- No state change
- Returns list of matching keys
- Logs scan operation

**Algorithm Outline:**
```
1. Validate pattern is non-empty
2. Initialize cursor to 0
3. Loop: Call SCAN with cursor, pattern, count
4. Add returned keys to result list
5. Update cursor from SCAN result
6. If cursor == 0 → done, exit loop
7. Return accumulated key list
```

**Edge Cases:**
1. Empty pattern → ValueError
2. No matching keys → return empty list []
3. Very large key set → iterate efficiently (cursor-based, non-blocking)
4. Connection lost during scan → CacheError

**Related Methods:**
- Calls: `client.scan_iter()`
- Called by: Cache cleanup, statistics, admin tools

### Method 6: `info`

**Signature:**
```python
def info(self) -> Dict[str, Any]
```

**Purpose:** Get Redis cache statistics and configuration

**Returns:**
- Type: `Dict[str, Any]`
- Dictionary with cache statistics

**Raises:**
- `CacheError`: If Redis INFO command fails

**Preconditions:**
- Client connected

**Postconditions:**
- No state change
- Returns current statistics

**Algorithm Outline:**
```
1. Call Redis INFO command (section: memory and keyspace)
2. Parse INFO response
3. Extract: used_memory, total_keys, maxmemory, eviction_policy
4. Convert memory to MB (divide by 1024^2)
5. Build result dictionary
6. Return result
```

**Edge Cases:**
1. INFO command disabled (security) → CacheError
2. Connection lost → CacheError

**Related Methods:**
- Calls: `client.info()`
- Called by: Monitoring, statistics reporting

## Error Handling

### Exceptions Defined

```python
class CacheError(Exception):
    """Raised when Redis cache operation fails."""
    pass

class SerializationError(CacheError):
    """Raised when value cannot be serialized to JSON."""
    pass

class DeserializationError(CacheError):
    """Raised when cached data cannot be deserialized from JSON."""
    pass

class ConnectionError(CacheError):
    """Raised when cannot connect to Redis server."""
    pass
```

### Error Recovery

**Retry Strategy:**
- Connection errors: Retry up to 3 times with exponential backoff (1s, 2s, 4s)
- Serialization errors: No retry (fix at caller)
- Deserialization errors: No retry (data corruption, log and return None)

**Fallback Behavior:**
- If cache unavailable: SemanticCache degrades gracefully (continues without cache)
- If deserialize fails: Log warning, return None (cache miss)
- If set fails: Log error, continue (non-critical)

**Error Propagation:**
- All Redis errors wrapped in CacheError subclasses
- ConnectionError propagates to SemanticCache
- SerializationError propagates to caller (fix input)

## Usage Examples

### Basic Usage

```python
from zapomni_db.redis_cache import RedisClient

# Initialize client
cache = RedisClient(
    host="localhost",
    port=6380,
    ttl_seconds=3600  # 1 hour
)

# Store embedding
embedding = [0.1, 0.2, 0.3] * 256  # 768-dimensional
cache.set("embedding_abc123", embedding)

# Retrieve embedding
cached = cache.get("embedding_abc123")
if cached:
    print(f"Cache hit! Embedding has {len(cached)} dimensions")
else:
    print("Cache miss - need to generate embedding")

# Delete when done
cache.delete("embedding_abc123")

# Close connection
cache.close()
```

### Advanced Usage

```python
from zapomni_db.redis_cache import RedisClient
import hashlib

# Initialize with custom config
cache = RedisClient(
    host="redis.example.com",
    port=6379,
    password="secret",
    ttl_seconds=86400,  # 24 hours
    max_connections=20
)

# Cache embeddings with hash-based keys
def get_cached_embedding(text: str):
    """Get embedding from cache or return None."""
    text_hash = hashlib.sha256(text.encode()).hexdigest()
    cache_key = f"embedding_{text_hash}"
    return cache.get(cache_key)

def cache_embedding(text: str, embedding: list):
    """Store embedding in cache."""
    text_hash = hashlib.sha256(text.encode()).hexdigest()
    cache_key = f"embedding_{text_hash}"
    cache.set(cache_key, embedding, ttl=7200)  # 2 hours

# Use in embedding workflow
text = "Python is a programming language"
embedding = get_cached_embedding(text)

if embedding is None:
    # Cache miss - generate embedding
    embedding = generate_embedding(text)  # Call Ollama
    cache_embedding(text, embedding)
    print("Generated and cached embedding")
else:
    print("Using cached embedding")

# Cleanup old embeddings
old_keys = cache.scan("embedding_*")
print(f"Found {len(old_keys)} cached embeddings")

for key in old_keys[:10]:  # Delete first 10
    cache.delete(key)

# Get cache statistics
stats = cache.info()
print(f"Cache memory: {stats['used_memory_mb']:.2f} MB")
print(f"Total keys: {stats['total_keys']}")

# Close
cache.close()
```

## Testing Approach

### Unit Tests Required

1. **test_init_success()** - Normal initialization with defaults
2. **test_init_with_password()** - Initialization with authentication
3. **test_init_invalid_db()** - ValueError when db > 15
4. **test_init_connection_failure()** - ConnectionError when Redis down
5. **test_get_success()** - Get existing key returns value
6. **test_get_miss()** - Get non-existent key returns None
7. **test_get_empty_key_raises()** - ValueError on empty key
8. **test_set_success()** - Set value with default TTL
9. **test_set_custom_ttl()** - Set value with custom TTL
10. **test_set_empty_key_raises()** - ValueError on empty key
11. **test_set_non_serializable_raises()** - SerializationError on bad value
12. **test_delete_success()** - Delete existing key returns True
13. **test_delete_miss()** - Delete non-existent key returns False
14. **test_scan_all_keys()** - Scan with "*" returns all keys
15. **test_scan_pattern()** - Scan with pattern filters correctly
16. **test_info_success()** - Info returns valid statistics
17. **test_serialize_list()** - Serialize list to JSON
18. **test_deserialize_list()** - Deserialize JSON to list
19. **test_close_success()** - Close releases connections

### Mocking Strategy

- **Mock redis.Redis** - Return fake responses for get/set/delete
- **Mock redis.ConnectionPool** - Avoid real connections
- **Mock time.sleep** - Speed up retry tests
- **Mock json.dumps/loads** - Test serialization errors

### Integration Tests

1. **test_redis_roundtrip()** - Set then get, verify data integrity
2. **test_ttl_expiration()** - Set with TTL, wait, verify expires
3. **test_connection_recovery()** - Disconnect Redis, reconnect, verify works
4. **test_large_embedding_cache()** - Cache 768-dim embedding, retrieve
5. **test_concurrent_access()** - Multiple threads accessing cache

## Performance Considerations

### Time Complexity

- `get()`: O(1) - Redis GET operation
- `set()`: O(1) - Redis SET operation
- `delete()`: O(1) - Redis DEL operation
- `scan()`: O(N) where N = total keys (cursor-based, non-blocking)
- `info()`: O(1) - Redis INFO operation

### Space Complexity

- Memory per connection: ~50KB (Redis connection overhead)
- Memory for 10 connections: ~500KB
- Cached embedding (768 floats): ~6KB per entry
- 10,000 cached embeddings: ~60MB + Redis overhead

### Optimization Opportunities

1. **Connection pooling** - Reuse connections (already implemented)
2. **Pipelining** - Batch multiple operations (future enhancement)
3. **Compression** - Compress large embeddings (future, trade CPU for memory)
4. **Binary serialization** - Use msgpack instead of JSON (future, faster)

**Trade-offs:**
- JSON vs msgpack: JSON is human-readable, msgpack is faster/smaller
- Compression: Saves memory but increases CPU usage
- Connection pool size: More connections = higher concurrency but more memory

## References

- **Module spec:** /home/dev/zapomni/.spec-workflow/specs/level1/zapomni_db_module.md
- **Related components:** SemanticCache (zapomni_core), FalkorDBClient (zapomni_db)
- **External docs:**
  - redis-py documentation: https://redis-py.readthedocs.io/
  - Redis commands: https://redis.io/commands/
  - Redis best practices: https://redis.io/docs/manual/patterns/

---

**Document Status:** Draft v1.0
**Created:** 2025-11-23
**Author:** Goncharenko Anton aka alienxs2
**License:** MIT License
**Last Updated:** 2025-11-23

**Completeness:** 100% (all sections filled)
**API Completeness:** 100% (all public methods documented)
**Test Coverage:** 100% (19 test scenarios defined)
**Implementation Readiness:** High (detailed signatures, edge cases, examples)
