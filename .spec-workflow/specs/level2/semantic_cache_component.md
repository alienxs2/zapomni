# SemanticCache - Component Specification

**Level:** 2 (Component)
**Module:** zapomni_core
**Author:** Goncharenko Anton aka alienxs2
**Status:** Draft
**Version:** 1.0
**Created:** 2025-11-23

---

## Overview

### Purpose

SemanticCache is a **performance optimization component** that caches embeddings and search results to avoid redundant computation. It dramatically improves system responsiveness by serving frequently requested embeddings from cache instead of regenerating them via Ollama.

This component is critical for achieving the **60-68% cache hit rate** target specified in the product requirements, which translates to:
- 3-4x faster embedding retrieval (< 5ms cache hit vs. 150-200ms Ollama generation)
- Reduced Ollama API load (fewer requests)
- Better user experience (faster search and add operations)

### Responsibilities

1. **Embedding Caching:** Store and retrieve embeddings by content hash
2. **Search Result Caching:** Cache search results for repeated queries (Phase 2 enhancement)
3. **Cache Invalidation:** Remove stale or outdated entries by pattern or TTL
4. **Hit Rate Tracking:** Monitor cache effectiveness and provide statistics
5. **Memory Management:** Enforce LRU eviction policy and size limits

### Position in Module

SemanticCache sits between the EmbeddingService and OllamaAPI:

```
┌─────────────────────────────────────────────────────┐
│           MemoryProcessor (Core Pipeline)           │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Input → Chunker → [SemanticCache] → Ollama         │
│                         ↓                            │
│                    Redis Backend                     │
│                                                      │
└─────────────────────────────────────────────────────┘

Flow with Cache:
1. Chunk text: "Python is great" → hash: abc123...
2. Check cache: cache.get_embedding(hash) → HIT or MISS
3. If HIT: Return cached embedding (< 5ms)
4. If MISS: Call Ollama → Cache result → Return embedding
```

---

## Class Definition

### Class Diagram

```
┌─────────────────────────────────────────────────┐
│            SemanticCache                        │
├─────────────────────────────────────────────────┤
│ - redis_client: RedisClient                     │
│ - ttl: int (default: 3600)                      │
│ - max_entries: int (default: 10000)             │
│ - hit_count: int                                │
│ - miss_count: int                               │
│ - _key_prefix: str = "zapomni:cache:"           │
├─────────────────────────────────────────────────┤
│ + __init__(redis_client, ttl, max_entries)      │
│ + get_embedding(text: str) -> Optional[List]    │
│ + set_embedding(text: str, embedding: List)     │
│ + get_search_results(query_hash: str) -> ...    │
│ + set_search_results(query_hash: str, ...)      │
│ + invalidate(pattern: str) -> int               │
│ + get_stats() -> CacheStats                     │
│ + clear() -> None                               │
│ - _normalize_text(text: str) -> str             │
│ - _compute_hash(text: str) -> str               │
└─────────────────────────────────────────────────┘
```

### Full Class Signature

```python
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import hashlib
import json

@dataclass
class CacheStats:
    """
    Cache statistics.

    Attributes:
        total_hits: Number of cache hits
        total_misses: Number of cache misses
        hit_rate: Hit rate percentage (0-1)
        total_entries: Current number of cached entries
        size_mb: Approximate cache size in MB
        avg_hit_latency_ms: Average cache hit latency
        avg_miss_latency_ms: Average cache miss latency
    """
    total_hits: int
    total_misses: int
    hit_rate: float
    total_entries: int
    size_mb: float
    avg_hit_latency_ms: float
    avg_miss_latency_ms: float


class SemanticCache:
    """
    Redis-backed cache for embeddings and search results.

    Implements LRU eviction, TTL-based expiration, and performance monitoring.
    Designed for high-throughput read operations with sub-5ms cache hits.

    Key Design Principles:
    - **Text Normalization:** Strip whitespace, lowercase, consistent encoding
    - **Content Hashing:** SHA256 for collision-free keys
    - **TTL Management:** 1-hour default (3600s), configurable
    - **LRU Eviction:** Redis handles automatically with MAXMEMORY-POLICY=allkeys-lru
    - **Size Limits:** max_entries prevents unbounded growth

    Attributes:
        redis_client: Redis client for storage backend
        ttl: Time-to-live for cache entries (seconds)
        max_entries: Maximum number of entries (LRU eviction after)
        hit_count: Running count of cache hits
        miss_count: Running count of cache misses
        _key_prefix: Redis key namespace prefix

    Example:
        ```python
        from zapomni_db.redis import RedisClient
        from zapomni_core.cache import SemanticCache

        # Initialize cache
        redis = RedisClient(host="localhost", port=6379)
        cache = SemanticCache(
            redis_client=redis,
            ttl=3600,  # 1 hour
            max_entries=10000
        )

        # Cache embedding
        text = "Python is a programming language"
        embedding = [0.1, 0.2, 0.3, ...]  # 768-dim vector

        cache.set_embedding(text, embedding)

        # Retrieve from cache
        cached_embedding = cache.get_embedding(text)
        if cached_embedding:
            print("Cache HIT")
        else:
            print("Cache MISS")

        # Get stats
        stats = cache.get_stats()
        print(f"Hit rate: {stats.hit_rate:.2%}")
        ```
    """

    def __init__(
        self,
        redis_client: "RedisClient",
        ttl: int = 3600,
        max_entries: int = 10000
    ) -> None:
        """
        Initialize semantic cache.

        Args:
            redis_client: Redis client instance (from zapomni_db.redis)
            ttl: Time-to-live for cache entries in seconds (default: 3600 = 1 hour)
            max_entries: Maximum cache entries before LRU eviction (default: 10,000)

        Raises:
            ValueError: If ttl <= 0 or max_entries <= 0
            ConnectionError: If redis_client cannot connect

        Performance Notes:
            - Cache initialization is fast (< 1ms)
            - Redis connection is lazy (connected on first operation)
            - max_entries enforced via ZREMRANGEBYRANK on insertion
        """
        if ttl <= 0:
            raise ValueError("ttl must be positive")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")

        self.redis_client = redis_client
        self.ttl = ttl
        self.max_entries = max_entries
        self.hit_count = 0
        self.miss_count = 0
        self._key_prefix = "zapomni:cache:"

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Retrieve cached embedding for text.

        Workflow:
        1. Normalize text (strip, lowercase)
        2. Compute SHA256 hash
        3. Query Redis: GET zapomni:cache:embedding:{hash}
        4. If found: Deserialize JSON → increment hit_count → return
        5. If not found: increment miss_count → return None

        Args:
            text: Input text (will be normalized)

        Returns:
            Cached embedding (List of floats, 768 dimensions for nomic-embed-text)
            or None if not cached

        Raises:
            ValueError: If text is empty after normalization
            RedisError: If Redis operation fails (logged, returns None)

        Performance Target:
            - Cache HIT: < 5ms (P95)
            - Cache MISS: < 2ms (P95)

        Example:
            ```python
            embedding = cache.get_embedding("Python is great")
            if embedding:
                print(f"HIT: {len(embedding)}-dim vector")
            else:
                print("MISS: Generate with Ollama")
            ```
        """

    def set_embedding(
        self,
        text: str,
        embedding: List[float]
    ) -> None:
        """
        Cache embedding for text.

        Workflow:
        1. Normalize text
        2. Compute SHA256 hash
        3. Serialize embedding to JSON
        4. SET zapomni:cache:embedding:{hash} {json_data} EX {ttl}
        5. ZADD zapomni:cache:lru {timestamp} {hash} (for LRU tracking)
        6. Enforce max_entries: ZREMRANGEBYRANK (remove oldest if over limit)

        Args:
            text: Input text (will be normalized)
            embedding: Embedding vector (768 floats for nomic-embed-text)

        Raises:
            ValueError: If text empty or embedding invalid (wrong dimension)
            RedisError: If Redis operation fails

        Performance Target:
            - Execution time: < 10ms (P95)

        Example:
            ```python
            embedding = ollama.embed("Python is great")  # 768 dimensions
            cache.set_embedding("Python is great", embedding)
            # Now cached for 1 hour (default TTL)
            ```
        """

    def get_search_results(
        self,
        query_hash: str
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve cached search results for query.

        Phase 2 feature for caching full search results (not just embeddings).

        Args:
            query_hash: SHA256 hash of search query + filters

        Returns:
            Cached search results (list of SearchResult dicts)
            or None if not cached

        Raises:
            RedisError: If Redis operation fails

        Performance Target:
            - Cache HIT: < 10ms

        Example:
            ```python
            query_hash = cache._compute_hash(
                f"{query}:{filters}:{search_mode}"
            )
            results = cache.get_search_results(query_hash)
            if results:
                return [SearchResult(**r) for r in results]
            ```
        """

    def set_search_results(
        self,
        query_hash: str,
        results: List[Dict[str, Any]]
    ) -> None:
        """
        Cache search results for query.

        Phase 2 feature.

        Args:
            query_hash: SHA256 hash of search query + filters
            results: Search results as list of dicts

        Raises:
            RedisError: If Redis operation fails

        Performance Target:
            - Execution time: < 15ms
        """

    def invalidate(self, pattern: str) -> int:
        """
        Invalidate cache entries matching pattern.

        Uses Redis SCAN + DEL for safe deletion without blocking.

        Args:
            pattern: Redis key pattern (e.g., "embedding:*", "search:*")

        Returns:
            Number of entries deleted

        Raises:
            RedisError: If Redis operation fails

        Performance Target:
            - Deletion rate: ~1000 keys/sec

        Example:
            ```python
            # Invalidate all embedding cache
            deleted = cache.invalidate("embedding:*")
            print(f"Deleted {deleted} entries")

            # Invalidate search cache only
            deleted = cache.invalidate("search:*")
            ```
        """

    def get_stats(self) -> CacheStats:
        """
        Get cache statistics.

        Returns:
            CacheStats object with:
            - total_hits, total_misses, hit_rate
            - total_entries (via DBSIZE)
            - size_mb (via INFO MEMORY)
            - avg_hit_latency_ms, avg_miss_latency_ms

        Performance Target:
            - Execution time: < 50ms

        Example:
            ```python
            stats = cache.get_stats()
            print(f"Hit rate: {stats.hit_rate:.1%}")
            print(f"Size: {stats.size_mb:.2f} MB")
            print(f"Entries: {stats.total_entries}")
            ```
        """

    def clear(self) -> None:
        """
        Clear all cache entries.

        Deletes all keys with prefix "zapomni:cache:".
        Resets hit/miss counters.

        Raises:
            RedisError: If Redis operation fails

        Performance Notes:
            - Uses SCAN + DEL for safe deletion
            - May take 1-2 seconds for large caches (10K+ entries)

        Example:
            ```python
            cache.clear()  # Fresh start
            stats = cache.get_stats()
            assert stats.total_entries == 0
            ```
        """

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for consistent hashing.

        Transformations:
        1. Strip leading/trailing whitespace
        2. Convert to lowercase
        3. Collapse multiple spaces to single space
        4. Remove zero-width characters

        Args:
            text: Raw input text

        Returns:
            Normalized text

        Example:
            ```python
            norm = cache._normalize_text("  Python   IS  Great  ")
            # Returns: "python is great"
            ```
        """

    def _compute_hash(self, text: str) -> str:
        """
        Compute SHA256 hash of text.

        Args:
            text: Normalized text

        Returns:
            Hex-encoded SHA256 hash (64 characters)

        Example:
            ```python
            hash_key = cache._compute_hash("python is great")
            # Returns: "a3f2b1c4d5e6..." (64 chars)
            ```
        """
```

---

## Dependencies

### Component Dependencies

**From zapomni_db:**
- `RedisClient` - Redis connection manager and client wrapper
  - Used for: All cache storage operations (GET, SET, DEL, SCAN)
  - Interface: Standard Redis commands via redis-py

**From zapomni_core:**
- None directly (SemanticCache is standalone)
- **Used by:** `OllamaEmbedder`, `SearchEngine` (they inject SemanticCache)

### External Libraries

- `redis>=5.0.0` - Python Redis client (via zapomni_db.redis)
  - Used for: All Redis operations
  - Configuration: Connection pooling, timeouts, retry logic handled by RedisClient

- `hashlib` (stdlib) - Cryptographic hashing
  - Used for: SHA256 hash computation

- `json` (stdlib) - JSON serialization
  - Used for: Embedding and result serialization/deserialization

### Dependency Injection

SemanticCache receives `RedisClient` via constructor injection:

```python
# Good: Dependency injection ✅
redis_client = RedisClient(host="localhost", port=6379)
cache = SemanticCache(redis_client=redis_client)

# Bad: Hardcoded dependency ❌
class SemanticCache:
    def __init__(self):
        self.redis = redis.Redis(host="localhost")  # Don't do this
```

**Rationale:**
- Testability: Easy to mock RedisClient in unit tests
- Flexibility: Can swap Redis implementations (cluster, sentinel)
- Configuration: Connection settings managed externally

---

## State Management

### Attributes

**Instance State:**
- `redis_client: RedisClient` - Redis connection (lifetime: instance lifetime)
- `ttl: int` - Cache TTL in seconds (immutable after init)
- `max_entries: int` - Maximum cache size (immutable after init)
- `hit_count: int` - Cache hits counter (mutable, incremented on get_embedding HIT)
- `miss_count: int` - Cache misses counter (mutable, incremented on get_embedding MISS)
- `_key_prefix: str` - Redis key namespace (constant: "zapomni:cache:")

**Redis State (External):**
- Embedding cache: `zapomni:cache:embedding:{hash}` → JSON embedding
- Search cache: `zapomni:cache:search:{hash}` → JSON search results
- LRU tracker: `zapomni:cache:lru` → Sorted set (timestamp → hash)

### State Transitions

```
INITIAL STATE (Empty Cache)
    ↓
[User adds memory] → get_embedding(text) → MISS
    ↓
    miss_count += 1
    ↓
[Ollama generates embedding] → set_embedding(text, embedding)
    ↓
    Redis: SET zapomni:cache:embedding:{hash} {json}
    ZADD zapomni:cache:lru {timestamp} {hash}
    ↓
STATE: Cache has 1 entry

[Same text requested again] → get_embedding(text) → HIT
    ↓
    hit_count += 1
    ↓
    Return cached embedding (< 5ms)
    ↓
STATE: hit_count=1, miss_count=1, hit_rate=50%

[TTL expires (1 hour later)]
    ↓
    Redis auto-deletes key (TTL expired)
    ↓
STATE: Cache entry removed

[Max entries exceeded]
    ↓
    ZREMRANGEBYRANK removes oldest entries (LRU)
    ↓
STATE: Cache size maintained at max_entries
```

### Thread Safety

**Is this component thread-safe?** **YES**

**Concurrency Guarantees:**
- `redis_client` is thread-safe (uses connection pooling)
- Redis operations are atomic (SET, GET, DEL)
- `hit_count` and `miss_count` are **not thread-safe** (but acceptable for statistics)
  - Race condition possible: Two threads increment simultaneously → one increment lost
  - Impact: Minimal (stats slightly inaccurate)
  - Fix (if needed): Use `threading.Lock` or Redis INCR command

**Thread-Safe Usage:**
```python
# Multiple threads can safely call get_embedding
cache = SemanticCache(redis_client)

# Thread 1
embedding1 = cache.get_embedding("text A")

# Thread 2 (concurrent)
embedding2 = cache.get_embedding("text B")

# Both safe - no shared mutable state accessed
```

**Not Thread-Safe (but acceptable):**
```python
# hit_count/miss_count may be slightly inaccurate
# if 1000 concurrent requests happen simultaneously
stats = cache.get_stats()
# stats.hit_rate might be 65.3% instead of 65.5% (acceptable)
```

---

## Public Methods (Detailed)

### Method 1: `get_embedding`

**Signature:**
```python
def get_embedding(self, text: str) -> Optional[List[float]]
```

**Purpose:** Retrieve cached embedding for text to avoid regenerating via Ollama.

**Parameters:**

- `text`: str
  - Description: Input text to retrieve embedding for
  - Constraints:
    - Must not be empty after normalization (strip + lowercase)
    - No max length (any length accepted)
    - Will be normalized before hashing
  - Example: `"Python is a programming language"`

**Returns:**
- Type: `Optional[List[float]]`
- Success Case:
  - Returns: `[0.123, 0.456, ..., 0.789]` (768 floats for nomic-embed-text)
  - Condition: Embedding found in cache and not expired
- Miss Case:
  - Returns: `None`
  - Condition: Embedding not cached or expired

**Raises:**
- `ValueError`: When text is empty after normalization
- `RedisError`: When Redis connection fails (logged, returns None gracefully)

**Preconditions:**
- SemanticCache initialized with valid RedisClient
- Redis server reachable

**Postconditions:**
- If HIT: `hit_count` incremented by 1
- If MISS: `miss_count` incremented by 1
- Redis connection state unchanged (no writes)

**Algorithm Outline:**
```
1. Validate text (not empty after strip)
2. Normalize text: text.strip().lower()
3. Compute hash: SHA256(normalized_text).hexdigest()
4. Build Redis key: f"{_key_prefix}embedding:{hash}"
5. Query Redis: GET key
6. If found:
   a. Deserialize JSON to List[float]
   b. Increment hit_count
   c. Return embedding
7. If not found:
   a. Increment miss_count
   b. Return None
```

**Edge Cases:**

1. **Empty text:** `text = ""`
   - Behavior: Raise `ValueError("Text cannot be empty")`
   - Test: `test_get_embedding_empty_raises()`

2. **Whitespace-only text:** `text = "   "`
   - Behavior: After normalize → empty → Raise `ValueError`
   - Test: `test_get_embedding_whitespace_raises()`

3. **Text with special chars:** `text = "Python™ is great!"`
   - Behavior: Normalize to `"python™ is great!"` → hash → query
   - Test: `test_get_embedding_special_chars()`

4. **Very long text (1MB):**
   - Behavior: Hash successfully (SHA256 handles any size), query Redis
   - Test: `test_get_embedding_large_text()`

5. **Redis connection lost:**
   - Behavior: Log error, return None (graceful degradation)
   - Test: `test_get_embedding_redis_error_returns_none()`

6. **Cached but expired (TTL passed):**
   - Behavior: Redis returns None (auto-deleted), miss_count incremented
   - Test: `test_get_embedding_expired_returns_none()`

**Related Methods:**
- Calls: `_normalize_text()`, `_compute_hash()`
- Called by: `OllamaEmbedder.embed()`, `MemoryProcessor.add_memory()`

---

### Method 2: `set_embedding`

**Signature:**
```python
def set_embedding(self, text: str, embedding: List[float]) -> None
```

**Purpose:** Cache embedding for text to speed up future retrievals.

**Parameters:**

- `text`: str
  - Description: Input text corresponding to embedding
  - Constraints: Same as `get_embedding()` (non-empty after normalization)
  - Example: `"Python is great"`

- `embedding`: List[float]
  - Description: Embedding vector from Ollama or sentence-transformers
  - Constraints:
    - Length must be 768 (nomic-embed-text) or 384 (sentence-transformers)
    - All elements must be finite floats (no NaN, no Inf)
  - Example: `[0.1, 0.2, ..., 0.768]`

**Returns:**
- Type: `None` (void method)

**Raises:**
- `ValueError`: When text empty or embedding invalid (wrong dimension)
- `RedisError`: When Redis operation fails

**Preconditions:**
- SemanticCache initialized
- `embedding` is valid (768 or 384 dimensions, all finite)

**Postconditions:**
- Embedding stored in Redis with TTL
- LRU tracker updated (timestamp added)
- If cache exceeds `max_entries`, oldest entry evicted

**Algorithm Outline:**
```
1. Validate text (not empty)
2. Validate embedding:
   a. Length in [384, 768] (common dimensions)
   b. All elements finite (math.isfinite)
3. Normalize text
4. Compute hash
5. Serialize embedding to JSON: json.dumps(embedding)
6. Redis pipeline (atomic):
   a. SET zapomni:cache:embedding:{hash} {json} EX {ttl}
   b. ZADD zapomni:cache:lru {timestamp} {hash}
   c. ZCARD lru_key → if count > max_entries:
      ZREMRANGEBYRANK lru_key 0 (count - max_entries - 1)
7. Execute pipeline
```

**Edge Cases:**

1. **Empty text:**
   - Behavior: Raise `ValueError("Text cannot be empty")`

2. **Embedding wrong dimension (e.g., 512):**
   - Behavior: Raise `ValueError("Embedding dimension must be 384 or 768")`

3. **Embedding contains NaN:**
   - Behavior: Raise `ValueError("Embedding contains invalid values (NaN/Inf)")`

4. **Cache at max capacity:**
   - Behavior: Evict oldest entry via ZREMRANGEBYRANK, then insert new
   - Test: `test_set_embedding_lru_eviction()`

5. **Redis full (out of memory):**
   - Behavior: Redis raises OOM error → logged, raise RedisError
   - Test: `test_set_embedding_redis_oom()`

**Related Methods:**
- Calls: `_normalize_text()`, `_compute_hash()`
- Called by: `OllamaEmbedder.embed()` (after generating embedding)

---

### Method 3: `invalidate`

**Signature:**
```python
def invalidate(self, pattern: str) -> int
```

**Purpose:** Remove cache entries matching pattern (for cache freshness management).

**Parameters:**

- `pattern`: str
  - Description: Redis key pattern for matching
  - Constraints:
    - Must be valid Redis pattern syntax (`*`, `?`, `[abc]`)
    - Relative to `_key_prefix` (e.g., "embedding:*" → "zapomni:cache:embedding:*")
  - Examples:
    - `"embedding:*"` - All embedding cache
    - `"search:*"` - All search cache
    - `"*"` - Entire cache

**Returns:**
- Type: `int`
- Value: Number of keys deleted

**Raises:**
- `RedisError`: If Redis SCAN or DEL fails

**Preconditions:**
- SemanticCache initialized
- Redis reachable

**Postconditions:**
- Matching keys deleted from Redis
- LRU tracker updated (removed hashes deleted)

**Algorithm Outline:**
```
1. Build full pattern: f"{_key_prefix}{pattern}"
2. SCAN Redis for matching keys:
   a. SCAN cursor=0, MATCH=full_pattern, COUNT=1000
   b. Collect keys
   c. Repeat until cursor=0 (full scan)
3. For each batch of keys:
   a. DEL key1 key2 ... key100 (batch delete)
4. Update LRU tracker:
   a. Extract hashes from deleted keys
   b. ZREM zapomni:cache:lru {hash1} {hash2} ...
5. Return total deleted count
```

**Edge Cases:**

1. **No matching keys:**
   - Behavior: SCAN returns empty, return 0
   - Test: `test_invalidate_no_matches()`

2. **Pattern matches entire cache:**
   - Behavior: Delete all keys (similar to `clear()`), return count
   - Test: `test_invalidate_all()`

3. **Invalid pattern syntax:**
   - Behavior: Redis handles gracefully (matches nothing), return 0
   - Test: `test_invalidate_invalid_pattern()`

4. **Redis connection lost during SCAN:**
   - Behavior: Raise RedisError (partial delete possible)
   - Test: `test_invalidate_redis_error()`

**Related Methods:**
- Calls: Redis `SCAN`, `DEL`, `ZREM`
- Called by: `MemoryProcessor.update_memory()` (when memory updated, invalidate old embeddings)

---

### Method 4: `get_stats`

**Signature:**
```python
def get_stats(self) -> CacheStats
```

**Purpose:** Provide cache performance metrics for monitoring and optimization.

**Returns:**
- Type: `CacheStats`
- Fields:
  - `total_hits`: Current `hit_count` value
  - `total_misses`: Current `miss_count` value
  - `hit_rate`: `hits / (hits + misses)` (0-1 range)
  - `total_entries`: Redis `DBSIZE` (approximate)
  - `size_mb`: Redis `INFO memory` → used_memory / 1MB
  - `avg_hit_latency_ms`: Average time for cache hits (tracked internally)
  - `avg_miss_latency_ms`: Average time for cache misses

**Raises:**
- `RedisError`: If Redis INFO command fails

**Preconditions:**
- SemanticCache initialized

**Postconditions:**
- No state changes (read-only operation)

**Algorithm Outline:**
```
1. Compute hit_rate: hit_count / (hit_count + miss_count) if total > 0 else 0.0
2. Query Redis:
   a. DBSIZE → total_entries (all keys, not just cache)
   b. INFO memory → parse "used_memory" field → convert to MB
3. Compute latencies:
   a. avg_hit_latency_ms: SUM(hit_latencies) / hit_count (if tracked)
   b. avg_miss_latency_ms: SUM(miss_latencies) / miss_count
4. Construct CacheStats object
5. Return
```

**Edge Cases:**

1. **No cache operations yet (hit_count=0, miss_count=0):**
   - Behavior: hit_rate = 0.0, latencies = 0.0
   - Test: `test_get_stats_empty_cache()`

2. **Only hits (miss_count=0):**
   - Behavior: hit_rate = 1.0
   - Test: `test_get_stats_100_percent_hit_rate()`

3. **Only misses (hit_count=0):**
   - Behavior: hit_rate = 0.0
   - Test: `test_get_stats_zero_hit_rate()`

4. **Redis DBSIZE unavailable:**
   - Behavior: total_entries = 0 (graceful fallback), log warning
   - Test: `test_get_stats_redis_error()`

**Related Methods:**
- Called by: `MemoryProcessor.get_stats()` (to include cache stats in system stats)

---

### Method 5: `clear`

**Signature:**
```python
def clear(self) -> None
```

**Purpose:** Clear all cache entries (for testing or fresh start).

**Returns:**
- Type: `None`

**Raises:**
- `RedisError`: If Redis operations fail

**Preconditions:**
- SemanticCache initialized

**Postconditions:**
- All keys with prefix `zapomni:cache:` deleted
- `hit_count` and `miss_count` reset to 0
- LRU tracker cleared

**Algorithm Outline:**
```
1. SCAN Redis for all keys matching "zapomni:cache:*"
2. Delete in batches (1000 keys per DEL command)
3. Reset counters:
   a. self.hit_count = 0
   b. self.miss_count = 0
4. Delete LRU tracker: DEL zapomni:cache:lru
```

**Edge Cases:**

1. **Cache already empty:**
   - Behavior: No-op, return silently
   - Test: `test_clear_empty_cache()`

2. **Partial delete (Redis fails mid-operation):**
   - Behavior: Raise RedisError, some keys deleted (partial state)
   - Test: `test_clear_redis_error()`

**Related Methods:**
- Calls: `invalidate("*")` (internally)
- Called by: Unit tests, admin operations

---

## Error Handling

### Exceptions Defined

```python
# zapomni_core/exceptions.py

class CacheError(ZapomniCoreError):
    """Base exception for cache operations."""
    pass

class CacheConnectionError(CacheError):
    """Redis connection failed."""
    pass

class CacheInvalidationError(CacheError):
    """Cache invalidation failed."""
    pass
```

### Error Recovery

**Transient Errors (Retry):**
- Redis connection timeout → Retry 3x with exponential backoff (100ms, 200ms, 400ms)
- Redis server busy → Retry 2x

**Permanent Errors (Fail Fast):**
- Invalid embedding dimension → Raise `ValueError` immediately
- Empty text → Raise `ValueError` immediately

**Graceful Degradation:**
- Redis unavailable → Log error, return `None` from `get_embedding()` (allow system to continue without cache)
- Redis OOM → Log error, skip caching (don't block memory addition)

### Error Propagation

**What exceptions bubble up:**
- `ValueError`: Input validation failures (caller should fix input)
- `RedisError`: Infrastructure failures (caller should handle or escalate)

**What exceptions are caught and logged:**
- Transient Redis errors (after retry exhausted) → Log warning, return None

---

## Usage Examples

### Basic Usage

```python
from zapomni_db.redis import RedisClient
from zapomni_core.cache import SemanticCache

# Initialize Redis client
redis = RedisClient(host="localhost", port=6379, db=0)

# Initialize cache with 1-hour TTL and 10K max entries
cache = SemanticCache(
    redis_client=redis,
    ttl=3600,
    max_entries=10000
)

# Example 1: Cache embedding
text = "Python is a programming language"
embedding = [0.1, 0.2, 0.3, ...]  # 768-dim vector from Ollama

cache.set_embedding(text, embedding)
print("Embedding cached")

# Example 2: Retrieve from cache
cached = cache.get_embedding(text)
if cached:
    print(f"Cache HIT: {len(cached)}-dim vector")
else:
    print("Cache MISS: Generate with Ollama")

# Example 3: Get stats
stats = cache.get_stats()
print(f"Hit rate: {stats.hit_rate:.1%}")
print(f"Total entries: {stats.total_entries}")
print(f"Cache size: {stats.size_mb:.2f} MB")
```

### Advanced Usage (Integration with OllamaEmbedder)

```python
class OllamaEmbedder:
    """Embedding service with cache integration."""

    def __init__(
        self,
        ollama_host: str,
        cache: Optional[SemanticCache] = None
    ):
        self.ollama_host = ollama_host
        self.cache = cache

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings with caching."""
        embeddings = []

        for text in texts:
            # Check cache first
            if self.cache:
                cached_embedding = self.cache.get_embedding(text)
                if cached_embedding:
                    embeddings.append(cached_embedding)
                    continue  # Cache HIT

            # Cache MISS - generate with Ollama
            embedding = await self._call_ollama(text)
            embeddings.append(embedding)

            # Cache result
            if self.cache:
                self.cache.set_embedding(text, embedding)

        return embeddings

    async def _call_ollama(self, text: str) -> List[float]:
        """Call Ollama API to generate embedding."""
        # Implementation: httpx POST to Ollama /api/embeddings
        ...


# Usage
cache = SemanticCache(redis_client=redis)
embedder = OllamaEmbedder(
    ollama_host="http://localhost:11434",
    cache=cache
)

texts = ["Python is great", "Python is great", "JavaScript too"]
embeddings = await embedder.embed(texts)
# First "Python is great" → MISS → Ollama
# Second "Python is great" → HIT → < 5ms
# "JavaScript too" → MISS → Ollama

stats = cache.get_stats()
print(f"Hit rate: {stats.hit_rate:.1%}")  # 33.3% (1 hit out of 3)
```

### Advanced Usage (Cache Invalidation)

```python
# Scenario: User updates a memory, need to invalidate related cache

# Invalidate specific embedding
cache.invalidate(f"embedding:{hash_of_old_text}")

# Invalidate all search results (after re-indexing)
deleted = cache.invalidate("search:*")
print(f"Deleted {deleted} search cache entries")

# Clear entire cache (for testing)
cache.clear()
stats = cache.get_stats()
assert stats.total_entries == 0
```

---

## Testing Approach

### Unit Tests Required

**Happy Path Tests:**
1. `test_init_success()` - Normal initialization with valid params
2. `test_get_embedding_hit()` - Retrieve cached embedding successfully
3. `test_get_embedding_miss()` - Retrieve non-cached embedding returns None
4. `test_set_embedding_success()` - Cache embedding successfully
5. `test_get_stats_basic()` - Get stats with hits and misses
6. `test_invalidate_pattern()` - Invalidate with pattern works
7. `test_clear_cache()` - Clear all entries

**Error Tests:**
8. `test_init_invalid_ttl_raises()` - ttl <= 0 raises ValueError
9. `test_init_invalid_max_entries_raises()` - max_entries <= 0 raises ValueError
10. `test_get_embedding_empty_raises()` - Empty text raises ValueError
11. `test_set_embedding_invalid_dimension_raises()` - Wrong embedding size raises ValueError
12. `test_set_embedding_nan_raises()` - Embedding with NaN raises ValueError
13. `test_get_embedding_redis_error_returns_none()` - Redis failure returns None gracefully

**Edge Case Tests:**
14. `test_get_embedding_expired()` - Expired entry returns None
15. `test_set_embedding_lru_eviction()` - Exceeding max_entries evicts oldest
16. `test_get_stats_empty_cache()` - Stats with no operations (hit_rate=0)
17. `test_get_stats_100_percent_hit_rate()` - Only cache hits
18. `test_invalidate_no_matches()` - Pattern matches nothing returns 0
19. `test_normalize_text_whitespace()` - Whitespace normalization works
20. `test_compute_hash_collision()` - Different texts produce different hashes

**Performance Tests:**
21. `test_get_embedding_latency()` - Cache hit < 5ms (P95)
22. `test_set_embedding_latency()` - Cache set < 10ms (P95)
23. `test_lru_eviction_performance()` - Evicting 1000 entries < 100ms

### Mocking Strategy

**Mock RedisClient:**
```python
import pytest
from unittest.mock import Mock, MagicMock

@pytest.fixture
def mock_redis():
    """Mock Redis client for unit tests."""
    redis = Mock()
    redis.get = MagicMock(return_value=None)  # Default: cache miss
    redis.set = MagicMock(return_value=True)
    redis.delete = MagicMock(return_value=1)
    redis.scan_iter = MagicMock(return_value=[])
    redis.dbsize = MagicMock(return_value=0)
    redis.info = MagicMock(return_value={"used_memory": 1024})
    return redis

def test_get_embedding_hit(mock_redis):
    """Test cache hit returns embedding."""
    # Setup mock
    embedding_json = json.dumps([0.1, 0.2, 0.3])
    mock_redis.get.return_value = embedding_json

    # Test
    cache = SemanticCache(redis_client=mock_redis)
    result = cache.get_embedding("test")

    # Assert
    assert result == [0.1, 0.2, 0.3]
    assert cache.hit_count == 1
    assert cache.miss_count == 0
```

**Don't Mock (Use Real):**
- `hashlib.sha256` - Fast, no network, no reason to mock
- `json.dumps/loads` - Standard library, deterministic

### Integration Tests

**With Real Redis (Docker Compose):**
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_cache_integration_with_redis(redis_client):
    """Test SemanticCache with real Redis instance."""
    cache = SemanticCache(redis_client=redis_client, ttl=60)

    # Add embedding
    text = "Integration test"
    embedding = [0.1] * 768
    cache.set_embedding(text, embedding)

    # Retrieve from cache
    result = cache.get_embedding(text)
    assert result == embedding

    # Verify in Redis
    key = f"zapomni:cache:embedding:{cache._compute_hash('integration test')}"
    raw = redis_client.get(key)
    assert raw is not None

    # Wait for TTL expiration
    await asyncio.sleep(61)
    result = cache.get_embedding(text)
    assert result is None  # Expired
```

---

## Performance Considerations

### Time Complexity

**get_embedding():**
- Normalize text: O(n) where n = text length
- Compute hash: O(n)
- Redis GET: O(1) average
- JSON deserialize: O(m) where m = embedding dimension (768)
- **Total: O(n + m) ≈ O(n)** for typical text sizes

**set_embedding():**
- Normalize + hash: O(n)
- JSON serialize: O(m)
- Redis SET: O(1)
- Redis ZADD: O(log k) where k = LRU set size
- ZREMRANGEBYRANK (if LRU full): O(log k)
- **Total: O(n + m + log k)**

**invalidate():**
- Redis SCAN: O(N) where N = total keys in DB
- Redis DEL (batch): O(p) where p = matching keys
- **Total: O(N + p)**

**get_stats():**
- DBSIZE: O(1)
- INFO memory: O(1)
- **Total: O(1)**

### Space Complexity

**Memory Usage:**
- Per embedding entry:
  - Key: ~80 bytes (prefix + hash)
  - Value: ~6KB (768 floats × 8 bytes)
  - Total: ~6.1KB per entry

**Cache Size Estimate:**
- 1,000 entries: ~6.1 MB
- 10,000 entries: ~61 MB
- 100,000 entries: ~610 MB

**Redis Configuration:**
- Set `maxmemory` to 2GB (allows ~300K embeddings)
- Set `maxmemory-policy` to `allkeys-lru` (automatic eviction)

### Optimization Opportunities

**Current Optimizations:**
- ✅ JSON serialization (fast, built-in)
- ✅ SHA256 hashing (hardware-accelerated)
- ✅ Redis pipelining (batch operations)
- ✅ LRU eviction (Redis built-in)

**Future Optimizations (if needed):**
- MessagePack instead of JSON (25% smaller, faster)
- Compression (zlib) for embeddings (50% smaller, slight CPU cost)
- Redis Cluster for horizontal scaling (multi-node)
- Bloom filter for quick miss detection (before Redis query)

**Trade-offs:**
- JSON vs. MessagePack: JSON is human-readable (debugging), MessagePack is faster
- Compression vs. Speed: Compression saves memory but adds 2-3ms latency
- LRU vs. LFU: LRU simpler, LFU better for skewed access patterns

---

## References

### Module Spec
- [zapomni_core_module.md](../level1/zapomni_core_module.md) - Parent module specification

### Related Components
- [OllamaEmbedder](./ollama_embedder_component.md) - Uses SemanticCache for embedding caching (future spec)
- [SearchEngine](./search_engine_component.md) - Uses SemanticCache for search result caching (future spec)

### External Documentation
- Redis Commands: https://redis.io/commands/
- Redis LRU Eviction: https://redis.io/docs/manual/eviction/
- SHA256 Performance: https://en.wikipedia.org/wiki/SHA-2

### Research References
- Cache Hit Rate Analysis: "Characterizing Database Queries from a Large-scale Workload" (Li et al., 2019)
- LRU vs. LFU: "ARC: A Self-Tuning, Low Overhead Replacement Cache" (Megiddo & Modha, 2003)

---

## Appendix: Performance Benchmarks (Expected)

### Target Metrics (Phase 2)

| Operation | Latency (P50) | Latency (P95) | Throughput |
|-----------|---------------|---------------|------------|
| get_embedding (hit) | 2ms | 5ms | 5000 ops/sec |
| get_embedding (miss) | 1ms | 2ms | 10000 ops/sec |
| set_embedding | 5ms | 10ms | 2000 ops/sec |
| invalidate (100 keys) | 20ms | 50ms | - |
| get_stats | 5ms | 10ms | 1000 ops/sec |

### Cache Hit Rate Target

**Goal:** 60-68% hit rate for typical workloads

**Rationale:**
- Research shows 60-70% hit rate is common for LRU caches with 1-hour TTL
- Zapomni workload: Users query similar topics repeatedly (high temporal locality)
- Example: User researches "Python" → generates 10 queries → 8-9 hit cache

**Measurement:**
```python
stats = cache.get_stats()
print(f"Hit rate: {stats.hit_rate:.1%}")
# Target: 60-68% after 1000+ operations
```

---

## Appendix: Redis Configuration

### Recommended Redis Settings

```conf
# /etc/redis/redis.conf

# Memory management
maxmemory 2gb
maxmemory-policy allkeys-lru  # LRU eviction

# Persistence (optional - cache can be ephemeral)
save ""  # Disable RDB snapshots (cache is transient)
appendonly no  # Disable AOF (cache is transient)

# Performance
tcp-backlog 511
timeout 300
tcp-keepalive 60
```

### Docker Compose (Testing)

```yaml
# docker-compose.yml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: >
      redis-server
      --maxmemory 2gb
      --maxmemory-policy allkeys-lru
      --save ""
      --appendonly no
    volumes:
      - redis_data:/data

volumes:
  redis_data:
```

---

**Document Status:** Draft v1.0
**Created:** 2025-11-23
**Author:** Goncharenko Anton aka alienxs2
**Copyright:** Copyright (c) 2025 Goncharenko Anton aka alienxs2
**License:** MIT License

**Total Sections:** 15
**Total Methods:** 8 (5 public, 2 private, 1 clear)
**Test Scenarios:** 23+
**Ready for Review:** Yes ✅
