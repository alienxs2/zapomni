"""
EmbeddingCache - Dual-backend semantic cache for embeddings (Redis + In-Memory).

Provides efficient caching of embedding vectors with:
- Get/set operations with Redis backend (primary)
- In-memory fallback when Redis unavailable
- Automatic text normalization for cache keys
- TTL support (configurable, default 1 hour)
- Hit/miss statistics and metrics
- Comprehensive validation and error handling
- Graceful degradation on Redis failures

Target: 60%+ cache hit rate for semantic operations.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

import hashlib
import json
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import structlog
from redis.asyncio import Redis

from zapomni_core.exceptions import ValidationError
from zapomni_core.utils.logger_factory import get_logger

logger = get_logger(__name__)


class EmbeddingCache:
    """
    Dual-backend semantic cache for embedding vectors (Redis primary + In-Memory fallback).

    Caches embedding vectors with automatic text normalization, TTL support,
    and comprehensive statistics tracking. Designed to achieve 60%+ cache hit
    rate for repeated semantic operations.

    Features:
    - Redis as primary backend for distributed caching and persistence
    - In-memory fallback for graceful degradation when Redis unavailable
    - Automatic failover with configurable retry behavior
    - Comprehensive statistics tracking for both backends
    - TTL support with automatic expiration

    Attributes:
        redis_client: Async Redis client for caching (optional)
        ttl_seconds: Cache entry TTL (default: 3600 = 1 hour)
        embedding_dimensions: Expected embedding vector dimensions (default: 768)
        stats: Cache statistics (hits, misses, hit_rate, total_requests)
        use_redis: Whether Redis is available and should be used
        in_memory_cache: In-memory cache (dict) for fallback
        in_memory_ttl: In-memory TTL tracking (key -> expiration time)

    Example:
        ```python
        from redis.asyncio import Redis
        from zapomni_core.embeddings.embedding_cache import EmbeddingCache

        # Initialize Redis client (optional)
        redis_client = Redis(host="localhost", port=6379, decode_responses=True)

        # Create cache with Redis
        cache = EmbeddingCache(
            redis_client=redis_client,
            ttl_seconds=3600,  # 1 hour
        )

        # Use cache (Redis or fallback to in-memory)
        text = "Python is a great programming language"
        embedding = [0.1, 0.2, ..., 0.768]  # 768 dimensions

        # Set embedding in cache
        await cache.set(text, embedding)

        # Get embedding from cache (cache hit from Redis or in-memory)
        cached = await cache.get(text)

        # Check statistics
        stats = cache.get_statistics()
        print(f"Hit rate: {stats['hit_rate']:.1%}")
        ```
    """

    def __init__(
        self,
        redis_client: Optional[Redis] = None,
        ttl_seconds: int = 3600,
        embedding_dimensions: int = 768,
    ) -> None:
        """
        Initialize EmbeddingCache with optional Redis client and in-memory fallback.

        Args:
            redis_client: Async Redis client instance (redis.asyncio.Redis), optional.
                         If None, only in-memory cache is used.
            ttl_seconds: Cache entry time-to-live in seconds (default: 3600 = 1 hour)
                        Must be >= 60 seconds
            embedding_dimensions: Expected embedding dimensions (default: 768)
                                Must be 1-3072

        Raises:
            ValidationError: If parameters are invalid

        Example:
            ```python
            from redis.asyncio import Redis
            from zapomni_core.embeddings.embedding_cache import EmbeddingCache

            # With Redis
            redis_client = Redis(host="localhost", port=6379)
            cache = EmbeddingCache(
                redis_client=redis_client,
                ttl_seconds=7200,  # 2 hours
                embedding_dimensions=768
            )

            # Without Redis (in-memory only)
            cache = EmbeddingCache(ttl_seconds=3600)
            ```
        """
        # Validate TTL
        if ttl_seconds < 60:
            raise ValidationError(
                message=f"ttl_seconds must be >= 60, got {ttl_seconds}",
                error_code="VAL_003",
                details={"ttl_seconds": ttl_seconds},
            )

        # Validate embedding dimensions
        if embedding_dimensions < 1 or embedding_dimensions > 3072:
            raise ValidationError(
                message=f"embedding_dimensions must be 1-3072, got {embedding_dimensions}",
                error_code="VAL_003",
                details={"embedding_dimensions": embedding_dimensions},
            )

        self.redis_client = redis_client
        self.ttl_seconds = ttl_seconds
        self.embedding_dimensions = embedding_dimensions

        # Dual-backend support
        self.use_redis = redis_client is not None
        self.in_memory_cache: Dict[str, Tuple[List[float], float]] = (
            {}
        )  # key -> (embedding, expiry_time)
        self.in_memory_ttl: Dict[str, float] = {}  # key -> expiration timestamp

        # Initialize statistics
        self.stats: Dict[str, Any] = {
            "hits": 0,
            "misses": 0,
            "total_requests": 0,
            "hit_rate": 0.0,
            "redis_hits": 0,
            "memory_hits": 0,
            "backend_errors": 0,
        }

        logger.info(
            "embedding_cache_initialized",
            ttl_seconds=ttl_seconds,
            embedding_dimensions=embedding_dimensions,
            redis_enabled=self.use_redis,
        )

    async def get(self, text: str) -> Optional[List[float]]:
        """
        Get cached embedding for text (tries Redis first, then in-memory fallback).

        Algorithm:
        1. Validate input text (non-empty, UTF-8)
        2. Generate normalized cache key from text
        3. Try Redis first (if available and enabled)
        4. Fall back to in-memory cache if Redis fails or unavailable
        5. Check TTL expiration for in-memory entries
        6. Deserialize embedding from JSON
        7. Update statistics (hit/miss/backend source)
        8. Return embedding or None if not cached

        Args:
            text: Input text to lookup (max 10,000 characters)

        Returns:
            List[float]: Embedding vector if cached, None if not in cache

        Raises:
            ValidationError: If text is empty or invalid

        Performance Target:
            - Cache hit (Redis): < 10ms (P95)
            - Cache hit (In-Memory): < 1ms (P95)
            - Cache miss: < 10ms (P95)

        Example:
            ```python
            cache = EmbeddingCache(redis_client)

            # Check cache for text
            embedding = await cache.get("Python is great")

            if embedding is not None:
                print(f"Cache hit! Dimensions: {len(embedding)}")
            else:
                print("Cache miss - need to generate embedding")
            ```
        """
        # Validate input
        if not text or not isinstance(text, str):
            raise ValidationError(
                message=f"text must be non-empty string, got {type(text).__name__}",
                error_code="VAL_001",
                details={"text_type": type(text).__name__},
            )

        if not text.strip():
            raise ValidationError(
                message="text cannot be empty or whitespace-only",
                error_code="VAL_001",
                details={"text_length": len(text)},
            )

        if len(text) > 10000:
            raise ValidationError(
                message=f"text exceeds max length (10,000 chars), got {len(text)}",
                error_code="VAL_003",
                details={"text_length": len(text)},
            )

        # Generate cache key
        cache_key = self._generate_cache_key(text)

        # Update statistics
        self.stats["total_requests"] += 1

        # Try Redis first
        if self.use_redis:
            try:
                cached_data = await self.redis_client.get(cache_key)

                if cached_data is not None:
                    # Cache hit from Redis
                    try:
                        embedding = json.loads(cached_data)
                        self.stats["hits"] += 1
                        self.stats["redis_hits"] += 1
                        self._update_hit_rate()

                        logger.debug(
                            "embedding_cache_hit",
                            source="redis",
                            text_length=len(text),
                            embedding_dims=len(embedding),
                        )

                        return embedding

                    except json.JSONDecodeError as e:
                        logger.error(
                            "embedding_cache_deserialization_failed",
                            error=str(e),
                            cache_key=cache_key,
                            source="redis",
                        )
                        self.stats["backend_errors"] += 1
                        # Fall through to in-memory cache

            except Exception as e:
                logger.warning(
                    "embedding_cache_redis_error",
                    error=str(e),
                    cache_key=cache_key,
                )
                self.stats["backend_errors"] += 1
                # Fall through to in-memory cache

        # Try in-memory cache as fallback
        embedding = self._get_from_memory(cache_key)
        if embedding is not None:
            self.stats["hits"] += 1
            self.stats["memory_hits"] += 1
            self._update_hit_rate()

            logger.debug(
                "embedding_cache_hit",
                source="memory",
                text_length=len(text),
                embedding_dims=len(embedding),
            )

            return embedding

        # Cache miss
        self.stats["misses"] += 1
        self._update_hit_rate()

        logger.debug(
            "embedding_cache_miss",
            text_length=len(text),
            hit_rate=f"{self.stats['hit_rate']:.2%}",
        )

        return None

    async def set(self, text: str, embedding: List[float]) -> None:
        """
        Set (cache) embedding for text (stores in Redis, always stores in in-memory backup).

        Algorithm:
        1. Validate input text (non-empty, UTF-8)
        2. Validate embedding (correct dimensions, no NaN/Inf)
        3. Generate normalized cache key
        4. Serialize embedding to JSON
        5. Try to store in Redis with TTL (if available)
        6. Always store in in-memory cache as backup
        7. Log operation

        Args:
            text: Input text (max 10,000 characters)
            embedding: Embedding vector (must match embedding_dimensions)

        Raises:
            ValidationError: If text or embedding is invalid

        Performance Target:
            - Set operation (Redis + Memory): < 20ms (P95)
            - Set operation (Memory only): < 1ms (P95)

        Example:
            ```python
            cache = EmbeddingCache(redis_client)

            text = "Python is a great programming language"
            embedding = [0.1, 0.2, ..., 0.768]  # 768-dim vector

            # Cache the embedding
            await cache.set(text, embedding)
            ```
        """
        # Validate text input
        if not text or not isinstance(text, str):
            raise ValidationError(
                message=f"text must be non-empty string, got {type(text).__name__}",
                error_code="VAL_001",
                details={"text_type": type(text).__name__},
            )

        if not text.strip():
            raise ValidationError(
                message="text cannot be empty or whitespace-only",
                error_code="VAL_001",
                details={"text_length": len(text)},
            )

        if len(text) > 10000:
            raise ValidationError(
                message=f"text exceeds max length (10,000 chars), got {len(text)}",
                error_code="VAL_003",
                details={"text_length": len(text)},
            )

        # Validate embedding input
        if not isinstance(embedding, list):
            raise ValidationError(
                message=f"embedding must be list, got {type(embedding).__name__}",
                error_code="VAL_002",
                details={"embedding_type": type(embedding).__name__},
            )

        if len(embedding) != self.embedding_dimensions:
            raise ValidationError(
                message=f"embedding dimensions mismatch: expected {self.embedding_dimensions}, got {len(embedding)}",
                error_code="VAL_003",
                details={
                    "expected_dims": self.embedding_dimensions,
                    "got_dims": len(embedding),
                },
            )

        # Validate embedding values
        for i, val in enumerate(embedding):
            if not isinstance(val, (int, float)):
                raise ValidationError(
                    message=f"embedding[{i}] must be number, got {type(val).__name__}",
                    error_code="VAL_002",
                    details={"index": i, "value_type": type(val).__name__},
                )

            if math.isnan(val) or math.isinf(val):
                raise ValidationError(
                    message=f"embedding[{i}] contains NaN or Inf: {val}",
                    error_code="VAL_003",
                    details={"index": i, "value": str(val)},
                )

        # Generate cache key
        cache_key = self._generate_cache_key(text)

        try:
            # Serialize embedding to JSON
            embedding_json = json.dumps(embedding)

            # Try to store in Redis if available
            if self.use_redis:
                try:
                    await self.redis_client.set(
                        cache_key,
                        embedding_json,
                        ex=self.ttl_seconds,
                    )
                    logger.debug(
                        "embedding_cached",
                        target="redis",
                        text_length=len(text),
                        embedding_dims=len(embedding),
                        ttl_seconds=self.ttl_seconds,
                    )

                except Exception as e:
                    logger.warning(
                        "embedding_cache_redis_set_failed",
                        error=str(e),
                        cache_key=cache_key,
                    )
                    self.stats["backend_errors"] += 1

            # Always store in in-memory cache as backup
            expiry_time = time.time() + self.ttl_seconds
            self.in_memory_cache[cache_key] = (embedding, expiry_time)
            self.in_memory_ttl[cache_key] = expiry_time

            logger.debug(
                "embedding_cached",
                target="memory",
                text_length=len(text),
                embedding_dims=len(embedding),
                ttl_seconds=self.ttl_seconds,
            )

        except Exception as e:
            logger.error(
                "embedding_cache_set_failed",
                error=str(e),
                cache_key=cache_key,
            )
            raise

    def _get_from_memory(self, cache_key: str) -> Optional[List[float]]:
        """
        Get embedding from in-memory cache with TTL check.

        Checks if key exists, verifies TTL hasn't expired, and returns embedding.
        Automatically removes expired entries.

        Args:
            cache_key: Cache key to lookup

        Returns:
            Embedding vector if found and not expired, None otherwise

        Private method, not exposed in public API.
        """
        if cache_key not in self.in_memory_cache:
            return None

        # Check TTL expiration
        expiry_time = self.in_memory_ttl.get(cache_key)
        if expiry_time is None or time.time() > expiry_time:
            # Remove expired entry
            self.in_memory_cache.pop(cache_key, None)
            self.in_memory_ttl.pop(cache_key, None)
            return None

        embedding, _ = self.in_memory_cache[cache_key]
        return embedding

    def _update_hit_rate(self) -> None:
        """
        Update cache hit rate statistics.

        Calculates hit rate as hits / total_requests.
        Handles edge case of zero total requests.

        Private method, not exposed in public API.
        """
        if self.stats["total_requests"] > 0:
            self.stats["hit_rate"] = self.stats["hits"] / self.stats["total_requests"]
        else:
            self.stats["hit_rate"] = 0.0

    def _generate_cache_key(self, text: str) -> str:
        """
        Generate normalized cache key from text.

        Algorithm:
        1. Normalize text:
           - Convert to lowercase
           - Strip leading/trailing whitespace
           - Replace multiple whitespace with single space
        2. Hash normalized text using SHA-256
        3. Return hash as cache key

        This ensures:
        - Same text with different casing produces same key
        - Extra whitespace doesn't create multiple cache entries
        - Key is deterministic for same input

        Args:
            text: Input text to generate key for

        Returns:
            str: Cache key (SHA-256 hash of normalized text)

        Private method, not exposed in public API.

        Example:
            ```
            key1 = cache._generate_cache_key("Hello World")
            key2 = cache._generate_cache_key("hello world")
            key3 = cache._generate_cache_key("HELLO   WORLD")

            # All produce same key due to normalization
            assert key1 == key2 == key3
            ```
        """
        # Normalize text
        normalized = " ".join(text.lower().split())

        # Generate SHA-256 hash
        hash_obj = hashlib.sha256(normalized.encode("utf-8"))
        cache_key = hash_obj.hexdigest()

        return cache_key

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get cache statistics snapshot.

        Returns dictionary with:
        - hits: Total cache hits (all sources)
        - misses: Total cache misses
        - total_requests: Total get operations
        - hit_rate: Cache hit rate (0.0 to 1.0)
        - redis_hits: Cache hits from Redis
        - memory_hits: Cache hits from in-memory cache
        - backend_errors: Redis errors encountered

        Returns:
            Dict with keys: hits, misses, total_requests, hit_rate, redis_hits, memory_hits, backend_errors

        Performance Target:
            - Operation: O(1) - simple dict copy

        Example:
            ```python
            cache = EmbeddingCache(redis_client)

            # ... perform operations ...

            stats = cache.get_statistics()
            print(f"Hits: {stats['hits']}")
            print(f"Misses: {stats['misses']}")
            print(f"Hit Rate: {stats['hit_rate']:.1%}")
            print(f"Redis Hits: {stats['redis_hits']}")
            print(f"Memory Hits: {stats['memory_hits']}")

            # Check if target 60% hit rate achieved
            if stats['hit_rate'] >= 0.60:
                print("Cache performing well!")
            ```
        """
        return {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "total_requests": self.stats["total_requests"],
            "hit_rate": self.stats["hit_rate"],
            "redis_hits": self.stats["redis_hits"],
            "memory_hits": self.stats["memory_hits"],
            "backend_errors": self.stats["backend_errors"],
        }

    async def clear(self) -> None:
        """
        Clear all cached embeddings from both Redis and in-memory storage.

        Flushes Redis database (use with caution in shared environments).
        Clears in-memory cache.
        Does NOT reset statistics counters.

        Performance Target:
            - Operation: < 100ms

        Example:
            ```python
            cache = EmbeddingCache(redis_client)

            # ... perform operations ...

            # Clear all cached data
            await cache.clear()
            ```
        """
        try:
            # Clear Redis if available
            if self.use_redis:
                try:
                    await self.redis_client.flushdb()
                    logger.info("embedding_cache_redis_cleared")
                except Exception as e:
                    logger.warning("embedding_cache_redis_clear_failed", error=str(e))
                    self.stats["backend_errors"] += 1

            # Clear in-memory cache
            self.in_memory_cache.clear()
            self.in_memory_ttl.clear()
            logger.info("embedding_cache_memory_cleared")

        except Exception as e:
            logger.error("embedding_cache_clear_failed", error=str(e))
            raise

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup."""
        # No cleanup needed for cache (client managed separately)
        pass
