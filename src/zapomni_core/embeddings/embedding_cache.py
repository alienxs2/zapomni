"""
EmbeddingCache - Redis-backed semantic cache for embeddings.

Provides efficient caching of embedding vectors with:
- Get/set operations with Redis backend
- Automatic text normalization for cache keys
- TTL support (configurable, default 1 hour)
- Hit/miss statistics and metrics
- Comprehensive validation and error handling

Target: 60%+ cache hit rate for semantic operations.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

import hashlib
import json
import math
from typing import List, Optional, Dict, Any
from redis.asyncio import Redis

import structlog

from zapomni_core.exceptions import ValidationError
from zapomni_core.utils.logger_factory import get_logger

logger = get_logger(__name__)


class EmbeddingCache:
    """
    Redis-backed semantic cache for embedding vectors.

    Caches embedding vectors with automatic text normalization, TTL support,
    and comprehensive statistics tracking. Designed to achieve 60%+ cache hit
    rate for repeated semantic operations.

    Attributes:
        redis_client: Async Redis client for caching
        ttl_seconds: Cache entry TTL (default: 3600 = 1 hour)
        embedding_dimensions: Expected embedding vector dimensions (default: 768)
        stats: Cache statistics (hits, misses, hit_rate, total_requests)

    Example:
        ```python
        from redis.asyncio import Redis
        from zapomni_core.embeddings.embedding_cache import EmbeddingCache

        # Initialize Redis client
        redis_client = Redis(host="localhost", port=6379, decode_responses=True)

        # Create cache
        cache = EmbeddingCache(
            redis_client=redis_client,
            ttl_seconds=3600,  # 1 hour
        )

        # Use cache
        text = "Python is a great programming language"
        embedding = [0.1, 0.2, ..., 0.768]  # 768 dimensions

        # Set embedding in cache
        await cache.set(text, embedding)

        # Get embedding from cache (cache hit)
        cached = await cache.get(text)

        # Check statistics
        stats = cache.get_statistics()
        print(f"Hit rate: {stats['hit_rate']:.1%}")
        ```
    """

    def __init__(
        self,
        redis_client: Redis,
        ttl_seconds: int = 3600,
        embedding_dimensions: int = 768,
    ) -> None:
        """
        Initialize EmbeddingCache with Redis client.

        Args:
            redis_client: Async Redis client instance (redis.asyncio.Redis)
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

            redis_client = Redis(host="localhost", port=6379)
            cache = EmbeddingCache(
                redis_client=redis_client,
                ttl_seconds=7200,  # 2 hours
                embedding_dimensions=768
            )
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

        # Initialize statistics
        self.stats: Dict[str, Any] = {
            "hits": 0,
            "misses": 0,
            "total_requests": 0,
            "hit_rate": 0.0,
        }

        logger.info(
            "embedding_cache_initialized",
            ttl_seconds=ttl_seconds,
            embedding_dimensions=embedding_dimensions,
        )

    async def get(self, text: str) -> Optional[List[float]]:
        """
        Get cached embedding for text.

        Algorithm:
        1. Validate input text (non-empty, UTF-8)
        2. Generate normalized cache key from text
        3. Query Redis for cached embedding
        4. Deserialize embedding from JSON
        5. Update statistics (hit/miss)
        6. Return embedding or None if not cached

        Args:
            text: Input text to lookup (max 10,000 characters)

        Returns:
            List[float]: Embedding vector if cached, None if not in cache

        Raises:
            ValidationError: If text is empty or invalid

        Performance Target:
            - Cache hit: < 10ms (P95) - Redis query
            - Cache miss: < 10ms (P95) - Redis query

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

        try:
            # Query Redis
            cached_data = await self.redis_client.get(cache_key)

            # Update statistics
            self.stats["total_requests"] += 1

            if cached_data is not None:
                # Cache hit
                try:
                    embedding = json.loads(cached_data)
                    self.stats["hits"] += 1

                    # Update hit rate
                    self.stats["hit_rate"] = (
                        self.stats["hits"] / self.stats["total_requests"]
                    )

                    logger.debug(
                        "embedding_cache_hit",
                        text_length=len(text),
                        embedding_dims=len(embedding),
                    )

                    return embedding

                except json.JSONDecodeError as e:
                    logger.error(
                        "embedding_cache_deserialization_failed",
                        error=str(e),
                        cache_key=cache_key,
                    )
                    raise Exception(
                        f"Failed to deserialize cached embedding: {str(e)}"
                    )
            else:
                # Cache miss
                self.stats["misses"] += 1

                # Update hit rate
                self.stats["hit_rate"] = (
                    self.stats["hits"] / self.stats["total_requests"]
                )

                logger.debug(
                    "embedding_cache_miss",
                    text_length=len(text),
                    hit_rate=f"{self.stats['hit_rate']:.2%}",
                )

                return None

        except json.JSONDecodeError as e:
            raise Exception(f"Failed to deserialize cached embedding: {str(e)}")
        except Exception as e:
            logger.error(
                "embedding_cache_get_failed",
                error=str(e),
                cache_key=cache_key,
            )
            raise

    async def set(self, text: str, embedding: List[float]) -> None:
        """
        Set (cache) embedding for text.

        Algorithm:
        1. Validate input text (non-empty, UTF-8)
        2. Validate embedding (correct dimensions, no NaN/Inf)
        3. Generate normalized cache key
        4. Serialize embedding to JSON
        5. Store in Redis with TTL
        6. Log operation

        Args:
            text: Input text (max 10,000 characters)
            embedding: Embedding vector (must be 768-dimensional)

        Raises:
            ValidationError: If text or embedding is invalid

        Performance Target:
            - Set operation: < 20ms (P95) - Redis write + serialization

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

            # Store in Redis with TTL
            await self.redis_client.set(
                cache_key,
                embedding_json,
                ex=self.ttl_seconds,
            )

            logger.debug(
                "embedding_cached",
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
        - hits: Total cache hits
        - misses: Total cache misses
        - total_requests: Total get operations
        - hit_rate: Cache hit rate (0.0 to 1.0)

        Returns:
            Dict with keys: hits, misses, total_requests, hit_rate

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
        }

    async def clear(self) -> None:
        """
        Clear all cached embeddings.

        Flushes Redis database (use with caution in shared environments).
        Resets statistics counters.

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
            await self.redis_client.flushdb()
            logger.info("embedding_cache_cleared")

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
