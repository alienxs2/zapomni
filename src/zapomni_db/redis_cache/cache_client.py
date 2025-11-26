"""
RedisClient - Lightweight wrapper for Redis operations.

Provides a simplified interface for storing and retrieving embeddings with TTL
(Time-To-Live) support, enabling fast cache lookups to reduce Ollama API calls.

Author: Goncharenko Anton aka alienxs2
TDD Implementation: Code written to pass tests from specifications.
"""

import json
import time
from abc import ABC
from typing import Any, Dict, List, Optional

import redis
import structlog
from redis import ConnectionPool, Redis

logger = structlog.get_logger(__name__)


# Custom Exceptions
class CacheError(Exception):
    """Raised when Redis cache operation fails."""

    pass


class SerializationError(CacheError):
    """Raised when value cannot be serialized to JSON."""

    pass


class DeserializationError(CacheError):
    """Raised when cached data cannot be deserialized from JSON."""

    pass


class RedisConnectionError(CacheError):
    """Raised when cannot connect to Redis server."""

    pass


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
        decode_responses: bool = False,
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
            RedisConnectionError: If cannot connect to Redis server
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
        # Validate parameters
        if db < 0 or db > 15:
            raise ValueError(f"db must be in range 0-15, got {db}")

        if port <= 0 or port > 65535:
            raise ValueError(f"port must be in range 1-65535, got {port}")

        if ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds must be positive, got {ttl_seconds}")

        # Store configuration
        self.host = host
        self.port = port
        self.db = db
        self.ttl_seconds = ttl_seconds
        self.password = password
        self.max_connections = max_connections
        self.decode_responses = decode_responses

        # Initialize state
        self._pool: Optional[ConnectionPool] = None
        self.client: Optional[Redis] = None
        self._closed = False
        self._logger = logger.bind(host=host, port=port, db=db)

        # Initialize connection
        try:
            self._init_connection()
            self._logger.info("redis_client_initialized")
        except redis.ConnectionError as e:
            self._logger.error("connection_failed", error=str(e))
            raise RedisConnectionError(f"Failed to connect to Redis: {e}")

    def _init_connection(self) -> None:
        """Initialize Redis connection with pooling and test connectivity."""
        try:
            # Create connection pool
            self._pool = ConnectionPool(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                max_connections=self.max_connections,
                socket_connect_timeout=self.CONNECTION_TIMEOUT,
                socket_keepalive=True,
                decode_responses=self.decode_responses,
            )

            # Create Redis client from pool
            self.client = Redis(connection_pool=self._pool)

            # Test connection
            self.client.ping()

        except redis.ConnectionError as e:
            self._logger.error("connection_init_failed", error=str(e))
            raise
        except Exception as e:
            self._logger.error("unexpected_error_during_init", error=str(e))
            raise redis.ConnectionError(f"Unexpected error during connection init: {e}")

    def ping(self) -> bool:
        """
        Test Redis connectivity with ping command.

        Returns:
            True if Redis server responds to PING command

        Raises:
            CacheError: If ping fails after retries
        """
        try:
            response = self.client.ping()
            self._logger.debug("ping_success", response=str(response))
            return response is True or response == b"PONG" or response == "PONG"
        except redis.RedisError as e:
            self._logger.error("redis_error_ping", error=str(e))
            raise CacheError(f"Redis PING failed: {e}")
        except Exception as e:
            self._logger.error("unexpected_error_ping", error=str(e))
            raise CacheError(f"Unexpected error during PING: {e}")

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache by key with automatic retry on connection errors.

        Retrieves cached value, deserializes from JSON, and returns.
        Returns None if key doesn't exist or has expired.

        Args:
            key: Cache key (string identifier)

        Returns:
            Cached value (deserialized from JSON) if found, None otherwise

        Raises:
            ValueError: If key is empty or invalid
            DeserializationError: If cached data is corrupted
            CacheError: If Redis operation fails after retries

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
        # Validate key
        if not key or not isinstance(key, str):
            raise ValueError("key must be non-empty string")

        retry_count = 0
        last_error = None

        while retry_count <= self.MAX_RETRIES:
            try:
                # Get value from Redis
                data = self.client.get(key)

                if data is None:
                    self._logger.debug("cache_miss", key=key)
                    return None

                # Deserialize from JSON
                result = self._deserialize(data)
                self._logger.debug("cache_hit", key=key)
                return result

            except DeserializationError:
                raise
            except redis.ConnectionError as e:
                retry_count += 1
                last_error = e
                if retry_count > self.MAX_RETRIES:
                    self._logger.error("get_failed_after_retries", key=key, retries=retry_count)
                    raise CacheError(
                        f"Redis GET operation failed after {self.MAX_RETRIES} retries: {e}"
                    )

                backoff_seconds = 2 ** (retry_count - 1)
                self._logger.warning(
                    "get_retry", key=key, attempt=retry_count, delay=backoff_seconds
                )
                time.sleep(backoff_seconds)

            except redis.RedisError as e:
                self._logger.error("redis_error_get", key=key, error=str(e))
                raise CacheError(f"Redis GET operation failed: {e}")
            except Exception as e:
                self._logger.error("unexpected_error_get", key=key, error=str(e))
                raise CacheError(f"Unexpected error during GET: {e}")

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache with TTL and automatic retry on connection errors.

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
            CacheError: If Redis write operation fails after retries

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
        # Validate key
        if not key or not isinstance(key, str):
            raise ValueError("key must be non-empty string")

        # Determine TTL
        effective_ttl = ttl if ttl is not None else self.ttl_seconds

        # Validate TTL
        if effective_ttl <= 0:
            raise ValueError(f"ttl must be positive, got {effective_ttl}")

        # Serialize value to JSON (do this before retrying, serialization errors don't get retried)
        try:
            serialized = self._serialize(value)
        except SerializationError:
            raise

        retry_count = 0

        while retry_count <= self.MAX_RETRIES:
            try:
                # Set in Redis with EX (expiration in seconds)
                result = self.client.set(key, serialized, ex=effective_ttl)

                if result:
                    self._logger.debug("cache_set", key=key, ttl=effective_ttl)
                    return True
                else:
                    self._logger.warning("cache_set_failed", key=key)
                    return False

            except redis.ConnectionError as e:
                retry_count += 1
                if retry_count > self.MAX_RETRIES:
                    self._logger.error("set_failed_after_retries", key=key, retries=retry_count)
                    raise CacheError(
                        f"Redis SET operation failed after {self.MAX_RETRIES} retries: {e}"
                    )

                backoff_seconds = 2 ** (retry_count - 1)
                self._logger.warning(
                    "set_retry", key=key, attempt=retry_count, delay=backoff_seconds
                )
                time.sleep(backoff_seconds)

            except redis.RedisError as e:
                self._logger.error("redis_error_set", key=key, error=str(e))
                raise CacheError(f"Redis SET operation failed: {e}")
            except Exception as e:
                self._logger.error("unexpected_error_set", key=key, error=str(e))
                raise CacheError(f"Unexpected error during SET: {e}")

    def delete(self, key: str) -> bool:
        """
        Delete key from cache with automatic retry on connection errors.

        Removes the specified key and its value from Redis.

        Args:
            key: Cache key to delete

        Returns:
            True if key was deleted, False if key didn't exist

        Raises:
            ValueError: If key is empty
            CacheError: If Redis delete operation fails after retries

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
        # Validate key
        if not key or not isinstance(key, str):
            raise ValueError("key must be non-empty string")

        retry_count = 0

        while retry_count <= self.MAX_RETRIES:
            try:
                # Delete key from Redis
                result = self.client.delete(key)

                if result > 0:
                    self._logger.debug("cache_deleted", key=key)
                    return True
                else:
                    self._logger.debug("cache_delete_miss", key=key)
                    return False

            except redis.ConnectionError as e:
                retry_count += 1
                if retry_count > self.MAX_RETRIES:
                    self._logger.error("delete_failed_after_retries", key=key, retries=retry_count)
                    raise CacheError(
                        f"Redis DELETE operation failed after {self.MAX_RETRIES} retries: {e}"
                    )

                backoff_seconds = 2 ** (retry_count - 1)
                self._logger.warning(
                    "delete_retry", key=key, attempt=retry_count, delay=backoff_seconds
                )
                time.sleep(backoff_seconds)

            except redis.RedisError as e:
                self._logger.error("redis_error_delete", key=key, error=str(e))
                raise CacheError(f"Redis DELETE operation failed: {e}")
            except Exception as e:
                self._logger.error("unexpected_error_delete", key=key, error=str(e))
                raise CacheError(f"Unexpected error during DELETE: {e}")

    def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.

        Args:
            key: Cache key to check

        Returns:
            True if key exists, False otherwise

        Raises:
            ValueError: If key is empty
            CacheError: If Redis operation fails
        """
        # Validate key
        if not key or not isinstance(key, str):
            raise ValueError("key must be non-empty string")

        try:
            result = self.client.exists(key)
            return result > 0

        except redis.RedisError as e:
            self._logger.error("redis_error_exists", key=key, error=str(e))
            raise CacheError(f"Redis EXISTS operation failed: {e}")
        except Exception as e:
            self._logger.error("unexpected_error_exists", key=key, error=str(e))
            raise CacheError(f"Unexpected error during EXISTS: {e}")

    def scan(self, pattern: str = "*", count: int = 100) -> List[str]:
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
        # Validate pattern
        if not pattern or not isinstance(pattern, str):
            raise ValueError("pattern must be non-empty string")

        try:
            # Use scan_iter for cursor-based iteration
            keys = []
            for key in self.client.scan_iter(match=pattern, count=count):
                keys.append(key if isinstance(key, str) else key.decode())

            self._logger.debug("scan_complete", pattern=pattern, count=len(keys))
            return keys

        except redis.RedisError as e:
            self._logger.error("redis_error_scan", pattern=pattern, error=str(e))
            raise CacheError(f"Redis SCAN operation failed: {e}")
        except Exception as e:
            self._logger.error("unexpected_error_scan", pattern=pattern, error=str(e))
            raise CacheError(f"Unexpected error during SCAN: {e}")

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
        try:
            # Get INFO from Redis
            info_dict = self.client.info()

            # Extract memory info
            used_memory = info_dict.get("used_memory", 0)
            used_memory_mb = used_memory / (1024 * 1024)
            used_memory_human = info_dict.get("used_memory_human", "0B")

            # Get keyspace info for current db
            keyspace_key = f"db{self.db}"
            keyspace_info = info_dict.get("keyspace", {})
            db_info = keyspace_info.get(keyspace_key, {})
            total_keys = db_info.get("keys", 0)

            # Get memory config
            maxmemory = info_dict.get("maxmemory", 0)
            maxmemory_mb = maxmemory / (1024 * 1024) if maxmemory > 0 else 0.0

            # Get eviction policy
            eviction_policy = info_dict.get("maxmemory_policy", "noeviction")

            result = {
                "used_memory_mb": used_memory_mb,
                "used_memory_human": used_memory_human,
                "total_keys": total_keys,
                "db_name": keyspace_key,
                "maxmemory_mb": maxmemory_mb,
                "eviction_policy": eviction_policy,
            }

            self._logger.debug("info_retrieved", total_keys=total_keys)
            return result

        except redis.RedisError as e:
            self._logger.error("redis_error_info", error=str(e))
            raise CacheError(f"Redis INFO command failed: {e}")
        except Exception as e:
            self._logger.error("unexpected_error_info", error=str(e))
            raise CacheError(f"Unexpected error during INFO: {e}")

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
        if self._closed:
            return

        try:
            if self.client:
                self.client.close()
            if self._pool:
                self._pool.disconnect()

            self._closed = True
            self._logger.info("connection_closed")

        except redis.RedisError as e:
            self._logger.error("redis_error_close", error=str(e))
            raise CacheError(f"Redis close operation failed: {e}")
        except Exception as e:
            self._logger.error("unexpected_error_close", error=str(e))
            raise CacheError(f"Unexpected error during close: {e}")

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
        try:
            return json.dumps(value)
        except (TypeError, ValueError) as e:
            self._logger.error("serialization_failed", error=str(e))
            raise SerializationError(f"Failed to serialize value to JSON: {e}")

    def _deserialize(self, data: Any) -> Any:
        """
        Deserialize JSON string to Python object.

        Internal method for converting JSON strings back to Python objects.

        Args:
            data: JSON string to deserialize (bytes or str)

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
        try:
            # Handle both bytes and str
            if isinstance(data, bytes):
                data = data.decode("utf-8")

            return json.loads(data)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as e:
            self._logger.error("deserialization_failed", error=str(e))
            raise DeserializationError(f"Failed to deserialize JSON: {e}")
