"""
Unit tests for RedisClient.

TDD Approach: Tests written FIRST based on specifications.
Tests use unittest.mock to mock Redis without requiring a running server.

Test Coverage:
- __init__: Connection initialization, validation, pooling
- get(): Cache hits/misses, deserialization, error handling, retries
- set(): TTL handling, serialization, error handling, retries
- delete(): Key deletion, non-existent keys, error handling, retries
- exists(): Key existence checks, error handling
- ping(): Connectivity tests
- scan(): Pattern matching, cursor-based iteration
- info(): Statistics retrieval
- close(): Connection cleanup
- _serialize(): JSON serialization
- _deserialize(): JSON deserialization

Total: 15+ comprehensive tests
"""

import pytest
import json
import time
from unittest.mock import Mock, MagicMock, patch, call
from typing import Any

from zapomni_db.redis_cache.cache_client import (
    RedisClient,
    CacheError,
    SerializationError,
    DeserializationError,
    RedisConnectionError
)
import redis


# ============================================================================
# __init__ TESTS
# ============================================================================

class TestRedisClientInit:
    """Test RedisClient initialization."""

    @patch('zapomni_db.redis_cache.cache_client.ConnectionPool')
    @patch('zapomni_db.redis_cache.cache_client.Redis')
    def test_init_defaults(self, mock_redis_class, mock_pool_class):
        """Test initialization with default parameters."""
        mock_redis_instance = MagicMock()
        mock_redis_class.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True

        client = RedisClient()

        assert client.host == "localhost"
        assert client.port == 6380
        assert client.db == 0
        assert client.ttl_seconds == 86400
        assert client.max_connections == 10

    @patch('zapomni_db.redis_cache.cache_client.ConnectionPool')
    @patch('zapomni_db.redis_cache.cache_client.Redis')
    def test_init_custom_params(self, mock_redis_class, mock_pool_class):
        """Test initialization with custom parameters."""
        mock_redis_instance = MagicMock()
        mock_redis_class.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True

        client = RedisClient(
            host="custom-host",
            port=6379,
            db=5,
            ttl_seconds=3600,
            max_connections=20
        )

        assert client.host == "custom-host"
        assert client.port == 6379
        assert client.db == 5
        assert client.ttl_seconds == 3600
        assert client.max_connections == 20

    def test_init_invalid_db_raises(self):
        """Test that db > 15 raises ValueError."""
        with pytest.raises(ValueError, match="db must be in range 0-15"):
            RedisClient(db=16)

    def test_init_negative_db_raises(self):
        """Test that db < 0 raises ValueError."""
        with pytest.raises(ValueError, match="db must be in range 0-15"):
            RedisClient(db=-1)

    def test_init_invalid_port_raises(self):
        """Test that invalid port raises ValueError."""
        with pytest.raises(ValueError, match="port must be in range 1-65535"):
            RedisClient(port=0)

    def test_init_port_too_large_raises(self):
        """Test that port > 65535 raises ValueError."""
        with pytest.raises(ValueError, match="port must be in range 1-65535"):
            RedisClient(port=65536)

    def test_init_invalid_ttl_raises(self):
        """Test that ttl_seconds <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="ttl_seconds must be positive"):
            RedisClient(ttl_seconds=0)

    @patch('zapomni_db.redis_cache.cache_client.ConnectionPool')
    @patch('zapomni_db.redis_cache.cache_client.Redis')
    def test_init_connection_failure_raises(self, mock_redis_class, mock_pool_class):
        """Test that connection failure raises RedisConnectionError."""
        mock_redis_instance = MagicMock()
        mock_redis_class.return_value = mock_redis_instance
        mock_redis_instance.ping.side_effect = redis.ConnectionError("Connection failed")

        with pytest.raises(RedisConnectionError):
            RedisClient()


# ============================================================================
# GET() TESTS
# ============================================================================

class TestRedisClientGet:
    """Test RedisClient.get() method."""

    @patch('zapomni_db.redis_cache.cache_client.ConnectionPool')
    @patch('zapomni_db.redis_cache.cache_client.Redis')
    def test_get_success_with_data(self, mock_redis_class, mock_pool_class):
        """Test successful get with existing key."""
        mock_redis_instance = MagicMock()
        mock_redis_class.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True
        mock_redis_instance.get.return_value = b'[0.1, 0.2, 0.3]'

        client = RedisClient()
        result = client.get("embedding_key")

        assert result == [0.1, 0.2, 0.3]
        mock_redis_instance.get.assert_called_once_with("embedding_key")

    @patch('zapomni_db.redis_cache.cache_client.ConnectionPool')
    @patch('zapomni_db.redis_cache.cache_client.Redis')
    def test_get_cache_miss(self, mock_redis_class, mock_pool_class):
        """Test get with non-existent key returns None."""
        mock_redis_instance = MagicMock()
        mock_redis_class.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True
        mock_redis_instance.get.return_value = None

        client = RedisClient()
        result = client.get("nonexistent_key")

        assert result is None

    @patch('zapomni_db.redis_cache.cache_client.ConnectionPool')
    @patch('zapomni_db.redis_cache.cache_client.Redis')
    def test_get_empty_key_raises(self, mock_redis_class, mock_pool_class):
        """Test that empty key raises ValueError."""
        mock_redis_instance = MagicMock()
        mock_redis_class.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True

        client = RedisClient()
        with pytest.raises(ValueError, match="key must be non-empty string"):
            client.get("")

    @patch('zapomni_db.redis_cache.cache_client.ConnectionPool')
    @patch('zapomni_db.redis_cache.cache_client.Redis')
    def test_get_invalid_json_raises(self, mock_redis_class, mock_pool_class):
        """Test that corrupted JSON raises DeserializationError."""
        mock_redis_instance = MagicMock()
        mock_redis_class.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True
        mock_redis_instance.get.return_value = b'invalid json {{'

        client = RedisClient()
        with pytest.raises(DeserializationError):
            client.get("bad_key")

    @patch('zapomni_db.redis_cache.cache_client.ConnectionPool')
    @patch('zapomni_db.redis_cache.cache_client.Redis')
    @patch('zapomni_db.redis_cache.cache_client.time.sleep')
    def test_get_retry_on_connection_error(self, mock_sleep, mock_redis_class, mock_pool_class):
        """Test that get retries on connection error with exponential backoff."""
        mock_redis_instance = MagicMock()
        mock_redis_class.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True

        # Fail twice, then succeed
        mock_redis_instance.get.side_effect = [
            redis.ConnectionError("Connection lost"),
            redis.ConnectionError("Connection lost"),
            b'{"success": true}'
        ]

        client = RedisClient()
        result = client.get("test_key")

        assert result == {"success": True}
        assert mock_redis_instance.get.call_count == 3
        # Check exponential backoff: 2^0=1s, 2^1=2s
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1)
        mock_sleep.assert_any_call(2)

    @patch('zapomni_db.redis_cache.cache_client.ConnectionPool')
    @patch('zapomni_db.redis_cache.cache_client.Redis')
    @patch('zapomni_db.redis_cache.cache_client.time.sleep')
    def test_get_fails_after_max_retries(self, mock_sleep, mock_redis_class, mock_pool_class):
        """Test that get fails after max retries exceeded."""
        mock_redis_instance = MagicMock()
        mock_redis_class.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True
        mock_redis_instance.get.side_effect = redis.ConnectionError("Connection lost")

        client = RedisClient()
        with pytest.raises(CacheError, match="failed after.*retries"):
            client.get("test_key")


# ============================================================================
# SET() TESTS
# ============================================================================

class TestRedisClientSet:
    """Test RedisClient.set() method."""

    @patch('zapomni_db.redis_cache.cache_client.ConnectionPool')
    @patch('zapomni_db.redis_cache.cache_client.Redis')
    def test_set_success_with_default_ttl(self, mock_redis_class, mock_pool_class):
        """Test successful set with default TTL."""
        mock_redis_instance = MagicMock()
        mock_redis_class.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True
        mock_redis_instance.set.return_value = True

        client = RedisClient()
        result = client.set("key1", [0.1, 0.2, 0.3])

        assert result is True
        # Verify set was called with default TTL (86400)
        mock_redis_instance.set.assert_called_once()
        call_args = mock_redis_instance.set.call_args
        assert call_args[0][0] == "key1"
        assert call_args[1]["ex"] == 86400

    @patch('zapomni_db.redis_cache.cache_client.ConnectionPool')
    @patch('zapomni_db.redis_cache.cache_client.Redis')
    def test_set_success_with_custom_ttl(self, mock_redis_class, mock_pool_class):
        """Test successful set with custom TTL."""
        mock_redis_instance = MagicMock()
        mock_redis_class.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True
        mock_redis_instance.set.return_value = True

        client = RedisClient()
        result = client.set("key1", {"data": "value"}, ttl=3600)

        assert result is True
        call_args = mock_redis_instance.set.call_args
        assert call_args[1]["ex"] == 3600

    @patch('zapomni_db.redis_cache.cache_client.ConnectionPool')
    @patch('zapomni_db.redis_cache.cache_client.Redis')
    def test_set_empty_key_raises(self, mock_redis_class, mock_pool_class):
        """Test that empty key raises ValueError."""
        mock_redis_instance = MagicMock()
        mock_redis_class.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True

        client = RedisClient()
        with pytest.raises(ValueError, match="key must be non-empty string"):
            client.set("", "value")

    @patch('zapomni_db.redis_cache.cache_client.ConnectionPool')
    @patch('zapomni_db.redis_cache.cache_client.Redis')
    def test_set_negative_ttl_raises(self, mock_redis_class, mock_pool_class):
        """Test that negative TTL raises ValueError."""
        mock_redis_instance = MagicMock()
        mock_redis_class.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True

        client = RedisClient()
        with pytest.raises(ValueError, match="ttl must be positive"):
            client.set("key1", "value", ttl=-1)

    @patch('zapomni_db.redis_cache.cache_client.ConnectionPool')
    @patch('zapomni_db.redis_cache.cache_client.Redis')
    def test_set_non_serializable_raises(self, mock_redis_class, mock_pool_class):
        """Test that non-serializable value raises SerializationError."""
        mock_redis_instance = MagicMock()
        mock_redis_class.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True

        client = RedisClient()

        # Create object that cannot be JSON serialized
        class NonSerializable:
            pass

        with pytest.raises(SerializationError):
            client.set("key1", NonSerializable())

    @patch('zapomni_db.redis_cache.cache_client.ConnectionPool')
    @patch('zapomni_db.redis_cache.cache_client.Redis')
    @patch('zapomni_db.redis_cache.cache_client.time.sleep')
    def test_set_retry_on_connection_error(self, mock_sleep, mock_redis_class, mock_pool_class):
        """Test that set retries on connection error."""
        mock_redis_instance = MagicMock()
        mock_redis_class.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True

        # Fail once, then succeed
        mock_redis_instance.set.side_effect = [
            redis.ConnectionError("Connection lost"),
            True
        ]

        client = RedisClient()
        result = client.set("key1", "value")

        assert result is True
        assert mock_redis_instance.set.call_count == 2
        assert mock_sleep.call_count == 1
        mock_sleep.assert_called_with(1)


# ============================================================================
# DELETE() TESTS
# ============================================================================

class TestRedisClientDelete:
    """Test RedisClient.delete() method."""

    @patch('zapomni_db.redis_cache.cache_client.ConnectionPool')
    @patch('zapomni_db.redis_cache.cache_client.Redis')
    def test_delete_success(self, mock_redis_class, mock_pool_class):
        """Test successful deletion of existing key."""
        mock_redis_instance = MagicMock()
        mock_redis_class.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True
        mock_redis_instance.delete.return_value = 1

        client = RedisClient()
        result = client.delete("key1")

        assert result is True

    @patch('zapomni_db.redis_cache.cache_client.ConnectionPool')
    @patch('zapomni_db.redis_cache.cache_client.Redis')
    def test_delete_miss(self, mock_redis_class, mock_pool_class):
        """Test delete of non-existent key returns False."""
        mock_redis_instance = MagicMock()
        mock_redis_class.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True
        mock_redis_instance.delete.return_value = 0

        client = RedisClient()
        result = client.delete("nonexistent")

        assert result is False

    @patch('zapomni_db.redis_cache.cache_client.ConnectionPool')
    @patch('zapomni_db.redis_cache.cache_client.Redis')
    def test_delete_empty_key_raises(self, mock_redis_class, mock_pool_class):
        """Test that empty key raises ValueError."""
        mock_redis_instance = MagicMock()
        mock_redis_class.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True

        client = RedisClient()
        with pytest.raises(ValueError, match="key must be non-empty string"):
            client.delete("")

    @patch('zapomni_db.redis_cache.cache_client.ConnectionPool')
    @patch('zapomni_db.redis_cache.cache_client.Redis')
    @patch('zapomni_db.redis_cache.cache_client.time.sleep')
    def test_delete_retry_on_connection_error(self, mock_sleep, mock_redis_class, mock_pool_class):
        """Test that delete retries on connection error."""
        mock_redis_instance = MagicMock()
        mock_redis_class.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True

        # Fail once, then succeed
        mock_redis_instance.delete.side_effect = [
            redis.ConnectionError("Connection lost"),
            1
        ]

        client = RedisClient()
        result = client.delete("key1")

        assert result is True
        assert mock_redis_instance.delete.call_count == 2


# ============================================================================
# EXISTS() TESTS
# ============================================================================

class TestRedisClientExists:
    """Test RedisClient.exists() method."""

    @patch('zapomni_db.redis_cache.cache_client.ConnectionPool')
    @patch('zapomni_db.redis_cache.cache_client.Redis')
    def test_exists_success(self, mock_redis_class, mock_pool_class):
        """Test exists returns True for existing key."""
        mock_redis_instance = MagicMock()
        mock_redis_class.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True
        mock_redis_instance.exists.return_value = 1

        client = RedisClient()
        result = client.exists("key1")

        assert result is True

    @patch('zapomni_db.redis_cache.cache_client.ConnectionPool')
    @patch('zapomni_db.redis_cache.cache_client.Redis')
    def test_exists_miss(self, mock_redis_class, mock_pool_class):
        """Test exists returns False for non-existent key."""
        mock_redis_instance = MagicMock()
        mock_redis_class.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True
        mock_redis_instance.exists.return_value = 0

        client = RedisClient()
        result = client.exists("nonexistent")

        assert result is False


# ============================================================================
# PING() TESTS
# ============================================================================

class TestRedisClientPing:
    """Test RedisClient.ping() method."""

    @patch('zapomni_db.redis_cache.cache_client.ConnectionPool')
    @patch('zapomni_db.redis_cache.cache_client.Redis')
    def test_ping_success(self, mock_redis_class, mock_pool_class):
        """Test ping returns True on success."""
        mock_redis_instance = MagicMock()
        mock_redis_class.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True

        client = RedisClient()
        result = client.ping()

        assert result is True

    @patch('zapomni_db.redis_cache.cache_client.ConnectionPool')
    @patch('zapomni_db.redis_cache.cache_client.Redis')
    def test_ping_pong_response(self, mock_redis_class, mock_pool_class):
        """Test ping handles PONG response."""
        mock_redis_instance = MagicMock()
        mock_redis_class.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = b"PONG"

        client = RedisClient()
        result = client.ping()

        assert result is True


# ============================================================================
# SCAN() TESTS
# ============================================================================

class TestRedisClientScan:
    """Test RedisClient.scan() method."""

    @patch('zapomni_db.redis_cache.cache_client.ConnectionPool')
    @patch('zapomni_db.redis_cache.cache_client.Redis')
    def test_scan_all_keys(self, mock_redis_class, mock_pool_class):
        """Test scan with * pattern returns all keys."""
        mock_redis_instance = MagicMock()
        mock_redis_class.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True
        mock_redis_instance.scan_iter.return_value = iter([
            b'embedding_1', b'embedding_2', b'metadata_1'
        ])

        client = RedisClient()
        result = client.scan("*")

        assert len(result) == 3
        assert 'embedding_1' in result or b'embedding_1' in result

    @patch('zapomni_db.redis_cache.cache_client.ConnectionPool')
    @patch('zapomni_db.redis_cache.cache_client.Redis')
    def test_scan_with_pattern(self, mock_redis_class, mock_pool_class):
        """Test scan with pattern filters keys."""
        mock_redis_instance = MagicMock()
        mock_redis_class.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True
        mock_redis_instance.scan_iter.return_value = iter([
            'embedding_1', 'embedding_2'
        ])

        client = RedisClient()
        result = client.scan("embedding_*")

        assert len(result) == 2
        mock_redis_instance.scan_iter.assert_called_once()


# ============================================================================
# INFO() TESTS
# ============================================================================

class TestRedisClientInfo:
    """Test RedisClient.info() method."""

    @patch('zapomni_db.redis_cache.cache_client.ConnectionPool')
    @patch('zapomni_db.redis_cache.cache_client.Redis')
    def test_info_success(self, mock_redis_class, mock_pool_class):
        """Test info returns valid statistics."""
        mock_redis_instance = MagicMock()
        mock_redis_class.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True
        mock_redis_instance.info.return_value = {
            'used_memory': 1024 * 1024,  # 1 MB
            'used_memory_human': '1M',
            'maxmemory': 100 * 1024 * 1024,  # 100 MB
            'maxmemory_policy': 'allkeys-lru',
            'keyspace': {
                'db0': {'keys': 10}
            }
        }

        client = RedisClient()
        result = client.info()

        assert 'used_memory_mb' in result
        assert 'total_keys' in result
        assert 'eviction_policy' in result
        assert result['used_memory_mb'] == 1.0
        assert result['total_keys'] == 10


# ============================================================================
# CLOSE() TESTS
# ============================================================================

class TestRedisClientClose:
    """Test RedisClient.close() method."""

    @patch('zapomni_db.redis_cache.cache_client.ConnectionPool')
    @patch('zapomni_db.redis_cache.cache_client.Redis')
    def test_close_success(self, mock_redis_class, mock_pool_class):
        """Test close releases connections."""
        mock_redis_instance = MagicMock()
        mock_pool_instance = MagicMock()
        mock_redis_class.return_value = mock_redis_instance
        mock_pool_class.return_value = mock_pool_instance
        mock_redis_instance.ping.return_value = True

        client = RedisClient()
        client.close()

        assert client._closed is True
        mock_redis_instance.close.assert_called_once()
        mock_pool_instance.disconnect.assert_called_once()

    @patch('zapomni_db.redis_cache.cache_client.ConnectionPool')
    @patch('zapomni_db.redis_cache.cache_client.Redis')
    def test_close_idempotent(self, mock_redis_class, mock_pool_class):
        """Test close can be called multiple times safely."""
        mock_redis_instance = MagicMock()
        mock_pool_instance = MagicMock()
        mock_redis_class.return_value = mock_redis_instance
        mock_pool_class.return_value = mock_pool_instance
        mock_redis_instance.ping.return_value = True

        client = RedisClient()
        client.close()
        client.close()  # Should not raise

        # close() should only be called once
        assert mock_redis_instance.close.call_count == 1


# ============================================================================
# SERIALIZATION TESTS
# ============================================================================

class TestRedisClientSerialization:
    """Test RedisClient serialization/deserialization."""

    @patch('zapomni_db.redis_cache.cache_client.ConnectionPool')
    @patch('zapomni_db.redis_cache.cache_client.Redis')
    def test_serialize_list(self, mock_redis_class, mock_pool_class):
        """Test _serialize converts list to JSON."""
        mock_redis_instance = MagicMock()
        mock_redis_class.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True

        client = RedisClient()
        result = client._serialize([0.1, 0.2, 0.3])

        assert result == '[0.1, 0.2, 0.3]'
        assert isinstance(result, str)

    @patch('zapomni_db.redis_cache.cache_client.ConnectionPool')
    @patch('zapomni_db.redis_cache.cache_client.Redis')
    def test_serialize_dict(self, mock_redis_class, mock_pool_class):
        """Test _serialize converts dict to JSON."""
        mock_redis_instance = MagicMock()
        mock_redis_class.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True

        client = RedisClient()
        result = client._serialize({"key": "value", "num": 42})

        data = json.loads(result)
        assert data["key"] == "value"
        assert data["num"] == 42

    @patch('zapomni_db.redis_cache.cache_client.ConnectionPool')
    @patch('zapomni_db.redis_cache.cache_client.Redis')
    def test_deserialize_list(self, mock_redis_class, mock_pool_class):
        """Test _deserialize converts JSON to list."""
        mock_redis_instance = MagicMock()
        mock_redis_class.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True

        client = RedisClient()
        result = client._deserialize('[0.1, 0.2, 0.3]')

        assert result == [0.1, 0.2, 0.3]

    @patch('zapomni_db.redis_cache.cache_client.ConnectionPool')
    @patch('zapomni_db.redis_cache.cache_client.Redis')
    def test_deserialize_bytes(self, mock_redis_class, mock_pool_class):
        """Test _deserialize handles bytes input."""
        mock_redis_instance = MagicMock()
        mock_redis_class.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True

        client = RedisClient()
        result = client._deserialize(b'{"key": "value"}')

        assert result == {"key": "value"}

    @patch('zapomni_db.redis_cache.cache_client.ConnectionPool')
    @patch('zapomni_db.redis_cache.cache_client.Redis')
    def test_deserialize_invalid_json_raises(self, mock_redis_class, mock_pool_class):
        """Test _deserialize raises on invalid JSON."""
        mock_redis_instance = MagicMock()
        mock_redis_class.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True

        client = RedisClient()
        with pytest.raises(DeserializationError):
            client._deserialize('invalid json {{')


# ============================================================================
# INTEGRATION-LIKE TESTS
# ============================================================================

class TestRedisClientIntegrationScenarios:
    """Test realistic usage scenarios."""

    @patch('zapomni_db.redis_cache.cache_client.ConnectionPool')
    @patch('zapomni_db.redis_cache.cache_client.Redis')
    def test_roundtrip_embedding(self, mock_redis_class, mock_pool_class):
        """Test set then get roundtrip with embedding."""
        mock_redis_instance = MagicMock()
        mock_redis_class.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True

        embedding = [0.1] * 768
        serialized = json.dumps(embedding)

        mock_redis_instance.set.return_value = True
        mock_redis_instance.get.return_value = serialized.encode()

        client = RedisClient()

        # Set embedding
        success = client.set("emb_123", embedding)
        assert success is True

        # Get embedding back
        result = client.get("emb_123")
        assert result == embedding

    @patch('zapomni_db.redis_cache.cache_client.ConnectionPool')
    @patch('zapomni_db.redis_cache.cache_client.Redis')
    def test_cache_workflow(self, mock_redis_class, mock_pool_class):
        """Test typical cache workflow: set, exists, get, delete."""
        mock_redis_instance = MagicMock()
        mock_redis_class.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True
        mock_redis_instance.set.return_value = True
        mock_redis_instance.exists.return_value = 1
        mock_redis_instance.get.return_value = b'{"cached": true}'
        mock_redis_instance.delete.return_value = 1

        client = RedisClient()

        # Set
        assert client.set("key1", {"cached": True}) is True

        # Exists
        assert client.exists("key1") is True

        # Get
        assert client.get("key1") == {"cached": True}

        # Delete
        assert client.delete("key1") is True
