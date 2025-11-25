"""
Unit tests for FalkorDB connection pooling functionality.

Tests cover:
- PoolConfig validation and initialization
- RetryConfig validation and initialization
- FalkorDBClient async pool initialization
- Connection pool statistics
- Retry logic with exponential backoff
- Async close() method
- Pool utilization monitoring

Copyright (c) 2025 Goncharenko Anton aka alienxs2
License: MIT
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from zapomni_db.exceptions import ConnectionError, ValidationError
from zapomni_db.pool_config import PoolConfig, RetryConfig

# ============================================================================
# POOLCONFIG TESTS
# ============================================================================


class TestPoolConfig:
    """Test PoolConfig dataclass and validation."""

    def test_pool_config_defaults(self):
        """Test PoolConfig initializes with correct defaults."""
        config = PoolConfig()

        assert config.min_size == 5
        assert config.max_size == 20
        assert config.timeout == 10.0
        assert config.socket_timeout == 30.0
        assert config.socket_connect_timeout == 5.0
        assert config.health_check_interval == 30

    def test_pool_config_custom_values(self):
        """Test PoolConfig accepts custom values."""
        config = PoolConfig(
            min_size=10,
            max_size=50,
            timeout=15.0,
            socket_timeout=60.0,
            socket_connect_timeout=10.0,
            health_check_interval=60,
        )

        assert config.min_size == 10
        assert config.max_size == 50
        assert config.timeout == 15.0
        assert config.socket_timeout == 60.0
        assert config.socket_connect_timeout == 10.0
        assert config.health_check_interval == 60

    def test_pool_config_min_size_validation(self):
        """Test min_size must be >= 1."""
        with pytest.raises(ValidationError, match="min_size"):
            PoolConfig(min_size=0)

    def test_pool_config_max_size_must_be_gte_min(self):
        """Test max_size must be >= min_size."""
        with pytest.raises(ValidationError, match="max_size"):
            PoolConfig(min_size=10, max_size=5)

    def test_pool_config_max_size_cap(self):
        """Test max_size cannot exceed 200."""
        with pytest.raises(ValidationError, match="max_size"):
            PoolConfig(max_size=201)

    def test_pool_config_timeout_positive(self):
        """Test timeout must be > 0."""
        with pytest.raises(ValidationError, match="timeout"):
            PoolConfig(timeout=0)

        with pytest.raises(ValidationError, match="timeout"):
            PoolConfig(timeout=-1.0)

    def test_pool_config_socket_timeout_positive(self):
        """Test socket_timeout must be > 0."""
        with pytest.raises(ValidationError, match="socket_timeout"):
            PoolConfig(socket_timeout=0)

    def test_pool_config_socket_connect_timeout_positive(self):
        """Test socket_connect_timeout must be > 0."""
        with pytest.raises(ValidationError, match="socket_connect_timeout"):
            PoolConfig(socket_connect_timeout=0)

    def test_pool_config_health_check_interval_minimum(self):
        """Test health_check_interval must be >= 10."""
        with pytest.raises(ValidationError, match="health_check_interval"):
            PoolConfig(health_check_interval=5)

    def test_pool_config_from_env(self):
        """Test PoolConfig.from_env() loads from environment."""
        with patch.dict(
            "os.environ",
            {
                "FALKORDB_POOL_MIN_SIZE": "8",
                "FALKORDB_POOL_MAX_SIZE": "40",
                "FALKORDB_POOL_TIMEOUT": "20.0",
                "FALKORDB_SOCKET_TIMEOUT": "45.0",
                "FALKORDB_HEALTH_CHECK_INTERVAL": "60",
            },
        ):
            config = PoolConfig.from_env()

            assert config.min_size == 8
            assert config.max_size == 40
            assert config.timeout == 20.0
            assert config.socket_timeout == 45.0
            assert config.health_check_interval == 60

    def test_pool_config_from_env_defaults(self):
        """Test PoolConfig.from_env() uses defaults when env vars not set."""
        with patch.dict("os.environ", {}, clear=True):
            config = PoolConfig.from_env()

            assert config.min_size == 5
            assert config.max_size == 20


# ============================================================================
# RETRYCONFIG TESTS
# ============================================================================


class TestRetryConfig:
    """Test RetryConfig dataclass and validation."""

    def test_retry_config_defaults(self):
        """Test RetryConfig initializes with correct defaults."""
        config = RetryConfig()

        assert config.max_retries == 3
        assert config.initial_delay == 0.1
        assert config.max_delay == 2.0
        assert config.exponential_base == 2.0

    def test_retry_config_custom_values(self):
        """Test RetryConfig accepts custom values."""
        config = RetryConfig(
            max_retries=5,
            initial_delay=0.5,
            max_delay=10.0,
            exponential_base=3.0,
        )

        assert config.max_retries == 5
        assert config.initial_delay == 0.5
        assert config.max_delay == 10.0
        assert config.exponential_base == 3.0

    def test_retry_config_max_retries_non_negative(self):
        """Test max_retries must be >= 0."""
        # 0 retries is valid (no retries)
        config = RetryConfig(max_retries=0)
        assert config.max_retries == 0

        with pytest.raises(ValidationError, match="max_retries"):
            RetryConfig(max_retries=-1)

    def test_retry_config_max_retries_cap(self):
        """Test max_retries cannot exceed 10."""
        with pytest.raises(ValidationError, match="max_retries"):
            RetryConfig(max_retries=11)

    def test_retry_config_initial_delay_positive(self):
        """Test initial_delay must be > 0."""
        with pytest.raises(ValidationError, match="initial_delay"):
            RetryConfig(initial_delay=0)

        with pytest.raises(ValidationError, match="initial_delay"):
            RetryConfig(initial_delay=-0.1)

    def test_retry_config_max_delay_gte_initial(self):
        """Test max_delay must be >= initial_delay."""
        with pytest.raises(ValidationError, match="max_delay"):
            RetryConfig(initial_delay=1.0, max_delay=0.5)

    def test_retry_config_exponential_base_minimum(self):
        """Test exponential_base must be >= 1.0."""
        # 1.0 is valid (no exponential growth)
        config = RetryConfig(exponential_base=1.0)
        assert config.exponential_base == 1.0

        with pytest.raises(ValidationError, match="exponential_base"):
            RetryConfig(exponential_base=0.5)

    def test_retry_config_from_env(self):
        """Test RetryConfig.from_env() loads from environment."""
        with patch.dict(
            "os.environ",
            {
                "FALKORDB_MAX_RETRIES": "5",
                "FALKORDB_RETRY_INITIAL_DELAY": "0.2",
                "FALKORDB_RETRY_MAX_DELAY": "5.0",
            },
        ):
            config = RetryConfig.from_env()

            assert config.max_retries == 5
            assert config.initial_delay == 0.2
            assert config.max_delay == 5.0


# ============================================================================
# FALKORDB CLIENT POOL TESTS
# ============================================================================


class TestFalkorDBClientPool:
    """Test FalkorDBClient connection pool functionality."""

    def test_client_accepts_pool_config(self):
        """Test FalkorDBClient accepts PoolConfig parameter."""
        from zapomni_db.falkordb_client import FalkorDBClient

        pool_config = PoolConfig(min_size=10, max_size=30)
        client = FalkorDBClient(pool_config=pool_config)

        assert client.pool_config.min_size == 10
        assert client.pool_config.max_size == 30

    def test_client_accepts_retry_config(self):
        """Test FalkorDBClient accepts RetryConfig parameter."""
        from zapomni_db.falkordb_client import FalkorDBClient

        retry_config = RetryConfig(max_retries=5)
        client = FalkorDBClient(retry_config=retry_config)

        assert client.retry_config.max_retries == 5

    def test_client_backwards_compat_pool_size(self):
        """Test FalkorDBClient maintains backwards compatibility with pool_size."""
        from zapomni_db.falkordb_client import FalkorDBClient

        client = FalkorDBClient(pool_size=50)

        assert client.pool_size == 50
        assert client.pool_config.max_size == 50

    def test_client_backwards_compat_max_retries(self):
        """Test FalkorDBClient maintains backwards compatibility with max_retries."""
        from zapomni_db.falkordb_client import FalkorDBClient

        client = FalkorDBClient(max_retries=7)

        assert client.max_retries == 7
        assert client.retry_config.max_retries == 7

    def test_client_not_initialized_until_init_async(self):
        """Test client is not initialized after __init__."""
        from zapomni_db.falkordb_client import FalkorDBClient

        client = FalkorDBClient()

        assert client._initialized is False
        assert client._closed is False

    @pytest.mark.asyncio
    async def test_client_close_is_async(self):
        """Test that close() is now async."""
        from zapomni_db.falkordb_client import FalkorDBClient

        client = FalkorDBClient()

        # close() should be a coroutine
        assert asyncio.iscoroutinefunction(client.close)

        # Should be safe to call even when not initialized
        await client.close()

        assert client._closed is True

    @pytest.mark.asyncio
    async def test_client_close_idempotent(self):
        """Test that close() can be called multiple times safely."""
        from zapomni_db.falkordb_client import FalkorDBClient

        client = FalkorDBClient()

        # Call close multiple times
        await client.close()
        await client.close()
        await client.close()

        assert client._closed is True

    @pytest.mark.asyncio
    async def test_get_pool_stats_returns_dict(self):
        """Test get_pool_stats() returns expected statistics."""
        from zapomni_db.falkordb_client import FalkorDBClient

        client = FalkorDBClient(pool_size=25)

        stats = await client.get_pool_stats()

        assert isinstance(stats, dict)
        assert "max_connections" in stats
        assert stats["max_connections"] == 25
        assert "active_connections" in stats
        assert stats["active_connections"] == 0
        assert "total_queries" in stats
        assert "total_retries" in stats
        assert "utilization_percent" in stats
        assert "initialized" in stats
        assert "closed" in stats

    @pytest.mark.asyncio
    async def test_get_pool_stats_utilization_calculation(self):
        """Test utilization_percent is calculated correctly."""
        from zapomni_db.falkordb_client import FalkorDBClient

        client = FalkorDBClient(pool_size=10)

        # Manually set active connections to test utilization
        client._active_connections = 5

        stats = await client.get_pool_stats()

        assert stats["utilization_percent"] == 50.0

    @pytest.mark.asyncio
    async def test_cannot_execute_when_closed(self):
        """Test that queries fail after close()."""
        from zapomni_db.falkordb_client import FalkorDBClient

        client = FalkorDBClient()
        client._initialized = True  # Pretend initialized
        await client.close()

        # After close(), _initialized is set to False, so we get the "not initialized" error
        # This is correct behavior - closed client should not execute queries
        with pytest.raises(ConnectionError):
            await client._execute_cypher("RETURN 1", {})

    @pytest.mark.asyncio
    async def test_cannot_init_after_close(self):
        """Test that init_async fails after close()."""
        from zapomni_db.falkordb_client import FalkorDBClient

        client = FalkorDBClient()
        await client.close()

        with pytest.raises(ConnectionError, match="closed"):
            await client.init_async()


# ============================================================================
# RETRY LOGIC TESTS
# ============================================================================


class TestRetryLogic:
    """Test exponential backoff retry logic."""

    @pytest.mark.asyncio
    async def test_retry_on_connection_error(self):
        """Test that ConnectionError triggers retry."""
        from zapomni_db.falkordb_client import FalkorDBClient

        client = FalkorDBClient()
        client._initialized = True
        retry_config = RetryConfig(max_retries=2, initial_delay=0.01)
        client.retry_config = retry_config

        # Mock graph.query to fail twice then succeed
        call_count = 0

        async def mock_query(query, params):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Simulated connection error")
            # Return a mock result
            mock_result = MagicMock()
            mock_result.result_set = [[1]]
            mock_result.header = [("col", "value")]
            mock_result.run_time_ms = 1.0
            return mock_result

        client.graph = MagicMock()
        client.graph.query = mock_query

        result = await client._execute_with_retry("RETURN 1", {})

        assert call_count == 3
        assert result.row_count == 1

    @pytest.mark.asyncio
    async def test_retry_exhausted_raises_error(self):
        """Test that max retries exhausted raises ConnectionError."""
        from zapomni_db.falkordb_client import FalkorDBClient

        client = FalkorDBClient()
        client._initialized = True
        retry_config = RetryConfig(max_retries=2, initial_delay=0.01)
        client.retry_config = retry_config

        # Mock graph.query to always fail
        async def mock_query(query, params):
            raise ConnectionError("Persistent connection error")

        client.graph = MagicMock()
        client.graph.query = mock_query

        with pytest.raises(ConnectionError, match="attempts"):
            await client._execute_with_retry("RETURN 1", {})

    @pytest.mark.asyncio
    async def test_no_retry_on_query_error(self):
        """Test that QueryError does NOT trigger retry."""
        from zapomni_db.exceptions import QueryError
        from zapomni_db.falkordb_client import FalkorDBClient

        client = FalkorDBClient()
        client._initialized = True
        retry_config = RetryConfig(max_retries=5, initial_delay=0.01)
        client.retry_config = retry_config

        call_count = 0

        async def mock_query(query, params):
            nonlocal call_count
            call_count += 1
            raise Exception("Invalid Cypher syntax")  # Non-retryable

        client.graph = MagicMock()
        client.graph.query = mock_query

        with pytest.raises(QueryError):
            await client._execute_with_retry("INVALID CYPHER", {})

        # Should have been called only once (no retries)
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_tracks_total_retries(self):
        """Test that total_retries counter is incremented."""
        from zapomni_db.falkordb_client import FalkorDBClient

        client = FalkorDBClient()
        client._initialized = True
        retry_config = RetryConfig(max_retries=3, initial_delay=0.01)
        client.retry_config = retry_config

        call_count = 0

        async def mock_query(query, params):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Transient error")
            mock_result = MagicMock()
            mock_result.result_set = [[1]]
            mock_result.header = [("col", "value")]
            mock_result.run_time_ms = 1.0
            return mock_result

        client.graph = MagicMock()
        client.graph.query = mock_query

        await client._execute_with_retry("RETURN 1", {})

        assert client._total_retries == 2  # Two retries before success

    @pytest.mark.asyncio
    async def test_zero_retries_no_retry(self):
        """Test that max_retries=0 means no retries."""
        from zapomni_db.falkordb_client import FalkorDBClient

        client = FalkorDBClient()
        client._initialized = True
        retry_config = RetryConfig(max_retries=0, initial_delay=0.01)
        client.retry_config = retry_config

        call_count = 0

        async def mock_query(query, params):
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Fail")

        client.graph = MagicMock()
        client.graph.query = mock_query

        with pytest.raises(ConnectionError):
            await client._execute_with_retry("RETURN 1", {})

        # Only one call, no retries
        assert call_count == 1


# ============================================================================
# POOL MONITORING TESTS
# ============================================================================


class TestPoolMonitoring:
    """Test pool utilization monitoring."""

    @pytest.mark.asyncio
    async def test_high_utilization_warning(self):
        """Test that high utilization logs a warning."""
        from zapomni_db.falkordb_client import FalkorDBClient

        client = FalkorDBClient(pool_size=10)

        # Simulate high utilization (>80%)
        client._active_connections = 9  # 90%

        # Check utilization (this should log warning)
        await client._check_pool_utilization()

        assert client._utilization_warning_logged is True

    @pytest.mark.asyncio
    async def test_utilization_warning_resets(self):
        """Test that utilization warning resets when usage drops."""
        from zapomni_db.falkordb_client import FalkorDBClient

        client = FalkorDBClient(pool_size=10)

        # High utilization
        client._active_connections = 9
        await client._check_pool_utilization()
        assert client._utilization_warning_logged is True

        # Low utilization
        client._active_connections = 5  # 50% - below reset threshold of 60%
        await client._check_pool_utilization()
        assert client._utilization_warning_logged is False

    @pytest.mark.asyncio
    async def test_query_tracking(self):
        """Test that queries are tracked."""
        from zapomni_db.falkordb_client import FalkorDBClient

        client = FalkorDBClient()
        client._initialized = True

        async def mock_query(query, params):
            mock_result = MagicMock()
            mock_result.result_set = [[1]]
            mock_result.header = [("col", "value")]
            mock_result.run_time_ms = 1.0
            return mock_result

        client.graph = MagicMock()
        client.graph.query = mock_query

        initial_queries = client._total_queries

        await client._execute_cypher("RETURN 1", {})
        await client._execute_cypher("RETURN 2", {})

        assert client._total_queries == initial_queries + 2


# ============================================================================
# INTEGRATION WITH STATS
# ============================================================================


class TestPoolStatsInGetStats:
    """Test pool statistics are included in get_stats()."""

    @pytest.mark.asyncio
    async def test_get_stats_includes_pool_stats(self):
        """Test that get_stats() includes pool statistics."""
        from zapomni_db.falkordb_client import FalkorDBClient

        client = FalkorDBClient(pool_size=15)
        client._initialized = True

        # Mock _execute_cypher to return empty results
        async def mock_execute(query, params):
            from zapomni_db.models import QueryResult

            return QueryResult(rows=[], row_count=0, execution_time_ms=0)

        client._execute_cypher = mock_execute

        stats = await client.get_stats()

        assert "pool" in stats
        assert stats["pool"]["size"] == 15
        assert stats["pool"]["active_connections"] == 0
        assert stats["pool"]["total_queries"] >= 0
        assert stats["pool"]["initialized"] is True
        assert stats["pool"]["closed"] is False
