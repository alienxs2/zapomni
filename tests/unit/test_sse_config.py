"""
Unit tests for SSEConfig component.

Tests the SSE configuration including:
- Default values
- Environment variable loading via from_env()
- Validation of configuration values
- Edge cases and error handling

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

import os
from unittest.mock import patch

import pytest

from zapomni_core.exceptions import ValidationError
from zapomni_mcp.config import SSEConfig

# Default Values Tests


class TestSSEConfigDefaults:
    """Test suite for SSEConfig default values."""

    def test_default_host(self):
        """Should default to 127.0.0.1 (localhost only)."""
        config = SSEConfig()
        assert config.host == "127.0.0.1"

    def test_default_port(self):
        """Should default to port 8000."""
        config = SSEConfig()
        assert config.port == 8000

    def test_default_cors_origins(self):
        """Should default to wildcard CORS origins."""
        config = SSEConfig()
        assert config.cors_origins == ["*"]

    def test_default_heartbeat_interval(self):
        """Should default to 30 seconds heartbeat."""
        config = SSEConfig()
        assert config.heartbeat_interval == 30

    def test_default_max_connection_lifetime(self):
        """Should default to 3600 seconds (1 hour)."""
        config = SSEConfig()
        assert config.max_connection_lifetime == 3600


# Custom Values Tests


class TestSSEConfigCustomValues:
    """Test suite for SSEConfig with custom values."""

    def test_custom_host(self):
        """Should accept custom host address."""
        config = SSEConfig(host="0.0.0.0")
        assert config.host == "0.0.0.0"

    def test_custom_port(self):
        """Should accept custom port number."""
        config = SSEConfig(port=9000)
        assert config.port == 9000

    def test_custom_cors_origins(self):
        """Should accept custom CORS origins list."""
        origins = ["http://localhost:3000", "http://localhost:5000"]
        config = SSEConfig(cors_origins=origins)
        assert config.cors_origins == origins

    def test_custom_heartbeat_interval(self):
        """Should accept custom heartbeat interval."""
        config = SSEConfig(heartbeat_interval=60)
        assert config.heartbeat_interval == 60

    def test_custom_max_connection_lifetime(self):
        """Should accept custom max connection lifetime."""
        config = SSEConfig(max_connection_lifetime=7200)
        assert config.max_connection_lifetime == 7200

    def test_all_custom_values(self):
        """Should accept all custom values together."""
        config = SSEConfig(
            host="0.0.0.0",
            port=9000,
            cors_origins=["http://example.com"],
            heartbeat_interval=45,
            max_connection_lifetime=1800,
        )

        assert config.host == "0.0.0.0"
        assert config.port == 9000
        assert config.cors_origins == ["http://example.com"]
        assert config.heartbeat_interval == 45
        assert config.max_connection_lifetime == 1800


# Validation Tests


class TestSSEConfigValidation:
    """Test suite for SSEConfig validation."""

    def test_port_too_low_raises(self):
        """Should raise ValidationError for port 0."""
        with pytest.raises(ValidationError, match="port must be between"):
            SSEConfig(port=0)

    def test_port_too_high_raises(self):
        """Should raise ValidationError for port > 65535."""
        with pytest.raises(ValidationError, match="port must be between"):
            SSEConfig(port=65536)

    def test_port_negative_raises(self):
        """Should raise ValidationError for negative port."""
        with pytest.raises(ValidationError, match="port must be between"):
            SSEConfig(port=-1)

    def test_valid_port_range(self):
        """Should accept valid port numbers."""
        # Test boundary values
        config_min = SSEConfig(port=1)
        config_max = SSEConfig(port=65535)

        assert config_min.port == 1
        assert config_max.port == 65535

    def test_heartbeat_interval_too_low_raises(self):
        """Should raise ValidationError for heartbeat < 5 seconds."""
        with pytest.raises(ValidationError, match="heartbeat_interval must be at least 5"):
            SSEConfig(heartbeat_interval=4)

    def test_heartbeat_interval_minimum(self):
        """Should accept minimum heartbeat interval of 5 seconds."""
        config = SSEConfig(heartbeat_interval=5)
        assert config.heartbeat_interval == 5

    def test_max_connection_lifetime_too_low_raises(self):
        """Should raise ValidationError for lifetime < 60 seconds."""
        with pytest.raises(ValidationError, match="max_connection_lifetime must be at least 60"):
            SSEConfig(max_connection_lifetime=59)

    def test_max_connection_lifetime_minimum(self):
        """Should accept minimum lifetime of 60 seconds."""
        config = SSEConfig(max_connection_lifetime=60)
        assert config.max_connection_lifetime == 60


# Environment Variable Tests


class TestSSEConfigFromEnv:
    """Test suite for SSEConfig.from_env() method."""

    def test_from_env_default_values(self):
        """Should use defaults when env vars not set."""
        # Clear any existing env vars
        env_vars = [
            "ZAPOMNI_SSE_HOST",
            "ZAPOMNI_SSE_PORT",
            "ZAPOMNI_SSE_CORS_ORIGINS",
            "ZAPOMNI_SSE_HEARTBEAT_INTERVAL",
            "ZAPOMNI_SSE_MAX_CONNECTION_LIFETIME",
        ]

        with patch.dict(os.environ, {}, clear=True):
            # Ensure test env vars are not set
            for var in env_vars:
                os.environ.pop(var, None)

            config = SSEConfig.from_env()

            assert config.host == "127.0.0.1"
            assert config.port == 8000
            assert config.cors_origins == ["*"]
            assert config.heartbeat_interval == 30
            assert config.max_connection_lifetime == 3600

    def test_from_env_custom_host(self):
        """Should read host from ZAPOMNI_SSE_HOST."""
        with patch.dict(os.environ, {"ZAPOMNI_SSE_HOST": "0.0.0.0"}):
            config = SSEConfig.from_env()
            assert config.host == "0.0.0.0"

    def test_from_env_custom_port(self):
        """Should read port from ZAPOMNI_SSE_PORT."""
        with patch.dict(os.environ, {"ZAPOMNI_SSE_PORT": "9000"}):
            config = SSEConfig.from_env()
            assert config.port == 9000

    def test_from_env_custom_cors_origins(self):
        """Should parse comma-separated CORS origins."""
        with patch.dict(
            os.environ,
            {"ZAPOMNI_SSE_CORS_ORIGINS": "http://localhost:3000,http://localhost:5000"},
        ):
            config = SSEConfig.from_env()
            assert config.cors_origins == [
                "http://localhost:3000",
                "http://localhost:5000",
            ]

    def test_from_env_single_cors_origin(self):
        """Should handle single CORS origin."""
        with patch.dict(os.environ, {"ZAPOMNI_SSE_CORS_ORIGINS": "http://example.com"}):
            config = SSEConfig.from_env()
            assert config.cors_origins == ["http://example.com"]

    def test_from_env_cors_origins_strips_whitespace(self):
        """Should strip whitespace from CORS origins."""
        with patch.dict(
            os.environ,
            {"ZAPOMNI_SSE_CORS_ORIGINS": "http://localhost:3000 , http://localhost:5000 "},
        ):
            config = SSEConfig.from_env()
            assert config.cors_origins == [
                "http://localhost:3000",
                "http://localhost:5000",
            ]

    def test_from_env_custom_heartbeat_interval(self):
        """Should read heartbeat interval from env."""
        with patch.dict(os.environ, {"ZAPOMNI_SSE_HEARTBEAT_INTERVAL": "60"}):
            config = SSEConfig.from_env()
            assert config.heartbeat_interval == 60

    def test_from_env_custom_max_connection_lifetime(self):
        """Should read max connection lifetime from env."""
        with patch.dict(os.environ, {"ZAPOMNI_SSE_MAX_CONNECTION_LIFETIME": "7200"}):
            config = SSEConfig.from_env()
            assert config.max_connection_lifetime == 7200

    def test_from_env_all_custom_values(self):
        """Should read all values from environment."""
        env_vars = {
            "ZAPOMNI_SSE_HOST": "0.0.0.0",
            "ZAPOMNI_SSE_PORT": "9000",
            "ZAPOMNI_SSE_CORS_ORIGINS": "http://app.example.com",
            "ZAPOMNI_SSE_HEARTBEAT_INTERVAL": "45",
            "ZAPOMNI_SSE_MAX_CONNECTION_LIFETIME": "1800",
        }

        with patch.dict(os.environ, env_vars):
            config = SSEConfig.from_env()

            assert config.host == "0.0.0.0"
            assert config.port == 9000
            assert config.cors_origins == ["http://app.example.com"]
            assert config.heartbeat_interval == 45
            assert config.max_connection_lifetime == 1800

    def test_from_env_invalid_port_raises(self):
        """Should raise when port env var is invalid."""
        with patch.dict(os.environ, {"ZAPOMNI_SSE_PORT": "99999"}):
            with pytest.raises(ValidationError):
                SSEConfig.from_env()

    def test_from_env_invalid_heartbeat_raises(self):
        """Should raise when heartbeat env var is invalid."""
        with patch.dict(os.environ, {"ZAPOMNI_SSE_HEARTBEAT_INTERVAL": "1"}):
            with pytest.raises(ValidationError):
                SSEConfig.from_env()

    def test_from_env_invalid_lifetime_raises(self):
        """Should raise when lifetime env var is invalid."""
        with patch.dict(os.environ, {"ZAPOMNI_SSE_MAX_CONNECTION_LIFETIME": "30"}):
            with pytest.raises(ValidationError):
                SSEConfig.from_env()


# Edge Cases


class TestSSEConfigEdgeCases:
    """Test suite for SSEConfig edge cases."""

    def test_empty_cors_origins_list(self):
        """Should accept empty CORS origins list."""
        config = SSEConfig(cors_origins=[])
        assert config.cors_origins == []

    def test_ipv6_host(self):
        """Should accept IPv6 host address."""
        config = SSEConfig(host="::1")
        assert config.host == "::1"

    def test_hostname_as_host(self):
        """Should accept hostname as host."""
        config = SSEConfig(host="localhost")
        assert config.host == "localhost"

    def test_large_heartbeat_interval(self):
        """Should accept large heartbeat interval."""
        config = SSEConfig(heartbeat_interval=300)  # 5 minutes
        assert config.heartbeat_interval == 300

    def test_large_connection_lifetime(self):
        """Should accept large connection lifetime."""
        config = SSEConfig(max_connection_lifetime=86400)  # 24 hours
        assert config.max_connection_lifetime == 86400

    def test_multiple_cors_origins(self):
        """Should accept multiple CORS origins."""
        origins = [
            "http://localhost:3000",
            "http://localhost:5000",
            "https://app.example.com",
            "https://admin.example.com",
        ]
        config = SSEConfig(cors_origins=origins)
        assert len(config.cors_origins) == 4


# Dataclass Behavior Tests


class TestSSEConfigDataclass:
    """Test suite for SSEConfig dataclass behavior."""

    def test_equality(self):
        """Should support equality comparison."""
        config1 = SSEConfig(port=8000)
        config2 = SSEConfig(port=8000)
        config3 = SSEConfig(port=9000)

        assert config1 == config2
        assert config1 != config3

    def test_immutability_after_creation(self):
        """Should allow field modification (not frozen)."""
        config = SSEConfig()
        config.port = 9000

        assert config.port == 9000

    def test_repr(self):
        """Should have readable repr."""
        config = SSEConfig()
        repr_str = repr(config)

        assert "SSEConfig" in repr_str
        assert "host" in repr_str
        assert "port" in repr_str
