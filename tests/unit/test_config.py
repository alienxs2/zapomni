"""
Unit tests for ConfigurationManager (ZapomniSettings).

Tests configuration loading, validation, computed properties,
and environment variable handling.

Copyright (c) 2025 Goncharenko Anton aka alienxs2
License: MIT
"""

import os
import tempfile
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from zapomni_core.config import ZapomniSettings, get_config_summary, validate_configuration

# ============================================================
# CONFIGURATION LOADING TESTS
# ============================================================


def test_default_configuration():
    """Test that default configuration loads successfully with all defaults."""
    settings = ZapomniSettings()

    # FalkorDB defaults
    assert settings.falkordb_host == "localhost"
    assert settings.falkordb_port == 6381
    assert settings.graph_name == "zapomni_memory"
    assert settings.falkordb_connection_timeout == 30
    assert settings.falkordb_pool_size == 20  # Increased for SSE concurrency

    # Ollama defaults
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.ollama_embedding_model == "nomic-embed-text"
    assert settings.ollama_llm_model == "llama3.1:8b"

    # Performance defaults
    assert settings.max_chunk_size == 512
    assert settings.chunk_overlap == 50
    assert settings.vector_dimensions == 768

    # Logging defaults
    assert settings.log_level == "INFO"
    assert settings.log_format == "json"

    # Feature flags defaults (core features enabled by default)
    assert settings.enable_hybrid_search == True
    assert settings.enable_knowledge_graph == True
    assert settings.enable_code_indexing == True
    assert settings.enable_semantic_cache == True  # Enabled by default for performance (Redis + in-memory fallback)


def test_environment_override(monkeypatch):
    """Test that environment variables override defaults."""
    monkeypatch.setenv("FALKORDB_HOST", "custom.host")
    monkeypatch.setenv("FALKORDB_PORT", "7000")
    monkeypatch.setenv("MAX_CHUNK_SIZE", "1024")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    settings = ZapomniSettings()

    assert settings.falkordb_host == "custom.host"
    assert settings.falkordb_port == 7000
    assert settings.max_chunk_size == 1024
    assert settings.log_level == "DEBUG"


def test_dotenv_loading(tmp_path, monkeypatch):
    """Test loading configuration from .env file."""
    # Set environment variables directly (Pydantic v2 settings behavior)
    monkeypatch.setenv("FALKORDB_PORT", "7001")
    monkeypatch.setenv("GRAPH_NAME", "test_graph")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")

    settings = ZapomniSettings()
    assert settings.falkordb_port == 7001
    assert settings.graph_name == "test_graph"
    assert settings.log_level == "WARNING"


def test_system_env_overrides_dotenv(tmp_path, monkeypatch):
    """Test that system environment variables have higher priority than .env file."""
    env_file = tmp_path / ".env"
    env_file.write_text("FALKORDB_HOST=dotenv.host\n")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FALKORDB_HOST", "system.host")

    settings = ZapomniSettings()
    assert settings.falkordb_host == "system.host"  # System env wins


# ============================================================
# VALIDATION TESTS
# ============================================================


def test_invalid_port_zero():
    """Test that port 0 is invalid."""
    with pytest.raises(ValidationError) as exc_info:
        ZapomniSettings(falkordb_port=0)
    assert "falkordb_port" in str(exc_info.value)


def test_invalid_port_negative():
    """Test that negative port is invalid."""
    with pytest.raises(ValidationError) as exc_info:
        ZapomniSettings(falkordb_port=-1)
    assert "falkordb_port" in str(exc_info.value)


def test_invalid_port_too_large():
    """Test that port > 65535 is invalid."""
    with pytest.raises(ValidationError) as exc_info:
        ZapomniSettings(falkordb_port=99999)
    assert "falkordb_port" in str(exc_info.value)


def test_invalid_log_level():
    """Test that invalid log level raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        ZapomniSettings(log_level="INVALID")
    assert "log_level" in str(exc_info.value)


def test_log_level_case_insensitive():
    """Test that log level is case-insensitive and normalized to uppercase."""
    settings = ZapomniSettings(log_level="debug")
    assert settings.log_level == "DEBUG"

    settings = ZapomniSettings(log_level="Info")
    assert settings.log_level == "INFO"


def test_invalid_log_format():
    """Test that invalid log format raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        ZapomniSettings(log_format="xml")
    assert "log_format" in str(exc_info.value)


def test_log_format_case_insensitive():
    """Test that log format is case-insensitive and normalized to lowercase."""
    settings = ZapomniSettings(log_format="JSON")
    assert settings.log_format == "json"

    settings = ZapomniSettings(log_format="Text")
    assert settings.log_format == "text"


def test_invalid_url_no_protocol():
    """Test that URL without http:// or https:// is invalid."""
    with pytest.raises(ValidationError) as exc_info:
        ZapomniSettings(ollama_base_url="localhost:11434")
    assert "ollama_base_url" in str(exc_info.value)


def test_url_trailing_slash_removed():
    """Test that trailing slash is removed from URL."""
    settings = ZapomniSettings(ollama_base_url="http://localhost:11434/")
    assert settings.ollama_base_url == "http://localhost:11434"


def test_chunk_overlap_equals_chunk_size():
    """Test that overlap >= chunk_size is invalid."""
    with pytest.raises(ValidationError) as exc_info:
        ZapomniSettings(max_chunk_size=512, chunk_overlap=512)
    assert "chunk_overlap" in str(exc_info.value)


def test_chunk_overlap_greater_than_chunk_size():
    """Test that overlap > chunk_size is invalid."""
    with pytest.raises(ValidationError) as exc_info:
        ZapomniSettings(max_chunk_size=512, chunk_overlap=600)
    assert "chunk_overlap" in str(exc_info.value)


def test_chunk_overlap_warning_over_50_percent(monkeypatch):
    """Test that warning is issued if overlap > 50% of chunk_size."""
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        settings = ZapomniSettings(max_chunk_size=512, chunk_overlap=300)

        # Check that warning was issued
        assert len(w) > 0
        assert "50%" in str(w[0].message)


def test_vector_dimensions_non_standard_warning():
    """Test that non-standard vector dimensions issue a warning."""
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        settings = ZapomniSettings(vector_dimensions=999)

        # Check that warning was issued
        assert len(w) > 0
        assert "non-standard" in str(w[0].message).lower()


# ============================================================
# COMPUTED PROPERTIES TESTS
# ============================================================


def test_falkordb_connection_string_no_password():
    """Test connection string generation without password."""
    settings = ZapomniSettings(falkordb_host="localhost", falkordb_port=6379)

    assert settings.falkordb_connection_string == "redis://localhost:6379"


def test_falkordb_connection_string_with_password():
    """Test connection string generation with password."""
    settings = ZapomniSettings(
        falkordb_host="localhost", falkordb_port=6379, falkordb_password="secret123"
    )

    assert settings.falkordb_connection_string == "redis://secret123@localhost:6379"


def test_redis_connection_string():
    """Test Redis connection string format."""
    settings = ZapomniSettings(redis_host="redis.host", redis_port=6380)

    assert settings.redis_connection_string == "redis://redis.host:6380"


def test_is_development_true():
    """Test development mode detection (DEBUG log level)."""
    settings = ZapomniSettings(log_level="DEBUG")
    assert settings.is_development == True
    assert settings.is_production == False


def test_is_development_false():
    """Test production mode detection (INFO, WARNING, ERROR)."""
    settings = ZapomniSettings(log_level="INFO")
    assert settings.is_development == False
    assert settings.is_production == True

    settings = ZapomniSettings(log_level="WARNING")
    assert settings.is_development == False
    assert settings.is_production == True


# ============================================================
# SECRET HANDLING TESTS
# ============================================================


def test_secret_masking():
    """Test that SecretStr fields are masked in string representation."""
    settings = ZapomniSettings(falkordb_password="super_secret")

    # SecretStr should mask the value
    settings_str = str(settings)
    assert "super_secret" not in settings_str


def test_secret_access():
    """Test that get_secret_value() works for accessing password."""
    settings = ZapomniSettings(falkordb_password="secret123")

    # Should be able to access via get_secret_value()
    assert settings.falkordb_password.get_secret_value() == "secret123"


# ============================================================
# DIRECTORY CREATION TESTS
# ============================================================


def test_data_dir_created(tmp_path):
    """Test that data_dir is created if it doesn't exist."""
    data_dir = tmp_path / "data"
    assert not data_dir.exists()

    settings = ZapomniSettings(data_dir=data_dir)

    # Directory should be created
    assert data_dir.exists()
    assert data_dir.is_dir()


def test_temp_dir_created(tmp_path):
    """Test that temp_dir is created if it doesn't exist."""
    temp_dir = tmp_path / "temp"
    assert not temp_dir.exists()

    settings = ZapomniSettings(temp_dir=temp_dir)

    # Directory should be created
    assert temp_dir.exists()
    assert temp_dir.is_dir()


def test_directory_creation_nested(tmp_path):
    """Test that nested directories are created."""
    nested_dir = tmp_path / "level1" / "level2" / "data"
    assert not nested_dir.exists()

    settings = ZapomniSettings(data_dir=nested_dir)

    # All parent directories should be created
    assert nested_dir.exists()
    assert nested_dir.is_dir()


# ============================================================
# HELPER FUNCTIONS TESTS
# ============================================================


def test_get_config_summary():
    """Test configuration summary structure."""
    settings = ZapomniSettings()
    summary = get_config_summary(settings)

    # Check that summary has expected categories
    assert "database" in summary
    assert "ollama" in summary
    assert "performance" in summary
    assert "features" in summary
    assert "logging" in summary

    # Check database section
    assert summary["database"]["falkordb_host"] == "localhost"
    assert summary["database"]["falkordb_port"] == 6381

    # Check features section (core features enabled by default)
    assert summary["features"]["hybrid_search"] == True
    assert summary["features"]["knowledge_graph"] == True


def test_validate_configuration_success(tmp_path):
    """Test configuration validation passes for valid settings."""
    settings = ZapomniSettings(data_dir=tmp_path / "data")
    is_valid, errors = validate_configuration(settings)

    assert is_valid == True
    assert len(errors) == 0


def test_validate_configuration_detects_issues(tmp_path):
    """Test configuration validation detects issues."""
    # Create a read-only directory
    readonly_dir = tmp_path / "readonly"
    readonly_dir.mkdir()
    readonly_dir.chmod(0o444)  # Read-only

    settings = ZapomniSettings(data_dir=readonly_dir)
    is_valid, errors = validate_configuration(settings)

    # Should detect that directory is not writable
    assert is_valid == False
    assert len(errors) > 0


# ============================================================
# INTEGRATION TESTS
# ============================================================


def test_full_configuration_lifecycle(tmp_path, monkeypatch):
    """Test complete configuration lifecycle with all features."""
    # Setup environment variables
    monkeypatch.setenv("FALKORDB_HOST", "testdb.local")
    monkeypatch.setenv("FALKORDB_PORT", "6381")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("MAX_CHUNK_SIZE", "1024")

    # Create settings
    settings = ZapomniSettings(data_dir=tmp_path / "data", temp_dir=tmp_path / "temp")

    # Verify loaded correctly
    assert settings.falkordb_host == "testdb.local"
    assert settings.falkordb_port == 6381
    assert settings.log_level == "DEBUG"
    assert settings.max_chunk_size == 1024

    # Verify directories created
    assert (tmp_path / "data").exists()
    assert (tmp_path / "temp").exists()

    # Verify computed properties
    assert settings.is_development == True
    assert "testdb.local:6381" in settings.falkordb_connection_string

    # Verify configuration summary
    summary = get_config_summary(settings)
    assert summary["database"]["falkordb_host"] == "testdb.local"
    assert summary["logging"]["level"] == "DEBUG"
