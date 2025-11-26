"""
Unit tests for ConfigManager re-export.

Tests that zapomni_mcp.config correctly re-exports ZapomniSettings from zapomni_core.config
and provides the ConfigManager alias.

Copyright (c) 2025 Goncharenko Anton aka alienxs2
License: MIT
"""

from pathlib import Path

import pytest


def test_config_manager_alias_exists():
    """Test that ConfigManager alias is available."""
    from zapomni_mcp.config import ConfigManager

    assert ConfigManager is not None


def test_config_manager_is_zapomni_settings():
    """Test that ConfigManager is an alias for ZapomniSettings."""
    from zapomni_core.config import ZapomniSettings
    from zapomni_mcp.config import ConfigManager

    # ConfigManager should be the same class as ZapomniSettings
    assert ConfigManager is ZapomniSettings


def test_zapomni_settings_reexport():
    """Test that ZapomniSettings is re-exported from zapomni_mcp."""
    from zapomni_core.config import ZapomniSettings as CoreSettings
    from zapomni_mcp.config import ZapomniSettings

    # Should be the same class
    assert ZapomniSettings is CoreSettings


def test_get_config_summary_reexport():
    """Test that get_config_summary is re-exported."""
    from zapomni_core.config import get_config_summary as core_get_config_summary
    from zapomni_mcp.config import get_config_summary

    # Should be the same function
    assert get_config_summary is core_get_config_summary


def test_validate_configuration_reexport():
    """Test that validate_configuration is re-exported."""
    from zapomni_core.config import validate_configuration as core_validate_configuration
    from zapomni_mcp.config import validate_configuration

    # Should be the same function
    assert validate_configuration is core_validate_configuration


def test_config_manager_instantiation():
    """Test that ConfigManager can be instantiated."""
    from zapomni_mcp.config import ConfigManager

    # Create instance with custom values to avoid side effects
    config = ConfigManager(
        data_dir=Path("/tmp/test_zapomni_data"), temp_dir=Path("/tmp/test_zapomni_temp")
    )

    assert config is not None
    assert hasattr(config, "falkordb_host")
    assert hasattr(config, "ollama_base_url")
    assert hasattr(config, "max_chunk_size")


def test_config_manager_defaults():
    """Test that ConfigManager has expected default values."""
    from zapomni_mcp.config import ConfigManager

    config = ConfigManager(
        data_dir=Path("/tmp/test_zapomni_data_defaults"),
        temp_dir=Path("/tmp/test_zapomni_temp_defaults"),
    )

    # Test FalkorDB defaults
    assert config.falkordb_host == "localhost"
    assert config.falkordb_port == 6379
    assert config.graph_name == "zapomni_memory"

    # Test Ollama defaults
    assert config.ollama_base_url == "http://localhost:11434"
    assert config.ollama_embedding_model == "nomic-embed-text"
    assert config.ollama_llm_model == "llama3.1:8b"

    # Test performance defaults
    assert config.max_chunk_size == 512
    assert config.chunk_overlap == 50
    assert config.vector_dimensions == 768


def test_config_manager_computed_properties():
    """Test that ConfigManager has computed properties."""
    from zapomni_mcp.config import ConfigManager

    config = ConfigManager(
        data_dir=Path("/tmp/test_zapomni_data_props"), temp_dir=Path("/tmp/test_zapomni_temp_props")
    )

    # Test computed properties exist and work
    assert hasattr(config, "falkordb_connection_string")
    assert hasattr(config, "redis_connection_string")
    assert hasattr(config, "is_development")
    assert hasattr(config, "is_production")

    # Test connection string format
    conn_str = config.falkordb_connection_string
    assert conn_str.startswith("redis://")
    assert "localhost" in conn_str
    assert "6379" in conn_str


def test_module_all_export():
    """Test that __all__ is properly defined."""
    import zapomni_mcp.config as config_module

    assert hasattr(config_module, "__all__")
    expected_exports = [
        "ConfigManager",
        "ZapomniSettings",
        "get_config_summary",
        "validate_configuration",
    ]

    for export in expected_exports:
        assert export in config_module.__all__, f"{export} not in __all__"


def test_helper_functions_work():
    """Test that re-exported helper functions work correctly."""
    from zapomni_mcp.config import ConfigManager, get_config_summary, validate_configuration

    config = ConfigManager(
        data_dir=Path("/tmp/test_zapomni_data_helpers"),
        temp_dir=Path("/tmp/test_zapomni_temp_helpers"),
    )

    # Test get_config_summary
    summary = get_config_summary(config)
    assert isinstance(summary, dict)
    assert "database" in summary
    assert "ollama" in summary
    assert "performance" in summary

    # Test validate_configuration
    is_valid, errors = validate_configuration(config)
    assert isinstance(is_valid, bool)
    assert isinstance(errors, list)


def test_no_import_errors():
    """Test that importing config module doesn't raise errors."""
    try:
        import zapomni_mcp.config

        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


def test_config_manager_validation():
    """Test that ConfigManager validates configuration."""
    from pydantic import ValidationError

    from zapomni_mcp.config import ConfigManager

    # Test invalid port raises ValidationError
    with pytest.raises(ValidationError):
        ConfigManager(
            falkordb_port=99999,  # Out of valid range
            data_dir=Path("/tmp/test_zapomni_invalid"),
            temp_dir=Path("/tmp/test_zapomni_temp_invalid"),
        )


def test_config_manager_custom_values():
    """Test that ConfigManager accepts custom configuration values."""
    from zapomni_mcp.config import ConfigManager

    config = ConfigManager(
        falkordb_host="custom.db.host",
        falkordb_port=7000,
        max_chunk_size=1024,
        log_level="DEBUG",
        data_dir=Path("/tmp/test_zapomni_custom"),
        temp_dir=Path("/tmp/test_zapomni_temp_custom"),
    )

    assert config.falkordb_host == "custom.db.host"
    assert config.falkordb_port == 7000
    assert config.max_chunk_size == 1024
    assert config.log_level == "DEBUG"
    assert config.is_development is True


def test_import_from_mcp_package():
    """Test that all exports can be imported from zapomni_mcp.config."""
    # This should not raise any ImportError
    from zapomni_mcp.config import (
        ConfigManager,
        ZapomniSettings,
        get_config_summary,
        validate_configuration,
    )

    assert ConfigManager is not None
    assert ZapomniSettings is not None
    assert get_config_summary is not None
    assert validate_configuration is not None
