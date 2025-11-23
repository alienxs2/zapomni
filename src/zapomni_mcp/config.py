"""
ConfigManager re-export from zapomni_core.

This module provides the ConfigManager class for the zapomni_mcp layer
by re-exporting ZapomniSettings from zapomni_core.config. It also re-exports
helper functions for configuration management.

The ConfigManager alias makes it clear that this is the configuration
manager component while maintaining compatibility with the core implementation.

Copyright (c) 2025 Goncharenko Anton aka alienxs2
License: MIT
"""

from zapomni_core.config import ZapomniSettings, get_config_summary, validate_configuration

# Alias for MCP layer - makes it clear this is the configuration manager
ConfigManager = ZapomniSettings

__all__ = ["ConfigManager", "ZapomniSettings", "get_config_summary", "validate_configuration"]
