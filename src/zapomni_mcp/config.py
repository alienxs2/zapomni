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

from dataclasses import dataclass
from zapomni_core.config import ZapomniSettings, get_config_summary, validate_configuration
from zapomni_core.exceptions import ValidationError

# Alias for MCP layer - makes it clear this is the configuration manager
ConfigManager = ZapomniSettings


@dataclass
class Settings:
    """
    MCP Server configuration settings.

    Attributes:
        server_name: Name of the MCP server (default: "zapomni-memory")
        version: Server version string (default: "0.1.0")
        log_level: Logging level (default: "INFO")
        max_concurrent_tasks: Max concurrent background tasks (default: 4)
        request_timeout_seconds: Request timeout in seconds (default: 300)
    """

    server_name: str = "zapomni-memory"
    version: str = "0.1.0"
    log_level: str = "INFO"
    max_concurrent_tasks: int = 4
    request_timeout_seconds: int = 300

    def __post_init__(self):
        """Validate configuration values."""
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_log_levels:
            raise ValidationError(
                f"log_level must be one of {valid_log_levels}, got '{self.log_level}'"
            )

        if self.max_concurrent_tasks <= 0:
            raise ValidationError("max_concurrent_tasks must be positive")

        if self.request_timeout_seconds <= 0:
            raise ValidationError("request_timeout_seconds must be positive")


__all__ = [
    "ConfigManager",
    "ZapomniSettings",
    "Settings",
    "get_config_summary",
    "validate_configuration",
]
