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

import os
from dataclasses import dataclass, field
from typing import List

from zapomni_core.config import ZapomniSettings, get_config_summary, validate_configuration
from zapomni_core.exceptions import ValidationError

# Alias for MCP layer - makes it clear this is the configuration manager
ConfigManager = ZapomniSettings


# Default allowed hosts for DNS rebinding protection (localhost variants)
DEFAULT_ALLOWED_HOSTS: List[str] = ["localhost", "127.0.0.1", "::1"]


@dataclass
class SSEConfig:
    """
    Configuration for SSE transport.

    Attributes:
        host: SSE server bind address (default: 127.0.0.1 for local only)
        port: SSE server port (default: 8000)
        cors_origins: List of allowed CORS origins (default: ["*"])
        heartbeat_interval: SSE heartbeat interval in seconds (default: 30)
        max_connection_lifetime: Maximum SSE connection lifetime in seconds (default: 3600)
        allowed_hosts: List of allowed Host header values for DNS rebinding protection.
            When binding to localhost (127.0.0.1 or localhost), defaults to
            ["localhost", "127.0.0.1", "::1"]. For non-localhost binding,
            this must be explicitly configured.
        dns_rebinding_protection: Whether to enable DNS rebinding protection (default: True)
    """

    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    heartbeat_interval: int = 30
    max_connection_lifetime: int = 3600
    allowed_hosts: List[str] = field(default_factory=list)
    dns_rebinding_protection: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not 1 <= self.port <= 65535:
            raise ValidationError(f"port must be between 1 and 65535, got {self.port}")

        if self.heartbeat_interval < 5:
            raise ValidationError(
                f"heartbeat_interval must be at least 5 seconds, got {self.heartbeat_interval}"
            )

        if self.max_connection_lifetime < 60:
            raise ValidationError(
                f"max_connection_lifetime must be at least 60 seconds, "
                f"got {self.max_connection_lifetime}"
            )

        # Set default allowed_hosts based on bind address if not explicitly provided
        if not self.allowed_hosts:
            if self._is_localhost_binding():
                self.allowed_hosts = DEFAULT_ALLOWED_HOSTS.copy()
            else:
                # For non-localhost binding, require explicit configuration
                # if DNS rebinding protection is enabled
                if self.dns_rebinding_protection:
                    raise ValidationError(
                        f"allowed_hosts must be explicitly configured when binding to "
                        f"non-localhost address '{self.host}' with dns_rebinding_protection enabled. "
                        f"Example: allowed_hosts=['yourdomain.com', 'api.yourdomain.com']"
                    )

    def _is_localhost_binding(self) -> bool:
        """Check if the host is a localhost variant."""
        localhost_variants = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}
        return self.host.lower() in localhost_variants

    def get_effective_allowed_hosts(self) -> List[str]:
        """
        Get the effective list of allowed hosts for DNS rebinding protection.

        Returns:
            List of allowed hosts. Empty list if protection is disabled.
        """
        if not self.dns_rebinding_protection:
            return []
        return self.allowed_hosts

    @classmethod
    def from_env(cls) -> "SSEConfig":
        """
        Create SSEConfig from environment variables.

        Environment Variables:
            ZAPOMNI_SSE_HOST: SSE server bind address (default: 127.0.0.1)
            ZAPOMNI_SSE_PORT: SSE server port (default: 8000)
            ZAPOMNI_SSE_CORS_ORIGINS: Comma-separated CORS origins (default: *)
            ZAPOMNI_SSE_HEARTBEAT_INTERVAL: Heartbeat interval in seconds (default: 30)
            ZAPOMNI_SSE_MAX_CONNECTION_LIFETIME: Max connection lifetime in seconds (default: 3600)
            ZAPOMNI_SSE_ALLOWED_HOSTS: Comma-separated allowed hosts for DNS rebinding protection
                (default: auto-configured based on host)
            ZAPOMNI_SSE_DNS_REBINDING_PROTECTION: Enable DNS rebinding protection (default: true)

        Returns:
            SSEConfig instance populated from environment variables
        """
        host = os.getenv("ZAPOMNI_SSE_HOST", "127.0.0.1")
        port = int(os.getenv("ZAPOMNI_SSE_PORT", "8000"))

        # Parse comma-separated CORS origins
        cors_origins_str = os.getenv("ZAPOMNI_SSE_CORS_ORIGINS", "*")
        cors_origins = [origin.strip() for origin in cors_origins_str.split(",")]

        heartbeat_interval = int(os.getenv("ZAPOMNI_SSE_HEARTBEAT_INTERVAL", "30"))
        max_connection_lifetime = int(os.getenv("ZAPOMNI_SSE_MAX_CONNECTION_LIFETIME", "3600"))

        # Parse allowed hosts for DNS rebinding protection
        allowed_hosts_str = os.getenv("ZAPOMNI_SSE_ALLOWED_HOSTS", "")
        allowed_hosts = (
            [h.strip() for h in allowed_hosts_str.split(",") if h.strip()]
            if allowed_hosts_str
            else []
        )

        # Parse DNS rebinding protection flag (default: true)
        dns_rebinding_protection_str = os.getenv(
            "ZAPOMNI_SSE_DNS_REBINDING_PROTECTION", "true"
        ).lower()
        dns_rebinding_protection = dns_rebinding_protection_str in ("true", "1", "yes", "on")

        return cls(
            host=host,
            port=port,
            cors_origins=cors_origins,
            heartbeat_interval=heartbeat_interval,
            max_connection_lifetime=max_connection_lifetime,
            allowed_hosts=allowed_hosts,
            dns_rebinding_protection=dns_rebinding_protection,
        )


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

    def __post_init__(self) -> None:
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
    "SSEConfig",
    "DEFAULT_ALLOWED_HOSTS",
    "get_config_summary",
    "validate_configuration",
]
