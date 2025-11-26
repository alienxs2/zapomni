"""
Logger Factory - Convenience wrapper for LoggingService.

Provides a simple get_logger() function that wraps LoggingService.get_logger()
for convenient structured logging access throughout the application.

This module serves as a utility layer that simplifies the import path
and provides a familiar factory pattern for obtaining loggers.

Copyright (c) 2025 Goncharenko Anton aka alienxs2
License: MIT
"""

from typing import Optional

import structlog

from zapomni_core.config import settings
from zapomni_core.logging_service import LoggingService


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a module/component-specific logger.

    This is a convenience wrapper around LoggingService.get_logger() that
    provides a simpler import path and familiar factory pattern.

    The function returns a cached logger if already created, otherwise creates
    a new logger with the given name for context enrichment.

    Logger names typically follow Python module path convention:
    "zapomni.mcp.tools.add_memory"

    Args:
        name: Logger name (typically module path or __name__)

    Returns:
        BoundLogger instance with automatic context enrichment,
        structured logging support, and JSON output formatting

    Raises:
        RuntimeError: If logging not configured yet (call configure_logging() first)
        ValueError: If name is empty or exceeds maximum length (200 chars)

    Example:
        ```python
        from zapomni_core.utils import get_logger

        # In a module
        logger = get_logger(__name__)

        # Log structured data
        logger.info(
            "operation_started",
            correlation_id="uuid-123",
            input_size=1024,
            user_id="user-456"
        )

        # Log with different levels
        logger.debug("debug_info", step=1)
        logger.warning("rate_limit_approaching", current=95, limit=100)
        logger.error("operation_failed", error_code="ERR_001")
        ```

    Notes:
        - Loggers are cached for performance (dict lookup O(1))
        - Output goes to stderr (MCP compatible, doesn't interfere with stdout)
        - All logs are JSON formatted by default (machine-readable)
        - Sensitive data is automatically sanitized (passwords, tokens, etc.)
        - Thread-safe for concurrent async operations
    """
    # Delegate to LoggingService
    return LoggingService.get_logger(name)


def configure_logging(level: Optional[str] = None, format: str = "json") -> None:
    """
    Configure structured logging infrastructure.

    Convenience wrapper for LoggingService.configure_logging() that
    automatically uses settings.log_level if no level is provided.

    This should be called ONCE at application startup before any logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               If None, uses settings.log_level from configuration.
        format: Output format ("json" or "console" for development).
                Default is "json" for production use.

    Raises:
        ValueError: If level or format is invalid
        RuntimeError: If called after logging already configured

    Example:
        ```python
        from zapomni_core.utils import configure_logging, get_logger

        # At application startup
        configure_logging()  # Uses settings.log_level

        # Or with explicit level
        configure_logging(level="DEBUG", format="console")

        # Then get loggers
        logger = get_logger(__name__)
        ```

    Notes:
        - If level is None, automatically reads from settings.log_level
        - This provides zero-config logging based on environment variables
        - Calling twice will raise RuntimeError to prevent misconfiguration
    """
    # Use settings.log_level if not provided
    if level is None:
        level = settings.log_level

    # Delegate to LoggingService
    LoggingService.configure_logging(level=level, format=format)
