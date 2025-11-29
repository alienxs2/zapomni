"""
LoggingService - Centralized structured logging for Zapomni.

Provides consistent, context-enriched, machine-readable logging
across all modules using structlog.

Copyright (c) 2025 Goncharenko Anton aka alienxs2
License: MIT
"""

import logging
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import structlog
from structlog.types import Processor


@dataclass
class LoggingConfig:
    """
    Configuration for LoggingService.

    Attributes:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Output format ("json" or "console" for dev)
        output_stream: Output destination (default: sys.stderr)
        sanitize_sensitive: Whether to sanitize sensitive data
        sensitive_keys: Set of keys to sanitize (e.g., "password", "api_key")

    Example:
        config = LoggingConfig(
            level="INFO",
            format="json",
            sensitive_keys={"password", "api_key", "token"}
        )
    """

    level: str = "INFO"
    format: str = "json"  # "json" or "console"
    output_stream: Any = sys.stderr
    sanitize_sensitive: bool = True
    sensitive_keys: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if not self.sensitive_keys:
            self.sensitive_keys = {
                "password",
                "passwd",
                "pwd",
                "api_key",
                "apikey",
                "key",
                "token",
                "access_token",
                "refresh_token",
                "secret",
                "auth",
                "authorization",
                "credit_card",
                "ssn",
                "social_security",
            }


class LoggingService:
    """
    Centralized structured logging service using structlog.

    Provides consistent, context-enriched, machine-readable logging
    across all Zapomni modules. Outputs JSON logs to stderr (MCP-compatible).

    Features:
        - Structured JSON logging
        - Automatic context enrichment (timestamps, correlation IDs)
        - Module-specific loggers
        - Sensitive data sanitization
        - Thread-safe operation
        - Performance metric logging

    Example:
        # Setup logging once at startup
        LoggingService.configure_logging(level="INFO", format="json")

        # Get logger for a module
        logger = LoggingService.get_logger("zapomni.mcp.tools")

        # Log operations
        logger.info(
            "tool_executed",
            tool_name="add_memory",
            correlation_id="uuid-here",
            duration_ms=123
        )
    """

    # Class-level state
    _configured: bool = False
    _log_level: str = "INFO"
    _config: Optional[LoggingConfig] = None
    _loggers: dict[str, structlog.BoundLogger] = {}
    _sensitive_keys: set[str] = set()

    @classmethod
    def configure_logging(
        cls, level: str = "INFO", format: str = "json", config: Optional[LoggingConfig] = None
    ) -> None:
        """
        Configure global structured logging infrastructure.

        Sets up structlog with JSON output to stderr, configures log level,
        and initializes processors for context enrichment.

        This should be called ONCE at application startup before any logging.

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            format: Output format ("json" or "console")
            config: Optional LoggingConfig for advanced configuration

        Raises:
            ValueError: If level or format is invalid
            RuntimeError: If called after logging already configured

        Example:
            # Simple usage
            LoggingService.configure_logging(level="DEBUG", format="json")

            # Advanced usage
            config = LoggingConfig(
                level="INFO",
                format="json",
                sensitive_keys={"password", "secret"}
            )
            LoggingService.configure_logging(config=config)
        """
        # Check if already configured
        if cls._configured:
            raise RuntimeError("Logging already configured")

        # Use provided config or create from parameters
        if config is not None:
            cfg = config
        else:
            # Validate level
            level_upper = level.upper()
            if level_upper not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
                raise ValueError(
                    f"Invalid log level: {level}. "
                    "Must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL"
                )

            # Validate format
            format_lower = format.lower()
            if format_lower not in ["json", "console"]:
                raise ValueError(f"Invalid format: {format}. Must be 'json' or 'console'")

            cfg = LoggingConfig(level=level_upper, format=format_lower)

        # Store config
        cls._config = cfg
        cls._log_level = cfg.level
        cls._sensitive_keys = cfg.sensitive_keys

        # Setup processors
        processors = cls._setup_processors()

        # Configure structlog
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, cfg.level)),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(file=cfg.output_stream),
            cache_logger_on_first_use=True,
        )

        # Mark as configured
        cls._configured = True

    @classmethod
    def get_logger(cls, name: str) -> structlog.BoundLogger:
        """
        Get a module/component-specific logger.

        Returns a cached logger if already created, otherwise creates
        a new logger with the given name for context.

        Args:
            name: Logger name (typically module path)

        Returns:
            BoundLogger instance with automatic context enrichment

        Raises:
            RuntimeError: If logging not configured yet
            ValueError: If name is empty or too long

        Example:
            logger = LoggingService.get_logger(__name__)
            logger.info("processing_started", correlation_id="uuid")
        """
        # Check if configured
        if not cls._configured:
            raise RuntimeError("Logging not configured. Call configure_logging() first.")

        # Validate name
        if not name:
            raise ValueError("Logger name cannot be empty")

        if len(name) > 200:
            raise ValueError("Logger name exceeds maximum length (200)")

        # Check cache
        if name in cls._loggers:
            return cls._loggers[name]

        # Create new logger
        logger = structlog.get_logger(name)
        cls._loggers[name] = logger

        return logger

    @classmethod
    def log_operation(
        cls,
        operation: str,
        correlation_id: str,
        metadata: Optional[dict[str, Any]] = None,
        logger_name: str = "zapomni",
        level: str = "info",
    ) -> None:
        """
        Log an operation with full context.

        Convenience method for logging operations with standardized structure.
        Automatically sanitizes sensitive data in metadata.

        Args:
            operation: Operation name (e.g., "add_memory", "search_memory")
            correlation_id: UUID for tracing this operation
            metadata: Additional context (parameters, results, metrics)
            logger_name: Which logger to use (default: "zapomni")
            level: Log level (default: "info")

        Raises:
            ValueError: If operation or correlation_id is empty

        Example:
            LoggingService.log_operation(
                operation="chunk_document",
                correlation_id="uuid-here",
                metadata={
                    "document_length": 5000,
                    "chunk_size": 512,
                    "chunks_created": 10
                }
            )
        """
        # Validate parameters
        if not operation:
            raise ValueError("operation cannot be empty")

        if not correlation_id:
            raise ValueError("correlation_id cannot be empty")

        # Get logger
        logger = cls.get_logger(logger_name)

        # Sanitize metadata
        if metadata:
            metadata = cls._sanitize_metadata(metadata)

        # Build context
        context = {
            "operation": operation,
            "correlation_id": correlation_id,
        }

        if metadata:
            context.update(metadata)

        # Log at specified level
        log_method = getattr(logger, level.lower())
        log_method(operation, **context)

    @classmethod
    def log_error(
        cls,
        error: Exception,
        correlation_id: str,
        context: Optional[dict[str, Any]] = None,
        logger_name: str = "zapomni",
        include_stack_trace: bool = True,
    ) -> None:
        """
        Log an error with full context and stack trace.

        Convenience method for consistent error logging. Automatically
        extracts error type, message, error code (if ZapomniError),
        and optionally includes stack trace.

        Args:
            error: Exception instance
            correlation_id: UUID for tracing
            context: Additional context about where/why error occurred
            logger_name: Which logger to use (default: "zapomni")
            include_stack_trace: Whether to include full stack trace

        Raises:
            ValueError: If correlation_id is empty

        Example:
            try:
                result = dangerous_operation()
            except DatabaseError as e:
                LoggingService.log_error(
                    error=e,
                    correlation_id="uuid-here",
                    context={"operation": "insert_memory"}
                )
                raise
        """
        # Validate
        if not correlation_id:
            raise ValueError("correlation_id cannot be empty")

        # Get logger
        logger = cls.get_logger(logger_name)

        # Extract error details
        error_type = type(error).__name__
        error_message = str(error)
        error_code = getattr(error, "error_code", None)

        # Build log context
        log_context = {
            "error_type": error_type,
            "error_message": error_message,
            "correlation_id": correlation_id,
        }

        if error_code:
            log_context["error_code"] = error_code

        if context:
            log_context.update(cls._sanitize_metadata(context))

        # Log error
        if include_stack_trace:
            log_context["stack_trace"] = traceback.format_exc()
            logger.error("error_occurred", **log_context)
        else:
            logger.error("error_occurred", **log_context)

    @classmethod
    def log_performance(
        cls,
        operation: str,
        duration_ms: float,
        correlation_id: str,
        metadata: Optional[dict[str, Any]] = None,
        logger_name: str = "zapomni",
    ) -> None:
        """
        Log performance metrics for an operation.

        Records operation duration and optional resource usage metrics.
        Useful for identifying slow operations and optimization opportunities.

        Args:
            operation: Operation name
            duration_ms: Duration in milliseconds
            correlation_id: UUID for tracing
            metadata: Additional metrics (memory usage, CPU, etc.)
            logger_name: Which logger to use

        Raises:
            ValueError: If operation/correlation_id empty or duration_ms < 0

        Example:
            import time

            start = time.perf_counter()
            result = embed_texts(texts)
            duration_ms = (time.perf_counter() - start) * 1000

            LoggingService.log_performance(
                operation="embed_texts",
                duration_ms=duration_ms,
                correlation_id="uuid-here",
                metadata={"text_count": len(texts)}
            )
        """
        # Validate
        if not operation:
            raise ValueError("operation cannot be empty")

        if not correlation_id:
            raise ValueError("correlation_id cannot be empty")

        if duration_ms < 0:
            raise ValueError("duration_ms cannot be negative")

        # Get logger
        logger = cls.get_logger(logger_name)

        # Build context
        context = {
            "operation": operation,
            "duration_ms": duration_ms,
            "correlation_id": correlation_id,
        }

        if metadata:
            context.update(cls._sanitize_metadata(metadata))

        # Log performance
        logger.info("performance_metric", **context)

    @classmethod
    def _sanitize_metadata(cls, data: dict[str, Any]) -> dict[str, Any]:
        """
        Sanitize sensitive data from metadata before logging.

        Replaces values for sensitive keys with "[REDACTED]".
        Recursively processes nested dictionaries and lists.

        Args:
            data: Metadata dictionary to sanitize

        Returns:
            Sanitized copy of metadata

        Example:
            metadata = {
                "username": "alice",
                "password": "secret123",
                "api_key": "sk-abc123"
            }

            sanitized = LoggingService._sanitize_metadata(metadata)
            # Returns:
            # {
            #     "username": "alice",
            #     "password": "[REDACTED]",
            #     "api_key": "[REDACTED]"
            # }
        """
        if not isinstance(data, dict):
            return data

        sanitized: Dict[str, Any] = {}

        for key, value in data.items():
            # Check if key is sensitive
            if key.lower() in cls._sensitive_keys:
                sanitized[key] = "[REDACTED]"
            # Recursively sanitize nested dicts
            elif isinstance(value, dict):
                sanitized[key] = cls._sanitize_metadata(value)
            # Recursively sanitize lists
            elif isinstance(value, list):
                sanitized[key] = [
                    cls._sanitize_metadata(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                sanitized[key] = value

        return sanitized

    @classmethod
    def _setup_processors(cls) -> list[Processor]:
        """
        Setup structlog processors based on configuration.

        Returns:
            List of structlog processors for log processing pipeline

        Processors (in order):
            1. add_log_level: Add log level to context
            2. TimeStamper: Add ISO timestamp
            3. StackInfoRenderer: Render stack info if requested
            4. format_exc_info: Format exception info
            5. JSONRenderer or ConsoleRenderer: Final output format
        """
        processors: list[Processor] = [
            # Add log level
            structlog.stdlib.add_log_level,
            # Add timestamp
            structlog.processors.TimeStamper(fmt="iso"),
            # Stack info renderer
            structlog.processors.StackInfoRenderer(),
            # Exception formatter
            structlog.processors.format_exc_info,
        ]

        # Add renderer based on format
        if cls._config and cls._config.format == "console":
            processors.append(structlog.dev.ConsoleRenderer(colors=True))
        else:
            # Default to JSON
            processors.append(structlog.processors.JSONRenderer())

        return processors
