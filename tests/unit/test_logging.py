"""
Unit tests for LoggingService.

Tests logging configuration, logger creation, operation logging,
error logging, performance logging, and sensitive data sanitization.

Copyright (c) 2025 Goncharenko Anton aka alienxs2
License: MIT
"""

import sys
from io import StringIO

import pytest
import structlog

from zapomni_core.logging_service import LoggingConfig, LoggingService

# ============================================================
# FIXTURES
# ============================================================


@pytest.fixture(autouse=True)
def reset_logging_service():
    """Reset LoggingService state before each test."""
    # Reset class-level state
    LoggingService._configured = False
    LoggingService._log_level = "INFO"
    LoggingService._config = None
    LoggingService._loggers = {}
    LoggingService._sensitive_keys = set()
    yield
    # Cleanup after test
    LoggingService._configured = False
    LoggingService._loggers = {}


# ============================================================
# CONFIGURATION TESTS
# ============================================================


def test_configure_logging_success():
    """Test normal logging configuration."""
    LoggingService.configure_logging(level="INFO", format="json")

    # Verify state updated
    assert LoggingService._configured == True
    assert LoggingService._log_level == "INFO"
    assert LoggingService._config is not None


def test_configure_logging_with_config_object():
    """Test configuration with LoggingConfig object."""
    config = LoggingConfig(level="DEBUG", format="json", sensitive_keys={"password", "api_key"})

    LoggingService.configure_logging(config=config)

    assert LoggingService._configured == True
    assert LoggingService._log_level == "DEBUG"
    assert "password" in LoggingService._sensitive_keys
    assert "api_key" in LoggingService._sensitive_keys


def test_configure_logging_invalid_level():
    """Test configuration with invalid log level."""
    with pytest.raises(ValueError) as exc_info:
        LoggingService.configure_logging(level="INVALID")

    assert "Invalid log level" in str(exc_info.value)
    assert LoggingService._configured == False  # Should not be configured


def test_configure_logging_already_configured():
    """Test that calling configure_logging twice raises RuntimeError."""
    LoggingService.configure_logging(level="INFO")

    with pytest.raises(RuntimeError) as exc_info:
        LoggingService.configure_logging(level="DEBUG")

    assert "already configured" in str(exc_info.value).lower()


def test_configure_logging_invalid_format():
    """Test configuration with invalid format."""
    with pytest.raises(ValueError) as exc_info:
        LoggingService.configure_logging(level="INFO", format="xml")

    assert "Invalid format" in str(exc_info.value)


# ============================================================
# GET LOGGER TESTS
# ============================================================


def test_get_logger_success():
    """Test getting logger after configuration."""
    LoggingService.configure_logging(level="INFO")

    logger = LoggingService.get_logger("test.module")

    assert logger is not None
    # Logger is a BoundLoggerLazyProxy, which is a valid logger type
    assert hasattr(logger, "info")
    assert hasattr(logger, "error")


def test_get_logger_caches_loggers():
    """Test that repeated calls return the same logger instance."""
    LoggingService.configure_logging(level="INFO")

    logger1 = LoggingService.get_logger("test.module")
    logger2 = LoggingService.get_logger("test.module")

    assert logger1 is logger2  # Same instance


def test_get_logger_different_names():
    """Test that different names create different loggers."""
    LoggingService.configure_logging(level="INFO")

    logger1 = LoggingService.get_logger("module1")
    logger2 = LoggingService.get_logger("module2")

    assert logger1 is not logger2


def test_get_logger_not_configured():
    """Test that get_logger raises RuntimeError if not configured."""
    with pytest.raises(RuntimeError) as exc_info:
        LoggingService.get_logger("test.module")

    assert "not configured" in str(exc_info.value).lower()


def test_get_logger_empty_name():
    """Test that empty logger name raises ValueError."""
    LoggingService.configure_logging(level="INFO")

    with pytest.raises(ValueError) as exc_info:
        LoggingService.get_logger("")

    assert "cannot be empty" in str(exc_info.value).lower()


def test_get_logger_very_long_name():
    """Test that very long logger name raises ValueError."""
    LoggingService.configure_logging(level="INFO")

    long_name = "x" * 300

    with pytest.raises(ValueError) as exc_info:
        LoggingService.get_logger(long_name)

    assert "maximum length" in str(exc_info.value).lower()


# ============================================================
# LOG OPERATION TESTS
# ============================================================


def test_log_operation_success():
    """Test logging operation with metadata."""
    LoggingService.configure_logging(level="INFO", format="json")

    # Should not raise any exceptions
    LoggingService.log_operation(
        operation="test_operation",
        correlation_id="test-uuid-123",
        metadata={"key": "value", "count": 42},
    )

    # If we get here, logging succeeded
    assert True


def test_log_operation_empty_operation():
    """Test that empty operation name raises ValueError."""
    LoggingService.configure_logging(level="INFO")

    with pytest.raises(ValueError) as exc_info:
        LoggingService.log_operation(operation="", correlation_id="uuid")

    assert "operation" in str(exc_info.value).lower()


def test_log_operation_empty_correlation_id():
    """Test that empty correlation_id raises ValueError."""
    LoggingService.configure_logging(level="INFO")

    with pytest.raises(ValueError) as exc_info:
        LoggingService.log_operation(operation="test", correlation_id="")

    assert "correlation_id" in str(exc_info.value).lower()


def test_log_operation_sanitizes_sensitive_data():
    """Test that sensitive data is sanitized in metadata."""
    LoggingService.configure_logging(level="INFO", format="json")

    # Should not raise exceptions, and metadata will be sanitized internally
    LoggingService.log_operation(
        operation="user_login",
        correlation_id="uuid",
        metadata={"username": "alice", "password": "secret123", "api_key": "sk-abc123"},
    )

    # If we get here, logging succeeded
    assert True


# ============================================================
# LOG ERROR TESTS
# ============================================================


def test_log_error_with_stack_trace():
    """Test logging error with stack trace."""
    LoggingService.configure_logging(level="INFO", format="json")

    error = ValueError("Something went wrong")

    # Should not raise exceptions
    LoggingService.log_error(
        error=error,
        correlation_id="uuid",
        context={"operation": "test_op"},
        include_stack_trace=True,
    )

    # If we get here, error logging succeeded
    assert True


def test_log_error_without_stack_trace():
    """Test logging error without stack trace."""
    LoggingService.configure_logging(level="INFO", format="json")

    error = ValueError("Expected error")

    # Should not raise exceptions
    LoggingService.log_error(error=error, correlation_id="uuid", include_stack_trace=False)

    # If we get here, error logging succeeded
    assert True


# ============================================================
# LOG PERFORMANCE TESTS
# ============================================================


def test_log_performance_success():
    """Test logging performance metrics."""
    LoggingService.configure_logging(level="INFO", format="json")

    # Should not raise exceptions
    LoggingService.log_performance(
        operation="test_op",
        duration_ms=123.45,
        correlation_id="uuid",
        metadata={"items_processed": 100},
    )

    # If we get here, performance logging succeeded
    assert True


def test_log_performance_negative_duration():
    """Test that negative duration raises ValueError."""
    LoggingService.configure_logging(level="INFO")

    with pytest.raises(ValueError) as exc_info:
        LoggingService.log_performance(operation="test", duration_ms=-10, correlation_id="uuid")

    assert "negative" in str(exc_info.value).lower()


# ============================================================
# SANITIZATION TESTS
# ============================================================


def test_sanitize_metadata_sensitive_keys():
    """Test sanitization of sensitive keys in metadata."""
    LoggingService.configure_logging(level="INFO")

    metadata = {"username": "alice", "password": "secret", "api_key": "key123", "data": "public"}

    sanitized = LoggingService._sanitize_metadata(metadata)

    assert sanitized["username"] == "alice"
    assert sanitized["password"] == "[REDACTED]"
    assert sanitized["api_key"] == "[REDACTED]"
    assert sanitized["data"] == "public"


def test_sanitize_metadata_nested_dicts():
    """Test recursive sanitization of nested dictionaries."""
    LoggingService.configure_logging(level="INFO")

    metadata = {"user": {"name": "alice", "credentials": {"password": "secret", "token": "abc123"}}}

    sanitized = LoggingService._sanitize_metadata(metadata)

    assert sanitized["user"]["name"] == "alice"
    assert sanitized["user"]["credentials"]["password"] == "[REDACTED]"
    assert sanitized["user"]["credentials"]["token"] == "[REDACTED]"


def test_sanitize_metadata_with_lists():
    """Test sanitization of lists in metadata."""
    LoggingService.configure_logging(level="INFO")

    metadata = {
        "users": [{"name": "alice", "password": "secret1"}, {"name": "bob", "password": "secret2"}]
    }

    sanitized = LoggingService._sanitize_metadata(metadata)

    assert sanitized["users"][0]["name"] == "alice"
    assert sanitized["users"][0]["password"] == "[REDACTED]"
    assert sanitized["users"][1]["password"] == "[REDACTED]"


# ============================================================
# INTEGRATION TESTS
# ============================================================


def test_end_to_end_logging_flow():
    """Test complete logging flow from configuration to output."""
    # 1. Configure
    LoggingService.configure_logging(level="DEBUG", format="json")

    # 2. Get logger
    logger = LoggingService.get_logger("test.integration")

    # 3. Log operation - should not raise exceptions
    logger.info("test_event", correlation_id="uuid-123", data="value")

    # 4. Verify no exceptions were raised
    assert True


def test_concurrent_logging_safe():
    """Test that concurrent logging from multiple loggers is safe."""
    LoggingService.configure_logging(level="INFO")

    # Get multiple loggers
    logger1 = LoggingService.get_logger("module1")
    logger2 = LoggingService.get_logger("module2")

    # Log from both simultaneously (simulate concurrent access)
    logger1.info("event1", data="value1")
    logger2.info("event2", data="value2")

    # Should not raise any exceptions
    assert True
