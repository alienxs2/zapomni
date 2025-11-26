"""
Unit tests for logger_factory utility.

Tests the convenience wrapper for LoggingService that provides
easy access to structured logging throughout the application.

Copyright (c) 2025 Goncharenko Anton aka alienxs2
License: MIT
"""

import pytest
import structlog

from zapomni_core.logging_service import LoggingService
from zapomni_core.utils import get_logger

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


@pytest.fixture
def configured_logging():
    """Fixture that configures logging before test."""
    LoggingService.configure_logging(level="INFO", format="json")
    yield
    # Cleanup handled by reset_logging_service


# ============================================================
# LOGGER CREATION TESTS
# ============================================================


def test_get_logger_returns_bound_logger(configured_logging):
    """Test that get_logger returns a valid BoundLogger instance."""
    logger = get_logger("test.module")

    assert logger is not None
    assert hasattr(logger, "info")
    assert hasattr(logger, "debug")
    assert hasattr(logger, "warning")
    assert hasattr(logger, "error")
    assert hasattr(logger, "critical")


def test_get_logger_with_module_name(configured_logging):
    """Test getting logger with __name__ pattern."""
    logger = get_logger("zapomni.core.chunking")

    assert logger is not None


def test_get_logger_caches_instances(configured_logging):
    """Test that multiple calls with same name return cached instance."""
    logger1 = get_logger("test.module")
    logger2 = get_logger("test.module")

    assert logger1 is logger2


def test_get_logger_different_names_different_instances(configured_logging):
    """Test that different names create different logger instances."""
    logger1 = get_logger("module.one")
    logger2 = get_logger("module.two")

    assert logger1 is not logger2


# ============================================================
# CONFIGURATION TESTS
# ============================================================


def test_get_logger_requires_configuration():
    """Test that get_logger raises error if logging not configured."""
    with pytest.raises(RuntimeError) as exc_info:
        get_logger("test.module")

    assert "not configured" in str(exc_info.value).lower()


def test_get_logger_respects_log_level(configured_logging):
    """Test that loggers respect the configured log level."""
    # Reconfigure with DEBUG level
    LoggingService._configured = False
    LoggingService.configure_logging(level="DEBUG", format="json")

    logger = get_logger("test.debug")

    # Logger should have debug method available
    assert hasattr(logger, "debug")


# ============================================================
# STRUCTURED LOGGING OUTPUT TESTS
# ============================================================


def test_logger_supports_structured_logging(configured_logging):
    """Test that logger supports structured key-value logging."""
    logger = get_logger("test.structured")

    # Should not raise exception with structured args
    logger.info("test_event", correlation_id="test-uuid-123", user_id="user-456", count=42)

    assert True  # If we reach here, logging succeeded


def test_logger_handles_nested_context(configured_logging):
    """Test that logger handles nested context dictionaries."""
    logger = get_logger("test.nested")

    # Should handle nested structures
    logger.info(
        "complex_event",
        metadata={
            "user": {"id": "123", "name": "Alice"},
            "stats": {"count": 10, "duration_ms": 150.5},
        },
    )

    assert True


# ============================================================
# MULTIPLE LOGGER INSTANCES TESTS
# ============================================================


def test_multiple_loggers_independent_context(configured_logging):
    """Test that multiple loggers maintain independent contexts."""
    logger1 = get_logger("module.one")
    logger2 = get_logger("module.two")

    # Each logger should maintain its own context
    logger1.info("event_one", data="value1")
    logger2.info("event_two", data="value2")

    # Both should work independently without interference
    assert True


def test_many_logger_instances(configured_logging):
    """Test creating many logger instances for different modules."""
    loggers = []

    for i in range(10):
        logger = get_logger(f"module.test_{i}")
        loggers.append(logger)

    # All loggers should be valid
    assert len(loggers) == 10

    # Each should be unique (except duplicates)
    logger_duplicate = get_logger("module.test_0")
    assert logger_duplicate is loggers[0]


# ============================================================
# VALIDATION TESTS
# ============================================================


def test_get_logger_empty_name_raises_error(configured_logging):
    """Test that empty logger name raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        get_logger("")

    assert "cannot be empty" in str(exc_info.value).lower()


def test_get_logger_very_long_name_raises_error(configured_logging):
    """Test that overly long logger name raises ValueError."""
    long_name = "x" * 300

    with pytest.raises(ValueError) as exc_info:
        get_logger(long_name)

    assert "maximum length" in str(exc_info.value).lower()


# ============================================================
# INTEGRATION TESTS
# ============================================================


def test_end_to_end_logger_factory_usage(configured_logging):
    """Test complete workflow: configure, get logger, log messages."""
    # Get logger
    logger = get_logger("integration.test")

    # Log at different levels
    logger.debug("debug_message", step=1)
    logger.info("info_message", step=2)
    logger.warning("warning_message", step=3)
    logger.error("error_message", step=4)

    # Should complete without exceptions
    assert True


def test_logger_factory_with_settings_integration():
    """Test that logger_factory works with settings from config."""
    from zapomni_core.config import settings

    # Configure using settings.log_level
    LoggingService.configure_logging(level=settings.log_level, format="json")

    logger = get_logger("settings.integration")

    logger.info("configured_from_settings", level=settings.log_level)

    assert True


# ============================================================
# PERFORMANCE TESTS
# ============================================================


def test_get_logger_performance_cached(configured_logging):
    """Test that cached logger access is fast."""
    import time

    # First call (creates logger)
    start = time.perf_counter()
    logger = get_logger("performance.test")
    first_call_duration = time.perf_counter() - start

    # Subsequent calls (cached)
    start = time.perf_counter()
    for _ in range(100):
        get_logger("performance.test")
    cached_duration = time.perf_counter() - start

    # Cached access should be much faster
    # Average per call should be well under 1ms
    avg_cached_time = (cached_duration / 100) * 1000  # Convert to ms

    assert avg_cached_time < 1.0  # Less than 1ms per cached call
