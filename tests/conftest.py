"""
Pytest configuration and fixtures for all tests.

Provides shared setup/teardown and fixtures for logging, database, etc.

Copyright (c) 2025 Goncharenko Anton aka alienxs2
License: MIT
"""

import pytest
from zapomni_core.logging_service import LoggingService


def pytest_configure(config):
    """Configure logging before any tests are collected."""
    # Configure logging at session start
    LoggingService.configure_logging(level="DEBUG", format="json")


@pytest.fixture(autouse=True)
def reset_logging_service():
    """Reset LoggingService state before each test."""
    # Reset class-level state
    LoggingService._configured = False
    LoggingService._log_level = "INFO"
    LoggingService._config = None
    LoggingService._loggers = {}
    LoggingService._sensitive_keys = set()

    # Configure for tests
    LoggingService.configure_logging(level="DEBUG", format="json")

    yield

    # Cleanup after test
    LoggingService._configured = False
    LoggingService._loggers = {}
