"""
Zapomni Core Layer.

Middle layer in dependency hierarchy. Contains:
- Business logic and processing
- Exception hierarchy
- Configuration management
- Logging service
- Chunking, embedding, extraction
- Search algorithms

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from .exceptions import (
    ZapomniError,
    ValidationError,
    ProcessingError,
    EmbeddingError,
    ExtractionError,
    SearchError,
    DatabaseError,
    ConnectionError,
    QueryError,
    TimeoutError,
)

from .config import (
    ZapomniSettings,
    settings,
    get_config_summary,
    validate_configuration,
)

from .logging_service import (
    LoggingService,
    LoggingConfig,
)

__all__ = [
    # Exceptions
    "ZapomniError",
    "ValidationError",
    "ProcessingError",
    "EmbeddingError",
    "ExtractionError",
    "SearchError",
    "DatabaseError",
    "ConnectionError",
    "QueryError",
    "TimeoutError",
    # Configuration
    "ZapomniSettings",
    "settings",
    "get_config_summary",
    "validate_configuration",
    # Logging
    "LoggingService",
    "LoggingConfig",
]
