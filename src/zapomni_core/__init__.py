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

from .config import (
    ZapomniSettings,
    get_config_summary,
    settings,
    validate_configuration,
)
from .exceptions import (
    ConnectionError,
    DatabaseError,
    EmbeddingError,
    ExtractionError,
    ProcessingError,
    QueryError,
    SearchError,
    TimeoutError,
    ValidationError,
    ZapomniError,
)
from .logging_service import (
    LoggingConfig,
    LoggingService,
)
from .runtime_config import RuntimeConfig


def __getattr__(name):
    """Lazy import for components to avoid circular imports."""
    if name == "code_processor":
        from .processors import code_processor

        return code_processor
    elif name == "MemoryProcessor":
        from .memory_processor import MemoryProcessor

        return MemoryProcessor
    elif name == "ProcessorConfig":
        from .memory_processor import ProcessorConfig

        return ProcessorConfig
    elif name == "SearchResultItem":
        from .memory_processor import SearchResultItem

        return SearchResultItem
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    "RuntimeConfig",
    # Logging
    "LoggingService",
    "LoggingConfig",
    # Memory Processor
    "MemoryProcessor",
    "ProcessorConfig",
    "SearchResultItem",
    # Processors
    "code_processor",
]
