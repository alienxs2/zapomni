"""
Exception hierarchy for Zapomni.

Defines all exception types with error codes, transient flags, and correlation IDs.
Follows the error handling strategy from error_handling_strategy.md.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

import uuid
from typing import Any, Dict, Optional


class ZapomniError(Exception):
    """
    Base exception for all Zapomni errors.

    All Zapomni exceptions inherit from this class. Provides standard
    error attributes: message, error_code, details, correlation_id.

    Attributes:
        message: Human-readable error message
        error_code: Programmatic error code (e.g., "ERR_001")
        details: Additional context (dict)
        correlation_id: UUID for tracing across layers
        original_exception: Wrapped exception (if any)
        is_transient: Whether error is transient (retryable)

    Example:
        raise ZapomniError(
            message="Operation failed",
            error_code="ERR_UNKNOWN",
            details={"param": "value"},
            correlation_id=uuid.uuid4()
        )
    """

    def __init__(
        self,
        message: str,
        error_code: str = "ERR_UNKNOWN",
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        original_exception: Optional[Exception] = None,
    ):
        """
        Initialize ZapomniError.

        Args:
            message: Error message
            error_code: Error code for programmatic handling
            details: Additional context dict
            correlation_id: UUID for request tracing
            original_exception: Original wrapped exception
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.original_exception = original_exception
        self.is_transient = False  # Default: not retryable

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary for logging/serialization.

        Returns:
            Dictionary with all error information
        """
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
            "correlation_id": self.correlation_id,
            "original_error": str(self.original_exception) if self.original_exception else None,
        }


# === Core Layer Exceptions ===


class ValidationError(ZapomniError):
    """
    Raised when input validation fails.

    Error Codes:
        VAL_001: Missing required field
        VAL_002: Invalid field type
        VAL_003: Field value out of range
        VAL_004: Invalid field format

    Not transient (user input errors should not be retried).
    """

    def __init__(self, message: str, error_code: str = "VAL_001", **kwargs):
        super().__init__(message=message, error_code=error_code, **kwargs)
        self.is_transient = False


class ProcessingError(ZapomniError):
    """
    Raised when document processing fails.

    Error Codes:
        PROC_001: Chunking failed
        PROC_002: Text extraction failed
        PROC_003: Invalid document format

    Not transient by default (processing logic errors).
    """

    def __init__(self, message: str, error_code: str = "PROC_001", **kwargs):
        super().__init__(message=message, error_code=error_code, **kwargs)
        self.is_transient = False


class EmbeddingError(ZapomniError):
    """
    Raised when embedding generation fails.

    Error Codes:
        EMB_001: Ollama connection failed
        EMB_002: Embedding timeout
        EMB_003: Invalid embedding dimensions
        EMB_004: Model not found

    Transient (network/timeout issues are retryable).
    """

    def __init__(self, message: str, error_code: str = "EMB_001", **kwargs):
        super().__init__(message=message, error_code=error_code, **kwargs)
        self.is_transient = True  # Retryable


class ExtractionError(ZapomniError):
    """
    Raised when entity/relationship extraction fails.

    Error Codes:
        EXTR_001: Entity extraction failed
        EXTR_002: Relationship detection failed
        EXTR_003: LLM response parsing failed

    Transient (LLM errors often transient).
    """

    def __init__(self, message: str, error_code: str = "EXTR_001", **kwargs):
        super().__init__(message=message, error_code=error_code, **kwargs)
        self.is_transient = True  # Retryable


class SearchError(ZapomniError):
    """
    Raised when search operation fails.

    Error Codes:
        SEARCH_001: Vector search failed
        SEARCH_002: BM25 search failed
        SEARCH_003: Reranking failed
        SEARCH_004: No results found (not an error, but info)

    Not transient by default.
    """

    def __init__(self, message: str, error_code: str = "SEARCH_001", **kwargs):
        super().__init__(message=message, error_code=error_code, **kwargs)
        self.is_transient = False


# === Database Layer Exceptions ===


class DatabaseError(ZapomniError):
    """
    Base exception for database operations.

    Not transient by default (most DB errors are not retryable).
    """

    def __init__(self, message: str, error_code: str = "DB_001", **kwargs):
        super().__init__(message=message, error_code=error_code, **kwargs)
        self.is_transient = False


class ConnectionError(DatabaseError):
    """
    Raised when database connection fails.

    Error Codes:
        CONN_001: FalkorDB connection refused
        CONN_002: Connection timeout
        CONN_003: Authentication failed
        CONN_004: Connection pool exhausted

    Transient (network issues are retryable).
    """

    def __init__(self, message: str, error_code: str = "CONN_001", **kwargs):
        super().__init__(message=message, error_code=error_code, **kwargs)
        self.is_transient = True  # Retryable


class QueryError(DatabaseError):
    """
    Raised when database query fails.

    Error Codes:
        QUERY_001: Syntax error in Cypher query
        QUERY_002: Query timeout
        QUERY_003: Index not found
        QUERY_004: Constraint violation

    Conditionally transient (only for timeouts).
    """

    def __init__(
        self, message: str, error_code: str = "QUERY_001", is_timeout: bool = False, **kwargs
    ):
        """
        Initialize QueryError.

        Args:
            message: Error message
            error_code: Error code
            is_timeout: Whether this is a timeout error (transient)
            **kwargs: Additional arguments
        """
        super().__init__(message=message, error_code=error_code, **kwargs)
        self.is_transient = is_timeout  # Only timeouts are retryable


class TimeoutError(ZapomniError):
    """
    Raised when operation times out.

    Error Codes:
        TIMEOUT_001: Operation timeout
        TIMEOUT_002: Network timeout
        TIMEOUT_003: Query timeout

    Transient (timeouts are retryable).
    """

    def __init__(self, message: str, error_code: str = "TIMEOUT_001", **kwargs):
        super().__init__(message=message, error_code=error_code, **kwargs)
        self.is_transient = True  # Retryable
