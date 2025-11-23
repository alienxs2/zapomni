"""
Unit tests for exception hierarchy and error handling.

Tests exception creation, error codes, transient flags, and serialization.
"""

import pytest
import uuid

# Import will fail initially (TDD RED phase)
from zapomni_core.exceptions import (
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


class TestZapomniErrorBase:
    """Test base ZapomniError exception."""

    def test_base_exception_creation(self):
        """Test creating base ZapomniError."""
        error = ZapomniError(
            message="Test error",
            error_code="ERR_001",
        )

        assert error.message == "Test error"
        assert error.error_code == "ERR_001"
        assert str(error) == "Test error"

    def test_base_exception_with_details(self):
        """Test ZapomniError with details dict."""
        error = ZapomniError(
            message="Test error",
            error_code="ERR_001",
            details={"key": "value", "param": 123},
        )

        assert error.details == {"key": "value", "param": 123}

    def test_base_exception_default_details(self):
        """Test ZapomniError with default empty details."""
        error = ZapomniError(
            message="Test error",
            error_code="ERR_001",
        )

        assert error.details == {}

    def test_base_exception_correlation_id(self):
        """Test ZapomniError with correlation_id."""
        correlation_id = str(uuid.uuid4())
        error = ZapomniError(
            message="Test error",
            error_code="ERR_001",
            correlation_id=correlation_id,
        )

        assert error.correlation_id == correlation_id

    def test_base_exception_auto_correlation_id(self):
        """Test ZapomniError auto-generates correlation_id."""
        error = ZapomniError(
            message="Test error",
            error_code="ERR_001",
        )

        # Should have auto-generated UUID
        assert error.correlation_id is not None
        # Verify it's a valid UUID string
        uuid.UUID(error.correlation_id)

    def test_base_exception_to_dict(self):
        """Test ZapomniError serialization to dict."""
        correlation_id = str(uuid.uuid4())
        error = ZapomniError(
            message="Test error",
            error_code="ERR_001",
            details={"key": "value"},
            correlation_id=correlation_id,
        )

        error_dict = error.to_dict()

        assert error_dict["error"] == "ZapomniError"
        assert error_dict["message"] == "Test error"
        assert error_dict["error_code"] == "ERR_001"
        assert error_dict["details"] == {"key": "value"}
        assert error_dict["correlation_id"] == correlation_id
        assert error_dict["original_error"] is None

    def test_base_exception_is_transient_default(self):
        """Test ZapomniError default is_transient flag."""
        error = ZapomniError(
            message="Test error",
            error_code="ERR_001",
        )

        assert error.is_transient is False


class TestValidationError:
    """Test ValidationError exception."""

    def test_validation_error_creation(self):
        """Test creating ValidationError."""
        error = ValidationError(
            message="Invalid input",
            error_code="VAL_001",
        )

        assert isinstance(error, ZapomniError)
        assert error.message == "Invalid input"
        assert error.error_code == "VAL_001"

    def test_validation_error_not_transient(self):
        """Test ValidationError is not transient."""
        error = ValidationError(
            message="Invalid input",
            error_code="VAL_001",
        )

        assert error.is_transient is False


class TestProcessingError:
    """Test ProcessingError exception."""

    def test_processing_error_creation(self):
        """Test creating ProcessingError."""
        error = ProcessingError(
            message="Processing failed",
            error_code="PROC_001",
        )

        assert isinstance(error, ZapomniError)
        assert error.message == "Processing failed"
        assert error.error_code == "PROC_001"

    def test_processing_error_not_transient(self):
        """Test ProcessingError is not transient by default."""
        error = ProcessingError(
            message="Processing failed",
            error_code="PROC_001",
        )

        assert error.is_transient is False


class TestEmbeddingError:
    """Test EmbeddingError exception."""

    def test_embedding_error_creation(self):
        """Test creating EmbeddingError."""
        error = EmbeddingError(
            message="Embedding generation failed",
            error_code="EMB_001",
        )

        assert isinstance(error, ZapomniError)
        assert error.message == "Embedding generation failed"
        assert error.error_code == "EMB_001"

    def test_embedding_error_is_transient(self):
        """Test EmbeddingError is transient (retryable)."""
        error = EmbeddingError(
            message="Embedding timeout",
            error_code="EMB_002",
        )

        assert error.is_transient is True


class TestExtractionError:
    """Test ExtractionError exception."""

    def test_extraction_error_creation(self):
        """Test creating ExtractionError."""
        error = ExtractionError(
            message="Entity extraction failed",
            error_code="EXTR_001",
        )

        assert isinstance(error, ZapomniError)
        assert error.message == "Entity extraction failed"
        assert error.error_code == "EXTR_001"

    def test_extraction_error_is_transient(self):
        """Test ExtractionError is transient (LLM errors often transient)."""
        error = ExtractionError(
            message="LLM timeout",
            error_code="EXTR_002",
        )

        assert error.is_transient is True


class TestSearchError:
    """Test SearchError exception."""

    def test_search_error_creation(self):
        """Test creating SearchError."""
        error = SearchError(
            message="Search failed",
            error_code="SEARCH_001",
        )

        assert isinstance(error, ZapomniError)
        assert error.message == "Search failed"
        assert error.error_code == "SEARCH_001"

    def test_search_error_not_transient(self):
        """Test SearchError is not transient by default."""
        error = SearchError(
            message="Search failed",
            error_code="SEARCH_001",
        )

        assert error.is_transient is False


class TestDatabaseError:
    """Test DatabaseError exception."""

    def test_database_error_creation(self):
        """Test creating DatabaseError."""
        error = DatabaseError(
            message="Database operation failed",
            error_code="DB_001",
        )

        assert isinstance(error, ZapomniError)
        assert error.message == "Database operation failed"
        assert error.error_code == "DB_001"

    def test_database_error_not_transient(self):
        """Test DatabaseError is not transient by default."""
        error = DatabaseError(
            message="Database error",
            error_code="DB_001",
        )

        assert error.is_transient is False


class TestConnectionError:
    """Test ConnectionError exception."""

    def test_connection_error_creation(self):
        """Test creating ConnectionError."""
        error = ConnectionError(
            message="Connection failed",
            error_code="CONN_001",
        )

        assert isinstance(error, DatabaseError)
        assert error.message == "Connection failed"
        assert error.error_code == "CONN_001"

    def test_connection_error_is_transient(self):
        """Test ConnectionError is transient (network issues)."""
        error = ConnectionError(
            message="Connection timeout",
            error_code="CONN_002",
        )

        assert error.is_transient is True


class TestQueryError:
    """Test QueryError exception."""

    def test_query_error_creation(self):
        """Test creating QueryError."""
        error = QueryError(
            message="Query failed",
            error_code="QUERY_001",
        )

        assert isinstance(error, DatabaseError)
        assert error.message == "Query failed"
        assert error.error_code == "QUERY_001"

    def test_query_error_not_transient_by_default(self):
        """Test QueryError is not transient by default (syntax errors)."""
        error = QueryError(
            message="Syntax error",
            error_code="QUERY_001",
        )

        assert error.is_transient is False

    def test_query_error_transient_on_timeout(self):
        """Test QueryError can be transient for timeouts."""
        error = QueryError(
            message="Query timeout",
            error_code="QUERY_002",
            is_timeout=True,
        )

        assert error.is_transient is True


class TestTimeoutError:
    """Test TimeoutError exception."""

    def test_timeout_error_creation(self):
        """Test creating TimeoutError."""
        error = TimeoutError(
            message="Operation timed out",
            error_code="TIMEOUT_001",
        )

        assert isinstance(error, ZapomniError)
        assert error.message == "Operation timed out"
        assert error.error_code == "TIMEOUT_001"

    def test_timeout_error_is_transient(self):
        """Test TimeoutError is transient."""
        error = TimeoutError(
            message="Operation timed out",
            error_code="TIMEOUT_001",
        )

        assert error.is_transient is True


class TestExceptionSerialization:
    """Test exception serialization and metadata."""

    def test_exception_with_original_exception(self):
        """Test wrapping original exception."""
        original = ValueError("Original error")
        error = ProcessingError(
            message="Processing failed",
            error_code="PROC_001",
            original_exception=original,
        )

        assert error.original_exception is original
        error_dict = error.to_dict()
        assert error_dict["original_error"] == "Original error"

    def test_exception_hierarchy_preserved(self):
        """Test exception inheritance hierarchy."""
        # ValidationError -> ZapomniError
        assert issubclass(ValidationError, ZapomniError)

        # ProcessingError -> ZapomniError
        assert issubclass(ProcessingError, ZapomniError)

        # DatabaseError -> ZapomniError
        assert issubclass(DatabaseError, ZapomniError)

        # ConnectionError -> DatabaseError -> ZapomniError
        assert issubclass(ConnectionError, DatabaseError)
        assert issubclass(ConnectionError, ZapomniError)

        # QueryError -> DatabaseError -> ZapomniError
        assert issubclass(QueryError, DatabaseError)
        assert issubclass(QueryError, ZapomniError)

    def test_exception_to_dict_complete(self):
        """Test complete exception serialization."""
        original = ValueError("Original")
        correlation_id = str(uuid.uuid4())

        error = EmbeddingError(
            message="Embedding failed",
            error_code="EMB_001",
            details={"model": "nomic-embed-text", "timeout": 30},
            correlation_id=correlation_id,
            original_exception=original,
        )

        error_dict = error.to_dict()

        assert error_dict["error"] == "EmbeddingError"
        assert error_dict["message"] == "Embedding failed"
        assert error_dict["error_code"] == "EMB_001"
        assert error_dict["details"]["model"] == "nomic-embed-text"
        assert error_dict["details"]["timeout"] == 30
        assert error_dict["correlation_id"] == correlation_id
        assert error_dict["original_error"] == "Original"


class TestErrorCodes:
    """Test error codes are properly assigned."""

    def test_validation_error_codes(self):
        """Test ValidationError uses VAL_* codes."""
        error = ValidationError(
            message="Missing required field",
            error_code="VAL_001",
        )
        assert error.error_code.startswith("VAL_")

    def test_embedding_error_codes(self):
        """Test EmbeddingError uses EMB_* codes."""
        error = EmbeddingError(
            message="Ollama connection failed",
            error_code="EMB_001",
        )
        assert error.error_code.startswith("EMB_")

    def test_database_error_codes(self):
        """Test DatabaseError uses DB_* codes."""
        error = DatabaseError(
            message="Database query failed",
            error_code="DB_001",
        )
        assert error.error_code.startswith("DB_")

    def test_connection_error_codes(self):
        """Test ConnectionError uses CONN_* codes."""
        error = ConnectionError(
            message="Connection timeout",
            error_code="CONN_001",
        )
        assert error.error_code.startswith("CONN_")


class TestTransientFlags:
    """Test is_transient flag for retry logic."""

    def test_transient_exceptions_list(self):
        """Test which exceptions are marked as transient."""
        # Transient (should retry)
        transient_errors = [
            EmbeddingError("Test", "EMB_001"),
            ExtractionError("Test", "EXTR_001"),
            ConnectionError("Test", "CONN_001"),
            TimeoutError("Test", "TIMEOUT_001"),
        ]

        for error in transient_errors:
            assert error.is_transient is True, f"{error.__class__.__name__} should be transient"

    def test_non_transient_exceptions_list(self):
        """Test which exceptions are NOT transient."""
        # Non-transient (should not retry)
        non_transient_errors = [
            ValidationError("Test", "VAL_001"),
            ProcessingError("Test", "PROC_001"),
            SearchError("Test", "SEARCH_001"),
            DatabaseError("Test", "DB_001"),
            QueryError("Test", "QUERY_001"),  # Unless is_timeout=True
        ]

        for error in non_transient_errors:
            assert error.is_transient is False, f"{error.__class__.__name__} should not be transient"
