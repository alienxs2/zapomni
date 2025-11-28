"""Unit tests for zapomni_core.treesitter.exceptions module."""

import pytest

from zapomni_core.treesitter.exceptions import (
    ExtractorNotFoundError,
    LanguageNotSupportedError,
    ParseError,
    TreeSitterError,
)
from zapomni_core.exceptions import ProcessingError


class TestTreeSitterError:
    """Tests for TreeSitterError base exception."""

    def test_default_message(self):
        """Test default error message."""
        error = TreeSitterError()
        assert str(error) == "Tree-sitter operation failed"

    def test_custom_message(self):
        """Test custom error message."""
        error = TreeSitterError(message="Custom error message")
        assert "Custom error message" in str(error)

    def test_default_error_code(self):
        """Test default error code."""
        error = TreeSitterError()
        assert error.error_code == "TS_001"

    def test_custom_error_code(self):
        """Test custom error code."""
        error = TreeSitterError(error_code="TS_CUSTOM")
        assert error.error_code == "TS_CUSTOM"

    def test_is_transient_false(self):
        """Test that is_transient is False by default."""
        error = TreeSitterError()
        assert error.is_transient is False

    def test_inherits_from_processing_error(self):
        """Test that TreeSitterError inherits from ProcessingError."""
        error = TreeSitterError()
        assert isinstance(error, ProcessingError)

    def test_with_details(self):
        """Test error with additional details."""
        error = TreeSitterError(
            message="Error with details",
            details={"key": "value"},
        )
        assert error.details["key"] == "value"


class TestLanguageNotSupportedError:
    """Tests for LanguageNotSupportedError."""

    def test_default_message(self):
        """Test default error message contains language name."""
        error = LanguageNotSupportedError(language="obscure_lang")
        assert "obscure_lang" in str(error)
        assert "not supported" in str(error)

    def test_custom_message(self):
        """Test custom error message."""
        error = LanguageNotSupportedError(
            language="test_lang",
            message="Custom language error",
        )
        assert "Custom language error" in str(error)

    def test_language_attribute(self):
        """Test language attribute is set."""
        error = LanguageNotSupportedError(language="cobol")
        assert error.language == "cobol"

    def test_error_code(self):
        """Test default error code is TS_002."""
        error = LanguageNotSupportedError(language="test")
        assert error.error_code == "TS_002"

    def test_language_in_details(self):
        """Test language is added to details."""
        error = LanguageNotSupportedError(language="dart")
        assert error.details["language"] == "dart"

    def test_inherits_from_treesitter_error(self):
        """Test inheritance from TreeSitterError."""
        error = LanguageNotSupportedError(language="test")
        assert isinstance(error, TreeSitterError)

    def test_with_additional_details(self):
        """Test error with additional details."""
        error = LanguageNotSupportedError(
            language="test",
            details={"available_languages": ["python", "javascript"]},
        )
        assert "available_languages" in error.details
        assert error.details["language"] == "test"


class TestParseError:
    """Tests for ParseError."""

    def test_default_message(self):
        """Test default error message contains file path."""
        error = ParseError(file_path="/path/to/file.py")
        assert "/path/to/file.py" in str(error)

    def test_message_with_details(self):
        """Test message includes parse details."""
        error = ParseError(
            file_path="/test.py",
            parse_details="Syntax error at line 10",
        )
        assert "Syntax error at line 10" in str(error)

    def test_custom_message(self):
        """Test custom error message."""
        error = ParseError(
            file_path="/test.py",
            message="Custom parse error",
        )
        assert "Custom parse error" in str(error)

    def test_file_path_attribute(self):
        """Test file_path attribute is set."""
        error = ParseError(file_path="/path/to/code.js")
        assert error.file_path == "/path/to/code.js"

    def test_parse_details_attribute(self):
        """Test parse_details attribute is set."""
        error = ParseError(
            file_path="/test.py",
            parse_details="Unexpected token",
        )
        assert error.parse_details == "Unexpected token"

    def test_error_code(self):
        """Test default error code is TS_003."""
        error = ParseError(file_path="/test.py")
        assert error.error_code == "TS_003"

    def test_file_path_in_details(self):
        """Test file_path is added to details."""
        error = ParseError(file_path="/path/to/test.py")
        assert error.details["file_path"] == "/path/to/test.py"

    def test_is_transient_false(self):
        """Test that syntax errors are not transient."""
        error = ParseError(file_path="/test.py")
        assert error.is_transient is False

    def test_inherits_from_treesitter_error(self):
        """Test inheritance from TreeSitterError."""
        error = ParseError(file_path="/test.py")
        assert isinstance(error, TreeSitterError)


class TestExtractorNotFoundError:
    """Tests for ExtractorNotFoundError."""

    def test_default_message(self):
        """Test default error message."""
        error = ExtractorNotFoundError(language="dart")
        assert "dart" in str(error)
        assert "GenericExtractor" in str(error)

    def test_custom_message(self):
        """Test custom error message."""
        error = ExtractorNotFoundError(
            language="test",
            message="Custom extractor error",
        )
        assert "Custom extractor error" in str(error)

    def test_language_attribute(self):
        """Test language attribute is set."""
        error = ExtractorNotFoundError(language="kotlin")
        assert error.language == "kotlin"

    def test_error_code(self):
        """Test default error code is TS_004."""
        error = ExtractorNotFoundError(language="test")
        assert error.error_code == "TS_004"

    def test_fallback_in_details(self):
        """Test fallback extractor is mentioned in details."""
        error = ExtractorNotFoundError(language="scala")
        assert error.details["fallback"] == "GenericExtractor"
        assert error.details["language"] == "scala"

    def test_inherits_from_treesitter_error(self):
        """Test inheritance from TreeSitterError."""
        error = ExtractorNotFoundError(language="test")
        assert isinstance(error, TreeSitterError)
