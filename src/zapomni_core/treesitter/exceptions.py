"""
Exception hierarchy for Tree-sitter module.

Defines all exception types specific to tree-sitter parsing and code extraction.
Follows the error handling strategy from error_handling_strategy.md.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from typing import Optional

from zapomni_core.exceptions import ProcessingError


class TreeSitterError(ProcessingError):
    """
    Base exception for all tree-sitter related errors.

    All tree-sitter exceptions inherit from this class. Provides standard
    error handling for parsing, extraction, and language support issues.

    Error Code: TS_001

    Attributes:
        message: Human-readable error message
        error_code: Programmatic error code (default: "TS_001")
        details: Additional context (dict)
        correlation_id: UUID for tracing across layers
        original_exception: Wrapped exception (if any)
        is_transient: Whether error is transient (retryable) - default False

    Example:
        raise TreeSitterError(
            message="Tree-sitter operation failed",
            details={"operation": "parse"}
        )
    """

    def __init__(
        self,
        message: str = "Tree-sitter operation failed",
        error_code: str = "TS_001",
        **kwargs,
    ):
        """
        Initialize TreeSitterError.

        Args:
            message: Error message describing the tree-sitter error
            error_code: Error code for programmatic handling
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(message=message, error_code=error_code, **kwargs)
        self.is_transient = False


class LanguageNotSupportedError(TreeSitterError):
    """
    Raised when a programming language is not supported by tree-sitter-language-pack.

    This error indicates that the requested language does not have a grammar
    available in the tree-sitter-language-pack library.

    Error Code: TS_002

    Attributes:
        language: The unsupported language identifier
        message: Human-readable error message
        is_transient: False (language support is static)

    Example:
        raise LanguageNotSupportedError(
            language="obscure_lang",
            details={"available_languages": ["python", "javascript", "rust"]}
        )
    """

    def __init__(
        self,
        language: str,
        message: Optional[str] = None,
        error_code: str = "TS_002",
        **kwargs,
    ):
        """
        Initialize LanguageNotSupportedError.

        Args:
            language: The language identifier that is not supported
            message: Optional custom error message
            error_code: Error code for programmatic handling
            **kwargs: Additional arguments passed to parent
        """
        self.language = language
        if message is None:
            message = f"Language '{language}' is not supported by tree-sitter-language-pack"

        # Add language to details
        details = kwargs.pop("details", {})
        details["language"] = language

        super().__init__(message=message, error_code=error_code, details=details, **kwargs)


class ParseError(TreeSitterError):
    """
    Raised when tree-sitter fails to parse source code.

    This error indicates a parsing failure, typically due to syntax errors
    in the source code or issues with the language grammar.

    Error Code: TS_003

    Attributes:
        file_path: Path to the file that failed to parse
        details: Optional additional error details from the parser
        message: Human-readable error message
        is_transient: False (syntax errors require code changes)

    Example:
        raise ParseError(
            file_path="/path/to/broken_file.py",
            details="Unexpected token at line 42"
        )
    """

    def __init__(
        self,
        file_path: str,
        parse_details: Optional[str] = None,
        message: Optional[str] = None,
        error_code: str = "TS_003",
        **kwargs,
    ):
        """
        Initialize ParseError.

        Args:
            file_path: Path to the file that failed to parse
            parse_details: Optional details about the parsing failure
            message: Optional custom error message
            error_code: Error code for programmatic handling
            **kwargs: Additional arguments passed to parent
        """
        self.file_path = file_path
        self.parse_details = parse_details

        if message is None:
            if parse_details:
                message = f"Failed to parse file '{file_path}': {parse_details}"
            else:
                message = f"Failed to parse file '{file_path}'"

        # Add file_path and details to details dict
        details = kwargs.pop("details", {})
        details["file_path"] = file_path
        if parse_details:
            details["parse_details"] = parse_details

        super().__init__(message=message, error_code=error_code, details=details, **kwargs)
        self.is_transient = False  # Syntax errors are not retryable


class ExtractorNotFoundError(TreeSitterError):
    """
    Raised when no specific extractor is available for a language.

    This is a warning-level error indicating that the system will fall back
    to GenericExtractor for the requested language. The GenericExtractor
    provides basic extraction capabilities for all tree-sitter supported languages.

    Error Code: TS_004

    Attributes:
        language: The language for which no specific extractor exists
        message: Human-readable error message
        is_transient: False (extractor availability is static)

    Example:
        raise ExtractorNotFoundError(
            language="dart",
            details={"fallback": "GenericExtractor"}
        )
    """

    def __init__(
        self,
        language: str,
        message: Optional[str] = None,
        error_code: str = "TS_004",
        **kwargs,
    ):
        """
        Initialize ExtractorNotFoundError.

        Args:
            language: The language for which no extractor was found
            message: Optional custom error message
            error_code: Error code for programmatic handling
            **kwargs: Additional arguments passed to parent
        """
        self.language = language
        if message is None:
            message = (
                f"No specific extractor found for language '{language}'. "
                f"GenericExtractor will be used instead."
            )

        # Add language to details
        details = kwargs.pop("details", {})
        details["language"] = language
        details["fallback"] = "GenericExtractor"

        super().__init__(message=message, error_code=error_code, details=details, **kwargs)


__all__ = [
    "TreeSitterError",
    "LanguageNotSupportedError",
    "ParseError",
    "ExtractorNotFoundError",
]
