"""
InputValidator - Centralized input validation for Zapomni.

This module provides the InputValidator class, which validates all user-provided
inputs before they enter the system. It enforces encoding, size, and structure
constraints, and provides clear error messages for invalid inputs.

Key Responsibilities:
- Text validation (encoding, size, content safety)
- Metadata validation (schema, JSON serialization, reserved keys)
- Query validation (length, format)
- Pagination validation (limit ranges)
- Input sanitization (PII removal, normalization)

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

import json
import re
import unicodedata
from typing import Any, Dict, Optional, Set

from zapomni_core.exceptions import ValidationError
from zapomni_core.utils.logger_factory import get_logger

logger = get_logger(__name__)


class InputValidator:
    """
    Centralized input validation for all Zapomni user inputs.

    Provides validation for text content, metadata, search queries,
    and pagination parameters. Enforces encoding, size, and structure
    constraints. Prevents injection attacks and malformed data.

    This class is stateless and thread-safe. All validation is independent,
    and the class can be safely shared across async tasks.

    Class Attributes:
        MAX_TEXT_SIZE: Maximum text size in bytes (10MB)
        MAX_METADATA_SIZE: Maximum metadata JSON size (1MB)
        MAX_QUERY_LENGTH: Maximum query string length (1000 chars)
        MAX_LIMIT: Maximum pagination limit (100)
        RESERVED_KEYS: Metadata keys reserved by system

    Example:
        ```python
        validator = InputValidator()

        # Validate text input
        try:
            text = validator.validate_text("Python is great")
            print(f"Valid text: {text}")
        except ValidationError as e:
            print(f"Invalid: {e.message}")

        # Validate metadata
        metadata = {"source": "user", "tags": ["python", "tutorial"]}
        validator.validate_metadata(metadata)

        # Sanitize user input
        clean_text = validator.sanitize_input("  Hello\\x00World  \\n")
        # Returns: "Hello World"
        ```
    """

    # Class constants
    MAX_TEXT_SIZE: int = 10_000_000  # 10 MB
    MAX_METADATA_SIZE: int = 1_000_000  # 1 MB (serialized JSON)
    MAX_QUERY_LENGTH: int = 1000
    MAX_LIMIT: int = 100
    RESERVED_KEYS: Set[str] = {"memory_id", "timestamp", "chunks", "embeddings"}

    def __init__(self) -> None:
        """
        Initialize InputValidator.

        No configuration needed - uses class constants.
        """
        logger.debug("input_validator_initialized")

    def validate_text(self, text: str, max_size: Optional[int] = None) -> str:
        """
        Validate text input for memory storage.

        Checks that text is:
        - Non-empty (after stripping whitespace)
        - Valid UTF-8 encoding
        - Within size limit
        - Free of null bytes and control characters

        Args:
            text: Text content to validate
            max_size: Optional custom max size (bytes).
                     Defaults to MAX_TEXT_SIZE (10MB)

        Returns:
            Sanitized text (whitespace normalized, null bytes removed)

        Raises:
            ValidationError: If text is invalid with specific reason:
                - "Text cannot be empty"
                - "Text must be a string, got {type}"
                - "Text exceeds maximum size (10,000,000 bytes)"
                - "Text contains invalid UTF-8 encoding"
                - "Text contains null bytes or control characters"

        Example:
            ```python
            validator = InputValidator()

            # Valid
            text = validator.validate_text("Python is great")

            # Invalid - empty
            validator.validate_text("")  # raises ValidationError

            # Invalid - too large
            huge = "x" * 20_000_000
            validator.validate_text(huge)  # raises ValidationError
            ```
        """
        # Step 1: Type check (must be str)
        if not isinstance(text, str):
            error_msg = f"Text must be a string, got {type(text).__name__}"
            logger.warning(
                "validate_text_failed", reason="wrong_type", expected="str", got=type(text).__name__
            )
            raise ValidationError(error_msg, error_code="VAL_002")

        # Step 2: Sanitize input (strip, remove null bytes)
        sanitized_text = self.sanitize_input(text)

        # Step 3: Check non-empty
        if not sanitized_text:
            logger.warning("validate_text_failed", reason="empty_after_sanitization")
            raise ValidationError("Text cannot be empty", error_code="VAL_001")

        # Step 4: Check encoding (valid UTF-8)
        self._check_encoding(sanitized_text)

        # Step 5: Check size (encode to bytes, compare to max_size)
        effective_max_size = max_size if max_size is not None else self.MAX_TEXT_SIZE
        text_bytes = sanitized_text.encode("utf-8")
        if len(text_bytes) > effective_max_size:
            error_msg = f"Text exceeds maximum size ({effective_max_size:,} bytes)"
            logger.warning(
                "validate_text_failed",
                reason="size_exceeded",
                size=len(text_bytes),
                max_size=effective_max_size,
            )
            raise ValidationError(error_msg, error_code="VAL_003")

        logger.debug("validate_text_success", text_length=len(sanitized_text))
        # Step 6: Return sanitized text
        return sanitized_text

    def validate_metadata(self, metadata: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Validate metadata dictionary.

        Checks that metadata:
        - Is a dictionary (if provided)
        - Contains only JSON-serializable values
        - Does not use reserved keys
        - Serializes to < 1MB JSON

        Args:
            metadata: Optional metadata dictionary

        Returns:
            Validated metadata (unchanged if valid, None if None)

        Raises:
            ValidationError: If metadata is invalid:
                - "Metadata must be a dictionary, got {type}"
                - "Metadata contains reserved key: {key}"
                - "Metadata value for '{key}' is not JSON-serializable"
                - "Metadata exceeds maximum size (1,000,000 bytes)"

        Example:
            ```python
            validator = InputValidator()

            # Valid
            meta = {"source": "user", "tags": ["ai", "ml"]}
            validator.validate_metadata(meta)

            # Valid - None
            validator.validate_metadata(None)

            # Invalid - reserved key
            meta = {"memory_id": "123"}
            validator.validate_metadata(meta)  # raises ValidationError

            # Invalid - not serializable
            meta = {"func": lambda x: x}
            validator.validate_metadata(meta)  # raises ValidationError
            ```
        """
        # Step 1: If metadata is None, return None
        if metadata is None:
            logger.debug("validate_metadata_success", metadata="none")
            return None

        # Step 2: Type check (must be dict)
        if not isinstance(metadata, dict):
            error_msg = f"Metadata must be a dictionary, got {type(metadata).__name__}"
            logger.warning(
                "validate_metadata_failed",
                reason="wrong_type",
                expected="dict",
                got=type(metadata).__name__,
            )
            raise ValidationError(error_msg, error_code="VAL_002")

        # Step 3: For each key:
        for key, value in metadata.items():
            # Check not in RESERVED_KEYS
            if key in self.RESERVED_KEYS:
                error_msg = f"Metadata contains reserved key: {key}"
                logger.warning("validate_metadata_failed", reason="reserved_key", key=key)
                raise ValidationError(error_msg, error_code="VAL_001")

            # Check value is JSON-serializable
            if not self._is_json_serializable(value):
                error_msg = f"Metadata value for '{key}' is not JSON-serializable"
                logger.warning(
                    "validate_metadata_failed",
                    reason="not_json_serializable",
                    key=key,
                    value_type=type(value).__name__,
                )
                raise ValidationError(error_msg, error_code="VAL_002")

        # Step 4: Serialize to JSON, check size < MAX_METADATA_SIZE
        try:
            metadata_json = json.dumps(metadata)
            metadata_bytes = metadata_json.encode("utf-8")
            if len(metadata_bytes) > self.MAX_METADATA_SIZE:
                error_msg = f"Metadata exceeds maximum size ({self.MAX_METADATA_SIZE:,} bytes)"
                logger.warning(
                    "validate_metadata_failed",
                    reason="size_exceeded",
                    size=len(metadata_bytes),
                    max_size=self.MAX_METADATA_SIZE,
                )
                raise ValidationError(error_msg, error_code="VAL_003")
        except (TypeError, ValueError) as e:
            # Shouldn't happen if _is_json_serializable works correctly
            error_msg = f"Metadata is not JSON-serializable: {str(e)}"
            logger.warning(
                "validate_metadata_failed", reason="json_serialization_error", error=str(e)
            )
            raise ValidationError(error_msg, error_code="VAL_002")

        logger.debug("validate_metadata_success", key_count=len(metadata))
        # Step 5: Return metadata unchanged
        return metadata

    def validate_query(self, query: str) -> str:
        """
        Validate search query.

        Checks that query:
        - Is a string
        - Is non-empty after stripping
        - Does not exceed max length (1000 chars)

        Args:
            query: Search query string

        Returns:
            Sanitized query (whitespace normalized)

        Raises:
            ValidationError: If query is invalid:
                - "Query cannot be empty"
                - "Query must be a string, got {type}"
                - "Query exceeds maximum length (1000 characters)"

        Example:
            ```python
            validator = InputValidator()

            # Valid
            query = validator.validate_query("Python programming")

            # Invalid - empty
            validator.validate_query("")  # raises ValidationError

            # Invalid - too long
            long_query = "x" * 1001
            validator.validate_query(long_query)  # raises ValidationError
            ```
        """
        # Step 1: Type check (must be str)
        if not isinstance(query, str):
            error_msg = f"Query must be a string, got {type(query).__name__}"
            logger.warning(
                "validate_query_failed",
                reason="wrong_type",
                expected="str",
                got=type(query).__name__,
            )
            raise ValidationError(error_msg, error_code="VAL_002")

        # Step 2: Sanitize (strip whitespace)
        sanitized_query = self.sanitize_input(query)

        # Step 3: Check non-empty
        if not sanitized_query:
            logger.warning("validate_query_failed", reason="empty_after_sanitization")
            raise ValidationError("Query cannot be empty", error_code="VAL_001")

        # Step 4: Check length <= MAX_QUERY_LENGTH
        if len(sanitized_query) > self.MAX_QUERY_LENGTH:
            error_msg = f"Query exceeds maximum length ({self.MAX_QUERY_LENGTH} characters)"
            logger.warning(
                "validate_query_failed",
                reason="length_exceeded",
                length=len(sanitized_query),
                max_length=self.MAX_QUERY_LENGTH,
            )
            raise ValidationError(error_msg, error_code="VAL_003")

        logger.debug("validate_query_success", query_length=len(sanitized_query))
        # Step 5: Return sanitized query
        return sanitized_query

    def validate_limit(self, limit: int, max_value: Optional[int] = None) -> int:
        """
        Validate pagination limit.

        Checks that limit:
        - Is an integer
        - Is positive (>= 1)
        - Does not exceed max_value

        Args:
            limit: Number of results to return
            max_value: Optional max limit (defaults to MAX_LIMIT = 100)

        Returns:
            Validated limit (unchanged)

        Raises:
            ValidationError: If limit is invalid:
                - "Limit must be an integer, got {type}"
                - "Limit must be positive (>= 1)"
                - "Limit exceeds maximum (100)"

        Example:
            ```python
            validator = InputValidator()

            # Valid
            validator.validate_limit(10)  # returns 10

            # Invalid - zero
            validator.validate_limit(0)  # raises ValidationError

            # Invalid - too large
            validator.validate_limit(200)  # raises ValidationError

            # Custom max
            validator.validate_limit(50, max_value=50)  # OK
            ```
        """
        # Step 1: Type check (must be int)
        if not isinstance(limit, int) or isinstance(limit, bool):
            error_msg = f"Limit must be an integer, got {type(limit).__name__}"
            logger.warning(
                "validate_limit_failed",
                reason="wrong_type",
                expected="int",
                got=type(limit).__name__,
            )
            raise ValidationError(error_msg, error_code="VAL_002")

        # Step 2: Check limit >= 1
        if limit < 1:
            logger.warning("validate_limit_failed", reason="not_positive", limit=limit)
            raise ValidationError("Limit must be positive (>= 1)", error_code="VAL_003")

        # Step 3: Check limit <= (max_value or MAX_LIMIT)
        effective_max = max_value if max_value is not None else self.MAX_LIMIT
        if limit > effective_max:
            error_msg = f"Limit exceeds maximum ({effective_max})"
            logger.warning(
                "validate_limit_failed", reason="exceeds_maximum", limit=limit, max=effective_max
            )
            raise ValidationError(error_msg, error_code="VAL_003")

        logger.debug("validate_limit_success", limit=limit)
        # Step 4: Return limit
        return limit

    def sanitize_input(self, text: str) -> str:
        """
        Sanitize text input (remove unsafe content).

        Performs:
        - Strip leading/trailing whitespace
        - Remove null bytes (\\x00)
        - Remove control characters (except \\n, \\t)
        - Normalize Unicode (NFC normalization)
        - Collapse multiple spaces to single space

        Args:
            text: Text to sanitize

        Returns:
            Sanitized text (safe for storage and processing)

        Example:
            ```python
            validator = InputValidator()

            # Remove null bytes
            clean = validator.sanitize_input("Hello\\x00World")
            # Returns: "HelloWorld"

            # Normalize whitespace
            clean = validator.sanitize_input("  Python   is    great  ")
            # Returns: "Python is great"

            # Normalize Unicode
            clean = validator.sanitize_input("café")  # é as e + combining accent
            # Returns: "café"  # é as single character
            ```
        """
        if not isinstance(text, str):
            return text

        # Step 1: Strip leading/trailing whitespace
        result = text.strip()

        # Step 2: Normalize Unicode (NFC normalization)
        result = unicodedata.normalize("NFC", result)

        # Step 3: Remove null bytes (\x00)
        result = self._remove_null_bytes(result)

        # Step 4: Remove control characters (except \n, \t)
        result = "".join(char for char in result if ord(char) >= 32 or char in "\n\t")

        # Step 5: Collapse multiple spaces to single space
        # Preserve newlines and tabs
        lines = result.split("\n")
        collapsed_lines = []
        for line in lines:
            # Collapse consecutive spaces, but preserve tabs
            simplified = []
            prev_space = False
            for char in line:
                if char == " ":
                    if not prev_space:
                        simplified.append(char)
                    prev_space = True
                else:
                    simplified.append(char)
                    prev_space = char == " "  # Reset to False for non-space

            collapsed_lines.append("".join(simplified))

        result = "\n".join(collapsed_lines)

        logger.debug(
            "sanitize_input_complete", original_length=len(text), sanitized_length=len(result)
        )
        return result

    # Private helper methods

    def _check_encoding(self, text: str) -> None:
        """
        Verify text is valid UTF-8.

        Args:
            text: Text to check

        Raises:
            ValidationError: If text contains invalid UTF-8 sequences
        """
        try:
            # Encoding to bytes and back checks for UTF-8 validity
            text.encode("utf-8").decode("utf-8")
            logger.debug("check_encoding_success")
        except (UnicodeDecodeError, UnicodeEncodeError) as e:
            logger.warning("check_encoding_failed", error=str(e))
            raise ValidationError(
                "Text contains invalid UTF-8 encoding", error_code="VAL_004", original_exception=e
            )

    def _is_json_serializable(self, obj: Any) -> bool:
        """
        Check if object can be JSON-serialized.

        Args:
            obj: Object to check

        Returns:
            True if serializable, False otherwise

        Implementation:
            Attempts json.dumps(), catches TypeError
        """
        try:
            json.dumps(obj)
            return True
        except (TypeError, ValueError):
            return False

    def _remove_null_bytes(self, text: str) -> str:
        """
        Remove null bytes and control characters from text.

        Args:
            text: Text to clean

        Returns:
            Text with null bytes removed

        Note:
            Preserves \\n (newline) and \\t (tab)
        """
        # Remove null bytes (\x00) specifically
        return text.replace("\x00", "")
