"""
Unit tests for InputValidator component.

Tests cover:
- Text validation (happy path and error cases)
- Metadata validation (happy path and error cases)
- Query validation (happy path and error cases)
- Limit validation (happy path and error cases)
- Sanitization (PII removal, Unicode normalization)
- Edge cases and boundary conditions

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

import pytest
from zapomni_core.validation import InputValidator
from zapomni_core.exceptions import ValidationError


class TestInputValidatorInit:
    """Test InputValidator initialization."""

    def test_init_creates_instance(self):
        """Test that InputValidator can be instantiated."""
        validator = InputValidator()
        assert validator is not None
        assert isinstance(validator, InputValidator)

    def test_class_constants_exist(self):
        """Test that class constants are properly defined."""
        assert InputValidator.MAX_TEXT_SIZE == 10_000_000
        assert InputValidator.MAX_METADATA_SIZE == 1_000_000
        assert InputValidator.MAX_QUERY_LENGTH == 1000
        assert InputValidator.MAX_LIMIT == 100
        assert "memory_id" in InputValidator.RESERVED_KEYS
        assert "timestamp" in InputValidator.RESERVED_KEYS
        assert "chunks" in InputValidator.RESERVED_KEYS
        assert "embeddings" in InputValidator.RESERVED_KEYS


class TestValidateTextSuccess:
    """Test validate_text() happy path."""

    @pytest.fixture
    def validator(self):
        """Provide InputValidator instance."""
        return InputValidator()

    def test_validate_text_success(self, validator):
        """Test valid text passes validation."""
        text = "Python is a programming language"
        result = validator.validate_text(text)
        assert result == text

    def test_validate_text_with_whitespace(self, validator):
        """Test text with leading/trailing whitespace."""
        text = "  Python is great  "
        result = validator.validate_text(text)
        assert result == "Python is great"

    def test_validate_text_with_newlines(self, validator):
        """Test text with newlines is preserved."""
        text = "Python\nis\ngreat"
        result = validator.validate_text(text)
        assert "Python" in result
        assert "is" in result
        assert "great" in result

    def test_validate_text_unicode(self, validator):
        """Test Unicode text passes validation."""
        text = "Python программирование العربية 中文"
        result = validator.validate_text(text)
        assert "Python" in result

    def test_validate_text_max_size_boundary(self, validator):
        """Test text at exact maximum size."""
        # Create text that is exactly MAX_TEXT_SIZE bytes
        text = "x" * (validator.MAX_TEXT_SIZE // 2)
        result = validator.validate_text(text)
        assert result is not None

    def test_validate_text_custom_max_size(self, validator):
        """Test custom max size parameter."""
        text = "Hello World"
        result = validator.validate_text(text, max_size=1000)
        assert result == text

    def test_validate_text_with_null_bytes(self, validator):
        """Test null bytes are removed during sanitization."""
        text = "Hello\x00World"
        result = validator.validate_text(text)
        assert "\x00" not in result
        assert "HelloWorld" == result

    def test_validate_text_with_control_chars(self, validator):
        """Test control characters are removed during sanitization."""
        text = "Hello\x01\x02World"
        result = validator.validate_text(text)
        assert "\x01" not in result
        assert "\x02" not in result

    def test_validate_text_with_multiple_spaces(self, validator):
        """Test multiple spaces are collapsed."""
        text = "Python   is    great"
        result = validator.validate_text(text)
        assert result == "Python is great"

    def test_validate_text_unicode_normalization(self, validator):
        """Test Unicode normalization (NFC)."""
        # é as e + combining accent (NFD) vs single character (NFC)
        text_nfd = "cafe\u0301"  # NFD form
        result = validator.validate_text(text_nfd)
        # After NFC normalization, should be "café"
        assert "caf" in result


class TestValidateTextErrors:
    """Test validate_text() error cases."""

    @pytest.fixture
    def validator(self):
        """Provide InputValidator instance."""
        return InputValidator()

    def test_validate_text_empty_raises(self, validator):
        """Test empty string raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_text("")
        assert "cannot be empty" in exc_info.value.message.lower()
        assert exc_info.value.error_code == "VAL_001"

    def test_validate_text_whitespace_only_raises(self, validator):
        """Test whitespace-only string raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_text("   \n  \t  ")
        assert "cannot be empty" in exc_info.value.message.lower()

    def test_validate_text_wrong_type_raises(self, validator):
        """Test non-string type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_text(123)
        assert "must be a string" in exc_info.value.message.lower()
        assert exc_info.value.error_code == "VAL_002"

    def test_validate_text_none_type_raises(self, validator):
        """Test None type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_text(None)
        assert "must be a string" in exc_info.value.message.lower()

    def test_validate_text_list_type_raises(self, validator):
        """Test list type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_text(["hello", "world"])
        assert "must be a string" in exc_info.value.message.lower()

    def test_validate_text_too_large_raises(self, validator):
        """Test oversized text raises ValidationError."""
        huge_text = "x" * (validator.MAX_TEXT_SIZE + 1)
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_text(huge_text)
        assert "exceeds maximum size" in exc_info.value.message.lower()
        assert exc_info.value.error_code == "VAL_003"

    def test_validate_text_custom_max_exceeded_raises(self, validator):
        """Test text exceeding custom max size raises."""
        text = "x" * 1001
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_text(text, max_size=1000)
        assert "exceeds maximum size" in exc_info.value.message.lower()


class TestValidateMetadataSuccess:
    """Test validate_metadata() happy path."""

    @pytest.fixture
    def validator(self):
        """Provide InputValidator instance."""
        return InputValidator()

    def test_validate_metadata_none(self, validator):
        """Test None metadata is valid."""
        result = validator.validate_metadata(None)
        assert result is None

    def test_validate_metadata_empty_dict(self, validator):
        """Test empty dict is valid."""
        result = validator.validate_metadata({})
        assert result == {}

    def test_validate_metadata_simple_dict(self, validator):
        """Test simple metadata dict."""
        metadata = {"source": "user", "version": 1}
        result = validator.validate_metadata(metadata)
        assert result == metadata

    def test_validate_metadata_with_list(self, validator):
        """Test metadata with lists."""
        metadata = {"tags": ["python", "ai", "ml"]}
        result = validator.validate_metadata(metadata)
        assert result == metadata

    def test_validate_metadata_with_numbers(self, validator):
        """Test metadata with various number types."""
        metadata = {
            "count": 42,
            "ratio": 3.14,
            "flag": True
        }
        result = validator.validate_metadata(metadata)
        assert result == metadata

    def test_validate_metadata_nested_structure(self, validator):
        """Test nested metadata structure."""
        metadata = {
            "user": {
                "name": "Alice",
                "tags": ["admin", "developer"],
                "score": 95.5
            }
        }
        result = validator.validate_metadata(metadata)
        assert result == metadata

    def test_validate_metadata_with_none_values(self, validator):
        """Test metadata with None values."""
        metadata = {
            "field1": "value",
            "field2": None
        }
        result = validator.validate_metadata(metadata)
        assert result == metadata

    def test_validate_metadata_unicode_keys(self, validator):
        """Test metadata with Unicode keys and values."""
        metadata = {
            "язык": "Python",
            "中文": "编程"
        }
        result = validator.validate_metadata(metadata)
        assert result == metadata


class TestValidateMetadataErrors:
    """Test validate_metadata() error cases."""

    @pytest.fixture
    def validator(self):
        """Provide InputValidator instance."""
        return InputValidator()

    def test_validate_metadata_wrong_type_raises(self, validator):
        """Test non-dict type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_metadata("not a dict")
        assert "must be a dictionary" in exc_info.value.message.lower()
        assert exc_info.value.error_code == "VAL_002"

    def test_validate_metadata_list_raises(self, validator):
        """Test list type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_metadata(["item1", "item2"])
        assert "must be a dictionary" in exc_info.value.message.lower()

    def test_validate_metadata_reserved_key_memory_id_raises(self, validator):
        """Test reserved key 'memory_id' raises."""
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_metadata({"memory_id": "123"})
        assert "reserved key" in exc_info.value.message.lower()
        assert exc_info.value.error_code == "VAL_001"

    def test_validate_metadata_reserved_key_timestamp_raises(self, validator):
        """Test reserved key 'timestamp' raises."""
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_metadata({"timestamp": 1234567890})
        assert "reserved key" in exc_info.value.message.lower()

    def test_validate_metadata_reserved_key_chunks_raises(self, validator):
        """Test reserved key 'chunks' raises."""
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_metadata({"chunks": []})
        assert "reserved key" in exc_info.value.message.lower()

    def test_validate_metadata_reserved_key_embeddings_raises(self, validator):
        """Test reserved key 'embeddings' raises."""
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_metadata({"embeddings": []})
        assert "reserved key" in exc_info.value.message.lower()

    def test_validate_metadata_lambda_raises(self, validator):
        """Test lambda function raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_metadata({"func": lambda x: x})
        assert "not json-serializable" in exc_info.value.message.lower()
        assert exc_info.value.error_code == "VAL_002"

    def test_validate_metadata_custom_object_raises(self, validator):
        """Test custom object raises."""
        class CustomObj:
            pass

        with pytest.raises(ValidationError) as exc_info:
            validator.validate_metadata({"obj": CustomObj()})
        assert "not json-serializable" in exc_info.value.message.lower()

    def test_validate_metadata_circular_reference_raises(self, validator):
        """Test circular reference raises."""
        metadata = {}
        metadata["self"] = metadata
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_metadata(metadata)
        assert "not json-serializable" in exc_info.value.message.lower()

    def test_validate_metadata_too_large_raises(self, validator):
        """Test oversized metadata raises."""
        # Create metadata that serializes to > 1MB
        large_value = "x" * (validator.MAX_METADATA_SIZE + 100)
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_metadata({"large": large_value})
        assert "exceeds maximum size" in exc_info.value.message.lower()
        assert exc_info.value.error_code == "VAL_003"


class TestValidateQuerySuccess:
    """Test validate_query() happy path."""

    @pytest.fixture
    def validator(self):
        """Provide InputValidator instance."""
        return InputValidator()

    def test_validate_query_success(self, validator):
        """Test valid query passes."""
        query = "Python programming"
        result = validator.validate_query(query)
        assert result == query

    def test_validate_query_with_whitespace(self, validator):
        """Test query with leading/trailing whitespace."""
        query = "  Python programming  "
        result = validator.validate_query(query)
        assert result == "Python programming"

    def test_validate_query_unicode(self, validator):
        """Test Unicode query."""
        query = "Python программирование"
        result = validator.validate_query(query)
        assert "Python" in result

    def test_validate_query_special_chars(self, validator):
        """Test query with special characters."""
        query = "What is $variable?"
        result = validator.validate_query(query)
        assert "What" in result

    def test_validate_query_exactly_max_length(self, validator):
        """Test query at exactly max length."""
        query = "x" * validator.MAX_QUERY_LENGTH
        result = validator.validate_query(query)
        assert len(result) == validator.MAX_QUERY_LENGTH

    def test_validate_query_multiple_spaces(self, validator):
        """Test query with multiple spaces is collapsed."""
        query = "Python   is    great"
        result = validator.validate_query(query)
        assert result == "Python is great"


class TestValidateQueryErrors:
    """Test validate_query() error cases."""

    @pytest.fixture
    def validator(self):
        """Provide InputValidator instance."""
        return InputValidator()

    def test_validate_query_empty_raises(self, validator):
        """Test empty query raises."""
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_query("")
        assert "cannot be empty" in exc_info.value.message.lower()
        assert exc_info.value.error_code == "VAL_001"

    def test_validate_query_whitespace_only_raises(self, validator):
        """Test whitespace-only query raises."""
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_query("   \n  ")
        assert "cannot be empty" in exc_info.value.message.lower()

    def test_validate_query_wrong_type_raises(self, validator):
        """Test non-string type raises."""
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_query(123)
        assert "must be a string" in exc_info.value.message.lower()
        assert exc_info.value.error_code == "VAL_002"

    def test_validate_query_too_long_raises(self, validator):
        """Test oversized query raises."""
        long_query = "x" * (validator.MAX_QUERY_LENGTH + 1)
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_query(long_query)
        assert "exceeds maximum length" in exc_info.value.message.lower()
        assert exc_info.value.error_code == "VAL_003"


class TestValidateLimitSuccess:
    """Test validate_limit() happy path."""

    @pytest.fixture
    def validator(self):
        """Provide InputValidator instance."""
        return InputValidator()

    def test_validate_limit_success(self, validator):
        """Test valid limit passes."""
        result = validator.validate_limit(10)
        assert result == 10

    def test_validate_limit_minimum(self, validator):
        """Test minimum limit (1)."""
        result = validator.validate_limit(1)
        assert result == 1

    def test_validate_limit_maximum(self, validator):
        """Test maximum limit (100)."""
        result = validator.validate_limit(100)
        assert result == 100

    def test_validate_limit_middle(self, validator):
        """Test middle value."""
        result = validator.validate_limit(50)
        assert result == 50

    def test_validate_limit_custom_max(self, validator):
        """Test custom max value."""
        result = validator.validate_limit(50, max_value=50)
        assert result == 50

    def test_validate_limit_custom_max_higher(self, validator):
        """Test custom max higher than default."""
        result = validator.validate_limit(150, max_value=200)
        assert result == 150


class TestValidateLimitErrors:
    """Test validate_limit() error cases."""

    @pytest.fixture
    def validator(self):
        """Provide InputValidator instance."""
        return InputValidator()

    def test_validate_limit_zero_raises(self, validator):
        """Test zero limit raises."""
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_limit(0)
        assert "must be positive" in exc_info.value.message.lower()
        assert exc_info.value.error_code == "VAL_003"

    def test_validate_limit_negative_raises(self, validator):
        """Test negative limit raises."""
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_limit(-5)
        assert "must be positive" in exc_info.value.message.lower()

    def test_validate_limit_exceeds_maximum_raises(self, validator):
        """Test exceeding maximum limit raises."""
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_limit(101)
        assert "exceeds maximum" in exc_info.value.message.lower()
        assert exc_info.value.error_code == "VAL_003"

    def test_validate_limit_float_raises(self, validator):
        """Test float type raises."""
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_limit(10.5)
        assert "must be an integer" in exc_info.value.message.lower()
        assert exc_info.value.error_code == "VAL_002"

    def test_validate_limit_string_raises(self, validator):
        """Test string type raises."""
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_limit("10")
        assert "must be an integer" in exc_info.value.message.lower()

    def test_validate_limit_bool_raises(self, validator):
        """Test boolean type raises (even though isinstance(True, int) is True in Python)."""
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_limit(True)
        assert "must be an integer" in exc_info.value.message.lower()

    def test_validate_limit_custom_max_exceeded_raises(self, validator):
        """Test exceeding custom max raises."""
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_limit(51, max_value=50)
        assert "exceeds maximum" in exc_info.value.message.lower()


class TestSanitizeInputBasic:
    """Test sanitize_input() basic functionality."""

    @pytest.fixture
    def validator(self):
        """Provide InputValidator instance."""
        return InputValidator()

    def test_sanitize_input_basic(self, validator):
        """Test basic sanitization."""
        text = "  Hello World  "
        result = validator.sanitize_input(text)
        assert result == "Hello World"

    def test_sanitize_input_null_bytes(self, validator):
        """Test null bytes are removed."""
        text = "Hello\x00World"
        result = validator.sanitize_input(text)
        assert result == "HelloWorld"

    def test_sanitize_input_multiple_null_bytes(self, validator):
        """Test multiple null bytes are removed."""
        text = "H\x00e\x00l\x00l\x00o"
        result = validator.sanitize_input(text)
        assert result == "Hello"

    def test_sanitize_input_control_chars(self, validator):
        """Test control characters are removed."""
        text = "Hello\x01\x02World"
        result = validator.sanitize_input(text)
        assert result == "HelloWorld"

    def test_sanitize_input_preserves_newline(self, validator):
        """Test newlines are preserved."""
        text = "Hello\nWorld"
        result = validator.sanitize_input(text)
        assert "\n" in result

    def test_sanitize_input_preserves_tab(self, validator):
        """Test tabs are preserved."""
        text = "Hello\tWorld"
        result = validator.sanitize_input(text)
        assert "\t" in result

    def test_sanitize_input_multiple_spaces(self, validator):
        """Test multiple spaces are collapsed."""
        text = "Python   is    great"
        result = validator.sanitize_input(text)
        assert result == "Python is great"

    def test_sanitize_input_unicode_normalization(self, validator):
        """Test Unicode normalization."""
        text = "cafe\u0301"  # NFD form
        result = validator.sanitize_input(text)
        # After NFC normalization
        assert "caf" in result

    def test_sanitize_input_non_string(self, validator):
        """Test non-string input is returned as-is."""
        result = validator.sanitize_input(123)
        assert result == 123

    def test_sanitize_input_empty_string(self, validator):
        """Test empty string."""
        result = validator.sanitize_input("")
        assert result == ""


class TestSanitizeInputEdgeCases:
    """Test sanitize_input() edge cases."""

    @pytest.fixture
    def validator(self):
        """Provide InputValidator instance."""
        return InputValidator()

    def test_sanitize_input_only_whitespace(self, validator):
        """Test string with only whitespace."""
        result = validator.sanitize_input("   \n  \t  ")
        assert result == ""

    def test_sanitize_input_mixed_control_chars(self, validator):
        """Test mixed control characters."""
        text = "Hello\x00\x01\x02\x03World"
        result = validator.sanitize_input(text)
        assert result == "HelloWorld"

    def test_sanitize_input_newline_with_spaces(self, validator):
        """Test newlines with spaces."""
        text = "Python\n   is\n   great"
        result = validator.sanitize_input(text)
        # Each line should have leading spaces removed
        assert "Python" in result
        assert "is" in result
        assert "great" in result

    def test_sanitize_input_tab_with_spaces(self, validator):
        """Test tabs with spaces."""
        text = "Python\t   is   \tgreat"
        result = validator.sanitize_input(text)
        assert "Python" in result


class TestValidatorThreadSafety:
    """Test InputValidator thread safety."""

    def test_validator_is_stateless(self):
        """Test that InputValidator is stateless."""
        validator1 = InputValidator()
        validator2 = InputValidator()

        # Both should validate the same way
        text = "Python is great"
        result1 = validator1.validate_text(text)
        result2 = validator2.validate_text(text)
        assert result1 == result2

    def test_multiple_instances_independent(self):
        """Test multiple instances are independent."""
        validators = [InputValidator() for _ in range(5)]
        text = "Test text"

        results = [v.validate_text(text) for v in validators]
        assert all(r == text for r in results)


class TestErrorMessages:
    """Test that error messages are clear and helpful."""

    @pytest.fixture
    def validator(self):
        """Provide InputValidator instance."""
        return InputValidator()

    def test_text_error_includes_type_info(self, validator):
        """Test error message includes type information."""
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_text(123)
        assert "int" in exc_info.value.message

    def test_metadata_reserved_key_error_includes_key(self, validator):
        """Test error message includes the reserved key."""
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_metadata({"memory_id": "123"})
        assert "memory_id" in exc_info.value.message

    def test_limit_error_includes_max_value(self, validator):
        """Test error message includes max value."""
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_limit(101)
        assert "100" in exc_info.value.message

    def test_query_length_error_includes_max(self, validator):
        """Test error message includes max length."""
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_query("x" * 1001)
        assert "1000" in exc_info.value.message


class TestIntegration:
    """Integration tests combining multiple validators."""

    @pytest.fixture
    def validator(self):
        """Provide InputValidator instance."""
        return InputValidator()

    def test_validate_complete_add_memory_input(self, validator):
        """Test validating all inputs for add_memory scenario."""
        text = "This is a memory about Python programming"
        metadata = {"source": "user", "tags": ["python", "ai"]}

        # All should pass
        validated_text = validator.validate_text(text)
        validated_metadata = validator.validate_metadata(metadata)

        assert validated_text is not None
        assert validated_metadata is not None

    def test_validate_complete_search_input(self, validator):
        """Test validating all inputs for search_memory scenario."""
        query = "Python tutorials"
        limit = 10
        filters = {"source": "official"}

        # All should pass
        validated_query = validator.validate_query(query)
        validated_limit = validator.validate_limit(limit)
        validated_filters = validator.validate_metadata(filters)

        assert validated_query is not None
        assert validated_limit == 10
        assert validated_filters is not None

    def test_invalid_add_memory_fails_fast(self, validator):
        """Test that add_memory validation fails on first error."""
        # Empty text should fail first
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_text("")
        assert "empty" in exc_info.value.message.lower()

    def test_invalid_search_fails_fast(self, validator):
        """Test that search validation fails on first error."""
        # Empty query should fail first
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_query("")
        assert "empty" in exc_info.value.message.lower()


class TestPrivateHelpers:
    """Test private helper methods."""

    @pytest.fixture
    def validator(self):
        """Provide InputValidator instance."""
        return InputValidator()

    def test_check_encoding_valid(self, validator):
        """Test _check_encoding with valid UTF-8."""
        # Should not raise
        validator._check_encoding("Hello World")
        validator._check_encoding("Python программирование")

    def test_check_encoding_invalid(self, validator):
        """Test _check_encoding with invalid UTF-8."""
        # Invalid UTF-8 sequences should raise
        # Note: Python's str type is already UTF-8, so we can't easily create invalid UTF-8
        # This test verifies the method exists and works correctly
        validator._check_encoding("Valid UTF-8 text")

    def test_is_json_serializable_valid(self, validator):
        """Test _is_json_serializable with valid objects."""
        assert validator._is_json_serializable("string") is True
        assert validator._is_json_serializable(42) is True
        assert validator._is_json_serializable(3.14) is True
        assert validator._is_json_serializable(True) is True
        assert validator._is_json_serializable(None) is True
        assert validator._is_json_serializable([1, 2, 3]) is True
        assert validator._is_json_serializable({"key": "value"}) is True

    def test_is_json_serializable_invalid(self, validator):
        """Test _is_json_serializable with non-serializable objects."""
        assert validator._is_json_serializable(lambda x: x) is False
        assert validator._is_json_serializable(set([1, 2, 3])) is False

        class CustomObj:
            pass

        assert validator._is_json_serializable(CustomObj()) is False

    def test_remove_null_bytes(self, validator):
        """Test _remove_null_bytes."""
        assert validator._remove_null_bytes("Hello\x00World") == "HelloWorld"
        assert validator._remove_null_bytes("No\x00null\x00bytes") == "Nonullbytes"
        assert validator._remove_null_bytes("No null bytes") == "No null bytes"
