"""Unit tests for zapomni_core.treesitter.models module."""

import pytest
from pydantic import ValidationError

from zapomni_core.treesitter.models import (
    ASTNodeLocation,
    CodeElementType,
    ExtractedCode,
    ParameterInfo,
    ParseResult,
)


class TestCodeElementType:
    """Tests for CodeElementType enum."""

    def test_function_value(self):
        """Test FUNCTION enum value."""
        assert CodeElementType.FUNCTION.value == "function"

    def test_method_value(self):
        """Test METHOD enum value."""
        assert CodeElementType.METHOD.value == "method"

    def test_class_value(self):
        """Test CLASS enum value."""
        assert CodeElementType.CLASS.value == "class"

    def test_interface_value(self):
        """Test INTERFACE enum value."""
        assert CodeElementType.INTERFACE.value == "interface"

    def test_struct_value(self):
        """Test STRUCT enum value."""
        assert CodeElementType.STRUCT.value == "struct"

    def test_enum_value(self):
        """Test ENUM enum value."""
        assert CodeElementType.ENUM.value == "enum"

    def test_all_enum_values(self):
        """Test all enum values are unique."""
        values = [e.value for e in CodeElementType]
        assert len(values) == len(set(values))

    def test_string_enum_behavior(self):
        """Test that CodeElementType is a string enum."""
        assert isinstance(CodeElementType.FUNCTION.value, str)
        # CodeElementType is (str, Enum), so it can be compared to strings
        assert CodeElementType.FUNCTION == "function"


class TestASTNodeLocation:
    """Tests for ASTNodeLocation model."""

    def test_valid_location(self, sample_location):
        """Test creating a valid location."""
        assert sample_location.start_line == 0
        assert sample_location.end_line == 5
        assert sample_location.start_column == 0
        assert sample_location.end_column == 10
        assert sample_location.start_byte == 0
        assert sample_location.end_byte == 100

    def test_location_with_zero_values(self):
        """Test location with all zero values (valid)."""
        location = ASTNodeLocation(
            start_line=0,
            end_line=0,
            start_column=0,
            end_column=0,
            start_byte=0,
            end_byte=0,
        )
        assert location.start_line == 0

    def test_negative_start_line_fails(self):
        """Test that negative start_line fails validation."""
        with pytest.raises(ValidationError):
            ASTNodeLocation(
                start_line=-1,
                end_line=5,
                start_column=0,
                end_column=10,
                start_byte=0,
                end_byte=100,
            )

    def test_negative_end_line_fails(self):
        """Test that negative end_line fails validation."""
        with pytest.raises(ValidationError):
            ASTNodeLocation(
                start_line=0,
                end_line=-1,
                start_column=0,
                end_column=10,
                start_byte=0,
                end_byte=100,
            )

    def test_negative_byte_offset_fails(self):
        """Test that negative byte offset fails validation."""
        with pytest.raises(ValidationError):
            ASTNodeLocation(
                start_line=0,
                end_line=5,
                start_column=0,
                end_column=10,
                start_byte=-1,
                end_byte=100,
            )

    def test_location_is_frozen(self, sample_location):
        """Test that location is immutable (frozen)."""
        with pytest.raises(ValidationError):
            sample_location.start_line = 10


class TestParameterInfo:
    """Tests for ParameterInfo model."""

    def test_valid_parameter(self, sample_parameter):
        """Test creating a valid parameter."""
        assert sample_parameter.name == "param1"
        assert sample_parameter.type_annotation == "str"
        assert sample_parameter.default_value is None

    def test_parameter_with_default(self):
        """Test parameter with default value."""
        param = ParameterInfo(
            name="count",
            type_annotation="int",
            default_value="0",
        )
        assert param.default_value == "0"

    def test_parameter_without_type(self):
        """Test parameter without type annotation."""
        param = ParameterInfo(name="arg")
        assert param.type_annotation is None
        assert param.default_value is None

    def test_empty_name_fails(self):
        """Test that empty name fails validation."""
        with pytest.raises(ValidationError):
            ParameterInfo(name="")

    def test_parameter_is_frozen(self, sample_parameter):
        """Test that parameter is immutable (frozen)."""
        with pytest.raises(ValidationError):
            sample_parameter.name = "new_name"


class TestExtractedCode:
    """Tests for ExtractedCode model."""

    def test_valid_extracted_code(self, sample_extracted_code):
        """Test creating valid extracted code."""
        assert sample_extracted_code.name == "test_function"
        assert sample_extracted_code.qualified_name == "test_module.test_function"
        assert sample_extracted_code.element_type == CodeElementType.FUNCTION
        assert sample_extracted_code.language == "python"

    def test_empty_name_fails(self, sample_location):
        """Test that empty name fails validation."""
        with pytest.raises(ValidationError):
            ExtractedCode(
                name="",
                qualified_name="test",
                element_type=CodeElementType.FUNCTION,
                language="python",
                file_path="/test.py",
                location=sample_location,
                source_code="def test(): pass",
            )

    def test_empty_source_code_fails(self, sample_location):
        """Test that empty source code fails validation."""
        with pytest.raises(ValidationError):
            ExtractedCode(
                name="test",
                qualified_name="test",
                element_type=CodeElementType.FUNCTION,
                language="python",
                file_path="/test.py",
                location=sample_location,
                source_code="",
            )

    def test_default_values(self, sample_location):
        """Test default values for optional fields."""
        code = ExtractedCode(
            name="test",
            qualified_name="test",
            element_type=CodeElementType.FUNCTION,
            language="python",
            file_path="/test.py",
            location=sample_location,
            source_code="def test(): pass",
        )
        assert code.docstring is None
        assert code.parent_class is None
        assert code.methods == []
        assert code.bases == []
        assert code.parameters == []
        assert code.return_type is None
        assert code.decorators == []
        assert code.is_async is False
        assert code.is_generator is False
        assert code.is_static is False
        assert code.is_abstract is False
        assert code.is_private is False
        assert code.line_count == 0

    def test_extracted_code_with_docstring(self, sample_location):
        """Test extracted code with docstring."""
        code = ExtractedCode(
            name="test",
            qualified_name="test",
            element_type=CodeElementType.FUNCTION,
            language="python",
            file_path="/test.py",
            location=sample_location,
            source_code="def test(): pass",
            docstring="Test docstring.",
        )
        assert code.docstring == "Test docstring."

    def test_extracted_code_with_parameters(self, sample_location, sample_parameter):
        """Test extracted code with parameters."""
        code = ExtractedCode(
            name="test",
            qualified_name="test",
            element_type=CodeElementType.FUNCTION,
            language="python",
            file_path="/test.py",
            location=sample_location,
            source_code="def test(param1: str): pass",
            parameters=[sample_parameter],
        )
        assert len(code.parameters) == 1
        assert code.parameters[0].name == "param1"

    def test_extracted_code_with_decorators(self, sample_location):
        """Test extracted code with decorators."""
        code = ExtractedCode(
            name="test",
            qualified_name="test",
            element_type=CodeElementType.METHOD,
            language="python",
            file_path="/test.py",
            location=sample_location,
            source_code="@staticmethod\ndef test(): pass",
            decorators=["staticmethod"],
            is_static=True,
        )
        assert "staticmethod" in code.decorators
        assert code.is_static is True

    def test_class_with_methods_and_bases(self, sample_location):
        """Test class with methods and base classes."""
        code = ExtractedCode(
            name="MyClass",
            qualified_name="module.MyClass",
            element_type=CodeElementType.CLASS,
            language="python",
            file_path="/test.py",
            location=sample_location,
            source_code="class MyClass(BaseClass): pass",
            methods=["__init__", "process"],
            bases=["BaseClass"],
        )
        assert code.methods == ["__init__", "process"]
        assert code.bases == ["BaseClass"]


class TestParseResult:
    """Tests for ParseResult model."""

    def test_valid_parse_result(self, sample_parse_result):
        """Test creating valid parse result."""
        assert sample_parse_result.file_path == "/path/to/test.py"
        assert sample_parse_result.language == "python"
        assert len(sample_parse_result.functions) == 1
        assert len(sample_parse_result.classes) == 0
        assert sample_parse_result.parse_time_ms == 10.5
        assert sample_parse_result.errors == []

    def test_empty_file_path_fails(self):
        """Test that empty file path fails validation."""
        with pytest.raises(ValidationError):
            ParseResult(
                file_path="",
                language="python",
                parse_time_ms=10.0,
            )

    def test_negative_parse_time_fails(self):
        """Test that negative parse time fails validation."""
        with pytest.raises(ValidationError):
            ParseResult(
                file_path="/test.py",
                language="python",
                parse_time_ms=-1.0,
            )

    def test_parse_result_with_errors(self):
        """Test parse result with errors."""
        result = ParseResult(
            file_path="/test.py",
            language="python",
            parse_time_ms=5.0,
            errors=["Syntax error at line 10", "Unexpected token"],
        )
        assert len(result.errors) == 2
        assert "Syntax error at line 10" in result.errors

    def test_default_empty_lists(self):
        """Test that lists default to empty."""
        result = ParseResult(
            file_path="/test.py",
            language="python",
            parse_time_ms=1.0,
        )
        assert result.functions == []
        assert result.classes == []
        assert result.errors == []
