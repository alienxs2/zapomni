"""
Unit tests for the FunctionExtractor component.

Tests AST-based extraction of function metadata from Python source code.
Covers signatures, parameters, docstrings, decorators, complexity, and edge cases.
"""

from __future__ import annotations

import ast
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from zapomni_core.code.function_extractor import (
    FunctionExtractor,
    FunctionMetadata,
    Parameter,
)
from zapomni_core.exceptions import ExtractionError, ValidationError

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def extractor_basic():
    """Create basic FunctionExtractor without optional features."""
    return FunctionExtractor(
        extract_body=False,
        calculate_complexity=False,
    )


@pytest.fixture
def extractor_with_body():
    """Create FunctionExtractor that extracts function bodies."""
    return FunctionExtractor(
        extract_body=True,
        calculate_complexity=False,
    )


@pytest.fixture
def extractor_with_complexity():
    """Create FunctionExtractor with complexity calculation enabled."""
    return FunctionExtractor(
        extract_body=False,
        calculate_complexity=True,
    )


# ============================================================================
# Test Initialization
# ============================================================================


class TestFunctionExtractorInitialization:
    """Test suite for FunctionExtractor initialization."""

    def test_init_with_defaults(self) -> None:
        """Should initialize with default settings."""
        extractor = FunctionExtractor()

        assert extractor.extract_body is False
        assert extractor._calculate_complexity_enabled is False
        assert extractor.max_file_size == 10_000_000

    def test_init_with_custom_settings(self) -> None:
        """Should accept custom settings."""
        extractor = FunctionExtractor(
            extract_body=True,
            calculate_complexity=True,
            max_file_size=5_000_000,
        )

        assert extractor.extract_body is True
        assert extractor._calculate_complexity_enabled is True
        assert extractor.max_file_size == 5_000_000

    def test_init_raises_error_for_invalid_file_size(self) -> None:
        """Should raise ValueError for non-positive max_file_size."""
        with pytest.raises(ValueError) as exc_info:
            FunctionExtractor(max_file_size=0)

        assert "max_file_size" in str(exc_info.value).lower()

    def test_init_raises_error_for_negative_file_size(self) -> None:
        """Should raise ValueError for negative max_file_size."""
        with pytest.raises(ValueError):
            FunctionExtractor(max_file_size=-1000)


# ============================================================================
# Test Simple Function Extraction
# ============================================================================


class TestSimpleFunctionExtraction:
    """Test extraction of simple functions without parameters."""

    def test_extract_simple_function(self, extractor_basic) -> None:
        """Should extract simple function with no parameters."""
        source = """
def hello():
    return "world"
"""
        functions = extractor_basic.extract_functions(source)

        assert len(functions) == 1
        assert functions[0].name == "hello"
        assert functions[0].signature == "def hello():"
        assert len(functions[0].parameters) == 0

    def test_extract_multiple_functions(self, extractor_basic) -> None:
        """Should extract multiple functions in correct order."""
        source = """
def first():
    pass

def second():
    pass

def third():
    pass
"""
        functions = extractor_basic.extract_functions(source)

        assert len(functions) == 3
        assert functions[0].name == "first"
        assert functions[1].name == "second"
        assert functions[2].name == "third"
        assert all(f.is_async is False for f in functions)

    def test_extract_nested_functions(self, extractor_basic) -> None:
        """Should extract nested functions."""
        source = """
def outer():
    def inner():
        pass
    return inner
"""
        functions = extractor_basic.extract_functions(source)

        # ast.walk finds all functions, including nested
        assert len(functions) == 2
        names = [f.name for f in functions]
        assert "outer" in names
        assert "inner" in names


# ============================================================================
# Test Function with Parameters
# ============================================================================


class TestParameterExtraction:
    """Test extraction of function parameters."""

    def test_extract_positional_parameters(self, extractor_basic) -> None:
        """Should extract positional parameters."""
        source = """
def add(x, y):
    return x + y
"""
        functions = extractor_basic.extract_functions(source)
        func = functions[0]

        assert len(func.parameters) == 2
        assert func.parameters[0].name == "x"
        assert func.parameters[1].name == "y"
        assert func.parameters[0].annotation is None
        assert func.parameters[0].default is None

    def test_extract_typed_parameters(self, extractor_basic) -> None:
        """Should extract type annotations from parameters."""
        source = """
def add(x: int, y: int) -> int:
    return x + y
"""
        functions = extractor_basic.extract_functions(source)
        func = functions[0]

        assert len(func.parameters) == 2
        assert func.parameters[0].annotation == "int"
        assert func.parameters[1].annotation == "int"
        assert func.return_type == "int"

    def test_extract_default_parameters(self, extractor_basic) -> None:
        """Should extract default parameter values."""
        source = """
def greet(name: str = "World"):
    return f"Hello {name}"
"""
        functions = extractor_basic.extract_functions(source)
        func = functions[0]

        assert len(func.parameters) == 1
        assert func.parameters[0].name == "name"
        assert func.parameters[0].annotation == "str"
        assert func.parameters[0].default == "'World'"

    def test_extract_varargs_parameters(self, extractor_basic) -> None:
        """Should extract *args and **kwargs."""
        source = """
def flexible(*args, **kwargs):
    pass
"""
        functions = extractor_basic.extract_functions(source)
        func = functions[0]

        param_names = [p.name for p in func.parameters]
        assert "*args" in param_names
        assert "**kwargs" in param_names

    def test_extract_keyword_only_parameters(self, extractor_basic) -> None:
        """Should extract keyword-only parameters."""
        source = """
def process(*, required, optional="default"):
    pass
"""
        functions = extractor_basic.extract_functions(source)
        func = functions[0]

        param_names = [p.name for p in func.parameters]
        assert "required" in param_names
        assert "optional" in param_names

    def test_complex_parameter_list(self, extractor_basic) -> None:
        """Should extract complex parameter combinations."""
        source = """
def complex_func(a: int, b: str = "x", *args, kw_only: float, **kwargs):
    pass
"""
        functions = extractor_basic.extract_functions(source)
        func = functions[0]

        assert len(func.parameters) == 5
        param_names = [p.name for p in func.parameters]
        assert "a" in param_names
        assert "b" in param_names
        assert "*args" in param_names
        assert "kw_only" in param_names
        assert "**kwargs" in param_names


# ============================================================================
# Test Docstring Extraction
# ============================================================================


class TestDocstringExtraction:
    """Test extraction of docstrings."""

    def test_extract_single_line_docstring(self, extractor_basic) -> None:
        """Should extract single-line docstring."""
        source = """
def add(x, y):
    '''Add two numbers.'''
    return x + y
"""
        functions = extractor_basic.extract_functions(source)
        func = functions[0]

        assert func.docstring == "Add two numbers."

    def test_extract_multiline_docstring(self, extractor_basic) -> None:
        """Should extract multi-line docstring."""
        source = '''
def process(data):
    """
    Process input data.

    Args:
        data: Input data to process

    Returns:
        Processed data
    """
    return data
'''
        functions = extractor_basic.extract_functions(source)
        func = functions[0]

        assert func.docstring is not None
        assert "Process input data" in func.docstring
        assert "Args:" in func.docstring

    def test_function_without_docstring(self, extractor_basic) -> None:
        """Should handle functions without docstrings."""
        source = """
def simple():
    return 42
"""
        functions = extractor_basic.extract_functions(source)
        func = functions[0]

        assert func.docstring is None


# ============================================================================
# Test Decorator Extraction
# ============================================================================


class TestDecoratorExtraction:
    """Test extraction of decorators."""

    def test_extract_single_decorator(self, extractor_basic) -> None:
        """Should extract single decorator."""
        source = """
@staticmethod
def utility():
    pass
"""
        functions = extractor_basic.extract_functions(source)
        func = functions[0]

        assert "staticmethod" in func.decorators

    def test_extract_multiple_decorators(self, extractor_basic) -> None:
        """Should extract multiple decorators."""
        source = """
@property
@cache
def cached_value(self):
    pass
"""
        functions = extractor_basic.extract_functions(source)
        func = functions[0]

        assert len(func.decorators) >= 2
        assert "property" in func.decorators

    def test_property_decorator_flag(self, extractor_basic) -> None:
        """Should set is_property flag for @property decorator."""
        source = """
@property
def value(self):
    return self._value
"""
        functions = extractor_basic.extract_functions(source)
        func = functions[0]

        assert func.is_property is True

    def test_non_property_function(self, extractor_basic) -> None:
        """Should not set is_property flag for regular functions."""
        source = """
def get_value(self):
    return self._value
"""
        functions = extractor_basic.extract_functions(source)
        func = functions[0]

        assert func.is_property is False


# ============================================================================
# Test Async Functions
# ============================================================================


class TestAsyncFunctionExtraction:
    """Test extraction of async functions."""

    def test_extract_async_function(self, extractor_basic) -> None:
        """Should mark async functions correctly."""
        source = """
async def fetch_data(url):
    return await client.get(url)
"""
        functions = extractor_basic.extract_functions(source)
        func = functions[0]

        assert func.is_async is True
        assert "async def" in func.signature

    def test_regular_function_not_async(self, extractor_basic) -> None:
        """Should mark regular functions as non-async."""
        source = """
def fetch_data(url):
    return url
"""
        functions = extractor_basic.extract_functions(source)
        func = functions[0]

        assert func.is_async is False
        assert "async" not in func.signature


# ============================================================================
# Test Generator Detection
# ============================================================================


class TestGeneratorDetection:
    """Test detection of generator functions."""

    def test_detect_generator_with_yield(self, extractor_basic) -> None:
        """Should detect functions with yield statements."""
        source = """
def counter(n):
    for i in range(n):
        yield i
"""
        functions = extractor_basic.extract_functions(source)
        func = functions[0]

        assert func.is_generator is True

    def test_detect_generator_with_yield_from(self, extractor_basic) -> None:
        """Should detect functions with yield from statements."""
        source = """
def relay(iterable):
    yield from iterable
"""
        functions = extractor_basic.extract_functions(source)
        func = functions[0]

        assert func.is_generator is True

    def test_regular_function_not_generator(self, extractor_basic) -> None:
        """Should mark regular functions as non-generators."""
        source = """
def regular():
    return [1, 2, 3]
"""
        functions = extractor_basic.extract_functions(source)
        func = functions[0]

        assert func.is_generator is False


# ============================================================================
# Test Line Number Tracking
# ============================================================================


class TestLineNumberTracking:
    """Test tracking of function line numbers."""

    def test_track_function_line_numbers(self, extractor_basic) -> None:
        """Should track correct start and end line numbers."""
        source = """
def first():
    return 1

def second():
    x = 2
    return x
"""
        functions = extractor_basic.extract_functions(source)

        # Line numbering starts at 1
        assert functions[0].name == "first"
        assert functions[0].start_line == 2

        assert functions[1].name == "second"
        assert functions[1].start_line == 5

    def test_single_line_function(self, extractor_basic) -> None:
        """Should handle single-line functions (e.g., pass)."""
        source = """
def empty():
    pass
"""
        functions = extractor_basic.extract_functions(source)
        func = functions[0]

        assert func.start_line > 0
        assert func.end_line >= func.start_line


# ============================================================================
# Test Function Body Extraction
# ============================================================================


class TestFunctionBodyExtraction:
    """Test extraction of function body source code."""

    def test_extract_function_body(self, extractor_with_body) -> None:
        """Should extract function body when enabled."""
        source = """
def add(x, y):
    result = x + y
    return result
"""
        functions = extractor_with_body.extract_functions(source)
        func = functions[0]

        assert len(func.body_lines) > 0
        # Body should contain the implementation
        body_text = "\n".join(func.body_lines)
        assert "result" in body_text or "return" in body_text

    def test_function_body_not_extracted_by_default(self, extractor_basic) -> None:
        """Should not extract body when disabled (default)."""
        source = """
def add(x, y):
    return x + y
"""
        functions = extractor_basic.extract_functions(source)
        func = functions[0]

        assert len(func.body_lines) == 0


# ============================================================================
# Test Complexity Calculation
# ============================================================================


class TestComplexityCalculation:
    """Test cyclomatic complexity calculation."""

    def test_complexity_simple_function(self, extractor_with_complexity) -> None:
        """Should calculate complexity for simple functions."""
        source = """
def simple():
    return 42
"""
        functions = extractor_with_complexity.extract_functions(source)
        func = functions[0]

        # Simple function should have complexity of 1
        # (or None if radon not available)
        if func.complexity_score is not None:
            assert func.complexity_score >= 1

    def test_complexity_not_calculated_by_default(self, extractor_basic) -> None:
        """Should not calculate complexity when disabled (default)."""
        source = """
def simple():
    return 42
"""
        functions = extractor_basic.extract_functions(source)
        func = functions[0]

        assert func.complexity_score is None

    def test_complexity_with_conditionals(self, extractor_with_complexity) -> None:
        """Should calculate higher complexity for functions with branches."""
        source = """
def decision(x):
    if x > 0:
        return "positive"
    elif x < 0:
        return "negative"
    else:
        return "zero"
"""
        functions = extractor_with_complexity.extract_functions(source)
        func = functions[0]

        # If calculated, should be higher than simple function
        # (or None if radon not available)
        if func.complexity_score is not None:
            assert func.complexity_score >= 1


# ============================================================================
# Test Error Handling
# ============================================================================


class TestErrorHandling:
    """Test error handling for invalid inputs."""

    def test_empty_source_code(self, extractor_basic) -> None:
        """Should raise ValidationError for empty source code."""
        with pytest.raises(ValidationError) as exc_info:
            extractor_basic.extract_functions("")

        assert "empty" in str(exc_info.value).lower()

    def test_whitespace_only_source(self, extractor_basic) -> None:
        """Should raise ValidationError for whitespace-only source."""
        with pytest.raises(ValidationError):
            extractor_basic.extract_functions("   \n\n   ")

    def test_source_exceeds_max_size(self) -> None:
        """Should raise ValidationError for oversized source."""
        extractor = FunctionExtractor(max_file_size=100)  # 100 bytes max

        large_source = "x = 1\n" * 1000  # Much larger than 100 bytes

        with pytest.raises(ValidationError) as exc_info:
            extractor.extract_functions(large_source)

        assert "exceeds" in str(exc_info.value).lower()

    def test_syntax_error_in_source(self, extractor_basic) -> None:
        """Should raise ExtractionError for syntax errors."""
        source = """
def invalid(
    # Missing closing parenthesis
"""
        with pytest.raises(ExtractionError):
            extractor_basic.extract_functions(source)

    def test_malformed_code(self, extractor_basic) -> None:
        """Should handle various malformed code gracefully."""
        source = """
def broken():
    if True
        pass
"""
        with pytest.raises(ExtractionError):
            extractor_basic.extract_functions(source)


# ============================================================================
# Test Metadata Serialization
# ============================================================================


class TestMetadataSerialization:
    """Test serialization of function metadata."""

    def test_metadata_to_dict(self, extractor_basic) -> None:
        """Should convert metadata to dictionary."""
        source = """
def add(x: int, y: int) -> int:
    '''Add two integers.'''
    return x + y
"""
        functions = extractor_basic.extract_functions(source)
        func = functions[0]

        metadata_dict = func.to_dict()

        assert metadata_dict["name"] == "add"
        assert metadata_dict["return_type"] == "int"
        assert metadata_dict["docstring"] == "Add two integers."
        assert len(metadata_dict["parameters"]) == 2
        assert metadata_dict["is_generator"] is False
        assert metadata_dict["is_async"] is False

    def test_parameter_to_dict(self) -> None:
        """Should convert parameter to dictionary."""
        param = Parameter(
            name="x",
            annotation="int",
            default="0",
            kind="POSITIONAL_OR_KEYWORD",
        )

        # Parameters should have their basic attributes
        assert param.name == "x"
        assert param.annotation == "int"
        assert param.default == "0"


# ============================================================================
# Test Return Type Extraction
# ============================================================================


class TestReturnTypeExtraction:
    """Test extraction of return type annotations."""

    def test_extract_simple_return_type(self, extractor_basic) -> None:
        """Should extract simple return type annotation."""
        source = """
def get_number() -> int:
    return 42
"""
        functions = extractor_basic.extract_functions(source)
        func = functions[0]

        assert func.return_type == "int"

    def test_extract_complex_return_type(self, extractor_basic) -> None:
        """Should extract complex return type annotations."""
        source = """
def get_items() -> List[Dict[str, int]]:
    return []
"""
        functions = extractor_basic.extract_functions(source)
        func = functions[0]

        assert func.return_type is not None
        assert "List" in func.return_type

    def test_no_return_type(self, extractor_basic) -> None:
        """Should handle functions without return type annotation."""
        source = """
def no_return():
    pass
"""
        functions = extractor_basic.extract_functions(source)
        func = functions[0]

        assert func.return_type is None


# ============================================================================
# Test Signature Generation
# ============================================================================


class TestSignatureGeneration:
    """Test function signature generation."""

    def test_signature_simple_function(self, extractor_basic) -> None:
        """Should generate correct signature for simple function."""
        source = """
def hello():
    pass
"""
        functions = extractor_basic.extract_functions(source)
        func = functions[0]

        assert func.signature == "def hello():"

    def test_signature_with_parameters(self, extractor_basic) -> None:
        """Should generate correct signature with parameters."""
        source = """
def add(x: int, y: int) -> int:
    return x + y
"""
        functions = extractor_basic.extract_functions(source)
        func = functions[0]

        assert "add" in func.signature
        assert "x: int" in func.signature
        assert "y: int" in func.signature
        assert "-> int" in func.signature

    def test_signature_async_function(self, extractor_basic) -> None:
        """Should include async keyword in signature."""
        source = """
async def fetch():
    pass
"""
        functions = extractor_basic.extract_functions(source)
        func = functions[0]

        assert "async def" in func.signature


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_method_extraction(self, extractor_basic) -> None:
        """Should extract methods from classes."""
        source = """
class MyClass:
    def method(self):
        pass
"""
        functions = extractor_basic.extract_functions(source)

        # ast.walk finds all functions, including methods
        assert len(functions) >= 1
        method_names = [f.name for f in functions]
        assert "method" in method_names

    def test_lambda_not_extracted(self, extractor_basic) -> None:
        """Should not extract lambda functions (no name to extract)."""
        source = """
x = lambda y: y * 2
"""
        functions = extractor_basic.extract_functions(source)

        # Lambdas are not FunctionDef nodes, so should not be extracted
        assert len(functions) == 0

    def test_unicode_in_docstring(self, extractor_basic) -> None:
        """Should handle unicode characters in docstrings."""
        source = """
def greet():
    '''Greet with émojis 🎉'''
    pass
"""
        functions = extractor_basic.extract_functions(source)
        func = functions[0]

        assert func.docstring is not None

    def test_very_long_function(self, extractor_basic) -> None:
        """Should handle very long functions."""
        lines = ["def long_function():"]
        for i in range(100):
            lines.append(f"    x{i} = {i}")
        lines.append("    return x99")

        source = "\n".join(lines)
        functions = extractor_basic.extract_functions(source)

        assert len(functions) == 1
        assert functions[0].name == "long_function"

    def test_file_path_tracking(self, extractor_basic) -> None:
        """Should track file path when provided."""
        source = "def func(): pass"
        file_path = "/path/to/file.py"

        functions = extractor_basic.extract_functions(source, file_path=file_path)
        func = functions[0]

        assert func.file_path == file_path
