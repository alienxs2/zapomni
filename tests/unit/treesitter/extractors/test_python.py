"""Tests for PythonExtractor.

This module contains comprehensive tests for the Python-specific code extractor,
covering docstrings, decorators, type hints, async functions, generators, and more.

Test count target: 40+ tests as per Issue #19 requirements.
"""

import pytest
from tree_sitter import Tree
from tree_sitter_language_pack import get_parser

from zapomni_core.treesitter.extractors.python import PythonExtractor
from zapomni_core.treesitter.models import CodeElementType, ParameterInfo

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def extractor():
    """Create a PythonExtractor instance."""
    return PythonExtractor()


@pytest.fixture
def parse_python():
    """Factory fixture to parse Python source code."""
    parser = get_parser("python")

    def _parse(source: str) -> tuple[Tree, bytes]:
        source_bytes = source.encode("utf-8")
        tree = parser.parse(source_bytes)
        return tree, source_bytes

    return _parse


# =============================================================================
# Test PythonExtractor Properties
# =============================================================================


class TestPythonExtractorProperties:
    """Tests for PythonExtractor basic properties."""

    def test_language_name(self, extractor):
        """Test that language name is 'python'."""
        assert extractor.language_name == "python"

    def test_supported_node_types(self, extractor):
        """Test supported node types include function and class definitions."""
        node_types = extractor.supported_node_types
        assert "function_definition" in node_types
        assert "class_definition" in node_types
        assert "decorated_definition" in node_types


# =============================================================================
# Test Function Extraction - Basic
# =============================================================================


class TestExtractFunctionsBasic:
    """Tests for basic function extraction."""

    def test_extract_simple_function(self, extractor, parse_python):
        """Test extraction of a simple function."""
        source = """
def hello():
    pass
"""
        tree, source_bytes = parse_python(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.py")

        assert len(functions) == 1
        assert functions[0].name == "hello"
        assert functions[0].element_type == CodeElementType.FUNCTION

    def test_extract_multiple_functions(self, extractor, parse_python):
        """Test extraction of multiple functions."""
        source = """
def func1():
    pass

def func2():
    pass

def func3():
    pass
"""
        tree, source_bytes = parse_python(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.py")

        assert len(functions) == 3
        names = [f.name for f in functions]
        assert "func1" in names
        assert "func2" in names
        assert "func3" in names

    def test_function_has_location(self, extractor, parse_python):
        """Test that extracted function has location info."""
        source = """def hello():
    pass
"""
        tree, source_bytes = parse_python(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.py")

        assert len(functions) == 1
        func = functions[0]
        assert func.location is not None
        assert func.location.start_line == 0
        assert func.location.start_column == 0

    def test_function_has_source_code(self, extractor, parse_python):
        """Test that extracted function has source code."""
        source = """def hello():
    print("Hello")
"""
        tree, source_bytes = parse_python(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.py")

        assert len(functions) == 1
        assert "def hello():" in functions[0].source_code
        assert 'print("Hello")' in functions[0].source_code


# =============================================================================
# Test Function Extraction - Docstrings
# =============================================================================


class TestExtractFunctionsDocstrings:
    """Tests for docstring extraction."""

    def test_extract_function_with_docstring(self, extractor, parse_python):
        """Test extraction of function with docstring."""
        source = '''
def hello():
    """Say hello."""
    print("Hello")
'''
        tree, source_bytes = parse_python(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.py")

        assert len(functions) == 1
        assert functions[0].docstring == "Say hello."

    def test_extract_function_with_multiline_docstring(self, extractor, parse_python):
        """Test extraction of function with multiline docstring."""
        source = '''
def calculate(x, y):
    """
    Calculate the sum of x and y.

    Args:
        x: First number
        y: Second number

    Returns:
        The sum of x and y
    """
    return x + y
'''
        tree, source_bytes = parse_python(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.py")

        assert len(functions) == 1
        docstring = functions[0].docstring
        assert docstring is not None
        assert "Calculate the sum" in docstring
        assert "Args:" in docstring
        assert "Returns:" in docstring

    def test_extract_function_with_single_quote_docstring(self, extractor, parse_python):
        """Test extraction of function with single-quote docstring."""
        source = """
def hello():
    '''Single quote docstring.'''
    pass
"""
        tree, source_bytes = parse_python(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.py")

        assert len(functions) == 1
        assert functions[0].docstring == "Single quote docstring."

    def test_function_without_docstring(self, extractor, parse_python):
        """Test function without docstring has None."""
        source = """
def hello():
    pass
"""
        tree, source_bytes = parse_python(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.py")

        assert len(functions) == 1
        assert functions[0].docstring is None

    def test_google_style_docstring(self, extractor, parse_python):
        """Test Google-style docstring extraction."""
        source = '''
def fetch_data(url, timeout=30):
    """Fetch data from a URL.

    Args:
        url: The URL to fetch from.
        timeout: Request timeout in seconds.

    Returns:
        The response data as bytes.

    Raises:
        ConnectionError: If the connection fails.
    """
    pass
'''
        tree, source_bytes = parse_python(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.py")

        assert len(functions) == 1
        docstring = functions[0].docstring
        assert "Args:" in docstring
        assert "Returns:" in docstring
        assert "Raises:" in docstring

    def test_numpy_style_docstring(self, extractor, parse_python):
        """Test NumPy-style docstring extraction."""
        source = '''
def calculate_mean(values):
    """
    Calculate the arithmetic mean.

    Parameters
    ----------
    values : array-like
        Input values.

    Returns
    -------
    float
        The arithmetic mean.
    """
    pass
'''
        tree, source_bytes = parse_python(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.py")

        assert len(functions) == 1
        docstring = functions[0].docstring
        assert "Parameters" in docstring
        assert "Returns" in docstring

    def test_sphinx_style_docstring(self, extractor, parse_python):
        """Test Sphinx-style docstring extraction."""
        source = '''
def divide(a, b):
    """
    Divide a by b.

    :param a: The dividend.
    :param b: The divisor.
    :return: The quotient.
    :raises ZeroDivisionError: If b is zero.
    """
    return a / b
'''
        tree, source_bytes = parse_python(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.py")

        assert len(functions) == 1
        docstring = functions[0].docstring
        assert ":param a:" in docstring
        assert ":return:" in docstring


# =============================================================================
# Test Function Extraction - Parameters
# =============================================================================


class TestExtractFunctionsParameters:
    """Tests for parameter extraction."""

    def test_extract_function_with_params(self, extractor, parse_python):
        """Test extraction of function with parameters."""
        source = """
def add(a, b):
    return a + b
"""
        tree, source_bytes = parse_python(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.py")

        assert len(functions) == 1
        params = functions[0].parameters
        assert len(params) == 2
        assert params[0].name == "a"
        assert params[1].name == "b"

    def test_extract_function_with_typed_params(self, extractor, parse_python):
        """Test extraction of function with type annotations."""
        source = """
def add(a: int, b: int) -> int:
    return a + b
"""
        tree, source_bytes = parse_python(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.py")

        assert len(functions) == 1
        params = functions[0].parameters
        assert len(params) == 2
        assert params[0].name == "a"
        assert params[0].type_annotation == "int"
        assert params[1].name == "b"
        assert params[1].type_annotation == "int"

    def test_extract_function_with_default_params(self, extractor, parse_python):
        """Test extraction of function with default values."""
        source = """
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"
"""
        tree, source_bytes = parse_python(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.py")

        assert len(functions) == 1
        params = functions[0].parameters
        assert len(params) == 2
        assert params[0].name == "name"
        assert params[0].default_value is None
        assert params[1].name == "greeting"
        assert params[1].default_value == '"Hello"'

    def test_extract_function_with_typed_default_params(self, extractor, parse_python):
        """Test extraction of function with typed default params."""
        source = """
def process(data: list, count: int = 10) -> None:
    pass
"""
        tree, source_bytes = parse_python(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.py")

        assert len(functions) == 1
        params = functions[0].parameters
        assert len(params) == 2
        assert params[0].name == "data"
        assert params[0].type_annotation == "list"
        assert params[1].name == "count"
        assert params[1].type_annotation == "int"
        assert params[1].default_value == "10"

    def test_extract_function_with_args_kwargs(self, extractor, parse_python):
        """Test extraction of function with *args and **kwargs."""
        source = """
def variadic(*args, **kwargs):
    pass
"""
        tree, source_bytes = parse_python(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.py")

        assert len(functions) == 1
        params = functions[0].parameters
        param_names = [p.name for p in params]
        assert "*args" in param_names
        assert "**kwargs" in param_names

    def test_extract_function_with_complex_type_hints(self, extractor, parse_python):
        """Test extraction of function with complex type hints."""
        source = """
from typing import List, Optional, Dict

def process(
    items: List[str],
    config: Optional[Dict[str, int]] = None
) -> List[int]:
    pass
"""
        tree, source_bytes = parse_python(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.py")

        assert len(functions) == 1
        params = functions[0].parameters
        assert params[0].type_annotation == "List[str]"
        assert "Optional" in params[1].type_annotation or "Dict" in params[1].type_annotation


# =============================================================================
# Test Function Extraction - Return Types
# =============================================================================


class TestExtractFunctionsReturnTypes:
    """Tests for return type extraction."""

    def test_extract_function_with_return_type(self, extractor, parse_python):
        """Test extraction of function with return type."""
        source = """
def add(a: int, b: int) -> int:
    return a + b
"""
        tree, source_bytes = parse_python(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.py")

        assert len(functions) == 1
        assert functions[0].return_type == "int"

    def test_function_without_return_type(self, extractor, parse_python):
        """Test function without return type has None."""
        source = """
def hello():
    print("Hello")
"""
        tree, source_bytes = parse_python(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.py")

        assert len(functions) == 1
        assert functions[0].return_type is None

    def test_function_with_none_return_type(self, extractor, parse_python):
        """Test function with None return type."""
        source = """
def process() -> None:
    pass
"""
        tree, source_bytes = parse_python(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.py")

        assert len(functions) == 1
        assert functions[0].return_type == "None"


# =============================================================================
# Test Function Extraction - Decorators
# =============================================================================


class TestExtractFunctionsDecorators:
    """Tests for decorator extraction."""

    def test_extract_decorated_function(self, extractor, parse_python):
        """Test extraction of decorated function."""
        source = """
@decorator
def hello():
    pass
"""
        tree, source_bytes = parse_python(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.py")

        assert len(functions) == 1
        assert "decorator" in functions[0].decorators

    def test_extract_function_with_multiple_decorators(self, extractor, parse_python):
        """Test extraction of function with multiple decorators."""
        source = """
@decorator1
@decorator2
@decorator3
def hello():
    pass
"""
        tree, source_bytes = parse_python(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.py")

        assert len(functions) == 1
        decorators = functions[0].decorators
        assert len(decorators) == 3
        assert "decorator1" in decorators
        assert "decorator2" in decorators
        assert "decorator3" in decorators

    def test_extract_staticmethod(self, extractor, parse_python):
        """Test extraction of staticmethod decorator."""
        source = """
class MyClass:
    @staticmethod
    def static_func():
        pass
"""
        tree, source_bytes = parse_python(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.py")

        assert len(functions) == 1
        assert "staticmethod" in functions[0].decorators
        assert functions[0].is_static is True

    def test_extract_classmethod(self, extractor, parse_python):
        """Test extraction of classmethod decorator."""
        source = """
class MyClass:
    @classmethod
    def class_func(cls):
        pass
"""
        tree, source_bytes = parse_python(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.py")

        assert len(functions) == 1
        assert "classmethod" in functions[0].decorators

    def test_extract_property_decorator(self, extractor, parse_python):
        """Test extraction of property decorator."""
        source = """
class MyClass:
    @property
    def value(self):
        return self._value
"""
        tree, source_bytes = parse_python(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.py")

        assert len(functions) == 1
        assert "property" in functions[0].decorators

    def test_extract_abstractmethod(self, extractor, parse_python):
        """Test extraction of abstractmethod decorator."""
        source = """
from abc import ABC, abstractmethod

class Base(ABC):
    @abstractmethod
    def process(self):
        pass
"""
        tree, source_bytes = parse_python(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.py")

        assert len(functions) == 1
        assert "abstractmethod" in functions[0].decorators
        assert functions[0].is_abstract is True

    def test_extract_decorator_with_args(self, extractor, parse_python):
        """Test extraction of decorator with arguments."""
        source = """
@decorator(arg1, arg2="value")
def hello():
    pass
"""
        tree, source_bytes = parse_python(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.py")

        assert len(functions) == 1
        assert "decorator" in functions[0].decorators

    def test_extract_dotted_decorator(self, extractor, parse_python):
        """Test extraction of dotted decorator like functools.wraps."""
        source = """
import functools

@functools.wraps
def hello():
    pass
"""
        tree, source_bytes = parse_python(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.py")

        assert len(functions) == 1
        assert "functools.wraps" in functions[0].decorators


# =============================================================================
# Test Function Extraction - Async and Generators
# =============================================================================


class TestExtractFunctionsAsyncGenerators:
    """Tests for async functions and generators."""

    def test_extract_async_function(self, extractor, parse_python):
        """Test extraction of async function."""
        source = """
async def fetch_data():
    pass
"""
        tree, source_bytes = parse_python(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.py")

        assert len(functions) == 1
        assert functions[0].is_async is True

    def test_extract_async_method(self, extractor, parse_python):
        """Test extraction of async method in class."""
        source = """
class Client:
    async def fetch(self, url):
        pass
"""
        tree, source_bytes = parse_python(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.py")

        assert len(functions) == 1
        assert functions[0].is_async is True
        assert functions[0].parent_class == "Client"

    def test_extract_generator_function(self, extractor, parse_python):
        """Test extraction of generator function."""
        source = """
def generate_numbers(n):
    for i in range(n):
        yield i
"""
        tree, source_bytes = parse_python(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.py")

        assert len(functions) == 1
        assert functions[0].is_generator is True

    def test_extract_generator_with_yield_from(self, extractor, parse_python):
        """Test extraction of generator with yield from."""
        source = """
def chain(*iterables):
    for it in iterables:
        yield from it
"""
        tree, source_bytes = parse_python(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.py")

        assert len(functions) == 1
        assert functions[0].is_generator is True

    def test_non_generator_function(self, extractor, parse_python):
        """Test that regular function is not marked as generator."""
        source = """
def regular():
    return 42
"""
        tree, source_bytes = parse_python(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.py")

        assert len(functions) == 1
        assert functions[0].is_generator is False


# =============================================================================
# Test Method Extraction
# =============================================================================


class TestExtractMethods:
    """Tests for method extraction from classes."""

    def test_method_has_parent_class(self, extractor, parse_python):
        """Test that method has parent_class set."""
        source = """
class Calculator:
    def add(self, a, b):
        return a + b
"""
        tree, source_bytes = parse_python(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.py")

        assert len(functions) == 1
        assert functions[0].parent_class == "Calculator"

    def test_method_has_correct_type(self, extractor, parse_python):
        """Test that method has CodeElementType.METHOD."""
        source = """
class MyClass:
    def my_method(self):
        pass
"""
        tree, source_bytes = parse_python(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.py")

        assert len(functions) == 1
        assert functions[0].element_type == CodeElementType.METHOD

    def test_method_qualified_name(self, extractor, parse_python):
        """Test method qualified_name includes class."""
        source = """
class MyClass:
    def my_method(self):
        pass
"""
        tree, source_bytes = parse_python(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.py")

        assert len(functions) == 1
        assert functions[0].qualified_name == "MyClass.my_method"

    def test_private_method(self, extractor, parse_python):
        """Test private method detection."""
        source = """
class MyClass:
    def _private_method(self):
        pass
"""
        tree, source_bytes = parse_python(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.py")

        assert len(functions) == 1
        assert functions[0].is_private is True

    def test_dunder_method_not_private(self, extractor, parse_python):
        """Test that dunder methods are not marked as private."""
        source = """
class MyClass:
    def __init__(self):
        pass
"""
        tree, source_bytes = parse_python(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.py")

        assert len(functions) == 1
        # __init__ should NOT be marked as private
        assert functions[0].is_private is False


# =============================================================================
# Test Class Extraction
# =============================================================================


class TestExtractClasses:
    """Tests for class extraction."""

    def test_extract_simple_class(self, extractor, parse_python):
        """Test extraction of simple class."""
        source = """
class MyClass:
    pass
"""
        tree, source_bytes = parse_python(source)
        classes = extractor.extract_classes(tree, source_bytes, "/test.py")

        assert len(classes) == 1
        assert classes[0].name == "MyClass"
        assert classes[0].element_type == CodeElementType.CLASS

    def test_extract_class_with_docstring(self, extractor, parse_python):
        """Test extraction of class with docstring."""
        source = '''
class Calculator:
    """A simple calculator class."""
    pass
'''
        tree, source_bytes = parse_python(source)
        classes = extractor.extract_classes(tree, source_bytes, "/test.py")

        assert len(classes) == 1
        assert classes[0].docstring == "A simple calculator class."

    def test_extract_class_with_bases(self, extractor, parse_python):
        """Test extraction of class with base classes."""
        source = """
class Child(Parent):
    pass
"""
        tree, source_bytes = parse_python(source)
        classes = extractor.extract_classes(tree, source_bytes, "/test.py")

        assert len(classes) == 1
        assert "Parent" in classes[0].bases

    def test_extract_class_with_multiple_bases(self, extractor, parse_python):
        """Test extraction of class with multiple inheritance."""
        source = """
class Child(Parent1, Parent2, Parent3):
    pass
"""
        tree, source_bytes = parse_python(source)
        classes = extractor.extract_classes(tree, source_bytes, "/test.py")

        assert len(classes) == 1
        bases = classes[0].bases
        assert "Parent1" in bases
        assert "Parent2" in bases
        assert "Parent3" in bases

    def test_extract_class_methods_list(self, extractor, parse_python):
        """Test that class has methods list."""
        source = """
class Calculator:
    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b
"""
        tree, source_bytes = parse_python(source)
        classes = extractor.extract_classes(tree, source_bytes, "/test.py")

        assert len(classes) == 1
        methods = classes[0].methods
        assert "add" in methods
        assert "subtract" in methods

    def test_decorated_class(self, extractor, parse_python):
        """Test extraction of decorated class."""
        source = """
@dataclass
class Point:
    x: int
    y: int
"""
        tree, source_bytes = parse_python(source)
        classes = extractor.extract_classes(tree, source_bytes, "/test.py")

        assert len(classes) == 1
        assert "dataclass" in classes[0].decorators

    def test_nested_class(self, extractor, parse_python):
        """Test extraction of nested class."""
        source = """
class Outer:
    class Inner:
        pass
"""
        tree, source_bytes = parse_python(source)
        classes = extractor.extract_classes(tree, source_bytes, "/test.py")

        assert len(classes) == 2
        names = [c.name for c in classes]
        assert "Outer" in names
        assert "Inner" in names

    def test_private_class(self, extractor, parse_python):
        """Test private class detection."""
        source = """
class _PrivateClass:
    pass
"""
        tree, source_bytes = parse_python(source)
        classes = extractor.extract_classes(tree, source_bytes, "/test.py")

        assert len(classes) == 1
        assert classes[0].is_private is True

    def test_abstract_class_with_abc(self, extractor, parse_python):
        """Test abstract class detection with ABC base."""
        source = """
from abc import ABC

class AbstractBase(ABC):
    pass
"""
        tree, source_bytes = parse_python(source)
        classes = extractor.extract_classes(tree, source_bytes, "/test.py")

        assert len(classes) == 1
        assert classes[0].is_abstract is True


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_empty_source_code(self, extractor, parse_python):
        """Test extraction from empty source."""
        source = ""
        tree, source_bytes = parse_python(source)

        functions = extractor.extract_functions(tree, source_bytes, "/test.py")
        classes = extractor.extract_classes(tree, source_bytes, "/test.py")

        assert len(functions) == 0
        assert len(classes) == 0

    def test_only_comments(self, extractor, parse_python):
        """Test extraction from source with only comments."""
        source = """
# This is a comment
# Another comment
"""
        tree, source_bytes = parse_python(source)

        functions = extractor.extract_functions(tree, source_bytes, "/test.py")
        classes = extractor.extract_classes(tree, source_bytes, "/test.py")

        assert len(functions) == 0
        assert len(classes) == 0

    def test_nested_function(self, extractor, parse_python):
        """Test extraction of nested function."""
        source = """
def outer():
    def inner():
        pass
    return inner
"""
        tree, source_bytes = parse_python(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.py")

        # Both outer and inner should be extracted
        assert len(functions) == 2
        names = [f.name for f in functions]
        assert "outer" in names
        assert "inner" in names

    def test_lambda_not_extracted(self, extractor, parse_python):
        """Test that lambda expressions are not extracted as functions."""
        source = """
add = lambda x, y: x + y
"""
        tree, source_bytes = parse_python(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.py")

        # Lambdas should not be extracted
        assert len(functions) == 0

    def test_unicode_in_names(self, extractor, parse_python):
        """Test extraction with unicode in identifiers."""
        source = '''
def привет():
    """Greet in Russian."""
    pass
'''
        tree, source_bytes = parse_python(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.py")

        assert len(functions) == 1
        assert functions[0].name == "привет"

    def test_line_count(self, extractor, parse_python):
        """Test line_count is calculated correctly."""
        source = """def multi_line():
    line1 = 1
    line2 = 2
    line3 = 3
    return line1 + line2 + line3
"""
        tree, source_bytes = parse_python(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.py")

        assert len(functions) == 1
        assert functions[0].line_count == 5

    def test_extract_all_combined(self, extractor, parse_python):
        """Test extract_all returns both functions and classes."""
        source = """
def standalone():
    pass

class MyClass:
    def method(self):
        pass
"""
        tree, source_bytes = parse_python(source)
        all_elements = extractor.extract_all(tree, source_bytes, "/test.py")

        # standalone function + MyClass + method
        assert len(all_elements) >= 2
        names = [e.name for e in all_elements]
        assert "standalone" in names
        assert "MyClass" in names


# =============================================================================
# Test Integration with Registry
# =============================================================================


class TestRegistryIntegration:
    """Tests for PythonExtractor integration with registry."""

    def test_extractor_registered(self):
        """Test that PythonExtractor can be registered in the registry."""
        from zapomni_core.treesitter.parser.registry import LanguageParserRegistry

        # Register extractor (auto-registration happens at import, but
        # reset_registry fixture clears it for test isolation)
        registry = LanguageParserRegistry()
        registry.register_extractor("python", PythonExtractor())

        extractor = registry.get_extractor("python")

        assert extractor is not None
        assert isinstance(extractor, PythonExtractor)
        assert extractor.language_name == "python"

    def test_language_in_config(self):
        """Test that 'python' is in LANGUAGES_WITH_EXTRACTORS."""
        from zapomni_core.treesitter.config import LANGUAGES_WITH_EXTRACTORS

        assert "python" in LANGUAGES_WITH_EXTRACTORS
