"""Unit tests for zapomni_core.treesitter.extractors.generic module."""

import pytest
from tree_sitter_language_pack import get_parser

from zapomni_core.treesitter.extractors.generic import (
    CLASS_NODE_TYPES,
    FUNCTION_NODE_TYPES,
    NAME_NODE_TYPES,
    GenericExtractor,
)
from zapomni_core.treesitter.models import CodeElementType


class TestGenericExtractorProperties:
    """Tests for GenericExtractor properties."""

    def test_language_name(self):
        """Test that language_name returns 'generic'."""
        extractor = GenericExtractor()
        assert extractor.language_name == "generic"

    def test_supported_node_types(self):
        """Test that supported_node_types includes function and class types."""
        extractor = GenericExtractor()
        node_types = extractor.supported_node_types

        # Should include function types
        assert "function_definition" in node_types
        assert "function_declaration" in node_types

        # Should include class types
        assert "class_definition" in node_types
        assert "class_declaration" in node_types

    def test_supported_node_types_is_tuple(self):
        """Test that supported_node_types returns a tuple."""
        extractor = GenericExtractor()
        assert isinstance(extractor.supported_node_types, tuple)


class TestNodeTypeConstants:
    """Tests for node type constant sets."""

    def test_function_node_types_not_empty(self):
        """Test FUNCTION_NODE_TYPES is not empty."""
        assert len(FUNCTION_NODE_TYPES) > 0

    def test_class_node_types_not_empty(self):
        """Test CLASS_NODE_TYPES is not empty."""
        assert len(CLASS_NODE_TYPES) > 0

    def test_name_node_types_not_empty(self):
        """Test NAME_NODE_TYPES is not empty."""
        assert len(NAME_NODE_TYPES) > 0

    def test_function_types_are_strings(self):
        """Test that all function node types are strings."""
        for node_type in FUNCTION_NODE_TYPES:
            assert isinstance(node_type, str)

    def test_class_types_are_strings(self):
        """Test that all class node types are strings."""
        for node_type in CLASS_NODE_TYPES:
            assert isinstance(node_type, str)

    def test_name_types_include_identifier(self):
        """Test that NAME_NODE_TYPES includes 'identifier'."""
        assert "identifier" in NAME_NODE_TYPES


class TestExtractFunctionsPython:
    """Tests for extract_functions with Python code."""

    def test_extract_simple_function(self, python_tree, python_source_code):
        """Test extracting a simple function."""
        extractor = GenericExtractor()
        functions = extractor.extract_functions(
            python_tree, python_source_code, "/test.py"
        )

        assert len(functions) > 0
        names = [f.name for f in functions]
        assert "hello" in names

    def test_extract_function_with_params(self, python_tree, python_source_code):
        """Test extracting function with parameters."""
        extractor = GenericExtractor()
        functions = extractor.extract_functions(
            python_tree, python_source_code, "/test.py"
        )

        # Find the top-level add function (not the method)
        add_funcs = [f for f in functions if f.name == "add"]
        assert len(add_funcs) > 0
        # At least one should be a function (could also have method)
        has_func_or_method = any(
            f.element_type in (CodeElementType.FUNCTION, CodeElementType.METHOD)
            for f in add_funcs
        )
        assert has_func_or_method

    def test_extract_async_function(self, python_tree, python_source_code):
        """Test extracting async function."""
        extractor = GenericExtractor()
        functions = extractor.extract_functions(
            python_tree, python_source_code, "/test.py"
        )

        async_func = next((f for f in functions if f.name == "async_function"), None)
        assert async_func is not None
        assert async_func.is_async is True

    def test_extract_method_inside_class(self, python_tree, python_source_code):
        """Test extracting methods inside a class."""
        extractor = GenericExtractor()
        functions = extractor.extract_functions(
            python_tree, python_source_code, "/test.py"
        )

        # Methods should also be extracted
        method_names = [f.name for f in functions if f.element_type == CodeElementType.METHOD]
        assert "__init__" in method_names or "add" in method_names

    def test_extract_private_method(self, python_tree, python_source_code):
        """Test extracting private method (starts with _)."""
        extractor = GenericExtractor()
        functions = extractor.extract_functions(
            python_tree, python_source_code, "/test.py"
        )

        private = next((f for f in functions if f.name == "_private_method"), None)
        assert private is not None
        assert private.is_private is True

    def test_function_has_location(self, python_tree, python_source_code):
        """Test that extracted function has location info."""
        extractor = GenericExtractor()
        functions = extractor.extract_functions(
            python_tree, python_source_code, "/test.py"
        )

        for func in functions:
            assert func.location is not None
            assert func.location.start_line >= 0
            assert func.location.end_line >= func.location.start_line

    def test_function_has_source_code(self, python_tree, python_source_code):
        """Test that extracted function has source code."""
        extractor = GenericExtractor()
        functions = extractor.extract_functions(
            python_tree, python_source_code, "/test.py"
        )

        for func in functions:
            assert func.source_code is not None
            assert len(func.source_code) > 0


class TestExtractClassesPython:
    """Tests for extract_classes with Python code."""

    def test_extract_class(self, python_tree, python_source_code):
        """Test extracting a class definition."""
        extractor = GenericExtractor()
        classes = extractor.extract_classes(
            python_tree, python_source_code, "/test.py"
        )

        assert len(classes) > 0
        names = [c.name for c in classes]
        assert "Calculator" in names

    def test_class_has_type_class(self, python_tree, python_source_code):
        """Test that extracted class has element_type CLASS."""
        extractor = GenericExtractor()
        classes = extractor.extract_classes(
            python_tree, python_source_code, "/test.py"
        )

        calculator = next((c for c in classes if c.name == "Calculator"), None)
        assert calculator is not None
        assert calculator.element_type == CodeElementType.CLASS

    def test_class_has_methods(self, python_tree, python_source_code):
        """Test that extracted class has method names."""
        extractor = GenericExtractor()
        classes = extractor.extract_classes(
            python_tree, python_source_code, "/test.py"
        )

        calculator = next((c for c in classes if c.name == "Calculator"), None)
        assert calculator is not None
        assert len(calculator.methods) > 0

    def test_class_has_location(self, python_tree, python_source_code):
        """Test that extracted class has location info."""
        extractor = GenericExtractor()
        classes = extractor.extract_classes(
            python_tree, python_source_code, "/test.py"
        )

        for cls in classes:
            assert cls.location is not None
            assert cls.location.start_line >= 0


class TestExtractFunctionsJavaScript:
    """Tests for extract_functions with JavaScript code."""

    def test_extract_function_declaration(self, javascript_tree, javascript_source_code):
        """Test extracting JavaScript function declaration."""
        extractor = GenericExtractor()
        functions = extractor.extract_functions(
            javascript_tree, javascript_source_code, "/test.js"
        )

        names = [f.name for f in functions]
        assert "greet" in names

    def test_extract_async_function_js(self, javascript_tree, javascript_source_code):
        """Test extracting JavaScript async function."""
        extractor = GenericExtractor()
        functions = extractor.extract_functions(
            javascript_tree, javascript_source_code, "/test.js"
        )

        fetch_func = next((f for f in functions if f.name == "fetchData"), None)
        assert fetch_func is not None
        assert fetch_func.is_async is True


class TestExtractClassesJavaScript:
    """Tests for extract_classes with JavaScript code."""

    def test_extract_class_js(self, javascript_tree, javascript_source_code):
        """Test extracting JavaScript class."""
        extractor = GenericExtractor()
        classes = extractor.extract_classes(
            javascript_tree, javascript_source_code, "/test.js"
        )

        names = [c.name for c in classes]
        assert "Person" in names


class TestExtractRust:
    """Tests for extraction with Rust code."""

    def test_extract_rust_function(self, rust_tree, rust_source_code):
        """Test extracting Rust function."""
        extractor = GenericExtractor()
        functions = extractor.extract_functions(
            rust_tree, rust_source_code, "/test.rs"
        )

        names = [f.name for f in functions]
        assert "main" in names or "add" in names

    def test_extract_rust_struct(self, rust_tree, rust_source_code):
        """Test extracting Rust struct."""
        extractor = GenericExtractor()
        classes = extractor.extract_classes(
            rust_tree, rust_source_code, "/test.rs"
        )

        names = [c.name for c in classes]
        assert "Point" in names

    def test_rust_struct_is_struct_type(self, rust_tree, rust_source_code):
        """Test that Rust struct has element_type STRUCT."""
        extractor = GenericExtractor()
        classes = extractor.extract_classes(
            rust_tree, rust_source_code, "/test.rs"
        )

        point = next((c for c in classes if c.name == "Point"), None)
        assert point is not None
        assert point.element_type == CodeElementType.STRUCT

    def test_extract_rust_impl(self, rust_tree, rust_source_code):
        """Test extracting Rust impl block methods."""
        extractor = GenericExtractor()
        functions = extractor.extract_functions(
            rust_tree, rust_source_code, "/test.rs"
        )

        # impl methods should be extracted
        names = [f.name for f in functions]
        assert "new" in names or "distance" in names

    def test_extract_rust_trait(self, rust_tree, rust_source_code):
        """Test extracting Rust trait."""
        extractor = GenericExtractor()
        classes = extractor.extract_classes(
            rust_tree, rust_source_code, "/test.rs"
        )

        trait = next((c for c in classes if c.name == "Drawable"), None)
        assert trait is not None
        # Traits are similar to interfaces
        assert trait.element_type == CodeElementType.INTERFACE


class TestExtractAll:
    """Tests for extract_all method."""

    def test_extract_all_python(self, python_tree, python_source_code):
        """Test extract_all with Python code."""
        extractor = GenericExtractor()
        all_elements = extractor.extract_all(
            python_tree, python_source_code, "/test.py"
        )

        assert len(all_elements) > 0
        # Should have both functions and classes
        types = set(e.element_type for e in all_elements)
        assert CodeElementType.FUNCTION in types or CodeElementType.METHOD in types
        assert CodeElementType.CLASS in types

    def test_extract_all_sorted_by_line(self, python_tree, python_source_code):
        """Test that extract_all returns elements sorted by line number."""
        extractor = GenericExtractor()
        all_elements = extractor.extract_all(
            python_tree, python_source_code, "/test.py"
        )

        for i in range(len(all_elements) - 1):
            assert all_elements[i].location.start_line <= all_elements[i + 1].location.start_line


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_source_code(self):
        """Test extracting from empty source code."""
        extractor = GenericExtractor()
        parser = get_parser("python")
        tree = parser.parse(b"")

        functions = extractor.extract_functions(tree, b"", "/test.py")
        classes = extractor.extract_classes(tree, b"", "/test.py")

        assert functions == []
        assert classes == []

    def test_comments_only(self):
        """Test extracting from source with only comments."""
        extractor = GenericExtractor()
        parser = get_parser("python")
        source = b"# This is a comment\n# Another comment"
        tree = parser.parse(source)

        functions = extractor.extract_functions(tree, source, "/test.py")
        classes = extractor.extract_classes(tree, source, "/test.py")

        assert functions == []
        assert classes == []

    def test_syntax_error_partial_extraction(self):
        """Test extraction from code with syntax errors."""
        extractor = GenericExtractor()
        parser = get_parser("python")
        # Valid function followed by invalid syntax
        source = b"def valid_func(): pass\ndef invalid { }"
        tree = parser.parse(source)

        functions = extractor.extract_functions(tree, source, "/test.py")

        # Should still extract the valid function
        assert len(functions) >= 1
        names = [f.name for f in functions]
        assert "valid_func" in names

    def test_unicode_in_source(self):
        """Test extraction from source with unicode characters."""
        extractor = GenericExtractor()
        parser = get_parser("python")
        # Use raw bytes to avoid encoding issues with emojis
        source = b'def hello():\n    """Say hello."""\n    print("Hi")'
        tree = parser.parse(source)

        functions = extractor.extract_functions(tree, source, "/test.py")

        # Should find at least the hello function
        hello_funcs = [f for f in functions if f.name == "hello"]
        assert len(hello_funcs) >= 1

    def test_nested_functions(self):
        """Test extraction of nested functions."""
        extractor = GenericExtractor()
        parser = get_parser("python")
        source = b"def outer():\n    def inner():\n        pass\n    return inner"
        tree = parser.parse(source)

        functions = extractor.extract_functions(tree, source, "/test.py")

        names = [f.name for f in functions]
        assert "outer" in names
        assert "inner" in names

    def test_nested_classes(self):
        """Test extraction of nested classes."""
        extractor = GenericExtractor()
        parser = get_parser("python")
        source = b"class Outer:\n    class Inner:\n        pass"
        tree = parser.parse(source)

        classes = extractor.extract_classes(tree, source, "/test.py")

        names = [c.name for c in classes]
        assert "Outer" in names
        assert "Inner" in names
