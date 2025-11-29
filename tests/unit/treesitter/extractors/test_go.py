"""Tests for GoExtractor.

This module contains comprehensive tests for the Go-specific code extractor,
covering functions, methods, structs, interfaces, doc comments, generics, and more.

Test count target: 50+ tests as per Issue #22 requirements.
"""

import pytest
from tree_sitter import Tree
from tree_sitter_language_pack import get_parser

from zapomni_core.treesitter.extractors.go import GoExtractor
from zapomni_core.treesitter.models import CodeElementType, ParameterInfo

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def extractor():
    """Create a GoExtractor instance."""
    return GoExtractor()


@pytest.fixture
def parse_go():
    """Factory fixture to parse Go source code."""
    parser = get_parser("go")

    def _parse(source: str) -> tuple[Tree, bytes]:
        source_bytes = source.encode("utf-8")
        tree = parser.parse(source_bytes)
        return tree, source_bytes

    return _parse


# =============================================================================
# Test GoExtractor Properties
# =============================================================================


class TestGoExtractorProperties:
    """Tests for GoExtractor basic properties."""

    def test_language_name(self, extractor):
        """Test that language name is 'go'."""
        assert extractor.language_name == "go"

    def test_supported_node_types(self, extractor):
        """Test supported node types include function and type declarations."""
        node_types = extractor.supported_node_types
        assert "function_declaration" in node_types
        assert "method_declaration" in node_types
        assert "type_declaration" in node_types


# =============================================================================
# Test Function Extraction - Basic
# =============================================================================


class TestExtractFunctionsBasic:
    """Tests for basic function extraction."""

    def test_extract_simple_function(self, extractor, parse_go):
        """Test extraction of a simple function."""
        source = """
package main

func hello() {
}
"""
        tree, source_bytes = parse_go(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.go")

        assert len(functions) == 1
        assert functions[0].name == "hello"
        assert functions[0].element_type == CodeElementType.FUNCTION

    def test_extract_multiple_functions(self, extractor, parse_go):
        """Test extraction of multiple functions."""
        source = """
package main

func func1() {}

func func2() {}

func func3() {}
"""
        tree, source_bytes = parse_go(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.go")

        assert len(functions) == 3
        names = [f.name for f in functions]
        assert "func1" in names
        assert "func2" in names
        assert "func3" in names

    def test_function_has_location(self, extractor, parse_go):
        """Test that extracted function has location info."""
        source = """package main

func hello() {}
"""
        tree, source_bytes = parse_go(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.go")

        assert len(functions) == 1
        func = functions[0]
        assert func.location is not None
        assert func.location.start_line == 2

    def test_function_has_source_code(self, extractor, parse_go):
        """Test that extracted function has source code."""
        source = """package main

func hello() {
    fmt.Println("Hello")
}
"""
        tree, source_bytes = parse_go(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.go")

        assert len(functions) == 1
        assert "func hello()" in functions[0].source_code
        assert 'fmt.Println("Hello")' in functions[0].source_code


# =============================================================================
# Test Function Extraction - Parameters
# =============================================================================


class TestExtractFunctionsParameters:
    """Tests for parameter extraction."""

    def test_extract_function_with_params(self, extractor, parse_go):
        """Test extraction of function with parameters."""
        source = """
package main

func add(a int, b int) int {
    return a + b
}
"""
        tree, source_bytes = parse_go(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.go")

        assert len(functions) == 1
        params = functions[0].parameters
        assert len(params) == 2
        assert params[0].name == "a"
        assert params[0].type_annotation == "int"
        assert params[1].name == "b"
        assert params[1].type_annotation == "int"

    def test_extract_function_with_grouped_params(self, extractor, parse_go):
        """Test extraction of function with grouped parameters (a, b int)."""
        source = """
package main

func add(a, b int) int {
    return a + b
}
"""
        tree, source_bytes = parse_go(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.go")

        assert len(functions) == 1
        params = functions[0].parameters
        assert len(params) == 2
        assert params[0].name == "a"
        assert params[1].name == "b"

    def test_extract_function_with_variadic_params(self, extractor, parse_go):
        """Test extraction of function with variadic parameters."""
        source = """
package main

func sum(nums ...int) int {
    return 0
}
"""
        tree, source_bytes = parse_go(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.go")

        assert len(functions) == 1
        params = functions[0].parameters
        assert len(params) == 1
        assert params[0].name == "nums"
        assert "...int" in params[0].type_annotation

    def test_extract_function_no_params(self, extractor, parse_go):
        """Test extraction of function with no parameters."""
        source = """
package main

func noParams() {}
"""
        tree, source_bytes = parse_go(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.go")

        assert len(functions) == 1
        assert len(functions[0].parameters) == 0


# =============================================================================
# Test Function Extraction - Return Types
# =============================================================================


class TestExtractFunctionsReturnTypes:
    """Tests for return type extraction."""

    def test_extract_function_with_return_type(self, extractor, parse_go):
        """Test extraction of function with single return type."""
        source = """
package main

func add(a, b int) int {
    return a + b
}
"""
        tree, source_bytes = parse_go(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.go")

        assert len(functions) == 1
        assert functions[0].return_type == "int"

    def test_function_without_return_type(self, extractor, parse_go):
        """Test function without return type has None."""
        source = """
package main

func hello() {
    fmt.Println("Hello")
}
"""
        tree, source_bytes = parse_go(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.go")

        assert len(functions) == 1
        assert functions[0].return_type is None

    def test_function_with_multiple_returns(self, extractor, parse_go):
        """Test function with multiple return values."""
        source = """
package main

func divide(a, b int) (int, error) {
    return a / b, nil
}
"""
        tree, source_bytes = parse_go(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.go")

        assert len(functions) == 1
        return_type = functions[0].return_type
        assert return_type is not None
        assert "int" in return_type
        assert "error" in return_type

    def test_function_with_named_returns(self, extractor, parse_go):
        """Test function with named return values."""
        source = """
package main

func divide(a, b int) (result int, err error) {
    result = a / b
    return
}
"""
        tree, source_bytes = parse_go(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.go")

        assert len(functions) == 1
        return_type = functions[0].return_type
        assert return_type is not None
        assert "result" in return_type
        assert "err" in return_type


# =============================================================================
# Test Function Extraction - Doc Comments
# =============================================================================


class TestExtractFunctionsDocComments:
    """Tests for doc comment extraction."""

    def test_extract_function_with_doc_comment(self, extractor, parse_go):
        """Test extraction of function with doc comment."""
        source = """
package main

// Add adds two integers and returns the sum.
func Add(a, b int) int {
    return a + b
}
"""
        tree, source_bytes = parse_go(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.go")

        assert len(functions) == 1
        assert functions[0].docstring is not None
        assert "Add adds two integers" in functions[0].docstring

    def test_extract_function_with_multiline_doc(self, extractor, parse_go):
        """Test extraction of function with multiline doc comment."""
        source = """
package main

// Divide divides a by b.
// It returns the quotient and an error if b is zero.
func Divide(a, b int) (int, error) {
    return a / b, nil
}
"""
        tree, source_bytes = parse_go(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.go")

        assert len(functions) == 1
        docstring = functions[0].docstring
        assert docstring is not None
        assert "Divide divides a by b" in docstring
        assert "quotient" in docstring

    def test_function_without_doc_comment(self, extractor, parse_go):
        """Test function without doc comment has None."""
        source = """
package main

func noDoc() {}
"""
        tree, source_bytes = parse_go(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.go")

        assert len(functions) == 1
        assert functions[0].docstring is None


# =============================================================================
# Test Method Extraction
# =============================================================================


class TestExtractMethods:
    """Tests for method extraction with receivers."""

    def test_extract_method_with_value_receiver(self, extractor, parse_go):
        """Test extraction of method with value receiver."""
        source = """
package main

type Calculator struct{}

func (c Calculator) Add(a, b int) int {
    return a + b
}
"""
        tree, source_bytes = parse_go(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.go")

        methods = [f for f in functions if f.element_type == CodeElementType.METHOD]
        assert len(methods) == 1
        assert methods[0].name == "Add"
        assert methods[0].parent_class == "Calculator"
        assert "pointer_receiver" not in methods[0].decorators

    def test_extract_method_with_pointer_receiver(self, extractor, parse_go):
        """Test extraction of method with pointer receiver."""
        source = """
package main

type Calculator struct {
    value int
}

func (c *Calculator) SetValue(v int) {
    c.value = v
}
"""
        tree, source_bytes = parse_go(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.go")

        methods = [f for f in functions if f.element_type == CodeElementType.METHOD]
        assert len(methods) == 1
        assert methods[0].name == "SetValue"
        assert methods[0].parent_class == "Calculator"
        assert "pointer_receiver" in methods[0].decorators

    def test_method_qualified_name(self, extractor, parse_go):
        """Test method qualified_name includes receiver type."""
        source = """
package main

type MyType struct{}

func (m MyType) Process() {}
"""
        tree, source_bytes = parse_go(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.go")

        methods = [f for f in functions if f.element_type == CodeElementType.METHOD]
        assert len(methods) == 1
        assert methods[0].qualified_name == "MyType.Process"

    def test_method_has_correct_type(self, extractor, parse_go):
        """Test that method has CodeElementType.METHOD."""
        source = """
package main

type MyType struct{}

func (m MyType) Method() {}
"""
        tree, source_bytes = parse_go(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.go")

        methods = [f for f in functions if f.element_type == CodeElementType.METHOD]
        assert len(methods) == 1
        assert methods[0].element_type == CodeElementType.METHOD


# =============================================================================
# Test Struct Extraction
# =============================================================================


class TestExtractStructs:
    """Tests for struct extraction."""

    def test_extract_simple_struct(self, extractor, parse_go):
        """Test extraction of simple struct."""
        source = """
package main

type Point struct {
    X int
    Y int
}
"""
        tree, source_bytes = parse_go(source)
        structs = extractor.extract_structs(tree, source_bytes, "/test.go")

        assert len(structs) == 1
        assert structs[0].name == "Point"
        assert structs[0].element_type == CodeElementType.STRUCT

    def test_struct_with_fields(self, extractor, parse_go):
        """Test struct fields are extracted."""
        source = """
package main

type Person struct {
    Name string
    Age  int
}
"""
        tree, source_bytes = parse_go(source)
        structs = extractor.extract_structs(tree, source_bytes, "/test.go")

        assert len(structs) == 1
        fields = structs[0].methods  # Fields stored in methods
        assert "Name" in fields
        assert "Age" in fields

    def test_struct_with_embedded_type(self, extractor, parse_go):
        """Test struct with embedded type."""
        source = """
package main

type Base struct {
    ID int
}

type Extended struct {
    Base
    Name string
}
"""
        tree, source_bytes = parse_go(source)
        structs = extractor.extract_structs(tree, source_bytes, "/test.go")

        assert len(structs) == 2
        extended = [s for s in structs if s.name == "Extended"][0]
        assert "Base" in extended.bases

    def test_struct_with_doc_comment(self, extractor, parse_go):
        """Test struct with doc comment."""
        source = """
package main

// Config represents application configuration.
type Config struct {
    Port int
}
"""
        tree, source_bytes = parse_go(source)
        structs = extractor.extract_structs(tree, source_bytes, "/test.go")

        assert len(structs) == 1
        assert structs[0].docstring is not None
        assert "Config represents" in structs[0].docstring

    def test_empty_struct(self, extractor, parse_go):
        """Test extraction of empty struct."""
        source = """
package main

type Empty struct{}
"""
        tree, source_bytes = parse_go(source)
        structs = extractor.extract_structs(tree, source_bytes, "/test.go")

        assert len(structs) == 1
        assert structs[0].name == "Empty"
        assert len(structs[0].methods) == 0


# =============================================================================
# Test Interface Extraction
# =============================================================================


class TestExtractInterfaces:
    """Tests for interface extraction."""

    def test_extract_simple_interface(self, extractor, parse_go):
        """Test extraction of simple interface."""
        source = """
package main

type Reader interface {
    Read(p []byte) (n int, err error)
}
"""
        tree, source_bytes = parse_go(source)
        interfaces = extractor.extract_interfaces(tree, source_bytes, "/test.go")

        assert len(interfaces) == 1
        assert interfaces[0].name == "Reader"
        assert interfaces[0].element_type == CodeElementType.INTERFACE

    def test_interface_with_methods(self, extractor, parse_go):
        """Test interface method extraction."""
        source = """
package main

type ReadWriter interface {
    Read(p []byte) (n int, err error)
    Write(p []byte) (n int, err error)
}
"""
        tree, source_bytes = parse_go(source)
        interfaces = extractor.extract_interfaces(tree, source_bytes, "/test.go")

        assert len(interfaces) == 1
        methods = interfaces[0].methods
        assert "Read" in methods
        assert "Write" in methods

    def test_interface_with_embedded_interface(self, extractor, parse_go):
        """Test interface with embedded interface."""
        source = """
package main

type Reader interface {
    Read(p []byte) (n int, err error)
}

type ReadCloser interface {
    Reader
    Close() error
}
"""
        tree, source_bytes = parse_go(source)
        interfaces = extractor.extract_interfaces(tree, source_bytes, "/test.go")

        assert len(interfaces) == 2
        read_closer = [i for i in interfaces if i.name == "ReadCloser"][0]
        assert "Reader" in read_closer.bases

    def test_interface_with_doc_comment(self, extractor, parse_go):
        """Test interface with doc comment."""
        source = """
package main

// Stringer is the interface implemented by types that can convert themselves to a string.
type Stringer interface {
    String() string
}
"""
        tree, source_bytes = parse_go(source)
        interfaces = extractor.extract_interfaces(tree, source_bytes, "/test.go")

        assert len(interfaces) == 1
        assert interfaces[0].docstring is not None
        assert "Stringer is the interface" in interfaces[0].docstring

    def test_empty_interface(self, extractor, parse_go):
        """Test extraction of empty interface."""
        source = """
package main

type Any interface{}
"""
        tree, source_bytes = parse_go(source)
        interfaces = extractor.extract_interfaces(tree, source_bytes, "/test.go")

        assert len(interfaces) == 1
        assert interfaces[0].name == "Any"
        assert len(interfaces[0].methods) == 0

    def test_interface_is_abstract(self, extractor, parse_go):
        """Test that interfaces are marked as abstract."""
        source = """
package main

type Service interface {
    Process()
}
"""
        tree, source_bytes = parse_go(source)
        interfaces = extractor.extract_interfaces(tree, source_bytes, "/test.go")

        assert len(interfaces) == 1
        assert interfaces[0].is_abstract is True


# =============================================================================
# Test Private Detection
# =============================================================================


class TestPrivateDetection:
    """Tests for private (unexported) detection."""

    def test_lowercase_function_is_private(self, extractor, parse_go):
        """Test that lowercase function is marked as private."""
        source = """
package main

func privateFunc() {}
"""
        tree, source_bytes = parse_go(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.go")

        assert len(functions) == 1
        assert functions[0].is_private is True

    def test_uppercase_function_is_public(self, extractor, parse_go):
        """Test that uppercase function is not marked as private."""
        source = """
package main

func PublicFunc() {}
"""
        tree, source_bytes = parse_go(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.go")

        assert len(functions) == 1
        assert functions[0].is_private is False

    def test_lowercase_struct_is_private(self, extractor, parse_go):
        """Test that lowercase struct is marked as private."""
        source = """
package main

type privateStruct struct{}
"""
        tree, source_bytes = parse_go(source)
        structs = extractor.extract_structs(tree, source_bytes, "/test.go")

        assert len(structs) == 1
        assert structs[0].is_private is True

    def test_uppercase_struct_is_public(self, extractor, parse_go):
        """Test that uppercase struct is not marked as private."""
        source = """
package main

type PublicStruct struct{}
"""
        tree, source_bytes = parse_go(source)
        structs = extractor.extract_structs(tree, source_bytes, "/test.go")

        assert len(structs) == 1
        assert structs[0].is_private is False

    def test_lowercase_interface_is_private(self, extractor, parse_go):
        """Test that lowercase interface is marked as private."""
        source = """
package main

type privateInterface interface{}
"""
        tree, source_bytes = parse_go(source)
        interfaces = extractor.extract_interfaces(tree, source_bytes, "/test.go")

        assert len(interfaces) == 1
        assert interfaces[0].is_private is True

    def test_uppercase_method_is_public(self, extractor, parse_go):
        """Test that uppercase method is not marked as private."""
        source = """
package main

type T struct{}

func (t T) PublicMethod() {}
"""
        tree, source_bytes = parse_go(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.go")

        methods = [f for f in functions if f.element_type == CodeElementType.METHOD]
        assert len(methods) == 1
        assert methods[0].is_private is False


# =============================================================================
# Test Generics (Go 1.18+)
# =============================================================================


class TestGenerics:
    """Tests for generics (type parameters) extraction."""

    def test_generic_function(self, extractor, parse_go):
        """Test extraction of generic function."""
        source = """
package main

func Map[T, U any](slice []T, fn func(T) U) []U {
    return nil
}
"""
        tree, source_bytes = parse_go(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.go")

        assert len(functions) == 1
        # Type params should be in decorators
        decorators = functions[0].decorators
        type_param_decorator = [d for d in decorators if d.startswith("type_params:")]
        assert len(type_param_decorator) == 1
        assert "T" in type_param_decorator[0]
        assert "U" in type_param_decorator[0]

    def test_generic_struct(self, extractor, parse_go):
        """Test extraction of generic struct."""
        source = """
package main

type Box[T any] struct {
    value T
}
"""
        tree, source_bytes = parse_go(source)
        structs = extractor.extract_structs(tree, source_bytes, "/test.go")

        assert len(structs) == 1
        assert structs[0].name == "Box"
        decorators = structs[0].decorators
        type_param_decorator = [d for d in decorators if d.startswith("type_params:")]
        assert len(type_param_decorator) == 1
        assert "T" in type_param_decorator[0]

    def test_generic_interface(self, extractor, parse_go):
        """Test extraction of generic interface."""
        source = """
package main

type Container[T any] interface {
    Get() T
    Set(T)
}
"""
        tree, source_bytes = parse_go(source)
        interfaces = extractor.extract_interfaces(tree, source_bytes, "/test.go")

        assert len(interfaces) == 1
        assert interfaces[0].name == "Container"
        decorators = interfaces[0].decorators
        type_param_decorator = [d for d in decorators if d.startswith("type_params:")]
        assert len(type_param_decorator) == 1


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_empty_source_code(self, extractor, parse_go):
        """Test extraction from empty source."""
        source = ""
        tree, source_bytes = parse_go(source)

        functions = extractor.extract_functions(tree, source_bytes, "/test.go")
        structs = extractor.extract_structs(tree, source_bytes, "/test.go")
        interfaces = extractor.extract_interfaces(tree, source_bytes, "/test.go")

        assert len(functions) == 0
        assert len(structs) == 0
        assert len(interfaces) == 0

    def test_only_package_declaration(self, extractor, parse_go):
        """Test extraction from source with only package declaration."""
        source = """
package main
"""
        tree, source_bytes = parse_go(source)

        functions = extractor.extract_functions(tree, source_bytes, "/test.go")
        structs = extractor.extract_structs(tree, source_bytes, "/test.go")
        interfaces = extractor.extract_interfaces(tree, source_bytes, "/test.go")

        assert len(functions) == 0
        assert len(structs) == 0
        assert len(interfaces) == 0

    def test_line_count(self, extractor, parse_go):
        """Test line_count is calculated correctly."""
        source = """package main

func multiLine() {
    line1 := 1
    line2 := 2
    line3 := 3
}
"""
        tree, source_bytes = parse_go(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.go")

        assert len(functions) == 1
        assert functions[0].line_count == 5

    def test_extract_all_combined(self, extractor, parse_go):
        """Test extract_all returns functions, structs, and interfaces."""
        source = """
package main

func standalone() {}

type MyStruct struct {
    Field int
}

type MyInterface interface {
    Method()
}
"""
        tree, source_bytes = parse_go(source)
        all_elements = extractor.extract_all(tree, source_bytes, "/test.go")

        assert len(all_elements) == 3
        names = [e.name for e in all_elements]
        assert "standalone" in names
        assert "MyStruct" in names
        assert "MyInterface" in names

    def test_extract_classes_returns_structs_and_interfaces(self, extractor, parse_go):
        """Test extract_classes returns both structs and interfaces."""
        source = """
package main

type MyStruct struct{}

type MyInterface interface{}
"""
        tree, source_bytes = parse_go(source)
        classes = extractor.extract_classes(tree, source_bytes, "/test.go")

        assert len(classes) == 2
        types = [c.element_type for c in classes]
        assert CodeElementType.STRUCT in types
        assert CodeElementType.INTERFACE in types

    def test_multiple_type_declarations(self, extractor, parse_go):
        """Test extraction from multiple type declarations."""
        source = """
package main

type (
    Point struct {
        X, Y int
    }

    Named interface {
        Name() string
    }
)
"""
        tree, source_bytes = parse_go(source)
        structs = extractor.extract_structs(tree, source_bytes, "/test.go")
        interfaces = extractor.extract_interfaces(tree, source_bytes, "/test.go")

        assert len(structs) == 1
        assert structs[0].name == "Point"
        assert len(interfaces) == 1
        assert interfaces[0].name == "Named"


# =============================================================================
# Test Complex Go Features
# =============================================================================


class TestComplexFeatures:
    """Tests for complex Go language features."""

    def test_function_with_slice_params(self, extractor, parse_go):
        """Test function with slice parameters."""
        source = """
package main

func process(data []byte) []byte {
    return data
}
"""
        tree, source_bytes = parse_go(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.go")

        assert len(functions) == 1
        params = functions[0].parameters
        assert len(params) == 1
        assert params[0].name == "data"
        assert "[]byte" in params[0].type_annotation

    def test_function_with_map_params(self, extractor, parse_go):
        """Test function with map parameters."""
        source = """
package main

func lookup(m map[string]int, key string) int {
    return m[key]
}
"""
        tree, source_bytes = parse_go(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.go")

        assert len(functions) == 1
        params = functions[0].parameters
        assert len(params) == 2

    def test_function_with_channel_params(self, extractor, parse_go):
        """Test function with channel parameters."""
        source = """
package main

func send(ch chan int, value int) {
    ch <- value
}
"""
        tree, source_bytes = parse_go(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.go")

        assert len(functions) == 1
        params = functions[0].parameters
        assert len(params) == 2
        assert params[0].name == "ch"

    def test_function_with_func_params(self, extractor, parse_go):
        """Test function with function type parameters."""
        source = """
package main

func apply(fn func(int) int, value int) int {
    return fn(value)
}
"""
        tree, source_bytes = parse_go(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.go")

        assert len(functions) == 1
        params = functions[0].parameters
        assert len(params) == 2

    def test_struct_with_pointer_field(self, extractor, parse_go):
        """Test struct with pointer field."""
        source = """
package main

type Node struct {
    Value int
    Next  *Node
}
"""
        tree, source_bytes = parse_go(source)
        structs = extractor.extract_structs(tree, source_bytes, "/test.go")

        assert len(structs) == 1
        fields = structs[0].methods
        assert "Value" in fields
        assert "Next" in fields

    def test_struct_with_embedded_pointer(self, extractor, parse_go):
        """Test struct with embedded pointer type."""
        source = """
package main

type Base struct {
    ID int
}

type Extended struct {
    *Base
    Name string
}
"""
        tree, source_bytes = parse_go(source)
        structs = extractor.extract_structs(tree, source_bytes, "/test.go")

        assert len(structs) == 2
        extended = [s for s in structs if s.name == "Extended"][0]
        # Embedded pointer type should be in bases
        assert any("Base" in b for b in extended.bases)


# =============================================================================
# Test Registry Integration
# =============================================================================


class TestRegistryIntegration:
    """Tests for GoExtractor integration with registry."""

    def test_extractor_registered(self):
        """Test that GoExtractor can be registered in the registry."""
        from zapomni_core.treesitter.parser.registry import LanguageParserRegistry

        registry = LanguageParserRegistry()
        registry.register_extractor("go", GoExtractor())

        extractor = registry.get_extractor("go")

        assert extractor is not None
        assert isinstance(extractor, GoExtractor)
        assert extractor.language_name == "go"

    def test_language_in_config(self):
        """Test that 'go' is in LANGUAGES_WITH_EXTRACTORS."""
        from zapomni_core.treesitter.config import LANGUAGES_WITH_EXTRACTORS

        assert "go" in LANGUAGES_WITH_EXTRACTORS
