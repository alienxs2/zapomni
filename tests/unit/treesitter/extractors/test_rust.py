"""Tests for RustExtractor.

This module contains comprehensive tests for the Rust-specific code extractor,
covering functions, methods, structs, traits, enums, doc comments, generics,
lifetimes, attributes, and more.

Test count target: 50+ tests as per Issue #23 requirements.
"""

import pytest
from tree_sitter import Tree
from tree_sitter_language_pack import get_parser

from zapomni_core.treesitter.extractors.rust import RustExtractor
from zapomni_core.treesitter.models import CodeElementType

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def extractor():
    """Create a RustExtractor instance."""
    return RustExtractor()


@pytest.fixture
def parse_rust():
    """Factory fixture to parse Rust source code."""
    parser = get_parser("rust")

    def _parse(source: str) -> tuple[Tree, bytes]:
        source_bytes = source.encode("utf-8")
        tree = parser.parse(source_bytes)
        return tree, source_bytes

    return _parse


# =============================================================================
# Test RustExtractor Properties
# =============================================================================


class TestRustExtractorProperties:
    """Tests for RustExtractor basic properties."""

    def test_language_name(self, extractor):
        """Test that language name is 'rust'."""
        assert extractor.language_name == "rust"

    def test_supported_node_types(self, extractor):
        """Test supported node types include function, impl, struct, trait, enum."""
        node_types = extractor.supported_node_types
        assert "function_item" in node_types
        assert "impl_item" in node_types
        assert "struct_item" in node_types
        assert "trait_item" in node_types
        assert "enum_item" in node_types


# =============================================================================
# Test Function Extraction - Basic
# =============================================================================


class TestExtractFunctionsBasic:
    """Tests for basic function extraction."""

    def test_extract_simple_function(self, extractor, parse_rust):
        """Test extraction of a simple function."""
        source = """
fn hello() {
}
"""
        tree, source_bytes = parse_rust(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.rs")

        assert len(functions) == 1
        assert functions[0].name == "hello"
        assert functions[0].element_type == CodeElementType.FUNCTION

    def test_extract_multiple_functions(self, extractor, parse_rust):
        """Test extraction of multiple functions."""
        source = """
fn func1() {}

fn func2() {}

fn func3() {}
"""
        tree, source_bytes = parse_rust(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.rs")

        assert len(functions) == 3
        names = [f.name for f in functions]
        assert "func1" in names
        assert "func2" in names
        assert "func3" in names

    def test_function_has_location(self, extractor, parse_rust):
        """Test that extracted function has location info."""
        source = """fn hello() {}
"""
        tree, source_bytes = parse_rust(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.rs")

        assert len(functions) == 1
        func = functions[0]
        assert func.location is not None
        assert func.location.start_line == 0

    def test_function_has_source_code(self, extractor, parse_rust):
        """Test that extracted function has source code."""
        source = """fn hello() {
    println!("Hello");
}
"""
        tree, source_bytes = parse_rust(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.rs")

        assert len(functions) == 1
        assert "fn hello()" in functions[0].source_code
        assert 'println!("Hello")' in functions[0].source_code


# =============================================================================
# Test Function Extraction - Parameters
# =============================================================================


class TestExtractFunctionsParameters:
    """Tests for parameter extraction."""

    def test_extract_function_with_params(self, extractor, parse_rust):
        """Test extraction of function with parameters."""
        source = """
fn add(a: i32, b: i32) -> i32 {
    a + b
}
"""
        tree, source_bytes = parse_rust(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.rs")

        assert len(functions) == 1
        params = functions[0].parameters
        assert len(params) == 2
        assert params[0].name == "a"
        assert params[0].type_annotation == "i32"
        assert params[1].name == "b"
        assert params[1].type_annotation == "i32"

    def test_extract_function_with_reference_params(self, extractor, parse_rust):
        """Test extraction of function with reference parameters."""
        source = """
fn process(data: &str) -> usize {
    data.len()
}
"""
        tree, source_bytes = parse_rust(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.rs")

        assert len(functions) == 1
        params = functions[0].parameters
        assert len(params) == 1
        assert params[0].name == "data"
        assert "&str" in params[0].type_annotation

    def test_extract_function_with_mutable_reference(self, extractor, parse_rust):
        """Test extraction of function with mutable reference parameter."""
        source = """
fn mutate(data: &mut Vec<i32>) {
    data.push(42);
}
"""
        tree, source_bytes = parse_rust(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.rs")

        assert len(functions) == 1
        params = functions[0].parameters
        assert len(params) == 1
        assert params[0].name == "data"
        assert "&mut" in params[0].type_annotation

    def test_extract_function_no_params(self, extractor, parse_rust):
        """Test extraction of function with no parameters."""
        source = """
fn empty() {}
"""
        tree, source_bytes = parse_rust(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.rs")

        assert len(functions) == 1
        assert functions[0].parameters == []


# =============================================================================
# Test Function Extraction - Return Types
# =============================================================================


class TestExtractFunctionsReturnTypes:
    """Tests for return type extraction."""

    def test_extract_function_with_simple_return(self, extractor, parse_rust):
        """Test extraction of function with simple return type."""
        source = """
fn answer() -> i32 {
    42
}
"""
        tree, source_bytes = parse_rust(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.rs")

        assert len(functions) == 1
        assert functions[0].return_type == "i32"

    def test_extract_function_with_result_return(self, extractor, parse_rust):
        """Test extraction of function with Result return type."""
        source = """
fn might_fail() -> Result<String, Error> {
    Ok(String::from("success"))
}
"""
        tree, source_bytes = parse_rust(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.rs")

        assert len(functions) == 1
        assert "Result" in functions[0].return_type

    def test_extract_function_with_option_return(self, extractor, parse_rust):
        """Test extraction of function with Option return type."""
        source = """
fn find_item() -> Option<i32> {
    Some(42)
}
"""
        tree, source_bytes = parse_rust(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.rs")

        assert len(functions) == 1
        assert "Option" in functions[0].return_type

    def test_extract_function_no_return(self, extractor, parse_rust):
        """Test extraction of function with no return type."""
        source = """
fn side_effect() {
    println!("done");
}
"""
        tree, source_bytes = parse_rust(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.rs")

        assert len(functions) == 1
        assert functions[0].return_type is None


# =============================================================================
# Test Function Extraction - Async Functions
# =============================================================================


class TestExtractFunctionsAsync:
    """Tests for async function extraction."""

    def test_extract_async_function(self, extractor, parse_rust):
        """Test extraction of async function."""
        source = """
async fn fetch_data() -> String {
    String::from("data")
}
"""
        tree, source_bytes = parse_rust(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.rs")

        assert len(functions) == 1
        assert functions[0].name == "fetch_data"
        assert functions[0].is_async is True

    def test_extract_non_async_function(self, extractor, parse_rust):
        """Test extraction of non-async function."""
        source = """
fn sync_func() -> i32 {
    42
}
"""
        tree, source_bytes = parse_rust(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.rs")

        assert len(functions) == 1
        assert functions[0].is_async is False


# =============================================================================
# Test Function Extraction - Doc Comments
# =============================================================================


class TestExtractFunctionsDocComments:
    """Tests for doc comment extraction."""

    def test_extract_function_with_doc_comment(self, extractor, parse_rust):
        """Test extraction of function with /// doc comment."""
        source = """
/// Adds two numbers together.
fn add(a: i32, b: i32) -> i32 {
    a + b
}
"""
        tree, source_bytes = parse_rust(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.rs")

        assert len(functions) == 1
        assert functions[0].docstring is not None
        assert "Adds two numbers together" in functions[0].docstring

    def test_extract_function_with_multiline_doc(self, extractor, parse_rust):
        """Test extraction of function with multiline doc comment."""
        source = """
/// Adds two numbers.
/// Returns the sum.
fn add(a: i32, b: i32) -> i32 {
    a + b
}
"""
        tree, source_bytes = parse_rust(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.rs")

        assert len(functions) == 1
        assert "Adds two numbers" in functions[0].docstring
        assert "Returns the sum" in functions[0].docstring

    def test_extract_function_no_doc_comment(self, extractor, parse_rust):
        """Test extraction of function without doc comment."""
        source = """
fn no_doc() {}
"""
        tree, source_bytes = parse_rust(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.rs")

        assert len(functions) == 1
        assert functions[0].docstring is None


# =============================================================================
# Test Function Extraction - Visibility
# =============================================================================


class TestExtractFunctionsVisibility:
    """Tests for visibility detection."""

    def test_public_function_is_not_private(self, extractor, parse_rust):
        """Test that pub function is not marked as private."""
        source = """
pub fn public_func() {}
"""
        tree, source_bytes = parse_rust(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.rs")

        assert len(functions) == 1
        assert functions[0].is_private is False

    def test_private_function_no_modifier(self, extractor, parse_rust):
        """Test that function without modifier is marked as private."""
        source = """
fn private_func() {}
"""
        tree, source_bytes = parse_rust(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.rs")

        assert len(functions) == 1
        assert functions[0].is_private is True

    def test_pub_crate_function_is_not_private(self, extractor, parse_rust):
        """Test that pub(crate) function is not marked as private."""
        source = """
pub(crate) fn crate_func() {}
"""
        tree, source_bytes = parse_rust(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.rs")

        assert len(functions) == 1
        assert functions[0].is_private is False


# =============================================================================
# Test Method Extraction - impl Blocks
# =============================================================================


class TestExtractMethods:
    """Tests for method extraction from impl blocks."""

    def test_extract_method_from_impl(self, extractor, parse_rust):
        """Test extraction of methods from impl block."""
        source = """
struct Point {
    x: i32,
    y: i32,
}

impl Point {
    fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}
"""
        tree, source_bytes = parse_rust(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.rs")

        assert len(functions) == 1
        assert functions[0].name == "new"
        assert functions[0].element_type == CodeElementType.METHOD
        assert functions[0].parent_class == "Point"

    def test_extract_method_with_self(self, extractor, parse_rust):
        """Test extraction of method with &self receiver."""
        source = """
struct Counter {
    value: i32,
}

impl Counter {
    fn get(&self) -> i32 {
        self.value
    }
}
"""
        tree, source_bytes = parse_rust(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.rs")

        assert len(functions) == 1
        assert functions[0].name == "get"
        assert any("receiver:&self" in d for d in functions[0].decorators)

    def test_extract_method_with_mut_self(self, extractor, parse_rust):
        """Test extraction of method with &mut self receiver."""
        source = """
struct Counter {
    value: i32,
}

impl Counter {
    fn increment(&mut self) {
        self.value += 1;
    }
}
"""
        tree, source_bytes = parse_rust(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.rs")

        assert len(functions) == 1
        assert functions[0].name == "increment"
        assert any("receiver:&mut self" in d for d in functions[0].decorators)

    def test_extract_static_method(self, extractor, parse_rust):
        """Test extraction of static method (no self)."""
        source = """
struct Math;

impl Math {
    fn add(a: i32, b: i32) -> i32 {
        a + b
    }
}
"""
        tree, source_bytes = parse_rust(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.rs")

        assert len(functions) == 1
        assert functions[0].name == "add"
        assert functions[0].is_static is True

    def test_extract_trait_impl_method(self, extractor, parse_rust):
        """Test extraction of trait impl method."""
        source = """
struct Circle {
    radius: f64,
}

impl Display for Circle {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "Circle({})", self.radius)
    }
}
"""
        tree, source_bytes = parse_rust(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.rs")

        assert len(functions) == 1
        assert functions[0].name == "fmt"
        assert functions[0].parent_class == "Circle"

    def test_extract_qualified_name_for_method(self, extractor, parse_rust):
        """Test that methods have qualified names."""
        source = """
struct Foo;

impl Foo {
    fn bar(&self) {}
}
"""
        tree, source_bytes = parse_rust(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.rs")

        assert len(functions) == 1
        assert "Foo.bar" in functions[0].qualified_name


# =============================================================================
# Test Struct Extraction
# =============================================================================


class TestExtractStructs:
    """Tests for struct extraction."""

    def test_extract_simple_struct(self, extractor, parse_rust):
        """Test extraction of simple struct."""
        source = """
struct Point {
    x: i32,
    y: i32,
}
"""
        tree, source_bytes = parse_rust(source)
        structs = extractor.extract_structs(tree, source_bytes, "/test.rs")

        assert len(structs) == 1
        assert structs[0].name == "Point"
        assert structs[0].element_type == CodeElementType.STRUCT

    def test_extract_struct_fields(self, extractor, parse_rust):
        """Test extraction of struct fields."""
        source = """
struct Rectangle {
    width: u32,
    height: u32,
}
"""
        tree, source_bytes = parse_rust(source)
        structs = extractor.extract_structs(tree, source_bytes, "/test.rs")

        assert len(structs) == 1
        fields = structs[0].methods  # Fields stored in methods
        assert "width" in fields
        assert "height" in fields

    def test_extract_tuple_struct(self, extractor, parse_rust):
        """Test extraction of tuple struct."""
        source = """
struct Color(u8, u8, u8);
"""
        tree, source_bytes = parse_rust(source)
        structs = extractor.extract_structs(tree, source_bytes, "/test.rs")

        assert len(structs) == 1
        assert structs[0].name == "Color"

    def test_extract_unit_struct(self, extractor, parse_rust):
        """Test extraction of unit struct."""
        source = """
struct Marker;
"""
        tree, source_bytes = parse_rust(source)
        structs = extractor.extract_structs(tree, source_bytes, "/test.rs")

        assert len(structs) == 1
        assert structs[0].name == "Marker"

    def test_extract_struct_with_derive(self, extractor, parse_rust):
        """Test extraction of struct with #[derive] attribute."""
        source = """
#[derive(Debug, Clone)]
struct Data {
    value: i32,
}
"""
        tree, source_bytes = parse_rust(source)
        structs = extractor.extract_structs(tree, source_bytes, "/test.rs")

        assert len(structs) == 1
        assert any("derive" in d for d in structs[0].decorators)

    def test_extract_struct_with_doc(self, extractor, parse_rust):
        """Test extraction of struct with doc comment."""
        source = """
/// A 2D point.
struct Point {
    x: f64,
    y: f64,
}
"""
        tree, source_bytes = parse_rust(source)
        structs = extractor.extract_structs(tree, source_bytes, "/test.rs")

        assert len(structs) == 1
        assert "2D point" in structs[0].docstring


# =============================================================================
# Test Trait Extraction
# =============================================================================


class TestExtractTraits:
    """Tests for trait extraction."""

    def test_extract_simple_trait(self, extractor, parse_rust):
        """Test extraction of simple trait."""
        source = """
trait Greet {
    fn greet(&self) -> String;
}
"""
        tree, source_bytes = parse_rust(source)
        traits = extractor.extract_traits(tree, source_bytes, "/test.rs")

        assert len(traits) == 1
        assert traits[0].name == "Greet"
        assert traits[0].element_type == CodeElementType.INTERFACE
        assert traits[0].is_abstract is True

    def test_extract_trait_methods(self, extractor, parse_rust):
        """Test extraction of trait method signatures."""
        source = """
trait Animal {
    fn name(&self) -> &str;
    fn speak(&self);
}
"""
        tree, source_bytes = parse_rust(source)
        traits = extractor.extract_traits(tree, source_bytes, "/test.rs")

        assert len(traits) == 1
        methods = traits[0].methods
        assert "name" in methods
        assert "speak" in methods

    def test_extract_trait_with_supertraits(self, extractor, parse_rust):
        """Test extraction of trait with supertraits."""
        source = """
trait DebugPrint: Debug + Display {
    fn debug_print(&self);
}
"""
        tree, source_bytes = parse_rust(source)
        traits = extractor.extract_traits(tree, source_bytes, "/test.rs")

        assert len(traits) == 1
        bases = traits[0].bases
        assert "Debug" in bases or "Display" in bases

    def test_extract_trait_with_doc(self, extractor, parse_rust):
        """Test extraction of trait with doc comment."""
        source = """
/// Represents something that can be drawn.
trait Drawable {
    fn draw(&self);
}
"""
        tree, source_bytes = parse_rust(source)
        traits = extractor.extract_traits(tree, source_bytes, "/test.rs")

        assert len(traits) == 1
        assert "can be drawn" in traits[0].docstring


# =============================================================================
# Test Enum Extraction
# =============================================================================


class TestExtractEnums:
    """Tests for enum extraction."""

    def test_extract_simple_enum(self, extractor, parse_rust):
        """Test extraction of simple enum."""
        source = """
enum Direction {
    North,
    South,
    East,
    West,
}
"""
        tree, source_bytes = parse_rust(source)
        enums = extractor.extract_enums(tree, source_bytes, "/test.rs")

        assert len(enums) == 1
        assert enums[0].name == "Direction"
        assert enums[0].element_type == CodeElementType.ENUM

    def test_extract_enum_variants(self, extractor, parse_rust):
        """Test extraction of enum variants."""
        source = """
enum Color {
    Red,
    Green,
    Blue,
}
"""
        tree, source_bytes = parse_rust(source)
        enums = extractor.extract_enums(tree, source_bytes, "/test.rs")

        assert len(enums) == 1
        variants = enums[0].methods  # Variants stored in methods
        assert "Red" in variants
        assert "Green" in variants
        assert "Blue" in variants

    def test_extract_enum_with_data(self, extractor, parse_rust):
        """Test extraction of enum with data variants."""
        source = """
enum Message {
    Quit,
    Move { x: i32, y: i32 },
    Write(String),
}
"""
        tree, source_bytes = parse_rust(source)
        enums = extractor.extract_enums(tree, source_bytes, "/test.rs")

        assert len(enums) == 1
        variants = enums[0].methods
        assert "Quit" in variants
        assert "Move" in variants
        assert "Write" in variants

    def test_extract_enum_with_derive(self, extractor, parse_rust):
        """Test extraction of enum with #[derive] attribute."""
        source = """
#[derive(Debug, PartialEq)]
enum Status {
    Active,
    Inactive,
}
"""
        tree, source_bytes = parse_rust(source)
        enums = extractor.extract_enums(tree, source_bytes, "/test.rs")

        assert len(enums) == 1
        assert any("derive" in d for d in enums[0].decorators)


# =============================================================================
# Test Generics and Lifetimes
# =============================================================================


class TestGenericsAndLifetimes:
    """Tests for generics and lifetime extraction."""

    def test_extract_generic_function(self, extractor, parse_rust):
        """Test extraction of generic function."""
        source = """
fn identity<T>(value: T) -> T {
    value
}
"""
        tree, source_bytes = parse_rust(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.rs")

        assert len(functions) == 1
        assert any("generics:" in d for d in functions[0].decorators)

    def test_extract_generic_with_bounds(self, extractor, parse_rust):
        """Test extraction of generic with trait bounds."""
        source = """
fn print_all<T: Display>(items: Vec<T>) {
    for item in items {
        println!("{}", item);
    }
}
"""
        tree, source_bytes = parse_rust(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.rs")

        assert len(functions) == 1
        assert any("generics:" in d and "T" in d for d in functions[0].decorators)

    def test_extract_generic_struct(self, extractor, parse_rust):
        """Test extraction of generic struct."""
        source = """
struct Wrapper<T> {
    value: T,
}
"""
        tree, source_bytes = parse_rust(source)
        structs = extractor.extract_structs(tree, source_bytes, "/test.rs")

        assert len(structs) == 1
        assert any("generics:" in d for d in structs[0].decorators)


# =============================================================================
# Test Attributes
# =============================================================================


class TestAttributes:
    """Tests for attribute extraction."""

    def test_extract_derive_attribute(self, extractor, parse_rust):
        """Test extraction of #[derive(...)] attribute."""
        source = """
#[derive(Debug, Clone, Copy)]
struct Point {
    x: i32,
    y: i32,
}
"""
        tree, source_bytes = parse_rust(source)
        structs = extractor.extract_structs(tree, source_bytes, "/test.rs")

        assert len(structs) == 1
        assert any("derive(Debug, Clone, Copy)" in d for d in structs[0].decorators)

    def test_extract_cfg_attribute(self, extractor, parse_rust):
        """Test extraction of #[cfg(...)] attribute."""
        source = """
#[cfg(test)]
fn test_only() {}
"""
        tree, source_bytes = parse_rust(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.rs")

        assert len(functions) == 1
        assert any("cfg(test)" in d for d in functions[0].decorators)

    def test_extract_multiple_attributes(self, extractor, parse_rust):
        """Test extraction of multiple attributes."""
        source = """
#[derive(Debug)]
#[allow(dead_code)]
struct Unused {
    data: i32,
}
"""
        tree, source_bytes = parse_rust(source)
        structs = extractor.extract_structs(tree, source_bytes, "/test.rs")

        assert len(structs) == 1
        decorators = structs[0].decorators
        assert any("derive" in d for d in decorators)
        assert any("allow" in d for d in decorators)


# =============================================================================
# Test extract_all and extract_classes
# =============================================================================


class TestExtractAllAndClasses:
    """Tests for extract_all and extract_classes methods."""

    def test_extract_all_combines_elements(self, extractor, parse_rust):
        """Test that extract_all returns all element types."""
        source = """
fn standalone() {}

struct Data {
    value: i32,
}

trait Processor {
    fn process(&self);
}

enum Status {
    Active,
    Inactive,
}

impl Data {
    fn new() -> Self {
        Self { value: 0 }
    }
}
"""
        tree, source_bytes = parse_rust(source)
        all_elements = extractor.extract_all(tree, source_bytes, "/test.rs")

        # Should have: 1 function + 1 method + 1 struct + 1 trait + 1 enum = 5
        assert len(all_elements) >= 5

        types = [e.element_type for e in all_elements]
        assert CodeElementType.FUNCTION in types
        assert CodeElementType.METHOD in types
        assert CodeElementType.STRUCT in types
        assert CodeElementType.INTERFACE in types
        assert CodeElementType.ENUM in types

    def test_extract_classes_returns_structs_traits_enums(self, extractor, parse_rust):
        """Test that extract_classes returns structs, traits, and enums."""
        source = """
struct Point { x: i32 }
trait Shape {}
enum Color { Red }
"""
        tree, source_bytes = parse_rust(source)
        classes = extractor.extract_classes(tree, source_bytes, "/test.rs")

        assert len(classes) == 3
        types = [c.element_type for c in classes]
        assert CodeElementType.STRUCT in types
        assert CodeElementType.INTERFACE in types
        assert CodeElementType.ENUM in types

    def test_extract_all_sorted_by_line(self, extractor, parse_rust):
        """Test that extract_all returns elements sorted by line number."""
        source = """
struct First {}

fn second() {}

enum Third { A }
"""
        tree, source_bytes = parse_rust(source)
        all_elements = extractor.extract_all(tree, source_bytes, "/test.rs")

        # Verify sorted by start_line
        lines = [e.location.start_line for e in all_elements]
        assert lines == sorted(lines)


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_empty_source(self, extractor, parse_rust):
        """Test extraction from empty source."""
        source = ""
        tree, source_bytes = parse_rust(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.rs")
        structs = extractor.extract_structs(tree, source_bytes, "/test.rs")

        assert functions == []
        assert structs == []

    def test_nested_modules(self, extractor, parse_rust):
        """Test extraction handles nested modules."""
        source = """
mod inner {
    fn nested_func() {}
}
"""
        tree, source_bytes = parse_rust(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.rs")

        # Should find the nested function
        assert len(functions) == 1
        assert functions[0].name == "nested_func"

    def test_impl_for_generic_type(self, extractor, parse_rust):
        """Test extraction of impl for generic type."""
        source = """
struct Container<T> {
    value: T,
}

impl<T> Container<T> {
    fn get(&self) -> &T {
        &self.value
    }
}
"""
        tree, source_bytes = parse_rust(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.rs")

        assert len(functions) == 1
        assert functions[0].name == "get"

    def test_multiple_impls_for_same_type(self, extractor, parse_rust):
        """Test extraction with multiple impl blocks for same type."""
        source = """
struct Foo;

impl Foo {
    fn method1(&self) {}
}

impl Foo {
    fn method2(&self) {}
}
"""
        tree, source_bytes = parse_rust(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.rs")

        assert len(functions) == 2
        names = [f.name for f in functions]
        assert "method1" in names
        assert "method2" in names
