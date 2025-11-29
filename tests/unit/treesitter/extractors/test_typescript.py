"""Tests for TypeScriptExtractor.

This module contains comprehensive tests for the TypeScript/JavaScript code extractor,
covering functions, classes, interfaces, type aliases, enums, JSDoc, decorators,
type annotations, async functions, generators, and more.

Test count target: 40+ tests as per Issue #20 requirements.
"""

import pytest
from tree_sitter import Tree
from tree_sitter_language_pack import get_parser

from zapomni_core.treesitter.extractors.typescript import TypeScriptExtractor
from zapomni_core.treesitter.models import CodeElementType

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def extractor():
    """Create a TypeScriptExtractor instance."""
    return TypeScriptExtractor()


@pytest.fixture
def parse_typescript():
    """Factory fixture to parse TypeScript source code."""
    parser = get_parser("typescript")

    def _parse(source: str) -> tuple[Tree, bytes]:
        source_bytes = source.encode("utf-8")
        tree = parser.parse(source_bytes)
        return tree, source_bytes

    return _parse


@pytest.fixture
def parse_javascript():
    """Factory fixture to parse JavaScript source code."""
    parser = get_parser("javascript")

    def _parse(source: str) -> tuple[Tree, bytes]:
        source_bytes = source.encode("utf-8")
        tree = parser.parse(source_bytes)
        return tree, source_bytes

    return _parse


# =============================================================================
# Test TypeScriptExtractor Properties
# =============================================================================


class TestTypeScriptExtractorProperties:
    """Tests for TypeScriptExtractor basic properties."""

    def test_language_name(self, extractor):
        """Test that language name is 'typescript'."""
        assert extractor.language_name == "typescript"

    def test_supported_node_types(self, extractor):
        """Test supported node types include all expected types."""
        node_types = extractor.supported_node_types
        # Functions
        assert "function_declaration" in node_types
        assert "generator_function_declaration" in node_types
        assert "arrow_function" in node_types
        assert "method_definition" in node_types
        # Classes
        assert "class_declaration" in node_types
        assert "abstract_class_declaration" in node_types
        # TypeScript-specific
        assert "interface_declaration" in node_types
        assert "type_alias_declaration" in node_types
        assert "enum_declaration" in node_types


# =============================================================================
# Test Function Extraction - Basic
# =============================================================================


class TestExtractFunctionsBasic:
    """Tests for basic function extraction."""

    def test_extract_simple_function(self, extractor, parse_typescript):
        """Test extraction of a simple function."""
        source = """
function hello() {
    console.log("Hello");
}
"""
        tree, source_bytes = parse_typescript(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.ts")

        assert len(functions) == 1
        assert functions[0].name == "hello"
        assert functions[0].element_type == CodeElementType.FUNCTION

    def test_extract_multiple_functions(self, extractor, parse_typescript):
        """Test extraction of multiple functions."""
        source = """
function func1() {}
function func2() {}
function func3() {}
"""
        tree, source_bytes = parse_typescript(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.ts")

        assert len(functions) == 3
        names = [f.name for f in functions]
        assert "func1" in names
        assert "func2" in names
        assert "func3" in names

    def test_function_has_location(self, extractor, parse_typescript):
        """Test that extracted function has location info."""
        source = """function hello() {}
"""
        tree, source_bytes = parse_typescript(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.ts")

        assert len(functions) == 1
        func = functions[0]
        assert func.location is not None
        assert func.location.start_line == 0
        assert func.location.start_column == 0

    def test_function_has_source_code(self, extractor, parse_typescript):
        """Test that extracted function has source code."""
        source = """function hello() {
    console.log("Hello");
}
"""
        tree, source_bytes = parse_typescript(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.ts")

        assert len(functions) == 1
        assert "function hello()" in functions[0].source_code
        assert 'console.log("Hello")' in functions[0].source_code


# =============================================================================
# Test Function Extraction - JSDoc
# =============================================================================


class TestExtractFunctionsJSDoc:
    """Tests for JSDoc extraction."""

    def test_extract_function_with_jsdoc(self, extractor, parse_typescript):
        """Test extraction of function with JSDoc."""
        source = """
/** Say hello. */
function hello() {
    console.log("Hello");
}
"""
        tree, source_bytes = parse_typescript(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.ts")

        assert len(functions) == 1
        assert functions[0].docstring == "Say hello."

    def test_extract_function_with_multiline_jsdoc(self, extractor, parse_typescript):
        """Test extraction of function with multiline JSDoc."""
        source = """
/**
 * Calculate the sum of x and y.
 *
 * @param x - First number
 * @param y - Second number
 * @returns The sum of x and y
 */
function add(x: number, y: number): number {
    return x + y;
}
"""
        tree, source_bytes = parse_typescript(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.ts")

        assert len(functions) == 1
        docstring = functions[0].docstring
        assert docstring is not None
        assert "Calculate the sum" in docstring
        assert "@param x" in docstring
        assert "@returns" in docstring

    def test_function_without_jsdoc(self, extractor, parse_typescript):
        """Test function without JSDoc has None."""
        source = """
function hello() {}
"""
        tree, source_bytes = parse_typescript(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.ts")

        assert len(functions) == 1
        assert functions[0].docstring is None


# =============================================================================
# Test Function Extraction - Parameters
# =============================================================================


class TestExtractFunctionsParameters:
    """Tests for parameter extraction."""

    def test_extract_function_with_params(self, extractor, parse_typescript):
        """Test extraction of function with parameters."""
        source = """
function add(a: number, b: number): number {
    return a + b;
}
"""
        tree, source_bytes = parse_typescript(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.ts")

        assert len(functions) == 1
        params = functions[0].parameters
        assert len(params) == 2
        assert params[0].name == "a"
        assert params[0].type_annotation == "number"
        assert params[1].name == "b"
        assert params[1].type_annotation == "number"

    def test_extract_function_with_optional_params(self, extractor, parse_typescript):
        """Test extraction of function with optional parameters."""
        source = """
function greet(name: string, greeting?: string): string {
    return greeting ? `${greeting}, ${name}!` : `Hello, ${name}!`;
}
"""
        tree, source_bytes = parse_typescript(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.ts")

        assert len(functions) == 1
        params = functions[0].parameters
        assert len(params) == 2
        assert params[0].name == "name"
        assert params[1].name == "greeting"

    def test_extract_function_with_rest_params(self, extractor, parse_typescript):
        """Test extraction of function with rest parameters."""
        source = """
function sum(...numbers: number[]): number {
    return numbers.reduce((a, b) => a + b, 0);
}
"""
        tree, source_bytes = parse_typescript(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.ts")

        assert len(functions) == 1
        params = functions[0].parameters
        assert len(params) >= 1
        # Rest parameter should have ... prefix
        assert any("...numbers" in p.name for p in params)


# =============================================================================
# Test Function Extraction - Return Types
# =============================================================================


class TestExtractFunctionsReturnTypes:
    """Tests for return type extraction."""

    def test_extract_function_with_return_type(self, extractor, parse_typescript):
        """Test extraction of function with return type."""
        source = """
function add(a: number, b: number): number {
    return a + b;
}
"""
        tree, source_bytes = parse_typescript(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.ts")

        assert len(functions) == 1
        assert functions[0].return_type == "number"

    def test_function_without_return_type(self, extractor, parse_typescript):
        """Test function without return type has None."""
        source = """
function hello() {
    console.log("Hello");
}
"""
        tree, source_bytes = parse_typescript(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.ts")

        assert len(functions) == 1
        assert functions[0].return_type is None

    def test_function_with_void_return_type(self, extractor, parse_typescript):
        """Test function with void return type."""
        source = """
function process(): void {
    console.log("Processing");
}
"""
        tree, source_bytes = parse_typescript(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.ts")

        assert len(functions) == 1
        assert functions[0].return_type == "void"

    def test_function_with_promise_return_type(self, extractor, parse_typescript):
        """Test function with Promise return type."""
        source = """
async function fetchData(): Promise<string> {
    return "data";
}
"""
        tree, source_bytes = parse_typescript(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.ts")

        assert len(functions) == 1
        assert "Promise" in functions[0].return_type


# =============================================================================
# Test Function Extraction - Async and Generators
# =============================================================================


class TestExtractFunctionsAsyncGenerators:
    """Tests for async functions and generators."""

    def test_extract_async_function(self, extractor, parse_typescript):
        """Test extraction of async function."""
        source = """
async function fetchData() {
    return await fetch("url");
}
"""
        tree, source_bytes = parse_typescript(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.ts")

        assert len(functions) == 1
        assert functions[0].is_async is True

    def test_extract_generator_function(self, extractor, parse_typescript):
        """Test extraction of generator function."""
        source = """
function* generateNumbers() {
    yield 1;
    yield 2;
    yield 3;
}
"""
        tree, source_bytes = parse_typescript(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.ts")

        assert len(functions) == 1
        assert functions[0].is_generator is True

    def test_non_async_function(self, extractor, parse_typescript):
        """Test that regular function is not marked as async."""
        source = """
function regular() {
    return 42;
}
"""
        tree, source_bytes = parse_typescript(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.ts")

        assert len(functions) == 1
        assert functions[0].is_async is False


# =============================================================================
# Test Arrow Function Extraction
# =============================================================================


class TestExtractArrowFunctions:
    """Tests for arrow function extraction."""

    def test_extract_arrow_function(self, extractor, parse_typescript):
        """Test extraction of arrow function."""
        source = """
const multiply = (a: number, b: number): number => a * b;
"""
        tree, source_bytes = parse_typescript(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.ts")

        assert len(functions) == 1
        assert functions[0].name == "multiply"
        assert functions[0].element_type == CodeElementType.FUNCTION

    def test_extract_arrow_function_with_body(self, extractor, parse_typescript):
        """Test extraction of arrow function with block body."""
        source = """
const calculate = (x: number): number => {
    const result = x * 2;
    return result;
};
"""
        tree, source_bytes = parse_typescript(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.ts")

        assert len(functions) == 1
        assert functions[0].name == "calculate"

    def test_extract_async_arrow_function(self, extractor, parse_typescript):
        """Test extraction of async arrow function."""
        source = """
const fetchUser = async (id: string): Promise<User> => {
    return await api.getUser(id);
};
"""
        tree, source_bytes = parse_typescript(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.ts")

        assert len(functions) == 1
        assert functions[0].name == "fetchUser"
        assert functions[0].is_async is True


# =============================================================================
# Test Method Extraction
# =============================================================================


class TestExtractMethods:
    """Tests for method extraction from classes."""

    def test_method_has_parent_class(self, extractor, parse_typescript):
        """Test that method has parent_class set."""
        source = """
class Calculator {
    add(a: number, b: number): number {
        return a + b;
    }
}
"""
        tree, source_bytes = parse_typescript(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.ts")

        assert len(functions) == 1
        assert functions[0].parent_class == "Calculator"

    def test_method_has_correct_type(self, extractor, parse_typescript):
        """Test that method has CodeElementType.METHOD."""
        source = """
class MyClass {
    myMethod(): void {}
}
"""
        tree, source_bytes = parse_typescript(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.ts")

        assert len(functions) == 1
        assert functions[0].element_type == CodeElementType.METHOD

    def test_method_qualified_name(self, extractor, parse_typescript):
        """Test method qualified_name includes class."""
        source = """
class MyClass {
    myMethod(): void {}
}
"""
        tree, source_bytes = parse_typescript(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.ts")

        assert len(functions) == 1
        assert functions[0].qualified_name == "MyClass.myMethod"

    def test_static_method(self, extractor, parse_typescript):
        """Test static method detection."""
        source = """
class Utils {
    static format(value: string): string {
        return value.trim();
    }
}
"""
        tree, source_bytes = parse_typescript(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.ts")

        assert len(functions) == 1
        assert functions[0].is_static is True

    def test_private_method(self, extractor, parse_typescript):
        """Test private method detection."""
        source = """
class Service {
    private internalProcess(): void {}
}
"""
        tree, source_bytes = parse_typescript(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.ts")

        assert len(functions) == 1
        assert functions[0].is_private is True

    def test_getter_method(self, extractor, parse_typescript):
        """Test getter method detection."""
        source = """
class Person {
    get fullName(): string {
        return `${this.firstName} ${this.lastName}`;
    }
}
"""
        tree, source_bytes = parse_typescript(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.ts")

        assert len(functions) == 1
        assert "get" in functions[0].decorators

    def test_setter_method(self, extractor, parse_typescript):
        """Test setter method detection."""
        source = """
class Person {
    set age(value: number) {
        this._age = value;
    }
}
"""
        tree, source_bytes = parse_typescript(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.ts")

        assert len(functions) == 1
        assert "set" in functions[0].decorators


# =============================================================================
# Test Class Extraction
# =============================================================================


class TestExtractClasses:
    """Tests for class extraction."""

    def test_extract_simple_class(self, extractor, parse_typescript):
        """Test extraction of simple class."""
        source = """
class MyClass {}
"""
        tree, source_bytes = parse_typescript(source)
        classes = extractor.extract_classes(tree, source_bytes, "/test.ts")

        assert len(classes) == 1
        assert classes[0].name == "MyClass"
        assert classes[0].element_type == CodeElementType.CLASS

    def test_extract_class_with_jsdoc(self, extractor, parse_typescript):
        """Test extraction of class with JSDoc."""
        source = """
/** A simple calculator class. */
class Calculator {
    add(a: number, b: number): number {
        return a + b;
    }
}
"""
        tree, source_bytes = parse_typescript(source)
        classes = extractor.extract_classes(tree, source_bytes, "/test.ts")

        assert len(classes) == 1
        assert classes[0].docstring == "A simple calculator class."

    def test_extract_class_with_extends(self, extractor, parse_typescript):
        """Test extraction of class with extends."""
        source = """
class Child extends Parent {}
"""
        tree, source_bytes = parse_typescript(source)
        classes = extractor.extract_classes(tree, source_bytes, "/test.ts")

        assert len(classes) == 1
        assert "Parent" in classes[0].bases

    def test_extract_class_with_implements(self, extractor, parse_typescript):
        """Test extraction of class with implements."""
        source = """
class UserService implements IUserService {}
"""
        tree, source_bytes = parse_typescript(source)
        classes = extractor.extract_classes(tree, source_bytes, "/test.ts")

        assert len(classes) == 1
        assert "IUserService" in classes[0].bases

    def test_extract_class_methods_list(self, extractor, parse_typescript):
        """Test that class has methods list."""
        source = """
class Calculator {
    add(a: number, b: number): number { return a + b; }
    subtract(a: number, b: number): number { return a - b; }
}
"""
        tree, source_bytes = parse_typescript(source)
        classes = extractor.extract_classes(tree, source_bytes, "/test.ts")

        assert len(classes) == 1
        methods = classes[0].methods
        assert "add" in methods
        assert "subtract" in methods

    def test_extract_abstract_class(self, extractor, parse_typescript):
        """Test extraction of abstract class."""
        source = """
abstract class BaseService {
    abstract process(): void;
}
"""
        tree, source_bytes = parse_typescript(source)
        classes = extractor.extract_classes(tree, source_bytes, "/test.ts")

        assert len(classes) == 1
        assert classes[0].is_abstract is True


# =============================================================================
# Test Interface Extraction
# =============================================================================


class TestExtractInterfaces:
    """Tests for interface extraction."""

    def test_extract_simple_interface(self, extractor, parse_typescript):
        """Test extraction of simple interface."""
        source = """
interface IUser {
    name: string;
    age: number;
}
"""
        tree, source_bytes = parse_typescript(source)
        interfaces = extractor.extract_interfaces(tree, source_bytes, "/test.ts")

        assert len(interfaces) == 1
        assert interfaces[0].name == "IUser"
        assert interfaces[0].element_type == CodeElementType.INTERFACE

    def test_extract_interface_with_jsdoc(self, extractor, parse_typescript):
        """Test extraction of interface with JSDoc."""
        source = """
/** User interface. */
interface IUser {
    name: string;
}
"""
        tree, source_bytes = parse_typescript(source)
        interfaces = extractor.extract_interfaces(tree, source_bytes, "/test.ts")

        assert len(interfaces) == 1
        assert interfaces[0].docstring == "User interface."

    def test_extract_interface_with_extends(self, extractor, parse_typescript):
        """Test extraction of interface with extends."""
        source = """
interface IAdmin extends IUser {
    permissions: string[];
}
"""
        tree, source_bytes = parse_typescript(source)
        interfaces = extractor.extract_interfaces(tree, source_bytes, "/test.ts")

        assert len(interfaces) == 1
        assert "IUser" in interfaces[0].bases

    def test_interface_is_abstract(self, extractor, parse_typescript):
        """Test that interfaces are marked as abstract."""
        source = """
interface IService {}
"""
        tree, source_bytes = parse_typescript(source)
        interfaces = extractor.extract_interfaces(tree, source_bytes, "/test.ts")

        assert len(interfaces) == 1
        assert interfaces[0].is_abstract is True


# =============================================================================
# Test Type Alias Extraction
# =============================================================================


class TestExtractTypeAliases:
    """Tests for type alias extraction."""

    def test_extract_simple_type_alias(self, extractor, parse_typescript):
        """Test extraction of simple type alias."""
        source = """
type ID = string;
"""
        tree, source_bytes = parse_typescript(source)
        types = extractor.extract_types(tree, source_bytes, "/test.ts")

        assert len(types) == 1
        assert types[0].name == "ID"
        assert types[0].element_type == CodeElementType.TYPE_ALIAS

    def test_extract_union_type_alias(self, extractor, parse_typescript):
        """Test extraction of union type alias."""
        source = """
type Status = "active" | "inactive" | "pending";
"""
        tree, source_bytes = parse_typescript(source)
        types = extractor.extract_types(tree, source_bytes, "/test.ts")

        assert len(types) == 1
        assert types[0].name == "Status"

    def test_extract_type_alias_with_jsdoc(self, extractor, parse_typescript):
        """Test extraction of type alias with JSDoc."""
        source = """
/** User status type. */
type Status = "active" | "inactive";
"""
        tree, source_bytes = parse_typescript(source)
        types = extractor.extract_types(tree, source_bytes, "/test.ts")

        assert len(types) == 1
        assert types[0].docstring == "User status type."


# =============================================================================
# Test Enum Extraction
# =============================================================================


class TestExtractEnums:
    """Tests for enum extraction."""

    def test_extract_simple_enum(self, extractor, parse_typescript):
        """Test extraction of simple enum."""
        source = """
enum Color {
    Red,
    Green,
    Blue
}
"""
        tree, source_bytes = parse_typescript(source)
        enums = extractor.extract_enums(tree, source_bytes, "/test.ts")

        assert len(enums) == 1
        assert enums[0].name == "Color"
        assert enums[0].element_type == CodeElementType.ENUM

    def test_extract_enum_members(self, extractor, parse_typescript):
        """Test extraction of enum members."""
        source = """
enum Direction {
    Up,
    Down,
    Left,
    Right
}
"""
        tree, source_bytes = parse_typescript(source)
        enums = extractor.extract_enums(tree, source_bytes, "/test.ts")

        assert len(enums) == 1
        members = enums[0].methods  # Members stored in methods field
        assert "Up" in members
        assert "Down" in members
        assert "Left" in members
        assert "Right" in members

    def test_extract_enum_with_jsdoc(self, extractor, parse_typescript):
        """Test extraction of enum with JSDoc."""
        source = """
/** Color enumeration. */
enum Color {
    Red,
    Green,
    Blue
}
"""
        tree, source_bytes = parse_typescript(source)
        enums = extractor.extract_enums(tree, source_bytes, "/test.ts")

        assert len(enums) == 1
        assert enums[0].docstring == "Color enumeration."

    def test_extract_const_enum(self, extractor, parse_typescript):
        """Test extraction of const enum."""
        source = """
const enum HttpStatus {
    OK = 200,
    NotFound = 404
}
"""
        tree, source_bytes = parse_typescript(source)
        enums = extractor.extract_enums(tree, source_bytes, "/test.ts")

        assert len(enums) == 1
        assert "const" in enums[0].decorators


# =============================================================================
# Test Export Handling
# =============================================================================


class TestExportHandling:
    """Tests for exported declarations."""

    def test_extract_exported_function(self, extractor, parse_typescript):
        """Test extraction of exported function."""
        source = """
export function greet(name: string): string {
    return `Hello, ${name}!`;
}
"""
        tree, source_bytes = parse_typescript(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.ts")

        assert len(functions) == 1
        assert functions[0].name == "greet"

    def test_extract_exported_class(self, extractor, parse_typescript):
        """Test extraction of exported class."""
        source = """
export class UserService {
    getUser(id: string): User {
        return {} as User;
    }
}
"""
        tree, source_bytes = parse_typescript(source)
        classes = extractor.extract_classes(tree, source_bytes, "/test.ts")

        assert len(classes) == 1
        assert classes[0].name == "UserService"

    def test_extract_exported_arrow_function(self, extractor, parse_typescript):
        """Test extraction of exported arrow function."""
        source = """
export const multiply = (a: number, b: number): number => a * b;
"""
        tree, source_bytes = parse_typescript(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.ts")

        assert len(functions) == 1
        assert functions[0].name == "multiply"


# =============================================================================
# Test JavaScript Support
# =============================================================================


class TestJavaScriptSupport:
    """Tests for JavaScript file support."""

    def test_extract_javascript_function(self, extractor, parse_javascript):
        """Test extraction from JavaScript file."""
        source = """
function hello() {
    console.log("Hello");
}
"""
        tree, source_bytes = parse_javascript(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.js")

        assert len(functions) == 1
        assert functions[0].name == "hello"

    def test_extract_javascript_class(self, extractor, parse_javascript):
        """Test extraction of JavaScript class."""
        source = """
class Calculator {
    add(a, b) {
        return a + b;
    }
}
"""
        tree, source_bytes = parse_javascript(source)
        classes = extractor.extract_classes(tree, source_bytes, "/test.js")

        assert len(classes) == 1
        assert classes[0].name == "Calculator"


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_empty_source_code(self, extractor, parse_typescript):
        """Test extraction from empty source."""
        source = ""
        tree, source_bytes = parse_typescript(source)

        functions = extractor.extract_functions(tree, source_bytes, "/test.ts")
        classes = extractor.extract_classes(tree, source_bytes, "/test.ts")
        interfaces = extractor.extract_interfaces(tree, source_bytes, "/test.ts")

        assert len(functions) == 0
        assert len(classes) == 0
        assert len(interfaces) == 0

    def test_only_comments(self, extractor, parse_typescript):
        """Test extraction from source with only comments."""
        source = """
// This is a comment
/* Another comment */
"""
        tree, source_bytes = parse_typescript(source)

        functions = extractor.extract_functions(tree, source_bytes, "/test.ts")
        classes = extractor.extract_classes(tree, source_bytes, "/test.ts")

        assert len(functions) == 0
        assert len(classes) == 0

    def test_line_count(self, extractor, parse_typescript):
        """Test line_count is calculated correctly."""
        source = """function multiLine(): void {
    const line1 = 1;
    const line2 = 2;
    const line3 = 3;
    console.log(line1 + line2 + line3);
}
"""
        tree, source_bytes = parse_typescript(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.ts")

        assert len(functions) == 1
        assert functions[0].line_count == 6

    def test_extract_all_combined(self, extractor, parse_typescript):
        """Test extract_all returns all element types."""
        source = """
function standalone(): void {}

class MyClass {
    method(): void {}
}

interface IService {
    process(): void;
}

type ID = string;

enum Status {
    Active,
    Inactive
}
"""
        tree, source_bytes = parse_typescript(source)
        all_elements = extractor.extract_all(tree, source_bytes, "/test.ts")

        # Should have function + class + method + interface + type + enum
        assert len(all_elements) >= 5
        types = [e.element_type for e in all_elements]
        assert CodeElementType.FUNCTION in types
        assert CodeElementType.CLASS in types
        assert CodeElementType.INTERFACE in types
        assert CodeElementType.TYPE_ALIAS in types
        assert CodeElementType.ENUM in types

    def test_generic_function(self, extractor, parse_typescript):
        """Test extraction of generic function."""
        source = """
function identity<T>(arg: T): T {
    return arg;
}
"""
        tree, source_bytes = parse_typescript(source)
        functions = extractor.extract_functions(tree, source_bytes, "/test.ts")

        assert len(functions) == 1
        assert functions[0].name == "identity"

    def test_generic_class(self, extractor, parse_typescript):
        """Test extraction of generic class."""
        source = """
class Container<T> {
    private value: T;

    constructor(value: T) {
        this.value = value;
    }
}
"""
        tree, source_bytes = parse_typescript(source)
        classes = extractor.extract_classes(tree, source_bytes, "/test.ts")

        assert len(classes) == 1
        assert classes[0].name == "Container"


# =============================================================================
# Test Integration with Registry
# =============================================================================


class TestRegistryIntegration:
    """Tests for TypeScriptExtractor integration with registry."""

    def test_extractor_registered_for_typescript(self):
        """Test that TypeScriptExtractor can be registered for typescript."""
        # Import the module to trigger auto-registration
        from zapomni_core.treesitter.extractors import typescript  # noqa: F401
        from zapomni_core.treesitter.parser.registry import LanguageParserRegistry

        registry = LanguageParserRegistry()
        # Manually register since tests reset the registry
        registry.register_extractor("typescript", TypeScriptExtractor())
        extractor = registry.get_extractor("typescript")

        assert extractor is not None
        assert isinstance(extractor, TypeScriptExtractor)

    def test_extractor_registered_for_javascript(self):
        """Test that TypeScriptExtractor can be registered for javascript."""
        from zapomni_core.treesitter.parser.registry import LanguageParserRegistry

        registry = LanguageParserRegistry()
        # Manually register since tests reset the registry
        registry.register_extractor("javascript", TypeScriptExtractor())
        extractor = registry.get_extractor("javascript")

        assert extractor is not None
        assert isinstance(extractor, TypeScriptExtractor)

    def test_language_in_config(self):
        """Test that 'typescript' is in LANGUAGES_WITH_EXTRACTORS."""
        from zapomni_core.treesitter.config import LANGUAGES_WITH_EXTRACTORS

        assert "typescript" in LANGUAGES_WITH_EXTRACTORS
        assert "javascript" in LANGUAGES_WITH_EXTRACTORS
