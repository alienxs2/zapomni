"""Pytest fixtures for treesitter tests."""

import pytest
from tree_sitter import Tree
from tree_sitter_language_pack import get_parser

from zapomni_core.treesitter.models import (
    ASTNodeLocation,
    CodeElementType,
    ExtractedCode,
    ParameterInfo,
    ParseResult,
)
from zapomni_core.treesitter.parser.registry import LanguageParserRegistry
from zapomni_core.treesitter.parser.factory import ParserFactory


@pytest.fixture
def sample_location():
    """Create a sample ASTNodeLocation for testing."""
    return ASTNodeLocation(
        start_line=0,
        end_line=5,
        start_column=0,
        end_column=10,
        start_byte=0,
        end_byte=100,
    )


@pytest.fixture
def sample_parameter():
    """Create a sample ParameterInfo for testing."""
    return ParameterInfo(
        name="param1",
        type_annotation="str",
        default_value=None,
    )


@pytest.fixture
def sample_extracted_code(sample_location):
    """Create a sample ExtractedCode for testing."""
    return ExtractedCode(
        name="test_function",
        qualified_name="test_module.test_function",
        element_type=CodeElementType.FUNCTION,
        language="python",
        file_path="/path/to/test.py",
        location=sample_location,
        source_code="def test_function(): pass",
    )


@pytest.fixture
def sample_parse_result(sample_extracted_code):
    """Create a sample ParseResult for testing."""
    return ParseResult(
        file_path="/path/to/test.py",
        language="python",
        functions=[sample_extracted_code],
        classes=[],
        parse_time_ms=10.5,
    )


@pytest.fixture
def python_source_code():
    """Sample Python source code for testing."""
    return b'''
def hello():
    """Say hello."""
    print("Hello, World!")

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

class Calculator:
    """A simple calculator."""

    def __init__(self, value: int = 0):
        self.value = value

    def add(self, x: int) -> int:
        self.value += x
        return self.value

    def _private_method(self):
        pass

async def async_function():
    pass
'''


@pytest.fixture
def python_tree(python_source_code) -> Tree:
    """Parse Python source code into a tree."""
    parser = get_parser("python")
    return parser.parse(python_source_code)


@pytest.fixture
def javascript_source_code():
    """Sample JavaScript source code for testing."""
    return b'''
function greet(name) {
    console.log("Hello, " + name);
}

const add = (a, b) => a + b;

class Person {
    constructor(name) {
        this.name = name;
    }

    sayHello() {
        console.log("Hello, I'm " + this.name);
    }
}

async function fetchData() {
    return await fetch("/api/data");
}
'''


@pytest.fixture
def javascript_tree(javascript_source_code) -> Tree:
    """Parse JavaScript source code into a tree."""
    parser = get_parser("javascript")
    return parser.parse(javascript_source_code)


@pytest.fixture
def rust_source_code():
    """Sample Rust source code for testing."""
    return b'''
fn main() {
    println!("Hello, World!");
}

fn add(a: i32, b: i32) -> i32 {
    a + b
}

struct Point {
    x: f64,
    y: f64,
}

impl Point {
    fn new(x: f64, y: f64) -> Self {
        Point { x, y }
    }

    fn distance(&self, other: &Point) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

trait Drawable {
    fn draw(&self);
}
'''


@pytest.fixture
def rust_tree(rust_source_code) -> Tree:
    """Parse Rust source code into a tree."""
    parser = get_parser("rust")
    return parser.parse(rust_source_code)


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the LanguageParserRegistry singleton before each test."""
    # Store original state
    LanguageParserRegistry.reset_instance()
    ParserFactory.reset()
    yield
    # Clean up after test
    LanguageParserRegistry.reset_instance()
    ParserFactory.reset()
