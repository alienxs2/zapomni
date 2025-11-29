"""Tests for CallGraphAnalyzer - Rust language.

This module contains comprehensive tests for Rust call extraction
in the CallGraphAnalyzer, covering function calls, method calls,
static method calls, macros, chained calls, async/await, and more.

Test count: 10 tests as per Issue #24 requirements.
"""

import pytest
from tree_sitter import Tree
from tree_sitter_language_pack import get_parser

from zapomni_core.treesitter.analyzers.call_graph import CallGraphAnalyzer
from zapomni_core.treesitter.models import (
    ASTNodeLocation,
    CallType,
    CodeElementType,
    ExtractedCode,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def analyzer():
    """Create a CallGraphAnalyzer instance."""
    return CallGraphAnalyzer()


@pytest.fixture
def parse_rust():
    """Factory fixture to parse Rust source code."""
    parser = get_parser("rust")

    def _parse(source: str) -> tuple[Tree, bytes]:
        source_bytes = source.encode("utf-8")
        tree = parser.parse(source_bytes)
        return tree, source_bytes

    return _parse


def make_rust_func(
    name: str,
    qualified_name: str,
    source: str,
    start_line: int,
    start_col: int,
    end_line: int,
    end_col: int,
    start_byte: int,
    end_byte: int,
    file_path: str = "/test.rs",
) -> ExtractedCode:
    """Helper to create ExtractedCode for Rust functions."""
    return ExtractedCode(
        name=name,
        qualified_name=qualified_name,
        element_type=CodeElementType.FUNCTION,
        language="rust",
        file_path=file_path,
        location=ASTNodeLocation(
            start_line=start_line,
            end_line=end_line,
            start_column=start_col,
            end_column=end_col,
            start_byte=start_byte,
            end_byte=end_byte,
        ),
        source_code=source,
    )


# =============================================================================
# Test Rust Call Extraction
# =============================================================================


class TestRustCallExtraction:
    """Tests for Rust call extraction."""

    def test_rust_function_call(self, analyzer, parse_rust):
        """Test extraction of simple function call: foo()."""
        source = """fn caller() {
    foo();
}

fn foo() {}
"""
        tree, source_bytes = parse_rust(source)

        func = make_rust_func(
            name="caller",
            qualified_name="caller",
            source="fn caller() {\n    foo();\n}",
            start_line=0,
            start_col=0,
            end_line=2,
            end_col=1,
            start_byte=0,
            end_byte=26,
        )

        calls = analyzer.analyze_function(func, tree, source_bytes, "rust")

        assert len(calls) == 1
        assert calls[0].callee_name == "foo"
        assert calls[0].call_type == CallType.FUNCTION
        assert calls[0].receiver is None
        assert calls[0].arguments_count == 0

    def test_rust_method_call(self, analyzer, parse_rust):
        """Test extraction of method call: obj.method()."""
        source = """struct Data {
    value: i32,
}

impl Data {
    fn get_value(&self) -> i32 {
        self.value
    }
}

fn caller() {
    let obj = Data { value: 42 };
    obj.get_value();
}
"""
        tree, source_bytes = parse_rust(source)

        func = make_rust_func(
            name="caller",
            qualified_name="caller",
            source="fn caller() {\n    let obj = Data { value: 42 };\n    obj.get_value();\n}",
            start_line=10,
            start_col=0,
            end_line=13,
            end_col=1,
            start_byte=110,
            end_byte=179,
        )

        calls = analyzer.analyze_function(func, tree, source_bytes, "rust")

        method_calls = [c for c in calls if c.callee_name == "get_value"]
        assert len(method_calls) == 1
        assert method_calls[0].receiver == "obj"
        assert method_calls[0].call_type == CallType.METHOD

    def test_rust_static_method_call(self, analyzer, parse_rust):
        """Test extraction of static method call: Type::new()."""
        source = """struct Point {
    x: f64,
    y: f64,
}

impl Point {
    fn new(x: f64, y: f64) -> Self {
        Point { x, y }
    }
}

fn caller() {
    let p = Point::new(1.0, 2.0);
}
"""
        tree, source_bytes = parse_rust(source)

        func = make_rust_func(
            name="caller",
            qualified_name="caller",
            source="fn caller() {\n    let p = Point::new(1.0, 2.0);\n}",
            start_line=11,
            start_col=0,
            end_line=13,
            end_col=1,
            start_byte=129,
            end_byte=177,
        )

        calls = analyzer.analyze_function(func, tree, source_bytes, "rust")

        assert len(calls) == 1
        assert calls[0].callee_name == "new"
        assert calls[0].receiver == "Point"
        assert calls[0].call_type == CallType.STATIC
        assert calls[0].arguments_count == 2

    def test_rust_macro_call(self, analyzer, parse_rust):
        """Test extraction of macro call: println!()."""
        source = """fn caller() {
    println!("Hello, World!");
}
"""
        tree, source_bytes = parse_rust(source)

        func = make_rust_func(
            name="caller",
            qualified_name="caller",
            source='fn caller() {\n    println!("Hello, World!");\n}',
            start_line=0,
            start_col=0,
            end_line=2,
            end_col=1,
            start_byte=0,
            end_byte=46,
        )

        calls = analyzer.analyze_function(func, tree, source_bytes, "rust")

        assert len(calls) == 1
        assert calls[0].callee_name == "println!"
        assert calls[0].call_type == CallType.MACRO

    def test_rust_vec_macro(self, analyzer, parse_rust):
        """Test extraction of vec! macro: vec![1, 2, 3]."""
        source = """fn caller() {
    let v = vec![1, 2, 3];
}
"""
        tree, source_bytes = parse_rust(source)

        func = make_rust_func(
            name="caller",
            qualified_name="caller",
            source="fn caller() {\n    let v = vec![1, 2, 3];\n}",
            start_line=0,
            start_col=0,
            end_line=2,
            end_col=1,
            start_byte=0,
            end_byte=41,
        )

        calls = analyzer.analyze_function(func, tree, source_bytes, "rust")

        assert len(calls) == 1
        assert calls[0].callee_name == "vec!"
        assert calls[0].call_type == CallType.MACRO

    def test_rust_chained_call(self, analyzer, parse_rust):
        """Test extraction of chained calls: foo().bar().baz()."""
        source = """struct Builder;

impl Builder {
    fn new() -> Self { Builder }
    fn step1(self) -> Self { self }
    fn step2(self) -> Self { self }
}

fn caller() {
    Builder::new().step1().step2();
}
"""
        tree, source_bytes = parse_rust(source)

        func = make_rust_func(
            name="caller",
            qualified_name="caller",
            source="fn caller() {\n    Builder::new().step1().step2();\n}",
            start_line=8,
            start_col=0,
            end_line=10,
            end_col=1,
            start_byte=136,
            end_byte=186,
        )

        calls = analyzer.analyze_function(func, tree, source_bytes, "rust")

        # Should find: new, step1, step2
        assert len(calls) == 3
        callee_names = {c.callee_name for c in calls}
        assert "new" in callee_names
        assert "step1" in callee_names
        assert "step2" in callee_names

        # Check chaining detection
        chained_calls = [c for c in calls if c.is_chained]
        assert len(chained_calls) >= 1

    def test_rust_await_call(self, analyzer, parse_rust):
        """Test extraction of async call with await: foo().await."""
        source = """async fn fetch_data() -> String {
    String::new()
}

async fn caller() {
    let data = fetch_data().await;
}
"""
        tree, source_bytes = parse_rust(source)

        func = make_rust_func(
            name="caller",
            qualified_name="caller",
            source="async fn caller() {\n    let data = fetch_data().await;\n}",
            start_line=4,
            start_col=0,
            end_line=6,
            end_col=1,
            start_byte=51,
            end_byte=107,
        )

        calls = analyzer.analyze_function(func, tree, source_bytes, "rust")

        fetch_calls = [c for c in calls if c.callee_name == "fetch_data"]
        assert len(fetch_calls) == 1
        assert fetch_calls[0].is_await is True

    def test_rust_trait_method_call(self, analyzer, parse_rust):
        """Test extraction of trait method call."""
        source = """trait Greet {
    fn greet(&self);
}

struct Person;

impl Greet for Person {
    fn greet(&self) {}
}

fn caller(p: impl Greet) {
    p.greet();
}
"""
        tree, source_bytes = parse_rust(source)

        func = make_rust_func(
            name="caller",
            qualified_name="caller",
            source="fn caller(p: impl Greet) {\n    p.greet();\n}",
            start_line=10,
            start_col=0,
            end_line=12,
            end_col=1,
            start_byte=113,
            end_byte=156,
        )

        calls = analyzer.analyze_function(func, tree, source_bytes, "rust")

        assert len(calls) == 1
        assert calls[0].callee_name == "greet"
        assert calls[0].receiver == "p"
        assert calls[0].call_type == CallType.METHOD

    def test_rust_closure_boundary(self, analyzer, parse_rust):
        """Test that closures are treated as boundaries (calls inside closures not extracted)."""
        source = """fn caller() {
    outer_call();
    let closure = || {
        inner_call();
    };
}

fn outer_call() {}
fn inner_call() {}
"""
        tree, source_bytes = parse_rust(source)

        func = make_rust_func(
            name="caller",
            qualified_name="caller",
            source="fn caller() {\n    outer_call();\n    let closure = || {\n        inner_call();\n    };\n}",
            start_line=0,
            start_col=0,
            end_line=5,
            end_col=1,
            start_byte=0,
            end_byte=85,
        )

        calls = analyzer.analyze_function(func, tree, source_bytes, "rust")

        # Should only find outer_call, not inner_call (inside closure)
        callee_names = {c.callee_name for c in calls}
        assert "outer_call" in callee_names
        assert "inner_call" not in callee_names

    def test_rust_multiple_calls(self, analyzer, parse_rust):
        """Test extraction of multiple calls in one function."""
        source = """struct Data;

impl Data {
    fn new() -> Self { Data }
    fn process(&self) {}
}

fn caller() {
    println!("start");
    let d = Data::new();
    d.process();
    helper(1, 2);
    println!("end");
}

fn helper(a: i32, b: i32) {}
"""
        tree, source_bytes = parse_rust(source)

        func = make_rust_func(
            name="caller",
            qualified_name="caller",
            source='fn caller() {\n    println!("start");\n    let d = Data::new();\n    d.process();\n    helper(1, 2);\n    println!("end");\n}',
            start_line=7,
            start_col=0,
            end_line=13,
            end_col=1,
            start_byte=75,
            end_byte=195,
        )

        calls = analyzer.analyze_function(func, tree, source_bytes, "rust")

        # Should find: 2x println!, new, process, helper
        assert len(calls) >= 5
        callee_names = [c.callee_name for c in calls]
        assert callee_names.count("println!") == 2
        assert "new" in callee_names
        assert "process" in callee_names
        assert "helper" in callee_names

        # Check call types
        macro_calls = [c for c in calls if c.call_type == CallType.MACRO]
        assert len(macro_calls) == 2
