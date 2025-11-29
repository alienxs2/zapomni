"""Tests for CallGraphAnalyzer - Go language.

This module contains comprehensive tests for Go call extraction
in the CallGraphAnalyzer, covering function calls, method calls,
package calls, deferred calls, goroutines, and more.

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
def parse_go():
    """Factory fixture to parse Go source code."""
    parser = get_parser("go")

    def _parse(source: str) -> tuple[Tree, bytes]:
        source_bytes = source.encode("utf-8")
        tree = parser.parse(source_bytes)
        return tree, source_bytes

    return _parse


def make_go_func(
    name: str,
    qualified_name: str,
    source: str,
    start_line: int,
    start_col: int,
    end_line: int,
    end_col: int,
    start_byte: int,
    end_byte: int,
    file_path: str = "/test.go",
) -> ExtractedCode:
    """Helper to create ExtractedCode for Go functions."""
    return ExtractedCode(
        name=name,
        qualified_name=qualified_name,
        element_type=CodeElementType.FUNCTION,
        language="go",
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
# Test Go Call Extraction
# =============================================================================


class TestGoCallExtraction:
    """Tests for Go call extraction."""

    def test_go_simple_call(self, analyzer, parse_go):
        """Test extraction of simple function call: foo()."""
        source = """package main

func caller() {
    foo()
}

func foo() {}
"""
        tree, source_bytes = parse_go(source)

        # Create ExtractedCode for caller function
        func = make_go_func(
            name="caller",
            qualified_name="main.caller",
            source="func caller() {\n    foo()\n}",
            start_line=2,
            start_col=0,
            end_line=4,
            end_col=1,
            start_byte=14,
            end_byte=40,
        )

        calls = analyzer.analyze_function(func, tree, source_bytes, "go")

        assert len(calls) == 1
        assert calls[0].callee_name == "foo"
        assert calls[0].call_type == CallType.FUNCTION
        assert calls[0].receiver is None
        assert calls[0].arguments_count == 0

    def test_go_package_call(self, analyzer, parse_go):
        """Test extraction of package call: fmt.Println()."""
        source = """package main

import "fmt"

func caller() {
    fmt.Println("hello")
}
"""
        tree, source_bytes = parse_go(source)

        func = make_go_func(
            name="caller",
            qualified_name="main.caller",
            source='func caller() {\n    fmt.Println("hello")\n}',
            start_line=4,
            start_col=0,
            end_line=6,
            end_col=1,
            start_byte=29,
            end_byte=68,
        )

        calls = analyzer.analyze_function(func, tree, source_bytes, "go")

        assert len(calls) == 1
        assert calls[0].callee_name == "Println"
        assert calls[0].receiver == "fmt"
        # Heuristic: lowercase receiver (fmt) is detected as METHOD
        # (can't distinguish package from variable without type info)
        assert calls[0].call_type == CallType.METHOD
        assert calls[0].arguments_count == 1

    def test_go_method_call(self, analyzer, parse_go):
        """Test extraction of method call: obj.Method()."""
        source = """package main

type MyType struct{}

func (m *MyType) Method() {}

func caller() {
    obj := &MyType{}
    obj.Method()
}
"""
        tree, source_bytes = parse_go(source)

        func = make_go_func(
            name="caller",
            qualified_name="main.caller",
            source="func caller() {\n    obj := &MyType{}\n    obj.Method()\n}",
            start_line=6,
            start_col=0,
            end_line=9,
            end_col=1,
            start_byte=68,
            end_byte=123,
        )

        calls = analyzer.analyze_function(func, tree, source_bytes, "go")

        # Should find the method call
        method_calls = [c for c in calls if c.callee_name == "Method"]
        assert len(method_calls) == 1
        assert method_calls[0].receiver == "obj"
        assert method_calls[0].call_type == CallType.METHOD

    def test_go_pointer_method_call(self, analyzer, parse_go):
        """Test extraction of pointer method call: ptr.Method()."""
        source = """package main

type Data struct {
    value int
}

func (d *Data) SetValue(v int) {
    d.value = v
}

func caller() {
    ptr := &Data{}
    ptr.SetValue(42)
}
"""
        tree, source_bytes = parse_go(source)

        func = make_go_func(
            name="caller",
            qualified_name="main.caller",
            source="func caller() {\n    ptr := &Data{}\n    ptr.SetValue(42)\n}",
            start_line=10,
            start_col=0,
            end_line=13,
            end_col=1,
            start_byte=97,
            end_byte=155,
        )

        calls = analyzer.analyze_function(func, tree, source_bytes, "go")

        method_calls = [c for c in calls if c.callee_name == "SetValue"]
        assert len(method_calls) == 1
        assert method_calls[0].receiver == "ptr"
        assert method_calls[0].call_type == CallType.METHOD
        assert method_calls[0].arguments_count == 1

    def test_go_deferred_call(self, analyzer, parse_go):
        """Test extraction of deferred call: defer Close()."""
        source = """package main

func caller() {
    defer Close()
}

func Close() {}
"""
        tree, source_bytes = parse_go(source)

        func = make_go_func(
            name="caller",
            qualified_name="main.caller",
            source="func caller() {\n    defer Close()\n}",
            start_line=2,
            start_col=0,
            end_line=4,
            end_col=1,
            start_byte=14,
            end_byte=48,
        )

        calls = analyzer.analyze_function(func, tree, source_bytes, "go")

        assert len(calls) == 1
        assert calls[0].callee_name == "Close"
        # Close is PascalCase so detected as CONSTRUCTOR
        assert calls[0].call_type == CallType.CONSTRUCTOR

    def test_go_goroutine_call(self, analyzer, parse_go):
        """Test extraction of goroutine call: go run()."""
        source = """package main

func caller() {
    go run()
}

func run() {}
"""
        tree, source_bytes = parse_go(source)

        func = make_go_func(
            name="caller",
            qualified_name="main.caller",
            source="func caller() {\n    go run()\n}",
            start_line=2,
            start_col=0,
            end_line=4,
            end_col=1,
            start_byte=14,
            end_byte=43,
        )

        calls = analyzer.analyze_function(func, tree, source_bytes, "go")

        assert len(calls) == 1
        assert calls[0].callee_name == "run"
        assert calls[0].call_type == CallType.FUNCTION

    def test_go_nested_call(self, analyzer, parse_go):
        """Test extraction of nested calls: foo(bar())."""
        source = """package main

func caller() {
    foo(bar())
}

func foo(x int) {}
func bar() int { return 1 }
"""
        tree, source_bytes = parse_go(source)

        func = make_go_func(
            name="caller",
            qualified_name="main.caller",
            source="func caller() {\n    foo(bar())\n}",
            start_line=2,
            start_col=0,
            end_line=4,
            end_col=1,
            start_byte=14,
            end_byte=46,
        )

        calls = analyzer.analyze_function(func, tree, source_bytes, "go")

        assert len(calls) == 2
        callee_names = {c.callee_name for c in calls}
        assert "foo" in callee_names
        assert "bar" in callee_names

    def test_go_interface_method_call(self, analyzer, parse_go):
        """Test extraction of interface method call."""
        source = """package main

type Reader interface {
    Read() []byte
}

func caller(r Reader) {
    r.Read()
}
"""
        tree, source_bytes = parse_go(source)

        func = make_go_func(
            name="caller",
            qualified_name="main.caller",
            source="func caller(r Reader) {\n    r.Read()\n}",
            start_line=6,
            start_col=0,
            end_line=8,
            end_col=1,
            start_byte=52,
            end_byte=90,
        )

        calls = analyzer.analyze_function(func, tree, source_bytes, "go")

        assert len(calls) == 1
        assert calls[0].callee_name == "Read"
        assert calls[0].receiver == "r"
        assert calls[0].call_type == CallType.METHOD

    def test_go_constructor_pattern(self, analyzer, parse_go):
        """Test extraction of constructor pattern: NewMyType()."""
        source = """package main

type MyType struct{}

func NewMyType() *MyType {
    return &MyType{}
}

func caller() {
    obj := NewMyType()
    _ = obj
}
"""
        tree, source_bytes = parse_go(source)

        func = make_go_func(
            name="caller",
            qualified_name="main.caller",
            source="func caller() {\n    obj := NewMyType()\n    _ = obj\n}",
            start_line=8,
            start_col=0,
            end_line=11,
            end_col=1,
            start_byte=81,
            end_byte=131,
        )

        calls = analyzer.analyze_function(func, tree, source_bytes, "go")

        assert len(calls) == 1
        assert calls[0].callee_name == "NewMyType"
        # PascalCase indicates constructor
        assert calls[0].call_type == CallType.CONSTRUCTOR

    def test_go_multiple_calls(self, analyzer, parse_go):
        """Test extraction of multiple calls in one function."""
        source = """package main

import "fmt"

func caller() {
    fmt.Println("start")
    foo()
    bar(1, 2)
    obj.Method()
    fmt.Println("end")
}

func foo() {}
func bar(a, b int) {}
"""
        tree, source_bytes = parse_go(source)

        func = make_go_func(
            name="caller",
            qualified_name="main.caller",
            source=(
                'func caller() {\n    fmt.Println("start")\n    foo()\n    '
                'bar(1, 2)\n    obj.Method()\n    fmt.Println("end")\n}'
            ),
            start_line=4,
            start_col=0,
            end_line=10,
            end_col=1,
            start_byte=29,
            end_byte=130,
        )

        calls = analyzer.analyze_function(func, tree, source_bytes, "go")

        # Should find: 2x Println, foo, bar, Method
        assert len(calls) >= 5
        callee_names = [c.callee_name for c in calls]
        assert callee_names.count("Println") == 2
        assert "foo" in callee_names
        assert "bar" in callee_names
        assert "Method" in callee_names
