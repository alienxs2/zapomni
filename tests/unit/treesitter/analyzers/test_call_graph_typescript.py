"""Tests for CallGraphAnalyzer - TypeScript language.

This module contains comprehensive tests for TypeScript call extraction
in the CallGraphAnalyzer, covering function calls, method calls,
new expressions, async/await, optional chaining, arrow functions, and more.

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
def parse_ts():
    """Factory fixture to parse TypeScript source code."""
    parser = get_parser("typescript")

    def _parse(source: str) -> tuple[Tree, bytes]:
        source_bytes = source.encode("utf-8")
        tree = parser.parse(source_bytes)
        return tree, source_bytes

    return _parse


def make_ts_func(
    name: str,
    qualified_name: str,
    source: str,
    start_line: int,
    start_col: int,
    end_line: int,
    end_col: int,
    start_byte: int,
    end_byte: int,
    file_path: str = "/test.ts",
) -> ExtractedCode:
    """Helper to create ExtractedCode for TypeScript functions."""
    return ExtractedCode(
        name=name,
        qualified_name=qualified_name,
        element_type=CodeElementType.FUNCTION,
        language="typescript",
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
# Test TypeScript Call Extraction
# =============================================================================


class TestTypeScriptCallExtraction:
    """Tests for TypeScript call extraction."""

    def test_ts_function_call(self, analyzer, parse_ts):
        """Test extraction of simple function call: foo()."""
        source = """function caller() {
    foo();
}

function foo() {}
"""
        tree, source_bytes = parse_ts(source)

        func = make_ts_func(
            name="caller",
            qualified_name="caller",
            source="function caller() {\n    foo();\n}",
            start_line=0,
            start_col=0,
            end_line=2,
            end_col=1,
            start_byte=0,
            end_byte=32,
        )

        calls = analyzer.analyze_function(func, tree, source_bytes, "typescript")

        assert len(calls) == 1
        assert calls[0].callee_name == "foo"
        assert calls[0].call_type == CallType.FUNCTION
        assert calls[0].receiver is None
        assert calls[0].arguments_count == 0

    def test_ts_method_call(self, analyzer, parse_ts):
        """Test extraction of method call: obj.method()."""
        source = """class MyClass {
    method() {}
}

function caller() {
    const obj = new MyClass();
    obj.method();
}
"""
        tree, source_bytes = parse_ts(source)

        func = make_ts_func(
            name="caller",
            qualified_name="caller",
            source="function caller() {\n    const obj = new MyClass();\n    obj.method();\n}",
            start_line=4,
            start_col=0,
            end_line=7,
            end_col=1,
            start_byte=36,
            end_byte=107,
        )

        calls = analyzer.analyze_function(func, tree, source_bytes, "typescript")

        method_calls = [c for c in calls if c.callee_name == "method"]
        assert len(method_calls) == 1
        assert method_calls[0].receiver == "obj"
        assert method_calls[0].call_type == CallType.METHOD

    def test_ts_new_expression(self, analyzer, parse_ts):
        """Test extraction of new expression: new MyClass()."""
        source = """class MyClass {
    constructor(value: number) {}
}

function caller() {
    const obj = new MyClass(42);
}
"""
        tree, source_bytes = parse_ts(source)

        func = make_ts_func(
            name="caller",
            qualified_name="caller",
            source="function caller() {\n    const obj = new MyClass(42);\n}",
            start_line=4,
            start_col=0,
            end_line=6,
            end_col=1,
            start_byte=52,
            end_byte=105,
        )

        calls = analyzer.analyze_function(func, tree, source_bytes, "typescript")

        assert len(calls) == 1
        assert calls[0].callee_name == "MyClass"
        assert calls[0].call_type == CallType.CONSTRUCTOR
        assert calls[0].arguments_count == 1

    def test_ts_async_await(self, analyzer, parse_ts):
        """Test extraction of async/await call: await fetch()."""
        source = """async function caller() {
    const response = await fetch('/api/data');
}
"""
        tree, source_bytes = parse_ts(source)

        func = make_ts_func(
            name="caller",
            qualified_name="caller",
            source="async function caller() {\n    const response = await fetch('/api/data');\n}",
            start_line=0,
            start_col=0,
            end_line=2,
            end_col=1,
            start_byte=0,
            end_byte=73,
        )

        calls = analyzer.analyze_function(func, tree, source_bytes, "typescript")

        fetch_calls = [c for c in calls if c.callee_name == "fetch"]
        assert len(fetch_calls) == 1
        assert fetch_calls[0].is_await is True
        # fetch is a builtin
        assert fetch_calls[0].call_type == CallType.BUILTIN

    def test_ts_optional_chain(self, analyzer, parse_ts):
        """Test extraction of optional chaining: obj?.method()."""
        source = """interface Data {
    method?(): void;
}

function caller(obj: Data | undefined) {
    obj?.method?.();
}
"""
        tree, source_bytes = parse_ts(source)

        func = make_ts_func(
            name="caller",
            qualified_name="caller",
            source="function caller(obj: Data | undefined) {\n    obj?.method?.();\n}",
            start_line=4,
            start_col=0,
            end_line=6,
            end_col=1,
            start_byte=46,
            end_byte=110,
        )

        calls = analyzer.analyze_function(func, tree, source_bytes, "typescript")

        # Should detect the optional chain call
        assert len(calls) >= 1
        # The method call should be detected
        method_calls = [c for c in calls if c.callee_name == "method"]
        assert len(method_calls) == 1

    def test_ts_arrow_boundary(self, analyzer, parse_ts):
        """Test that arrow functions are treated as boundaries."""
        source = """function caller() {
    outerCall();
    const arrow = () => {
        innerCall();
    };
}

function outerCall() {}
function innerCall() {}
"""
        tree, source_bytes = parse_ts(source)

        func = make_ts_func(
            name="caller",
            qualified_name="caller",
            source="function caller() {\n    outerCall();\n    const arrow = () => {\n        innerCall();\n    };\n}",
            start_line=0,
            start_col=0,
            end_line=5,
            end_col=1,
            start_byte=0,
            end_byte=93,
        )

        calls = analyzer.analyze_function(func, tree, source_bytes, "typescript")

        # Should only find outerCall, not innerCall (inside arrow function)
        callee_names = {c.callee_name for c in calls}
        assert "outerCall" in callee_names
        assert "innerCall" not in callee_names

    def test_ts_promise_then(self, analyzer, parse_ts):
        """Test extraction of promise.then() call."""
        source = """function caller() {
    fetch('/api')
        .then(response => response.json())
        .then(data => console.log(data));
}
"""
        tree, source_bytes = parse_ts(source)

        func = make_ts_func(
            name="caller",
            qualified_name="caller",
            source="function caller() {\n    fetch('/api')\n        .then(response => response.json())\n        .then(data => console.log(data));\n}",
            start_line=0,
            start_col=0,
            end_line=4,
            end_col=1,
            start_byte=0,
            end_byte=121,
        )

        calls = analyzer.analyze_function(func, tree, source_bytes, "typescript")

        # Should find fetch and two then calls (but not json/log inside arrows)
        callee_names = [c.callee_name for c in calls]
        assert "fetch" in callee_names
        assert callee_names.count("then") == 2

    def test_ts_console_log(self, analyzer, parse_ts):
        """Test extraction of console.log() as builtin."""
        source = """function caller() {
    console.log("Hello");
    console.error("Error");
    console.warn("Warning");
}
"""
        tree, source_bytes = parse_ts(source)

        func = make_ts_func(
            name="caller",
            qualified_name="caller",
            source='function caller() {\n    console.log("Hello");\n    console.error("Error");\n    console.warn("Warning");\n}',
            start_line=0,
            start_col=0,
            end_line=4,
            end_col=1,
            start_byte=0,
            end_byte=104,
        )

        calls = analyzer.analyze_function(func, tree, source_bytes, "typescript")

        assert len(calls) == 3

        # All should have console as receiver
        for call in calls:
            assert call.receiver == "console"
            # console methods are builtins
            assert call.call_type == CallType.BUILTIN

        callee_names = {c.callee_name for c in calls}
        assert "log" in callee_names
        assert "error" in callee_names
        assert "warn" in callee_names

    def test_ts_chained_calls(self, analyzer, parse_ts):
        """Test extraction of chained calls: a().b().c()."""
        source = """class Builder {
    static create(): Builder { return new Builder(); }
    step1(): Builder { return this; }
    step2(): Builder { return this; }
}

function caller() {
    Builder.create().step1().step2();
}
"""
        tree, source_bytes = parse_ts(source)

        func = make_ts_func(
            name="caller",
            qualified_name="caller",
            source="function caller() {\n    Builder.create().step1().step2();\n}",
            start_line=6,
            start_col=0,
            end_line=8,
            end_col=1,
            start_byte=148,
            end_byte=206,
        )

        calls = analyzer.analyze_function(func, tree, source_bytes, "typescript")

        # Should find: create, step1, step2
        assert len(calls) == 3
        callee_names = {c.callee_name for c in calls}
        assert "create" in callee_names
        assert "step1" in callee_names
        assert "step2" in callee_names

        # Check chaining detection
        chained_calls = [c for c in calls if c.is_chained]
        assert len(chained_calls) >= 1

    def test_ts_multiple_calls(self, analyzer, parse_ts):
        """Test extraction of multiple calls in one function."""
        source = """class Service {
    constructor() {}
    getData(): string { return ''; }
}

function helper(x: number): number { return x; }

function caller() {
    console.log("start");
    const svc = new Service();
    const data = svc.getData();
    const result = helper(42);
    console.log("end");
}
"""
        tree, source_bytes = parse_ts(source)

        func = make_ts_func(
            name="caller",
            qualified_name="caller",
            source='function caller() {\n    console.log("start");\n    const svc = new Service();\n    const data = svc.getData();\n    const result = helper(42);\n    console.log("end");\n}',
            start_line=7,
            start_col=0,
            end_line=13,
            end_col=1,
            start_byte=121,
            end_byte=289,
        )

        calls = analyzer.analyze_function(func, tree, source_bytes, "typescript")

        # Should find: 2x log, Service (new), getData, helper
        assert len(calls) >= 5
        callee_names = [c.callee_name for c in calls]
        assert callee_names.count("log") == 2
        assert "Service" in callee_names
        assert "getData" in callee_names
        assert "helper" in callee_names

        # Check call types
        constructor_calls = [c for c in calls if c.call_type == CallType.CONSTRUCTOR]
        assert len(constructor_calls) == 1
        assert constructor_calls[0].callee_name == "Service"
