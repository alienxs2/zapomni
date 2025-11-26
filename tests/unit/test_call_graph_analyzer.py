"""
Unit tests for the CallGraphAnalyzer component.

Tests function call relationship analysis, import resolution, and graph building.
Covers both simple dict graphs and optional NetworkX graph generation.
"""

from __future__ import annotations

import ast
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, List

import pytest

from zapomni_core.code.call_graph_analyzer import (
    CallGraphAnalyzer,
    FunctionCall,
    FunctionDef,
    ImportMapping,
)
from zapomni_core.exceptions import ValidationError

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def analyzer() -> CallGraphAnalyzer:
    """Create a fresh CallGraphAnalyzer instance."""
    return CallGraphAnalyzer()


@pytest.fixture
def temp_py_file():
    """Create a temporary Python file."""
    with NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        yield f
    # Cleanup happens automatically with delete=False and manual deletion in tests


# ============================================================================
# Test Classes
# ============================================================================


class TestCallGraphAnalyzerInitialization:
    """Test suite for CallGraphAnalyzer initialization."""

    def test_init_creates_fresh_analyzer(self) -> None:
        """Should initialize with empty state."""
        analyzer = CallGraphAnalyzer()

        assert analyzer.file_path is None
        assert analyzer.ast_tree is None
        assert analyzer.functions == {}
        assert analyzer.calls == []
        assert analyzer.imports == {}
        assert analyzer.call_graph == {}
        assert analyzer.networkx_graph is None

    def test_init_is_reusable(self) -> None:
        """Should be able to use analyzer for multiple files."""
        analyzer = CallGraphAnalyzer()

        # Analyzer should be ready for use
        assert callable(analyzer.analyze_file)
        assert callable(analyzer.find_function_calls)


class TestAnalyzeFile:
    """Test suite for analyze_file method."""

    def test_analyze_simple_python_file(self, analyzer: CallGraphAnalyzer) -> None:
        """Should analyze a simple Python file."""
        code = """
def func_a():
    pass

def func_b():
    func_a()
"""
        with NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()

            graph = analyzer.analyze_file(f.name)

            assert isinstance(graph, dict)
            assert "func_a" in graph
            assert "func_b" in graph
            Path(f.name).unlink()

    def test_analyze_file_with_multiple_calls(self, analyzer: CallGraphAnalyzer) -> None:
        """Should capture multiple function calls."""
        code = """
def func_a():
    pass

def func_b():
    func_a()

def func_c():
    func_a()
    func_b()
"""
        with NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()

            graph = analyzer.analyze_file(f.name)

            assert graph["func_c"] == ["func_a", "func_b"]
            Path(f.name).unlink()

    def test_analyze_file_with_provided_ast(self, analyzer: CallGraphAnalyzer) -> None:
        """Should accept pre-parsed AST instead of parsing file."""
        code = """
def test_func():
    pass

def caller():
    test_func()
"""
        ast_tree = ast.parse(code)

        with NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()

            graph = analyzer.analyze_file(f.name, ast_tree=ast_tree)

            assert "caller" in graph
            assert "test_func" in graph["caller"]
            Path(f.name).unlink()

    def test_analyze_file_returns_dict_graph(self, analyzer: CallGraphAnalyzer) -> None:
        """Should return call graph as dictionary."""
        code = "def func(): pass"

        with NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()

            result = analyzer.analyze_file(f.name)

            assert isinstance(result, dict)
            assert result == {"func": []}
            Path(f.name).unlink()

    def test_analyze_file_raises_error_for_nonexistent_file(
        self, analyzer: CallGraphAnalyzer
    ) -> None:
        """Should raise ValidationError for nonexistent file."""
        with pytest.raises(ValidationError) as exc_info:
            analyzer.analyze_file("/nonexistent/path/file.py")

        assert exc_info.value.error_code == "VAL_001"
        assert "not found" in exc_info.value.message.lower()

    def test_analyze_file_raises_error_for_non_python_file(
        self, analyzer: CallGraphAnalyzer
    ) -> None:
        """Should raise ValidationError for non-.py files."""
        with NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"not python")
            f.flush()

            with pytest.raises(ValidationError) as exc_info:
                analyzer.analyze_file(f.name)

            assert exc_info.value.error_code == "VAL_002"
            assert ".py" in exc_info.value.message.lower()
            Path(f.name).unlink()

    def test_analyze_file_raises_error_for_invalid_python(
        self, analyzer: CallGraphAnalyzer
    ) -> None:
        """Should raise SyntaxError for invalid Python code."""
        code = "def func(: pass"  # Invalid syntax

        with NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()

            with pytest.raises(SyntaxError):
                analyzer.analyze_file(f.name)

            Path(f.name).unlink()

    def test_analyze_file_with_path_object(self, analyzer: CallGraphAnalyzer) -> None:
        """Should accept Path objects in addition to strings."""
        code = "def func(): pass"

        with NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()

            path_obj = Path(f.name)
            graph = analyzer.analyze_file(path_obj)

            assert isinstance(graph, dict)
            path_obj.unlink()


class TestFindFunctionCalls:
    """Test suite for find_function_calls method."""

    def test_find_simple_function_call(self, analyzer: CallGraphAnalyzer) -> None:
        """Should find simple function calls."""
        code = """
def func_a():
    pass

def func_b():
    func_a()
"""
        tree = ast.parse(code)
        calls = analyzer.find_function_calls(tree)

        assert len(calls) > 0
        assert any(c.callee == "func_a" for c in calls)

    def test_find_multiple_calls_in_function(self, analyzer: CallGraphAnalyzer) -> None:
        """Should find multiple calls within a single function."""
        code = """
def a(): pass
def b(): pass
def c():
    a()
    b()
"""
        tree = ast.parse(code)
        calls = analyzer.find_function_calls(tree)

        c_calls = [c for c in calls if c.caller == "c"]
        assert len(c_calls) >= 2

    def test_find_calls_returns_function_call_objects(self, analyzer: CallGraphAnalyzer) -> None:
        """Should return FunctionCall objects with required attributes."""
        code = """
def helper():
    pass

def main():
    helper()
"""
        tree = ast.parse(code)
        calls = analyzer.find_function_calls(tree)

        for call in calls:
            assert isinstance(call, FunctionCall)
            assert hasattr(call, "caller")
            assert hasattr(call, "callee")
            assert hasattr(call, "lineno")
            assert isinstance(call.lineno, int)

    def test_find_nested_function_calls(self, analyzer: CallGraphAnalyzer) -> None:
        """Should find calls in nested functions."""
        code = """
def outer():
    def inner():
        pass
    inner()
"""
        tree = ast.parse(code)
        calls = analyzer.find_function_calls(tree)

        assert any(c.caller == "outer" and c.callee == "inner" for c in calls)

    def test_find_method_calls(self, analyzer: CallGraphAnalyzer) -> None:
        """Should identify method calls (is_method=True)."""
        code = """
def main():
    obj.method()
"""
        tree = ast.parse(code)
        calls = analyzer.find_function_calls(tree)

        method_calls = [c for c in calls if c.is_method]
        assert len(method_calls) > 0
        assert any(c.callee == "method" for c in method_calls)

    def test_find_calls_raises_error_for_non_ast_node(self, analyzer: CallGraphAnalyzer) -> None:
        """Should raise ValidationError for non-AST input."""
        with pytest.raises(ValidationError) as exc_info:
            analyzer.find_function_calls("not an ast node")

        assert exc_info.value.error_code == "VAL_003"


class TestResolveImports:
    """Test suite for resolve_imports method."""

    def test_resolve_simple_import(self, analyzer: CallGraphAnalyzer) -> None:
        """Should resolve simple import statements."""
        code = "import os"
        tree = ast.parse(code)

        imports = analyzer.resolve_imports(tree)

        assert "os" in imports
        assert imports["os"].module_name == "os"
        assert imports["os"].alias == "os"

    def test_resolve_import_with_alias(self, analyzer: CallGraphAnalyzer) -> None:
        """Should resolve import with 'as' alias."""
        code = "import numpy as np"
        tree = ast.parse(code)

        imports = analyzer.resolve_imports(tree)

        assert "np" in imports
        assert imports["np"].module_name == "numpy"
        assert imports["np"].alias == "np"

    def test_resolve_from_import(self, analyzer: CallGraphAnalyzer) -> None:
        """Should resolve from-import statements."""
        code = "from os import path"
        tree = ast.parse(code)

        imports = analyzer.resolve_imports(tree)

        assert "path" in imports
        assert imports["path"].module_name == "os"
        assert imports["path"].is_from_import is True

    def test_resolve_from_import_with_alias(self, analyzer: CallGraphAnalyzer) -> None:
        """Should resolve from-import with alias."""
        code = "from os.path import join as path_join"
        tree = ast.parse(code)

        imports = analyzer.resolve_imports(tree)

        assert "path_join" in imports
        assert imports["path_join"].module_name == "os.path"
        assert imports["path_join"].is_from_import is True

    def test_resolve_multiple_imports(self, analyzer: CallGraphAnalyzer) -> None:
        """Should resolve multiple import statements."""
        code = """
import os
import sys
from pathlib import Path
"""
        tree = ast.parse(code)

        imports = analyzer.resolve_imports(tree)

        assert "os" in imports
        assert "sys" in imports
        assert "Path" in imports
        assert len(imports) >= 3

    def test_resolve_imports_returns_import_mapping(self, analyzer: CallGraphAnalyzer) -> None:
        """Should return ImportMapping objects."""
        code = "import json"
        tree = ast.parse(code)

        imports = analyzer.resolve_imports(tree)

        for mapping in imports.values():
            assert isinstance(mapping, ImportMapping)
            assert hasattr(mapping, "module_name")
            assert hasattr(mapping, "alias")
            assert hasattr(mapping, "is_from_import")

    def test_resolve_imports_raises_error_for_non_module(self, analyzer: CallGraphAnalyzer) -> None:
        """Should raise ValidationError for non-Module input."""
        with pytest.raises(ValidationError) as exc_info:
            analyzer.resolve_imports("not a module")  # type: ignore

        assert exc_info.value.error_code == "VAL_003"


class TestBuildDependencyGraph:
    """Test suite for build_dependency_graph method."""

    def test_build_simple_graph(self, analyzer: CallGraphAnalyzer) -> None:
        """Should build simple dependency graph."""
        functions = {
            "func_a": FunctionDef(name="func_a", lineno=1, col_offset=0),
            "func_b": FunctionDef(name="func_b", lineno=3, col_offset=0),
        }
        calls = [
            FunctionCall(caller="func_b", callee="func_a", lineno=4),
        ]

        graph = analyzer.build_dependency_graph(functions, calls)

        assert graph["func_a"] == []
        assert graph["func_b"] == ["func_a"]

    def test_build_graph_with_multiple_calls(self, analyzer: CallGraphAnalyzer) -> None:
        """Should handle functions with multiple callees."""
        functions = {
            "a": FunctionDef(name="a", lineno=1, col_offset=0),
            "b": FunctionDef(name="b", lineno=3, col_offset=0),
            "c": FunctionDef(name="c", lineno=5, col_offset=0),
        }
        calls = [
            FunctionCall(caller="c", callee="a", lineno=6),
            FunctionCall(caller="c", callee="b", lineno=7),
        ]

        graph = analyzer.build_dependency_graph(functions, calls)

        assert set(graph["c"]) == {"a", "b"}

    def test_build_graph_avoids_duplicate_edges(self, analyzer: CallGraphAnalyzer) -> None:
        """Should not create duplicate edges for repeated calls."""
        functions = {
            "a": FunctionDef(name="a", lineno=1, col_offset=0),
            "b": FunctionDef(name="b", lineno=3, col_offset=0),
        }
        calls = [
            FunctionCall(caller="b", callee="a", lineno=4),
            FunctionCall(caller="b", callee="a", lineno=5),  # Duplicate
        ]

        graph = analyzer.build_dependency_graph(functions, calls)

        # Should have "a" in callees only once
        assert graph["b"].count("a") == 1

    def test_build_graph_sorts_callees(self, analyzer: CallGraphAnalyzer) -> None:
        """Should return sorted lists of callees."""
        functions = {
            "a": FunctionDef(name="a", lineno=1, col_offset=0),
            "b": FunctionDef(name="b", lineno=3, col_offset=0),
            "c": FunctionDef(name="c", lineno=5, col_offset=0),
            "main": FunctionDef(name="main", lineno=7, col_offset=0),
        }
        calls = [
            FunctionCall(caller="main", callee="c", lineno=8),
            FunctionCall(caller="main", callee="a", lineno=9),
            FunctionCall(caller="main", callee="b", lineno=10),
        ]

        graph = analyzer.build_dependency_graph(functions, calls)

        assert graph["main"] == ["a", "b", "c"]  # Sorted

    def test_build_graph_initializes_all_functions(self, analyzer: CallGraphAnalyzer) -> None:
        """Should initialize all functions with empty lists."""
        functions = {
            "a": FunctionDef(name="a", lineno=1, col_offset=0),
            "b": FunctionDef(name="b", lineno=3, col_offset=0),
            "c": FunctionDef(name="c", lineno=5, col_offset=0),
        }
        calls: List[FunctionCall] = []

        graph = analyzer.build_dependency_graph(functions, calls)

        assert set(graph.keys()) == {"a", "b", "c"}
        assert all(graph[f] == [] for f in graph)


class TestCompleteWorkflow:
    """Test suite for complete analysis workflow."""

    def test_full_analysis_workflow(self, analyzer: CallGraphAnalyzer) -> None:
        """Should complete full analysis workflow on real file."""
        code = """
def helper():
    return 42

def process(data):
    result = helper()
    return result + data

def main():
    x = process(10)
    return x
"""
        with NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()

            graph = analyzer.analyze_file(f.name)

            # Verify structure
            assert "helper" in graph
            assert "process" in graph
            assert "main" in graph

            # Verify relationships
            assert "helper" in graph["process"]
            assert "process" in graph["main"]

            # Verify no call to undefined functions (exclude builtin functions)
            builtin_functions = {"print", "len", "range", "enumerate", "map", "filter", "zip"}
            for func, callees in graph.items():
                for callee in callees:
                    is_defined = callee in graph
                    is_imported = callee in analyzer.imports
                    is_builtin = callee in builtin_functions
                    assert (
                        is_defined or is_imported or is_builtin
                    ), f"{callee} called by {func} but not defined, imported, or builtin"

            Path(f.name).unlink()

    def test_analysis_with_imports(self, analyzer: CallGraphAnalyzer) -> None:
        """Should handle analysis with import statements."""
        code = """
import os
from pathlib import Path

def get_home():
    return os.path.expanduser('~')

def create_dir(path):
    p = Path(path)
    return p
"""
        with NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()

            graph = analyzer.analyze_file(f.name)

            assert "os" in analyzer.imports
            assert "Path" in analyzer.imports
            assert len(graph) >= 2

            Path(f.name).unlink()

    def test_analysis_with_nested_functions(self, analyzer: CallGraphAnalyzer) -> None:
        """Should handle nested function definitions."""
        code = """
def outer():
    def inner():
        return 42
    return inner()

outer()
"""
        with NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()

            graph = analyzer.analyze_file(f.name)

            assert "outer" in graph
            Path(f.name).unlink()


class TestGetterMethods:
    """Test suite for getter methods."""

    def test_get_call_graph(self, analyzer: CallGraphAnalyzer) -> None:
        """Should return current call graph."""
        code = """
def a(): pass
def b(): a()
"""
        with NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()

            analyzer.analyze_file(f.name)
            graph = analyzer.get_call_graph()

            assert isinstance(graph, dict)
            assert "a" in graph
            assert "b" in graph
            Path(f.name).unlink()

    def test_get_callees_of(self, analyzer: CallGraphAnalyzer) -> None:
        """Should return callees of a function."""
        code = """
def a(): pass
def b(): pass
def c():
    a()
    b()
"""
        with NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()

            analyzer.analyze_file(f.name)
            callees = analyzer.get_callees_of("c")

            assert set(callees) == {"a", "b"}
            Path(f.name).unlink()

    def test_get_callers_of(self, analyzer: CallGraphAnalyzer) -> None:
        """Should return functions that call a given function."""
        code = """
def target(): pass
def caller1(): target()
def caller2(): target()
"""
        with NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()

            analyzer.analyze_file(f.name)
            callers = analyzer.get_callers_of("target")

            assert set(callers) == {"caller1", "caller2"}
            Path(f.name).unlink()

    def test_get_callees_of_returns_copy(self, analyzer: CallGraphAnalyzer) -> None:
        """Should return copy of callees list (not reference)."""
        code = """
def a(): pass
def b(): a()
"""
        with NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()

            analyzer.analyze_file(f.name)
            callees = analyzer.get_callees_of("b")
            callees.append("modified")

            # Original should not be modified
            original_callees = analyzer.get_callees_of("b")
            assert "modified" not in original_callees
            Path(f.name).unlink()


class TestEdgeCases:
    """Test suite for edge cases and special scenarios."""

    def test_empty_file(self, analyzer: CallGraphAnalyzer) -> None:
        """Should handle empty Python files."""
        code = ""

        with NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()

            graph = analyzer.analyze_file(f.name)

            assert graph == {}
            Path(f.name).unlink()

    def test_file_with_only_comments(self, analyzer: CallGraphAnalyzer) -> None:
        """Should handle files with only comments."""
        code = """
# This is a comment
# Another comment
"""
        with NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()

            graph = analyzer.analyze_file(f.name)

            assert graph == {}
            Path(f.name).unlink()

    def test_file_with_lambda_functions(self, analyzer: CallGraphAnalyzer) -> None:
        """Should handle lambda expressions."""
        code = """
def map_values(data):
    result = list(map(lambda x: x * 2, data))
    return result
"""
        with NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()

            graph = analyzer.analyze_file(f.name)

            assert "map_values" in graph
            Path(f.name).unlink()

    def test_file_with_async_functions(self, analyzer: CallGraphAnalyzer) -> None:
        """Should handle async function definitions."""
        code = """
async def fetch_data():
    pass

async def process():
    await fetch_data()
"""
        with NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()

            graph = analyzer.analyze_file(f.name)

            assert "fetch_data" in graph
            assert "process" in graph
            Path(f.name).unlink()

    def test_file_with_class_methods(self, analyzer: CallGraphAnalyzer) -> None:
        """Should handle class methods."""
        code = """
class MyClass:
    def method_a(self):
        pass

    def method_b(self):
        self.method_a()
"""
        with NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()

            graph = analyzer.analyze_file(f.name)

            assert "method_a" in graph
            assert "method_b" in graph
            Path(f.name).unlink()
