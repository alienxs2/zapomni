"""Unit tests for CallGraphAnalyzer - Python call extraction.

This module contains comprehensive tests for the CallGraphAnalyzer's Python
call extraction functionality, covering various call types, patterns, and
edge cases.

Test count: 15 tests as per Issue #24 requirements.

Author: Goncharenko Anton aka alienxs2
License: MIT
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
def parse_python():
    """Factory fixture to parse Python source code."""
    parser = get_parser("python")

    def _parse(source: str) -> tuple[Tree, bytes]:
        source_bytes = source.encode("utf-8")
        tree = parser.parse(source_bytes)
        return tree, source_bytes

    return _parse


def create_extracted_function(
    name: str,
    qualified_name: str,
    source: bytes,
    start_line: int,
    end_line: int,
    start_column: int = 0,
    end_column: int = 0,
    element_type: CodeElementType = CodeElementType.FUNCTION,
    parent_class: str | None = None,
    file_path: str = "/test.py",
) -> ExtractedCode:
    """Helper to create an ExtractedCode object for testing."""
    return ExtractedCode(
        name=name,
        qualified_name=qualified_name,
        element_type=element_type,
        language="python",
        file_path=file_path,
        location=ASTNodeLocation(
            start_line=start_line,
            end_line=end_line,
            start_column=start_column,
            end_column=end_column,
            start_byte=0,
            end_byte=len(source),
        ),
        source_code=source.decode("utf-8"),
        parent_class=parent_class,
    )


# =============================================================================
# Test Simple Function Call
# =============================================================================


class TestSimpleFunctionCall:
    """Tests for simple function calls: foo()"""

    def test_simple_function_call(self, analyzer, parse_python):
        """Test extraction of simple function call: foo()"""
        source = """def caller():
    foo()
"""
        tree, source_bytes = parse_python(source)

        func = create_extracted_function(
            name="caller",
            qualified_name="caller",
            source=source_bytes,
            start_line=0,
            end_line=1,
        )

        calls = analyzer.analyze_function(func, tree, source_bytes, "python")

        assert len(calls) == 1
        call = calls[0]
        assert call.callee_name == "foo"
        assert call.call_type == CallType.FUNCTION
        assert call.receiver is None
        assert call.arguments_count == 0
        assert call.is_await is False
        assert call.caller_qualified_name == "caller"


# =============================================================================
# Test Function Call with Arguments
# =============================================================================


class TestFunctionCallWithArgs:
    """Tests for function calls with arguments: foo(a, b, c)"""

    def test_function_call_with_args(self, analyzer, parse_python):
        """Test extraction of function call with positional arguments: foo(a, b, c)"""
        source = """def caller():
    foo(a, b, c)
"""
        tree, source_bytes = parse_python(source)

        func = create_extracted_function(
            name="caller",
            qualified_name="caller",
            source=source_bytes,
            start_line=0,
            end_line=1,
        )

        calls = analyzer.analyze_function(func, tree, source_bytes, "python")

        assert len(calls) == 1
        call = calls[0]
        assert call.callee_name == "foo"
        assert call.call_type == CallType.FUNCTION
        assert call.arguments_count == 3


# =============================================================================
# Test Method Call
# =============================================================================


class TestMethodCall:
    """Tests for method calls: obj.method()"""

    def test_method_call(self, analyzer, parse_python):
        """Test extraction of method call: obj.method()"""
        source = """def caller():
    obj.method()
"""
        tree, source_bytes = parse_python(source)

        func = create_extracted_function(
            name="caller",
            qualified_name="caller",
            source=source_bytes,
            start_line=0,
            end_line=1,
        )

        calls = analyzer.analyze_function(func, tree, source_bytes, "python")

        assert len(calls) == 1
        call = calls[0]
        assert call.callee_name == "method"
        assert call.call_type == CallType.METHOD
        assert call.receiver == "obj"


# =============================================================================
# Test Chained Method Call
# =============================================================================


class TestChainedMethodCall:
    """Tests for chained method calls: obj.foo().bar()"""

    def test_chained_method_call(self, analyzer, parse_python):
        """Test extraction of chained method call: obj.foo().bar()"""
        source = """def caller():
    obj.foo().bar()
"""
        tree, source_bytes = parse_python(source)

        func = create_extracted_function(
            name="caller",
            qualified_name="caller",
            source=source_bytes,
            start_line=0,
            end_line=1,
        )

        calls = analyzer.analyze_function(func, tree, source_bytes, "python")

        # Should detect both calls: foo() and bar()
        assert len(calls) == 2
        call_names = [c.callee_name for c in calls]
        assert "foo" in call_names
        assert "bar" in call_names

        # At least one should be marked as chained
        chained_calls = [c for c in calls if c.is_chained]
        assert len(chained_calls) >= 1


# =============================================================================
# Test Nested Call
# =============================================================================


class TestNestedCall:
    """Tests for nested calls: foo(bar())"""

    def test_nested_call(self, analyzer, parse_python):
        """Test extraction of nested call: foo(bar())"""
        source = """def caller():
    foo(bar())
"""
        tree, source_bytes = parse_python(source)

        func = create_extracted_function(
            name="caller",
            qualified_name="caller",
            source=source_bytes,
            start_line=0,
            end_line=1,
        )

        calls = analyzer.analyze_function(func, tree, source_bytes, "python")

        # Should detect both calls: foo() and bar()
        assert len(calls) == 2
        call_names = [c.callee_name for c in calls]
        assert "foo" in call_names
        assert "bar" in call_names


# =============================================================================
# Test Await Call
# =============================================================================


class TestAwaitCall:
    """Tests for await expressions: await async_func()"""

    def test_await_call(self, analyzer, parse_python):
        """Test extraction of await call: await async_func()"""
        source = """async def caller():
    await async_func()
"""
        tree, source_bytes = parse_python(source)

        func = create_extracted_function(
            name="caller",
            qualified_name="caller",
            source=source_bytes,
            start_line=0,
            end_line=1,
        )

        calls = analyzer.analyze_function(func, tree, source_bytes, "python")

        assert len(calls) == 1
        call = calls[0]
        assert call.callee_name == "async_func"
        assert call.is_await is True


# =============================================================================
# Test Call in Class Method
# =============================================================================


class TestCallInClassMethod:
    """Tests for calls inside class methods"""

    def test_call_in_class_method(self, analyzer, parse_python):
        """Test extraction of calls inside class method"""
        source = """class MyClass:
    def method(self):
        self.other_method()
        helper_func()
"""
        tree, source_bytes = parse_python(source)

        # The method starts at line 1 (0-indexed)
        func = create_extracted_function(
            name="method",
            qualified_name="MyClass.method",
            source=source_bytes,
            start_line=1,
            end_line=3,
            start_column=4,
            element_type=CodeElementType.METHOD,
            parent_class="MyClass",
        )

        calls = analyzer.analyze_function(func, tree, source_bytes, "python")

        assert len(calls) == 2
        call_names = [c.callee_name for c in calls]
        assert "other_method" in call_names
        assert "helper_func" in call_names

        # Check the method call has self as receiver
        method_call = next(c for c in calls if c.callee_name == "other_method")
        assert method_call.call_type == CallType.METHOD
        assert method_call.receiver == "self"


# =============================================================================
# Test Constructor Call
# =============================================================================


class TestConstructorCall:
    """Tests for constructor calls: MyClass()"""

    def test_constructor_call(self, analyzer, parse_python):
        """Test extraction of constructor call: MyClass()"""
        source = """def caller():
    obj = MyClass()
"""
        tree, source_bytes = parse_python(source)

        func = create_extracted_function(
            name="caller",
            qualified_name="caller",
            source=source_bytes,
            start_line=0,
            end_line=1,
        )

        calls = analyzer.analyze_function(func, tree, source_bytes, "python")

        assert len(calls) == 1
        call = calls[0]
        assert call.callee_name == "MyClass"
        # PascalCase names are detected as constructors
        assert call.call_type == CallType.CONSTRUCTOR


# =============================================================================
# Test Builtin Call
# =============================================================================


class TestBuiltinCall:
    """Tests for builtin function calls: print(), len()"""

    def test_builtin_call(self, analyzer, parse_python):
        """Test extraction of builtin calls: print(), len()"""
        source = """def caller():
    print("hello")
    x = len(items)
"""
        tree, source_bytes = parse_python(source)

        func = create_extracted_function(
            name="caller",
            qualified_name="caller",
            source=source_bytes,
            start_line=0,
            end_line=2,
        )

        calls = analyzer.analyze_function(func, tree, source_bytes, "python")

        assert len(calls) == 2
        call_names = [c.callee_name for c in calls]
        assert "print" in call_names
        assert "len" in call_names

        # Check builtin detection
        print_call = next(c for c in calls if c.callee_name == "print")
        assert print_call.call_type == CallType.BUILTIN

        len_call = next(c for c in calls if c.callee_name == "len")
        assert len_call.call_type == CallType.BUILTIN


# =============================================================================
# Test Lambda Ignored
# =============================================================================


class TestLambdaIgnored:
    """Tests for lambda expressions - calls inside lambda should not be from parent"""

    def test_lambda_ignored(self, analyzer, parse_python):
        """Test that calls inside lambda are not attributed to parent function"""
        source = """def caller():
    func = lambda x: inner_call(x)
    outer_call()
"""
        tree, source_bytes = parse_python(source)

        func = create_extracted_function(
            name="caller",
            qualified_name="caller",
            source=source_bytes,
            start_line=0,
            end_line=2,
        )

        calls = analyzer.analyze_function(func, tree, source_bytes, "python")

        # Only outer_call should be detected from caller
        # inner_call is inside lambda which is a boundary
        call_names = [c.callee_name for c in calls]
        assert "outer_call" in call_names
        # inner_call should NOT be in calls from caller (it's inside lambda)
        assert "inner_call" not in call_names


# =============================================================================
# Test Call with Kwargs
# =============================================================================


class TestCallWithKwargs:
    """Tests for function calls with keyword arguments: foo(a=1, b=2)"""

    def test_call_with_kwargs(self, analyzer, parse_python):
        """Test extraction of function call with kwargs: foo(a=1, b=2)"""
        source = """def caller():
    foo(a=1, b=2)
"""
        tree, source_bytes = parse_python(source)

        func = create_extracted_function(
            name="caller",
            qualified_name="caller",
            source=source_bytes,
            start_line=0,
            end_line=1,
        )

        calls = analyzer.analyze_function(func, tree, source_bytes, "python")

        assert len(calls) == 1
        call = calls[0]
        assert call.callee_name == "foo"
        assert call.arguments_count == 2  # Two keyword arguments


# =============================================================================
# Test Call in Comprehension
# =============================================================================


class TestCallInComprehension:
    """Tests for calls inside comprehensions: [f(x) for x in items]"""

    def test_call_in_comprehension(self, analyzer, parse_python):
        """Test extraction of calls inside list comprehension: [f(x) for x in items]"""
        source = """def caller():
    result = [f(x) for x in items]
"""
        tree, source_bytes = parse_python(source)

        func = create_extracted_function(
            name="caller",
            qualified_name="caller",
            source=source_bytes,
            start_line=0,
            end_line=1,
        )

        calls = analyzer.analyze_function(func, tree, source_bytes, "python")

        assert len(calls) == 1
        call = calls[0]
        assert call.callee_name == "f"
        assert call.arguments_count == 1


# =============================================================================
# Test Super Call
# =============================================================================


class TestSuperCall:
    """Tests for super() calls: super().__init__()"""

    def test_super_call(self, analyzer, parse_python):
        """Test extraction of super() call: super().__init__()"""
        source = """class Child(Parent):
    def __init__(self):
        super().__init__()
"""
        tree, source_bytes = parse_python(source)

        # Method starts at line 1
        func = create_extracted_function(
            name="__init__",
            qualified_name="Child.__init__",
            source=source_bytes,
            start_line=1,
            end_line=2,
            start_column=4,
            element_type=CodeElementType.METHOD,
            parent_class="Child",
        )

        calls = analyzer.analyze_function(func, tree, source_bytes, "python")

        # Should detect super() and __init__()
        assert len(calls) == 2
        call_names = [c.callee_name for c in calls]
        assert "super" in call_names
        assert "__init__" in call_names

        # super is a builtin
        super_call = next(c for c in calls if c.callee_name == "super")
        assert super_call.call_type == CallType.BUILTIN

        # __init__ is a method call (chained from super())
        init_call = next(c for c in calls if c.callee_name == "__init__")
        assert init_call.call_type == CallType.METHOD


# =============================================================================
# Test Multiple Calls in Function
# =============================================================================


class TestMultipleCallsInFunction:
    """Tests for functions with multiple calls (5+)"""

    def test_multiple_calls_in_function(self, analyzer, parse_python):
        """Test extraction of multiple calls (5+) in a function"""
        source = """def caller():
    call1()
    call2(a)
    obj.call3()
    call4(x, y)
    result = call5(z)
    call6()
"""
        tree, source_bytes = parse_python(source)

        func = create_extracted_function(
            name="caller",
            qualified_name="caller",
            source=source_bytes,
            start_line=0,
            end_line=6,
        )

        calls = analyzer.analyze_function(func, tree, source_bytes, "python")

        # Should detect all 6 calls
        assert len(calls) == 6
        call_names = [c.callee_name for c in calls]
        assert "call1" in call_names
        assert "call2" in call_names
        assert "call3" in call_names
        assert "call4" in call_names
        assert "call5" in call_names
        assert "call6" in call_names


# =============================================================================
# Test No Calls Empty Function
# =============================================================================


class TestNoCallsEmptyFunction:
    """Tests for functions with no calls"""

    def test_no_calls_empty_function(self, analyzer, parse_python):
        """Test extraction from function with no calls"""
        source = """def caller():
    x = 1
    y = 2
    return x + y
"""
        tree, source_bytes = parse_python(source)

        func = create_extracted_function(
            name="caller",
            qualified_name="caller",
            source=source_bytes,
            start_line=0,
            end_line=3,
        )

        calls = analyzer.analyze_function(func, tree, source_bytes, "python")

        # No calls should be detected
        assert len(calls) == 0
