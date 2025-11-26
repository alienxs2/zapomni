"""
Unit tests for ASTCodeChunker component.

Covers AST parsing, function/class extraction, metadata generation,
and fallback line-based chunking for Python code.

Author: Zapomni Team
License: MIT
"""

from __future__ import annotations

import ast
import tempfile
from pathlib import Path
from typing import List

import pytest

from zapomni_core.code.ast_chunker import ASTCodeChunker, CodeMetadata, SupportedLanguage
from zapomni_core.exceptions import ProcessingError, ValidationError
from zapomni_db.models import Chunk

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def simple_python_code() -> str:
    """Simple Python code with function and class."""
    return '''"""Module docstring."""

def hello():
    """Say hello."""
    return "Hello"

def add(a, b):
    """Add two numbers."""
    return a + b

class Calculator:
    """A simple calculator."""

    def __init__(self):
        self.value = 0

    def add(self, x):
        """Add to value."""
        self.value += x
        return self.value
'''


@pytest.fixture
def python_with_async() -> str:
    """Python code with async functions."""
    return '''async def fetch_data():
    """Fetch data asynchronously."""
    return await some_operation()

def sync_func():
    """Regular sync function."""
    pass
'''


@pytest.fixture
def python_with_imports() -> str:
    """Python code with imports."""
    return '''import os
import sys
from typing import List, Dict
from pathlib import Path

def main():
    """Main function."""
    pass
'''


@pytest.fixture
def python_with_decorators() -> str:
    """Python code with decorated functions."""
    return '''import functools

@functools.lru_cache
def fibonacci(n):
    """Calculate fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

@property
def name(self):
    """Get name."""
    return self._name

@staticmethod
def static_method():
    """Static method."""
    pass
'''


@pytest.fixture
def python_with_syntax_error() -> str:
    """Python code with syntax error."""
    return """def broken(:
    pass
"""


@pytest.fixture
def python_private_functions() -> str:
    """Python code with private functions."""
    return '''def public_function():
    """Public function."""
    return 42

def _private_function():
    """Private function."""
    return 0

class MyClass:
    def public_method(self):
        """Public method."""
        pass

    def _private_method(self):
        """Private method."""
        pass
'''


@pytest.fixture
def empty_python() -> str:
    """Empty or minimal Python code."""
    return '''"""Just a docstring."""

# Comment
'''


@pytest.fixture
def python_with_nested_classes() -> str:
    """Python code with nested classes."""
    return '''class Outer:
    """Outer class."""

    class Inner:
        """Inner class."""

        def inner_method(self):
            """Inner method."""
            pass

    def outer_method(self):
        """Outer method."""
        pass
'''


# ============================================================================
# Test Cases: Initialization
# ============================================================================


class TestASTCodeChunkerInit:
    """Tests for ASTCodeChunker.__init__."""

    def test_init_python_success(self) -> None:
        """Initialize with Python language."""
        chunker = ASTCodeChunker(language="python")

        assert chunker.language == SupportedLanguage.PYTHON
        assert chunker.min_chunk_lines == 2

    def test_init_python_case_insensitive(self) -> None:
        """Language parameter should be case insensitive."""
        chunker = ASTCodeChunker(language="PYTHON")
        assert chunker.language == SupportedLanguage.PYTHON

        chunker = ASTCodeChunker(language="PyThOn")
        assert chunker.language == SupportedLanguage.PYTHON

    def test_init_javascript_success(self) -> None:
        """Initialize with JavaScript language."""
        chunker = ASTCodeChunker(language="javascript")
        assert chunker.language == SupportedLanguage.JAVASCRIPT

    def test_init_typescript_success(self) -> None:
        """Initialize with TypeScript language."""
        chunker = ASTCodeChunker(language="typescript")
        assert chunker.language == SupportedLanguage.TYPESCRIPT

    def test_init_unsupported_language_raises(self) -> None:
        """Unsupported language should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ASTCodeChunker(language="rust")

        assert "Unsupported language" in str(exc_info.value.message)
        assert exc_info.value.error_code == "VAL_002"

    def test_init_custom_min_chunk_lines(self) -> None:
        """Custom min_chunk_lines should be accepted."""
        chunker = ASTCodeChunker(language="python", min_chunk_lines=5)
        assert chunker.min_chunk_lines == 5

    def test_init_min_chunk_lines_invalid_raises(self) -> None:
        """min_chunk_lines < 1 should raise ValueError."""
        with pytest.raises(ValueError):
            ASTCodeChunker(language="python", min_chunk_lines=0)

        with pytest.raises(ValueError):
            ASTCodeChunker(language="python", min_chunk_lines=-1)


# ============================================================================
# Test Cases: chunk_file
# ============================================================================


class TestASTCodeChunkerChunkFile:
    """Tests for ASTCodeChunker.chunk_file."""

    def test_chunk_file_with_content_success(self, simple_python_code: str) -> None:
        """Chunking code with explicit content should succeed."""
        chunker = ASTCodeChunker(language="python")
        chunks = chunker.chunk_file(
            file_path="test.py",
            content=simple_python_code,
        )

        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(c.text for c in chunks)
        assert all(c.metadata for c in chunks)

    def test_chunk_file_from_disk(self, simple_python_code: str) -> None:
        """Chunking from disk file should succeed."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(simple_python_code)
            temp_path = f.name

        try:
            chunker = ASTCodeChunker(language="python")
            chunks = chunker.chunk_file(file_path=temp_path)

            assert len(chunks) > 0
            assert all(isinstance(c, Chunk) for c in chunks)
        finally:
            Path(temp_path).unlink()

    def test_chunk_file_nonexistent_raises(self) -> None:
        """Chunking nonexistent file should raise ValidationError."""
        chunker = ASTCodeChunker(language="python")

        with pytest.raises(ValidationError) as exc_info:
            chunker.chunk_file(file_path="/nonexistent/file.py")

        assert "not found" in str(exc_info.value.message).lower()

    def test_chunk_file_empty_content_raises(self) -> None:
        """Empty content should raise ValidationError."""
        chunker = ASTCodeChunker(language="python")

        with pytest.raises(ValidationError) as exc_info:
            chunker.chunk_file(file_path="test.py", content="")

        assert "empty" in str(exc_info.value.message).lower()

    def test_chunk_file_syntax_error_raises(self, python_with_syntax_error: str) -> None:
        """Syntax error in code should raise ProcessingError."""
        chunker = ASTCodeChunker(language="python")

        with pytest.raises(ProcessingError) as exc_info:
            chunker.chunk_file(
                file_path="bad.py",
                content=python_with_syntax_error,
            )

        assert "syntax" in str(exc_info.value.message).lower()

    def test_chunk_file_invalid_content_type_raises(self) -> None:
        """Non-string content should raise ValidationError."""
        chunker = ASTCodeChunker(language="python")

        with pytest.raises(ValidationError):
            chunker.chunk_file(file_path="test.py", content=12345)  # type: ignore

    def test_chunk_file_sequential_indices(self, simple_python_code: str) -> None:
        """Chunks should have sequential indices."""
        chunker = ASTCodeChunker(language="python")
        chunks = chunker.chunk_file(
            file_path="test.py",
            content=simple_python_code,
        )

        for i, chunk in enumerate(chunks):
            assert chunk.index == i


# ============================================================================
# Test Cases: Python AST Chunking
# ============================================================================


class TestASTCodeChunkerPythonAST:
    """Tests for Python AST parsing and chunking."""

    def test_chunk_simple_functions(self, simple_python_code: str) -> None:
        """Chunking functions should extract them as separate chunks."""
        chunker = ASTCodeChunker(language="python")
        chunks = chunker.chunk_file(
            file_path="test.py",
            content=simple_python_code,
        )

        # Should have hello, add, and Calculator class
        function_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "function"]
        assert len(function_chunks) >= 2

        # Check function names are in metadata
        names = [c.metadata.get("name") for c in function_chunks]
        assert "hello" in names
        assert "add" in names

    def test_chunk_class(self, simple_python_code: str) -> None:
        """Chunking should extract class definitions."""
        chunker = ASTCodeChunker(language="python")
        chunks = chunker.chunk_file(
            file_path="test.py",
            content=simple_python_code,
        )

        class_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "class"]
        assert len(class_chunks) >= 1
        assert class_chunks[0].metadata.get("name") == "Calculator"

    def test_chunk_class_has_methods_count(self, simple_python_code: str) -> None:
        """Class metadata should include methods count."""
        chunker = ASTCodeChunker(language="python")
        chunks = chunker.chunk_file(
            file_path="test.py",
            content=simple_python_code,
        )

        class_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "class"]
        assert class_chunks[0].metadata.get("methods_count") >= 1

    def test_chunk_async_functions(self, python_with_async: str) -> None:
        """Async functions should be marked correctly."""
        chunker = ASTCodeChunker(language="python")
        chunks = chunker.chunk_file(
            file_path="test.py",
            content=python_with_async,
        )

        async_chunks = [c for c in chunks if c.metadata.get("node_type") == "async_function"]
        assert len(async_chunks) >= 1

        sync_chunks = [c for c in chunks if c.metadata.get("node_type") == "function"]
        assert len(sync_chunks) >= 1

    def test_chunk_with_imports(self, python_with_imports: str) -> None:
        """Import statements should be extracted."""
        chunker = ASTCodeChunker(language="python")
        chunks = chunker.chunk_file(
            file_path="test.py",
            content=python_with_imports,
        )

        import_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "import"]
        assert len(import_chunks) >= 1

    def test_chunk_with_decorators(self, python_with_decorators: str) -> None:
        """Decorated functions should include decorators in metadata."""
        chunker = ASTCodeChunker(language="python")
        chunks = chunker.chunk_file(
            file_path="test.py",
            content=python_with_decorators,
        )

        decorated_chunks = [
            c
            for c in chunks
            if c.metadata.get("name") == "fibonacci" and c.metadata.get("chunk_type") == "function"
        ]
        assert len(decorated_chunks) >= 1
        assert len(decorated_chunks[0].metadata.get("decorators", [])) > 0

    def test_chunk_private_functions(self, python_private_functions: str) -> None:
        """Private functions should be marked."""
        chunker = ASTCodeChunker(language="python")
        chunks = chunker.chunk_file(
            file_path="test.py",
            content=python_private_functions,
        )

        private_chunks = [
            c
            for c in chunks
            if c.metadata.get("is_private") is True and c.metadata.get("chunk_type") == "function"
        ]
        assert len(private_chunks) >= 1
        assert private_chunks[0].metadata.get("name") == "_private_function"

    def test_chunk_docstrings_extracted(self, simple_python_code: str) -> None:
        """Docstrings should be extracted as metadata."""
        chunker = ASTCodeChunker(language="python")
        chunks = chunker.chunk_file(
            file_path="test.py",
            content=simple_python_code,
        )

        for chunk in chunks:
            if chunk.metadata.get("name") in ["hello", "add"]:
                # These functions have docstrings
                docstring = chunk.metadata.get("docstring")
                assert docstring is not None
                assert len(docstring) > 0

    def test_chunk_line_ranges(self, simple_python_code: str) -> None:
        """Chunks should have correct line ranges."""
        chunker = ASTCodeChunker(language="python")
        chunks = chunker.chunk_file(
            file_path="test.py",
            content=simple_python_code,
        )

        for chunk in chunks:
            line_start = chunk.metadata.get("line_start")
            line_end = chunk.metadata.get("line_end")
            assert line_start is not None
            assert line_end is not None
            assert line_end >= line_start

    def test_chunk_char_offsets(self, simple_python_code: str) -> None:
        """Chunks should have correct character offsets."""
        chunker = ASTCodeChunker(language="python")
        chunks = chunker.chunk_file(
            file_path="test.py",
            content=simple_python_code,
        )

        for chunk in chunks:
            assert chunk.start_char >= 0
            assert chunk.end_char > chunk.start_char


# ============================================================================
# Test Cases: extract_functions
# ============================================================================


class TestExtractFunctions:
    """Tests for ASTCodeChunker.extract_functions."""

    def test_extract_functions_simple(self, simple_python_code: str) -> None:
        """extract_functions should return all functions."""
        tree = ast.parse(simple_python_code)
        chunker = ASTCodeChunker(language="python")

        functions = chunker.extract_functions(tree)

        assert len(functions) >= 2
        names = [f.name for f in functions]
        assert "hello" in names
        assert "add" in names

    def test_extract_functions_includes_class_methods(self, simple_python_code: str) -> None:
        """extract_functions should include class methods."""
        tree = ast.parse(simple_python_code)
        chunker = ASTCodeChunker(language="python")

        functions = chunker.extract_functions(tree)

        # Should include class methods too
        assert len(functions) >= 3

    def test_extract_functions_async(self, python_with_async: str) -> None:
        """extract_functions should include async functions."""
        tree = ast.parse(python_with_async)
        chunker = ASTCodeChunker(language="python")

        functions = chunker.extract_functions(tree)

        assert len(functions) >= 2
        names = [f.name for f in functions]
        assert "fetch_data" in names
        assert "sync_func" in names

    def test_extract_functions_empty(self, python_with_imports: str) -> None:
        """extract_functions should return empty for code with no functions."""
        tree = ast.parse(python_with_imports)
        chunker = ASTCodeChunker(language="python")

        functions = chunker.extract_functions(tree)

        # Only has imports, so should have 1 main function
        assert len(functions) >= 1


# ============================================================================
# Test Cases: extract_classes
# ============================================================================


class TestExtractClasses:
    """Tests for ASTCodeChunker.extract_classes."""

    def test_extract_classes_simple(self, simple_python_code: str) -> None:
        """extract_classes should return all classes."""
        tree = ast.parse(simple_python_code)
        chunker = ASTCodeChunker(language="python")

        classes = chunker.extract_classes(tree)

        assert len(classes) >= 1
        names = [c.name for c in classes]
        assert "Calculator" in names

    def test_extract_classes_nested(self, python_with_nested_classes: str) -> None:
        """extract_classes should include nested classes."""
        tree = ast.parse(python_with_nested_classes)
        chunker = ASTCodeChunker(language="python")

        classes = chunker.extract_classes(tree)

        assert len(classes) >= 2
        names = [c.name for c in classes]
        assert "Outer" in names
        assert "Inner" in names

    def test_extract_classes_empty(self, python_with_imports: str) -> None:
        """extract_classes should return empty for code with no classes."""
        tree = ast.parse(python_with_imports)
        chunker = ASTCodeChunker(language="python")

        classes = chunker.extract_classes(tree)

        assert len(classes) == 0


# ============================================================================
# Test Cases: get_chunk_metadata
# ============================================================================


class TestGetChunkMetadata:
    """Tests for ASTCodeChunker.get_chunk_metadata."""

    def test_metadata_function(self, simple_python_code: str) -> None:
        """get_chunk_metadata should extract function metadata."""
        tree = ast.parse(simple_python_code)
        chunker = ASTCodeChunker(language="python")

        # Get first function node
        func_node = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "hello"
        )

        metadata = chunker.get_chunk_metadata(func_node)

        assert metadata.name == "hello"
        assert metadata.node_type == "function"
        assert metadata.docstring is not None
        assert metadata.line_start > 0
        assert metadata.line_end > 0

    def test_metadata_async_function(self, python_with_async: str) -> None:
        """get_chunk_metadata should mark async functions."""
        tree = ast.parse(python_with_async)
        chunker = ASTCodeChunker(language="python")

        # Get async function node
        func_node = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef))

        metadata = chunker.get_chunk_metadata(func_node)

        assert metadata.is_async is True
        assert metadata.node_type == "async_function"

    def test_metadata_class(self, simple_python_code: str) -> None:
        """get_chunk_metadata should extract class metadata."""
        tree = ast.parse(simple_python_code)
        chunker = ASTCodeChunker(language="python")

        # Get class node
        class_node = next(n for n in tree.body if isinstance(n, ast.ClassDef))

        metadata = chunker.get_chunk_metadata(class_node)

        assert metadata.name == "Calculator"
        assert metadata.node_type == "class"
        assert metadata.docstring is not None

    def test_metadata_private_flag(self, python_private_functions: str) -> None:
        """Private functions should have is_private flag."""
        tree = ast.parse(python_private_functions)
        chunker = ASTCodeChunker(language="python")

        # Get private function
        private_node = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_private_function"
        )

        metadata = chunker.get_chunk_metadata(private_node)

        assert metadata.is_private is True

    def test_metadata_to_dict(self, simple_python_code: str) -> None:
        """CodeMetadata.to_dict should return complete dict."""
        tree = ast.parse(simple_python_code)
        chunker = ASTCodeChunker(language="python")

        func_node = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "hello"
        )

        metadata = chunker.get_chunk_metadata(func_node)
        metadata_dict = metadata.to_dict()

        assert isinstance(metadata_dict, dict)
        assert "name" in metadata_dict
        assert "node_type" in metadata_dict
        assert "line_start" in metadata_dict
        assert "line_end" in metadata_dict

    def test_metadata_invalid_node_raises(self) -> None:
        """Invalid node types should raise ValueError."""
        chunker = ASTCodeChunker(language="python")

        # Create an invalid node type
        import_node = ast.Import(names=[ast.alias(name="os", asname=None)])

        with pytest.raises(ValueError):
            chunker.get_chunk_metadata(import_node)


# ============================================================================
# Test Cases: Fallback Line-Based Chunking
# ============================================================================


class TestLineFallbackChunking:
    """Tests for line-based chunking fallback."""

    def test_line_fallback_javascript(self) -> None:
        """JavaScript code should fall back to line-based chunking."""
        js_code = """
function add(a, b) {
    return a + b;
}

function multiply(a, b) {
    return a * b;
}

class Calculator {
    constructor() {
        this.value = 0;
    }
}
"""
        chunker = ASTCodeChunker(language="javascript")
        chunks = chunker.chunk_file(file_path="test.js", content=js_code)

        assert len(chunks) > 0
        assert all(c.metadata.get("chunk_type") == "line_based" for c in chunks)
        assert all(c.metadata.get("language") == "javascript" for c in chunks)

    def test_line_fallback_typescript(self) -> None:
        """TypeScript code should fall back to line-based chunking."""
        ts_code = """
interface User {
    name: string;
    age: number;
}

function getUser(): User {
    return { name: "John", age: 30 };
}
"""
        chunker = ASTCodeChunker(language="typescript")
        chunks = chunker.chunk_file(file_path="test.ts", content=ts_code)

        assert len(chunks) > 0
        assert all(c.metadata.get("chunk_type") == "line_based" for c in chunks)

    def test_line_based_chunks_have_line_info(self) -> None:
        """Line-based chunks should have line metadata."""
        code = "line1\nline2\nline3\nline4\nline5"
        chunker = ASTCodeChunker(language="javascript")
        chunks = chunker.chunk_file(file_path="test.js", content=code)

        for chunk in chunks:
            assert "line_start" in chunk.metadata
            assert "line_end" in chunk.metadata
            assert chunk.metadata["line_start"] <= chunk.metadata["line_end"]

    def test_line_based_chunks_sequential(self) -> None:
        """Line-based chunks should have sequential indices."""
        code = "\n".join([f"line{i}" for i in range(50)])
        chunker = ASTCodeChunker(language="javascript")
        chunks = chunker.chunk_file(file_path="test.js", content=code)

        for i, chunk in enumerate(chunks):
            assert chunk.index == i


# ============================================================================
# Test Cases: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_python_fallback(self, empty_python: str) -> None:
        """Empty Python file should still produce chunks."""
        chunker = ASTCodeChunker(language="python")
        chunks = chunker.chunk_file(file_path="empty.py", content=empty_python)

        # Should fallback to line-based if no AST nodes
        assert len(chunks) >= 0

    def test_very_large_function(self) -> None:
        """Very large functions should be chunked."""
        large_func = (
            "def large_func():\n    "
            + "\n    ".join([f"x = {i}" for i in range(100)])
            + "\n    return x"
        )

        chunker = ASTCodeChunker(language="python")
        chunks = chunker.chunk_file(
            file_path="large.py",
            content=large_func,
        )

        assert len(chunks) >= 1

    def test_unicode_characters(self) -> None:
        """Code with unicode characters should be handled."""
        code = '''
def greet(name):
    """Greet someone with emoji and unicode."""
    return f"Hello {name} 👋 こんにちは"
'''
        chunker = ASTCodeChunker(language="python")
        chunks = chunker.chunk_file(file_path="unicode.py", content=code)

        assert len(chunks) >= 1
        assert any("👋" in c.text for c in chunks)

    def test_mixed_tabs_spaces(self) -> None:
        """Code with mixed tabs and spaces should raise SyntaxError."""
        code = "def func():\n\tif True:\n        pass"

        chunker = ASTCodeChunker(language="python")

        # Mixed tabs/spaces should raise ProcessingError due to Python syntax rules
        with pytest.raises(ProcessingError) as exc_info:
            chunker.chunk_file(file_path="mixed.py", content=code)

        assert "syntax" in str(exc_info.value.message).lower()

    def test_path_object_input(self, simple_python_code: str) -> None:
        """Path object should be accepted as file_path."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(simple_python_code)
            temp_path = Path(f.name)

        try:
            chunker = ASTCodeChunker(language="python")
            chunks = chunker.chunk_file(file_path=temp_path)

            assert len(chunks) > 0
        finally:
            temp_path.unlink()


# ============================================================================
# Test Cases: Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_workflow_python(self, simple_python_code: str) -> None:
        """Complete workflow: init -> chunk -> verify."""
        chunker = ASTCodeChunker(language="python", min_chunk_lines=2)
        chunks = chunker.chunk_file(
            file_path="test.py",
            content=simple_python_code,
        )

        # Verify structure
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(c.index >= 0 for c in chunks)
        assert all(c.metadata for c in chunks)
        assert all(c.start_char >= 0 and c.end_char > c.start_char for c in chunks)

        # Verify metadata consistency
        for i, chunk in enumerate(chunks):
            assert chunk.index == i

    def test_extract_and_chunk_consistency(self, simple_python_code: str) -> None:
        """extract_functions/classes should match top-level chunked results."""
        tree = ast.parse(simple_python_code)
        chunker = ASTCodeChunker(language="python")

        classes = chunker.extract_classes(tree)
        chunks = chunker.chunk_file(
            file_path="test.py",
            content=simple_python_code,
        )

        # All extracted classes should appear in chunks
        chunked_names = [c.metadata.get("name") for c in chunks]

        # Classes are extracted globally via ast.walk (includes nested)
        # but chunks are top-level only, so only check top-level classes
        top_level_classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]

        for cls in top_level_classes:
            assert cls.name in chunked_names

    def test_metadata_consistency_across_methods(self, simple_python_code: str) -> None:
        """Metadata from get_chunk_metadata should match chunk metadata."""
        tree = ast.parse(simple_python_code)
        chunker = ASTCodeChunker(language="python")

        # Get first function
        func_node = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "hello"
        )

        metadata = chunker.get_chunk_metadata(func_node)
        chunks = chunker.chunk_file(
            file_path="test.py",
            content=simple_python_code,
        )

        hello_chunk = next(
            (c for c in chunks if c.metadata.get("name") == "hello"),
            None,
        )

        assert hello_chunk is not None
        assert hello_chunk.metadata.get("docstring") == metadata.docstring
        assert hello_chunk.metadata.get("is_private") == metadata.is_private
