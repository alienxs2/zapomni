"""
Integration tests for index_codebase extractor integration.

Tests verify that language-specific extractors (PythonExtractor, TypeScriptExtractor)
are properly used instead of GenericExtractor for supported languages.

Issue #21: Tree-sitter Integration
"""

import tempfile
from unittest.mock import AsyncMock, Mock

import pytest

from zapomni_core.code.repository_indexer import CodeRepositoryIndexer
from zapomni_core.memory_processor import MemoryProcessor
from zapomni_core.treesitter.extractors import (
    GenericExtractor,
    GoExtractor,
    PythonExtractor,
    RustExtractor,
    TypeScriptExtractor,
)
from zapomni_core.treesitter.parser.registry import LanguageParserRegistry
from zapomni_mcp.tools.index_codebase import IndexCodebaseTool


class TestExtractorRegistration:
    """Test that extractors are properly registered."""

    def test_registry_contains_all_extractors(self):
        """Verify all expected extractors are registered."""
        registry = LanguageParserRegistry()
        extractors = registry.list_registered_extractors()

        assert "python" in extractors
        assert "typescript" in extractors
        assert "javascript" in extractors
        assert "generic" in extractors

    def test_python_extractor_registered(self):
        """Verify PythonExtractor is registered for Python."""
        registry = LanguageParserRegistry()
        extractor = registry.get_extractor("python")

        assert extractor is not None
        assert isinstance(extractor, PythonExtractor)

    def test_typescript_extractor_registered(self):
        """Verify TypeScriptExtractor is registered for TypeScript."""
        registry = LanguageParserRegistry()
        extractor = registry.get_extractor("typescript")

        assert extractor is not None
        assert isinstance(extractor, TypeScriptExtractor)

    def test_javascript_uses_typescript_extractor(self):
        """Verify JavaScript uses TypeScriptExtractor."""
        registry = LanguageParserRegistry()
        extractor = registry.get_extractor("javascript")

        assert extractor is not None
        assert isinstance(extractor, TypeScriptExtractor)

    def test_unknown_language_falls_back_to_generic(self):
        """Verify unknown languages fall back to GenericExtractor."""
        registry = LanguageParserRegistry()
        extractor = registry.get_extractor("cobol")

        assert extractor is not None
        assert isinstance(extractor, GenericExtractor)

    def test_go_extractor_registered(self):
        """Verify GoExtractor is registered for Go."""
        registry = LanguageParserRegistry()
        extractor = registry.get_extractor("go")

        assert extractor is not None
        assert isinstance(extractor, GoExtractor)

    def test_rust_extractor_registered(self):
        """Verify RustExtractor is registered for Rust."""
        registry = LanguageParserRegistry()
        extractor = registry.get_extractor("rust")

        assert extractor is not None
        assert isinstance(extractor, RustExtractor)


class TestParseFileAstIntegration:
    """Test _parse_file_ast uses correct extractors."""

    @pytest.fixture
    def tool(self):
        """Create IndexCodebaseTool with mocks."""
        mock_indexer = Mock(spec=CodeRepositoryIndexer)
        mock_processor = Mock(spec=MemoryProcessor)
        mock_processor.add_memory = AsyncMock(return_value="test-id")
        return IndexCodebaseTool(
            repository_indexer=mock_indexer,
            memory_processor=mock_processor,
        )

    def test_python_file_extracts_docstring(self, tool):
        """Verify Python files use PythonExtractor (extracts docstrings)."""
        python_code = '''
def hello_world():
    """This is a docstring that only PythonExtractor extracts."""
    print("Hello, World!")

class MyClass:
    """Class docstring."""

    def method(self):
        """Method docstring."""
        pass
'''
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(python_code.encode("utf-8"))
            f.flush()

            result = tool._parse_file_ast(f.name, python_code.encode("utf-8"))

            assert len(result["errors"]) == 0
            assert len(result["functions"]) >= 1

            # PythonExtractor extracts docstrings
            hello_func = next((f for f in result["functions"] if f.name == "hello_world"), None)
            assert hello_func is not None
            assert hello_func.docstring is not None
            assert "docstring" in hello_func.docstring.lower()

    def test_typescript_file_extracts_jsdoc(self, tool):
        """Verify TypeScript files use TypeScriptExtractor (extracts JSDoc)."""
        ts_code = """
/**
 * This is a JSDoc comment that only TypeScriptExtractor extracts.
 * @param name The name parameter
 * @returns A greeting string
 */
function greet(name: string): string {
    return `Hello, ${name}!`;
}

class Greeter {
    /** Greeting message */
    message: string;

    constructor() {
        this.message = "Hello";
    }
}
"""
        with tempfile.NamedTemporaryFile(suffix=".ts", delete=False) as f:
            f.write(ts_code.encode("utf-8"))
            f.flush()

            result = tool._parse_file_ast(f.name, ts_code.encode("utf-8"))

            assert len(result["errors"]) == 0
            assert len(result["functions"]) >= 1

            # TypeScriptExtractor extracts JSDoc
            greet_func = next((f for f in result["functions"] if f.name == "greet"), None)
            assert greet_func is not None
            assert greet_func.docstring is not None
            assert "JSDoc" in greet_func.docstring or "parameter" in greet_func.docstring.lower()

    def test_javascript_file_uses_typescript_extractor(self, tool):
        """Verify JavaScript files use TypeScriptExtractor."""
        js_code = """
/**
 * Add two numbers
 * @param {number} a First number
 * @param {number} b Second number
 * @returns {number} Sum
 */
function add(a, b) {
    return a + b;
}
"""
        with tempfile.NamedTemporaryFile(suffix=".js", delete=False) as f:
            f.write(js_code.encode("utf-8"))
            f.flush()

            result = tool._parse_file_ast(f.name, js_code.encode("utf-8"))

            assert len(result["errors"]) == 0
            assert len(result["functions"]) >= 1

            add_func = next((f for f in result["functions"] if f.name == "add"), None)
            assert add_func is not None

    def test_go_file_uses_go_extractor(self, tool):
        """Verify Go files use GoExtractor."""
        go_code = """
package main

// Add adds two integers and returns the result.
func Add(a, b int) int {
    return a + b
}

func main() {
    println("Hello, World!")
}
"""
        with tempfile.NamedTemporaryFile(suffix=".go", delete=False) as f:
            f.write(go_code.encode("utf-8"))
            f.flush()

            result = tool._parse_file_ast(f.name, go_code.encode("utf-8"))

            assert len(result["errors"]) == 0
            assert len(result["functions"]) >= 2

            # GoExtractor extracts doc comments
            add_func = next((f for f in result["functions"] if f.name == "Add"), None)
            assert add_func is not None
            assert add_func.docstring is not None
            assert "adds two integers" in add_func.docstring.lower()
