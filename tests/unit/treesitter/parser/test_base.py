"""Unit tests for zapomni_core.treesitter.parser.base module."""

import os
import tempfile

import pytest
from tree_sitter import Tree

from zapomni_core.treesitter.exceptions import LanguageNotSupportedError, ParseError
from zapomni_core.treesitter.parser.base import BaseLanguageParser


class ConcreteParser(BaseLanguageParser):
    """Concrete implementation of BaseLanguageParser for testing."""

    @property
    def language_name(self) -> str:
        return "python"

    @property
    def file_extensions(self) -> tuple[str, ...]:
        return (".py", ".pyw", ".pyi")


class UnsupportedLanguageParser(BaseLanguageParser):
    """Parser for an unsupported language (for testing errors)."""

    @property
    def language_name(self) -> str:
        return "totally_unsupported_language_xyz"

    @property
    def file_extensions(self) -> tuple[str, ...]:
        return (".xyz",)


class TestBaseLanguageParser:
    """Tests for BaseLanguageParser abstract class."""

    def test_concrete_parser_initialization(self):
        """Test that concrete parser can be initialized."""
        parser = ConcreteParser()
        assert parser.language_name == "python"
        assert parser.file_extensions == (".py", ".pyw", ".pyi")

    def test_unsupported_language_raises_error(self):
        """Test that unsupported language raises LanguageNotSupportedError."""
        with pytest.raises(LanguageNotSupportedError) as exc_info:
            UnsupportedLanguageParser()
        assert "totally_unsupported_language_xyz" in str(exc_info.value)

    def test_get_parser_returns_parser(self):
        """Test that get_parser returns a tree-sitter Parser."""
        parser = ConcreteParser()
        ts_parser = parser.get_parser()
        assert ts_parser is not None

    def test_get_parser_caches_parser(self):
        """Test that get_parser returns the same parser instance."""
        parser = ConcreteParser()
        ts_parser1 = parser.get_parser()
        ts_parser2 = parser.get_parser()
        assert ts_parser1 is ts_parser2

    def test_parse_returns_tree(self):
        """Test that parse returns a Tree object."""
        parser = ConcreteParser()
        source = b"def hello(): pass"
        tree = parser.parse(source)
        assert isinstance(tree, Tree)
        assert tree.root_node is not None

    def test_parse_simple_function(self):
        """Test parsing a simple function."""
        parser = ConcreteParser()
        source = b"def hello():\n    print('Hello')"
        tree = parser.parse(source)
        assert tree.root_node.type == "module"

    def test_parse_class_definition(self):
        """Test parsing a class definition."""
        parser = ConcreteParser()
        source = b"class MyClass:\n    def __init__(self):\n        pass"
        tree = parser.parse(source)
        assert tree.root_node.type == "module"

    def test_parse_invalid_syntax_returns_tree(self):
        """Test that parse returns tree even with syntax errors (tree-sitter behavior)."""
        parser = ConcreteParser()
        source = b"def invalid syntax here {{"
        tree = parser.parse(source)
        # Tree-sitter returns a tree even with errors
        assert tree is not None
        # The tree should have error nodes
        assert tree.root_node.has_error

    def test_parse_empty_source(self):
        """Test parsing empty source code."""
        parser = ConcreteParser()
        tree = parser.parse(b"")
        assert tree is not None
        assert tree.root_node.type == "module"


class TestParseFile:
    """Tests for parse_file method."""

    def test_parse_file_success(self):
        """Test parsing an existing file."""
        parser = ConcreteParser()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def hello(): pass")
            f.flush()
            try:
                tree = parser.parse_file(f.name)
                assert isinstance(tree, Tree)
                assert tree.root_node.type == "module"
            finally:
                os.unlink(f.name)

    def test_parse_file_not_found(self):
        """Test that non-existent file raises ParseError."""
        parser = ConcreteParser()
        with pytest.raises(ParseError) as exc_info:
            parser.parse_file("/non/existent/file.py")
        assert "File not found" in str(exc_info.value)

    def test_parse_file_not_a_file(self):
        """Test that directory path raises ParseError."""
        parser = ConcreteParser()
        with pytest.raises(ParseError) as exc_info:
            parser.parse_file(tempfile.gettempdir())
        assert "not a file" in str(exc_info.value)

    def test_parse_file_permission_denied(self):
        """Test handling of permission denied error."""
        parser = ConcreteParser()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def hello(): pass")
            f.flush()
            try:
                # Make file unreadable
                os.chmod(f.name, 0o000)
                with pytest.raises(ParseError) as exc_info:
                    parser.parse_file(f.name)
                assert "Permission denied" in str(exc_info.value) or "Failed to read" in str(
                    exc_info.value
                )
            finally:
                # Restore permissions and delete
                os.chmod(f.name, 0o644)
                os.unlink(f.name)


class TestSupportsExtension:
    """Tests for supports_extension method."""

    def test_supports_py_with_dot(self):
        """Test supports_extension with .py."""
        parser = ConcreteParser()
        assert parser.supports_extension(".py") is True

    def test_supports_py_without_dot(self):
        """Test supports_extension with py (no dot)."""
        parser = ConcreteParser()
        assert parser.supports_extension("py") is True

    def test_supports_pyw(self):
        """Test supports_extension with .pyw."""
        parser = ConcreteParser()
        assert parser.supports_extension(".pyw") is True

    def test_does_not_support_js(self):
        """Test that .js is not supported by Python parser."""
        parser = ConcreteParser()
        assert parser.supports_extension(".js") is False

    def test_does_not_support_unknown(self):
        """Test that unknown extension is not supported."""
        parser = ConcreteParser()
        assert parser.supports_extension(".xyz") is False


class TestRepr:
    """Tests for __repr__ method."""

    def test_repr_contains_class_name(self):
        """Test that repr contains class name."""
        parser = ConcreteParser()
        repr_str = repr(parser)
        assert "ConcreteParser" in repr_str

    def test_repr_contains_language(self):
        """Test that repr contains language name."""
        parser = ConcreteParser()
        repr_str = repr(parser)
        assert "python" in repr_str
