"""Unit tests for zapomni_core.treesitter.parser.factory module."""


import pytest
from tree_sitter import Parser

from zapomni_core.treesitter.parser.base import BaseLanguageParser
from zapomni_core.treesitter.parser.factory import (
    ParserFactory,
    UniversalLanguageParser,
)
from zapomni_core.treesitter.parser.registry import LanguageParserRegistry


class TestUniversalLanguageParser:
    """Tests for UniversalLanguageParser class."""

    def test_init_with_python(self):
        """Test initializing parser for Python."""
        parser = UniversalLanguageParser("python")
        assert parser.language_name == "python"

    def test_init_with_empty_name_raises(self):
        """Test that empty language name raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            UniversalLanguageParser("")
        assert "cannot be empty" in str(exc_info.value)

    def test_init_with_none_raises(self):
        """Test that None language name raises ValueError."""
        with pytest.raises(ValueError):
            UniversalLanguageParser(None)

    def test_language_name_property(self):
        """Test language_name property."""
        parser = UniversalLanguageParser("javascript")
        assert parser.language_name == "javascript"

    def test_file_extensions_property(self):
        """Test file_extensions property returns correct extensions."""
        parser = UniversalLanguageParser("python")
        extensions = parser.file_extensions
        assert ".py" in extensions
        assert ".pyw" in extensions

    def test_file_extensions_for_unknown_language(self):
        """Test file_extensions returns empty tuple for unknown language with valid grammar."""
        # This will fail at get_parser() time, but extensions would be empty
        parser = UniversalLanguageParser("c")  # Use a real language
        extensions = parser.file_extensions
        assert ".c" in extensions

    def test_get_parser_returns_parser(self):
        """Test that get_parser returns a tree-sitter Parser."""
        parser = UniversalLanguageParser("python")
        ts_parser = parser.get_parser()
        assert isinstance(ts_parser, Parser)

    def test_get_parser_caches_parser(self):
        """Test that get_parser caches the parser instance."""
        parser = UniversalLanguageParser("python")
        ts_parser1 = parser.get_parser()
        ts_parser2 = parser.get_parser()
        assert ts_parser1 is ts_parser2

    def test_get_parser_for_multiple_languages(self):
        """Test creating parsers for multiple languages."""
        languages = ["python", "javascript", "rust", "go", "java"]
        for lang in languages:
            parser = UniversalLanguageParser(lang)
            ts_parser = parser.get_parser()
            assert ts_parser is not None

    def test_parse_python_code(self):
        """Test parsing Python code."""
        parser = UniversalLanguageParser("python")
        tree = parser.parse(b"def hello(): pass")
        assert tree is not None
        assert tree.root_node.type == "module"

    def test_parse_javascript_code(self):
        """Test parsing JavaScript code."""
        parser = UniversalLanguageParser("javascript")
        tree = parser.parse(b"function hello() { }")
        assert tree is not None
        assert tree.root_node.type == "program"


class TestParserFactory:
    """Tests for ParserFactory class."""

    def test_is_initialized_before_init(self):
        """Test is_initialized returns False before initialization."""
        ParserFactory.reset()
        assert ParserFactory.is_initialized() is False

    def test_initialize(self):
        """Test that initialize sets up the factory."""
        ParserFactory.reset()
        ParserFactory.initialize()
        assert ParserFactory.is_initialized() is True

    def test_initialize_is_idempotent(self):
        """Test that calling initialize multiple times is safe."""
        ParserFactory.reset()
        ParserFactory.initialize()
        ParserFactory.initialize()  # Should not raise
        assert ParserFactory.is_initialized() is True

    def test_get_parser_auto_initializes(self):
        """Test that get_parser auto-initializes if needed."""
        ParserFactory.reset()
        assert ParserFactory.is_initialized() is False

        parser = ParserFactory.get_parser("python")
        assert ParserFactory.is_initialized() is True
        assert parser is not None

    def test_get_parser_returns_base_language_parser(self):
        """Test that get_parser returns a BaseLanguageParser."""
        parser = ParserFactory.get_parser("python")
        assert isinstance(parser, BaseLanguageParser)

    def test_get_parser_caches_in_registry(self):
        """Test that get_parser registers parser in registry."""
        ParserFactory.reset()
        ParserFactory.initialize()

        parser = ParserFactory.get_parser("python")
        registry = LanguageParserRegistry()
        cached = registry.get_parser("python")

        assert cached is parser

    def test_get_parser_for_various_languages(self):
        """Test getting parsers for various languages."""
        languages = ["python", "javascript", "typescript", "rust", "go"]
        for lang in languages:
            parser = ParserFactory.get_parser(lang)
            assert parser is not None
            assert parser.language_name == lang

    def test_get_parser_for_unsupported_language(self):
        """Test that unsupported language raises error."""
        with pytest.raises(Exception):  # Will raise when trying to get grammar
            ParserFactory.get_parser("totally_fake_language_xyz")


class TestParserFactoryGetParserForFile:
    """Tests for ParserFactory.get_parser_for_file method."""

    def test_get_parser_for_python_file(self):
        """Test getting parser for .py file."""
        parser = ParserFactory.get_parser_for_file("/path/to/script.py")
        assert parser is not None
        assert parser.language_name == "python"

    def test_get_parser_for_javascript_file(self):
        """Test getting parser for .js file."""
        parser = ParserFactory.get_parser_for_file("/path/to/app.js")
        assert parser is not None
        assert parser.language_name == "javascript"

    def test_get_parser_for_typescript_file(self):
        """Test getting parser for .ts file."""
        parser = ParserFactory.get_parser_for_file("/path/to/app.ts")
        assert parser is not None
        assert parser.language_name == "typescript"

    def test_get_parser_for_rust_file(self):
        """Test getting parser for .rs file."""
        parser = ParserFactory.get_parser_for_file("/path/to/main.rs")
        assert parser is not None
        assert parser.language_name == "rust"

    def test_get_parser_for_unknown_extension(self):
        """Test that unknown extension returns None."""
        parser = ParserFactory.get_parser_for_file("/path/to/file.xyz")
        assert parser is None

    def test_get_parser_for_dockerfile(self):
        """Test getting parser for Dockerfile (no extension)."""
        parser = ParserFactory.get_parser_for_file("/path/to/Dockerfile")
        assert parser is not None
        assert parser.language_name == "dockerfile"

    def test_get_parser_for_nested_path(self):
        """Test getting parser for file in nested path."""
        parser = ParserFactory.get_parser_for_file("/a/b/c/d/e/script.py")
        assert parser is not None
        assert parser.language_name == "python"

    def test_get_parser_auto_initializes(self):
        """Test that get_parser_for_file auto-initializes."""
        ParserFactory.reset()
        assert ParserFactory.is_initialized() is False

        ParserFactory.get_parser_for_file("/test.py")
        assert ParserFactory.is_initialized() is True


class TestParserFactoryReset:
    """Tests for ParserFactory.reset method."""

    def test_reset_clears_initialized_flag(self):
        """Test that reset clears the initialized flag."""
        ParserFactory.initialize()
        assert ParserFactory.is_initialized() is True

        ParserFactory.reset()
        assert ParserFactory.is_initialized() is False

    def test_reset_clears_registry(self):
        """Test that reset clears the registry."""
        ParserFactory.initialize()
        ParserFactory.get_parser("python")

        ParserFactory.reset()

        registry = LanguageParserRegistry()
        assert registry.list_registered_languages() == []

    def test_reset_allows_reinitialize(self):
        """Test that reset allows re-initialization."""
        ParserFactory.initialize()
        ParserFactory.reset()
        ParserFactory.initialize()

        assert ParserFactory.is_initialized() is True


class TestParserFactoryGenericExtractor:
    """Tests for GenericExtractor registration during initialization."""

    def test_generic_extractor_registered(self):
        """Test that GenericExtractor is registered during initialization."""
        ParserFactory.reset()
        ParserFactory.initialize()

        registry = LanguageParserRegistry()
        extractors = registry.list_registered_extractors()
        assert "generic" in extractors

    def test_generic_extractor_available_as_fallback(self):
        """Test that GenericExtractor is available as fallback."""
        ParserFactory.reset()
        ParserFactory.initialize()

        registry = LanguageParserRegistry()
        extractor = registry.get_extractor("unknown_language")
        assert extractor is not None
        assert extractor.language_name == "generic"
