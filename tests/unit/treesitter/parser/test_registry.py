"""Unit tests for zapomni_core.treesitter.parser.registry module."""

from unittest.mock import MagicMock

import pytest

from zapomni_core.treesitter.extractors.base import BaseCodeExtractor
from zapomni_core.treesitter.parser.base import BaseLanguageParser
from zapomni_core.treesitter.parser.registry import LanguageParserRegistry


class MockParser(BaseLanguageParser):
    """Mock parser for testing."""

    def __init__(self, lang_name: str = "mock"):
        self._lang_name = lang_name
        # Skip parent __init__ to avoid language validation
        self._parser = None
        self._log = MagicMock()

    @property
    def language_name(self) -> str:
        return self._lang_name

    @property
    def file_extensions(self) -> tuple[str, ...]:
        return (".mock",)

    def get_parser(self):
        return MagicMock()


class MockExtractor:
    """Mock extractor for testing."""

    def __init__(self, lang_name: str = "mock"):
        self._lang_name = lang_name

    @property
    def language_name(self) -> str:
        return self._lang_name


class TestLanguageParserRegistrySingleton:
    """Tests for singleton behavior of LanguageParserRegistry."""

    def test_singleton_same_instance(self):
        """Test that multiple instantiations return the same instance."""
        registry1 = LanguageParserRegistry()
        registry2 = LanguageParserRegistry()
        assert registry1 is registry2

    def test_reset_instance_creates_new(self):
        """Test that reset_instance allows creating a new instance."""
        registry1 = LanguageParserRegistry()
        registry1.register_parser("test", MockParser("test"))

        LanguageParserRegistry.reset_instance()

        registry2 = LanguageParserRegistry()
        # Should be a new instance with empty registrations
        assert registry2.get_parser("test") is None

    def test_initialized_flag(self):
        """Test that _initialized flag prevents re-initialization."""
        registry = LanguageParserRegistry()
        assert registry._initialized is True


class TestParserRegistration:
    """Tests for parser registration functionality."""

    def test_register_parser(self):
        """Test registering a parser."""
        registry = LanguageParserRegistry()
        parser = MockParser("python")
        registry.register_parser("python", parser)

        retrieved = registry.get_parser("python")
        assert retrieved is parser

    def test_get_parser_not_found(self):
        """Test getting a parser that doesn't exist."""
        registry = LanguageParserRegistry()
        result = registry.get_parser("nonexistent")
        assert result is None

    def test_register_multiple_parsers(self):
        """Test registering multiple parsers."""
        registry = LanguageParserRegistry()
        python_parser = MockParser("python")
        js_parser = MockParser("javascript")

        registry.register_parser("python", python_parser)
        registry.register_parser("javascript", js_parser)

        assert registry.get_parser("python") is python_parser
        assert registry.get_parser("javascript") is js_parser

    def test_overwrite_parser(self):
        """Test that registering with same name overwrites."""
        registry = LanguageParserRegistry()
        parser1 = MockParser("python")
        parser2 = MockParser("python")

        registry.register_parser("python", parser1)
        registry.register_parser("python", parser2)

        assert registry.get_parser("python") is parser2


class TestExtractorRegistration:
    """Tests for extractor registration functionality."""

    def test_register_extractor(self):
        """Test registering an extractor."""
        registry = LanguageParserRegistry()
        extractor = MockExtractor("python")
        registry.register_extractor("python", extractor)

        retrieved = registry.get_extractor("python")
        assert retrieved is extractor

    def test_get_extractor_fallback_to_generic(self):
        """Test that get_extractor falls back to generic."""
        registry = LanguageParserRegistry()
        generic = MockExtractor("generic")
        registry.register_extractor("generic", generic)

        # Request extractor for unregistered language
        result = registry.get_extractor("unknown_language")
        assert result is generic

    def test_get_extractor_no_fallback(self):
        """Test get_extractor returns None when no generic available."""
        registry = LanguageParserRegistry()
        result = registry.get_extractor("unknown_language")
        assert result is None

    def test_specific_extractor_over_generic(self):
        """Test that specific extractor is returned over generic."""
        registry = LanguageParserRegistry()
        generic = MockExtractor("generic")
        python = MockExtractor("python")

        registry.register_extractor("generic", generic)
        registry.register_extractor("python", python)

        assert registry.get_extractor("python") is python


class TestGetLanguageByExtension:
    """Tests for get_language_by_extension method."""

    def test_py_extension(self):
        """Test .py extension lookup."""
        registry = LanguageParserRegistry()
        assert registry.get_language_by_extension(".py") == "python"

    def test_js_extension(self):
        """Test .js extension lookup."""
        registry = LanguageParserRegistry()
        assert registry.get_language_by_extension(".js") == "javascript"

    def test_without_dot(self):
        """Test extension without leading dot."""
        registry = LanguageParserRegistry()
        assert registry.get_language_by_extension("py") == "python"

    def test_unknown_extension(self):
        """Test unknown extension returns None."""
        registry = LanguageParserRegistry()
        assert registry.get_language_by_extension(".xyz") is None


class TestListMethods:
    """Tests for list_registered_languages and list_registered_extractors."""

    def test_list_registered_languages_empty(self):
        """Test list_registered_languages when empty."""
        registry = LanguageParserRegistry()
        assert registry.list_registered_languages() == []

    def test_list_registered_languages(self):
        """Test list_registered_languages returns sorted list."""
        registry = LanguageParserRegistry()
        registry.register_parser("python", MockParser("python"))
        registry.register_parser("javascript", MockParser("javascript"))
        registry.register_parser("rust", MockParser("rust"))

        languages = registry.list_registered_languages()
        assert languages == ["javascript", "python", "rust"]

    def test_list_registered_extractors_empty(self):
        """Test list_registered_extractors when empty."""
        registry = LanguageParserRegistry()
        assert registry.list_registered_extractors() == []

    def test_list_registered_extractors(self):
        """Test list_registered_extractors returns sorted list."""
        registry = LanguageParserRegistry()
        registry.register_extractor("python", MockExtractor("python"))
        registry.register_extractor("generic", MockExtractor("generic"))

        extractors = registry.list_registered_extractors()
        assert extractors == ["generic", "python"]


class TestClear:
    """Tests for clear method."""

    def test_clear_removes_parsers(self):
        """Test that clear removes all parsers."""
        registry = LanguageParserRegistry()
        registry.register_parser("python", MockParser("python"))
        registry.register_parser("javascript", MockParser("javascript"))

        registry.clear()

        assert registry.get_parser("python") is None
        assert registry.get_parser("javascript") is None
        assert registry.list_registered_languages() == []

    def test_clear_removes_extractors(self):
        """Test that clear removes all extractors."""
        registry = LanguageParserRegistry()
        registry.register_extractor("python", MockExtractor("python"))
        registry.register_extractor("generic", MockExtractor("generic"))

        registry.clear()

        assert registry.get_extractor("python") is None
        assert registry.list_registered_extractors() == []

    def test_clear_is_idempotent(self):
        """Test that calling clear multiple times is safe."""
        registry = LanguageParserRegistry()
        registry.register_parser("python", MockParser("python"))

        registry.clear()
        registry.clear()  # Should not raise

        assert registry.list_registered_languages() == []
