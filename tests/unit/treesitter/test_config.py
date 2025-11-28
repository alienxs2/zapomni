"""Unit tests for zapomni_core.treesitter.config module."""

import pytest

from zapomni_core.treesitter.config import (
    EXTENSION_TO_LANGUAGE,
    LANGUAGE_EXTENSIONS,
    LANGUAGES_WITH_EXTRACTORS,
    get_config_stats,
    get_extensions_by_language,
    get_language_by_extension,
    is_supported_extension,
    is_supported_language,
)


class TestLanguageExtensions:
    """Tests for LANGUAGE_EXTENSIONS mapping."""

    def test_python_extensions(self):
        """Test Python has correct extensions."""
        assert ".py" in LANGUAGE_EXTENSIONS["python"]
        assert ".pyw" in LANGUAGE_EXTENSIONS["python"]
        assert ".pyi" in LANGUAGE_EXTENSIONS["python"]

    def test_javascript_extensions(self):
        """Test JavaScript has correct extensions."""
        assert ".js" in LANGUAGE_EXTENSIONS["javascript"]
        assert ".jsx" in LANGUAGE_EXTENSIONS["javascript"]
        assert ".mjs" in LANGUAGE_EXTENSIONS["javascript"]

    def test_typescript_extensions(self):
        """Test TypeScript has correct extensions."""
        assert ".ts" in LANGUAGE_EXTENSIONS["typescript"]
        assert ".tsx" in LANGUAGE_EXTENSIONS["typescript"]

    def test_rust_extensions(self):
        """Test Rust has correct extensions."""
        assert ".rs" in LANGUAGE_EXTENSIONS["rust"]

    def test_go_extensions(self):
        """Test Go has correct extensions."""
        assert ".go" in LANGUAGE_EXTENSIONS["go"]

    def test_popular_languages_present(self):
        """Test that popular languages are present."""
        popular = [
            "python", "javascript", "typescript", "java", "go",
            "rust", "c", "cpp", "ruby", "php", "swift",
        ]
        for lang in popular:
            assert lang in LANGUAGE_EXTENSIONS, f"{lang} not in LANGUAGE_EXTENSIONS"

    def test_web_technologies_present(self):
        """Test that web technologies are present."""
        web_tech = ["html", "css", "scss", "json", "yaml", "xml"]
        for tech in web_tech:
            assert tech in LANGUAGE_EXTENSIONS, f"{tech} not in LANGUAGE_EXTENSIONS"

    def test_special_files(self):
        """Test special files without extensions."""
        assert "Dockerfile" in LANGUAGE_EXTENSIONS["dockerfile"]
        assert "Makefile" in LANGUAGE_EXTENSIONS["makefile"]


class TestExtensionToLanguage:
    """Tests for EXTENSION_TO_LANGUAGE reverse mapping."""

    def test_py_maps_to_python(self):
        """Test .py maps to python."""
        assert EXTENSION_TO_LANGUAGE[".py"] == "python"

    def test_js_maps_to_javascript(self):
        """Test .js maps to javascript."""
        assert EXTENSION_TO_LANGUAGE[".js"] == "javascript"

    def test_ts_maps_to_typescript(self):
        """Test .ts maps to typescript."""
        assert EXTENSION_TO_LANGUAGE[".ts"] == "typescript"

    def test_rs_maps_to_rust(self):
        """Test .rs maps to rust."""
        assert EXTENSION_TO_LANGUAGE[".rs"] == "rust"

    def test_all_extensions_mapped(self):
        """Test all extensions from LANGUAGE_EXTENSIONS are in reverse mapping."""
        for lang, extensions in LANGUAGE_EXTENSIONS.items():
            for ext in extensions:
                assert ext in EXTENSION_TO_LANGUAGE, f"{ext} not mapped"


class TestGetLanguageByExtension:
    """Tests for get_language_by_extension function."""

    def test_with_dot(self):
        """Test extension with leading dot."""
        assert get_language_by_extension(".py") == "python"
        assert get_language_by_extension(".js") == "javascript"
        assert get_language_by_extension(".ts") == "typescript"

    def test_without_dot(self):
        """Test extension without leading dot."""
        assert get_language_by_extension("py") == "python"
        assert get_language_by_extension("js") == "javascript"
        assert get_language_by_extension("rs") == "rust"

    def test_unknown_extension(self):
        """Test unknown extension returns None."""
        assert get_language_by_extension(".unknown") is None
        assert get_language_by_extension("xyz") is None

    def test_special_files(self):
        """Test special files like Dockerfile."""
        assert get_language_by_extension("Dockerfile") == "dockerfile"
        assert get_language_by_extension("Makefile") == "makefile"


class TestGetExtensionsByLanguage:
    """Tests for get_extensions_by_language function."""

    def test_python_extensions(self):
        """Test getting Python extensions."""
        extensions = get_extensions_by_language("python")
        assert ".py" in extensions
        assert ".pyw" in extensions
        assert ".pyi" in extensions

    def test_javascript_extensions(self):
        """Test getting JavaScript extensions."""
        extensions = get_extensions_by_language("javascript")
        assert ".js" in extensions
        assert ".jsx" in extensions

    def test_unknown_language(self):
        """Test unknown language returns empty tuple."""
        extensions = get_extensions_by_language("unknown_language")
        assert extensions == ()

    def test_returns_tuple(self):
        """Test that function returns a tuple."""
        extensions = get_extensions_by_language("python")
        assert isinstance(extensions, tuple)


class TestIsSupportedLanguage:
    """Tests for is_supported_language function."""

    def test_supported_languages(self):
        """Test that known languages are supported."""
        assert is_supported_language("python") is True
        assert is_supported_language("javascript") is True
        assert is_supported_language("rust") is True
        assert is_supported_language("go") is True

    def test_unsupported_languages(self):
        """Test that unknown languages are not supported."""
        assert is_supported_language("cobol") is False
        assert is_supported_language("fortran77") is False
        assert is_supported_language("unknown") is False

    def test_case_sensitive(self):
        """Test that language names are case-sensitive."""
        assert is_supported_language("Python") is False
        assert is_supported_language("PYTHON") is False
        assert is_supported_language("python") is True


class TestIsSupportedExtension:
    """Tests for is_supported_extension function."""

    def test_supported_with_dot(self):
        """Test supported extensions with dot."""
        assert is_supported_extension(".py") is True
        assert is_supported_extension(".js") is True
        assert is_supported_extension(".rs") is True

    def test_supported_without_dot(self):
        """Test supported extensions without dot."""
        assert is_supported_extension("py") is True
        assert is_supported_extension("js") is True
        assert is_supported_extension("ts") is True

    def test_unsupported_extensions(self):
        """Test unsupported extensions."""
        assert is_supported_extension(".xyz") is False
        assert is_supported_extension("unknown") is False

    def test_special_files(self):
        """Test special files without extensions."""
        assert is_supported_extension("Dockerfile") is True
        assert is_supported_extension("Makefile") is True


class TestGetConfigStats:
    """Tests for get_config_stats function."""

    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        stats = get_config_stats()
        assert isinstance(stats, dict)

    def test_contains_required_keys(self):
        """Test that stats contains required keys."""
        stats = get_config_stats()
        assert "total_languages" in stats
        assert "total_extensions" in stats
        assert "languages_with_extractors" in stats

    def test_total_languages_positive(self):
        """Test that total_languages is positive."""
        stats = get_config_stats()
        assert stats["total_languages"] > 0
        assert stats["total_languages"] == len(LANGUAGE_EXTENSIONS)

    def test_total_extensions_positive(self):
        """Test that total_extensions is positive."""
        stats = get_config_stats()
        assert stats["total_extensions"] > 0
        assert stats["total_extensions"] == len(EXTENSION_TO_LANGUAGE)

    def test_extractors_count(self):
        """Test languages_with_extractors count."""
        stats = get_config_stats()
        assert stats["languages_with_extractors"] == len(LANGUAGES_WITH_EXTRACTORS)


class TestLanguagesWithExtractors:
    """Tests for LANGUAGES_WITH_EXTRACTORS set."""

    def test_is_set(self):
        """Test that LANGUAGES_WITH_EXTRACTORS is a set."""
        assert isinstance(LANGUAGES_WITH_EXTRACTORS, set)

    def test_initially_empty_or_populated(self):
        """Test that set is initially empty (will be populated later)."""
        # As per design, this starts empty and is populated as extractors are added
        # This test documents the expected behavior
        assert isinstance(LANGUAGES_WITH_EXTRACTORS, set)
