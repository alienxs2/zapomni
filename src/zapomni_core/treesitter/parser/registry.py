"""
Central registry for language parsers and extractors.

This module provides a singleton registry for managing tree-sitter parsers
and code extractors. It supports lazy registration and lookup of parsers
and extractors by language name or file extension.

Usage:
    from zapomni_core.treesitter.parser.registry import LanguageParserRegistry

    registry = LanguageParserRegistry()
    registry.register_parser("python", python_parser)
    parser = registry.get_parser("python")
"""

from typing import TYPE_CHECKING, Dict, List, Optional

import structlog

from ..config import get_language_by_extension

if TYPE_CHECKING:
    from ..extractors.base import BaseCodeExtractor
    from .base import BaseLanguageParser


class LanguageParserRegistry:
    """
    Singleton registry for language parsers and code extractors.

    This registry provides a centralized location for registering and
    retrieving tree-sitter parsers and code extractors. It follows the
    singleton pattern to ensure a single shared instance across the
    application.

    The registry supports:
    - Registering parsers by language name
    - Registering extractors by language name
    - Retrieving parsers and extractors with fallback support
    - Looking up languages by file extension
    - Listing all registered languages

    Attributes:
        _parsers: Dictionary mapping language names to parsers.
        _extractors: Dictionary mapping language names to extractors.
        logger: Structured logger instance for this class.

    Example:
        >>> registry = LanguageParserRegistry()
        >>> registry.register_parser("python", python_parser)
        >>> parser = registry.get_parser("python")
        >>> extractor = registry.get_extractor("python")  # falls back to "generic"
    """

    _instance: Optional["LanguageParserRegistry"] = None

    def __new__(cls) -> "LanguageParserRegistry":
        """
        Create or return the singleton instance.

        Returns:
            The singleton LanguageParserRegistry instance.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """
        Initialize the registry (only on first instantiation).

        Subsequent instantiations will skip initialization due to
        the singleton pattern.
        """
        if self._initialized:
            return

        self._parsers: Dict[str, "BaseLanguageParser"] = {}
        self._extractors: Dict[str, "BaseCodeExtractor"] = {}
        self._initialized = True
        self.logger = structlog.get_logger(__name__)

        self.logger.debug("LanguageParserRegistry initialized")

    def register_parser(self, language: str, parser: "BaseLanguageParser") -> None:
        """
        Register a language parser.

        Args:
            language: The programming language name (e.g., "python", "javascript").
            parser: The parser instance implementing BaseLanguageParser.

        Example:
            >>> registry.register_parser("python", PythonParser())
        """
        self._parsers[language] = parser
        self.logger.debug("Parser registered", language=language)

    def register_extractor(self, language: str, extractor: "BaseCodeExtractor") -> None:
        """
        Register a code extractor for a language.

        Args:
            language: The programming language name (e.g., "python", "javascript").
                      Use "generic" for the fallback extractor.
            extractor: The extractor instance implementing BaseCodeExtractor.

        Example:
            >>> registry.register_extractor("python", PythonExtractor())
            >>> registry.register_extractor("generic", GenericExtractor())
        """
        self._extractors[language] = extractor
        self.logger.debug("Extractor registered", language=language)

    def get_parser(self, language: str) -> Optional["BaseLanguageParser"]:
        """
        Get a parser for the specified language.

        Args:
            language: The programming language name (e.g., "python", "go").

        Returns:
            The registered parser if found, None otherwise.

        Example:
            >>> parser = registry.get_parser("python")
            >>> if parser:
            ...     tree = parser.parse(source_code)
        """
        parser = self._parsers.get(language)
        if parser is None:
            self.logger.debug("Parser not found", language=language)
        return parser

    def get_extractor(self, language: str) -> Optional["BaseCodeExtractor"]:
        """
        Get an extractor for the specified language with fallback to generic.

        If no specific extractor is registered for the language, this method
        will attempt to return the "generic" extractor as a fallback.

        Args:
            language: The programming language name (e.g., "python", "go").

        Returns:
            The language-specific extractor if registered, otherwise the
            generic extractor if available, or None if neither exists.

        Example:
            >>> extractor = registry.get_extractor("cobol")  # Not registered
            >>> # Returns generic extractor if available
        """
        extractor = self._extractors.get(language)
        if extractor is not None:
            return extractor

        # Fallback to generic extractor
        self.logger.debug(
            "Specific extractor not found, trying generic fallback",
            language=language,
        )
        return self._extractors.get("generic")

    def get_language_by_extension(self, extension: str) -> Optional[str]:
        """
        Get the programming language name for a file extension.

        Delegates to the config module's get_language_by_extension function.

        Args:
            extension: File extension (with or without leading dot).
                       Examples: ".py", "py", ".ts", "ts"

        Returns:
            Language name if found, None otherwise.

        Example:
            >>> registry.get_language_by_extension(".py")
            'python'
            >>> registry.get_language_by_extension("ts")
            'typescript'
        """
        return get_language_by_extension(extension)

    def list_registered_languages(self) -> List[str]:
        """
        Get a list of all languages with registered parsers.

        Returns:
            Sorted list of language names that have parsers registered.

        Example:
            >>> registry.list_registered_languages()
            ['go', 'javascript', 'python', 'rust', 'typescript']
        """
        return sorted(self._parsers.keys())

    def list_registered_extractors(self) -> List[str]:
        """
        Get a list of all languages with registered extractors.

        Returns:
            Sorted list of language names that have extractors registered.
            May include "generic" if a fallback extractor is registered.

        Example:
            >>> registry.list_registered_extractors()
            ['generic', 'javascript', 'python', 'typescript']
        """
        return sorted(self._extractors.keys())

    def clear(self) -> None:
        """
        Clear all registered parsers and extractors.

        This method is primarily intended for testing purposes to reset
        the registry state between tests.

        Warning:
            This will remove all registered parsers and extractors.
            Use with caution in production code.

        Example:
            >>> registry.clear()
            >>> registry.list_registered_languages()
            []
        """
        self._parsers.clear()
        self._extractors.clear()
        self.logger.debug("Registry cleared")

    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset the singleton instance.

        This class method completely resets the singleton, allowing a
        new instance to be created on the next instantiation. This is
        primarily intended for testing purposes.

        Warning:
            This will destroy the current singleton instance and all
            registered parsers and extractors. Use with caution.

        Example:
            >>> LanguageParserRegistry.reset_instance()
            >>> # Next instantiation creates a fresh instance
            >>> registry = LanguageParserRegistry()
        """
        cls._instance = None
