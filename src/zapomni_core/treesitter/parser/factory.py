"""
Parser factory module for tree-sitter integration.

Provides UniversalLanguageParser for parsing any language supported by
tree-sitter-language-pack, and ParserFactory for managing parser instances.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from pathlib import Path
from typing import Optional

import structlog
from tree_sitter import Parser
from tree_sitter_language_pack import get_language

from ..config import LANGUAGE_EXTENSIONS, get_language_by_extension
from .base import BaseLanguageParser
from .registry import LanguageParserRegistry


class UniversalLanguageParser(BaseLanguageParser):
    """
    Universal parser that works with any language from tree-sitter-language-pack.

    This parser dynamically loads language grammars from tree-sitter-language-pack
    based on the language name provided at construction time. It provides a unified
    interface for parsing source code in any of the 165+ supported languages.

    Attributes:
        language_name: The name of the programming language (e.g., "python", "rust")
        file_extensions: Tuple of file extensions associated with this language

    Example:
        >>> parser = UniversalLanguageParser("python")
        >>> ts_parser = parser.get_parser()
        >>> tree = ts_parser.parse(b"def hello(): pass")
    """

    def __init__(self, language_name: str):
        """
        Initialize the universal parser for a specific language.

        Args:
            language_name: Name of the programming language (e.g., "python", "go").
                          Must be supported by tree-sitter-language-pack.

        Raises:
            ValueError: If language_name is empty or None.
        """
        if not language_name:
            raise ValueError("language_name cannot be empty")

        self._language_name = language_name
        self._parser: Optional[Parser] = None
        self._file_extensions = self._get_extensions_for_language(language_name)
        self._logger = structlog.get_logger(__name__)
        # _log is used by BaseLanguageParser.parse() method
        self._log = self._logger.bind(parser=self.__class__.__name__, language=language_name)

    @property
    def language_name(self) -> str:
        """
        Get the name of the programming language.

        Returns:
            The language name string (e.g., "python", "rust").
        """
        return self._language_name

    @property
    def file_extensions(self) -> tuple[str, ...]:
        """
        Get file extensions associated with this language.

        Returns:
            Tuple of file extensions (e.g., (".py", ".pyw", ".pyi") for Python).
        """
        return self._file_extensions

    def get_parser(self) -> Parser:
        """
        Get the tree-sitter parser for this language.

        Creates the parser lazily on first call and caches it for subsequent calls.
        The parser is configured with the appropriate language grammar from
        tree-sitter-language-pack.

        Returns:
            Configured tree-sitter Parser instance.

        Raises:
            LanguageNotSupportedError: If the language is not available in
                tree-sitter-language-pack.
        """
        if self._parser is None:
            self._parser = Parser()
            # get_language() will raise an exception if language not supported
            language = get_language(self._language_name)
            self._parser.language = language
            self._logger.debug(
                "Created parser for language",
                language=self._language_name
            )
        return self._parser

    @staticmethod
    def _get_extensions_for_language(language_name: str) -> tuple[str, ...]:
        """
        Get file extensions for a given language.

        Args:
            language_name: Programming language name.

        Returns:
            Tuple of file extensions, or empty tuple if not found in config.
        """
        return LANGUAGE_EXTENSIONS.get(language_name, ())


class ParserFactory:
    """
    Factory for creating and managing language parsers.

    This factory provides a centralized way to obtain parser instances for
    different programming languages. It uses lazy initialization to only
    create parsers when they are first requested, and registers them with
    the LanguageParserRegistry for reuse.

    The factory initializes parsers for all languages defined in LANGUAGE_EXTENSIONS
    that are supported by tree-sitter-language-pack.

    Class Attributes:
        _initialized: Whether the factory has been initialized.
        _logger: Structured logger instance.

    Example:
        >>> ParserFactory.initialize()
        >>> parser = ParserFactory.get_parser("python")
        >>> # Or get parser by file extension
        >>> parser = ParserFactory.get_parser_for_file("/path/to/code.py")
    """

    _initialized: bool = False
    _logger = structlog.get_logger(__name__)

    @classmethod
    def initialize(cls) -> None:
        """
        Initialize the factory by registering parsers for all supported languages.

        This method registers a UniversalLanguageParser for each language defined
        in LANGUAGE_EXTENSIONS that is supported by tree-sitter-language-pack.
        Languages that are not available in the pack are skipped with a warning.

        This method is idempotent - calling it multiple times has no effect
        after the first successful initialization.

        Example:
            >>> ParserFactory.initialize()
            >>> ParserFactory.is_initialized()
            True
        """
        if cls._initialized:
            cls._logger.debug("ParserFactory already initialized, skipping")
            return

        registry = LanguageParserRegistry()
        registered_count = 0
        skipped_count = 0

        for language in LANGUAGE_EXTENSIONS.keys():
            try:
                parser = UniversalLanguageParser(language)
                # Verify the language is supported by actually creating the parser
                # This will raise an exception if the language is not in the pack
                parser.get_parser()
                registry.register_parser(language, parser)
                registered_count += 1
                cls._logger.debug(
                    "Registered parser for language",
                    language=language
                )
            except Exception as e:
                skipped_count += 1
                cls._logger.warning(
                    "Language not available in tree-sitter-language-pack",
                    language=language,
                    error=str(e)
                )

        # Register GenericExtractor as fallback
        try:
            from ..extractors.generic import GenericExtractor
            registry.register_extractor("generic", GenericExtractor())
            cls._logger.debug("GenericExtractor registered as fallback")
        except ImportError as e:
            cls._logger.warning("Could not register GenericExtractor", error=str(e))

        cls._initialized = True
        cls._logger.info(
            "ParserFactory initialized",
            registered=registered_count,
            skipped=skipped_count,
            total_languages=len(LANGUAGE_EXTENSIONS)
        )

    @classmethod
    def get_parser(cls, language: str) -> BaseLanguageParser:
        """
        Get a parser for the specified programming language.

        If the factory is not initialized, it will be initialized automatically.
        If no parser exists for the language, a new UniversalLanguageParser
        is created and registered.

        Args:
            language: Programming language name (e.g., "python", "rust").

        Returns:
            BaseLanguageParser instance for the requested language.

        Raises:
            LanguageNotSupportedError: If the language is not supported by
                tree-sitter-language-pack.

        Example:
            >>> parser = ParserFactory.get_parser("python")
            >>> ts_parser = parser.get_parser()
        """
        if not cls._initialized:
            cls.initialize()

        registry = LanguageParserRegistry()
        existing_parser = registry.get_parser(language)

        if existing_parser is not None:
            return existing_parser

        # Create new parser on demand (lazy creation)
        cls._logger.debug(
            "Creating new parser on demand",
            language=language
        )
        parser = UniversalLanguageParser(language)
        # Verify it works before registering
        parser.get_parser()
        registry.register_parser(language, parser)
        return parser

    @classmethod
    def get_parser_for_file(cls, file_path: str) -> Optional[BaseLanguageParser]:
        """
        Get a parser based on the file extension.

        Determines the programming language from the file extension and returns
        the appropriate parser. Returns None if the file extension is not
        recognized.

        Args:
            file_path: Path to the source file (absolute or relative).

        Returns:
            BaseLanguageParser if a parser is available for the file type,
            None otherwise.

        Example:
            >>> parser = ParserFactory.get_parser_for_file("/path/to/main.py")
            >>> if parser:
            ...     ts_parser = parser.get_parser()
        """
        if not cls._initialized:
            cls.initialize()

        # Extract extension from file path
        path = Path(file_path)
        extension = path.suffix  # e.g., ".py"

        # Handle special files without extensions (Dockerfile, Makefile)
        if not extension:
            extension = path.name

        # Look up language by extension
        language = get_language_by_extension(extension)

        if language is None:
            cls._logger.debug(
                "No parser found for file extension",
                file_path=file_path,
                extension=extension
            )
            return None

        try:
            return cls.get_parser(language)
        except Exception as e:
            cls._logger.warning(
                "Failed to get parser for file",
                file_path=file_path,
                language=language,
                error=str(e)
            )
            return None

    @classmethod
    def is_initialized(cls) -> bool:
        """
        Check if the factory has been initialized.

        Returns:
            True if initialize() has been called successfully, False otherwise.

        Example:
            >>> ParserFactory.is_initialized()
            False
            >>> ParserFactory.initialize()
            >>> ParserFactory.is_initialized()
            True
        """
        return cls._initialized

    @classmethod
    def reset(cls) -> None:
        """
        Reset the factory state for testing purposes.

        This method clears the initialization flag and resets the registry,
        allowing the factory to be re-initialized. This is primarily useful
        for unit tests that need to test initialization behavior.

        Warning:
            This method should only be used in testing. Using it in production
            code may lead to unexpected behavior.

        Example:
            >>> ParserFactory.initialize()
            >>> ParserFactory.is_initialized()
            True
            >>> ParserFactory.reset()
            >>> ParserFactory.is_initialized()
            False
        """
        cls._initialized = False
        LanguageParserRegistry.reset_instance()
        cls._logger.debug("ParserFactory reset")


__all__ = [
    "UniversalLanguageParser",
    "ParserFactory",
]
