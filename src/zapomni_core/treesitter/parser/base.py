"""
Base classes for Tree-sitter language parsers.

Provides abstract base class for implementing language-specific parsers
using tree-sitter and tree-sitter-language-pack.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import structlog
from tree_sitter import Parser, Tree
from tree_sitter_language_pack import get_language, get_parser

from zapomni_core.treesitter.exceptions import (
    LanguageNotSupportedError,
    ParseError,
)

logger = structlog.get_logger(__name__)


class BaseLanguageParser(ABC):
    """
    Abstract base class for language-specific tree-sitter parsers.

    Provides common parsing functionality using tree-sitter-language-pack.
    Subclasses must define the language name and supported file extensions.

    Example:
        class PythonParser(BaseLanguageParser):
            @property
            def language_name(self) -> str:
                return "python"

            @property
            def file_extensions(self) -> tuple[str, ...]:
                return (".py", ".pyw", ".pyi")

        parser = PythonParser()
        tree = parser.parse(b"def hello(): pass")
    """

    def __init__(self) -> None:
        """
        Initialize the parser with tree-sitter language support.

        Creates and configures the tree-sitter Parser for the specific language.

        Raises:
            LanguageNotSupportedError: If the language is not supported by
                tree-sitter-language-pack.
        """
        self._parser: Optional[Parser] = None
        self._log = logger.bind(parser=self.__class__.__name__)

        # Validate language is supported
        try:
            # Test that language exists in tree-sitter-language-pack
            _ = get_language(self.language_name)  # type: ignore[arg-type]
            self._log.debug("language_validated", language=self.language_name)
        except Exception as e:
            self._log.error(
                "language_not_supported",
                language=self.language_name,
                error=str(e),
            )
            raise LanguageNotSupportedError(
                language=self.language_name,
                details={"error": str(e)},
            ) from e

    @property
    @abstractmethod
    def language_name(self) -> str:
        """
        Get the language name for this parser.

        Returns:
            Language name as used by tree-sitter-language-pack
            (e.g., "python", "javascript", "rust").
        """
        ...

    @property
    @abstractmethod
    def file_extensions(self) -> tuple[str, ...]:
        """
        Get the file extensions supported by this parser.

        Returns:
            Tuple of file extensions including the dot
            (e.g., (".py", ".pyw", ".pyi")).
        """
        ...

    def get_parser(self) -> Parser:
        """
        Get the configured tree-sitter Parser instance.

        Lazily initializes the parser on first call. Uses get_parser from
        tree-sitter-language-pack which returns a pre-configured Parser.

        Returns:
            Configured tree-sitter Parser for this language.
        """
        if self._parser is None:
            self._parser = get_parser(self.language_name)
            self._log.debug("parser_initialized", language=self.language_name)
        return self._parser

    def parse(self, source_code: bytes) -> Optional[Tree]:
        """
        Parse source code into an AST tree.

        Args:
            source_code: Source code as bytes (UTF-8 encoded).

        Returns:
            Parsed AST Tree if successful, None if parsing fails.

        Example:
            tree = parser.parse(b"def hello(): pass")
            if tree:
                print(tree.root_node.type)  # "module"
        """
        try:
            parser = self.get_parser()
            tree = parser.parse(source_code)
            self._log.debug(
                "parse_success",
                language=self.language_name,
                source_length=len(source_code),
                has_errors=tree.root_node.has_error if tree else True,
            )
            return tree
        except Exception as e:
            self._log.error(
                "parse_failed",
                language=self.language_name,
                source_length=len(source_code),
                error=str(e),
            )
            return None

    def parse_file(self, file_path: str) -> Optional[Tree]:
        """
        Parse a source file into an AST tree.

        Reads the file content and parses it using the configured parser.

        Args:
            file_path: Absolute or relative path to the source file.

        Returns:
            Parsed AST Tree if successful, None if reading or parsing fails.

        Raises:
            ParseError: If the file cannot be read (not found, permission denied).

        Example:
            tree = parser.parse_file("/path/to/script.py")
            if tree:
                for child in tree.root_node.children:
                    print(child.type)
        """
        path = Path(file_path)

        # Check file exists
        if not path.exists():
            self._log.error(
                "file_not_found",
                file_path=file_path,
            )
            raise ParseError(
                file_path=file_path,
                parse_details="File not found",
            )

        # Check file is readable
        if not path.is_file():
            self._log.error(
                "not_a_file",
                file_path=file_path,
            )
            raise ParseError(
                file_path=file_path,
                parse_details="Path is not a file",
            )

        try:
            source_code = path.read_bytes()
            self._log.debug(
                "file_read",
                file_path=file_path,
                size_bytes=len(source_code),
            )
        except PermissionError as e:
            self._log.error(
                "permission_denied",
                file_path=file_path,
                error=str(e),
            )
            raise ParseError(
                file_path=file_path,
                parse_details="Permission denied",
            ) from e
        except OSError as e:
            self._log.error(
                "read_error",
                file_path=file_path,
                error=str(e),
            )
            raise ParseError(
                file_path=file_path,
                parse_details=f"Failed to read file: {e}",
            ) from e

        return self.parse(source_code)

    def supports_extension(self, extension: str) -> bool:
        """
        Check if this parser supports a given file extension.

        Args:
            extension: File extension with or without leading dot
                (e.g., ".py" or "py").

        Returns:
            True if the extension is supported by this parser.

        Example:
            parser.supports_extension(".py")  # True
            parser.supports_extension("py")   # True
            parser.supports_extension(".js")  # False
        """
        # Normalize extension
        if not extension.startswith("."):
            extension = f".{extension}"
        return extension in self.file_extensions

    def __repr__(self) -> str:
        """Return string representation of the parser."""
        return f"{self.__class__.__name__}(language='{self.language_name}')"


__all__ = ["BaseLanguageParser"]
