"""
Tree-sitter parser module.

Provides base classes and utilities for language-specific parsing using tree-sitter.

Exports:
    BaseLanguageParser: Abstract base class for language parsers.
    LanguageParserRegistry: Central registry for parsers and extractors.
    ParserFactory: Factory for creating and managing language parsers.
    UniversalLanguageParser: Universal parser for any supported language.
"""

from zapomni_core.treesitter.parser.base import BaseLanguageParser
from zapomni_core.treesitter.parser.factory import ParserFactory, UniversalLanguageParser
from zapomni_core.treesitter.parser.registry import LanguageParserRegistry

__all__ = [
    "BaseLanguageParser",
    "LanguageParserRegistry",
    "ParserFactory",
    "UniversalLanguageParser",
]
