"""
Tree-sitter AST extraction module for Zapomni.

Provides high-quality code parsing and extraction using Tree-sitter
for 165+ programming languages.

Key components:
- config: Language mappings and configuration
- models: Data models for extracted code elements
- exceptions: Tree-sitter specific exceptions
- parser: Language parser registry and factory
- extractors: Code element extractors
"""

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
from zapomni_core.treesitter.exceptions import (
    ExtractorNotFoundError,
    LanguageNotSupportedError,
    ParseError,
    TreeSitterError,
)
from zapomni_core.treesitter.extractors import (
    CLASS_NODE_TYPES,
    FUNCTION_NODE_TYPES,
    BaseCodeExtractor,
    GenericExtractor,
)
from zapomni_core.treesitter.models import (
    ASTNodeLocation,
    CodeElementType,
    ExtractedCode,
    ParameterInfo,
    ParseResult,
)
from zapomni_core.treesitter.parser import (
    BaseLanguageParser,
    LanguageParserRegistry,
    ParserFactory,
    UniversalLanguageParser,
)

__all__ = [
    # Config constants
    "LANGUAGE_EXTENSIONS",
    "LANGUAGES_WITH_EXTRACTORS",
    "EXTENSION_TO_LANGUAGE",
    # Config functions
    "get_language_by_extension",
    "get_extensions_by_language",
    "is_supported_language",
    "is_supported_extension",
    "get_config_stats",
    # Exceptions
    "TreeSitterError",
    "LanguageNotSupportedError",
    "ParseError",
    "ExtractorNotFoundError",
    # Models
    "CodeElementType",
    "ASTNodeLocation",
    "ParameterInfo",
    "ExtractedCode",
    "ParseResult",
    # Parser
    "BaseLanguageParser",
    "LanguageParserRegistry",
    "ParserFactory",
    "UniversalLanguageParser",
    # Extractors
    "BaseCodeExtractor",
    "GenericExtractor",
    "FUNCTION_NODE_TYPES",
    "CLASS_NODE_TYPES",
]
