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
    LANGUAGE_EXTENSIONS,
    LANGUAGES_WITH_EXTRACTORS,
    EXTENSION_TO_LANGUAGE,
    get_language_by_extension,
    get_extensions_by_language,
    is_supported_language,
    is_supported_extension,
    get_config_stats,
)
from zapomni_core.treesitter.exceptions import (
    TreeSitterError,
    LanguageNotSupportedError,
    ParseError,
    ExtractorNotFoundError,
)
from zapomni_core.treesitter.models import (
    CodeElementType,
    ASTNodeLocation,
    ParameterInfo,
    ExtractedCode,
    ParseResult,
)
from zapomni_core.treesitter.parser import (
    BaseLanguageParser,
    LanguageParserRegistry,
    ParserFactory,
    UniversalLanguageParser,
)
from zapomni_core.treesitter.extractors import (
    BaseCodeExtractor,
    GenericExtractor,
    FUNCTION_NODE_TYPES,
    CLASS_NODE_TYPES,
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
