"""
Tree-sitter configuration module.

Contains language mappings, file extension associations, and helper functions
for working with supported programming languages.
"""

from typing import Dict, Optional, Set, Tuple


# =============================================================================
# LANGUAGE EXTENSIONS MAPPING
# =============================================================================
# Maps programming language names to their file extensions.
# Includes 40+ popular languages across different domains.

LANGUAGE_EXTENSIONS: Dict[str, Tuple[str, ...]] = {
    # -------------------------------------------------------------------------
    # Popular Languages
    # -------------------------------------------------------------------------
    "python": (".py", ".pyw", ".pyi"),
    "javascript": (".js", ".jsx", ".mjs", ".cjs"),
    "typescript": (".ts", ".tsx", ".mts", ".cts"),
    "java": (".java",),
    "kotlin": (".kt", ".kts"),
    "go": (".go",),
    "rust": (".rs",),
    "c": (".c", ".h"),
    "cpp": (".cpp", ".cc", ".cxx", ".hpp", ".hxx", ".h"),
    "csharp": (".cs",),
    "ruby": (".rb", ".rake"),
    "php": (".php",),
    "swift": (".swift",),
    "scala": (".scala", ".sc"),

    # -------------------------------------------------------------------------
    # Web Technologies
    # -------------------------------------------------------------------------
    "html": (".html", ".htm"),
    "css": (".css",),
    "scss": (".scss",),
    "json": (".json",),
    "yaml": (".yaml", ".yml"),
    "xml": (".xml",),

    # -------------------------------------------------------------------------
    # Scripting Languages
    # -------------------------------------------------------------------------
    "bash": (".sh", ".bash"),
    "powershell": (".ps1", ".psm1"),
    "lua": (".lua",),
    "perl": (".pl", ".pm"),
    "r": (".r", ".R"),

    # -------------------------------------------------------------------------
    # Systems Programming
    # -------------------------------------------------------------------------
    "zig": (".zig",),
    "nim": (".nim",),
    "haskell": (".hs",),
    "elixir": (".ex", ".exs"),
    "erlang": (".erl", ".hrl"),
    "clojure": (".clj", ".cljs", ".cljc"),
    "ocaml": (".ml", ".mli"),
    "fsharp": (".fs", ".fsi", ".fsx"),

    # -------------------------------------------------------------------------
    # Data & Configuration
    # -------------------------------------------------------------------------
    "sql": (".sql",),
    "graphql": (".graphql", ".gql"),
    "toml": (".toml",),
    "dockerfile": ("Dockerfile",),
    "makefile": ("Makefile", "makefile"),

    # -------------------------------------------------------------------------
    # Other Languages
    # -------------------------------------------------------------------------
    "dart": (".dart",),
    "julia": (".jl",),
    "v": (".v",),
    "solidity": (".sol",),
}


# =============================================================================
# LANGUAGES WITH CUSTOM EXTRACTORS
# =============================================================================
# Set of languages that have specialized code extractors.
# Initially empty, will be populated as extractors are implemented.
# Example: after implementing PythonExtractor, add "python" here.

LANGUAGES_WITH_EXTRACTORS: Set[str] = {
    "python",  # PythonExtractor - Issue #19
    "typescript",  # TypeScriptExtractor - Issue #20
    "javascript",  # TypeScriptExtractor - Issue #20 (same extractor)
    "go",  # GoExtractor - Issue #22
}
# Will be populated as extractors are added:
# "java", "rust", etc.


# =============================================================================
# EXTENSION TO LANGUAGE MAPPING (Reverse Lookup)
# =============================================================================
# Generated automatically from LANGUAGE_EXTENSIONS.
# Maps file extension -> language name for quick lookups.
# Note: .h maps to both "c" and "cpp", last one wins (cpp)

EXTENSION_TO_LANGUAGE: Dict[str, str] = {
    ext: lang
    for lang, extensions in LANGUAGE_EXTENSIONS.items()
    for ext in extensions
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_language_by_extension(extension: str) -> Optional[str]:
    """
    Get the programming language name for a given file extension.

    Args:
        extension: File extension (with or without leading dot).
                   Examples: ".py", "py", ".ts", "ts"

    Returns:
        Language name if found, None otherwise.

    Examples:
        >>> get_language_by_extension(".py")
        'python'
        >>> get_language_by_extension("ts")
        'typescript'
        >>> get_language_by_extension(".unknown")
        None
    """
    # Normalize: ensure extension starts with a dot (unless it's a special file)
    if not extension.startswith(".") and extension not in ("Dockerfile", "Makefile", "makefile"):
        extension = f".{extension}"

    return EXTENSION_TO_LANGUAGE.get(extension)


def get_extensions_by_language(language: str) -> Tuple[str, ...]:
    """
    Get all file extensions associated with a programming language.

    Args:
        language: Programming language name (e.g., "python", "typescript").
                  Case-sensitive.

    Returns:
        Tuple of file extensions, or empty tuple if language not found.

    Examples:
        >>> get_extensions_by_language("python")
        ('.py', '.pyw', '.pyi')
        >>> get_extensions_by_language("unknown")
        ()
    """
    return LANGUAGE_EXTENSIONS.get(language, ())


def is_supported_language(language: str) -> bool:
    """
    Check if a programming language is supported.

    Args:
        language: Programming language name (e.g., "python", "go").
                  Case-sensitive.

    Returns:
        True if the language is supported, False otherwise.

    Examples:
        >>> is_supported_language("python")
        True
        >>> is_supported_language("COBOL")
        False
    """
    return language in LANGUAGE_EXTENSIONS


def is_supported_extension(extension: str) -> bool:
    """
    Check if a file extension is supported.

    Args:
        extension: File extension (with or without leading dot).
                   Examples: ".py", "py", ".ts", "ts"

    Returns:
        True if the extension is supported, False otherwise.

    Examples:
        >>> is_supported_extension(".py")
        True
        >>> is_supported_extension("ts")
        True
        >>> is_supported_extension(".unknown")
        False
    """
    # Normalize: ensure extension starts with a dot (unless it's a special file)
    if not extension.startswith(".") and extension not in ("Dockerfile", "Makefile", "makefile"):
        extension = f".{extension}"

    return extension in EXTENSION_TO_LANGUAGE


# =============================================================================
# MODULE STATISTICS (for introspection)
# =============================================================================

def get_config_stats() -> Dict[str, int]:
    """
    Get statistics about the configuration.

    Returns:
        Dictionary with:
        - total_languages: Number of supported languages
        - total_extensions: Number of file extension mappings
        - languages_with_extractors: Number of languages with custom extractors

    Examples:
        >>> stats = get_config_stats()
        >>> stats['total_languages']
        40
    """
    return {
        "total_languages": len(LANGUAGE_EXTENSIONS),
        "total_extensions": len(EXTENSION_TO_LANGUAGE),
        "languages_with_extractors": len(LANGUAGES_WITH_EXTRACTORS),
    }
