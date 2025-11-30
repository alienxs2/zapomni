"""
CodeTokenizer - Code-aware tokenizer for BM25 search.

Provides specialized tokenization for code and technical content:
- Split camelCase: "calculateUserScore" -> ["calculate", "user", "score"]
- Split snake_case: "calculate_user_score" -> ["calculate", "user", "score"]
- Handle acronyms: "HTTPResponse" -> ["http", "response"]
- Preserve programming keywords

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

import re
from typing import List, Optional, Set

import structlog

logger = structlog.get_logger(__name__)


class CodeTokenizer:
    """
    Code-aware tokenizer for improved BM25 search on source code.

    This tokenizer understands programming conventions and produces
    better tokens for code search than naive whitespace splitting.

    Features:
        - CamelCase splitting: "getUserName" -> ["get", "user", "name"]
        - snake_case splitting: "get_user_name" -> ["get", "user", "name"]
        - Acronym handling: "HTTPResponse" -> ["http", "response"]
        - Programming keyword preservation
        - Numeric token handling

    Example:
        ```python
        tokenizer = CodeTokenizer()

        # CamelCase
        tokens = tokenizer.tokenize("calculateUserScore")
        # -> ["calculate", "user", "score"]

        # snake_case
        tokens = tokenizer.tokenize("calculate_user_score")
        # -> ["calculate", "user", "score"]

        # Mixed
        tokens = tokenizer.tokenize("def getUserHTTPResponse(user_id):")
        # -> ["def", "get", "user", "http", "response", "user", "id"]
        ```
    """

    # Common programming keywords to preserve
    PROGRAMMING_KEYWORDS: Set[str] = {
        # Python
        "def",
        "class",
        "import",
        "from",
        "return",
        "yield",
        "async",
        "await",
        "if",
        "elif",
        "else",
        "for",
        "while",
        "try",
        "except",
        "finally",
        "with",
        "as",
        "lambda",
        "pass",
        "break",
        "continue",
        "raise",
        "assert",
        "global",
        "nonlocal",
        "del",
        "true",
        "false",
        "none",
        "and",
        "or",
        "not",
        "in",
        "is",
        # JavaScript/TypeScript
        "const",
        "let",
        "var",
        "function",
        "interface",
        "type",
        "export",
        "default",
        "extends",
        "implements",
        "new",
        "this",
        "super",
        "static",
        "public",
        "private",
        "protected",
        "readonly",
        "abstract",
        "void",
        "null",
        "undefined",
        "typeof",
        "instanceof",
        # Go
        "func",
        "package",
        "struct",
        "chan",
        "select",
        "case",
        "defer",
        "go",
        "range",
        "map",
        "make",
        "append",
        "len",
        "cap",
        # Rust
        "fn",
        "impl",
        "trait",
        "enum",
        "mod",
        "use",
        "pub",
        "mut",
        "ref",
        "self",
        "match",
        "loop",
        "move",
        "box",
        "where",
        "unsafe",
        # Common types
        "int",
        "str",
        "string",
        "bool",
        "float",
        "double",
        "byte",
        "char",
        "list",
        "dict",
        "array",
        "tuple",
        "set",
        "vector",
        "hashmap",
        "option",
        "result",
        "error",
        # Common terms
        "api",
        "http",
        "url",
        "uri",
        "json",
        "xml",
        "html",
        "css",
        "sql",
        "db",
        "io",
        "ui",
        "id",
        "uuid",
        "async",
        "sync",
        "config",
        "env",
        "test",
        "mock",
        "stub",
        "spec",
    }

    # Minimum token length to keep (except keywords)
    MIN_TOKEN_LENGTH: int = 2

    def __init__(
        self,
        min_token_length: int = 2,
        preserve_keywords: bool = True,
        custom_keywords: Optional[Set[str]] = None,
    ) -> None:
        """
        Initialize CodeTokenizer.

        Args:
            min_token_length: Minimum length for tokens (default: 2)
            preserve_keywords: Whether to preserve programming keywords (default: True)
            custom_keywords: Additional keywords to preserve

        Example:
            ```python
            # Default tokenizer
            tokenizer = CodeTokenizer()

            # Custom settings
            tokenizer = CodeTokenizer(
                min_token_length=3,
                custom_keywords={"myKeyword", "customTerm"}
            )
            ```
        """
        self.min_token_length = min_token_length
        self.preserve_keywords = preserve_keywords
        self.keywords = self.PROGRAMMING_KEYWORDS.copy()

        if custom_keywords:
            self.keywords.update(kw.lower() for kw in custom_keywords)

        logger.debug(
            "code_tokenizer_initialized",
            min_token_length=min_token_length,
            preserve_keywords=preserve_keywords,
            custom_keywords_count=len(custom_keywords) if custom_keywords else 0,
        )

    def _split_camel_case(self, word: str) -> List[str]:
        """
        Split camelCase and PascalCase identifiers.

        Handles:
            - camelCase: "getUserName" -> ["get", "User", "Name"]
            - PascalCase: "UserService" -> ["User", "Service"]
            - Acronyms: "HTTPResponse" -> ["HTTP", "Response"]
            - Mixed: "getHTTPResponse" -> ["get", "HTTP", "Response"]

        Args:
            word: Word to split

        Returns:
            List of split parts (not yet lowercased)
        """
        if not word:
            return []

        # Pattern to split on:
        # - Before uppercase letter preceded by lowercase
        # - Before uppercase letter followed by lowercase (for acronyms)
        # This handles: "getUserName" -> ["get", "User", "Name"]
        # And: "HTTPResponse" -> ["HTTP", "Response"]
        # And: "getHTTPResponse" -> ["get", "HTTP", "Response"]

        # First, insert markers before transitions
        # lowercase to uppercase: aB -> a|B
        result = re.sub(r"([a-z])([A-Z])", r"\1|\2", word)
        # uppercase to uppercase+lowercase: ABc -> A|Bc (for acronyms like HTTP)
        result = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1|\2", result)

        parts = result.split("|")
        return [p for p in parts if p]

    def _split_snake_case(self, word: str) -> List[str]:
        """
        Split snake_case and SCREAMING_SNAKE_CASE identifiers.

        Handles:
            - snake_case: "get_user_name" -> ["get", "user", "name"]
            - SCREAMING_SNAKE_CASE: "MAX_RETRY_COUNT" -> ["MAX", "RETRY", "COUNT"]
            - Mixed: "get_HTTP_response" -> ["get", "HTTP", "response"]

        Args:
            word: Word to split

        Returns:
            List of split parts
        """
        if not word:
            return []

        # Split on underscores
        parts = word.split("_")
        return [p for p in parts if p]

    def _split_identifier(self, identifier: str) -> List[str]:
        """
        Split a programming identifier into components.

        Combines camelCase and snake_case splitting, handling mixed styles.

        Args:
            identifier: Identifier to split

        Returns:
            List of lowercased component parts
        """
        if not identifier:
            return []

        # First split by snake_case
        snake_parts = self._split_snake_case(identifier)

        # Then split each part by camelCase
        result: List[str] = []
        for part in snake_parts:
            camel_parts = self._split_camel_case(part)
            result.extend(camel_parts)

        # Lowercase all parts
        return [p.lower() for p in result if p]

    def _is_valid_token(self, token: str) -> bool:
        """
        Check if a token is valid for indexing.

        A token is valid if:
            - It's a programming keyword (regardless of length)
            - It meets the minimum length requirement
            - It contains at least one alphanumeric character

        Args:
            token: Token to validate

        Returns:
            True if token is valid
        """
        if not token:
            return False

        # Always keep programming keywords
        if self.preserve_keywords and token.lower() in self.keywords:
            return True

        # Check minimum length
        if len(token) < self.min_token_length:
            return False

        # Must contain at least one alphanumeric character
        return any(c.isalnum() for c in token)

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text with code-aware splitting.

        Process:
            1. Replace punctuation with spaces (preserve underscores temporarily)
            2. Split on whitespace
            3. For each word, split by snake_case then camelCase
            4. Lowercase all tokens
            5. Filter by minimum length and validity

        Args:
            text: Text to tokenize

        Returns:
            List of lowercase tokens

        Example:
            ```python
            tokenizer = CodeTokenizer()

            # Code-aware tokenization
            tokens = tokenizer.tokenize("def getUserHTTPResponse(user_id):")
            # -> ["def", "get", "user", "http", "response", "user", "id"]

            # Natural text
            tokens = tokenizer.tokenize("Python is a programming language")
            # -> ["python", "is", "programming", "language"]
            ```
        """
        if not text:
            return []

        # Replace most punctuation with spaces, but keep underscores for now
        # to properly handle snake_case
        cleaned = re.sub(r"[^\w\s]", " ", text)

        # Split on whitespace
        words = cleaned.split()

        # Process each word
        tokens: List[str] = []
        for word in words:
            # Split the identifier
            parts = self._split_identifier(word)

            # Filter and add valid tokens
            for part in parts:
                if self._is_valid_token(part):
                    tokens.append(part.lower())

        logger.debug(
            "text_tokenized",
            input_length=len(text),
            token_count=len(tokens),
        )

        return tokens

    def tokenize_batch(self, texts: List[str]) -> List[List[str]]:
        """
        Tokenize multiple texts efficiently.

        Args:
            texts: List of texts to tokenize

        Returns:
            List of token lists, one per input text

        Example:
            ```python
            tokenizer = CodeTokenizer()
            texts = ["getUserName", "calculate_score", "HTTPClient"]
            all_tokens = tokenizer.tokenize_batch(texts)
            # -> [["get", "user", "name"], ["calculate", "score"], ["http", "client"]]
            ```
        """
        return [self.tokenize(text) for text in texts]
