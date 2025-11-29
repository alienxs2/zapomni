"""
TokenCounter utility for counting tokens in text.

Provides accurate token counting using tiktoken with graceful fallback
to simple whitespace-based approximation when tiktoken is unavailable.
Caches encoding objects for performance.
"""

from __future__ import annotations

import logging
from typing import Any, List

logger = logging.getLogger(__name__)

try:  # pragma: no cover - third-party import
    import tiktoken

    _TIKTOKEN_AVAILABLE = True
except Exception:  # pragma: no cover - environment without tiktoken
    tiktoken = None  # type: ignore[assignment]
    _TIKTOKEN_AVAILABLE = False


class TokenCounter:
    """
    Counts tokens in text using tiktoken with fallback to whitespace splitting.

    Caches encoding objects for performance and supports different encoding
    types. Default encoding is cl100k_base which matches GPT-4 and Llama.

    Attributes:
        encoding_name: Name of the encoding being used.
        _encoding: Cached encoding object from tiktoken.
        _fallback_mode: Whether using fallback whitespace-based tokenization.
    """

    def __init__(self, encoding: str = "cl100k_base") -> None:
        """
        Initialize TokenCounter with specified encoding.

        Args:
            encoding: Encoding name (default: "cl100k_base" for GPT-4/Llama).

        Note:
            If tiktoken is unavailable or the specified encoding cannot be loaded,
            falls back to simple whitespace-based token approximation.
        """
        self.encoding_name = encoding
        self._encoding: Any = None
        self._fallback_mode = False

        if _TIKTOKEN_AVAILABLE:
            try:
                self._encoding = tiktoken.get_encoding(encoding)
                logger.debug(f"TokenCounter initialized with encoding: {encoding}")
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(
                    f"Failed to load encoding '{encoding}': {exc}. "
                    "Using fallback whitespace-based tokenization."
                )
                self._fallback_mode = True
        else:
            logger.debug("tiktoken not available. Using fallback whitespace-based tokenization.")
            self._fallback_mode = True

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Uses tiktoken if available, otherwise falls back to whitespace splitting.
        Handles empty strings gracefully.

        Args:
            text: Text to count tokens in.

        Returns:
            Number of tokens in the text.
        """
        if not text or not text.strip():
            return 0

        if not self._fallback_mode and self._encoding is not None:
            try:
                encoded = self._encoding.encode(text)
                return len(encoded)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(f"Token counting with tiktoken failed: {exc}. Using fallback.")
                self._fallback_mode = True

        # Fallback: whitespace-based approximation
        return len(text.split())

    def estimate_tokens(self, texts: List[str]) -> List[int]:
        """
        Estimate tokens for a batch of texts.

        Applies count_tokens to each text in the list, maintaining order.
        Efficient for batch operations and integrates well with chunking.

        Args:
            texts: List of texts to estimate tokens for.

        Returns:
            List of token counts corresponding to input texts.

        Example:
            >>> counter = TokenCounter()
            >>> chunks = ["First chunk", "Second chunk", "Third"]
            >>> counts = counter.estimate_tokens(chunks)
            >>> counts
            [3, 3, 1]
        """
        return [self.count_tokens(text) for text in texts]
