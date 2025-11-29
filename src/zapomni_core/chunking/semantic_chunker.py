"""
SemanticChunker implementation.

Splits text into semantically meaningful chunks using configurable size and
overlap. Integrates with zapomni_db Chunk model and uses langchain/tiktoken
when available, with lightweight fallbacks for local testing environments.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence

from zapomni_core.exceptions import ProcessingError, ValidationError
from zapomni_db.models import Chunk

logger = logging.getLogger(__name__)

try:  # pragma: no cover - third-party import
    import tiktoken

    _TIKTOKEN_AVAILABLE = True
except Exception:  # pragma: no cover - environment without tiktoken
    tiktoken = None  # type: ignore[assignment]
    _TIKTOKEN_AVAILABLE = False

try:  # pragma: no cover - third-party import
    from langchain.text_splitter import (  # type: ignore[import-not-found]
        RecursiveCharacterTextSplitter,
    )

    _LANGCHAIN_AVAILABLE = True
except Exception:  # pragma: no cover - environment without langchain
    RecursiveCharacterTextSplitter = None  # type: ignore[assignment]
    _LANGCHAIN_AVAILABLE = False


class SimpleTokenizer:
    """Fallback tokenizer used when tiktoken is not available.

    Uses a simple whitespace split to approximate token counts.
    """

    def encode(self, text: str) -> List[str]:
        return text.split()


@dataclass
class SimpleRecursiveSplitter:
    """Fallback splitter used when langchain is not available.

    Splits text into overlapping character windows based on chunk_size and
    chunk_overlap. This keeps behaviour predictable for tests without
    requiring external dependencies.
    """

    chunk_size: int
    chunk_overlap: int
    separators: Sequence[str]
    length_function: Optional[Callable[[str], int]] = None

    def split_text(self, text: str) -> List[str]:
        if not text:
            return []

        if self.chunk_size <= 0:
            return [text]

        if len(text) <= self.chunk_size:
            return [text]

        chunks: List[str] = []
        step = max(self.chunk_size - self.chunk_overlap, 1)
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start = start + step

        return chunks


class SemanticChunker:
    """
    Intelligent text chunking with semantic boundary detection.

    Uses LangChain's RecursiveCharacterTextSplitter when available to split
    text at natural boundaries while maintaining target chunk size. Includes
    configurable overlap to preserve context at chunk boundaries.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: Optional[int] = None,
        separators: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize SemanticChunker with chunking configuration.

        Args:
            chunk_size: Target chunk size in tokens (default: 512).
            chunk_overlap: Overlap between chunks in tokens (default: 50).
            min_chunk_size: Minimum chunk size in tokens (default: chunk_size // 5).
                If None, computed as chunk_size // 5, with minimum of 1.
            separators: Custom separators; defaults to paragraph/sentence
                boundaries if None.

        Raises:
            ValueError: If configuration values are out of allowed ranges.
        """
        if chunk_size < 10 or chunk_size > 2048:
            raise ValueError("chunk_size must be between 10 and 2048")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

        # Use provided min_chunk_size or compute default
        if min_chunk_size is None:
            # Default to 100 for normal usage, or proportional for small chunk_size
            min_chunk_size = 100 if chunk_size >= 100 else max(1, chunk_size // 5)
        else:
            # Explicit min_chunk_size: allow down to 1 for testing flexibility
            if min_chunk_size < 1 or min_chunk_size > chunk_size:
                raise ValueError("min_chunk_size must be between 1 and chunk_size")
        if separators is not None and len(separators) == 0:
            raise ValueError("separators cannot be empty")

        logger.debug(
            f"Initializing SemanticChunker: chunk_size={chunk_size}, overlap={chunk_overlap}"
        )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.separators: List[str] = separators or ["\n\n", "\n", ". ", "! ", "? "]

        if _TIKTOKEN_AVAILABLE:  # pragma: no cover - external dependency
            self.tokenizer: Any = tiktoken.get_encoding("cl100k_base")
        else:
            self.tokenizer = SimpleTokenizer()

        if _LANGCHAIN_AVAILABLE and RecursiveCharacterTextSplitter is not None:  # pragma: no cover
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=self._count_tokens,
                separators=self.separators,
            )
        else:
            self.splitter = SimpleRecursiveSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=self.separators,
                length_function=self._count_tokens,
            )

    def _validate_input(self, text: str) -> None:
        """
        Validate text input before chunking.

        Args:
            text: Text to validate.

        Raises:
            ValidationError: If validation fails.
        """
        if not isinstance(text, str):
            raise ValidationError("Text must be a string")

        stripped = text.strip()
        if not stripped:
            raise ValidationError("Text cannot be empty")

        if len(text) > 10_000_000:
            raise ValidationError(f"Text exceeds maximum length (10,000,000 chars): {len(text)}")

    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken when available.

        Falls back to simple whitespace-based token counting if tiktoken
        is not installed.
        """
        if not text:
            return 0

        try:
            encode = getattr(self.tokenizer, "encode", None)
            if encode is not None:
                encoded = encode(text)
                return len(encoded)
        except Exception:  # pragma: no cover - defensive
            logger.debug("Token counting failed with tokenizer; falling back to split()")

        return len(text.split())

    def chunk_text(self, text: str) -> List[Chunk]:
        """
        Split text into semantic chunks with overlap.

        Args:
            text: Input text to chunk (max 10,000,000 characters).

        Returns:
            List of Chunk objects with sequential indices and character offsets.

        Raises:
            ValidationError: If text is invalid.
            ProcessingError: If chunking fails due to internal error.
        """
        self._validate_input(text)

        logger.debug(f"Chunking text of {len(text)} chars")

        try:
            raw_chunks = self.splitter.split_text(text)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Text splitting failed: %s", exc)
            raise ProcessingError(
                message=f"Failed to split text: {exc}",
                error_code="PROC_001",
                original_exception=exc,
            )

        if not raw_chunks:
            raise ProcessingError("Chunking produced empty result", error_code="PROC_001")

        logger.debug(f"Initial split created {len(raw_chunks)} chunks")

        chunks: List[Chunk] = []
        current_offset = 0

        for index, chunk_text in enumerate(raw_chunks):
            if not chunk_text:
                continue

            start_char = text.find(chunk_text, current_offset)
            if start_char == -1:
                # Fallback to global search if overlapping chunks confuse the offset
                start_char = text.find(chunk_text)
                if start_char == -1:
                    raise ProcessingError(
                        message=f"Chunk positioning error at index {index}",
                        error_code="PROC_001",
                    )

            end_char = start_char + len(chunk_text)

            chunk = Chunk(
                text=chunk_text,
                index=index,
                start_char=start_char,
                end_char=end_char,
                metadata={},
            )
            chunks.append(chunk)

            # Advance offset; overlap is handled by the splitter itself
            current_offset = end_char

        merged_chunks = self.merge_small_chunks(chunks)

        logger.info(f"Successfully chunked text into {len(merged_chunks)} chunks")

        return merged_chunks

    def merge_small_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Merge chunks smaller than min_chunk_size with adjacent chunks.

        Args:
            chunks: List of chunks to process.

        Returns:
            List of chunks where all chunks are at least min_chunk_size tokens,
            except possibly the last one.

        Raises:
            ValueError: If chunks list is empty.
        """
        if not chunks:
            raise ValueError("chunks cannot be empty")

        if len(chunks) == 1:
            # Single chunk - nothing to merge
            single = chunks[0]
            return [
                Chunk(
                    text=single.text,
                    index=0,
                    start_char=single.start_char,
                    end_char=single.end_char,
                    metadata=single.metadata,
                )
            ]

        merged: List[Chunk] = []
        i = 0

        while i < len(chunks):
            current = chunks[i]
            # Use character length as the primary measure for merge decisions
            # This is more reliable when chunks have position metadata
            chunk_size = len(current.text)
            is_last = i == len(chunks) - 1

            # Determine if chunk is large enough
            chunk_is_large = chunk_size >= self.min_chunk_size

            # If current chunk is large enough, append as-is
            if chunk_is_large:
                merged.append(current)
                i += 1
                continue

            # Current chunk is small
            if not is_last:
                # Merge with next chunk (regardless of next chunk size)
                next_chunk = chunks[i + 1]
                merged_text = current.text + next_chunk.text
                merged_chunk = Chunk(
                    text=merged_text,
                    index=0,  # reindexed later
                    start_char=current.start_char,
                    end_char=next_chunk.end_char,
                    metadata=(current.metadata or {}) | (next_chunk.metadata or {}),
                )
                merged.append(merged_chunk)
                i += 2  # Skip both current and next chunk - they've been merged
            else:
                # Last chunk is small: merge with previous merged chunk if possible
                if merged:
                    previous = merged.pop()
                    merged_text = previous.text + current.text
                    merged_chunk = Chunk(
                        text=merged_text,
                        index=0,
                        start_char=previous.start_char,
                        end_char=current.end_char,
                        metadata=(previous.metadata or {}) | (current.metadata or {}),
                    )
                    merged.append(merged_chunk)
                else:
                    # No previous chunk, so append current as-is (even though small)
                    merged.append(current)
                i += 1

        # Reindex sequentially without mutating original Chunk instances
        reindexed: List[Chunk] = []
        for new_index, chunk in enumerate(merged):
            reindexed.append(
                Chunk(
                    text=chunk.text,
                    index=new_index,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    metadata=chunk.metadata,
                )
            )

        logger.debug(f"Merging small chunks: input={len(chunks)}, output={len(reindexed)}")

        return reindexed

    def chunk_code(self, code: str, language: str = "python") -> List[Chunk]:
        """
        Split code using AST-based chunking (Phase 3 feature).

        Currently not implemented; reserved for future work.
        """
        raise NotImplementedError("AST-based code chunking is not implemented yet")
