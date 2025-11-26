"""
Unit tests for SemanticChunker component.

Covers initialization, text chunking behaviour, input validation,
and merge_small_chunks logic based on level2/level3 specs.
"""

from __future__ import annotations

from typing import List

import pytest

from zapomni_core.chunking.semantic_chunker import SemanticChunker
from zapomni_core.exceptions import ValidationError
from zapomni_db.models import Chunk


class TestSemanticChunkerInit:
    """Tests for SemanticChunker.__init__ configuration and validation."""

    def test_init_defaults_success(self) -> None:
        """Default parameters should be valid and create splitter/tokenizer."""
        chunker = SemanticChunker()

        assert chunker.chunk_size == 512
        assert chunker.chunk_overlap == 50
        assert chunker.min_chunk_size == 100

        # Splitter and tokenizer should be initialized
        assert chunker.splitter is not None
        assert chunker.tokenizer is not None

        # Default separators should be configured
        assert chunker.separators == ["\n\n", "\n", ". ", "! ", "? "]

    def test_init_custom_params_success(self) -> None:
        """Custom configuration parameters should be accepted."""
        chunker = SemanticChunker(
            chunk_size=256,
            chunk_overlap=25,
            min_chunk_size=80,
            separators=["\n\n", "\n"],
        )

        assert chunker.chunk_size == 256
        assert chunker.chunk_overlap == 25
        assert chunker.min_chunk_size == 80
        assert chunker.separators == ["\n\n", "\n"]

    def test_init_chunk_size_too_small_raises(self) -> None:
        """chunk_size < 10 should raise ValueError."""
        with pytest.raises(ValueError):
            SemanticChunker(chunk_size=9)

    def test_init_chunk_size_too_large_raises(self) -> None:
        """chunk_size > 2048 should raise ValueError."""
        with pytest.raises(ValueError):
            SemanticChunker(chunk_size=2049)

    def test_init_chunk_overlap_negative_raises(self) -> None:
        """Negative chunk_overlap should raise ValueError."""
        with pytest.raises(ValueError):
            SemanticChunker(chunk_overlap=-1)

    def test_init_chunk_overlap_too_large_raises(self) -> None:
        """chunk_overlap >= chunk_size should raise ValueError."""
        with pytest.raises(ValueError):
            SemanticChunker(chunk_size=256, chunk_overlap=256)

    def test_init_min_chunk_size_too_small_raises(self) -> None:
        """min_chunk_size < 1 should raise ValueError."""
        with pytest.raises(ValueError):
            SemanticChunker(min_chunk_size=0)

    def test_init_min_chunk_size_too_large_raises(self) -> None:
        """min_chunk_size > chunk_size should raise ValueError."""
        with pytest.raises(ValueError):
            SemanticChunker(chunk_size=200, min_chunk_size=201)

    def test_init_empty_separators_raises(self) -> None:
        """Empty separators list should raise ValueError."""
        with pytest.raises(ValueError):
            SemanticChunker(separators=[])

    def test_init_creates_text_splitter(self) -> None:
        """Splitter instance should be created during initialization."""
        chunker = SemanticChunker()

        assert chunker.splitter is not None


class TestSemanticChunkerChunkText:
    """Tests for SemanticChunker.chunk_text behaviour and validation."""

    def test_chunk_text_basic_success(self) -> None:
        """Chunking normal text should return non-empty list of Chunk."""
        chunker = SemanticChunker()
        text = (
            "Python is a high-level programming language.\n\n"
            "It was created by Guido van Rossum.\n"
            "Python emphasizes readability."
        )

        chunks = chunker.chunk_text(text)

        assert isinstance(chunks, list)
        assert len(chunks) >= 1
        assert all(isinstance(chunk, Chunk) for chunk in chunks)

        # Indices should be sequential starting from 0
        indices = [chunk.index for chunk in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_text_single_character(self) -> None:
        """Single-character text should produce a single chunk."""
        chunker = SemanticChunker()
        text = "A"

        chunks = chunker.chunk_text(text)

        assert len(chunks) == 1
        assert chunks[0].text == "A"
        assert chunks[0].index == 0

    def test_chunk_text_small_text_single_chunk(self) -> None:
        """Text shorter than chunk_size should return a single chunk."""
        chunker = SemanticChunker(chunk_size=1000)
        text = "Short text that should fit into one chunk."

        chunks = chunker.chunk_text(text)

        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].index == 0

    def test_chunk_text_repeated_text_multiple_chunks(self) -> None:
        """Long repeated text with small chunk_size should produce multiple chunks."""
        chunker = SemanticChunker(chunk_size=50, chunk_overlap=10)
        sentence = "Python is a programming language. "
        text = sentence * 100  # Long text

        chunks = chunker.chunk_text(text)

        assert len(chunks) > 1
        assert all(chunk.text for chunk in chunks)

    def test_chunk_text_empty_raises(self) -> None:
        """Empty string should raise ValidationError."""
        chunker = SemanticChunker()

        with pytest.raises(ValidationError):
            chunker.chunk_text("")

    def test_chunk_text_whitespace_only_raises(self) -> None:
        """Whitespace-only text should raise ValidationError."""
        chunker = SemanticChunker()

        with pytest.raises(ValidationError):
            chunker.chunk_text("   \n\t  ")

    def test_chunk_text_max_length_allowed(self) -> None:
        """Text exactly at max length (10_000_000 chars) should be accepted."""
        chunker = SemanticChunker()
        text = "A" * 10_000_000

        chunks = chunker.chunk_text(text)

        assert len(chunks) >= 1

    def test_chunk_text_too_large_raises(self) -> None:
        """Text exceeding max length should raise ValidationError."""
        chunker = SemanticChunker()
        text = "A" * 10_000_001

        with pytest.raises(ValidationError):
            chunker.chunk_text(text)

    def test_chunk_text_non_string_input_raises(self) -> None:
        """Non-string input should raise ValidationError."""
        chunker = SemanticChunker()

        with pytest.raises(ValidationError):
            chunker.chunk_text(b"bytes are not allowed")  # type: ignore[arg-type]

    def test_chunk_text_chunks_cover_original_text(self) -> None:
        """All chunk texts should be substrings of original text."""
        chunker = SemanticChunker(chunk_size=64, chunk_overlap=16)
        text = ("Python is great. " * 50).strip()

        chunks = chunker.chunk_text(text)

        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.text
            assert chunk.text in text

    def test_chunk_text_indices_sequential_after_processing(self) -> None:
        """Indices should remain sequential even after internal merging."""
        chunker = SemanticChunker(chunk_size=64, chunk_overlap=16, min_chunk_size=10)
        text = ("Python is great. " * 40).strip()

        chunks = chunker.chunk_text(text)

        indices = [chunk.index for chunk in chunks]
        assert indices == list(range(len(chunks)))


class TestSemanticChunkerMergeSmallChunks:
    """Tests for merge_small_chunks helper behaviour."""

    def _make_chunk(self, text: str, index: int, start: int, end: int) -> Chunk:
        """Helper to create Chunk with required metadata."""
        return Chunk(text=text, index=index, start_char=start, end_char=end, metadata={})

    def test_merge_small_chunks_raises_on_empty_list(self) -> None:
        """Empty chunks list should raise ValueError."""
        chunker = SemanticChunker()

        with pytest.raises(ValueError):
            chunker.merge_small_chunks([])

    def test_merge_small_chunks_no_small_chunks_returns_same_list(self) -> None:
        """When all chunks are large enough, list should be unchanged."""
        chunker = SemanticChunker(min_chunk_size=5)

        chunks: List[Chunk] = [
            self._make_chunk("A" * 20, 0, 0, 20),
            self._make_chunk("B" * 20, 1, 20, 40),
        ]

        merged = chunker.merge_small_chunks(chunks)

        assert len(merged) == 2
        assert [chunk.text for chunk in merged] == [chunk.text for chunk in chunks]
        assert [chunk.index for chunk in merged] == [0, 1]

    def test_merge_small_chunks_merges_single_small_middle_chunk(self) -> None:
        """Single small chunk in the middle should be merged with a neighbour."""
        chunker = SemanticChunker(min_chunk_size=50)

        chunks: List[Chunk] = [
            self._make_chunk("A" * 100, 0, 0, 100),
            self._make_chunk("B" * 10, 1, 100, 110),
            self._make_chunk("C" * 100, 2, 110, 210),
        ]

        merged = chunker.merge_small_chunks(chunks)

        assert len(merged) == 2
        assert merged[0].index == 0
        assert merged[1].index == 1
        # Ensure small chunk text was merged into one of the neighbours
        combined_texts = "".join(chunk.text for chunk in merged)
        assert "B" * 10 in combined_texts

    def test_merge_small_chunks_merges_all_small_chunks(self) -> None:
        """All-small chunk list should be merged into a single chunk."""
        chunker = SemanticChunker(min_chunk_size=50)

        chunks: List[Chunk] = [
            self._make_chunk("A" * 10, 0, 0, 10),
            self._make_chunk("B" * 10, 1, 10, 20),
            self._make_chunk("C" * 10, 2, 20, 30),
        ]

        merged = chunker.merge_small_chunks(chunks)

        assert len(merged) == 1
        assert merged[0].index == 0
        assert "A" * 10 in merged[0].text
        assert "B" * 10 in merged[0].text
        assert "C" * 10 in merged[0].text

    def test_merge_small_chunks_merges_last_small_chunk_with_previous(self) -> None:
        """Last small chunk should be merged with previous one."""
        chunker = SemanticChunker(min_chunk_size=50)

        chunks: List[Chunk] = [
            self._make_chunk("A" * 100, 0, 0, 100),
            self._make_chunk("B" * 10, 1, 100, 110),
        ]

        merged = chunker.merge_small_chunks(chunks)

        assert len(merged) == 1
        assert merged[0].index == 0
        assert "A" * 100 in merged[0].text
        assert "B" * 10 in merged[0].text

    def test_merge_small_chunks_preserves_order(self) -> None:
        """Order of content should be preserved after merging."""
        chunker = SemanticChunker(min_chunk_size=50)

        chunks: List[Chunk] = [
            self._make_chunk("AAA", 0, 0, 3),
            self._make_chunk("BBB", 1, 3, 6),
            self._make_chunk("CCC" * 30, 2, 6, 96),
        ]

        merged = chunker.merge_small_chunks(chunks)

        assert len(merged) == 2
        assert merged[0].index == 0
        assert merged[1].index == 1
        assert merged[0].text.startswith("AAA")
        assert "BBB" in merged[0].text
