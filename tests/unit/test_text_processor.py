"""
Unit tests for TextProcessor component.

Tests the orchestration of text processing pipeline:
chunking → embedding → storage.

Follows TDD approach with comprehensive mocking of dependencies.
"""

import uuid
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from zapomni_core.exceptions import (
    DatabaseError,
    EmbeddingError,
    ProcessingError,
    ValidationError,
)
from zapomni_core.processors.text_processor import TextProcessor
from zapomni_db.models import Chunk, Memory


class TestTextProcessorInit:
    """Tests for TextProcessor.__init__ configuration and validation."""

    def test_init_defaults_success(self):
        """Default initialization should create instance with dependencies."""
        processor = TextProcessor()

        assert processor is not None
        assert processor.chunker is not None
        assert processor.embedder is not None
        assert processor.db_client is not None
        assert processor.logger is not None

    def test_init_custom_dependencies(self):
        """Custom dependencies should be accepted."""
        mock_chunker = MagicMock()
        mock_embedder = AsyncMock()
        mock_db_client = AsyncMock()

        processor = TextProcessor(
            chunker=mock_chunker, embedder=mock_embedder, db_client=mock_db_client
        )

        assert processor.chunker is mock_chunker
        assert processor.embedder is mock_embedder
        assert processor.db_client is mock_db_client


class TestTextProcessorAddText:
    """Tests for TextProcessor.add_text() method."""

    @pytest.mark.asyncio
    async def test_add_text_success_happy_path(self):
        """Happy path: text → chunks → embeddings → storage → memory_id."""
        # Setup mocks
        mock_chunker = MagicMock()
        mock_embedder = AsyncMock()
        mock_db_client = AsyncMock()

        # Mock chunker output
        chunk1 = Chunk(text="Python is great.", index=0, start_char=0, end_char=16, metadata={})
        chunk2 = Chunk(text="It's easy to learn.", index=1, start_char=17, end_char=36, metadata={})
        mock_chunker.chunk_text.return_value = [chunk1, chunk2]

        # Mock embedder output
        embedding1 = [0.1] * 768
        embedding2 = [0.2] * 768
        mock_embedder.embed_batch.return_value = [embedding1, embedding2]

        # Mock database output
        expected_memory_id = str(uuid.uuid4())
        mock_db_client.add_memory.return_value = expected_memory_id

        # Create processor with mocks
        processor = TextProcessor(
            chunker=mock_chunker, embedder=mock_embedder, db_client=mock_db_client
        )

        # Execute
        text = "Python is great. It's easy to learn."
        metadata = {"source": "test", "author": "tester"}
        memory_id = await processor.add_text(text, metadata)

        # Verify
        assert memory_id == expected_memory_id

        # Verify chunker was called
        mock_chunker.chunk_text.assert_called_once_with(text)

        # Verify embedder was called with chunk texts
        mock_embedder.embed_batch.assert_called_once()
        call_args = mock_embedder.embed_batch.call_args
        assert call_args[0][0] == ["Python is great.", "It's easy to learn."]

        # Verify database was called with Memory
        mock_db_client.add_memory.assert_called_once()
        memory_arg = mock_db_client.add_memory.call_args[0][0]
        assert isinstance(memory_arg, Memory)
        assert memory_arg.text == text
        assert memory_arg.metadata == metadata
        assert len(memory_arg.chunks) == 2
        assert len(memory_arg.embeddings) == 2

    @pytest.mark.asyncio
    async def test_add_text_empty_text_raises_validation_error(self):
        """Empty text should raise ValidationError."""
        processor = TextProcessor()

        with pytest.raises(ValidationError) as exc_info:
            await processor.add_text("", {})

        assert "text cannot be empty" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_add_text_whitespace_only_raises_validation_error(self):
        """Whitespace-only text should raise ValidationError."""
        processor = TextProcessor()

        with pytest.raises(ValidationError) as exc_info:
            await processor.add_text("   \n\t  ", {})

        assert "text cannot be empty" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_add_text_non_string_raises_validation_error(self):
        """Non-string text should raise ValidationError."""
        processor = TextProcessor()

        with pytest.raises(ValidationError):
            await processor.add_text(123, {})  # type: ignore

    @pytest.mark.asyncio
    async def test_add_text_invalid_metadata_raises_validation_error(self):
        """Non-dict metadata should raise ValidationError."""
        processor = TextProcessor()

        with pytest.raises(ValidationError):
            await processor.add_text("Valid text", "invalid metadata")  # type: ignore

    @pytest.mark.asyncio
    async def test_add_text_chunking_fails_raises_processing_error(self):
        """Chunking failure should raise ProcessingError."""
        mock_chunker = MagicMock()
        mock_chunker.chunk_text.side_effect = ProcessingError(
            message="Chunking failed", error_code="PROC_001"
        )

        processor = TextProcessor(chunker=mock_chunker)

        with pytest.raises(ProcessingError) as exc_info:
            await processor.add_text("Valid text", {})

        assert "chunking failed" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_add_text_embedding_fails_raises_embedding_error(self):
        """Embedding failure should raise EmbeddingError."""
        mock_chunker = MagicMock()
        mock_embedder = AsyncMock()

        chunk = Chunk(text="Test", index=0, start_char=0, end_char=4, metadata={})
        mock_chunker.chunk_text.return_value = [chunk]
        mock_embedder.embed_batch.side_effect = EmbeddingError(
            message="Embedding failed", error_code="EMB_001"
        )

        processor = TextProcessor(chunker=mock_chunker, embedder=mock_embedder)

        with pytest.raises(EmbeddingError) as exc_info:
            await processor.add_text("Test text", {})

        assert "embedding failed" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_add_text_storage_fails_raises_database_error(self):
        """Storage failure should raise DatabaseError."""
        mock_chunker = MagicMock()
        mock_embedder = AsyncMock()
        mock_db_client = AsyncMock()

        chunk = Chunk(text="Test", index=0, start_char=0, end_char=4, metadata={})
        mock_chunker.chunk_text.return_value = [chunk]
        mock_embedder.embed_batch.return_value = [[0.1] * 768]
        mock_db_client.add_memory.side_effect = DatabaseError(
            message="Storage failed", error_code="DB_001"
        )

        processor = TextProcessor(
            chunker=mock_chunker, embedder=mock_embedder, db_client=mock_db_client
        )

        with pytest.raises(DatabaseError) as exc_info:
            await processor.add_text("Test text", {})

        assert "storage failed" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_add_text_single_chunk_success(self):
        """Single chunk should be processed correctly."""
        mock_chunker = MagicMock()
        mock_embedder = AsyncMock()
        mock_db_client = AsyncMock()

        chunk = Chunk(text="Short text.", index=0, start_char=0, end_char=11, metadata={})
        mock_chunker.chunk_text.return_value = [chunk]
        mock_embedder.embed_batch.return_value = [[0.5] * 768]
        expected_memory_id = str(uuid.uuid4())
        mock_db_client.add_memory.return_value = expected_memory_id

        processor = TextProcessor(
            chunker=mock_chunker, embedder=mock_embedder, db_client=mock_db_client
        )

        memory_id = await processor.add_text("Short text.", {})

        assert memory_id == expected_memory_id
        mock_embedder.embed_batch.assert_called_once_with(["Short text."])

    @pytest.mark.asyncio
    async def test_add_text_multiple_chunks_success(self):
        """Multiple chunks should be processed in order."""
        mock_chunker = MagicMock()
        mock_embedder = AsyncMock()
        mock_db_client = AsyncMock()

        chunks = [
            Chunk(text=f"Chunk {i}", index=i, start_char=i * 10, end_char=(i + 1) * 10, metadata={})
            for i in range(5)
        ]
        mock_chunker.chunk_text.return_value = chunks
        mock_embedder.embed_batch.return_value = [[0.1 * (i + 1)] * 768 for i in range(5)]
        expected_memory_id = str(uuid.uuid4())
        mock_db_client.add_memory.return_value = expected_memory_id

        processor = TextProcessor(
            chunker=mock_chunker, embedder=mock_embedder, db_client=mock_db_client
        )

        memory_id = await processor.add_text("Long text with multiple chunks.", {})

        assert memory_id == expected_memory_id

        # Verify chunks were processed in order
        call_args = mock_embedder.embed_batch.call_args[0][0]
        assert call_args == [f"Chunk {i}" for i in range(5)]

    @pytest.mark.asyncio
    async def test_add_text_metadata_preserved(self):
        """Metadata should be preserved through pipeline."""
        mock_chunker = MagicMock()
        mock_embedder = AsyncMock()
        mock_db_client = AsyncMock()

        chunk = Chunk(text="Test", index=0, start_char=0, end_char=4, metadata={})
        mock_chunker.chunk_text.return_value = [chunk]
        mock_embedder.embed_batch.return_value = [[0.1] * 768]
        expected_memory_id = str(uuid.uuid4())
        mock_db_client.add_memory.return_value = expected_memory_id

        processor = TextProcessor(
            chunker=mock_chunker, embedder=mock_embedder, db_client=mock_db_client
        )

        metadata = {
            "source": "test.txt",
            "author": "test_user",
            "timestamp": "2025-11-23T10:00:00Z",
            "tags": ["test", "unittest"],
        }

        await processor.add_text("Test text", metadata)

        # Verify metadata was passed to database
        memory_arg = mock_db_client.add_memory.call_args[0][0]
        assert memory_arg.metadata == metadata

    @pytest.mark.asyncio
    async def test_add_text_logging_success_path(self):
        """Successful processing should log appropriate messages."""
        mock_chunker = MagicMock()
        mock_embedder = AsyncMock()
        mock_db_client = AsyncMock()

        chunk = Chunk(text="Test", index=0, start_char=0, end_char=4, metadata={})
        mock_chunker.chunk_text.return_value = [chunk]
        mock_embedder.embed_batch.return_value = [[0.1] * 768]
        expected_memory_id = str(uuid.uuid4())
        mock_db_client.add_memory.return_value = expected_memory_id

        processor = TextProcessor(
            chunker=mock_chunker, embedder=mock_embedder, db_client=mock_db_client
        )

        with patch.object(processor.logger, "info") as mock_log:
            memory_id = await processor.add_text("Test text", {})

            # Verify logging was called
            assert mock_log.call_count >= 1

            # Check for success log with memory_id
            log_calls = [call[0] for call in mock_log.call_args_list]
            assert any(
                "text_processed" in str(call) or "success" in str(call) for call in log_calls
            )

    @pytest.mark.asyncio
    async def test_add_text_logging_error_path(self):
        """Failed processing should log error messages."""
        mock_chunker = MagicMock()
        mock_chunker.chunk_text.side_effect = ProcessingError(
            message="Chunking failed", error_code="PROC_001"
        )

        processor = TextProcessor(chunker=mock_chunker)

        with patch.object(processor.logger, "error") as mock_error_log:
            with pytest.raises(ProcessingError):
                await processor.add_text("Test text", {})

            # Verify error was logged
            assert mock_error_log.call_count >= 1

    @pytest.mark.asyncio
    async def test_add_text_chunks_embeddings_count_match(self):
        """Number of chunks must equal number of embeddings."""
        mock_chunker = MagicMock()
        mock_embedder = AsyncMock()
        mock_db_client = AsyncMock()

        chunks = [
            Chunk(text=f"Chunk {i}", index=i, start_char=i * 10, end_char=(i + 1) * 10, metadata={})
            for i in range(3)
        ]
        mock_chunker.chunk_text.return_value = chunks

        # Intentionally return wrong number of embeddings
        mock_embedder.embed_batch.return_value = [[0.1] * 768, [0.2] * 768]  # Only 2 instead of 3

        processor = TextProcessor(
            chunker=mock_chunker, embedder=mock_embedder, db_client=mock_db_client
        )

        # Should raise error due to mismatch
        with pytest.raises(ProcessingError) as exc_info:
            await processor.add_text("Test text", {})

        assert "mismatch" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_add_text_embedding_dimension_validation(self):
        """Embeddings must be 768-dimensional."""
        mock_chunker = MagicMock()
        mock_embedder = AsyncMock()
        mock_db_client = AsyncMock()

        chunk = Chunk(text="Test", index=0, start_char=0, end_char=4, metadata={})
        mock_chunker.chunk_text.return_value = [chunk]

        # Return wrong dimension embedding
        mock_embedder.embed_batch.return_value = [[0.1] * 384]  # Wrong dimension

        processor = TextProcessor(
            chunker=mock_chunker, embedder=mock_embedder, db_client=mock_db_client
        )

        with pytest.raises(ProcessingError) as exc_info:
            await processor.add_text("Test text", {})

        assert "dimension" in str(exc_info.value).lower()


class TestTextProcessorIntegration:
    """Integration-style tests with realistic scenarios."""

    @pytest.mark.asyncio
    async def test_full_pipeline_realistic_text(self):
        """Test full pipeline with realistic multi-paragraph text."""
        mock_chunker = MagicMock()
        mock_embedder = AsyncMock()
        mock_db_client = AsyncMock()

        # Realistic chunks
        chunks = [
            Chunk(
                text="Python is a high-level programming language.",
                index=0,
                start_char=0,
                end_char=45,
                metadata={},
            ),
            Chunk(
                text="It was created by Guido van Rossum in 1991.",
                index=1,
                start_char=46,
                end_char=90,
                metadata={},
            ),
            Chunk(
                text="Python emphasizes code readability and simplicity.",
                index=2,
                start_char=91,
                end_char=141,
                metadata={},
            ),
        ]
        mock_chunker.chunk_text.return_value = chunks

        # Realistic embeddings (different values)
        embeddings = [[0.1 + i * 0.01] * 768 for i in range(3)]
        mock_embedder.embed_batch.return_value = embeddings

        expected_memory_id = str(uuid.uuid4())
        mock_db_client.add_memory.return_value = expected_memory_id

        processor = TextProcessor(
            chunker=mock_chunker, embedder=mock_embedder, db_client=mock_db_client
        )

        text = (
            "Python is a high-level programming language. "
            "It was created by Guido van Rossum in 1991. "
            "Python emphasizes code readability and simplicity."
        )
        metadata = {"source": "python_intro.txt", "author": "wikipedia", "category": "programming"}

        memory_id = await processor.add_text(text, metadata)

        # Verify complete pipeline
        assert memory_id == expected_memory_id
        mock_chunker.chunk_text.assert_called_once_with(text)
        mock_embedder.embed_batch.assert_called_once()
        mock_db_client.add_memory.assert_called_once()

        # Verify Memory object structure
        memory_arg = mock_db_client.add_memory.call_args[0][0]
        assert isinstance(memory_arg, Memory)
        assert memory_arg.text == text
        assert memory_arg.metadata == metadata
        assert len(memory_arg.chunks) == 3
        assert len(memory_arg.embeddings) == 3
        assert all(len(emb) == 768 for emb in memory_arg.embeddings)
