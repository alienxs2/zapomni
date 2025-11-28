"""
Unit tests for MemoryProcessor component.

Tests cover initialization, add_memory flow, search_memory flow, get_stats,
error handling, and edge cases.

Target coverage: 80%+

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

import asyncio
from datetime import datetime, timezone
from typing import List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from zapomni_core.exceptions import (
    DatabaseError,
    EmbeddingError,
    ProcessingError,
    SearchError,
    ValidationError,
)
from zapomni_core.memory_processor import (
    MemoryProcessor,
    ProcessorConfig,
    SearchResultItem,
)
from zapomni_db.models import Chunk, SearchResult

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_db_client():
    """Mock FalkorDB client."""
    client = AsyncMock()
    client.add_memory = AsyncMock(return_value="550e8400-e29b-41d4-a716-446655440000")
    client.vector_search = AsyncMock(
        return_value=[
            SearchResult(
                memory_id="550e8400-e29b-41d4-a716-446655440000",
                content="Python is a programming language",  # Required field
                relevance_score=0.95,  # Required field
                chunk_id="chunk-1",
                text="Python is a programming language",
                similarity_score=0.95,
                tags=["python", "programming"],
                source="user",
                timestamp=datetime.now(timezone.utc),
                chunk_index=0,
            )
        ]
    )
    client.get_stats = AsyncMock(
        return_value={
            "total_memories": 5,
            "total_chunks": 20,
            "database_size_mb": 10.5,
            "avg_query_latency_ms": 150,
            "oldest_memory_date": datetime.now(timezone.utc),
            "newest_memory_date": datetime.now(timezone.utc),
        }
    )
    return client


@pytest.fixture
def mock_chunker():
    """Mock SemanticChunker."""
    chunker = Mock()
    chunker.chunk_text = Mock(
        return_value=[
            Chunk(
                text="Python is a programming language",
                index=0,
                start_char=0,
                end_char=33,
                metadata={},
            ),
            Chunk(
                text="created by Guido van Rossum.",
                index=1,
                start_char=34,
                end_char=62,
                metadata={},
            ),
        ]
    )
    return chunker


@pytest.fixture
def mock_embedder():
    """Mock OllamaEmbedder."""
    embedder = AsyncMock()
    embedder.embed_text = AsyncMock(return_value=[0.1] * 768)  # 768-dimensional embedding
    # embed_batch returns a list of embeddings (one per input text)
    embedder.embed_batch = AsyncMock(side_effect=lambda texts, **kwargs: [[0.1] * 768 for _ in texts])
    return embedder


@pytest.fixture
def processor_config():
    """Standard processor configuration."""
    return ProcessorConfig(
        enable_cache=False,
        enable_extraction=False,
        enable_graph=False,
        max_text_length=10_000_000,
        batch_size=32,
        search_mode="vector",
    )


@pytest.fixture
def memory_processor(mock_db_client, mock_chunker, mock_embedder, processor_config):
    """Create a MemoryProcessor with mocked dependencies."""
    return MemoryProcessor(
        db_client=mock_db_client,
        chunker=mock_chunker,
        embedder=mock_embedder,
        config=processor_config,
    )


# ============================================================================
# Initialization Tests
# ============================================================================


class TestMemoryProcessorInit:
    """Test MemoryProcessor initialization."""

    def test_init_minimal(self, mock_db_client, mock_chunker, mock_embedder):
        """Test minimal initialization with required dependencies only."""
        processor = MemoryProcessor(
            db_client=mock_db_client,
            chunker=mock_chunker,
            embedder=mock_embedder,
        )

        assert processor.db_client is mock_db_client
        assert processor.chunker is mock_chunker
        assert processor.embedder is mock_embedder
        # Note: extractor is now lazy-loaded when enable_extraction=True (default)
        # So it won't be None after first access
        assert processor.cache is None
        assert processor.task_manager is None
        assert isinstance(processor.config, ProcessorConfig)

    def test_init_full(self, mock_db_client, mock_chunker, mock_embedder):
        """Test full initialization with all dependencies."""
        mock_extractor = Mock()
        mock_cache = Mock()
        mock_task_manager = Mock()
        config = ProcessorConfig(enable_cache=True, enable_extraction=True)

        processor = MemoryProcessor(
            db_client=mock_db_client,
            chunker=mock_chunker,
            embedder=mock_embedder,
            extractor=mock_extractor,
            cache=mock_cache,
            task_manager=mock_task_manager,
            config=config,
        )

        assert processor.extractor is mock_extractor
        assert processor.cache is mock_cache
        assert processor.task_manager is mock_task_manager

    def test_init_missing_db_client_raises(self, mock_chunker, mock_embedder):
        """Test that missing db_client raises ValueError."""
        with pytest.raises(ValueError, match="db_client is required"):
            MemoryProcessor(
                db_client=None,
                chunker=mock_chunker,
                embedder=mock_embedder,
            )

    def test_init_missing_chunker_raises(self, mock_db_client, mock_embedder):
        """Test that missing chunker raises ValueError."""
        with pytest.raises(ValueError, match="chunker is required"):
            MemoryProcessor(
                db_client=mock_db_client,
                chunker=None,
                embedder=mock_embedder,
            )

    def test_init_missing_embedder_raises(self, mock_db_client, mock_chunker):
        """Test that missing embedder raises ValueError."""
        with pytest.raises(ValueError, match="embedder is required"):
            MemoryProcessor(
                db_client=mock_db_client,
                chunker=mock_chunker,
                embedder=None,
            )

    def test_init_invalid_config_max_text_length_raises(
        self, mock_db_client, mock_chunker, mock_embedder
    ):
        """Test that invalid max_text_length raises ValueError."""
        config = ProcessorConfig(max_text_length=0)
        with pytest.raises(ValueError, match="max_text_length must be positive"):
            MemoryProcessor(
                db_client=mock_db_client,
                chunker=mock_chunker,
                embedder=mock_embedder,
                config=config,
            )

    def test_init_invalid_config_batch_size_raises(
        self, mock_db_client, mock_chunker, mock_embedder
    ):
        """Test that invalid batch_size raises ValueError."""
        config = ProcessorConfig(batch_size=0)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            MemoryProcessor(
                db_client=mock_db_client,
                chunker=mock_chunker,
                embedder=mock_embedder,
                config=config,
            )

    def test_init_invalid_search_mode_raises(self, mock_db_client, mock_chunker, mock_embedder):
        """Test that invalid search_mode raises ValueError."""
        config = ProcessorConfig(search_mode="invalid_mode")
        with pytest.raises(ValueError, match="Invalid search_mode"):
            MemoryProcessor(
                db_client=mock_db_client,
                chunker=mock_chunker,
                embedder=mock_embedder,
                config=config,
            )


# ============================================================================
# add_memory Tests - Happy Path
# ============================================================================


class TestAddMemoryHappyPath:
    """Test add_memory with valid inputs."""

    @pytest.mark.asyncio
    async def test_add_memory_success(self, memory_processor, mock_db_client):
        """Test adding memory successfully."""
        text = "Python is a programming language created by Guido van Rossum."
        memory_id = await memory_processor.add_memory(text=text)

        assert memory_id == "550e8400-e29b-41d4-a716-446655440000"
        assert mock_db_client.add_memory.called

    @pytest.mark.asyncio
    async def test_add_memory_with_metadata(self, memory_processor, mock_db_client):
        """Test adding memory with metadata."""
        text = "Python is a programming language."
        metadata = {
            "tags": ["python", "programming"],
            "source": "wikipedia",
            "date": "2025-11-23",
        }

        memory_id = await memory_processor.add_memory(text=text, metadata=metadata)

        assert memory_id == "550e8400-e29b-41d4-a716-446655440000"
        assert mock_db_client.add_memory.called

        # Verify metadata was stored
        call_args = mock_db_client.add_memory.call_args
        assert call_args is not None
        memory_arg = call_args[0][0]
        assert "timestamp" in memory_arg.metadata

    @pytest.mark.asyncio
    async def test_add_memory_returns_uuid(self, memory_processor):
        """Test that add_memory returns valid UUID."""
        text = "Test memory content."
        memory_id = await memory_processor.add_memory(text=text)

        # Verify it's a valid UUID format
        parts = memory_id.split("-")
        assert len(parts) == 5
        assert len(parts[0]) == 8
        assert len(parts[1]) == 4
        assert len(parts[2]) == 4
        assert len(parts[3]) == 4
        assert len(parts[4]) == 12

    @pytest.mark.asyncio
    async def test_add_memory_chunks_created(self, memory_processor, mock_chunker):
        """Test that chunks are created during add_memory."""
        text = "Python is great. It is simple and powerful."
        await memory_processor.add_memory(text=text)

        assert mock_chunker.chunk_text.called

    @pytest.mark.asyncio
    async def test_add_memory_embeddings_generated(self, memory_processor, mock_embedder):
        """Test that embeddings are generated during add_memory."""
        text = "Python is a programming language."
        await memory_processor.add_memory(text=text)

        # Should use batch API for efficiency
        assert mock_embedder.embed_batch.call_count >= 1


# ============================================================================
# add_memory Tests - Validation
# ============================================================================


class TestAddMemoryValidation:
    """Test add_memory input validation."""

    @pytest.mark.asyncio
    async def test_add_memory_empty_text_raises(self, memory_processor):
        """Test that empty text raises ValidationError."""
        with pytest.raises(ValidationError, match="Text cannot be empty"):
            await memory_processor.add_memory(text="")

    @pytest.mark.asyncio
    async def test_add_memory_whitespace_only_raises(self, memory_processor):
        """Test that whitespace-only text raises ValidationError."""
        with pytest.raises(ValidationError, match="Text cannot be empty"):
            await memory_processor.add_memory(text="   \n  \t  ")

    @pytest.mark.asyncio
    async def test_add_memory_too_large_raises(self, memory_processor):
        """Test that text exceeding max_text_length raises ValidationError."""
        # Create text larger than config max (10MB)
        config = ProcessorConfig(max_text_length=100)
        processor = MemoryProcessor(
            db_client=memory_processor.db_client,
            chunker=memory_processor.chunker,
            embedder=memory_processor.embedder,
            config=config,
        )

        large_text = "x" * 101
        with pytest.raises(ValidationError, match="Text exceeds maximum length"):
            await processor.add_memory(text=large_text)

    @pytest.mark.asyncio
    async def test_add_memory_non_string_text_raises(self, memory_processor):
        """Test that non-string text raises ValidationError."""
        with pytest.raises(ValidationError, match="Text must be a string"):
            await memory_processor.add_memory(text=123)  # type: ignore

    @pytest.mark.asyncio
    async def test_add_memory_reserved_metadata_key_raises(self, memory_processor):
        """Test that reserved metadata key raises ValidationError."""
        reserved_keys = ["memory_id", "timestamp", "chunks"]
        for key in reserved_keys:
            with pytest.raises(ValidationError, match=f"Reserved key in metadata"):
                await memory_processor.add_memory(text="Test", metadata={key: "value"})

    @pytest.mark.asyncio
    async def test_add_memory_non_serializable_metadata_raises(self, memory_processor):
        """Test that non-JSON-serializable metadata raises ValidationError."""

        def custom_func():
            pass

        with pytest.raises(ValidationError, match="JSON-serializable"):
            await memory_processor.add_memory(
                text="Test",
                metadata={"func": custom_func},  # type: ignore
            )


# ============================================================================
# search_memory Tests - Happy Path
# ============================================================================


class TestSearchMemoryHappyPath:
    """Test search_memory with valid inputs."""

    @pytest.mark.asyncio
    async def test_search_memory_success(self, memory_processor):
        """Test searching memories successfully."""
        query = "Who created Python?"
        results = await memory_processor.search_memory(query=query, limit=5)

        assert isinstance(results, list)
        assert len(results) >= 1
        assert isinstance(results[0], SearchResultItem)

    @pytest.mark.asyncio
    async def test_search_memory_with_filters(self, memory_processor):
        """Test search with metadata filters."""
        query = "Python web frameworks"
        filters = {"tags": ["python", "web"], "source": "documentation"}

        results = await memory_processor.search_memory(
            query=query,
            limit=10,
            filters=filters,
        )

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_memory_limit_enforced(self, memory_processor, mock_db_client):
        """Test that limit is enforced."""
        # Create many mock results
        mock_results = [
            SearchResult(
                memory_id=f"id-{i}",
                content=f"Result {i}",  # Required field
                relevance_score=0.9 - i * 0.01,  # Required field
                chunk_id=f"chunk-{i}",
                text=f"Result {i}",
                similarity_score=0.9 - i * 0.01,
                tags=["python"],
                source="test",
                timestamp=datetime.now(timezone.utc),
                chunk_index=0,
            )
            for i in range(20)
        ]
        mock_db_client.vector_search = AsyncMock(return_value=mock_results)

        query = "test query"
        results = await memory_processor.search_memory(query=query, limit=5)

        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_search_memory_sorted_by_similarity(self, memory_processor):
        """Test that results are sorted by similarity score (descending)."""
        query = "test"
        results = await memory_processor.search_memory(query=query, limit=10)

        # Check that scores are in descending order
        for i in range(len(results) - 1):
            assert results[i].similarity_score >= results[i + 1].similarity_score


# ============================================================================
# search_memory Tests - Validation
# ============================================================================


class TestSearchMemoryValidation:
    """Test search_memory input validation."""

    @pytest.mark.asyncio
    async def test_search_memory_empty_query_raises(self, memory_processor):
        """Test that empty query raises ValidationError."""
        with pytest.raises(ValidationError, match="Query cannot be empty"):
            await memory_processor.search_memory(query="")

    @pytest.mark.asyncio
    async def test_search_memory_whitespace_query_raises(self, memory_processor):
        """Test that whitespace-only query raises ValidationError."""
        with pytest.raises(ValidationError, match="Query cannot be empty"):
            await memory_processor.search_memory(query="   \n  ")

    @pytest.mark.asyncio
    async def test_search_memory_query_too_long_raises(self, memory_processor):
        """Test that query exceeding max length raises ValidationError."""
        long_query = "a" * 1001
        with pytest.raises(ValidationError, match="Query exceeds maximum length"):
            await memory_processor.search_memory(query=long_query)

    @pytest.mark.asyncio
    async def test_search_memory_limit_below_1_raises(self, memory_processor):
        """Test that limit < 1 raises ValidationError."""
        with pytest.raises(ValidationError, match="Limit must be between 1 and 100"):
            await memory_processor.search_memory(query="test", limit=0)

    @pytest.mark.asyncio
    async def test_search_memory_limit_above_100_raises(self, memory_processor):
        """Test that limit > 100 raises ValidationError."""
        with pytest.raises(ValidationError, match="Limit must be between 1 and 100"):
            await memory_processor.search_memory(query="test", limit=101)

    @pytest.mark.asyncio
    async def test_search_memory_invalid_search_mode_raises(self, memory_processor):
        """Test that invalid search_mode raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid search mode"):
            await memory_processor.search_memory(
                query="test",
                search_mode="invalid_mode",
            )

    @pytest.mark.asyncio
    async def test_search_memory_invalid_filter_key_raises(self, memory_processor):
        """Test that invalid filter key raises ValidationError."""
        with pytest.raises(ValidationError, match="Unknown filter key"):
            await memory_processor.search_memory(
                query="test",
                filters={"invalid_filter": "value"},
            )


# ============================================================================
# get_stats Tests
# ============================================================================


class TestGetStats:
    """Test get_stats functionality."""

    @pytest.mark.asyncio
    async def test_get_stats_success(self, memory_processor):
        """Test getting statistics successfully."""
        stats = await memory_processor.get_stats()

        assert isinstance(stats, dict)
        assert "total_memories" in stats
        assert "total_chunks" in stats
        assert "database_size_mb" in stats
        assert "avg_chunks_per_memory" in stats
        assert "avg_query_latency_ms" in stats

    @pytest.mark.asyncio
    async def test_get_stats_no_memories(self, memory_processor, mock_db_client):
        """Test get_stats with empty database."""
        mock_db_client.get_stats = AsyncMock(
            return_value={
                "total_memories": 0,
                "total_chunks": 0,
                "database_size_mb": 0.0,
                "avg_query_latency_ms": 0,
            }
        )

        stats = await memory_processor.get_stats()

        assert stats["total_memories"] == 0
        assert stats["total_chunks"] == 0
        assert stats["avg_chunks_per_memory"] == 0.0

    @pytest.mark.asyncio
    async def test_get_stats_calculates_avg_chunks(self, memory_processor, mock_db_client):
        """Test that avg_chunks_per_memory is calculated correctly."""
        mock_db_client.get_stats = AsyncMock(
            return_value={
                "total_memories": 10,
                "total_chunks": 50,
                "database_size_mb": 5.0,
                "avg_query_latency_ms": 100,
            }
        )

        stats = await memory_processor.get_stats()

        assert stats["avg_chunks_per_memory"] == 5.0  # 50 / 10

    @pytest.mark.asyncio
    async def test_get_stats_with_dates(self, memory_processor, mock_db_client):
        """Test that date range is included in stats."""
        now = datetime.now(timezone.utc)
        mock_db_client.get_stats = AsyncMock(
            return_value={
                "total_memories": 5,
                "total_chunks": 20,
                "database_size_mb": 10.5,
                "avg_query_latency_ms": 150,
                "oldest_memory_date": now,
                "newest_memory_date": now,
            }
        )

        stats = await memory_processor.get_stats()

        assert "oldest_memory_date" in stats
        assert "newest_memory_date" in stats


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Test error handling and recovery."""

    @pytest.mark.asyncio
    async def test_add_memory_chunking_error(self, memory_processor, mock_chunker):
        """Test error handling when chunking fails."""
        mock_chunker.chunk_text = Mock(
            side_effect=ProcessingError(
                message="Chunking failed",
                error_code="PROC_001",
            )
        )

        with pytest.raises(ProcessingError):
            await memory_processor.add_memory(text="Test")

    @pytest.mark.asyncio
    async def test_add_memory_embedding_error(self, memory_processor, mock_embedder):
        """Test error handling when embedding generation fails."""
        mock_embedder.embed_batch = AsyncMock(
            side_effect=EmbeddingError(
                message="Ollama unavailable",
                error_code="EMB_001",
            )
        )

        with pytest.raises(EmbeddingError):
            await memory_processor.add_memory(text="Test")

    @pytest.mark.asyncio
    async def test_add_memory_storage_error(self, memory_processor, mock_db_client):
        """Test error handling when storage fails."""
        mock_db_client.add_memory = AsyncMock(
            side_effect=DatabaseError(
                message="Database error",
                error_code="DB_001",
            )
        )

        with pytest.raises(DatabaseError):
            await memory_processor.add_memory(text="Test")

    @pytest.mark.asyncio
    async def test_search_memory_embedding_error(self, memory_processor, mock_embedder):
        """Test error handling when query embedding generation fails."""
        mock_embedder.embed_text = AsyncMock(
            side_effect=EmbeddingError(
                message="Embedding failed",
                error_code="EMB_001",
            )
        )

        with pytest.raises(EmbeddingError):
            await memory_processor.search_memory(query="test")


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_add_memory_single_char(self, memory_processor):
        """Test adding memory with single character."""
        memory_id = await memory_processor.add_memory(text="A")
        assert memory_id is not None

    @pytest.mark.asyncio
    async def test_add_memory_exact_max_length(self, memory_processor):
        """Test adding memory with exactly max length."""
        config = ProcessorConfig(max_text_length=100)
        processor = MemoryProcessor(
            db_client=memory_processor.db_client,
            chunker=memory_processor.chunker,
            embedder=memory_processor.embedder,
            config=config,
        )

        text = "x" * 100
        memory_id = await processor.add_memory(text=text)
        assert memory_id is not None

    @pytest.mark.asyncio
    async def test_search_memory_no_results(self, memory_processor, mock_db_client):
        """Test search that returns no results."""
        mock_db_client.vector_search = AsyncMock(return_value=[])

        results = await memory_processor.search_memory(query="test")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_memory_limit_1(self, memory_processor):
        """Test search with limit=1."""
        results = await memory_processor.search_memory(query="test", limit=1)
        assert len(results) <= 1

    @pytest.mark.asyncio
    async def test_search_memory_limit_100(self, memory_processor):
        """Test search with limit=100."""
        results = await memory_processor.search_memory(query="test", limit=100)
        assert len(results) <= 100

    @pytest.mark.asyncio
    async def test_add_memory_none_metadata(self, memory_processor):
        """Test adding memory with None metadata."""
        memory_id = await memory_processor.add_memory(text="Test", metadata=None)
        assert memory_id is not None

    @pytest.mark.asyncio
    async def test_add_memory_empty_metadata(self, memory_processor):
        """Test adding memory with empty metadata dict."""
        memory_id = await memory_processor.add_memory(text="Test", metadata={})
        assert memory_id is not None


# ============================================================================
# build_knowledge_graph Tests
# ============================================================================


class TestBuildKnowledgeGraph:
    """Test build_knowledge_graph functionality."""

    @pytest.mark.asyncio
    async def test_build_knowledge_graph_not_implemented(self, memory_processor):
        """Test that build_knowledge_graph raises NotImplementedError in Phase 1."""
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            await memory_processor.build_knowledge_graph()


# ============================================================================
# Integration-like Tests
# ============================================================================


class TestAddMemoryIntegration:
    """Test add_memory with more realistic scenarios."""

    @pytest.mark.asyncio
    async def test_add_memory_full_pipeline(
        self, memory_processor, mock_db_client, mock_chunker, mock_embedder
    ):
        """Test complete add_memory pipeline with all stages."""
        text = "Django is a web framework for Python."
        metadata = {"tags": ["django", "web", "python"], "source": "docs"}

        memory_id = await memory_processor.add_memory(text=text, metadata=metadata)

        # Verify all pipeline stages were executed
        assert mock_chunker.chunk_text.called
        assert mock_embedder.embed_batch.called  # Uses batch API for efficiency
        assert mock_db_client.add_memory.called
        assert memory_id is not None


# ============================================================================
# Configuration Tests
# ============================================================================


class TestProcessorConfig:
    """Test ProcessorConfig dataclass."""

    def test_config_defaults(self):
        """Test ProcessorConfig default values."""
        config = ProcessorConfig()

        assert config.enable_cache is True  # Changed: cache enabled by default for performance
        # Note: enable_extraction and enable_graph are now True by default
        assert config.enable_extraction is True
        assert config.enable_graph is True
        assert config.max_text_length == 10_000_000
        assert config.batch_size == 32
        assert config.search_mode == "vector"

    def test_config_custom_values(self):
        """Test ProcessorConfig with custom values."""
        config = ProcessorConfig(
            enable_cache=True,
            enable_extraction=True,
            max_text_length=5_000_000,
            batch_size=64,
            search_mode="hybrid",
        )

        assert config.enable_cache is True
        assert config.enable_extraction is True
        assert config.max_text_length == 5_000_000
        assert config.batch_size == 64
        assert config.search_mode == "hybrid"


# ============================================================================
# SearchResultItem Tests
# ============================================================================


class TestSearchResultItem:
    """Test SearchResultItem dataclass."""

    def test_search_result_creation(self):
        """Test creating a SearchResultItem."""
        now = datetime.now(timezone.utc)
        result = SearchResultItem(
            memory_id="test-id",
            text="Test text",
            similarity_score=0.95,
            tags=["test", "python"],
            source="user",
            timestamp=now,
        )

        assert result.memory_id == "test-id"
        assert result.text == "Test text"
        assert result.similarity_score == 0.95
        assert result.tags == ["test", "python"]
        assert result.source == "user"
        assert result.timestamp == now
        assert result.highlight is None

    def test_search_result_with_highlight(self):
        """Test creating SearchResultItem with highlight."""
        now = datetime.now(timezone.utc)
        result = SearchResultItem(
            memory_id="test-id",
            text="Test text",
            similarity_score=0.95,
            tags=["test"],
            source="user",
            timestamp=now,
            highlight="<em>Test</em> text",
        )

        assert result.highlight == "<em>Test</em> text"


# ============================================================================
# Date Filter Timezone Tests (Issue #17 - BUG-001)
# ============================================================================


class TestParseFilterDate:
    """Test _parse_filter_date method for timezone handling."""

    @pytest.fixture
    def processor(self, mock_db_client, mock_chunker, mock_embedder):
        """Create MemoryProcessor instance for testing."""
        return MemoryProcessor(
            db_client=mock_db_client,
            chunker=mock_chunker,
            embedder=mock_embedder,
        )

    def test_parse_date_only_string(self, processor):
        """Test parsing date-only string (naive) - Issue #17 main case."""
        result = processor._parse_filter_date("2025-11-01")

        assert result.year == 2025
        assert result.month == 11
        assert result.day == 1
        assert result.hour == 0
        assert result.minute == 0
        assert result.second == 0
        # Must be timezone-aware (UTC)
        assert result.tzinfo == timezone.utc

    def test_parse_datetime_string_naive(self, processor):
        """Test parsing datetime string without timezone."""
        result = processor._parse_filter_date("2025-11-01T12:30:45")

        assert result.year == 2025
        assert result.month == 11
        assert result.day == 1
        assert result.hour == 12
        assert result.minute == 30
        assert result.second == 45
        # Must be timezone-aware (UTC)
        assert result.tzinfo == timezone.utc

    def test_parse_datetime_with_z_suffix(self, processor):
        """Test parsing datetime string with Z suffix."""
        result = processor._parse_filter_date("2025-11-01T12:30:45Z")

        assert result.year == 2025
        assert result.month == 11
        assert result.day == 1
        assert result.hour == 12
        assert result.minute == 30
        assert result.second == 45
        # Z means UTC
        assert result.tzinfo is not None
        assert result.utcoffset().total_seconds() == 0

    def test_parse_datetime_with_utc_offset(self, processor):
        """Test parsing datetime string with explicit UTC offset."""
        result = processor._parse_filter_date("2025-11-01T12:30:45+00:00")

        assert result.year == 2025
        assert result.month == 11
        assert result.day == 1
        assert result.hour == 12
        assert result.minute == 30
        assert result.second == 45
        # Already timezone-aware
        assert result.tzinfo is not None

    def test_parse_datetime_with_positive_offset(self, processor):
        """Test parsing datetime string with positive timezone offset."""
        result = processor._parse_filter_date("2025-11-01T12:30:45+03:00")

        assert result.year == 2025
        assert result.month == 11
        assert result.day == 1
        assert result.hour == 12
        assert result.minute == 30
        assert result.second == 45
        # Preserves timezone info
        assert result.tzinfo is not None
        assert result.utcoffset().total_seconds() == 3 * 3600

    def test_parse_datetime_with_negative_offset(self, processor):
        """Test parsing datetime string with negative timezone offset."""
        result = processor._parse_filter_date("2025-11-01T12:30:45-05:00")

        assert result.year == 2025
        assert result.month == 11
        assert result.hour == 12
        # Preserves timezone info
        assert result.tzinfo is not None
        assert result.utcoffset().total_seconds() == -5 * 3600


class TestApplyFiltersDateRange:
    """Test _apply_filters with date range filters - Issue #17."""

    @pytest.fixture
    def processor(self, mock_db_client, mock_chunker, mock_embedder):
        """Create MemoryProcessor instance for testing."""
        return MemoryProcessor(
            db_client=mock_db_client,
            chunker=mock_chunker,
            embedder=mock_embedder,
        )

    def test_apply_date_from_filter_naive_string(self, processor):
        """Test date_from filter with naive date string (main Issue #17 case)."""
        # Results with timezone-aware timestamps (as returned from DB)
        results = [
            SearchResultItem(
                memory_id="id-1",
                text="Memory from October",
                similarity_score=0.9,
                tags=["test"],
                source="user",
                timestamp=datetime(2025, 10, 15, 12, 0, 0, tzinfo=timezone.utc),
            ),
            SearchResultItem(
                memory_id="id-2",
                text="Memory from November",
                similarity_score=0.85,
                tags=["test"],
                source="user",
                timestamp=datetime(2025, 11, 15, 12, 0, 0, tzinfo=timezone.utc),
            ),
        ]

        # Filter with naive date string (no timezone info)
        filters = {"date_from": "2025-11-01"}

        # Should NOT raise TypeError: can't compare offset-naive and offset-aware datetimes
        filtered = processor._apply_filters(results, filters)

        assert len(filtered) == 1
        assert filtered[0].memory_id == "id-2"

    def test_apply_date_to_filter_naive_string(self, processor):
        """Test date_to filter with naive date string."""
        results = [
            SearchResultItem(
                memory_id="id-1",
                text="Memory from October",
                similarity_score=0.9,
                tags=["test"],
                source="user",
                timestamp=datetime(2025, 10, 15, 12, 0, 0, tzinfo=timezone.utc),
            ),
            SearchResultItem(
                memory_id="id-2",
                text="Memory from November",
                similarity_score=0.85,
                tags=["test"],
                source="user",
                timestamp=datetime(2025, 11, 15, 12, 0, 0, tzinfo=timezone.utc),
            ),
        ]

        filters = {"date_to": "2025-10-31"}

        filtered = processor._apply_filters(results, filters)

        assert len(filtered) == 1
        assert filtered[0].memory_id == "id-1"

    def test_apply_date_range_filter(self, processor):
        """Test both date_from and date_to filters together."""
        results = [
            SearchResultItem(
                memory_id="id-1",
                text="Memory from September",
                similarity_score=0.9,
                tags=["test"],
                source="user",
                timestamp=datetime(2025, 9, 15, 12, 0, 0, tzinfo=timezone.utc),
            ),
            SearchResultItem(
                memory_id="id-2",
                text="Memory from October",
                similarity_score=0.85,
                tags=["test"],
                source="user",
                timestamp=datetime(2025, 10, 15, 12, 0, 0, tzinfo=timezone.utc),
            ),
            SearchResultItem(
                memory_id="id-3",
                text="Memory from November",
                similarity_score=0.8,
                tags=["test"],
                source="user",
                timestamp=datetime(2025, 11, 15, 12, 0, 0, tzinfo=timezone.utc),
            ),
        ]

        filters = {"date_from": "2025-10-01", "date_to": "2025-10-31"}

        filtered = processor._apply_filters(results, filters)

        assert len(filtered) == 1
        assert filtered[0].memory_id == "id-2"

    def test_apply_date_filter_with_z_suffix(self, processor):
        """Test date filter with Z suffix (UTC indicator)."""
        results = [
            SearchResultItem(
                memory_id="id-1",
                text="Memory from October",
                similarity_score=0.9,
                tags=["test"],
                source="user",
                timestamp=datetime(2025, 10, 15, 12, 0, 0, tzinfo=timezone.utc),
            ),
        ]

        filters = {"date_from": "2025-10-01T00:00:00Z"}

        # Should handle Z suffix correctly
        filtered = processor._apply_filters(results, filters)

        assert len(filtered) == 1

    def test_apply_date_filter_with_offset(self, processor):
        """Test date filter with explicit timezone offset."""
        results = [
            SearchResultItem(
                memory_id="id-1",
                text="Memory from October",
                similarity_score=0.9,
                tags=["test"],
                source="user",
                timestamp=datetime(2025, 10, 15, 12, 0, 0, tzinfo=timezone.utc),
            ),
        ]

        filters = {"date_from": "2025-10-01T00:00:00+00:00"}

        filtered = processor._apply_filters(results, filters)

        assert len(filtered) == 1
