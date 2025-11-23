"""
Unit tests for VectorSearch component.

Tests the vector similarity search wrapper that orchestrates
FalkorDBClient and OllamaEmbedder for semantic search operations.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, timezone
from typing import List

from zapomni_core.search.vector_search import VectorSearch
from zapomni_db.models import SearchResult
from zapomni_core.exceptions import ValidationError, SearchError


class TestVectorSearchInit:
    """Test VectorSearch initialization."""

    def test_init_with_dependencies(self):
        """Test initialization with provided dependencies."""
        mock_db_client = Mock()
        mock_embedder = Mock()

        search = VectorSearch(
            db_client=mock_db_client,
            embedder=mock_embedder
        )

        assert search.db_client is mock_db_client
        assert search.embedder is mock_embedder

    def test_init_with_none_dependencies_raises_error(self):
        """Test initialization fails with None dependencies."""
        with pytest.raises(ValidationError) as exc_info:
            VectorSearch(db_client=None, embedder=None)

        assert "db_client cannot be None" in str(exc_info.value)


class TestVectorSearchSearch:
    """Test VectorSearch.search() method."""

    @pytest.fixture
    def mock_db_client(self):
        """Mock FalkorDB client."""
        client = Mock()
        client.vector_search = AsyncMock()
        return client

    @pytest.fixture
    def mock_embedder(self):
        """Mock Ollama embedder."""
        embedder = Mock()
        embedder.embed_text = AsyncMock()
        return embedder

    @pytest.fixture
    def vector_search(self, mock_db_client, mock_embedder):
        """VectorSearch instance with mocked dependencies."""
        return VectorSearch(
            db_client=mock_db_client,
            embedder=mock_embedder
        )

    @pytest.mark.asyncio
    async def test_search_basic_query(self, vector_search, mock_embedder, mock_db_client):
        """Test basic search with text query."""
        # Setup
        query_text = "Python programming"
        mock_embedding = [0.1] * 768
        mock_embedder.embed_text.return_value = mock_embedding

        mock_results = [
            SearchResult(
                memory_id="mem-1",
                chunk_id="chunk-1",
                text="Python is a programming language",
                similarity_score=0.95,
                tags=["python", "programming"],
                source="doc1.txt",
                timestamp=datetime.now(timezone.utc),
                chunk_index=0
            )
        ]
        mock_db_client.vector_search.return_value = mock_results

        # Execute
        results = await vector_search.search(query_text)

        # Assert
        assert len(results) == 1
        assert results[0].text == "Python is a programming language"
        assert results[0].similarity_score == 0.95

        # Verify calls
        mock_embedder.embed_text.assert_called_once_with(query_text)
        mock_db_client.vector_search.assert_called_once_with(
            embedding=mock_embedding,
            limit=10,
            filters=None
        )

    @pytest.mark.asyncio
    async def test_search_with_limit(self, vector_search, mock_embedder, mock_db_client):
        """Test search with custom limit."""
        query_text = "Python programming"
        mock_embedding = [0.1] * 768
        mock_embedder.embed_text.return_value = mock_embedding
        mock_db_client.vector_search.return_value = []

        await vector_search.search(query_text, limit=5)

        mock_db_client.vector_search.assert_called_once_with(
            embedding=mock_embedding,
            limit=5,
            filters=None
        )

    @pytest.mark.asyncio
    async def test_search_with_filters(self, vector_search, mock_embedder, mock_db_client):
        """Test search with metadata filters."""
        query_text = "Python programming"
        mock_embedding = [0.1] * 768
        mock_embedder.embed_text.return_value = mock_embedding
        mock_db_client.vector_search.return_value = []

        filters = {"tags": ["python"], "source": "docs"}
        await vector_search.search(query_text, filters=filters)

        mock_db_client.vector_search.assert_called_once_with(
            embedding=mock_embedding,
            limit=10,
            filters=filters
        )

    @pytest.mark.asyncio
    async def test_search_empty_query_raises_error(self, vector_search):
        """Test search with empty query raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            await vector_search.search("")

        assert "query cannot be empty" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_search_invalid_limit_raises_error(self, vector_search):
        """Test search with invalid limit raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            await vector_search.search("Python", limit=0)

        assert "limit must be >= 1" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_search_limit_too_large_raises_error(self, vector_search):
        """Test search with limit > 1000 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            await vector_search.search("Python", limit=1001)

        assert "limit cannot exceed 1000" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_search_empty_results(self, vector_search, mock_embedder, mock_db_client):
        """Test search returns empty list when no results found."""
        mock_embedding = [0.1] * 768
        mock_embedder.embed_text.return_value = mock_embedding
        mock_db_client.vector_search.return_value = []

        results = await vector_search.search("nonexistent query")

        assert results == []
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_multiple_results_sorted(self, vector_search, mock_embedder, mock_db_client):
        """Test search returns results sorted by similarity score."""
        mock_embedding = [0.1] * 768
        mock_embedder.embed_text.return_value = mock_embedding

        # Results should be sorted by similarity_score descending
        mock_results = [
            SearchResult(
                memory_id="mem-1",
                chunk_id="chunk-1",
                text="Result 1",
                similarity_score=0.95,
                tags=[],
                source="doc1.txt",
                timestamp=datetime.now(timezone.utc),
                chunk_index=0
            ),
            SearchResult(
                memory_id="mem-2",
                chunk_id="chunk-2",
                text="Result 2",
                similarity_score=0.85,
                tags=[],
                source="doc2.txt",
                timestamp=datetime.now(timezone.utc),
                chunk_index=0
            ),
            SearchResult(
                memory_id="mem-3",
                chunk_id="chunk-3",
                text="Result 3",
                similarity_score=0.75,
                tags=[],
                source="doc3.txt",
                timestamp=datetime.now(timezone.utc),
                chunk_index=0
            )
        ]
        mock_db_client.vector_search.return_value = mock_results

        results = await vector_search.search("test query")

        assert len(results) == 3
        assert results[0].similarity_score == 0.95
        assert results[1].similarity_score == 0.85
        assert results[2].similarity_score == 0.75

    @pytest.mark.asyncio
    async def test_search_embedder_failure_propagates(self, vector_search, mock_embedder):
        """Test embedding failure propagates as SearchError."""
        from zapomni_core.exceptions import EmbeddingError

        mock_embedder.embed_text.side_effect = EmbeddingError(
            message="Ollama connection failed",
            error_code="EMB_001"
        )

        with pytest.raises(SearchError) as exc_info:
            await vector_search.search("test query")

        assert "embedding generation failed" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_search_db_failure_propagates(self, vector_search, mock_embedder, mock_db_client):
        """Test database failure propagates as SearchError."""
        from zapomni_db.exceptions import DatabaseError

        mock_embedding = [0.1] * 768
        mock_embedder.embed_text.return_value = mock_embedding

        mock_db_client.vector_search.side_effect = DatabaseError(
            "Database connection failed"
        )

        with pytest.raises(SearchError) as exc_info:
            await vector_search.search("test query")

        assert "vector search failed" in str(exc_info.value).lower()


class TestVectorSearchPreprocessing:
    """Test query preprocessing and validation."""

    @pytest.fixture
    def vector_search(self):
        """VectorSearch instance with mocked dependencies."""
        mock_db_client = Mock()
        mock_db_client.vector_search = AsyncMock(return_value=[])
        mock_embedder = Mock()
        mock_embedder.embed_text = AsyncMock(return_value=[0.1] * 768)
        return VectorSearch(db_client=mock_db_client, embedder=mock_embedder)

    @pytest.mark.asyncio
    async def test_query_whitespace_trimmed(self, vector_search):
        """Test query whitespace is trimmed."""
        query = "  Python programming  "

        await vector_search.search(query)

        # Verify embedder was called with trimmed query
        vector_search.embedder.embed_text.assert_called_once_with("Python programming")

    @pytest.mark.asyncio
    async def test_query_only_whitespace_raises_error(self, vector_search):
        """Test query with only whitespace raises ValidationError."""
        with pytest.raises(ValidationError):
            await vector_search.search("   ")


class TestVectorSearchResultFormatting:
    """Test result ranking and formatting."""

    @pytest.fixture
    def vector_search(self):
        """VectorSearch instance with mocked dependencies."""
        mock_db_client = Mock()
        mock_embedder = Mock()
        return VectorSearch(db_client=mock_db_client, embedder=mock_embedder)

    @pytest.mark.asyncio
    async def test_results_returned_as_is(self, vector_search):
        """Test results from DB client are returned without modification."""
        mock_embedding = [0.1] * 768
        vector_search.embedder.embed_text = AsyncMock(return_value=mock_embedding)

        expected_results = [
            SearchResult(
                memory_id="mem-1",
                chunk_id="chunk-1",
                text="Test",
                similarity_score=0.95,
                tags=["test"],
                source="test.txt",
                timestamp=datetime.now(timezone.utc),
                chunk_index=0
            )
        ]
        vector_search.db_client.vector_search = AsyncMock(return_value=expected_results)

        results = await vector_search.search("test")

        assert results == expected_results
        assert results[0] is expected_results[0]  # Same object reference


class TestVectorSearchEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def vector_search(self):
        """VectorSearch instance with mocked dependencies."""
        mock_db_client = Mock()
        mock_db_client.vector_search = AsyncMock()
        mock_embedder = Mock()
        mock_embedder.embed_text = AsyncMock()
        return VectorSearch(db_client=mock_db_client, embedder=mock_embedder)

    @pytest.mark.asyncio
    async def test_search_with_none_filters(self, vector_search):
        """Test search with None filters (should use default None)."""
        mock_embedding = [0.1] * 768
        vector_search.embedder.embed_text.return_value = mock_embedding
        vector_search.db_client.vector_search.return_value = []

        await vector_search.search("test", filters=None)

        vector_search.db_client.vector_search.assert_called_once_with(
            embedding=mock_embedding,
            limit=10,
            filters=None
        )

    @pytest.mark.asyncio
    async def test_search_with_empty_filters_dict(self, vector_search):
        """Test search with empty filters dict."""
        mock_embedding = [0.1] * 768
        vector_search.embedder.embed_text.return_value = mock_embedding
        vector_search.db_client.vector_search.return_value = []

        await vector_search.search("test", filters={})

        vector_search.db_client.vector_search.assert_called_once_with(
            embedding=mock_embedding,
            limit=10,
            filters={}
        )

    @pytest.mark.asyncio
    async def test_search_max_limit(self, vector_search):
        """Test search with maximum allowed limit (1000)."""
        mock_embedding = [0.1] * 768
        vector_search.embedder.embed_text.return_value = mock_embedding
        vector_search.db_client.vector_search.return_value = []

        # Should not raise error
        await vector_search.search("test", limit=1000)

        vector_search.db_client.vector_search.assert_called_once_with(
            embedding=mock_embedding,
            limit=1000,
            filters=None
        )

    @pytest.mark.asyncio
    async def test_search_min_limit(self, vector_search):
        """Test search with minimum allowed limit (1)."""
        mock_embedding = [0.1] * 768
        vector_search.embedder.embed_text.return_value = mock_embedding
        vector_search.db_client.vector_search.return_value = []

        # Should not raise error
        await vector_search.search("test", limit=1)

        vector_search.db_client.vector_search.assert_called_once_with(
            embedding=mock_embedding,
            limit=1,
            filters=None
        )
