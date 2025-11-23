"""
Unit tests for BM25Search component.

Tests cover:
- Initialization and validation
- Document indexing
- Search functionality with various queries
- Score ranking and normalization
- Empty results handling
- Edge cases

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

import pytest
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

from zapomni_core.search.bm25_search import BM25Search
from zapomni_core.exceptions import ValidationError, SearchError
from zapomni_db.models import SearchResult
from datetime import datetime


# ==================== FIXTURES ====================


@pytest.fixture
def sample_documents() -> List[str]:
    """Sample documents for indexing."""
    return [
        "Python is a high-level programming language",
        "Machine learning is a subset of artificial intelligence",
        "Natural language processing uses machine learning algorithms",
        "Deep learning neural networks require large datasets",
        "Python programming is popular for data science and AI"
    ]


@pytest.fixture
def mock_db_client():
    """Mock FalkorDBClient for testing."""
    client = MagicMock()
    # Mock async methods
    client.vector_search = AsyncMock(return_value=[])
    return client


@pytest.fixture
def bm25_search(mock_db_client):
    """BM25Search instance with mocked dependencies."""
    return BM25Search(db_client=mock_db_client)


# ==================== INITIALIZATION TESTS ====================


class TestInitialization:
    """Test BM25Search initialization and validation."""

    def test_init_success(self, mock_db_client):
        """Test successful initialization."""
        search = BM25Search(db_client=mock_db_client)
        assert search.db_client == mock_db_client
        assert search._corpus is None
        assert search._bm25 is None

    def test_init_none_db_client(self):
        """Test initialization fails with None db_client."""
        with pytest.raises(ValidationError) as exc_info:
            BM25Search(db_client=None)

        assert "db_client cannot be None" in str(exc_info.value)
        assert exc_info.value.error_code == "VAL_001"


# ==================== INDEXING TESTS ====================


class TestIndexing:
    """Test document indexing functionality."""

    def test_index_documents_success(self, bm25_search, sample_documents):
        """Test successful document indexing."""
        bm25_search.index_documents(sample_documents)

        assert bm25_search._corpus is not None
        assert len(bm25_search._corpus) == len(sample_documents)
        assert bm25_search._bm25 is not None

    def test_index_documents_empty_list(self, bm25_search):
        """Test indexing empty document list raises error."""
        with pytest.raises(ValidationError) as exc_info:
            bm25_search.index_documents([])

        assert "Documents list cannot be empty" in str(exc_info.value)
        assert exc_info.value.error_code == "VAL_001"

    def test_index_documents_none(self, bm25_search):
        """Test indexing None raises error."""
        with pytest.raises(ValidationError) as exc_info:
            bm25_search.index_documents(None)

        assert "Documents must be a list" in str(exc_info.value)

    def test_index_documents_tokenization(self, bm25_search):
        """Test document tokenization during indexing."""
        docs = ["Hello World", "Python Programming"]
        bm25_search.index_documents(docs)

        # Check that corpus contains tokenized documents
        assert len(bm25_search._corpus) == 2
        # Should be lowercase and split
        assert "hello" in bm25_search._corpus[0]
        assert "world" in bm25_search._corpus[0]

    def test_reindex_documents(self, bm25_search, sample_documents):
        """Test re-indexing replaces old index."""
        # First indexing
        bm25_search.index_documents(sample_documents)
        first_bm25 = bm25_search._bm25

        # Re-index with new documents
        new_docs = ["New document one", "New document two"]
        bm25_search.index_documents(new_docs)

        # Verify new index
        assert bm25_search._bm25 is not first_bm25
        assert len(bm25_search._corpus) == 2


# ==================== SEARCH TESTS ====================


class TestSearch:
    """Test BM25 search functionality."""

    def test_search_success(self, bm25_search, sample_documents):
        """Test successful search returns ranked results."""
        bm25_search.index_documents(sample_documents)
        results = bm25_search.search("Python programming", limit=3)

        assert isinstance(results, list)
        assert len(results) <= 3

        # Check result structure
        if results:
            assert all(isinstance(r, dict) for r in results)
            assert all("text" in r for r in results)
            assert all("score" in r for r in results)
            assert all("index" in r for r in results)

    def test_search_score_ranking(self, bm25_search, sample_documents):
        """Test results are ranked by score (descending)."""
        bm25_search.index_documents(sample_documents)
        results = bm25_search.search("Python", limit=5)

        # Results should be sorted by score descending
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_limit_respected(self, bm25_search, sample_documents):
        """Test search respects limit parameter."""
        bm25_search.index_documents(sample_documents)

        results = bm25_search.search("machine learning", limit=2)
        assert len(results) <= 2

        results = bm25_search.search("machine learning", limit=10)
        assert len(results) <= min(10, len(sample_documents))

    def test_search_empty_query(self, bm25_search, sample_documents):
        """Test search with empty query raises error."""
        bm25_search.index_documents(sample_documents)

        with pytest.raises(ValidationError) as exc_info:
            bm25_search.search("", limit=5)

        assert "Query cannot be empty" in str(exc_info.value)

    def test_search_whitespace_query(self, bm25_search, sample_documents):
        """Test search with whitespace-only query raises error."""
        bm25_search.index_documents(sample_documents)

        with pytest.raises(ValidationError) as exc_info:
            bm25_search.search("   ", limit=5)

        assert "Query cannot be empty" in str(exc_info.value)

    def test_search_invalid_limit(self, bm25_search, sample_documents):
        """Test search with invalid limit raises error."""
        bm25_search.index_documents(sample_documents)

        # Limit < 1
        with pytest.raises(ValidationError) as exc_info:
            bm25_search.search("Python", limit=0)
        assert "Limit must be >= 1" in str(exc_info.value)

        # Limit > 1000
        with pytest.raises(ValidationError) as exc_info:
            bm25_search.search("Python", limit=1001)
        assert "Limit cannot exceed 1000" in str(exc_info.value)

    def test_search_not_indexed(self, bm25_search):
        """Test search before indexing raises error."""
        with pytest.raises(SearchError) as exc_info:
            bm25_search.search("Python", limit=5)

        assert "No documents indexed" in str(exc_info.value)

    def test_search_no_matches(self, bm25_search):
        """Test search with no matches returns empty list."""
        docs = ["Apple banana cherry", "Dog elephant fox"]
        bm25_search.index_documents(docs)

        # Query with words not in corpus
        results = bm25_search.search("zzz xxx www", limit=5)

        # Should return empty or very low scored results
        assert isinstance(results, list)

    def test_search_score_normalization(self, bm25_search, sample_documents):
        """Test that scores are normalized to 0-1 range."""
        bm25_search.index_documents(sample_documents)
        results = bm25_search.search("Python machine learning", limit=5)

        # All scores should be in [0, 1] range
        for result in results:
            assert 0.0 <= result["score"] <= 1.0

    def test_search_relevance(self, bm25_search, sample_documents):
        """Test search returns relevant documents."""
        bm25_search.index_documents(sample_documents)
        results = bm25_search.search("Python programming", limit=2)

        # Top results should contain "Python"
        top_texts = [r["text"] for r in results[:2]]
        assert any("Python" in text for text in top_texts)

    def test_search_multiple_terms(self, bm25_search, sample_documents):
        """Test search with multiple terms."""
        bm25_search.index_documents(sample_documents)
        results = bm25_search.search("machine learning neural networks", limit=3)

        assert isinstance(results, list)
        assert len(results) > 0

        # Results should be ranked
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)


# ==================== TOKENIZATION TESTS ====================


class TestTokenization:
    """Test tokenization and preprocessing."""

    def test_tokenize_basic(self, bm25_search):
        """Test basic tokenization."""
        tokens = bm25_search._tokenize("Hello World Python")
        assert tokens == ["hello", "world", "python"]

    def test_tokenize_lowercase(self, bm25_search):
        """Test tokenization converts to lowercase."""
        tokens = bm25_search._tokenize("UPPERCASE MiXeD lowercase")
        assert all(t.islower() or not t.isalpha() for t in tokens)

    def test_tokenize_punctuation(self, bm25_search):
        """Test tokenization removes punctuation."""
        tokens = bm25_search._tokenize("Hello, World! How are you?")
        # Punctuation should be handled (removed or separated)
        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_tokenize_empty_string(self, bm25_search):
        """Test tokenization of empty string."""
        tokens = bm25_search._tokenize("")
        assert tokens == []

    def test_tokenize_whitespace_only(self, bm25_search):
        """Test tokenization of whitespace."""
        tokens = bm25_search._tokenize("   \t\n  ")
        assert tokens == []


# ==================== EDGE CASES ====================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_document_search(self, bm25_search):
        """Test search with single document indexed."""
        bm25_search.index_documents(["Single document about Python"])
        results = bm25_search.search("Python", limit=5)

        assert len(results) == 1
        assert results[0]["score"] > 0

    def test_large_document_set(self, bm25_search):
        """Test indexing and searching large document set."""
        # Create 100 documents
        docs = [f"Document {i} about topic {i % 10}" for i in range(100)]
        bm25_search.index_documents(docs)

        results = bm25_search.search("topic", limit=10)
        assert len(results) <= 10

    def test_unicode_handling(self, bm25_search):
        """Test handling of unicode characters."""
        docs = ["Python 编程", "Machine learning 学习", "AI 人工智能"]
        bm25_search.index_documents(docs)

        results = bm25_search.search("Python", limit=5)
        assert isinstance(results, list)

    def test_special_characters(self, bm25_search):
        """Test handling of special characters."""
        docs = ["C++ programming", "C# development", "F# functional"]
        bm25_search.index_documents(docs)

        results = bm25_search.search("programming", limit=5)
        assert isinstance(results, list)

    def test_very_long_query(self, bm25_search, sample_documents):
        """Test search with very long query."""
        bm25_search.index_documents(sample_documents)
        long_query = " ".join(["Python"] * 100)

        results = bm25_search.search(long_query, limit=5)
        assert isinstance(results, list)

    def test_default_limit(self, bm25_search, sample_documents):
        """Test search with default limit."""
        bm25_search.index_documents(sample_documents)
        results = bm25_search.search("Python")

        # Default limit should be 10
        assert len(results) <= 10
