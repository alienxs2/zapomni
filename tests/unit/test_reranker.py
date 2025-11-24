"""
Unit tests for CrossEncoderReranker component.

Tests the reranking functionality that improves search result relevance
using cross-encoder model for semantic similarity scoring.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from typing import List
import numpy as np

from zapomni_core.search.reranker import CrossEncoderReranker
from zapomni_db.models import SearchResult
from zapomni_core.exceptions import ValidationError, SearchError


class TestCrossEncoderRerankerInit:
    """Test CrossEncoderReranker initialization."""

    def test_init_with_default_model(self):
        """Test initialization with default model."""
        with patch('zapomni_core.search.reranker.CrossEncoder'):
            reranker = CrossEncoderReranker()

            assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
            assert reranker.fallback_enabled is True
            assert reranker.model is None  # Lazy loaded

    def test_init_with_custom_model_name(self):
        """Test initialization with custom model name."""
        custom_model = "cross-encoder/ms-marco-MiniLM-L-12-v2"
        with patch('zapomni_core.search.reranker.CrossEncoder'):
            reranker = CrossEncoderReranker(model_name=custom_model)

            assert reranker.model_name == custom_model
            assert reranker.fallback_enabled is True

    def test_init_with_fallback_disabled(self):
        """Test initialization with fallback disabled."""
        with patch('zapomni_core.search.reranker.CrossEncoder'):
            reranker = CrossEncoderReranker(fallback_enabled=False)

            assert reranker.fallback_enabled is False

    def test_init_with_empty_model_name_raises_error(self):
        """Test initialization fails with empty model name."""
        with pytest.raises(ValidationError) as exc_info:
            CrossEncoderReranker(model_name="")

        assert "non-empty string" in str(exc_info.value.message)

    def test_init_with_none_model_name_uses_default(self):
        """Test initialization with None model_name uses default."""
        with patch('zapomni_core.search.reranker.CrossEncoder'):
            reranker = CrossEncoderReranker(model_name=None)

            assert reranker.model_name == CrossEncoderReranker.DEFAULT_MODEL

    def test_init_with_invalid_model_name_type_raises_error(self):
        """Test initialization fails with invalid model name type."""
        with pytest.raises(ValidationError) as exc_info:
            CrossEncoderReranker(model_name=123)

        assert "non-empty string" in str(exc_info.value.message)


class TestCrossEncoderRerankerModelLoading:
    """Test model loading and caching."""

    def test_load_model_loads_cross_encoder(self):
        """Test that _load_model loads CrossEncoder."""
        mock_encoder = MagicMock()
        with patch('zapomni_core.search.reranker.CrossEncoder', return_value=mock_encoder):
            reranker = CrossEncoderReranker()
            reranker._load_model()

            assert reranker.model == mock_encoder

    def test_load_model_caches_model(self):
        """Test that _load_model caches the model."""
        mock_encoder = MagicMock()
        with patch('zapomni_core.search.reranker.CrossEncoder', return_value=mock_encoder) as mock_ce:
            reranker = CrossEncoderReranker()
            reranker._load_model()
            reranker._load_model()  # Call again

            # CrossEncoder should only be called once (cached)
            assert mock_ce.call_count == 1

    def test_load_model_failure_raises_search_error(self):
        """Test that model loading failure raises SearchError."""
        with patch('zapomni_core.search.reranker.CrossEncoder', side_effect=OSError("Model not found")):
            reranker = CrossEncoderReranker()

            with pytest.raises(SearchError) as exc_info:
                reranker._load_model()

            assert "Failed to load cross-encoder model" in str(exc_info.value.message)


class TestCrossEncoderRerankerRerank:
    """Test CrossEncoderReranker.rerank() method."""

    @pytest.fixture
    def mock_reranker(self):
        """Reranker instance with mocked model."""
        mock_encoder = MagicMock()
        with patch('zapomni_core.search.reranker.CrossEncoder', return_value=mock_encoder):
            reranker = CrossEncoderReranker()
            reranker._load_model()
            return reranker

    @pytest.fixture
    def sample_search_results(self):
        """Sample search results for testing."""
        return [
            SearchResult(
                memory_id="mem-1",
                content="Python is a programming language",
                relevance_score=0.8,
                metadata={"source": "docs"}
            ),
            SearchResult(
                memory_id="mem-2",
                content="Java is an object-oriented programming language",
                relevance_score=0.7,
                metadata={"source": "wiki"}
            ),
            SearchResult(
                memory_id="mem-3",
                content="C++ is a compiled programming language",
                relevance_score=0.6,
                metadata={"source": "manual"}
            ),
        ]

    @pytest.mark.asyncio
    async def test_rerank_with_query(self, mock_reranker, sample_search_results):
        """Test reranking search results with query."""
        query = "Python programming"

        # Mock the model predictions (raw scores)
        mock_reranker.model.predict.return_value = np.array([2.0, -1.0, -2.0])

        result = await mock_reranker.rerank(query=query, results=sample_search_results)

        assert len(result) == 3
        assert result[0].memory_id == "mem-1"  # Highest score
        # Verify scores were updated
        assert result[0].relevance_score > result[1].relevance_score

    @pytest.mark.asyncio
    async def test_rerank_empty_results(self, mock_reranker):
        """Test reranking with empty results list."""
        query = "test query"

        result = await mock_reranker.rerank(query=query, results=[])

        assert result == []

    @pytest.mark.asyncio
    async def test_rerank_single_result(self, mock_reranker, sample_search_results):
        """Test reranking with single result."""
        query = "test query"
        single_result = [sample_search_results[0]]

        mock_reranker.model.predict.return_value = np.array([1.5])

        result = await mock_reranker.rerank(query=query, results=single_result)

        assert len(result) == 1
        assert result[0].memory_id == "mem-1"

    @pytest.mark.asyncio
    async def test_rerank_with_top_k(self, mock_reranker, sample_search_results):
        """Test reranking with top_k limit."""
        query = "test query"

        mock_reranker.model.predict.return_value = np.array([2.0, 1.0, 0.5])

        result = await mock_reranker.rerank(
            query=query,
            results=sample_search_results,
            top_k=2
        )

        assert len(result) == 2
        assert result[0].memory_id == "mem-1"
        assert result[1].memory_id == "mem-2"

    @pytest.mark.asyncio
    async def test_rerank_empty_query_raises_error(self, mock_reranker):
        """Test reranking fails with empty query."""
        with pytest.raises(ValidationError):
            await mock_reranker.rerank(query="", results=[])

    @pytest.mark.asyncio
    async def test_rerank_invalid_results_type_raises_error(self, mock_reranker):
        """Test reranking fails with invalid results type."""
        with pytest.raises(ValidationError):
            await mock_reranker.rerank(query="test", results="not a list")

    @pytest.mark.asyncio
    async def test_rerank_result_without_content_raises_error(self, mock_reranker):
        """Test reranking fails when result has no content."""
        result = SearchResult(
            memory_id="mem-1",
            content="",  # Empty content
            relevance_score=0.8,
            metadata={}
        )

        with pytest.raises(ValidationError):
            await mock_reranker.rerank(query="test", results=[result])

    @pytest.mark.asyncio
    async def test_rerank_preserves_metadata(self, mock_reranker, sample_search_results):
        """Test that reranking preserves metadata."""
        query = "test"

        mock_reranker.model.predict.return_value = np.array([2.0, 1.0, 0.5])

        result = await mock_reranker.rerank(query=query, results=sample_search_results)

        for original, reranked in zip(sample_search_results, result):
            assert reranked.metadata == original.metadata

    @pytest.mark.asyncio
    async def test_rerank_sorts_descending(self, mock_reranker, sample_search_results):
        """Test that reranking sorts results in descending order."""
        query = "test"

        # Scores in descending order
        mock_reranker.model.predict.return_value = np.array([3.0, 1.0, 2.0])

        result = await mock_reranker.rerank(query=query, results=sample_search_results)

        # Should be sorted: 3.0, 2.0, 1.0
        scores = [r.relevance_score for r in result]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_rerank_with_fallback_on_model_error(self, mock_reranker, sample_search_results):
        """Test fallback to original scores when model fails."""
        query = "test"
        mock_reranker.model.predict.side_effect = Exception("Model error")

        result = await mock_reranker.rerank(query=query, results=sample_search_results)

        # Should return results sorted by original scores
        assert len(result) == 3
        # Should be sorted by original relevance_score (0.8, 0.7, 0.6)
        assert result[0].memory_id == "mem-1"

    @pytest.mark.asyncio
    async def test_rerank_without_fallback_raises_error(self, sample_search_results):
        """Test that reranking raises error without fallback on failure."""
        mock_encoder = MagicMock()
        mock_encoder.predict.side_effect = Exception("Model error")

        with patch('zapomni_core.search.reranker.CrossEncoder', return_value=mock_encoder):
            reranker = CrossEncoderReranker(fallback_enabled=False)
            reranker._load_model()

            with pytest.raises(SearchError):
                await reranker.rerank(query="test", results=sample_search_results)

    @pytest.mark.asyncio
    async def test_rerank_invalid_top_k_raises_error(self, mock_reranker, sample_search_results):
        """Test reranking fails with invalid top_k."""
        query = "test"

        with pytest.raises(ValidationError):
            await mock_reranker.rerank(
                query=query,
                results=sample_search_results,
                top_k=0
            )


class TestCrossEncoderRerankerScoring:
    """Test CrossEncoderReranker.score() method."""

    @pytest.fixture
    def mock_reranker(self):
        """Reranker instance with mocked model."""
        mock_encoder = MagicMock()
        with patch('zapomni_core.search.reranker.CrossEncoder', return_value=mock_encoder):
            reranker = CrossEncoderReranker()
            reranker._load_model()
            return reranker

    @pytest.mark.asyncio
    async def test_score_single_content(self, mock_reranker):
        """Test scoring a single query-content pair."""
        query = "Python programming"
        content = "Python is a programming language"

        mock_reranker.model.predict.return_value = np.array([2.0])

        score = await mock_reranker.score(query, content)

        assert isinstance(score, (float, np.floating))
        assert 0 <= score <= 1

    @pytest.mark.asyncio
    async def test_score_multiple_contents(self, mock_reranker):
        """Test scoring multiple query-content pairs."""
        query = "Python"
        contents = [
            "Python is a programming language",
            "Java is an object-oriented language",
            "C++ is compiled"
        ]

        mock_reranker.model.predict.return_value = np.array([2.0, -1.0, -2.0])

        scores = await mock_reranker.score(query, contents)

        assert len(scores) == 3
        assert all(0 <= s <= 1 for s in scores)

    @pytest.mark.asyncio
    async def test_score_empty_query_raises_error(self, mock_reranker):
        """Test scoring with empty query raises error."""
        with pytest.raises(ValidationError):
            await mock_reranker.score(query="", content="test")

    @pytest.mark.asyncio
    async def test_score_empty_content_raises_error(self, mock_reranker):
        """Test scoring with empty content raises error."""
        with pytest.raises(ValidationError):
            await mock_reranker.score(query="test", content="")

    @pytest.mark.asyncio
    async def test_score_empty_content_list_raises_error(self, mock_reranker):
        """Test scoring with empty content list raises error."""
        with pytest.raises(ValidationError):
            await mock_reranker.score(query="test", content=[])

    @pytest.mark.asyncio
    async def test_score_returns_float_for_string(self, mock_reranker):
        """Test that score returns float for string content."""
        query = "test"
        content = "test content"

        mock_reranker.model.predict.return_value = np.array([1.5])

        score = await mock_reranker.score(query, content)

        # Should be a float, not a list
        assert isinstance(score, (float, np.floating))

    @pytest.mark.asyncio
    async def test_score_returns_list_for_list(self, mock_reranker):
        """Test that score returns list for list content."""
        query = "test"
        contents = ["content1", "content2"]

        mock_reranker.model.predict.return_value = np.array([1.5, 2.0])

        scores = await mock_reranker.score(query, contents)

        # Should be a list
        assert isinstance(scores, list)
        assert len(scores) == 2


class TestCrossEncoderRerankerBatchProcessing:
    """Test batch processing in CrossEncoderReranker."""

    @pytest.fixture
    def mock_reranker(self):
        """Reranker instance with mocked model."""
        mock_encoder = MagicMock()
        with patch('zapomni_core.search.reranker.CrossEncoder', return_value=mock_encoder):
            reranker = CrossEncoderReranker()
            reranker._load_model()
            return reranker

    @pytest.mark.asyncio
    async def test_batch_rerank_multiple_queries(self, mock_reranker):
        """Test reranking multiple queries in batch."""
        batch_queries = [
            {
                "query": "Python",
                "results": [
                    SearchResult(memory_id="m1", content="Python code", relevance_score=0.8, metadata={})
                ]
            },
            {
                "query": "Java",
                "results": [
                    SearchResult(memory_id="m2", content="Java code", relevance_score=0.7, metadata={})
                ]
            }
        ]

        mock_reranker.model.predict.side_effect = [
            np.array([2.0]),
            np.array([1.5])
        ]

        results = await mock_reranker.rerank_batch(batch_queries)

        assert len(results) == 2
        assert results[0][0].memory_id == "m1"
        assert results[1][0].memory_id == "m2"

    @pytest.mark.asyncio
    async def test_batch_rerank_empty_queries_raises_error(self, mock_reranker):
        """Test batch reranking with empty queries raises error."""
        with pytest.raises(ValidationError):
            await mock_reranker.rerank_batch([])


class TestCrossEncoderRerankerErrorHandling:
    """Test error handling in CrossEncoderReranker."""

    @pytest.fixture
    def mock_reranker(self):
        """Reranker instance with mocked model."""
        mock_encoder = MagicMock()
        with patch('zapomni_core.search.reranker.CrossEncoder', return_value=mock_encoder):
            reranker = CrossEncoderReranker()
            reranker._load_model()
            return reranker

    @pytest.mark.asyncio
    async def test_rerank_model_loading_error_handled(self):
        """Test handling of model loading errors."""
        with patch('zapomni_core.search.reranker.CrossEncoder', side_effect=OSError("Model not found")):
            reranker = CrossEncoderReranker()

            with pytest.raises(SearchError):
                reranker._load_model()

    @pytest.mark.asyncio
    async def test_rerank_invalid_result_type(self, mock_reranker):
        """Test handling of invalid result type."""
        with pytest.raises(ValidationError):
            await mock_reranker.rerank(query="test", results=[{"not": "a_result"}])


class TestCrossEncoderRerankerEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def mock_reranker(self):
        """Reranker instance with mocked model."""
        mock_encoder = MagicMock()
        with patch('zapomni_core.search.reranker.CrossEncoder', return_value=mock_encoder):
            reranker = CrossEncoderReranker()
            reranker._load_model()
            return reranker

    @pytest.mark.asyncio
    async def test_rerank_very_long_query(self, mock_reranker):
        """Test reranking with very long query."""
        long_query = "Python programming language " * 100
        results = [
            SearchResult(memory_id="m1", content="Python", relevance_score=0.8, metadata={})
        ]

        mock_reranker.model.predict.return_value = np.array([1.5])

        result = await mock_reranker.rerank(query=long_query, results=results)

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_rerank_many_results(self, mock_reranker):
        """Test reranking with many results."""
        query = "test"
        results = [
            SearchResult(
                memory_id=f"m{i}",
                content=f"content {i}",
                relevance_score=0.5,
                metadata={}
            )
            for i in range(100)
        ]

        mock_reranker.model.predict.return_value = np.array(list(range(100, 0, -1)))

        result = await mock_reranker.rerank(query=query, results=results)

        assert len(result) == 100

    @pytest.mark.asyncio
    async def test_rerank_special_characters_in_content(self, mock_reranker):
        """Test reranking with special characters in content."""
        query = "test"
        results = [
            SearchResult(
                memory_id="m1",
                content="Test with special chars: !@#$%^&*()",
                relevance_score=0.8,
                metadata={}
            )
        ]

        mock_reranker.model.predict.return_value = np.array([1.5])

        result = await mock_reranker.rerank(query=query, results=results)

        assert len(result) == 1
        assert "!@#$%^&*()" in result[0].content

    @pytest.mark.asyncio
    async def test_rerank_unicode_content(self, mock_reranker):
        """Test reranking with unicode content."""
        query = "test"
        results = [
            SearchResult(
                memory_id="m1",
                content="Test with unicode: 中文, العربية, Ελληνικά",
                relevance_score=0.8,
                metadata={}
            )
        ]

        mock_reranker.model.predict.return_value = np.array([1.5])

        result = await mock_reranker.rerank(query=query, results=results)

        assert len(result) == 1
        assert "中文" in result[0].content

    @pytest.mark.asyncio
    async def test_score_normalization(self, mock_reranker):
        """Test that raw scores are properly normalized to [0, 1]."""
        query = "test"
        content = "test content"

        # Test various raw scores
        mock_reranker.model.predict.return_value = np.array([-10.0])  # Very negative

        score = await mock_reranker.score(query, content)

        assert 0 <= score <= 1  # Should be normalized


class TestCrossEncoderRerankerIntegration:
    """Integration tests for reranker with multiple scenarios."""

    @pytest.fixture
    def mock_reranker(self):
        """Reranker instance with mocked model."""
        mock_encoder = MagicMock()
        with patch('zapomni_core.search.reranker.CrossEncoder', return_value=mock_encoder):
            reranker = CrossEncoderReranker()
            reranker._load_model()
            return reranker

    @pytest.mark.asyncio
    async def test_rerank_improves_relevance_ranking(self, mock_reranker):
        """Test that reranking improves relevance ranking."""
        query = "Python Django web framework"
        results = [
            SearchResult(
                memory_id="m1",
                content="JavaScript frameworks",
                relevance_score=0.9,  # High initial score but low relevance
                metadata={}
            ),
            SearchResult(
                memory_id="m2",
                content="Django is a Python web framework",
                relevance_score=0.7,  # Lower initial score but high relevance
                metadata={}
            ),
        ]

        # Reranker scores: m2 is actually more relevant
        mock_reranker.model.predict.return_value = np.array([0.3, 2.0])

        reranked = await mock_reranker.rerank(query=query, results=results)

        # m2 should be ranked first after reranking
        assert reranked[0].memory_id == "m2"
        assert reranked[0].relevance_score > reranked[1].relevance_score

    @pytest.mark.asyncio
    async def test_rerank_pipeline_with_top_k_and_metadata(self, mock_reranker):
        """Test complete reranking pipeline with top_k and metadata preservation."""
        query = "machine learning"
        results = [
            SearchResult(
                memory_id="m1",
                content="Content 1",
                relevance_score=0.5,
                metadata={"source": "paper1", "year": 2020}
            ),
            SearchResult(
                memory_id="m2",
                content="Content 2",
                relevance_score=0.6,
                metadata={"source": "paper2", "year": 2021}
            ),
            SearchResult(
                memory_id="m3",
                content="Content 3",
                relevance_score=0.4,
                metadata={"source": "paper3", "year": 2022}
            ),
        ]

        mock_reranker.model.predict.return_value = np.array([2.0, 1.5, 0.5])

        reranked = await mock_reranker.rerank(
            query=query,
            results=results,
            top_k=2
        )

        assert len(reranked) == 2
        assert reranked[0].metadata["source"] == "paper1"
        assert reranked[1].metadata["source"] == "paper2"
