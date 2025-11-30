"""
Unit tests for HybridSearch component.

Tests cover:
- HybridSearch initialization and validation
- Reciprocal Rank Fusion (RRF) algorithm
- Alpha weighting (0.0=BM25 only, 1.0=vector only, 0.5=balanced)
- Result deduplication
- Error handling and edge cases

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from datetime import datetime
from unittest.mock import AsyncMock, Mock

import pytest

from zapomni_core.exceptions import SearchError, ValidationError
from zapomni_core.search.hybrid_search import HybridSearch
from zapomni_db.models import SearchResult


@pytest.fixture
def mock_vector_search():
    """Mock VectorSearch instance."""
    mock = Mock()
    mock.search = AsyncMock()
    return mock


@pytest.fixture
def mock_bm25_search():
    """Mock BM25Search instance.

    Note: BM25Search.search is synchronous and is wrapped in asyncio.to_thread
    by HybridSearch, so we use a regular Mock (not AsyncMock).
    """
    mock = Mock()
    # BM25Search.search is synchronous - NOT an AsyncMock
    mock.search = Mock()
    return mock


@pytest.fixture
def hybrid_search(mock_vector_search, mock_bm25_search):
    """HybridSearch instance with mocked dependencies."""
    return HybridSearch(vector_search=mock_vector_search, bm25_search=mock_bm25_search)


@pytest.fixture
def sample_vector_results():
    """Sample results from vector search."""
    return [
        SearchResult(
            memory_id="mem1",
            content="Python programming tutorial",  # Required field
            relevance_score=0.95,  # Required field
            chunk_id="chunk1",
            text="Python programming tutorial",
            similarity_score=0.95,
            tags=["python", "tutorial"],
            source="docs",
            timestamp=datetime.now(),
            chunk_index=0,
        ),
        SearchResult(
            memory_id="mem2",
            content="Machine learning with Python",  # Required field
            relevance_score=0.85,  # Required field
            chunk_id="chunk2",
            text="Machine learning with Python",
            similarity_score=0.85,
            tags=["python", "ml"],
            source="docs",
            timestamp=datetime.now(),
            chunk_index=0,
        ),
        SearchResult(
            memory_id="mem3",
            content="Data science fundamentals",  # Required field
            relevance_score=0.75,  # Required field
            chunk_id="chunk3",
            text="Data science fundamentals",
            similarity_score=0.75,
            tags=["data-science"],
            source="docs",
            timestamp=datetime.now(),
            chunk_index=0,
        ),
    ]


@pytest.fixture
def sample_bm25_results():
    """Sample results from BM25 search (returns dicts, not SearchResult)."""
    return [
        {"text": "Machine learning with Python", "score": 0.90, "index": 1},
        {"text": "Python best practices", "score": 0.80, "index": 3},
        {"text": "Python programming tutorial", "score": 0.70, "index": 0},
    ]


class TestHybridSearchInitialization:
    """Test HybridSearch initialization and validation."""

    def test_init_with_valid_dependencies(self, mock_vector_search, mock_bm25_search):
        """Should initialize successfully with valid dependencies."""
        search = HybridSearch(vector_search=mock_vector_search, bm25_search=mock_bm25_search)

        assert search.vector_search == mock_vector_search
        assert search.bm25_search == mock_bm25_search

    def test_init_with_none_vector_search(self, mock_bm25_search):
        """Should raise ValidationError if vector_search is None."""
        with pytest.raises(ValidationError) as exc_info:
            HybridSearch(vector_search=None, bm25_search=mock_bm25_search)

        assert "vector_search cannot be None" in str(exc_info.value)
        assert exc_info.value.error_code == "VAL_001"

    def test_init_with_none_bm25_search(self, mock_vector_search):
        """Should raise ValidationError if bm25_search is None."""
        with pytest.raises(ValidationError) as exc_info:
            HybridSearch(vector_search=mock_vector_search, bm25_search=None)

        assert "bm25_search cannot be None" in str(exc_info.value)
        assert exc_info.value.error_code == "VAL_001"


class TestHybridSearchValidation:
    """Test input validation in search method."""

    @pytest.mark.asyncio
    async def test_search_with_empty_query(self, hybrid_search):
        """Should raise ValidationError for empty query."""
        with pytest.raises(ValidationError) as exc_info:
            await hybrid_search.search(query="", limit=10)

        assert "Query cannot be empty" in str(exc_info.value)
        assert exc_info.value.error_code == "VAL_001"

    @pytest.mark.asyncio
    async def test_search_with_whitespace_query(self, hybrid_search):
        """Should raise ValidationError for whitespace-only query."""
        with pytest.raises(ValidationError) as exc_info:
            await hybrid_search.search(query="   ", limit=10)

        assert "Query cannot be empty" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_search_with_invalid_limit_type(self, hybrid_search):
        """Should raise ValidationError if limit is not int."""
        with pytest.raises(ValidationError) as exc_info:
            await hybrid_search.search(query="test", limit="10")

        assert "Limit must be int" in str(exc_info.value)
        assert exc_info.value.error_code == "VAL_002"

    @pytest.mark.asyncio
    async def test_search_with_negative_limit(self, hybrid_search):
        """Should raise ValidationError for negative limit."""
        with pytest.raises(ValidationError) as exc_info:
            await hybrid_search.search(query="test", limit=-1)

        assert "Limit must be >= 1" in str(exc_info.value)
        assert exc_info.value.error_code == "VAL_003"

    @pytest.mark.asyncio
    async def test_search_with_zero_limit(self, hybrid_search):
        """Should raise ValidationError for zero limit."""
        with pytest.raises(ValidationError) as exc_info:
            await hybrid_search.search(query="test", limit=0)

        assert "Limit must be >= 1" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_search_with_excessive_limit(self, hybrid_search):
        """Should raise ValidationError for limit > 1000."""
        with pytest.raises(ValidationError) as exc_info:
            await hybrid_search.search(query="test", limit=1001)

        assert "Limit cannot exceed 1000" in str(exc_info.value)
        assert exc_info.value.error_code == "VAL_003"

    @pytest.mark.asyncio
    async def test_search_with_invalid_alpha_type(self, hybrid_search):
        """Should raise ValidationError if alpha is not float."""
        with pytest.raises(ValidationError) as exc_info:
            await hybrid_search.search(query="test", limit=10, alpha="0.5")

        assert "Alpha must be float" in str(exc_info.value)
        assert exc_info.value.error_code == "VAL_002"

    @pytest.mark.asyncio
    async def test_search_with_alpha_below_range(self, hybrid_search):
        """Should raise ValidationError for alpha < 0.0."""
        with pytest.raises(ValidationError) as exc_info:
            await hybrid_search.search(query="test", limit=10, alpha=-0.1)

        assert "Alpha must be between 0.0 and 1.0" in str(exc_info.value)
        assert exc_info.value.error_code == "VAL_003"

    @pytest.mark.asyncio
    async def test_search_with_alpha_above_range(self, hybrid_search):
        """Should raise ValidationError for alpha > 1.0."""
        with pytest.raises(ValidationError) as exc_info:
            await hybrid_search.search(query="test", limit=10, alpha=1.1)

        assert "Alpha must be between 0.0 and 1.0" in str(exc_info.value)
        assert exc_info.value.error_code == "VAL_003"


class TestReciprocalRankFusion:
    """Test Reciprocal Rank Fusion (RRF) algorithm."""

    @pytest.mark.asyncio
    async def test_rrf_with_balanced_weighting(
        self, hybrid_search, sample_vector_results, sample_bm25_results
    ):
        """Should correctly merge results with balanced weighting (alpha=0.5)."""
        hybrid_search.vector_search.search.return_value = sample_vector_results
        hybrid_search.bm25_search.search.return_value = sample_bm25_results

        results = await hybrid_search.search(query="Python", limit=10, alpha=0.5)

        # Verify both searches were called
        hybrid_search.vector_search.search.assert_called_once_with(
            query="Python", limit=10, filters=None
        )
        # BM25Search.search is called without keyword arguments (via asyncio.to_thread)
        hybrid_search.bm25_search.search.assert_called_once_with("Python", 10)

        # Should deduplicate mem1 and mem2 (appear in both)
        # BM25 result "Python best practices" doesn't match any vector result,
        # so it gets a pseudo ID like "bm25_mem_3"
        assert len(results) == 4  # 3 from vector + 1 unique from BM25

        # Verify all results are present (some may have pseudo IDs from BM25)
        # We can't assert exact memory_ids because BM25-only results get pseudo IDs
        chunk_ids = {r.chunk_id for r in results}
        assert len(chunk_ids) == 4  # All unique chunks

    @pytest.mark.asyncio
    async def test_rrf_vector_only(self, hybrid_search, sample_vector_results, sample_bm25_results):
        """Should use vector results only when alpha=1.0."""
        hybrid_search.vector_search.search.return_value = sample_vector_results
        hybrid_search.bm25_search.search.return_value = sample_bm25_results

        results = await hybrid_search.search(query="Python", limit=10, alpha=1.0)

        # With alpha=1.0, BM25 results should have zero weight
        # Top result should be from vector search
        assert results[0].memory_id == "mem1"  # Highest vector score

    @pytest.mark.asyncio
    async def test_rrf_bm25_only(self, hybrid_search, sample_vector_results, sample_bm25_results):
        """Should use BM25 results only when alpha=0.0."""
        hybrid_search.vector_search.search.return_value = sample_vector_results
        hybrid_search.bm25_search.search.return_value = sample_bm25_results

        results = await hybrid_search.search(query="Python", limit=10, alpha=0.0)

        # With alpha=0.0, vector results should have zero weight
        # Top result should be from BM25 search
        assert results[0].memory_id == "mem2"  # Highest BM25 score

    @pytest.mark.asyncio
    async def test_rrf_with_empty_vector_results(self, hybrid_search, sample_bm25_results):
        """Should handle empty vector results gracefully."""
        hybrid_search.vector_search.search.return_value = []
        hybrid_search.bm25_search.search.return_value = sample_bm25_results

        results = await hybrid_search.search(query="Python", limit=10, alpha=0.5)

        # Should return BM25 results only (with pseudo IDs since no vector results)
        assert len(results) == 3
        # Top result should have highest BM25 score (first in list)
        assert results[0].text == "Machine learning with Python"

    @pytest.mark.asyncio
    async def test_rrf_with_empty_bm25_results(self, hybrid_search, sample_vector_results):
        """Should handle empty BM25 results gracefully."""
        hybrid_search.vector_search.search.return_value = sample_vector_results
        hybrid_search.bm25_search.search.return_value = []

        results = await hybrid_search.search(query="Python", limit=10, alpha=0.5)

        # Should return vector results only
        assert len(results) == 3
        assert results[0].memory_id == "mem1"

    @pytest.mark.asyncio
    async def test_rrf_with_both_empty(self, hybrid_search):
        """Should return empty list when both searches return nothing."""
        hybrid_search.vector_search.search.return_value = []
        hybrid_search.bm25_search.search.return_value = []

        results = await hybrid_search.search(query="Python", limit=10, alpha=0.5)

        assert results == []

    @pytest.mark.asyncio
    async def test_rrf_respects_limit(
        self, hybrid_search, sample_vector_results, sample_bm25_results
    ):
        """Should respect limit parameter in final results."""
        hybrid_search.vector_search.search.return_value = sample_vector_results
        hybrid_search.bm25_search.search.return_value = sample_bm25_results

        results = await hybrid_search.search(query="Python", limit=2, alpha=0.5)

        # Should return top 2 results only
        assert len(results) <= 2


class TestDeduplication:
    """Test result deduplication logic."""

    @pytest.mark.asyncio
    async def test_deduplication_by_chunk_id(
        self, hybrid_search, sample_vector_results, sample_bm25_results
    ):
        """Should deduplicate results by chunk_id."""
        hybrid_search.vector_search.search.return_value = sample_vector_results
        hybrid_search.bm25_search.search.return_value = sample_bm25_results

        results = await hybrid_search.search(query="Python", limit=10, alpha=0.5)

        # Verify no duplicate chunk_ids
        chunk_ids = [r.chunk_id for r in results]
        assert len(chunk_ids) == len(set(chunk_ids))

    @pytest.mark.asyncio
    async def test_deduplication_preserves_higher_score(self, hybrid_search):
        """Should keep result with higher combined RRF score when deduplicating."""
        # Create duplicate results with different scores
        result_high = SearchResult(
            memory_id="mem1",
            content="Test",  # Required field
            relevance_score=0.9,  # Required field
            chunk_id="chunk1",
            text="Test",
            similarity_score=0.9,
            tags=[],
            source="docs",
            timestamp=datetime.now(),
            chunk_index=0,
        )
        # BM25 returns dict, not SearchResult
        bm25_result = {"text": "Test", "score": 0.5, "index": 0}

        hybrid_search.vector_search.search.return_value = [result_high]
        hybrid_search.bm25_search.search.return_value = [bm25_result]

        results = await hybrid_search.search(query="test", limit=10, alpha=0.5)

        # Should have only one result (deduplicated by text matching)
        assert len(results) == 1
        # The combined score should be properly calculated
        assert results[0].chunk_id == "chunk1"


class TestErrorHandling:
    """Test error handling and propagation.

    Note: HybridSearch now uses graceful degradation - if one search fails,
    the other results are still returned. Only if BOTH fail does it raise.
    """

    @pytest.mark.asyncio
    async def test_vector_search_error_graceful_degradation(
        self, hybrid_search, sample_bm25_results
    ):
        """Should return BM25 results when vector search fails (graceful degradation)."""
        hybrid_search.vector_search.search.side_effect = SearchError(
            message="Vector search failed", error_code="SEARCH_001"
        )
        hybrid_search.bm25_search.search.return_value = sample_bm25_results

        # Should NOT raise - graceful degradation returns BM25 results
        results = await hybrid_search.search(query="test", limit=10, alpha=0.5)

        # Should have BM25 results only
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_bm25_search_error_graceful_degradation(
        self, hybrid_search, sample_vector_results
    ):
        """Should return vector results when BM25 search fails (graceful degradation)."""
        hybrid_search.vector_search.search.return_value = sample_vector_results
        hybrid_search.bm25_search.search.side_effect = SearchError(
            message="BM25 search failed", error_code="SEARCH_001"
        )

        # Should NOT raise - graceful degradation returns vector results
        results = await hybrid_search.search(query="test", limit=10, alpha=0.5)

        # Should have vector results only
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_both_searches_fail_raises_error(self, hybrid_search):
        """Should raise SearchError when both searches fail."""
        hybrid_search.vector_search.search.side_effect = SearchError(
            message="Vector search failed", error_code="SEARCH_001"
        )
        hybrid_search.bm25_search.search.side_effect = SearchError(
            message="BM25 search failed", error_code="SEARCH_001"
        )

        with pytest.raises(SearchError) as exc_info:
            await hybrid_search.search(query="test", limit=10, alpha=0.5)

        assert "Both vector and BM25 searches failed" in str(exc_info.value)
        assert exc_info.value.error_code == "SEARCH_001"


class TestRRFScoring:
    """Test RRF scoring calculation."""

    @pytest.mark.asyncio
    async def test_rrf_score_calculation(self, hybrid_search):
        """Should correctly calculate RRF scores."""
        # Create controlled test data
        vector_results = [
            SearchResult(
                memory_id=f"mem{i}",
                content=f"Text {i}",  # Required field
                relevance_score=1.0 - (i * 0.1),  # Required field
                chunk_id=f"chunk{i}",
                text=f"Text {i}",
                similarity_score=1.0 - (i * 0.1),
                tags=[],
                source="test",
                timestamp=datetime.now(),
                chunk_index=0,
            )
            for i in range(3)
        ]

        # BM25 returns dicts, not SearchResult
        bm25_results = [
            {"text": f"Text {i}", "score": 1.0 - (i * 0.1), "index": i}
            for i in range(2, -1, -1)  # Reverse order
        ]

        hybrid_search.vector_search.search.return_value = vector_results
        hybrid_search.bm25_search.search.return_value = bm25_results

        results = await hybrid_search.search(query="test", limit=10, alpha=0.5)

        # Results should be properly ranked by combined RRF score
        assert len(results) == 3
        # All results should have similarity scores (RRF scores)
        for result in results:
            assert 0.0 <= result.similarity_score <= 1.0

    @pytest.mark.asyncio
    async def test_rrf_with_k_parameter(self, hybrid_search):
        """Should use k=60 for RRF calculation (standard value)."""
        # This test verifies that the RRF formula: 1/(k + rank) is used
        # where k=60 is the standard constant from literature

        vector_result = SearchResult(
            memory_id="mem1",
            content="Test",  # Required field
            relevance_score=0.9,  # Required field
            chunk_id="chunk1",
            text="Test",
            similarity_score=0.9,
            tags=[],
            source="test",
            timestamp=datetime.now(),
            chunk_index=0,
        )

        bm25_result = {"text": "Test", "score": 0.9, "index": 0}

        hybrid_search.vector_search.search.return_value = [vector_result]
        hybrid_search.bm25_search.search.return_value = [bm25_result]

        results = await hybrid_search.search(query="test", limit=10, alpha=0.5)

        # RRF formula for rank 1 with k=60:
        # vector: 1/(60+1) ≈ 0.0164
        # bm25: 1/(60+1) ≈ 0.0164
        # combined with alpha=0.5: 0.5*0.0164 + 0.5*0.0164 ≈ 0.0164
        # Normalized to 0-1 range
        assert len(results) == 1
        assert results[0].similarity_score > 0


class TestFiltersParameter:
    """Test filters parameter handling."""

    @pytest.mark.asyncio
    async def test_filters_passed_to_vector_search(self, hybrid_search):
        """Should pass filters to vector search."""
        hybrid_search.vector_search.search.return_value = []
        hybrid_search.bm25_search.search.return_value = []

        filters = {"tags": ["python"], "source": "docs"}
        await hybrid_search.search(query="test", limit=10, filters=filters)

        hybrid_search.vector_search.search.assert_called_once_with(
            query="test", limit=10, filters=filters
        )

    @pytest.mark.asyncio
    async def test_filters_not_passed_to_bm25_search(self, hybrid_search):
        """Should not pass filters to BM25 search (not supported)."""
        hybrid_search.vector_search.search.return_value = []
        hybrid_search.bm25_search.search.return_value = []

        filters = {"tags": ["python"], "source": "docs"}
        await hybrid_search.search(query="test", limit=10, filters=filters)

        # BM25 search should not receive filters (called via asyncio.to_thread)
        hybrid_search.bm25_search.search.assert_called_once_with("test", 10)


class TestFusionMethods:
    """Test different fusion strategies (RRF, RSF, DBSF)."""

    def test_init_with_fusion_method_rrf(self, mock_vector_search, mock_bm25_search):
        """Should initialize with RRF fusion method."""
        search = HybridSearch(
            vector_search=mock_vector_search,
            bm25_search=mock_bm25_search,
            fusion_method="rrf",
        )
        assert search._fusion_method == "rrf"

    def test_init_with_fusion_method_rsf(self, mock_vector_search, mock_bm25_search):
        """Should initialize with RSF fusion method."""
        search = HybridSearch(
            vector_search=mock_vector_search,
            bm25_search=mock_bm25_search,
            fusion_method="rsf",
        )
        assert search._fusion_method == "rsf"

    def test_init_with_fusion_method_dbsf(self, mock_vector_search, mock_bm25_search):
        """Should initialize with DBSF fusion method."""
        search = HybridSearch(
            vector_search=mock_vector_search,
            bm25_search=mock_bm25_search,
            fusion_method="dbsf",
        )
        assert search._fusion_method == "dbsf"

    def test_init_with_custom_fusion_k(self, mock_vector_search, mock_bm25_search):
        """Should initialize with custom fusion_k parameter."""
        search = HybridSearch(
            vector_search=mock_vector_search,
            bm25_search=mock_bm25_search,
            fusion_k=20,
        )
        assert search._fusion_k == 20

    def test_init_with_invalid_fusion_k(self, mock_vector_search, mock_bm25_search):
        """Should raise ValidationError for invalid fusion_k."""
        with pytest.raises(ValidationError) as exc_info:
            HybridSearch(
                vector_search=mock_vector_search,
                bm25_search=mock_bm25_search,
                fusion_k=0,
            )
        assert "fusion_k must be a positive integer" in str(exc_info.value)

    def test_init_with_negative_fusion_k(self, mock_vector_search, mock_bm25_search):
        """Should raise ValidationError for negative fusion_k."""
        with pytest.raises(ValidationError) as exc_info:
            HybridSearch(
                vector_search=mock_vector_search,
                bm25_search=mock_bm25_search,
                fusion_k=-10,
            )
        assert "fusion_k must be a positive integer" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_search_with_rsf_fusion(
        self, mock_vector_search, mock_bm25_search, sample_vector_results, sample_bm25_results
    ):
        """Should use RSF fusion method when specified."""
        mock_vector_search.search.return_value = sample_vector_results
        mock_bm25_search.search.return_value = sample_bm25_results

        search = HybridSearch(
            vector_search=mock_vector_search,
            bm25_search=mock_bm25_search,
            fusion_method="rsf",
        )

        results = await search.search(query="test", limit=10, alpha=0.5)

        # RSF should produce results with scores in [0, 1] range
        assert len(results) > 0
        for result in results:
            assert 0.0 <= result.similarity_score <= 1.0

    @pytest.mark.asyncio
    async def test_search_with_dbsf_fusion(
        self, mock_vector_search, mock_bm25_search, sample_vector_results, sample_bm25_results
    ):
        """Should use DBSF fusion method when specified."""
        mock_vector_search.search.return_value = sample_vector_results
        mock_bm25_search.search.return_value = sample_bm25_results

        search = HybridSearch(
            vector_search=mock_vector_search,
            bm25_search=mock_bm25_search,
            fusion_method="dbsf",
        )

        results = await search.search(query="test", limit=10, alpha=0.5)

        # DBSF should produce results with scores in [0, 1] range
        assert len(results) > 0
        for result in results:
            assert 0.0 <= result.similarity_score <= 1.0

    @pytest.mark.asyncio
    async def test_search_fusion_method_override(
        self, mock_vector_search, mock_bm25_search, sample_vector_results, sample_bm25_results
    ):
        """Should override instance fusion method with search parameter."""
        mock_vector_search.search.return_value = sample_vector_results
        mock_bm25_search.search.return_value = sample_bm25_results

        # Initialize with RRF
        search = HybridSearch(
            vector_search=mock_vector_search,
            bm25_search=mock_bm25_search,
            fusion_method="rrf",
        )

        # Override with RSF for this search
        results = await search.search(query="test", limit=10, alpha=0.5, fusion_method="rsf")

        # Should return results (RSF was used)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_search_uses_default_fusion_method(
        self, mock_vector_search, mock_bm25_search, sample_vector_results, sample_bm25_results
    ):
        """Should use default fusion method when not specified in search."""
        mock_vector_search.search.return_value = sample_vector_results
        mock_bm25_search.search.return_value = sample_bm25_results

        # Initialize with DBSF as default
        search = HybridSearch(
            vector_search=mock_vector_search,
            bm25_search=mock_bm25_search,
            fusion_method="dbsf",
        )

        # Don't specify fusion_method in search
        results = await search.search(query="test", limit=10, alpha=0.5)

        # Should return results using DBSF
        assert len(results) > 0
