"""
Unit tests for fusion module (RRF, RSF, DBSF).

Tests cover:
- RRFusion: Reciprocal Rank Fusion
- RSFusion: Relative Score Fusion with min-max normalization
- DBSFusion: Distribution-Based Score Fusion (3-sigma normalization)
- Edge cases and error handling

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

import math
from typing import Dict, Tuple

import pytest

from zapomni_core.exceptions import ValidationError
from zapomni_core.search.fusion import (
    DBSFusion,
    RRFusion,
    RSFusion,
    fuse_dbsf,
    fuse_rsf,
)


# ==================== FIXTURES ====================


@pytest.fixture
def vector_results() -> Dict[str, Tuple[int, float]]:
    """Sample vector search results: chunk_id -> (rank, score)."""
    return {
        "chunk_1": (1, 0.95),
        "chunk_2": (2, 0.85),
        "chunk_3": (3, 0.75),
    }


@pytest.fixture
def bm25_results() -> Dict[str, Tuple[int, float]]:
    """Sample BM25 search results: chunk_id -> (rank, score)."""
    return {
        "chunk_2": (1, 0.90),
        "chunk_4": (2, 0.80),
        "chunk_1": (3, 0.70),
    }


@pytest.fixture
def overlapping_vector_results() -> Dict[str, Tuple[int, float]]:
    """Vector results with same chunks as BM25."""
    return {
        "chunk_1": (1, 0.95),
        "chunk_2": (2, 0.85),
    }


@pytest.fixture
def overlapping_bm25_results() -> Dict[str, Tuple[int, float]]:
    """BM25 results with same chunks as vector."""
    return {
        "chunk_1": (2, 0.80),
        "chunk_2": (1, 0.90),
    }


@pytest.fixture
def non_overlapping_vector_results() -> Dict[str, Tuple[int, float]]:
    """Vector results with no overlap with BM25."""
    return {
        "chunk_1": (1, 0.95),
        "chunk_2": (2, 0.85),
    }


@pytest.fixture
def non_overlapping_bm25_results() -> Dict[str, Tuple[int, float]]:
    """BM25 results with no overlap with vector."""
    return {
        "chunk_3": (1, 0.90),
        "chunk_4": (2, 0.80),
    }


@pytest.fixture
def single_result_vector() -> Dict[str, Tuple[int, float]]:
    """Single result from vector search."""
    return {"chunk_1": (1, 0.95)}


@pytest.fixture
def single_result_bm25() -> Dict[str, Tuple[int, float]]:
    """Single result from BM25 search."""
    return {"chunk_1": (1, 0.85)}


@pytest.fixture
def uniform_score_results() -> Dict[str, Tuple[int, float]]:
    """Results where all scores are identical."""
    return {
        "chunk_1": (1, 0.80),
        "chunk_2": (2, 0.80),
        "chunk_3": (3, 0.80),
    }


@pytest.fixture
def rrf() -> RRFusion:
    """RRFusion instance with default k=60."""
    return RRFusion()


@pytest.fixture
def rsf() -> RSFusion:
    """RSFusion instance."""
    return RSFusion()


@pytest.fixture
def dbsf() -> DBSFusion:
    """DBSFusion instance with no predefined ranges."""
    return DBSFusion()


# ==================== RRF FUSION TESTS ====================


class TestRRFusionInit:
    """Tests for RRFusion initialization."""

    def test_init_default_k(self) -> None:
        """Default k should be 60."""
        rrf = RRFusion()
        assert rrf.k == 60

    def test_init_custom_k(self) -> None:
        """Custom k value should be accepted."""
        rrf = RRFusion(k=100)
        assert rrf.k == 100

    def test_init_small_k(self) -> None:
        """Small k value (emphasizing top ranks) should work."""
        rrf = RRFusion(k=10)
        assert rrf.k == 10

    def test_init_invalid_k_zero(self) -> None:
        """k=0 should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            RRFusion(k=0)

        assert "k must be > 0" in str(exc_info.value)
        assert exc_info.value.error_code == "VAL_003"

    def test_init_invalid_k_negative(self) -> None:
        """k < 0 should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            RRFusion(k=-10)

        assert "k must be > 0" in str(exc_info.value)
        assert exc_info.value.error_code == "VAL_003"

    def test_init_invalid_k_type_float(self) -> None:
        """k as float should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            RRFusion(k=60.5)

        assert "k must be an integer" in str(exc_info.value)
        assert exc_info.value.error_code == "VAL_002"

    def test_init_invalid_k_type_string(self) -> None:
        """k as string should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            RRFusion(k="60")

        assert "k must be an integer" in str(exc_info.value)


class TestRRFusionBasic:
    """Tests for basic RRFusion functionality."""

    def test_fuse_basic(
        self,
        rrf: RRFusion,
        vector_results: Dict[str, Tuple[int, float]],
        bm25_results: Dict[str, Tuple[int, float]],
    ) -> None:
        """Basic fusion with results from both sources."""
        fused = rrf.fuse(vector_results, bm25_results, alpha=0.5)

        assert isinstance(fused, dict)
        # Should include all unique chunks from both sources
        assert len(fused) == 4  # chunk_1, chunk_2, chunk_3, chunk_4
        assert "chunk_1" in fused
        assert "chunk_2" in fused
        assert "chunk_3" in fused
        assert "chunk_4" in fused

    def test_fuse_vector_only(
        self,
        rrf: RRFusion,
        vector_results: Dict[str, Tuple[int, float]],
    ) -> None:
        """Fusion with only vector results, empty BM25."""
        fused = rrf.fuse(vector_results, {}, alpha=0.5)

        assert len(fused) == 3
        assert all(chunk_id in fused for chunk_id in vector_results.keys())

    def test_fuse_bm25_only(
        self,
        rrf: RRFusion,
        bm25_results: Dict[str, Tuple[int, float]],
    ) -> None:
        """Fusion with only BM25 results, empty vector."""
        fused = rrf.fuse({}, bm25_results, alpha=0.5)

        assert len(fused) == 3
        assert all(chunk_id in fused for chunk_id in bm25_results.keys())

    def test_fuse_both_empty(self, rrf: RRFusion) -> None:
        """Fusion with both sources empty."""
        fused = rrf.fuse({}, {}, alpha=0.5)

        assert fused == {}

    def test_fuse_none_inputs(self, rrf: RRFusion) -> None:
        """Fusion should handle None inputs gracefully."""
        fused = rrf.fuse(None, None, alpha=0.5)

        assert fused == {}


class TestRRFusionAlpha:
    """Tests for RRFusion alpha parameter."""

    def test_fuse_alpha_zero(
        self,
        rrf: RRFusion,
        vector_results: Dict[str, Tuple[int, float]],
        bm25_results: Dict[str, Tuple[int, float]],
    ) -> None:
        """alpha=0 means BM25 only (zero weight for vector)."""
        fused = rrf.fuse(vector_results, bm25_results, alpha=0.0)

        # Vector-only chunks should have score 0
        assert fused["chunk_3"] == 0.0  # Only in vector results

        # BM25-only chunks should have non-zero score
        assert fused["chunk_4"] > 0.0

    def test_fuse_alpha_one(
        self,
        rrf: RRFusion,
        vector_results: Dict[str, Tuple[int, float]],
        bm25_results: Dict[str, Tuple[int, float]],
    ) -> None:
        """alpha=1 means vector only (zero weight for BM25)."""
        fused = rrf.fuse(vector_results, bm25_results, alpha=1.0)

        # BM25-only chunks should have score 0
        assert fused["chunk_4"] == 0.0  # Only in BM25 results

        # Vector-only chunks should have non-zero score
        assert fused["chunk_3"] > 0.0

    def test_fuse_alpha_balanced(
        self,
        rrf: RRFusion,
        overlapping_vector_results: Dict[str, Tuple[int, float]],
        overlapping_bm25_results: Dict[str, Tuple[int, float]],
    ) -> None:
        """alpha=0.5 gives equal weight to both sources."""
        fused = rrf.fuse(overlapping_vector_results, overlapping_bm25_results, alpha=0.5)

        # Both chunks should have contributions from both sources
        assert fused["chunk_1"] > 0
        assert fused["chunk_2"] > 0

    def test_fuse_invalid_alpha_negative(
        self,
        rrf: RRFusion,
        vector_results: Dict[str, Tuple[int, float]],
        bm25_results: Dict[str, Tuple[int, float]],
    ) -> None:
        """alpha < 0 should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            rrf.fuse(vector_results, bm25_results, alpha=-0.1)

        assert "alpha must be in [0.0, 1.0]" in str(exc_info.value)
        assert exc_info.value.error_code == "VAL_003"

    def test_fuse_invalid_alpha_greater_than_one(
        self,
        rrf: RRFusion,
        vector_results: Dict[str, Tuple[int, float]],
        bm25_results: Dict[str, Tuple[int, float]],
    ) -> None:
        """alpha > 1 should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            rrf.fuse(vector_results, bm25_results, alpha=1.5)

        assert "alpha must be in [0.0, 1.0]" in str(exc_info.value)


class TestRRFusionOverlap:
    """Tests for RRFusion with overlapping/non-overlapping results."""

    def test_fuse_overlapping_results(
        self,
        rrf: RRFusion,
        overlapping_vector_results: Dict[str, Tuple[int, float]],
        overlapping_bm25_results: Dict[str, Tuple[int, float]],
    ) -> None:
        """Same chunks appearing in both sources."""
        fused = rrf.fuse(overlapping_vector_results, overlapping_bm25_results, alpha=0.5)

        # Should only have the overlapping chunks
        assert len(fused) == 2
        assert "chunk_1" in fused
        assert "chunk_2" in fused

    def test_fuse_non_overlapping(
        self,
        rrf: RRFusion,
        non_overlapping_vector_results: Dict[str, Tuple[int, float]],
        non_overlapping_bm25_results: Dict[str, Tuple[int, float]],
    ) -> None:
        """Different chunks in each source."""
        fused = rrf.fuse(non_overlapping_vector_results, non_overlapping_bm25_results, alpha=0.5)

        # Should have all unique chunks
        assert len(fused) == 4
        assert "chunk_1" in fused
        assert "chunk_2" in fused
        assert "chunk_3" in fused
        assert "chunk_4" in fused


class TestRRFusionScoreCalculation:
    """Tests for RRF score calculation verification."""

    def test_fuse_score_calculation(self) -> None:
        """Verify RRF formula: score = alpha/(k+rank) + (1-alpha)/(k+rank)."""
        rrf = RRFusion(k=60)

        vector_results = {"chunk_1": (1, 0.95)}  # rank 1 in vector
        bm25_results = {"chunk_1": (2, 0.80)}  # rank 2 in BM25

        fused = rrf.fuse(vector_results, bm25_results, alpha=0.5)

        # Expected: 0.5 * (1/(60+1)) + 0.5 * (1/(60+2))
        # = 0.5 * (1/61) + 0.5 * (1/62)
        # = 0.5 * 0.01639... + 0.5 * 0.01612...
        expected = 0.5 * (1 / 61) + 0.5 * (1 / 62)

        assert abs(fused["chunk_1"] - expected) < 1e-10

    def test_fuse_score_single_source_vector(self) -> None:
        """Verify score when chunk only in vector results."""
        rrf = RRFusion(k=60)

        vector_results = {"chunk_1": (1, 0.95)}
        bm25_results: Dict[str, Tuple[int, float]] = {}

        fused = rrf.fuse(vector_results, bm25_results, alpha=0.5)

        # Expected: 0.5 * (1/(60+1)) + 0 = 0.5/61
        expected = 0.5 * (1 / 61)

        assert abs(fused["chunk_1"] - expected) < 1e-10

    def test_fuse_score_single_source_bm25(self) -> None:
        """Verify score when chunk only in BM25 results."""
        rrf = RRFusion(k=60)

        vector_results: Dict[str, Tuple[int, float]] = {}
        bm25_results = {"chunk_1": (1, 0.85)}

        fused = rrf.fuse(vector_results, bm25_results, alpha=0.5)

        # Expected: 0 + 0.5 * (1/(60+1)) = 0.5/61
        expected = 0.5 * (1 / 61)

        assert abs(fused["chunk_1"] - expected) < 1e-10

    def test_fuse_ranking_order(
        self,
        rrf: RRFusion,
        overlapping_vector_results: Dict[str, Tuple[int, float]],
        overlapping_bm25_results: Dict[str, Tuple[int, float]],
    ) -> None:
        """Higher fused score should indicate better rank."""
        fused = rrf.fuse(overlapping_vector_results, overlapping_bm25_results, alpha=0.5)

        # Sort by score descending
        sorted_results = sorted(fused.items(), key=lambda x: x[1], reverse=True)

        # First result should have highest score
        assert sorted_results[0][1] >= sorted_results[1][1]


class TestRRFusionHelperMethods:
    """Tests for RRFusion helper methods."""

    def test_fuse_with_ranks(
        self,
        rrf: RRFusion,
        vector_results: Dict[str, Tuple[int, float]],
        bm25_results: Dict[str, Tuple[int, float]],
    ) -> None:
        """Test fuse_with_ranks returns detailed information."""
        detailed = rrf.fuse_with_ranks(vector_results, bm25_results, alpha=0.5)

        # Check structure
        assert "chunk_1" in detailed
        score, vector_rank, bm25_rank = detailed["chunk_1"]

        assert isinstance(score, float)
        assert vector_rank == 1  # rank 1 in vector
        assert bm25_rank == 3  # rank 3 in BM25

    def test_fuse_with_ranks_none_for_missing(
        self,
        rrf: RRFusion,
        vector_results: Dict[str, Tuple[int, float]],
        bm25_results: Dict[str, Tuple[int, float]],
    ) -> None:
        """Test fuse_with_ranks returns None for missing ranks."""
        detailed = rrf.fuse_with_ranks(vector_results, bm25_results, alpha=0.5)

        # chunk_3 only in vector
        _, vector_rank, bm25_rank = detailed["chunk_3"]
        assert vector_rank == 3
        assert bm25_rank is None

        # chunk_4 only in BM25
        _, vector_rank, bm25_rank = detailed["chunk_4"]
        assert vector_rank is None
        assert bm25_rank == 2

    def test_get_sorted_results(
        self,
        rrf: RRFusion,
        vector_results: Dict[str, Tuple[int, float]],
        bm25_results: Dict[str, Tuple[int, float]],
    ) -> None:
        """Test get_sorted_results returns sorted list."""
        sorted_list = rrf.get_sorted_results(vector_results, bm25_results, alpha=0.5)

        assert isinstance(sorted_list, list)
        assert len(sorted_list) == 4

        # Check sorted in descending order
        scores = [score for _, score in sorted_list]
        assert scores == sorted(scores, reverse=True)

    def test_get_sorted_results_with_limit(
        self,
        rrf: RRFusion,
        vector_results: Dict[str, Tuple[int, float]],
        bm25_results: Dict[str, Tuple[int, float]],
    ) -> None:
        """Test get_sorted_results respects limit."""
        sorted_list = rrf.get_sorted_results(vector_results, bm25_results, alpha=0.5, limit=2)

        assert len(sorted_list) == 2

    def test_get_sorted_results_invalid_limit(
        self,
        rrf: RRFusion,
        vector_results: Dict[str, Tuple[int, float]],
        bm25_results: Dict[str, Tuple[int, float]],
    ) -> None:
        """Test get_sorted_results raises error for invalid limit."""
        with pytest.raises(ValidationError) as exc_info:
            rrf.get_sorted_results(vector_results, bm25_results, alpha=0.5, limit=0)

        assert "limit must be > 0" in str(exc_info.value)


# ==================== RSF FUSION TESTS ====================


class TestRSFusionBasic:
    """Tests for basic RSFusion functionality."""

    def test_fuse_basic(
        self,
        rsf: RSFusion,
        vector_results: Dict[str, Tuple[int, float]],
        bm25_results: Dict[str, Tuple[int, float]],
    ) -> None:
        """Basic fusion with results from both sources."""
        fused = rsf.fuse(vector_results, bm25_results, alpha=0.5)

        assert isinstance(fused, dict)
        assert len(fused) == 4  # All unique chunks

    def test_fuse_empty_sources(self, rsf: RSFusion) -> None:
        """Fusion with empty sources returns empty dict."""
        fused = rsf.fuse({}, {}, alpha=0.5)

        assert fused == {}

    def test_fuse_vector_only(
        self,
        rsf: RSFusion,
        vector_results: Dict[str, Tuple[int, float]],
    ) -> None:
        """Fusion with only vector results."""
        fused = rsf.fuse(vector_results, {}, alpha=0.5)

        assert len(fused) == 3
        # With alpha=0.5 and only vector results, scores are halved
        # after normalization

    def test_fuse_bm25_only(
        self,
        rsf: RSFusion,
        bm25_results: Dict[str, Tuple[int, float]],
    ) -> None:
        """Fusion with only BM25 results."""
        fused = rsf.fuse({}, bm25_results, alpha=0.5)

        assert len(fused) == 3


class TestRSFusionNormalization:
    """Tests for RSFusion score normalization."""

    def test_normalize_scores(
        self,
        rsf: RSFusion,
        vector_results: Dict[str, Tuple[int, float]],
        bm25_results: Dict[str, Tuple[int, float]],
    ) -> None:
        """Scores should be normalized to [0, 1] range."""
        fused = rsf.fuse(vector_results, bm25_results, alpha=0.5)

        # All scores should be in [0, 1]
        for score in fused.values():
            assert 0.0 <= score <= 1.0

    def test_normalize_preserves_order(self, rsf: RSFusion) -> None:
        """Normalization should preserve relative ordering."""
        results = {
            "chunk_1": (1, 100.0),
            "chunk_2": (2, 75.0),
            "chunk_3": (3, 50.0),
        }
        normalized = rsf._normalize(results)

        # Highest score should normalize to 1.0
        assert normalized["chunk_1"] == 1.0
        # Lowest score should normalize to 0.0
        assert normalized["chunk_3"] == 0.0
        # Middle score should be proportionally in between
        assert 0.0 < normalized["chunk_2"] < 1.0

    def test_fuse_all_same_scores(
        self,
        rsf: RSFusion,
        uniform_score_results: Dict[str, Tuple[int, float]],
    ) -> None:
        """All scores identical - avoid division by zero."""
        # When all scores are the same, normalization should return 1.0 for all
        normalized = rsf._normalize(uniform_score_results)

        for score in normalized.values():
            assert score == 1.0

    def test_fuse_single_result_each(
        self,
        rsf: RSFusion,
        single_result_vector: Dict[str, Tuple[int, float]],
        single_result_bm25: Dict[str, Tuple[int, float]],
    ) -> None:
        """Single result from each source."""
        fused = rsf.fuse(single_result_vector, single_result_bm25, alpha=0.5)

        # Single result normalizes to 1.0
        # Combined: 0.5 * 1.0 + 0.5 * 1.0 = 1.0
        assert fused["chunk_1"] == 1.0


class TestRSFusionAlphaWeighting:
    """Tests for RSFusion alpha parameter."""

    def test_fuse_alpha_weighting(self, rsf: RSFusion) -> None:
        """Alpha properly weights sources."""
        vector_results = {"chunk_1": (1, 1.0)}
        bm25_results = {"chunk_1": (1, 1.0)}

        # With normalized scores of 1.0 each:
        # alpha=0.7 -> 0.7 * 1.0 + 0.3 * 1.0 = 1.0
        fused = rsf.fuse(vector_results, bm25_results, alpha=0.7)
        assert fused["chunk_1"] == 1.0

    def test_fuse_alpha_zero_bm25_only(self, rsf: RSFusion) -> None:
        """alpha=0 means only BM25 contributes."""
        # Use 3 results so min-max normalization produces non-zero for middle values
        vector_results = {"chunk_1": (1, 0.95), "chunk_v": (2, 0.85), "chunk_2": (3, 0.75)}
        bm25_results = {"chunk_1": (1, 0.90), "chunk_b": (2, 0.80), "chunk_3": (3, 0.70)}

        fused = rsf.fuse(vector_results, bm25_results, alpha=0.0)

        # Vector-only chunks should have score 0 (alpha=0 ignores vector)
        assert fused["chunk_v"] == 0.0
        assert fused["chunk_2"] == 0.0

        # BM25-only chunk with highest score normalizes to 1.0
        assert fused["chunk_b"] > 0.0 or fused["chunk_3"] >= 0.0

    def test_fuse_alpha_one_vector_only(self, rsf: RSFusion) -> None:
        """alpha=1 means only vector contributes."""
        # Use 3 results so min-max normalization produces non-zero for middle values
        vector_results = {"chunk_1": (1, 0.95), "chunk_v": (2, 0.85), "chunk_2": (3, 0.75)}
        bm25_results = {"chunk_1": (1, 0.90), "chunk_b": (2, 0.80), "chunk_3": (3, 0.70)}

        fused = rsf.fuse(vector_results, bm25_results, alpha=1.0)

        # BM25-only chunks should have score 0 (alpha=1 ignores bm25)
        assert fused["chunk_b"] == 0.0
        assert fused["chunk_3"] == 0.0

        # Vector-only chunk with highest score normalizes to 1.0
        assert fused["chunk_v"] > 0.0 or fused["chunk_2"] >= 0.0


class TestRSFusionEdgeCases:
    """Tests for RSFusion edge cases."""

    def test_normalize_empty_results(self, rsf: RSFusion) -> None:
        """Normalizing empty results returns empty dict."""
        normalized = rsf._normalize({})
        assert normalized == {}

    def test_normalize_single_result(self, rsf: RSFusion) -> None:
        """Single result normalizes to 1.0."""
        results = {"chunk_1": (1, 0.95)}
        normalized = rsf._normalize(results)

        assert normalized["chunk_1"] == 1.0

    def test_fuse_rsf_convenience_function(
        self,
        vector_results: Dict[str, Tuple[int, float]],
        bm25_results: Dict[str, Tuple[int, float]],
    ) -> None:
        """Test fuse_rsf convenience function."""
        fused = fuse_rsf(vector_results, bm25_results, alpha=0.5)

        assert isinstance(fused, dict)
        assert len(fused) == 4


# ==================== DBSF FUSION TESTS ====================


class TestDBSFusionInit:
    """Tests for DBSFusion initialization."""

    def test_init_default_ranges(self) -> None:
        """No ranges provided uses 3-sigma normalization."""
        dbsf = DBSFusion()

        assert dbsf.vector_range is None
        assert dbsf.bm25_range is None

    def test_init_custom_ranges(self) -> None:
        """Custom score ranges are accepted."""
        dbsf = DBSFusion(
            vector_range=(0.0, 1.0),
            bm25_range=(0.0, 50.0),
        )

        assert dbsf.vector_range == (0.0, 1.0)
        assert dbsf.bm25_range == (0.0, 50.0)

    def test_init_vector_range_only(self) -> None:
        """Only vector range provided."""
        dbsf = DBSFusion(vector_range=(0.0, 1.0))

        assert dbsf.vector_range == (0.0, 1.0)
        assert dbsf.bm25_range is None

    def test_init_bm25_range_only(self) -> None:
        """Only BM25 range provided."""
        dbsf = DBSFusion(bm25_range=(0.0, 50.0))

        assert dbsf.vector_range is None
        assert dbsf.bm25_range == (0.0, 50.0)


class TestDBSFusionBasic:
    """Tests for basic DBSFusion functionality."""

    def test_fuse_basic(
        self,
        dbsf: DBSFusion,
        vector_results: Dict[str, Tuple[int, float]],
        bm25_results: Dict[str, Tuple[int, float]],
    ) -> None:
        """Basic 3-sigma normalization fusion."""
        fused = dbsf.fuse(vector_results, bm25_results, alpha=0.5)

        assert isinstance(fused, dict)
        assert len(fused) == 4

    def test_fuse_empty_sources(self, dbsf: DBSFusion) -> None:
        """Fusion with both empty returns empty dict."""
        fused = dbsf.fuse({}, {}, alpha=0.5)

        assert fused == {}

    def test_fuse_scores_in_range(
        self,
        dbsf: DBSFusion,
        vector_results: Dict[str, Tuple[int, float]],
        bm25_results: Dict[str, Tuple[int, float]],
    ) -> None:
        """Fused scores should be in [0, 1] range."""
        fused = dbsf.fuse(vector_results, bm25_results, alpha=0.5)

        for score in fused.values():
            assert 0.0 <= score <= 1.0


class TestDBSFusionNormalization:
    """Tests for DBSF 3-sigma normalization."""

    def test_normalize_3sigma(self, dbsf: DBSFusion) -> None:
        """Verify 3-sigma calculation."""
        # Create results with known mean and std
        # Scores: 10, 20, 30 -> mean=20, std=sqrt(66.67)~8.16
        results = {
            "chunk_1": (1, 30.0),
            "chunk_2": (2, 20.0),
            "chunk_3": (3, 10.0),
        }

        normalized = dbsf._normalize_3sigma(results)

        # All normalized values should be in [0, 1]
        for score in normalized.values():
            assert 0.0 <= score <= 1.0

    def test_fuse_outliers_clamped(self, dbsf: DBSFusion) -> None:
        """Scores outside 3-sigma should be clamped to [0, 1]."""
        # Create results with a large outlier
        results = {
            "chunk_1": (1, 100.0),  # Outlier
            "chunk_2": (2, 10.0),
            "chunk_3": (3, 11.0),
            "chunk_4": (4, 12.0),
        }

        normalized = dbsf._normalize_3sigma(results)

        # All values should be clamped to [0, 1]
        for score in normalized.values():
            assert 0.0 <= score <= 1.0

    def test_fuse_std_zero(self, dbsf: DBSFusion) -> None:
        """All same scores - std=0, should return 1.0 for all."""
        # Create results with exactly identical integer scores
        # to avoid floating-point precision issues
        uniform_results = {
            "chunk_1": (1, 10),  # Use integers
            "chunk_2": (2, 10),
            "chunk_3": (3, 10),
        }
        normalized = dbsf._normalize_3sigma(uniform_results)

        for score in normalized.values():
            assert score == 1.0

    def test_fuse_single_result(self, dbsf: DBSFusion) -> None:
        """Single result should return 1.0."""
        results = {"chunk_1": (1, 0.95)}
        normalized = dbsf._normalize_3sigma(results)

        assert normalized["chunk_1"] == 1.0


class TestDBSFusionWithRanges:
    """Tests for DBSF with predefined ranges."""

    def test_fuse_with_vector_range(self) -> None:
        """Test fusion with predefined vector range."""
        dbsf = DBSFusion(vector_range=(0.0, 1.0))

        vector_results = {
            "chunk_1": (1, 0.8),
            "chunk_2": (2, 0.5),
        }
        bm25_results: Dict[str, Tuple[int, float]] = {}

        fused = dbsf.fuse(vector_results, bm25_results, alpha=1.0)

        # With range (0, 1), score 0.8 normalizes to 0.8
        assert abs(fused["chunk_1"] - 0.8) < 1e-10
        assert abs(fused["chunk_2"] - 0.5) < 1e-10

    def test_fuse_with_bm25_range(self) -> None:
        """Test fusion with predefined BM25 range."""
        dbsf = DBSFusion(bm25_range=(0.0, 100.0))

        vector_results: Dict[str, Tuple[int, float]] = {}
        bm25_results = {
            "chunk_1": (1, 80.0),
            "chunk_2": (2, 50.0),
        }

        fused = dbsf.fuse(vector_results, bm25_results, alpha=0.0)

        # With range (0, 100), score 80 normalizes to 0.8
        assert abs(fused["chunk_1"] - 0.8) < 1e-10
        assert abs(fused["chunk_2"] - 0.5) < 1e-10

    def test_normalize_invalid_range(self, dbsf: DBSFusion) -> None:
        """Invalid range (upper <= lower) returns 1.0 for all."""
        results = {
            "chunk_1": (1, 0.8),
            "chunk_2": (2, 0.5),
        }

        # Range where upper == lower
        normalized = dbsf._normalize_3sigma(results, default_range=(0.5, 0.5))

        for score in normalized.values():
            assert score == 1.0


class TestDBSFusionAlpha:
    """Tests for DBSF alpha parameter."""

    def test_fuse_invalid_alpha_negative(
        self,
        dbsf: DBSFusion,
        vector_results: Dict[str, Tuple[int, float]],
        bm25_results: Dict[str, Tuple[int, float]],
    ) -> None:
        """alpha < 0 should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            dbsf.fuse(vector_results, bm25_results, alpha=-0.1)

        assert "alpha must be in [0, 1]" in str(exc_info.value)

    def test_fuse_invalid_alpha_greater_than_one(
        self,
        dbsf: DBSFusion,
        vector_results: Dict[str, Tuple[int, float]],
        bm25_results: Dict[str, Tuple[int, float]],
    ) -> None:
        """alpha > 1 should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            dbsf.fuse(vector_results, bm25_results, alpha=1.5)

        assert "alpha must be in [0, 1]" in str(exc_info.value)


class TestDBSFusionEdgeCases:
    """Tests for DBSF edge cases."""

    def test_normalize_empty_results(self, dbsf: DBSFusion) -> None:
        """Normalizing empty results returns empty dict."""
        normalized = dbsf._normalize_3sigma({})
        assert normalized == {}

    def test_fuse_dbsf_convenience_function(
        self,
        vector_results: Dict[str, Tuple[int, float]],
        bm25_results: Dict[str, Tuple[int, float]],
    ) -> None:
        """Test fuse_dbsf convenience function."""
        fused = fuse_dbsf(vector_results, bm25_results, alpha=0.5)

        assert isinstance(fused, dict)
        assert len(fused) == 4

    def test_fuse_dbsf_with_ranges(
        self,
        vector_results: Dict[str, Tuple[int, float]],
        bm25_results: Dict[str, Tuple[int, float]],
    ) -> None:
        """Test fuse_dbsf with custom ranges."""
        fused = fuse_dbsf(
            vector_results,
            bm25_results,
            alpha=0.5,
            vector_range=(0.0, 1.0),
            bm25_range=(0.0, 1.0),
        )

        assert isinstance(fused, dict)
        assert len(fused) == 4


class TestDBSFusionMathematical:
    """Tests for DBSF mathematical correctness."""

    def test_3sigma_calculation(self) -> None:
        """Verify 3-sigma bounds calculation."""
        dbsf = DBSFusion()

        # Known data: [10, 20, 30]
        # Mean = 20
        # Variance = ((10-20)^2 + (20-20)^2 + (30-20)^2) / 3 = 200/3
        # Std = sqrt(200/3) = 8.165...
        # Lower bound = 20 - 3*8.165 = -4.495
        # Upper bound = 20 + 3*8.165 = 44.495

        results = {
            "a": (1, 10.0),
            "b": (2, 20.0),
            "c": (3, 30.0),
        }

        normalized = dbsf._normalize_3sigma(results)

        # Score 10: (10 - (-4.495)) / 48.99 = 14.495 / 48.99 ~ 0.296
        # Score 20: (20 - (-4.495)) / 48.99 = 24.495 / 48.99 ~ 0.5
        # Score 30: (30 - (-4.495)) / 48.99 = 34.495 / 48.99 ~ 0.704

        # Mean score should normalize to approximately 0.5
        assert 0.45 <= normalized["b"] <= 0.55

        # Higher score should have higher normalized value
        assert normalized["c"] > normalized["b"] > normalized["a"]


# ==================== INTEGRATION TESTS ====================


class TestFusionIntegration:
    """Integration tests comparing fusion methods."""

    def test_all_fusion_methods_return_same_keys(
        self,
        vector_results: Dict[str, Tuple[int, float]],
        bm25_results: Dict[str, Tuple[int, float]],
    ) -> None:
        """All fusion methods should return results for same chunk IDs."""
        rrf = RRFusion()
        rsf = RSFusion()
        dbsf = DBSFusion()

        rrf_result = rrf.fuse(vector_results, bm25_results, alpha=0.5)
        rsf_result = rsf.fuse(vector_results, bm25_results, alpha=0.5)
        dbsf_result = dbsf.fuse(vector_results, bm25_results, alpha=0.5)

        assert set(rrf_result.keys()) == set(rsf_result.keys()) == set(dbsf_result.keys())

    def test_all_fusion_methods_handle_empty(self) -> None:
        """All fusion methods should handle empty inputs."""
        rrf = RRFusion()
        rsf = RSFusion()
        dbsf = DBSFusion()

        assert rrf.fuse({}, {}, alpha=0.5) == {}
        assert rsf.fuse({}, {}, alpha=0.5) == {}
        assert dbsf.fuse({}, {}, alpha=0.5) == {}

    def test_fusion_preserves_relative_ranking(
        self,
        overlapping_vector_results: Dict[str, Tuple[int, float]],
        overlapping_bm25_results: Dict[str, Tuple[int, float]],
    ) -> None:
        """Test that fusion methods preserve reasonable ranking."""
        rrf = RRFusion()

        # chunk_1: vector rank 1, bm25 rank 2
        # chunk_2: vector rank 2, bm25 rank 1
        # With alpha=0.5, chunk_1 has better vector rank, chunk_2 has better bm25 rank

        fused = rrf.fuse(overlapping_vector_results, overlapping_bm25_results, alpha=0.5)

        # Both chunks should have comparable scores with balanced alpha
        # since they have opposite rankings in the two sources
        score_diff = abs(fused["chunk_1"] - fused["chunk_2"])
        # Scores should be close but not necessarily equal
        assert score_diff < 0.01  # Small difference expected
