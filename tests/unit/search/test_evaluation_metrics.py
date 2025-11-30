"""
Unit tests for search evaluation metrics module.

Tests cover:
- Recall@K: Measures how many relevant items were retrieved
- Precision@K: Measures fraction of retrieved items that are relevant
- MRR: Mean Reciprocal Rank - how quickly first relevant result appears
- NDCG@K: Normalized Discounted Cumulative Gain with graded relevance
- Average Precision: Used for computing MAP over queries
- evaluate_search: Aggregates all metrics in one call

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from typing import Dict, List, Set

import pytest

from zapomni_core.search.evaluation.metrics import (
    average_precision,
    evaluate_search,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


# ==================== FIXTURES ====================


@pytest.fixture
def retrieved_docs() -> List[str]:
    """Standard retrieved documents for testing."""
    return ["doc1", "doc2", "doc3", "doc4", "doc5"]


@pytest.fixture
def relevant_docs() -> Set[str]:
    """Standard relevant documents for testing."""
    return {"doc1", "doc3", "doc6"}


@pytest.fixture
def all_relevant_in_top() -> tuple[List[str], Set[str]]:
    """All relevant documents are in top positions."""
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = {"doc1", "doc2", "doc3"}
    return retrieved, relevant


@pytest.fixture
def graded_relevance() -> Dict[str, float]:
    """Graded relevance scores for NDCG testing."""
    return {
        "doc1": 1.0,
        "doc2": 0.8,
        "doc3": 0.5,
        "doc4": 0.3,
        "doc5": 0.0,
    }


@pytest.fixture
def binary_relevance() -> Dict[str, float]:
    """Binary relevance scores (0 or 1)."""
    return {
        "doc1": 1.0,
        "doc2": 0.0,
        "doc3": 1.0,
        "doc4": 0.0,
        "doc5": 1.0,
    }


# ==================== RECALL@K TESTS ====================


class TestRecallAtK:
    """Tests for recall_at_k function."""

    def test_perfect_recall(self) -> None:
        """All relevant items in top K should give recall 1.0."""
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc1", "doc2"}

        result = recall_at_k(retrieved, relevant, k=5)

        assert result == pytest.approx(1.0)

    def test_partial_recall(self, retrieved_docs: List[str], relevant_docs: Set[str]) -> None:
        """Some relevant items in top K should give partial recall."""
        # relevant = {"doc1", "doc3", "doc6"}
        # In retrieved[:5] we have doc1, doc3 but not doc6
        result = recall_at_k(retrieved_docs, relevant_docs, k=5)

        # 2 out of 3 relevant found
        assert result == pytest.approx(2 / 3)

    def test_zero_recall(self) -> None:
        """No relevant items in top K should give recall 0."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc10", "doc11", "doc12"}

        result = recall_at_k(retrieved, relevant, k=3)

        assert result == pytest.approx(0.0)

    def test_empty_relevant(self, retrieved_docs: List[str]) -> None:
        """Empty ground truth should return 0."""
        result = recall_at_k(retrieved_docs, set(), k=5)

        assert result == pytest.approx(0.0)

    def test_empty_retrieved(self, relevant_docs: Set[str]) -> None:
        """Empty retrieved list should return 0."""
        result = recall_at_k([], relevant_docs, k=5)

        assert result == pytest.approx(0.0)

    def test_k_larger_than_retrieved(self) -> None:
        """K larger than retrieved list should use all retrieved."""
        retrieved = ["doc1", "doc2"]
        relevant = {"doc1", "doc2", "doc3"}

        result = recall_at_k(retrieved, relevant, k=10)

        # 2 out of 3 relevant found
        assert result == pytest.approx(2 / 3)

    def test_k_zero(self, retrieved_docs: List[str], relevant_docs: Set[str]) -> None:
        """K=0 should return 0."""
        result = recall_at_k(retrieved_docs, relevant_docs, k=0)

        assert result == pytest.approx(0.0)

    def test_k_negative(self, retrieved_docs: List[str], relevant_docs: Set[str]) -> None:
        """Negative K should return 0."""
        result = recall_at_k(retrieved_docs, relevant_docs, k=-5)

        assert result == pytest.approx(0.0)

    def test_different_k_values(self) -> None:
        """Different K values should give different recall."""
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc1", "doc3", "doc5"}

        recall_k1 = recall_at_k(retrieved, relevant, k=1)
        recall_k3 = recall_at_k(retrieved, relevant, k=3)
        recall_k5 = recall_at_k(retrieved, relevant, k=5)

        assert recall_k1 == pytest.approx(1 / 3)  # doc1 found
        assert recall_k3 == pytest.approx(2 / 3)  # doc1, doc3 found
        assert recall_k5 == pytest.approx(1.0)  # all found


# ==================== PRECISION@K TESTS ====================


class TestPrecisionAtK:
    """Tests for precision_at_k function."""

    def test_perfect_precision(self) -> None:
        """All top K are relevant should give precision 1.0."""
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc1", "doc2", "doc3", "doc4", "doc5"}

        result = precision_at_k(retrieved, relevant, k=3)

        assert result == pytest.approx(1.0)

    def test_partial_precision(self, retrieved_docs: List[str], relevant_docs: Set[str]) -> None:
        """Some top K are relevant should give partial precision."""
        # relevant = {"doc1", "doc3", "doc6"}
        # In retrieved[:5] we have doc1, doc3 which are relevant (2 out of 5)
        result = precision_at_k(retrieved_docs, relevant_docs, k=5)

        assert result == pytest.approx(2 / 5)

    def test_zero_precision(self) -> None:
        """No top K are relevant should give precision 0."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc10", "doc11", "doc12"}

        result = precision_at_k(retrieved, relevant, k=3)

        assert result == pytest.approx(0.0)

    def test_k_zero(self, retrieved_docs: List[str], relevant_docs: Set[str]) -> None:
        """K=0 should return 0."""
        result = precision_at_k(retrieved_docs, relevant_docs, k=0)

        assert result == pytest.approx(0.0)

    def test_k_negative(self, retrieved_docs: List[str], relevant_docs: Set[str]) -> None:
        """Negative K should return 0."""
        result = precision_at_k(retrieved_docs, relevant_docs, k=-3)

        assert result == pytest.approx(0.0)

    def test_empty_retrieved(self, relevant_docs: Set[str]) -> None:
        """Empty retrieved list should return 0."""
        result = precision_at_k([], relevant_docs, k=5)

        assert result == pytest.approx(0.0)

    def test_empty_relevant(self, retrieved_docs: List[str]) -> None:
        """Empty relevant set should return 0."""
        result = precision_at_k(retrieved_docs, set(), k=5)

        assert result == pytest.approx(0.0)

    def test_k_larger_than_retrieved(self) -> None:
        """K larger than retrieved uses K for denominator."""
        retrieved = ["doc1", "doc2"]  # only 2 docs
        relevant = {"doc1", "doc2"}

        result = precision_at_k(retrieved, relevant, k=10)

        # 2 relevant found out of K=10
        assert result == pytest.approx(2 / 10)

    def test_precision_decreases_with_k(self) -> None:
        """Precision typically decreases as K increases with non-relevant docs."""
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc1"}  # only first is relevant

        p_at_1 = precision_at_k(retrieved, relevant, k=1)
        p_at_3 = precision_at_k(retrieved, relevant, k=3)
        p_at_5 = precision_at_k(retrieved, relevant, k=5)

        assert p_at_1 == pytest.approx(1.0)
        assert p_at_3 == pytest.approx(1 / 3)
        assert p_at_5 == pytest.approx(1 / 5)


# ==================== MRR TESTS ====================


class TestMRR:
    """Tests for Mean Reciprocal Rank (MRR) function."""

    def test_first_is_relevant(self) -> None:
        """Relevant item at rank 1 should give MRR 1.0."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1"}

        result = mrr(retrieved, relevant)

        assert result == pytest.approx(1.0)

    def test_second_is_relevant(self) -> None:
        """Relevant item at rank 2 should give MRR 0.5."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc2"}

        result = mrr(retrieved, relevant)

        assert result == pytest.approx(0.5)

    def test_third_is_relevant(self) -> None:
        """Relevant item at rank 3 should give MRR ~0.333."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc3"}

        result = mrr(retrieved, relevant)

        assert result == pytest.approx(1 / 3)

    def test_no_relevant(self) -> None:
        """No relevant items found should give MRR 0."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc10", "doc11"}

        result = mrr(retrieved, relevant)

        assert result == pytest.approx(0.0)

    def test_multiple_relevant(self) -> None:
        """Only first relevant item matters for MRR."""
        retrieved = ["doc1", "doc2", "doc3", "doc4"]
        relevant = {"doc2", "doc4"}  # doc2 is first at rank 2

        result = mrr(retrieved, relevant)

        # First relevant is doc2 at position 2
        assert result == pytest.approx(0.5)

    def test_empty_retrieved(self) -> None:
        """Empty retrieved list should return 0."""
        relevant = {"doc1", "doc2"}

        result = mrr([], relevant)

        assert result == pytest.approx(0.0)

    def test_empty_relevant(self, retrieved_docs: List[str]) -> None:
        """Empty relevant set should return 0."""
        result = mrr(retrieved_docs, set())

        assert result == pytest.approx(0.0)

    def test_both_empty(self) -> None:
        """Both empty should return 0."""
        result = mrr([], set())

        assert result == pytest.approx(0.0)

    def test_late_relevant(self) -> None:
        """Relevant item at later positions gives low MRR."""
        retrieved = [
            "doc1",
            "doc2",
            "doc3",
            "doc4",
            "doc5",
            "doc6",
            "doc7",
            "doc8",
            "doc9",
            "doc10",
        ]
        relevant = {"doc10"}

        result = mrr(retrieved, relevant)

        assert result == pytest.approx(0.1)


# ==================== NDCG@K TESTS ====================


class TestNDCGAtK:
    """Tests for Normalized Discounted Cumulative Gain at K."""

    def test_perfect_ranking(self) -> None:
        """Ideal ranking should give NDCG 1.0."""
        # Relevance scores in descending order - ideal ranking
        retrieved = ["doc1", "doc2", "doc3"]
        relevance = {"doc1": 1.0, "doc2": 0.5, "doc3": 0.0}

        result = ndcg_at_k(retrieved, relevance, k=3)

        assert result == pytest.approx(1.0)

    def test_reversed_ranking(self) -> None:
        """Worst ranking should give NDCG < 1.0."""
        # Relevance scores in ascending order - worst possible
        retrieved = ["doc3", "doc2", "doc1"]
        relevance = {"doc1": 1.0, "doc2": 0.5, "doc3": 0.0}

        result = ndcg_at_k(retrieved, relevance, k=3)

        # Should be less than 1.0 since ranking is reversed
        assert result < 1.0
        assert result > 0.0

    def test_partial_relevant(self) -> None:
        """Mixed relevance scores should give partial NDCG."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevance = {"doc1": 0.5, "doc2": 1.0, "doc3": 0.0}

        result = ndcg_at_k(retrieved, relevance, k=3)

        # Not ideal order (doc2 should be first), so < 1.0
        assert result < 1.0
        assert result > 0.0

    def test_graded_relevance(self, graded_relevance: Dict[str, float]) -> None:
        """Different relevance values should be handled correctly."""
        # Ideal order: doc1(1.0), doc2(0.8), doc3(0.5), doc4(0.3), doc5(0.0)
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]

        result = ndcg_at_k(retrieved, graded_relevance, k=5)

        assert result == pytest.approx(1.0)

    def test_graded_relevance_suboptimal(self, graded_relevance: Dict[str, float]) -> None:
        """Suboptimal ranking with graded relevance."""
        # Not ideal order
        retrieved = ["doc5", "doc4", "doc3", "doc2", "doc1"]

        result = ndcg_at_k(retrieved, graded_relevance, k=5)

        assert result < 1.0
        assert result >= 0.0

    def test_empty_relevance(self, retrieved_docs: List[str]) -> None:
        """No relevance scores should return 0."""
        result = ndcg_at_k(retrieved_docs, {}, k=5)

        assert result == pytest.approx(0.0)

    def test_binary_relevance(self, binary_relevance: Dict[str, float]) -> None:
        """Binary (0 and 1) relevance scores."""
        # Ideal order: doc1, doc3, doc5 (all with 1.0)
        retrieved = ["doc1", "doc3", "doc5", "doc2", "doc4"]

        result = ndcg_at_k(retrieved, binary_relevance, k=5)

        assert result == pytest.approx(1.0)

    def test_k_zero(self, graded_relevance: Dict[str, float]) -> None:
        """K=0 should return 0."""
        retrieved = ["doc1", "doc2", "doc3"]

        result = ndcg_at_k(retrieved, graded_relevance, k=0)

        assert result == pytest.approx(0.0)

    def test_k_negative(self, graded_relevance: Dict[str, float]) -> None:
        """Negative K should return 0."""
        retrieved = ["doc1", "doc2", "doc3"]

        result = ndcg_at_k(retrieved, graded_relevance, k=-5)

        assert result == pytest.approx(0.0)

    def test_k_larger_than_retrieved(self) -> None:
        """K larger than retrieved should use all available."""
        retrieved = ["doc1", "doc2"]
        relevance = {"doc1": 1.0, "doc2": 0.5}

        result = ndcg_at_k(retrieved, relevance, k=10)

        # Should still work, using only available docs
        assert result == pytest.approx(1.0)

    def test_all_zero_relevance(self) -> None:
        """All zero relevance scores should return 0."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevance = {"doc1": 0.0, "doc2": 0.0, "doc3": 0.0}

        result = ndcg_at_k(retrieved, relevance, k=3)

        assert result == pytest.approx(0.0)

    def test_single_item(self) -> None:
        """Single item ranking."""
        retrieved = ["doc1"]
        relevance = {"doc1": 1.0}

        result = ndcg_at_k(retrieved, relevance, k=1)

        assert result == pytest.approx(1.0)

    def test_retrieved_not_in_relevance(self) -> None:
        """Retrieved items not in relevance dict treated as 0."""
        retrieved = ["doc10", "doc11", "doc1"]
        relevance = {"doc1": 1.0}  # Only doc1 has relevance

        result = ndcg_at_k(retrieved, relevance, k=3)

        # doc1 is at position 3, not ideal (should be position 1)
        assert result < 1.0
        assert result > 0.0


# ==================== AVERAGE PRECISION TESTS ====================


class TestAveragePrecision:
    """Tests for Average Precision function."""

    def test_all_relevant_at_top(self) -> None:
        """All relevant items at top positions - best case."""
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc1", "doc2", "doc3"}

        result = average_precision(retrieved, relevant)

        # Precision at each relevant position:
        # doc1: 1/1 = 1.0
        # doc2: 2/2 = 1.0
        # doc3: 3/3 = 1.0
        # AP = (1.0 + 1.0 + 1.0) / 3 = 1.0
        assert result == pytest.approx(1.0)

    def test_relevant_scattered(self) -> None:
        """Relevant items at various positions."""
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc1", "doc3", "doc5"}

        result = average_precision(retrieved, relevant)

        # Precision at each relevant position:
        # doc1 (pos 1): 1/1 = 1.0
        # doc3 (pos 3): 2/3 = 0.667
        # doc5 (pos 5): 3/5 = 0.6
        # AP = (1.0 + 0.667 + 0.6) / 3 = 0.756
        expected = (1.0 + 2 / 3 + 3 / 5) / 3
        assert result == pytest.approx(expected)

    def test_no_relevant(self) -> None:
        """No relevant items should give AP 0."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc10", "doc11"}

        result = average_precision(retrieved, relevant)

        assert result == pytest.approx(0.0)

    def test_empty_relevant(self, retrieved_docs: List[str]) -> None:
        """Empty relevant set should return 0."""
        result = average_precision(retrieved_docs, set())

        assert result == pytest.approx(0.0)

    def test_empty_retrieved(self, relevant_docs: Set[str]) -> None:
        """Empty retrieved list should return 0."""
        result = average_precision([], relevant_docs)

        assert result == pytest.approx(0.0)

    def test_all_relevant_at_bottom(self) -> None:
        """All relevant items at bottom positions - worst case."""
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc4", "doc5"}

        result = average_precision(retrieved, relevant)

        # Precision at each relevant position:
        # doc4 (pos 4): 1/4 = 0.25
        # doc5 (pos 5): 2/5 = 0.4
        # AP = (0.25 + 0.4) / 2 = 0.325
        expected = (1 / 4 + 2 / 5) / 2
        assert result == pytest.approx(expected)

    def test_single_relevant_first(self) -> None:
        """Single relevant item at first position."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1"}

        result = average_precision(retrieved, relevant)

        assert result == pytest.approx(1.0)

    def test_single_relevant_last(self) -> None:
        """Single relevant item at last position."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc3"}

        result = average_precision(retrieved, relevant)

        # Precision at doc3: 1/3
        assert result == pytest.approx(1 / 3)

    def test_not_all_relevant_found(self) -> None:
        """Some relevant items not in retrieved list."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1", "doc2", "doc10"}  # doc10 not in retrieved

        result = average_precision(retrieved, relevant)

        # Precision at doc1: 1/1 = 1.0
        # Precision at doc2: 2/2 = 1.0
        # AP = (1.0 + 1.0) / 3 = 0.667 (divide by total relevant, not found)
        expected = (1.0 + 1.0) / 3
        assert result == pytest.approx(expected)


# ==================== EVALUATE_SEARCH TESTS ====================


class TestEvaluateSearch:
    """Tests for evaluate_search aggregator function."""

    def test_returns_all_metrics(self, retrieved_docs: List[str], relevant_docs: Set[str]) -> None:
        """Returns dict with all expected metric keys."""
        result = evaluate_search(retrieved_docs, relevant_docs, k=5)

        assert "recall@5" in result
        assert "precision@5" in result
        assert "mrr" in result
        assert "ndcg@5" in result
        assert "ap" in result
        assert len(result) == 5

    def test_with_relevance_scores(
        self, retrieved_docs: List[str], relevant_docs: Set[str], graded_relevance: Dict[str, float]
    ) -> None:
        """Uses graded relevance for NDCG when provided."""
        result = evaluate_search(
            retrieved_docs, relevant_docs, k=5, relevance_scores=graded_relevance
        )

        # Should have all metrics
        assert "ndcg@5" in result
        # NDCG should be computed with graded relevance
        assert result["ndcg@5"] >= 0.0
        assert result["ndcg@5"] <= 1.0

    def test_without_relevance_scores(
        self, retrieved_docs: List[str], relevant_docs: Set[str]
    ) -> None:
        """Uses binary relevance for NDCG when not provided."""
        result = evaluate_search(retrieved_docs, relevant_docs, k=5)

        # Should use binary relevance (1.0 for relevant items)
        assert "ndcg@5" in result
        assert result["ndcg@5"] >= 0.0
        assert result["ndcg@5"] <= 1.0

    def test_different_k_values(self, retrieved_docs: List[str], relevant_docs: Set[str]) -> None:
        """Different K values should change metric keys."""
        result_k3 = evaluate_search(retrieved_docs, relevant_docs, k=3)
        result_k10 = evaluate_search(retrieved_docs, relevant_docs, k=10)

        assert "recall@3" in result_k3
        assert "precision@3" in result_k3
        assert "ndcg@3" in result_k3

        assert "recall@10" in result_k10
        assert "precision@10" in result_k10
        assert "ndcg@10" in result_k10

    def test_metrics_consistency(self) -> None:
        """Individual metric functions should match evaluate_search results."""
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc1", "doc3", "doc6"}
        k = 5

        result = evaluate_search(retrieved, relevant, k=k)

        # Compare with individual functions
        assert result[f"recall@{k}"] == pytest.approx(recall_at_k(retrieved, relevant, k))
        assert result[f"precision@{k}"] == pytest.approx(precision_at_k(retrieved, relevant, k))
        assert result["mrr"] == pytest.approx(mrr(retrieved, relevant))
        assert result["ap"] == pytest.approx(average_precision(retrieved, relevant))

    def test_empty_inputs(self) -> None:
        """Empty inputs should return all zeros."""
        result = evaluate_search([], set(), k=5)

        assert result["recall@5"] == pytest.approx(0.0)
        assert result["precision@5"] == pytest.approx(0.0)
        assert result["mrr"] == pytest.approx(0.0)
        assert result["ndcg@5"] == pytest.approx(0.0)
        assert result["ap"] == pytest.approx(0.0)

    def test_perfect_search(self) -> None:
        """Perfect search results should give high scores."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1", "doc2", "doc3"}

        result = evaluate_search(retrieved, relevant, k=3)

        assert result["recall@3"] == pytest.approx(1.0)
        assert result["precision@3"] == pytest.approx(1.0)
        assert result["mrr"] == pytest.approx(1.0)
        assert result["ndcg@3"] == pytest.approx(1.0)
        assert result["ap"] == pytest.approx(1.0)


# ==================== EDGE CASE TESTS ====================


class TestEdgeCases:
    """Additional edge case tests across all metrics."""

    def test_duplicate_ids_in_retrieved(self) -> None:
        """Duplicate IDs in retrieved list are counted once per position."""
        retrieved = ["doc1", "doc1", "doc2", "doc1", "doc3"]
        relevant = {"doc1", "doc2"}

        # Each position is evaluated independently
        recall = recall_at_k(retrieved, relevant, k=5)
        precision = precision_at_k(retrieved, relevant, k=5)

        # Set operations deduplicate for recall
        assert recall == pytest.approx(1.0)
        assert precision == pytest.approx(2 / 5)

    def test_large_k_value(self) -> None:
        """Very large K value should not cause errors."""
        retrieved = ["doc1", "doc2"]
        relevant = {"doc1", "doc2"}

        recall = recall_at_k(retrieved, relevant, k=1000000)
        precision = precision_at_k(retrieved, relevant, k=1000000)

        assert recall == pytest.approx(1.0)
        assert precision == pytest.approx(2 / 1000000)

    def test_unicode_document_ids(self) -> None:
        """Unicode document IDs should work correctly."""
        retrieved = ["doc_\u0442\u0435\u0441\u0442", "\u6587\u6863_2", "emoji_\U0001f600"]
        relevant = {"doc_\u0442\u0435\u0441\u0442", "\u6587\u6863_2"}

        recall = recall_at_k(retrieved, relevant, k=3)
        precision = precision_at_k(retrieved, relevant, k=3)
        mrr_score = mrr(retrieved, relevant)

        assert recall == pytest.approx(1.0)
        assert precision == pytest.approx(2 / 3)
        assert mrr_score == pytest.approx(1.0)

    def test_special_characters_in_ids(self) -> None:
        """Special characters in document IDs."""
        retrieved = ["doc-1", "doc.2", "doc_3", "doc/4", "doc@5"]
        relevant = {"doc-1", "doc.2"}

        recall = recall_at_k(retrieved, relevant, k=5)

        assert recall == pytest.approx(1.0)

    def test_numeric_string_ids(self) -> None:
        """Numeric string IDs."""
        retrieved = ["1", "2", "3", "4", "5"]
        relevant = {"1", "3", "5"}

        recall = recall_at_k(retrieved, relevant, k=5)

        assert recall == pytest.approx(1.0)

    def test_all_metrics_with_single_doc(self) -> None:
        """All metrics with single document."""
        retrieved = ["doc1"]
        relevant = {"doc1"}
        relevance = {"doc1": 1.0}

        assert recall_at_k(retrieved, relevant, k=1) == pytest.approx(1.0)
        assert precision_at_k(retrieved, relevant, k=1) == pytest.approx(1.0)
        assert mrr(retrieved, relevant) == pytest.approx(1.0)
        assert ndcg_at_k(retrieved, relevance, k=1) == pytest.approx(1.0)
        assert average_precision(retrieved, relevant) == pytest.approx(1.0)

    def test_very_small_relevance_values(self) -> None:
        """Very small non-zero relevance values."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevance = {"doc1": 0.001, "doc2": 0.0001, "doc3": 0.00001}

        result = ndcg_at_k(retrieved, relevance, k=3)

        # Ideal order is already achieved
        assert result == pytest.approx(1.0)

    def test_relevance_values_greater_than_one(self) -> None:
        """Relevance values > 1.0 should still work (not typical but valid)."""
        retrieved = ["doc1", "doc2"]
        relevance = {"doc1": 2.0, "doc2": 1.5}

        result = ndcg_at_k(retrieved, relevance, k=2)

        # Ideal order is already achieved
        assert result == pytest.approx(1.0)
