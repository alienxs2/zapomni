"""
Search evaluation metrics for measuring retrieval quality.

Provides standard information retrieval metrics for evaluating
search result quality:
    - Recall@K: Measures how many relevant items were retrieved
    - Precision@K: Measures fraction of retrieved items that are relevant
    - MRR: Mean Reciprocal Rank - how quickly first relevant result appears
    - NDCG@K: Normalized Discounted Cumulative Gain with graded relevance
    - Average Precision: Used for computing MAP over queries

These metrics are used to evaluate and compare different search
strategies (vector, BM25, hybrid) and fusion methods (RRF, RSF, DBSF).

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

import math
from typing import Dict, List, Optional, Set


def recall_at_k(
    retrieved: List[str],
    relevant: Set[str],
    k: int = 10,
) -> float:
    """
    Calculate Recall@K metric.

    Recall@K = |relevant intersection retrieved[:k]| / |relevant|

    Measures how many of the relevant items were retrieved in top K results.
    This metric answers: "Of all the relevant items, how many did we find?"

    Args:
        retrieved: List of chunk_ids in ranked order (most relevant first)
        relevant: Set of relevant chunk_ids (ground truth)
        k: Number of top results to consider (default: 10)

    Returns:
        Recall score between 0.0 and 1.0.
        Returns 0.0 if relevant set is empty.

    Examples:
        >>> retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        >>> relevant = {"doc1", "doc3", "doc6"}
        >>> recall_at_k(retrieved, relevant, k=3)
        0.6666666666666666
        >>> recall_at_k(retrieved, relevant, k=5)
        0.6666666666666666
        >>> recall_at_k([], relevant, k=5)
        0.0
        >>> recall_at_k(retrieved, set(), k=5)
        0.0
    """
    if not relevant:
        return 0.0

    if k <= 0:
        return 0.0

    top_k = set(retrieved[:k])
    relevant_found = len(relevant & top_k)

    return relevant_found / len(relevant)


def precision_at_k(
    retrieved: List[str],
    relevant: Set[str],
    k: int = 10,
) -> float:
    """
    Calculate Precision@K metric.

    Precision@K = |relevant intersection retrieved[:k]| / K

    Measures what fraction of the top K results are relevant.
    This metric answers: "Of the items we retrieved, how many are relevant?"

    Args:
        retrieved: List of chunk_ids in ranked order (most relevant first)
        relevant: Set of relevant chunk_ids (ground truth)
        k: Number of top results to consider (default: 10)

    Returns:
        Precision score between 0.0 and 1.0.
        Returns 0.0 if k <= 0.

    Examples:
        >>> retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        >>> relevant = {"doc1", "doc3", "doc6"}
        >>> precision_at_k(retrieved, relevant, k=3)
        0.6666666666666666
        >>> precision_at_k(retrieved, relevant, k=5)
        0.4
        >>> precision_at_k([], relevant, k=5)
        0.0
        >>> precision_at_k(retrieved, set(), k=5)
        0.0
    """
    if k <= 0:
        return 0.0

    top_k = set(retrieved[:k])
    relevant_found = len(relevant & top_k)

    return relevant_found / k


def mrr(
    retrieved: List[str],
    relevant: Set[str],
) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR) metric.

    MRR = 1 / rank_of_first_relevant_item

    Measures how quickly we find the first relevant result.
    Higher MRR means relevant items appear earlier in the ranking.

    Args:
        retrieved: List of chunk_ids in ranked order (most relevant first)
        relevant: Set of relevant chunk_ids (ground truth)

    Returns:
        Reciprocal rank between 0.0 and 1.0.
        Returns 0.0 if no relevant items are found in retrieved list.

    Examples:
        >>> retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        >>> relevant = {"doc2", "doc4"}
        >>> mrr(retrieved, relevant)  # First relevant at position 2
        0.5
        >>> relevant = {"doc1"}
        >>> mrr(retrieved, relevant)  # First relevant at position 1
        1.0
        >>> relevant = {"doc6"}
        >>> mrr(retrieved, relevant)  # No relevant found
        0.0
        >>> mrr([], relevant)
        0.0
    """
    if not relevant or not retrieved:
        return 0.0

    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            # Rank is 1-indexed
            return 1.0 / (i + 1)

    return 0.0


def _dcg_at_k(
    retrieved: List[str],
    relevance: Dict[str, float],
    k: int,
) -> float:
    """
    Calculate Discounted Cumulative Gain at K.

    DCG@K = sum_{i=0}^{k-1} (2^rel_i - 1) / log2(i + 2)

    Internal helper function for NDCG calculation.

    Args:
        retrieved: List of chunk_ids in ranked order
        relevance: Dict mapping chunk_id to relevance score (0.0 to 1.0)
        k: Number of top results to consider

    Returns:
        DCG score (unbounded, depends on relevance values)
    """
    dcg = 0.0
    for i in range(min(k, len(retrieved))):
        doc_id = retrieved[i]
        rel = relevance.get(doc_id, 0.0)
        # Gain: 2^rel - 1, Discount: log2(rank + 1) where rank is 1-indexed
        # Position i is 0-indexed, so rank = i + 1, log2(i + 2)
        dcg += (2**rel - 1) / math.log2(i + 2)
    return dcg


def ndcg_at_k(
    retrieved: List[str],
    relevance: Dict[str, float],
    k: int = 10,
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at K (NDCG@K).

    NDCG@K = DCG@K / IDCG@K

    Where:
        DCG@K = sum_{i=0}^{k-1} (2^rel_i - 1) / log2(i + 2)
        IDCG@K = DCG@K for ideal ranking (sorted by relevance descending)

    Measures ranking quality with graded relevance scores.
    Takes into account both the relevance of items and their positions.

    Args:
        retrieved: List of chunk_ids in ranked order (most relevant first)
        relevance: Dict mapping chunk_id to relevance score (0.0 to 1.0)
        k: Number of top results to consider (default: 10)

    Returns:
        NDCG score between 0.0 and 1.0.
        Returns 0.0 if IDCG is 0 (no relevant items) or k <= 0.

    Examples:
        >>> retrieved = ["doc1", "doc2", "doc3"]
        >>> relevance = {"doc1": 0.5, "doc2": 1.0, "doc3": 0.0}
        >>> # Ideal order would be ["doc2", "doc1", "doc3"]
        >>> round(ndcg_at_k(retrieved, relevance, k=3), 4)
        0.8286
        >>> ndcg_at_k(["doc2", "doc1", "doc3"], relevance, k=3)  # Ideal ordering
        1.0
        >>> ndcg_at_k(retrieved, {}, k=3)  # No relevance info
        0.0
    """
    if k <= 0 or not relevance:
        return 0.0

    # Calculate actual DCG
    dcg = _dcg_at_k(retrieved, relevance, k)

    # Calculate ideal DCG (sorted by relevance descending)
    # Only consider items that have relevance scores
    ideal_ranking = sorted(relevance.keys(), key=lambda x: relevance[x], reverse=True)
    idcg = _dcg_at_k(ideal_ranking, relevance, k)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def average_precision(
    retrieved: List[str],
    relevant: Set[str],
) -> float:
    """
    Calculate Average Precision (AP) metric.

    AP = (1/|relevant|) * sum_{k=1}^{n} (Precision@k * rel_k)

    Where rel_k = 1 if item at position k is relevant, else 0.

    Used to compute Mean Average Precision (MAP) over multiple queries.
    Combines precision at each relevant item position.

    Args:
        retrieved: List of chunk_ids in ranked order (most relevant first)
        relevant: Set of relevant chunk_ids (ground truth)

    Returns:
        Average Precision score between 0.0 and 1.0.
        Returns 0.0 if relevant set is empty or no relevant items found.

    Examples:
        >>> retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        >>> relevant = {"doc1", "doc3", "doc5"}
        >>> # Precision at position 1 (doc1): 1/1 = 1.0
        >>> # Precision at position 3 (doc3): 2/3 = 0.667
        >>> # Precision at position 5 (doc5): 3/5 = 0.6
        >>> # AP = (1.0 + 0.667 + 0.6) / 3 = 0.756
        >>> round(average_precision(retrieved, relevant), 3)
        0.756
        >>> average_precision([], relevant)
        0.0
        >>> average_precision(retrieved, set())
        0.0
    """
    if not relevant or not retrieved:
        return 0.0

    precision_sum = 0.0
    relevant_count = 0

    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            relevant_count += 1
            # Precision at this position (1-indexed)
            precision_at_i = relevant_count / (i + 1)
            precision_sum += precision_at_i

    if relevant_count == 0:
        return 0.0

    return precision_sum / len(relevant)


def evaluate_search(
    retrieved: List[str],
    relevant: Set[str],
    k: int = 10,
    relevance_scores: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics for a single query.

    Aggregates all standard IR metrics in one convenient function call.
    Use this for quick evaluation of search results quality.

    Args:
        retrieved: List of chunk_ids in ranked order (most relevant first)
        relevant: Set of relevant chunk_ids (ground truth)
        k: Number of top results to consider for @K metrics (default: 10)
        relevance_scores: Optional dict mapping chunk_id to relevance score
            (0.0 to 1.0). If not provided, binary relevance (1.0 for all
            relevant items) is used for NDCG calculation.

    Returns:
        Dict with the following keys:
            - recall@{k}: Recall at K
            - precision@{k}: Precision at K
            - mrr: Mean Reciprocal Rank
            - ndcg@{k}: Normalized Discounted Cumulative Gain at K
            - ap: Average Precision

    Examples:
        >>> retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        >>> relevant = {"doc1", "doc3", "doc6"}
        >>> metrics = evaluate_search(retrieved, relevant, k=5)
        >>> sorted(metrics.keys())
        ['ap', 'mrr', 'ndcg@5', 'precision@5', 'recall@5']
        >>> metrics["recall@5"]
        0.6666666666666666
        >>> metrics["precision@5"]
        0.4

        >>> # With graded relevance scores
        >>> relevance = {"doc1": 1.0, "doc3": 0.5, "doc6": 0.8}
        >>> metrics = evaluate_search(retrieved, relevant, k=5, relevance_scores=relevance)
        >>> "ndcg@5" in metrics
        True
    """
    # Create binary relevance if not provided
    if relevance_scores is None:
        relevance_scores = {doc_id: 1.0 for doc_id in relevant}

    return {
        f"recall@{k}": recall_at_k(retrieved, relevant, k),
        f"precision@{k}": precision_at_k(retrieved, relevant, k),
        "mrr": mrr(retrieved, relevant),
        f"ndcg@{k}": ndcg_at_k(retrieved, relevance_scores, k),
        "ap": average_precision(retrieved, relevant),
    }
