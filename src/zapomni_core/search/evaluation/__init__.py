"""
Search evaluation metrics module.

Provides standard information retrieval metrics for measuring
search result quality:
    - recall_at_k: Measures how many relevant items were retrieved
    - precision_at_k: Measures fraction of retrieved items that are relevant
    - mrr: Mean Reciprocal Rank - how quickly first relevant result appears
    - ndcg_at_k: Normalized Discounted Cumulative Gain with graded relevance
    - average_precision: Used for computing MAP over queries
    - evaluate_search: Compute all metrics for a single query

These metrics are used to evaluate and compare different search
strategies (vector, BM25, hybrid) and fusion methods (RRF, RSF, DBSF).

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from zapomni_core.search.evaluation.metrics import (
    average_precision,
    evaluate_search,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

__all__ = [
    "recall_at_k",
    "precision_at_k",
    "mrr",
    "ndcg_at_k",
    "average_precision",
    "evaluate_search",
]
