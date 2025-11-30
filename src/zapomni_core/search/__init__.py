"""
Search module for Zapomni.

Provides multiple search strategies for information retrieval:
- Vector similarity search (semantic)
- BM25 keyword search (lexical) with code-aware tokenization
- Hybrid search combining both with RRF
- Cross-encoder reranking for improved relevance
- Fusion strategies for combining multiple search results
- Evaluation metrics for search quality assessment

Components:
    - VectorSearch: Semantic search using embeddings
    - BM25Search: Keyword-based search using BM25 algorithm (bm25s)
    - CodeTokenizer: Code-aware tokenizer for programming identifiers
    - HybridSearch: Combines vector + BM25 with Reciprocal Rank Fusion
    - CrossEncoderReranker: Query-aware semantic reranking

Fusion Strategies:
    - FusionStrategy: Abstract base class for fusion strategies
    - RRFusion: Reciprocal Rank Fusion (RRF)
    - RSFusion: Relative Score Fusion
    - DBSFusion: Distribution-Based Score Fusion

Evaluation Metrics:
    - recall_at_k: Recall at K
    - precision_at_k: Precision at K
    - mrr: Mean Reciprocal Rank
    - ndcg_at_k: Normalized Discounted Cumulative Gain
    - average_precision: Average Precision
    - evaluate_search: Comprehensive search evaluation

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from zapomni_core.search.bm25_search import BM25Search
from zapomni_core.search.bm25_tokenizer import CodeTokenizer
from zapomni_core.search.evaluation import (
    average_precision,
    evaluate_search,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from zapomni_core.search.fusion import (
    DBSFusion,
    FusionStrategy,
    RRFusion,
    RSFusion,
)
from zapomni_core.search.hybrid_search import HybridSearch
from zapomni_core.search.reranker import CrossEncoderReranker
from zapomni_core.search.vector_search import VectorSearch

__all__ = [
    # Core search components
    "VectorSearch",
    "BM25Search",
    "CodeTokenizer",
    "HybridSearch",
    "CrossEncoderReranker",
    # Fusion strategies
    "FusionStrategy",
    "RRFusion",
    "RSFusion",
    "DBSFusion",
    # Evaluation metrics
    "recall_at_k",
    "precision_at_k",
    "mrr",
    "ndcg_at_k",
    "average_precision",
    "evaluate_search",
]
