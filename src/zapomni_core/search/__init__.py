"""
Search module for Zapomni.

Provides multiple search strategies for information retrieval:
- Vector similarity search (semantic)
- BM25 keyword search (lexical)
- Hybrid search combining both with RRF
- Cross-encoder reranking for improved relevance

Components:
    - VectorSearch: Semantic search using embeddings
    - BM25Search: Keyword-based search using BM25 algorithm
    - HybridSearch: Combines vector + BM25 with Reciprocal Rank Fusion
    - CrossEncoderReranker: Query-aware semantic reranking

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from zapomni_core.search.vector_search import VectorSearch
from zapomni_core.search.bm25_search import BM25Search
from zapomni_core.search.hybrid_search import HybridSearch
from zapomni_core.search.reranker import CrossEncoderReranker

__all__ = ["VectorSearch", "BM25Search", "HybridSearch", "CrossEncoderReranker"]
