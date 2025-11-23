"""
Search module for Zapomni.

Provides multiple search strategies for information retrieval:
- Vector similarity search (semantic)
- BM25 keyword search (lexical)

Components:
    - VectorSearch: Main search interface for semantic search
    - BM25Search: Keyword-based search using BM25 algorithm

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from zapomni_core.search.vector_search import VectorSearch
from zapomni_core.search.bm25_search import BM25Search

__all__ = ["VectorSearch", "BM25Search"]
