"""
Search module for Zapomni.

Provides vector similarity search functionality wrapping FalkorDB HNSW
with intelligent query preprocessing and result ranking.

Components:
    - VectorSearch: Main search interface for semantic search

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from zapomni_core.search.vector_search import VectorSearch

__all__ = ["VectorSearch"]
