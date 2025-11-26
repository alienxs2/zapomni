"""
Embeddings module for Zapomni.

Provides embedding generation via Ollama API with fallback to sentence-transformers,
and Redis-backed semantic caching for embeddings.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from zapomni_core.embeddings.embedding_cache import EmbeddingCache
from zapomni_core.embeddings.ollama_embedder import OllamaEmbedder

__all__ = ["OllamaEmbedder", "EmbeddingCache"]
