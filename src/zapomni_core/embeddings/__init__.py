"""
Embeddings module for Zapomni.

Provides embedding generation via Ollama API with fallback to sentence-transformers,
and Redis-backed semantic caching for embeddings.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from zapomni_core.embeddings.ollama_embedder import OllamaEmbedder
from zapomni_core.embeddings.embedding_cache import EmbeddingCache

__all__ = ["OllamaEmbedder", "EmbeddingCache"]
