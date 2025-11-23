"""
Embeddings module for Zapomni.

Provides embedding generation via Ollama API with fallback to sentence-transformers.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from zapomni_core.embeddings.ollama_embedder import OllamaEmbedder

__all__ = ["OllamaEmbedder"]
