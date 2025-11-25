"""
LLM module for Zapomni.

Provides LLM inference capabilities via Ollama for:
- Entity refinement (enhancing SpaCy NER results)
- Relationship extraction (detecting connections between entities)

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from zapomni_core.llm.ollama_llm import OllamaLLMClient

__all__ = ["OllamaLLMClient"]
