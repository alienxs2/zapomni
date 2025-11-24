"""
Entity extraction components for zapomni_core.

Provides hybrid NER (Named Entity Recognition) for knowledge graph construction
using fast SpaCy-based extraction with optional LLM refinement (Phase 2).

Modules:
    entity_extractor: Main EntityExtractor component

Classes:
    EntityExtractor: Hybrid entity extraction using SpaCy + LLM
    Entity: Data model for extracted entities
    Relationship: Data model for entity relationships

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from __future__ import annotations

from zapomni_core.extractors.entity_extractor import (
    Entity,
    EntityExtractor,
    Relationship,
)

__all__ = [
    "EntityExtractor",
    "Entity",
    "Relationship",
]
