"""
Core data models for the zapomni_core module.

This module re-exports the canonical Chunk model from zapomni_db.models
to ensure a single, consistent definition across core and database layers.
"""

from __future__ import annotations

from zapomni_db.models import Chunk

__all__ = ["Chunk"]
