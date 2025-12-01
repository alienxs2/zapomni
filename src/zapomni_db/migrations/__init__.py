"""
Database migrations for Zapomni.

This module provides idempotent database migrations for schema updates.
Each migration is versioned and can be run multiple times safely.

Migrations:
- 001_add_bitemporal: Add bi-temporal fields to Memory and Entity nodes (v0.8.0)
"""

from zapomni_db.migrations.migration_001_bitemporal import (
    MigrationResult,
    migrate_to_bitemporal,
)

__all__ = [
    "migrate_to_bitemporal",
    "MigrationResult",
]
