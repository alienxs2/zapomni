"""
zapomni_db - Database layer for Zapomni memory system.

Provides FalkorDB client for unified vector + graph database operations.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from zapomni_db.falkordb_client import FalkorDBClient
from zapomni_db.models import Memory, Chunk, SearchResult, QueryResult
from zapomni_db.exceptions import (
    ValidationError,
    DatabaseError,
    ConnectionError,
    QueryError,
    TransactionError
)

__all__ = [
    "FalkorDBClient",
    "Memory",
    "Chunk",
    "SearchResult",
    "QueryResult",
    "ValidationError",
    "DatabaseError",
    "ConnectionError",
    "QueryError",
    "TransactionError",
]
