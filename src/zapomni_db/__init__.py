"""
zapomni_db - Database layer for Zapomni memory system.

Provides FalkorDB client for unified vector + graph database operations.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from zapomni_db.exceptions import (
    ConnectionError,
    DatabaseError,
    QueryError,
    TransactionError,
    ValidationError,
)
from zapomni_db.falkordb_client import FalkorDBClient
from zapomni_db.models import Chunk, Memory, QueryResult, SearchResult
from zapomni_db.pool_config import PoolConfig, RetryConfig

__all__ = [
    "FalkorDBClient",
    "Memory",
    "Chunk",
    "SearchResult",
    "QueryResult",
    "PoolConfig",
    "RetryConfig",
    "ValidationError",
    "DatabaseError",
    "ConnectionError",
    "QueryError",
    "TransactionError",
]
