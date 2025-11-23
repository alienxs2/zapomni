"""
Custom exceptions for zapomni_db module.
"""


class DatabaseError(Exception):
    """Raised when database operation fails after retries."""
    pass


class ConnectionError(DatabaseError):
    """Raised when cannot connect to FalkorDB."""
    pass


class QueryError(DatabaseError):
    """Raised when Cypher query execution fails."""
    pass


class QuerySyntaxError(QueryError):
    """Raised when Cypher query syntax is invalid."""
    pass


class TransactionError(DatabaseError):
    """Raised when transaction state is invalid."""
    pass


class ValidationError(Exception):
    """Raised when input validation fails."""
    pass
