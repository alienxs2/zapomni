"""
VectorSearch - Vector similarity search wrapper for FalkorDB HNSW.

Orchestrates query preprocessing, embedding generation via OllamaEmbedder,
and vector search via FalkorDBClient. Provides a clean interface for
semantic search operations.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from typing import Any, Dict, List, Optional

import structlog

from zapomni_core.exceptions import EmbeddingError, SearchError, ValidationError
from zapomni_db.exceptions import DatabaseError
from zapomni_db.models import SearchResult

logger = structlog.get_logger(__name__)


class VectorSearch:
    """
    Vector similarity search wrapper for FalkorDB HNSW.

    Provides high-level semantic search interface by orchestrating:
    1. Query text preprocessing and validation
    2. Embedding generation via OllamaEmbedder
    3. Vector similarity search via FalkorDBClient
    4. Result ranking and formatting

    Attributes:
        db_client: FalkorDBClient instance for vector search operations
        embedder: OllamaEmbedder instance for text-to-embedding conversion

    Example:
        ```python
        from zapomni_core.search import VectorSearch
        from zapomni_db.falkordb_client import FalkorDBClient
        from zapomni_core.embeddings.ollama_embedder import OllamaEmbedder

        db_client = FalkorDBClient()
        embedder = OllamaEmbedder()
        search = VectorSearch(db_client=db_client, embedder=embedder)

        # Search with defaults
        results = await search.search("Python programming")

        # Search with limit and filters
        results = await search.search(
            query="machine learning",
            limit=5,
            filters={"tags": ["python"], "source": "docs"}
        )

        for result in results:
            print(f"{result.similarity_score:.2f}: {result.text}")
        ```
    """

    def __init__(self, db_client: Any, embedder: Any) -> None:
        """
        Initialize VectorSearch with dependencies.

        Args:
            db_client: FalkorDBClient instance (must have vector_search method)
            embedder: OllamaEmbedder instance (must have embed_text method)

        Raises:
            ValidationError: If dependencies are None or invalid

        Example:
            ```python
            from zapomni_db.falkordb_client import FalkorDBClient
            from zapomni_core.embeddings.ollama_embedder import OllamaEmbedder

            db_client = FalkorDBClient()
            embedder = OllamaEmbedder()
            search = VectorSearch(db_client=db_client, embedder=embedder)
            ```
        """
        # Validate dependencies
        if db_client is None:
            raise ValidationError(
                message="db_client cannot be None",
                error_code="VAL_001",
                details={"parameter": "db_client"},
            )

        if embedder is None:
            raise ValidationError(
                message="embedder cannot be None",
                error_code="VAL_001",
                details={"parameter": "embedder"},
            )

        self.db_client = db_client
        self.embedder = embedder

        logger.info(
            "vector_search_initialized",
            db_client_type=type(db_client).__name__,
            embedder_type=type(embedder).__name__,
        )

    async def search(
        self, query: str, limit: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Perform semantic search on stored memories.

        Algorithm:
        1. Validate and preprocess query (trim whitespace, check empty)
        2. Validate limit (1-1000 range)
        3. Convert query text to embedding via OllamaEmbedder
        4. Execute vector similarity search via FalkorDBClient
        5. Return ranked results (sorted by similarity score descending)

        Args:
            query: Natural language search query (e.g., "Python programming")
            limit: Maximum number of results to return (1-1000, default: 10)
            filters: Optional metadata filters for results:
                - tags: List[str] - Filter by tags
                - source: str - Filter by source document
                - date_from: str - ISO 8601 date string
                - date_to: str - ISO 8601 date string
                - min_similarity: float - Minimum similarity score (0.0-1.0)

        Returns:
            List[SearchResult]: Ranked search results sorted by similarity score.
                Returns empty list if no results found (not an error).

        Raises:
            ValidationError: If query is empty or limit is invalid
            SearchError: If embedding generation or database search fails

        Performance Target:
            - Single search: < 200ms (P95) including embedding generation
            - Empty result: < 50ms

        Example:
            ```python
            search = VectorSearch(db_client, embedder)

            # Basic search
            results = await search.search("Python programming")
            for result in results:
                print(f"{result.similarity_score:.2f}: {result.text}")

            # Search with limit
            top_5 = await search.search("machine learning", limit=5)

            # Search with filters
            results = await search.search(
                query="data science",
                limit=20,
                filters={
                    "tags": ["python", "tutorial"],
                    "source": "documentation",
                    "min_similarity": 0.8
                }
            )

            # Handle no results
            results = await search.search("nonexistent topic")
            if not results:
                print("No results found")
            ```
        """
        # STEP 1: QUERY PREPROCESSING AND VALIDATION

        # Trim whitespace
        query = query.strip()

        # Validate query not empty
        if not query:
            raise ValidationError(
                message="Query cannot be empty", error_code="VAL_001", details={"query": query}
            )

        # Validate limit range
        if not isinstance(limit, int):
            raise ValidationError(
                message=f"Limit must be int, got {type(limit).__name__}",
                error_code="VAL_002",
                details={"limit": limit, "type": type(limit).__name__},
            )

        if limit < 1:
            raise ValidationError(
                message=f"Limit must be >= 1, got {limit}",
                error_code="VAL_003",
                details={"limit": limit},
            )

        if limit > 1000:
            raise ValidationError(
                message=f"Limit cannot exceed 1000, got {limit}",
                error_code="VAL_003",
                details={"limit": limit, "max": 1000},
            )

        # STEP 2: GENERATE QUERY EMBEDDING

        try:
            logger.debug(
                "generating_query_embedding",
                query_length=len(query),
                limit=limit,
                has_filters=filters is not None,
            )

            query_embedding = await self.embedder.embed_text(query)

            logger.debug("query_embedding_generated", embedding_dimensions=len(query_embedding))

        except EmbeddingError as e:
            logger.error(
                "embedding_generation_failed",
                error=str(e),
                query=query[:100],  # Log first 100 chars
            )
            raise SearchError(
                message=f"Embedding generation failed: {str(e)}",
                error_code="SEARCH_001",
                details={"query": query[:100], "embedding_error": str(e)},
                original_exception=e,
            )

        except Exception as e:
            logger.error("unexpected_embedding_error", error=str(e), error_type=type(e).__name__)
            raise SearchError(
                message=f"Unexpected error during embedding generation: {str(e)}",
                error_code="SEARCH_001",
                details={"error_type": type(e).__name__},
                original_exception=e,
            )

        # STEP 3: EXECUTE VECTOR SEARCH

        try:
            logger.debug(
                "executing_vector_search",
                embedding_dimensions=len(query_embedding),
                limit=limit,
                filters=filters,
            )

            results = await self.db_client.vector_search(
                embedding=query_embedding, limit=limit, filters=filters
            )

            logger.info(
                "vector_search_completed",
                query=query[:100],
                result_count=len(results),
                limit=limit,
                has_filters=filters is not None,
            )

            return results

        except DatabaseError as e:
            logger.error("vector_search_failed", error=str(e), query=query[:100])
            raise SearchError(
                message=f"Vector search failed: {str(e)}",
                error_code="SEARCH_001",
                details={"query": query[:100], "database_error": str(e)},
                original_exception=e,
            )

        except Exception as e:
            logger.error("unexpected_search_error", error=str(e), error_type=type(e).__name__)
            raise SearchError(
                message=f"Unexpected error during vector search: {str(e)}",
                error_code="SEARCH_001",
                details={"error_type": type(e).__name__},
                original_exception=e,
            )
