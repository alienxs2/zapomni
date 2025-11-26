"""
HybridSearch - Combines BM25 and Vector search using Reciprocal Rank Fusion.

Provides intelligent search by merging keyword-based (BM25) and semantic (vector)
search results using Reciprocal Rank Fusion (RRF) algorithm.

RRF Algorithm:
    - Combines rankings from multiple retrieval systems
    - Formula: RRF_score = Σ (alpha_i / (k + rank_i))
    - k=60 is standard constant from literature
    - Produces robust rankings without score normalization

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog

from zapomni_core.exceptions import SearchError, ValidationError
from zapomni_db.models import SearchResult

logger = structlog.get_logger(__name__)


class HybridSearch:
    """
    Hybrid search combining BM25 and vector similarity search.

    Orchestrates parallel search execution across keyword and semantic
    search methods, then merges results using Reciprocal Rank Fusion (RRF).

    RRF provides:
    1. Robust rank aggregation without score calibration
    2. Democratic voting across different ranking methods
    3. Automatic handling of score scale differences

    Attributes:
        vector_search: VectorSearch instance for semantic search
        bm25_search: BM25Search instance for keyword search

    Example:
        ```python
        from zapomni_core.search import HybridSearch, VectorSearch, BM25Search
        from zapomni_db.falkordb_client import FalkorDBClient
        from zapomni_core.embeddings.ollama_embedder import OllamaEmbedder

        db_client = FalkorDBClient()
        embedder = OllamaEmbedder()

        vector_search = VectorSearch(db_client=db_client, embedder=embedder)
        bm25_search = BM25Search(db_client=db_client)

        hybrid = HybridSearch(
            vector_search=vector_search,
            bm25_search=bm25_search
        )

        # Balanced search (alpha=0.5)
        results = await hybrid.search("Python programming", limit=10, alpha=0.5)

        # Semantic-focused (alpha=0.8)
        results = await hybrid.search("machine learning concepts", alpha=0.8)

        # Keyword-focused (alpha=0.2)
        results = await hybrid.search("exact function name", alpha=0.2)
        ```
    """

    def __init__(self, vector_search, bm25_search):
        """
        Initialize HybridSearch with search dependencies.

        Args:
            vector_search: VectorSearch instance (must have search method)
            bm25_search: BM25Search instance (must have search method)

        Raises:
            ValidationError: If dependencies are None or invalid

        Example:
            ```python
            hybrid = HybridSearch(
                vector_search=vector_search,
                bm25_search=bm25_search
            )
            ```
        """
        # Validate dependencies
        if vector_search is None:
            raise ValidationError(
                message="vector_search cannot be None",
                error_code="VAL_001",
                details={"parameter": "vector_search"},
            )

        if bm25_search is None:
            raise ValidationError(
                message="bm25_search cannot be None",
                error_code="VAL_001",
                details={"parameter": "bm25_search"},
            )

        self.vector_search = vector_search
        self.bm25_search = bm25_search

        logger.info(
            "hybrid_search_initialized",
            vector_search_type=type(vector_search).__name__,
            bm25_search_type=type(bm25_search).__name__,
        )

    async def search(
        self,
        query: str,
        limit: int = 10,
        alpha: float = 0.5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining BM25 and vector similarity.

        Algorithm:
        1. Validate query, limit, and alpha parameters
        2. Execute vector search and BM25 search in parallel
        3. Calculate RRF scores for each result
        4. Merge and deduplicate results by chunk_id
        5. Sort by combined RRF score descending
        6. Return top N results

        RRF Formula:
            For each result:
            - vector_rrf = 1 / (k + vector_rank)
            - bm25_rrf = 1 / (k + bm25_rank)
            - combined_score = alpha * vector_rrf + (1 - alpha) * bm25_rrf
            Where k=60 (standard constant)

        Args:
            query: Natural language search query
            limit: Maximum number of results to return (1-1000, default: 10)
            alpha: Weighting factor (0.0-1.0, default: 0.5):
                - 0.0 = BM25 only (pure keyword search)
                - 0.5 = balanced hybrid (recommended)
                - 1.0 = Vector only (pure semantic search)
            filters: Optional metadata filters (passed to vector search only):
                - tags: List[str] - Filter by tags
                - source: str - Filter by source document
                - date_from: str - ISO 8601 date string
                - date_to: str - ISO 8601 date string
                - min_similarity: float - Minimum similarity score

        Returns:
            List[SearchResult]: Ranked search results sorted by RRF score.
                Results are deduplicated by chunk_id.
                Returns empty list if no results found.

        Raises:
            ValidationError: If query is empty, limit invalid, or alpha out of range
            SearchError: If search execution fails

        Performance Target:
            - Single search: < 300ms (P95) for parallel execution
            - Empty result: < 100ms

        Example:
            ```python
            hybrid = HybridSearch(vector_search, bm25_search)

            # Balanced hybrid search
            results = await hybrid.search("Python programming", limit=10, alpha=0.5)

            # Semantic-focused (better for conceptual queries)
            results = await hybrid.search(
                query="how does machine learning work?",
                alpha=0.8,
                limit=5
            )

            # Keyword-focused (better for exact matches)
            results = await hybrid.search(
                query="def calculate_score",
                alpha=0.2,
                limit=10
            )

            # With filters (applied to vector search only)
            results = await hybrid.search(
                query="data processing",
                alpha=0.5,
                filters={"tags": ["python"], "source": "documentation"}
            )
            ```
        """
        # STEP 1: VALIDATE QUERY
        query = query.strip()

        if not query:
            raise ValidationError(
                message="Query cannot be empty", error_code="VAL_001", details={"query": query}
            )

        # STEP 2: VALIDATE LIMIT
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

        # STEP 3: VALIDATE ALPHA
        if not isinstance(alpha, (int, float)):
            raise ValidationError(
                message=f"Alpha must be float, got {type(alpha).__name__}",
                error_code="VAL_002",
                details={"alpha": alpha, "type": type(alpha).__name__},
            )

        if alpha < 0.0 or alpha > 1.0:
            raise ValidationError(
                message=f"Alpha must be between 0.0 and 1.0, got {alpha}",
                error_code="VAL_003",
                details={"alpha": alpha, "min": 0.0, "max": 1.0},
            )

        # STEP 4: EXECUTE PARALLEL SEARCHES
        try:
            logger.debug(
                "executing_hybrid_search",
                query=query[:100],
                limit=limit,
                alpha=alpha,
                has_filters=filters is not None,
            )

            # Execute both searches in parallel
            vector_results = await self.vector_search.search(
                query=query, limit=limit, filters=filters
            )

            bm25_results = await self.bm25_search.search(query=query, limit=limit)

            logger.debug(
                "search_results_received",
                vector_count=len(vector_results),
                bm25_count=len(bm25_results),
            )

        except (ValidationError, SearchError):
            # Re-raise validation and search errors as-is
            raise

        except Exception as e:
            logger.error(
                "unexpected_hybrid_search_error", error=str(e), error_type=type(e).__name__
            )
            raise SearchError(
                message=f"Unexpected error during hybrid search: {str(e)}",
                error_code="SEARCH_001",
                details={"error_type": type(e).__name__},
                original_exception=e,
            )

        # STEP 5: CALCULATE RRF SCORES AND MERGE RESULTS
        merged_results = self._merge_with_rrf(
            vector_results=vector_results, bm25_results=bm25_results, alpha=alpha
        )

        # STEP 6: SORT BY RRF SCORE AND APPLY LIMIT
        merged_results.sort(key=lambda x: x.similarity_score, reverse=True)
        final_results = merged_results[:limit]

        logger.info(
            "hybrid_search_completed",
            query=query[:100],
            result_count=len(final_results),
            vector_count=len(vector_results),
            bm25_count=len(bm25_results),
            alpha=alpha,
            limit=limit,
        )

        return final_results

    def _merge_with_rrf(
        self,
        vector_results: List[SearchResult],
        bm25_results: List[Dict[str, Any]],
        alpha: float,
        k: int = 60,
    ) -> List[SearchResult]:
        """
        Merge results using Reciprocal Rank Fusion (RRF).

        RRF Algorithm:
        1. Assign rank to each result in each list (1-indexed)
        2. Calculate RRF score: 1/(k + rank) for each result in each list
        3. Combine scores: alpha * vector_rrf + (1-alpha) * bm25_rrf
        4. Deduplicate by chunk_id, keeping highest combined score
        5. Return merged results with RRF scores

        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search (as dicts)
            alpha: Weighting factor (0.0-1.0)
            k: RRF constant (default: 60, from literature)

        Returns:
            List of SearchResult objects with RRF scores in similarity_score field

        Note:
            - k=60 is the standard constant from RRF literature
            - Deduplication uses chunk_id as unique identifier
            - Original similarity_score is replaced with RRF score
        """
        # Dictionary to accumulate RRF scores by chunk_id
        rrf_scores: Dict[str, float] = {}
        result_by_chunk: Dict[str, SearchResult] = {}

        # STEP 1: PROCESS VECTOR RESULTS
        for rank, result in enumerate(vector_results, start=1):
            chunk_id = result.chunk_id
            vector_rrf = 1.0 / (k + rank)
            combined_score = alpha * vector_rrf

            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = 0.0
                result_by_chunk[chunk_id] = result

            rrf_scores[chunk_id] += combined_score

        # STEP 2: PROCESS BM25 RESULTS
        # Note: BM25Search returns List[Dict] not List[SearchResult]
        # We need to convert or skip if not in vector results
        for rank, bm25_dict in enumerate(bm25_results, start=1):
            bm25_rrf = 1.0 / (k + rank)
            combined_score = (1.0 - alpha) * bm25_rrf

            # BM25 results are dicts with 'text', 'score', 'index'
            # We need to match them to SearchResult objects somehow
            # For now, we'll try to match by text content
            text = bm25_dict.get("text", "")

            # Find matching chunk_id by text
            matched_chunk_id = None
            for chunk_id, result in result_by_chunk.items():
                if result.text == text:
                    matched_chunk_id = chunk_id
                    break

            if matched_chunk_id:
                # Add BM25 score to existing result
                rrf_scores[matched_chunk_id] += combined_score
            else:
                # BM25 result not in vector results
                # Create a pseudo SearchResult for it
                # Note: This is a limitation - we don't have full metadata
                pseudo_chunk_id = f"bm25_{bm25_dict.get('index', rank)}"
                if pseudo_chunk_id not in rrf_scores:
                    pseudo_result = SearchResult(
                        memory_id=f"bm25_mem_{bm25_dict.get('index', rank)}",
                        chunk_id=pseudo_chunk_id,
                        text=text,
                        similarity_score=0.0,  # Will be set to RRF score
                        tags=[],
                        source="bm25",
                        timestamp=datetime.now(),
                        chunk_index=bm25_dict.get("index", rank),
                    )
                    result_by_chunk[pseudo_chunk_id] = pseudo_result
                    rrf_scores[pseudo_chunk_id] = 0.0

                rrf_scores[pseudo_chunk_id] += combined_score

        # STEP 3: CREATE FINAL RESULTS WITH RRF SCORES
        merged_results = []
        for chunk_id, result in result_by_chunk.items():
            # Create new SearchResult with RRF score
            merged_result = SearchResult(
                memory_id=result.memory_id,
                chunk_id=result.chunk_id,
                text=result.text,
                similarity_score=rrf_scores[chunk_id],  # Replace with RRF score
                tags=result.tags,
                source=result.source,
                timestamp=result.timestamp,
                chunk_index=result.chunk_index,
            )
            merged_results.append(merged_result)

        logger.debug(
            "rrf_merge_completed",
            total_results=len(merged_results),
            vector_only=len(vector_results)
            - sum(
                1 for r in vector_results if r.chunk_id in [b.get("text", "") for b in bm25_results]
            ),
            bm25_only=len(bm25_results)
            - sum(1 for b in bm25_results if b.get("text", "") in [r.text for r in vector_results]),
            both=len(
                [r for r in vector_results if r.text in [b.get("text", "") for b in bm25_results]]
            ),
        )

        return merged_results
