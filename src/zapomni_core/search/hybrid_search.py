"""
HybridSearch - Combines BM25 and Vector search with pluggable fusion strategies.

Provides intelligent search by merging keyword-based (BM25) and semantic (vector)
search results using configurable fusion algorithms. Supports parallel execution
for optimal performance.

Fusion Strategies:
    - RRF (Reciprocal Rank Fusion): Rank-based fusion, standard k=60
    - RSF (Relative Score Fusion): Min-max normalization score fusion
    - DBSF (Distribution-Based Score Fusion): 3-sigma normalization

Parallel Execution:
    - Vector search and BM25 search run concurrently via asyncio.gather()
    - BM25 (synchronous) wrapped with asyncio.to_thread()
    - Graceful error handling for individual search failures

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

import structlog

from zapomni_core.exceptions import SearchError, ValidationError
from zapomni_core.search.fusion import DBSFusion, RRFusion, RSFusion
from zapomni_db.models import SearchResult

logger = structlog.get_logger(__name__)


class HybridSearch:
    """
    Hybrid search combining BM25 and vector similarity search.

    Orchestrates TRUE PARALLEL search execution across keyword and semantic
    search methods using asyncio.gather(), then merges results using
    configurable fusion strategies.

    Fusion Strategies:
        - RRF (Reciprocal Rank Fusion): Rank-based, robust without calibration
        - RSF (Relative Score Fusion): Score-based with min-max normalization
        - DBSF (Distribution-Based Score Fusion): 3-sigma normalization

    Performance:
        - Parallel execution reduces latency by ~50%
        - BM25 (synchronous) wrapped in asyncio.to_thread()
        - Graceful degradation on individual search failures

    Attributes:
        vector_search: VectorSearch instance for semantic search
        bm25_search: BM25Search instance for keyword search
        fusion_method: Default fusion strategy ("rrf", "rsf", "dbsf")
        fusion_k: RRF smoothing constant (default: 60)

    Example:
        ```python
        from zapomni_core.search import HybridSearch, VectorSearch, BM25Search
        from zapomni_db.falkordb_client import FalkorDBClient
        from zapomni_core.embeddings.ollama_embedder import OllamaEmbedder

        db_client = FalkorDBClient()
        embedder = OllamaEmbedder()

        vector_search = VectorSearch(db_client=db_client, embedder=embedder)
        bm25_search = BM25Search(db_client=db_client)

        # Default: RRF fusion with k=60
        hybrid = HybridSearch(
            vector_search=vector_search,
            bm25_search=bm25_search
        )

        # Custom fusion strategy
        hybrid = HybridSearch(
            vector_search=vector_search,
            bm25_search=bm25_search,
            fusion_method="dbsf",  # Use 3-sigma normalization
            fusion_k=60,
        )

        # Balanced search (alpha=0.5)
        results = await hybrid.search("Python programming", limit=10, alpha=0.5)

        # Override fusion method per-query
        results = await hybrid.search(
            "machine learning concepts",
            alpha=0.8,
            fusion_method="rsf"
        )
        ```
    """

    def __init__(
        self,
        vector_search: Any,
        bm25_search: Any,
        fusion_method: Literal["rrf", "rsf", "dbsf"] = "rrf",
        fusion_k: int = 60,
    ) -> None:
        """
        Initialize HybridSearch with search dependencies and fusion configuration.

        Args:
            vector_search: VectorSearch instance (must have async search method)
            bm25_search: BM25Search instance (must have sync search method)
            fusion_method: Default fusion strategy to use:
                - "rrf": Reciprocal Rank Fusion (default, recommended)
                - "rsf": Relative Score Fusion (min-max normalization)
                - "dbsf": Distribution-Based Score Fusion (3-sigma)
            fusion_k: RRF smoothing constant (default: 60).
                Only used when fusion_method="rrf".
                Lower values emphasize top ranks more strongly.

        Raises:
            ValidationError: If dependencies are None or fusion_k is invalid

        Example:
            ```python
            # Default RRF with k=60
            hybrid = HybridSearch(
                vector_search=vector_search,
                bm25_search=bm25_search
            )

            # Custom: DBSF fusion
            hybrid = HybridSearch(
                vector_search=vector_search,
                bm25_search=bm25_search,
                fusion_method="dbsf"
            )

            # Custom: RRF with k=20 (emphasize top ranks more)
            hybrid = HybridSearch(
                vector_search=vector_search,
                bm25_search=bm25_search,
                fusion_method="rrf",
                fusion_k=20
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

        # Validate fusion_k
        if not isinstance(fusion_k, int) or fusion_k <= 0:
            raise ValidationError(
                message=f"fusion_k must be a positive integer, got {fusion_k}",
                error_code="VAL_003",
                details={"fusion_k": fusion_k},
            )

        self.vector_search = vector_search
        self.bm25_search = bm25_search
        self._fusion_method: Literal["rrf", "rsf", "dbsf"] = fusion_method
        self._fusion_k = fusion_k

        logger.info(
            "hybrid_search_initialized",
            vector_search_type=type(vector_search).__name__,
            bm25_search_type=type(bm25_search).__name__,
            fusion_method=fusion_method,
            fusion_k=fusion_k,
        )

    async def search(
        self,
        query: str,
        limit: int = 10,
        alpha: float = 0.5,
        filters: Optional[Dict[str, Any]] = None,
        fusion_method: Optional[Literal["rrf", "rsf", "dbsf"]] = None,
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining BM25 and vector similarity.

        Algorithm:
        1. Validate query, limit, alpha, and fusion_method parameters
        2. Execute vector search and BM25 search in TRUE PARALLEL
        3. Fuse results using selected fusion strategy (RRF/RSF/DBSF)
        4. Merge and deduplicate results by chunk_id
        5. Sort by fused score descending
        6. Return top N results

        Parallel Execution:
            - Vector search (async) runs concurrently with BM25 search
            - BM25 search (sync) is wrapped in asyncio.to_thread()
            - Individual search failures are handled gracefully
            - Reduces overall latency by ~50%

        Fusion Strategies:
            - RRF: score = alpha/(k+rank_v) + (1-alpha)/(k+rank_b), k=60
            - RSF: score = alpha*norm_v + (1-alpha)*norm_b (min-max normalized)
            - DBSF: score = alpha*norm_v + (1-alpha)*norm_b (3-sigma normalized)

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
            fusion_method: Override instance default fusion strategy:
                - "rrf": Reciprocal Rank Fusion (rank-based)
                - "rsf": Relative Score Fusion (score-based, min-max)
                - "dbsf": Distribution-Based Score Fusion (3-sigma)
                - None: Use instance default (self._fusion_method)

        Returns:
            List[SearchResult]: Ranked search results sorted by fused score.
                Results are deduplicated by chunk_id.
                Returns empty list if no results found.

        Raises:
            ValidationError: If query is empty, limit invalid, or alpha out of range
            SearchError: If both searches fail

        Performance Target:
            - Single search: < 200ms (P95) with parallel execution
            - Empty result: < 50ms

        Example:
            ```python
            hybrid = HybridSearch(vector_search, bm25_search)

            # Balanced hybrid search with default RRF
            results = await hybrid.search("Python programming", limit=10, alpha=0.5)

            # Override fusion method per-query
            results = await hybrid.search(
                query="machine learning concepts",
                alpha=0.8,
                fusion_method="dbsf"  # Use 3-sigma normalization
            )

            # Keyword-focused with RSF
            results = await hybrid.search(
                query="def calculate_score",
                alpha=0.2,
                fusion_method="rsf"
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

        # STEP 4: DETERMINE FUSION METHOD
        effective_fusion = fusion_method if fusion_method is not None else self._fusion_method

        # STEP 5: EXECUTE TRUE PARALLEL SEARCHES
        logger.debug(
            "executing_hybrid_search",
            query=query[:100],
            limit=limit,
            alpha=alpha,
            has_filters=filters is not None,
            fusion_method=effective_fusion,
        )

        # Create async tasks for parallel execution
        # Vector search is async, BM25 search is sync (wrapped in to_thread)
        vector_task = self.vector_search.search(query=query, limit=limit, filters=filters)
        bm25_task = asyncio.to_thread(self.bm25_search.search, query, limit)

        # Execute both searches in TRUE parallel with asyncio.gather
        # return_exceptions=True allows us to handle individual failures gracefully
        results = await asyncio.gather(vector_task, bm25_task, return_exceptions=True)

        vector_results_raw, bm25_results_raw = results

        # Handle vector search exceptions
        vector_results: List[SearchResult]
        if isinstance(vector_results_raw, BaseException):
            logger.error(
                "vector_search_failed",
                error=str(vector_results_raw),
                error_type=type(vector_results_raw).__name__,
            )
            vector_results = []
        else:
            vector_results = vector_results_raw

        # Handle BM25 search exceptions
        bm25_results: List[Dict[str, Any]]
        if isinstance(bm25_results_raw, BaseException):
            logger.error(
                "bm25_search_failed",
                error=str(bm25_results_raw),
                error_type=type(bm25_results_raw).__name__,
            )
            bm25_results = []
        else:
            bm25_results = bm25_results_raw

        # If both searches failed, raise an error
        if isinstance(vector_results_raw, Exception) and isinstance(bm25_results_raw, Exception):
            raise SearchError(
                message="Both vector and BM25 searches failed",
                error_code="SEARCH_001",
                details={
                    "vector_error": str(vector_results_raw),
                    "bm25_error": str(bm25_results_raw),
                },
            )

        logger.debug(
            "search_results_received",
            vector_count=len(vector_results),
            bm25_count=len(bm25_results),
        )

        # STEP 6: FUSE RESULTS USING SELECTED STRATEGY
        merged_results = self._fuse_results(
            vector_results=vector_results,
            bm25_results=bm25_results,
            alpha=alpha,
            fusion_method=effective_fusion,
        )

        # STEP 7: SORT BY FUSED SCORE AND APPLY LIMIT
        merged_results.sort(key=lambda x: x.similarity_score or 0.0, reverse=True)
        final_results = merged_results[:limit]

        logger.info(
            "hybrid_search_completed",
            query=query[:100],
            result_count=len(final_results),
            vector_count=len(vector_results),
            bm25_count=len(bm25_results),
            alpha=alpha,
            limit=limit,
            fusion_method=effective_fusion,
        )

        return final_results

    def _fuse_results(
        self,
        vector_results: List[SearchResult],
        bm25_results: List[Dict[str, Any]],
        alpha: float,
        fusion_method: Literal["rrf", "rsf", "dbsf"],
    ) -> List[SearchResult]:
        """
        Fuse results using the specified fusion strategy.

        Supports three fusion strategies:
        - RRF (Reciprocal Rank Fusion): Rank-based, robust without calibration
        - RSF (Relative Score Fusion): Score-based with min-max normalization
        - DBSF (Distribution-Based Score Fusion): 3-sigma normalization

        Algorithm:
        1. Build chunk_id -> (rank, score) mappings from both result sets
        2. Apply selected fusion strategy to compute combined scores
        3. Create SearchResult objects with fused scores
        4. Return merged results (unsorted - caller handles sorting)

        Args:
            vector_results: Results from vector search (SearchResult objects)
            bm25_results: Results from BM25 search (dicts with 'id', 'text', 'score')
            alpha: Weighting factor (0.0-1.0):
                - 0.0 = BM25 only
                - 0.5 = balanced
                - 1.0 = Vector only
            fusion_method: Which fusion strategy to use:
                - "rrf": Reciprocal Rank Fusion
                - "rsf": Relative Score Fusion
                - "dbsf": Distribution-Based Score Fusion

        Returns:
            List of SearchResult objects with fused scores in similarity_score field.
            Results are NOT sorted (caller is responsible for sorting).

        Note:
            - BM25 results should have "id" field (chunk_id) for proper matching
            - Results without chunk_id are skipped from vector results
            - BM25-only results get a pseudo SearchResult created
        """
        # STEP 1: BUILD VECTOR RESULTS MAPPING
        # chunk_id -> (rank, score) for fusion strategies
        vector_map: Dict[str, Tuple[int, float]] = {}
        vector_by_id: Dict[str, SearchResult] = {}

        for rank, result in enumerate(vector_results, start=1):
            if result.chunk_id is None:
                continue  # Skip results without chunk_id
            vector_map[result.chunk_id] = (rank, result.similarity_score or 0.0)
            vector_by_id[result.chunk_id] = result

        # STEP 2: BUILD BM25 RESULTS MAPPING
        # BM25Search returns dicts with 'id' (chunk_id), 'text', 'score', 'index'
        bm25_map: Dict[str, Tuple[int, float]] = {}
        bm25_by_id: Dict[str, Dict[str, Any]] = {}

        for rank, bm25_dict in enumerate(bm25_results, start=1):
            # Get chunk_id from 'id' field (set during index_documents with document_ids)
            chunk_id = bm25_dict.get("id")

            if chunk_id is None:
                # Fallback: Try to match by text content
                text = bm25_dict.get("text", "")
                for v_chunk_id, v_result in vector_by_id.items():
                    if v_result.text == text:
                        chunk_id = v_chunk_id
                        break

            if chunk_id is None:
                # Create pseudo chunk_id for BM25-only results
                chunk_id = f"bm25_{bm25_dict.get('index', rank)}"

            bm25_map[chunk_id] = (rank, bm25_dict.get("score", 0.0))
            bm25_by_id[chunk_id] = bm25_dict

        # STEP 3: SELECT AND APPLY FUSION STRATEGY
        fused_scores: Dict[str, float]
        if fusion_method == "rrf":
            fused_scores = RRFusion(k=self._fusion_k).fuse(vector_map, bm25_map, alpha)
        elif fusion_method == "rsf":
            fused_scores = RSFusion().fuse(vector_map, bm25_map, alpha)
        else:  # dbsf
            fused_scores = DBSFusion().fuse(vector_map, bm25_map, alpha)

        # STEP 4: BUILD FINAL RESULTS WITH FUSED SCORES
        merged_results: List[SearchResult] = []
        all_chunk_ids = set(vector_map.keys()) | set(bm25_map.keys())

        for chunk_id in all_chunk_ids:
            fused_score = fused_scores.get(chunk_id, 0.0)

            if chunk_id in vector_by_id:
                # Use vector result as base (has full metadata)
                result = vector_by_id[chunk_id]
                merged_result = SearchResult(
                    memory_id=result.memory_id,
                    content=result.text or result.content,
                    relevance_score=fused_score,
                    chunk_id=result.chunk_id,
                    text=result.text,
                    similarity_score=fused_score,
                    tags=result.tags,
                    source=result.source,
                    timestamp=result.timestamp,
                    chunk_index=result.chunk_index,
                )
            else:
                # BM25-only result: create pseudo SearchResult
                bm25_dict = bm25_by_id.get(chunk_id, {})
                text = bm25_dict.get("text", "")
                index = bm25_dict.get("index", 0)

                merged_result = SearchResult(
                    memory_id=f"bm25_mem_{index}",
                    content=text,
                    relevance_score=fused_score,
                    chunk_id=chunk_id,
                    text=text,
                    similarity_score=fused_score,
                    tags=[],
                    source="bm25",
                    timestamp=datetime.now(),
                    chunk_index=index,
                )

            merged_results.append(merged_result)

        # STEP 5: LOG FUSION STATISTICS
        vector_only = len(set(vector_map.keys()) - set(bm25_map.keys()))
        bm25_only = len(set(bm25_map.keys()) - set(vector_map.keys()))
        overlap = len(set(vector_map.keys()) & set(bm25_map.keys()))

        logger.debug(
            "fusion_completed",
            fusion_method=fusion_method,
            total_results=len(merged_results),
            vector_only=vector_only,
            bm25_only=bm25_only,
            overlap=overlap,
        )

        return merged_results
