"""
RRF (Reciprocal Rank Fusion) - Rank-based fusion for hybrid search.

Implements Reciprocal Rank Fusion algorithm for combining results from
multiple ranking sources (vector search and BM25) into a unified ranking.

Algorithm:
    RRF_score(d) = sum(weight_i / (k + rank_i(d)))

    Where:
    - k is a smoothing constant (default: 60)
    - alpha controls weight between vector (alpha) and BM25 (1-alpha)
    - rank_i(d) is the rank of document d in ranking source i

Reference:
    Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009).
    Reciprocal rank fusion outperforms condorcet and individual rank
    learning methods. In SIGIR.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from typing import Dict, Optional, Set, Tuple

import structlog

from zapomni_core.exceptions import ValidationError

logger = structlog.get_logger(__name__)


class RRFusion:
    """
    Reciprocal Rank Fusion (RRF) for combining search results.

    RRF is a simple yet effective method for combining rankings from
    multiple sources. It uses reciprocal ranks to give higher weight
    to top-ranked documents while dampening the influence of low-ranked
    documents.

    The formula for each document d is:
        score(d) = alpha * (1/(k + rank_vector)) + (1-alpha) * (1/(k + rank_bm25))

    Attributes:
        k: Smoothing constant (default: 60). Higher values reduce the
           difference between adjacent ranks.

    Example:
        ```python
        from zapomni_core.search.fusion.rrf import RRFusion

        # Initialize RRF with default k=60
        rrf = RRFusion()

        # Results from vector search: {chunk_id: (rank, score)}
        vector_results = {
            "chunk_1": (1, 0.95),
            "chunk_2": (2, 0.87),
            "chunk_3": (3, 0.75),
        }

        # Results from BM25 search: {chunk_id: (rank, score)}
        bm25_results = {
            "chunk_1": (2, 0.82),
            "chunk_3": (1, 0.91),
            "chunk_4": (3, 0.65),
        }

        # Fuse with equal weighting (alpha=0.5)
        fused = rrf.fuse(vector_results, bm25_results, alpha=0.5)
        # Returns: {"chunk_1": 0.0163, "chunk_3": 0.0161, ...}

        # Prefer vector results (alpha=0.7)
        fused = rrf.fuse(vector_results, bm25_results, alpha=0.7)
        ```
    """

    def __init__(self, k: int = 60) -> None:
        """
        Initialize RRFusion with smoothing constant k.

        Args:
            k: Smoothing constant for rank fusion. Must be > 0.
               Default: 60 (standard value from original RRF paper).

               Effects of k:
               - Lower k (e.g., 10): Emphasizes top ranks more strongly
               - Higher k (e.g., 100): More uniform weighting across ranks

        Raises:
            ValidationError: If k is not a positive integer

        Example:
            ```python
            # Default k=60 (recommended)
            rrf = RRFusion()

            # Emphasize top results more
            rrf = RRFusion(k=20)

            # More uniform weighting
            rrf = RRFusion(k=100)
            ```
        """
        # Validate k parameter
        if not isinstance(k, int):
            raise ValidationError(
                message=f"k must be an integer, got {type(k).__name__}",
                error_code="VAL_002",
                details={"k": k, "type": type(k).__name__},
            )

        if k <= 0:
            raise ValidationError(
                message=f"k must be > 0, got {k}",
                error_code="VAL_003",
                details={"k": k},
            )

        self.k = k

        logger.info("rrf_fusion_initialized", k=k)

    def fuse(
        self,
        vector_results: Dict[str, Tuple[int, float]],
        bm25_results: Dict[str, Tuple[int, float]],
        alpha: float = 0.5,
    ) -> Dict[str, float]:
        """
        Fuse results from vector and BM25 search using Reciprocal Rank Fusion.

        Combines rankings from two sources using weighted RRF:
            score(d) = alpha * (1/(k + rank_vector)) + (1-alpha) * (1/(k + rank_bm25))

        For documents appearing in only one source, only that source's
        contribution is used (the other term is 0).

        Args:
            vector_results: Vector search results as {chunk_id: (rank, score)}.
                           Rank is 1-indexed (1 = best match).
            bm25_results: BM25 search results as {chunk_id: (rank, score)}.
                         Rank is 1-indexed (1 = best match).
            alpha: Weight for vector results. Range: [0.0, 1.0].
                   - alpha=0.5: Equal weighting (default)
                   - alpha=0.7: Prefer vector/semantic results
                   - alpha=0.3: Prefer BM25/keyword results

        Returns:
            Dictionary mapping chunk_id to fused RRF score.
            Higher scores indicate better combined ranking.
            Scores are in range (0, 1/k] theoretically, but typically
            much smaller due to the k smoothing constant.

        Raises:
            ValidationError: If alpha is not in [0.0, 1.0] range

        Performance:
            - O(n + m) where n = len(vector_results), m = len(bm25_results)
            - Handles up to millions of results efficiently

        Example:
            ```python
            rrf = RRFusion(k=60)

            # Both sources have chunk_1 and chunk_3
            vector_results = {
                "chunk_1": (1, 0.95),  # Rank 1 in vector
                "chunk_2": (2, 0.87),  # Only in vector
                "chunk_3": (3, 0.75),
            }
            bm25_results = {
                "chunk_1": (2, 0.82),  # Rank 2 in BM25
                "chunk_3": (1, 0.91),  # Rank 1 in BM25
                "chunk_4": (3, 0.65),  # Only in BM25
            }

            # Equal weighting
            fused = rrf.fuse(vector_results, bm25_results, alpha=0.5)

            # chunk_1 score: 0.5*(1/61) + 0.5*(1/62) = 0.00819 + 0.00806 = 0.01625
            # chunk_3 score: 0.5*(1/63) + 0.5*(1/61) = 0.00793 + 0.00819 = 0.01612
            # chunk_2 score: 0.5*(1/62) + 0 = 0.00806
            # chunk_4 score: 0 + 0.5*(1/63) = 0.00793

            # Sort by score to get final ranking
            ranking = sorted(fused.items(), key=lambda x: x[1], reverse=True)
            ```
        """
        # Validate alpha parameter
        if not isinstance(alpha, (int, float)):
            raise ValidationError(
                message=f"alpha must be a number, got {type(alpha).__name__}",
                error_code="VAL_002",
                details={"alpha": alpha, "type": type(alpha).__name__},
            )

        if alpha < 0.0 or alpha > 1.0:
            raise ValidationError(
                message=f"alpha must be in [0.0, 1.0], got {alpha}",
                error_code="VAL_003",
                details={"alpha": alpha, "min": 0.0, "max": 1.0},
            )

        # Handle None inputs gracefully
        if vector_results is None:
            vector_results = {}
        if bm25_results is None:
            bm25_results = {}

        # Handle empty results
        if not vector_results and not bm25_results:
            logger.debug("rrf_fuse_empty_inputs")
            return {}

        # Get all unique chunk IDs from both sources
        all_chunk_ids: Set[str] = set(vector_results.keys()) | set(bm25_results.keys())

        logger.debug(
            "rrf_fuse_start",
            vector_count=len(vector_results),
            bm25_count=len(bm25_results),
            total_unique=len(all_chunk_ids),
            alpha=alpha,
            k=self.k,
        )

        # Calculate RRF scores for each chunk
        fused_scores: Dict[str, float] = {}

        for chunk_id in all_chunk_ids:
            score = 0.0

            # Vector contribution
            if chunk_id in vector_results:
                vector_rank, _ = vector_results[chunk_id]
                vector_contribution = alpha / (self.k + vector_rank)
                score += vector_contribution

            # BM25 contribution
            if chunk_id in bm25_results:
                bm25_rank, _ = bm25_results[chunk_id]
                bm25_contribution = (1 - alpha) / (self.k + bm25_rank)
                score += bm25_contribution

            fused_scores[chunk_id] = score

        logger.info(
            "rrf_fuse_completed",
            result_count=len(fused_scores),
            vector_only=len(set(vector_results.keys()) - set(bm25_results.keys())),
            bm25_only=len(set(bm25_results.keys()) - set(vector_results.keys())),
            overlap=len(set(vector_results.keys()) & set(bm25_results.keys())),
        )

        return fused_scores

    def fuse_with_ranks(
        self,
        vector_results: Dict[str, Tuple[int, float]],
        bm25_results: Dict[str, Tuple[int, float]],
        alpha: float = 0.5,
    ) -> Dict[str, Tuple[float, Optional[int], Optional[int]]]:
        """
        Fuse results and return detailed information including original ranks.

        Similar to fuse(), but returns additional context about each result
        including the original ranks from both sources.

        Args:
            vector_results: Vector search results as {chunk_id: (rank, score)}
            bm25_results: BM25 search results as {chunk_id: (rank, score)}
            alpha: Weight for vector results. Range: [0.0, 1.0]

        Returns:
            Dictionary mapping chunk_id to tuple of:
                (fused_score, vector_rank, bm25_rank)
            Where ranks are None if the chunk wasn't in that source.

        Example:
            ```python
            rrf = RRFusion()

            vector_results = {"chunk_1": (1, 0.95), "chunk_2": (2, 0.87)}
            bm25_results = {"chunk_1": (2, 0.82), "chunk_3": (1, 0.91)}

            detailed = rrf.fuse_with_ranks(vector_results, bm25_results)
            # Returns:
            # {
            #     "chunk_1": (0.01625, 1, 2),   # In both
            #     "chunk_2": (0.00806, 2, None), # Vector only
            #     "chunk_3": (0.00819, None, 1), # BM25 only
            # }
            ```
        """
        # Use main fuse method for score calculation
        fused_scores = self.fuse(vector_results, bm25_results, alpha)

        # Handle None inputs
        if vector_results is None:
            vector_results = {}
        if bm25_results is None:
            bm25_results = {}

        # Build detailed results with ranks
        detailed_results: Dict[str, Tuple[float, Optional[int], Optional[int]]] = {}

        for chunk_id, score in fused_scores.items():
            vector_rank: Optional[int] = None
            bm25_rank: Optional[int] = None

            if chunk_id in vector_results:
                vector_rank = vector_results[chunk_id][0]

            if chunk_id in bm25_results:
                bm25_rank = bm25_results[chunk_id][0]

            detailed_results[chunk_id] = (score, vector_rank, bm25_rank)

        return detailed_results

    def get_sorted_results(
        self,
        vector_results: Dict[str, Tuple[int, float]],
        bm25_results: Dict[str, Tuple[int, float]],
        alpha: float = 0.5,
        limit: Optional[int] = None,
    ) -> list[Tuple[str, float]]:
        """
        Fuse results and return sorted list of (chunk_id, score) tuples.

        Convenience method that fuses and sorts results in one call.

        Args:
            vector_results: Vector search results as {chunk_id: (rank, score)}
            bm25_results: BM25 search results as {chunk_id: (rank, score)}
            alpha: Weight for vector results. Range: [0.0, 1.0]
            limit: Maximum number of results to return. None for all.

        Returns:
            List of (chunk_id, fused_score) tuples, sorted by score descending.

        Raises:
            ValidationError: If limit is provided but not positive

        Example:
            ```python
            rrf = RRFusion()

            vector_results = {"chunk_1": (1, 0.95), "chunk_2": (2, 0.87)}
            bm25_results = {"chunk_1": (2, 0.82), "chunk_3": (1, 0.91)}

            # Get top 5 results
            top_results = rrf.get_sorted_results(
                vector_results, bm25_results, alpha=0.5, limit=5
            )
            # Returns: [("chunk_1", 0.01625), ("chunk_3", 0.00819), ...]
            ```
        """
        # Validate limit if provided
        if limit is not None:
            if not isinstance(limit, int):
                raise ValidationError(
                    message=f"limit must be an integer, got {type(limit).__name__}",
                    error_code="VAL_002",
                    details={"limit": limit, "type": type(limit).__name__},
                )
            if limit <= 0:
                raise ValidationError(
                    message=f"limit must be > 0, got {limit}",
                    error_code="VAL_003",
                    details={"limit": limit},
                )

        # Fuse results
        fused_scores = self.fuse(vector_results, bm25_results, alpha)

        # Sort by score descending
        sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

        # Apply limit if specified
        if limit is not None:
            sorted_results = sorted_results[:limit]

        return sorted_results
