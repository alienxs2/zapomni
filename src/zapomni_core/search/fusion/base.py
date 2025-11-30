"""
Fusion strategy base module for hybrid search.

Provides the abstract protocol for search result fusion strategies.
Fusion strategies combine results from multiple search methods (e.g., vector
and BM25) into a single ranked list using various algorithms.

Supported fusion strategies:
    - RRF (Reciprocal Rank Fusion): Rank-based fusion, parameter k
    - RSF (Relative Score Fusion): Score-based fusion with normalization
    - DBSF (Distribution-Based Score Fusion): 3-sigma normalization

The alpha parameter controls the balance between search methods:
    - alpha = 0.0: 100% BM25 (keyword-only)
    - alpha = 0.5: Equal weight (balanced)
    - alpha = 1.0: 100% vector (semantic-only)

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from typing import Dict, Protocol, Tuple

import structlog

logger = structlog.get_logger(__name__)


class FusionStrategy(Protocol):
    """
    Protocol defining the interface for search result fusion strategies.

    Fusion strategies combine ranked results from multiple search methods
    (vector search and BM25 keyword search) into a single unified ranking.
    Different strategies use different algorithms to calculate combined scores.

    Implementations:
        - RRFStrategy: Reciprocal Rank Fusion (rank-based)
        - RSFStrategy: Relative Score Fusion (score-based)
        - DBSFStrategy: Distribution-Based Score Fusion (3-sigma normalized)

    Example:
        ```python
        from zapomni_core.search.fusion import RRFStrategy

        strategy = RRFStrategy(k=60)

        # Results from vector search: {chunk_id: (rank, score)}
        vector_results = {
            "chunk_001": (1, 0.95),
            "chunk_002": (2, 0.87),
            "chunk_003": (3, 0.72),
        }

        # Results from BM25 search: {chunk_id: (rank, score)}
        bm25_results = {
            "chunk_002": (1, 0.88),
            "chunk_004": (2, 0.75),
            "chunk_001": (3, 0.65),
        }

        # Fuse results with equal weight
        fused = strategy.fuse(vector_results, bm25_results, alpha=0.5)
        # Returns: {"chunk_001": 0.82, "chunk_002": 0.91, ...}
        ```

    See Also:
        - HybridSearch: Uses fusion strategies for combining search results
        - VectorSearch: Provides vector_results input
        - BM25Search: Provides bm25_results input
    """

    def fuse(
        self,
        vector_results: Dict[str, Tuple[int, float]],
        bm25_results: Dict[str, Tuple[int, float]],
        alpha: float = 0.5,
    ) -> Dict[str, float]:
        """
        Fuse results from vector and BM25 search into combined scores.

        Combines ranked results from both search methods using the implemented
        fusion algorithm. Results are merged, with documents appearing in both
        result sets receiving contributions from both sources.

        Args:
            vector_results: Results from vector (semantic) search.
                Dictionary mapping chunk_id to (rank, score) tuple where:
                - rank: 1-based position in results (1 = best match)
                - score: Similarity score from vector search (0.0 to 1.0)
            bm25_results: Results from BM25 (keyword) search.
                Dictionary mapping chunk_id to (rank, score) tuple where:
                - rank: 1-based position in results (1 = best match)
                - score: BM25 score (normalized to 0.0 to 1.0)
            alpha: Weight for vector search results (default: 0.5).
                - 0.0: Use only BM25 results (pure keyword search)
                - 0.5: Equal weight to both methods (balanced hybrid)
                - 1.0: Use only vector results (pure semantic search)
                The BM25 weight is calculated as (1 - alpha).

        Returns:
            Dictionary mapping chunk_id to combined fusion score.
            Higher scores indicate more relevant documents.
            Score range depends on the fusion algorithm:
            - RRF: Scores are reciprocal rank sums
            - RSF: Scores are weighted averages (0.0 to 1.0)
            - DBSF: Scores are normalized weighted averages

        Raises:
            ValueError: If alpha is not in range [0.0, 1.0]

        Note:
            Documents appearing in only one result set will still be
            included in the fused results, with their single-source
            score weighted by the appropriate alpha value.

        Example:
            ```python
            # Equal weight fusion
            fused = strategy.fuse(vector_results, bm25_results, alpha=0.5)

            # Semantic-focused (70% vector, 30% BM25)
            fused = strategy.fuse(vector_results, bm25_results, alpha=0.7)

            # Keyword-focused (30% vector, 70% BM25)
            fused = strategy.fuse(vector_results, bm25_results, alpha=0.3)

            # Sort by combined score
            ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
            for chunk_id, score in ranked[:10]:
                print(f"{chunk_id}: {score:.4f}")
            ```
        """
        ...
