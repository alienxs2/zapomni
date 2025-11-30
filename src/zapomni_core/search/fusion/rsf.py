"""
RSFusion - Relative Score Fusion for hybrid search result merging.

Provides score-based fusion using min-max normalization to combine results
from different search sources (vector and BM25) with configurable weighting.

RSF Algorithm:
    1. Normalize scores from each source to [0, 1] using min-max normalization
    2. Combine using weighted sum: score = alpha * norm_vector + (1-alpha) * norm_bm25
    3. Handle edge cases for missing results and zero-range scores

Comparison with RRF:
    - RSF uses actual scores, preserving relative importance within each source
    - RRF uses only ranks, treating all score differences equally
    - RSF is more sensitive to score distributions
    - RSF requires calibrated scores from sources

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from typing import Dict, Tuple

import structlog

logger = structlog.get_logger(__name__)


class RSFusion:
    """
    Relative Score Fusion for combining search results from multiple sources.

    Uses min-max normalization to scale scores to [0, 1] range, then combines
    them using a weighted sum. This preserves the relative importance of
    scores within each source while allowing configurable weighting between
    sources.

    Normalization Formula:
        norm_score = (score - min_score) / (max_score - min_score)

    Fusion Formula:
        combined_score = alpha * norm_vector + (1 - alpha) * norm_bm25

    Where:
        - alpha = 1.0: Pure vector search
        - alpha = 0.5: Balanced hybrid
        - alpha = 0.0: Pure BM25 search

    Attributes:
        None (stateless fusion algorithm)

    Example:
        ```python
        from zapomni_core.search.fusion import RSFusion

        rsf = RSFusion()

        # Results from vector search: chunk_id -> (rank, score)
        vector_results = {
            "chunk_1": (1, 0.95),
            "chunk_2": (2, 0.85),
            "chunk_3": (3, 0.70),
        }

        # Results from BM25 search: chunk_id -> (rank, score)
        bm25_results = {
            "chunk_1": (2, 12.5),
            "chunk_2": (1, 15.0),
            "chunk_4": (3, 8.0),
        }

        # Fuse with balanced weighting
        fused = rsf.fuse(vector_results, bm25_results, alpha=0.5)

        # Results: chunk_id -> combined_score
        # chunk_1: 0.5 * 1.0 + 0.5 * 0.643 = 0.821
        # chunk_2: 0.5 * 0.6 + 0.5 * 1.0 = 0.8
        # chunk_3: 0.5 * 0.0 + 0.5 * 0.0 = 0.0 (only in vector)
        # chunk_4: 0.5 * 0.0 + 0.5 * 0.0 = 0.0 (only in bm25)

        for chunk_id, score in sorted(fused.items(), key=lambda x: x[1], reverse=True):
            print(f"{chunk_id}: {score:.3f}")
        ```
    """

    def fuse(
        self,
        vector_results: Dict[str, Tuple[int, float]],
        bm25_results: Dict[str, Tuple[int, float]],
        alpha: float = 0.5,
    ) -> Dict[str, float]:
        """
        Fuse results using Relative Score Fusion.

        Uses min-max normalization to scale scores to [0, 1], then combines
        them using weighted sum based on alpha parameter.

        Algorithm:
            1. Normalize vector scores to [0, 1] using min-max
            2. Normalize BM25 scores to [0, 1] using min-max
            3. For each unique chunk_id:
               - Get normalized vector score (0.0 if not in vector results)
               - Get normalized BM25 score (0.0 if not in BM25 results)
               - Compute: combined = alpha * norm_vector + (1-alpha) * norm_bm25
            4. Return combined scores for all chunks

        Args:
            vector_results: Results from vector search.
                Dictionary mapping chunk_id to (rank, score) tuple.
                Rank is 1-indexed position in result list.
                Score is the raw similarity score from vector search.
            bm25_results: Results from BM25 search.
                Dictionary mapping chunk_id to (rank, score) tuple.
                Rank is 1-indexed position in result list.
                Score is the raw BM25 score.
            alpha: Weighting factor for vector vs BM25 (default: 0.5).
                - alpha = 1.0: Pure vector search (semantic)
                - alpha = 0.5: Balanced hybrid (recommended)
                - alpha = 0.0: Pure BM25 search (lexical)

        Returns:
            Dictionary mapping chunk_id to combined fusion score.
            Scores are in [0, 1] range, higher is better.
            Empty dict if both inputs are empty.

        Edge Cases:
            - Empty results from one source: Uses 0.0 for missing chunks
            - Empty results from both sources: Returns empty dict
            - All scores identical in a source: Uses 1.0 for all chunks
            - Chunk in only one source: Uses 0.0 for missing source

        Example:
            ```python
            rsf = RSFusion()

            # Vector search results
            vector = {
                "doc_1": (1, 0.95),  # rank 1, score 0.95
                "doc_2": (2, 0.80),  # rank 2, score 0.80
            }

            # BM25 search results
            bm25 = {
                "doc_1": (2, 10.0),  # rank 2, score 10.0
                "doc_3": (1, 15.0),  # rank 1, score 15.0
            }

            # Balanced fusion
            fused = rsf.fuse(vector, bm25, alpha=0.5)
            # doc_1: appears in both sources
            # doc_2: only in vector (bm25 contribution = 0)
            # doc_3: only in bm25 (vector contribution = 0)

            # Semantic-focused fusion
            fused = rsf.fuse(vector, bm25, alpha=0.8)
            # Vector scores weighted 80%, BM25 weighted 20%
            ```
        """
        logger.debug(
            "rsf_fuse_start",
            vector_count=len(vector_results),
            bm25_count=len(bm25_results),
            alpha=alpha,
        )

        # Handle edge case: both empty
        if not vector_results and not bm25_results:
            logger.debug("rsf_fuse_empty_inputs")
            return {}

        # Normalize scores from each source
        norm_vector = self._normalize(vector_results)
        norm_bm25 = self._normalize(bm25_results)

        # Get all unique chunk IDs from both sources
        all_chunks = set(norm_vector.keys()) | set(norm_bm25.keys())

        # Compute combined scores
        combined_scores: Dict[str, float] = {}

        for chunk_id in all_chunks:
            # Get normalized score from each source (0.0 if missing)
            vector_score = norm_vector.get(chunk_id, 0.0)
            bm25_score = norm_bm25.get(chunk_id, 0.0)

            # Weighted combination
            combined = alpha * vector_score + (1.0 - alpha) * bm25_score
            combined_scores[chunk_id] = combined

        logger.debug(
            "rsf_fuse_complete",
            unique_chunks=len(all_chunks),
            vector_only=len(set(norm_vector.keys()) - set(norm_bm25.keys())),
            bm25_only=len(set(norm_bm25.keys()) - set(norm_vector.keys())),
            overlap=len(set(norm_vector.keys()) & set(norm_bm25.keys())),
        )

        return combined_scores

    def _normalize(
        self,
        results: Dict[str, Tuple[int, float]],
    ) -> Dict[str, float]:
        """
        Min-max normalize scores to [0, 1] range.

        Normalization Formula:
            norm_score = (score - min_score) / (max_score - min_score)

        This transforms all scores to the [0, 1] range while preserving
        the relative ordering and proportional differences between scores.

        Args:
            results: Dictionary mapping chunk_id to (rank, score) tuple.
                The rank value is ignored; only scores are used for normalization.

        Returns:
            Dictionary mapping chunk_id to normalized score in [0, 1] range.
            Returns empty dict if input is empty.

        Edge Cases:
            - Empty input: Returns empty dict
            - Single result: Returns {chunk_id: 1.0}
            - All scores identical: Returns {chunk_id: 1.0} for all chunks
              (avoids division by zero)

        Example:
            ```python
            # Normal case with score range
            results = {
                "doc_1": (1, 100.0),
                "doc_2": (2, 75.0),
                "doc_3": (3, 50.0),
            }
            normalized = rsf._normalize(results)
            # normalized = {"doc_1": 1.0, "doc_2": 0.5, "doc_3": 0.0}

            # Edge case: all same scores
            results = {
                "doc_1": (1, 50.0),
                "doc_2": (2, 50.0),
            }
            normalized = rsf._normalize(results)
            # normalized = {"doc_1": 1.0, "doc_2": 1.0}
            ```
        """
        # Handle empty input
        if not results:
            return {}

        # Extract scores from (rank, score) tuples
        scores = [score for _, (_, score) in results.items()]

        # Calculate min and max scores
        min_score = min(scores)
        max_score = max(scores)

        # Calculate score range
        score_range = max_score - min_score

        # Handle edge case: all scores are the same (avoid division by zero)
        if score_range == 0:
            # All scores are equal, assign 1.0 to all
            logger.debug(
                "rsf_normalize_uniform_scores",
                count=len(results),
                score=min_score,
            )
            return {chunk_id: 1.0 for chunk_id in results.keys()}

        # Apply min-max normalization
        normalized: Dict[str, float] = {}
        for chunk_id, (_, score) in results.items():
            normalized[chunk_id] = (score - min_score) / score_range

        logger.debug(
            "rsf_normalize_complete",
            count=len(normalized),
            min_score=min_score,
            max_score=max_score,
            score_range=score_range,
        )

        return normalized


# Module-level instance for convenience function
_default_rsf = RSFusion()


def fuse_rsf(
    vector_results: Dict[str, Tuple[int, float]],
    bm25_results: Dict[str, Tuple[int, float]],
    alpha: float = 0.5,
) -> Dict[str, float]:
    """
    Convenience function for Relative Score Fusion.

    Fuses results using min-max normalization with weighted combination.
    Uses a module-level RSFusion instance for stateless operation.

    Args:
        vector_results: Results from vector search.
            Dictionary mapping chunk_id to (rank, score) tuple.
        bm25_results: Results from BM25 search.
            Dictionary mapping chunk_id to (rank, score) tuple.
        alpha: Weighting factor for vector vs BM25 (default: 0.5).
            - alpha = 1.0: Pure vector search (semantic)
            - alpha = 0.5: Balanced hybrid (recommended)
            - alpha = 0.0: Pure BM25 search (lexical)

    Returns:
        Dictionary mapping chunk_id to combined fusion score.
        Scores are in [0, 1] range, higher is better.

    Example:
        ```python
        from zapomni_core.search.fusion import fuse_rsf

        # Functional interface
        fused = fuse_rsf(vector_results, bm25_results, alpha=0.5)

        # Sort by score
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        for chunk_id, score in ranked[:10]:
            print(f"{chunk_id}: {score:.4f}")
        ```
    """
    return _default_rsf.fuse(vector_results, bm25_results, alpha)
