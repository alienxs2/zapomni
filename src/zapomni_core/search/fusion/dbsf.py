"""
DBSFusion - Distribution-Based Score Fusion for hybrid search.

Implements DBSF (Distribution-Based Score Fusion) algorithm that uses
the 3-sigma (three-sigma) rule for score normalization. This provides
statistically robust normalization that handles outliers and varying
score distributions across different retrieval methods.

The 3-Sigma Rule:
    For a normal distribution, approximately:
    - 68% of values lie within 1 standard deviation of the mean
    - 95% of values lie within 2 standard deviations
    - 99.7% of values lie within 3 standard deviations

    DBSF uses this property to normalize scores to [0, 1] by mapping
    the range [mean - 3*std, mean + 3*std] to [0, 1]. This captures
    99.7% of expected score values and handles outliers gracefully.

Algorithm:
    1. Calculate mean (mu) and standard deviation (sigma) of scores
    2. Define normalization range: [mu - 3*sigma, mu + 3*sigma]
    3. Normalize each score: norm = (score - (mu - 3*sigma)) / (6 * sigma)
    4. Clamp normalized values to [0, 1] to handle outliers
    5. Combine: final_score = alpha * norm_vector + (1 - alpha) * norm_bm25

References:
    - Kurland, O., & Lee, L. (2006). "Corpus structure, language models,
      and ad hoc information retrieval"
    - Cormack, G. V., et al. (2009). "Reciprocal Rank Fusion outperforms
      Condorcet and individual Rank Learning Methods"

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

import math
from typing import Dict, Optional, Tuple

import structlog

logger = structlog.get_logger(__name__)


class DBSFusion:
    """
    Distribution-Based Score Fusion for combining vector and BM25 search results.

    Uses the 3-sigma rule for score normalization, which provides statistically
    robust handling of different score distributions. This is particularly useful
    when combining scores from different retrieval systems that may have vastly
    different score scales and distributions.

    Key Properties:
        - Statistically grounded normalization using 3-sigma rule
        - Handles outliers by clamping to [0, 1]
        - Optional pre-defined score ranges for consistent normalization
        - Graceful degradation for edge cases (empty results, single score)

    Attributes:
        vector_range: Optional expected score range for vector search (min, max)
        bm25_range: Optional expected score range for BM25 search (min, max)

    Example:
        ```python
        from zapomni_core.search.fusion import DBSFusion

        # Create fusion instance
        fusion = DBSFusion()

        # Vector search results: {doc_id: (rank, score)}
        vector_results = {
            "doc1": (1, 0.95),
            "doc2": (2, 0.80),
            "doc3": (3, 0.65),
        }

        # BM25 search results: {doc_id: (rank, score)}
        bm25_results = {
            "doc1": (2, 12.5),
            "doc2": (1, 15.8),
            "doc4": (3, 8.2),
        }

        # Fuse results with 60% weight on vector scores
        fused = fusion.fuse(vector_results, bm25_results, alpha=0.6)

        # Result: {doc_id: fused_score}
        for doc_id, score in sorted(fused.items(), key=lambda x: x[1], reverse=True):
            print(f"{doc_id}: {score:.4f}")
        ```
    """

    def __init__(
        self,
        vector_range: Optional[Tuple[float, float]] = None,
        bm25_range: Optional[Tuple[float, float]] = None,
    ) -> None:
        """
        Initialize DBSF with optional score ranges.

        If ranges are not provided, they are computed from the data using
        the 3-sigma rule. Pre-defining ranges can provide more consistent
        normalization across different queries and result sets.

        Args:
            vector_range: Expected score range for vector search as (min, max).
                If provided, scores are normalized using this range instead of
                computing from data. Typical vector similarity scores are in [0, 1].
            bm25_range: Expected score range for BM25 search as (min, max).
                If provided, scores are normalized using this range. BM25 scores
                typically vary widely based on document collection.

        Example:
            ```python
            # Use automatic 3-sigma normalization
            fusion = DBSFusion()

            # Use predefined ranges for consistent normalization
            fusion = DBSFusion(
                vector_range=(0.0, 1.0),  # Typical vector similarity range
                bm25_range=(0.0, 50.0),   # Estimated BM25 range
            )
            ```
        """
        self.vector_range = vector_range
        self.bm25_range = bm25_range

        logger.info(
            "dbsf_initialized",
            vector_range=vector_range,
            bm25_range=bm25_range,
        )

    def _normalize_3sigma(
        self,
        results: Dict[str, Tuple[int, float]],
        default_range: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, float]:
        """
        Normalize scores using the 3-sigma rule.

        The 3-sigma normalization works as follows:
        1. Extract all scores from results
        2. Calculate mean (mu) and standard deviation (sigma)
        3. Define normalization range: [mu - 3*sigma, mu + 3*sigma]
        4. Normalize: norm = (score - lower_bound) / (upper_bound - lower_bound)
        5. Clamp to [0, 1] to handle outliers beyond 3-sigma

        This method captures 99.7% of values within [0, 1] for normally
        distributed scores, while gracefully handling outliers.

        Args:
            results: Dictionary mapping document IDs to (rank, score) tuples.
                The rank is ignored; only scores are used for normalization.
            default_range: Optional pre-defined range (min, max) to use instead
                of computing from data. If provided, bypasses 3-sigma calculation.

        Returns:
            Dictionary mapping document IDs to normalized scores in [0, 1].
            Empty dict is returned for empty input.

        Edge Cases:
            - Empty results: Returns empty dict
            - Single result: Returns {doc_id: 1.0}
            - All same scores (std=0): Returns {doc_id: 1.0} for all
            - Outliers: Clamped to [0, 1]

        Example:
            ```python
            results = {
                "doc1": (1, 15.0),
                "doc2": (2, 12.0),
                "doc3": (3, 8.0),
            }
            normalized = self._normalize_3sigma(results)
            # Returns normalized scores in [0, 1]
            ```
        """
        # Handle empty results
        if not results:
            logger.debug("normalize_3sigma_empty_results")
            return {}

        # Extract scores from results
        scores = [score for _, (_, score) in results.items()]

        # Handle single result case
        if len(scores) == 1:
            doc_id = next(iter(results.keys()))
            logger.debug("normalize_3sigma_single_result", doc_id=doc_id)
            return {doc_id: 1.0}

        # Use default range if provided
        if default_range is not None:
            lower_bound, upper_bound = default_range
            range_size = upper_bound - lower_bound

            # Handle zero range (invalid configuration)
            if range_size <= 0:
                logger.warning(
                    "normalize_3sigma_invalid_range",
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                )
                return {doc_id: 1.0 for doc_id in results.keys()}

            normalized: Dict[str, float] = {}
            for doc_id, (_, score) in results.items():
                norm_score = (score - lower_bound) / range_size
                # Clamp to [0, 1]
                normalized[doc_id] = max(0.0, min(1.0, norm_score))

            logger.debug(
                "normalize_3sigma_with_range",
                num_results=len(normalized),
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )
            return normalized

        # Calculate mean
        mean = sum(scores) / len(scores)

        # Calculate standard deviation
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        std = math.sqrt(variance)

        # Handle zero standard deviation (all same scores)
        if std == 0:
            logger.debug(
                "normalize_3sigma_zero_std",
                num_results=len(results),
                constant_score=mean,
            )
            return {doc_id: 1.0 for doc_id in results.keys()}

        # Calculate 3-sigma bounds
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std
        range_size = 6 * std  # upper_bound - lower_bound = 6 * std

        # Normalize scores
        normalized = {}
        for doc_id, (_, score) in results.items():
            norm_score = (score - lower_bound) / range_size
            # Clamp to [0, 1] to handle outliers
            normalized[doc_id] = max(0.0, min(1.0, norm_score))

        logger.debug(
            "normalize_3sigma_completed",
            num_results=len(normalized),
            mean=mean,
            std=std,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

        return normalized

    def fuse(
        self,
        vector_results: Dict[str, Tuple[int, float]],
        bm25_results: Dict[str, Tuple[int, float]],
        alpha: float = 0.5,
    ) -> Dict[str, float]:
        """
        Fuse results using Distribution-Based Score Fusion.

        Combines vector search and BM25 search results using weighted sum
        of normalized scores. Each score set is normalized using the 3-sigma
        rule to ensure comparable scales.

        Algorithm:
            1. Normalize vector scores using 3-sigma (or predefined range)
            2. Normalize BM25 scores using 3-sigma (or predefined range)
            3. Collect all unique document IDs from both result sets
            4. For each document, compute:
               fused_score = alpha * norm_vector + (1 - alpha) * norm_bm25
               where missing scores default to 0.0

        Args:
            vector_results: Vector search results as {doc_id: (rank, score)}.
                Higher scores indicate better matches (similarity scores).
            bm25_results: BM25 search results as {doc_id: (rank, score)}.
                Higher scores indicate better matches (BM25 relevance).
            alpha: Weight for vector scores in [0, 1]. Default: 0.5.
                - alpha=1.0: Use only vector scores
                - alpha=0.0: Use only BM25 scores
                - alpha=0.5: Equal weighting (default)

        Returns:
            Dictionary mapping document IDs to fused scores in [0, 1].
            Sorted by score descending when iterating.

        Raises:
            ValueError: If alpha is not in [0, 1]

        Example:
            ```python
            fusion = DBSFusion()

            vector_results = {
                "doc1": (1, 0.95),  # Highest vector similarity
                "doc2": (2, 0.80),
                "doc3": (3, 0.65),
            }

            bm25_results = {
                "doc2": (1, 15.8),  # Highest BM25 score
                "doc1": (2, 12.5),
                "doc4": (3, 8.2),   # Only in BM25 results
            }

            # Equal weighting
            fused = fusion.fuse(vector_results, bm25_results, alpha=0.5)

            # Prioritize semantic similarity
            fused = fusion.fuse(vector_results, bm25_results, alpha=0.7)

            # Prioritize keyword matching
            fused = fusion.fuse(vector_results, bm25_results, alpha=0.3)
            ```
        """
        # Validate alpha
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        # Handle both empty case
        if not vector_results and not bm25_results:
            logger.debug("fuse_both_empty")
            return {}

        # Normalize vector scores
        norm_vector = self._normalize_3sigma(vector_results, self.vector_range)

        # Normalize BM25 scores
        norm_bm25 = self._normalize_3sigma(bm25_results, self.bm25_range)

        # Collect all document IDs
        all_doc_ids = set(norm_vector.keys()) | set(norm_bm25.keys())

        # Compute fused scores
        fused: Dict[str, float] = {}
        for doc_id in all_doc_ids:
            vector_score = norm_vector.get(doc_id, 0.0)
            bm25_score = norm_bm25.get(doc_id, 0.0)

            # Weighted combination
            fused[doc_id] = alpha * vector_score + (1 - alpha) * bm25_score

        logger.info(
            "dbsf_fusion_completed",
            vector_count=len(vector_results),
            bm25_count=len(bm25_results),
            fused_count=len(fused),
            alpha=alpha,
        )

        return fused


def fuse_dbsf(
    vector_results: Dict[str, Tuple[int, float]],
    bm25_results: Dict[str, Tuple[int, float]],
    alpha: float = 0.5,
    vector_range: Optional[Tuple[float, float]] = None,
    bm25_range: Optional[Tuple[float, float]] = None,
) -> Dict[str, float]:
    """
    Convenience function for one-shot DBSF fusion.

    Creates a DBSFusion instance and performs fusion in a single call.
    For repeated fusion with the same configuration, prefer creating
    a DBSFusion instance and reusing it.

    Args:
        vector_results: Vector search results as {doc_id: (rank, score)}
        bm25_results: BM25 search results as {doc_id: (rank, score)}
        alpha: Weight for vector scores in [0, 1]. Default: 0.5
        vector_range: Optional expected score range for vector search
        bm25_range: Optional expected score range for BM25 search

    Returns:
        Dictionary mapping document IDs to fused scores in [0, 1]

    Example:
        ```python
        from zapomni_core.search.fusion import fuse_dbsf

        fused = fuse_dbsf(
            vector_results={"doc1": (1, 0.95), "doc2": (2, 0.80)},
            bm25_results={"doc1": (2, 12.5), "doc3": (1, 15.8)},
            alpha=0.6,
        )
        ```
    """
    fusion = DBSFusion(vector_range=vector_range, bm25_range=bm25_range)
    return fusion.fuse(vector_results, bm25_results, alpha=alpha)
