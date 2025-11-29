"""
CrossEncoderReranker - Semantic reranking using cross-encoder models.

Improves search result relevance using sentence-transformers cross-encoder models.
Provides query-aware relevance scoring and batch processing with intelligent fallback
to original scores if model loading fails.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

import asyncio
from typing import Any, List, Optional, Union

import structlog
from sentence_transformers import CrossEncoder

from zapomni_core.exceptions import SearchError, ValidationError
from zapomni_db.models import SearchResult

logger = structlog.get_logger()


class CrossEncoderReranker:
    """
    Semantic reranking of search results using cross-encoder models.

    Uses sentence-transformers cross-encoder for query-aware relevance scoring.
    Cross-encoders provide superior relevance assessment compared to bi-encoders
    (embeddings) by directly modeling query-document pairs.

    Attributes:
        model_name: Cross-encoder model name (default: "cross-encoder/ms-marco-MiniLM-L-6-v2")
        model: Lazy-loaded CrossEncoder instance
        fallback_enabled: Whether to fallback to original scores if reranking fails

    Features:
        - Query-aware relevance scoring
        - Batch processing support
        - Lazy model loading (loaded on first use)
        - Intelligent fallback to original scores
        - Comprehensive error handling
        - Structured logging

    Example:
        ```python
        # Initialize reranker
        reranker = CrossEncoderReranker()

        # Rerank search results
        results = [
            SearchResult(
                memory_id="1",
                content="Python programming guide",
                relevance_score=0.8,
                metadata={}
            ),
            SearchResult(
                memory_id="2",
                content="Java tutorial",
                relevance_score=0.6,
                metadata={}
            ),
        ]

        reranked = await reranker.rerank("Python", results, top_k=10)
        print(f"Top result: {reranked[0].memory_id}")  # Should be "1" with higher score

        # Score individual query-content pairs
        score = await reranker.score("Python", "Python is a programming language")
        print(f"Score: {score}")  # Returns float between 0 and 1
        ```
    """

    # Default cross-encoder model: optimized for MS MARCO dataset
    # - Lightweight (135M parameters)
    # - Fast inference (~100ms per query)
    # - Good accuracy for passage ranking
    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        model_name: Optional[str] = None,
        fallback_enabled: bool = True,
    ) -> None:
        """
        Initialize CrossEncoderReranker with model configuration.

        Args:
            model_name: Cross-encoder model name from sentence-transformers
                       (default: "cross-encoder/ms-marco-MiniLM-L-6-v2")
            fallback_enabled: If True, use original scores if reranking fails
                             (default: True)

        Raises:
            ValidationError: If model_name is invalid

        Example:
            ```python
            # Default model
            reranker = CrossEncoderReranker()

            # Custom model
            reranker = CrossEncoderReranker(
                model_name="cross-encoder/ms-marco-MiniLM-L-12-v2"
            )

            # Without fallback (fail fast)
            reranker = CrossEncoderReranker(fallback_enabled=False)
            ```
        """
        # Validate model name before applying default
        if model_name is not None:
            if not isinstance(model_name, str):
                raise ValidationError(
                    message="model_name must be non-empty string",
                    error_code="VAL_001",
                    details={"model_name": model_name},
                )
            if not model_name or not model_name.strip():
                raise ValidationError(
                    message="model_name must be non-empty string",
                    error_code="VAL_001",
                    details={"model_name": model_name},
                )

        self.model_name = model_name or self.DEFAULT_MODEL
        self.fallback_enabled = fallback_enabled
        self.model: Optional[CrossEncoder] = None

        logger.info(
            "cross_encoder_reranker_initialized",
            model_name=self.model_name,
            fallback_enabled=fallback_enabled,
        )

    def _load_model(self) -> None:
        """
        Lazy load cross-encoder model on first use.

        Downloads model from Hugging Face if not cached locally.
        Uses sentence-transformers for automatic caching.

        Raises:
            SearchError: If model loading fails

        Private method, not exposed in public API.
        """
        if self.model is not None:
            return

        try:
            logger.info("loading_cross_encoder_model", model=self.model_name)
            self.model = CrossEncoder(self.model_name)
            logger.info(
                "cross_encoder_model_loaded",
                model=self.model_name,
                max_length=512,
            )
        except Exception as e:
            raise SearchError(
                message=f"Failed to load cross-encoder model '{self.model_name}': {str(e)}",
                error_code="SEARCH_001",
                details={"model": self.model_name, "error": str(e)},
                original_exception=e,
            )

    async def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Rerank search results using cross-encoder for improved relevance.

        Algorithm:
        1. Validate query and results
        2. Lazy load cross-encoder model
        3. Score each (query, result.content) pair
        4. Sort results by new scores (descending)
        5. Return top_k results or all if top_k is None
        6. On failure, return original results sorted by relevance_score

        Args:
            query: Search query string
            results: List of SearchResult objects to rerank
            top_k: Return top K results (default: None = return all)

        Returns:
            List[SearchResult]: Reranked results sorted by relevance_score (descending)

        Raises:
            ValidationError: If query is empty or results format invalid

        Performance Target:
            - Single query with 10 results: < 200ms (P95)
            - Batch processing: ~50ms per query

        Example:
            ```python
            results = [
                SearchResult(..., content="text1", relevance_score=0.8),
                SearchResult(..., content="text2", relevance_score=0.6),
            ]

            reranked = await reranker.rerank("query", results, top_k=10)
            assert reranked[0].relevance_score >= reranked[1].relevance_score
            ```
        """
        # Validate inputs
        if not query or not query.strip():
            raise ValidationError(
                message="Query cannot be empty",
                error_code="VAL_001",
                details={"query": query},
            )

        if not isinstance(results, list):
            raise ValidationError(
                message=f"Results must be list, got {type(results)}",
                error_code="VAL_002",
                details={"type": str(type(results))},
            )

        # Handle empty results
        if not results:
            logger.debug("reranking_empty_results")
            return []

        # Validate each result
        for i, result in enumerate(results):
            if not isinstance(result, SearchResult):
                raise ValidationError(
                    message=f"Result {i} is not SearchResult instance",
                    error_code="VAL_002",
                    details={"index": i, "type": str(type(result))},
                )
            if not hasattr(result, "content") or not result.content:
                raise ValidationError(
                    message=f"Result {i} missing or empty content",
                    error_code="VAL_001",
                    details={"index": i},
                )

        try:
            # Load model if needed
            self._load_model()

            if self.model is None:
                raise SearchError(
                    message="Model failed to load",
                    error_code="SEARCH_001",
                )

            # Score all results
            query_content_pairs = [(query, result.content) for result in results]
            scores = await self._score_batch(query_content_pairs)

            # Update relevance scores
            for result, score in zip(results, scores):
                result.relevance_score = score

            # Sort by relevance_score descending
            reranked = sorted(
                results,
                key=lambda x: x.relevance_score,
                reverse=True,
            )

            # Apply top_k limit if specified
            if top_k is not None:
                if top_k < 1:
                    raise ValidationError(
                        message=f"top_k must be >= 1, got {top_k}",
                        error_code="VAL_003",
                        details={"top_k": top_k},
                    )
                reranked = reranked[:top_k]

            logger.info(
                "reranking_completed",
                query_length=len(query),
                input_count=len(results),
                output_count=len(reranked),
                top_k=top_k,
            )

            return reranked

        except ValidationError:
            # Re-raise validation errors
            raise

        except SearchError as e:
            logger.warning(
                "reranking_failed",
                error=str(e),
                fallback_enabled=self.fallback_enabled,
            )

            if self.fallback_enabled:
                # Fallback: return results sorted by original scores
                logger.info("using_fallback_scores")
                return (
                    sorted(
                        results,
                        key=lambda x: x.relevance_score,
                        reverse=True,
                    )[:top_k]
                    if top_k
                    else sorted(
                        results,
                        key=lambda x: x.relevance_score,
                        reverse=True,
                    )
                )
            else:
                raise

        except Exception as e:
            logger.warning(
                "reranking_failed",
                error=str(e),
                fallback_enabled=self.fallback_enabled,
            )

            if self.fallback_enabled:
                # Fallback: return results sorted by original scores
                logger.info("using_fallback_scores")
                return (
                    sorted(
                        results,
                        key=lambda x: x.relevance_score,
                        reverse=True,
                    )[:top_k]
                    if top_k
                    else sorted(
                        results,
                        key=lambda x: x.relevance_score,
                        reverse=True,
                    )
                )
            else:
                raise SearchError(
                    message=f"Reranking failed: {str(e)}",
                    error_code="SEARCH_003",
                    details={"error": str(e)},
                    original_exception=e,
                )

    async def score(
        self,
        query: str,
        content: Union[str, List[str]],
    ) -> Union[float, List[float]]:
        """
        Score query-content relevance pair(s).

        Directly uses cross-encoder to score semantic relevance between
        query and content. Returns scores normalized to [0, 1].

        Args:
            query: Search query
            content: Single string or list of strings to score

        Returns:
            float: If content is single string
            List[float]: If content is list of strings
            Scores are in range [0, 1] (0 = no relevance, 1 = perfect match)

        Raises:
            ValidationError: If inputs are invalid

        Example:
            ```python
            # Single content
            score = await reranker.score(
                "Python",
                "Python is a programming language"
            )
            print(f"Score: {score}")  # e.g., 0.95

            # Multiple contents
            scores = await reranker.score(
                "Python",
                ["Python guide", "Java tutorial", "C++ book"]
            )
            print(f"Scores: {scores}")  # [0.95, 0.3, 0.2]
            ```
        """
        # Validate query
        if not query or not query.strip():
            raise ValidationError(
                message="Query cannot be empty",
                error_code="VAL_001",
                details={"query": query},
            )

        # Handle single content vs list
        if isinstance(content, str):
            is_single = True
            contents: List[str] = [content]
        else:
            is_single = False
            contents = list(content)

        # Validate contents
        if not contents:
            raise ValidationError(
                message="Content list cannot be empty",
                error_code="VAL_001",
                details={"content_count": 0},
            )

        for i, c in enumerate(contents):
            if not isinstance(c, str) or not c.strip():
                raise ValidationError(
                    message=f"Content at index {i} must be non-empty string",
                    error_code="VAL_001",
                    details={"index": i, "type": str(type(c))},
                )

        try:
            # Load model if needed
            self._load_model()

            if self.model is None:
                raise SearchError(
                    message="Model failed to load",
                    error_code="SEARCH_001",
                )

            # Score all pairs
            scores = await self._score_batch([(query, c) for c in contents])

            logger.debug(
                "content_scored",
                query_length=len(query),
                content_count=len(contents),
            )

            return scores[0] if is_single else scores

        except (ValidationError, SearchError):
            raise

        except Exception as e:
            raise SearchError(
                message=f"Failed to score content: {str(e)}",
                error_code="SEARCH_003",
                details={"error": str(e)},
                original_exception=e,
            )

    async def rerank_batch(
        self,
        queries: List[dict[str, Any]],
    ) -> List[List[SearchResult]]:
        """
        Rerank multiple queries in batch.

        Processes multiple (query, results) pairs efficiently using asyncio.
        Each query reranks its own results independently.

        Args:
            queries: List of dicts with 'query' and 'results' keys
                    Example: [
                        {"query": "Python", "results": [...]},
                        {"query": "Java", "results": [...]}
                    ]

        Returns:
            List[List[SearchResult]]: Reranked results for each query

        Example:
            ```python
            batch_queries = [
                {"query": "Python", "results": [...]},
                {"query": "Java", "results": [...]}
            ]

            batch_results = await reranker.rerank_batch(batch_queries)
            ```
        """
        # Validate input
        if not queries:
            raise ValidationError(
                message="Queries list cannot be empty",
                error_code="VAL_001",
                details={"queries_count": 0},
            )

        # Process queries concurrently
        tasks = [self.rerank(q["query"], q["results"]) for q in queries]

        results = await asyncio.gather(*tasks, return_exceptions=False)

        logger.info(
            "batch_reranking_completed",
            query_count=len(queries),
        )

        return results

    async def _score_batch(
        self,
        pairs: List[tuple[str, str]],
    ) -> List[float]:
        """
        Internal method: Score multiple query-content pairs efficiently.

        Uses CrossEncoder.predict() to score all pairs in one batch.
        Applies sigmoid normalization to convert raw scores to [0, 1] range.

        Args:
            pairs: List of (query, content) tuples

        Returns:
            List[float]: Scores for each pair, normalized to [0, 1]

        Private method, not exposed in public API.
        """
        if not pairs:
            return []

        try:
            # Score all pairs at once
            if self.model is None:
                raise SearchError(
                    message="Model not loaded",
                    error_code="SEARCH_001",
                )
            raw_scores = self.model.predict(pairs)

            # Convert raw scores to probabilities via sigmoid
            # CrossEncoder returns raw logits, normalize to [0, 1]
            import numpy as np

            if isinstance(raw_scores, np.ndarray):
                # Sigmoid: 1 / (1 + e^-x)
                normalized_arr = 1 / (1 + np.exp(-raw_scores))
                return list(normalized_arr.tolist())
            else:
                # Handle list of scores
                normalized_list: List[float] = [
                    float(1 / (1 + np.exp(-s))) for s in raw_scores
                ]
                return normalized_list

        except Exception as e:
            logger.warning("batch_scoring_failed", error=str(e))
            raise SearchError(
                message=f"Batch scoring failed: {str(e)}",
                error_code="SEARCH_003",
                details={"pair_count": len(pairs), "error": str(e)},
                original_exception=e,
            )
