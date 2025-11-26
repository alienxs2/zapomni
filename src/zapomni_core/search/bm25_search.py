"""
BM25Search - TF-IDF keyword-based search using rank-bm25.

Provides BM25 (Best Matching 25) ranking algorithm for keyword-based
document retrieval. Complements vector search with traditional IR approach.

BM25 Algorithm:
    - TF-IDF based ranking with document length normalization
    - Parameters: k1 (term saturation), b (length normalization)
    - Fast and effective for exact keyword matching

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from typing import Any, Dict, List, Optional

import structlog
from rank_bm25 import BM25Okapi

from zapomni_core.exceptions import SearchError, ValidationError

logger = structlog.get_logger(__name__)


class BM25Search:
    """
    BM25 keyword-based search implementation.

    Uses rank-bm25 library for efficient BM25 ranking. Provides:
    1. Document indexing with tokenization
    2. Keyword-based search with score ranking
    3. Score normalization to [0, 1] range

    BM25 Parameters:
        - k1=1.5: Term frequency saturation parameter
        - b=0.75: Length normalization parameter

    Attributes:
        db_client: FalkorDBClient instance for loading documents
        _corpus: Tokenized document corpus
        _bm25: BM25Okapi index instance

    Example:
        ```python
        from zapomni_core.search import BM25Search
        from zapomni_db.falkordb_client import FalkorDBClient

        db_client = FalkorDBClient()
        search = BM25Search(db_client=db_client)

        # Index documents
        documents = [
            "Python is a programming language",
            "Machine learning uses algorithms",
            "Natural language processing"
        ]
        search.index_documents(documents)

        # Search
        results = search.search("Python programming", limit=5)
        for result in results:
            print(f"{result['score']:.2f}: {result['text']}")
        ```
    """

    def __init__(self, db_client):
        """
        Initialize BM25Search with dependencies.

        Args:
            db_client: FalkorDBClient instance (for loading documents)

        Raises:
            ValidationError: If db_client is None

        Example:
            ```python
            from zapomni_db.falkordb_client import FalkorDBClient

            db_client = FalkorDBClient()
            search = BM25Search(db_client=db_client)
            ```
        """
        # Validate dependencies
        if db_client is None:
            raise ValidationError(
                message="db_client cannot be None",
                error_code="VAL_001",
                details={"parameter": "db_client"},
            )

        self.db_client = db_client
        self._corpus: Optional[List[List[str]]] = None
        self._bm25: Optional[BM25Okapi] = None
        self._documents: Optional[List[str]] = None

        logger.info("bm25_search_initialized", db_client_type=type(db_client).__name__)

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into lowercase words.

        Simple tokenization strategy:
        1. Convert to lowercase
        2. Split on whitespace and punctuation
        3. Filter out empty tokens

        Args:
            text: Text to tokenize

        Returns:
            List of lowercase tokens

        Example:
            >>> search._tokenize("Hello, World!")
            ['hello', 'world']
        """
        if not text:
            return []

        # Convert to lowercase and split
        # Remove punctuation by keeping only alphanumeric and spaces
        import re

        # Replace punctuation with spaces, then split
        text = re.sub(r"[^\w\s]", " ", text.lower())
        tokens = text.split()

        return [t for t in tokens if t]

    def index_documents(self, documents: List[str]) -> None:
        """
        Index documents for BM25 search.

        Tokenizes documents and builds BM25 index. Previous index is replaced.

        Args:
            documents: List of document strings to index

        Raises:
            ValidationError: If documents is None, not a list, or empty

        Example:
            ```python
            search = BM25Search(db_client)

            docs = [
                "Python programming language",
                "Machine learning algorithms",
                "Natural language processing"
            ]
            search.index_documents(docs)
            ```
        """
        # Validate documents
        if documents is None:
            raise ValidationError(
                message="Documents must be a list",
                error_code="VAL_001",
                details={"documents": documents},
            )

        if not isinstance(documents, list):
            raise ValidationError(
                message="Documents must be a list",
                error_code="VAL_002",
                details={"type": type(documents).__name__},
            )

        if len(documents) == 0:
            raise ValidationError(
                message="Documents list cannot be empty",
                error_code="VAL_001",
                details={"documents": documents},
            )

        logger.debug("indexing_documents", document_count=len(documents))

        # Tokenize all documents
        self._documents = documents
        self._corpus = [self._tokenize(doc) for doc in documents]

        # Build BM25 index
        self._bm25 = BM25Okapi(self._corpus)

        logger.info(
            "documents_indexed",
            document_count=len(documents),
            total_tokens=sum(len(tokens) for tokens in self._corpus),
        )

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform BM25 keyword search.

        Algorithm:
        1. Validate query and limit
        2. Tokenize query
        3. Calculate BM25 scores for all documents
        4. Normalize scores to [0, 1] range
        5. Sort by score descending
        6. Return top N results

        Args:
            query: Search query string
            limit: Maximum number of results (1-1000, default: 10)

        Returns:
            List of dicts with keys:
                - text: Document text
                - score: Normalized BM25 score (0-1)
                - index: Document index in corpus

        Raises:
            ValidationError: If query is empty or limit is invalid
            SearchError: If no documents indexed

        Performance Target:
            - Single search: < 50ms (P95)
            - Empty result: < 10ms

        Example:
            ```python
            search = BM25Search(db_client)
            search.index_documents(documents)

            # Basic search
            results = search.search("Python programming")
            for r in results:
                print(f"{r['score']:.2f}: {r['text']}")

            # Search with limit
            top_3 = search.search("machine learning", limit=3)

            # Handle no results
            results = search.search("nonexistent")
            if not results:
                print("No results found")
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

        # STEP 3: CHECK INDEX EXISTS

        if self._bm25 is None or self._corpus is None:
            raise SearchError(
                message="No documents indexed. Call index_documents() first.",
                error_code="SEARCH_002",
                details={"query": query[:100]},
            )

        # STEP 4: TOKENIZE QUERY

        logger.debug("executing_bm25_search", query=query[:100], limit=limit)

        query_tokens = self._tokenize(query)

        if not query_tokens:
            # Query has no valid tokens after tokenization
            logger.debug("query_has_no_tokens", query=query)
            return []

        # STEP 5: CALCULATE BM25 SCORES

        scores = self._bm25.get_scores(query_tokens)

        # STEP 6: NORMALIZE SCORES TO [0, 1]

        # BM25 scores can be negative. Shift to positive range first.
        min_score = min(scores) if len(scores) > 0 else 0.0
        max_score = max(scores) if len(scores) > 0 else 1.0

        # Handle case where all scores are the same
        if max_score == min_score:
            normalized_scores = [1.0 for _ in scores]
        else:
            # Min-max normalization: (x - min) / (max - min)
            normalized_scores = [(s - min_score) / (max_score - min_score) for s in scores]

        # STEP 7: CREATE RESULTS WITH SCORES AND INDICES

        results = [
            {"text": self._documents[i], "score": normalized_scores[i], "index": i}
            for i in range(len(self._documents))
        ]

        # STEP 8: SORT BY SCORE DESCENDING

        results.sort(key=lambda x: x["score"], reverse=True)

        # STEP 9: APPLY LIMIT

        results = results[:limit]

        logger.info(
            "bm25_search_completed", query=query[:100], result_count=len(results), limit=limit
        )

        return results
