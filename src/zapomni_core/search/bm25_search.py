"""
BM25Search - Enhanced BM25 keyword-based search using bm25s library.

Provides BM25 (Best Matching 25) ranking algorithm for keyword-based
document retrieval with code-aware tokenization and persistence support.

Features:
    - Multiple BM25 variants: lucene, robertson, bm25+, bm25l
    - Code-aware tokenization (camelCase, snake_case, acronyms)
    - Memory-mapped index persistence for fast loading
    - Backward compatible with existing search() interface

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import bm25s
import structlog

from zapomni_core.exceptions import SearchError, ValidationError
from zapomni_core.search.bm25_tokenizer import CodeTokenizer

logger = structlog.get_logger(__name__)

# Valid BM25 methods supported by bm25s
VALID_BM25_METHODS = frozenset({"lucene", "robertson", "bm25+", "bm25l", "atire"})


class BM25Search:
    """
    Enhanced BM25 keyword-based search implementation using bm25s.

    Uses bm25s library for efficient BM25 ranking with memory-mapped
    index persistence. Provides code-aware tokenization for better
    search results on source code.

    BM25 Variants:
        - lucene: Default, Lucene's practical BM25 implementation
        - robertson: Original BM25 (Robertson et al.)
        - bm25+: BM25+ with lower-bounding term frequency
        - bm25l: BM25L with length normalization adjustment
        - atire: ATIRE's BM25 implementation

    Attributes:
        db_client: FalkorDBClient instance for loading documents
        tokenizer: CodeTokenizer for code-aware tokenization
        method: BM25 variant to use
        index_path: Optional path for index persistence

    Example:
        ```python
        from zapomni_core.search import BM25Search
        from zapomni_db.falkordb_client import FalkorDBClient

        db_client = FalkorDBClient()
        search = BM25Search(
            db_client=db_client,
            index_path=Path("./bm25_index"),
            method="lucene"
        )

        # Index documents
        documents = [
            "def calculate_user_score(user_id): pass",
            "class HTTPResponseHandler: pass",
            "async def fetch_data_from_api(): pass"
        ]
        search.index_documents(documents)

        # Search with code-aware tokenization
        results = search.search("calculateUserScore", limit=5)
        for result in results:
            print(f"{result['score']:.2f}: {result['text']}")

        # Save index for later
        search.save_index()
        ```
    """

    def __init__(
        self,
        db_client: Any,
        index_path: Optional[Path] = None,
        method: str = "lucene",
    ) -> None:
        """
        Initialize BM25Search with dependencies.

        Args:
            db_client: FalkorDBClient instance (for loading documents)
            index_path: Optional path for index persistence
            method: BM25 variant to use (lucene, robertson, bm25+, bm25l, atire)

        Raises:
            ValidationError: If db_client is None or method is invalid

        Example:
            ```python
            from zapomni_db.falkordb_client import FalkorDBClient

            db_client = FalkorDBClient()
            search = BM25Search(
                db_client=db_client,
                index_path=Path("./indexes/bm25"),
                method="lucene"
            )
            ```
        """
        # Validate dependencies
        if db_client is None:
            raise ValidationError(
                message="db_client cannot be None",
                error_code="VAL_001",
                details={"parameter": "db_client"},
            )

        # Validate method
        if method not in VALID_BM25_METHODS:
            raise ValidationError(
                message=f"Invalid BM25 method: {method}",
                error_code="VAL_004",
                details={"method": method, "valid_methods": list(VALID_BM25_METHODS)},
            )

        self.db_client = db_client
        self.index_path = index_path
        self.method = method
        self.tokenizer = CodeTokenizer()

        # Internal state
        self._retriever: Optional[bm25s.BM25] = None
        self._documents: Optional[List[str]] = None
        self._document_ids: Optional[List[str]] = None
        self._corpus: Optional[List[List[str]]] = None

        logger.info(
            "bm25_search_initialized",
            db_client_type=type(db_client).__name__,
            method=method,
            index_path=str(index_path) if index_path else None,
        )

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using code-aware tokenizer.

        Delegates to CodeTokenizer for proper handling of:
        - camelCase identifiers
        - snake_case identifiers
        - Acronyms (HTTP, API, etc.)
        - Programming keywords

        Args:
            text: Text to tokenize

        Returns:
            List of lowercase tokens

        Example:
            >>> search._tokenize("getUserHTTPResponse")
            ['get', 'user', 'http', 'response']
        """
        return self.tokenizer.tokenize(text)

    def index_documents(
        self,
        documents: List[str],
        document_ids: Optional[List[str]] = None,
    ) -> None:
        """
        Index documents for BM25 search.

        Tokenizes documents using code-aware tokenization and builds
        BM25 index. Previous index is replaced.

        Args:
            documents: List of document strings to index
            document_ids: Optional list of document IDs (must match documents length)

        Raises:
            ValidationError: If documents is None, not a list, empty,
                           or document_ids length doesn't match

        Example:
            ```python
            search = BM25Search(db_client)

            docs = [
                "def calculate_score(user_id): pass",
                "class UserScoreCalculator: pass",
                "HTTP_TIMEOUT = 30"
            ]
            doc_ids = ["file1.py:1", "file1.py:10", "config.py:5"]

            search.index_documents(docs, doc_ids)
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

        # Validate document_ids if provided
        if document_ids is not None:
            if not isinstance(document_ids, list):
                raise ValidationError(
                    message="document_ids must be a list",
                    error_code="VAL_002",
                    details={"type": type(document_ids).__name__},
                )
            if len(document_ids) != len(documents):
                raise ValidationError(
                    message="document_ids length must match documents length",
                    error_code="VAL_003",
                    details={
                        "documents_length": len(documents),
                        "document_ids_length": len(document_ids),
                    },
                )

        logger.debug("indexing_documents", document_count=len(documents))

        # Tokenize all documents using code-aware tokenizer
        self._documents = documents
        self._document_ids = document_ids
        self._corpus = self.tokenizer.tokenize_batch(documents)

        # Build BM25 index using bm25s
        self._retriever = bm25s.BM25(method=self.method)
        self._retriever.index(self._corpus)

        total_tokens = sum(len(tokens) for tokens in self._corpus)
        logger.info(
            "documents_indexed",
            document_count=len(documents),
            total_tokens=total_tokens,
            method=self.method,
        )

    def search(
        self,
        query: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Perform BM25 keyword search.

        Algorithm:
        1. Validate query and limit
        2. Tokenize query with code-aware tokenizer
        3. Retrieve top-k documents using bm25s
        4. Normalize scores to [0, 1] range
        5. Return results with text, score, and index

        Args:
            query: Search query string
            limit: Maximum number of results (1-1000, default: 10)

        Returns:
            List of dicts with keys:
                - text: Document text
                - score: Normalized BM25 score (0-1)
                - index: Document index in corpus
                - id: Document ID if provided during indexing

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

            # Code-aware search
            results = search.search("getUserScore")
            for r in results:
                print(f"{r['score']:.2f}: {r['text']}")

            # Search with limit
            top_3 = search.search("calculate_user", limit=3)

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
                message="Query cannot be empty",
                error_code="VAL_001",
                details={"query": query},
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

        if self._retriever is None or self._corpus is None:
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

        # STEP 5: RETRIEVE TOP-K DOCUMENTS USING BM25S

        # Ensure we don't request more than available documents
        documents = self._documents or []
        actual_limit = min(limit, len(documents))

        if actual_limit == 0:
            return []

        # bm25s retrieve expects list of queries, returns (docs, scores)
        # docs shape: (num_queries, k), scores shape: (num_queries, k)
        results_tuple = self._retriever.retrieve(
            [query_tokens],
            k=actual_limit,
            corpus=documents,
        )

        # Unpack results - first element is docs, second is scores
        retrieved_docs = results_tuple[0][0]  # First query, all results
        scores = results_tuple[1][0]  # First query, all scores

        # STEP 6: NORMALIZE SCORES TO [0, 1]

        # Get min and max scores for normalization
        if len(scores) > 0:
            scores_list = list(scores)
            min_score = float(min(scores_list))
            max_score = float(max(scores_list))
        else:
            min_score = 0.0
            max_score = 1.0

        # Handle case where all scores are the same
        if max_score == min_score:
            normalized_scores = [1.0 for _ in scores]
        else:
            # Min-max normalization: (x - min) / (max - min)
            normalized_scores = [
                (float(s) - min_score) / (max_score - min_score) for s in scores
            ]

        # STEP 7: CREATE RESULTS

        results: List[Dict[str, Any]] = []

        for i, (doc, norm_score) in enumerate(zip(retrieved_docs, normalized_scores)):
            # Find the original index of this document
            doc_str = str(doc)
            try:
                original_index = documents.index(doc_str)
            except ValueError:
                original_index = i

            result: Dict[str, Any] = {
                "text": doc_str,
                "score": norm_score,
                "index": original_index,
            }

            # Add document ID if available
            if self._document_ids and original_index < len(self._document_ids):
                result["id"] = self._document_ids[original_index]

            results.append(result)

        # STEP 8: SORT BY SCORE DESCENDING (already sorted by bm25s, but ensure)

        results.sort(key=lambda x: cast(float, x["score"]), reverse=True)

        logger.info(
            "bm25_search_completed",
            query=query[:100],
            result_count=len(results),
            limit=limit,
        )

        return results

    def save_index(self, path: Optional[Path] = None) -> None:
        """
        Save the BM25 index to disk for persistence.

        Uses memory-mapped storage for efficient loading.

        Args:
            path: Path to save index (uses self.index_path if not provided)

        Raises:
            SearchError: If no index to save or path not specified
            ValidationError: If path is invalid

        Example:
            ```python
            search = BM25Search(db_client, index_path=Path("./index"))
            search.index_documents(documents)
            search.save_index()  # Uses index_path

            # Or specify path explicitly
            search.save_index(Path("./backup/bm25_index"))
            ```
        """
        save_path = path or self.index_path

        if save_path is None:
            raise ValidationError(
                message="No path specified for saving index",
                error_code="VAL_001",
                details={"path": None, "index_path": None},
            )

        if self._retriever is None:
            raise SearchError(
                message="No index to save. Call index_documents() first.",
                error_code="SEARCH_002",
                details={},
            )

        if self._documents is None:
            raise SearchError(
                message="No documents to save with index.",
                error_code="SEARCH_002",
                details={},
            )

        # Create parent directories if needed
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the index with corpus
        self._retriever.save(str(save_path), corpus=self._documents)

        logger.info(
            "bm25_index_saved",
            path=str(save_path),
            document_count=len(self._documents),
        )

    def load_index(self, path: Optional[Path] = None) -> bool:
        """
        Load a previously saved BM25 index from disk.

        Uses memory-mapping for efficient loading of large indexes.

        Args:
            path: Path to load index from (uses self.index_path if not provided)

        Returns:
            True if index loaded successfully, False otherwise

        Raises:
            ValidationError: If path is invalid or not specified

        Example:
            ```python
            search = BM25Search(db_client, index_path=Path("./index"))

            if search.load_index():
                # Index loaded, ready to search
                results = search.search("query")
            else:
                # Index not found, need to rebuild
                search.index_documents(documents)
                search.save_index()
            ```
        """
        load_path = path or self.index_path

        if load_path is None:
            raise ValidationError(
                message="No path specified for loading index",
                error_code="VAL_001",
                details={"path": None, "index_path": None},
            )

        # Check if path exists
        if not load_path.exists():
            logger.warning(
                "bm25_index_not_found",
                path=str(load_path),
            )
            return False

        try:
            # Load index with memory mapping for efficiency
            self._retriever = bm25s.BM25.load(
                str(load_path),
                load_corpus=True,
                mmap=True,
            )

            # Restore documents from corpus
            # The corpus might be a JsonlCorpus object that needs iteration
            if hasattr(self._retriever, "corpus") and self._retriever.corpus is not None:
                corpus = self._retriever.corpus
                # Convert corpus to list of strings
                # bm25s JsonlCorpus returns dicts with 'id' and 'text' keys
                # or can be a list of strings
                documents: List[str] = []
                try:
                    for doc in corpus:
                        if isinstance(doc, dict) and "text" in doc:
                            documents.append(str(doc["text"]))
                        elif isinstance(doc, str):
                            documents.append(doc)
                        else:
                            documents.append(str(doc))
                except TypeError:
                    # corpus is not iterable
                    pass
                self._documents = documents
                self._corpus = self.tokenizer.tokenize_batch(self._documents)
            else:
                self._documents = []
                self._corpus = []

            logger.info(
                "bm25_index_loaded",
                path=str(load_path),
                document_count=len(self._documents) if self._documents else 0,
            )

            return True

        except Exception as e:
            logger.error(
                "bm25_index_load_failed",
                path=str(load_path),
                error=str(e),
            )
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current BM25 index.

        Returns:
            Dictionary with index statistics

        Example:
            ```python
            stats = search.get_stats()
            print(f"Documents: {stats['document_count']}")
            print(f"Total tokens: {stats['total_tokens']}")
            print(f"Avg tokens/doc: {stats['avg_tokens_per_doc']:.1f}")
            ```
        """
        if self._corpus is None or self._documents is None:
            return {
                "indexed": False,
                "document_count": 0,
                "total_tokens": 0,
                "avg_tokens_per_doc": 0.0,
                "method": self.method,
            }

        total_tokens = sum(len(tokens) for tokens in self._corpus)
        avg_tokens = total_tokens / len(self._documents) if self._documents else 0.0

        return {
            "indexed": True,
            "document_count": len(self._documents),
            "total_tokens": total_tokens,
            "avg_tokens_per_doc": avg_tokens,
            "method": self.method,
            "has_document_ids": self._document_ids is not None,
            "index_path": str(self.index_path) if self.index_path else None,
        }

    @property
    def is_indexed(self) -> bool:
        """Check if documents have been indexed."""
        return self._retriever is not None and self._documents is not None

    # Backward compatibility: expose _bm25 property for existing tests
    @property
    def _bm25(self) -> Optional[bm25s.BM25]:
        """Backward compatibility property for existing code."""
        return self._retriever

    @_bm25.setter
    def _bm25(self, value: Optional[bm25s.BM25]) -> None:
        """Backward compatibility setter."""
        self._retriever = value
