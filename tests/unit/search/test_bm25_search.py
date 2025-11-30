"""
Unit tests for BM25Search and CodeTokenizer components.

Tests cover:
- CodeTokenizer: camelCase, snake_case, acronym handling
- BM25Search: indexing, search, persistence
- Backward compatibility with existing interface
- Edge cases and error handling

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

import tempfile
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import pytest

from zapomni_core.exceptions import SearchError, ValidationError
from zapomni_core.search.bm25_search import BM25Search, VALID_BM25_METHODS
from zapomni_core.search.bm25_tokenizer import CodeTokenizer


# ==================== FIXTURES ====================


@pytest.fixture
def tokenizer() -> CodeTokenizer:
    """Default CodeTokenizer instance."""
    return CodeTokenizer()


@pytest.fixture
def sample_documents() -> List[str]:
    """Sample documents for indexing."""
    return [
        "Python is a high-level programming language",
        "Machine learning is a subset of artificial intelligence",
        "Natural language processing uses machine learning algorithms",
        "Deep learning neural networks require large datasets",
        "Python programming is popular for data science and AI",
    ]


@pytest.fixture
def code_documents() -> List[str]:
    """Code-specific documents for testing."""
    return [
        "def calculate_user_score(user_id): return score",
        "class HTTPResponseHandler(BaseHandler): pass",
        "async def fetchDataFromAPI(): await response",
        "const getUserName = (userId) => user.name",
        "func ProcessHTTPRequest(r *Request) error",
    ]


@pytest.fixture
def mock_db_client() -> MagicMock:
    """Mock FalkorDBClient for testing."""
    return MagicMock()


@pytest.fixture
def bm25_search(mock_db_client: MagicMock) -> BM25Search:
    """BM25Search instance with mocked dependencies."""
    return BM25Search(db_client=mock_db_client)


# ==================== CODE TOKENIZER TESTS ====================


class TestCodeTokenizerCamelCase:
    """Test CodeTokenizer camelCase splitting."""

    def test_simple_camel_case(self, tokenizer: CodeTokenizer) -> None:
        """Test simple camelCase splitting."""
        tokens = tokenizer.tokenize("calculateUserScore")
        assert "calculate" in tokens
        assert "user" in tokens
        assert "score" in tokens

    def test_pascal_case(self, tokenizer: CodeTokenizer) -> None:
        """Test PascalCase splitting."""
        tokens = tokenizer.tokenize("UserScoreCalculator")
        assert "user" in tokens
        assert "score" in tokens
        assert "calculator" in tokens

    def test_mixed_case(self, tokenizer: CodeTokenizer) -> None:
        """Test mixed case identifiers."""
        tokens = tokenizer.tokenize("getUserName")
        assert "get" in tokens
        assert "user" in tokens
        assert "name" in tokens

    def test_single_word(self, tokenizer: CodeTokenizer) -> None:
        """Test single word without case transitions."""
        tokens = tokenizer.tokenize("calculate")
        assert tokens == ["calculate"]


class TestCodeTokenizerSnakeCase:
    """Test CodeTokenizer snake_case splitting."""

    def test_simple_snake_case(self, tokenizer: CodeTokenizer) -> None:
        """Test simple snake_case splitting."""
        tokens = tokenizer.tokenize("calculate_user_score")
        assert tokens == ["calculate", "user", "score"]

    def test_screaming_snake_case(self, tokenizer: CodeTokenizer) -> None:
        """Test SCREAMING_SNAKE_CASE splitting."""
        tokens = tokenizer.tokenize("MAX_RETRY_COUNT")
        assert "max" in tokens
        assert "retry" in tokens
        assert "count" in tokens

    def test_leading_underscore(self, tokenizer: CodeTokenizer) -> None:
        """Test identifiers with leading underscore."""
        tokens = tokenizer.tokenize("_private_method")
        assert "private" in tokens
        assert "method" in tokens

    def test_double_underscore(self, tokenizer: CodeTokenizer) -> None:
        """Test identifiers with double underscore."""
        tokens = tokenizer.tokenize("__init__")
        assert "init" in tokens


class TestCodeTokenizerAcronyms:
    """Test CodeTokenizer acronym handling."""

    def test_http_response(self, tokenizer: CodeTokenizer) -> None:
        """Test HTTP acronym handling."""
        tokens = tokenizer.tokenize("HTTPResponse")
        assert "http" in tokens
        assert "response" in tokens

    def test_get_http_response(self, tokenizer: CodeTokenizer) -> None:
        """Test acronym in middle of identifier."""
        tokens = tokenizer.tokenize("getHTTPResponse")
        assert "get" in tokens
        assert "http" in tokens
        assert "response" in tokens

    def test_api_handler(self, tokenizer: CodeTokenizer) -> None:
        """Test API acronym."""
        tokens = tokenizer.tokenize("APIHandler")
        assert "api" in tokens
        assert "handler" in tokens

    def test_url_parser(self, tokenizer: CodeTokenizer) -> None:
        """Test URL acronym."""
        tokens = tokenizer.tokenize("URLParser")
        assert "url" in tokens
        assert "parser" in tokens

    def test_mixed_acronyms(self, tokenizer: CodeTokenizer) -> None:
        """Test multiple acronyms."""
        tokens = tokenizer.tokenize("HTTPURLConnection")
        # When two acronyms are adjacent (HTTPURL), they may be merged
        # but the Connection part should be split
        assert "connection" in tokens
        # Either both are split or merged together
        assert "httpurl" in tokens or ("http" in tokens and "url" in tokens)


class TestCodeTokenizerKeywords:
    """Test CodeTokenizer programming keyword preservation."""

    def test_python_keywords(self, tokenizer: CodeTokenizer) -> None:
        """Test Python keywords are preserved."""
        tokens = tokenizer.tokenize("def class import from return")
        assert "def" in tokens
        assert "class" in tokens
        assert "import" in tokens
        assert "from" in tokens
        assert "return" in tokens

    def test_javascript_keywords(self, tokenizer: CodeTokenizer) -> None:
        """Test JavaScript keywords are preserved."""
        tokens = tokenizer.tokenize("const let var function async")
        assert "const" in tokens
        assert "let" in tokens
        assert "var" in tokens
        assert "function" in tokens
        assert "async" in tokens

    def test_go_keywords(self, tokenizer: CodeTokenizer) -> None:
        """Test Go keywords are preserved."""
        tokens = tokenizer.tokenize("func package struct chan defer")
        assert "func" in tokens
        assert "package" in tokens
        assert "struct" in tokens
        assert "chan" in tokens
        assert "defer" in tokens


class TestCodeTokenizerEdgeCases:
    """Test CodeTokenizer edge cases."""

    def test_empty_string(self, tokenizer: CodeTokenizer) -> None:
        """Test empty string tokenization."""
        tokens = tokenizer.tokenize("")
        assert tokens == []

    def test_whitespace_only(self, tokenizer: CodeTokenizer) -> None:
        """Test whitespace-only string."""
        tokens = tokenizer.tokenize("   \t\n  ")
        assert tokens == []

    def test_numbers(self, tokenizer: CodeTokenizer) -> None:
        """Test strings with numbers."""
        tokens = tokenizer.tokenize("user123 count456")
        # Should handle numeric suffixes
        assert len(tokens) > 0

    def test_special_characters(self, tokenizer: CodeTokenizer) -> None:
        """Test special characters are handled."""
        tokens = tokenizer.tokenize("user@email.com")
        # Punctuation should be removed
        assert "user" in tokens
        assert "email" in tokens
        assert "com" in tokens

    def test_unicode(self, tokenizer: CodeTokenizer) -> None:
        """Test unicode character handling."""
        tokens = tokenizer.tokenize("user_name variable")
        assert isinstance(tokens, list)

    def test_batch_tokenize(self, tokenizer: CodeTokenizer) -> None:
        """Test batch tokenization."""
        texts = ["getUserName", "calculate_score", "HTTPClient"]
        all_tokens = tokenizer.tokenize_batch(texts)
        assert len(all_tokens) == 3
        assert "get" in all_tokens[0]
        assert "calculate" in all_tokens[1]
        assert "http" in all_tokens[2]


# ==================== BM25 SEARCH INITIALIZATION TESTS ====================


class TestBM25Initialization:
    """Test BM25Search initialization and validation."""

    def test_init_success(self, mock_db_client: MagicMock) -> None:
        """Test successful initialization."""
        search = BM25Search(db_client=mock_db_client)
        assert search.db_client == mock_db_client
        assert search._corpus is None
        assert search._retriever is None
        assert search.method == "lucene"

    def test_init_with_method(self, mock_db_client: MagicMock) -> None:
        """Test initialization with custom method."""
        search = BM25Search(db_client=mock_db_client, method="robertson")
        assert search.method == "robertson"

    def test_init_with_index_path(self, mock_db_client: MagicMock) -> None:
        """Test initialization with index path."""
        path = Path("/tmp/test_index")
        search = BM25Search(db_client=mock_db_client, index_path=path)
        assert search.index_path == path

    def test_init_none_db_client(self) -> None:
        """Test initialization fails with None db_client."""
        with pytest.raises(ValidationError) as exc_info:
            BM25Search(db_client=None)

        assert "db_client cannot be None" in str(exc_info.value)
        assert exc_info.value.error_code == "VAL_001"

    def test_init_invalid_method(self, mock_db_client: MagicMock) -> None:
        """Test initialization fails with invalid method."""
        with pytest.raises(ValidationError) as exc_info:
            BM25Search(db_client=mock_db_client, method="invalid")

        assert "Invalid BM25 method" in str(exc_info.value)
        assert exc_info.value.error_code == "VAL_004"

    def test_all_valid_methods(self, mock_db_client: MagicMock) -> None:
        """Test all valid BM25 methods work."""
        for method in VALID_BM25_METHODS:
            search = BM25Search(db_client=mock_db_client, method=method)
            assert search.method == method


# ==================== BM25 INDEXING TESTS ====================


class TestBM25Indexing:
    """Test BM25Search document indexing."""

    def test_index_documents_success(
        self, bm25_search: BM25Search, sample_documents: List[str]
    ) -> None:
        """Test successful document indexing."""
        bm25_search.index_documents(sample_documents)

        assert bm25_search._corpus is not None
        assert len(bm25_search._corpus) == len(sample_documents)
        assert bm25_search._retriever is not None
        assert bm25_search.is_indexed

    def test_index_documents_with_ids(
        self, bm25_search: BM25Search, sample_documents: List[str]
    ) -> None:
        """Test indexing with document IDs."""
        doc_ids = [f"doc_{i}" for i in range(len(sample_documents))]
        bm25_search.index_documents(sample_documents, document_ids=doc_ids)

        assert bm25_search._document_ids == doc_ids

    def test_index_documents_empty_list(self, bm25_search: BM25Search) -> None:
        """Test indexing empty document list raises error."""
        with pytest.raises(ValidationError) as exc_info:
            bm25_search.index_documents([])

        assert "Documents list cannot be empty" in str(exc_info.value)
        assert exc_info.value.error_code == "VAL_001"

    def test_index_documents_none(self, bm25_search: BM25Search) -> None:
        """Test indexing None raises error."""
        with pytest.raises(ValidationError) as exc_info:
            bm25_search.index_documents(None)  # type: ignore

        assert "Documents must be a list" in str(exc_info.value)

    def test_index_documents_mismatched_ids(
        self, bm25_search: BM25Search, sample_documents: List[str]
    ) -> None:
        """Test indexing with mismatched document IDs raises error."""
        with pytest.raises(ValidationError) as exc_info:
            bm25_search.index_documents(sample_documents, document_ids=["id1", "id2"])

        assert "document_ids length must match" in str(exc_info.value)

    def test_reindex_documents(
        self, bm25_search: BM25Search, sample_documents: List[str]
    ) -> None:
        """Test re-indexing replaces old index."""
        bm25_search.index_documents(sample_documents)
        first_retriever = bm25_search._retriever

        new_docs = ["New document one", "New document two"]
        bm25_search.index_documents(new_docs)

        assert bm25_search._retriever is not first_retriever
        assert len(bm25_search._corpus or []) == 2


# ==================== BM25 SEARCH TESTS ====================


class TestBM25SearchBasic:
    """Test basic BM25 search functionality."""

    def test_search_success(
        self, bm25_search: BM25Search, sample_documents: List[str]
    ) -> None:
        """Test successful search returns ranked results."""
        bm25_search.index_documents(sample_documents)
        results = bm25_search.search("Python programming", limit=3)

        assert isinstance(results, list)
        assert len(results) <= 3

        if results:
            assert all(isinstance(r, dict) for r in results)
            assert all("text" in r for r in results)
            assert all("score" in r for r in results)
            assert all("index" in r for r in results)

    def test_search_score_ranking(
        self, bm25_search: BM25Search, sample_documents: List[str]
    ) -> None:
        """Test results are ranked by score (descending)."""
        bm25_search.index_documents(sample_documents)
        results = bm25_search.search("Python", limit=5)

        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_limit_respected(
        self, bm25_search: BM25Search, sample_documents: List[str]
    ) -> None:
        """Test search respects limit parameter."""
        bm25_search.index_documents(sample_documents)

        results = bm25_search.search("machine learning", limit=2)
        assert len(results) <= 2

    def test_search_score_normalization(
        self, bm25_search: BM25Search, sample_documents: List[str]
    ) -> None:
        """Test that scores are normalized to 0-1 range."""
        bm25_search.index_documents(sample_documents)
        results = bm25_search.search("Python machine learning", limit=5)

        for result in results:
            assert 0.0 <= result["score"] <= 1.0


class TestBM25SearchCode:
    """Test BM25 search with code-specific queries."""

    def test_search_camel_case_query(
        self, bm25_search: BM25Search, code_documents: List[str]
    ) -> None:
        """Test searching with camelCase query."""
        bm25_search.index_documents(code_documents)
        results = bm25_search.search("calculateUserScore", limit=3)

        assert isinstance(results, list)
        # Should find the calculate_user_score function
        if results:
            assert any("calculate" in r["text"].lower() for r in results)

    def test_search_snake_case_query(
        self, bm25_search: BM25Search, code_documents: List[str]
    ) -> None:
        """Test searching with snake_case query."""
        bm25_search.index_documents(code_documents)
        results = bm25_search.search("calculate_user_score", limit=3)

        assert isinstance(results, list)
        if results:
            assert any("calculate" in r["text"].lower() for r in results)

    def test_search_http_handler(
        self, bm25_search: BM25Search, code_documents: List[str]
    ) -> None:
        """Test searching for HTTP handler."""
        bm25_search.index_documents(code_documents)
        results = bm25_search.search("HTTPHandler", limit=3)

        assert isinstance(results, list)
        if results:
            assert any("http" in r["text"].lower() for r in results)

    def test_search_with_document_ids(
        self, bm25_search: BM25Search, code_documents: List[str]
    ) -> None:
        """Test search returns document IDs when indexed."""
        doc_ids = [f"file.py:{i}" for i in range(len(code_documents))]
        bm25_search.index_documents(code_documents, document_ids=doc_ids)

        results = bm25_search.search("calculate", limit=3)
        if results:
            assert "id" in results[0]


class TestBM25SearchValidation:
    """Test BM25 search input validation."""

    def test_search_empty_query(
        self, bm25_search: BM25Search, sample_documents: List[str]
    ) -> None:
        """Test search with empty query raises error."""
        bm25_search.index_documents(sample_documents)

        with pytest.raises(ValidationError) as exc_info:
            bm25_search.search("", limit=5)

        assert "Query cannot be empty" in str(exc_info.value)

    def test_search_whitespace_query(
        self, bm25_search: BM25Search, sample_documents: List[str]
    ) -> None:
        """Test search with whitespace-only query raises error."""
        bm25_search.index_documents(sample_documents)

        with pytest.raises(ValidationError) as exc_info:
            bm25_search.search("   ", limit=5)

        assert "Query cannot be empty" in str(exc_info.value)

    def test_search_invalid_limit(
        self, bm25_search: BM25Search, sample_documents: List[str]
    ) -> None:
        """Test search with invalid limit raises error."""
        bm25_search.index_documents(sample_documents)

        with pytest.raises(ValidationError) as exc_info:
            bm25_search.search("Python", limit=0)
        assert "Limit must be >= 1" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            bm25_search.search("Python", limit=1001)
        assert "Limit cannot exceed 1000" in str(exc_info.value)

    def test_search_not_indexed(self, bm25_search: BM25Search) -> None:
        """Test search before indexing raises error."""
        with pytest.raises(SearchError) as exc_info:
            bm25_search.search("Python", limit=5)

        assert "No documents indexed" in str(exc_info.value)


class TestBM25SearchEdgeCases:
    """Test BM25 search edge cases."""

    def test_search_no_results(self, bm25_search: BM25Search) -> None:
        """Test search with no matches returns empty list."""
        docs = ["Apple banana cherry", "Dog elephant fox"]
        bm25_search.index_documents(docs)

        # Query with words that won't produce valid tokens
        results = bm25_search.search("z", limit=5)
        assert isinstance(results, list)

    def test_single_document_search(self, bm25_search: BM25Search) -> None:
        """Test search with single document indexed."""
        bm25_search.index_documents(["Single document about Python"])
        results = bm25_search.search("Python", limit=5)

        assert len(results) == 1
        assert results[0]["score"] >= 0

    def test_large_document_set(self, bm25_search: BM25Search) -> None:
        """Test indexing and searching large document set."""
        docs = [f"Document {i} about topic {i % 10}" for i in range(100)]
        bm25_search.index_documents(docs)

        results = bm25_search.search("topic", limit=10)
        assert len(results) <= 10

    def test_default_limit(
        self, bm25_search: BM25Search, sample_documents: List[str]
    ) -> None:
        """Test search with default limit."""
        bm25_search.index_documents(sample_documents)
        results = bm25_search.search("Python")

        assert len(results) <= 10


# ==================== BM25 PERSISTENCE TESTS ====================


class TestBM25Persistence:
    """Test BM25 index persistence (save/load)."""

    def test_save_index_success(
        self, mock_db_client: MagicMock, sample_documents: List[str]
    ) -> None:
        """Test saving index to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "bm25_index"
            search = BM25Search(db_client=mock_db_client, index_path=index_path)
            search.index_documents(sample_documents)
            search.save_index()

            # Verify index files were created
            assert index_path.exists()

    def test_save_index_custom_path(
        self, mock_db_client: MagicMock, sample_documents: List[str]
    ) -> None:
        """Test saving index to custom path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            search = BM25Search(db_client=mock_db_client)
            search.index_documents(sample_documents)

            custom_path = Path(tmpdir) / "custom" / "path" / "index"
            search.save_index(custom_path)

            assert custom_path.exists()

    def test_save_index_no_path(
        self, bm25_search: BM25Search, sample_documents: List[str]
    ) -> None:
        """Test saving without path raises error."""
        bm25_search.index_documents(sample_documents)

        with pytest.raises(ValidationError) as exc_info:
            bm25_search.save_index()

        assert "No path specified" in str(exc_info.value)

    def test_save_index_not_indexed(self, mock_db_client: MagicMock) -> None:
        """Test saving without index raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "bm25_index"
            search = BM25Search(db_client=mock_db_client, index_path=index_path)

            with pytest.raises(SearchError) as exc_info:
                search.save_index()

            assert "No index to save" in str(exc_info.value)

    def test_load_index_success(
        self, mock_db_client: MagicMock, sample_documents: List[str]
    ) -> None:
        """Test loading index from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "bm25_index"

            # Create and save index
            search1 = BM25Search(db_client=mock_db_client, index_path=index_path)
            search1.index_documents(sample_documents)
            search1.save_index()

            # Load index in new instance
            search2 = BM25Search(db_client=mock_db_client, index_path=index_path)
            loaded = search2.load_index()

            assert loaded is True
            assert search2.is_indexed

    def test_load_index_not_found(self, mock_db_client: MagicMock) -> None:
        """Test loading non-existent index returns False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "nonexistent_index"
            search = BM25Search(db_client=mock_db_client, index_path=index_path)

            loaded = search.load_index()
            assert loaded is False

    def test_load_index_no_path(self, bm25_search: BM25Search) -> None:
        """Test loading without path raises error."""
        with pytest.raises(ValidationError) as exc_info:
            bm25_search.load_index()

        assert "No path specified" in str(exc_info.value)

    def test_save_load_roundtrip(
        self, mock_db_client: MagicMock, sample_documents: List[str]
    ) -> None:
        """Test save/load roundtrip preserves search functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "bm25_index"

            # Create, index, and save
            search1 = BM25Search(db_client=mock_db_client, index_path=index_path)
            search1.index_documents(sample_documents)
            results1 = search1.search("Python", limit=3)
            search1.save_index()

            # Load and search
            search2 = BM25Search(db_client=mock_db_client, index_path=index_path)
            search2.load_index()
            results2 = search2.search("Python", limit=3)

            # Results should be similar (same documents found)
            assert len(results1) == len(results2)


# ==================== BM25 STATS TESTS ====================


class TestBM25Stats:
    """Test BM25 index statistics."""

    def test_get_stats_indexed(
        self, bm25_search: BM25Search, sample_documents: List[str]
    ) -> None:
        """Test stats after indexing."""
        bm25_search.index_documents(sample_documents)
        stats = bm25_search.get_stats()

        assert stats["indexed"] is True
        assert stats["document_count"] == len(sample_documents)
        assert stats["total_tokens"] > 0
        assert stats["avg_tokens_per_doc"] > 0
        assert stats["method"] == "lucene"

    def test_get_stats_not_indexed(self, bm25_search: BM25Search) -> None:
        """Test stats before indexing."""
        stats = bm25_search.get_stats()

        assert stats["indexed"] is False
        assert stats["document_count"] == 0
        assert stats["total_tokens"] == 0

    def test_is_indexed_property(
        self, bm25_search: BM25Search, sample_documents: List[str]
    ) -> None:
        """Test is_indexed property."""
        assert bm25_search.is_indexed is False

        bm25_search.index_documents(sample_documents)
        assert bm25_search.is_indexed is True


# ==================== BACKWARD COMPATIBILITY TESTS ====================


class TestBackwardCompatibility:
    """Test backward compatibility with existing interface."""

    def test_bm25_property(
        self, bm25_search: BM25Search, sample_documents: List[str]
    ) -> None:
        """Test _bm25 property for backward compatibility."""
        assert bm25_search._bm25 is None

        bm25_search.index_documents(sample_documents)
        assert bm25_search._bm25 is not None

    def test_corpus_property(
        self, bm25_search: BM25Search, sample_documents: List[str]
    ) -> None:
        """Test _corpus property after indexing."""
        bm25_search.index_documents(sample_documents)
        assert bm25_search._corpus is not None
        assert len(bm25_search._corpus) == len(sample_documents)

    def test_documents_property(
        self, bm25_search: BM25Search, sample_documents: List[str]
    ) -> None:
        """Test _documents property after indexing."""
        bm25_search.index_documents(sample_documents)
        assert bm25_search._documents == sample_documents

    def test_tokenize_method(self, bm25_search: BM25Search) -> None:
        """Test _tokenize method works."""
        tokens = bm25_search._tokenize("getUserName")
        assert "get" in tokens
        assert "user" in tokens
        assert "name" in tokens
