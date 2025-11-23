# VectorSearchEngine - Component Specification

**Level:** 2 (Component)
**Module:** zapomni_core
**Author:** Goncharenko Anton aka alienxs2
**Status:** Draft
**Version:** 1.0
**Created:** 2025-11-23

---

## Overview

### Purpose

VectorSearchEngine implements vector similarity search with optional reranking capabilities for the Zapomni memory system. It performs semantic search over embedded document chunks using cosine similarity and provides hybrid search combining BM25 keyword matching with vector similarity.

### Responsibilities

1. **Vector Similarity Search:** Execute cosine similarity search over embedded chunks stored in FalkorDB
2. **Hybrid Search (Phase 2):** Combine BM25 keyword search with vector search using RRF (Reciprocal Rank Fusion)
3. **Result Reranking:** Apply cross-encoder reranking to improve top-K result quality
4. **Query Embedding:** Generate embeddings for search queries via OllamaEmbedder
5. **Performance Optimization:** Ensure sub-200ms latency for 10K document corpus

### Position in Module

VectorSearchEngine is a core search component within zapomni_core module. It sits between the MemoryProcessor (which orchestrates the full pipeline) and the database layer (FalkorDBClient for storage).

```
┌──────────────────┐
│ MemoryProcessor  │  ← Orchestrator
└────────┬─────────┘
         │ uses
         ↓
┌──────────────────┐
│VectorSearchEngine│  ← THIS COMPONENT
└────────┬─────────┘
         │ uses
         ↓
┌──────────────────┐  ┌──────────────────┐
│ FalkorDBClient   │  │ OllamaEmbedder   │
└──────────────────┘  └──────────────────┘
```

---

## Class Definition

### Class Diagram

```
┌─────────────────────────────────────┐
│      VectorSearchEngine             │
├─────────────────────────────────────┤
│ - db_client: FalkorDBClient         │
│ - embedder: OllamaEmbedder          │
│ - bm25_index: Optional[BM25]        │
│ - reranker: Optional[CrossEncoder]  │
├─────────────────────────────────────┤
│ + __init__(db_client, embedder)     │
│ + search(query_text, limit)         │
│ + hybrid_search(query_text, limit)  │
│ + rerank_results(results, query)    │
│ - _build_bm25_index()               │
│ - _compute_cosine_similarity()      │
└─────────────────────────────────────┘
```

### Full Class Signature

```python
from typing import Optional, List, Dict, Any
from zapomni_db.falkordb import FalkorDBClient
from zapomni_core.embeddings import OllamaEmbedder
from zapomni_core.models import SearchResult
from rank_bm25 import BM25Okapi  # Phase 2


class VectorSearchEngine:
    """
    Semantic search engine using vector similarity and hybrid search.

    Performs vector similarity search over embedded document chunks using
    cosine distance. Supports hybrid search combining BM25 keyword matching
    with vector search for improved accuracy (Phase 2).

    Attributes:
        db_client: FalkorDB client for vector storage queries
        embedder: OllamaEmbedder for query embedding generation
        bm25_index: BM25 index for keyword search (Phase 2, optional)
        reranker: Cross-encoder model for reranking (Phase 2, optional)

    Performance Target:
        - Vector search: < 200ms for 10K documents
        - Hybrid search: < 300ms for 10K documents (Phase 2)
        - Query embedding: < 100ms

    Example:
        ```python
        from zapomni_db.falkordb import FalkorDBClient
        from zapomni_core.embeddings import OllamaEmbedder

        db_client = FalkorDBClient(host="localhost", port=6379)
        embedder = OllamaEmbedder(
            host="http://localhost:11434",
            model="nomic-embed-text"
        )

        engine = VectorSearchEngine(
            db_client=db_client,
            embedder=embedder
        )

        # Phase 1: Vector search only
        results = await engine.search(
            query_text="What is Python?",
            limit=10
        )

        # Phase 2: Hybrid search
        results = await engine.hybrid_search(
            query_text="Python programming language",
            limit=10
        )

        # Phase 2: Rerank top results
        reranked = await engine.rerank_results(
            results=results[:20],
            query="Python programming language"
        )
        ```
    """

    def __init__(
        self,
        db_client: FalkorDBClient,
        embedder: OllamaEmbedder,
        enable_hybrid: bool = False,  # Phase 2
        enable_reranking: bool = False  # Phase 2
    ) -> None:
        """
        Initialize VectorSearchEngine with dependencies.

        Args:
            db_client: FalkorDB client for storage and retrieval
            embedder: OllamaEmbedder for query embedding generation
            enable_hybrid: Enable BM25 hybrid search (Phase 2, default: False)
            enable_reranking: Enable cross-encoder reranking (Phase 2, default: False)

        Raises:
            ValueError: If db_client or embedder is None
            ConnectionError: If FalkorDB connection cannot be established

        Example:
            ```python
            engine = VectorSearchEngine(
                db_client=FalkorDBClient(),
                embedder=OllamaEmbedder()
            )
            ```
        """

    async def search(
        self,
        query_text: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        min_similarity: float = 0.5
    ) -> List[SearchResult]:
        """
        Execute vector similarity search (Phase 1).

        Embeds query using OllamaEmbedder, performs cosine similarity search
        against stored chunk embeddings in FalkorDB, and returns top-K results.

        Args:
            query_text: Natural language search query (non-empty, max 10,000 chars)
            limit: Maximum number of results to return (1-100, default: 10)
            filters: Optional metadata filters (tags, date_from, date_to, source)
            min_similarity: Minimum similarity threshold (0.0-1.0, default: 0.5)

        Returns:
            List of SearchResult objects sorted by similarity (descending):
            - memory_id: UUID of matching memory
            - chunk_text: Matching chunk content
            - similarity_score: Cosine similarity (0-1)
            - metadata: Original metadata (tags, source, timestamp)

        Raises:
            ValidationError: If query_text empty or limit out of range
            EmbeddingError: If query embedding generation fails
            SearchError: If database query fails
            TimeoutError: If search exceeds 1000ms timeout

        Performance Target:
            - P50 latency: < 100ms
            - P95 latency: < 200ms
            - P99 latency: < 500ms

        Algorithm:
            1. Validate inputs (query_text, limit, min_similarity)
            2. Generate query embedding via embedder.embed([query_text])
            3. Build Cypher query with vector similarity clause
            4. Apply metadata filters if provided
            5. Execute query in FalkorDB with LIMIT
            6. Compute cosine similarity for results
            7. Filter by min_similarity threshold
            8. Sort by similarity descending
            9. Return top-K results

        Example:
            ```python
            results = await engine.search(
                query_text="machine learning algorithms",
                limit=5,
                filters={"tags": ["AI", "ML"]},
                min_similarity=0.7
            )

            for result in results:
                print(f"[{result.similarity_score:.2f}] {result.chunk_text[:100]}")
            ```
        """

    async def hybrid_search(
        self,
        query_text: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        alpha: float = 0.5
    ) -> List[SearchResult]:
        """
        Execute hybrid search: BM25 + vector similarity with RRF fusion (Phase 2).

        Combines keyword-based BM25 search with semantic vector search using
        Reciprocal Rank Fusion (RRF) to merge results. Provides best of both
        approaches: exact keyword matching + semantic understanding.

        Args:
            query_text: Natural language search query (non-empty, max 10,000 chars)
            limit: Maximum number of results to return (1-100, default: 10)
            filters: Optional metadata filters (tags, date_from, date_to, source)
            alpha: Fusion weight (0.0-1.0):
                   0.0 = BM25 only
                   1.0 = vector only
                   0.5 = balanced (default)

        Returns:
            List of SearchResult objects sorted by fused score (descending):
            - memory_id: UUID of matching memory
            - chunk_text: Matching chunk content
            - similarity_score: Fused relevance score (0-1)
            - metadata: Original metadata + fusion_details

        Raises:
            ValidationError: If query_text empty or alpha out of range
            EmbeddingError: If query embedding generation fails
            SearchError: If either BM25 or vector search fails
            NotImplementedError: If BM25 index not built (enable_hybrid=False)

        Performance Target:
            - P50 latency: < 150ms
            - P95 latency: < 300ms
            - P99 latency: < 600ms

        Algorithm:
            1. Validate inputs (query_text, limit, alpha)
            2. Execute vector search (get top 2*limit results)
            3. Execute BM25 search (get top 2*limit results)
            4. Apply RRF fusion:
               - For each result in both lists:
                 fused_score = alpha * vector_score + (1-alpha) * bm25_score
            5. Merge and deduplicate results
            6. Sort by fused_score descending
            7. Apply metadata filters
            8. Return top-K results

        RRF Formula:
            For result at rank r in list:
            RRF_score = 1 / (k + r)  where k = 60 (constant)

        Example:
            ```python
            # Balanced hybrid search
            results = await engine.hybrid_search(
                query_text="Python async await",
                limit=10,
                alpha=0.5  # 50% vector, 50% BM25
            )

            # Favor semantic search
            results = await engine.hybrid_search(
                query_text="asynchronous programming concepts",
                limit=10,
                alpha=0.8  # 80% vector, 20% BM25
            )

            # Favor keyword search
            results = await engine.hybrid_search(
                query_text="asyncio.run() function",
                limit=10,
                alpha=0.2  # 20% vector, 80% BM25
            )
            ```
        """

    async def rerank_results(
        self,
        results: List[SearchResult],
        query: str,
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Rerank search results using cross-encoder for improved relevance (Phase 2).

        Applies a more sophisticated cross-encoder model to refine the ranking
        of search results. Cross-encoders jointly encode query and result,
        providing higher accuracy than bi-encoder vector similarity.

        Args:
            results: List of SearchResult objects to rerank (max 100)
            query: Original search query text
            top_k: Number of top results to return after reranking (1-100)

        Returns:
            Reranked list of SearchResult objects (top_k items):
            - similarity_score updated to cross-encoder score
            - metadata includes reranking_details

        Raises:
            ValidationError: If results empty or top_k out of range
            ProcessingError: If cross-encoder model fails
            NotImplementedError: If reranking not enabled (enable_reranking=False)

        Performance Target:
            - Latency: < 200ms for 20 results
            - Throughput: Supports up to 100 results per call

        Algorithm:
            1. Validate inputs (results, query, top_k)
            2. Extract (query, chunk_text) pairs from results
            3. Score all pairs using cross-encoder model
            4. Update similarity_score with cross-encoder scores
            5. Sort by new scores descending
            6. Return top_k results

        Use Cases:
            - Refine top 20 results from hybrid_search to best 10
            - Improve precision for complex queries
            - Filter out false positives from vector search

        Example:
            ```python
            # Get top 20 candidates from hybrid search
            candidates = await engine.hybrid_search(
                query_text="Python dependency injection patterns",
                limit=20
            )

            # Rerank to best 10 using cross-encoder
            final_results = await engine.rerank_results(
                results=candidates,
                query="Python dependency injection patterns",
                top_k=10
            )

            for i, result in enumerate(final_results, 1):
                print(f"{i}. [{result.similarity_score:.3f}] {result.chunk_text[:80]}")
            ```
        """
```

---

## Dependencies

### Component Dependencies

**From zapomni_db:**
- `FalkorDBClient` - For vector storage queries and graph operations
  - Used in: `__init__`, `search()`, `hybrid_search()`
  - Purpose: Execute Cypher queries with vector similarity

**From zapomni_core:**
- `OllamaEmbedder` - For query embedding generation
  - Used in: `__init__`, `search()`, `hybrid_search()`
  - Purpose: Convert query text to embedding vectors

- `SearchResult` (data model) - Standardized search result format
  - Used in: Return types for all search methods
  - Purpose: Consistent result structure across search types

### External Libraries

**Phase 1 (Core):**
- `numpy>=1.24.0` - For cosine similarity computation
  - Used in: `_compute_cosine_similarity()`
  - Purpose: Efficient vector operations

**Phase 2 (Hybrid Search):**
- `rank-bm25>=0.2.2` - BM25 keyword search implementation
  - Used in: `_build_bm25_index()`, `hybrid_search()`
  - Purpose: Keyword-based search

- `sentence-transformers>=2.2.0` - Cross-encoder reranking
  - Used in: `rerank_results()`
  - Purpose: High-precision result reranking
  - Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`

### Dependency Injection

Dependencies are injected via constructor (`__init__`):

```python
# Dependencies provided by caller (MemoryProcessor)
db_client = FalkorDBClient(host="localhost")
embedder = OllamaEmbedder(host="http://localhost:11434")

# Inject into VectorSearchEngine
engine = VectorSearchEngine(
    db_client=db_client,
    embedder=embedder
)
```

---

## State Management

### Attributes

**Core Attributes (Phase 1):**

- `db_client: FalkorDBClient`
  - **Type:** FalkorDBClient instance
  - **Purpose:** Execute vector similarity queries against FalkorDB
  - **Lifetime:** Entire object lifetime (set in `__init__`, never changes)
  - **Thread Safety:** Shared read-only (immutable after init)

- `embedder: OllamaEmbedder`
  - **Type:** OllamaEmbedder instance
  - **Purpose:** Generate embeddings for search queries
  - **Lifetime:** Entire object lifetime
  - **Thread Safety:** Shared read-only (embedder is thread-safe)

**Phase 2 Attributes:**

- `bm25_index: Optional[BM25Okapi]`
  - **Type:** BM25Okapi instance or None
  - **Purpose:** Pre-built BM25 index for keyword search
  - **Lifetime:** Built lazily on first hybrid_search call, cached
  - **Thread Safety:** NOT thread-safe (rebuild required if corpus changes)
  - **Invalidation:** Rebuild when new documents added

- `reranker: Optional[CrossEncoder]`
  - **Type:** CrossEncoder model instance or None
  - **Purpose:** Cross-encoder model for result reranking
  - **Lifetime:** Loaded lazily on first rerank_results call, cached
  - **Thread Safety:** Thread-safe (model inference is read-only)

### State Transitions

```
┌─────────────────┐
│   INITIALIZED   │  ← __init__ called with db_client, embedder
└────────┬────────┘
         │
         ├─→ search() → SEARCHING → RESULTS_READY → return results
         │
         ├─→ hybrid_search() (Phase 2)
         │       ↓
         │   BM25_INDEX_NEEDED?
         │       ├─→ Yes: BUILD_BM25_INDEX → HYBRID_SEARCHING → RESULTS_READY
         │       └─→ No:  HYBRID_SEARCHING → RESULTS_READY
         │
         └─→ rerank_results() (Phase 2)
                 ↓
             RERANKER_NEEDED?
                 ├─→ Yes: LOAD_RERANKER → RERANKING → RESULTS_READY
                 └─→ No:  RERANKING → RESULTS_READY
```

### Thread Safety

**Thread-Safe Operations:**
- ✅ `search()` - Read-only query execution
- ✅ `rerank_results()` - Read-only model inference (Phase 2)

**NOT Thread-Safe:**
- ❌ `hybrid_search()` - BM25 index building (use mutex if concurrent)
- ❌ `_build_bm25_index()` - Modifies internal state

**Concurrency Strategy:**
- Phase 1: Fully thread-safe (no mutable state)
- Phase 2: Use `asyncio.Lock` around BM25 index building
  ```python
  self._bm25_lock = asyncio.Lock()

  async def hybrid_search(...):
      async with self._bm25_lock:
          if self.bm25_index is None:
              await self._build_bm25_index()
  ```

---

## Public Methods (Detailed)

### Method 1: `search`

**Signature:**
```python
async def search(
    self,
    query_text: str,
    limit: int = 10,
    filters: Optional[Dict[str, Any]] = None,
    min_similarity: float = 0.5
) -> List[SearchResult]
```

**Purpose:**
Execute vector similarity search over embedded chunks using cosine similarity. This is the core Phase 1 search functionality.

**Parameters:**

- `query_text`: str
  - **Description:** Natural language search query
  - **Constraints:**
    - Must not be empty (after strip)
    - Maximum length: 10,000 characters
    - Must be valid UTF-8 text
  - **Example:** `"What is Python programming language?"`

- `limit`: int (default: 10)
  - **Description:** Maximum number of results to return
  - **Constraints:**
    - Range: 1 to 100 (inclusive)
    - Typical values: 5, 10, 20
  - **Example:** `10`

- `filters`: Optional[Dict[str, Any]] (default: None)
  - **Description:** Optional metadata filters to narrow results
  - **Structure (when provided):**
    ```python
    {
        "tags": List[str],          # Filter by tags (AND logic)
        "source": str,              # Filter by source
        "date_from": str,           # ISO 8601 date (e.g., "2025-01-01")
        "date_to": str              # ISO 8601 date
    }
    ```
  - **Example:** `{"tags": ["python", "programming"], "source": "documentation"}`

- `min_similarity`: float (default: 0.5)
  - **Description:** Minimum cosine similarity threshold
  - **Constraints:**
    - Range: 0.0 to 1.0 (inclusive)
    - Recommended: 0.5 (balanced), 0.7 (high precision), 0.3 (high recall)
  - **Example:** `0.7`

**Returns:**
- Type: `List[SearchResult]`
- Structure:
  ```python
  @dataclass
  class SearchResult:
      memory_id: str              # UUID of matching memory
      chunk_text: str             # Matching chunk content
      similarity_score: float     # Cosine similarity (0-1)
      metadata: Dict[str, Any]    # tags, source, timestamp
      chunk_index: int            # Position in original document
  ```
- **Guarantees:**
  - Results sorted by similarity_score descending
  - All results have similarity_score >= min_similarity
  - Maximum length = limit
  - No duplicate memory_id + chunk_index combinations

**Raises:**
- `ValidationError`:
  - When query_text is empty or exceeds 10,000 chars
  - When limit < 1 or limit > 100
  - When min_similarity < 0.0 or min_similarity > 1.0
  - When filters have invalid structure

- `EmbeddingError`:
  - When embedder.embed() fails (Ollama unavailable, timeout)
  - When generated embedding has wrong dimensions

- `SearchError`:
  - When FalkorDB query execution fails
  - When database connection lost

- `TimeoutError`:
  - When search operation exceeds 1000ms timeout

**Preconditions:**
- ✅ VectorSearchEngine initialized with valid db_client and embedder
- ✅ FalkorDB contains at least one embedded chunk
- ✅ Vector index exists in FalkorDB (created during add_memory)

**Postconditions:**
- ✅ Query embedding generated and cached (if caching enabled, Phase 2)
- ✅ Search completed within performance target (< 200ms P95)
- ✅ All results match filters (if provided)
- ✅ No side effects on database state

**Algorithm Outline:**
```
FUNCTION search(query_text, limit, filters, min_similarity):
    # Step 1: Validate inputs
    VALIDATE query_text is non-empty and <= 10,000 chars
    VALIDATE limit in range [1, 100]
    VALIDATE min_similarity in range [0.0, 1.0]
    IF filters provided:
        VALIDATE filters structure (tags, source, date_from, date_to)

    # Step 2: Generate query embedding
    start_time = current_timestamp()
    TRY:
        query_embedding = await embedder.embed([query_text])
        query_vector = query_embedding[0]  # Extract single embedding
    CATCH EmbeddingError:
        RAISE EmbeddingError("Failed to generate query embedding")

    # Step 3: Build Cypher query
    cypher_query = """
        MATCH (c:Chunk)
        WHERE c.embedding IS NOT NULL
        WITH c, vector.similarity(c.embedding, $query_vector) AS similarity
        WHERE similarity >= $min_similarity
    """

    # Step 4: Apply metadata filters
    IF filters.tags:
        cypher_query += " AND ALL(tag IN $tags WHERE tag IN c.tags)"
    IF filters.source:
        cypher_query += " AND c.source = $source"
    IF filters.date_from:
        cypher_query += " AND c.timestamp >= $date_from"
    IF filters.date_to:
        cypher_query += " AND c.timestamp <= $date_to"

    # Step 5: Complete query with ordering and limit
    cypher_query += """
        RETURN c.memory_id, c.text, similarity, c.metadata, c.chunk_index
        ORDER BY similarity DESC
        LIMIT $limit
    """

    # Step 6: Execute query in FalkorDB
    TRY:
        results = await db_client.execute_query(
            query=cypher_query,
            params={
                "query_vector": query_vector,
                "min_similarity": min_similarity,
                "limit": limit,
                "tags": filters.get("tags") if filters else None,
                "source": filters.get("source") if filters else None,
                "date_from": filters.get("date_from") if filters else None,
                "date_to": filters.get("date_to") if filters else None
            },
            timeout_ms=1000
        )
    CATCH DatabaseError:
        RAISE SearchError("Database query failed")

    # Step 7: Convert to SearchResult objects
    search_results = []
    FOR row IN results:
        search_results.append(SearchResult(
            memory_id=row.memory_id,
            chunk_text=row.text,
            similarity_score=row.similarity,
            metadata=row.metadata,
            chunk_index=row.chunk_index
        ))

    # Step 8: Log performance metrics
    elapsed_ms = (current_timestamp() - start_time) * 1000
    logger.info("search_completed",
                query_length=len(query_text),
                result_count=len(search_results),
                latency_ms=elapsed_ms)

    # Step 9: Return results
    RETURN search_results
END FUNCTION
```

**Edge Cases:**

1. **Empty query_text:**
   - Behavior: Raise ValidationError
   - Test: `await engine.search(query_text="", limit=10)`

2. **Query with only whitespace:**
   - Behavior: Raise ValidationError (after strip, empty)
   - Test: `await engine.search(query_text="   ", limit=10)`

3. **Extremely long query_text (> 10,000 chars):**
   - Behavior: Raise ValidationError
   - Test: `await engine.search(query_text="x" * 10001, limit=10)`

4. **limit = 0:**
   - Behavior: Raise ValidationError
   - Test: `await engine.search(query_text="test", limit=0)`

5. **limit > 100:**
   - Behavior: Raise ValidationError
   - Test: `await engine.search(query_text="test", limit=101)`

6. **min_similarity = 1.1 (out of range):**
   - Behavior: Raise ValidationError
   - Test: `await engine.search(query_text="test", limit=10, min_similarity=1.1)`

7. **No results above min_similarity:**
   - Behavior: Return empty list []
   - Test: `await engine.search(query_text="xyz", limit=10, min_similarity=0.99)`

8. **Embedder fails (Ollama unavailable):**
   - Behavior: Raise EmbeddingError with helpful message
   - Test: Mock embedder to raise exception

9. **Database connection lost:**
   - Behavior: Raise SearchError
   - Test: Mock db_client to raise ConnectionError

10. **Search timeout (> 1000ms):**
    - Behavior: Raise TimeoutError
    - Test: Mock slow database query

**Related Methods:**
- Calls: `embedder.embed()`, `db_client.execute_query()`
- Called by: `MemoryProcessor.search_memory()` (when search_mode="vector")

---

### Method 2: `hybrid_search` (Phase 2)

**Signature:**
```python
async def hybrid_search(
    self,
    query_text: str,
    limit: int = 10,
    filters: Optional[Dict[str, Any]] = None,
    alpha: float = 0.5
) -> List[SearchResult]
```

**Purpose:**
Combine BM25 keyword search with vector similarity search using Reciprocal Rank Fusion (RRF). Provides best of both: exact keyword matching + semantic understanding.

**Parameters:**

- `query_text`: str
  - Same constraints as `search()`
  - Example: `"Python asyncio await syntax"`

- `limit`: int (default: 10)
  - Same constraints as `search()`

- `filters`: Optional[Dict[str, Any]]
  - Same structure as `search()`

- `alpha`: float (default: 0.5)
  - **Description:** Fusion weight between vector and BM25 scores
  - **Constraints:** Range [0.0, 1.0]
  - **Interpretation:**
    - 0.0 = BM25 only (pure keyword search)
    - 1.0 = Vector only (pure semantic search)
    - 0.5 = Balanced (50% each)
  - **Recommendations:**
    - Use 0.8-1.0 for conceptual queries ("machine learning concepts")
    - Use 0.0-0.2 for exact matches ("asyncio.run() function")
    - Use 0.5 when unsure (balanced default)

**Returns:**
- Type: `List[SearchResult]`
- Same structure as `search()`, but:
  - `similarity_score` = fused score combining BM25 + vector
  - `metadata` includes `fusion_details`:
    ```python
    {
        "vector_score": 0.85,
        "bm25_score": 0.72,
        "alpha": 0.5,
        "fused_score": 0.785
    }
    ```

**Raises:**
- Same exceptions as `search()`, plus:
  - `NotImplementedError`: If `enable_hybrid=False` in `__init__`
  - `ValidationError`: If alpha < 0.0 or alpha > 1.0

**Preconditions:**
- ✅ VectorSearchEngine initialized with `enable_hybrid=True`
- ✅ FalkorDB contains embedded chunks
- ✅ BM25 index built (or will be built lazily)

**Postconditions:**
- ✅ BM25 index built and cached (if first call)
- ✅ Results contain fusion_details in metadata

**Algorithm Outline:**
```
FUNCTION hybrid_search(query_text, limit, filters, alpha):
    # Step 1: Validate inputs
    VALIDATE query_text, limit, filters (same as search())
    VALIDATE alpha in range [0.0, 1.0]

    IF NOT enable_hybrid:
        RAISE NotImplementedError("Hybrid search not enabled")

    # Step 2: Build BM25 index if needed
    IF bm25_index is None:
        await _build_bm25_index()

    # Step 3: Execute vector search (get 2*limit candidates)
    vector_results = await search(
        query_text=query_text,
        limit=limit * 2,
        filters=filters,
        min_similarity=0.0  # No threshold for hybrid
    )

    # Step 4: Execute BM25 search (get 2*limit candidates)
    tokenized_query = query_text.lower().split()
    bm25_scores = bm25_index.get_scores(tokenized_query)

    # Get top 2*limit BM25 results
    bm25_indices = argsort(bm25_scores)[::-1][:limit * 2]
    bm25_results = [
        {
            "memory_id": corpus[idx].memory_id,
            "chunk_text": corpus[idx].text,
            "bm25_score": bm25_scores[idx]
        }
        for idx in bm25_indices
    ]

    # Step 5: Apply RRF fusion
    # RRF formula: score(doc) = 1 / (k + rank)  where k=60
    K = 60
    fused_scores = {}

    # Add vector scores
    FOR rank, result IN enumerate(vector_results):
        doc_id = (result.memory_id, result.chunk_index)
        vector_rrf = 1.0 / (K + rank)
        fused_scores[doc_id] = {
            "result": result,
            "vector_rrf": vector_rrf,
            "bm25_rrf": 0.0
        }

    # Add BM25 scores
    FOR rank, result IN enumerate(bm25_results):
        doc_id = (result["memory_id"], result["chunk_index"])
        bm25_rrf = 1.0 / (K + rank)

        IF doc_id IN fused_scores:
            fused_scores[doc_id]["bm25_rrf"] = bm25_rrf
        ELSE:
            fused_scores[doc_id] = {
                "result": result,  # Need to fetch from DB
                "vector_rrf": 0.0,
                "bm25_rrf": bm25_rrf
            }

    # Step 6: Compute final fused scores
    final_results = []
    FOR doc_id, scores IN fused_scores.items():
        fused_score = alpha * scores["vector_rrf"] + (1 - alpha) * scores["bm25_rrf"]

        result = scores["result"]
        result.similarity_score = fused_score
        result.metadata["fusion_details"] = {
            "vector_score": scores["vector_rrf"],
            "bm25_score": scores["bm25_rrf"],
            "alpha": alpha,
            "fused_score": fused_score
        }
        final_results.append(result)

    # Step 7: Sort by fused score and return top-K
    final_results.sort(key=lambda x: x.similarity_score, reverse=True)
    RETURN final_results[:limit]
END FUNCTION
```

**Edge Cases:**
1. **alpha = 0.0 (BM25 only):** Should return pure BM25 results
2. **alpha = 1.0 (vector only):** Should return pure vector results
3. **No overlap between vector and BM25 results:** Fusion still works (union)
4. **BM25 index not built:** Build lazily on first call
5. **Query with rare words not in BM25 vocabulary:** BM25 returns zero scores

**Related Methods:**
- Calls: `search()`, `_build_bm25_index()`
- Called by: `MemoryProcessor.search_memory()` (when search_mode="hybrid")

---

### Method 3: `rerank_results` (Phase 2)

**Signature:**
```python
async def rerank_results(
    self,
    results: List[SearchResult],
    query: str,
    top_k: int = 10
) -> List[SearchResult]
```

**Purpose:**
Apply cross-encoder reranking to improve the precision of search results. Cross-encoders jointly encode query + document, providing higher accuracy than bi-encoder cosine similarity.

**Parameters:**

- `results`: List[SearchResult]
  - **Description:** List of search results to rerank
  - **Constraints:**
    - Must not be empty
    - Maximum length: 100 results (performance limit)
  - **Example:** Results from `hybrid_search()` or `search()`

- `query`: str
  - **Description:** Original search query text
  - **Constraints:** Same as `search()` query_text
  - **Example:** `"Python dependency injection patterns"`

- `top_k`: int (default: 10)
  - **Description:** Number of top results to return after reranking
  - **Constraints:** Range [1, len(results)]
  - **Example:** `10`

**Returns:**
- Type: `List[SearchResult]`
- Structure: Same as input results, but:
  - Reordered by cross-encoder scores
  - `similarity_score` updated to cross-encoder score
  - `metadata["reranking_details"]` added:
    ```python
    {
        "original_score": 0.85,
        "reranker_score": 0.92,
        "model": "cross-encoder/ms-marco-MiniLM-L-6-v2"
    }
    ```

**Raises:**
- `ValidationError`: If results empty or top_k out of range
- `ProcessingError`: If cross-encoder inference fails
- `NotImplementedError`: If `enable_reranking=False` in `__init__`

**Preconditions:**
- ✅ VectorSearchEngine initialized with `enable_reranking=True`
- ✅ Results list is non-empty

**Postconditions:**
- ✅ Cross-encoder model loaded and cached (if first call)
- ✅ Results reordered by cross-encoder scores
- ✅ Original scores preserved in metadata

**Algorithm Outline:**
```
FUNCTION rerank_results(results, query, top_k):
    # Step 1: Validate inputs
    VALIDATE results is not empty
    VALIDATE top_k in range [1, len(results)]

    IF NOT enable_reranking:
        RAISE NotImplementedError("Reranking not enabled")

    # Step 2: Load cross-encoder model if needed
    IF reranker is None:
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Step 3: Prepare (query, document) pairs
    pairs = [
        (query, result.chunk_text)
        for result in results
    ]

    # Step 4: Score all pairs with cross-encoder
    TRY:
        scores = reranker.predict(pairs)
    CATCH Exception:
        RAISE ProcessingError("Cross-encoder inference failed")

    # Step 5: Update results with new scores
    FOR i, result IN enumerate(results):
        original_score = result.similarity_score
        reranker_score = scores[i]

        result.similarity_score = reranker_score
        result.metadata["reranking_details"] = {
            "original_score": original_score,
            "reranker_score": reranker_score,
            "model": "cross-encoder/ms-marco-MiniLM-L-6-v2"
        }

    # Step 6: Sort by reranker scores
    results.sort(key=lambda x: x.similarity_score, reverse=True)

    # Step 7: Return top-K
    RETURN results[:top_k]
END FUNCTION
```

**Edge Cases:**
1. **results list has 1 item:** Return that item (no reranking needed)
2. **top_k > len(results):** Return all results
3. **top_k = len(results):** Rerank all, return all
4. **Cross-encoder model fails to load:** Raise ProcessingError
5. **Query very long (> 512 tokens):** Truncate to model max length

**Related Methods:**
- Calls: `CrossEncoder.predict()`
- Called by: User code for precision refinement

---

## Error Handling

### Exceptions Defined

```python
# Defined in zapomni_core/exceptions.py

class SearchError(ZapomniCoreError):
    """
    Raised when search operation fails.

    Examples:
    - Database query execution failed
    - Vector index not found
    - Search timeout exceeded
    """

class ValidationError(ZapomniCoreError):
    """
    Raised when input validation fails.

    Examples:
    - Empty query_text
    - limit out of range
    - Invalid filter structure
    """

class EmbeddingError(ZapomniCoreError):
    """
    Raised when embedding generation fails.

    Examples:
    - Ollama service unavailable
    - Embedding model not found
    - Wrong embedding dimensions
    """

class ProcessingError(ZapomniCoreError):
    """
    Raised when processing operation fails.

    Examples:
    - Cross-encoder inference failed
    - BM25 index building failed
    """

class NotImplementedError(Exception):
    """
    Raised when optional feature not enabled.

    Examples:
    - hybrid_search() called but enable_hybrid=False
    - rerank_results() called but enable_reranking=False
    """
```

### Error Recovery

**Transient Errors (Retry):**
- `EmbeddingError` from Ollama timeout → Retry 3x with exponential backoff
- `SearchError` from database connection → Retry 3x
- `ProcessingError` from temporary model loading → Retry 1x

**Permanent Errors (Fail Fast):**
- `ValidationError` → No retry, return error to caller immediately
- `NotImplementedError` → No retry, feature not available

**Error Propagation:**
- All exceptions bubble up to `MemoryProcessor.search_memory()`
- MemoryProcessor wraps in MCP-friendly error messages
- No exception swallowing (always log and propagate)

**Logging:**
```python
# Example error logging
logger.error(
    "search_failed",
    error_type=type(e).__name__,
    error_message=str(e),
    query_length=len(query_text),
    limit=limit
)
```

---

## Usage Examples

### Basic Usage (Phase 1)

```python
from zapomni_db.falkordb import FalkorDBClient
from zapomni_core.embeddings import OllamaEmbedder
from zapomni_core.search import VectorSearchEngine

# Initialize dependencies
db_client = FalkorDBClient(host="localhost", port=6379)
embedder = OllamaEmbedder(
    host="http://localhost:11434",
    model="nomic-embed-text"
)

# Create search engine
engine = VectorSearchEngine(
    db_client=db_client,
    embedder=embedder
)

# Execute vector search
results = await engine.search(
    query_text="What is Python?",
    limit=10,
    min_similarity=0.7
)

# Display results
for i, result in enumerate(results, 1):
    print(f"{i}. [{result.similarity_score:.2f}] {result.chunk_text[:100]}...")
```

### Advanced Usage with Filters

```python
# Search with metadata filters
results = await engine.search(
    query_text="machine learning algorithms",
    limit=20,
    filters={
        "tags": ["AI", "ML"],
        "source": "documentation",
        "date_from": "2024-01-01"
    },
    min_similarity=0.6
)

for result in results:
    print(f"[{result.similarity_score:.2f}] {result.metadata['tags']}")
    print(f"  {result.chunk_text[:150]}")
    print()
```

### Hybrid Search (Phase 2)

```python
# Initialize with hybrid search enabled
engine = VectorSearchEngine(
    db_client=db_client,
    embedder=embedder,
    enable_hybrid=True
)

# Balanced hybrid search
results = await engine.hybrid_search(
    query_text="Python asyncio programming",
    limit=10,
    alpha=0.5  # 50% vector, 50% BM25
)

# Favor semantic search for conceptual queries
results = await engine.hybrid_search(
    query_text="asynchronous programming concepts",
    limit=10,
    alpha=0.8  # 80% vector, 20% BM25
)

# Favor keyword search for exact terms
results = await engine.hybrid_search(
    query_text="asyncio.run() syntax",
    limit=10,
    alpha=0.2  # 20% vector, 80% BM25
)
```

### Reranking Pipeline (Phase 2)

```python
# Initialize with reranking enabled
engine = VectorSearchEngine(
    db_client=db_client,
    embedder=embedder,
    enable_hybrid=True,
    enable_reranking=True
)

# Step 1: Get top 20 candidates from hybrid search
candidates = await engine.hybrid_search(
    query_text="Python dependency injection patterns",
    limit=20,
    alpha=0.5
)

# Step 2: Rerank to best 10 using cross-encoder
final_results = await engine.rerank_results(
    results=candidates,
    query="Python dependency injection patterns",
    top_k=10
)

# Display with reranking details
for i, result in enumerate(final_results, 1):
    rerank = result.metadata["reranking_details"]
    print(f"{i}. Original: {rerank['original_score']:.3f} → "
          f"Reranked: {rerank['reranker_score']:.3f}")
    print(f"   {result.chunk_text[:80]}...")
```

---

## Testing Approach

### Unit Tests Required

**Core Functionality (Phase 1):**
1. `test_init_success()` - Normal initialization with valid dependencies
2. `test_init_invalid_db_client()` - None db_client raises ValueError
3. `test_init_invalid_embedder()` - None embedder raises ValueError
4. `test_search_success()` - Happy path with valid query
5. `test_search_empty_query_raises()` - Empty query_text raises ValidationError
6. `test_search_long_query_raises()` - Query > 10,000 chars raises ValidationError
7. `test_search_invalid_limit_raises()` - limit=0 or limit=101 raises ValidationError
8. `test_search_invalid_min_similarity_raises()` - min_similarity=1.5 raises ValidationError
9. `test_search_no_results()` - Query with no matches returns empty list
10. `test_search_with_filters()` - Filters applied correctly
11. `test_search_embedding_error()` - Embedder failure raises EmbeddingError
12. `test_search_database_error()` - DB query failure raises SearchError
13. `test_search_timeout()` - Timeout exceeded raises TimeoutError

**Hybrid Search (Phase 2):**
14. `test_hybrid_search_not_enabled_raises()` - enable_hybrid=False raises NotImplementedError
15. `test_hybrid_search_alpha_zero()` - alpha=0.0 returns BM25 only
16. `test_hybrid_search_alpha_one()` - alpha=1.0 returns vector only
17. `test_hybrid_search_balanced()` - alpha=0.5 fuses correctly
18. `test_hybrid_search_builds_bm25_index()` - Lazy index building works
19. `test_hybrid_search_invalid_alpha_raises()` - alpha=1.5 raises ValidationError

**Reranking (Phase 2):**
20. `test_rerank_not_enabled_raises()` - enable_reranking=False raises NotImplementedError
21. `test_rerank_success()` - Reranking improves order
22. `test_rerank_empty_results_raises()` - Empty results raises ValidationError
23. `test_rerank_invalid_top_k_raises()` - top_k > len(results) raises ValidationError
24. `test_rerank_model_failure()` - Cross-encoder failure raises ProcessingError

### Mocking Strategy

**Mock Dependencies:**
```python
import pytest
from unittest.mock import Mock, AsyncMock, patch

@pytest.fixture
def mock_db_client():
    """Mock FalkorDBClient."""
    client = Mock(spec=FalkorDBClient)
    client.execute_query = AsyncMock(return_value=[
        {
            "memory_id": "uuid-1",
            "text": "Python is a programming language",
            "similarity": 0.95,
            "metadata": {"tags": ["python"]},
            "chunk_index": 0
        }
    ])
    return client

@pytest.fixture
def mock_embedder():
    """Mock OllamaEmbedder."""
    embedder = Mock(spec=OllamaEmbedder)
    embedder.embed = AsyncMock(return_value=[
        [0.1] * 768  # Mock 768-dim embedding
    ])
    return embedder

@pytest.mark.asyncio
async def test_search_success(mock_db_client, mock_embedder):
    """Test successful search."""
    engine = VectorSearchEngine(
        db_client=mock_db_client,
        embedder=mock_embedder
    )

    results = await engine.search(
        query_text="What is Python?",
        limit=10
    )

    assert len(results) == 1
    assert results[0].similarity_score == 0.95
    assert "Python" in results[0].chunk_text
    mock_embedder.embed.assert_called_once()
    mock_db_client.execute_query.assert_called_once()
```

### Integration Tests

**Test with Real Dependencies:**
1. `test_search_with_real_falkordb()` - Full search against real FalkorDB instance
2. `test_search_with_real_ollama()` - Real embedding generation
3. `test_hybrid_search_end_to_end()` - Full hybrid pipeline
4. `test_rerank_with_real_model()` - Real cross-encoder inference

**Test Environment:**
- Docker Compose: FalkorDB + Ollama containers
- Test data: Pre-indexed sample documents
- Cleanup: Clear DB after each test

---

## Performance Considerations

### Time Complexity

**search():**
- Query embedding: O(n) where n = query length (Ollama API call)
- Vector similarity: O(k * log(N)) where N = corpus size, k = limit (HNSW index)
- Total: O(n + k * log(N)) ≈ **O(k * log(N))** for typical queries

**hybrid_search():**
- Vector search: O(k * log(N))
- BM25 search: O(N) for scoring all documents
- RRF fusion: O(k) for merging results
- Total: **O(N)** dominated by BM25

**rerank_results():**
- Cross-encoder: O(k * m) where k = results, m = model inference time
- Total: **O(k)** linear with number of results

### Space Complexity

**VectorSearchEngine instance:**
- `db_client`: O(1) - reference only
- `embedder`: O(1) - reference only
- `bm25_index` (Phase 2): **O(N * M)** where N = docs, M = avg doc length
  - Example: 10K docs × 500 tokens = 5M tokens ≈ 20MB RAM
- `reranker` (Phase 2): O(1) - model loaded once, ~40MB

**Per-query memory:**
- Query embedding: O(d) where d = 768 dimensions ≈ 3KB
- Results: O(k * chunk_size) ≈ k × 500 chars ≈ 5KB for k=10

### Optimization Opportunities

**Phase 1 Optimizations:**
1. **Semantic caching:** Cache query embeddings for repeated queries
   - Hit rate target: 30-40% (common queries)
   - TTL: 1 hour

2. **Batch embedding:** Process multiple queries in single Ollama call
   - Throughput: 5x improvement for batches of 10+

3. **HNSW index tuning:** Optimize FalkorDB vector index parameters
   - `ef_construction`: 200 (build quality)
   - `M`: 16 (graph connectivity)

**Phase 2 Optimizations:**
1. **BM25 incremental updates:** Update index without full rebuild
   - Use inverted index structure
   - Add new docs in O(M) time

2. **Reranker batching:** Process results in batches for GPU efficiency
   - Batch size: 32 (GPU memory optimized)

3. **Hybrid search caching:** Cache BM25 scores for frequent terms
   - Cache top 1000 terms
   - Hit rate: 60-70%

**Trade-offs:**
- Caching increases memory usage but reduces latency
- BM25 index adds O(N) space but enables hybrid search
- Cross-encoder is slower but more accurate than bi-encoder

### Performance Targets (Revisited)

| Operation | Input Size | P50 | P95 | P99 |
|-----------|-----------|-----|-----|-----|
| search() | 10K docs | 50ms | 100ms | 200ms |
| search() | 100K docs | 100ms | 200ms | 500ms |
| hybrid_search() | 10K docs | 100ms | 200ms | 400ms |
| rerank_results() | 20 results | 50ms | 100ms | 200ms |

**Latency Breakdown (search on 10K docs):**
- Query embedding: 30ms (Ollama)
- Vector search: 40ms (FalkorDB HNSW)
- Result marshalling: 10ms
- **Total P50: 80ms** ✅ (within 100ms target)

---

## References

### Module Spec
- [zapomni_core_module.md](../level1/zapomni_core_module.md) - Parent module specification

### Related Components
- `OllamaEmbedder` - Embedding generation service
- `FalkorDBClient` (zapomni_db) - Vector storage client
- `MemoryProcessor` - Orchestrator that uses VectorSearchEngine

### External Documentation
- FalkorDB Vector Search: https://docs.falkordb.com/vector-search.html
- Rank BM25 Library: https://github.com/dorianbrown/rank_bm25
- Cross-Encoder Models: https://www.sbert.net/docs/pretrained_cross-encoders.html
- RRF Algorithm: "Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods" (Cormack et al., 2009)

### Research References
- [03_best_practices_patterns.md](/home/dev/zapomni/research/03_best_practices_patterns.md) - Hybrid RAG best practices
- MTEB Benchmark: https://huggingface.co/spaces/mteb/leaderboard

---

**Document Status:** Draft v1.0
**Created:** 2025-11-23
**Author:** Goncharenko Anton aka alienxs2
**Copyright:** Copyright (c) 2025 Goncharenko Anton aka alienxs2
**License:** MIT License

**Ready for Review:** Yes ✅
**Next Steps:** Multi-agent verification (Level 2 verification process)
