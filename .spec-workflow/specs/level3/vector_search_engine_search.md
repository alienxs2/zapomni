# VectorSearchEngine.search - Function Specification

**Level:** 3 (Function)
**Component:** VectorSearchEngine
**Module:** zapomni_core
**Parent Spec:** [vector_search_engine_component.md](../level2/vector_search_engine_component.md)
**Author:** Goncharenko Anton aka alienxs2
**Status:** Draft
**Version:** 1.0
**Created:** 2025-11-23

---

## Function Signature

```python
async def search(
    self,
    query_text: str,
    limit: int = 10,
    filters: Optional[Dict[str, Any]] = None,
    min_similarity: float = 0.5
) -> List[SearchResult]:
    """
    Execute vector similarity search over embedded chunks using cosine similarity.

    This is the core Phase 1 search functionality that converts natural language
    queries into embeddings, performs vector similarity search in FalkorDB, and
    returns ranked results. Designed for sub-200ms latency on 10K document corpus.

    The function implements a 9-step algorithm:
    1. Input validation (query_text, limit, min_similarity, filters)
    2. Query embedding generation via OllamaEmbedder
    3. Cypher query construction with vector similarity clause
    4. Metadata filter application (tags, source, dates)
    5. Query execution in FalkorDB with timeout
    6. Cosine similarity computation for results
    7. Filtering by min_similarity threshold
    8. Sorting by similarity score (descending)
    9. Return top-K SearchResult objects

    Args:
        query_text: Natural language search query
            - Constraints: Non-empty, max 10,000 characters, valid UTF-8
            - Example: "What are Python asyncio patterns?"

        limit: Maximum number of results to return (default: 10)
            - Constraints: Range [1, 100] inclusive
            - Typical values: 5, 10, 20
            - Example: 10

        filters: Optional metadata filters to narrow results (default: None)
            - Structure when provided:
              {
                  "tags": List[str],      # Filter by tags (AND logic)
                  "source": str,          # Filter by source
                  "date_from": str,       # ISO 8601 date (e.g., "2025-01-01")
                  "date_to": str          # ISO 8601 date
              }
            - Example: {"tags": ["python", "async"], "source": "docs"}

        min_similarity: Minimum cosine similarity threshold (default: 0.5)
            - Constraints: Range [0.0, 1.0] inclusive
            - Recommendations:
              * 0.5 = balanced (default)
              * 0.7 = high precision
              * 0.3 = high recall
            - Example: 0.7

    Returns:
        List[SearchResult]: Sorted list of matching results
            - Maximum length: min(limit, total_matches)
            - Sorted by similarity_score descending
            - All results have similarity_score >= min_similarity
            - No duplicate (memory_id, chunk_index) combinations

        SearchResult structure:
        @dataclass
        class SearchResult:
            memory_id: str              # UUID of matching memory
            chunk_text: str             # Matching chunk content
            similarity_score: float     # Cosine similarity (0-1)
            metadata: Dict[str, Any]    # Original metadata (tags, source, timestamp)
            chunk_index: int            # Position in original document

    Raises:
        ValidationError: When input validation fails
            - Empty or whitespace-only query_text
            - query_text exceeds 10,000 characters
            - limit < 1 or limit > 100
            - min_similarity < 0.0 or min_similarity > 1.0
            - filters have invalid structure (missing required keys, wrong types)

        EmbeddingError: When query embedding generation fails
            - Ollama service unavailable or timeout
            - Embedding model not found
            - Generated embedding has wrong dimensions
            - Network connectivity issues

        SearchError: When database query fails
            - FalkorDB query execution error
            - Database connection lost
            - Vector index not found
            - Cypher syntax error

        TimeoutError: When search exceeds timeout
            - Operation exceeds 1000ms timeout
            - Database query hangs

    Performance Targets:
        - P50 latency: < 100ms for 10K documents
        - P95 latency: < 200ms for 10K documents
        - P99 latency: < 500ms for 10K documents
        - Throughput: Support 100 concurrent searches/sec

    Example:
        ```python
        from zapomni_core.search import VectorSearchEngine
        from zapomni_db.falkordb import FalkorDBClient
        from zapomni_core.embeddings import OllamaEmbedder

        # Initialize engine
        db_client = FalkorDBClient(host="localhost")
        embedder = OllamaEmbedder(host="http://localhost:11434")
        engine = VectorSearchEngine(db_client=db_client, embedder=embedder)

        # Basic search
        results = await engine.search(
            query_text="machine learning algorithms",
            limit=10,
            min_similarity=0.7
        )

        # Search with filters
        results = await engine.search(
            query_text="Python async programming",
            limit=20,
            filters={
                "tags": ["python", "async"],
                "source": "documentation",
                "date_from": "2024-01-01"
            },
            min_similarity=0.6
        )

        # Display results
        for i, result in enumerate(results, 1):
            print(f"{i}. [{result.similarity_score:.2f}] {result.chunk_text[:100]}...")
            print(f"   Source: {result.metadata.get('source', 'unknown')}")
            print(f"   Tags: {', '.join(result.metadata.get('tags', []))}")
        ```
    """
```

---

## Purpose & Context

### What It Does

Executes semantic vector similarity search over embedded document chunks stored in FalkorDB. The function:

1. **Converts** natural language query to embedding vector via OllamaEmbedder
2. **Searches** FalkorDB vector index using cosine similarity
3. **Filters** results by similarity threshold and optional metadata constraints
4. **Returns** ranked list of matching chunks with metadata

This is the core retrieval mechanism for Phase 1 of the Zapomni memory system.

### Why It Exists

**Business Need:**
- Enable semantic search over stored memories
- Support natural language queries (not just keyword matching)
- Power the search_memory MCP tool

**Technical Need:**
- Provide vector similarity search abstraction
- Integrate OllamaEmbedder + FalkorDB components
- Ensure consistent error handling and validation

### When To Use

**Use this function when:**
- User performs search via MCP search_memory tool
- MemoryProcessor needs to retrieve relevant context
- Testing semantic search functionality
- Benchmarking search performance

**Do NOT use when:**
- Need hybrid search (BM25 + vector) → Use `hybrid_search()` (Phase 2)
- Need reranked results → Use `rerank_results()` after search (Phase 2)
- Need to search code repositories → Use codify pipeline instead

### When NOT To Use

- **Empty knowledge base:** Returns empty list [] (no error)
- **Hybrid search needed:** Use `hybrid_search()` for BM25 + vector fusion
- **High precision required:** Chain with `rerank_results()` for cross-encoder refinement

---

## Parameters (Detailed)

### query_text: str

**Type:** `str` (required)

**Purpose:** Natural language search query to convert to embedding and match against stored chunks

**Constraints:**
- **Non-empty:** After `.strip()`, must have length > 0
- **Maximum length:** 10,000 characters (prevents Ollama timeout)
- **Encoding:** Valid UTF-8 text (no binary data)
- **No special restrictions:** Can contain any natural language, including punctuation, emojis, code snippets

**Validation Logic:**
```python
# Step 1: Strip whitespace
query_text = query_text.strip()

# Step 2: Check non-empty
if not query_text:
    raise ValidationError("query_text cannot be empty")

# Step 3: Check maximum length
if len(query_text) > 10_000:
    raise ValidationError(
        f"query_text exceeds maximum length (10,000 chars): got {len(query_text)}"
    )

# Step 4: Verify UTF-8 (Python strings are always valid UTF-8)
# No explicit check needed unless dealing with bytes
```

**Examples:**

✅ **Valid:**
- `"What is Python?"`
- `"machine learning algorithms for NLP"`
- `"async def main():\n    await process()"`
- `"How to fix ImportError in Python 🐍"`
- Query with 9,999 characters (within limit)

❌ **Invalid:**
- `""` (empty string) → ValidationError
- `"   "` (whitespace only) → ValidationError (becomes empty after strip)
- `"x" * 10_001` (10,001 chars) → ValidationError
- `None` (not a string) → TypeError (before validation)

**Edge Cases:**
1. **Single character:** `"a"` → Valid (minimum meaningful query)
2. **Only punctuation:** `"???"` → Valid (will likely return no results due to low semantic content)
3. **Mixed languages:** `"Python программирование"` → Valid (multilingual embeddings supported by nomic-embed-text)
4. **Code snippets:** `"def foo(): pass"` → Valid (code can be embedded)
5. **Very long query (9,999 chars):** Valid but may have slower embedding generation

---

### limit: int

**Type:** `int` (default: 10)

**Purpose:** Maximum number of results to return from search

**Constraints:**
- **Minimum:** 1 (must request at least one result)
- **Maximum:** 100 (prevents excessive memory usage and latency)
- **Default:** 10 (balanced for most use cases)

**Validation Logic:**
```python
if limit < 1:
    raise ValidationError(
        f"limit must be at least 1: got {limit}"
    )

if limit > 100:
    raise ValidationError(
        f"limit cannot exceed 100: got {limit}"
    )
```

**Examples:**

✅ **Valid:**
- `1` (minimum, get single best match)
- `10` (default, typical use case)
- `20` (larger result set)
- `100` (maximum allowed)

❌ **Invalid:**
- `0` → ValidationError
- `-5` → ValidationError
- `101` → ValidationError
- `1000` → ValidationError

**Use Cases by Value:**
- **limit=1:** Find single best match (e.g., "most relevant document")
- **limit=5:** Quick preview of top results
- **limit=10:** Standard search results (default)
- **limit=20-50:** Comprehensive search for reranking pipeline
- **limit=100:** Maximum recall for analysis

**Edge Cases:**
1. **limit > total_results:** Returns all available results (e.g., limit=100 but only 5 matches)
2. **limit=1 with ties:** Returns first result (deterministic ordering by similarity DESC, then by memory_id ASC)

---

### filters: Optional[Dict[str, Any]]

**Type:** `Optional[Dict[str, Any]]` (default: None)

**Purpose:** Optional metadata filters to narrow search results by tags, source, or date range

**Default Behavior:**
- When `None` (default): No filtering, search all chunks
- When provided: Apply AND logic for all specified filters

**Structure (when provided):**
```python
{
    "tags": List[str],          # Optional: Filter by tags (chunk must have ALL tags)
    "source": str,              # Optional: Filter by exact source match
    "date_from": str,           # Optional: ISO 8601 date (YYYY-MM-DD), inclusive
    "date_to": str              # Optional: ISO 8601 date (YYYY-MM-DD), inclusive
}
```

**All keys are optional** - can provide any combination.

**Validation Logic:**
```python
if filters is not None:
    # Validate filters is a dict
    if not isinstance(filters, dict):
        raise ValidationError("filters must be a dictionary")

    # Validate tags (if provided)
    if "tags" in filters:
        tags = filters["tags"]
        if not isinstance(tags, list):
            raise ValidationError("filters['tags'] must be a list")
        if not all(isinstance(tag, str) for tag in tags):
            raise ValidationError("All tags must be strings")
        if not tags:  # Empty list
            raise ValidationError("filters['tags'] cannot be empty list")

    # Validate source (if provided)
    if "source" in filters:
        source = filters["source"]
        if not isinstance(source, str):
            raise ValidationError("filters['source'] must be a string")
        if not source.strip():
            raise ValidationError("filters['source'] cannot be empty")

    # Validate date_from (if provided)
    if "date_from" in filters:
        date_from = filters["date_from"]
        if not isinstance(date_from, str):
            raise ValidationError("filters['date_from'] must be a string")
        try:
            datetime.fromisoformat(date_from)  # Validate ISO 8601
        except ValueError:
            raise ValidationError(
                f"filters['date_from'] must be ISO 8601 date: got '{date_from}'"
            )

    # Validate date_to (if provided)
    if "date_to" in filters:
        date_to = filters["date_to"]
        if not isinstance(date_to, str):
            raise ValidationError("filters['date_to'] must be a string")
        try:
            datetime.fromisoformat(date_to)
        except ValueError:
            raise ValidationError(
                f"filters['date_to'] must be ISO 8601 date: got '{date_to}'"
            )

    # Validate date range (if both provided)
    if "date_from" in filters and "date_to" in filters:
        date_from_dt = datetime.fromisoformat(filters["date_from"])
        date_to_dt = datetime.fromisoformat(filters["date_to"])
        if date_from_dt > date_to_dt:
            raise ValidationError(
                "filters['date_from'] cannot be after filters['date_to']"
            )
```

**Examples:**

✅ **Valid:**
- `None` (no filters, search all)
- `{"tags": ["python"]}` (single tag)
- `{"tags": ["python", "async"]}` (multiple tags)
- `{"source": "documentation"}` (exact source match)
- `{"date_from": "2024-01-01"}` (from date only)
- `{"date_to": "2025-12-31"}` (to date only)
- `{"date_from": "2024-01-01", "date_to": "2024-12-31"}` (date range)
- `{"tags": ["ML"], "source": "research", "date_from": "2024-06-01"}` (combined filters)

❌ **Invalid:**
- `"not_a_dict"` → ValidationError
- `{"tags": "python"}` → ValidationError (tags must be list)
- `{"tags": []}` → ValidationError (empty list not allowed)
- `{"tags": [123]}` → ValidationError (tags must be strings)
- `{"source": ""}` → ValidationError (empty source)
- `{"date_from": "01-01-2024"}` → ValidationError (not ISO 8601)
- `{"date_from": "2024-12-31", "date_to": "2024-01-01"}` → ValidationError (from > to)
- `{"unknown_key": "value"}` → Ignored (unknown keys are allowed but ignored)

**Filter Semantics:**

**tags (AND logic):**
```python
# filters={"tags": ["python", "async"]}
# Matches chunks with BOTH tags
# chunk.tags = ["python", "async", "web"] → Match ✅
# chunk.tags = ["python"] → No match ❌
# chunk.tags = ["async"] → No match ❌
```

**source (exact match):**
```python
# filters={"source": "documentation"}
# chunk.source = "documentation" → Match ✅
# chunk.source = "Documentation" → No match ❌ (case-sensitive)
# chunk.source = "docs" → No match ❌
```

**date_from/date_to (inclusive range):**
```python
# filters={"date_from": "2024-01-01", "date_to": "2024-12-31"}
# chunk.timestamp = "2024-01-01" → Match ✅ (inclusive)
# chunk.timestamp = "2024-06-15" → Match ✅
# chunk.timestamp = "2024-12-31" → Match ✅ (inclusive)
# chunk.timestamp = "2023-12-31" → No match ❌
# chunk.timestamp = "2025-01-01" → No match ❌
```

**Edge Cases:**
1. **Empty filters dict `{}`:** Valid, no filtering (equivalent to `None`)
2. **Unknown keys:** Ignored (e.g., `{"tags": ["x"], "foo": "bar"}` → only tags filter applied)
3. **Conflicting filters:** All must match (AND logic) - may return zero results

---

### min_similarity: float

**Type:** `float` (default: 0.5)

**Purpose:** Minimum cosine similarity threshold to include result in output

**Constraints:**
- **Minimum:** 0.0 (no similarity required, essentially disabled)
- **Maximum:** 1.0 (perfect match required)
- **Default:** 0.5 (balanced threshold)

**Validation Logic:**
```python
if min_similarity < 0.0:
    raise ValidationError(
        f"min_similarity cannot be negative: got {min_similarity}"
    )

if min_similarity > 1.0:
    raise ValidationError(
        f"min_similarity cannot exceed 1.0: got {min_similarity}"
    )
```

**Examples:**

✅ **Valid:**
- `0.0` (include all results, no threshold)
- `0.3` (high recall, low precision)
- `0.5` (balanced, default)
- `0.7` (high precision, low recall)
- `0.9` (very high precision, may return few results)
- `1.0` (perfect match only, extremely rare)

❌ **Invalid:**
- `-0.1` → ValidationError
- `1.5` → ValidationError
- `float('inf')` → ValidationError

**Threshold Recommendations:**

| Threshold | Use Case | Expected Behavior |
|-----------|----------|-------------------|
| 0.0-0.3 | Maximum recall | Returns many results, some may be irrelevant |
| 0.4-0.6 | Balanced (default 0.5) | Good mix of precision and recall |
| 0.7-0.8 | High precision | Returns only highly relevant results |
| 0.9-1.0 | Exact/near-exact match | May return very few or zero results |

**Cosine Similarity Interpretation:**
- **1.0:** Identical vectors (exact semantic match)
- **0.9-0.99:** Very high similarity (near-duplicates, paraphrases)
- **0.7-0.89:** High similarity (related concepts)
- **0.5-0.69:** Moderate similarity (some relevance)
- **0.3-0.49:** Low similarity (weak connection)
- **0.0-0.29:** Very low similarity (likely irrelevant)

**Edge Cases:**
1. **min_similarity=1.0:** Valid but will likely return zero results (perfect match extremely rare)
2. **min_similarity=0.0 with limit=100:** Returns top 100 by similarity, no filtering
3. **High threshold with no matches:** Returns empty list [] (not an error)

---

## Return Value

**Type:** `List[SearchResult]`

**Guaranteed Properties:**
- ✅ Sorted by `similarity_score` **descending** (highest first)
- ✅ All results have `similarity_score >= min_similarity`
- ✅ Maximum length: `min(limit, total_matches_above_threshold)`
- ✅ No duplicate `(memory_id, chunk_index)` combinations
- ✅ All `SearchResult` objects fully populated (no None values)

**SearchResult Structure:**
```python
@dataclass
class SearchResult:
    """Single search result from vector similarity search."""

    memory_id: str              # UUID of memory (e.g., "550e8400-e29b-41d4-a716-446655440000")
    chunk_text: str             # Full text of matching chunk (500-1000 chars typical)
    similarity_score: float     # Cosine similarity score [0.0, 1.0]
    metadata: Dict[str, Any]    # Original metadata from chunk
    chunk_index: int            # Position of chunk in original document (0-based)
```

**metadata Field Contents:**
```python
{
    "tags": List[str],          # Tags associated with chunk (e.g., ["python", "async"])
    "source": str,              # Source of chunk (e.g., "documentation", "user_input")
    "timestamp": str,           # ISO 8601 timestamp when memory was created
    # ... other custom metadata fields
}
```

**Return Value Examples:**

**Example 1: Normal search with 3 results**
```python
[
    SearchResult(
        memory_id="uuid-1",
        chunk_text="Python is a high-level programming language...",
        similarity_score=0.95,
        metadata={"tags": ["python"], "source": "docs", "timestamp": "2024-06-15T10:30:00Z"},
        chunk_index=0
    ),
    SearchResult(
        memory_id="uuid-2",
        chunk_text="Asyncio provides infrastructure for asynchronous programming...",
        similarity_score=0.87,
        metadata={"tags": ["python", "async"], "source": "docs", "timestamp": "2024-06-16T14:20:00Z"},
        chunk_index=2
    ),
    SearchResult(
        memory_id="uuid-3",
        chunk_text="Type hints in Python enable static type checking...",
        similarity_score=0.72,
        metadata={"tags": ["python", "typing"], "source": "tutorial", "timestamp": "2024-06-17T09:15:00Z"},
        chunk_index=0
    )
]
```

**Example 2: Empty results (no matches above threshold)**
```python
[]  # Empty list, not None
```

**Example 3: Single result (limit=1)**
```python
[
    SearchResult(
        memory_id="uuid-42",
        chunk_text="The best match for your query...",
        similarity_score=0.89,
        metadata={"tags": ["ai"], "source": "research", "timestamp": "2024-07-01T12:00:00Z"},
        chunk_index=5
    )
]
```

**Ordering Guarantees:**
- **Primary sort:** `similarity_score` descending (highest first)
- **Secondary sort (tie-breaker):** `memory_id` ascending (deterministic ordering)

**Example with ties:**
```python
# Two results with same similarity (rare but possible)
[
    SearchResult(memory_id="uuid-aaa", similarity_score=0.85, ...),  # "aaa" < "bbb" lexicographically
    SearchResult(memory_id="uuid-bbb", similarity_score=0.85, ...),
    SearchResult(memory_id="uuid-ccc", similarity_score=0.80, ...)
]
```

---

## Exceptions

### ValidationError

**Module:** `zapomni_core.exceptions`

**Base Class:** `ZapomniCoreError`

**When Raised:**
1. **Empty query_text:** `query_text.strip() == ""`
2. **Query too long:** `len(query_text) > 10_000`
3. **Invalid limit:** `limit < 1` or `limit > 100`
4. **Invalid min_similarity:** `min_similarity < 0.0` or `min_similarity > 1.0`
5. **Invalid filters structure:** See "filters" parameter validation above

**Message Format:**
```python
f"Validation failed: {specific_reason}"
```

**Examples:**
```python
ValidationError("Validation failed: query_text cannot be empty")
ValidationError("Validation failed: query_text exceeds maximum length (10,000 chars): got 15000")
ValidationError("Validation failed: limit must be at least 1: got 0")
ValidationError("Validation failed: min_similarity cannot exceed 1.0: got 1.5")
ValidationError("Validation failed: filters['tags'] must be a list")
```

**Recovery Strategy:**
- **Caller should NOT retry:** Fix input and resubmit
- **User feedback required:** Display validation error to user
- **Logging:** Log at INFO level (not ERROR, as it's expected user error)

---

### EmbeddingError

**Module:** `zapomni_core.exceptions`

**Base Class:** `ZapomniCoreError`

**When Raised:**
1. **Ollama unavailable:** Service not running or network unreachable
2. **Embedding timeout:** Ollama takes > 5 seconds to respond
3. **Model not found:** Embedding model not pulled in Ollama
4. **Wrong dimensions:** Generated embedding has unexpected dimensions (e.g., got 384 instead of 768)
5. **Invalid response:** Ollama returns malformed JSON or missing data

**Message Format:**
```python
f"Embedding generation failed: {specific_reason}"
```

**Examples:**
```python
EmbeddingError("Embedding generation failed: Ollama service unavailable at http://localhost:11434")
EmbeddingError("Embedding generation failed: Timeout after 5000ms")
EmbeddingError("Embedding generation failed: Model 'nomic-embed-text' not found")
EmbeddingError("Embedding generation failed: Expected 768 dimensions, got 384")
```

**Recovery Strategy:**
- **Retry with exponential backoff:** Up to 3 attempts for transient errors (timeout, network)
- **Do NOT retry for:** Model not found (permanent error)
- **Caller should propagate:** After max retries, escalate to user
- **Logging:** Log at ERROR level with full traceback

**Retry Logic:**
```python
max_retries = 3
for attempt in range(max_retries):
    try:
        embedding = await self.embedder.embed([query_text])
        break
    except EmbeddingError as e:
        if attempt == max_retries - 1:
            raise  # Final attempt failed, propagate
        await asyncio.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
```

---

### SearchError

**Module:** `zapomni_core.exceptions`

**Base Class:** `ZapomniCoreError`

**When Raised:**
1. **Database connection lost:** FalkorDB unavailable during query
2. **Query execution failed:** Cypher syntax error or runtime error
3. **Vector index not found:** HNSW index missing in database
4. **Transaction failure:** Database transaction rollback

**Message Format:**
```python
f"Search operation failed: {specific_reason}"
```

**Examples:**
```python
SearchError("Search operation failed: Database connection lost")
SearchError("Search operation failed: Vector index 'chunk_embedding_idx' not found")
SearchError("Search operation failed: Cypher query execution error: {cypher_error}")
```

**Recovery Strategy:**
- **Retry with exponential backoff:** Up to 3 attempts for connection errors
- **Do NOT retry for:** Cypher syntax errors (permanent)
- **Caller should propagate:** After max retries
- **Logging:** Log at ERROR level with query details

---

### TimeoutError

**Module:** `asyncio` (Python standard library)

**When Raised:**
1. **Search exceeds 1000ms:** Total operation time (embedding + query + processing) > 1000ms
2. **Database query hangs:** FalkorDB query takes too long

**Message Format:**
```python
"Search operation timed out after 1000ms"
```

**Recovery Strategy:**
- **Retry once:** Single retry attempt (timeout may be transient)
- **Simplify query:** Consider reducing `limit` or adding more filters
- **Caller should propagate:** After retry
- **Logging:** Log at WARNING level (may indicate performance issue)

**Timeout Implementation:**
```python
async def search(...) -> List[SearchResult]:
    start_time = time.time()

    try:
        async with asyncio.timeout(1.0):  # 1000ms timeout
            # ... search logic ...
    except asyncio.TimeoutError:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.warning(
            "search_timeout",
            query_length=len(query_text),
            limit=limit,
            elapsed_ms=elapsed_ms
        )
        raise TimeoutError(f"Search operation timed out after {elapsed_ms:.0f}ms")
```

---

## Algorithm (Pseudocode)

```
FUNCTION search(query_text, limit, filters, min_similarity):
    # ===== STEP 1: Input Validation =====
    start_time = current_timestamp()

    # 1.1: Validate query_text
    query_text = query_text.strip()
    IF query_text is empty:
        RAISE ValidationError("query_text cannot be empty")

    IF length(query_text) > 10,000:
        RAISE ValidationError(f"query_text exceeds max length: {len}")

    # 1.2: Validate limit
    IF limit < 1 OR limit > 100:
        RAISE ValidationError(f"limit must be in range [1, 100]: got {limit}")

    # 1.3: Validate min_similarity
    IF min_similarity < 0.0 OR min_similarity > 1.0:
        RAISE ValidationError(f"min_similarity must be in [0.0, 1.0]: got {min_similarity}")

    # 1.4: Validate filters (if provided)
    IF filters is not None:
        VALIDATE_FILTERS(filters)  # See "filters" parameter section

    LOG_INFO("search_started", query_length=len(query_text), limit=limit)

    # ===== STEP 2: Generate Query Embedding =====
    TRY:
        # Call OllamaEmbedder with retry logic
        query_embeddings = await embedder.embed(
            texts=[query_text],
            timeout=5000  # 5 second timeout for embedding
        )
        query_vector = query_embeddings[0]  # Extract single embedding

        # Verify embedding dimensions
        IF length(query_vector) != EXPECTED_DIM (768 for nomic-embed-text):
            RAISE EmbeddingError(f"Wrong embedding dimensions: expected 768, got {len}")

    CATCH OllamaConnectionError:
        LOG_ERROR("embedding_failed", reason="Ollama unavailable")
        RAISE EmbeddingError("Embedding generation failed: Ollama service unavailable")

    CATCH OllamaTimeoutError:
        LOG_ERROR("embedding_timeout", query_length=len(query_text))
        RAISE EmbeddingError("Embedding generation failed: Timeout after 5000ms")

    embedding_time_ms = (current_timestamp() - start_time) * 1000
    LOG_DEBUG("embedding_generated", latency_ms=embedding_time_ms)

    # ===== STEP 3: Build Cypher Query =====
    # Base query with vector similarity
    cypher_query = """
        MATCH (c:Chunk)
        WHERE c.embedding IS NOT NULL
        WITH c, vector.similarity(c.embedding, $query_vector) AS similarity
        WHERE similarity >= $min_similarity
    """

    # Dictionary to hold query parameters
    params = {
        "query_vector": query_vector,
        "min_similarity": min_similarity,
        "limit": limit
    }

    # ===== STEP 4: Apply Metadata Filters =====
    IF filters is not None:
        # 4.1: Apply tags filter (AND logic)
        IF "tags" in filters AND filters["tags"]:
            cypher_query += """
                AND ALL(tag IN $tags WHERE tag IN c.tags)
            """
            params["tags"] = filters["tags"]

        # 4.2: Apply source filter (exact match)
        IF "source" in filters:
            cypher_query += """
                AND c.source = $source
            """
            params["source"] = filters["source"]

        # 4.3: Apply date_from filter (inclusive)
        IF "date_from" in filters:
            cypher_query += """
                AND c.timestamp >= $date_from
            """
            params["date_from"] = filters["date_from"]

        # 4.4: Apply date_to filter (inclusive)
        IF "date_to" in filters:
            cypher_query += """
                AND c.timestamp <= $date_to
            """
            params["date_to"] = filters["date_to"]

    # ===== STEP 5: Complete Query with Ordering and Limit =====
    cypher_query += """
        RETURN c.memory_id AS memory_id,
               c.text AS chunk_text,
               similarity AS similarity_score,
               c.metadata AS metadata,
               c.chunk_index AS chunk_index
        ORDER BY similarity DESC, c.memory_id ASC
        LIMIT $limit
    """

    LOG_DEBUG("cypher_query_built", has_filters=filters is not None)

    # ===== STEP 6: Execute Query in FalkorDB =====
    TRY:
        # Execute with timeout
        db_results = await db_client.execute_query(
            query=cypher_query,
            params=params,
            timeout_ms=1000  # 1 second timeout
        )

    CATCH DatabaseConnectionError:
        LOG_ERROR("database_connection_lost")
        RAISE SearchError("Search operation failed: Database connection lost")

    CATCH CypherQueryError as e:
        LOG_ERROR("cypher_query_failed", error=str(e), query=cypher_query)
        RAISE SearchError(f"Search operation failed: Cypher error: {e}")

    CATCH DatabaseTimeoutError:
        LOG_WARNING("database_timeout", query_length=len(query_text), limit=limit)
        RAISE TimeoutError("Search operation timed out: Database query exceeded 1000ms")

    query_time_ms = (current_timestamp() - start_time) * 1000
    LOG_DEBUG("database_query_completed", latency_ms=query_time_ms, result_count=len(db_results))

    # ===== STEP 7: Convert DB Results to SearchResult Objects =====
    search_results = []

    FOR row IN db_results:
        # Create SearchResult from database row
        result = SearchResult(
            memory_id=row["memory_id"],
            chunk_text=row["chunk_text"],
            similarity_score=row["similarity_score"],
            metadata=row["metadata"],
            chunk_index=row["chunk_index"]
        )
        search_results.append(result)

    # ===== STEP 8: Validate Results (Sanity Checks) =====
    # Ensure no duplicates (should be guaranteed by DB, but verify)
    seen_keys = set()
    FOR result IN search_results:
        key = (result.memory_id, result.chunk_index)
        IF key IN seen_keys:
            LOG_WARNING("duplicate_result_detected", memory_id=result.memory_id, chunk_index=result.chunk_index)
            # Remove duplicate (keep first occurrence)
            search_results.remove(result)
        ELSE:
            seen_keys.add(key)

    # ===== STEP 9: Log Performance Metrics & Return =====
    total_time_ms = (current_timestamp() - start_time) * 1000

    LOG_INFO(
        "search_completed",
        query_length=len(query_text),
        result_count=len(search_results),
        total_latency_ms=total_time_ms,
        embedding_latency_ms=embedding_time_ms,
        query_latency_ms=query_time_ms,
        has_filters=filters is not None
    )

    # Performance warning if too slow
    IF total_time_ms > 500:
        LOG_WARNING(
            "search_slow",
            latency_ms=total_time_ms,
            query_length=len(query_text),
            limit=limit
        )

    RETURN search_results

END FUNCTION


# Helper function: Validate filters structure
FUNCTION VALIDATE_FILTERS(filters):
    IF NOT isinstance(filters, dict):
        RAISE ValidationError("filters must be a dictionary")

    # Validate tags
    IF "tags" in filters:
        tags = filters["tags"]
        IF NOT isinstance(tags, list):
            RAISE ValidationError("filters['tags'] must be a list")
        IF length(tags) == 0:
            RAISE ValidationError("filters['tags'] cannot be empty list")
        FOR tag IN tags:
            IF NOT isinstance(tag, str):
                RAISE ValidationError("All tags must be strings")

    # Validate source
    IF "source" in filters:
        source = filters["source"]
        IF NOT isinstance(source, str):
            RAISE ValidationError("filters['source'] must be a string")
        IF source.strip() is empty:
            RAISE ValidationError("filters['source'] cannot be empty")

    # Validate date_from
    IF "date_from" in filters:
        date_from = filters["date_from"]
        IF NOT isinstance(date_from, str):
            RAISE ValidationError("filters['date_from'] must be a string")
        TRY:
            datetime.fromisoformat(date_from)
        CATCH ValueError:
            RAISE ValidationError(f"filters['date_from'] must be ISO 8601 date: got '{date_from}'")

    # Validate date_to
    IF "date_to" in filters:
        date_to = filters["date_to"]
        IF NOT isinstance(date_to, str):
            RAISE ValidationError("filters['date_to'] must be a string")
        TRY:
            datetime.fromisoformat(date_to)
        CATCH ValueError:
            RAISE ValidationError(f"filters['date_to'] must be ISO 8601 date: got '{date_to}'")

    # Validate date range
    IF "date_from" in filters AND "date_to" in filters:
        date_from_dt = datetime.fromisoformat(filters["date_from"])
        date_to_dt = datetime.fromisoformat(filters["date_to"])
        IF date_from_dt > date_to_dt:
            RAISE ValidationError("filters['date_from'] cannot be after filters['date_to']")

END FUNCTION
```

---

## Preconditions

**Must be true before calling `search()`:**

1. ✅ **VectorSearchEngine initialized:**
   - `__init__()` successfully called
   - `self.db_client` is valid FalkorDBClient instance
   - `self.embedder` is valid OllamaEmbedder instance

2. ✅ **FalkorDB available:**
   - Database server running and accessible
   - Connection established (may reconnect automatically)
   - Vector index exists (created during `add_memory` operations)

3. ✅ **Ollama service available:**
   - Ollama server running at configured host
   - Embedding model pulled (e.g., `nomic-embed-text`)
   - Network connectivity to Ollama

4. ✅ **Knowledge base not empty (for meaningful results):**
   - At least one memory with embedded chunks exists in FalkorDB
   - Not a hard requirement (returns empty list if no data)

**Not required (optional):**
- 🔹 Existing search history
- 🔹 Pre-warmed caches
- 🔹 Specific database schema version (handled by migrations)

---

## Postconditions

**Guaranteed to be true after successful `search()` completion:**

1. ✅ **Valid return value:**
   - Returns `List[SearchResult]` (never None)
   - List length: 0 to `limit` (inclusive)
   - All results sorted by `similarity_score` descending

2. ✅ **Data integrity:**
   - No duplicate `(memory_id, chunk_index)` tuples
   - All `similarity_score` values >= `min_similarity`
   - All `SearchResult` fields populated (no None values)

3. ✅ **Filters applied (if provided):**
   - All results match filter criteria
   - AND logic for multiple filters

4. ✅ **Performance logged:**
   - Latency metrics recorded
   - Slow queries logged (> 500ms)

5. ✅ **No side effects:**
   - Database state unchanged (read-only query)
   - No caching state modified (Phase 1)
   - No resource leaks (connections closed)

**On failure (exception raised):**
- ❌ No partial results returned
- ❌ Exception propagated to caller
- ❌ Error logged with context
- ✅ Resources cleaned up (connections, timeouts)

---

## Edge Cases & Handling

### Edge Case 1: Empty query_text

**Scenario:** User passes empty string `""`

**Expected Behavior:**
```python
raise ValidationError("Validation failed: query_text cannot be empty")
```

**Why It's an Edge Case:**
- Cannot generate meaningful embedding for empty text
- Ollama would reject empty input
- No semantic meaning to search for

**Test Scenario:**
```python
async def test_search_empty_query_raises():
    """Test that empty query_text raises ValidationError."""
    engine = VectorSearchEngine(db_client=mock_db, embedder=mock_embedder)

    with pytest.raises(ValidationError, match="query_text cannot be empty"):
        await engine.search(query_text="", limit=10)
```

---

### Edge Case 2: Query with Only Whitespace

**Scenario:** User passes whitespace-only string `"   \n\t   "`

**Expected Behavior:**
```python
# After strip(), becomes empty
raise ValidationError("Validation failed: query_text cannot be empty")
```

**Why It's an Edge Case:**
- Common user input mistake
- Must strip before validation
- Equivalent to empty query semantically

**Test Scenario:**
```python
async def test_search_whitespace_query_raises():
    """Test that whitespace-only query raises ValidationError."""
    engine = VectorSearchEngine(db_client=mock_db, embedder=mock_embedder)

    with pytest.raises(ValidationError, match="query_text cannot be empty"):
        await engine.search(query_text="   \n\t   ", limit=10)
```

---

### Edge Case 3: Extremely Long query_text (> 10,000 chars)

**Scenario:** User passes query with 15,000 characters

**Expected Behavior:**
```python
raise ValidationError(
    "Validation failed: query_text exceeds maximum length (10,000 chars): got 15000"
)
```

**Why It's an Edge Case:**
- Prevents Ollama timeout (very long text takes minutes to embed)
- Prevents memory exhaustion
- User likely pasted entire document by mistake

**Test Scenario:**
```python
async def test_search_long_query_raises():
    """Test that query > 10,000 chars raises ValidationError."""
    engine = VectorSearchEngine(db_client=mock_db, embedder=mock_embedder)

    huge_query = "x" * 15_000

    with pytest.raises(ValidationError, match="exceeds maximum length"):
        await engine.search(query_text=huge_query, limit=10)
```

---

### Edge Case 4: limit = 0

**Scenario:** User passes `limit=0`

**Expected Behavior:**
```python
raise ValidationError("Validation failed: limit must be at least 1: got 0")
```

**Why It's an Edge Case:**
- No point in searching if zero results requested
- May indicate caller logic bug
- Prevents unnecessary database query

**Test Scenario:**
```python
async def test_search_zero_limit_raises():
    """Test that limit=0 raises ValidationError."""
    engine = VectorSearchEngine(db_client=mock_db, embedder=mock_embedder)

    with pytest.raises(ValidationError, match="limit must be at least 1"):
        await engine.search(query_text="test", limit=0)
```

---

### Edge Case 5: limit > 100

**Scenario:** User passes `limit=500`

**Expected Behavior:**
```python
raise ValidationError("Validation failed: limit cannot exceed 100: got 500")
```

**Why It's an Edge Case:**
- Prevents excessive memory usage (500 × 1KB chunks = 500KB per search)
- Prevents slow queries (retrieving hundreds of results)
- User should use pagination instead

**Test Scenario:**
```python
async def test_search_large_limit_raises():
    """Test that limit > 100 raises ValidationError."""
    engine = VectorSearchEngine(db_client=mock_db, embedder=mock_embedder)

    with pytest.raises(ValidationError, match="limit cannot exceed 100"):
        await engine.search(query_text="test", limit=500)
```

---

### Edge Case 6: min_similarity = 1.1 (out of range)

**Scenario:** User passes `min_similarity=1.1`

**Expected Behavior:**
```python
raise ValidationError("Validation failed: min_similarity cannot exceed 1.0: got 1.1")
```

**Why It's an Edge Case:**
- Cosine similarity range is [0.0, 1.0]
- Value > 1.0 is mathematically impossible
- May indicate caller misunderstanding (e.g., using percentage 110% instead of 1.1)

**Test Scenario:**
```python
async def test_search_invalid_min_similarity_raises():
    """Test that min_similarity > 1.0 raises ValidationError."""
    engine = VectorSearchEngine(db_client=mock_db, embedder=mock_embedder)

    with pytest.raises(ValidationError, match="min_similarity cannot exceed 1.0"):
        await engine.search(query_text="test", limit=10, min_similarity=1.1)
```

---

### Edge Case 7: No Results Above min_similarity

**Scenario:** Query "xyz123abc" with `min_similarity=0.99` returns zero matches

**Expected Behavior:**
```python
# Returns empty list (not an error)
results = []
assert len(results) == 0
```

**Why It's an Edge Case:**
- Common scenario with high thresholds or rare queries
- NOT an error condition (valid result)
- Caller should handle gracefully

**Test Scenario:**
```python
async def test_search_no_results_returns_empty_list():
    """Test that no matches returns empty list (not error)."""
    mock_db.execute_query = AsyncMock(return_value=[])  # No results from DB
    mock_embedder.embed = AsyncMock(return_value=[[0.1] * 768])

    engine = VectorSearchEngine(db_client=mock_db, embedder=mock_embedder)

    results = await engine.search(
        query_text="nonexistent_query_xyz",
        limit=10,
        min_similarity=0.99
    )

    assert results == []
    assert len(results) == 0
```

---

### Edge Case 8: Ollama Service Unavailable

**Scenario:** Ollama server is down or unreachable during `embedder.embed()` call

**Expected Behavior:**
```python
# After 3 retry attempts with exponential backoff
raise EmbeddingError(
    "Embedding generation failed: Ollama service unavailable at http://localhost:11434"
)
```

**Why It's an Edge Case:**
- Transient failure (Ollama crashed or network issue)
- Requires retry logic before failing
- Caller should escalate to user

**Test Scenario:**
```python
async def test_search_ollama_unavailable_raises():
    """Test that Ollama unavailable raises EmbeddingError after retries."""
    mock_embedder.embed = AsyncMock(
        side_effect=OllamaConnectionError("Connection refused")
    )

    engine = VectorSearchEngine(db_client=mock_db, embedder=mock_embedder)

    with pytest.raises(EmbeddingError, match="Ollama service unavailable"):
        await engine.search(query_text="test", limit=10)

    # Verify retry attempts (3 calls)
    assert mock_embedder.embed.call_count == 3
```

---

### Edge Case 9: Database Connection Lost During Query

**Scenario:** FalkorDB connection drops while executing Cypher query

**Expected Behavior:**
```python
# After 3 retry attempts
raise SearchError("Search operation failed: Database connection lost")
```

**Why It's an Edge Case:**
- Transient failure (database restarted, network issue)
- Requires retry logic
- Different from permanent errors (e.g., syntax error)

**Test Scenario:**
```python
async def test_search_database_connection_lost_raises():
    """Test that DB connection lost raises SearchError after retries."""
    mock_db.execute_query = AsyncMock(
        side_effect=DatabaseConnectionError("Connection reset by peer")
    )
    mock_embedder.embed = AsyncMock(return_value=[[0.1] * 768])

    engine = VectorSearchEngine(db_client=mock_db, embedder=mock_embedder)

    with pytest.raises(SearchError, match="Database connection lost"):
        await engine.search(query_text="test", limit=10)

    # Verify retry attempts
    assert mock_db.execute_query.call_count == 3
```

---

### Edge Case 10: Search Timeout (> 1000ms)

**Scenario:** Total search operation takes 1200ms (exceeds 1000ms timeout)

**Expected Behavior:**
```python
raise TimeoutError("Search operation timed out after 1200ms")
```

**Why It's an Edge Case:**
- Performance degradation (database slow or overloaded)
- Prevents hanging requests
- Indicates need for optimization

**Test Scenario:**
```python
async def test_search_timeout_raises():
    """Test that search > 1000ms raises TimeoutError."""
    async def slow_query(*args, **kwargs):
        await asyncio.sleep(1.5)  # Simulate 1500ms query
        return []

    mock_db.execute_query = AsyncMock(side_effect=slow_query)
    mock_embedder.embed = AsyncMock(return_value=[[0.1] * 768])

    engine = VectorSearchEngine(db_client=mock_db, embedder=mock_embedder)

    with pytest.raises(TimeoutError, match="timed out"):
        await engine.search(query_text="test", limit=10)
```

---

### Edge Case 11: filters with Empty tags List

**Scenario:** User passes `filters={"tags": []}`

**Expected Behavior:**
```python
raise ValidationError("Validation failed: filters['tags'] cannot be empty list")
```

**Why It's an Edge Case:**
- Empty tags list has no filtering effect (ambiguous intent)
- User likely meant to omit tags key entirely
- Prevents confusion

**Test Scenario:**
```python
async def test_search_empty_tags_list_raises():
    """Test that empty tags list raises ValidationError."""
    engine = VectorSearchEngine(db_client=mock_db, embedder=mock_embedder)

    with pytest.raises(ValidationError, match="tags cannot be empty list"):
        await engine.search(
            query_text="test",
            limit=10,
            filters={"tags": []}
        )
```

---

### Edge Case 12: filters with date_from > date_to

**Scenario:** User passes `filters={"date_from": "2024-12-31", "date_to": "2024-01-01"}`

**Expected Behavior:**
```python
raise ValidationError(
    "Validation failed: filters['date_from'] cannot be after filters['date_to']"
)
```

**Why It's an Edge Case:**
- Invalid date range (from after to)
- Will always return zero results
- Likely user mistake (swapped dates)

**Test Scenario:**
```python
async def test_search_invalid_date_range_raises():
    """Test that date_from > date_to raises ValidationError."""
    engine = VectorSearchEngine(db_client=mock_db, embedder=mock_embedder)

    with pytest.raises(ValidationError, match="date_from cannot be after date_to"):
        await engine.search(
            query_text="test",
            limit=10,
            filters={
                "date_from": "2024-12-31",
                "date_to": "2024-01-01"
            }
        )
```

---

## Test Scenarios (Complete List)

### Happy Path Tests

#### 1. test_search_success_minimal
**Input:**
- query_text: `"What is Python?"`
- limit: `10`
- filters: `None`
- min_similarity: `0.5` (default)

**Expected:**
- Returns `List[SearchResult]` with 1-10 results
- All results have `similarity_score >= 0.5`
- Sorted by similarity descending
- `embedder.embed()` called once
- `db_client.execute_query()` called once

**Mock Setup:**
```python
mock_embedder.embed.return_value = [[0.1] * 768]
mock_db.execute_query.return_value = [
    {
        "memory_id": "uuid-1",
        "chunk_text": "Python is a programming language",
        "similarity_score": 0.95,
        "metadata": {"tags": ["python"]},
        "chunk_index": 0
    }
]
```

---

#### 2. test_search_success_with_filters
**Input:**
- query_text: `"machine learning"`
- limit: `20`
- filters: `{"tags": ["AI", "ML"], "source": "research"}`
- min_similarity: `0.6`

**Expected:**
- Returns filtered results matching tags AND source
- Cypher query includes filter clauses
- All results have both tags "AI" and "ML"
- All results have source = "research"

---

#### 3. test_search_success_boundary_limit
**Input:**
- query_text: `"test"`
- limit: `1` (minimum)
- min_similarity: `0.0`

**Expected:**
- Returns exactly 1 result (highest similarity)
- Single best match

**Input:**
- limit: `100` (maximum)

**Expected:**
- Returns up to 100 results
- All available results if < 100 matches

---

#### 4. test_search_success_high_threshold
**Input:**
- query_text: `"Python async"`
- min_similarity: `0.9` (very high)

**Expected:**
- Returns only very high similarity results
- May return 0 results (valid)
- All results have similarity >= 0.9

---

#### 5. test_search_success_date_filters
**Input:**
- filters: `{"date_from": "2024-01-01", "date_to": "2024-12-31"}`

**Expected:**
- Returns results within date range (inclusive)
- Cypher includes timestamp comparisons

---

### Error Tests

#### 6. test_search_empty_query_raises
**Input:**
- query_text: `""`

**Expected:**
- Raises `ValidationError` with message "query_text cannot be empty"
- No embedder call
- No database call

---

#### 7. test_search_whitespace_query_raises
**Input:**
- query_text: `"   \n\t   "`

**Expected:**
- Raises `ValidationError` (after strip becomes empty)

---

#### 8. test_search_long_query_raises
**Input:**
- query_text: `"x" * 15_000` (15,000 chars)

**Expected:**
- Raises `ValidationError` with message "exceeds maximum length"

---

#### 9. test_search_zero_limit_raises
**Input:**
- limit: `0`

**Expected:**
- Raises `ValidationError` with message "limit must be at least 1"

---

#### 10. test_search_large_limit_raises
**Input:**
- limit: `101`

**Expected:**
- Raises `ValidationError` with message "limit cannot exceed 100"

---

#### 11. test_search_negative_limit_raises
**Input:**
- limit: `-5`

**Expected:**
- Raises `ValidationError`

---

#### 12. test_search_invalid_min_similarity_raises
**Input:**
- min_similarity: `1.5` (> 1.0)

**Expected:**
- Raises `ValidationError` with message "min_similarity cannot exceed 1.0"

**Input:**
- min_similarity: `-0.1` (< 0.0)

**Expected:**
- Raises `ValidationError` with message "min_similarity cannot be negative"

---

#### 13. test_search_invalid_filters_structure_raises
**Input:**
- filters: `"not_a_dict"`

**Expected:**
- Raises `ValidationError` with message "filters must be a dictionary"

---

#### 14. test_search_empty_tags_list_raises
**Input:**
- filters: `{"tags": []}`

**Expected:**
- Raises `ValidationError` with message "tags cannot be empty list"

---

#### 15. test_search_invalid_date_format_raises
**Input:**
- filters: `{"date_from": "01-01-2024"}` (not ISO 8601)

**Expected:**
- Raises `ValidationError` with message "must be ISO 8601 date"

---

#### 16. test_search_invalid_date_range_raises
**Input:**
- filters: `{"date_from": "2024-12-31", "date_to": "2024-01-01"}`

**Expected:**
- Raises `ValidationError` with message "date_from cannot be after date_to"

---

### Dependency Failure Tests

#### 17. test_search_ollama_unavailable_raises
**Mock:**
- `embedder.embed()` raises `OllamaConnectionError`

**Expected:**
- Raises `EmbeddingError` with message "Ollama service unavailable"
- Retry 3 times before failing
- `embedder.embed.call_count == 3`

---

#### 18. test_search_embedding_timeout_raises
**Mock:**
- `embedder.embed()` times out after 5 seconds

**Expected:**
- Raises `EmbeddingError` with message "Timeout after 5000ms"

---

#### 19. test_search_database_connection_lost_raises
**Mock:**
- `db_client.execute_query()` raises `DatabaseConnectionError`

**Expected:**
- Raises `SearchError` with message "Database connection lost"
- Retry 3 times

---

#### 20. test_search_cypher_error_raises
**Mock:**
- `db_client.execute_query()` raises `CypherQueryError`

**Expected:**
- Raises `SearchError` with message "Cypher error"
- No retry (permanent error)

---

#### 21. test_search_timeout_raises
**Mock:**
- Total operation takes > 1000ms

**Expected:**
- Raises `TimeoutError` with message "timed out after 1000ms"

---

### Integration Tests (With Real Dependencies)

#### 22. test_search_with_real_falkordb
**Setup:**
- Real FalkorDB instance with test data
- 100 embedded chunks in database

**Input:**
- query_text: `"Python programming"`
- limit: `10`

**Expected:**
- Returns actual search results
- Latency < 200ms (P95 target)
- Results semantically relevant

---

#### 23. test_search_with_real_ollama
**Setup:**
- Real Ollama service with nomic-embed-text model

**Input:**
- query_text: `"machine learning"`

**Expected:**
- Real embedding generated (768 dimensions)
- Embedding time < 100ms

---

#### 24. test_search_end_to_end_performance
**Setup:**
- Real FalkorDB + Ollama
- 10,000 chunks in database

**Input:**
- query_text: `"async programming patterns"`
- limit: `10`

**Expected:**
- P50 < 100ms
- P95 < 200ms
- P99 < 500ms
- Results accurate (manually verified)

---

### Performance Tests

#### 25. test_search_performance_within_sla
**Input:**
- Normal query (100 chars)
- limit: `10`

**Expected:**
- Total latency < 200ms (P95 target)
- Embedding latency < 100ms
- Query latency < 100ms

---

#### 26. test_search_large_input_performance
**Input:**
- query_text: `"x" * 9_999` (near max length)

**Expected:**
- Still completes (no timeout)
- Latency < 500ms (P99 target)

---

#### 27. test_search_concurrent_throughput
**Setup:**
- 100 concurrent search requests

**Expected:**
- All complete successfully
- No errors
- Average latency < 300ms
- Throughput >= 100 searches/sec

---

## Performance Requirements

### Latency Targets

**Normal Input (< 1KB query):**
- **P50:** < 50ms
- **P95:** < 100ms
- **P99:** < 200ms

**Large Input (1KB - 10KB query):**
- **P50:** < 100ms
- **P95:** < 200ms
- **P99:** < 500ms

**Maximum Allowed:**
- **Timeout:** 1000ms (1 second)

### Latency Breakdown

**For 10K document corpus:**

| Operation | P50 | P95 | P99 |
|-----------|-----|-----|-----|
| Input validation | 1ms | 2ms | 5ms |
| Query embedding (Ollama) | 30ms | 80ms | 150ms |
| Vector search (FalkorDB) | 20ms | 50ms | 100ms |
| Result marshalling | 5ms | 10ms | 20ms |
| **Total** | **56ms** | **142ms** | **275ms** |

**Bottlenecks:**
1. **Embedding generation:** Dominant factor (50-60% of latency)
2. **Vector search:** Second factor (30-40% of latency)
3. **Validation + marshalling:** Negligible (< 10%)

### Throughput

**Concurrent Searches:**
- **Target:** 100 searches/sec
- **Bottleneck:** Ollama embedding throughput
- **Scaling:** Horizontal (multiple Ollama instances)

### Resource Usage

**Memory:**
- **Per query:** ~5KB (query embedding + result objects)
- **Peak:** 500KB for 100 concurrent searches
- **Limit:** 10MB max (prevents OOM)

**CPU:**
- **Vector search:** O(k * log(N)) where k=limit, N=corpus_size
- **Embedding:** Offloaded to Ollama (GPU)

**Network:**
- **Ollama:** 10KB request + 5KB response per query
- **FalkorDB:** 5KB request + 50KB response (for limit=10)

---

## Security Considerations

### Input Validation

✅ **All inputs validated before use:**
- query_text: Length, encoding, non-empty
- limit: Range check [1, 100]
- min_similarity: Range check [0.0, 1.0]
- filters: Structure, types, date formats

✅ **No injection vulnerabilities:**
- Cypher queries use parameterized queries (no string concatenation)
- All user inputs passed as `$params` to prevent injection

**Example safe query:**
```python
# SAFE: Parameterized query
cypher_query = "MATCH (c:Chunk) WHERE c.text CONTAINS $search_term"
params = {"search_term": user_input}  # User input passed as parameter
```

**Example unsafe query (NOT used):**
```python
# UNSAFE: String concatenation (vulnerable to injection)
cypher_query = f"MATCH (c:Chunk) WHERE c.text CONTAINS '{user_input}'"  # ❌ NEVER DO THIS
```

✅ **Safe error messages (no sensitive data leaked):**
- Errors do NOT include query embeddings (768 floats)
- Errors do NOT include database connection strings
- Errors do NOT include chunk content (only count)

**Example safe error:**
```python
# SAFE
raise SearchError("Database connection lost")

# UNSAFE (leaks sensitive data)
raise SearchError(f"DB error at {db_host}:{db_password}@{db_port}")  # ❌
```

### Data Protection

**Sensitive data in parameters:**
- **query_text:** May contain PII (user's search query)
  - ✅ Logged only in DEBUG mode (disabled in production)
  - ✅ Not included in error messages
  - ✅ Embeddings not logged (too large, not human-readable)

- **filters:** May contain sensitive metadata
  - ✅ Logged only in DEBUG mode
  - ✅ Not propagated in exceptions

**Logging restrictions:**
```python
# Production logging (INFO level)
logger.info(
    "search_completed",
    query_length=len(query_text),  # Safe: only length
    result_count=len(results),      # Safe: only count
    latency_ms=total_time_ms
)

# Debug logging (disabled in production)
logger.debug(
    "search_debug",
    query_text=query_text[:100],  # Truncated for safety
    filters=filters
)
```

### Rate Limiting

**Not implemented in Phase 1** (handled by MCP server layer)

**Future consideration (Phase 2):**
- Per-user rate limiting (e.g., 100 searches/hour)
- IP-based rate limiting
- Exponential backoff for repeated failures

---

## Related Functions

### Calls (Dependencies)

#### 1. `embedder.embed(texts: List[str]) -> List[List[float]]`
**Purpose:** Generate embedding for query text

**Called from:** Step 2 of algorithm

**Example:**
```python
query_embeddings = await self.embedder.embed([query_text])
query_vector = query_embeddings[0]  # Extract single embedding (768 dims)
```

**Error handling:**
- Wraps `OllamaConnectionError` → `EmbeddingError`
- Wraps `OllamaTimeoutError` → `EmbeddingError`
- Retries 3 times with exponential backoff

---

#### 2. `db_client.execute_query(query: str, params: dict, timeout_ms: int) -> List[dict]`
**Purpose:** Execute Cypher query with vector similarity search

**Called from:** Step 6 of algorithm

**Example:**
```python
results = await self.db_client.execute_query(
    query=cypher_query,
    params={
        "query_vector": query_vector,
        "min_similarity": min_similarity,
        "limit": limit
    },
    timeout_ms=1000
)
```

**Error handling:**
- Wraps `DatabaseConnectionError` → `SearchError`
- Wraps `CypherQueryError` → `SearchError`
- Wraps `DatabaseTimeoutError` → `TimeoutError`
- Retries 3 times for connection errors (not syntax errors)

---

### Called By (Dependents)

#### 1. `MemoryProcessor.search_memory(query: str, mode: str, limit: int) -> List[SearchResult]`
**Module:** `zapomni_core.processor`

**Purpose:** High-level search orchestrator that routes to appropriate search method

**Example:**
```python
# In MemoryProcessor
async def search_memory(self, query: str, mode: str = "vector", limit: int = 10):
    if mode == "vector":
        return await self.search_engine.search(
            query_text=query,
            limit=limit
        )
    elif mode == "hybrid":
        return await self.search_engine.hybrid_search(...)  # Phase 2
```

---

#### 2. `SearchMemoryTool.execute(arguments: dict) -> dict`
**Module:** `zapomni_mcp.tools.search_memory`

**Purpose:** MCP tool wrapper that exposes search to Claude

**Example:**
```python
# In SearchMemoryTool
async def execute(self, arguments: dict) -> dict:
    results = await self.core.search_engine.search(
        query_text=arguments["query"],
        limit=arguments.get("limit", 10),
        filters=arguments.get("filters")
    )

    # Format as MCP response
    return format_mcp_response(results)
```

---

## Implementation Notes

### Libraries Used

**Python Standard Library:**
- `asyncio` - Async/await support, timeout handling
- `time` - Performance timing
- `datetime` - Date validation for filters
- `typing` - Type hints (List, Dict, Optional, Any)

**External Libraries:**
- `numpy>=1.24.0` (optional) - Cosine similarity computation (if not using FalkorDB built-in)
  - Used for: Vector normalization, dot product
  - Alternative: FalkorDB's `vector.similarity()` function (preferred)

**Internal Dependencies:**
- `zapomni_core.embeddings.OllamaEmbedder` - Query embedding generation
- `zapomni_db.falkordb.FalkorDBClient` - Vector database operations
- `zapomni_core.models.SearchResult` - Result data structure
- `zapomni_core.exceptions` - Custom exceptions (ValidationError, etc.)

### Known Limitations

1. **Text-only search:**
   - Cannot handle binary data in query_text (UTF-8 only)
   - No support for image or audio queries (future: multimodal embeddings)

2. **Single embedding model:**
   - Tied to nomic-embed-text (768 dimensions)
   - Changing model requires reindexing entire corpus

3. **No query caching (Phase 1):**
   - Repeated identical queries regenerate embeddings
   - Phase 2: Add semantic cache for common queries

4. **Synchronous embedding:**
   - One query at a time to Ollama
   - Phase 2: Batch embedding for multiple concurrent queries

5. **Limited filter operators:**
   - Tags: Only AND logic (all tags must match)
   - No OR, NOT, or complex boolean expressions
   - Dates: Only inclusive range (>=, <=)

6. **No pagination:**
   - Must fetch all results in single call
   - Maximum 100 results enforced
   - Phase 2: Add cursor-based pagination

### Future Enhancements

**Phase 2 (Hybrid Search):**
- Combine with BM25 keyword search using RRF fusion
- Enable `alpha` parameter to balance vector vs. keyword

**Phase 2 (Semantic Caching):**
- Cache query embeddings for repeated queries
- TTL: 1 hour
- Hit rate target: 30-40%

**Phase 3 (Advanced Features):**
- Streaming results (yield results as they arrive)
- Async batch embedding (process multiple queries in parallel)
- Complex filter expressions (OR, NOT, nested conditions)
- Relevance feedback (learn from user clicks)
- Personalized search (user-specific embeddings)

---

## References

### Parent Specifications
- **Component Spec:** [vector_search_engine_component.md](../level2/vector_search_engine_component.md)
- **Module Spec:** [zapomni_core_module.md](../level1/zapomni_core_module.md)

### Related Function Specs
- `VectorSearchEngine.hybrid_search()` (Phase 2, Level 3)
- `VectorSearchEngine.rerank_results()` (Phase 2, Level 3)
- `OllamaEmbedder.embed()` (zapomni_core, Level 3)
- `FalkorDBClient.execute_query()` (zapomni_db, Level 3)

### External Documentation
- **FalkorDB Vector Search:** https://docs.falkordb.com/vector-search.html
- **Cosine Similarity:** https://en.wikipedia.org/wiki/Cosine_similarity
- **nomic-embed-text Model:** https://ollama.com/library/nomic-embed-text
- **HNSW Index:** https://arxiv.org/abs/1603.09320

### Research References
- **Hybrid RAG Best Practices:** [03_best_practices_patterns.md](/home/dev/zapomni/research/03_best_practices_patterns.md)
- **MTEB Benchmark:** https://huggingface.co/spaces/mteb/leaderboard
- **Vector Search Optimization:** https://www.pinecone.io/learn/vector-search/

---

**Document Status:** Draft v1.0
**Created:** 2025-11-23
**Author:** Goncharenko Anton aka alienxs2
**Copyright:** Copyright (c) 2025 Goncharenko Anton aka alienxs2
**License:** MIT License

**Ready for Review:** Yes ✅
**Next Steps:** Multi-agent verification (Level 3 verification process)
