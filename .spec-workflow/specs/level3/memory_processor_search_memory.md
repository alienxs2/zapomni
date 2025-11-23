# MemoryProcessor.search_memory() - Function Specification

**Level:** 3 (Function)
**Component:** MemoryProcessor
**Module:** zapomni_core
**Author:** Goncharenko Anton aka alienxs2
**Status:** Draft
**Version:** 1.0
**Created:** 2025-11-23

---

## Function Signature

```python
async def search_memory(
    self,
    query: str,
    limit: int = 10,
    filters: Optional[Dict[str, Any]] = None,
    search_mode: str = "vector"
) -> List[SearchResult]:
    """
    Search memories using specified search mode.

    Executes search across stored memories and returns ranked results.
    Supports multiple search modes (vector, BM25, hybrid, graph) with
    consistent result format.

    Args:
        query: Natural language search query
            - Constraints: Non-empty, max 1000 chars
            - Format: Natural language question or keywords
            - Example: "Who created Python?", "Python web frameworks"

        limit: Maximum number of results to return
            - Constraints: 1 <= limit <= 100
            - Default: 10
            - Note: Actual results may be fewer if not enough matches

        filters: Optional metadata filters to narrow results
            - Structure: Dict with filter conditions
            - Supported filters:
                - "tags": List[str] - Match ANY of these tags
                - "source": str - Exact source match
                - "date_from": str - ISO date, >= this date
                - "date_to": str - ISO date, <= this date
            - Example: {"tags": ["python"], "source": "wikipedia"}

        search_mode: Search algorithm to use
            - Options: "vector" (Phase 1), "bm25" (Phase 2), "hybrid" (Phase 2), "graph" (Phase 2)
            - Default: "vector"
            - "vector": Semantic similarity via embeddings (cosine distance)
            - "bm25": Keyword matching via BM25 algorithm (Phase 2)
            - "hybrid": RRF fusion of vector + BM25 (Phase 2)
            - "graph": Knowledge graph traversal (Phase 2)

    Returns:
        List of SearchResult objects, sorted by similarity_score (descending)
        - Length: 0 to limit (may be fewer than limit)
        - Sorted: Highest similarity first
        - Filtered: Only results matching filters (if provided)
        - Minimum similarity: 0.5 (configurable, results below threshold excluded)

    Raises:
        ValidationError: If query empty or limit out of range [1, 100]
        ValidationError: If search_mode invalid or unsupported in current phase
        ValidationError: If filters have invalid structure
        EmbeddingError: If query embedding generation fails
        SearchError: If search operation fails (DB error, timeout)
    """
```

## Purpose & Context

### What It Does

Executes semantic search across stored memories by:
1. Validating search parameters (query, limit, filters, mode)
2. Generating query embedding via embedder
3. Executing database search (vector, BM25, hybrid, or graph mode)
4. Applying metadata filters to results
5. Formatting and returning ranked SearchResult objects

### Why It Exists

Core search functionality for Zapomni memory system. Enables users to find relevant memories using natural language queries. Critical for MCP SearchMemoryTool and interactive search features.

### When To Use

- User wants to search stored memories
- MCP client (Claude Desktop) executes search_memory tool
- Application needs to retrieve relevant context

### When NOT To Use

- Getting stats (use `get_stats()` instead)
- Retrieving specific memory by ID (use direct DB lookup)

---

## Parameters (Detailed)

### query: str

**Type:** `str`

**Purpose:** Natural language search query to find relevant memories

**Constraints:**
- Must not be empty after `.strip()`
- Maximum length: 1000 characters
- Format: Natural language text or keywords
- Encoding: Valid UTF-8

**Validation:**
```python
if not query or not query.strip():
    raise ValidationError("Query cannot be empty")
if len(query) > 1000:
    raise ValidationError("Query exceeds maximum length (1000 characters)")
```

**Examples:**
- Valid: `"Who created Python?"`
- Valid: `"web frameworks"`
- Valid: `"Python Django ORM features"`
- Invalid: `""` (empty)
- Invalid: `"   "` (whitespace only)
- Invalid: `"a" * 1001` (too long)

---

### limit: int

**Type:** `int`

**Purpose:** Maximum number of search results to return

**Constraints:**
- Must be >= 1
- Must be <= 100
- Default: 10

**Validation:**
```python
if limit < 1:
    raise ValidationError("limit must be >= 1")
if limit > 100:
    raise ValidationError("limit must be <= 100")
```

**Examples:**
- Valid: `1`, `10`, `100`
- Invalid: `0`, `-5`, `101`

---

### filters: Optional[Dict[str, Any]]

**Type:** `Optional[Dict[str, Any]]`

**Purpose:** Optional metadata filters to narrow search results

**Default:** `None` (no filtering)

**Structure (when provided):**
```python
{
    "tags": ["python", "web"],  # Match ANY tag (OR logic)
    "source": "wikipedia",      # Exact source match
    "date_from": "2025-01-01",  # >= this date (ISO format)
    "date_to": "2025-12-31"     # <= this date (ISO format)
}
```

**Validation:**
```python
if filters is not None:
    if not isinstance(filters, dict):
        raise ValidationError("filters must be a dictionary")

    # Validate known filter keys
    valid_keys = {"tags", "source", "date_from", "date_to"}
    for key in filters:
        if key not in valid_keys:
            raise ValidationError(f"Unknown filter key: {key}")

    # Validate tags filter
    if "tags" in filters:
        if not isinstance(filters["tags"], list):
            raise ValidationError("filters['tags'] must be a list")
        if not all(isinstance(t, str) for t in filters["tags"]):
            raise ValidationError("All tags must be strings")

    # Validate source filter
    if "source" in filters:
        if not isinstance(filters["source"], str):
            raise ValidationError("filters['source'] must be a string")

    # Validate date filters
    for date_key in ["date_from", "date_to"]:
        if date_key in filters:
            if not isinstance(filters[date_key], str):
                raise ValidationError(f"filters['{date_key}'] must be a string")
            # Further validation: ISO date format (YYYY-MM-DD)
```

**Examples:**
- Valid: `None`
- Valid: `{"tags": ["python"]}`
- Valid: `{"tags": ["python", "web"], "source": "docs"}`
- Valid: `{"date_from": "2025-01-01", "date_to": "2025-12-31"}`
- Invalid: `{"invalid_key": "value"}`
- Invalid: `{"tags": "not_a_list"}`

---

### search_mode: str

**Type:** `str`

**Purpose:** Search algorithm to use

**Default:** `"vector"`

**Options:**
- `"vector"`: Semantic similarity via cosine distance (Phase 1)
- `"bm25"`: Keyword matching via BM25 algorithm (Phase 2)
- `"hybrid"`: RRF fusion of vector + BM25 (Phase 2)
- `"graph"`: Knowledge graph traversal (Phase 2)

**Validation:**
```python
valid_modes = {"vector", "bm25", "hybrid", "graph"}
if search_mode not in valid_modes:
    raise ValidationError(f"Invalid search_mode: {search_mode}. Must be one of {valid_modes}")

# Phase check
if search_mode in {"bm25", "hybrid", "graph"} and not self.config.enable_hybrid_search:
    raise ValidationError(f"search_mode '{search_mode}' not available in Phase 1")
```

**Examples:**
- Valid: `"vector"`
- Valid: `"hybrid"` (Phase 2 only)
- Invalid: `"invalid_mode"`

---

## Return Value

**Type:** `List[SearchResult]`

**Structure:**
```python
@dataclass
class SearchResult:
    memory_id: str              # UUID of matching memory
    text: str                   # Chunk text
    similarity_score: float     # Relevance score (0-1)
    tags: List[str]             # Tags from metadata
    source: str                 # Source identifier
    timestamp: datetime         # When memory was created
    highlight: Optional[str]    # Optional highlighted excerpt (Phase 2)
```

**Properties:**
- Length: 0 to `limit` (may be fewer)
- Sorted: Descending by `similarity_score`
- Filtered: Only results matching `filters`
- Threshold: Results with `similarity_score < 0.5` excluded

**Examples:**
```python
[
    SearchResult(
        memory_id="uuid-1",
        text="Python was created by Guido van Rossum.",
        similarity_score=0.95,
        tags=["python", "programming"],
        source="user",
        timestamp=datetime(2025, 11, 23, 10, 0, 0),
        highlight=None
    ),
    SearchResult(
        memory_id="uuid-2",
        text="Django is a Python web framework.",
        similarity_score=0.87,
        tags=["python", "web", "django"],
        source="documentation",
        timestamp=datetime(2025, 11, 22, 15, 30, 0),
        highlight=None
    )
]
```

---

## Exceptions

### ValidationError

**When Raised:**
- Query is empty or too long
- Limit is out of range [1, 100]
- search_mode is invalid
- filters have invalid structure

**Example Messages:**
```python
"Query cannot be empty"
"Query exceeds maximum length (1000 characters)"
"limit must be >= 1"
"limit must be <= 100"
"Invalid search_mode: invalid. Must be one of {'vector', 'bm25', 'hybrid', 'graph'}"
"search_mode 'hybrid' not available in Phase 1"
"Unknown filter key: invalid_key"
```

---

### EmbeddingError

**When Raised:**
- Query embedding generation fails
- Ollama service unavailable
- Embedding model not loaded

**Handling:**
Should retry with exponential backoff (handled by embedder)

---

### SearchError

**When Raised:**
- Database search operation fails
- Query timeout (> 5 seconds)
- Index unavailable

**Handling:**
Propagate to caller (MCP tool formats as error response)

---

## Algorithm (Pseudocode)

```
FUNCTION search_memory(query, limit, filters, search_mode):
    # Step 1: Validate Input
    VALIDATE query is not empty and length <= 1000
    VALIDATE limit in range [1, 100]
    VALIDATE search_mode in allowed modes
    VALIDATE filters structure (if provided)

    # Step 2: Generate Query Embedding
    TRY:
        query_embedding = AWAIT embedder.embed([query])
        query_vector = query_embedding[0]  # Extract first (only) embedding
    CATCH EmbeddingError:
        RAISE EmbeddingError("Failed to generate query embedding")

    # Step 3: Execute Search (mode-dependent)
    IF search_mode == "vector":
        results = AWAIT db_client.vector_search(
            query_vector=query_vector,
            limit=limit,
            min_similarity=0.5
        )
    ELSE IF search_mode == "bm25":
        results = AWAIT db_client.bm25_search(query=query, limit=limit)
    ELSE IF search_mode == "hybrid":
        vector_results = AWAIT db_client.vector_search(query_vector, limit=limit*2)
        bm25_results = AWAIT db_client.bm25_search(query, limit=limit*2)
        results = rrf_merge(vector_results, bm25_results, limit)
    ELSE IF search_mode == "graph":
        results = AWAIT db_client.graph_search(query=query, limit=limit)

    # Step 4: Apply Filters
    IF filters:
        filtered_results = []
        FOR result IN results:
            IF matches_filters(result, filters):
                filtered_results.append(result)
        results = filtered_results[:limit]  # Re-apply limit after filtering

    # Step 5: Format Results
    search_results = []
    FOR result IN results:
        search_results.append(SearchResult(
            memory_id=result.memory_id,
            text=result.text,
            similarity_score=result.score,
            tags=result.metadata.get("tags", []),
            source=result.metadata.get("source", "unknown"),
            timestamp=result.metadata.get("timestamp"),
            highlight=None
        ))

    # Step 6: Return Results
    RETURN search_results
END FUNCTION
```

---

## Preconditions

- ✅ MemoryProcessor initialized with embedder and db_client
- ✅ Database contains indexed memories (may return empty if none)
- ✅ Ollama service running (for embedding generation)

---

## Postconditions

- ✅ Query embedding generated and logged
- ✅ Search executed successfully
- ✅ Results filtered and sorted
- ✅ No state changes (read-only operation)

---

## Edge Cases & Handling

### Edge Case 1: Empty Query

**Scenario:** `query = ""`

**Behavior:** Raise ValidationError

**Test:**
```python
def test_search_memory_empty_query():
    with pytest.raises(ValidationError, match="Query cannot be empty"):
        await processor.search_memory(query="", limit=10)
```

---

### Edge Case 2: Query Too Long

**Scenario:** `query = "a" * 1001`

**Behavior:** Raise ValidationError

**Test:**
```python
def test_search_memory_query_too_long():
    huge_query = "a" * 1001
    with pytest.raises(ValidationError, match="exceeds maximum length"):
        await processor.search_memory(query=huge_query, limit=10)
```

---

### Edge Case 3: Limit Out of Range

**Scenario:** `limit = 0` or `limit = 101`

**Behavior:** Raise ValidationError

**Test:**
```python
def test_search_memory_invalid_limit():
    with pytest.raises(ValidationError, match="limit must be >= 1"):
        await processor.search_memory(query="test", limit=0)

    with pytest.raises(ValidationError, match="limit must be <= 100"):
        await processor.search_memory(query="test", limit=101)
```

---

### Edge Case 4: No Matching Results

**Scenario:** Search returns 0 results

**Behavior:** Return empty list `[]`

**Test:**
```python
async def test_search_memory_no_results():
    results = await processor.search_memory(query="nonexistent query", limit=10)
    assert results == []
```

---

### Edge Case 5: Fewer Results Than Limit

**Scenario:** Database has 3 memories, limit=10

**Behavior:** Return 3 results (all available)

**Test:**
```python
async def test_search_memory_fewer_than_limit():
    # Database has only 3 memories
    results = await processor.search_memory(query="python", limit=10)
    assert len(results) <= 10
```

---

### Edge Case 6: All Results Filtered Out

**Scenario:** Search finds 10 results, but filters exclude all

**Behavior:** Return empty list `[]`

**Test:**
```python
async def test_search_memory_all_filtered():
    results = await processor.search_memory(
        query="python",
        limit=10,
        filters={"tags": ["nonexistent_tag"]}
    )
    assert results == []
```

---

## Test Scenarios (Complete List)

### Happy Path Tests

1. **test_search_memory_success**
2. **test_search_memory_with_filters**
3. **test_search_memory_sorted_by_similarity**
4. **test_search_memory_limit_enforced**

### Validation Tests

5. **test_search_memory_empty_query_raises**
6. **test_search_memory_query_too_long_raises**
7. **test_search_memory_limit_zero_raises**
8. **test_search_memory_limit_negative_raises**
9. **test_search_memory_limit_too_large_raises**
10. **test_search_memory_invalid_mode_raises**

### Edge Case Tests

11. **test_search_memory_no_results**
12. **test_search_memory_fewer_than_limit**
13. **test_search_memory_all_filtered_out**

---

## Performance Requirements

**Latency:**
- P50: < 200ms
- P95: < 500ms
- P99: < 1000ms

**Throughput:**
- 10-20 searches/sec (single process)

---

## References

- Component spec: [memory_processor_component.md](../level2/memory_processor_component.md)
- Module spec: [zapomni_core_module.md](../level1/zapomni_core_module.md)

---

**Status:** Draft v1.0
**Created:** 2025-11-23
**Author:** Goncharenko Anton aka alienxs2
**License:** MIT
