# search_memory Tool

Semantic search across stored memories using vector similarity, keyword matching, and optional filtering.

## Overview

The `search_memory` tool finds relevant information from your knowledge graph using natural language queries. It uses semantic embeddings for intelligent retrieval, understanding meaning rather than just keywords.

Search Process:

1. **Query Embedding** - Convert search query to vector embedding
2. **Vector Search** - Find most similar chunks using cosine similarity
3. **Filtering** - Apply optional metadata filters (tags, source, dates)
4. **Ranking** - Sort results by relevance score
5. **Formatting** - Return formatted results with metadata

## Tool Identifier

```
name: "search_memory"
```

## Description

"Search your personal memory graph for information. Performs semantic search to find relevant memories based on meaning, not just keyword matching. Returns ranked results with similarity scores. Use this when you need to recall previously stored information."

## Parameters

### query (Required)

**Type**: `string`
**Min Length**: 1 character
**Max Length**: 1,000 characters
**Validation**: Non-empty, not just whitespace

Natural language search query. The system understands meaning and context, not just keywords.

**Examples**:
- `"What did I learn about Python?"`
- `"How do decorators work?"`
- `"information about machine learning"`
- `"who created Python?"`
- `"Docker containers explained"`

**Validation Rules**:

```
✓ Non-empty string
✓ 1-1000 characters
✓ Natural language or keywords
✗ Empty string or whitespace only
✗ Exceeds 1000 characters
```

**Query Tips**:
- More specific queries give better results: `"Python async/await syntax"` vs `"Python"`
- Natural language queries work well: `"How do I...?"` vs `"how to..."`
- Can use both keywords and phrases: `"decorators in Python"`

### limit (Optional)

**Type**: `integer`
**Default**: `10`
**Min**: `1`
**Max**: `100`
**Validation**: Must be between 1 and 100

Maximum number of results to return. Each result is a matching chunk from memory.

**Examples**:
- `5` - Return top 5 most relevant results
- `10` - Return top 10 (default)
- `50` - Return many results for comprehensive search
- `1` - Return only the most relevant result

**Note**: More results = slower query, but more comprehensive coverage.

### filters (Optional)

**Type**: `object` (dictionary/map)
**Default**: `null` (no filtering)
**Additional Properties**: Not allowed

Optional metadata filters to narrow search results. All filter fields are optional.

#### Supported Filter Fields

**tags** (array of strings)
- Filter to memories with all specified tags
- Example: `["python", "tutorial"]` - finds memories tagged with BOTH python AND tutorial
- Empty array `[]` means no tag filtering
- Case-sensitive matching

**source** (string)
- Filter to memories from specific source
- Example: `"github:username/repo"` or `"documentation"`
- Exact string matching
- Case-sensitive

**date_from** (string, YYYY-MM-DD format)
- Only return memories created on or after this date
- Example: `"2025-01-01"` - memories from Jan 1, 2025 onward
- ISO 8601 date format
- Inclusive (date_from is included in results)

**date_to** (string, YYYY-MM-DD format)
- Only return memories created on or before this date
- Example: `"2025-12-31"` - memories until Dec 31, 2025
- ISO 8601 date format
- Inclusive (date_to is included in results)

#### Filter Examples

```json
{
  "tags": ["python"]
}
```
Returns memories tagged with "python"

```json
{
  "tags": ["python", "tutorial"]
}
```
Returns memories tagged with BOTH "python" AND "tutorial"

```json
{
  "source": "github:username/repo"
}
```
Returns memories from specific GitHub repository

```json
{
  "date_from": "2025-01-01",
  "date_to": "2025-12-31"
}
```
Returns memories created in 2025

```json
{
  "tags": ["research"],
  "source": "documentation",
  "date_from": "2025-06-01"
}
```
Complex filter: memories tagged "research" from documentation source after June 1, 2025

## Request Example

### Basic Search

```json
{
  "query": "What is machine learning?"
}
```

### Search with Limit

```json
{
  "query": "Python programming",
  "limit": 5
}
```

### Search with Filters

```json
{
  "query": "decorators",
  "limit": 10,
  "filters": {
    "tags": ["python"]
  }
}
```

### Complex Search

```json
{
  "query": "How do decorators work?",
  "limit": 20,
  "filters": {
    "tags": ["python", "patterns"],
    "source": "tutorials",
    "date_from": "2025-01-01"
  }
}
```

## Response Format

### Success Response (With Results)

Status: `isError: false`

```json
{
  "content": [
    {
      "type": "text",
      "text": "Found 3 results:\n\n1. [Score: 0.92]\nPython decorators are functions that modify other functions without permanently changing the source code...\n\n2. [Score: 0.87]\nDecorators provide a \"Pythonic\" way to wrap a function or class with another function to provide additional functionality...\n\n3. [Score: 0.81]\nYou can create your own decorators by defining a function that takes a function as an argument..."
    }
  ],
  "isError": false
}
```

### Success Response (No Results)

Status: `isError: false`

```json
{
  "content": [
    {
      "type": "text",
      "text": "No results found matching your query."
    }
  ],
  "isError": false
}
```

### Error Response

Status: `isError: true`

```json
{
  "content": [
    {
      "type": "text",
      "text": "Error: Invalid input - query: query cannot be empty or contain only whitespace"
    }
  ],
  "isError": true
}
```

## Response Fields Explained

Each search result includes:

| Field | Format | Example | Description |
|-------|--------|---------|-------------|
| **Position** | Integer | `1.` `2.` | Result number (1-indexed) |
| **Score** | Float (0-1) | `[Score: 0.92]` | Relevance score (higher = more relevant) |
| **Tags** | Comma-separated | `[Tags: python, tutorial]` | Memory tags (if any) |
| **Text Preview** | String (200 chars) | `"Decorators provide..."` | First 200 characters of matching chunk |

### Score Interpretation

- **0.95-1.00**: Excellent match - exactly what you're looking for
- **0.85-0.95**: Good match - highly relevant
- **0.70-0.85**: Fair match - related to query
- **0.50-0.70**: Weak match - tangentially related
- **< 0.50**: Poor match - not included in results

## Ranking and Scoring

Results are ranked by **cosine similarity** score:

```
Relevance = cosine(query_embedding, chunk_embedding)
```

Higher scores indicate:
- More semantically similar to your query
- Better semantic alignment
- More likely to contain the information you're looking for

### Example: Query "Python"

```
Chunks ranked by similarity:
1. "Python is a programming language" → 0.95
2. "Guido van Rossum created Python in 1991" → 0.93
3. "Python uses indentation for code blocks" → 0.91
4. "The Monty Python comedy group created sketches" → 0.45
5. "Burmese pythons are large snakes" → 0.35
```

### Ranking Algorithm

1. **Embedding**: Query converted to 768-dimensional vector
2. **Vector Search**: HNSW index finds nearest neighbors (O(log n))
3. **Cosine Similarity**: Score calculated for top candidates
4. **Sorting**: Results sorted descending by score
5. **Filtering**: Apply metadata filters (tags, source, dates)
6. **Limiting**: Return top N results (limit parameter)

## Error Codes

### Validation Errors (VAL_*)

| Code | Message | Cause | Action |
|------|---------|-------|--------|
| **VAL_001** | Missing required field | `query` not provided | Add `query` parameter |
| **VAL_002** | Invalid field type | Wrong parameter type | Ensure `limit` is integer, `filters` is object |
| **VAL_003** | Field value out of range | limit < 1 or > 100 | Use limit between 1 and 100 |
| **VAL_004** | Invalid field format | Date not YYYY-MM-DD format | Use ISO 8601 dates: "2025-11-24" |

### Embedding Errors (EMB_*)

| Code | Message | Cause | Action |
|------|---------|-------|--------|
| **EMB_001** | Ollama connection failed | Can't reach embedding service | Start Ollama: `ollama serve` |
| **EMB_002** | Embedding timeout | Embedding took too long | Retry query, check system load |
| **EMB_003** | Invalid embedding dimensions | Wrong vector size | Verify nomic-embed-text model |
| **EMB_004** | Model not found | Embedding model missing | `ollama pull nomic-embed-text` |

### Search Errors (SEARCH_*)

| Code | Message | Cause | Action |
|------|---------|-------|--------|
| **SEARCH_001** | Vector search failed | Query couldn't be executed | Check query format, retry |
| **SEARCH_002** | BM25 search failed | Keyword search error (Phase 2) | Retry with simpler query |
| **SEARCH_003** | Reranking failed | Result reranking failed (Phase 2) | Retry search |

### Database Errors (DB_*, CONN_*)

| Code | Message | Cause | Action |
|------|---------|-------|--------|
| **CONN_001** | FalkorDB connection refused | Database down | Start FalkorDB: `docker-compose up -d` |
| **CONN_002** | Connection timeout | Network issue | Check connectivity to database |

## Error Handling Examples

### Missing Query

```python
result = await search_memory.execute({
    "limit": 10
})
# Returns:
# Error: Missing required field 'query'
# Error Code: VAL_001
```

### Invalid Limit

```python
result = await search_memory.execute({
    "query": "Python",
    "limit": 200  # Max is 100
})
# Returns:
# Error: limit cannot exceed 100
# Error Code: VAL_003
```

### Invalid Date Format

```python
result = await search_memory.execute({
    "query": "Python",
    "filters": {
        "date_from": "2025/11/24"  # Wrong format
    }
})
# Returns:
# Error: Invalid field format (use YYYY-MM-DD)
# Error Code: VAL_004
```

### Ollama Service Down

```python
result = await search_memory.execute({
    "query": "machine learning"
})
# Returns:
# Error: Failed to process search query. Please try again.
# Error Code: EMB_001
# Action: Start Ollama with: ollama serve
```

## Usage Examples

### Python (Direct Library)

```python
from zapomni_mcp.tools import SearchMemoryTool
from zapomni_core.memory_processor import MemoryProcessor

processor = MemoryProcessor(...)
tool = SearchMemoryTool(processor)

# Simple search
result = await tool.execute({
    "query": "What is Python?"
})

# Search with limit
result = await tool.execute({
    "query": "decorators",
    "limit": 5
})

# Search with filters
result = await tool.execute({
    "query": "machine learning",
    "limit": 10,
    "filters": {
        "tags": ["ai", "ml"],
        "source": "tutorials",
        "date_from": "2025-01-01"
    }
})

# Check results
if not result["isError"]:
    print(result["content"][0]["text"])
```

### Python (MCP Client)

```python
from mcp import create_client

async with create_client("zapomni") as client:
    # Basic search
    result = await client.call("search_memory", {
        "query": "How do I use async/await in Python?"
    })

    # Advanced search
    result = await client.call("search_memory", {
        "query": "Docker containers",
        "limit": 20,
        "filters": {
            "tags": ["devops", "containerization"],
            "date_from": "2025-01-01"
        }
    })

    print(result.content[0].text)
```

### JavaScript/Node.js

```javascript
const { createClient } = require("@anthropic-ai/sdk/mcp");

const client = await createClient("zapomni");

// Basic search
const result = await client.call("search_memory", {
  query: "Tell me about REST APIs"
});

// Advanced search with filters
const result = await client.call("search_memory", {
  query: "authentication methods",
  limit: 15,
  filters: {
    tags: ["security", "auth"],
    source: "documentation",
    date_from: "2025-06-01"
  }
});

console.log(result.content[0].text);
```

### cURL (If HTTP transport available)

```bash
# Basic search
curl -X POST http://localhost:5000/mcp/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "search_memory",
    "arguments": {
      "query": "What is REST?"
    }
  }'

# Advanced search with filters
curl -X POST http://localhost:5000/mcp/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "search_memory",
    "arguments": {
      "query": "Python decorators",
      "limit": 10,
      "filters": {
        "tags": ["python"],
        "date_from": "2025-01-01"
      }
    }
  }'
```

## Query Examples and Results

### Example 1: Simple Fact Query

**Query**: `"Who created Python?"`

**Response**:
```
Found 2 results:

1. [Score: 0.94]
Python was created by Guido van Rossum in 1991

2. [Score: 0.87]
Guido van Rossum is the original author and "Benevolent Dictator For Life" (BDFL) of Python
```

### Example 2: Concept Query

**Query**: `"How do decorators work?"`

**Response**:
```
Found 3 results:

1. [Score: 0.93]
Decorators are functions that modify other functions without permanently changing the source code

2. [Score: 0.88]
@property is a built-in decorator that converts a method into a property

3. [Score: 0.82]
You can create your own decorators by defining a function that takes a function as an argument
```

### Example 3: Filtered Search

**Query**: `"testing"` with `filters: {"tags": ["python"]}`

**Response**:
```
Found 2 results:

1. [Score: 0.91] [Tags: python, testing]
unittest is Python's built-in testing framework

2. [Score: 0.86] [Tags: python, testing, pytest]
pytest provides a more modern and flexible testing framework for Python
```

### Example 4: Date Range Search

**Query**: `"async/await"` with `filters: {"date_from": "2025-06-01"}`

**Response**:
```
Found 1 result:

1. [Score: 0.95]
async and await keywords enable asynchronous programming in Python 3.5+
```

## Search Behavior and Edge Cases

### Empty Results

When no memories match the query:

```json
{
  "content": [{"type": "text", "text": "No results found matching your query."}],
  "isError": false
}
```

**Common causes**:
- Query too specific for stored memories
- No memories match filter criteria
- Memory hasn't been added yet

**Solution**: Try broader query or check with `get_stats`

### Whitespace-Only Query

```python
result = await search_memory.execute({
    "query": "   "  # Only spaces
})
# Error: query cannot be empty or contain only whitespace
```

### Very Broad Query

Query: `"the"`

**Result**: May match many chunks, top 10 most relevant returned

### Semantic Similarity

Query: `"large language models"`

**Matches**:
- "LLM" (semantic equivalence)
- "transformers and neural networks" (related concept)
- "AI assistants" (related domain)
- "natural language processing" (semantic similarity)

## Filter Behavior Details

### Tag Filtering (AND logic)

```python
filters = {"tags": ["python", "tutorial"]}
```

Returns memories that have **BOTH** tags:
- ✓ Memory with tags: `["python", "tutorial", "beginner"]`
- ✓ Memory with tags: `["python", "tutorial"]`
- ✗ Memory with tags: `["python"]` (missing "tutorial")
- ✗ Memory with tags: `["javascript", "tutorial"]` (missing "python")

### Source Filtering (exact match)

```python
filters = {"source": "github:username/repo"}
```

Returns memories from exact source:
- ✓ Source: `"github:username/repo"`
- ✗ Source: `"github:other/repo"`
- ✗ Source: `"github:username/repo/file"` (if stored differently)

### Date Range Filtering (inclusive)

```python
filters = {
    "date_from": "2025-01-01",
    "date_to": "2025-12-31"
}
```

Returns memories created between these dates (inclusive):
- ✓ Created: `2025-01-01T00:00:00Z`
- ✓ Created: `2025-06-15T12:30:00Z`
- ✓ Created: `2025-12-31T23:59:59Z`
- ✗ Created: `2024-12-31T23:59:59Z` (before range)
- ✗ Created: `2026-01-01T00:00:00Z` (after range)

## Performance Characteristics

| Operation | Time (P95) | Notes |
|-----------|-----------|-------|
| Query embedding | 50-200ms | Depends on query length |
| Vector search | 10-100ms | HNSW index lookup |
| Filtering | < 10ms | Metadata filtering |
| Total latency | **< 500ms** | For typical query |

## Best Practices

### 1. Query Formulation

```python
# Good: Natural language, specific
await search_memory.execute({"query": "How do I handle errors in Python?"})

# Good: Concept-based
await search_memory.execute({"query": "dependency injection patterns"})

# Bad: Too vague
await search_memory.execute({"query": "stuff"})

# Bad: Too long/complex
await search_memory.execute({
    "query": "Tell me everything you know about every aspect of programming including but not limited to..."
})
```

### 2. Limit Usage

```python
# Good: Appropriate limit for use case
await search_memory.execute({
    "query": "Python",
    "limit": 5  # For quick answer
})

await search_memory.execute({
    "query": "Python",
    "limit": 50  # For comprehensive research
})

# Bad: Always using max limit (slower)
await search_memory.execute({
    "query": "anything",
    "limit": 100  # Every query
})
```

### 3. Smart Filtering

```python
# Good: Narrow scope with filters
await search_memory.execute({
    "query": "authentication",
    "filters": {
        "tags": ["security"],
        "source": "documentation"
    }
})

# Bad: Overly restrictive filters
await search_memory.execute({
    "query": "authentication",
    "filters": {
        "tags": ["security", "auth", "login", "password"],  # Too many tags
        "source": "exact/source/path/that/may/not/exist",
        "date_from": "2025-11-24",
        "date_to": "2025-11-24"  # Single day - too narrow
    }
})
```

### 4. Error Handling

```python
# Good: Handle different error cases
try:
    result = await search_memory.execute({
        "query": query,
        "limit": min(limit, 100),  # Validate limit
        "filters": filters
    })

    if result["isError"]:
        error_msg = result["content"][0]["text"]
        if "Ollama" in error_msg:
            raise ServiceError("Embedding service unavailable")
        elif "Database" in error_msg:
            raise ServiceError("Database unavailable")
        else:
            raise ValueError(error_msg)

    return result["content"][0]["text"]

except ValueError as e:
    print(f"Invalid input: {e}")
except ServiceError as e:
    print(f"Service error: {e}")
```

## Related Tools

- **[add_memory](./add_memory.md)** - Store memories to search
- **[get_stats](./get_stats.md)** - Check total memories available

## See Also

- **[Request/Response Schemas](../schemas.md)** - JSON schema definitions
- **[Error Reference](../errors.md)** - All error codes and meanings
- **[Ranking Explanation](../README.md#ranking-and-scoring)** - How results are ranked

## Troubleshooting

### "No results found matching your query"

**Cause**: Query doesn't match any stored memories

**Solutions**:
1. Try simpler, more general query
2. Check with `get_stats` that memories exist
3. Use broader tag filters
4. Remove date restrictions

### "Query cannot be empty"

**Cause**: Empty or whitespace-only query string

**Solution**: Provide non-empty query

### "limit cannot exceed 100"

**Cause**: Requested limit > 100

**Solution**: Use limit between 1 and 100

### "Embedding timeout"

**Cause**: Query embedding took too long

**Solution**:
1. Retry query (might be transient)
2. Check system load (CPU/memory)
3. Restart Ollama if problem persists

---

**Tool Version**: 1.0
**Phase**: MVP (Current)
**Status**: Stable
**Last Updated**: 2025-11-24
