# add_memory Tool

Store text content with automatic semantic chunking, embedding generation, and graph storage.

## Overview

The `add_memory` tool is the primary way to add information to Zapomni's memory system. It accepts text in any format (natural language, code, documentation) and processes it through a complete pipeline:

1. **Validation** - Ensures input meets requirements
2. **Chunking** - Splits text into semantic chunks (typically 256-512 tokens)
3. **Embedding** - Generates vector embeddings using Ollama
4. **Extraction** - Extracts entities and relationships (Phase 2)
5. **Storage** - Stores memory in FalkorDB with full-text and vector indexing

## Tool Identifier

```
name: "add_memory"
```

## Description

"Add a memory (text or code) to the knowledge graph. The memory will be processed, chunked, embedded, and stored for later retrieval."

## Parameters

### text (Required)

**Type**: `string`
**Min Length**: 1 character
**Max Length**: 10,000,000 characters (10 MB)
**Validation**: Non-empty after whitespace stripping

The content to remember. Can be:

- Natural language text (articles, notes, conversations)
- Source code (Python, JavaScript, SQL, etc.)
- Documentation (markdown, html, plaintext)
- Structured data (JSON, YAML, CSV)
- Mixed content

**Examples**:
- `"Python was created by Guido van Rossum in 1991"`
- Complete source code files or snippets
- Entire Wikipedia articles
- Meeting notes or transcripts

**Validation Rules**:

```
✓ Non-empty string
✓ Valid UTF-8 encoding
✓ Between 1 and 10,000,000 characters
✗ Empty after stripping whitespace
✗ Binary data without encoding
✗ Exceeds 10MB size limit
```

### metadata (Optional)

**Type**: `object` (dictionary/map)
**Default**: `{}`
**Additional Properties**: Allowed (`true`)

Metadata to attach to the memory for filtering, organization, and context. All fields are optional.

#### Supported Metadata Fields

**source** (string)
- Identifier for where memory came from
- Examples: `"user"`, `"api"`, `"file:///path/to/file"`, `"github:username/repo"`
- Used for filtering and tracking provenance

**tags** (array of strings)
- List of tags for categorization
- Examples: `["python", "programming", "tutorial"]`
- Useful for organizing related memories
- Retrievable in search results

**timestamp** (string, ISO 8601 format)
- When the memory was created/acquired
- Format: `"2025-11-24T10:30:00Z"` or `"2025-11-24T10:30:00+00:00"`
- If not provided, current time is used
- Supports date range filtering in search

**language** (string)
- Programming language if text is code
- Examples: `"python"`, `"javascript"`, `"sql"`, `"go"`, `"rust"`
- Used to select appropriate syntax-aware processing
- Helps with code chunking and analysis

**custom fields** (any type)
- You can add any custom fields to metadata
- Will be stored and returned in search results
- Examples: `"author"`, `"category"`, `"priority"`, `"version"`, etc.

**Example Metadata**:

```json
{
  "source": "github:torvalds/linux",
  "tags": ["kernel", "c", "linux", "systems"],
  "timestamp": "2025-11-24T10:30:00Z",
  "language": "c",
  "author": "Linus Torvalds",
  "custom_field": "any value"
}
```

## Request Example

### Minimal Request

```json
{
  "text": "Claude is an AI assistant created by Anthropic"
}
```

### Complete Request

```json
{
  "text": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
  "metadata": {
    "source": "code-example",
    "tags": ["python", "algorithms", "fibonacci"],
    "language": "python",
    "timestamp": "2025-11-24T10:30:00Z",
    "difficulty": "intermediate"
  }
}
```

## Response Format

### Success Response

Status: `isError: false`

```json
{
  "content": [
    {
      "type": "text",
      "text": "Memory stored successfully.\nID: 550e8400-e29b-41d4-a716-446655440000\nPreview: Python was created by Guido van Rossum in 1991..."
    }
  ],
  "isError": false
}
```

**Response Fields**:

- **ID**: UUID of the stored memory (v4 format)
- **Preview**: First 100 characters of the text
- **Status**: Always "Memory stored successfully" on success

### Error Response

Status: `isError: true`

```json
{
  "content": [
    {
      "type": "text",
      "text": "Error: text cannot be empty or contain only whitespace"
    }
  ],
  "isError": true
}
```

## Response Data Structure

The tool returns a formatted text response. Internally, the system processes:

```
Input Text
  ↓
[Validation]
  ↓
[Semantic Chunking] → chunks: [Chunk1, Chunk2, ..., ChunkN]
  ↓
[Embedding Generation] → embeddings: [Vector1, Vector2, ..., VectorN]
  ↓
[Entity Extraction] → entities: [Entity1, Entity2, ...] (Phase 2)
  ↓
[Database Storage] → Memory ID: UUID
```

## Error Codes

### Validation Errors (VAL_*)

| Code | Message | Cause | Action |
|------|---------|-------|--------|
| **VAL_001** | Missing required field | `text` parameter not provided | Add `text` parameter |
| **VAL_002** | Invalid field type | Parameter has wrong type | Ensure `text` is string, `metadata` is object |
| **VAL_003** | Field value out of range | Text too long (>10MB) | Reduce text size or split into multiple memories |
| **VAL_004** | Invalid field format | Non-UTF8 bytes, invalid format | Ensure valid UTF-8 encoding |

### Processing Errors (PROC_*)

| Code | Message | Cause | Action |
|------|---------|-------|--------|
| **PROC_001** | Chunking failed | Text couldn't be split semantically | Check text format, try simpler text |
| **PROC_002** | Text extraction failed | Problem processing text content | Verify text is valid UTF-8 |
| **PROC_003** | Invalid document format | Unsupported content type | Use plaintext, code, or markdown |

### Embedding Errors (EMB_*)

| Code | Message | Cause | Action |
|------|---------|-------|--------|
| **EMB_001** | Ollama connection failed | Can't reach embedding service | Start Ollama: `ollama serve` |
| **EMB_002** | Embedding timeout | Embedding generation took too long | Retry, or reduce chunk size |
| **EMB_003** | Invalid embedding dimensions | Wrong vector size | Verify `nomic-embed-text` model loaded |
| **EMB_004** | Model not found | Required embedding model missing | `ollama pull nomic-embed-text` |

### Database Errors (DB_*, CONN_*, QUERY_*)

| Code | Message | Cause | Action |
|------|---------|-------|--------|
| **CONN_001** | FalkorDB connection refused | Database service down | Start FalkorDB: `docker-compose up -d` |
| **CONN_002** | Connection timeout | Network issue with database | Check network connectivity |
| **QUERY_001** | Syntax error in Cypher query | Database query malformed | Rare - file a bug report |
| **QUERY_004** | Constraint violation | Duplicate memory ID (extremely rare) | Retry operation |

## Error Handling Examples

### Invalid Input

```python
# Missing text parameter
response = await add_memory.execute({
    "metadata": {"source": "test"}
})
# Returns:
# Error: Missing required field 'text'
# Error Code: VAL_001
```

### Text Too Large

```python
# Text exceeds 10MB limit
huge_text = "x" * 10_000_001
response = await add_memory.execute({
    "text": huge_text
})
# Returns:
# Error: Field value out of range (max 10,000,000 chars)
# Error Code: VAL_003
```

### Ollama Service Down

```python
# Embedding service not running
response = await add_memory.execute({
    "text": "Some important memory"
})
# Returns:
# Error: Failed to process text for embedding. Please try again.
# Error Code: EMB_001
# Action: Start Ollama with: ollama serve
```

### Database Connection Lost

```python
# FalkorDB service unavailable
response = await add_memory.execute({
    "text": "Some important memory"
})
# Returns:
# Error: Database temporarily unavailable. Please retry in a few seconds.
# Error Code: CONN_001
# Action: Restart FalkorDB with: docker-compose up -d
```

## Usage Examples

### Python (Direct Library)

```python
from zapomni_mcp.tools import AddMemoryTool
from zapomni_core.memory_processor import MemoryProcessor

# Initialize (assumes services running)
processor = MemoryProcessor(...)
tool = AddMemoryTool(processor)

# Add simple memory
result = await tool.execute({
    "text": "The Earth orbits the Sun"
})

# With metadata
result = await tool.execute({
    "text": "Python 3.10 introduced structural pattern matching",
    "metadata": {
        "source": "documentation",
        "tags": ["python", "3.10", "features"],
        "language": "en"
    }
})

# Check result
if result["isError"]:
    print(f"Error: {result['content'][0]['text']}")
else:
    print(f"Stored: {result['content'][0]['text']}")
```

### Python (MCP Client)

```python
from mcp import create_client

async with create_client("zapomni") as client:
    result = await client.call("add_memory", {
        "text": "Machine learning models learn patterns from data",
        "metadata": {
            "source": "learning",
            "tags": ["ml", "ai", "fundamentals"]
        }
    })
    print(result.content[0].text)
```

### JavaScript/Node.js (MCP Client)

```javascript
const { createClient } = require("@anthropic-ai/sdk/mcp");

const client = await createClient("zapomni");

const result = await client.call("add_memory", {
  text: "Decorators in Python provide a clean way to modify functions",
  metadata: {
    source: "tutorial",
    tags: ["python", "decorators", "design-patterns"],
    language: "python"
  }
});

console.log(result.content[0].text);
```

### cURL (If HTTP transport available)

```bash
curl -X POST http://localhost:5000/mcp/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "add_memory",
    "arguments": {
      "text": "REST stands for Representational State Transfer",
      "metadata": {
        "source": "api",
        "tags": ["rest", "api", "web"]
      }
    }
  }'
```

## Processing Details

### Text Chunking

The text is split into semantic chunks (typically 256-512 tokens, ~1-2KB):

- Chunks overlap by ~50 tokens for context continuity
- Chunking is semantically aware (breaks on sentence boundaries when possible)
- Each chunk is independently embedded
- Chunk boundaries preserve code structure

**Example Chunking**:

```
Input: "def hello():\n    print('world')\n\ndef goodbye():\n    print('farewell')"

Chunks:
1. "def hello():\n    print('world')"
2. "def goodbye():\n    print('farewell')"
```

### Embedding Generation

Each chunk is embedded using the `nomic-embed-text` model:

- **Dimensions**: 768 dimensions
- **Model**: `nomic-embed-text` (1.4B parameters, optimized for retrieval)
- **Speed**: ~100-500ms per 1000 tokens
- **Quality**: State-of-the-art for semantic similarity

### Storage

Stored in FalkorDB with multiple indices:

- **Vector Index**: HNSW index for fast similarity search (O(log n) average case)
- **Full-Text Index**: BM25 index for keyword search (Phase 2)
- **Graph Index**: Cypher queries on knowledge graph (Phase 2)

## Performance Characteristics

| Operation | Latency (P95) | Notes |
|-----------|---------------|-------|
| Chunking | < 50ms | Fast, local operation |
| Embedding | 100-500ms | Depends on chunk count |
| Storage | < 100ms | Database write |
| **Total** | **< 2 seconds** | For typical 1-5KB text |

## Limits and Constraints

| Limit | Value | Notes |
|-------|-------|-------|
| Max text size | 10 MB | Enforced at schema |
| Max chunks per memory | 100 | Soft limit, configurable |
| Max metadata size | Unlimited | Serialized with memory |
| Max metadata fields | Unlimited | No schema restriction |
| Vector dimensions | 768 | Fixed by nomic-embed-text |

## Best Practices

### 1. Metadata Organization

```python
# Good: Informative metadata
{
    "text": "...",
    "metadata": {
        "source": "github:username/repo",
        "tags": ["category1", "category2"],
        "timestamp": "2025-11-24T10:30:00Z"
    }
}

# Bad: Vague metadata
{
    "text": "...",
    "metadata": {
        "info": "stuff"
    }
}
```

### 2. Text Organization

```python
# Good: Coherent, focused content
{
    "text": "Python decorators are functions that modify other functions..."
}

# Bad: Too diverse
{
    "text": "Python decorators... Also, the weather is nice... And SQL queries..."
}
```

### 3. Batch Operations

```python
# Good: Process one document at a time for better chunking
for document in documents:
    await add_memory.execute({
        "text": document,
        "metadata": {"source": document.filename}
    })

# Bad: Concatenate many documents
all_text = "\n".join(documents)
await add_memory.execute({"text": all_text})
```

### 4. Error Handling

```python
# Good: Catch and handle errors
try:
    result = await add_memory.execute({"text": text})
    if result["isError"]:
        error_msg = result["content"][0]["text"]
        if "Ollama" in error_msg:
            print("Start Ollama service")
        elif "Database" in error_msg:
            print("Start FalkorDB")
except Exception as e:
    print(f"Unexpected error: {e}")

# Bad: Assume success
result = await add_memory.execute({"text": text})
memory_id = extract_id_from_result(result)  # Will fail if error
```

## Troubleshooting

### "Ollama connection failed"

**Symptom**: Memory can't be stored, embedding service error

**Solution**:
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running, start it
ollama serve

# If model not loaded
ollama pull nomic-embed-text
```

### "Database temporarily unavailable"

**Symptom**: Can't write to database

**Solution**:
```bash
# Check FalkorDB status
docker-compose ps

# Restart if needed
docker-compose down
docker-compose up -d

# Verify connectivity
redis-cli ping  # Should return PONG
```

### "Text too large"

**Symptom**: VAL_003 error on large files

**Solution**: Split into multiple smaller memories:
```python
# Instead of:
await add_memory.execute({"text": huge_text})

# Do this:
chunk_size = 5_000_000  # 5MB per memory
for i in range(0, len(huge_text), chunk_size):
    chunk = huge_text[i:i+chunk_size]
    await add_memory.execute({
        "text": chunk,
        "metadata": {"part": i // chunk_size + 1}
    })
```

### "Invalid UTF-8 encoding"

**Symptom**: VAL_004 error on text input

**Solution**: Ensure proper encoding:
```python
# Convert bytes to string if needed
if isinstance(text, bytes):
    text = text.decode('utf-8', errors='replace')

# Or
text = text.encode('utf-8', errors='ignore').decode('utf-8')
```

## Related Tools

- **[search_memory](./search_memory.md)** - Retrieve stored memories
- **[get_stats](./get_stats.md)** - View memory statistics

## See Also

- **[Request/Response Schemas](../schemas.md)** - JSON schema definitions
- **[Error Reference](../errors.md)** - All error codes and meanings
- **[Data Models](../schemas.md#data-models)** - Memory, Chunk, SearchResult definitions

---

**Tool Version**: 1.0
**Phase**: MVP (Current)
**Status**: Stable
**Last Updated**: 2025-11-24
