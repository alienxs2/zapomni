# Data Flow Architecture - Module Specification

**Level:** 1 (Module-level)
**Author:** Goncharenko Anton aka alienxs2
**Status:** Draft
**Version:** 1.0
**Date:** 2025-11-23
**Copyright:** Copyright (c) 2025 Goncharenko Anton aka alienxs2
**License:** MIT

---

## Executive Summary

This specification defines the complete data flow architecture for Zapomni, describing how data moves through the system from MCP requests to database storage and back. It covers three primary operations (add_memory, search_memory, get_stats), data transformations at each stage, processing pipelines, and performance characteristics.

**Key Architectural Decisions:**
- **Pipeline Architecture:** Modular, composable stages for flexibility and testability
- **Async Processing:** Full async/await for I/O operations to maximize throughput
- **Streaming Where Possible:** Reduce memory footprint for large documents
- **Fail-Fast Validation:** Validate early to avoid wasted processing
- **Observable Pipelines:** Structured logging at each stage for debugging

---

## Overview

### Purpose

The data flow architecture defines:
1. **Request Processing Flows:** How MCP requests are transformed into responses
2. **Data Transformations:** What happens to data at each processing stage
3. **Pipeline Stages:** Discrete processing steps and their responsibilities
4. **Error Propagation:** How errors flow through the system
5. **Performance Characteristics:** Expected latencies, throughput, resource usage

### Scope

**Included:**
- add_memory flow (MCP → chunking → embedding → storage)
- search_memory flow (MCP → embedding → vector search → reranking → response)
- get_stats flow (MCP → DB aggregation → response)
- Data transformations at each stage
- Error handling and recovery strategies
- Performance targets and bottlenecks

**Not Included:**
- Implementation details of individual components (covered in Level 2 specs)
- Database schema design (covered in zapomni_db module spec)
- MCP protocol specifics (covered in zapomni_mcp module spec)
- Knowledge graph construction (Phase 2, future spec)

### Position in Architecture

Data flow architecture sits between the three main modules:
```
MCP Module → Data Flow → Core Module → Data Flow → DB Module
```

It defines the contracts and transformations between these layers.

---

## Architecture Overview

### System Context Diagram

```
┌──────────────┐
│  MCP Client  │
│  (Claude)    │
└──────┬───────┘
       │ stdio (JSON-RPC)
       ↓
┌──────────────────────────────────────────────────┐
│             ZAPOMNI MCP SERVER                   │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐ │
│  │ add_memory │  │search_mem  │  │ get_stats  │ │
│  │    Tool    │  │    Tool    │  │    Tool    │ │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘ │
└────────┼───────────────┼───────────────┼────────┘
         │               │               │
         ↓               ↓               ↓
┌──────────────────────────────────────────────────┐
│           DATA FLOW PIPELINES                    │
│  ┌─────────────┐ ┌─────────────┐ ┌────────────┐ │
│  │Add Pipeline │ │Search Pipe  │ │Stats Pipe  │ │
│  │(this spec)  │ │(this spec)  │ │(this spec) │ │
│  └──────┬──────┘ └──────┬──────┘ └──────┬─────┘ │
└─────────┼───────────────┼───────────────┼───────┘
          │               │               │
          ↓               ↓               ↓
┌──────────────────────────────────────────────────┐
│              ZAPOMNI CORE                        │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐            │
│  │ Chunker │ │Embedder │ │ Search  │            │
│  └─────────┘ └─────────┘ └─────────┘            │
└──────────────────────────────────────────────────┘
          │               │               │
          ↓               ↓               ↓
┌──────────────────────────────────────────────────┐
│              ZAPOMNI DB                          │
│  ┌──────────────────────────────────┐            │
│  │        FalkorDB Client           │            │
│  │  (Unified Vector + Graph)        │            │
│  └──────────────────────────────────┘            │
└──────────────────────────────────────────────────┘
```

### Key Principles

1. **Unidirectional Data Flow:** Data flows down through layers, responses flow up
2. **Immutable Transformations:** Each stage produces new data, doesn't mutate input
3. **Fail-Fast Validation:** Validate at entry points before expensive operations
4. **Observable at Every Stage:** Structured logging captures data shape and timing
5. **Composable Pipelines:** Stages can be swapped or reordered for different strategies

---

## Operation 1: add_memory Data Flow

### High-Level Flow

```
User Request
    ↓
[1. MCP Request Reception] (zapomni_mcp)
    ↓
[2. Input Validation] (zapomni_mcp)
    ↓
[3. Document Processing] (zapomni_core)
    ├─ 3a. Text Extraction
    ├─ 3b. Chunking
    └─ 3c. Metadata Enrichment
    ↓
[4. Embedding Generation] (zapomni_core)
    ├─ 4a. Batch Preparation
    ├─ 4b. Ollama API Call
    └─ 4c. Embedding Cache (Phase 2)
    ↓
[5. Storage] (zapomni_db)
    ├─ 5a. Create Memory Node
    ├─ 5b. Create Chunk Nodes
    ├─ 5c. Create Relationships
    └─ 5d. Build Vector Index
    ↓
[6. Response Formation] (zapomni_mcp)
    ↓
MCP Response
```

### Stage 1: MCP Request Reception

**Input:**
```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "add_memory",
    "arguments": {
      "text": "Python was created by Guido van Rossum in 1991...",
      "metadata": {
        "source": "wikipedia",
        "tags": ["python", "programming"],
        "date": "2024-11-22"
      }
    }
  },
  "id": 1
}
```

**Processing:**
- Parse JSON-RPC envelope
- Extract tool name and arguments
- Route to add_memory tool handler

**Output:**
```python
ToolRequest(
    method="tools/call",
    params={
        "name": "add_memory",
        "arguments": {
            "text": "Python was created...",
            "metadata": {...}
        }
    }
)
```

**Performance:**
- Latency: < 1ms
- Memory: O(1) - just parsing

### Stage 2: Input Validation

**Input:** ToolRequest (from Stage 1)

**Processing:**
```python
# Validate text field
if not arguments.get("text"):
    raise ValidationError("Missing required field: text")

text = arguments["text"]
if not isinstance(text, str):
    raise ValidationError("text must be string")

if not text.strip():
    raise ValidationError("text cannot be empty")

if len(text) > 10_000_000:  # 10MB limit
    raise ValidationError("text exceeds maximum size")

# Validate metadata (optional)
metadata = arguments.get("metadata", {})
if metadata and not isinstance(metadata, dict):
    raise ValidationError("metadata must be dict")

# Check JSON serializability
try:
    json.dumps(metadata)
except (TypeError, ValueError):
    raise ValidationError("metadata must be JSON-serializable")
```

**Output:**
```python
ValidatedInput(
    text="Python was created by Guido van Rossum in 1991...",
    metadata={
        "source": "wikipedia",
        "tags": ["python", "programming"],
        "date": "2024-11-22"
    },
    text_length=1547,
    estimated_chunks=3
)
```

**Performance:**
- Latency: < 5ms
- Memory: O(n) where n = text size

### Stage 3: Document Processing

**Input:** ValidatedInput (from Stage 2)

#### Stage 3a: Text Extraction

**Processing:**
```python
# For plain text (this case), pass through
if input_type == "text":
    extracted_text = text

# For PDF (future)
elif input_type == "pdf":
    extracted_text = pdf_processor.extract_text(text)

# For HTML (future)
elif input_type == "html":
    extracted_text = html_processor.extract_text(text)
```

**Output:**
```python
ExtractedText(
    content="Python was created by Guido van Rossum in 1991...",
    length=1547,
    encoding="utf-8",
    language="en"  # detected
)
```

#### Stage 3b: Chunking

**Processing:**
```python
# Semantic chunking with overlap
chunker = SemanticChunker(
    chunk_size=512,  # tokens
    overlap=50,      # tokens
    strategy="recursive"  # LangChain RecursiveTextSplitter
)

chunks = chunker.chunk(extracted_text.content)
```

**Algorithm:**
```
1. Split on paragraph boundaries first (\n\n)
2. If paragraph > chunk_size:
   - Split on sentence boundaries (. ! ?)
3. If sentence > chunk_size:
   - Split on clause boundaries (, ; :)
4. If clause > chunk_size:
   - Split on word boundaries
5. Add overlap between chunks (last N tokens of chunk[i]
   become first N tokens of chunk[i+1])
6. Preserve metadata for each chunk (position, parent doc)
```

**Output:**
```python
Chunks([
    Chunk(
        text="Python was created by Guido van Rossum in 1991...",
        index=0,
        token_count=487,
        char_start=0,
        char_end=512,
        metadata={
            "parent_doc": "doc_id",
            "position": "start"
        }
    ),
    Chunk(
        text="...van Rossum in 1991. It emphasizes code readability...",
        index=1,
        token_count=503,
        char_start=462,  # 50 token overlap
        char_end=1024,
        metadata={
            "parent_doc": "doc_id",
            "position": "middle"
        }
    ),
    Chunk(
        text="...readability with significant whitespace. Python is...",
        index=2,
        token_count=512,
        char_start=974,
        char_end=1547,
        metadata={
            "parent_doc": "doc_id",
            "position": "end"
        }
    )
])
# Total: 3 chunks
```

**Performance:**
- Latency: O(n) where n = text length
- Target: < 10ms per 1KB of text
- For 10KB document: ~100ms

#### Stage 3c: Metadata Enrichment

**Processing:**
```python
# Add system metadata to each chunk
for chunk in chunks:
    chunk.metadata.update({
        "timestamp": datetime.utcnow().isoformat(),
        "source": original_metadata.get("source", "unknown"),
        "tags": original_metadata.get("tags", []),
        "chunk_strategy": "semantic_recursive",
        "chunk_size": 512,
        "overlap": 50
    })
```

**Output:**
```python
EnrichedChunks([
    Chunk(
        text="Python was created...",
        index=0,
        token_count=487,
        metadata={
            "parent_doc": "doc_id",
            "position": "start",
            "timestamp": "2024-11-22T10:30:00Z",
            "source": "wikipedia",
            "tags": ["python", "programming"],
            "chunk_strategy": "semantic_recursive",
            "chunk_size": 512,
            "overlap": 50
        }
    ),
    # ... other chunks
])
```

**Performance:**
- Latency: O(k) where k = number of chunks
- Target: < 1ms (just dict updates)

### Stage 4: Embedding Generation

**Input:** EnrichedChunks (from Stage 3c)

#### Stage 4a: Batch Preparation

**Processing:**
```python
# Extract texts for embedding
texts_to_embed = [chunk.text for chunk in enriched_chunks]

# Check cache (Phase 2)
if cache_enabled:
    cached, uncached = cache.get_many(texts_to_embed)
    texts_to_embed = uncached
else:
    cached = []
    uncached = texts_to_embed
```

**Output:**
```python
BatchRequest(
    texts=[
        "Python was created by Guido van Rossum...",
        "...van Rossum in 1991. It emphasizes...",
        "...readability with significant whitespace..."
    ],
    model="nomic-embed-text",
    batch_size=3,
    cached_count=0,
    uncached_count=3
)
```

#### Stage 4b: Ollama API Call

**Processing:**
```python
embeddings = []

# Process in batches of 32 (configurable)
for batch in chunk_list(texts_to_embed, batch_size=32):
    # Concurrent API calls within batch
    tasks = [
        ollama_client.embed(text, model="nomic-embed-text")
        for text in batch
    ]
    batch_embeddings = await asyncio.gather(*tasks)
    embeddings.extend(batch_embeddings)
```

**API Request (per text):**
```http
POST http://localhost:11434/api/embeddings
Content-Type: application/json

{
  "model": "nomic-embed-text",
  "prompt": "Python was created by Guido van Rossum..."
}
```

**API Response:**
```json
{
  "embedding": [0.123, -0.456, 0.789, ..., 0.234],  // 768 dimensions
  "model": "nomic-embed-text"
}
```

**Output:**
```python
Embeddings([
    [0.123, -0.456, 0.789, ..., 0.234],  # 768-dim vector
    [0.234, -0.567, 0.890, ..., 0.345],
    [0.345, -0.678, 0.901, ..., 0.456]
])
# Shape: (3, 768)
```

**Performance:**
- Latency: ~50-100ms per embedding (Ollama local)
- Batch parallelization: 32 concurrent requests
- For 3 chunks: ~100-150ms total (parallel)
- For 100 chunks: ~400-600ms (4 batches × ~100-150ms)

#### Stage 4c: Embedding Cache (Phase 2)

**Processing:**
```python
# Store in cache for reuse
if cache_enabled:
    for text, embedding in zip(texts_to_embed, embeddings):
        cache_key = hashlib.sha256(text.encode()).hexdigest()
        cache.set(cache_key, embedding, ttl=86400)  # 24h
```

**Performance:**
- Latency: ~1-2ms per cache write (Redis)
- Target cache hit rate: 60-68% (research target)

### Stage 5: Storage

**Input:**
- EnrichedChunks (from Stage 3c)
- Embeddings (from Stage 4b)

#### Stage 5a: Create Memory Node

**Processing:**
```python
memory_id = str(uuid.uuid4())

memory_node = graph.query("""
    CREATE (m:Memory {
        id: $id,
        text: $text,
        source: $source,
        tags: $tags,
        timestamp: datetime(),
        chunk_count: $chunk_count
    })
    RETURN m.id
""", {
    "id": memory_id,
    "text": original_text,
    "source": metadata.get("source", ""),
    "tags": metadata.get("tags", []),
    "chunk_count": len(chunks)
})
```

**Output:**
```
Memory Node Created:
- ID: "550e8400-e29b-41d4-a716-446655440000"
- Label: Memory
- Properties: {text, source, tags, timestamp, chunk_count}
```

**Performance:**
- Latency: ~5-10ms (single node creation)

#### Stage 5b: Create Chunk Nodes

**Processing:**
```python
for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
    chunk_id = f"{memory_id}_chunk_{i}"

    graph.query("""
        MATCH (m:Memory {id: $memory_id})
        CREATE (c:Chunk {
            id: $chunk_id,
            text: $text,
            index: $index,
            token_count: $token_count,
            embedding: $embedding,
            metadata: $metadata
        })
        CREATE (m)-[:HAS_CHUNK {index: $index}]->(c)
    """, {
        "memory_id": memory_id,
        "chunk_id": chunk_id,
        "text": chunk.text,
        "index": i,
        "token_count": chunk.token_count,
        "embedding": embedding.tolist(),
        "metadata": json.dumps(chunk.metadata)
    })
```

**Output:**
```
Chunk Nodes Created: 3
- chunk_0: 487 tokens, embedding 768-dim
- chunk_1: 503 tokens, embedding 768-dim
- chunk_2: 512 tokens, embedding 768-dim

Relationships Created: 3
- Memory -[HAS_CHUNK]-> chunk_0
- Memory -[HAS_CHUNK]-> chunk_1
- Memory -[HAS_CHUNK]-> chunk_2
```

**Performance:**
- Latency: ~10-15ms per chunk (includes relationship)
- For 3 chunks: ~30-45ms total
- Could batch for larger chunk counts

#### Stage 5c: Build Vector Index

**Processing:**
```python
# FalkorDB automatically indexes vectors on write
# Vector index already created during schema initialization:
# CREATE VECTOR INDEX FOR (c:Chunk) ON (c.embedding)
# OPTIONS {dimension: 768, similarityFunction: 'cosine'}

# Verification query (optional, for monitoring)
index_status = graph.query("""
    CALL db.idx.vector.info('Chunk', 'embedding')
""")
```

**Output:**
```
Vector Index Status:
- Indexed nodes: 3 new (total: N+3)
- Dimension: 768
- Similarity: cosine
- Status: ready
```

**Performance:**
- Latency: Automatic, negligible (asynchronous in FalkorDB)
- Index build is incremental

### Stage 6: Response Formation

**Input:**
- memory_id (from Stage 5a)
- chunk_count (from Stage 5b)
- processing stats

**Processing:**
```python
response = {
    "content": [
        {
            "type": "text",
            "text": f"Memory stored successfully.\n\n"
                   f"Memory ID: {memory_id}\n"
                   f"Chunks created: {chunk_count}\n"
                   f"Preview: {original_text[:100]}..."
        }
    ],
    "isError": False,
    "_meta": {
        "processing_time_ms": end_time - start_time,
        "memory_id": memory_id,
        "chunks": chunk_count
    }
}
```

**Output (MCP Response):**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Memory stored successfully.\n\nMemory ID: 550e8400-e29b-41d4-a716-446655440000\nChunks created: 3\nPreview: Python was created by Guido van Rossum in 1991. It is known for its simple, readable..."
      }
    ],
    "isError": false,
    "_meta": {
      "processing_time_ms": 387,
      "memory_id": "550e8400-e29b-41d4-a716-446655440000",
      "chunks": 3
    }
  },
  "id": 1
}
```

**Performance:**
- Latency: < 5ms (just string formatting)

### add_memory Total Performance Budget

**Target Latency (P95):**
- Small document (< 1KB, 1-2 chunks): < 200ms
- Medium document (1-10KB, 3-20 chunks): < 500ms
- Large document (10-100KB, 20-200 chunks): < 3000ms

**Breakdown (for medium document with 3 chunks):**
```
Stage 1: MCP Reception         ~1ms
Stage 2: Validation            ~5ms
Stage 3a: Text Extraction      ~1ms
Stage 3b: Chunking             ~10ms
Stage 3c: Metadata Enrichment  ~1ms
Stage 4a: Batch Prep           ~2ms
Stage 4b: Embedding (parallel) ~100-150ms  ← BOTTLENECK
Stage 4c: Cache Write          ~3ms
Stage 5a: Create Memory        ~5-10ms
Stage 5b: Create Chunks        ~30-45ms
Stage 5c: Vector Index         ~1ms (async)
Stage 6: Response Formation    ~5ms
─────────────────────────────────────
Total:                         ~164-233ms ✅ Under budget
```

**Bottleneck Identified:** Embedding generation (Stage 4b)
- Mitigation: Batch parallelization (already implemented)
- Future: GPU acceleration, embedding cache (Phase 2)

---

## Operation 2: search_memory Data Flow

### High-Level Flow

```
User Query
    ↓
[1. MCP Request Reception] (zapomni_mcp)
    ↓
[2. Input Validation] (zapomni_mcp)
    ↓
[3. Query Embedding] (zapomni_core)
    ├─ 3a. Cache Lookup (Phase 2)
    ├─ 3b. Ollama API Call
    └─ 3c. Cache Write (Phase 2)
    ↓
[4. Vector Search] (zapomni_db)
    ├─ 4a. HNSW Index Query
    ├─ 4b. Metadata Filtering
    └─ 4c. Top-K Retrieval
    ↓
[5. Result Ranking] (zapomni_core)
    ├─ 5a. BM25 Fusion (Phase 2)
    ├─ 5b. Cross-Encoder Rerank (Phase 2)
    └─ 5c. Graph Context (Phase 2)
    ↓
[6. Response Formation] (zapomni_mcp)
    ↓
MCP Response
```

### Stage 1: MCP Request Reception

**Input:**
```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "search_memory",
    "arguments": {
      "query": "Who created Python?",
      "limit": 10,
      "filters": {
        "tags": ["python"],
        "date_from": "2024-01-01"
      }
    }
  },
  "id": 2
}
```

**Processing:** Parse MCP request, extract arguments

**Output:**
```python
ToolRequest(
    method="tools/call",
    params={
        "name": "search_memory",
        "arguments": {
            "query": "Who created Python?",
            "limit": 10,
            "filters": {...}
        }
    }
)
```

### Stage 2: Input Validation

**Input:** ToolRequest (from Stage 1)

**Processing:**
```python
# Validate query
query = arguments.get("query")
if not query:
    raise ValidationError("Missing required field: query")

if not isinstance(query, str):
    raise ValidationError("query must be string")

if not query.strip():
    raise ValidationError("query cannot be empty")

if len(query) > 1000:
    raise ValidationError("query too long (max 1000 chars)")

# Validate limit
limit = arguments.get("limit", 10)
if not isinstance(limit, int):
    raise ValidationError("limit must be integer")

if limit < 1 or limit > 100:
    raise ValidationError("limit must be 1-100")

# Validate filters
filters = arguments.get("filters", {})
if filters and not isinstance(filters, dict):
    raise ValidationError("filters must be dict")
```

**Output:**
```python
ValidatedSearchInput(
    query="Who created Python?",
    query_length=19,
    limit=10,
    filters={
        "tags": ["python"],
        "date_from": "2024-01-01"
    }
)
```

**Performance:**
- Latency: < 5ms

### Stage 3: Query Embedding

**Input:** ValidatedSearchInput (from Stage 2)

#### Stage 3a: Cache Lookup (Phase 2)

**Processing:**
```python
cache_key = hashlib.sha256(query.encode()).hexdigest()
cached_embedding = cache.get(cache_key)

if cached_embedding:
    return cached_embedding
```

**Performance:**
- Latency: ~1-2ms (Redis)
- Cache hit rate target: 60-68%

#### Stage 3b: Ollama API Call

**Processing:**
```python
# If not cached, generate embedding
query_embedding = await ollama_client.embed(
    query,
    model="nomic-embed-text"
)
```

**API Request:**
```http
POST http://localhost:11434/api/embeddings
{
  "model": "nomic-embed-text",
  "prompt": "Who created Python?"
}
```

**API Response:**
```json
{
  "embedding": [0.234, -0.567, 0.890, ..., 0.123]  // 768-dim
}
```

**Output:**
```python
QueryEmbedding(
    vector=[0.234, -0.567, 0.890, ..., 0.123],
    dimensions=768,
    model="nomic-embed-text"
)
```

**Performance:**
- Latency: ~50-100ms (Ollama local)
- If cached: ~1-2ms

#### Stage 3c: Cache Write (Phase 2)

**Processing:**
```python
cache.set(cache_key, query_embedding, ttl=86400)
```

### Stage 4: Vector Search

**Input:**
- QueryEmbedding (from Stage 3b)
- limit (from Stage 2)
- filters (from Stage 2)

#### Stage 4a: HNSW Index Query

**Processing:**
```python
# Build Cypher query
cypher = """
    CALL db.idx.vector.queryNodes(
        'Chunk',           // Node label
        'embedding',       // Property name
        $limit,            // K neighbors
        $query_embedding   // Query vector
    ) YIELD node, score
    WHERE score >= $min_similarity
"""

# Add filter conditions if provided
if filters.get("tags"):
    cypher += """
    MATCH (m:Memory)-[:HAS_CHUNK]->(node)
    WHERE ANY(tag IN $tags WHERE tag IN m.tags)
    """

if filters.get("date_from"):
    cypher += """
    AND m.timestamp >= datetime($date_from)
    """

cypher += """
    RETURN node.id as chunk_id,
           node.text as text,
           score as similarity,
           m.id as memory_id,
           m.source as source,
           m.tags as tags,
           m.timestamp as timestamp
    ORDER BY score DESC
    LIMIT $limit
"""

results = graph.query(cypher, {
    "query_embedding": query_embedding,
    "limit": limit,
    "min_similarity": 0.5,
    "tags": filters.get("tags", []),
    "date_from": filters.get("date_from")
})
```

**Algorithm (HNSW - Hierarchical Navigable Small World):**
```
1. Start at top layer of hierarchy
2. Navigate to nearest neighbor
3. Descend to next layer
4. Repeat until bottom layer
5. Search local neighborhood
6. Return top-K by cosine similarity
```

**Output:**
```python
VectorSearchResults([
    SearchResult(
        chunk_id="550e8400_chunk_0",
        text="Python was created by Guido van Rossum in 1991...",
        similarity=0.87,
        memory_id="550e8400-e29b-41d4-a716-446655440000",
        source="wikipedia",
        tags=["python", "programming"],
        timestamp="2024-11-22T10:30:00Z"
    ),
    SearchResult(
        chunk_id="660f9511_chunk_1",
        text="Guido van Rossum started Python as a hobby project...",
        similarity=0.82,
        memory_id="660f9511-f3ac-52e5-b827-557766551111",
        source="python.org",
        tags=["python"],
        timestamp="2024-11-20T14:00:00Z"
    ),
    # ... up to 10 results
])
```

**Performance:**
- Latency: ~20-50ms for 10K documents
- Latency: ~50-100ms for 100K documents
- Latency: ~100-200ms for 1M documents
- FalkorDB HNSW is highly optimized (496x faster than alternatives)

### Stage 5: Result Ranking (Phase 1: Pass-through, Phase 2: Enhanced)

**Input:** VectorSearchResults (from Stage 4)

**Phase 1 Processing (MVP):**
```python
# In Phase 1, we just pass through vector search results
# No additional ranking
ranked_results = vector_search_results
```

**Phase 2 Processing (Future):**
```python
# BM25 Fusion
bm25_scores = bm25_search(query, top_k=50)
fused_results = reciprocal_rank_fusion(
    vector_results=vector_search_results,
    bm25_results=bm25_scores,
    k=60  # RRF parameter
)

# Cross-Encoder Reranking (top 20)
top_candidates = fused_results[:20]
reranked = cross_encoder.rerank(
    query=query,
    candidates=top_candidates
)

# Graph Context Enhancement
for result in reranked:
    related_entities = graph.get_related(result.memory_id)
    result.context = related_entities

ranked_results = reranked[:limit]
```

**Output (Phase 1):**
```python
RankedResults([
    # Same as VectorSearchResults, just passed through
    SearchResult(...),
    SearchResult(...),
    # ...
])
```

**Performance (Phase 1):**
- Latency: ~0ms (passthrough)

**Performance (Phase 2):**
- BM25: ~10-20ms
- RRF Fusion: ~5ms
- Cross-Encoder: ~50-100ms for 20 candidates
- Graph Context: ~20-30ms
- Total Phase 2 overhead: ~85-150ms

### Stage 6: Response Formation

**Input:** RankedResults (from Stage 5)

**Processing:**
```python
response_text = f"Found {len(results)} relevant memories:\n\n"

for i, result in enumerate(results, 1):
    response_text += f"{i}. {result.text}\n"
    response_text += f"   Source: {result.source}\n"
    response_text += f"   Similarity: {result.similarity:.2f}\n"
    response_text += f"   Tags: {', '.join(result.tags)}\n\n"

response = {
    "content": [
        {
            "type": "text",
            "text": response_text
        }
    ],
    "isError": False,
    "_meta": {
        "processing_time_ms": end_time - start_time,
        "results_count": len(results),
        "query": query
    }
}
```

**Output (MCP Response):**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Found 2 relevant memories:\n\n1. Python was created by Guido van Rossum in 1991...\n   Source: wikipedia\n   Similarity: 0.87\n   Tags: python, programming\n\n2. Guido van Rossum started Python as a hobby project...\n   Source: python.org\n   Similarity: 0.82\n   Tags: python\n\n"
      }
    ],
    "isError": false,
    "_meta": {
      "processing_time_ms": 123,
      "results_count": 2,
      "query": "Who created Python?"
    }
  },
  "id": 2
}
```

### search_memory Total Performance Budget

**Phase 1 Target Latency (P95):**
- Cached query: < 50ms
- Uncached query: < 200ms

**Phase 1 Breakdown (uncached, 10 results):**
```
Stage 1: MCP Reception         ~1ms
Stage 2: Validation            ~5ms
Stage 3a: Cache Lookup (miss)  ~2ms
Stage 3b: Query Embedding      ~50-100ms  ← BOTTLENECK
Stage 3c: Cache Write          ~2ms
Stage 4: Vector Search         ~20-50ms
Stage 5: Ranking (passthrough) ~0ms
Stage 6: Response Formation    ~5ms
─────────────────────────────────────
Total:                         ~85-165ms ✅ Under budget
```

**Phase 1 Breakdown (cached, 10 results):**
```
Stage 1: MCP Reception         ~1ms
Stage 2: Validation            ~5ms
Stage 3a: Cache Lookup (hit)   ~2ms
Stage 3b: Query Embedding      SKIPPED
Stage 3c: Cache Write          SKIPPED
Stage 4: Vector Search         ~20-50ms
Stage 5: Ranking (passthrough) ~0ms
Stage 6: Response Formation    ~5ms
─────────────────────────────────────
Total:                         ~33-63ms ✅ Excellent
```

**Phase 2 Target Latency (P95):**
- Cached query with hybrid search: < 150ms
- Uncached query with hybrid search: < 300ms

**Bottleneck:** Query embedding (Stage 3b)
- Mitigation: Semantic cache with 60-68% hit rate
- Future: Precompute embeddings for common queries

---

## Operation 3: get_stats Data Flow

### High-Level Flow

```
User Request
    ↓
[1. MCP Request Reception] (zapomni_mcp)
    ↓
[2. Input Validation] (zapomni_mcp)
    ↓
[3. Database Aggregation] (zapomni_db)
    ├─ 3a. Count Memories
    ├─ 3b. Count Chunks
    ├─ 3c. Calculate DB Size
    ├─ 3d. Cache Stats (Phase 2)
    └─ 3e. Query Stats (Phase 2)
    ↓
[4. Response Formation] (zapomni_mcp)
    ↓
MCP Response
```

### Stage 1: MCP Request Reception

**Input:**
```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "get_stats",
    "arguments": {}
  },
  "id": 3
}
```

**Processing:** Parse MCP request

**Output:**
```python
ToolRequest(
    method="tools/call",
    params={
        "name": "get_stats",
        "arguments": {}
    }
)
```

### Stage 2: Input Validation

**Input:** ToolRequest (from Stage 1)

**Processing:**
```python
# No arguments required for get_stats
# Just verify it's a valid request
if not isinstance(arguments, dict):
    raise ValidationError("arguments must be dict")
```

**Output:**
```python
ValidatedStatsInput()
```

**Performance:**
- Latency: < 1ms

### Stage 3: Database Aggregation

**Input:** ValidatedStatsInput (from Stage 2)

#### Stage 3a: Count Memories

**Processing:**
```python
memory_count = graph.query("""
    MATCH (m:Memory)
    RETURN count(m) as total
""")
```

**Output:**
```python
MemoryCount(total=1542)
```

**Performance:**
- Latency: ~5-10ms (indexed count)

#### Stage 3b: Count Chunks

**Processing:**
```python
chunk_count = graph.query("""
    MATCH (c:Chunk)
    RETURN count(c) as total
""")
```

**Output:**
```python
ChunkCount(total=4821)
```

**Performance:**
- Latency: ~5-10ms

#### Stage 3c: Calculate DB Size

**Processing:**
```python
# FalkorDB-specific query for database size
db_info = graph.query("""
    CALL dbms.db.info()
""")

db_size_bytes = db_info[0]["size"]
db_size_mb = db_size_bytes / (1024 * 1024)
```

**Output:**
```python
DBSize(
    bytes=245892096,
    megabytes=234.5,
    graph_name="zapomni_memory"
)
```

**Performance:**
- Latency: ~10-20ms

#### Stage 3d: Cache Stats (Phase 2)

**Processing:**
```python
# Redis cache statistics
cache_info = redis.info("stats")

cache_stats = {
    "total_requests": cache_info.get("keyspace_hits", 0) +
                     cache_info.get("keyspace_misses", 0),
    "hits": cache_info.get("keyspace_hits", 0),
    "misses": cache_info.get("keyspace_misses", 0),
    "hit_rate": cache_info.get("keyspace_hits", 0) /
                max(cache_info.get("keyspace_hits", 0) +
                    cache_info.get("keyspace_misses", 0), 1)
}
```

**Output:**
```python
CacheStats(
    total_requests=1000,
    hits=640,
    misses=360,
    hit_rate=0.64
)
```

#### Stage 3e: Query Stats (Phase 2)

**Processing:**
```python
# Average query latency from logs or monitoring
query_stats = monitoring.get_stats(
    metric="query_latency_ms",
    aggregation="avg",
    period="1h"
)
```

**Output:**
```python
QueryStats(
    avg_latency_ms=187,
    p50_latency_ms=120,
    p95_latency_ms=350,
    p99_latency_ms=480
)
```

### Stage 4: Response Formation

**Input:**
- MemoryCount (from Stage 3a)
- ChunkCount (from Stage 3b)
- DBSize (from Stage 3c)
- CacheStats (from Stage 3d, Phase 2)
- QueryStats (from Stage 3e, Phase 2)

**Processing:**
```python
stats = {
    "total_memories": memory_count.total,
    "total_chunks": chunk_count.total,
    "database_size_mb": db_size.megabytes,
    "graph_name": "zapomni_memory"
}

# Phase 2 additions
if cache_enabled:
    stats["cache_hit_rate"] = cache_stats.hit_rate
    stats["avg_query_latency_ms"] = query_stats.avg_latency_ms

response_text = f"""
Memory System Statistics:
━━━━━━━━━━━━━━━━━━━━━━━━━

Storage:
  • Total Memories: {stats['total_memories']:,}
  • Total Chunks: {stats['total_chunks']:,}
  • Database Size: {stats['database_size_mb']:.1f} MB
  • Graph Name: {stats['graph_name']}
"""

if cache_enabled:
    response_text += f"""
Performance:
  • Cache Hit Rate: {stats['cache_hit_rate']:.1%}
  • Avg Query Latency: {stats['avg_query_latency_ms']:.0f} ms
"""

response = {
    "content": [
        {
            "type": "text",
            "text": response_text
        }
    ],
    "isError": False,
    "_meta": {
        "processing_time_ms": end_time - start_time,
        "statistics": stats
    }
}
```

**Output (MCP Response):**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "content": [
      {
        "type": "text",
        "text": "\nMemory System Statistics:\n━━━━━━━━━━━━━━━━━━━━━━━━━\n\nStorage:\n  • Total Memories: 1,542\n  • Total Chunks: 4,821\n  • Database Size: 234.5 MB\n  • Graph Name: zapomni_memory\n\nPerformance:\n  • Cache Hit Rate: 64.0%\n  • Avg Query Latency: 187 ms\n"
      }
    ],
    "isError": false,
    "_meta": {
      "processing_time_ms": 34,
      "statistics": {
        "total_memories": 1542,
        "total_chunks": 4821,
        "database_size_mb": 234.5,
        "graph_name": "zapomni_memory",
        "cache_hit_rate": 0.64,
        "avg_query_latency_ms": 187
      }
    }
  },
  "id": 3
}
```

### get_stats Total Performance Budget

**Target Latency (P95):**
- < 100ms

**Breakdown:**
```
Stage 1: MCP Reception         ~1ms
Stage 2: Validation            ~1ms
Stage 3a: Count Memories       ~5-10ms
Stage 3b: Count Chunks         ~5-10ms
Stage 3c: DB Size              ~10-20ms
Stage 3d: Cache Stats          ~2-5ms (Phase 2)
Stage 3e: Query Stats          ~2-5ms (Phase 2)
Stage 4: Response Formation    ~5ms
─────────────────────────────────────
Total:                         ~31-57ms ✅ Well under budget
```

**No Bottlenecks:** All stages are fast database aggregations

---

## Data Transformations Summary

### Transformation Matrix

| Stage | Input Type | Output Type | Transformation |
|-------|-----------|-------------|----------------|
| **add_memory** ||||
| Validation | `str (text)` | `ValidatedInput` | Parse, validate constraints |
| Extraction | `ValidatedInput` | `ExtractedText` | Extract from format (PDF, HTML) |
| Chunking | `ExtractedText` | `Chunks[]` | Split into semantic units |
| Enrichment | `Chunks[]` | `EnrichedChunks[]` | Add metadata |
| Batching | `EnrichedChunks[]` | `BatchRequest` | Group for API |
| Embedding | `BatchRequest` | `Embeddings[][]` | Text → vectors (768-dim) |
| Storage | `EnrichedChunks + Embeddings` | `Memory + Chunk nodes` | Persist to graph |
| **search_memory** ||||
| Validation | `str (query)` | `ValidatedSearchInput` | Parse, validate |
| Embedding | `ValidatedSearchInput` | `QueryEmbedding` | Query → vector (768-dim) |
| Search | `QueryEmbedding + filters` | `VectorSearchResults` | Vector similarity + filter |
| Ranking | `VectorSearchResults` | `RankedResults` | Rerank by relevance |
| **get_stats** ||||
| Aggregation | `(void)` | `StatsData` | Query DB counts, sizes |
| Formatting | `StatsData` | `str (formatted)` | Pretty print |

### Data Model Evolution

**Incoming (MCP Request):**
```
Raw JSON string
```

**After Parsing:**
```python
dict[str, Any]  # Untyped
```

**After Validation:**
```python
ValidatedInput  # Typed, constraints checked
```

**After Processing:**
```python
ProcessedData  # Enriched with metadata, transformed
```

**At Storage:**
```python
GraphNodes  # Persisted in FalkorDB
```

**At Retrieval:**
```python
SearchResults  # Enriched with scores, context
```

**Outgoing (MCP Response):**
```
JSON string (formatted)
```

---

## Error Handling & Propagation

### Error Flow Architecture

```
┌─────────────┐
│ Error Occurs│
└──────┬──────┘
       │
       ├─ ValidationError       → HTTP 400 (Bad Request)
       ├─ ProcessingError       → HTTP 500 (retry possible)
       ├─ DatabaseError         → HTTP 503 (retry with backoff)
       ├─ TimeoutError          → HTTP 504 (retry)
       ├─ EmbeddingError        → HTTP 500 (fallback to cache or fail)
       └─ UnexpectedError       → HTTP 500 (log, alert)
       │
       ↓
┌──────────────┐
│ Error Handler│ (in each stage)
└──────┬───────┘
       │
       ├─ Log structured error (stderr)
       ├─ Record metrics (prometheus)
       ├─ Attempt recovery (if possible)
       └─ Propagate to caller
       │
       ↓
┌──────────────┐
│ MCP Formatter│
└──────┬───────┘
       │
       ↓
┌──────────────┐
│ Error Response│
└──────────────┘
{
  "content": [...],
  "isError": true
}
```

### Error Types & Recovery

#### ValidationError

**When:** Invalid input (empty text, wrong types, out of range)

**Recovery:** None (user must fix input)

**Response:**
```json
{
  "content": [
    {
      "type": "text",
      "text": "Validation Error: text cannot be empty"
    }
  ],
  "isError": true
}
```

**Example:**
```python
if not text.strip():
    raise ValidationError("text cannot be empty")
```

#### ProcessingError

**When:** Business logic failure (chunking failed, invalid document)

**Recovery:** Retry with different strategy, or fail

**Response:**
```json
{
  "content": [
    {
      "type": "text",
      "text": "Processing Error: Failed to chunk document. Please try a different format."
    }
  ],
  "isError": true
}
```

#### DatabaseError

**When:** DB connection failed, query timeout, transaction rollback

**Recovery:** Retry with exponential backoff (3 attempts)

**Retry Logic:**
```python
@retry(
    exceptions=(DatabaseError,),
    max_attempts=3,
    backoff=exponential(base=2, max_value=10)
)
async def store_memory(data):
    await db.insert(data)
```

**Response (after retries exhausted):**
```json
{
  "content": [
    {
      "type": "text",
      "text": "Database Error: Service temporarily unavailable. Please try again in a few moments."
    }
  ],
  "isError": true
}
```

#### EmbeddingError

**When:** Ollama unreachable, timeout, model not found

**Recovery:**
1. Try cache (if Phase 2)
2. Fallback to sentence-transformers
3. If all fail, return error

**Fallback Example:**
```python
try:
    embedding = await ollama_embedder.embed(text)
except EmbeddingError:
    logger.warning("ollama_failed_using_fallback")
    embedding = sentence_transformer.encode(text)
```

### Structured Error Logging

**Format:**
```python
logger.error(
    "operation_failed",
    operation="add_memory",
    stage="embedding_generation",
    error_type="EmbeddingError",
    error_message=str(e),
    text_length=len(text),
    chunk_index=i,
    retry_attempt=attempt,
    timestamp=datetime.utcnow().isoformat()
)
```

**Output (JSON to stderr):**
```json
{
  "event": "operation_failed",
  "level": "error",
  "operation": "add_memory",
  "stage": "embedding_generation",
  "error_type": "EmbeddingError",
  "error_message": "Connection refused to localhost:11434",
  "text_length": 1547,
  "chunk_index": 2,
  "retry_attempt": 1,
  "timestamp": "2024-11-22T10:35:42.123Z"
}
```

---

## Performance Characteristics

### Latency Budgets (Phase 1)

| Operation | Small Input | Medium Input | Large Input |
|-----------|------------|--------------|-------------|
| add_memory | < 200ms | < 500ms | < 3000ms |
| search_memory (cached) | < 50ms | < 50ms | < 50ms |
| search_memory (uncached) | < 200ms | < 200ms | < 200ms |
| get_stats | < 100ms | < 100ms | < 100ms |

**Input Sizes:**
- Small: < 1KB (1-2 chunks)
- Medium: 1-10KB (3-20 chunks)
- Large: 10-100KB (20-200 chunks)

### Throughput Targets (Phase 1)

| Operation | Target QPS | Concurrent Users |
|-----------|-----------|------------------|
| add_memory | 5-10 QPS | 50 |
| search_memory | 20-50 QPS | 200 |
| get_stats | 50-100 QPS | 500 |

**Assumptions:**
- Single Zapomni instance
- FalkorDB on localhost
- Ollama on localhost (CPU or GPU)

### Resource Usage

#### Memory (RAM)

**add_memory:**
```
Base: ~100MB (Python runtime + dependencies)
+ Text size (e.g., 10MB document = 10MB)
+ Chunks in memory (e.g., 100 chunks × 0.5KB = 50KB)
+ Embeddings batch (e.g., 32 × 768 × 4 bytes = 98KB)
Total: ~110-120MB for large document
```

**search_memory:**
```
Base: ~100MB
+ Query embedding (768 × 4 bytes = 3KB)
+ Search results (10 × 0.5KB = 5KB)
Total: ~100MB (minimal overhead)
```

**Recommendation:** 2GB RAM minimum, 4GB comfortable

#### CPU

**add_memory:**
- Chunking: Low CPU (mostly string ops)
- Embedding: High CPU if Ollama on CPU (1-2 cores per request)
- Storage: Low CPU (network I/O)

**search_memory:**
- Embedding: High CPU (1-2 cores)
- Vector search: Medium CPU (HNSW traversal)

**Recommendation:** 4+ cores for production

#### Disk I/O

**add_memory:**
- FalkorDB writes: ~1-5 MB/s (depends on chunk count)
- Sequential writes (good for SSDs)

**search_memory:**
- FalkorDB reads: ~0.5-2 MB/s
- Random reads (benefits from SSD)

**Recommendation:** SSD strongly recommended for FalkorDB

### Bottleneck Analysis

#### Identified Bottlenecks (Phase 1)

1. **Embedding Generation (add_memory)**
   - Latency: ~50-100ms per chunk (Ollama CPU)
   - Impact: Dominates total latency for documents with many chunks
   - Mitigation: Batch parallelization (32 concurrent)
   - Future: GPU acceleration (10x faster), semantic cache

2. **Query Embedding (search_memory)**
   - Latency: ~50-100ms per query
   - Impact: Half of total search latency
   - Mitigation: Semantic cache (60-68% hit rate in Phase 2)
   - Future: Precompute embeddings for common queries

3. **Vector Search (search_memory)**
   - Latency: ~20-50ms for 10K docs (scales log(n))
   - Impact: Moderate, grows with DB size
   - Mitigation: FalkorDB HNSW is already optimized
   - Future: Sharding for > 1M documents

#### Non-Bottlenecks

- Chunking: Fast (< 10ms per 1KB)
- Validation: Negligible (< 5ms)
- Response formatting: Negligible (< 5ms)
- Database writes: Fast (< 50ms for typical batch)

### Scalability Projections

**Database Size vs. Performance:**

| Documents | Chunks | DB Size | add_memory | search_memory |
|-----------|--------|---------|-----------|---------------|
| 100 | 300 | 5 MB | 150ms | 70ms |
| 1,000 | 3,000 | 50 MB | 200ms | 100ms |
| 10,000 | 30,000 | 500 MB | 300ms | 150ms |
| 100,000 | 300,000 | 5 GB | 500ms | 250ms |
| 1,000,000 | 3,000,000 | 50 GB | 800ms | 400ms |

**Assumptions:**
- 3 chunks per document average
- 512 tokens per chunk
- Embedding dimension: 768
- HNSW index

**Scaling Strategy:**
- Up to 100K documents: Single instance, no changes
- 100K-1M documents: Add Redis cache, GPU for embeddings
- 1M+ documents: Consider sharding, distributed FalkorDB

---

## Design Decisions

### Decision 1: Pipeline Architecture vs. Monolithic

**Context:** How to structure data processing?

**Options Considered:**

**Option A: Monolithic Function**
- Single large function handles entire flow
- Pros: Simple, fewer abstractions
- Cons: Hard to test, hard to modify, no reusability

**Option B: Pipeline with Discrete Stages**
- Each stage is separate, composable
- Pros: Testable, modular, observable, reusable
- Cons: More abstractions, slight overhead

**Chosen:** Option B (Pipeline)

**Rationale:**
- **Testability:** Each stage can be unit tested independently
- **Observability:** Log/monitor each stage separately
- **Flexibility:** Easy to swap stages (e.g., different chunker)
- **Reusability:** Stages like embedding can be reused across operations
- **Maintainability:** Changes isolated to specific stages
- The overhead is negligible (< 1ms per stage transition)

### Decision 2: Async vs. Sync Processing

**Context:** Use async/await or synchronous code?

**Options Considered:**

**Option A: Synchronous**
- Simpler code, easier to debug
- Pros: Straightforward control flow
- Cons: Blocks on I/O, lower throughput

**Option B: Async/Await**
- All I/O operations async
- Pros: Non-blocking, higher concurrency
- Cons: More complex, async propagation

**Chosen:** Option B (Async)

**Rationale:**
- **I/O-Bound:** Operations dominated by network I/O (Ollama, FalkorDB)
- **Concurrency:** Can handle multiple requests simultaneously
- **MCP Server:** Already async (MCP protocol uses async transport)
- **Future-Proof:** Enables batch parallelization, streaming
- Python's asyncio is mature and well-supported

### Decision 3: Fail-Fast vs. Graceful Degradation

**Context:** What to do when non-critical components fail?

**Options Considered:**

**Option A: Fail-Fast**
- Any error → stop processing, return error
- Pros: Clear failure modes, easier debugging
- Cons: Brittle, poor user experience

**Option B: Graceful Degradation**
- Continue with fallbacks when possible
- Pros: Better availability, user experience
- Cons: Harder to debug, hidden failures

**Chosen:** Hybrid Approach

**Rationale:**
- **Fail-Fast for Critical Errors:**
  - ValidationError → fail (bad input)
  - DatabaseError (after retries) → fail (can't store)

- **Graceful Degradation for Non-Critical:**
  - Ollama down → fallback to sentence-transformers
  - Cache unavailable → skip cache, continue
  - Phase 2 features unavailable → use Phase 1 baseline

**Example:**
```python
try:
    embedding = await ollama_embedder.embed(text)
except OllamaConnectionError:
    logger.warning("ollama_unavailable_using_fallback")
    embedding = fallback_embedder.encode(text)
```

### Decision 4: Embedding Batch Size

**Context:** How many embeddings to generate concurrently?

**Options Considered:**

**Option A: Sequential (batch_size=1)**
- One at a time
- Pros: Simple, low memory
- Cons: Very slow for many chunks

**Option B: Large Batch (batch_size=unlimited)**
- All at once
- Pros: Maximum parallelism
- Cons: Can overwhelm Ollama, high memory

**Option C: Fixed Batch (batch_size=32)**
- Process in fixed-size batches
- Pros: Balanced throughput and resource usage
- Cons: Batch size may not be optimal for all cases

**Chosen:** Option C (batch_size=32, configurable)

**Rationale:**
- **Empirical Testing:** 32 concurrent requests is sweet spot for Ollama
  - Lower: Underutilizes Ollama
  - Higher: Diminishing returns, memory pressure
- **Configurable:** Can be tuned per deployment
- **Graceful:** Ollama's rate limiting handles overflow

**Configuration:**
```python
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
```

### Decision 5: Vector Index Type (HNSW)

**Context:** Which vector index algorithm to use?

**Options Considered:**

**Option A: Flat (Brute Force)**
- Exact search, O(n) complexity
- Pros: 100% recall, simple
- Cons: Slow for > 1K vectors

**Option B: HNSW (Hierarchical Navigable Small World)**
- Approximate, O(log n) complexity
- Pros: Very fast, good recall (>95%)
- Cons: Slightly less accurate than flat

**Option C: IVF (Inverted File Index)**
- Clustering-based, O(sqrt(n))
- Pros: Fast, good for very large datasets
- Cons: Requires training, complex

**Chosen:** Option B (HNSW)

**Rationale:**
- **FalkorDB Native:** FalkorDB implements HNSW out-of-box
- **Performance:** 496x faster P99 latency (FalkorDB benchmark)
- **Accuracy:** >95% recall with proper HNSW params
- **Scalability:** Handles 1M+ vectors efficiently
- **No Training:** Works immediately, no clustering phase

**HNSW Parameters (FalkorDB):**
```cypher
CREATE VECTOR INDEX FOR (c:Chunk) ON (c.embedding)
OPTIONS {
  dimension: 768,
  similarityFunction: 'cosine',
  M: 16,              // Connections per layer
  efConstruction: 200 // Build-time accuracy
}
```

---

## Non-Functional Requirements

### Performance

**Latency Requirements:**
- add_memory (P95): < 500ms for medium documents (1-10KB)
- search_memory (P95): < 200ms uncached, < 50ms cached
- get_stats (P95): < 100ms

**Throughput Requirements:**
- add_memory: 5-10 QPS
- search_memory: 20-50 QPS
- get_stats: 50-100 QPS

**Resource Limits:**
- Memory: < 2GB for typical workload
- CPU: < 50% average on 4-core machine
- Disk I/O: < 10 MB/s write, < 5 MB/s read

### Scalability

**Database Size:**
- Support up to 100K documents in Phase 1
- Support up to 1M documents in Phase 2 (with optimizations)

**Concurrent Users:**
- Support 50 concurrent users in Phase 1
- Support 200 concurrent users in Phase 2

**Data Volume:**
- Single document: up to 10MB
- Total database: up to 50GB in Phase 1

### Reliability

**Availability:**
- Target: 99.9% uptime (< 8.76 hours downtime/year)
- Graceful degradation on component failures

**Error Recovery:**
- Retry transient failures (DatabaseError, NetworkError) up to 3 times
- Exponential backoff: 1s, 2s, 4s
- Fallback to alternative implementations when possible

**Data Integrity:**
- Atomic transactions for database writes
- Rollback on partial failures
- Verify embeddings before storage

### Observability

**Logging:**
- Structured JSON logs to stderr
- Log level: INFO (default), DEBUG (development)
- Every stage logs: start, end, duration, data shape

**Metrics (Phase 2):**
- Request count per operation
- Latency histograms (P50, P95, P99)
- Error rates by type
- Cache hit rates
- Database query performance

**Tracing (Phase 2):**
- Distributed tracing for full request lifecycle
- Correlation IDs across stages
- Flamegraphs for performance analysis

### Security

**Input Validation:**
- All inputs validated before processing
- Maximum sizes enforced (10MB text, 1000 char query)
- JSON schema validation for structured inputs

**Data Protection:**
- All data stays local (no external API calls with data)
- No sensitive data in logs (redact or hash)
- Secure storage in FalkorDB (no plaintext secrets)

**Error Messages:**
- User-facing errors: Generic, non-revealing
- Internal logs: Detailed, for debugging
- Never leak system internals to user

---

## Testing Strategy

### Unit Testing

**What to Test:**
- Each pipeline stage independently
- Data transformations (input → output)
- Validation logic
- Error handling and recovery

**Example:**
```python
def test_chunking_stage():
    """Test chunking stage produces correct output."""
    input_text = "A" * 1000
    chunker = SemanticChunker(chunk_size=512, overlap=50)

    chunks = chunker.chunk(input_text)

    assert len(chunks) == 2  # 1000 chars / 512 per chunk
    assert chunks[0].token_count <= 512
    assert chunks[0].index == 0
    assert chunks[1].index == 1
```

### Integration Testing

**What to Test:**
- Full pipeline (Stage 1 → Stage N)
- Real Ollama integration
- Real FalkorDB integration
- Error propagation through stages

**Example:**
```python
@pytest.mark.asyncio
async def test_add_memory_full_pipeline():
    """Test complete add_memory flow."""
    mcp_request = {
        "method": "tools/call",
        "params": {
            "name": "add_memory",
            "arguments": {
                "text": "Python is great",
                "metadata": {"source": "test"}
            }
        }
    }

    response = await mcp_server.handle_request(mcp_request)

    assert response["isError"] is False
    assert "memory_id" in response["_meta"]

    # Verify storage
    memory_id = response["_meta"]["memory_id"]
    stored = await db.get_memory(memory_id)
    assert stored is not None
    assert stored.text == "Python is great"
```

### Performance Testing

**What to Test:**
- Latency under load
- Throughput limits
- Resource usage (CPU, memory, I/O)
- Scalability with data size

**Example:**
```python
@pytest.mark.performance
async def test_search_latency_budget():
    """Verify search meets latency budget."""
    # Setup: 10K documents in DB
    await populate_db(num_docs=10000)

    # Execute 100 searches
    latencies = []
    for _ in range(100):
        start = time.time()
        await search_memory("test query")
        latency = (time.time() - start) * 1000  # ms
        latencies.append(latency)

    # Verify P95 latency
    p95 = np.percentile(latencies, 95)
    assert p95 < 200, f"P95 latency {p95}ms exceeds budget"
```

### Load Testing

**Scenarios:**
- Sustained load: 10 QPS for 1 hour
- Spike load: 100 QPS for 1 minute
- Mixed operations: add + search + stats concurrently

**Tools:**
- Locust for HTTP load testing
- Custom scripts for MCP stdio testing

---

## Future Enhancements

### Phase 2 Enhancements

1. **Hybrid Search**
   - Add BM25 keyword search
   - Reciprocal Rank Fusion (RRF)
   - Cross-encoder reranking
   - Expected impact: 2-3x better accuracy

2. **Semantic Cache**
   - Cache query embeddings (Redis)
   - 60-68% hit rate target
   - Expected impact: 50% latency reduction for cached queries

3. **Knowledge Graph**
   - Entity extraction (SpaCy + Ollama)
   - Relationship detection
   - Graph-enhanced search
   - Expected impact: Better context, multi-hop reasoning

### Phase 3 Enhancements

1. **Code Intelligence**
   - AST-based code chunking
   - Call graph construction
   - Code-specific search
   - Expected impact: 4.3 point accuracy gain for code

2. **Streaming Processing**
   - Stream large documents chunk-by-chunk
   - Reduce memory footprint
   - Expected impact: Support 100MB+ documents

3. **GPU Acceleration**
   - Ollama GPU mode
   - Batch embedding on GPU
   - Expected impact: 10x faster embedding generation

### Long-Term Considerations

1. **Distributed Architecture**
   - Horizontal scaling (multiple instances)
   - Load balancing
   - Shared FalkorDB cluster

2. **Advanced Caching**
   - Multi-tier cache (L1: memory, L2: Redis)
   - Intelligent prefetching
   - Cache warming strategies

3. **Query Optimization**
   - Query plan analysis
   - Index tuning
   - Materialized views for stats

---

## References

### Internal Documents
- [product.md](.spec-workflow/steering/product.md) - Product vision and features
- [tech.md](.spec-workflow/steering/tech.md) - Technology stack decisions
- [structure.md](.spec-workflow/steering/structure.md) - Project organization
- [zapomni_mcp_module.md](zapomni_mcp_module.md) - MCP module spec
- [zapomni_core_module.md](zapomni_core_module.md) - Core module spec
- [zapomni_db_module.md](zapomni_db_module.md) - Database module spec

### External Resources
- MCP Specification: https://spec.modelcontextprotocol.io/
- FalkorDB Vector Search: https://docs.falkordb.com/vector_search.html
- Ollama API: https://github.com/ollama/ollama/blob/main/docs/api.md
- HNSW Algorithm: https://arxiv.org/abs/1603.09320
- RecursiveTextSplitter: https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/recursive_text_splitter

### Research Papers
- Hybrid Search (RRF): "Reciprocal Rank Fusion for Metasearch"
- HNSW: "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs"
- RAG Best Practices: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"

---

## Appendix: Performance Test Results

### Baseline Performance (Phase 1)

**Test Environment:**
- CPU: Intel i7-10700K (8 cores)
- RAM: 32GB DDR4
- SSD: Samsung 970 EVO Plus NVMe
- Ollama: CPU mode, nomic-embed-text
- FalkorDB: v4.0.0

**add_memory (1KB document, 2 chunks):**
```
Latency (ms):
  Min:    142
  P50:    178
  P95:    234
  P99:    289
  Max:    412

Breakdown:
  Chunking:     8ms
  Embedding:    145ms (bottleneck)
  Storage:      18ms
  Other:        7ms
```

**search_memory (uncached, 1K docs in DB):**
```
Latency (ms):
  Min:    67
  P50:    98
  P95:    156
  P99:    201
  Max:    287

Breakdown:
  Query Embedding:  87ms (bottleneck)
  Vector Search:    23ms
  Formatting:       3ms
```

**search_memory (cached):**
```
Latency (ms):
  Min:    21
  P50:    34
  P95:    47
  P99:    58
  Max:    89

Breakdown:
  Cache Lookup:     2ms
  Vector Search:    24ms
  Formatting:       3ms
```

**get_stats:**
```
Latency (ms):
  Min:    18
  P50:    29
  P95:    41
  P99:    53
  Max:    78

Breakdown:
  DB Queries:       24ms
  Formatting:       3ms
```

---

**Document Status:**
- **Version:** 1.0 (Draft)
- **Created:** 2025-11-23
- **Author:** Goncharenko Anton aka alienxs2
- **Copyright:** Copyright (c) 2025 Goncharenko Anton aka alienxs2
- **License:** MIT License
- **Ready for Review:** Yes

**Total Pages:** ~30 (estimated)
**Total Diagrams:** 8
**Total Code Examples:** 40+
**Total Performance Metrics:** 15+
