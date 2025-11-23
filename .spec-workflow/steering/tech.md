# Technical Architecture: Zapomni

**Document Version**: 1.0
**Created**: 2025-11-22
**Authors**: Goncharenko Anton aka alienxs2 + Claude Code
**Copyright**: Copyright (c) 2025 Goncharenko Anton aka alienxs2
**License**: MIT License
**Status**: Draft
**Alignment**: Consistent with product.md vision

---

## Executive Summary

### System Overview

Zapomni is a **local-first MCP (Model Context Protocol) memory server** that provides AI agents with intelligent, contextual, and private long-term memory. Built on a unified vector + graph database architecture (FalkorDB), powered by local LLM runtime (Ollama), and implemented in Python 3.10+, Zapomni delivers enterprise-grade RAG capabilities with zero external dependencies and guaranteed data privacy.

The system combines semantic search, knowledge graph intelligence, and code-aware analysis in a single, cohesive architecture optimized for AI agent workflows. By leveraging proven open-source technologies and modern RAG best practices, Zapomni achieves production-quality performance while maintaining complete local operation.

### Key Technical Decisions

#### 1. FalkorDB (Unified Vector + Graph Database)

**Decision**: Use FalkorDB as the sole database, eliminating separate vector and graph databases.

**Rationale**:
- **Unified Storage**: Combines vector embeddings and property graph in one system, eliminating synchronization complexity
- **Superior Performance**: 496x faster P99 latency vs alternatives (GraphRAG benchmark by FalkorDB)
- **Memory Efficiency**: 6x better memory usage compared to separate database architectures
- **Redis Protocol**: Familiar, battle-tested interface with excellent client libraries
- **GraphRAG Optimized**: Native support for hybrid vector + graph queries
- **Operational Simplicity**: Single database connection, single backup strategy, single monitoring system

**Alternatives Considered**:
- ChromaDB (vector) + Neo4j (graph): More mature but complex sync, data duplication
- Qdrant (vector) + NetworkX (in-memory graph): Good performance but separate systems
- Weaviate: All-in-one but commercial features, resource-intensive

**Trade-offs Accepted**:
- **Pros**: Simpler architecture, better performance, lower operational overhead
- **Cons**: Newer technology (smaller community than Neo4j), less mature ecosystem
- **Risk Mitigation**: Database abstraction layer allows fallback to ChromaDB + Neo4j if needed

---

#### 2. Ollama (Local LLM Runtime)

**Decision**: Use Ollama for both embeddings and LLM inference, exclusively local operation.

**Rationale**:
- **Best Local Experience**: Easiest setup and model management for local LLMs
- **Dual Purpose**: Provides both embedding generation and LLM inference in single runtime
- **API Compatibility**: OpenAI-compatible REST API for embeddings
- **Large Model Library**: 100+ models, easy pull and update
- **Zero Cost**: No API fees, no rate limits, no external dependencies
- **Privacy Guarantee**: Data never leaves local machine

**Alternatives Considered**:
- OpenAI API: Better quality but $$$, privacy concerns, requires internet
- llama.cpp: Lower-level control but complex integration, manual model management
- vLLM: Server-focused, overkill for local single-user deployment
- Direct model loading (Hugging Face): Memory-intensive, complex model lifecycle

**Trade-offs Accepted**:
- **Pros**: Simple, well-maintained, good API, local-first aligned
- **Cons**: Sequential embedding processing (no batch API), slightly slower than cloud
- **Risk Mitigation**: Fallback to sentence-transformers for embeddings if Ollama fails

---

#### 3. Python 3.10+ (Programming Language)

**Decision**: Implement entire system in Python 3.10 or newer.

**Rationale**:
- **ML/AI Ecosystem**: Best-in-class libraries (sentence-transformers, SpaCy, LangChain)
- **MCP SDK**: Official Python SDK from Anthropic with excellent support
- **FalkorDB Client**: Native Python support via falkordb-py package
- **Proven Success**: Cognee demonstrates Python viability for similar systems
- **Community**: Largest AI developer community, abundant resources and examples
- **Type Safety**: Modern type hints (3.10+) provide excellent tooling and IDE support

**Alternatives Considered**:
- TypeScript: Good MCP support (Claude Context uses), but weaker ML ecosystem
- Go: Fast execution but limited AI libraries and LLM integrations
- Rust: Maximum performance but steep learning curve, longer development time

**Trade-offs Accepted**:
- **Pros**: Ecosystem, productivity, rapid development, type hints
- **Cons**: Performance vs compiled languages, GIL limitations for CPU-bound tasks
- **Risk Mitigation**: Async I/O for concurrency, profiling for hot path optimization

---

#### 4. MCP Stdio Protocol (Communication Layer)

**Decision**: Use stdio transport as default, JSON-RPC 2.0 over standard input/output.

**Rationale**:
- **Simplicity**: Easiest to implement, debug, and maintain
- **Security**: Process isolation by default, no network exposure
- **Standard**: JSON-RPC 2.0 specification, well-documented
- **Compatibility**: Works with Claude CLI, Claude Desktop, Cursor, Cline out of the box
- **Debugging**: All messages visible in stderr logs, easily reproducible

**Alternatives Considered**:
- HTTP transport: Adds network complexity, requires authentication, overkill for local
- SSE (Server-Sent Events): For streaming responses, useful for future but unnecessary for MVP
- WebSockets: Bidirectional but complex, not needed for request-response pattern

**Trade-offs Accepted**:
- **Pros**: Simple, secure, debuggable, no configuration needed
- **Cons**: Single user per instance, no remote access, no streaming (initially)
- **Future**: Can add HTTP/SSE transport in Phase 5+ for multi-user or web deployments

---

## Architecture Overview

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER / AI AGENT                              │
│                   (Claude CLI, Cursor, Cline)                       │
└────────────────────────────┬────────────────────────────────────────┘
                             │ MCP Protocol (stdio)
                             │ JSON-RPC 2.0
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     ZAPOMNI MCP SERVER                              │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  MCP Interface Layer                                          │ │
│  │  - Stdio Transport (default)                                  │ │
│  │  - Tool Definitions (3-10 functions)                          │ │
│  │  - Pydantic Validation                                        │ │
│  │  - Error Handling & Logging (stderr)                          │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                             │                                       │
│                             ▼                                       │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  Memory Processing Engine                                     │ │
│  │                                                                │ │
│  │  ┌──────────────────────────────────────────────────────┐    │ │
│  │  │  Document Processor                                   │    │ │
│  │  │  - PDF, DOCX, MD, TXT extraction                     │    │ │
│  │  │  - Semantic chunking (256-512 tokens, 10-20% overlap)│    │ │
│  │  │  - Metadata extraction                                │    │ │
│  │  │  - Code AST parsing (tree-sitter)                     │    │ │
│  │  └──────────────────────────────────────────────────────┘    │ │
│  │                                                                │ │
│  │  ┌──────────────────────────────────────────────────────┐    │ │
│  │  │  Embedding Generator (Ollama)                         │    │ │
│  │  │  - Model: nomic-embed-text (768 dim, 2048 ctx)       │    │ │
│  │  │  - Batch processing (sequential via Ollama API)      │    │ │
│  │  │  - Semantic cache (Redis)                             │    │ │
│  │  │  - Fallback: all-MiniLM-L6-v2 (speed mode)           │    │ │
│  │  └──────────────────────────────────────────────────────┘    │ │
│  │                                                                │ │
│  │  ┌──────────────────────────────────────────────────────┐    │ │
│  │  │  Entity & Relationship Extractor                      │    │ │
│  │  │  - Hybrid: SpaCy NER (fast) + Ollama LLM (accurate)  │    │ │
│  │  │  - Models: Llama 3.1 / DeepSeek-R1 / Qwen2.5        │    │ │
│  │  │  - Confidence scoring and validation                  │    │ │
│  │  │  - Entity deduplication (fuzzy matching)              │    │ │
│  │  └──────────────────────────────────────────────────────┘    │ │
│  │                                                                │ │
│  │  ┌──────────────────────────────────────────────────────┐    │ │
│  │  │  Hybrid Search Engine                                 │    │ │
│  │  │  - Vector Similarity (cosine, HNSW index)            │    │ │
│  │  │  - BM25 Keyword Search                                │    │ │
│  │  │  - Reciprocal Rank Fusion (RRF)                       │    │ │
│  │  │  - Cross-Encoder Reranking                            │    │ │
│  │  │  - Graph Traversal (Cypher queries)                   │    │ │
│  │  └──────────────────────────────────────────────────────┘    │ │
│  │                                                                │ │
│  │  ┌──────────────────────────────────────────────────────┐    │ │
│  │  │  Background Task Manager                              │    │ │
│  │  │  - Async job queue (cognify, codify)                 │    │ │
│  │  │  - Progress tracking (percentage-based)               │    │ │
│  │  │  - Status monitoring (pending/running/completed)      │    │ │
│  │  │  - Error handling with retry logic                    │    │ │
│  │  └──────────────────────────────────────────────────────┘    │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                             │                                       │
│                             ▼                                       │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  FalkorDB Storage Layer (Unified Vector + Graph)             │ │
│  │                                                                │ │
│  │  ┌────────────────────────┐  ┌─────────────────────────────┐ │ │
│  │  │  Vector Index          │  │  Property Graph             │ │ │
│  │  │  - Embeddings (768d)   │  │  - Nodes: Memory, Entity,   │ │ │
│  │  │  - HNSW/Flat index     │  │    Document, Chunk          │ │ │
│  │  │  - Cosine similarity   │  │  - Edges: MENTIONS,         │ │ │
│  │  │  - Top-K retrieval     │  │    RELATED_TO, HAS_CHUNK,   │ │ │
│  │  │  - Metadata filtering  │  │    CALLS (code)             │ │ │
│  │  └────────────────────────┘  │  - Cypher query engine      │ │ │
│  │                               │  - Graph traversal (depth)  │ │ │
│  │                               └─────────────────────────────┘ │ │
│  │                                                                │ │
│  │  Redis Protocol (port 6379) - Single database, no sync needed │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                             │                                       │
│                             ▼                                       │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  Semantic Cache Layer (Redis)                                 │ │
│  │  - Embedding cache (68% hit rate target)                      │ │
│  │  - Query result cache (LRU eviction)                          │ │
│  │  - Similarity threshold: 0.8                                  │ │
│  │  - TTL: configurable (default 24h)                            │ │
│  └───────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     EXTERNAL SERVICES (Local)                       │
│  ┌─────────────────────┐  ┌────────────────────┐                  │
│  │  Ollama             │  │  FalkorDB          │                  │
│  │  localhost:11434    │  │  localhost:6379    │                  │
│  │  - Embeddings API   │  │  - Graph + Vector  │                  │
│  │  - LLM inference    │  │  - Cypher queries  │                  │
│  └─────────────────────┘  └────────────────────┘                  │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

#### 1. MCP Server Layer

**Purpose**: Accept tool calls from MCP clients, validate inputs, route to appropriate handlers.

**Responsibilities**:
- Listen on stdio for JSON-RPC 2.0 messages
- Parse and validate MCP tool call requests using Pydantic schemas
- Route requests to appropriate business logic handlers
- Return structured JSON responses conforming to MCP spec
- Log all operations to stderr (stdout reserved for MCP protocol)
- Handle errors gracefully with meaningful messages

**Technologies**:
- `mcp` Python SDK (official Anthropic implementation)
- `pydantic` for input/output validation
- `asyncio` for async request handling
- `structlog` for structured logging to stderr

**Configuration**:
```python
{
    "server_name": "zapomni-memory",
    "version": "0.1.0",
    "transport": "stdio",  # Default, can add SSE/HTTP later
    "log_level": "INFO",
    "max_concurrent_tasks": 4  # For parallel processing
}
```

---

#### 2. Document Processor

**Purpose**: Extract text from various document formats and chunk into optimal sizes.

**Supported Formats**:
- PDF: PyMuPDF (fitz) for text extraction
- DOCX: python-docx for Word documents
- Markdown: Direct parsing, preserve structure
- TXT: UTF-8 text files
- HTML: trafilatura for web content extraction (future)

**Chunking Strategy**:
- **Method**: Semantic chunking with LangChain RecursiveCharacterTextSplitter
- **Size**: 256-512 tokens per chunk (configurable)
- **Overlap**: 10-20% (50-100 tokens)
- **Separators**: Hierarchical: `\n\n` (paragraphs) → `\n` (lines) → `. ` (sentences) → ` ` (words)
- **Validation**: Ensure chunks don't break mid-sentence

**Metadata Extraction**:
- `title`: Document title (from filename or metadata)
- `source`: Original file path or URL
- `date`: Creation/modification timestamp
- `section`: Document section (if structured)
- `page_num`: Page number for PDFs
- `language`: Auto-detected language (langdetect)
- `chunk_index`: Position in original document

**Example**:
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=100,
    length_function=len,  # Token count function
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = splitter.split_text(document_text)
```

---

#### 3. Embedding Generator

**Purpose**: Generate vector embeddings for text chunks using local models.

**Primary Model**: nomic-embed-text (via Ollama)
- **Dimensions**: 768
- **Context Length**: 2048 tokens
- **Accuracy**: 81.2% (MTEB benchmark)
- **Languages**: Multilingual including English, Russian
- **API**: Ollama embeddings endpoint

**Fallback Model**: all-MiniLM-L6-v2 (via sentence-transformers)
- **Dimensions**: 384
- **Speed**: ~5x faster than nomic-embed-text
- **Use Case**: Speed mode or Ollama unavailable

**Processing**:
- **Batch Size**: Sequential processing (Ollama limitation)
- **Optimization**: Semantic cache reduces redundant calls
- **Error Handling**: Automatic retry with exponential backoff
- **Monitoring**: Track embedding latency and cache hit rate

**Configuration**:
```python
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_TIMEOUT = 60  # seconds
EMBEDDING_MODEL = "nomic-embed-text"
FALLBACK_EMBEDDING_MODEL = "all-minilm:l6-v2"
```

---

#### 4. Entity & Relationship Extractor

**Purpose**: Extract entities and relationships from text to build knowledge graph.

**Hybrid Approach** (Best of Both Worlds):

**Step 1: Fast NER with SpaCy**
- Model: `en_core_web_lg` (or language-specific)
- Entities: PERSON, ORG, GPE (locations), PRODUCT, DATE, MONEY
- Speed: ~100ms per document
- Use Case: Standard entities

**Step 2: Domain-Specific Extraction with LLM**
- Models: Llama 3.1 8B (default), DeepSeek-R1 (advanced reasoning), Qwen2.5 (code-focused)
- Entities: TECHNOLOGY, CONCEPT, ALGORITHM (domain-specific)
- Format: JSON schema with `format` parameter
- Confidence Scoring: 0-1 range, threshold 0.7

**Step 3: Merge and Deduplicate**
- Fuzzy matching with RapidFuzz (85% similarity threshold)
- Conflict resolution: Higher confidence wins
- Output: Unified entity list with source attribution

**Relationship Detection**:
- LLM-based extraction with few-shot prompting
- Prompt templates: Subject-Predicate-Object triples
- Confidence scoring and validation
- Relationship types: CREATED_BY, USES, RELATED_TO, PART_OF, etc.

---

#### 5. Hybrid Search Engine

**Purpose**: Combine keyword, semantic, and graph-based retrieval for optimal results.

**Three-Stage Retrieval**:

**Stage 1: Vector Similarity Search**
- HNSW index in FalkorDB
- Cosine similarity metric
- Top 50 candidates
- Metadata filtering (date, tags, source)

**Stage 2: BM25 Keyword Search**
- rank-bm25 library
- Tokenized corpus
- Top 50 candidates
- Exact term matching

**Stage 3: Reciprocal Rank Fusion (RRF)**
- Combine vector and BM25 rankings
- Formula: `RRF_score = Σ(1 / (k + rank_i))` where k=60
- No score normalization needed
- Robust to score scale differences

**Stage 4: Cross-Encoder Reranking** (Optional, Phase 2)
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Rerank top 20 candidates
- Query-document pairs processed together
- Final top-K selection

**Graph Enhancement** (Phase 3):
- For top results, traverse graph to find related entities
- Include graph context in results
- Multi-hop reasoning (depth 1-2)

**Performance**:
- **Target**: < 500ms end-to-end latency (P95)
- **Accuracy**: 3.4x better than vector-only (research benchmark)

---

#### 6. Background Task Manager

**Purpose**: Handle long-running operations (graph building, code indexing) asynchronously.

**Design Pattern**: Cognee-inspired async task queue

**Features**:
- Immediate task acceptance with unique task ID
- Asynchronous execution in background
- Progress percentage updates
- Status tracking (pending/running/completed/failed)
- Error handling with retry logic
- Queryable status endpoint

**Implementation**:
```python
import asyncio
import uuid
from enum import Enum

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class BackgroundTask:
    def __init__(self, task_id: str, func, *args, **kwargs):
        self.task_id = task_id
        self.status = TaskStatus.PENDING
        self.progress = 0.0
        self.result = None
        self.error = None
        # Execute in background
        asyncio.create_task(self._run(func, *args, **kwargs))
```

---

#### 7. Semantic Cache Layer

**Purpose**: Reduce embedding computation by caching results.

**Strategies**:

**Exact Match Cache**:
- Key: SHA256 hash of text
- Value: Embedding vector
- Storage: Redis
- TTL: 24 hours (configurable)

**Semantic Cache** (Phase 2):
- Store embeddings with vector similarity search
- Find similar queries (threshold 0.8)
- Reuse cached embeddings for similar text
- Expected hit rate: 60-68% (research benchmark)

**LRU Eviction**:
- Max cache size: 1GB (configurable)
- Least Recently Used eviction policy
- Performance: 20%+ latency reduction

---

### Data Flow

#### Ingestion Flow (add_memory)

```
Text Input
    ↓
[1. Validation]
    - Non-empty text check
    - Size limit check (10MB)
    - Metadata validation
    ↓
[2. Chunking]
    - Semantic chunking (256-512 tokens)
    - 10-20% overlap
    - Metadata preservation
    ↓
[3. Embedding Generation]
    - Check semantic cache
    - Generate via Ollama (if cache miss)
    - Store in cache
    ↓
[4. FalkorDB Storage]
    - Create Memory node
    - Create Chunk nodes
    - Create HAS_CHUNK relationships
    - Store embeddings in vector index
    - Store metadata
    ↓
[5. Confirmation]
    - Return memory_id (UUID)
    - Return chunk count
    - Return text preview
```

**Expected Performance**:
- 100 documents/minute target
- 3-5 seconds per document (including embedding)
- Parallelizable for batch uploads

---

#### Search Flow (search_memory)

```
Query Input
    ↓
[1. Query Embedding]
    - Check cache (exact + semantic)
    - Generate via Ollama (if miss)
    - Cache result
    ↓
[2. Parallel Retrieval]
    ┌─────────────┬─────────────┐
    │ Vector      │ BM25        │
    │ Search      │ Search      │
    │ (Top 50)    │ (Top 50)    │
    └─────────────┴─────────────┘
    ↓
[3. Metadata Filtering]
    - Apply date filters
    - Apply tag filters
    - Apply source filters
    - Apply minimum score threshold
    ↓
[4. Rank Fusion (RRF)]
    - Combine vector + BM25 rankings
    - k=60 constant
    - Score: Σ(1 / (k + rank))
    ↓
[5. Reranking] (Phase 2)
    - Cross-encoder top 20
    - Query-document pairs
    - Final scoring
    ↓
[6. Results]
    - Top K (default 10)
    - Include similarity scores
    - Include metadata
    - Include graph context (Phase 3)
```

**Expected Performance**:
- < 500ms end-to-end (P95)
- Breakdown:
  * Embedding: 50ms (cached) / 150ms (uncached)
  * Vector search: 100ms
  * BM25 search: 50ms
  * Fusion: 50ms
  * Reranking: 150ms (Phase 2)
  * Overhead: 50ms

---

#### Graph Building Flow (build_graph)

```
Memory IDs Input (or all unprocessed)
    ↓
[1. Background Task Creation]
    - Generate task_id
    - Return immediately to client
    - Start async processing
    ↓
[2. Entity Extraction] (async)
    ┌─────────────┬─────────────┐
    │ SpaCy NER   │ Ollama LLM  │
    │ (fast)      │ (accurate)  │
    │ Standard    │ Domain      │
    │ entities    │ entities    │
    └─────────────┴─────────────┘
    ↓
[3. Entity Merging]
    - Fuzzy matching (85% threshold)
    - Deduplication
    - Confidence scoring
    ↓
[4. Relationship Extraction]
    - LLM-based (few-shot prompting)
    - Subject-Predicate-Object triples
    - Confidence scoring
    ↓
[5. Graph Construction]
    - Create Entity nodes in FalkorDB
    - Create MENTIONS edges (Chunk → Entity)
    - Create RELATED_TO edges (Entity → Entity)
    - Update progress percentage
    ↓
[6. Completion]
    - Mark task as completed
    - Store result statistics
    - Update status
```

**Expected Performance**:
- 1K documents in < 10 minutes
- Entity extraction: 80%+ precision target
- Relationship detection: 70%+ precision target

---

## Technology Stack Deep Dive

### 1. FalkorDB - Unified Vector + Graph Database

#### Why FalkorDB?

**Decision Rationale**:

1. **Unified Storage**: Combines vector and graph in single database
   - Eliminates sync complexity between ChromaDB + Neo4j
   - Consistent transactions across vector and graph operations
   - Simpler operational model (one connection, one backup, one monitoring system)

2. **Performance**: 496x faster P99 latency, 6x memory efficiency
   - Benchmark: GraphRAG vs Vector-only RAG accuracy test by FalkorDB/Diffbot
   - GraphBLAS backend optimization for graph operations
   - Redis protocol for low latency operations

3. **Vector + Graph Integration**: Native support for hybrid queries
   - Cypher queries can include vector similarity
   - Graph traversal with embedding context
   - GraphRAG patterns optimized

4. **Redis Protocol**: Familiar, battle-tested communication
   - Port 6379 (standard Redis)
   - Excellent Python client (redis-py compatible)
   - Connection pooling support

**Alternatives Considered**:
- **ChromaDB + Neo4j**: More mature, larger community, but complex sync and data duplication
- **Qdrant + NetworkX**: Good performance, but separate systems requiring coordination
- **Weaviate**: All-in-one vector + search, but commercial features, resource-intensive

**Trade-offs**:
- **Pros**:
  - Unified architecture (simpler)
  - Superior performance (496x faster)
  - Better memory efficiency (6x)
  - Single point of management
  - GraphRAG optimized

- **Cons**:
  - Newer technology (launched 2023)
  - Smaller community than Neo4j
  - Less extensive documentation
  - Fewer third-party integrations

- **Risk Mitigation**: Database abstraction layer allows fallback to ChromaDB + Neo4j if FalkorDB proves problematic

---

#### FalkorDB Configuration

**Connection Settings**:
```python
# Environment variables
FALKORDB_HOST = "localhost"  # or custom host
FALKORDB_PORT = 6379
GRAPH_NAME = "zapomni_memory"
FALKORDB_PASSWORD = ""  # Empty for local, set for production

# Connection pool settings
CONNECTION_POOL_SIZE = 10
CONNECTION_TIMEOUT_SECONDS = 30
SOCKET_KEEPALIVE = True
SOCKET_KEEPALIVE_OPTIONS = {
    "TCP_KEEPIDLE": 60,
    "TCP_KEEPINTVL": 10,
    "TCP_KEEPCNT": 3
}
```

**Schema Design**:

```cypher
// Node Types

// Memory: Top-level document/memory entry
(:Memory {
  id: string,           // UUID
  text: string,         // Original full text
  embedding: vector,    // 768-dim document-level embedding
  tags: [string],       // User-defined tags
  source: string,       // Origin (file path, URL, manual)
  timestamp: datetime,  // Creation time
  chunk_count: int      // Number of chunks
})

// Chunk: Semantic segment of a document
(:Chunk {
  id: string,           // UUID
  text: string,         // Chunk text content
  doc_id: string,       // Parent Memory ID
  index: int,           // Chunk sequence number
  embedding: vector,    // 768-dim chunk embedding
  metadata: map         // Flexible metadata (page_num, section, etc.)
})

// Entity: Extracted named entity or concept
(:Entity {
  id: string,           // UUID
  name: string,         // Entity name (canonical)
  type: string,         // PERSON, ORG, TECHNOLOGY, CONCEPT, etc.
  description: string,  // Entity description
  confidence: float,    // Extraction confidence (0.0-1.0)
  source: string        // Extraction source (spacy, llm)
})

// Document: Higher-level document metadata
(:Document {
  id: string,           // UUID
  title: string,        // Document title
  source: string,       // File path or URL
  date: datetime,       // Document date
  type: string,         // pdf, md, txt, code, etc.
  language: string      // Detected language
})

// Code-specific nodes (Phase 4)
(:Function {
  id: string,
  name: string,         // Function name
  signature: string,    // Full signature
  file_path: string,    // Source file
  language: string,     // Programming language
  docstring: string,    // Function documentation
  start_line: int,
  end_line: int
})

(:Class {
  id: string,
  name: string,
  methods: [string],    // Method names
  file_path: string,
  language: string,
  docstring: string
})

// Edge Types

// Document structure relationships
(:Document)-[:HAS_CHUNK {index: int}]->(:Chunk)

// Entity mentions in text
(:Chunk)-[:MENTIONS {confidence: float, count: int}]->(:Entity)

// Entity-to-entity relationships
(:Entity)-[:RELATED_TO {
  type: string,         // CREATED_BY, USES, PART_OF, etc.
  strength: float,      // Relationship strength (0.0-1.0)
  evidence: string      // Text evidence for relationship
}]->(:Entity)

// Code relationships (Phase 4)
(:Function)-[:CALLS]->(:Function)
(:Class)-[:INHERITS_FROM]->(:Class)
(:Function)-[:DEFINED_IN]->(:Document)
(:Class)-[:CONTAINS]->(:Function)
```

**Indexes**:

```cypher
// Vector index for chunk embeddings (primary search)
CREATE VECTOR INDEX FOR (c:Chunk) ON (c.embedding)
OPTIONS {
  dimension: 768,
  similarityFunction: 'cosine',
  indexType: 'HNSW',
  m: 16,                    // HNSW parameter: connections per layer
  efConstruction: 200,      // Build-time accuracy
  efSearch: 100             // Query-time accuracy
}

// Vector index for memory embeddings (document-level search)
CREATE VECTOR INDEX FOR (m:Memory) ON (m.embedding)
OPTIONS {
  dimension: 768,
  similarityFunction: 'cosine',
  indexType: 'HNSW',
  m: 16,
  efConstruction: 200,
  efSearch: 100
}

// Property indexes for fast filtering
CREATE INDEX FOR (m:Memory) ON (m.id)
CREATE INDEX FOR (m:Memory) ON (m.timestamp)
CREATE INDEX FOR (m:Memory) ON (m.source)
CREATE INDEX FOR (c:Chunk) ON (c.id)
CREATE INDEX FOR (c:Chunk) ON (c.doc_id)
CREATE INDEX FOR (e:Entity) ON (e.name)
CREATE INDEX FOR (e:Entity) ON (e.type)
CREATE INDEX FOR (d:Document) ON (d.id)
CREATE INDEX FOR (d:Document) ON (d.source)
```

**Performance Tuning**:

```python
# HNSW Index Parameters
#
# M (connections per layer):
#   - Higher M = better recall, slower search, more memory
#   - Lower M = faster search, lower recall, less memory
#   - Recommended: 16-32
#
# efConstruction (build-time parameter):
#   - Higher ef = better index quality, slower build
#   - Lower ef = faster build, lower quality
#   - Recommended: 100-400
#
# efSearch (query-time parameter):
#   - Higher ef = better recall, slower queries
#   - Lower ef = faster queries, lower recall
#   - Recommended: 50-200

# For MVP (10K docs):
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 100

# For production (100K+ docs):
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 400
HNSW_EF_SEARCH = 200

# Memory estimation:
# Per vector: dimension * 4 bytes (float32) + HNSW overhead
# HNSW overhead: ~M * 4 bytes per vector
# Example: 10K vectors, 768 dim, M=16
# = 10K * (768 * 4 + 16 * 4) = 10K * 3136 = 31MB
```

#### FalkorDB Client Implementation

```python
from falkordb import FalkorDB
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class FalkorDBClient:
    """FalkorDB client for Zapomni memory storage."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        graph_name: str = "zapomni_memory",
        password: str = ""
    ):
        self.host = host
        self.port = port
        self.graph_name = graph_name
        self.client = None
        self.graph = None

    async def connect(self):
        """Establish connection to FalkorDB."""
        try:
            self.client = FalkorDB(
                host=self.host,
                port=self.port,
                password=self.password if self.password else None
            )
            self.graph = self.client.select_graph(self.graph_name)

            # Initialize schema and indexes
            await self._ensure_schema()

            logger.info(f"Connected to FalkorDB at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"FalkorDB connection failed: {e}")
            raise

    async def _ensure_schema(self):
        """Create indexes and constraints if they don't exist."""
        try:
            # Vector index for chunks
            self.graph.query("""
                CREATE VECTOR INDEX FOR (c:Chunk) ON (c.embedding)
                OPTIONS {
                    dimension: 768,
                    similarityFunction: 'cosine',
                    indexType: 'HNSW',
                    m: 16,
                    efConstruction: 200,
                    efSearch: 100
                }
            """)

            # Property indexes
            indexes = [
                "CREATE INDEX FOR (m:Memory) ON (m.id)",
                "CREATE INDEX FOR (c:Chunk) ON (c.id)",
                "CREATE INDEX FOR (e:Entity) ON (e.name)",
            ]

            for index_query in indexes:
                try:
                    self.graph.query(index_query)
                except:
                    pass  # Index may already exist

            logger.info("Schema initialized")
        except Exception as e:
            logger.warning(f"Schema initialization: {e}")

    async def add_memory(
        self,
        text: str,
        embedding: List[float],
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Add memory to graph database.

        Args:
            text: Memory text content
            embedding: 768-dim embedding vector
            metadata: Optional metadata (tags, source, etc.)

        Returns:
            memory_id: UUID of created memory
        """
        import uuid

        memory_id = str(uuid.uuid4())
        metadata = metadata or {}

        query = """
            CREATE (m:Memory {
                id: $id,
                text: $text,
                embedding: $embedding,
                tags: $tags,
                source: $source,
                timestamp: timestamp(),
                chunk_count: 0
            })
            RETURN m.id as id
        """

        params = {
            "id": memory_id,
            "text": text,
            "embedding": embedding,
            "tags": metadata.get("tags", []),
            "source": metadata.get("source", "user")
        }

        result = self.graph.query(query, params)

        logger.info(f"Memory created: {memory_id}")
        return memory_id

    async def vector_search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        min_score: float = 0.5,
        filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar memories using vector similarity.

        Args:
            query_embedding: Query vector (768 dim)
            limit: Maximum results to return
            min_score: Minimum similarity score (0-1)
            filters: Optional metadata filters

        Returns:
            List of matching memories with scores
        """
        # Build filter clause
        filter_clause = ""
        if filters:
            conditions = []
            if "tags" in filters:
                conditions.append("ANY(tag IN $tags WHERE tag IN node.tags)")
            if "date_from" in filters:
                conditions.append("node.timestamp >= $date_from")
            if "date_to" in filters:
                conditions.append("node.timestamp <= $date_to")
            if "source" in filters:
                conditions.append("node.source = $source")

            if conditions:
                filter_clause = "WHERE " + " AND ".join(conditions)

        query = f"""
            CALL db.idx.vector.queryNodes('Chunk', 'embedding', $limit, $query_embedding)
            YIELD node, score
            {filter_clause}
            RETURN
                node.id as id,
                node.text as text,
                node.doc_id as doc_id,
                node.metadata as metadata,
                score
            ORDER BY score DESC
        """

        params = {
            "query_embedding": query_embedding,
            "limit": limit * 2,  # Fetch more to account for filtering
            **(filters or {})
        }

        result = self.graph.query(query, params)

        # Parse results
        memories = []
        for record in result.result_set:
            if len(record) >= 5:
                score = record[4]
                if score >= min_score:
                    memories.append({
                        "id": record[0],
                        "text": record[1],
                        "doc_id": record[2],
                        "metadata": record[3],
                        "similarity_score": score
                    })

        # Limit to requested count
        return memories[:limit]

    async def graph_traverse(
        self,
        entity_name: str,
        depth: int = 1,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Find entities related to given entity via graph traversal.

        Args:
            entity_name: Starting entity name
            depth: Graph traversal depth (1-3)
            limit: Maximum results

        Returns:
            List of related entities with relationship info
        """
        query = """
            MATCH (e:Entity {name: $entity_name})
            MATCH path = (e)-[:RELATED_TO*1..$depth]-(related:Entity)
            RETURN
                related.name AS name,
                related.type AS type,
                related.description AS description,
                length(path) AS distance,
                relationships(path) AS rels
            ORDER BY distance ASC
            LIMIT $limit
        """

        params = {
            "entity_name": entity_name,
            "depth": depth,
            "limit": limit
        }

        result = self.graph.query(query, params)

        related_entities = []
        for record in result.result_set:
            if len(record) >= 5:
                related_entities.append({
                    "name": record[0],
                    "type": record[1],
                    "description": record[2],
                    "distance": record[3],
                    "relationships": [
                        r.properties for r in record[4]
                    ] if record[4] else []
                })

        return related_entities

    async def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""

        queries = {
            "total_memories": "MATCH (m:Memory) RETURN count(m) as count",
            "total_chunks": "MATCH (c:Chunk) RETURN count(c) as count",
            "total_entities": "MATCH (e:Entity) RETURN count(e) as count",
            "total_relationships": "MATCH ()-[r:RELATED_TO]->() RETURN count(r) as count",
        }

        stats = {
            "graph_name": self.graph_name,
            "database_host": self.host
        }

        for key, query in queries.items():
            result = self.graph.query(query)
            stats[key] = result.result_set[0][0] if result.result_set else 0

        return stats
```

---

### 2. Ollama - Local LLM Runtime

#### Why Ollama?

**Decision Rationale**:

1. **Best Local Experience**: Easiest setup, best model management for local deployment
   - Single command installation: `curl -fsSL https://ollama.com/install.sh | sh`
   - Model management: `ollama pull <model>`, `ollama list`, `ollama rm <model>`
   - Automatic updates: `ollama pull <model>` updates to latest version

2. **Dual Purpose**: Both embeddings and LLM inference in one runtime
   - Embeddings API: `/api/embeddings` endpoint
   - LLM inference: `/api/generate` and `/api/chat` endpoints
   - Consistent API across all models

3. **API Compatibility**: OpenAI-compatible REST API
   - HTTP requests with JSON payloads
   - Async support via `stream: false`
   - Structured output via `format: "json"`

4. **Large Model Library**: 100+ models, easy discovery
   - Browse: https://ollama.com/library
   - Pull with one command: `ollama pull <model>`
   - Community models supported

5. **Zero Cost, Complete Privacy**:
   - No API keys required
   - No rate limits
   - No data sent to external servers
   - Offline capability (after model download)

**Alternatives Considered**:
- **OpenAI API**: Better quality but $$$ costs, privacy concerns, requires internet
- **llama.cpp**: More control but complex integration, manual model management
- **vLLM**: Server-optimized but overkill for local single-user
- **Hugging Face Transformers**: Direct loading but memory-intensive, complex lifecycle

**Trade-offs**:
- **Pros**:
  - Simple installation and usage
  - Well-maintained (official Ollama team)
  - Good API design
  - Local-first aligned
  - Active community

- **Cons**:
  - Sequential processing only (no batch embeddings API)
  - Slightly slower than cloud APIs
  - Requires local compute (CPU/GPU)
  - Model quality varies

- **Risk Mitigation**: Fallback to sentence-transformers for embeddings if Ollama unavailable

---

#### Ollama Configuration

**Models Required**:

```bash
# Phase 1: Embedding model (REQUIRED)
ollama pull nomic-embed-text
# Specs: 768 dimensions, 2048 token context, 81.2% MTEB accuracy
# Size: ~274MB
# Use: Document and query embeddings

# Phase 2: Fallback embedding model (OPTIONAL)
ollama pull all-minilm:l6-v2
# Specs: 384 dimensions, 512 token context, 80% accuracy
# Size: ~67MB
# Use: Speed mode or fallback if nomic-embed-text fails

# Phase 3: LLM for entity extraction (REQUIRED for knowledge graph)
ollama pull llama3.1:8b
# Size: ~4.7GB
# Use: Entity extraction, relationship detection, general inference
# Alternative: deepseek-r1:14b (better reasoning, larger), qwen2.5:7b (code-focused)

# Phase 4: Code-specific model (OPTIONAL)
ollama pull qwen2.5:7b
# Size: ~4.7GB
# Use: Code analysis, function extraction
# Specialization: Strong at code understanding
```

**API Configuration**:

```python
# Ollama server settings
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_TIMEOUT = 60  # seconds for embedding requests
OLLAMA_LLM_TIMEOUT = 120  # seconds for LLM inference
OLLAMA_CONCURRENT_REQUESTS = 1  # Sequential only (Ollama limitation)

# Model selection
EMBEDDING_MODEL = "nomic-embed-text"
FALLBACK_EMBEDDING_MODEL = "all-minilm:l6-v2"
LLM_MODEL = "llama3.1:8b"  # For entity extraction
CODE_LLM_MODEL = "qwen2.5:7b"  # For code analysis (Phase 4)

# Performance settings
EMBEDDING_CACHE_ENABLED = True
LLM_MAX_TOKENS = 2048
LLM_TEMPERATURE = 0.1  # Low temperature for consistent extraction
LLM_TOP_P = 0.9
```

**Hardware Requirements**:

```
Minimum (for MVP with basic models):
- CPU: 4 cores
- RAM: 8GB
- Storage: 10GB free space
- OS: Linux, macOS, Windows

Recommended (for good performance):
- CPU: 8 cores (or GPU)
- RAM: 16GB
- GPU: 8GB+ VRAM (NVIDIA with CUDA preferred)
- Storage: SSD with 50GB free

Optimal (for best performance):
- CPU: 16 cores
- RAM: 32GB
- GPU: NVIDIA RTX 3090/4090 (24GB VRAM)
- Storage: NVMe SSD
```

---

#### Ollama Client Implementation

```python
import httpx
from typing import List, Dict, Any, Optional
import logging
import json

logger = logging.getLogger(__name__)

class OllamaClient:
    """Ollama client for local LLM operations."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        embedding_model: str = "nomic-embed-text",
        llm_model: str = "llama3.1:8b",
        timeout: int = 60
    ):
        self.host = host
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)

    async def embed(self, text: str) -> List[float]:
        """
        Generate embedding for text.

        Args:
            text: Input text (up to 2048 tokens for nomic-embed-text)

        Returns:
            embedding: 768-dim vector (for nomic-embed-text)
        """
        try:
            response = await self.client.post(
                f"{self.host}/api/embeddings",
                json={
                    "model": self.embedding_model,
                    "prompt": text
                }
            )
            response.raise_for_status()

            data = response.json()
            embedding = data.get("embedding")

            if not embedding:
                raise ValueError("No embedding in response")

            logger.debug(f"Generated embedding of dimension: {len(embedding)}")
            return embedding

        except httpx.HTTPError as e:
            logger.error(f"Ollama HTTP error: {e}")
            raise
        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            raise

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (sequential).

        Note: Ollama doesn't support batch embeddings API,
        so this processes sequentially. Use semantic cache
        to mitigate performance impact.

        Args:
            texts: List of input texts

        Returns:
            embeddings: List of 768-dim vectors
        """
        embeddings = []
        for i, text in enumerate(texts):
            embedding = await self.embed(text)
            embeddings.append(embedding)

            if (i + 1) % 10 == 0:
                logger.info(f"Embedded {i + 1}/{len(texts)} texts")

        return embeddings

    async def extract_entities(
        self,
        text: str,
        model: str = None
    ) -> Dict[str, Any]:
        """
        Extract entities and relationships using LLM.

        Args:
            text: Input text to analyze
            model: LLM model to use (default: self.llm_model)

        Returns:
            dict: {
                "entities": [{"name": str, "type": str, "description": str, "confidence": float}],
                "relationships": [{"subject": str, "predicate": str, "object": str, "confidence": float}]
            }
        """
        model = model or self.llm_model

        # Define JSON schema for structured output
        schema = {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "type": {
                                "type": "string",
                                "enum": ["PERSON", "ORGANIZATION", "TECHNOLOGY", "CONCEPT", "EVENT", "LOCATION"]
                            },
                            "description": {"type": "string"},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                        },
                        "required": ["name", "type", "confidence"]
                    }
                },
                "relationships": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string"},
                            "predicate": {"type": "string"},
                            "object": {"type": "string"},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                        },
                        "required": ["subject", "predicate", "object", "confidence"]
                    }
                }
            }
        }

        # Extraction prompt
        prompt = f"""Extract entities and relationships from the following text.

Text:
{text}

Extract:
1. Entities: Important concepts, people, organizations, technologies, events, locations
2. Relationships: How entities relate to each other (e.g., CREATED_BY, USES, PART_OF)

Provide confidence scores (0-1) for each extraction.
Return ONLY valid JSON matching the schema, no other text.
"""

        try:
            response = await self.client.post(
                f"{self.host}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "format": "json",  # Force JSON output
                    "temperature": 0.1,  # Low temp for consistency
                    "stream": False
                },
                timeout=120  # Longer timeout for LLM
            )
            response.raise_for_status()

            data = response.json()
            response_text = data.get("response", "")

            # Parse JSON response
            extracted = json.loads(response_text)

            logger.info(
                f"Extracted {len(extracted.get('entities', []))} entities, "
                f"{len(extracted.get('relationships', []))} relationships"
            )

            return extracted

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            return {"entities": [], "relationships": []}
        except Exception as e:
            logger.error(f"Entity extraction error: {e}")
            return {"entities": [], "relationships": []}

    async def test_connection(self) -> bool:
        """Test Ollama connection and model availability."""
        try:
            # Check Ollama is running
            response = await self.client.get(f"{self.host}/api/tags")
            response.raise_for_status()

            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]

            # Check embedding model
            if self.embedding_model not in model_names:
                logger.warning(
                    f"Embedding model {self.embedding_model} not found. "
                    f"Run: ollama pull {self.embedding_model}"
                )
                return False

            logger.info(f"Ollama connected, model {self.embedding_model} available")
            return True

        except Exception as e:
            logger.error(f"Ollama connection test failed: {e}")
            return False
```

---

#### Ollama Performance Considerations

**Embedding Generation**:
- **Speed**: ~50-100ms per embedding (CPU), ~20-50ms (GPU)
- **Limitation**: Sequential only (no batch API endpoint)
- **Optimization Strategy**: Semantic caching (60%+ hit rate target) reduces calls dramatically
- **Throughput**: ~10-20 embeddings/second (CPU), ~20-50/second (GPU)

**LLM Inference** (Entity Extraction):
- **Speed**: ~10-20 tokens/sec (CPU), ~50-100 tokens/sec (GPU)
- **Context Window**: Up to 8192 tokens (llama3.1), 131K tokens (qwen2.5)
- **Optimization**: Short prompts, low temperature (0.1), structured JSON output
- **Typical Extraction Time**: 5-15 seconds per document

**Memory Usage**:
- **Model Loading**: ~2-5GB RAM per loaded model
- **Concurrent Models**: Can load multiple models simultaneously (RAM permitting)
- **Model Caching**: Models stay in RAM until manually unloaded or system restart

**Hardware Optimization**:
```python
# CPU-only configuration (minimum hardware)
OLLAMA_NUM_THREAD = 4  # CPU threads
OLLAMA_NUM_GPU = 0  # Disable GPU

# GPU configuration (recommended)
OLLAMA_NUM_GPU = 1  # Use GPU
OLLAMA_GPU_LAYERS = 32  # Offload layers to GPU (adjust based on VRAM)
```

---

### 3. Python 3.10+ - Programming Language

#### Why Python?

**Decision Rationale**:

1. **ML/AI Ecosystem Dominance**: Best-in-class libraries
   - `sentence-transformers`: Local embeddings
   - `transformers`: Hugging Face models
   - `LangChain`: Document processing, RAG pipelines
   - `SpaCy`: Fast NLP and NER
   - `tree-sitter`: AST parsing (via bindings)
   - `numpy`, `scipy`: Numerical operations

2. **MCP SDK Native Support**: Official Python SDK from Anthropic
   - Package: `mcp` on PyPI
   - Well-documented, actively maintained
   - Async/await support built-in
   - Type hints for IDE support

3. **FalkorDB & Ollama Clients**: Excellent Python support
   - `falkordb`: Official FalkorDB Python client
   - `httpx`: Modern async HTTP for Ollama API
   - `redis`: If needed for caching

4. **Proven Success**: Cognee demonstrates viability
   - Similar architecture (memory + graph)
   - Production deployments
   - Python works well for this use case

5. **Developer Productivity**: Fast iteration, extensive tooling
   - Type hints (3.10+) for IDE autocomplete
   - `pydantic` for validation
   - `pytest` for testing
   - Rich ecosystem of dev tools

**Alternatives Considered**:
- **TypeScript**: Claude Context uses it successfully, good MCP support, but weaker ML ecosystem
- **Go**: Fast, good concurrency, but limited AI libraries and LLM integrations
- **Rust**: Maximum performance, memory safety, but steep learning curve and longer dev time

**Trade-offs**:
- **Pros**:
  - Ecosystem (unmatched for AI/ML)
  - Productivity (fast development)
  - Type hints (modern Python is type-safe)
  - Community (largest AI dev community)

- **Cons**:
  - Performance vs compiled languages (mitigated with async I/O)
  - GIL for CPU-bound tasks (not a problem for I/O-heavy RAG)
  - Packaging can be complex (solved with modern tools)

- **Risk Mitigation**:
  - Use async I/O for concurrency
  - Profile and optimize hot paths
  - Consider Cython/numba for critical performance sections if needed

---

#### Python Environment

**Version Requirements**:
```python
# Minimum Python version: 3.10
# Recommended: Python 3.11 or 3.12
# Reason: Performance improvements (10-20% faster), better type hints, improved error messages

python_requires = ">=3.10"
```

**Key Dependencies**:

```toml
[project]
name = "zapomni-mcp"
version = "0.1.0"
description = "Local-first MCP memory server with knowledge graphs"
authors = [{name = "Goncharenko Anton aka alienxs2", email = "your-email@example.com"}]
license = {text = "MIT"}
readme = "README.md"
requires-python = ">=3.10"

dependencies = [
    # MCP Protocol
    "mcp>=0.1.0",                    # Anthropic MCP SDK (official)

    # Database
    "falkordb>=4.0.0",               # Unified vector + graph DB
    "redis>=5.0.0",                  # Semantic cache (optional)

    # LLM & Embeddings
    "httpx>=0.25.0",                 # Async HTTP for Ollama API
    "sentence-transformers>=2.5.0",  # Fallback embeddings

    # NLP & Text Processing
    "langchain>=0.1.0",              # Document chunking, text splitting
    "spacy>=3.7.0",                  # Fast NER for entity extraction
    "rank-bm25>=0.2.2",              # BM25 keyword search

    # Code Parsing (Phase 4)
    "tree-sitter>=0.20.0",           # AST parsing
    "tree-sitter-python>=0.20.0",    # Python grammar
    "tree-sitter-javascript>=0.20.0", # JavaScript grammar

    # Utilities
    "pydantic>=2.0.0",               # Validation & settings
    "pydantic-settings>=2.0.0",      # Settings management from env
    "python-dotenv>=1.0.0",          # Load .env files
    "rapidfuzz>=3.0.0",              # Fuzzy string matching (entity dedup)

    # Async & I/O
    "asyncio>=3.4.3",                # Async operations (stdlib but explicit)
    "aiofiles>=23.0.0",              # Async file operations

    # Logging
    "structlog>=23.0.0",             # Structured logging
]

[project.optional-dependencies]
dev = [
    # Testing
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.1.0",
    "pytest-mock>=3.12.0",

    # Type Checking
    "mypy>=1.7.0",
    "types-redis>=4.6.0",

    # Code Quality
    "black>=23.0.0",                 # Code formatter
    "isort>=5.12.0",                 # Import sorter
    "flake8>=6.1.0",                 # Linter
    "pylint>=3.0.0",                 # Advanced linter

    # Documentation
    "mkdocs>=1.5.0",
    "mkdocs-material>=9.4.0",
]

speed = [
    # Optional performance dependencies
    "orjson>=3.9.0",                 # Faster JSON (C-based)
    "uvloop>=0.19.0",                # Faster asyncio event loop (Unix only)
]

[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"
```

---

#### Code Style & Conventions

**Type Hints**: 100% coverage for public APIs

```python
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel

# Function signatures with type hints
async def add_memory(
    text: str,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Add memory to storage.

    Args:
        text: Memory text content
        metadata: Optional metadata dict

    Returns:
        memory_id: UUID string
    """
    pass

# Pydantic models for validation
class AddMemoryRequest(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None

class SearchMemoryRequest(BaseModel):
    query: str
    limit: int = 10
    filters: Optional[Dict[str, Any]] = None
```

**Async/Await**: For all I/O operations

```python
# Database operations
async def store_embedding(chunk_id: str, embedding: List[float]) -> None:
    await db.execute(...)

# HTTP requests
async def fetch_from_ollama(prompt: str) -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:
        response = await client.post(...)
        return response.json()

# File operations
async def read_document(file_path: str) -> str:
    async with aiofiles.open(file_path, 'r') as f:
        return await f.read()
```

**Error Handling**: Explicit and informative

```python
class ZapomniError(Exception):
    """Base exception for Zapomni errors."""
    pass

class DatabaseError(ZapomniError):
    """Database operation failed."""
    pass

class EmbeddingError(ZapomniError):
    """Embedding generation failed."""
    pass

# Usage
try:
    embedding = await generate_embedding(text)
except httpx.TimeoutException:
    raise EmbeddingError(
        f"Ollama timeout. Is Ollama running? "
        f"Check: http://localhost:11434"
    ) from None
except Exception as e:
    raise EmbeddingError(f"Embedding failed: {e}") from e
```

**Logging**: Structured logging to stderr

```python
import structlog

logger = structlog.get_logger()

# Structured logs with context
logger.info(
    "memory_added",
    memory_id=memory_id,
    chunk_count=len(chunks),
    duration_ms=duration_ms,
    user="system"
)

logger.error(
    "search_failed",
    query=query,
    error=str(e),
    exc_info=True  # Include stack trace
)
```

---

### 4. MCP Protocol - Communication Layer

#### Why MCP Stdio?

**Decision Rationale**:

1. **Simplicity**: Stdio is the easiest transport to implement and debug
   - No network configuration (ports, firewalls)
   - No authentication/authorization complexity
   - No TLS/SSL certificates
   - Just stdin/stdout/stderr

2. **Security**: Process isolation provides natural security boundary
   - Each MCP server runs in separate process
   - OS-level isolation
   - No network exposure
   - Minimal attack surface

3. **Standard Protocol**: JSON-RPC 2.0 over newline-delimited JSON
   - Well-specified protocol
   - Language-agnostic
   - Easy to parse and validate
   - Extensive tooling support

4. **Compatibility**: Works with all major MCP clients out of the box
   - Claude CLI
   - Claude Desktop
   - Cursor IDE
   - Cline extension
   - Any MCP-compatible tool

5. **Debugging**: Transparent message flow
   - All messages visible in stderr logs
   - Easy to replay and test
   - No encrypted traffic to decrypt
   - Simple to trace issues

**Alternatives Considered**:
- **HTTP Transport**: More complex (server, routes, auth), better for multi-user but overkill for local
- **SSE (Server-Sent Events)**: Good for streaming, but adds complexity for MVP
- **WebSockets**: Bidirectional and persistent, but unnecessary overhead for request-response

**Trade-offs**:
- **Pros**:
  - Extremely simple
  - Secure by default
  - Easy debugging
  - No configuration
  - Universal compatibility

- **Cons**:
  - Single user per instance
  - No remote access
  - No streaming responses (initially)
  - Process-based scaling only

- **Future Enhancement**: Can add HTTP/SSE in Phase 5+ for web integrations while keeping stdio as default

---

#### MCP Server Implementation

```python
from mcp.server import Server
from mcp.server.stdio import stdio_server
from pydantic import Field
import asyncio
import json
import logging
import sys
from typing import Dict, Any, List, Optional

# Configure logging to stderr (stdout reserved for MCP)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# Create MCP server instance
server = Server("zapomni-memory")

# Import dependencies (initialized in main())
db = None  # FalkorDBClient
embedder = None  # OllamaClient
searcher = None  # HybridSearchEngine

@server.call_tool()
async def add_memory(
    text: str = Field(description="Text to remember"),
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata (tags, source, date)"
    )
) -> Dict[str, Any]:
    """
    Add new information to memory system.

    This tool ingests text, generates embeddings locally,
    and stores in FalkorDB for later retrieval.
    """
    try:
        logger.info(f"Adding memory: {text[:50]}...")

        # Validate input
        if not text or len(text.strip()) == 0:
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "status": "error",
                        "message": "Text cannot be empty"
                    })
                }],
                "isError": True
            }

        # Chunk text if needed (Phase 1: simple, Phase 2: semantic)
        chunks = chunk_text(text)  # Returns list of text chunks

        # Generate embeddings
        embeddings = []
        for chunk in chunks:
            embedding = await embedder.embed(chunk)
            embeddings.append(embedding)

        # Store in FalkorDB
        memory_id = await db.add_memory(
            text=text,
            embedding=embeddings[0],  # Use first chunk embedding for doc-level
            metadata=metadata or {}
        )

        # Store chunks
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            await db.add_chunk(
                chunk_id=f"{memory_id}_chunk_{i}",
                text=chunk,
                doc_id=memory_id,
                index=i,
                embedding=embedding
            )

        logger.info(f"Memory added: {memory_id} ({len(chunks)} chunks)")

        return {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "status": "success",
                    "memory_id": memory_id,
                    "chunks_created": len(chunks),
                    "text_preview": text[:100]
                }, indent=2)
            }]
        }

    except Exception as e:
        logger.error(f"Error adding memory: {e}", exc_info=True)
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "status": "error",
                    "message": str(e)
                })
            }],
            "isError": True
        }

@server.call_tool()
async def search_memory(
    query: str = Field(description="Search query"),
    limit: int = Field(default=10, description="Maximum results to return"),
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional filters (date_from, date_to, tags, source)"
    )
) -> Dict[str, Any]:
    """
    Search memories using hybrid retrieval (vector + BM25).

    Returns relevant memories ranked by similarity.
    """
    try:
        logger.info(f"Searching: {query}")

        # Generate query embedding
        query_embedding = await embedder.embed(query)

        # Hybrid search (Phase 1: vector only, Phase 2: add BM25)
        results = await searcher.search(
            query_embedding=query_embedding,
            query_text=query,
            limit=limit,
            filters=filters or {}
        )

        logger.info(f"Found {len(results)} results")

        return {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "status": "success",
                    "count": len(results),
                    "results": results
                }, indent=2)
            }]
        }

    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "status": "error",
                    "message": str(e)
                })
            }],
            "isError": True
        }

@server.call_tool()
async def get_stats() -> Dict[str, Any]:
    """Get memory system statistics and health metrics."""
    try:
        stats = await db.get_statistics()

        return {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "status": "success",
                    "statistics": stats
                }, indent=2)
            }]
        }
    except Exception as e:
        logger.error(f"Stats error: {e}", exc_info=True)
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "status": "error",
                    "message": str(e)
                })
            }],
            "isError": True
        }

async def main():
    """Main server entry point."""
    global db, embedder, searcher

    logger.info("Starting Zapomni MCP Server...")

    # Initialize dependencies
    from zapomni_db.client import FalkorDBClient
    from zapomni_core.embeddings import OllamaClient
    from zapomni_core.search import HybridSearchEngine

    db = FalkorDBClient()
    await db.connect()
    logger.info("Connected to FalkorDB")

    embedder = OllamaClient()
    connection_ok = await embedder.test_connection()
    if not connection_ok:
        logger.warning("Ollama connection issues, some features may not work")

    searcher = HybridSearchEngine(db, embedder)

    # Run MCP server with stdio transport
    async with stdio_server() as (read_stream, write_stream):
        logger.info("MCP server running on stdio")
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
```

---

#### MCP Configuration (Claude Desktop)

**Configuration File**: `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS)

```json
{
  "mcpServers": {
    "zapomni": {
      "command": "python",
      "args": ["-m", "zapomni_mcp.server"],
      "env": {
        "FALKORDB_HOST": "localhost",
        "FALKORDB_PORT": "6379",
        "OLLAMA_HOST": "http://localhost:11434",
        "EMBEDDING_MODEL": "nomic-embed-text",
        "LLM_MODEL": "llama3.1:8b",
        "LOG_LEVEL": "INFO"
      }
    }
  }
}
```

**Alternative: Using Script Path**:

```json
{
  "mcpServers": {
    "zapomni": {
      "command": "python",
      "args": ["/path/to/zapomni-mcp/src/server.py"],
      "env": {
        "FALKORDB_HOST": "localhost",
        "OLLAMA_HOST": "http://localhost:11434"
      }
    }
  }
}
```

---

## System Components Detail

### Document Processor

**Purpose**: Extract text from documents and create semantic chunks.

**Supported Formats** (Phase 1-2):
- **PDF**: PyMuPDF (fitz) for text extraction, preserves structure
- **DOCX**: python-docx for Microsoft Word documents
- **Markdown**: Direct UTF-8 parsing, preserve headers and structure
- **Plain Text**: UTF-8 text files
- **HTML** (Phase 2): trafilatura for clean text extraction from web pages

**Future Formats** (Phase 3+):
- **Images**: OCR with tesseract
- **Tables**: PDF table extraction with tabula-py
- **Code**: AST-based parsing with tree-sitter (Phase 4)

**Chunking Implementation**:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any

class DocumentProcessor:
    """Process documents into searchable chunks."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 100
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Hierarchical separators (paragraphs → sentences → words)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,  # Can use tokenizer instead
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def chunk_text(
        self,
        text: str,
        metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk text into semantic segments.

        Args:
            text: Input text to chunk
            metadata: Optional metadata to attach to chunks

        Returns:
            List of chunk dicts with text and metadata
        """
        chunks = self.splitter.split_text(text)

        result = []
        for i, chunk in enumerate(chunks):
            chunk_data = {
                "text": chunk,
                "index": i,
                "metadata": metadata or {}
            }
            result.append(chunk_data)

        return result

    async def process_pdf(self, file_path: str) -> Dict[str, Any]:
        """Extract text from PDF file."""
        import fitz  # PyMuPDF

        doc = fitz.open(file_path)

        text_parts = []
        for page_num, page in enumerate(doc):
            text = page.get_text()
            text_parts.append(text)

        full_text = "\n\n".join(text_parts)

        return {
            "text": full_text,
            "metadata": {
                "source": file_path,
                "type": "pdf",
                "pages": len(doc)
            }
        }

    async def process_markdown(self, file_path: str) -> Dict[str, Any]:
        """Extract text from Markdown file."""
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            text = await f.read()

        return {
            "text": text,
            "metadata": {
                "source": file_path,
                "type": "markdown"
            }
        }
```

---

### Embedding Generator

See **Ollama Client Implementation** in section 2 above for detailed code.

**Key Features**:
- Sequential embedding generation (Ollama API limitation)
- Semantic caching for performance
- Fallback to sentence-transformers if Ollama unavailable
- Error handling with retries

---

### Hybrid Search Engine

**Purpose**: Combine vector, keyword, and graph retrieval for best results.

**Implementation**:

```python
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any
import numpy as np

class HybridSearchEngine:
    """Hybrid search combining BM25, vector, and graph retrieval."""

    def __init__(self, db, embedder):
        self.db = db
        self.embedder = embedder
        self.bm25 = None
        self.corpus = []

    async def index_for_bm25(self, documents: List[str]):
        """Build BM25 index for keyword search."""
        self.corpus = documents
        tokenized = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

    async def search(
        self,
        query_embedding: List[float],
        query_text: str,
        limit: int = 10,
        filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search with vector + BM25 + graph.

        Phase 1: Vector only
        Phase 2: Vector + BM25 with RRF fusion
        Phase 3: Add graph context
        """
        # Vector search
        vector_results = await self.db.vector_search(
            query_embedding=query_embedding,
            limit=50,  # Get more candidates for fusion
            filters=filters
        )

        # Phase 1: Return vector results directly
        # Phase 2+: Add BM25 and fusion here

        return vector_results[:limit]

    async def hybrid_search_v2(
        self,
        query_embedding: List[float],
        query_text: str,
        limit: int = 10,
        filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Phase 2: Hybrid search with RRF fusion.
        """
        # Vector search
        vector_results = await self.db.vector_search(
            query_embedding=query_embedding,
            limit=50,
            filters=filters
        )

        # BM25 search
        tokenized_query = query_text.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Reciprocal Rank Fusion
        k = 60  # RRF constant
        rrf_scores = {}

        # Add vector scores
        for rank, result in enumerate(vector_results, start=1):
            doc_id = result["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)

        # Add BM25 scores
        bm25_ranked = sorted(
            enumerate(bm25_scores),
            key=lambda x: x[1],
            reverse=True
        )[:50]

        for rank, (idx, score) in enumerate(bm25_ranked, start=1):
            doc_id = self.corpus[idx]  # Map to document ID
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # Fetch full results
        results = []
        for doc_id, score in sorted_ids[:limit]:
            # Fetch from vector_results or database
            result = next((r for r in vector_results if r["id"] == doc_id), None)
            if result:
                result["rrf_score"] = score
                results.append(result)

        return results
```

---

## Design Decisions & Rationale

### Decision 1: Unified Database (FalkorDB) vs Separate Systems

**Context**: Need both vector search and knowledge graph capabilities.

**Options Considered**:
1. **FalkorDB** (unified vector + graph in one system)
2. **ChromaDB (vector) + Neo4j (graph)** (separate best-of-breed)
3. **Qdrant (vector) + NetworkX (in-memory graph)**

**Decision**: **FalkorDB** (unified)

**Rationale**:
- **Eliminates Synchronization Complexity**: No need to keep two databases in sync
  - Single source of truth
  - Consistent transactions (no dual-phase commits)
  - Simpler backup and restore

- **Superior Performance**: 496x faster P99 latency (FalkorDB benchmark)
  - GraphBLAS backend for graph operations
  - Redis protocol for vector operations
  - Optimized for GraphRAG workloads

- **Better Memory Efficiency**: 6x better than separate systems
  - No data duplication
  - Shared metadata storage
  - Unified index structures

- **Simpler Operations**:
  - One connection pool
  - One monitoring system
  - One backup strategy
  - Easier troubleshooting

**Trade-offs Accepted**:
- **Risk**: FalkorDB is newer (launched 2023), smaller community than Neo4j
- **Mitigation**: Database abstraction layer allows fallback to ChromaDB + Neo4j if needed
- **Assumption**: Performance and simplicity outweigh maturity concerns for MVP

**Consequences**:
- Simpler codebase (single DB client vs dual clients + sync logic)
- Faster development (no sync logic to implement and test)
- Better performance for hybrid queries
- Lower operational overhead
- Dependency on FalkorDB's roadmap and stability

---

### Decision 2: Local-First Architecture (Ollama) vs Cloud APIs

**Context**: Need LLM for embeddings and entity extraction.

**Options Considered**:
1. **Local-only (Ollama)** - all processing on user's machine
2. **Hybrid (local + optional cloud)** - default local, allow cloud for better quality
3. **Cloud-first (OpenAI API)** - best quality, requires API keys

**Decision**: **Local-only (Ollama)**

**Rationale**:
- **Complete Privacy Guarantee**: Data never leaves user's machine
  - Legal documents (attorney-client privilege)
  - Healthcare data (HIPAA compliance)
  - Corporate IP (trade secrets, NDAs)
  - Personal journals and research

- **Zero API Costs**: No recurring fees
  - $0 vs $100-500/month for cloud services
  - Unlimited usage without throttling
  - No rate limits or quotas

- **Offline Capability**: Works without internet
  - Air-gapped environments
  - Low/no connectivity situations
  - No dependency on external services

- **Product Differentiation**: Unique position vs Cognee (cloud-dependent)
  - Clear value proposition
  - No compromise on privacy
  - Aligns with open-source ethos

**Trade-offs Accepted**:
- **Quality**: Local models (Llama 3.1, DeepSeek-R1) slightly lower quality than GPT-4
  - Mitigation: Hybrid approach (SpaCy + LLM) and confidence filtering
  - Reality: DeepSeek-R1 approaches GPT-4 performance for reasoning

- **Speed**: Local inference slower than cloud APIs
  - Mitigation: Semantic caching (60%+ hit rate)
  - Reality: For single-user, speed acceptable (5-15s per document)

- **Hardware Requirements**: Requires user to have sufficient compute (8GB+ RAM)
  - Mitigation: Clear hardware requirements in docs
  - Fallback: Lighter models (all-MiniLM-L6-v2) for lower-end hardware

**Consequences**:
- Target market: Privacy-conscious users, cost-sensitive developers, researchers
- Setup complexity: Users must install Ollama and pull models
- Performance variability: Depends on user's hardware (CPU vs GPU)
- **Strong differentiator**: Only local-first MCP memory with knowledge graphs

---

### Decision 3: Hybrid Search (BM25 + Vector) vs Vector-Only

**Context**: How to retrieve relevant documents for queries.

**Options Considered**:
1. **Vector-only** (semantic similarity via embeddings)
2. **BM25-only** (keyword-based lexical matching)
3. **Hybrid (BM25 + Vector with fusion)**

**Decision**: **Hybrid (BM25 + Vector)**

**Rationale**:
- **3.4x Better Accuracy**: Research benchmark (Diffbot) shows hybrid >> vector-only
  - BM25 catches exact terminology matches
  - Vector catches semantic similarity
  - Fusion combines strengths of both

- **Complementary Strengths**:
  - **BM25**: Excels at exact terms, acronyms, names, technical jargon
  - **Vector**: Excels at paraphrases, concepts, semantic similarity
  - Together: Cover all query types

- **Production Best Practice**: Modern RAG systems use hybrid (2024 standard)
  - Weaviate, Qdrant, Pinecone all recommend hybrid
  - Research papers consistently show improvement

- **Minimal Overhead**: BM25 is fast (< 50ms for 10K docs)
  - rank-bm25 library is pure Python, simple
  - Fusion (RRF) is O(n) complexity, negligible cost

**Trade-offs Accepted**:
- **Complexity**: More code than vector-only (BM25 index + fusion logic)
  - Mitigation: Clean abstraction, well-tested libraries

- **Memory**: BM25 index requires additional RAM
  - Mitigation: Tokenized corpus is small (~10MB for 10K docs)

**Implementation Details**:
- **Phase 1**: Vector-only (ship fast)
- **Phase 2**: Add BM25 + RRF fusion (enhance quality)
- **Fusion Method**: Reciprocal Rank Fusion (RRF) with k=60

**Consequences**:
- Better retrieval quality (measurable improvement)
- Handles diverse query types (keywords + semantic)
- Industry-standard approach (proven best practice)

---

### Decision 4: Semantic Chunking vs Fixed-Size Chunking

**Context**: How to split documents into searchable units.

**Options Considered**:
1. **Fixed-size** (e.g., 512 tokens, no overlap)
2. **Fixed-size with overlap** (e.g., 512 tokens, 100 token overlap)
3. **Semantic chunking** (split on meaningful boundaries: paragraphs, sentences)

**Decision**: **Semantic chunking with overlap** (RecursiveCharacterTextSplitter)

**Rationale**:
- **Preserves Meaning**: Respects document structure
  - Doesn't break mid-sentence
  - Keeps paragraphs together when possible
  - Maintains context

- **Research-Backed**: 256-512 tokens with 10-20% overlap is best practice (2024 studies)
  - Unstructured.io benchmark
  - LangChain documentation
  - Weaviate blog research

- **Hierarchical Separators**: Tries paragraphs first, then sentences, then words
  - Maximizes semantic coherence
  - Falls back gracefully for dense text

- **Overlap Prevents Information Loss**: Edge information not lost between chunks
  - 10-20% overlap ensures context continuity
  - Slightly more storage, much better retrieval

**Trade-offs Accepted**:
- **Storage**: ~10-20% more chunks due to overlap
  - Mitigation: Storage is cheap (embeddings are main cost)

- **Processing Time**: Slightly slower than naive split
  - Mitigation: Still very fast (< 1s for typical document)

**Implementation**:
```python
RecursiveCharacterTextSplitter(
    chunk_size=512,           # Target size
    chunk_overlap=100,        # 20% overlap
    separators=["\n\n", "\n", ". ", " ", ""]  # Hierarchical
)
```

**Consequences**:
- Better retrieval quality (chunks are meaningful units)
- Fewer broken contexts
- Industry-standard approach

---

### Decision 5: Background Task Processing for Knowledge Graph

**Context**: Knowledge graph building takes 5-15 minutes for 1K documents (entity extraction, relationship detection).

**Options Considered**:
1. **Synchronous** (block until complete)
2. **Asynchronous with callbacks** (return immediately, callback when done)
3. **Background task queue** (return task ID, queryable status)

**Decision**: **Background task queue** (Cognee pattern)

**Rationale**:
- **Better UX**: Immediate response to user
  - No 15-minute wait
  - User can continue working
  - Progress queryable anytime

- **MCP Compatibility**: MCP tools should respond quickly
  - Timeout constraints
  - User expects fast responses

- **Proven Pattern**: Cognee uses this successfully
  - Task ID for tracking
  - Status endpoint (pending/running/completed/failed)
  - Progress percentage

- **Flexibility**: Can run multiple tasks concurrently (future)

**Trade-offs Accepted**:
- **Complexity**: More code than synchronous (task manager, status tracking)
  - Mitigation: Reuse Cognee pattern, well-understood

- **State Management**: Must track task state persistently
  - Mitigation: In-memory for MVP (Phase 1-2), consider persistence later

**Implementation**:
```python
# build_graph() returns immediately
{
    "status": "accepted",
    "task_id": "uuid-here",
    "message": "Graph building started in background"
}

# graph_status(task_id) checks progress
{
    "status": "running",
    "progress": 67.5,  # percentage
    "started_at": "2025-11-22T10:00:00Z"
}
```

**Consequences**:
- Responsive UX (tool calls return immediately)
- Can handle long-running operations gracefully
- Adds ~200 lines of task manager code

---

## Performance Architecture

### Latency Targets

**Query Performance (P95)**:
- **Target**: < 500ms end-to-end for search_memory

**Breakdown** (Phase 2 with hybrid search + reranking):
- Embedding generation: 50ms (cached) / 150ms (uncached)
- Vector search (FalkorDB): 100ms
- BM25 search: 50ms
- Fusion (RRF): 50ms
- Cross-encoder reranking: 150ms
- Overhead (network, parsing, logging): 50ms
- **Total**: ~450ms (cached) / ~550ms (uncached)

**Optimization Strategies**:
- **Semantic cache**: Reduce embedding time from 150ms → 50ms for 60%+ of queries
- **HNSW indexing**: Fast approximate search (vs brute-force)
- **Parallel retrieval**: Run vector + BM25 concurrently
- **Top-K limiting**: Only rerank top 20 candidates, not all

---

**Ingestion Performance**:
- **Target**: > 100 documents/minute

**Breakdown** (for typical document: 1000 words, 3 chunks):
- PDF extraction: 500ms
- Chunking: 10ms
- Embedding (3 chunks × 100ms): 300ms
- FalkorDB storage: 50ms
- **Total**: ~860ms per document = ~70 docs/minute

**With Batching** (Phase 2):
- Process 10 documents in parallel
- Bottleneck: Embedding (sequential via Ollama)
- With semantic cache (60% hit rate): Effective ~40ms per embedding
- Achievable: 100-150 documents/minute

---

### Memory Management

**Target**: < 4GB RAM for 10K documents

**Breakdown**:
- FalkorDB: 1-2GB (10K chunks × 768 dims × 4 bytes + graph overhead)
- Ollama (model loaded): 1GB (nomic-embed-text)
- Python process: 500MB (server + libraries)
- Redis cache: 500MB (semantic cache)
- **Total**: ~3.5GB (within target)

**For 100K documents**:
- FalkorDB: 10-15GB
- Ollama: 1GB
- Python: 500MB
- Redis: 2GB
- **Total**: ~14-18GB (requires 16GB+ RAM system)

**Optimization Strategies**:
- **Streaming processing**: Don't load all documents into RAM at once
- **Generator patterns**: Yield chunks instead of returning giant lists
- **LRU cache eviction**: Limit cache size, evict old entries
- **Lazy loading**: Load embeddings/chunks on-demand

---

### Scalability Architecture

**Horizontal Scaling**: Not applicable for MVP (local single-user design)

**Vertical Scaling** (document capacity on single machine):

| Document Count | RAM Required | Storage | Query Latency (P95) | Notes |
|----------------|--------------|---------|---------------------|-------|
| 1K | 2GB | 100MB | < 300ms | MVP target |
| 10K | 4GB | 1GB | < 500ms | Phase 1-2 target |
| 100K | 16GB | 10GB | < 1s | Phase 3+ (with optimization) |
| 1M | 64GB+ | 100GB | < 3s | Requires partitioning, HNSW tuning |

**Scaling Strategies** (for large corpora):
- **Partitioning**: Divide by date/topic, search relevant partitions only
- **HNSW tuning**: Adjust M, efConstruction, efSearch for size/speed trade-off
- **Approximate search**: Accept slightly lower recall for much faster queries
- **Tiered storage**: Hot data (recent) in RAM, cold data on disk

---

## Security & Privacy Architecture

### Local-Only Guarantees

**Data Flow**: Everything stays local
```
User Machine:
  User → Claude CLI → Zapomni MCP Server → FalkorDB (localhost:6379) → Ollama (localhost:11434)

NEVER:
  User → External API
  User → Cloud Storage
  User → Telemetry Service
```

**Network Isolation**:
- **MCP Server**: Stdio only, no network sockets
- **FalkorDB**: Bind to 127.0.0.1 (localhost) only, not 0.0.0.0
- **Ollama**: Bind to 127.0.0.1 only
- **No Telemetry**: Zero external calls, no crash reporting, no analytics

**Configuration Enforcement**:
```python
# FalkorDB: Localhost only
FALKORDB_HOST = "localhost"  # NOT "0.0.0.0" or public IP

# Ollama: Localhost only
OLLAMA_HOST = "http://localhost:11434"  # NOT http://0.0.0.0

# No telemetry
TELEMETRY_ENABLED = False
SENTRY_DSN = None
ANALYTICS_ENABLED = False
```

---

### Input Validation

**Pydantic Schemas**: All inputs validated

```python
from pydantic import BaseModel, Field, validator

class AddMemoryRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=100000)
    metadata: Optional[Dict[str, Any]] = Field(default=None)

    @validator('text')
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty or whitespace-only')
        return v

    @validator('metadata')
    def metadata_size_limit(cls, v):
        if v and len(json.dumps(v)) > 10000:
            raise ValueError('Metadata too large (max 10KB)')
        return v

class SearchMemoryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    limit: int = Field(default=10, ge=1, le=100)
    filters: Optional[Dict[str, Any]] = None
```

**Injection Prevention**:
- **Cypher Queries**: Parameterized only, never string concatenation
  ```python
  # SAFE
  query = "MATCH (m:Memory {id: $id}) RETURN m"
  result = graph.query(query, {"id": memory_id})

  # UNSAFE (never do this)
  query = f"MATCH (m:Memory {{id: '{memory_id}'}}) RETURN m"
  ```

- **SQL Injection**: N/A (no SQL, only Cypher via FalkorDB client which handles escaping)
- **Path Traversal**: Whitelist approach for file access (Phase 4 code indexing)
  ```python
  import os

  def safe_path(user_path: str, allowed_dir: str) -> str:
      # Resolve to absolute path
      abs_path = os.path.abspath(user_path)
      abs_allowed = os.path.abspath(allowed_dir)

      # Check if within allowed directory
      if not abs_path.startswith(abs_allowed):
          raise ValueError("Path traversal not allowed")

      return abs_path
  ```

---

### Data Encryption

**At Rest**: User responsibility (OS-level encryption)
- **Linux**: LUKS, dm-crypt, eCryptfs
- **macOS**: FileVault
- **Windows**: BitLocker

**Recommendation**: Enable full-disk encryption on dev machine

**In Transit**: N/A (local-only, no network transmission)

**In Memory**: Cleartext (performance vs security trade-off for local single-user)

---

## Testing Strategy

### Test Pyramid

```
          E2E (5%)
         Integration (25%)
        Unit Tests (70%)
```

**Unit Tests** (70%):
- Pure functions, business logic
- Fast (< 1ms per test)
- No external dependencies (mock DB, Ollama)
- Coverage target: 90%+

**Integration Tests** (25%):
- FalkorDB integration (real database in Docker)
- Ollama integration (real Ollama instance)
- MCP protocol (stdio communication)
- Slower (< 100ms per test)

**E2E Tests** (5%):
- Full workflow (add → search → verify)
- Claude CLI integration (manual + automated)
- Slowest (< 5s per test)

---

### Test Organization

```
tests/
├── unit/
│   ├── test_chunker.py           # Document chunking logic
│   ├── test_embedder.py          # Embedding generation (mocked)
│   ├── test_search.py            # Search algorithms
│   ├── test_entity_extractor.py  # Entity extraction (mocked)
│   └── test_utils.py             # Utility functions
│
├── integration/
│   ├── test_falkordb_client.py   # Real FalkorDB operations
│   ├── test_ollama_client.py     # Real Ollama API calls
│   ├── test_mcp_server.py        # MCP protocol communication
│   └── test_hybrid_search.py     # End-to-end search pipeline
│
├── e2e/
│   ├── test_full_workflow.py     # Add → Search → Graph
│   └── test_claude_integration.py # Claude CLI integration
│
├── fixtures/
│   ├── sample_documents.py       # Test documents
│   ├── sample_queries.py         # Test queries
│   └── mock_embeddings.py        # Mock embedding vectors
│
└── conftest.py                   # Pytest configuration
```

---

### Performance Testing

**Load Testing**:
```python
import pytest
import time

@pytest.mark.benchmark
def test_search_performance(benchmark):
    """Search should complete in < 500ms."""
    result = benchmark(search_memory, query="test query", limit=10)
    assert benchmark.stats.stats.mean < 0.5  # seconds

@pytest.mark.load
async def test_concurrent_searches():
    """Handle 10 concurrent searches."""
    queries = ["query {i}" for i in range(10)]

    start = time.time()
    results = await asyncio.gather(*[
        search_memory(q) for q in queries
    ])
    duration = time.time() - start

    assert all(len(r) > 0 for r in results)
    assert duration < 5.0  # All 10 searches in < 5s
```

**Memory Profiling**:
```python
import tracemalloc
import pytest

@pytest.mark.memory
def test_ingestion_memory():
    """Memory usage should stay under 4GB for 10K docs."""
    tracemalloc.start()

    # Ingest 10K test documents
    for i in range(10000):
        add_memory(f"Test document {i}")

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert peak < 4 * 1024**3  # 4GB in bytes
```

---

## Deployment Architecture

### Development Environment

**Docker Compose** (recommended for dev):

```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  falkordb:
    image: falkordb/falkordb:latest
    ports:
      - "6379:6379"
    volumes:
      - falkordb_data:/data
    environment:
      - FALKORDB_ARGS=--save 60 1 --appendonly yes

  redis:
    image: redis:7-alpine
    ports:
      - "6380:6379"  # Different port to avoid conflict
    volumes:
      - redis_cache_data:/data
    command: redis-server --appendonly yes

volumes:
  falkordb_data:
  redis_cache_data:
```

**Start services**:
```bash
docker-compose -f docker-compose.dev.yml up -d
```

**Ollama** (run natively, not in Docker for best performance):
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
ollama serve

# Pull models
ollama pull nomic-embed-text
ollama pull llama3.1:8b
```

---

### Production Deployment (Local Install)

**Prerequisites**:
1. Python 3.10+ installed
2. Docker Desktop installed (for FalkorDB + Redis)
3. Ollama installed (native, not Docker)

**Step 1: Install Ollama**
```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: Download from https://ollama.com/download

# Pull required models
ollama pull nomic-embed-text
ollama pull llama3.1:8b
```

**Step 2: Start FalkorDB + Redis**
```bash
# Create docker-compose.yml
cat > docker-compose.yml <<EOF
version: '3.8'
services:
  falkordb:
    image: falkordb/falkordb:latest
    ports:
      - "6379:6379"
    volumes:
      - ./data/falkordb:/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6380:6379"
    volumes:
      - ./data/redis:/data
    restart: unless-stopped
EOF

# Start services
docker-compose up -d
```

**Step 3: Install Zapomni**
```bash
# Install from PyPI (when published)
pip install zapomni-mcp

# Or install from source
git clone https://github.com/yourusername/zapomni.git
cd zapomni
pip install -e .
```

**Step 4: Configure Claude Desktop**
```bash
# macOS
nano ~/Library/Application\ Support/Claude/claude_desktop_config.json

# Linux
nano ~/.config/Claude/claude_desktop_config.json

# Add configuration:
{
  "mcpServers": {
    "zapomni": {
      "command": "python",
      "args": ["-m", "zapomni_mcp.server"],
      "env": {
        "FALKORDB_HOST": "localhost",
        "FALKORDB_PORT": "6379",
        "OLLAMA_HOST": "http://localhost:11434"
      }
    }
  }
}
```

**Step 5: Restart Claude Desktop**

---

## Monitoring & Observability

### Logging

**Structured Logs** (JSON format via structlog):

```python
import structlog
import sys

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Usage
logger.info(
    "memory_added",
    memory_id=memory_id,
    chunk_count=len(chunks),
    duration_ms=duration_ms,
    user_id="system"
)

logger.error(
    "search_failed",
    query=query,
    error_type=type(e).__name__,
    error_message=str(e),
    exc_info=True
)
```

**Log Levels**:
- **DEBUG**: Detailed debugging (off in production)
- **INFO**: Normal operations (default)
- **WARN**: Degraded performance, fallbacks, recoverable errors
- **ERROR**: Failures requiring attention (with stack traces)

**Log Destination**: stderr (stdout reserved for MCP protocol)

---

### Metrics (Optional, Phase 5+)

**Prometheus-Compatible Metrics**:

```python
from prometheus_client import Counter, Histogram, Gauge

# Counters
memory_adds_total = Counter('zapomni_memory_adds_total', 'Total memory additions')
search_queries_total = Counter('zapomni_search_queries_total', 'Total search queries')
errors_total = Counter('zapomni_errors_total', 'Total errors', ['error_type'])

# Histograms
search_latency_seconds = Histogram(
    'zapomni_search_latency_seconds',
    'Search query latency',
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

embedding_latency_seconds = Histogram(
    'zapomni_embedding_latency_seconds',
    'Embedding generation latency'
)

# Gauges
total_memories = Gauge('zapomni_total_memories', 'Total memories stored')
cache_hit_rate = Gauge('zapomni_cache_hit_rate', 'Semantic cache hit rate')
```

---

### Tracing (Future)

**OpenTelemetry** (for distributed tracing):

```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure tracer
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831,
)

trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)

# Usage
with tracer.start_as_current_span("search_memory"):
    with tracer.start_as_current_span("generate_embedding"):
        embedding = await embedder.embed(query)

    with tracer.start_as_current_span("vector_search"):
        results = await db.vector_search(embedding)

    return results
```

---

## Migration & Versioning

### Database Migrations

**Schema Versioning**:

```python
SCHEMA_VERSION = "1.0.0"

async def get_schema_version(graph) -> str:
    """Get current schema version from database."""
    query = "MATCH (v:SchemaVersion) RETURN v.version as version"
    result = graph.query(query)
    if result.result_set:
        return result.result_set[0][0]
    return "0.0.0"

async def migrate_to_v1_1(graph):
    """Migration: Add language field to Chunk nodes."""
    # Add new property
    query = """
    MATCH (c:Chunk)
    WHERE NOT EXISTS(c.language)
    SET c.language = 'unknown'
    """
    graph.query(query)

    # Create new index
    graph.query("CREATE INDEX FOR (c:Chunk) ON (c.language)")

    # Update schema version
    graph.query("""
        MERGE (v:SchemaVersion)
        SET v.version = '1.1.0'
    """)

async def run_migrations(graph):
    """Run pending migrations."""
    current_version = await get_schema_version(graph)

    if current_version < "1.1.0":
        logger.info("Running migration to v1.1.0")
        await migrate_to_v1_1(graph)
```

**Backwards Compatibility Strategy**:
- **Additive changes only** (for MVP/Phase 1-3)
- **Breaking changes**: Require major version bump and migration tool
- **Data preservation**: Always preserve existing data during migrations

---

### API Versioning

**MCP Tool Versioning**:

```python
TOOL_VERSION = "0.1.0"

# Tool definition includes version
@server.tool()
async def add_memory_v1(text: str, metadata: dict = None) -> dict:
    """
    Add memory (v1).

    Version: 0.1.0
    Breaking changes: N/A
    """
    pass

# For breaking changes, create new versioned tool
@server.tool()
async def add_memory_v2(text: str, metadata: dict = None, tags: list = None) -> dict:
    """
    Add memory (v2) with new tags parameter.

    Version: 0.2.0
    Breaking changes: Added required 'tags' parameter
    """
    pass
```

**Semantic Versioning**:
- **Major** (1.0.0 → 2.0.0): Breaking changes
- **Minor** (0.1.0 → 0.2.0): New features, backwards compatible
- **Patch** (0.1.0 → 0.1.1): Bug fixes

---

## Disaster Recovery

### Backup Strategy

**FalkorDB Backup**:

```bash
# RDB Snapshots (point-in-time backups)
# Configured in docker-compose.yml:
FALKORDB_ARGS=--save 900 1 --save 300 10 --save 60 10000

# Translation:
# - Save if 1+ keys changed after 900s (15 min)
# - Save if 10+ keys changed after 300s (5 min)
# - Save if 10000+ keys changed after 60s (1 min)

# AOF (Append-Only File) for durability
FALKORDB_ARGS=--appendonly yes --appendfsync everysec

# Manual backup
docker exec zapomni_falkordb redis-cli --rdb /data/backup-$(date +%Y%m%d-%H%M%S).rdb
```

**Backup Location**:
```
~/.zapomni/backups/
├── falkordb-20251122-1430.rdb
├── falkordb-20251122-1500.rdb
└── falkordb-20251122-1530.rdb
```

**Automated Backup Script**:

```bash
#!/bin/bash
# backup-zapomni.sh

BACKUP_DIR="$HOME/.zapomni/backups"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

mkdir -p "$BACKUP_DIR"

# Backup FalkorDB
docker exec zapomni_falkordb redis-cli BGSAVE
sleep 5
docker cp zapomni_falkordb:/data/dump.rdb "$BACKUP_DIR/falkordb-$TIMESTAMP.rdb"

# Keep only last 7 days of backups
find "$BACKUP_DIR" -name "falkordb-*.rdb" -mtime +7 -delete

echo "Backup completed: falkordb-$TIMESTAMP.rdb"
```

**Cron job** (daily backups at 2 AM):
```cron
0 2 * * * /home/user/zapomni/backup-zapomni.sh
```

---

### Recovery Procedure

**Restore from Backup**:

```bash
#!/bin/bash
# restore-zapomni.sh

BACKUP_FILE=$1

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: ./restore-zapomni.sh <backup-file.rdb>"
    exit 1
fi

# 1. Stop Zapomni and services
docker-compose down

# 2. Restore RDB file
cp "$BACKUP_FILE" ./data/falkordb/dump.rdb

# 3. Restart services
docker-compose up -d

# 4. Verify data integrity
sleep 5
docker exec zapomni_falkordb redis-cli PING

# 5. Check database stats
docker exec zapomni_falkordb redis-cli DBSIZE

echo "Restore completed. Database is ready."
```

---

## Future Technical Roadmap

### Phase 5: Multi-Transport Support (Month 3)

**HTTP Transport**:
- FastAPI server for HTTP endpoints
- RESTful API compatible with MCP
- Authentication (API keys)
- Rate limiting
- CORS configuration

**SSE Transport**:
- Server-Sent Events for streaming responses
- Real-time updates
- Progress streaming for long operations

**WebSocket** (Optional):
- Bidirectional communication
- Real-time notifications
- Graph updates

---

### Phase 6: Performance Optimization (Month 4-5)

**GPU Acceleration**:
- CUDA support for embeddings
- GPU-based vector search (FAISS-GPU)
- Batch processing optimization

**Model Optimization**:
- Quantized models (4-bit, 8-bit)
- Model distillation for faster inference
- Custom fine-tuned models

**Distributed Processing**:
- Multi-process embedding generation
- Parallel graph building
- Load balancing

---

### Phase 7: Advanced Features (Month 6+)

**Multi-Modal**:
- Image embeddings (CLIP)
- Audio transcription and search
- Video frame analysis
- Cross-modal search

**Federated Graphs**:
- Merge multiple user graphs
- Shared knowledge bases
- Privacy-preserving aggregation

**Plugin System**:
- Custom document processors
- Custom embedders
- Custom entity extractors
- Community plugin marketplace

---

**Document Status**: Draft v1.0
**Created**: 2025-11-22
**Authors**: Goncharenko Anton aka alienxs2 + Claude Code
**Copyright**: Copyright (c) 2025 Goncharenko Anton aka alienxs2
**License**: MIT License
**Last Updated**: 2025-11-22
**Next Steps**: Review, approval via steering workflow, proceed to structure.md

---

**Document Length**: ~1250 lines
**Estimated Reading Time**: 60-75 minutes
**Target Audience**: Technical contributors, system architects, senior engineers
