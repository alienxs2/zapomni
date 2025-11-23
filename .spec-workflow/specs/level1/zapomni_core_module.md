# Zapomni Core Module - Module Specification

**Level:** 1 (Module)
**Author:** Goncharenko Anton aka alienxs2
**Status:** Draft
**Version:** 1.0
**Created:** 2025-11-23

---

## Overview

### Purpose

The Zapomni Core module serves as the **business logic and processing engine** for the entire memory system. It provides the essential intelligence that transforms raw text and code into searchable, semantically-rich memory chunks with embedded knowledge graphs.

This module is the "brain" of Zapomni - it handles all document processing, embedding generation, entity extraction, search algorithms, and knowledge graph construction. It is designed to be **framework-agnostic** and reusable, working independently of the MCP protocol layer.

### Scope

**Included:**
- **Document Processing:** Text and code chunking with semantic boundary detection
- **Embedding Generation:** Local embedding creation via Ollama integration
- **Entity Extraction:** NER (Named Entity Recognition) and relationship detection using hybrid SpaCy + LLM approach
- **Search Engines:** Vector similarity, BM25 keyword search, hybrid fusion, and graph traversal
- **Memory Processing Pipeline:** Orchestration of the complete add → chunk → embed → store workflow
- **Semantic Caching:** Embedding cache for performance optimization
- **Background Task Management:** Async task queue for knowledge graph construction

**Not Included:**
- MCP protocol handling (in `zapomni_mcp`)
- Database client implementations (in `zapomni_db`)
- HTTP/network transports
- User authentication/authorization
- Multi-user or collaboration features

### Position in Architecture

Zapomni Core sits in the **middle layer** of the architecture:

```
┌──────────────────┐
│   zapomni_mcp    │  ← Protocol adapter (MCP tools)
└────────┬─────────┘
         │ calls
         ↓
┌──────────────────┐
│  zapomni_core    │  ← THIS MODULE (business logic)
│  (Processing)    │
└────────┬─────────┘
         │ uses
         ↓
┌──────────────────┐
│   zapomni_db     │  ← Storage layer (FalkorDB, Redis)
└──────────────────┘
```

**Key Relationships:**
- **zapomni_mcp → zapomni_core:** MCP tools delegate all processing to core services
- **zapomni_core → zapomni_db:** Core uses DB clients for persistence and retrieval
- **zapomni_core → Ollama:** Direct integration for embeddings and LLM calls (via httpx)

---

## Architecture

### High-Level Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      Zapomni Core Module                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │             Memory Processing Pipeline              │    │
│  │  Input → Validate → Chunk → Embed → Extract → Store│    │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Processors  │  │  Embeddings  │  │  Extractors  │      │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤      │
│  │ TextProc     │  │ OllamaEmbed  │  │ EntityExtract│      │
│  │ PDFProc      │  │ STEmbed      │  │ RelExtract   │      │
│  │ CodeProc     │  │ SemanticCache│  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Chunking   │  │    Search    │  │    Tasks     │      │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤      │
│  │ SemanticChunk│  │ VectorSearch │  │ TaskManager  │      │
│  │ ASTChunk     │  │ BM25Search   │  │ StatusTracker│      │
│  │              │  │ HybridSearch │  │              │      │
│  │              │  │ GraphSearch  │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                  Utilities                          │    │
│  │  TextUtils | Validation | ErrorHandling            │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
         ↓                          ↓                     ↓
   FalkorDBClient            OllamaAPI              RedisCache
```

### Key Responsibilities

1. **Document Processing**
   - Accept text, PDF, code files
   - Perform semantic chunking (text) or AST-based chunking (code)
   - Preserve structural integrity (functions, paragraphs, sections)
   - Handle edge cases (empty docs, binary data, encoding issues)

2. **Embedding Generation**
   - Generate embeddings locally via Ollama (nomic-embed-text)
   - Fallback to sentence-transformers if Ollama unavailable
   - Implement semantic caching (60%+ hit rate target)
   - Batch processing for efficiency

3. **Entity & Relationship Extraction**
   - Extract entities using hybrid SpaCy + LLM approach
   - Detect relationships between entities
   - Build knowledge graph structures
   - Achieve 80%+ precision for entities, 70%+ for relationships

4. **Search Intelligence**
   - Vector similarity search (cosine distance, HNSW index)
   - BM25 keyword search (rank-bm25 library)
   - Hybrid search with RRF (Reciprocal Rank Fusion)
   - Cross-encoder reranking for top-K refinement
   - Graph traversal queries for relationship discovery

5. **Memory Processing Pipeline**
   - Orchestrate end-to-end workflow: input → chunks → embeddings → graph → storage
   - Validate inputs at each stage
   - Handle errors gracefully with informative messages
   - Provide progress tracking for long-running operations

6. **Performance Optimization**
   - Semantic caching for embeddings
   - Batch processing for multiple documents
   - Async I/O for network calls (Ollama, DB)
   - Efficient data structures (minimize memory footprint)

---

## Public API

### Main Entry Point: MemoryProcessor

```python
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

class MemoryProcessor:
    """
    Main entry point for memory processing operations.

    Orchestrates the complete pipeline: chunking → embedding → extraction → storage.
    All MCP tools delegate to this processor.

    Attributes:
        chunker: Text/code chunking service
        embedder: Embedding generation service
        extractor: Entity extraction service (optional, Phase 2)
        db: Database client for persistence
        cache: Semantic cache (optional, Phase 2)
        task_manager: Background task manager (optional, Phase 2)

    Example:
        >>> from zapomni_core import MemoryProcessor
        >>> from zapomni_db.falkordb import FalkorDBClient
        >>>
        >>> processor = MemoryProcessor(
        ...     db=FalkorDBClient(host="localhost"),
        ...     ollama_host="http://localhost:11434"
        ... )
        >>>
        >>> memory_id = await processor.add_memory(
        ...     text="Python is a programming language",
        ...     metadata={"source": "user"}
        ... )
        >>> print(f"Stored with ID: {memory_id}")
    """

    def __init__(
        self,
        db: FalkorDBClient,
        ollama_host: str = "http://localhost:11434",
        embedding_model: str = "nomic-embed-text",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        enable_cache: bool = True,
        enable_extraction: bool = False  # Phase 2
    ) -> None:
        """
        Initialize memory processor with dependencies.

        Args:
            db: FalkorDB client for storage
            ollama_host: Ollama API URL
            embedding_model: Ollama embedding model name
            chunk_size: Text chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            enable_cache: Enable semantic caching (Phase 2)
            enable_extraction: Enable entity extraction (Phase 2)
        """

    async def add_memory(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add memory to system with full processing pipeline.

        Workflow:
        1. Validate input (non-empty, max 10MB)
        2. Chunk text (semantic boundaries, 256-512 tokens)
        3. Generate embeddings (via Ollama)
        4. Store in FalkorDB (with vector index)
        5. Return memory ID

        Args:
            text: Text content to remember (max 10MB)
            metadata: Optional metadata (tags, source, date, custom fields)

        Returns:
            memory_id: UUID string identifying stored memory

        Raises:
            ValidationError: If text empty or exceeds 10MB
            EmbeddingError: If embedding generation fails
            DatabaseError: If storage fails

        Performance Target:
            - Normal input (< 1KB): < 100ms
            - Large input (< 100KB): < 500ms
            - Maximum allowed: < 1000ms
        """

    async def search_memory(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        search_mode: str = "vector"  # "vector", "bm25", "hybrid" (Phase 2)
    ) -> List[SearchResult]:
        """
        Search memories using specified search mode.

        Args:
            query: Natural language search query
            limit: Maximum number of results (1-100)
            filters: Optional metadata filters (tags, date_from, date_to, source)
            search_mode: "vector" (Phase 1), "bm25", "hybrid" (Phase 2)

        Returns:
            List of SearchResult objects with:
            - memory_id: UUID of matching memory
            - text: Matching chunk text
            - similarity_score: 0-1 relevance score
            - metadata: Original metadata (tags, source, timestamp)

        Raises:
            ValidationError: If query empty or limit out of range
            EmbeddingError: If query embedding fails
            SearchError: If search operation fails

        Performance Target:
            - P50 latency: < 200ms
            - P95 latency: < 500ms
            - P99 latency: < 1000ms
        """

    async def build_knowledge_graph(
        self,
        memory_ids: Optional[List[str]] = None,
        mode: str = "full"  # "entities_only", "relationships_only", "full"
    ) -> str:
        """
        Build knowledge graph from memories (async background task).

        Phase 2 feature. Extracts entities and relationships, constructs graph.

        Args:
            memory_ids: Specific memories to process (None = all unprocessed)
            mode: What to extract (entities, relationships, or both)

        Returns:
            task_id: UUID for tracking background task progress

        Raises:
            TaskError: If task queue is full

        Performance Target:
            - 1K documents in < 10 minutes
            - Entity extraction: 80%+ precision, 75%+ recall
            - Relationship detection: 70%+ precision, 65%+ recall
        """

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get memory system statistics.

        Returns:
            Dictionary with:
            - total_memories: int
            - total_chunks: int
            - database_size_mb: float
            - cache_hit_rate: float (0-1, if caching enabled)
            - avg_query_latency_ms: int
            - total_entities: int (if graph built)
            - total_relationships: int (if graph built)

        Performance Target:
            - Execution time: < 100ms
        """
```

### Data Models

**NOTE**: The `Chunk` model is defined in `zapomni_db.models` and imported by Core.

```python
from typing import List, Dict, Any, Optional
from datetime import datetime
from zapomni_db.models import Chunk  # Shared Chunk model (Pydantic)

@dataclass
class MemoryInput:
    """
    Input model for memory processing.

    Attributes:
        text: Raw text content
        metadata: Optional metadata (tags, source, date, etc.)
    """
    text: str
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ProcessedMemory:
    """
    Fully processed memory ready for storage.

    Attributes:
        original_text: Original input text
        chunks: List of semantic chunks
        embeddings: Embeddings for each chunk (List of 768-dim vectors)
        metadata: Original + computed metadata
        entities: Extracted entities (Phase 2, optional)
        relationships: Detected relationships (Phase 2, optional)
    """
    original_text: str
    chunks: List[Chunk]
    embeddings: List[List[float]]
    metadata: Dict[str, Any]
    entities: Optional[List["Entity"]] = None
    relationships: Optional[List["Relationship"]] = None

@dataclass
class SearchResult:
    """
    Single search result.

    Attributes:
        memory_id: UUID of matching memory
        text: Chunk text
        similarity_score: Relevance score (0-1)
        tags: List of tags from metadata
        source: Source identifier
        timestamp: When memory was created
        highlight: Optional highlighted excerpt (Phase 2)
    """
    memory_id: str
    text: str
    similarity_score: float
    tags: List[str]
    source: str
    timestamp: datetime
    highlight: Optional[str] = None

@dataclass
class Entity:
    """
    Knowledge graph entity (Phase 2).

    Attributes:
        name: Entity name (e.g., "Python")
        type: Entity type (PERSON, ORG, TECHNOLOGY, etc.)
        description: Brief description
        confidence: Extraction confidence (0-1)
        mentions: Number of times mentioned
    """
    name: str
    type: str
    description: str
    confidence: float
    mentions: int = 1

@dataclass
class Relationship:
    """
    Knowledge graph relationship (Phase 2).

    Attributes:
        source_entity: Source entity name
        target_entity: Target entity name
        relationship_type: Type (CREATED_BY, USES, IS_A, etc.)
        confidence: Detection confidence (0-1)
        evidence: Text snippet supporting relationship
    """
    source_entity: str
    target_entity: str
    relationship_type: str
    confidence: float
    evidence: str
```

### Service Interfaces

```python
from typing import Protocol, List

class ChunkerProtocol(Protocol):
    """Interface for chunking services."""

    def chunk(self, text: str) -> List[Chunk]:
        """Split text into semantic chunks."""
        ...

class EmbedderProtocol(Protocol):
    """Interface for embedding services."""

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        ...

class ExtractorProtocol(Protocol):
    """Interface for entity extraction services."""

    async def extract_entities(self, text: str) -> List[Entity]:
        """Extract entities from text."""
        ...

    async def extract_relationships(
        self,
        text: str,
        entities: List[Entity]
    ) -> List[Relationship]:
        """Detect relationships between entities."""
        ...

class SearchEngineProtocol(Protocol):
    """Interface for search engines."""

    async def search(
        self,
        query: str,
        limit: int,
        filters: Optional[Dict[str, Any]]
    ) -> List[SearchResult]:
        """Execute search and return results."""
        ...
```

---

## Dependencies

### External Dependencies

**Core Processing:**
- `langchain>=0.1.0` - For RecursiveCharacterTextSplitter (semantic chunking)
- `sentence-transformers>=2.2.0` - Fallback embedding model (all-MiniLM-L6-v2)
- `httpx>=0.25.0` - Async HTTP client for Ollama API calls
- `rank-bm25>=0.2.2` - BM25 keyword search implementation (Phase 2)

**Entity Extraction (Phase 2):**
- `spacy>=3.7.0` - NER (Named Entity Recognition)
- `en_core_web_sm` - English language model for SpaCy

**Document Processing:**
- `pymupdf>=1.23.0` - PDF text extraction (fitz library)
- `python-docx>=1.0.0` - DOCX document processing
- `tree-sitter>=0.20.0` - AST parsing for code (Phase 3)
- `tree-sitter-python>=0.20.0` - Python grammar for tree-sitter

**Utilities:**
- `pydantic>=2.5.0` - Data validation and settings management
- `structlog>=23.2.0` - Structured logging
- `tiktoken>=0.5.0` - Token counting (OpenAI's tokenizer)

### Internal Dependencies

**From zapomni_db:**
- `FalkorDBClient` - For vector storage and graph operations
- `RedisCacheClient` - For semantic cache (Phase 2, optional)
- `models` - Shared data models (Memory, Chunk, Entity, Relationship)

### Dependency Rationale

**Why LangChain?**
- Industry-standard RecursiveCharacterTextSplitter
- Semantic-aware chunking (respects paragraphs, sentences)
- Well-tested, maintained
- Minimal overhead (we use only text splitting, not full LangChain)

**Why sentence-transformers?**
- Reliable fallback if Ollama unavailable
- all-MiniLM-L6-v2 is fast (< 50ms per embedding)
- 384 dimensions (lightweight)
- Works offline

**Why rank-bm25?**
- Pure Python implementation (no external dependencies)
- Fast BM25 keyword search
- Simple API, easy to integrate

**Why SpaCy?**
- State-of-the-art NER
- Fast (C extensions)
- Pre-trained models available
- Hybrid with LLM gives best precision

**Why tree-sitter?**
- Universal parser for 50+ languages
- AST-based chunking preserves code structure
- Fast incremental parsing
- Better than regex-based approaches

---

## Data Flow

### Input → Processing → Output

```
┌─────────────────────────────────────────────────────────────┐
│                     MEMORY PROCESSING PIPELINE               │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  INPUT: text, metadata                                       │
│    ↓                                                          │
│  VALIDATE: Check non-empty, max 10MB, UTF-8                  │
│    ↓                                                          │
│  CHUNK: Semantic boundaries (256-512 tokens, 10-20% overlap) │
│    ↓                                                          │
│  EMBED: Generate embeddings via Ollama (batch)               │
│    ↓ (check cache first if enabled)                          │
│  EXTRACT (Phase 2): Entities & relationships via SpaCy + LLM │
│    ↓                                                          │
│  STORE: FalkorDB (vector index + graph nodes)                │
│    ↓                                                          │
│  OUTPUT: memory_id (UUID)                                    │
│                                                               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      SEARCH PIPELINE                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  INPUT: query, limit, filters, search_mode                   │
│    ↓                                                          │
│  VALIDATE: Check non-empty, limit 1-100                      │
│    ↓                                                          │
│  EMBED QUERY: Generate query embedding                       │
│    ↓ (check cache if enabled)                                │
│  SEARCH:                                                      │
│    - Vector: Cosine similarity via FalkorDB vector index     │
│    - BM25 (Phase 2): Keyword matching via rank-bm25          │
│    - Hybrid (Phase 2): RRF fusion of both                    │
│    - Graph (Phase 2): Traverse knowledge graph               │
│    ↓                                                          │
│  FILTER: Apply metadata filters (tags, date, source)         │
│    ↓                                                          │
│  RERANK (Phase 2): Cross-encoder refinement                  │
│    ↓                                                          │
│  OUTPUT: List[SearchResult] (top K results)                  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Input Validation

**Text Input:**
- **Min length:** 1 character (after strip)
- **Max length:** 10,000,000 characters (~10MB)
- **Encoding:** UTF-8 required
- **Allowed content:** Alphanumeric, whitespace, punctuation
- **Rejected:** Binary data, null bytes

**Metadata:**
- **Type:** dict[str, Any]
- **Keys:** Must be JSON-serializable (str, int, float, bool, list, dict)
- **Reserved keys:** `memory_id`, `timestamp`, `chunks` (automatically added)
- **Custom keys:** User-defined (tags, source, date, author, etc.)

### Processing Transformations

**Chunking:**
1. Tokenize text (tiktoken, cl100k_base)
2. Split at semantic boundaries (RecursiveCharacterTextSplitter)
3. Chunk size: 256-512 tokens (configurable)
4. Overlap: 10-20% (50-100 tokens)
5. Preserve metadata (chunk index, start/end char offsets)

**Embedding:**
1. Batch chunks (up to 32 per request for efficiency)
2. Check semantic cache (Phase 2):
   - SHA256 hash of chunk text
   - If hit, return cached embedding
3. Call Ollama API:
   - Model: nomic-embed-text (768 dimensions)
   - Timeout: 30s per request
   - Retry: 3 attempts with exponential backoff
4. Fallback to sentence-transformers if Ollama fails
5. Cache result (Phase 2)

**Entity Extraction (Phase 2):**
1. SpaCy NER pass (fast, 80% recall):
   - Extract PERSON, ORG, GPE, DATE, TECHNOLOGY
   - Confidence from SpaCy pipeline
2. LLM refinement (slower, 90% precision):
   - Send top entities to Ollama
   - Ask LLM to validate + extract missed entities
   - Combine results
3. Deduplicate entities (same name → merge)
4. Filter by confidence threshold (> 0.7)

**Relationship Detection (Phase 2):**
1. For each entity pair:
   - Find co-occurrence in text
   - Extract surrounding context (±100 tokens)
2. LLM-based detection:
   - Prompt: "What relationship exists between X and Y in this context?"
   - Parse LLM response (relationship type + confidence)
3. Filter by confidence (> 0.6)
4. Deduplicate (same source-target-type → keep highest confidence)

### Output Guarantees

**Memory ID:**
- UUID v4 format (e.g., `550e8400-e29b-41d4-a716-446655440000`)
- Unique across all memories
- Stable (same memory always has same ID after storage)
- Opaque (no semantic meaning)

**Search Results:**
- Sorted by similarity score (descending)
- Similarity score: 0.0 to 1.0 (1.0 = perfect match)
- Minimum similarity: 0.5 (configurable)
- Limit enforced (never returns more than requested)
- Metadata preserved exactly as stored

---

## Design Decisions

### Decision 1: Ollama for Embeddings (vs. OpenAI API)

**Context:**
We need to generate embeddings for text chunks. Options include cloud APIs (OpenAI, Cohere) or local models (Ollama, sentence-transformers).

**Options Considered:**

**Option A: OpenAI Embeddings API**
- **Pros:** High quality (text-embedding-3-large), fast, scalable
- **Cons:** Costs money ($0.13 per 1M tokens), requires internet, privacy concerns, vendor lock-in
- **Cost Example:** 1K documents (5M tokens) = $0.65

**Option B: Cohere Embeddings API**
- **Pros:** Competitive quality, good docs
- **Cons:** Similar cost structure, cloud dependency

**Option C: Ollama (nomic-embed-text)**
- **Pros:** 100% local, zero cost, privacy-first, 768 dimensions, 81.2% accuracy (MTEB), good community support
- **Cons:** Requires Ollama installed, slower than cloud (but still < 200ms per batch), limited to local compute

**Option D: sentence-transformers (all-MiniLM-L6-v2)**
- **Pros:** 100% local, zero cost, no external dependencies, fast (< 50ms), lightweight (384 dimensions)
- **Cons:** Lower quality than Ollama (78% MTEB), smaller embedding dimension

**Chosen:** **Option C (Ollama)** with **Option D (sentence-transformers) as fallback**

**Rationale:**
1. **Privacy-first mission:** Aligns with Zapomni's core value (data never leaves machine)
2. **Zero cost:** Critical for democratization (from product.md)
3. **Quality sufficient:** 81.2% MTEB accuracy is production-grade for RAG
4. **Fallback safety:** sentence-transformers ensures system works even if Ollama unavailable
5. **Community alignment:** Ollama is popular in local-LLM community (r/LocalLLaMA)

**Trade-offs Accepted:**
- Slower than cloud APIs (200ms vs. 50ms) - acceptable for local use case
- Requires user to install Ollama - documented in setup guide
- Limited to single-machine scaling - out of scope (no distributed mode)

### Decision 2: Semantic Chunking (vs. Fixed-Size)

**Context:**
Text must be split into chunks for embedding. Options include fixed-size, sentence-based, or semantic-aware chunking.

**Options Considered:**

**Option A: Fixed-Size (e.g., 512 characters)**
- **Pros:** Simple, fast, predictable
- **Cons:** Splits mid-sentence, mid-paragraph, breaks semantic units

**Option B: Sentence-Based**
- **Pros:** Respects sentence boundaries
- **Cons:** Sentence length varies (50-500 chars), inconsistent chunk sizes, poor for code

**Option C: Semantic (RecursiveCharacterTextSplitter)**
- **Pros:** Respects paragraphs, sentences, and semantic boundaries; configurable chunk size; 10-20% overlap for context
- **Cons:** Slightly slower, requires LangChain dependency

**Chosen:** **Option C (Semantic)**

**Rationale:**
1. **Better retrieval quality:** Preserving semantic units improves search accuracy (measured in RAG benchmarks)
2. **LangChain is proven:** Widely used, well-tested, minimal overhead
3. **Overlap improves recall:** 10-20% overlap ensures context isn't lost at boundaries
4. **Configurable:** chunk_size and overlap are tunable parameters

**Configuration:**
- Default chunk_size: 512 tokens
- Default overlap: 50 tokens (10%)
- Separator priority: "\n\n" (paragraphs) → "\n" (lines) → " " (words)

### Decision 3: Hybrid SpaCy + LLM for Entity Extraction

**Context (Phase 2):**
Entity extraction can use rule-based (regex), ML-based (SpaCy), or LLM-based approaches.

**Options Considered:**

**Option A: SpaCy NER only**
- **Pros:** Fast (< 10ms per doc), offline, high recall (80%)
- **Cons:** Lower precision (60-70%), misses domain-specific entities

**Option B: LLM-based only (Ollama)**
- **Pros:** High precision (90%), flexible, catches domain-specific entities
- **Cons:** Slow (1-2s per doc), requires LLM inference

**Option C: Hybrid (SpaCy + LLM refinement)**
- **Pros:** Best of both - high recall (SpaCy) + high precision (LLM), faster than LLM-only
- **Cons:** More complex implementation

**Chosen:** **Option C (Hybrid)**

**Rationale:**
1. **Performance target:** Need 80%+ precision AND 75%+ recall (from product.md)
2. **Speed matters:** Pure LLM too slow (1K docs = 30+ minutes); hybrid is 3x faster
3. **Best practices:** Hybrid approach recommended in research (e.g., "LLMs for Information Extraction" papers)

**Workflow:**
1. SpaCy NER first (fast pass, 80% recall)
2. LLM refinement (validate top entities, extract missed ones)
3. Merge results (deduplicate, confidence-weighted)

### Decision 4: Vector Search in Phase 1, Hybrid in Phase 2

**Context:**
Search can be vector-only, keyword-only (BM25), or hybrid.

**Options Considered:**

**Option A: Vector-only (Phase 1)**
- **Pros:** Simple, semantic search, proven
- **Cons:** Misses exact keyword matches (acronyms, names)

**Option B: BM25-only**
- **Pros:** Good for exact matches
- **Cons:** No semantic understanding, misses paraphrases

**Option C: Hybrid (Vector + BM25 + RRF fusion)**
- **Pros:** Best accuracy (3.4x better in benchmarks), handles both semantic and exact
- **Cons:** More complex, slower

**Chosen:** **Option A (Phase 1) → Option C (Phase 2)**

**Rationale:**
1. **MVP simplicity:** Vector-only sufficient for MVP (proven by Claude Context)
2. **Phased complexity:** Add hybrid in Phase 2 after core is stable
3. **Performance budget:** Hybrid adds latency; need to optimize before adding
4. **Research-backed:** Hybrid RAG is state-of-the-art (from research/03_best_practices_patterns.md)

### Decision 5: Async Task Queue for Graph Construction

**Context (Phase 2):**
Knowledge graph construction is slow (1K docs = 10 minutes). Options: synchronous, background thread, or async task queue.

**Options Considered:**

**Option A: Synchronous**
- **Pros:** Simple, immediate feedback
- **Cons:** Blocks MCP server (unacceptable), timeout issues

**Option B: Python threading**
- **Pros:** Built-in, no dependencies
- **Cons:** GIL limits concurrency, complex state management

**Option C: Async task queue (asyncio + task manager)**
- **Pros:** Non-blocking, progress tracking, cancel support, Pythonic (async/await)
- **Cons:** Requires task manager implementation

**Chosen:** **Option C (Async task queue)**

**Rationale:**
1. **User experience:** Must not block MCP server (from product.md UX goals)
2. **Progress tracking:** Users want to see "60% complete" status
3. **Cancellation:** Users should be able to cancel long operations
4. **Python-native:** asyncio is standard, no external dependencies

**Implementation:**
- TaskManager class (stores task state)
- StatusTracker (provides progress updates)
- Return task_id immediately, poll via graph_status()

---

## Non-Functional Requirements

### Performance

**Latency Targets:**

| Operation | P50 | P95 | P99 | Max |
|-----------|-----|-----|-----|-----|
| add_memory (< 1KB) | 50ms | 100ms | 200ms | 500ms |
| add_memory (< 100KB) | 200ms | 400ms | 800ms | 1000ms |
| search_memory (vector) | 100ms | 200ms | 500ms | 1000ms |
| search_memory (hybrid, P2) | 150ms | 300ms | 600ms | 1200ms |
| get_stats | 10ms | 20ms | 50ms | 100ms |
| build_graph (per doc, P2) | 300ms | 600ms | 1000ms | 2000ms |

**Throughput:**
- Concurrent add_memory: 10 ops/sec (single process)
- Concurrent search_memory: 20 ops/sec
- Batch embedding (32 chunks): < 1 second

**Resource Usage:**
- Memory: < 4GB RAM for 10K documents indexed
- CPU: Embedding generation is CPU-bound (use batching to amortize)
- Disk: Minimal (data stored in FalkorDB, not in this module)

### Scalability

**Current Scope (Phase 1-3):**
- Single-user, single-machine (local-first)
- 1K-10K documents (target use case)
- Linear scaling O(n) with document count

**Bottlenecks:**
1. **Ollama embedding generation:** Sequential API calls
   - Mitigation: Batch processing (32 chunks per request)
2. **SpaCy NER (Phase 2):** CPU-bound
   - Mitigation: Process documents in parallel (async tasks)
3. **FalkorDB vector search:** Query time increases with corpus size
   - Mitigation: FalkorDB uses HNSW index (sublinear search)

**Out of Scope:**
- Distributed processing (multiple machines)
- Horizontal scaling (load balancing)
- Multi-tenant isolation

### Security

**Input Validation:**
- ✅ All inputs validated before processing
- ✅ Max size enforced (10MB)
- ✅ UTF-8 encoding validated
- ✅ No code execution from user input

**Data Protection:**
- ✅ Data never sent to external APIs (except local Ollama)
- ✅ No logging of sensitive data (PII redacted if detected)
- ✅ Embeddings are one-way (cannot reconstruct original text)

**Error Handling:**
- ✅ No stack traces in user-facing errors
- ✅ Generic error messages (no internal paths leaked)
- ✅ Structured logging to stderr (separate from stdout)

### Reliability

**Error Handling Strategy:**

**Transient Errors (retry):**
- Ollama API timeout → retry 3x with exponential backoff (1s, 2s, 4s)
- FalkorDB connection error → retry 3x
- Network errors → retry with backoff

**Permanent Errors (fail fast):**
- Invalid input (empty text) → ValidationError immediately
- Ollama model not found → EmbeddingError, suggest model download
- Encoding error (non-UTF-8) → ValidationError with helpful message

**Graceful Degradation:**
- Ollama unavailable → fallback to sentence-transformers
- SpaCy model missing (Phase 2) → skip entity extraction, log warning
- Cache unavailable (Phase 2) → disable caching, continue without

**Recovery Strategies:**
- Partial processing success: Store what succeeded, report what failed
- Long-running tasks (Phase 2): Persist state, resume on crash
- Transaction safety: Use DB transactions where applicable

**Fail-Safe Mechanisms:**
- Input validation prevents bad data from entering system
- Timeouts prevent infinite loops
- Resource limits (max 10MB input) prevent OOM errors

---

## Testing Strategy

### Unit Testing (70% of tests)

**Components to Test:**
- `SemanticChunker.chunk()` - chunking logic
- `OllamaEmbedder.embed()` - embedding generation
- `EntityExtractor.extract_entities()` - entity extraction (Phase 2)
- `VectorSearch.search()` - search algorithm
- All validators (input validation functions)

**Mocking Strategy:**
- Mock Ollama API responses (httpx)
- Mock FalkorDB client (return fake data)
- Mock Redis cache (in-memory dict)
- Use fixtures for sample text, embeddings, entities

**Test Coverage:**
- Happy path (valid inputs, expected outputs)
- Edge cases (empty input, max size, special chars)
- Error cases (API failures, timeouts, invalid data)
- Performance tests (verify latency targets with pytest-benchmark)

**Example Unit Tests:**
```python
# tests/unit/test_chunker.py

def test_semantic_chunker_basic():
    """Test basic chunking with default settings."""
    chunker = SemanticChunker(chunk_size=512, overlap=50)
    text = "A" * 1000  # 1000 chars

    chunks = chunker.chunk(text)

    assert len(chunks) >= 1
    assert all(isinstance(c, Chunk) for c in chunks)
    assert chunks[0].index == 0

def test_semantic_chunker_overlap():
    """Test that chunks have proper overlap."""
    chunker = SemanticChunker(chunk_size=100, overlap=20)
    text = "A" * 100 + "B" * 100

    chunks = chunker.chunk(text)

    # Verify overlap exists
    assert len(chunks) >= 2
    # Overlap should contain chars from both chunks
    overlap_text = text[chunks[0].end_char - 20:chunks[0].end_char]
    assert "A" in overlap_text or "B" in overlap_text

def test_semantic_chunker_empty_raises():
    """Test that empty input raises ValidationError."""
    chunker = SemanticChunker()

    with pytest.raises(ValidationError, match="empty"):
        chunker.chunk("")
```

### Integration Testing (25% of tests)

**Integration Points to Test:**
1. **MemoryProcessor + FalkorDB**: Full add_memory → storage → retrieval flow
2. **MemoryProcessor + Ollama**: Real embedding generation (requires Ollama running)
3. **MemoryProcessor + Cache**: Semantic cache hit/miss behavior
4. **Search + FalkorDB**: Vector search with real DB

**Test Environment:**
- Docker Compose with FalkorDB, Redis, Ollama (optional)
- Test data fixtures (sample documents, embeddings)
- Cleanup after each test (clear DB)

**Example Integration Tests:**
```python
# tests/integration/test_memory_processor.py

@pytest.mark.asyncio
async def test_add_and_search_memory(db_client, ollama_host):
    """Test full workflow: add → search → retrieve."""
    processor = MemoryProcessor(db=db_client, ollama_host=ollama_host)

    # Add memory
    memory_id = await processor.add_memory(
        text="Python is a programming language created by Guido van Rossum",
        metadata={"source": "test"}
    )

    assert memory_id is not None

    # Search for it
    results = await processor.search_memory(
        query="Who created Python?",
        limit=5
    )

    assert len(results) >= 1
    assert any(memory_id in r.memory_id for r in results)
    assert results[0].similarity_score > 0.5
```

### Component Testing (Overlap with Integration)

**Focus:** Test components with real dependencies (not fully mocked)

**Examples:**
- `OllamaEmbedder` with real Ollama API
- `FalkorDBClient` with real FalkorDB instance
- `SemanticChunker` with real LangChain splitter

**Value:** Catches integration issues that unit tests miss

---

## Future Considerations

### Phase 2 Enhancements (Weeks 3-4)

**Semantic Caching:**
- Implement Redis-backed cache for embeddings
- Target: 60%+ cache hit rate
- LRU eviction policy, 24h TTL
- Cache key: SHA256 of normalized text

**Hybrid Search:**
- Add BM25 keyword search (rank-bm25)
- Implement RRF (Reciprocal Rank Fusion) for result merging
- Cross-encoder reranking (top 20 → refine to top 10)
- Configurable fusion weights (alpha parameter)

**Knowledge Graph Construction:**
- Hybrid SpaCy + LLM entity extraction
- Relationship detection via LLM
- Async task queue for background processing
- Progress tracking and cancellation support

### Phase 3 Enhancements (Weeks 5-6)

**Code Intelligence:**
- AST-based chunking with tree-sitter
- Extract functions, classes, imports
- Build call graphs (CALLS, INHERITS relationships)
- Support Python, JavaScript, TypeScript, Go, Rust

**Advanced Search:**
- Graph traversal queries (find related entities)
- Multi-hop reasoning (Python → Django → PostgreSQL)
- Personalized ranking (user preferences)

### Known Limitations

**Current (Phase 1):**
- Vector-only search (no keyword matching yet)
- No entity extraction (just text + embeddings)
- No code understanding (treats code as text)
- No caching (every embedding generated fresh)
- Single-threaded processing (no parallelism)

**Architectural:**
- Single-machine only (no distributed mode)
- CPU-bound embedding generation (limited by Ollama speed)
- Linear scaling with document count (no partitioning)

### Evolution Path

**Beyond Phase 3:**
1. **Multi-modal support:** PDF tables, images (OCR), audio (transcription)
2. **Advanced caching:** Hierarchical cache (L1 memory, L2 Redis, L3 disk)
3. **Incremental processing:** Update existing memories instead of full reindex
4. **Query optimization:** Query plan analysis, automatic index selection
5. **Plugin system:** Custom chunkers, embedders, extractors

---

## References

### Internal Documents
- [product.md](../../steering/product.md) - Product vision, features, success criteria
- [tech.md](../../steering/tech.md) - Technology stack, Ollama + FalkorDB rationale
- [structure.md](../../steering/structure.md) - Project structure, coding conventions
- [SPEC_METHODOLOGY.md](/home/dev/zapomni/SPEC_METHODOLOGY.md) - Specification approach

### Research Documents
- [00_final_synthesis.md](/home/dev/zapomni/research/00_final_synthesis.md) - Implementation roadmap
- [01_tech_stack_infrastructure.md](/home/dev/zapomni/research/01_tech_stack_infrastructure.md) - Tech evaluation
- [03_best_practices_patterns.md](/home/dev/zapomni/research/03_best_practices_patterns.md) - RAG best practices

### External Resources
- LangChain Text Splitters: https://python.langchain.com/docs/modules/data_connection/document_transformers/
- Ollama Embeddings API: https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
- Nomic Embed Text: https://huggingface.co/nomic-ai/nomic-embed-text-v1
- SpaCy NER: https://spacy.io/usage/linguistic-features#named-entities
- Rank BM25: https://github.com/dorianbrown/rank_bm25
- Hybrid RAG Research: "Dense Passage Retrieval for Open-Domain Question Answering" (Karpukhin et al., 2020)

### Related Specifications (Future)
- zapomni_mcp_module.md - MCP protocol adapter (upstream)
- zapomni_db_module.md - Database clients (downstream)
- cross_module_interfaces.md - How zapomni_core interfaces with other modules

---

## Appendix: Error Handling Details

### Exception Hierarchy

```python
# zapomni_core/exceptions.py

class ZapomniCoreError(Exception):
    """Base exception for zapomni_core module."""
    pass

class ValidationError(ZapomniCoreError):
    """Input validation failed."""
    pass

class EmbeddingError(ZapomniCoreError):
    """Embedding generation failed."""
    pass

class ChunkingError(ZapomniCoreError):
    """Text chunking failed."""
    pass

class ExtractionError(ZapomniCoreError):
    """Entity/relationship extraction failed."""
    pass

class SearchError(ZapomniCoreError):
    """Search operation failed."""
    pass

class TaskError(ZapomniCoreError):
    """Background task operation failed."""
    pass

class TimeoutError(ZapomniCoreError):
    """Operation exceeded timeout."""
    pass
```

### Error Messages (User-Friendly)

**ValidationError examples:**
- "Text cannot be empty"
- "Text exceeds maximum length (10,000,000 characters)"
- "Text must be valid UTF-8"
- "Query cannot be empty"
- "Limit must be between 1 and 100"

**EmbeddingError examples:**
- "Failed to generate embedding: Ollama model 'nomic-embed-text' not found. Please run: ollama pull nomic-embed-text"
- "Embedding generation timed out after 30 seconds. Check Ollama service."
- "Ollama service unavailable. Falling back to sentence-transformers."

**SearchError examples:**
- "Vector search failed: Database connection lost"
- "Search timeout exceeded (1000ms)"

### Logging Standards

**Structured Logging (structlog):**
```python
import structlog

logger = structlog.get_logger()

# Good: Structured ✅
logger.info(
    "memory_added",
    memory_id=memory_id,
    chunk_count=len(chunks),
    embedding_dim=len(embeddings[0]),
    duration_ms=duration
)

# Bad: Unstructured ❌
logger.info(f"Added memory {memory_id} with {len(chunks)} chunks")
```

**Log Levels:**
- DEBUG: Detailed flow (chunk sizes, embedding dimensions, cache hits)
- INFO: Normal operations (memory added, search completed)
- WARNING: Degraded performance (fallback to sentence-transformers, cache disabled)
- ERROR: Operation failed (embedding error, DB connection lost)
- CRITICAL: System failure (should never happen in zapomni_core)

---

**Document Status:** Draft v1.0
**Created:** 2025-11-23
**Author:** Goncharenko Anton aka alienxs2
**Copyright:** Copyright (c) 2025 Goncharenko Anton aka alienxs2
**License:** MIT License

**Total Sections:** 14
**Total Code Examples:** 25+
**Total Data Models:** 7
**Ready for Review:** Yes ✅
