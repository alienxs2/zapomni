# MemoryProcessor - Component Specification

**Level:** 2 (Component)
**Module:** zapomni_core
**Author:** Goncharenko Anton aka alienxs2
**Status:** Draft
**Version:** 1.0
**Created:** 2025-11-23

---

## Overview

### Purpose

MemoryProcessor is the **main orchestrator and entry point** for all memory operations in the Zapomni system. It implements the MemoryEngine protocol and coordinates the complete end-to-end pipeline for adding, searching, and managing memories.

This component serves as the **central brain** that integrates all specialized components (chunking, embedding, extraction, storage) into a cohesive memory processing system. It provides the high-level API that MCP tools and other clients use to interact with the memory system.

### Responsibilities

1. **Memory Addition Pipeline:** Orchestrate full workflow: validate → chunk → embed → extract → store
2. **Search Coordination:** Execute search queries across multiple search modes (vector, BM25, hybrid, graph)
3. **Dependency Management:** Inject and coordinate all dependent components (chunker, embedder, extractor, DB client)
4. **Error Handling:** Catch and handle errors from all pipeline stages, provide meaningful error messages
5. **Statistics Aggregation:** Collect and return system-wide statistics from all components
6. **Background Task Management (Phase 2):** Launch and track async knowledge graph construction tasks

### Position in Module

MemoryProcessor sits at the **top of the core module hierarchy** and is the primary interface:

```
MCP Tools (add_memory, search_memory, get_stats)
    ↓ calls
MemoryProcessor (THIS) ← Main orchestrator
    ↓ coordinates
┌───────────────┬─────────────────┬──────────────────┬─────────────┐
│               │                 │                  │             │
SemanticChunker OllamaEmbedder   EntityExtractor    FalkorDBClient
                                  (Phase 2)
```

**Key Relationships:**
- **Used by:** MCP Tools (AddMemoryTool, SearchMemoryTool, GetStatsTool)
- **Uses:** SemanticChunker (chunking), OllamaEmbedder (embeddings), EntityExtractor (entities, Phase 2), FalkorDBClient (storage)
- **Implements:** MemoryEngine protocol (defines standard memory operations interface)

---

## Class Definition

### Class Diagram

```
┌─────────────────────────────────────────────────┐
│              MemoryProcessor                    │
├─────────────────────────────────────────────────┤
│ - db_client: FalkorDBClient                     │
│ - chunker: SemanticChunker                      │
│ - embedder: OllamaEmbedder                      │
│ - extractor: Optional[EntityExtractor]          │
│ - cache: Optional[SemanticCache]                │
│ - task_manager: Optional[TaskManager]           │
│ - config: ProcessorConfig                       │
│ - logger: structlog.BoundLogger                 │
├─────────────────────────────────────────────────┤
│ + __init__(db_client, chunker, embedder, ...)   │
│ + add_memory(text, metadata) -> str             │
│ + search_memory(query, limit, filters, mode)    │
│   -> List[SearchResult]                         │
│ + get_stats() -> Dict[str, Any]                 │
│ + build_knowledge_graph(memory_ids, mode)       │
│   -> str                                         │
│ - _validate_text(text) -> None                  │
│ - _validate_metadata(metadata) -> None          │
│ - _create_memory_id() -> str                    │
│ - _process_chunks(text) -> List[Chunk]          │
│ - _generate_embeddings(chunks) -> List[List[float]] │
│ - _extract_entities(text) -> List[Entity]       │
│ - _store_memory(chunks, embeddings, metadata)   │
│   -> str                                         │
└─────────────────────────────────────────────────┘
```

### Full Class Signature

```python
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import structlog
import uuid
from datetime import datetime

from zapomni_core.chunking import SemanticChunker, Chunk
from zapomni_core.embeddings import OllamaEmbedder
from zapomni_core.extraction import EntityExtractor
from zapomni_core.cache import SemanticCache
from zapomni_core.tasks import TaskManager
from zapomni_db import FalkorDBClient
from zapomni_core.exceptions import (
    ValidationError,
    EmbeddingError,
    ExtractionError,
    StorageError
)


@dataclass
class ProcessorConfig:
    """
    Configuration for MemoryProcessor.

    Attributes:
        enable_cache: Enable semantic embedding cache (Phase 2)
        enable_extraction: Enable entity extraction (Phase 2)
        enable_graph: Enable knowledge graph construction (Phase 2)
        max_text_length: Maximum text length in characters (default: 10MB)
        batch_size: Batch size for embedding generation (default: 32)
        search_mode: Default search mode ("vector", "bm25", "hybrid")
    """
    enable_cache: bool = False
    enable_extraction: bool = False
    enable_graph: bool = False
    max_text_length: int = 10_000_000
    batch_size: int = 32
    search_mode: str = "vector"


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


class MemoryProcessor:
    """
    Main orchestrator for memory processing operations.

    Coordinates the complete pipeline: chunking → embedding → extraction → storage.
    Provides high-level API for adding memories, searching, and retrieving statistics.

    This class implements dependency injection - all components are provided
    at initialization, making it easy to test and configure.

    Attributes:
        db_client: FalkorDB client for storage and retrieval
        chunker: SemanticChunker for text chunking
        embedder: OllamaEmbedder for embedding generation
        extractor: EntityExtractor for entity/relationship extraction (Phase 2, optional)
        cache: SemanticCache for embedding caching (Phase 2, optional)
        task_manager: TaskManager for background tasks (Phase 2, optional)
        config: ProcessorConfig with system configuration
        logger: Structured logger for operations tracking

    Example:
        ```python
        from zapomni_core import MemoryProcessor
        from zapomni_core.chunking import SemanticChunker
        from zapomni_core.embeddings import OllamaEmbedder
        from zapomni_db import FalkorDBClient

        # Initialize dependencies
        db = FalkorDBClient(host="localhost", port=6379)
        chunker = SemanticChunker(chunk_size=512, chunk_overlap=50)
        embedder = OllamaEmbedder(
            host="http://localhost:11434",
            model="nomic-embed-text"
        )

        # Create processor
        processor = MemoryProcessor(
            db_client=db,
            chunker=chunker,
            embedder=embedder
        )

        # Add memory
        memory_id = await processor.add_memory(
            text="Python is a programming language created by Guido van Rossum.",
            metadata={"source": "user", "tags": ["python", "programming"]}
        )
        print(f"Stored with ID: {memory_id}")

        # Search memories
        results = await processor.search_memory(
            query="Who created Python?",
            limit=5
        )
        for result in results:
            print(f"{result.similarity_score:.2f}: {result.text}")
        ```
    """

    def __init__(
        self,
        db_client: FalkorDBClient,
        chunker: SemanticChunker,
        embedder: OllamaEmbedder,
        extractor: Optional[EntityExtractor] = None,
        cache: Optional[SemanticCache] = None,
        task_manager: Optional[TaskManager] = None,
        config: Optional[ProcessorConfig] = None
    ) -> None:
        """
        Initialize MemoryProcessor with all dependencies.

        This constructor uses **dependency injection** - all components are
        provided externally, making the processor testable and flexible.

        Args:
            db_client: FalkorDB client for storage (required)
            chunker: SemanticChunker for text chunking (required)
            embedder: OllamaEmbedder for embedding generation (required)
            extractor: EntityExtractor for entity extraction (optional, Phase 2)
            cache: SemanticCache for embedding caching (optional, Phase 2)
            task_manager: TaskManager for background tasks (optional, Phase 2)
            config: ProcessorConfig with system configuration (optional, uses defaults)

        Raises:
            ValueError: If any required dependency is None
            ValueError: If config has invalid values (e.g., max_text_length <= 0)

        Example:
            ```python
            # Minimal configuration (Phase 1)
            processor = MemoryProcessor(
                db_client=FalkorDBClient(),
                chunker=SemanticChunker(),
                embedder=OllamaEmbedder()
            )

            # Full configuration (Phase 2+)
            processor = MemoryProcessor(
                db_client=FalkorDBClient(),
                chunker=SemanticChunker(),
                embedder=OllamaEmbedder(),
                extractor=EntityExtractor(),
                cache=SemanticCache(redis_client=RedisClient()),
                task_manager=TaskManager(),
                config=ProcessorConfig(
                    enable_cache=True,
                    enable_extraction=True,
                    enable_graph=True
                )
            )
            ```
        """

    async def add_memory(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add memory to system with full processing pipeline.

        This is the main entry point for adding new memories. It orchestrates
        the complete 6-stage pipeline:
        1. Validate input (text non-empty, metadata valid)
        2. Chunk text (semantic boundaries, 256-512 tokens)
        3. Generate embeddings (via Ollama, with cache check if enabled)
        4. Extract entities (Phase 2, if enabled)
        5. Store in FalkorDB (chunks + embeddings + metadata + entities)
        6. Return memory ID (UUID)

        The pipeline is **transactional** - if any stage fails, no data is stored.
        All errors are caught, logged, and re-raised with context.

        Args:
            text: Text content to remember
                - Constraints: Non-empty, max 10,000,000 chars (~10MB)
                - Encoding: Must be valid UTF-8
                - Content: Natural language or code (treated as text in Phase 1)
            metadata: Optional metadata dict
                - Structure: Flat dict with string keys
                - Supported types: str, int, float, bool, list, dict (JSON-serializable)
                - Reserved keys: "memory_id", "timestamp", "chunks" (auto-added)
                - Common fields: "tags" (list), "source" (str), "date" (str), "author" (str)

        Returns:
            memory_id: UUID string identifying stored memory (format: UUID v4)
                Example: "550e8400-e29b-41d4-a716-446655440000"

        Raises:
            ValidationError: If text empty, too large, or non-UTF-8
            ValidationError: If metadata contains reserved keys or non-JSON-serializable values
            EmbeddingError: If embedding generation fails (Ollama unavailable)
            ExtractionError: If entity extraction fails (Phase 2)
            StorageError: If FalkorDB storage fails (connection error, transaction failed)

        Performance Target:
            - Small input (< 1KB): < 100ms
            - Medium input (< 10KB): < 300ms
            - Large input (< 100KB): < 500ms
            - Maximum allowed: < 1000ms

        Example:
            ```python
            processor = MemoryProcessor(...)

            # Simple add
            memory_id = await processor.add_memory(
                text="Python is a high-level programming language."
            )

            # Add with metadata
            memory_id = await processor.add_memory(
                text="The Django framework was created in 2005.",
                metadata={
                    "tags": ["django", "python", "web"],
                    "source": "wikipedia",
                    "date": "2025-11-23",
                    "author": "user123"
                }
            )

            # Error handling
            try:
                memory_id = await processor.add_memory("")
            except ValidationError as e:
                print(f"Invalid input: {e}")
            ```
        """

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

        Performance Target:
            - P50 latency: < 200ms
            - P95 latency: < 500ms
            - P99 latency: < 1000ms

        Example:
            ```python
            processor = MemoryProcessor(...)

            # Simple search
            results = await processor.search_memory(
                query="What is Python?",
                limit=5
            )
            for result in results:
                print(f"{result.similarity_score:.2f}: {result.text}")

            # Filtered search
            results = await processor.search_memory(
                query="web frameworks",
                limit=10,
                filters={"tags": ["python", "web"], "source": "documentation"}
            )

            # Hybrid search (Phase 2)
            results = await processor.search_memory(
                query="Django ORM features",
                limit=20,
                search_mode="hybrid"
            )

            # Error handling
            try:
                results = await processor.search_memory("", limit=5)
            except ValidationError as e:
                print(f"Invalid query: {e}")
            ```
        """

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive memory system statistics.

        Aggregates statistics from all components (DB, cache, extractor) and
        returns a single dict with system-wide metrics.

        Returns:
            Dictionary with statistics:
            - total_memories: int - Total number of unique memories stored
            - total_chunks: int - Total number of chunks across all memories
            - database_size_mb: float - Total size of FalkorDB data in MB
            - avg_chunks_per_memory: float - Average chunks per memory
            - cache_hit_rate: float - Embedding cache hit rate 0-1 (if caching enabled, Phase 2)
            - cache_size_mb: float - Cache size in MB (if caching enabled, Phase 2)
            - avg_query_latency_ms: int - Average search latency in milliseconds
            - total_entities: int - Total entities in knowledge graph (if graph built, Phase 2)
            - total_relationships: int - Total relationships in knowledge graph (if graph built, Phase 2)
            - oldest_memory_date: datetime - Timestamp of oldest memory
            - newest_memory_date: datetime - Timestamp of newest memory

        Raises:
            DatabaseError: If DB query fails
            StatisticsError: If statistics calculation fails

        Performance Target:
            - Execution time: < 100ms
            - No expensive computations (all stats pre-computed or cached)

        Example:
            ```python
            processor = MemoryProcessor(...)

            stats = await processor.get_stats()

            print(f"Total memories: {stats['total_memories']}")
            print(f"Total chunks: {stats['total_chunks']}")
            print(f"Database size: {stats['database_size_mb']:.2f} MB")
            print(f"Average query latency: {stats['avg_query_latency_ms']} ms")

            if stats.get('cache_hit_rate') is not None:
                print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")

            if stats.get('total_entities') is not None:
                print(f"Entities: {stats['total_entities']}")
                print(f"Relationships: {stats['total_relationships']}")
            ```
        """

    async def build_knowledge_graph(
        self,
        memory_ids: Optional[List[str]] = None,
        mode: str = "full"
    ) -> str:
        """
        Build knowledge graph from memories (async background task, Phase 2).

        Launches a background task to extract entities and relationships from
        specified memories, then constructs/updates the knowledge graph in FalkorDB.

        This operation is **asynchronous** - it returns immediately with a task_id
        that can be used to track progress via task_manager.get_status(task_id).

        Args:
            memory_ids: Specific memories to process
                - None: Process all unprocessed memories (default)
                - List[str]: Process only these memory IDs
                - Empty list: No-op, returns task_id immediately

            mode: What to extract and build
                - "entities_only": Extract entities, skip relationships
                - "relationships_only": Extract relationships (requires entities exist)
                - "full": Extract both entities and relationships (default)

        Returns:
            task_id: UUID string for tracking background task progress
                - Format: UUID v4
                - Use: task_manager.get_status(task_id) to check progress
                - Example: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

        Raises:
            ValidationError: If memory_ids contains invalid UUIDs
            ValidationError: If mode not in ["entities_only", "relationships_only", "full"]
            TaskError: If task queue is full (max concurrent tasks exceeded)
            NotImplementedError: In Phase 1 (feature not implemented yet)

        Performance Target:
            - Launch overhead: < 50ms (just creates task, returns immediately)
            - Processing time: ~600ms per document (background)
            - 1K documents: < 10 minutes (background processing)
            - Entity extraction: 80%+ precision, 75%+ recall
            - Relationship detection: 70%+ precision, 65%+ recall

        Background Task Behavior:
            - Non-blocking: Returns immediately, processing happens in background
            - Progress tracking: task_manager provides progress updates (0-100%)
            - Cancellable: Can cancel via task_manager.cancel(task_id)
            - Error handling: Task failures logged, error stored in task status

        Example:
            ```python
            processor = MemoryProcessor(...)

            # Build graph for all unprocessed memories
            task_id = await processor.build_knowledge_graph()
            print(f"Started task {task_id}")

            # Check progress
            while True:
                status = await task_manager.get_status(task_id)
                if status['state'] == 'completed':
                    print(f"Graph built! Processed {status['items_processed']} memories")
                    break
                elif status['state'] == 'failed':
                    print(f"Task failed: {status['error']}")
                    break
                else:
                    print(f"Progress: {status['progress']}%")
                    await asyncio.sleep(1)

            # Build graph for specific memories
            task_id = await processor.build_knowledge_graph(
                memory_ids=["uuid1", "uuid2", "uuid3"],
                mode="full"
            )

            # Extract only entities (faster)
            task_id = await processor.build_knowledge_graph(
                mode="entities_only"
            )
            ```
        """

    def _validate_text(self, text: str) -> None:
        """
        Validate text input before processing (private helper).

        Checks:
        - Non-empty (after strip)
        - Valid UTF-8 encoding
        - Length <= max_text_length (default 10MB)

        Args:
            text: Text to validate

        Raises:
            ValidationError: If validation fails
        """

    def _validate_metadata(self, metadata: Optional[Dict[str, Any]]) -> None:
        """
        Validate metadata structure and content (private helper).

        Checks:
        - All keys are strings
        - No reserved keys ("memory_id", "timestamp", "chunks")
        - All values are JSON-serializable

        Args:
            metadata: Metadata dict to validate

        Raises:
            ValidationError: If validation fails
        """

    def _create_memory_id(self) -> str:
        """
        Generate unique memory ID (private helper).

        Returns:
            UUID v4 string (e.g., "550e8400-e29b-41d4-a716-446655440000")
        """

    async def _process_chunks(self, text: str) -> List[Chunk]:
        """
        Chunk text using SemanticChunker (private helper).

        Args:
            text: Text to chunk

        Returns:
            List of Chunk objects

        Raises:
            ChunkingError: If chunking fails
        """

    async def _generate_embeddings(
        self,
        chunks: List[Chunk]
    ) -> List[List[float]]:
        """
        Generate embeddings for chunks using OllamaEmbedder (private helper).

        Checks cache first (if enabled), generates embeddings for cache misses,
        stores in cache (if enabled).

        Args:
            chunks: List of Chunk objects

        Returns:
            List of embedding vectors (one per chunk)

        Raises:
            EmbeddingError: If embedding generation fails
        """

    async def _extract_entities(
        self,
        text: str,
        chunks: List[Chunk]
    ) -> tuple[List[Entity], List[Relationship]]:
        """
        Extract entities and relationships from text (private helper, Phase 2).

        Args:
            text: Original text
            chunks: Chunks for context

        Returns:
            Tuple of (entities, relationships)

        Raises:
            ExtractionError: If extraction fails
            NotImplementedError: In Phase 1
        """

    async def _store_memory(
        self,
        memory_id: str,
        chunks: List[Chunk],
        embeddings: List[List[float]],
        metadata: Dict[str, Any],
        entities: Optional[List[Entity]] = None,
        relationships: Optional[List[Relationship]] = None
    ) -> str:
        """
        Store memory in FalkorDB (private helper).

        Creates memory node, chunk nodes, embedding vectors, metadata,
        and optionally entity/relationship nodes.

        Args:
            memory_id: Pre-generated UUID
            chunks: List of chunks
            embeddings: List of embeddings (same length as chunks)
            metadata: Metadata dict (includes timestamp)
            entities: Optional entities (Phase 2)
            relationships: Optional relationships (Phase 2)

        Returns:
            memory_id: Same as input (for consistency)

        Raises:
            StorageError: If DB operation fails
        """
```

---

## Dependencies

### Component Dependencies

**Internal (from zapomni_core):**
- `SemanticChunker` - Text chunking service (required)
- `OllamaEmbedder` - Embedding generation service (required)
- `EntityExtractor` - Entity extraction service (optional, Phase 2)
- `SemanticCache` - Embedding cache (optional, Phase 2)
- `TaskManager` - Background task management (optional, Phase 2)
- `Chunk`, `Entity`, `Relationship` dataclasses (from zapomni_core.models)
- `ValidationError`, `EmbeddingError`, `ExtractionError`, `StorageError` exceptions (from zapomni_core.exceptions)

**From zapomni_db:**
- `FalkorDBClient` - For vector storage and graph operations (required)

### External Libraries

**Required:**
- `structlog>=23.2.0` - Structured logging for operations tracking
- `uuid` - UUID generation (Python stdlib)
- `datetime` - Timestamp handling (Python stdlib)

**Phase 2:**
- No additional external dependencies (all features use existing components)

### Dependency Injection

MemoryProcessor uses **constructor-based dependency injection**:

```python
def __init__(
    self,
    db_client: FalkorDBClient,
    chunker: SemanticChunker,
    embedder: OllamaEmbedder,
    extractor: Optional[EntityExtractor] = None,
    cache: Optional[SemanticCache] = None,
    task_manager: Optional[TaskManager] = None,
    config: Optional[ProcessorConfig] = None
) -> None:
    """All dependencies provided externally."""
    self.db_client = db_client
    self.chunker = chunker
    self.embedder = embedder
    self.extractor = extractor
    self.cache = cache
    self.task_manager = task_manager
    self.config = config or ProcessorConfig()
    self.logger = structlog.get_logger()
```

**Rationale:**
- **Testability:** Easy to mock dependencies in tests
- **Flexibility:** Can swap implementations (e.g., different embedders)
- **Clarity:** All dependencies explicit (no hidden coupling)
- **Best practice:** Industry-standard DI pattern

---

## State Management

### Attributes

**Dependencies (injected, immutable):**
- `db_client: FalkorDBClient` - Database client
  - Lifetime: Set at initialization, never changes
  - Purpose: Storage and retrieval operations

- `chunker: SemanticChunker` - Text chunker
  - Lifetime: Set at initialization, never changes
  - Purpose: Text splitting into chunks

- `embedder: OllamaEmbedder` - Embedding generator
  - Lifetime: Set at initialization, never changes
  - Purpose: Generate embeddings for chunks

- `extractor: Optional[EntityExtractor]` - Entity extractor (Phase 2)
  - Lifetime: Set at initialization, never changes
  - Purpose: Extract entities and relationships

- `cache: Optional[SemanticCache]` - Embedding cache (Phase 2)
  - Lifetime: Set at initialization, never changes
  - Purpose: Cache embeddings for performance

- `task_manager: Optional[TaskManager]` - Task manager (Phase 2)
  - Lifetime: Set at initialization, never changes
  - Purpose: Manage background tasks

**Configuration (immutable after __init__):**
- `config: ProcessorConfig` - System configuration
  - Lifetime: Set at initialization, never changes
  - Purpose: Control behavior flags and limits

**Internal State:**
- `logger: structlog.BoundLogger` - Structured logger
  - Lifetime: Created in __init__, reused for all operations
  - Purpose: Structured logging with context

### State Transitions

MemoryProcessor is **functionally stateless** - each method call is independent:

```
Initial State (after __init__)
    ↓
Ready State (can call add_memory(), search_memory(), etc.)
    ↓ method call
Processing (temporary, within method execution)
    ↓ method returns
Ready State (no state changes, ready for next call)
```

**No Persistent State:**
- Each `add_memory()` call is independent (no shared state)
- Each `search_memory()` call is independent
- No mutable state between calls (dependencies are stateless services)

### Thread Safety

**Thread-Safe:** ✅ Yes (with caveats)

**Reasoning:**
- MemoryProcessor has no mutable state (all attributes final after __init__)
- All dependencies are thread-safe:
  - SemanticChunker: Thread-safe (stateless)
  - OllamaEmbedder: Thread-safe (httpx client is thread-safe)
  - FalkorDBClient: Thread-safe (connection pooling)
- No shared mutable state between method calls

**Concurrency Support:**
- Multiple threads can call methods simultaneously
- Multiple coroutines (async) can call methods simultaneously
- Safe to share single MemoryProcessor instance across threads/coroutines

**Caveats:**
- If using shared cache (Phase 2), cache must be thread-safe (Redis is)
- If using task_manager (Phase 2), task_manager must be thread-safe
- Database transactions are isolated (FalkorDB handles concurrency)

---

## Public Methods (Detailed)

### Method 1: `__init__`

**Signature:**
```python
def __init__(
    self,
    db_client: FalkorDBClient,
    chunker: SemanticChunker,
    embedder: OllamaEmbedder,
    extractor: Optional[EntityExtractor] = None,
    cache: Optional[SemanticCache] = None,
    task_manager: Optional[TaskManager] = None,
    config: Optional[ProcessorConfig] = None
) -> None
```

**Purpose:** Initialize MemoryProcessor with all dependencies via dependency injection

**Parameters:**

- `db_client: FalkorDBClient` (required)
  - Description: FalkorDB client for storage and retrieval
  - Constraints: Must be initialized and connected
  - Example: `FalkorDBClient(host="localhost", port=6379)`

- `chunker: SemanticChunker` (required)
  - Description: Text chunker
  - Constraints: Must be initialized with valid config
  - Example: `SemanticChunker(chunk_size=512, chunk_overlap=50)`

- `embedder: OllamaEmbedder` (required)
  - Description: Embedding generator
  - Constraints: Must be initialized with Ollama host
  - Example: `OllamaEmbedder(host="http://localhost:11434", model="nomic-embed-text")`

- `extractor: Optional[EntityExtractor]` (optional, Phase 2)
  - Description: Entity and relationship extractor
  - Constraints: None if None, must be initialized if provided
  - Example: `EntityExtractor(spacy_model="en_core_web_sm", ollama_host="...")`

- `cache: Optional[SemanticCache]` (optional, Phase 2)
  - Description: Semantic embedding cache
  - Constraints: None if None, must have Redis client if provided
  - Example: `SemanticCache(redis_client=RedisClient())`

- `task_manager: Optional[TaskManager]` (optional, Phase 2)
  - Description: Background task manager
  - Constraints: None if None, must be initialized if provided
  - Example: `TaskManager()`

- `config: Optional[ProcessorConfig]` (optional)
  - Description: System configuration
  - Constraints: None (uses defaults) or valid ProcessorConfig
  - Example: `ProcessorConfig(enable_cache=True, max_text_length=5_000_000)`

**Returns:** None (constructor)

**Raises:**
- `ValueError`: If db_client is None
- `ValueError`: If chunker is None
- `ValueError`: If embedder is None
- `ValueError`: If config.max_text_length <= 0
- `ValueError`: If config.batch_size <= 0
- `ValueError`: If config.search_mode not in ["vector", "bm25", "hybrid", "graph"]

**Preconditions:**
- All required dependencies instantiated
- Database client connected (optional: can fail later in add_memory/search)

**Postconditions:**
- MemoryProcessor instance ready to use
- All dependencies stored as instance attributes
- Logger initialized with context

**Algorithm Outline:**
```
1. Validate required dependencies (db_client, chunker, embedder not None)
2. Validate config (if provided, check max_text_length > 0, batch_size > 0, search_mode valid)
3. Store all dependencies as instance attributes
4. Create default config if None provided
5. Initialize structured logger with component context
6. Log initialization success with config details
```

**Edge Cases:**
1. **db_client = None** → ValueError("db_client is required")
2. **chunker = None** → ValueError("chunker is required")
3. **embedder = None** → ValueError("embedder is required")
4. **config.max_text_length = 0** → ValueError("max_text_length must be positive")
5. **config.search_mode = "invalid"** → ValueError("Invalid search_mode")

**Related Methods:**
- Called by: MCP Tools constructors, MemoryProcessor factory functions
- Calls: ProcessorConfig() (default), structlog.get_logger()

---

### Method 2: `add_memory`

**Signature:**
```python
async def add_memory(
    self,
    text: str,
    metadata: Optional[Dict[str, Any]] = None
) -> str
```

**Purpose:** Execute complete memory addition pipeline (6 stages)

**Parameters:**

- `text: str`
  - Description: Text content to remember
  - Constraints:
    - Non-empty (after strip)
    - Valid UTF-8 encoding
    - Max length: config.max_text_length (default 10,000,000 chars)
    - Min length: 1 character
  - Example: "Python is a programming language."

- `metadata: Optional[Dict[str, Any]]`
  - Description: Optional metadata (tags, source, date, custom fields)
  - Constraints:
    - All keys must be strings
    - No reserved keys: "memory_id", "timestamp", "chunks"
    - All values must be JSON-serializable
  - Structure:
    ```python
    {
        "tags": ["python", "programming"],  # Common
        "source": "wikipedia",              # Common
        "date": "2025-11-23",               # ISO date
        "author": "user123",                # Custom
        "custom_field": "any value"         # Custom
    }
    ```

**Returns:**
- Type: `str`
- Format: UUID v4 (e.g., "550e8400-e29b-41d4-a716-446655440000")
- Uniqueness: Guaranteed unique across all memories
- Stability: Same memory_id for this memory forever

**Raises:**
- `ValidationError`: If text empty, too large, or non-UTF-8
- `ValidationError`: If metadata has reserved keys or non-JSON-serializable values
- `ChunkingError`: If chunking fails (internal error)
- `EmbeddingError`: If embedding generation fails (Ollama unavailable)
- `ExtractionError`: If entity extraction fails (Phase 2)
- `StorageError`: If FalkorDB storage fails

**Preconditions:**
- MemoryProcessor initialized (__init__ called)
- Dependencies available (DB connected, Ollama running)

**Postconditions:**
- If success: Memory stored in FalkorDB with chunks, embeddings, metadata
- If error: No data stored (transactional behavior)
- State unchanged (no side effects on MemoryProcessor instance)

**Algorithm Outline:**
```
1. STAGE 1: Validate Input
   - Check text non-empty (after strip)
   - Check text length <= max_text_length
   - Verify UTF-8 encoding
   - Validate metadata structure (no reserved keys, JSON-serializable)
   - Generate memory_id (UUID v4)
   - Add timestamp to metadata

2. STAGE 2: Chunk Text
   - Call chunker.chunk_text(text)
   - Get List[Chunk] with semantic boundaries
   - Log chunk count

3. STAGE 3: Generate Embeddings
   - Extract chunk texts: [chunk.text for chunk in chunks]
   - Check cache (if enabled, Phase 2):
     - For each chunk text, check if embedding cached
     - Collect cache hits and misses
   - Generate embeddings for cache misses:
     - Call embedder.embed(uncached_texts)
     - Batch processing (config.batch_size chunks per request)
   - Store embeddings in cache (if enabled)
   - Combine cached + generated embeddings

4. STAGE 4: Extract Entities (Phase 2, if enabled)
   - Call extractor.extract_entities(text, chunks)
   - Get List[Entity] and List[Relationship]
   - Log entity and relationship counts

5. STAGE 5: Store in FalkorDB
   - Begin transaction
   - Create memory node with memory_id, original text, metadata
   - Create chunk nodes (linked to memory)
   - Store embeddings (vector index)
   - Create entity and relationship nodes (if Phase 2)
   - Commit transaction
   - Log storage success

6. STAGE 6: Return memory_id
   - Return UUID string
```

**Edge Cases:**
1. **Empty text ""** → ValidationError("Text cannot be empty")
2. **Whitespace only "   "** → ValidationError("Text cannot be empty") (after strip)
3. **Single char "A"** → Success, creates 1 chunk
4. **Text = max_text_length** → Success, allowed
5. **Text = max_text_length + 1** → ValidationError("Text exceeds maximum length")
6. **Non-UTF-8 bytes** → ValidationError("Text must be valid UTF-8")
7. **metadata = {"memory_id": "x"}** → ValidationError("Reserved key: memory_id")
8. **metadata = {"tags": lambda x: x}** → ValidationError("Value not JSON-serializable")
9. **Ollama unavailable** → EmbeddingError → fallback to sentence-transformers (if configured)
10. **DB connection lost mid-transaction** → StorageError, transaction rolled back

**Related Methods:**
- Called by: AddMemoryTool.execute() (MCP tool)
- Calls: _validate_text(), _validate_metadata(), _create_memory_id(), _process_chunks(), _generate_embeddings(), _extract_entities(), _store_memory()

---

### Method 3: `search_memory`

**Signature:**
```python
async def search_memory(
    self,
    query: str,
    limit: int = 10,
    filters: Optional[Dict[str, Any]] = None,
    search_mode: str = "vector"
) -> List[SearchResult]
```

**Purpose:** Execute search across memories with ranking and filtering

**Parameters:**

- `query: str`
  - Description: Natural language search query
  - Constraints:
    - Non-empty (after strip)
    - Max length: 1000 chars
    - Format: Natural language or keywords
  - Example: "Who created Python?", "web frameworks"

- `limit: int` (default: 10)
  - Description: Maximum results to return
  - Constraints: 1 <= limit <= 100
  - Example: 5, 20, 100

- `filters: Optional[Dict[str, Any]]` (default: None)
  - Description: Metadata filters to narrow results
  - Structure:
    ```python
    {
        "tags": ["python", "web"],     # Match ANY tag
        "source": "wikipedia",          # Exact source match
        "date_from": "2025-01-01",     # >= this date (ISO)
        "date_to": "2025-12-31"        # <= this date (ISO)
    }
    ```

- `search_mode: str` (default: "vector")
  - Description: Search algorithm
  - Options: "vector" (Phase 1), "bm25" (Phase 2), "hybrid" (Phase 2), "graph" (Phase 2)

**Returns:**
- Type: `List[SearchResult]`
- Length: 0 to limit (may be fewer)
- Sorted: By similarity_score descending (highest first)
- Filtered: Only results matching filters

**Raises:**
- `ValidationError`: If query empty or limit out of range
- `ValidationError`: If search_mode invalid
- `ValidationError`: If filters have invalid structure
- `EmbeddingError`: If query embedding generation fails
- `SearchError`: If search operation fails

**Preconditions:**
- MemoryProcessor initialized
- Database has memories (otherwise returns empty list)

**Postconditions:**
- No state changes (read-only operation)
- Results cached (if caching enabled)

**Algorithm Outline:**
```
1. Validate Input
   - Check query non-empty, <= 1000 chars
   - Check limit in range [1, 100]
   - Validate search_mode in allowed modes
   - Validate filters structure (if provided)

2. Generate Query Embedding
   - Call embedder.embed([query])
   - Get query embedding vector (768-dim)

3. Execute Search (mode-dependent)
   - If "vector":
     - Call db_client.vector_search(query_embedding, limit)
     - Cosine similarity via HNSW index
   - If "bm25" (Phase 2):
     - Call db_client.bm25_search(query, limit)
     - Keyword matching
   - If "hybrid" (Phase 2):
     - Call vector_search + bm25_search in parallel
     - Merge with RRF (Reciprocal Rank Fusion)
   - If "graph" (Phase 2):
     - Call db_client.graph_search(query, limit)
     - Traverse knowledge graph

4. Apply Filters
   - For each result:
     - Check if matches filters (tags, source, date range)
     - Exclude if doesn't match
   - Keep up to limit results

5. Format Results
   - Convert DB results to SearchResult objects
   - Extract metadata (tags, source, timestamp)
   - Calculate similarity_score

6. Return Results
   - Return List[SearchResult] (sorted, filtered)
```

**Edge Cases:**
1. **Empty query ""** → ValidationError("Query cannot be empty")
2. **query = "a" * 1001** → ValidationError("Query exceeds max length")
3. **limit = 0** → ValidationError("limit must be >= 1")
4. **limit = 101** → ValidationError("limit must be <= 100")
5. **search_mode = "invalid"** → ValidationError("Invalid search_mode")
6. **filters = {"invalid_key": "x"}** → ValidationError("Unknown filter key")
7. **No memories in DB** → Return []
8. **No matches** → Return []
9. **Fewer matches than limit** → Return all matches (< limit)
10. **All results filtered out** → Return []

**Related Methods:**
- Called by: SearchMemoryTool.execute() (MCP tool)
- Calls: embedder.embed(), db_client.vector_search(), db_client.apply_filters()

---

### Method 4: `get_stats`

**Signature:**
```python
async def get_stats(self) -> Dict[str, Any]
```

**Purpose:** Aggregate system-wide statistics from all components

**Parameters:** None

**Returns:**
- Type: `Dict[str, Any]`
- Structure:
  ```python
  {
      "total_memories": 1234,               # int
      "total_chunks": 5678,                 # int
      "database_size_mb": 45.6,             # float
      "avg_chunks_per_memory": 4.6,         # float
      "cache_hit_rate": 0.63,               # float 0-1 (Phase 2, if caching)
      "cache_size_mb": 12.3,                # float (Phase 2, if caching)
      "avg_query_latency_ms": 156,          # int
      "total_entities": 987,                # int (Phase 2, if graph)
      "total_relationships": 543,           # int (Phase 2, if graph)
      "oldest_memory_date": "2025-01-01T10:00:00Z",  # ISO datetime
      "newest_memory_date": "2025-11-23T15:30:00Z"   # ISO datetime
  }
  ```

**Raises:**
- `DatabaseError`: If DB query fails
- `StatisticsError`: If statistics calculation fails

**Preconditions:**
- MemoryProcessor initialized
- Database accessible

**Postconditions:**
- No state changes (read-only operation)

**Algorithm Outline:**
```
1. Query Database
   - Get total_memories count
   - Get total_chunks count
   - Get database_size_mb (disk usage)
   - Get oldest_memory_date and newest_memory_date

2. Calculate Derived Stats
   - avg_chunks_per_memory = total_chunks / total_memories (handle division by zero)

3. Query Cache (if enabled, Phase 2)
   - Get cache_hit_rate from cache.get_stats()
   - Get cache_size_mb

4. Query Graph (if enabled, Phase 2)
   - Get total_entities count
   - Get total_relationships count

5. Query Search Latency
   - Get avg_query_latency_ms from recent queries (last 100)

6. Format and Return
   - Create dict with all stats
   - Convert timestamps to ISO format
   - Return dict
```

**Edge Cases:**
1. **No memories in DB** → total_memories = 0, avg_chunks_per_memory = 0
2. **Cache disabled** → cache_hit_rate and cache_size_mb not in result
3. **Graph not built** → total_entities and total_relationships not in result
4. **DB query fails** → DatabaseError with details
5. **No queries yet** → avg_query_latency_ms = 0

**Related Methods:**
- Called by: GetStatsTool.execute() (MCP tool)
- Calls: db_client.get_stats(), cache.get_stats() (if Phase 2), task_manager.get_stats() (if Phase 2)

---

### Method 5: `build_knowledge_graph` (Phase 2)

**Signature:**
```python
async def build_knowledge_graph(
    self,
    memory_ids: Optional[List[str]] = None,
    mode: str = "full"
) -> str
```

**Purpose:** Launch background task to build knowledge graph (entities + relationships)

**Parameters:**

- `memory_ids: Optional[List[str]]`
  - Description: Specific memories to process
  - Constraints: None (all unprocessed) or list of valid UUIDs
  - Example: None, ["uuid1", "uuid2"]

- `mode: str` (default: "full")
  - Description: What to extract
  - Options: "entities_only", "relationships_only", "full"

**Returns:**
- Type: `str`
- Format: UUID v4 task ID
- Example: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

**Raises:**
- `ValidationError`: If memory_ids contains invalid UUIDs
- `ValidationError`: If mode invalid
- `TaskError`: If task queue full
- `NotImplementedError`: In Phase 1

**Algorithm Outline:**
```
1. Validate Input
   - Check mode in ["entities_only", "relationships_only", "full"]
   - If memory_ids provided, validate all are valid UUIDs

2. Determine Memories to Process
   - If memory_ids = None:
     - Query DB for all unprocessed memories (no entities/relationships)
   - Else:
     - Use provided memory_ids

3. Create Background Task
   - Generate task_id (UUID v4)
   - Create task with:
     - task_id
     - type: "build_knowledge_graph"
     - params: {memory_ids, mode}
     - state: "pending"
     - progress: 0

4. Submit to TaskManager
   - task_manager.submit(task_id, task_func, params)
   - task_func: async function that processes memories

5. Return task_id
   - Return immediately (non-blocking)
```

**Edge Cases:**
1. **memory_ids = []** → No-op, return task_id immediately (completes instantly)
2. **memory_ids = ["invalid"]** → ValidationError("Invalid UUID")
3. **mode = "invalid"** → ValidationError("Invalid mode")
4. **Task queue full** → TaskError("Task queue full, max 10 concurrent tasks")
5. **extractor = None** → NotImplementedError("Entity extraction not enabled")

**Related Methods:**
- Called by: MCP tool (if implemented), user code
- Calls: task_manager.submit(), extractor.extract_entities(), extractor.extract_relationships()

---

## Error Handling

### Exceptions Defined

```python
# From zapomni_core.exceptions

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

class StorageError(ZapomniCoreError):
    """FalkorDB storage operation failed."""
    pass

class SearchError(ZapomniCoreError):
    """Search operation failed."""
    pass

class TaskError(ZapomniCoreError):
    """Background task operation failed."""
    pass

class StatisticsError(ZapomniCoreError):
    """Statistics calculation failed."""
    pass
```

### Error Recovery

**Validation Errors (fail fast, no recovery):**
- Empty text → Raise ValidationError immediately
- Invalid metadata → Raise ValidationError immediately
- Propagate to caller (MCP tool formats as MCP error)

**Transient Errors (retry):**
- Ollama API timeout → Retry 3x with exponential backoff (embedder handles this)
- FalkorDB connection error → Retry 3x (db_client handles this)
- Fallback: If Ollama unavailable → use sentence-transformers (embedder fallback)

**Permanent Errors (fail fast):**
- Chunking error → Raise ChunkingError (no retry, likely bug)
- Entity extraction error → Log warning, continue without entities (Phase 2)
- Storage error (transaction failed) → Rollback, raise StorageError

**Transaction Safety:**
- add_memory() uses DB transactions: all-or-nothing storage
- If any stage fails, rollback DB transaction (no partial data)

### Error Propagation

```
MemoryProcessor.add_memory()
    ↓ Stage 1: Validate → ValidationError
    ↓ Stage 2: Chunk → ChunkingError
    ↓ Stage 3: Embed → EmbeddingError (with fallback)
    ↓ Stage 4: Extract → ExtractionError (continue without)
    ↓ Stage 5: Store → StorageError (rollback)
    ↓ All errors logged with context
AddMemoryTool.execute()
    ↓ Catches exceptions, formats as MCP error
MCP Client (Claude Desktop)
    ↓ Displays error to user
```

---

## Usage Examples

### Basic Usage (Phase 1)

```python
from zapomni_core import MemoryProcessor
from zapomni_core.chunking import SemanticChunker
from zapomni_core.embeddings import OllamaEmbedder
from zapomni_db import FalkorDBClient

# Initialize dependencies
db = FalkorDBClient(host="localhost", port=6379)
chunker = SemanticChunker(chunk_size=512, chunk_overlap=50)
embedder = OllamaEmbedder(
    host="http://localhost:11434",
    model="nomic-embed-text"
)

# Create processor
processor = MemoryProcessor(
    db_client=db,
    chunker=chunker,
    embedder=embedder
)

# Add memory
memory_id = await processor.add_memory(
    text="Python is a high-level programming language created by Guido van Rossum.",
    metadata={"tags": ["python", "programming"], "source": "user"}
)
print(f"Stored: {memory_id}")

# Search memories
results = await processor.search_memory(
    query="Who created Python?",
    limit=5
)
for result in results:
    print(f"{result.similarity_score:.2f}: {result.text}")

# Get statistics
stats = await processor.get_stats()
print(f"Total memories: {stats['total_memories']}")
print(f"Database size: {stats['database_size_mb']:.2f} MB")
```

### Advanced Usage (Phase 2 with all features)

```python
from zapomni_core import MemoryProcessor, ProcessorConfig
from zapomni_core.chunking import SemanticChunker
from zapomni_core.embeddings import OllamaEmbedder
from zapomni_core.extraction import EntityExtractor
from zapomni_core.cache import SemanticCache
from zapomni_core.tasks import TaskManager
from zapomni_db import FalkorDBClient, RedisClient

# Initialize all dependencies
db = FalkorDBClient(host="localhost", port=6379)
chunker = SemanticChunker(chunk_size=512, chunk_overlap=50)
embedder = OllamaEmbedder(
    host="http://localhost:11434",
    model="nomic-embed-text"
)
extractor = EntityExtractor(
    spacy_model="en_core_web_sm",
    ollama_host="http://localhost:11434"
)
cache = SemanticCache(redis_client=RedisClient())
task_manager = TaskManager()

# Create processor with all features enabled
processor = MemoryProcessor(
    db_client=db,
    chunker=chunker,
    embedder=embedder,
    extractor=extractor,
    cache=cache,
    task_manager=task_manager,
    config=ProcessorConfig(
        enable_cache=True,
        enable_extraction=True,
        enable_graph=True,
        search_mode="hybrid"
    )
)

# Add memory (with entity extraction)
memory_id = await processor.add_memory(
    text="Django was created by Adrian Holovaty and Simon Willison in 2005 as a web framework for the Python programming language.",
    metadata={"tags": ["django", "python", "web"], "source": "wikipedia"}
)

# Build knowledge graph (background task)
task_id = await processor.build_knowledge_graph(mode="full")
print(f"Started graph build: {task_id}")

# Check progress
while True:
    status = await task_manager.get_status(task_id)
    if status['state'] == 'completed':
        print(f"Graph built! {status['items_processed']} memories processed")
        break
    print(f"Progress: {status['progress']}%")
    await asyncio.sleep(1)

# Hybrid search (vector + BM25)
results = await processor.search_memory(
    query="Django web framework",
    limit=10,
    search_mode="hybrid"
)

# Get comprehensive stats (with cache and graph)
stats = await processor.get_stats()
print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")
print(f"Entities: {stats['total_entities']}")
print(f"Relationships: {stats['total_relationships']}")
```

### Error Handling Example

```python
from zapomni_core import MemoryProcessor
from zapomni_core.exceptions import (
    ValidationError,
    EmbeddingError,
    StorageError
)

processor = MemoryProcessor(...)

# Validation error
try:
    memory_id = await processor.add_memory("")
except ValidationError as e:
    print(f"Validation failed: {e}")
    # Handle: Ask user for valid input

# Embedding error (Ollama unavailable)
try:
    memory_id = await processor.add_memory("Valid text")
except EmbeddingError as e:
    print(f"Embedding failed: {e}")
    # Handle: Check Ollama service, suggest model download

# Storage error (DB connection lost)
try:
    memory_id = await processor.add_memory("Valid text")
except StorageError as e:
    print(f"Storage failed: {e}")
    # Handle: Retry later, check DB connection

# Search error
try:
    results = await processor.search_memory("", limit=5)
except ValidationError as e:
    print(f"Invalid query: {e}")

# Comprehensive error handling
try:
    memory_id = await processor.add_memory(
        text="Python is great",
        metadata={"tags": ["python"]}
    )
    print(f"Success: {memory_id}")
except ValidationError as e:
    print(f"Input error: {e}")
except EmbeddingError as e:
    print(f"Embedding error: {e} - Check Ollama")
except StorageError as e:
    print(f"Storage error: {e} - Check database")
except Exception as e:
    print(f"Unexpected error: {e}")
    # Log and report
```

---

## Testing Approach

### Unit Tests Required

**Initialization:**
1. `test_init_minimal()` - Minimal config (Phase 1 dependencies only)
2. `test_init_full()` - Full config (all dependencies, Phase 2)
3. `test_init_missing_required_raises()` - Missing db_client raises ValueError
4. `test_init_invalid_config_raises()` - Invalid config raises ValueError

**add_memory() - Happy Path:**
5. `test_add_memory_success()` - Normal text added successfully
6. `test_add_memory_with_metadata()` - Metadata stored correctly
7. `test_add_memory_returns_uuid()` - Returns valid UUID v4
8. `test_add_memory_chunks_created()` - Chunks created and stored
9. `test_add_memory_embeddings_generated()` - Embeddings generated

**add_memory() - Validation:**
10. `test_add_memory_empty_raises()` - Empty text raises ValidationError
11. `test_add_memory_too_large_raises()` - Text > max_text_length raises ValidationError
12. `test_add_memory_non_utf8_raises()` - Non-UTF-8 raises ValidationError
13. `test_add_memory_reserved_key_raises()` - Reserved metadata key raises ValidationError
14. `test_add_memory_non_serializable_raises()` - Non-JSON-serializable metadata raises ValidationError

**add_memory() - Pipeline Stages:**
15. `test_add_memory_calls_chunker()` - Chunker called correctly
16. `test_add_memory_calls_embedder()` - Embedder called correctly
17. `test_add_memory_calls_db_store()` - DB storage called correctly

**search_memory() - Happy Path:**
18. `test_search_memory_success()` - Normal search returns results
19. `test_search_memory_with_filters()` - Filters applied correctly
20. `test_search_memory_sorted()` - Results sorted by similarity
21. `test_search_memory_limit_enforced()` - Limit enforced

**search_memory() - Validation:**
22. `test_search_memory_empty_query_raises()` - Empty query raises ValidationError
23. `test_search_memory_invalid_limit_raises()` - Limit out of range raises ValidationError
24. `test_search_memory_invalid_mode_raises()` - Invalid search_mode raises ValidationError

**get_stats():**
25. `test_get_stats_success()` - Returns valid stats dict
26. `test_get_stats_no_memories()` - Handles empty DB (total_memories=0)
27. `test_get_stats_with_cache()` - Includes cache stats (Phase 2)
28. `test_get_stats_with_graph()` - Includes graph stats (Phase 2)

**build_knowledge_graph() (Phase 2):**
29. `test_build_graph_returns_task_id()` - Returns valid UUID
30. `test_build_graph_invalid_mode_raises()` - Invalid mode raises ValidationError
31. `test_build_graph_not_implemented_phase1()` - Raises NotImplementedError in Phase 1

### Mocking Strategy

**Mock Dependencies:**
- Mock `FalkorDBClient` (return fake data, simulate errors)
- Mock `SemanticChunker` (return predefined chunks)
- Mock `OllamaEmbedder` (return fake embeddings)
- Mock `EntityExtractor` (return fake entities, Phase 2)
- Mock `SemanticCache` (simulate cache hits/misses, Phase 2)
- Mock `TaskManager` (return task_id, Phase 2)

**Use Fixtures:**
- Sample text (short, medium, large)
- Sample metadata (valid, invalid, edge cases)
- Sample chunks (from chunker)
- Sample embeddings (768-dim vectors)

### Integration Tests

**With Real Dependencies:**
- Test add_memory() → search_memory() flow with real DB
- Test embedding generation with real Ollama
- Test chunking with real LangChain
- Verify end-to-end pipeline works

**Test Scenarios:**
1. Add memory → search → verify found
2. Add multiple memories → search → verify ranking
3. Add with metadata → search with filters → verify filtered
4. Ollama unavailable → fallback to sentence-transformers

### Performance Tests

**Benchmarks:**
- `test_add_memory_performance_small()` - < 1KB in < 100ms
- `test_add_memory_performance_large()` - < 100KB in < 500ms
- `test_search_memory_performance()` - P95 < 500ms

---

## Performance Considerations

### Time Complexity

**add_memory():**
- Validation: O(1)
- Chunking: O(n) where n = len(text)
- Embedding: O(k * e) where k = num chunks, e = embedding time (~50ms per batch)
- Extraction: O(n) (Phase 2, SpaCy + LLM)
- Storage: O(k) for chunks
- **Total: O(n) + O(k * e) ≈ O(n) for text, O(k) for embeddings**

**search_memory():**
- Query embedding: O(e) (~50ms)
- Vector search: O(log n) (HNSW index)
- Filter application: O(m) where m = num results
- **Total: O(e) + O(log n) + O(m) ≈ O(log n)**

**get_stats():**
- DB queries: O(1) (aggregates pre-computed)
- **Total: O(1)**

### Space Complexity

**add_memory():**
- Text: n bytes
- Chunks: O(k * avg_chunk_size) ≈ O(n)
- Embeddings: O(k * 768 * 4 bytes) ≈ O(k)
- **Total: O(n) for text + O(k) for embeddings**

**search_memory():**
- Query embedding: 768 * 4 bytes = 3KB
- Results: O(limit * avg_result_size)
- **Total: O(limit)**

### Optimization Opportunities

**Phase 1:**
- Already optimized (batch embedding, efficient chunking)

**Phase 2:**
- **Semantic cache:** 60%+ hit rate → 3x faster embedding (cache.get() vs. Ollama API)
- **Parallel processing:** Process multiple add_memory() calls in parallel (async)

**Phase 3+:**
1. **Incremental processing:** Update existing memories instead of full reindex
2. **Adaptive batching:** Dynamically adjust batch size based on Ollama load
3. **Pre-fetching:** Pre-fetch likely chunks for faster search

---

## Performance Requirements

**Latency Targets:**

| Operation | Input Size | P50 | P95 | P99 | Max |
|-----------|-----------|-----|-----|-----|-----|
| add_memory | < 1KB | 50ms | 100ms | 200ms | 500ms |
| add_memory | < 10KB | 150ms | 300ms | 600ms | 1000ms |
| add_memory | < 100KB | 300ms | 500ms | 800ms | 1000ms |
| search_memory | any | 100ms | 200ms | 500ms | 1000ms |
| get_stats | any | 10ms | 20ms | 50ms | 100ms |
| build_graph (per doc) | any | 300ms | 600ms | 1000ms | 2000ms |

**Throughput:**
- Concurrent add_memory: 10 ops/sec (single process)
- Concurrent search_memory: 20 ops/sec
- build_knowledge_graph: 1-2 docs/sec (background)

**Resource Usage:**
- Memory: < 4GB RAM for 10K documents indexed
- CPU: Embedding generation is CPU-bound (use batching)
- Disk: Minimal (data in FalkorDB, not in memory)

---

## References

**Module Spec:**
- [zapomni_core_module.md](../level1/zapomni_core_module.md) - Parent module specification

**Related Components:**
- [semantic_chunker_component.md](./semantic_chunker_component.md) - Text chunking
- OllamaEmbedder (next component spec to create)
- EntityExtractor (Phase 2)
- FalkorDBClient (from zapomni_db module)

**External Documentation:**
- Dependency Injection Pattern: https://en.wikipedia.org/wiki/Dependency_injection
- Python asyncio: https://docs.python.org/3/library/asyncio.html
- UUID v4: https://docs.python.org/3/library/uuid.html

---

**Document Status:** Draft v1.0
**Created:** 2025-11-23
**Author:** Goncharenko Anton aka alienxs2
**Copyright:** Copyright (c) 2025 Goncharenko Anton aka alienxs2
**License:** MIT License

**Ready for Review:** Yes ✅
**Next Steps:**
1. Review and approve this component spec
2. Create function-level specs for each public method (Level 3)
3. Implement MemoryProcessor class
4. Write tests (31 unit tests defined above)
5. Integration tests with real dependencies
