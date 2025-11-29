"""
MemoryProcessor - Main orchestrator for memory processing operations.

Coordinates the complete pipeline: chunking → embedding → extraction → storage.
Provides high-level API for adding memories, searching, and retrieving statistics.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import structlog

from zapomni_core.chunking import SemanticChunker
from zapomni_core.embeddings.ollama_embedder import OllamaEmbedder
from zapomni_core.exceptions import (
    DatabaseError,
    EmbeddingError,
    ExtractionError,
    ProcessingError,
    SearchError,
    ValidationError,
)
from zapomni_core.graph.graph_builder import GraphBuilder
from zapomni_db import FalkorDBClient
from zapomni_db.models import DEFAULT_WORKSPACE_ID, Chunk, Memory

logger = structlog.get_logger(__name__)


@dataclass
class ProcessorConfig:
    """
    Configuration for MemoryProcessor.

    Attributes:
        enable_cache: Enable semantic embedding cache (Phase 2)
        enable_extraction: Enable entity extraction (Phase 2)
        enable_graph: Enable knowledge graph construction (Phase 2)
        enable_llm_refinement: Enable LLM-based entity refinement (Phase 2)
        max_text_length: Maximum text length in characters (default: 10MB)
        batch_size: Batch size for embedding generation (default: 32)
        search_mode: Default search mode ("vector", "bm25", "hybrid")
        llm_model: Ollama LLM model for entity refinement (default: qwen2.5:latest)
    """

    enable_cache: bool = True  # Enabled by default for performance
    enable_extraction: bool = True  # Phase 2: Entity extraction enabled
    enable_graph: bool = True  # Phase 2: Graph building enabled
    enable_llm_refinement: bool = False  # Phase 2: LLM refinement (off by default)
    max_text_length: int = 10_000_000
    batch_size: int = 32
    search_mode: str = "vector"
    llm_model: str = "qwen2.5:latest"  # Ollama LLM model for refinement


@dataclass
class SearchResultItem:
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
        from zapomni_core.memory_processor import MemoryProcessor
        from zapomni_core.chunking import SemanticChunker
        from zapomni_core.embeddings import OllamaEmbedder
        from zapomni_db import FalkorDBClient

        # Initialize dependencies
        db = FalkorDBClient(host="localhost", port=6381)
        chunker = SemanticChunker(chunk_size=512, chunk_overlap=50)
        embedder = OllamaEmbedder(
            base_url="http://localhost:11434",
            model_name="nomic-embed-text"
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
        extractor: Optional[Any] = None,
        cache: Optional[Any] = None,
        task_manager: Optional[Any] = None,
        config: Optional[ProcessorConfig] = None,
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
        # Validate required dependencies
        if db_client is None:
            raise ValueError("db_client is required")
        if chunker is None:
            raise ValueError("chunker is required")
        if embedder is None:
            raise ValueError("embedder is required")

        # Store dependencies
        self.db_client = db_client
        self.chunker = chunker
        self.embedder = embedder
        self.cache = cache
        self.task_manager = task_manager

        # Lazy-loaded components (Phase 2)
        # These are loaded on first access to speed up MCP server startup
        self._extractor = extractor  # May be None, loaded lazily
        self._graph_builder: Optional[GraphBuilder] = None
        self._spacy_model = None  # Cached SpaCy model
        self._llm_client = None  # Cached LLM client for entity refinement

        # Handle config
        self.config = config or ProcessorConfig()

        # Validate config
        if self.config.max_text_length <= 0:
            raise ValueError("max_text_length must be positive")
        if self.config.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.config.search_mode not in ["vector", "bm25", "hybrid", "graph"]:
            raise ValueError("Invalid search_mode")

        # Initialize logger with context
        self.logger = structlog.get_logger(__name__).bind(
            processor="memory_processor",
            config=asdict_safe(self.config),
        )

        self.logger.info(
            "memory_processor_initialized",
            max_text_length=self.config.max_text_length,
            batch_size=self.config.batch_size,
            search_mode=self.config.search_mode,
            cache_enabled=self.config.enable_cache,
            extraction_enabled=self.config.enable_extraction,
            graph_enabled=self.config.enable_graph,
        )

    def _load_spacy_model(self):
        """
        Load SpaCy model lazily on first use.

        This is called internally when extractor or graph_builder is first accessed.
        Loading SpaCy takes ~2-3 seconds, so we defer it until actually needed.
        """
        if self._spacy_model is None:
            self.logger.info("lazy_loading_spacy_model", model="en_core_web_sm")
            import spacy

            self._spacy_model = spacy.load("en_core_web_sm")
            self.logger.info("spacy_model_loaded")
        return self._spacy_model

    @property
    def llm_client(self):
        """
        Lazy-loaded OllamaLLMClient for entity refinement.

        Only loaded if enable_llm_refinement is True.
        """
        if self._llm_client is None and self.config.enable_llm_refinement:
            self.logger.info(
                "lazy_loading_llm_client",
                model=self.config.llm_model,
            )
            from zapomni_core.llm import OllamaLLMClient

            self._llm_client = OllamaLLMClient(
                base_url=self.embedder.base_url,  # Use same Ollama instance
                model_name=self.config.llm_model,
                timeout=120,
                temperature=0.1,
            )
            self.logger.info("llm_client_loaded")
        return self._llm_client

    @property
    def extractor(self):
        """
        Lazy-loaded EntityExtractor.

        SpaCy model is loaded on first access (~2-3 seconds).
        If LLM refinement is enabled, OllamaLLMClient is also attached.
        Subsequent accesses return cached instance.
        """
        if self._extractor is None:
            self.logger.info("lazy_loading_entity_extractor")
            from zapomni_core.extractors.entity_extractor import EntityExtractor

            spacy_model = self._load_spacy_model()

            # Create extractor with optional LLM client
            llm = self.llm_client if self.config.enable_llm_refinement else None
            self._extractor = EntityExtractor(
                spacy_model=spacy_model,
                ollama_client=llm,
                enable_llm_refinement=self.config.enable_llm_refinement,
            )
            self.logger.info(
                "entity_extractor_loaded",
                llm_refinement_enabled=self.config.enable_llm_refinement,
            )
        return self._extractor

    @property
    def graph_builder(self) -> Optional[GraphBuilder]:
        """
        Lazy-loaded GraphBuilder.

        Requires EntityExtractor, which requires SpaCy model.
        All are loaded on first access if not already initialized.
        """
        if self._graph_builder is None:
            self.logger.info("lazy_loading_graph_builder")
            # This will trigger lazy loading of extractor if needed
            extractor = self.extractor
            self._graph_builder = GraphBuilder(
                entity_extractor=extractor,
                db_client=self.db_client,
            )
            self.logger.info("graph_builder_loaded")
        return self._graph_builder

    async def add_memory(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        workspace_id: Optional[str] = None,
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

        Args:
            text: Text content to remember
            metadata: Optional metadata dict
            workspace_id: Workspace ID for data isolation. Defaults to "default".

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
            DatabaseError: If FalkorDB storage fails (connection error, transaction failed)

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
        correlation_id = str(uuid.uuid4())
        # Determine effective workspace_id
        effective_workspace_id = workspace_id or DEFAULT_WORKSPACE_ID
        log = self.logger.bind(
            correlation_id=correlation_id,
            operation="add_memory",
            workspace_id=effective_workspace_id,
        )

        try:
            if not isinstance(text, str):
                raise ValidationError(
                    message="Text must be a string",
                    error_code="VAL_002",
                    details={"text_type": str(type(text))},
                )

            log.info(
                "add_memory_started",
                text_length=len(text) if text else 0,
                workspace_id=effective_workspace_id,
            )

            # STAGE 1: Validate Input
            log.debug("validating_input")
            self._validate_text(text)
            self._validate_metadata(metadata)

            memory_id = self._create_memory_id()
            timestamp = datetime.now(timezone.utc)
            log.debug("input_validated", memory_id=memory_id)

            # Prepare metadata with system fields
            final_metadata = metadata.copy() if metadata else {}
            final_metadata["timestamp"] = timestamp.isoformat()

            # STAGE 2: Chunk Text
            log.debug("processing_chunks")
            chunks = await self._process_chunks(text)
            log.info("chunks_created", chunk_count=len(chunks))

            # STAGE 3: Generate Embeddings
            log.debug("generating_embeddings")
            embeddings = await self._generate_embeddings(chunks)
            log.info("embeddings_generated", embedding_count=len(embeddings))

            # STAGE 4: Extract Entities (Phase 2, optional)
            entities = None
            relationships = None
            if self.config.enable_extraction and self._extractor is not None:
                try:
                    log.debug("extracting_entities")
                    entities, relationships = await self._extract_entities(text, chunks)
                    log.info(
                        "entities_extracted",
                        entity_count=len(entities) if entities else 0,
                        relationship_count=len(relationships) if relationships else 0,
                    )
                except ExtractionError as e:
                    # Log warning but continue without entities
                    log.warning(
                        "entity_extraction_failed",
                        error=str(e),
                        error_code=getattr(e, "error_code", "EXTR_001"),
                    )

            # STAGE 5: Store in FalkorDB
            log.debug("storing_memory")
            stored_id = await self._store_memory(
                memory_id=memory_id,
                chunks=chunks,
                embeddings=embeddings,
                metadata=final_metadata,
                entities=entities,
                relationships=relationships,
                workspace_id=effective_workspace_id,
            )
            log.info("memory_stored", memory_id=stored_id)

            # STAGE 6: Return memory_id
            log.info(
                "add_memory_completed",
                memory_id=stored_id,
                chunks=len(chunks),
                workspace_id=effective_workspace_id,
                success=True,
            )
            return stored_id

        except ValidationError as e:
            log.error(
                "validation_error", error=str(e), error_code=getattr(e, "error_code", "VAL_001")
            )
            raise
        except ProcessingError as e:
            log.error(
                "processing_error", error=str(e), error_code=getattr(e, "error_code", "PROC_001")
            )
            raise
        except EmbeddingError as e:
            log.error(
                "embedding_error", error=str(e), error_code=getattr(e, "error_code", "EMB_001")
            )
            raise
        except DatabaseError as e:
            log.error("database_error", error=str(e), error_code=getattr(e, "error_code", "DB_001"))
            raise
        except Exception as e:
            log.error("unexpected_error", error=str(e), error_type=type(e).__name__)
            raise DatabaseError(
                message=f"Unexpected error in add_memory: {str(e)}",
                error_code="DB_999",
                original_exception=e,
            )

    async def search_memory(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        search_mode: str = "vector",
        workspace_id: Optional[str] = None,
    ) -> List[SearchResultItem]:
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
                - Options: "vector" (Phase 1), "bm25", "hybrid", "graph" (all Phase 2)
                - Default: "vector"
                - "vector": Semantic similarity via embeddings (cosine distance)
                - "bm25": Keyword matching via BM25 algorithm (Phase 2)
                - "hybrid": RRF fusion of vector + BM25 (Phase 2)
                - "graph": Knowledge graph traversal (Phase 2)

        Returns:
            List of SearchResultItem objects, sorted by similarity_score (descending)
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
        correlation_id = str(uuid.uuid4())
        # Determine effective workspace_id
        effective_workspace_id = workspace_id or DEFAULT_WORKSPACE_ID
        log = self.logger.bind(
            correlation_id=correlation_id,
            operation="search_memory",
            workspace_id=effective_workspace_id,
        )

        try:
            log.info(
                "search_memory_started",
                query_length=len(query),
                limit=limit,
                search_mode=search_mode,
                workspace_id=effective_workspace_id,
            )

            # Validate input
            log.debug("validating_search_input")
            self._validate_search_query(query)
            self._validate_search_limit(limit)
            self._validate_search_mode(search_mode)
            if filters:
                self._validate_search_filters(filters)
            log.debug("search_input_validated")

            # Generate query embedding (with cache if enabled)
            log.debug("generating_query_embedding")
            query_embedding = None

            # Check cache first
            if self.config.enable_cache and self.cache is not None:
                query_embedding = await self.cache.get(query)
                if query_embedding is not None:
                    log.debug("query_embedding_cache_hit")

            # Generate if not cached
            if query_embedding is None:
                query_embedding = await self.embedder.embed_text(query)
                # Store in cache for future queries
                if self.config.enable_cache and self.cache is not None:
                    await self.cache.set(query, query_embedding)

            log.debug("query_embedding_generated", embedding_dim=len(query_embedding))

            # Execute search based on mode
            log.debug("executing_search", search_mode=search_mode)
            if search_mode == "vector":
                db_results = await self.db_client.vector_search(
                    query_embedding,
                    limit=limit,
                    workspace_id=effective_workspace_id,
                )
            else:
                # Phase 2: Other search modes not yet implemented
                log.warning("search_mode_not_implemented", search_mode=search_mode)
                raise ValidationError(
                    message=f"Search mode '{search_mode}' not yet implemented",
                    error_code="VAL_004",
                )

            log.info("search_completed", result_count=len(db_results))

            # Convert to SearchResultItem objects
            results = []
            for db_result in db_results:
                # Extract metadata
                tags = []
                source = "unknown"
                if hasattr(db_result, "tags"):
                    tags = db_result.tags
                if hasattr(db_result, "source"):
                    source = db_result.source

                result = SearchResultItem(
                    memory_id=db_result.memory_id,
                    text=db_result.text,
                    similarity_score=db_result.similarity_score,
                    tags=tags,
                    source=source,
                    timestamp=db_result.timestamp,
                )
                results.append(result)

            # Apply filters if provided
            if filters:
                log.debug("applying_filters", filter_count=len(filters))
                results = self._apply_filters(results, filters)
                log.info("filters_applied", result_count=len(results))

            # Ensure we don't exceed limit
            results = results[:limit]

            log.info("search_memory_completed", result_count=len(results), success=True)
            return results

        except ValidationError as e:
            log.error(
                "validation_error", error=str(e), error_code=getattr(e, "error_code", "VAL_001")
            )
            raise
        except EmbeddingError as e:
            log.error(
                "embedding_error", error=str(e), error_code=getattr(e, "error_code", "EMB_001")
            )
            raise
        except SearchError as e:
            log.error(
                "search_error", error=str(e), error_code=getattr(e, "error_code", "SEARCH_001")
            )
            raise
        except Exception as e:
            log.error("unexpected_error", error=str(e), error_type=type(e).__name__)
            raise SearchError(
                message=f"Unexpected error in search_memory: {str(e)}",
                error_code="SEARCH_999",
                original_exception=e,
            )

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
            - total_entities: int - Total entities in knowledge graph (Phase 2)
            - total_relationships: int - Total relationships in graph (Phase 2)
            - oldest_memory_date: datetime - Timestamp of oldest memory
            - newest_memory_date: datetime - Timestamp of newest memory

        Raises:
            DatabaseError: If DB query fails
            Exception: If statistics calculation fails

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
        correlation_id = str(uuid.uuid4())
        log = self.logger.bind(correlation_id=correlation_id, operation="get_stats")

        try:
            log.info("get_stats_started")

            # Query database for base stats
            log.debug("querying_database_stats")
            db_stats = await self.db_client.get_stats()
            log.debug("database_stats_retrieved")

            # Build result dict with db stats
            stats: Dict[str, Any] = {
                "total_memories": db_stats.get("total_memories", 0),
                "total_chunks": db_stats.get("total_chunks", 0),
                "database_size_mb": db_stats.get("database_size_mb", 0.0),
                "avg_chunks_per_memory": 0.0,
                "avg_query_latency_ms": db_stats.get("avg_query_latency_ms", 0),
            }

            # Calculate derived stats
            if stats["total_memories"] > 0:
                stats["avg_chunks_per_memory"] = stats["total_chunks"] / stats["total_memories"]

            # Add date range
            if "oldest_memory_date" in db_stats:
                stats["oldest_memory_date"] = db_stats["oldest_memory_date"]
            if "newest_memory_date" in db_stats:
                stats["newest_memory_date"] = db_stats["newest_memory_date"]

            # Add cache stats if enabled (Phase 2)
            if self.config.enable_cache and self.cache is not None:
                try:
                    log.debug("querying_cache_stats")
                    cache_stats = self.cache.get_stats()
                    stats["cache_hit_rate"] = cache_stats.get("hit_rate", 0.0)
                    stats["cache_size_mb"] = cache_stats.get("size_mb", 0.0)
                    log.debug("cache_stats_retrieved")
                except Exception as e:
                    log.warning("cache_stats_failed", error=str(e))

            # Add graph stats if enabled (Phase 2)
            if self.config.enable_graph and self.db_client is not None:
                try:
                    log.debug("querying_graph_stats")
                    # This would be implemented in FalkorDBClient
                    if hasattr(self.db_client, "get_graph_stats"):
                        graph_stats = await self.db_client.get_graph_stats()
                        stats["total_entities"] = graph_stats.get("total_entities", 0)
                        stats["total_relationships"] = graph_stats.get("total_relationships", 0)
                        log.debug("graph_stats_retrieved")
                except Exception as e:
                    log.warning("graph_stats_failed", error=str(e))

            log.info("get_stats_completed", total_memories=stats["total_memories"], success=True)
            return stats

        except DatabaseError as e:
            log.error("database_error", error=str(e), error_code=getattr(e, "error_code", "DB_001"))
            raise
        except Exception as e:
            log.error("unexpected_error", error=str(e), error_type=type(e).__name__)
            raise DatabaseError(
                message=f"Unexpected error in get_stats: {str(e)}",
                error_code="DB_999",
                original_exception=e,
            )

    async def build_knowledge_graph(
        self,
        memory_ids: Optional[List[str]] = None,
        mode: str = "full",
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
            Exception: If task queue is full (max concurrent tasks exceeded)
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
        correlation_id = str(uuid.uuid4())
        log = self.logger.bind(correlation_id=correlation_id, operation="build_knowledge_graph")

        try:
            log.info(
                "build_knowledge_graph_started",
                mode=mode,
                memory_count=len(memory_ids) if memory_ids else None,
            )

            # Phase 1: Not implemented
            raise NotImplementedError(
                "build_knowledge_graph is not yet implemented (Phase 2 feature). "
                "Please upgrade to Phase 2 when available."
            )

        except NotImplementedError:
            raise
        except Exception as e:
            log.error("unexpected_error", error=str(e), error_type=type(e).__name__)
            raise NotImplementedError(f"build_knowledge_graph not implemented: {str(e)}")

    async def get_related_entities(
        self,
        entity_id: str,
        depth: int = 2,
        limit: int = 20,
    ) -> List[Any]:
        """
        Get entities related to a given entity through graph traversal (Phase 2).

        Performs breadth-first graph traversal to find connected entities.

        Args:
            entity_id: UUID of the entity to find related entities for
            depth: Maximum traversal depth in hops (1-5, default: 2)
            limit: Maximum number of related entities to return (1-50, default: 20)

        Returns:
            List of related Entity objects sorted by relationship strength

        Raises:
            ValidationError: If entity_id is invalid or depth/limit out of range
            DatabaseError: If graph traversal fails
            NotImplementedError: In Phase 1 (feature not implemented yet)

        Example:
            ```python
            processor = MemoryProcessor(...)

            # Get directly related entities (depth 1)
            related = await processor.get_related_entities(
                entity_id="550e8400-e29b-41d4-a716-446655440000",
                depth=1,
                limit=10
            )

            # Get entities 2 hops away
            related = await processor.get_related_entities(
                entity_id="550e8400-e29b-41d4-a716-446655440000",
                depth=2,
                limit=20
            )
            ```
        """
        correlation_id = str(uuid.uuid4())
        log = self.logger.bind(correlation_id=correlation_id, operation="get_related_entities")

        try:
            log.info(
                "get_related_entities_started",
                entity_id=entity_id,
                depth=depth,
                limit=limit,
            )

            results = await self.db_client.get_related_entities(
                entity_id=entity_id,
                depth=depth,
                limit=limit,
            )
            log.info("get_related_entities_completed", count=len(results))
            return results

        except Exception as e:
            log.error("unexpected_error", error=str(e), error_type=type(e).__name__)
            return []

    # Private helper methods

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
        if not text or not text.strip():
            raise ValidationError(
                message="Text cannot be empty",
                error_code="VAL_001",
                details={"text_length": len(text)},
            )

        if len(text) > self.config.max_text_length:
            raise ValidationError(
                message=f"Text exceeds maximum length ({self.config.max_text_length} chars)",
                error_code="VAL_003",
                details={"text_length": len(text), "max_length": self.config.max_text_length},
            )

        # Validate UTF-8 encoding
        try:
            text.encode("utf-8")
        except UnicodeEncodeError as e:
            raise ValidationError(
                message="Text must be valid UTF-8",
                error_code="VAL_004",
                details={"encoding_error": str(e)},
                original_exception=e,
            )

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
        if metadata is None:
            return

        if not isinstance(metadata, dict):
            raise ValidationError(
                message="Metadata must be a dictionary",
                error_code="VAL_002",
                details={"metadata_type": str(type(metadata))},
            )

        # Check reserved keys
        reserved_keys = {"memory_id", "timestamp", "chunks"}
        for key in metadata.keys():
            if key in reserved_keys:
                raise ValidationError(
                    message=f"Reserved key in metadata: {key}",
                    error_code="VAL_004",
                    details={"reserved_key": key},
                )

        # Check JSON serializability
        try:
            json.dumps(metadata)
        except (TypeError, ValueError) as e:
            raise ValidationError(
                message="Metadata values must be JSON-serializable",
                error_code="VAL_004",
                details={"serialization_error": str(e)},
                original_exception=e,
            )

    def _create_memory_id(self) -> str:
        """
        Generate unique memory ID (private helper).

        Returns:
            UUID v4 string (e.g., "550e8400-e29b-41d4-a716-446655440000")
        """
        return str(uuid.uuid4())

    async def _process_chunks(self, text: str) -> List[Chunk]:
        """
        Chunk text using SemanticChunker (private helper).

        Args:
            text: Text to chunk

        Returns:
            List of Chunk objects

        Raises:
            ProcessingError: If chunking fails
        """
        try:
            chunks = self.chunker.chunk_text(text)
            return chunks
        except Exception as e:
            self.logger.error("chunking_failed", error=str(e), error_type=type(e).__name__)
            raise ProcessingError(
                message=f"Failed to chunk text: {str(e)}",
                error_code="PROC_001",
                original_exception=e,
            )

    async def _generate_embeddings(self, chunks: List[Chunk]) -> List[List[float]]:
        """
        Generate embeddings for chunks using OllamaEmbedder with caching.

        Uses batch API for efficiency. Checks cache first (if enabled),
        generates embeddings only for cache misses, stores results in cache.

        Args:
            chunks: List of Chunk objects

        Returns:
            List of embedding vectors (one per chunk)

        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not chunks:
            return []

        chunk_texts = [chunk.text for chunk in chunks]

        try:
            # If cache is enabled, check for cached embeddings first
            if self.config.enable_cache and self.cache is not None:
                embeddings: List[List[float]] = []
                texts_to_embed: List[str] = []
                cache_indices: List[int] = []  # Track which indices need embedding

                # Check cache for each text
                for i, text in enumerate(chunk_texts):
                    cached = await self.cache.get(text)
                    if cached is not None:
                        embeddings.append(cached)
                        self.logger.debug("embedding_cache_hit", index=i)
                    else:
                        embeddings.append([])  # Placeholder
                        texts_to_embed.append(text)
                        cache_indices.append(i)

                # Generate embeddings only for cache misses
                if texts_to_embed:
                    self.logger.info(
                        "generating_embeddings_for_cache_misses",
                        total=len(chunk_texts),
                        cache_hits=len(chunk_texts) - len(texts_to_embed),
                        cache_misses=len(texts_to_embed),
                    )
                    # Use batch API for efficiency
                    new_embeddings = await self.embedder.embed_batch(
                        texts_to_embed, batch_size=self.config.batch_size
                    )

                    # Fill in the embeddings and cache them
                    for idx, (text, embedding) in enumerate(zip(texts_to_embed, new_embeddings)):
                        original_idx = cache_indices[idx]
                        embeddings[original_idx] = embedding
                        # Store in cache
                        await self.cache.set(text, embedding)

                return embeddings

            else:
                # No cache - use batch API directly
                embeddings = await self.embedder.embed_batch(
                    chunk_texts, batch_size=self.config.batch_size
                )
                return embeddings

        except EmbeddingError:
            raise
        except Exception as e:
            self.logger.error("embedding_failed", error=str(e), error_type=type(e).__name__)
            raise EmbeddingError(
                message=f"Failed to generate embeddings: {str(e)}",
                error_code="EMB_001",
                original_exception=e,
            )

    async def _extract_entities(
        self,
        text: str,
        chunks: List[Chunk],
    ) -> tuple[List[Any], List[Any]]:
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
        if self._extractor is None:
            raise ExtractionError(
                message="Entity extractor not configured",
                error_code="EXTR_001",
            )

        try:
            # Phase 2: Would implement entity extraction here
            return [], []
        except Exception as e:
            self.logger.error("extraction_failed", error=str(e), error_type=type(e).__name__)
            raise ExtractionError(
                message=f"Failed to extract entities: {str(e)}",
                error_code="EXTR_001",
                original_exception=e,
            )

    async def _store_memory(
        self,
        memory_id: str,
        chunks: List[Chunk],
        embeddings: List[List[float]],
        metadata: Dict[str, Any],
        entities: Optional[List[Any]] = None,
        relationships: Optional[List[Any]] = None,
        workspace_id: Optional[str] = None,
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
            workspace_id: Workspace ID for data isolation. Defaults to "default".

        Returns:
            memory_id: Same as input (for consistency)

        Raises:
            DatabaseError: If DB operation fails
        """
        effective_workspace_id = workspace_id or DEFAULT_WORKSPACE_ID

        try:
            # Extract original text from chunks
            text = "".join([chunk.text for chunk in chunks])

            # Create Memory model with workspace_id
            memory = Memory(
                text=text,
                chunks=chunks,
                embeddings=embeddings,
                metadata=metadata,
                workspace_id=effective_workspace_id,
            )

            # Store in database with workspace_id
            stored_id = await self.db_client.add_memory(memory, workspace_id=effective_workspace_id)
            return stored_id

        except DatabaseError:
            raise
        except Exception as e:
            self.logger.error("storage_failed", error=str(e), error_type=type(e).__name__)
            raise DatabaseError(
                message=f"Failed to store memory: {str(e)}",
                error_code="DB_001",
                original_exception=e,
            )

    def _validate_search_query(self, query: str) -> None:
        """Validate search query input."""
        if not isinstance(query, str):
            raise ValidationError(
                message="Query must be a string",
                error_code="VAL_002",
                details={"query_type": str(type(query))},
            )

        if not query or not query.strip():
            raise ValidationError(
                message="Query cannot be empty",
                error_code="VAL_001",
                details={"query_length": len(query)},
            )

        if len(query) > 1000:
            raise ValidationError(
                message="Query exceeds maximum length (1000 chars)",
                error_code="VAL_003",
                details={"query_length": len(query), "max_length": 1000},
            )

    def _validate_search_limit(self, limit: int) -> None:
        """Validate search limit."""
        if not isinstance(limit, int):
            raise ValidationError(
                message="Limit must be an integer",
                error_code="VAL_002",
                details={"limit_type": str(type(limit))},
            )

        if limit < 1 or limit > 100:
            raise ValidationError(
                message="Limit must be between 1 and 100",
                error_code="VAL_003",
                details={"limit": limit, "min": 1, "max": 100},
            )

    def _validate_search_mode(self, search_mode: str) -> None:
        """Validate search mode."""
        valid_modes = ["vector", "bm25", "hybrid", "graph"]
        if search_mode not in valid_modes:
            raise ValidationError(
                message=f"Invalid search mode '{search_mode}'",
                error_code="VAL_004",
                details={"search_mode": search_mode, "valid_modes": valid_modes},
            )

    def _validate_search_filters(self, filters: Dict[str, Any]) -> None:
        """Validate search filters."""
        valid_filter_keys = {"tags", "source", "date_from", "date_to"}
        for key in filters.keys():
            if key not in valid_filter_keys:
                raise ValidationError(
                    message=f"Unknown filter key '{key}'",
                    error_code="VAL_004",
                    details={"filter_key": key, "valid_keys": list(valid_filter_keys)},
                )

    def _parse_filter_date(self, date_str: str) -> datetime:
        """
        Parse date string from filter and normalize to UTC timezone-aware datetime.

        Handles both timezone-aware and timezone-naive ISO 8601 date strings.
        If the input is naive (no timezone), assumes UTC.

        Args:
            date_str: ISO 8601 date string (e.g., "2025-11-01", "2025-11-01T12:00:00",
                     "2025-11-01T12:00:00Z", "2025-11-01T12:00:00+00:00")

        Returns:
            datetime: UTC timezone-aware datetime

        Examples:
            - "2025-11-01" -> datetime(2025, 11, 1, 0, 0, 0, tzinfo=timezone.utc)
            - "2025-11-01T12:00:00Z" -> datetime(2025, 11, 1, 12, 0, 0, tzinfo=timezone.utc)
        """
        # Handle 'Z' suffix which fromisoformat doesn't support directly
        normalized_str = date_str.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized_str)

        # If naive (no timezone info), assume UTC
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)

        return parsed

    def _apply_filters(
        self, results: List[SearchResultItem], filters: Dict[str, Any]
    ) -> List[SearchResultItem]:
        """Apply metadata filters to search results."""
        filtered = results

        # Filter by tags (match ANY)
        if "tags" in filters and filters["tags"]:
            filter_tags = set(filters["tags"])
            filtered = [r for r in filtered if r.tags and any(tag in filter_tags for tag in r.tags)]

        # Filter by source (exact match)
        if "source" in filters:
            filter_source = filters["source"]
            filtered = [r for r in filtered if r.source == filter_source]

        # Filter by date range
        if "date_from" in filters or "date_to" in filters:
            if "date_from" in filters:
                date_from = self._parse_filter_date(filters["date_from"])
                filtered = [r for r in filtered if r.timestamp >= date_from]
            if "date_to" in filters:
                date_to = self._parse_filter_date(filters["date_to"])
                filtered = [r for r in filtered if r.timestamp <= date_to]

        return filtered


def asdict_safe(obj: Any) -> Dict[str, Any]:
    """Safely convert dataclass to dict for logging."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: getattr(obj, k) for k in obj.__dataclass_fields__}
    return {}
