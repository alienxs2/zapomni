# Cross-Module Interfaces - Module Specification

**Level:** 1 (Module)
**Author:** Goncharenko Anton aka alienxs2
**Status:** Draft
**Version:** 1.0
**Date:** 2025-11-23

---

## Overview

### Purpose

This specification defines the **interface contracts, data models, and communication patterns** that govern interactions between Zapomni's three core modules: `zapomni_mcp`, `zapomni_core`, and `zapomni_db`. It establishes the architectural boundaries, dependency directions, and integration points that ensure loose coupling, modularity, and testability.

### Scope

**Included:**
- Interface protocols and abstract classes for all cross-module communication
- Shared data models (Pydantic schemas) used across module boundaries
- Dependency injection patterns and configuration
- Error propagation and handling strategies
- Communication patterns (synchronous, asynchronous)
- Module boundary definitions and responsibilities

**Not Included:**
- Internal module implementation details (covered in individual module specs)
- MCP protocol specifics (covered in zapomni_mcp_module.md)
- Database schema details (covered in zapomni_db_module.md)
- Algorithm implementations (covered in zapomni_core_module.md)

### Position in Architecture

This specification serves as the **architectural contract** that all three modules must adhere to. It sits at the intersection of:
- MCP layer (client-facing protocol adapter)
- Core layer (business logic and processing)
- DB layer (persistence and storage)

```
┌─────────────────────────────────────────────────────────────┐
│                    MCP Client                               │
│                (Claude, Cursor, Cline)                      │
└────────────────────────┬────────────────────────────────────┘
                         │ MCP Protocol (stdio)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  zapomni_mcp                                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         MCP Tools (add_memory, search, etc.)         │  │
│  └──────────────────┬───────────────────────────────────┘  │
│                     │                                       │
│                     │ MemoryEngine Protocol                 │
│                     │ (THIS SPEC)                           │
│                     ▼                                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              zapomni_core                             │  │
│  │  ┌─────────────────────────────────────────────────┐ │  │
│  │  │  DocumentProcessor, SearchEngine, etc.          │ │  │
│  │  └────────────────┬────────────────────────────────┘ │  │
│  │                   │                                   │  │
│  │                   │ StorageProvider Protocol          │  │
│  │                   │ (THIS SPEC)                       │  │
│  │                   ▼                                   │  │
│  │  ┌─────────────────────────────────────────────────┐ │  │
│  │  │              zapomni_db                          │ │  │
│  │  │  ┌─────────────────────────────────────────┐    │ │  │
│  │  │  │  FalkorDBClient, RedisCache            │    │ │  │
│  │  │  └─────────────────────────────────────────┘    │ │  │
│  │  └─────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Architecture Principles

### 1. Dependency Direction

**Rule**: Dependencies flow **downward only** (no circular dependencies).

```
zapomni_mcp
    ↓ depends on
zapomni_core
    ↓ depends on
zapomni_db
```

- **zapomni_mcp** imports from `zapomni_core`, never vice versa
- **zapomni_core** imports from `zapomni_db`, never vice versa
- **zapomni_db** imports NO other Zapomni modules (leaf module)

**Enforcement**:
- Import checks in pre-commit hooks
- Pytest tests to verify no reverse imports
- Module graph visualization in CI

### 2. Protocol-Based Interfaces

**Rule**: Use `typing.Protocol` for interface definitions, NOT inheritance.

**Rationale**:
- **Structural subtyping** (duck typing with type checking)
- **No runtime overhead** (no ABC metaclass magic)
- **Easier testing** (no need to inherit from abstract classes)
- **Flexibility** (implementations can use any base class)

**Example**:
```python
from typing import Protocol, List

class Embedder(Protocol):
    """Protocol for embedding generators."""

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        ...
```

Any class with an `async def embed(...)` method satisfies this protocol, regardless of inheritance.

### 3. Data Transfer Objects (DTOs)

**Rule**: Use **immutable Pydantic models** for all data crossing module boundaries.

**Benefits**:
- **Validation** at boundaries
- **Serialization** for logging, debugging, caching
- **Immutability** prevents accidental mutation
- **Type safety** with IDE support

**Example**:
```python
from pydantic import BaseModel, Field

class MemoryRequest(BaseModel):
    """Data model for add_memory requests."""

    text: str = Field(..., min_length=1, max_length=10_000_000)
    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        frozen = True  # Immutable
```

### 4. Error Handling Strategy

**Rule**: Each module defines its own exception hierarchy; higher modules catch and wrap.

**Pattern**:
```python
# zapomni_db/exceptions.py
class DatabaseError(Exception):
    """Base exception for database layer."""

class ConnectionError(DatabaseError):
    """Database connection failed."""

# zapomni_core/exceptions.py
class CoreError(Exception):
    """Base exception for core layer."""

class ProcessingError(CoreError):
    """Processing operation failed."""

# zapomni_mcp/tools/add_memory.py
try:
    result = await core_engine.add_memory(...)
except CoreError as e:
    logger.error("core_error", error=str(e))
    return {"status": "error", "message": str(e)}
```

**No Exception Chaining Across Modules**: Each module logs original exception, then raises its own exception type.

---

## Interface Contracts

### 1. MCP → Core: MemoryEngine Protocol

**Purpose**: Defines how MCP tools interact with core memory processing engine.

**Interface Definition**:

```python
# File: zapomni_core/interfaces.py

from typing import Protocol, List, Dict, Any, Optional
from pydantic import BaseModel

class AddMemoryRequest(BaseModel):
    """Request to add memory."""
    text: str
    metadata: dict[str, Any] = {}

    class Config:
        frozen = True

class AddMemoryResponse(BaseModel):
    """Response from add_memory."""
    memory_id: str
    chunks_created: int
    text_preview: str  # First 100 chars

    class Config:
        frozen = True

class SearchRequest(BaseModel):
    """Request to search memories."""
    query: str
    limit: int = 10
    filters: Optional[dict[str, Any]] = None

    class Config:
        frozen = True

class SearchResult(BaseModel):
    """Single search result."""
    memory_id: str
    text: str
    similarity_score: float
    metadata: dict[str, Any]

    class Config:
        frozen = True

class SearchResponse(BaseModel):
    """Response from search."""
    results: List[SearchResult]
    count: int

    class Config:
        frozen = True

class MemoryEngine(Protocol):
    """
    Protocol for core memory processing engine.

    All MCP tools interact with the core engine through this interface.
    Implementations must provide these async methods.
    """

    async def add_memory(
        self,
        request: AddMemoryRequest
    ) -> AddMemoryResponse:
        """
        Add text to memory system.

        Args:
            request: AddMemoryRequest with text and metadata

        Returns:
            AddMemoryResponse with memory_id and details

        Raises:
            ValidationError: If request data invalid
            ProcessingError: If embedding/storage fails
        """
        ...

    async def search_memory(
        self,
        request: SearchRequest
    ) -> SearchResponse:
        """
        Search memory system for relevant information.

        Args:
            request: SearchRequest with query and filters

        Returns:
            SearchResponse with ranked results

        Raises:
            ValidationError: If request data invalid
            SearchError: If search operation fails
        """
        ...

    async def get_stats(self) -> dict[str, Any]:
        """
        Get memory system statistics.

        Returns:
            Dictionary with:
                - total_memories: int
                - total_chunks: int
                - database_size_mb: float
                - graph_name: str
        """
        ...
```

**Usage in MCP Layer**:

```python
# File: zapomni_mcp/tools/add_memory.py

from zapomni_core.interfaces import MemoryEngine, AddMemoryRequest
from zapomni_core.engine import ZapomniEngine  # Concrete implementation

async def add_memory_tool(text: str, metadata: dict) -> dict:
    """MCP tool for adding memory."""

    # Create request DTO
    request = AddMemoryRequest(text=text, metadata=metadata)

    # Get engine instance (dependency injection)
    engine: MemoryEngine = get_engine()

    # Call through protocol interface
    response = await engine.add_memory(request)

    # Convert to MCP response format
    return {
        "status": "success",
        "memory_id": response.memory_id,
        "chunks": response.chunks_created,
        "preview": response.text_preview
    }
```

**Design Decision: Why Protocol?**

**Alternatives Considered**:
- **Abstract Base Class (ABC)**: Requires inheritance, runtime overhead
- **Direct import of concrete class**: Tight coupling, hard to test
- **Duck typing**: No type safety

**Chosen: Protocol**
- Type safety without inheritance
- Easy mocking in tests
- No runtime overhead
- Follows PEP 544 (Structural Subtyping)

---

### 2. Core → DB: StorageProvider Protocol

**Purpose**: Defines how core engine interacts with database storage layer.

**Interface Definition**:

```python
# File: zapomni_db/interfaces.py

from typing import Protocol, List, Dict, Any, Optional
from pydantic import BaseModel
import datetime

class ChunkData(BaseModel):
    """Data for a single text chunk."""
    text: str
    index: int
    embedding: List[float]
    metadata: dict[str, Any] = {}

    class Config:
        frozen = True

class MemoryData(BaseModel):
    """Data for storing a memory."""
    memory_id: str
    text: str
    chunks: List[ChunkData]
    metadata: dict[str, Any]
    timestamp: datetime.datetime

    class Config:
        frozen = True

class VectorSearchRequest(BaseModel):
    """Request for vector similarity search."""
    query_embedding: List[float]
    limit: int = 10
    min_similarity: float = 0.5
    filters: Optional[dict[str, Any]] = None

    class Config:
        frozen = True

class VectorSearchResult(BaseModel):
    """Result from vector search."""
    chunk_id: str
    memory_id: str
    text: str
    similarity_score: float
    metadata: dict[str, Any]

    class Config:
        frozen = True

class EntityData(BaseModel):
    """Data for a knowledge graph entity."""
    entity_id: str
    name: str
    type: str  # PERSON, ORG, TECHNOLOGY, etc.
    description: str
    confidence: float

    class Config:
        frozen = True

class RelationshipData(BaseModel):
    """Data for entity relationship."""
    source_entity_id: str
    target_entity_id: str
    relationship_type: str  # CREATED_BY, USES, RELATED_TO, etc.
    strength: float
    evidence: str  # Text evidence for relationship

    class Config:
        frozen = True

class StorageProvider(Protocol):
    """
    Protocol for database storage operations.

    Core engine uses this interface to persist and retrieve data.
    Implementations handle specific database technology (FalkorDB, etc.).
    """

    async def store_memory(self, memory: MemoryData) -> str:
        """
        Store memory with chunks and embeddings.

        Args:
            memory: MemoryData with all information

        Returns:
            memory_id: Confirmed memory ID

        Raises:
            DatabaseError: If storage fails
            ConnectionError: If DB unavailable
        """
        ...

    async def vector_search(
        self,
        request: VectorSearchRequest
    ) -> List[VectorSearchResult]:
        """
        Perform vector similarity search.

        Args:
            request: Search parameters

        Returns:
            List of matching results, sorted by similarity

        Raises:
            DatabaseError: If search fails
        """
        ...

    async def store_entities(
        self,
        entities: List[EntityData],
        relationships: List[RelationshipData]
    ) -> int:
        """
        Store entities and relationships in graph.

        Args:
            entities: List of entities to create/update
            relationships: List of relationships to create

        Returns:
            count: Number of entities stored

        Raises:
            DatabaseError: If storage fails
        """
        ...

    async def get_stats(self) -> dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Statistics dictionary with:
                - total_memories: int
                - total_chunks: int
                - total_entities: int
                - database_size_mb: float
        """
        ...

    async def health_check(self) -> bool:
        """
        Check if database is healthy and accessible.

        Returns:
            True if healthy, False otherwise
        """
        ...
```

**Usage in Core Layer**:

```python
# File: zapomni_core/engine.py

from zapomni_db.interfaces import StorageProvider, MemoryData, ChunkData
from zapomni_db.falkordb_client import FalkorDBClient  # Concrete impl

class ZapomniEngine:
    """Core memory processing engine."""

    def __init__(self, storage: StorageProvider):
        """
        Initialize engine with storage provider.

        Args:
            storage: Any object satisfying StorageProvider protocol
        """
        self.storage = storage
        self.chunker = SemanticChunker()
        self.embedder = OllamaEmbedder()

    async def add_memory(self, request: AddMemoryRequest) -> AddMemoryResponse:
        """Add memory to system."""

        # 1. Chunk text
        chunks = self.chunker.chunk(request.text)

        # 2. Generate embeddings
        chunk_texts = [c.text for c in chunks]
        embeddings = await self.embedder.embed(chunk_texts)

        # 3. Create MemoryData DTO
        chunk_data = [
            ChunkData(
                text=chunk.text,
                index=i,
                embedding=embedding,
                metadata=chunk.metadata
            )
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]

        memory = MemoryData(
            memory_id=str(uuid.uuid4()),
            text=request.text,
            chunks=chunk_data,
            metadata=request.metadata,
            timestamp=datetime.datetime.now()
        )

        # 4. Store via protocol interface
        memory_id = await self.storage.store_memory(memory)

        # 5. Return response
        return AddMemoryResponse(
            memory_id=memory_id,
            chunks_created=len(chunks),
            text_preview=request.text[:100]
        )
```

**Design Decision: Why Separate Request/Response DTOs?**

**Rationale**:
- **Validation**: Different constraints for input vs output
- **Versioning**: Can evolve independently
- **Clarity**: Explicit about what crosses boundary
- **Testing**: Easy to create test fixtures

---

### 3. Core → Core: Internal Protocols

**Purpose**: Define interfaces for internal core components.

**Interfaces**:

```python
# File: zapomni_core/interfaces.py

from typing import Protocol, List

class TextChunker(Protocol):
    """Protocol for text chunking strategies."""

    def chunk(self, text: str) -> List[ChunkInfo]:
        """
        Split text into semantic chunks.

        Args:
            text: Input text to chunk

        Returns:
            List of ChunkInfo objects
        """
        ...

class Embedder(Protocol):
    """Protocol for embedding generation."""

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for text list.

        Args:
            texts: List of text strings

        Returns:
            List of embedding vectors (same order)

        Raises:
            EmbeddingError: If generation fails
        """
        ...

class EntityExtractor(Protocol):
    """Protocol for entity extraction."""

    async def extract_entities(
        self,
        text: str
    ) -> List[EntityData]:
        """
        Extract named entities from text.

        Args:
            text: Input text

        Returns:
            List of EntityData objects
        """
        ...

class RelationshipExtractor(Protocol):
    """Protocol for relationship extraction."""

    async def extract_relationships(
        self,
        text: str,
        entities: List[EntityData]
    ) -> List[RelationshipData]:
        """
        Extract relationships between entities.

        Args:
            text: Source text
            entities: Previously extracted entities

        Returns:
            List of RelationshipData objects
        """
        ...
```

**Design Decision: Protocols for Plugins**

**Benefit**: Easy to swap implementations:
- `SemanticChunker` (LangChain) → `ASTChunker` (code-specific)
- `OllamaEmbedder` → `SentenceTransformerEmbedder` (fallback)
- `HybridEntityExtractor` (SpaCy + LLM) → `LLMOnlyExtractor`

---

## Data Models (Shared)

### Shared Data Package

**Location**: `zapomni_db/models.py` (shared by all modules)

**Rationale**: Database layer is leaf module, so shared models there prevent circular imports.

**Models**:

```python
# File: zapomni_db/models.py

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

# === Memory Models ===

class Chunk(BaseModel):
    """Information about a text chunk.

    This is the canonical Chunk model used across all modules.
    Defined in zapomni_db.models and imported by zapomni_core and zapomni_mcp.
    """
    text: str
    index: int
    start_char: int
    end_char: int
    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        frozen = True

class ChunkData(BaseModel):
    """Chunk with embedding (used for storage operations)."""
    text: str
    index: int
    start_char: int
    end_char: int
    embedding: List[float]
    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        frozen = True

class MemoryData(BaseModel):
    """Complete memory data."""
    memory_id: str
    text: str
    chunks: List[ChunkData]
    metadata: dict[str, Any]
    timestamp: datetime

    class Config:
        frozen = True

# === Search Models ===

class VectorSearchResult(BaseModel):
    """Result from vector search."""
    chunk_id: str
    memory_id: str
    text: str
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    metadata: dict[str, Any]

    class Config:
        frozen = True

class BM25SearchResult(BaseModel):
    """Result from BM25 keyword search."""
    chunk_id: str
    memory_id: str
    text: str
    bm25_score: float
    metadata: dict[str, Any]

    class Config:
        frozen = True

class HybridSearchResult(BaseModel):
    """Fused result from hybrid search."""
    chunk_id: str
    memory_id: str
    text: str
    similarity_score: float
    bm25_score: float
    fused_score: float
    metadata: dict[str, Any]

    class Config:
        frozen = True

# === Knowledge Graph Models ===

class EntityData(BaseModel):
    """Knowledge graph entity."""
    entity_id: str
    name: str
    type: str  # PERSON, ORG, TECHNOLOGY, CONCEPT, etc.
    description: str = ""
    confidence: float = Field(..., ge=0.0, le=1.0)
    source: str = "unknown"  # spacy, llm, manual

    class Config:
        frozen = True

class RelationshipData(BaseModel):
    """Entity relationship."""
    source_entity_id: str
    target_entity_id: str
    relationship_type: str  # CREATED_BY, USES, RELATED_TO, etc.
    strength: float = Field(..., ge=0.0, le=1.0)
    evidence: str = ""

    class Config:
        frozen = True

# === Statistics Models ===

class MemoryStats(BaseModel):
    """Memory system statistics."""
    total_memories: int
    total_chunks: int
    total_entities: int
    database_size_mb: float
    graph_name: str
    cache_hit_rate: Optional[float] = None
    avg_query_latency_ms: Optional[float] = None

    class Config:
        frozen = True

# === Task Models ===

class TaskStatus(BaseModel):
    """Background task status."""
    task_id: str
    status: str  # pending, running, completed, failed
    progress: float = Field(..., ge=0.0, le=1.0)
    message: str = ""
    error: Optional[str] = None

    class Config:
        frozen = True
```

**Import Pattern**:

```python
# In zapomni_mcp:
from zapomni_db.models import MemoryData, SearchResult

# In zapomni_core:
from zapomni_db.models import ChunkData, EntityData, RelationshipData

# In zapomni_db:
from .models import *  # Local import
```

**Design Decision: Why Pydantic?**

**Benefits**:
- **Validation**: Automatic type checking and constraint validation
- **Serialization**: Easy JSON serialization for logging, caching
- **Immutability**: `frozen = True` prevents accidental mutation
- **IDE Support**: Excellent autocomplete and type hints
- **Documentation**: Self-documenting with Field descriptions

**Alternatives Considered**:
- **dataclasses**: Less validation, no JSON schema generation
- **attrs**: Similar but less ecosystem integration
- **Plain dicts**: No type safety, error-prone

---

## Communication Patterns

### Pattern 1: Synchronous Request-Response

**When to Use**: Simple, fast operations (< 500ms)

**Example**: `add_memory`, `search_memory`, `get_stats`

**Flow**:
```
MCP Tool
    ↓ (async call)
Core Engine
    ↓ (async call)
DB Client
    ↓ (await result)
Core Engine
    ↓ (return response)
MCP Tool
```

**Code**:
```python
# Synchronous-looking async code
response = await engine.add_memory(request)
return format_mcp_response(response)
```

---

### Pattern 2: Asynchronous Task Queue

**When to Use**: Long-running operations (> 1 second)

**Example**: `build_graph`, `index_codebase`

**Flow**:
```
MCP Tool
    ↓ (submit task)
Core Engine (creates task)
    ↓ (return task_id immediately)
MCP Tool (returns task_id)

Meanwhile:
Background Worker
    ↓ (execute task)
Core Engine
    ↓ (update progress)
DB Client (store results)
```

**Code**:
```python
# In MCP tool:
task_id = await engine.build_graph_async(memory_ids)
return {"task_id": task_id, "status": "pending"}

# User can query status later:
status = await engine.get_task_status(task_id)
# Returns: {"status": "running", "progress": 0.65, ...}
```

**Implementation**:
```python
# File: zapomni_core/tasks.py

import asyncio
from typing import Callable, Any
from zapomni_db.models import TaskStatus

class TaskManager:
    """Manages background async tasks."""

    def __init__(self):
        self.tasks: dict[str, TaskStatus] = {}

    async def submit_task(
        self,
        task_func: Callable,
        *args,
        **kwargs
    ) -> str:
        """
        Submit task for background execution.

        Args:
            task_func: Async function to execute
            *args, **kwargs: Arguments for task_func

        Returns:
            task_id: Unique task identifier
        """
        task_id = str(uuid.uuid4())

        # Create task status
        self.tasks[task_id] = TaskStatus(
            task_id=task_id,
            status="pending",
            progress=0.0,
            message="Task queued"
        )

        # Start background execution
        asyncio.create_task(self._execute_task(task_id, task_func, *args, **kwargs))

        return task_id

    async def _execute_task(
        self,
        task_id: str,
        task_func: Callable,
        *args,
        **kwargs
    ):
        """Execute task in background."""
        try:
            # Update to running
            self.tasks[task_id] = TaskStatus(
                task_id=task_id,
                status="running",
                progress=0.0,
                message="Task started"
            )

            # Execute task (with progress callback)
            def update_progress(progress: float, message: str = ""):
                self.tasks[task_id] = TaskStatus(
                    task_id=task_id,
                    status="running",
                    progress=progress,
                    message=message
                )

            result = await task_func(*args, progress_callback=update_progress, **kwargs)

            # Mark completed
            self.tasks[task_id] = TaskStatus(
                task_id=task_id,
                status="completed",
                progress=1.0,
                message="Task completed successfully"
            )

        except Exception as e:
            # Mark failed
            self.tasks[task_id] = TaskStatus(
                task_id=task_id,
                status="failed",
                progress=self.tasks[task_id].progress,
                message="Task failed",
                error=str(e)
            )

    def get_task_status(self, task_id: str) -> TaskStatus:
        """Get current task status."""
        return self.tasks.get(task_id)
```

---

### Pattern 3: Dependency Injection

**When to Use**: Providing dependencies to constructors

**Example**: Engine initialization

**Code**:
```python
# File: zapomni_mcp/server.py

from zapomni_core.engine import ZapomniEngine
from zapomni_db.falkordb_client import FalkorDBClient
from zapomni_core.chunking import SemanticChunker
from zapomni_core.embeddings import OllamaEmbedder

async def create_engine() -> ZapomniEngine:
    """Create configured engine instance."""

    # Create dependencies
    storage = FalkorDBClient(
        host=settings.falkordb_host,
        port=settings.falkordb_port,
        graph_name=settings.graph_name
    )
    await storage.connect()

    chunker = SemanticChunker(
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap
    )

    embedder = OllamaEmbedder(
        host=settings.ollama_host,
        model=settings.embedding_model
    )

    # Inject into engine
    engine = ZapomniEngine(
        storage=storage,
        chunker=chunker,
        embedder=embedder
    )

    return engine
```

**Benefits**:
- **Testability**: Easy to inject mocks
- **Flexibility**: Swap implementations without changing engine code
- **Configuration**: Dependencies configured externally

---

## Error Handling

### Exception Hierarchy

```python
# zapomni_db/exceptions.py
class ZapomniDBError(Exception):
    """Base exception for database layer."""

class DatabaseConnectionError(ZapomniDBError):
    """Database connection failed."""

class DatabaseQueryError(ZapomniDBError):
    """Database query execution failed."""

class DatabaseTimeoutError(ZapomniDBError):
    """Database operation timed out."""

# zapomni_core/exceptions.py
class ZapomniCoreError(Exception):
    """Base exception for core layer."""

class ValidationError(ZapomniCoreError):
    """Input validation failed."""

class ProcessingError(ZapomniCoreError):
    """Processing operation failed."""

class EmbeddingError(ZapomniCoreError):
    """Embedding generation failed."""

class SearchError(ZapomniCoreError):
    """Search operation failed."""

# zapomni_mcp/exceptions.py
class ZapomniMCPError(Exception):
    """Base exception for MCP layer."""

class ToolExecutionError(ZapomniMCPError):
    """MCP tool execution failed."""
```

### Error Propagation Strategy

**Rule**: Catch at boundary, log, wrap in layer-specific exception.

**Example**:
```python
# In zapomni_core/engine.py
from zapomni_db.exceptions import DatabaseQueryError
from zapomni_core.exceptions import ProcessingError

async def add_memory(self, request: AddMemoryRequest) -> AddMemoryResponse:
    """Add memory to system."""
    try:
        # ... processing ...
        memory_id = await self.storage.store_memory(memory)

    except DatabaseQueryError as e:
        # Log original error with full context
        logger.error(
            "database_storage_failed",
            error=str(e),
            memory_id=memory.memory_id,
            chunk_count=len(memory.chunks)
        )
        # Wrap in core-level exception
        raise ProcessingError(f"Failed to store memory: {e}") from e
```

**Benefits**:
- **Isolation**: Each layer handles its own error types
- **Logging**: Full context logged at error site
- **Abstraction**: Higher layers don't know about DB details
- **Debugging**: `from e` preserves traceback

---

## Module Boundaries

### Responsibility Matrix

| Responsibility | MCP | Core | DB |
|----------------|-----|------|----|
| MCP protocol compliance | ✅ | ❌ | ❌ |
| Input validation (MCP) | ✅ | ❌ | ❌ |
| Tool routing | ✅ | ❌ | ❌ |
| Document chunking | ❌ | ✅ | ❌ |
| Embedding generation | ❌ | ✅ | ❌ |
| Entity extraction | ❌ | ✅ | ❌ |
| Search algorithms (BM25, RRF) | ❌ | ✅ | ❌ |
| Vector similarity search | ❌ | ❌ | ✅ |
| Graph traversal | ❌ | ❌ | ✅ |
| Data persistence | ❌ | ❌ | ✅ |
| Cypher query generation | ❌ | ❌ | ✅ |
| Background task management | ❌ | ✅ | ❌ |

### What Belongs Where?

**zapomni_mcp**:
- MCP server setup (stdio transport)
- Tool definitions and schemas
- MCP request/response formatting
- Logging to stderr (MCP compliance)
- Configuration loading

**zapomni_core**:
- All business logic
- Document processing pipelines
- Embedding coordination
- Search algorithm implementations
- Entity and relationship extraction
- Background task orchestration

**zapomni_db**:
- Database client implementations
- Cypher query generation
- Vector index management
- Graph schema enforcement
- Connection pooling
- Health checks

---

## Design Decisions

### Decision 1: Protocol-Based Interfaces vs Inheritance

**Context**: Need to define interfaces between modules without tight coupling.

**Options Considered**:
1. **Abstract Base Classes (ABC)**: Traditional OOP approach
2. **Protocol (PEP 544)**: Structural subtyping
3. **Duck typing**: No formal interface

**Chosen**: Protocol (PEP 544)

**Rationale**:
- **No inheritance required**: Implementations don't need to inherit from ABC
- **Structural subtyping**: If it has the right methods, it satisfies the protocol
- **No runtime overhead**: No metaclass magic
- **Type safety**: MyPy checks protocol compliance
- **Testability**: Easy to create mock objects

**Trade-offs**:
- **Pros**: Flexibility, testability, no runtime cost
- **Cons**: Newer Python feature (3.8+), less familiar to some developers

### Decision 2: Shared Data Models in DB Package

**Context**: Need shared Pydantic models across all modules.

**Options Considered**:
1. **Separate models package**: `zapomni_models` as 4th module
2. **Models in Core**: Core is center, MCP and DB import from it
3. **Models in DB**: DB is leaf, no circular imports

**Chosen**: Models in DB (`zapomni_db/models.py`)

**Rationale**:
- **No circular imports**: DB is leaf module, never imports other modules
- **Single source of truth**: All modules import from same place
- **Simplicity**: Avoid creating 4th module just for models

**Trade-offs**:
- **Pros**: Simple, no circular imports, single source
- **Cons**: DB package has responsibility beyond just storage (minor concern)

### Decision 3: Immutable DTOs (frozen Pydantic)

**Context**: Data crossing module boundaries should be safe from mutation.

**Options Considered**:
1. **Mutable Pydantic models**: Default behavior
2. **Immutable Pydantic** (`frozen = True`)
3. **Copy on boundary**: Clone objects when passing

**Chosen**: Immutable Pydantic (`frozen = True`)

**Rationale**:
- **Safety**: Prevents accidental mutation across modules
- **Clarity**: Explicit that data is read-only
- **Performance**: No need for defensive copying
- **Functional style**: Encourages immutable data patterns

**Trade-offs**:
- **Pros**: Safe, clear intent, performant
- **Cons**: Slightly less convenient (can't modify, must create new instance)

### Decision 4: Async/Await Throughout

**Context**: I/O operations (DB, Ollama) are blocking.

**Options Considered**:
1. **Synchronous**: Simple but blocking
2. **Async/Await**: Non-blocking I/O
3. **Thread pool**: Parallel execution

**Chosen**: Async/Await

**Rationale**:
- **Non-blocking**: Multiple operations can run concurrently
- **MCP SDK**: Anthropic's MCP SDK uses async
- **Python 3.10+**: Excellent async support
- **Natural fit**: I/O-bound operations benefit most from async

**Trade-offs**:
- **Pros**: Scalable, efficient, modern Python
- **Cons**: More complex than sync code, async "color" spreads

---

## Non-Functional Requirements

### Performance

**Latency Targets**:
- add_memory: < 5 seconds per document (including embedding)
- search_memory: < 500ms (P95)
- get_stats: < 100ms

**Throughput**:
- add_memory: 100+ documents/minute
- search_memory: 10+ queries/second

**Resource Usage**:
- Memory: < 4GB RAM for 10K documents
- CPU: < 50% during normal operations

### Scalability

**Supported Scale (MVP)**:
- 10,000 documents
- 50,000 chunks
- 10,000 entities

**Bottlenecks**:
- Embedding generation: Sequential via Ollama (no batch API)
- Graph queries: Can slow down with deep traversals (limit depth to 3)

**Mitigation**:
- Semantic cache for embeddings (60-68% hit rate target)
- Parallel search (vector + BM25 concurrent)
- Connection pooling for DB

### Reliability

**Error Handling**:
- All exceptions logged with full context
- Graceful degradation (e.g., fallback embedder if Ollama fails)
- Retry logic with exponential backoff for transient errors

**Health Checks**:
- DB connection health: Ping every 30 seconds
- Ollama availability: Check before embedding
- Automatic reconnection on connection loss

**Data Integrity**:
- Pydantic validation at all boundaries
- Database transactions for multi-step operations
- No silent data loss (all errors logged and raised)

---

## Testing Strategy

### Unit Testing

**What to Test**:
- Interface contracts (protocols are satisfied)
- Data model validation (Pydantic constraints)
- Error handling (exceptions raised correctly)

**Mocking Strategy**:
- Mock StorageProvider for testing Core layer
- Mock MemoryEngine for testing MCP layer
- Use in-memory implementations for fast tests

**Example**:
```python
# Test that ZapomniEngine satisfies MemoryEngine protocol
def test_engine_satisfies_protocol():
    from zapomni_core.interfaces import MemoryEngine
    from zapomni_core.engine import ZapomniEngine

    # If this type-checks, protocol is satisfied
    engine: MemoryEngine = ZapomniEngine(
        storage=MockStorage(),
        chunker=MockChunker(),
        embedder=MockEmbedder()
    )
```

### Integration Testing

**What to Test**:
- Cross-module data flow (MCP → Core → DB)
- Real database operations (with test FalkorDB instance)
- Real Ollama calls (if available, else skip)

**Test Environment**:
- Docker Compose with test services
- Separate graph name (`test_graph`)
- Cleanup after each test

### Contract Testing

**What to Test**:
- Request/Response DTOs are compatible
- Protocol methods have correct signatures
- Error types match across modules

**Tools**:
- MyPy for type checking
- Pytest for runtime checks

---

## Future Considerations

### Potential Enhancements

1. **Streaming Responses**: For large search results
2. **Batch Operations**: Bulk add_memory for efficiency
3. **Query Planning**: Optimize complex searches
4. **Plugin System**: Custom embedders, chunkers, extractors
5. **Multi-tenancy**: Support multiple isolated graphs

### Known Limitations

1. **Sequential Embedding**: Ollama doesn't support batch API
2. **No Distributed Mode**: Single-instance only (local-first design)
3. **No Schema Migrations**: Manual graph updates if schema changes

### Evolution Path

**Phase 1 → 2**: Add knowledge graph interfaces
**Phase 2 → 3**: Add code-aware interfaces
**Phase 3 → 4**: Add multi-modal interfaces (images, tables)

---

## References

### Internal Documents
- [product.md](../../steering/product.md) - Product vision and features
- [tech.md](../../steering/tech.md) - Technology stack decisions
- [structure.md](../../steering/structure.md) - Project structure and conventions

### External Resources
- **PEP 544**: Protocol (Structural Subtyping): https://peps.python.org/pep-0544/
- **Pydantic**: Data validation library: https://docs.pydantic.dev/
- **MCP Specification**: https://spec.modelcontextprotocol.io/
- **FalkorDB Documentation**: https://docs.falkordb.com/

---

## Appendix: Complete Interface Example

### Full Example: add_memory Flow

```python
# === MCP Layer (zapomni_mcp/tools/add_memory.py) ===

from zapomni_core.interfaces import MemoryEngine, AddMemoryRequest
from zapomni_core.engine import ZapomniEngine
from zapomni_core.exceptions import ValidationError, ProcessingError

async def add_memory_tool(text: str, metadata: dict) -> dict:
    """MCP tool for adding memory."""

    try:
        # 1. Create request DTO
        request = AddMemoryRequest(text=text, metadata=metadata)

        # 2. Get engine (dependency injection)
        engine: MemoryEngine = get_engine_instance()

        # 3. Call through protocol interface
        response = await engine.add_memory(request)

        # 4. Format MCP response
        return {
            "content": [{
                "type": "text",
                "text": f"Memory stored with ID: {response.memory_id}"
            }],
            "isError": False
        }

    except ValidationError as e:
        logger.error("validation_failed", error=str(e))
        return {
            "content": [{"type": "text", "text": f"Validation error: {e}"}],
            "isError": True
        }

    except ProcessingError as e:
        logger.error("processing_failed", error=str(e))
        return {
            "content": [{"type": "text", "text": f"Processing error: {e}"}],
            "isError": True
        }

# === Core Layer (zapomni_core/engine.py) ===

from zapomni_db.interfaces import StorageProvider
from zapomni_db.models import MemoryData, ChunkData
from zapomni_core.exceptions import ProcessingError, ValidationError
from zapomni_db.exceptions import DatabaseError

class ZapomniEngine:
    """Core memory processing engine."""

    def __init__(
        self,
        storage: StorageProvider,
        chunker: TextChunker,
        embedder: Embedder
    ):
        self.storage = storage
        self.chunker = chunker
        self.embedder = embedder

    async def add_memory(
        self,
        request: AddMemoryRequest
    ) -> AddMemoryResponse:
        """Add memory to system."""

        # 1. Validate
        if not request.text.strip():
            raise ValidationError("Text cannot be empty")

        try:
            # 2. Chunk
            chunks = self.chunker.chunk(request.text)
            logger.debug("text_chunked", chunk_count=len(chunks))

            # 3. Embed
            chunk_texts = [c.text for c in chunks]
            embeddings = await self.embedder.embed(chunk_texts)
            logger.debug("embeddings_generated", count=len(embeddings))

            # 4. Create MemoryData DTO
            chunk_data = [
                ChunkData(
                    text=chunk.text,
                    index=i,
                    embedding=embedding,
                    metadata=chunk.metadata
                )
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
            ]

            memory = MemoryData(
                memory_id=str(uuid.uuid4()),
                text=request.text,
                chunks=chunk_data,
                metadata=request.metadata,
                timestamp=datetime.now()
            )

            # 5. Store via protocol
            memory_id = await self.storage.store_memory(memory)

            # 6. Return response
            return AddMemoryResponse(
                memory_id=memory_id,
                chunks_created=len(chunks),
                text_preview=request.text[:100]
            )

        except DatabaseError as e:
            logger.error("storage_failed", error=str(e))
            raise ProcessingError(f"Failed to store memory: {e}") from e

# === DB Layer (zapomni_db/falkordb_client.py) ===

from zapomni_db.models import MemoryData
from zapomni_db.exceptions import DatabaseError, DatabaseConnectionError

class FalkorDBClient:
    """FalkorDB storage implementation."""

    async def store_memory(self, memory: MemoryData) -> str:
        """Store memory in FalkorDB."""

        if not self.graph:
            raise DatabaseConnectionError("Not connected to database")

        try:
            # Create Memory node
            query = """
                CREATE (m:Memory {
                    id: $id,
                    text: $text,
                    timestamp: timestamp(),
                    chunk_count: $chunk_count
                })
                RETURN m.id
            """

            params = {
                "id": memory.memory_id,
                "text": memory.text,
                "chunk_count": len(memory.chunks)
            }

            result = self.graph.query(query, params)

            # Create Chunk nodes
            for chunk in memory.chunks:
                chunk_query = """
                    MATCH (m:Memory {id: $memory_id})
                    CREATE (c:Chunk {
                        id: $chunk_id,
                        text: $text,
                        index: $index,
                        embedding: $embedding
                    })
                    CREATE (m)-[:HAS_CHUNK]->(c)
                """

                chunk_params = {
                    "memory_id": memory.memory_id,
                    "chunk_id": str(uuid.uuid4()),
                    "text": chunk.text,
                    "index": chunk.index,
                    "embedding": chunk.embedding
                }

                self.graph.query(chunk_query, chunk_params)

            logger.info("memory_stored", memory_id=memory.memory_id)
            return memory.memory_id

        except Exception as e:
            logger.error("database_query_failed", error=str(e))
            raise DatabaseError(f"Failed to store memory: {e}") from e
```

**This example demonstrates**:
- Protocol-based interfaces (MemoryEngine, StorageProvider)
- Immutable DTOs (AddMemoryRequest, MemoryData)
- Error handling at boundaries
- Dependency injection
- Logging at each layer
- Type safety throughout

---

**Document Status**: Draft v1.0
**Date**: 2025-11-23
**Author**: Goncharenko Anton aka alienxs2
**Copyright**: Copyright (c) 2025 Goncharenko Anton aka alienxs2
**License**: MIT License

**Ready for Verification**: Yes
