# Project Structure: Zapomni

**Document Version**: 1.0
**Created**: 2025-11-22
**Authors**: Goncharenko Anton aka alienxs2 + Claude Code
**Copyright**: Copyright (c) 2025 Goncharenko Anton aka alienxs2
**License**: MIT License
**Status**: Draft
**Alignment**: Consistent with product.md and tech.md

---

## Executive Summary

### Purpose of This Document

This document provides the **definitive guide to Zapomni's codebase organization, conventions, and development workflows**. It serves as the primary reference for:

- **New developers** setting up local development environments
- **Contributors** understanding code organization and contribution guidelines
- **Maintainers** making architectural decisions and reviewing PRs
- **Future Claude agents** navigating the codebase and maintaining consistency

The goal is **actionable practicality**: every section includes concrete examples, copy-paste ready configurations, and clear rationale. This is not abstract theory—this is how we build Zapomni.

### Key Organizational Principles

1. **Monorepo Structure**: Single repository with three distinct Python packages in `/home/dev/zapomni/src/`
2. **Package Separation**: Clean separation between MCP server, core logic, and database layers
3. **Convention Over Configuration**: Sensible defaults, minimal setup required
4. **Easy Onboarding**: New developer from zero to running tests in < 30 minutes

### Project Philosophy

- **Local-First**: All code runs locally, zero cloud dependencies
- **Privacy by Design**: Data never leaves the machine
- **Performance from Day One**: Measure, optimize, maintain speed budgets
- **Developer-Friendly**: Clear errors, good docs, fast feedback loops
- **Open Source**: Transparent development, welcoming community

---

## Directory Layout

### Complete Project Structure

```
/home/dev/zapomni/                          # Project root
├── .spec-workflow/                         # Spec workflow system
│   ├── specs/                              # Feature specifications
│   ├── steering/                           # Steering documents (this file)
│   │   ├── product.md                      # Product vision ✅ APPROVED
│   │   ├── tech.md                         # Technical architecture ✅ APPROVED
│   │   └── structure.md                    # THIS DOCUMENT
│   ├── templates/                          # Spec templates
│   └── approvals/                          # Approval tracking
│
├── research/                               # Research reports (preserved)
│   ├── 00_final_synthesis.md               # Implementation roadmap
│   ├── 01_tech_stack_infrastructure.md     # Tech evaluation
│   ├── 02_mcp_solutions_architectures.md   # MCP patterns
│   └── 03_best_practices_patterns.md       # RAG best practices
│
├── src/                                    # Source code (main packages)
│   ├── zapomni_mcp/                        # MCP server package
│   │   ├── __init__.py                     # Package exports
│   │   ├── server.py                       # Main MCP server entry point
│   │   ├── tools/                          # MCP tool implementations
│   │   │   ├── __init__.py                 # Tool registry
│   │   │   ├── add_memory.py               # Phase 1: add_memory tool
│   │   │   ├── search_memory.py            # Phase 1: search_memory tool
│   │   │   ├── get_stats.py                # Phase 1: get_stats tool
│   │   │   ├── build_graph.py              # Phase 2: build_graph tool
│   │   │   ├── get_related.py              # Phase 2: get_related tool
│   │   │   ├── graph_status.py             # Phase 2: graph_status tool
│   │   │   ├── index_codebase.py           # Phase 3: index_codebase tool
│   │   │   ├── delete_memory.py            # Phase 3: delete_memory tool
│   │   │   ├── clear_all.py                # Phase 3: clear_all tool
│   │   │   └── export_graph.py             # Phase 3: export_graph tool
│   │   ├── schemas/                        # Pydantic models for validation
│   │   │   ├── __init__.py                 # Schema exports
│   │   │   ├── requests.py                 # Request schemas
│   │   │   └── responses.py                # Response schemas
│   │   ├── config.py                       # Configuration management
│   │   └── logging.py                      # Logging setup (structlog)
│   │
│   ├── zapomni_core/                       # Core processing logic
│   │   ├── __init__.py                     # Package exports
│   │   ├── processors/                     # Document processors
│   │   │   ├── __init__.py                 # Processor registry
│   │   │   ├── base.py                     # Base processor interface
│   │   │   ├── text_processor.py           # Text/markdown processing
│   │   │   ├── pdf_processor.py            # PDF processing (PyMuPDF)
│   │   │   └── code_processor.py           # Code AST processing (Phase 3)
│   │   ├── embeddings/                     # Embedding generation
│   │   │   ├── __init__.py                 # Embedder registry
│   │   │   ├── ollama_embedder.py          # Ollama client (nomic-embed-text)
│   │   │   ├── sentence_transformer.py     # Fallback embedder (all-MiniLM)
│   │   │   └── cache.py                    # Semantic cache (Phase 2)
│   │   ├── extractors/                     # Entity & relationship extraction
│   │   │   ├── __init__.py                 # Extractor registry
│   │   │   ├── entity_extractor.py         # Hybrid SpaCy + LLM (Phase 2)
│   │   │   └── relationship_extractor.py   # Relationship detection (Phase 2)
│   │   ├── search/                         # Search engines
│   │   │   ├── __init__.py                 # Search registry
│   │   │   ├── vector_search.py            # Vector similarity search (Phase 1)
│   │   │   ├── bm25_search.py              # BM25 keyword search (Phase 2)
│   │   │   ├── hybrid_search.py            # Fusion (RRF) (Phase 2)
│   │   │   ├── reranker.py                 # Cross-encoder reranking (Phase 2)
│   │   │   └── graph_search.py             # Graph traversal queries (Phase 2)
│   │   ├── chunking/                       # Document chunking
│   │   │   ├── __init__.py                 # Chunker registry
│   │   │   ├── semantic_chunker.py         # LangChain RecursiveTextSplitter
│   │   │   └── ast_chunker.py              # Code AST chunking (Phase 3)
│   │   ├── tasks/                          # Background tasks
│   │   │   ├── __init__.py                 # Task manager exports
│   │   │   ├── task_manager.py             # Async task queue (Phase 2)
│   │   │   └── status_tracker.py           # Progress tracking (Phase 2)
│   │   └── utils/                          # Utilities
│   │       ├── __init__.py                 # Utility exports
│   │       ├── text_utils.py               # Text processing utilities
│   │       └── validation.py               # Input validation helpers
│   │
│   └── zapomni_db/                         # Database integrations
│       ├── __init__.py                     # Package exports
│       ├── falkordb/                       # FalkorDB client
│       │   ├── __init__.py                 # Client exports
│       │   ├── client.py                   # Main FalkorDB client wrapper
│       │   ├── schema.py                   # Schema definitions (nodes, edges)
│       │   ├── queries.py                  # Cypher query templates
│       │   └── migrations.py               # Schema migrations (future)
│       ├── redis_cache/                    # Redis semantic cache
│       │   ├── __init__.py                 # Cache exports
│       │   └── cache_client.py             # Redis client wrapper
│       └── models.py                       # Shared data models (Pydantic)
│
├── tests/                                  # Test suite
│   ├── __init__.py                         # Test package
│   ├── unit/                               # Unit tests (70% of tests)
│   │   ├── __init__.py
│   │   ├── test_chunker.py                 # Chunking tests
│   │   ├── test_embedder.py                # Embedding tests
│   │   ├── test_search.py                  # Search algorithm tests
│   │   ├── test_entity_extractor.py        # Entity extraction tests
│   │   └── ...                             # One test file per module
│   ├── integration/                        # Integration tests (25% of tests)
│   │   ├── __init__.py
│   │   ├── test_falkordb_client.py         # FalkorDB integration
│   │   ├── test_ollama_client.py           # Ollama integration
│   │   ├── test_mcp_server.py              # MCP server integration
│   │   └── ...
│   ├── e2e/                                # End-to-end tests (5% of tests)
│   │   ├── __init__.py
│   │   └── test_full_workflow.py           # Complete add→search workflow
│   ├── fixtures/                           # Test fixtures and data
│   │   ├── sample_docs.py                  # Sample documents
│   │   ├── mock_embeddings.py              # Mock embedding data
│   │   └── test_data/                      # Static test files
│   │       ├── sample.pdf
│   │       ├── sample.md
│   │       └── sample_code.py
│   └── conftest.py                         # Pytest configuration and fixtures
│
├── docs/                                   # Documentation
│   ├── README.md                           # Main documentation entry point
│   ├── architecture/                       # Architecture docs
│   │   ├── overview.md                     # System overview
│   │   ├── mcp_protocol.md                 # MCP protocol integration
│   │   └── data_flow.md                    # Data flow diagrams
│   ├── api/                                # API reference
│   │   ├── tools.md                        # MCP tools reference
│   │   └── python_api.md                   # Python API (if exposed)
│   ├── guides/                             # How-to guides
│   │   ├── quick_start.md                  # 5-minute quickstart
│   │   ├── installation.md                 # Detailed installation
│   │   ├── configuration.md                # Configuration guide
│   │   └── contributing.md                 # Contribution guidelines
│   └── benchmarks/                         # Performance benchmarks
│       └── results.md                      # Benchmark results
│
├── docker/                                 # Docker configurations
│   ├── docker-compose.yml                  # Development environment
│   ├── docker-compose.prod.yml             # Production (optional)
│   ├── Dockerfile.falkordb                 # Custom FalkorDB image (if needed)
│   └── .env.example                        # Environment variables template
│
├── scripts/                                # Utility scripts
│   ├── setup.sh                            # Initial setup script
│   ├── dev.sh                              # Start dev environment
│   ├── test.sh                             # Run all tests
│   ├── benchmark.sh                        # Run benchmarks
│   └── cleanup.sh                          # Clean up docker volumes
│
├── .github/                                # GitHub workflows (future)
│   └── workflows/
│       ├── test.yml                        # CI tests
│       └── publish.yml                     # PyPI publish
│
├── pyproject.toml                          # Python project config
├── README.md                               # Project README
├── LICENSE                                 # MIT license
├── .gitignore                              # Git ignore rules
├── .env.example                            # Environment template
├── .python-version                         # Python version (3.10+)
├── .pre-commit-config.yaml                 # Pre-commit hooks
└── AGENT_WORKFLOW.md                       # Agent workflow rules
```

### Directory Purpose Explanations

#### Root Level

**`.spec-workflow/`**: Spec workflow system
- `specs/`: Feature specifications (requirements, design, tasks)
- `steering/`: High-level steering documents (product, tech, structure)
- `templates/`: Templates for specs and steering docs
- `approvals/`: Approval tracking for documents

**`research/`**: Research documents (preserved for reference)
- Final synthesis and technical research reports
- **Purpose**: Historical context, decision rationale, future reference
- **Status**: Read-only archive, not actively maintained

**`src/`**: Source code packages (main development here)
- Three distinct Python packages: `zapomni_mcp`, `zapomni_core`, `zapomni_db`
- **Rationale**: Clean separation of concerns, modularity, testability

**`tests/`**: All tests (unit, integration, e2e)
- 70% unit, 25% integration, 5% e2e (test pyramid)
- Mirrors `src/` structure for easy navigation

**`docs/`**: User and developer documentation
- Architecture, API reference, guides, benchmarks
- **Audience**: Users (quick start), developers (API), contributors (guides)

**`docker/`**: Docker configurations for services
- FalkorDB, Redis, Ollama (future: containerized)
- Development and production compose files

**`scripts/`**: Automation scripts
- Setup, development, testing, benchmarking
- **Goal**: One-command operations

**`.github/`**: CI/CD workflows (future)
- Automated testing, linting, publishing
- **Phase**: Post-MVP

#### Source Packages (`src/`)

**`zapomni_mcp/`**: MCP server implementation
- **Purpose**: MCP protocol handling, tool definitions, stdio transport
- **Responsibilities**: Thin layer that delegates to `zapomni_core`
- **Key Files**:
  - `server.py`: Main entry point, server setup, tool registration
  - `tools/`: Individual MCP tool implementations (one file per tool)
  - `schemas/`: Pydantic request/response schemas
  - `config.py`: Configuration loading (environment variables)
  - `logging.py`: Structured logging setup (stderr only)

**`zapomni_core/`**: Core business logic and processing
- **Purpose**: Document processing, search, entity extraction, task management
- **Responsibilities**: All business logic, algorithms, processing pipelines
- **Key Modules**:
  - `processors/`: Document processing (PDF, text, code)
  - `embeddings/`: Embedding generation (Ollama, fallback)
  - `search/`: Search implementations (vector, BM25, hybrid, graph)
  - `extractors/`: Entity and relationship extraction
  - `chunking/`: Text and code chunking strategies
  - `tasks/`: Background task management
  - `utils/`: Shared utilities

**`zapomni_db/`**: Database client implementations
- **Purpose**: Database abstractions, storage layer
- **Responsibilities**: FalkorDB client, Redis cache, data models
- **Key Modules**:
  - `falkordb/`: FalkorDB client, schema, queries
  - `redis_cache/`: Redis cache client
  - `models.py`: Shared Pydantic data models

#### Separation Rationale

**Why Three Packages?**

1. **zapomni_mcp**: MCP-specific code
   - Easy to swap transport layer (stdio → HTTP)
   - Clear boundary between protocol and logic
   - Testable in isolation

2. **zapomni_core**: Reusable business logic
   - Can be used standalone (without MCP)
   - Clean algorithms independent of transport
   - Maximum testability (pure functions)

3. **zapomni_db**: Database abstraction
   - Easy to swap backends (FalkorDB → ChromaDB + Neo4j)
   - Clean separation of storage concerns
   - Mockable for testing

**Benefits**:
- **Modularity**: Change one layer without affecting others
- **Testability**: Mock dependencies easily
- **Reusability**: Core logic can be used in other contexts
- **Clarity**: Clear responsibilities per package

---

## Module Organization

### Package: zapomni_mcp

**Purpose**: MCP protocol implementation and server

**Design Pattern**: Thin adapter layer that delegates to `zapomni_core`

#### Key Files

**`server.py`**: Main entry point
```python
"""Zapomni MCP Server - Local-first AI Memory System

This is the main entry point for the Zapomni MCP server. It sets up the
stdio transport, registers all tools, and handles incoming MCP requests.

Usage:
    python -m zapomni_mcp.server

Environment Variables:
    FALKORDB_HOST: FalkorDB host (default: localhost)
    FALKORDB_PORT: FalkorDB port (default: 6379)
    OLLAMA_HOST: Ollama API URL (default: http://localhost:11434)
    LOG_LEVEL: Logging level (default: INFO)
"""

import asyncio
import sys
from mcp.server import Server
from mcp.server.stdio import stdio_server

from .config import settings
from .logging import setup_logging
from .tools import register_tools

# Setup logging to stderr (stdout reserved for MCP)
setup_logging(level=settings.log_level)

async def main():
    """Main entry point for Zapomni MCP server."""
    # Create MCP server
    server = Server("zapomni-memory")

    # Register all tools
    register_tools(server)

    # Start stdio transport
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
```

**`tools/__init__.py`**: Tool registry
```python
"""MCP Tool Registry

Registers all available tools with the MCP server.
"""

from mcp.server import Server

from .add_memory import add_memory
from .search_memory import search_memory
from .get_stats import get_stats

def register_tools(server: Server) -> None:
    """Register all MCP tools with the server.

    Args:
        server: MCP Server instance
    """
    # Phase 1: Essential tools
    server.tool()(add_memory)
    server.tool()(search_memory)
    server.tool()(get_stats)

    # Phase 2: Knowledge graph tools (conditional import)
    try:
        from .build_graph import build_graph
        from .get_related import get_related
        from .graph_status import graph_status

        server.tool()(build_graph)
        server.tool()(get_related)
        server.tool()(graph_status)
    except ImportError:
        pass  # Phase 2 tools not yet implemented
```

**`tools/add_memory.py`**: Individual tool implementation
```python
"""add_memory MCP Tool

Store information in memory system with automatic chunking and embedding.
"""

from typing import Dict, Any, Optional
import structlog

from ..schemas.requests import AddMemoryRequest
from ..schemas.responses import AddMemoryResponse
from zapomni_core.processors import DocumentProcessor
from zapomni_db.falkordb import FalkorDBClient
from ..config import settings

logger = structlog.get_logger()

async def add_memory(
    text: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Store information in memory system.

    Processes the provided text by chunking, embedding, and storing in FalkorDB.
    Returns a unique memory ID for later retrieval.

    Args:
        text: The text content to remember. Must be non-empty.
        metadata: Optional metadata to attach (tags, source, date, etc.)

    Returns:
        Dictionary with:
            - status: "success" or "error"
            - memory_id: UUID string identifying the stored memory
            - chunks_created: Number of chunks generated
            - text_preview: First 100 chars of text

    Raises:
        ValueError: If text is empty or exceeds max length (10MB).
        DatabaseError: If FalkorDB storage fails.

    Example:
        >>> result = await add_memory(
        ...     "RAG systems benefit from hybrid search",
        ...     {"tags": ["rag", "research"], "source": "paper.pdf"}
        ... )
        >>> print(result["memory_id"])
        '550e8400-e29b-41d4-a716-446655440000'
    """
    logger.info("add_memory_called", text_length=len(text), has_metadata=bool(metadata))

    try:
        # Validate request
        request = AddMemoryRequest(text=text, metadata=metadata or {})

        # Initialize processor (delegates to core logic)
        processor = DocumentProcessor(
            embedder=None,  # Uses default from config
            db=FalkorDBClient(
                host=settings.falkordb_host,
                port=settings.falkordb_port,
                graph_name=settings.graph_name
            )
        )

        # Process and store
        memory_id = await processor.add(request.text, request.metadata)

        # Return success response
        response = AddMemoryResponse(
            status="success",
            memory_id=str(memory_id),
            chunks_created=len(processor.last_chunks),  # Set by processor
            text_preview=text[:100]
        )

        logger.info(
            "add_memory_success",
            memory_id=memory_id,
            chunks=response.chunks_created
        )

        return response.dict()

    except ValueError as e:
        logger.error("add_memory_validation_error", error=str(e))
        return {"status": "error", "message": str(e)}

    except Exception as e:
        logger.exception("add_memory_failed", error=str(e))
        return {"status": "error", "message": f"Unexpected error: {e}"}
```

**`schemas/requests.py`**: Request validation
```python
"""Request Schemas

Pydantic models for validating MCP tool inputs.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, validator

class AddMemoryRequest(BaseModel):
    """Request schema for add_memory tool."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=10_000_000,  # 10MB
        description="Text content to remember"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata (tags, source, date, etc.)"
    )

    @validator("text")
    def validate_text(cls, v):
        """Ensure text is not just whitespace."""
        if not v.strip():
            raise ValueError("Text cannot be empty or whitespace-only")
        return v.strip()

class SearchMemoryRequest(BaseModel):
    """Request schema for search_memory tool."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Natural language search query"
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results to return"
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional filters (date_from, date_to, tags, source, etc.)"
    )
```

**`config.py`**: Configuration management
```python
"""Configuration Management

Centralized configuration using Pydantic Settings.
Loads from environment variables with sensible defaults.
"""

from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """Zapomni configuration settings."""

    # FalkorDB
    falkordb_host: str = Field(
        default="localhost",
        env="FALKORDB_HOST",
        description="FalkorDB host address"
    )
    falkordb_port: int = Field(
        default=6379,
        env="FALKORDB_PORT",
        description="FalkorDB port"
    )
    graph_name: str = Field(
        default="zapomni_memory",
        env="GRAPH_NAME",
        description="FalkorDB graph name"
    )

    # Ollama
    ollama_host: str = Field(
        default="http://localhost:11434",
        env="OLLAMA_HOST",
        description="Ollama API URL"
    )
    embedding_model: str = Field(
        default="nomic-embed-text",
        env="EMBEDDING_MODEL",
        description="Ollama embedding model"
    )
    llm_model: str = Field(
        default="llama3.1:8b",
        env="LLM_MODEL",
        description="Ollama LLM model for entity extraction"
    )

    # Performance
    chunk_size: int = Field(
        default=512,
        ge=100,
        le=2000,
        env="CHUNK_SIZE",
        description="Text chunk size in tokens"
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        le=500,
        env="CHUNK_OVERLAP",
        description="Chunk overlap in tokens"
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        env="LOG_LEVEL",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# Singleton instance
settings = Settings()
```

---

### Package: zapomni_core

**Purpose**: Core business logic and processing

**Design Pattern**: Service layer with dependency injection

#### Key Modules

**`processors/text_processor.py`**: Document processing
```python
"""Text Document Processor

Handles text document ingestion: chunking, embedding, storage.
"""

from typing import List, Dict, Any, Optional
import structlog

from ..chunking import SemanticChunker
from ..embeddings import OllamaEmbedder
from ...zapomni_db.falkordb import FalkorDBClient
from ...zapomni_db.models import Chunk, Memory

logger = structlog.get_logger()

class TextProcessor:
    """Process text documents for storage in memory system.

    This class handles the complete document ingestion pipeline:
    1. Text chunking (semantic boundaries)
    2. Embedding generation (via Ollama)
    3. Metadata extraction
    4. FalkorDB storage

    Attributes:
        chunker: SemanticChunker instance for text chunking
        embedder: OllamaEmbedder instance for embedding generation
        db: FalkorDBClient instance for storage

    Example:
        >>> chunker = SemanticChunker(chunk_size=512, overlap=50)
        >>> embedder = OllamaEmbedder(model="nomic-embed-text")
        >>> db = FalkorDBClient(host="localhost", port=6379)
        >>> processor = TextProcessor(chunker, embedder, db)
        >>> memory_id = await processor.add("Sample text", {})
    """

    def __init__(
        self,
        chunker: SemanticChunker,
        embedder: OllamaEmbedder,
        db: FalkorDBClient
    ):
        self.chunker = chunker
        self.embedder = embedder
        self.db = db
        self.last_chunks: List[Chunk] = []  # For testing/debugging

    async def add(
        self,
        text: str,
        metadata: Dict[str, Any]
    ) -> str:
        """Add text document to memory.

        Args:
            text: Text content to process
            metadata: Metadata dictionary (tags, source, date, etc.)

        Returns:
            memory_id: UUID string identifying the stored memory

        Raises:
            ValueError: If text is invalid
            DatabaseError: If storage fails
        """
        logger.info("text_processor_add", text_length=len(text))

        # 1. Chunk text
        chunks = self.chunker.chunk(text)
        self.last_chunks = chunks
        logger.debug("text_chunked", chunk_count=len(chunks))

        # 2. Generate embeddings (batch)
        chunk_texts = [c.text for c in chunks]
        embeddings = await self.embedder.embed(chunk_texts)
        logger.debug("embeddings_generated", count=len(embeddings))

        # 3. Create memory object
        memory = Memory(
            text=text,
            chunks=chunks,
            embeddings=embeddings,
            metadata=metadata
        )

        # 4. Store in FalkorDB
        memory_id = await self.db.add_memory(memory)
        logger.info("memory_stored", memory_id=memory_id)

        return memory_id
```

**`embeddings/ollama_embedder.py`**: Embedding generation
```python
"""Ollama Embedding Generator

Generates embeddings using Ollama's embedding API.
"""

from typing import List
import httpx
import structlog

logger = structlog.get_logger()

class OllamaEmbedder:
    """Generate embeddings using Ollama.

    Uses Ollama's embedding API to generate vector embeddings for text.
    Supports any Ollama embedding model (nomic-embed-text recommended).

    Attributes:
        host: Ollama API URL (e.g., http://localhost:11434)
        model: Embedding model name (e.g., nomic-embed-text)
        timeout: API timeout in seconds

    Example:
        >>> embedder = OllamaEmbedder(
        ...     host="http://localhost:11434",
        ...     model="nomic-embed-text"
        ... )
        >>> embeddings = await embedder.embed(["Hello world", "Goodbye"])
        >>> len(embeddings[0])  # nomic-embed-text is 768-dimensional
        768
    """

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
        timeout: int = 60
    ):
        self.host = host
        self.model = model
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (each is List[float])

        Raises:
            httpx.HTTPError: If Ollama API request fails
        """
        embeddings = []

        for text in texts:
            response = await self.client.post(
                f"{self.host}/api/embeddings",
                json={"model": self.model, "prompt": text}
            )
            response.raise_for_status()

            data = response.json()
            embeddings.append(data["embedding"])

        logger.debug("ollama_embed", count=len(embeddings), model=self.model)
        return embeddings

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()
```

**`search/vector_search.py`**: Vector similarity search
```python
"""Vector Similarity Search

Implements vector similarity search using FalkorDB's vector index.
"""

from typing import List, Dict, Any, Optional
import structlog

from ...zapomni_db.falkordb import FalkorDBClient
from ...zapomni_db.models import SearchResult

logger = structlog.get_logger()

class VectorSearch:
    """Vector similarity search engine.

    Performs cosine similarity search using FalkorDB's HNSW vector index.

    Attributes:
        db: FalkorDBClient instance
        similarity_threshold: Minimum similarity score (0-1)

    Example:
        >>> search = VectorSearch(db=db_client, similarity_threshold=0.5)
        >>> query_embedding = [0.1, 0.2, ...]  # 768-dimensional
        >>> results = await search.search(
        ...     query_embedding=query_embedding,
        ...     limit=10,
        ...     filters={"tags": ["python"]}
        ... )
    """

    def __init__(
        self,
        db: FalkorDBClient,
        similarity_threshold: float = 0.5
    ):
        self.db = db
        self.similarity_threshold = similarity_threshold

    async def search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Perform vector similarity search.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            filters: Optional filters (tags, date_from, date_to, etc.)

        Returns:
            List of SearchResult objects, sorted by similarity (descending)
        """
        logger.info("vector_search", limit=limit, has_filters=bool(filters))

        # Delegate to database
        results = await self.db.vector_search(
            query_embedding=query_embedding,
            limit=limit,
            filters=filters or {},
            min_similarity=self.similarity_threshold
        )

        logger.info("vector_search_complete", results=len(results))
        return results
```

---

### Package: zapomni_db

**Purpose**: Database client implementations

**Design Pattern**: Repository pattern with abstraction layer

#### Key Modules

**`falkordb/client.py`**: FalkorDB client
```python
"""FalkorDB Client

Main interface to FalkorDB database (unified vector + graph).
"""

from typing import List, Dict, Any, Optional
import uuid
from falkordb import FalkorDB
import structlog

from ..models import Memory, SearchResult, Entity

logger = structlog.get_logger()

class FalkorDBClient:
    """FalkorDB client for unified vector + graph storage.

    Provides high-level interface to FalkorDB for storing memories,
    performing vector search, and querying knowledge graph.

    Attributes:
        host: FalkorDB host (default: localhost)
        port: FalkorDB port (default: 6379)
        graph_name: Graph name (default: zapomni_memory)
        db: FalkorDB connection instance
        graph: Graph instance

    Example:
        >>> client = FalkorDBClient(
        ...     host="localhost",
        ...     port=6379,
        ...     graph_name="zapomni_memory"
        ... )
        >>> await client.add_memory(memory)
        '550e8400-e29b-41d4-a716-446655440000'
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        graph_name: str = "zapomni_memory"
    ):
        self.host = host
        self.port = port
        self.graph_name = graph_name

        # Connect to FalkorDB
        self.db = FalkorDB(host=host, port=port)
        self.graph = self.db.select_graph(graph_name)

        # Initialize schema (idempotent)
        self._init_schema()

        logger.info(
            "falkordb_connected",
            host=host,
            port=port,
            graph=graph_name
        )

    def _init_schema(self):
        """Initialize graph schema (nodes, edges, indexes)."""
        # Create vector index for embeddings
        self.graph.query("""
            CREATE VECTOR INDEX FOR (c:Chunk) ON (c.embedding)
            OPTIONS {dimension: 768, similarityFunction: 'cosine'}
        """)

        # Create property indexes
        self.graph.query("CREATE INDEX FOR (m:Memory) ON (m.id)")
        self.graph.query("CREATE INDEX FOR (e:Entity) ON (e.name)")
        self.graph.query("CREATE INDEX FOR (d:Document) ON (d.id)")

        logger.debug("falkordb_schema_initialized")

    async def add_memory(self, memory: Memory) -> str:
        """Store memory with chunks and embeddings.

        Args:
            memory: Memory object with text, chunks, embeddings, metadata

        Returns:
            memory_id: UUID string
        """
        memory_id = str(uuid.uuid4())

        # Create Memory node
        self.graph.query(
            """
            CREATE (m:Memory {
                id: $id,
                text: $text,
                tags: $tags,
                source: $source,
                timestamp: datetime()
            })
            """,
            {
                "id": memory_id,
                "text": memory.text,
                "tags": memory.metadata.get("tags", []),
                "source": memory.metadata.get("source", ""),
            }
        )

        # Create Chunk nodes with embeddings
        for i, (chunk, embedding) in enumerate(zip(memory.chunks, memory.embeddings)):
            chunk_id = f"{memory_id}_chunk_{i}"

            self.graph.query(
                """
                MATCH (m:Memory {id: $memory_id})
                CREATE (c:Chunk {
                    id: $chunk_id,
                    text: $text,
                    index: $index,
                    embedding: $embedding
                })
                CREATE (m)-[:HAS_CHUNK]->(c)
                """,
                {
                    "memory_id": memory_id,
                    "chunk_id": chunk_id,
                    "text": chunk.text,
                    "index": i,
                    "embedding": embedding
                }
            )

        logger.info(
            "memory_added",
            memory_id=memory_id,
            chunks=len(memory.chunks)
        )

        return memory_id

    async def vector_search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        filters: Dict[str, Any] = None,
        min_similarity: float = 0.5
    ) -> List[SearchResult]:
        """Perform vector similarity search.

        Args:
            query_embedding: Query vector (768-dimensional)
            limit: Max results
            filters: Optional metadata filters
            min_similarity: Minimum similarity threshold (0-1)

        Returns:
            List of SearchResult objects
        """
        # Build Cypher query with filters
        where_clause = ""
        if filters:
            # TODO: Build WHERE clause from filters
            pass

        # Vector search query
        query = f"""
            CALL db.idx.vector.queryNodes(
                'Chunk',
                'embedding',
                {limit},
                $query_embedding
            ) YIELD node, score
            WHERE score >= $min_similarity
            MATCH (m:Memory)-[:HAS_CHUNK]->(node)
            RETURN m.id as memory_id,
                   node.text as text,
                   score as similarity,
                   m.tags as tags,
                   m.source as source,
                   m.timestamp as timestamp
            ORDER BY score DESC
        """

        result = self.graph.query(query, {
            "query_embedding": query_embedding,
            "min_similarity": min_similarity
        })

        # Convert to SearchResult objects
        results = [
            SearchResult(
                memory_id=row[0],
                text=row[1],
                similarity_score=row[2],
                tags=row[3],
                source=row[4],
                timestamp=row[5]
            )
            for row in result.result_set
        ]

        logger.debug("vector_search_complete", results=len(results))
        return results
```

**`models.py`**: Shared data models
```python
"""Shared Data Models

Pydantic models used across packages.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class Chunk(BaseModel):
    """Text chunk model."""
    text: str
    index: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Memory(BaseModel):
    """Memory model (document + chunks + embeddings)."""
    text: str
    chunks: List[Chunk]
    embeddings: List[List[float]]
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SearchResult(BaseModel):
    """Search result model."""
    memory_id: str
    text: str
    similarity_score: float
    tags: List[str] = Field(default_factory=list)
    source: str = ""
    timestamp: Optional[datetime] = None

class Entity(BaseModel):
    """Knowledge graph entity model."""
    name: str
    type: str  # PERSON, ORG, TECHNOLOGY, etc.
    description: str = ""
    confidence: float = 1.0
```

---

## Coding Conventions

### Python Style Guide

**Standard**: PEP 8 + Black + Type Hints

**Tools**:
- **Black**: Code formatting (line length 100)
- **isort**: Import sorting
- **mypy**: Type checking (strict mode)
- **flake8**: Linting
- **pylint**: Additional linting (optional)

**Configuration**: See `pyproject.toml` section below

### File Naming

**Modules**: `snake_case.py`

✅ **Good**:
```
text_processor.py
ollama_embedder.py
vector_search.py
```

❌ **Bad**:
```
TextProcessor.py
OllamaEmbedder.py
vectorSearch.py
```

**Classes**: `PascalCase`

✅ **Good**:
```python
class DocumentProcessor:
class OllamaEmbedder:
class FalkorDBClient:
```

❌ **Bad**:
```python
class document_processor:
class ollama_embedder:
class falkordb_client:
```

**Functions/Variables**: `snake_case`

✅ **Good**:
```python
def add_memory(text: str) -> str:
    memory_id = uuid.uuid4()
    chunk_size = 512
```

❌ **Bad**:
```python
def AddMemory(text: str) -> str:
    memoryID = uuid.uuid4()
    chunkSize = 512
```

**Constants**: `UPPER_SNAKE_CASE`

✅ **Good**:
```python
MAX_CHUNK_SIZE = 512
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_DIMENSIONS = 768
```

❌ **Bad**:
```python
max_chunk_size = 512
defaultEmbeddingModel = "nomic-embed-text"
```

### Type Hints

**Requirement**: 100% coverage for public APIs

**Examples**:

```python
from typing import List, Dict, Optional, Any, Union

# Function signatures
def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50
) -> List[str]:
    """Chunk text into smaller pieces."""
    ...

# Complex types
from pydantic import BaseModel

class Memory(BaseModel):
    id: str
    text: str
    embedding: List[float]
    metadata: Optional[Dict[str, Any]] = None
    tags: List[str] = []

# Async functions
async def embed(self, texts: List[str]) -> List[List[float]]:
    """Generate embeddings asynchronously."""
    ...

# Generic types
from typing import TypeVar, Generic

T = TypeVar('T')

class Cache(Generic[T]):
    def get(self, key: str) -> Optional[T]:
        ...
```

**mypy Configuration** (`pyproject.toml`):
```toml
[tool.mypy]
python_version = "3.10"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_any_unimported = false
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
check_untyped_defs = true
```

### Docstrings

**Standard**: Google Style

**Required For**:
- All public functions
- All classes
- All modules (module-level docstring)

**Function Docstring Example**:
```python
def add_memory(text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Store information in memory system.

    Processes the provided text by chunking, embedding, and storing in FalkorDB.
    Returns a unique memory ID for later retrieval.

    Args:
        text: The text content to remember. Must be non-empty.
        metadata: Optional metadata to attach (tags, source, date, etc.)

    Returns:
        A UUID string identifying the stored memory.

    Raises:
        ValueError: If text is empty or exceeds max length (10MB).
        DatabaseError: If FalkorDB storage fails.

    Example:
        >>> memory_id = add_memory("Important information", {"source": "notes"})
        >>> print(memory_id)
        '550e8400-e29b-41d4-a716-446655440000'
    """
    ...
```

**Class Docstring Example**:
```python
class TextProcessor:
    """Processes text documents for storage in memory system.

    This class handles the complete document ingestion pipeline:
    1. Text chunking (semantic boundaries)
    2. Embedding generation (via Ollama)
    3. Metadata extraction
    4. FalkorDB storage

    Attributes:
        chunker: SemanticChunker instance for text chunking
        embedder: OllamaEmbedder instance for embedding generation
        db: FalkorDBClient instance for storage

    Example:
        >>> processor = TextProcessor(chunker, embedder, db)
        >>> memory_id = await processor.add("Sample text", {})
    """
    ...
```

**Module Docstring Example**:
```python
"""Text Document Processor

This module provides the TextProcessor class for handling text document
ingestion into the Zapomni memory system.

Usage:
    from zapomni_core.processors import TextProcessor

    processor = TextProcessor(chunker, embedder, db)
    memory_id = await processor.add(text, metadata)
"""
```

### Import Order

**Standard**: isort with Black compatibility

**Order**:
1. Standard library
2. Third-party packages
3. Local application imports

**Example**:
```python
# Standard library
import os
import sys
from typing import List, Dict, Optional

# Third-party
import numpy as np
from pydantic import BaseModel
from mcp.server import Server

# Local
from zapomni_core.processors import DocumentProcessor
from zapomni_db.falkordb import FalkorDBClient
from .schemas import AddMemoryRequest
```

**isort Configuration** (`pyproject.toml`):
```toml
[tool.isort]
profile = "black"
line_length = 100
multi_line_output = 3
include_trailing_comma = true
force_grid_wrap = 0
use_parentheses = true
ensure_newline_before_comments = true
```

### Error Handling

**Pattern**: Explicit exception handling with context

**Good** ✅:
```python
import structlog

logger = structlog.get_logger()

try:
    embedding = await embedder.embed(text)
except OllamaConnectionError as e:
    logger.error("ollama_connection_failed", error=str(e), text_length=len(text))
    raise EmbeddingError(f"Failed to generate embedding: {e}") from e
except OllamaTimeoutError as e:
    logger.warn("ollama_timeout_retrying_with_fallback", error=str(e))
    embedding = await fallback_embedder.embed(text)
```

**Bad** ❌:
```python
try:
    embedding = await embedder.embed(text)
except Exception as e:  # Too broad
    print(f"Error: {e}")  # Don't use print
    pass  # Don't silently fail
```

**Custom Exceptions**:
```python
# zapomni_core/exceptions.py

class ZapomniError(Exception):
    """Base exception for Zapomni."""
    pass

class EmbeddingError(ZapomniError):
    """Embedding generation failed."""
    pass

class DatabaseError(ZapomniError):
    """Database operation failed."""
    pass

class ValidationError(ZapomniError):
    """Input validation failed."""
    pass
```

### Logging

**Standard**: Structured logging with structlog

**Configuration**:
```python
# zapomni_mcp/logging.py

import sys
import structlog

def setup_logging(level: str = "INFO"):
    """Setup structured logging to stderr."""
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ],
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        wrapper_class=structlog.BoundLogger,
        context_class=dict,
        cache_logger_on_first_use=True,
    )
```

**Usage**:
```python
import structlog

logger = structlog.get_logger()

# Good: Structured ✅
logger.info(
    "memory_added",
    memory_id=memory_id,
    chunk_count=len(chunks),
    duration_ms=duration
)

# Bad: Unstructured ❌
logger.info(f"Added memory {memory_id} with {len(chunks)} chunks in {duration}ms")
```

**Log Levels**:
- **DEBUG**: Detailed debugging (off in production)
- **INFO**: Normal operations (default)
- **WARNING**: Degraded performance, fallbacks used
- **ERROR**: Operation failed, needs attention
- **CRITICAL**: System failure, immediate action required

**Why stderr?**: MCP protocol uses stdout for communication, so all logging must go to stderr.

### Async/Await

**Pattern**: Use async I/O for all database and network calls

**Good** ✅:
```python
async def add_memory(text: str) -> str:
    embedding = await embedder.embed(text)  # Network I/O
    memory_id = await db.store(embedding)  # Database I/O
    return memory_id
```

**Bad** ❌:
```python
def add_memory(text: str) -> str:
    embedding = embedder.embed(text)  # Blocking
    memory_id = db.store(embedding)  # Blocking
    return memory_id
```

**Running Async**:
```python
# In MCP server (already async context)
memory_id = await add_memory(text)

# In scripts (create async context)
import asyncio
memory_id = asyncio.run(add_memory(text))

# In tests (pytest-asyncio)
@pytest.mark.asyncio
async def test_add_memory():
    memory_id = await add_memory("test")
    assert memory_id is not None
```

---

## Configuration Management

### Environment Variables

**File**: `.env` (local, gitignored) + `.env.example` (template, committed)

**Template** (`.env.example`):
```bash
# FalkorDB Configuration
FALKORDB_HOST=localhost
FALKORDB_PORT=6379
GRAPH_NAME=zapomni_memory

# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text
LLM_MODEL=llama3.1:8b

# Redis Cache (Phase 2)
REDIS_HOST=localhost
REDIS_PORT=6380
REDIS_TTL_SECONDS=86400

# Performance Tuning
CHUNK_SIZE=512
CHUNK_OVERLAP=50
EMBEDDING_BATCH_SIZE=32
SEARCH_LIMIT_DEFAULT=10

# Logging
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# Features (Phase-gated)
ENABLE_HYBRID_SEARCH=false  # Phase 2
ENABLE_KNOWLEDGE_GRAPH=false  # Phase 2
ENABLE_CODE_INDEXING=false  # Phase 3
```

### Pydantic Settings

**Implementation** (`zapomni_mcp/config.py` - already shown above)

**Usage**:
```python
from zapomni_mcp.config import settings

# Use in code
client = FalkorDBClient(
    host=settings.falkordb_host,
    port=settings.falkordb_port,
    graph_name=settings.graph_name
)

embedder = OllamaEmbedder(
    host=settings.ollama_host,
    model=settings.embedding_model
)

chunker = SemanticChunker(
    chunk_size=settings.chunk_size,
    overlap=settings.chunk_overlap
)
```

### Docker Compose Configuration

**File**: `docker/docker-compose.yml`

```yaml
version: '3.8'

services:
  falkordb:
    image: falkordb/falkordb:latest
    container_name: zapomni-falkordb
    ports:
      - "6379:6379"
    volumes:
      - falkordb_data:/data
    environment:
      - FALKORDB_ARGS=--save 60 1
    restart: unless-stopped

  redis-cache:
    image: redis:7-alpine
    container_name: zapomni-redis-cache
    ports:
      - "6380:6379"
    volumes:
      - redis_data:/data
    command: redis-server --maxmemory 1gb --maxmemory-policy allkeys-lru
    restart: unless-stopped

  # Ollama (future: if containerized)
  # ollama:
  #   image: ollama/ollama:latest
  #   container_name: zapomni-ollama
  #   ports:
  #     - "11434:11434"
  #   volumes:
  #     - ollama_data:/root/.ollama
  #   restart: unless-stopped

volumes:
  falkordb_data:
  redis_data:
  # ollama_data:
```

**Usage**:
```bash
# Start all services
docker-compose -f docker/docker-compose.yml up -d

# View logs
docker-compose -f docker/docker-compose.yml logs -f

# Stop services
docker-compose -f docker/docker-compose.yml down

# Clean up (WARNING: deletes data)
docker-compose -f docker/docker-compose.yml down -v
```

---

## Testing Organization

### Test Structure

**Pyramid**: 70% unit, 25% integration, 5% E2E

**Directory Mapping**:
```
src/zapomni_core/processors/text_processor.py
→ tests/unit/test_text_processor.py

src/zapomni_db/falkordb/client.py
→ tests/integration/test_falkordb_client.py

End-to-end workflow
→ tests/e2e/test_full_workflow.py
```

### Unit Tests (70%)

**Characteristics**:
- Fast (< 1ms per test)
- No external dependencies (mock everything)
- Test single function/class
- High coverage (90%+)

**Example**:
```python
# tests/unit/test_chunker.py

import pytest
from zapomni_core.chunking import SemanticChunker

def test_chunk_text_basic():
    """Test basic text chunking."""
    chunker = SemanticChunker(chunk_size=100, overlap=10)
    text = "A" * 250

    chunks = chunker.chunk(text)

    assert len(chunks) == 3  # 250 chars / 100 per chunk
    assert len(chunks[0].text) == 100
    assert chunks[0].index == 0

def test_chunk_text_overlap():
    """Test chunking with overlap."""
    chunker = SemanticChunker(chunk_size=100, overlap=20)
    text = "A" * 100 + "B" * 100

    chunks = chunker.chunk(text)

    # Verify overlap
    assert "A" in chunks[1].text  # Overlap from first chunk
    assert "B" in chunks[1].text

def test_chunk_text_empty():
    """Test chunking empty text raises ValueError."""
    chunker = SemanticChunker(chunk_size=100, overlap=10)

    with pytest.raises(ValueError, match="empty"):
        chunker.chunk("")

@pytest.mark.parametrize("chunk_size,expected_chunks", [
    (100, 3),
    (200, 2),
    (500, 1),
])
def test_chunk_text_sizes(chunk_size, expected_chunks):
    """Test different chunk sizes."""
    chunker = SemanticChunker(chunk_size=chunk_size, overlap=0)
    text = "A" * 250

    chunks = chunker.chunk(text)

    assert len(chunks) == expected_chunks
```

### Integration Tests (25%)

**Characteristics**:
- Slower (< 100ms per test)
- Use real services (FalkorDB, Ollama via Docker)
- Test interactions between components
- Medium coverage (70%+)

**Example**:
```python
# tests/integration/test_falkordb_client.py

import pytest
from zapomni_db.falkordb import FalkorDBClient
from zapomni_db.models import Memory, Chunk

@pytest.fixture
def db_client():
    """FalkorDB test instance (Docker)."""
    client = FalkorDBClient(
        host="localhost",
        port=6379,
        graph_name="test_graph"
    )
    yield client
    # Cleanup
    client.graph.query("MATCH (n) DETACH DELETE n")

@pytest.mark.asyncio
async def test_add_and_search_memory(db_client):
    """Test full add → search workflow."""
    # Add memory
    embedding = [0.1] * 768
    memory = Memory(
        text="Test memory",
        chunks=[Chunk(text="Test chunk", index=0)],
        embeddings=[embedding],
        metadata={"source": "test"}
    )

    memory_id = await db_client.add_memory(memory)

    assert memory_id is not None

    # Search
    results = await db_client.vector_search(
        query_embedding=embedding,
        limit=10
    )

    assert len(results) == 1
    assert results[0].text == "Test chunk"
    assert results[0].memory_id == memory_id

@pytest.mark.asyncio
async def test_vector_search_with_filters(db_client):
    """Test vector search with metadata filters."""
    # Add multiple memories with different tags
    for tag in ["python", "javascript", "rust"]:
        embedding = [0.1 * ord(tag[0])] * 768  # Different embeddings
        memory = Memory(
            text=f"Memory about {tag}",
            chunks=[Chunk(text=f"{tag} content", index=0)],
            embeddings=[embedding],
            metadata={"tags": [tag]}
        )
        await db_client.add_memory(memory)

    # Search with filter
    results = await db_client.vector_search(
        query_embedding=[0.1 * ord('p')] * 768,
        limit=10,
        filters={"tags": ["python"]}
    )

    assert len(results) == 1
    assert "python" in results[0].tags
```

### E2E Tests (5%)

**Characteristics**:
- Slowest (< 5s per test)
- Full workflow from MCP call to result
- All services running
- Low coverage (critical paths only)

**Example**:
```python
# tests/e2e/test_full_workflow.py

import pytest
import asyncio
from zapomni_mcp.server import create_server
from mcp.client import Client

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_add_and_search_workflow():
    """Test full MCP workflow: add → search."""
    # Setup MCP server and client
    server = create_server()
    # (In real test, would use stdio pipes)

    # Add memory via MCP
    add_response = await server.call_tool("add_memory", {
        "text": "Paris is the capital of France",
        "metadata": {"source": "geography"}
    })

    assert add_response["status"] == "success"
    memory_id = add_response["memory_id"]

    # Wait for processing
    await asyncio.sleep(0.5)

    # Search for it
    search_response = await server.call_tool("search_memory", {
        "query": "What is the capital of France?",
        "limit": 5
    })

    assert search_response["status"] == "success"
    assert len(search_response["results"]) >= 1
    assert any(r["memory_id"] == memory_id for r in search_response["results"])

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_stats_workflow():
    """Test get_stats returns valid statistics."""
    server = create_server()

    # Add some memories first
    for i in range(5):
        await server.call_tool("add_memory", {
            "text": f"Test memory {i}",
            "metadata": {}
        })

    # Get stats
    stats_response = await server.call_tool("get_stats", {})

    assert stats_response["status"] == "success"
    assert stats_response["statistics"]["total_memories"] >= 5
    assert "database_size_mb" in stats_response["statistics"]
```

### Test Fixtures

**Shared Fixtures** (`tests/conftest.py`):
```python
import pytest
import asyncio
from zapomni_db.falkordb import FalkorDBClient
from zapomni_core.embeddings import OllamaEmbedder

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def docker_services():
    """Start Docker services for tests."""
    # Use pytest-docker or manual docker-compose
    import subprocess
    subprocess.run(["docker-compose", "-f", "docker/docker-compose.yml", "up", "-d"])
    yield
    subprocess.run(["docker-compose", "-f", "docker/docker-compose.yml", "down"])

@pytest.fixture
async def db_client(docker_services):
    """FalkorDB client fixture."""
    client = FalkorDBClient(
        host="localhost",
        port=6379,
        graph_name="test_graph"
    )
    yield client
    # Cleanup
    client.graph.query("MATCH (n) DETACH DELETE n")

@pytest.fixture
async def embedder(docker_services):
    """Ollama embedder fixture."""
    embedder = OllamaEmbedder(
        host="http://localhost:11434",
        model="nomic-embed-text"
    )
    yield embedder
    await embedder.close()

@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return """
    The Python programming language was created by Guido van Rossum in 1991.
    It is known for its simple, readable syntax and powerful standard library.
    Python is widely used in web development, data science, and machine learning.
    """
```

### Running Tests

**Commands**:
```bash
# All tests
pytest

# Unit tests only (fast)
pytest tests/unit

# Integration tests (requires Docker)
pytest tests/integration

# E2E tests (slow, requires full stack)
pytest tests/e2e
pytest -m e2e

# With coverage
pytest --cov=src --cov-report=html

# Specific test
pytest tests/unit/test_chunker.py::test_chunk_text_basic

# Parallel execution (requires pytest-xdist)
pytest -n auto

# Verbose output
pytest -v

# Stop on first failure
pytest -x

# Run only failed tests from last run
pytest --lf
```

**Configuration** (`pyproject.toml`):
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
markers = [
    "unit: Unit tests (fast, no external dependencies)",
    "integration: Integration tests (medium speed, uses services)",
    "e2e: End-to-end tests (slow, full stack)",
]
addopts = [
    "--strict-markers",
    "--tb=short",
    "--cov-report=term-missing",
]
asyncio_mode = "auto"
```

---

## Documentation Structure

### User Documentation (`docs/`)

**README.md**: Quick start, installation, basic usage

```markdown
# Zapomni - Local-First AI Memory

Zapomni is a local-first MCP memory server that gives AI agents intelligent,
contextual, and private long-term memory.

## Quick Start

### 1. Install Services

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull models
ollama pull nomic-embed-text
ollama pull llama3.1:8b

# Start FalkorDB (Docker)
docker run -d -p 6379:6379 falkordb/falkordb
```

### 2. Install Zapomni

```bash
pip install zapomni-mcp
```

### 3. Configure Claude

Add to `~/.config/claude/config.json`:

```json
{
  "mcpServers": {
    "zapomni": {
      "command": "python",
      "args": ["-m", "zapomni_mcp.server"],
      "env": {
        "FALKORDB_HOST": "localhost",
        "OLLAMA_HOST": "http://localhost:11434"
      }
    }
  }
}
```

### 4. Use

```
User: Remember that Python was created by Guido van Rossum in 1991
Claude: [Calls add_memory tool]
  ✓ Memory stored with ID: 550e8400-...

User: Who created Python?
Claude: [Calls search_memory tool]
  Python was created by Guido van Rossum in 1991.
```

## Learn More

- [Installation Guide](guides/installation.md)
- [Configuration](guides/configuration.md)
- [API Reference](api/tools.md)
- [Architecture](architecture/overview.md)
```

**guides/**: How-to guides for common tasks
- `quick_start.md`: 5-minute setup
- `installation.md`: Detailed installation steps
- `configuration.md`: Configuration options
- `contributing.md`: Contribution guidelines

**api/**: API reference for MCP tools
- `tools.md`: All MCP tools with parameters, examples
- `python_api.md`: Python API (if exposed for non-MCP use)

### Developer Documentation

**architecture/**: System design docs
- `overview.md`: High-level architecture
- `mcp_protocol.md`: MCP integration details
- `data_flow.md`: Data flow diagrams

**benchmarks/**: Performance benchmarks
- `results.md`: Benchmark results, charts

### Code Documentation

**Docstrings**: Google style in code (already covered)
**Type hints**: Self-documenting types (already covered)

### Keeping Docs Updated

**When to Update**:
- **New feature**: Update `guides/`, `api/`
- **Architecture change**: Update `architecture/`
- **Performance change**: Update `benchmarks/`
- **API change**: Update `api/`

**Doc Review**: Part of PR checklist (see below)

---

## Development Workflow

### Initial Setup (New Developer)

**Prerequisites**:
```bash
# 1. Python 3.10+
python --version  # Should be 3.10 or newer

# 2. Docker Desktop
docker --version

# 3. Git
git --version
```

**Setup Steps**:
```bash
# 1. Clone repository
git clone https://github.com/your-org/zapomni.git
cd zapomni

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# OR
.venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -e ".[dev]"  # Editable install with dev dependencies

# 4. Install pre-commit hooks
pre-commit install

# 5. Copy environment template
cp .env.example .env
# Edit .env with your settings (defaults work for local dev)

# 6. Start services (FalkorDB, Redis)
./scripts/dev.sh

# 7. Pull Ollama models
ollama pull nomic-embed-text
ollama pull llama3.1:8b

# 8. Run tests to verify setup
pytest tests/unit  # Should pass
```

**Expected Time**: < 30 minutes

### Daily Development Workflow

**1. Start Services**:
```bash
./scripts/dev.sh  # Starts Docker services (FalkorDB, Redis)
```

**2. Make Changes**:
```bash
# Edit code in src/
# Edit tests in tests/
```

**3. Run Tests** (continuous):
```bash
# Unit tests (fast feedback loop)
pytest tests/unit -v

# Specific module
pytest tests/unit/test_chunker.py -v

# Watch mode (re-run on file change)
pytest-watch tests/unit
```

**4. Pre-Commit Checks** (automatic):
```bash
# Triggers on git commit
# Runs: black, isort, mypy, flake8
git add .
git commit -m "feat: add semantic chunking"
# Pre-commit hooks run automatically
```

**5. Manual Checks** (before PR):
```bash
# Full test suite
pytest

# Type checking
mypy src/

# Code coverage
pytest --cov=src --cov-report=html
open htmlcov/index.html

# Linting
flake8 src/ tests/
```

### Git Workflow

**Branching Strategy**:
- `main`: Stable, production-ready
- `feature/*`: New features (e.g., `feature/hybrid-search`)
- `fix/*`: Bug fixes (e.g., `fix/embedding-timeout`)
- `docs/*`: Documentation updates

**Commit Messages**: Conventional Commits

**Format**:
```
type(scope): description

[optional body]

[optional footer]
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Tooling, dependencies

**Examples**:
```bash
feat(mcp): add search_memory tool
fix(embedder): handle Ollama timeout gracefully
docs(api): update MCP tools reference
refactor(search): extract RRF fusion to separate function
test(chunker): add edge case tests for empty input
chore(deps): upgrade FalkorDB client to 4.1.0
```

**Good Commit Messages** ✅:
```
feat(search): implement hybrid BM25 + vector search

- Add BM25 keyword search using rank-bm25 library
- Implement Reciprocal Rank Fusion (RRF) for result merging
- Add cross-encoder reranking for top-K refinement
- Achieves 3.4x better accuracy vs vector-only (benchmark)

Closes #42
```

**Bad Commit Messages** ❌:
```
fixed stuff
updated files
wip
asdf
```

### Pull Request Process

**Steps**:
1. Create feature branch: `git checkout -b feature/my-feature`
2. Make changes + write tests
3. Run pre-commit: `pre-commit run --all-files`
4. Run full test suite: `pytest`
5. Push branch: `git push origin feature/my-feature`
6. Create PR on GitHub
7. Request review
8. Address feedback
9. Merge after approval

**PR Checklist**:
- [ ] Tests added/updated (unit + integration)
- [ ] Documentation updated (API, guides, architecture)
- [ ] Type hints added to new functions
- [ ] Docstrings added (Google style)
- [ ] Pre-commit hooks passing
- [ ] All tests passing (unit, integration, e2e)
- [ ] No decrease in code coverage
- [ ] CHANGELOG.md updated (if user-facing change)

**PR Template** (`.github/pull_request_template.md`):
```markdown
## Description

Brief description of what this PR does.

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing

How was this tested?

- [ ] Unit tests
- [ ] Integration tests
- [ ] E2E tests
- [ ] Manual testing

## Checklist

- [ ] Tests pass locally (`pytest`)
- [ ] Pre-commit hooks pass (`pre-commit run --all-files`)
- [ ] Documentation updated
- [ ] Type hints added
- [ ] Docstrings added
- [ ] CHANGELOG.md updated (if needed)

## Related Issues

Closes #123
```

### Release Process (Future)

**Versioning**: Semantic Versioning (MAJOR.MINOR.PATCH)

**Steps**:
1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md` with release notes
3. Create git tag: `git tag v0.1.0`
4. Push tag: `git push origin v0.1.0`
5. GitHub Actions builds and publishes to PyPI (automated)

**CHANGELOG.md Format**:
```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.2.0] - 2025-12-01

### Added
- Hybrid search (BM25 + vector + reranking)
- Semantic caching with 68% hit rate
- Knowledge graph construction (entities + relationships)

### Changed
- Improved chunking algorithm (semantic boundaries)
- Upgraded FalkorDB to 4.1.0

### Fixed
- Ollama timeout handling
- Memory leak in embedding cache

## [0.1.0] - 2025-11-22

### Added
- Initial MVP release
- add_memory, search_memory, get_stats tools
- FalkorDB integration
- Ollama embedding support
```

---

## Code Review Guidelines

### What Reviewers Check

**Functionality**:
- Does it work as intended?
- Are edge cases handled?
- Are there tests?
- Do tests pass?

**Code Quality**:
- Type hints present?
- Docstrings clear?
- Follows coding conventions?
- Error handling appropriate?

**Architecture**:
- Fits with existing design?
- Doesn't duplicate existing code?
- Proper layer separation (MCP → Core → DB)?
- Swappable components respected?

**Performance**:
- No obvious performance issues?
- Async I/O used appropriately?
- Database queries optimized?

**Security**:
- Input validation?
- No injection vulnerabilities?
- Secrets not hardcoded?

### Review Process

**Time Expectations**:
- Small PR (< 100 lines): 1 day
- Medium PR (100-500 lines): 2 days
- Large PR (> 500 lines): Split into smaller PRs

**Approval Required**: 1 reviewer minimum

**Merge**: Squash and merge (clean history)

**Review Comments**:
- **Blocking**: Must be addressed before merge
- **Non-blocking**: Suggestions, can be addressed later
- **Nit**: Minor style issues

**Example Review Comment** (good):
```markdown
**Blocking**: This function lacks error handling for database failures.

Suggestion: Add try-except block and return meaningful error:

```python
try:
    result = await db.query(...)
except DatabaseError as e:
    logger.error("query_failed", error=str(e))
    raise
```
```

---

## Continuous Integration (Future)

### GitHub Actions

**Workflow**: `.github/workflows/test.yml`

```yaml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    services:
      falkordb:
        image: falkordb/falkordb:latest
        ports:
          - 6379:6379

      redis:
        image: redis:7-alpine
        ports:
          - 6380:6379

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"

      - name: Run linters
        run: |
          black --check src/ tests/
          isort --check src/ tests/
          flake8 src/ tests/
          mypy src/

      - name: Run tests
        run: |
          pytest --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```

**On Push**:
1. Run linters (black, isort, flake8, mypy)
2. Run unit tests
3. Run integration tests (Docker services)
4. Upload coverage to Codecov

**On PR**:
- Same as push
- Require passing before merge

**On Tag** (Release):
1. Build package
2. Run full test suite
3. Publish to PyPI (if tests pass)

---

## Project Configuration Files

### pyproject.toml

**Complete Configuration** (`pyproject.toml`):
```toml
[build-system]
requires = ["setuptools>=65.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "zapomni-mcp"
version = "0.1.0"
description = "Local-first MCP memory server for AI agents"
authors = [
    {name = "Goncharenko Anton aka alienxs2", email = "your-email@example.com"}
]
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.10"
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
keywords = ["mcp", "rag", "memory", "ai", "agents", "local-first"]

dependencies = [
    # MCP
    "mcp>=0.1.0",

    # Database
    "falkordb>=4.0.0",
    "redis>=5.0.0",

    # LLM & Embeddings
    "httpx>=0.25.0",  # For Ollama API

    # Processing
    "langchain>=0.1.0",
    "sentence-transformers>=2.2.0",
    "spacy>=3.7.0",
    "rank-bm25>=0.2.2",

    # Document Processing
    "pymupdf>=1.23.0",  # PDF
    "python-docx>=1.0.0",  # DOCX
    "trafilatura>=1.6.0",  # HTML

    # Utilities
    "pydantic>=2.5.0",
    "pydantic-settings>=2.1.0",
    "structlog>=23.2.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    # Testing
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.1.0",
    "pytest-watch>=4.2.0",
    "pytest-xdist>=3.5.0",

    # Linting & Formatting
    "black>=23.12.0",
    "isort>=5.13.0",
    "flake8>=7.0.0",
    "mypy>=1.8.0",
    "pylint>=3.0.0",

    # Pre-commit
    "pre-commit>=3.6.0",

    # Documentation
    "mkdocs>=1.5.0",
    "mkdocs-material>=9.5.0",
]

[project.urls]
Homepage = "https://github.com/your-org/zapomni"
Documentation = "https://zapomni.readthedocs.io"
Repository = "https://github.com/your-org/zapomni"
Issues = "https://github.com/your-org/zapomni/issues"

[project.scripts]
zapomni-mcp = "zapomni_mcp.server:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.black]
line-length = 100
target-version = ['py310']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
line_length = 100
multi_line_output = 3
include_trailing_comma = true
force_grid_wrap = 0
use_parentheses = true
ensure_newline_before_comments = true

[tool.mypy]
python_version = "3.10"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_any_unimported = false
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
check_untyped_defs = true

[[tool.mypy.overrides]]
module = [
    "falkordb.*",
    "rank_bm25.*",
    "trafilatura.*",
]
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
markers = [
    "unit: Unit tests (fast, no external dependencies)",
    "integration: Integration tests (medium speed, uses services)",
    "e2e: End-to-end tests (slow, full stack)",
]
addopts = [
    "--strict-markers",
    "--tb=short",
    "--cov-report=term-missing",
]
asyncio_mode = "auto"

[tool.coverage.run]
source = ["src"]
omit = [
    "*/tests/*",
    "*/conftest.py",
    "*/__pycache__/*",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
    "@abstractmethod",
]
```

### .gitignore

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.venv
venv/
ENV/
env/

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/
.hypothesis/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Environment
.env
.env.local

# Logs
*.log
logs/

# Database
*.db
*.sqlite

# Docker
docker-compose.override.yml

# Documentation
docs/_build/
site/

# Temporary
tmp/
temp/
*.tmp
```

### .pre-commit-config.yaml

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict

  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        language_version: python3.10

  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
        args: ["--max-line-length=100", "--extend-ignore=E203"]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [pydantic>=2.5.0]
```

---

## Summary

### Key Takeaways

1. **Monorepo Structure**: Three packages (`zapomni_mcp`, `zapomni_core`, `zapomni_db`) for clean separation
2. **Conventions Matter**: Black, isort, mypy, type hints, docstrings—consistency is key
3. **Testing is Essential**: 70% unit, 25% integration, 5% E2E
4. **Documentation is Code**: Keep docs updated, examples everywhere
5. **Developer Experience**: Fast feedback loops, clear errors, easy onboarding

### Next Steps

1. **Review and Approve** this document
2. **Create Project Structure**: Run initial setup
3. **Begin Development**: Start with Phase 1 (MVP)
4. **Iterate**: Refine conventions based on experience

---

**Document Status**: Draft v1.0
**Created**: 2025-11-22
**Authors**: Goncharenko Anton aka alienxs2 + Claude Code
**Copyright**: Copyright (c) 2025 Goncharenko Anton aka alienxs2
**License**: MIT License
**Last Updated**: 2025-11-22

**Total Lines**: 1800+
**Total Code Examples**: 50+
**Total Configurations**: 10+

**Ready for Review**: Yes ✅
