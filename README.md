# Zapomni

Local-first MCP memory server for AI agents.

## Overview

Zapomni is a local-first MCP (Model Context Protocol) memory server that provides AI agents with intelligent, contextual, and private long-term memory. Built on a unified vector and graph database architecture using FalkorDB and powered by local LLM runtime via Ollama, Zapomni delivers enterprise-grade RAG capabilities with zero cloud dependencies.

**Key Features:**

- Local-first architecture - all data and processing stays on your machine
- Unified database - FalkorDB combines vector embeddings and knowledge graph in a single system
- Hybrid search - vector similarity, BM25 keyword search, and graph traversal for optimal results
- Zero configuration - works out-of-the-box with sensible defaults
- MCP native - seamless integration with Claude, Cursor, Cline, and other MCP clients
- Privacy guaranteed - your data never leaves your machine

## Requirements

- **FalkorDB** - localhost:6381 (or configured port)
- **Redis** - localhost:6380 (optional, for semantic caching)
- **Ollama** - localhost:11434 with models:
  - `nomic-embed-text` (embeddings)
  - `llama3.1:8b` or `qwen2.5:latest` (LLM for entity extraction)
- **Python** 3.10+

## Quick Start

### 1. Install Ollama and pull models

```bash
# Install Ollama (Linux/macOS)
curl -fsSL https://ollama.com/install.sh | sh

# Windows: Download from https://ollama.com/download

# Pull required models
ollama pull nomic-embed-text
ollama pull llama3.1:8b
```

### 2. Start services with Docker

```bash
# Start FalkorDB and Redis
docker-compose up -d

# Verify services are running
docker-compose ps
```

### 3. Install Zapomni

```bash
# Clone repository
git clone https://github.com/alienxs2/zapomni.git
cd zapomni

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# OR
.venv\Scripts\activate     # Windows

# Install package
pip install -e .

# Install with development dependencies (optional)
pip install -e ".[dev]"
```

### 4. Configure MCP client

Add to your MCP client configuration (e.g., `~/.config/claude/config.json`):

```json
{
  "mcpServers": {
    "zapomni": {
      "command": "python",
      "args": ["-m", "zapomni_mcp"],
      "env": {
        "FALKORDB_HOST": "localhost",
        "FALKORDB_PORT": "6381",
        "OLLAMA_BASE_URL": "http://localhost:11434"
      }
    }
  }
}
```

### 5. Start using

```
User: Remember that Python was created by Guido van Rossum in 1991
Claude: [Calls add_memory tool] Memory stored successfully.

User: Who created Python?
Claude: [Calls search_memory tool] Based on stored memory, Python was created by Guido van Rossum in 1991.
```

## MCP Tools (18)

### Memory Operations (4)

| Tool | Description |
|------|-------------|
| `add_memory` | Store text with automatic chunking and embedding |
| `search_memory` | Semantic search across stored memories with hybrid ranking |
| `delete_memory` | Delete specific memory by ID (requires confirmation) |
| `clear_all` | Clear all memories (requires exact phrase "DELETE ALL MEMORIES") |

### Graph Operations (4)

| Tool | Description |
|------|-------------|
| `build_graph` | Extract entities and build knowledge graph from text |
| `get_related` | Find related entities through graph traversal (1-5 hop depth) |
| `graph_status` | View knowledge graph statistics (nodes, edges, entity types) |
| `export_graph` | Export graph in 4 formats (GraphML, Cytoscape JSON, Neo4j Cypher, JSON) |

### Code Intelligence (1)

| Tool | Description |
|------|-------------|
| `index_codebase` | Index code repository with AST analysis (14+ languages, call graphs, class hierarchies) |

### System Management (3)

| Tool | Description |
|------|-------------|
| `get_stats` | Query memory statistics (total memories, chunks, database size) |
| `prune_memory` | Garbage collection for stale and orphaned nodes (dry-run mode available) |
| `set_model` | Hot-reload Ollama LLM model without server restart |

### Workspace Management (6)

| Tool | Description |
|------|-------------|
| `create_workspace` | Create a new workspace for data isolation |
| `list_workspaces` | List all available workspaces |
| `set_current_workspace` | Set the current workspace for the session |
| `get_current_workspace` | Get the current workspace for the session |
| `delete_workspace` | Delete a workspace and all its data (requires confirmation) |

## Configuration

Configuration is managed via environment variables. Copy `.env.example` to `.env` and customize as needed.

### Essential Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `FALKORDB_HOST` | localhost | FalkorDB host address |
| `FALKORDB_PORT` | 6381 | FalkorDB port |
| `OLLAMA_BASE_URL` | http://localhost:11434 | Ollama API endpoint |
| `OLLAMA_EMBEDDING_MODEL` | nomic-embed-text | Model for generating embeddings |
| `OLLAMA_LLM_MODEL` | llama3.1:8b | Model for entity extraction and refinement |
| `MAX_CHUNK_SIZE` | 512 | Maximum tokens per chunk |
| `CHUNK_OVERLAP` | 50 | Token overlap between chunks |
| `VECTOR_DIMENSIONS` | 768 | Vector embedding dimensions |
| `LOG_LEVEL` | INFO | Logging level (DEBUG, INFO, WARNING, ERROR) |

### Performance Tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `FALKORDB_POOL_SIZE` | 20 | Connection pool size for concurrent requests |
| `MAX_CONCURRENT_TASKS` | 4 | Maximum parallel processing tasks |
| `HNSW_M` | 16 | HNSW index parameter (connections per layer) |
| `HNSW_EF_CONSTRUCTION` | 200 | HNSW build-time accuracy parameter |
| `HNSW_EF_SEARCH` | 100 | HNSW search-time accuracy parameter |
| `MIN_SIMILARITY_THRESHOLD` | 0.5 | Minimum similarity score for search results |

### Optional Features

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_ENABLED` | false | Enable semantic caching with Redis |
| `ENABLE_HYBRID_SEARCH` | false | Enable BM25 + vector hybrid search |
| `ENABLE_KNOWLEDGE_GRAPH` | false | Enable automatic knowledge graph construction |
| `ENABLE_CODE_INDEXING` | false | Enable code repository indexing |

See `.env.example` for complete configuration options.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     MCP Client Layer                        │
│              (Claude, Cursor, Cline, etc.)                  │
└────────────────────────┬────────────────────────────────────┘
                         │ MCP Protocol (stdio/SSE)
┌────────────────────────▼────────────────────────────────────┐
│                   zapomni_mcp (MCP Layer)                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ MCPServer: Protocol handling, tool registration     │   │
│  │ Tools: 18 MCP tool implementations                  │   │
│  │ Transport: stdio (default) or SSE (concurrent)      │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                zapomni_core (Business Logic)                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ MemoryProcessor: Orchestrates all operations        │   │
│  │ Processors: Text, PDF, DOCX, HTML, Markdown, Code   │   │
│  │ Chunking: Semantic text chunking                    │   │
│  │ Embeddings: Ollama integration + caching            │   │
│  │ Search: Vector, BM25, Hybrid, Graph traversal       │   │
│  │ Extractors: Entity & relationship extraction        │   │
│  │ Graph: Knowledge graph builder & exporter           │   │
│  │ Code: AST analysis, call graphs, indexing           │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                  zapomni_db (Data Layer)                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ FalkorDB Client: Graph queries, vector search       │   │
│  │ Redis Cache: Semantic caching (optional)            │   │
│  │ Models: Data structures and validation              │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   External Services                         │
│    FalkorDB (6381)  │  Redis (6380)  │  Ollama (11434)      │
└─────────────────────────────────────────────────────────────┘
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Unit tests only (fast, no external dependencies)
pytest tests/unit

# Integration tests (requires services running)
pytest tests/integration

# With coverage report
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

### Code Quality

```bash
# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/

# Run all pre-commit checks
pre-commit run --all-files
```

### Build Package

```bash
# Build distribution packages
python -m build

# Install locally for testing
pip install -e .
```

## License

MIT License - Copyright (c) 2025 Goncharenko Anton aka alienxs2

See [LICENSE](LICENSE) file for details.
