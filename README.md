# Zapomni

Local-first MCP memory server for AI agents.

## Overview

Zapomni is a local-first MCP (Model Context Protocol) memory server that provides AI agents with intelligent, contextual, and private long-term memory. Built on a unified vector and graph database architecture using FalkorDB and powered by local LLM runtime via Ollama, Zapomni delivers enterprise-grade RAG capabilities with zero cloud dependencies.

**Key Features:**

- **Local-first architecture** - all data and processing stays on your machine
- **Unified database** - FalkorDB combines vector embeddings and knowledge graph in a single system
- **Hybrid search** - vector similarity, BM25 keyword search, and graph traversal
- **Knowledge graph** - automatic entity extraction and relationship mapping
- **Code intelligence** - AST-based code analysis and indexing (41+ languages, Python extractor with full AST support)
- **Git Hooks integration** - automatic re-indexing on code changes
- **MCP native** - seamless integration with Claude, Cursor, Cline, and other MCP clients
- **Privacy guaranteed** - your data never leaves your machine

**All features enabled by default:**

Advanced features (hybrid search, knowledge graph, code indexing) are **enabled by default**. To disable them, set to `false` in your `.env` file:

```bash
ENABLE_HYBRID_SEARCH=false
ENABLE_KNOWLEDGE_GRAPH=false
ENABLE_CODE_INDEXING=false
```

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

### 4. Configure environment (optional)

```bash
# Copy example configuration
cp .env.example .env

# Advanced features are enabled by default
# To disable, uncomment and set to false:
# ENABLE_HYBRID_SEARCH=false
# ENABLE_KNOWLEDGE_GRAPH=false
# ENABLE_CODE_INDEXING=false
```

### 5. Configure MCP client

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

### 6. Start using

```
User: Remember that Python was created by Guido van Rossum in 1991
Claude: [Calls add_memory tool] Memory stored successfully.

User: Who created Python?
Claude: [Calls search_memory tool] Based on stored memory, Python was created by Guido van Rossum in 1991.
```

## Configuration

Configuration is managed via environment variables. Copy `.env.example` to `.env` and customize as needed.

**Note**: Advanced features (hybrid search, knowledge graph, code indexing) are **enabled by default**. To disable them in `.env`:

```bash
ENABLE_HYBRID_SEARCH=false
ENABLE_KNOWLEDGE_GRAPH=false
ENABLE_CODE_INDEXING=false
```

### Essential Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `FALKORDB_HOST` | localhost | FalkorDB host address |
| `FALKORDB_PORT` | 6381 | FalkorDB port |
| `OLLAMA_BASE_URL` | http://localhost:11434 | Ollama API endpoint |
| `OLLAMA_EMBEDDING_MODEL` | nomic-embed-text | Model for generating embeddings |
| `OLLAMA_LLM_MODEL` | llama3.1:8b | Model for entity extraction |
| `MAX_CHUNK_SIZE` | 512 | Maximum tokens per chunk |
| `CHUNK_OVERLAP` | 50 | Token overlap between chunks |
| `LOG_LEVEL` | INFO | Logging level (DEBUG, INFO, WARNING, ERROR) |

**Note**: The project uses 43 environment variables total. For complete configuration options, see the [Configuration Guide](docs/CONFIGURATION.md).

## MCP Tools

Zapomni provides 17 MCP tools organized into 5 categories. Some tools require feature flags to be enabled.

### Memory Operations (4 tools)

| Tool | Description | Requires Flag |
|------|-------------|---------------|
| `add_memory` | Store text with automatic chunking and embedding | - |
| `search_memory` | Semantic search across stored memories | - |
| `delete_memory` | Delete specific memory by ID | - |
| `clear_all` | Clear all memories (safety confirmation required) | - |

### Graph Operations (4 tools)

| Tool | Description | Requires Flag |
|------|-------------|---------------|
| `build_graph` | Extract entities and build knowledge graph | `ENABLE_KNOWLEDGE_GRAPH` |
| `get_related` | Find related entities through graph traversal | `ENABLE_KNOWLEDGE_GRAPH` |
| `graph_status` | View knowledge graph statistics | `ENABLE_KNOWLEDGE_GRAPH` |
| `export_graph` | Export graph (GraphML, Cytoscape, Neo4j, JSON) | `ENABLE_KNOWLEDGE_GRAPH` |

### Code Intelligence (1 tool)

| Tool | Description | Requires Flag |
|------|-------------|---------------|
| `index_codebase` | Index code repository (18 file extensions supported, AST analysis for Python) | `ENABLE_CODE_INDEXING` |

### System Management (3 tools)

| Tool | Description | Requires Flag |
|------|-------------|---------------|
| `get_stats` | Query memory statistics | - |
| `prune_memory` | Garbage collection for stale nodes | - |
| `set_model` | Hot-reload Ollama LLM model | - |

### Workspace Management (5 tools)

| Tool | Description | Requires Flag |
|------|-------------|---------------|
| `create_workspace` | Create a new workspace for data isolation | - |
| `list_workspaces` | List all available workspaces | - |
| `set_current_workspace` | Set the current workspace | - |
| `get_current_workspace` | Get the current workspace | - |
| `delete_workspace` | Delete a workspace and all its data | - |

For detailed API documentation, see the [API Reference](docs/API.md).

## Architecture

Zapomni consists of 4 layers:

```
┌─────────────────────────────────────────────────────┐
│              MCP Client Layer                       │
│         (Claude, Cursor, Cline, etc.)               │
└──────────────────┬──────────────────────────────────┘
                   │ MCP Protocol (stdio/SSE)
┌──────────────────▼──────────────────────────────────┐
│           zapomni_mcp (MCP Layer)                   │
│  • MCPServer: Protocol handling                     │
│  • Tools: 17 MCP tool implementations               │
│  • Transport: stdio (default) or SSE (concurrent)   │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│         zapomni_core (Business Logic)               │
│  • MemoryProcessor: Orchestrates operations         │
│  • Processors: Text, PDF, DOCX, HTML, Code          │
│  • Search: Vector, BM25, Hybrid, Graph traversal    │
│  • Extractors: Entity & relationship extraction     │
│  • Code: AST analysis, call graphs, indexing        │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│           zapomni_db (Data Layer)                   │
│  • FalkorDB Client: Graph queries, vector search    │
│  • Redis Cache: Semantic caching (optional)         │
│  • Models: Data structures and validation           │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│            External Services                        │
│  FalkorDB (6381) │ Redis (6380) │ Ollama (11434)   │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│          zapomni_cli (CLI Tools)                    │
│  • install-hooks: Git hooks installation            │
│  • Git hooks: Auto-indexing triggers                │
└─────────────────────────────────────────────────────┘
```

For detailed architecture documentation, see [ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Git Hooks Integration

Automatically re-index your codebase when files change:

```bash
# Install Git hooks in your repository
zapomni install-hooks [--repo-path PATH]
```

After installation, every git commit/merge/checkout automatically updates the knowledge graph. This ensures your AI assistant always has the latest code context.

**Supported hooks:**
- `post-commit` - Re-indexes changed files after commit
- `post-merge` - Updates index after merge operations
- `post-checkout` - Refreshes index when switching branches

**Note**: Code indexing is enabled by default. Set `ENABLE_CODE_INDEXING=false` to disable.

For more details, see the [CLI Guide](docs/CLI.md).

## Development

### Running Tests

The project includes **2134 tests** (unit + E2E + integration) with high coverage (74-89% depending on module).

```bash
# Run all tests
pytest

# Unit tests only (fast, no external dependencies)
pytest tests/unit

# E2E tests (requires MCP server running)
pytest tests/e2e

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

# Type checking (strict mode)
mypy src/

# Linting
flake8 src/ tests/

# Run all pre-commit checks
pre-commit run --all-files
```

For detailed development setup and guidelines, see [DEVELOPMENT.md](docs/DEVELOPMENT.md).

## Project Status

**Current Version**: v0.5.0-alpha

**What's Working**:
- Core memory operations (add, search, statistics)
- Knowledge graph construction and traversal
- Workspace isolation (fixed in v0.5.0-alpha)
- Git hooks integration
- All 17 MCP tools available
- Tree-sitter AST parsing (41 languages, 279 tests)
- Language-specific extractors: Python (58 tests, full AST support)
- Comprehensive test suite (2192 tests)

**Recent Fixes (v0.5.0-alpha)**:
- Workspace isolation (Issue #12)
- Performance 7-45x improvement (Issue #13)
- Code indexing with Tree-sitter (Issues #14, #15)
- Instance-level workspace state (Issue #16)
- Timezone handling in date filters (Issue #17)
- Model existence validation (Issue #18)

**Note**: All advanced features are enabled by default.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Development Setup**:
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Ensure all tests pass (`pytest`)
5. Run code quality checks (`pre-commit run --all-files`)
6. Submit a pull request

## License

MIT License - Copyright (c) 2025 Goncharenko Anton aka alienxs2

See [LICENSE](LICENSE) file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/alienxs2/zapomni/issues)
- **Discussions**: [GitHub Discussions](https://github.com/alienxs2/zapomni/discussions)

## Acknowledgments

Built with:
- [FalkorDB](https://www.falkordb.com/) - Unified graph and vector database
- [Ollama](https://ollama.com/) - Local LLM runtime
- [MCP](https://modelcontextprotocol.io/) - Model Context Protocol
