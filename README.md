```
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ███████╗ █████╗ ██████╗  ██████╗ ███╗   ███╗███╗   ██╗██╗ ║
║   ╚══███╔╝██╔══██╗██╔══██╗██╔═══██╗████╗ ████║████╗  ██║██║ ║
║     ███╔╝ ███████║██████╔╝██║   ██║██╔████╔██║██╔██╗ ██║██║ ║
║    ███╔╝  ██╔══██║██╔═══╝ ██║   ██║██║╚██╔╝██║██║╚██╗██║██║ ║
║   ███████╗██║  ██║██║     ╚██████╔╝██║ ╚═╝ ██║██║ ╚████║██║ ║
║   ╚══════╝╚═╝  ╚═╝╚═╝      ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═══╝╚═╝ ║
║                                                              ║
║           Local-First MCP Memory for AI Agents              ║
╚══════════════════════════════════════════════════════════════╝
```

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/alienxs2/zapomni/workflows/Tests/badge.svg)](https://github.com/alienxs2/zapomni/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

**Version:** 0.1.0 | **Author:** Goncharenko Anton (alienxs2) | **Status:** Alpha

[Quick Start](#quick-start) • [Features](#key-features) • [Documentation](docs/) • [Contributing](CONTRIBUTING.md)

</div>

---

## Overview

Zapomni is a **local-first MCP (Model Context Protocol) memory server** that provides AI agents with intelligent, contextual, and private long-term memory. Built on a unified vector + graph database architecture (FalkorDB) and powered by local LLM runtime (Ollama), Zapomni delivers enterprise-grade RAG capabilities with zero external dependencies and guaranteed data privacy.

### Key Features

- **Local-First Architecture**: All data and processing stays on your machine - no cloud dependencies
- **Unified Database**: FalkorDB combines vector embeddings and knowledge graph in single system
- **Hybrid Search**: Vector similarity + BM25 keyword search + graph traversal for optimal results
- **Zero Configuration**: Works out-of-the-box with sensible defaults
- **MCP Native**: Seamless integration with Claude, Cursor, Cline, and other MCP clients
- **Privacy Guaranteed**: Your data never leaves your machine

---

## Quick Start

### Prerequisites

- **Python 3.10+**
- **Docker Desktop** (for FalkorDB and Redis)
- **Ollama** (for embeddings and LLM)

### Installation

#### 1. Install Ollama

```bash
# Linux / macOS
curl -fsSL https://ollama.com/install.sh | sh

# Windows: Download from https://ollama.com/download
```

Pull required models:
```bash
ollama pull nomic-embed-text
ollama pull llama3.1:8b
```

#### 2. Install Zapomni

```bash
# Clone repository
git clone https://github.com/alienxs2/zapomni.git
cd zapomni

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# OR
.venv\Scripts\activate  # Windows

# Install with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

#### 3. Start Services

```bash
# Copy environment template
cp .env.example .env

# Start FalkorDB and Redis
docker-compose up -d

# Verify services are running
docker-compose ps
```

#### 4. Configure MCP Client

Add to your MCP client configuration (e.g., `~/.config/claude/config.json`):

```json
{
  "mcpServers": {
    "zapomni": {
      "command": "python",
      "args": ["-m", "zapomni_mcp.server"],
      "env": {
        "FALKORDB_HOST": "localhost",
        "OLLAMA_BASE_URL": "http://localhost:11434"
      }
    }
  }
}
```

---

## Usage

### Basic Memory Operations

```
User: Remember that Python was created by Guido van Rossum in 1991
Claude: [Calls add_memory tool]
  ✓ Memory stored with ID: 550e8400-e29b-41d4-a716-446655440000

User: Who created Python?
Claude: [Calls search_memory tool]
  Based on stored memory, Python was created by Guido van Rossum in 1991.

User: What do you remember about programming languages?
Claude: [Calls search_memory tool]
  I found information about Python: It was created by Guido van Rossum in 1991...
```

### Available MCP Tools

**Phase 1 (MVP):**
- `add_memory` - Store text with automatic chunking and embedding
- `search_memory` - Semantic search across stored memories
- `get_stats` - View memory statistics (total memories, chunks, size)

**Phase 2 (Future):**
- `build_graph` - Extract entities and build knowledge graph
- `get_related` - Find related entities via graph traversal
- `graph_status` - View knowledge graph statistics

**Phase 3 (Future):**
- `index_codebase` - Index code repository with AST analysis
- `delete_memory` - Delete specific memory by ID
- `clear_all` - Clear all memories (with confirmation)
- `export_graph` - Export knowledge graph in various formats

---

## Development

### Running Tests

```bash
# All tests
pytest

# Unit tests only (fast)
pytest tests/unit

# Integration tests (requires services)
pytest tests/integration

# With coverage
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

# Run all checks
pre-commit run --all-files
```

### Development Workflow

```bash
# Start services
docker-compose up -d

# Run in development mode
python -m zapomni_mcp.server

# Monitor logs
docker-compose logs -f
```

---

## Architecture

### Tech Stack

- **Language:** Python 3.10+
- **Database:** FalkorDB (unified vector + graph)
- **Cache:** Redis (semantic caching)
- **LLM Runtime:** Ollama (local embeddings + inference)
- **MCP SDK:** Official Anthropic Python SDK
- **Vector Search:** HNSW index (FalkorDB native)
- **Keyword Search:** BM25 (rank-bm25 library)
- **Entity Extraction:** SpaCy NER + Ollama LLM (hybrid)

### Package Structure

```
zapomni/
├── src/
│   ├── zapomni_mcp/        # MCP server layer
│   │   ├── server.py       # Main entry point
│   │   ├── tools/          # MCP tool implementations
│   │   ├── schemas/        # Request/response validation
│   │   └── config.py       # Configuration management
│   │
│   ├── zapomni_core/       # Core business logic
│   │   ├── processors/     # Document processing
│   │   ├── embeddings/     # Embedding generation
│   │   ├── search/         # Search algorithms
│   │   ├── extractors/     # Entity/relationship extraction
│   │   └── chunking/       # Text chunking
│   │
│   └── zapomni_db/         # Database layer
│       ├── falkordb/       # FalkorDB client
│       ├── redis_cache/    # Redis cache client
│       └── models.py       # Data models
│
├── tests/                  # Test suite
│   ├── unit/              # Unit tests (70%)
│   ├── integration/       # Integration tests (25%)
│   └── e2e/               # End-to-end tests (5%)
│
├── docs/                  # Documentation
└── docker/                # Docker configurations
```

---

## Configuration

All configuration via environment variables (`.env` file or system environment).

### Essential Settings

```bash
# Database
FALKORDB_HOST=localhost
FALKORDB_PORT=6379

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# Performance
MAX_CHUNK_SIZE=512
CHUNK_OVERLAP=50
```

See `.env.example` for complete configuration options.

---

## Performance

### Targets (MVP)

- **Search Latency:** < 500ms (P95)
- **Embedding Cache Hit Rate:** 60-68%
- **Chunk Size:** 256-512 tokens
- **Vector Dimensions:** 768 (nomic-embed-text)
- **Concurrent Tasks:** 4 (configurable)

### Benchmarks

Coming soon - will include comparisons with other RAG systems.

---

## Roadmap

### Phase 1: MVP (Current)
- [x] Project infrastructure setup
- [ ] Basic MCP server (stdio transport)
- [ ] Document chunking and embedding
- [ ] Vector similarity search
- [ ] Memory storage in FalkorDB
- [ ] Core MCP tools (add_memory, search_memory, get_stats)

### Phase 2: Enhanced Search
- [ ] BM25 keyword search
- [ ] Hybrid search (RRF fusion)
- [ ] Entity extraction (SpaCy + LLM)
- [ ] Knowledge graph construction
- [ ] Semantic caching (Redis)
- [ ] Graph traversal queries

### Phase 3: Code Intelligence
- [ ] Code repository indexing
- [ ] AST-based code chunking
- [ ] Function/class entity extraction
- [ ] Call graph analysis
- [ ] Code-specific search

### Phase 4+: Advanced Features
- [ ] Multi-language support
- [ ] Document format expansion
- [ ] Performance optimization
- [ ] HTTP/SSE transport
- [ ] Configuration profiles
- [ ] Secrets management integration

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/zapomni.git
cd zapomni

# Setup development environment
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install

# Start services
docker-compose up -d

# Run tests
pytest
```

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

**Copyright (c) 2025 Goncharenko Anton aka alienxs2**

---

## Support

- **Issues:** [GitHub Issues](https://github.com/alienxs2/zapomni/issues)
- **Discussions:** [GitHub Discussions](https://github.com/alienxs2/zapomni/discussions)
- **Documentation:** [docs/README.md](docs/README.md)

---

## Acknowledgments

- **Anthropic** for the MCP protocol and SDK
- **FalkorDB** for unified vector + graph database
- **Ollama** for local LLM runtime
- **Cognee** for inspiration on async task management
- Open-source community for excellent libraries

---

**Built with ❤️ for the local-first AI community**
