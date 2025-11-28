# Zapomni - Project Structure

**Last Updated**: 2025-11-28
**Version**: v0.4.0 Foundation

---

## Directory Overview

```
zapomni/
├── src/                           # Source code
│   ├── zapomni_core/              # Core business logic
│   ├── zapomni_mcp/               # MCP server layer
│   ├── zapomni_db/                # Database clients
│   └── zapomni_cli/               # CLI tools
├── tests/                         # Test suites
│   ├── unit/                      # Unit tests (2089+)
│   ├── e2e/                       # End-to-end tests (88)
│   └── integration/               # Integration tests
├── .shashka/                      # Project management
│   ├── state/                     # Project state files
│   ├── steering/                  # Strategy documents
│   ├── specs/                     # Feature specifications
│   └── log/                       # Session logs
├── docs/                          # Documentation
├── docker/                        # Docker configurations
└── scripts/                       # Utility scripts
```

---

## Source Code Structure

### zapomni_core/ - Business Logic Layer

```
src/zapomni_core/
├── __init__.py
├── config.py                      # Configuration management
├── memory_processor.py            # Main orchestrator (1200+ lines)
│
├── embeddings/                    # Embedding services
│   ├── __init__.py
│   ├── base_embedder.py           # Abstract interface
│   ├── ollama_embedder.py         # Ollama implementation + batch API
│   └── embedding_cache.py         # Redis/memory cache
│
├── treesitter/                    # AST parsing (41 languages)
│   ├── __init__.py
│   ├── config.py                  # Tree-sitter configuration
│   │
│   ├── parser/                    # Language parsers
│   │   ├── __init__.py
│   │   ├── base_parser.py         # Abstract parser interface
│   │   ├── parser_registry.py     # Singleton registry
│   │   └── language_config.py     # Language definitions
│   │
│   └── extractors/                # Code element extractors
│       ├── __init__.py
│       ├── base_extractor.py      # Abstract extractor
│       ├── python_extractor.py    # Python AST (planned v0.5)
│       └── typescript_extractor.py # TypeScript AST (planned v0.5)
│
├── chunking/                      # Text chunking
│   ├── __init__.py
│   ├── base_chunker.py            # Abstract chunker
│   └── semantic_chunker.py        # Semantic-aware splitting
│
├── graph/                         # Knowledge graph
│   ├── __init__.py
│   ├── entity_extractor.py        # LLM-powered extraction
│   └── graph_builder.py           # Graph construction
│
├── search/                        # Search services
│   ├── __init__.py
│   ├── vector_search.py           # Vector similarity
│   └── hybrid_search.py           # Vector + graph search
│
└── workspace/                     # Workspace management
    ├── __init__.py
    └── workspace_manager.py       # Isolation logic
```

### zapomni_mcp/ - MCP Server Layer

```
src/zapomni_mcp/
├── __init__.py
├── __main__.py                    # Entry point, service initialization
├── server.py                      # MCP server implementation
│
└── tools/                         # 17 MCP tools
    ├── __init__.py
    │
    ├── memory/                    # Memory operations
    │   ├── add_memory.py          # Add new memories
    │   ├── delete_memory.py       # Delete by ID
    │   ├── search_memory.py       # Semantic search
    │   └── clear_all.py           # Destructive clear
    │
    ├── graph/                     # Knowledge graph
    │   ├── build_graph.py         # Build from text
    │   ├── get_related.py         # Graph traversal
    │   ├── graph_status.py        # Statistics
    │   └── export_graph.py        # Export formats
    │
    ├── code/                      # Code intelligence
    │   ├── index_codebase.py      # Repository indexing
    │   └── prune_memory.py        # Stale code cleanup
    │
    ├── workspace/                 # Workspace management
    │   ├── create_workspace.py
    │   ├── list_workspaces.py
    │   ├── set_current_workspace.py
    │   ├── get_current_workspace.py
    │   └── delete_workspace.py
    │
    └── system/                    # System tools
        ├── get_stats.py           # Memory statistics
        └── set_model.py           # Hot-reload Ollama model
```

### zapomni_db/ - Data Layer

```
src/zapomni_db/
├── __init__.py
│
├── clients/                       # Database clients
│   ├── __init__.py
│   ├── falkordb_client.py         # FalkorDB graph + vector
│   └── base_client.py             # Abstract interface
│
└── redis_cache/                   # Redis caching
    ├── __init__.py
    └── cache_client.py            # Redis connection management
```

### zapomni_cli/ - CLI Tools

```
src/zapomni_cli/
├── __init__.py
│
└── hooks/                         # Git hooks
    ├── __init__.py
    ├── pre_commit.py              # Pre-commit hook
    └── post_commit.py             # Post-commit indexing
```

---

## Test Structure

```
tests/
├── conftest.py                    # Shared fixtures
├── pytest.ini                     # Pytest configuration
│
├── unit/                          # Unit tests (2089+)
│   ├── conftest.py
│   │
│   ├── core/                      # Core logic tests
│   │   ├── test_memory_processor.py
│   │   ├── test_embeddings.py
│   │   ├── test_chunking.py
│   │   └── treesitter/            # Tree-sitter tests (221)
│   │       ├── test_parser_registry.py
│   │       ├── test_language_config.py
│   │       └── test_extractors.py
│   │
│   ├── mcp/                       # MCP tool tests
│   │   ├── test_add_memory.py
│   │   ├── test_search_memory.py
│   │   └── ...
│   │
│   └── db/                        # Database client tests
│       ├── test_falkordb_client.py
│       └── test_redis_cache.py
│
├── integration/                   # Integration tests
│   ├── test_memory_flow.py
│   └── test_indexing_flow.py
│
└── e2e/                           # End-to-end tests (88)
    ├── conftest.py
    ├── test_full_workflow.py
    └── test_mcp_protocol.py
```

---

## Project Management (.shashka/)

```
.shashka/
├── state/                         # Project state
│   ├── SNAPSHOT.md                # Current state snapshot
│   ├── HANDOFF.md                 # Session handoff notes
│   └── ACTIVE.md                  # Active work tracking
│
├── steering/                      # Strategy documents
│   ├── product.md                 # Product strategy
│   ├── tech.md                    # Technical architecture
│   └── structure.md               # This file
│
├── specs/                         # Feature specifications
│   └── <feature>/
│       ├── requirements.md
│       ├── design.md
│       ├── tasks.md
│       └── implementation-log.md
│
└── log/                           # Session logs
    └── session-YYYY-MM-DD-NN.md
```

---

## Configuration Files

```
zapomni/
├── pyproject.toml                 # Python project config
├── setup.py                       # Legacy setup (if needed)
├── Makefile                       # Build/test commands
├── docker-compose.yml             # Local services
│
├── .env.example                   # Environment template
├── .gitignore                     # Git ignore rules
├── .pre-commit-config.yaml        # Pre-commit hooks
│
├── config.yaml                    # Application config
└── logging.yaml                   # Logging configuration
```

---

## Key Files Reference

### Most Important Files

| File | Purpose | Lines |
|------|---------|-------|
| `src/zapomni_core/memory_processor.py` | Main orchestrator | ~1200 |
| `src/zapomni_mcp/server.py` | MCP server | ~500 |
| `src/zapomni_mcp/__main__.py` | Entry point | ~350 |
| `src/zapomni_core/embeddings/ollama_embedder.py` | Embeddings | ~470 |
| `src/zapomni_db/clients/falkordb_client.py` | Database | ~400 |

### Bug Fix Locations (Current Issues)

| Issue | Bug | Primary File | Secondary Files |
|-------|-----|--------------|-----------------|
| #12 | BUG-005 | `server.py` | `add_memory.py`, `search_memory.py` |
| #14 | BUG-002 | `index_codebase.py` | `memory_processor.py` |
| #15 | BUG-003 | `index_codebase.py` | `treesitter/extractors/` |
| #16 | BUG-004 | `workspace_tools.py` | `server.py` |
| #17 | BUG-001 | `prune_memory.py` | - |
| #18 | BUG-006 | `export_graph.py` | - |

---

## Module Dependencies

```
                    +------------------+
                    |   zapomni_mcp    |
                    |   (MCP Server)   |
                    +--------+---------+
                             |
                             | imports
                             v
                    +------------------+
                    |  zapomni_core    |
                    | (Business Logic) |
                    +--------+---------+
                             |
              +--------------+--------------+
              |              |              |
              v              v              v
      +-------+------+ +-----+------+ +----+-----+
      | zapomni_db   | | Tree-sitter| | Ollama   |
      | (FalkorDB)   | | (AST)      | | (LLM)    |
      +--------------+ +------------+ +----------+
```

---

## Development Workflow

### Local Setup
```bash
# Clone and setup
git clone https://github.com/alienxs2/zapomni
cd zapomni
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Start services
make docker-up

# Run tests
make test
make e2e
```

### Common Commands
```bash
make test           # Run unit tests
make e2e            # Run E2E tests
make lint           # Run linters
make format         # Format code
make docker-up      # Start FalkorDB + Redis
make server         # Start MCP server
```

---

## File Naming Conventions

### Python Files
- `snake_case.py` for all modules
- `test_*.py` for test files
- `_internal.py` suffix for private modules

### Classes
- `PascalCase` for class names
- `Base*` prefix for abstract classes
- `*Service`, `*Client`, `*Manager` suffixes

### Functions/Methods
- `snake_case` for all functions
- `_private_method` with underscore prefix
- `async_*` prefix NOT used (use `async def` keyword)

---

## Import Order

```python
# 1. Standard library
import asyncio
from typing import Dict, List, Optional

# 2. Third-party packages
import httpx
from pydantic import BaseModel

# 3. Local imports (absolute)
from zapomni_core.config import Config
from zapomni_core.embeddings import OllamaEmbedder
from zapomni_db.clients import FalkorDBClient
```

---

## Version Information

- **Current Version**: v0.4.0 Foundation
- **Python Version**: 3.11+
- **Tree-sitter**: 41 languages supported
- **MCP Tools**: 17 tools exposed
- **Test Count**: 2089+ unit, 88 E2E
