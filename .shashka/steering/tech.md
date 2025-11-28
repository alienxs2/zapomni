# Zapomni - Technical Architecture

**Last Updated**: 2025-11-28
**Version**: v0.4.0 Foundation

---

## Technology Stack

### Core Runtime
| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| Language | Python | 3.11+ | Core implementation |
| Runtime | asyncio | stdlib | Async I/O |
| Package Manager | uv/pip | latest | Dependency management |

### Data Layer
| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| Graph DB | FalkorDB | 1.0.0+ | Graph + Vector storage |
| Cache | Redis | 7.0+ | Embedding cache, session state |
| In-Memory | LRU Cache | stdlib | Fallback cache |

### AI/ML
| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| Embeddings | Ollama | latest | Local embedding generation |
| LLM | Ollama | latest | Entity extraction, refinement |
| Default Model | nomic-embed-text | - | 768-dim embeddings |

### Code Analysis
| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| AST Parser | Tree-sitter | 0.25.0+ | Language-agnostic parsing |
| Language Pack | tree-sitter-language-pack | 0.13.0+ | 41 language grammars |

### Protocol
| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| Interface | MCP Protocol | 1.0 | Claude/AI integration |
| Transport | stdio/SSE | - | Communication layer |

---

## Architecture Overview

```
+------------------+     +------------------+     +------------------+
|   AI Agents      |     |   Claude Code    |     |   Cursor/etc     |
|   (Claude, etc)  |     |   CLI            |     |                  |
+--------+---------+     +--------+---------+     +--------+---------+
         |                        |                        |
         +------------------------+------------------------+
                                  |
                          [MCP Protocol]
                                  |
+-------------------------------------------------------------------------+
|                         zapomni_mcp (Server Layer)                       |
|  +------------------+  +------------------+  +------------------+        |
|  | add_memory       |  | search_memory    |  | index_codebase   |        |
|  | delete_memory    |  | get_related      |  | prune_memory     |        |
|  | build_graph      |  | graph_status     |  | export_graph     |        |
|  | get_stats        |  | clear_all        |  | set_model        |        |
|  | create_workspace |  | list_workspaces  |  | set_workspace    |        |
|  | get_workspace    |  | delete_workspace |  |                  |        |
|  +------------------+  +------------------+  +------------------+        |
+-------------------------------------------------------------------------+
                                  |
+-------------------------------------------------------------------------+
|                         zapomni_core (Business Logic)                    |
|  +------------------+  +------------------+  +------------------+        |
|  | MemoryProcessor  |  | EmbeddingService |  | ChunkingService  |        |
|  | (orchestrator)   |  | (Ollama client)  |  | (text splitting) |        |
|  +------------------+  +------------------+  +------------------+        |
|                                                                          |
|  +------------------+  +------------------+  +------------------+        |
|  | TreeSitter       |  | EntityExtractor  |  | GraphBuilder     |        |
|  | Parser Registry  |  | (LLM-powered)    |  | (knowledge graph)|        |
|  +------------------+  +------------------+  +------------------+        |
|                                                                          |
|  +------------------+  +------------------+                              |
|  | SearchService    |  | EmbeddingCache   |                              |
|  | (hybrid search)  |  | (Redis/memory)   |                              |
|  +------------------+  +------------------+                              |
+-------------------------------------------------------------------------+
                                  |
+-------------------------------------------------------------------------+
|                         zapomni_db (Data Layer)                          |
|  +------------------+  +------------------+                              |
|  | FalkorDB Client  |  | Redis Client     |                              |
|  | (graph + vector) |  | (cache)          |                              |
|  +------------------+  +------------------+                              |
+-------------------------------------------------------------------------+
                                  |
+-------------------------------------------------------------------------+
|                         External Services                                |
|  +------------------+  +------------------+  +------------------+        |
|  | FalkorDB         |  | Redis            |  | Ollama           |        |
|  | (Docker)         |  | (Docker)         |  | (local)          |        |
|  +------------------+  +------------------+  +------------------+        |
+-------------------------------------------------------------------------+
```

---

## Design Patterns

### Registry + Factory Pattern (Tree-sitter)
```python
# Parser Registry - singleton managing language parsers
class ParserRegistry:
    _instance = None
    _parsers: Dict[str, Parser] = {}

    @classmethod
    def get_parser(cls, language: str) -> Parser:
        if language not in cls._parsers:
            cls._parsers[language] = cls._create_parser(language)
        return cls._parsers[language]
```

**Used in**: `zapomni_core/treesitter/parser/`

### Strategy Pattern (Embedders)
```python
# Abstract embedder interface
class BaseEmbedder(ABC):
    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        pass

# Concrete implementations
class OllamaEmbedder(BaseEmbedder):
    async def embed(self, texts: List[str]) -> List[List[float]]:
        return await self._call_ollama_batch(texts)
```

**Used in**: `zapomni_core/embeddings/`

### Singleton Pattern (Services)
```python
# Single instance of memory processor per server
class MemoryProcessor:
    _instance = None

    @classmethod
    def get_instance(cls, config: Config) -> "MemoryProcessor":
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance
```

**Used in**: `zapomni_core/memory_processor.py`

### Repository Pattern (Database)
```python
# Abstract data access
class MemoryRepository:
    async def save(self, memory: Memory) -> str: ...
    async def find_by_id(self, id: str) -> Memory: ...
    async def search(self, query: SearchQuery) -> List[Memory]: ...
```

**Used in**: `zapomni_db/clients/`

---

## Key Components

### 1. Memory Processor (`memory_processor.py`)
- Main orchestrator for memory operations
- Coordinates chunking, embedding, and storage
- Handles workspace isolation
- Integrates embedding cache

### 2. Tree-sitter Module (`treesitter/`)
- Language-agnostic AST parsing
- 41 language support via language pack
- Extractor framework for code elements
- Ready for PythonExtractor, TypeScriptExtractor

### 3. Embedding Service (`embeddings/`)
- Ollama integration with batch API
- Redis-backed embedding cache
- In-memory fallback cache
- Semantic similarity scoring

### 4. MCP Server (`server.py`)
- 17 MCP tools exposed
- stdio and SSE transport support
- Workspace management
- Session state handling

---

## Code Quality Standards

### Linting & Formatting
```yaml
tools:
  - black          # Code formatting
  - isort          # Import sorting
  - flake8         # Linting
  - mypy           # Type checking (strict mode)

mypy_config:
  strict: true
  disallow_untyped_defs: true
  disallow_any_generics: true
```

### Pre-commit Hooks
```yaml
repos:
  - repo: local
    hooks:
      - id: black
      - id: isort
      - id: flake8
      - id: mypy
      - id: pytest
```

### Testing Strategy
```
tests/
├── unit/           # 2089+ fast unit tests
│   ├── core/       # Core logic tests
│   ├── mcp/        # MCP tool tests
│   └── db/         # Database client tests
├── integration/    # Service integration tests
└── e2e/            # 88 end-to-end tests
```

**Test Framework**: pytest
**Coverage Target**: >80%
**TDD Approach**: Write tests first for new features

---

## Dependencies

### Core Dependencies
```toml
[project.dependencies]
python = ">=3.11"
falkordb = ">=1.0.0"
redis = ">=5.0.0"
httpx = ">=0.27.0"
pydantic = ">=2.0.0"
mcp = ">=1.0.0"
```

### Tree-sitter Dependencies
```toml
tree-sitter = ">=0.25.0"
tree-sitter-language-pack = ">=0.13.0"
```

### Development Dependencies
```toml
[project.optional-dependencies.dev]
pytest = ">=8.0.0"
pytest-asyncio = ">=0.23.0"
pytest-cov = ">=4.0.0"
black = ">=24.0.0"
mypy = ">=1.8.0"
```

---

## Configuration

### Environment Variables
```bash
# Database
FALKORDB_HOST=localhost
FALKORDB_PORT=6379

# Cache
REDIS_HOST=localhost
REDIS_PORT=6380
REDIS_ENABLED=true

# Ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=nomic-embed-text

# Server
MCP_TRANSPORT=stdio  # or "sse"
LOG_LEVEL=INFO
```

### Config File (`config.yaml`)
```yaml
embeddings:
  model: nomic-embed-text
  dimension: 768
  batch_size: 100
  enable_semantic_cache: true

chunking:
  max_chunk_size: 1000
  overlap: 100

graph:
  max_depth: 5
  relationship_threshold: 0.7
```

---

## Deployment

### Docker Compose (Development)
```yaml
services:
  falkordb:
    image: falkordb/falkordb:latest
    ports:
      - "6379:6379"

  redis:
    image: redis:7-alpine
    ports:
      - "6380:6379"
```

### Production Recommendations
- Use managed FalkorDB or Redis Enterprise
- Run Ollama on GPU-enabled instance
- Configure proper resource limits
- Enable TLS for all connections

---

## Performance Optimizations

### Session #12 Improvements
1. **Ollama Batch API**: Single request for multiple embeddings
2. **Redis Embedding Cache**: Skip re-computation for known texts
3. **In-Memory Fallback**: Works without Redis
4. **Semantic Cache**: Enabled by default

### Future Optimizations (v0.9.0)
- Incremental indexing
- Connection pooling
- Query result caching
- Parallel chunk processing

---

## Security Considerations

### Local-First Principle
- No external API calls for core functionality
- All data stays on local machine
- Ollama runs locally

### Best Practices
- No secrets in code
- Environment variable configuration
- Input validation on all MCP tools
- Workspace isolation for multi-tenant use

---

## Troubleshooting

### Common Issues

**FalkorDB connection refused**
```bash
docker compose up -d falkordb
```

**Ollama not responding**
```bash
ollama serve
ollama pull nomic-embed-text
```

**Tests failing**
```bash
make docker-up   # Start services
make test        # Run tests
```

---

## References

- [MCP Protocol Specification](https://modelcontextprotocol.io/)
- [Tree-sitter Documentation](https://tree-sitter.github.io/)
- [FalkorDB Documentation](https://docs.falkordb.com/)
- [Ollama API Reference](https://ollama.ai/docs/)
