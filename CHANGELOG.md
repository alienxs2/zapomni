# Changelog

All notable changes to Zapomni will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned

#### Phase 4+: Advanced Features (Q3+ 2025)
- Multi-language embeddings support
- Additional document formats (video transcripts, audio)
- Performance optimization (query caching, index tuning)
- HTTP/SSE transport alternatives
- Configuration profiles (dev/staging/prod)
- Secrets management integration
- Real-time collaboration features
- Advanced analytics dashboard

## [0.2.0] - 2025-11-24

### Phase 2: Enhanced Search - COMPLETE

**Added:**
- `build_graph` tool - Extract entities and build knowledge graph
  - SpaCy NER + LLM entity extraction
  - Graph construction with relationships
  - Entity confidence scoring
- `get_related` tool - Find related entities via graph traversal
  - 1-5 hop depth traversal
  - Relationship type filtering
  - Score-based ranking
- `graph_status` tool - View knowledge graph statistics
  - Node and edge counts
  - Entity type breakdown
  - Graph health indicators

**Core Implementation:**
- EntityExtractor with SpaCy + normalization
- GraphBuilder for entity relationships
- Graph traversal in FalkorDB
- Semantic caching with Redis
- Feature flags enabled: enable_extraction, enable_graph

**Tests:**
- 115 Phase 2 tool tests
- Integration test for full workflow
- No regressions in Phase 1

### Phase 3: Code Intelligence - COMPLETE

**Added:**
- `export_graph` tool - Export knowledge graph in 4 formats
  - GraphML (XML) for Gephi, yEd
  - Cytoscape JSON for web visualization
  - Neo4j Cypher for database import
  - Simple JSON for backup
- `index_codebase` tool - Index code repositories with AST
  - 14+ programming languages
  - Function and class extraction
  - Call graph analysis
- `delete_memory` tool - Delete specific memory with confirmation
  - UUID validation
  - Safety confirmation required
  - Audit logging
- `clear_all` tool - Clear all memories with strict phrase match
  - Requires EXACT phrase: "DELETE ALL MEMORIES"
  - Case-sensitive validation
  - Comprehensive logging

**Core Implementation:**
- GraphExporter with 4 format support (GraphML, Cytoscape, Neo4j, JSON)
- RepositoryIndexer for 14+ languages
- AST-based code chunking
- Delete operations with safety mechanisms

**Tests:**
- 155 Phase 3 tool tests
- Integration test for workflows
- Safety mechanism verification

**Total Added:**
- 7 new MCP tools (3 Phase 2 + 4 Phase 3)
- 1 core component (GraphExporter)
- 270+ unit tests
- 2 integration test suites
- All tests passing (>95% pass rate)

## [0.1.0] - 2025-11-24

### Initial Alpha Release

This is the first release of Zapomni, establishing the core foundation for local-first AI memory with the Model Context Protocol (MCP).

### Added

#### Core Infrastructure
- Project structure and package organization
- Python 3.10+ support with comprehensive type hints
- MIT License and open-source governance
- Contributing guidelines and community standards

#### MCP Server (Phase 1)
- Full MCP stdio transport implementation
- Three core MCP tools:
  - `add_memory` - Store text with automatic chunking and embedding
  - `search_memory` - Semantic search across stored memories
  - `get_stats` - View memory statistics (total memories, chunks, size)
- Request/response validation with Pydantic
- Proper error handling and MCP error responses
- Configuration management via environment variables

#### Database Layer
- FalkorDB integration for unified vector + graph storage
- Memory model with embeddings and metadata
- CRUD operations for memory persistence
- Redis caching layer infrastructure (prepared for Phase 2)
- Database initialization and schema setup

#### Embeddings Engine
- Ollama integration for local embeddings
- Nomic-embed-text (768-dim) as default model
- Embedding caching for performance optimization
- Fallback embedding generation mechanisms
- Configurable embedding parameters

#### Document Processing
- Smart document chunking (configurable size and overlap)
- Multi-format input support (plain text, JSON, structured data)
- Metadata preservation and enrichment
- Chunk validation and error handling
- Configurable chunk parameters (256-512 tokens recommended)

#### Search Capabilities
- Vector similarity search with HNSW index
- Configurable search parameters (top_k, similarity thresholds)
- Score normalization and result ranking
- Support for multi-chunk results with context
- Search result pagination ready

#### Testing Infrastructure
- Comprehensive test suite structure
- Unit tests (70% coverage target)
- Integration tests with services
- End-to-end test examples
- Test fixtures and mocking utilities
- PyTest configuration and plugins
- 80%+ code coverage target

#### Code Quality Tools
- Black code formatter (100 char line length)
- isort import sorting (Black profile)
- Flake8 linting configuration
- MyPy strict type checking
- Pre-commit hooks for automated checks
- GitHub Actions CI/CD pipeline
- Code coverage reporting

#### Documentation
- Comprehensive README.md with quick start
- API documentation in docstrings
- Architecture overview and design decisions
- Configuration guide with environment variables
- Development workflow documentation
- MCP tool usage examples
- Contributing guidelines
- Code of Conduct (Contributor Covenant 2.1)

#### Development Experience
- Development mode with auto-reload ready
- Docker Compose for service orchestration
- Environment configuration templates
- Local development instructions
- Pre-commit hooks for code quality
- Multiple test execution modes

### Fixed
- N/A (initial release)

### Changed
- N/A (initial release)

### Deprecated
- N/A (initial release)

### Removed
- N/A (initial release)

### Security
- No external API dependencies (local-first)
- All data processing on user's machine
- No telemetry or data collection
- No authentication required for local use
- Secure environment variable handling

### Known Issues

#### FalkorDB Compatibility
- Limited testing with FalkorDB 4.0+ versions
- Some edge cases in complex graph traversals not yet covered
- Graph schema validation may need refinement for Phase 2

#### Performance Baseline
- Search latency targets not yet benchmarked against competing systems
- Concurrent request handling (4 max) is conservative for initial release
- Memory usage not yet optimized for large knowledge graphs

#### Feature Limitations
- Phase 1 focuses on vector search only (keyword search in Phase 2)
- No entity extraction in Phase 1 (available in Phase 2)
- No knowledge graph construction yet (Phase 2 feature)
- No code indexing capabilities (Phase 3 feature)
- Graph traversal not implemented (Phase 2)

#### Operational
- No persistence of application state between restarts
- Limited monitoring and observability in Phase 1
- No backup/restore functionality yet
- Docker services require manual management

### Performance

#### Targets (MVP)
- **Search Latency:** < 500ms (P95)
- **Embedding Cache Hit Rate:** 60-68%
- **Chunk Size:** 256-512 tokens
- **Vector Dimensions:** 768 (nomic-embed-text)
- **Concurrent Tasks:** 4 (configurable)

#### Baseline Metrics
- Memory storage: Tested with documents up to 100KB
- Embedding generation: ~50-100ms per document (via Ollama)
- Vector search: <100ms for <10K memories
- Concurrent requests: Stable up to 4 parallel requests

### Roadmap

#### Phase 2: Enhanced Search (Q4 2025)
- [ ] BM25 keyword search implementation
- [ ] Hybrid search with reciprocal rank fusion
- [ ] SpaCy NER for entity extraction
- [ ] LLM-based entity enhancement
- [ ] Knowledge graph construction from entities
- [ ] Semantic caching with Redis
- [ ] Graph traversal queries

#### Phase 3: Code Intelligence (Q1 2026)
- [ ] Code repository indexing
- [ ] AST-based code chunking
- [ ] Function/class entity extraction
- [ ] Call graph analysis
- [ ] Code-specific search

#### Phase 4+: Advanced Features (Future)
- [ ] Multi-language support
- [ ] Document format expansion (PDF, Markdown, Office)
- [ ] Performance optimization
- [ ] HTTP/SSE transport alternatives
- [ ] Configuration profiles
- [ ] Secrets management integration

### Contributors

- **Goncharenko Anton** (alienxs2) - Author & Maintainer

### Dependencies

#### Core
- `python` >= 3.10
- `pydantic` >= 2.5.0 - Data validation
- `pydantic-settings` >= 2.0.0 - Configuration management
- `anthropic` >= 0.25.0 - MCP protocol and SDK

#### Database & Storage
- `redis` >= 5.0.0 - Caching layer
- `falkordb` >= 4.0.0 - Vector + graph database

#### Embeddings & LLM
- `ollama-python` >= 0.1.0 - Ollama integration
- `spacy` >= 3.7.0 - NLP (prepared for Phase 2)

#### Search & Ranking
- `rank-bm25` >= 0.2.2 - BM25 implementation (prepared for Phase 2)

#### Development
- `pytest` >= 7.4.0 - Testing framework
- `pytest-cov` >= 4.1.0 - Coverage reporting
- `black` >= 24.1.0 - Code formatting
- `isort` >= 5.13.0 - Import sorting
- `flake8` >= 7.0.0 - Linting
- `mypy` >= 1.8.0 - Type checking
- `pre-commit` >= 3.6.0 - Git hooks

### Links

- **Repository:** https://github.com/alienxs2/zapomni
- **Issues:** https://github.com/alienxs2/zapomni/issues
- **Discussions:** https://github.com/alienxs2/zapomni/discussions
- **License:** MIT

---

## Template for Future Releases

```markdown
## [X.Y.Z] - YYYY-MM-DD

### Added
- Specific feature descriptions

### Changed
- Behavioral changes

### Fixed
- Bug fixes with references

### Deprecated
- Features being phased out

### Removed
- Removed functionality

### Security
- Security fixes and improvements

### Known Issues
- Documented limitations

### Performance
- Performance improvements or regressions
```

---

**Zapomni v0.1.0** - Building the foundation for local-first AI memory.
