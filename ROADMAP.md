# Roadmap

This document outlines the development roadmap and current status of Zapomni.

## Current Status

**Version**: v0.3.1 (2025-11-28)
**Stage**: Alpha - Production-ready core functionality

| Metric | Value |
|--------|-------|
| MCP Tools | 17 registered |
| Unit Tests | 1,868 passed, 11 skipped |
| E2E Tests | **88 passed, 1 xfailed** |
| Total Tests | **~1,957** |
| Coverage | 74-89% (module-dependent) |
| Test Runtime | ~37 seconds (unit) |
| Python Files | 80 |
| Lines of Code | ~29,500 |

### Feature Status

```mermaid
pie title Feature Maturity
    "Stable" : 8
```

| Feature | Status | Flag |
|---------|--------|------|
| Memory Operations | Stable | - |
| Hybrid Search | Stable | `ENABLE_HYBRID_SEARCH=true` |
| Knowledge Graph | Stable | `ENABLE_KNOWLEDGE_GRAPH=true` |
| Code Indexing | Stable | `ENABLE_CODE_INDEXING=true` |
| Semantic Cache | Stable | `ENABLE_SEMANTIC_CACHE=true` |
| Git Hooks | Stable | - |
| Workspaces | Stable | - |
| E2E Testing | Stable | 115 tests |

## Version History

### v0.3.x - Performance & Bug Fixes

| Version | Date | Highlights |
|---------|------|------------|
| v0.3.1 | 2025-11-28 | **Fix**: index_codebase now stores file content (Issue #2) |
| v0.3.0 | 2025-11-27 | Performance benchmarking, E2E tests complete |

### v0.2.x - Foundation Complete

| Version | Date | Highlights |
|---------|------|------------|
| v0.2.2 | 2025-11-27 | Feature flags connected to code, enabled by default |
| v0.2.1 | 2025-11-27 | 4 missing tools registered, 95+ test fixes |
| v0.2.0 | 2025-11-25 | Phase 2 (Enhanced Search) + Phase 3 (Code Intelligence) |

### v0.1.x - Initial Release

| Version | Date | Highlights |
|---------|------|------------|
| v0.1.0 | 2025-11-24 | Initial release: 17 MCP tools, 2135 tests, Git Hooks |

## Roadmap

```mermaid
gantt
    title Zapomni Development Roadmap
    dateFormat  YYYY-MM
    section Released
    v0.1.0 Initial Release     :done, v01, 2025-11, 1d
    v0.2.x Foundation          :done, v02, 2025-11, 3d
    v0.3.x Performance         :done, v03, 2025-11, 1d
    section In Progress
    v0.4.0 Tree-sitter AST     :active, v04, 2025-11, 2025-12
    section Planned
    v0.5.0 Multi-language      :v05, 2025-05, 2025-06
    v0.6.0 Transports          :v06, 2025-07, 2025-09
    v1.0.0 Stable Release      :milestone, v10, 2025-10, 2025-12
```

### v0.3.0 - Performance & Stability

**Target**: Q1 2025
**Focus**: Production hardening

**Planned**:
- [ ] Performance benchmarking and optimization
- [ ] Query caching and index tuning
- [ ] Memory usage optimization for large graphs
- [ ] Connection pooling improvements
- [ ] Load testing validation (Locust)

**KPIs**:
- Search latency < 200ms (P95)
- Support 100K+ memories
- Concurrent requests: 8+

### v0.4.0 - Tree-sitter AST Integration

**Target**: Q4 2025 (Active Development)
**Focus**: Advanced code intelligence with AST parsing
**Issue**: [#5](https://github.com/alienxs2/zapomni/issues/5)

**Architecture Decisions**:
- Integration: Full replacement of `index_codebase` (Breaking Change)
- Granularity: Hybrid (file + top-level elements separately)
- Patterns: Registry + Factory for extensibility
- Fallback: GenericExtractor for all 165+ languages

**Foundation Phase (F1-F11)**:
- [ ] F1: Add tree-sitter dependencies
- [ ] F2: Create models.py (ExtractedCode, ASTNodeLocation, etc.)
- [ ] F3: Create exceptions.py (TreeSitterError hierarchy)
- [ ] F4: Create config.py (165+ language mappings)
- [ ] F5: Create parser/base.py (BaseLanguageParser ABC)
- [ ] F6: Create parser/registry.py (LanguageParserRegistry Singleton)
- [ ] F7: Create parser/factory.py (ParserFactory with lazy init)
- [ ] F8: Create extractors/base.py (BaseCodeExtractor ABC)
- [ ] F9: Create extractors/generic.py (GenericExtractor fallback)
- [ ] F10: Unit tests (~115 new tests)
- [ ] F11: Documentation updates

**Next Phases**:
- [ ] Python extractor (semantic extraction)
- [ ] JavaScript/TypeScript extractors
- [ ] Integration with index_codebase tool
- [ ] E2E tests for AST functionality

**KPIs**:
- 165+ languages supported (via tree-sitter-language-pack)
- Parse time < 100ms per file
- ~115 new unit tests

### v0.5.0 - Multi-language & i18n

**Target**: Q2 2025
**Focus**: International support

**Planned**:
- [ ] Multi-language embeddings (multilingual-e5)
- [ ] Non-English entity extraction
- [ ] Language detection
- [ ] Cross-language search

**KPIs**:
- Support 10+ languages
- Cross-language search accuracy > 80%

### v0.6.0 - Advanced Transports

**Target**: Q3 2025
**Focus**: Deployment flexibility

**Planned**:
- [ ] HTTP/SSE transport (production-ready)
- [ ] WebSocket support
- [ ] gRPC transport option
- [ ] Docker image optimization
- [ ] Kubernetes deployment manifests

**KPIs**:
- < 100ms transport overhead
- Container image < 500MB

### v1.0.0 - Stable Release

**Target**: Q4 2025
**Focus**: Production stability

**Requirements for v1.0**:
- [ ] API stability guarantee
- [ ] Comprehensive documentation
- [ ] Migration guides
- [ ] Security audit complete
- [ ] Performance benchmarks published
- [ ] 90%+ test coverage
- [ ] Zero critical/high CVEs

## Future Considerations (v1.x+)

These features are under consideration for post-1.0 releases:

### Shadow Documentation
Automatic documentation generation from code changes:
- Track code modifications
- Generate change summaries
- Build documentation graphs

### Cross-Project Intelligence
Link knowledge across multiple repositories:
- Shared entity resolution
- Cross-project search
- Dependency analysis

### Real-time Collaboration
Multi-user memory sharing:
- Workspace permissions
- Change synchronization
- Conflict resolution

### Advanced Analytics
Insights and visualization:
- Usage analytics dashboard
- Knowledge graph visualization
- Memory access patterns

## Success Metrics

### Quality KPIs

| Metric | Current | Target (v1.0) |
|--------|---------|---------------|
| Test Coverage | 74-89% | 90%+ |
| Tests Passing | 99.7% | 100% |
| Critical Bugs | 0 | 0 |
| Documentation | Complete | Complete |

### Performance KPIs

| Metric | Current | Target (v1.0) |
|--------|---------|---------------|
| Search Latency (P95) | < 500ms | < 200ms |
| Memory Capacity | 10K+ | 100K+ |
| Concurrent Users | 4 | 8+ |
| Embedding Cache Hit | 60-68% | 80%+ |

### Adoption KPIs

| Metric | Current | Target (v1.0) |
|--------|---------|---------------|
| GitHub Stars | - | 100+ |
| Active Users | - | 50+ |
| Community PRs | - | 10+ |

## Contributing to the Roadmap

We welcome community input on the roadmap! To suggest features or changes:

1. Open a [GitHub Discussion](https://github.com/alienxs2/zapomni/discussions) for feature ideas
2. Create an [Issue](https://github.com/alienxs2/zapomni/issues) for specific proposals
3. Submit a PR for documentation or implementation

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

**Last Updated**: 2025-11-28
