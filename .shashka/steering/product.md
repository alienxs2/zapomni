# Zapomni - Product Strategy

**Last Updated**: 2025-11-28
**Version**: v0.4.0 Foundation

---

## Vision

> **"The AI memory that truly understands your code"**

Zapomni is a local-first MCP memory server designed to give AI agents persistent, intelligent memory with deep code understanding capabilities.

---

## Unique Value Proposition

### 1. Deep Code Intelligence (41 Languages)
- **Tree-sitter AST parsing** for 41 programming languages
- Extracts functions, classes, methods, imports with full context
- Understands code structure, not just text
- **Unique on market** - no competitor offers this depth

### 2. 100% Local-First Architecture
- All data stays on your machine
- No cloud dependencies for core functionality
- Complete privacy for sensitive codebases
- Ideal for enterprise and security-conscious developers

### 3. Knowledge Graph Foundation
- FalkorDB-powered graph storage
- Semantic relationships between code elements
- Vector embeddings via Ollama (local LLMs)
- Hybrid search (vector + graph traversal)

---

## Target Users

### Primary
- **AI-assisted developers** using Claude Code, Cursor, Windsurf, Cline
- **Enterprise developers** requiring local-first solutions
- **Open source maintainers** managing large codebases

### Secondary
- **DevOps engineers** automating code analysis
- **Technical writers** documenting codebases
- **Security researchers** analyzing code patterns

---

## Competitive Landscape

| Feature | Zapomni | Mem0 | Zep | Cognee |
|---------|---------|------|-----|--------|
| **GitHub Stars** | ~100 | 43.6k | 23.8k | 9.3k |
| **Code Intelligence** | **41 languages** | No | No | ~10 languages |
| **100% Local** | **Yes** | No | No | Partial |
| **Tree-sitter AST** | **Yes** | No | No | No |
| **Knowledge Graph** | Yes | Yes | Yes | Yes |
| **MCP Protocol** | **Native** | No | No | No |
| **Temporal Model** | Planned | No | **Yes** | No |

### Key Competitors

1. **Mem0** (43.6k stars)
   - Strengths: Large community, good documentation
   - Weaknesses: Cloud-dependent, no code understanding

2. **Zep** (23.8k stars)
   - Strengths: Temporal memory model, enterprise focus
   - Weaknesses: Cloud-first, limited code support

3. **Cognee** (9.3k stars)
   - Strengths: Growing community, some code support
   - Weaknesses: ~10 languages only, not fully local

### Our Differentiation
- **41 language support** vs competitors' 0-10
- **Native MCP integration** for Claude ecosystem
- **100% local** vs cloud-dependent solutions
- **Tree-sitter foundation** for semantic code understanding

---

## Product Roadmap

### Phase 1: Solid Foundation (v0.5.0)
**Timeline**: 3-4 weeks
**Focus**: Bug fixes + Python/TypeScript extractors

- Fix 6 remaining bugs (#12-18)
- PythonExtractor with full AST support
- TypeScriptExtractor with full AST support
- Tree-sitter integration complete

### Phase 2: Code Intelligence (v0.6.0)
**Timeline**: 4-5 weeks
**Focus**: Multi-language support + intelligent indexing

- 10 most popular languages fully supported
- Cross-file reference tracking
- Dependency analysis
- Smart change detection

### Phase 3: Search Excellence (v0.7.0)
**Timeline**: 3-4 weeks
**Focus**: Advanced search capabilities

- Hybrid search (vector + graph)
- Code-aware ranking
- Semantic code search
- Natural language queries

### Phase 4: Knowledge Graph 2.0 (v0.8.0)
**Timeline**: 4-5 weeks
**Focus**: Deep relationships

- Call graph analysis
- Type hierarchy tracking
- Import/export relationships
- Cross-repository linking

### Phase 5: Scale & Performance (v0.9.0)
**Timeline**: 3-4 weeks
**Focus**: Production readiness

- Large repository support (1M+ files)
- Incremental indexing
- Memory optimization
- Benchmark suite

### Phase 6: Production Ready (v1.0.0)
**Timeline**: 4-5 weeks
**Focus**: Polish and launch

- Comprehensive documentation
- Migration tools
- Enterprise features
- Community launch

**Total timeline to v1.0: ~5-6 months**

---

## Success Metrics

### Technical KPIs
- Test coverage: >80%
- Indexing speed: <1min for 10k files
- Query latency: <100ms p95
- Memory usage: <2GB for 100k files

### Community KPIs
- GitHub stars: 1k by v1.0
- Weekly active users: 100+
- Discord community: 500+ members
- Documentation completeness: 100%

---

## Go-to-Market Strategy

### Phase 1: Developer Preview (v0.5-v0.7)
- Focus on early adopters
- Gather feedback via GitHub issues
- Iterate on core features

### Phase 2: Community Growth (v0.8-v0.9)
- Launch blog posts and tutorials
- Conference talks (AI/Developer conferences)
- Integration guides for popular tools

### Phase 3: Production Launch (v1.0)
- Product Hunt launch
- Hacker News announcement
- Partnership with Claude/Anthropic ecosystem

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Tree-sitter breaking changes | Medium | High | Pin versions, extensive tests |
| Ollama performance issues | Low | Medium | Caching layer, batch API |
| Competition from Mem0/Zep | High | Medium | Focus on code intelligence niche |
| MCP protocol changes | Low | High | Abstract MCP layer |

---

## Contact

- **Repository**: https://github.com/alienxs2/zapomni
- **Owner**: Goncharenko Anton (alienxs2)
- **Issues**: https://github.com/alienxs2/zapomni/issues
