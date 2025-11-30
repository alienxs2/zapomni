# Project Snapshot

**Project**: Zapomni
**Version**: v0.7.0-dev
**Status**: v0.7.0 IN PROGRESS | Issue #25 COMPLETE | mypy 100% CLEAN
**Last Updated**: 2025-11-30 (Session #24)

## Quick Stats

| Metric | Value |
|--------|-------|
| Unit Tests | 2501 passed, 11 skipped |
| Integration Tests | 115 passed (51 skipped in CI) |
| E2E Tests | 88 passed, 1 xfailed |
| Tree-sitter | 41 languages, 449 tests |
| PythonExtractor | 58 tests |
| TypeScriptExtractor | 60 tests |
| GoExtractor | 55 tests |
| RustExtractor | 55 tests |
| CallGraphAnalyzer | 74 tests |
| **BM25Search** | **65 tests** (NEW!) |
| **mypy errors** | **0** |
| Known Bugs | **0 remaining** |
| Open Issues | 6 (features only) |

## CI/CD Status

| Workflow | Status | Notes |
|----------|--------|-------|
| **Build & Package** | **SUCCESS** | All green |
| **Lint & Code Quality** | **SUCCESS** | mypy: 0 errors |
| **Tests** | **SUCCESS** | 2501 unit tests |

## Session #24 Summary - BM25 Search Enhanced!

**Issue #25: BM25 Search Index - COMPLETE**

### Files Created
| File | Purpose |
|------|---------|
| `src/zapomni_core/search/bm25_tokenizer.py` | CodeTokenizer class |
| `tests/unit/search/__init__.py` | Test package |
| `tests/unit/search/test_bm25_search.py` | 65 comprehensive tests |

### Files Modified
| File | Changes |
|------|---------|
| `src/zapomni_core/search/bm25_search.py` | bm25s + persistence |
| `src/zapomni_core/search/__init__.py` | Export CodeTokenizer |
| `pyproject.toml` | bm25s[full]>=0.2.0 |

### New Features
1. **bm25s library** - 100-500x faster than rank-bm25
2. **CodeTokenizer** - camelCase/snake_case/acronym splitting
3. **Persistence** - save_index() / load_index() with mmap
4. **BM25 variants** - lucene, robertson, bm25+, bm25l, atire
5. **Backward compatible** - All 29 original tests pass

---

## v0.7.0 Progress - Search Excellence

| Issue | Title | Status | Tests |
|-------|-------|--------|-------|
| #25 | BM25 Search Index | **COMPLETE** | 65 |
| #26 | Hybrid Search with RRF | **NEXT** | - |

---

## Architecture Overview

```
zapomni/
├── src/
│   ├── zapomni_core/       # Core memory processing
│   │   ├── embeddings/     # Ollama embeddings + cache
│   │   ├── treesitter/     # AST parsing (41 languages)
│   │   │   ├── extractors/ # Python, TS, Go, Rust
│   │   │   └── analyzers/  # Call graph analyzer
│   │   ├── search/         # Search modules
│   │   │   ├── bm25_search.py      # Enhanced with bm25s
│   │   │   ├── bm25_tokenizer.py   # NEW: CodeTokenizer
│   │   │   ├── vector_search.py    # Vector search
│   │   │   ├── hybrid_search.py    # Hybrid (Issue #26)
│   │   │   └── reranker.py         # Cross-encoder
│   │   └── memory_processor.py
│   ├── zapomni_mcp/        # MCP server (17 tools)
│   ├── zapomni_db/         # FalkorDB + Redis clients
│   └── zapomni_cli/        # CLI tools + Git hooks
├── .github/workflows/      # CI/CD (All green!)
│   ├── build.yml           # Build & Package
│   ├── lint.yml            # Lint & Code Quality
│   └── tests.yml           # Tests
└── tests/                  # 2501+ unit tests
    ├── unit/
    │   └── search/         # NEW: Search tests (65)
    └── integration/        # 115 tests
```

## Roadmap to v1.0

| Milestone | Focus | Status |
|-----------|-------|--------|
| Bug Fixing | 8 bugs | **COMPLETE** |
| v0.5.0 | Solid Foundation | **COMPLETE** |
| v0.6.0 | Code Intelligence | **COMPLETE** |
| v0.7.0 | Search Excellence | **IN PROGRESS** |
| v0.8.0 | Knowledge Graph 2.0 | Planned |
| v0.9.0 | Scale & Performance | Planned |
| v1.0.0 | Production Ready | Target |

## Open Issues (6)

| Issue | Title | Milestone |
|-------|-------|-----------|
| #26 | Hybrid search with RRF fusion | v0.7.0 |
| #27 | Bi-temporal model | v0.8.0 |
| #28 | Support 100k+ files indexing | v0.9.0 |
| #29 | Web UI Dashboard | v1.0.0 |
| #30 | Complete documentation | v1.0.0 |
| #1 | Featured on cursor.store | Info |

## Key Documents

| Document | Path | Purpose |
|----------|------|---------|
| Product Strategy | `.shashka/steering/product.md` | Vision, positioning |
| Technical Spec | `.shashka/steering/tech.md` | Architecture, stack |
| Project Structure | `.shashka/steering/structure.md` | Code organization |
| Resume Prompt | `.claude/resume-prompt.md` | AI agent handoff |
| Session Handoff | `.shashka/state/HANDOFF.md` | Session details |
