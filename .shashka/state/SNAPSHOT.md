# Project Snapshot

**Project**: Zapomni
**Version**: v0.7.0
**Status**: v0.7.0 COMPLETE | Issue #26 COMPLETE | mypy 100% CLEAN
**Last Updated**: 2025-11-30 (Session #25)

## Quick Stats

| Metric | Value |
|--------|-------|
| Unit Tests | 2640 passed, 11 skipped |
| Integration Tests | 115 passed (51 skipped in CI) |
| E2E Tests | 88 passed, 1 xfailed |
| Tree-sitter | 41 languages, 449 tests |
| PythonExtractor | 58 tests |
| TypeScriptExtractor | 60 tests |
| GoExtractor | 55 tests |
| RustExtractor | 55 tests |
| CallGraphAnalyzer | 74 tests |
| BM25Search | 65 tests |
| **HybridSearch + Fusion** | **139 tests** (NEW!) |
| **mypy errors** | **0** |
| Known Bugs | **0 remaining** |
| Open Issues | 5 (features only) |

## CI/CD Status

| Workflow | Status | Notes |
|----------|--------|-------|
| **Build & Package** | **SUCCESS** | All green |
| **Lint & Code Quality** | **SUCCESS** | mypy: 0 errors |
| **Tests** | **SUCCESS** | 2640 unit tests |

## Session #25 Summary - Hybrid Search with RRF Fusion!

**Issue #26: Hybrid Search with RRF Fusion - COMPLETE**

### Files Created (fusion/ module)
| File | Purpose |
|------|---------|
| `src/zapomni_core/search/fusion/__init__.py` | Package exports |
| `src/zapomni_core/search/fusion/base.py` | Base fusion class |
| `src/zapomni_core/search/fusion/rrf.py` | Reciprocal Rank Fusion |
| `src/zapomni_core/search/fusion/rsf.py` | Relative Score Fusion |
| `src/zapomni_core/search/fusion/dbsf.py` | Distribution-Based Score Fusion |

### Files Created (evaluation/ module)
| File | Purpose |
|------|---------|
| `src/zapomni_core/search/evaluation/__init__.py` | Package exports |
| `src/zapomni_core/search/evaluation/metrics.py` | MRR, NDCG@K, Recall@K metrics |

### Files Modified
| File | Changes |
|------|---------|
| `src/zapomni_core/search/hybrid_search.py` | Parallel execution + fusion options |
| `src/zapomni_core/search/__init__.py` | Export fusion classes |
| `tests/unit/search/` | 139 new tests |

### New Features
1. **RRF (Reciprocal Rank Fusion)** - Configurable k parameter (default: 60)
2. **RSF (Relative Score Fusion)** - Score-based normalization fusion
3. **DBSF (Distribution-Based Score Fusion)** - 3-sigma normalization
4. **Parallel execution** - asyncio.gather() for true parallelism
5. **Evaluation metrics** - MRR, NDCG@K, Recall@K for search quality

---

## v0.7.0 Progress - Search Excellence (COMPLETE)

| Issue | Title | Status | Tests |
|-------|-------|--------|-------|
| #25 | BM25 Search Index | **COMPLETE** | 65 |
| #26 | Hybrid Search with RRF | **COMPLETE** | 139 |

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
│   │   ├── search/         # Search modules (v0.7.0 COMPLETE)
│   │   │   ├── bm25_search.py      # Enhanced with bm25s
│   │   │   ├── bm25_tokenizer.py   # CodeTokenizer
│   │   │   ├── vector_search.py    # Vector search
│   │   │   ├── hybrid_search.py    # Parallel + fusion
│   │   │   ├── reranker.py         # Cross-encoder
│   │   │   ├── fusion/             # NEW: Fusion algorithms
│   │   │   │   ├── rrf.py          # Reciprocal Rank Fusion
│   │   │   │   ├── rsf.py          # Relative Score Fusion
│   │   │   │   └── dbsf.py         # Distribution-Based Score Fusion
│   │   │   └── evaluation/         # NEW: Metrics
│   │   │       └── metrics.py      # MRR, NDCG@K, Recall@K
│   │   └── memory_processor.py
│   ├── zapomni_mcp/        # MCP server (17 tools)
│   ├── zapomni_db/         # FalkorDB + Redis clients
│   └── zapomni_cli/        # CLI tools + Git hooks
├── .github/workflows/      # CI/CD (All green!)
│   ├── build.yml           # Build & Package
│   ├── lint.yml            # Lint & Code Quality
│   └── tests.yml           # Tests
└── tests/                  # 2640+ unit tests
    ├── unit/
    │   └── search/         # Search tests (204 total)
    └── integration/        # 115 tests
```

## Roadmap to v1.0

| Milestone | Focus | Status |
|-----------|-------|--------|
| Bug Fixing | 8 bugs | **COMPLETE** |
| v0.5.0 | Solid Foundation | **COMPLETE** |
| v0.6.0 | Code Intelligence | **COMPLETE** |
| v0.7.0 | Search Excellence | **COMPLETE** |
| v0.8.0 | Knowledge Graph 2.0 | **NEXT** |
| v0.9.0 | Scale & Performance | Planned |
| v1.0.0 | Production Ready | Target |

## Open Issues (5)

| Issue | Title | Milestone |
|-------|-------|-----------|
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
