# Project Snapshot

**Project**: Zapomni
**Version**: v0.7.0 (v0.8.0 in progress)
**Status**: Issue #27 Phase 1 COMPLETE | mypy 100% CLEAN
**Last Updated**: 2025-12-01 (Session #26)

## Quick Stats

| Metric | Value |
|--------|-------|
| Unit Tests | 2688 passed, 11 skipped |
| Integration Tests | 115 passed (51 skipped in CI) |
| E2E Tests | 88 passed, 1 xfailed |
| Tree-sitter | 41 languages, 449 tests |
| PythonExtractor | 58 tests |
| TypeScriptExtractor | 60 tests |
| GoExtractor | 55 tests |
| RustExtractor | 55 tests |
| CallGraphAnalyzer | 74 tests |
| BM25Search | 65 tests |
| HybridSearch + Fusion | 139 tests |
| **Bi-temporal Models** | **48 tests** (NEW!) |
| **mypy errors** | **0** |
| Known Bugs | **0 remaining** |
| Open Issues | 9 (5 features + 4 phase issues) |

## CI/CD Status

| Workflow | Status | Notes |
|----------|--------|-------|
| **Build & Package** | **SUCCESS** | All green |
| **Lint & Code Quality** | **SUCCESS** | mypy: 0 errors |
| **Tests** | **SUCCESS** | 2688 unit tests |

## Session #26 Summary - Bi-temporal Model Phase 1!

**Issue #27 Phase 1: Schema & Models - COMPLETE**

### Files Created
| File | Purpose |
|------|---------|
| `src/zapomni_db/migrations/__init__.py` | Migration module exports |
| `src/zapomni_db/migrations/migration_001_bitemporal.py` | Bi-temporal migration script |
| `tests/unit/db/test_models_temporal.py` | 26 tests for temporal models |
| `tests/unit/db/test_migration_bitemporal.py` | 22 tests for migration |
| `.shashka/specs/issue-27-bitemporal/` | Full specification (4 files) |

### Files Modified
| File | Changes |
|------|---------|
| `src/zapomni_db/models.py` | Added MemoryVersion, TemporalQuery, Entity temporal fields |
| `src/zapomni_db/schema_manager.py` | Version 2.0.0, 4 new temporal indexes |
| `tests/unit/test_schema_manager.py` | Updated for new schema version |
| `CHANGELOG.md` | Added Phase 1 entry |

### New Features
1. **MemoryVersion model** - Full bi-temporal support with valid_from/to, transaction_to
2. **TemporalQuery model** - Parameters for point-in-time and history queries
3. **Entity temporal fields** - valid_from, valid_to, is_current
4. **Migration script** - Idempotent migration for existing data
5. **Schema 2.0.0** - 4 new indexes for temporal queries

---

## v0.8.0 Progress - Knowledge Graph 2.0 (IN PROGRESS)

| Issue | Title | Status | Tests |
|-------|-------|--------|-------|
| #37 | Phase 1: Schema & Models | **COMPLETE** | 48 |
| #38 | Phase 2: Database Layer | NEXT | - |
| #39 | Phase 3: Git Integration | Pending | - |
| #40 | Phase 4: MCP Tools | Pending | - |
| #41 | Phase 5: Documentation & Release | Pending | - |

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
│   │   │   ├── fusion/             # RRF, RSF, DBSF
│   │   │   └── evaluation/         # MRR, NDCG, Recall
│   │   └── memory_processor.py
│   ├── zapomni_mcp/        # MCP server (17 tools)
│   ├── zapomni_db/         # FalkorDB + Redis clients
│   │   ├── models.py       # Data models (+ bi-temporal!)
│   │   ├── schema_manager.py # Schema v2.0.0
│   │   └── migrations/     # NEW: Migration scripts
│   └── zapomni_cli/        # CLI tools + Git hooks
├── .github/workflows/      # CI/CD (All green!)
│   ├── build.yml           # Build & Package
│   ├── lint.yml            # Lint & Code Quality
│   └── tests.yml           # Tests
├── .shashka/               # Project management
│   ├── specs/issue-27-bitemporal/  # Bi-temporal spec
│   └── state/              # Handoff files
└── tests/                  # 2688+ unit tests
    ├── unit/
    │   ├── db/             # DB tests (+ temporal!)
    │   └── search/         # Search tests
    └── integration/        # 115 tests
```

## Roadmap to v1.0

| Milestone | Focus | Status |
|-----------|-------|--------|
| Bug Fixing | 8 bugs | **COMPLETE** |
| v0.5.0 | Solid Foundation | **COMPLETE** |
| v0.6.0 | Code Intelligence | **COMPLETE** |
| v0.7.0 | Search Excellence | **COMPLETE** |
| v0.8.0 | Knowledge Graph 2.0 | **IN PROGRESS** (Phase 1 done) |
| v0.9.0 | Scale & Performance | Planned |
| v1.0.0 | Production Ready | Target |

## Open Issues (9)

| Issue | Title | Milestone |
|-------|-------|-----------|
| #27 | Bi-temporal model | v0.8.0 (parent) |
| #37 | Phase 1: Schema & Models | v0.8.0 **COMPLETE** |
| #38 | Phase 2: Database Layer | v0.8.0 **NEXT** |
| #39 | Phase 3: Git Integration | v0.8.0 |
| #40 | Phase 4: MCP Tools | v0.8.0 |
| #41 | Phase 5: Documentation | v0.8.0 |
| #28 | Support 100k+ files indexing | v0.9.0 |
| #29 | Web UI Dashboard | v1.0.0 |
| #30 | Complete documentation | v1.0.0 |

## Key Documents

| Document | Path | Purpose |
|----------|------|---------|
| Product Strategy | `.shashka/steering/product.md` | Vision, positioning |
| Technical Spec | `.shashka/steering/tech.md` | Architecture, stack |
| Project Structure | `.shashka/steering/structure.md` | Code organization |
| **Bi-temporal Spec** | `.shashka/specs/issue-27-bitemporal/` | **Full implementation plan** |
| Resume Prompt | `.claude/resume-prompt.md` | AI agent handoff |
| Session Handoff | `.shashka/state/HANDOFF.md` | Session details |
