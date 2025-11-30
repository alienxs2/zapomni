# Zapomni Project - AI Agent Handoff

**Last Updated**: 2025-11-30 (Session #25)
**Project Status**: v0.7.0 COMPLETE | Issue #26 COMPLETE | mypy 100% CLEAN
**Version**: v0.7.0
**Branch**: `main`

---

## START HERE (New AI Agent)

### Current State Summary

**Session #25: Hybrid Search with RRF Fusion!**

Major achievement:
- **Issue #26 Hybrid Search with RRF Fusion**: COMPLETE
- Added `fusion/` module (RRF, RSF, DBSF algorithms)
- Added `evaluation/` module (MRR, NDCG@K, Recall@K)
- True parallel execution with `asyncio.gather()`
- **139 new tests**, all passing
- **2640 total unit tests** passing
- **v0.7.0 Search Excellence: COMPLETE**

**CI/CD Status:**
| Workflow | Status |
|----------|--------|
| **Build & Package** | **SUCCESS** |
| **Lint & Code Quality** | **SUCCESS** (mypy: 0 errors!) |
| Tests | **SUCCESS** (2640 passed) |

**Test Status:**
- Unit Tests: **2640 passed**, 11 skipped
- Integration Tests: **115 passed** (51 skip in CI)
- E2E Tests: 88 passed, 1 xfailed

---

## NEXT STEPS

### Priority 1: v0.8.0 - Knowledge Graph 2.0

#### Issue #27 - Bi-temporal model

```bash
gh issue view 27
```

Key areas:
1. **Valid time vs transaction time**
2. **Historical queries**
3. **Time-travel debugging**

### Other milestones
- v0.9.0: #28 Support 100k+ files
- v1.0.0: #29 Web UI, #30 Documentation

---

## Reference Files

```
src/zapomni_core/search/
├── bm25_search.py        # Enhanced with bm25s (Session #24)
├── bm25_tokenizer.py     # CodeTokenizer (Session #24)
├── vector_search.py      # Vector search
├── hybrid_search.py      # Parallel + fusion (Session #25)
├── reranker.py           # Cross-encoder
├── fusion/               # NEW: Fusion algorithms (Session #25)
│   ├── rrf.py            # Reciprocal Rank Fusion
│   ├── rsf.py            # Relative Score Fusion
│   └── dbsf.py           # Distribution-Based Score Fusion
└── evaluation/           # NEW: Metrics (Session #25)
    └── metrics.py        # MRR, NDCG@K, Recall@K
```

### How to Start

```bash
cd /home/dev/zapomni
git pull origin main
source .venv/bin/activate
make test                     # Should see 2640 passed
mypy src/                     # Should see 0 errors!
gh run list --limit 5         # Check CI status
```

---

## PROJECT MANAGEMENT

### SHASHKA System

```
.shashka/
├── state/
│   ├── HANDOFF.md        # Session handoff (START HERE)
│   └── SNAPSHOT.md       # Project snapshot
├── log/
│   └── 2025-11-30-session-25.md  # Latest session log
└── config.yaml           # Project config
```

### Claude Slash Commands

| Command | Description |
|---------|-------------|
| /pm | Project management tasks |
| /dev | Development workflow |
| /review | Code review checklist |
| /test | Testing guidance |

---

## QUICK COMMANDS

```bash
# Development
make test                     # All unit tests (2640)
make lint                     # Run linter
make format                   # Format code
mypy src/                     # Type checking (0 errors!)

# Search tests
pytest tests/unit/search/     # Search tests (204)

# Services
make docker-up                # Start FalkorDB + Redis
make server                   # Start MCP server

# GitHub
gh run list --limit 5         # Check CI status
gh issue list --state open    # All open issues (5)
```

---

## ARCHITECTURE

```
src/
├── zapomni_core/
│   ├── search/             # Search module (v0.7.0 COMPLETE)
│   │   ├── bm25_search.py      # Enhanced with bm25s
│   │   ├── bm25_tokenizer.py   # CodeTokenizer
│   │   ├── vector_search.py    # Vector search
│   │   ├── hybrid_search.py    # Parallel + fusion
│   │   ├── fusion/             # RRF, RSF, DBSF
│   │   └── evaluation/         # MRR, NDCG, Recall
│   ├── treesitter/         # Tree-sitter module
│   │   ├── extractors/     # Language extractors
│   │   └── analyzers/      # Code analysis
│   └── memory_processor.py
├── zapomni_mcp/
│   └── tools/
│       └── index_codebase.py  # INTEGRATION POINT
└── zapomni_db/
```

---

## SESSION HISTORY

| Session | Date | Focus | Result |
|---------|------|-------|--------|
| **#25** | 2025-11-30 | Issue #26 Hybrid | **RRF/RSF/DBSF fusion, 139 tests** |
| #24 | 2025-11-30 | Issue #25 BM25 | bm25s + CodeTokenizer, 65 tests |
| #23 | 2025-11-29 | mypy cleanup | 141->0 errors, 9 issues closed |
| #22 | 2025-11-29 | mypy + Integration | 64 mypy fixed |
| #21 | 2025-11-29 | CI/CD Fixes | Build SUCCESS |
| #20 | 2025-11-29 | Issue #24 | CallGraphAnalyzer (74 tests) |
| #19 | 2025-11-29 | Issue #23 | RustExtractor (55 tests) |
| #18 | 2025-11-28 | Issue #22 | GoExtractor (55 tests) |

---

## CONTACTS

- **Repository**: https://github.com/alienxs2/zapomni
- **Issues**: https://github.com/alienxs2/zapomni/issues
- **Owner**: Goncharenko Anton (alienxs2)

---

**v0.7.0 Search Excellence COMPLETE! Next: v0.8.0 Knowledge Graph 2.0 (Issue #27).**
