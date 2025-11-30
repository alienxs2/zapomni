# Zapomni Project - AI Agent Handoff

**Last Updated**: 2025-11-30 (Session #24)
**Project Status**: v0.7.0 IN PROGRESS | Issue #25 COMPLETE | mypy 100% CLEAN
**Version**: v0.7.0-dev
**Branch**: `main`

---

## START HERE (New AI Agent)

### Current State Summary

**Session #24: BM25 Search Enhanced!**

Major achievement:
- **Issue #25 BM25 Search**: COMPLETE
- Replaced `rank-bm25` with `bm25s` (100-500x faster)
- Added `CodeTokenizer` for code-aware tokenization
- **65 new tests**, all passing
- **2501 total unit tests** passing

**CI/CD Status:**
| Workflow | Status |
|----------|--------|
| **Build & Package** | **SUCCESS** |
| **Lint & Code Quality** | **SUCCESS** (mypy: 0 errors!) |
| Tests | **SUCCESS** (2501 passed) |

**Test Status:**
- Unit Tests: **2501 passed**, 11 skipped
- Integration Tests: **115 passed** (51 skip in CI)
- E2E Tests: 88 passed, 1 xfailed

---

## NEXT STEPS

### Priority 1: Issue #26 - Hybrid Search with RRF

```bash
gh issue view 26
```

Based on research from Session #24:
1. **True parallel execution** with `asyncio.gather()`
2. **Fusion method options**: RRF, RSF, DBSF
3. **Configurable RRF k parameter** (default: 60)
4. **DBSF implementation** (3-sigma normalization)
5. **Evaluation metrics**: MRR, NDCG@K, Recall@K

### Other milestones
- v0.8.0: #27 Bi-temporal model
- v0.9.0: #28 Support 100k+ files
- v1.0.0: #29 Web UI, #30 Documentation

---

## Reference Files

```
src/zapomni_core/search/
├── bm25_search.py        # Enhanced with bm25s (Session #24)
├── bm25_tokenizer.py     # NEW: CodeTokenizer (Session #24)
├── vector_search.py      # Vector search
├── hybrid_search.py      # Needs Issue #26
└── reranker.py           # Cross-encoder
```

### How to Start

```bash
cd /home/dev/zapomni
git pull origin main
source .venv/bin/activate
make test                     # Should see 2501 passed
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
│   └── 2025-11-30-session-24.md  # Latest session log
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
make test                     # All unit tests (2501)
make lint                     # Run linter
make format                   # Format code
mypy src/                     # Type checking (0 errors!)

# Search tests
pytest tests/unit/search/     # BM25 search tests (65)

# Services
make docker-up                # Start FalkorDB + Redis
make server                   # Start MCP server

# GitHub
gh run list --limit 5         # Check CI status
gh issue list --state open    # All open issues (6)
```

---

## ARCHITECTURE

```
src/
├── zapomni_core/
│   ├── search/             # Search module (v0.7.0 focus)
│   │   ├── bm25_search.py      # Enhanced with bm25s
│   │   ├── bm25_tokenizer.py   # CodeTokenizer
│   │   ├── vector_search.py    # Vector search
│   │   └── hybrid_search.py    # Issue #26
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
| **#24** | 2025-11-30 | Issue #25 BM25 | **bm25s + CodeTokenizer, 65 tests** |
| #23 | 2025-11-29 | mypy cleanup | 141→0 errors, 9 issues closed |
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

**Issue #25 complete! Next: Issue #26 Hybrid Search with RRF fusion.**
