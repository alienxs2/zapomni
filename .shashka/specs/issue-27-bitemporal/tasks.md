# Issue #27: Bi-temporal Model - Implementation Tasks

## Overview

| Phase | Description | Estimate | Status |
|-------|-------------|----------|--------|
| Phase 1 | Schema & Models | 2 days | [ ] Pending |
| Phase 2 | Database Layer | 2 days | [ ] Pending |
| Phase 3 | Git Integration | 1 day | [ ] Pending |
| Phase 4 | MCP Tools | 2 days | [ ] Pending |
| Phase 5 | Documentation & Release | 1 day | [ ] Pending |
| **Total** | | **8 days** | |

---

## Phase 1: Schema & Models (2 days)

### Task 1.1: Update Memory Model
- [ ] Add `valid_from: str` field
- [ ] Add `valid_to: Optional[str]` field
- [ ] Add `transaction_to: Optional[str]` field
- [ ] Add `version: int = 1` field
- [ ] Add `previous_version_id: Optional[str]` field
- [ ] Add `is_current: bool = True` field
- [ ] Update docstrings

**File:** `src/zapomni_db/models.py`

### Task 1.2: Update Entity Model
- [ ] Add `valid_from: str` field
- [ ] Add `valid_to: Optional[str]` field
- [ ] Add `is_current: bool = True` field

**File:** `src/zapomni_db/models.py`

### Task 1.3: Create Migration Script
- [ ] Create migrations directory structure
- [ ] Implement `001_add_bitemporal.py`
- [ ] Add idempotent checks
- [ ] Add rollback support (if possible)

**File:** `src/zapomni_db/migrations/001_add_bitemporal.py`

### Task 1.4: Update Schema Manager
- [ ] Add temporal indexes definitions
- [ ] Update `ensure_schema()` method
- [ ] Add schema version check
- [ ] Run migration on version mismatch

**File:** `src/zapomni_db/schema_manager.py`

### Task 1.5: Unit Tests for Models
- [ ] Test Memory model with temporal fields
- [ ] Test Entity model with temporal fields
- [ ] Test default values
- [ ] Test serialization/deserialization

**File:** `tests/unit/db/test_models_temporal.py`

### Task 1.6: Unit Tests for Migration
- [ ] Test migration on empty DB
- [ ] Test migration on existing data
- [ ] Test idempotency (run twice)
- [ ] Test index creation

**File:** `tests/unit/db/test_migration_bitemporal.py`

---

## Phase 2: Database Layer (2 days)

### Task 2.1: Add Temporal Query Methods to FalkorDBClient
- [ ] `get_memory_at_time(workspace_id, file_path, as_of, time_type)`
- [ ] `get_memory_history(workspace_id, file_path, limit)`
- [ ] `get_changes(workspace_id, since, until, change_type, path_pattern, limit)`
- [ ] `close_version(memory_id, valid_to, transaction_to)`
- [ ] `create_new_version(previous, new_content, valid_from)`

**File:** `src/zapomni_db/falkordb_client.py`

### Task 2.2: Update CypherQueryBuilder
- [ ] Add `build_point_in_time_query()` method
- [ ] Add `build_history_query()` method
- [ ] Add `build_changes_query()` method
- [ ] Add temporal WHERE clause builders

**File:** `src/zapomni_db/cypher_query_builder.py`

### Task 2.3: Update Existing Methods for Backwards Compatibility
- [ ] Update `get_memory()` to use is_current filter
- [ ] Update `search_memories()` to use is_current filter
- [ ] Update `delete_memory()` to set transaction_to instead of DELETE
- [ ] Add optional `as_of` parameter to read methods

**File:** `src/zapomni_db/falkordb_client.py`

### Task 2.4: Unit Tests for Temporal Queries
- [ ] Test current state query
- [ ] Test point-in-time valid query
- [ ] Test point-in-time transaction query
- [ ] Test full bi-temporal query
- [ ] Test history query
- [ ] Test changes query

**File:** `tests/unit/db/test_falkordb_temporal.py`

### Task 2.5: Unit Tests for Version Management
- [ ] Test close_version
- [ ] Test create_new_version
- [ ] Test version chain integrity
- [ ] Test is_current flag updates

**File:** `tests/unit/db/test_version_management.py`

---

## Phase 3: Git Integration (1 day)

### Task 3.1: Create GitTemporalExtractor Class
- [ ] Implement `is_git_repo()` method
- [ ] Implement `get_commit_date(file_path)` method
- [ ] Implement `get_file_history(file_path, limit)` method
- [ ] Implement `get_file_at_commit(file_path, commit)` method
- [ ] Add proper error handling

**File:** `src/zapomni_core/temporal/git_temporal.py`

### Task 3.2: Create Fallback Strategy
- [ ] Implement `get_valid_from()` with fallbacks
- [ ] Git commit date → file mtime → current time
- [ ] Add logging for fallback usage

**File:** `src/zapomni_core/temporal/git_temporal.py`

### Task 3.3: Integrate with index_codebase
- [ ] Initialize GitTemporalExtractor in IndexCodebaseTool
- [ ] Get valid_from for each indexed file
- [ ] Detect file changes and create new versions
- [ ] Close old versions when file changes

**File:** `src/zapomni_mcp/tools/index_codebase.py`

### Task 3.4: Unit Tests for Git Integration
- [ ] Test is_git_repo (git dir and non-git dir)
- [ ] Test get_commit_date
- [ ] Test get_file_history
- [ ] Test fallback strategy
- [ ] Mock git commands for testing

**File:** `tests/unit/temporal/test_git_temporal.py`

### Task 3.5: Integration Tests for Git + Indexer
- [ ] Test indexing git repo with history
- [ ] Test re-indexing changed file
- [ ] Test version creation on change

**File:** `tests/integration/test_git_indexing.py`

---

## Phase 4: MCP Tools (2 days)

### Task 4.1: Implement get_timeline Tool
- [ ] Create GetTimelineTool class
- [ ] Implement input validation (Pydantic)
- [ ] Implement execute() method
- [ ] Format output for readability
- [ ] Add workspace_id resolution

**File:** `src/zapomni_mcp/tools/get_timeline.py`

### Task 4.2: Implement get_changes Tool
- [ ] Create GetChangesTool class
- [ ] Implement input validation
- [ ] Implement execute() method
- [ ] Support path_pattern glob filtering
- [ ] Support change_type filtering
- [ ] Format output for readability

**File:** `src/zapomni_mcp/tools/get_changes.py`

### Task 4.3: Implement get_snapshot Tool
- [ ] Create GetSnapshotTool class
- [ ] Implement input validation
- [ ] Implement execute() method
- [ ] Support valid/transaction time_type
- [ ] Format output for readability

**File:** `src/zapomni_mcp/tools/get_snapshot.py`

### Task 4.4: Register Tools in Server
- [ ] Import new tools in __init__.py
- [ ] Add to register_all_tools() in server.py
- [ ] Add to __all__ exports

**Files:**
- `src/zapomni_mcp/tools/__init__.py`
- `src/zapomni_mcp/server.py`

### Task 4.5: Unit Tests for get_timeline
- [ ] Test with file_path
- [ ] Test with qualified_name
- [ ] Test with entity_id
- [ ] Test limit parameter
- [ ] Test empty history
- [ ] Test error handling

**File:** `tests/unit/mcp/tools/test_get_timeline.py`

### Task 4.6: Unit Tests for get_changes
- [ ] Test with since only
- [ ] Test with since + until
- [ ] Test change_type filtering
- [ ] Test path_pattern filtering
- [ ] Test empty results
- [ ] Test error handling

**File:** `tests/unit/mcp/tools/test_get_changes.py`

### Task 4.7: Unit Tests for get_snapshot
- [ ] Test valid time query
- [ ] Test transaction time query
- [ ] Test path_pattern filtering
- [ ] Test empty results
- [ ] Test error handling

**File:** `tests/unit/mcp/tools/test_get_snapshot.py`

### Task 4.8: Integration Tests for Temporal Tools
- [ ] Test get_timeline with real version history
- [ ] Test get_changes after re-indexing
- [ ] Test get_snapshot at different times

**File:** `tests/integration/test_temporal_tools.py`

---

## Phase 5: Documentation & Release (1 day)

### Task 5.1: Update README
- [ ] Add bi-temporal model section
- [ ] Document new MCP tools
- [ ] Add usage examples
- [ ] Update feature list

**File:** `README.md`

### Task 5.2: Update CHANGELOG
- [ ] Add v0.8.0 section
- [ ] List all new features
- [ ] List breaking changes (if any)
- [ ] Add migration instructions

**File:** `CHANGELOG.md`

### Task 5.3: Create Migration Guide
- [ ] Explain bi-temporal model
- [ ] Document migration steps
- [ ] Document new fields
- [ ] Provide troubleshooting tips

**File:** `docs/migration-v0.8.0.md`

### Task 5.4: Update API Documentation
- [ ] Document new MCP tools
- [ ] Document new FalkorDBClient methods
- [ ] Add query examples

**File:** `docs/api.md`

### Task 5.5: Bump Version
- [ ] Update version in pyproject.toml
- [ ] Update version in __init__.py
- [ ] Update version in config.yaml
- [ ] Create git tag v0.8.0

**Files:**
- `pyproject.toml`
- `src/zapomni/__init__.py`
- `.shashka/config.yaml`

### Task 5.6: Final Testing
- [ ] Run full test suite (target: 2740+ tests)
- [ ] Run mypy (0 errors)
- [ ] Run linter
- [ ] Manual testing of new tools
- [ ] Performance benchmarks

### Task 5.7: Update SHASHKA State
- [ ] Update SNAPSHOT.md
- [ ] Update HANDOFF.md
- [ ] Create session log
- [ ] Close Issue #27

---

## Test Coverage Target

| Category | Current | Target | New Tests |
|----------|---------|--------|-----------|
| Unit Tests | 2640 | 2740+ | 100+ |
| Integration Tests | 115 | 125+ | 10+ |
| Total | 2755 | 2865+ | 110+ |

---

## Dependencies

```
Phase 1 ──► Phase 2 ──► Phase 3 ──┐
                                  ├──► Phase 5
Phase 4 (can start after Phase 2)─┘
```

Phase 4 can run in parallel with Phase 3 after Phase 2 completes.

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Performance regression | is_current index, benchmarks |
| Git integration failures | Robust fallback strategy |
| Migration data loss | Idempotent migration, backup |
| Breaking changes | Optional parameters, defaults |
