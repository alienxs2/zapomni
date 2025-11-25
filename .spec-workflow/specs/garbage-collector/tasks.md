# Garbage Collector Implementation Tasks

**Version:** 1.0.0
**Date:** 2025-11-25
**Status:** Draft
**Task:** 04 - Implement Garbage Collector

---

## Task Overview

| Phase | Tasks | Estimated Effort |
|-------|-------|------------------|
| 1. Schema Updates | 2 tasks | 0.5 day |
| 2. Database Methods | 4 tasks | 1 day |
| 3. MCP Tool | 3 tasks | 1 day |
| 4. Delta Indexing | 2 tasks | 0.5 day |
| 5. Testing | 3 tasks | 1 day |
| **Total** | **14 tasks** | **4 days** |

---

## Phase 1: Schema Updates

### Task 1.1: Add stale Property Index

**File:** `/src/zapomni_db/schema_manager.py`

**Description:** Add property index for the `stale` flag on Memory nodes to enable efficient garbage collection queries.

**Changes:**

1. Add new index to `create_property_indexes()` method:

```python
indexes = [
    (self.INDEX_MEMORY_ID, self.NODE_MEMORY, "id"),
    (self.INDEX_ENTITY_NAME, self.NODE_ENTITY, "name"),
    (self.INDEX_TIMESTAMP, self.NODE_MEMORY, "timestamp"),
    ("chunk_memory_id_idx", self.NODE_CHUNK, "memory_id"),
    # NEW: Index for garbage collection
    ("memory_stale_idx", self.NODE_MEMORY, "stale"),
]
```

2. Add class constant for index name:

```python
INDEX_MEMORY_STALE: str = "memory_stale_idx"
```

**Acceptance Criteria:**
- [ ] `memory_stale_idx` index is created on schema init
- [ ] Index creation is idempotent (no error on re-run)
- [ ] Existing tests pass

**Dependencies:** None

**Status:** [ ] Pending

---

### Task 1.2: Update Memory Creation with last_seen_at

**File:** `/src/zapomni_db/falkordb_client.py`

**Description:** Update `_execute_transaction()` to include `last_seen_at` and `stale` properties when creating Memory nodes.

**Changes:**

Update the Memory creation Cypher query:

```python
cypher = """
// Create Memory node with workspace_id and GC properties
CREATE (m:Memory {
    id: $memory_id,
    text: $text,
    tags: $tags,
    source: $source,
    metadata: $metadata,
    workspace_id: $workspace_id,
    created_at: $timestamp,
    stale: false,
    last_seen_at: $timestamp
})
// ... rest of query ...
"""
```

**Acceptance Criteria:**
- [ ] New memories have `stale=false` property
- [ ] New memories have `last_seen_at` property set to creation time
- [ ] Existing `add_memory` tests pass

**Dependencies:** Task 1.1

**Status:** [ ] Pending

---

## Phase 2: Database Methods

### Task 2.1: Implement mark_code_memories_stale()

**File:** `/src/zapomni_db/falkordb_client.py`

**Description:** Add method to mark all code indexer memories as stale before re-indexing.

**Method Signature:**

```python
async def mark_code_memories_stale(
    self,
    workspace_id: str,
) -> int:
    """
    Mark all code memories as stale before re-indexing.

    Args:
        workspace_id: Workspace to mark

    Returns:
        Count of memories marked as stale
    """
```

**Cypher Query:**

```cypher
MATCH (m:Memory)
WHERE m.source = 'code_indexer'
  AND m.workspace_id = $workspace_id
SET m.stale = true
RETURN count(m) AS marked_count
```

**Acceptance Criteria:**
- [ ] Method marks all code memories as stale
- [ ] Returns accurate count
- [ ] Logs operation with count
- [ ] Only affects code_indexer source

**Dependencies:** Task 1.1

**Status:** [ ] Pending

---

### Task 2.2: Implement mark_memory_fresh()

**File:** `/src/zapomni_db/falkordb_client.py`

**Description:** Add method to mark a specific memory as fresh (not stale) during indexing.

**Method Signature:**

```python
async def mark_memory_fresh(
    self,
    file_path: str,
    workspace_id: str,
) -> Optional[str]:
    """
    Mark a specific memory as fresh (not stale) during indexing.
    Also updates last_seen_at timestamp.

    Args:
        file_path: Absolute file path
        workspace_id: Workspace ID

    Returns:
        Memory ID if found and updated, None otherwise
    """
```

**Implementation Notes:**
- File path is stored in JSON metadata
- Use CONTAINS for pattern matching in metadata
- Update both `stale` and `last_seen_at`

**Acceptance Criteria:**
- [ ] Method finds memory by file_path in metadata
- [ ] Sets `stale=false`
- [ ] Updates `last_seen_at` to current datetime
- [ ] Returns memory ID if found, None otherwise
- [ ] Only affects code_indexer source

**Dependencies:** Task 1.2

**Status:** [ ] Pending

---

### Task 2.3: Implement Stale Memory Preview/Delete Methods

**File:** `/src/zapomni_db/falkordb_client.py`

**Description:** Add methods for previewing and deleting stale memories.

**Methods:**

```python
async def get_stale_memories_preview(
    self,
    workspace_id: str,
    limit: int = 20,
) -> Dict[str, Any]:
    """Get preview of stale memories for dry-run."""

async def delete_stale_memories(
    self,
    workspace_id: str,
) -> Dict[str, int]:
    """Delete all stale memories and their chunks."""

async def count_stale_memories(
    self,
    workspace_id: str,
) -> int:
    """Count stale memories in workspace."""
```

**Acceptance Criteria:**
- [ ] `get_stale_memories_preview` returns accurate count and preview
- [ ] `delete_stale_memories` removes memories and their chunks
- [ ] `count_stale_memories` returns accurate count
- [ ] All methods are workspace-scoped
- [ ] Deletion uses DETACH DELETE

**Dependencies:** Task 2.1

**Status:** [ ] Pending

---

### Task 2.4: Implement Orphan Detection/Delete Methods

**File:** `/src/zapomni_db/falkordb_client.py`

**Description:** Add methods for detecting and deleting orphaned chunks and entities.

**Methods:**

```python
async def get_orphaned_chunks_preview(
    self,
    workspace_id: str,
    limit: int = 20,
) -> Dict[str, Any]:
    """Get preview of orphaned chunks."""

async def delete_orphaned_chunks(
    self,
    workspace_id: str,
) -> int:
    """Delete chunks without parent memories."""

async def get_orphaned_entities_preview(
    self,
    workspace_id: str,
    limit: int = 20,
) -> Dict[str, Any]:
    """Get preview of orphaned entities."""

async def delete_orphaned_entities(
    self,
    workspace_id: str,
) -> int:
    """Delete entities without mentions."""
```

**Acceptance Criteria:**
- [ ] Orphaned chunk detection finds chunks without HAS_CHUNK edge
- [ ] Orphaned entity detection finds entities without MENTIONS and RELATED_TO
- [ ] Delete methods return accurate counts
- [ ] All methods are workspace-scoped

**Dependencies:** Task 1.1

**Status:** [ ] Pending

---

## Phase 3: MCP Tool

### Task 3.1: Create PruneMemoryTool Class

**File:** `/src/zapomni_mcp/tools/prune_memory.py` (NEW FILE)

**Description:** Create the main MCP tool class for garbage collection.

**Implementation:**

1. Create Pydantic models:
   - `PruneStrategy` enum
   - `PruneMemoryRequest` model
   - `PrunePreviewItem` model
   - `PrunePreviewResponse` model
   - `PruneResultResponse` model

2. Create `PruneMemoryTool` class with:
   - `name = "prune_memory"`
   - `description` with safety note
   - `input_schema` JSON schema
   - `__init__(self, db_client: FalkorDBClient)`

**Acceptance Criteria:**
- [ ] File created with proper docstrings
- [ ] All Pydantic models validate correctly
- [ ] Tool schema matches requirements
- [ ] Default values are `dry_run=true`, `confirm=false`

**Dependencies:** Phase 2 complete

**Status:** [ ] Pending

---

### Task 3.2: Implement PruneMemoryTool.execute()

**File:** `/src/zapomni_mcp/tools/prune_memory.py`

**Description:** Implement the main execute method with safety checks.

**Implementation Flow:**

1. Validate arguments with Pydantic
2. Resolve workspace_id
3. Get preview (always, for logging)
4. If dry_run: return preview response
5. If not confirm: return error requiring confirmation
6. Execute deletion based on strategy
7. Return result response

**Acceptance Criteria:**
- [ ] Dry run returns preview without deleting
- [ ] Requires `confirm=true` for deletion
- [ ] All strategies work correctly
- [ ] Proper error handling and logging

**Dependencies:** Task 3.1

**Status:** [ ] Pending

---

### Task 3.3: Register PruneMemoryTool

**File:** `/src/zapomni_mcp/tools/__init__.py`

**Description:** Export and register the tool for MCP.

**Changes:**

```python
from zapomni_mcp.tools.prune_memory import PruneMemoryTool

__all__ = [
    # ... existing tools ...
    "PruneMemoryTool",
]
```

**Acceptance Criteria:**
- [ ] Tool is exported from tools module
- [ ] Tool is registered with MCP server
- [ ] Tool is callable via MCP protocol

**Dependencies:** Task 3.2

**Status:** [ ] Pending

---

## Phase 4: Delta Indexing Integration

### Task 4.1: Update IndexCodebaseTool for Delta Indexing

**File:** `/src/zapomni_mcp/tools/index_codebase.py`

**Description:** Integrate mark-and-sweep delta indexing into the code indexer.

**Changes to `execute()`:**

1. Before indexing: Call `mark_code_memories_stale()`
2. During indexing: Call `mark_memory_fresh()` for each file
3. After indexing: Call `count_stale_memories()` for report
4. Update `_format_success()` to include stale count

**Implementation:**

```python
async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
    # ... existing validation ...

    # NEW: Mark all existing code memories as stale
    stale_marked = await self.memory_processor.db_client.mark_code_memories_stale(
        workspace_id=effective_workspace_id
    )

    # ... existing indexing ...

    # NEW: Mark each file as fresh
    for file_dict in files:
        await self.memory_processor.db_client.mark_memory_fresh(
            file_path=file_dict.get("path"),
            workspace_id=effective_workspace_id,
        )

    # NEW: Count remaining stale
    stale_remaining = await self.memory_processor.db_client.count_stale_memories(
        workspace_id=effective_workspace_id
    )

    return self._format_success(
        # ... existing params ...
        stale_memories=stale_remaining,
    )
```

**Acceptance Criteria:**
- [ ] All code memories marked stale before indexing
- [ ] Processed files marked as fresh
- [ ] Stale count reported in response
- [ ] Existing indexing functionality unchanged

**Dependencies:** Phase 2 complete

**Status:** [ ] Pending

---

### Task 4.2: Update _format_success() for Stale Count

**File:** `/src/zapomni_mcp/tools/index_codebase.py`

**Description:** Update the success response to include stale memory information.

**Changes:**

1. Add `stale_memories` parameter to method
2. Add conditional message about stale memories

```python
def _format_success(
    self,
    # ... existing params ...
    stale_memories: int = 0,
) -> Dict[str, Any]:
    # ... existing message building ...

    if stale_memories > 0:
        message += f"\nStale memories detected: {stale_memories} (run prune_memory to clean up)"

    return {"content": [{"type": "text", "text": message}], "isError": False}
```

**Acceptance Criteria:**
- [ ] Message includes stale count when > 0
- [ ] Message suggests running prune_memory
- [ ] No change to message when stale_memories = 0

**Dependencies:** Task 4.1

**Status:** [ ] Pending

---

## Phase 5: Testing

### Task 5.1: Unit Tests for PruneMemoryTool

**File:** `/tests/unit/test_prune_memory_tool.py` (NEW FILE)

**Description:** Create unit tests for the prune_memory tool.

**Test Cases:**

```python
class TestPruneMemoryTool:
    async def test_dry_run_returns_preview(self):
        """Dry run should return preview without deleting."""

    async def test_dry_run_is_default(self):
        """Default should be dry_run=true."""

    async def test_deletion_requires_confirmation(self):
        """Deletion without confirm=true should fail."""

    async def test_deletion_with_confirmation(self):
        """Deletion with dry_run=false and confirm=true should work."""

    async def test_stale_code_strategy(self):
        """stale_code strategy should only affect code memories."""

    async def test_orphaned_chunks_strategy(self):
        """orphaned_chunks strategy should find parentless chunks."""

    async def test_orphaned_entities_strategy(self):
        """orphaned_entities strategy should find unconnected entities."""

    async def test_all_strategy(self):
        """all strategy should run all cleanup operations."""

    async def test_workspace_isolation(self):
        """Operations should be scoped to workspace_id."""

    async def test_invalid_strategy_error(self):
        """Invalid strategy should return error."""
```

**Acceptance Criteria:**
- [ ] All test cases implemented
- [ ] Tests use mocking for database
- [ ] Tests cover edge cases
- [ ] All tests pass

**Dependencies:** Phase 3 complete

**Status:** [ ] Pending

---

### Task 5.2: Unit Tests for FalkorDBClient GC Methods

**File:** `/tests/unit/test_falkordb_gc.py` (NEW FILE)

**Description:** Create unit tests for the garbage collection database methods.

**Test Cases:**

```python
class TestFalkorDBGarbageCollection:
    async def test_mark_code_memories_stale(self):
        """Should mark all code memories as stale."""

    async def test_mark_memory_fresh(self):
        """Should mark specific memory as fresh."""

    async def test_get_stale_memories_preview(self):
        """Should return preview of stale memories."""

    async def test_delete_stale_memories(self):
        """Should delete stale memories and chunks."""

    async def test_get_orphaned_chunks_preview(self):
        """Should find chunks without parent."""

    async def test_delete_orphaned_chunks(self):
        """Should delete orphaned chunks."""

    async def test_get_orphaned_entities_preview(self):
        """Should find entities without mentions."""

    async def test_delete_orphaned_entities(self):
        """Should delete orphaned entities."""

    async def test_count_stale_memories(self):
        """Should return accurate count."""
```

**Acceptance Criteria:**
- [ ] All test cases implemented
- [ ] Tests cover Cypher query correctness
- [ ] Tests verify workspace isolation
- [ ] All tests pass

**Dependencies:** Phase 2 complete

**Status:** [ ] Pending

---

### Task 5.3: Integration Tests for GC Workflow

**File:** `/tests/integration/test_gc_workflow.py` (NEW FILE)

**Description:** Create integration tests for the full garbage collection workflow.

**Test Cases:**

```python
class TestGCWorkflow:
    async def test_full_gc_workflow(self):
        """
        Test complete workflow:
        1. Index repository with files
        2. Delete file from filesystem
        3. Re-index repository
        4. Verify file marked as stale
        5. Preview prune operation
        6. Execute prune operation
        7. Verify node removed
        8. Verify stats updated
        """

    async def test_acceptance_criteria(self):
        """
        Test acceptance criteria from task:
        - Index file with hello() function
        - Delete hello() from file, re-index
        - Node is removed after prune
        - get_stats shows decreased count
        - No broken edges
        """

    async def test_delta_indexing_marks_stale(self):
        """Re-indexing should mark removed files as stale."""

    async def test_no_broken_edges_after_prune(self):
        """Graph should have no broken edges after prune."""

    async def test_workspace_isolation_in_workflow(self):
        """GC should not affect other workspaces."""
```

**Acceptance Criteria:**
- [ ] Full workflow test passes
- [ ] Acceptance criteria test passes
- [ ] Graph integrity verified
- [ ] All tests pass

**Dependencies:** Phase 4 complete

**Status:** [ ] Pending

---

## Task Summary

| Task ID | Description | Status | Dependencies |
|---------|-------------|--------|--------------|
| 1.1 | Add stale Property Index | [ ] | None |
| 1.2 | Update Memory Creation with last_seen_at | [ ] | 1.1 |
| 2.1 | Implement mark_code_memories_stale() | [ ] | 1.1 |
| 2.2 | Implement mark_memory_fresh() | [ ] | 1.2 |
| 2.3 | Implement Stale Memory Preview/Delete | [ ] | 2.1 |
| 2.4 | Implement Orphan Detection/Delete | [ ] | 1.1 |
| 3.1 | Create PruneMemoryTool Class | [ ] | Phase 2 |
| 3.2 | Implement PruneMemoryTool.execute() | [ ] | 3.1 |
| 3.3 | Register PruneMemoryTool | [ ] | 3.2 |
| 4.1 | Update IndexCodebaseTool | [ ] | Phase 2 |
| 4.2 | Update _format_success() | [ ] | 4.1 |
| 5.1 | Unit Tests for PruneMemoryTool | [ ] | Phase 3 |
| 5.2 | Unit Tests for FalkorDBClient GC | [ ] | Phase 2 |
| 5.3 | Integration Tests for GC Workflow | [ ] | Phase 4 |

---

## Implementation Order

Recommended implementation sequence:

1. **Day 1 Morning:** Tasks 1.1, 1.2 (Schema updates)
2. **Day 1 Afternoon:** Tasks 2.1, 2.2 (Core GC methods)
3. **Day 2 Morning:** Tasks 2.3, 2.4 (Preview/Delete methods)
4. **Day 2 Afternoon:** Tasks 3.1, 3.2, 3.3 (MCP Tool)
5. **Day 3 Morning:** Tasks 4.1, 4.2 (Delta indexing)
6. **Day 3 Afternoon:** Tasks 5.1, 5.2 (Unit tests)
7. **Day 4:** Task 5.3 (Integration tests) + Bug fixes

---

## Verification Checklist

After implementation, verify:

- [ ] `prune_memory` tool is callable via MCP
- [ ] Default is `dry_run=true` - no deletion without explicit params
- [ ] Requires `confirm=true` for actual deletion
- [ ] `index_codebase` marks stale nodes during re-indexing
- [ ] `index_codebase` reports stale count in response
- [ ] `get_stats` shows accurate counts before/after GC
- [ ] No broken edges after GC operations
- [ ] All existing tests still pass
- [ ] New tests all pass

---

**Document History:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-11-25 | Specification Writer | Initial draft |
