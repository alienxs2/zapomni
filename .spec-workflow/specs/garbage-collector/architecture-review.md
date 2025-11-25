# Garbage Collector Architecture Review

**Date:** 2025-11-25
**Project:** Zapomni
**Task:** Implement Garbage Collector for Knowledge Graph (Task 04)
**Scope:** MVP - Manual Garbage Collection Only

---

## 1. Executive Summary

This document provides an architecture review for implementing a garbage collection system for Zapomni's FalkorDB knowledge graph. The MVP focuses on a **manual-only, dry-run-by-default** approach via a `prune_memory()` MCP tool with safety mechanisms.

### Key Design Decisions

1. **Manual Trigger Only** - No automated background GC for MVP
2. **Dry Run by Default** - `dry_run=True` prevents accidental deletions
3. **Explicit Confirmation** - Requires `confirm=True` for actual deletion
4. **Delta Indexing Integration** - Mark-and-sweep during `index_codebase`
5. **Workspace Isolation** - All operations scoped to workspace_id

---

## 2. Affected Modules and Files

### 2.1 Core Files to Modify

| File | Changes Required | Priority |
|------|------------------|----------|
| `/src/zapomni_db/schema_manager.py` | Add `stale` property to Memory nodes | High |
| `/src/zapomni_db/falkordb_client.py` | Add `prune_stale_memories()` method | High |
| `/src/zapomni_mcp/tools/__init__.py` | Export `PruneMemoryTool` | High |
| `/src/zapomni_mcp/tools/index_codebase.py` | Add delta indexing (mark stale) | Medium |
| `/src/zapomni_core/code/repository_indexer.py` | No changes needed | - |
| `/src/zapomni_core/memory_processor.py` | No changes needed | - |

### 2.2 New Files to Create

| File | Purpose | Priority |
|------|---------|----------|
| `/src/zapomni_mcp/tools/prune_memory.py` | MCP tool implementation | High |
| `/src/zapomni_db/garbage_collector.py` | GC query logic (optional) | Low |

### 2.3 Test Files to Create

| File | Purpose |
|------|---------|
| `/tests/unit/test_prune_memory_tool.py` | Unit tests for MCP tool |
| `/tests/unit/test_falkordb_gc.py` | Unit tests for GC queries |
| `/tests/integration/test_gc_workflow.py` | End-to-end GC tests |

---

## 3. Schema Changes

### 3.1 New Properties on Memory Nodes

Add these properties to the `Memory` node schema:

```cypher
// Memory node with new GC properties
CREATE (m:Memory {
    id: $memory_id,
    text: $text,
    tags: $tags,
    source: $source,
    metadata: $metadata,
    workspace_id: $workspace_id,
    created_at: $timestamp,
    // NEW: Garbage collection properties
    stale: false,              // Mark for deletion during delta indexing
    last_seen_at: $timestamp   // Optional: TTL-based cleanup (future)
})
```

### 3.2 SchemaManager Updates

**File:** `/src/zapomni_db/schema_manager.py`

Add a new property index for the `stale` flag:

```python
# In create_property_indexes():
indexes = [
    (self.INDEX_MEMORY_ID, self.NODE_MEMORY, "id"),
    (self.INDEX_ENTITY_NAME, self.NODE_ENTITY, "name"),
    (self.INDEX_TIMESTAMP, self.NODE_MEMORY, "timestamp"),
    ("chunk_memory_id_idx", self.NODE_CHUNK, "memory_id"),
    # NEW: Index for garbage collection
    ("memory_stale_idx", self.NODE_MEMORY, "stale"),
]
```

### 3.3 Schema Migration (Future)

For production systems, add migration support:

```python
# schema_manager.py
SCHEMA_VERSION = "1.1.0"  # Bump from 1.0.0

def migrate_1_0_to_1_1(self) -> None:
    """Add stale property to existing Memory nodes."""
    cypher = """
    MATCH (m:Memory)
    WHERE m.stale IS NULL
    SET m.stale = false,
        m.last_seen_at = m.created_at
    RETURN count(m) AS migrated
    """
    self._execute_cypher(cypher)
```

---

## 4. FalkorDBClient Additions

### 4.1 New Methods

**File:** `/src/zapomni_db/falkordb_client.py`

```python
async def mark_code_memories_stale(
    self,
    workspace_id: str,
) -> int:
    """
    Mark all code memories as stale before re-indexing.

    Returns:
        Count of memories marked as stale
    """

async def mark_memory_fresh(
    self,
    file_path: str,
    workspace_id: str,
) -> Optional[str]:
    """
    Mark a specific memory as fresh (not stale) during indexing.

    Returns:
        Memory ID if found, None otherwise
    """

async def get_stale_memories_preview(
    self,
    workspace_id: str,
    source: Optional[str] = None,
    limit: int = 20,
) -> Dict[str, Any]:
    """
    Get preview of stale memories for dry-run.

    Returns:
        {
            "memory_count": int,
            "chunk_count": int,
            "preview": List[Dict],  # First N memories
        }
    """

async def delete_stale_memories(
    self,
    workspace_id: str,
    source: Optional[str] = None,
) -> Dict[str, int]:
    """
    Delete all stale memories and their chunks.

    Returns:
        {"deleted_memories": int, "deleted_chunks": int}
    """

async def get_orphaned_chunks(
    self,
    workspace_id: str,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Find chunks without parent memories.
    """

async def get_orphaned_entities(
    self,
    workspace_id: str,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Find entities without mentions from chunks.
    """

async def delete_orphaned_chunks(
    self,
    workspace_id: str,
) -> int:
    """
    Delete chunks without parent memories.
    """

async def delete_orphaned_entities(
    self,
    workspace_id: str,
) -> int:
    """
    Delete entities without mentions.
    """
```

---

## 5. Cypher Query Patterns

### 5.1 Mark All Code Memories as Stale

```cypher
MATCH (m:Memory)
WHERE m.source = 'code_indexer'
  AND m.workspace_id = $workspace_id
SET m.stale = true
RETURN count(m) AS marked_count
```

### 5.2 Mark Memory as Fresh During Indexing

```cypher
MATCH (m:Memory {file_path: $file_path, workspace_id: $workspace_id})
WHERE m.source = 'code_indexer'
SET m.stale = false,
    m.last_seen_at = datetime()
RETURN m.id AS memory_id
```

### 5.3 Preview Stale Memories (Dry Run)

```cypher
MATCH (m:Memory)
WHERE m.source = $source
  AND m.workspace_id = $workspace_id
  AND m.stale = true
OPTIONAL MATCH (m)-[:HAS_CHUNK]->(c:Chunk)
WITH m, count(c) AS chunk_count
RETURN m.id AS memory_id,
       m.file_path AS file_path,
       m.relative_path AS relative_path,
       m.created_at AS created_at,
       m.lines AS lines,
       chunk_count
ORDER BY m.created_at DESC
LIMIT $limit
```

### 5.4 Count Stale Memories

```cypher
MATCH (m:Memory)
WHERE m.source = $source
  AND m.workspace_id = $workspace_id
  AND m.stale = true
OPTIONAL MATCH (m)-[:HAS_CHUNK]->(c:Chunk)
RETURN count(DISTINCT m) AS memory_count,
       count(DISTINCT c) AS chunk_count
```

### 5.5 Delete Stale Memories and Chunks

```cypher
MATCH (m:Memory)
WHERE m.source = $source
  AND m.workspace_id = $workspace_id
  AND m.stale = true
OPTIONAL MATCH (m)-[:HAS_CHUNK]->(c:Chunk)
DETACH DELETE m, c
RETURN count(DISTINCT m) AS deleted_memories,
       count(DISTINCT c) AS deleted_chunks
```

### 5.6 Find Orphaned Chunks

```cypher
MATCH (c:Chunk)
WHERE c.workspace_id = $workspace_id
  AND NOT (:Memory)-[:HAS_CHUNK]->(c)
RETURN c.id AS chunk_id,
       c.workspace_id AS workspace_id,
       size(c.text) AS text_length
LIMIT $limit
```

### 5.7 Delete Orphaned Chunks

```cypher
MATCH (c:Chunk)
WHERE c.workspace_id = $workspace_id
  AND NOT (:Memory)-[:HAS_CHUNK]->(c)
DETACH DELETE c
RETURN count(c) AS deleted_count
```

### 5.8 Find Orphaned Entities

```cypher
MATCH (e:Entity)
WHERE e.workspace_id = $workspace_id
  AND NOT (:Chunk)-[:MENTIONS]->(e)
  AND NOT (e)-[:RELATED_TO]-(:Entity)
RETURN e.id AS entity_id,
       e.name AS name,
       e.type AS type
LIMIT $limit
```

### 5.9 Delete Orphaned Entities

```cypher
MATCH (e:Entity)
WHERE e.workspace_id = $workspace_id
  AND NOT (:Chunk)-[:MENTIONS]->(e)
  AND NOT (e)-[:RELATED_TO]-(:Entity)
DETACH DELETE e
RETURN count(e) AS deleted_count
```

---

## 6. MCP Tool Design: `prune_memory`

### 6.1 Tool Specification

**File:** `/src/zapomni_mcp/tools/prune_memory.py`

```python
class PruneMemoryTool:
    """
    MCP tool for pruning stale memory nodes from knowledge graph.

    Supports multiple GC strategies with dry-run preview and
    explicit confirmation for safety.
    """

    name = "prune_memory"
    description = (
        "Prune stale or orphaned memory nodes from the knowledge graph. "
        "SAFETY: Defaults to dry_run=true (preview only). "
        "Set dry_run=false and confirm=true to delete. "
        "Use strategy parameter to select what to clean up."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "workspace_id": {
                "type": "string",
                "description": "Workspace to prune (default: current workspace)",
                "default": "",
            },
            "dry_run": {
                "type": "boolean",
                "description": (
                    "Preview mode - show what would be deleted without deleting. "
                    "DEFAULT: true. Set to false to perform actual deletion."
                ),
                "default": True,
            },
            "confirm": {
                "type": "boolean",
                "description": (
                    "REQUIRED for deletion: Must be true to confirm deletion. "
                    "Ignored when dry_run=true. Safety feature to prevent accidents."
                ),
                "default": False,
            },
            "strategy": {
                "type": "string",
                "description": (
                    "What to prune:\n"
                    "- 'stale_code': Delete stale code indexer memories\n"
                    "- 'orphaned_chunks': Delete chunks without parent memories\n"
                    "- 'orphaned_entities': Delete entities without mentions\n"
                    "- 'all': Run all strategies"
                ),
                "enum": ["stale_code", "orphaned_chunks", "orphaned_entities", "all"],
                "default": "stale_code",
            },
        },
        "required": [],
        "additionalProperties": False,
    }
```

### 6.2 Tool Response Format

**Dry Run Response:**
```json
{
    "content": [{
        "type": "text",
        "text": "Dry run complete. Preview of items to delete:\n\n..."
    }],
    "isError": false
}
```

**Deletion Response:**
```json
{
    "content": [{
        "type": "text",
        "text": "Successfully deleted 42 stale memory nodes and 156 chunks."
    }],
    "isError": false
}
```

**Error Response (no confirmation):**
```json
{
    "content": [{
        "type": "text",
        "text": "Error: Deletion requires explicit confirmation. Set confirm=true to proceed."
    }],
    "isError": true
}
```

---

## 7. Delta Indexing Integration

### 7.1 Changes to `index_codebase.py`

**File:** `/src/zapomni_mcp/tools/index_codebase.py`

Modify `IndexCodebaseTool.execute()` to implement delta indexing:

```python
async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
    # ... existing validation ...

    # NEW: Step 0 - Mark all existing code memories as stale
    log.info("marking_existing_memories_stale")
    stale_count = await self.memory_processor.db_client.mark_code_memories_stale(
        workspace_id=effective_workspace_id
    )
    log.info("memories_marked_stale", count=stale_count)

    # ... existing indexing logic ...

    # Step 5: Store code as memories
    for file_dict in files:
        # ... create memory ...

        # NEW: Check if memory exists and mark as fresh
        existing_id = await self.memory_processor.db_client.mark_memory_fresh(
            file_path=file_dict["path"],
            workspace_id=effective_workspace_id,
        )

        if existing_id:
            # File unchanged, skip re-indexing
            log.debug("file_unchanged", file_path=file_dict["path"])
            continue

        # File is new or changed, create new memory
        await self.memory_processor.add_memory(...)

    # NEW: Report stale count in response
    stale_remaining = await self._count_stale_memories(effective_workspace_id)

    return self._format_success(
        # ... existing fields ...
        stale_memories=stale_remaining,  # NEW field
    )
```

### 7.2 Updated Response Format

```
Repository indexed successfully.
Path: /path/to/repo
Files indexed: 42
Functions: 150
Classes: 28
Languages: Python (35), JavaScript (7)
Total lines: 8,500
Indexing time: 2.35s
Stale memories: 5 (run prune_memory to clean up)  <-- NEW
```

---

## 8. Safety Mechanisms

### 8.1 Multi-Layer Safety

1. **Layer 1: Dry Run Default**
   - `dry_run=True` by default
   - Shows preview without deleting anything
   - User must explicitly set `dry_run=False`

2. **Layer 2: Explicit Confirmation**
   - `confirm=False` by default
   - Even with `dry_run=False`, requires `confirm=True`
   - Prevents accidental double-click or script errors

3. **Layer 3: Workspace Isolation**
   - All operations scoped to `workspace_id`
   - Cannot accidentally delete across workspaces
   - Default workspace has extra protections

4. **Layer 4: Preview Before Delete**
   - Always show what will be deleted
   - Include file paths, counts, timestamps
   - Allow user to review before confirming

### 8.2 Validation Flow

```python
async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
    # Parse and validate
    request = PruneMemoryRequest(**arguments)

    # Get preview (always)
    preview = await self._get_preview(request)

    # Dry run - return preview only
    if request.dry_run:
        return self._format_preview(preview)

    # Not dry run - require confirmation
    if not request.confirm:
        return self._format_error(
            "Deletion requires explicit confirmation. "
            "Set confirm=True to proceed. "
            "Run with dry_run=True first to preview."
        )

    # Execute deletion
    result = await self._execute_deletion(request)
    return self._format_result(result)
```

### 8.3 Audit Logging

All GC operations are logged with:
- Operation type (preview/delete)
- Workspace ID
- Strategy used
- Counts (memories, chunks, entities)
- Timestamp
- User/session ID (if available)

```python
log.info(
    "prune_memory_executed",
    workspace_id=request.workspace_id,
    dry_run=request.dry_run,
    confirm=request.confirm,
    strategy=request.strategy,
    deleted_memories=result.deleted_memories,
    deleted_chunks=result.deleted_chunks,
)
```

---

## 9. Implementation Approach

### 9.1 Phase 1: Core Infrastructure (Day 1)

1. **Update Schema**
   - Add `stale` property to Memory nodes
   - Add property index for `stale` flag
   - Test schema migration

2. **Add FalkorDBClient Methods**
   - `mark_code_memories_stale()`
   - `mark_memory_fresh()`
   - `get_stale_memories_preview()`
   - `delete_stale_memories()`

### 9.2 Phase 2: MCP Tool (Day 1-2)

1. **Create PruneMemoryTool**
   - Implement `execute()` method
   - Add Pydantic request/response models
   - Implement all strategies

2. **Register Tool**
   - Add to `__init__.py` exports
   - Register with MCP server

### 9.3 Phase 3: Delta Indexing (Day 2)

1. **Update index_codebase**
   - Add mark stale step before indexing
   - Add mark fresh step during indexing
   - Update response with stale count

2. **Test Integration**
   - Test full workflow: index -> modify -> re-index -> prune

### 9.4 Phase 4: Testing (Day 2-3)

1. **Unit Tests**
   - Test each Cypher query
   - Test dry run vs actual deletion
   - Test confirmation requirement

2. **Integration Tests**
   - Test full GC workflow
   - Test edge cases (empty workspace, all stale, none stale)

---

## 10. Testing Strategy

### 10.1 Unit Tests

```python
# tests/unit/test_prune_memory_tool.py

class TestPruneMemoryTool:
    async def test_dry_run_returns_preview(self):
        """Dry run should return preview without deleting."""

    async def test_deletion_requires_confirmation(self):
        """Deletion without confirm=True should fail."""

    async def test_deletion_with_confirmation_succeeds(self):
        """Deletion with dry_run=False and confirm=True should work."""

    async def test_workspace_isolation(self):
        """Operations should be scoped to workspace_id."""

    async def test_stale_code_strategy(self):
        """stale_code strategy should only affect code memories."""

    async def test_orphaned_chunks_strategy(self):
        """orphaned_chunks strategy should find parentless chunks."""

    async def test_all_strategy(self):
        """all strategy should run all cleanup operations."""
```

### 10.2 Integration Tests

```python
# tests/integration/test_gc_workflow.py

class TestGCWorkflow:
    async def test_full_gc_workflow(self):
        """Test: index -> delete file -> re-index -> prune."""
        # 1. Index repository
        # 2. Verify memories created
        # 3. Delete a file from repo
        # 4. Re-index repository
        # 5. Verify file marked as stale
        # 6. Run prune (dry run)
        # 7. Verify preview correct
        # 8. Run prune (actual)
        # 9. Verify stale memories deleted
```

---

## 11. Dependencies

### 11.1 No New External Dependencies

The implementation uses existing project dependencies:
- `falkordb` - Database client
- `pydantic` - Request validation
- `structlog` - Logging

### 11.2 Internal Dependencies

```
prune_memory.py
    -> FalkorDBClient (existing)
    -> MemoryProcessor (existing, for workspace context)

index_codebase.py (modified)
    -> FalkorDBClient.mark_code_memories_stale() (new)
    -> FalkorDBClient.mark_memory_fresh() (new)
```

---

## 12. Risks and Mitigations

### 12.1 Risk: Accidental Data Loss

**Mitigation:**
- Dry run by default
- Explicit confirmation required
- Preview before deletion
- Comprehensive logging

### 12.2 Risk: Performance on Large Graphs

**Mitigation:**
- Batch deletion (LIMIT in Cypher)
- Index on `stale` property
- Async execution with timeout

### 12.3 Risk: Orphaned Entities After Memory Deletion

**Mitigation:**
- Include `orphaned_entities` strategy
- Run after `stale_code` cleanup
- Document cleanup order

### 12.4 Risk: Race Condition During Re-indexing

**Mitigation:**
- Use MERGE instead of CREATE for idempotent operations
- Atomic mark-and-sweep within single transaction
- Document that concurrent re-indexing is not supported

---

## 13. Future Enhancements (Not MVP)

1. **Automated Background GC** - Scheduled task with configurable interval
2. **Soft Delete** - Mark as deleted, purge after grace period
3. **TTL-based Cleanup** - Delete memories older than threshold
4. **Backup Before Prune** - Export graph before destructive operations
5. **Audit Log Table** - Persistent record of all GC operations
6. **Entity Deduplication** - Merge duplicate entities during cleanup

---

## 14. Summary

### Files to Create
- `/src/zapomni_mcp/tools/prune_memory.py`

### Files to Modify
- `/src/zapomni_db/schema_manager.py` - Add `stale` index
- `/src/zapomni_db/falkordb_client.py` - Add GC methods
- `/src/zapomni_mcp/tools/__init__.py` - Export tool
- `/src/zapomni_mcp/tools/index_codebase.py` - Delta indexing

### Key Patterns
- Dry run by default (`dry_run=True`)
- Explicit confirmation (`confirm=True`)
- Mark-and-sweep during indexing
- Workspace-scoped operations

### Estimated Effort
- Core implementation: 1-2 days
- Testing: 1 day
- Total: 2-3 days

---

**Prepared by:** Architecture Agent
**Date:** 2025-11-25
**Version:** 1.0
**Status:** Ready for Implementation
