# Garbage Collection for Knowledge Graphs - Research Report

**Date:** 2025-11-25
**Project:** Zapomni
**Phase:** MVP - Manual Garbage Collection Only

---

## Executive Summary

This research investigates garbage collection (GC) strategies for Zapomni's FalkorDB-based knowledge graph. When files are deleted or modified, old nodes (functions, classes, entities) become orphaned and need cleanup. For MVP, we implement a **manual-only, dry-run-by-default** approach via a `prune_memory()` MCP tool with safety mechanisms to prevent accidental data loss.

### Key Findings
1. **Orphan Detection:** Use Cypher pattern matching to find nodes without incoming edges or references
2. **Delta Indexing:** Mark nodes as "dirty" during re-indexing, clean up unmarked nodes after
3. **Safety First:** Default to `dry_run=True`, require explicit confirmation for deletion
4. **Timestamp Approach:** Add `last_seen_at` timestamps for TTL-based cleanup (future enhancement)
5. **Soft Delete:** Mark nodes as `deleted=true` before hard deletion (optional safety layer)

---

## 1. Current Architecture Analysis

### 1.1 Database Schema (FalkorDB/Cypher)

From `/home/dev/zapomni/src/zapomni_db/schema_manager.py`:

**Node Types:**
- `Memory` - Main memory/document nodes with metadata
- `Chunk` - Text chunks with embeddings (768-dim vectors)
- `Entity` - Named entities extracted from text
- `Document` - Source document metadata (future)

**Relationship Types:**
- `HAS_CHUNK` - Memory → Chunk (one-to-many)
- `MENTIONS` - Chunk → Entity (many-to-many)
- `RELATED_TO` - Entity → Entity (semantic relationships)

**Properties Available:**
- `workspace_id` - Data isolation by workspace
- `created_at` - Node creation timestamp
- `updated_at` - Node update timestamp (on Workspace, Entity)
- `id` - UUID for all nodes

**Current Indexes:**
- Vector index: `chunk_embedding_idx` on `Chunk.embedding` (HNSW, cosine, 768-dim)
- Property indexes: `memory_id_idx`, `entity_name_idx`, `timestamp_idx`

### 1.2 Code Indexing Flow

From `/home/dev/zapomni/src/zapomni_mcp/tools/index_codebase.py` and `/home/dev/zapomni/src/zapomni_core/code/repository_indexer.py`:

**Current Process:**
1. Scan repository using `CodeRepositoryIndexer`
2. Extract functions, classes, methods via AST parsing
3. Create chunks and embeddings
4. Store as `Memory` nodes with metadata:
   - `source: "code_indexer"`
   - `file_path: "/absolute/path/to/file.py"`
   - `relative_path: "src/module/file.py"`
   - `extension: ".py"`
   - `lines: 150`

**Problem:** When a file is deleted or modified:
- Old `Memory` nodes remain in graph
- Old `Chunk` nodes with embeddings persist
- Old `Entity` nodes (if extracted) become orphaned
- No mechanism to detect or clean up stale data

### 1.3 Memory Storage Architecture

From `/home/dev/zapomni/src/zapomni_core/memory_processor.py`:

**Storage Pipeline:**
1. Text → Chunks (SemanticChunker)
2. Chunks → Embeddings (OllamaEmbedder)
3. Text → Entities/Relationships (EntityExtractor, optional)
4. Store in FalkorDB via `add_memory()`

**Current Cypher for Memory Creation:**
```cypher
CREATE (m:Memory {
    id: $memory_id,
    text: $text,
    tags: $tags,
    source: $source,
    metadata: $metadata,
    workspace_id: $workspace_id,
    created_at: $timestamp
})
WITH m
UNWIND $chunks_data AS chunk_data
CREATE (c:Chunk {
    id: chunk_data.id,
    text: chunk_data.text,
    index: chunk_data.index,
    workspace_id: $workspace_id,
    embedding: vecf32(chunk_data.embedding)
})
CREATE (m)-[:HAS_CHUNK {index: chunk_data.index}]->(c)
```

**Deletion Operations:**
From `/home/dev/zapomni/src/zapomni_db/falkordb_client.py`:

```cypher
-- Delete single memory and its chunks
MATCH (m:Memory {id: $memory_id})
OPTIONAL MATCH (m)-[:HAS_CHUNK]->(c:Chunk)
DETACH DELETE m, c
```

```cypher
-- Clear all data in workspace
MATCH (n)
WHERE n.workspace_id = $workspace_id
DETACH DELETE n
```

**Gap:** No mechanism to identify orphaned nodes or mark stale data for cleanup.

---

## 2. Orphan Detection Algorithms

### 2.1 Reference Counting Approach

**Concept:** Track incoming edge counts; nodes with zero incoming edges are orphans.

**Cypher Query:**
```cypher
MATCH (n:Memory)
WHERE NOT (n)<-[:HAS_CHUNK]-()
  AND NOT (n)<-[:MENTIONS]-()
  AND NOT (n)<-[:RELATED_TO]-()
RETURN n.id AS orphan_id, n.source, n.created_at
```

**Pros:**
- Simple and fast
- Works for any node type
- Easy to extend to other relationships

**Cons:**
- False positives if node legitimately has no incoming edges (e.g., root nodes)
- Doesn't detect circular references (A→B→A, both orphaned from main graph)
- Requires knowledge of all relationship types

**For Code Entities:**
```cypher
-- Find code memories without parent references
MATCH (m:Memory)
WHERE m.source = 'code_indexer'
  AND NOT EXISTS((m)<-[:INDEXED_FROM]-())
RETURN m.id, m.metadata
```

### 2.2 Reachability Analysis (Mark-and-Sweep)

**Concept:** Start from root nodes (e.g., workspace), traverse all reachable nodes, delete unreachable ones.

**Cypher Query (Phase 1 - Mark Reachable):**
```cypher
MATCH (w:Workspace {id: $workspace_id})
MATCH (w)-[*0..10]-(n)
SET n.reachable = true
RETURN count(n) AS reachable_count
```

**Cypher Query (Phase 2 - Find Unreachable):**
```cypher
MATCH (n)
WHERE n.workspace_id = $workspace_id
  AND (n.reachable IS NULL OR n.reachable = false)
RETURN n.id, labels(n), n.created_at
```

**Cypher Query (Phase 3 - Sweep/Delete):**
```cypher
MATCH (n)
WHERE n.workspace_id = $workspace_id
  AND (n.reachable IS NULL OR n.reachable = false)
DETACH DELETE n
RETURN count(n) AS deleted_count
```

**Pros:**
- Handles circular references correctly
- Comprehensive (finds all disconnected subgraphs)
- Works even if relationship types are unknown

**Cons:**
- More complex implementation
- Requires multiple queries (mark, sweep, cleanup)
- Performance impact on large graphs
- Temporary property modification (`reachable`) may conflict with concurrent operations

**Recommendation for Zapomni:** Use **reference counting** for MVP simplicity. Upgrade to reachability analysis if circular references become an issue.

### 2.3 File-Based Orphan Detection (Specific to Code Indexing)

**Concept:** Track which files exist on disk, mark nodes referencing non-existent files as orphans.

**Cypher Query:**
```cypher
MATCH (m:Memory)
WHERE m.source = 'code_indexer'
  AND m.file_path NOT IN $existing_file_paths
RETURN m.id, m.file_path, m.created_at
```

**Implementation Steps:**
1. Scan repository, collect all current file paths
2. Query graph for all `Memory` nodes with `source='code_indexer'`
3. Compare file paths, mark nodes referencing deleted files
4. Delete marked nodes

**Pros:**
- Directly addresses code indexing use case
- Fast (single query + set comparison)
- Accurate (file existence is ground truth)

**Cons:**
- Only works for code indexing, not general memories
- Requires file system access
- Doesn't handle moved files (appears as delete + create)

---

## 3. Delta Indexing Patterns

### 3.1 Mark Dirty/Clean During Re-indexing

**Concept:** During re-indexing, mark nodes as "seen" if they still exist, delete "unseen" nodes after.

**Workflow:**
1. **Before re-index:** Mark all code nodes as `stale=true`
2. **During re-index:** For each file:
   - Check if corresponding `Memory` node exists
   - If exists and unchanged: Set `stale=false` (keep it)
   - If exists and changed: Delete old, create new
   - If new file: Create new node
3. **After re-index:** Delete all nodes with `stale=true`

**Cypher Queries:**

**Step 1 - Mark All as Stale:**
```cypher
MATCH (m:Memory)
WHERE m.source = 'code_indexer'
  AND m.workspace_id = $workspace_id
SET m.stale = true
RETURN count(m) AS marked_count
```

**Step 2 - Mark File as Fresh (during indexing):**
```cypher
MATCH (m:Memory {file_path: $file_path, workspace_id: $workspace_id})
WHERE m.source = 'code_indexer'
SET m.stale = false,
    m.last_indexed_at = datetime()
RETURN m.id
```

**Step 3 - Delete Stale Nodes:**
```cypher
MATCH (m:Memory)
WHERE m.source = 'code_indexer'
  AND m.workspace_id = $workspace_id
  AND m.stale = true
OPTIONAL MATCH (m)-[:HAS_CHUNK]->(c:Chunk)
DETACH DELETE m, c
RETURN count(m) AS deleted_count
```

**Pros:**
- Simple and effective
- Works incrementally during indexing
- Low false positive rate

**Cons:**
- Requires `stale` property on nodes
- Need to handle concurrent re-indexing (multiple processes)
- Transaction safety concerns (mark+delete must be atomic)

**Recommendation:** This is the **best approach for code indexing** use case. Implement for MVP.

### 3.2 Timestamp-Based Detection

**Concept:** Track `last_seen_at` timestamp, delete nodes not seen in X days.

**Cypher Query:**
```cypher
MATCH (m:Memory)
WHERE m.source = 'code_indexer'
  AND m.workspace_id = $workspace_id
  AND datetime(m.last_seen_at) < datetime() - duration({days: 30})
RETURN m.id, m.file_path, m.last_seen_at
```

**Workflow:**
1. Add `last_seen_at` property to schema
2. Update `last_seen_at = datetime()` during each re-index
3. Run cleanup job to delete nodes with `last_seen_at` older than threshold

**Pros:**
- Simple to implement
- Works well with incremental indexing
- Provides audit trail (when was node last valid)

**Cons:**
- Requires clock synchronization
- False positives if indexing is delayed
- Doesn't distinguish between "deleted" and "not checked yet"

**Recommendation:** Add `last_seen_at` as **future enhancement** for TTL-based cleanup. Not critical for MVP.

### 3.3 Version-Based Tracking

**Concept:** Track indexing version number, delete nodes from old versions.

**Schema Change:**
```cypher
CREATE (m:Memory {
    id: $memory_id,
    source: 'code_indexer',
    file_path: $file_path,
    index_version: $index_version,  // e.g., "2025-11-25T10:30:00Z"
    workspace_id: $workspace_id
})
```

**Cleanup Query:**
```cypher
MATCH (m:Memory)
WHERE m.source = 'code_indexer'
  AND m.workspace_id = $workspace_id
  AND m.index_version < $current_version
DETACH DELETE m
RETURN count(m) AS deleted_count
```

**Pros:**
- Clear versioning for auditing
- Easy rollback to previous index version
- No timing issues

**Cons:**
- Requires version management
- All-or-nothing (can't partially update)
- May delete nodes unnecessarily if indexing is incremental

**Recommendation:** Consider for **future enhancement** if audit requirements increase. Not needed for MVP.

---

## 4. Soft Delete vs Hard Delete

### 4.1 Soft Delete Strategy

**Concept:** Mark nodes as `deleted=true` instead of removing them immediately. Allows recovery and auditing.

**Cypher Query:**
```cypher
MATCH (m:Memory {id: $memory_id})
SET m.deleted = true,
    m.deleted_at = datetime(),
    m.deleted_by = $user_id
RETURN m.id
```

**Hard Delete After Grace Period:**
```cypher
MATCH (m:Memory)
WHERE m.deleted = true
  AND datetime(m.deleted_at) < datetime() - duration({days: 30})
DETACH DELETE m
RETURN count(m) AS purged_count
```

**Pros:**
- Safe recovery window
- Audit trail (who deleted what, when)
- Rollback capability
- Compliance-friendly

**Cons:**
- Storage overhead (deleted nodes remain)
- Query complexity (need `WHERE deleted = false` everywhere)
- Vector index pollution (deleted embeddings remain searchable)
- Requires periodic purge job

**Recommendation for Zapomni:**
- **MVP:** Hard delete only (simpler, faster)
- **Future:** Add soft delete if compliance or audit requirements emerge

### 4.2 Hard Delete Strategy

**Concept:** Immediately remove nodes and relationships with `DETACH DELETE`.

**Cypher Query:**
```cypher
MATCH (m:Memory {id: $memory_id})
OPTIONAL MATCH (m)-[:HAS_CHUNK]->(c:Chunk)
OPTIONAL MATCH (m)-[:MENTIONS]->(e:Entity)
DETACH DELETE m, c
RETURN count(m) AS deleted_count
```

**Pros:**
- Simple implementation
- Immediate space reclamation
- No query overhead
- Cleaner graph structure

**Cons:**
- No recovery after deletion
- Requires robust validation before deletion
- Must handle orphaned entities separately

**Recommendation:** Use **hard delete for MVP** with strong safety checks (dry_run, confirmation).

---

## 5. Timestamp-Based TTL Cleanup

### 5.1 Schema Enhancement

**Add Properties:**
```cypher
CREATE (m:Memory {
    id: $memory_id,
    created_at: datetime(),
    updated_at: datetime(),
    last_seen_at: datetime(),  // NEW: Last time node was verified
    workspace_id: $workspace_id
})
```

### 5.2 Update Query (During Re-indexing)

```cypher
MATCH (m:Memory {file_path: $file_path, workspace_id: $workspace_id})
SET m.last_seen_at = datetime(),
    m.updated_at = datetime()
RETURN m.id
```

### 5.3 TTL Cleanup Query

```cypher
MATCH (m:Memory)
WHERE m.source = 'code_indexer'
  AND m.workspace_id = $workspace_id
  AND datetime(m.last_seen_at) < datetime() - duration({days: $ttl_days})
OPTIONAL MATCH (m)-[:HAS_CHUNK]->(c:Chunk)
DETACH DELETE m, c
RETURN count(m) AS deleted_count
```

### 5.4 Implementation Notes

**Pros:**
- Automatic cleanup of stale data
- Works with incremental indexing
- Configurable retention policy

**Cons:**
- Requires background job (scheduled task)
- Clock drift issues across distributed systems
- False positives if indexing is delayed

**Recommendation:**
- **MVP:** Not required, implement as **Phase 2 enhancement**
- **Future:** Add `last_seen_at` property during next schema migration

---

## 6. FalkorDB/Cypher Patterns for Garbage Collection

### 6.1 Find Orphaned Memory Nodes

**Orphaned Memories (no chunks):**
```cypher
MATCH (m:Memory)
WHERE NOT (m)-[:HAS_CHUNK]->(:Chunk)
RETURN m.id, m.source, m.created_at, m.workspace_id
ORDER BY m.created_at DESC
LIMIT 100
```

**Orphaned Chunks (no parent memory):**
```cypher
MATCH (c:Chunk)
WHERE NOT (:Memory)-[:HAS_CHUNK]->(c)
RETURN c.id, c.workspace_id, size(c.text) AS text_length
LIMIT 100
```

**Orphaned Entities (no mentions):**
```cypher
MATCH (e:Entity)
WHERE NOT (:Chunk)-[:MENTIONS]->(e)
  AND NOT (e)-[:RELATED_TO]-(:Entity)
RETURN e.id, e.name, e.type, e.workspace_id
LIMIT 100
```

### 6.2 Find Stale Code Nodes (Non-Existent Files)

**Query for Code Memories:**
```cypher
MATCH (m:Memory)
WHERE m.source = 'code_indexer'
  AND m.workspace_id = $workspace_id
RETURN m.id, m.file_path, m.relative_path, m.created_at
```

**Filter in Application Code:**
```python
import os
from pathlib import Path

# Query graph for all code memories
result = await db_client.graph_query(query, {"workspace_id": workspace_id})

orphaned_ids = []
for row in result.rows:
    file_path = row["file_path"]
    if not Path(file_path).exists():
        orphaned_ids.append(row["id"])
```

**Batch Delete Orphaned Nodes:**
```cypher
UNWIND $orphaned_ids AS memory_id
MATCH (m:Memory {id: memory_id, workspace_id: $workspace_id})
OPTIONAL MATCH (m)-[:HAS_CHUNK]->(c:Chunk)
DETACH DELETE m, c
RETURN count(m) AS deleted_count
```

### 6.3 Delta Indexing Pattern (Mark-and-Sweep)

**Step 1 - Mark All as Stale:**
```cypher
MATCH (m:Memory)
WHERE m.source = 'code_indexer'
  AND m.workspace_id = $workspace_id
SET m.stale = true
RETURN count(m) AS marked_count
```

**Step 2 - Mark Fresh During Indexing:**
```cypher
MERGE (m:Memory {file_path: $file_path, workspace_id: $workspace_id})
ON MATCH SET m.stale = false,
             m.last_indexed_at = datetime(),
             m.updated_at = datetime()
ON CREATE SET m.id = $memory_id,
              m.source = 'code_indexer',
              m.stale = false,
              m.created_at = datetime(),
              m.last_indexed_at = datetime()
RETURN m.id, m.stale
```

**Step 3 - Delete Stale Nodes:**
```cypher
MATCH (m:Memory)
WHERE m.source = 'code_indexer'
  AND m.workspace_id = $workspace_id
  AND m.stale = true
OPTIONAL MATCH (m)-[:HAS_CHUNK]->(c:Chunk)
DETACH DELETE m, c
RETURN count(m) AS deleted_count
```

### 6.4 Dry Run Pattern (Preview Deletions)

**Count Nodes to Delete:**
```cypher
MATCH (m:Memory)
WHERE m.source = 'code_indexer'
  AND m.workspace_id = $workspace_id
  AND m.stale = true
OPTIONAL MATCH (m)-[:HAS_CHUNK]->(c:Chunk)
RETURN count(DISTINCT m) AS memory_count,
       count(DISTINCT c) AS chunk_count
```

**Preview Node Details:**
```cypher
MATCH (m:Memory)
WHERE m.source = 'code_indexer'
  AND m.workspace_id = $workspace_id
  AND m.stale = true
RETURN m.id, m.file_path, m.relative_path, m.created_at, m.lines
ORDER BY m.created_at DESC
LIMIT 20
```

---

## 7. Safety Mechanisms

### 7.1 Dry Run by Default

**Implementation:**
```python
async def prune_memory(
    workspace_id: str = "default",
    dry_run: bool = True,  # DEFAULT: True
    confirm: bool = False,
) -> Dict[str, Any]:
    """
    Prune stale memory nodes from knowledge graph.

    Args:
        workspace_id: Workspace to prune (default: "default")
        dry_run: If True, only preview deletions (default: True)
        confirm: If False, require user confirmation (default: False)

    Returns:
        {
            "dry_run": bool,
            "nodes_to_delete": int,
            "chunks_to_delete": int,
            "preview": List[Dict],  // First 20 nodes
            "deleted_count": int,   // Only if dry_run=False
        }
    """
    if dry_run:
        # Preview only
        result = await db_client.graph_query(preview_query, params)
        return format_preview(result)

    if not confirm:
        raise ValueError(
            "Deletion requires explicit confirmation. "
            "Set confirm=True to proceed."
        )

    # Execute deletion
    result = await db_client.graph_query(delete_query, params)
    return format_result(result)
```

**Safety Checks:**
1. **Default to dry_run=True** - Users must opt-in to deletion
2. **Require explicit confirmation** - `confirm=True` required for deletion
3. **Preview before deletion** - Show exactly what will be deleted
4. **Workspace isolation** - Cannot accidentally delete across workspaces

### 7.2 Confirmation Workflow

**Step 1 - Preview (Dry Run):**
```bash
# User calls: prune_memory(workspace_id="my_project")
# Response:
{
  "dry_run": true,
  "nodes_to_delete": 42,
  "chunks_to_delete": 156,
  "preview": [
    {
      "id": "uuid-1",
      "file_path": "/path/to/deleted_file.py",
      "created_at": "2025-11-20T10:30:00Z"
    },
    ...
  ],
  "message": "Dry run complete. Set dry_run=False and confirm=True to delete."
}
```

**Step 2 - Confirm Deletion:**
```bash
# User calls: prune_memory(workspace_id="my_project", dry_run=False, confirm=True)
# Response:
{
  "dry_run": false,
  "deleted_count": 42,
  "chunks_deleted": 156,
  "message": "Successfully deleted 42 stale memory nodes and 156 chunks."
}
```

### 7.3 Validation Before Deletion

**Pre-deletion Checks:**
```python
async def validate_deletion_safety(
    workspace_id: str,
    memory_ids: List[str],
) -> Dict[str, Any]:
    """
    Validate deletion is safe before executing.

    Returns:
        {
            "is_safe": bool,
            "warnings": List[str],
            "blocked_ids": List[str],  // IDs that should not be deleted
        }
    """
    warnings = []
    blocked_ids = []

    # Check 1: Verify workspace exists
    workspace = await db_client.get_workspace(workspace_id)
    if not workspace:
        return {
            "is_safe": False,
            "warnings": [f"Workspace '{workspace_id}' does not exist"],
            "blocked_ids": memory_ids,
        }

    # Check 2: Verify memories belong to workspace
    for memory_id in memory_ids:
        query = "MATCH (m:Memory {id: $id}) RETURN m.workspace_id AS ws"
        result = await db_client.graph_query(query, {"id": memory_id})

        if result.row_count == 0:
            warnings.append(f"Memory {memory_id} not found")
            blocked_ids.append(memory_id)
        elif result.rows[0]["ws"] != workspace_id:
            warnings.append(f"Memory {memory_id} belongs to different workspace")
            blocked_ids.append(memory_id)

    # Check 3: Prevent deletion of default workspace
    if workspace_id == "default":
        warnings.append("WARNING: Deleting from default workspace")

    return {
        "is_safe": len(blocked_ids) == 0,
        "warnings": warnings,
        "blocked_ids": blocked_ids,
    }
```

### 7.4 Transaction Safety

**Use Database Transactions:**
```python
async def prune_stale_nodes(
    workspace_id: str,
    dry_run: bool,
    confirm: bool,
) -> Dict[str, Any]:
    """
    Prune stale nodes with transaction safety.
    """
    if dry_run:
        return await preview_deletion(workspace_id)

    if not confirm:
        raise ValueError("Deletion requires confirm=True")

    # Execute in transaction
    try:
        # FalkorDB queries are atomic within a single query
        result = await db_client.graph_query(delete_query, params)
        return {"deleted_count": result.row_count}
    except Exception as e:
        # Transaction automatically rolled back on error
        logger.error("deletion_failed", error=str(e))
        raise DatabaseError(f"Failed to prune nodes: {e}")
```

**Note:** FalkorDB/Cypher queries are atomic. Multi-step operations should be combined into single query when possible.

### 7.5 Backup Before Deletion (Optional)

**Export Graph Before Pruning:**
```python
async def backup_before_prune(workspace_id: str) -> str:
    """
    Export graph to JSON before pruning.

    Returns:
        backup_path: Path to backup file
    """
    from zapomni_core.graph.graph_exporter import GraphExporter

    exporter = GraphExporter(db_client=db_client)

    backup_path = f"/tmp/zapomni_backup_{workspace_id}_{datetime.now().isoformat()}.json"

    await exporter.export_graph(
        workspace_id=workspace_id,
        format="json",
        output_path=backup_path,
    )

    return backup_path
```

**Recommendation:**
- **MVP:** Not required, too complex for manual-only operation
- **Future:** Add if automated GC is implemented

---

## 8. Recommended GC Strategy for Zapomni MVP

### 8.1 Implementation Plan

**Phase 1: Manual Garbage Collection (MVP)**

**MCP Tool:** `prune_memory`

**Parameters:**
- `workspace_id` (string, default="default") - Workspace to prune
- `dry_run` (bool, default=True) - Preview deletions only
- `confirm` (bool, default=False) - Explicit confirmation required
- `strategy` (string, default="stale_files") - GC strategy to use
  - `"stale_files"` - Delete code nodes for non-existent files
  - `"orphaned_chunks"` - Delete chunks without parent memories
  - `"orphaned_entities"` - Delete entities without mentions
  - `"all"` - Run all strategies

**Implementation Steps:**

1. **Add `stale` property to schema**
   - Modify `SchemaManager` to add `stale: bool` property to `Memory` nodes
   - Default to `false` on creation

2. **Update `index_codebase` tool:**
   - Before indexing: Mark all code memories as `stale=true`
   - During indexing: Set `stale=false` for files that still exist
   - After indexing: Return count of stale nodes (for user awareness)

3. **Implement `prune_memory` MCP tool:**
   ```python
   async def prune_memory(
       workspace_id: str = "default",
       dry_run: bool = True,
       confirm: bool = False,
       strategy: str = "stale_files",
   ) -> Dict[str, Any]:
       """Prune stale memory nodes."""

       # Validate inputs
       if workspace_id == "":
           workspace_id = "default"

       # Select strategy
       if strategy == "stale_files":
           query = QUERY_STALE_FILES
       elif strategy == "orphaned_chunks":
           query = QUERY_ORPHANED_CHUNKS
       elif strategy == "orphaned_entities":
           query = QUERY_ORPHANED_ENTITIES
       elif strategy == "all":
           # Run all strategies sequentially
           results = []
           for s in ["stale_files", "orphaned_chunks", "orphaned_entities"]:
               result = await prune_memory(workspace_id, dry_run, confirm, s)
               results.append(result)
           return aggregate_results(results)
       else:
           raise ValueError(f"Unknown strategy: {strategy}")

       # Preview or execute
       if dry_run:
           result = await db_client.graph_query(query, params)
           return format_preview(result)

       if not confirm:
           raise ValueError("Deletion requires confirm=True")

       # Execute deletion
       result = await db_client.graph_query(query, params)
       return format_result(result)
   ```

4. **Cypher Queries:**
   ```cypher
   -- Query: Find stale code files
   MATCH (m:Memory)
   WHERE m.source = 'code_indexer'
     AND m.workspace_id = $workspace_id
     AND m.stale = true
   OPTIONAL MATCH (m)-[:HAS_CHUNK]->(c:Chunk)
   RETURN m.id, m.file_path, m.created_at,
          count(c) AS chunk_count
   ```

   ```cypher
   -- Query: Delete stale code files
   MATCH (m:Memory)
   WHERE m.source = 'code_indexer'
     AND m.workspace_id = $workspace_id
     AND m.stale = true
   OPTIONAL MATCH (m)-[:HAS_CHUNK]->(c:Chunk)
   DETACH DELETE m, c
   RETURN count(DISTINCT m) AS deleted_memories,
          count(DISTINCT c) AS deleted_chunks
   ```

### 8.2 User Workflow

**Step 1: Re-index Codebase**
```bash
index_codebase(repo_path="/path/to/project", workspace_id="my_project")
# Response: "42 files indexed, 5 stale nodes detected"
```

**Step 2: Preview Deletions**
```bash
prune_memory(workspace_id="my_project", dry_run=True)
# Response:
# {
#   "dry_run": true,
#   "nodes_to_delete": 5,
#   "chunks_to_delete": 18,
#   "preview": [
#     {"id": "uuid-1", "file_path": "/path/to/deleted_file.py"},
#     ...
#   ]
# }
```

**Step 3: Confirm Deletion**
```bash
prune_memory(workspace_id="my_project", dry_run=False, confirm=True)
# Response:
# {
#   "deleted_count": 5,
#   "chunks_deleted": 18,
#   "message": "Successfully deleted 5 stale memory nodes."
# }
```

### 8.3 Safety Guarantees

1. **Default to Safe Mode:** `dry_run=True` by default
2. **Explicit Confirmation:** Require `confirm=True` for deletion
3. **Workspace Isolation:** Cannot delete across workspaces
4. **Preview Before Delete:** Show exactly what will be removed
5. **Atomic Deletion:** Single Cypher query ensures consistency
6. **Validation:** Check workspace exists, verify node ownership

### 8.4 Edge Cases Handled

1. **Concurrent Re-indexing:**
   - Use `MERGE` instead of `CREATE` to avoid duplicates
   - `stale` flag handles concurrent updates gracefully

2. **Deleted Files:**
   - `stale=true` marks them for cleanup
   - User decides when to prune (manual trigger)

3. **Modified Files:**
   - Re-indexing creates new nodes
   - Old nodes marked as `stale=true`
   - Prune removes old versions

4. **Circular References:**
   - `DETACH DELETE` removes all relationships
   - No orphaned edges left behind

5. **Empty Workspaces:**
   - Query returns zero nodes to delete
   - Safe to run repeatedly

---

## 9. Risks and Mitigations

### 9.1 Risk: Accidental Data Loss

**Scenario:** User runs `prune_memory()` with `dry_run=False` and `confirm=True` by mistake.

**Mitigation:**
1. **Default to `dry_run=True`** - Force explicit opt-in
2. **Require `confirm=True`** - Double confirmation
3. **Preview before deletion** - Show what will be deleted
4. **Logging** - Record all deletion operations with timestamps
5. **Workspace isolation** - Prevent cross-workspace deletion

**Implementation:**
```python
if not dry_run and not confirm:
    raise ValueError(
        "Deletion requires explicit confirmation. "
        "Set confirm=True to proceed. "
        "Run with dry_run=True first to preview."
    )
```

### 9.2 Risk: Deleting Valid Data

**Scenario:** A file is temporarily moved or renamed, marked as stale, then deleted.

**Mitigation:**
1. **Manual trigger only (MVP)** - User controls when to prune
2. **Grace period** - Don't prune immediately after re-indexing
3. **Preview** - User reviews list before confirming
4. **Timestamp tracking** - Add `last_seen_at` to detect recent changes

**Recommendation:** Add warning if nodes are less than 24 hours old:
```python
recent_threshold = datetime.now() - timedelta(days=1)
recent_nodes = [n for n in nodes if n.created_at > recent_threshold]
if recent_nodes:
    warnings.append(f"WARNING: {len(recent_nodes)} nodes are less than 24h old")
```

### 9.3 Risk: Performance Impact on Large Graphs

**Scenario:** Pruning operation takes too long or locks database.

**Mitigation:**
1. **Batch deletion** - Delete in chunks of 100-1000 nodes
2. **Timeout limits** - Fail fast if query exceeds threshold
3. **Background job** - Run as async task (future enhancement)
4. **Index optimization** - Ensure `stale` property is indexed

**Implementation:**
```cypher
-- Batch delete (100 nodes at a time)
MATCH (m:Memory)
WHERE m.source = 'code_indexer'
  AND m.workspace_id = $workspace_id
  AND m.stale = true
WITH m LIMIT 100
OPTIONAL MATCH (m)-[:HAS_CHUNK]->(c:Chunk)
DETACH DELETE m, c
RETURN count(m) AS deleted_count
```

### 9.4 Risk: Orphaned Entities After Deletion

**Scenario:** Deleting memories leaves orphaned entities without mentions.

**Mitigation:**
1. **Cascade deletion** - Use `DETACH DELETE` to remove relationships
2. **Separate orphan cleanup** - Add `strategy="orphaned_entities"`
3. **Reference counting** - Track incoming edges to entities
4. **Manual review** - Preview entities before deletion

**Implementation:**
```cypher
-- Find and delete orphaned entities
MATCH (e:Entity)
WHERE NOT (:Chunk)-[:MENTIONS]->(e)
  AND NOT (e)-[:RELATED_TO]-(:Entity)
  AND e.workspace_id = $workspace_id
DETACH DELETE e
RETURN count(e) AS deleted_count
```

### 9.5 Risk: Vector Index Desynchronization

**Scenario:** Deleting chunks but vector index entries remain.

**Mitigation:**
1. **FalkorDB handles this automatically** - Deleting nodes removes from vector index
2. **Verify after deletion** - Check index stats
3. **Manual index rebuild** - Add admin tool if needed (future)

**Verification Query:**
```cypher
CALL db.indexes() YIELD name, type, properties, status
WHERE name = 'chunk_embedding_idx'
RETURN name, status, properties
```

---

## 10. Future Enhancements (Post-MVP)

### 10.1 Automated Background GC

**Concept:** Scheduled task runs GC automatically.

**Implementation:**
- Add background job runner (e.g., APScheduler)
- Run `prune_memory()` nightly at 2 AM
- Send summary email/notification
- Configurable via `config.yaml`

**Configuration:**
```yaml
garbage_collection:
  enabled: true
  schedule: "0 2 * * *"  # Cron: 2 AM daily
  strategy: "stale_files"
  workspace_id: "default"
  ttl_days: 30
  notify_email: "admin@example.com"
```

### 10.2 Soft Delete with Recovery

**Implementation:**
- Add `deleted: bool` property
- Add `deleted_at: datetime` property
- Add `deleted_by: str` property
- Modify queries to filter `WHERE deleted = false`
- Add recovery tool: `recover_memory(memory_id)`

**Recovery Window:** 30 days, then purge permanently.

### 10.3 Graph Compaction

**Concept:** Rebuild graph to reclaim space and optimize structure.

**Implementation:**
- Export graph to JSON
- Drop and recreate graph
- Re-import optimized structure
- Rebuild indexes

**Use Case:** After large deletions to reclaim disk space.

### 10.4 Entity Merging

**Concept:** Merge duplicate entities detected after cleanup.

**Implementation:**
- Find entities with same `name` and `type`
- Merge relationships
- Update `MENTIONS` edges
- Delete duplicate entity

**Example:**
```cypher
MATCH (e1:Entity {name: "Python"}), (e2:Entity {name: "Python"})
WHERE e1.id < e2.id
MATCH (e2)-[r]->(other)
CREATE (e1)-[r2:RELATED_TO]->(other)
SET r2 = properties(r)
DETACH DELETE e2
RETURN count(e2) AS merged_count
```

### 10.5 Audit Log

**Concept:** Track all GC operations for compliance.

**Schema:**
```cypher
CREATE (log:GCLog {
    id: $log_id,
    operation: "prune_memory",
    workspace_id: $workspace_id,
    deleted_count: $deleted_count,
    strategy: $strategy,
    executed_by: $user_id,
    executed_at: datetime(),
    dry_run: $dry_run
})
```

**Query Logs:**
```cypher
MATCH (log:GCLog)
WHERE log.workspace_id = $workspace_id
RETURN log.operation, log.deleted_count, log.executed_at
ORDER BY log.executed_at DESC
LIMIT 20
```

---

## 11. Conclusion

### 11.1 Recommended MVP Implementation

**For Zapomni MVP, implement:**

1. **Manual GC via `prune_memory()` MCP tool**
   - Default: `dry_run=True`, `confirm=False`
   - Strategies: `stale_files`, `orphaned_chunks`, `orphaned_entities`
   - Safety: Preview, confirmation, workspace isolation

2. **Delta indexing in `index_codebase`**
   - Mark all as `stale=true` before indexing
   - Mark fresh as `stale=false` during indexing
   - User manually runs `prune_memory()` to cleanup

3. **Cypher queries for GC**
   - Find orphaned nodes (no incoming edges)
   - Find stale code files (deleted from disk)
   - Batch delete with `DETACH DELETE`

4. **Safety mechanisms**
   - Dry run by default
   - Explicit confirmation required
   - Preview before deletion
   - Transaction safety

**Do NOT implement (defer to future):**
- Automated background GC
- Soft delete with recovery
- TTL-based cleanup
- Graph compaction
- Audit logging

### 11.2 Implementation Checklist

- [ ] Add `stale: bool` property to `Memory` schema
- [ ] Update `index_codebase` tool to mark nodes as stale
- [ ] Implement `prune_memory()` MCP tool with strategies
- [ ] Write Cypher queries for orphan detection
- [ ] Add validation and safety checks
- [ ] Write unit tests for GC logic
- [ ] Write integration tests for end-to-end workflow
- [ ] Update documentation with user workflow
- [ ] Add logging for all GC operations

### 11.3 Success Metrics

**MVP Success Criteria:**
- ✅ User can preview deletions before executing
- ✅ User must explicitly confirm deletion
- ✅ Stale code nodes are correctly identified
- ✅ Deletion is atomic and consistent
- ✅ No valid data is accidentally deleted
- ✅ GC completes in < 5 seconds for < 1000 nodes

### 11.4 Next Steps

1. **Create design doc** from this research
2. **Implement schema changes** (`stale` property)
3. **Update `index_codebase`** tool with marking logic
4. **Implement `prune_memory`** MCP tool
5. **Write tests** (unit + integration)
6. **Document user workflow** in README
7. **Review and merge** to main branch

---

## References

### Graph Database Garbage Collection Patterns
- [Neo4j Manual: Delete](https://neo4j.com/docs/cypher-manual/current/clauses/delete/)
- [FalkorDB Documentation](https://docs.falkordb.com/)
- [Graph Database Internals](https://www.oreilly.com/library/view/graph-databases/9781449356262/)

### Delta Indexing Patterns
- [Incremental Indexing in Elasticsearch](https://www.elastic.co/guide/en/elasticsearch/reference/current/docs-update.html)
- [Delta Lake: ACID Transactions](https://delta.io/)
- [Lucene Index Lifecycle](https://lucene.apache.org/core/9_0_0/core/org/apache/lucene/index/IndexWriter.html)

### Code Indexing Best Practices
- [Sourcegraph Indexing Architecture](https://docs.sourcegraph.com/)
- [GitHub Code Search](https://github.blog/2021-12-08-improving-github-code-search/)
- [Kythe: Code Indexing Protocol](https://kythe.io/)

---

**Report prepared by:** Research Agent
**Date:** 2025-11-25
**Version:** 1.0
**Status:** Final
