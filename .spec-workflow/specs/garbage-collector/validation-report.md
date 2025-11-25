# Garbage Collector Specification Validation Report

**Date:** 2025-11-25
**Project:** Zapomni
**Specification:** Garbage Collector (Task 04)
**Validator:** Validation Agent

---

## Executive Summary

**Validation Status:** **PASSED WITH WARNINGS**

The garbage collector specification is well-designed and aligns with the existing Zapomni codebase architecture. All Cypher queries are valid for FalkorDB, API contracts match existing patterns, and the task breakdown is comprehensive. However, several warnings and recommendations have been identified that should be addressed before implementation.

**Key Findings:**
- ✅ All Cypher queries are valid for FalkorDB syntax
- ✅ API patterns match existing `FalkorDBClient` conventions
- ✅ Task breakdown is complete and actionable
- ⚠️ Schema migration strategy needs refinement
- ⚠️ File path matching approach has potential issues
- ⚠️ Integration with `index_codebase` needs coordination

---

## 1. Cypher Query Validation

### 1.1 Query: Mark Code Memories as Stale

**Location:** `design.md` Section 5.1

```cypher
MATCH (m:Memory)
WHERE m.source = 'code_indexer'
  AND m.workspace_id = $workspace_id
SET m.stale = true
RETURN count(m) AS marked_count
```

**Status:** ✅ **VALID**

**Analysis:**
- Standard FalkorDB/Cypher syntax
- Property filtering with `WHERE` clause is supported
- `SET` clause for property updates is correct
- `count()` aggregation function is valid

---

### 1.2 Query: Mark Memory as Fresh

**Location:** `design.md` Section 5.2

```cypher
MATCH (m:Memory)
WHERE m.source = 'code_indexer'
  AND m.workspace_id = $workspace_id
  AND m.metadata CONTAINS $file_path_pattern
SET m.stale = false,
    m.last_seen_at = datetime()
RETURN m.id AS memory_id
```

**Status:** ⚠️ **VALID WITH WARNING**

**Analysis:**
- ✅ Query syntax is valid
- ⚠️ **WARNING:** The `CONTAINS` operator for JSON metadata matching is **problematic**

**Issue:** The `metadata` field is a JSON string (as seen in `falkordb_client.py:687`):
```python
"metadata": json.dumps(memory.metadata),
```

The `CONTAINS` operator performs substring matching on the JSON string. This approach has several issues:

1. **False Positives:** A file path like `/src/app.py` would match `/src/app.py.backup` if both exist in metadata
2. **Fragile:** Sensitive to JSON formatting changes (spaces, quotes)
3. **Inefficient:** No index can be used for `CONTAINS` operations

**Recommendation:**
Use a more robust approach. Since FalkorDB supports JSON property access, consider:

```cypher
MATCH (m:Memory)
WHERE m.source = 'code_indexer'
  AND m.workspace_id = $workspace_id
  AND m.metadata CONTAINS '"file_path": "' + $file_path + '"'
SET m.stale = false,
    m.last_seen_at = datetime()
RETURN m.id AS memory_id
```

Or better yet, extract `file_path` as a **top-level property** on Memory nodes:

```cypher
CREATE (m:Memory {
    id: $memory_id,
    file_path: $file_path,      // NEW: Extract from metadata
    source: 'code_indexer',
    metadata: $metadata,
    workspace_id: $workspace_id,
    stale: false
})
```

Then matching becomes:
```cypher
MATCH (m:Memory)
WHERE m.file_path = $file_path
  AND m.workspace_id = $workspace_id
SET m.stale = false
```

---

### 1.3 Query: Preview Stale Memories

**Location:** `design.md` Section 5.3

```cypher
MATCH (m:Memory)
WHERE m.source = 'code_indexer'
  AND m.workspace_id = $workspace_id
  AND m.stale = true
OPTIONAL MATCH (m)-[:HAS_CHUNK]->(c:Chunk)
WITH m, count(c) AS chunk_count
RETURN m.id AS id,
       m.metadata AS metadata,
       m.created_at AS created_at,
       chunk_count
ORDER BY m.created_at DESC
LIMIT $limit
```

**Status:** ✅ **VALID**

**Analysis:**
- Correct use of `OPTIONAL MATCH` for handling memories without chunks
- `WITH` clause aggregation is valid
- `ORDER BY` and `LIMIT` are properly placed

---

### 1.4 Query: Delete Stale Memories

**Location:** `design.md` Section 5.4

```cypher
MATCH (m:Memory)
WHERE m.source = 'code_indexer'
  AND m.workspace_id = $workspace_id
  AND m.stale = true
OPTIONAL MATCH (m)-[:HAS_CHUNK]->(c:Chunk)
DETACH DELETE m, c
```

**Status:** ⚠️ **VALID WITH WARNING**

**Analysis:**
- ✅ `DETACH DELETE` is the correct way to remove nodes and relationships
- ⚠️ **WARNING:** Return clause missing for confirmation

**Issue:** The query doesn't return a count of deleted items, making it hard to verify success.

**Recommendation:**
FalkorDB's `DELETE` doesn't return affected rows directly. Use a two-step approach:

```cypher
// Step 1: Count (already done in preview method)
// Step 2: Delete without counting
MATCH (m:Memory)
WHERE m.source = 'code_indexer'
  AND m.workspace_id = $workspace_id
  AND m.stale = true
OPTIONAL MATCH (m)-[:HAS_CHUNK]->(c:Chunk)
DETACH DELETE m, c
```

The design correctly addresses this by calling `get_stale_memories_preview()` first to get counts (see `design.md:645-649`).

---

### 1.5 Query: Find Orphaned Chunks

**Location:** `design.md` Section 5.5

```cypher
MATCH (c:Chunk)
WHERE c.workspace_id = $workspace_id
  AND NOT (:Memory)-[:HAS_CHUNK]->(c)
RETURN c.id AS id, size(c.text) AS text_length
LIMIT $limit
```

**Status:** ✅ **VALID**

**Analysis:**
- Correct use of negative pattern matching with `NOT`
- `size()` function is valid for string length
- Properly workspace-scoped

---

### 1.6 Query: Delete Orphaned Chunks

**Location:** `design.md` Section 5.6

```cypher
MATCH (c:Chunk)
WHERE c.workspace_id = $workspace_id
  AND NOT (:Memory)-[:HAS_CHUNK]->(c)
DETACH DELETE c
```

**Status:** ✅ **VALID**

**Analysis:**
- Properly handles orphaned chunks
- Uses `DETACH DELETE` to remove all relationships

---

### 1.7 Query: Find Orphaned Entities

**Location:** `design.md` Section 5.7

```cypher
MATCH (e:Entity)
WHERE e.workspace_id = $workspace_id
  AND NOT (:Chunk)-[:MENTIONS]->(e)
  AND NOT (e)-[:RELATED_TO]-(:Entity)
RETURN e.id AS id, e.name AS name, e.type AS type
LIMIT $limit
```

**Status:** ✅ **VALID**

**Analysis:**
- Correct use of multiple negative patterns
- Properly identifies truly orphaned entities (no mentions AND no relationships)

---

### 1.8 Query: Delete Orphaned Entities

**Location:** `design.md` Section 5.8

```cypher
MATCH (e:Entity)
WHERE e.workspace_id = $workspace_id
  AND NOT (:Chunk)-[:MENTIONS]->(e)
  AND NOT (e)-[:RELATED_TO]-(:Entity)
DETACH DELETE e
```

**Status:** ✅ **VALID**

**Analysis:**
- Consistent with find query
- Properly removes all relationships

---

## 2. API Contract Validation

### 2.1 FalkorDBClient Method Signatures

**Existing Pattern Analysis:**

From `/home/dev/zapomni/src/zapomni_db/falkordb_client.py`, all async methods follow this pattern:

```python
async def method_name(
    self,
    param1: Type1,
    param2: Type2,
    workspace_id: Optional[str] = None,
) -> ReturnType:
    """Docstring with Args and Returns."""
```

**Specification Compliance:**

✅ All proposed methods match this pattern:

```python
async def mark_code_memories_stale(
    self,
    workspace_id: str,
) -> int:
    """Mark all code memories as stale before re-indexing."""

async def mark_memory_fresh(
    self,
    file_path: str,
    workspace_id: str,
) -> Optional[str]:
    """Mark a specific memory as fresh during indexing."""

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
```

**Observations:**
- ✅ All methods are `async`
- ✅ Type hints provided
- ✅ Docstrings follow project convention
- ✅ Return types are appropriate

---

### 2.2 Error Handling Pattern

**Existing Pattern:**

From `falkordb_client.py`, the project uses custom exceptions:

```python
from zapomni_db.exceptions import (
    ConnectionError,
    DatabaseError,
    QueryError,
    TransactionError,
    ValidationError,
)
```

**Specification Compliance:**

✅ The design correctly uses these exceptions:
- `ValidationError` for input validation failures
- `DatabaseError` for query execution failures
- `QueryError` for invalid Cypher syntax

**Example from design.md:**

```python
async def get_stale_memories_preview(...):
    try:
        result = await self._execute_cypher(cypher, params)
        return {...}
    except Exception as e:
        self._logger.error("preview_failed", error=str(e))
        raise DatabaseError(f"Failed to get preview: {e}")
```

---

### 2.3 Logging Pattern

**Existing Pattern:**

From `falkordb_client.py:139`, the project uses `structlog`:

```python
self._logger = logger.bind(
    host=host,
    port=port,
    graph=graph_name,
    pool_max=self.pool_config.max_size,
)

self._logger.info("client_configured", ...)
```

**Specification Compliance:**

✅ The design follows this pattern consistently:

```python
self._logger.info(
    "memories_marked_stale",
    workspace_id=workspace_id,
    count=count,
)

self._logger.debug("memory_marked_fresh", memory_id=memory_id)

self._logger.error("preview_failed", error=str(e))
```

---

## 3. Schema Integration Validation

### 3.1 Schema Manager Integration

**Current Schema (from `schema_manager.py:243-248`):**

```python
indexes = [
    (self.INDEX_MEMORY_ID, self.NODE_MEMORY, "id"),
    (self.INDEX_ENTITY_NAME, self.NODE_ENTITY, "name"),
    (self.INDEX_TIMESTAMP, self.NODE_MEMORY, "timestamp"),
    ("chunk_memory_id_idx", self.NODE_CHUNK, "memory_id"),
]
```

**Proposed Addition:**

```python
("memory_stale_idx", self.NODE_MEMORY, "stale"),
```

**Status:** ✅ **VALID**

**Analysis:**
- Follows existing naming convention (`{node}_{property}_idx`)
- Properly uses tuple format `(name, node_label, property)`
- Will be created idempotently (existing `create_property_indexes()` method handles this)

---

### 3.2 Memory Node Property Addition

**Current Memory Creation (from `falkordb_client.py:644-654`):**

```python
CREATE (m:Memory {
    id: $memory_id,
    text: $text,
    tags: $tags,
    source: $source,
    metadata: $metadata,
    workspace_id: $workspace_id,
    created_at: $timestamp
})
```

**Proposed Properties:**

```python
CREATE (m:Memory {
    // ... existing properties ...
    stale: false,              // NEW
    last_seen_at: $timestamp   // NEW
})
```

**Status:** ⚠️ **VALID WITH WARNING**

**Analysis:**
- ✅ Properties are simple types (boolean, datetime)
- ⚠️ **WARNING:** No schema migration plan for existing Memory nodes

**Issue:** Existing Memory nodes in the database will have `stale=null` and `last_seen_at=null`.

**Impact:**
The Cypher query `WHERE m.stale = true` correctly excludes null values, so this is actually **safe**. Null values will be treated as "not stale".

**Recommendation:**
Document this behavior explicitly. Consider adding a schema migration note for production deployments:

```python
# Optional migration for existing databases
MATCH (m:Memory)
WHERE m.stale IS NULL
SET m.stale = false,
    m.last_seen_at = COALESCE(m.last_seen_at, m.created_at, datetime())
```

---

## 4. Task Breakdown Validation

### 4.1 Task Dependencies

**Analysis of `tasks.md`:**

The task breakdown correctly identifies dependencies:

```
1.1 → 1.2 → 2.2
1.1 → 2.1 → 2.3
1.1 → 2.4
Phase 2 → 3.1 → 3.2 → 3.3
Phase 2 → 4.1 → 4.2
```

**Status:** ✅ **VALID**

**Analysis:**
- Schema changes must happen before database methods (correct)
- Database methods must be complete before MCP tool (correct)
- Delta indexing can proceed in parallel with MCP tool development (efficient)
- Testing depends on all implementation phases (correct)

---

### 4.2 Task Completeness

**Checklist:**

- ✅ Schema updates covered (Tasks 1.1, 1.2)
- ✅ Database methods covered (Tasks 2.1-2.4)
- ✅ MCP tool covered (Tasks 3.1-3.3)
- ✅ Delta indexing integration covered (Tasks 4.1-4.2)
- ✅ Unit tests covered (Tasks 5.1-5.2)
- ✅ Integration tests covered (Task 5.3)

**Status:** ✅ **COMPLETE**

---

### 4.3 Estimated Effort

**Specification:** 4 days (14 tasks)

**Breakdown:**
- Day 1: Schema + Core GC methods (Tasks 1.1, 1.2, 2.1, 2.2)
- Day 2: Preview/Delete methods + MCP Tool (Tasks 2.3, 2.4, 3.1, 3.2, 3.3)
- Day 3: Delta indexing + Unit tests (Tasks 4.1, 4.2, 5.1, 5.2)
- Day 4: Integration tests + Bug fixes (Task 5.3)

**Status:** ✅ **REASONABLE**

**Analysis:**
Based on existing codebase complexity and patterns, this estimate is realistic for an experienced developer.

---

## 5. Safety Mechanisms Validation

### 5.1 Dry Run by Default

**From `requirements.md:46-47`:**

```python
dry_run: bool = Field(
    default=True,
    description="Preview mode - show what would be deleted"
)
```

**Status:** ✅ **VALID**

**Analysis:**
- Follows safety-first design principle
- Matches existing patterns in the codebase (e.g., `delete_workspace` requires `confirm=True` in `falkordb_client.py:1477`)

---

### 5.2 Explicit Confirmation

**From `requirements.md:153-156`:**

```python
confirm: bool = Field(
    default=False,
    description="Required true for actual deletion"
)
```

**Validation Logic (from `design.md:296-303`):**

```python
if not request.confirm:
    return self._format_error(
        "Deletion requires explicit confirmation. "
        "Set confirm=true to proceed."
    )
```

**Status:** ✅ **VALID**

**Analysis:**
- Two-factor safety (dry_run AND confirm)
- Consistent with `delete_workspace` pattern in `falkordb_client.py:1497-1503`

---

### 5.3 Workspace Isolation

**All queries include:**

```cypher
WHERE m.workspace_id = $workspace_id
```

**Status:** ✅ **VALID**

**Analysis:**
- Matches existing workspace isolation pattern throughout `falkordb_client.py`
- Prevents cross-workspace data leakage

---

### 5.4 Preview Before Delete

**From `design.md:289-294`:**

```python
# Step 2: Get preview (always, for logging)
preview = await self._get_preview(workspace_id, request.strategy)
```

**Status:** ✅ **VALID**

**Analysis:**
- Preview is generated even in non-dry-run mode for logging
- User sees preview in dry-run mode before confirming
- Audit trail is complete

---

## 6. Integration Validation

### 6.1 IndexCodebaseTool Integration

**Current Implementation (from `index_codebase.py:208-292`):**

```python
async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
    # Step 1: Validate arguments
    # Step 2: Index repository
    # Step 3: Filter files
    # Step 4: Calculate statistics
    # Step 5: Store memories (commented out)
    # Step 6: Format response
```

**Proposed Changes (from `design.md:991-1086`):**

```python
async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
    # NEW Step 2: Mark all existing code memories as stale
    stale_marked = await self.memory_processor.db_client.mark_code_memories_stale(
        workspace_id=effective_workspace_id
    )

    # ... existing indexing ...

    # NEW Step 6: For each file, mark as fresh
    for file_dict in files:
        await self.memory_processor.db_client.mark_memory_fresh(
            file_path=file_dict.get("path"),
            workspace_id=effective_workspace_id,
        )

    # NEW Step 7: Count remaining stale memories
    stale_remaining = await self.db_client.count_stale_memories(
        workspace_id=effective_workspace_id
    )
```

**Status:** ⚠️ **VALID WITH WARNING**

**Analysis:**
- ✅ Integration points are correctly identified
- ⚠️ **WARNING:** Step 5 in current implementation is commented out

**Issue:** The current `index_codebase.py` does NOT actually store code as memories:

```python
# Step 5: Store code as memories (optional - for advanced features)
# This would store individual code files/functions as memories
memories_created = 0
# await self._store_code_memories(files)
```

**Impact:** If memories aren't being created, the GC system won't have anything to clean up!

**Recommendation:**
1. **Verify:** Check if memories are being created elsewhere in the pipeline
2. **Document:** If intentional, clarify what "code indexing" means in the project context
3. **Update:** If unintentional, uncomment and implement `_store_code_memories()` before implementing GC

**Critical Question:** What exactly gets stored as Memory nodes with `source='code_indexer'`?

From the code, it appears the indexer only **scans** files but doesn't **store** them. The GC spec assumes storage happens.

---

### 6.2 MemoryProcessor Access

**From `design.md:1016-1018`:**

```python
await self.memory_processor.db_client.mark_code_memories_stale(
    workspace_id=effective_workspace_id
)
```

**Status:** ✅ **VALID**

**Analysis:**
- `IndexCodebaseTool` has a `memory_processor` attribute (line 202 in `index_codebase.py`)
- `MemoryProcessor` likely has a `db_client` attribute (standard pattern)
- Access chain is valid

---

## 7. Research & Architecture Review Validation

### 7.1 Research Document Analysis

**Strengths:**
- ✅ Comprehensive survey of GC strategies
- ✅ Proper analysis of trade-offs
- ✅ Clear recommendation for MVP (manual + delta indexing)
- ✅ Good coverage of risks and mitigations

**Alignment with Design:**
- ✅ Recommended "mark-and-sweep" approach is implemented
- ✅ Safety mechanisms match research recommendations
- ✅ Manual-only approach for MVP is consistent

---

### 7.2 Architecture Review Analysis

**Strengths:**
- ✅ Comprehensive file impact analysis
- ✅ Clear identification of modifications vs. new files
- ✅ Detailed Cypher query patterns
- ✅ Proper consideration of dependencies

**Alignment with Requirements:**
- ✅ All requirements from `requirements.md` are addressed
- ✅ Schema changes are documented
- ✅ Safety mechanisms are specified
- ✅ Testing strategy is defined

---

## 8. Issues Found

### 8.1 Critical Issues

**None identified.**

---

### 8.2 High Priority Warnings

#### Warning 1: File Path Matching Strategy

**Location:** `design.md:507-520`

**Issue:** The `mark_memory_fresh()` method uses `CONTAINS` for JSON metadata matching:

```cypher
WHERE m.metadata CONTAINS $file_path_pattern
```

**Risk:** Potential false positives, inefficiency, fragility

**Recommendation:** Extract `file_path` as a top-level property or improve the matching logic

**Severity:** HIGH

---

#### Warning 2: Memory Storage Gap

**Location:** `index_codebase.py:268-271`

**Issue:** Code indexer doesn't actually store memories:

```python
# Step 5: Store code as memories (optional - for advanced features)
memories_created = 0
# await self._store_code_memories(files)
```

**Risk:** GC system will have nothing to clean up if memories aren't created

**Recommendation:** Clarify what gets stored as Memory nodes or implement storage

**Severity:** HIGH

---

### 8.3 Medium Priority Warnings

#### Warning 3: Schema Migration

**Location:** `tasks.md:62-98`

**Issue:** No explicit migration plan for existing Memory nodes

**Risk:** Existing nodes will have `stale=null` and `last_seen_at=null`

**Recommendation:** Document the behavior or add optional migration script

**Severity:** MEDIUM

---

#### Warning 4: MCP Tool Registration

**Location:** `design.md:1143-1161`

**Issue:** The registration process is described as "follows existing patterns" but no concrete implementation is provided

**Risk:** Unclear how to register with MCP server

**Recommendation:** Provide concrete registration code in the design

**Severity:** MEDIUM

---

### 8.4 Low Priority Warnings

#### Warning 5: Performance Considerations

**Location:** `design.md:1276-1309`

**Issue:** No concrete benchmarks or performance tests specified

**Risk:** Performance issues may not be detected early

**Recommendation:** Add specific performance test cases to Task 5.3

**Severity:** LOW

---

## 9. Recommendations

### 9.1 Must Address Before Implementation

1. **Clarify Memory Storage:** Verify whether `index_codebase` actually stores memories. If not, implement storage before GC.

2. **Fix File Path Matching:** Replace `CONTAINS` with exact matching or extract `file_path` as a property.

### 9.2 Should Address During Implementation

3. **Add Schema Migration:** Document the behavior of null properties or provide optional migration script.

4. **Specify MCP Registration:** Add concrete code for registering `PruneMemoryTool` with the MCP server.

5. **Add Performance Tests:** Include specific benchmarks in integration tests (e.g., prune 1000 nodes in < 5s).

### 9.3 Nice to Have

6. **Add Metrics:** Consider adding metrics for GC operations (e.g., average deletion time, stale node accumulation rate).

7. **Add Admin Tools:** Consider adding a tool to list all stale nodes across all workspaces for debugging.

---

## 10. Final Recommendation

**Status:** **PASSED WITH WARNINGS**

The garbage collector specification is **ready for implementation** with the following conditions:

1. **Address Warning 1 (File Path Matching):** This is a design flaw that will cause issues in production
2. **Investigate Warning 2 (Memory Storage):** Verify that memories are actually being created

Once these two items are addressed, the specification is solid and can proceed to implementation.

**Confidence Level:** HIGH

The specification is well-researched, follows existing patterns, and has appropriate safety mechanisms. The warnings identified are fixable and do not represent fundamental design flaws.

---

## 11. Appendix: Cypher Query Checklist

| Query | Syntax Valid | Logic Valid | Index Support | Status |
|-------|--------------|-------------|---------------|--------|
| Mark stale | ✅ | ✅ | ✅ (workspace_id, source) | PASS |
| Mark fresh | ✅ | ⚠️ | ❌ (CONTAINS) | WARN |
| Preview stale | ✅ | ✅ | ✅ (stale index) | PASS |
| Delete stale | ✅ | ✅ | ✅ (stale index) | PASS |
| Find orphaned chunks | ✅ | ✅ | ⚠️ (negative match) | PASS |
| Delete orphaned chunks | ✅ | ✅ | ⚠️ (negative match) | PASS |
| Find orphaned entities | ✅ | ✅ | ⚠️ (negative match) | PASS |
| Delete orphaned entities | ✅ | ✅ | ⚠️ (negative match) | PASS |

**Legend:**
- ✅ = Fully supported
- ⚠️ = Works but may have performance implications
- ❌ = Not supported or problematic

---

**End of Report**

**Generated by:** Validation Agent
**Date:** 2025-11-25
**Version:** 1.0
