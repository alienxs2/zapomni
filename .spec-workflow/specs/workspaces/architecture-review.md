# Workspace Isolation - Architecture Review

**Date:** 2025-11-25
**Reviewer:** Architecture Agent
**Project:** Zapomni Memory Server
**Spec:** Workspace/Project Isolation for Knowledge Graph

---

## 1. Executive Summary

This architecture review analyzes how workspace isolation integrates with the existing Zapomni codebase. The recommended approach is **property-based filtering** with a **hybrid argument + session state** pattern for workspace resolution.

**Key Findings:**
- Current architecture supports workspace isolation with minimal structural changes
- Existing `SessionManager` handles SSE connection lifecycle but needs workspace state extension
- All database operations flow through `FalkorDBClient` and `MemoryProcessor` - ideal injection points
- MCP tools are stateless and delegate to `MemoryProcessor` - workspace context flows naturally

**Architectural Alignment:** HIGH - The proposed design follows existing patterns and requires no breaking changes to public APIs.

---

## 2. Architecture Alignment Analysis

### 2.1 Current Architecture Layers

```
+------------------+     +------------------+     +------------------+
|   MCP Transport  | --> |   MCP Server     | --> |   MCP Tools      |
| (stdio/SSE)      |     | (server.py)      |     | (tools/*.py)     |
+------------------+     +------------------+     +------------------+
                                  |                      |
                                  v                      v
                         +------------------+     +------------------+
                         | Session Manager  |     | Memory Processor |
                         | (SSE only)       |     | (core logic)     |
                         +------------------+     +------------------+
                                                        |
                                                        v
                                                 +------------------+
                                                 | FalkorDB Client  |
                                                 | (database)       |
                                                 +------------------+
```

### 2.2 Workspace Context Flow (Proposed)

```
+------------------+     +------------------+     +------------------+
|   MCP Transport  | --> |   MCP Server     | --> |   MCP Tools      |
|                  |     | + workspace ctx  |     | + workspace_id   |
+------------------+     +------------------+     +------------------+
        |                       |                      |
        v                       v                      v
+------------------+     +------------------+     +------------------+
| Workspace State  | <-- | Session Manager  |     | Memory Processor |
| (per session)    |     | + workspace mgmt |     | + workspace_id   |
+------------------+     +------------------+     +------------------+
                                                        |
                                                        v
                                                 +------------------+
                                                 | FalkorDB Client  |
                                                 | + workspace_id   |
                                                 | + Cypher filters |
                                                 +------------------+
```

### 2.3 Alignment with Existing Patterns

| Pattern | Current Implementation | Workspace Extension | Alignment |
|---------|----------------------|---------------------|-----------|
| Dependency Injection | MemoryProcessor receives all deps | Add workspace context injection | HIGH |
| Session Management | SessionManager tracks SSE sessions | Add workspace_id to SessionState | HIGH |
| Tool Delegation | Tools delegate to MemoryProcessor | Pass workspace_id through | HIGH |
| Query Building | CypherQueryBuilder parameterizes | Add workspace_id filter | HIGH |
| Validation | Pydantic models + custom validators | Add workspace_id validation | HIGH |

---

## 3. Affected Modules and Files

### 3.1 Core Database Layer

#### `/home/dev/zapomni/src/zapomni_db/falkordb_client.py`

**Impact Level:** HIGH - Core workspace injection point

**Modifications Required:**

| Line Range | Method | Change Description |
|------------|--------|-------------------|
| 484-609 | `add_memory()` | Add `workspace_id: str = "default"` parameter; inject into Memory node creation |
| 611-681 | `_execute_transaction()` | Add `workspace_id` to Cypher CREATE statements |
| 683-793 | `vector_search()` | Add `workspace_id: str = "default"` parameter; add WHERE filter |
| 829-955 | `get_stats()` | Add optional `workspace_id` for workspace-scoped stats |
| 961-1007 | `add_entity()` | Add `workspace_id: str = "default"` parameter |
| 1098-1176 | `get_related_entities()` | Add `workspace_id: str = "default"` parameter |
| 1178-1224 | `graph_query()` | Keep as-is (raw query interface, caller provides workspace filter) |
| 1226-1267 | `delete_memory()` | Add workspace validation (prevent cross-workspace delete) |

**New Methods to Add:**
```python
# After line 1287 (after clear_all)
async def create_workspace(self, workspace_id: str, metadata: Dict = None) -> Dict
async def list_workspaces(self) -> List[Dict]
async def delete_workspace(self, workspace_id: str, cascade: bool = False) -> Dict
async def get_workspace_stats(self, workspace_id: str) -> Dict
```

#### `/home/dev/zapomni/src/zapomni_db/models.py`

**Impact Level:** LOW - Optional model extension

**Modifications Required:**

| Line Range | Model | Change Description |
|------------|-------|-------------------|
| 32-53 | `Memory` | Add optional `workspace_id: str = "default"` field |
| 56-70 | `SearchResult` | Add optional `workspace_id: str = None` field |
| 81-87 | `Entity` | Add optional `workspace_id: str = "default"` field |

**New Models to Add:**
```python
# After line 104
class Workspace(BaseModel):
    """Workspace metadata node."""
    id: str = Field(..., min_length=1, max_length=63)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None

class WorkspaceSession(BaseModel):
    """Workspace context for MCP session."""
    session_id: str
    workspace_id: str = "default"
    created_at: datetime
    updated_at: datetime
```

#### `/home/dev/zapomni/src/zapomni_db/cypher_query_builder.py`

**Impact Level:** HIGH - All queries need workspace filter

**Modifications Required:**

| Line Range | Method | Change Description |
|------------|--------|-------------------|
| 56-160 | `build_add_memory_query()` | Add `workspace_id` to Memory and Chunk node creation |
| 162-254 | `build_vector_search_query()` | Add `workspace_id` filter in WHERE clause |
| 256-337 | `build_graph_traversal_query()` | Add `workspace_id` filter for Entity traversal |
| 339-374 | `build_stats_query()` | Add optional `workspace_id` for scoped stats |
| 376-421 | `build_delete_memory_query()` | Add `workspace_id` validation in MATCH |
| 423-494 | `build_add_entity_query()` | Add `workspace_id` to Entity creation |
| 496-600 | `build_add_relationship_query()` | Add `workspace_id` consistency check |
| 643-716 | `_build_filter_clause()` | Extend to include workspace filter as mandatory |

**New Methods to Add:**
```python
# Workspace CRUD query builders
def build_create_workspace_query(self, workspace_id: str, metadata: Dict) -> Tuple[str, Dict]
def build_list_workspaces_query(self) -> Tuple[str, Dict]
def build_delete_workspace_query(self, workspace_id: str, cascade: bool) -> Tuple[str, Dict]
def build_workspace_stats_query(self, workspace_id: str) -> Tuple[str, Dict]
```

### 3.2 Core Processing Layer

#### `/home/dev/zapomni/src/zapomni_core/memory_processor.py`

**Impact Level:** HIGH - Primary API layer for workspace context

**Modifications Required:**

| Line Range | Method | Change Description |
|------------|--------|-------------------|
| 335-511 | `add_memory()` | Add `workspace_id: Optional[str] = None` parameter; resolve from context |
| 513-700 | `search_memory()` | Add `workspace_id: Optional[str] = None` parameter |
| 702-816 | `get_stats()` | Add `workspace_id: Optional[str] = None` for scoped stats |
| 818-921 | `build_knowledge_graph()` | Add `workspace_id: Optional[str] = None` parameter |
| 923-983 | `get_related_entities()` | Add `workspace_id: Optional[str] = None` parameter |
| 1186-1239 | `_store_memory()` | Pass `workspace_id` to `db_client.add_memory()` |

**New Class Attribute:**
```python
# In __init__ (around line 220)
self._current_workspace: str = "default"  # Default workspace context
```

**New Methods to Add:**
```python
# Workspace context management
def set_current_workspace(self, workspace_id: str) -> None
def get_current_workspace(self) -> str

# Resolve workspace from explicit arg vs context
def _resolve_workspace(self, workspace_id: Optional[str]) -> str
```

### 3.3 MCP Server Layer

#### `/home/dev/zapomni/src/zapomni_mcp/server.py`

**Impact Level:** MEDIUM - Tool call interception for workspace injection

**Modifications Required:**

| Line Range | Method | Change Description |
|------------|--------|-------------------|
| 109-167 | `__init__()` | Add `_workspace_manager: WorkspaceManager` instance |
| 276-359 | `run()` (stdio) | Workspace context not applicable (no session state) |
| 361-474 | `run_sse()` | Inject workspace context from session into tool calls |
| 420-444 | `handle_call_tool()` | Extract/inject `workspace_id` from arguments/session |

**New Import to Add:**
```python
from zapomni_mcp.workspace_manager import WorkspaceManager
```

**Key Change in `handle_call_tool()` (lines 420-444):**
```python
@self._server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list:
    # Resolve workspace: explicit arg > session state > default
    workspace_id = arguments.get("workspace_id")
    if workspace_id is None and hasattr(self, "_session_manager"):
        session_id = get_current_session_id()  # From request context
        if session_id:
            workspace_id = self._workspace_manager.get_workspace(session_id)
    if workspace_id is None:
        workspace_id = "default"

    # Inject into arguments
    arguments["workspace_id"] = workspace_id

    # Execute tool
    result = await self._tools[name].execute(arguments)
    return result.get("content", [])
```

#### `/home/dev/zapomni/src/zapomni_mcp/session_manager.py`

**Impact Level:** MEDIUM - Extend SessionState with workspace

**Modifications Required:**

| Line Range | Class/Method | Change Description |
|------------|--------------|-------------------|
| 68-91 | `SessionState` | Add `workspace_id: str = "default"` attribute |
| 94-130 | `SessionManager` | Add workspace methods |
| 151-202 | `create_session()` | Initialize workspace_id to "default" |

**New Methods to Add:**
```python
def set_workspace(self, session_id: str, workspace_id: str) -> bool:
    """Set workspace for session."""
    session = self._sessions.get(session_id)
    if session:
        session.workspace_id = workspace_id
        session.last_activity = time.monotonic()
        return True
    return False

def get_workspace(self, session_id: str) -> str:
    """Get workspace for session, default if not found."""
    session = self._sessions.get(session_id)
    return session.workspace_id if session else "default"
```

#### NEW FILE: `/home/dev/zapomni/src/zapomni_mcp/workspace_manager.py`

**Impact Level:** NEW - Dedicated workspace context management

**Purpose:** Separate workspace logic from session management for cleaner architecture.

```python
"""
Workspace Manager for MCP Server.

Manages workspace context for sessions and provides workspace CRUD operations.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, List, Any
import re
import structlog

from zapomni_db import FalkorDBClient

logger = structlog.get_logger(__name__)

RESERVED_WORKSPACES = {"default", "system", "admin", "test", "global"}
WORKSPACE_ID_PATTERN = re.compile(r'^[a-z0-9][a-z0-9_-]{0,62}$')


class WorkspaceValidationError(Exception):
    """Raised when workspace validation fails."""
    pass


class WorkspaceManager:
    """
    Manages workspace context and CRUD operations.

    Responsibilities:
    - Workspace context tracking per session
    - Workspace validation
    - Workspace CRUD via FalkorDBClient
    """

    def __init__(self, db_client: FalkorDBClient):
        self.db_client = db_client
        self._session_workspaces: Dict[str, str] = {}
        self._logger = logger.bind(component="WorkspaceManager")

    def validate_workspace_id(self, workspace_id: str) -> None:
        """Validate workspace ID format and reserved names."""
        if workspace_id in RESERVED_WORKSPACES and workspace_id != "default":
            raise WorkspaceValidationError(
                f"Workspace name '{workspace_id}' is reserved"
            )
        if not WORKSPACE_ID_PATTERN.match(workspace_id):
            raise WorkspaceValidationError(
                "Workspace ID must be lowercase alphanumeric, hyphens, underscores, 1-63 chars"
            )

    def set_workspace(self, session_id: str, workspace_id: str) -> None:
        """Set workspace for session."""
        self._session_workspaces[session_id] = workspace_id

    def get_workspace(self, session_id: str) -> str:
        """Get workspace for session, default if not set."""
        return self._session_workspaces.get(session_id, "default")

    def clear_session(self, session_id: str) -> None:
        """Clear workspace state for session."""
        self._session_workspaces.pop(session_id, None)

    async def create_workspace(
        self,
        workspace_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create new workspace."""
        self.validate_workspace_id(workspace_id)
        return await self.db_client.create_workspace(workspace_id, metadata)

    async def list_workspaces(self) -> List[Dict[str, Any]]:
        """List all workspaces."""
        return await self.db_client.list_workspaces()

    async def delete_workspace(
        self,
        workspace_id: str,
        cascade: bool = False
    ) -> Dict[str, Any]:
        """Delete workspace."""
        if workspace_id == "default":
            raise WorkspaceValidationError("Cannot delete default workspace")
        return await self.db_client.delete_workspace(workspace_id, cascade)
```

### 3.4 MCP Tools Layer

All tools need `workspace_id` parameter added to input schema and passed to MemoryProcessor.

#### `/home/dev/zapomni/src/zapomni_mcp/tools/add_memory.py`

**Impact Level:** MEDIUM

**Modifications Required:**

| Line Range | Change Description |
|------------|-------------------|
| 67-110 | Add `workspace_id` to input_schema properties |
| 138-189 | Pass `workspace_id` from arguments to `memory_processor.add_memory()` |

**Schema Addition:**
```python
"workspace_id": {
    "type": "string",
    "description": "Target workspace ID. If not specified, uses session workspace or 'default'.",
    "pattern": "^[a-z0-9][a-z0-9_-]{0,62}$"
},
```

#### `/home/dev/zapomni/src/zapomni_mcp/tools/search_memory.py`

**Impact Level:** MEDIUM

**Modifications Required:**

| Line Range | Change Description |
|------------|-------------------|
| 60-111 | Add `workspace_id` to input_schema properties |
| 138-185 | Pass `workspace_id` from arguments to `memory_processor.search_memory()` |

#### `/home/dev/zapomni/src/zapomni_mcp/tools/get_stats.py`

**Impact Level:** LOW

**Modifications Required:**
- Add optional `workspace_id` parameter for workspace-scoped stats

#### `/home/dev/zapomni/src/zapomni_mcp/tools/delete_memory.py`

**Impact Level:** MEDIUM

**Modifications Required:**
- Add `workspace_id` validation to ensure memory belongs to workspace

#### NEW TOOLS TO ADD:

| Tool | File | Description |
|------|------|-------------|
| `create_workspace` | `tools/create_workspace.py` | Create new workspace |
| `list_workspaces` | `tools/list_workspaces.py` | List all workspaces |
| `set_current_workspace` | `tools/set_workspace.py` | Switch session workspace |
| `delete_workspace` | `tools/delete_workspace.py` | Delete workspace (with cascade option) |
| `get_current_workspace` | `tools/get_workspace.py` | Get current session workspace |

### 3.5 SSE Transport Layer

#### `/home/dev/zapomni/src/zapomni_mcp/sse_transport.py`

**Impact Level:** LOW - Session integration

**Modifications Required:**

| Line Range | Method | Change Description |
|------------|--------|-------------------|
| 186-286 | `handle_sse()` | Initialize workspace context for new session |
| 288-356 | `handle_messages()` | Extract session_id for workspace lookup |

---

## 4. Implementation Recommendations

### 4.1 Phase 1: Foundation (Estimated: 2-3 days)

**Goal:** Core workspace support without breaking changes

1. **Add workspace_id to FalkorDBClient** (Day 1)
   - Add `workspace_id` parameter to `add_memory()`, `vector_search()`
   - Update Cypher queries with workspace filter
   - Default to `"default"` for backward compatibility

2. **Add workspace_id to MemoryProcessor** (Day 1)
   - Add `workspace_id` parameter to public methods
   - Implement `_resolve_workspace()` for context resolution

3. **Create WorkspaceManager** (Day 2)
   - New class for workspace context tracking
   - Validation utilities
   - Session-workspace mapping

4. **Extend SessionState** (Day 2)
   - Add `workspace_id` attribute
   - Add workspace getter/setter methods

5. **Create Migration Script** (Day 2-3)
   - Add `workspace_id = "default"` to existing nodes
   - Verify data integrity

### 4.2 Phase 2: MCP Integration (Estimated: 2 days)

**Goal:** Full MCP tool support for workspaces

1. **Update existing tools** (Day 1)
   - Add `workspace_id` to input schemas
   - Pass workspace context to MemoryProcessor

2. **Create new workspace tools** (Day 2)
   - `create_workspace`
   - `list_workspaces`
   - `set_current_workspace`
   - `delete_workspace`
   - `get_current_workspace`

3. **Update MCPServer** (Day 2)
   - Integrate WorkspaceManager
   - Inject workspace context in tool calls

### 4.3 Phase 3: Testing & Migration (Estimated: 2 days)

1. **Unit tests** - Workspace isolation verification
2. **Integration tests** - Cross-workspace operations
3. **Performance tests** - Query overhead measurement
4. **Production migration** - Run migration script with backups

---

## 5. Integration Points

### 5.1 Workspace Context Resolution

**Priority Order:**
1. Explicit `workspace_id` in tool arguments
2. Session workspace state (SSE mode only)
3. Default workspace `"default"`

**Implementation:**
```python
def _resolve_workspace(
    self,
    explicit_workspace_id: Optional[str],
    session_id: Optional[str]
) -> str:
    """Resolve workspace from explicit arg, session, or default."""
    if explicit_workspace_id is not None:
        return explicit_workspace_id

    if session_id is not None and self._workspace_manager:
        return self._workspace_manager.get_workspace(session_id)

    return "default"
```

### 5.2 Cross-Workspace Operations

**Supported via explicit `workspace_id`:**
```python
# Read from workspace A
results = await processor.search_memory(
    query="Python frameworks",
    workspace_id="project_a"
)

# Write to workspace B
memory_id = await processor.add_memory(
    text="Flask framework notes",
    workspace_id="project_b"
)
```

### 5.3 Cypher Query Integration

**Vector Search with Workspace Filter:**
```cypher
CALL db.idx.vector.queryNodes('Chunk', 'embedding', $limit, vecf32($query_embedding))
YIELD node AS c, score
MATCH (m:Memory)-[:HAS_CHUNK]->(c)
WHERE c.workspace_id = $workspace_id
  AND m.workspace_id = $workspace_id
  AND score <= (1.0 - $min_similarity)
RETURN m.id, c.text, (1.0 - score) AS similarity_score
ORDER BY score ASC
```

---

## 6. Potential Breaking Changes

### 6.1 Database Schema Changes

| Change | Breaking? | Mitigation |
|--------|-----------|------------|
| Add `workspace_id` property to nodes | NO | Default value `"default"` |
| Add workspace filter to queries | NO | Filter includes `workspace_id IS NULL` during migration |

### 6.2 API Changes

| Change | Breaking? | Mitigation |
|--------|-----------|------------|
| Add `workspace_id` parameter to methods | NO | Optional with default `None` |
| New `set_current_workspace()` method | NO | New method, not modifying existing |
| New workspace tools | NO | Additional tools |

### 6.3 Behavioral Changes

| Change | Impact | Mitigation |
|--------|--------|------------|
| Queries return only workspace-scoped data | LOW | Default workspace `"default"` contains all pre-migration data |
| Session state includes workspace | NONE | Additional state, transparent to clients |

**Conclusion:** No breaking changes if migration script is run first.

---

## 7. Session Management Approach

### 7.1 Stdio Transport (No Session)

- No persistent session state
- Workspace resolved per-call from explicit `workspace_id` argument
- Default to `"default"` if not specified

### 7.2 SSE Transport (Session State)

- Session created on SSE connection
- `workspace_id` stored in `SessionState`
- Workspace state cleared on session close
- In-memory storage (lost on restart) - acceptable for MVP

### 7.3 Future Considerations

- Redis-backed session storage for multi-instance deployments
- Workspace persistence across restarts
- Session TTL and cleanup

---

## 8. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Data leakage across workspaces | LOW | HIGH | Mandatory workspace filter in all queries |
| Query performance degradation | LOW | MEDIUM | Index on workspace_id; benchmark after |
| Migration failure | LOW | HIGH | Backup before migration; rollback script |
| Session state loss on restart | MEDIUM | LOW | Default workspace fallback; document behavior |
| Complex cross-workspace logic | MEDIUM | LOW | Explicit workspace_id always works |

---

## 9. Architectural Concerns

### 9.1 Addressed Concerns

1. **Query Safety**: All Cypher queries built via `CypherQueryBuilder` - add workspace filter centrally
2. **Validation**: `WorkspaceManager.validate_workspace_id()` ensures consistent validation
3. **Backward Compatibility**: Default workspace `"default"` preserves existing behavior

### 9.2 Outstanding Questions

1. **Workspace quotas**: Should workspaces have memory limits?
   - Recommendation: Defer to Phase 2

2. **Workspace metadata**: What metadata should workspaces store?
   - Recommendation: `owner`, `created_at`, custom metadata dict

3. **Workspace deletion**: Soft delete vs. hard delete?
   - Recommendation: Hard delete with `cascade` flag; no soft delete for MVP

---

## 10. Summary

### 10.1 Files to Modify

| File | Priority | Estimated LOC Changes |
|------|----------|----------------------|
| `src/zapomni_db/falkordb_client.py` | HIGH | +150 lines |
| `src/zapomni_db/cypher_query_builder.py` | HIGH | +100 lines |
| `src/zapomni_core/memory_processor.py` | HIGH | +80 lines |
| `src/zapomni_mcp/server.py` | MEDIUM | +40 lines |
| `src/zapomni_mcp/session_manager.py` | MEDIUM | +30 lines |
| `src/zapomni_mcp/tools/add_memory.py` | MEDIUM | +20 lines |
| `src/zapomni_mcp/tools/search_memory.py` | MEDIUM | +20 lines |
| `src/zapomni_db/models.py` | LOW | +30 lines |

### 10.2 New Files

| File | Purpose |
|------|---------|
| `src/zapomni_mcp/workspace_manager.py` | Workspace context management |
| `src/zapomni_mcp/tools/create_workspace.py` | Create workspace tool |
| `src/zapomni_mcp/tools/list_workspaces.py` | List workspaces tool |
| `src/zapomni_mcp/tools/set_workspace.py` | Set current workspace tool |
| `src/zapomni_mcp/tools/delete_workspace.py` | Delete workspace tool |
| `src/zapomni_mcp/tools/get_workspace.py` | Get current workspace tool |
| `scripts/migrate_to_workspaces.py` | Migration script |

### 10.3 Estimated Total Effort

- **Phase 1 (Foundation):** 2-3 days
- **Phase 2 (MCP Integration):** 2 days
- **Phase 3 (Testing & Migration):** 2 days
- **Total:** 6-7 days

---

**Prepared by:** Architecture Agent
**For:** Zapomni Development Team
**Next Steps:** Review findings -> Approve design -> Begin Phase 1 implementation
