# Workspace/Project Isolation - Requirements Specification

**Spec ID:** workspaces
**Version:** 1.0
**Date:** 2025-11-25
**Status:** Draft

---

## 1. Overview

### 1.1 Purpose

Implement workspace isolation for the Zapomni memory server to enable multi-tenant operation where different projects/contexts can maintain separate knowledge graphs without data interference.

### 1.2 Scope

This specification covers:
- Workspace CRUD operations
- Data isolation at the graph database level
- Session-based workspace state management
- Migration of existing data to default workspace

### 1.3 Definitions

| Term | Definition |
|------|------------|
| Workspace | A logical partition of the knowledge graph that isolates memories, chunks, and entities |
| Session | An MCP connection instance (SSE mode) with associated state |
| Default Workspace | The "default" workspace used when no workspace is specified |

---

## 2. User Stories

### 2.1 Workspace Management

**US-01: Create Workspace**
> As a user, I want to create a new workspace with a name and description so that I can organize my memories by project.

**Acceptance Criteria:**
- [ ] Can call `create_workspace(name, description)` to create a new workspace
- [ ] Workspace name is validated (lowercase alphanumeric, hyphens, underscores, 1-63 chars)
- [ ] Reserved names ("system", "admin", "test", "global") are rejected
- [ ] Returns workspace ID and creation timestamp on success
- [ ] Returns error if workspace with same name already exists

**US-02: List Workspaces**
> As a user, I want to list all available workspaces so that I can see what projects exist.

**Acceptance Criteria:**
- [ ] Can call `list_workspaces()` to get all workspaces
- [ ] Each workspace entry includes: id, name, description, created_at, memory_count
- [ ] Workspaces are sorted by creation date (newest first)
- [ ] "default" workspace is always included in the list

**US-03: Delete Workspace**
> As a user, I want to delete a workspace and all its data so that I can clean up unused projects.

**Acceptance Criteria:**
- [ ] Can call `delete_workspace(id)` to delete a workspace
- [ ] Deletes all memories, chunks, and entities in the workspace
- [ ] Cannot delete the "default" workspace
- [ ] Returns count of deleted items
- [ ] No data from other workspaces is affected

**US-04: Switch Workspace**
> As a user, I want to switch my current workspace so that subsequent operations target that workspace.

**Acceptance Criteria:**
- [ ] Can call `set_current_workspace(workspace_id)` to switch workspace
- [ ] Subsequent `add_memory()` and `search_memory()` calls use the set workspace
- [ ] Switching to non-existent workspace returns an error
- [ ] Current workspace is reported in `get_current_workspace()` response

**US-05: Get Current Workspace**
> As a user, I want to know which workspace I'm currently in so that I understand where my data will be stored.

**Acceptance Criteria:**
- [ ] Can call `get_current_workspace()` to see current workspace
- [ ] Returns "default" if no workspace has been set
- [ ] Returns the workspace set by `set_current_workspace()`

### 2.2 Data Isolation

**US-06: Isolated Memory Storage**
> As a user, I want memories added to a workspace to be stored only in that workspace so that projects don't mix.

**Acceptance Criteria:**
- [ ] Memory added to "Project A" is stored with `workspace_id = "project_a"`
- [ ] Memory cannot be accessed from other workspaces
- [ ] All chunks and entities from the memory inherit the same workspace_id

**US-07: Isolated Memory Search**
> As a user, I want search results to only return memories from my current workspace so that I get relevant results.

**Acceptance Criteria:**
- [ ] Search in "Project A" only returns memories from "Project A"
- [ ] Search in "Project B" does not return memories from "Project A"
- [ ] Vector similarity search respects workspace boundaries
- [ ] Graph traversal (entities, relationships) respects workspace boundaries

**US-08: Explicit Workspace Override**
> As a user, I want to specify an explicit workspace_id in any operation so that I can perform cross-context operations.

**Acceptance Criteria:**
- [ ] Every tool accepts optional `workspace_id` parameter
- [ ] Explicit `workspace_id` takes priority over session state
- [ ] Can read from workspace A while session is set to workspace B
- [ ] Can write to workspace B while session is set to workspace A

### 2.3 Default Behavior

**US-09: Default Workspace Fallback**
> As a user, I want operations without a workspace_id to use a default workspace so that the system works without explicit configuration.

**Acceptance Criteria:**
- [ ] If no `workspace_id` argument and no session workspace set, use "default"
- [ ] "default" workspace is created automatically on first use
- [ ] Existing data (pre-migration) lives in "default" workspace
- [ ] New users start with "default" workspace

---

## 3. Functional Requirements

### 3.1 Workspace CRUD Operations

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-01 | System SHALL provide `create_workspace(name, description)` tool | MUST |
| FR-02 | System SHALL validate workspace names against pattern `^[a-z0-9][a-z0-9_-]{0,62}$` | MUST |
| FR-03 | System SHALL reject reserved workspace names: "system", "admin", "test", "global" | MUST |
| FR-04 | System SHALL allow "default" as a valid workspace name | MUST |
| FR-05 | System SHALL provide `list_workspaces()` tool returning all workspaces with stats | MUST |
| FR-06 | System SHALL provide `delete_workspace(id)` tool with cascade delete | MUST |
| FR-07 | System SHALL prevent deletion of "default" workspace | MUST |
| FR-08 | System SHALL provide `set_current_workspace(workspace_id)` tool | MUST |
| FR-09 | System SHALL provide `get_current_workspace()` tool | MUST |

### 3.2 Data Isolation

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-10 | All Memory nodes SHALL include `workspace_id` property | MUST |
| FR-11 | All Chunk nodes SHALL include `workspace_id` property | MUST |
| FR-12 | All Entity nodes SHALL include `workspace_id` property | MUST |
| FR-13 | All search queries SHALL filter by `workspace_id` | MUST |
| FR-14 | Vector search SHALL return only results from specified workspace | MUST |
| FR-15 | Graph traversal SHALL not cross workspace boundaries | MUST |
| FR-16 | Delete operations SHALL validate workspace ownership | MUST |

### 3.3 Session State Management

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-17 | SSE sessions SHALL maintain current workspace state | MUST |
| FR-18 | Session workspace state SHALL default to "default" | MUST |
| FR-19 | Session workspace state SHALL persist for duration of connection | MUST |
| FR-20 | Session workspace state SHALL clear on disconnect | MUST |
| FR-21 | Stdio mode SHALL use explicit `workspace_id` or "default" | MUST |

### 3.4 Workspace Resolution Priority

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-22 | Explicit `workspace_id` argument SHALL take priority | MUST |
| FR-23 | Session workspace state SHALL be second priority | MUST |
| FR-24 | "default" workspace SHALL be fallback | MUST |

### 3.5 Migration Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-25 | Migration script SHALL add `workspace_id = "default"` to all existing nodes | MUST |
| FR-26 | Migration SHALL be idempotent (safe to run multiple times) | MUST |
| FR-27 | Migration SHALL support batch processing for large datasets | SHOULD |
| FR-28 | Migration SHALL provide progress reporting | SHOULD |
| FR-29 | Migration SHALL include rollback capability | SHOULD |

---

## 4. Non-Functional Requirements

### 4.1 Performance

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-01 | Workspace filtering overhead on queries | < 15% latency increase |
| NFR-02 | Workspace CRUD operation latency | < 100ms |
| NFR-03 | Session workspace lookup | < 1ms |

### 4.2 Scalability

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-04 | Number of concurrent workspaces | Unlimited |
| NFR-05 | Memories per workspace | Unlimited |
| NFR-06 | Concurrent sessions with different workspaces | 100+ |

### 4.3 Reliability

| ID | Requirement | Description |
|----|-------------|-------------|
| NFR-07 | Data isolation guarantee | Zero data leakage between workspaces |
| NFR-08 | Graceful degradation | Missing workspace_id defaults to "default" |
| NFR-09 | Session state loss | Acceptable (defaults to "default" on reconnect) |

### 4.4 Security

| ID | Requirement | Description |
|----|-------------|-------------|
| NFR-10 | Workspace access | No authentication (single-user mode for MVP) |
| NFR-11 | Workspace name validation | Prevent injection via strict pattern matching |
| NFR-12 | Reserved names | Prevent use of system-reserved names |

---

## 5. Constraints

### 5.1 Technical Constraints

| ID | Constraint | Rationale |
|----|------------|-----------|
| C-01 | Must use property-based filtering (not separate graphs) | Simpler implementation, lower memory overhead |
| C-02 | Session state stored in-memory only | Redis integration deferred to future phase |
| C-03 | No workspace authentication/authorization | Single-user mode for MVP |
| C-04 | Workspace ID is immutable after creation | Simplifies data integrity |

### 5.2 Backward Compatibility

| ID | Constraint | Rationale |
|----|------------|-----------|
| C-05 | All existing APIs must continue to work without modification | Zero breaking changes |
| C-06 | Existing data must remain accessible | Migration to "default" workspace |
| C-07 | `workspace_id` parameter must be optional | Backward compatibility |

---

## 6. Acceptance Criteria (Integration Test)

The following end-to-end scenario MUST pass:

```python
# 1. Create two workspaces
await create_workspace("project_a", "Project A workspace")
await create_workspace("project_b", "Project B workspace")

# 2. Add memory to Project A
await set_current_workspace("project_a")
memory_a = await add_memory("This is Project A content about Python")

# 3. Add memory to Project B
await set_current_workspace("project_b")
memory_b = await add_memory("This is Project B content about Python")

# 4. Search in Project A - should only find A's content
await set_current_workspace("project_a")
results_a = await search_memory("Python")
assert len(results_a) == 1
assert "Project A" in results_a[0].text

# 5. Search in Project B - should only find B's content
await set_current_workspace("project_b")
results_b = await search_memory("Python")
assert len(results_b) == 1
assert "Project B" in results_b[0].text

# 6. Cross-workspace read (explicit workspace_id)
results_cross = await search_memory("Python", workspace_id="project_a")
assert "Project A" in results_cross[0].text

# 7. Delete Project A - only A's data affected
await delete_workspace("project_a")

# 8. Verify Project B still has its data
results_b_after = await search_memory("Python", workspace_id="project_b")
assert len(results_b_after) == 1

# 9. Default workspace behavior
await set_current_workspace("default")
memory_default = await add_memory("Default workspace content")
results_default = await search_memory("Default")
assert len(results_default) == 1
```

---

## 7. Out of Scope

The following are explicitly out of scope for this specification:

1. **Workspace Authentication/Authorization** - No per-workspace access control
2. **Workspace Quotas** - No memory limits per workspace
3. **Workspace Sharing** - No multi-user workspace sharing
4. **Workspace Templates** - No pre-configured workspace templates
5. **Redis-backed Session Storage** - In-memory only for MVP
6. **Workspace Import/Export** - No bulk data transfer between workspaces
7. **Workspace Analytics** - No usage dashboards or metrics
8. **Separate Graph Instances** - Property-based filtering only

---

## 8. Dependencies

| Dependency | Type | Description |
|------------|------|-------------|
| FalkorDB | External | Graph database with vector search |
| MCP Protocol | External | Model Context Protocol for tool interface |
| SessionManager | Internal | Existing SSE session management |

---

## 9. Glossary

| Term | Definition |
|------|------------|
| Cascade Delete | Deleting a workspace and all its contained data |
| Property-based Filtering | Using node properties (workspace_id) to isolate data in a shared graph |
| Session State | In-memory state associated with an SSE connection |
| Workspace Context | The current workspace for operations in a session |

---

**Prepared by:** Specification Agent
**Reviewed by:** (pending)
**Approved by:** (pending)
