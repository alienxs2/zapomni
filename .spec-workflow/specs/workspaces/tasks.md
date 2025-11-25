# Workspace/Project Isolation - Implementation Tasks

**Spec ID:** workspaces
**Version:** 1.0
**Date:** 2025-11-25
**Status:** Draft

---

## Phase 1: Database Layer (Foundation)

### 1.1 Data Models

- [ ] **1.1.1** Add `workspace_id: str = "default"` field to `Memory` model in `src/zapomni_db/models.py`
- [ ] **1.1.2** Add `workspace_id: str = "default"` field to `Chunk` model in `src/zapomni_db/models.py`
- [ ] **1.1.3** Add `workspace_id: str = "default"` field to `Entity` model in `src/zapomni_db/models.py`
- [ ] **1.1.4** Create `Workspace` model with id, name, description, metadata, created_at fields
- [ ] **1.1.5** Create `WorkspaceInfo` dataclass for workspace metadata responses

### 1.2 Cypher Query Builder

- [ ] **1.2.1** Update `build_add_memory_query()` to include `workspace_id` in Memory and Chunk node creation
- [ ] **1.2.2** Update `build_vector_search_query()` to add `WHERE c.workspace_id = $workspace_id` filter
- [ ] **1.2.3** Update `build_graph_traversal_query()` to filter entities by workspace_id
- [ ] **1.2.4** Update `build_delete_memory_query()` to validate workspace ownership
- [ ] **1.2.5** Update `build_add_entity_query()` to include workspace_id in Entity creation
- [ ] **1.2.6** Update `build_add_relationship_query()` to check workspace consistency
- [ ] **1.2.7** Add `build_create_workspace_query()` for Workspace node creation
- [ ] **1.2.8** Add `build_list_workspaces_query()` with stats aggregation
- [ ] **1.2.9** Add `build_delete_workspace_query()` with cascade delete logic
- [ ] **1.2.10** Add `build_workspace_stats_query()` for workspace-scoped statistics

### 1.3 FalkorDB Client

- [ ] **1.3.1** Add `workspace_id: str = "default"` parameter to `add_memory()` method
- [ ] **1.3.2** Add `workspace_id: str = "default"` parameter to `vector_search()` method
- [ ] **1.3.3** Add `workspace_id: str = "default"` parameter to `add_entity()` method
- [ ] **1.3.4** Add `workspace_id: str = "default"` parameter to `get_related_entities()` method
- [ ] **1.3.5** Add `workspace_id: Optional[str] = None` parameter to `get_stats()` for scoped stats
- [ ] **1.3.6** Add `workspace_id` validation to `delete_memory()` to prevent cross-workspace delete
- [ ] **1.3.7** Implement `create_workspace()` method
- [ ] **1.3.8** Implement `list_workspaces()` method
- [ ] **1.3.9** Implement `delete_workspace()` method with cascade option
- [ ] **1.3.10** Implement `workspace_exists()` method
- [ ] **1.3.11** Implement `ensure_default_workspace()` method for initialization

---

## Phase 2: Core Processing Layer

### 2.1 Memory Processor

- [ ] **2.1.1** Add `workspace_id: Optional[str] = None` parameter to `add_memory()` method
- [ ] **2.1.2** Add `workspace_id: Optional[str] = None` parameter to `search_memory()` method
- [ ] **2.1.3** Add `workspace_id: Optional[str] = None` parameter to `get_stats()` method
- [ ] **2.1.4** Add `workspace_id: Optional[str] = None` parameter to `build_knowledge_graph()` method
- [ ] **2.1.5** Add `workspace_id: Optional[str] = None` parameter to `get_related_entities()` method
- [ ] **2.1.6** Implement `_resolve_workspace()` helper for context resolution
- [ ] **2.1.7** Update `_store_memory()` to pass workspace_id to db_client
- [ ] **2.1.8** Update `_store_entities()` to pass workspace_id to db_client

---

## Phase 3: Workspace Manager

### 3.1 Create WorkspaceManager Class

- [ ] **3.1.1** Create `src/zapomni_mcp/workspace_manager.py` file
- [ ] **3.1.2** Implement `RESERVED_WORKSPACES` constant and `WORKSPACE_ID_PATTERN` regex
- [ ] **3.1.3** Implement `WorkspaceValidationError` and `WorkspaceNotFoundError` exceptions
- [ ] **3.1.4** Implement `WorkspaceInfo` dataclass
- [ ] **3.1.5** Implement `WorkspaceManager.__init__()` with db_client and session dict
- [ ] **3.1.6** Implement `validate_workspace_id()` method
- [ ] **3.1.7** Implement `set_workspace()` for session context
- [ ] **3.1.8** Implement `get_workspace()` for session context
- [ ] **3.1.9** Implement `clear_session()` for session cleanup
- [ ] **3.1.10** Implement `resolve_workspace()` for priority resolution (explicit > session > default)
- [ ] **3.1.11** Implement `create_workspace()` async method
- [ ] **3.1.12** Implement `list_workspaces()` async method
- [ ] **3.1.13** Implement `get_workspace_by_id()` async method
- [ ] **3.1.14** Implement `delete_workspace()` async method
- [ ] **3.1.15** Implement `workspace_exists()` async method
- [ ] **3.1.16** Implement `ensure_default_workspace()` async method

---

## Phase 4: Session Manager Integration

### 4.1 Extend SessionState

- [ ] **4.1.1** Add `workspace_id: str = "default"` attribute to `SessionState` dataclass
- [ ] **4.1.2** Add `set_workspace()` method to `SessionManager`
- [ ] **4.1.3** Add `get_workspace()` method to `SessionManager`
- [ ] **4.1.4** Update session creation to initialize workspace_id to "default"

---

## Phase 5: MCP Server Integration

### 5.1 Server Updates

- [ ] **5.1.1** Import `WorkspaceManager` in `src/zapomni_mcp/server.py`
- [ ] **5.1.2** Initialize `WorkspaceManager` in `MCPServer.__init__()`
- [ ] **5.1.3** Update `handle_call_tool()` to resolve workspace from arguments/session
- [ ] **5.1.4** Inject resolved `workspace_id` into tool arguments before execution
- [ ] **5.1.5** Pass `session_id` to tools that need session context
- [ ] **5.1.6** Call `ensure_default_workspace()` on server startup

---

## Phase 6: MCP Tools - New Workspace Tools

### 6.1 Create Workspace Tool

- [ ] **6.1.1** Create `src/zapomni_mcp/tools/create_workspace.py`
- [ ] **6.1.2** Implement `CreateWorkspaceTool` class with input schema
- [ ] **6.1.3** Implement `execute()` method calling WorkspaceManager
- [ ] **6.1.4** Register tool in `__init__.py`

### 6.2 List Workspaces Tool

- [ ] **6.2.1** Create `src/zapomni_mcp/tools/list_workspaces.py`
- [ ] **6.2.2** Implement `ListWorkspacesTool` class with input schema
- [ ] **6.2.3** Implement `execute()` method returning workspace list with stats
- [ ] **6.2.4** Register tool in `__init__.py`

### 6.3 Set Current Workspace Tool

- [ ] **6.3.1** Create `src/zapomni_mcp/tools/set_workspace.py`
- [ ] **6.3.2** Implement `SetCurrentWorkspaceTool` class with input schema
- [ ] **6.3.3** Implement `execute()` method updating session workspace
- [ ] **6.3.4** Add workspace existence validation
- [ ] **6.3.5** Register tool in `__init__.py`

### 6.4 Get Current Workspace Tool

- [ ] **6.4.1** Create `src/zapomni_mcp/tools/get_workspace.py`
- [ ] **6.4.2** Implement `GetCurrentWorkspaceTool` class with input schema
- [ ] **6.4.3** Implement `execute()` method returning current session workspace
- [ ] **6.4.4** Register tool in `__init__.py`

### 6.5 Delete Workspace Tool

- [ ] **6.5.1** Create `src/zapomni_mcp/tools/delete_workspace.py`
- [ ] **6.5.2** Implement `DeleteWorkspaceTool` class with input schema
- [ ] **6.5.3** Implement `execute()` method with cascade delete
- [ ] **6.5.4** Add validation to prevent deleting "default" workspace
- [ ] **6.5.5** Register tool in `__init__.py`

---

## Phase 7: MCP Tools - Update Existing Tools

### 7.1 Add Memory Tool

- [ ] **7.1.1** Add `workspace_id` to `AddMemoryTool.input_schema`
- [ ] **7.1.2** Extract `workspace_id` from arguments in `execute()`
- [ ] **7.1.3** Pass `workspace_id` to `memory_processor.add_memory()`

### 7.2 Search Memory Tool

- [ ] **7.2.1** Add `workspace_id` to `SearchMemoryTool.input_schema`
- [ ] **7.2.2** Extract `workspace_id` from arguments in `execute()`
- [ ] **7.2.3** Pass `workspace_id` to `memory_processor.search_memory()`

### 7.3 Delete Memory Tool

- [ ] **7.3.1** Add `workspace_id` to `DeleteMemoryTool.input_schema`
- [ ] **7.3.2** Extract `workspace_id` from arguments in `execute()`
- [ ] **7.3.3** Pass `workspace_id` to validation and delete operation

### 7.4 Get Stats Tool

- [ ] **7.4.1** Add optional `workspace_id` to `GetStatsTool.input_schema`
- [ ] **7.4.2** Pass `workspace_id` to `memory_processor.get_stats()` if provided

### 7.5 Build Knowledge Graph Tool

- [ ] **7.5.1** Add `workspace_id` to tool input schema
- [ ] **7.5.2** Pass `workspace_id` to `memory_processor.build_knowledge_graph()`

---

## Phase 8: Migration Scripts

### 8.1 Migration Script

- [ ] **8.1.1** Create `scripts/migrate_to_workspaces.py`
- [ ] **8.1.2** Implement CLI argument parsing (--batch-size, --dry-run, --host, --port)
- [ ] **8.1.3** Implement `ensure_default_workspace()` function
- [ ] **8.1.4** Implement `migrate_nodes()` function for batch migration
- [ ] **8.1.5** Add progress reporting with remaining count
- [ ] **8.1.6** Add support for Memory, Chunk, and Entity node types
- [ ] **8.1.7** Implement idempotent migration (skip nodes with existing workspace_id)

### 8.2 Rollback Script

- [ ] **8.2.1** Create `scripts/rollback_workspaces.py`
- [ ] **8.2.2** Implement workspace_id property removal from all nodes
- [ ] **8.2.3** Implement Workspace node deletion
- [ ] **8.2.4** Add confirmation prompt for destructive operation

---

## Phase 9: Testing

### 9.1 Unit Tests - WorkspaceManager

- [ ] **9.1.1** Create `tests/unit/test_workspace_manager.py`
- [ ] **9.1.2** Test `validate_workspace_id()` with valid IDs
- [ ] **9.1.3** Test `validate_workspace_id()` with invalid IDs (uppercase, special chars)
- [ ] **9.1.4** Test `validate_workspace_id()` with reserved names
- [ ] **9.1.5** Test `set_workspace()` and `get_workspace()` for session tracking
- [ ] **9.1.6** Test `resolve_workspace()` priority (explicit > session > default)
- [ ] **9.1.7** Test `clear_session()` removes workspace state

### 9.2 Unit Tests - Cypher Queries

- [ ] **9.2.1** Test `build_add_memory_query()` includes workspace_id
- [ ] **9.2.2** Test `build_vector_search_query()` includes workspace filter
- [ ] **9.2.3** Test `build_create_workspace_query()` generates correct Cypher
- [ ] **9.2.4** Test `build_delete_workspace_query()` with cascade option

### 9.3 Integration Tests - Workspace Isolation

- [ ] **9.3.1** Create `tests/integration/test_workspace_isolation.py`
- [ ] **9.3.2** Test memory added to workspace A is NOT found in workspace B search
- [ ] **9.3.3** Test memory added to workspace B is NOT found in workspace A search
- [ ] **9.3.4** Test explicit `workspace_id` parameter overrides session workspace
- [ ] **9.3.5** Test `delete_workspace()` deletes only that workspace's data
- [ ] **9.3.6** Test default workspace behavior when no workspace specified

### 9.4 Integration Tests - Workspace CRUD

- [ ] **9.4.1** Create `tests/integration/test_workspace_crud.py`
- [ ] **9.4.2** Test `create_workspace()` creates Workspace node
- [ ] **9.4.3** Test `create_workspace()` rejects duplicate names
- [ ] **9.4.4** Test `create_workspace()` rejects reserved names
- [ ] **9.4.5** Test `list_workspaces()` returns all workspaces with stats
- [ ] **9.4.6** Test `delete_workspace()` removes workspace and data
- [ ] **9.4.7** Test `delete_workspace()` rejects "default" workspace deletion

### 9.5 Integration Tests - MCP Tools

- [ ] **9.5.1** Create `tests/integration/test_workspace_mcp_tools.py`
- [ ] **9.5.2** Test `create_workspace` tool via MCP server
- [ ] **9.5.3** Test `list_workspaces` tool via MCP server
- [ ] **9.5.4** Test `set_current_workspace` tool updates session state
- [ ] **9.5.5** Test `get_current_workspace` tool returns correct workspace
- [ ] **9.5.6** Test `delete_workspace` tool removes workspace
- [ ] **9.5.7** Test `add_memory` tool respects session workspace
- [ ] **9.5.8** Test `search_memory` tool respects session workspace

### 9.6 Migration Tests

- [ ] **9.6.1** Create `tests/integration/test_migration.py`
- [ ] **9.6.2** Test migration script adds workspace_id to existing nodes
- [ ] **9.6.3** Test migration is idempotent (safe to run twice)
- [ ] **9.6.4** Test rollback script removes workspace_id

---

## Phase 10: Documentation

### 10.1 Update Existing Documentation

- [ ] **10.1.1** Update MCP tools documentation with workspace parameters
- [ ] **10.1.2** Add workspace section to README

### 10.2 Migration Guide

- [ ] **10.2.1** Create migration instructions for existing deployments

---

## Summary

| Phase | Tasks | Estimated Effort |
|-------|-------|------------------|
| Phase 1: Database Layer | 21 tasks | 1-2 days |
| Phase 2: Core Processing | 8 tasks | 0.5 days |
| Phase 3: WorkspaceManager | 16 tasks | 1 day |
| Phase 4: Session Manager | 4 tasks | 0.5 days |
| Phase 5: MCP Server | 6 tasks | 0.5 days |
| Phase 6: New MCP Tools | 17 tasks | 1 day |
| Phase 7: Update Existing Tools | 11 tasks | 0.5 days |
| Phase 8: Migration Scripts | 9 tasks | 0.5 days |
| Phase 9: Testing | 25 tasks | 1-2 days |
| Phase 10: Documentation | 3 tasks | 0.5 days |
| **Total** | **120 tasks** | **7-9 days** |

---

## Task Dependencies

```
Phase 1 (Database)
    └── Phase 2 (Core Processing)
          └── Phase 3 (WorkspaceManager)
                ├── Phase 4 (Session Manager)
                └── Phase 5 (MCP Server)
                      ├── Phase 6 (New Tools)
                      └── Phase 7 (Update Tools)
                            └── Phase 8 (Migration)
                                  └── Phase 9 (Testing)
                                        └── Phase 10 (Docs)
```

---

**Prepared by:** Specification Agent
**Reviewed by:** (pending)
**Approved by:** (pending)
