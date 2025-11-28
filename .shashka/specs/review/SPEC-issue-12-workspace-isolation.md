# SPEC: Issue #12 - Workspace Isolation Fix (BUG-005)

**Status**: Review
**Priority**: P0 CRITICAL
**Created**: 2025-11-28
**Author**: Dev Agent (Claude)

## Summary

Fix critical workspace isolation bug where data from one workspace was accessible when searching from another workspace.

## Problem Statement

From GitHub Issue #12:

```
Steps to Reproduce:
1. set_current_workspace("project-a")
2. add_memory("secret data") without workspace_id
3. set_current_workspace("project-b")
4. search_memory("secret") - **finds data from project-a!**
```

Root cause: `add_memory` and `search_memory` tools were NOT integrated with session state. They ignored `set_current_workspace()` and always used DEFAULT workspace unless workspace_id was explicitly passed.

## Solution

1. Added optional `mcp_server` parameter to `AddMemoryTool` and `SearchMemoryTool`
2. Tools now call `mcp_server.resolve_workspace_id()` when workspace_id not provided
3. Server passes `self` reference when creating tools in `register_all_tools()`
4. Maintains backwards compatibility - tools work without `mcp_server`

## Implementation Details

### Files Modified

| File | Lines Changed | Description |
|------|---------------|-------------|
| `add_memory.py` | +20/-5 | Added mcp_server param, workspace resolution |
| `search_memory.py` | +18/-4 | Added mcp_server param, workspace resolution |
| `server.py` | +3/-3 | Pass mcp_server=self to tools |

### Key Code Changes

**add_memory.py:121-153** - Updated `__init__`:
```python
def __init__(
    self,
    memory_processor: MemoryProcessor,
    mcp_server: Any = None,
) -> None:
    ...
    self.mcp_server = mcp_server
```

**add_memory.py:188-193** - Workspace resolution:
```python
workspace_id = arguments.get("workspace_id")
if workspace_id is None and self.mcp_server is not None:
    workspace_id = self.mcp_server.resolve_workspace_id()
```

**server.py:259-261** - Tool registration:
```python
AddMemoryTool(memory_processor=memory_processor, mcp_server=self),
SearchMemoryTool(memory_processor=memory_processor, mcp_server=self),
```

## Testing

### TDD Tests Added (10 tests)

File: `tests/unit/test_workspace_isolation_bug005.py`

1. `test_add_memory_accepts_mcp_server_parameter`
2. `test_add_memory_uses_resolve_workspace_id_when_not_provided`
3. `test_add_memory_uses_explicit_workspace_id_when_provided`
4. `test_add_memory_falls_back_to_default_without_mcp_server`
5. `test_search_memory_accepts_mcp_server_parameter`
6. `test_search_memory_uses_resolve_workspace_id_when_not_provided`
7. `test_search_memory_uses_explicit_workspace_id_when_provided`
8. `test_search_memory_falls_back_to_default_without_mcp_server`
9. `test_workspace_isolation_scenario` (end-to-end)
10. `test_register_all_tools_passes_mcp_server`

### Test Results

```
tests/unit/test_workspace_isolation_bug005.py: 10 passed
tests/unit/test_add_memory_tool.py: 27 passed
tests/unit/test_search_memory_tool.py: 28 passed
Full unit test suite: 2109 passed, 11 skipped
```

## Acceptance Criteria

- [x] `add_memory` without workspace_id uses session workspace
- [x] `search_memory` without workspace_id uses session workspace
- [x] Explicit workspace_id overrides session workspace
- [x] Backwards compatible without mcp_server
- [x] All existing tests pass
- [x] 10 new TDD tests pass

---

## Dev Agent Report

**Timestamp:** 2025-11-28 15:45 UTC
**Status:** Completed
**PR:** #31

### Files Changed
- `src/zapomni_mcp/tools/add_memory.py`
- `src/zapomni_mcp/tools/search_memory.py`
- `src/zapomni_mcp/server.py`

### Files Created
- `tests/unit/test_workspace_isolation_bug005.py`

### Tests
- 10/10 new tests passed
- 2109/2109 unit tests passed (11 skipped)

### Statistics
- Lines added: ~320
- Lines removed: ~12
- Files changed: 4

### Notes
- Used TDD approach: wrote tests first, then implemented fix
- Maintained backwards compatibility
- No breaking changes to existing API
