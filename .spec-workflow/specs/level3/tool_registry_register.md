# ToolRegistry.register() - Function Specification

**Level:** 3 (Function)
**Component:** ToolRegistry
**Module:** zapomni_mcp
**Author:** Goncharenko Anton aka alienxs2
**Status:** Draft
**Version:** 1.0
**Created:** 2025-11-23

## Function Signature

```python
def register(self, tool: MCPTool) -> None:
    """Register a single MCP tool."""
```

## Purpose

Validate tool protocol compliance and name uniqueness, then add to internal registry.

## Parameters

### tool: MCPTool
- **Type:** Object implementing MCPTool protocol
- **Required attributes:**
  - `name: str` (non-empty, unique)
  - `description: str` (non-empty)
  - `input_schema: dict[str, Any]` (valid JSON schema)
- **Required methods:**
  - `async def execute(arguments: dict) -> dict`

## Returns

None (void method)

## Raises

- `TypeError`: Tool doesn't implement MCPTool protocol
- `ValueError`: Tool name empty or duplicate
- `AttributeError`: Tool missing required attributes

## Algorithm

```
1. Validate tool type (isinstance check)
2. Check required attributes exist:
   - tool.name
   - tool.description
   - tool.input_schema
3. Validate name:
   - Non-empty string
   - Not already registered
4. Validate execute method exists
5. Add to registry: self._tools[tool.name] = tool
6. Log registration
```

## Edge Cases

1. **Duplicate name** → ValueError
2. **Empty name** → ValueError
3. **Missing attribute** → AttributeError
4. **Invalid protocol** → TypeError
5. **None as tool** → TypeError

## Test Scenarios (10)

1. test_register_success
2. test_register_duplicate_raises
3. test_register_empty_name_raises
4. test_register_missing_name_raises
5. test_register_missing_description_raises
6. test_register_missing_schema_raises
7. test_register_missing_execute_raises
8. test_register_invalid_protocol_raises
9. test_register_none_raises
10. test_register_stores_in_registry

## Performance

- Target: < 1ms (dict insertion)

**Status:** Draft v1.0 | **Author:** Goncharenko Anton aka alienxs2 | **License:** MIT
