# ToolRegistry.__init__() - Function Specification

**Level:** 3 | **Component:** ToolRegistry | **Module:** zapomni_mcp
**Author:** Goncharenko Anton aka alienxs2 | **Status:** Draft | **V:** 1.0

## Signature
```python
def __init__(self, tools: Optional[List[MCPTool]] = None) -> None:
    """Initialize tool registry with optional pre-registered tools."""
```

## Parameters
- **tools**: Optional list of MCPTool instances to register (default: None = empty registry)

## Edge Cases
1. tools = None → Valid (empty registry)
2. tools = [] → Valid (empty registry)
3. tools = [tool1, tool2] → Valid (both registered)
4. tools contains duplicate names → ValueError
5. tools contains non-MCPTool → TypeError
6. tools contains tool with invalid schema → ValueError

## Tests (10)
1. test_init_no_tools, 2. test_init_empty_list, 3. test_init_with_tools,
4. test_init_duplicate_names_raises, 5. test_init_wrong_type_raises,
6. test_init_invalid_schema_raises, 7. test_init_creates_tool_dict,
8. test_init_stores_tools, 9. test_init_creates_logger,
10. test_init_tool_count_correct
