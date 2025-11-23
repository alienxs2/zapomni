# ToolRegistry.get_tool() - Function Specification

**Level:** 3 | **Component:** ToolRegistry | **Module:** zapomni_mcp
**Author:** Goncharenko Anton aka alienxs2 | **Status:** Draft | **V:** 1.0

## Signature
```python
def get_tool(self, name: str) -> MCPTool:
    """
    Retrieve registered tool by name.
    
    Args:
        name: Tool name (e.g., "add_memory", "search_memory")
        
    Returns:
        MCPTool instance
        
    Raises:
        KeyError: If tool not registered
        ValueError: If name empty or invalid
    """
```

## Purpose

Lookup tool by name for request routing in MCP server.

## Edge Cases
1. name = "" → ValueError("Tool name cannot be empty")
2. name = "unknown" → KeyError("Tool not found: unknown")
3. name = "add_memory" → Returns AddMemoryTool instance
4. name with spaces → ValueError
5. Case-sensitive lookup → "Add_Memory" != "add_memory"

## Algorithm
```
1. Validate name non-empty
2. Validate name alphanumeric + underscore only
3. Lookup in tool_dict
4. If found, return tool
5. If not found, raise KeyError
```

## Tests (10)
1. test_get_tool_success
2. test_get_tool_not_found_raises
3. test_get_tool_empty_name_raises
4. test_get_tool_invalid_chars_raises
5. test_get_tool_case_sensitive
6. test_get_tool_returns_correct_instance
7. test_get_tool_multiple_lookups
8. test_get_tool_all_registered_tools
9. test_get_tool_whitespace_raises
10. test_get_tool_special_chars_raises
