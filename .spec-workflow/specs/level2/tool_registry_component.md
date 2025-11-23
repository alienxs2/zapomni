# ToolRegistry - Component Specification

**Level:** 2 (Component)
**Module:** zapomni_mcp
**Author:** Goncharenko Anton aka alienxs2
**Status:** Draft
**Version:** 1.0
**Created:** 2025-11-23

---

## Overview

### Purpose

The **ToolRegistry** component manages registration, storage, and retrieval of MCP tools within the Zapomni MCP server. It acts as a centralized registry that ensures tool name uniqueness, validates tool protocol compliance, and provides efficient tool lookup during MCP request handling.

**Key Responsibilities:**
- Register individual MCP tools
- Batch register multiple tools
- Retrieve tools by name
- List all registered tools
- Validate tool protocol compliance
- Prevent duplicate tool names

**Design Philosophy:** Simple in-memory registry with strict validation. Zero business logic - purely organizational.

### Responsibilities

1. **Tool Storage:**
   - Maintain in-memory dictionary of tools
   - Map tool names to tool instances
   - Support fast O(1) lookup by name

2. **Tool Validation:**
   - Verify tool implements MCPTool protocol
   - Check required attributes exist (name, description, input_schema)
   - Validate execute() method signature
   - Ensure tool name uniqueness

3. **Tool Registration:**
   - Register single tool with validation
   - Batch register multiple tools
   - Reject duplicate names
   - Provide clear error messages

4. **Tool Discovery:**
   - List all registered tool names
   - Retrieve tool by name
   - Return None for unknown tools (safe lookup)

### Position in Module

**Component Context within zapomni_mcp:**
```
┌─────────────────────────────────────┐
│         MCPServer                    │
│  (main entry point)                  │
└──────────────┬──────────────────────┘
               │ uses
               ↓
┌─────────────────────────────────────┐
│       ToolRegistry                   │ ← THIS COMPONENT
│  (tool management)                   │
└──────────────┬──────────────────────┘
               │ stores
               ↓
┌─────────────────────────────────────┐
│  MCPTool Instances                   │
│  - AddMemoryTool                     │
│  - SearchMemoryTool                  │
│  - GetStatsTool                      │
└─────────────────────────────────────┘
```

**Interaction Flow:**
1. MCPServer creates ToolRegistry instance
2. MCPServer registers all tools via `register_all()`
3. During request handling, MCPServer calls `get_tool(name)`
4. ToolRegistry returns tool instance or None
5. MCPServer executes tool if found

---

## Class Definition

### Class Diagram

```
┌───────────────────────────────────────┐
│         ToolRegistry                  │
├───────────────────────────────────────┤
│ - _tools: dict[str, MCPTool]          │
├───────────────────────────────────────┤
│ + __init__() -> None                  │
│ + register(tool: MCPTool) -> None     │
│ + register_all(tools: List[MCPTool])  │
│ + get_tool(name: str) -> MCPTool?     │
│ + list_tools() -> List[str]           │
│ + validate_tool(tool: MCPTool) -> bool│
│ - _check_protocol(tool: Any) -> None  │
└───────────────────────────────────────┘
```

### Full Class Signature

```python
from typing import Protocol, Any, Optional, List, runtime_checkable

@runtime_checkable
class MCPTool(Protocol):
    """Protocol defining the interface for MCP tools.

    All tools must implement these attributes and methods to be
    registered with the ToolRegistry.

    Attributes:
        name: Unique tool identifier (e.g., 'add_memory')
        description: Human-readable description
        input_schema: JSON Schema for tool arguments
    """

    name: str
    description: str
    input_schema: dict[str, Any]

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute tool with given arguments.

        Args:
            arguments: Tool arguments matching input_schema

        Returns:
            MCP response dictionary with content and isError
        """
        ...


class ToolRegistry:
    """
    Registry for managing MCP tools with validation and lookup.

    Provides centralized tool management for the MCP server, ensuring
    all registered tools comply with the MCPTool protocol and have
    unique names.

    Attributes:
        _tools: Internal dictionary mapping tool names to instances

    Example:
        ```python
        # Create registry
        registry = ToolRegistry()

        # Register tools
        registry.register(AddMemoryTool(core))
        registry.register(SearchMemoryTool(core))

        # Or batch register
        registry.register_all([
            AddMemoryTool(core),
            SearchMemoryTool(core),
            GetStatsTool(core)
        ])

        # Retrieve tool
        tool = registry.get_tool("add_memory")
        if tool:
            result = await tool.execute({"text": "..."})

        # List all tools
        tool_names = registry.list_tools()
        # Returns: ['add_memory', 'search_memory', 'get_stats']
        ```

    Thread Safety:
        NOT thread-safe. Intended for single-threaded stdio MCP server.
        All registration should occur during initialization.
    """

    def __init__(self) -> None:
        """
        Initialize empty tool registry.

        Creates internal storage dictionary. No tools are registered
        by default - must be added via register() or register_all().

        Example:
            >>> registry = ToolRegistry()
            >>> registry.list_tools()
            []
        """

    def register(self, tool: MCPTool) -> None:
        """
        Register a single MCP tool.

        Validates tool protocol compliance and name uniqueness before
        registration. Raises exception if validation fails.

        Args:
            tool: Tool instance implementing MCPTool protocol

        Raises:
            TypeError: If tool doesn't implement MCPTool protocol
            ValueError: If tool.name is empty or already registered
            AttributeError: If tool missing required attributes

        Example:
            >>> registry = ToolRegistry()
            >>> tool = AddMemoryTool(core)
            >>> registry.register(tool)  # Success
            >>> registry.register(tool)  # Raises ValueError (duplicate)
        """

    def register_all(self, tools: List[MCPTool]) -> None:
        """
        Batch register multiple tools.

        Registers tools in order. If any registration fails, stops and
        raises exception. Already-registered tools remain registered.

        Args:
            tools: List of tool instances to register

        Raises:
            TypeError: If any tool doesn't implement MCPTool protocol
            ValueError: If any tool has duplicate name
            AttributeError: If any tool missing required attributes

        Example:
            >>> registry = ToolRegistry()
            >>> tools = [
            ...     AddMemoryTool(core),
            ...     SearchMemoryTool(core),
            ...     GetStatsTool(core)
            ... ]
            >>> registry.register_all(tools)
            >>> len(registry.list_tools())
            3
        """

    def get_tool(self, name: str) -> Optional[MCPTool]:
        """
        Retrieve tool by name.

        Returns tool instance if registered, None otherwise.
        Safe to call with unknown names - no exceptions.

        Args:
            name: Tool name to look up (case-sensitive)

        Returns:
            Tool instance if found, None if not registered

        Example:
            >>> registry = ToolRegistry()
            >>> registry.register(AddMemoryTool(core))
            >>> tool = registry.get_tool("add_memory")
            >>> tool is not None
            True
            >>> registry.get_tool("unknown_tool")
            None
        """

    def list_tools(self) -> List[str]:
        """
        List all registered tool names.

        Returns list of tool names in registration order.
        Empty list if no tools registered.

        Returns:
            List of tool names (strings)

        Example:
            >>> registry = ToolRegistry()
            >>> registry.register_all([
            ...     AddMemoryTool(core),
            ...     SearchMemoryTool(core)
            ... ])
            >>> registry.list_tools()
            ['add_memory', 'search_memory']
        """

    def validate_tool(self, tool: MCPTool) -> bool:
        """
        Validate tool implements MCPTool protocol.

        Checks for required attributes and method signatures.
        Returns True if valid, False if invalid (does not raise).

        Args:
            tool: Tool instance to validate

        Returns:
            True if tool valid, False otherwise

        Example:
            >>> registry = ToolRegistry()
            >>> tool = AddMemoryTool(core)
            >>> registry.validate_tool(tool)
            True
            >>>
            >>> class BadTool:
            ...     pass
            >>> registry.validate_tool(BadTool())
            False
        """

    def _check_protocol(self, tool: Any) -> None:
        """
        Internal method to check protocol compliance.

        Verifies tool has all required attributes with correct types.
        Raises detailed exceptions for missing/invalid attributes.

        Args:
            tool: Object to check

        Raises:
            TypeError: If tool doesn't match MCPTool protocol
            AttributeError: If missing required attribute
            ValueError: If attribute has invalid type

        Note:
            This is an internal helper. Use validate_tool() or register()
            for public validation.
        """
```

---

## Dependencies

### Component Dependencies

**None.** ToolRegistry is dependency-free by design.

**Rationale:**
- Pure organizational component
- No external state or I/O
- Depends only on typing.Protocol (stdlib)

### External Libraries

**Standard Library Only:**
- `typing` (Protocol, runtime_checkable, Any, Optional, List)
  - **Purpose:** Type hints and protocol definition
  - **Why:** Zero-dependency validation using structural typing

**No Third-Party Dependencies:**
- Intentionally minimal
- Easy to test in isolation
- No version conflicts

### Dependency Injection

**None Required.** ToolRegistry is self-contained.

**Tools are injected TO registry, not BY registry:**
```python
# MCPServer creates tools and injects into registry
core = ZapomniCore(config)
registry = ToolRegistry()

# Tools depend on core, but registry doesn't
registry.register(AddMemoryTool(core))  # Tool has dependency
registry.register(SearchMemoryTool(core))  # Tool has dependency
```

---

## State Management

### Attributes

**`_tools: dict[str, MCPTool]`**
- **Type:** `dict[str, MCPTool]`
- **Purpose:** Internal storage mapping tool names to instances
- **Lifetime:** Lives as long as ToolRegistry instance
- **Visibility:** Private (name-mangled with underscore)
- **Access:** Only via public methods (get_tool, list_tools)

**State Transitions:**
```
Empty Registry
    ↓ register(tool1)
Registry with {tool1.name: tool1}
    ↓ register(tool2)
Registry with {tool1.name: tool1, tool2.name: tool2}
    ↓ register_all([tool3, tool4])
Registry with {tool1.name: tool1, tool2.name: tool2, tool3.name: tool3, tool4.name: tool4}
```

**Invariants:**
- All keys in `_tools` equal the `.name` attribute of corresponding value
- No duplicate names (enforced by dict and validation)
- All values implement MCPTool protocol (enforced by validation)

### Thread Safety

**NOT Thread-Safe.**

**Rationale:**
- MCP stdio server is single-threaded by design
- Tool registration happens during initialization (before request handling)
- No concurrent access expected

**If Threading Needed (Future):**
- Add `threading.Lock` around `_tools` access
- Or use `threading.RLock` for reentrancy
- Or make `_tools` a `dict` subclass with locking

**Current Design:** Optimized for simplicity and performance in single-threaded context.

---

## Public Methods (Detailed)

### Method 1: `__init__`

**Signature:**
```python
def __init__(self) -> None
```

**Purpose:** Initialize empty tool registry with no registered tools.

**Parameters:** None

**Returns:** None (constructor)

**Raises:** None (never fails)

**Preconditions:** None

**Postconditions:**
- `_tools` is empty dict `{}`
- `list_tools()` returns `[]`
- `get_tool(any_name)` returns `None`

**Algorithm Outline:**
```
1. Create empty dictionary for _tools
2. Return (implicit)
```

**Edge Cases:**
1. Multiple instantiations → each instance has separate storage
2. No edge cases (trivial constructor)

**Related Methods:**
- Called by: MCPServer during initialization
- Calls: None

**Example:**
```python
registry = ToolRegistry()
assert registry.list_tools() == []
```

---

### Method 2: `register`

**Signature:**
```python
def register(self, tool: MCPTool) -> None
```

**Purpose:** Register single MCP tool after validation.

**Parameters:**

- `tool`: MCPTool
  - **Description:** Tool instance to register
  - **Constraints:**
    - Must implement MCPTool protocol
    - Must have non-empty `name` attribute
    - `name` must not already be registered
    - Must have `description` (str)
    - Must have `input_schema` (dict)
    - Must have `execute` method (async callable)
  - **Example:** `AddMemoryTool(core_engine)`

**Returns:** None (mutates internal state)

**Raises:**

- `TypeError`:
  - When: `tool` doesn't implement MCPTool protocol
  - Message: `"Tool must implement MCPTool protocol"`
  - Example: `register(object())` → TypeError

- `ValueError`:
  - When: `tool.name` is empty string
  - Message: `"Tool name cannot be empty"`
  - Example: `tool.name = ""` → ValueError

- `ValueError`:
  - When: `tool.name` already registered
  - Message: `f"Tool '{tool.name}' already registered"`
  - Example: `register(tool); register(tool)` → ValueError

- `AttributeError`:
  - When: `tool` missing required attribute
  - Message: `f"Tool missing required attribute: {attr_name}"`
  - Example: Tool without `description` → AttributeError

**Preconditions:**
- ToolRegistry initialized
- `tool` object exists

**Postconditions:**
- `tool.name` in `_tools` keys
- `get_tool(tool.name)` returns `tool`
- `list_tools()` includes `tool.name`

**Algorithm Outline:**
```
1. Validate tool protocol compliance (call _check_protocol)
2. Check tool.name is non-empty string
   - If empty → raise ValueError
3. Check tool.name not in _tools
   - If duplicate → raise ValueError
4. Store tool in _tools[tool.name] = tool
5. Return (implicit)
```

**Edge Cases:**

1. **Empty tool name:**
   - Input: `tool.name = ""`
   - Expected: `ValueError("Tool name cannot be empty")`
   - Test:
     ```python
     def test_register_empty_name():
         registry = ToolRegistry()
         tool = Mock(spec=MCPTool)
         tool.name = ""
         with pytest.raises(ValueError, match="cannot be empty"):
             registry.register(tool)
     ```

2. **Duplicate registration:**
   - Input: Same tool registered twice
   - Expected: `ValueError("Tool 'add_memory' already registered")`
   - Test:
     ```python
     def test_register_duplicate():
         registry = ToolRegistry()
         tool = AddMemoryTool(core)
         registry.register(tool)
         with pytest.raises(ValueError, match="already registered"):
             registry.register(tool)
     ```

3. **Invalid protocol:**
   - Input: Object without `execute` method
   - Expected: `TypeError("Tool must implement MCPTool protocol")`
   - Test:
     ```python
     def test_register_invalid_protocol():
         registry = ToolRegistry()
         invalid_tool = object()
         with pytest.raises(TypeError, match="MCPTool protocol"):
             registry.register(invalid_tool)
     ```

4. **Missing attribute:**
   - Input: Tool without `description`
   - Expected: `AttributeError("Tool missing required attribute: description")`
   - Test:
     ```python
     def test_register_missing_attribute():
         registry = ToolRegistry()
         tool = Mock()
         tool.name = "test"
         # No description attribute
         with pytest.raises(AttributeError, match="description"):
             registry.register(tool)
     ```

**Related Methods:**
- Calls: `_check_protocol(tool)`
- Called by: `register_all(tools)`
- Called by: MCPServer during setup

---

### Method 3: `register_all`

**Signature:**
```python
def register_all(self, tools: List[MCPTool]) -> None
```

**Purpose:** Batch register multiple tools in one call.

**Parameters:**

- `tools`: List[MCPTool]
  - **Description:** List of tool instances to register
  - **Constraints:**
    - Each tool must meet `register()` requirements
    - No duplicate names within list
    - Can be empty list (no-op)
  - **Example:** `[AddMemoryTool(core), SearchMemoryTool(core)]`

**Returns:** None (mutates internal state)

**Raises:**

- **Same exceptions as `register()`** for individual tools:
  - `TypeError`: If any tool invalid
  - `ValueError`: If any tool has duplicate name
  - `AttributeError`: If any tool missing attribute

- **Partial Registration Behavior:**
  - If tool N fails, tools 0..N-1 remain registered
  - Caller should handle failure (re-initialize registry?)

**Preconditions:**
- ToolRegistry initialized
- `tools` is valid list (can be empty)

**Postconditions:**
- All tools in list are registered (if no exceptions)
- `list_tools()` includes all tool names from list

**Algorithm Outline:**
```
1. Iterate over tools list
2. For each tool:
   a. Call register(tool)
   b. If exception → propagate immediately
3. Return (implicit)
```

**Edge Cases:**

1. **Empty list:**
   - Input: `register_all([])`
   - Expected: No-op, no errors
   - Test:
     ```python
     def test_register_all_empty():
         registry = ToolRegistry()
         registry.register_all([])
         assert registry.list_tools() == []
     ```

2. **Partial failure:**
   - Input: `[valid_tool, invalid_tool, another_tool]`
   - Expected: `valid_tool` registered, then exception, `another_tool` not registered
   - Test:
     ```python
     def test_register_all_partial_failure():
         registry = ToolRegistry()
         valid = AddMemoryTool(core)
         invalid = object()
         tools = [valid, invalid]

         with pytest.raises(TypeError):
             registry.register_all(tools)

         # First tool still registered
         assert "add_memory" in registry.list_tools()
     ```

3. **Duplicate within list:**
   - Input: `[tool, tool]` (same instance twice)
   - Expected: First succeeds, second raises ValueError
   - Test:
     ```python
     def test_register_all_duplicate_in_list():
         registry = ToolRegistry()
         tool = AddMemoryTool(core)

         with pytest.raises(ValueError, match="already registered"):
             registry.register_all([tool, tool])
     ```

**Related Methods:**
- Calls: `register(tool)` for each tool
- Called by: MCPServer during initialization

**Performance:**
- Time Complexity: O(n) where n = len(tools)
- Space Complexity: O(n) for storing tools

---

### Method 4: `get_tool`

**Signature:**
```python
def get_tool(self, name: str) -> Optional[MCPTool]
```

**Purpose:** Retrieve tool by name with safe lookup (no exceptions).

**Parameters:**

- `name`: str
  - **Description:** Tool name to look up
  - **Constraints:**
    - Case-sensitive match
    - Must be exact string match
  - **Examples:**
    - Valid: `"add_memory"`
    - Invalid (not found): `"Add_Memory"` (case mismatch)

**Returns:**
- **Type:** `Optional[MCPTool]`
- **Success:** Tool instance if name registered
- **Failure:** `None` if name not found

**Raises:** None (safe lookup)

**Preconditions:**
- ToolRegistry initialized
- `name` is valid string

**Postconditions:**
- Registry state unchanged (read-only operation)

**Algorithm Outline:**
```
1. Look up name in _tools dictionary
2. Return value if found, None if not found
```

**Edge Cases:**

1. **Unknown tool name:**
   - Input: `get_tool("nonexistent")`
   - Expected: `None`
   - Test:
     ```python
     def test_get_tool_unknown():
         registry = ToolRegistry()
         result = registry.get_tool("nonexistent")
         assert result is None
     ```

2. **Case sensitivity:**
   - Input: `get_tool("Add_Memory")` (registered as "add_memory")
   - Expected: `None` (case mismatch)
   - Test:
     ```python
     def test_get_tool_case_sensitive():
         registry = ToolRegistry()
         registry.register(AddMemoryTool(core))  # name="add_memory"

         assert registry.get_tool("add_memory") is not None
         assert registry.get_tool("Add_Memory") is None
     ```

3. **Empty registry:**
   - Input: `get_tool(any_name)` on empty registry
   - Expected: `None`
   - Test:
     ```python
     def test_get_tool_empty_registry():
         registry = ToolRegistry()
         assert registry.get_tool("anything") is None
     ```

**Related Methods:**
- Called by: MCPServer during request handling
- Calls: None (direct dict lookup)

**Performance:**
- Time Complexity: O(1) average case (dict lookup)
- Space Complexity: O(1)

---

### Method 5: `list_tools`

**Signature:**
```python
def list_tools(self) -> List[str]
```

**Purpose:** List all registered tool names for discovery.

**Parameters:** None

**Returns:**
- **Type:** `List[str]`
- **Content:** Tool names in registration order
- **Empty:** `[]` if no tools registered

**Raises:** None (always succeeds)

**Preconditions:**
- ToolRegistry initialized

**Postconditions:**
- Registry state unchanged (read-only)

**Algorithm Outline:**
```
1. Extract keys from _tools dictionary
2. Convert to list
3. Return list
```

**Edge Cases:**

1. **Empty registry:**
   - Input: `list_tools()` on empty registry
   - Expected: `[]`
   - Test:
     ```python
     def test_list_tools_empty():
         registry = ToolRegistry()
         assert registry.list_tools() == []
     ```

2. **Single tool:**
   - Input: Registry with one tool
   - Expected: `["tool_name"]`
   - Test:
     ```python
     def test_list_tools_single():
         registry = ToolRegistry()
         registry.register(AddMemoryTool(core))
         assert registry.list_tools() == ["add_memory"]
     ```

3. **Multiple tools:**
   - Input: Registry with 3 tools
   - Expected: All 3 names in list
   - Test:
     ```python
     def test_list_tools_multiple():
         registry = ToolRegistry()
         registry.register_all([
             AddMemoryTool(core),
             SearchMemoryTool(core),
             GetStatsTool(core)
         ])
         names = registry.list_tools()
         assert len(names) == 3
         assert "add_memory" in names
         assert "search_memory" in names
         assert "get_stats" in names
     ```

**Related Methods:**
- Called by: MCPServer for tool discovery
- Called by: Tests for verification
- Calls: None (direct dict access)

**Performance:**
- Time Complexity: O(n) where n = number of tools
- Space Complexity: O(n) for returned list

**Note:** Order is insertion order (Python 3.7+ dict guarantee).

---

### Method 6: `validate_tool`

**Signature:**
```python
def validate_tool(self, tool: MCPTool) -> bool
```

**Purpose:** Check if tool implements MCPTool protocol without raising exceptions.

**Parameters:**

- `tool`: MCPTool
  - **Description:** Tool instance to validate
  - **Type:** Any (accepts any object for checking)

**Returns:**
- **Type:** `bool`
- **True:** Tool valid (implements protocol)
- **False:** Tool invalid (missing attributes/methods)

**Raises:** None (safe validation)

**Preconditions:**
- ToolRegistry initialized

**Postconditions:**
- Registry state unchanged (read-only)

**Algorithm Outline:**
```
1. Try to call _check_protocol(tool)
2. If successful → return True
3. If exception caught → return False
```

**Edge Cases:**

1. **Valid tool:**
   - Input: Proper MCPTool instance
   - Expected: `True`
   - Test:
     ```python
     def test_validate_tool_valid():
         registry = ToolRegistry()
         tool = AddMemoryTool(core)
         assert registry.validate_tool(tool) is True
     ```

2. **Invalid tool (missing method):**
   - Input: Object without `execute` method
   - Expected: `False`
   - Test:
     ```python
     def test_validate_tool_missing_execute():
         registry = ToolRegistry()

         class BadTool:
             name = "bad"
             description = "Bad tool"
             input_schema = {}
             # No execute method

         assert registry.validate_tool(BadTool()) is False
     ```

3. **Invalid tool (missing attribute):**
   - Input: Object without `name`
   - Expected: `False`
   - Test:
     ```python
     def test_validate_tool_missing_name():
         registry = ToolRegistry()

         class BadTool:
             description = "Bad"
             input_schema = {}
             async def execute(self, args): pass

         assert registry.validate_tool(BadTool()) is False
     ```

4. **Completely invalid object:**
   - Input: `object()`
   - Expected: `False`
   - Test:
     ```python
     def test_validate_tool_plain_object():
         registry = ToolRegistry()
         assert registry.validate_tool(object()) is False
     ```

**Related Methods:**
- Calls: `_check_protocol(tool)`
- Called by: Tests, potentially by MCPServer for diagnostics

**Performance:**
- Time Complexity: O(1) (few attribute checks)
- Space Complexity: O(1)

---

### Method 7: `_check_protocol` (Private)

**Signature:**
```python
def _check_protocol(self, tool: Any) -> None
```

**Purpose:** Internal validation helper that raises detailed exceptions.

**Parameters:**

- `tool`: Any
  - **Description:** Object to validate
  - **Type:** Any (accepts anything for validation)

**Returns:** None (raises exception on failure)

**Raises:**

- `TypeError`:
  - When: Tool doesn't match MCPTool protocol
  - Message: `"Tool must implement MCPTool protocol"`

- `AttributeError`:
  - When: Missing required attribute
  - Message: `f"Tool missing required attribute: {attr_name}"`

- `ValueError`:
  - When: Attribute exists but wrong type
  - Message: `f"Tool.{attr_name} must be {expected_type}, got {actual_type}"`

**Algorithm Outline:**
```
1. Check isinstance(tool, MCPTool) using runtime_checkable
   - If False → raise TypeError
2. Check hasattr(tool, "name")
   - If False → raise AttributeError
3. Check isinstance(tool.name, str)
   - If False → raise ValueError
4. Repeat for "description", "input_schema"
5. Check hasattr(tool, "execute") and callable
   - If False → raise AttributeError
6. Return (implicit, validation passed)
```

**Edge Cases:** Same as `validate_tool()`, but raises instead of returning False.

**Related Methods:**
- Called by: `register()`, `validate_tool()`
- Calls: None (uses built-in type checks)

**Note:** This is private API. External callers should use `register()` or `validate_tool()`.

---

## Error Handling

### Exceptions Defined

**No custom exceptions.** Uses standard Python exceptions for clarity.

**Exception Types Used:**

1. **`TypeError`:**
   - **Reason:** Tool doesn't implement protocol
   - **When to Catch:** Never (indicates programmer error)
   - **Example:** `register(object())`

2. **`ValueError`:**
   - **Reason:** Tool name empty or duplicate
   - **When to Catch:** During batch registration (handle partial failure)
   - **Example:** `register(tool); register(tool)` → second raises

3. **`AttributeError`:**
   - **Reason:** Tool missing required attribute
   - **When to Catch:** Never (indicates malformed tool)
   - **Example:** Tool without `description`

### Error Recovery

**No automatic recovery.** All errors are caller's responsibility.

**Caller Strategies:**

1. **For duplicate names:**
   - Check `get_tool(name)` before `register()`
   - Or catch ValueError and skip duplicate

2. **For partial batch failure:**
   - Catch exception during `register_all()`
   - Re-initialize registry and retry
   - Or accept partial registration

3. **For invalid tools:**
   - Validate tools before registration using `validate_tool()`
   - Fix tool implementation

**Example Defensive Registration:**
```python
registry = ToolRegistry()
tools = [tool1, tool2, tool3]

for tool in tools:
    if registry.validate_tool(tool):
        try:
            registry.register(tool)
        except ValueError:
            logger.warning(f"Tool {tool.name} already registered, skipping")
    else:
        logger.error(f"Invalid tool: {tool}")
```

### Error Propagation

**All errors propagate to caller.** ToolRegistry does NOT:
- Log errors (logging is caller's responsibility)
- Retry operations
- Recover from invalid state

**Rationale:** Simple, predictable behavior. Caller knows best how to handle errors.

---

## Usage Examples

### Basic Usage

```python
from zapomni_mcp.tools import ToolRegistry, AddMemoryTool, SearchMemoryTool
from zapomni_core import ZapomniCore

# Initialize core engine
core = ZapomniCore(config)

# Create registry
registry = ToolRegistry()

# Register individual tools
add_tool = AddMemoryTool(core)
registry.register(add_tool)

search_tool = SearchMemoryTool(core)
registry.register(search_tool)

# List registered tools
print(registry.list_tools())
# Output: ['add_memory', 'search_memory']

# Retrieve tool by name
tool = registry.get_tool("add_memory")
if tool:
    result = await tool.execute({"text": "Sample memory"})
```

### Batch Registration

```python
from zapomni_mcp.tools import (
    ToolRegistry,
    AddMemoryTool,
    SearchMemoryTool,
    GetStatsTool
)

# Create all tools
core = ZapomniCore(config)
tools = [
    AddMemoryTool(core),
    SearchMemoryTool(core),
    GetStatsTool(core)
]

# Batch register
registry = ToolRegistry()
registry.register_all(tools)

# Verify all registered
assert len(registry.list_tools()) == 3
```

### Safe Tool Lookup Pattern

```python
# MCPServer request handling
async def handle_tool_request(tool_name: str, arguments: dict) -> dict:
    """Handle incoming MCP tool request."""

    # Safe lookup (returns None if not found)
    tool = registry.get_tool(tool_name)

    if tool is None:
        return {
            "error": {
                "code": -32601,
                "message": f"Tool not found: {tool_name}"
            }
        }

    # Execute tool
    try:
        result = await tool.execute(arguments)
        return {"result": result}
    except Exception as e:
        return {
            "error": {
                "code": -32603,
                "message": f"Tool execution failed: {str(e)}"
            }
        }
```

### Validation Before Registration

```python
from typing import List

def safe_register_tools(
    registry: ToolRegistry,
    tools: List[Any]
) -> List[str]:
    """Safely register tools, skipping invalid ones.

    Returns:
        List of successfully registered tool names
    """
    registered = []

    for tool in tools:
        # Validate first
        if not registry.validate_tool(tool):
            print(f"Skipping invalid tool: {tool}")
            continue

        # Register with duplicate handling
        try:
            registry.register(tool)
            registered.append(tool.name)
        except ValueError as e:
            print(f"Skipping duplicate: {e}")

    return registered
```

---

## Testing Approach

### Unit Tests Required

**Happy Path Tests:**

1. `test_init_creates_empty_registry()`
   - Verify `list_tools()` returns `[]`
   - Verify `get_tool(any_name)` returns `None`

2. `test_register_single_tool_success()`
   - Register valid tool
   - Verify `get_tool(name)` returns tool
   - Verify `list_tools()` includes name

3. `test_register_all_multiple_tools_success()`
   - Register 3 tools via `register_all()`
   - Verify all 3 in `list_tools()`
   - Verify each retrievable via `get_tool()`

4. `test_get_tool_returns_correct_tool()`
   - Register 2 tools
   - Verify `get_tool("tool1")` returns tool1, not tool2

5. `test_list_tools_returns_all_names()`
   - Register 3 tools
   - Verify `list_tools()` has length 3
   - Verify all names present

**Error Tests:**

6. `test_register_duplicate_name_raises_ValueError()`
   - Register tool
   - Register same tool again
   - Expect ValueError with "already registered"

7. `test_register_empty_name_raises_ValueError()`
   - Create tool with `name = ""`
   - Expect ValueError with "cannot be empty"

8. `test_register_invalid_protocol_raises_TypeError()`
   - Register `object()` (no protocol)
   - Expect TypeError with "MCPTool protocol"

9. `test_register_missing_attribute_raises_AttributeError()`
   - Create tool without `description`
   - Expect AttributeError with "description"

10. `test_register_all_partial_failure_registers_valid_tools()`
    - List: [valid_tool, invalid_tool]
    - Expect exception on second
    - Verify first tool still registered

**Edge Case Tests:**

11. `test_get_tool_unknown_name_returns_None()`
    - Call `get_tool("nonexistent")`
    - Expect `None`

12. `test_get_tool_case_sensitive()`
    - Register "add_memory"
    - `get_tool("Add_Memory")` returns `None`

13. `test_list_tools_empty_registry()`
    - New registry
    - `list_tools()` returns `[]`

14. `test_register_all_empty_list()`
    - `register_all([])`
    - No errors, registry still empty

15. `test_validate_tool_valid_returns_True()`
    - Valid tool → `validate_tool()` returns True

16. `test_validate_tool_invalid_returns_False()`
    - Invalid tool → `validate_tool()` returns False

### Mocking Strategy

**Mock MCPTool Instances:**

```python
from unittest.mock import Mock, AsyncMock
import pytest

@pytest.fixture
def mock_tool():
    """Create mock MCPTool for testing."""
    tool = Mock()
    tool.name = "test_tool"
    tool.description = "Test tool"
    tool.input_schema = {"type": "object"}
    tool.execute = AsyncMock(return_value={"status": "ok"})
    return tool
```

**Use Real Protocol Definition:**
- Don't mock `MCPTool` protocol itself
- Mock tool instances that implement protocol

### Integration Tests

**Test with Real Tools:**

```python
@pytest.mark.integration
def test_registry_with_real_tools(core_engine):
    """Integration test with actual AddMemoryTool."""
    from zapomni_mcp.tools import AddMemoryTool

    registry = ToolRegistry()
    tool = AddMemoryTool(core_engine)

    registry.register(tool)

    retrieved = registry.get_tool("add_memory")
    assert retrieved is tool
    assert retrieved.name == "add_memory"
```

**Test MCP Server Integration:**

```python
@pytest.mark.integration
async def test_mcp_server_uses_registry(core_engine):
    """Verify MCPServer integrates with ToolRegistry correctly."""
    from zapomni_mcp.server import MCPServer

    server = MCPServer(core_engine)

    # Verify all tools registered
    tools = server.registry.list_tools()
    assert "add_memory" in tools
    assert "search_memory" in tools
    assert "get_stats" in tools
```

---

## Performance Considerations

### Time Complexity

**Method Complexities:**

| Method         | Best Case | Worst Case | Average Case |
|----------------|-----------|------------|--------------|
| `register()`   | O(1)      | O(1)       | O(1)         |
| `register_all()`| O(n)     | O(n)       | O(n)         |
| `get_tool()`   | O(1)      | O(1)       | O(1)         |
| `list_tools()` | O(n)      | O(n)       | O(n)         |
| `validate_tool()` | O(1)   | O(1)       | O(1)         |

Where n = number of tools (expected: 3-10)

**Rationale:**
- Dict operations are O(1) average case
- Small n means O(n) operations are fast (< 1ms)

### Space Complexity

**Memory Usage:**
- Base: ~1KB (empty ToolRegistry instance)
- Per tool: ~100 bytes (dict entry overhead)
- Total for 10 tools: ~2KB

**No Memory Leaks:**
- No circular references
- Tools owned by registry (clear lifecycle)
- No external resources held

### Optimization Opportunities

**Current Design is Optimal for Use Case:**
- 3-10 tools → dict is perfect data structure
- No need for optimization

**Future Optimizations (If Needed):**
1. **Frozen Registry:**
   - After initialization, make `_tools` read-only
   - Prevent accidental modification
   - Enable further optimizations

2. **Tool Name Validation Cache:**
   - If validation becomes bottleneck
   - Cache validated tool classes
   - Skip re-validation for known types

**Trade-offs:**
- Current: Simple, maintainable, correct
- Optimized: More complex, marginal gains
- **Decision:** Keep simple (YAGNI)

---

## References

### Module Spec

- **zapomni_mcp_module.md** (Level 1)
  - Section: "Tool Registration & Routing"
  - Section: "Public API" → MCPTool protocol definition

### Related Components

- **mcp_server_component.md** (Level 2)
  - Uses ToolRegistry for tool management
  - Calls `get_tool()` during request handling

- **add_memory_tool_component.md** (Level 2)
  - Example of tool implementing MCPTool protocol

- **search_memory_tool_component.md** (Level 2)
  - Example of tool implementing MCPTool protocol

### External Docs

- **Python typing.Protocol:**
  - https://docs.python.org/3/library/typing.html#typing.Protocol
  - Used for structural typing without inheritance

- **Python runtime_checkable:**
  - https://docs.python.org/3/library/typing.html#typing.runtime_checkable
  - Enables isinstance() checks on protocols

---

## Document Status

**Version:** 1.0
**Created:** 2025-11-23
**Author:** Goncharenko Anton aka alienxs2
**Copyright:** Copyright (c) 2025 Goncharenko Anton aka alienxs2
**License:** MIT License
**Status:** Draft - Ready for Verification

**Next Steps:**
1. Multi-agent verification
2. Component-level approval
3. Proceed to function-level specs for each method

---

**Estimated Implementation Time:** 2-3 hours
**Estimated Test Time:** 2-4 hours
**Total Lines of Code (Estimated):** ~150 lines (simple component)
