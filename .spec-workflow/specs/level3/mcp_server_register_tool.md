# MCPServer.register_tool() - Function Specification

**Level:** 3 (Function)
**Component:** MCPServer
**Module:** zapomni_mcp
**Author:** Goncharenko Anton aka alienxs2
**Status:** Draft
**Version:** 1.0
**Created:** 2025-11-23

---

## Function Signature

```python
def register_tool(
    self,
    tool: MCPTool
) -> None:
    """
    Register an MCP tool with the server for client availability.

    This method validates the tool definition, ensures no duplicate registrations,
    and registers the tool with the MCP SDK server instance. Once registered,
    the tool becomes available to MCP clients via the tools/list protocol message.

    The method performs comprehensive validation:
    - Tool has required attributes (name, description, input_schema)
    - Tool name is unique (no duplicates allowed)
    - Input schema is valid JSON Schema format
    - Tool has executable handler (execute method)

    Args:
        tool: MCPTool instance to register, must implement MCPTool protocol:
            - name (str): Unique tool identifier (e.g., "add_memory")
            - description (str): Human-readable description
            - input_schema (dict): JSON Schema for tool arguments
            - execute(arguments: dict) -> dict: Async handler method

    Returns:
        None: Tool is registered and ready for use, no return value

    Raises:
        ValueError: If tool is invalid (missing attrs, duplicate name, bad schema)
        TypeError: If tool doesn't implement MCPTool protocol
        RuntimeError: If server already started (must register before run())

    Example:
        >>> from zapomni_mcp.tools import AddMemoryTool
        >>> server = MCPServer(core_engine=engine)
        >>>
        >>> # Register single tool
        >>> tool = AddMemoryTool(core_engine=engine)
        >>> server.register_tool(tool)
        >>>
        >>> # Tool now available to clients
        >>> assert "add_memory" in server._tools
        >>> assert len(server._tools) == 1

    Thread Safety:
        Not thread-safe. Must be called sequentially during initialization.
        Do not call after server.run() starts.

    Performance:
        - Validation overhead: < 1ms per tool
        - Registration: O(1) dictionary insertion
        - No I/O operations
    """
```

---

## Purpose & Context

### What It Does

The `register_tool()` method is responsible for **validating and registering MCP tools** with the server. It performs three key operations:

1. **Validates** the tool implementation (name, description, schema, execute method)
2. **Checks** for duplicate tool names to prevent conflicts
3. **Registers** the tool with the internal MCP SDK server instance

This method ensures that only properly structured, validated tools are exposed to MCP clients.

### Why It Exists

**MCP Protocol Requirement:**
- MCP servers must maintain a registry of available tools
- Tools must be declared before the server starts accepting requests
- The `tools/list` protocol message requires a complete tool inventory

**Security and Validation:**
- Prevents malformed tools from crashing the server
- Ensures input schemas are valid for client validation
- Enforces naming uniqueness to avoid routing conflicts

### When To Use

**Called During:**
- Server initialization (before `run()` is called)
- Manual tool registration in configuration code
- Typically called by `register_all_tools()` for batch registration

**Example Usage Pattern:**
```python
# Initialize server
server = MCPServer(core_engine=core)

# Register individual tools
server.register_tool(AddMemoryTool(core_engine=core))
server.register_tool(SearchMemoryTool(core_engine=core))
server.register_tool(GetStatsTool(core_engine=core))

# Now start server
await server.run()
```

### When NOT To Use

**Don't use this if:**
- Server is already running → raises RuntimeError
- Tool has already been registered → raises ValueError (duplicate)
- You want to register multiple tools → use `register_all_tools()` instead

---

## Parameters (Detailed)

### tool: MCPTool

**Type:** `MCPTool` (Protocol)

**Purpose:**
An MCP tool implementation that will be made available to clients. Must conform to the MCPTool protocol interface.

**Protocol Definition:**
```python
from typing import Protocol, Any

class MCPTool(Protocol):
    """Protocol that all MCP tools must implement."""

    name: str  # Unique tool identifier
    description: str  # Human-readable description
    input_schema: dict[str, Any]  # JSON Schema for arguments

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute tool with provided arguments."""
        ...
```

**Constraints:**

1. **name Field:**
   - Type: `str`
   - Pattern: `^[a-z][a-z0-9_]*$` (lowercase, alphanumeric, underscores)
   - Length: 1-50 characters
   - Must be unique across all registered tools
   - Examples: `"add_memory"`, `"search_memory"`, `"get_stats"`

2. **description Field:**
   - Type: `str`
   - Minimum length: 10 characters
   - Maximum length: 500 characters
   - Should be descriptive and user-friendly
   - Example: `"Add a memory to the knowledge graph with automatic chunking and embedding"`

3. **input_schema Field:**
   - Type: `dict[str, Any]`
   - Must be valid JSON Schema (draft 2020-12 or later)
   - Must have "type": "object" at root level
   - Must have "properties" field defining parameters
   - Should have "required" array for mandatory parameters
   - Example:
     ```python
     {
         "type": "object",
         "properties": {
             "text": {"type": "string", "description": "Memory content"},
             "metadata": {"type": "object", "description": "Optional metadata"}
         },
         "required": ["text"]
     }
     ```

4. **execute Method:**
   - Must be async (coroutine function)
   - Signature: `async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]`
   - Must return MCP-formatted response dict
   - Must never raise exceptions (return error responses instead)

**Validation Logic:**
```python
# Step 1: Type check
if not hasattr(tool, 'name') or not hasattr(tool, 'description'):
    raise TypeError(f"Tool must have 'name' and 'description' attributes")

if not hasattr(tool, 'input_schema') or not hasattr(tool, 'execute'):
    raise TypeError(f"Tool must have 'input_schema' and 'execute' attributes")

# Step 2: Name validation
if not isinstance(tool.name, str):
    raise ValueError(f"Tool name must be string, got {type(tool.name)}")

if not re.match(r'^[a-z][a-z0-9_]*$', tool.name):
    raise ValueError(f"Tool name '{tool.name}' invalid (must be lowercase, alphanumeric, underscores)")

if len(tool.name) < 1 or len(tool.name) > 50:
    raise ValueError(f"Tool name length must be 1-50 chars, got {len(tool.name)}")

# Step 3: Duplicate check
if tool.name in self._tools:
    raise ValueError(f"Tool '{tool.name}' already registered")

# Step 4: Description validation
if not isinstance(tool.description, str):
    raise ValueError(f"Tool description must be string")

if len(tool.description) < 10:
    raise ValueError(f"Tool description too short (min 10 chars)")

if len(tool.description) > 500:
    raise ValueError(f"Tool description too long (max 500 chars)")

# Step 5: Schema validation
if not isinstance(tool.input_schema, dict):
    raise ValueError(f"Tool input_schema must be dict")

if tool.input_schema.get("type") != "object":
    raise ValueError(f"Tool input_schema must have type='object'")

if "properties" not in tool.input_schema:
    raise ValueError(f"Tool input_schema must have 'properties' field")

# Step 6: Execute method validation
if not callable(getattr(tool, 'execute', None)):
    raise ValueError(f"Tool must have callable 'execute' method")

if not asyncio.iscoroutinefunction(tool.execute):
    raise ValueError(f"Tool execute method must be async")
```

**Examples:**

**Valid - AddMemoryTool:**
```python
class AddMemoryTool:
    name = "add_memory"
    description = "Add a memory to the knowledge graph"
    input_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string"}
        },
        "required": ["text"]
    }

    async def execute(self, arguments: dict) -> dict:
        # Implementation...
        return {"content": [...], "isError": False}

server.register_tool(AddMemoryTool(core_engine=engine))
# Success
```

**Invalid - Missing name:**
```python
class BadTool:
    description = "Missing name field"
    # name is missing!

server.register_tool(BadTool())
# Raises: TypeError("Tool must have 'name' and 'description' attributes")
```

**Invalid - Duplicate name:**
```python
tool1 = AddMemoryTool(core_engine=engine)
tool2 = AddMemoryTool(core_engine=engine)

server.register_tool(tool1)  # Success
server.register_tool(tool2)  # Raises: ValueError("Tool 'add_memory' already registered")
```

**Invalid - Bad name pattern:**
```python
class BadTool:
    name = "Add-Memory"  # Uppercase and hyphen not allowed
    description = "Tool with invalid name"
    input_schema = {"type": "object", "properties": {}}

server.register_tool(BadTool())
# Raises: ValueError("Tool name 'Add-Memory' invalid (must be lowercase, alphanumeric, underscores)")
```

**Invalid - Short description:**
```python
class BadTool:
    name = "bad_tool"
    description = "Too short"  # Only 9 chars
    input_schema = {"type": "object", "properties": {}}

server.register_tool(BadTool())
# Raises: ValueError("Tool description too short (min 10 chars)")
```

**Invalid - Missing execute method:**
```python
class BadTool:
    name = "bad_tool"
    description = "Tool without execute method"
    input_schema = {"type": "object", "properties": {}}
    # No execute method!

server.register_tool(BadTool())
# Raises: ValueError("Tool must have callable 'execute' method")
```

**Invalid - Synchronous execute (not async):**
```python
class BadTool:
    name = "bad_tool"
    description = "Tool with sync execute"
    input_schema = {"type": "object", "properties": {}}

    def execute(self, arguments: dict) -> dict:  # Not async!
        return {}

server.register_tool(BadTool())
# Raises: ValueError("Tool execute method must be async")
```

---

## Return Value

**Type:** `None`

**Purpose:**
This method performs a side effect (registering the tool) and returns nothing. Success is indicated by the absence of exceptions.

**Side Effects:**
1. Tool added to `self._tools` dictionary: `{tool.name: tool}`
2. Tool registered with MCP SDK server instance
3. Tool becomes available in `tools/list` protocol responses
4. Logger records registration event

**Post-Registration State:**
- `len(self._tools)` increased by 1
- `tool.name in self._tools` returns `True`
- Tool is callable via `self._tools[tool.name].execute()`

**No Return Value Examples:**
```python
# Correct usage (no return value expected)
server.register_tool(tool)
print("Tool registered successfully")

# Incorrect usage (don't try to capture return value)
result = server.register_tool(tool)  # result will be None
```

---

## Exceptions

### ValueError

**When Raised:**
- Tool name is empty, too long, or has invalid pattern
- Tool description is too short or too long
- Tool name is duplicate (already registered)
- Tool input_schema is invalid (missing type, properties, etc.)

**Message Formats:**
```python
# Name validation
f"Tool name '{tool.name}' invalid (must be lowercase, alphanumeric, underscores)"
f"Tool name length must be 1-50 chars, got {len(tool.name)}"

# Duplicate
f"Tool '{tool.name}' already registered"

# Description
f"Tool description too short (min 10 chars)"
f"Tool description too long (max 500 chars)"

# Schema
f"Tool input_schema must have type='object'"
f"Tool input_schema must have 'properties' field"

# Execute method
f"Tool must have callable 'execute' method"
f"Tool execute method must be async"
```

**Recovery:** Fix tool definition and retry registration

### TypeError

**When Raised:**
- Tool is missing required attributes (name, description, input_schema, execute)
- Tool attributes are wrong types (e.g., name is int instead of str)

**Message Formats:**
```python
f"Tool must have 'name' and 'description' attributes"
f"Tool must have 'input_schema' and 'execute' attributes"
f"Tool name must be string, got {type(tool.name)}"
```

**Recovery:** Implement proper MCPTool protocol

### RuntimeError

**When Raised:**
- Attempting to register tool after server has started
- Server is in shutdown state

**Message Formats:**
```python
"Cannot register tools after server has started"
"Server is shutting down, registration not allowed"
```

**Recovery:** Register tools before calling `server.run()`

---

## Algorithm (Pseudocode)

```
FUNCTION register_tool(self, tool):
    # Step 1: Check server state
    IF self._running IS True:
        RAISE RuntimeError("Cannot register tools after server has started")

    # Step 2: Validate tool has required protocol attributes
    IF NOT hasattr(tool, 'name') OR NOT hasattr(tool, 'description'):
        RAISE TypeError("Tool must have 'name' and 'description' attributes")

    IF NOT hasattr(tool, 'input_schema') OR NOT hasattr(tool, 'execute'):
        RAISE TypeError("Tool must have 'input_schema' and 'execute' attributes")

    # Step 3: Validate tool name
    IF NOT isinstance(tool.name, str):
        RAISE ValueError("Tool name must be string")

    IF NOT matches_pattern(tool.name, r'^[a-z][a-z0-9_]*$'):
        RAISE ValueError("Tool name invalid (must be lowercase, alphanumeric, underscores)")

    IF len(tool.name) < 1 OR len(tool.name) > 50:
        RAISE ValueError("Tool name length must be 1-50 chars")

    # Step 4: Check for duplicate registration
    IF tool.name IN self._tools:
        RAISE ValueError("Tool already registered")

    # Step 5: Validate description
    IF NOT isinstance(tool.description, str):
        RAISE ValueError("Tool description must be string")

    IF len(tool.description) < 10:
        RAISE ValueError("Tool description too short (min 10 chars)")

    IF len(tool.description) > 500:
        RAISE ValueError("Tool description too long (max 500 chars)")

    # Step 6: Validate input_schema
    IF NOT isinstance(tool.input_schema, dict):
        RAISE ValueError("Tool input_schema must be dict")

    IF tool.input_schema.get("type") != "object":
        RAISE ValueError("Tool input_schema must have type='object'")

    IF "properties" NOT IN tool.input_schema:
        RAISE ValueError("Tool input_schema must have 'properties' field")

    # Step 7: Validate execute method
    IF NOT callable(getattr(tool, 'execute', None)):
        RAISE ValueError("Tool must have callable 'execute' method")

    IF NOT is_coroutine_function(tool.execute):
        RAISE ValueError("Tool execute method must be async")

    # Step 8: Register tool in internal registry
    self._tools[tool.name] = tool

    # Step 9: Register with MCP SDK server
    self._server.register_tool(
        name=tool.name,
        description=tool.description,
        input_schema=tool.input_schema,
        handler=tool.execute
    )

    # Step 10: Log successful registration
    self._logger.info(
        "tool_registered",
        tool_name=tool.name,
        total_tools=len(self._tools)
    )

    # Step 11: Return (no value)
    RETURN None
END FUNCTION
```

---

## Preconditions

### Required State

✅ **Server State:**
- MCPServer instance must be initialized (`__init__` called)
- `self._server` (MCP SDK server) must be initialized
- `self._tools` dictionary must be initialized (empty or with tools)
- `self._running` must be `False` (server not started yet)

✅ **Tool Validity:**
- Tool must implement MCPTool protocol (name, description, input_schema, execute)
- All tool attributes must have correct types
- Tool name must be unique (not in `self._tools`)

### Not Required

❌ **Core Engine State:**
- Core engine can be in any state (tool validation is independent)

❌ **Network State:**
- No network connections required for registration

---

## Postconditions

### On Success

✅ **Tool Registered:**
- Tool added to `self._tools[tool.name]`
- Tool registered with MCP SDK server
- Tool count incremented

✅ **Tool Available:**
- Tool appears in `tools/list` responses (after server starts)
- Tool can be invoked via `tools/call` by clients
- Tool accessible via `self._tools[tool.name]`

✅ **Logging Complete:**
- Registration logged with tool name and total count

✅ **State Unchanged Elsewhere:**
- Server running state unchanged
- Other tools unaffected

### On Error (Exception Raised)

✅ **No Registration:**
- Tool NOT added to `self._tools`
- Tool NOT registered with MCP SDK
- Tool count unchanged

✅ **State Unchanged:**
- Server state unchanged
- Existing tools unaffected

---

## Edge Cases & Handling

### Edge Case 1: Empty Tool Name

**Scenario:** Tool has empty string as name

**Input:**
```python
class BadTool:
    name = ""
    description = "Tool with empty name"
    input_schema = {"type": "object", "properties": {}}
    async def execute(self, args): return {}
```

**Processing:**
1. Validate name length: `len("") = 0`
2. Check: `0 < 1` → True
3. Raise `ValueError("Tool name length must be 1-50 chars, got 0")`

**Expected Behavior:**
```python
server.register_tool(BadTool())
# Raises: ValueError("Tool name length must be 1-50 chars, got 0")
```

**Test Scenario:**
```python
def test_register_tool_empty_name():
    server = MCPServer(core_engine=mock_engine)

    class BadTool:
        name = ""
        description = "Valid description"
        input_schema = {"type": "object", "properties": {}}
        async def execute(self, args): return {}

    with pytest.raises(ValueError, match="name length must be 1-50"):
        server.register_tool(BadTool())
```

---

### Edge Case 2: Duplicate Tool Name

**Scenario:** Attempting to register two tools with same name

**Input:**
```python
tool1 = AddMemoryTool(core_engine=engine)
tool2 = AddMemoryTool(core_engine=engine)

server.register_tool(tool1)  # First registration
server.register_tool(tool2)  # Duplicate
```

**Processing:**
1. First registration: `tool1.name = "add_memory"` → Success
2. Second registration: Check `"add_memory" in self._tools` → True
3. Raise `ValueError("Tool 'add_memory' already registered")`

**Expected Behavior:**
```python
# First registration succeeds
server.register_tool(tool1)
assert "add_memory" in server._tools

# Second registration fails
with pytest.raises(ValueError, match="already registered"):
    server.register_tool(tool2)

# Only first tool remains
assert len(server._tools) == 1
```

**Test Scenario:**
```python
def test_register_tool_duplicate_name():
    server = MCPServer(core_engine=mock_engine)

    tool1 = AddMemoryTool(core_engine=mock_engine)
    tool2 = AddMemoryTool(core_engine=mock_engine)

    server.register_tool(tool1)  # Success

    with pytest.raises(ValueError, match="already registered"):
        server.register_tool(tool2)

    assert len(server._tools) == 1
```

---

### Edge Case 3: Tool Name with Invalid Characters

**Scenario:** Tool name contains uppercase or special characters

**Input:**
```python
class BadTool:
    name = "Add-Memory!"  # Uppercase, hyphen, exclamation
    description = "Tool with bad name"
    input_schema = {"type": "object", "properties": {}}
    async def execute(self, args): return {}
```

**Processing:**
1. Validate pattern: `re.match(r'^[a-z][a-z0-9_]*$', "Add-Memory!")` → None
2. Raise `ValueError("Tool name 'Add-Memory!' invalid...")`

**Expected Behavior:**
```python
server.register_tool(BadTool())
# Raises: ValueError("Tool name 'Add-Memory!' invalid (must be lowercase, alphanumeric, underscores)")
```

**Test Scenario:**
```python
def test_register_tool_invalid_name_pattern():
    server = MCPServer(core_engine=mock_engine)

    invalid_names = [
        "Add-Memory",      # Uppercase, hyphen
        "add_memory!",     # Exclamation
        "addMemory",       # camelCase
        "add memory",      # Space
        "123_add",         # Starts with number
    ]

    for bad_name in invalid_names:
        class BadTool:
            name = bad_name
            description = "Valid description"
            input_schema = {"type": "object", "properties": {}}
            async def execute(self, args): return {}

        with pytest.raises(ValueError, match="invalid.*lowercase"):
            server.register_tool(BadTool())
```

---

### Edge Case 4: Tool Missing input_schema

**Scenario:** Tool doesn't have input_schema attribute

**Input:**
```python
class BadTool:
    name = "bad_tool"
    description = "Tool without schema"
    # input_schema is missing!
    async def execute(self, args): return {}
```

**Processing:**
1. Check `hasattr(tool, 'input_schema')` → False
2. Raise `TypeError("Tool must have 'input_schema' and 'execute' attributes")`

**Expected Behavior:**
```python
server.register_tool(BadTool())
# Raises: TypeError("Tool must have 'input_schema' and 'execute' attributes")
```

**Test Scenario:**
```python
def test_register_tool_missing_input_schema():
    server = MCPServer(core_engine=mock_engine)

    class BadTool:
        name = "bad_tool"
        description = "Valid description"
        # Missing input_schema
        async def execute(self, args): return {}

    with pytest.raises(TypeError, match="must have 'input_schema'"):
        server.register_tool(BadTool())
```

---

### Edge Case 5: Tool input_schema Invalid (Not an Object)

**Scenario:** input_schema has wrong type (e.g., "array" instead of "object")

**Input:**
```python
class BadTool:
    name = "bad_tool"
    description = "Tool with invalid schema"
    input_schema = {
        "type": "array",  # Should be "object"
        "items": {"type": "string"}
    }
    async def execute(self, args): return {}
```

**Processing:**
1. Check `tool.input_schema.get("type")` → `"array"`
2. Compare: `"array" != "object"` → True
3. Raise `ValueError("Tool input_schema must have type='object'")`

**Expected Behavior:**
```python
server.register_tool(BadTool())
# Raises: ValueError("Tool input_schema must have type='object'")
```

**Test Scenario:**
```python
def test_register_tool_schema_not_object():
    server = MCPServer(core_engine=mock_engine)

    class BadTool:
        name = "bad_tool"
        description = "Valid description"
        input_schema = {"type": "array", "items": {}}
        async def execute(self, args): return {}

    with pytest.raises(ValueError, match="type='object'"):
        server.register_tool(BadTool())
```

---

### Edge Case 6: Tool execute Not Async

**Scenario:** Tool has synchronous execute method instead of async

**Input:**
```python
class BadTool:
    name = "bad_tool"
    description = "Tool with sync execute"
    input_schema = {"type": "object", "properties": {}}

    def execute(self, arguments: dict) -> dict:  # Not async!
        return {}
```

**Processing:**
1. Check `asyncio.iscoroutinefunction(tool.execute)` → False
2. Raise `ValueError("Tool execute method must be async")`

**Expected Behavior:**
```python
server.register_tool(BadTool())
# Raises: ValueError("Tool execute method must be async")
```

**Test Scenario:**
```python
def test_register_tool_execute_not_async():
    server = MCPServer(core_engine=mock_engine)

    class BadTool:
        name = "bad_tool"
        description = "Valid description"
        input_schema = {"type": "object", "properties": {}}

        def execute(self, args):  # Sync, not async
            return {}

    with pytest.raises(ValueError, match="must be async"):
        server.register_tool(BadTool())
```

---

### Edge Case 7: Register After Server Started

**Scenario:** Attempting to register tool after `run()` has been called

**Input:**
```python
server = MCPServer(core_engine=engine)
await server.run()  # Server now running

# Later, try to register tool
tool = AddMemoryTool(core_engine=engine)
server.register_tool(tool)  # Should fail
```

**Processing:**
1. Check `self._running` → True
2. Raise `RuntimeError("Cannot register tools after server has started")`

**Expected Behavior:**
```python
# Server already running
assert server._running is True

# Registration fails
with pytest.raises(RuntimeError, match="after server has started"):
    server.register_tool(tool)
```

**Test Scenario:**
```python
async def test_register_tool_after_server_started():
    server = MCPServer(core_engine=mock_engine)

    # Start server (mocked to set _running=True)
    server._running = True

    tool = AddMemoryTool(core_engine=mock_engine)

    with pytest.raises(RuntimeError, match="after server has started"):
        server.register_tool(tool)
```

---

## Test Scenarios (Complete List)

### Happy Path Tests

**1. test_register_tool_success_single**
- **Input:** Valid AddMemoryTool
- **Expected:** Tool registered, `len(self._tools) == 1`, no exception
- **Verifies:** Basic registration success

**2. test_register_tool_success_multiple**
- **Input:** Register 3 different tools sequentially
- **Expected:** All 3 registered, `len(self._tools) == 3`
- **Verifies:** Multiple tool registration

**3. test_register_tool_success_complex_schema**
- **Input:** Tool with nested input_schema (properties with nested objects)
- **Expected:** Registration succeeds
- **Verifies:** Complex schema handling

**4. test_register_tool_name_with_underscores**
- **Input:** Tool name `"search_memory_advanced"`
- **Expected:** Registration succeeds
- **Verifies:** Underscores allowed in names

**5. test_register_tool_name_boundary_length**
- **Input:** Tool name with exactly 50 characters (boundary)
- **Expected:** Registration succeeds
- **Verifies:** Max length boundary case

---

### Validation Error Tests

**6. test_register_tool_empty_name**
- **Input:** Tool with `name = ""`
- **Expected:** `ValueError` mentioning length
- **Verifies:** Edge case 1

**7. test_register_tool_duplicate_name**
- **Input:** Register same tool twice
- **Expected:** First succeeds, second raises `ValueError`
- **Verifies:** Edge case 2

**8. test_register_tool_invalid_name_pattern**
- **Input:** Names with uppercase, hyphens, special chars
- **Expected:** `ValueError` mentioning pattern
- **Verifies:** Edge case 3

**9. test_register_tool_name_too_long**
- **Input:** Tool name with 51 characters
- **Expected:** `ValueError` mentioning max 50 chars
- **Verifies:** Length validation

**10. test_register_tool_description_too_short**
- **Input:** Description with 9 characters
- **Expected:** `ValueError` mentioning min 10 chars
- **Verifies:** Description validation

**11. test_register_tool_description_too_long**
- **Input:** Description with 501 characters
- **Expected:** `ValueError` mentioning max 500 chars
- **Verifies:** Description boundary

---

### Type Error Tests

**12. test_register_tool_missing_name**
- **Input:** Tool without `name` attribute
- **Expected:** `TypeError` mentioning required attributes
- **Verifies:** Protocol validation

**13. test_register_tool_missing_input_schema**
- **Input:** Tool without `input_schema`
- **Expected:** `TypeError`
- **Verifies:** Edge case 4

**14. test_register_tool_name_wrong_type**
- **Input:** Tool with `name = 123` (int instead of str)
- **Expected:** `ValueError` about type
- **Verifies:** Type validation

---

### Schema Validation Tests

**15. test_register_tool_schema_not_object**
- **Input:** Schema with `type="array"`
- **Expected:** `ValueError` about type='object'
- **Verifies:** Edge case 5

**16. test_register_tool_schema_missing_properties**
- **Input:** Schema without "properties" field
- **Expected:** `ValueError` about properties
- **Verifies:** Schema structure

**17. test_register_tool_schema_not_dict**
- **Input:** `input_schema = "invalid"`
- **Expected:** `ValueError` about dict type
- **Verifies:** Schema type validation

---

### Execute Method Tests

**18. test_register_tool_execute_not_async**
- **Input:** Tool with sync execute method
- **Expected:** `ValueError` about async
- **Verifies:** Edge case 6

**19. test_register_tool_missing_execute**
- **Input:** Tool without execute method
- **Expected:** `ValueError` about callable
- **Verifies:** Execute presence

---

### State Tests

**20. test_register_tool_after_server_started**
- **Input:** Register tool when `_running=True`
- **Expected:** `RuntimeError`
- **Verifies:** Edge case 7

---

### Integration Tests

**21. test_register_tool_appears_in_tools_dict**
- **Verifies:** Tool accessible via `server._tools[tool.name]`

**22. test_register_tool_increments_count**
- **Verifies:** `len(server._tools)` incremented correctly

**23. test_register_tool_logs_registration**
- **Verifies:** Logger called with correct parameters

---

## Performance Requirements

### Latency Targets

**Registration Time:**
- Single tool: < 1ms (validation + dict insertion)
- Batch of 10 tools: < 10ms total

**Validation Overhead:**
- Name pattern check: < 0.1ms (regex match)
- Schema validation: < 0.5ms (dict traversal)
- Type checks: < 0.1ms (hasattr, isinstance)

### Memory Usage

**Per Tool:**
- Tool reference in dict: ~100 bytes
- Total for 10 tools: ~1KB

### Scalability

**Tool Count:**
- Supports 1-100 tools efficiently
- O(1) registration time (dict insertion)
- O(1) duplicate check (dict lookup)

---

## Security Considerations

### Input Validation

✅ **All tool attributes validated:**
- Name pattern prevents injection attacks
- Schema structure validated (JSON Schema format)
- Type checking prevents type confusion

✅ **Duplicate prevention:**
- Prevents malicious tool override attacks
- Ensures unique tool routing

### Error Message Safety

✅ **Safe to expose:**
- Validation error messages (controlled format)
- Tool name in errors (validated input)

❌ **Never expose:**
- Internal server state details
- Other registered tools in error messages (information leak)

---

## Related Functions

### Calls

**1. `hasattr(tool, attr)` (stdlib)**
- **Purpose:** Check tool has required attributes
- **When:** During protocol validation

**2. `re.match(pattern, tool.name)` (stdlib)**
- **Purpose:** Validate tool name pattern
- **When:** During name validation

**3. `asyncio.iscoroutinefunction(tool.execute)` (stdlib)**
- **Purpose:** Verify execute is async
- **When:** During execute validation

**4. `self._server.register_tool()` (MCP SDK)**
- **Purpose:** Register with MCP SDK server
- **When:** After all validation passes

**5. `self._logger.info()` (structlog)**
- **Purpose:** Log successful registration
- **When:** After registration complete

### Called By

**1. `MCPServer.register_all_tools()`**
- **Purpose:** Batch registration of all tools
- **When:** During server initialization
- **How:** Calls `register_tool()` for each tool in list

**2. Application initialization code**
- **Purpose:** Manual tool registration
- **When:** Server setup phase
- **How:** Direct call to `register_tool()`

---

## Implementation Notes

### Dependencies

**Standard Library:**
- `re` - Name pattern validation
- `asyncio` - Coroutine function checking
- `typing` - Type annotations

**External Libraries:**
- `structlog` - Logging
- `mcp.server` - MCP SDK server instance

**Internal Dependencies:**
- None (standalone validation logic)

### Known Limitations

**1. Static Registration Only:**
- Tools cannot be registered after server starts
- No dynamic tool loading at runtime
- Workaround: Restart server to add new tools

**2. No Unregistration:**
- Once registered, tools cannot be removed
- No `unregister_tool()` method
- Workaround: Restart server with different tool set

**3. No Tool Versioning:**
- Tool names must be unique (no versions)
- Cannot register `add_memory_v1` and `add_memory_v2`
- Workaround: Use different names

### Future Enhancements

**1. Dynamic Registration:**
- Allow tool registration/unregistration at runtime
- Requires server state management updates

**2. Tool Versioning:**
- Support multiple versions of same tool
- Include version in routing logic

**3. Enhanced Validation:**
- JSON Schema validation using `jsonschema` library
- Deeper schema structure checks

**4. Tool Discovery:**
- Auto-discover tools in package via introspection
- Reduce boilerplate registration code

---

## References

### Component Spec
- [MCPServer Component Specification](../level2/mcp_server_component.md)

### Module Spec
- [Zapomni MCP Module Specification](../level1/zapomni_mcp_module.md)

### Related Function Specs
- `MCPServer.register_all_tools()` (Level 3) - Batch registration wrapper
- `MCPServer.run()` (Level 3) - Server execution (calls after registration)
- `MCPServer._handle_request()` (Level 3) - Tool routing logic

### External Documentation
- [MCP Specification - Tools](https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/tools/)
- [Python Protocol - PEP 544](https://peps.python.org/pep-0544/)
- [JSON Schema Specification](https://json-schema.org/specification.html)

---

## Document Status

**Version:** 1.0
**Created:** 2025-11-23
**Author:** Goncharenko Anton aka alienxs2
**Copyright:** Copyright (c) 2025 Goncharenko Anton aka alienxs2
**License:** MIT License
**Status:** Draft - Ready for Review

**Next Steps:**
1. Review against function specification template
2. Verify alignment with component spec
3. Create test implementation from scenarios
4. Proceed to implementation

---

**Estimated Implementation Effort:** 1-2 hours
**Lines of Code (Estimated):** ~60 lines (including validation)
**Test Coverage Target:** 95%+ (23 test scenarios defined)
**Test File:** `tests/unit/mcp/test_mcp_server_register_tool.py`
