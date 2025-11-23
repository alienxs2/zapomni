# MCPServer.handle_request() - Function Specification

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
async def handle_request(
    self,
    request: dict
) -> dict:
    """
    Handle incoming MCP protocol request and route to appropriate tool.

    This is the core request routing method called by the MCP SDK server for
    each incoming JSON-RPC 2.0 message from stdin. It parses the request,
    validates the method and parameters, routes to the appropriate tool handler,
    and formats the response according to MCP protocol specification.

    The method implements comprehensive error handling with zero exception
    propagation - all errors are caught and converted to MCP error responses.

    Args:
        request: MCP request dictionary conforming to JSON-RPC 2.0 format:
            - method (str, required): MCP method ("tools/call", "tools/list", etc.)
            - params (dict, optional): Method-specific parameters
            - id (int|str, optional): Request identifier for correlation

            For "tools/call" method:
            - params.name (str): Tool name to execute
            - params.arguments (dict): Tool-specific arguments

    Returns:
        MCP response dictionary in JSON-RPC 2.0 format:

        Success response (tools/call):
        {
            "content": [...],  # MCP content blocks from tool
            "isError": False
        }

        Error response:
        {
            "error": {
                "code": int,  # Error code (-32600 to -32603, or custom)
                "message": str,  # Human-readable error message
                "data": {...}  # Optional additional error details
            }
        }

    Raises:
        Never raises exceptions. All errors returned as MCP error responses.

    Example:
        >>> request = {
        ...     "method": "tools/call",
        ...     "params": {
        ...         "name": "add_memory",
        ...         "arguments": {"text": "Python is great"}
        ...     },
        ...     "id": 1
        ... }
        >>> response = await server.handle_request(request)
        >>> print(response["isError"])
        False

    Thread Safety:
        Not thread-safe. MCP stdio transport ensures sequential request
        processing, so concurrent calls won't occur.

    Performance:
        - Request parsing: < 1ms
        - Routing overhead: < 1ms (dict lookup)
        - Total overhead: < 5ms (excluding tool execution)
    """
```

---

## Purpose & Context

### What It Does

The `handle_request()` method is the **central request dispatcher** for the MCP server. It performs:

1. **Request Parsing** - Extract method, params, and request ID from JSON-RPC message
2. **Method Routing** - Dispatch to appropriate handler (tools/call, tools/list, etc.)
3. **Tool Execution** - Call registered tool's execute() method for tools/call
4. **Response Formatting** - Convert tool responses to MCP protocol format
5. **Error Handling** - Catch all exceptions and return MCP error responses

This method acts as the **protocol adapter layer** between MCP clients and Zapomni tools.

### Why It Exists

**MCP Protocol Requirement:**
- MCP servers must handle standardized JSON-RPC 2.0 requests
- Each request type (tools/call, tools/list, etc.) requires specific handling
- Responses must conform to MCP content block format

**Separation of Concerns:**
- Request handling logic separated from tool business logic
- Protocol compliance isolated in this method
- Tools only need to implement execute(), not protocol details

### When To Use

**Called Automatically By:**
- MCP SDK server when receiving stdin messages from client
- Internal MCP request routing infrastructure

**Not Called Directly By:**
- Application code (handled by MCP SDK)
- Tool implementations (tools call each other via core engine, not requests)

### When NOT To Use

**Don't use this if:**
- You want to call a tool programmatically → use `tool.execute()` directly
- You're testing tool logic → test execute() method, not request handler
- You need synchronous execution → this is async only

---

## Parameters (Detailed)

### request: dict

**Type:** `dict`

**Purpose:**
Contains the complete MCP request message from the client, following JSON-RPC 2.0 specification.

**Structure:**
```python
{
    "jsonrpc": "2.0",  # Always "2.0" (validated by MCP SDK)
    "method": str,     # Required: MCP method name
    "params": dict,    # Optional: Method parameters
    "id": int | str    # Optional: Request identifier
}
```

**Method-Specific Formats:**

**1. tools/call (Tool Execution):**
```python
{
    "method": "tools/call",
    "params": {
        "name": str,        # Required: Tool name to execute
        "arguments": dict   # Required: Tool-specific arguments
    },
    "id": 1
}
```

**2. tools/list (List Available Tools):**
```python
{
    "method": "tools/list",
    "params": {},  # No parameters needed
    "id": 2
}
```

**3. resources/list (List Resources - Future):**
```python
{
    "method": "resources/list",
    "params": {},
    "id": 3
}
```

**Constraints:**

1. **method Field:**
   - Type: `str`
   - Required: Yes
   - Valid values: `"tools/call"`, `"tools/list"`, `"resources/list"`, etc.
   - Case-sensitive

2. **params Field:**
   - Type: `dict`
   - Required: No (defaults to empty dict)
   - For "tools/call": Must have "name" and "arguments" keys
   - For "tools/list": Can be empty

3. **id Field:**
   - Type: `int` or `str`
   - Required: No (notifications don't have IDs)
   - Used for request-response correlation

**Validation Logic:**
```python
# Step 1: Check method exists
if "method" not in request:
    return _error_response(-32600, "Invalid request: missing method field")

method = request["method"]

# Step 2: Extract params (default to empty dict)
params = request.get("params", {})

# Step 3: Extract request ID (for response correlation)
request_id = request.get("id")

# Step 4: Route based on method
if method == "tools/call":
    # Validate tools/call params
    if "name" not in params:
        return _error_response(-32602, "Invalid params: missing tool name")

    if "arguments" not in params:
        return _error_response(-32602, "Invalid params: missing arguments")

    tool_name = params["name"]
    tool_arguments = params["arguments"]

    # Check tool exists
    if tool_name not in self._tools:
        return _error_response(-32601, f"Tool not found: {tool_name}")

    # Execute tool
    tool = self._tools[tool_name]
    result = await tool.execute(tool_arguments)
    return result

elif method == "tools/list":
    # Return list of registered tools
    tools_list = [
        {
            "name": tool.name,
            "description": tool.description,
            "inputSchema": tool.input_schema
        }
        for tool in self._tools.values()
    ]
    return {"tools": tools_list}

else:
    return _error_response(-32601, f"Method not found: {method}")
```

**Examples:**

**Valid - tools/call:**
```python
request = {
    "method": "tools/call",
    "params": {
        "name": "add_memory",
        "arguments": {"text": "Python is great"}
    },
    "id": 1
}

response = await server.handle_request(request)
# Returns: {"content": [...], "isError": False}
```

**Valid - tools/list:**
```python
request = {
    "method": "tools/list",
    "params": {},
    "id": 2
}

response = await server.handle_request(request)
# Returns: {"tools": [...]}
```

**Invalid - Missing method:**
```python
request = {
    "params": {"name": "add_memory"},
    "id": 1
}

response = await server.handle_request(request)
# Returns: {"error": {"code": -32600, "message": "Invalid request: missing method field"}}
```

**Invalid - Unknown method:**
```python
request = {
    "method": "unknown/method",
    "id": 1
}

response = await server.handle_request(request)
# Returns: {"error": {"code": -32601, "message": "Method not found: unknown/method"}}
```

**Invalid - tools/call missing tool name:**
```python
request = {
    "method": "tools/call",
    "params": {
        "arguments": {"text": "test"}
        # Missing "name"
    },
    "id": 1
}

response = await server.handle_request(request)
# Returns: {"error": {"code": -32602, "message": "Invalid params: missing tool name"}}
```

**Invalid - Tool not registered:**
```python
request = {
    "method": "tools/call",
    "params": {
        "name": "nonexistent_tool",
        "arguments": {}
    },
    "id": 1
}

response = await server.handle_request(request)
# Returns: {"error": {"code": -32601, "message": "Tool not found: nonexistent_tool"}}
```

---

## Return Value

**Type:** `dict`

**Purpose:**
MCP-formatted response conforming to JSON-RPC 2.0 and MCP content block specifications.

### Success Response (tools/call)

**Structure:**
```python
{
    "content": [
        {
            "type": "text",
            "text": str  # Tool output message
        }
    ],
    "isError": False
}
```

**Example:**
```python
{
    "content": [
        {
            "type": "text",
            "text": "Memory stored successfully.\nID: 550e8400-e29b-41d4-a716-446655440000\nChunks created: 3"
        }
    ],
    "isError": False
}
```

### Success Response (tools/list)

**Structure:**
```python
{
    "tools": [
        {
            "name": str,
            "description": str,
            "inputSchema": dict
        },
        ...
    ]
}
```

**Example:**
```python
{
    "tools": [
        {
            "name": "add_memory",
            "description": "Add a memory to the knowledge graph",
            "inputSchema": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"]
            }
        },
        {
            "name": "search_memory",
            "description": "Search the knowledge graph",
            "inputSchema": {...}
        }
    ]
}
```

### Error Response (Any Error)

**Structure:**
```python
{
    "error": {
        "code": int,      # JSON-RPC error code
        "message": str,   # Human-readable error
        "data": dict      # Optional error details
    }
}
```

**Error Codes:**
- `-32600`: Invalid Request (malformed JSON, missing fields)
- `-32601`: Method Not Found (unknown method or tool)
- `-32602`: Invalid Params (missing required parameters)
- `-32603`: Internal Error (tool execution failed)
- `-32700`: Parse Error (JSON parsing failed - handled by SDK)

**Example:**
```python
{
    "error": {
        "code": -32601,
        "message": "Tool not found: nonexistent_tool",
        "data": {
            "available_tools": ["add_memory", "search_memory", "get_stats"]
        }
    }
}
```

---

## Exceptions

### Never Raises

**Critical Design Decision:**

This method **NEVER** raises exceptions. All errors are caught and returned as MCP error responses.

**Rationale:**
1. MCP protocol expects standard error format
2. Uncaught exceptions would crash stdio server loop
3. Clients need predictable response structure
4. Errors still logged to stderr for debugging

### Exception Handling Strategy

```python
try:
    # Parse and validate request
    method = request.get("method")
    if not method:
        return self._error_response(-32600, "Invalid request: missing method")

    params = request.get("params", {})

    # Route and execute
    if method == "tools/call":
        result = await self._handle_tool_call(params)
        return result
    elif method == "tools/list":
        return self._handle_tools_list()
    else:
        return self._error_response(-32601, f"Method not found: {method}")

except Exception as e:
    # Unexpected error (bug or system issue)
    self._logger.error(
        "request_handling_error",
        error_type=type(e).__name__,
        error=str(e),
        exc_info=True
    )
    return self._error_response(
        -32603,
        "Internal error occurred while processing request"
    )
```

---

## Algorithm (Pseudocode)

```
FUNCTION handle_request(self, request: dict) -> dict:
    # Step 1: Initialize request context
    request_id = request.get("id", "<no-id>")
    logger = self._logger.bind(request_id=request_id)
    logger.info("request_received")

    # Step 2: Validate and extract method
    TRY:
        method = request.get("method")
        IF method IS None:
            logger.warning("missing_method")
            RETURN _error_response(-32600, "Invalid request: missing method field")

        params = request.get("params", {})
        logger.info("request_parsed", method=method)

    CATCH Exception as e:
        logger.error("parse_error", error=str(e))
        RETURN _error_response(-32600, "Invalid request format")

    # Step 3: Route based on method
    TRY:
        IF method == "tools/call":
            # Step 3.1: Validate tools/call params
            IF "name" NOT IN params:
                logger.warning("missing_tool_name")
                RETURN _error_response(-32602, "Invalid params: missing tool name")

            IF "arguments" NOT IN params:
                logger.warning("missing_arguments")
                RETURN _error_response(-32602, "Invalid params: missing arguments")

            tool_name = params["name"]
            tool_arguments = params["arguments"]

            # Step 3.2: Check tool exists
            IF tool_name NOT IN self._tools:
                logger.warning("tool_not_found", tool_name=tool_name)
                RETURN _error_response(
                    -32601,
                    f"Tool not found: {tool_name}",
                    {"available_tools": list(self._tools.keys())}
                )

            # Step 3.3: Execute tool
            tool = self._tools[tool_name]
            logger.info("executing_tool", tool_name=tool_name)

            TRY:
                result = AWAIT tool.execute(tool_arguments)
                logger.info("tool_execution_success", tool_name=tool_name)
                RETURN result

            CATCH Exception as tool_error:
                # Tool execution failed (should not happen - tools handle errors internally)
                logger.error(
                    "tool_execution_failed",
                    tool_name=tool_name,
                    error=str(tool_error),
                    exc_info=True
                )
                RETURN _error_response(
                    -32603,
                    f"Tool execution failed: {tool_name}"
                )

        ELIF method == "tools/list":
            # Step 3.4: Return tools list
            logger.info("listing_tools", count=len(self._tools))

            tools_list = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.input_schema
                }
                FOR tool IN self._tools.values()
            ]

            RETURN {"tools": tools_list}

        ELSE:
            # Step 3.5: Unknown method
            logger.warning("method_not_found", method=method)
            RETURN _error_response(-32601, f"Method not found: {method}")

    CATCH Exception as unexpected_error:
        # Step 4: Catch-all for unexpected errors
        logger.error(
            "unexpected_request_error",
            error_type=type(unexpected_error).__name__,
            error=str(unexpected_error),
            exc_info=True
        )
        RETURN _error_response(-32603, "Internal error occurred")

END FUNCTION
```

---

## Preconditions

### Required State

✅ **Server Initialization:**
- `MCPServer.__init__()` must have been called
- `self._tools` dictionary must be initialized
- `self._logger` must be initialized

✅ **Tools Registered:**
- At least one tool should be registered (for tools/call to work)
- No strict requirement (empty tool list is valid)

✅ **Request Format:**
- `request` parameter must be a Python dict
- Dict must be JSON-deserializable (no circular refs)

### Not Required

❌ **Server Running:**
- Can be called before or after `run()` starts
- Typically called after run() starts processing

❌ **Pre-validation:**
- No need to validate request format before calling
- Method handles all validation internally

---

## Postconditions

### On Success (tools/call)

✅ **Tool Executed:**
- Registered tool's execute() method was called
- Tool processed the arguments
- Tool returned MCP-formatted response

✅ **Response Valid:**
- Return dict conforms to MCP response schema
- Contains "content" array or "error" object
- Ready for serialization to JSON

✅ **Logging Complete:**
- Request logged with method and request_id
- Tool execution logged with tool name
- Success/failure logged appropriately

### On Success (tools/list)

✅ **Tools Listed:**
- All registered tools included in response
- Each tool has name, description, inputSchema
- Response ready for client consumption

### On Error

✅ **Error Response:**
- Return dict contains "error" object
- Error has proper JSON-RPC error code
- Error message is user-friendly

✅ **No Side Effects:**
- No state changes on error
- Tools not executed if validation fails
- Server remains operational

---

## Edge Cases & Handling

### Edge Case 1: Empty Request Dictionary

**Scenario:** Request is `{}`

**Input:**
```python
request = {}
```

**Processing:**
1. Extract method: `request.get("method")` → None
2. Check: `method IS None` → True
3. Return error: code -32600, "Invalid request: missing method field"

**Expected Behavior:**
```python
response = await server.handle_request({})
assert response["error"]["code"] == -32600
assert "missing method" in response["error"]["message"].lower()
```

**Test Scenario:**
```python
async def test_handle_request_empty_dict():
    server = MCPServer(core_engine=mock_engine)

    response = await server.handle_request({})

    assert "error" in response
    assert response["error"]["code"] == -32600
    assert "method" in response["error"]["message"].lower()
```

---

### Edge Case 2: Unknown Method

**Scenario:** Method not in supported list

**Input:**
```python
request = {"method": "unknown/action", "id": 1}
```

**Processing:**
1. Parse method: `"unknown/action"`
2. Check routing: not "tools/call" or "tools/list"
3. Return error: code -32601, "Method not found: unknown/action"

**Expected Behavior:**
```python
response = await server.handle_request({"method": "unknown/action"})
assert response["error"]["code"] == -32601
assert "unknown/action" in response["error"]["message"]
```

**Test Scenario:**
```python
async def test_handle_request_unknown_method():
    server = MCPServer(core_engine=mock_engine)

    response = await server.handle_request({"method": "foo/bar", "id": 1})

    assert response["error"]["code"] == -32601
    assert "method not found" in response["error"]["message"].lower()
```

---

### Edge Case 3: tools/call Without Tool Name

**Scenario:** tools/call params missing "name"

**Input:**
```python
request = {
    "method": "tools/call",
    "params": {"arguments": {"text": "test"}},
    "id": 1
}
```

**Processing:**
1. Method = "tools/call"
2. Check "name" in params → False
3. Return error: code -32602, "Invalid params: missing tool name"

**Expected Behavior:**
```python
response = await server.handle_request(request)
assert response["error"]["code"] == -32602
assert "missing tool name" in response["error"]["message"].lower()
```

**Test Scenario:**
```python
async def test_handle_request_tools_call_missing_name():
    server = MCPServer(core_engine=mock_engine)

    request = {
        "method": "tools/call",
        "params": {"arguments": {}},
        "id": 1
    }

    response = await server.handle_request(request)

    assert response["error"]["code"] == -32602
    assert "tool name" in response["error"]["message"].lower()
```

---

### Edge Case 4: Tool Not Found

**Scenario:** Requesting tool that isn't registered

**Input:**
```python
request = {
    "method": "tools/call",
    "params": {
        "name": "nonexistent_tool",
        "arguments": {}
    },
    "id": 1
}
```

**Processing:**
1. Extract tool_name: "nonexistent_tool"
2. Check: `tool_name in self._tools` → False
3. Return error: code -32601, "Tool not found: nonexistent_tool"

**Expected Behavior:**
```python
response = await server.handle_request(request)
assert response["error"]["code"] == -32601
assert "nonexistent_tool" in response["error"]["message"]
assert "available_tools" in response["error"].get("data", {})
```

**Test Scenario:**
```python
async def test_handle_request_tool_not_found():
    server = MCPServer(core_engine=mock_engine)
    server.register_tool(AddMemoryTool(mock_engine))

    request = {
        "method": "tools/call",
        "params": {"name": "fake_tool", "arguments": {}},
        "id": 1
    }

    response = await server.handle_request(request)

    assert response["error"]["code"] == -32601
    assert "fake_tool" in response["error"]["message"]
    assert "add_memory" in response["error"]["data"]["available_tools"]
```

---

### Edge Case 5: Tool Execution Raises Exception

**Scenario:** Tool's execute() method raises unexpected exception

**Input:**
```python
request = {
    "method": "tools/call",
    "params": {
        "name": "buggy_tool",
        "arguments": {}
    },
    "id": 1
}

# Mock tool that raises
mock_tool.execute = AsyncMock(side_effect=RuntimeError("Tool bug"))
```

**Processing:**
1. Tool execution: `await tool.execute(...)` raises RuntimeError
2. Catch in exception handler
3. Log error with traceback
4. Return error: code -32603, "Tool execution failed: buggy_tool"

**Expected Behavior:**
```python
response = await server.handle_request(request)
assert response["error"]["code"] == -32603
assert "tool execution failed" in response["error"]["message"].lower()
# Internal error details NOT exposed
assert "Tool bug" not in response["error"]["message"]
```

**Test Scenario:**
```python
async def test_handle_request_tool_execution_error():
    server = MCPServer(core_engine=mock_engine)

    buggy_tool = Mock()
    buggy_tool.name = "buggy_tool"
    buggy_tool.execute = AsyncMock(side_effect=RuntimeError("Internal bug"))

    server._tools["buggy_tool"] = buggy_tool

    request = {
        "method": "tools/call",
        "params": {"name": "buggy_tool", "arguments": {}},
        "id": 1
    }

    response = await server.handle_request(request)

    assert response["error"]["code"] == -32603
    assert "Internal bug" not in response["error"]["message"]  # Not leaked
```

---

### Edge Case 6: tools/list With No Tools Registered

**Scenario:** Server has zero tools registered

**Input:**
```python
request = {"method": "tools/list", "id": 1}
```

**Processing:**
1. Method = "tools/list"
2. Build list: `[...for tool in self._tools.values()]` → []
3. Return: `{"tools": []}`

**Expected Behavior:**
```python
response = await server.handle_request(request)
assert "tools" in response
assert response["tools"] == []
assert "error" not in response  # Not an error
```

**Test Scenario:**
```python
async def test_handle_request_tools_list_empty():
    server = MCPServer(core_engine=mock_engine)
    # No tools registered

    request = {"method": "tools/list", "id": 1}
    response = await server.handle_request(request)

    assert "tools" in response
    assert len(response["tools"]) == 0
    assert "error" not in response
```

---

### Edge Case 7: Request With No ID (Notification)

**Scenario:** Request is a notification (no "id" field)

**Input:**
```python
request = {
    "method": "tools/call",
    "params": {
        "name": "add_memory",
        "arguments": {"text": "test"}
    }
    # No "id" field
}
```

**Processing:**
1. Extract request_id: `request.get("id")` → None
2. Logger binds: `request_id="<no-id>"`
3. Process normally
4. Return response (client won't correlate, but valid)

**Expected Behavior:**
```python
response = await server.handle_request(request)
assert "content" in response or "error" in response
# Response returned even though no ID
```

**Test Scenario:**
```python
async def test_handle_request_notification_no_id():
    server = MCPServer(core_engine=mock_engine)
    server.register_tool(AddMemoryTool(mock_engine))

    request = {
        "method": "tools/call",
        "params": {
            "name": "add_memory",
            "arguments": {"text": "test"}
        }
        # No "id"
    }

    response = await server.handle_request(request)
    # Should process normally
    assert "content" in response or "error" in response
```

---

## Test Scenarios (Complete List)

### Happy Path Tests

**1. test_handle_request_tools_call_success**
- **Input:** Valid tools/call request for add_memory
- **Expected:** Tool executed, success response returned
- **Verifies:** Normal tool execution flow

**2. test_handle_request_tools_list_success**
- **Input:** Valid tools/list request
- **Expected:** List of all registered tools
- **Verifies:** Tools listing

**3. test_handle_request_multiple_tools**
- **Input:** Calls to different tools sequentially
- **Expected:** Each tool executed correctly
- **Verifies:** Tool routing to multiple tools

**4. test_handle_request_with_request_id**
- **Input:** Request with numeric ID
- **Expected:** Response can be correlated (ID logged)
- **Verifies:** Request ID handling

**5. test_handle_request_with_string_id**
- **Input:** Request with string ID (e.g., "req-123")
- **Expected:** Accepted and logged
- **Verifies:** String ID support

---

### Error Handling Tests

**6. test_handle_request_empty_dict**
- **Input:** Empty request `{}`
- **Expected:** Error -32600, missing method
- **Verifies:** Edge case 1

**7. test_handle_request_unknown_method**
- **Input:** Method "foo/bar"
- **Expected:** Error -32601, method not found
- **Verifies:** Edge case 2

**8. test_handle_request_tools_call_missing_name**
- **Input:** tools/call without "name" in params
- **Expected:** Error -32602, missing tool name
- **Verifies:** Edge case 3

**9. test_handle_request_tools_call_missing_arguments**
- **Input:** tools/call without "arguments"
- **Expected:** Error -32602, missing arguments
- **Verifies:** Parameter validation

**10. test_handle_request_tool_not_found**
- **Input:** tools/call for unregistered tool
- **Expected:** Error -32601, tool not found + available tools list
- **Verifies:** Edge case 4

**11. test_handle_request_tool_execution_error**
- **Input:** Tool that raises exception
- **Expected:** Error -32603, internal error, no leak
- **Verifies:** Edge case 5

---

### Boundary Tests

**12. test_handle_request_tools_list_empty**
- **Input:** tools/list when no tools registered
- **Expected:** Empty tools array (not error)
- **Verifies:** Edge case 6

**13. test_handle_request_notification_no_id**
- **Input:** Request without "id" field
- **Expected:** Processed normally
- **Verifies:** Edge case 7

---

### Integration Tests

**14. test_handle_request_calls_tool_execute**
- **Verifies:** Tool's execute() method actually called
- **Mocks:** Tool execute method
- **Asserts:** Called with correct arguments

**15. test_handle_request_logs_request**
- **Verifies:** Logger records request details
- **Checks:** request_id, method logged

**16. test_handle_request_logs_tool_execution**
- **Verifies:** Tool execution logged
- **Checks:** tool_name, success/failure logged

---

### Performance Tests

**17. test_handle_request_performance_overhead**
- **Input:** Simple tools/list request
- **Expected:** < 5ms total time
- **Verifies:** Routing overhead minimal

---

## Performance Requirements

### Latency Targets

**Request Handling Overhead:**
- Parse and validate: < 1ms
- Routing decision: < 0.1ms (dict lookup)
- Response formatting: < 1ms
- **Total overhead (excluding tool execution): < 5ms**

**End-to-End (including tool):**
- tools/list: < 10ms total
- tools/call (simple): < 100ms total (depends on tool)
- tools/call (complex): Variable (depends on tool processing)

### Throughput

**Sequential Processing:**
- MCP stdio is inherently sequential (one request at a time)
- No concurrency control needed
- Throughput limited by tool execution time

---

## Security Considerations

### Input Validation

✅ **All request fields validated:**
- Method existence checked
- Params structure validated
- Tool name verified against registry

✅ **Injection Prevention:**
- No string concatenation of user input
- Tool names validated (must be registered)
- Arguments passed to tools for validation

### Error Message Safety

✅ **Safe to expose:**
- JSON-RPC error codes (standard)
- Method/tool not found errors
- Available tools list (public info)

❌ **Never expose:**
- Tool execution exception details (internal bugs)
- Stack traces
- Internal state information

---

## Related Functions

### Calls

**1. `tool.execute(arguments)` (Tool protocol)**
- **Purpose:** Execute registered tool
- **When:** For tools/call requests

**2. `self._error_response(code, message, data)` (helper)**
- **Purpose:** Format JSON-RPC error response
- **When:** Any error occurs

**3. `self._logger.info/warning/error()` (structlog)**
- **Purpose:** Log request processing
- **When:** Throughout request handling

### Called By

**1. MCP SDK server request handler**
- **Purpose:** Route incoming stdio messages
- **When:** Client sends JSON-RPC request

---

## Implementation Notes

### Dependencies

**External Libraries:**
- `mcp.server` - MCP SDK server
- `structlog` - Logging

**Internal Dependencies:**
- Registered tools (via `self._tools`)

### Known Limitations

**1. Sequential Processing Only:**
- Cannot handle concurrent requests (stdio limitation)
- Parallel tool execution not supported

**2. No Request Batching:**
- Each request processed individually
- No JSON-RPC batch support

### Future Enhancements

**1. Support More Methods:**
- `resources/list`, `resources/read`
- `prompts/list`, `prompts/get`
- Custom methods

**2. Request Middleware:**
- Request logging middleware
- Rate limiting
- Authentication

---

## References

### Component Spec
- [MCPServer Component Specification](../level2/mcp_server_component.md)

### Module Spec
- [Zapomni MCP Module Specification](../level1/zapomni_mcp_module.md)

### Related Function Specs
- `MCPServer.register_tool()` (Level 3) - Tool registration
- `MCPServer.run()` (Level 3) - Server execution loop
- `AddMemoryTool.execute()` (Level 3) - Example tool handler

### External Documentation
- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [JSON-RPC 2.0 Specification](https://www.jsonrpc.org/specification)

---

## Document Status

**Version:** 1.0
**Created:** 2025-11-23
**Author:** Goncharenko Anton aka alienxs2
**License:** MIT License
**Status:** Draft - Ready for Review

**Estimated Implementation Effort:** 2-3 hours
**Lines of Code (Estimated):** ~80 lines
**Test Coverage Target:** 95%+ (17 test scenarios defined)
**Test File:** `tests/unit/mcp/test_mcp_server_handle_request.py`
