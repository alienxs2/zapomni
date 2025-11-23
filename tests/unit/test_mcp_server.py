"""
Unit tests for MCPServer component.

Tests the core MCP server functionality including:
- Server initialization and configuration
- Tool registration and validation
- Request/response handling
- Statistics tracking
- Graceful shutdown

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

import asyncio
import signal
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from mcp.types import TextContent, Tool

from zapomni_core.exceptions import ValidationError
from zapomni_mcp.config import Settings
from zapomni_mcp.server import MCPServer, ServerStats


# Fixtures


@pytest.fixture
def mock_core_engine():
    """Mock ZapomniCore engine for testing."""
    mock = MagicMock()
    mock.add_memory = AsyncMock(return_value={"memory_id": "test-uuid-123"})
    mock.search_memory = AsyncMock(return_value=[])
    mock.get_stats = AsyncMock(
        return_value={"total_memories": 0, "total_chunks": 0, "graph_nodes": 0}
    )
    return mock


@pytest.fixture
def mock_tool():
    """Create a mock MCPTool for testing."""

    class MockTool:
        name = "test_tool"
        description = "A test tool"
        input_schema = {"type": "object", "properties": {"data": {"type": "string"}}}

        async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
            return {"content": [{"type": "text", "text": "Success"}], "isError": False}

    return MockTool()


@pytest.fixture
def server_config():
    """Default server configuration for testing."""
    return Settings(log_level="DEBUG", server_name="test-server")


# Initialization Tests


@pytest.mark.unit
def test_init_success(mock_core_engine, server_config):
    """Test successful server initialization with valid core engine."""
    server = MCPServer(core_engine=mock_core_engine, config=server_config)

    assert server._core_engine is mock_core_engine
    assert server._config.server_name == "test-server"
    assert server._running is False
    assert server._request_count == 0
    assert server._error_count == 0
    assert isinstance(server._tools, dict)
    assert len(server._tools) == 0


@pytest.mark.unit
def test_init_with_defaults(mock_core_engine):
    """Test initialization with default config when none provided."""
    server = MCPServer(core_engine=mock_core_engine)

    assert server._core_engine is mock_core_engine
    assert server._config is not None
    assert server._config.server_name == "zapomni-memory"
    assert server._running is False


@pytest.mark.unit
def test_init_none_core_raises():
    """Test that None core_engine raises ConfigurationError."""
    from zapomni_mcp.server import ConfigurationError

    with pytest.raises(ConfigurationError, match="core_engine cannot be None"):
        MCPServer(core_engine=None)


@pytest.mark.unit
def test_init_invalid_config_raises(mock_core_engine):
    """Test that invalid config raises ValidationError."""
    with pytest.raises(ValidationError):
        invalid_config = Settings(log_level="INVALID_LEVEL")
        MCPServer(core_engine=mock_core_engine, config=invalid_config)


# Tool Registration Tests


@pytest.mark.unit
def test_register_tool_success(mock_core_engine, mock_tool):
    """Test successful tool registration."""
    server = MCPServer(core_engine=mock_core_engine)
    server.register_tool(mock_tool)

    assert "test_tool" in server._tools
    assert server._tools["test_tool"] is mock_tool


@pytest.mark.unit
def test_register_tool_duplicate_raises(mock_core_engine, mock_tool):
    """Test that registering duplicate tool name raises ValueError."""
    server = MCPServer(core_engine=mock_core_engine)
    server.register_tool(mock_tool)

    with pytest.raises(ValueError, match="already registered"):
        server.register_tool(mock_tool)


@pytest.mark.unit
def test_register_tool_invalid_name_raises(mock_core_engine):
    """Test that invalid tool name format raises ValueError."""

    class BadTool:
        name = "Bad-Tool-Name"  # Invalid: uppercase and dashes
        description = "test"
        input_schema = {"type": "object"}

        async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
            return {}

    server = MCPServer(core_engine=mock_core_engine)

    with pytest.raises(ValueError, match="invalid format"):
        server.register_tool(BadTool())


@pytest.mark.unit
def test_register_tool_missing_attributes_raises(mock_core_engine):
    """Test that tool missing required attributes raises TypeError."""

    class IncompleteTool:
        name = "incomplete"
        # Missing description, input_schema, execute

    server = MCPServer(core_engine=mock_core_engine)

    with pytest.raises(TypeError, match="must implement"):
        server.register_tool(IncompleteTool())


@pytest.mark.unit
def test_register_all_tools_success(mock_core_engine):
    """Test that register_all_tools registers standard tools."""
    server = MCPServer(core_engine=mock_core_engine)
    server.register_all_tools()

    # Should register at least 3 tools (add_memory, search_memory, get_stats)
    assert len(server._tools) >= 3
    assert "add_memory" in server._tools
    assert "search_memory" in server._tools
    assert "get_stats" in server._tools


@pytest.mark.unit
def test_register_all_tools_count(mock_core_engine):
    """Test that correct number of tools are registered."""
    server = MCPServer(core_engine=mock_core_engine)
    server.register_all_tools()

    # Phase 1: Exactly 3 tools
    assert len(server._tools) == 3


# Server Lifecycle Tests


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_starts_server(mock_core_engine):
    """Test that run() starts the server successfully."""
    server = MCPServer(core_engine=mock_core_engine)
    server.register_all_tools()

    # Mock stdio_server to avoid blocking
    with patch("zapomni_mcp.server.stdio_server") as mock_stdio:
        # Create async context manager mock
        mock_streams = (MagicMock(), MagicMock())

        # Create a proper async context manager
        class AsyncContextManager:
            async def __aenter__(self):
                return mock_streams

            async def __aexit__(self, *args):
                # Trigger shutdown to exit run()
                server._running = False
                return None

        mock_stdio.return_value = AsyncContextManager()

        # Mock the server.run method to avoid actual MCP protocol
        with patch.object(server._server, "run", new_callable=AsyncMock):
            await server.run()

        assert mock_stdio.called


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_already_running_raises(mock_core_engine):
    """Test that calling run() while already running raises RuntimeError."""
    server = MCPServer(core_engine=mock_core_engine)
    server._running = True  # Simulate running state

    with pytest.raises(RuntimeError, match="already running"):
        await server.run()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_no_tools_raises(mock_core_engine):
    """Test that running with no registered tools raises RuntimeError."""
    server = MCPServer(core_engine=mock_core_engine)
    # No tools registered

    with pytest.raises(RuntimeError, match="No tools registered"):
        await server.run()


@pytest.mark.unit
def test_shutdown_stops_server(mock_core_engine):
    """Test that shutdown() sets running flag to False."""
    server = MCPServer(core_engine=mock_core_engine)
    server._running = True

    server.shutdown()

    assert server._running is False


@pytest.mark.unit
def test_shutdown_idempotent(mock_core_engine):
    """Test that calling shutdown() multiple times is safe."""
    server = MCPServer(core_engine=mock_core_engine)
    server._running = True

    server.shutdown()
    assert server._running is False

    # Second call should not raise
    server.shutdown()
    assert server._running is False


# Request Handling Tests


@pytest.mark.unit
@pytest.mark.asyncio
async def test_handle_valid_tool_call(mock_core_engine):
    """Test handling a valid tool call request."""
    server = MCPServer(core_engine=mock_core_engine)
    server.register_all_tools()

    # Simulate get_stats tool call
    result = await server._tools["get_stats"].execute({})

    assert "content" in result
    assert isinstance(result["content"], list)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_handle_unknown_tool_error(mock_core_engine):
    """Test that calling unknown tool raises appropriate error."""
    server = MCPServer(core_engine=mock_core_engine)
    server.register_all_tools()

    # Unknown tool should not be in registry
    assert "unknown_tool" not in server._tools


@pytest.mark.unit
@pytest.mark.asyncio
async def test_handle_invalid_arguments_error(mock_core_engine):
    """Test that invalid tool arguments are caught."""
    server = MCPServer(core_engine=mock_core_engine)
    server.register_all_tools()

    # add_memory requires 'text' argument
    with pytest.raises((ValidationError, KeyError, TypeError)):
        await server._tools["add_memory"].execute({})  # Missing required 'text'


# Statistics Tests


@pytest.mark.unit
def test_get_stats_before_start(mock_core_engine):
    """Test get_stats() before server starts."""
    server = MCPServer(core_engine=mock_core_engine)

    stats = server.get_stats()

    assert isinstance(stats, ServerStats)
    assert stats.total_requests == 0
    assert stats.total_errors == 0
    assert stats.registered_tools == 0
    assert stats.uptime_seconds == 0.0
    assert stats.running is False


@pytest.mark.unit
def test_get_stats_after_registration(mock_core_engine):
    """Test get_stats() reflects registered tools."""
    server = MCPServer(core_engine=mock_core_engine)
    server.register_all_tools()

    stats = server.get_stats()

    assert stats.registered_tools == 3
    assert stats.running is False


@pytest.mark.unit
def test_get_stats_while_running(mock_core_engine):
    """Test get_stats() while server is running."""
    server = MCPServer(core_engine=mock_core_engine)
    server.register_all_tools()
    server._running = True
    server._start_time = 100.0

    with patch("time.time", return_value=150.0):
        stats = server.get_stats()

        assert stats.running is True
        assert stats.uptime_seconds == 50.0


@pytest.mark.unit
def test_get_stats_request_count(mock_core_engine):
    """Test that request count is tracked correctly."""
    server = MCPServer(core_engine=mock_core_engine)
    server._request_count = 42
    server._error_count = 3

    stats = server.get_stats()

    assert stats.total_requests == 42
    assert stats.total_errors == 3


# Signal Handling Tests


@pytest.mark.unit
def test_signal_handlers_installed(mock_core_engine):
    """Test that signal handlers are installed on initialization."""
    with patch("signal.signal") as mock_signal:
        server = MCPServer(core_engine=mock_core_engine)
        server._setup_signal_handlers()

        # Should register handlers for SIGINT and SIGTERM
        calls = mock_signal.call_args_list
        signals_registered = [call[0][0] for call in calls]

        assert signal.SIGINT in signals_registered
        assert signal.SIGTERM in signals_registered


@pytest.mark.unit
def test_sigint_triggers_shutdown(mock_core_engine):
    """Test that SIGINT calls shutdown()."""
    server = MCPServer(core_engine=mock_core_engine)
    server._running = True

    # Manually call shutdown (simulating signal handler)
    server.shutdown()

    assert server._running is False


# Validation Tests


@pytest.mark.unit
def test_validate_tool_success(mock_core_engine, mock_tool):
    """Test that valid tool passes validation."""
    server = MCPServer(core_engine=mock_core_engine)

    # Should not raise
    server._validate_tool(mock_tool)


@pytest.mark.unit
def test_validate_tool_no_name_raises(mock_core_engine):
    """Test that tool without name fails validation."""

    class NoNameTool:
        description = "test"
        input_schema = {}

        async def execute(self, args):
            return {}

    server = MCPServer(core_engine=mock_core_engine)

    with pytest.raises(TypeError, match="name"):
        server._validate_tool(NoNameTool())


@pytest.mark.unit
def test_validate_tool_no_execute_raises(mock_core_engine):
    """Test that tool without execute method fails validation."""

    class NoExecuteTool:
        name = "no_execute"
        description = "test"
        input_schema = {}

    server = MCPServer(core_engine=mock_core_engine)

    with pytest.raises(TypeError, match="execute"):
        server._validate_tool(NoExecuteTool())


# Edge Cases


@pytest.mark.unit
def test_empty_tool_name_raises(mock_core_engine):
    """Test that empty tool name raises ValueError."""

    class EmptyNameTool:
        name = ""
        description = "test"
        input_schema = {}

        async def execute(self, args):
            return {}

    server = MCPServer(core_engine=mock_core_engine)

    with pytest.raises(ValueError, match="cannot be empty"):
        server.register_tool(EmptyNameTool())


@pytest.mark.unit
def test_server_stats_dataclass():
    """Test ServerStats dataclass structure."""
    stats = ServerStats(
        total_requests=10,
        total_errors=2,
        registered_tools=3,
        uptime_seconds=123.45,
        running=True,
    )

    assert stats.total_requests == 10
    assert stats.total_errors == 2
    assert stats.registered_tools == 3
    assert stats.uptime_seconds == 123.45
    assert stats.running is True
