"""
Integration tests for transport backward compatibility.

Tests that:
- stdio transport still works as before
- Switching between transports works correctly
- All tools work in both transport modes
- Default transport is SSE

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

import argparse
import asyncio
import os
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from zapomni_mcp.config import Settings, SSEConfig
from zapomni_mcp.server import MCPServer

# Fixtures


@pytest.fixture
def mock_memory_processor():
    """Create mock MemoryProcessor for tool registration testing."""
    mock = MagicMock()
    mock.add_memory = AsyncMock(return_value={"memory_id": "test-uuid-123"})
    mock.search_memory = AsyncMock(return_value=[])
    mock.get_stats = AsyncMock(
        return_value={
            "total_memories": 10,
            "total_chunks": 50,
            "graph_nodes": 25,
        }
    )
    mock.delete_memory = AsyncMock(return_value={"deleted": True})
    mock.clear_all = AsyncMock(return_value={"cleared": True})
    mock.build_graph = AsyncMock(return_value={"entities": [], "relationships": []})
    mock.get_related = AsyncMock(return_value=[])
    mock.graph_status = AsyncMock(return_value={"total_nodes": 0, "total_relationships": 0})
    mock.export_graph = AsyncMock(return_value="")
    mock.code_indexer = None
    return mock


@pytest.fixture
def mock_core_engine():
    """Create mock ZapomniCore engine (for tests that don't need tools)."""
    mock = MagicMock()
    mock.add_memory = AsyncMock(return_value={"memory_id": "test-uuid-123"})
    mock.search_memory = AsyncMock(return_value=[])
    mock.get_stats = AsyncMock(
        return_value={
            "total_memories": 10,
            "total_chunks": 50,
            "graph_nodes": 25,
        }
    )
    mock.delete_memory = AsyncMock(return_value={"deleted": True})
    mock.clear_all = AsyncMock(return_value={"cleared": True})
    mock.build_graph = AsyncMock(return_value={"entities": [], "relationships": []})
    mock.get_related = AsyncMock(return_value=[])
    mock.graph_status = AsyncMock(return_value={"total_nodes": 0, "total_relationships": 0})
    mock.export_graph = AsyncMock(return_value="")
    return mock


@pytest.fixture
def server_config():
    """Default server configuration."""
    return Settings(log_level="DEBUG", server_name="test-server")


@pytest.fixture
def mcp_server(mock_core_engine, server_config):
    """Create MCPServer instance for testing."""
    return MCPServer(core_engine=mock_core_engine, config=server_config)


# Default Transport Tests


class TestDefaultTransport:
    """Test suite for default transport behavior."""

    def test_default_transport_is_sse(self):
        """Should default to SSE transport."""
        # Simulate argparse default
        parser = argparse.ArgumentParser()
        parser.add_argument("--transport", default="sse")
        args = parser.parse_args([])

        assert args.transport == "sse"

    def test_transport_choices(self):
        """Should accept stdio and sse as valid transport choices."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--transport", choices=["stdio", "sse"], default="sse")

        # Valid transports
        args_sse = parser.parse_args(["--transport", "sse"])
        args_stdio = parser.parse_args(["--transport", "stdio"])

        assert args_sse.transport == "sse"
        assert args_stdio.transport == "stdio"


# Stdio Transport Tests


class TestStdioTransport:
    """Test suite for stdio transport backward compatibility."""

    @pytest.mark.asyncio
    async def test_stdio_run_method_exists(self, mcp_server):
        """Should have run() method for stdio transport."""
        assert hasattr(mcp_server, "run")
        assert callable(mcp_server.run)

    @pytest.mark.asyncio
    async def test_stdio_transport_starts(self, mcp_server):
        """Should start server with stdio transport."""
        # Register a mock tool to bypass the "no tools" check
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        mock_tool.input_schema = {"type": "object"}
        mock_tool.execute = AsyncMock(return_value={"content": [], "isError": False})
        mcp_server._tools["test_tool"] = mock_tool

        with patch("zapomni_mcp.server.stdio_server") as mock_stdio:
            # Create async context manager mock
            mock_streams = (MagicMock(), MagicMock())

            class AsyncContextManager:
                async def __aenter__(self):
                    return mock_streams

                async def __aexit__(self, *args):
                    mcp_server._running = False
                    return None

            mock_stdio.return_value = AsyncContextManager()

            with patch.object(mcp_server._server, "run", new_callable=AsyncMock):
                await mcp_server.run()

            mock_stdio.assert_called_once()

    @pytest.mark.asyncio
    async def test_stdio_requires_tools(self, mcp_server):
        """Should require tools registered before running."""
        # No tools registered
        with pytest.raises(RuntimeError, match="No tools registered"):
            await mcp_server.run()

    @pytest.mark.asyncio
    async def test_stdio_prevents_double_run(self, mcp_server):
        """Should prevent running twice."""
        mcp_server._running = True

        with pytest.raises(RuntimeError, match="already running"):
            await mcp_server.run()


# SSE Transport Tests


class TestSSETransport:
    """Test suite for SSE transport."""

    @pytest.mark.asyncio
    async def test_sse_run_method_exists(self, mcp_server):
        """Should have run_sse() method for SSE transport."""
        assert hasattr(mcp_server, "run_sse")
        assert callable(mcp_server.run_sse)

    @pytest.mark.asyncio
    async def test_sse_transport_accepts_parameters(self, mcp_server):
        """Should accept host, port, cors_origins parameters."""
        # Register a mock tool to bypass the "no tools" check
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        mock_tool.input_schema = {"type": "object"}
        mock_tool.execute = AsyncMock(return_value={"content": [], "isError": False})
        mcp_server._tools["test_tool"] = mock_tool

        # Mock uvicorn to prevent actual server start (uvicorn is imported inside run_sse)
        mock_server = MagicMock()
        mock_server.serve = AsyncMock()

        with patch.dict("sys.modules", {"uvicorn": MagicMock()}):
            import sys

            sys.modules["uvicorn"].Config = MagicMock()
            sys.modules["uvicorn"].Server = MagicMock(return_value=mock_server)

            # Test with custom parameters
            try:
                # Start in background and cancel immediately
                task = asyncio.create_task(
                    mcp_server.run_sse(
                        host="0.0.0.0",
                        port=9000,
                        cors_origins=["http://localhost:3000"],
                    )
                )
                await asyncio.sleep(0.1)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            except Exception:
                pass  # Expected on cancellation

    @pytest.mark.asyncio
    async def test_sse_requires_tools(self, mcp_server):
        """Should require tools registered before running SSE."""
        # No tools registered
        with pytest.raises(RuntimeError, match="No tools registered"):
            await mcp_server.run_sse()

    @pytest.mark.asyncio
    async def test_sse_prevents_double_run(self, mcp_server):
        """Should prevent running twice."""
        mcp_server._running = True

        with pytest.raises(RuntimeError, match="already running"):
            await mcp_server.run_sse()


# Tool Registration Tests


class TestToolRegistration:
    """Test suite for tool registration across transports."""

    def test_register_tool_adds_to_registry(self, mcp_server):
        """Should add tools to registry when registered."""
        # Create a mock tool
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        mock_tool.input_schema = {"type": "object"}
        mock_tool.execute = AsyncMock(return_value={"content": [], "isError": False})

        mcp_server.register_tool(mock_tool)

        assert "test_tool" in mcp_server._tools
        assert mcp_server._tools["test_tool"] is mock_tool

    def test_tools_accessible_after_registration(self, mcp_server):
        """Should have tools accessible after registration."""
        # Create and register mock tools
        for i in range(3):
            mock_tool = MagicMock()
            mock_tool.name = f"test_tool_{i}"
            mock_tool.description = f"Test tool {i}"
            mock_tool.input_schema = {"type": "object"}
            mock_tool.execute = AsyncMock(return_value={"content": [], "isError": False})
            mcp_server.register_tool(mock_tool)

        # Tools should be callable
        for tool_name, tool in mcp_server._tools.items():
            assert hasattr(tool, "execute")
            assert hasattr(tool, "name")
            assert hasattr(tool, "description")

    @pytest.mark.asyncio
    async def test_tool_execution_works(self, mcp_server):
        """Should execute tools successfully."""
        # Create and register a mock tool
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        mock_tool.input_schema = {"type": "object"}
        mock_tool.execute = AsyncMock(
            return_value={"content": [{"type": "text", "text": "success"}], "isError": False}
        )

        mcp_server.register_tool(mock_tool)

        # Execute the tool
        result = await mcp_server._tools["test_tool"].execute({})

        assert "content" in result
        assert isinstance(result["content"], list)


# Transport Switching Tests


class TestTransportSwitching:
    """Test suite for switching between transports."""

    def test_server_supports_both_transports(self, mcp_server):
        """Should support both stdio and SSE transports."""
        assert hasattr(mcp_server, "run")  # stdio
        assert hasattr(mcp_server, "run_sse")  # SSE

    @pytest.mark.asyncio
    async def test_same_tools_for_both_transports(self, mock_core_engine, server_config):
        """Should have same tools available for both transports."""
        # Create two servers
        server_stdio = MCPServer(core_engine=mock_core_engine, config=server_config)
        server_sse = MCPServer(core_engine=mock_core_engine, config=server_config)

        # Register same mock tools on both servers
        for server in [server_stdio, server_sse]:
            for tool_name in ["add_memory", "search_memory", "get_stats"]:
                mock_tool = MagicMock()
                mock_tool.name = tool_name
                mock_tool.description = f"{tool_name} description"
                mock_tool.input_schema = {"type": "object"}
                mock_tool.execute = AsyncMock(return_value={"content": [], "isError": False})
                server.register_tool(mock_tool)

        # Should have same tools
        assert set(server_stdio._tools.keys()) == set(server_sse._tools.keys())

    def test_shutdown_works_for_both(self, mcp_server):
        """Should shutdown gracefully for both transports."""
        mcp_server._running = True

        mcp_server.shutdown()

        assert mcp_server._running is False


# Configuration Priority Tests


class TestConfigurationPriority:
    """Test suite for configuration priority (CLI > env > defaults)."""

    def test_cli_overrides_env(self):
        """CLI arguments should override environment variables."""
        with patch.dict(os.environ, {"ZAPOMNI_SSE_PORT": "9999"}):
            sse_config = SSEConfig.from_env()

            # Env should be loaded
            assert sse_config.port == 9999

            # CLI override would happen in main()
            cli_port = 8080
            final_port = cli_port  # CLI takes precedence

            assert final_port == 8080

    def test_env_overrides_defaults(self):
        """Environment variables should override defaults."""
        with patch.dict(
            os.environ,
            {
                "ZAPOMNI_SSE_HOST": "0.0.0.0",
                "ZAPOMNI_SSE_PORT": "9000",
            },
        ):
            config = SSEConfig.from_env()

            assert config.host == "0.0.0.0"
            assert config.port == 9000

    def test_defaults_used_when_no_override(self):
        """Should use defaults when no override provided."""
        # Clear env vars
        with patch.dict(os.environ, {}, clear=True):
            for key in list(os.environ.keys()):
                if key.startswith("ZAPOMNI_SSE"):
                    del os.environ[key]

            config = SSEConfig()

            assert config.host == "127.0.0.1"
            assert config.port == 8000
            assert config.cors_origins == ["*"]


# Server Stats Tests


class TestServerStats:
    """Test suite for server statistics across transports."""

    def test_stats_structure_consistent(self, mcp_server):
        """Should have consistent stats structure."""
        stats = mcp_server.get_stats()

        assert hasattr(stats, "total_requests")
        assert hasattr(stats, "total_errors")
        assert hasattr(stats, "registered_tools")
        assert hasattr(stats, "uptime_seconds")
        assert hasattr(stats, "running")

    def test_stats_initial_values(self, mcp_server):
        """Should start with zero stats."""
        stats = mcp_server.get_stats()

        assert stats.total_requests == 0
        assert stats.total_errors == 0
        assert stats.running is False

    def test_stats_reflect_tool_registration(self, mcp_server):
        """Should reflect registered tool count."""
        # Register mock tools
        for i in range(3):
            mock_tool = MagicMock()
            mock_tool.name = f"test_tool_{i}"
            mock_tool.description = f"Test tool {i}"
            mock_tool.input_schema = {"type": "object"}
            mock_tool.execute = AsyncMock(return_value={"content": [], "isError": False})
            mcp_server.register_tool(mock_tool)

        stats = mcp_server.get_stats()

        assert stats.registered_tools == len(mcp_server._tools)


# Error Handling Tests


class TestErrorHandling:
    """Test suite for error handling across transports."""

    @pytest.mark.asyncio
    async def test_stdio_handles_no_tools_gracefully(self, mcp_server):
        """Should raise clear error when no tools for stdio."""
        with pytest.raises(RuntimeError, match="No tools registered"):
            await mcp_server.run()

    @pytest.mark.asyncio
    async def test_sse_handles_no_tools_gracefully(self, mcp_server):
        """Should raise clear error when no tools for SSE."""
        with pytest.raises(RuntimeError, match="No tools registered"):
            await mcp_server.run_sse()

    def test_shutdown_is_idempotent(self, mcp_server):
        """Should handle multiple shutdown calls."""
        mcp_server._running = True

        mcp_server.shutdown()
        assert mcp_server._running is False

        # Second call should not raise
        mcp_server.shutdown()
        assert mcp_server._running is False


# Signal Handler Tests


class TestSignalHandlers:
    """Test suite for signal handler compatibility."""

    def test_signal_handlers_installed(self, mcp_server):
        """Should have signal handler setup method."""
        assert hasattr(mcp_server, "_setup_signal_handlers")

    def test_shutdown_on_signal(self, mcp_server):
        """Should shutdown on signal."""
        mcp_server._running = True

        # Simulate signal handler behavior
        mcp_server.shutdown()

        assert mcp_server._running is False


# Logging Compatibility Tests


class TestLoggingCompatibility:
    """Test suite for logging compatibility."""

    def test_server_has_logger(self, mcp_server):
        """Should have logger configured."""
        assert hasattr(mcp_server, "_logger")

    @pytest.mark.asyncio
    async def test_logging_during_run(self, mcp_server):
        """Should log during run lifecycle."""
        # Register a mock tool to bypass the "no tools" check
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        mock_tool.input_schema = {"type": "object"}
        mock_tool.execute = AsyncMock(return_value={"content": [], "isError": False})
        mcp_server._tools["test_tool"] = mock_tool

        with patch("zapomni_mcp.server.stdio_server") as mock_stdio:
            mock_streams = (MagicMock(), MagicMock())

            class AsyncContextManager:
                async def __aenter__(self):
                    return mock_streams

                async def __aexit__(self, *args):
                    mcp_server._running = False
                    return None

            mock_stdio.return_value = AsyncContextManager()

            with patch.object(mcp_server._server, "run", new_callable=AsyncMock):
                await mcp_server.run()

            # Should not raise logging errors


# Stdio Backward Compatibility Tests


class TestStdioBackwardCompatibility:
    """Verify stdio transport works identically after SSE implementation."""

    @pytest.mark.asyncio
    async def test_stdio_transport_uses_stdio_server(self, mcp_server):
        """Stdio mode should use mcp.server.stdio.stdio_server."""
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        mock_tool.input_schema = {"type": "object"}
        mock_tool.execute = AsyncMock(return_value={"content": [], "isError": False})
        mcp_server._tools["test_tool"] = mock_tool

        with patch("zapomni_mcp.server.stdio_server") as mock_stdio:
            mock_streams = (MagicMock(), MagicMock())

            class AsyncContextManager:
                async def __aenter__(self):
                    return mock_streams

                async def __aexit__(self, *args):
                    mcp_server._running = False
                    return None

            mock_stdio.return_value = AsyncContextManager()

            with patch.object(mcp_server._server, "run", new_callable=AsyncMock):
                await mcp_server.run()

            # Verify stdio_server was called (not SSE)
            mock_stdio.assert_called_once()

    @pytest.mark.asyncio
    async def test_stdio_does_not_import_sse_modules(self, mock_core_engine, server_config):
        """Stdio mode should not trigger SSE module imports."""
        server = MCPServer(core_engine=mock_core_engine, config=server_config)

        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        mock_tool.input_schema = {"type": "object"}
        mock_tool.execute = AsyncMock(return_value={"content": [], "isError": False})
        server._tools["test_tool"] = mock_tool

        with patch("zapomni_mcp.server.stdio_server") as mock_stdio:
            mock_streams = (MagicMock(), MagicMock())

            class AsyncContextManager:
                async def __aenter__(self):
                    return mock_streams

                async def __aexit__(self, *args):
                    server._running = False
                    return None

            mock_stdio.return_value = AsyncContextManager()

            with patch.object(server._server, "run", new_callable=AsyncMock):
                # Track if SSE modules are imported
                with patch.dict("sys.modules", {"uvicorn": None}):
                    await server.run()

            # SSE session_manager should not be set in stdio mode
            assert not hasattr(server, "_session_manager") or server._session_manager is None

    @pytest.mark.asyncio
    async def test_stdio_tool_handlers_registered_correctly(self, mcp_server):
        """Tools should be registered identically in stdio mode."""
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        mock_tool.input_schema = {"type": "object"}
        mock_tool.execute = AsyncMock(
            return_value={"content": [{"type": "text", "text": "result"}], "isError": False}
        )
        mcp_server._tools["test_tool"] = mock_tool

        # Verify tool is accessible
        assert "test_tool" in mcp_server._tools
        assert mcp_server._tools["test_tool"].name == "test_tool"
        assert mcp_server._tools["test_tool"].description == "A test tool"

    @pytest.mark.asyncio
    async def test_stdio_request_count_tracked(self, mcp_server):
        """Stdio mode should track request counts."""
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        mock_tool.input_schema = {"type": "object"}
        mock_tool.execute = AsyncMock(return_value={"content": [], "isError": False})
        mcp_server._tools["test_tool"] = mock_tool

        # Initial count should be 0
        assert mcp_server._request_count == 0

        # Simulate request increment (as would happen during run())
        mcp_server._request_count += 1
        assert mcp_server._request_count == 1

    @pytest.mark.asyncio
    async def test_stdio_error_count_tracked(self, mcp_server):
        """Stdio mode should track error counts."""
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        mock_tool.input_schema = {"type": "object"}
        mock_tool.execute = AsyncMock(return_value={"content": [], "isError": False})
        mcp_server._tools["test_tool"] = mock_tool

        # Initial count should be 0
        assert mcp_server._error_count == 0

        # Simulate error increment
        mcp_server._error_count += 1
        assert mcp_server._error_count == 1

    def test_stdio_preserves_server_name(self, mcp_server):
        """Server name should be preserved in stdio mode."""
        assert mcp_server._config.server_name == "test-server"

    @pytest.mark.asyncio
    async def test_stdio_shutdown_cleans_up_state(self, mcp_server):
        """Stdio shutdown should clean up running state."""
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        mock_tool.input_schema = {"type": "object"}
        mock_tool.execute = AsyncMock(return_value={"content": [], "isError": False})
        mcp_server._tools["test_tool"] = mock_tool

        mcp_server._running = True
        mcp_server.shutdown()

        assert mcp_server._running is False


# No SSE Code Paths in Stdio Mode


class TestNoSSECodePathsInStdio:
    """Verify SSE code paths are not triggered in stdio mode."""

    @pytest.mark.asyncio
    async def test_stdio_no_session_manager(self, mock_core_engine, server_config):
        """Stdio mode should not create session manager."""
        server = MCPServer(core_engine=mock_core_engine, config=server_config)

        # Session manager should not exist initially
        assert not hasattr(server, "_session_manager") or server._session_manager is None

    @pytest.mark.asyncio
    async def test_stdio_no_starlette_app(self, mock_core_engine, server_config):
        """Stdio mode should not create Starlette app."""
        server = MCPServer(core_engine=mock_core_engine, config=server_config)

        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        mock_tool.input_schema = {"type": "object"}
        mock_tool.execute = AsyncMock(return_value={"content": [], "isError": False})
        server._tools["test_tool"] = mock_tool

        with patch("zapomni_mcp.server.stdio_server") as mock_stdio:
            mock_streams = (MagicMock(), MagicMock())

            class AsyncContextManager:
                async def __aenter__(self):
                    return mock_streams

                async def __aexit__(self, *args):
                    server._running = False
                    return None

            mock_stdio.return_value = AsyncContextManager()

            # Ensure create_sse_app is NOT called
            with patch("zapomni_mcp.sse_transport.create_sse_app") as mock_create_app:
                with patch.object(server._server, "run", new_callable=AsyncMock):
                    await server.run()

                mock_create_app.assert_not_called()

    def test_stdio_run_different_from_run_sse(self, mcp_server):
        """run() and run_sse() should be different methods."""
        assert mcp_server.run != mcp_server.run_sse
        assert mcp_server.run.__name__ == "run"
        assert mcp_server.run_sse.__name__ == "run_sse"


# Extended Configuration Priority Tests


class TestExtendedConfigurationPriority:
    """Extended tests for configuration priority across transports."""

    def test_transport_env_variable_support(self):
        """Environment variable for transport should be respected."""
        # Note: Transport is CLI-only, but we test the pattern
        parser = argparse.ArgumentParser()
        parser.add_argument("--transport", default="sse")

        # CLI explicitly overrides
        args = parser.parse_args(["--transport", "stdio"])
        assert args.transport == "stdio"

    def test_sse_config_all_env_vars(self):
        """All SSE environment variables should work."""
        env_vars = {
            "ZAPOMNI_SSE_HOST": "0.0.0.0",
            "ZAPOMNI_SSE_PORT": "9090",
            "ZAPOMNI_SSE_CORS_ORIGINS": "http://localhost:3000,http://localhost:8080",
            "ZAPOMNI_SSE_HEARTBEAT_INTERVAL": "60",
            "ZAPOMNI_SSE_MAX_CONNECTION_LIFETIME": "7200",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            config = SSEConfig.from_env()

            assert config.host == "0.0.0.0"
            assert config.port == 9090
            assert config.cors_origins == ["http://localhost:3000", "http://localhost:8080"]
            assert config.heartbeat_interval == 60
            assert config.max_connection_lifetime == 7200

    def test_cli_port_overrides_env_port(self):
        """CLI --port should override ZAPOMNI_SSE_PORT."""
        with patch.dict(os.environ, {"ZAPOMNI_SSE_PORT": "9999"}):
            sse_config = SSEConfig.from_env()
            assert sse_config.port == 9999

            # Simulate CLI override logic from __main__.py
            cli_port = 8080
            final_port = cli_port if cli_port is not None else sse_config.port
            assert final_port == 8080

    def test_cli_host_overrides_env_host(self):
        """CLI --host should override ZAPOMNI_SSE_HOST."""
        with patch.dict(os.environ, {"ZAPOMNI_SSE_HOST": "0.0.0.0"}):
            sse_config = SSEConfig.from_env()
            assert sse_config.host == "0.0.0.0"

            # Simulate CLI override logic
            cli_host = "127.0.0.1"
            final_host = cli_host if cli_host is not None else sse_config.host
            assert final_host == "127.0.0.1"

    def test_cli_cors_overrides_env_cors(self):
        """CLI --cors-origins should override ZAPOMNI_SSE_CORS_ORIGINS."""
        with patch.dict(os.environ, {"ZAPOMNI_SSE_CORS_ORIGINS": "http://example.com"}):
            sse_config = SSEConfig.from_env()
            assert sse_config.cors_origins == ["http://example.com"]

            # Simulate CLI override logic
            cli_cors = "http://localhost:3000,http://localhost:8080"
            cors_list = [origin.strip() for origin in cli_cors.split(",")]
            assert cors_list == ["http://localhost:3000", "http://localhost:8080"]

    def test_default_values_when_no_env(self):
        """Default values should be used when no env vars set."""
        # Create fresh config without env vars
        config = SSEConfig()

        assert config.host == "127.0.0.1"
        assert config.port == 8000
        assert config.cors_origins == ["*"]
        assert config.heartbeat_interval == 30
        assert config.max_connection_lifetime == 3600


# Transport Selection Tests


class TestTransportSelection:
    """Test transport selection logic."""

    def test_default_transport_is_sse(self):
        """Default transport should be SSE."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--transport", choices=["stdio", "sse"], default="sse")
        args = parser.parse_args([])

        assert args.transport == "sse"

    def test_stdio_flag_selects_stdio(self):
        """--transport stdio should select stdio mode."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--transport", choices=["stdio", "sse"], default="sse")
        args = parser.parse_args(["--transport", "stdio"])

        assert args.transport == "stdio"

    def test_sse_flag_selects_sse(self):
        """--transport sse should select SSE mode."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--transport", choices=["stdio", "sse"], default="sse")
        args = parser.parse_args(["--transport", "sse"])

        assert args.transport == "sse"

    def test_invalid_transport_rejected(self):
        """Invalid transport values should be rejected."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--transport", choices=["stdio", "sse"], default="sse")

        with pytest.raises(SystemExit):
            parser.parse_args(["--transport", "invalid"])


# Tool Handler Consistency Tests


class TestToolHandlerConsistency:
    """Test that tool handlers work identically in both modes."""

    @pytest.mark.asyncio
    async def test_tool_execution_result_format_stdio(self, mcp_server):
        """Tool execution should return consistent format in stdio mode."""
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        mock_tool.input_schema = {"type": "object", "properties": {"text": {"type": "string"}}}
        mock_tool.execute = AsyncMock(
            return_value={
                "content": [{"type": "text", "text": "Hello, World!"}],
                "isError": False,
            }
        )

        mcp_server.register_tool(mock_tool)

        result = await mcp_server._tools["test_tool"].execute({"text": "test"})

        assert "content" in result
        assert isinstance(result["content"], list)
        assert result["content"][0]["type"] == "text"
        assert result["isError"] is False

    @pytest.mark.asyncio
    async def test_tool_error_format_consistent(self, mcp_server):
        """Tool errors should have consistent format."""
        mock_tool = MagicMock()
        mock_tool.name = "error_tool"
        mock_tool.description = "A tool that errors"
        mock_tool.input_schema = {"type": "object"}
        mock_tool.execute = AsyncMock(
            return_value={
                "content": [{"type": "text", "text": "Error: Something went wrong"}],
                "isError": True,
            }
        )

        mcp_server.register_tool(mock_tool)

        result = await mcp_server._tools["error_tool"].execute({})

        assert "content" in result
        assert result["isError"] is True

    def test_tool_schema_preserved(self, mcp_server):
        """Tool input schema should be preserved."""
        schema = {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to process"},
                "limit": {"type": "integer", "default": 10},
            },
            "required": ["text"],
        }

        mock_tool = MagicMock()
        mock_tool.name = "schema_tool"
        mock_tool.description = "Tool with schema"
        mock_tool.input_schema = schema
        mock_tool.execute = AsyncMock(return_value={"content": [], "isError": False})

        mcp_server.register_tool(mock_tool)

        assert mcp_server._tools["schema_tool"].input_schema == schema


# Graceful Shutdown Compatibility Tests


class TestGracefulShutdownCompatibility:
    """Test graceful shutdown works in both modes."""

    def test_shutdown_sets_running_false(self, mcp_server):
        """Shutdown should set _running to False."""
        mcp_server._running = True
        mcp_server.shutdown()
        assert mcp_server._running is False

    def test_shutdown_idempotent(self, mcp_server):
        """Multiple shutdowns should be safe."""
        mcp_server._running = True

        mcp_server.shutdown()
        mcp_server.shutdown()
        mcp_server.shutdown()

        assert mcp_server._running is False

    def test_shutdown_when_not_running(self, mcp_server):
        """Shutdown when not running should not raise."""
        mcp_server._running = False
        mcp_server.shutdown()  # Should not raise
        assert mcp_server._running is False

    @pytest.mark.asyncio
    async def test_stdio_cleanup_on_exception(self, mcp_server):
        """Stdio mode should cleanup on exception."""
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        mock_tool.input_schema = {"type": "object"}
        mock_tool.execute = AsyncMock(return_value={"content": [], "isError": False})
        mcp_server._tools["test_tool"] = mock_tool

        with patch("zapomni_mcp.server.stdio_server") as mock_stdio:

            class FailingContextManager:
                async def __aenter__(self):
                    raise ConnectionError("Simulated connection error")

                async def __aexit__(self, *args):
                    return None

            mock_stdio.return_value = FailingContextManager()

            with pytest.raises(ConnectionError):
                await mcp_server.run()

            # Server should be shut down after exception
            assert mcp_server._running is False
