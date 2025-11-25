"""
MCPServer - Main MCP protocol server implementation.

Implements the Model Context Protocol (MCP) server for stdio transport.
Manages tool registration, request routing, and graceful shutdown.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

import asyncio
import re
import signal
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import structlog
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool

from zapomni_core.exceptions import ValidationError
from zapomni_core.workspace_manager import WorkspaceManager
from zapomni_db.models import DEFAULT_WORKSPACE_ID
from zapomni_mcp.config import Settings, SSEConfig
from zapomni_mcp.tools import AddMemoryTool, GetStatsTool, MCPTool, SearchMemoryTool
from zapomni_mcp.tools.build_graph import BuildGraphTool
from zapomni_mcp.tools.clear_all import ClearAllTool
from zapomni_mcp.tools.delete_memory import DeleteMemoryTool
from zapomni_mcp.tools.export_graph import ExportGraphTool
from zapomni_mcp.tools.get_related import GetRelatedTool
from zapomni_mcp.tools.graph_status import GraphStatusTool
from zapomni_mcp.tools.index_codebase import IndexCodebaseTool
from zapomni_mcp.tools.prune_memory import PruneMemoryTool
from zapomni_mcp.tools.set_model import SetModelTool
from zapomni_mcp.tools.workspace_tools import (
    CreateWorkspaceTool,
    DeleteWorkspaceTool,
    GetCurrentWorkspaceTool,
    ListWorkspacesTool,
    SetCurrentWorkspaceTool,
)

# Custom Exceptions


class MCPServerError(Exception):
    """Base exception for all MCP server errors."""

    pass


class ConfigurationError(MCPServerError):
    """Raised when server configuration is invalid."""

    pass


class ToolRegistrationError(MCPServerError):
    """Raised when tool registration fails."""

    pass


class RequestHandlingError(MCPServerError):
    """Raised when request processing fails."""

    pass


# Data Classes


@dataclass
class ServerStats:
    """Statistics about server operation."""

    total_requests: int
    total_errors: int
    registered_tools: int
    uptime_seconds: float
    running: bool


# Main Server Class


class MCPServer:
    """
    Main MCP protocol server implementing stdio transport.

    This is the central component of the zapomni_mcp module, responsible for:
    - Managing the MCP server lifecycle (start, run, stop)
    - Registering and routing MCP tools
    - Handling requests from MCP clients
    - Delegating business logic to ZapomniCore
    - Logging operations and errors

    The server follows the MCP specification and uses stdio transport
    for communication with clients.

    Attributes:
        _server: Internal mcp.server.Server instance
        _core_engine: ZapomniCore processing engine for business logic
        _tools: Registry of registered tools (name -> MCPTool instance)
        _config: Server configuration settings
        _running: Flag indicating if server is currently running
        _logger: Structured logger for stderr output
        _request_count: Total number of requests processed
        _error_count: Total number of errors encountered
        _start_time: Server start timestamp for uptime calculation

    Thread Safety:
        MCPServer is NOT thread-safe. Stdio transport is inherently sequential,
        so only one request is processed at a time.
    """

    def __init__(self, core_engine: Any, config: Optional[Settings] = None) -> None:
        """
        Initialize MCP server with core processing engine.

        Args:
            core_engine: ZapomniCore instance for business logic processing.
                Must be fully initialized and ready to use.
            config: Optional server configuration settings.
                If None, uses defaults from Settings class.

        Raises:
            ConfigurationError: If core_engine is None or invalid config provided
            ValidationError: If config contains invalid values
        """
        # Validate core_engine
        if core_engine is None:
            raise ConfigurationError("core_engine cannot be None")

        # Initialize config with defaults if not provided
        if config is None:
            config = Settings()

        # Store core dependencies
        self._core_engine = core_engine
        self._config = config

        # Initialize server state
        self._running = False
        self._tools: Dict[str, MCPTool] = {}
        self._request_count = 0
        self._error_count = 0
        self._start_time = 0.0

        # Initialize workspace manager (will be created when db_client is available)
        self._workspace_manager: Optional[WorkspaceManager] = None

        # Session manager will be set by SSE transport
        self._session_manager: Optional[Any] = None

        # Setup structured logging to stderr
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.add_log_level,
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.BoundLogger,
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
            cache_logger_on_first_use=True,
        )
        self._logger = structlog.get_logger()

        # Create MCP server instance
        self._server = Server(self._config.server_name)

        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()

        self._logger.info(
            "MCPServer initialized",
            server_name=self._config.server_name,
            log_level=self._config.log_level,
            core_engine_type=type(core_engine).__name__,
        )

    def register_tool(self, tool: MCPTool) -> None:
        """
        Register a single MCP tool with the server.

        Args:
            tool: Tool instance implementing MCPTool protocol.
                Must have unique name not already registered.

        Raises:
            ValueError: If tool.name is empty or already registered
            TypeError: If tool doesn't implement MCPTool protocol
            ValidationError: If tool.input_schema is invalid JSON Schema
        """
        # Validate tool implements protocol
        self._validate_tool(tool)

        # Validate tool name
        if not tool.name or len(tool.name.strip()) == 0:
            raise ValueError("Tool name cannot be empty")

        # Check name format (lowercase with underscores only)
        if not re.match(r"^[a-z_][a-z0-9_]*$", tool.name):
            raise ValueError(
                f"Tool name '{tool.name}' invalid format " "(must be lowercase with underscores)"
            )

        # Check for duplicates
        if tool.name in self._tools:
            raise ValueError(f"Tool name '{tool.name}' already registered")

        # Add to registry
        self._tools[tool.name] = tool

        # Register with MCP SDK
        # Note: MCP SDK registration happens when server starts
        # We just track tools in our registry for now

        self._logger.info("Tool registered", tool_name=tool.name)

    def register_all_tools(self, memory_processor: Optional[Any] = None) -> None:
        """
        Register all standard Zapomni MCP tools.

        Registers:
        - add_memory: Store new information in memory
        - search_memory: Retrieve relevant information
        - get_stats: Query system statistics

        Args:
            memory_processor: Optional MemoryProcessor instance.
                If None, attempts to use self._core_engine if it's a MemoryProcessor.
                If provided, uses this instance for all tools.

        Raises:
            ValueError: If any tool name conflicts
            TypeError: If memory_processor type is invalid
            ImportError: If tool modules cannot be imported
        """
        # Determine which processor to use
        if memory_processor is None:
            # Try to use core_engine if it's a MemoryProcessor
            from zapomni_core.memory_processor import MemoryProcessor

            if isinstance(self._core_engine, MemoryProcessor):
                memory_processor = self._core_engine
            else:
                raise TypeError(
                    f"core_engine must be MemoryProcessor or memory_processor must be provided, "
                    f"got {type(self._core_engine).__name__}"
                )

        # Instantiate all standard tools with MemoryProcessor
        tools = [
            # Phase 1: Core tools
            AddMemoryTool(memory_processor=memory_processor),
            SearchMemoryTool(memory_processor=memory_processor),
            # GetStatsTool gets mcp_server reference for dynamic session_manager access in SSE mode
            GetStatsTool(memory_processor=memory_processor, mcp_server=self),
            # Phase 2: Enhanced Search Tools
            BuildGraphTool(memory_processor=memory_processor),
            GetRelatedTool(memory_processor=memory_processor),
            GraphStatusTool(memory_processor=memory_processor),
            # Phase 3: Code Intelligence & Memory Management Tools
            ExportGraphTool(memory_processor=memory_processor),
            DeleteMemoryTool(memory_processor=memory_processor),
            ClearAllTool(memory_processor=memory_processor),
        ]

        # Phase 3 (Optional): Add IndexCodebaseTool if code indexer is available
        if hasattr(memory_processor, "code_indexer") and memory_processor.code_indexer is not None:
            tools.append(
                IndexCodebaseTool(
                    repository_indexer=memory_processor.code_indexer,
                    memory_processor=memory_processor,
                )
            )

        # Phase 3.5: Add PruneMemoryTool for garbage collection
        if hasattr(memory_processor, "db_client") and memory_processor.db_client is not None:
            tools.append(
                PruneMemoryTool(db_client=memory_processor.db_client)
            )

        # Phase 4: Workspace management tools
        workspace_manager = self.get_workspace_manager()
        if workspace_manager is not None:
            tools.extend(
                [
                    CreateWorkspaceTool(workspace_manager=workspace_manager),
                    ListWorkspacesTool(workspace_manager=workspace_manager),
                    SetCurrentWorkspaceTool(workspace_manager=workspace_manager, mcp_server=self),
                    GetCurrentWorkspaceTool(workspace_manager=workspace_manager, mcp_server=self),
                    DeleteWorkspaceTool(workspace_manager=workspace_manager),
                ]
            )
        else:
            self._logger.warning(
                "workspace_tools_not_registered",
                reason="WorkspaceManager not available (db_client missing)",
            )

        # Phase 5: Configuration hot-reload tools
        tools.append(SetModelTool())

        # Register each tool
        for tool in tools:
            self.register_tool(tool)

        self._logger.info(
            "All tools registered successfully",
            tool_count=len(self._tools),
            tools=list(self._tools.keys()),
        )

    async def run(self) -> None:
        """
        Start the MCP server and process requests from stdin.

        This is the main server loop that processes incoming JSON-RPC 2.0
        messages from stdin and routes them to registered tools.

        Raises:
            RuntimeError: If server is already running or no tools registered
            ConnectionError: If stdin/stdout unavailable
        """
        # Check if already running
        if self._running:
            raise RuntimeError("Server is already running")

        # Check if tools are registered
        if len(self._tools) == 0:
            raise RuntimeError("No tools registered. Call register_all_tools() first.")

        # Set running state
        self._running = True
        self._start_time = time.time()

        self._logger.info(
            "Starting MCP server",
            tool_count=len(self._tools),
            tools=list(self._tools.keys()),
        )

        try:
            # Register tools with MCP server
            for tool in self._tools.values():

                @self._server.call_tool()
                async def handle_call_tool(name: str, arguments: dict) -> list:
                    """Handle tool call from MCP client."""
                    self._request_count += 1

                    try:
                        if name not in self._tools:
                            self._error_count += 1
                            return [
                                {
                                    "type": "text",
                                    "text": f"Error: Unknown tool '{name}'",
                                }
                            ]

                        # Execute tool
                        result = await self._tools[name].execute(arguments)

                        # Return content from result
                        return result.get("content", [])

                    except Exception as e:
                        self._error_count += 1
                        self._logger.error("Tool execution error", tool=name, error=str(e))
                        return [{"type": "text", "text": f"Error: {str(e)}"}]

                @self._server.list_tools()
                async def handle_list_tools() -> list[Tool]:
                    """List all available tools."""
                    return [
                        Tool(
                            name=tool.name,
                            description=tool.description,
                            inputSchema=tool.input_schema,
                        )
                        for tool in self._tools.values()
                    ]

            # Start stdio server (blocks until shutdown)
            async with stdio_server() as streams:
                read_stream, write_stream = streams
                await self._server.run(
                    read_stream, write_stream, self._server.create_initialization_options()
                )

        except Exception as e:
            self._logger.error("Server error", error=str(e))
            raise
        finally:
            # Ensure shutdown is called
            self.shutdown()

    async def run_sse(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        cors_origins: list[str] = None,
    ) -> None:
        """
        Start the MCP server with SSE transport.

        This enables multiple concurrent client connections via HTTP.
        The SSE transport provides:
        - GET /sse: Establish SSE connection for MCP communication
        - POST /messages/{session_id}: Send JSON-RPC messages to session

        Args:
            host: Bind address (default: 127.0.0.1 for local only)
            port: HTTP port (default: 8000)
            cors_origins: Allowed CORS origins (default: ["*"])

        Raises:
            RuntimeError: If server is already running
            OSError: If port is already in use
        """
        import uvicorn

        from zapomni_mcp.sse_transport import create_sse_app

        # Check if already running
        if self._running:
            raise RuntimeError("Server is already running")

        # Check if tools are registered
        if len(self._tools) == 0:
            raise RuntimeError("No tools registered. Call register_all_tools() first.")

        # Create SSE configuration
        config = SSEConfig(
            host=host,
            port=port,
            cors_origins=cors_origins or ["*"],
        )

        # Create Starlette app with SSE routes
        app = create_sse_app(mcp_server=self, config=config)

        # Set running state
        self._running = True
        self._start_time = time.time()

        self._logger.info(
            "Starting SSE server",
            host=host,
            port=port,
            tool_count=len(self._tools),
            tools=list(self._tools.keys()),
            cors_origins=config.cors_origins,
        )

        # Register tools with MCP server (same as stdio mode)
        @self._server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> list:
            """Handle tool call from MCP client."""
            self._request_count += 1

            try:
                if name not in self._tools:
                    self._error_count += 1
                    return [
                        {
                            "type": "text",
                            "text": f"Error: Unknown tool '{name}'",
                        }
                    ]

                # Execute tool
                result = await self._tools[name].execute(arguments)

                # Return content from result
                return result.get("content", [])

            except Exception as e:
                self._error_count += 1
                self._logger.error("Tool execution error", tool=name, error=str(e))
                return [{"type": "text", "text": f"Error: {str(e)}"}]

        @self._server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            """List all available tools."""
            return [
                Tool(
                    name=tool.name,
                    description=tool.description,
                    inputSchema=tool.input_schema,
                )
                for tool in self._tools.values()
            ]

        # Configure uvicorn server
        uvicorn_config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="info",
        )
        server = uvicorn.Server(uvicorn_config)

        try:
            await server.serve()
        except Exception as e:
            self._logger.error("SSE server error", error=str(e))
            raise
        finally:
            # Graceful shutdown sequence
            await self._graceful_shutdown_sse()

    def shutdown(self) -> None:
        """
        Gracefully shut down the MCP server.

        This method is safe to call multiple times (idempotent).
        Can be called from signal handlers.
        """
        if not self._running:
            self._logger.debug("Shutdown called but server not running")
            return

        self._logger.info("Shutting down MCP server...")

        # Set running flag to False
        self._running = False

        # Calculate final statistics
        uptime = time.time() - self._start_time if self._start_time > 0 else 0

        # Log final statistics
        self._logger.info(
            "Server shutdown complete",
            total_requests=self._request_count,
            total_errors=self._error_count,
            uptime_seconds=round(uptime, 2),
            error_rate=(
                round(self._error_count / self._request_count * 100, 2)
                if self._request_count > 0
                else 0
            ),
        )

    async def _graceful_shutdown_sse(self) -> None:
        """
        Graceful shutdown sequence for SSE transport.

        This method performs cleanup in the correct order:
        1. Close all active SSE sessions
        2. Cleanup EntityExtractor thread pool
        3. Close database connection pool
        4. Call standard shutdown

        This is called automatically when the SSE server exits.
        """
        self._logger.info("Starting graceful SSE shutdown...")

        # Step 1: Close all active sessions
        if hasattr(self, "_session_manager") and self._session_manager is not None:
            try:
                closed_count = await self._session_manager.close_all_sessions()
                self._logger.info(
                    "SSE sessions closed",
                    closed_count=closed_count,
                )
            except Exception as e:
                self._logger.error(
                    "Error closing SSE sessions",
                    error=str(e),
                )

        # Step 2: Cleanup EntityExtractor thread pool
        await self._cleanup_entity_extractor()

        # Step 3: Close database connection pool (BREAKING CHANGE: close() is now async)
        await self._close_database_pool()

        # Step 4: Standard shutdown
        self.shutdown()

    async def _cleanup_entity_extractor(self) -> None:
        """
        Cleanup EntityExtractor thread pool during shutdown.

        Waits for pending extractions to complete (with timeout)
        before shutting down the executor.
        """
        try:
            # Check if we have a memory processor with entity extractor
            if hasattr(self._core_engine, "_entity_extractor"):
                extractor = self._core_engine._entity_extractor
                if extractor is not None and hasattr(extractor, "shutdown"):
                    self._logger.info("Shutting down EntityExtractor...")
                    extractor.shutdown()
                    self._logger.info("EntityExtractor shutdown complete")

            # Also check memory_processor.extractor pattern
            if hasattr(self._core_engine, "extractor"):
                extractor = self._core_engine.extractor
                if extractor is not None and hasattr(extractor, "shutdown"):
                    self._logger.info("Shutting down EntityExtractor (via extractor)...")
                    extractor.shutdown()
                    self._logger.info("EntityExtractor shutdown complete")

        except Exception as e:
            self._logger.warning(
                "Error during EntityExtractor cleanup",
                error=str(e),
            )

    async def _close_database_pool(self) -> None:
        """
        Close database connection pool during shutdown.

        Handles async close() method of FalkorDBClient.
        Waits for pending queries to complete before closing.
        """
        try:
            # Check if we have a memory processor with db_client
            if hasattr(self._core_engine, "db_client"):
                db_client = self._core_engine.db_client
                if db_client is not None and hasattr(db_client, "close"):
                    # Get pool stats before closing
                    if hasattr(db_client, "get_pool_stats"):
                        try:
                            stats = await db_client.get_pool_stats()
                            self._logger.info(
                                "Closing database pool",
                                total_queries=stats.get("total_queries", 0),
                                total_retries=stats.get("total_retries", 0),
                            )
                        except Exception:
                            pass

                    # close() is now async
                    self._logger.info("Closing FalkorDB connection pool...")
                    await db_client.close()
                    self._logger.info("FalkorDB connection pool closed")

        except Exception as e:
            self._logger.warning(
                "Error during database pool cleanup",
                error=str(e),
            )

    def get_stats(self) -> ServerStats:
        """
        Get current server statistics.

        Returns:
            ServerStats object containing operational metrics
        """
        uptime = time.time() - self._start_time if self._running and self._start_time > 0 else 0.0

        return ServerStats(
            total_requests=self._request_count,
            total_errors=self._error_count,
            registered_tools=len(self._tools),
            uptime_seconds=uptime,
            running=self._running,
        )

    def get_workspace_manager(self) -> Optional[WorkspaceManager]:
        """
        Get or create the WorkspaceManager instance.

        Creates the WorkspaceManager lazily when first accessed,
        using the db_client from the core engine.

        Returns:
            WorkspaceManager instance, or None if db_client not available
        """
        if self._workspace_manager is not None:
            return self._workspace_manager

        # Try to get db_client from core engine
        if hasattr(self._core_engine, "db_client"):
            db_client = self._core_engine.db_client
            if db_client is not None:
                self._workspace_manager = WorkspaceManager(db_client=db_client)
                self._logger.info("WorkspaceManager initialized")
                return self._workspace_manager

        return None

    def resolve_workspace_id(self, session_id: Optional[str] = None) -> str:
        """
        Resolve the current workspace_id for a request.

        Resolution order:
        1. If session_id provided and session_manager available,
           get workspace from session state
        2. Otherwise, return DEFAULT_WORKSPACE_ID

        Args:
            session_id: Optional session ID for SSE transport

        Returns:
            Resolved workspace_id string
        """
        # Try to get from session if available
        if session_id and self._session_manager is not None:
            try:
                workspace_id = self._session_manager.get_workspace_id(session_id)
                if workspace_id:
                    return workspace_id
            except Exception as e:
                self._logger.warning(
                    "Failed to get workspace from session",
                    session_id=session_id,
                    error=str(e),
                )

        return DEFAULT_WORKSPACE_ID

    def set_session_workspace(
        self,
        session_id: str,
        workspace_id: str,
    ) -> bool:
        """
        Set the workspace_id for a session.

        Args:
            session_id: Session ID
            workspace_id: Workspace ID to set

        Returns:
            True if successful, False otherwise
        """
        if self._session_manager is None:
            self._logger.warning("Session manager not available")
            return False

        return self._session_manager.set_workspace_id(session_id, workspace_id)

    def _setup_signal_handlers(self) -> None:
        """
        Install signal handlers for graceful shutdown.

        Registers handlers for SIGINT (Ctrl+C) and SIGTERM (kill).
        """

        def signal_handler(signum, frame):
            """Handle shutdown signals."""
            self._logger.info("Received shutdown signal", signal=signum)
            self.shutdown()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _validate_tool(self, tool: MCPTool) -> None:
        """
        Validate that a tool implements MCPTool protocol correctly.

        Args:
            tool: Tool instance to validate

        Raises:
            TypeError: If tool doesn't implement required attributes/methods
        """
        # Check for required attributes
        if not hasattr(tool, "name"):
            raise TypeError("Tool must implement 'name' attribute")

        if not hasattr(tool, "description"):
            raise TypeError("Tool must implement 'description' attribute")

        if not hasattr(tool, "input_schema"):
            raise TypeError("Tool must implement 'input_schema' attribute")

        # Check for execute method
        if not hasattr(tool, "execute") or not callable(getattr(tool, "execute")):
            raise TypeError("Tool must implement 'execute' method")


# Entry Point


async def main() -> None:
    """
    Main entry point for MCP server.

    This would be called from the command-line script.
    For now, this is a placeholder.
    """
    # This will be implemented when we have full ZapomniCore integration
    # For now, server is tested via unit tests with mocked core
    pass


if __name__ == "__main__":
    asyncio.run(main())
