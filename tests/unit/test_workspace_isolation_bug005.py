"""
Unit tests for Issue #12 - Workspace isolation bug (BUG-005).

Tests verify that:
1. AddMemoryTool and SearchMemoryTool accept mcp_server parameter
2. Tools call resolve_workspace_id() when workspace_id not provided
3. Tools correctly use session workspace for memory operations

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from zapomni_core.memory_processor import MemoryProcessor


class TestAddMemoryToolWorkspaceResolution:
    """Test AddMemoryTool workspace resolution via mcp_server."""

    @pytest.fixture
    def mock_processor(self):
        """Create a mock MemoryProcessor."""
        processor = Mock(spec=MemoryProcessor)
        processor.add_memory = AsyncMock(return_value="test-memory-id-123")
        return processor

    @pytest.fixture
    def mock_mcp_server(self):
        """Create a mock MCPServer with workspace resolution."""
        server = MagicMock()
        server.resolve_workspace_id = Mock(return_value="session-workspace")
        return server

    @pytest.mark.asyncio
    async def test_add_memory_accepts_mcp_server_parameter(self, mock_processor, mock_mcp_server):
        """Test that AddMemoryTool accepts mcp_server parameter."""
        from zapomni_mcp.tools.add_memory import AddMemoryTool

        # Should not raise - mcp_server is optional but accepted
        tool = AddMemoryTool(memory_processor=mock_processor, mcp_server=mock_mcp_server)

        assert tool.mcp_server == mock_mcp_server

    @pytest.mark.asyncio
    async def test_add_memory_uses_resolve_workspace_id_when_not_provided(
        self, mock_processor, mock_mcp_server
    ):
        """Test that add_memory calls resolve_workspace_id when workspace_id not in arguments."""
        from zapomni_mcp.tools.add_memory import AddMemoryTool

        tool = AddMemoryTool(memory_processor=mock_processor, mcp_server=mock_mcp_server)

        # Execute without workspace_id
        arguments = {"text": "Test memory content"}
        await tool.execute(arguments)

        # Verify resolve_workspace_id was called
        mock_mcp_server.resolve_workspace_id.assert_called_once()

        # Verify add_memory was called with resolved workspace_id
        call_args = mock_processor.add_memory.call_args
        assert call_args[1]["workspace_id"] == "session-workspace"

    @pytest.mark.asyncio
    async def test_add_memory_uses_explicit_workspace_id_when_provided(
        self, mock_processor, mock_mcp_server
    ):
        """Test that add_memory uses explicit workspace_id when provided in arguments."""
        from zapomni_mcp.tools.add_memory import AddMemoryTool

        tool = AddMemoryTool(memory_processor=mock_processor, mcp_server=mock_mcp_server)

        # Execute with explicit workspace_id
        arguments = {"text": "Test memory content", "workspace_id": "explicit-workspace"}
        await tool.execute(arguments)

        # Verify resolve_workspace_id was NOT called
        mock_mcp_server.resolve_workspace_id.assert_not_called()

        # Verify add_memory was called with explicit workspace_id
        call_args = mock_processor.add_memory.call_args
        assert call_args[1]["workspace_id"] == "explicit-workspace"

    @pytest.mark.asyncio
    async def test_add_memory_falls_back_to_default_without_mcp_server(self, mock_processor):
        """Test that add_memory uses default workspace when mcp_server not provided."""
        from zapomni_mcp.tools.add_memory import AddMemoryTool

        # Create tool without mcp_server (backwards compatibility)
        tool = AddMemoryTool(memory_processor=mock_processor)

        # Execute without workspace_id
        arguments = {"text": "Test memory content"}
        await tool.execute(arguments)

        # Verify add_memory was called with default workspace_id (None -> processor handles default)
        call_args = mock_processor.add_memory.call_args
        # When no mcp_server and no workspace_id provided, should pass None
        # (processor handles default)
        assert call_args[1]["workspace_id"] is None


class TestSearchMemoryToolWorkspaceResolution:
    """Test SearchMemoryTool workspace resolution via mcp_server."""

    @pytest.fixture
    def mock_processor(self):
        """Create a mock MemoryProcessor."""
        processor = Mock(spec=MemoryProcessor)
        processor.search_memory = AsyncMock(return_value=[])
        return processor

    @pytest.fixture
    def mock_mcp_server(self):
        """Create a mock MCPServer with workspace resolution."""
        server = MagicMock()
        server.resolve_workspace_id = Mock(return_value="session-workspace")
        return server

    @pytest.mark.asyncio
    async def test_search_memory_accepts_mcp_server_parameter(
        self, mock_processor, mock_mcp_server
    ):
        """Test that SearchMemoryTool accepts mcp_server parameter."""
        from zapomni_mcp.tools.search_memory import SearchMemoryTool

        # Should not raise - mcp_server is optional but accepted
        tool = SearchMemoryTool(memory_processor=mock_processor, mcp_server=mock_mcp_server)

        assert tool.mcp_server == mock_mcp_server

    @pytest.mark.asyncio
    async def test_search_memory_uses_resolve_workspace_id_when_not_provided(
        self, mock_processor, mock_mcp_server
    ):
        """Test that search_memory calls resolve_workspace_id when workspace_id not in arguments."""
        from zapomni_mcp.tools.search_memory import SearchMemoryTool

        tool = SearchMemoryTool(memory_processor=mock_processor, mcp_server=mock_mcp_server)

        # Execute without workspace_id
        arguments = {"query": "test query"}
        await tool.execute(arguments)

        # Verify resolve_workspace_id was called
        mock_mcp_server.resolve_workspace_id.assert_called_once()

        # Verify search_memory was called with resolved workspace_id
        call_args = mock_processor.search_memory.call_args
        assert call_args[1]["workspace_id"] == "session-workspace"

    @pytest.mark.asyncio
    async def test_search_memory_uses_explicit_workspace_id_when_provided(
        self, mock_processor, mock_mcp_server
    ):
        """Test that search_memory uses explicit workspace_id when provided in arguments."""
        from zapomni_mcp.tools.search_memory import SearchMemoryTool

        tool = SearchMemoryTool(memory_processor=mock_processor, mcp_server=mock_mcp_server)

        # Execute with explicit workspace_id
        arguments = {"query": "test query", "workspace_id": "explicit-workspace"}
        await tool.execute(arguments)

        # Verify resolve_workspace_id was NOT called
        mock_mcp_server.resolve_workspace_id.assert_not_called()

        # Verify search_memory was called with explicit workspace_id
        call_args = mock_processor.search_memory.call_args
        assert call_args[1]["workspace_id"] == "explicit-workspace"

    @pytest.mark.asyncio
    async def test_search_memory_falls_back_to_default_without_mcp_server(self, mock_processor):
        """Test that search_memory uses default workspace when mcp_server not provided."""
        from zapomni_mcp.tools.search_memory import SearchMemoryTool

        # Create tool without mcp_server (backwards compatibility)
        tool = SearchMemoryTool(memory_processor=mock_processor)

        # Execute without workspace_id
        arguments = {"query": "test query"}
        await tool.execute(arguments)

        # Verify search_memory was called with default workspace_id
        # (None -> processor handles default)
        call_args = mock_processor.search_memory.call_args
        # When no mcp_server and no workspace_id provided, should pass None
        # (processor handles default)
        assert call_args[1]["workspace_id"] is None


class TestWorkspaceIsolationEndToEnd:
    """End-to-end tests for workspace isolation scenario from Issue #12."""

    @pytest.fixture
    def mock_processor(self):
        """Create a mock MemoryProcessor."""
        processor = Mock(spec=MemoryProcessor)
        processor.add_memory = AsyncMock(return_value="test-memory-id")
        processor.search_memory = AsyncMock(return_value=[])
        return processor

    @pytest.fixture
    def mock_mcp_server_project_a(self):
        """Create mock MCPServer returning project-a workspace."""
        server = MagicMock()
        server.resolve_workspace_id = Mock(return_value="project-a")
        return server

    @pytest.fixture
    def mock_mcp_server_project_b(self):
        """Create mock MCPServer returning project-b workspace."""
        server = MagicMock()
        server.resolve_workspace_id = Mock(return_value="project-b")
        return server

    @pytest.mark.asyncio
    async def test_workspace_isolation_scenario(
        self, mock_processor, mock_mcp_server_project_a, mock_mcp_server_project_b
    ):
        """
        Test the exact scenario from Issue #12:
        1. set_current_workspace("project-a")
        2. add_memory("secret data") without workspace_id
        3. set_current_workspace("project-b")
        4. search_memory("secret") should NOT find data from project-a

        This test verifies that tools correctly use session workspace.
        """
        from zapomni_mcp.tools.add_memory import AddMemoryTool
        from zapomni_mcp.tools.search_memory import SearchMemoryTool

        # Step 1-2: Add memory in project-a context
        add_tool = AddMemoryTool(
            memory_processor=mock_processor, mcp_server=mock_mcp_server_project_a
        )
        await add_tool.execute({"text": "secret data"})

        # Verify memory was added to project-a
        add_call = mock_processor.add_memory.call_args
        assert add_call[1]["workspace_id"] == "project-a"

        # Step 3-4: Search in project-b context
        search_tool = SearchMemoryTool(
            memory_processor=mock_processor, mcp_server=mock_mcp_server_project_b
        )
        await search_tool.execute({"query": "secret"})

        # Verify search was performed in project-b (isolated from project-a)
        search_call = mock_processor.search_memory.call_args
        assert search_call[1]["workspace_id"] == "project-b"


class TestServerRegisterToolsWithMCPServer:
    """Test that server.register_all_tools passes mcp_server to tools."""

    @pytest.mark.asyncio
    async def test_register_all_tools_passes_mcp_server(self):
        """
        Test register_all_tools injects mcp_server into add_memory and search_memory tools.
        """
        from zapomni_db.falkordb_client import FalkorDBClient
        from zapomni_mcp.server import MCPServer

        # Create mock memory processor with db_client for ExportGraphTool
        mock_db_client = Mock(spec=FalkorDBClient)
        mock_db_client._graph_name = "test_graph"
        mock_db_client._execute_cypher = AsyncMock(return_value=MagicMock(rows=[]))

        mock_processor = Mock(spec=MemoryProcessor)
        mock_processor.db_client = mock_db_client  # For ExportGraphTool and PruneMemoryTool
        mock_processor.code_indexer = None  # Disable index tool

        # Create server
        server = MCPServer(core_engine=mock_processor)

        # Register tools
        server.register_all_tools(memory_processor=mock_processor)

        # Verify add_memory tool has mcp_server reference
        add_memory_tool = server._tools.get("add_memory")
        assert add_memory_tool is not None
        assert hasattr(add_memory_tool, "mcp_server")
        assert add_memory_tool.mcp_server == server

        # Verify search_memory tool has mcp_server reference
        search_memory_tool = server._tools.get("search_memory")
        assert search_memory_tool is not None
        assert hasattr(search_memory_tool, "mcp_server")
        assert search_memory_tool.mcp_server == server
