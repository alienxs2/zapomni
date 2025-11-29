"""
Tests for workspace state persistence (Issue #16 / BUG-004).

This module tests that workspace state persists correctly in stdio mode,
where session_id is None and session_manager is not available.

The bug was that resolve_workspace_id() always returned DEFAULT_WORKSPACE_ID
because:
1. session_id was never passed to tools from server.py
2. There was no instance-level state for stdio mode
3. set_session_workspace() required session_manager which is None in stdio mode

The fix adds instance-level state (_default_workspace_id) that persists
workspace changes in stdio mode.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from zapomni_db.models import DEFAULT_WORKSPACE_ID
from zapomni_mcp.server import MCPServer


class TestWorkspaceStatePersistence:
    """Tests for workspace state persistence in stdio mode (Issue #16 / BUG-004)."""

    @pytest.fixture
    def mock_processor(self):
        """Create mock memory processor."""
        processor = MagicMock()
        processor.db_client = MagicMock()
        return processor

    @pytest.fixture
    def server(self, mock_processor):
        """Create MCPServer instance."""
        return MCPServer(core_engine=mock_processor)

    def test_default_workspace_is_default(self, server):
        """Test that default workspace starts as DEFAULT_WORKSPACE_ID."""
        assert server.resolve_workspace_id(None) == DEFAULT_WORKSPACE_ID
        assert server._default_workspace_id == DEFAULT_WORKSPACE_ID

    def test_set_workspace_stdio_mode(self, server):
        """Test setting workspace in stdio mode (no session_id)."""
        result = server.set_session_workspace(None, "my-workspace")
        assert result is True
        assert server.resolve_workspace_id(None) == "my-workspace"
        assert server._default_workspace_id == "my-workspace"

    def test_workspace_persists_between_calls(self, server):
        """Test workspace persists between multiple resolve calls."""
        server.set_session_workspace(None, "persistent-ws")

        # Multiple calls should return same workspace
        assert server.resolve_workspace_id(None) == "persistent-ws"
        assert server.resolve_workspace_id(None) == "persistent-ws"
        assert server.resolve_workspace_id(None) == "persistent-ws"

    def test_workspace_can_be_changed(self, server):
        """Test workspace can be changed multiple times."""
        server.set_session_workspace(None, "ws-1")
        assert server.resolve_workspace_id(None) == "ws-1"

        server.set_session_workspace(None, "ws-2")
        assert server.resolve_workspace_id(None) == "ws-2"

        server.set_session_workspace(None, "ws-3")
        assert server.resolve_workspace_id(None) == "ws-3"

    def test_sse_mode_uses_session_manager(self, server):
        """Test SSE mode uses session manager when available."""
        # Setup mock session manager
        mock_session_manager = MagicMock()
        mock_session_manager.get_workspace_id.return_value = "session-workspace"
        server._session_manager = mock_session_manager

        # Should use session manager with session_id
        result = server.resolve_workspace_id("session-123")
        assert result == "session-workspace"
        mock_session_manager.get_workspace_id.assert_called_once_with("session-123")

    def test_stdio_mode_ignores_session_manager_when_no_session_id(self, server):
        """Test stdio mode (None session_id) uses instance state even if session_manager exists."""
        server._default_workspace_id = "stdio-workspace"

        # Even with session manager, None session_id should use instance state
        mock_session_manager = MagicMock()
        server._session_manager = mock_session_manager

        result = server.resolve_workspace_id(None)
        assert result == "stdio-workspace"
        mock_session_manager.get_workspace_id.assert_not_called()

    def test_sse_mode_set_workspace_uses_session_manager(self, server):
        """Test SSE mode set_session_workspace uses session manager."""
        mock_session_manager = MagicMock()
        mock_session_manager.set_workspace_id.return_value = True
        server._session_manager = mock_session_manager

        result = server.set_session_workspace("session-123", "sse-workspace")
        assert result is True
        mock_session_manager.set_workspace_id.assert_called_once_with(
            "session-123", "sse-workspace"
        )
        # Instance state should NOT be changed
        assert server._default_workspace_id == DEFAULT_WORKSPACE_ID

    def test_stdio_mode_set_workspace_ignores_session_manager(self, server):
        """Test stdio mode set_session_workspace uses instance state."""
        mock_session_manager = MagicMock()
        server._session_manager = mock_session_manager

        # With None session_id, should use instance state
        result = server.set_session_workspace(None, "stdio-workspace")
        assert result is True
        assert server._default_workspace_id == "stdio-workspace"
        mock_session_manager.set_workspace_id.assert_not_called()

    def test_session_manager_failure_falls_back_to_instance_state(self, server):
        """Test that if session_manager.get_workspace_id fails, we fall back to instance state."""
        server._default_workspace_id = "fallback-workspace"

        mock_session_manager = MagicMock()
        mock_session_manager.get_workspace_id.side_effect = Exception("Session error")
        server._session_manager = mock_session_manager

        # Should fall back to instance state on error
        result = server.resolve_workspace_id("session-123")
        assert result == "fallback-workspace"

    def test_session_manager_returns_none_falls_back_to_instance_state(self, server):
        """Test that if session_manager.get_workspace_id returns None, we fall back to instance state."""
        server._default_workspace_id = "fallback-workspace"

        mock_session_manager = MagicMock()
        mock_session_manager.get_workspace_id.return_value = None
        server._session_manager = mock_session_manager

        # Should fall back to instance state when session returns None
        result = server.resolve_workspace_id("session-123")
        assert result == "fallback-workspace"

    def test_instance_variable_initialized_correctly(self, mock_processor):
        """Test that _default_workspace_id is initialized to DEFAULT_WORKSPACE_ID."""
        server = MCPServer(core_engine=mock_processor)
        assert hasattr(server, "_default_workspace_id")
        assert server._default_workspace_id == DEFAULT_WORKSPACE_ID


class TestWorkspaceStateIntegration:
    """Integration tests for workspace state with SetCurrentWorkspaceTool."""

    @pytest.fixture
    def mock_processor(self):
        """Create mock memory processor."""
        processor = MagicMock()
        processor.db_client = MagicMock()
        return processor

    @pytest.fixture
    def server(self, mock_processor):
        """Create MCPServer instance."""
        return MCPServer(core_engine=mock_processor)

    @pytest.mark.asyncio
    async def test_set_current_workspace_tool_stdio_mode(self, server):
        """Test SetCurrentWorkspaceTool updates instance state in stdio mode."""
        from zapomni_core.workspace_manager import WorkspaceManager
        from zapomni_mcp.tools.workspace_tools import SetCurrentWorkspaceTool

        # Mock workspace manager
        mock_ws_manager = MagicMock(spec=WorkspaceManager)
        mock_workspace = MagicMock()
        mock_workspace.name = "Test Workspace"
        mock_ws_manager.get_workspace = AsyncMock(return_value=mock_workspace)

        tool = SetCurrentWorkspaceTool(
            workspace_manager=mock_ws_manager,
            mcp_server=server,
        )

        # Initial state
        assert server.resolve_workspace_id(None) == DEFAULT_WORKSPACE_ID

        # Execute tool with session_id=None (stdio mode)
        result = await tool.execute(
            {"workspace_id": "test-workspace"},
            session_id=None,  # stdio mode
        )

        # Should succeed
        assert result["isError"] is False
        assert "test-workspace" in result["content"][0]["text"]

        # Instance state should be updated
        assert server.resolve_workspace_id(None) == "test-workspace"
        assert server._default_workspace_id == "test-workspace"

    @pytest.mark.asyncio
    async def test_workflow_create_set_use_workspace(self, server):
        """Test complete workflow: create workspace, set it, verify it persists."""
        # Simulate the user workflow:
        # 1. User calls set_current_workspace("my-project")
        # 2. User calls add_memory (should use "my-project")
        # 3. User calls search_memory (should use "my-project")

        # Step 1: Set workspace
        server.set_session_workspace(None, "my-project")

        # Step 2 & 3: Verify workspace persists
        assert server.resolve_workspace_id(None) == "my-project"
        assert server.resolve_workspace_id(None) == "my-project"  # Still persists

        # Change to another workspace
        server.set_session_workspace(None, "another-project")
        assert server.resolve_workspace_id(None) == "another-project"
