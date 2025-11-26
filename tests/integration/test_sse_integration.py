"""
Integration tests for SSE Transport.

Tests the full SSE connection lifecycle including:
- Full SSE connection lifecycle (connect, call tools, disconnect)
- Concurrent client connections
- Tool execution through SSE
- Error propagation

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient

from zapomni_mcp.config import SSEConfig
from zapomni_mcp.session_manager import SessionManager, generate_session_id

# Fixtures


@pytest.fixture
def sse_config():
    """Create SSEConfig for integration testing."""
    return SSEConfig(
        host="127.0.0.1",
        port=8000,
        cors_origins=["*"],
        heartbeat_interval=30,
        max_connection_lifetime=3600,
    )


@pytest.fixture
def mock_core_engine():
    """Create mock ZapomniCore engine."""
    mock = MagicMock()
    mock.add_memory = AsyncMock(return_value={"memory_id": "test-uuid-123"})
    mock.search_memory = AsyncMock(return_value=[])
    mock.get_stats = AsyncMock(
        return_value={
            "total_memories": 10,
            "total_chunks": 50,
            "graph_nodes": 25,
            "database_size_mb": 1.5,
        }
    )
    return mock


@pytest.fixture
def mock_mcp_server(mock_core_engine):
    """Create a mock MCPServer with tools registered."""
    mock = MagicMock()
    mock._server = MagicMock()
    mock._server.run = AsyncMock()
    mock._server.create_initialization_options = MagicMock(return_value={})
    mock._session_manager = None
    mock._core_engine = mock_core_engine
    mock._running = True
    mock._request_count = 0
    mock._error_count = 0
    mock._tools = {}
    return mock


@pytest.fixture
def sse_app(mock_mcp_server, sse_config):
    """Create SSE Starlette app for integration testing."""
    from zapomni_mcp.sse_transport import create_sse_app

    return create_sse_app(mock_mcp_server, sse_config)


@pytest.fixture
def test_client(sse_app):
    """Create test client for SSE app."""
    return TestClient(sse_app, raise_server_exceptions=False)


# Full Lifecycle Tests


class TestSSELifecycle:
    """Test suite for full SSE connection lifecycle."""

    @pytest.mark.asyncio
    async def test_session_creation_and_cleanup(self, mock_mcp_server, sse_config):
        """Should create and cleanup session properly."""
        from zapomni_mcp.sse_transport import create_sse_app

        app = create_sse_app(mock_mcp_server, sse_config)
        session_manager = mock_mcp_server._session_manager

        # Simulate session creation
        mock_transport = MagicMock()
        session_id = generate_session_id()

        await session_manager.create_session(
            session_id=session_id,
            transport=mock_transport,
            client_ip="127.0.0.1",
        )

        assert session_manager.active_session_count == 1
        session = session_manager.get_session(session_id)
        assert session is not None
        assert session.client_ip == "127.0.0.1"

        # Simulate session cleanup
        await session_manager.remove_session(session_id)

        assert session_manager.active_session_count == 0
        assert session_manager.get_session(session_id) is None

    @pytest.mark.asyncio
    async def test_multiple_sessions_independent(self, mock_mcp_server, sse_config):
        """Should handle multiple independent sessions."""
        from zapomni_mcp.sse_transport import create_sse_app

        app = create_sse_app(mock_mcp_server, sse_config)
        session_manager = mock_mcp_server._session_manager

        # Create multiple sessions
        sessions = []
        for i in range(5):
            session_id = generate_session_id()
            await session_manager.create_session(
                session_id=session_id,
                transport=MagicMock(),
            )
            sessions.append(session_id)

        assert session_manager.active_session_count == 5

        # Remove sessions one by one
        for i, session_id in enumerate(sessions):
            await session_manager.remove_session(session_id)
            assert session_manager.active_session_count == 4 - i

    @pytest.mark.asyncio
    async def test_session_activity_tracking(self, mock_mcp_server, sse_config):
        """Should track session activity correctly."""
        from zapomni_mcp.sse_transport import create_sse_app

        app = create_sse_app(mock_mcp_server, sse_config)
        session_manager = mock_mcp_server._session_manager

        session_id = generate_session_id()
        mock_transport = MagicMock()
        mock_transport.handle_post_message = AsyncMock()

        await session_manager.create_session(
            session_id=session_id,
            transport=mock_transport,
        )

        # Simulate message handling
        session_manager.increment_request_count(session_id)
        session_manager.increment_request_count(session_id)

        session = session_manager.get_session(session_id)
        assert session.request_count == 2

        await session_manager.remove_session(session_id)


# Concurrent Connection Tests


class TestConcurrentConnections:
    """Test suite for concurrent client connections."""

    @pytest.mark.asyncio
    async def test_concurrent_session_creation(self, mock_mcp_server, sse_config):
        """Should handle concurrent session creation."""
        from zapomni_mcp.sse_transport import create_sse_app

        app = create_sse_app(mock_mcp_server, sse_config)
        session_manager = mock_mcp_server._session_manager

        async def create_session():
            session_id = generate_session_id()
            await session_manager.create_session(
                session_id=session_id,
                transport=MagicMock(),
            )
            return session_id

        # Create 20 sessions concurrently
        session_ids = await asyncio.gather(*[create_session() for _ in range(20)])

        assert session_manager.active_session_count == 20
        assert len(set(session_ids)) == 20  # All unique

        # Cleanup
        await session_manager.close_all_sessions()

    @pytest.mark.asyncio
    async def test_concurrent_message_handling(self, mock_mcp_server, sse_config):
        """Should handle concurrent messages to different sessions."""
        from zapomni_mcp.sse_transport import create_sse_app

        app = create_sse_app(mock_mcp_server, sse_config)
        session_manager = mock_mcp_server._session_manager

        # Create sessions
        sessions = []
        for i in range(10):
            session_id = generate_session_id()
            mock_transport = MagicMock()
            mock_transport.handle_post_message = AsyncMock()
            await session_manager.create_session(
                session_id=session_id,
                transport=mock_transport,
            )
            sessions.append(session_id)

        # Simulate concurrent message handling
        async def process_messages(session_id):
            for _ in range(5):
                session_manager.increment_request_count(session_id)
                await asyncio.sleep(0.001)

        await asyncio.gather(*[process_messages(sid) for sid in sessions])

        # Each session should have 5 requests
        for sid in sessions:
            session = session_manager.get_session(sid)
            assert session.request_count == 5

        # Cleanup
        await session_manager.close_all_sessions()

    @pytest.mark.asyncio
    async def test_peak_connections_tracking(self, mock_mcp_server, sse_config):
        """Should track peak concurrent connections."""
        from zapomni_mcp.sse_transport import create_sse_app

        app = create_sse_app(mock_mcp_server, sse_config)
        session_manager = mock_mcp_server._session_manager

        # Create 10 sessions
        sessions = []
        for i in range(10):
            session_id = generate_session_id()
            await session_manager.create_session(
                session_id=session_id,
                transport=MagicMock(),
            )
            sessions.append(session_id)

        metrics = session_manager.get_metrics()
        assert metrics.peak_connections == 10

        # Remove half
        for sid in sessions[:5]:
            await session_manager.remove_session(sid)

        # Peak should still be 10
        metrics = session_manager.get_metrics()
        assert metrics.peak_connections == 10
        assert metrics.current_active_connections == 5

        # Cleanup
        await session_manager.close_all_sessions()


# Tool Execution Tests


class TestToolExecution:
    """Test suite for tool execution through SSE."""

    @pytest.mark.asyncio
    async def test_message_forwarding_to_transport(self, mock_mcp_server, sse_config):
        """Should forward messages to the correct transport."""
        from zapomni_mcp.sse_transport import create_sse_app

        app = create_sse_app(mock_mcp_server, sse_config)
        session_manager = mock_mcp_server._session_manager

        # Create session with mock transport
        session_id = "tool-test-session"
        mock_transport = MagicMock()
        mock_transport.handle_post_message = AsyncMock()

        await session_manager.create_session(
            session_id=session_id,
            transport=mock_transport,
        )

        # Test through client
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                f"/messages/{session_id}",
                json={
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {"name": "get_stats"},
                    "id": 1,
                },
            )

            # Should accept the message
            # Note: actual forwarding depends on transport implementation
            session = session_manager.get_session(session_id)
            assert session.request_count >= 1

    @pytest.mark.asyncio
    async def test_session_isolation(self, mock_mcp_server, sse_config):
        """Should isolate sessions from each other."""
        from zapomni_mcp.sse_transport import create_sse_app

        app = create_sse_app(mock_mcp_server, sse_config)
        session_manager = mock_mcp_server._session_manager

        # Create two sessions
        session1_id = "session-1"
        session2_id = "session-2"

        mock_transport1 = MagicMock()
        mock_transport1.handle_post_message = AsyncMock()
        mock_transport2 = MagicMock()
        mock_transport2.handle_post_message = AsyncMock()

        await session_manager.create_session(
            session_id=session1_id,
            transport=mock_transport1,
        )
        await session_manager.create_session(
            session_id=session2_id,
            transport=mock_transport2,
        )

        # Send messages to session 1
        session_manager.increment_request_count(session1_id)
        session_manager.increment_request_count(session1_id)

        # Session 2 should not be affected
        session1 = session_manager.get_session(session1_id)
        session2 = session_manager.get_session(session2_id)

        assert session1.request_count == 2
        assert session2.request_count == 0

        # Cleanup
        await session_manager.close_all_sessions()


# Health Endpoint Integration Tests


class TestHealthIntegration:
    """Test suite for health endpoint integration."""

    def test_health_reflects_actual_state(self, test_client):
        """Should reflect actual server state."""
        response = test_client.get("/health")
        data = response.json()

        assert response.status_code == 200
        assert data["status"] == "healthy"
        assert data["transport"] == "sse"
        assert "version" in data

    @pytest.mark.asyncio
    async def test_health_updates_with_sessions(self, mock_mcp_server, sse_config):
        """Should update health as sessions change."""
        from zapomni_mcp.sse_transport import create_sse_app

        app = create_sse_app(mock_mcp_server, sse_config)
        session_manager = mock_mcp_server._session_manager

        with TestClient(app) as client:
            # Initial state
            response = client.get("/health")
            assert response.json()["active_connections"] == 0

            # Add sessions
            for i in range(3):
                await session_manager.create_session(
                    session_id=f"health-session-{i}",
                    transport=MagicMock(),
                )

            response = client.get("/health")
            assert response.json()["active_connections"] == 3

            # Remove one
            await session_manager.remove_session("health-session-0")

            response = client.get("/health")
            assert response.json()["active_connections"] == 2


# Error Propagation Tests


class TestErrorPropagation:
    """Test suite for error propagation through SSE."""

    def test_404_for_nonexistent_session(self, test_client):
        """Should return 404 for non-existent session."""
        response = test_client.post(
            "/messages/nonexistent-session-id",
            json={"jsonrpc": "2.0", "method": "test", "id": 1},
        )

        assert response.status_code == 404
        data = response.json()
        assert "Session not found" in data["error"]

    @pytest.mark.asyncio
    async def test_500_on_transport_error(self, mock_mcp_server, sse_config):
        """Should return 500 on transport error."""
        from zapomni_mcp.sse_transport import create_sse_app

        app = create_sse_app(mock_mcp_server, sse_config)
        session_manager = mock_mcp_server._session_manager

        # Create session with failing transport
        session_id = "error-session"
        mock_transport = MagicMock()
        mock_transport.handle_post_message = AsyncMock(side_effect=Exception("Transport failed"))

        await session_manager.create_session(
            session_id=session_id,
            transport=mock_transport,
        )

        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                f"/messages/{session_id}",
                json={"jsonrpc": "2.0", "method": "test", "id": 1},
            )

            assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_error_count_incremented(self, mock_mcp_server, sse_config):
        """Should increment error count on failures."""
        from zapomni_mcp.sse_transport import create_sse_app

        app = create_sse_app(mock_mcp_server, sse_config)
        session_manager = mock_mcp_server._session_manager

        # Create session with failing transport
        session_id = "error-count-session"
        mock_transport = MagicMock()
        mock_transport.handle_post_message = AsyncMock(side_effect=Exception("Error"))

        await session_manager.create_session(
            session_id=session_id,
            transport=mock_transport,
        )

        with TestClient(app, raise_server_exceptions=False) as client:
            # Make multiple failing requests
            for _ in range(3):
                client.post(
                    f"/messages/{session_id}",
                    json={"jsonrpc": "2.0", "method": "test", "id": 1},
                )

        session = session_manager.get_session(session_id)
        assert session.error_count >= 3


# Graceful Shutdown Tests


class TestGracefulShutdown:
    """Test suite for graceful shutdown."""

    @pytest.mark.asyncio
    async def test_close_all_sessions(self, mock_mcp_server, sse_config):
        """Should close all sessions on shutdown."""
        from zapomni_mcp.sse_transport import create_sse_app

        app = create_sse_app(mock_mcp_server, sse_config)
        session_manager = mock_mcp_server._session_manager

        # Create sessions
        for i in range(5):
            await session_manager.create_session(
                session_id=f"shutdown-session-{i}",
                transport=MagicMock(),
            )

        assert session_manager.active_session_count == 5

        # Shutdown
        closed = await session_manager.close_all_sessions()

        assert closed == 5
        assert session_manager.active_session_count == 0

    @pytest.mark.asyncio
    async def test_metrics_preserved_after_shutdown(self, mock_mcp_server, sse_config):
        """Should preserve metrics after shutdown."""
        from zapomni_mcp.sse_transport import create_sse_app

        app = create_sse_app(mock_mcp_server, sse_config)
        session_manager = mock_mcp_server._session_manager

        # Create and process sessions
        for i in range(3):
            session_id = f"metrics-session-{i}"
            await session_manager.create_session(
                session_id=session_id,
                transport=MagicMock(),
            )
            session_manager.increment_request_count(session_id)

        # Shutdown
        await session_manager.close_all_sessions()

        # Metrics should be preserved
        metrics = session_manager.get_metrics()
        assert metrics.total_connections_created == 3
        assert metrics.total_connections_closed == 3
        assert metrics.total_requests_processed == 3


# Stale Session Cleanup Tests


class TestStaleSessionCleanup:
    """Test suite for stale session cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_stale_sessions(self, mock_mcp_server, sse_config):
        """Should cleanup stale sessions."""
        import time

        from zapomni_mcp.sse_transport import create_sse_app

        app = create_sse_app(mock_mcp_server, sse_config)
        session_manager = mock_mcp_server._session_manager

        # Create session
        session_id = "stale-session"
        await session_manager.create_session(
            session_id=session_id,
            transport=MagicMock(),
        )

        # Make it stale
        session = session_manager.get_session(session_id)
        session.created_at = time.monotonic() - 4000  # 4000 seconds ago

        # Create fresh session
        await session_manager.create_session(
            session_id="fresh-session",
            transport=MagicMock(),
        )

        # Cleanup with 1 hour lifetime
        cleaned = await session_manager.cleanup_stale_sessions(3600)

        assert cleaned == 1
        assert session_manager.get_session(session_id) is None
        assert session_manager.get_session("fresh-session") is not None
