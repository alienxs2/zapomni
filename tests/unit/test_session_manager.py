"""
Unit tests for SessionManager component.

Tests the SSE session lifecycle management including:
- Session creation with unique IDs
- Session lookup (found/not found)
- Session deletion and cleanup
- Connection metrics tracking
- Heartbeat task management
- close_all_sessions() for graceful shutdown

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from zapomni_mcp.session_manager import (
    ConnectionMetrics,
    SessionManager,
    SessionState,
    generate_session_id,
)

# Fixtures


@pytest.fixture
def session_manager():
    """Create a SessionManager instance for testing."""
    return SessionManager(heartbeat_interval=30)


@pytest.fixture
def mock_transport():
    """Create a mock SseServerTransport for testing."""
    transport = MagicMock()
    transport.connect_sse = AsyncMock()
    transport.handle_post_message = AsyncMock()
    return transport


# Session Creation Tests


class TestSessionCreation:
    """Test suite for session creation functionality."""

    @pytest.mark.asyncio
    async def test_create_session_success(self, session_manager, mock_transport):
        """Should create session successfully with valid parameters."""
        session_id = "test-session-123"

        await session_manager.create_session(
            session_id=session_id,
            transport=mock_transport,
            client_ip="127.0.0.1",
        )

        assert session_manager.active_session_count == 1
        session = session_manager.get_session(session_id)
        assert session is not None
        assert session.session_id == session_id
        assert session.transport is mock_transport
        assert session.client_ip == "127.0.0.1"
        assert session.request_count == 0
        assert session.error_count == 0

    @pytest.mark.asyncio
    async def test_create_session_duplicate_raises(self, session_manager, mock_transport):
        """Should raise ValueError when creating session with duplicate ID."""
        session_id = "duplicate-session"

        await session_manager.create_session(
            session_id=session_id,
            transport=mock_transport,
        )

        with pytest.raises(ValueError, match="already exists"):
            await session_manager.create_session(
                session_id=session_id,
                transport=mock_transport,
            )

    @pytest.mark.asyncio
    async def test_create_session_tracks_timestamps(self, session_manager, mock_transport):
        """Should track created_at and last_activity timestamps."""
        session_id = "timestamp-session"

        await session_manager.create_session(
            session_id=session_id,
            transport=mock_transport,
        )

        session = session_manager.get_session(session_id)
        assert session is not None
        assert session.created_at > 0
        assert session.last_activity > 0
        assert session.last_activity >= session.created_at

    @pytest.mark.asyncio
    async def test_create_session_updates_metrics(self, session_manager, mock_transport):
        """Should update connection metrics on session creation."""
        await session_manager.create_session(
            session_id="metrics-session-1",
            transport=mock_transport,
        )

        metrics = session_manager.get_metrics()
        assert metrics.total_connections_created == 1
        assert metrics.current_active_connections == 1
        assert metrics.peak_connections == 1

    @pytest.mark.asyncio
    async def test_create_session_tracks_peak_connections(self, session_manager, mock_transport):
        """Should track peak concurrent connections."""
        # Create 3 sessions
        for i in range(3):
            await session_manager.create_session(
                session_id=f"peak-session-{i}",
                transport=MagicMock(),
            )

        metrics = session_manager.get_metrics()
        assert metrics.peak_connections == 3

        # Remove one session
        await session_manager.remove_session("peak-session-1")

        # Peak should still be 3
        metrics = session_manager.get_metrics()
        assert metrics.peak_connections == 3
        assert metrics.current_active_connections == 2


# Session Lookup Tests


class TestSessionLookup:
    """Test suite for session lookup functionality."""

    @pytest.mark.asyncio
    async def test_get_session_found(self, session_manager, mock_transport):
        """Should return session when it exists."""
        session_id = "lookup-session"
        await session_manager.create_session(
            session_id=session_id,
            transport=mock_transport,
        )

        session = session_manager.get_session(session_id)

        assert session is not None
        assert session.session_id == session_id
        assert session.transport is mock_transport

    def test_get_session_not_found(self, session_manager):
        """Should return None for non-existent session."""
        session = session_manager.get_session("non-existent-session")
        assert session is None

    @pytest.mark.asyncio
    async def test_get_all_session_ids(self, session_manager):
        """Should return list of all active session IDs."""
        # Create multiple sessions
        session_ids = ["session-a", "session-b", "session-c"]
        for sid in session_ids:
            await session_manager.create_session(
                session_id=sid,
                transport=MagicMock(),
            )

        all_ids = await session_manager.get_all_session_ids()

        assert len(all_ids) == 3
        for sid in session_ids:
            assert sid in all_ids


# Session Removal Tests


class TestSessionRemoval:
    """Test suite for session removal functionality."""

    @pytest.mark.asyncio
    async def test_remove_session_success(self, session_manager, mock_transport):
        """Should remove existing session and return True."""
        session_id = "remove-session"
        await session_manager.create_session(
            session_id=session_id,
            transport=mock_transport,
        )

        result = await session_manager.remove_session(session_id)

        assert result is True
        assert session_manager.get_session(session_id) is None
        assert session_manager.active_session_count == 0

    @pytest.mark.asyncio
    async def test_remove_session_not_found(self, session_manager):
        """Should return False when session doesn't exist."""
        result = await session_manager.remove_session("non-existent")
        assert result is False

    @pytest.mark.asyncio
    async def test_remove_session_updates_metrics(self, session_manager, mock_transport):
        """Should update metrics on session removal."""
        session_id = "remove-metrics-session"
        await session_manager.create_session(
            session_id=session_id,
            transport=mock_transport,
        )

        # Increment some counters
        session_manager.increment_request_count(session_id)
        session_manager.increment_request_count(session_id)
        session_manager.increment_error_count(session_id)

        await session_manager.remove_session(session_id)

        metrics = session_manager.get_metrics()
        assert metrics.total_connections_closed == 1
        assert metrics.current_active_connections == 0
        assert metrics.total_requests_processed == 2
        assert metrics.total_errors == 1


# Activity Tracking Tests


class TestActivityTracking:
    """Test suite for session activity tracking."""

    @pytest.mark.asyncio
    async def test_update_activity(self, session_manager, mock_transport):
        """Should update last_activity timestamp."""
        session_id = "activity-session"
        await session_manager.create_session(
            session_id=session_id,
            transport=mock_transport,
        )

        session = session_manager.get_session(session_id)
        original_activity = session.last_activity

        # Small delay to ensure different timestamp
        await asyncio.sleep(0.01)
        session_manager.update_activity(session_id)

        assert session.last_activity > original_activity

    @pytest.mark.asyncio
    async def test_increment_request_count(self, session_manager, mock_transport):
        """Should increment request count and update activity."""
        session_id = "request-count-session"
        await session_manager.create_session(
            session_id=session_id,
            transport=mock_transport,
        )

        session_manager.increment_request_count(session_id)
        session_manager.increment_request_count(session_id)
        session_manager.increment_request_count(session_id)

        session = session_manager.get_session(session_id)
        assert session.request_count == 3

    @pytest.mark.asyncio
    async def test_increment_error_count(self, session_manager, mock_transport):
        """Should increment error count."""
        session_id = "error-count-session"
        await session_manager.create_session(
            session_id=session_id,
            transport=mock_transport,
        )

        session_manager.increment_error_count(session_id)
        session_manager.increment_error_count(session_id)

        session = session_manager.get_session(session_id)
        assert session.error_count == 2

    def test_update_activity_nonexistent_session(self, session_manager):
        """Should handle update_activity for non-existent session gracefully."""
        # Should not raise
        session_manager.update_activity("non-existent")

    def test_increment_count_nonexistent_session(self, session_manager):
        """Should handle increment counts for non-existent session gracefully."""
        # Should not raise
        session_manager.increment_request_count("non-existent")
        session_manager.increment_error_count("non-existent")


# Stale Session Cleanup Tests


class TestStaleSessionCleanup:
    """Test suite for stale session cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_stale_sessions(self, session_manager):
        """Should remove sessions exceeding max lifetime."""
        # Create sessions with mocked timestamps
        mock_transport = MagicMock()

        await session_manager.create_session(
            session_id="stale-session",
            transport=mock_transport,
        )

        # Manually set created_at to simulate old session
        session = session_manager.get_session("stale-session")
        session.created_at = time.monotonic() - 4000  # 4000 seconds ago

        # Create a fresh session
        await session_manager.create_session(
            session_id="fresh-session",
            transport=mock_transport,
        )

        # Cleanup with 3600 second lifetime
        cleaned = await session_manager.cleanup_stale_sessions(3600)

        assert cleaned == 1
        assert session_manager.get_session("stale-session") is None
        assert session_manager.get_session("fresh-session") is not None

    @pytest.mark.asyncio
    async def test_cleanup_stale_sessions_none_stale(self, session_manager, mock_transport):
        """Should return 0 when no sessions are stale."""
        await session_manager.create_session(
            session_id="fresh-session",
            transport=mock_transport,
        )

        cleaned = await session_manager.cleanup_stale_sessions(3600)

        assert cleaned == 0
        assert session_manager.active_session_count == 1


# Close All Sessions Tests


class TestCloseAllSessions:
    """Test suite for close_all_sessions functionality."""

    @pytest.mark.asyncio
    async def test_close_all_sessions(self, session_manager):
        """Should close all active sessions."""
        # Create multiple sessions
        for i in range(5):
            await session_manager.create_session(
                session_id=f"close-session-{i}",
                transport=MagicMock(),
            )

        assert session_manager.active_session_count == 5

        closed = await session_manager.close_all_sessions()

        assert closed == 5
        assert session_manager.active_session_count == 0

    @pytest.mark.asyncio
    async def test_close_all_sessions_empty(self, session_manager):
        """Should return 0 when no sessions exist."""
        closed = await session_manager.close_all_sessions()
        assert closed == 0


# Heartbeat Tests


class TestHeartbeatManagement:
    """Test suite for heartbeat task management."""

    @pytest.mark.asyncio
    async def test_start_heartbeat_on_session_create(self, session_manager, mock_transport):
        """Should start heartbeat task when requested."""
        await session_manager.create_session(
            session_id="heartbeat-session",
            transport=mock_transport,
            start_heartbeat=True,
        )

        # Heartbeat task should be created
        assert "heartbeat-session" in session_manager._heartbeat_tasks

        # Clean up
        await session_manager.remove_session("heartbeat-session")

    @pytest.mark.asyncio
    async def test_stop_heartbeat_on_session_remove(self, session_manager, mock_transport):
        """Should cancel heartbeat task when session is removed."""
        await session_manager.create_session(
            session_id="heartbeat-stop-session",
            transport=mock_transport,
            start_heartbeat=True,
        )

        await session_manager.remove_session("heartbeat-stop-session")

        assert "heartbeat-stop-session" not in session_manager._heartbeat_tasks

    @pytest.mark.asyncio
    async def test_heartbeat_loop_updates_activity(self, session_manager, mock_transport):
        """Should update last_activity during heartbeat."""
        # Use short heartbeat interval for testing
        manager = SessionManager(heartbeat_interval=1)

        await manager.create_session(
            session_id="heartbeat-activity-session",
            transport=mock_transport,
            start_heartbeat=True,
        )

        session = manager.get_session("heartbeat-activity-session")
        original_activity = session.last_activity

        # Wait for heartbeat to run
        await asyncio.sleep(1.5)

        # Activity should be updated
        assert session.last_activity > original_activity

        # Clean up
        await manager.close_all_sessions()

    @pytest.mark.asyncio
    async def test_set_heartbeat_sender(self, session_manager, mock_transport):
        """Should set heartbeat sender callback on session."""
        session_id = "heartbeat-sender-session"
        await session_manager.create_session(
            session_id=session_id,
            transport=mock_transport,
        )

        # Create a mock heartbeat sender
        heartbeat_called = False

        async def mock_heartbeat_sender():
            nonlocal heartbeat_called
            heartbeat_called = True

        # Set the heartbeat sender
        result = session_manager.set_heartbeat_sender(session_id, mock_heartbeat_sender)
        assert result is True

        # Verify it's set on the session
        session = session_manager.get_session(session_id)
        assert session.heartbeat_sender is mock_heartbeat_sender

        # Clean up
        await session_manager.remove_session(session_id)

    @pytest.mark.asyncio
    async def test_set_heartbeat_sender_nonexistent_session(self, session_manager):
        """Should return False when setting sender for non-existent session."""

        async def mock_sender():
            pass

        result = session_manager.set_heartbeat_sender("nonexistent", mock_sender)
        assert result is False

    @pytest.mark.asyncio
    async def test_heartbeat_calls_sender(self, mock_transport):
        """Should call heartbeat sender when heartbeat fires."""
        # Use short heartbeat interval for testing
        manager = SessionManager(heartbeat_interval=1)
        session_id = "heartbeat-calls-sender"

        await manager.create_session(
            session_id=session_id,
            transport=mock_transport,
            start_heartbeat=True,
        )

        # Track heartbeat calls
        heartbeat_count = 0

        async def counting_heartbeat_sender():
            nonlocal heartbeat_count
            heartbeat_count += 1

        # Set the heartbeat sender
        manager.set_heartbeat_sender(session_id, counting_heartbeat_sender)

        # Wait for at least one heartbeat
        await asyncio.sleep(1.5)

        # Heartbeat sender should have been called
        assert heartbeat_count >= 1

        # Clean up
        await manager.close_all_sessions()

    @pytest.mark.asyncio
    async def test_heartbeat_removes_session_on_sender_failure(self, mock_transport):
        """Should remove session when heartbeat sender raises exception."""
        # Use short heartbeat interval for testing
        manager = SessionManager(heartbeat_interval=1)
        session_id = "heartbeat-failure-session"

        await manager.create_session(
            session_id=session_id,
            transport=mock_transport,
            start_heartbeat=True,
        )

        # Create a failing heartbeat sender
        async def failing_heartbeat_sender():
            raise ConnectionError("Connection lost")

        # Set the failing heartbeat sender
        manager.set_heartbeat_sender(session_id, failing_heartbeat_sender)

        # Wait for heartbeat to fire and fail
        await asyncio.sleep(1.5)

        # Session should be removed due to failed heartbeat
        assert manager.get_session(session_id) is None

    @pytest.mark.asyncio
    async def test_heartbeat_skips_when_no_sender(self, mock_transport):
        """Should skip heartbeat when no sender is set."""
        # Use short heartbeat interval for testing
        manager = SessionManager(heartbeat_interval=1)
        session_id = "no-sender-session"

        await manager.create_session(
            session_id=session_id,
            transport=mock_transport,
            start_heartbeat=True,
        )

        # Don't set a heartbeat sender

        session = manager.get_session(session_id)
        original_activity = session.last_activity

        # Wait for heartbeat to run
        await asyncio.sleep(1.5)

        # Activity should still be updated even without sender
        assert session.last_activity > original_activity

        # Session should still exist
        assert manager.get_session(session_id) is not None

        # Clean up
        await manager.close_all_sessions()


# Metrics Tests


class TestConnectionMetrics:
    """Test suite for ConnectionMetrics tracking."""

    def test_metrics_initial_state(self, session_manager):
        """Should start with zero metrics."""
        metrics = session_manager.get_metrics()

        assert metrics.total_connections_created == 0
        assert metrics.total_connections_closed == 0
        assert metrics.current_active_connections == 0
        assert metrics.peak_connections == 0
        assert metrics.total_requests_processed == 0
        assert metrics.total_errors == 0

    def test_metrics_to_dict(self):
        """Should convert metrics to dictionary."""
        metrics = ConnectionMetrics(
            total_connections_created=10,
            total_connections_closed=5,
            current_active_connections=5,
            peak_connections=8,
            total_requests_processed=100,
            total_errors=3,
        )

        result = metrics.to_dict()

        assert result["total_connections_created"] == 10
        assert result["total_connections_closed"] == 5
        assert result["current_active_connections"] == 5
        assert result["peak_connections"] == 8
        assert result["total_requests_processed"] == 100
        assert result["total_errors"] == 3


# Session State Tests


class TestSessionState:
    """Test suite for SessionState dataclass."""

    def test_session_state_creation(self, mock_transport):
        """Should create SessionState with all fields."""
        state = SessionState(
            session_id="test-state",
            transport=mock_transport,
            created_at=100.0,
            last_activity=100.0,
            client_ip="192.168.1.1",
            request_count=5,
            error_count=1,
        )

        assert state.session_id == "test-state"
        assert state.transport is mock_transport
        assert state.created_at == 100.0
        assert state.last_activity == 100.0
        assert state.client_ip == "192.168.1.1"
        assert state.request_count == 5
        assert state.error_count == 1

    def test_session_state_defaults(self, mock_transport):
        """Should use default values for optional fields."""
        state = SessionState(
            session_id="default-state",
            transport=mock_transport,
            created_at=100.0,
            last_activity=100.0,
        )

        assert state.client_ip == ""
        assert state.request_count == 0
        assert state.error_count == 0


# Session ID Generation Tests


class TestSessionIdGeneration:
    """Test suite for session ID generation."""

    def test_generate_session_id_unique(self):
        """Should generate unique session IDs."""
        ids = [generate_session_id() for _ in range(100)]
        unique_ids = set(ids)

        assert len(unique_ids) == 100

    def test_generate_session_id_format(self):
        """Should generate URL-safe session IDs."""
        session_id = generate_session_id()

        # URL-safe base64 characters
        valid_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")

        for char in session_id:
            assert char in valid_chars

    def test_generate_session_id_length(self):
        """Should generate session IDs of appropriate length."""
        session_id = generate_session_id()

        # 32 bytes -> ~43 characters in base64
        assert len(session_id) >= 40
        assert len(session_id) <= 50


# Concurrency Tests


class TestConcurrency:
    """Test suite for concurrent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_session_creation(self, session_manager):
        """Should handle concurrent session creation safely."""

        async def create_session(i):
            await session_manager.create_session(
                session_id=f"concurrent-session-{i}",
                transport=MagicMock(),
            )

        # Create 20 sessions concurrently
        await asyncio.gather(*[create_session(i) for i in range(20)])

        assert session_manager.active_session_count == 20
        metrics = session_manager.get_metrics()
        assert metrics.total_connections_created == 20

    @pytest.mark.asyncio
    async def test_concurrent_session_removal(self, session_manager):
        """Should handle concurrent session removal safely."""
        # Create sessions first
        for i in range(20):
            await session_manager.create_session(
                session_id=f"remove-concurrent-{i}",
                transport=MagicMock(),
            )

        async def remove_session(i):
            await session_manager.remove_session(f"remove-concurrent-{i}")

        # Remove all concurrently
        await asyncio.gather(*[remove_session(i) for i in range(20)])

        assert session_manager.active_session_count == 0
        metrics = session_manager.get_metrics()
        assert metrics.total_connections_closed == 20

    @pytest.mark.asyncio
    async def test_concurrent_mixed_operations(self, session_manager):
        """Should handle mixed concurrent operations safely."""

        async def create_and_remove(i):
            sid = f"mixed-session-{i}"
            await session_manager.create_session(
                session_id=sid,
                transport=MagicMock(),
            )
            session_manager.increment_request_count(sid)
            await asyncio.sleep(0.001)  # Small delay
            await session_manager.remove_session(sid)

        # Run mixed operations concurrently
        await asyncio.gather(*[create_and_remove(i) for i in range(10)])

        assert session_manager.active_session_count == 0
