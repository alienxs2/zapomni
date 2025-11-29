"""
Session Manager for SSE Transport.

Manages the lifecycle of SSE sessions including creation, tracking, and cleanup.
Thread-safe implementation using asyncio locks for concurrent access.

Features:
- Session lifecycle management (create, track, remove)
- Connection heartbeat for detecting stale connections
- Connection metrics tracking (active, total, peak, errors)
- Graceful shutdown with close_all_sessions()

Copyright (c) 2025 Goncharenko Anton aka alienxs2
License: MIT
"""

import asyncio
import secrets
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Dict, Optional

import structlog

from zapomni_db.models import DEFAULT_WORKSPACE_ID

if TYPE_CHECKING:
    from mcp.server.sse import SseServerTransport

# Type alias for async heartbeat sender callback
HeartbeatSender = Callable[[], Coroutine[Any, Any, None]]


logger = structlog.get_logger(__name__)


@dataclass
class ConnectionMetrics:
    """
    Metrics for SSE connection tracking.

    Attributes:
        total_connections_created: Total number of sessions created since server start
        total_connections_closed: Total number of sessions closed since server start
        current_active_connections: Number of currently active sessions
        peak_connections: Maximum concurrent connections seen
        total_requests_processed: Total number of requests processed across all sessions
        total_errors: Total number of errors across all sessions
    """

    total_connections_created: int = 0
    total_connections_closed: int = 0
    current_active_connections: int = 0
    peak_connections: int = 0
    total_requests_processed: int = 0
    total_errors: int = 0

    def to_dict(self) -> Dict[str, int]:
        """Convert metrics to dictionary."""
        return {
            "total_connections_created": self.total_connections_created,
            "total_connections_closed": self.total_connections_closed,
            "current_active_connections": self.current_active_connections,
            "peak_connections": self.peak_connections,
            "total_requests_processed": self.total_requests_processed,
            "total_errors": self.total_errors,
        }


@dataclass
class SessionState:
    """
    State for a single SSE session.

    Attributes:
        session_id: Unique identifier for the session
        transport: SseServerTransport instance for this session
        created_at: Monotonic time when session was created
        last_activity: Monotonic time of last activity
        client_ip: IP address of the client
        request_count: Number of requests processed in this session
        error_count: Number of errors encountered in this session
        heartbeat_sender: Optional callback to send heartbeat to client
        current_workspace_id: Current workspace for this session (defaults to "default")
    """

    session_id: str
    transport: "SseServerTransport"
    created_at: float
    last_activity: float
    client_ip: str = ""
    request_count: int = 0
    error_count: int = 0
    heartbeat_sender: Optional[HeartbeatSender] = None
    current_workspace_id: str = DEFAULT_WORKSPACE_ID


class SessionManager:
    """
    Manages SSE session lifecycle.

    This class is responsible for:
    - Creating new sessions with cryptographically secure IDs
    - Tracking active sessions
    - Providing thread-safe access to session data
    - Cleaning up stale sessions
    - Managing connection heartbeats to detect dead connections
    - Tracking connection metrics (active, total, peak, errors)
    - Graceful shutdown with close_all_sessions()
    - Logging session events

    Thread Safety:
        All mutating operations are protected by an asyncio lock to ensure
        thread-safe access from multiple coroutines.

    Example:
        ```python
        manager = SessionManager(heartbeat_interval=30)

        # Create a new session
        session_id = await manager.create_session(transport)

        # Get session for processing
        session = manager.get_session(session_id)
        if session:
            await process_message(session.transport)

        # Clean up when done
        await manager.remove_session(session_id)

        # On shutdown
        await manager.close_all_sessions()
        ```
    """

    def __init__(self, heartbeat_interval: int = 30) -> None:
        """
        Initialize session manager with empty session registry.

        Args:
            heartbeat_interval: Interval in seconds for heartbeat pings (default: 30)
        """
        self._sessions: Dict[str, SessionState] = {}
        self._lock = asyncio.Lock()
        self._logger = structlog.get_logger(__name__).bind(component="SessionManager")

        # Heartbeat configuration
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat_tasks: Dict[str, asyncio.Task[None]] = {}

        # Connection metrics
        self._metrics = ConnectionMetrics()
        self._metrics_lock = asyncio.Lock()

    async def create_session(
        self,
        session_id: str,
        transport: "SseServerTransport",
        client_ip: str = "",
        start_heartbeat: bool = False,
    ) -> None:
        """
        Create and register a new session.

        Args:
            session_id: Unique identifier for the session
            transport: SseServerTransport instance for this session
            client_ip: IP address of the client (optional)
            start_heartbeat: Whether to start heartbeat task for this session (optional)

        Raises:
            ValueError: If session_id already exists
        """
        current_time = time.monotonic()

        async with self._lock:
            if session_id in self._sessions:
                raise ValueError(f"Session {session_id} already exists")

            session_state = SessionState(
                session_id=session_id,
                transport=transport,
                created_at=current_time,
                last_activity=current_time,
                client_ip=client_ip,
            )
            self._sessions[session_id] = session_state

        # Update metrics
        async with self._metrics_lock:
            self._metrics.total_connections_created += 1
            self._metrics.current_active_connections = len(self._sessions)
            if self._metrics.current_active_connections > self._metrics.peak_connections:
                self._metrics.peak_connections = self._metrics.current_active_connections

        # Optionally start heartbeat task
        if start_heartbeat and self._heartbeat_interval > 0:
            await self._start_heartbeat(session_id)

        self._logger.info(
            "Session created",
            session_id=session_id,
            client_ip=client_ip,
            active_sessions=len(self._sessions),
            heartbeat_enabled=start_heartbeat,
        )

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """
        Get session by ID.

        This is a non-blocking read operation that returns the session
        state if found, or None if the session doesn't exist.

        Args:
            session_id: Unique identifier of the session

        Returns:
            SessionState if found, None otherwise
        """
        return self._sessions.get(session_id)

    async def remove_session(self, session_id: str) -> bool:
        """
        Remove session and cleanup resources.

        Args:
            session_id: Unique identifier of the session to remove

        Returns:
            True if session was removed, False if not found
        """
        # Cancel heartbeat task first
        await self._stop_heartbeat(session_id)

        async with self._lock:
            session = self._sessions.pop(session_id, None)

        if session is not None:
            # Calculate session duration
            duration = time.monotonic() - session.created_at

            # Update metrics
            async with self._metrics_lock:
                self._metrics.total_connections_closed += 1
                self._metrics.current_active_connections = len(self._sessions)
                self._metrics.total_requests_processed += session.request_count
                self._metrics.total_errors += session.error_count

            self._logger.info(
                "Session removed",
                session_id=session_id,
                duration_seconds=round(duration, 2),
                request_count=session.request_count,
                error_count=session.error_count,
                active_sessions=len(self._sessions),
            )
            return True

        self._logger.debug("Session not found for removal", session_id=session_id)
        return False

    async def get_active_count(self) -> int:
        """
        Get number of active sessions.

        Returns:
            Number of currently active sessions
        """
        return len(self._sessions)

    @property
    def active_session_count(self) -> int:
        """
        Get number of active sessions (synchronous property).

        Returns:
            Number of currently active sessions
        """
        return len(self._sessions)

    def update_activity(self, session_id: str) -> None:
        """
        Update last activity timestamp for a session.

        This is used to track session activity for cleanup purposes.

        Args:
            session_id: Unique identifier of the session
        """
        session = self._sessions.get(session_id)
        if session:
            session.last_activity = time.monotonic()

    def increment_request_count(self, session_id: str) -> None:
        """
        Increment request count for a session.

        Args:
            session_id: Unique identifier of the session
        """
        session = self._sessions.get(session_id)
        if session:
            session.request_count += 1
            session.last_activity = time.monotonic()

    def increment_error_count(self, session_id: str) -> None:
        """
        Increment error count for a session.

        Args:
            session_id: Unique identifier of the session
        """
        session = self._sessions.get(session_id)
        if session:
            session.error_count += 1

    def set_heartbeat_sender(self, session_id: str, heartbeat_sender: HeartbeatSender) -> bool:
        """
        Set the heartbeat sender callback for a session.

        The heartbeat sender is an async callback that sends SSE data to keep
        the connection alive. This should be called after the SSE streams are
        established.

        Args:
            session_id: Unique identifier of the session
            heartbeat_sender: Async callback that sends heartbeat to client

        Returns:
            True if session found and sender set, False otherwise
        """
        session = self._sessions.get(session_id)
        if session:
            session.heartbeat_sender = heartbeat_sender
            self._logger.debug(
                "Heartbeat sender set",
                session_id=session_id,
            )
            return True
        return False

    def get_workspace_id(self, session_id: str) -> str:
        """
        Get the current workspace ID for a session.

        Args:
            session_id: Unique identifier of the session

        Returns:
            Current workspace ID, or DEFAULT_WORKSPACE_ID if session not found
        """
        session = self._sessions.get(session_id)
        if session:
            return session.current_workspace_id
        return DEFAULT_WORKSPACE_ID

    def set_workspace_id(self, session_id: str, workspace_id: str) -> bool:
        """
        Set the current workspace ID for a session.

        Args:
            session_id: Unique identifier of the session
            workspace_id: New workspace ID to set

        Returns:
            True if session found and workspace set, False otherwise
        """
        session = self._sessions.get(session_id)
        if session:
            old_workspace = session.current_workspace_id
            session.current_workspace_id = workspace_id
            self._logger.info(
                "Workspace changed",
                session_id=session_id,
                old_workspace=old_workspace,
                new_workspace=workspace_id,
            )
            return True
        return False

    async def cleanup_stale_sessions(self, max_lifetime_seconds: int) -> int:
        """
        Remove sessions that have exceeded maximum lifetime.

        Args:
            max_lifetime_seconds: Maximum session lifetime in seconds

        Returns:
            Number of sessions cleaned up
        """
        current_time = time.monotonic()
        stale_sessions = []

        async with self._lock:
            for session_id, session in list(self._sessions.items()):
                if current_time - session.created_at > max_lifetime_seconds:
                    stale_sessions.append(session_id)
                    del self._sessions[session_id]

        for session_id in stale_sessions:
            self._logger.info(
                "Stale session cleaned up",
                session_id=session_id,
                max_lifetime_seconds=max_lifetime_seconds,
            )

        if stale_sessions:
            self._logger.info(
                "Stale session cleanup completed",
                cleaned_count=len(stale_sessions),
                remaining_sessions=len(self._sessions),
            )

        return len(stale_sessions)

    async def get_all_session_ids(self) -> list[str]:
        """
        Get list of all active session IDs.

        Returns:
            List of session IDs
        """
        return list(self._sessions.keys())

    def get_metrics(self) -> ConnectionMetrics:
        """
        Get current connection metrics.

        Returns:
            ConnectionMetrics object with current statistics
        """
        return self._metrics

    async def close_all_sessions(self) -> int:
        """
        Close all active sessions and cleanup resources.

        This method is used during graceful shutdown to ensure all
        sessions are properly closed and heartbeat tasks are cancelled.

        Returns:
            Number of sessions that were closed
        """
        self._logger.info(
            "Closing all sessions",
            active_sessions=len(self._sessions),
        )

        # Get all session IDs first
        session_ids = await self.get_all_session_ids()

        # Cancel all heartbeat tasks
        for session_id in list(self._heartbeat_tasks.keys()):
            await self._stop_heartbeat(session_id)

        # Remove all sessions
        closed_count = 0
        for session_id in session_ids:
            if await self.remove_session(session_id):
                closed_count += 1

        self._logger.info(
            "All sessions closed",
            closed_count=closed_count,
            remaining_sessions=len(self._sessions),
        )

        return closed_count

    async def _start_heartbeat(self, session_id: str) -> None:
        """
        Start heartbeat task for a session.

        The heartbeat sends SSE comments periodically to keep the connection
        alive and detect stale connections.

        Args:
            session_id: Session to start heartbeat for
        """
        if session_id in self._heartbeat_tasks:
            return  # Already running

        task = asyncio.create_task(
            self._heartbeat_loop(session_id),
            name=f"heartbeat_{session_id}",
        )
        self._heartbeat_tasks[session_id] = task

        self._logger.debug(
            "Heartbeat started",
            session_id=session_id,
            interval_seconds=self._heartbeat_interval,
        )

    async def _stop_heartbeat(self, session_id: str) -> None:
        """
        Stop heartbeat task for a session.

        Args:
            session_id: Session to stop heartbeat for
        """
        task = self._heartbeat_tasks.pop(session_id, None)
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

            self._logger.debug(
                "Heartbeat stopped",
                session_id=session_id,
            )

    async def _heartbeat_loop(self, session_id: str) -> None:
        """
        Send periodic heartbeats to keep connection alive.

        Heartbeats are sent via the heartbeat_sender callback if available.
        The callback should send SSE data (typically comments starting with ':')
        to keep the connection alive and detect stale connections.

        Args:
            session_id: Session to send heartbeats to
        """
        try:
            while session_id in self._sessions:
                await asyncio.sleep(self._heartbeat_interval)

                session = self._sessions.get(session_id)
                if not session:
                    break

                try:
                    # Send heartbeat via callback if available
                    if session.heartbeat_sender is not None:
                        await session.heartbeat_sender()
                        self._logger.debug(
                            "Heartbeat sent",
                            session_id=session_id,
                        )
                    else:
                        # No sender available yet, just log
                        self._logger.debug(
                            "Heartbeat skipped (no sender)",
                            session_id=session_id,
                        )

                    # Update activity timestamp
                    session.last_activity = time.monotonic()

                except Exception as e:
                    self._logger.warning(
                        "Heartbeat failed",
                        session_id=session_id,
                        error=str(e),
                    )
                    # Connection may be dead, remove session
                    await self.remove_session(session_id)
                    break

        except asyncio.CancelledError:
            # Normal cancellation during shutdown
            pass
        except Exception as e:
            self._logger.error(
                "Heartbeat loop error",
                session_id=session_id,
                error=str(e),
            )


def generate_session_id() -> str:
    """
    Generate cryptographically secure session ID.

    Uses secrets.token_urlsafe to generate a 256-bit random token
    that is safe for use in URLs.

    Returns:
        URL-safe session ID string
    """
    return secrets.token_urlsafe(32)


__all__ = [
    "SessionManager",
    "SessionState",
    "ConnectionMetrics",
    "HeartbeatSender",
    "generate_session_id",
]
