"""
SSE Transport implementation for Zapomni MCP Server.

Provides HTTP endpoints for SSE-based MCP communication:
- GET /sse: Establish SSE connection
- POST /messages/{session_id}: Receive JSON-RPC messages
- GET /health: Health check endpoint for monitoring

This module enables concurrent client connections via HTTP,
allowing multiple MCP clients to connect simultaneously.

Includes DNS rebinding protection middleware for security.

Copyright (c) 2025 Goncharenko Anton aka alienxs2
License: MIT
"""

import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Set

import anyio
import structlog
from mcp.server.sse import SseServerTransport
from mcp.shared.message import SessionMessage
from mcp.types import JSONRPCMessage, JSONRPCNotification
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from zapomni_mcp.config import SSEConfig
from zapomni_mcp.session_manager import SessionManager, generate_session_id

if TYPE_CHECKING:
    from zapomni_mcp.server import MCPServer


logger = structlog.get_logger(__name__)

# Server version for health endpoint
__version__ = "0.2.0"

# Default allowed hosts for DNS rebinding protection
DEFAULT_ALLOWED_HOSTS: List[str] = ["localhost", "127.0.0.1", "::1"]


@dataclass
class TransportSecuritySettings:
    """
    Security settings for SSE transport.

    Attributes:
        allowed_hosts: List of allowed Host header values for DNS rebinding protection.
            Defaults to localhost variants for local-only binding.
        check_origin: Whether to validate the Host header. Default True.
    """

    allowed_hosts: List[str] = field(default_factory=lambda: DEFAULT_ALLOWED_HOSTS.copy())
    check_origin: bool = True


class DNSRebindingProtectionMiddleware:
    """
    ASGI middleware for DNS rebinding attack protection.

    DNS rebinding attacks allow malicious websites to bypass same-origin policy
    by manipulating DNS. This middleware validates the Host header against a
    whitelist of allowed hosts to prevent such attacks.

    When deployed on localhost (127.0.0.1), the middleware automatically allows:
    - localhost
    - 127.0.0.1
    - ::1 (IPv6 localhost)

    For non-localhost deployments, allowed_hosts must be explicitly configured.
    """

    def __init__(self, app: ASGIApp, allowed_hosts: List[str]) -> None:
        """
        Initialize the DNS rebinding protection middleware.

        Args:
            app: The ASGI application to wrap.
            allowed_hosts: List of allowed Host header values (without port).
        """
        self.app = app
        self.allowed_hosts: Set[str] = set(host.lower() for host in allowed_hosts)
        self._logger = logger.bind(component="DNSRebindingProtection")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """
        Process the request and validate Host header.

        Args:
            scope: ASGI scope dictionary.
            receive: ASGI receive callable.
            send: ASGI send callable.
        """
        if scope["type"] == "http":
            # Extract Host header from scope
            headers = dict(scope.get("headers", []))
            host_header = headers.get(b"host", b"").decode("utf-8", errors="ignore")

            # Extract hostname, handling both IPv4 and IPv6
            # IPv6 addresses may be in bracket notation: [::1]:8000
            host = self._extract_hostname(host_header)

            if host not in self.allowed_hosts:
                self._logger.warning(
                    "DNS rebinding protection blocked request",
                    blocked_host=host,
                    allowed_hosts=list(self.allowed_hosts),
                )

                # Send 403 Forbidden response
                response = Response(
                    content="Host not allowed",
                    status_code=403,
                    media_type="text/plain",
                )
                await response(scope, receive, send)
                return

        # Host is allowed, proceed with the application
        await self.app(scope, receive, send)

    @staticmethod
    def _extract_hostname(host_header: str) -> str:
        """
        Extract hostname from Host header, handling IPv6 bracket notation.

        Args:
            host_header: The Host header value (e.g., "localhost:8000", "[::1]:8000")

        Returns:
            The hostname in lowercase, stripped of port and brackets.
        """
        host = host_header.lower()

        # Handle IPv6 addresses in bracket notation: [::1]:8000
        if host.startswith("["):
            # Find the closing bracket
            bracket_end = host.find("]")
            if bracket_end != -1:
                # Extract the IPv6 address without brackets
                return host[1:bracket_end]
            # Malformed, return as-is
            return host

        # Handle regular IPv4 or hostname with optional port
        return host.split(":")[0]


def create_sse_app(mcp_server: "MCPServer", config: SSEConfig) -> Starlette:
    """
    Create Starlette application with SSE routes.

    This factory function creates a fully configured Starlette application
    with SSE endpoints for MCP communication. The application includes:
    - GET /sse: SSE connection establishment endpoint
    - POST /messages/{session_id}: Message handling endpoint
    - GET /health: Health check endpoint for monitoring
    - CORS middleware for cross-origin requests

    Args:
        mcp_server: MCPServer instance (singleton) that handles tool execution
        config: SSE configuration settings

    Returns:
        Configured Starlette application ready to be run with uvicorn
    """
    # Track startup time for uptime calculation
    startup_time = time.time()

    session_manager = SessionManager(heartbeat_interval=config.heartbeat_interval)
    bound_logger = logger.bind(component="SSETransport")

    # Store session_manager reference on mcp_server for shutdown cleanup
    mcp_server._session_manager = session_manager

    async def handle_sse(request: Request) -> Response:
        """
        Handle SSE connection establishment.

        This endpoint:
        1. Creates a new SseServerTransport with a unique session endpoint
        2. Registers the session with SessionManager
        3. Connects the SSE stream to the MCP server
        4. Runs the MCP server with the SSE streams
        5. Cleans up the session when the connection closes

        Args:
            request: Starlette request object

        Returns:
            SSE response stream
        """
        # Generate unique session ID
        session_id = generate_session_id()

        # Get client IP for logging
        client_ip = request.client.host if request.client else "unknown"

        bound_logger.info(
            "SSE connection requested",
            session_id=session_id,
            client_ip=client_ip,
        )

        # Create SSE transport with session-specific endpoint
        sse_transport = SseServerTransport(f"/messages/{session_id}")

        # Register session with manager and start heartbeat
        await session_manager.create_session(
            session_id=session_id,
            transport=sse_transport,
            client_ip=client_ip,
            start_heartbeat=True,
        )

        try:
            # Connect SSE and run MCP server
            async with sse_transport.connect_sse(
                request.scope,
                request.receive,
                request._send,
            ) as streams:
                read_stream, write_stream = streams

                # Create heartbeat sender that sends SSE notification through write_stream
                async def heartbeat_sender() -> None:
                    """Send heartbeat notification through SSE stream."""
                    # Create a JSON-RPC notification for heartbeat
                    # Using 'notifications/heartbeat' as the method name
                    # This follows MCP notification naming convention
                    heartbeat_notification = JSONRPCNotification(
                        jsonrpc="2.0",
                        method="notifications/heartbeat",
                        params={"timestamp": time.time()},
                    )
                    heartbeat_message = SessionMessage(
                        message=JSONRPCMessage(heartbeat_notification)
                    )
                    await write_stream.send(heartbeat_message)

                # Register the heartbeat sender with the session
                session_manager.set_heartbeat_sender(session_id, heartbeat_sender)

                bound_logger.info(
                    "SSE connection established",
                    session_id=session_id,
                    active_sessions=session_manager.active_session_count,
                )

                # Run the MCP server with the SSE streams
                await mcp_server._server.run(
                    read_stream,
                    write_stream,
                    mcp_server._server.create_initialization_options(),
                )

        except Exception as e:
            bound_logger.error(
                "SSE connection error",
                session_id=session_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

        finally:
            # Clean up session
            await session_manager.remove_session(session_id)
            bound_logger.info(
                "SSE connection closed",
                session_id=session_id,
                remaining_sessions=session_manager.active_session_count,
            )

        # Return empty response - SSE streams are handled above
        return Response()

    async def handle_messages(request: Request) -> Response:
        """
        Handle POST messages to session.

        This endpoint:
        1. Extracts session_id from the path
        2. Looks up the transport via SessionManager
        3. Forwards the message to the transport
        4. Returns 202 Accepted on success

        Args:
            request: Starlette request object with session_id path parameter

        Returns:
            JSON response with status or error
        """
        session_id = request.path_params.get("session_id", "")

        if not session_id:
            return Response(
                content='{"error": "Session ID required"}',
                status_code=400,
                media_type="application/json",
            )

        # Look up session
        session = session_manager.get_session(session_id)
        if not session:
            bound_logger.debug(
                "Session not found",
                session_id=session_id,
            )
            return Response(
                content='{"error": "Session not found"}',
                status_code=404,
                media_type="application/json",
            )

        # Update activity and increment request count
        session_manager.increment_request_count(session_id)

        try:
            # Forward to transport for processing
            # Note: handle_post_message reads the body from receive internally
            await session.transport.handle_post_message(
                scope=request.scope,
                receive=request.receive,
                send=request._send,
            )

            return Response(
                content='{"status": "accepted"}',
                status_code=202,
                media_type="application/json",
            )

        except Exception as e:
            session_manager.increment_error_count(session_id)
            bound_logger.error(
                "Message handling error",
                session_id=session_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            return Response(
                content=f'{{"error": "Internal server error: {str(e)}"}}',
                status_code=500,
                media_type="application/json",
            )

    async def handle_health(request: Request) -> Response:
        """
        Health check endpoint for monitoring and service verification.

        Returns:
            JSON response with health status:
            - status: "healthy" or "unhealthy"
            - version: Server version
            - transport: Transport type (sse)
            - active_connections: Number of active SSE sessions
            - uptime_seconds: Server uptime in seconds
            - metrics: Connection metrics (total, peak, errors)
        """
        try:
            metrics = session_manager.get_metrics()
            uptime_seconds = time.time() - startup_time

            health_data = {
                "status": "healthy",
                "version": __version__,
                "transport": "sse",
                "active_connections": session_manager.active_session_count,
                "uptime_seconds": round(uptime_seconds, 2),
                "metrics": {
                    "total_connections_created": metrics.total_connections_created,
                    "total_connections_closed": metrics.total_connections_closed,
                    "peak_connections": metrics.peak_connections,
                    "total_requests_processed": metrics.total_requests_processed,
                    "total_errors": metrics.total_errors,
                },
            }

            bound_logger.debug(
                "Health check",
                status="healthy",
                active_connections=health_data["active_connections"],
            )

            return Response(
                content=json.dumps(health_data),
                status_code=200,
                media_type="application/json",
            )

        except Exception as e:
            bound_logger.error(
                "Health check failed",
                error=str(e),
                error_type=type(e).__name__,
            )

            error_data = {
                "status": "unhealthy",
                "version": __version__,
                "transport": "sse",
                "error": str(e),
            }

            return Response(
                content=json.dumps(error_data),
                status_code=503,
                media_type="application/json",
            )

    # Build middleware stack
    middleware = []

    # Add DNS rebinding protection middleware (runs first, before CORS)
    if config.dns_rebinding_protection:
        allowed_hosts = config.get_effective_allowed_hosts()
        middleware.append(
            Middleware(
                DNSRebindingProtectionMiddleware,
                allowed_hosts=allowed_hosts,
            )
        )
        bound_logger.info(
            "DNS rebinding protection enabled",
            allowed_hosts=allowed_hosts,
        )

    # CORS middleware configuration
    middleware.append(
        Middleware(
            CORSMiddleware,
            allow_origins=config.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
            expose_headers=["*"],
        )
    )

    # Define routes
    routes = [
        Route("/sse", endpoint=handle_sse, methods=["GET"]),
        Route("/messages/{session_id}", endpoint=handle_messages, methods=["POST"]),
        Route("/health", endpoint=handle_health, methods=["GET"]),
    ]

    bound_logger.info(
        "SSE application created",
        host=config.host,
        port=config.port,
        cors_origins=config.cors_origins,
        dns_rebinding_protection=config.dns_rebinding_protection,
        routes=["/sse", "/messages/{session_id}", "/health"],
    )

    return Starlette(routes=routes, middleware=middleware)


__all__ = [
    "create_sse_app",
    "SSEConfig",
    "TransportSecuritySettings",
    "DNSRebindingProtectionMiddleware",
    "DEFAULT_ALLOWED_HOSTS",
]
