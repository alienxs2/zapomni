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
from typing import TYPE_CHECKING, List, Set

import structlog
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Mount, Route
from starlette.types import ASGIApp, Receive, Scope, Send

from zapomni_mcp.config import SSEConfig

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
    - POST /messages: Message handling endpoint
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

    bound_logger = logger.bind(component="SSETransport")

    # Create a single shared SSE transport for all sessions
    # The SDK will manage multiple sessions internally using UUID-based session IDs
    # Use /messages/ (with trailing slash) to match the Mount route
    sse_transport = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> Response:
        """
        Handle SSE connection establishment.

        This endpoint:
        1. Uses the shared SseServerTransport to establish SSE connection
        2. The SDK creates a UUID session_id internally and manages it
        3. Connects the SSE stream to the MCP server
        4. Runs the MCP server with the SSE streams
        5. Cleans up when the connection closes

        Args:
            request: Starlette request object

        Returns:
            SSE response stream
        """
        # Get client IP for logging
        client_ip = request.client.host if request.client else "unknown"

        bound_logger.info(
            "SSE connection requested",
            client_ip=client_ip,
        )

        try:
            # Connect SSE and run MCP server
            # The SDK will create a UUID session_id internally
            async with sse_transport.connect_sse(
                request.scope,
                request.receive,
                request._send,
            ) as streams:
                read_stream, write_stream = streams

                bound_logger.info(
                    "SSE connection established",
                    client_ip=client_ip,
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
                client_ip=client_ip,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

        finally:
            bound_logger.info(
                "SSE connection closed",
                client_ip=client_ip,
            )

        # Return empty response - SSE streams are handled above
        return Response()

    # No wrapper needed - we'll pass the SDK's handler directly to the route

    async def handle_health(request: Request) -> Response:
        """
        Health check endpoint for monitoring and service verification.

        Returns:
            JSON response with health status:
            - status: "healthy" or "unhealthy"
            - version: Server version
            - transport: Transport type (sse)
            - uptime_seconds: Server uptime in seconds
            - database_pool: Connection pool statistics (if available)
        """
        try:
            uptime_seconds = time.time() - startup_time

            health_data = {
                "status": "healthy",
                "version": __version__,
                "transport": "sse",
                "uptime_seconds": round(uptime_seconds, 2),
            }

            # Add database pool statistics if available
            try:
                if hasattr(mcp_server, "_core_engine") and mcp_server._core_engine is not None:
                    core_engine = mcp_server._core_engine
                    if hasattr(core_engine, "db_client") and core_engine.db_client is not None:
                        db_client = core_engine.db_client
                        if hasattr(db_client, "get_pool_stats"):
                            pool_stats = await db_client.get_pool_stats()
                            health_data["database_pool"] = {
                                "max_connections": pool_stats.get("max_connections", 0),
                                "active_connections": pool_stats.get("active_connections", 0),
                                "total_queries": pool_stats.get("total_queries", 0),
                                "total_retries": pool_stats.get("total_retries", 0),
                                "utilization_percent": pool_stats.get("utilization_percent", 0.0),
                                "initialized": pool_stats.get("initialized", False),
                                "closed": pool_stats.get("closed", False),
                            }
            except Exception as pool_error:
                bound_logger.debug(
                    "Could not get pool stats for health check",
                    error=str(pool_error),
                )

            bound_logger.debug(
                "Health check",
                status="healthy",
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

    async def handle_dashboard(request: Request) -> Response:
        """Serve dashboard HTML page."""
        from pathlib import Path

        dashboard_path = Path(__file__).parent / "ui" / "dashboard.html"
        if dashboard_path.exists():
            return Response(dashboard_path.read_text(), media_type="text/html")
        return Response("Dashboard not found", status_code=404)

    async def handle_graph_api(request: Request) -> Response:
        """Return graph data for visualization."""
        workspace_id = request.query_params.get("workspace", "default")
        limit = int(request.query_params.get("limit", "100"))

        # Simple graph query - get Memory and Chunk nodes
        try:
            db_client = mcp_server._core_engine.db_client
            cypher = """
            MATCH (m:Memory)-[r:HAS_CHUNK]->(c:Chunk)
            WHERE m.workspace_id = $workspace_id
            RETURN m, r, c
            LIMIT $limit
            """
            result = await db_client._execute_cypher(
                cypher, {"workspace_id": workspace_id, "limit": limit}
            )

            nodes = []
            edges = []
            seen_nodes = set()

            for row in result.rows:
                # Extract Memory node
                m_node = row.get("m")
                if m_node and hasattr(m_node, "properties"):
                    m_props = m_node.properties
                    m_id = str(m_props.get("id", "unknown"))
                    if m_id not in seen_nodes:
                        nodes.append({"id": m_id, "label": f"Memory {m_id[:8]}", "type": "Memory"})
                        seen_nodes.add(m_id)

                # Extract Chunk node
                c_node = row.get("c")
                if c_node and hasattr(c_node, "properties"):
                    c_props = c_node.properties
                    c_id = str(c_props.get("id", "unknown"))
                    if c_id not in seen_nodes:
                        nodes.append({"id": c_id, "label": f"Chunk {c_id[:8]}", "type": "Chunk"})
                        seen_nodes.add(c_id)

                # Add edge if both nodes exist
                if m_node and c_node:
                    edges.append(
                        {
                            "id": f"{m_id}-{c_id}",
                            "source": m_id,
                            "target": c_id,
                            "type": "HAS_CHUNK",
                        }
                    )

            return Response(
                json.dumps({"nodes": nodes, "edges": edges, "workspace": workspace_id}),
                media_type="application/json",
            )
        except Exception as e:
            bound_logger.error("graph_api_error", error=str(e))
            return Response(
                json.dumps({"nodes": [], "edges": [], "error": str(e)}),
                media_type="application/json",
                status_code=500,
            )

    # Define routes
    # Note: We handle /messages with two routes - one for trailing slash, one without
    # Both delegate to the SDK's ASGI handler
    routes = [
        Route("/", endpoint=handle_dashboard, methods=["GET"]),
        Route("/api/graph", endpoint=handle_graph_api, methods=["GET"]),
        Route("/sse", endpoint=handle_sse, methods=["GET"]),
        # Mount the SDK handler at /messages/  (with trailing slash)
        # This avoids redirect issues while still using the SDK's ASGI app
        Mount("/messages/", app=sse_transport.handle_post_message),
        Route("/health", endpoint=handle_health, methods=["GET"]),
    ]

    bound_logger.info(
        "SSE application created",
        host=config.host,
        port=config.port,
        cors_origins=config.cors_origins,
        dns_rebinding_protection=config.dns_rebinding_protection,
        routes=["/sse", "/messages", "/health"],
    )

    return Starlette(routes=routes, middleware=middleware)


__all__ = [
    "create_sse_app",
    "SSEConfig",
    "TransportSecuritySettings",
    "DNSRebindingProtectionMiddleware",
    "DEFAULT_ALLOWED_HOSTS",
]
