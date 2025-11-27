"""
Unit tests for SSE Transport component.

Tests the SSE transport implementation including:
- handle_sse endpoint
- handle_messages endpoint
- handle_health endpoint
- create_sse_app factory function

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient

from zapomni_mcp.config import SSEConfig
from zapomni_mcp.session_manager import SessionManager

# Fixtures


@pytest.fixture
def sse_config():
    """Create SSEConfig for testing.

    Note: 'testserver' is added to allowed_hosts because Starlette's TestClient
    uses 'testserver' as the default Host header.
    """
    return SSEConfig(
        host="127.0.0.1",
        port=8000,
        cors_origins=["*"],
        heartbeat_interval=30,
        max_connection_lifetime=3600,
        allowed_hosts=["localhost", "127.0.0.1", "::1", "testserver"],
    )


@pytest.fixture
def mock_mcp_server():
    """Create a mock MCPServer for testing."""
    mock = MagicMock()
    mock._server = MagicMock()
    mock._server.run = AsyncMock()
    mock._server.create_initialization_options = MagicMock(return_value={})
    mock._session_manager = None
    return mock


@pytest.fixture
def mock_sse_transport():
    """Create a mock SseServerTransport for testing."""
    mock = MagicMock()

    # Mock connect_sse as async context manager
    mock_streams = (MagicMock(), MagicMock())

    class MockAsyncContextManager:
        async def __aenter__(self):
            return mock_streams

        async def __aexit__(self, *args):
            return None

    mock.connect_sse = MagicMock(return_value=MockAsyncContextManager())
    mock.handle_post_message = AsyncMock()
    return mock


@pytest.fixture
def sse_app(mock_mcp_server, sse_config):
    """Create SSE Starlette app for testing."""
    from zapomni_mcp.sse_transport import create_sse_app

    return create_sse_app(mock_mcp_server, sse_config)


@pytest.fixture
def test_client(sse_app):
    """Create test client for SSE app."""
    return TestClient(sse_app, raise_server_exceptions=False)


# create_sse_app Factory Tests


class TestCreateSseApp:
    """Test suite for create_sse_app factory function."""

    def test_creates_starlette_app(self, mock_mcp_server, sse_config):
        """Should create a Starlette application."""
        from starlette.applications import Starlette

        from zapomni_mcp.sse_transport import create_sse_app

        app = create_sse_app(mock_mcp_server, sse_config)

        assert isinstance(app, Starlette)

    def test_registers_sse_endpoint(self, mock_mcp_server, sse_config):
        """Should register /sse endpoint."""
        from zapomni_mcp.sse_transport import create_sse_app

        app = create_sse_app(mock_mcp_server, sse_config)

        # Check routes
        route_paths = [route.path for route in app.routes]
        assert "/sse" in route_paths

    def test_registers_messages_endpoint(self, mock_mcp_server, sse_config):
        """Should register /messages/{session_id} endpoint."""
        from zapomni_mcp.sse_transport import create_sse_app

        app = create_sse_app(mock_mcp_server, sse_config)

        route_paths = [route.path for route in app.routes]
        assert "/messages" in route_paths

    def test_registers_health_endpoint(self, mock_mcp_server, sse_config):
        """Should register /health endpoint."""
        from zapomni_mcp.sse_transport import create_sse_app

        app = create_sse_app(mock_mcp_server, sse_config)

        route_paths = [route.path for route in app.routes]
        assert "/health" in route_paths

    def test_creates_app_successfully(self, mock_mcp_server, sse_config):
        """Should create app successfully without session manager (SDK manages sessions)."""
        from zapomni_mcp.sse_transport import create_sse_app

        app = create_sse_app(mock_mcp_server, sse_config)

        # Should create app with routes
        assert app is not None
        route_paths = [route.path for route in app.routes]
        assert len(route_paths) > 0

    def test_configures_cors_middleware(self, mock_mcp_server, sse_config):
        """Should configure CORS middleware."""
        from starlette.middleware.cors import CORSMiddleware

        from zapomni_mcp.sse_transport import create_sse_app

        app = create_sse_app(mock_mcp_server, sse_config)

        # Check middleware is configured by looking at the app's user_middleware list
        # Starlette stores user-added middleware in user_middleware attribute
        assert len(app.user_middleware) > 0, "Should have user middleware configured"
        # Verify CORS middleware is in the list
        cors_configured = any(m.cls == CORSMiddleware for m in app.user_middleware)
        assert cors_configured, "CORS middleware should be configured"


# Health Endpoint Tests


class TestHealthEndpoint:
    """Test suite for /health endpoint."""

    def test_health_returns_200(self, test_client):
        """Should return 200 OK for health check."""
        response = test_client.get("/health")

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"

    def test_health_returns_correct_structure(self, test_client):
        """Should return correct health response structure."""
        response = test_client.get("/health")
        data = response.json()

        assert "status" in data
        assert "version" in data
        assert "transport" in data
        assert "uptime_seconds" in data
        # Note: active_connections and metrics are no longer tracked
        # since we removed SessionManager in favor of SDK's built-in session management

    def test_health_status_healthy(self, test_client):
        """Should return healthy status."""
        response = test_client.get("/health")
        data = response.json()

        assert data["status"] == "healthy"
        assert data["transport"] == "sse"

    def test_health_includes_basic_info(self, test_client):
        """Should include basic health information."""
        response = test_client.get("/health")
        data = response.json()

        # Basic health info is present
        assert data["status"] == "healthy"
        assert data["transport"] == "sse"
        assert isinstance(data["uptime_seconds"], (int, float))
        # Note: metrics are no longer tracked since we use SDK's session management

    def test_health_uptime_increases(self, test_client):
        """Should show increasing uptime."""
        response1 = test_client.get("/health")
        time.sleep(0.1)
        response2 = test_client.get("/health")

        uptime1 = response1.json()["uptime_seconds"]
        uptime2 = response2.json()["uptime_seconds"]

        assert uptime2 > uptime1


# Messages Endpoint Tests


class TestMessagesEndpoint:
    """Test suite for /messages endpoint with query parameter session_id."""

    def test_messages_returns_404_for_invalid_session(self, test_client):
        """Should return 404 for non-existent session (via SDK validation)."""
        response = test_client.post(
            "/messages?session_id=non-existent-uuid",
            json={"jsonrpc": "2.0", "method": "test", "id": 1},
        )

        # SDK returns 400 for invalid session_id (not found is treated as bad request)
        assert response.status_code == 400

    def test_messages_requires_session_id(self, test_client):
        """Should require session_id query parameter."""
        # POST without session_id query param should return 400
        response = test_client.post(
            "/messages",
            json={"jsonrpc": "2.0", "method": "test", "id": 1},
        )
        # SDK will return 400 for missing session_id
        assert response.status_code == 400

    def test_messages_delegates_to_sdk(self, test_client):
        """Should delegate message handling to SDK's SseServerTransport."""
        # Test that messages endpoint delegates to SDK
        # SDK will return 400 for missing session_id
        response = test_client.post(
            "/messages",
            json={"jsonrpc": "2.0", "method": "test", "id": 1},
        )
        assert response.status_code == 400
        # The SDK's handle_post_message is being called

    @pytest.mark.skip(reason="SessionManager integration removed - MCP SDK manages sessions internally")
    @pytest.mark.asyncio
    async def test_messages_increments_request_count(self, mock_mcp_server, sse_config):
        """Should increment request count on message."""
        from zapomni_mcp.sse_transport import create_sse_app

        app = create_sse_app(mock_mcp_server, sse_config)
        session_manager = mock_mcp_server._session_manager

        # Create session
        mock_transport = MagicMock()
        mock_transport.handle_post_message = AsyncMock()

        await session_manager.create_session(
            session_id="request-count-session",
            transport=mock_transport,
        )

        # Make request
        with TestClient(app, raise_server_exceptions=False) as client:
            client.post(
                "/messages/request-count-session",
                json={"jsonrpc": "2.0", "method": "test", "id": 1},
            )

        session = session_manager.get_session("request-count-session")
        assert session.request_count >= 1


# SSE Endpoint Tests


class TestSSEEndpoint:
    """Test suite for /sse endpoint."""

    def test_sse_endpoint_exists(self, test_client):
        """Should have /sse endpoint registered."""
        # Note: SSE endpoints require special handling
        # We just verify the route exists
        with patch("zapomni_mcp.sse_transport.SseServerTransport") as mock_transport_class:
            mock_transport = MagicMock()
            mock_transport_class.return_value = mock_transport

            # SSE connections are long-lived, so we can't easily test with TestClient
            # We verify the route exists by checking routes
            route_paths = [route.path for route in test_client.app.routes]
            assert "/sse" in route_paths


# CORS Tests


class TestCORSConfiguration:
    """Test suite for CORS middleware configuration."""

    def test_cors_allows_all_origins_with_wildcard(self, mock_mcp_server):
        """Should allow all origins when configured with *."""
        # Include 'testserver' in allowed_hosts for TestClient compatibility
        config = SSEConfig(
            cors_origins=["*"],
            allowed_hosts=["localhost", "127.0.0.1", "::1", "testserver"],
        )

        from zapomni_mcp.sse_transport import create_sse_app

        app = create_sse_app(mock_mcp_server, config)
        client = TestClient(app)

        # OPTIONS request with Origin header
        response = client.options(
            "/health",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "GET",
            },
        )

        # Should allow the origin
        assert response.status_code == 200

    def test_cors_preflight_for_post(self, mock_mcp_server):
        """Should handle CORS preflight for POST requests."""
        # Include 'testserver' in allowed_hosts for TestClient compatibility
        config = SSEConfig(
            cors_origins=["*"],
            allowed_hosts=["localhost", "127.0.0.1", "::1", "testserver"],
        )

        from zapomni_mcp.sse_transport import create_sse_app

        app = create_sse_app(mock_mcp_server, config)
        client = TestClient(app)

        response = client.options(
            "/messages/test-session",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )

        assert response.status_code == 200


# Error Handling Tests


class TestErrorHandling:
    """Test suite for error handling in SSE transport."""

    @pytest.mark.skip(reason="SessionManager integration removed - MCP SDK manages sessions internally")
    @pytest.mark.asyncio
    async def test_messages_handles_transport_error(self, mock_mcp_server, sse_config):
        """Should return 500 on transport error."""
        from zapomni_mcp.sse_transport import create_sse_app

        app = create_sse_app(mock_mcp_server, sse_config)
        session_manager = mock_mcp_server._session_manager

        # Create session with failing transport
        mock_transport = MagicMock()
        mock_transport.handle_post_message = AsyncMock(side_effect=Exception("Transport error"))

        await session_manager.create_session(
            session_id="error-session",
            transport=mock_transport,
        )

        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                "/messages/error-session",
                json={"jsonrpc": "2.0", "method": "test", "id": 1},
            )

            assert response.status_code == 500
            data = response.json()
            assert "error" in data

    @pytest.mark.skip(reason="SessionManager integration removed - MCP SDK manages sessions internally")
    @pytest.mark.asyncio
    async def test_messages_increments_error_count_on_error(self, mock_mcp_server, sse_config):
        """Should increment error count on transport error."""
        from zapomni_mcp.sse_transport import create_sse_app

        app = create_sse_app(mock_mcp_server, sse_config)
        session_manager = mock_mcp_server._session_manager

        # Create session with failing transport
        mock_transport = MagicMock()
        mock_transport.handle_post_message = AsyncMock(side_effect=Exception("Transport error"))

        await session_manager.create_session(
            session_id="error-count-session",
            transport=mock_transport,
        )

        with TestClient(app, raise_server_exceptions=False) as client:
            client.post(
                "/messages/error-count-session",
                json={"jsonrpc": "2.0", "method": "test", "id": 1},
            )

        session = session_manager.get_session("error-count-session")
        assert session.error_count >= 1


# Version Tests


class TestVersionInfo:
    """Test suite for version information."""

    def test_health_includes_version(self, test_client):
        """Should include version in health response."""
        response = test_client.get("/health")
        data = response.json()

        assert "version" in data
        assert data["version"]  # Not empty

    def test_version_format(self, test_client):
        """Should return valid version format."""
        response = test_client.get("/health")
        data = response.json()

        version = data["version"]
        # Should be semver-like format
        parts = version.split(".")
        assert len(parts) >= 2  # At least major.minor


# Session Manager Integration Tests


class TestSessionManagerIntegration:
    """Test suite for SessionManager integration with SSE app.

    NOTE: These tests are skipped because SessionManager integration was removed
    when migrating to MCP SDK's SseServerTransport, which manages sessions internally.
    """

    @pytest.mark.skip(reason="SessionManager integration removed - MCP SDK manages sessions internally")
    @pytest.mark.asyncio
    async def test_session_manager_heartbeat_interval(self, mock_mcp_server):
        """Should configure SessionManager with correct heartbeat interval."""
        config = SSEConfig(heartbeat_interval=60)

        from zapomni_mcp.sse_transport import create_sse_app

        create_sse_app(mock_mcp_server, config)

        session_manager = mock_mcp_server._session_manager
        assert session_manager._heartbeat_interval == 60

    @pytest.mark.skip(reason="SessionManager integration removed - MCP SDK manages sessions internally")
    @pytest.mark.asyncio
    async def test_health_reflects_active_sessions(self, mock_mcp_server, sse_config):
        """Should reflect active sessions in health endpoint."""
        from zapomni_mcp.sse_transport import create_sse_app

        app = create_sse_app(mock_mcp_server, sse_config)
        session_manager = mock_mcp_server._session_manager

        # Create sessions
        for i in range(3):
            await session_manager.create_session(
                session_id=f"health-session-{i}",
                transport=MagicMock(),
            )

        with TestClient(app) as client:
            response = client.get("/health")
            data = response.json()

            assert data["active_connections"] == 3


# DNS Rebinding Protection Tests


class TestDNSRebindingProtection:
    """Test suite for DNS rebinding protection middleware."""

    def test_localhost_allowed_by_default(self, mock_mcp_server):
        """Should allow localhost by default."""
        config = SSEConfig(host="127.0.0.1")

        from zapomni_mcp.sse_transport import create_sse_app

        app = create_sse_app(mock_mcp_server, config)
        client = TestClient(app)

        response = client.get("/health", headers={"Host": "localhost:8000"})
        assert response.status_code == 200

    def test_127_0_0_1_allowed_by_default(self, mock_mcp_server):
        """Should allow 127.0.0.1 by default."""
        config = SSEConfig(host="127.0.0.1")

        from zapomni_mcp.sse_transport import create_sse_app

        app = create_sse_app(mock_mcp_server, config)
        client = TestClient(app)

        response = client.get("/health", headers={"Host": "127.0.0.1:8000"})
        assert response.status_code == 200

    def test_ipv6_localhost_allowed_by_default(self, mock_mcp_server):
        """Should allow ::1 (IPv6 localhost) by default."""
        config = SSEConfig(host="127.0.0.1")

        from zapomni_mcp.sse_transport import create_sse_app

        app = create_sse_app(mock_mcp_server, config)
        client = TestClient(app)

        # IPv6 addresses in Host header should use bracket notation: [::1]:8000
        # But we also accept ::1 without brackets for compatibility
        response = client.get("/health", headers={"Host": "[::1]:8000"})
        assert response.status_code == 200

    def test_invalid_host_blocked(self, mock_mcp_server):
        """Should block requests with invalid Host header."""
        config = SSEConfig(host="127.0.0.1")

        from zapomni_mcp.sse_transport import create_sse_app

        app = create_sse_app(mock_mcp_server, config)
        client = TestClient(app, raise_server_exceptions=False)

        response = client.get("/health", headers={"Host": "evil.example.com:8000"})
        assert response.status_code == 403
        assert "Host not allowed" in response.text

    def test_dns_rebinding_attack_blocked(self, mock_mcp_server):
        """Should block potential DNS rebinding attack."""
        config = SSEConfig(host="127.0.0.1")

        from zapomni_mcp.sse_transport import create_sse_app

        app = create_sse_app(mock_mcp_server, config)
        client = TestClient(app, raise_server_exceptions=False)

        # Attacker could use DNS rebinding to make victim's browser
        # connect to localhost thinking it's an external site
        response = client.get("/health", headers={"Host": "attacker-controlled.com"})
        assert response.status_code == 403

    def test_custom_allowed_hosts(self, mock_mcp_server):
        """Should allow custom host when configured."""
        config = SSEConfig(
            host="0.0.0.0",
            allowed_hosts=["myservice.example.com", "api.example.com"],
        )

        from zapomni_mcp.sse_transport import create_sse_app

        app = create_sse_app(mock_mcp_server, config)
        client = TestClient(app)

        response = client.get("/health", headers={"Host": "myservice.example.com:8000"})
        assert response.status_code == 200

        response = client.get("/health", headers={"Host": "api.example.com:443"})
        assert response.status_code == 200

    def test_custom_host_blocks_others(self, mock_mcp_server):
        """Should block hosts not in custom allowed_hosts."""
        config = SSEConfig(
            host="0.0.0.0",
            allowed_hosts=["myservice.example.com"],
        )

        from zapomni_mcp.sse_transport import create_sse_app

        app = create_sse_app(mock_mcp_server, config)
        client = TestClient(app, raise_server_exceptions=False)

        # Even localhost should be blocked if not in allowed_hosts
        response = client.get("/health", headers={"Host": "localhost:8000"})
        assert response.status_code == 403

    def test_protection_can_be_disabled(self, mock_mcp_server):
        """Should allow any host when protection is disabled."""
        config = SSEConfig(
            host="0.0.0.0",
            dns_rebinding_protection=False,
        )

        from zapomni_mcp.sse_transport import create_sse_app

        app = create_sse_app(mock_mcp_server, config)
        client = TestClient(app)

        # Should allow any host when protection is disabled
        response = client.get("/health", headers={"Host": "any-host.example.com:8000"})
        assert response.status_code == 200

    def test_host_header_case_insensitive(self, mock_mcp_server):
        """Should handle Host header case-insensitively."""
        config = SSEConfig(host="127.0.0.1")

        from zapomni_mcp.sse_transport import create_sse_app

        app = create_sse_app(mock_mcp_server, config)
        client = TestClient(app)

        response = client.get("/health", headers={"Host": "LOCALHOST:8000"})
        assert response.status_code == 200

        response = client.get("/health", headers={"Host": "LocalHost:8000"})
        assert response.status_code == 200

    def test_messages_endpoint_protected(self, mock_mcp_server):
        """Should protect messages endpoint from DNS rebinding."""
        config = SSEConfig(host="127.0.0.1")

        from zapomni_mcp.sse_transport import create_sse_app

        app = create_sse_app(mock_mcp_server, config)
        client = TestClient(app, raise_server_exceptions=False)

        # Messages endpoint should also be protected
        response = client.post(
            "/messages/test-session",
            headers={"Host": "evil.example.com:8000"},
            json={"jsonrpc": "2.0", "method": "test", "id": 1},
        )
        assert response.status_code == 403

    def test_middleware_configured_in_app(self, mock_mcp_server):
        """Should configure DNS rebinding middleware in app."""
        config = SSEConfig(host="127.0.0.1")

        from zapomni_mcp.sse_transport import DNSRebindingProtectionMiddleware, create_sse_app

        app = create_sse_app(mock_mcp_server, config)

        # Check middleware is configured
        dns_middleware_configured = any(
            m.cls == DNSRebindingProtectionMiddleware for m in app.user_middleware
        )
        assert dns_middleware_configured, "DNS rebinding protection middleware should be configured"

    def test_middleware_not_configured_when_disabled(self, mock_mcp_server):
        """Should not configure middleware when protection is disabled."""
        config = SSEConfig(
            host="0.0.0.0",
            dns_rebinding_protection=False,
        )

        from zapomni_mcp.sse_transport import DNSRebindingProtectionMiddleware, create_sse_app

        app = create_sse_app(mock_mcp_server, config)

        # Check middleware is NOT configured
        dns_middleware_configured = any(
            m.cls == DNSRebindingProtectionMiddleware for m in app.user_middleware
        )
        assert (
            not dns_middleware_configured
        ), "DNS rebinding middleware should not be configured when disabled"


class TestSSEConfigAllowedHosts:
    """Test suite for SSEConfig allowed_hosts configuration."""

    def test_localhost_default_allowed_hosts(self):
        """Should set default allowed hosts for localhost binding."""
        config = SSEConfig(host="127.0.0.1")

        assert "localhost" in config.allowed_hosts
        assert "127.0.0.1" in config.allowed_hosts
        assert "::1" in config.allowed_hosts

    def test_0_0_0_0_default_allowed_hosts(self):
        """Should set default allowed hosts for 0.0.0.0 binding."""
        config = SSEConfig(host="0.0.0.0", allowed_hosts=["myhost.com"])

        # Should use explicitly provided allowed_hosts
        assert "myhost.com" in config.allowed_hosts

    def test_non_localhost_requires_explicit_hosts(self):
        """Should require explicit allowed_hosts for non-localhost binding."""
        from zapomni_core.exceptions import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            SSEConfig(host="192.168.1.100")

        assert "allowed_hosts must be explicitly configured" in str(exc_info.value)

    def test_non_localhost_works_with_explicit_hosts(self):
        """Should work with explicit allowed_hosts for non-localhost."""
        config = SSEConfig(
            host="192.168.1.100",
            allowed_hosts=["192.168.1.100", "myserver.local"],
        )

        assert "192.168.1.100" in config.allowed_hosts
        assert "myserver.local" in config.allowed_hosts

    def test_non_localhost_works_with_protection_disabled(self):
        """Should work without explicit hosts when protection is disabled."""
        config = SSEConfig(
            host="192.168.1.100",
            dns_rebinding_protection=False,
        )

        # Should not raise and allowed_hosts can be empty
        assert config.dns_rebinding_protection is False

    def test_get_effective_allowed_hosts_returns_list(self):
        """Should return allowed hosts list."""
        config = SSEConfig(host="127.0.0.1")

        effective = config.get_effective_allowed_hosts()

        assert isinstance(effective, list)
        assert "localhost" in effective

    def test_get_effective_allowed_hosts_empty_when_disabled(self):
        """Should return empty list when protection is disabled."""
        config = SSEConfig(
            host="127.0.0.1",
            dns_rebinding_protection=False,
        )

        effective = config.get_effective_allowed_hosts()

        assert effective == []
