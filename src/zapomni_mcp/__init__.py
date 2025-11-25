"""
Zapomni MCP Layer.

Top layer in dependency hierarchy. Contains:
- MCP server and protocol handling
- Tool definitions
- User-facing interface
- SSE transport for concurrent connections

Note: Imports are lazy to allow logging configuration before
module-level logger initialization in dependent modules.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .server import MCPServer, ServerStats, ConfigurationError
    from .config import Settings, SSEConfig
    from .session_manager import SessionManager, SessionState
    from .sse_transport import create_sse_app


def __getattr__(name: str) -> Any:
    """Lazy import to support deferred initialization."""
    if name == "MCPServer":
        from .server import MCPServer
        return MCPServer
    elif name == "ServerStats":
        from .server import ServerStats
        return ServerStats
    elif name == "ConfigurationError":
        from .server import ConfigurationError
        return ConfigurationError
    elif name == "Settings":
        from .config import Settings
        return Settings
    elif name == "SSEConfig":
        from .config import SSEConfig
        return SSEConfig
    elif name == "SessionManager":
        from .session_manager import SessionManager
        return SessionManager
    elif name == "SessionState":
        from .session_manager import SessionState
        return SessionState
    elif name == "create_sse_app":
        from .sse_transport import create_sse_app
        return create_sse_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MCPServer",
    "ServerStats",
    "ConfigurationError",
    "Settings",
    "SSEConfig",
    "SessionManager",
    "SessionState",
    "create_sse_app",
]
