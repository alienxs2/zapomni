"""
Zapomni MCP Layer.

Top layer in dependency hierarchy. Contains:
- MCP server and protocol handling
- Tool definitions
- User-facing interface

Note: Imports are lazy to allow logging configuration before
module-level logger initialization in dependent modules.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .server import MCPServer, ServerStats, ConfigurationError
    from .config import Settings


def __getattr__(name: str):
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["MCPServer", "ServerStats", "ConfigurationError", "Settings"]
