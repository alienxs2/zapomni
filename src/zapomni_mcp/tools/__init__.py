"""
MCP Tools for Zapomni.

This module contains all MCP tool implementations:
- add_memory: Store new information in memory
- search_memory: Retrieve relevant information
- get_stats: Query system statistics
- get_related: Find related entities through graph traversal

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from typing import Protocol, Any, Dict


class MCPTool(Protocol):
    """
    Protocol definition for MCP tools.

    All MCP tools must implement this interface to be registered
    with the MCPServer.
    """

    name: str
    description: str
    input_schema: Dict[str, Any]

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with given arguments."""
        ...


from .add_memory import AddMemoryTool
from .search_memory import SearchMemoryTool
from .get_stats import GetStatsTool
from .get_related import GetRelatedTool
from .build_graph import BuildGraphTool
from .graph_status import GraphStatusTool


__all__ = [
    "MCPTool",
    "AddMemoryTool",
    "SearchMemoryTool",
    "GetStatsTool",
    "GetRelatedTool",
    "BuildGraphTool",
    "GraphStatusTool",
]
