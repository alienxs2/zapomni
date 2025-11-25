"""
MCP Tools for Zapomni.

This module contains all MCP tool implementations:
- add_memory: Store new information in memory
- search_memory: Retrieve relevant information
- get_stats: Query system statistics
- get_related: Find related entities through graph traversal
- set_model: Hot-reload Ollama LLM model without server restart
- Workspace tools: create, list, set, get, delete workspaces

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from typing import Any, Dict, Protocol


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
from .build_graph import BuildGraphTool
from .get_related import GetRelatedTool
from .get_stats import GetStatsTool
from .graph_status import GraphStatusTool
from .prune_memory import PruneMemoryTool
from .search_memory import SearchMemoryTool
from .set_model import SetModelTool
from .workspace_tools import (
    CreateWorkspaceTool,
    DeleteWorkspaceTool,
    GetCurrentWorkspaceTool,
    ListWorkspacesTool,
    SetCurrentWorkspaceTool,
)

__all__ = [
    "MCPTool",
    "AddMemoryTool",
    "SearchMemoryTool",
    "GetStatsTool",
    "GetRelatedTool",
    "BuildGraphTool",
    "GraphStatusTool",
    "PruneMemoryTool",
    "SetModelTool",
    # Workspace tools
    "CreateWorkspaceTool",
    "ListWorkspacesTool",
    "SetCurrentWorkspaceTool",
    "GetCurrentWorkspaceTool",
    "DeleteWorkspaceTool",
]
