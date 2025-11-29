"""
MCP Tools for Zapomni.

This module contains all MCP tool implementations:
- add_memory: Store new information in memory
- search_memory: Retrieve relevant information
- get_stats: Query system statistics
- get_related: Find related entities through graph traversal
- set_model: Hot-reload Ollama LLM model without server restart
- get_callers: Find functions that call a specified function
- get_callees: Find functions called by a specified function
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


from .add_memory import AddMemoryTool  # noqa: E402
from .build_graph import BuildGraphTool  # noqa: E402
from .call_graph import GetCalleesTool, GetCallersTool  # noqa: E402
from .clear_all import ClearAllTool  # noqa: E402
from .delete_memory import DeleteMemoryTool  # noqa: E402
from .export_graph import ExportGraphTool  # noqa: E402
from .get_related import GetRelatedTool  # noqa: E402
from .get_stats import GetStatsTool  # noqa: E402
from .graph_status import GraphStatusTool  # noqa: E402
from .index_codebase import IndexCodebaseTool  # noqa: E402
from .prune_memory import PruneMemoryTool  # noqa: E402
from .search_memory import SearchMemoryTool  # noqa: E402
from .set_model import SetModelTool  # noqa: E402
from .workspace_tools import (  # noqa: E402
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
    "ClearAllTool",
    "DeleteMemoryTool",
    "ExportGraphTool",
    "IndexCodebaseTool",
    # Call graph tools
    "GetCallersTool",
    "GetCalleesTool",
    # Workspace tools
    "CreateWorkspaceTool",
    "ListWorkspacesTool",
    "SetCurrentWorkspaceTool",
    "GetCurrentWorkspaceTool",
    "DeleteWorkspaceTool",
]
