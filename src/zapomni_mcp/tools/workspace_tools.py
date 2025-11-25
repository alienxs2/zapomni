"""
Workspace MCP Tools - Workspace management operations.

Provides 5 MCP tools for workspace management:
1. create_workspace - Create a new workspace
2. list_workspaces - List all workspaces
3. set_current_workspace - Set the current workspace for the session
4. get_current_workspace - Get the current workspace for the session
5. delete_workspace - Delete a workspace and all its data

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from typing import Any, Dict, List, Optional

import structlog
from pydantic import BaseModel, ConfigDict, Field

from zapomni_core.exceptions import DatabaseError, ValidationError
from zapomni_core.workspace_manager import WorkspaceManager
from zapomni_db.models import DEFAULT_WORKSPACE_ID, Workspace, WorkspaceStats

logger = structlog.get_logger(__name__)


# ============================================================================
# CREATE WORKSPACE TOOL
# ============================================================================


class CreateWorkspaceRequest(BaseModel):
    """Pydantic model for create_workspace request."""

    model_config = ConfigDict(extra="forbid")

    workspace_id: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    name: str = Field(..., min_length=1, max_length=255)
    description: str = Field(default="", max_length=1000)


class CreateWorkspaceTool:
    """
    MCP tool for creating new workspaces.

    Workspaces provide data isolation for memories, allowing multiple
    projects or contexts to be managed independently.
    """

    name = "create_workspace"
    description = (
        "Create a new workspace for data isolation. "
        "Workspaces allow you to organize memories into separate contexts."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "workspace_id": {
                "type": "string",
                "description": (
                    "Unique identifier for the workspace. "
                    "Must be 1-64 characters, alphanumeric with hyphens and underscores."
                ),
                "minLength": 1,
                "maxLength": 64,
                "pattern": "^[a-zA-Z0-9_-]+$",
            },
            "name": {
                "type": "string",
                "description": "Human-readable name for the workspace.",
                "minLength": 1,
                "maxLength": 255,
            },
            "description": {
                "type": "string",
                "description": "Optional description of the workspace.",
                "maxLength": 1000,
            },
        },
        "required": ["workspace_id", "name"],
        "additionalProperties": False,
    }

    def __init__(self, workspace_manager: WorkspaceManager) -> None:
        """
        Initialize CreateWorkspaceTool.

        Args:
            workspace_manager: WorkspaceManager instance for operations.
        """
        if workspace_manager is None:
            raise ValueError("workspace_manager is required")

        self.workspace_manager = workspace_manager
        self.logger = logger.bind(tool=self.name)
        self.logger.info("create_workspace_tool_initialized")

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute create_workspace tool."""
        log = self.logger.bind(request_id=id(arguments))

        try:
            log.info("validating_arguments")
            request = CreateWorkspaceRequest(**arguments)

            log.info(
                "creating_workspace",
                workspace_id=request.workspace_id,
                name=request.name,
            )

            workspace_id = await self.workspace_manager.create_workspace(
                workspace_id=request.workspace_id,
                name=request.name,
                description=request.description,
            )

            log.info("workspace_created", workspace_id=workspace_id)

            message = (
                f"Workspace created successfully.\n"
                f"ID: {workspace_id}\n"
                f"Name: {request.name}\n"
                f"Description: {request.description or '(none)'}"
            )

            return {
                "content": [{"type": "text", "text": message}],
                "isError": False,
            }

        except ValidationError as e:
            log.warning("validation_error", error=str(e))
            return {
                "content": [{"type": "text", "text": f"Error: {e}"}],
                "isError": True,
            }
        except DatabaseError as e:
            log.error("database_error", error=str(e))
            return {
                "content": [{"type": "text", "text": f"Error: {e}"}],
                "isError": True,
            }
        except Exception as e:
            log.error("unexpected_error", error=str(e), exc_info=True)
            return {
                "content": [{"type": "text", "text": "Error: Failed to create workspace."}],
                "isError": True,
            }


# ============================================================================
# LIST WORKSPACES TOOL
# ============================================================================


class ListWorkspacesTool:
    """
    MCP tool for listing all workspaces.
    """

    name = "list_workspaces"
    description = (
        "List all available workspaces. "
        "Shows workspace ID, name, description, and creation date."
    )
    input_schema = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def __init__(self, workspace_manager: WorkspaceManager) -> None:
        """Initialize ListWorkspacesTool."""
        if workspace_manager is None:
            raise ValueError("workspace_manager is required")

        self.workspace_manager = workspace_manager
        self.logger = logger.bind(tool=self.name)
        self.logger.info("list_workspaces_tool_initialized")

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute list_workspaces tool."""
        log = self.logger.bind(request_id=id(arguments))

        try:
            log.info("listing_workspaces")
            workspaces = await self.workspace_manager.list_workspaces()

            if not workspaces:
                message = "No workspaces found. Use create_workspace to create one."
            else:
                lines = [f"Found {len(workspaces)} workspace(s):\n"]
                for ws in workspaces:
                    created = (
                        ws.created_at.strftime("%Y-%m-%d %H:%M") if ws.created_at else "unknown"
                    )
                    lines.append(
                        f"- {ws.id}: {ws.name}\n"
                        f"  Description: {ws.description or '(none)'}\n"
                        f"  Created: {created}"
                    )
                message = "\n".join(lines)

            log.info("workspaces_listed", count=len(workspaces))

            return {
                "content": [{"type": "text", "text": message}],
                "isError": False,
            }

        except DatabaseError as e:
            log.error("database_error", error=str(e))
            return {
                "content": [{"type": "text", "text": f"Error: {e}"}],
                "isError": True,
            }
        except Exception as e:
            log.error("unexpected_error", error=str(e), exc_info=True)
            return {
                "content": [{"type": "text", "text": "Error: Failed to list workspaces."}],
                "isError": True,
            }


# ============================================================================
# SET CURRENT WORKSPACE TOOL
# ============================================================================


class SetCurrentWorkspaceRequest(BaseModel):
    """Pydantic model for set_current_workspace request."""

    model_config = ConfigDict(extra="forbid")

    workspace_id: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")


class SetCurrentWorkspaceTool:
    """
    MCP tool for setting the current workspace for the session.

    All subsequent memory operations will use this workspace until changed.
    """

    name = "set_current_workspace"
    description = (
        "Set the current workspace for this session. "
        "All subsequent memory operations will use this workspace."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "workspace_id": {
                "type": "string",
                "description": "ID of the workspace to switch to.",
                "minLength": 1,
                "maxLength": 64,
                "pattern": "^[a-zA-Z0-9_-]+$",
            },
        },
        "required": ["workspace_id"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        workspace_manager: WorkspaceManager,
        mcp_server: Any,
    ) -> None:
        """
        Initialize SetCurrentWorkspaceTool.

        Args:
            workspace_manager: WorkspaceManager instance for operations.
            mcp_server: MCPServer instance for session management.
        """
        if workspace_manager is None:
            raise ValueError("workspace_manager is required")
        if mcp_server is None:
            raise ValueError("mcp_server is required")

        self.workspace_manager = workspace_manager
        self.mcp_server = mcp_server
        self.logger = logger.bind(tool=self.name)
        self.logger.info("set_current_workspace_tool_initialized")

    async def execute(
        self, arguments: Dict[str, Any], session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute set_current_workspace tool."""
        log = self.logger.bind(request_id=id(arguments), session_id=session_id)

        try:
            log.info("validating_arguments")
            request = SetCurrentWorkspaceRequest(**arguments)

            # Verify workspace exists
            workspace = await self.workspace_manager.get_workspace(request.workspace_id)
            if workspace is None:
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Error: Workspace '{request.workspace_id}' not found.",
                        }
                    ],
                    "isError": True,
                }

            # Set workspace in session
            if session_id:
                success = self.mcp_server.set_session_workspace(session_id, request.workspace_id)
                if not success:
                    log.warning("failed_to_set_session_workspace")

            log.info(
                "workspace_set",
                workspace_id=request.workspace_id,
            )

            message = (
                f"Current workspace set to '{request.workspace_id}'.\n"
                f"Name: {workspace.name}\n"
                f"All subsequent memory operations will use this workspace."
            )

            return {
                "content": [{"type": "text", "text": message}],
                "isError": False,
            }

        except ValidationError as e:
            log.warning("validation_error", error=str(e))
            return {
                "content": [{"type": "text", "text": f"Error: {e}"}],
                "isError": True,
            }
        except DatabaseError as e:
            log.error("database_error", error=str(e))
            return {
                "content": [{"type": "text", "text": f"Error: {e}"}],
                "isError": True,
            }
        except Exception as e:
            log.error("unexpected_error", error=str(e), exc_info=True)
            return {
                "content": [{"type": "text", "text": "Error: Failed to set workspace."}],
                "isError": True,
            }


# ============================================================================
# GET CURRENT WORKSPACE TOOL
# ============================================================================


class GetCurrentWorkspaceTool:
    """
    MCP tool for getting the current workspace for the session.
    """

    name = "get_current_workspace"
    description = (
        "Get the current workspace for this session. "
        "Shows the workspace ID and name, plus statistics about memories."
    )
    input_schema = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def __init__(
        self,
        workspace_manager: WorkspaceManager,
        mcp_server: Any,
    ) -> None:
        """Initialize GetCurrentWorkspaceTool."""
        if workspace_manager is None:
            raise ValueError("workspace_manager is required")
        if mcp_server is None:
            raise ValueError("mcp_server is required")

        self.workspace_manager = workspace_manager
        self.mcp_server = mcp_server
        self.logger = logger.bind(tool=self.name)
        self.logger.info("get_current_workspace_tool_initialized")

    async def execute(
        self, arguments: Dict[str, Any], session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute get_current_workspace tool."""
        log = self.logger.bind(request_id=id(arguments), session_id=session_id)

        try:
            # Get current workspace from session
            workspace_id = self.mcp_server.resolve_workspace_id(session_id)

            log.info("getting_workspace", workspace_id=workspace_id)

            # Get workspace details
            workspace = await self.workspace_manager.get_workspace(workspace_id)

            # Get workspace stats
            stats = await self.workspace_manager.get_workspace_stats(workspace_id)

            if workspace:
                message = (
                    f"Current Workspace: {workspace_id}\n"
                    f"Name: {workspace.name}\n"
                    f"Description: {workspace.description or '(none)'}\n"
                    f"\nStatistics:\n"
                    f"- Memories: {stats.total_memories}\n"
                    f"- Chunks: {stats.total_chunks}\n"
                    f"- Entities: {stats.total_entities}\n"
                    f"- Relationships: {stats.total_relationships}"
                )
            else:
                message = (
                    f"Current Workspace: {workspace_id}\n"
                    f"(Workspace not found in database - using default)\n"
                    f"\nStatistics:\n"
                    f"- Memories: {stats.total_memories}\n"
                    f"- Chunks: {stats.total_chunks}\n"
                    f"- Entities: {stats.total_entities}\n"
                    f"- Relationships: {stats.total_relationships}"
                )

            return {
                "content": [{"type": "text", "text": message}],
                "isError": False,
            }

        except DatabaseError as e:
            log.error("database_error", error=str(e))
            return {
                "content": [{"type": "text", "text": f"Error: {e}"}],
                "isError": True,
            }
        except Exception as e:
            log.error("unexpected_error", error=str(e), exc_info=True)
            return {
                "content": [{"type": "text", "text": "Error: Failed to get workspace."}],
                "isError": True,
            }


# ============================================================================
# DELETE WORKSPACE TOOL
# ============================================================================


class DeleteWorkspaceRequest(BaseModel):
    """Pydantic model for delete_workspace request."""

    model_config = ConfigDict(extra="forbid")

    workspace_id: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    confirm: bool = Field(
        default=False,
        description="Must be true to confirm deletion.",
    )


class DeleteWorkspaceTool:
    """
    MCP tool for deleting a workspace and all its data.

    WARNING: This is a destructive operation that cannot be undone.
    """

    name = "delete_workspace"
    description = (
        "Delete a workspace and ALL its data. "
        "WARNING: This is irreversible. Set confirm=true to proceed."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "workspace_id": {
                "type": "string",
                "description": "ID of the workspace to delete.",
                "minLength": 1,
                "maxLength": 64,
                "pattern": "^[a-zA-Z0-9_-]+$",
            },
            "confirm": {
                "type": "boolean",
                "description": "Must be true to confirm deletion.",
                "default": False,
            },
        },
        "required": ["workspace_id"],
        "additionalProperties": False,
    }

    def __init__(self, workspace_manager: WorkspaceManager) -> None:
        """Initialize DeleteWorkspaceTool."""
        if workspace_manager is None:
            raise ValueError("workspace_manager is required")

        self.workspace_manager = workspace_manager
        self.logger = logger.bind(tool=self.name)
        self.logger.info("delete_workspace_tool_initialized")

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute delete_workspace tool."""
        log = self.logger.bind(request_id=id(arguments))

        try:
            log.info("validating_arguments")
            request = DeleteWorkspaceRequest(**arguments)

            # Cannot delete default workspace
            if request.workspace_id == DEFAULT_WORKSPACE_ID:
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": "Error: Cannot delete the default workspace.",
                        }
                    ],
                    "isError": True,
                }

            # Require confirmation
            if not request.confirm:
                # Get stats for warning message
                stats = await self.workspace_manager.get_workspace_stats(request.workspace_id)

                message = (
                    f"WARNING: You are about to delete workspace '{request.workspace_id}'.\n"
                    f"\nThis workspace contains:\n"
                    f"- {stats.total_memories} memories\n"
                    f"- {stats.total_chunks} chunks\n"
                    f"- {stats.total_entities} entities\n"
                    f"- {stats.total_relationships} relationships\n"
                    f"\nThis action is IRREVERSIBLE.\n"
                    f"To confirm deletion, call again with confirm=true."
                )

                return {
                    "content": [{"type": "text", "text": message}],
                    "isError": False,
                }

            log.warning(
                "deleting_workspace",
                workspace_id=request.workspace_id,
            )

            deleted = await self.workspace_manager.delete_workspace(
                request.workspace_id, confirm=True
            )

            if deleted:
                log.warning("workspace_deleted", workspace_id=request.workspace_id)
                message = f"Workspace '{request.workspace_id}' has been deleted."
            else:
                log.info("workspace_not_found", workspace_id=request.workspace_id)
                message = f"Workspace '{request.workspace_id}' was not found."

            return {
                "content": [{"type": "text", "text": message}],
                "isError": False,
            }

        except ValidationError as e:
            log.warning("validation_error", error=str(e))
            return {
                "content": [{"type": "text", "text": f"Error: {e}"}],
                "isError": True,
            }
        except DatabaseError as e:
            log.error("database_error", error=str(e))
            return {
                "content": [{"type": "text", "text": f"Error: {e}"}],
                "isError": True,
            }
        except Exception as e:
            log.error("unexpected_error", error=str(e), exc_info=True)
            return {
                "content": [{"type": "text", "text": "Error: Failed to delete workspace."}],
                "isError": True,
            }


__all__ = [
    "CreateWorkspaceTool",
    "ListWorkspacesTool",
    "SetCurrentWorkspaceTool",
    "GetCurrentWorkspaceTool",
    "DeleteWorkspaceTool",
]
