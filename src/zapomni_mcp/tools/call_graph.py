"""
CallGraph MCP Tools - Query callers and callees in the knowledge graph.

Provides MCP tools for querying the call graph stored in FalkorDB:
- get_callers: Find all functions that call a specified function
- get_callees: Find all functions called by a specified function

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from typing import Any, Dict, List, Optional

import structlog
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from zapomni_db.exceptions import DatabaseError
from zapomni_db.exceptions import ValidationError as DBValidationError
from zapomni_db.falkordb_client import FalkorDBClient
from zapomni_db.models import DEFAULT_WORKSPACE_ID

logger = structlog.get_logger(__name__)


class GetCallersRequest(BaseModel):
    """Pydantic model for validating get_callers request."""

    model_config = ConfigDict(extra="forbid")

    qualified_name: str = Field(
        ..., min_length=1, description="Qualified name of the function to find callers for"
    )
    limit: int = Field(default=50, ge=1, le=100, description="Maximum number of callers to return")
    workspace_id: str = Field(default="", description="Workspace ID for data isolation")


class GetCalleesRequest(BaseModel):
    """Pydantic model for validating get_callees request."""

    model_config = ConfigDict(extra="forbid")

    qualified_name: str = Field(
        ..., min_length=1, description="Qualified name of the function to find callees for"
    )
    limit: int = Field(default=50, ge=1, le=100, description="Maximum number of callees to return")
    workspace_id: str = Field(default="", description="Workspace ID for data isolation")


class GetCallersTool:
    """
    MCP tool for finding functions that call a specified function.

    Uses FalkorDB CALLS relationships to traverse the call graph
    and find all callers of a given function.

    Attributes:
        name: Tool identifier ("get_callers")
        description: Human-readable tool description
        input_schema: JSON Schema for input validation
        db_client: FalkorDBClient instance for database queries
        logger: Structured logger for operations
    """

    name = "get_callers"
    description = (
        "Find all functions that call the specified function. "
        "Returns a list of caller functions with their file paths and call locations. "
        "Use this to understand how a function is used throughout the codebase."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "qualified_name": {
                "type": "string",
                "description": (
                    "Qualified name of the function to find callers for "
                    "(e.g., 'module.ClassName.method')"
                ),
                "minLength": 1,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of callers to return",
                "default": 50,
                "minimum": 1,
                "maximum": 100,
            },
            "workspace_id": {
                "type": "string",
                "description": (
                    "Optional workspace ID. If not specified, uses the default workspace."
                ),
                "default": "",
            },
        },
        "required": ["qualified_name"],
        "additionalProperties": False,
    }

    def __init__(self, db_client: FalkorDBClient) -> None:
        """
        Initialize GetCallersTool with database client.

        Args:
            db_client: FalkorDBClient instance for executing queries

        Raises:
            TypeError: If db_client is not a FalkorDBClient instance
        """
        if not isinstance(db_client, FalkorDBClient):
            raise TypeError(f"db_client must be FalkorDBClient instance, got {type(db_client)}")

        self.db_client = db_client
        self.logger = logger.bind(tool=self.name)
        self.logger.info("get_callers_tool_initialized")

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute get_callers tool with provided arguments.

        Args:
            arguments: Dictionary containing:
                - qualified_name (str, required): Function name to find callers for
                - limit (int, optional): Max results (default: 50)
                - workspace_id (str, optional): Workspace ID

        Returns:
            MCP-formatted response dictionary

        Example:
            >>> result = await tool.execute({
            ...     "qualified_name": "module.helper_func",
            ...     "limit": 20
            ... })
        """
        request_id = id(arguments)
        log = self.logger.bind(request_id=request_id)

        try:
            # Step 1: Validate input
            log.info("validating_get_callers_arguments")
            request = GetCallersRequest(**arguments)

            # Resolve workspace
            workspace_id = request.workspace_id or DEFAULT_WORKSPACE_ID

            log.info(
                "executing_get_callers",
                qualified_name=request.qualified_name,
                workspace_id=workspace_id,
                limit=request.limit,
            )

            # Step 2: Execute query
            callers = await self.db_client.get_callers(
                qualified_name=request.qualified_name,
                workspace_id=workspace_id,
                limit=request.limit,
            )

            log.info(
                "get_callers_complete",
                qualified_name=request.qualified_name,
                callers_count=len(callers),
            )

            # Step 3: Format response
            return self._format_response(request.qualified_name, callers)

        except ValidationError as e:
            log.warning("validation_error", error=str(e))
            return self._format_error(str(e))

        except (DBValidationError, DatabaseError) as e:
            log.error("database_error", error=str(e), exc_info=True)
            return self._format_error(f"Database error: {e}")

        except Exception as e:
            log.error("unexpected_error", error=str(e), exc_info=True)
            return self._format_error(f"Unexpected error: {e}")

    def _format_response(
        self,
        qualified_name: str,
        callers: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Format callers list as MCP response."""
        if not callers:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"No callers found for '{qualified_name}'.\n\n"
                        "This could mean:\n"
                        "- The function is not called by any other function\n"
                        "- The function name is incorrect\n"
                        "- The codebase has not been indexed with call graph analysis",
                    }
                ],
                "isError": False,
            }

        # Build formatted text response
        lines = [
            f"Found {len(callers)} caller(s) for '{qualified_name}':\n",
        ]

        for i, caller in enumerate(callers, start=1):
            caller_name = caller.get("caller_qualified_name", "unknown")
            file_path = caller.get("caller_file_path", "unknown")
            call_line = caller.get("call_line", 0)
            call_type = caller.get("call_type", "function")
            args_count = caller.get("arguments_count", 0)
            call_count = caller.get("call_count", 1)

            lines.append(f"\n{i}. {caller_name}")
            lines.append(f"   File: {file_path}")
            lines.append(f"   Line: {call_line}")
            lines.append(f"   Type: {call_type}")
            lines.append(f"   Arguments: {args_count}")
            if call_count > 1:
                lines.append(f"   Call count: {call_count}")

        formatted_text = "\n".join(lines)

        return {
            "content": [
                {
                    "type": "text",
                    "text": formatted_text,
                }
            ],
            "isError": False,
        }

    def _format_error(self, message: str) -> Dict[str, Any]:
        """Format error as MCP error response."""
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Error: {message}",
                }
            ],
            "isError": True,
        }


class GetCalleesTool:
    """
    MCP tool for finding functions called by a specified function.

    Uses FalkorDB CALLS relationships to traverse the call graph
    and find all functions that the specified function calls.

    Attributes:
        name: Tool identifier ("get_callees")
        description: Human-readable tool description
        input_schema: JSON Schema for input validation
        db_client: FalkorDBClient instance for database queries
        logger: Structured logger for operations
    """

    name = "get_callees"
    description = (
        "Find all functions called by the specified function. "
        "Returns a list of functions that are called from within the specified function. "
        "Use this to understand the dependencies of a function."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "qualified_name": {
                "type": "string",
                "description": (
                    "Qualified name of the function to find callees for "
                    "(e.g., 'module.ClassName.method')"
                ),
                "minLength": 1,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of callees to return",
                "default": 50,
                "minimum": 1,
                "maximum": 100,
            },
            "workspace_id": {
                "type": "string",
                "description": (
                    "Optional workspace ID. If not specified, uses the default workspace."
                ),
                "default": "",
            },
        },
        "required": ["qualified_name"],
        "additionalProperties": False,
    }

    def __init__(self, db_client: FalkorDBClient) -> None:
        """
        Initialize GetCalleesTool with database client.

        Args:
            db_client: FalkorDBClient instance for executing queries

        Raises:
            TypeError: If db_client is not a FalkorDBClient instance
        """
        if not isinstance(db_client, FalkorDBClient):
            raise TypeError(f"db_client must be FalkorDBClient instance, got {type(db_client)}")

        self.db_client = db_client
        self.logger = logger.bind(tool=self.name)
        self.logger.info("get_callees_tool_initialized")

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute get_callees tool with provided arguments.

        Args:
            arguments: Dictionary containing:
                - qualified_name (str, required): Function name to find callees for
                - limit (int, optional): Max results (default: 50)
                - workspace_id (str, optional): Workspace ID

        Returns:
            MCP-formatted response dictionary

        Example:
            >>> result = await tool.execute({
            ...     "qualified_name": "module.MyClass.process",
            ...     "limit": 20
            ... })
        """
        request_id = id(arguments)
        log = self.logger.bind(request_id=request_id)

        try:
            # Step 1: Validate input
            log.info("validating_get_callees_arguments")
            request = GetCalleesRequest(**arguments)

            # Resolve workspace
            workspace_id = request.workspace_id or DEFAULT_WORKSPACE_ID

            log.info(
                "executing_get_callees",
                qualified_name=request.qualified_name,
                workspace_id=workspace_id,
                limit=request.limit,
            )

            # Step 2: Execute query
            callees = await self.db_client.get_callees(
                qualified_name=request.qualified_name,
                workspace_id=workspace_id,
                limit=request.limit,
            )

            log.info(
                "get_callees_complete",
                qualified_name=request.qualified_name,
                callees_count=len(callees),
            )

            # Step 3: Format response
            return self._format_response(request.qualified_name, callees)

        except ValidationError as e:
            log.warning("validation_error", error=str(e))
            return self._format_error(str(e))

        except (DBValidationError, DatabaseError) as e:
            log.error("database_error", error=str(e), exc_info=True)
            return self._format_error(f"Database error: {e}")

        except Exception as e:
            log.error("unexpected_error", error=str(e), exc_info=True)
            return self._format_error(f"Unexpected error: {e}")

    def _format_response(
        self,
        qualified_name: str,
        callees: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Format callees list as MCP response."""
        if not callees:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"No callees found for '{qualified_name}'.\n\n"
                        "This could mean:\n"
                        "- The function does not call any other functions\n"
                        "- The function name is incorrect\n"
                        "- The codebase has not been indexed with call graph analysis",
                    }
                ],
                "isError": False,
            }

        # Build formatted text response
        lines = [
            f"'{qualified_name}' calls {len(callees)} function(s):\n",
        ]

        for i, callee in enumerate(callees, start=1):
            callee_name = callee.get("callee_qualified_name", "unknown")
            file_path = callee.get("callee_file_path", "unknown")
            call_line = callee.get("call_line", 0)
            call_type = callee.get("call_type", "function")
            args_count = callee.get("arguments_count", 0)
            call_count = callee.get("call_count", 1)

            lines.append(f"\n{i}. {callee_name}")
            lines.append(f"   File: {file_path}")
            lines.append(f"   Called at line: {call_line}")
            lines.append(f"   Type: {call_type}")
            lines.append(f"   Arguments: {args_count}")
            if call_count > 1:
                lines.append(f"   Call count: {call_count}")

        formatted_text = "\n".join(lines)

        return {
            "content": [
                {
                    "type": "text",
                    "text": formatted_text,
                }
            ],
            "isError": False,
        }

    def _format_error(self, message: str) -> Dict[str, Any]:
        """Format error as MCP error response."""
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Error: {message}",
                }
            ],
            "isError": True,
        }
