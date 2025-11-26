"""
DeleteMemory MCP Tool - Full Implementation.

Deletes a specific memory by ID with required safety confirmation.
Requires explicit confirmation (confirm=true) to prevent accidental deletions.
Delegates to MemoryProcessor for the database operation.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

import uuid
from typing import Any, Dict, Optional

import structlog
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from zapomni_core.exceptions import DatabaseError
from zapomni_core.exceptions import ValidationError as CoreValidationError
from zapomni_core.memory_processor import MemoryProcessor

logger = structlog.get_logger(__name__)


class DeleteMemoryRequest(BaseModel):
    """Pydantic model for validating delete_memory request."""

    model_config = ConfigDict(extra="forbid")

    memory_id: str
    confirm: bool

    @field_validator("memory_id")
    @classmethod
    def validate_memory_id(cls, v: str) -> str:
        """Validate that memory_id is a valid UUID format."""
        try:
            uuid.UUID(v)
        except ValueError as e:
            raise ValueError(f"Invalid UUID format: {e}")
        return v


class DeleteMemoryResponse(BaseModel):
    """Pydantic model for delete_memory response."""

    status: str
    deleted: bool
    memory_id: str
    message: str = ""
    error: Optional[str] = None


class DeleteMemoryTool:
    """
    MCP tool for deleting memories.

    This tool validates input with required safety confirmation,
    delegates to MemoryProcessor.db_client.delete_memory(),
    and formats the response according to MCP protocol.

    IMPORTANT: Requires explicit confirm=true to prevent accidental deletions.
    This is a safety feature to ensure users must explicitly confirm deletion.

    Attributes:
        name: Tool identifier ("delete_memory")
        description: Human-readable tool description
        input_schema: JSON Schema for input validation
        memory_processor: MemoryProcessor instance for database access
        logger: Structured logger for operations
    """

    name = "delete_memory"
    description = (
        "Delete a specific memory by ID. "
        "REQUIRES explicit confirmation (confirm=true) for safety. "
        "This operation is permanent and cannot be undone. "
        "Provide the exact memory UUID and confirm=true to proceed."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "string",
                "description": (
                    "UUID of the memory to delete. "
                    "Must be a valid UUID format (e.g., 550e8400-e29b-41d4-a716-446655440000)"
                ),
                "format": "uuid",
            },
            "confirm": {
                "type": "boolean",
                "description": (
                    "REQUIRED: Must be explicitly set to true to confirm deletion. "
                    "This is a safety feature to prevent accidental deletions."
                ),
            },
        },
        "required": ["memory_id", "confirm"],
        "additionalProperties": False,
    }

    def __init__(self, memory_processor: MemoryProcessor) -> None:
        """
        Initialize DeleteMemoryTool with MemoryProcessor.

        Args:
            memory_processor: MemoryProcessor instance for database operations.
                Must be initialized and connected to database.

        Raises:
            TypeError: If memory_processor is not a MemoryProcessor instance

        Example:
            >>> processor = MemoryProcessor(...)
            >>> tool = DeleteMemoryTool(memory_processor=processor)
        """
        if not isinstance(memory_processor, MemoryProcessor):
            raise TypeError(
                f"memory_processor must be MemoryProcessor instance, got {type(memory_processor)}"
            )

        self.memory_processor = memory_processor
        self.logger = logger.bind(tool=self.name)

        self.logger.info("delete_memory_tool_initialized")

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute delete_memory tool with provided arguments.

        This is the main entry point called by the MCP server when a client
        invokes the delete_memory tool. It validates inputs with mandatory
        confirmation, deletes the memory through the database client,
        and returns a formatted MCP response.

        Args:
            arguments: Dictionary containing:
                - memory_id (str, required): UUID of memory to delete
                - confirm (bool, required): Must be true to confirm deletion

        Returns:
            Dictionary in MCP response format:
            {
                "content": [
                    {
                        "type": "text",
                        "text": "Formatted response message"
                    }
                ],
                "isError": False or True
            }

        Example:
            >>> tool = DeleteMemoryTool(memory_processor=processor)
            >>> result = await tool.execute({
            ...     "memory_id": "550e8400-e29b-41d4-a716-446655440000",
            ...     "confirm": True
            ... })
            >>> print(result["isError"])
            False
        """
        log = self.logger.bind(request_id=id(arguments))

        try:
            # Validate arguments type
            if not isinstance(arguments, dict):
                log.warning(
                    "delete_memory_invalid_arguments_type",
                    type=type(arguments).__name__,
                )
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": "Error: Arguments must be a dictionary",
                        }
                    ],
                    "isError": True,
                }

            # Parse and validate with Pydantic
            log.debug("parsing_delete_memory_request", arguments=arguments)
            request = DeleteMemoryRequest(**arguments)

            # Safety check: confirm must be true
            if not request.confirm:
                log.warning(
                    "delete_memory_confirmation_failed",
                    memory_id=request.memory_id,
                    confirm=request.confirm,
                )
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Error: Deletion requires explicit confirmation. "
                                "Please set confirm=true to proceed. "
                                "This is a safety feature to prevent accidental deletions."
                            ),
                        }
                    ],
                    "isError": True,
                }

            # Log deletion request
            log.info(
                "delete_memory_requested",
                memory_id=request.memory_id,
                confirm=request.confirm,
            )

            # Execute deletion through database client
            deleted = await self.memory_processor.db_client.delete_memory(request.memory_id)

            # Format response
            if deleted:
                log.info("delete_memory_success", memory_id=request.memory_id)
                response = {
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Memory deleted successfully.\n"
                                f"ID: {request.memory_id}\n"
                                f"Status: Deleted"
                            ),
                        }
                    ],
                    "isError": False,
                }
            else:
                log.warning(
                    "delete_memory_not_found",
                    memory_id=request.memory_id,
                )
                response = {
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Memory not found.\n"
                                f"ID: {request.memory_id}\n"
                                f"Status: Not Found - No deletion performed"
                            ),
                        }
                    ],
                    "isError": True,
                }

            return response

        except ValidationError as e:
            # Pydantic validation error
            log.warning(
                "delete_memory_validation_error",
                error=str(e),
                error_type=type(e).__name__,
            )
            error_details = "; ".join([f"{err['loc'][0]}: {err['msg']}" for err in e.errors()])
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Error: Input validation failed - {error_details}",
                    }
                ],
                "isError": True,
            }

        except CoreValidationError as e:
            # Database validation error
            log.error(
                "delete_memory_database_validation_error",
                error=str(e),
                error_type=type(e).__name__,
            )
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Error: Invalid memory ID - {str(e)}",
                    }
                ],
                "isError": True,
            }

        except DatabaseError as e:
            # Database error during deletion
            log.error(
                "delete_memory_database_error",
                error=str(e),
                error_type=type(e).__name__,
            )
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Error: Failed to delete memory - {str(e)}",
                    }
                ],
                "isError": True,
            }

        except Exception as e:
            # Unexpected error
            log.error(
                "delete_memory_unexpected_error",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Error: An unexpected error occurred while deleting memory",
                    }
                ],
                "isError": True,
            }
