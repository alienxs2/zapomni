"""
ClearAll MCP Tool - Full Implementation.

Clears all data from the graph database with required safety confirmation.
Requires explicit confirmation phrase "DELETE ALL MEMORIES" to prevent accidental deletions.
Delegates to MemoryProcessor for the database operation.

WARNING: This is a HIGHLY DESTRUCTIVE operation that cannot be undone.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from typing import Any, Dict, Optional

import structlog
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from zapomni_core.exceptions import DatabaseError
from zapomni_core.exceptions import ValidationError as CoreValidationError
from zapomni_core.memory_processor import MemoryProcessor

logger = structlog.get_logger(__name__)

# Required phrase for confirmation
REQUIRED_CONFIRMATION_PHRASE = "DELETE ALL MEMORIES"


class ClearAllRequest(BaseModel):
    """Pydantic model for validating clear_all request."""

    model_config = ConfigDict(extra="forbid")

    confirm_phrase: str

    @field_validator("confirm_phrase")
    @classmethod
    def validate_confirm_phrase(cls, v: str) -> str:
        """Validate that confirm_phrase is provided."""
        if not v:
            raise ValueError("confirm_phrase cannot be empty")
        return v


class ClearAllResponse(BaseModel):
    """Pydantic model for clear_all response."""

    status: str
    cleared: bool
    deleted_count: int
    message: str = ""
    error: Optional[str] = None


class ClearAllTool:
    """
    MCP tool for clearing all data from the knowledge graph.

    This tool validates input with required safety confirmation phrase,
    delegates to MemoryProcessor.db_client.clear_all(),
    and formats the response according to MCP protocol.

    IMPORTANT: Requires EXACT phrase "DELETE ALL MEMORIES" to prevent accidental usage.
    This is a safety feature to ensure users explicitly understand the destructive nature.
    This operation is PERMANENT and IRREVERSIBLE - all memories will be deleted.

    Attributes:
        name: Tool identifier ("clear_all")
        description: Human-readable tool description
        input_schema: JSON Schema for input validation
        memory_processor: MemoryProcessor instance for database access
        logger: Structured logger for operations
    """

    name = "clear_all"
    description = (
        "DESTRUCTIVE: Clear ALL data from the knowledge graph. "
        "REQUIRES exact confirmation phrase 'DELETE ALL MEMORIES'. "
        "This operation is PERMANENT and IRREVERSIBLE - all stored memories will be deleted. "
        "This is a safety feature to ensure explicit user understanding and intent."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "confirm_phrase": {
                "type": "string",
                "description": (
                    "REQUIRED: Must be EXACTLY 'DELETE ALL MEMORIES' to proceed. "
                    "This is case-sensitive. "
                    "No partial matches or variations accepted. "
                    "This safety check ensures users understand the destructive nature."
                ),
            },
        },
        "required": ["confirm_phrase"],
        "additionalProperties": False,
    }

    def __init__(self, memory_processor: MemoryProcessor) -> None:
        """
        Initialize ClearAllTool with MemoryProcessor.

        Args:
            memory_processor: MemoryProcessor instance for database operations.
                Must be initialized and connected to database.

        Raises:
            TypeError: If memory_processor is not a MemoryProcessor instance

        Example:
            >>> processor = MemoryProcessor(...)
            >>> tool = ClearAllTool(memory_processor=processor)
        """
        if not isinstance(memory_processor, MemoryProcessor):
            raise TypeError(
                f"memory_processor must be MemoryProcessor instance, got {type(memory_processor)}"
            )

        self.memory_processor = memory_processor
        self.logger = logger.bind(tool=self.name)

        self.logger.info("clear_all_tool_initialized")

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute clear_all tool with provided arguments.

        This is the main entry point called by the MCP server when a client
        invokes the clear_all tool. It validates inputs with mandatory
        confirmation phrase, clears all data through the database client,
        and returns a formatted MCP response.

        Args:
            arguments: Dictionary containing:
                - confirm_phrase (str, required): MUST be "DELETE ALL MEMORIES"

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
            >>> tool = ClearAllTool(memory_processor=processor)
            >>> result = await tool.execute({
            ...     "confirm_phrase": "DELETE ALL MEMORIES"
            ... })
            >>> print(result["isError"])
            False
        """
        log = self.logger.bind(request_id=id(arguments))

        try:
            # Validate arguments type
            if not isinstance(arguments, dict):
                log.warning(
                    "clear_all_invalid_arguments_type",
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
            log.debug("parsing_clear_all_request", arguments=arguments)
            request = ClearAllRequest(**arguments)

            # Safety check: confirm_phrase must match EXACTLY
            if request.confirm_phrase != REQUIRED_CONFIRMATION_PHRASE:
                log.warning(
                    "clear_all_confirmation_failed",
                    provided_phrase=request.confirm_phrase,
                    required_phrase=REQUIRED_CONFIRMATION_PHRASE,
                )
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Error: Invalid confirmation phrase. "
                                f"To clear all memories, provide EXACT phrase:\n"
                                f'confirm_phrase: "{REQUIRED_CONFIRMATION_PHRASE}"\n\n'
                                f"This operation is PERMANENT and IRREVERSIBLE. "
                                f"All stored memories will be deleted. "
                                f"The phrase is case-sensitive and no partial matches are accepted."
                            ),
                        }
                    ],
                    "isError": True,
                }

            # Log clear operation with warning level (destructive operation)
            log.warning(
                "clear_all_requested",
                confirm_phrase=request.confirm_phrase,
            )

            # Execute clear operation through database client
            await self.memory_processor.db_client.clear_all()

            # Format response
            log.warning("clear_all_success")
            response = {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"All memories cleared successfully!\n"
                            f"Status: CLEARED\n"
                            f"All data has been permanently deleted from the knowledge graph.\n"
                            f"WARNING: This operation is irreversible."
                        ),
                    }
                ],
                "isError": False,
            }

            return response

        except ValidationError as e:
            # Pydantic validation error
            log.warning(
                "clear_all_validation_error",
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
            # Core validation error
            log.error(
                "clear_all_database_validation_error",
                error=str(e),
                error_type=type(e).__name__,
            )
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Error: Validation failed - {str(e)}",
                    }
                ],
                "isError": True,
            }

        except DatabaseError as e:
            # Database error during clear operation
            log.error(
                "clear_all_database_error",
                error=str(e),
                error_type=type(e).__name__,
            )
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Error: Failed to clear memories - {str(e)}",
                    }
                ],
                "isError": True,
            }

        except Exception as e:
            # Unexpected error
            log.error(
                "clear_all_unexpected_error",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Error: An unexpected error occurred while clearing memories",
                    }
                ],
                "isError": True,
            }
