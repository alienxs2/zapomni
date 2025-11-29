"""
GetRelated MCP Tool - Full Implementation.

Searches for entities related to a given entity through graph traversal.
Uses FalkorDB graph traversal up to 5 hops away.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

import uuid
from typing import Any, Dict, List, Optional

import structlog
from pydantic import BaseModel, ConfigDict, ValidationError

from zapomni_core.exceptions import (
    DatabaseError,
)
from zapomni_core.exceptions import ValidationError as CoreValidationError
from zapomni_core.memory_processor import MemoryProcessor

logger = structlog.get_logger(__name__)


class GetRelatedRequest(BaseModel):
    """Pydantic model for validating get_related request."""

    model_config = ConfigDict(extra="forbid")

    entity_id: str
    depth: int = 2
    limit: int = 20
    relationship_types: Optional[List[str]] = None


class GetRelatedTool:
    """
    MCP tool for finding related entities through graph traversal.

    This tool validates input, delegates to MemoryProcessor.get_related_entities(),
    and formats the response according to MCP protocol.

    Attributes:
        name: Tool identifier ("get_related")
        description: Human-readable tool description
        input_schema: JSON Schema for input validation
        memory_processor: MemoryProcessor instance for traversal
        logger: Structured logger for operations
    """

    name = "get_related"
    description = (
        "Find entities related to a given entity through graph traversal. "
        "Performs breadth-first search up to 5 hops away. "
        "Returns related entities sorted by relationship strength. "
        "Use this to explore connections and relationships in your knowledge graph."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "entity_id": {
                "type": "string",
                "description": (
                    "UUID of the entity to find related entities for. "
                    "The entity must exist in the knowledge graph."
                ),
            },
            "depth": {
                "type": "integer",
                "description": (
                    "Maximum traversal depth in hops (default: 2, range: 1-5). "
                    "Depth 1 returns direct, depth 2 includes connections of connections."
                ),
                "default": 2,
                "minimum": 1,
                "maximum": 5,
            },
            "limit": {
                "type": "integer",
                "description": (
                    "Maximum number of related entities to return (default: 20, range: 1-50). "
                    "Results are sorted by relationship strength."
                ),
                "default": 20,
                "minimum": 1,
                "maximum": 50,
            },
            "relationship_types": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional list of relationship types to filter by. "
                    "Examples: ['MENTIONS', 'RELATED_TO', 'DEPENDS_ON']. "
                    "If not provided, all relationship types are included."
                ),
            },
        },
        "required": ["entity_id"],
    }

    def __init__(self, memory_processor: MemoryProcessor) -> None:
        """
        Initialize GetRelatedTool with MemoryProcessor.

        Args:
            memory_processor: MemoryProcessor instance for executing traversal.
                Must be initialized and connected to database.

        Raises:
            TypeError: If memory_processor is not a MemoryProcessor instance

        Example:
            >>> processor = MemoryProcessor(...)
            >>> tool = GetRelatedTool(memory_processor=processor)
        """
        if not isinstance(memory_processor, MemoryProcessor):
            raise TypeError(
                f"memory_processor must be MemoryProcessor instance, got {type(memory_processor)}"
            )

        self.memory_processor = memory_processor
        self.logger = logger.bind(tool=self.name)

        self.logger.info("get_related_tool_initialized")

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute get_related tool with provided arguments.

        This is the main entry point called by the MCP server when a client
        invokes the get_related tool.

        Args:
            arguments: Dictionary containing:
                - entity_id (str, required): UUID of entity to find relations for
                - depth (int, optional): Traversal depth (default: 2)
                - limit (int, optional): Max results (default: 20)
                - relationship_types (list, optional): Filter by relationship types

        Returns:
            MCP-formatted response dictionary

        Example:
            >>> result = await tool.execute({
            ...     "entity_id": "550e8400-e29b-41d4-a716-446655440000",
            ...     "depth": 2,
            ...     "limit": 10
            ... })
            >>> print(result["isError"])
            False
        """
        request_id = id(arguments)  # Simple request ID for logging
        log = self.logger.bind(request_id=request_id)

        try:
            # Step 1: Validate input
            log.info("validating_arguments")
            request = self._validate_input(arguments)

            # Step 2: Execute graph traversal
            log.info(
                "executing_traversal",
                entity_id=request.entity_id,
                depth=request.depth,
                limit=request.limit,
                has_filters=request.relationship_types is not None,
            )
            results = await self.memory_processor.get_related_entities(
                entity_id=request.entity_id,
                depth=request.depth,
                limit=request.limit,
            )

            # Step 3: Format response
            log.info("traversal_completed", related_count=len(results))
            return self._format_response(request.entity_id, results)

        except (ValidationError, CoreValidationError) as e:
            # Input validation failed
            log.warning("validation_error", error=str(e))
            return self._format_error(e)

        except DatabaseError as e:
            # Database error
            log.error(
                "database_error",
                error=str(e),
                exc_info=True,
            )
            return self._format_error(e)

        except Exception as e:
            # Unexpected error
            log.error(
                "unexpected_error",
                error_type=type(e).__name__,
                error=str(e),
                exc_info=True,
            )
            return self._format_error(e)

    def _validate_input(self, arguments: Dict[str, Any]) -> GetRelatedRequest:
        """
        Validate and parse tool arguments into GetRelatedRequest model.

        Args:
            arguments: Raw arguments dictionary from MCP client

        Returns:
            GetRelatedRequest: Validated request model

        Raises:
            ValidationError: If arguments don't match schema
        """
        # Validate using Pydantic model
        try:
            request = GetRelatedRequest(**arguments)
        except ValidationError:
            # Re-raise as is for handling upstream
            raise

        # Additional validation: entity_id must be valid UUID
        if request.entity_id:
            try:
                uuid.UUID(request.entity_id)
            except ValueError:
                raise CoreValidationError(
                    message=f"entity_id must be a valid UUID, got: {request.entity_id}",
                    error_code="INVALID_UUID",
                )

        # Additional validation: depth bounds
        if request.depth < 1 or request.depth > 5:
            raise CoreValidationError(
                message=f"depth must be in range [1, 5], got: {request.depth}",
                error_code="INVALID_DEPTH",
            )

        # Additional validation: limit bounds
        if request.limit < 1 or request.limit > 50:
            raise CoreValidationError(
                message=f"limit must be in range [1, 50], got: {request.limit}",
                error_code="INVALID_LIMIT",
            )

        return request

    def _format_response(self, entity_id: str, results: List[Any]) -> Dict[str, Any]:
        """
        Format related entities as MCP response.

        Args:
            entity_id: The entity UUID that was queried
            results: List of Entity objects from processor

        Returns:
            MCP response dictionary
        """
        if not results:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"No related entities found for entity {entity_id}.",
                    }
                ],
                "isError": False,
            }

        # Build formatted text response
        lines = [f"Found {len(results)} related entities:\n"]

        for i, entity in enumerate(results, start=1):
            # Format: "N. [Type] Name (Confidence: 0.XX)"
            type_str = entity.type if hasattr(entity, "type") else "UNKNOWN"
            name_str = entity.name if hasattr(entity, "name") else "Unknown"
            conf = getattr(entity, "confidence", 1.0)
            conf_str = f"{conf:.2f}"

            description = ""
            if hasattr(entity, "description") and entity.description:
                description = f"\n  Description: {entity.description[:100]}"
                if len(entity.description) > 100:
                    description += "..."

            lines.append(f"\n{i}. [{type_str}] {name_str} (Confidence: {conf_str}){description}\n")

        formatted_text = "".join(lines)

        return {
            "content": [
                {
                    "type": "text",
                    "text": formatted_text,
                }
            ],
            "isError": False,
        }

    def _format_error(self, error: Exception) -> Dict[str, Any]:
        """
        Format exception into MCP error response.

        Args:
            error: Exception that occurred during execution

        Returns:
            MCP error response dictionary
        """
        # Determine error message based on exception type
        if isinstance(error, (ValidationError, CoreValidationError)):
            # Extract field-level errors if available
            if hasattr(error, "errors"):
                error_msgs = []
                for err in error.errors():
                    field = ".".join(str(loc) for loc in err["loc"])
                    message = err["msg"]
                    error_msgs.append(f"{field}: {message}")
                error_msg = f"Invalid input - {'; '.join(error_msgs)}"
            else:
                error_msg = f"Invalid input: {str(error)}"

        elif isinstance(error, DatabaseError):
            error_msg = f"Database error: {str(error)}"

        else:
            # Unknown error - generic message for security
            error_msg = "An unexpected error occurred during graph traversal."

        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Error: {error_msg}",
                }
            ],
            "isError": True,
        }
