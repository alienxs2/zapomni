"""
SearchMemory MCP Tool - Full Implementation.

Searches for relevant information in memory using semantic similarity.
Delegates to MemoryProcessor for all search operations.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

import time
from typing import Any, Dict, Optional

import structlog
from pydantic import BaseModel, ConfigDict, StringConstraints, ValidationError
from typing_extensions import Annotated

from zapomni_core.exceptions import (
    DatabaseError,
    EmbeddingError,
    SearchError,
)
from zapomni_core.exceptions import ValidationError as CoreValidationError
from zapomni_core.memory_processor import MemoryProcessor

logger = structlog.get_logger(__name__)


class SearchMemoryRequest(BaseModel):
    """Pydantic model for validating search_memory request."""

    model_config = ConfigDict(extra="forbid")

    query: Annotated[str, StringConstraints(min_length=1, max_length=1000)]
    limit: int = 10
    filters: Optional[Dict[str, Any]] = None
    workspace_id: Optional[str] = None  # Optional workspace ID override


class SearchMemoryTool:
    """
    MCP tool for searching memories.

    This tool validates input, delegates to MemoryProcessor.search_memory(),
    and formats the response according to MCP protocol.

    Attributes:
        name: Tool identifier ("search_memory")
        description: Human-readable tool description
        input_schema: JSON Schema for input validation
        memory_processor: MemoryProcessor instance for searching
        mcp_server: Optional MCPServer instance for workspace resolution
        logger: Structured logger for operations
    """

    name = "search_memory"
    description = (
        "Search your personal memory graph for information. "
        "Performs semantic search to find relevant memories based on meaning, "
        "not just keyword matching. Returns ranked results with similarity scores. "
        "Use this when you need to recall previously stored information."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Natural language search query (e.g., 'information about Python', "
                    "'my notes on machine learning', 'what did I learn about Docker?')"
                ),
                "minLength": 1,
                "maxLength": 1000,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return (default: 10, max: 100)",
                "default": 10,
                "minimum": 1,
                "maximum": 100,
            },
            "filters": {
                "type": "object",
                "description": (
                    "Optional metadata filters to narrow results. "
                    "Supported keys: 'tags' (list), 'source' (string), "
                    "'date_from' (ISO date), 'date_to' (ISO date)"
                ),
                "properties": {
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Only return memories with these tags",
                    },
                    "source": {
                        "type": "string",
                        "description": "Only return memories from this source",
                    },
                    "date_from": {
                        "type": "string",
                        "format": "date",
                        "description": "Only memories created after this date (YYYY-MM-DD)",
                    },
                    "date_to": {
                        "type": "string",
                        "format": "date",
                        "description": "Only memories created before this date (YYYY-MM-DD)",
                    },
                },
                "additionalProperties": False,
            },
            "workspace_id": {
                "type": "string",
                "description": (
                    "Optional workspace ID. If not specified, uses the current session workspace."
                ),
            },
        },
        "required": ["query"],
    }

    def __init__(
        self,
        memory_processor: MemoryProcessor,
        mcp_server: Any = None,
    ) -> None:
        """
        Initialize SearchMemoryTool with MemoryProcessor.

        Args:
            memory_processor: MemoryProcessor instance for executing searches.
                Must be initialized and connected to database.
            mcp_server: Optional MCPServer instance for workspace resolution.
                If provided, tool will use resolve_workspace_id() to get
                current session workspace when workspace_id not in arguments.

        Raises:
            TypeError: If memory_processor is not a MemoryProcessor instance

        Example:
            >>> processor = MemoryProcessor(...)
            >>> tool = SearchMemoryTool(memory_processor=processor, mcp_server=server)
        """
        if not isinstance(memory_processor, MemoryProcessor):
            raise TypeError(
                f"memory_processor must be MemoryProcessor instance, got {type(memory_processor)}"
            )

        self.memory_processor = memory_processor
        self.mcp_server = mcp_server
        self.logger = logger.bind(tool=self.name)

        self.logger.info("search_memory_tool_initialized")

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute search_memory tool with provided arguments.

        This is the main entry point called by the MCP server when a client
        invokes the search_memory tool.

        Args:
            arguments: Dictionary containing:
                - query (str, required): Search query
                - limit (int, optional): Max results (default: 10)
                - filters (dict, optional): Metadata filters

        Returns:
            MCP-formatted response dictionary

        Example:
            >>> result = await tool.execute({
            ...     "query": "Python programming",
            ...     "limit": 5
            ... })
            >>> print(result["isError"])
            False
        """
        request_id = id(arguments)  # Simple request ID for logging
        log = self.logger.bind(request_id=request_id)
        start_time = time.time()

        try:
            # Step 1: Validate input
            log.info("validating_arguments")
            request = self._validate_input(arguments)

            # Step 2: Resolve workspace_id
            workspace_id = arguments.get("workspace_id")
            if workspace_id is None and self.mcp_server is not None:
                # Use session workspace from mcp_server
                workspace_id = self.mcp_server.resolve_workspace_id()
                log.debug("resolved_workspace_id", workspace_id=workspace_id)

            # Step 3: Execute search
            log.info(
                "executing_search",
                query_length=len(request.query),
                limit=request.limit,
                has_filters=request.filters is not None,
                workspace_id=workspace_id,
            )
            results = await self.memory_processor.search_memory(
                query=request.query,
                limit=request.limit,
                filters=request.filters,
                workspace_id=workspace_id,
            )

            # Step 4: Format response
            processing_time_ms = (time.time() - start_time) * 1000
            log.info(
                "search_completed",
                result_count=len(results),
                processing_time_ms=processing_time_ms,
            )
            response = self._format_response(results)
            response["processing_time_ms"] = processing_time_ms
            return response

        except (ValidationError, CoreValidationError) as e:
            # Input validation failed
            processing_time_ms = (time.time() - start_time) * 1000
            log.warning("validation_error", error=str(e))
            response = self._format_error(e)
            response["processing_time_ms"] = processing_time_ms
            return response

        except (SearchError, EmbeddingError, DatabaseError) as e:
            # Search or processing error
            processing_time_ms = (time.time() - start_time) * 1000
            log.error(
                "search_error",
                error_type=type(e).__name__,
                error=str(e),
                exc_info=True,
            )
            response = self._format_error(e)
            response["processing_time_ms"] = processing_time_ms
            return response

        except Exception as e:
            # Unexpected error
            processing_time_ms = (time.time() - start_time) * 1000
            log.error(
                "unexpected_error",
                error_type=type(e).__name__,
                error=str(e),
                exc_info=True,
            )
            response = self._format_error(e)
            response["processing_time_ms"] = processing_time_ms
            return response

    def _validate_input(self, arguments: Dict[str, Any]) -> SearchMemoryRequest:
        """
        Validate and parse tool arguments into SearchMemoryRequest model.

        Args:
            arguments: Raw arguments dictionary from MCP client

        Returns:
            SearchMemoryRequest: Validated request model

        Raises:
            ValidationError: If arguments don't match schema
        """
        # Validate using Pydantic model
        try:
            request = SearchMemoryRequest(**arguments)
        except ValidationError:
            # Re-raise as is for handling upstream
            raise

        # Additional validation: query cannot be empty or whitespace only
        if not request.query or not request.query.strip():
            raise CoreValidationError(
                message="query cannot be empty or contain only whitespace",
                error_code="VAL_001",
            )

        # Additional validation: limit bounds
        if request.limit < 1:
            raise CoreValidationError(
                message="limit must be at least 1",
                error_code="VAL_001",
            )
        if request.limit > 100:
            raise CoreValidationError(
                message="limit cannot exceed 100",
                error_code="VAL_001",
            )

        return request

    def _format_response(self, results: list[Any]) -> Dict[str, Any]:
        """
        Format search results as MCP response.

        Args:
            results: List of SearchResultItem objects from processor

        Returns:
            MCP response dictionary
        """
        if not results:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "No results found matching your query.",
                    }
                ],
                "isError": False,
            }

        # Build formatted text response
        lines = [f"Found {len(results)} results:\n"]

        for i, result in enumerate(results, start=1):
            # Format: "N. [Score: 0.XX] text excerpt"
            score = f"{result.similarity_score:.2f}"
            text_preview = result.text[:200]  # First 200 chars
            if len(result.text) > 200:
                text_preview += "..."

            # Include tags if present
            tags_str = ""
            if result.tags:
                tags_str = f" [Tags: {', '.join(result.tags)}]"

            lines.append(f"\n{i}. [Score: {score}]{tags_str}\n{text_preview}\n")

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

        elif isinstance(error, SearchError):
            error_msg = f"Search failed: {str(error)}"

        elif isinstance(error, EmbeddingError):
            error_msg = "Failed to process search query. Please try again."

        elif isinstance(error, DatabaseError):
            error_msg = "Database error during search. Please try again."

        else:
            # Unknown error - generic message for security
            error_msg = "An unexpected error occurred during search."

        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Error: {error_msg}",
                }
            ],
            "isError": True,
        }
