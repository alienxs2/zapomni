"""
AddMemory MCP Tool - Full Implementation.

Stores new information in memory with semantic chunking and embeddings.
Delegates to MemoryProcessor for all processing operations.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

import time
from typing import Any, Dict, Tuple

import structlog
from pydantic import BaseModel, ConfigDict, StringConstraints, ValidationError
from typing_extensions import Annotated

from zapomni_core.exceptions import (
    DatabaseError,
    EmbeddingError,
    ProcessingError,
)
from zapomni_core.exceptions import ValidationError as CoreValidationError
from zapomni_core.memory_processor import MemoryProcessor

logger = structlog.get_logger(__name__)


class AddMemoryRequest(BaseModel):
    """Pydantic model for validating add_memory request."""

    model_config = ConfigDict(extra="forbid")

    text: Annotated[str, StringConstraints(min_length=1, max_length=10_000_000)]
    metadata: Dict[str, Any] = {}
    workspace_id: str = None  # Optional workspace ID override


class AddMemoryResponse(BaseModel):
    """Pydantic model for add_memory response."""

    status: str
    memory_id: str
    chunks_created: int
    text_preview: str
    error: str = None


class AddMemoryTool:
    """
    MCP tool for adding new memories.

    This tool validates input, delegates to MemoryProcessor.add_memory(),
    and formats the response according to MCP protocol.

    Attributes:
        name: Tool identifier ("add_memory")
        description: Human-readable tool description
        input_schema: JSON Schema for input validation
        memory_processor: MemoryProcessor instance for processing
        mcp_server: Optional MCPServer instance for workspace resolution
        logger: Structured logger for operations
    """

    name = "add_memory"
    description = (
        "Add a memory (text or code) to the knowledge graph. "
        "The memory will be processed, chunked, embedded, and stored for later retrieval."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": (
                    "Text content to remember. Can be natural language, code, "
                    "documentation, or any UTF-8 text. Maximum 10MB."
                ),
                "minLength": 1,
                "maxLength": 10_000_000,
            },
            "metadata": {
                "type": "object",
                "description": (
                    "Optional metadata to attach to this memory. "
                    "Useful for filtering and organization."
                ),
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Source of the memory (e.g., 'user', 'api', 'file')",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for categorization",
                    },
                    "timestamp": {
                        "type": "string",
                        "format": "date-time",
                        "description": "Optional timestamp (ISO 8601 format)",
                    },
                    "language": {
                        "type": "string",
                        "description": "Programming language if text is code (e.g., 'python')",
                    },
                },
                "additionalProperties": True,
            },
            "workspace_id": {
                "type": "string",
                "description": (
                    "Optional workspace ID. If not specified, uses the current session workspace."
                ),
            },
        },
        "required": ["text"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        memory_processor: MemoryProcessor,
        mcp_server: Any = None,
    ) -> None:
        """
        Initialize AddMemoryTool with MemoryProcessor.

        Args:
            memory_processor: MemoryProcessor instance for processing memories.
                Must be initialized and connected to database.
            mcp_server: Optional MCPServer instance for workspace resolution.
                If provided, tool will use resolve_workspace_id() to get
                current session workspace when workspace_id not in arguments.

        Raises:
            TypeError: If memory_processor is not a MemoryProcessor instance
            ValueError: If memory_processor is not initialized

        Example:
            >>> processor = MemoryProcessor(...)
            >>> tool = AddMemoryTool(memory_processor=processor, mcp_server=server)
        """
        if not isinstance(memory_processor, MemoryProcessor):
            raise TypeError(
                f"memory_processor must be MemoryProcessor instance, got {type(memory_processor)}"
            )

        self.memory_processor = memory_processor
        self.mcp_server = mcp_server
        self.logger = logger.bind(tool=self.name)

        self.logger.info("add_memory_tool_initialized")

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute add_memory tool with provided arguments.

        This is the main entry point called by the MCP server when a client
        invokes the add_memory tool. It validates inputs, processes the memory
        through the core engine, and returns a formatted MCP response.

        Args:
            arguments: Dictionary containing:
                - text (str, required): Memory content to store
                - metadata (dict, optional): Additional metadata

        Returns:
            MCP-formatted response dictionary

        Example:
            >>> result = await tool.execute({
            ...     "text": "Claude is an AI assistant",
            ...     "metadata": {"source": "user"}
            ... })
            >>> print(result["isError"])
            False
        """
        request_id = id(arguments)  # Simple request ID for logging
        log = self.logger.bind(request_id=request_id)
        start_time = time.time()

        try:
            # Step 1: Validate and extract arguments
            log.info("validating_arguments")
            text, metadata = self._validate_arguments(arguments)

            # Step 2: Resolve workspace_id
            workspace_id = arguments.get("workspace_id")
            if workspace_id is None and self.mcp_server is not None:
                # Use session workspace from mcp_server
                workspace_id = self.mcp_server.resolve_workspace_id()
                log.debug("resolved_workspace_id", workspace_id=workspace_id)

            # Step 3: Process memory via processor
            log.info(
                "processing_memory",
                text_length=len(text),
                has_metadata=bool(metadata),
                workspace_id=workspace_id,
            )
            memory_id = await self.memory_processor.add_memory(
                text=text,
                metadata=metadata or {},
                workspace_id=workspace_id,
            )

            # Step 4: Get preview of text for response
            text_preview = text[:100] + "..." if len(text) > 100 else text

            # Step 5: Format success response
            processing_time_ms = (time.time() - start_time) * 1000
            log.info(
                "memory_added_successfully",
                memory_id=memory_id,
                processing_time_ms=processing_time_ms,
            )
            response = self._format_success(memory_id, text_preview)
            response["processing_time_ms"] = processing_time_ms
            return response

        except (ValidationError, CoreValidationError) as e:
            # Input validation failed
            processing_time_ms = (time.time() - start_time) * 1000
            log.warning("validation_error", error=str(e))
            response = self._format_error(e)
            response["processing_time_ms"] = processing_time_ms
            return response

        except (EmbeddingError, ProcessingError, DatabaseError) as e:
            # Core processing error
            processing_time_ms = (time.time() - start_time) * 1000
            log.error(
                "processing_error",
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

    def _validate_arguments(
        self,
        arguments: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Validate and extract arguments from MCP request.

        Args:
            arguments: Raw arguments dictionary from MCP client

        Returns:
            Tuple of (text, metadata)

        Raises:
            ValidationError: If arguments don't match schema
        """
        # Validate using Pydantic model
        try:
            request = AddMemoryRequest(**arguments)
        except ValidationError as e:
            # Re-raise as is for handling upstream
            raise

        # Extract and sanitize
        text = request.text.strip()

        # Validate text is not empty after stripping
        if not text:
            raise ValidationError.from_exception_data(
                "AddMemoryRequest",
                [
                    {
                        "type": "value_error",
                        "loc": ("text",),
                        "msg": "text cannot be empty or contain only whitespace",
                        "input": arguments.get("text", ""),
                    }
                ],
            )

        metadata = request.metadata or {}

        return text, metadata

    def _format_success(self, memory_id: str, text_preview: str) -> Dict[str, Any]:
        """
        Format successful memory addition as MCP response.

        Args:
            memory_id: UUID of stored memory
            text_preview: Preview of text

        Returns:
            MCP response dictionary
        """
        message = f"Memory stored successfully.\n" f"ID: {memory_id}\n" f"Preview: {text_preview}"

        return {
            "content": [
                {
                    "type": "text",
                    "text": message,
                }
            ],
            "isError": False,
        }

    def _format_error(self, error: Exception) -> Dict[str, Any]:
        """
        Format error as MCP error response.

        Args:
            error: Exception that occurred during processing

        Returns:
            MCP error response dictionary
        """
        # Determine error message based on exception type
        if isinstance(error, (ValidationError, CoreValidationError)):
            # Validation error - safe to expose
            error_msg = str(error)
        elif isinstance(error, DatabaseError):
            # Database error - suggest retry
            error_msg = "Database temporarily unavailable. Please retry in a few seconds."
        elif isinstance(error, EmbeddingError):
            # Embedding error
            error_msg = "Failed to process text for embedding. Please try again."
        elif isinstance(error, ProcessingError):
            # Processing error
            error_msg = "Failed to process memory. Please try again."
        else:
            # Unknown error - generic message for security
            error_msg = "An internal error occurred while processing your memory."

        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Error: {error_msg}",
                }
            ],
            "isError": True,
        }
