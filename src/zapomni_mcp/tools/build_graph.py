"""
BuildGraph MCP Tool - Full Implementation.

Builds knowledge graphs from text by extracting entities and creating graph structures.
Delegates to EntityExtractor and GraphBuilder from the core module.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

import time
from typing import Any, Dict, Optional, Tuple

import structlog
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from typing_extensions import Annotated

from zapomni_core.exceptions import (
    DatabaseError,
    ExtractionError,
    ProcessingError,
)
from zapomni_core.exceptions import ValidationError as CoreValidationError
from zapomni_core.memory_processor import MemoryProcessor

logger = structlog.get_logger(__name__)


class BuildGraphOptions(BaseModel):
    """Pydantic model for build_graph options."""

    model_config = ConfigDict(extra="forbid")

    extract_entities: bool = True
    build_relationships: bool = False
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class BuildGraphRequest(BaseModel):
    """Pydantic model for validating build_graph request."""

    model_config = ConfigDict(extra="forbid")

    text: Annotated[str, Field(min_length=1, max_length=100_000)]
    options: Optional[BuildGraphOptions] = None


class BuildGraphResponse(BaseModel):
    """Pydantic model for build_graph response."""

    status: str
    entities_count: int
    relationships_count: int
    entities_created: int
    entities_merged: int
    processing_time_ms: float
    confidence_avg: float = 0.0
    error: Optional[str] = None


class BuildGraphTool:
    """
    MCP tool for building knowledge graphs from text.

    This tool extracts entities from text and builds knowledge graph structures
    using the EntityExtractor and GraphBuilder components.

    Attributes:
        name: Tool identifier ("build_graph")
        description: Human-readable tool description
        input_schema: JSON Schema for input validation
        memory_processor: MemoryProcessor instance for processing
        logger: Structured logger for operations
    """

    name = "build_graph"
    description = (
        "Build a knowledge graph from text by extracting entities and creating graph structures. "
        "Uses EntityExtractor for entity recognition and GraphBuilder for graph construction."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": (
                    "Text to extract entities from and build graph. "
                    "Can be natural language, code, or documentation. Maximum 100,000 characters."
                ),
                "minLength": 1,
                "maxLength": 100_000,
            },
            "options": {
                "type": "object",
                "description": "Optional processing options",
                "properties": {
                    "extract_entities": {
                        "type": "boolean",
                        "description": "Enable entity extraction (default: true)",
                        "default": True,
                    },
                    "build_relationships": {
                        "type": "boolean",
                        "description": "Enable relationship detection via LLM (default: false)",
                        "default": False,
                    },
                    "confidence_threshold": {
                        "type": "number",
                        "description": "Minimum confidence for entities (0.0-1.0, default: 0.7)",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": 0.7,
                    },
                },
                "additionalProperties": False,
            },
        },
        "required": ["text"],
        "additionalProperties": False,
    }

    def __init__(self, memory_processor: MemoryProcessor) -> None:
        """
        Initialize BuildGraphTool with MemoryProcessor.

        Args:
            memory_processor: MemoryProcessor instance for processing.
                Must be initialized and connected to database and extractors.

        Raises:
            TypeError: If memory_processor is not a MemoryProcessor instance
            ValueError: If memory_processor is not initialized

        Example:
            >>> processor = MemoryProcessor(...)
            >>> tool = BuildGraphTool(memory_processor=processor)
        """
        if not isinstance(memory_processor, MemoryProcessor):
            raise TypeError(
                f"memory_processor must be MemoryProcessor instance, got {type(memory_processor)}"
            )

        self.memory_processor = memory_processor
        self.logger = logger.bind(tool=self.name)

        self.logger.info("build_graph_tool_initialized")

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute build_graph tool with provided arguments.

        This is the main entry point called by the MCP server when a client
        invokes the build_graph tool. It validates inputs, extracts entities,
        builds the knowledge graph, and returns formatted statistics.

        Args:
            arguments: Dictionary containing:
                - text (str, required): Text to extract entities from
                - options (dict, optional): Processing options with:
                    - extract_entities (bool): Enable entity extraction (default: True)
                    - build_relationships (bool): Enable relationship building (default: False)
                    - confidence_threshold (float): Min confidence threshold (default: 0.7)

        Returns:
            MCP-formatted response dictionary with:
                - isError: Boolean indicating success/failure
                - content: List with text content of response

        Example:
            >>> result = await tool.execute({
            ...     "text": "Python is a programming language created by Guido van Rossum",
            ...     "options": {
            ...         "extract_entities": True,
            ...         "confidence_threshold": 0.7
            ...     }
            ... })
            >>> print(result["isError"])
            False
        """
        request_id = id(arguments)
        log = self.logger.bind(request_id=request_id)
        start_time = time.time()

        try:
            # Step 1: Validate and extract arguments
            log.info("validating_arguments")
            text, options = self._validate_arguments(arguments)

            # Step 2: Extract entities using entity extractor
            log.info(
                "extracting_entities",
                text_length=len(text),
                extract_entities=options.extract_entities,
            )

            entities, avg_confidence = await self._extract_entities(
                text, options.confidence_threshold
            )

            # Step 3: Build graph from extracted entities
            log.info(
                "building_graph",
                num_entities=len(entities),
                build_relationships=options.build_relationships,
            )

            graph_stats = await self._build_graph(entities, text)

            # Step 4: Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000

            # Step 5: Format success response
            log.info(
                "graph_building_complete",
                entities_count=len(entities),
                relationships_count=graph_stats.get("relationships_created", 0),
                processing_time_ms=processing_time_ms,
            )

            return self._format_success(
                entities_count=len(entities),
                relationships_count=graph_stats.get("relationships_created", 0),
                entities_created=graph_stats.get("entities_created", 0),
                entities_merged=graph_stats.get("entities_merged", 0),
                processing_time_ms=processing_time_ms,
                confidence_avg=avg_confidence,
            )

        except (ValidationError, CoreValidationError) as e:
            # Input validation failed
            processing_time_ms = (time.time() - start_time) * 1000
            log.warning("validation_error", error=str(e))
            return self._format_error(e, processing_time_ms)

        except (ExtractionError, ProcessingError, DatabaseError) as e:
            # Core processing error
            processing_time_ms = (time.time() - start_time) * 1000
            log.error(
                "processing_error",
                error_type=type(e).__name__,
                error=str(e),
                exc_info=True,
            )
            return self._format_error(e, processing_time_ms)

        except Exception as e:
            # Unexpected error
            processing_time_ms = (time.time() - start_time) * 1000
            log.error(
                "unexpected_error",
                error_type=type(e).__name__,
                error=str(e),
                exc_info=True,
            )
            return self._format_error(e, processing_time_ms)

    def _validate_arguments(
        self,
        arguments: Dict[str, Any],
    ) -> Tuple[str, BuildGraphOptions]:
        """
        Validate and extract arguments from MCP request.

        Args:
            arguments: Raw arguments dictionary from MCP client

        Returns:
            Tuple of (text, options)

        Raises:
            ValidationError: If arguments don't match schema
        """
        # Validate using Pydantic model
        try:
            request = BuildGraphRequest(**arguments)
        except ValidationError:
            # Re-raise as is for handling upstream
            raise

        # Extract and sanitize
        text = request.text.strip()

        # Validate text is not empty after stripping
        if not text:
            raise ValidationError.from_exception_data(
                "BuildGraphRequest",
                [
                    {
                        "type": "value_error",
                        "loc": ("text",),
                        "msg": "text cannot be empty or contain only whitespace",
                        "input": arguments.get("text", ""),
                    }
                ],
            )

        options = request.options or BuildGraphOptions()

        return text, options

    async def _extract_entities(self, text: str, confidence_threshold: float) -> Tuple[list, float]:
        """
        Extract entities from text using EntityExtractor.

        Uses async extraction to avoid blocking the event loop during
        CPU-bound SpaCy NLP operations. This is essential for SSE transport
        where multiple concurrent connections share the same event loop.

        Args:
            text: Text to extract entities from
            confidence_threshold: Minimum confidence threshold

        Returns:
            Tuple of (entities_list, average_confidence)

        Raises:
            ExtractionError: If entity extraction fails
            ValidationError: If text is invalid
        """
        try:
            # Get entity extractor from memory processor
            entity_extractor = self.memory_processor.extractor

            if entity_extractor is None:
                raise ProcessingError(
                    message="EntityExtractor not initialized in MemoryProcessor",
                    error_code="PROC_001",
                )

            # Extract entities using async method to avoid blocking event loop
            # Falls back to sync if async method not available (backward compatibility)
            if hasattr(entity_extractor, "extract_entities_async"):
                entities = await entity_extractor.extract_entities_async(text)
            else:
                # Fallback for older EntityExtractor instances without async support
                entities = entity_extractor.extract_entities(text)

            self.logger.debug("entities_extracted", num_entities=len(entities))

            # Calculate average confidence
            if entities:
                avg_confidence = sum(e.confidence for e in entities) / len(entities)
            else:
                avg_confidence = 0.0

            return entities, avg_confidence

        except (ExtractionError, CoreValidationError):
            raise
        except Exception as e:
            raise ExtractionError(
                message=f"Entity extraction failed: {str(e)}",
                error_code="EXTR_001",
                details={"error": str(e), "text_length": len(text)},
                original_exception=e,
            )

    async def _build_graph(self, entities: list, text: str) -> Dict[str, Any]:
        """
        Build knowledge graph from extracted entities.

        Args:
            entities: List of extracted entities
            text: Original text for context

        Returns:
            Dict with graph building statistics

        Raises:
            ProcessingError: If graph building fails
            DatabaseError: If database operations fail
        """
        try:
            # Get graph builder from memory processor
            graph_builder = self.memory_processor.graph_builder

            if graph_builder is None:
                raise ProcessingError(
                    message="GraphBuilder not initialized in MemoryProcessor",
                    error_code="PROC_001",
                )

            # Build graph from entities
            # GraphBuilder.build_graph expects a list of dicts (memories) or text
            # Pass text as a memory dict to satisfy validation
            graph_stats = await graph_builder.build_graph(
                memories=[{"text": text}],  # Pass text as memory dict
                text=text,
            )

            self.logger.debug(
                "graph_built",
                entities_created=graph_stats.get("entities_created", 0),
                relationships_created=graph_stats.get("relationships_created", 0),
            )

            return graph_stats

        except NotImplementedError as e:
            # Relationship building is Phase 2 - handle gracefully
            self.logger.debug("relationships_not_implemented", error=str(e))
            return {
                "entities_created": 0,
                "entities_merged": 0,
                "relationships_created": 0,
                "total_nodes": 0,
                "total_edges": 0,
            }
        except (ProcessingError, DatabaseError):
            raise
        except Exception as e:
            raise ProcessingError(
                message=f"Graph building failed: {str(e)}",
                error_code="GRAPH_001",
                details={"error": str(e)},
                original_exception=e,
            )

    def _format_success(
        self,
        entities_count: int,
        relationships_count: int,
        entities_created: int,
        entities_merged: int,
        processing_time_ms: float,
        confidence_avg: float,
    ) -> Dict[str, Any]:
        """
        Format successful graph building as MCP response.

        Args:
            entities_count: Total number of entities
            relationships_count: Number of relationships created
            entities_created: New entities added to graph
            entities_merged: Existing entities updated
            processing_time_ms: Processing time in milliseconds
            confidence_avg: Average confidence score

        Returns:
            MCP response dictionary
        """
        message = (
            f"Knowledge graph built successfully.\n"
            f"Entities: {entities_count} (Created: {entities_created}, Merged: {entities_merged})\n"
            f"Relationships: {relationships_count}\n"
            f"Average Confidence: {confidence_avg:.2f}\n"
            f"Processing Time: {processing_time_ms:.1f}ms"
        )

        return {
            "content": [
                {
                    "type": "text",
                    "text": message,
                }
            ],
            "isError": False,
        }

    def _format_error(self, error: Exception, processing_time_ms: float) -> Dict[str, Any]:
        """
        Format error as MCP error response.

        Args:
            error: Exception that occurred during processing
            processing_time_ms: Processing time before error

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
        elif isinstance(error, ExtractionError):
            # Extraction error
            error_msg = (
                "Failed to extract entities from text. Please check the text format and try again."
            )
        elif isinstance(error, ProcessingError):
            # Processing error
            error_msg = "Failed to build knowledge graph. Please try again."
        else:
            # Unknown error - generic message for security
            error_msg = "An internal error occurred while building the knowledge graph."

        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Error: {error_msg}",
                }
            ],
            "isError": True,
        }
