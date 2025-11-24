"""
ExportGraph MCP Tool - Export knowledge graph to various formats.

Provides MCP interface for GraphExporter functionality.
Supports export to GraphML, Cytoscape JSON, Neo4j Cypher, and simple JSON formats.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from typing import Any, Dict, Tuple, Optional, List
import structlog
from pathlib import Path
from pydantic import ValidationError, BaseModel, Field, ConfigDict, field_validator
from enum import Enum

from zapomni_core.graph.graph_exporter import GraphExporter, ExportResult
from zapomni_core.memory_processor import MemoryProcessor
from zapomni_core.exceptions import (
    ValidationError as CoreValidationError,
    DatabaseError,
    ProcessingError,
)


logger = structlog.get_logger(__name__)


class ExportFormat(str, Enum):
    """Supported export formats."""

    GRAPHML = "graphml"
    CYTOSCAPE = "cytoscape"
    NEO4J = "neo4j"
    JSON = "json"


class ExportOptions(BaseModel):
    """Options for graph export."""

    model_config = ConfigDict(extra="forbid")

    node_types: Optional[List[str]] = Field(
        default=None, description="Filter by node types (e.g., ['Entity', 'Memory'])"
    )
    pretty_print: Optional[bool] = Field(
        default=True, description="Pretty-print output (GraphML, JSON)"
    )
    include_metadata: Optional[bool] = Field(default=True, description="Include export metadata")
    include_style: Optional[bool] = Field(
        default=True, description="Include style hints (Cytoscape only)"
    )
    batch_size: Optional[int] = Field(
        default=1000, description="Batch size for Neo4j export", ge=1, le=10000
    )

    @field_validator("node_types")
    @classmethod
    def validate_node_types(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate node_types list."""
        if v is not None:
            if not isinstance(v, list):
                raise ValueError("node_types must be a list")
            if not all(isinstance(t, str) for t in v):
                raise ValueError("node_types must contain strings")
            if not all(t.strip() for t in v):
                raise ValueError("node_types cannot contain empty strings")
        return v


class ExportGraphRequest(BaseModel):
    """Pydantic model for validating export_graph request."""

    model_config = ConfigDict(extra="forbid")

    format: ExportFormat = Field(description="Export format")
    output_path: str = Field(min_length=1, description="Output file path")
    options: ExportOptions = Field(default_factory=ExportOptions, description="Export options")

    @field_validator("output_path")
    @classmethod
    def validate_output_path(cls, v: str) -> str:
        """Validate output_path is not empty after stripping."""
        if not v.strip():
            raise ValueError("output_path cannot be empty or whitespace")
        return v.strip()


class ExportGraphResponse(BaseModel):
    """Pydantic model for export_graph response."""

    success: bool
    format: str
    output_path: str
    nodes_count: int
    edges_count: int
    file_size_bytes: int
    export_time_ms: float
    error: Optional[str] = None


class ExportGraphTool:
    """
    MCP tool for exporting knowledge graphs to various formats.

    This tool validates input, delegates to GraphExporter methods based on format,
    and formats the response according to MCP protocol.

    Supported formats:
    - graphml: GraphML XML format for Gephi, yEd, NetworkX
    - cytoscape: Cytoscape JSON for Cytoscape.js web visualization
    - neo4j: Neo4j Cypher statements for Neo4j import
    - json: Simple JSON format for backup and processing

    Attributes:
        name: Tool identifier ("export_graph")
        description: Human-readable tool description
        input_schema: JSON Schema for input validation
        memory_processor: MemoryProcessor instance for database access
        logger: Structured logger for operations
    """

    name = "export_graph"
    description = (
        "Export the knowledge graph to various formats for visualization and analysis. "
        "Supports GraphML (Gephi/yEd), Cytoscape JSON (web), Neo4j Cypher, and simple JSON."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "format": {
                "type": "string",
                "description": "Export format: graphml, cytoscape, neo4j, or json",
                "enum": ["graphml", "cytoscape", "neo4j", "json"],
            },
            "output_path": {
                "type": "string",
                "description": (
                    "Absolute path to output file. Extension should match format "
                    "(.graphml, .json, or .cypher)"
                ),
                "minLength": 1,
            },
            "options": {
                "type": "object",
                "description": "Optional export settings",
                "properties": {
                    "node_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by node types (e.g., ['Entity', 'Memory'])",
                    },
                    "pretty_print": {
                        "type": "boolean",
                        "description": "Pretty-print output (default: true)",
                        "default": True,
                    },
                    "include_metadata": {
                        "type": "boolean",
                        "description": "Include export metadata (default: true)",
                        "default": True,
                    },
                    "include_style": {
                        "type": "boolean",
                        "description": "Include style hints for Cytoscape (default: true)",
                        "default": True,
                    },
                    "batch_size": {
                        "type": "integer",
                        "description": "Batch size for Neo4j export (default: 1000)",
                        "default": 1000,
                        "minimum": 1,
                        "maximum": 10000,
                    },
                },
                "additionalProperties": False,
            },
        },
        "required": ["format", "output_path"],
        "additionalProperties": False,
    }

    def __init__(self, memory_processor: MemoryProcessor) -> None:
        """
        Initialize ExportGraphTool with MemoryProcessor.

        Args:
            memory_processor: MemoryProcessor instance for database access.
                Must be initialized and connected to database.

        Raises:
            TypeError: If memory_processor is not a MemoryProcessor instance
            ValueError: If memory_processor is not initialized

        Example:
            >>> processor = MemoryProcessor(...)
            >>> tool = ExportGraphTool(memory_processor=processor)
        """
        if not isinstance(memory_processor, MemoryProcessor):
            raise TypeError(
                f"memory_processor must be MemoryProcessor instance, got {type(memory_processor)}"
            )

        self.memory_processor = memory_processor
        self._exporter = GraphExporter(db_client=memory_processor.db_client)
        self.logger = logger.bind(tool=self.name)

        self.logger.info("export_graph_tool_initialized")

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute export_graph tool with provided arguments.

        This is the main entry point called by the MCP server when a client
        invokes the export_graph tool. It validates inputs, routes to the
        appropriate export method, and returns a formatted MCP response.

        Args:
            arguments: Dictionary containing:
                - format (str, required): Export format (graphml|cytoscape|neo4j|json)
                - output_path (str, required): Path to output file
                - options (dict, optional): Export options

        Returns:
            MCP-formatted response dictionary with export statistics

        Example:
            >>> result = await tool.execute({
            ...     "format": "graphml",
            ...     "output_path": "/tmp/graph.graphml",
            ...     "options": {"pretty_print": True}
            ... })
            >>> print(result["isError"])
            False
        """
        request_id = id(arguments)
        log = self.logger.bind(request_id=request_id)

        try:
            # Step 1: Validate and extract arguments
            log.info("validating_arguments")
            format_type, output_path, options = self._validate_arguments(arguments)

            # Step 2: Route to appropriate export method
            log.info(
                "exporting_graph",
                format=format_type.value,
                output_path=output_path,
                has_options=bool(options),
            )
            export_result = await self._export_graph(format_type, output_path, options)

            # Step 3: Format success response
            log.info(
                "export_complete",
                format=format_type.value,
                nodes=export_result.nodes_count,
                edges=export_result.edges_count,
                size_kb=round(export_result.file_size_bytes / 1024, 2),
            )
            return self._format_success(export_result)

        except (ValidationError, CoreValidationError) as e:
            # Input validation failed
            log.warning("validation_error", error=str(e))
            return self._format_error(e)

        except (ProcessingError, DatabaseError) as e:
            # Core processing/database error
            log.error(
                "processing_error",
                error_type=type(e).__name__,
                error=str(e),
                exc_info=True,
            )
            return self._format_error(e)

        except IOError as e:
            # File write error
            log.error("io_error", error=str(e), exc_info=True)
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

    def _validate_arguments(
        self,
        arguments: Dict[str, Any],
    ) -> Tuple[ExportFormat, str, Dict[str, Any]]:
        """
        Validate and extract arguments from MCP request.

        Args:
            arguments: Raw arguments dictionary from MCP client

        Returns:
            Tuple of (format_type, output_path, options_dict)

        Raises:
            ValidationError: If arguments don't match schema
        """
        # Validate using Pydantic model
        try:
            request = ExportGraphRequest(**arguments)
        except ValidationError as e:
            # Re-raise as is for handling upstream
            raise

        # Extract validated values
        format_type = request.format
        output_path = request.output_path

        # Convert options to dict for GraphExporter
        options_dict = {}
        if request.options.node_types is not None:
            options_dict["node_types"] = request.options.node_types
        if request.options.pretty_print is not None:
            options_dict["pretty_print"] = request.options.pretty_print
        if request.options.include_metadata is not None:
            options_dict["include_metadata"] = request.options.include_metadata
        if request.options.include_style is not None:
            options_dict["include_style"] = request.options.include_style
        if request.options.batch_size is not None:
            options_dict["batch_size"] = request.options.batch_size

        return format_type, output_path, options_dict

    async def _export_graph(
        self, format_type: ExportFormat, output_path: str, options: Dict[str, Any]
    ) -> ExportResult:
        """
        Route export request to appropriate GraphExporter method.

        Args:
            format_type: Export format enum
            output_path: Output file path
            options: Export options dictionary

        Returns:
            ExportResult from GraphExporter

        Raises:
            ProcessingError: If export fails
            DatabaseError: If graph query fails
            IOError: If file write fails
        """
        # Route to appropriate export method based on format
        if format_type == ExportFormat.GRAPHML:
            return await self._exporter.export_graphml(output_path, options)
        elif format_type == ExportFormat.CYTOSCAPE:
            return await self._exporter.export_cytoscape(output_path, options)
        elif format_type == ExportFormat.NEO4J:
            return await self._exporter.export_neo4j(output_path, options)
        elif format_type == ExportFormat.JSON:
            return await self._exporter.export_json(output_path, options)
        else:
            raise CoreValidationError(
                message=f"Unsupported format: {format_type}",
                error_code="VAL_001",
            )

    def _format_success(self, export_result: ExportResult) -> Dict[str, Any]:
        """
        Format successful export as MCP response.

        Args:
            export_result: ExportResult from GraphExporter

        Returns:
            MCP response dictionary with export statistics
        """
        # Create user-friendly message
        size_kb = round(export_result.file_size_bytes / 1024, 2)
        time_sec = round(export_result.export_time_ms / 1000, 2)

        message = (
            f"Graph exported successfully!\n\n"
            f"Format: {export_result.format}\n"
            f"Output: {export_result.output_path}\n"
            f"Nodes: {export_result.nodes_count}\n"
            f"Edges: {export_result.edges_count}\n"
            f"File size: {size_kb} KB\n"
            f"Export time: {time_sec}s"
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
        elif isinstance(error, ProcessingError):
            # Processing error
            error_msg = f"Export failed: {str(error)}"
        elif isinstance(error, IOError):
            # File write error
            error_msg = f"Failed to write output file: {str(error)}"
        else:
            # Unknown error - generic message for security
            error_msg = "An internal error occurred while exporting the graph."

        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Error: {error_msg}",
                }
            ],
            "isError": True,
        }
