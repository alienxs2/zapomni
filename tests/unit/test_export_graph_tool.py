"""
Unit tests for ExportGraphTool MCP tool.

Tests cover:
- Tool initialization and validation
- Successful graph export for all formats
- Input validation and error handling
- Error response formatting
- Integration with GraphExporter

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import ValidationError

from zapomni_core.exceptions import (
    DatabaseError,
    ProcessingError,
)
from zapomni_core.graph.graph_exporter import ExportResult
from zapomni_core.memory_processor import MemoryProcessor
from zapomni_db import FalkorDBClient
from zapomni_mcp.tools.export_graph import (
    ExportFormat,
    ExportGraphRequest,
    ExportGraphTool,
    ExportOptions,
)


class TestExportGraphToolInit:
    """Test ExportGraphTool initialization."""

    @patch("zapomni_mcp.tools.export_graph.GraphExporter")
    def test_init_success_with_valid_processor(self, mock_graph_exporter):
        """Test successful initialization with valid MemoryProcessor."""
        # Setup
        mock_processor = Mock(spec=MemoryProcessor)
        mock_processor.db_client = Mock(spec=FalkorDBClient)
        mock_exporter_instance = Mock()
        mock_graph_exporter.return_value = mock_exporter_instance

        # Execute
        tool = ExportGraphTool(memory_processor=mock_processor)

        # Verify
        assert tool.memory_processor == mock_processor
        assert tool.name == "export_graph"
        assert tool.description is not None
        assert tool.input_schema is not None
        assert tool._exporter == mock_exporter_instance
        mock_graph_exporter.assert_called_once_with(db_client=mock_processor.db_client)

    def test_init_fails_with_invalid_processor_type(self):
        """Test initialization fails with non-MemoryProcessor type."""
        # Execute & Verify
        with pytest.raises(TypeError) as exc_info:
            ExportGraphTool(memory_processor="not a processor")

        assert "MemoryProcessor" in str(exc_info.value)

    def test_init_fails_with_none_processor(self):
        """Test initialization fails with None processor."""
        # Execute & Verify
        with pytest.raises(TypeError):
            ExportGraphTool(memory_processor=None)


class TestExportGraphToolExecute:
    """Test ExportGraphTool.execute() method."""

    @pytest.fixture
    def mock_processor(self):
        """Create a mock MemoryProcessor."""
        processor = Mock(spec=MemoryProcessor)
        processor.db_client = Mock(spec=FalkorDBClient)
        return processor

    @pytest.fixture
    @patch("zapomni_mcp.tools.export_graph.GraphExporter")
    def tool(self, mock_graph_exporter, mock_processor):
        """Create ExportGraphTool with mock processor."""
        mock_exporter_instance = Mock()
        mock_graph_exporter.return_value = mock_exporter_instance
        tool_instance = ExportGraphTool(memory_processor=mock_processor)
        return tool_instance

    @pytest.fixture
    def mock_export_result(self):
        """Create a mock ExportResult."""
        return ExportResult(
            format="graphml",
            output_path="/tmp/test.graphml",
            nodes_count=100,
            edges_count=150,
            file_size_bytes=12345,
            export_time_ms=123.45,
        )

    @pytest.mark.asyncio
    async def test_execute_success_graphml(self, tool, mock_export_result):
        """Test successful execution with GraphML format."""
        # Setup
        tool._exporter.export_graphml = AsyncMock(return_value=mock_export_result)
        arguments = {"format": "graphml", "output_path": "/tmp/test.graphml"}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is False
        assert "content" in result
        assert len(result["content"]) > 0
        assert result["content"][0]["type"] == "text"
        assert "100" in result["content"][0]["text"]  # nodes count
        assert "150" in result["content"][0]["text"]  # edges count
        tool._exporter.export_graphml.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_success_cytoscape(self, tool, mock_export_result):
        """Test successful execution with Cytoscape format."""
        # Setup
        mock_export_result.format = "cytoscape"
        mock_export_result.output_path = "/tmp/test.json"
        tool._exporter.export_cytoscape = AsyncMock(return_value=mock_export_result)
        arguments = {"format": "cytoscape", "output_path": "/tmp/test.json"}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is False
        assert "cytoscape" in result["content"][0]["text"]
        tool._exporter.export_cytoscape.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_success_neo4j(self, tool, mock_export_result):
        """Test successful execution with Neo4j format."""
        # Setup
        mock_export_result.format = "neo4j"
        mock_export_result.output_path = "/tmp/test.cypher"
        tool._exporter.export_neo4j = AsyncMock(return_value=mock_export_result)
        arguments = {"format": "neo4j", "output_path": "/tmp/test.cypher"}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is False
        assert "neo4j" in result["content"][0]["text"]
        tool._exporter.export_neo4j.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_success_json(self, tool, mock_export_result):
        """Test successful execution with JSON format."""
        # Setup
        mock_export_result.format = "json"
        mock_export_result.output_path = "/tmp/test.json"
        tool._exporter.export_json = AsyncMock(return_value=mock_export_result)
        arguments = {"format": "json", "output_path": "/tmp/test.json"}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is False
        assert "json" in result["content"][0]["text"]
        tool._exporter.export_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_success_with_options(self, tool, mock_export_result):
        """Test successful execution with export options."""
        # Setup
        tool._exporter.export_graphml = AsyncMock(return_value=mock_export_result)
        arguments = {
            "format": "graphml",
            "output_path": "/tmp/test.graphml",
            "options": {
                "pretty_print": True,
                "include_metadata": True,
                "node_types": ["Entity", "Memory"],
            },
        }

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is False
        # Verify options were passed to exporter
        call_args = tool._exporter.export_graphml.call_args
        assert call_args[0][0] == "/tmp/test.graphml"
        assert call_args[0][1]["pretty_print"] is True
        assert call_args[0][1]["include_metadata"] is True
        assert call_args[0][1]["node_types"] == ["Entity", "Memory"]

    @pytest.mark.asyncio
    async def test_execute_invalid_format(self, tool):
        """Test execution with invalid format."""
        # Setup
        arguments = {"format": "invalid", "output_path": "/tmp/test.txt"}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "Error" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_empty_output_path(self, tool):
        """Test execution with empty output path."""
        # Setup
        arguments = {"format": "graphml", "output_path": ""}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "Error" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_missing_format(self, tool):
        """Test execution with missing format parameter."""
        # Setup
        arguments = {"output_path": "/tmp/test.graphml"}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "Error" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_missing_output_path(self, tool):
        """Test execution with missing output_path parameter."""
        # Setup
        arguments = {"format": "graphml"}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "Error" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_invalid_options_type(self, tool):
        """Test execution with invalid options type."""
        # Setup
        arguments = {
            "format": "graphml",
            "output_path": "/tmp/test.graphml",
            "options": "invalid",
        }

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "Error" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_invalid_node_types(self, tool):
        """Test execution with invalid node_types in options."""
        # Setup
        arguments = {
            "format": "graphml",
            "output_path": "/tmp/test.graphml",
            "options": {"node_types": "invalid"},
        }

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "Error" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_database_error(self, tool):
        """Test execution when exporter raises DatabaseError."""
        # Setup
        tool._exporter.export_graphml = AsyncMock(
            side_effect=DatabaseError(message="Connection lost", error_code="DB_001")
        )
        arguments = {"format": "graphml", "output_path": "/tmp/test.graphml"}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "Error" in result["content"][0]["text"]
        assert "temporarily unavailable" in result["content"][0]["text"].lower()

    @pytest.mark.asyncio
    async def test_execute_processing_error(self, tool):
        """Test execution when exporter raises ProcessingError."""
        # Setup
        tool._exporter.export_graphml = AsyncMock(
            side_effect=ProcessingError(message="Export failed", error_code="PROC_001")
        )
        arguments = {"format": "graphml", "output_path": "/tmp/test.graphml"}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "Error" in result["content"][0]["text"]
        assert "export failed" in result["content"][0]["text"].lower()

    @pytest.mark.asyncio
    async def test_execute_io_error(self, tool):
        """Test execution when exporter raises IOError."""
        # Setup
        tool._exporter.export_graphml = AsyncMock(side_effect=IOError("Permission denied"))
        arguments = {"format": "graphml", "output_path": "/tmp/test.graphml"}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "Error" in result["content"][0]["text"]
        assert "write" in result["content"][0]["text"].lower()

    @pytest.mark.asyncio
    async def test_execute_unexpected_error(self, tool):
        """Test execution when exporter raises unexpected exception."""
        # Setup
        tool._exporter.export_graphml = AsyncMock(side_effect=RuntimeError("Boom"))
        arguments = {"format": "graphml", "output_path": "/tmp/test.graphml"}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "Error" in result["content"][0]["text"]
        assert "internal error" in result["content"][0]["text"].lower()


class TestExportGraphToolValidation:
    """Test ExportGraphTool._validate_arguments() method."""

    @pytest.fixture
    @patch("zapomni_mcp.tools.export_graph.GraphExporter")
    def tool(self, mock_graph_exporter):
        """Create ExportGraphTool with mock processor."""
        mock_processor = Mock(spec=MemoryProcessor)
        mock_processor.db_client = Mock(spec=FalkorDBClient)
        mock_exporter_instance = Mock()
        mock_graph_exporter.return_value = mock_exporter_instance
        return ExportGraphTool(memory_processor=mock_processor)

    def test_validate_arguments_success_minimal(self, tool):
        """Test successful validation with minimal input."""
        # Setup
        arguments = {"format": "graphml", "output_path": "/tmp/test.graphml"}

        # Execute
        format_type, output_path, options = tool._validate_arguments(arguments)

        # Verify
        assert format_type == ExportFormat.GRAPHML
        assert output_path == "/tmp/test.graphml"
        # Options dict contains default values from ExportOptions model
        assert "pretty_print" in options
        assert "include_metadata" in options
        assert "batch_size" in options
        assert options["pretty_print"] is True
        assert options["include_metadata"] is True

    def test_validate_arguments_success_with_all_options(self, tool):
        """Test successful validation with all options."""
        # Setup
        arguments = {
            "format": "graphml",
            "output_path": "/tmp/test.graphml",
            "options": {
                "node_types": ["Entity", "Memory"],
                "pretty_print": True,
                "include_metadata": True,
                "batch_size": 500,
            },
        }

        # Execute
        format_type, output_path, options = tool._validate_arguments(arguments)

        # Verify
        assert format_type == ExportFormat.GRAPHML
        assert output_path == "/tmp/test.graphml"
        assert options["node_types"] == ["Entity", "Memory"]
        assert options["pretty_print"] is True
        assert options["include_metadata"] is True
        assert options["batch_size"] == 500

    def test_validate_arguments_strips_whitespace(self, tool):
        """Test that validation strips whitespace from output_path."""
        # Setup
        arguments = {"format": "graphml", "output_path": "  /tmp/test.graphml  "}

        # Execute
        format_type, output_path, options = tool._validate_arguments(arguments)

        # Verify
        assert output_path == "/tmp/test.graphml"

    def test_validate_arguments_empty_path_raises_error(self, tool):
        """Test that empty output_path raises ValidationError."""
        # Setup
        arguments = {"format": "graphml", "output_path": ""}

        # Execute & Verify
        with pytest.raises(ValidationError):
            tool._validate_arguments(arguments)

    def test_validate_arguments_whitespace_only_path_raises_error(self, tool):
        """Test that whitespace-only output_path raises ValidationError."""
        # Setup
        arguments = {"format": "graphml", "output_path": "   "}

        # Execute & Verify
        with pytest.raises(ValidationError):
            tool._validate_arguments(arguments)

    def test_validate_arguments_invalid_format_raises_error(self, tool):
        """Test that invalid format raises ValidationError."""
        # Setup
        arguments = {"format": "invalid", "output_path": "/tmp/test.txt"}

        # Execute & Verify
        with pytest.raises(ValidationError):
            tool._validate_arguments(arguments)

    def test_validate_arguments_missing_format_raises_error(self, tool):
        """Test that missing format raises ValidationError."""
        # Setup
        arguments = {"output_path": "/tmp/test.graphml"}

        # Execute & Verify
        with pytest.raises(ValidationError):
            tool._validate_arguments(arguments)

    def test_validate_arguments_missing_path_raises_error(self, tool):
        """Test that missing output_path raises ValidationError."""
        # Setup
        arguments = {"format": "graphml"}

        # Execute & Verify
        with pytest.raises(ValidationError):
            tool._validate_arguments(arguments)


class TestExportGraphToolExport:
    """Test ExportGraphTool._export_graph() method."""

    @pytest.fixture
    @patch("zapomni_mcp.tools.export_graph.GraphExporter")
    def tool(self, mock_graph_exporter):
        """Create ExportGraphTool with mock processor."""
        mock_processor = Mock(spec=MemoryProcessor)
        mock_processor.db_client = Mock(spec=FalkorDBClient)
        mock_exporter_instance = Mock()
        mock_graph_exporter.return_value = mock_exporter_instance
        return ExportGraphTool(memory_processor=mock_processor)

    @pytest.fixture
    def mock_export_result(self):
        """Create a mock ExportResult."""
        return ExportResult(
            format="graphml",
            output_path="/tmp/test.graphml",
            nodes_count=100,
            edges_count=150,
            file_size_bytes=12345,
            export_time_ms=123.45,
        )

    @pytest.mark.asyncio
    async def test_export_graph_routes_graphml(self, tool, mock_export_result):
        """Test that _export_graph routes to export_graphml for GRAPHML format."""
        # Setup
        tool._exporter.export_graphml = AsyncMock(return_value=mock_export_result)

        # Execute
        result = await tool._export_graph(ExportFormat.GRAPHML, "/tmp/test.graphml", {})

        # Verify
        assert result == mock_export_result
        tool._exporter.export_graphml.assert_called_once_with("/tmp/test.graphml", {})

    @pytest.mark.asyncio
    async def test_export_graph_routes_cytoscape(self, tool, mock_export_result):
        """Test that _export_graph routes to export_cytoscape for CYTOSCAPE format."""
        # Setup
        tool._exporter.export_cytoscape = AsyncMock(return_value=mock_export_result)

        # Execute
        result = await tool._export_graph(ExportFormat.CYTOSCAPE, "/tmp/test.json", {})

        # Verify
        assert result == mock_export_result
        tool._exporter.export_cytoscape.assert_called_once_with("/tmp/test.json", {})

    @pytest.mark.asyncio
    async def test_export_graph_routes_neo4j(self, tool, mock_export_result):
        """Test that _export_graph routes to export_neo4j for NEO4J format."""
        # Setup
        tool._exporter.export_neo4j = AsyncMock(return_value=mock_export_result)

        # Execute
        result = await tool._export_graph(ExportFormat.NEO4J, "/tmp/test.cypher", {})

        # Verify
        assert result == mock_export_result
        tool._exporter.export_neo4j.assert_called_once_with("/tmp/test.cypher", {})

    @pytest.mark.asyncio
    async def test_export_graph_routes_json(self, tool, mock_export_result):
        """Test that _export_graph routes to export_json for JSON format."""
        # Setup
        tool._exporter.export_json = AsyncMock(return_value=mock_export_result)

        # Execute
        result = await tool._export_graph(ExportFormat.JSON, "/tmp/test.json", {})

        # Verify
        assert result == mock_export_result
        tool._exporter.export_json.assert_called_once_with("/tmp/test.json", {})

    @pytest.mark.asyncio
    async def test_export_graph_passes_options(self, tool, mock_export_result):
        """Test that _export_graph passes options to exporter."""
        # Setup
        tool._exporter.export_graphml = AsyncMock(return_value=mock_export_result)
        options = {"pretty_print": True, "node_types": ["Entity"]}

        # Execute
        await tool._export_graph(ExportFormat.GRAPHML, "/tmp/test.graphml", options)

        # Verify
        call_args = tool._exporter.export_graphml.call_args
        assert call_args[0][1] == options


class TestExportGraphToolFormatting:
    """Test ExportGraphTool response formatting methods."""

    @pytest.fixture
    @patch("zapomni_mcp.tools.export_graph.GraphExporter")
    def tool(self, mock_graph_exporter):
        """Create ExportGraphTool with mock processor."""
        mock_processor = Mock(spec=MemoryProcessor)
        mock_processor.db_client = Mock(spec=FalkorDBClient)
        mock_exporter_instance = Mock()
        mock_graph_exporter.return_value = mock_exporter_instance
        return ExportGraphTool(memory_processor=mock_processor)

    @pytest.fixture
    def mock_export_result(self):
        """Create a mock ExportResult."""
        return ExportResult(
            format="graphml",
            output_path="/tmp/test.graphml",
            nodes_count=100,
            edges_count=150,
            file_size_bytes=12345,
            export_time_ms=123.45,
        )

    def test_format_success(self, tool, mock_export_result):
        """Test success response formatting."""
        # Execute
        result = tool._format_success(mock_export_result)

        # Verify
        assert result["isError"] is False
        assert "content" in result
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "text"
        assert "graphml" in result["content"][0]["text"]
        assert "/tmp/test.graphml" in result["content"][0]["text"]
        assert "100" in result["content"][0]["text"]  # nodes
        assert "150" in result["content"][0]["text"]  # edges

    def test_format_success_includes_statistics(self, tool, mock_export_result):
        """Test that success response includes all statistics."""
        # Execute
        result = tool._format_success(mock_export_result)

        # Verify
        text = result["content"][0]["text"]
        assert "Nodes:" in text
        assert "Edges:" in text
        assert "File size:" in text
        assert "Export time:" in text

    def test_format_error_validation_error(self, tool):
        """Test error formatting for ValidationError."""
        # Setup - create a simple validation error by calling the Pydantic model incorrectly
        try:
            ExportGraphRequest(format="invalid", output_path="/test")
        except ValidationError as error:
            # Execute
            result = tool._format_error(error)

            # Verify
            assert result["isError"] is True
            assert "Error" in result["content"][0]["text"]

    def test_format_error_database_error(self, tool):
        """Test error formatting for DatabaseError."""
        # Setup
        error = DatabaseError(message="Connection lost", error_code="DB_001")

        # Execute
        result = tool._format_error(error)

        # Verify
        assert result["isError"] is True
        assert "temporarily unavailable" in result["content"][0]["text"].lower()

    def test_format_error_processing_error(self, tool):
        """Test error formatting for ProcessingError."""
        # Setup
        error = ProcessingError(message="Export failed", error_code="PROC_001")

        # Execute
        result = tool._format_error(error)

        # Verify
        assert result["isError"] is True
        assert "export failed" in result["content"][0]["text"].lower()

    def test_format_error_io_error(self, tool):
        """Test error formatting for IOError."""
        # Setup
        error = IOError("Permission denied")

        # Execute
        result = tool._format_error(error)

        # Verify
        assert result["isError"] is True
        assert "write" in result["content"][0]["text"].lower()

    def test_format_error_unexpected_error(self, tool):
        """Test error formatting for unexpected error."""
        # Setup
        error = RuntimeError("Boom")

        # Execute
        result = tool._format_error(error)

        # Verify
        assert result["isError"] is True
        assert "internal error" in result["content"][0]["text"].lower()


class TestExportGraphToolInputSchema:
    """Test ExportGraphTool input schema."""

    def test_input_schema_structure(self):
        """Test that input schema has correct structure."""
        schema = ExportGraphTool.input_schema

        # Verify schema structure
        assert schema["type"] == "object"
        assert "format" in schema["properties"]
        assert "output_path" in schema["properties"]
        assert "options" in schema["properties"]
        assert "format" in schema["required"]
        assert "output_path" in schema["required"]
        assert "options" not in schema["required"]

    def test_input_schema_format_enum(self):
        """Test that format field has correct enum values."""
        schema = ExportGraphTool.input_schema
        format_schema = schema["properties"]["format"]

        # Verify enum
        assert format_schema["enum"] == ["graphml", "cytoscape", "neo4j", "json"]

    def test_input_schema_output_path_constraints(self):
        """Test that output_path field has correct constraints."""
        schema = ExportGraphTool.input_schema
        path_schema = schema["properties"]["output_path"]

        # Verify constraints
        assert path_schema["minLength"] == 1
        assert path_schema["type"] == "string"

    def test_input_schema_options_structure(self):
        """Test that options field has correct structure."""
        schema = ExportGraphTool.input_schema
        options_schema = schema["properties"]["options"]

        # Verify options structure
        assert options_schema["type"] == "object"
        assert "node_types" in options_schema["properties"]
        assert "pretty_print" in options_schema["properties"]
        assert "include_metadata" in options_schema["properties"]
        assert "batch_size" in options_schema["properties"]


class TestExportOptionsModel:
    """Test ExportOptions Pydantic model."""

    def test_export_options_defaults(self):
        """Test ExportOptions default values."""
        # Execute
        options = ExportOptions()

        # Verify
        assert options.node_types is None
        assert options.pretty_print is True
        assert options.include_metadata is True
        assert options.include_style is True
        assert options.batch_size == 1000

    def test_export_options_valid_node_types(self):
        """Test ExportOptions with valid node_types."""
        # Execute
        options = ExportOptions(node_types=["Entity", "Memory"])

        # Verify
        assert options.node_types == ["Entity", "Memory"]

    def test_export_options_empty_node_types_list(self):
        """Test ExportOptions with empty node_types list."""
        # Execute
        options = ExportOptions(node_types=[])

        # Verify
        assert options.node_types == []

    def test_export_options_batch_size_validation(self):
        """Test ExportOptions batch_size validation."""
        # Valid batch size
        options = ExportOptions(batch_size=500)
        assert options.batch_size == 500

        # Too small
        with pytest.raises(ValidationError):
            ExportOptions(batch_size=0)

        # Too large
        with pytest.raises(ValidationError):
            ExportOptions(batch_size=20000)
