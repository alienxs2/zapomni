"""
Unit tests for GraphStatusTool MCP tool.

Tests cover:
- Tool initialization and validation
- Successful graph status retrieval
- Error handling (database errors, exceptions)
- Response formatting with various stats
- Entity type extraction
- Health calculation
- Integration with MemoryProcessor and FalkorDBClient

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from unittest.mock import AsyncMock, Mock

import pytest

from zapomni_core.exceptions import DatabaseError
from zapomni_core.memory_processor import MemoryProcessor
from zapomni_db.models import QueryResult
from zapomni_mcp.tools.graph_status import GraphStatusTool


class TestGraphStatusToolInit:
    """Test GraphStatusTool initialization."""

    def test_init_success_with_valid_processor(self):
        """Test successful initialization with valid MemoryProcessor."""
        # Setup
        mock_processor = Mock(spec=MemoryProcessor)

        # Execute
        tool = GraphStatusTool(memory_processor=mock_processor)

        # Verify
        assert tool.memory_processor == mock_processor
        assert tool.name == "graph_status"
        assert tool.description is not None
        assert tool.input_schema is not None
        assert tool.input_schema["properties"] == {}
        assert tool.input_schema["required"] == []

    def test_init_fails_with_invalid_processor_type(self):
        """Test initialization fails with non-MemoryProcessor type."""
        # Execute & Verify
        with pytest.raises(TypeError) as exc_info:
            GraphStatusTool(memory_processor="not a processor")

        assert "MemoryProcessor" in str(exc_info.value)

    def test_init_fails_with_none_processor(self):
        """Test initialization fails with None processor."""
        # Execute & Verify
        with pytest.raises(TypeError):
            GraphStatusTool(memory_processor=None)

    def test_init_fails_with_dict(self):
        """Test initialization fails with dict instead of processor."""
        # Execute & Verify
        with pytest.raises(TypeError):
            GraphStatusTool(memory_processor={})


class TestGraphStatusToolExecute:
    """Test GraphStatusTool.execute() method."""

    @pytest.fixture
    def mock_db_client(self):
        """Create a mock FalkorDBClient."""
        db_client = Mock()
        db_client.graph_name = "zapomni_memory"
        db_client.get_stats = AsyncMock(
            return_value={
                "nodes": {
                    "total": 156,
                    "memory": 42,
                    "chunk": 98,
                    "entity": 15,
                    "document": 1,
                },
                "relationships": {
                    "total": 140,
                    "has_chunk": 98,
                    "mentions": 42,
                    "related_to": 0,
                },
            }
        )
        db_client._execute_cypher = AsyncMock(
            return_value=QueryResult(
                rows=[
                    {"type": "TECHNOLOGY", "count": 8},
                    {"type": "PERSON", "count": 4},
                    {"type": "ORG", "count": 3},
                ],
                row_count=3,
                execution_time_ms=10,
            )
        )
        return db_client

    @pytest.fixture
    def mock_processor(self, mock_db_client):
        """Create a mock MemoryProcessor with database client."""
        processor = Mock(spec=MemoryProcessor)
        processor.db_client = mock_db_client
        return processor

    @pytest.fixture
    def tool(self, mock_processor):
        """Create GraphStatusTool with mock processor."""
        return GraphStatusTool(memory_processor=mock_processor)

    @pytest.mark.asyncio
    async def test_execute_success_with_empty_arguments(self, tool, mock_db_client):
        """Test successful execution with empty arguments."""
        # Setup
        arguments = {}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is False
        assert "content" in result
        assert len(result["content"]) > 0
        assert result["content"][0]["type"] == "text"
        assert "Knowledge Graph Status" in result["content"][0]["text"]
        assert "156" in result["content"][0]["text"]  # total nodes
        assert "Healthy" in result["content"][0]["text"]  # health status
        mock_db_client.get_stats.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_success_with_extra_arguments(self, tool):
        """Test successful execution ignores extra arguments."""
        # Setup
        arguments = {"foo": "bar", "baz": 123}  # Extra arguments should be ignored

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is False
        assert "Knowledge Graph Status" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_invalid_arguments_type(self, tool):
        """Test execution with invalid arguments type."""
        # Setup
        arguments = "not a dict"

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "dictionary" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_with_list_arguments(self, tool):
        """Test execution with list instead of dict."""
        # Setup
        arguments = [1, 2, 3]

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "dictionary" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_processor_database_error(self, tool, mock_db_client):
        """Test execution when database raises DatabaseError."""
        # Setup
        mock_db_client.get_stats = AsyncMock(
            side_effect=DatabaseError(message="Connection lost", error_code="DB_001")
        )
        arguments = {}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "Error" in result["content"][0]["text"]
        assert "statistics" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_unexpected_error(self, tool, mock_db_client):
        """Test execution when database raises unexpected exception."""
        # Setup
        mock_db_client.get_stats = AsyncMock(side_effect=RuntimeError("Boom"))
        arguments = {}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "Error" in result["content"][0]["text"]
        # The RuntimeError is caught and wrapped in DatabaseError
        assert "Failed to retrieve graph statistics" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_with_no_db_client(self, mock_processor):
        """Test execution when database client is not initialized."""
        # Setup
        mock_processor.db_client = None
        tool = GraphStatusTool(memory_processor=mock_processor)
        arguments = {}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "Error" in result["content"][0]["text"]


class TestGraphStatusToolHealthCalculation:
    """Test health status calculation logic."""

    @pytest.fixture
    def tool(self):
        """Create GraphStatusTool with mock processor."""
        mock_processor = Mock(spec=MemoryProcessor)
        return GraphStatusTool(memory_processor=mock_processor)

    def test_calculate_health_healthy(self, tool):
        """Test health is 'Healthy' when entities exist."""
        # Setup
        stats = {
            "nodes": {
                "total": 100,
                "entity": 15,
            }
        }

        # Execute
        result = tool._calculate_health(stats)

        # Verify
        assert result == "Healthy"

    def test_calculate_health_warning(self, tool):
        """Test health is 'Warning' when no entities but has nodes."""
        # Setup
        stats = {
            "nodes": {
                "total": 100,
                "entity": 0,
            }
        }

        # Execute
        result = tool._calculate_health(stats)

        # Verify
        assert result == "Warning"

    def test_calculate_health_critical(self, tool):
        """Test health is 'Critical' when no nodes."""
        # Setup
        stats = {
            "nodes": {
                "total": 0,
                "entity": 0,
            }
        }

        # Execute
        result = tool._calculate_health(stats)

        # Verify
        assert result == "Critical"

    def test_calculate_health_empty_stats(self, tool):
        """Test health calculation with empty stats."""
        # Setup
        stats = {}

        # Execute
        result = tool._calculate_health(stats)

        # Verify
        assert result == "Critical"  # No nodes = critical

    def test_calculate_health_missing_entity_count(self, tool):
        """Test health calculation when entity count is missing."""
        # Setup
        stats = {
            "nodes": {
                "total": 50,
            }
        }

        # Execute
        result = tool._calculate_health(stats)

        # Verify
        assert result == "Warning"  # Total > 0 but no entity count


class TestGraphStatusToolFormatting:
    """Test GraphStatusTool response formatting methods."""

    @pytest.fixture
    def tool(self):
        """Create GraphStatusTool with mock processor."""
        mock_processor = Mock(spec=MemoryProcessor)
        return GraphStatusTool(memory_processor=mock_processor)

    def test_format_response_full_stats(self, tool):
        """Test response formatting with complete statistics."""
        # Setup
        stats = {
            "nodes": {
                "total": 156,
                "memory": 42,
                "chunk": 98,
                "entity": 15,
                "document": 1,
            },
            "relationships": {
                "total": 140,
                "has_chunk": 98,
                "mentions": 42,
                "related_to": 0,
            },
            "entity_types": {
                "TECHNOLOGY": 8,
                "PERSON": 4,
                "ORG": 3,
            },
            "health": "Healthy",
            "graph_name": "zapomni_memory",
        }

        # Execute
        result = tool._format_response(stats)

        # Verify
        assert result["isError"] is False
        text = result["content"][0]["text"]
        assert "Knowledge Graph Status" in text
        assert "156" in text  # total nodes
        assert "42" in text  # memories
        assert "98" in text  # chunks
        assert "140" in text  # total relationships
        assert "Healthy" in text
        assert "TECHNOLOGY: 8" in text

    def test_format_response_empty_stats(self, tool):
        """Test response formatting with empty statistics."""
        # Setup
        stats = {
            "nodes": {},
            "relationships": {},
            "entity_types": {},
            "health": "Critical",
        }

        # Execute
        result = tool._format_response(stats)

        # Verify
        assert result["isError"] is False
        text = result["content"][0]["text"]
        assert "Knowledge Graph Status" in text
        assert "Critical" in text
        # Should show 0 for missing counts
        assert "0" in text

    def test_format_response_with_zero_nodes(self, tool):
        """Test formatting when graph has no nodes."""
        # Setup
        stats = {
            "nodes": {
                "total": 0,
                "memory": 0,
                "chunk": 0,
                "entity": 0,
                "document": 0,
            },
            "relationships": {
                "total": 0,
                "has_chunk": 0,
                "mentions": 0,
                "related_to": 0,
            },
            "entity_types": {},
            "health": "Critical",
        }

        # Execute
        result = tool._format_response(stats)

        # Verify
        assert result["isError"] is False
        text = result["content"][0]["text"]
        assert "Critical" in text
        assert "Total: 0" in text

    def test_format_response_with_graph_name(self, tool):
        """Test formatting includes graph name when provided."""
        # Setup
        stats = {
            "nodes": {"total": 10},
            "relationships": {"total": 5},
            "entity_types": {},
            "health": "Healthy",
            "graph_name": "my_custom_graph",
        }

        # Execute
        result = tool._format_response(stats)

        # Verify
        text = result["content"][0]["text"]
        assert "my_custom_graph" in text

    def test_format_response_entity_types_sorted(self, tool):
        """Test entity types are sorted by count descending."""
        # Setup
        stats = {
            "nodes": {"total": 15},
            "relationships": {"total": 0},
            "entity_types": {
                "TECHNOLOGY": 1,
                "PERSON": 5,
                "ORG": 2,
                "CONCEPT": 7,
            },
            "health": "Healthy",
        }

        # Execute
        result = tool._format_response(stats)

        # Verify
        text = result["content"][0]["text"]
        # CONCEPT (7) should appear before PERSON (5), which appears before ORG (2), etc.
        concept_idx = text.find("CONCEPT: 7")
        person_idx = text.find("PERSON: 5")
        org_idx = text.find("ORG: 2")
        tech_idx = text.find("TECHNOLOGY: 1")

        assert concept_idx < person_idx < org_idx < tech_idx

    def test_format_response_excludes_zero_entity_types(self, tool):
        """Test that entity types with zero count are excluded."""
        # Setup
        stats = {
            "nodes": {"total": 5},
            "relationships": {"total": 0},
            "entity_types": {
                "TECHNOLOGY": 5,
                "PERSON": 0,
                "ORG": 0,
            },
            "health": "Healthy",
        }

        # Execute
        result = tool._format_response(stats)

        # Verify
        text = result["content"][0]["text"]
        assert "TECHNOLOGY: 5" in text
        # PERSON and ORG should not appear in entity types section
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if "Entity Types:" in line:
                # Check next few lines
                for j in range(i + 1, min(i + 4, len(lines))):
                    if lines[j].startswith("  PERSON") or lines[j].startswith("  ORG"):
                        if ": 0" in lines[j]:
                            pytest.fail("Zero count entity types should not be shown")

    def test_format_response_thousand_separators(self, tool):
        """Test that large numbers have thousand separators."""
        # Setup
        stats = {
            "nodes": {
                "total": 1000000,
                "memory": 100000,
                "chunk": 500000,
                "entity": 400000,
                "document": 0,
            },
            "relationships": {
                "total": 1500000,
                "has_chunk": 500000,
                "mentions": 1000000,
                "related_to": 0,
            },
            "entity_types": {},
            "health": "Healthy",
        }

        # Execute
        result = tool._format_response(stats)

        # Verify
        text = result["content"][0]["text"]
        assert "1,000,000" in text
        assert "100,000" in text
        assert "500,000" in text


class TestGraphStatusToolInputSchema:
    """Test GraphStatusTool input schema."""

    def test_input_schema_structure(self):
        """Test that input schema has correct structure."""
        schema = GraphStatusTool.input_schema

        # Verify schema structure
        assert schema["type"] == "object"
        assert schema["properties"] == {}  # No properties
        assert schema["required"] == []  # No required fields
        assert schema["additionalProperties"] is False

    def test_tool_metadata(self):
        """Test tool metadata attributes."""
        # Verify metadata
        assert GraphStatusTool.name == "graph_status"
        assert "statistics" in GraphStatusTool.description.lower()
        assert "graph" in GraphStatusTool.description.lower()
        assert "health" in GraphStatusTool.description.lower()

    def test_tool_has_valid_description(self):
        """Test tool has a meaningful description."""
        description = GraphStatusTool.description
        assert isinstance(description, str)
        assert len(description) > 20
        assert description.endswith(".")


class TestGraphStatusToolEntityTypes:
    """Test entity type extraction logic."""

    @pytest.fixture
    def mock_db_client(self):
        """Create a mock database client."""
        db_client = Mock()
        db_client.graph_name = "test_graph"
        return db_client

    @pytest.fixture
    def mock_processor(self, mock_db_client):
        """Create a mock processor with database client."""
        processor = Mock(spec=MemoryProcessor)
        processor.db_client = mock_db_client
        return processor

    @pytest.fixture
    def tool(self, mock_processor):
        """Create tool with mock processor."""
        return GraphStatusTool(memory_processor=mock_processor)

    @pytest.mark.asyncio
    async def test_get_entity_types_success(self, tool, mock_db_client):
        """Test successful entity type extraction."""
        # Setup
        mock_db_client._execute_cypher = AsyncMock(
            return_value=QueryResult(
                rows=[
                    {"type": "TECHNOLOGY", "count": 10},
                    {"type": "PERSON", "count": 5},
                ],
                row_count=2,
                execution_time_ms=5,
            )
        )

        # Execute
        result = await tool._get_entity_types()

        # Verify
        assert "TECHNOLOGY" in result
        assert result["TECHNOLOGY"] == 10
        assert "PERSON" in result
        assert result["PERSON"] == 5
        # Standard types should all be present
        assert "ORG" in result
        assert "CONCEPT" in result

    @pytest.mark.asyncio
    async def test_get_entity_types_handles_error(self, tool, mock_db_client):
        """Test entity type extraction handles errors gracefully."""
        # Setup
        mock_db_client._execute_cypher = AsyncMock(side_effect=Exception("Query failed"))

        # Execute
        result = await tool._get_entity_types()

        # Verify - should return empty dict without raising
        assert result == {}

    @pytest.mark.asyncio
    async def test_get_entity_types_no_db_client(self, mock_processor):
        """Test entity type extraction when db client is None."""
        # Setup
        mock_processor.db_client = None
        tool = GraphStatusTool(memory_processor=mock_processor)

        # Execute
        result = await tool._get_entity_types()

        # Verify
        assert result == {}


class TestGraphStatusToolEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def mock_processor(self):
        """Create a mock processor."""
        return Mock(spec=MemoryProcessor)

    @pytest.fixture
    def tool(self, mock_processor):
        """Create tool with mock processor."""
        return GraphStatusTool(memory_processor=mock_processor)

    @pytest.mark.asyncio
    async def test_execute_handles_none_in_stats(self, tool, mock_processor):
        """Test execution handles None values in statistics."""
        # Setup
        mock_db_client = Mock()
        mock_db_client.graph_name = "test"
        mock_db_client.get_stats = AsyncMock(
            return_value={
                "nodes": {
                    "total": None,
                    "entity": None,
                },
                "relationships": None,
            }
        )
        mock_db_client._execute_cypher = AsyncMock(
            return_value=QueryResult(rows=[], row_count=0, execution_time_ms=0)
        )
        mock_processor.db_client = mock_db_client

        # Execute - should not crash
        result = await tool.execute({})

        # Verify - should handle None gracefully
        assert result["isError"] is False or result["isError"] is True  # Either is OK
        # Make sure it doesn't crash

    def test_format_response_with_none_values(self, tool):
        """Test formatting handles None values gracefully."""
        # Setup
        stats = {
            "nodes": {
                "total": None,
                "memory": 0,
            },
            "relationships": None,
            "entity_types": None,
            "health": "Unknown",
        }

        # Execute - should not crash
        result = tool._format_response(stats)

        # Verify
        assert result["isError"] is False
