"""
Unit tests for GetStatsTool MCP tool.

Tests cover:
- Tool initialization and validation
- Successful stats retrieval
- Error handling
- Response formatting
- Integration with MemoryProcessor

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

import pytest
from unittest.mock import AsyncMock, Mock

from zapomni_mcp.tools.get_stats import GetStatsTool
from zapomni_core.memory_processor import MemoryProcessor
from zapomni_core.exceptions import DatabaseError


class TestGetStatsToolInit:
    """Test GetStatsTool initialization."""

    def test_init_success_with_valid_processor(self):
        """Test successful initialization with valid MemoryProcessor."""
        # Setup
        mock_processor = Mock(spec=MemoryProcessor)

        # Execute
        tool = GetStatsTool(memory_processor=mock_processor)

        # Verify
        assert tool.memory_processor == mock_processor
        assert tool.name == "get_stats"
        assert tool.description is not None
        assert tool.input_schema is not None

    def test_init_fails_with_invalid_processor_type(self):
        """Test initialization fails with non-MemoryProcessor type."""
        # Execute & Verify
        with pytest.raises(TypeError) as exc_info:
            GetStatsTool(memory_processor="not a processor")

        assert "MemoryProcessor" in str(exc_info.value)

    def test_init_fails_with_none_processor(self):
        """Test initialization fails with None processor."""
        # Execute & Verify
        with pytest.raises(TypeError):
            GetStatsTool(memory_processor=None)


class TestGetStatsToolExecute:
    """Test GetStatsTool.execute() method."""

    @pytest.fixture
    def mock_processor(self):
        """Create a mock MemoryProcessor."""
        processor = Mock(spec=MemoryProcessor)
        processor.get_stats = AsyncMock(
            return_value={
                "total_memories": 100,
                "total_chunks": 500,
                "database_size_mb": 45.67,
                "avg_chunks_per_memory": 5.0,
                "avg_query_latency_ms": 23,
            }
        )
        return processor

    @pytest.fixture
    def tool(self, mock_processor):
        """Create GetStatsTool with mock processor."""
        return GetStatsTool(memory_processor=mock_processor)

    @pytest.mark.asyncio
    async def test_execute_success_with_empty_arguments(self, tool, mock_processor):
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
        assert "Memory System Statistics" in result["content"][0]["text"]
        assert "100" in result["content"][0]["text"]  # total_memories
        mock_processor.get_stats.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_success_with_extra_arguments(self, tool, mock_processor):
        """Test successful execution ignores extra arguments."""
        # Setup
        arguments = {"foo": "bar"}  # Extra arguments should be ignored

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is False
        assert "Memory System Statistics" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_success_with_optional_fields(self, tool, mock_processor):
        """Test successful execution with optional fields."""
        # Setup
        mock_processor.get_stats = AsyncMock(
            return_value={
                "total_memories": 100,
                "total_chunks": 500,
                "database_size_mb": 45.67,
                "avg_chunks_per_memory": 5.0,
                "avg_query_latency_ms": 23,
                "cache_hit_rate": 0.653,
                "total_entities": 250,
                "total_relationships": 500,
            }
        )
        arguments = {}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is False
        text = result["content"][0]["text"]
        assert "65.3%" in text  # cache_hit_rate as percentage
        assert "250" in text  # total_entities
        assert "500" in text  # total_relationships

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
    async def test_execute_processor_database_error(self, tool, mock_processor):
        """Test execution when processor raises DatabaseError."""
        # Setup
        mock_processor.get_stats = AsyncMock(
            side_effect=DatabaseError(
                message="Connection lost", error_code="DB_001"
            )
        )
        arguments = {}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "Error" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_unexpected_error(self, tool, mock_processor):
        """Test execution when processor raises unexpected exception."""
        # Setup
        mock_processor.get_stats = AsyncMock(side_effect=RuntimeError("Boom"))
        arguments = {}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "Error" in result["content"][0]["text"]
        assert "unexpected" in result["content"][0]["text"].lower()


class TestGetStatsToolFormatting:
    """Test GetStatsTool response formatting methods."""

    @pytest.fixture
    def tool(self):
        """Create GetStatsTool with mock processor."""
        mock_processor = Mock(spec=MemoryProcessor)
        return GetStatsTool(memory_processor=mock_processor)

    def test_format_response_required_fields_only(self, tool):
        """Test response formatting with required fields only."""
        # Setup
        stats = {
            "total_memories": 1234,
            "total_chunks": 5678,
            "database_size_mb": 45.67,
            "avg_chunks_per_memory": 4.6,
        }

        # Execute
        result = tool._format_response(stats)

        # Verify
        assert result["isError"] is False
        text = result["content"][0]["text"]
        assert "Memory System Statistics" in text
        assert "1,234" in text  # formatted with comma
        assert "5,678" in text
        assert "45.67 MB" in text
        assert "4.6" in text

    def test_format_response_all_fields(self, tool):
        """Test response formatting with all fields."""
        # Setup
        stats = {
            "total_memories": 1234,
            "total_chunks": 5678,
            "database_size_mb": 45.67,
            "avg_chunks_per_memory": 4.6,
            "total_entities": 999,
            "total_relationships": 2000,
            "cache_hit_rate": 0.653,
            "avg_query_latency_ms": 23,
            "oldest_memory_date": "2025-01-01",
            "newest_memory_date": "2025-11-24",
        }

        # Execute
        result = tool._format_response(stats)

        # Verify
        assert result["isError"] is False
        text = result["content"][0]["text"]
        assert "1,234" in text
        assert "999" in text
        assert "65.3%" in text
        assert "23.0 ms" in text
        assert "2025-01-01" in text

    def test_format_response_handles_missing_optional_fields(self, tool):
        """Test response formatting when optional fields are missing."""
        # Setup
        stats = {
            "total_memories": 100,
            "total_chunks": 500,
            "database_size_mb": 10.0,
            "avg_chunks_per_memory": 5.0,
            # No optional fields
        }

        # Execute
        result = tool._format_response(stats)

        # Verify
        assert result["isError"] is False
        text = result["content"][0]["text"]
        # Should have required fields
        assert "100" in text
        assert "500" in text
        # Should not crash with missing optional fields

    def test_format_response_handles_none_optional_fields(self, tool):
        """Test response formatting when optional fields are None."""
        # Setup
        stats = {
            "total_memories": 100,
            "total_chunks": 500,
            "database_size_mb": 10.0,
            "avg_chunks_per_memory": 5.0,
            "cache_hit_rate": None,
            "total_entities": None,
        }

        # Execute
        result = tool._format_response(stats)

        # Verify
        assert result["isError"] is False
        # Should not include None fields

    def test_format_response_thousand_separators(self, tool):
        """Test that large numbers have thousand separators."""
        # Setup
        stats = {
            "total_memories": 1000000,
            "total_chunks": 5000000,
            "database_size_mb": 1000.0,
            "avg_chunks_per_memory": 5.0,
        }

        # Execute
        result = tool._format_response(stats)

        # Verify
        text = result["content"][0]["text"]
        assert "1,000,000" in text
        assert "5,000,000" in text

    def test_format_response_percentage_formatting(self, tool):
        """Test that cache hit rate is formatted as percentage."""
        # Setup
        stats = {
            "total_memories": 100,
            "total_chunks": 500,
            "database_size_mb": 10.0,
            "avg_chunks_per_memory": 5.0,
            "cache_hit_rate": 0.653,
        }

        # Execute
        result = tool._format_response(stats)

        # Verify
        text = result["content"][0]["text"]
        assert "65.3%" in text

    def test_format_response_decimal_places(self, tool):
        """Test that numbers have correct decimal places."""
        # Setup
        stats = {
            "total_memories": 100,
            "total_chunks": 500,
            "database_size_mb": 45.678,
            "avg_chunks_per_memory": 4.567,
            "avg_query_latency_ms": 23.456,
        }

        # Execute
        result = tool._format_response(stats)

        # Verify
        text = result["content"][0]["text"]
        assert "45.68 MB" in text  # 2 decimal places
        assert "4.6" in text  # 1 decimal place
        assert "23.5 ms" in text  # 1 decimal place

    def test_format_response_handles_missing_required_fields(self, tool):
        """Test response formatting uses defaults for missing required fields."""
        # Setup
        stats = {}  # All fields missing

        # Execute
        result = tool._format_response(stats)

        # Verify
        assert result["isError"] is False
        text = result["content"][0]["text"]
        # Should use defaults and not crash
        assert "0" in text
        assert "unknown" not in text.lower()  # We use 0 for missing numbers


class TestGetStatsToolInputSchema:
    """Test GetStatsTool input schema."""

    def test_input_schema_structure(self):
        """Test that input schema has correct structure."""
        schema = GetStatsTool.input_schema

        # Verify schema structure
        assert schema["type"] == "object"
        assert schema["properties"] == {}  # No properties
        assert schema["required"] == []  # No required fields
        assert schema["additionalProperties"] is False

    def test_tool_metadata(self):
        """Test tool metadata attributes."""
        # Verify metadata
        assert GetStatsTool.name == "get_stats"
        assert "statistics" in GetStatsTool.description.lower()
        assert "memory" in GetStatsTool.description.lower()
