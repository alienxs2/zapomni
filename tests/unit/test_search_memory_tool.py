"""
Unit tests for SearchMemoryTool MCP tool.

Tests cover:
- Tool initialization and validation
- Successful search operations
- Input validation and error handling
- Result formatting
- Integration with MemoryProcessor

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

import pytest
from unittest.mock import AsyncMock, Mock
from pydantic import ValidationError
from datetime import datetime, timezone

from zapomni_mcp.tools.search_memory import SearchMemoryTool, SearchMemoryRequest
from zapomni_core.memory_processor import MemoryProcessor, SearchResultItem
from zapomni_core.exceptions import (
    ValidationError as CoreValidationError,
    SearchError,
    EmbeddingError,
    DatabaseError,
)


class TestSearchMemoryToolInit:
    """Test SearchMemoryTool initialization."""

    def test_init_success_with_valid_processor(self):
        """Test successful initialization with valid MemoryProcessor."""
        # Setup
        mock_processor = Mock(spec=MemoryProcessor)

        # Execute
        tool = SearchMemoryTool(memory_processor=mock_processor)

        # Verify
        assert tool.memory_processor == mock_processor
        assert tool.name == "search_memory"
        assert tool.description is not None
        assert tool.input_schema is not None

    def test_init_fails_with_invalid_processor_type(self):
        """Test initialization fails with non-MemoryProcessor type."""
        # Execute & Verify
        with pytest.raises(TypeError) as exc_info:
            SearchMemoryTool(memory_processor="not a processor")

        assert "MemoryProcessor" in str(exc_info.value)

    def test_init_fails_with_none_processor(self):
        """Test initialization fails with None processor."""
        # Execute & Verify
        with pytest.raises(TypeError):
            SearchMemoryTool(memory_processor=None)


class TestSearchMemoryToolExecute:
    """Test SearchMemoryTool.execute() method."""

    @pytest.fixture
    def mock_processor(self):
        """Create a mock MemoryProcessor."""
        processor = Mock(spec=MemoryProcessor)
        processor.search_memory = AsyncMock(return_value=[])
        return processor

    @pytest.fixture
    def tool(self, mock_processor):
        """Create SearchMemoryTool with mock processor."""
        return SearchMemoryTool(memory_processor=mock_processor)

    @pytest.fixture
    def sample_results(self):
        """Create sample search results."""
        return [
            SearchResultItem(
                memory_id="id-1",
                text="Python is a programming language",
                similarity_score=0.92,
                tags=["python", "programming"],
                source="documentation",
                timestamp=datetime.now(timezone.utc),
            ),
            SearchResultItem(
                memory_id="id-2",
                text="Python was created by Guido van Rossum",
                similarity_score=0.85,
                tags=["python", "history"],
                source="wikipedia",
                timestamp=datetime.now(timezone.utc),
            ),
        ]

    @pytest.mark.asyncio
    async def test_execute_success_minimal(self, tool, mock_processor, sample_results):
        """Test successful execution with minimal input."""
        # Setup
        mock_processor.search_memory = AsyncMock(return_value=sample_results)
        arguments = {"query": "What is Python?"}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is False
        assert "content" in result
        assert "Found 2 results" in result["content"][0]["text"]
        mock_processor.search_memory.assert_called_once()
        call_args = mock_processor.search_memory.call_args
        assert call_args[1]["query"] == "What is Python?"
        assert call_args[1]["limit"] == 10  # default

    @pytest.mark.asyncio
    async def test_execute_success_with_limit(self, tool, mock_processor, sample_results):
        """Test successful execution with custom limit."""
        # Setup
        mock_processor.search_memory = AsyncMock(return_value=sample_results[:1])
        arguments = {"query": "Python", "limit": 5}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is False
        call_args = mock_processor.search_memory.call_args
        assert call_args[1]["limit"] == 5

    @pytest.mark.asyncio
    async def test_execute_success_with_filters(self, tool, mock_processor, sample_results):
        """Test successful execution with filters."""
        # Setup
        mock_processor.search_memory = AsyncMock(return_value=sample_results)
        arguments = {
            "query": "Python",
            "filters": {"tags": ["python"], "source": "documentation"},
        }

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is False
        call_args = mock_processor.search_memory.call_args
        assert call_args[1]["filters"] == {"tags": ["python"], "source": "documentation"}

    @pytest.mark.asyncio
    async def test_execute_no_results(self, tool, mock_processor):
        """Test execution with no results."""
        # Setup
        mock_processor.search_memory = AsyncMock(return_value=[])
        arguments = {"query": "nonexistent topic"}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is False
        assert "No results found" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_empty_query_validation_error(self, tool):
        """Test execution with empty query."""
        # Setup
        arguments = {"query": ""}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "Error" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_query_too_long_validation_error(self, tool):
        """Test execution with query exceeding max length."""
        # Setup
        arguments = {"query": "x" * 1001}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True

    @pytest.mark.asyncio
    async def test_execute_invalid_limit_below_minimum(self, tool):
        """Test execution with limit below minimum."""
        # Setup
        arguments = {"query": "test", "limit": 0}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True

    @pytest.mark.asyncio
    async def test_execute_invalid_limit_above_maximum(self, tool):
        """Test execution with limit above maximum."""
        # Setup
        arguments = {"query": "test", "limit": 200}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True

    @pytest.mark.asyncio
    async def test_execute_missing_required_query(self, tool):
        """Test execution with missing required query."""
        # Setup
        arguments = {"limit": 5}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True

    @pytest.mark.asyncio
    async def test_execute_processor_search_error(self, tool, mock_processor):
        """Test execution when processor raises SearchError."""
        # Setup
        mock_processor.search_memory = AsyncMock(
            side_effect=SearchError(
                message="Vector index not found", error_code="SEARCH_001"
            )
        )
        arguments = {"query": "test"}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "Search failed" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_processor_database_error(self, tool, mock_processor):
        """Test execution when processor raises DatabaseError."""
        # Setup
        mock_processor.search_memory = AsyncMock(
            side_effect=DatabaseError(
                message="Connection lost", error_code="DB_001"
            )
        )
        arguments = {"query": "test"}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "Database error" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_unexpected_error(self, tool, mock_processor):
        """Test execution when processor raises unexpected exception."""
        # Setup
        mock_processor.search_memory = AsyncMock(side_effect=RuntimeError("Boom"))
        arguments = {"query": "test"}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "unexpected error" in result["content"][0]["text"].lower()


class TestSearchMemoryToolValidation:
    """Test SearchMemoryTool._validate_input() method."""

    @pytest.fixture
    def tool(self):
        """Create SearchMemoryTool with mock processor."""
        mock_processor = Mock(spec=MemoryProcessor)
        return SearchMemoryTool(memory_processor=mock_processor)

    def test_validate_input_success_minimal(self, tool):
        """Test successful validation with minimal input."""
        # Setup
        arguments = {"query": "test"}

        # Execute
        request = tool._validate_input(arguments)

        # Verify
        assert request.query == "test"
        assert request.limit == 10
        assert request.filters is None

    def test_validate_input_success_with_all_params(self, tool):
        """Test successful validation with all parameters."""
        # Setup
        arguments = {
            "query": "test",
            "limit": 20,
            "filters": {"tags": ["python"]},
        }

        # Execute
        request = tool._validate_input(arguments)

        # Verify
        assert request.query == "test"
        assert request.limit == 20
        assert request.filters == {"tags": ["python"]}

    def test_validate_input_empty_query_raises_error(self, tool):
        """Test that empty query raises ValidationError."""
        # Setup
        arguments = {"query": ""}

        # Execute & Verify
        with pytest.raises(ValidationError):
            tool._validate_input(arguments)

    def test_validate_input_missing_query_raises_error(self, tool):
        """Test that missing query raises ValidationError."""
        # Setup
        arguments = {"limit": 5}

        # Execute & Verify
        with pytest.raises(ValidationError):
            tool._validate_input(arguments)


class TestSearchMemoryToolFormatting:
    """Test SearchMemoryTool response formatting methods."""

    @pytest.fixture
    def tool(self):
        """Create SearchMemoryTool with mock processor."""
        mock_processor = Mock(spec=MemoryProcessor)
        return SearchMemoryTool(memory_processor=mock_processor)

    @pytest.fixture
    def sample_results(self):
        """Create sample search results."""
        return [
            SearchResultItem(
                memory_id="id-1",
                text="Python is a programming language" * 10,  # Long text for truncation test
                similarity_score=0.92,
                tags=["python", "programming"],
                source="documentation",
                timestamp=datetime.now(timezone.utc),
            ),
            SearchResultItem(
                memory_id="id-2",
                text="Short text",
                similarity_score=0.85,
                tags=[],
                source="wikipedia",
                timestamp=datetime.now(timezone.utc),
            ),
        ]

    def test_format_response_with_results(self, tool, sample_results):
        """Test response formatting with results."""
        # Execute
        result = tool._format_response(sample_results)

        # Verify
        assert result["isError"] is False
        assert "Found 2 results" in result["content"][0]["text"]
        assert "0.92" in result["content"][0]["text"]
        assert "0.85" in result["content"][0]["text"]

    def test_format_response_no_results(self, tool):
        """Test response formatting with no results."""
        # Execute
        result = tool._format_response([])

        # Verify
        assert result["isError"] is False
        assert "No results found" in result["content"][0]["text"]

    def test_format_response_includes_tags(self, tool, sample_results):
        """Test that response formatting includes tags."""
        # Execute
        result = tool._format_response(sample_results)

        # Verify
        assert "python" in result["content"][0]["text"]
        assert "programming" in result["content"][0]["text"]

    def test_format_response_truncates_long_text(self, tool, sample_results):
        """Test that response truncates long text."""
        # Execute
        result = tool._format_response(sample_results)

        # Verify
        text = result["content"][0]["text"]
        assert "..." in text
        # Verify text is truncated to 200 chars + "..."
        lines = text.split("\n")
        for line in lines:
            if "[Score:" in line:
                # Found a result line
                # Text should be truncated
                if len(sample_results[0].text) > 200:
                    assert "..." in line or line.index("...") > 0

    def test_format_error_validation_error(self, tool):
        """Test error formatting for ValidationError."""
        # Setup
        error = ValidationError.from_exception_data(
            "test",
            [{"loc": ("query",), "msg": "field required", "type": "value_error"}],
        )

        # Execute
        result = tool._format_error(error)

        # Verify
        assert result["isError"] is True
        assert "Error" in result["content"][0]["text"]
        assert "Invalid input" in result["content"][0]["text"]

    def test_format_error_search_error(self, tool):
        """Test error formatting for SearchError."""
        # Setup
        error = SearchError(message="Vector index not found", error_code="SEARCH_001")

        # Execute
        result = tool._format_error(error)

        # Verify
        assert result["isError"] is True
        assert "Search failed" in result["content"][0]["text"]

    def test_format_error_unexpected_error(self, tool):
        """Test error formatting for unexpected error."""
        # Setup
        error = RuntimeError("Boom")

        # Execute
        result = tool._format_error(error)

        # Verify
        assert result["isError"] is True
        assert "unexpected error" in result["content"][0]["text"].lower()


class TestSearchMemoryToolInputSchema:
    """Test SearchMemoryTool input schema."""

    def test_input_schema_structure(self):
        """Test that input schema has correct structure."""
        schema = SearchMemoryTool.input_schema

        # Verify schema structure
        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert "limit" in schema["properties"]
        assert "filters" in schema["properties"]
        assert "query" in schema["required"]

    def test_input_schema_query_constraints(self):
        """Test that query field has correct constraints."""
        schema = SearchMemoryTool.input_schema
        query_schema = schema["properties"]["query"]

        # Verify constraints
        assert query_schema["minLength"] == 1
        assert query_schema["maxLength"] == 1000
        assert query_schema["type"] == "string"

    def test_input_schema_limit_constraints(self):
        """Test that limit field has correct constraints."""
        schema = SearchMemoryTool.input_schema
        limit_schema = schema["properties"]["limit"]

        # Verify constraints
        assert limit_schema["minimum"] == 1
        assert limit_schema["maximum"] == 100
        assert limit_schema["default"] == 10
