"""
Unit tests for AddMemoryTool MCP tool.

Tests cover:
- Tool initialization and validation
- Successful memory addition
- Input validation and error handling
- Error response formatting
- Integration with MemoryProcessor

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import ValidationError

from zapomni_core.exceptions import (
    DatabaseError,
    EmbeddingError,
    ProcessingError,
)
from zapomni_core.exceptions import ValidationError as CoreValidationError
from zapomni_core.memory_processor import MemoryProcessor
from zapomni_mcp.tools.add_memory import AddMemoryRequest, AddMemoryTool


class TestAddMemoryToolInit:
    """Test AddMemoryTool initialization."""

    def test_init_success_with_valid_processor(self):
        """Test successful initialization with valid MemoryProcessor."""
        # Setup
        mock_processor = Mock(spec=MemoryProcessor)

        # Execute
        tool = AddMemoryTool(memory_processor=mock_processor)

        # Verify
        assert tool.memory_processor == mock_processor
        assert tool.name == "add_memory"
        assert tool.description is not None
        assert tool.input_schema is not None

    def test_init_fails_with_invalid_processor_type(self):
        """Test initialization fails with non-MemoryProcessor type."""
        # Execute & Verify
        with pytest.raises(TypeError) as exc_info:
            AddMemoryTool(memory_processor="not a processor")

        assert "MemoryProcessor" in str(exc_info.value)

    def test_init_fails_with_none_processor(self):
        """Test initialization fails with None processor."""
        # Execute & Verify
        with pytest.raises(TypeError):
            AddMemoryTool(memory_processor=None)


class TestAddMemoryToolExecute:
    """Test AddMemoryTool.execute() method."""

    @pytest.fixture
    def mock_processor(self):
        """Create a mock MemoryProcessor."""
        processor = Mock(spec=MemoryProcessor)
        processor.add_memory = AsyncMock(return_value="test-memory-id-123")
        return processor

    @pytest.fixture
    def tool(self, mock_processor):
        """Create AddMemoryTool with mock processor."""
        return AddMemoryTool(memory_processor=mock_processor)

    @pytest.mark.asyncio
    async def test_execute_success_minimal(self, tool, mock_processor):
        """Test successful execution with minimal input."""
        # Setup
        arguments = {"text": "Test memory"}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is False
        assert "content" in result
        assert len(result["content"]) > 0
        assert result["content"][0]["type"] == "text"
        assert "test-memory-id-123" in result["content"][0]["text"]
        mock_processor.add_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_success_with_metadata(self, tool, mock_processor):
        """Test successful execution with metadata."""
        # Setup
        arguments = {
            "text": "Test memory",
            "metadata": {"source": "user", "tags": ["test"]},
        }

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is False
        assert "content" in result
        # Verify metadata was passed to processor
        call_args = mock_processor.add_memory.call_args
        assert call_args[1]["metadata"] == {"source": "user", "tags": ["test"]}

    @pytest.mark.asyncio
    async def test_execute_empty_text_validation_error(self, tool):
        """Test execution with empty text raises validation error."""
        # Setup
        arguments = {"text": ""}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "Error" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_whitespace_only_text_validation_error(self, tool):
        """Test execution with whitespace-only text raises validation error."""
        # Setup
        arguments = {"text": "   "}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "Error" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_missing_required_text(self, tool):
        """Test execution with missing required text parameter."""
        # Setup
        arguments = {"metadata": {}}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "Error" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_invalid_metadata_type(self, tool):
        """Test execution with invalid metadata type."""
        # Setup
        arguments = {"text": "Test", "metadata": "invalid"}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "Error" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_processor_embedding_error(self, tool, mock_processor):
        """Test execution when processor raises EmbeddingError."""
        # Setup
        mock_processor.add_memory = AsyncMock(
            side_effect=EmbeddingError(message="Embedding failed", error_code="EMB_001")
        )
        arguments = {"text": "Test memory"}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "Error" in result["content"][0]["text"]
        assert "embedding" in result["content"][0]["text"].lower()

    @pytest.mark.asyncio
    async def test_execute_processor_database_error(self, tool, mock_processor):
        """Test execution when processor raises DatabaseError."""
        # Setup
        mock_processor.add_memory = AsyncMock(
            side_effect=DatabaseError(message="DB connection lost", error_code="DB_001")
        )
        arguments = {"text": "Test memory"}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "Error" in result["content"][0]["text"]
        assert "temporarily unavailable" in result["content"][0]["text"].lower()

    @pytest.mark.asyncio
    async def test_execute_unexpected_error(self, tool, mock_processor):
        """Test execution when processor raises unexpected exception."""
        # Setup
        mock_processor.add_memory = AsyncMock(side_effect=RuntimeError("Boom"))
        arguments = {"text": "Test memory"}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "Error" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_text_preview_truncation(self, tool, mock_processor):
        """Test that text preview is truncated correctly."""
        # Setup
        long_text = "x" * 200
        arguments = {"text": long_text}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is False
        assert "..." in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_text_preview_no_truncation(self, tool, mock_processor):
        """Test that short text is not truncated."""
        # Setup
        short_text = "Short text"
        arguments = {"text": short_text}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is False
        assert short_text in result["content"][0]["text"]
        # Check that we don't have extra dots for short text
        text_content = result["content"][0]["text"]
        assert not (text_content.endswith("...") and "Short text..." in text_content)


class TestAddMemoryToolValidation:
    """Test AddMemoryTool._validate_arguments() method."""

    @pytest.fixture
    def tool(self):
        """Create AddMemoryTool with mock processor."""
        mock_processor = Mock(spec=MemoryProcessor)
        return AddMemoryTool(memory_processor=mock_processor)

    def test_validate_arguments_success_minimal(self, tool):
        """Test successful validation with minimal input."""
        # Setup
        arguments = {"text": "Test"}

        # Execute
        text, metadata = tool._validate_arguments(arguments)

        # Verify
        assert text == "Test"
        assert metadata == {}

    def test_validate_arguments_success_with_metadata(self, tool):
        """Test successful validation with metadata."""
        # Setup
        arguments = {"text": "Test", "metadata": {"source": "user"}}

        # Execute
        text, metadata = tool._validate_arguments(arguments)

        # Verify
        assert text == "Test"
        assert metadata == {"source": "user"}

    def test_validate_arguments_strips_whitespace(self, tool):
        """Test that validation strips whitespace from text."""
        # Setup
        arguments = {"text": "  Test  "}

        # Execute
        text, metadata = tool._validate_arguments(arguments)

        # Verify
        assert text == "Test"

    def test_validate_arguments_empty_text_raises_error(self, tool):
        """Test that empty text raises ValidationError."""
        # Setup
        arguments = {"text": ""}

        # Execute & Verify
        with pytest.raises(ValidationError):
            tool._validate_arguments(arguments)

    def test_validate_arguments_missing_text_raises_error(self, tool):
        """Test that missing text raises ValidationError."""
        # Setup
        arguments = {"metadata": {}}

        # Execute & Verify
        with pytest.raises(ValidationError):
            tool._validate_arguments(arguments)


class TestAddMemoryToolFormatting:
    """Test AddMemoryTool response formatting methods."""

    @pytest.fixture
    def tool(self):
        """Create AddMemoryTool with mock processor."""
        mock_processor = Mock(spec=MemoryProcessor)
        return AddMemoryTool(memory_processor=mock_processor)

    def test_format_success(self, tool):
        """Test success response formatting."""
        # Execute
        result = tool._format_success("test-id-123", "Test preview")

        # Verify
        assert result["isError"] is False
        assert "content" in result
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "text"
        assert "test-id-123" in result["content"][0]["text"]
        assert "Test preview" in result["content"][0]["text"]

    def test_format_error_validation_error(self, tool):
        """Test error formatting for ValidationError."""
        # Setup
        error = ValidationError.from_exception_data(
            "test",
            [
                {
                    "type": "value_error",
                    "loc": ("text",),
                    "msg": "field required",
                    "input": {},
                    "ctx": {"error": "field required"},
                }
            ],
        )

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

    def test_format_error_embedding_error(self, tool):
        """Test error formatting for EmbeddingError."""
        # Setup
        error = EmbeddingError(message="Embedding failed", error_code="EMB_001")

        # Execute
        result = tool._format_error(error)

        # Verify
        assert result["isError"] is True
        assert "embedding" in result["content"][0]["text"].lower()

    def test_format_error_unexpected_error(self, tool):
        """Test error formatting for unexpected error."""
        # Setup
        error = RuntimeError("Boom")

        # Execute
        result = tool._format_error(error)

        # Verify
        assert result["isError"] is True
        assert "internal error" in result["content"][0]["text"].lower()


class TestAddMemoryToolInputSchema:
    """Test AddMemoryTool input schema."""

    def test_input_schema_structure(self):
        """Test that input schema has correct structure."""
        schema = AddMemoryTool.input_schema

        # Verify schema structure
        assert schema["type"] == "object"
        assert "text" in schema["properties"]
        assert "metadata" in schema["properties"]
        assert "text" in schema["required"]
        assert "metadata" not in schema["required"]

    def test_input_schema_text_constraints(self):
        """Test that text field has correct constraints."""
        schema = AddMemoryTool.input_schema
        text_schema = schema["properties"]["text"]

        # Verify constraints
        assert text_schema["minLength"] == 1
        assert text_schema["maxLength"] == 10_000_000
        assert text_schema["type"] == "string"
