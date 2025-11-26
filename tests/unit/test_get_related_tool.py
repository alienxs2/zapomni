"""
Unit tests for GetRelatedTool MCP tool.

Tests cover:
- Tool initialization and validation
- Successful graph traversal operations
- Input validation and error handling
- Result formatting
- Integration with MemoryProcessor

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from zapomni_core.exceptions import (
    DatabaseError,
)
from zapomni_core.exceptions import ValidationError as CoreValidationError
from zapomni_core.memory_processor import MemoryProcessor
from zapomni_db.models import Entity
from zapomni_mcp.tools.get_related import GetRelatedRequest, GetRelatedTool


class TestGetRelatedToolInit:
    """Test GetRelatedTool initialization."""

    def test_init_success_with_valid_processor(self):
        """Test successful initialization with valid MemoryProcessor."""
        # Setup
        mock_processor = Mock(spec=MemoryProcessor)

        # Execute
        tool = GetRelatedTool(memory_processor=mock_processor)

        # Verify
        assert tool.memory_processor == mock_processor
        assert tool.name == "get_related"
        assert tool.description is not None
        assert tool.input_schema is not None

    def test_init_fails_with_invalid_processor_type(self):
        """Test initialization fails with non-MemoryProcessor type."""
        # Execute & Verify
        with pytest.raises(TypeError) as exc_info:
            GetRelatedTool(memory_processor="not a processor")

        assert "MemoryProcessor" in str(exc_info.value)

    def test_init_fails_with_none_processor(self):
        """Test initialization fails with None processor."""
        # Execute & Verify
        with pytest.raises(TypeError):
            GetRelatedTool(memory_processor=None)

    def test_init_sets_logger(self):
        """Test that init properly sets logger."""
        # Setup
        mock_processor = Mock(spec=MemoryProcessor)

        # Execute
        tool = GetRelatedTool(memory_processor=mock_processor)

        # Verify
        assert tool.logger is not None


class TestGetRelatedToolExecute:
    """Test GetRelatedTool.execute() method."""

    @pytest.fixture
    def mock_processor(self):
        """Create a mock MemoryProcessor."""
        processor = Mock(spec=MemoryProcessor)
        processor.get_related_entities = AsyncMock(return_value=[])
        return processor

    @pytest.fixture
    def tool(self, mock_processor):
        """Create GetRelatedTool with mock processor."""
        return GetRelatedTool(memory_processor=mock_processor)

    @pytest.fixture
    def sample_entity_id(self):
        """Generate a sample entity UUID."""
        return str(uuid4())

    @pytest.fixture
    def sample_results(self):
        """Create sample related entities."""
        return [
            Entity(
                name="Python",
                type="TECHNOLOGY",
                description="Programming language",
                confidence=0.92,
            ),
            Entity(
                name="Machine Learning",
                type="CONCEPT",
                description="Subset of AI focused on learning from data",
                confidence=0.85,
            ),
            Entity(
                name="Data Science",
                type="FIELD",
                description="",
                confidence=0.78,
            ),
        ]

    @pytest.mark.asyncio
    async def test_execute_success_minimal(
        self, tool, mock_processor, sample_entity_id, sample_results
    ):
        """Test successful execution with minimal input."""
        # Setup
        mock_processor.get_related_entities = AsyncMock(return_value=sample_results)
        arguments = {"entity_id": sample_entity_id}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is False
        assert "content" in result
        assert "Found 3 related entities" in result["content"][0]["text"]
        mock_processor.get_related_entities.assert_called_once()
        call_args = mock_processor.get_related_entities.call_args
        assert call_args[1]["entity_id"] == sample_entity_id
        assert call_args[1]["depth"] == 2  # default
        assert call_args[1]["limit"] == 20  # default

    @pytest.mark.asyncio
    async def test_execute_success_custom_depth(
        self, tool, mock_processor, sample_entity_id, sample_results
    ):
        """Test successful execution with custom depth."""
        # Setup
        mock_processor.get_related_entities = AsyncMock(return_value=sample_results)
        arguments = {"entity_id": sample_entity_id, "depth": 4}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is False
        call_args = mock_processor.get_related_entities.call_args
        assert call_args[1]["depth"] == 4

    @pytest.mark.asyncio
    async def test_execute_success_custom_limit(
        self, tool, mock_processor, sample_entity_id, sample_results
    ):
        """Test successful execution with custom limit."""
        # Setup
        mock_processor.get_related_entities = AsyncMock(return_value=sample_results[:1])
        arguments = {"entity_id": sample_entity_id, "limit": 5}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is False
        call_args = mock_processor.get_related_entities.call_args
        assert call_args[1]["limit"] == 5

    @pytest.mark.asyncio
    async def test_execute_no_results(self, tool, mock_processor, sample_entity_id):
        """Test execution with no related entities."""
        # Setup
        mock_processor.get_related_entities = AsyncMock(return_value=[])
        arguments = {"entity_id": sample_entity_id}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is False
        assert "No related entities found" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_invalid_entity_id_format(self, tool):
        """Test execution with invalid entity UUID format."""
        # Setup
        arguments = {"entity_id": "not-a-uuid"}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "Error" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_missing_entity_id(self, tool):
        """Test execution with missing required entity_id."""
        # Setup
        arguments = {"depth": 2}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True

    @pytest.mark.asyncio
    async def test_execute_invalid_depth_below_minimum(self, tool, sample_entity_id):
        """Test execution with depth below minimum."""
        # Setup
        arguments = {"entity_id": sample_entity_id, "depth": 0}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True

    @pytest.mark.asyncio
    async def test_execute_invalid_depth_above_maximum(self, tool, sample_entity_id):
        """Test execution with depth above maximum."""
        # Setup
        arguments = {"entity_id": sample_entity_id, "depth": 6}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True

    @pytest.mark.asyncio
    async def test_execute_invalid_limit_below_minimum(self, tool, sample_entity_id):
        """Test execution with limit below minimum."""
        # Setup
        arguments = {"entity_id": sample_entity_id, "limit": 0}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True

    @pytest.mark.asyncio
    async def test_execute_invalid_limit_above_maximum(self, tool, sample_entity_id):
        """Test execution with limit above maximum."""
        # Setup
        arguments = {"entity_id": sample_entity_id, "limit": 100}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True

    @pytest.mark.asyncio
    async def test_execute_processor_database_error(self, tool, mock_processor, sample_entity_id):
        """Test execution when processor raises DatabaseError."""
        # Setup
        mock_processor.get_related_entities = AsyncMock(
            side_effect=DatabaseError(message="Connection lost", error_code="DB_001")
        )
        arguments = {"entity_id": sample_entity_id}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "Database error" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_processor_validation_error(self, tool, mock_processor, sample_entity_id):
        """Test execution when processor raises ValidationError."""
        # Setup
        mock_processor.get_related_entities = AsyncMock(
            side_effect=CoreValidationError(message="Invalid entity ID", error_code="VAL_001")
        )
        arguments = {"entity_id": sample_entity_id}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True

    @pytest.mark.asyncio
    async def test_execute_unexpected_error(self, tool, mock_processor, sample_entity_id):
        """Test execution when processor raises unexpected exception."""
        # Setup
        mock_processor.get_related_entities = AsyncMock(side_effect=RuntimeError("Boom"))
        arguments = {"entity_id": sample_entity_id}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "unexpected error" in result["content"][0]["text"].lower()


class TestGetRelatedToolValidation:
    """Test GetRelatedTool._validate_input() method."""

    @pytest.fixture
    def tool(self):
        """Create GetRelatedTool with mock processor."""
        mock_processor = Mock(spec=MemoryProcessor)
        return GetRelatedTool(memory_processor=mock_processor)

    @pytest.fixture
    def sample_entity_id(self):
        """Generate a sample entity UUID."""
        return str(uuid4())

    def test_validate_input_success_minimal(self, tool, sample_entity_id):
        """Test successful validation with minimal input."""
        # Setup
        arguments = {"entity_id": sample_entity_id}

        # Execute
        request = tool._validate_input(arguments)

        # Verify
        assert request.entity_id == sample_entity_id
        assert request.depth == 2
        assert request.limit == 20
        assert request.relationship_types is None

    def test_validate_input_success_with_all_params(self, tool, sample_entity_id):
        """Test successful validation with all parameters."""
        # Setup
        arguments = {
            "entity_id": sample_entity_id,
            "depth": 3,
            "limit": 15,
            "relationship_types": ["MENTIONS", "RELATED_TO"],
        }

        # Execute
        request = tool._validate_input(arguments)

        # Verify
        assert request.entity_id == sample_entity_id
        assert request.depth == 3
        assert request.limit == 15
        assert request.relationship_types == ["MENTIONS", "RELATED_TO"]

    def test_validate_input_invalid_uuid_raises_error(self, tool):
        """Test that invalid UUID raises error."""
        # Setup
        arguments = {"entity_id": "not-a-uuid"}

        # Execute & Verify
        with pytest.raises((ValidationError, CoreValidationError)):
            tool._validate_input(arguments)

    def test_validate_input_missing_entity_id_raises_error(self, tool):
        """Test that missing entity_id raises ValidationError."""
        # Setup
        arguments = {"depth": 2}

        # Execute & Verify
        with pytest.raises(ValidationError):
            tool._validate_input(arguments)

    def test_validate_input_invalid_depth_low_raises_error(self, tool, sample_entity_id):
        """Test that depth below 1 raises error."""
        # Setup
        arguments = {"entity_id": sample_entity_id, "depth": 0}

        # Execute & Verify
        with pytest.raises(CoreValidationError):
            tool._validate_input(arguments)

    def test_validate_input_invalid_depth_high_raises_error(self, tool, sample_entity_id):
        """Test that depth above 5 raises error."""
        # Setup
        arguments = {"entity_id": sample_entity_id, "depth": 6}

        # Execute & Verify
        with pytest.raises(CoreValidationError):
            tool._validate_input(arguments)

    def test_validate_input_invalid_limit_low_raises_error(self, tool, sample_entity_id):
        """Test that limit below 1 raises error."""
        # Setup
        arguments = {"entity_id": sample_entity_id, "limit": 0}

        # Execute & Verify
        with pytest.raises(CoreValidationError):
            tool._validate_input(arguments)

    def test_validate_input_invalid_limit_high_raises_error(self, tool, sample_entity_id):
        """Test that limit above 50 raises error."""
        # Setup
        arguments = {"entity_id": sample_entity_id, "limit": 51}

        # Execute & Verify
        with pytest.raises(CoreValidationError):
            tool._validate_input(arguments)

    def test_validate_input_valid_depth_boundaries(self, tool, sample_entity_id):
        """Test that valid depth boundaries work."""
        # Test depth 1
        request1 = tool._validate_input({"entity_id": sample_entity_id, "depth": 1})
        assert request1.depth == 1

        # Test depth 5
        request5 = tool._validate_input({"entity_id": sample_entity_id, "depth": 5})
        assert request5.depth == 5

    def test_validate_input_valid_limit_boundaries(self, tool, sample_entity_id):
        """Test that valid limit boundaries work."""
        # Test limit 1
        request1 = tool._validate_input({"entity_id": sample_entity_id, "limit": 1})
        assert request1.limit == 1

        # Test limit 50
        request50 = tool._validate_input({"entity_id": sample_entity_id, "limit": 50})
        assert request50.limit == 50


class TestGetRelatedToolFormatting:
    """Test GetRelatedTool response formatting methods."""

    @pytest.fixture
    def tool(self):
        """Create GetRelatedTool with mock processor."""
        mock_processor = Mock(spec=MemoryProcessor)
        return GetRelatedTool(memory_processor=mock_processor)

    @pytest.fixture
    def sample_entity_id(self):
        """Generate a sample entity UUID."""
        return str(uuid4())

    @pytest.fixture
    def sample_results(self):
        """Create sample related entities."""
        return [
            Entity(
                name="Python",
                type="TECHNOLOGY",
                description="Programming language",
                confidence=0.92,
            ),
            Entity(
                name="Machine Learning",
                type="CONCEPT",
                description="Subset of AI",
                confidence=0.85,
            ),
            Entity(
                name="Data Science",
                type="FIELD",
                description="",
                confidence=0.78,
            ),
        ]

    def test_format_response_with_results(self, tool, sample_entity_id, sample_results):
        """Test response formatting with results."""
        # Execute
        result = tool._format_response(sample_entity_id, sample_results)

        # Verify
        assert result["isError"] is False
        assert "Found 3 related entities" in result["content"][0]["text"]
        assert "Python" in result["content"][0]["text"]
        assert "0.92" in result["content"][0]["text"]

    def test_format_response_no_results(self, tool, sample_entity_id):
        """Test response formatting with no results."""
        # Execute
        result = tool._format_response(sample_entity_id, [])

        # Verify
        assert result["isError"] is False
        assert "No related entities found" in result["content"][0]["text"]

    def test_format_response_includes_entity_data(self, tool, sample_entity_id, sample_results):
        """Test that response includes entity names, types, and confidence."""
        # Execute
        result = tool._format_response(sample_entity_id, sample_results)

        # Verify
        text = result["content"][0]["text"]
        assert "Python" in text
        assert "TECHNOLOGY" in text
        assert "Machine Learning" in text
        assert "CONCEPT" in text
        assert "Confidence" in text

    def test_format_response_truncates_long_descriptions(self, tool, sample_entity_id):
        """Test that long descriptions are truncated."""
        # Setup
        long_desc = "x" * 200
        entities = [
            Entity(
                name="Test",
                type="TEST",
                description=long_desc,
                confidence=0.9,
            )
        ]

        # Execute
        result = tool._format_response(sample_entity_id, entities)

        # Verify
        text = result["content"][0]["text"]
        # Should have truncation indicator
        assert "..." in text
        # Should not include full description
        assert long_desc not in text

    def test_format_response_handles_empty_description(self, tool, sample_entity_id):
        """Test that empty descriptions are handled gracefully."""
        # Setup
        entities = [
            Entity(
                name="Test",
                type="TEST",
                description="",
                confidence=0.9,
            )
        ]

        # Execute
        result = tool._format_response(sample_entity_id, entities)

        # Verify
        assert result["isError"] is False
        assert "Test" in result["content"][0]["text"]

    def test_format_error_core_validation_error(self, tool):
        """Test error formatting for CoreValidationError."""
        # Setup
        error = CoreValidationError(
            message="Invalid input: entity_id must be a valid UUID", error_code="INVALID_UUID"
        )

        # Execute
        result = tool._format_error(error)

        # Verify
        assert result["isError"] is True
        assert "Error" in result["content"][0]["text"]
        assert "Invalid input" in result["content"][0]["text"]

    def test_format_error_database_error(self, tool):
        """Test error formatting for DatabaseError."""
        # Setup
        error = DatabaseError(message="Connection lost", error_code="DB_001")

        # Execute
        result = tool._format_error(error)

        # Verify
        assert result["isError"] is True
        assert "Database error" in result["content"][0]["text"]

    def test_format_error_unexpected_error(self, tool):
        """Test error formatting for unexpected error."""
        # Setup
        error = RuntimeError("Boom")

        # Execute
        result = tool._format_error(error)

        # Verify
        assert result["isError"] is True
        assert "unexpected error" in result["content"][0]["text"].lower()


class TestGetRelatedToolInputSchema:
    """Test GetRelatedTool input schema."""

    def test_input_schema_structure(self):
        """Test that input schema has correct structure."""
        schema = GetRelatedTool.input_schema

        # Verify schema structure
        assert schema["type"] == "object"
        assert "entity_id" in schema["properties"]
        assert "depth" in schema["properties"]
        assert "limit" in schema["properties"]
        assert "entity_id" in schema["required"]

    def test_input_schema_entity_id_required(self):
        """Test that entity_id is required."""
        schema = GetRelatedTool.input_schema
        assert "entity_id" in schema["required"]

    def test_input_schema_depth_constraints(self):
        """Test that depth field has correct constraints."""
        schema = GetRelatedTool.input_schema
        depth_schema = schema["properties"]["depth"]

        # Verify constraints
        assert depth_schema["minimum"] == 1
        assert depth_schema["maximum"] == 5
        assert depth_schema["default"] == 2

    def test_input_schema_limit_constraints(self):
        """Test that limit field has correct constraints."""
        schema = GetRelatedTool.input_schema
        limit_schema = schema["properties"]["limit"]

        # Verify constraints
        assert limit_schema["minimum"] == 1
        assert limit_schema["maximum"] == 50
        assert limit_schema["default"] == 20

    def test_input_schema_relationship_types_is_array(self):
        """Test that relationship_types is an optional array."""
        schema = GetRelatedTool.input_schema
        rel_types_schema = schema["properties"]["relationship_types"]

        # Verify structure
        assert rel_types_schema["type"] == "array"
        assert rel_types_schema["items"]["type"] == "string"
