"""
Unit tests for BuildGraphTool MCP tool.

Tests cover:
- Tool initialization and validation
- Successful graph building
- Input validation and error handling
- Error response formatting
- Integration with EntityExtractor and GraphBuilder
- Entity extraction and graph construction

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import ValidationError

from zapomni_core.exceptions import (
    DatabaseError,
    ExtractionError,
    ProcessingError,
)
from zapomni_core.exceptions import ValidationError as CoreValidationError
from zapomni_core.extractors.entity_extractor import Entity
from zapomni_core.memory_processor import MemoryProcessor
from zapomni_mcp.tools.build_graph import (
    BuildGraphOptions,
    BuildGraphRequest,
    BuildGraphResponse,
    BuildGraphTool,
)


class TestBuildGraphToolInit:
    """Test BuildGraphTool initialization."""

    def test_init_success_with_valid_processor(self):
        """Test successful initialization with valid MemoryProcessor."""
        # Setup
        mock_processor = Mock(spec=MemoryProcessor)

        # Execute
        tool = BuildGraphTool(memory_processor=mock_processor)

        # Verify
        assert tool.memory_processor == mock_processor
        assert tool.name == "build_graph"
        assert tool.description is not None
        assert tool.input_schema is not None

    def test_init_fails_with_invalid_processor_type(self):
        """Test initialization fails with non-MemoryProcessor type."""
        # Execute & Verify
        with pytest.raises(TypeError) as exc_info:
            BuildGraphTool(memory_processor="not a processor")

        assert "MemoryProcessor" in str(exc_info.value)

    def test_init_fails_with_none_processor(self):
        """Test initialization fails with None processor."""
        # Execute & Verify
        with pytest.raises(TypeError):
            BuildGraphTool(memory_processor=None)

    def test_init_sets_logger(self):
        """Test that initialization sets up logger."""
        # Setup
        mock_processor = Mock(spec=MemoryProcessor)

        # Execute
        tool = BuildGraphTool(memory_processor=mock_processor)

        # Verify
        assert tool.logger is not None


class TestBuildGraphToolExecute:
    """Test BuildGraphTool.execute() method."""

    @pytest.fixture
    def mock_entity_extractor(self):
        """Create a mock EntityExtractor."""
        entities = [
            Entity(
                name="Python",
                type="TECHNOLOGY",
                description="Programming language",
                confidence=0.95,
            ),
            Entity(
                name="Guido van Rossum",
                type="PERSON",
                description="Python creator",
                confidence=0.90,
            ),
        ]
        extractor = Mock()
        # Sync method for backward compatibility
        extractor.extract_entities = Mock(return_value=entities)
        # Async method for SSE transport (Phase 2)
        extractor.extract_entities_async = AsyncMock(return_value=entities)
        return extractor

    @pytest.fixture
    def mock_graph_builder(self):
        """Create a mock GraphBuilder."""
        builder = Mock()
        builder.build_graph = AsyncMock(
            return_value={
                "entities_created": 2,
                "entities_merged": 0,
                "relationships_created": 0,
                "total_nodes": 2,
                "total_edges": 0,
            }
        )
        return builder

    @pytest.fixture
    def mock_processor(self, mock_entity_extractor, mock_graph_builder):
        """Create a mock MemoryProcessor."""
        processor = Mock(spec=MemoryProcessor)
        # Both property names for compatibility (code uses .extractor)
        processor.extractor = mock_entity_extractor
        processor.entity_extractor = mock_entity_extractor
        processor.graph_builder = mock_graph_builder
        return processor

    @pytest.fixture
    def tool(self, mock_processor):
        """Create BuildGraphTool with mock processor."""
        return BuildGraphTool(memory_processor=mock_processor)

    @pytest.mark.asyncio
    async def test_execute_success_minimal(self, tool, mock_processor):
        """Test successful execution with minimal input."""
        # Setup
        arguments = {"text": "Python is a programming language"}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is False
        assert "content" in result
        assert len(result["content"]) > 0
        assert result["content"][0]["type"] == "text"
        assert "successfully" in result["content"][0]["text"].lower()
        assert "Entities: 2" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_success_with_options(self, tool, mock_processor):
        """Test successful execution with options."""
        # Setup
        arguments = {
            "text": "Python is a programming language",
            "options": {
                "extract_entities": True,
                "build_relationships": False,
                "confidence_threshold": 0.8,
            },
        }

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is False
        assert "content" in result
        assert "successfully" in result["content"][0]["text"].lower()

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
        arguments = {"options": {}}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "Error" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_text_too_long(self, tool):
        """Test execution with text exceeding max length."""
        # Setup
        arguments = {"text": "x" * 100_001}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "Error" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_invalid_options_type(self, tool):
        """Test execution with invalid options type."""
        # Setup
        arguments = {"text": "Test", "options": "invalid"}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "Error" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_invalid_confidence_threshold(self, tool):
        """Test execution with confidence threshold out of range."""
        # Setup
        arguments = {
            "text": "Test",
            "options": {"confidence_threshold": 1.5},
        }

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "Error" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_extractor_error(self, tool, mock_processor):
        """Test execution when entity extractor raises ExtractionError."""
        # Setup - async method since code uses extract_entities_async
        mock_processor.extractor.extract_entities_async = AsyncMock(
            side_effect=ExtractionError(message="Extraction failed", error_code="EXTR_001")
        )
        arguments = {"text": "Test text"}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "Error" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_database_error(self, tool, mock_processor):
        """Test execution when graph builder raises DatabaseError."""
        # Setup
        mock_processor.graph_builder.build_graph = AsyncMock(
            side_effect=DatabaseError(message="DB connection lost", error_code="DB_001")
        )
        arguments = {"text": "Test text"}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "temporarily unavailable" in result["content"][0]["text"].lower()

    @pytest.mark.asyncio
    async def test_execute_processing_error(self, tool, mock_processor):
        """Test execution when graph builder raises ProcessingError."""
        # Setup
        mock_processor.graph_builder.build_graph = AsyncMock(
            side_effect=ProcessingError(message="Processing failed", error_code="PROC_001")
        )
        arguments = {"text": "Test text"}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "Error" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_unexpected_error(self, tool, mock_processor):
        """Test execution when unexpected exception is raised."""
        # Setup - set extractor to None to trigger error
        mock_processor.extractor = None
        arguments = {"text": "Test text"}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "Error" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_includes_processing_time(self, tool):
        """Test that response includes processing time."""
        # Setup
        arguments = {"text": "Test text"}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is False
        assert "ms" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_with_no_entities(self, tool, mock_processor):
        """Test execution when no entities are extracted."""
        # Setup - use async method since code uses extract_entities_async
        mock_processor.extractor.extract_entities_async = AsyncMock(return_value=[])
        arguments = {"text": "No entities here"}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is False
        assert "Entities: 0" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_shows_entity_count(self, tool):
        """Test that response shows entity count."""
        # Setup
        arguments = {"text": "Python was created by Guido van Rossum"}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is False
        assert "Entities: 2" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_shows_relationships_count(self, tool):
        """Test that response shows relationships count."""
        # Setup
        arguments = {"text": "Test"}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is False
        assert "Relationships:" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_shows_confidence_score(self, tool):
        """Test that response shows average confidence score."""
        # Setup
        arguments = {"text": "Test"}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is False
        assert "Confidence:" in result["content"][0]["text"]


class TestBuildGraphToolValidation:
    """Test BuildGraphTool._validate_arguments() method."""

    @pytest.fixture
    def tool(self):
        """Create BuildGraphTool with mock processor."""
        mock_processor = Mock(spec=MemoryProcessor)
        return BuildGraphTool(memory_processor=mock_processor)

    def test_validate_arguments_success_minimal(self, tool):
        """Test successful validation with minimal input."""
        # Setup
        arguments = {"text": "Test"}

        # Execute
        text, options = tool._validate_arguments(arguments)

        # Verify
        assert text == "Test"
        assert isinstance(options, BuildGraphOptions)
        assert options.extract_entities is True
        assert options.build_relationships is False
        assert options.confidence_threshold == 0.7

    def test_validate_arguments_success_with_options(self, tool):
        """Test successful validation with all options."""
        # Setup
        arguments = {
            "text": "Test",
            "options": {
                "extract_entities": False,
                "build_relationships": True,
                "confidence_threshold": 0.5,
            },
        }

        # Execute
        text, options = tool._validate_arguments(arguments)

        # Verify
        assert text == "Test"
        assert options.extract_entities is False
        assert options.build_relationships is True
        assert options.confidence_threshold == 0.5

    def test_validate_arguments_strips_whitespace(self, tool):
        """Test that validation strips whitespace from text."""
        # Setup
        arguments = {"text": "  Test  "}

        # Execute
        text, options = tool._validate_arguments(arguments)

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
        arguments = {"options": {}}

        # Execute & Verify
        with pytest.raises(ValidationError):
            tool._validate_arguments(arguments)

    def test_validate_arguments_extra_fields_rejected(self, tool):
        """Test that extra fields in options are rejected."""
        # Setup
        arguments = {
            "text": "Test",
            "options": {"extra_field": "value"},
        }

        # Execute & Verify
        with pytest.raises(ValidationError):
            tool._validate_arguments(arguments)


class TestBuildGraphToolFormatting:
    """Test BuildGraphTool response formatting methods."""

    @pytest.fixture
    def tool(self):
        """Create BuildGraphTool with mock processor."""
        mock_processor = Mock(spec=MemoryProcessor)
        return BuildGraphTool(memory_processor=mock_processor)

    def test_format_success(self, tool):
        """Test success response formatting."""
        # Execute
        result = tool._format_success(
            entities_count=5,
            relationships_count=3,
            entities_created=4,
            entities_merged=1,
            processing_time_ms=123.45,
            confidence_avg=0.85,
        )

        # Verify
        assert result["isError"] is False
        assert "content" in result
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "text"
        assert "successfully" in result["content"][0]["text"].lower()
        assert "5" in result["content"][0]["text"]
        assert "0.85" in result["content"][0]["text"]

    def test_format_success_includes_all_stats(self, tool):
        """Test that success format includes all statistics."""
        # Execute
        result = tool._format_success(
            entities_count=10,
            relationships_count=5,
            entities_created=8,
            entities_merged=2,
            processing_time_ms=456.78,
            confidence_avg=0.92,
        )

        # Verify
        text = result["content"][0]["text"]
        assert "Entities: 10" in text
        assert "Created: 8" in text
        assert "Merged: 2" in text
        assert "Relationships: 5" in text
        assert "0.92" in text
        assert "456.8" in text

    def test_format_error_validation_error(self, tool):
        """Test error formatting for ValidationError."""
        # Setup - create a ValidationError by triggering Pydantic validation
        try:
            BuildGraphRequest(text="")
        except ValidationError as error:
            # Execute
            result = tool._format_error(error, 50.0)

            # Verify
            assert result["isError"] is True
            assert "Error" in result["content"][0]["text"]

    def test_format_error_database_error(self, tool):
        """Test error formatting for DatabaseError."""
        # Setup
        error = DatabaseError(message="Connection lost", error_code="DB_001")

        # Execute
        result = tool._format_error(error, 100.0)

        # Verify
        assert result["isError"] is True
        assert "temporarily unavailable" in result["content"][0]["text"].lower()

    def test_format_error_extraction_error(self, tool):
        """Test error formatting for ExtractionError."""
        # Setup
        error = ExtractionError(message="Extraction failed", error_code="EXTR_001")

        # Execute
        result = tool._format_error(error, 75.0)

        # Verify
        assert result["isError"] is True
        assert "extract" in result["content"][0]["text"].lower()

    def test_format_error_processing_error(self, tool):
        """Test error formatting for ProcessingError."""
        # Setup
        error = ProcessingError(message="Processing failed", error_code="PROC_001")

        # Execute
        result = tool._format_error(error, 200.0)

        # Verify
        assert result["isError"] is True
        assert "Error" in result["content"][0]["text"]

    def test_format_error_unexpected_error(self, tool):
        """Test error formatting for unexpected error."""
        # Setup
        error = RuntimeError("Boom")

        # Execute
        result = tool._format_error(error, 150.0)

        # Verify
        assert result["isError"] is True
        assert "internal error" in result["content"][0]["text"].lower()


class TestBuildGraphToolInputSchema:
    """Test BuildGraphTool input schema."""

    def test_input_schema_structure(self):
        """Test that input schema has correct structure."""
        schema = BuildGraphTool.input_schema

        # Verify schema structure
        assert schema["type"] == "object"
        assert "text" in schema["properties"]
        assert "options" in schema["properties"]
        assert "text" in schema["required"]
        assert "options" not in schema["required"]

    def test_input_schema_text_constraints(self):
        """Test that text field has correct constraints."""
        schema = BuildGraphTool.input_schema
        text_schema = schema["properties"]["text"]

        # Verify constraints
        assert text_schema["minLength"] == 1
        assert text_schema["maxLength"] == 100_000
        assert text_schema["type"] == "string"

    def test_input_schema_options_structure(self):
        """Test that options field has correct structure."""
        schema = BuildGraphTool.input_schema
        options_schema = schema["properties"]["options"]

        # Verify options properties
        assert "extract_entities" in options_schema["properties"]
        assert "build_relationships" in options_schema["properties"]
        assert "confidence_threshold" in options_schema["properties"]

    def test_input_schema_confidence_threshold_range(self):
        """Test that confidence threshold has correct constraints."""
        schema = BuildGraphTool.input_schema
        options_schema = schema["properties"]["options"]
        threshold_schema = options_schema["properties"]["confidence_threshold"]

        # Verify range
        assert threshold_schema["minimum"] == 0.0
        assert threshold_schema["maximum"] == 1.0

    def test_input_schema_no_additional_properties(self):
        """Test that schema rejects additional properties."""
        schema = BuildGraphTool.input_schema

        # Verify
        assert schema["additionalProperties"] is False
        options_schema = schema["properties"]["options"]
        assert options_schema["additionalProperties"] is False


class TestBuildGraphToolDataModels:
    """Test BuildGraphTool Pydantic models."""

    def test_build_graph_options_defaults(self):
        """Test BuildGraphOptions default values."""
        # Execute
        options = BuildGraphOptions()

        # Verify
        assert options.extract_entities is True
        assert options.build_relationships is False
        assert options.confidence_threshold == 0.7

    def test_build_graph_options_custom_values(self):
        """Test BuildGraphOptions with custom values."""
        # Execute
        options = BuildGraphOptions(
            extract_entities=False,
            build_relationships=True,
            confidence_threshold=0.5,
        )

        # Verify
        assert options.extract_entities is False
        assert options.build_relationships is True
        assert options.confidence_threshold == 0.5

    def test_build_graph_options_invalid_threshold(self):
        """Test that invalid confidence threshold is rejected."""
        # Execute & Verify
        with pytest.raises(ValidationError):
            BuildGraphOptions(confidence_threshold=1.5)

    def test_build_graph_request_validation(self):
        """Test BuildGraphRequest validation."""
        # Setup
        request_data = {
            "text": "Test",
            "options": {"confidence_threshold": 0.8},
        }

        # Execute
        request = BuildGraphRequest(**request_data)

        # Verify
        assert request.text == "Test"
        assert request.options.confidence_threshold == 0.8

    def test_build_graph_response_creation(self):
        """Test BuildGraphResponse creation."""
        # Setup & Execute
        response = BuildGraphResponse(
            status="success",
            entities_count=5,
            relationships_count=3,
            entities_created=4,
            entities_merged=1,
            processing_time_ms=100.5,
            confidence_avg=0.85,
        )

        # Verify
        assert response.status == "success"
        assert response.entities_count == 5
        assert response.processing_time_ms == 100.5
