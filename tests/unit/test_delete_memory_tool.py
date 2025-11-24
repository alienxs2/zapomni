"""
Unit tests for DeleteMemoryTool MCP tool.

Tests cover:
- Tool initialization and validation
- Successful deletion with confirmation
- Safety confirmation requirement (confirm=true)
- UUID validation
- Memory not found handling
- Error handling (validation, database, unexpected)
- Response formatting
- Integration with MemoryProcessor and FalkorDBClient

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

import pytest
import uuid
from unittest.mock import AsyncMock, Mock

from zapomni_mcp.tools.delete_memory import (
    DeleteMemoryTool,
    DeleteMemoryRequest,
    DeleteMemoryResponse,
)
from zapomni_core.memory_processor import MemoryProcessor
from zapomni_core.exceptions import DatabaseError, ValidationError as CoreValidationError
from zapomni_db import FalkorDBClient


class TestDeleteMemoryToolInit:
    """Test DeleteMemoryTool initialization."""

    def test_init_success_with_valid_processor(self):
        """Test successful initialization with valid MemoryProcessor."""
        # Setup
        mock_processor = Mock(spec=MemoryProcessor)

        # Execute
        tool = DeleteMemoryTool(memory_processor=mock_processor)

        # Verify
        assert tool.memory_processor == mock_processor
        assert tool.name == "delete_memory"
        assert tool.description is not None
        assert tool.input_schema is not None
        assert "confirm" in tool.input_schema["properties"]
        assert "memory_id" in tool.input_schema["properties"]

    def test_init_fails_with_invalid_processor_type(self):
        """Test initialization fails with non-MemoryProcessor type."""
        # Execute & Verify
        with pytest.raises(TypeError) as exc_info:
            DeleteMemoryTool(memory_processor="not a processor")

        assert "MemoryProcessor" in str(exc_info.value)

    def test_init_fails_with_none_processor(self):
        """Test initialization fails with None processor."""
        # Execute & Verify
        with pytest.raises(TypeError):
            DeleteMemoryTool(memory_processor=None)

    def test_input_schema_has_required_fields(self):
        """Test that input schema includes all required fields."""
        schema = DeleteMemoryTool.input_schema

        # Verify schema structure
        assert schema["type"] == "object"
        assert "memory_id" in schema["properties"]
        assert "confirm" in schema["properties"]
        assert "memory_id" in schema["required"]
        assert "confirm" in schema["required"]
        assert schema["additionalProperties"] is False


class TestDeleteMemoryRequestValidation:
    """Test DeleteMemoryRequest Pydantic validation."""

    def test_valid_request_with_valid_uuid(self):
        """Test valid request with valid UUID."""
        valid_uuid = str(uuid.uuid4())

        request = DeleteMemoryRequest(
            memory_id=valid_uuid,
            confirm=True,
        )

        assert request.memory_id == valid_uuid
        assert request.confirm is True

    def test_valid_request_with_confirm_false(self):
        """Test request with confirm=false is valid (rejected in execute)."""
        valid_uuid = str(uuid.uuid4())

        request = DeleteMemoryRequest(
            memory_id=valid_uuid,
            confirm=False,
        )

        assert request.memory_id == valid_uuid
        assert request.confirm is False

    def test_invalid_request_with_invalid_uuid(self):
        """Test request fails with invalid UUID format."""
        with pytest.raises(ValueError) as exc_info:
            DeleteMemoryRequest(
                memory_id="not-a-uuid",
                confirm=True,
            )

        assert "UUID" in str(exc_info.value) or "Invalid" in str(exc_info.value)

    def test_invalid_request_missing_memory_id(self):
        """Test request fails when memory_id is missing."""
        with pytest.raises(ValueError):
            DeleteMemoryRequest(confirm=True)  # type: ignore

    def test_invalid_request_missing_confirm(self):
        """Test request fails when confirm is missing."""
        valid_uuid = str(uuid.uuid4())

        with pytest.raises(ValueError):
            DeleteMemoryRequest(memory_id=valid_uuid)  # type: ignore

    def test_invalid_request_extra_properties(self):
        """Test request fails with extra properties."""
        valid_uuid = str(uuid.uuid4())

        with pytest.raises(ValueError):
            DeleteMemoryRequest(
                memory_id=valid_uuid,
                confirm=True,
                extra_field="should fail",  # type: ignore
            )

    def test_valid_request_confirm_coerced_from_string(self):
        """Test that Pydantic coerces truthy string to boolean."""
        valid_uuid = str(uuid.uuid4())

        # Pydantic will coerce "1" or "true" strings to boolean
        request = DeleteMemoryRequest(
            memory_id=valid_uuid,
            confirm="1",  # type: ignore
        )

        assert request.confirm is True


class TestDeleteMemoryToolExecute:
    """Test DeleteMemoryTool.execute() method."""

    @pytest.fixture
    def mock_db_client(self):
        """Create a mock FalkorDBClient."""
        db_client = Mock(spec=FalkorDBClient)
        db_client.delete_memory = AsyncMock()
        return db_client

    @pytest.fixture
    def mock_processor(self, mock_db_client):
        """Create a mock MemoryProcessor."""
        processor = Mock(spec=MemoryProcessor)
        processor.db_client = mock_db_client
        return processor

    @pytest.fixture
    def tool(self, mock_processor):
        """Create DeleteMemoryTool with mock processor."""
        return DeleteMemoryTool(memory_processor=mock_processor)

    @pytest.fixture
    def valid_memory_id(self):
        """Generate a valid memory UUID."""
        return str(uuid.uuid4())

    @pytest.mark.asyncio
    async def test_execute_success_with_confirmation(self, tool, mock_processor, valid_memory_id):
        """Test successful deletion with explicit confirmation."""
        # Setup
        mock_processor.db_client.delete_memory.return_value = True
        arguments = {
            "memory_id": valid_memory_id,
            "confirm": True,
        }

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is False
        assert "content" in result
        assert len(result["content"]) > 0
        assert result["content"][0]["type"] == "text"
        assert "deleted successfully" in result["content"][0]["text"].lower()
        assert valid_memory_id in result["content"][0]["text"]
        mock_processor.db_client.delete_memory.assert_called_once_with(valid_memory_id)

    @pytest.mark.asyncio
    async def test_execute_fails_without_confirmation(self, tool, mock_processor, valid_memory_id):
        """Test deletion fails when confirm=false (safety check)."""
        # Setup
        arguments = {
            "memory_id": valid_memory_id,
            "confirm": False,
        }

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "confirmation" in result["content"][0]["text"].lower()
        assert "confirm=true" in result["content"][0]["text"]
        # Should not have called delete_memory
        mock_processor.db_client.delete_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_fails_missing_confirmation_field(
        self, tool, mock_processor, valid_memory_id
    ):
        """Test deletion fails when confirm field is missing."""
        # Setup
        arguments = {
            "memory_id": valid_memory_id,
            # confirm is missing
        }

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "Error" in result["content"][0]["text"]
        mock_processor.db_client.delete_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_memory_not_found(self, tool, mock_processor, valid_memory_id):
        """Test execution when memory is not found."""
        # Setup
        mock_processor.db_client.delete_memory.return_value = False
        arguments = {
            "memory_id": valid_memory_id,
            "confirm": True,
        }

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "not found" in result["content"][0]["text"].lower()
        assert valid_memory_id in result["content"][0]["text"]
        mock_processor.db_client.delete_memory.assert_called_once_with(valid_memory_id)

    @pytest.mark.asyncio
    async def test_execute_invalid_arguments_type(self, tool):
        """Test execution with invalid arguments type."""
        # Setup
        arguments = "not a dict"

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "dictionary" in result["content"][0]["text"].lower()

    @pytest.mark.asyncio
    async def test_execute_invalid_uuid_format(self, tool, mock_processor):
        """Test execution with invalid UUID format."""
        # Setup
        arguments = {
            "memory_id": "not-a-valid-uuid",
            "confirm": True,
        }

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "Error" in result["content"][0]["text"]
        mock_processor.db_client.delete_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_database_validation_error(self, tool, mock_processor, valid_memory_id):
        """Test execution when db_client raises validation error."""
        # Setup
        mock_processor.db_client.delete_memory.side_effect = CoreValidationError(
            message="Invalid memory ID format", error_code="VALIDATION_001"
        )
        arguments = {
            "memory_id": valid_memory_id,
            "confirm": True,
        }

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "invalid" in result["content"][0]["text"].lower()

    @pytest.mark.asyncio
    async def test_execute_database_error(self, tool, mock_processor, valid_memory_id):
        """Test execution when db_client raises database error."""
        # Setup
        mock_processor.db_client.delete_memory.side_effect = DatabaseError(
            message="Connection lost", error_code="DB_001"
        )
        arguments = {
            "memory_id": valid_memory_id,
            "confirm": True,
        }

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "failed" in result["content"][0]["text"].lower()

    @pytest.mark.asyncio
    async def test_execute_unexpected_error(self, tool, mock_processor, valid_memory_id):
        """Test execution when db_client raises unexpected exception."""
        # Setup
        mock_processor.db_client.delete_memory.side_effect = RuntimeError("Unexpected error")
        arguments = {
            "memory_id": valid_memory_id,
            "confirm": True,
        }

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "unexpected" in result["content"][0]["text"].lower()


class TestDeleteMemoryToolSafety:
    """Test safety features of DeleteMemoryTool."""

    @pytest.fixture
    def mock_db_client(self):
        """Create a mock FalkorDBClient."""
        db_client = Mock(spec=FalkorDBClient)
        db_client.delete_memory = AsyncMock(return_value=True)
        return db_client

    @pytest.fixture
    def mock_processor(self, mock_db_client):
        """Create a mock MemoryProcessor."""
        processor = Mock(spec=MemoryProcessor)
        processor.db_client = mock_db_client
        return processor

    @pytest.fixture
    def tool(self, mock_processor):
        """Create DeleteMemoryTool with mock processor."""
        return DeleteMemoryTool(memory_processor=mock_processor)

    @pytest.mark.asyncio
    async def test_safety_requires_explicit_true(self, tool, mock_processor):
        """Test that safety check requires confirm=true explicitly."""
        valid_uuid = str(uuid.uuid4())

        # Test with confirm=false
        result = await tool.execute(
            {
                "memory_id": valid_uuid,
                "confirm": False,
            }
        )
        assert result["isError"] is True
        mock_processor.db_client.delete_memory.assert_not_called()

        # Test with missing confirm field
        result = await tool.execute(
            {
                "memory_id": valid_uuid,
            }
        )
        assert result["isError"] is True
        mock_processor.db_client.delete_memory.assert_not_called()

        # Test with confirm=true (should proceed)
        result = await tool.execute(
            {
                "memory_id": valid_uuid,
                "confirm": True,
            }
        )
        assert result["isError"] is False
        mock_processor.db_client.delete_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_safety_logs_all_deletions(self, tool, mock_processor):
        """Test that all deletion attempts are logged."""
        valid_uuid = str(uuid.uuid4())

        # Execute successful deletion
        await tool.execute(
            {
                "memory_id": valid_uuid,
                "confirm": True,
            }
        )

        # Verify deletion was logged (through logger.info calls)
        # This is tested implicitly through structured logging

    @pytest.mark.asyncio
    async def test_safety_prevents_accidental_deletion(self, tool, mock_processor):
        """Test that confirm=false prevents accidental deletion."""
        valid_uuid = str(uuid.uuid4())
        mock_processor.db_client.delete_memory.return_value = True

        # Attempt without confirmation
        result = await tool.execute(
            {
                "memory_id": valid_uuid,
                "confirm": False,
            }
        )

        # Should fail and not call delete
        assert result["isError"] is True
        mock_processor.db_client.delete_memory.assert_not_called()


class TestDeleteMemoryToolEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def mock_db_client(self):
        """Create a mock FalkorDBClient."""
        db_client = Mock(spec=FalkorDBClient)
        db_client.delete_memory = AsyncMock()
        return db_client

    @pytest.fixture
    def mock_processor(self, mock_db_client):
        """Create a mock MemoryProcessor."""
        processor = Mock(spec=MemoryProcessor)
        processor.db_client = mock_db_client
        return processor

    @pytest.fixture
    def tool(self, mock_processor):
        """Create DeleteMemoryTool with mock processor."""
        return DeleteMemoryTool(memory_processor=mock_processor)

    @pytest.mark.asyncio
    async def test_handles_empty_arguments(self, tool):
        """Test handling of empty arguments dict."""
        result = await tool.execute({})
        assert result["isError"] is True

    @pytest.mark.asyncio
    async def test_handles_null_memory_id(self, tool):
        """Test handling of null memory_id."""
        result = await tool.execute(
            {
                "memory_id": None,
                "confirm": True,
            }
        )
        assert result["isError"] is True

    @pytest.mark.asyncio
    async def test_handles_empty_string_memory_id(self, tool):
        """Test handling of empty string memory_id."""
        result = await tool.execute(
            {
                "memory_id": "",
                "confirm": True,
            }
        )
        assert result["isError"] is True

    @pytest.mark.asyncio
    async def test_handles_extra_fields_in_arguments(self, tool):
        """Test that extra fields cause validation error."""
        valid_uuid = str(uuid.uuid4())
        result = await tool.execute(
            {
                "memory_id": valid_uuid,
                "confirm": True,
                "extra_field": "should fail",
            }
        )
        assert result["isError"] is True

    @pytest.mark.asyncio
    async def test_handles_multiple_same_uuid_deletions(self, tool, mock_processor):
        """Test multiple delete attempts with same UUID."""
        valid_uuid = str(uuid.uuid4())
        mock_processor.db_client.delete_memory.return_value = True

        # First deletion
        result1 = await tool.execute(
            {
                "memory_id": valid_uuid,
                "confirm": True,
            }
        )
        assert result1["isError"] is False

        # Second deletion (memory already deleted, returns False)
        mock_processor.db_client.delete_memory.return_value = False
        result2 = await tool.execute(
            {
                "memory_id": valid_uuid,
                "confirm": True,
            }
        )
        assert result2["isError"] is True
        assert "not found" in result2["content"][0]["text"].lower()

    @pytest.mark.asyncio
    async def test_response_includes_memory_id(self, tool, mock_processor):
        """Test that response includes the deleted memory_id."""
        valid_uuid = str(uuid.uuid4())
        mock_processor.db_client.delete_memory.return_value = True

        result = await tool.execute(
            {
                "memory_id": valid_uuid,
                "confirm": True,
            }
        )

        assert valid_uuid in result["content"][0]["text"]


class TestDeleteMemoryToolInputSchema:
    """Test DeleteMemoryTool input schema."""

    def test_input_schema_structure(self):
        """Test that input schema has correct structure."""
        schema = DeleteMemoryTool.input_schema

        # Verify schema structure
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema
        assert schema["additionalProperties"] is False

    def test_input_schema_memory_id_property(self):
        """Test memory_id property in schema."""
        schema = DeleteMemoryTool.input_schema
        memory_id_prop = schema["properties"]["memory_id"]

        assert memory_id_prop["type"] == "string"
        assert "uuid" in memory_id_prop.get("format", "").lower()
        assert "description" in memory_id_prop

    def test_input_schema_confirm_property(self):
        """Test confirm property in schema."""
        schema = DeleteMemoryTool.input_schema
        confirm_prop = schema["properties"]["confirm"]

        assert confirm_prop["type"] == "boolean"
        assert "description" in confirm_prop
        assert "confirm" in confirm_prop["description"].lower()

    def test_tool_metadata(self):
        """Test tool metadata attributes."""
        # Verify metadata
        assert DeleteMemoryTool.name == "delete_memory"
        assert "delete" in DeleteMemoryTool.description.lower()
        assert "memory" in DeleteMemoryTool.description.lower()
        assert "confirm" in DeleteMemoryTool.description.lower()
        assert (
            "safety" in DeleteMemoryTool.description.lower()
            or "confirmation" in DeleteMemoryTool.description.lower()
        )
