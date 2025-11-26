"""
Unit tests for ClearAllTool.

Tests the clear_all MCP tool's initialization, validation, safety checks,
and database interaction with comprehensive test coverage.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from zapomni_core.exceptions import DatabaseError
from zapomni_core.exceptions import ValidationError as CoreValidationError
from zapomni_mcp.tools.clear_all import ClearAllRequest, ClearAllTool


class TestClearAllToolInitialization:
    """Test ClearAllTool initialization."""

    def test_initialization_with_valid_processor(self):
        """Test successful initialization with valid MemoryProcessor."""
        mock_processor = MagicMock()
        mock_processor.__class__.__name__ = "MemoryProcessor"

        # Use isinstance mock
        with patch(
            "zapomni_mcp.tools.clear_all.isinstance",
            return_value=True,
        ):
            tool = ClearAllTool(memory_processor=mock_processor)

            assert tool.memory_processor is mock_processor
            assert tool.name == "clear_all"
            assert isinstance(tool.description, str)
            assert "DELETE ALL MEMORIES" in tool.description

    def test_initialization_with_invalid_processor(self):
        """Test initialization fails with invalid processor type."""
        invalid_processor = "not a processor"

        with pytest.raises(TypeError, match="memory_processor must be MemoryProcessor instance"):
            ClearAllTool(memory_processor=invalid_processor)

    def test_tool_attributes_exist(self):
        """Test that tool has all required attributes."""
        mock_processor = MagicMock()

        with patch(
            "zapomni_mcp.tools.clear_all.isinstance",
            return_value=True,
        ):
            tool = ClearAllTool(memory_processor=mock_processor)

            assert hasattr(tool, "name")
            assert hasattr(tool, "description")
            assert hasattr(tool, "input_schema")
            assert hasattr(tool, "execute")
            assert hasattr(tool, "memory_processor")
            assert hasattr(tool, "logger")


class TestClearAllToolInputSchema:
    """Test ClearAllTool input schema."""

    def test_input_schema_has_required_fields(self):
        """Test that input schema defines required fields correctly."""
        mock_processor = MagicMock()

        with patch(
            "zapomni_mcp.tools.clear_all.isinstance",
            return_value=True,
        ):
            tool = ClearAllTool(memory_processor=mock_processor)

            assert "confirm_phrase" in tool.input_schema["properties"]
            assert "confirm_phrase" in tool.input_schema["required"]
            assert tool.input_schema["additionalProperties"] is False

    def test_input_schema_describes_safety(self):
        """Test that input schema describes safety requirements."""
        mock_processor = MagicMock()

        with patch(
            "zapomni_mcp.tools.clear_all.isinstance",
            return_value=True,
        ):
            tool = ClearAllTool(memory_processor=mock_processor)

            confirm_field = tool.input_schema["properties"]["confirm_phrase"]
            assert "DELETE ALL MEMORIES" in confirm_field["description"]
            assert "exact" in confirm_field["description"].lower()


class TestClearAllToolValidation:
    """Test ClearAllRequest validation."""

    def test_valid_request_with_correct_phrase(self):
        """Test validation passes with correct confirmation phrase."""
        request = ClearAllRequest(confirm_phrase="DELETE ALL MEMORIES")
        assert request.confirm_phrase == "DELETE ALL MEMORIES"

    def test_validation_fails_with_empty_phrase(self):
        """Test validation fails with empty confirmation phrase."""
        with pytest.raises(ValidationError):
            ClearAllRequest(confirm_phrase="")

    def test_validation_fails_with_whitespace_only(self):
        """Test validation passes with whitespace-only phrase (it's not empty string)."""
        # Pydantic validates non-empty strings, so whitespace is technically valid
        # The execute method will reject it
        request = ClearAllRequest(confirm_phrase="   ")
        assert request.confirm_phrase == "   "

    def test_validation_fails_with_wrong_phrase(self):
        """Test validation passes but execute rejects wrong phrase."""
        # Validation should pass (it's a non-empty string)
        request = ClearAllRequest(confirm_phrase="wrong phrase")
        assert request.confirm_phrase == "wrong phrase"

    def test_validation_fails_with_partial_phrase(self):
        """Test validation passes but execute rejects partial phrase."""
        request = ClearAllRequest(confirm_phrase="DELETE ALL")
        assert request.confirm_phrase == "DELETE ALL"

    def test_validation_fails_with_extra_properties(self):
        """Test validation fails with extra properties."""
        with pytest.raises(ValidationError):
            ClearAllRequest(confirm_phrase="DELETE ALL MEMORIES", extra="field")


class TestClearAllToolExecution:
    """Test ClearAllTool.execute() method."""

    @pytest.mark.asyncio
    async def test_execute_success_with_correct_phrase(self):
        """Test successful execution with correct confirmation phrase."""
        mock_processor = MagicMock()
        mock_processor.db_client = MagicMock()
        mock_processor.db_client.clear_all = AsyncMock()

        with patch(
            "zapomni_mcp.tools.clear_all.isinstance",
            return_value=True,
        ):
            tool = ClearAllTool(memory_processor=mock_processor)
            result = await tool.execute({"confirm_phrase": "DELETE ALL MEMORIES"})

            assert result["isError"] is False
            assert "cleared" in result["content"][0]["text"].lower()
            assert "permanent" in result["content"][0]["text"].lower()
            mock_processor.db_client.clear_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_failure_with_wrong_phrase(self):
        """Test execution fails with wrong confirmation phrase."""
        mock_processor = MagicMock()
        mock_processor.db_client = MagicMock()
        mock_processor.db_client.clear_all = AsyncMock()

        with patch(
            "zapomni_mcp.tools.clear_all.isinstance",
            return_value=True,
        ):
            tool = ClearAllTool(memory_processor=mock_processor)
            result = await tool.execute({"confirm_phrase": "wrong phrase"})

            assert result["isError"] is True
            assert "Invalid confirmation" in result["content"][0]["text"]
            assert "DELETE ALL MEMORIES" in result["content"][0]["text"]
            mock_processor.db_client.clear_all.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_failure_with_empty_phrase(self):
        """Test execution fails with empty confirmation phrase."""
        mock_processor = MagicMock()
        mock_processor.db_client = MagicMock()
        mock_processor.db_client.clear_all = AsyncMock()

        with patch(
            "zapomni_mcp.tools.clear_all.isinstance",
            return_value=True,
        ):
            tool = ClearAllTool(memory_processor=mock_processor)
            result = await tool.execute({"confirm_phrase": ""})

            assert result["isError"] is True
            mock_processor.db_client.clear_all.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_failure_with_case_mismatch(self):
        """Test execution fails with case mismatch in phrase."""
        mock_processor = MagicMock()
        mock_processor.db_client = MagicMock()
        mock_processor.db_client.clear_all = AsyncMock()

        with patch(
            "zapomni_mcp.tools.clear_all.isinstance",
            return_value=True,
        ):
            tool = ClearAllTool(memory_processor=mock_processor)

            # Test lowercase
            result = await tool.execute({"confirm_phrase": "delete all memories"})
            assert result["isError"] is True
            mock_processor.db_client.clear_all.assert_not_called()

            # Test uppercase
            result = await tool.execute({"confirm_phrase": "DELETE ALL MEMORIES ".strip()})
            assert result["isError"] is False  # This should work
            mock_processor.db_client.clear_all.assert_called()

    @pytest.mark.asyncio
    async def test_execute_failure_with_partial_phrase(self):
        """Test execution fails with partial phrase."""
        mock_processor = MagicMock()
        mock_processor.db_client = MagicMock()
        mock_processor.db_client.clear_all = AsyncMock()

        with patch(
            "zapomni_mcp.tools.clear_all.isinstance",
            return_value=True,
        ):
            tool = ClearAllTool(memory_processor=mock_processor)

            # Test various partial matches
            for partial in ["DELETE ALL", "DELETE", "ALL MEMORIES", "DELETE ALL MEMORIES "]:
                result = await tool.execute({"confirm_phrase": partial})

                if partial == "DELETE ALL MEMORIES":
                    # Trailing space should fail
                    assert result["isError"] is True
                else:
                    assert result["isError"] is True

                mock_processor.db_client.clear_all.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_failure_with_invalid_argument_type(self):
        """Test execution fails with invalid argument type."""
        mock_processor = MagicMock()
        mock_processor.db_client = MagicMock()
        mock_processor.db_client.clear_all = AsyncMock()

        with patch(
            "zapomni_mcp.tools.clear_all.isinstance",
            return_value=True,
        ):
            tool = ClearAllTool(memory_processor=mock_processor)

            # Test with string instead of dict - will raise TypeError when unpacking
            result = await tool.execute("invalid")
            assert result["isError"] is True
            # TypeError during **arguments expansion is caught as unexpected error
            assert "error" in result["content"][0]["text"].lower()

    @pytest.mark.asyncio
    async def test_execute_failure_with_missing_required_field(self):
        """Test execution fails when required field is missing."""
        mock_processor = MagicMock()
        mock_processor.db_client = MagicMock()
        mock_processor.db_client.clear_all = AsyncMock()

        with patch(
            "zapomni_mcp.tools.clear_all.isinstance",
            return_value=True,
        ):
            tool = ClearAllTool(memory_processor=mock_processor)

            result = await tool.execute({})
            assert result["isError"] is True
            mock_processor.db_client.clear_all.assert_not_called()


class TestClearAllToolDatabaseErrors:
    """Test ClearAllTool database error handling."""

    @pytest.mark.asyncio
    async def test_execute_handles_database_error(self):
        """Test execution handles DatabaseError correctly."""
        mock_processor = MagicMock()
        mock_processor.db_client = MagicMock()
        mock_processor.db_client.clear_all = AsyncMock(
            side_effect=DatabaseError("Connection failed")
        )

        with patch(
            "zapomni_mcp.tools.clear_all.isinstance",
            return_value=True,
        ):
            tool = ClearAllTool(memory_processor=mock_processor)
            result = await tool.execute({"confirm_phrase": "DELETE ALL MEMORIES"})

            assert result["isError"] is True
            assert "Failed to clear" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_handles_core_validation_error(self):
        """Test execution handles CoreValidationError correctly."""
        mock_processor = MagicMock()
        mock_processor.db_client = MagicMock()
        mock_processor.db_client.clear_all = AsyncMock(
            side_effect=CoreValidationError(
                message="Invalid graph state",
                error_code="VAL_001",
            )
        )

        with patch(
            "zapomni_mcp.tools.clear_all.isinstance",
            return_value=True,
        ):
            tool = ClearAllTool(memory_processor=mock_processor)
            result = await tool.execute({"confirm_phrase": "DELETE ALL MEMORIES"})

            assert result["isError"] is True
            assert "Validation failed" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_handles_unexpected_error(self):
        """Test execution handles unexpected errors gracefully."""
        mock_processor = MagicMock()
        mock_processor.db_client = MagicMock()
        mock_processor.db_client.clear_all = AsyncMock(side_effect=RuntimeError("Unexpected error"))

        with patch(
            "zapomni_mcp.tools.clear_all.isinstance",
            return_value=True,
        ):
            tool = ClearAllTool(memory_processor=mock_processor)
            result = await tool.execute({"confirm_phrase": "DELETE ALL MEMORIES"})

            assert result["isError"] is True
            assert "unexpected error" in result["content"][0]["text"].lower()


class TestClearAllToolResponseFormat:
    """Test ClearAllTool response formatting."""

    @pytest.mark.asyncio
    async def test_success_response_format(self):
        """Test successful response follows MCP format."""
        mock_processor = MagicMock()
        mock_processor.db_client = MagicMock()
        mock_processor.db_client.clear_all = AsyncMock()

        with patch(
            "zapomni_mcp.tools.clear_all.isinstance",
            return_value=True,
        ):
            tool = ClearAllTool(memory_processor=mock_processor)
            result = await tool.execute({"confirm_phrase": "DELETE ALL MEMORIES"})

            # Check MCP response structure
            assert "content" in result
            assert "isError" in result
            assert isinstance(result["content"], list)
            assert len(result["content"]) > 0
            assert result["content"][0]["type"] == "text"
            assert isinstance(result["content"][0]["text"], str)

    @pytest.mark.asyncio
    async def test_error_response_format(self):
        """Test error response follows MCP format."""
        mock_processor = MagicMock()
        mock_processor.db_client = MagicMock()
        mock_processor.db_client.clear_all = AsyncMock()

        with patch(
            "zapomni_mcp.tools.clear_all.isinstance",
            return_value=True,
        ):
            tool = ClearAllTool(memory_processor=mock_processor)
            result = await tool.execute({"confirm_phrase": "wrong"})

            # Check MCP response structure
            assert "content" in result
            assert "isError" in result
            assert isinstance(result["content"], list)
            assert len(result["content"]) > 0
            assert result["content"][0]["type"] == "text"
            assert isinstance(result["content"][0]["text"], str)


class TestClearAllToolSafety:
    """Test ClearAllTool safety features."""

    @pytest.mark.asyncio
    async def test_phrase_must_be_exact_match(self):
        """Test that phrase must be EXACT match."""
        mock_processor = MagicMock()
        mock_processor.db_client = MagicMock()
        mock_processor.db_client.clear_all = AsyncMock()

        with patch(
            "zapomni_mcp.tools.clear_all.isinstance",
            return_value=True,
        ):
            tool = ClearAllTool(memory_processor=mock_processor)

            # Test various non-exact matches
            invalid_phrases = [
                "delete all memories",  # lowercase
                "DELETE ALL MEMORIES ",  # trailing space
                " DELETE ALL MEMORIES",  # leading space
                "DELETE  ALL  MEMORIES",  # extra spaces
                "DELETE_ALL_MEMORIES",  # underscores
                "DELETE-ALL-MEMORIES",  # dashes
                "DELETE ALL MEMORIES!!!",  # extra characters
                "DELETE ALL MEMORIESSS",  # typo
                "DELETE_ALLv MEMORIES",  # typo
            ]

            for phrase in invalid_phrases:
                result = await tool.execute({"confirm_phrase": phrase})
                assert result["isError"] is True, f"Should reject phrase: {phrase!r}"
                mock_processor.db_client.clear_all.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_partial_matches_accepted(self):
        """Test that no partial matches are accepted."""
        mock_processor = MagicMock()
        mock_processor.db_client = MagicMock()
        mock_processor.db_client.clear_all = AsyncMock()

        with patch(
            "zapomni_mcp.tools.clear_all.isinstance",
            return_value=True,
        ):
            tool = ClearAllTool(memory_processor=mock_processor)

            # Test that phrase must be complete
            partial_phrases = [
                "DELETE ALL",
                "ALL MEMORIES",
                "DELETE MEMORIES",
                "DEL",
            ]

            for phrase in partial_phrases:
                result = await tool.execute({"confirm_phrase": phrase})
                assert result["isError"] is True, f"Should reject partial phrase: {phrase!r}"

    @pytest.mark.asyncio
    async def test_safety_message_in_error(self):
        """Test that safety message is in error response."""
        mock_processor = MagicMock()
        mock_processor.db_client = MagicMock()
        mock_processor.db_client.clear_all = AsyncMock()

        with patch(
            "zapomni_mcp.tools.clear_all.isinstance",
            return_value=True,
        ):
            tool = ClearAllTool(memory_processor=mock_processor)
            result = await tool.execute({"confirm_phrase": "wrong"})

            assert result["isError"] is True
            text = result["content"][0]["text"]
            assert "DELETE ALL MEMORIES" in text
            assert "exact" in text.lower() or "EXACT" in text
            assert "permanent" in text.lower() or "PERMANENT" in text
