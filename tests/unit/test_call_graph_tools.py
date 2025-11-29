"""
Unit tests for call graph MCP tools (get_callers, get_callees).

Tests the GetCallersTool and GetCalleesTool classes which provide
MCP tools for querying the call graph in FalkorDB.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from zapomni_db.exceptions import DatabaseError
from zapomni_db.exceptions import ValidationError as DBValidationError
from zapomni_db.falkordb_client import FalkorDBClient
from zapomni_mcp.tools.call_graph import (
    GetCallersRequest,
    GetCalleesRequest,
    GetCallersTool,
    GetCalleesTool,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_db_client():
    """Create a mock FalkorDBClient."""
    client = MagicMock(spec=FalkorDBClient)
    client.get_callers = AsyncMock(return_value=[])
    client.get_callees = AsyncMock(return_value=[])
    return client


@pytest.fixture
def get_callers_tool(mock_db_client):
    """Create GetCallersTool with mocked db_client."""
    return GetCallersTool(db_client=mock_db_client)


@pytest.fixture
def get_callees_tool(mock_db_client):
    """Create GetCalleesTool with mocked db_client."""
    return GetCalleesTool(db_client=mock_db_client)


# =============================================================================
# GetCallersTool Tests
# =============================================================================


class TestGetCallersToolInit:
    """Tests for GetCallersTool initialization."""

    def test_init_success(self, mock_db_client):
        """Test successful initialization."""
        tool = GetCallersTool(db_client=mock_db_client)
        assert tool.db_client == mock_db_client
        assert tool.name == "get_callers"

    def test_init_wrong_type_raises(self):
        """Test initialization with wrong type raises TypeError."""
        with pytest.raises(TypeError, match="must be FalkorDBClient instance"):
            GetCallersTool(db_client="not_a_client")

    def test_tool_attributes(self, mock_db_client):
        """Test tool has required attributes."""
        tool = GetCallersTool(db_client=mock_db_client)
        assert tool.name == "get_callers"
        assert "Find all functions that call" in tool.description
        assert "qualified_name" in tool.input_schema["properties"]
        assert "limit" in tool.input_schema["properties"]
        assert "workspace_id" in tool.input_schema["properties"]


class TestGetCallersToolExecute:
    """Tests for GetCallersTool.execute()."""

    @pytest.mark.asyncio
    async def test_execute_success_with_results(self, get_callers_tool, mock_db_client):
        """Test successful execution with callers found."""
        mock_db_client.get_callers.return_value = [
            {
                "caller_qualified_name": "module.MyClass.process",
                "caller_id": "caller-uuid-1",
                "caller_file_path": "/path/to/file.py",
                "call_line": 42,
                "call_type": "function",
                "arguments_count": 2,
                "call_count": 1,
            },
            {
                "caller_qualified_name": "module.another_func",
                "caller_id": "caller-uuid-2",
                "caller_file_path": "/path/to/other.py",
                "call_line": 15,
                "call_type": "method",
                "arguments_count": 1,
                "call_count": 3,
            },
        ]

        result = await get_callers_tool.execute(
            {
                "qualified_name": "module.helper_func",
                "limit": 20,
            }
        )

        assert result["isError"] is False
        assert len(result["content"]) == 1
        assert "Found 2 caller(s)" in result["content"][0]["text"]
        assert "module.MyClass.process" in result["content"][0]["text"]
        assert "module.another_func" in result["content"][0]["text"]
        assert "Line: 42" in result["content"][0]["text"]
        assert "Call count: 3" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_success_no_results(self, get_callers_tool, mock_db_client):
        """Test successful execution with no callers found."""
        mock_db_client.get_callers.return_value = []

        result = await get_callers_tool.execute(
            {
                "qualified_name": "module.unused_func",
            }
        )

        assert result["isError"] is False
        assert "No callers found" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_with_workspace_id(self, get_callers_tool, mock_db_client):
        """Test execution with explicit workspace_id."""
        mock_db_client.get_callers.return_value = []

        await get_callers_tool.execute(
            {
                "qualified_name": "module.func",
                "workspace_id": "my-workspace",
            }
        )

        mock_db_client.get_callers.assert_called_once_with(
            qualified_name="module.func",
            workspace_id="my-workspace",
            limit=50,
        )

    @pytest.mark.asyncio
    async def test_execute_validation_error_empty_name(self, get_callers_tool):
        """Test validation error for empty qualified_name."""
        result = await get_callers_tool.execute(
            {
                "qualified_name": "",
            }
        )

        assert result["isError"] is True
        assert "Error:" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_validation_error_invalid_limit(self, get_callers_tool):
        """Test validation error for invalid limit."""
        result = await get_callers_tool.execute(
            {
                "qualified_name": "module.func",
                "limit": 0,  # Invalid: min is 1
            }
        )

        assert result["isError"] is True
        assert "Error:" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_validation_error_limit_too_high(self, get_callers_tool):
        """Test validation error for limit exceeding maximum."""
        result = await get_callers_tool.execute(
            {
                "qualified_name": "module.func",
                "limit": 200,  # Invalid: max is 100
            }
        )

        assert result["isError"] is True
        assert "Error:" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_database_error(self, get_callers_tool, mock_db_client):
        """Test database error handling."""
        mock_db_client.get_callers.side_effect = DatabaseError("Connection lost")

        result = await get_callers_tool.execute(
            {
                "qualified_name": "module.func",
            }
        )

        assert result["isError"] is True
        assert "Database error" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_unexpected_error(self, get_callers_tool, mock_db_client):
        """Test unexpected error handling."""
        mock_db_client.get_callers.side_effect = RuntimeError("Unexpected!")

        result = await get_callers_tool.execute(
            {
                "qualified_name": "module.func",
            }
        )

        assert result["isError"] is True
        assert "Unexpected error" in result["content"][0]["text"]


# =============================================================================
# GetCalleesTool Tests
# =============================================================================


class TestGetCalleesToolInit:
    """Tests for GetCalleesTool initialization."""

    def test_init_success(self, mock_db_client):
        """Test successful initialization."""
        tool = GetCalleesTool(db_client=mock_db_client)
        assert tool.db_client == mock_db_client
        assert tool.name == "get_callees"

    def test_init_wrong_type_raises(self):
        """Test initialization with wrong type raises TypeError."""
        with pytest.raises(TypeError, match="must be FalkorDBClient instance"):
            GetCalleesTool(db_client="not_a_client")

    def test_tool_attributes(self, mock_db_client):
        """Test tool has required attributes."""
        tool = GetCalleesTool(db_client=mock_db_client)
        assert tool.name == "get_callees"
        assert "Find all functions called by" in tool.description
        assert "qualified_name" in tool.input_schema["properties"]
        assert "limit" in tool.input_schema["properties"]
        assert "workspace_id" in tool.input_schema["properties"]


class TestGetCalleesToolExecute:
    """Tests for GetCalleesTool.execute()."""

    @pytest.mark.asyncio
    async def test_execute_success_with_results(self, get_callees_tool, mock_db_client):
        """Test successful execution with callees found."""
        mock_db_client.get_callees.return_value = [
            {
                "callee_qualified_name": "module.helper_func",
                "callee_id": "callee-uuid-1",
                "callee_file_path": "/path/to/helper.py",
                "call_line": 10,
                "call_type": "function",
                "arguments_count": 2,
                "call_count": 1,
            },
            {
                "callee_qualified_name": "module.logger.info",
                "callee_id": "callee-uuid-2",
                "callee_file_path": "/path/to/logger.py",
                "call_line": 25,
                "call_type": "method",
                "arguments_count": 1,
                "call_count": 5,
            },
        ]

        result = await get_callees_tool.execute(
            {
                "qualified_name": "module.MyClass.process",
                "limit": 20,
            }
        )

        assert result["isError"] is False
        assert len(result["content"]) == 1
        assert "calls 2 function(s)" in result["content"][0]["text"]
        assert "module.helper_func" in result["content"][0]["text"]
        assert "module.logger.info" in result["content"][0]["text"]
        assert "Called at line: 10" in result["content"][0]["text"]
        assert "Call count: 5" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_success_no_results(self, get_callees_tool, mock_db_client):
        """Test successful execution with no callees found."""
        mock_db_client.get_callees.return_value = []

        result = await get_callees_tool.execute(
            {
                "qualified_name": "module.simple_func",
            }
        )

        assert result["isError"] is False
        assert "No callees found" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_with_workspace_id(self, get_callees_tool, mock_db_client):
        """Test execution with explicit workspace_id."""
        mock_db_client.get_callees.return_value = []

        await get_callees_tool.execute(
            {
                "qualified_name": "module.func",
                "workspace_id": "my-workspace",
            }
        )

        mock_db_client.get_callees.assert_called_once_with(
            qualified_name="module.func",
            workspace_id="my-workspace",
            limit=50,
        )

    @pytest.mark.asyncio
    async def test_execute_validation_error_empty_name(self, get_callees_tool):
        """Test validation error for empty qualified_name."""
        result = await get_callees_tool.execute(
            {
                "qualified_name": "",
            }
        )

        assert result["isError"] is True
        assert "Error:" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_validation_error_invalid_limit(self, get_callees_tool):
        """Test validation error for invalid limit."""
        result = await get_callees_tool.execute(
            {
                "qualified_name": "module.func",
                "limit": -1,  # Invalid: min is 1
            }
        )

        assert result["isError"] is True
        assert "Error:" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_database_error(self, get_callees_tool, mock_db_client):
        """Test database error handling."""
        mock_db_client.get_callees.side_effect = DatabaseError("Query failed")

        result = await get_callees_tool.execute(
            {
                "qualified_name": "module.func",
            }
        )

        assert result["isError"] is True
        assert "Database error" in result["content"][0]["text"]


# =============================================================================
# Request Model Tests
# =============================================================================


class TestGetCallersRequest:
    """Tests for GetCallersRequest Pydantic model."""

    def test_valid_request(self):
        """Test valid request creation."""
        request = GetCallersRequest(
            qualified_name="module.func",
            limit=25,
            workspace_id="my-workspace",
        )
        assert request.qualified_name == "module.func"
        assert request.limit == 25
        assert request.workspace_id == "my-workspace"

    def test_default_values(self):
        """Test default values are applied."""
        request = GetCallersRequest(qualified_name="module.func")
        assert request.limit == 50
        assert request.workspace_id == ""

    def test_empty_name_raises(self):
        """Test empty qualified_name raises validation error."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            GetCallersRequest(qualified_name="")

    def test_extra_fields_forbidden(self):
        """Test extra fields are forbidden."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            GetCallersRequest(
                qualified_name="module.func",
                extra_field="value",
            )


class TestGetCalleesRequest:
    """Tests for GetCalleesRequest Pydantic model."""

    def test_valid_request(self):
        """Test valid request creation."""
        request = GetCalleesRequest(
            qualified_name="module.MyClass.method",
            limit=30,
            workspace_id="workspace-1",
        )
        assert request.qualified_name == "module.MyClass.method"
        assert request.limit == 30
        assert request.workspace_id == "workspace-1"

    def test_default_values(self):
        """Test default values are applied."""
        request = GetCalleesRequest(qualified_name="module.func")
        assert request.limit == 50
        assert request.workspace_id == ""

    def test_limit_bounds(self):
        """Test limit bounds validation."""
        # Valid bounds
        GetCalleesRequest(qualified_name="func", limit=1)
        GetCalleesRequest(qualified_name="func", limit=100)

        # Invalid: below minimum
        with pytest.raises(Exception):
            GetCalleesRequest(qualified_name="func", limit=0)

        # Invalid: above maximum
        with pytest.raises(Exception):
            GetCalleesRequest(qualified_name="func", limit=101)


# =============================================================================
# Tool Registration Tests
# =============================================================================


class TestToolRegistration:
    """Tests for tool registration with MCPServer."""

    def test_tools_have_required_attributes(self, mock_db_client):
        """Test tools have all attributes required for MCP registration."""
        callers_tool = GetCallersTool(db_client=mock_db_client)
        callees_tool = GetCalleesTool(db_client=mock_db_client)

        for tool in [callers_tool, callees_tool]:
            # Required attributes
            assert hasattr(tool, "name")
            assert hasattr(tool, "description")
            assert hasattr(tool, "input_schema")
            assert hasattr(tool, "execute")

            # name should be lowercase with underscores
            assert tool.name.islower() or "_" in tool.name
            assert len(tool.name) > 0

            # description should be non-empty
            assert len(tool.description) > 0

            # input_schema should be valid JSON Schema
            assert "type" in tool.input_schema
            assert tool.input_schema["type"] == "object"
            assert "properties" in tool.input_schema

            # execute should be callable
            assert callable(tool.execute)

    def test_tool_names_unique(self, mock_db_client):
        """Test tool names are unique."""
        callers_tool = GetCallersTool(db_client=mock_db_client)
        callees_tool = GetCalleesTool(db_client=mock_db_client)

        assert callers_tool.name != callees_tool.name
        assert callers_tool.name == "get_callers"
        assert callees_tool.name == "get_callees"
