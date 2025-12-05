"""
Unit tests for FalkorDBClient bi-temporal methods.

Tests for:
- get_memory_at_time()
- get_memory_history()
- get_changes()
- close_version()
- create_new_version()
- soft_delete_memory()
"""

import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from zapomni_db.exceptions import DatabaseError, ValidationError
from zapomni_db.falkordb_client import FalkorDBClient
from zapomni_db.models import MemoryVersion, QueryResult


@pytest.fixture
def mock_query_result() -> MagicMock:
    """Create a mock QueryResult with default values."""
    result = MagicMock(spec=QueryResult)
    result.rows = []
    result.row_count = 0
    result.execution_time_ms = 1
    return result


@pytest.fixture
def sample_memory_row() -> Dict[str, Any]:
    """Create a sample database row for a memory."""
    now = datetime.now(timezone.utc).isoformat()
    return {
        "memory_id": str(uuid.uuid4()),
        "id": str(uuid.uuid4()),
        "text": "def hello(): pass",
        "file_path": "/project/src/main.py",
        "qualified_name": "main.hello",
        "workspace_id": "default",
        "version": 1,
        "source": "code_indexer",
        "tags": ["python", "function"],
        "metadata": '{"language": "python"}',
        "created_at": now,
        "transaction_to": None,
        "valid_from": now,
        "valid_to": None,
        "is_current": True,
        "stale": False,
        "last_seen_at": now,
        "previous_version_id": None,
    }


@pytest.fixture
def mock_falkordb_client(mocker) -> FalkorDBClient:
    """Create FalkorDBClient with mocked connection and query execution."""
    client = FalkorDBClient(host="localhost", port=6379)

    # Mock the _execute_cypher method
    mock_execute = AsyncMock()
    mocker.patch.object(client, "_execute_cypher", mock_execute)

    # Mark as initialized
    mocker.patch.object(client, "_initialized", True)
    mocker.patch.object(client, "_schema_ready", True)

    return client


# ============================================================================
# get_memory_at_time TESTS
# ============================================================================


class TestGetMemoryAtTime:
    """Tests for get_memory_at_time()."""

    @pytest.mark.asyncio
    async def test_valid_time_query_returns_memory(
        self, mock_falkordb_client: FalkorDBClient, sample_memory_row: Dict[str, Any], mocker
    ) -> None:
        """Test that valid time query returns a MemoryVersion when found."""
        # Setup mock result
        mock_result = MagicMock()
        mock_result.row_count = 1
        mock_result.rows = [sample_memory_row]
        mock_falkordb_client._execute_cypher.return_value = mock_result

        # Call method
        memory = await mock_falkordb_client.get_memory_at_time(
            workspace_id="default",
            file_path="/project/src/main.py",
            as_of=datetime(2025, 11, 15, tzinfo=timezone.utc),
            time_type="valid",
        )

        # Assertions
        assert memory is not None
        assert isinstance(memory, MemoryVersion)
        assert memory.text == "def hello(): pass"
        assert memory.file_path == "/project/src/main.py"

    @pytest.mark.asyncio
    async def test_transaction_time_query_returns_memory(
        self, mock_falkordb_client: FalkorDBClient, sample_memory_row: Dict[str, Any]
    ) -> None:
        """Test that transaction time query returns a MemoryVersion when found."""
        mock_result = MagicMock()
        mock_result.row_count = 1
        mock_result.rows = [sample_memory_row]
        mock_falkordb_client._execute_cypher.return_value = mock_result

        memory = await mock_falkordb_client.get_memory_at_time(
            workspace_id="default",
            file_path="/project/src/main.py",
            as_of=datetime(2025, 11, 15, tzinfo=timezone.utc),
            time_type="transaction",
        )

        assert memory is not None
        assert isinstance(memory, MemoryVersion)

    @pytest.mark.asyncio
    async def test_both_time_query_returns_memory(
        self, mock_falkordb_client: FalkorDBClient, sample_memory_row: Dict[str, Any]
    ) -> None:
        """Test that bi-temporal query returns a MemoryVersion when found."""
        mock_result = MagicMock()
        mock_result.row_count = 1
        mock_result.rows = [sample_memory_row]
        mock_falkordb_client._execute_cypher.return_value = mock_result

        memory = await mock_falkordb_client.get_memory_at_time(
            workspace_id="default",
            file_path="/project/src/main.py",
            as_of=datetime(2025, 11, 15, tzinfo=timezone.utc),
            time_type="both",
        )

        assert memory is not None
        assert isinstance(memory, MemoryVersion)

    @pytest.mark.asyncio
    async def test_no_match_returns_none(self, mock_falkordb_client: FalkorDBClient) -> None:
        """Test that query returns None when no memory found at that time."""
        mock_result = MagicMock()
        mock_result.row_count = 0
        mock_result.rows = []
        mock_falkordb_client._execute_cypher.return_value = mock_result

        memory = await mock_falkordb_client.get_memory_at_time(
            workspace_id="default",
            file_path="/project/src/nonexistent.py",
            as_of=datetime(2025, 11, 15, tzinfo=timezone.utc),
        )

        assert memory is None

    @pytest.mark.asyncio
    async def test_converts_datetime_to_iso(
        self, mock_falkordb_client: FalkorDBClient, sample_memory_row: Dict[str, Any]
    ) -> None:
        """Test that datetime is converted to ISO 8601 string for query."""
        mock_result = MagicMock()
        mock_result.row_count = 1
        mock_result.rows = [sample_memory_row]
        mock_falkordb_client._execute_cypher.return_value = mock_result

        as_of = datetime(2025, 11, 15, 10, 30, 0, tzinfo=timezone.utc)
        await mock_falkordb_client.get_memory_at_time(
            workspace_id="default",
            file_path="/project/src/main.py",
            as_of=as_of,
        )

        # Verify _execute_cypher was called with ISO formatted timestamp in params
        call_args = mock_falkordb_client._execute_cypher.call_args
        params = call_args[0][1]
        assert "as_of_valid" in params
        assert "2025-11-15" in params["as_of_valid"]

    @pytest.mark.asyncio
    async def test_empty_workspace_id_raises_error(
        self, mock_falkordb_client: FalkorDBClient
    ) -> None:
        """Test that empty workspace_id raises ValidationError."""
        with pytest.raises(ValidationError, match="workspace_id cannot be empty"):
            await mock_falkordb_client.get_memory_at_time(
                workspace_id="",
                file_path="/project/src/main.py",
                as_of=datetime.now(timezone.utc),
            )

    @pytest.mark.asyncio
    async def test_empty_file_path_raises_error(
        self, mock_falkordb_client: FalkorDBClient
    ) -> None:
        """Test that empty file_path raises ValidationError."""
        with pytest.raises(ValidationError, match="file_path cannot be empty"):
            await mock_falkordb_client.get_memory_at_time(
                workspace_id="default",
                file_path="",
                as_of=datetime.now(timezone.utc),
            )

    @pytest.mark.asyncio
    async def test_database_error_wrapped(self, mock_falkordb_client: FalkorDBClient) -> None:
        """Test that database errors are wrapped in DatabaseError."""
        mock_falkordb_client._execute_cypher.side_effect = RuntimeError("Connection lost")

        with pytest.raises(DatabaseError, match="Failed to get memory at time"):
            await mock_falkordb_client.get_memory_at_time(
                workspace_id="default",
                file_path="/project/src/main.py",
                as_of=datetime.now(timezone.utc),
            )


# ============================================================================
# get_memory_history TESTS
# ============================================================================


class TestGetMemoryHistory:
    """Tests for get_memory_history()."""

    @pytest.mark.asyncio
    async def test_returns_all_versions(
        self, mock_falkordb_client: FalkorDBClient, sample_memory_row: Dict[str, Any]
    ) -> None:
        """Test that history returns all versions."""
        # Create multiple version rows
        row1 = sample_memory_row.copy()
        row1["version"] = 3
        row1["is_current"] = True
        row2 = sample_memory_row.copy()
        row2["version"] = 2
        row2["is_current"] = False
        row3 = sample_memory_row.copy()
        row3["version"] = 1
        row3["is_current"] = False

        mock_result = MagicMock()
        mock_result.row_count = 3
        mock_result.rows = [row1, row2, row3]
        mock_falkordb_client._execute_cypher.return_value = mock_result

        history = await mock_falkordb_client.get_memory_history(
            workspace_id="default",
            file_path="/project/src/main.py",
        )

        assert len(history) == 3
        assert all(isinstance(m, MemoryVersion) for m in history)

    @pytest.mark.asyncio
    async def test_ordered_by_valid_from_desc(
        self, mock_falkordb_client: FalkorDBClient, sample_memory_row: Dict[str, Any]
    ) -> None:
        """Test that history is ordered by valid_from DESC."""
        # The mock returns in the order we set, so we verify the query has ORDER BY
        mock_result = MagicMock()
        mock_result.row_count = 1
        mock_result.rows = [sample_memory_row]
        mock_falkordb_client._execute_cypher.return_value = mock_result

        await mock_falkordb_client.get_memory_history(
            workspace_id="default",
            file_path="/project/src/main.py",
        )

        # Verify the cypher query contains ORDER BY valid_from DESC
        call_args = mock_falkordb_client._execute_cypher.call_args
        cypher = call_args[0][0]
        assert "ORDER BY" in cypher
        assert "valid_from DESC" in cypher

    @pytest.mark.asyncio
    async def test_with_limit(
        self, mock_falkordb_client: FalkorDBClient, sample_memory_row: Dict[str, Any]
    ) -> None:
        """Test that limit is passed to query."""
        mock_result = MagicMock()
        mock_result.row_count = 1
        mock_result.rows = [sample_memory_row]
        mock_falkordb_client._execute_cypher.return_value = mock_result

        await mock_falkordb_client.get_memory_history(
            workspace_id="default",
            file_path="/project/src/main.py",
            limit=25,
        )

        call_args = mock_falkordb_client._execute_cypher.call_args
        params = call_args[0][1]
        assert params["limit"] == 25

    @pytest.mark.asyncio
    async def test_empty_history_returns_empty_list(
        self, mock_falkordb_client: FalkorDBClient
    ) -> None:
        """Test that empty history returns empty list."""
        mock_result = MagicMock()
        mock_result.row_count = 0
        mock_result.rows = []
        mock_falkordb_client._execute_cypher.return_value = mock_result

        history = await mock_falkordb_client.get_memory_history(
            workspace_id="default",
            file_path="/project/src/nonexistent.py",
        )

        assert history == []

    @pytest.mark.asyncio
    async def test_file_path_filter(
        self, mock_falkordb_client: FalkorDBClient, sample_memory_row: Dict[str, Any]
    ) -> None:
        """Test that file_path filter is applied."""
        mock_result = MagicMock()
        mock_result.row_count = 1
        mock_result.rows = [sample_memory_row]
        mock_falkordb_client._execute_cypher.return_value = mock_result

        await mock_falkordb_client.get_memory_history(
            workspace_id="default",
            file_path="/project/src/main.py",
        )

        call_args = mock_falkordb_client._execute_cypher.call_args
        params = call_args[0][1]
        assert params["file_path"] == "/project/src/main.py"

    @pytest.mark.asyncio
    async def test_entity_id_filter(
        self, mock_falkordb_client: FalkorDBClient, sample_memory_row: Dict[str, Any]
    ) -> None:
        """Test that entity_id filter is applied."""
        entity_id = "550e8400-e29b-41d4-a716-446655440000"
        mock_result = MagicMock()
        mock_result.row_count = 1
        mock_result.rows = [sample_memory_row]
        mock_falkordb_client._execute_cypher.return_value = mock_result

        await mock_falkordb_client.get_memory_history(
            workspace_id="default",
            entity_id=entity_id,
        )

        call_args = mock_falkordb_client._execute_cypher.call_args
        params = call_args[0][1]
        assert params["entity_id"] == entity_id

    @pytest.mark.asyncio
    async def test_requires_file_path_or_entity_id(
        self, mock_falkordb_client: FalkorDBClient
    ) -> None:
        """Test that either file_path or entity_id is required."""
        with pytest.raises(ValidationError, match="Either file_path or entity_id"):
            await mock_falkordb_client.get_memory_history(workspace_id="default")

    @pytest.mark.asyncio
    async def test_database_error_wrapped(self, mock_falkordb_client: FalkorDBClient) -> None:
        """Test that database errors are wrapped in DatabaseError."""
        mock_falkordb_client._execute_cypher.side_effect = RuntimeError("Connection lost")

        with pytest.raises(DatabaseError, match="Failed to get memory history"):
            await mock_falkordb_client.get_memory_history(
                workspace_id="default",
                file_path="/project/src/main.py",
            )


# ============================================================================
# get_changes TESTS
# ============================================================================


class TestGetChanges:
    """Tests for get_changes()."""

    @pytest.mark.asyncio
    async def test_returns_changes_in_range(self, mock_falkordb_client: FalkorDBClient) -> None:
        """Test that changes are returned in the time range."""
        mock_result = MagicMock()
        mock_result.row_count = 2
        mock_result.rows = [
            {
                "memory_id": str(uuid.uuid4()),
                "file_path": "/src/a.py",
                "version": 1,
                "created_at": "2025-11-15T10:00:00Z",
                "change_type": "created",
                "qualified_name": "a.main",
                "valid_from": "2025-11-15T10:00:00Z",
                "valid_to": None,
                "transaction_to": None,
            },
            {
                "memory_id": str(uuid.uuid4()),
                "file_path": "/src/b.py",
                "version": 2,
                "created_at": "2025-11-15T11:00:00Z",
                "change_type": "modified",
                "qualified_name": "b.func",
                "valid_from": "2025-11-15T11:00:00Z",
                "valid_to": None,
                "transaction_to": None,
            },
        ]
        mock_falkordb_client._execute_cypher.return_value = mock_result

        since = datetime(2025, 11, 15, tzinfo=timezone.utc)
        changes = await mock_falkordb_client.get_changes(
            workspace_id="default",
            since=since,
        )

        assert len(changes) == 2
        assert changes[0]["change_type"] == "created"
        assert changes[1]["change_type"] == "modified"

    @pytest.mark.asyncio
    async def test_filter_by_change_type(self, mock_falkordb_client: FalkorDBClient) -> None:
        """Test filtering by change_type."""
        mock_result = MagicMock()
        mock_result.row_count = 1
        mock_result.rows = [
            {
                "memory_id": str(uuid.uuid4()),
                "file_path": "/src/new.py",
                "version": 1,
                "created_at": "2025-11-15T10:00:00Z",
                "change_type": "created",
                "qualified_name": None,
                "valid_from": "2025-11-15T10:00:00Z",
                "valid_to": None,
                "transaction_to": None,
            },
        ]
        mock_falkordb_client._execute_cypher.return_value = mock_result

        changes = await mock_falkordb_client.get_changes(
            workspace_id="default",
            since=datetime(2025, 11, 1, tzinfo=timezone.utc),
            change_type="created",
        )

        # Verify cypher query includes version = 1 filter
        call_args = mock_falkordb_client._execute_cypher.call_args
        cypher = call_args[0][0]
        assert "version = 1" in cypher

    @pytest.mark.asyncio
    async def test_filter_by_path_pattern(self, mock_falkordb_client: FalkorDBClient) -> None:
        """Test filtering by path_pattern."""
        mock_result = MagicMock()
        mock_result.row_count = 1
        mock_result.rows = [
            {
                "memory_id": str(uuid.uuid4()),
                "file_path": "/project/src/utils.py",
                "version": 1,
                "created_at": "2025-11-15T10:00:00Z",
                "change_type": "created",
                "qualified_name": None,
                "valid_from": "2025-11-15T10:00:00Z",
                "valid_to": None,
                "transaction_to": None,
            },
        ]
        mock_falkordb_client._execute_cypher.return_value = mock_result

        await mock_falkordb_client.get_changes(
            workspace_id="default",
            since=datetime(2025, 11, 1, tzinfo=timezone.utc),
            path_pattern="/project/src/",
        )

        call_args = mock_falkordb_client._execute_cypher.call_args
        params = call_args[0][1]
        assert params["path_pattern"] == "/project/src/"

    @pytest.mark.asyncio
    async def test_respects_limit(self, mock_falkordb_client: FalkorDBClient) -> None:
        """Test that limit is respected."""
        mock_result = MagicMock()
        mock_result.row_count = 0
        mock_result.rows = []
        mock_falkordb_client._execute_cypher.return_value = mock_result

        await mock_falkordb_client.get_changes(
            workspace_id="default",
            since=datetime(2025, 11, 1, tzinfo=timezone.utc),
            limit=50,
        )

        call_args = mock_falkordb_client._execute_cypher.call_args
        params = call_args[0][1]
        assert params["limit"] == 50

    @pytest.mark.asyncio
    async def test_no_changes_returns_empty_list(
        self, mock_falkordb_client: FalkorDBClient
    ) -> None:
        """Test that no changes returns empty list."""
        mock_result = MagicMock()
        mock_result.row_count = 0
        mock_result.rows = []
        mock_falkordb_client._execute_cypher.return_value = mock_result

        changes = await mock_falkordb_client.get_changes(
            workspace_id="default",
            since=datetime(2025, 12, 1, tzinfo=timezone.utc),
        )

        assert changes == []

    @pytest.mark.asyncio
    async def test_with_until_parameter(self, mock_falkordb_client: FalkorDBClient) -> None:
        """Test that until parameter is passed to query."""
        mock_result = MagicMock()
        mock_result.row_count = 0
        mock_result.rows = []
        mock_falkordb_client._execute_cypher.return_value = mock_result

        since = datetime(2025, 11, 1, tzinfo=timezone.utc)
        until = datetime(2025, 11, 30, tzinfo=timezone.utc)

        await mock_falkordb_client.get_changes(
            workspace_id="default",
            since=since,
            until=until,
        )

        call_args = mock_falkordb_client._execute_cypher.call_args
        params = call_args[0][1]
        assert "until" in params

    @pytest.mark.asyncio
    async def test_database_error_wrapped(self, mock_falkordb_client: FalkorDBClient) -> None:
        """Test that database errors are wrapped in DatabaseError."""
        mock_falkordb_client._execute_cypher.side_effect = RuntimeError("Connection lost")

        with pytest.raises(DatabaseError, match="Failed to get changes"):
            await mock_falkordb_client.get_changes(
                workspace_id="default",
                since=datetime(2025, 11, 1, tzinfo=timezone.utc),
            )


# ============================================================================
# close_version TESTS
# ============================================================================


class TestCloseVersion:
    """Tests for close_version()."""

    @pytest.mark.asyncio
    async def test_closes_existing_version(self, mock_falkordb_client: FalkorDBClient) -> None:
        """Test that existing version is closed successfully."""
        mock_result = MagicMock()
        mock_result.row_count = 1
        mock_result.rows = [{"memory_id": "uuid-123", "is_current": False}]
        mock_falkordb_client._execute_cypher.return_value = mock_result

        memory_id = "550e8400-e29b-41d4-a716-446655440000"
        result = await mock_falkordb_client.close_version(memory_id=memory_id)

        assert result is True

    @pytest.mark.asyncio
    async def test_sets_valid_to(self, mock_falkordb_client: FalkorDBClient) -> None:
        """Test that valid_to is set in the query."""
        mock_result = MagicMock()
        mock_result.row_count = 1
        mock_result.rows = [{"memory_id": "uuid-123", "is_current": False}]
        mock_falkordb_client._execute_cypher.return_value = mock_result

        valid_to = datetime(2025, 12, 5, 10, 30, tzinfo=timezone.utc)
        await mock_falkordb_client.close_version(
            memory_id="550e8400-e29b-41d4-a716-446655440000",
            valid_to=valid_to,
        )

        call_args = mock_falkordb_client._execute_cypher.call_args
        params = call_args[0][1]
        assert "valid_to" in params
        assert "2025-12-05" in params["valid_to"]

    @pytest.mark.asyncio
    async def test_sets_transaction_to(self, mock_falkordb_client: FalkorDBClient) -> None:
        """Test that transaction_to is set in the query."""
        mock_result = MagicMock()
        mock_result.row_count = 1
        mock_result.rows = [{"memory_id": "uuid-123", "is_current": False}]
        mock_falkordb_client._execute_cypher.return_value = mock_result

        transaction_to = datetime(2025, 12, 5, 10, 30, tzinfo=timezone.utc)
        await mock_falkordb_client.close_version(
            memory_id="550e8400-e29b-41d4-a716-446655440000",
            transaction_to=transaction_to,
        )

        call_args = mock_falkordb_client._execute_cypher.call_args
        params = call_args[0][1]
        assert "transaction_to" in params
        assert "2025-12-05" in params["transaction_to"]

    @pytest.mark.asyncio
    async def test_sets_is_current_false(self, mock_falkordb_client: FalkorDBClient) -> None:
        """Test that is_current is set to false in the query."""
        mock_result = MagicMock()
        mock_result.row_count = 1
        mock_result.rows = [{"memory_id": "uuid-123", "is_current": False}]
        mock_falkordb_client._execute_cypher.return_value = mock_result

        await mock_falkordb_client.close_version(
            memory_id="550e8400-e29b-41d4-a716-446655440000",
        )

        call_args = mock_falkordb_client._execute_cypher.call_args
        cypher = call_args[0][0]
        assert "is_current = false" in cypher

    @pytest.mark.asyncio
    async def test_nonexistent_version_returns_false(
        self, mock_falkordb_client: FalkorDBClient
    ) -> None:
        """Test that closing nonexistent version returns False."""
        mock_result = MagicMock()
        mock_result.row_count = 0
        mock_result.rows = []
        mock_falkordb_client._execute_cypher.return_value = mock_result

        result = await mock_falkordb_client.close_version(
            memory_id="550e8400-e29b-41d4-a716-446655440000",
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_invalid_uuid_raises_error(self, mock_falkordb_client: FalkorDBClient) -> None:
        """Test that invalid UUID raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid memory UUID"):
            await mock_falkordb_client.close_version(memory_id="not-a-valid-uuid")

    @pytest.mark.asyncio
    async def test_defaults_to_now(self, mock_falkordb_client: FalkorDBClient) -> None:
        """Test that valid_to and transaction_to default to now."""
        mock_result = MagicMock()
        mock_result.row_count = 1
        mock_result.rows = [{"memory_id": "uuid-123", "is_current": False}]
        mock_falkordb_client._execute_cypher.return_value = mock_result

        await mock_falkordb_client.close_version(
            memory_id="550e8400-e29b-41d4-a716-446655440000",
        )

        call_args = mock_falkordb_client._execute_cypher.call_args
        params = call_args[0][1]
        # Both should be set (defaulted to now)
        assert "valid_to" in params
        assert "transaction_to" in params
        assert params["valid_to"] is not None
        assert params["transaction_to"] is not None


# ============================================================================
# create_new_version TESTS
# ============================================================================


class TestCreateNewVersion:
    """Tests for create_new_version()."""

    @pytest.mark.asyncio
    async def test_creates_new_version(
        self, mock_falkordb_client: FalkorDBClient, sample_memory_row: Dict[str, Any]
    ) -> None:
        """Test that new version is created successfully."""
        # First call returns previous memory
        prev_result = MagicMock()
        prev_result.row_count = 1
        prev_result.rows = [
            {
                "id": "prev-uuid",
                "file_path": "/src/main.py",
                "qualified_name": "main.hello",
                "workspace_id": "default",
                "version": 1,
                "source": "code_indexer",
                "tags": ["python"],
                "metadata": "{}",
            }
        ]

        # Second call closes previous version
        close_result = MagicMock()
        close_result.row_count = 1
        close_result.rows = [{"memory_id": "prev-uuid", "is_current": False}]

        # Third call creates new version
        new_row = sample_memory_row.copy()
        new_row["version"] = 2
        new_row["previous_version_id"] = "prev-uuid"
        create_result = MagicMock()
        create_result.row_count = 1
        create_result.rows = [new_row]

        mock_falkordb_client._execute_cypher.side_effect = [
            prev_result,
            close_result,
            create_result,
        ]

        new_memory = await mock_falkordb_client.create_new_version(
            previous_memory_id="550e8400-e29b-41d4-a716-446655440000",
            new_text="def hello_updated(): pass",
        )

        assert new_memory is not None
        assert isinstance(new_memory, MemoryVersion)

    @pytest.mark.asyncio
    async def test_increments_version_number(
        self, mock_falkordb_client: FalkorDBClient, sample_memory_row: Dict[str, Any]
    ) -> None:
        """Test that version number is incremented."""
        prev_result = MagicMock()
        prev_result.row_count = 1
        prev_result.rows = [
            {
                "id": "prev-uuid",
                "file_path": "/src/main.py",
                "qualified_name": "main.hello",
                "workspace_id": "default",
                "version": 5,  # Previous version
                "source": "code_indexer",
                "tags": [],
                "metadata": "{}",
            }
        ]

        close_result = MagicMock()
        close_result.row_count = 1
        close_result.rows = [{"memory_id": "prev-uuid", "is_current": False}]

        new_row = sample_memory_row.copy()
        new_row["version"] = 6  # New version should be prev + 1
        create_result = MagicMock()
        create_result.row_count = 1
        create_result.rows = [new_row]

        mock_falkordb_client._execute_cypher.side_effect = [
            prev_result,
            close_result,
            create_result,
        ]

        new_memory = await mock_falkordb_client.create_new_version(
            previous_memory_id="550e8400-e29b-41d4-a716-446655440000",
            new_text="updated text",
        )

        assert new_memory.version == 6

    @pytest.mark.asyncio
    async def test_sets_previous_version_id(
        self, mock_falkordb_client: FalkorDBClient, sample_memory_row: Dict[str, Any]
    ) -> None:
        """Test that previous_version_id is set on new version."""
        prev_id = "550e8400-e29b-41d4-a716-446655440000"

        prev_result = MagicMock()
        prev_result.row_count = 1
        prev_result.rows = [
            {
                "id": prev_id,
                "file_path": "/src/main.py",
                "qualified_name": None,
                "workspace_id": "default",
                "version": 1,
                "source": "code_indexer",
                "tags": [],
                "metadata": "{}",
            }
        ]

        close_result = MagicMock()
        close_result.row_count = 1
        close_result.rows = [{"memory_id": prev_id, "is_current": False}]

        new_row = sample_memory_row.copy()
        new_row["version"] = 2
        new_row["previous_version_id"] = prev_id
        create_result = MagicMock()
        create_result.row_count = 1
        create_result.rows = [new_row]

        mock_falkordb_client._execute_cypher.side_effect = [
            prev_result,
            close_result,
            create_result,
        ]

        new_memory = await mock_falkordb_client.create_new_version(
            previous_memory_id=prev_id,
            new_text="updated text",
        )

        assert new_memory.previous_version_id == prev_id

    @pytest.mark.asyncio
    async def test_closes_previous_version(
        self, mock_falkordb_client: FalkorDBClient, sample_memory_row: Dict[str, Any]
    ) -> None:
        """Test that previous version is closed."""
        prev_result = MagicMock()
        prev_result.row_count = 1
        prev_result.rows = [
            {
                "id": "prev-uuid",
                "file_path": "/src/main.py",
                "qualified_name": None,
                "workspace_id": "default",
                "version": 1,
                "source": "code_indexer",
                "tags": [],
                "metadata": "{}",
            }
        ]

        close_result = MagicMock()
        close_result.row_count = 1
        close_result.rows = [{"memory_id": "prev-uuid", "is_current": False}]

        create_result = MagicMock()
        create_result.row_count = 1
        create_result.rows = [sample_memory_row]

        mock_falkordb_client._execute_cypher.side_effect = [
            prev_result,
            close_result,
            create_result,
        ]

        await mock_falkordb_client.create_new_version(
            previous_memory_id="550e8400-e29b-41d4-a716-446655440000",
            new_text="updated text",
        )

        # Verify close_version was called (second call to _execute_cypher)
        assert mock_falkordb_client._execute_cypher.call_count >= 2
        # The close query should set is_current = false
        second_call = mock_falkordb_client._execute_cypher.call_args_list[1]
        cypher = second_call[0][0]
        assert "is_current = false" in cypher

    @pytest.mark.asyncio
    async def test_new_version_is_current(
        self, mock_falkordb_client: FalkorDBClient, sample_memory_row: Dict[str, Any]
    ) -> None:
        """Test that new version has is_current = True."""
        prev_result = MagicMock()
        prev_result.row_count = 1
        prev_result.rows = [
            {
                "id": "prev-uuid",
                "file_path": "/src/main.py",
                "qualified_name": None,
                "workspace_id": "default",
                "version": 1,
                "source": "code_indexer",
                "tags": [],
                "metadata": "{}",
            }
        ]

        close_result = MagicMock()
        close_result.row_count = 1
        close_result.rows = [{"memory_id": "prev-uuid", "is_current": False}]

        new_row = sample_memory_row.copy()
        new_row["is_current"] = True
        create_result = MagicMock()
        create_result.row_count = 1
        create_result.rows = [new_row]

        mock_falkordb_client._execute_cypher.side_effect = [
            prev_result,
            close_result,
            create_result,
        ]

        new_memory = await mock_falkordb_client.create_new_version(
            previous_memory_id="550e8400-e29b-41d4-a716-446655440000",
            new_text="updated text",
        )

        assert new_memory.is_current is True

    @pytest.mark.asyncio
    async def test_nonexistent_previous_returns_none(
        self, mock_falkordb_client: FalkorDBClient
    ) -> None:
        """Test that nonexistent previous memory returns None."""
        mock_result = MagicMock()
        mock_result.row_count = 0
        mock_result.rows = []
        mock_falkordb_client._execute_cypher.return_value = mock_result

        result = await mock_falkordb_client.create_new_version(
            previous_memory_id="550e8400-e29b-41d4-a716-446655440000",
            new_text="new text",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_invalid_uuid_raises_error(self, mock_falkordb_client: FalkorDBClient) -> None:
        """Test that invalid previous_memory_id raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid previous_memory_id UUID"):
            await mock_falkordb_client.create_new_version(
                previous_memory_id="not-a-uuid",
                new_text="new text",
            )

    @pytest.mark.asyncio
    async def test_empty_text_raises_error(self, mock_falkordb_client: FalkorDBClient) -> None:
        """Test that empty new_text raises ValidationError."""
        with pytest.raises(ValidationError, match="new_text cannot be empty"):
            await mock_falkordb_client.create_new_version(
                previous_memory_id="550e8400-e29b-41d4-a716-446655440000",
                new_text="",
            )


# ============================================================================
# soft_delete_memory TESTS
# ============================================================================


class TestSoftDeleteMemory:
    """Tests for soft_delete_memory()."""

    @pytest.mark.asyncio
    async def test_soft_delete_sets_transaction_to(
        self, mock_falkordb_client: FalkorDBClient
    ) -> None:
        """Test that soft delete sets transaction_to."""
        mock_result = MagicMock()
        mock_result.row_count = 1
        mock_result.rows = [{"deleted_count": 1}]
        mock_falkordb_client._execute_cypher.return_value = mock_result

        await mock_falkordb_client.soft_delete_memory(
            memory_id="550e8400-e29b-41d4-a716-446655440000",
        )

        call_args = mock_falkordb_client._execute_cypher.call_args
        cypher = call_args[0][0]
        params = call_args[0][1]
        assert "transaction_to = $transaction_to" in cypher
        assert "transaction_to" in params

    @pytest.mark.asyncio
    async def test_soft_delete_sets_is_current_false(
        self, mock_falkordb_client: FalkorDBClient
    ) -> None:
        """Test that soft delete sets is_current to false."""
        mock_result = MagicMock()
        mock_result.row_count = 1
        mock_result.rows = [{"deleted_count": 1}]
        mock_falkordb_client._execute_cypher.return_value = mock_result

        await mock_falkordb_client.soft_delete_memory(
            memory_id="550e8400-e29b-41d4-a716-446655440000",
        )

        call_args = mock_falkordb_client._execute_cypher.call_args
        cypher = call_args[0][0]
        assert "is_current = false" in cypher

    @pytest.mark.asyncio
    async def test_soft_delete_preserves_data(
        self, mock_falkordb_client: FalkorDBClient
    ) -> None:
        """Test that soft delete does not remove the node."""
        mock_result = MagicMock()
        mock_result.row_count = 1
        mock_result.rows = [{"deleted_count": 1}]
        mock_falkordb_client._execute_cypher.return_value = mock_result

        result = await mock_falkordb_client.soft_delete_memory(
            memory_id="550e8400-e29b-41d4-a716-446655440000",
        )

        # Verify we use SET not DELETE
        call_args = mock_falkordb_client._execute_cypher.call_args
        cypher = call_args[0][0]
        assert "SET" in cypher
        assert "DELETE" not in cypher.upper().split("RETURN")[0]  # No DELETE before RETURN
        assert result is True

    @pytest.mark.asyncio
    async def test_nonexistent_memory_returns_false(
        self, mock_falkordb_client: FalkorDBClient
    ) -> None:
        """Test that soft delete of nonexistent memory returns False."""
        mock_result = MagicMock()
        mock_result.row_count = 1
        mock_result.rows = [{"deleted_count": 0}]
        mock_falkordb_client._execute_cypher.return_value = mock_result

        result = await mock_falkordb_client.soft_delete_memory(
            memory_id="550e8400-e29b-41d4-a716-446655440000",
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_invalid_uuid_raises_error(self, mock_falkordb_client: FalkorDBClient) -> None:
        """Test that invalid memory_id raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid memory UUID"):
            await mock_falkordb_client.soft_delete_memory(memory_id="not-a-uuid")

    @pytest.mark.asyncio
    async def test_workspace_validation(
        self, mock_falkordb_client: FalkorDBClient
    ) -> None:
        """Test that workspace_id is validated if provided."""
        # First call checks workspace
        check_result = MagicMock()
        check_result.row_count = 1
        check_result.rows = [{"workspace_id": "other-workspace"}]

        mock_falkordb_client._execute_cypher.return_value = check_result

        with pytest.raises(ValidationError, match="Memory belongs to workspace"):
            await mock_falkordb_client.soft_delete_memory(
                memory_id="550e8400-e29b-41d4-a716-446655440000",
                workspace_id="default",
            )

    @pytest.mark.asyncio
    async def test_soft_delete_success_returns_true(
        self, mock_falkordb_client: FalkorDBClient
    ) -> None:
        """Test that successful soft delete returns True."""
        mock_result = MagicMock()
        mock_result.row_count = 1
        mock_result.rows = [{"deleted_count": 1}]
        mock_falkordb_client._execute_cypher.return_value = mock_result

        result = await mock_falkordb_client.soft_delete_memory(
            memory_id="550e8400-e29b-41d4-a716-446655440000",
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_database_error_wrapped(self, mock_falkordb_client: FalkorDBClient) -> None:
        """Test that database errors are wrapped in DatabaseError."""
        mock_falkordb_client._execute_cypher.side_effect = RuntimeError("Connection lost")

        with pytest.raises(DatabaseError, match="Failed to soft delete memory"):
            await mock_falkordb_client.soft_delete_memory(
                memory_id="550e8400-e29b-41d4-a716-446655440000",
            )
