"""
Unit tests for Garbage Collection functionality.

Tests for:
- mark_code_memories_stale
- mark_memory_fresh
- count_stale_memories
- get_stale_memories_preview
- delete_stale_memories
- get_orphaned_chunks_preview
- delete_orphaned_chunks
- get_orphaned_entities_preview
- delete_orphaned_entities
- PruneMemoryTool
"""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from zapomni_db.exceptions import ValidationError
from zapomni_db.falkordb_client import FalkorDBClient
from zapomni_mcp.tools.prune_memory import PruneMemoryTool

# ============================================================================
# mark_code_memories_stale TESTS
# ============================================================================


class TestMarkCodeMemoriesStale:
    """Test FalkorDBClient.mark_code_memories_stale()."""

    @pytest.fixture
    def mock_client(self, mocker):
        """Create mocked client."""
        client = FalkorDBClient()

        mocker.patch.object(client, "_initialized", True)
        mocker.patch.object(client, "_schema_ready", True)

        return client

    @pytest.mark.asyncio
    async def test_mark_stale_success(self, mock_client, mocker):
        """Test successfully marking memories as stale."""
        # Mock result with count
        mock_result = MagicMock()
        mock_result.rows = [{"marked_count": 5}]
        mock_result.row_count = 1

        mocker.patch.object(mock_client, "_execute_cypher", return_value=mock_result)

        count = await mock_client.mark_code_memories_stale("workspace_1")

        assert count == 5

    @pytest.mark.asyncio
    async def test_mark_stale_no_memories(self, mock_client, mocker):
        """Test when no memories exist."""
        mock_result = MagicMock()
        mock_result.rows = [{"marked_count": 0}]
        mock_result.row_count = 1

        mocker.patch.object(mock_client, "_execute_cypher", return_value=mock_result)

        count = await mock_client.mark_code_memories_stale("workspace_1")

        assert count == 0

    @pytest.mark.asyncio
    async def test_mark_stale_filters_by_source(self, mock_client, mocker):
        """Test that only code_indexer memories are marked."""
        execute_spy = mocker.patch.object(
            mock_client,
            "_execute_cypher",
            return_value=MagicMock(rows=[{"marked_count": 3}], row_count=1),
        )

        await mock_client.mark_code_memories_stale("ws_1")

        # Verify query filters by source = 'code_indexer'
        call_args = execute_spy.call_args
        query = call_args[0][0]
        assert "source = 'code_indexer'" in query

    @pytest.mark.asyncio
    async def test_mark_stale_filters_by_workspace(self, mock_client, mocker):
        """Test that memories are filtered by workspace."""
        execute_spy = mocker.patch.object(
            mock_client,
            "_execute_cypher",
            return_value=MagicMock(rows=[{"marked_count": 2}], row_count=1),
        )

        await mock_client.mark_code_memories_stale("my_workspace")

        # Verify workspace_id parameter is passed
        call_args = execute_spy.call_args
        params = call_args[0][1]
        assert params["workspace_id"] == "my_workspace"


# ============================================================================
# mark_memory_fresh TESTS
# ============================================================================


class TestMarkMemoryFresh:
    """Test FalkorDBClient.mark_memory_fresh()."""

    @pytest.fixture
    def mock_client(self, mocker):
        """Create mocked client."""
        client = FalkorDBClient()
        mocker.patch.object(client, "_initialized", True)
        mocker.patch.object(client, "_schema_ready", True)
        return client

    @pytest.mark.asyncio
    async def test_mark_fresh_existing_memory(self, mock_client, mocker):
        """Test marking an existing memory as fresh."""
        memory_id = str(uuid.uuid4())
        mock_result = MagicMock()
        mock_result.rows = [{"memory_id": memory_id}]
        mock_result.row_count = 1

        mocker.patch.object(mock_client, "_execute_cypher", return_value=mock_result)

        result = await mock_client.mark_memory_fresh(
            file_path="/path/to/file.py",
            workspace_id="workspace_1",
        )

        assert result == memory_id

    @pytest.mark.asyncio
    async def test_mark_fresh_not_found(self, mock_client, mocker):
        """Test when memory doesn't exist."""
        mock_result = MagicMock()
        mock_result.rows = []
        mock_result.row_count = 0

        mocker.patch.object(mock_client, "_execute_cypher", return_value=mock_result)

        result = await mock_client.mark_memory_fresh(
            file_path="/path/to/nonexistent.py",
            workspace_id="workspace_1",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_mark_fresh_exact_file_path_match(self, mock_client, mocker):
        """Test that file_path uses exact match (addresses validation warning)."""
        execute_spy = mocker.patch.object(
            mock_client, "_execute_cypher", return_value=MagicMock(rows=[], row_count=0)
        )

        await mock_client.mark_memory_fresh(
            file_path="/exact/path/file.py",
            workspace_id="ws_1",
        )

        # Verify query uses exact match (=) not CONTAINS
        call_args = execute_spy.call_args
        query = call_args[0][0]
        assert "m.file_path = $file_path" in query
        assert "CONTAINS" not in query


# ============================================================================
# count_stale_memories TESTS
# ============================================================================


class TestCountStaleMemories:
    """Test FalkorDBClient.count_stale_memories()."""

    @pytest.fixture
    def mock_client(self, mocker):
        """Create mocked client."""
        client = FalkorDBClient()
        mocker.patch.object(client, "_initialized", True)
        return client

    @pytest.mark.asyncio
    async def test_count_stale_with_results(self, mock_client, mocker):
        """Test counting stale memories."""
        mock_result = MagicMock()
        mock_result.rows = [{"count": 10}]
        mock_result.row_count = 1

        mocker.patch.object(mock_client, "_execute_cypher", return_value=mock_result)

        count = await mock_client.count_stale_memories("workspace_1")

        assert count == 10

    @pytest.mark.asyncio
    async def test_count_stale_zero(self, mock_client, mocker):
        """Test when no stale memories exist."""
        mock_result = MagicMock()
        mock_result.rows = [{"count": 0}]
        mock_result.row_count = 1

        mocker.patch.object(mock_client, "_execute_cypher", return_value=mock_result)

        count = await mock_client.count_stale_memories("workspace_1")

        assert count == 0


# ============================================================================
# get_stale_memories_preview TESTS
# ============================================================================


class TestGetStaleMemoriesPreview:
    """Test FalkorDBClient.get_stale_memories_preview()."""

    @pytest.fixture
    def mock_client(self, mocker):
        """Create mocked client."""
        client = FalkorDBClient()
        mocker.patch.object(client, "_initialized", True)
        return client

    @pytest.mark.asyncio
    async def test_preview_with_items(self, mock_client, mocker):
        """Test preview returns counts and items."""
        # Mock count result
        count_result = MagicMock()
        count_result.rows = [{"memory_count": 3, "chunk_count": 9}]
        count_result.row_count = 1

        # Mock preview result
        preview_result = MagicMock()
        preview_result.rows = [
            {
                "id": "mem_1",
                "file_path": "/path/file1.py",
                "metadata": '{"relative_path": "file1.py"}',
                "created_at": "2024-01-01T00:00:00Z",
                "chunk_count": 3,
            },
            {
                "id": "mem_2",
                "file_path": "/path/file2.py",
                "metadata": '{"relative_path": "file2.py"}',
                "created_at": "2024-01-02T00:00:00Z",
                "chunk_count": 3,
            },
        ]
        preview_result.row_count = 2

        # Return different results for different queries
        call_count = 0

        async def mock_execute(query, params):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return count_result
            return preview_result

        mocker.patch.object(mock_client, "_execute_cypher", side_effect=mock_execute)

        result = await mock_client.get_stale_memories_preview("workspace_1")

        assert result["memory_count"] == 3
        assert result["chunk_count"] == 9
        assert len(result["preview"]) == 2
        assert result["preview"][0]["type"] == "Memory"

    @pytest.mark.asyncio
    async def test_preview_empty(self, mock_client, mocker):
        """Test preview when no stale memories."""
        mock_result = MagicMock()
        mock_result.rows = [{"memory_count": 0, "chunk_count": 0}]
        mock_result.row_count = 1

        mocker.patch.object(mock_client, "_execute_cypher", return_value=mock_result)

        result = await mock_client.get_stale_memories_preview("workspace_1")

        assert result["memory_count"] == 0
        assert result["chunk_count"] == 0


# ============================================================================
# delete_stale_memories TESTS
# ============================================================================


class TestDeleteStaleMemories:
    """Test FalkorDBClient.delete_stale_memories()."""

    @pytest.fixture
    def mock_client(self, mocker):
        """Create mocked client."""
        client = FalkorDBClient()
        mocker.patch.object(client, "_initialized", True)
        return client

    @pytest.mark.asyncio
    async def test_delete_requires_confirmation(self, mock_client):
        """Test that deletion without confirm raises error."""
        with pytest.raises(ValidationError, match="confirmation"):
            await mock_client.delete_stale_memories("workspace_1", confirm=False)

    @pytest.mark.asyncio
    async def test_delete_success(self, mock_client, mocker):
        """Test successful deletion with confirmation."""
        # Mock preview for counts
        preview_result = MagicMock()
        preview_result.rows = [{"memory_count": 5, "chunk_count": 15}]
        preview_result.row_count = 1

        mocker.patch.object(
            mock_client,
            "get_stale_memories_preview",
            return_value={"memory_count": 5, "chunk_count": 15, "preview": []},
        )
        mocker.patch.object(mock_client, "_execute_cypher", return_value=MagicMock())

        result = await mock_client.delete_stale_memories("workspace_1", confirm=True)

        assert result["deleted_memories"] == 5
        assert result["deleted_chunks"] == 15

    @pytest.mark.asyncio
    async def test_delete_nothing_to_delete(self, mock_client, mocker):
        """Test deletion when no stale memories exist."""
        mocker.patch.object(
            mock_client,
            "get_stale_memories_preview",
            return_value={"memory_count": 0, "chunk_count": 0, "preview": []},
        )

        result = await mock_client.delete_stale_memories("workspace_1", confirm=True)

        assert result["deleted_memories"] == 0
        assert result["deleted_chunks"] == 0


# ============================================================================
# get_orphaned_chunks_preview TESTS
# ============================================================================


class TestGetOrphanedChunksPreview:
    """Test FalkorDBClient.get_orphaned_chunks_preview()."""

    @pytest.fixture
    def mock_client(self, mocker):
        """Create mocked client."""
        client = FalkorDBClient()
        mocker.patch.object(client, "_initialized", True)
        return client

    @pytest.mark.asyncio
    async def test_preview_orphaned_chunks(self, mock_client, mocker):
        """Test preview of orphaned chunks."""
        count_result = MagicMock()
        count_result.rows = [{"count": 5}]
        count_result.row_count = 1

        preview_result = MagicMock()
        preview_result.rows = [
            {"id": "chunk_1", "text_length": 500},
            {"id": "chunk_2", "text_length": 300},
        ]
        preview_result.row_count = 2

        call_count = 0

        async def mock_execute(query, params):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return count_result
            return preview_result

        mocker.patch.object(mock_client, "_execute_cypher", side_effect=mock_execute)

        result = await mock_client.get_orphaned_chunks_preview("workspace_1")

        assert result["count"] == 5
        assert len(result["preview"]) == 2
        assert result["preview"][0]["type"] == "Chunk"


# ============================================================================
# delete_orphaned_chunks TESTS
# ============================================================================


class TestDeleteOrphanedChunks:
    """Test FalkorDBClient.delete_orphaned_chunks()."""

    @pytest.fixture
    def mock_client(self, mocker):
        """Create mocked client."""
        client = FalkorDBClient()
        mocker.patch.object(client, "_initialized", True)
        return client

    @pytest.mark.asyncio
    async def test_delete_requires_confirmation(self, mock_client):
        """Test that deletion without confirm raises error."""
        with pytest.raises(ValidationError, match="confirmation"):
            await mock_client.delete_orphaned_chunks("workspace_1", confirm=False)

    @pytest.mark.asyncio
    async def test_delete_orphaned_chunks_success(self, mock_client, mocker):
        """Test successful orphaned chunk deletion."""
        mocker.patch.object(
            mock_client, "get_orphaned_chunks_preview", return_value={"count": 10, "preview": []}
        )
        mocker.patch.object(mock_client, "_execute_cypher", return_value=MagicMock())

        count = await mock_client.delete_orphaned_chunks("workspace_1", confirm=True)

        assert count == 10


# ============================================================================
# get_orphaned_entities_preview TESTS
# ============================================================================


class TestGetOrphanedEntitiesPreview:
    """Test FalkorDBClient.get_orphaned_entities_preview()."""

    @pytest.fixture
    def mock_client(self, mocker):
        """Create mocked client."""
        client = FalkorDBClient()
        mocker.patch.object(client, "_initialized", True)
        return client

    @pytest.mark.asyncio
    async def test_preview_orphaned_entities(self, mock_client, mocker):
        """Test preview of orphaned entities."""
        count_result = MagicMock()
        count_result.rows = [{"count": 3}]
        count_result.row_count = 1

        preview_result = MagicMock()
        preview_result.rows = [
            {"id": "ent_1", "name": "Python", "type": "TECHNOLOGY"},
        ]
        preview_result.row_count = 1

        call_count = 0

        async def mock_execute(query, params):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return count_result
            return preview_result

        mocker.patch.object(mock_client, "_execute_cypher", side_effect=mock_execute)

        result = await mock_client.get_orphaned_entities_preview("workspace_1")

        assert result["count"] == 3
        assert len(result["preview"]) == 1
        assert result["preview"][0]["type"] == "Entity"
        assert result["preview"][0]["name"] == "Python"


# ============================================================================
# delete_orphaned_entities TESTS
# ============================================================================


class TestDeleteOrphanedEntities:
    """Test FalkorDBClient.delete_orphaned_entities()."""

    @pytest.fixture
    def mock_client(self, mocker):
        """Create mocked client."""
        client = FalkorDBClient()
        mocker.patch.object(client, "_initialized", True)
        return client

    @pytest.mark.asyncio
    async def test_delete_requires_confirmation(self, mock_client):
        """Test that deletion without confirm raises error."""
        with pytest.raises(ValidationError, match="confirmation"):
            await mock_client.delete_orphaned_entities("workspace_1", confirm=False)

    @pytest.mark.asyncio
    async def test_delete_orphaned_entities_success(self, mock_client, mocker):
        """Test successful orphaned entity deletion."""
        mocker.patch.object(
            mock_client, "get_orphaned_entities_preview", return_value={"count": 7, "preview": []}
        )
        mocker.patch.object(mock_client, "_execute_cypher", return_value=MagicMock())

        count = await mock_client.delete_orphaned_entities("workspace_1", confirm=True)

        assert count == 7


# ============================================================================
# PruneMemoryTool TESTS
# ============================================================================


class TestPruneMemoryTool:
    """Test PruneMemoryTool MCP tool."""

    @pytest.fixture
    def mock_db_client(self, mocker):
        """Create mocked db_client."""
        client = MagicMock()

        # Setup async methods
        client.get_stale_memories_preview = AsyncMock(
            return_value={"memory_count": 5, "chunk_count": 15, "preview": []}
        )
        client.get_orphaned_chunks_preview = AsyncMock(return_value={"count": 3, "preview": []})
        client.get_orphaned_entities_preview = AsyncMock(return_value={"count": 2, "preview": []})
        client.delete_stale_memories = AsyncMock(
            return_value={"deleted_memories": 5, "deleted_chunks": 15}
        )
        client.delete_orphaned_chunks = AsyncMock(return_value=3)
        client.delete_orphaned_entities = AsyncMock(return_value=2)

        return client

    @pytest.fixture
    def tool(self, mock_db_client):
        """Create PruneMemoryTool with mocked client."""
        return PruneMemoryTool(db_client=mock_db_client)

    @pytest.mark.asyncio
    async def test_dry_run_default(self, tool):
        """Test that dry_run is True by default."""
        result = await tool.execute({"strategy": "stale_code"})

        assert result["isError"] is False
        # Should be a preview, not an actual deletion
        text = result["content"][0]["text"]
        assert "Preview" in text or "Dry run" in text

    @pytest.mark.asyncio
    async def test_dry_run_shows_preview(self, tool):
        """Test dry run shows preview information."""
        result = await tool.execute(
            {
                "strategy": "stale_code",
                "dry_run": True,
            }
        )

        assert result["isError"] is False
        text = result["content"][0]["text"]
        assert "stale_code" in text

    @pytest.mark.asyncio
    async def test_deletion_requires_confirm(self, tool):
        """Test deletion without confirm returns error."""
        result = await tool.execute(
            {
                "strategy": "stale_code",
                "dry_run": False,
                "confirm": False,
            }
        )

        assert result["isError"] is True
        text = result["content"][0]["text"]
        assert "confirm" in text.lower()

    @pytest.mark.asyncio
    async def test_deletion_with_confirm(self, tool, mock_db_client):
        """Test deletion works with confirm=True."""
        result = await tool.execute(
            {
                "strategy": "stale_code",
                "dry_run": False,
                "confirm": True,
            }
        )

        assert result["isError"] is False
        mock_db_client.delete_stale_memories.assert_called_once()
        text = result["content"][0]["text"]
        assert "deleted" in text.lower() or "complete" in text.lower()

    @pytest.mark.asyncio
    async def test_all_strategy(self, tool, mock_db_client):
        """Test 'all' strategy runs all deletion types."""
        result = await tool.execute(
            {
                "strategy": "all",
                "dry_run": False,
                "confirm": True,
            }
        )

        assert result["isError"] is False
        mock_db_client.delete_stale_memories.assert_called_once()
        mock_db_client.delete_orphaned_chunks.assert_called_once()
        mock_db_client.delete_orphaned_entities.assert_called_once()

    @pytest.mark.asyncio
    async def test_orphaned_chunks_strategy(self, tool, mock_db_client):
        """Test orphaned_chunks strategy."""
        result = await tool.execute(
            {
                "strategy": "orphaned_chunks",
                "dry_run": False,
                "confirm": True,
            }
        )

        assert result["isError"] is False
        mock_db_client.delete_orphaned_chunks.assert_called_once()
        mock_db_client.delete_stale_memories.assert_not_called()

    @pytest.mark.asyncio
    async def test_orphaned_entities_strategy(self, tool, mock_db_client):
        """Test orphaned_entities strategy."""
        result = await tool.execute(
            {
                "strategy": "orphaned_entities",
                "dry_run": False,
                "confirm": True,
            }
        )

        assert result["isError"] is False
        mock_db_client.delete_orphaned_entities.assert_called_once()
        mock_db_client.delete_stale_memories.assert_not_called()

    @pytest.mark.asyncio
    async def test_invalid_strategy(self, tool):
        """Test invalid strategy returns error."""
        result = await tool.execute(
            {
                "strategy": "invalid_strategy",
            }
        )

        assert result["isError"] is True

    def test_tool_schema(self, tool):
        """Test tool has correct schema."""
        assert tool.name == "prune_memory"
        assert "dry_run" in tool.input_schema["properties"]
        assert "confirm" in tool.input_schema["properties"]
        assert "strategy" in tool.input_schema["properties"]
        assert tool.input_schema["properties"]["dry_run"]["default"] is True
