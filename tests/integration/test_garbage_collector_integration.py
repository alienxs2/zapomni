"""
Integration tests for Garbage Collection workflow.

Tests the complete GC workflow including:
- Schema initialization with new indexes
- Mark stale -> Index -> Mark fresh workflow
- Stale detection after file deletion
- Preview and deletion operations
- Delta indexing integration

Requirements:
- Running FalkorDB instance
- Correct FALKORDB_HOST, FALKORDB_PORT environment variables
"""

import asyncio
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pytest

from zapomni_db.falkordb_client import FalkorDBClient
from zapomni_db.models import DEFAULT_WORKSPACE_ID
from zapomni_db.schema_manager import SchemaManager


# Skip tests if no FalkorDB available
def falkordb_available():
    """Check if FalkorDB is available."""
    host = os.environ.get("FALKORDB_HOST", "localhost")
    port = int(os.environ.get("FALKORDB_PORT", 6381))
    try:
        import redis
        r = redis.Redis(host=host, port=port)
        r.ping()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not falkordb_available(),
    reason="FalkorDB not available"
)


@pytest.fixture
def workspace_id():
    """Create unique workspace for test isolation."""
    return f"test_gc_{uuid.uuid4().hex[:8]}"


@pytest.fixture
async def db_client():
    """Create and initialize FalkorDB client."""
    client = FalkorDBClient()
    await client.init_async()
    yield client
    await client.close()


@pytest.fixture
def temp_repo():
    """Create temporary repository for code indexing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some Python files
        files = [
            ("main.py", "def main():\n    print('Hello')\n"),
            ("utils.py", "def helper():\n    return 42\n"),
            ("models.py", "class User:\n    pass\n"),
        ]
        for filename, content in files:
            path = Path(tmpdir) / filename
            path.write_text(content)
        yield tmpdir


class TestSchemaWithGCIndexes:
    """Test schema includes GC indexes."""

    @pytest.mark.asyncio
    async def test_schema_has_stale_index(self, db_client):
        """Test schema creates stale index."""
        schema_manager = SchemaManager(graph=db_client.graph)

        # Verify index constants exist
        assert hasattr(schema_manager, "INDEX_MEMORY_STALE")
        assert hasattr(schema_manager, "INDEX_MEMORY_FILE_PATH")

    @pytest.mark.asyncio
    async def test_schema_init_creates_gc_indexes(self, db_client):
        """Test schema init creates GC-related indexes."""
        schema_manager = SchemaManager(graph=db_client.graph)
        schema_manager.init_schema()

        # Verify schema can be verified without errors
        status = schema_manager.verify_schema()
        assert status["initialized"] is True or len(status["issues"]) == 0


class TestMarkStaleWorkflow:
    """Test the mark stale workflow."""

    @pytest.mark.asyncio
    async def test_mark_stale_marks_code_memories(self, db_client, workspace_id):
        """Test marking code memories as stale."""
        # First, create a test memory with source='code_indexer'
        from zapomni_db.models import Chunk, Memory

        memory = Memory(
            text="Test code file",
            chunks=[Chunk(text="def test(): pass", embedding=[0.1] * 768)],
            embeddings=[[0.1] * 768],
            metadata={
                "source": "code_indexer",
                "file_path": "/test/path.py",
            },
        )

        memory_id = await db_client.add_memory(memory, workspace_id=workspace_id)
        assert memory_id is not None

        # Mark all code memories as stale
        marked_count = await db_client.mark_code_memories_stale(workspace_id)

        assert marked_count >= 1

        # Verify memory is now stale
        stale_count = await db_client.count_stale_memories(workspace_id)
        assert stale_count >= 1

    @pytest.mark.asyncio
    async def test_mark_fresh_clears_stale(self, db_client, workspace_id):
        """Test marking a memory fresh clears stale flag."""
        from zapomni_db.models import Chunk, Memory

        file_path = f"/test/fresh_{uuid.uuid4().hex[:8]}.py"

        memory = Memory(
            text="Test code file",
            chunks=[Chunk(text="def test(): pass", embedding=[0.1] * 768)],
            embeddings=[[0.1] * 768],
            metadata={
                "source": "code_indexer",
                "file_path": file_path,
            },
        )

        await db_client.add_memory(memory, workspace_id=workspace_id)

        # Mark all as stale
        await db_client.mark_code_memories_stale(workspace_id)

        # Verify stale
        stale_before = await db_client.count_stale_memories(workspace_id)
        assert stale_before >= 1

        # Mark fresh
        result = await db_client.mark_memory_fresh(file_path, workspace_id)
        assert result is not None  # Memory found and marked fresh

        # Verify stale count decreased
        stale_after = await db_client.count_stale_memories(workspace_id)
        assert stale_after < stale_before


class TestDeltaIndexingWorkflow:
    """Test the complete delta indexing workflow."""

    @pytest.mark.asyncio
    async def test_delta_indexing_detects_deleted_files(self, db_client, workspace_id):
        """Test delta indexing detects files that no longer exist."""
        from zapomni_db.models import Chunk, Memory

        # Simulate initial indexing with 3 files
        initial_files = [
            "/repo/file1.py",
            "/repo/file2.py",
            "/repo/file3.py",  # Will be "deleted"
        ]

        for file_path in initial_files:
            memory = Memory(
                text=f"Code: {file_path}",
                chunks=[Chunk(text="code", embedding=[0.1] * 768)],
                embeddings=[[0.1] * 768],
                metadata={
                    "source": "code_indexer",
                    "file_path": file_path,
                },
            )
            await db_client.add_memory(memory, workspace_id=workspace_id)

        # Simulate re-indexing with only 2 files (file3.py deleted)
        # Step 1: Mark all as stale
        await db_client.mark_code_memories_stale(workspace_id)

        # Step 2: Mark existing files as fresh
        remaining_files = ["/repo/file1.py", "/repo/file2.py"]
        for file_path in remaining_files:
            await db_client.mark_memory_fresh(file_path, workspace_id)

        # Step 3: Count stale (should be 1 - the deleted file)
        stale_count = await db_client.count_stale_memories(workspace_id)
        assert stale_count == 1

        # Step 4: Get preview of stale
        preview = await db_client.get_stale_memories_preview(workspace_id)
        assert preview["memory_count"] == 1
        # Verify the stale memory is for file3.py
        assert any(
            "file3.py" in (p.get("file_path") or "")
            for p in preview["preview"]
        )


class TestDeletionSafety:
    """Test deletion safety features."""

    @pytest.mark.asyncio
    async def test_delete_without_confirm_fails(self, db_client, workspace_id):
        """Test deletion fails without confirmation."""
        from zapomni_db.exceptions import ValidationError

        with pytest.raises(ValidationError, match="confirmation"):
            await db_client.delete_stale_memories(workspace_id, confirm=False)

        with pytest.raises(ValidationError, match="confirmation"):
            await db_client.delete_orphaned_chunks(workspace_id, confirm=False)

        with pytest.raises(ValidationError, match="confirmation"):
            await db_client.delete_orphaned_entities(workspace_id, confirm=False)

    @pytest.mark.asyncio
    async def test_delete_stale_with_confirm_works(self, db_client, workspace_id):
        """Test deletion works with confirmation."""
        from zapomni_db.models import Chunk, Memory

        # Create and mark stale
        memory = Memory(
            text="To be deleted",
            chunks=[Chunk(text="code", embedding=[0.1] * 768)],
            embeddings=[[0.1] * 768],
            metadata={
                "source": "code_indexer",
                "file_path": f"/delete/{uuid.uuid4().hex}.py",
            },
        )
        await db_client.add_memory(memory, workspace_id=workspace_id)
        await db_client.mark_code_memories_stale(workspace_id)

        # Verify stale exists
        stale_before = await db_client.count_stale_memories(workspace_id)
        assert stale_before >= 1

        # Delete with confirmation
        result = await db_client.delete_stale_memories(workspace_id, confirm=True)

        assert result["deleted_memories"] >= 1

        # Verify deleted
        stale_after = await db_client.count_stale_memories(workspace_id)
        assert stale_after == 0


class TestOrphanDetection:
    """Test orphan detection and cleanup."""

    @pytest.mark.asyncio
    async def test_orphaned_chunks_detection(self, db_client, workspace_id):
        """Test detection of orphaned chunks."""
        # This test requires directly creating orphaned chunks
        # which is complex in an integration test.
        # For now, verify the method runs without error
        result = await db_client.get_orphaned_chunks_preview(workspace_id)

        assert "count" in result
        assert "preview" in result
        assert isinstance(result["count"], int)

    @pytest.mark.asyncio
    async def test_orphaned_entities_detection(self, db_client, workspace_id):
        """Test detection of orphaned entities."""
        result = await db_client.get_orphaned_entities_preview(workspace_id)

        assert "count" in result
        assert "preview" in result
        assert isinstance(result["count"], int)


class TestPruneMemoryToolIntegration:
    """Integration tests for PruneMemoryTool."""

    @pytest.mark.asyncio
    async def test_prune_tool_dry_run(self, db_client, workspace_id):
        """Test prune tool in dry run mode."""
        from zapomni_mcp.tools.prune_memory import PruneMemoryTool

        tool = PruneMemoryTool(db_client=db_client)

        result = await tool.execute({
            "workspace_id": workspace_id,
            "strategy": "stale_code",
            "dry_run": True,
        })

        assert result["isError"] is False
        text = result["content"][0]["text"]
        assert "stale_code" in text.lower() or "preview" in text.lower()

    @pytest.mark.asyncio
    async def test_prune_tool_all_strategies(self, db_client, workspace_id):
        """Test prune tool with all strategies."""
        from zapomni_mcp.tools.prune_memory import PruneMemoryTool

        tool = PruneMemoryTool(db_client=db_client)

        # Test dry run for each strategy
        for strategy in ["stale_code", "orphaned_chunks", "orphaned_entities", "all"]:
            result = await tool.execute({
                "workspace_id": workspace_id,
                "strategy": strategy,
                "dry_run": True,
            })

            assert result["isError"] is False, f"Strategy {strategy} failed"
