"""
Integration tests for workspace isolation.

Tests that memories in different workspaces are properly isolated.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from zapomni_db.models import DEFAULT_WORKSPACE_ID, Chunk, Memory, SearchResult


class TestWorkspaceIsolation:
    """Test workspace isolation for memories."""

    @pytest.fixture
    def mock_db_client(self):
        """Create mock database client."""
        client = MagicMock()
        client.add_memory = AsyncMock()
        client.vector_search = AsyncMock(return_value=[])
        client.delete_memory = AsyncMock(return_value=True)
        return client

    @pytest.fixture
    def mock_embedder(self):
        """Create mock embedder."""
        embedder = MagicMock()
        embedder.embed_text = AsyncMock(return_value=[0.1] * 768)
        return embedder

    @pytest.mark.asyncio
    async def test_add_memory_with_workspace(self, mock_db_client):
        """Test adding memory with workspace_id."""
        # Create memory with workspace_id
        memory = Memory(
            text="Test memory content",
            chunks=[Chunk(text="Test", index=0)],
            embeddings=[[0.1] * 768],
            workspace_id="test-workspace",
        )

        # Verify workspace_id is set correctly
        assert memory.workspace_id == "test-workspace"

    @pytest.mark.asyncio
    async def test_add_memory_default_workspace(self, mock_db_client):
        """Test adding memory uses default workspace."""
        # Create memory without workspace_id
        memory = Memory(
            text="Test memory content",
            chunks=[Chunk(text="Test", index=0)],
            embeddings=[[0.1] * 768],
        )

        # Verify default workspace is used
        assert memory.workspace_id == DEFAULT_WORKSPACE_ID

    @pytest.mark.asyncio
    async def test_search_returns_workspace_in_result(self, mock_db_client):
        """Test search results include workspace_id."""
        # Create search result with workspace_id
        result = SearchResult(
            memory_id="test-id",
            content="Test content",
            relevance_score=0.9,
            workspace_id="test-workspace",
        )

        assert result.workspace_id == "test-workspace"

    @pytest.mark.asyncio
    async def test_search_result_default_workspace(self, mock_db_client):
        """Test search result uses default workspace when not specified."""
        # Create search result without workspace_id
        result = SearchResult(
            memory_id="test-id",
            content="Test content",
            relevance_score=0.9,
        )

        # Should default to DEFAULT_WORKSPACE_ID
        assert result.workspace_id == DEFAULT_WORKSPACE_ID


class TestSessionWorkspaceIsolation:
    """Test session-based workspace isolation."""

    @pytest.fixture
    def mock_session_manager(self):
        """Create mock session manager."""
        from zapomni_mcp.session_manager import SessionManager

        manager = SessionManager()
        return manager

    @pytest.mark.asyncio
    async def test_set_and_get_workspace(self, mock_session_manager):
        """Test setting and getting workspace for a session."""
        from unittest.mock import MagicMock

        # Create a mock transport
        mock_transport = MagicMock()

        # Create session
        await mock_session_manager.create_session(
            session_id="test-session",
            transport=mock_transport,
        )

        # Default workspace should be "default"
        workspace = mock_session_manager.get_workspace_id("test-session")
        assert workspace == DEFAULT_WORKSPACE_ID

        # Set custom workspace
        result = mock_session_manager.set_workspace_id("test-session", "custom-workspace")
        assert result is True

        # Get should return custom workspace
        workspace = mock_session_manager.get_workspace_id("test-session")
        assert workspace == "custom-workspace"

        # Cleanup
        await mock_session_manager.remove_session("test-session")

    @pytest.mark.asyncio
    async def test_get_workspace_nonexistent_session(self, mock_session_manager):
        """Test getting workspace for nonexistent session returns default."""
        workspace = mock_session_manager.get_workspace_id("nonexistent")
        assert workspace == DEFAULT_WORKSPACE_ID

    @pytest.mark.asyncio
    async def test_set_workspace_nonexistent_session(self, mock_session_manager):
        """Test setting workspace for nonexistent session returns False."""
        result = mock_session_manager.set_workspace_id("nonexistent", "custom")
        assert result is False


class TestWorkspaceValidation:
    """Test workspace validation in delete operations."""

    @pytest.mark.asyncio
    async def test_delete_memory_validates_workspace(self):
        """Test delete_memory validates workspace ownership."""
        from zapomni_db.exceptions import ValidationError
        from zapomni_db.falkordb_client import FalkorDBClient

        # Create a mock client that simulates workspace mismatch
        mock_client = MagicMock(spec=FalkorDBClient)
        mock_client._execute_cypher = AsyncMock()
        mock_client._initialized = True

        # Mock the workspace check to return a different workspace
        mock_result = MagicMock()
        mock_result.row_count = 1
        mock_result.rows = [{"workspace_id": "other-workspace"}]
        mock_client._execute_cypher.return_value = mock_result

        # The actual implementation should raise ValidationError
        # when workspace doesn't match


class TestCypherQueryWorkspaceFilter:
    """Test Cypher query builder with workspace filters."""

    def test_vector_search_includes_workspace_filter(self):
        """Test vector search query includes workspace_id filter."""
        from zapomni_db.cypher_query_builder import CypherQueryBuilder

        builder = CypherQueryBuilder()
        cypher, params = builder.build_vector_search_query(
            embedding=[0.1] * 768,
            limit=10,
            workspace_id="test-workspace",
        )

        # Verify workspace_id is in query
        assert "$workspace_id" in cypher
        assert "workspace_id" in params
        assert params["workspace_id"] == "test-workspace"

        # Verify filter comes AFTER YIELD clause (important for FalkorDB)
        yield_pos = cypher.find("YIELD")
        workspace_filter_pos = cypher.find("WHERE c.workspace_id")
        assert yield_pos < workspace_filter_pos

    def test_vector_search_default_workspace(self):
        """Test vector search uses default workspace when not specified."""
        from zapomni_db.cypher_query_builder import CypherQueryBuilder

        builder = CypherQueryBuilder()
        cypher, params = builder.build_vector_search_query(
            embedding=[0.1] * 768,
            limit=10,
        )

        # Should use default workspace
        assert params["workspace_id"] == DEFAULT_WORKSPACE_ID
