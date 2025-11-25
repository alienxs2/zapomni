"""
Unit tests for WorkspaceManager.

Tests workspace CRUD operations with mocked FalkorDBClient.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from zapomni_core.exceptions import DatabaseError, ValidationError
from zapomni_core.workspace_manager import WorkspaceManager
from zapomni_db.models import DEFAULT_WORKSPACE_ID, Workspace, WorkspaceStats


class TestWorkspaceManagerInit:
    """Test WorkspaceManager initialization."""

    def test_init_with_valid_client(self):
        """Test initialization with valid client."""
        mock_client = MagicMock()
        manager = WorkspaceManager(db_client=mock_client)
        assert manager.db_client is mock_client

    def test_init_with_none_client_raises_error(self):
        """Test initialization with None client raises ValueError."""
        with pytest.raises(ValueError, match="db_client is required"):
            WorkspaceManager(db_client=None)


class TestCreateWorkspace:
    """Test workspace creation."""

    @pytest.fixture
    def manager(self):
        """Create WorkspaceManager with mock client."""
        mock_client = MagicMock()
        mock_client.create_workspace = AsyncMock(return_value="test-workspace")
        mock_client.get_workspace = AsyncMock(return_value=None)
        return WorkspaceManager(db_client=mock_client)

    @pytest.mark.asyncio
    async def test_create_workspace_success(self, manager):
        """Test successful workspace creation."""
        result = await manager.create_workspace(
            workspace_id="test-workspace",
            name="Test Workspace",
            description="A test workspace",
        )
        assert result == "test-workspace"
        manager.db_client.create_workspace.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_workspace_invalid_id_empty(self, manager):
        """Test creating workspace with empty ID raises error."""
        with pytest.raises(ValidationError):
            await manager.create_workspace(
                workspace_id="",
                name="Test",
            )

    @pytest.mark.asyncio
    async def test_create_workspace_invalid_id_special_chars(self, manager):
        """Test creating workspace with invalid characters raises error."""
        with pytest.raises(ValidationError):
            await manager.create_workspace(
                workspace_id="test workspace!",
                name="Test",
            )

    @pytest.mark.asyncio
    async def test_create_workspace_empty_name_raises_error(self, manager):
        """Test creating workspace with empty name raises error."""
        with pytest.raises(ValidationError):
            await manager.create_workspace(
                workspace_id="test-workspace",
                name="",
            )

    @pytest.mark.asyncio
    async def test_create_workspace_valid_ids(self, manager):
        """Test creating workspace with various valid IDs."""
        valid_ids = [
            "demo",
            "demo-workspace",
            "my-project",
            "workspace123",
            "a",
            "a" * 63,  # max 63 chars
        ]
        for ws_id in valid_ids:
            manager.db_client.create_workspace.reset_mock()
            result = await manager.create_workspace(
                workspace_id=ws_id,
                name="Test",
            )
            assert result == "test-workspace"


class TestGetWorkspace:
    """Test workspace retrieval."""

    @pytest.fixture
    def manager(self):
        """Create WorkspaceManager with mock client."""
        mock_client = MagicMock()
        return WorkspaceManager(db_client=mock_client)

    @pytest.mark.asyncio
    async def test_get_workspace_found(self, manager):
        """Test getting existing workspace."""
        mock_workspace = Workspace(
            id="test-workspace",
            name="Test Workspace",
            description="Test",
            created_at=datetime.now(timezone.utc),
        )
        manager.db_client.get_workspace = AsyncMock(return_value=mock_workspace)

        result = await manager.get_workspace("test-workspace")
        assert result is not None
        assert result.id == "test-workspace"

    @pytest.mark.asyncio
    async def test_get_workspace_not_found(self, manager):
        """Test getting non-existent workspace."""
        manager.db_client.get_workspace = AsyncMock(return_value=None)

        result = await manager.get_workspace("nonexistent")
        assert result is None


class TestListWorkspaces:
    """Test workspace listing."""

    @pytest.fixture
    def manager(self):
        """Create WorkspaceManager with mock client."""
        mock_client = MagicMock()
        return WorkspaceManager(db_client=mock_client)

    @pytest.mark.asyncio
    async def test_list_workspaces_empty(self, manager):
        """Test listing when no workspaces exist."""
        manager.db_client.list_workspaces = AsyncMock(return_value=[])

        result = await manager.list_workspaces()
        assert result == []

    @pytest.mark.asyncio
    async def test_list_workspaces_multiple(self, manager):
        """Test listing multiple workspaces."""
        workspaces = [
            Workspace(
                id="ws1",
                name="Workspace 1",
                created_at=datetime.now(timezone.utc),
            ),
            Workspace(
                id="ws2",
                name="Workspace 2",
                created_at=datetime.now(timezone.utc),
            ),
        ]
        manager.db_client.list_workspaces = AsyncMock(return_value=workspaces)

        result = await manager.list_workspaces()
        assert len(result) == 2
        assert result[0].id == "ws1"
        assert result[1].id == "ws2"


class TestDeleteWorkspace:
    """Test workspace deletion."""

    @pytest.fixture
    def manager(self):
        """Create WorkspaceManager with mock client."""
        mock_client = MagicMock()
        mock_client.delete_workspace = AsyncMock(return_value=True)
        return WorkspaceManager(db_client=mock_client)

    @pytest.mark.asyncio
    async def test_delete_workspace_success(self, manager):
        """Test successful workspace deletion."""
        result = await manager.delete_workspace("test-workspace", confirm=True)
        assert result is True
        manager.db_client.delete_workspace.assert_called_once_with("test-workspace", confirm=True)

    @pytest.mark.asyncio
    async def test_delete_workspace_without_confirm_raises_error(self, manager):
        """Test deleting without confirm raises error."""
        with pytest.raises(ValidationError):
            await manager.delete_workspace("test-workspace", confirm=False)

    @pytest.mark.asyncio
    async def test_delete_default_workspace_raises_error(self, manager):
        """Test deleting default workspace raises error."""
        with pytest.raises(ValidationError):
            await manager.delete_workspace(DEFAULT_WORKSPACE_ID, confirm=True)


class TestGetWorkspaceStats:
    """Test workspace statistics."""

    @pytest.fixture
    def manager(self):
        """Create WorkspaceManager with mock client."""
        mock_client = MagicMock()
        return WorkspaceManager(db_client=mock_client)

    @pytest.mark.asyncio
    async def test_get_workspace_stats(self, manager):
        """Test getting workspace statistics."""
        mock_stats = WorkspaceStats(
            workspace_id="test-workspace",
            total_memories=10,
            total_chunks=50,
            total_entities=25,
            total_relationships=15,
        )
        manager.db_client.get_workspace_stats = AsyncMock(return_value=mock_stats)

        result = await manager.get_workspace_stats("test-workspace")
        assert result.workspace_id == "test-workspace"
        assert result.total_memories == 10
        assert result.total_chunks == 50

    @pytest.mark.asyncio
    async def test_get_workspace_stats_default(self, manager):
        """Test getting stats for default workspace."""
        mock_stats = WorkspaceStats(
            workspace_id=DEFAULT_WORKSPACE_ID,
            total_memories=0,
            total_chunks=0,
            total_entities=0,
            total_relationships=0,
        )
        manager.db_client.get_workspace_stats = AsyncMock(return_value=mock_stats)

        result = await manager.get_workspace_stats()
        assert result.workspace_id == DEFAULT_WORKSPACE_ID


class TestWorkspaceExists:
    """Test workspace existence check."""

    @pytest.fixture
    def manager(self):
        """Create WorkspaceManager with mock client."""
        mock_client = MagicMock()
        return WorkspaceManager(db_client=mock_client)

    @pytest.mark.asyncio
    async def test_workspace_exists_true(self, manager):
        """Test workspace exists returns True."""
        mock_workspace = Workspace(
            id="test-workspace",
            name="Test",
            created_at=datetime.now(timezone.utc),
        )
        manager.db_client.get_workspace = AsyncMock(return_value=mock_workspace)

        result = await manager.workspace_exists("test-workspace")
        assert result is True

    @pytest.mark.asyncio
    async def test_workspace_exists_false(self, manager):
        """Test workspace exists returns False."""
        manager.db_client.get_workspace = AsyncMock(return_value=None)

        result = await manager.workspace_exists("nonexistent")
        assert result is False


class TestEnsureDefaultWorkspace:
    """Test default workspace creation."""

    @pytest.fixture
    def manager(self):
        """Create WorkspaceManager with mock client."""
        mock_client = MagicMock()
        return WorkspaceManager(db_client=mock_client)

    @pytest.mark.asyncio
    async def test_ensure_default_workspace_already_exists(self, manager):
        """Test ensuring default workspace when it already exists."""
        mock_workspace = Workspace(
            id=DEFAULT_WORKSPACE_ID,
            name="Default Workspace",
            created_at=datetime.now(timezone.utc),
        )
        manager.db_client.get_workspace = AsyncMock(return_value=mock_workspace)
        manager.db_client.create_workspace = AsyncMock()

        await manager.ensure_default_workspace()

        # Should not try to create if already exists
        manager.db_client.create_workspace.assert_not_called()

    @pytest.mark.asyncio
    async def test_ensure_default_workspace_creates_new(self, manager):
        """Test ensuring default workspace creates new one."""
        manager.db_client.get_workspace = AsyncMock(return_value=None)
        manager.db_client.create_workspace = AsyncMock(return_value=DEFAULT_WORKSPACE_ID)

        await manager.ensure_default_workspace()

        manager.db_client.create_workspace.assert_called_once()


class TestValidateWorkspaceId:
    """Test workspace ID validation."""

    @pytest.fixture
    def manager(self):
        """Create WorkspaceManager with mock client."""
        mock_client = MagicMock()
        return WorkspaceManager(db_client=mock_client)

    def test_validate_empty_id_raises_error(self, manager):
        """Test empty ID raises error."""
        with pytest.raises(ValidationError):
            manager._validate_workspace_id("")

    def test_validate_none_id_raises_error(self, manager):
        """Test None ID raises error."""
        with pytest.raises(ValidationError):
            manager._validate_workspace_id(None)

    def test_validate_id_with_spaces_raises_error(self, manager):
        """Test ID with spaces raises error."""
        with pytest.raises(ValidationError):
            manager._validate_workspace_id("test workspace")

    def test_validate_id_with_special_chars_raises_error(self, manager):
        """Test ID with special chars raises error."""
        with pytest.raises(ValidationError):
            manager._validate_workspace_id("test@workspace!")

    def test_validate_id_too_long_raises_error(self, manager):
        """Test ID that's too long raises error."""
        with pytest.raises(ValidationError):
            manager._validate_workspace_id("a" * 65)

    def test_validate_valid_ids(self, manager):
        """Test valid IDs pass validation."""
        valid_ids = [
            "demo",
            "demo-workspace",
            "demo_workspace",
            "myworkspace",
            "demo123",
            "a",
            "a" * 63,  # max 63 chars
        ]
        for ws_id in valid_ids:
            # Should not raise
            manager._validate_workspace_id(ws_id)

    def test_validate_reserved_names_raises_error(self, manager):
        """Test reserved workspace names raise error."""
        reserved_names = ["system", "admin", "test", "global", "root", "internal"]
        for name in reserved_names:
            with pytest.raises(ValidationError) as exc_info:
                manager._validate_workspace_id(name)
            assert "reserved" in str(exc_info.value).lower()

    def test_validate_uppercase_raises_error(self, manager):
        """Test uppercase IDs raise error (lowercase only)."""
        with pytest.raises(ValidationError):
            manager._validate_workspace_id("MyWorkspace")
        with pytest.raises(ValidationError):
            manager._validate_workspace_id("TEST")
