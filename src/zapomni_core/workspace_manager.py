"""
WorkspaceManager - High-level workspace operations for Zapomni.

Provides workspace CRUD operations with validation and error handling.
Delegates database operations to FalkorDBClient.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import structlog

from zapomni_core.exceptions import DatabaseError, ValidationError
from zapomni_db import FalkorDBClient
from zapomni_db.models import DEFAULT_WORKSPACE_ID, Workspace, WorkspaceStats

logger = structlog.get_logger(__name__)


class WorkspaceManager:
    """
    High-level workspace management operations.

    Provides workspace CRUD operations with validation, error handling,
    and structured logging. Delegates database operations to FalkorDBClient.

    Attributes:
        db_client: FalkorDB client for storage operations
        logger: Structured logger for operations tracking

    Example:
        ```python
        from zapomni_core.workspace_manager import WorkspaceManager
        from zapomni_db import FalkorDBClient

        db = FalkorDBClient(host="localhost", port=6381)
        await db.init_async()

        manager = WorkspaceManager(db_client=db)

        # Create a workspace
        workspace_id = await manager.create_workspace(
            workspace_id="my-project",
            name="My Project",
            description="Project workspace"
        )

        # List all workspaces
        workspaces = await manager.list_workspaces()

        # Get workspace stats
        stats = await manager.get_workspace_stats("my-project")
        ```
    """

    # Regex pattern for valid workspace IDs (lowercase, starts with alphanumeric)
    WORKSPACE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")

    # Reserved workspace names that cannot be created
    RESERVED_WORKSPACES = frozenset({"system", "admin", "test", "global", "root", "internal"})

    def __init__(self, db_client: FalkorDBClient) -> None:
        """
        Initialize WorkspaceManager with database client.

        Args:
            db_client: FalkorDBClient instance for database operations.
                Must be initialized and connected.

        Raises:
            ValueError: If db_client is None
        """
        if db_client is None:
            raise ValueError("db_client is required")

        self.db_client = db_client
        self.logger = logger.bind(component="workspace_manager")

        self.logger.info("workspace_manager_initialized")

    async def create_workspace(
        self,
        workspace_id: str,
        name: str,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a new workspace.

        Args:
            workspace_id: Unique identifier (alphanumeric, hyphen, underscore)
            name: Human-readable workspace name
            description: Optional description
            metadata: Optional additional metadata

        Returns:
            The workspace_id of the created workspace

        Raises:
            ValidationError: If workspace_id format is invalid
            ValidationError: If workspace_id already exists
            DatabaseError: If database operation fails
        """
        log = self.logger.bind(operation="create_workspace", workspace_id=workspace_id)

        try:
            # Validate workspace_id format
            self._validate_workspace_id(workspace_id)

            # Validate name
            if not name or not name.strip():
                raise ValidationError(
                    message="Workspace name cannot be empty",
                    error_code="WS_VAL_002",
                )

            log.info("creating_workspace", name=name)

            # Create Workspace object
            workspace = Workspace(
                id=workspace_id,
                name=name.strip(),
                description=description.strip() if description else "",
                created_at=datetime.now(timezone.utc),
                metadata=metadata or {},
            )

            # Delegate to database client
            result_id = await self.db_client.create_workspace(workspace)

            log.info("workspace_created", workspace_id=result_id)
            return result_id

        except ValidationError:
            raise
        except DatabaseError:
            raise
        except Exception as e:
            log.error("create_workspace_error", error=str(e))
            raise DatabaseError(
                message=f"Failed to create workspace: {e}",
                error_code="WS_DB_001",
                original_exception=e,
            )

    async def get_workspace(self, workspace_id: str) -> Optional[Workspace]:
        """
        Get a workspace by ID.

        Args:
            workspace_id: Workspace ID to retrieve

        Returns:
            Workspace object if found, None otherwise

        Raises:
            ValidationError: If workspace_id format is invalid
            DatabaseError: If database operation fails
        """
        log = self.logger.bind(operation="get_workspace", workspace_id=workspace_id)

        try:
            # Validate workspace_id format
            self._validate_workspace_id(workspace_id)

            log.debug("getting_workspace")
            workspace = await self.db_client.get_workspace(workspace_id)

            if workspace:
                log.debug("workspace_found")
            else:
                log.debug("workspace_not_found")

            return workspace

        except ValidationError:
            raise
        except DatabaseError:
            raise
        except Exception as e:
            log.error("get_workspace_error", error=str(e))
            raise DatabaseError(
                message=f"Failed to get workspace: {e}",
                error_code="WS_DB_002",
                original_exception=e,
            )

    async def list_workspaces(self) -> List[Workspace]:
        """
        List all workspaces.

        Returns:
            List of Workspace objects sorted by creation date

        Raises:
            DatabaseError: If database operation fails
        """
        log = self.logger.bind(operation="list_workspaces")

        try:
            log.debug("listing_workspaces")
            workspaces = await self.db_client.list_workspaces()

            log.info("workspaces_listed", count=len(workspaces))
            return workspaces

        except DatabaseError:
            raise
        except Exception as e:
            log.error("list_workspaces_error", error=str(e))
            raise DatabaseError(
                message=f"Failed to list workspaces: {e}",
                error_code="WS_DB_003",
                original_exception=e,
            )

    async def delete_workspace(
        self,
        workspace_id: str,
        confirm: bool = False,
    ) -> bool:
        """
        Delete a workspace and all its data.

        WARNING: This is a destructive operation that cannot be undone.

        Args:
            workspace_id: Workspace ID to delete
            confirm: Must be True to confirm deletion

        Returns:
            True if deleted, False if workspace not found

        Raises:
            ValidationError: If workspace_id is the default workspace
            ValidationError: If confirm is not True
            ValidationError: If workspace_id format is invalid
            DatabaseError: If database operation fails
        """
        log = self.logger.bind(operation="delete_workspace", workspace_id=workspace_id)

        try:
            # Validate workspace_id format
            self._validate_workspace_id(workspace_id)

            # Cannot delete default workspace
            if workspace_id == DEFAULT_WORKSPACE_ID:
                raise ValidationError(
                    message="Cannot delete the default workspace",
                    error_code="WS_VAL_003",
                )

            # Require explicit confirmation
            if not confirm:
                raise ValidationError(
                    message="Deletion requires explicit confirmation (confirm=True)",
                    error_code="WS_VAL_004",
                )

            log.warning("deleting_workspace", workspace_id=workspace_id)

            # Delegate to database client
            deleted = await self.db_client.delete_workspace(workspace_id, confirm=confirm)

            if deleted:
                log.warning("workspace_deleted", workspace_id=workspace_id)
            else:
                log.info("workspace_not_found", workspace_id=workspace_id)

            return deleted

        except ValidationError:
            raise
        except DatabaseError:
            raise
        except Exception as e:
            log.error("delete_workspace_error", error=str(e))
            raise DatabaseError(
                message=f"Failed to delete workspace: {e}",
                error_code="WS_DB_004",
                original_exception=e,
            )

    async def get_workspace_stats(
        self,
        workspace_id: Optional[str] = None,
    ) -> WorkspaceStats:
        """
        Get statistics for a workspace.

        Args:
            workspace_id: Workspace ID. Defaults to "default".

        Returns:
            WorkspaceStats object with memory/chunk/entity counts

        Raises:
            ValidationError: If workspace_id format is invalid
            DatabaseError: If database operation fails
        """
        effective_workspace_id = workspace_id or DEFAULT_WORKSPACE_ID
        log = self.logger.bind(operation="get_workspace_stats", workspace_id=effective_workspace_id)

        try:
            # Validate workspace_id format
            self._validate_workspace_id(effective_workspace_id)

            log.debug("getting_workspace_stats")
            stats = await self.db_client.get_workspace_stats(effective_workspace_id)

            log.info(
                "workspace_stats_retrieved",
                total_memories=stats.total_memories,
                total_chunks=stats.total_chunks,
            )
            return stats

        except ValidationError:
            raise
        except DatabaseError:
            raise
        except Exception as e:
            log.error("get_workspace_stats_error", error=str(e))
            raise DatabaseError(
                message=f"Failed to get workspace stats: {e}",
                error_code="WS_DB_005",
                original_exception=e,
            )

    async def workspace_exists(self, workspace_id: str) -> bool:
        """
        Check if a workspace exists.

        Args:
            workspace_id: Workspace ID to check

        Returns:
            True if workspace exists, False otherwise

        Raises:
            ValidationError: If workspace_id format is invalid
            DatabaseError: If database operation fails
        """
        workspace = await self.get_workspace(workspace_id)
        return workspace is not None

    async def ensure_default_workspace(self) -> None:
        """
        Ensure the default workspace exists.

        Creates the default workspace if it doesn't exist.
        This is idempotent - safe to call multiple times.

        Raises:
            DatabaseError: If database operation fails
        """
        log = self.logger.bind(operation="ensure_default_workspace")

        try:
            existing = await self.db_client.get_workspace(DEFAULT_WORKSPACE_ID)
            if existing:
                log.debug("default_workspace_exists")
                return

            log.info("creating_default_workspace")

            workspace = Workspace(
                id=DEFAULT_WORKSPACE_ID,
                name="Default Workspace",
                description="Default workspace for memories without explicit workspace",
                created_at=datetime.now(timezone.utc),
                metadata={},
            )

            await self.db_client.create_workspace(workspace)
            log.info("default_workspace_created")

        except DatabaseError as e:
            # Workspace might already exist from concurrent creation
            if "already exists" in str(e).lower():
                log.debug("default_workspace_already_exists")
                return
            raise
        except Exception as e:
            log.error("ensure_default_workspace_error", error=str(e))
            raise DatabaseError(
                message=f"Failed to ensure default workspace: {e}",
                error_code="WS_DB_006",
                original_exception=e,
            )

    def _validate_workspace_id(self, workspace_id: str) -> None:
        """
        Validate workspace ID format.

        Args:
            workspace_id: Workspace ID to validate

        Raises:
            ValidationError: If format is invalid or reserved name
        """
        if not workspace_id:
            raise ValidationError(
                message="Workspace ID cannot be empty",
                error_code="WS_VAL_001",
            )

        # Check reserved names (case-insensitive)
        if workspace_id.lower() in self.RESERVED_WORKSPACES:
            raise ValidationError(
                message=f"Workspace ID '{workspace_id}' is reserved and cannot be used",
                error_code="WS_VAL_005",
                details={"workspace_id": workspace_id, "reserved": list(self.RESERVED_WORKSPACES)},
            )

        if not self.WORKSPACE_ID_PATTERN.match(workspace_id):
            raise ValidationError(
                message=(
                    "Workspace ID must be 1-63 lowercase characters, starting with "
                    "alphanumeric, containing only letters, numbers, hyphens, underscores"
                ),
                error_code="WS_VAL_001",
                details={"workspace_id": workspace_id},
            )
