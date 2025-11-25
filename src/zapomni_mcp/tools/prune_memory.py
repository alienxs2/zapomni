"""
PruneMemory MCP Tool - Manual garbage collection for knowledge graph.

Provides safe, explicit cleanup of stale nodes with dry-run preview
and confirmation requirements.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from enum import Enum
from typing import Any, Dict, List, Optional

import structlog
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from zapomni_db.falkordb_client import FalkorDBClient
from zapomni_db.models import DEFAULT_WORKSPACE_ID

logger = structlog.get_logger(__name__)


class PruneStrategy(str, Enum):
    """Garbage collection strategies."""

    STALE_CODE = "stale_code"
    ORPHANED_CHUNKS = "orphaned_chunks"
    ORPHANED_ENTITIES = "orphaned_entities"
    ALL = "all"


class PruneMemoryRequest(BaseModel):
    """Pydantic model for validating prune_memory request."""

    model_config = ConfigDict(extra="forbid")

    workspace_id: str = Field(
        default="",
        description="Workspace to prune (empty = current workspace)"
    )
    dry_run: bool = Field(
        default=True,
        description="Preview mode - show what would be deleted"
    )
    confirm: bool = Field(
        default=False,
        description="Required true for actual deletion"
    )
    strategy: PruneStrategy = Field(
        default=PruneStrategy.STALE_CODE,
        description="GC strategy to apply"
    )


class PrunePreviewItem(BaseModel):
    """Single item in prune preview."""

    id: str
    type: str  # "Memory", "Chunk", "Entity"
    file_path: Optional[str] = None
    relative_path: Optional[str] = None
    name: Optional[str] = None  # For entities
    created_at: Optional[str] = None
    chunk_count: int = 0


class PrunePreviewResponse(BaseModel):
    """Response for dry run preview."""

    dry_run: bool = True
    strategy: str
    nodes_to_delete: int = 0
    chunks_to_delete: int = 0
    entities_to_delete: int = 0
    preview: List[Dict[str, Any]] = []
    message: str = ""


class PruneResultResponse(BaseModel):
    """Response for actual deletion."""

    dry_run: bool = False
    strategy: str
    deleted_memories: int = 0
    deleted_chunks: int = 0
    deleted_entities: int = 0
    message: str = ""


class PruneMemoryTool:
    """
    MCP tool for pruning stale memory nodes from knowledge graph.

    Supports multiple GC strategies with dry-run preview and
    explicit confirmation for safety.

    Strategies:
        - stale_code: Delete Memory nodes marked as stale from code indexer
        - orphaned_chunks: Delete Chunk nodes without parent Memory
        - orphaned_entities: Delete Entity nodes without MENTIONS or RELATED_TO edges
        - all: Execute all strategies sequentially

    Safety Features:
        - Defaults to dry_run=true (preview only)
        - Requires confirm=true for actual deletion
        - Workspace-scoped operations
        - Comprehensive logging
    """

    name = "prune_memory"
    description = (
        "Prune stale or orphaned nodes from the knowledge graph. "
        "SAFETY: Defaults to dry_run=true (preview only). "
        "Set dry_run=false and confirm=true to delete. "
        "Strategies: stale_code, orphaned_chunks, orphaned_entities, all."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "workspace_id": {
                "type": "string",
                "description": "Workspace to prune (default: current workspace)",
                "default": "",
            },
            "dry_run": {
                "type": "boolean",
                "description": (
                    "Preview mode - show what would be deleted. "
                    "DEFAULT: true. Set to false to perform actual deletion."
                ),
                "default": True,
            },
            "confirm": {
                "type": "boolean",
                "description": (
                    "REQUIRED for deletion: Must be true to confirm deletion. "
                    "Ignored when dry_run=true."
                ),
                "default": False,
            },
            "strategy": {
                "type": "string",
                "description": (
                    "What to prune:\n"
                    "- 'stale_code': Delete stale code indexer memories\n"
                    "- 'orphaned_chunks': Delete chunks without parent memories\n"
                    "- 'orphaned_entities': Delete entities without mentions\n"
                    "- 'all': Run all strategies"
                ),
                "enum": ["stale_code", "orphaned_chunks", "orphaned_entities", "all"],
                "default": "stale_code",
            },
        },
        "required": [],
        "additionalProperties": False,
    }

    def __init__(self, db_client: FalkorDBClient) -> None:
        """
        Initialize with database client.

        Args:
            db_client: FalkorDBClient instance for database operations
        """
        self.db_client = db_client
        self.logger = logger.bind(tool=self.name)

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute prune_memory tool with provided arguments.

        Args:
            arguments: Dictionary containing:
                - workspace_id (str, optional): Workspace to prune
                - dry_run (bool, optional): Preview mode (default: true)
                - confirm (bool, optional): Confirm deletion (default: false)
                - strategy (str, optional): GC strategy (default: stale_code)

        Returns:
            MCP-formatted response dictionary
        """
        request_id = id(arguments)
        log = self.logger.bind(request_id=request_id)

        try:
            # Step 1: Validate arguments
            log.info("validating_prune_arguments")
            request = PruneMemoryRequest(**arguments)

            # Resolve workspace
            workspace_id = request.workspace_id or DEFAULT_WORKSPACE_ID

            log.info(
                "prune_request_validated",
                workspace_id=workspace_id,
                strategy=request.strategy.value,
                dry_run=request.dry_run,
            )

            # Step 2: Get preview (always, for logging)
            preview = await self._get_preview(workspace_id, request.strategy)

            log.info(
                "prune_preview_complete",
                workspace_id=workspace_id,
                strategy=request.strategy.value,
                nodes_to_delete=preview.nodes_to_delete,
                chunks_to_delete=preview.chunks_to_delete,
                entities_to_delete=preview.entities_to_delete,
            )

            # Step 3: If dry run, return preview
            if request.dry_run:
                return self._format_preview_response(preview)

            # Step 4: Check confirmation
            if not request.confirm:
                log.warning("deletion_not_confirmed", workspace_id=workspace_id)
                return self._format_error(
                    "Deletion requires explicit confirmation. "
                    "Set confirm=true to proceed. "
                    "Run with dry_run=true first to preview."
                )

            # Step 5: Execute deletion
            log.info(
                "executing_prune_deletion",
                workspace_id=workspace_id,
                strategy=request.strategy.value,
            )

            result = await self._execute_deletion(workspace_id, request.strategy)

            log.info(
                "prune_deletion_complete",
                workspace_id=workspace_id,
                strategy=request.strategy.value,
                deleted_memories=result.deleted_memories,
                deleted_chunks=result.deleted_chunks,
                deleted_entities=result.deleted_entities,
            )

            return self._format_result_response(result)

        except ValidationError as e:
            log.warning("validation_error", error=str(e))
            return self._format_error(str(e))

        except Exception as e:
            log.error("prune_error", error=str(e), exc_info=True)
            return self._format_error(f"Prune operation failed: {e}")

    async def _get_preview(
        self,
        workspace_id: str,
        strategy: PruneStrategy,
    ) -> PrunePreviewResponse:
        """Get preview of nodes to delete."""

        if strategy == PruneStrategy.ALL:
            # Aggregate all strategies
            stale = await self.db_client.get_stale_memories_preview(workspace_id)
            orphan_chunks = await self.db_client.get_orphaned_chunks_preview(workspace_id)
            orphan_entities = await self.db_client.get_orphaned_entities_preview(workspace_id)

            return PrunePreviewResponse(
                strategy="all",
                nodes_to_delete=stale["memory_count"],
                chunks_to_delete=stale["chunk_count"] + orphan_chunks["count"],
                entities_to_delete=orphan_entities["count"],
                preview=(
                    stale["preview"][:10] +
                    orphan_chunks["preview"][:5] +
                    orphan_entities["preview"][:5]
                ),
                message="Dry run complete. Set dry_run=false and confirm=true to delete.",
            )

        elif strategy == PruneStrategy.STALE_CODE:
            result = await self.db_client.get_stale_memories_preview(workspace_id)
            return PrunePreviewResponse(
                strategy="stale_code",
                nodes_to_delete=result["memory_count"],
                chunks_to_delete=result["chunk_count"],
                preview=result["preview"],
                message="Dry run complete. Set dry_run=false and confirm=true to delete.",
            )

        elif strategy == PruneStrategy.ORPHANED_CHUNKS:
            result = await self.db_client.get_orphaned_chunks_preview(workspace_id)
            return PrunePreviewResponse(
                strategy="orphaned_chunks",
                chunks_to_delete=result["count"],
                preview=result["preview"],
                message="Dry run complete. Set dry_run=false and confirm=true to delete.",
            )

        elif strategy == PruneStrategy.ORPHANED_ENTITIES:
            result = await self.db_client.get_orphaned_entities_preview(workspace_id)
            return PrunePreviewResponse(
                strategy="orphaned_entities",
                entities_to_delete=result["count"],
                preview=result["preview"],
                message="Dry run complete. Set dry_run=false and confirm=true to delete.",
            )

        # Should not reach here, but just in case
        return PrunePreviewResponse(
            strategy=strategy.value,
            message="Unknown strategy",
        )

    async def _execute_deletion(
        self,
        workspace_id: str,
        strategy: PruneStrategy,
    ) -> PruneResultResponse:
        """Execute actual deletion."""

        if strategy == PruneStrategy.ALL:
            # Execute all strategies
            stale_result = await self.db_client.delete_stale_memories(
                workspace_id, confirm=True
            )
            chunk_result = await self.db_client.delete_orphaned_chunks(
                workspace_id, confirm=True
            )
            entity_result = await self.db_client.delete_orphaned_entities(
                workspace_id, confirm=True
            )

            total_memories = stale_result["deleted_memories"]
            total_chunks = stale_result["deleted_chunks"] + chunk_result
            total_entities = entity_result

            return PruneResultResponse(
                strategy="all",
                deleted_memories=total_memories,
                deleted_chunks=total_chunks,
                deleted_entities=total_entities,
                message=(
                    f"Successfully deleted {total_memories} memories, "
                    f"{total_chunks} chunks, and {total_entities} entities."
                ),
            )

        elif strategy == PruneStrategy.STALE_CODE:
            result = await self.db_client.delete_stale_memories(
                workspace_id, confirm=True
            )
            return PruneResultResponse(
                strategy="stale_code",
                deleted_memories=result["deleted_memories"],
                deleted_chunks=result["deleted_chunks"],
                message=(
                    f"Successfully deleted {result['deleted_memories']} stale memory nodes "
                    f"and {result['deleted_chunks']} chunks."
                ),
            )

        elif strategy == PruneStrategy.ORPHANED_CHUNKS:
            count = await self.db_client.delete_orphaned_chunks(
                workspace_id, confirm=True
            )
            return PruneResultResponse(
                strategy="orphaned_chunks",
                deleted_chunks=count,
                message=f"Successfully deleted {count} orphaned chunks.",
            )

        elif strategy == PruneStrategy.ORPHANED_ENTITIES:
            count = await self.db_client.delete_orphaned_entities(
                workspace_id, confirm=True
            )
            return PruneResultResponse(
                strategy="orphaned_entities",
                deleted_entities=count,
                message=f"Successfully deleted {count} orphaned entities.",
            )

        # Should not reach here
        return PruneResultResponse(
            strategy=strategy.value,
            message="Unknown strategy",
        )

    def _format_preview_response(
        self,
        preview: PrunePreviewResponse,
    ) -> Dict[str, Any]:
        """Format preview response as MCP response."""
        # Build summary text
        summary_lines = [
            f"Prune Preview (Strategy: {preview.strategy})",
            "",
            "Nodes to delete:",
        ]

        if preview.nodes_to_delete > 0:
            summary_lines.append(f"  - Memory nodes: {preview.nodes_to_delete}")
        if preview.chunks_to_delete > 0:
            summary_lines.append(f"  - Chunk nodes: {preview.chunks_to_delete}")
        if preview.entities_to_delete > 0:
            summary_lines.append(f"  - Entity nodes: {preview.entities_to_delete}")

        total = preview.nodes_to_delete + preview.chunks_to_delete + preview.entities_to_delete
        if total == 0:
            summary_lines.append("  (none)")

        summary_lines.append("")

        # Add preview items
        if preview.preview:
            summary_lines.append("Preview of items to delete:")
            for item in preview.preview[:10]:  # Limit to 10 for readability
                item_type = item.get("type", "Unknown")
                if item_type == "Memory":
                    path = item.get("relative_path") or item.get("file_path") or "unknown"
                    chunks = item.get("chunk_count", 0)
                    summary_lines.append(f"  - [Memory] {path} ({chunks} chunks)")
                elif item_type == "Chunk":
                    chunk_id = item.get("id", "unknown")[:8]
                    length = item.get("text_length", 0)
                    summary_lines.append(f"  - [Chunk] {chunk_id}... ({length} chars)")
                elif item_type == "Entity":
                    name = item.get("name", "unknown")
                    entity_type = item.get("entity_type", "unknown")
                    summary_lines.append(f"  - [Entity] {name} ({entity_type})")

            if len(preview.preview) > 10:
                summary_lines.append(f"  ... and {len(preview.preview) - 10} more")

        summary_lines.append("")
        summary_lines.append(preview.message)

        return {
            "content": [
                {
                    "type": "text",
                    "text": "\n".join(summary_lines),
                }
            ],
            "isError": False,
        }

    def _format_result_response(
        self,
        result: PruneResultResponse,
    ) -> Dict[str, Any]:
        """Format deletion result as MCP response."""
        summary_lines = [
            f"Prune Complete (Strategy: {result.strategy})",
            "",
            "Deleted:",
        ]

        if result.deleted_memories > 0:
            summary_lines.append(f"  - Memory nodes: {result.deleted_memories}")
        if result.deleted_chunks > 0:
            summary_lines.append(f"  - Chunk nodes: {result.deleted_chunks}")
        if result.deleted_entities > 0:
            summary_lines.append(f"  - Entity nodes: {result.deleted_entities}")

        total = result.deleted_memories + result.deleted_chunks + result.deleted_entities
        if total == 0:
            summary_lines.append("  (none)")

        summary_lines.append("")
        summary_lines.append(result.message)

        return {
            "content": [
                {
                    "type": "text",
                    "text": "\n".join(summary_lines),
                }
            ],
            "isError": False,
        }

    def _format_error(self, message: str) -> Dict[str, Any]:
        """Format error as MCP error response."""
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Error: {message}",
                }
            ],
            "isError": True,
        }
