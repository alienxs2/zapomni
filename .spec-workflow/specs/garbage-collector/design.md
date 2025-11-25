# Garbage Collector Technical Design

**Version:** 1.0.0
**Date:** 2025-11-25
**Status:** Draft
**Task:** 04 - Implement Garbage Collector

---

## 1. Architecture Overview

### 1.1 Component Diagram

```
+------------------+     +-------------------+     +------------------+
|   MCP Server     |---->|  PruneMemoryTool  |---->|  FalkorDBClient  |
+------------------+     +-------------------+     +------------------+
                                  |                        |
                                  v                        v
                         +----------------+       +----------------+
                         | PruneRequest   |       | Cypher Queries |
                         | PruneResponse  |       | (GC Methods)   |
                         +----------------+       +----------------+
                                                          |
                                                          v
                                                 +----------------+
                                                 |   FalkorDB     |
                                                 |   (Graph DB)   |
                                                 +----------------+
```

### 1.2 Data Flow

```
User Request
    |
    v
PruneMemoryTool.execute(arguments)
    |
    v
Validate Input (Pydantic: PruneMemoryRequest)
    |
    +--[dry_run=true]--> Preview Query --> Format Preview Response
    |
    +--[dry_run=false, confirm=false]--> Error Response (require confirmation)
    |
    +--[dry_run=false, confirm=true]--> Execute Deletion --> Format Result Response
```

---

## 2. Schema Changes

### 2.1 Memory Node Properties

Add two new properties to Memory nodes:

```cypher
// Memory node with GC properties
CREATE (m:Memory {
    id: $memory_id,
    text: $text,
    tags: $tags,
    source: $source,
    metadata: $metadata,
    workspace_id: $workspace_id,
    created_at: $timestamp,
    // NEW: Garbage collection properties
    stale: false,                    // Boolean - mark for deletion
    last_seen_at: $timestamp         // DateTime - last verification time
})
```

### 2.2 New Index

Add property index for `stale` flag in `schema_manager.py`:

```python
# In create_property_indexes():
indexes = [
    # ... existing indexes ...
    ("memory_stale_idx", self.NODE_MEMORY, "stale"),
]
```

### 2.3 Schema Migration (Implicit)

Existing Memory nodes will have `stale=null` and `last_seen_at=null`. The GC queries handle this gracefully:

```cypher
// Treat null as not stale
WHERE m.stale = true  -- null values are excluded

// Treat null last_seen_at as created_at
COALESCE(m.last_seen_at, m.created_at)
```

---

## 3. PruneMemoryTool Class Design

### 3.1 File Location

`/src/zapomni_mcp/tools/prune_memory.py`

### 3.2 Class Definition

```python
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
from pydantic import BaseModel, ConfigDict, Field, field_validator

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
    preview: List[PrunePreviewItem] = []
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
        """Initialize with database client."""
        self.db_client = db_client
        self.logger = logger.bind(tool=self.name)

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute prune_memory tool with provided arguments."""
        # Implementation details in section 3.3
        pass
```

### 3.3 Execute Method Flow

```python
async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Execute prune_memory tool with provided arguments."""
    request_id = id(arguments)
    log = self.logger.bind(request_id=request_id)

    try:
        # Step 1: Validate arguments
        log.info("validating_prune_arguments")
        request = PruneMemoryRequest(**arguments)

        # Resolve workspace
        workspace_id = request.workspace_id or DEFAULT_WORKSPACE_ID

        # Step 2: Get preview (always, for logging)
        preview = await self._get_preview(workspace_id, request.strategy)

        log.info(
            "prune_preview_complete",
            workspace_id=workspace_id,
            strategy=request.strategy.value,
            nodes_to_delete=preview.nodes_to_delete,
            chunks_to_delete=preview.chunks_to_delete,
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
```

### 3.4 Preview Method

```python
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
            preview=stale["preview"][:10] + orphan_chunks["preview"][:5] + orphan_entities["preview"][:5],
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
```

### 3.5 Deletion Method

```python
async def _execute_deletion(
    self,
    workspace_id: str,
    strategy: PruneStrategy,
) -> PruneResultResponse:
    """Execute actual deletion."""

    if strategy == PruneStrategy.ALL:
        # Execute all strategies
        stale_result = await self.db_client.delete_stale_memories(workspace_id)
        chunk_result = await self.db_client.delete_orphaned_chunks(workspace_id)
        entity_result = await self.db_client.delete_orphaned_entities(workspace_id)

        return PruneResultResponse(
            strategy="all",
            deleted_memories=stale_result["deleted_memories"],
            deleted_chunks=stale_result["deleted_chunks"] + chunk_result,
            deleted_entities=entity_result,
            message=f"Successfully deleted {stale_result['deleted_memories']} memories, "
                    f"{stale_result['deleted_chunks'] + chunk_result} chunks, "
                    f"and {entity_result} entities.",
        )

    elif strategy == PruneStrategy.STALE_CODE:
        result = await self.db_client.delete_stale_memories(workspace_id)
        return PruneResultResponse(
            strategy="stale_code",
            deleted_memories=result["deleted_memories"],
            deleted_chunks=result["deleted_chunks"],
            message=f"Successfully deleted {result['deleted_memories']} stale memory nodes "
                    f"and {result['deleted_chunks']} chunks.",
        )

    elif strategy == PruneStrategy.ORPHANED_CHUNKS:
        count = await self.db_client.delete_orphaned_chunks(workspace_id)
        return PruneResultResponse(
            strategy="orphaned_chunks",
            deleted_chunks=count,
            message=f"Successfully deleted {count} orphaned chunks.",
        )

    elif strategy == PruneStrategy.ORPHANED_ENTITIES:
        count = await self.db_client.delete_orphaned_entities(workspace_id)
        return PruneResultResponse(
            strategy="orphaned_entities",
            deleted_entities=count,
            message=f"Successfully deleted {count} orphaned entities.",
        )
```

---

## 4. FalkorDBClient GC Methods

### 4.1 New Methods to Add

Add these methods to `/src/zapomni_db/falkordb_client.py`:

```python
# ========================================
# GARBAGE COLLECTION OPERATIONS
# ========================================

async def mark_code_memories_stale(
    self,
    workspace_id: str,
) -> int:
    """
    Mark all code memories as stale before re-indexing.

    Args:
        workspace_id: Workspace to mark

    Returns:
        Count of memories marked as stale
    """
    cypher = """
    MATCH (m:Memory)
    WHERE m.source = 'code_indexer'
      AND m.workspace_id = $workspace_id
    SET m.stale = true
    RETURN count(m) AS marked_count
    """

    result = await self._execute_cypher(cypher, {"workspace_id": workspace_id})

    if result.row_count > 0:
        count = result.rows[0].get("marked_count", 0)
        self._logger.info(
            "memories_marked_stale",
            workspace_id=workspace_id,
            count=count,
        )
        return count
    return 0


async def mark_memory_fresh(
    self,
    file_path: str,
    workspace_id: str,
) -> Optional[str]:
    """
    Mark a specific memory as fresh (not stale) during indexing.
    Also updates last_seen_at timestamp.

    Args:
        file_path: Absolute file path
        workspace_id: Workspace ID

    Returns:
        Memory ID if found and updated, None otherwise
    """
    cypher = """
    MATCH (m:Memory)
    WHERE m.source = 'code_indexer'
      AND m.workspace_id = $workspace_id
      AND m.metadata CONTAINS $file_path_pattern
    SET m.stale = false,
        m.last_seen_at = datetime()
    RETURN m.id AS memory_id
    """

    # The file_path is stored in metadata JSON
    file_path_pattern = f'"file_path": "{file_path}"'

    result = await self._execute_cypher(cypher, {
        "workspace_id": workspace_id,
        "file_path_pattern": file_path_pattern,
    })

    if result.row_count > 0:
        memory_id = result.rows[0].get("memory_id")
        self._logger.debug(
            "memory_marked_fresh",
            memory_id=memory_id,
            file_path=file_path,
        )
        return memory_id
    return None


async def get_stale_memories_preview(
    self,
    workspace_id: str,
    limit: int = 20,
) -> Dict[str, Any]:
    """
    Get preview of stale memories for dry-run.

    Args:
        workspace_id: Workspace to query
        limit: Maximum preview items

    Returns:
        {
            "memory_count": int,
            "chunk_count": int,
            "preview": List[Dict]
        }
    """
    # Count query
    count_cypher = """
    MATCH (m:Memory)
    WHERE m.source = 'code_indexer'
      AND m.workspace_id = $workspace_id
      AND m.stale = true
    OPTIONAL MATCH (m)-[:HAS_CHUNK]->(c:Chunk)
    RETURN count(DISTINCT m) AS memory_count,
           count(DISTINCT c) AS chunk_count
    """

    count_result = await self._execute_cypher(count_cypher, {"workspace_id": workspace_id})

    memory_count = 0
    chunk_count = 0
    if count_result.row_count > 0:
        memory_count = count_result.rows[0].get("memory_count", 0) or 0
        chunk_count = count_result.rows[0].get("chunk_count", 0) or 0

    # Preview query
    preview_cypher = """
    MATCH (m:Memory)
    WHERE m.source = 'code_indexer'
      AND m.workspace_id = $workspace_id
      AND m.stale = true
    OPTIONAL MATCH (m)-[:HAS_CHUNK]->(c:Chunk)
    WITH m, count(c) AS chunk_count
    RETURN m.id AS id,
           m.metadata AS metadata,
           m.created_at AS created_at,
           chunk_count
    ORDER BY m.created_at DESC
    LIMIT $limit
    """

    preview_result = await self._execute_cypher(preview_cypher, {
        "workspace_id": workspace_id,
        "limit": limit,
    })

    preview = []
    for row in preview_result.rows:
        metadata = row.get("metadata", "{}")
        if isinstance(metadata, str):
            import json
            metadata = json.loads(metadata)

        preview.append({
            "id": row.get("id"),
            "type": "Memory",
            "file_path": metadata.get("file_path"),
            "relative_path": metadata.get("relative_path"),
            "created_at": row.get("created_at"),
            "chunk_count": row.get("chunk_count", 0),
        })

    return {
        "memory_count": memory_count,
        "chunk_count": chunk_count,
        "preview": preview,
    }


async def delete_stale_memories(
    self,
    workspace_id: str,
) -> Dict[str, int]:
    """
    Delete all stale memories and their chunks.

    Args:
        workspace_id: Workspace to clean

    Returns:
        {"deleted_memories": int, "deleted_chunks": int}
    """
    cypher = """
    MATCH (m:Memory)
    WHERE m.source = 'code_indexer'
      AND m.workspace_id = $workspace_id
      AND m.stale = true
    OPTIONAL MATCH (m)-[:HAS_CHUNK]->(c:Chunk)
    WITH m, collect(c) AS chunks
    DETACH DELETE m
    WITH chunks
    UNWIND chunks AS c
    DELETE c
    RETURN count(DISTINCT m) AS deleted_memories,
           count(DISTINCT c) AS deleted_chunks
    """

    # Note: The above query has issues with counting after deletion
    # Better approach: count first, then delete

    # Count first
    count_result = await self.get_stale_memories_preview(workspace_id, limit=1)
    memory_count = count_result["memory_count"]
    chunk_count = count_result["chunk_count"]

    # Delete
    delete_cypher = """
    MATCH (m:Memory)
    WHERE m.source = 'code_indexer'
      AND m.workspace_id = $workspace_id
      AND m.stale = true
    OPTIONAL MATCH (m)-[:HAS_CHUNK]->(c:Chunk)
    DETACH DELETE m, c
    """

    await self._execute_cypher(delete_cypher, {"workspace_id": workspace_id})

    self._logger.info(
        "stale_memories_deleted",
        workspace_id=workspace_id,
        deleted_memories=memory_count,
        deleted_chunks=chunk_count,
    )

    return {
        "deleted_memories": memory_count,
        "deleted_chunks": chunk_count,
    }


async def get_orphaned_chunks_preview(
    self,
    workspace_id: str,
    limit: int = 20,
) -> Dict[str, Any]:
    """
    Get preview of orphaned chunks.

    Args:
        workspace_id: Workspace to query
        limit: Maximum preview items

    Returns:
        {"count": int, "preview": List[Dict]}
    """
    count_cypher = """
    MATCH (c:Chunk)
    WHERE c.workspace_id = $workspace_id
      AND NOT (:Memory)-[:HAS_CHUNK]->(c)
    RETURN count(c) AS count
    """

    count_result = await self._execute_cypher(count_cypher, {"workspace_id": workspace_id})
    count = count_result.rows[0].get("count", 0) if count_result.row_count > 0 else 0

    preview_cypher = """
    MATCH (c:Chunk)
    WHERE c.workspace_id = $workspace_id
      AND NOT (:Memory)-[:HAS_CHUNK]->(c)
    RETURN c.id AS id,
           size(c.text) AS text_length
    LIMIT $limit
    """

    preview_result = await self._execute_cypher(preview_cypher, {
        "workspace_id": workspace_id,
        "limit": limit,
    })

    preview = [
        {
            "id": row.get("id"),
            "type": "Chunk",
            "text_length": row.get("text_length", 0),
        }
        for row in preview_result.rows
    ]

    return {"count": count, "preview": preview}


async def delete_orphaned_chunks(
    self,
    workspace_id: str,
) -> int:
    """
    Delete chunks without parent memories.

    Args:
        workspace_id: Workspace to clean

    Returns:
        Number of chunks deleted
    """
    # Count first
    count_result = await self.get_orphaned_chunks_preview(workspace_id, limit=1)
    count = count_result["count"]

    # Delete
    delete_cypher = """
    MATCH (c:Chunk)
    WHERE c.workspace_id = $workspace_id
      AND NOT (:Memory)-[:HAS_CHUNK]->(c)
    DETACH DELETE c
    """

    await self._execute_cypher(delete_cypher, {"workspace_id": workspace_id})

    self._logger.info(
        "orphaned_chunks_deleted",
        workspace_id=workspace_id,
        count=count,
    )

    return count


async def get_orphaned_entities_preview(
    self,
    workspace_id: str,
    limit: int = 20,
) -> Dict[str, Any]:
    """
    Get preview of orphaned entities.

    Args:
        workspace_id: Workspace to query
        limit: Maximum preview items

    Returns:
        {"count": int, "preview": List[Dict]}
    """
    count_cypher = """
    MATCH (e:Entity)
    WHERE e.workspace_id = $workspace_id
      AND NOT (:Chunk)-[:MENTIONS]->(e)
      AND NOT (e)-[:RELATED_TO]-(:Entity)
    RETURN count(e) AS count
    """

    count_result = await self._execute_cypher(count_cypher, {"workspace_id": workspace_id})
    count = count_result.rows[0].get("count", 0) if count_result.row_count > 0 else 0

    preview_cypher = """
    MATCH (e:Entity)
    WHERE e.workspace_id = $workspace_id
      AND NOT (:Chunk)-[:MENTIONS]->(e)
      AND NOT (e)-[:RELATED_TO]-(:Entity)
    RETURN e.id AS id,
           e.name AS name,
           e.type AS type
    LIMIT $limit
    """

    preview_result = await self._execute_cypher(preview_cypher, {
        "workspace_id": workspace_id,
        "limit": limit,
    })

    preview = [
        {
            "id": row.get("id"),
            "type": "Entity",
            "name": row.get("name"),
            "entity_type": row.get("type"),
        }
        for row in preview_result.rows
    ]

    return {"count": count, "preview": preview}


async def delete_orphaned_entities(
    self,
    workspace_id: str,
) -> int:
    """
    Delete entities without mentions.

    Args:
        workspace_id: Workspace to clean

    Returns:
        Number of entities deleted
    """
    # Count first
    count_result = await self.get_orphaned_entities_preview(workspace_id, limit=1)
    count = count_result["count"]

    # Delete
    delete_cypher = """
    MATCH (e:Entity)
    WHERE e.workspace_id = $workspace_id
      AND NOT (:Chunk)-[:MENTIONS]->(e)
      AND NOT (e)-[:RELATED_TO]-(:Entity)
    DETACH DELETE e
    """

    await self._execute_cypher(delete_cypher, {"workspace_id": workspace_id})

    self._logger.info(
        "orphaned_entities_deleted",
        workspace_id=workspace_id,
        count=count,
    )

    return count


async def count_stale_memories(
    self,
    workspace_id: str,
) -> int:
    """
    Count stale memories in workspace.

    Args:
        workspace_id: Workspace to query

    Returns:
        Count of stale memories
    """
    cypher = """
    MATCH (m:Memory)
    WHERE m.source = 'code_indexer'
      AND m.workspace_id = $workspace_id
      AND m.stale = true
    RETURN count(m) AS count
    """

    result = await self._execute_cypher(cypher, {"workspace_id": workspace_id})
    return result.rows[0].get("count", 0) if result.row_count > 0 else 0
```

---

## 5. Cypher Queries Reference

### 5.1 Mark Stale Query

```cypher
// Mark all code memories as stale before re-indexing
MATCH (m:Memory)
WHERE m.source = 'code_indexer'
  AND m.workspace_id = $workspace_id
SET m.stale = true
RETURN count(m) AS marked_count
```

### 5.2 Mark Fresh Query

```cypher
// Mark memory as fresh during indexing
MATCH (m:Memory)
WHERE m.source = 'code_indexer'
  AND m.workspace_id = $workspace_id
  AND m.metadata CONTAINS $file_path_pattern
SET m.stale = false,
    m.last_seen_at = datetime()
RETURN m.id AS memory_id
```

### 5.3 Preview Stale Memories Query

```cypher
// Preview stale memories
MATCH (m:Memory)
WHERE m.source = 'code_indexer'
  AND m.workspace_id = $workspace_id
  AND m.stale = true
OPTIONAL MATCH (m)-[:HAS_CHUNK]->(c:Chunk)
WITH m, count(c) AS chunk_count
RETURN m.id AS id,
       m.metadata AS metadata,
       m.created_at AS created_at,
       chunk_count
ORDER BY m.created_at DESC
LIMIT $limit
```

### 5.4 Delete Stale Memories Query

```cypher
// Delete stale memories and their chunks
MATCH (m:Memory)
WHERE m.source = 'code_indexer'
  AND m.workspace_id = $workspace_id
  AND m.stale = true
OPTIONAL MATCH (m)-[:HAS_CHUNK]->(c:Chunk)
DETACH DELETE m, c
```

### 5.5 Orphaned Chunks Query

```cypher
// Find orphaned chunks (no parent memory)
MATCH (c:Chunk)
WHERE c.workspace_id = $workspace_id
  AND NOT (:Memory)-[:HAS_CHUNK]->(c)
RETURN c.id AS id, size(c.text) AS text_length
LIMIT $limit
```

### 5.6 Delete Orphaned Chunks Query

```cypher
// Delete orphaned chunks
MATCH (c:Chunk)
WHERE c.workspace_id = $workspace_id
  AND NOT (:Memory)-[:HAS_CHUNK]->(c)
DETACH DELETE c
```

### 5.7 Orphaned Entities Query

```cypher
// Find orphaned entities (no mentions, no relationships)
MATCH (e:Entity)
WHERE e.workspace_id = $workspace_id
  AND NOT (:Chunk)-[:MENTIONS]->(e)
  AND NOT (e)-[:RELATED_TO]-(:Entity)
RETURN e.id AS id, e.name AS name, e.type AS type
LIMIT $limit
```

### 5.8 Delete Orphaned Entities Query

```cypher
// Delete orphaned entities
MATCH (e:Entity)
WHERE e.workspace_id = $workspace_id
  AND NOT (:Chunk)-[:MENTIONS]->(e)
  AND NOT (e)-[:RELATED_TO]-(:Entity)
DETACH DELETE e
```

---

## 6. Delta Indexing Integration

### 6.1 Modified IndexCodebaseTool.execute()

Update `/src/zapomni_mcp/tools/index_codebase.py`:

```python
async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Execute index_codebase tool with delta indexing support."""
    request_id = id(arguments)
    log = self.logger.bind(request_id=request_id)
    start_time = time.time()

    try:
        # Step 1: Validate arguments
        log.info("validating_arguments")
        (
            repo_path,
            languages,
            recursive,
            max_file_size,
            include_tests,
        ) = self._validate_arguments(arguments)

        # Get effective workspace
        effective_workspace_id = getattr(
            self.memory_processor, 'workspace_id', DEFAULT_WORKSPACE_ID
        )

        # NEW Step 2: Mark all existing code memories as stale
        log.info("marking_existing_memories_stale")
        stale_marked = await self.memory_processor.db_client.mark_code_memories_stale(
            workspace_id=effective_workspace_id
        )
        log.info("memories_marked_stale", count=stale_marked)

        # Step 3: Index repository
        log.info(
            "indexing_repository",
            repo_path=repo_path,
            languages=languages,
            recursive=recursive,
        )
        index_result = self.repository_indexer.index_repository(repo_path)

        # Step 4: Filter files
        files = index_result.get("files", [])
        files = self._filter_files(files, languages, include_tests, max_file_size)

        # Step 5: Calculate statistics
        stats = self._calculate_statistics(files, index_result)

        # NEW Step 6: For each file, mark as fresh or create new memory
        for file_dict in files:
            file_path = file_dict.get("path")

            # Try to mark existing memory as fresh
            existing_id = await self.memory_processor.db_client.mark_memory_fresh(
                file_path=file_path,
                workspace_id=effective_workspace_id,
            )

            if existing_id:
                # Memory exists and was marked fresh - could skip re-indexing
                # For MVP, we continue to re-index anyway
                log.debug("memory_marked_fresh", file_path=file_path)
            # else: new file, will be created

        # Store code as memories
        memories_created = 0
        # memories_created = await self._store_code_memories(files)

        # NEW Step 7: Count remaining stale memories
        stale_remaining = await self.memory_processor.db_client.count_stale_memories(
            workspace_id=effective_workspace_id
        )

        # Step 8: Format success response
        processing_time_ms = (time.time() - start_time) * 1000

        log.info(
            "repository_indexed_successfully",
            files_indexed=stats["files_indexed"],
            functions=stats["functions_found"],
            classes=stats["classes_found"],
            stale_memories=stale_remaining,
            processing_time_ms=processing_time_ms,
        )

        return self._format_success(
            repo_path=repo_path,
            files_indexed=stats["files_indexed"],
            functions_found=stats["functions_found"],
            classes_found=stats["classes_found"],
            languages=stats["languages"],
            total_lines=stats["total_lines"],
            processing_time_ms=processing_time_ms,
            stale_memories=stale_remaining,  # NEW parameter
        )

    except Exception as e:
        log.error("indexing_error", error=str(e), exc_info=True)
        return self._format_error(e)
```

### 6.2 Updated _format_success Method

```python
def _format_success(
    self,
    repo_path: str,
    files_indexed: int,
    functions_found: int,
    classes_found: int,
    languages: Dict[str, int],
    total_lines: int,
    processing_time_ms: float,
    stale_memories: int = 0,  # NEW parameter
) -> Dict[str, Any]:
    """Format successful indexing as MCP response."""
    # Format language summary
    lang_summary = (
        ", ".join(
            [f"{lang.capitalize()} ({count})" for lang, count in sorted(languages.items())]
        )
        or "None"
    )

    processing_time_sec = processing_time_ms / 1000.0

    message = (
        f"Repository indexed successfully.\n"
        f"Path: {repo_path}\n"
        f"Files indexed: {files_indexed}\n"
        f"Functions: {functions_found}\n"
        f"Classes: {classes_found}\n"
        f"Languages: {lang_summary}\n"
        f"Total lines: {total_lines:,}\n"
        f"Indexing time: {processing_time_sec:.2f}s"
    )

    # NEW: Add stale memories info if any
    if stale_memories > 0:
        message += f"\nStale memories detected: {stale_memories} (run prune_memory to clean up)"

    return {
        "content": [
            {
                "type": "text",
                "text": message,
            }
        ],
        "isError": False,
    }
```

---

## 7. Tool Registration

### 7.1 Update __init__.py

Update `/src/zapomni_mcp/tools/__init__.py`:

```python
from zapomni_mcp.tools.prune_memory import PruneMemoryTool

# Add to __all__
__all__ = [
    # ... existing tools ...
    "PruneMemoryTool",
]
```

### 7.2 Register in MCP Server

The MCP server registration follows existing patterns in the project.

---

## 8. Error Handling

### 8.1 Error Response Format

```python
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
```

### 8.2 Error Conditions

| Condition | Response |
|-----------|----------|
| Invalid strategy | `"Error: Invalid strategy 'xyz'. Valid: stale_code, orphaned_chunks, orphaned_entities, all"` |
| No confirmation | `"Error: Deletion requires explicit confirmation. Set confirm=true to proceed."` |
| Database error | `"Error: Prune operation failed: <database error message>"` |
| Validation error | `"Error: <pydantic validation message>"` |

---

## 9. Testing Strategy

### 9.1 Unit Test Cases

```python
# tests/unit/test_prune_memory_tool.py

class TestPruneMemoryTool:
    """Unit tests for PruneMemoryTool."""

    async def test_dry_run_returns_preview(self):
        """Dry run should return preview without deleting."""
        pass

    async def test_dry_run_is_default(self):
        """Default should be dry_run=true."""
        pass

    async def test_deletion_requires_confirmation(self):
        """Deletion without confirm=true should fail."""
        pass

    async def test_deletion_with_confirmation(self):
        """Deletion with dry_run=false and confirm=true should work."""
        pass

    async def test_stale_code_strategy(self):
        """stale_code strategy should only affect code memories."""
        pass

    async def test_orphaned_chunks_strategy(self):
        """orphaned_chunks strategy should find parentless chunks."""
        pass

    async def test_orphaned_entities_strategy(self):
        """orphaned_entities strategy should find unconnected entities."""
        pass

    async def test_all_strategy(self):
        """all strategy should run all cleanup operations."""
        pass

    async def test_workspace_isolation(self):
        """Operations should be scoped to workspace_id."""
        pass

    async def test_invalid_strategy_error(self):
        """Invalid strategy should return error."""
        pass
```

### 9.2 Integration Test Cases

```python
# tests/integration/test_gc_workflow.py

class TestGCWorkflow:
    """Integration tests for garbage collection workflow."""

    async def test_full_gc_workflow(self):
        """Test: index -> delete file -> re-index -> prune."""
        pass

    async def test_delta_indexing_marks_stale(self):
        """Re-indexing should mark removed files as stale."""
        pass

    async def test_prune_removes_stale_nodes(self):
        """Prune should remove stale nodes and their chunks."""
        pass

    async def test_stats_reflect_deletion(self):
        """get_stats should show decreased counts after prune."""
        pass

    async def test_no_broken_edges_after_prune(self):
        """Graph should have no broken edges after prune."""
        pass
```

---

## 10. Performance Considerations

### 10.1 Index Usage

The `memory_stale_idx` index enables efficient filtering:

```cypher
-- Index-assisted query
MATCH (m:Memory)
WHERE m.stale = true  -- Uses memory_stale_idx
  AND m.workspace_id = $workspace_id
```

### 10.2 Batch Operations

For large graphs, consider batch deletion:

```cypher
-- Batch delete (future enhancement)
MATCH (m:Memory)
WHERE m.source = 'code_indexer'
  AND m.workspace_id = $workspace_id
  AND m.stale = true
WITH m LIMIT 100
OPTIONAL MATCH (m)-[:HAS_CHUNK]->(c:Chunk)
DETACH DELETE m, c
RETURN count(m) AS deleted
```

### 10.3 Query Optimization

- Count queries use aggregation without fetching data
- Preview queries use LIMIT to bound results
- Deletion queries use DETACH DELETE for atomic cleanup

---

**Document History:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-11-25 | Specification Writer | Initial draft |
