# Workspace/Project Isolation - Technical Design

**Spec ID:** workspaces
**Version:** 1.0
**Date:** 2025-11-25
**Status:** Draft

---

## 1. Architecture Overview

### 1.1 High-Level Design

The workspace isolation feature uses **property-based filtering** with a `workspace_id` property on all data nodes. A hybrid resolution strategy combines explicit arguments with session state.

```
+------------------+     +------------------+     +------------------+
|   MCP Transport  | --> |   MCP Server     | --> |   MCP Tools      |
|  (stdio/SSE)     |     | + WorkspaceMgr   |     | + workspace_id   |
+------------------+     +------------------+     +------------------+
        |                       |                      |
        v                       v                      v
+------------------+     +------------------+     +------------------+
| (SSE only)       | <-- | SessionManager   |     | MemoryProcessor  |
| Workspace State  |     | + workspace_id   |     | + workspace_id   |
+------------------+     +------------------+     +------------------+
                                                        |
                                                        v
                                                 +------------------+
                                                 | FalkorDBClient   |
                                                 | + workspace_id   |
                                                 | + Cypher filters |
                                                 +------------------+
```

### 1.2 Design Principles

1. **Explicit Over Implicit**: Explicit `workspace_id` argument always wins
2. **Backward Compatible**: All parameters optional with sensible defaults
3. **Fail-Safe**: Missing workspace defaults to "default"
4. **Zero Breaking Changes**: Existing APIs work without modification

---

## 2. Database Schema Changes

### 2.1 Node Property Additions

All data nodes gain a `workspace_id` property:

```
+------------------+              +------------------+
|     Memory       |              |      Chunk       |
+------------------+              +------------------+
| id: string (PK)  |              | id: string (PK)  |
| workspace_id: *  | <-- NEW      | workspace_id: *  | <-- NEW
| text: string     |              | text: string     |
| tags: list       |              | index: int       |
| source: string   |              | embedding: vec   |
| created_at: ts   |              +------------------+
+------------------+                     ^
        |                                |
        | HAS_CHUNK                      |
        +--------------------------------+
        |
        | (via Chunk) MENTIONS
        v
+------------------+
|     Entity       |
+------------------+
| id: string (PK)  |
| workspace_id: *  | <-- NEW
| name: string     |
| type: string     |
| properties: dict |
+------------------+
```

### 2.2 Workspace Metadata Node

New `Workspace` node type for storing workspace metadata:

```cypher
(:Workspace {
    id: "project_a",           // Primary key, matches workspace_id in data nodes
    name: "Project A",         // Display name
    description: "...",        // Optional description
    metadata: "{}",            // JSON metadata
    created_at: "2025-11-25T10:00:00Z"
})
```

### 2.3 Cypher Queries with Workspace Filter

**Memory Creation:**
```cypher
CREATE (m:Memory {
    id: $memory_id,
    workspace_id: $workspace_id,
    text: $text,
    tags: $tags,
    source: $source,
    created_at: $timestamp
})
WITH m
UNWIND $chunks_data AS chunk_data
CREATE (c:Chunk {
    id: chunk_data.id,
    workspace_id: $workspace_id,
    text: chunk_data.text,
    index: chunk_data.index,
    embedding: vecf32(chunk_data.embedding)
})
CREATE (m)-[:HAS_CHUNK {index: chunk_data.index}]->(c)
RETURN m.id AS memory_id
```

**Vector Search with Workspace Filter:**
```cypher
CALL db.idx.vector.queryNodes('Chunk', 'embedding', $limit * 3, vecf32($query_embedding))
YIELD node AS c, score
WHERE c.workspace_id = $workspace_id
WITH c, score
MATCH (m:Memory {workspace_id: $workspace_id})-[:HAS_CHUNK]->(c)
WHERE score <= (1.0 - $min_similarity)
RETURN DISTINCT
    m.id AS memory_id,
    c.id AS chunk_id,
    c.text AS text,
    c.index AS chunk_index,
    (1.0 - score) AS similarity_score,
    m.tags AS tags,
    m.source AS source,
    m.created_at AS timestamp
ORDER BY score ASC
LIMIT $limit
```

**Entity Query with Workspace Filter:**
```cypher
MATCH (e:Entity {workspace_id: $workspace_id})
WHERE e.name = $name OR e.id = $id
OPTIONAL MATCH (e)-[r]-(related:Entity {workspace_id: $workspace_id})
RETURN e, type(r) AS relationship, related
```

---

## 3. WorkspaceManager Class Design

### 3.1 Class Definition

**File:** `src/zapomni_mcp/workspace_manager.py`

```python
"""
Workspace Manager for MCP Server.

Manages workspace context for sessions and provides workspace CRUD operations.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional, List, Any
import re
import json
import structlog

from zapomni_db import FalkorDBClient

logger = structlog.get_logger(__name__)

# Reserved workspace names that cannot be created
RESERVED_WORKSPACES = frozenset({"system", "admin", "test", "global"})

# Allowed workspace name: "default" or lowercase alphanumeric with hyphens/underscores
WORKSPACE_ID_PATTERN = re.compile(r'^[a-z0-9][a-z0-9_-]{0,62}$')


class WorkspaceValidationError(Exception):
    """Raised when workspace validation fails."""
    pass


class WorkspaceNotFoundError(Exception):
    """Raised when workspace does not exist."""
    pass


@dataclass
class WorkspaceInfo:
    """Workspace metadata."""
    id: str
    name: str
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    memory_count: int = 0
    entity_count: int = 0


class WorkspaceManager:
    """
    Manages workspace context and CRUD operations.

    Responsibilities:
    - Workspace context tracking per session
    - Workspace ID validation
    - Workspace CRUD via FalkorDBClient
    - Workspace resolution (explicit > session > default)
    """

    DEFAULT_WORKSPACE = "default"

    def __init__(self, db_client: FalkorDBClient):
        """
        Initialize WorkspaceManager.

        Args:
            db_client: FalkorDB client for database operations
        """
        self._db_client = db_client
        self._session_workspaces: Dict[str, str] = {}
        self._logger = logger.bind(component="WorkspaceManager")

    # ========== Validation ==========

    def validate_workspace_id(self, workspace_id: str) -> None:
        """
        Validate workspace ID format and reserved names.

        Args:
            workspace_id: Workspace ID to validate

        Raises:
            WorkspaceValidationError: If validation fails
        """
        if workspace_id in RESERVED_WORKSPACES:
            raise WorkspaceValidationError(
                f"Workspace name '{workspace_id}' is reserved. "
                f"Reserved names: {', '.join(sorted(RESERVED_WORKSPACES))}"
            )
        if not WORKSPACE_ID_PATTERN.match(workspace_id):
            raise WorkspaceValidationError(
                "Workspace ID must be 1-63 characters, lowercase alphanumeric, "
                "hyphens, or underscores. Must start with letter or number."
            )

    # ========== Session Context ==========

    def set_workspace(self, session_id: str, workspace_id: str) -> None:
        """
        Set current workspace for session.

        Args:
            session_id: MCP session ID
            workspace_id: Workspace ID to set
        """
        self._session_workspaces[session_id] = workspace_id
        self._logger.debug(
            "workspace_set",
            session_id=session_id,
            workspace_id=workspace_id
        )

    def get_workspace(self, session_id: Optional[str]) -> str:
        """
        Get current workspace for session.

        Args:
            session_id: MCP session ID (None for stdio mode)

        Returns:
            Workspace ID, defaults to "default" if not set
        """
        if session_id is None:
            return self.DEFAULT_WORKSPACE
        return self._session_workspaces.get(session_id, self.DEFAULT_WORKSPACE)

    def clear_session(self, session_id: str) -> None:
        """
        Clear workspace state for session (on disconnect).

        Args:
            session_id: MCP session ID to clear
        """
        removed = self._session_workspaces.pop(session_id, None)
        if removed:
            self._logger.debug(
                "session_cleared",
                session_id=session_id,
                workspace_id=removed
            )

    def resolve_workspace(
        self,
        explicit_workspace_id: Optional[str],
        session_id: Optional[str]
    ) -> str:
        """
        Resolve workspace using priority: explicit > session > default.

        Args:
            explicit_workspace_id: Explicitly provided workspace_id (highest priority)
            session_id: Session ID for looking up session state

        Returns:
            Resolved workspace ID
        """
        if explicit_workspace_id is not None:
            return explicit_workspace_id
        return self.get_workspace(session_id)

    # ========== CRUD Operations ==========

    async def create_workspace(
        self,
        workspace_id: str,
        name: Optional[str] = None,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> WorkspaceInfo:
        """
        Create new workspace.

        Args:
            workspace_id: Unique workspace identifier
            name: Display name (defaults to workspace_id)
            description: Optional description
            metadata: Optional metadata dict

        Returns:
            WorkspaceInfo for created workspace

        Raises:
            WorkspaceValidationError: If validation fails or workspace exists
        """
        self.validate_workspace_id(workspace_id)

        # Check if workspace already exists
        existing = await self._get_workspace_node(workspace_id)
        if existing:
            raise WorkspaceValidationError(
                f"Workspace '{workspace_id}' already exists"
            )

        created_at = datetime.now(timezone.utc)

        cypher = """
        CREATE (w:Workspace {
            id: $workspace_id,
            name: $name,
            description: $description,
            metadata: $metadata,
            created_at: $created_at
        })
        RETURN w
        """

        params = {
            "workspace_id": workspace_id,
            "name": name or workspace_id,
            "description": description,
            "metadata": json.dumps(metadata or {}),
            "created_at": created_at.isoformat()
        }

        await self._db_client.graph_query(cypher, params)

        self._logger.info(
            "workspace_created",
            workspace_id=workspace_id,
            name=name or workspace_id
        )

        return WorkspaceInfo(
            id=workspace_id,
            name=name or workspace_id,
            description=description,
            metadata=metadata or {},
            created_at=created_at
        )

    async def list_workspaces(self) -> List[WorkspaceInfo]:
        """
        List all workspaces with statistics.

        Returns:
            List of WorkspaceInfo objects
        """
        cypher = """
        MATCH (w:Workspace)
        OPTIONAL MATCH (m:Memory {workspace_id: w.id})
        OPTIONAL MATCH (e:Entity {workspace_id: w.id})
        RETURN
            w.id AS id,
            w.name AS name,
            w.description AS description,
            w.metadata AS metadata,
            w.created_at AS created_at,
            count(DISTINCT m) AS memory_count,
            count(DISTINCT e) AS entity_count
        ORDER BY w.created_at DESC
        """

        result = await self._db_client.graph_query(cypher)

        workspaces = []
        for row in result.result_set:
            workspaces.append(WorkspaceInfo(
                id=row[0],
                name=row[1],
                description=row[2] or "",
                metadata=json.loads(row[3]) if row[3] else {},
                created_at=datetime.fromisoformat(row[4]) if row[4] else None,
                memory_count=row[5],
                entity_count=row[6]
            ))

        return workspaces

    async def get_workspace(self, workspace_id: str) -> Optional[WorkspaceInfo]:
        """
        Get workspace by ID.

        Args:
            workspace_id: Workspace ID to retrieve

        Returns:
            WorkspaceInfo if found, None otherwise
        """
        cypher = """
        MATCH (w:Workspace {id: $workspace_id})
        OPTIONAL MATCH (m:Memory {workspace_id: w.id})
        OPTIONAL MATCH (e:Entity {workspace_id: w.id})
        RETURN
            w.id AS id,
            w.name AS name,
            w.description AS description,
            w.metadata AS metadata,
            w.created_at AS created_at,
            count(DISTINCT m) AS memory_count,
            count(DISTINCT e) AS entity_count
        """

        result = await self._db_client.graph_query(
            cypher,
            {"workspace_id": workspace_id}
        )

        if not result.result_set:
            return None

        row = result.result_set[0]
        return WorkspaceInfo(
            id=row[0],
            name=row[1],
            description=row[2] or "",
            metadata=json.loads(row[3]) if row[3] else {},
            created_at=datetime.fromisoformat(row[4]) if row[4] else None,
            memory_count=row[5],
            entity_count=row[6]
        )

    async def delete_workspace(
        self,
        workspace_id: str,
        cascade: bool = True
    ) -> Dict[str, Any]:
        """
        Delete workspace and optionally all its data.

        Args:
            workspace_id: Workspace to delete
            cascade: If True, delete all data in workspace (default: True)

        Returns:
            Dict with deletion statistics

        Raises:
            WorkspaceValidationError: If trying to delete "default" workspace
            WorkspaceNotFoundError: If workspace doesn't exist
        """
        if workspace_id == self.DEFAULT_WORKSPACE:
            raise WorkspaceValidationError(
                "Cannot delete the 'default' workspace"
            )

        # Check workspace exists
        existing = await self._get_workspace_node(workspace_id)
        if not existing:
            raise WorkspaceNotFoundError(
                f"Workspace '{workspace_id}' not found"
            )

        deleted_counts = {
            "memories": 0,
            "chunks": 0,
            "entities": 0
        }

        if cascade:
            # Delete all data in workspace
            # Delete chunks first (they reference memories)
            chunk_result = await self._db_client.graph_query(
                """
                MATCH (c:Chunk {workspace_id: $workspace_id})
                DETACH DELETE c
                RETURN count(c) AS count
                """,
                {"workspace_id": workspace_id}
            )
            deleted_counts["chunks"] = chunk_result.result_set[0][0] if chunk_result.result_set else 0

            # Delete memories
            memory_result = await self._db_client.graph_query(
                """
                MATCH (m:Memory {workspace_id: $workspace_id})
                DETACH DELETE m
                RETURN count(m) AS count
                """,
                {"workspace_id": workspace_id}
            )
            deleted_counts["memories"] = memory_result.result_set[0][0] if memory_result.result_set else 0

            # Delete entities
            entity_result = await self._db_client.graph_query(
                """
                MATCH (e:Entity {workspace_id: $workspace_id})
                DETACH DELETE e
                RETURN count(e) AS count
                """,
                {"workspace_id": workspace_id}
            )
            deleted_counts["entities"] = entity_result.result_set[0][0] if entity_result.result_set else 0

        # Delete workspace metadata node
        await self._db_client.graph_query(
            "MATCH (w:Workspace {id: $workspace_id}) DELETE w",
            {"workspace_id": workspace_id}
        )

        self._logger.info(
            "workspace_deleted",
            workspace_id=workspace_id,
            cascade=cascade,
            deleted_counts=deleted_counts
        )

        return {
            "workspace_id": workspace_id,
            "deleted": deleted_counts,
            "status": "deleted"
        }

    async def workspace_exists(self, workspace_id: str) -> bool:
        """
        Check if workspace exists.

        Args:
            workspace_id: Workspace ID to check

        Returns:
            True if workspace exists
        """
        # "default" always exists (virtual workspace)
        if workspace_id == self.DEFAULT_WORKSPACE:
            return True
        return await self._get_workspace_node(workspace_id) is not None

    async def ensure_default_workspace(self) -> None:
        """
        Ensure the default workspace exists.
        Creates it if it doesn't exist.
        """
        existing = await self._get_workspace_node(self.DEFAULT_WORKSPACE)
        if not existing:
            await self._db_client.graph_query(
                """
                CREATE (w:Workspace {
                    id: $workspace_id,
                    name: 'Default',
                    description: 'Default workspace',
                    metadata: '{}',
                    created_at: $created_at
                })
                """,
                {
                    "workspace_id": self.DEFAULT_WORKSPACE,
                    "created_at": datetime.now(timezone.utc).isoformat()
                }
            )
            self._logger.info("default_workspace_created")

    # ========== Private Helpers ==========

    async def _get_workspace_node(self, workspace_id: str) -> Optional[Dict]:
        """Get raw workspace node data."""
        result = await self._db_client.graph_query(
            "MATCH (w:Workspace {id: $workspace_id}) RETURN w",
            {"workspace_id": workspace_id}
        )
        if result.result_set:
            return result.result_set[0][0]
        return None
```

---

## 4. MCP Tools Interface Design

### 4.1 Tool: create_workspace

**File:** `src/zapomni_mcp/tools/create_workspace.py`

```python
class CreateWorkspaceTool(BaseTool):
    """MCP tool to create a new workspace."""

    name = "create_workspace"
    description = "Create a new workspace for organizing memories by project"

    input_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Workspace ID (lowercase alphanumeric, hyphens, underscores, 1-63 chars)",
                "pattern": "^[a-z0-9][a-z0-9_-]{0,62}$"
            },
            "description": {
                "type": "string",
                "description": "Optional description of the workspace",
                "default": ""
            }
        },
        "required": ["name"]
    }

    async def execute(self, arguments: dict) -> dict:
        workspace_id = arguments["name"]
        description = arguments.get("description", "")

        result = await self._workspace_manager.create_workspace(
            workspace_id=workspace_id,
            name=workspace_id,
            description=description
        )

        return {
            "content": [{
                "type": "text",
                "text": f"Created workspace: {workspace_id}"
            }],
            "workspace_id": result.id,
            "created_at": result.created_at.isoformat()
        }
```

### 4.2 Tool: list_workspaces

```python
class ListWorkspacesTool(BaseTool):
    """MCP tool to list all workspaces."""

    name = "list_workspaces"
    description = "List all available workspaces with memory counts"

    input_schema = {
        "type": "object",
        "properties": {},
        "required": []
    }

    async def execute(self, arguments: dict) -> dict:
        workspaces = await self._workspace_manager.list_workspaces()

        workspace_list = [
            {
                "id": w.id,
                "name": w.name,
                "description": w.description,
                "memory_count": w.memory_count,
                "entity_count": w.entity_count,
                "created_at": w.created_at.isoformat() if w.created_at else None
            }
            for w in workspaces
        ]

        return {
            "content": [{
                "type": "text",
                "text": f"Found {len(workspaces)} workspace(s)"
            }],
            "workspaces": workspace_list
        }
```

### 4.3 Tool: set_current_workspace

```python
class SetCurrentWorkspaceTool(BaseTool):
    """MCP tool to switch the current workspace for this session."""

    name = "set_current_workspace"
    description = "Switch to a different workspace for this session. Subsequent operations will target this workspace."

    input_schema = {
        "type": "object",
        "properties": {
            "workspace_id": {
                "type": "string",
                "description": "Workspace ID to switch to",
                "pattern": "^[a-z0-9][a-z0-9_-]{0,62}$"
            }
        },
        "required": ["workspace_id"]
    }

    async def execute(self, arguments: dict, session_id: str = None) -> dict:
        workspace_id = arguments["workspace_id"]

        # Verify workspace exists (or is "default")
        exists = await self._workspace_manager.workspace_exists(workspace_id)
        if not exists:
            raise ValueError(f"Workspace '{workspace_id}' not found")

        # Set session workspace
        if session_id:
            self._workspace_manager.set_workspace(session_id, workspace_id)

        return {
            "content": [{
                "type": "text",
                "text": f"Switched to workspace: {workspace_id}"
            }],
            "workspace_id": workspace_id,
            "session_id": session_id
        }
```

### 4.4 Tool: delete_workspace

```python
class DeleteWorkspaceTool(BaseTool):
    """MCP tool to delete a workspace and all its data."""

    name = "delete_workspace"
    description = "Delete a workspace and all its memories, chunks, and entities"

    input_schema = {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "Workspace ID to delete"
            }
        },
        "required": ["id"]
    }

    async def execute(self, arguments: dict) -> dict:
        workspace_id = arguments["id"]

        result = await self._workspace_manager.delete_workspace(
            workspace_id=workspace_id,
            cascade=True  # Always cascade for simplicity
        )

        return {
            "content": [{
                "type": "text",
                "text": f"Deleted workspace: {workspace_id}"
            }],
            "deleted": result["deleted"],
            "status": result["status"]
        }
```

### 4.5 Tool: get_current_workspace

```python
class GetCurrentWorkspaceTool(BaseTool):
    """MCP tool to get the current workspace for this session."""

    name = "get_current_workspace"
    description = "Get the current workspace ID for this session"

    input_schema = {
        "type": "object",
        "properties": {},
        "required": []
    }

    async def execute(self, arguments: dict, session_id: str = None) -> dict:
        workspace_id = self._workspace_manager.get_workspace(session_id)

        return {
            "content": [{
                "type": "text",
                "text": f"Current workspace: {workspace_id}"
            }],
            "workspace_id": workspace_id
        }
```

### 4.6 Updated Existing Tools

All existing memory tools gain an optional `workspace_id` parameter:

```python
# In add_memory.py, search_memory.py, etc.
input_schema = {
    "type": "object",
    "properties": {
        # ... existing properties ...
        "workspace_id": {
            "type": "string",
            "description": "Target workspace ID. If not specified, uses session workspace or 'default'.",
            "pattern": "^[a-z0-9][a-z0-9_-]{0,62}$"
        }
    }
}
```

---

## 5. Session Integration Design

### 5.1 SessionState Extension

**File:** `src/zapomni_mcp/session_manager.py`

Add `workspace_id` to existing `SessionState`:

```python
@dataclass
class SessionState:
    """State for an MCP session."""
    session_id: str
    transport: str  # "sse" or "stdio"
    created_at: float
    last_activity: float
    request_count: int = 0
    workspace_id: str = "default"  # <-- NEW FIELD
```

### 5.2 MCP Server Integration

**File:** `src/zapomni_mcp/server.py`

Modify `handle_call_tool()` to inject workspace context:

```python
class MCPServer:
    def __init__(self, ...):
        # ... existing init ...
        self._workspace_manager = WorkspaceManager(db_client)

    async def _handle_call_tool(
        self,
        name: str,
        arguments: dict,
        session_id: Optional[str] = None
    ) -> list:
        """Handle tool call with workspace context injection."""

        # Resolve workspace: explicit > session > default
        explicit_workspace = arguments.get("workspace_id")
        resolved_workspace = self._workspace_manager.resolve_workspace(
            explicit_workspace_id=explicit_workspace,
            session_id=session_id
        )

        # Inject resolved workspace into arguments
        arguments["workspace_id"] = resolved_workspace
        arguments["_session_id"] = session_id  # For tools that need session context

        # Execute tool
        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"Unknown tool: {name}")

        result = await tool.execute(arguments)
        return result.get("content", [])
```

---

## 6. Migration Design

### 6.1 Migration Script

**File:** `scripts/migrate_to_workspaces.py`

```python
#!/usr/bin/env python3
"""
Migration script to add workspace_id to existing data.

Usage:
    python scripts/migrate_to_workspaces.py [--batch-size 10000] [--dry-run]
"""
import asyncio
import argparse
from datetime import datetime, timezone

from zapomni_db import FalkorDBClient


async def migrate_nodes(
    db: FalkorDBClient,
    label: str,
    batch_size: int,
    dry_run: bool
) -> int:
    """Migrate nodes of given label to default workspace."""

    total_migrated = 0

    while True:
        # Find nodes without workspace_id
        count_query = f"""
        MATCH (n:{label})
        WHERE n.workspace_id IS NULL
        RETURN count(n) AS remaining
        """
        result = await db.graph_query(count_query)
        remaining = result.result_set[0][0] if result.result_set else 0

        if remaining == 0:
            break

        print(f"  {label}: {remaining} nodes remaining...")

        if dry_run:
            total_migrated += remaining
            break

        # Migrate batch
        migrate_query = f"""
        MATCH (n:{label})
        WHERE n.workspace_id IS NULL
        WITH n LIMIT $batch_size
        SET n.workspace_id = 'default'
        RETURN count(n) AS migrated
        """
        result = await db.graph_query(migrate_query, {"batch_size": batch_size})
        migrated = result.result_set[0][0] if result.result_set else 0
        total_migrated += migrated

        if migrated == 0:
            break

    return total_migrated


async def ensure_default_workspace(db: FalkorDBClient, dry_run: bool) -> None:
    """Ensure default workspace exists."""

    result = await db.graph_query(
        "MATCH (w:Workspace {id: 'default'}) RETURN w"
    )

    if result.result_set:
        print("Default workspace already exists")
        return

    if dry_run:
        print("Would create default workspace")
        return

    await db.graph_query(
        """
        CREATE (w:Workspace {
            id: 'default',
            name: 'Default',
            description: 'Default workspace for pre-migration data',
            metadata: '{}',
            created_at: $created_at
        })
        """,
        {"created_at": datetime.now(timezone.utc).isoformat()}
    )
    print("Created default workspace")


async def main():
    parser = argparse.ArgumentParser(
        description="Migrate existing data to workspaces"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Number of nodes to process per batch"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without making changes"
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="FalkorDB host"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6379,
        help="FalkorDB port"
    )

    args = parser.parse_args()

    print(f"Workspace Migration Script")
    print(f"=" * 40)
    print(f"Batch size: {args.batch_size}")
    print(f"Dry run: {args.dry_run}")
    print()

    # Connect to database
    db = FalkorDBClient(host=args.host, port=args.port)
    await db.init_async()

    try:
        # Ensure default workspace exists
        print("Step 1: Ensure default workspace exists")
        await ensure_default_workspace(db, args.dry_run)
        print()

        # Migrate each node type
        print("Step 2: Migrate nodes to default workspace")

        labels = ["Memory", "Chunk", "Entity"]
        total = 0

        for label in labels:
            count = await migrate_nodes(db, label, args.batch_size, args.dry_run)
            total += count
            print(f"  {label}: {count} nodes {'would be ' if args.dry_run else ''}migrated")

        print()
        print(f"Total: {total} nodes {'would be ' if args.dry_run else ''}migrated")

        if args.dry_run:
            print()
            print("This was a dry run. No changes were made.")
            print("Run without --dry-run to apply changes.")

    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
```

### 6.2 Rollback Script

**File:** `scripts/rollback_workspaces.py`

```python
#!/usr/bin/env python3
"""
Rollback script to remove workspace_id from all nodes.

WARNING: This removes workspace isolation. Use with caution.
"""
import asyncio
from zapomni_db import FalkorDBClient


async def rollback():
    db = FalkorDBClient()
    await db.init_async()

    try:
        for label in ["Memory", "Chunk", "Entity"]:
            result = await db.graph_query(
                f"MATCH (n:{label}) REMOVE n.workspace_id RETURN count(n)"
            )
            count = result.result_set[0][0] if result.result_set else 0
            print(f"Removed workspace_id from {count} {label} nodes")

        # Remove workspace metadata nodes
        result = await db.graph_query(
            "MATCH (w:Workspace) DELETE w RETURN count(w)"
        )
        count = result.result_set[0][0] if result.result_set else 0
        print(f"Deleted {count} Workspace nodes")

    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(rollback())
```

---

## 7. Testing Strategy

### 7.1 Unit Tests

```python
# tests/unit/test_workspace_manager.py

class TestWorkspaceManager:

    def test_validate_workspace_id_valid(self):
        """Valid workspace IDs should pass validation."""
        manager = WorkspaceManager(mock_db)

        # Should not raise
        manager.validate_workspace_id("project-a")
        manager.validate_workspace_id("project_b")
        manager.validate_workspace_id("123project")
        manager.validate_workspace_id("default")

    def test_validate_workspace_id_reserved(self):
        """Reserved names should be rejected."""
        manager = WorkspaceManager(mock_db)

        with pytest.raises(WorkspaceValidationError):
            manager.validate_workspace_id("system")
        with pytest.raises(WorkspaceValidationError):
            manager.validate_workspace_id("admin")

    def test_resolve_workspace_priority(self):
        """Explicit argument should take priority over session."""
        manager = WorkspaceManager(mock_db)
        manager.set_workspace("session1", "project_a")

        # Explicit wins
        assert manager.resolve_workspace("project_b", "session1") == "project_b"

        # Session if no explicit
        assert manager.resolve_workspace(None, "session1") == "project_a"

        # Default if neither
        assert manager.resolve_workspace(None, "unknown_session") == "default"
```

### 7.2 Integration Tests

```python
# tests/integration/test_workspace_isolation.py

@pytest.mark.asyncio
async def test_workspace_isolation():
    """Memories in different workspaces should be isolated."""

    # Add memory to workspace A
    await processor.add_memory(
        text="Workspace A content about Python",
        workspace_id="workspace_a"
    )

    # Add memory to workspace B
    await processor.add_memory(
        text="Workspace B content about Python",
        workspace_id="workspace_b"
    )

    # Search in workspace A
    results_a = await processor.search_memory(
        query="Python",
        workspace_id="workspace_a"
    )
    assert len(results_a) == 1
    assert "Workspace A" in results_a[0].text

    # Search in workspace B
    results_b = await processor.search_memory(
        query="Python",
        workspace_id="workspace_b"
    )
    assert len(results_b) == 1
    assert "Workspace B" in results_b[0].text
```

---

## 8. Error Handling

### 8.1 Error Types

| Error | HTTP Code | Description |
|-------|-----------|-------------|
| `WorkspaceValidationError` | 400 | Invalid workspace ID format or reserved name |
| `WorkspaceNotFoundError` | 404 | Workspace does not exist |
| `WorkspaceDeleteError` | 400 | Cannot delete default workspace |

### 8.2 Error Messages

```python
ERROR_MESSAGES = {
    "invalid_format": "Workspace ID must be 1-63 characters, lowercase alphanumeric, hyphens, or underscores. Must start with letter or number.",
    "reserved_name": "Workspace name '{name}' is reserved. Reserved names: system, admin, test, global",
    "already_exists": "Workspace '{name}' already exists",
    "not_found": "Workspace '{name}' not found",
    "cannot_delete_default": "Cannot delete the 'default' workspace"
}
```

---

## 9. Configuration

No new configuration options required. All defaults are:
- Default workspace: `"default"`
- Session storage: In-memory
- Cascade delete: Always on

Future configuration options (not in scope):
- Redis session storage URL
- Workspace quotas
- Default workspace name override

---

**Prepared by:** Specification Agent
**Reviewed by:** (pending)
**Approved by:** (pending)
