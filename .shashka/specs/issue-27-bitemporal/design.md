# Issue #27: Bi-temporal Model - Technical Design

## 1. Data Model Changes

### 1.1 Memory Node Schema

```python
# src/zapomni_db/models.py

class Memory(BaseModel):
    """Memory node with bi-temporal support."""

    # Existing fields
    id: str                           # UUID
    text: str
    tags: List[str] = []
    source: str = "user"
    metadata: Optional[str] = None    # JSON string
    workspace_id: str
    file_path: Optional[str] = None
    qualified_name: Optional[str] = None

    # Transaction Time (when recorded in DB)
    created_at: str                   # EXISTING - transaction time start
    transaction_to: Optional[str] = None  # NEW - NULL = current

    # Valid Time (when true in reality)
    valid_from: str                   # NEW - git commit date or created_at
    valid_to: Optional[str] = None    # NEW - NULL = still valid

    # Version Control
    version: int = 1                  # NEW - version number
    previous_version_id: Optional[str] = None  # NEW - link to previous

    # Optimization
    is_current: bool = True           # NEW - fast filter

    # GC fields (keep for compatibility)
    stale: bool = False
    last_seen_at: Optional[str] = None
```

### 1.2 Entity Node Schema

```python
class Entity(BaseModel):
    """Entity node with bi-temporal support."""

    # Existing fields
    id: str
    name: str
    type: str
    description: Optional[str] = None
    confidence: float = 1.0
    workspace_id: Optional[str] = None

    # Temporal fields
    created_at: str
    updated_at: Optional[str] = None
    valid_from: str                   # NEW
    valid_to: Optional[str] = None    # NEW
    is_current: bool = True           # NEW
```

## 2. Database Indexes

### 2.1 New Indexes (schema_manager.py)

```python
TEMPORAL_INDEXES = [
    # Fast current state queries (most common)
    IndexDef("memory_current_idx", "Memory", ["workspace_id", "is_current"]),

    # Valid time range queries
    IndexDef("memory_valid_from_idx", "Memory", ["valid_from"]),
    IndexDef("memory_valid_range_idx", "Memory", ["workspace_id", "valid_from", "valid_to"]),

    # File history queries
    IndexDef("memory_file_history_idx", "Memory", ["file_path", "workspace_id", "valid_from"]),

    # Version chain traversal
    IndexDef("memory_version_idx", "Memory", ["previous_version_id"]),

    # Entity temporal
    IndexDef("entity_current_idx", "Entity", ["workspace_id", "is_current"]),
]
```

### 2.2 Cypher Index Creation

```cypher
CREATE INDEX memory_current_idx IF NOT EXISTS
FOR (m:Memory) ON (m.workspace_id, m.is_current)

CREATE INDEX memory_valid_from_idx IF NOT EXISTS
FOR (m:Memory) ON (m.valid_from)

CREATE INDEX memory_file_history_idx IF NOT EXISTS
FOR (m:Memory) ON (m.file_path, m.workspace_id, m.valid_from)

CREATE INDEX memory_version_idx IF NOT EXISTS
FOR (m:Memory) ON (m.previous_version_id)
```

## 3. Query Patterns

### 3.1 Current State (Default)

```cypher
// Most common query - optimized with is_current index
MATCH (m:Memory)
WHERE m.workspace_id = $workspace_id
  AND m.is_current = true
RETURN m
```

### 3.2 Point-in-Time Valid

```cypher
// What was the state at time T in reality?
MATCH (m:Memory)
WHERE m.workspace_id = $workspace_id
  AND m.file_path = $file_path
  AND m.valid_from <= $point_in_time
  AND (m.valid_to IS NULL OR m.valid_to > $point_in_time)
RETURN m
ORDER BY m.valid_from DESC
LIMIT 1
```

### 3.3 Point-in-Time Transaction

```cypher
// What did we know at time T?
MATCH (m:Memory)
WHERE m.workspace_id = $workspace_id
  AND m.file_path = $file_path
  AND m.created_at <= $point_in_time
  AND (m.transaction_to IS NULL OR m.transaction_to > $point_in_time)
RETURN m
ORDER BY m.created_at DESC
LIMIT 1
```

### 3.4 Full Bi-temporal

```cypher
// What did we know about state X at time Y?
MATCH (m:Memory)
WHERE m.workspace_id = $workspace_id
  AND m.valid_from <= $valid_time
  AND (m.valid_to IS NULL OR m.valid_to > $valid_time)
  AND m.created_at <= $transaction_time
  AND (m.transaction_to IS NULL OR m.transaction_to > $transaction_time)
RETURN m
```

### 3.5 Version History

```cypher
// Get all versions of a file
MATCH (m:Memory)
WHERE m.file_path = $file_path
  AND m.workspace_id = $workspace_id
RETURN m.id, m.version, m.valid_from, m.valid_to,
       m.created_at, m.transaction_to, m.is_current
ORDER BY m.valid_from DESC
```

### 3.6 Changes in Time Range

```cypher
// Get changes since timestamp
MATCH (m:Memory)
WHERE m.workspace_id = $workspace_id
  AND m.source = 'code_indexer'
  AND m.created_at >= $since
RETURN m.id, m.file_path, m.version, m.created_at,
       CASE
         WHEN m.version = 1 THEN 'created'
         ELSE 'modified'
       END AS change_type
ORDER BY m.created_at DESC
LIMIT $limit
```

## 4. FalkorDBClient Methods

### 4.1 New Methods

```python
class FalkorDBClient:

    async def get_memory_at_time(
        self,
        workspace_id: str,
        file_path: str,
        as_of: datetime,
        time_type: Literal["valid", "transaction", "both"] = "valid"
    ) -> Optional[Memory]:
        """Get memory state at a specific point in time."""

    async def get_memory_history(
        self,
        workspace_id: str,
        file_path: Optional[str] = None,
        entity_id: Optional[str] = None,
        limit: int = 50
    ) -> List[Memory]:
        """Get version history of a memory."""

    async def get_changes(
        self,
        workspace_id: str,
        since: datetime,
        until: Optional[datetime] = None,
        change_type: Optional[str] = None,
        path_pattern: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get changes in time range."""

    async def close_version(
        self,
        memory_id: str,
        valid_to: datetime,
        transaction_to: datetime
    ) -> bool:
        """Close a memory version (mark as superseded)."""

    async def create_new_version(
        self,
        previous: Memory,
        new_content: str,
        valid_from: datetime
    ) -> Memory:
        """Create a new version of a memory."""
```

## 5. Git Integration

### 5.1 GitTemporalExtractor

```python
# src/zapomni_core/temporal/git_temporal.py

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import asyncio

@dataclass
class GitVersion:
    commit_hash: str
    commit_date: datetime
    author: str
    message: str

class GitTemporalExtractor:
    """Extract temporal information from git history."""

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self._git_available: Optional[bool] = None

    async def is_git_repo(self) -> bool:
        """Check if directory is a git repository."""

    async def get_commit_date(self, file_path: str) -> Optional[datetime]:
        """Get the commit date for current file version.

        Returns None if:
        - Not a git repo
        - File not tracked
        - Git command fails
        """

    async def get_file_history(
        self,
        file_path: str,
        limit: int = 100
    ) -> List[GitVersion]:
        """Get all versions of a file from git log."""

    async def get_file_at_commit(
        self,
        file_path: str,
        commit: str
    ) -> Optional[str]:
        """Get file content at specific commit."""
```

### 5.2 Fallback Strategy

```python
async def get_valid_from(
    self,
    file_path: str,
    git_extractor: GitTemporalExtractor
) -> datetime:
    """Get valid_from timestamp with fallbacks."""

    # 1. Try git commit date
    git_date = await git_extractor.get_commit_date(file_path)
    if git_date:
        return git_date

    # 2. Fallback to file mtime
    try:
        stat = Path(file_path).stat()
        return datetime.fromtimestamp(stat.st_mtime)
    except OSError:
        pass

    # 3. Final fallback to current time
    return datetime.now()
```

## 6. MCP Tools Design

### 6.1 get_timeline Tool

```python
# src/zapomni_mcp/tools/get_timeline.py

class GetTimelineTool:
    name = "get_timeline"
    description = """
    Get the history of changes for a specific entity (file, function, class).
    Shows all versions with their valid time periods and changes.
    Useful for understanding how code evolved over time.
    """

    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file"
            },
            "qualified_name": {
                "type": "string",
                "description": "Qualified name (e.g., module.Class.method)"
            },
            "entity_id": {
                "type": "string",
                "description": "Entity ID (UUID)"
            },
            "limit": {
                "type": "integer",
                "default": 20,
                "minimum": 1,
                "maximum": 100
            },
            "workspace_id": {
                "type": "string",
                "description": "Workspace ID (optional)"
            }
        },
        "anyOf": [
            {"required": ["file_path"]},
            {"required": ["qualified_name"]},
            {"required": ["entity_id"]}
        ]
    }
```

### 6.2 get_changes Tool

```python
# src/zapomni_mcp/tools/get_changes.py

class GetChangesTool:
    name = "get_changes"
    description = """
    Get changes in the codebase within a time range.
    Shows created, modified, and deleted files/entities.
    Supports filtering by change type and path pattern.
    """

    input_schema = {
        "type": "object",
        "properties": {
            "since": {
                "type": "string",
                "format": "date-time",
                "description": "Start of time range (ISO 8601)"
            },
            "until": {
                "type": "string",
                "format": "date-time",
                "description": "End of time range (optional, default: now)"
            },
            "change_type": {
                "type": "string",
                "enum": ["created", "modified", "deleted", "all"],
                "default": "all"
            },
            "path_pattern": {
                "type": "string",
                "description": "Glob pattern (e.g., 'src/**/*.py')"
            },
            "limit": {
                "type": "integer",
                "default": 100,
                "minimum": 1,
                "maximum": 500
            },
            "workspace_id": {
                "type": "string"
            }
        },
        "required": ["since"]
    }
```

### 6.3 get_snapshot Tool

```python
# src/zapomni_mcp/tools/get_snapshot.py

class GetSnapshotTool:
    name = "get_snapshot"
    description = """
    Get the state of the codebase at a specific point in time.
    Time-travel query for debugging and analysis.
    Supports both valid time and transaction time dimensions.
    """

    input_schema = {
        "type": "object",
        "properties": {
            "as_of": {
                "type": "string",
                "format": "date-time",
                "description": "Point in time (ISO 8601)"
            },
            "time_type": {
                "type": "string",
                "enum": ["valid", "transaction"],
                "default": "valid",
                "description": "Which time dimension to query"
            },
            "path_pattern": {
                "type": "string",
                "description": "Filter by path pattern"
            },
            "limit": {
                "type": "integer",
                "default": 100
            },
            "workspace_id": {
                "type": "string"
            }
        },
        "required": ["as_of"]
    }
```

## 7. Migration Strategy

### 7.1 Migration Script

```python
# src/zapomni_db/migrations/001_add_bitemporal.py

async def migrate_to_bitemporal(db: FalkorDBClient) -> MigrationResult:
    """
    Add bi-temporal fields to existing data.
    This migration is idempotent - safe to run multiple times.
    """

    result = MigrationResult()

    # Step 1: Add temporal fields to Memory nodes
    memory_count = await db.execute("""
        MATCH (m:Memory)
        WHERE m.valid_from IS NULL
        SET m.valid_from = m.created_at,
            m.valid_to = null,
            m.transaction_to = null,
            m.version = 1,
            m.previous_version_id = null,
            m.is_current = true
        RETURN count(m) AS count
    """)
    result.memories_migrated = memory_count

    # Step 2: Add temporal fields to Entity nodes
    entity_count = await db.execute("""
        MATCH (e:Entity)
        WHERE e.valid_from IS NULL
        SET e.valid_from = coalesce(e.created_at, datetime()),
            e.valid_to = null,
            e.is_current = true
        RETURN count(e) AS count
    """)
    result.entities_migrated = entity_count

    # Step 3: Create new indexes
    await create_temporal_indexes(db)

    # Step 4: Update schema version
    await db.set_schema_version("2.0.0")

    return result
```

### 7.2 Index Creation

```python
async def create_temporal_indexes(db: FalkorDBClient) -> None:
    """Create indexes for temporal queries."""

    indexes = [
        "CREATE INDEX memory_current_idx IF NOT EXISTS FOR (m:Memory) ON (m.workspace_id, m.is_current)",
        "CREATE INDEX memory_valid_from_idx IF NOT EXISTS FOR (m:Memory) ON (m.valid_from)",
        "CREATE INDEX memory_file_history_idx IF NOT EXISTS FOR (m:Memory) ON (m.file_path, m.workspace_id, m.valid_from)",
        "CREATE INDEX memory_version_idx IF NOT EXISTS FOR (m:Memory) ON (m.previous_version_id)",
        "CREATE INDEX entity_current_idx IF NOT EXISTS FOR (e:Entity) ON (e.workspace_id, e.is_current)",
    ]

    for index_query in indexes:
        await db.execute(index_query)
```

## 8. Performance Considerations

### 8.1 Query Optimization

1. **is_current index** - Most queries (99%) are for current state
2. **Composite indexes** - For common query patterns
3. **LIMIT clauses** - Always limit history queries

### 8.2 Storage Optimization

1. **Anchor+Delta strategy** (future) - Periodically create full snapshots
2. **Compaction** (future) - Merge old versions to reduce storage
3. **Retention policy** (future) - Auto-delete versions older than X

### 8.3 Benchmarks to Track

| Query Type | Target | Acceptable |
|------------|--------|------------|
| Current state | < 10ms | < 50ms |
| Point-in-time | < 50ms | < 200ms |
| History (20 versions) | < 100ms | < 500ms |
| Changes (100 items) | < 200ms | < 1s |
