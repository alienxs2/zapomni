"""
Data models for zapomni_db module.

Defines Pydantic models for Memory, Chunk, SearchResult, Workspace, etc.

Bi-temporal Model Support (v0.8.0):
- Valid Time: When the fact was true in reality (valid_from, valid_to)
- Transaction Time: When we recorded the fact (created_at, transaction_to)

See: .shashka/specs/issue-27-bitemporal/ for full specification.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

# Default workspace ID - used when no workspace is specified
DEFAULT_WORKSPACE_ID = "default"


class Chunk(BaseModel):
    """Text chunk with positional metadata."""

    text: str = Field(..., min_length=1)
    index: int = Field(..., ge=0)
    start_char: Optional[int] = Field(default=None, ge=0)
    end_char: Optional[int] = Field(default=None, ge=0)
    metadata: Optional[Dict[str, Any]] = None
    workspace_id: str = Field(default=DEFAULT_WORKSPACE_ID)


class ChunkData(BaseModel):
    """Chunk with embedding vector."""

    text: str = Field(..., min_length=1)
    index: int = Field(..., ge=0)
    start_char: Optional[int] = Field(default=None, ge=0)
    end_char: Optional[int] = Field(default=None, ge=0)
    embedding: List[float] = Field(..., min_length=1)
    metadata: Optional[Dict[str, Any]] = None
    workspace_id: str = Field(default=DEFAULT_WORKSPACE_ID)


class Memory(BaseModel):
    """Complete memory with chunks, embeddings, and metadata."""

    text: str = Field(..., min_length=1, max_length=1_000_000)
    chunks: List[Chunk] = Field(..., min_length=1, max_length=100)
    embeddings: List[List[float]] = Field(..., min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    workspace_id: str = Field(default=DEFAULT_WORKSPACE_ID)

    @field_validator("chunks")
    @classmethod
    def validate_chunks(cls, v: List[str]) -> List[str]:
        """Ensure chunks are not empty."""
        if not v:
            raise ValueError("chunks list cannot be empty")
        return v

    @field_validator("embeddings")
    @classmethod
    def validate_embeddings(cls, v: List[List[float]]) -> List[List[float]]:
        """Ensure embeddings are not empty."""
        if not v:
            raise ValueError("embeddings list cannot be empty")
        return v


@dataclass
class SearchResult:
    """Single search result from vector similarity search."""

    memory_id: str
    content: str
    relevance_score: float
    metadata: Optional[Dict[str, Any]] = None
    workspace_id: str = DEFAULT_WORKSPACE_ID
    # Legacy fields for backward compatibility
    chunk_id: Optional[str] = None
    text: Optional[str] = None
    similarity_score: Optional[float] = None
    tags: Optional[List[str]] = None
    source: Optional[str] = None
    timestamp: Optional[datetime] = None
    chunk_index: Optional[int] = None


@dataclass
class QueryResult:
    """Result from Cypher query execution."""

    rows: List[Dict[str, Any]]
    row_count: int
    execution_time_ms: int


class Entity(BaseModel):
    """
    Entity node in knowledge graph.

    Bi-temporal Support (v0.8.0):
    - valid_from: When entity appeared in reality
    - valid_to: When entity ceased to exist (None = current)
    - is_current: Fast filter for current version
    """

    name: str = Field(..., min_length=1)
    type: str = Field(..., min_length=1)
    description: Optional[str] = Field(default="")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    workspace_id: str = Field(default=DEFAULT_WORKSPACE_ID)

    # Bi-temporal fields (v0.8.0)
    valid_from: Optional[str] = Field(
        default=None,
        description="When entity appeared in reality (ISO 8601 timestamp)"
    )
    valid_to: Optional[str] = Field(
        default=None,
        description="When entity ceased to exist (None = still valid)"
    )
    is_current: bool = Field(
        default=True,
        description="True if this is the current version"
    )


class Relationship(BaseModel):
    """Relationship edge in knowledge graph."""

    from_entity_id: str = Field(..., min_length=1)
    to_entity_id: str = Field(..., min_length=1)
    relationship_type: str = Field(..., min_length=1)
    strength: float = Field(default=1.0, ge=0.0, le=1.0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    context: Optional[str] = None
    workspace_id: str = Field(default=DEFAULT_WORKSPACE_ID)


class MemoryResult(BaseModel):
    """Result of memory creation operation."""

    id: str = Field(..., min_length=1)
    chunks_created: int = Field(..., ge=0)
    processing_time_ms: float = Field(..., gt=0)
    workspace_id: str = Field(default=DEFAULT_WORKSPACE_ID)


@dataclass
class Workspace:
    """Workspace for data isolation."""

    id: str
    name: str
    description: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkspaceStats:
    """Statistics for a specific workspace."""

    workspace_id: str
    total_memories: int = 0
    total_chunks: int = 0
    total_entities: int = 0
    total_relationships: int = 0


# =============================================================================
# Bi-temporal Models (v0.8.0 - Issue #27)
# =============================================================================


class MemoryVersion(BaseModel):
    """
    Memory node with full bi-temporal support.

    This model represents a versioned memory with two temporal dimensions:
    - Transaction Time: When the record was created/superseded in the database
    - Valid Time: When the content was true in reality (e.g., git commit date)

    Use Cases:
    - Time-travel debugging: Query code state at any point in time
    - Historical queries: "What changed in the last week?"
    - Version tracking: Full history of each file/function

    Example:
        ```python
        memory = MemoryVersion(
            id="uuid-123",
            text="def hello(): pass",
            file_path="src/main.py",
            workspace_id="default",
            # Transaction time (when recorded)
            created_at="2025-12-01T10:00:00Z",
            transaction_to=None,  # Still current in DB
            # Valid time (when true in reality)
            valid_from="2025-11-30T15:30:00Z",  # Git commit date
            valid_to=None,  # Still valid
            # Version info
            version=2,
            previous_version_id="uuid-122",
            is_current=True,
        )
        ```
    """

    # Identity
    id: str = Field(..., min_length=1, description="UUID of this memory version")

    # Content
    text: str = Field(..., min_length=1, max_length=1_000_000)
    tags: List[str] = Field(default_factory=list)
    source: str = Field(default="user")
    metadata: Optional[str] = Field(default=None, description="JSON string of metadata")
    workspace_id: str = Field(default=DEFAULT_WORKSPACE_ID)

    # Code-specific fields
    file_path: Optional[str] = Field(default=None, description="Absolute file path")
    qualified_name: Optional[str] = Field(
        default=None,
        description="Fully qualified name (e.g., module.Class.method)"
    )

    # Transaction Time (when recorded in database)
    created_at: str = Field(
        ...,
        description="When this version was created in DB (ISO 8601)"
    )
    transaction_to: Optional[str] = Field(
        default=None,
        description="When this version was superseded in DB (None = current)"
    )

    # Valid Time (when true in reality)
    valid_from: str = Field(
        ...,
        description="When content became valid in reality (e.g., git commit date)"
    )
    valid_to: Optional[str] = Field(
        default=None,
        description="When content ceased to be valid (None = still valid)"
    )

    # Version Control
    version: int = Field(default=1, ge=1, description="Version number (1, 2, 3...)")
    previous_version_id: Optional[str] = Field(
        default=None,
        description="UUID of previous version (None if first version)"
    )

    # Optimization
    is_current: bool = Field(
        default=True,
        description="True if this is the current version (for fast queries)"
    )

    # GC fields (compatibility)
    stale: bool = Field(default=False, description="Marked for garbage collection")
    last_seen_at: Optional[str] = Field(default=None, description="Last indexing time")


class TemporalQuery(BaseModel):
    """
    Parameters for bi-temporal queries.

    Supports three query modes:
    - current: Latest version (is_current = true)
    - point_in_time: State at specific time
    - history: All versions of an entity

    Example:
        ```python
        # Current state (default)
        query = TemporalQuery()

        # State at specific valid time
        query = TemporalQuery(
            as_of_valid="2025-11-15T00:00:00Z",
            time_type="valid"
        )

        # What we knew at transaction time
        query = TemporalQuery(
            as_of_transaction="2025-11-20T00:00:00Z",
            time_type="transaction"
        )

        # Full history
        query = TemporalQuery(mode="history", limit=50)
        ```
    """

    mode: Literal["current", "point_in_time", "history"] = Field(
        default="current",
        description="Query mode: current, point_in_time, or history"
    )
    time_type: Literal["valid", "transaction", "both"] = Field(
        default="valid",
        description="Which time dimension to query (for point_in_time mode)"
    )
    as_of_valid: Optional[str] = Field(
        default=None,
        description="Point in valid time (ISO 8601)"
    )
    as_of_transaction: Optional[str] = Field(
        default=None,
        description="Point in transaction time (ISO 8601)"
    )
    limit: int = Field(default=50, ge=1, le=1000, description="Max results for history")


@dataclass
class VersionInfo:
    """Information about a specific version in history."""

    version: int
    memory_id: str
    valid_from: str
    valid_to: Optional[str]
    created_at: str
    transaction_to: Optional[str]
    is_current: bool
    change_type: str = "modified"  # created, modified, deleted


@dataclass
class ChangeRecord:
    """Record of a change in the codebase."""

    memory_id: str
    file_path: str
    change_type: str  # created, modified, deleted
    timestamp: str  # When the change occurred
    version: int
    previous_version_id: Optional[str] = None
    qualified_name: Optional[str] = None


@dataclass
class TimelineEntry:
    """Entry in an entity's timeline."""

    version: int
    valid_from: str
    valid_to: Optional[str]
    change_type: str
    summary: str  # Brief description of changes
    memory_id: str
    file_path: Optional[str] = None
