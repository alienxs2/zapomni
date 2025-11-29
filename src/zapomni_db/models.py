"""
Data models for zapomni_db module.

Defines Pydantic models for Memory, Chunk, SearchResult, Workspace, etc.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

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
    """Entity node in knowledge graph."""

    name: str = Field(..., min_length=1)
    type: str = Field(..., min_length=1)
    description: Optional[str] = Field(default="")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    workspace_id: str = Field(default=DEFAULT_WORKSPACE_ID)


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
