"""
Data models for zapomni_db module.

Defines Pydantic models for Memory, Chunk, SearchResult, etc.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator


class Chunk(BaseModel):
    """Text chunk with positional metadata."""
    text: str = Field(..., min_length=1)
    index: int = Field(..., ge=0)
    start_char: Optional[int] = Field(default=None, ge=0)
    end_char: Optional[int] = Field(default=None, ge=0)
    metadata: Optional[Dict[str, Any]] = None


class Memory(BaseModel):
    """Complete memory with chunks, embeddings, and metadata."""
    text: str = Field(..., min_length=1, max_length=1_000_000)
    chunks: List[Chunk] = Field(..., min_length=1, max_length=100)
    embeddings: List[List[float]] = Field(..., min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('chunks')
    @classmethod
    def validate_chunks(cls, v):
        """Ensure chunks are not empty."""
        if not v:
            raise ValueError("chunks list cannot be empty")
        return v

    @field_validator('embeddings')
    @classmethod
    def validate_embeddings(cls, v):
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


class Relationship(BaseModel):
    """Relationship edge in knowledge graph."""
    from_entity_id: str = Field(..., min_length=1)
    to_entity_id: str = Field(..., min_length=1)
    relationship_type: str = Field(..., min_length=1)
    strength: float = Field(default=1.0, ge=0.0, le=1.0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    context: Optional[str] = None
