"""
Unit tests for shared data models (Pydantic schemas).

Tests data model validation, constraints, serialization.
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

# Import models from zapomni_db
from zapomni_db.models import (
    Chunk,
    ChunkData,
    Entity,
    Memory,
    MemoryResult,
    Relationship,
    SearchResult,
)


class TestChunkModel:
    """Test Chunk data model."""

    def test_chunk_valid_creation(self):
        """Test creating valid Chunk."""
        chunk = Chunk(
            text="This is a test chunk.",
            index=0,
            start_char=0,
            end_char=21,
            metadata={"source": "test"},
        )

        assert chunk.text == "This is a test chunk."
        assert chunk.index == 0
        assert chunk.start_char == 0
        assert chunk.end_char == 21
        assert chunk.metadata == {"source": "test"}

    def test_chunk_default_metadata(self):
        """Test Chunk with default None metadata."""
        chunk = Chunk(
            text="Test",
            index=0,
            start_char=0,
            end_char=4,
        )

        # Current implementation uses None as default
        assert chunk.metadata is None

    def test_chunk_mutable(self):
        """Test that Chunk is mutable (not frozen)."""
        chunk = Chunk(
            text="Test",
            index=0,
            start_char=0,
            end_char=4,
        )

        # Current models are NOT frozen, so this should work
        chunk.text = "Modified"
        assert chunk.text == "Modified"

    def test_chunk_required_fields(self):
        """Test that required fields are enforced."""
        with pytest.raises(ValidationError):
            Chunk(text="Test")  # Missing index

    def test_chunk_serialization(self):
        """Test Chunk JSON serialization."""
        chunk = Chunk(
            text="Test",
            index=1,
            start_char=10,
            end_char=14,
            metadata={"key": "value"},
        )

        chunk_dict = chunk.model_dump()

        assert chunk_dict["text"] == "Test"
        assert chunk_dict["index"] == 1
        assert chunk_dict["start_char"] == 10
        assert chunk_dict["end_char"] == 14
        assert chunk_dict["metadata"] == {"key": "value"}


class TestChunkDataModel:
    """Test ChunkData model (Chunk with embedding)."""

    def test_chunkdata_valid_creation(self):
        """Test creating valid ChunkData with embedding."""
        chunk_data = ChunkData(
            text="Test chunk",
            index=0,
            start_char=0,
            end_char=10,
            embedding=[0.1, 0.2, 0.3],
            metadata={"source": "test"},
        )

        assert chunk_data.text == "Test chunk"
        assert chunk_data.embedding == [0.1, 0.2, 0.3]

    def test_chunkdata_embedding_validation(self):
        """Test that embedding must be list of floats."""
        chunk_data = ChunkData(
            text="Test",
            index=0,
            start_char=0,
            end_char=4,
            embedding=[1.0, 2.0, 3.0],
        )

        assert len(chunk_data.embedding) == 3
        assert all(isinstance(x, float) for x in chunk_data.embedding)

    def test_chunkdata_mutable(self):
        """Test ChunkData is mutable (not frozen)."""
        chunk_data = ChunkData(
            text="Test",
            index=0,
            start_char=0,
            end_char=4,
            embedding=[1.0],
        )

        # Current models are NOT frozen
        chunk_data.embedding = [2.0]
        assert chunk_data.embedding == [2.0]


class TestMemoryModel:
    """Test Memory data model."""

    def test_memory_valid_creation(self):
        """Test creating valid Memory."""
        chunks = [
            Chunk(text="Chunk 1", index=0, start_char=0, end_char=7),
            Chunk(text="Chunk 2", index=1, start_char=8, end_char=15),
        ]

        memory = Memory(
            text="Chunk 1 Chunk 2",
            chunks=chunks,
            embeddings=[[0.1, 0.2], [0.3, 0.4]],  # Required field
            metadata={"source": "test"},
        )

        assert memory.text == "Chunk 1 Chunk 2"
        assert len(memory.chunks) == 2
        assert memory.metadata == {"source": "test"}
        assert len(memory.embeddings) == 2

    def test_memory_required_fields(self):
        """Test Memory required fields."""
        with pytest.raises(ValidationError):
            Memory(text="Test")  # Missing chunks, embeddings

    def test_memory_mutable(self):
        """Test Memory is mutable (not frozen)."""
        memory = Memory(
            text="Test",
            chunks=[Chunk(text="Test", index=0, start_char=0, end_char=4)],
            embeddings=[[0.1, 0.2]],
            metadata={},
        )

        # Current models are NOT frozen
        memory.text = "Modified"
        assert memory.text == "Modified"


class TestEntityModel:
    """Test Entity data model."""

    def test_entity_valid_creation(self):
        """Test creating valid Entity."""
        entity = Entity(
            name="Python",  # Uses 'name' not 'text'
            type="TECHNOLOGY",
            confidence=0.95,
            description="Programming language",
        )

        assert entity.name == "Python"
        assert entity.type == "TECHNOLOGY"
        assert entity.confidence == 0.95
        assert entity.description == "Programming language"

    def test_entity_confidence_bounds(self):
        """Test that confidence is between 0 and 1."""
        # Valid confidence
        entity = Entity(
            name="Test",  # Uses 'name' not 'text'
            type="CONCEPT",
            confidence=0.5,
        )
        assert entity.confidence == 0.5

        # Invalid confidence (too high)
        with pytest.raises(ValidationError):
            Entity(
                name="Test",
                type="CONCEPT",
                confidence=1.5,
            )

        # Invalid confidence (negative)
        with pytest.raises(ValidationError):
            Entity(
                name="Test",
                type="CONCEPT",
                confidence=-0.1,
            )

    def test_entity_default_description(self):
        """Test Entity with default description."""
        entity = Entity(
            name="Test",
            type="PERSON",
            confidence=0.9,
        )

        # Default description is empty string
        assert entity.description == ""


class TestRelationshipModel:
    """Test Relationship data model."""

    def test_relationship_valid_creation(self):
        """Test creating valid Relationship."""
        rel = Relationship(
            from_entity_id="ent_1",  # New field names
            to_entity_id="ent_2",
            relationship_type="CREATED_BY",
            confidence=0.85,
            context="Python was created by Guido",
        )

        assert rel.from_entity_id == "ent_1"
        assert rel.to_entity_id == "ent_2"
        assert rel.relationship_type == "CREATED_BY"
        assert rel.confidence == 0.85

    def test_relationship_confidence_bounds(self):
        """Test relationship confidence validation."""
        # Valid
        rel = Relationship(
            from_entity_id="ent_1",
            to_entity_id="ent_2",
            relationship_type="RELATED_TO",
            confidence=0.5,
        )
        assert rel.confidence == 0.5

        # Invalid (out of bounds)
        with pytest.raises(ValidationError):
            Relationship(
                from_entity_id="ent_1",
                to_entity_id="ent_2",
                relationship_type="RELATED_TO",
                confidence=2.0,
            )


class TestSearchResultModel:
    """Test SearchResult data model (dataclass)."""

    def test_searchresult_valid_creation(self):
        """Test creating valid SearchResult."""
        result = SearchResult(
            memory_id="mem_123",
            content="Relevant text",  # Required field
            relevance_score=0.92,  # Required field
            metadata={"key": "value"},
            chunk_id="chunk_456",
            text="Relevant text chunk",
            similarity_score=0.92,
            tags=["python", "programming"],
            source="document.txt",
            timestamp=datetime.now(),
            chunk_index=0,
        )

        assert result.memory_id == "mem_123"
        assert result.content == "Relevant text"
        assert result.relevance_score == 0.92
        assert result.chunk_id == "chunk_456"
        assert result.text == "Relevant text chunk"
        assert result.similarity_score == 0.92
        assert result.tags == ["python", "programming"]
        assert result.source == "document.txt"
        assert result.chunk_index == 0

    def test_searchresult_required_fields(self):
        """Test SearchResult required fields."""
        # SearchResult is a dataclass with required fields
        result = SearchResult(
            memory_id="mem_123",
            content="Test content",
            relevance_score=0.5,
        )
        assert result.memory_id == "mem_123"
        assert result.content == "Test content"
        assert result.relevance_score == 0.5

    def test_searchresult_default_values(self):
        """Test SearchResult with default values."""
        result = SearchResult(
            memory_id="mem_123",
            content="Test",
            relevance_score=0.8,
        )

        # Optional fields should have None as default
        assert result.tags is None
        assert result.source is None
        assert result.timestamp is None


class TestMemoryResultModel:
    """Test MemoryResult data model."""

    def test_memoryresult_valid_creation(self):
        """Test creating valid MemoryResult."""
        result = MemoryResult(
            id="mem_123",
            chunks_created=5,
            processing_time_ms=1234.5,
        )

        assert result.id == "mem_123"
        assert result.chunks_created == 5
        assert result.processing_time_ms == 1234.5

    def test_memoryresult_required_fields(self):
        """Test MemoryResult required fields."""
        with pytest.raises(ValidationError):
            MemoryResult(id="mem_123")  # Missing chunks_created, processing_time_ms

    def test_memoryresult_positive_values(self):
        """Test that chunks and time are positive."""
        # Valid
        result = MemoryResult(
            id="mem_123",
            chunks_created=1,
            processing_time_ms=1.0,
        )
        assert result.chunks_created >= 0
        assert result.processing_time_ms >= 0

        # chunks_created should be non-negative integer
        with pytest.raises(ValidationError):
            MemoryResult(
                id="mem_123",
                chunks_created=-1,
                processing_time_ms=100.0,
            )


class TestModelSerialization:
    """Test model serialization and deserialization."""

    def test_chunk_json_roundtrip(self):
        """Test Chunk JSON serialization roundtrip."""
        chunk = Chunk(
            text="Test",
            index=0,
            start_char=0,
            end_char=4,
            metadata={"key": "value"},
        )

        # Serialize to dict
        chunk_dict = chunk.model_dump()

        # Deserialize back
        chunk_restored = Chunk(**chunk_dict)

        assert chunk_restored.text == chunk.text
        assert chunk_restored.index == chunk.index
        assert chunk_restored.metadata == chunk.metadata

    def test_memory_json_roundtrip(self):
        """Test Memory JSON serialization roundtrip."""
        memory = Memory(
            text="Test",
            chunks=[
                Chunk(text="Test", index=0, start_char=0, end_char=4),
            ],
            embeddings=[[0.1, 0.2, 0.3]],  # Required field
            metadata={"source": "test"},
        )

        # Serialize
        memory_dict = memory.model_dump()

        # Deserialize
        memory_restored = Memory(**memory_dict)

        assert memory_restored.text == memory.text
        assert len(memory_restored.chunks) == 1
        assert len(memory_restored.embeddings) == 1
