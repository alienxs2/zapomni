"""
Unit tests for shared data models (Pydantic schemas).

Tests data model validation, constraints, serialization, and immutability.
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

# Import will fail initially (TDD RED phase)
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
        """Test Chunk with default empty metadata."""
        chunk = Chunk(
            text="Test",
            index=0,
            start_char=0,
            end_char=4,
        )

        assert chunk.metadata == {}

    def test_chunk_immutable(self):
        """Test that Chunk is immutable (frozen)."""
        chunk = Chunk(
            text="Test",
            index=0,
            start_char=0,
            end_char=4,
        )

        with pytest.raises(ValidationError):
            chunk.text = "Modified"

    def test_chunk_required_fields(self):
        """Test that required fields are enforced."""
        with pytest.raises(ValidationError):
            Chunk(text="Test")  # Missing index, start_char, end_char

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

    def test_chunkdata_immutable(self):
        """Test ChunkData immutability."""
        chunk_data = ChunkData(
            text="Test",
            index=0,
            start_char=0,
            end_char=4,
            embedding=[1.0],
        )

        with pytest.raises(ValidationError):
            chunk_data.embedding = [2.0]


class TestMemoryModel:
    """Test Memory data model."""

    def test_memory_valid_creation(self):
        """Test creating valid Memory."""
        chunks = [
            Chunk(text="Chunk 1", index=0, start_char=0, end_char=7),
            Chunk(text="Chunk 2", index=1, start_char=8, end_char=15),
        ]

        memory = Memory(
            id="mem_123",
            text="Chunk 1 Chunk 2",
            chunks=chunks,
            metadata={"source": "test"},
            created_at=datetime.now(),
        )

        assert memory.id == "mem_123"
        assert memory.text == "Chunk 1 Chunk 2"
        assert len(memory.chunks) == 2
        assert memory.metadata == {"source": "test"}
        assert isinstance(memory.created_at, datetime)

    def test_memory_required_fields(self):
        """Test Memory required fields."""
        with pytest.raises(ValidationError):
            Memory(id="mem_123")  # Missing text, chunks, metadata, created_at

    def test_memory_immutable(self):
        """Test Memory immutability."""
        memory = Memory(
            id="mem_123",
            text="Test",
            chunks=[],
            metadata={},
            created_at=datetime.now(),
        )

        with pytest.raises(ValidationError):
            memory.id = "mem_456"


class TestEntityModel:
    """Test Entity data model."""

    def test_entity_valid_creation(self):
        """Test creating valid Entity."""
        entity = Entity(
            id="ent_123",
            text="Python",
            type="TECHNOLOGY",
            confidence=0.95,
            metadata={"source": "spacy"},
        )

        assert entity.id == "ent_123"
        assert entity.text == "Python"
        assert entity.type == "TECHNOLOGY"
        assert entity.confidence == 0.95

    def test_entity_confidence_bounds(self):
        """Test that confidence is between 0 and 1."""
        # Valid confidence
        entity = Entity(
            id="ent_123",
            text="Test",
            type="CONCEPT",
            confidence=0.5,
        )
        assert entity.confidence == 0.5

        # Invalid confidence (too high)
        with pytest.raises(ValidationError):
            Entity(
                id="ent_123",
                text="Test",
                type="CONCEPT",
                confidence=1.5,
            )

        # Invalid confidence (negative)
        with pytest.raises(ValidationError):
            Entity(
                id="ent_123",
                text="Test",
                type="CONCEPT",
                confidence=-0.1,
            )

    def test_entity_default_metadata(self):
        """Test Entity with default metadata."""
        entity = Entity(
            id="ent_123",
            text="Test",
            type="PERSON",
            confidence=0.9,
        )

        assert entity.metadata == {}


class TestRelationshipModel:
    """Test Relationship data model."""

    def test_relationship_valid_creation(self):
        """Test creating valid Relationship."""
        rel = Relationship(
            id="rel_123",
            from_entity="ent_1",
            to_entity="ent_2",
            type="CREATED_BY",
            confidence=0.85,
            metadata={"evidence": "Python was created by Guido"},
        )

        assert rel.id == "rel_123"
        assert rel.from_entity == "ent_1"
        assert rel.to_entity == "ent_2"
        assert rel.type == "CREATED_BY"
        assert rel.confidence == 0.85

    def test_relationship_confidence_bounds(self):
        """Test relationship confidence validation."""
        # Valid
        rel = Relationship(
            id="rel_123",
            from_entity="ent_1",
            to_entity="ent_2",
            type="RELATED_TO",
            confidence=0.5,
        )
        assert rel.confidence == 0.5

        # Invalid (out of bounds)
        with pytest.raises(ValidationError):
            Relationship(
                id="rel_123",
                from_entity="ent_1",
                to_entity="ent_2",
                type="RELATED_TO",
                confidence=2.0,
            )


class TestSearchResultModel:
    """Test SearchResult data model."""

    def test_searchresult_valid_creation(self):
        """Test creating valid SearchResult."""
        result = SearchResult(
            memory_id="mem_123",
            chunk_id="chunk_456",
            text="Relevant text chunk",
            similarity_score=0.92,
            tags=["python", "programming"],
            source="document.txt",
            timestamp=datetime.now(),
            chunk_index=0,
        )

        assert result.memory_id == "mem_123"
        assert result.chunk_id == "chunk_456"
        assert result.text == "Relevant text chunk"
        assert result.similarity_score == 0.92
        assert result.tags == ["python", "programming"]
        assert result.source == "document.txt"
        assert result.chunk_index == 0

    def test_searchresult_similarity_score_bounds(self):
        """Test similarity score is between 0 and 1."""
        # Valid
        result = SearchResult(
            memory_id="mem_123",
            chunk_id="chunk_456",
            text="Test",
            similarity_score=0.5,
            tags=[],
            source="test",
            timestamp=datetime.now(),
            chunk_index=0,
        )
        assert result.similarity_score == 0.5

        # Invalid (too high)
        with pytest.raises(ValidationError):
            SearchResult(
                memory_id="mem_123",
                chunk_id="chunk_456",
                text="Test",
                similarity_score=1.5,
                tags=[],
                source="test",
                timestamp=datetime.now(),
                chunk_index=0,
            )

    def test_searchresult_default_tags(self):
        """Test SearchResult with default empty tags."""
        result = SearchResult(
            memory_id="mem_123",
            chunk_id="chunk_456",
            text="Test",
            similarity_score=0.8,
            source="test",
            timestamp=datetime.now(),
            chunk_index=0,
        )

        assert result.tags == []


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
            id="mem_123",
            text="Test",
            chunks=[
                Chunk(text="Test", index=0, start_char=0, end_char=4),
            ],
            metadata={"source": "test"},
            created_at=datetime.now(),
        )

        # Serialize
        memory_dict = memory.model_dump()

        # Deserialize
        memory_restored = Memory(**memory_dict)

        assert memory_restored.id == memory.id
        assert memory_restored.text == memory.text
        assert len(memory_restored.chunks) == 1
