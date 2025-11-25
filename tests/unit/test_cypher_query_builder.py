"""
Unit tests for CypherQueryBuilder component.

Tests query generation, parameter injection safety, validation, and edge cases.
Focuses on security (no SQL injection), correctness, and parameter handling.
"""

import pytest
import uuid
from datetime import datetime
from typing import Dict, Any, List
from pydantic import ValidationError as PydanticValidationError

from zapomni_db.cypher_query_builder import CypherQueryBuilder
from zapomni_db.models import Memory, Chunk, Entity
from zapomni_db.exceptions import ValidationError


class TestCypherQueryBuilderBasics:
    """Test CypherQueryBuilder instantiation and basic functionality."""

    def test_builder_instantiation(self):
        """Test creating CypherQueryBuilder instance."""
        builder = CypherQueryBuilder()
        assert builder is not None
        assert isinstance(builder, CypherQueryBuilder)

    def test_builder_constants(self):
        """Test that class constants are properly set."""
        assert CypherQueryBuilder.VECTOR_INDEX_NAME == "chunk_embedding_idx"
        assert CypherQueryBuilder.VECTOR_DIMENSION == 768
        assert CypherQueryBuilder.DEFAULT_SIMILARITY_FUNCTION == "cosine"

    def test_builder_is_stateless(self):
        """Test that builder instances are stateless and reusable."""
        builder1 = CypherQueryBuilder()
        builder2 = CypherQueryBuilder()

        # Both should produce identical results
        assert builder1.VECTOR_INDEX_NAME == builder2.VECTOR_INDEX_NAME
        assert builder1.VECTOR_DIMENSION == builder2.VECTOR_DIMENSION


class TestBuildAddMemoryQuery:
    """Test build_add_memory_query method."""

    def test_build_add_memory_query_single_chunk(self):
        """Test add memory query with single chunk."""
        builder = CypherQueryBuilder()
        memory = Memory(
            text="Python is great",
            chunks=[Chunk(text="Python is great", index=0)],
            embeddings=[[0.1] * 768],
            metadata={"source": "user", "tags": ["python"]}
        )

        cypher, params = builder.build_add_memory_query(memory)

        # Verify return types
        assert isinstance(cypher, str)
        assert isinstance(params, dict)

        # Verify cypher string contains expected keywords
        assert "CREATE (m:Memory" in cypher
        assert "UNWIND $chunks" in cypher
        assert "CREATE (c:Chunk" in cypher
        assert "HAS_CHUNK" in cypher

        # Verify parameters
        assert "memory_id" in params
        assert "text" in params
        assert params["text"] == "Python is great"
        assert "source" in params
        assert "tags" in params
        assert "chunks" in params
        assert len(params["chunks"]) == 1

    def test_build_add_memory_query_multiple_chunks(self):
        """Test add memory query with multiple chunks."""
        builder = CypherQueryBuilder()
        memory = Memory(
            text="Python is a programming language",
            chunks=[
                Chunk(text="Python is a", index=0),
                Chunk(text="programming language", index=1),
                Chunk(text="great language", index=2)
            ],
            embeddings=[
                [0.1] * 768,
                [0.2] * 768,
                [0.3] * 768
            ],
            metadata={"source": "docs"}
        )

        cypher, params = builder.build_add_memory_query(memory)

        # Verify chunks data
        assert len(params["chunks"]) == 3
        assert params["chunks"][0]["index"] == 0
        assert params["chunks"][1]["index"] == 1
        assert params["chunks"][2]["index"] == 2

    def test_build_add_memory_query_invalid_embedding_dimension(self):
        """Test wrong embedding dimension raises error."""
        builder = CypherQueryBuilder()
        memory = Memory(
            text="Test",
            chunks=[Chunk(text="test", index=0)],
            embeddings=[[0.1] * 512],  # Wrong dimension (512 instead of 768)
            metadata={}
        )

        with pytest.raises(ValidationError) as exc_info:
            builder.build_add_memory_query(memory)
        assert "Embedding dimension mismatch" in str(exc_info.value)

    def test_build_add_memory_query_generates_unique_ids(self):
        """Test that each query gets unique memory_id and chunk_ids."""
        builder = CypherQueryBuilder()
        memory = Memory(
            text="Test",
            chunks=[Chunk(text="test", index=0)],
            embeddings=[[0.1] * 768],
            metadata={}
        )

        cypher1, params1 = builder.build_add_memory_query(memory)
        cypher2, params2 = builder.build_add_memory_query(memory)

        # IDs should be different
        assert params1["memory_id"] != params2["memory_id"]
        assert params1["chunks"][0]["id"] != params2["chunks"][0]["id"]

    def test_build_add_memory_query_empty_metadata(self):
        """Test handling of empty metadata."""
        builder = CypherQueryBuilder()
        memory = Memory(
            text="Test",
            chunks=[Chunk(text="test", index=0)],
            embeddings=[[0.1] * 768],
            metadata={}
        )

        cypher, params = builder.build_add_memory_query(memory)

        # Should have default source and tags
        assert params["source"] == ""
        assert params["tags"] == []

    def test_build_add_memory_query_preserves_metadata(self):
        """Test that metadata is preserved in parameters."""
        builder = CypherQueryBuilder()
        metadata = {"source": "custom", "tags": ["python", "coding"]}
        memory = Memory(
            text="Test",
            chunks=[Chunk(text="test", index=0)],
            embeddings=[[0.1] * 768],
            metadata=metadata
        )

        cypher, params = builder.build_add_memory_query(memory)

        assert params["source"] == "custom"
        assert params["tags"] == ["python", "coding"]


class TestBuildVectorSearchQuery:
    """Test build_vector_search_query method."""

    def test_build_vector_search_query_basic(self):
        """Test basic vector search query."""
        builder = CypherQueryBuilder()
        embedding = [0.1] * 768

        cypher, params = builder.build_vector_search_query(
            embedding=embedding,
            limit=10,
            min_similarity=0.5
        )

        # Verify cypher structure
        assert "CALL db.idx.vector.queryNodes" in cypher
        assert "'Chunk'" in cypher
        assert "'embedding'" in cypher
        assert "WHERE score >= $min_similarity" in cypher
        assert "ORDER BY score DESC" in cypher

        # Verify parameters
        assert params["query_embedding"] == embedding
        assert params["limit"] == 10
        assert params["min_similarity"] == 0.5

    def test_build_vector_search_query_with_filters(self):
        """Test vector search with metadata filters."""
        builder = CypherQueryBuilder()
        filters = {
            "tags": ["python", "coding"],
            "source": "docs"
        }

        cypher, params = builder.build_vector_search_query(
            embedding=[0.1] * 768,
            filters=filters
        )

        # Verify filter parameters
        assert "tag_0" in params
        assert "tag_1" in params
        assert "source" in params
        assert params["tag_0"] == "python"
        assert params["tag_1"] == "coding"
        assert params["source"] == "docs"

        # Verify filter logic in cypher
        assert "$tag_0 IN m.tags" in cypher
        assert "$tag_1 IN m.tags" in cypher
        assert "m.source = $source" in cypher

    def test_build_vector_search_query_with_date_filters(self):
        """Test vector search with date range filters."""
        builder = CypherQueryBuilder()
        date_from = datetime(2024, 1, 1)
        date_to = datetime(2025, 12, 31)

        filters = {
            "date_from": date_from,
            "date_to": date_to
        }

        cypher, params = builder.build_vector_search_query(
            embedding=[0.1] * 768,
            filters=filters
        )

        # Verify date parameters are converted to ISO strings
        assert "date_from" in params
        assert "date_to" in params
        assert isinstance(params["date_from"], str)
        assert isinstance(params["date_to"], str)

        # Verify date filter logic
        assert "m.created_at >= $date_from" in cypher
        assert "m.created_at <= $date_to" in cypher

    def test_build_vector_search_query_invalid_embedding_dimension(self):
        """Test that wrong embedding dimension raises error."""
        builder = CypherQueryBuilder()

        with pytest.raises(ValidationError) as exc_info:
            builder.build_vector_search_query(
                embedding=[0.1] * 512,  # Wrong dimension
                limit=10
            )
        assert "Embedding dimension mismatch" in str(exc_info.value)

    def test_build_vector_search_query_invalid_limit(self):
        """Test that invalid limit raises error."""
        builder = CypherQueryBuilder()

        # Limit too small
        with pytest.raises(ValidationError) as exc_info:
            builder.build_vector_search_query(
                embedding=[0.1] * 768,
                limit=0
            )
        assert "limit must be int in range [1, 1000]" in str(exc_info.value)

        # Limit too large
        with pytest.raises(ValidationError) as exc_info:
            builder.build_vector_search_query(
                embedding=[0.1] * 768,
                limit=1001
            )
        assert "limit must be int in range [1, 1000]" in str(exc_info.value)

    def test_build_vector_search_query_invalid_min_similarity(self):
        """Test that invalid min_similarity raises error."""
        builder = CypherQueryBuilder()

        # Below range
        with pytest.raises(ValidationError) as exc_info:
            builder.build_vector_search_query(
                embedding=[0.1] * 768,
                min_similarity=-0.1
            )
        assert "min_similarity must be in [0.0, 1.0]" in str(exc_info.value)

        # Above range
        with pytest.raises(ValidationError) as exc_info:
            builder.build_vector_search_query(
                embedding=[0.1] * 768,
                min_similarity=1.1
            )
        assert "min_similarity must be in [0.0, 1.0]" in str(exc_info.value)

    def test_build_vector_search_query_no_filters(self):
        """Test vector search without filters."""
        builder = CypherQueryBuilder()

        cypher, params = builder.build_vector_search_query(
            embedding=[0.1] * 768,
            filters=None
        )

        # Should not have filter parameters
        assert "tag_" not in str(params)


class TestBuildGraphTraversalQuery:
    """Test build_graph_traversal_query method."""

    def test_build_graph_traversal_query_depth_1(self):
        """Test graph traversal with depth 1 (direct neighbors)."""
        builder = CypherQueryBuilder()
        entity_id = str(uuid.uuid4())

        cypher, params = builder.build_graph_traversal_query(
            entity_id=entity_id,
            depth=1,
            limit=20
        )

        # Verify cypher structure
        assert "MATCH (start:Entity {id: $entity_id})" in cypher
        assert "[rels*1..1]" in cypher
        assert "WHERE related.id <> $entity_id" in cypher
        assert "ORDER BY path_strength DESC" in cypher

        # Verify parameters
        assert params["entity_id"] == entity_id
        assert params["limit"] == 20

    def test_build_graph_traversal_query_depth_3(self):
        """Test graph traversal with depth 3."""
        builder = CypherQueryBuilder()
        entity_id = str(uuid.uuid4())

        cypher, params = builder.build_graph_traversal_query(
            entity_id=entity_id,
            depth=3,
            limit=50
        )

        # Verify correct depth pattern
        assert "[rels*1..3]" in cypher

    def test_build_graph_traversal_query_invalid_entity_id(self):
        """Test that invalid UUID raises error."""
        builder = CypherQueryBuilder()

        with pytest.raises(ValidationError) as exc_info:
            builder.build_graph_traversal_query(
                entity_id="not-a-uuid",
                depth=1
            )
        assert "Invalid UUID format" in str(exc_info.value)

    def test_build_graph_traversal_query_invalid_depth(self):
        """Test that invalid depth raises error."""
        builder = CypherQueryBuilder()
        entity_id = str(uuid.uuid4())

        # Depth too small
        with pytest.raises(ValidationError) as exc_info:
            builder.build_graph_traversal_query(
                entity_id=entity_id,
                depth=0
            )
        assert "depth must be int in range [1, 5]" in str(exc_info.value)

        # Depth too large
        with pytest.raises(ValidationError) as exc_info:
            builder.build_graph_traversal_query(
                entity_id=entity_id,
                depth=6
            )
        assert "depth must be int in range [1, 5]" in str(exc_info.value)

    def test_build_graph_traversal_query_invalid_limit(self):
        """Test that invalid limit raises error."""
        builder = CypherQueryBuilder()
        entity_id = str(uuid.uuid4())

        # Limit too small
        with pytest.raises(ValidationError) as exc_info:
            builder.build_graph_traversal_query(
                entity_id=entity_id,
                depth=1,
                limit=0
            )
        assert "limit must be int in range [1, 100]" in str(exc_info.value)

        # Limit too large
        with pytest.raises(ValidationError) as exc_info:
            builder.build_graph_traversal_query(
                entity_id=entity_id,
                depth=1,
                limit=101
            )
        assert "limit must be int in range [1, 100]" in str(exc_info.value)


class TestBuildStatsQuery:
    """Test build_stats_query method."""

    def test_build_stats_query(self):
        """Test stats query generation."""
        builder = CypherQueryBuilder()

        cypher, params = builder.build_stats_query()

        # Verify cypher structure
        assert "MATCH (m:Memory)" in cypher
        assert "MATCH (c:Chunk)" in cypher
        assert "MATCH (e:Entity)" in cypher
        assert "MATCH ()-[r]->()" in cypher
        assert "total_memories" in cypher
        assert "total_chunks" in cypher
        assert "total_entities" in cypher
        assert "total_relationships" in cypher

        # Verify no parameters needed
        assert params == {}


class TestBuildDeleteMemoryQuery:
    """Test build_delete_memory_query method."""

    def test_build_delete_memory_query(self):
        """Test delete memory query generation."""
        builder = CypherQueryBuilder()
        memory_id = str(uuid.uuid4())

        cypher, params = builder.build_delete_memory_query(memory_id)

        # Verify cypher structure
        assert "MATCH (m:Memory {id: $memory_id})" in cypher
        assert "OPTIONAL MATCH (m)-[:HAS_CHUNK]->(c:Chunk)" in cypher
        assert "DETACH DELETE m, c" in cypher

        # Verify parameters
        assert params["memory_id"] == memory_id

    def test_build_delete_memory_query_invalid_uuid(self):
        """Test that invalid UUID raises error."""
        builder = CypherQueryBuilder()

        with pytest.raises(ValidationError) as exc_info:
            builder.build_delete_memory_query("invalid-uuid")
        assert "Invalid UUID format" in str(exc_info.value)


class TestBuildAddEntityQuery:
    """Test build_add_entity_query method."""

    def test_build_add_entity_query(self):
        """Test add entity query generation."""
        builder = CypherQueryBuilder()
        entity = Entity(
            name="Python",
            type="TECHNOLOGY",
            description="Programming language",
            confidence=0.95
        )

        cypher, params = builder.build_add_entity_query(entity)

        # Verify cypher structure
        assert "CREATE (e:Entity" in cypher
        assert "id: $entity_id" in cypher
        assert "name: $name" in cypher
        assert "type: $type" in cypher
        assert "confidence: $confidence" in cypher

        # Verify parameters
        assert "entity_id" in params
        assert params["name"] == "Python"
        assert params["type"] == "TECHNOLOGY"
        assert params["description"] == "Programming language"
        assert params["confidence"] == 0.95

    def test_build_add_entity_query_validation_happens_at_pydantic(self):
        """Test that validation happens at Pydantic level for models."""
        builder = CypherQueryBuilder()

        # Pydantic validates before we even get to the builder
        with pytest.raises(PydanticValidationError):
            Entity(name="", type="TECHNOLOGY")

        with pytest.raises(PydanticValidationError):
            Entity(name="Python", type="", confidence=0.95)

        with pytest.raises(PydanticValidationError):
            Entity(name="Python", type="TECHNOLOGY", confidence=1.5)


class TestBuildAddRelationshipQuery:
    """Test build_add_relationship_query method."""

    def test_build_add_relationship_query(self):
        """Test add relationship query generation."""
        builder = CypherQueryBuilder()
        from_id = str(uuid.uuid4())
        to_id = str(uuid.uuid4())

        cypher, params = builder.build_add_relationship_query(
            from_entity_id=from_id,
            to_entity_id=to_id,
            relationship_type="MENTIONS",
            properties={"strength": 0.8, "confidence": 0.9}
        )

        # Verify cypher structure
        assert "MATCH (from:Entity {id: $from_id})" in cypher
        assert "MATCH (to:Entity {id: $to_id})" in cypher
        assert "CREATE (from)-[r:MENTIONS" in cypher

        # Verify parameters
        assert params["from_id"] == from_id
        assert params["to_id"] == to_id
        assert params["strength"] == 0.8
        assert params["confidence"] == 0.9

    def test_build_add_relationship_query_default_properties(self):
        """Test relationship with default properties."""
        builder = CypherQueryBuilder()
        from_id = str(uuid.uuid4())
        to_id = str(uuid.uuid4())

        cypher, params = builder.build_add_relationship_query(
            from_entity_id=from_id,
            to_entity_id=to_id,
            relationship_type="RELATED_TO",
            properties=None
        )

        # Verify defaults
        assert params["strength"] == 1.0
        assert params["confidence"] == 1.0
        assert params["context"] == ""

    def test_build_add_relationship_query_invalid_entity_id(self):
        """Test that invalid entity IDs raise error."""
        builder = CypherQueryBuilder()

        with pytest.raises(ValidationError) as exc_info:
            builder.build_add_relationship_query(
                from_entity_id="invalid",
                to_entity_id=str(uuid.uuid4()),
                relationship_type="MENTIONS"
            )
        assert "Invalid UUID format" in str(exc_info.value)

    def test_build_add_relationship_query_invalid_type_format(self):
        """Test that invalid relationship type format raises error."""
        builder = CypherQueryBuilder()
        from_id = str(uuid.uuid4())
        to_id = str(uuid.uuid4())

        # Lowercase letters not allowed
        with pytest.raises(ValidationError) as exc_info:
            builder.build_add_relationship_query(
                from_entity_id=from_id,
                to_entity_id=to_id,
                relationship_type="mentions"  # lowercase
            )
        assert "relationship_type must match pattern" in str(exc_info.value)

        # Special characters not allowed
        with pytest.raises(ValidationError) as exc_info:
            builder.build_add_relationship_query(
                from_entity_id=from_id,
                to_entity_id=to_id,
                relationship_type="MENTIONS-TYPE"  # hyphen not allowed
            )
        assert "relationship_type must match pattern" in str(exc_info.value)

    def test_build_add_relationship_query_invalid_strength(self):
        """Test that invalid strength raises error."""
        builder = CypherQueryBuilder()
        from_id = str(uuid.uuid4())
        to_id = str(uuid.uuid4())

        with pytest.raises(ValidationError) as exc_info:
            builder.build_add_relationship_query(
                from_entity_id=from_id,
                to_entity_id=to_id,
                relationship_type="MENTIONS",
                properties={"strength": 1.5}
            )
        assert "[0.0, 1.0]" in str(exc_info.value)


class TestParameterInjectionSafety:
    """Test that builder prevents Cypher injection attacks."""

    def test_malicious_filter_injection(self):
        """Test that filter values cannot inject code."""
        builder = CypherQueryBuilder()

        # Try to inject malicious filter
        filters = {
            "tags": ["python'; DETACH DELETE *; //"]
        }

        cypher, params = builder.build_vector_search_query(
            embedding=[0.1] * 768,
            filters=filters
        )

        # Verify the malicious string is in params, not in cypher
        assert params["tag_0"] == "python'; DETACH DELETE *; //"
        # The malicious code should NOT be in the cypher string itself
        assert "DETACH DELETE" not in cypher or cypher.count("DETACH DELETE") == 0

    def test_entity_id_injection_attempt(self):
        """Test that entity IDs are validated and parameterized."""
        builder = CypherQueryBuilder()

        # Try to inject via entity_id
        with pytest.raises(ValidationError):
            builder.build_graph_traversal_query(
                entity_id="550e8400-e29b-41d4-a716-446655440000; MATCH (n) DETACH DELETE n",
                depth=1
            )

    def test_relationship_type_validation(self):
        """Test that relationship types are validated against injection."""
        builder = CypherQueryBuilder()
        from_id = str(uuid.uuid4())
        to_id = str(uuid.uuid4())

        # Try to inject via relationship_type
        with pytest.raises(ValidationError):
            builder.build_add_relationship_query(
                from_entity_id=from_id,
                to_entity_id=to_id,
                relationship_type="MENTIONS}; DETACH DELETE *; {"  # Invalid format
            )


class TestValidationHelpers:
    """Test private validation helper methods."""

    def test_validate_uuid_valid(self):
        """Test UUID validation with valid UUID."""
        builder = CypherQueryBuilder()
        valid_uuid = str(uuid.uuid4())

        # Should not raise
        builder._validate_uuid(valid_uuid)

    def test_validate_uuid_invalid(self):
        """Test UUID validation with invalid strings."""
        builder = CypherQueryBuilder()

        invalid_uuids = [
            "not-a-uuid",
            "12345",
            "",
            "550e8400-e29b-41d4-a716-44665544000",  # Missing one char
            "550e8400-e29b-41d4-a716-4466554400000"  # Extra char
        ]

        for invalid_uuid in invalid_uuids:
            with pytest.raises(ValidationError):
                builder._validate_uuid(invalid_uuid)

    def test_validate_embedding_valid(self):
        """Test embedding validation with valid embedding."""
        builder = CypherQueryBuilder()
        valid_embedding = [0.1] * 768

        # Should not raise
        builder._validate_embedding(valid_embedding)

    def test_validate_embedding_wrong_dimension(self):
        """Test embedding validation with wrong dimension."""
        builder = CypherQueryBuilder()

        with pytest.raises(ValidationError) as exc_info:
            builder._validate_embedding([0.1] * 512)
        assert "Embedding dimension mismatch" in str(exc_info.value)

    def test_validate_embedding_empty(self):
        """Test embedding validation with empty embedding."""
        builder = CypherQueryBuilder()

        with pytest.raises(ValidationError) as exc_info:
            builder._validate_embedding([])
        assert "Embedding cannot be empty" in str(exc_info.value)

    def test_validate_embedding_non_numeric(self):
        """Test embedding validation with non-numeric values."""
        builder = CypherQueryBuilder()

        embedding = [0.1] * 767 + ["not-a-number"]

        with pytest.raises(ValidationError) as exc_info:
            builder._validate_embedding(embedding)
        assert "numeric values" in str(exc_info.value)


class TestBuildFilterClause:
    """Test _build_filter_clause private method."""

    def test_build_filter_clause_empty(self):
        """Test filter clause with no filters."""
        builder = CypherQueryBuilder()

        clause, params = builder._build_filter_clause(None)

        assert clause == ""
        assert params == {}

    def test_build_filter_clause_tags_only(self):
        """Test filter clause with tags only."""
        builder = CypherQueryBuilder()

        clause, params = builder._build_filter_clause({"tags": ["python", "coding"]})

        # Should use OR logic for tags
        assert "$tag_0 IN m.tags" in clause
        assert "$tag_1 IN m.tags" in clause
        assert " OR " in clause

        assert params["tag_0"] == "python"
        assert params["tag_1"] == "coding"

    def test_build_filter_clause_source_only(self):
        """Test filter clause with source only."""
        builder = CypherQueryBuilder()

        clause, params = builder._build_filter_clause({"source": "user_input"})

        assert "m.source = $source" in clause
        assert params["source"] == "user_input"

    def test_build_filter_clause_multiple_filters(self):
        """Test filter clause with multiple filter types."""
        builder = CypherQueryBuilder()

        filters = {
            "tags": ["python"],
            "source": "docs",
            "date_from": datetime(2024, 1, 1),
            "date_to": datetime(2025, 12, 31)
        }

        clause, params = builder._build_filter_clause(filters)

        # All filters should be present
        assert "$tag_0 IN m.tags" in clause
        assert "m.source = $source" in clause
        assert "m.created_at >= $date_from" in clause
        assert "m.created_at <= $date_to" in clause

        # All parameters should be present
        assert params["tag_0"] == "python"
        assert params["source"] == "docs"
        assert "date_from" in params
        assert "date_to" in params

    def test_build_filter_clause_datetime_conversion(self):
        """Test that datetime objects are converted to ISO strings."""
        builder = CypherQueryBuilder()

        dt = datetime(2024, 1, 15, 12, 30, 45)
        filters = {"date_from": dt}

        clause, params = builder._build_filter_clause(filters)

        # Should be converted to ISO string
        assert isinstance(params["date_from"], str)
        assert "2024-01-15" in params["date_from"]


class TestReturnTypes:
    """Test that all methods return correct types."""

    def test_all_methods_return_tuple(self):
        """Test that all build_* methods return (str, dict) tuple."""
        builder = CypherQueryBuilder()

        # Test build_add_memory_query
        memory = Memory(
            text="Test",
            chunks=[Chunk(text="test", index=0)],
            embeddings=[[0.1] * 768],
            metadata={}
        )
        result = builder.build_add_memory_query(memory)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], dict)

        # Test build_vector_search_query
        result = builder.build_vector_search_query([0.1] * 768)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], dict)

        # Test build_graph_traversal_query
        result = builder.build_graph_traversal_query(str(uuid.uuid4()))
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], dict)

        # Test build_stats_query
        result = builder.build_stats_query()
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], dict)

        # Test build_delete_memory_query
        result = builder.build_delete_memory_query(str(uuid.uuid4()))
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], dict)

        # Test build_add_entity_query
        entity = Entity(name="Test", type="CONCEPT", confidence=0.9)
        result = builder.build_add_entity_query(entity)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], dict)

        # Test build_add_relationship_query
        result = builder.build_add_relationship_query(
            str(uuid.uuid4()),
            str(uuid.uuid4()),
            "MENTIONS"
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], dict)


class TestParameterization:
    """Test that all parameters are properly parameterized."""

    def test_memory_text_is_parameterized(self):
        """Test that memory text uses parameter, not concatenation."""
        builder = CypherQueryBuilder()
        memory = Memory(
            text="Injected code: MATCH (n) DETACH DELETE n",
            chunks=[Chunk(text="test", index=0)],
            embeddings=[[0.1] * 768],
            metadata={}
        )

        cypher, params = builder.build_add_memory_query(memory)

        # The malicious text should be in parameters, not in cypher string
        assert "DETACH DELETE" in params["text"]
        # But NOT in the cypher query itself (except maybe in comments)
        assert "DETACH DELETE" not in cypher.split("--")[0]  # Ignore comment sections

    def test_filter_values_parameterized(self):
        """Test that filter values use parameters."""
        builder = CypherQueryBuilder()

        filters = {
            "tags": ["python'; DELETE *; --"],
            "source": "docs' OR '1'='1"
        }

        cypher, params = builder.build_vector_search_query(
            [0.1] * 768,
            filters=filters
        )

        # Values should be in params
        assert params["tag_0"] == "python'; DELETE *; --"
        assert params["source"] == "docs' OR '1'='1"

        # But not directly in cypher string
        assert "DELETE *" not in cypher.split("--")[0]

