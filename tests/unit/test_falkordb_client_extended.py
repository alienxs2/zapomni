"""
Extended unit tests for FalkorDBClient - new methods.

Tests for:
- add_entity
- add_relationship
- get_related_entities
- delete_memory
- clear_all
- graph_query
"""

import asyncio
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from zapomni_db.exceptions import ConnectionError, DatabaseError, ValidationError
from zapomni_db.falkordb_client import FalkorDBClient
from zapomni_db.models import Entity, QueryResult, Relationship

# ============================================================================
# add_entity TESTS
# ============================================================================


class TestFalkorDBClientAddEntity:
    """Test FalkorDBClient.add_entity()."""

    @pytest.fixture
    def mock_client(self, mocker):
        """Create mocked client."""
        client = FalkorDBClient()

        # Mock successful entity creation
        mock_result = MagicMock()
        mock_result.rows = [{"entity_id": str(uuid.uuid4())}]
        mock_result.row_count = 1
        mock_result.execution_time_ms = 10

        async def mock_execute(query, params):
            return mock_result

        mocker.patch.object(client, "_execute_cypher", side_effect=mock_execute)
        mocker.patch.object(client, "_initialized", True)
        mocker.patch.object(client, "_schema_ready", True)

        return client

    @pytest.mark.asyncio
    async def test_add_entity_success(self, mock_client):
        """Test successful entity addition."""
        entity = Entity(
            name="Python", type="TECHNOLOGY", description="Programming language", confidence=0.95
        )

        entity_id = await mock_client.add_entity(entity)

        assert isinstance(entity_id, str)
        assert len(entity_id) == 36  # UUID format
        uuid.UUID(entity_id)  # Validate UUID

    @pytest.mark.asyncio
    async def test_add_entity_minimal(self, mock_client):
        """Test entity with minimal required fields."""
        entity = Entity(name="AI", type="CONCEPT")

        entity_id = await mock_client.add_entity(entity)
        assert entity_id is not None

    def test_add_entity_empty_name_raises(self):
        """Test ValidationError on empty name."""
        with pytest.raises(Exception, match="at least 1 character"):
            entity = Entity(name="", type="CONCEPT")

    def test_add_entity_empty_type_raises(self):
        """Test ValidationError on empty type."""
        with pytest.raises(Exception, match="at least 1 character"):
            entity = Entity(name="Test", type="")

    def test_add_entity_invalid_confidence_raises(self):
        """Test ValidationError on confidence out of range."""
        with pytest.raises(Exception, match="greater than or equal to 0"):
            entity = Entity(name="Test", type="CONCEPT", confidence=-0.1)

        with pytest.raises(Exception, match="less than or equal to 1"):
            entity = Entity(name="Test", type="CONCEPT", confidence=1.5)


# ============================================================================
# add_relationship TESTS
# ============================================================================


class TestFalkorDBClientAddRelationship:
    """Test FalkorDBClient.add_relationship()."""

    @pytest.fixture
    def mock_client(self, mocker):
        """Create mocked client."""
        client = FalkorDBClient()

        # Mock successful relationship creation
        mock_result = MagicMock()
        mock_result.rows = [{"relationship_id": str(uuid.uuid4())}]
        mock_result.row_count = 1
        mock_result.execution_time_ms = 10

        async def mock_execute(query, params):
            return mock_result

        mocker.patch.object(client, "_execute_cypher", side_effect=mock_execute)
        mocker.patch.object(client, "_initialized", True)
        mocker.patch.object(client, "_schema_ready", True)

        return client

    @pytest.mark.asyncio
    async def test_add_relationship_success(self, mock_client):
        """Test successful relationship addition."""
        from_id = str(uuid.uuid4())
        to_id = str(uuid.uuid4())

        rel_id = await mock_client.add_relationship(
            from_entity_id=from_id, to_entity_id=to_id, relationship_type="USES"
        )

        assert isinstance(rel_id, str)
        assert len(rel_id) == 36  # UUID format

    @pytest.mark.asyncio
    async def test_add_relationship_with_properties(self, mock_client):
        """Test relationship with properties."""
        from_id = str(uuid.uuid4())
        to_id = str(uuid.uuid4())

        rel_id = await mock_client.add_relationship(
            from_entity_id=from_id,
            to_entity_id=to_id,
            relationship_type="RELATED_TO",
            properties={"strength": 0.8, "confidence": 0.9, "context": "Test context"},
        )

        assert rel_id is not None

    def test_add_relationship_invalid_from_uuid(self):
        """Test ValidationError on invalid from_entity_id."""
        client = FalkorDBClient()

        with pytest.raises(ValidationError, match="Invalid entity UUID"):
            asyncio.run(
                client.add_relationship(
                    from_entity_id="invalid-uuid",
                    to_entity_id=str(uuid.uuid4()),
                    relationship_type="USES",
                )
            )

    def test_add_relationship_invalid_to_uuid(self):
        """Test ValidationError on invalid to_entity_id."""
        client = FalkorDBClient()

        with pytest.raises(ValidationError, match="Invalid entity UUID"):
            asyncio.run(
                client.add_relationship(
                    from_entity_id=str(uuid.uuid4()),
                    to_entity_id="invalid-uuid",
                    relationship_type="USES",
                )
            )

    def test_add_relationship_invalid_strength(self):
        """Test ValidationError on invalid strength."""
        client = FalkorDBClient()

        with pytest.raises(ValidationError, match="strength must be in"):
            asyncio.run(
                client.add_relationship(
                    from_entity_id=str(uuid.uuid4()),
                    to_entity_id=str(uuid.uuid4()),
                    relationship_type="USES",
                    properties={"strength": 1.5},
                )
            )

    def test_add_relationship_invalid_confidence(self):
        """Test ValidationError on invalid confidence."""
        client = FalkorDBClient()

        with pytest.raises(ValidationError, match="confidence must be in"):
            asyncio.run(
                client.add_relationship(
                    from_entity_id=str(uuid.uuid4()),
                    to_entity_id=str(uuid.uuid4()),
                    relationship_type="USES",
                    properties={"confidence": -0.1},
                )
            )


# ============================================================================
# get_related_entities TESTS
# ============================================================================


class TestFalkorDBClientGetRelatedEntities:
    """Test FalkorDBClient.get_related_entities()."""

    @pytest.fixture
    def mock_client(self, mocker):
        """Create mocked client."""
        client = FalkorDBClient()

        # Mock successful query
        mock_result = MagicMock()
        mock_result.rows = [
            {
                "id": str(uuid.uuid4()),
                "name": "Python",
                "type": "TECHNOLOGY",
                "description": "Programming language",
                "confidence": 0.95,
                "avg_strength": 0.8,
            },
            {
                "id": str(uuid.uuid4()),
                "name": "AI",
                "type": "CONCEPT",
                "description": "Artificial Intelligence",
                "confidence": 0.90,
                "avg_strength": 0.7,
            },
        ]
        mock_result.row_count = 2
        mock_result.execution_time_ms = 15

        async def mock_execute(query, params):
            return mock_result

        mocker.patch.object(client, "_execute_cypher", side_effect=mock_execute)
        mocker.patch.object(client, "_initialized", True)
        mocker.patch.object(client, "_schema_ready", True)

        return client

    @pytest.mark.asyncio
    async def test_get_related_entities_success(self, mock_client):
        """Test successful related entities retrieval."""
        entity_id = str(uuid.uuid4())

        entities = await mock_client.get_related_entities(entity_id=entity_id, depth=1, limit=10)

        assert isinstance(entities, list)
        assert len(entities) == 2
        assert all(isinstance(e, Entity) for e in entities)

    @pytest.mark.asyncio
    async def test_get_related_entities_depth_2(self, mock_client):
        """Test with depth=2."""
        entity_id = str(uuid.uuid4())

        entities = await mock_client.get_related_entities(entity_id=entity_id, depth=2, limit=20)

        assert isinstance(entities, list)

    def test_get_related_entities_invalid_uuid(self):
        """Test ValidationError on invalid UUID."""
        client = FalkorDBClient()

        with pytest.raises(ValidationError, match="Invalid entity UUID"):
            asyncio.run(client.get_related_entities(entity_id="invalid-uuid", depth=1))

    def test_get_related_entities_depth_too_low(self):
        """Test ValidationError on depth < 1."""
        client = FalkorDBClient()

        with pytest.raises(ValidationError, match="depth must be in"):
            asyncio.run(client.get_related_entities(entity_id=str(uuid.uuid4()), depth=0))

    def test_get_related_entities_depth_too_high(self):
        """Test ValidationError on depth > 5."""
        client = FalkorDBClient()

        with pytest.raises(ValidationError, match="depth must be in"):
            asyncio.run(client.get_related_entities(entity_id=str(uuid.uuid4()), depth=6))

    def test_get_related_entities_limit_too_high(self):
        """Test ValidationError on limit > 100."""
        client = FalkorDBClient()

        with pytest.raises(ValidationError, match="limit must be in"):
            asyncio.run(
                client.get_related_entities(entity_id=str(uuid.uuid4()), depth=1, limit=101)
            )


# ============================================================================
# graph_query TESTS
# ============================================================================


class TestFalkorDBClientGraphQuery:
    """Test FalkorDBClient.graph_query()."""

    @pytest.fixture
    def mock_client(self, mocker):
        """Create mocked client."""
        client = FalkorDBClient()

        # Mock successful query
        mock_result = MagicMock()
        mock_result.rows = [
            {"m.id": "uuid-1", "m.text": "Test 1"},
            {"m.id": "uuid-2", "m.text": "Test 2"},
        ]
        mock_result.row_count = 2
        mock_result.execution_time_ms = 20

        async def mock_execute(query, params):
            return mock_result

        mocker.patch.object(client, "_execute_cypher", side_effect=mock_execute)
        mocker.patch.object(client, "_initialized", True)
        mocker.patch.object(client, "_schema_ready", True)

        return client

    @pytest.mark.asyncio
    async def test_graph_query_success(self, mock_client):
        """Test successful graph query."""
        result = await mock_client.graph_query(
            cypher="MATCH (m:Memory) RETURN m.id, m.text LIMIT 10"
        )

        # The result is a QueryResult object
        assert hasattr(result, "rows")
        assert hasattr(result, "row_count")
        assert hasattr(result, "execution_time_ms")

    @pytest.mark.asyncio
    async def test_graph_query_with_parameters(self, mock_client):
        """Test query with parameters."""
        result = await mock_client.graph_query(
            cypher="MATCH (m:Memory {id: $id}) RETURN m", parameters={"id": "test-id"}
        )

        assert result.row_count == 2

    def test_graph_query_empty_cypher_raises(self):
        """Test ValidationError on empty cypher."""
        client = FalkorDBClient()

        with pytest.raises(ValidationError, match="cannot be empty"):
            asyncio.run(client.graph_query(cypher=""))

    def test_graph_query_too_long_raises(self):
        """Test ValidationError on query too long."""
        client = FalkorDBClient()

        long_query = "MATCH (n) " * 50000  # > 100K chars

        with pytest.raises(ValidationError, match="exceeds max length"):
            asyncio.run(client.graph_query(cypher=long_query))

    def test_graph_query_non_serializable_params_raises(self):
        """Test ValidationError on non-serializable parameters."""
        client = FalkorDBClient()

        with pytest.raises(ValidationError, match="JSON-serializable"):
            asyncio.run(
                client.graph_query(cypher="MATCH (n) RETURN n", parameters={"binary": b"bytes"})
            )


# ============================================================================
# delete_memory TESTS
# ============================================================================


class TestFalkorDBClientDeleteMemory:
    """Test FalkorDBClient.delete_memory()."""

    @pytest.fixture
    def mock_client_found(self, mocker):
        """Create mocked client for memory found case."""
        client = FalkorDBClient()

        # Mock successful deletion
        mock_result = MagicMock()
        mock_result.rows = [{"deleted_count": 1}]
        mock_result.row_count = 1
        mock_result.execution_time_ms = 5

        async def mock_execute(query, params):
            return mock_result

        mocker.patch.object(client, "_execute_cypher", side_effect=mock_execute)
        mocker.patch.object(client, "_initialized", True)
        mocker.patch.object(client, "_schema_ready", True)

        return client

    @pytest.fixture
    def mock_client_not_found(self, mocker):
        """Create mocked client for memory not found case."""
        client = FalkorDBClient()

        # Mock no deletion
        mock_result = MagicMock()
        mock_result.rows = [{"deleted_count": 0}]
        mock_result.row_count = 1
        mock_result.execution_time_ms = 5

        async def mock_execute(query, params):
            return mock_result

        mocker.patch.object(client, "_execute_cypher", side_effect=mock_execute)
        mocker.patch.object(client, "_initialized", True)
        mocker.patch.object(client, "_schema_ready", True)

        return client

    @pytest.mark.asyncio
    async def test_delete_memory_success(self, mock_client_found):
        """Test successful memory deletion."""
        memory_id = str(uuid.uuid4())

        deleted = await mock_client_found.delete_memory(memory_id)

        assert deleted is True

    @pytest.mark.asyncio
    async def test_delete_memory_not_found(self, mock_client_not_found):
        """Test deletion when memory not found."""
        memory_id = str(uuid.uuid4())

        deleted = await mock_client_not_found.delete_memory(memory_id)

        assert deleted is False

    def test_delete_memory_invalid_uuid_raises(self):
        """Test ValidationError on invalid UUID."""
        client = FalkorDBClient()

        with pytest.raises(ValidationError, match="Invalid memory UUID"):
            asyncio.run(client.delete_memory("invalid-uuid"))


# ============================================================================
# clear_all TESTS
# ============================================================================


class TestFalkorDBClientClearAll:
    """Test FalkorDBClient.clear_all()."""

    @pytest.fixture
    def mock_client(self, mocker):
        """Create mocked client."""
        client = FalkorDBClient()

        # Mock successful clear
        mock_result = MagicMock()
        mock_result.rows = []
        mock_result.row_count = 0
        mock_result.execution_time_ms = 100

        async def mock_execute(query, params):
            return mock_result

        mocker.patch.object(client, "_execute_cypher", side_effect=mock_execute)
        mocker.patch.object(client, "_initialized", True)
        mocker.patch.object(client, "_schema_ready", True)

        return client

    @pytest.mark.asyncio
    async def test_clear_all_success(self, mock_client):
        """Test successful clear all."""
        # Should not raise
        await mock_client.clear_all()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
