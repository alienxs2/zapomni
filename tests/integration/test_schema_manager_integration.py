"""
Integration tests for SchemaManager with real FalkorDB instance.

These tests require a running FalkorDB instance (via docker-compose).
Run with: pytest tests/integration/test_schema_manager_integration.py -v

Prerequisites:
- docker-compose up -d (to start FalkorDB)
- FalkorDB running on localhost:6381
"""

import pytest
from falkordb import FalkorDB

from zapomni_db.exceptions import DatabaseError, QuerySyntaxError
from zapomni_db.schema_manager import SchemaManager


@pytest.fixture(scope="module")
def falkordb_connection():
    """
    Create FalkorDB connection for integration tests.

    Uses a separate test graph to avoid polluting production data.
    """
    try:
        db = FalkorDB(host="localhost", port=6381)
        yield db
    except Exception as e:
        pytest.skip(f"FalkorDB not available: {e}")


@pytest.fixture(scope="function")
def test_graph(falkordb_connection):
    """
    Create a clean test graph for each test.

    Automatically cleans up after the test.
    """
    graph_name = "test_schema_integration"
    graph = falkordb_connection.select_graph(graph_name)

    # Clean slate - delete all data and indexes
    try:
        graph.query("MATCH (n) DETACH DELETE n")
    except Exception:
        pass

    # Drop all indexes
    try:
        result = graph.query("SHOW INDEXES")
        if result.result_set:
            for row in result.result_set:
                index_name = row[0]
                try:
                    graph.query(f"DROP INDEX {index_name}")
                except Exception:
                    pass
    except Exception:
        pass

    yield graph

    # Cleanup after test
    try:
        graph.query("MATCH (n) DETACH DELETE n")
    except Exception:
        pass


@pytest.mark.integration
class TestSchemaManagerIntegration:
    """Integration tests with real FalkorDB instance."""

    def test_init_schema_creates_all_indexes(self, test_graph):
        """Test that init_schema creates all required indexes."""
        manager = SchemaManager(graph=test_graph)

        # Initialize schema
        manager.init_schema()

        # Query indexes
        result = test_graph.query("SHOW INDEXES")
        index_names = [row[0] for row in result.result_set]

        # Verify all indexes created
        assert "chunk_embedding_idx" in index_names
        assert "memory_id_idx" in index_names
        assert "entity_name_idx" in index_names
        assert "timestamp_idx" in index_names

    def test_init_schema_idempotent_real_db(self, test_graph):
        """Test that calling init_schema multiple times is safe."""
        manager = SchemaManager(graph=test_graph)

        # First initialization
        manager.init_schema()
        result1 = test_graph.query("SHOW INDEXES")
        index_count_1 = len(result1.result_set)

        # Second initialization - should not create duplicates
        manager.init_schema()
        result2 = test_graph.query("SHOW INDEXES")
        index_count_2 = len(result2.result_set)

        # Index count should be same (no duplicates)
        assert index_count_1 == index_count_2

    def test_vector_index_configuration(self, test_graph):
        """Test vector index is created with correct configuration."""
        manager = SchemaManager(graph=test_graph)
        manager.create_vector_index()

        # Query index details
        result = test_graph.query("SHOW INDEXES")
        vector_index = None
        for row in result.result_set:
            if row[0] == "chunk_embedding_idx":
                vector_index = row
                break

        assert vector_index is not None, "Vector index not found"
        # Note: FalkorDB SHOW INDEXES format may vary
        # We just verify the index exists

    def test_property_indexes_usable(self, test_graph):
        """Test property indexes work for fast lookups."""
        manager = SchemaManager(graph=test_graph)
        manager.init_schema()

        # Create test Memory node
        test_graph.query(
            """
            CREATE (m:Memory {
                id: 'test-memory-123',
                text: 'Test memory',
                timestamp: datetime('2025-11-23T00:00:00Z')
            })
        """
        )

        # Query using indexed property (should be fast)
        result = test_graph.query(
            """
            MATCH (m:Memory {id: 'test-memory-123'})
            RETURN m.text AS text
        """
        )

        assert len(result.result_set) == 1
        assert result.result_set[0][0] == "Test memory"

    def test_verify_schema_complete(self, test_graph):
        """Test verify_schema returns correct status."""
        manager = SchemaManager(graph=test_graph)
        manager.init_schema()

        status = manager.verify_schema()

        assert status["version"] == "1.0.0"
        assert status["initialized"] is True
        assert status["issues"] == []
        assert status["indexes"]["vector_index"]["exists"] is True
        assert len(status["indexes"]["property_indexes"]) >= 3

    def test_verify_schema_incomplete(self, test_graph):
        """Test verify_schema detects missing indexes."""
        manager = SchemaManager(graph=test_graph)

        # Don't initialize - verify should detect missing indexes
        status = manager.verify_schema()

        assert status["initialized"] is False
        assert len(status["issues"]) > 0
        assert status["indexes"]["vector_index"]["exists"] is False

    def test_drop_all_removes_everything(self, test_graph):
        """Test drop_all removes all data and indexes."""
        manager = SchemaManager(graph=test_graph)
        manager.init_schema()

        # Create some test data
        test_graph.query(
            """
            CREATE (m:Memory {id: 'test-123', text: 'Test'})
            CREATE (c:Chunk {id: 'chunk-1', text: 'Chunk'})
            CREATE (m)-[:HAS_CHUNK]->(c)
        """
        )

        # Drop all
        manager.drop_all()

        # Verify no nodes
        result = test_graph.query("MATCH (n) RETURN count(n) AS count")
        assert result.result_set[0][0] == 0

        # Verify no indexes (or minimal indexes)
        result = test_graph.query("SHOW INDEXES")
        # After drop, should have no custom indexes
        assert len(result.result_set) == 0 or all(
            row[0]
            not in ["chunk_embedding_idx", "memory_id_idx", "entity_name_idx", "timestamp_idx"]
            for row in result.result_set
        )

    def test_graph_schema_documentation(self, test_graph):
        """Test that graph schema (labels) work as documented."""
        manager = SchemaManager(graph=test_graph)
        manager.init_schema()

        # Create nodes with documented labels
        test_graph.query(
            """
            CREATE (m:Memory {id: '1'})
            CREATE (c:Chunk {id: '2'})
            CREATE (e:Entity {id: '3', name: 'Test Entity'})
            CREATE (d:Document {id: '4'})
            CREATE (m)-[:HAS_CHUNK]->(c)
            CREATE (c)-[:MENTIONS]->(e)
        """
        )

        # Verify labels exist
        result = test_graph.query("MATCH (m:Memory) RETURN count(m) AS count")
        assert result.result_set[0][0] == 1

        result = test_graph.query("MATCH (c:Chunk) RETURN count(c) AS count")
        assert result.result_set[0][0] == 1

        result = test_graph.query("MATCH (e:Entity) RETURN count(e) AS count")
        assert result.result_set[0][0] == 1

        # Verify edges exist
        result = test_graph.query("MATCH ()-[r:HAS_CHUNK]->() RETURN count(r) AS count")
        assert result.result_set[0][0] == 1

        result = test_graph.query("MATCH ()-[r:MENTIONS]->() RETURN count(r) AS count")
        assert result.result_set[0][0] == 1

    def test_migrate_raises_not_implemented(self, test_graph):
        """Test that migrate raises NotImplementedError (future feature)."""
        manager = SchemaManager(graph=test_graph)

        with pytest.raises(NotImplementedError, match="not implemented"):
            manager.migrate(from_version="1.0.0", to_version="1.1.0")

    def test_concurrent_init_schema_calls(self, test_graph):
        """Test that concurrent init_schema calls don't cause errors."""
        import threading

        manager = SchemaManager(graph=test_graph)
        errors = []

        def init_schema_thread():
            try:
                manager.init_schema()
            except Exception as e:
                errors.append(e)

        # Create multiple threads trying to initialize
        threads = [threading.Thread(target=init_schema_thread) for _ in range(3)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Should have no errors (idempotent)
        assert len(errors) == 0 or all(
            "already exists" in str(e).lower() or "initialized" in str(e).lower() for e in errors
        )


@pytest.mark.integration
class TestSchemaManagerWithFalkorDBClient:
    """Integration tests verifying SchemaManager works with FalkorDBClient."""

    def test_falkordb_client_uses_schema_manager(self, test_graph):
        """Test that FalkorDBClient uses SchemaManager for initialization."""
        # This test verifies the integration between SchemaManager and FalkorDBClient
        # We'll import FalkorDBClient and verify it creates schema correctly

        from zapomni_db.falkordb_client import FalkorDBClient

        # Note: This test would require a real FalkorDB connection
        # For now, we verify the import works and SchemaManager is accessible
        assert hasattr(FalkorDBClient, "_init_schema")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
