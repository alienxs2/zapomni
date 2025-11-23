"""
Unit tests for SchemaManager component.

TDD Approach: Tests written FIRST based on Level 2 & Level 3 specifications.
All tests should FAIL initially (no implementation yet).

Test Coverage:
- __init__: 5 tests
- init_schema: 10 tests (from Level 3 spec)
- create_vector_index: 5 tests
- create_graph_schema: 3 tests
- create_property_indexes: 4 tests
- verify_schema: 5 tests
- migrate: 2 tests (future - not implemented)
- drop_all: 3 tests
- _index_exists: 2 tests (private helper)
- _execute_cypher: 2 tests (private helper)

Total: 41 comprehensive tests
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from falkordb import Graph

# Will be implemented - TDD RED phase
from zapomni_db.schema_manager import SchemaManager
from zapomni_db.exceptions import DatabaseError, QuerySyntaxError


# ============================================================================
# __init__ TESTS (5 tests)
# ============================================================================

class TestSchemaManagerInit:
    """Test SchemaManager initialization."""

    def test_init_with_graph_instance(self):
        """Test initialization with valid Graph instance."""
        mock_graph = MagicMock(spec=Graph)

        manager = SchemaManager(graph=mock_graph)

        assert manager.graph is mock_graph
        assert manager.schema_version == "1.0.0"
        assert manager.initialized is False
        assert manager.logger is not None

    def test_init_with_custom_logger(self):
        """Test initialization with custom logger."""
        mock_graph = MagicMock(spec=Graph)
        mock_logger = MagicMock()

        manager = SchemaManager(graph=mock_graph, logger=mock_logger)

        assert manager.logger is mock_logger

    def test_init_invalid_graph_type_raises(self):
        """Test TypeError when graph is not Graph instance."""
        with pytest.raises(TypeError, match="must be a FalkorDB Graph instance"):
            SchemaManager(graph="not a graph")

    def test_init_none_graph_raises(self):
        """Test TypeError when graph is None."""
        with pytest.raises(TypeError, match="must be a FalkorDB Graph instance"):
            SchemaManager(graph=None)

    def test_init_stores_constants(self):
        """Test that class constants are accessible."""
        mock_graph = MagicMock(spec=Graph)
        manager = SchemaManager(graph=mock_graph)

        assert manager.SCHEMA_VERSION == "1.0.0"
        assert manager.VECTOR_DIMENSION == 768
        assert manager.SIMILARITY_FUNCTION == "cosine"
        assert manager.NODE_MEMORY == "Memory"
        assert manager.NODE_CHUNK == "Chunk"
        assert manager.NODE_ENTITY == "Entity"
        assert manager.NODE_DOCUMENT == "Document"
        assert manager.EDGE_HAS_CHUNK == "HAS_CHUNK"
        assert manager.EDGE_MENTIONS == "MENTIONS"
        assert manager.EDGE_RELATED_TO == "RELATED_TO"
        assert manager.INDEX_VECTOR == "chunk_embedding_idx"
        assert manager.INDEX_MEMORY_ID == "memory_id_idx"
        assert manager.INDEX_ENTITY_NAME == "entity_name_idx"
        assert manager.INDEX_TIMESTAMP == "timestamp_idx"


# ============================================================================
# init_schema TESTS (10 tests from Level 3 spec)
# ============================================================================

class TestSchemaManagerInitSchema:
    """Test init_schema() based on Level 3 function specification."""

    @pytest.fixture
    def mock_manager(self):
        """Create SchemaManager with mocked graph."""
        mock_graph = MagicMock(spec=Graph)
        manager = SchemaManager(graph=mock_graph)

        # Mock helper methods
        manager.create_vector_index = Mock()
        manager.create_graph_schema = Mock()
        manager.create_property_indexes = Mock()

        return manager

    def test_init_schema_success(self, mock_manager):
        """Test successful schema initialization."""
        mock_manager.init_schema()

        # Verify all sub-methods called
        mock_manager.create_vector_index.assert_called_once()
        mock_manager.create_graph_schema.assert_called_once()
        mock_manager.create_property_indexes.assert_called_once()

        # Verify initialized flag set
        assert mock_manager.initialized is True

    def test_init_schema_idempotent(self, mock_manager):
        """Test calling init_schema() multiple times is safe."""
        # First call
        mock_manager.init_schema()
        assert mock_manager.initialized is True

        # Reset mocks
        mock_manager.create_vector_index.reset_mock()
        mock_manager.create_graph_schema.reset_mock()
        mock_manager.create_property_indexes.reset_mock()

        # Second call - should skip (already initialized)
        mock_manager.init_schema()

        # Should not call sub-methods again
        mock_manager.create_vector_index.assert_not_called()
        mock_manager.create_graph_schema.assert_not_called()
        mock_manager.create_property_indexes.assert_not_called()

    def test_init_schema_partial_indexes(self, mock_manager):
        """Test schema creation with some indexes already existing."""
        # Simulate partial initialization - vector index exists
        mock_manager.create_vector_index = Mock()  # No-op if exists
        mock_manager.create_graph_schema = Mock()
        mock_manager.create_property_indexes = Mock()

        mock_manager.init_schema()

        # All methods still called (idempotent checks inside)
        mock_manager.create_vector_index.assert_called_once()
        mock_manager.create_graph_schema.assert_called_once()
        mock_manager.create_property_indexes.assert_called_once()

    def test_init_schema_database_error(self, mock_manager):
        """Test DatabaseError propagates correctly."""
        # Simulate database error during vector index creation
        mock_manager.create_vector_index = Mock(
            side_effect=DatabaseError("Connection lost")
        )

        with pytest.raises(DatabaseError, match="Connection lost"):
            mock_manager.init_schema()

        # initialized flag should remain False
        assert mock_manager.initialized is False

    def test_init_schema_query_syntax_error(self, mock_manager):
        """Test QuerySyntaxError propagates correctly."""
        # Simulate syntax error in property index creation
        mock_manager.create_property_indexes = Mock(
            side_effect=QuerySyntaxError("Invalid Cypher syntax")
        )

        with pytest.raises(QuerySyntaxError, match="Invalid Cypher"):
            mock_manager.init_schema()

        assert mock_manager.initialized is False

    def test_init_schema_sets_initialized_flag(self, mock_manager):
        """Test that initialized flag is set to True on success."""
        assert mock_manager.initialized is False

        mock_manager.init_schema()

        assert mock_manager.initialized is True

    def test_init_schema_creates_vector_index(self, mock_manager):
        """Test that vector index creation is called."""
        mock_manager.init_schema()

        mock_manager.create_vector_index.assert_called_once()

    def test_init_schema_creates_property_indexes(self, mock_manager):
        """Test that property indexes creation is called."""
        mock_manager.init_schema()

        mock_manager.create_property_indexes.assert_called_once()

    def test_init_schema_logging(self, mock_manager):
        """Test that schema initialization is logged."""
        mock_logger = MagicMock()
        mock_manager.logger = mock_logger

        mock_manager.init_schema()

        # Should have log calls (start and success)
        assert mock_logger.info.called or mock_logger.debug.called

    def test_init_schema_integration_real_db(self):
        """Test schema initialization with real FalkorDB (integration test)."""
        # This test requires real FalkorDB connection
        # Skipped in unit tests, run in integration tests
        pytest.skip("Integration test - requires FalkorDB instance")


# ============================================================================
# create_vector_index TESTS (5 tests)
# ============================================================================

class TestSchemaManagerCreateVectorIndex:
    """Test create_vector_index() method."""

    @pytest.fixture
    def mock_manager(self):
        """Create SchemaManager with mocked helpers."""
        mock_graph = MagicMock(spec=Graph)
        manager = SchemaManager(graph=mock_graph)
        manager._index_exists = Mock(return_value=False)
        manager._execute_cypher = Mock()
        return manager

    def test_create_vector_index_success(self, mock_manager):
        """Test successful vector index creation."""
        mock_manager.create_vector_index()

        # Should check if index exists
        mock_manager._index_exists.assert_called_once_with("chunk_embedding_idx")

        # Should execute CREATE VECTOR INDEX
        mock_manager._execute_cypher.assert_called_once()
        cypher_query = mock_manager._execute_cypher.call_args[0][0]
        assert "CREATE VECTOR INDEX" in cypher_query
        assert "chunk_embedding_idx" in cypher_query
        assert "Chunk" in cypher_query
        assert "embedding" in cypher_query

    def test_create_vector_index_already_exists(self, mock_manager):
        """Test index creation skipped when already exists (idempotent)."""
        mock_manager._index_exists = Mock(return_value=True)

        mock_manager.create_vector_index()

        # Should check existence
        mock_manager._index_exists.assert_called_once()

        # Should NOT execute CREATE INDEX (already exists)
        mock_manager._execute_cypher.assert_not_called()

    def test_create_vector_index_correct_config(self, mock_manager):
        """Test vector index created with correct configuration."""
        mock_manager.create_vector_index()

        cypher_query = mock_manager._execute_cypher.call_args[0][0]

        # Verify dimension: 768
        assert "768" in cypher_query

        # Verify similarity: cosine
        assert "cosine" in cypher_query

    def test_create_vector_index_database_error(self, mock_manager):
        """Test DatabaseError on index creation failure."""
        mock_manager._execute_cypher = Mock(
            side_effect=DatabaseError("Disk full")
        )

        with pytest.raises(DatabaseError, match="Disk full"):
            mock_manager.create_vector_index()

    def test_create_vector_index_query_syntax_error(self, mock_manager):
        """Test QuerySyntaxError on invalid Cypher."""
        mock_manager._execute_cypher = Mock(
            side_effect=QuerySyntaxError("Invalid syntax")
        )

        with pytest.raises(QuerySyntaxError, match="Invalid syntax"):
            mock_manager.create_vector_index()


# ============================================================================
# create_graph_schema TESTS (3 tests)
# ============================================================================

class TestSchemaManagerCreateGraphSchema:
    """Test create_graph_schema() method."""

    @pytest.fixture
    def mock_manager(self):
        """Create SchemaManager with mocked graph."""
        mock_graph = MagicMock(spec=Graph)
        return SchemaManager(graph=mock_graph)

    def test_create_graph_schema_success(self, mock_manager):
        """Test successful graph schema definition."""
        # Should not raise
        mock_manager.create_graph_schema()

        # Verify logger was used
        assert True  # Schema definition is mostly documentation

    def test_create_graph_schema_idempotent(self, mock_manager):
        """Test calling multiple times is safe."""
        mock_manager.create_graph_schema()
        mock_manager.create_graph_schema()

        # Should not raise (labels are implicit in FalkorDB)
        assert True

    def test_create_graph_schema_logging(self, mock_manager):
        """Test that schema definition is logged."""
        mock_logger = MagicMock()
        mock_manager.logger = mock_logger

        mock_manager.create_graph_schema()

        # Should log schema info
        assert mock_logger.info.called or mock_logger.debug.called


# ============================================================================
# create_property_indexes TESTS (4 tests)
# ============================================================================

class TestSchemaManagerCreatePropertyIndexes:
    """Test create_property_indexes() method."""

    @pytest.fixture
    def mock_manager(self):
        """Create SchemaManager with mocked helpers."""
        mock_graph = MagicMock(spec=Graph)
        manager = SchemaManager(graph=mock_graph)
        manager._index_exists = Mock(return_value=False)
        manager._execute_cypher = Mock()
        return manager

    def test_create_property_indexes_success(self, mock_manager):
        """Test all property indexes created successfully."""
        mock_manager.create_property_indexes()

        # Should check for all 4 indexes
        expected_indexes = [
            "memory_id_idx",
            "entity_name_idx",
            "timestamp_idx",
            "chunk_memory_id_idx"
        ]

        actual_calls = [call[0][0] for call in mock_manager._index_exists.call_args_list]

        for idx_name in expected_indexes:
            assert idx_name in actual_calls, f"Missing index check: {idx_name}"

    def test_create_property_indexes_skip_existing(self, mock_manager):
        """Test existing indexes are skipped (idempotent)."""
        # Simulate memory_id_idx already exists
        def index_exists_side_effect(index_name):
            return index_name == "memory_id_idx"

        mock_manager._index_exists = Mock(side_effect=index_exists_side_effect)

        mock_manager.create_property_indexes()

        # Should execute CREATE INDEX for 3 missing indexes (not for memory_id_idx)
        assert mock_manager._execute_cypher.call_count == 3

    def test_create_property_indexes_correct_cypher(self, mock_manager):
        """Test property indexes created with correct Cypher syntax."""
        mock_manager.create_property_indexes()

        # Check that Cypher queries contain expected patterns
        cypher_calls = [call[0][0] for call in mock_manager._execute_cypher.call_args_list]

        # Should have 4 CREATE INDEX queries
        assert len(cypher_calls) == 4

        # All should be CREATE INDEX
        for query in cypher_calls:
            assert "CREATE INDEX" in query

    def test_create_property_indexes_database_error(self, mock_manager):
        """Test DatabaseError propagates correctly."""
        mock_manager._execute_cypher = Mock(
            side_effect=DatabaseError("Index creation failed")
        )

        with pytest.raises(DatabaseError, match="Index creation failed"):
            mock_manager.create_property_indexes()


# ============================================================================
# verify_schema TESTS (5 tests)
# ============================================================================

class TestSchemaManagerVerifySchema:
    """Test verify_schema() method."""

    @pytest.fixture
    def mock_manager(self):
        """Create SchemaManager with mocked graph."""
        mock_graph = MagicMock(spec=Graph)
        manager = SchemaManager(graph=mock_graph)

        # Mock SHOW INDEXES query result
        mock_result = MagicMock()
        mock_result.result_set = [
            ["chunk_embedding_idx", "vector", "Chunk", "embedding"],
            ["memory_id_idx", "property", "Memory", "id"],
            ["entity_name_idx", "property", "Entity", "name"],
            ["timestamp_idx", "property", "Memory", "timestamp"],
        ]
        mock_graph.query = Mock(return_value=mock_result)

        return manager

    def test_verify_schema_complete(self, mock_manager):
        """Test verification returns initialized=True when all indexes exist."""
        status = mock_manager.verify_schema()

        assert isinstance(status, dict)
        assert status["version"] == "1.0.0"
        assert status["initialized"] is True
        assert "indexes" in status
        assert "node_labels" in status
        assert "edge_labels" in status
        assert status["issues"] == []

    def test_verify_schema_incomplete(self):
        """Test verification detects missing indexes."""
        mock_graph = MagicMock(spec=Graph)
        manager = SchemaManager(graph=mock_graph)

        # Mock empty SHOW INDEXES result (no indexes)
        mock_result = MagicMock()
        mock_result.result_set = []
        mock_graph.query = Mock(return_value=mock_result)

        status = manager.verify_schema()

        assert status["initialized"] is False
        assert len(status["issues"]) > 0
        assert any("Vector index not found" in issue for issue in status["issues"])

    def test_verify_schema_vector_index_config(self, mock_manager):
        """Test vector index configuration is reported."""
        status = mock_manager.verify_schema()

        vector_idx = status["indexes"]["vector_index"]
        assert vector_idx["exists"] is True
        assert vector_idx["name"] == "chunk_embedding_idx"
        assert vector_idx["dimension"] == 768
        assert vector_idx["similarity"] == "cosine"

    def test_verify_schema_property_indexes(self, mock_manager):
        """Test property indexes are verified."""
        status = mock_manager.verify_schema()

        prop_indexes = status["indexes"]["property_indexes"]

        assert "memory_id_idx" in prop_indexes
        assert prop_indexes["memory_id_idx"]["exists"] is True
        assert prop_indexes["memory_id_idx"]["property"] == "id"

        assert "entity_name_idx" in prop_indexes
        assert prop_indexes["entity_name_idx"]["exists"] is True

    def test_verify_schema_database_error(self):
        """Test DatabaseError when verification fails."""
        mock_graph = MagicMock(spec=Graph)
        manager = SchemaManager(graph=mock_graph)

        # Simulate query failure
        mock_graph.query = Mock(side_effect=Exception("Query failed"))

        with pytest.raises(DatabaseError):
            manager.verify_schema()


# ============================================================================
# migrate TESTS (2 tests - future feature)
# ============================================================================

class TestSchemaManagerMigrate:
    """Test migrate() method (future feature)."""

    def test_migrate_not_implemented(self):
        """Test migrate raises NotImplementedError."""
        mock_graph = MagicMock(spec=Graph)
        manager = SchemaManager(graph=mock_graph)

        with pytest.raises(NotImplementedError, match="not implemented"):
            manager.migrate(from_version="1.0.0", to_version="1.1.0")

    def test_migrate_placeholder_signature(self):
        """Test migrate has correct signature (for future)."""
        mock_graph = MagicMock(spec=Graph)
        manager = SchemaManager(graph=mock_graph)

        # Should accept from_version and to_version
        import inspect
        sig = inspect.signature(manager.migrate)
        params = list(sig.parameters.keys())

        assert "from_version" in params
        assert "to_version" in params


# ============================================================================
# drop_all TESTS (3 tests)
# ============================================================================

class TestSchemaManagerDropAll:
    """Test drop_all() method."""

    @pytest.fixture
    def mock_manager(self):
        """Create SchemaManager with mocked graph."""
        mock_graph = MagicMock(spec=Graph)
        manager = SchemaManager(graph=mock_graph)
        manager._execute_cypher = Mock()
        return manager

    def test_drop_all_success(self, mock_manager):
        """Test successful drop of all data and schema."""
        mock_manager.initialized = True

        mock_manager.drop_all()

        # Should execute delete queries
        assert mock_manager._execute_cypher.called

        # Should set initialized to False
        assert mock_manager.initialized is False

    def test_drop_all_idempotent(self, mock_manager):
        """Test calling drop_all on empty graph doesn't error."""
        mock_manager.drop_all()
        mock_manager.drop_all()

        # Should not raise (idempotent)
        assert True

    def test_drop_all_database_error(self, mock_manager):
        """Test DatabaseError when drop operation fails."""
        mock_manager._execute_cypher = Mock(
            side_effect=DatabaseError("Connection lost during delete")
        )

        with pytest.raises(DatabaseError, match="Connection lost"):
            mock_manager.drop_all()


# ============================================================================
# _index_exists TESTS (2 tests - private helper)
# ============================================================================

class TestSchemaManagerIndexExists:
    """Test _index_exists() private helper."""

    def test_index_exists_true(self):
        """Test _index_exists returns True when index found."""
        mock_graph = MagicMock(spec=Graph)
        manager = SchemaManager(graph=mock_graph)

        # Mock query result with index
        mock_result = MagicMock()
        mock_result.result_set = [["chunk_embedding_idx"]]
        mock_graph.query = Mock(return_value=mock_result)

        exists = manager._index_exists("chunk_embedding_idx")

        assert exists is True

    def test_index_exists_false(self):
        """Test _index_exists returns False when index not found."""
        mock_graph = MagicMock(spec=Graph)
        manager = SchemaManager(graph=mock_graph)

        # Mock query result with no index
        mock_result = MagicMock()
        mock_result.result_set = []
        mock_graph.query = Mock(return_value=mock_result)

        exists = manager._index_exists("nonexistent_idx")

        assert exists is False


# ============================================================================
# _execute_cypher TESTS (2 tests - private helper)
# ============================================================================

class TestSchemaManagerExecuteCypher:
    """Test _execute_cypher() private helper."""

    def test_execute_cypher_success(self):
        """Test successful Cypher execution."""
        mock_graph = MagicMock(spec=Graph)
        manager = SchemaManager(graph=mock_graph)

        manager._execute_cypher("CREATE INDEX test_idx FOR (n:Node) ON (n.id)")

        # Should call graph.query
        mock_graph.query.assert_called_once()

    def test_execute_cypher_database_error(self):
        """Test DatabaseError when execution fails."""
        mock_graph = MagicMock(spec=Graph)
        manager = SchemaManager(graph=mock_graph)

        # Simulate query failure
        mock_graph.query = Mock(side_effect=Exception("Execution failed"))

        with pytest.raises(DatabaseError):
            manager._execute_cypher("INVALID CYPHER")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
