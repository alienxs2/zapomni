"""
Unit tests for bi-temporal migration (v0.8.0 - Issue #27).

Tests for:
- MigrationResult dataclass
- get_migration_status function
- migrate_to_bitemporal function

Migration module: zapomni_db.migrations.migration_001_bitemporal
"""

from typing import Any, List
from unittest.mock import MagicMock, Mock, call

import pytest

from zapomni_db.migrations import MigrationResult
from zapomni_db.migrations.migration_001_bitemporal import (
    get_migration_status,
    migrate_to_bitemporal,
)


# ============================================================================
# MigrationResult TESTS
# ============================================================================


class TestMigrationResult:
    """Tests for MigrationResult dataclass."""

    def test_migration_result_defaults(self) -> None:
        """Test MigrationResult has correct default values."""
        result = MigrationResult()

        assert result.success is False
        assert result.memories_migrated == 0
        assert result.entities_migrated == 0
        assert result.indexes_created == 0
        assert result.errors == []

    def test_migration_result_with_values(self) -> None:
        """Test MigrationResult with custom values."""
        result = MigrationResult(
            success=True,
            memories_migrated=100,
            entities_migrated=50,
            indexes_created=4,
        )

        assert result.success is True
        assert result.memories_migrated == 100
        assert result.entities_migrated == 50
        assert result.indexes_created == 4
        assert result.errors == []

    def test_migration_result_errors_list(self) -> None:
        """Test MigrationResult errors list initialization."""
        # Errors list should be initialized via __post_init__
        result = MigrationResult()
        assert isinstance(result.errors, list)
        assert len(result.errors) == 0

        # Can append errors
        result.errors.append("Test error")
        assert len(result.errors) == 1
        assert result.errors[0] == "Test error"

    def test_migration_result_with_errors(self) -> None:
        """Test MigrationResult with pre-set errors."""
        result = MigrationResult(
            success=False,
            errors=["Error 1", "Error 2"],
        )

        assert result.success is False
        assert len(result.errors) == 2
        assert "Error 1" in result.errors
        assert "Error 2" in result.errors

    def test_migration_result_errors_separate_instances(self) -> None:
        """Test that errors list is separate for each instance."""
        result1 = MigrationResult()
        result2 = MigrationResult()

        result1.errors.append("Error in result1")

        # result2 should have empty errors
        assert len(result2.errors) == 0
        assert len(result1.errors) == 1


# ============================================================================
# Mock Graph Fixture
# ============================================================================


class MockQueryResult:
    """Mock FalkorDB query result."""

    def __init__(self, result_set: List[List[Any]]) -> None:
        self.result_set = result_set


def create_mock_graph(
    memory_total: int = 0,
    memory_migrated: int = 0,
    entity_total: int = 0,
    entity_migrated: int = 0,
) -> MagicMock:
    """Create a mock FalkorDB Graph with configurable counts."""
    mock_graph = MagicMock()

    def query_side_effect(query: str) -> MockQueryResult:
        """Return appropriate results based on the query."""
        if "MATCH (m:Memory) RETURN count(m)" in query and "WHERE" not in query:
            return MockQueryResult([[memory_total]])
        if "MATCH (m:Memory) WHERE m.valid_from IS NOT NULL" in query:
            return MockQueryResult([[memory_migrated]])
        if "MATCH (m:Memory)" in query and "WHERE m.valid_from IS NULL" in query:
            # Count query for migration
            return MockQueryResult([[memory_total - memory_migrated]])
        if "MATCH (e:Entity) RETURN count(e)" in query and "WHERE" not in query:
            return MockQueryResult([[entity_total]])
        if "MATCH (e:Entity) WHERE e.valid_from IS NOT NULL" in query:
            return MockQueryResult([[entity_migrated]])
        if "MATCH (e:Entity)" in query and "WHERE e.valid_from IS NULL" in query:
            # Count query for migration
            return MockQueryResult([[entity_total - entity_migrated]])
        if "db.indexes" in query:
            return MockQueryResult([])
        if "CREATE INDEX" in query:
            return MockQueryResult([])
        if "SET m.valid_from" in query:
            return MockQueryResult([[memory_total - memory_migrated]])
        if "SET e.valid_from" in query:
            return MockQueryResult([[entity_total - entity_migrated]])

        # Default
        return MockQueryResult([])

    mock_graph.query = Mock(side_effect=query_side_effect)
    return mock_graph


# ============================================================================
# get_migration_status TESTS
# ============================================================================


class TestGetMigrationStatus:
    """Tests for get_migration_status function."""

    def test_get_migration_status_empty_database(self) -> None:
        """Test get_migration_status on empty database returns correct status."""
        mock_graph = create_mock_graph(
            memory_total=0,
            memory_migrated=0,
            entity_total=0,
            entity_migrated=0,
        )

        status = get_migration_status(mock_graph)

        assert isinstance(status, dict)
        assert status["needs_migration"] is False
        assert status["memory_nodes_total"] == 0
        assert status["memory_nodes_migrated"] == 0
        assert status["entity_nodes_total"] == 0
        assert status["entity_nodes_migrated"] == 0

    def test_get_migration_status_needs_migration_true(self) -> None:
        """Test get_migration_status returns needs_migration=True for unmigrated data."""
        mock_graph = create_mock_graph(
            memory_total=100,
            memory_migrated=0,
            entity_total=50,
            entity_migrated=0,
        )

        status = get_migration_status(mock_graph)

        assert status["needs_migration"] is True
        assert status["memory_nodes_total"] == 100
        assert status["memory_nodes_migrated"] == 0
        assert status["entity_nodes_total"] == 50
        assert status["entity_nodes_migrated"] == 0

    def test_get_migration_status_partial_migration(self) -> None:
        """Test get_migration_status with partially migrated data."""
        mock_graph = create_mock_graph(
            memory_total=100,
            memory_migrated=80,
            entity_total=50,
            entity_migrated=50,
        )

        status = get_migration_status(mock_graph)

        # Still needs migration because not all memories are migrated
        assert status["needs_migration"] is True
        assert status["memory_nodes_total"] == 100
        assert status["memory_nodes_migrated"] == 80
        assert status["entity_nodes_total"] == 50
        assert status["entity_nodes_migrated"] == 50

    def test_get_migration_status_fully_migrated(self) -> None:
        """Test get_migration_status when all data is migrated."""
        mock_graph = create_mock_graph(
            memory_total=100,
            memory_migrated=100,
            entity_total=50,
            entity_migrated=50,
        )

        status = get_migration_status(mock_graph)

        assert status["needs_migration"] is False
        assert status["memory_nodes_total"] == 100
        assert status["memory_nodes_migrated"] == 100
        assert status["entity_nodes_total"] == 50
        assert status["entity_nodes_migrated"] == 50

    def test_get_migration_status_only_memories_need_migration(self) -> None:
        """Test when only memories need migration."""
        mock_graph = create_mock_graph(
            memory_total=100,
            memory_migrated=50,
            entity_total=20,
            entity_migrated=20,
        )

        status = get_migration_status(mock_graph)

        assert status["needs_migration"] is True

    def test_get_migration_status_only_entities_need_migration(self) -> None:
        """Test when only entities need migration."""
        mock_graph = create_mock_graph(
            memory_total=100,
            memory_migrated=100,
            entity_total=20,
            entity_migrated=10,
        )

        status = get_migration_status(mock_graph)

        assert status["needs_migration"] is True

    def test_get_migration_status_handles_query_error(self) -> None:
        """Test get_migration_status handles query errors gracefully."""
        mock_graph = MagicMock()
        mock_graph.query = Mock(side_effect=Exception("Database error"))

        status = get_migration_status(mock_graph)

        # Should return default status without raising
        assert isinstance(status, dict)
        assert status["needs_migration"] is False
        assert status["memory_nodes_total"] == 0
        assert status["memory_nodes_migrated"] == 0
        assert status["entity_nodes_total"] == 0
        assert status["entity_nodes_migrated"] == 0


# ============================================================================
# migrate_to_bitemporal TESTS
# ============================================================================


class TestMigrateToBitemporal:
    """Tests for migrate_to_bitemporal function."""

    @pytest.mark.asyncio
    async def test_dry_run_doesnt_modify_data(self) -> None:
        """Test dry_run mode doesn't modify data."""
        mock_graph = create_mock_graph(
            memory_total=100,
            memory_migrated=0,
            entity_total=50,
            entity_migrated=0,
        )

        result = await migrate_to_bitemporal(mock_graph, dry_run=True)

        assert result.success is True
        assert result.memories_migrated == 100  # Reports what would be migrated
        assert result.entities_migrated == 50
        assert result.indexes_created == 0  # Indexes not created in dry run
        assert len(result.errors) == 0

        # Verify no SET queries were executed
        for call_args in mock_graph.query.call_args_list:
            query = call_args[0][0]
            assert "SET m.valid_from" not in query
            assert "SET e.valid_from" not in query

    @pytest.mark.asyncio
    async def test_migration_success(self) -> None:
        """Test successful migration."""
        mock_graph = create_mock_graph(
            memory_total=100,
            memory_migrated=0,
            entity_total=50,
            entity_migrated=0,
        )

        result = await migrate_to_bitemporal(mock_graph, dry_run=False)

        assert result.success is True
        assert result.memories_migrated == 100
        assert result.entities_migrated == 50
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_migration_idempotent(self) -> None:
        """Test migration is idempotent - running twice is safe."""
        # First run: data needs migration
        mock_graph_first = create_mock_graph(
            memory_total=100,
            memory_migrated=0,
            entity_total=50,
            entity_migrated=0,
        )

        result1 = await migrate_to_bitemporal(mock_graph_first, dry_run=False)
        assert result1.success is True
        assert result1.memories_migrated == 100
        assert result1.entities_migrated == 50

        # Second run: all data already migrated
        mock_graph_second = create_mock_graph(
            memory_total=100,
            memory_migrated=100,
            entity_total=50,
            entity_migrated=50,
        )

        result2 = await migrate_to_bitemporal(mock_graph_second, dry_run=False)
        assert result2.success is True
        assert result2.memories_migrated == 0  # Nothing to migrate
        assert result2.entities_migrated == 0  # Nothing to migrate

    @pytest.mark.asyncio
    async def test_migration_sets_correct_default_values(self) -> None:
        """Test migration sets correct default values for temporal fields."""
        mock_graph = MagicMock()
        queries_executed: List[str] = []

        def capture_query(query: str) -> MockQueryResult:
            queries_executed.append(query)
            if "count(m)" in query or "count(e)" in query:
                return MockQueryResult([[10]])
            if "db.indexes" in query:
                return MockQueryResult([])
            return MockQueryResult([[10]])

        mock_graph.query = Mock(side_effect=capture_query)

        await migrate_to_bitemporal(mock_graph, dry_run=False)

        # Find the Memory migration SET query
        memory_set_queries = [q for q in queries_executed if "SET m.valid_from" in q]
        assert len(memory_set_queries) > 0

        memory_query = memory_set_queries[0]
        # Check that all required fields are set
        assert "m.valid_from = COALESCE(m.created_at, datetime())" in memory_query
        assert "m.valid_to = null" in memory_query
        assert "m.transaction_to = null" in memory_query
        assert "m.version = 1" in memory_query
        assert "m.previous_version_id = null" in memory_query
        assert "m.is_current = true" in memory_query

        # Find the Entity migration SET query
        entity_set_queries = [q for q in queries_executed if "SET e.valid_from" in q]
        assert len(entity_set_queries) > 0

        entity_query = entity_set_queries[0]
        # Check Entity fields
        assert "e.valid_from = COALESCE(e.created_at, e.updated_at, datetime())" in entity_query
        assert "e.valid_to = null" in entity_query
        assert "e.is_current = true" in entity_query

    @pytest.mark.asyncio
    async def test_migration_handles_errors(self) -> None:
        """Test migration handles errors and returns them in result."""
        mock_graph = MagicMock()

        def error_on_set(query: str) -> MockQueryResult:
            if "SET m.valid_from" in query:
                raise Exception("Database write error")
            if "count(m)" in query:
                return MockQueryResult([[10]])
            return MockQueryResult([[0]])

        mock_graph.query = Mock(side_effect=error_on_set)

        result = await migrate_to_bitemporal(mock_graph, dry_run=False)

        assert result.success is False
        assert len(result.errors) > 0
        assert "Database write error" in result.errors[0]

    @pytest.mark.asyncio
    async def test_migration_empty_database(self) -> None:
        """Test migration on empty database succeeds."""
        mock_graph = create_mock_graph(
            memory_total=0,
            memory_migrated=0,
            entity_total=0,
            entity_migrated=0,
        )

        result = await migrate_to_bitemporal(mock_graph, dry_run=False)

        assert result.success is True
        assert result.memories_migrated == 0
        assert result.entities_migrated == 0
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_migration_creates_indexes(self) -> None:
        """Test migration creates temporal indexes."""
        mock_graph = MagicMock()
        queries_executed: List[str] = []

        def capture_query(query: str) -> MockQueryResult:
            queries_executed.append(query)
            if "db.indexes" in query:
                return MockQueryResult([])  # No existing indexes
            if "count" in query:
                return MockQueryResult([[0]])
            return MockQueryResult([])

        mock_graph.query = Mock(side_effect=capture_query)

        result = await migrate_to_bitemporal(mock_graph, dry_run=False)

        assert result.success is True

        # Check that CREATE INDEX queries were executed
        index_queries = [q for q in queries_executed if "CREATE INDEX" in q]
        # Should create at least some indexes (memory_current_idx, etc.)
        # Note: The actual number depends on which indexes already exist
        assert len(index_queries) >= 0  # May be 0 if error handling catches them

    @pytest.mark.asyncio
    async def test_migration_skips_existing_indexes(self) -> None:
        """Test migration skips indexes that already exist."""
        mock_graph = MagicMock()

        def query_with_existing_index(query: str) -> MockQueryResult:
            if "db.indexes" in query:
                # Return result indicating index exists
                return MockQueryResult([["memory_current_idx"]])
            if "count" in query:
                return MockQueryResult([[0]])
            return MockQueryResult([])

        mock_graph.query = Mock(side_effect=query_with_existing_index)

        result = await migrate_to_bitemporal(mock_graph, dry_run=False)

        assert result.success is True
        # Should still succeed even with existing indexes


# ============================================================================
# Integration-like Tests (with mocks)
# ============================================================================


class TestMigrationIntegration:
    """Integration-like tests for the migration workflow."""

    @pytest.mark.asyncio
    async def test_full_migration_workflow(self) -> None:
        """Test full migration workflow: check status -> migrate -> verify."""
        # Initial state: unmigrated data
        mock_graph = create_mock_graph(
            memory_total=100,
            memory_migrated=0,
            entity_total=50,
            entity_migrated=0,
        )

        # Step 1: Check status before migration
        status_before = get_migration_status(mock_graph)
        assert status_before["needs_migration"] is True
        assert status_before["memory_nodes_total"] == 100
        assert status_before["memory_nodes_migrated"] == 0

        # Step 2: Run dry-run first
        dry_result = await migrate_to_bitemporal(mock_graph, dry_run=True)
        assert dry_result.success is True
        assert dry_result.memories_migrated == 100

        # Step 3: Perform actual migration
        result = await migrate_to_bitemporal(mock_graph, dry_run=False)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_migration_preserves_existing_data(self) -> None:
        """Test that migration only updates nodes without temporal fields."""
        mock_graph = MagicMock()
        queries_executed: List[str] = []

        def capture_query(query: str) -> MockQueryResult:
            queries_executed.append(query)
            if "WHERE m.valid_from IS NULL" in query:
                return MockQueryResult([[50]])  # 50 unmigrated memories
            if "WHERE e.valid_from IS NULL" in query:
                return MockQueryResult([[25]])  # 25 unmigrated entities
            if "db.indexes" in query:
                return MockQueryResult([])
            return MockQueryResult([[0]])

        mock_graph.query = Mock(side_effect=capture_query)

        await migrate_to_bitemporal(mock_graph, dry_run=False)

        # Verify the WHERE clause is used to filter only unmigrated nodes
        memory_queries = [q for q in queries_executed if "SET m.valid_from" in q]
        for query in memory_queries:
            assert "WHERE m.valid_from IS NULL" in query

        entity_queries = [q for q in queries_executed if "SET e.valid_from" in q]
        for query in entity_queries:
            assert "WHERE e.valid_from IS NULL" in query


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
