"""
Unit tests for CypherQueryBuilder bi-temporal methods.

Tests for:
- _build_temporal_filter_clause()
- build_point_in_time_query()
- build_history_query()
- build_changes_query()
- build_close_version_query()
"""

import pytest

from zapomni_db.cypher_query_builder import CypherQueryBuilder
from zapomni_db.exceptions import ValidationError


@pytest.fixture
def query_builder() -> CypherQueryBuilder:
    """Create a CypherQueryBuilder instance for testing."""
    return CypherQueryBuilder()


# ============================================================================
# _build_temporal_filter_clause TESTS
# ============================================================================


class TestBuildTemporalFilterClause:
    """Tests for _build_temporal_filter_clause()."""

    def test_current_mode_returns_is_current_filter(self, query_builder: CypherQueryBuilder) -> None:
        """Test that time_type='current' returns is_current = true filter."""
        clause, params = query_builder._build_temporal_filter_clause(time_type="current")

        assert "is_current = true" in clause
        assert clause.startswith("AND")
        assert params == {}

    def test_valid_mode_returns_valid_time_filter(self, query_builder: CypherQueryBuilder) -> None:
        """Test that time_type='valid' returns valid time range filter."""
        as_of = "2025-11-15T00:00:00Z"
        clause, params = query_builder._build_temporal_filter_clause(
            time_type="valid",
            as_of_valid=as_of,
        )

        assert "valid_from <= $as_of_valid" in clause
        assert "valid_to IS NULL OR" in clause
        assert "valid_to > $as_of_valid" in clause
        assert params["as_of_valid"] == as_of

    def test_valid_mode_requires_as_of_valid(self, query_builder: CypherQueryBuilder) -> None:
        """Test that time_type='valid' raises error without as_of_valid."""
        with pytest.raises(ValidationError, match="as_of_valid is required"):
            query_builder._build_temporal_filter_clause(time_type="valid")

    def test_transaction_mode_returns_transaction_time_filter(
        self, query_builder: CypherQueryBuilder
    ) -> None:
        """Test that time_type='transaction' returns transaction time range filter."""
        as_of = "2025-11-15T00:00:00Z"
        clause, params = query_builder._build_temporal_filter_clause(
            time_type="transaction",
            as_of_transaction=as_of,
        )

        assert "created_at <= $as_of_transaction" in clause
        assert "transaction_to IS NULL OR" in clause
        assert "transaction_to > $as_of_transaction" in clause
        assert params["as_of_transaction"] == as_of

    def test_transaction_mode_requires_as_of_transaction(
        self, query_builder: CypherQueryBuilder
    ) -> None:
        """Test that time_type='transaction' raises error without as_of_transaction."""
        with pytest.raises(ValidationError, match="as_of_transaction is required"):
            query_builder._build_temporal_filter_clause(time_type="transaction")

    def test_both_mode_returns_combined_filter(self, query_builder: CypherQueryBuilder) -> None:
        """Test that time_type='both' returns combined valid and transaction filters."""
        as_of_valid = "2025-11-15T00:00:00Z"
        as_of_transaction = "2025-11-16T00:00:00Z"
        clause, params = query_builder._build_temporal_filter_clause(
            time_type="both",
            as_of_valid=as_of_valid,
            as_of_transaction=as_of_transaction,
        )

        # Check valid time part
        assert "valid_from <= $as_of_valid" in clause
        assert "valid_to IS NULL OR" in clause
        # Check transaction time part
        assert "created_at <= $as_of_transaction" in clause
        assert "transaction_to IS NULL OR" in clause
        # Check params
        assert params["as_of_valid"] == as_of_valid
        assert params["as_of_transaction"] == as_of_transaction

    def test_both_mode_requires_both_timestamps(self, query_builder: CypherQueryBuilder) -> None:
        """Test that time_type='both' raises error without both timestamps."""
        # Missing as_of_valid
        with pytest.raises(ValidationError, match="as_of_valid is required"):
            query_builder._build_temporal_filter_clause(
                time_type="both",
                as_of_transaction="2025-11-15T00:00:00Z",
            )

        # Missing as_of_transaction
        with pytest.raises(ValidationError, match="as_of_transaction is required"):
            query_builder._build_temporal_filter_clause(
                time_type="both",
                as_of_valid="2025-11-15T00:00:00Z",
            )

    def test_custom_node_alias(self, query_builder: CypherQueryBuilder) -> None:
        """Test that custom node alias is used in the clause."""
        clause, _ = query_builder._build_temporal_filter_clause(
            time_type="current",
            node_alias="memory",
        )

        assert "memory.is_current = true" in clause

    def test_invalid_time_type_raises_error(self, query_builder: CypherQueryBuilder) -> None:
        """Test that invalid time_type raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid time_type"):
            query_builder._build_temporal_filter_clause(time_type="invalid")


# ============================================================================
# build_point_in_time_query TESTS
# ============================================================================


class TestBuildPointInTimeQuery:
    """Tests for build_point_in_time_query()."""

    def test_valid_time_query_structure(self, query_builder: CypherQueryBuilder) -> None:
        """Test valid time query contains correct structure."""
        cypher, params = query_builder.build_point_in_time_query(
            workspace_id="default",
            file_path="/project/src/main.py",
            as_of="2025-11-15T00:00:00Z",
            time_type="valid",
        )

        # Check query structure
        assert "MATCH (m:Memory)" in cypher
        assert "m.workspace_id = $workspace_id" in cypher
        assert "m.file_path = $file_path" in cypher
        assert "valid_from <= $as_of_valid" in cypher
        assert "ORDER BY" in cypher
        assert "LIMIT 1" in cypher

        # Check params
        assert params["workspace_id"] == "default"
        assert params["file_path"] == "/project/src/main.py"
        assert params["as_of_valid"] == "2025-11-15T00:00:00Z"

    def test_transaction_time_query_structure(self, query_builder: CypherQueryBuilder) -> None:
        """Test transaction time query contains correct structure."""
        cypher, params = query_builder.build_point_in_time_query(
            workspace_id="default",
            file_path="/project/src/main.py",
            as_of="2025-11-15T00:00:00Z",
            time_type="transaction",
        )

        # Check query uses transaction time fields
        assert "created_at <= $as_of_transaction" in cypher
        assert "transaction_to IS NULL OR" in cypher
        assert params["as_of_transaction"] == "2025-11-15T00:00:00Z"

    def test_both_time_query_structure(self, query_builder: CypherQueryBuilder) -> None:
        """Test bi-temporal query contains both time filters."""
        cypher, params = query_builder.build_point_in_time_query(
            workspace_id="default",
            file_path="/project/src/main.py",
            as_of="2025-11-15T00:00:00Z",
            time_type="both",
        )

        # Check both temporal filters
        assert "valid_from <= $as_of_valid" in cypher
        assert "created_at <= $as_of_transaction" in cypher

        # For 'both' mode, as_of is used for both dimensions
        assert params["as_of_valid"] == "2025-11-15T00:00:00Z"
        assert params["as_of_transaction"] == "2025-11-15T00:00:00Z"

    def test_empty_workspace_id_raises_error(self, query_builder: CypherQueryBuilder) -> None:
        """Test that empty workspace_id raises ValidationError."""
        with pytest.raises(ValidationError, match="workspace_id cannot be empty"):
            query_builder.build_point_in_time_query(
                workspace_id="",
                file_path="/project/src/main.py",
                as_of="2025-11-15T00:00:00Z",
            )

    def test_whitespace_workspace_id_raises_error(self, query_builder: CypherQueryBuilder) -> None:
        """Test that whitespace-only workspace_id raises ValidationError."""
        with pytest.raises(ValidationError, match="workspace_id cannot be empty"):
            query_builder.build_point_in_time_query(
                workspace_id="   ",
                file_path="/project/src/main.py",
                as_of="2025-11-15T00:00:00Z",
            )

    def test_empty_file_path_raises_error(self, query_builder: CypherQueryBuilder) -> None:
        """Test that empty file_path raises ValidationError."""
        with pytest.raises(ValidationError, match="file_path cannot be empty"):
            query_builder.build_point_in_time_query(
                workspace_id="default",
                file_path="",
                as_of="2025-11-15T00:00:00Z",
            )

    def test_empty_as_of_raises_error(self, query_builder: CypherQueryBuilder) -> None:
        """Test that empty as_of raises ValidationError."""
        with pytest.raises(ValidationError, match="as_of cannot be empty"):
            query_builder.build_point_in_time_query(
                workspace_id="default",
                file_path="/project/src/main.py",
                as_of="",
            )

    def test_invalid_time_type_raises_error(self, query_builder: CypherQueryBuilder) -> None:
        """Test that invalid time_type raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid time_type"):
            query_builder.build_point_in_time_query(
                workspace_id="default",
                file_path="/project/src/main.py",
                as_of="2025-11-15T00:00:00Z",
                time_type="invalid",  # type: ignore
            )

    def test_parameters_correct(self, query_builder: CypherQueryBuilder) -> None:
        """Test that all parameters are correctly populated."""
        cypher, params = query_builder.build_point_in_time_query(
            workspace_id="my-workspace",
            file_path="/my/file.py",
            as_of="2025-12-01T12:00:00Z",
            time_type="valid",
        )

        assert params["workspace_id"] == "my-workspace"
        assert params["file_path"] == "/my/file.py"
        assert params["as_of_valid"] == "2025-12-01T12:00:00Z"

    def test_query_returns_expected_fields(self, query_builder: CypherQueryBuilder) -> None:
        """Test that query RETURN clause includes all expected fields."""
        cypher, _ = query_builder.build_point_in_time_query(
            workspace_id="default",
            file_path="/project/src/main.py",
            as_of="2025-11-15T00:00:00Z",
            time_type="valid",
        )

        # Check all expected return fields
        assert "memory_id" in cypher
        assert "text" in cypher
        assert "file_path" in cypher
        assert "version" in cypher
        assert "valid_from" in cypher
        assert "valid_to" in cypher
        assert "created_at" in cypher
        assert "transaction_to" in cypher
        assert "is_current" in cypher


# ============================================================================
# build_history_query TESTS
# ============================================================================


class TestBuildHistoryQuery:
    """Tests for build_history_query()."""

    def test_query_with_file_path(self, query_builder: CypherQueryBuilder) -> None:
        """Test history query with file_path filter."""
        cypher, params = query_builder.build_history_query(
            workspace_id="default",
            file_path="/project/src/main.py",
        )

        assert "MATCH (m:Memory)" in cypher
        assert "m.workspace_id = $workspace_id" in cypher
        assert "m.file_path = $file_path" in cypher
        assert params["file_path"] == "/project/src/main.py"
        assert params["workspace_id"] == "default"

    def test_query_with_entity_id(self, query_builder: CypherQueryBuilder) -> None:
        """Test history query with entity_id filter."""
        entity_id = "550e8400-e29b-41d4-a716-446655440000"
        cypher, params = query_builder.build_history_query(
            workspace_id="default",
            entity_id=entity_id,
        )

        assert "m.id = $entity_id" in cypher
        assert params["entity_id"] == entity_id

    def test_query_requires_file_path_or_entity_id(
        self, query_builder: CypherQueryBuilder
    ) -> None:
        """Test that query raises error without file_path or entity_id."""
        with pytest.raises(ValidationError, match="Either file_path or entity_id must be provided"):
            query_builder.build_history_query(workspace_id="default")

    def test_empty_workspace_id_raises_error(self, query_builder: CypherQueryBuilder) -> None:
        """Test that empty workspace_id raises ValidationError."""
        with pytest.raises(ValidationError, match="workspace_id cannot be empty"):
            query_builder.build_history_query(
                workspace_id="",
                file_path="/project/src/main.py",
            )

    def test_invalid_entity_id_raises_error(self, query_builder: CypherQueryBuilder) -> None:
        """Test that invalid entity_id UUID raises ValidationError."""
        with pytest.raises(ValidationError, match="UUID"):
            query_builder.build_history_query(
                workspace_id="default",
                entity_id="not-a-valid-uuid",
            )

    def test_limit_applied(self, query_builder: CypherQueryBuilder) -> None:
        """Test that limit is applied to query."""
        cypher, params = query_builder.build_history_query(
            workspace_id="default",
            file_path="/project/src/main.py",
            limit=25,
        )

        assert "LIMIT $limit" in cypher
        assert params["limit"] == 25

    def test_default_limit(self, query_builder: CypherQueryBuilder) -> None:
        """Test that default limit is 50."""
        _, params = query_builder.build_history_query(
            workspace_id="default",
            file_path="/project/src/main.py",
        )

        assert params["limit"] == 50

    def test_limit_out_of_range_raises_error(self, query_builder: CypherQueryBuilder) -> None:
        """Test that limit out of range raises ValidationError."""
        # Too small
        with pytest.raises(ValidationError, match="limit"):
            query_builder.build_history_query(
                workspace_id="default",
                file_path="/project/src/main.py",
                limit=0,
            )

        # Too large
        with pytest.raises(ValidationError, match="limit"):
            query_builder.build_history_query(
                workspace_id="default",
                file_path="/project/src/main.py",
                limit=1001,
            )

    def test_order_by_valid_from_desc(self, query_builder: CypherQueryBuilder) -> None:
        """Test that results are ordered by valid_from DESC."""
        cypher, _ = query_builder.build_history_query(
            workspace_id="default",
            file_path="/project/src/main.py",
        )

        assert "ORDER BY m.valid_from DESC" in cypher

    def test_query_returns_version_fields(self, query_builder: CypherQueryBuilder) -> None:
        """Test that query returns all version-related fields."""
        cypher, _ = query_builder.build_history_query(
            workspace_id="default",
            file_path="/project/src/main.py",
        )

        assert "memory_id" in cypher
        assert "version" in cypher
        assert "valid_from" in cypher
        assert "valid_to" in cypher
        assert "created_at" in cypher
        assert "transaction_to" in cypher
        assert "is_current" in cypher


# ============================================================================
# build_changes_query TESTS
# ============================================================================


class TestBuildChangesQuery:
    """Tests for build_changes_query()."""

    def test_basic_changes_query(self, query_builder: CypherQueryBuilder) -> None:
        """Test basic changes query structure."""
        cypher, params = query_builder.build_changes_query(
            workspace_id="default",
            since="2025-11-01T00:00:00Z",
        )

        assert "MATCH (m:Memory)" in cypher
        assert "m.workspace_id = $workspace_id" in cypher
        assert "m.created_at >= $since" in cypher
        assert params["workspace_id"] == "default"
        assert params["since"] == "2025-11-01T00:00:00Z"

    def test_with_until_parameter(self, query_builder: CypherQueryBuilder) -> None:
        """Test changes query with until parameter."""
        cypher, params = query_builder.build_changes_query(
            workspace_id="default",
            since="2025-11-01T00:00:00Z",
            until="2025-12-01T00:00:00Z",
        )

        assert "m.created_at <= $until" in cypher
        assert params["until"] == "2025-12-01T00:00:00Z"

    def test_filter_by_change_type_created(self, query_builder: CypherQueryBuilder) -> None:
        """Test filtering by change_type='created'."""
        cypher, _ = query_builder.build_changes_query(
            workspace_id="default",
            since="2025-11-01T00:00:00Z",
            change_type="created",
        )

        assert "m.version = 1" in cypher

    def test_filter_by_change_type_modified(self, query_builder: CypherQueryBuilder) -> None:
        """Test filtering by change_type='modified'."""
        cypher, _ = query_builder.build_changes_query(
            workspace_id="default",
            since="2025-11-01T00:00:00Z",
            change_type="modified",
        )

        assert "m.version > 1" in cypher

    def test_filter_by_change_type_deleted(self, query_builder: CypherQueryBuilder) -> None:
        """Test filtering by change_type='deleted'."""
        cypher, _ = query_builder.build_changes_query(
            workspace_id="default",
            since="2025-11-01T00:00:00Z",
            change_type="deleted",
        )

        assert "m.transaction_to IS NOT NULL" in cypher

    def test_filter_by_path_pattern(self, query_builder: CypherQueryBuilder) -> None:
        """Test filtering by path_pattern."""
        cypher, params = query_builder.build_changes_query(
            workspace_id="default",
            since="2025-11-01T00:00:00Z",
            path_pattern="/project/src/",
        )

        assert "m.file_path STARTS WITH $path_pattern" in cypher
        assert params["path_pattern"] == "/project/src/"

    def test_limit_applied(self, query_builder: CypherQueryBuilder) -> None:
        """Test that limit is applied to query."""
        cypher, params = query_builder.build_changes_query(
            workspace_id="default",
            since="2025-11-01T00:00:00Z",
            limit=50,
        )

        assert "LIMIT $limit" in cypher
        assert params["limit"] == 50

    def test_default_limit(self, query_builder: CypherQueryBuilder) -> None:
        """Test that default limit is 100."""
        _, params = query_builder.build_changes_query(
            workspace_id="default",
            since="2025-11-01T00:00:00Z",
        )

        assert params["limit"] == 100

    def test_empty_workspace_id_raises_error(self, query_builder: CypherQueryBuilder) -> None:
        """Test that empty workspace_id raises ValidationError."""
        with pytest.raises(ValidationError, match="workspace_id cannot be empty"):
            query_builder.build_changes_query(
                workspace_id="",
                since="2025-11-01T00:00:00Z",
            )

    def test_empty_since_raises_error(self, query_builder: CypherQueryBuilder) -> None:
        """Test that empty since raises ValidationError."""
        with pytest.raises(ValidationError, match="since cannot be empty"):
            query_builder.build_changes_query(
                workspace_id="default",
                since="",
            )

    def test_invalid_change_type_raises_error(self, query_builder: CypherQueryBuilder) -> None:
        """Test that invalid change_type raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid change_type"):
            query_builder.build_changes_query(
                workspace_id="default",
                since="2025-11-01T00:00:00Z",
                change_type="invalid",
            )

    def test_limit_out_of_range_raises_error(self, query_builder: CypherQueryBuilder) -> None:
        """Test that limit out of range raises ValidationError."""
        with pytest.raises(ValidationError, match="limit"):
            query_builder.build_changes_query(
                workspace_id="default",
                since="2025-11-01T00:00:00Z",
                limit=0,
            )

    def test_query_returns_change_type_field(self, query_builder: CypherQueryBuilder) -> None:
        """Test that query returns computed change_type field."""
        cypher, _ = query_builder.build_changes_query(
            workspace_id="default",
            since="2025-11-01T00:00:00Z",
        )

        # Check CASE statement for change_type
        assert "CASE" in cypher
        assert "change_type" in cypher

    def test_order_by_created_at_desc(self, query_builder: CypherQueryBuilder) -> None:
        """Test that results are ordered by created_at DESC."""
        cypher, _ = query_builder.build_changes_query(
            workspace_id="default",
            since="2025-11-01T00:00:00Z",
        )

        assert "ORDER BY m.created_at DESC" in cypher


# ============================================================================
# build_close_version_query TESTS
# ============================================================================


class TestBuildCloseVersionQuery:
    """Tests for build_close_version_query()."""

    def test_sets_all_fields(self, query_builder: CypherQueryBuilder) -> None:
        """Test that query sets valid_to, transaction_to, and is_current."""
        memory_id = "550e8400-e29b-41d4-a716-446655440000"
        valid_to = "2025-12-05T10:30:00Z"
        transaction_to = "2025-12-05T10:30:00Z"

        cypher, params = query_builder.build_close_version_query(
            memory_id=memory_id,
            valid_to=valid_to,
            transaction_to=transaction_to,
        )

        # Check SET clause
        assert "SET m.valid_to = $valid_to" in cypher
        assert "SET m.transaction_to = $transaction_to" in cypher or "transaction_to = $transaction_to" in cypher
        assert "is_current = false" in cypher

        # Check params
        assert params["memory_id"] == memory_id
        assert params["valid_to"] == valid_to
        assert params["transaction_to"] == transaction_to

    def test_matches_by_memory_id(self, query_builder: CypherQueryBuilder) -> None:
        """Test that query matches by memory_id."""
        memory_id = "550e8400-e29b-41d4-a716-446655440000"
        cypher, _ = query_builder.build_close_version_query(
            memory_id=memory_id,
            valid_to="2025-12-05T10:30:00Z",
            transaction_to="2025-12-05T10:30:00Z",
        )

        assert "MATCH (m:Memory {id: $memory_id})" in cypher

    def test_invalid_memory_id_raises_error(self, query_builder: CypherQueryBuilder) -> None:
        """Test that invalid memory_id UUID raises ValidationError."""
        with pytest.raises(ValidationError, match="UUID"):
            query_builder.build_close_version_query(
                memory_id="not-a-valid-uuid",
                valid_to="2025-12-05T10:30:00Z",
                transaction_to="2025-12-05T10:30:00Z",
            )

    def test_empty_memory_id_raises_error(self, query_builder: CypherQueryBuilder) -> None:
        """Test that empty memory_id raises ValidationError."""
        with pytest.raises(ValidationError):
            query_builder.build_close_version_query(
                memory_id="",
                valid_to="2025-12-05T10:30:00Z",
                transaction_to="2025-12-05T10:30:00Z",
            )

    def test_empty_valid_to_raises_error(self, query_builder: CypherQueryBuilder) -> None:
        """Test that empty valid_to raises ValidationError."""
        with pytest.raises(ValidationError, match="valid_to cannot be empty"):
            query_builder.build_close_version_query(
                memory_id="550e8400-e29b-41d4-a716-446655440000",
                valid_to="",
                transaction_to="2025-12-05T10:30:00Z",
            )

    def test_empty_transaction_to_raises_error(self, query_builder: CypherQueryBuilder) -> None:
        """Test that empty transaction_to raises ValidationError."""
        with pytest.raises(ValidationError, match="transaction_to cannot be empty"):
            query_builder.build_close_version_query(
                memory_id="550e8400-e29b-41d4-a716-446655440000",
                valid_to="2025-12-05T10:30:00Z",
                transaction_to="",
            )

    def test_parameters_correct(self, query_builder: CypherQueryBuilder) -> None:
        """Test that all parameters are correctly populated."""
        memory_id = "550e8400-e29b-41d4-a716-446655440000"
        valid_to = "2025-12-05T10:30:00Z"
        transaction_to = "2025-12-05T10:35:00Z"

        _, params = query_builder.build_close_version_query(
            memory_id=memory_id,
            valid_to=valid_to,
            transaction_to=transaction_to,
        )

        assert params["memory_id"] == memory_id
        assert params["valid_to"] == valid_to
        assert params["transaction_to"] == transaction_to

    def test_query_returns_updated_fields(self, query_builder: CypherQueryBuilder) -> None:
        """Test that query RETURN clause includes updated fields."""
        cypher, _ = query_builder.build_close_version_query(
            memory_id="550e8400-e29b-41d4-a716-446655440000",
            valid_to="2025-12-05T10:30:00Z",
            transaction_to="2025-12-05T10:30:00Z",
        )

        assert "RETURN" in cypher
        assert "memory_id" in cypher
        assert "is_current" in cypher
        assert "valid_to" in cypher
        assert "transaction_to" in cypher
