"""
Unit tests for bi-temporal models (v0.8.0 - Issue #27).

Tests for:
- MemoryVersion model
- TemporalQuery model
- VersionInfo dataclass
- ChangeRecord dataclass
- TimelineEntry dataclass
- Entity temporal fields
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from zapomni_db.models import (
    ChangeRecord,
    Entity,
    MemoryVersion,
    TemporalQuery,
    TimelineEntry,
    VersionInfo,
)


class TestMemoryVersion:
    """Tests for MemoryVersion model."""

    def test_create_basic_memory_version(self) -> None:
        """Test creating a basic MemoryVersion."""
        now = datetime.utcnow().isoformat() + "Z"
        memory = MemoryVersion(
            id="uuid-123",
            text="def hello(): pass",
            created_at=now,
            valid_from=now,
        )

        assert memory.id == "uuid-123"
        assert memory.text == "def hello(): pass"
        assert memory.created_at == now
        assert memory.valid_from == now
        assert memory.valid_to is None
        assert memory.transaction_to is None
        assert memory.version == 1
        assert memory.previous_version_id is None
        assert memory.is_current is True
        assert memory.stale is False

    def test_create_full_memory_version(self) -> None:
        """Test creating a MemoryVersion with all fields."""
        memory = MemoryVersion(
            id="uuid-456",
            text="class Foo: pass",
            tags=["python", "class"],
            source="code_indexer",
            metadata='{"language": "python"}',
            workspace_id="my-workspace",
            file_path="/src/foo.py",
            qualified_name="foo.Foo",
            created_at="2025-12-01T10:00:00Z",
            transaction_to="2025-12-02T10:00:00Z",
            valid_from="2025-11-30T15:30:00Z",
            valid_to="2025-12-01T08:00:00Z",
            version=3,
            previous_version_id="uuid-455",
            is_current=False,
            stale=False,
            last_seen_at="2025-12-01T12:00:00Z",
        )

        assert memory.id == "uuid-456"
        assert memory.tags == ["python", "class"]
        assert memory.source == "code_indexer"
        assert memory.file_path == "/src/foo.py"
        assert memory.qualified_name == "foo.Foo"
        assert memory.transaction_to == "2025-12-02T10:00:00Z"
        assert memory.valid_to == "2025-12-01T08:00:00Z"
        assert memory.version == 3
        assert memory.previous_version_id == "uuid-455"
        assert memory.is_current is False

    def test_memory_version_requires_id(self) -> None:
        """Test that id is required."""
        with pytest.raises(ValidationError) as exc_info:
            MemoryVersion(
                text="test",
                created_at="2025-12-01T10:00:00Z",
                valid_from="2025-12-01T10:00:00Z",
            )  # type: ignore
        assert "id" in str(exc_info.value)

    def test_memory_version_requires_text(self) -> None:
        """Test that text is required."""
        with pytest.raises(ValidationError) as exc_info:
            MemoryVersion(
                id="uuid-123",
                created_at="2025-12-01T10:00:00Z",
                valid_from="2025-12-01T10:00:00Z",
            )  # type: ignore
        assert "text" in str(exc_info.value)

    def test_memory_version_requires_created_at(self) -> None:
        """Test that created_at is required."""
        with pytest.raises(ValidationError) as exc_info:
            MemoryVersion(
                id="uuid-123",
                text="test",
                valid_from="2025-12-01T10:00:00Z",
            )  # type: ignore
        assert "created_at" in str(exc_info.value)

    def test_memory_version_requires_valid_from(self) -> None:
        """Test that valid_from is required."""
        with pytest.raises(ValidationError) as exc_info:
            MemoryVersion(
                id="uuid-123",
                text="test",
                created_at="2025-12-01T10:00:00Z",
            )  # type: ignore
        assert "valid_from" in str(exc_info.value)

    def test_memory_version_defaults(self) -> None:
        """Test default values for MemoryVersion."""
        memory = MemoryVersion(
            id="uuid-123",
            text="test",
            created_at="2025-12-01T10:00:00Z",
            valid_from="2025-12-01T10:00:00Z",
        )

        assert memory.tags == []
        assert memory.source == "user"
        assert memory.metadata is None
        assert memory.workspace_id == "default"
        assert memory.file_path is None
        assert memory.qualified_name is None
        assert memory.transaction_to is None
        assert memory.valid_to is None
        assert memory.version == 1
        assert memory.previous_version_id is None
        assert memory.is_current is True
        assert memory.stale is False
        assert memory.last_seen_at is None

    def test_memory_version_min_version(self) -> None:
        """Test that version must be >= 1."""
        with pytest.raises(ValidationError) as exc_info:
            MemoryVersion(
                id="uuid-123",
                text="test",
                created_at="2025-12-01T10:00:00Z",
                valid_from="2025-12-01T10:00:00Z",
                version=0,
            )
        assert "version" in str(exc_info.value)

    def test_memory_version_serialization(self) -> None:
        """Test MemoryVersion can be serialized to dict."""
        memory = MemoryVersion(
            id="uuid-123",
            text="test",
            created_at="2025-12-01T10:00:00Z",
            valid_from="2025-12-01T10:00:00Z",
        )

        data = memory.model_dump()

        assert isinstance(data, dict)
        assert data["id"] == "uuid-123"
        assert data["text"] == "test"
        assert data["is_current"] is True
        assert data["version"] == 1


class TestTemporalQuery:
    """Tests for TemporalQuery model."""

    def test_create_default_query(self) -> None:
        """Test creating a default TemporalQuery."""
        query = TemporalQuery()

        assert query.mode == "current"
        assert query.time_type == "valid"
        assert query.as_of_valid is None
        assert query.as_of_transaction is None
        assert query.limit == 50

    def test_create_point_in_time_query(self) -> None:
        """Test creating a point-in-time query."""
        query = TemporalQuery(
            mode="point_in_time",
            time_type="valid",
            as_of_valid="2025-11-15T00:00:00Z",
        )

        assert query.mode == "point_in_time"
        assert query.time_type == "valid"
        assert query.as_of_valid == "2025-11-15T00:00:00Z"

    def test_create_transaction_time_query(self) -> None:
        """Test creating a transaction time query."""
        query = TemporalQuery(
            mode="point_in_time",
            time_type="transaction",
            as_of_transaction="2025-11-20T00:00:00Z",
        )

        assert query.mode == "point_in_time"
        assert query.time_type == "transaction"
        assert query.as_of_transaction == "2025-11-20T00:00:00Z"

    def test_create_history_query(self) -> None:
        """Test creating a history query."""
        query = TemporalQuery(mode="history", limit=100)

        assert query.mode == "history"
        assert query.limit == 100

    def test_query_limit_bounds(self) -> None:
        """Test that limit has proper bounds."""
        # Min limit
        with pytest.raises(ValidationError):
            TemporalQuery(limit=0)

        # Max limit
        with pytest.raises(ValidationError):
            TemporalQuery(limit=1001)

        # Valid limits
        query_min = TemporalQuery(limit=1)
        assert query_min.limit == 1

        query_max = TemporalQuery(limit=1000)
        assert query_max.limit == 1000

    def test_query_invalid_mode(self) -> None:
        """Test that invalid mode raises error."""
        with pytest.raises(ValidationError):
            TemporalQuery(mode="invalid")  # type: ignore

    def test_query_invalid_time_type(self) -> None:
        """Test that invalid time_type raises error."""
        with pytest.raises(ValidationError):
            TemporalQuery(time_type="invalid")  # type: ignore

    def test_query_both_time_types(self) -> None:
        """Test query with both time dimensions."""
        query = TemporalQuery(
            mode="point_in_time",
            time_type="both",
            as_of_valid="2025-11-15T00:00:00Z",
            as_of_transaction="2025-11-20T00:00:00Z",
        )

        assert query.time_type == "both"
        assert query.as_of_valid is not None
        assert query.as_of_transaction is not None


class TestVersionInfo:
    """Tests for VersionInfo dataclass."""

    def test_create_version_info(self) -> None:
        """Test creating a VersionInfo."""
        info = VersionInfo(
            version=2,
            memory_id="uuid-123",
            valid_from="2025-11-30T10:00:00Z",
            valid_to="2025-12-01T10:00:00Z",
            created_at="2025-12-01T10:00:00Z",
            transaction_to=None,
            is_current=True,
            change_type="modified",
        )

        assert info.version == 2
        assert info.memory_id == "uuid-123"
        assert info.valid_from == "2025-11-30T10:00:00Z"
        assert info.valid_to == "2025-12-01T10:00:00Z"
        assert info.is_current is True
        assert info.change_type == "modified"

    def test_version_info_defaults(self) -> None:
        """Test VersionInfo default values."""
        info = VersionInfo(
            version=1,
            memory_id="uuid-123",
            valid_from="2025-11-30T10:00:00Z",
            valid_to=None,
            created_at="2025-11-30T10:00:00Z",
            transaction_to=None,
            is_current=True,
        )

        assert info.change_type == "modified"  # default


class TestChangeRecord:
    """Tests for ChangeRecord dataclass."""

    def test_create_change_record(self) -> None:
        """Test creating a ChangeRecord."""
        record = ChangeRecord(
            memory_id="uuid-123",
            file_path="/src/foo.py",
            change_type="created",
            timestamp="2025-12-01T10:00:00Z",
            version=1,
        )

        assert record.memory_id == "uuid-123"
        assert record.file_path == "/src/foo.py"
        assert record.change_type == "created"
        assert record.version == 1
        assert record.previous_version_id is None
        assert record.qualified_name is None

    def test_change_record_with_previous(self) -> None:
        """Test ChangeRecord with previous version."""
        record = ChangeRecord(
            memory_id="uuid-124",
            file_path="/src/foo.py",
            change_type="modified",
            timestamp="2025-12-02T10:00:00Z",
            version=2,
            previous_version_id="uuid-123",
            qualified_name="foo.bar",
        )

        assert record.change_type == "modified"
        assert record.version == 2
        assert record.previous_version_id == "uuid-123"
        assert record.qualified_name == "foo.bar"


class TestTimelineEntry:
    """Tests for TimelineEntry dataclass."""

    def test_create_timeline_entry(self) -> None:
        """Test creating a TimelineEntry."""
        entry = TimelineEntry(
            version=3,
            valid_from="2025-12-01T10:00:00Z",
            valid_to=None,
            change_type="modified",
            summary="Added error handling",
            memory_id="uuid-123",
            file_path="/src/foo.py",
        )

        assert entry.version == 3
        assert entry.valid_from == "2025-12-01T10:00:00Z"
        assert entry.valid_to is None
        assert entry.change_type == "modified"
        assert entry.summary == "Added error handling"
        assert entry.memory_id == "uuid-123"
        assert entry.file_path == "/src/foo.py"

    def test_timeline_entry_without_file_path(self) -> None:
        """Test TimelineEntry without file_path."""
        entry = TimelineEntry(
            version=1,
            valid_from="2025-12-01T10:00:00Z",
            valid_to=None,
            change_type="created",
            summary="Initial version",
            memory_id="uuid-123",
        )

        assert entry.file_path is None


class TestEntityTemporalFields:
    """Tests for Entity temporal fields."""

    def test_entity_temporal_fields(self) -> None:
        """Test Entity has temporal fields."""
        entity = Entity(
            name="UserService",
            type="CLASS",
            valid_from="2025-11-30T10:00:00Z",
            valid_to=None,
            is_current=True,
        )

        assert entity.valid_from == "2025-11-30T10:00:00Z"
        assert entity.valid_to is None
        assert entity.is_current is True

    def test_entity_temporal_defaults(self) -> None:
        """Test Entity temporal field defaults."""
        entity = Entity(name="foo", type="FUNCTION")

        assert entity.valid_from is None  # Optional
        assert entity.valid_to is None
        assert entity.is_current is True  # Default

    def test_entity_non_current(self) -> None:
        """Test Entity with is_current=False."""
        entity = Entity(
            name="OldService",
            type="CLASS",
            valid_from="2025-11-01T10:00:00Z",
            valid_to="2025-11-30T10:00:00Z",
            is_current=False,
        )

        assert entity.is_current is False
        assert entity.valid_to is not None
