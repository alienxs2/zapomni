"""
Migration 001: Add bi-temporal fields to Memory and Entity nodes.

This migration adds support for bi-temporal queries by adding:
- valid_from, valid_to: Valid time dimension (when true in reality)
- transaction_to: Transaction time end (created_at is already transaction start)
- version, previous_version_id: Version chain for history
- is_current: Fast filter for current records

This migration is IDEMPOTENT - safe to run multiple times.

Author: Goncharenko Anton aka alienxs2
License: MIT
Version: 0.8.0
Issue: #27 (Bi-temporal model)
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import structlog

if TYPE_CHECKING:
    from falkordb import Graph

logger = structlog.get_logger(__name__)


@dataclass
class MigrationResult:
    """Result of migration execution."""

    success: bool = False
    memories_migrated: int = 0
    entities_migrated: int = 0
    indexes_created: int = 0
    errors: List[str] = None  # type: ignore

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []


async def migrate_to_bitemporal(
    graph: "Graph",
    dry_run: bool = False,
) -> MigrationResult:
    """
    Add bi-temporal fields to existing Memory and Entity nodes.

    This migration:
    1. Adds temporal fields to Memory nodes (valid_from, valid_to, etc.)
    2. Adds temporal fields to Entity nodes
    3. Creates new indexes for temporal queries
    4. Sets default values for existing records

    Args:
        graph: FalkorDB Graph instance
        dry_run: If True, only report what would be done without making changes

    Returns:
        MigrationResult with counts and status

    Note:
        This migration is idempotent - running it multiple times is safe.
        Existing temporal fields are not overwritten.
    """
    result = MigrationResult()
    log = logger.bind(migration="001_bitemporal", dry_run=dry_run)

    log.info("Starting bi-temporal migration")

    try:
        # Step 1: Migrate Memory nodes
        memories_count = await _migrate_memory_nodes(graph, dry_run, log)
        result.memories_migrated = memories_count

        # Step 2: Migrate Entity nodes
        entities_count = await _migrate_entity_nodes(graph, dry_run, log)
        result.entities_migrated = entities_count

        # Step 3: Create new indexes (if not dry run)
        if not dry_run:
            indexes_count = await _create_temporal_indexes(graph, log)
            result.indexes_created = indexes_count

        result.success = True
        log.info(
            "Bi-temporal migration completed",
            memories=result.memories_migrated,
            entities=result.entities_migrated,
            indexes=result.indexes_created,
        )

    except Exception as e:
        result.errors.append(str(e))
        log.error("Migration failed", error=str(e))

    return result


async def _migrate_memory_nodes(
    graph: "Graph",
    dry_run: bool,
    log: Any,
) -> int:
    """
    Add bi-temporal fields to Memory nodes.

    Fields added:
    - valid_from: Set to created_at (or current time if missing)
    - valid_to: Set to null (still valid)
    - transaction_to: Set to null (still current in DB)
    - version: Set to 1
    - previous_version_id: Set to null (first version)
    - is_current: Set to true

    Returns:
        Number of Memory nodes migrated
    """
    log.debug("Migrating Memory nodes")

    # Count nodes that need migration (don't have valid_from)
    count_query = """
        MATCH (m:Memory)
        WHERE m.valid_from IS NULL
        RETURN count(m) AS count
    """

    try:
        result = graph.query(count_query)
        count = result.result_set[0][0] if result.result_set else 0
    except Exception as e:
        log.warning("Could not count Memory nodes", error=str(e))
        count = 0

    log.info("Memory nodes to migrate", count=count)

    if dry_run or count == 0:
        return count

    # Perform migration
    migrate_query = """
        MATCH (m:Memory)
        WHERE m.valid_from IS NULL
        SET m.valid_from = COALESCE(m.created_at, datetime()),
            m.valid_to = null,
            m.transaction_to = null,
            m.version = 1,
            m.previous_version_id = null,
            m.is_current = true
        RETURN count(m) AS migrated
    """

    try:
        result = graph.query(migrate_query)
        migrated = result.result_set[0][0] if result.result_set else 0
        log.info("Memory nodes migrated", count=migrated)
        return migrated
    except Exception as e:
        log.error("Failed to migrate Memory nodes", error=str(e))
        raise


async def _migrate_entity_nodes(
    graph: "Graph",
    dry_run: bool,
    log: Any,
) -> int:
    """
    Add bi-temporal fields to Entity nodes.

    Fields added:
    - valid_from: Set to created_at or updated_at (or current time)
    - valid_to: Set to null (still valid)
    - is_current: Set to true

    Returns:
        Number of Entity nodes migrated
    """
    log.debug("Migrating Entity nodes")

    # Count nodes that need migration
    count_query = """
        MATCH (e:Entity)
        WHERE e.valid_from IS NULL
        RETURN count(e) AS count
    """

    try:
        result = graph.query(count_query)
        count = result.result_set[0][0] if result.result_set else 0
    except Exception as e:
        log.warning("Could not count Entity nodes", error=str(e))
        count = 0

    log.info("Entity nodes to migrate", count=count)

    if dry_run or count == 0:
        return count

    # Perform migration
    migrate_query = """
        MATCH (e:Entity)
        WHERE e.valid_from IS NULL
        SET e.valid_from = COALESCE(e.created_at, e.updated_at, datetime()),
            e.valid_to = null,
            e.is_current = true
        RETURN count(e) AS migrated
    """

    try:
        result = graph.query(migrate_query)
        migrated = result.result_set[0][0] if result.result_set else 0
        log.info("Entity nodes migrated", count=migrated)
        return migrated
    except Exception as e:
        log.error("Failed to migrate Entity nodes", error=str(e))
        raise


async def _create_temporal_indexes(
    graph: "Graph",
    log: Any,
) -> int:
    """
    Create indexes for bi-temporal queries.

    Indexes created:
    - memory_current_idx: Fast current state queries
    - memory_valid_from_idx: Valid time range queries
    - memory_version_idx: Version chain traversal
    - entity_current_idx: Entity current state

    Returns:
        Number of indexes created
    """
    log.debug("Creating temporal indexes")

    indexes = [
        ("memory_current_idx", "Memory", "is_current"),
        ("memory_valid_from_idx", "Memory", "valid_from"),
        ("memory_version_idx", "Memory", "previous_version_id"),
        ("entity_current_idx", "Entity", "is_current"),
    ]

    created = 0

    for index_name, node_label, property_name in indexes:
        try:
            # Check if index exists
            check_query = "CALL db.indexes()"
            result = graph.query(check_query)
            exists = any(
                index_name in str(row)
                for row in (result.result_set or [])
            )

            if exists:
                log.debug("Index already exists", index=index_name)
                continue

            # Create index
            create_query = f"""
                CREATE INDEX FOR (n:{node_label})
                ON (n.{property_name})
            """
            graph.query(create_query)
            created += 1
            log.info("Index created", index=index_name)

        except Exception as e:
            # "already indexed" is expected for idempotent operation
            if "already indexed" in str(e).lower():
                log.debug("Index already exists (detected via error)", index=index_name)
            else:
                log.warning("Failed to create index", index=index_name, error=str(e))

    log.info("Temporal indexes created", count=created)
    return created


def get_migration_status(graph: "Graph") -> Dict[str, Any]:
    """
    Check migration status by counting nodes with/without temporal fields.

    Returns:
        Dictionary with migration status:
        - needs_migration: bool
        - memory_nodes_total: int
        - memory_nodes_migrated: int
        - entity_nodes_total: int
        - entity_nodes_migrated: int
    """
    status: Dict[str, Any] = {
        "needs_migration": False,
        "memory_nodes_total": 0,
        "memory_nodes_migrated": 0,
        "entity_nodes_total": 0,
        "entity_nodes_migrated": 0,
    }

    try:
        # Count Memory nodes
        result = graph.query("MATCH (m:Memory) RETURN count(m) AS total")
        status["memory_nodes_total"] = result.result_set[0][0] if result.result_set else 0

        result = graph.query(
            "MATCH (m:Memory) WHERE m.valid_from IS NOT NULL RETURN count(m) AS migrated"
        )
        status["memory_nodes_migrated"] = result.result_set[0][0] if result.result_set else 0

        # Count Entity nodes
        result = graph.query("MATCH (e:Entity) RETURN count(e) AS total")
        status["entity_nodes_total"] = result.result_set[0][0] if result.result_set else 0

        result = graph.query(
            "MATCH (e:Entity) WHERE e.valid_from IS NOT NULL RETURN count(e) AS migrated"
        )
        status["entity_nodes_migrated"] = result.result_set[0][0] if result.result_set else 0

        # Check if migration is needed
        status["needs_migration"] = (
            status["memory_nodes_total"] > status["memory_nodes_migrated"]
            or status["entity_nodes_total"] > status["entity_nodes_migrated"]
        )

    except Exception:
        pass

    return status
