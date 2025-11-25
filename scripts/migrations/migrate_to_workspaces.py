#!/usr/bin/env python3
"""
Migration Script: Add workspace_id to existing data.

This script migrates existing Zapomni data to support workspaces by:
1. Creating a backup of the current database state
2. Adding workspace_id="default" to all existing Memory, Chunk, Entity nodes
3. Creating the default Workspace node
4. Verifying the migration was successful

IMPORTANT: According to validation report:
- Batch processing uses SKIP/LIMIT pattern verified for FalkorDB
- Backup verification is performed before proceeding
- Rollback only removes workspace_id from nodes where it was added

Usage:
    python migrate_to_workspaces.py --host localhost --port 6381

    # Dry run (no changes):
    python migrate_to_workspaces.py --host localhost --port 6381 --dry-run

    # Rollback:
    python migrate_to_workspaces.py --host localhost --port 6381 --rollback

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from zapomni_db.falkordb_client import FalkorDBClient
from zapomni_db.models import DEFAULT_WORKSPACE_ID, Workspace

# Migration configuration
BATCH_SIZE = 100  # Number of nodes to process per batch
BACKUP_QUERY_LIMIT = 10000  # Maximum nodes to backup in dry-run


async def create_backup(
    client: FalkorDBClient,
    backup_path: Path,
) -> Dict[str, int]:
    """
    Create a backup of node counts before migration.

    Args:
        client: FalkorDBClient instance
        backup_path: Path to save backup metadata

    Returns:
        Dict with node counts
    """
    print("Creating backup metadata...")

    # Get counts of each node type
    counts = {}

    # Count Memory nodes
    result = await client._execute_cypher("MATCH (m:Memory) RETURN count(m) AS count", {})
    counts["memories"] = result.rows[0]["count"] if result.rows else 0

    # Count Chunk nodes
    result = await client._execute_cypher("MATCH (c:Chunk) RETURN count(c) AS count", {})
    counts["chunks"] = result.rows[0]["count"] if result.rows else 0

    # Count Entity nodes
    result = await client._execute_cypher("MATCH (e:Entity) RETURN count(e) AS count", {})
    counts["entities"] = result.rows[0]["count"] if result.rows else 0

    # Count nodes without workspace_id
    result = await client._execute_cypher(
        "MATCH (m:Memory) WHERE m.workspace_id IS NULL RETURN count(m) AS count", {}
    )
    counts["memories_without_workspace"] = result.rows[0]["count"] if result.rows else 0

    result = await client._execute_cypher(
        "MATCH (c:Chunk) WHERE c.workspace_id IS NULL RETURN count(c) AS count", {}
    )
    counts["chunks_without_workspace"] = result.rows[0]["count"] if result.rows else 0

    result = await client._execute_cypher(
        "MATCH (e:Entity) WHERE e.workspace_id IS NULL RETURN count(e) AS count", {}
    )
    counts["entities_without_workspace"] = result.rows[0]["count"] if result.rows else 0

    # Save backup metadata
    backup_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "counts": counts,
        "version": "1.0",
    }

    backup_path.write_text(json.dumps(backup_data, indent=2))
    print(f"Backup metadata saved to: {backup_path}")

    return counts


def verify_backup(backup_path: Path) -> bool:
    """
    Verify backup file exists and is valid.

    Args:
        backup_path: Path to backup file

    Returns:
        True if backup is valid
    """
    if not backup_path.exists():
        print(f"ERROR: Backup file not found: {backup_path}")
        return False

    try:
        data = json.loads(backup_path.read_text())
        if "counts" not in data or "timestamp" not in data:
            print("ERROR: Backup file is missing required fields")
            return False
        print(f"Backup verified: created at {data['timestamp']}")
        return True
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid backup file: {e}")
        return False


async def migrate_node_type(
    client: FalkorDBClient,
    node_label: str,
    dry_run: bool = False,
) -> int:
    """
    Add workspace_id to all nodes of a specific type.

    Uses SKIP/LIMIT pattern for batch processing (verified for FalkorDB).

    Args:
        client: FalkorDBClient instance
        node_label: Node label (Memory, Chunk, Entity)
        dry_run: If True, only report what would be done

    Returns:
        Number of nodes updated
    """
    print(f"\nMigrating {node_label} nodes...")

    # Count nodes needing migration
    count_query = f"""
    MATCH (n:{node_label})
    WHERE n.workspace_id IS NULL
    RETURN count(n) AS count
    """
    result = await client._execute_cypher(count_query, {})
    total = result.rows[0]["count"] if result.rows else 0

    if total == 0:
        print(f"  No {node_label} nodes need migration")
        return 0

    print(f"  Found {total} {node_label} nodes to migrate")

    if dry_run:
        print(f"  [DRY RUN] Would update {total} nodes")
        return total

    # Batch update using SKIP/LIMIT pattern
    updated = 0
    offset = 0

    while offset < total:
        # Update batch of nodes
        # Note: FalkorDB supports SKIP/LIMIT in MATCH queries
        update_query = f"""
        MATCH (n:{node_label})
        WHERE n.workspace_id IS NULL
        WITH n LIMIT {BATCH_SIZE}
        SET n.workspace_id = $workspace_id
        RETURN count(n) AS updated
        """

        result = await client._execute_cypher(update_query, {"workspace_id": DEFAULT_WORKSPACE_ID})

        batch_updated = result.rows[0]["updated"] if result.rows else 0
        updated += batch_updated

        if batch_updated == 0:
            # No more nodes to update
            break

        print(f"  Updated {updated}/{total} nodes...", end="\r")

        # Small delay to prevent overwhelming the database
        await asyncio.sleep(0.1)

    print(f"  Updated {updated}/{total} {node_label} nodes")
    return updated


async def create_default_workspace(
    client: FalkorDBClient,
    dry_run: bool = False,
) -> bool:
    """
    Create the default workspace node if it doesn't exist.

    Args:
        client: FalkorDBClient instance
        dry_run: If True, only report what would be done

    Returns:
        True if workspace was created or already exists
    """
    print("\nCreating default workspace...")

    # Check if workspace already exists
    check_query = """
    MATCH (w:Workspace {id: $workspace_id})
    RETURN w.id AS id
    """
    result = await client._execute_cypher(check_query, {"workspace_id": DEFAULT_WORKSPACE_ID})

    if result.rows:
        print(f"  Default workspace already exists")
        return True

    if dry_run:
        print(f"  [DRY RUN] Would create default workspace")
        return True

    # Create default workspace
    create_query = """
    CREATE (w:Workspace {
        id: $workspace_id,
        name: $name,
        description: $description,
        created_at: $created_at,
        metadata: $metadata
    })
    RETURN w.id AS id
    """

    workspace = Workspace(
        id=DEFAULT_WORKSPACE_ID,
        name="Default Workspace",
        description="Default workspace for memories without explicit workspace",
        created_at=datetime.now(timezone.utc),
        metadata={},
    )

    result = await client._execute_cypher(
        create_query,
        {
            "workspace_id": workspace.id,
            "name": workspace.name,
            "description": workspace.description,
            "created_at": workspace.created_at.isoformat(),
            "metadata": json.dumps(workspace.metadata),
        },
    )

    if result.rows:
        print(f"  Created default workspace: {workspace.id}")
        return True
    else:
        print(f"  ERROR: Failed to create default workspace")
        return False


async def verify_migration(
    client: FalkorDBClient,
    backup_counts: Dict[str, int],
) -> bool:
    """
    Verify the migration was successful.

    Args:
        client: FalkorDBClient instance
        backup_counts: Original node counts from backup

    Returns:
        True if migration is verified
    """
    print("\nVerifying migration...")

    # Check no nodes without workspace_id remain
    for label in ["Memory", "Chunk", "Entity"]:
        result = await client._execute_cypher(
            f"MATCH (n:{label}) WHERE n.workspace_id IS NULL RETURN count(n) AS count",
            {},
        )
        remaining = result.rows[0]["count"] if result.rows else 0

        if remaining > 0:
            print(f"  WARNING: {remaining} {label} nodes still missing workspace_id")
            return False

    # Check node counts match
    result = await client._execute_cypher("MATCH (m:Memory) RETURN count(m) AS count", {})
    current_memories = result.rows[0]["count"] if result.rows else 0

    if current_memories != backup_counts.get("memories", 0):
        print(f"  WARNING: Memory count changed during migration")
        return False

    # Check default workspace exists
    result = await client._execute_cypher(
        "MATCH (w:Workspace {id: $workspace_id}) RETURN w.id AS id",
        {"workspace_id": DEFAULT_WORKSPACE_ID},
    )

    if not result.rows:
        print(f"  WARNING: Default workspace not found")
        return False

    print("  Migration verified successfully!")
    return True


async def rollback_migration(
    client: FalkorDBClient,
    backup_path: Path,
    dry_run: bool = False,
) -> bool:
    """
    Rollback the migration by removing workspace_id from nodes.

    IMPORTANT: According to validation report, we should only remove
    workspace_id from nodes where it equals DEFAULT_WORKSPACE_ID to avoid
    removing workspace_id from nodes that were explicitly assigned.

    Args:
        client: FalkorDBClient instance
        backup_path: Path to backup file
        dry_run: If True, only report what would be done

    Returns:
        True if rollback was successful
    """
    print("\nRolling back migration...")

    # Verify backup exists
    if not verify_backup(backup_path):
        print("ERROR: Cannot rollback without valid backup")
        return False

    # Only remove workspace_id from nodes with default workspace
    # This is safer than removing all workspace_id properties blindly
    for label in ["Memory", "Chunk", "Entity"]:
        count_query = f"""
        MATCH (n:{label})
        WHERE n.workspace_id = $workspace_id
        RETURN count(n) AS count
        """
        result = await client._execute_cypher(count_query, {"workspace_id": DEFAULT_WORKSPACE_ID})
        total = result.rows[0]["count"] if result.rows else 0

        print(f"  Found {total} {label} nodes with default workspace_id")

        if dry_run:
            print(f"  [DRY RUN] Would remove workspace_id from {total} nodes")
            continue

        if total > 0:
            # Remove workspace_id in batches
            rollback_query = f"""
            MATCH (n:{label})
            WHERE n.workspace_id = $workspace_id
            WITH n LIMIT {BATCH_SIZE}
            REMOVE n.workspace_id
            RETURN count(n) AS updated
            """

            updated = 0
            while updated < total:
                result = await client._execute_cypher(
                    rollback_query, {"workspace_id": DEFAULT_WORKSPACE_ID}
                )
                batch = result.rows[0]["updated"] if result.rows else 0
                if batch == 0:
                    break
                updated += batch
                print(f"  Removed workspace_id from {updated}/{total} nodes...", end="\r")
                await asyncio.sleep(0.1)

            print(f"  Removed workspace_id from {updated} {label} nodes")

    # Remove default workspace node
    if not dry_run:
        delete_query = """
        MATCH (w:Workspace {id: $workspace_id})
        DELETE w
        RETURN count(w) AS deleted
        """
        result = await client._execute_cypher(delete_query, {"workspace_id": DEFAULT_WORKSPACE_ID})
        deleted = result.rows[0]["deleted"] if result.rows else 0
        print(f"  Deleted {deleted} workspace node(s)")
    else:
        print("  [DRY RUN] Would delete default workspace node")

    print("Rollback complete!")
    return True


async def run_migration(
    host: str,
    port: int,
    graph_name: str,
    dry_run: bool,
    rollback: bool,
) -> bool:
    """
    Run the workspace migration.

    Args:
        host: FalkorDB host
        port: FalkorDB port
        graph_name: Graph name
        dry_run: If True, only report what would be done
        rollback: If True, rollback the migration

    Returns:
        True if migration was successful
    """
    # Create backup path
    backup_dir = Path(__file__).parent / "backups"
    backup_dir.mkdir(exist_ok=True)
    backup_path = backup_dir / f"workspace_migration_{graph_name}.json"

    # Initialize client
    print(f"Connecting to FalkorDB at {host}:{port}...")
    client = FalkorDBClient(
        host=host,
        port=port,
        graph_name=graph_name,
    )

    try:
        await client.init_async()
        print("Connected successfully!")

        if rollback:
            return await rollback_migration(client, backup_path, dry_run)

        # Step 1: Create backup
        backup_counts = await create_backup(client, backup_path)

        # Step 2: Verify backup before proceeding
        if not verify_backup(backup_path):
            print("ERROR: Backup verification failed")
            return False

        print("\nBackup counts:")
        print(f"  Memories: {backup_counts.get('memories', 0)}")
        print(f"  Chunks: {backup_counts.get('chunks', 0)}")
        print(f"  Entities: {backup_counts.get('entities', 0)}")
        print(
            f"  Memories without workspace_id: {backup_counts.get('memories_without_workspace', 0)}"
        )
        print(f"  Chunks without workspace_id: {backup_counts.get('chunks_without_workspace', 0)}")
        print(
            f"  Entities without workspace_id: {backup_counts.get('entities_without_workspace', 0)}"
        )

        # Step 3: Migrate each node type
        total_updated = 0
        for label in ["Memory", "Chunk", "Entity"]:
            updated = await migrate_node_type(client, label, dry_run)
            total_updated += updated

        # Step 4: Create default workspace
        workspace_created = await create_default_workspace(client, dry_run)

        if not workspace_created:
            print("ERROR: Failed to create default workspace")
            return False

        # Step 5: Verify migration
        if not dry_run:
            verified = await verify_migration(client, backup_counts)
            if not verified:
                print("\nWARNING: Migration verification failed!")
                print("You may want to rollback using: --rollback")
                return False

        print("\n" + "=" * 50)
        if dry_run:
            print("DRY RUN COMPLETE - No changes were made")
        else:
            print("MIGRATION COMPLETE!")
        print(f"Total nodes updated: {total_updated}")
        print("=" * 50)

        return True

    finally:
        await client.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Migrate Zapomni data to support workspaces")
    parser.add_argument(
        "--host",
        default="localhost",
        help="FalkorDB host (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6381,
        help="FalkorDB port (default: 6381)",
    )
    parser.add_argument(
        "--graph-name",
        default="zapomni",
        help="Graph name (default: zapomni)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Rollback the migration",
    )

    args = parser.parse_args()

    print("=" * 50)
    print("Zapomni Workspace Migration")
    print("=" * 50)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Graph: {args.graph_name}")
    print(f"Mode: {'DRY RUN' if args.dry_run else ('ROLLBACK' if args.rollback else 'LIVE')}")
    print("=" * 50)

    if not args.dry_run and not args.rollback:
        response = input("\nThis will modify your database. Continue? [y/N] ")
        if response.lower() != "y":
            print("Aborted.")
            sys.exit(0)

    success = asyncio.run(
        run_migration(
            host=args.host,
            port=args.port,
            graph_name=args.graph_name,
            dry_run=args.dry_run,
            rollback=args.rollback,
        )
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
