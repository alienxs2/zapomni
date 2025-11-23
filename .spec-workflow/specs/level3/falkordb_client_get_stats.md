# FalkorDBClient.get_stats() - Function Specification

**Level:** 3 (Function)
**Component:** FalkorDBClient
**Module:** zapomni_db
**Author:** Goncharenko Anton aka alienxs2
**Status:** Draft
**Version:** 1.0
**Created:** 2025-11-23

---

## Function Signature

```python
async def get_stats(
    self
) -> Dict[str, Any]:
    """
    Retrieve comprehensive statistics about the knowledge graph.

    Executes multiple Cypher queries to gather metrics about nodes, relationships,
    memory usage, and database health. Provides a complete snapshot of the graph
    state for monitoring, debugging, and user-facing statistics.

    Returns aggregated statistics including:
    - Node counts by type (Memory, Chunk, Entity, etc.)
    - Relationship counts by type (HAS_CHUNK, MENTIONS, etc.)
    - Storage metrics (total memories, chunks, entities)
    - Index statistics (vector index size, performance)
    - Database health indicators

    Args:
        None: This method takes no parameters

    Returns:
        Dict[str, Any]: Comprehensive statistics dictionary with structure:
        {
            "nodes": {
                "total": int,  # Total node count across all types
                "memory": int,  # Number of Memory nodes
                "chunk": int,  # Number of Chunk nodes
                "entity": int,  # Number of Entity nodes
                "document": int  # Number of Document nodes
            },
            "relationships": {
                "total": int,  # Total relationship count
                "has_chunk": int,  # Memory -> Chunk relationships
                "mentions": int,  # Chunk -> Entity relationships
                "related_to": int  # Entity -> Entity relationships
            },
            "storage": {
                "total_memories": int,  # Same as nodes.memory
                "total_chunks": int,  # Same as nodes.chunk
                "total_entities": int,  # Same as nodes.entity
                "avg_chunks_per_memory": float  # Average chunking rate
            },
            "indexes": {
                "vector_index_size": int,  # Number of vectors indexed
                "vector_index_name": str  # Index name
            },
            "health": {
                "connected": bool,  # Database connection status
                "graph_name": str,  # Current graph name
                "query_latency_ms": float  # Average query time
            }
        }

    Raises:
        DatabaseError: If database queries fail
        ConnectionError: If database connection is lost

    Example:
        >>> client = FalkorDBClient(host="localhost", port=6379)
        >>> stats = await client.get_stats()
        >>> print(stats["nodes"]["total"])
        1523
        >>> print(stats["storage"]["avg_chunks_per_memory"])
        3.2
        >>> print(stats["health"]["connected"])
        True

    Thread Safety:
        Thread-safe. Multiple coroutines can call concurrently (read-only).

    Performance:
        - Latency: < 50ms for typical graphs (< 100k nodes)
        - Latency: < 200ms for large graphs (100k-1M nodes)
        - Query count: 5-7 Cypher queries executed
    """
```

---

## Purpose & Context

### What It Does

`get_stats()` provides **comprehensive visibility** into the knowledge graph by:

1. **Counting Nodes** by type (Memory, Chunk, Entity, Document)
2. **Counting Relationships** by type (HAS_CHUNK, MENTIONS, etc.)
3. **Calculating Metrics** (averages, ratios, health indicators)
4. **Checking Indexes** (vector index size and status)
5. **Verifying Health** (connection status, latency)

Returns a **structured statistics dictionary** for dashboards, monitoring, and debugging.

### Why It Exists

**Monitoring Requirements:**
- Users need visibility into system state
- Operators need health/performance metrics
- Debugging requires detailed graph statistics

**MCP Tool Integration:**
- `GetStatsTool` exposes this via MCP protocol
- Claude can query statistics during conversations

### When To Use

**Called By:**
- `GetStatsTool.execute()` - MCP tool for statistics
- `MemoryProcessor.get_stats()` - Core stats wrapper
- Admin/monitoring dashboards
- Health check endpoints

**Use When:**
- Displaying system status to users
- Monitoring graph growth over time
- Debugging data issues
- Verifying successful ingestion

### When NOT To Use

**Don't use for:**
- Real-time per-request metrics → use PerformanceMonitor
- Frequent polling (< 1s intervals) → too expensive
- Critical path operations → stats are read-heavy

---

## Parameters (Detailed)

### No Parameters

This method takes **no parameters**. It operates on the current database connection and graph name configured during initialization.

**Why No Parameters:**
- Statistics are for the entire graph (no filtering)
- Graph name is instance property (`self.graph_name`)
- Connection is instance property (`self.graph`)

**Future Enhancement:**
Could add optional parameters like:
- `include_indexes: bool = True` - Skip index stats for speed
- `include_health: bool = True` - Skip health checks
- `node_type: str = None` - Filter to specific node type

---

## Return Value

**Type:** `Dict[str, Any]`

**Purpose:**
Comprehensive statistics dictionary with nested structure for different metric categories.

### Complete Structure

```python
{
    # Node statistics
    "nodes": {
        "total": 1523,  # Sum of all node types
        "memory": 247,  # Count of Memory nodes
        "chunk": 912,  # Count of Chunk nodes
        "entity": 364,  # Count of Entity nodes
        "document": 0   # Count of Document nodes (Phase 2)
    },

    # Relationship statistics
    "relationships": {
        "total": 1756,  # Sum of all relationship types
        "has_chunk": 912,  # Memory -> Chunk
        "mentions": 728,  # Chunk -> Entity
        "related_to": 116  # Entity -> Entity (Phase 2)
    },

    # Storage metrics
    "storage": {
        "total_memories": 247,  # Same as nodes.memory
        "total_chunks": 912,  # Same as nodes.chunk
        "total_entities": 364,  # Same as nodes.entity
        "avg_chunks_per_memory": 3.69  # 912 / 247
    },

    # Index statistics
    "indexes": {
        "vector_index_size": 912,  # Number of vectors in HNSW index
        "vector_index_name": "chunk_embedding_idx"  # Index name
    },

    # Health indicators
    "health": {
        "connected": True,  # Database connection OK
        "graph_name": "zapomni_memory",  # Active graph
        "query_latency_ms": 12.5  # Average latency of stat queries
    }
}
```

### Field Descriptions

**nodes.***:
- `total`: Count of all nodes in graph
- `memory`: Number of Memory nodes (top-level memories)
- `chunk`: Number of Chunk nodes (text segments)
- `entity`: Number of Entity nodes (extracted entities)
- `document`: Number of Document nodes (Phase 2 feature)

**relationships.***:
- `total`: Count of all relationships
- `has_chunk`: Memory-to-Chunk edges
- `mentions`: Chunk-to-Entity edges (entity references)
- `related_to`: Entity-to-Entity edges (Phase 2)

**storage.***:
- `total_memories`: Convenience alias for nodes.memory
- `total_chunks`: Convenience alias for nodes.chunk
- `total_entities`: Convenience alias for nodes.entity
- `avg_chunks_per_memory`: Average chunking rate (chunks/memory)

**indexes.***:
- `vector_index_size`: Number of vectors indexed (should equal chunk count)
- `vector_index_name`: Name of the HNSW vector index

**health.***:
- `connected`: Boolean indicating DB connection status
- `graph_name`: Name of active graph
- `query_latency_ms`: Average query execution time for stats

### Example Responses

**Small Graph (< 100 memories):**
```python
{
    "nodes": {"total": 423, "memory": 87, "chunk": 312, "entity": 24, "document": 0},
    "relationships": {"total": 336, "has_chunk": 312, "mentions": 24, "related_to": 0},
    "storage": {
        "total_memories": 87,
        "total_chunks": 312,
        "total_entities": 24,
        "avg_chunks_per_memory": 3.59
    },
    "indexes": {"vector_index_size": 312, "vector_index_name": "chunk_embedding_idx"},
    "health": {"connected": True, "graph_name": "zapomni_memory", "query_latency_ms": 8.3}
}
```

**Empty Graph (Fresh Installation):**
```python
{
    "nodes": {"total": 0, "memory": 0, "chunk": 0, "entity": 0, "document": 0},
    "relationships": {"total": 0, "has_chunk": 0, "mentions": 0, "related_to": 0},
    "storage": {
        "total_memories": 0,
        "total_chunks": 0,
        "total_entities": 0,
        "avg_chunks_per_memory": 0.0
    },
    "indexes": {"vector_index_size": 0, "vector_index_name": "chunk_embedding_idx"},
    "health": {"connected": True, "graph_name": "zapomni_memory", "query_latency_ms": 5.1}
}
```

**Large Graph (1000+ memories):**
```python
{
    "nodes": {"total": 15234, "memory": 2456, "chunk": 10123, "entity": 2655, "document": 0},
    "relationships": {"total": 12778, "has_chunk": 10123, "mentions": 2655, "related_to": 0},
    "storage": {
        "total_memories": 2456,
        "total_chunks": 10123,
        "total_entities": 2655,
        "avg_chunks_per_memory": 4.12
    },
    "indexes": {"vector_index_size": 10123, "vector_index_name": "chunk_embedding_idx"},
    "health": {"connected": True, "graph_name": "zapomni_memory", "query_latency_ms": 45.7}
}
```

---

## Exceptions

### DatabaseError

**When Raised:**
- Cypher query execution fails
- Graph is corrupted or inaccessible
- Transaction errors

**Message Format:**
```python
f"Failed to retrieve graph statistics: {error_details}"
```

**Recovery:** Check database health, retry connection

### ConnectionError

**When Raised:**
- Database connection lost during stats collection
- FalkorDB service is down
- Network issues

**Message Format:**
```python
"Database connection lost while fetching statistics"
```

**Recovery:** Reconnect to database, verify service status

---

## Algorithm (Pseudocode)

```
ASYNC FUNCTION get_stats(self) -> dict:
    # Step 1: Initialize stats dictionary
    stats = {
        "nodes": {},
        "relationships": {},
        "storage": {},
        "indexes": {},
        "health": {}
    }

    # Step 2: Record start time for latency calculation
    start_time = time.time()

    TRY:
        # Step 3: Count nodes by type
        node_count_query = """
            MATCH (n)
            RETURN
                labels(n)[0] AS node_type,
                count(n) AS count
        """
        node_results = AWAIT self._execute_cypher(node_count_query, {})

        # Parse node counts
        stats["nodes"]["total"] = 0
        stats["nodes"]["memory"] = 0
        stats["nodes"]["chunk"] = 0
        stats["nodes"]["entity"] = 0
        stats["nodes"]["document"] = 0

        FOR row IN node_results:
            node_type = row["node_type"].lower()
            count = row["count"]
            stats["nodes"][node_type] = count
            stats["nodes"]["total"] += count

        # Step 4: Count relationships by type
        rel_count_query = """
            MATCH ()-[r]->()
            RETURN
                type(r) AS rel_type,
                count(r) AS count
        """
        rel_results = AWAIT self._execute_cypher(rel_count_query, {})

        # Parse relationship counts
        stats["relationships"]["total"] = 0
        stats["relationships"]["has_chunk"] = 0
        stats["relationships"]["mentions"] = 0
        stats["relationships"]["related_to"] = 0

        FOR row IN rel_results:
            rel_type = row["rel_type"].lower()
            count = row["count"]
            stats["relationships"][rel_type] = count
            stats["relationships"]["total"] += count

        # Step 5: Calculate storage metrics
        stats["storage"]["total_memories"] = stats["nodes"]["memory"]
        stats["storage"]["total_chunks"] = stats["nodes"]["chunk"]
        stats["storage"]["total_entities"] = stats["nodes"]["entity"]

        # Calculate average chunks per memory
        IF stats["nodes"]["memory"] > 0:
            stats["storage"]["avg_chunks_per_memory"] = (
                stats["nodes"]["chunk"] / stats["nodes"]["memory"]
            )
        ELSE:
            stats["storage"]["avg_chunks_per_memory"] = 0.0

        # Step 6: Get vector index stats
        index_query = """
            CALL db.indexes()
            YIELD name, type, entityType, properties
            WHERE name = 'chunk_embedding_idx'
            RETURN name, type
        """
        index_results = AWAIT self._execute_cypher(index_query, {})

        IF len(index_results) > 0:
            stats["indexes"]["vector_index_name"] = index_results[0]["name"]
            stats["indexes"]["vector_index_size"] = stats["nodes"]["chunk"]
        ELSE:
            stats["indexes"]["vector_index_name"] = "chunk_embedding_idx"
            stats["indexes"]["vector_index_size"] = 0

        # Step 7: Health metrics
        stats["health"]["connected"] = True
        stats["health"]["graph_name"] = self.graph_name

        # Calculate average query latency
        end_time = time.time()
        total_latency_ms = (end_time - start_time) * 1000
        num_queries = 3  # node count, rel count, index query
        stats["health"]["query_latency_ms"] = round(total_latency_ms / num_queries, 2)

        # Step 8: Log success
        self._logger.info(
            "stats_retrieved",
            total_nodes=stats["nodes"]["total"],
            total_relationships=stats["relationships"]["total"]
        )

        RETURN stats

    CATCH DatabaseError as db_error:
        self._logger.error(
            "stats_database_error",
            error=str(db_error),
            exc_info=True
        )
        RAISE DatabaseError(f"Failed to retrieve graph statistics: {db_error}")

    CATCH Exception as unexpected:
        self._logger.error(
            "stats_unexpected_error",
            error=str(unexpected),
            exc_info=True
        )
        RAISE DatabaseError(f"Unexpected error retrieving statistics: {unexpected}")

END FUNCTION
```

---

## Preconditions

✅ **Client Initialized:**
- `FalkorDBClient.__init__()` called
- Database connection established
- Graph schema initialized

✅ **Database Available:**
- FalkorDB service running
- Graph accessible

---

## Postconditions

### On Success

✅ **Stats Returned:**
- Complete statistics dictionary
- All fields populated
- Valid data types

✅ **No State Changes:**
- Read-only operation
- Database unchanged

### On Error

❌ **Exception Raised:**
- DatabaseError or ConnectionError
- Error logged to stderr

---

## Edge Cases & Handling

### Edge Case 1: Empty Graph (No Data)

**Scenario:** Brand new graph with zero nodes

**Input:** (no parameters)

**Processing:**
1. Count nodes: all queries return 0
2. Calculate averages: 0 / 0 → 0.0 (handled)
3. Return stats with all zeros

**Expected:**
```python
stats = await client.get_stats()
assert stats["nodes"]["total"] == 0
assert stats["storage"]["avg_chunks_per_memory"] == 0.0
```

**Test:**
```python
async def test_get_stats_empty_graph():
    client = FalkorDBClient()
    await client.clear_all()  # Empty the graph

    stats = await client.get_stats()

    assert stats["nodes"]["total"] == 0
    assert stats["relationships"]["total"] == 0
    assert stats["storage"]["avg_chunks_per_memory"] == 0.0
```

---

### Edge Case 2: Memories Without Chunks (Data Inconsistency)

**Scenario:** Memory nodes exist but HAS_CHUNK relationships missing

**Processing:**
1. Count memories: 10
2. Count chunks: 0
3. Average: 0 / 10 = 0.0

**Expected:**
```python
stats["storage"]["avg_chunks_per_memory"] == 0.0
# Indicates data quality issue
```

**Test:**
```python
async def test_get_stats_memories_without_chunks():
    client = FalkorDBClient()

    # Create memories without chunks (invalid state)
    for i in range(5):
        await client.graph_query("CREATE (:Memory {id: $id})", {"id": f"mem-{i}"})

    stats = await client.get_stats()

    assert stats["nodes"]["memory"] == 5
    assert stats["nodes"]["chunk"] == 0
    assert stats["storage"]["avg_chunks_per_memory"] == 0.0
```

---

### Edge Case 3: Large Graph (100k+ Nodes)

**Scenario:** Graph with 100,000+ nodes

**Processing:**
1. Queries take longer (50-200ms each)
2. Aggregation still O(N) but acceptable
3. Return accurate stats (may be slow)

**Expected:** Stats returned, latency higher

**Test:**
```python
async def test_get_stats_large_graph():
    # Note: This test would be slow, use in performance suite only
    client = FalkorDBClient()

    # Simulate large graph (or use pre-populated test DB)
    stats = await client.get_stats()

    assert stats["nodes"]["total"] > 100000
    assert stats["health"]["query_latency_ms"] < 500  # Still reasonable
```

---

### Edge Case 4: Database Connection Lost Mid-Query

**Scenario:** Connection drops during stats collection

**Processing:**
1. First query succeeds (node count)
2. Connection lost
3. Second query fails
4. Raise ConnectionError

**Expected:**
```python
# Raises: ConnectionError("Database connection lost while fetching statistics")
```

**Test:**
```python
async def test_get_stats_connection_lost(mocker):
    client = FalkorDBClient()

    # Mock to fail on second query
    call_count = 0
    async def mock_execute(query, params):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise ConnectionError("Connection lost")
        return []

    mocker.patch.object(client, '_execute_cypher', side_effect=mock_execute)

    with pytest.raises(ConnectionError, match="connection lost"):
        await client.get_stats()
```

---

### Edge Case 5: Vector Index Missing (Schema Issue)

**Scenario:** Vector index not created properly

**Processing:**
1. Index query returns empty result
2. Set index_size to 0
3. Set index_name to default

**Expected:**
```python
stats["indexes"]["vector_index_size"] == 0
stats["indexes"]["vector_index_name"] == "chunk_embedding_idx"
```

**Test:**
```python
async def test_get_stats_missing_index(mocker):
    client = FalkorDBClient()

    # Mock index query to return empty
    async def mock_execute(query, params):
        if "db.indexes()" in query:
            return []
        # Other queries return normal data
        return [{"node_type": "Memory", "count": 10}]

    mocker.patch.object(client, '_execute_cypher', side_effect=mock_execute)

    stats = await client.get_stats()

    assert stats["indexes"]["vector_index_size"] == 0
```

---

## Test Scenarios (Complete List)

### Happy Path Tests

**1. test_get_stats_success_normal_graph**
- Graph with memories, chunks, entities
- Expected: All stats populated correctly

**2. test_get_stats_calculates_averages**
- 100 memories, 350 chunks
- Expected: avg_chunks_per_memory = 3.5

**3. test_get_stats_all_node_types**
- Memories, chunks, entities present
- Expected: Each type counted

**4. test_get_stats_health_indicators**
- Expected: connected=True, latency reasonable

---

### Edge Case Tests

**5. test_get_stats_empty_graph**
- Edge case 1

**6. test_get_stats_memories_without_chunks**
- Edge case 2

**7. test_get_stats_large_graph**
- Edge case 3 (performance suite)

**8. test_get_stats_connection_lost**
- Edge case 4

**9. test_get_stats_missing_index**
- Edge case 5

---

### Error Tests

**10. test_get_stats_database_error**
- Query execution fails
- Expected: DatabaseError raised

**11. test_get_stats_logs_error**
- Error logged to stderr

---

### Integration Tests

**12. test_get_stats_after_add_memory**
- Add memory, check stats updated

**13. test_get_stats_after_delete_memory**
- Delete memory, check stats decreased

---

## Performance Requirements

### Latency Targets

- **Small graph (< 1k nodes):** < 20ms
- **Medium graph (1k-100k nodes):** < 50ms
- **Large graph (100k-1M nodes):** < 200ms

### Query Count

- **Total queries:** 3-5 Cypher queries
- **Aggregation:** O(N) for node/relationship counts

---

## Security Considerations

✅ **Read-Only:** No mutations, safe to expose
✅ **No Sensitive Data:** Stats are aggregate counts
✅ **Rate Limiting:** Should be rate-limited in production

---

## Related Functions

### Calls

**1. `self._execute_cypher(query, params)`**
- Purpose: Execute Cypher queries

**2. `self._logger.info/error()`**
- Purpose: Log stats retrieval

### Called By

**1. `GetStatsTool.execute()`**
- Purpose: MCP tool for stats

**2. `MemoryProcessor.get_stats()`**
- Purpose: Core stats wrapper

---

## Implementation Notes

### Dependencies

- FalkorDB Python client
- Cypher query language

### Known Limitations

**1. No Caching:**
- Stats recalculated every call
- Could cache for N seconds

**2. No Filtering:**
- All stats or nothing
- Could add node_type filter

---

## References

### Component Spec
- [FalkorDBClient Component](../level2/falkordb_client_component.md)

### Related Functions
- `FalkorDBClient.add_memory()` (Level 3)
- `FalkorDBClient.vector_search()` (Level 3)

---

## Document Status

**Version:** 1.0
**Created:** 2025-11-23
**Author:** Goncharenko Anton aka alienxs2
**License:** MIT
**Status:** Draft

**Estimated Implementation:** 1-2 hours
**Lines of Code:** ~80 lines
**Test Coverage Target:** 95%+
**Test File:** `tests/unit/db/test_falkordb_client_get_stats.py`
