# Workspace/Project Isolation for Knowledge Graph Systems - Research Report

**Date:** 2025-11-25
**Project:** Zapomni Memory Server
**Objective:** Research multi-tenant patterns and workspace isolation strategies for FalkorDB-based knowledge graph system

---

## Executive Summary

This research investigates workspace isolation patterns for Zapomni, a knowledge graph memory server using FalkorDB (graph database) and vector embeddings. The system currently stores all data in a flat namespace without tenant separation. This report analyzes multi-tenant approaches, session state management patterns, and migration strategies to implement workspace isolation.

**Key Recommendation:** Implement a **hybrid property-based filtering approach** with `workspace_id` as a node property, combined with session-aware MCP server state management. This provides strong isolation, efficient queries, and minimal migration complexity.

---

## 1. Multi-Tenant Patterns in Graph Databases

### 1.1 Overview of Approaches

Based on industry research and FalkorDB/Neo4j best practices, there are three primary multi-tenancy patterns for graph databases:

#### **Pattern 1: Separate Graphs (Hard Isolation)**
- **Description:** Each workspace gets a dedicated graph instance
- **FalkorDB Support:** Native multi-graph support (10K+ graphs per instance)
- **Pros:**
  - Complete data isolation (zero commingling)
  - Independent backups and migrations
  - No query overhead from filtering
  - Tenant-specific schema customization
- **Cons:**
  - Higher memory overhead per workspace
  - No cross-workspace queries
  - Complex resource management for many workspaces
  - Increased operational complexity

**FalkorDB Example:**
```python
# Create separate graphs
db = FalkorDB(host="localhost", port=6381)
workspace_a = db.select_graph("zapomni_workspace_a")
workspace_b = db.select_graph("zapomni_workspace_b")
```

#### **Pattern 2: Property-Based Filtering (Soft Isolation)**
- **Description:** Single graph with `workspace_id` property on all nodes/edges
- **Implementation:** Add `workspace_id` to Memory, Chunk, Entity nodes
- **Pros:**
  - Efficient resource utilization (single graph)
  - Cross-workspace analytics possible
  - Simpler operational model
  - Lower memory footprint
- **Cons:**
  - Query overhead from filtering every operation
  - Risk of data leakage if filters forgotten
  - Shared schema constraints across workspaces
  - Index size grows with all workspaces

**Cypher Query Example:**
```cypher
// Vector search with workspace filter
MATCH (m:Memory {workspace_id: $workspace_id})-[:HAS_CHUNK]->(c:Chunk)
WHERE c.workspace_id = $workspace_id
CALL db.idx.vector.queryNodes('chunk_embedding_idx', 10, $embedding)
YIELD node AS c, score
WHERE c.workspace_id = $workspace_id
RETURN c.text, score
```

#### **Pattern 3: Hybrid Approach (Recommended)**
- **Description:** Combine property filtering with logical graph partitioning
- **Implementation:**
  - Use `workspace_id` property for queries
  - Optionally create separate graphs for large/premium workspaces
  - Default workspace for quick start ("default")
- **Pros:**
  - Flexible isolation levels per use case
  - Efficient for most workspaces, isolated for critical ones
  - Gradual migration path (start simple, scale complexity)
- **Cons:**
  - Dual-mode complexity in code
  - Needs routing logic to select graph

**Recommendation for Zapomni:** Start with **Pattern 2 (Property-Based Filtering)** for simplicity, with architecture allowing migration to Pattern 3 if needed.

### 1.2 Performance Characteristics

Based on [Memgraph's multi-tenancy research](https://memgraph.com/blog/why-multi-tenancy-matters-in-graph-databases) and [AWS Neptune guidance](https://aws.amazon.com/blogs/database/build-multi-tenant-architectures-on-amazon-neptune/):

| Approach | Query Overhead | Memory Usage | Isolation Level | Scalability |
|----------|---------------|--------------|-----------------|-------------|
| Separate Graphs | None (0%) | High (100% per tenant) | Complete | Limited by instance |
| Property Filtering | 5-15% | Low (shared) | Logical | Excellent |
| Hybrid | Variable | Medium | Configurable | Good |

**Key Insight:** Property-based filtering adds 5-15% query overhead but provides 10x better memory efficiency. For Zapomni's use case (moderate workspace count, shared infrastructure), this trade-off is favorable.

### 1.3 FalkorDB-Specific Considerations

From [FalkorDB's multi-tenant blog](https://www.falkordb.com/blog/graph-database-multi-tenant-cloud-security/):

- **Native Multi-Graph:** FalkorDB supports 10,000+ graphs per Redis instance with zero overhead
- **Resource Sharing:** Shared compute resources across graphs with dedicated instances
- **Tenant Isolation:** Dedicated graph instances eliminate update conflicts
- **DevOps Simplicity:** No need for multiple Redis instances

**Implementation Note:** FalkorDB's `select_graph(name)` API makes multi-graph switching trivial:

```python
db = AsyncFalkorDB(connection_pool=pool)
graph = db.select_graph(f"zapomni_{workspace_id}")  # Automatic creation
```

---

## 2. Vector Search Filtering with HNSW Indexes

### 2.1 Challenge: Filtering HNSW Indexes

HNSW (Hierarchical Navigable Small World) indexes optimize for approximate nearest neighbor search but **complicate filtering**. From [Bits & Backprops analysis](https://yudhiesh.github.io/2025/05/09/the-achilles-heel-of-vector-search-filters/):

> "Unlike a traditional RDBMS, adding a filter to a vector search often slows it down, not speeds it up."

#### **Pre-Filtering vs. In-Filtering**

Based on [Oracle's HNSW optimizer plans](https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/optimizer-plans-hnsw-vector-indexes.html):

1. **Pre-Filtering:**
   - Filter rows BEFORE vector search
   - Best when filter is highly selective (<10% of data)
   - Can miss nearest neighbors outside filtered set

2. **In-Filtering (Recommended for Zapomni):**
   - Traverse HNSW graph, check `workspace_id` at each candidate
   - More accurate (considers full graph)
   - Better when many rows pass filter (>10% of data)

**FalkorDB Vector Search with Filtering:**
```cypher
// In-filtering approach - check workspace_id during traversal
CALL db.idx.vector.queryNodes('chunk_embedding_idx', $limit, $embedding)
YIELD node AS c, score
WHERE c.workspace_id = $workspace_id  // Filter during traversal
RETURN c.id, c.text, score
ORDER BY score DESC
```

### 2.2 Semantic Filtering Strategy

From [Emiliano Billi's semantic filtering article](https://emilianobilli.medium.com/semantic-filtering-in-hnsw-structuring-vector-indexes-by-domain-bd447695cfb7):

**Embed Domain Tags in Vectors:**
Instead of post-processing, attach `workspace_id` metadata to each vector and filter during graph traversal:

```python
# During embedding storage
chunk_node = {
    "id": chunk_id,
    "text": chunk_text,
    "embedding": vecf32(embedding),
    "workspace_id": workspace_id,  # Filter tag
    "memory_id": memory_id
}
```

**Benefits:**
- Maintains graph connectivity
- Filters at candidate evaluation (not post-search)
- Top-K results guaranteed within workspace

### 2.3 Index Partitioning (Advanced Option)

For future optimization, consider **separate HNSW indexes per workspace**:

```cypher
// Create workspace-specific vector index
CREATE VECTOR INDEX workspace_a_chunks FOR (c:Chunk)
ON (c.embedding)
WHERE c.workspace_id = 'workspace_a'
OPTIONS {dimension: 768, similarityFunction: 'cosine'}
```

**Trade-offs:**
- **Pros:** No filtering overhead, faster search
- **Cons:** Increased memory (N indexes), complex management

**Recommendation:** Start with single index + filtering. Consider partitioned indexes if workspace count > 100 or performance degrades.

---

## 3. Session State Management in MCP Servers

### 3.1 MCP Session Isolation Patterns

Based on [MCP GitHub issue #1087](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1087) and [CodeSignal's stateful server guide](https://codesignal.com/learn/courses/developing-and-integrating-an-mcp-server-in-typescript/lessons/stateful-mcp-server-sessions):

**Key Question:** Does MCP protocol guarantee session isolation, or is it implementation-specific?

**Answer:** Session isolation is **implementation-specific**. MCP provides `sessionId` but doesn't enforce state separation.

### 3.2 Recommended Pattern: Session Manager

From [MCPcat's configuration guide](https://mcpcat.io/guides/configuring-mcp-servers-multiple-simultaneous-connections/):

```python
class SessionManager:
    """Manages isolated workspace state per MCP session."""

    def __init__(self):
        self._sessions: Dict[str, WorkspaceSession] = {}

    def get_or_create_session(self, session_id: str) -> WorkspaceSession:
        """Get existing session or create new one with default workspace."""
        if session_id not in self._sessions:
            self._sessions[session_id] = WorkspaceSession(
                session_id=session_id,
                workspace_id="default",  # Default workspace
                created_at=datetime.now(timezone.utc)
            )
        return self._sessions[session_id]

    def set_workspace(self, session_id: str, workspace_id: str) -> None:
        """Switch workspace for session."""
        session = self.get_or_create_session(session_id)
        session.workspace_id = workspace_id
        session.updated_at = datetime.now(timezone.utc)

    def get_workspace(self, session_id: str) -> str:
        """Get current workspace for session."""
        return self.get_or_create_session(session_id).workspace_id
```

### 3.3 Hybrid Argument + Session State

User requirement: **Explicit `workspace_id` argument takes priority over session state.**

```python
async def add_memory(
    self,
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    workspace_id: Optional[str] = None,  # Explicit argument
    session_id: Optional[str] = None     # From MCP context
) -> str:
    """
    Add memory with workspace isolation.

    Priority:
    1. Explicit workspace_id argument (if provided)
    2. Session workspace state (if session exists)
    3. Default workspace ("default")
    """
    # Determine workspace
    if workspace_id is not None:
        target_workspace = workspace_id
    elif session_id is not None:
        target_workspace = self.session_manager.get_workspace(session_id)
    else:
        target_workspace = "default"

    # Add workspace_id to metadata
    final_metadata = metadata or {}
    final_metadata["workspace_id"] = target_workspace

    # Store with workspace
    return await self._store_memory(..., workspace_id=target_workspace)
```

### 3.4 Cross-Context Operations

User requirement: **Support reading from workspace A while writing to workspace B.**

```python
# Cross-workspace search
results_a = await processor.search_memory(
    query="Python frameworks",
    workspace_id="project_a"  # Explicit read workspace
)

# Write to different workspace
memory_id = await processor.add_memory(
    text="Flask is a Python microframework",
    workspace_id="project_b"  # Explicit write workspace
)
```

**Implementation Note:** Every database operation must accept `workspace_id` parameter and inject it into Cypher WHERE clauses.

### 3.5 Session Storage Options

From [Byteplus MCP session management](https://www.byteplus.com/en/topic/541419):

| Storage Type | Use Case | Pros | Cons |
|--------------|----------|------|------|
| **In-Memory** | Single server, moderate load | Simple, fast | Lost on restart |
| **Redis** | Multi-server, high availability | Persistent, shared | External dependency |
| **Database** | Audit trail, compliance | Durable, queryable | Slower access |

**Recommendation for Zapomni:** Start with **in-memory** SessionManager. Migrate to Redis if deploying multi-instance load-balanced setup.

---

## 4. Migration Strategies

### 4.1 Adding `workspace_id` to Existing Nodes

**Challenge:** Zapomni already has data without `workspace_id`. How to migrate without downtime?

#### **Strategy 1: Default Workspace Migration (Recommended)**

```cypher
// Add workspace_id property to all existing Memory nodes
MATCH (m:Memory)
WHERE m.workspace_id IS NULL
SET m.workspace_id = 'default'
RETURN count(m) AS migrated_memories

// Add workspace_id to all existing Chunk nodes
MATCH (c:Chunk)
WHERE c.workspace_id IS NULL
SET c.workspace_id = 'default'
RETURN count(c) AS migrated_chunks

// Add workspace_id to all existing Entity nodes
MATCH (e:Entity)
WHERE e.workspace_id IS NULL
SET e.workspace_id = 'default'
RETURN count(e) AS migrated_entities
```

**Execution Plan:**
1. Run migration script during maintenance window (low traffic)
2. Apply to Memory, Chunk, Entity nodes
3. Verify counts match expected totals
4. Update application code to require `workspace_id`

**Downtime:** None (SET operations are atomic per node)

**Rollback:** Remove property: `MATCH (n) REMOVE n.workspace_id`

#### **Strategy 2: Dual-Read Migration (Zero Downtime)**

For large datasets (>1M nodes), use phased migration:

**Phase 1: Write workspace_id, read with fallback**
```python
# Write: Always include workspace_id
metadata["workspace_id"] = workspace_id or "default"

# Read: Support both old and new data
query = """
MATCH (c:Chunk)
WHERE c.workspace_id = $workspace_id OR c.workspace_id IS NULL
CALL db.idx.vector.queryNodes('chunk_embedding_idx', $limit, $embedding)
YIELD node AS c, score
RETURN c.id, c.text, score
"""
```

**Phase 2: Background migration**
```python
# Migrate in batches
batch_size = 10000
while True:
    result = await db.graph_query(
        """
        MATCH (m:Memory)
        WHERE m.workspace_id IS NULL
        WITH m LIMIT $batch_size
        SET m.workspace_id = 'default'
        RETURN count(m) AS migrated
        """,
        {"batch_size": batch_size}
    )
    if result.rows[0]["migrated"] == 0:
        break  # All nodes migrated
```

**Phase 3: Enforce workspace_id**
Remove fallback logic, require `workspace_id` in all queries.

### 4.2 Schema Migration Script

```python
# scripts/migrate_to_workspaces.py

import asyncio
from zapomni_db import FalkorDBClient

async def migrate_to_workspaces():
    """Add workspace_id to all existing nodes."""

    db = FalkorDBClient(host="localhost", port=6381)
    await db.init_async()

    try:
        # Memory nodes
        result = await db.graph_query(
            "MATCH (m:Memory) WHERE m.workspace_id IS NULL "
            "SET m.workspace_id = 'default' RETURN count(m)"
        )
        print(f"Migrated {result.rows[0]['count(m)']} Memory nodes")

        # Chunk nodes
        result = await db.graph_query(
            "MATCH (c:Chunk) WHERE c.workspace_id IS NULL "
            "SET c.workspace_id = 'default' RETURN count(c)"
        )
        print(f"Migrated {result.rows[0]['count(c)']} Chunk nodes")

        # Entity nodes
        result = await db.graph_query(
            "MATCH (e:Entity) WHERE e.workspace_id IS NULL "
            "SET e.workspace_id = 'default' RETURN count(e)"
        )
        print(f"Migrated {result.rows[0]['count(e)']} Entity nodes")

    finally:
        await db.close()

if __name__ == "__main__":
    asyncio.run(migrate_to_workspaces())
```

### 4.3 Index Migration

**Current Index:**
```cypher
CREATE VECTOR INDEX FOR (c:Chunk) ON (c.embedding)
```

**Post-Migration (No Change Required):**
The existing vector index continues to work. Filtering happens at query time:

```cypher
CALL db.idx.vector.queryNodes('chunk_embedding_idx', $limit, $embedding)
YIELD node AS c, score
WHERE c.workspace_id = $workspace_id  // Filter during search
RETURN c
```

**Optional Optimization (Future):**
Create composite index on `(workspace_id, embedding)` if FalkorDB supports it in future versions.

---

## 5. Best Practices for Workspace APIs

### 5.1 CRUD Operations

Based on industry patterns and [Multi-Tenant Database Design 2024](https://daily.dev/blog/multi-tenant-database-design-patterns-2024):

#### **Create Workspace**

```python
async def create_workspace(
    self,
    workspace_id: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create new workspace (logical partition).

    Args:
        workspace_id: Unique workspace identifier
        metadata: Workspace metadata (owner, created_at, etc.)

    Returns:
        Workspace info dict

    Raises:
        ValidationError: If workspace_id invalid or already exists
    """
    # Validate workspace_id
    if not re.match(r'^[a-z0-9][a-z0-9_-]{0,62}$', workspace_id):
        raise ValidationError(
            "workspace_id must be lowercase alphanumeric, hyphens, "
            "underscores, max 63 chars"
        )

    # Check if exists
    result = await self.db_client.graph_query(
        "MATCH (w:Workspace {id: $workspace_id}) RETURN w",
        {"workspace_id": workspace_id}
    )
    if result.row_count > 0:
        raise ValidationError(f"Workspace {workspace_id} already exists")

    # Create workspace metadata node
    await self.db_client.graph_query(
        """
        CREATE (w:Workspace {
            id: $workspace_id,
            metadata: $metadata,
            created_at: $created_at
        })
        RETURN w
        """,
        {
            "workspace_id": workspace_id,
            "metadata": json.dumps(metadata or {}),
            "created_at": datetime.now(timezone.utc).isoformat()
        }
    )

    return {
        "workspace_id": workspace_id,
        "metadata": metadata,
        "status": "created"
    }
```

#### **List Workspaces**

```python
async def list_workspaces(self) -> List[Dict[str, Any]]:
    """List all workspaces with stats."""

    result = await self.db_client.graph_query(
        """
        MATCH (w:Workspace)
        OPTIONAL MATCH (m:Memory {workspace_id: w.id})
        RETURN w.id AS workspace_id,
               w.metadata AS metadata,
               w.created_at AS created_at,
               count(m) AS memory_count
        ORDER BY w.created_at DESC
        """
    )

    return [
        {
            "workspace_id": row["workspace_id"],
            "metadata": json.loads(row["metadata"]),
            "created_at": row["created_at"],
            "memory_count": row["memory_count"]
        }
        for row in result.rows
    ]
```

#### **Delete Workspace**

```python
async def delete_workspace(
    self,
    workspace_id: str,
    cascade: bool = False
) -> Dict[str, Any]:
    """
    Delete workspace and optionally all data.

    Args:
        workspace_id: Workspace to delete
        cascade: If True, delete all memories/chunks/entities

    Returns:
        Delete statistics
    """
    if cascade:
        # Delete all data in workspace
        result = await self.db_client.graph_query(
            """
            MATCH (n)
            WHERE n.workspace_id = $workspace_id
            DETACH DELETE n
            RETURN count(n) AS deleted_nodes
            """,
            {"workspace_id": workspace_id}
        )
        deleted_count = result.rows[0]["deleted_nodes"]
    else:
        # Check if workspace has data
        result = await self.db_client.graph_query(
            "MATCH (n) WHERE n.workspace_id = $workspace_id RETURN count(n)",
            {"workspace_id": workspace_id}
        )
        if result.rows[0]["count(n)"] > 0:
            raise ValidationError(
                f"Workspace {workspace_id} contains data. "
                "Use cascade=True to delete."
            )
        deleted_count = 0

    # Delete workspace metadata
    await self.db_client.graph_query(
        "MATCH (w:Workspace {id: $workspace_id}) DELETE w",
        {"workspace_id": workspace_id}
    )

    return {
        "workspace_id": workspace_id,
        "deleted_nodes": deleted_count,
        "status": "deleted"
    }
```

### 5.2 Validation and Naming Conventions

**Workspace ID Rules:**
- Format: `^[a-z0-9][a-z0-9_-]{0,62}$`
- Length: 1-63 characters
- Characters: Lowercase letters, numbers, hyphens, underscores
- Start: Must start with letter or number (not hyphen/underscore)
- Reserved: `default`, `system`, `admin`, `test`

**Examples:**
- ✅ Valid: `project-a`, `user_123`, `finance2024`, `dev`
- ❌ Invalid: `Project-A` (uppercase), `-temp` (starts with hyphen), `a` (too short with underscore)

### 5.3 Set Current Workspace (MCP Tool)

```python
class SetWorkspaceTool(MCPTool):
    """MCP tool to switch workspace for current session."""

    name = "set_current_workspace"
    description = "Switch to a different workspace for this session"

    input_schema = {
        "type": "object",
        "properties": {
            "workspace_id": {
                "type": "string",
                "description": "Workspace ID to switch to",
                "pattern": "^[a-z0-9][a-z0-9_-]{0,62}$"
            }
        },
        "required": ["workspace_id"]
    }

    async def execute(self, arguments: dict) -> dict:
        workspace_id = arguments["workspace_id"]
        session_id = self.get_session_id()  # From MCP context

        # Validate workspace exists
        result = await self.memory_processor.db_client.graph_query(
            "MATCH (w:Workspace {id: $workspace_id}) RETURN w",
            {"workspace_id": workspace_id}
        )
        if result.row_count == 0:
            raise ValidationError(f"Workspace {workspace_id} not found")

        # Update session state
        self.session_manager.set_workspace(session_id, workspace_id)

        return {
            "content": [{
                "type": "text",
                "text": f"Switched to workspace: {workspace_id}"
            }],
            "workspace_id": workspace_id,
            "session_id": session_id
        }
```

---

## 6. Implementation Plan

### 6.1 Phase 1: Foundation (Week 1)

**Tasks:**
1. Add `SessionManager` class to `zapomni_mcp/server.py`
2. Add `workspace_id` parameter to `MemoryProcessor` methods
3. Update `FalkorDBClient.add_memory()` to accept and store `workspace_id`
4. Update `FalkorDBClient.vector_search()` to filter by `workspace_id`
5. Create migration script `scripts/migrate_to_workspaces.py`

**Deliverables:**
- Updated core classes with workspace support
- Migration script tested on dev database
- Unit tests for workspace filtering

### 6.2 Phase 2: MCP Tools (Week 2)

**Tasks:**
1. Create `CreateWorkspaceTool`
2. Create `ListWorkspacesTool`
3. Create `DeleteWorkspaceTool`
4. Create `SetWorkspaceTool`
5. Update existing tools (`add_memory`, `search_memory`) to use session workspace

**Deliverables:**
- 4 new MCP tools registered
- Integration tests for workspace CRUD
- Documentation for MCP tool usage

### 6.3 Phase 3: Migration & Testing (Week 3)

**Tasks:**
1. Run migration on production database (backup first!)
2. Verify all existing data moved to "default" workspace
3. Create sample workspaces for testing
4. Performance testing with multiple workspaces
5. Documentation updates

**Deliverables:**
- Production data migrated
- Performance benchmarks (query overhead < 10%)
- User documentation for workspace features

### 6.4 Phase 4: Optimization (Week 4)

**Tasks:**
1. Add workspace validation middleware
2. Implement workspace usage statistics
3. Add workspace-level quotas (optional)
4. Consider workspace-specific HNSW indexes if needed
5. Monitor query performance

**Deliverables:**
- Workspace analytics dashboard
- Quota enforcement (if needed)
- Performance tuning recommendations

---

## 7. Risks and Mitigations

### 7.1 Risk: Data Leakage

**Description:** Queries missing `workspace_id` filter could leak data across workspaces.

**Mitigation:**
- Add validation layer that injects `workspace_id` into ALL queries
- Create query builder helper that enforces workspace filtering
- Add integration tests that verify cross-workspace isolation
- Code review checklist for workspace filtering

**Example Validation Layer:**
```python
class WorkspaceAwareQueryBuilder:
    """Ensures all queries include workspace_id filter."""

    def build_query(self, cypher: str, params: dict, workspace_id: str) -> tuple:
        # Parse Cypher and inject workspace filter
        if "WHERE" in cypher.upper():
            cypher = cypher.replace(
                "WHERE",
                f"WHERE n.workspace_id = $__workspace_id AND",
                1
            )
        else:
            cypher = cypher.replace(
                "RETURN",
                f"WHERE n.workspace_id = $__workspace_id RETURN",
                1
            )
        params["__workspace_id"] = workspace_id
        return cypher, params
```

### 7.2 Risk: Performance Degradation

**Description:** Adding `workspace_id` filtering to every query may slow search by 5-15%.

**Mitigation:**
- Benchmark current performance (baseline)
- Monitor P95/P99 latencies post-migration
- Create property index on `workspace_id` if needed
- Consider workspace-specific indexes for large workspaces (>100K nodes)

**Monitoring Query:**
```python
# Before migration
baseline_latency = measure_search_latency(query, iterations=1000)

# After migration
workspace_latency = measure_search_latency(query, workspace_id="test", iterations=1000)

overhead_percent = ((workspace_latency - baseline_latency) / baseline_latency) * 100
assert overhead_percent < 15, f"Overhead too high: {overhead_percent}%"
```

### 7.3 Risk: Migration Failure

**Description:** Large dataset migration could timeout or corrupt data.

**Mitigation:**
- Backup database before migration
- Run migration on staging environment first
- Use batched migration (10K nodes at a time)
- Add rollback script
- Monitor migration progress with counts

**Rollback Script:**
```python
# Rollback: Remove workspace_id property
async def rollback_migration():
    await db.graph_query("MATCH (n) REMOVE n.workspace_id")
    print("Rolled back workspace_id property")
```

### 7.4 Risk: Session State Loss

**Description:** In-memory session state lost on server restart.

**Mitigation:**
- Accept this trade-off for MVP (session resets to "default")
- Document session behavior in user guide
- Plan Redis-backed session storage for v2
- Add session persistence flag to configuration

### 7.5 Risk: Workspace Naming Conflicts

**Description:** Users create workspaces with reserved names or invalid characters.

**Mitigation:**
- Enforce strict validation regex
- Maintain reserved name list
- Return clear error messages
- Suggest alternative names in error

**Validation:**
```python
RESERVED_WORKSPACES = {"default", "system", "admin", "test", "global"}

def validate_workspace_id(workspace_id: str):
    if workspace_id in RESERVED_WORKSPACES:
        raise ValidationError(
            f"Workspace name '{workspace_id}' is reserved. "
            f"Try: {workspace_id}_workspace"
        )
    if not re.match(r'^[a-z0-9][a-z0-9_-]{0,62}$', workspace_id):
        raise ValidationError(
            "Workspace ID must be lowercase alphanumeric, "
            "hyphens, underscores, 1-63 chars"
        )
```

---

## 8. Code Examples

### 8.1 Updated FalkorDBClient.add_memory()

```python
async def add_memory(
    self,
    memory: Memory,
    workspace_id: str = "default"
) -> str:
    """
    Store memory with workspace isolation.

    Args:
        memory: Memory object with text, chunks, embeddings
        workspace_id: Workspace to store memory in

    Returns:
        memory_id: UUID of stored memory
    """
    memory_id = str(uuid.uuid4())
    chunk_ids = [str(uuid.uuid4()) for _ in range(len(memory.chunks))]

    cypher = """
    // Create Memory node with workspace_id
    CREATE (m:Memory {
        id: $memory_id,
        workspace_id: $workspace_id,
        text: $text,
        tags: $tags,
        source: $source,
        metadata: $metadata,
        created_at: $timestamp
    })

    // Create Chunk nodes with workspace_id
    WITH m
    UNWIND $chunks_data AS chunk_data
    CREATE (c:Chunk {
        id: chunk_data.id,
        workspace_id: $workspace_id,
        text: chunk_data.text,
        index: chunk_data.index,
        embedding: vecf32(chunk_data.embedding)
    })
    CREATE (m)-[:HAS_CHUNK {index: chunk_data.index}]->(c)

    RETURN m.id AS memory_id
    """

    chunks_data = [
        {
            "id": chunk_ids[i],
            "text": chunk.text,
            "index": chunk.index,
            "embedding": embedding,
        }
        for i, (chunk, embedding) in enumerate(zip(memory.chunks, memory.embeddings))
    ]

    parameters = {
        "memory_id": memory_id,
        "workspace_id": workspace_id,  # Inject workspace
        "text": memory.text,
        "tags": memory.metadata.get("tags", []),
        "source": memory.metadata.get("source", ""),
        "metadata": json.dumps(memory.metadata),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "chunks_data": chunks_data,
    }

    result = await self._execute_cypher(cypher, parameters)
    return memory_id
```

### 8.2 Updated FalkorDBClient.vector_search()

```python
async def vector_search(
    self,
    embedding: List[float],
    limit: int = 10,
    workspace_id: str = "default",
    filters: Optional[Dict[str, Any]] = None,
) -> List[SearchResult]:
    """
    Vector search with workspace filtering.

    Args:
        embedding: Query embedding vector
        limit: Max results
        workspace_id: Workspace to search in
        filters: Additional filters (tags, source, etc.)

    Returns:
        List of SearchResult objects from specified workspace
    """
    # Build query with workspace filter
    cypher = """
    // Vector search with workspace isolation
    CALL db.idx.vector.queryNodes('chunk_embedding_idx', $limit * 3, $embedding)
    YIELD node AS c, score
    WHERE c.workspace_id = $workspace_id
    WITH c, score
    MATCH (m:Memory {workspace_id: $workspace_id})-[:HAS_CHUNK]->(c)
    RETURN DISTINCT
        m.id AS memory_id,
        c.id AS chunk_id,
        c.text AS text,
        c.index AS chunk_index,
        score AS similarity_score,
        m.tags AS tags,
        m.source AS source,
        m.created_at AS timestamp
    ORDER BY similarity_score DESC
    LIMIT $limit
    """

    params = {
        "embedding": embedding,
        "limit": limit,
        "workspace_id": workspace_id
    }

    result = await self._execute_cypher(cypher, params)
    return self._parse_search_results(result)
```

### 8.3 SessionManager Implementation

```python
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional

@dataclass
class WorkspaceSession:
    """Session state for workspace context."""
    session_id: str
    workspace_id: str
    created_at: datetime
    updated_at: datetime = None

    def __post_init__(self):
        if self.updated_at is None:
            self.updated_at = self.created_at

class SessionManager:
    """
    Manages workspace state per MCP session.

    Ensures session isolation and workspace context tracking.
    """

    def __init__(self):
        self._sessions: Dict[str, WorkspaceSession] = {}
        self._default_workspace = "default"

    def get_or_create_session(self, session_id: str) -> WorkspaceSession:
        """Get existing session or create with default workspace."""
        if session_id not in self._sessions:
            self._sessions[session_id] = WorkspaceSession(
                session_id=session_id,
                workspace_id=self._default_workspace,
                created_at=datetime.now(timezone.utc)
            )
        return self._sessions[session_id]

    def set_workspace(self, session_id: str, workspace_id: str) -> None:
        """Switch workspace for session."""
        session = self.get_or_create_session(session_id)
        session.workspace_id = workspace_id
        session.updated_at = datetime.now(timezone.utc)

    def get_workspace(self, session_id: str) -> str:
        """Get current workspace for session."""
        return self.get_or_create_session(session_id).workspace_id

    def delete_session(self, session_id: str) -> None:
        """Remove session state (on disconnect)."""
        self._sessions.pop(session_id, None)

    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        return {
            "total_sessions": len(self._sessions),
            "active_sessions": len([
                s for s in self._sessions.values()
                if (datetime.now(timezone.utc) - s.updated_at).seconds < 3600
            ]),
            "default_workspace": self._default_workspace
        }
```

### 8.4 Integration with MCP Server

```python
# In zapomni_mcp/server.py

class MCPServer:
    def __init__(self, core_engine: Any, config: Optional[Settings] = None):
        # ... existing init ...

        # Add session manager
        self._session_manager = SessionManager()

    async def run_sse(self, host: str = "127.0.0.1", port: int = 8000):
        """SSE mode with session management."""

        @self._server.call_tool()
        async def handle_call_tool(
            name: str,
            arguments: dict,
            session_id: str  # From SSE context
        ) -> list:
            # Get workspace from session or argument
            workspace_id = arguments.get("workspace_id")
            if workspace_id is None:
                workspace_id = self._session_manager.get_workspace(session_id)

            # Inject workspace into tool arguments
            arguments["workspace_id"] = workspace_id
            arguments["session_id"] = session_id

            # Execute tool
            result = await self._tools[name].execute(arguments)
            return result.get("content", [])
```

---

## 9. Testing Strategy

### 9.1 Unit Tests

```python
# tests/unit/test_workspace_isolation.py

import pytest
from zapomni_core import MemoryProcessor
from zapomni_db import FalkorDBClient

@pytest.mark.asyncio
async def test_workspace_isolation():
    """Verify memories in different workspaces are isolated."""

    db = FalkorDBClient()
    await db.init_async()
    processor = MemoryProcessor(db_client=db, ...)

    # Add memory to workspace A
    id_a = await processor.add_memory(
        text="Workspace A content",
        workspace_id="workspace_a"
    )

    # Add memory to workspace B
    id_b = await processor.add_memory(
        text="Workspace B content",
        workspace_id="workspace_b"
    )

    # Search in workspace A - should only find A
    results_a = await processor.search_memory(
        query="content",
        workspace_id="workspace_a"
    )
    assert len(results_a) == 1
    assert results_a[0].memory_id == id_a

    # Search in workspace B - should only find B
    results_b = await processor.search_memory(
        query="content",
        workspace_id="workspace_b"
    )
    assert len(results_b) == 1
    assert results_b[0].memory_id == id_b

@pytest.mark.asyncio
async def test_cross_workspace_operations():
    """Verify cross-context read/write operations."""

    processor = ...

    # Read from workspace A
    results_a = await processor.search_memory(
        query="Python",
        workspace_id="project_a"
    )

    # Write to workspace B
    memory_id = await processor.add_memory(
        text="Django framework info",
        workspace_id="project_b"
    )

    # Verify written to B, not A
    results_b = await processor.search_memory(
        query="Django",
        workspace_id="project_b"
    )
    assert len(results_b) == 1

    results_a = await processor.search_memory(
        query="Django",
        workspace_id="project_a"
    )
    assert len(results_a) == 0
```

### 9.2 Integration Tests

```python
# tests/integration/test_workspace_mcp_tools.py

@pytest.mark.asyncio
async def test_mcp_workspace_lifecycle():
    """Test full workspace lifecycle via MCP tools."""

    server = MCPServer(...)

    # Create workspace
    result = await server.execute_tool(
        "create_workspace",
        {"workspace_id": "test_project", "metadata": {"owner": "alice"}}
    )
    assert result["status"] == "created"

    # Set current workspace
    result = await server.execute_tool(
        "set_current_workspace",
        {"workspace_id": "test_project"},
        session_id="session_123"
    )

    # Add memory (uses session workspace)
    result = await server.execute_tool(
        "add_memory",
        {"text": "Test memory"},
        session_id="session_123"
    )
    memory_id = result["memory_id"]

    # Verify memory in correct workspace
    result = await server.execute_tool(
        "search_memory",
        {"query": "test", "workspace_id": "test_project"}
    )
    assert len(result["results"]) == 1

    # Delete workspace
    result = await server.execute_tool(
        "delete_workspace",
        {"workspace_id": "test_project", "cascade": True}
    )
    assert result["deleted_nodes"] > 0
```

### 9.3 Performance Tests

```python
# tests/performance/test_workspace_overhead.py

@pytest.mark.benchmark
async def test_vector_search_overhead():
    """Measure query overhead from workspace filtering."""

    # Setup: 10K memories across 10 workspaces
    for i in range(10):
        workspace_id = f"workspace_{i}"
        for j in range(1000):
            await processor.add_memory(
                text=f"Memory {j} in workspace {i}",
                workspace_id=workspace_id
            )

    # Baseline: Search without filter (legacy mode)
    start = time.time()
    for _ in range(100):
        await db.vector_search(embedding, limit=10)
    baseline_time = time.time() - start

    # With workspace filter
    start = time.time()
    for _ in range(100):
        await db.vector_search(
            embedding,
            limit=10,
            workspace_id="workspace_5"
        )
    filtered_time = time.time() - start

    # Calculate overhead
    overhead_percent = ((filtered_time - baseline_time) / baseline_time) * 100

    print(f"Workspace filter overhead: {overhead_percent:.1f}%")
    assert overhead_percent < 15, "Overhead exceeds 15% threshold"
```

---

## 10. Recommendations Summary

### 10.1 Immediate Actions (Week 1)

1. **Implement Property-Based Filtering**
   - Add `workspace_id: str` parameter to all database methods
   - Update Cypher queries to include `WHERE n.workspace_id = $workspace_id`
   - Default to `"default"` workspace for backward compatibility

2. **Add SessionManager**
   - Create `SessionManager` class for MCP session tracking
   - Integrate with `MCPServer` for automatic workspace context
   - Support explicit `workspace_id` override in tool arguments

3. **Create Migration Script**
   - Batch migration to add `workspace_id = "default"` to existing nodes
   - Test on staging database first
   - Verify data integrity with count checks

### 10.2 Short-Term Priorities (Weeks 2-3)

4. **Implement Workspace CRUD Tools**
   - `create_workspace`: Create new logical workspace
   - `list_workspaces`: List all workspaces with stats
   - `set_current_workspace`: Switch session workspace
   - `delete_workspace`: Remove workspace and data

5. **Add Validation Layer**
   - Workspace ID format validation
   - Reserved name checking
   - Query builder with automatic workspace injection

6. **Performance Testing**
   - Benchmark query overhead (<15% acceptable)
   - Monitor vector search latency
   - Consider workspace-specific indexes if needed

### 10.3 Long-Term Considerations (Weeks 4+)

7. **Session Persistence**
   - Evaluate Redis-backed session storage for multi-instance deployments
   - Document session behavior (in-memory vs. persistent)

8. **Advanced Isolation**
   - Consider separate graph instances for large/premium workspaces
   - Implement workspace quotas (memory limits, rate limits)

9. **Monitoring & Analytics**
   - Workspace usage dashboards
   - Cross-workspace analytics (aggregated insights)
   - Quota enforcement and alerts

### 10.4 Architecture Decision

**Recommended Approach:** **Property-Based Filtering (Pattern 2)** with session-aware MCP server.

**Rationale:**
- ✅ Simplest implementation (minimal code changes)
- ✅ Efficient resource usage (single graph)
- ✅ Supports cross-workspace analytics
- ✅ Low migration complexity
- ✅ Future-proof (can migrate to multi-graph if needed)

**Trade-offs Accepted:**
- 5-15% query overhead (acceptable for moderate workspace count)
- Requires careful validation to prevent data leakage
- Shared schema across workspaces

---

## 11. Sources and References

### Research Sources

1. **FalkorDB Multi-Tenancy:**
   - [Graph Database Multi-Tenant Cloud Security Architecture](https://www.falkordb.com/blog/graph-database-multi-tenant-cloud-security/)
   - [Graphiti + FalkorDB: Integration for Multi-Agent Systems](https://www.falkordb.com/blog/graphiti-falkordb-multi-agent-performance/)

2. **Neo4j Patterns:**
   - [Multi Tenancy in Neo4j: A Worked Example](https://neo4j.com/developer/multi-tenancy-worked-example/)
   - [Multi Tenancy on Neo4j - Community Discussion](https://community.neo4j.com/t/multi-tenancy-on-neo4j/10627)

3. **Graph Database Multi-Tenancy:**
   - [Multi-Tenancy in Graph Databases and Why Should You Care?](https://memgraph.com/blog/why-multi-tenancy-matters-in-graph-databases)
   - [AWS Neptune Multi-Tenant Architectures](https://aws.amazon.com/blogs/database/build-multi-tenant-architectures-on-amazon-neptune/)

4. **Vector Index Filtering:**
   - [Semantic Filtering in HNSW: Structuring Vector Indexes by Domain](https://emilianobilli.medium.com/semantic-filtering-in-hnsw-structuring-vector-indexes-by-domain-bd447695cfb7)
   - [The Achilles Heel of Vector Search: Filters](https://yudhiesh.github.io/2025/05/09/the-achilles-heel-of-vector-search-filters/)
   - [Oracle HNSW Optimizer Plans](https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/optimizer-plans-hnsw-vector-indexes.html)

5. **MCP Session Management:**
   - [MCP Session Isolation - GitHub Issue #1087](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1087)
   - [Managing Stateful MCP Server Sessions](https://codesignal.com/learn/courses/developing-and-integrating-an-mcp-server-in-typescript/lessons/stateful-mcp-server-sessions)
   - [Configure MCP Servers for Multiple Connections](https://mcpcat.io/guides/configuring-mcp-servers-multiple-simultaneous-connections/)
   - [MCP Session Management: Best Practices 2025](https://www.byteplus.com/en/topic/541419)

6. **General Multi-Tenant Patterns:**
   - [Multi-Tenant Database Design Patterns 2024](https://daily.dev/blog/multi-tenant-database-design-patterns-2024)

### Technical Documentation

- FalkorDB Documentation: https://www.falkordb.com/
- Neo4j Developer Guides: https://neo4j.com/developer/
- MCP Specification: https://github.com/modelcontextprotocol/modelcontextprotocol

---

## Appendix A: Workspace Data Model

```
┌─────────────────┐
│   Workspace     │
├─────────────────┤
│ id: string      │ (PK)
│ metadata: json  │
│ created_at: ts  │
└─────────────────┘
        │
        │ (logical grouping)
        ↓
┌─────────────────┐
│     Memory      │
├─────────────────┤
│ id: uuid        │ (PK)
│ workspace_id: * │ (FK - logical)
│ text: string    │
│ tags: list      │
│ source: string  │
│ created_at: ts  │
└─────────────────┘
        │
        │ HAS_CHUNK
        ↓
┌─────────────────┐
│     Chunk       │
├─────────────────┤
│ id: uuid        │ (PK)
│ workspace_id: * │ (FK - logical)
│ text: string    │
│ index: int      │
│ embedding: vec  │ (768-dim)
└─────────────────┘
        │
        │ MENTIONS
        ↓
┌─────────────────┐
│     Entity      │
├─────────────────┤
│ id: uuid        │ (PK)
│ workspace_id: * │ (FK - logical)
│ name: string    │
│ type: string    │
│ confidence: f   │
└─────────────────┘
```

**Note:** `workspace_id` is a property filter, not a foreign key constraint (graph DB pattern).

---

## Appendix B: Example Cypher Queries

### Create Workspace
```cypher
CREATE (w:Workspace {
    id: 'project_alpha',
    metadata: '{"owner": "alice", "team": "engineering"}',
    created_at: '2025-11-25T10:00:00Z'
})
RETURN w
```

### Add Memory with Workspace
```cypher
CREATE (m:Memory {
    id: 'mem-123',
    workspace_id: 'project_alpha',
    text: 'Important project documentation',
    tags: ['docs', 'project'],
    created_at: '2025-11-25T10:05:00Z'
})
RETURN m
```

### Vector Search with Workspace Filter
```cypher
CALL db.idx.vector.queryNodes('chunk_embedding_idx', 20, $embedding)
YIELD node AS c, score
WHERE c.workspace_id = 'project_alpha'
WITH c, score
MATCH (m:Memory {workspace_id: 'project_alpha'})-[:HAS_CHUNK]->(c)
RETURN m.id, c.text, score
ORDER BY score DESC
LIMIT 10
```

### List Workspaces with Stats
```cypher
MATCH (w:Workspace)
OPTIONAL MATCH (m:Memory {workspace_id: w.id})
OPTIONAL MATCH (e:Entity {workspace_id: w.id})
RETURN
    w.id AS workspace_id,
    w.metadata AS metadata,
    count(DISTINCT m) AS memory_count,
    count(DISTINCT e) AS entity_count
ORDER BY w.created_at DESC
```

### Delete Workspace (Cascade)
```cypher
// Delete all nodes with workspace_id
MATCH (n)
WHERE n.workspace_id = 'project_alpha'
DETACH DELETE n

// Delete workspace metadata
MATCH (w:Workspace {id: 'project_alpha'})
DELETE w
```

---

**End of Research Report**

---

**Prepared by:** Research Agent
**For:** Zapomni Development Team
**Next Steps:** Review findings → Create design spec → Implement Phase 1
