# get_stats Tool

Retrieve comprehensive statistics about the memory system including total memories, chunks, database size, and performance metrics.

## Overview

The `get_stats` tool provides visibility into your memory system's state and health. It returns statistics about:

- **Capacity**: Total memories and chunks stored
- **Storage**: Database size in MB
- **Performance**: Query latency and cache hit rates
- **Graph**: Entities and relationships (Phase 2)

## Tool Identifier

```
name: "get_stats"
```

## Description

"Get statistics about the memory system including total memories, chunks, database size, and performance metrics."

## Parameters

This tool takes **no parameters**. The arguments object should be empty `{}`.

```json
{
}
```

## Request Example

### Standard Request

```json
{
}
```

**Note**: Any provided arguments are ignored as per tool specification.

## Response Format

### Success Response

Status: `isError: false`

```json
{
  "content": [
    {
      "type": "text",
      "text": "Memory System Statistics:\nTotal Memories: 42\nTotal Chunks: 156\nDatabase Size: 12.45 MB\nAverage Chunks per Memory: 3.7\nCache Hit Rate: 65.3%\nAvg Query Latency: 245 ms\nOldest Memory: 2025-01-15T10:30:00Z\nNewest Memory: 2025-11-24T16:45:00Z"
    }
  ],
  "isError": false
}
```

### Error Response

Status: `isError: true`

```json
{
  "content": [
    {
      "type": "text",
      "text": "Error: Failed to retrieve statistics - Connection refused"
    }
  ],
  "isError": true
}
```

## Response Fields

### Required Statistics (Always Present)

| Field | Type | Example | Description |
|-------|------|---------|-------------|
| **Total Memories** | integer | `42` | Number of distinct memory entries stored |
| **Total Chunks** | integer | `156` | Total text chunks across all memories |
| **Database Size** | float (MB) | `12.45` | Total database size in megabytes |
| **Average Chunks per Memory** | float | `3.7` | Mean chunks per memory entry |

### Optional Statistics (If Available)

| Field | Type | Example | Description |
|-------|------|---------|-------------|
| **Total Entities** | integer | `127` | Number of entities in knowledge graph (Phase 2) |
| **Total Relationships** | integer | `89` | Number of relationships in knowledge graph (Phase 2) |
| **Cache Hit Rate** | float (0-1) | `0.653` | Embedding cache hit rate percentage |
| **Avg Query Latency** | float (ms) | `245.5` | Average search query latency in milliseconds |
| **Oldest Memory** | ISO 8601 | `2025-01-15T10:30:00Z` | Timestamp of first memory stored |
| **Newest Memory** | ISO 8601 | `2025-11-24T16:45:00Z` | Timestamp of most recent memory |

## Statistics Explained

### Total Memories

**Definition**: Number of distinct `add_memory` operations completed successfully.

Each call to `add_memory` creates one memory, even if it generates multiple chunks.

**Example**:
- Added "Python tutorial" → 1 memory (5 chunks)
- Added "JavaScript guide" → 1 memory (3 chunks)
- Added "Go reference" → 1 memory (4 chunks)
- Total Memories: 3

**Use for**: Understanding overall volume of stored information

### Total Chunks

**Definition**: Sum of chunks across all memories.

Chunks are created during semantic chunking of input text.

**Example**:
- Memory 1: 5 chunks
- Memory 2: 3 chunks
- Memory 3: 4 chunks
- Total Chunks: 12

**Use for**: Understanding processing load and search coverage

### Database Size (MB)

**Definition**: Total size of FalkorDB (graph + vector storage) in megabytes.

Includes:
- Chunk text content
- Vector embeddings (768 dimensions × 4 bytes per float)
- Metadata
- Graph nodes and edges
- Indices

**Estimation**:
- Per chunk: ~2 KB text + 3 KB embeddings + 1 KB metadata = ~6 KB/chunk
- 1000 chunks ≈ 6 MB database

**Use for**: Storage planning and capacity monitoring

### Average Chunks per Memory

**Definition**: `Total Chunks ÷ Total Memories`

Indicates typical memory size (shorter = smaller memories, longer = larger memories).

**Examples**:
- 2.5 avg chunks → Short memories (500-1000 tokens each)
- 5.0 avg chunks → Medium memories (1000-2500 tokens each)
- 10+ avg chunks → Large memories (2500+ tokens each)

**Use for**: Understanding memory composition

### Cache Hit Rate (Phase 2)

**Definition**: Percentage of embedding requests served from cache vs. computed fresh.

Cached queries are faster (< 1ms) vs. fresh embeddings (100-200ms).

**Interpretation**:
- **< 30%**: Low reuse, consider more varied queries
- **30-60%**: Normal usage pattern
- **> 60%**: High efficiency, good cache utilization

**Use for**: Performance optimization decisions

### Average Query Latency (Phase 2)

**Definition**: Mean latency of all search queries in milliseconds.

Breakdown:
- Query embedding: 50-200ms
- Vector search: 10-100ms
- Result formatting: < 10ms
- **Total**: 100-500ms typical

**Use for**: User experience monitoring and optimization

### Oldest Memory

**Definition**: Timestamp (ISO 8601) of first memory stored.

Shows when memory system was first populated.

**Example**: `2025-01-15T10:30:00Z`

**Use for**: Tracking retention period and data aging

### Newest Memory

**Definition**: Timestamp (ISO 8601) of most recent memory stored.

Shows system activity recency.

**Use for**: Verifying recent memory additions

### Total Entities (Phase 2)

**Definition**: Number of entities extracted into knowledge graph.

Entities are concepts like:
- People: "Guido van Rossum"
- Tools: "Python", "Docker"
- Concepts: "Machine Learning", "REST APIs"

**Use for**: Understanding knowledge graph size (Phase 2+)

### Total Relationships (Phase 2)

**Definition**: Number of relationships between entities in knowledge graph.

Examples:
- "Python" → created_by → "Guido van Rossum"
- "Docker" → used_for → "containerization"
- "Machine Learning" → subset_of → "AI"

**Use for**: Understanding knowledge graph density (Phase 2+)

## Error Codes

### Database Errors (DB_*, CONN_*)

| Code | Message | Cause | Action |
|------|---------|-------|--------|
| **DB_001** | Generic database error | Database query failed | Check logs, restart FalkorDB |
| **CONN_001** | FalkorDB connection refused | Database service down | Start FalkorDB: `docker-compose up -d` |
| **CONN_002** | Connection timeout | Network issue with database | Check connectivity, restart services |
| **QUERY_001** | Syntax error in Cypher query | Database query malformed | File bug report, contact maintainer |
| **QUERY_002** | Query timeout | Stats query took too long | Reduce dataset, increase timeout |

## Error Handling Examples

### Database Down

```python
result = await get_stats.execute({})
# Returns:
# Error: Failed to retrieve statistics - Connection refused
# Error Code: CONN_001
# Action: docker-compose up -d
```

### Connection Timeout

```python
result = await get_stats.execute({})
# Returns:
# Error: Failed to retrieve statistics - Connection timeout
# Error Code: CONN_002
# Action: Check network connectivity
```

### Query Timeout

```python
result = await get_stats.execute({})
# Returns:
# Error: Failed to retrieve statistics - Query timeout
# Error Code: QUERY_002
# Action: Wait and retry, or contact admin
```

## Usage Examples

### Python (Direct Library)

```python
from zapomni_mcp.tools import GetStatsTool
from zapomni_core.memory_processor import MemoryProcessor

processor = MemoryProcessor(...)
tool = GetStatsTool(processor)

# Get statistics
result = await tool.execute({})

# Parse results
if not result["isError"]:
    print(result["content"][0]["text"])
else:
    print(f"Error: {result['content'][0]['text']}")
```

### Python (MCP Client)

```python
from mcp import create_client

async with create_client("zapomni") as client:
    result = await client.call("get_stats", {})
    print(result.content[0].text)
```

### Python (Parsing Results)

```python
async def get_memory_stats():
    result = await get_stats.execute({})

    if result["isError"]:
        raise Exception(result["content"][0]["text"])

    stats_text = result["content"][0]["text"]

    # Parse the formatted output
    stats = {}
    for line in stats_text.split("\n"):
        if ": " in line:
            key, value = line.split(": ", 1)
            stats[key.strip()] = value.strip()

    return stats

stats = await get_memory_stats()
print(f"Stored {stats['Total Memories']} memories")
```

### JavaScript/Node.js

```javascript
const { createClient } = require("@anthropic-ai/sdk/mcp");

const client = await createClient("zapomni");

const result = await client.call("get_stats", {});
console.log(result.content[0].text);
```

### JavaScript (Parsing Results)

```javascript
async function getMemoryStats() {
  const result = await client.call("get_stats", {});

  if (result.isError) {
    throw new Error(result.content[0].text);
  }

  const statsText = result.content[0].text;
  const stats = {};

  for (const line of statsText.split("\n")) {
    const [key, value] = line.split(": ");
    if (key && value) {
      stats[key.trim()] = value.trim();
    }
  }

  return stats;
}

const stats = await getMemoryStats();
console.log(`Stored ${stats["Total Memories"]} memories`);
```

### cURL (If HTTP transport available)

```bash
curl -X POST http://localhost:5000/mcp/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "get_stats",
    "arguments": {}
  }'
```

## Real-World Examples

### Example 1: Fresh Install

```
Memory System Statistics:
Total Memories: 0
Total Chunks: 0
Database Size: 0.05 MB
Average Chunks per Memory: 0.0
```

**Interpretation**: No memories added yet, database ready for use

### Example 2: Small Knowledge Base

```
Memory System Statistics:
Total Memories: 5
Total Chunks: 18
Database Size: 0.55 MB
Average Chunks per Memory: 3.6
Cache Hit Rate: 42.3%
Avg Query Latency: 312 ms
Oldest Memory: 2025-11-20T09:15:00Z
Newest Memory: 2025-11-24T14:30:00Z
```

**Interpretation**:
- 5 memories with moderate content
- Good cache efficiency (42%)
- Acceptable query latency
- Recent activity (last 4 days)

### Example 3: Production System

```
Memory System Statistics:
Total Memories: 287
Total Chunks: 1,456
Database Size: 45.23 MB
Average Chunks per Memory: 5.1
Total Entities: 342
Total Relationships: 521
Cache Hit Rate: 68.5%
Avg Query Latency: 187 ms
Oldest Memory: 2024-06-01T00:00:00Z
Newest Memory: 2025-11-24T16:45:00Z
```

**Interpretation**:
- Large knowledge base (287 memories, 1.4K chunks)
- Good performance (187ms latency)
- Excellent cache efficiency (68.5%)
- Rich knowledge graph (342 entities, 521 relationships)
- Active use over 18 months

## Use Cases

### 1. Monitor System Health

```python
async def check_system_health():
    stats = await get_stats.execute({})

    if stats["isError"]:
        return False, "Database offline"

    # Parse stats...
    latency = parse_latency(stats)

    if latency > 1000:
        return False, f"High latency: {latency}ms"

    return True, "System healthy"
```

### 2. Capacity Planning

```python
async def should_archive_old_memories():
    stats = await get_stats.execute({})

    # Parse stats...
    db_size = parse_size(stats)  # MB

    if db_size > 1000:  # 1GB threshold
        return True, f"Database is {db_size}MB"

    return False, f"Database is {db_size}MB, still OK"
```

### 3. Performance Monitoring

```python
async def log_metrics():
    stats = await get_stats.execute({})

    # Parse stats...
    cache_hit = parse_cache_hit(stats)
    latency = parse_latency(stats)

    print(f"Cache hit rate: {cache_hit}%")
    print(f"Query latency: {latency}ms")

    # Send to monitoring system
    send_to_prometheus({
        "memory.cache_hit_rate": cache_hit,
        "memory.query_latency_ms": latency
    })
```

### 4. User Information

```python
async def summarize_for_user():
    stats = await get_stats.execute({})

    # Parse stats...
    memories = parse_memories(stats)
    chunks = parse_chunks(stats)
    size = parse_size(stats)

    return f"""
    You have stored:
    - {memories} memories
    - {chunks} chunks
    - Using {size:.2f}MB storage
    """
```

## Performance Characteristics

| Operation | Time (P95) | Notes |
|-----------|-----------|-------|
| Statistics query | < 100ms | Cached/optimized query |
| Result parsing | < 10ms | Simple formatting |
| **Total latency** | **< 100ms** | Very fast operation |

## Limits and Constraints

| Limit | Value | Notes |
|-------|-------|-------|
| Max memories | Unlimited | Limited by database storage |
| Max chunks | Unlimited | Limited by database storage |
| Max database size | Limited by disk | Typical 1GB per 3000 chunks |
| Frequency | No limit | Safe to call frequently |

## Best Practices

### 1. Regular Monitoring

```python
# Good: Check health periodically
async def monitor_system():
    while True:
        stats = await get_stats.execute({})
        if not stats["isError"]:
            log_metrics(stats)
        await asyncio.sleep(300)  # Every 5 minutes

# Bad: Only check when problems occur
# (reactive vs proactive)
```

### 2. Error Handling

```python
# Good: Handle gracefully
try:
    stats = await get_stats.execute({})
    if stats["isError"]:
        log_error("Stats unavailable")
    else:
        log_info(stats["content"][0]["text"])
except Exception as e:
    log_error(f"Unexpected error: {e}")

# Bad: Assume always works
stats = await get_stats.execute({})
print(stats["content"][0]["text"])
```

### 3. Caching Results

```python
# Good: Cache results with TTL
import time
last_stats = None
last_update = 0

async def get_cached_stats():
    global last_stats, last_update

    if time.time() - last_update > 60:  # Refresh every 60s
        last_stats = await get_stats.execute({})
        last_update = time.time()

    return last_stats

# Bad: Query on every request
async def get_fresh_stats():
    return await get_stats.execute({})  # Every request!
```

## Related Tools

- **[add_memory](./add_memory.md)** - Add memories (increases stats)
- **[search_memory](./search_memory.md)** - Search memories (measured in latency)

## See Also

- **[Request/Response Schemas](../schemas.md)** - Data model definitions
- **[Error Reference](../errors.md)** - Error codes and meanings
- **[Performance Tuning](../../development/performance.md)** - Optimization guide

## Troubleshooting

### "Connection refused"

**Cause**: FalkorDB service is down

**Solution**:
```bash
# Check status
docker-compose ps

# Restart if needed
docker-compose down
docker-compose up -d

# Verify
redis-cli ping  # Should return PONG
```

### "Connection timeout"

**Cause**: Network issue or service overloaded

**Solution**:
1. Check network connectivity
2. Wait a moment and retry
3. Restart services if problem persists

### "Query timeout"

**Cause**: Statistics query took too long (rare)

**Solution**:
1. Wait and retry
2. Check system load
3. Contact administrator if persistent

### High Query Latency (> 1000ms)

**Causes**:
- System under heavy load
- Database fragmented
- Large dataset

**Solutions**:
1. Archive old memories
2. Reduce dataset size
3. Optimize database (reindex)

## Version Information

- **Tool Version**: 1.0
- **Phase**: MVP (Current)
- **Status**: Stable
- **Last Updated**: 2025-11-24
