# MemoryProcessor.get_stats() - Function Specification

**Level:** 3 (Function)
**Component:** MemoryProcessor
**Module:** zapomni_core
**Author:** Goncharenko Anton aka alienxs2
**Status:** Draft
**Version:** 1.0
**Created:** 2025-11-23

## Function Signature

```python
async def get_stats(self) -> Dict[str, Any]:
    """Get comprehensive memory system statistics."""
```

## Purpose

Aggregate statistics from all components (DB, cache, extractor) and return comprehensive system metrics.

## Parameters

None

## Returns

```python
{
    "total_memories": 1234,           # int
    "total_chunks": 5678,             # int
    "database_size_mb": 45.6,         # float
    "avg_chunks_per_memory": 4.6,    # float
    "cache_hit_rate": 0.63,          # float 0-1 (Phase 2)
    "avg_query_latency_ms": 156,     # int
    "total_entities": 987,           # int (Phase 2)
    "oldest_memory_date": datetime,  # datetime
    "newest_memory_date": datetime   # datetime
}
```

## Raises

- `DatabaseError`: If DB query fails
- `StatisticsError`: If calculation fails

## Algorithm

```
1. Query database for total_memories, total_chunks
2. Calculate avg_chunks_per_memory
3. Get cache stats (if enabled)
4. Get graph stats (if enabled)
5. Get latency metrics
6. Return formatted dict
```

## Edge Cases

1. **No memories** → total_memories=0, avg=0
2. **Cache disabled** → cache stats not in result
3. **Graph not built** → entity stats not in result

## Test Scenarios (10)

1. test_get_stats_success
2. test_get_stats_no_memories
3. test_get_stats_with_cache
4. test_get_stats_with_graph
5. test_get_stats_database_error
6. test_get_stats_calculation_error
7. test_get_stats_missing_timestamps
8. test_get_stats_performance
9. test_get_stats_concurrent_calls
10. test_get_stats_cache_disabled

## Performance

- Target: < 100ms
- No expensive computations

**Status:** Draft v1.0 | **Author:** Goncharenko Anton aka alienxs2 | **License:** MIT
