# TaskManager.submit_task() - Function Specification

**Level:** 3 (Function)
**Component:** TaskManager
**Module:** zapomni_core
**Author:** Goncharenko Anton aka alienxs2
**Status:** Draft
**Version:** 1.0
**Created:** 2025-11-23

## Function Signature

```python
async def submit_task(
    self,
    task_id: str,
    coro: Coroutine,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """Submit an async task for background execution."""
```

## Purpose

Create TaskInfo entry, schedule coroutine for background execution, return task_id immediately for tracking.

## Parameters

### task_id: str
- **Constraints:** Unique, non-empty
- **Format:** `"build_graph_001"`, `"codify_repo_xyz"`
- **Validation:** Must not already exist

### coro: Coroutine
- **Type:** Async coroutine to execute
- **Example:** `build_graph()`

### metadata: Optional[Dict[str, Any]]
- **Purpose:** Task-specific metadata
- **Fields:** `task_type`, `description`, custom fields
- **Example:** `{"task_type": "graph_build", "repo": "zapomni"}`

## Returns

- **Type:** `str`
- **Value:** The same `task_id` provided (for convenience)

## Raises

- `ValueError`: If task_id empty or already exists
- `TaskError`: If task queue full (max_concurrent limit reached)

## Algorithm

```
1. Validate task_id (non-empty, unique)
2. Create TaskInfo:
   - state = PENDING
   - progress = 0
   - created_at = now()
   - metadata = provided metadata
3. Store in self.tasks[task_id]
4. Add to queue: self.queue.put(task_id, coro)
5. Log task submission
6. Return task_id
```

## Edge Cases

1. **Duplicate task_id** → ValueError
2. **Empty task_id** → ValueError
3. **Queue full** → TaskError
4. **Invalid coroutine** → TypeError (caught by asyncio)

## Test Scenarios (10)

1. test_submit_task_success
2. test_submit_task_returns_task_id
3. test_submit_task_duplicate_raises
4. test_submit_task_empty_id_raises
5. test_submit_task_queue_full_raises
6. test_submit_task_with_metadata
7. test_submit_task_creates_task_info
8. test_submit_task_sets_pending_state
9. test_submit_task_logging
10. test_submit_task_performance

## Performance

- Target: < 10ms (non-blocking)

**Status:** Draft v1.0 | **Author:** Goncharenko Anton aka alienxs2 | **License:** MIT
