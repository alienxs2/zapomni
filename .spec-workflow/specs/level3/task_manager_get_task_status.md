# TaskManager.get_task_status() - Function Specification

**Level:** 3 | **Component:** TaskManager | **Module:** zapomni_core
**Author:** Goncharenko Anton aka alienxs2 | **Status:** Draft | **V:** 1.0

## Signature
```python
async def get_task_status(self, task_id: str) -> Dict[str, Any]:
    """
    Get status of background task.
    
    Args:
        task_id: UUID of task to query
        
    Returns:
        Dict with task status:
        {
            "task_id": str,
            "state": "pending" | "running" | "completed" | "failed" | "cancelled",
            "progress": int (0-100),
            "items_processed": int,
            "items_total": int,
            "started_at": Optional[datetime],
            "completed_at": Optional[datetime],
            "error": Optional[str]
        }
        
    Raises:
        KeyError: If task_id not found
        ValueError: If task_id invalid UUID format
    """
```

## Purpose

Query status of background task for progress tracking and completion detection.

## Edge Cases
1. task_id not found → KeyError
2. task_id invalid UUID → ValueError
3. Task just started → state="pending", progress=0
4. Task completed → state="completed", progress=100
5. Task failed → state="failed", error set

## Algorithm
```
1. Validate task_id is valid UUID
2. Lookup task in task_dict
3. If not found, raise KeyError
4. Read task state and metrics
5. Format status dict
6. Return status
```

## Tests (10)
1. test_get_status_pending_task
2. test_get_status_running_task
3. test_get_status_completed_task
4. test_get_status_failed_task
5. test_get_status_cancelled_task
6. test_get_status_not_found_raises
7. test_get_status_invalid_uuid_raises
8. test_get_status_progress_updated
9. test_get_status_timestamps
10. test_get_status_error_message
