# TaskManager.cancel_task() - Function Specification

**Level:** 3 | **Component:** TaskManager | **Module:** zapomni_core
**Author:** Goncharenko Anton aka alienxs2 | **Status:** Draft | **V:** 1.0

## Signature
```python
async def cancel_task(self, task_id: str) -> bool:
    """
    Cancel running or pending background task.
    
    Args:
        task_id: UUID of task to cancel
        
    Returns:
        True if task was cancelled
        False if task already completed/failed/cancelled
        
    Raises:
        KeyError: If task_id not found
        ValueError: If task_id invalid UUID
    """
```

## Purpose

Cancel background task to free resources. Best-effort cancellation - running tasks may complete before cancellation takes effect.

## Edge Cases
1. Task not found → KeyError
2. Task already completed → Returns False (cannot cancel)
3. Task already failed → Returns False
4. Task already cancelled → Returns False (idempotent)
5. Task pending → Returns True (cancelled before start)
6. Task running → Returns True (cancel requested, may complete)

## Algorithm
```
1. Validate task_id UUID
2. Lookup task
3. Check current state
4. If completed/failed/cancelled, return False
5. If pending, remove from queue, set cancelled, return True
6. If running, request cancellation, set cancelled, return True
```

## Tests (10)
1. test_cancel_pending_task
2. test_cancel_running_task
3. test_cancel_completed_task_returns_false
4. test_cancel_failed_task_returns_false
5. test_cancel_already_cancelled_returns_false
6. test_cancel_not_found_raises
7. test_cancel_invalid_uuid_raises
8. test_cancel_sets_cancelled_state
9. test_cancel_idempotent
10. test_cancel_removes_from_queue
