# TaskManager.__init__() - Function Specification

**Level:** 3 | **Component:** TaskManager | **Module:** zapomni_core
**Author:** Goncharenko Anton aka alienxs2 | **Status:** Draft | **V:** 1.0

## Signature
```python
def __init__(self, max_workers: int = 5, max_queue_size: int = 100) -> None:
    """Initialize background task manager."""
```

## Parameters
- **max_workers**: Maximum concurrent tasks (default: 5, range: 1-20)
- **max_queue_size**: Maximum queued tasks (default: 100, range: 1-1000)

## Edge Cases
1. max_workers = 0 → ValueError
2. max_workers = 21 → ValueError
3. max_queue_size = 0 → ValueError
4. max_queue_size = 1001 → ValueError
5. max_workers = 1 → Valid (sequential processing)

## Tests (10)
1. test_init_defaults, 2. test_init_custom_params, 3. test_init_zero_workers_raises,
4. test_init_too_many_workers_raises, 5. test_init_zero_queue_raises,
6. test_init_queue_too_large_raises, 7. test_init_single_worker_valid,
8. test_init_creates_executor, 9. test_init_creates_task_dict,
10. test_init_creates_logger
