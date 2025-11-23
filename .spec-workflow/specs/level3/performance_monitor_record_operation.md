# PerformanceMonitor.record_operation() - Function Specification

**Level:** 3 (Function)
**Component:** PerformanceMonitor
**Module:** zapomni_core
**Author:** Goncharenko Anton aka alienxs2
**Status:** Draft
**Version:** 1.0
**Created:** 2025-11-23

---

## Function Signature

```python
def record_operation(
    self,
    operation_name: str,
    duration_ms: float
) -> None:
    """
    Record execution time for an operation.

    Stores operation metric in circular buffer (last 10,000 operations)
    for performance analysis and statistics calculation. Used by all
    critical operations (add_memory, search_memory, embed_text, etc.)

    Args:
        operation_name: Operation identifier (e.g., "add_memory", "embed_text")
            - Must be non-empty string
            - Recommended: lowercase, underscores
        duration_ms: Execution time in milliseconds
            - Must be >= 0
            - Typical range: 0.1 - 10,000ms

    Returns:
        None: Side effect only (stores metric)

    Raises:
        ValueError: If operation_name empty or duration_ms negative

    Example:
        >>> monitor = PerformanceMonitor()
        >>> start = time.time()
        >>> # ... perform operation ...
        >>> duration = (time.time() - start) * 1000
        >>> monitor.record_operation("add_memory", duration)
    """
```

## Algorithm

```
FUNCTION record_operation(operation_name: str, duration_ms: float) -> None:
    # Validate inputs
    IF NOT operation_name:
        RAISE ValueError("operation_name cannot be empty")
    IF duration_ms < 0:
        RAISE ValueError("duration_ms must be >= 0")

    # Create metric
    metric = OperationMetric(
        operation_name=operation_name,
        duration_ms=duration_ms,
        timestamp=time.time()
    )

    # Add to circular buffer (auto-evicts oldest if full)
    self._operations.append(metric)

    # If buffer full (10,000 items), oldest automatically removed
    IF len(self._operations) > self._max_operations:
        self._operations.popleft()

    # Log if slow operation (> 1000ms)
    IF duration_ms > 1000:
        self._logger.warning(
            "slow_operation",
            operation=operation_name,
            duration_ms=duration_ms
        )
END
```

## Edge Cases

1. **First operation:** Buffer empty → metric added
2. **Buffer full (10,000):** Oldest evicted automatically
3. **Very fast operation (< 0.1ms):** Recorded normally
4. **Very slow operation (> 10s):** Recorded, logged as warning
5. **Same operation_name repeated:** Multiple entries OK
6. **Empty operation_name:** ValueError

## Test Scenarios

1. test_record_operation_success
2. test_record_operation_adds_to_buffer
3. test_record_operation_circular_buffer_eviction
4. test_record_operation_empty_name_raises
5. test_record_operation_negative_duration_raises
6. test_record_operation_logs_slow_operation
7. test_record_operation_timestamp_set

## Performance

- Record time: < 0.01ms (deque append is O(1))
- Memory: ~10MB for 10,000 operations
- Thread-safe: Yes (deque is thread-safe for append/popleft)

---

**Estimated Implementation:** 30 min | **LoC:** ~20 | **Test File:** `test_performance_monitor_record_operation.py`
