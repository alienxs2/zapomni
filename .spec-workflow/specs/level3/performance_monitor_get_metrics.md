# PerformanceMonitor.get_metrics() - Function Specification

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
def get_metrics(
    self,
    operation_name: str
) -> Optional[PerformanceMetrics]:
    """
    Calculate aggregated performance metrics for a specific operation.

    Analyzes all recorded operations matching operation_name and computes
    statistics including min, max, mean, and percentiles (P50, P95, P99).

    Args:
        operation_name: Operation to analyze (e.g., "add_memory")
            - Must match an operation that has been recorded
            - Case-sensitive

    Returns:
        Optional[PerformanceMetrics]: Metrics object or None if no data
            - count: Number of operations recorded
            - min_ms: Fastest execution time
            - max_ms: Slowest execution time
            - mean_ms: Average execution time
            - p50_ms: Median (50th percentile)
            - p95_ms: 95th percentile
            - p99_ms: 99th percentile
            - throughput_ops_per_sec: Operations per second

    Example:
        >>> monitor = PerformanceMonitor()
        >>> # ... record many operations ...
        >>> metrics = monitor.get_metrics("add_memory")
        >>> if metrics:
        ...     print(f"P95 latency: {metrics.p95_ms}ms")
        ...     print(f"Throughput: {metrics.throughput_ops_per_sec} ops/sec")
    """
```

## Algorithm

```
FUNCTION get_metrics(operation_name: str) -> Optional[PerformanceMetrics]:
    # Filter operations by name
    matching_ops = [op FOR op IN self._operations IF op.operation_name == operation_name]

    IF len(matching_ops) == 0:
        RETURN None

    # Extract durations
    durations = [op.duration_ms FOR op IN matching_ops]

    # Calculate statistics
    count = len(durations)
    min_ms = min(durations)
    max_ms = max(durations)
    mean_ms = sum(durations) / count

    # Calculate percentiles
    sorted_durations = sorted(durations)
    p50_ms = sorted_durations[count // 2]
    p95_ms = sorted_durations[int(count * 0.95)]
    p99_ms = sorted_durations[int(count * 0.99)]

    # Calculate throughput (ops/sec over measurement window)
    first_timestamp = matching_ops[0].timestamp
    last_timestamp = matching_ops[-1].timestamp
    time_window_sec = last_timestamp - first_timestamp

    IF time_window_sec > 0:
        throughput = count / time_window_sec
    ELSE:
        throughput = 0.0

    RETURN PerformanceMetrics(
        operation_name=operation_name,
        count=count,
        min_ms=min_ms,
        max_ms=max_ms,
        mean_ms=mean_ms,
        p50_ms=p50_ms,
        p95_ms=p95_ms,
        p99_ms=p99_ms,
        throughput_ops_per_sec=throughput
    )
END
```

## Edge Cases

1. **No operations recorded:** Return None
2. **Only 1 operation:** All percentiles = same value
3. **< 100 operations:** Percentile approximations
4. **All operations same duration:** Min = Max = Mean = P50 = P95 = P99
5. **Unknown operation_name:** Return None

## Test Scenarios

1. test_get_metrics_success
2. test_get_metrics_no_data_returns_none
3. test_get_metrics_single_operation
4. test_get_metrics_calculates_percentiles
5. test_get_metrics_calculates_throughput
6. test_get_metrics_unknown_operation_returns_none
7. test_get_metrics_percentiles_ordered (p50 <= p95 <= p99)

## Performance

- Calculation time: O(N log N) where N = operations for that name
- Typical: < 10ms for 10,000 operations
- Memory: Temporary sorted copy of durations

---

**Estimated Implementation:** 1 hour | **LoC:** ~50 | **Test File:** `test_performance_monitor_get_metrics.py`
