# PerformanceMonitor - Component Specification

**Level:** 2 (Component)
**Module:** zapomni_core
**Author:** Goncharenko Anton aka alienxs2
**Status:** Draft
**Version:** 1.0
**Created:** 2025-11-23

---

## Overview

### Purpose

The PerformanceMonitor component provides **real-time performance tracking and analysis** for the Zapomni memory system. It monitors key operational metrics including latency, throughput, and resource usage to ensure the system meets its performance targets and to identify bottlenecks.

This component acts as the **observability layer** for Zapomni Core, collecting metrics from all critical operations (add_memory, search_memory, embedding generation, chunking) and exposing aggregated statistics for debugging, optimization, and health monitoring.

### Responsibilities

1. **Operation Latency Tracking:** Record execution time for all critical operations with P50/P95/P99 percentile calculations
2. **Throughput Monitoring:** Track operations per second for add_memory and search_memory workflows
3. **Resource Usage Monitoring:** Capture memory and CPU usage via psutil integration
4. **In-Memory Metrics Storage:** Maintain circular buffer of last 10,000 operations for efficient analysis
5. **Statistics Aggregation:** Calculate summary statistics (min, max, mean, percentiles) for health reporting
6. **Metrics Reset:** Provide capability to clear all metrics for testing or fresh monitoring periods

### Position in Module

The PerformanceMonitor is a **utility component** used by the MemoryProcessor and other core services:

```
┌─────────────────────────────────────────────┐
│         zapomni_core Module                 │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────────────┐                       │
│  │ MemoryProcessor  │                       │
│  │ (Main Service)   │                       │
│  └────────┬─────────┘                       │
│           │ uses                            │
│           ↓                                 │
│  ┌──────────────────┐                       │
│  │PerformanceMonitor│ ← THIS COMPONENT     │
│  │ (Metrics)        │                       │
│  └──────────────────┘                       │
│           ↓ uses                            │
│  ┌──────────────────┐                       │
│  │    psutil        │                       │
│  │  (System Metrics)│                       │
│  └──────────────────┘                       │
└─────────────────────────────────────────────┘
```

**Relationships:**
- **Used by:** MemoryProcessor, OllamaEmbedder, SemanticChunker, SearchEngines
- **Uses:** psutil (resource monitoring), statistics (percentile calculations)
- **Called at:** Start and end of every critical operation

---

## Class Definition

### Class Diagram

```
┌─────────────────────────────────────────┐
│       PerformanceMonitor                │
├─────────────────────────────────────────┤
│ - _operations: deque                    │
│ - _resource_samples: deque              │
│ - _max_operations: int                  │
│ - _start_time: float                    │
│ - _process: psutil.Process              │
├─────────────────────────────────────────┤
│ + record_operation(name, duration)      │
│ + get_metrics(operation_name)           │
│ + record_resource_usage()               │
│ + get_stats_summary()                   │
│ + reset_metrics()                       │
│ - _calculate_percentiles(durations)     │
│ - _get_current_resources()              │
└─────────────────────────────────────────┘
```

### Full Class Signature

```python
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from statistics import mean, median, quantiles
import time
import psutil

@dataclass
class OperationMetric:
    """
    Represents a single operation execution metric.

    Attributes:
        operation_name: Name of the operation (e.g., "add_memory", "search_memory")
        duration_ms: Execution time in milliseconds
        timestamp: Unix timestamp when operation completed
    """
    operation_name: str
    duration_ms: float
    timestamp: float

@dataclass
class ResourceSnapshot:
    """
    Snapshot of system resource usage at a point in time.

    Attributes:
        timestamp: Unix timestamp of snapshot
        memory_mb: Memory usage in megabytes
        cpu_percent: CPU usage percentage (0-100)
    """
    timestamp: float
    memory_mb: float
    cpu_percent: float

@dataclass
class PerformanceMetrics:
    """
    Aggregated performance metrics for a specific operation.

    Attributes:
        operation_name: Operation being measured
        count: Total number of operations recorded
        min_ms: Minimum latency
        max_ms: Maximum latency
        mean_ms: Average latency
        p50_ms: 50th percentile (median)
        p95_ms: 95th percentile
        p99_ms: 99th percentile
        throughput_ops_per_sec: Operations per second (calculated over measurement window)
    """
    operation_name: str
    count: int
    min_ms: float
    max_ms: float
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    throughput_ops_per_sec: float

@dataclass
class SystemStats:
    """
    Overall system health and performance statistics.

    Attributes:
        uptime_seconds: Time since monitor was initialized
        total_operations: Total operations across all types
        avg_memory_mb: Average memory usage
        avg_cpu_percent: Average CPU usage
        operation_metrics: Dict mapping operation names to their PerformanceMetrics
    """
    uptime_seconds: float
    total_operations: int
    avg_memory_mb: float
    avg_cpu_percent: float
    operation_metrics: Dict[str, PerformanceMetrics]


class PerformanceMonitor:
    """
    Tracks performance metrics for Zapomni operations.

    Monitors latency (P50/P95/P99), throughput (ops/sec), and resource usage
    (memory/CPU) for all critical operations. Maintains in-memory circular buffer
    of last 10K operations for efficient real-time analysis.

    Thread-safe: No (designed for single-threaded asyncio usage in MemoryProcessor)

    Attributes:
        _operations: Circular buffer of operation metrics (max 10K)
        _resource_samples: Circular buffer of resource snapshots (max 1K)
        _max_operations: Maximum operations to store (default: 10000)
        _start_time: Monitor initialization timestamp
        _process: psutil Process handle for resource monitoring

    Example:
        ```python
        # Initialize monitor
        monitor = PerformanceMonitor(max_operations=10000)

        # Record an operation
        start = time.time()
        # ... do work ...
        duration_ms = (time.time() - start) * 1000
        monitor.record_operation("add_memory", duration_ms)

        # Get metrics for specific operation
        metrics = monitor.get_metrics("add_memory")
        print(f"P95 latency: {metrics.p95_ms}ms")

        # Record resource usage
        monitor.record_resource_usage()

        # Get overall system stats
        stats = monitor.get_stats_summary()
        print(f"Total operations: {stats.total_operations}")
        print(f"Avg memory: {stats.avg_memory_mb}MB")
        ```
    """

    def __init__(self, max_operations: int = 10000) -> None:
        """
        Initialize PerformanceMonitor with circular buffer.

        Args:
            max_operations: Maximum number of operations to store in buffer.
                Older operations are automatically evicted when buffer is full.
                Default: 10000 (sufficient for ~1 hour of heavy usage)

        Raises:
            ValueError: If max_operations < 100 (too small to be useful)

        Example:
            ```python
            # Default buffer size
            monitor = PerformanceMonitor()

            # Custom buffer size (larger for longer analysis windows)
            monitor = PerformanceMonitor(max_operations=50000)
            ```
        """

    def record_operation(
        self,
        operation_name: str,
        duration_ms: float
    ) -> None:
        """
        Record latency for a completed operation.

        Stores operation in circular buffer. If buffer is full (10K operations),
        oldest operation is automatically evicted (FIFO).

        Args:
            operation_name: Name of the operation (e.g., "add_memory", "search_memory",
                "embed_text", "chunk_text"). Use consistent naming for accurate aggregation.
            duration_ms: Execution time in milliseconds. Must be positive.

        Raises:
            ValueError: If operation_name is empty or duration_ms is negative

        Side Effects:
            - Appends OperationMetric to _operations deque
            - Evicts oldest metric if buffer is full

        Example:
            ```python
            import time

            monitor = PerformanceMonitor()

            # Measure add_memory operation
            start = time.time()
            await processor.add_memory(text="...", metadata={})
            duration_ms = (time.time() - start) * 1000
            monitor.record_operation("add_memory", duration_ms)

            # Measure search operation
            start = time.time()
            results = await processor.search_memory(query="...")
            duration_ms = (time.time() - start) * 1000
            monitor.record_operation("search_memory", duration_ms)
            ```
        """

    def get_metrics(self, operation_name: str) -> Optional[PerformanceMetrics]:
        """
        Get aggregated performance metrics for a specific operation.

        Calculates percentiles (P50/P95/P99), min/max/mean latency, and throughput
        from all recorded instances of the operation.

        Args:
            operation_name: Name of the operation to analyze (must match name used
                in record_operation calls)

        Returns:
            PerformanceMetrics object with aggregated statistics, or None if no
            operations with that name have been recorded.

        Raises:
            ValueError: If operation_name is empty

        Performance:
            - Time complexity: O(n log n) where n = count of matching operations
              (due to sorting for percentile calculation)
            - Space complexity: O(n) for temporary duration list
            - Typical execution: < 10ms for 10K operations

        Example:
            ```python
            # Record some operations first
            for _ in range(100):
                monitor.record_operation("search_memory", random.uniform(50, 500))

            # Get aggregated metrics
            metrics = monitor.get_metrics("search_memory")

            if metrics:
                print(f"Operation: {metrics.operation_name}")
                print(f"Count: {metrics.count}")
                print(f"P50 latency: {metrics.p50_ms:.2f}ms")
                print(f"P95 latency: {metrics.p95_ms:.2f}ms")
                print(f"P99 latency: {metrics.p99_ms:.2f}ms")
                print(f"Throughput: {metrics.throughput_ops_per_sec:.2f} ops/sec")
            else:
                print("No metrics found for operation")
            ```
        """

    def record_resource_usage(self) -> None:
        """
        Record current system resource usage (memory and CPU).

        Captures snapshot of memory consumption (RSS) and CPU utilization via psutil.
        Stores in circular buffer of last 1,000 samples for trend analysis.

        Raises:
            RuntimeError: If psutil fails to access process information
                (rare, usually indicates permission issues)

        Side Effects:
            - Appends ResourceSnapshot to _resource_samples deque
            - Evicts oldest sample if buffer is full (> 1000 samples)

        Performance:
            - Execution time: < 5ms (fast, psutil is efficient)
            - Recommended frequency: Every 1-10 seconds (not per-operation)

        Example:
            ```python
            import asyncio

            monitor = PerformanceMonitor()

            # Record resource usage periodically in background task
            async def monitor_resources():
                while True:
                    monitor.record_resource_usage()
                    await asyncio.sleep(5)  # Sample every 5 seconds

            # Start background monitoring
            asyncio.create_task(monitor_resources())

            # Later, analyze resource trends
            stats = monitor.get_stats_summary()
            print(f"Avg memory: {stats.avg_memory_mb:.2f}MB")
            print(f"Avg CPU: {stats.avg_cpu_percent:.1f}%")
            ```
        """

    def get_stats_summary(self) -> SystemStats:
        """
        Get comprehensive system health and performance statistics.

        Aggregates all recorded metrics into a single summary including:
        - Uptime (time since monitor initialization)
        - Total operation count across all types
        - Average memory and CPU usage
        - Per-operation performance metrics (P50/P95/P99 latencies)

        Returns:
            SystemStats object with complete system health snapshot

        Performance:
            - Time complexity: O(n * m) where n = unique operations, m = ops per type
            - Typical execution: < 50ms for 10K total operations

        Example:
            ```python
            monitor = PerformanceMonitor()

            # ... record operations and resources over time ...

            # Get comprehensive summary
            stats = monitor.get_stats_summary()

            print(f"System uptime: {stats.uptime_seconds:.0f}s")
            print(f"Total operations: {stats.total_operations}")
            print(f"Avg memory: {stats.avg_memory_mb:.2f}MB")
            print(f"Avg CPU: {stats.avg_cpu_percent:.1f}%")

            print("\nPer-Operation Metrics:")
            for op_name, metrics in stats.operation_metrics.items():
                print(f"  {op_name}:")
                print(f"    Count: {metrics.count}")
                print(f"    P50: {metrics.p50_ms:.2f}ms")
                print(f"    P95: {metrics.p95_ms:.2f}ms")
                print(f"    P99: {metrics.p99_ms:.2f}ms")
                print(f"    Throughput: {metrics.throughput_ops_per_sec:.2f} ops/sec")
            ```
        """

    def reset_metrics(self) -> None:
        """
        Clear all recorded metrics and reset monitor to initial state.

        Removes all operation metrics and resource snapshots from buffers.
        Resets start_time to current time. Useful for:
        - Starting fresh monitoring window
        - Testing (clear between test runs)
        - Memory management (free up buffer space)

        Side Effects:
            - Clears _operations deque (all operation metrics lost)
            - Clears _resource_samples deque (all resource snapshots lost)
            - Resets _start_time to time.time()

        Warning:
            This operation is irreversible. All metrics are permanently deleted.

        Example:
            ```python
            monitor = PerformanceMonitor()

            # Record some test data
            for i in range(100):
                monitor.record_operation("test_op", 100.0)

            # Verify data exists
            metrics = monitor.get_metrics("test_op")
            assert metrics.count == 100

            # Reset for clean slate
            monitor.reset_metrics()

            # Verify reset
            metrics = monitor.get_metrics("test_op")
            assert metrics is None  # No data

            stats = monitor.get_stats_summary()
            assert stats.total_operations == 0
            assert stats.uptime_seconds < 1  # Just reset
            ```
        """

    def _calculate_percentiles(
        self,
        durations: List[float]
    ) -> Dict[str, float]:
        """
        Calculate P50/P95/P99 percentiles from duration list.

        Private helper method using statistics.quantiles() for accurate percentile
        calculation. Uses exclusive method (type 7) which is standard in NumPy/Pandas.

        Args:
            durations: List of duration values (in milliseconds). Must have at least
                2 elements for percentile calculation.

        Returns:
            Dictionary with keys:
            - "p50": 50th percentile (median)
            - "p95": 95th percentile
            - "p99": 99th percentile

            If fewer than 2 durations, returns {"p50": dur, "p95": dur, "p99": dur}

        Algorithm:
            1. Sort durations (O(n log n))
            2. Use statistics.quantiles() with n=100 (percentiles 1-99)
            3. Extract P50 (index 49), P95 (index 94), P99 (index 98)

        Example:
            ```python
            # Private method - called internally by get_metrics()
            durations = [10.5, 20.3, 30.1, 40.8, 50.2, 100.5, 200.7, 500.9]
            percentiles = self._calculate_percentiles(durations)
            # Returns: {"p50": ~35.45, "p95": ~450.8, "p99": ~490.86}
            ```
        """

    def _get_current_resources(self) -> ResourceSnapshot:
        """
        Get current memory and CPU usage snapshot.

        Private helper method that wraps psutil calls with error handling.

        Returns:
            ResourceSnapshot with current timestamp, memory (MB), and CPU (%)

        Raises:
            RuntimeError: If psutil fails (permission denied, process not found)

        Implementation:
            - Memory: self._process.memory_info().rss / (1024 * 1024)  # Bytes to MB
            - CPU: self._process.cpu_percent(interval=0.1)  # 100ms sample

        Example:
            ```python
            # Private method - called internally by record_resource_usage()
            snapshot = self._get_current_resources()
            # Returns: ResourceSnapshot(timestamp=..., memory_mb=234.5, cpu_percent=12.3)
            ```
        """
```

---

## Dependencies

### Component Dependencies

**None** - PerformanceMonitor is a standalone utility component with no dependencies on other Zapomni components.

### External Libraries

**Required:**
- `psutil>=5.9.0` - System and process monitoring (memory, CPU usage)
  - Reason: Industry-standard, cross-platform, efficient C-based implementation
  - Alternatives considered: Native os/resource module (Linux-only, limited metrics)

**Standard Library:**
- `collections.deque` - Circular buffer implementation (O(1) append/pop)
- `statistics` - Percentile calculations (quantiles, mean, median)
- `time` - Timestamp generation
- `dataclasses` - Data model definitions
- `typing` - Type hints (Dict, List, Optional, Any)

### Dependency Injection

**No dependency injection** - PerformanceMonitor is initialized directly with configuration:

```python
# Instantiated by MemoryProcessor
monitor = PerformanceMonitor(max_operations=10000)
```

**Optional configuration:**
- `max_operations`: Buffer size (default: 10000)

---

## State Management

### Attributes

**Private Attributes:**

1. **`_operations: deque[OperationMetric]`**
   - Purpose: Circular buffer storing last N operation metrics
   - Lifetime: Persists until reset_metrics() called or object destroyed
   - Max size: Configurable via max_operations (default: 10000)
   - Thread safety: Not thread-safe (single-threaded asyncio usage)

2. **`_resource_samples: deque[ResourceSnapshot]`**
   - Purpose: Circular buffer storing resource usage snapshots
   - Lifetime: Persists until reset_metrics() called
   - Max size: 1000 samples (hardcoded, ~1 hour at 5s intervals)

3. **`_max_operations: int`**
   - Purpose: Maximum operations to store before eviction
   - Lifetime: Immutable after initialization
   - Typical value: 10000

4. **`_start_time: float`**
   - Purpose: Monitor initialization timestamp (for uptime calculation)
   - Lifetime: Reset on reset_metrics() call
   - Format: Unix timestamp (time.time())

5. **`_process: psutil.Process`**
   - Purpose: Handle to current process for resource monitoring
   - Lifetime: Persists for object lifetime
   - Initialized: In __init__() as psutil.Process()

### State Transitions

```
┌─────────────────┐
│  INITIALIZED    │  (Empty buffers, start_time set)
└────────┬────────┘
         │
         │ record_operation() called
         ↓
┌─────────────────┐
│  COLLECTING     │  (Buffers filling with metrics)
└────────┬────────┘
         │
         │ record_resource_usage() called
         ↓
┌─────────────────┐
│  MONITORING     │  (Both operation and resource data collected)
└────────┬────────┘
         │
         │ reset_metrics() called
         ↓
┌─────────────────┐
│  RESET          │  (Back to INITIALIZED state)
└─────────────────┘

Note: get_metrics() and get_stats_summary() are read-only operations
      and do not change state.
```

### Thread Safety

**Not thread-safe** - Designed for single-threaded asyncio usage:
- All methods are synchronous (no async/await)
- Called from MemoryProcessor (single asyncio event loop)
- No locks required (no concurrent access)

**If multi-threaded usage is required (future):**
- Add threading.Lock around _operations and _resource_samples modifications
- Use thread-safe collections.deque (already thread-safe for append/pop from opposite ends)

---

## Public Methods (Detailed)

### Method 1: `record_operation`

**Signature:**
```python
def record_operation(self, operation_name: str, duration_ms: float) -> None
```

**Purpose:** Record latency for a completed operation in the circular buffer.

**Parameters:**

- `operation_name`: str
  - Description: Name of the operation being measured
  - Constraints: Non-empty, alphanumeric + underscores, max 100 chars
  - Recommended names: "add_memory", "search_memory", "embed_text", "chunk_text", "extract_entities"
  - Example: "add_memory"

- `duration_ms`: float
  - Description: Execution time in milliseconds
  - Constraints: Must be >= 0 (negative durations rejected)
  - Typical range: 1-2000ms for Zapomni operations
  - Example: 123.45

**Returns:** None (side effect: appends to _operations buffer)

**Raises:**
- `ValueError`: If operation_name is empty or duration_ms < 0

**Preconditions:**
- Monitor must be initialized (__init__ called)

**Postconditions:**
- OperationMetric appended to _operations deque
- If buffer full, oldest metric evicted (FIFO)
- Total operation count increased by 1

**Algorithm Outline:**
```
1. Validate operation_name (non-empty)
2. Validate duration_ms (>= 0)
3. Get current timestamp (time.time())
4. Create OperationMetric(operation_name, duration_ms, timestamp)
5. Append to _operations deque
6. If len(_operations) > _max_operations:
       _operations.popleft()  # Evict oldest
7. Return None
```

**Edge Cases:**

1. Empty operation_name → ValueError
2. Negative duration_ms → ValueError
3. Buffer full (10K operations) → automatic eviction
4. Very large duration (> 1 hour) → accepted (valid for long-running tasks)
5. Duplicate operation names → accepted (aggregated in get_metrics)

**Related Methods:**
- Called by: MemoryProcessor (around all critical operations)
- Calls: None (leaf method)
- Data used by: get_metrics(), get_stats_summary()

---

### Method 2: `get_metrics`

**Signature:**
```python
def get_metrics(self, operation_name: str) -> Optional[PerformanceMetrics]
```

**Purpose:** Calculate and return aggregated performance metrics (P50/P95/P99, throughput) for a specific operation.

**Parameters:**

- `operation_name`: str
  - Description: Operation to analyze (must match name used in record_operation)
  - Constraints: Non-empty
  - Example: "search_memory"

**Returns:**
- Type: `Optional[PerformanceMetrics]`
- If operation found: PerformanceMetrics with complete statistics
- If no operations recorded: None

**Raises:**
- `ValueError`: If operation_name is empty

**Preconditions:**
- At least one operation with operation_name must have been recorded (otherwise returns None)

**Postconditions:**
- No state changes (read-only operation)
- Returns fresh calculation (not cached)

**Algorithm Outline:**
```
1. Validate operation_name (non-empty)
2. Filter _operations for matching operation_name
3. If no matches:
       RETURN None
4. Extract durations list from filtered operations
5. Calculate statistics:
   - count = len(durations)
   - min_ms = min(durations)
   - max_ms = max(durations)
   - mean_ms = statistics.mean(durations)
   - percentiles = _calculate_percentiles(durations)
   - p50_ms = percentiles["p50"]
   - p95_ms = percentiles["p95"]
   - p99_ms = percentiles["p99"]
6. Calculate throughput:
   - time_window_sec = (last_timestamp - first_timestamp)
   - If time_window_sec > 0:
         throughput = count / time_window_sec
     Else:
         throughput = 0.0  # Single operation, no time window
7. Create PerformanceMetrics object
8. RETURN PerformanceMetrics
```

**Edge Cases:**

1. No operations recorded for operation_name → Returns None
2. Single operation recorded → P50/P95/P99 all equal to duration, throughput = 0
3. Two operations recorded → P50 = median, P95/P99 approximated
4. All operations have same duration → P50/P95/P99 identical
5. Operations spanning < 1 second → Throughput may be very high (e.g., 1000 ops/sec)

**Performance:**
- Time complexity: O(n log n) where n = count of matching operations (due to sorting in _calculate_percentiles)
- Space complexity: O(n) for temporary durations list
- Typical execution: < 10ms for 10K operations

**Related Methods:**
- Called by: get_stats_summary(), external monitoring tools
- Calls: _calculate_percentiles()

---

### Method 3: `record_resource_usage`

**Signature:**
```python
def record_resource_usage(self) -> None
```

**Purpose:** Capture and store current system resource usage (memory and CPU).

**Parameters:** None

**Returns:** None (side effect: appends to _resource_samples buffer)

**Raises:**
- `RuntimeError`: If psutil fails to access process information (rare, permission issue)

**Preconditions:**
- Monitor initialized (psutil.Process created)

**Postconditions:**
- ResourceSnapshot appended to _resource_samples deque
- If buffer full (> 1000 samples), oldest evicted

**Algorithm Outline:**
```
1. Try:
       snapshot = _get_current_resources()
   Catch psutil.Error:
       Raise RuntimeError("Failed to access process info")
2. Append snapshot to _resource_samples deque
3. If len(_resource_samples) > 1000:
       _resource_samples.popleft()  # Evict oldest
4. Return None
```

**Edge Cases:**

1. psutil permission denied → RuntimeError (caller should handle)
2. Process terminated (zombie) → RuntimeError
3. Buffer full → automatic eviction (oldest sample removed)
4. CPU spike (100%) → Recorded accurately
5. Memory spike (OOM condition) → Recorded (may be last sample before crash)

**Performance:**
- Execution time: < 5ms (psutil is fast)
- Recommended frequency: Every 1-10 seconds (not per-operation, too expensive)

**Related Methods:**
- Called by: Background monitoring task (periodic)
- Calls: _get_current_resources()
- Data used by: get_stats_summary()

---

### Method 4: `get_stats_summary`

**Signature:**
```python
def get_stats_summary(self) -> SystemStats
```

**Purpose:** Generate comprehensive system health statistics including uptime, operation metrics, and resource usage.

**Parameters:** None

**Returns:**
- Type: `SystemStats`
- Contains: Uptime, total operations, avg memory/CPU, per-operation metrics

**Raises:** None (always returns valid SystemStats, even if no data)

**Preconditions:** None (works with empty buffers)

**Postconditions:** No state changes (read-only)

**Algorithm Outline:**
```
1. Calculate uptime_seconds = time.time() - _start_time
2. total_operations = len(_operations)
3. If _resource_samples is not empty:
       avg_memory_mb = mean([s.memory_mb for s in _resource_samples])
       avg_cpu_percent = mean([s.cpu_percent for s in _resource_samples])
   Else:
       avg_memory_mb = 0.0
       avg_cpu_percent = 0.0
4. operation_metrics = {}
5. For each unique operation_name in _operations:
       metrics = get_metrics(operation_name)
       If metrics is not None:
           operation_metrics[operation_name] = metrics
6. Create SystemStats object
7. RETURN SystemStats
```

**Edge Cases:**

1. No operations recorded → total_operations=0, operation_metrics={}
2. No resource samples → avg_memory_mb=0.0, avg_cpu_percent=0.0
3. Single operation type → operation_metrics has 1 entry
4. Just initialized (< 1s uptime) → uptime_seconds < 1.0
5. Long-running monitor (days) → uptime_seconds > 86400

**Performance:**
- Time complexity: O(n * m) where n = unique operations, m = average ops per type
- Typical execution: < 50ms for 10K total operations

**Related Methods:**
- Called by: MCP get_stats tool, health check endpoints
- Calls: get_metrics() (for each unique operation)

---

### Method 5: `reset_metrics`

**Signature:**
```python
def reset_metrics(self) -> None
```

**Purpose:** Clear all metrics and reset monitor to initial state.

**Parameters:** None

**Returns:** None (side effect: clears buffers, resets start_time)

**Raises:** None

**Preconditions:** None

**Postconditions:**
- _operations deque cleared (len=0)
- _resource_samples deque cleared (len=0)
- _start_time reset to time.time()

**Algorithm Outline:**
```
1. _operations.clear()
2. _resource_samples.clear()
3. _start_time = time.time()
4. Return None
```

**Edge Cases:**

1. Already empty → No-op (harmless)
2. Called during operation recording → Race condition (caller should avoid)
3. Called multiple times in succession → Idempotent (safe)

**Performance:**
- Execution time: < 1ms (deque.clear() is O(n) but fast)

**Related Methods:**
- Called by: Test teardown, manual reset via MCP tool (future)
- Calls: None

---

## Error Handling

### Exceptions Defined

```python
# No custom exceptions - uses standard library exceptions

# ValueError: Input validation failures
#   - Empty operation_name
#   - Negative duration_ms
#   - Invalid max_operations (< 100)

# RuntimeError: System failures
#   - psutil permission denied
#   - Process access failure
```

### Error Recovery

**Transient Errors:**
- `RuntimeError` from psutil → Caller should retry resource monitoring (not critical)

**Permanent Errors:**
- `ValueError` from invalid inputs → Caller should fix input and retry
- Empty operation_name → Fix at call site
- Negative duration → Fix timing logic

**Error Propagation:**
- All exceptions bubble up to caller (no internal catching)
- MemoryProcessor should log errors and continue (don't crash on metrics failure)

---

## Usage Examples

### Basic Usage

```python
from zapomni_core.performance_monitor import PerformanceMonitor
import time

# Initialize monitor
monitor = PerformanceMonitor(max_operations=10000)

# Simulate some operations
for i in range(100):
    # Measure add_memory operation
    start = time.time()
    # ... do work (simulated) ...
    time.sleep(0.05)  # 50ms operation
    duration_ms = (time.time() - start) * 1000
    monitor.record_operation("add_memory", duration_ms)

# Get metrics for add_memory
metrics = monitor.get_metrics("add_memory")
if metrics:
    print(f"add_memory metrics:")
    print(f"  Count: {metrics.count}")
    print(f"  P50: {metrics.p50_ms:.2f}ms")
    print(f"  P95: {metrics.p95_ms:.2f}ms")
    print(f"  P99: {metrics.p99_ms:.2f}ms")
    print(f"  Throughput: {metrics.throughput_ops_per_sec:.2f} ops/sec")
```

### Advanced Usage: Multi-Operation Monitoring

```python
import asyncio
from zapomni_core.performance_monitor import PerformanceMonitor

monitor = PerformanceMonitor()

async def monitor_system_resources():
    """Background task to record resource usage."""
    while True:
        monitor.record_resource_usage()
        await asyncio.sleep(5)  # Sample every 5 seconds

# Start background monitoring
asyncio.create_task(monitor_system_resources())

# Simulate various operations
operations = ["add_memory", "search_memory", "embed_text", "chunk_text"]

for _ in range(500):
    op = random.choice(operations)
    start = time.time()
    # ... do work ...
    await asyncio.sleep(random.uniform(0.01, 0.2))  # 10-200ms
    duration_ms = (time.time() - start) * 1000
    monitor.record_operation(op, duration_ms)

# Get comprehensive stats
stats = monitor.get_stats_summary()

print(f"System Stats:")
print(f"  Uptime: {stats.uptime_seconds:.0f}s")
print(f"  Total operations: {stats.total_operations}")
print(f"  Avg memory: {stats.avg_memory_mb:.2f}MB")
print(f"  Avg CPU: {stats.avg_cpu_percent:.1f}%")

print(f"\nPer-Operation Breakdown:")
for op_name, metrics in stats.operation_metrics.items():
    print(f"  {op_name}:")
    print(f"    Count: {metrics.count}")
    print(f"    P50: {metrics.p50_ms:.2f}ms | P95: {metrics.p95_ms:.2f}ms | P99: {metrics.p99_ms:.2f}ms")
    print(f"    Throughput: {metrics.throughput_ops_per_sec:.2f} ops/sec")
```

### Testing with Reset

```python
import pytest
from zapomni_core.performance_monitor import PerformanceMonitor

def test_performance_monitor_reset():
    """Test that reset clears all metrics."""
    monitor = PerformanceMonitor()

    # Record data
    for i in range(50):
        monitor.record_operation("test_op", 100.0)
    monitor.record_resource_usage()

    # Verify data exists
    metrics = monitor.get_metrics("test_op")
    assert metrics is not None
    assert metrics.count == 50

    stats = monitor.get_stats_summary()
    assert stats.total_operations == 50

    # Reset
    monitor.reset_metrics()

    # Verify reset
    metrics_after = monitor.get_metrics("test_op")
    assert metrics_after is None  # No data

    stats_after = monitor.get_stats_summary()
    assert stats_after.total_operations == 0
    assert stats_after.uptime_seconds < 1  # Freshly reset
```

---

## Testing Approach

### Unit Tests Required

**Initialization:**
- `test_init_default_params()` - Normal initialization with defaults
- `test_init_custom_max_operations()` - Custom buffer size
- `test_init_invalid_max_operations_raises()` - max_operations < 100 → ValueError

**record_operation:**
- `test_record_operation_success()` - Valid operation recorded
- `test_record_operation_empty_name_raises()` - Empty operation_name → ValueError
- `test_record_operation_negative_duration_raises()` - duration_ms < 0 → ValueError
- `test_record_operation_buffer_overflow()` - Buffer full → oldest evicted
- `test_record_operation_multiple_types()` - Different operation names coexist

**get_metrics:**
- `test_get_metrics_success()` - Valid metrics returned
- `test_get_metrics_no_data_returns_none()` - Unknown operation → None
- `test_get_metrics_empty_name_raises()` - Empty operation_name → ValueError
- `test_get_metrics_single_operation()` - Edge case: 1 operation
- `test_get_metrics_percentiles_accuracy()` - Verify P50/P95/P99 calculations
- `test_get_metrics_throughput_calculation()` - Verify ops/sec calculation

**record_resource_usage:**
- `test_record_resource_usage_success()` - Snapshot recorded
- `test_record_resource_usage_buffer_overflow()` - Buffer full → eviction
- `test_record_resource_usage_psutil_failure()` - psutil error → RuntimeError

**get_stats_summary:**
- `test_get_stats_summary_empty()` - No data → valid empty stats
- `test_get_stats_summary_with_data()` - Multiple operations and resources
- `test_get_stats_summary_uptime_calculation()` - Verify uptime accuracy

**reset_metrics:**
- `test_reset_metrics_clears_operations()` - Operations cleared
- `test_reset_metrics_clears_resources()` - Resources cleared
- `test_reset_metrics_resets_start_time()` - Uptime resets
- `test_reset_metrics_idempotent()` - Can be called multiple times

**Private methods:**
- `test_calculate_percentiles_normal()` - Normal distribution
- `test_calculate_percentiles_single_value()` - Edge case: 1 value
- `test_calculate_percentiles_two_values()` - Edge case: 2 values
- `test_get_current_resources_success()` - Valid snapshot

### Mocking Strategy

**Mock psutil:**
```python
@pytest.fixture
def mock_process(mocker):
    """Mock psutil.Process for resource monitoring."""
    mock = mocker.MagicMock()
    mock.memory_info.return_value = mocker.MagicMock(rss=100 * 1024 * 1024)  # 100MB
    mock.cpu_percent.return_value = 15.5  # 15.5% CPU
    mocker.patch('psutil.Process', return_value=mock)
    return mock
```

**Mock time:**
```python
@pytest.fixture
def mock_time(mocker):
    """Mock time.time() for deterministic timestamps."""
    mock_time = mocker.MagicMock()
    mock_time.return_value = 1000.0  # Fixed timestamp
    mocker.patch('time.time', mock_time)
    return mock_time
```

### Integration Tests

**With MemoryProcessor:**
- Test PerformanceMonitor integrated with real MemoryProcessor operations
- Verify metrics match actual operation timing
- Check resource usage trends during heavy load

**Performance Benchmarks:**
- Benchmark get_metrics() with 10K operations → < 10ms
- Benchmark record_operation() → < 0.1ms
- Benchmark get_stats_summary() with 10K ops → < 50ms

---

## Performance Considerations

### Time Complexity

**Operations:**
- `record_operation()`: O(1) - deque append is constant time
- `get_metrics(operation_name)`: O(n log n) where n = count of matching ops (sorting for percentiles)
- `record_resource_usage()`: O(1) - deque append + psutil call
- `get_stats_summary()`: O(n * m) where n = unique operations, m = avg ops per type
- `reset_metrics()`: O(n) where n = buffer size (deque.clear)

### Space Complexity

**Memory Usage:**
- Operations buffer: O(max_operations) - default 10K * ~50 bytes = ~500KB
- Resource samples buffer: O(1000) - 1K * ~24 bytes = ~24KB
- Total: < 1MB for typical configuration

**Optimization Opportunities:**
- Cache percentile calculations (trade-off: stale data vs. speed)
- Use NumPy for faster percentile calculations (trade-off: dependency vs. 2x speed)
- Implement sampling (store every Nth operation) for very high throughput

**Trade-offs:**
- Larger buffer (max_operations > 10K) → More memory, longer percentile calculations
- Smaller buffer (max_operations < 1K) → Less memory, but shorter analysis window
- More frequent resource sampling → Better trends, but more overhead

---

## References

- **Module spec:** [zapomni_core_module.md](/home/dev/zapomni/.spec-workflow/specs/level1/zapomni_core_module.md)
- **Related components:** MemoryProcessor (uses PerformanceMonitor)
- **External docs:**
  - psutil documentation: https://psutil.readthedocs.io/
  - Python statistics module: https://docs.python.org/3/library/statistics.html
  - collections.deque: https://docs.python.org/3/library/collections.html#collections.deque

---

**Document Status:** Draft v1.0
**Created:** 2025-11-23
**Author:** Goncharenko Anton aka alienxs2
**Copyright:** Copyright (c) 2025 Goncharenko Anton aka alienxs2
**License:** MIT License

**Ready for Implementation:** Yes ✅
