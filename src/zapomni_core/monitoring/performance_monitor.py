"""Performance monitoring component for Zapomni Core.

Provides real-time performance tracking and analysis for the Zapomni memory system.
Monitors key operational metrics including latency, throughput, and resource usage.

Author: Implementation based on specification
License: MIT
"""

import time
from collections import deque
from dataclasses import dataclass
from statistics import mean, quantiles
from typing import Dict, List, Optional

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
        if max_operations < 100:
            raise ValueError(f"max_operations must be >= 100, got {max_operations}")

        self._operations: deque[OperationMetric] = deque(maxlen=max_operations)
        self._resource_samples: deque[ResourceSnapshot] = deque(maxlen=1000)
        self._max_operations = max_operations
        self._start_time = time.time()
        self._process = psutil.Process()

    def record_operation(self, operation_name: str, duration_ms: float) -> None:
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
        if not operation_name:
            raise ValueError("operation_name cannot be empty")
        if duration_ms < 0:
            raise ValueError(f"duration_ms must be >= 0, got {duration_ms}")

        metric = OperationMetric(
            operation_name=operation_name, duration_ms=duration_ms, timestamp=time.time()
        )
        self._operations.append(metric)

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
        if not operation_name:
            raise ValueError("operation_name cannot be empty")

        # Filter operations for matching operation_name
        matching_ops = [op for op in self._operations if op.operation_name == operation_name]

        if not matching_ops:
            return None

        # Extract durations
        durations = [op.duration_ms for op in matching_ops]

        # Calculate basic statistics
        count = len(durations)
        min_ms = min(durations)
        max_ms = max(durations)
        mean_ms = mean(durations)

        # Calculate percentiles
        percentiles = self._calculate_percentiles(durations)

        # Calculate throughput
        timestamps = [op.timestamp for op in matching_ops]
        first_timestamp = min(timestamps)
        last_timestamp = max(timestamps)
        time_window_sec = last_timestamp - first_timestamp

        if time_window_sec > 0:
            throughput_ops_per_sec = count / time_window_sec
        else:
            throughput_ops_per_sec = 0.0

        return PerformanceMetrics(
            operation_name=operation_name,
            count=count,
            min_ms=min_ms,
            max_ms=max_ms,
            mean_ms=mean_ms,
            p50_ms=percentiles["p50"],
            p95_ms=percentiles["p95"],
            p99_ms=percentiles["p99"],
            throughput_ops_per_sec=throughput_ops_per_sec,
        )

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
        try:
            snapshot = self._get_current_resources()
            self._resource_samples.append(snapshot)
        except (psutil.Error, Exception) as e:
            raise RuntimeError(f"Failed to access process information: {e}")

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
        # Calculate uptime
        uptime_seconds = time.time() - self._start_time

        # Calculate total operations
        total_operations = len(self._operations)

        # Calculate average resource usage
        if self._resource_samples:
            avg_memory_mb = mean([s.memory_mb for s in self._resource_samples])
            avg_cpu_percent = mean([s.cpu_percent for s in self._resource_samples])
        else:
            avg_memory_mb = 0.0
            avg_cpu_percent = 0.0

        # Get per-operation metrics
        operation_metrics: Dict[str, PerformanceMetrics] = {}
        unique_operations = set(op.operation_name for op in self._operations)

        for op_name in unique_operations:
            metrics = self.get_metrics(op_name)
            if metrics is not None:
                operation_metrics[op_name] = metrics

        return SystemStats(
            uptime_seconds=uptime_seconds,
            total_operations=total_operations,
            avg_memory_mb=avg_memory_mb,
            avg_cpu_percent=avg_cpu_percent,
            operation_metrics=operation_metrics,
        )

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
        self._operations.clear()
        self._resource_samples.clear()
        self._start_time = time.time()

    def _calculate_percentiles(self, durations: List[float]) -> Dict[str, float]:
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
        if len(durations) < 2:
            # For single value, all percentiles are that value
            duration = durations[0] if durations else 0.0
            return {"p50": duration, "p95": duration, "p99": duration}

        # Use quantiles to calculate percentiles
        # quantiles() returns n-1 cut points dividing sorted data into n equal groups
        # For n=100, we get 99 percentile values
        sorted_durations = sorted(durations)
        percentile_values = quantiles(sorted_durations, n=100)

        return {
            "p50": percentile_values[49],  # 50th percentile (index 49)
            "p95": percentile_values[94],  # 95th percentile (index 94)
            "p99": percentile_values[98],  # 99th percentile (index 98)
        }

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
        try:
            # Get memory usage in MB
            memory_info = self._process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)

            # Get CPU percentage
            cpu_percent = self._process.cpu_percent(interval=0.1)

            return ResourceSnapshot(
                timestamp=time.time(), memory_mb=memory_mb, cpu_percent=cpu_percent
            )
        except (psutil.Error, Exception) as e:
            raise RuntimeError(f"Failed to get resource information: {e}")
