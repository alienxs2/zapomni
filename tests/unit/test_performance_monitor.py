"""Unit tests for PerformanceMonitor component.

Tests cover:
- Initialization (default and custom parameters)
- Operation recording (valid, invalid, buffer overflow)
- Metrics calculation (percentiles, throughput, statistics)
- Resource monitoring (snapshots, buffer management)
- System stats aggregation
- Reset functionality
- Edge cases and error handling
"""

import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from zapomni_core.monitoring.performance_monitor import (
    OperationMetric,
    PerformanceMetrics,
    PerformanceMonitor,
    ResourceSnapshot,
    SystemStats,
)


class TestPerformanceMonitorInit:
    """Tests for PerformanceMonitor initialization."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        monitor = PerformanceMonitor()
        assert monitor._max_operations == 10000
        assert len(monitor._operations) == 0
        assert len(monitor._resource_samples) == 0
        assert monitor._start_time > 0

    def test_init_custom_max_operations(self):
        """Test initialization with custom max_operations."""
        monitor = PerformanceMonitor(max_operations=5000)
        assert monitor._max_operations == 5000

    def test_init_invalid_max_operations_too_small(self):
        """Test that max_operations < 100 raises ValueError."""
        with pytest.raises(ValueError, match="max_operations must be >= 100"):
            PerformanceMonitor(max_operations=50)

    def test_init_invalid_max_operations_exactly_100(self):
        """Test that max_operations = 100 is valid (minimum)."""
        monitor = PerformanceMonitor(max_operations=100)
        assert monitor._max_operations == 100

    def test_init_psutil_process_created(self):
        """Test that psutil Process is created during init."""
        monitor = PerformanceMonitor()
        assert monitor._process is not None


class TestRecordOperation:
    """Tests for record_operation method."""

    def test_record_operation_success(self):
        """Test successful operation recording."""
        monitor = PerformanceMonitor()
        monitor.record_operation("test_op", 100.5)

        assert len(monitor._operations) == 1
        op = list(monitor._operations)[0]
        assert op.operation_name == "test_op"
        assert op.duration_ms == 100.5
        assert op.timestamp > 0

    def test_record_operation_empty_name_raises(self):
        """Test that empty operation_name raises ValueError."""
        monitor = PerformanceMonitor()
        with pytest.raises(ValueError, match="operation_name cannot be empty"):
            monitor.record_operation("", 100.0)

    def test_record_operation_negative_duration_raises(self):
        """Test that negative duration_ms raises ValueError."""
        monitor = PerformanceMonitor()
        with pytest.raises(ValueError, match="duration_ms must be >= 0"):
            monitor.record_operation("test_op", -10.0)

    def test_record_operation_zero_duration(self):
        """Test that zero duration is valid."""
        monitor = PerformanceMonitor()
        monitor.record_operation("test_op", 0.0)
        assert len(monitor._operations) == 1

    def test_record_operation_multiple_calls(self):
        """Test recording multiple operations."""
        monitor = PerformanceMonitor()
        for i in range(100):
            monitor.record_operation("test_op", float(i))

        assert len(monitor._operations) == 100

    def test_record_operation_buffer_overflow(self):
        """Test that buffer evicts oldest operation when full."""
        monitor = PerformanceMonitor(max_operations=100)

        # Add more than max_operations
        for i in range(150):
            monitor.record_operation("test_op", float(i))

        # Only 100 most recent operations should be kept
        assert len(monitor._operations) == 100

        # Verify oldest operations (0-49) are evicted, newest (50-149) are kept
        durations = [op.duration_ms for op in monitor._operations]
        assert durations == [float(i) for i in range(50, 150)]

    def test_record_operation_multiple_operation_types(self):
        """Test recording different operation types."""
        monitor = PerformanceMonitor()
        monitor.record_operation("add_memory", 50.0)
        monitor.record_operation("search_memory", 100.0)
        monitor.record_operation("embed_text", 75.0)

        assert len(monitor._operations) == 3
        op_names = {op.operation_name for op in monitor._operations}
        assert op_names == {"add_memory", "search_memory", "embed_text"}

    def test_record_operation_large_duration(self):
        """Test recording very large duration (e.g., long-running operation)."""
        monitor = PerformanceMonitor()
        large_duration = 60000.0  # 60 seconds
        monitor.record_operation("long_op", large_duration)

        assert len(monitor._operations) == 1
        op = list(monitor._operations)[0]
        assert op.duration_ms == large_duration


class TestGetMetrics:
    """Tests for get_metrics method."""

    def test_get_metrics_success(self):
        """Test getting metrics for recorded operation."""
        monitor = PerformanceMonitor()
        for i in range(10):
            monitor.record_operation("test_op", float(i * 10))

        metrics = monitor.get_metrics("test_op")

        assert metrics is not None
        assert metrics.operation_name == "test_op"
        assert metrics.count == 10
        assert metrics.min_ms == 0.0
        assert metrics.max_ms == 90.0
        assert 40.0 < metrics.mean_ms < 50.0  # Average of 0,10,20,...,90 is 45

    def test_get_metrics_no_data_returns_none(self):
        """Test that getting metrics for unknown operation returns None."""
        monitor = PerformanceMonitor()
        metrics = monitor.get_metrics("nonexistent")
        assert metrics is None

    def test_get_metrics_empty_name_raises(self):
        """Test that empty operation_name raises ValueError."""
        monitor = PerformanceMonitor()
        with pytest.raises(ValueError, match="operation_name cannot be empty"):
            monitor.get_metrics("")

    def test_get_metrics_single_operation(self):
        """Test metrics for single operation."""
        monitor = PerformanceMonitor()
        monitor.record_operation("test_op", 100.0)

        metrics = monitor.get_metrics("test_op")

        assert metrics.count == 1
        assert metrics.min_ms == 100.0
        assert metrics.max_ms == 100.0
        assert metrics.mean_ms == 100.0
        assert metrics.p50_ms == 100.0
        assert metrics.p95_ms == 100.0
        assert metrics.p99_ms == 100.0
        assert metrics.throughput_ops_per_sec == 0.0  # No time window for single op

    def test_get_metrics_percentiles_accuracy(self):
        """Test that percentiles are calculated accurately."""
        monitor = PerformanceMonitor()
        # Record durations: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
        for i in range(1, 11):
            monitor.record_operation("test_op", float(i * 10))

        metrics = monitor.get_metrics("test_op")

        # P50 should be around 50-55 (median)
        assert 45 < metrics.p50_ms < 60
        # P95 should be high (around 95+)
        assert metrics.p95_ms > 85
        # P99 should be highest
        assert metrics.p99_ms > metrics.p95_ms

    def test_get_metrics_throughput_calculation(self):
        """Test that throughput is calculated correctly."""
        monitor = PerformanceMonitor()

        # Record operations with controlled timestamps
        start_time = time.time()

        with patch("time.time") as mock_time:
            mock_time.return_value = start_time
            monitor.record_operation("test_op", 10.0)

            # Advance time by 10 seconds
            mock_time.return_value = start_time + 10
            for i in range(99):
                monitor.record_operation("test_op", 10.0)

        metrics = monitor.get_metrics("test_op")

        # 100 operations over 10 seconds = 10 ops/sec
        assert 9.5 < metrics.throughput_ops_per_sec < 10.5

    def test_get_metrics_filters_by_operation_name(self):
        """Test that metrics only include specified operation."""
        monitor = PerformanceMonitor()

        # Record different operation types
        for i in range(5):
            monitor.record_operation("op_a", 100.0)
            monitor.record_operation("op_b", 200.0)

        metrics_a = monitor.get_metrics("op_a")
        metrics_b = monitor.get_metrics("op_b")

        assert metrics_a.count == 5
        assert metrics_a.mean_ms == 100.0
        assert metrics_b.count == 5
        assert metrics_b.mean_ms == 200.0

    def test_get_metrics_with_variable_durations(self):
        """Test metrics with variable duration values."""
        monitor = PerformanceMonitor()

        # Record variable durations
        durations = [10.0, 15.0, 20.0, 100.0, 200.0]
        for duration in durations:
            monitor.record_operation("test_op", duration)

        metrics = monitor.get_metrics("test_op")

        assert metrics.count == 5
        assert metrics.min_ms == 10.0
        assert metrics.max_ms == 200.0
        assert metrics.mean_ms == 69.0  # (10+15+20+100+200)/5
        assert 10 < metrics.p50_ms < 100
        assert metrics.p95_ms > metrics.p50_ms


class TestRecordResourceUsage:
    """Tests for record_resource_usage method."""

    @patch("psutil.Process")
    def test_record_resource_usage_success(self, mock_process_class):
        """Test successful resource usage recording."""
        mock_process = MagicMock()
        mock_process.memory_info.return_value = MagicMock(rss=100 * 1024 * 1024)  # 100MB
        mock_process.cpu_percent.return_value = 15.5
        mock_process_class.return_value = mock_process

        monitor = PerformanceMonitor()
        monitor.record_resource_usage()

        assert len(monitor._resource_samples) == 1
        snapshot = list(monitor._resource_samples)[0]
        assert snapshot.memory_mb == 100.0
        assert snapshot.cpu_percent == 15.5
        assert snapshot.timestamp > 0

    @patch("psutil.Process")
    def test_record_resource_usage_buffer_overflow(self, mock_process_class):
        """Test that resource buffer evicts oldest sample when full."""
        mock_process = MagicMock()
        mock_process.memory_info.return_value = MagicMock(rss=100 * 1024 * 1024)
        mock_process.cpu_percent.return_value = 10.0
        mock_process_class.return_value = mock_process

        monitor = PerformanceMonitor()

        # Add more samples than buffer capacity (1000)
        for i in range(1500):
            monitor.record_resource_usage()

        # Only 1000 most recent samples should be kept
        assert len(monitor._resource_samples) == 1000

    @patch("psutil.Process")
    def test_record_resource_usage_psutil_failure(self, mock_process_class):
        """Test that psutil error raises RuntimeError."""
        mock_process = MagicMock()
        mock_process.memory_info.side_effect = Exception("Permission denied")
        mock_process_class.return_value = mock_process

        monitor = PerformanceMonitor()

        with pytest.raises(RuntimeError, match="Failed to access process information"):
            monitor.record_resource_usage()

    @patch("psutil.Process")
    def test_record_resource_usage_multiple_samples(self, mock_process_class):
        """Test recording multiple resource samples."""
        mock_process = MagicMock()
        mock_process.memory_info.return_value = MagicMock(rss=100 * 1024 * 1024)
        mock_process.cpu_percent.return_value = 10.0
        mock_process_class.return_value = mock_process

        monitor = PerformanceMonitor()

        for i in range(10):
            monitor.record_resource_usage()

        assert len(monitor._resource_samples) == 10


class TestGetStatsSummary:
    """Tests for get_stats_summary method."""

    @patch("psutil.Process")
    def test_get_stats_summary_empty(self, mock_process_class):
        """Test stats summary when no data recorded."""
        mock_process = MagicMock()
        mock_process_class.return_value = mock_process

        monitor = PerformanceMonitor()
        stats = monitor.get_stats_summary()

        assert stats.total_operations == 0
        assert stats.avg_memory_mb == 0.0
        assert stats.avg_cpu_percent == 0.0
        assert len(stats.operation_metrics) == 0
        assert stats.uptime_seconds >= 0

    @patch("psutil.Process")
    def test_get_stats_summary_with_data(self, mock_process_class):
        """Test stats summary with recorded operations and resources."""
        mock_process = MagicMock()
        mock_process.memory_info.return_value = MagicMock(rss=100 * 1024 * 1024)
        mock_process.cpu_percent.return_value = 15.0
        mock_process_class.return_value = mock_process

        monitor = PerformanceMonitor()

        # Record operations
        for i in range(10):
            monitor.record_operation("test_op", 100.0)

        # Record resources
        for i in range(5):
            monitor.record_resource_usage()

        stats = monitor.get_stats_summary()

        assert stats.total_operations == 10
        assert "test_op" in stats.operation_metrics
        assert stats.operation_metrics["test_op"].count == 10
        assert stats.avg_memory_mb == 100.0
        assert stats.avg_cpu_percent == 15.0
        assert stats.uptime_seconds > 0

    @patch("psutil.Process")
    def test_get_stats_summary_uptime_calculation(self, mock_process_class):
        """Test that uptime is calculated correctly."""
        mock_process = MagicMock()
        mock_process_class.return_value = mock_process

        monitor = PerformanceMonitor()

        with patch("time.time") as mock_time:
            start_time = 1000.0
            mock_time.return_value = start_time

            # Simulate advancing time
            monitor._start_time = start_time
            mock_time.return_value = start_time + 5.5

            stats = monitor.get_stats_summary()
            assert 5.4 < stats.uptime_seconds < 5.6

    @patch("psutil.Process")
    def test_get_stats_summary_multiple_operations(self, mock_process_class):
        """Test stats summary with multiple operation types."""
        mock_process = MagicMock()
        mock_process.memory_info.return_value = MagicMock(rss=100 * 1024 * 1024)
        mock_process.cpu_percent.return_value = 10.0
        mock_process_class.return_value = mock_process

        monitor = PerformanceMonitor()

        # Record different operation types
        for i in range(5):
            monitor.record_operation("add_memory", 50.0)
            monitor.record_operation("search_memory", 100.0)
            monitor.record_operation("embed_text", 75.0)

        stats = monitor.get_stats_summary()

        assert stats.total_operations == 15
        assert len(stats.operation_metrics) == 3
        assert "add_memory" in stats.operation_metrics
        assert "search_memory" in stats.operation_metrics
        assert "embed_text" in stats.operation_metrics


class TestResetMetrics:
    """Tests for reset_metrics method."""

    def test_reset_metrics_clears_operations(self):
        """Test that reset clears all operations."""
        monitor = PerformanceMonitor()

        for i in range(50):
            monitor.record_operation("test_op", 100.0)

        assert len(monitor._operations) == 50

        monitor.reset_metrics()

        assert len(monitor._operations) == 0
        assert monitor.get_metrics("test_op") is None

    @patch("psutil.Process")
    def test_reset_metrics_clears_resources(self, mock_process_class):
        """Test that reset clears all resource samples."""
        mock_process = MagicMock()
        mock_process.memory_info.return_value = MagicMock(rss=100 * 1024 * 1024)
        mock_process.cpu_percent.return_value = 10.0
        mock_process_class.return_value = mock_process

        monitor = PerformanceMonitor()

        for i in range(10):
            monitor.record_resource_usage()

        assert len(monitor._resource_samples) == 10

        monitor.reset_metrics()

        assert len(monitor._resource_samples) == 0

    def test_reset_metrics_resets_start_time(self):
        """Test that reset resets start_time."""
        monitor = PerformanceMonitor()
        old_start_time = monitor._start_time

        time.sleep(0.1)  # Wait a bit
        monitor.reset_metrics()

        assert monitor._start_time > old_start_time

    def test_reset_metrics_idempotent(self):
        """Test that reset can be called multiple times safely."""
        monitor = PerformanceMonitor()

        monitor.record_operation("test_op", 100.0)
        monitor.reset_metrics()
        monitor.reset_metrics()  # Should not raise

        assert len(monitor._operations) == 0

    @patch("psutil.Process")
    def test_reset_metrics_stats_after_reset(self, mock_process_class):
        """Test that stats are empty after reset."""
        mock_process = MagicMock()
        mock_process.memory_info.return_value = MagicMock(rss=100 * 1024 * 1024)
        mock_process.cpu_percent.return_value = 10.0
        mock_process_class.return_value = mock_process

        monitor = PerformanceMonitor()

        monitor.record_operation("test_op", 100.0)
        monitor.record_resource_usage()

        monitor.reset_metrics()

        stats = monitor.get_stats_summary()
        assert stats.total_operations == 0
        assert stats.avg_memory_mb == 0.0
        assert stats.avg_cpu_percent == 0.0
        assert len(stats.operation_metrics) == 0


class TestCalculatePercentiles:
    """Tests for _calculate_percentiles private method."""

    def test_calculate_percentiles_normal(self):
        """Test percentile calculation with normal distribution."""
        monitor = PerformanceMonitor()
        durations = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]

        percentiles = monitor._calculate_percentiles(durations)

        assert "p50" in percentiles
        assert "p95" in percentiles
        assert "p99" in percentiles
        assert percentiles["p50"] < percentiles["p95"]
        assert percentiles["p95"] < percentiles["p99"]

    def test_calculate_percentiles_single_value(self):
        """Test percentile calculation with single value."""
        monitor = PerformanceMonitor()
        durations = [50.0]

        percentiles = monitor._calculate_percentiles(durations)

        assert percentiles["p50"] == 50.0
        assert percentiles["p95"] == 50.0
        assert percentiles["p99"] == 50.0

    def test_calculate_percentiles_two_values(self):
        """Test percentile calculation with two values."""
        monitor = PerformanceMonitor()
        durations = [10.0, 100.0]

        percentiles = monitor._calculate_percentiles(durations)

        assert percentiles["p50"] > 10.0
        assert percentiles["p95"] > percentiles["p50"]
        assert percentiles["p99"] >= percentiles["p95"]

    def test_calculate_percentiles_identical_values(self):
        """Test percentile calculation when all values are identical."""
        monitor = PerformanceMonitor()
        durations = [50.0] * 100

        percentiles = monitor._calculate_percentiles(durations)

        assert percentiles["p50"] == 50.0
        assert percentiles["p95"] == 50.0
        assert percentiles["p99"] == 50.0


class TestGetCurrentResources:
    """Tests for _get_current_resources private method."""

    @patch("psutil.Process")
    def test_get_current_resources_success(self, mock_process_class):
        """Test getting resource snapshot."""
        mock_process = MagicMock()
        mock_process.memory_info.return_value = MagicMock(rss=256 * 1024 * 1024)  # 256MB
        mock_process.cpu_percent.return_value = 25.5
        mock_process_class.return_value = mock_process

        monitor = PerformanceMonitor()
        snapshot = monitor._get_current_resources()

        assert isinstance(snapshot, ResourceSnapshot)
        assert snapshot.memory_mb == 256.0
        assert snapshot.cpu_percent == 25.5
        assert snapshot.timestamp > 0

    @patch("psutil.Process")
    def test_get_current_resources_zero_memory(self, mock_process_class):
        """Test getting resource snapshot with zero memory."""
        mock_process = MagicMock()
        mock_process.memory_info.return_value = MagicMock(rss=0)
        mock_process.cpu_percent.return_value = 0.0
        mock_process_class.return_value = mock_process

        monitor = PerformanceMonitor()
        snapshot = monitor._get_current_resources()

        assert snapshot.memory_mb == 0.0
        assert snapshot.cpu_percent == 0.0

    @patch("psutil.Process")
    def test_get_current_resources_psutil_error(self, mock_process_class):
        """Test that psutil error raises RuntimeError."""
        mock_process = MagicMock()
        mock_process.memory_info.side_effect = Exception("Access denied")
        mock_process_class.return_value = mock_process

        monitor = PerformanceMonitor()

        with pytest.raises(RuntimeError, match="Failed to get resource information"):
            monitor._get_current_resources()


class TestDataclasses:
    """Tests for data classes."""

    def test_operation_metric_dataclass(self):
        """Test OperationMetric dataclass."""
        metric = OperationMetric(operation_name="test_op", duration_ms=100.5, timestamp=1000.0)

        assert metric.operation_name == "test_op"
        assert metric.duration_ms == 100.5
        assert metric.timestamp == 1000.0

    def test_resource_snapshot_dataclass(self):
        """Test ResourceSnapshot dataclass."""
        snapshot = ResourceSnapshot(timestamp=1000.0, memory_mb=256.0, cpu_percent=15.5)

        assert snapshot.timestamp == 1000.0
        assert snapshot.memory_mb == 256.0
        assert snapshot.cpu_percent == 15.5

    def test_performance_metrics_dataclass(self):
        """Test PerformanceMetrics dataclass."""
        metrics = PerformanceMetrics(
            operation_name="test_op",
            count=100,
            min_ms=10.0,
            max_ms=500.0,
            mean_ms=150.0,
            p50_ms=100.0,
            p95_ms=400.0,
            p99_ms=480.0,
            throughput_ops_per_sec=10.0,
        )

        assert metrics.operation_name == "test_op"
        assert metrics.count == 100
        assert metrics.min_ms == 10.0
        assert metrics.p99_ms == 480.0

    @patch("psutil.Process")
    def test_system_stats_dataclass(self, mock_process_class):
        """Test SystemStats dataclass."""
        mock_process = MagicMock()
        mock_process_class.return_value = mock_process

        perf_metrics = PerformanceMetrics(
            operation_name="test_op",
            count=100,
            min_ms=10.0,
            max_ms=500.0,
            mean_ms=150.0,
            p50_ms=100.0,
            p95_ms=400.0,
            p99_ms=480.0,
            throughput_ops_per_sec=10.0,
        )

        stats = SystemStats(
            uptime_seconds=60.0,
            total_operations=100,
            avg_memory_mb=256.0,
            avg_cpu_percent=15.0,
            operation_metrics={"test_op": perf_metrics},
        )

        assert stats.uptime_seconds == 60.0
        assert stats.total_operations == 100
        assert "test_op" in stats.operation_metrics


class TestIntegration:
    """Integration tests combining multiple features."""

    @patch("psutil.Process")
    def test_full_monitoring_workflow(self, mock_process_class):
        """Test complete monitoring workflow."""
        mock_process = MagicMock()
        mock_process.memory_info.return_value = MagicMock(rss=100 * 1024 * 1024)
        mock_process.cpu_percent.return_value = 10.0
        mock_process_class.return_value = mock_process

        monitor = PerformanceMonitor()

        # Record operations
        for i in range(50):
            monitor.record_operation("add_memory", 50.0)
            monitor.record_operation("search_memory", 100.0)

        # Record resources
        for i in range(10):
            monitor.record_resource_usage()

        # Get individual metrics
        add_metrics = monitor.get_metrics("add_memory")
        search_metrics = monitor.get_metrics("search_memory")

        assert add_metrics.count == 50
        assert search_metrics.count == 50

        # Get summary
        stats = monitor.get_stats_summary()

        assert stats.total_operations == 100
        assert len(stats.operation_metrics) == 2
        assert stats.avg_memory_mb == 100.0

        # Reset and verify
        monitor.reset_metrics()

        assert monitor.get_metrics("add_memory") is None
        stats_after = monitor.get_stats_summary()
        assert stats_after.total_operations == 0

    @patch("psutil.Process")
    def test_monitoring_with_errors(self, mock_process_class):
        """Test monitoring with various error conditions."""
        mock_process = MagicMock()
        mock_process.memory_info.return_value = MagicMock(rss=100 * 1024 * 1024)
        mock_process.cpu_percent.return_value = 10.0
        mock_process_class.return_value = mock_process

        monitor = PerformanceMonitor()

        # Try various invalid operations
        with pytest.raises(ValueError):
            monitor.record_operation("", 100.0)

        with pytest.raises(ValueError):
            monitor.record_operation("test", -10.0)

        with pytest.raises(ValueError):
            monitor.get_metrics("")

        # Valid operations should still work
        monitor.record_operation("valid_op", 100.0)
        assert monitor.get_metrics("valid_op") is not None
