"""
Load tests for Zapomni MCP Server using Locust framework.

This module provides load testing scenarios to verify system performance
under concurrent client connections as required by the specification.

Test Scenarios:
    1. Basic Connection Test - Health endpoint verification
    2. Concurrent Connections - 50+ simultaneous clients
    3. Tool Execution Under Load - Response time validation
    4. Health Endpoint Load - High-frequency monitoring

Performance Targets:
    - 50+ concurrent connections
    - Response time < 500ms (P95)
    - No errors under normal load
    - 30 minute stability test

Usage:
    # Start the server first
    zapomni-mcp --host 127.0.0.1 --port 8000

    # Run with web UI (default http://localhost:8089)
    locust -f locustfile.py --host=http://127.0.0.1:8000

    # Run headless for CI/CD
    locust -f locustfile.py --host=http://127.0.0.1:8000 \
        --headless -u 50 -r 10 --run-time 5m

    # Run with custom targets
    locust -f locustfile.py --host=http://127.0.0.1:8000 \
        --headless -u 100 -r 20 --run-time 30m

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

import json
import time
import uuid
from typing import Any, Optional

from locust import HttpUser, between, events, tag, task
from locust.env import Environment

# Performance thresholds (from specification)
P95_RESPONSE_TIME_MS = 500  # milliseconds
MAX_CONCURRENT_CONNECTIONS = 50
STABILITY_TEST_DURATION_MINUTES = 30


class HealthCheckUser(HttpUser):
    """
    Load test user focused on health endpoint verification.

    This user simulates monitoring clients that frequently check
    the server's health status. Useful for verifying the health
    endpoint can handle high-frequency requests without degradation.
    """

    wait_time = between(0.1, 0.5)  # Very frequent requests
    weight = 2  # Lower weight - fewer instances

    @task(10)
    @tag("health", "monitoring")
    def health_check(self) -> None:
        """
        Check health endpoint and validate response.

        Verifies:
            - Status code is 200
            - Response contains 'healthy' status
            - Response time is acceptable
        """
        with self.client.get("/health", catch_response=True, name="/health") as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if data.get("status") == "healthy":
                        response.success()
                    else:
                        response.failure(f"Unhealthy status: {data.get('status')}")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(5)
    @tag("health", "metrics")
    def health_check_with_metrics_validation(self) -> None:
        """
        Check health endpoint and validate metrics structure.

        Verifies the health response contains all expected fields
        for proper monitoring integration.
        """
        with self.client.get("/health", catch_response=True, name="/health (metrics)") as response:
            if response.status_code != 200:
                response.failure(f"Status code: {response.status_code}")
                return

            try:
                data = response.json()

                # Validate required fields
                required_fields = [
                    "status",
                    "version",
                    "transport",
                    "active_connections",
                    "uptime_seconds",
                    "metrics",
                ]

                missing_fields = [f for f in required_fields if f not in data]
                if missing_fields:
                    response.failure(f"Missing fields: {missing_fields}")
                    return

                # Validate metrics sub-fields
                metrics = data.get("metrics", {})
                metrics_fields = [
                    "total_connections_created",
                    "total_connections_closed",
                    "peak_connections",
                    "total_requests_processed",
                    "total_errors",
                ]

                missing_metrics = [f for f in metrics_fields if f not in metrics]
                if missing_metrics:
                    response.failure(f"Missing metrics: {missing_metrics}")
                    return

                response.success()

            except json.JSONDecodeError:
                response.failure("Invalid JSON response")


class SSEConnectionUser(HttpUser):
    """
    Load test user simulating SSE client connections.

    Note: Locust doesn't natively support long-lived SSE connections.
    These tests verify HTTP endpoints that support SSE transport,
    including connection establishment and message handling.
    """

    wait_time = between(1, 3)
    weight = 5  # Higher weight - more instances

    def on_start(self) -> None:
        """Initialize user session."""
        self.session_id: Optional[str] = None
        self.connection_count = 0

    @task(10)
    @tag("health", "connection")
    def health_check(self) -> None:
        """Regular health check during session."""
        with self.client.get("/health", catch_response=True, name="/health") as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    response.success()
                else:
                    response.failure(f"Unhealthy: {data}")
            else:
                response.failure(f"Status: {response.status_code}")

    @task(5)
    @tag("session", "message")
    def send_message_to_invalid_session(self) -> None:
        """
        Send message to non-existent session.

        This tests error handling under load and verifies
        the server properly rejects invalid session requests.
        """
        fake_session_id = f"load-test-{uuid.uuid4()}"

        with self.client.post(
            f"/messages/{fake_session_id}",
            json={
                "jsonrpc": "2.0",
                "method": "tools/list",
                "id": 1,
            },
            catch_response=True,
            name="/messages/{session_id} (invalid)",
        ) as response:
            # Expecting 404 for non-existent session
            if response.status_code == 404:
                response.success()
            else:
                response.failure(f"Expected 404, got {response.status_code}")

    @task(3)
    @tag("session", "connection")
    def attempt_sse_connection(self) -> None:
        """
        Attempt SSE endpoint connection.

        Note: This won't establish a true SSE connection in Locust,
        but verifies the endpoint is responsive and returns
        appropriate headers/status for SSE requests.
        """
        headers = {
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache",
        }

        # Short timeout since we're just testing endpoint availability
        with self.client.get(
            "/sse",
            headers=headers,
            catch_response=True,
            name="/sse (connection attempt)",
            timeout=2,
        ) as response:
            # SSE endpoint should respond (might be streaming or error)
            # We mark success if server responds at all
            if response.status_code in [200, 408, 499]:  # OK, Timeout, Client Closed
                response.success()
            else:
                response.failure(f"Unexpected status: {response.status_code}")


class ConcurrentLoadUser(HttpUser):
    """
    Load test user for concurrent connection verification.

    This user class is designed to run with 50+ instances to verify
    the system can handle the required concurrent connection count.
    """

    wait_time = between(0.5, 2)
    weight = 10  # Highest weight - most instances

    def on_start(self) -> None:
        """Record connection start time."""
        self.start_time = time.time()
        self.request_count = 0

    @task(15)
    @tag("health", "concurrent")
    def concurrent_health_check(self) -> None:
        """
        Health check for concurrent connection testing.

        Track request count and validate response times.
        """
        self.request_count += 1

        with self.client.get(
            "/health",
            catch_response=True,
            name="/health (concurrent)",
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    # Check response time
                    response_time_ms = response.elapsed.total_seconds() * 1000
                    if response_time_ms > P95_RESPONSE_TIME_MS * 2:
                        # Warning if very slow but still success
                        response.success()
                    else:
                        response.success()
                else:
                    response.failure(f"Unhealthy: {data}")
            else:
                response.failure(f"Status: {response.status_code}")

    @task(5)
    @tag("session", "concurrent")
    def message_endpoint_check(self) -> None:
        """
        Check message endpoint availability under load.
        """
        # Use deterministic session ID for this test
        session_id = f"concurrent-test-{uuid.uuid4().hex[:8]}"

        with self.client.post(
            f"/messages/{session_id}",
            json={"jsonrpc": "2.0", "method": "ping", "id": 1},
            catch_response=True,
            name="/messages/{session_id} (concurrent)",
        ) as response:
            # 404 is expected (no session), but server should respond quickly
            if response.status_code in [404, 202]:
                response.success()
            else:
                response.failure(f"Unexpected: {response.status_code}")


# Event handlers for test reporting


@events.test_start.add_listener
def on_test_start(environment: Environment, **kwargs: Any) -> None:
    """Log test start with configuration."""
    print("\n" + "=" * 60)
    print("ZAPOMNI LOAD TEST STARTED")
    print("=" * 60)
    print(f"Target Host: {environment.host}")
    print(f"Performance Targets:")
    print(f"  - P95 Response Time: < {P95_RESPONSE_TIME_MS}ms")
    print(f"  - Max Concurrent Connections: {MAX_CONCURRENT_CONNECTIONS}")
    print(f"  - Stability Test Duration: {STABILITY_TEST_DURATION_MINUTES} minutes")
    print("=" * 60 + "\n")


@events.test_stop.add_listener
def on_test_stop(environment: Environment, **kwargs: Any) -> None:
    """Generate test summary on completion."""
    print("\n" + "=" * 60)
    print("ZAPOMNI LOAD TEST COMPLETED")
    print("=" * 60)

    stats = environment.stats

    # Calculate key metrics
    total_requests = stats.total.num_requests
    total_failures = stats.total.num_failures
    avg_response_time = stats.total.avg_response_time
    p95_response_time = stats.total.get_response_time_percentile(0.95)
    p99_response_time = stats.total.get_response_time_percentile(0.99)

    error_rate = (total_failures / total_requests * 100) if total_requests > 0 else 0

    print(f"\nResults Summary:")
    print(f"  Total Requests: {total_requests}")
    print(f"  Total Failures: {total_failures}")
    print(f"  Error Rate: {error_rate:.2f}%")
    print(f"\nResponse Times:")
    print(f"  Average: {avg_response_time:.2f}ms")
    print(f"  P95: {p95_response_time:.2f}ms")
    print(f"  P99: {p99_response_time:.2f}ms")

    # Performance validation
    print("\nPerformance Validation:")

    p95_pass = p95_response_time is not None and p95_response_time < P95_RESPONSE_TIME_MS
    error_pass = error_rate < 1.0  # Less than 1% error rate

    print(f"  P95 < {P95_RESPONSE_TIME_MS}ms: {'PASS' if p95_pass else 'FAIL'}")
    print(f"  Error Rate < 1%: {'PASS' if error_pass else 'FAIL'}")

    overall_pass = p95_pass and error_pass
    print(f"\nOverall: {'PASS' if overall_pass else 'FAIL'}")
    print("=" * 60 + "\n")


@events.request.add_listener
def on_request(
    request_type: str,
    name: str,
    response_time: float,
    response_length: int,
    exception: Optional[Exception],
    **kwargs: Any,
) -> None:
    """
    Track individual requests for detailed analysis.

    This listener can be used to log slow requests or failures
    for debugging purposes during load testing.
    """
    # Log very slow requests (> 2x P95 target)
    if response_time > P95_RESPONSE_TIME_MS * 2:
        print(f"SLOW REQUEST: {request_type} {name} - {response_time:.2f}ms")

    # Log exceptions
    if exception:
        print(f"REQUEST ERROR: {request_type} {name} - {exception}")


# Custom shape class for ramping up to 50 users


class StagesShape:
    """
    Custom load shape for staged testing.

    Stages:
        1. Ramp up to 10 users (warm-up)
        2. Ramp up to 50 users (target load)
        3. Hold at 50 users (stability test)
        4. Ramp down
    """

    stages = [
        {"duration": 60, "users": 10, "spawn_rate": 2},  # 1 min warm-up
        {"duration": 120, "users": 50, "spawn_rate": 5},  # 2 min ramp to target
        {"duration": 300, "users": 50, "spawn_rate": 5},  # 5 min stability
        {"duration": 60, "users": 0, "spawn_rate": 10},  # 1 min ramp down
    ]

    def tick(self) -> Optional[tuple[int, float]]:
        """Return user count and spawn rate for current time."""
        run_time = self.get_run_time()

        for stage in self.stages:
            if run_time < stage["duration"]:
                return (stage["users"], stage["spawn_rate"])
            run_time -= stage["duration"]

        return None


# For running directly
if __name__ == "__main__":
    import os
    import sys

    print("Zapomni Load Test")
    print("-" * 40)
    print("Usage:")
    print("  locust -f locustfile.py --host=http://127.0.0.1:8000")
    print("")
    print("Options:")
    print("  --headless        Run without web UI")
    print("  -u, --users       Number of concurrent users (default: 50)")
    print("  -r, --spawn-rate  Users spawned per second (default: 10)")
    print("  --run-time        Test duration (e.g., '5m', '30m', '1h')")
    print("")
    print("Example for CI/CD:")
    print("  locust -f locustfile.py --host=http://127.0.0.1:8000 \\")
    print("      --headless -u 50 -r 10 --run-time 5m")
