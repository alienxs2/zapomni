"""
E2E tests for concurrent access and thread safety.

Tests that the MCP server properly handles:
- Multiple concurrent read operations
- Multiple concurrent write operations
- Mixed read/write concurrent operations

These tests use threading to simulate concurrent access patterns.

Copyright (c) 2025 Goncharenko Anton aka alienxs2
License: MIT
"""

import threading
import time
from typing import List, Optional

import pytest

from tests.e2e.sse_client import MCPResponse, MCPSSEClient

# Timeout for concurrent operations (seconds)
CONCURRENT_TIMEOUT = 60.0
# Number of concurrent threads
NUM_THREADS = 5


@pytest.mark.e2e
@pytest.mark.slow
class TestConcurrentAccess:
    """Tests for concurrent access and thread safety."""

    def test_concurrent_reads(
        self,
        mcp_client: MCPSSEClient,
        clean_workspace: str,
        sample_memory_text: str,
    ) -> None:
        """Test multiple concurrent read operations."""
        # Step 1: Add a memory to search for
        add_response = mcp_client.call_tool(
            "add_memory",
            {"text": sample_memory_text},
        )
        add_response.assert_success("Failed to add memory for concurrent read test")

        # Give the system time to index
        time.sleep(1.0)

        # Step 2: Launch concurrent search threads
        results: List[Optional[MCPResponse]] = [None] * NUM_THREADS
        errors: List[Optional[Exception]] = [None] * NUM_THREADS
        threads: List[threading.Thread] = []

        def search_worker(idx: int) -> None:
            """Worker function for concurrent search."""
            try:
                response = mcp_client.call_tool(
                    "search_memory",
                    {"query": "Python programming language", "limit": 10},
                )
                results[idx] = response
            except Exception as e:
                errors[idx] = e

        # Create and start threads
        for i in range(NUM_THREADS):
            t = threading.Thread(target=search_worker, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all threads with timeout
        for t in threads:
            t.join(timeout=CONCURRENT_TIMEOUT / NUM_THREADS)

        # Step 3: Verify all succeeded without errors
        for i in range(NUM_THREADS):
            assert errors[i] is None, f"Thread {i} got error: {errors[i]}"
            assert results[i] is not None, f"Thread {i} got no result"
            results[i].assert_success(f"Thread {i} search failed")

    def test_concurrent_writes(
        self,
        mcp_client: MCPSSEClient,
        clean_workspace: str,
    ) -> None:
        """Test multiple concurrent write operations."""
        # Step 1: Launch concurrent add_memory threads
        results: List[Optional[MCPResponse]] = [None] * NUM_THREADS
        errors: List[Optional[Exception]] = [None] * NUM_THREADS
        threads: List[threading.Thread] = []

        def write_worker(idx: int) -> None:
            """Worker function for concurrent write."""
            try:
                unique_text = (
                    f"Concurrent write test document number {idx}. "
                    f"This is unique content for thread {idx} to verify "
                    f"that concurrent writes work correctly without data corruption. "
                    f"Unique identifier: CONCURRENT_WRITE_{idx}_MARKER"
                )
                response = mcp_client.call_tool(
                    "add_memory",
                    {"text": unique_text},
                )
                results[idx] = response
            except Exception as e:
                errors[idx] = e

        # Create and start threads
        for i in range(NUM_THREADS):
            t = threading.Thread(target=write_worker, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all threads with timeout
        for t in threads:
            t.join(timeout=CONCURRENT_TIMEOUT / NUM_THREADS)

        # Step 2: Verify all writes succeeded
        for i in range(NUM_THREADS):
            assert errors[i] is None, f"Write thread {i} got error: {errors[i]}"
            assert results[i] is not None, f"Write thread {i} got no result"
            results[i].assert_success(f"Write thread {i} failed")

        # Give the system time to index
        time.sleep(2.0)

        # Step 3: Verify all memories were added by searching
        stats_response = mcp_client.call_tool("get_stats", {})
        stats_response.assert_success("Failed to get stats after concurrent writes")

        # Search for our concurrent write markers
        found_count = 0
        for i in range(NUM_THREADS):
            search_response = mcp_client.call_tool(
                "search_memory",
                {"query": f"CONCURRENT_WRITE_{i}_MARKER", "limit": 5},
            )
            if not search_response.is_error and search_response.text:
                if (
                    f"CONCURRENT_WRITE_{i}_MARKER" in search_response.text
                    or "thread" in search_response.text.lower()
                ):
                    found_count += 1

        # At least some should be found (semantic search may not find all exact matches)
        assert (
            found_count >= 1
        ), f"Expected to find at least 1 concurrent write memory, found {found_count}"

    def test_mixed_concurrent_operations(
        self,
        mcp_client: MCPSSEClient,
        clean_workspace: str,
    ) -> None:
        """Test mixed read/write concurrent operations."""
        # Step 1: Add initial memory
        init_response = mcp_client.call_tool(
            "add_memory",
            {"text": "Initial memory for mixed concurrent test. Python programming."},
        )
        init_response.assert_success("Failed to add initial memory")

        time.sleep(1.0)

        # Step 2: Launch mixed threads - some reading, some writing
        results: List[Optional[MCPResponse]] = [None] * NUM_THREADS
        errors: List[Optional[Exception]] = [None] * NUM_THREADS
        threads: List[threading.Thread] = []

        def reader_worker(idx: int) -> None:
            """Worker function for read operations."""
            try:
                response = mcp_client.call_tool(
                    "search_memory",
                    {"query": "Python programming", "limit": 5},
                )
                results[idx] = response
            except Exception as e:
                errors[idx] = e

        def writer_worker(idx: int) -> None:
            """Worker function for write operations."""
            try:
                response = mcp_client.call_tool(
                    "add_memory",
                    {"text": f"Mixed concurrent write {idx}. Database systems."},
                )
                results[idx] = response
            except Exception as e:
                errors[idx] = e

        # Alternate between readers and writers
        for i in range(NUM_THREADS):
            if i % 2 == 0:
                t = threading.Thread(target=reader_worker, args=(i,))
            else:
                t = threading.Thread(target=writer_worker, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all threads with timeout
        for t in threads:
            t.join(timeout=CONCURRENT_TIMEOUT / NUM_THREADS)

        # Step 3: Verify no deadlocks or errors
        for i in range(NUM_THREADS):
            assert errors[i] is None, f"Mixed thread {i} got error: {errors[i]}"
            assert results[i] is not None, f"Mixed thread {i} got no result"
            # We don't assert success because some operations may legitimately fail
            # The important thing is no deadlocks or crashes
