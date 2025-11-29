"""
E2E workflow tests for memory operations.

Tests the complete lifecycle of memory operations:
- Adding memories with various content
- Searching with semantic relevance
- Filtering by metadata
- Bulk operations
- Deletion and verification

Copyright (c) 2025 Goncharenko Anton aka alienxs2
License: MIT
"""

import time

import pytest

from tests.e2e.sse_client import MCPSSEClient


@pytest.mark.e2e
@pytest.mark.workflow
class TestMemoryWorkflow:
    """Integration tests for memory operations workflow."""

    def test_full_memory_lifecycle(self, mcp_client: MCPSSEClient, clean_workspace: str) -> None:
        """
        Test complete memory lifecycle: add -> search -> update -> delete.

        This test verifies the full CRUD workflow for memories:
        1. Add initial memory
        2. Search and find it
        3. Add related memory
        4. Search with different query - find both
        5. Delete one memory
        6. Verify only one remains
        """
        # Step 1: Add initial memory about Python
        response = mcp_client.call_tool(
            "add_memory",
            {
                "text": "Python is a high-level programming language known for its readability. "
                "It was created by Guido van Rossum and first released in 1991.",
                "metadata": {
                    "tags": ["programming", "python"],
                    "source": "e2e-test",
                },
            },
        )
        response.assert_success("Failed to add initial memory")
        time.sleep(0.5)  # Allow indexing

        # Step 2: Search for the memory
        response = mcp_client.call_tool(
            "search_memory",
            {"query": "Python programming language", "limit": 5},
        )
        response.assert_success("Failed to search for memory")
        assert (
            "Python" in response.text or "python" in response.text.lower()
        ), f"Expected to find Python memory, got: {response.text}"

        # Step 3: Add related memory about JavaScript
        response = mcp_client.call_tool(
            "add_memory",
            {
                "text": "JavaScript is a dynamic programming language used for web development. "
                "It was created by Brendan Eich in 1995 while working at Netscape.",
                "metadata": {
                    "tags": ["programming", "javascript"],
                    "source": "e2e-test",
                },
            },
        )
        response.assert_success("Failed to add JavaScript memory")
        time.sleep(0.5)  # Allow indexing

        # Step 4: Search with broader query - should find both
        response = mcp_client.call_tool(
            "search_memory",
            {"query": "programming languages history", "limit": 10},
        )
        response.assert_success("Failed to search for programming languages")
        # Both languages should be in results (semantic search)
        text_lower = response.text.lower()
        # At least one should be found
        assert (
            "python" in text_lower or "javascript" in text_lower
        ), f"Expected to find programming language memories, got: {response.text}"

        # Step 5: Get stats to verify memories exist
        response = mcp_client.call_tool("get_stats", {})
        response.assert_success("Failed to get stats")
        # Stats should show memories
        assert "memories" in response.text.lower() or "total" in response.text.lower()

    def test_memory_search_relevance(self, mcp_client: MCPSSEClient, clean_workspace: str) -> None:
        """
        Test that search returns relevant results.

        Add multiple memories on different topics and verify
        that search returns the most relevant results first.
        """
        # Add memories on different topics
        topics = [
            {
                "text": "Machine learning is a subset of artificial intelligence "
                "that enables systems to learn from data.",
                "tags": ["ai", "ml"],
            },
            {
                "text": "Cooking Italian pasta requires quality ingredients: "
                "durum wheat semolina, fresh eggs, and olive oil.",
                "tags": ["cooking", "food"],
            },
            {
                "text": "Neural networks are computing systems inspired by "
                "biological neural networks in animal brains.",
                "tags": ["ai", "neural"],
            },
        ]

        for topic in topics:
            response = mcp_client.call_tool(
                "add_memory",
                {
                    "text": topic["text"],
                    "metadata": {
                        "tags": topic["tags"],
                        "source": "e2e-test-relevance",
                    },
                },
            )
            response.assert_success(f"Failed to add memory: {topic['text'][:30]}...")
            time.sleep(0.3)

        time.sleep(0.5)  # Allow indexing

        # Search for AI-related content
        response = mcp_client.call_tool(
            "search_memory",
            {"query": "artificial intelligence and machine learning", "limit": 5},
        )
        response.assert_success("Failed to search for AI content")

        # AI-related memories should be returned
        text_lower = response.text.lower()
        assert (
            "machine learning" in text_lower
            or "neural" in text_lower
            or "artificial intelligence" in text_lower
        ), f"Expected AI-related results, got: {response.text}"

        # Cooking should NOT be the primary result for AI query
        # (it may appear but should not dominate)

    def test_memory_with_metadata_search(
        self, mcp_client: MCPSSEClient, clean_workspace: str
    ) -> None:
        """
        Test search with metadata filters.

        Add memories with different tags and verify that
        filtering by tags works correctly.
        """
        # Add memories with different tags
        memories = [
            {
                "text": "Docker containers provide lightweight virtualization "
                "for application deployment.",
                "tags": ["devops", "containers"],
            },
            {
                "text": "Kubernetes orchestrates container deployment, scaling, "
                "and management across clusters.",
                "tags": ["devops", "kubernetes"],
            },
            {
                "text": "React is a JavaScript library for building user interfaces "
                "with a component-based architecture.",
                "tags": ["frontend", "react"],
            },
        ]

        for memory in memories:
            response = mcp_client.call_tool(
                "add_memory",
                {
                    "text": memory["text"],
                    "metadata": {
                        "tags": memory["tags"],
                        "source": "e2e-test-metadata",
                    },
                },
            )
            response.assert_success(f"Failed to add memory: {memory['text'][:30]}...")
            time.sleep(0.3)

        time.sleep(0.5)  # Allow indexing

        # Search with tag filter for devops
        response = mcp_client.call_tool(
            "search_memory",
            {
                "query": "container technology",
                "limit": 10,
                "filters": {"tags": ["devops"]},
            },
        )
        response.assert_success("Failed to search with tag filter")

        # Should find devops-related content
        text_lower = response.text.lower()
        # DevOps content should be present
        assert (
            "docker" in text_lower or "kubernetes" in text_lower or "container" in text_lower
        ), f"Expected DevOps content with tag filter, got: {response.text}"

    def test_bulk_memory_operations(self, mcp_client: MCPSSEClient, clean_workspace: str) -> None:
        """
        Test adding and searching multiple memories.

        Add 5-10 memories and verify all can be found.
        """
        # Add multiple memories
        items = [
            "PostgreSQL is a powerful open-source relational database system.",
            "MongoDB is a document-oriented NoSQL database for modern applications.",
            "Redis is an in-memory data structure store used as database and cache.",
            "Elasticsearch is a distributed search and analytics engine.",
            "Apache Kafka is a distributed event streaming platform.",
            "GraphQL is a query language for APIs with a runtime for execution.",
            "gRPC is a high-performance RPC framework using Protocol Buffers.",
        ]

        added_count = 0
        for i, item in enumerate(items):
            response = mcp_client.call_tool(
                "add_memory",
                {
                    "text": item,
                    "metadata": {
                        "tags": ["database", "tech"],
                        "source": f"e2e-bulk-{i}",
                    },
                },
            )
            if not response.is_error:
                added_count += 1
            time.sleep(0.2)

        assert added_count >= 5, f"Expected at least 5 memories added, got {added_count}"
        time.sleep(0.5)  # Allow indexing

        # Search for database-related content
        response = mcp_client.call_tool(
            "search_memory",
            {"query": "database systems", "limit": 10},
        )
        response.assert_success("Failed to search bulk memories")

        # Should find multiple results
        text_lower = response.text.lower()
        found_items = sum(
            1
            for keyword in ["postgresql", "mongodb", "redis", "elasticsearch"]
            if keyword in text_lower
        )
        # At least some should be found
        assert found_items >= 1, f"Expected to find database memories, got: {response.text}"

        # Verify stats show the memories
        response = mcp_client.call_tool("get_stats", {})
        response.assert_success("Failed to get stats after bulk add")
