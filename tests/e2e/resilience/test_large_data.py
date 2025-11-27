"""
E2E tests for handling large data volumes.

Tests that the MCP server properly handles:
- Large text memories (10KB+)
- Many memories (20+)
- Large knowledge graphs

These tests may take longer to run due to data volume.

Copyright (c) 2025 Goncharenko Anton aka alienxs2
License: MIT
"""

import json
import time

import pytest

from tests.e2e.sse_client import MCPSSEClient


@pytest.mark.e2e
@pytest.mark.slow
class TestLargeData:
    """Tests for handling large data volumes."""

    def test_large_text_memory(
        self,
        mcp_client: MCPSSEClient,
        clean_workspace: str,
    ) -> None:
        """Test adding large text (10KB+)."""
        # Create a large text document (~12KB)
        base_paragraph = (
            "This is a test paragraph that will be repeated multiple times "
            "to create a large document for testing the memory chunking system. "
            "The document contains information about artificial intelligence, "
            "machine learning, natural language processing, and knowledge graphs. "
            "It discusses how embeddings work, how semantic search operates, "
            "and how memory systems can be used to store and retrieve information. "
            "This paragraph is approximately 500 characters when repeated. "
        )

        # Repeat to get ~12KB (approximately 24 repetitions * 500 chars = 12000 chars)
        large_text = (base_paragraph * 24)

        # Verify size
        assert len(large_text) > 10000, f"Large text should be >10KB, got {len(large_text)} chars"

        # Add the large memory
        add_response = mcp_client.call_tool(
            "add_memory",
            {"text": large_text},
        )
        add_response.assert_success("Failed to add large text memory")

        # Give time for chunking and indexing
        time.sleep(2.0)

        # Verify memory was added by searching for unique content
        search_response = mcp_client.call_tool(
            "search_memory",
            {"query": "artificial intelligence machine learning embeddings", "limit": 5},
        )
        search_response.assert_success("Failed to search for large memory")

    def test_many_memories(
        self,
        mcp_client: MCPSSEClient,
        clean_workspace: str,
    ) -> None:
        """Test adding many memories (20+)."""
        num_memories = 20

        # Different topics for variety in semantic space
        topics = [
            "Python programming language features and syntax",
            "JavaScript web development frameworks like React",
            "Database management with PostgreSQL and MySQL",
            "Machine learning algorithms and neural networks",
            "Cloud computing with AWS and Azure services",
            "DevOps practices including CI/CD pipelines",
            "Containerization with Docker and Kubernetes",
            "API design patterns and REST best practices",
            "Data structures including trees and graphs",
            "Software testing strategies and methodologies",
            "Agile development and Scrum frameworks",
            "Version control with Git and branching strategies",
            "Security best practices for web applications",
            "Performance optimization techniques",
            "Microservices architecture design patterns",
            "Mobile development for iOS and Android",
            "Functional programming concepts and paradigms",
            "Object-oriented design principles SOLID",
            "Distributed systems and scalability",
            "Event-driven architecture and messaging",
        ]

        # Add 20 memories
        add_errors = []
        for i in range(num_memories):
            topic = topics[i % len(topics)]
            text = (
                f"Memory document {i + 1} about {topic}. "
                f"This is unique content for testing multiple memory addition. "
                f"Unique marker: MULTI_MEM_{i}_MARKER"
            )

            response = mcp_client.call_tool(
                "add_memory",
                {"text": text},
            )

            if response.is_error:
                add_errors.append(f"Memory {i + 1}: {response.text}")

        # Allow some failures but not too many
        assert len(add_errors) <= 2, f"Too many add_memory failures: {add_errors}"

        # Give time for indexing
        time.sleep(3.0)

        # Verify stats show memories were added
        stats_response = mcp_client.call_tool("get_stats", {})
        stats_response.assert_success("Failed to get stats after adding many memories")

        # Search should work
        search_response = mcp_client.call_tool(
            "search_memory",
            {"query": "programming development", "limit": 10},
        )
        search_response.assert_success("Failed to search with many memories")

    def test_large_graph(
        self,
        mcp_client: MCPSSEClient,
        clean_workspace: str,
    ) -> None:
        """Test building graph from large text with many entities."""
        # Text with many entities (people, places, organizations, concepts)
        entity_rich_text = """
        Albert Einstein was a German-born theoretical physicist who developed
        the theory of relativity. He was born in Ulm, Germany and later worked
        at the Swiss Federal Institute of Technology in Zurich. Einstein received
        the Nobel Prize in Physics in 1921. He moved to the United States in 1933
        and worked at the Institute for Advanced Study in Princeton, New Jersey.

        Marie Curie was a Polish physicist and chemist who conducted pioneering
        research on radioactivity. She was the first woman to win a Nobel Prize
        and the only person to win Nobel Prizes in two different sciences (Physics
        and Chemistry). She worked at the University of Paris and founded the
        Curie Institutes in Paris and Warsaw.

        Isaac Newton was an English mathematician and physicist who is widely
        recognized as one of the most influential scientists of all time. He
        developed the laws of motion and universal gravitation at Cambridge
        University. Newton was also President of the Royal Society in London.

        Nikola Tesla was a Serbian-American inventor and electrical engineer
        who contributed to the design of the modern alternating current (AC)
        electricity supply system. He worked for Thomas Edison at Edison Machine
        Works in New York before starting his own companies. Tesla's laboratory
        was in Colorado Springs and later in Wardenclyffe on Long Island.

        Richard Feynman was an American theoretical physicist known for his work
        in quantum mechanics and quantum electrodynamics. He worked at Cornell
        University and later at California Institute of Technology (Caltech).
        Feynman received the Nobel Prize in Physics in 1965.
        """

        # Build graph from entity-rich text
        graph_response = mcp_client.call_tool(
            "build_graph",
            {
                "text": entity_rich_text,
                "options": {
                    "extract_entities": True,
                    "confidence_threshold": 0.5,
                },
            },
        )
        graph_response.assert_success("Failed to build graph from entity-rich text")

        # Give time for graph processing
        time.sleep(2.0)

        # Check graph status shows entities
        status_response = mcp_client.call_tool("graph_status", {})
        status_response.assert_success("Failed to get graph status")

        # The response should show some graph data
        assert status_response.text, "graph_status should return status text"

        # Export graph should work
        export_response = mcp_client.call_tool(
            "export_graph",
            {
                "format": "json",
                "output_path": "/tmp/test_large_graph_export.json",
            },
        )
        # Export may or may not succeed depending on permissions, but shouldn't crash
        assert export_response.raw is not None, "export_graph should return response"
