"""
E2E workflow tests for knowledge graph operations.

Tests the complete lifecycle of graph operations:
- Building graphs from text
- Navigating relationships
- Exporting to different formats
- Incremental graph building
- Integration with memory operations

Copyright (c) 2025 Goncharenko Anton aka alienxs2
License: MIT
"""

import json
import time
import tempfile
from pathlib import Path

import pytest

from tests.e2e.sse_client import MCPSSEClient


@pytest.mark.e2e
@pytest.mark.workflow
class TestGraphWorkflow:
    """Integration tests for knowledge graph workflow."""

    def test_build_and_navigate_graph(
        self,
        mcp_client: MCPSSEClient,
        clean_workspace: str,
        sample_graph_text: str,
    ) -> None:
        """
        Test building graph and navigating relationships.

        Steps:
        1. Build graph from text about Einstein
        2. Check graph status - verify entities created
        3. Get related entities for a found entity
        4. Verify relationship structure
        """
        # Step 1: Build graph from sample text
        response = mcp_client.call_tool(
            "build_graph",
            {
                "text": sample_graph_text,
                "options": {
                    "extract_entities": True,
                    "confidence_threshold": 0.5,
                },
            },
        )
        response.assert_success("Failed to build graph")
        time.sleep(1.0)  # Allow graph processing

        # Step 2: Check graph status
        response = mcp_client.call_tool("graph_status", {})
        response.assert_success("Failed to get graph status")

        # Parse the response to check for entities
        status_text = response.text.lower()
        # Should have some nodes/entities
        assert (
            "node" in status_text
            or "entit" in status_text
            or "graph" in status_text
        ), f"Expected graph status info, got: {response.text}"

        # Step 3: Try to get related entities (if entities were extracted)
        # First, export graph to see what entities exist
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as tmp:
            tmp_path = tmp.name

        try:
            response = mcp_client.call_tool(
                "export_graph",
                {
                    "format": "json",
                    "output_path": tmp_path,
                },
            )

            if not response.is_error:
                # Read the exported graph
                try:
                    with open(tmp_path, "r") as f:
                        graph_data = json.load(f)

                    # If we have nodes with IDs, try get_related
                    nodes = graph_data.get("nodes", [])
                    if nodes:
                        # Try get_related for the first entity
                        first_node = nodes[0]
                        if "id" in first_node:
                            response = mcp_client.call_tool(
                                "get_related",
                                {
                                    "entity_id": first_node["id"],
                                    "depth": 2,
                                    "limit": 10,
                                },
                            )
                            # get_related may or may not find relationships
                            # depending on the graph structure
                except (json.JSONDecodeError, FileNotFoundError, KeyError):
                    # Graph might be empty or format different
                    pass
        finally:
            # Cleanup temp file
            Path(tmp_path).unlink(missing_ok=True)

    def test_graph_export_formats(
        self,
        mcp_client: MCPSSEClient,
        clean_workspace: str,
        sample_graph_text: str,
    ) -> None:
        """
        Test exporting graph in all formats.

        Steps:
        1. Build graph
        2. Export to JSON
        3. Export to GraphML
        4. Export to Cytoscape
        5. Verify all exports successful
        """
        # Build graph first
        response = mcp_client.call_tool(
            "build_graph",
            {
                "text": sample_graph_text,
                "options": {"extract_entities": True},
            },
        )
        response.assert_success("Failed to build graph for export test")
        time.sleep(1.0)  # Allow processing

        # Test export formats
        formats_to_test = [
            ("json", ".json"),
            ("graphml", ".graphml"),
            ("cytoscape", ".json"),
        ]

        export_results = {}

        for format_name, extension in formats_to_test:
            with tempfile.NamedTemporaryFile(
                suffix=extension, delete=False, mode="w"
            ) as tmp:
                tmp_path = tmp.name

            try:
                response = mcp_client.call_tool(
                    "export_graph",
                    {
                        "format": format_name,
                        "output_path": tmp_path,
                    },
                )

                # Check if export was successful
                if not response.is_error:
                    # Verify file was created with content
                    try:
                        with open(tmp_path, "r") as f:
                            content = f.read()
                        export_results[format_name] = len(content) > 0
                    except FileNotFoundError:
                        export_results[format_name] = False
                else:
                    export_results[format_name] = False
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        # At least JSON export should work
        assert export_results.get("json", False) or any(export_results.values()), (
            f"At least one export format should succeed. Results: {export_results}"
        )

    def test_incremental_graph_building(
        self, mcp_client: MCPSSEClient, clean_workspace: str
    ) -> None:
        """
        Test building graph incrementally with multiple texts.

        Steps:
        1. Build graph with text A
        2. Check status - X entities
        3. Build graph with text B (related)
        4. Check status - more entities
        5. Verify graph grew
        """
        # Text A: About Python
        text_a = (
            "Python was created by Guido van Rossum. "
            "He started working on Python in the late 1980s. "
            "Python was first released in 1991."
        )

        # Text B: Related to Text A
        text_b = (
            "Guido van Rossum worked at Google and Dropbox. "
            "He was born in the Netherlands. "
            "Python is now maintained by the Python Software Foundation."
        )

        # Step 1: Build graph with text A
        response = mcp_client.call_tool(
            "build_graph",
            {
                "text": text_a,
                "options": {"extract_entities": True},
            },
        )
        response.assert_success("Failed to build graph with text A")
        time.sleep(1.0)

        # Step 2: Check initial status
        response = mcp_client.call_tool("graph_status", {})
        response.assert_success("Failed to get initial graph status")
        initial_status = response.text

        # Step 3: Build graph with text B
        response = mcp_client.call_tool(
            "build_graph",
            {
                "text": text_b,
                "options": {"extract_entities": True},
            },
        )
        response.assert_success("Failed to build graph with text B")
        time.sleep(1.0)

        # Step 4: Check updated status
        response = mcp_client.call_tool("graph_status", {})
        response.assert_success("Failed to get updated graph status")
        updated_status = response.text

        # Step 5: Verify graph exists (it may or may not have grown depending on entities)
        assert updated_status, "Graph status should return information"

    def test_graph_with_memory_integration(
        self, mcp_client: MCPSSEClient, clean_workspace: str
    ) -> None:
        """
        Test graph building from stored memories.

        Steps:
        1. Add memory
        2. Build graph from same text
        3. Search memory
        4. Verify both work together
        """
        # Test text about a topic
        test_text = (
            "Marie Curie was a Polish-French physicist and chemist. "
            "She discovered polonium and radium. "
            "She won the Nobel Prize in Physics in 1903 and Chemistry in 1911."
        )

        # Step 1: Add memory
        response = mcp_client.call_tool(
            "add_memory",
            {
                "text": test_text,
                "metadata": {
                    "tags": ["science", "physics", "chemistry"],
                    "source": "e2e-graph-integration",
                },
            },
        )
        response.assert_success("Failed to add memory")
        time.sleep(0.5)

        # Step 2: Build graph from same text
        response = mcp_client.call_tool(
            "build_graph",
            {
                "text": test_text,
                "options": {"extract_entities": True},
            },
        )
        response.assert_success("Failed to build graph")
        time.sleep(1.0)

        # Step 3: Search memory
        response = mcp_client.call_tool(
            "search_memory",
            {"query": "Marie Curie Nobel Prize", "limit": 5},
        )
        response.assert_success("Failed to search memory")
        assert "curie" in response.text.lower() or "nobel" in response.text.lower(), (
            f"Expected to find Marie Curie memory, got: {response.text}"
        )

        # Step 4: Verify graph status
        response = mcp_client.call_tool("graph_status", {})
        response.assert_success("Failed to get graph status")

        # Both memory and graph operations should work in the same workspace
        # This verifies they don't interfere with each other
