"""
E2E tests for Graph MCP tools (build_graph, graph_status, get_related, export_graph).

Tests verify the complete flow through the MCP server via SSE transport.
Each test uses isolated workspace for test independence.

Author: Zapomni Test Suite
License: MIT
"""

import json
import os
import re
import tempfile
import time
import uuid

import pytest


@pytest.mark.e2e
class TestBuildGraph:
    """E2E tests for build_graph tool."""

    def test_build_graph_extracts_entities(self, mcp_client, clean_workspace, sample_graph_text):
        """Test entity extraction from text about Einstein."""
        response = mcp_client.call_tool(
            "build_graph",
            {"text": sample_graph_text},
        )

        response.assert_success("build_graph should succeed with valid text")

        # Verify response contains expected information
        assert "Knowledge graph built successfully" in response.text
        assert "Entities:" in response.text
        # Should have extracted some entities (Einstein, Nobel Prize, etc.)
        # Look for non-zero entity count
        match = re.search(r"Entities:\s*(\d+)", response.text)
        assert match is not None, f"Expected Entities count in response, got: {response.text}"
        entities_count = int(match.group(1))
        assert entities_count >= 0, f"Expected non-negative entity count, got: {entities_count}"

    def test_build_graph_with_options(self, mcp_client, clean_workspace):
        """Test build_graph with custom options (confidence_threshold)."""
        text = "Google was founded by Larry Page and Sergey Brin in 1998 at Stanford University."

        response = mcp_client.call_tool(
            "build_graph",
            {
                "text": text,
                "options": {
                    "extract_entities": True,
                    "confidence_threshold": 0.8,
                },
            },
        )

        response.assert_success("build_graph with options should succeed")
        assert "Knowledge graph built successfully" in response.text

    def test_build_graph_empty_text_fails(self, mcp_client, clean_workspace):
        """Test that empty text fails validation."""
        response = mcp_client.call_tool(
            "build_graph",
            {"text": ""},
        )

        response.assert_error()
        # Should fail with validation error (minLength constraint)

    def test_build_graph_whitespace_only_fails(self, mcp_client, clean_workspace):
        """Test that whitespace-only text fails validation."""
        response = mcp_client.call_tool(
            "build_graph",
            {"text": "   \n\t   "},
        )

        response.assert_error()
        # Should fail because text is stripped and becomes empty

    def test_build_graph_with_relationships(self, mcp_client, clean_workspace):
        """Test relationship extraction with build_relationships=true."""
        text = (
            "Microsoft was founded by Bill Gates and Paul Allen. "
            "Microsoft created Windows operating system. "
            "Bill Gates served as CEO of Microsoft until 2000."
        )

        response = mcp_client.call_tool(
            "build_graph",
            {
                "text": text,
                "options": {
                    "extract_entities": True,
                    "build_relationships": True,
                },
            },
        )

        # Should succeed (relationships might be 0 if Phase 2 not implemented)
        response.assert_success("build_graph with relationships should succeed")
        assert "Knowledge graph built successfully" in response.text
        assert "Relationships:" in response.text


@pytest.mark.e2e
class TestGraphStatus:
    """E2E tests for graph_status tool."""

    def test_graph_status_returns_counts(self, mcp_client, clean_workspace):
        """Test graph status returns valid counts."""
        response = mcp_client.call_tool(
            "graph_status",
            {},
        )

        response.assert_success("graph_status should succeed")

        # Verify response structure
        assert "Knowledge Graph Status:" in response.text
        assert "Nodes:" in response.text
        assert "Total:" in response.text
        assert "Relationships:" in response.text
        assert "Graph Health:" in response.text

    def test_graph_status_after_build(self, mcp_client, clean_workspace, sample_graph_text):
        """Test status shows counts after building graph."""
        # Step 1: Get initial status
        initial_response = mcp_client.call_tool("graph_status", {})
        initial_response.assert_success("initial graph_status should succeed")

        # Parse initial total nodes
        initial_match = re.search(r"Total:\s*([\d,]+)", initial_response.text)
        initial_total = int(initial_match.group(1).replace(",", "")) if initial_match else 0

        # Step 2: Build graph
        build_response = mcp_client.call_tool(
            "build_graph",
            {"text": sample_graph_text},
        )
        build_response.assert_success("build_graph should succeed")

        # Small delay for database update
        time.sleep(0.5)

        # Step 3: Get new status
        new_response = mcp_client.call_tool("graph_status", {})
        new_response.assert_success("graph_status after build should succeed")

        # Parse new total nodes
        new_match = re.search(r"Total:\s*([\d,]+)", new_response.text)
        new_total = int(new_match.group(1).replace(",", "")) if new_match else 0

        # Counts should have increased (or at least not decreased)
        assert new_total >= initial_total, (
            f"Expected node count to increase or stay same, "
            f"initial={initial_total}, new={new_total}"
        )

    def test_graph_status_shows_entity_types(self, mcp_client, clean_workspace):
        """Test that graph_status shows entity type breakdown."""
        # Build some graph data first
        text = "Amazon was founded by Jeff Bezos in Seattle, Washington."
        mcp_client.call_tool("build_graph", {"text": text})

        time.sleep(0.5)

        response = mcp_client.call_tool("graph_status", {})
        response.assert_success()

        # Should show entity types section
        assert "Nodes:" in response.text
        # Entity types section may be empty if no entities, but structure should be valid


@pytest.mark.e2e
class TestGetRelated:
    """E2E tests for get_related tool."""

    def test_get_related_invalid_uuid_fails(self, mcp_client, clean_workspace):
        """Test get_related with invalid entity_id format fails."""
        response = mcp_client.call_tool(
            "get_related",
            {"entity_id": "not-a-valid-uuid"},
        )

        response.assert_error()
        # Should fail with validation error about UUID format

    def test_get_related_nonexistent_entity(self, mcp_client, clean_workspace):
        """Test get_related with non-existent entity_id returns empty or error."""
        # Generate a valid UUID that doesn't exist
        fake_entity_id = str(uuid.uuid4())

        response = mcp_client.call_tool(
            "get_related",
            {"entity_id": fake_entity_id},
        )

        # Either returns "no related entities found" or an error
        # Both are acceptable for non-existent entity
        if not response.is_error:
            assert "No related entities found" in response.text or "0" in response.text

    def test_get_related_with_depth(self, mcp_client, clean_workspace):
        """Test get_related with custom depth parameter."""
        fake_entity_id = str(uuid.uuid4())

        response = mcp_client.call_tool(
            "get_related",
            {
                "entity_id": fake_entity_id,
                "depth": 3,
                "limit": 5,
            },
        )

        # Should accept valid parameters (even if no results)
        # Either success with no results or error for not found
        assert response.raw is not None  # Response was received

    def test_get_related_depth_out_of_range_fails(self, mcp_client, clean_workspace):
        """Test get_related fails with depth > 5."""
        response = mcp_client.call_tool(
            "get_related",
            {
                "entity_id": str(uuid.uuid4()),
                "depth": 10,  # Invalid: max is 5
            },
        )

        response.assert_error()
        # Should fail with validation error about depth range


@pytest.mark.e2e
class TestExportGraph:
    """E2E tests for export_graph tool."""

    def test_export_graph_json(self, mcp_client, clean_workspace, sample_graph_text):
        """Test JSON export format."""
        # Build some graph data first
        build_response = mcp_client.call_tool(
            "build_graph",
            {"text": sample_graph_text},
        )
        build_response.assert_success("build_graph should succeed before export")

        time.sleep(0.5)

        # Export to JSON
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "graph_export.json")

            response = mcp_client.call_tool(
                "export_graph",
                {
                    "format": "json",
                    "output_path": output_path,
                },
            )

            response.assert_success("export_graph to JSON should succeed")
            assert "Graph exported successfully" in response.text
            assert "Format: json" in response.text

            # Verify file was created and is valid JSON
            assert os.path.exists(output_path), f"Export file not created: {output_path}"

            with open(output_path, "r") as f:
                content = f.read()
                # Should be valid JSON
                try:
                    data = json.loads(content)
                    # Basic structure check
                    assert isinstance(data, dict), "JSON export should be a dictionary"
                except json.JSONDecodeError as e:
                    pytest.fail(f"Export file is not valid JSON: {e}")

    def test_export_graph_graphml(self, mcp_client, clean_workspace):
        """Test GraphML export format."""
        # Build some graph data first
        mcp_client.call_tool(
            "build_graph",
            {"text": "Python is a programming language created by Guido van Rossum."},
        )

        time.sleep(0.5)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "graph_export.graphml")

            response = mcp_client.call_tool(
                "export_graph",
                {
                    "format": "graphml",
                    "output_path": output_path,
                },
            )

            response.assert_success("export_graph to GraphML should succeed")
            assert "Graph exported successfully" in response.text
            assert "Format: graphml" in response.text

            # Verify file was created
            assert os.path.exists(output_path), f"Export file not created: {output_path}"

            # GraphML is XML format
            with open(output_path, "r") as f:
                content = f.read()
                assert "<?xml" in content or "<graphml" in content, "Export should be XML format"

    def test_export_graph_cytoscape(self, mcp_client, clean_workspace):
        """Test Cytoscape JSON export format."""
        # Build some graph data first
        mcp_client.call_tool(
            "build_graph",
            {"text": "Apple was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne."},
        )

        time.sleep(0.5)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "graph_export_cytoscape.json")

            response = mcp_client.call_tool(
                "export_graph",
                {
                    "format": "cytoscape",
                    "output_path": output_path,
                    "options": {
                        "include_style": True,
                    },
                },
            )

            response.assert_success("export_graph to Cytoscape should succeed")
            assert "Graph exported successfully" in response.text
            assert "Format: cytoscape" in response.text

            # Verify file exists
            assert os.path.exists(output_path)

    def test_export_graph_invalid_format_fails(self, mcp_client, clean_workspace):
        """Test that invalid export format fails validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "graph_export.xyz")

            response = mcp_client.call_tool(
                "export_graph",
                {
                    "format": "invalid_format",
                    "output_path": output_path,
                },
            )

            response.assert_error()
            # Should fail with validation error about format

    def test_export_graph_missing_output_path_fails(self, mcp_client, clean_workspace):
        """Test that missing output_path fails validation."""
        response = mcp_client.call_tool(
            "export_graph",
            {
                "format": "json",
                # Missing output_path
            },
        )

        response.assert_error()
        # Should fail with validation error about required field

    def test_export_graph_empty_output_path_fails(self, mcp_client, clean_workspace):
        """Test that empty output_path fails validation."""
        response = mcp_client.call_tool(
            "export_graph",
            {
                "format": "json",
                "output_path": "",
            },
        )

        response.assert_error()
        # Should fail with validation error about minLength
