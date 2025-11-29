"""
E2E tests for error handling and edge cases.

Tests that the MCP server properly handles:
- Invalid tool names
- Missing required parameters
- Invalid parameter types
- Edge case inputs (unicode, special characters)
- Empty database operations

Copyright (c) 2025 Goncharenko Anton aka alienxs2
License: MIT
"""

import pytest

from tests.e2e.sse_client import MCPSSEClient


@pytest.mark.e2e
class TestErrorHandling:
    """Tests for error handling and graceful degradation."""

    def test_invalid_tool_name(self, mcp_client: MCPSSEClient) -> None:
        """Test calling non-existent tool returns error."""
        # Call a tool that doesn't exist
        response = mcp_client.call_tool(
            "nonexistent_tool_xyz_12345",
            {},
        )

        # Should return an error response
        assert response.is_error, f"Expected error for invalid tool, got: {response.text}"
        # The error message should indicate the tool is unknown
        assert (
            "nonexistent_tool_xyz_12345" in response.text.lower()
            or "unknown" in response.text.lower()
            or "not found" in response.text.lower()
        ), f"Error message should mention the tool name or 'unknown', got: {response.text}"

    def test_missing_required_parameters(self, mcp_client: MCPSSEClient) -> None:
        """Test missing required parameters returns error."""
        # add_memory requires 'text' parameter
        response = mcp_client.call_tool(
            "add_memory",
            {},  # Missing required 'text' parameter
        )

        # Should return an error about missing parameter
        assert response.is_error, f"Expected error for missing parameter, got: {response.text}"
        error_text = response.text.lower()
        assert (
            "text" in error_text or "required" in error_text or "missing" in error_text
        ), f"Error should mention 'text' or 'required', got: {response.text}"

    def test_invalid_parameter_types(self, mcp_client: MCPSSEClient) -> None:
        """Test invalid parameter types are handled gracefully."""
        # search_memory 'limit' should be an integer, not a string
        response = mcp_client.call_tool(
            "search_memory",
            {
                "query": "test query",
                "limit": "not_a_number",  # Invalid type: should be int
            },
        )

        # Should handle gracefully - either error or coerce type
        # We just verify it doesn't crash the server
        # The response could be an error or success depending on implementation
        assert response.raw is not None, "Response should have raw data"

    def test_invalid_uuid_parameter(self, mcp_client: MCPSSEClient) -> None:
        """Test invalid UUID format is handled gracefully."""
        # delete_memory requires a valid UUID
        response = mcp_client.call_tool(
            "delete_memory",
            {
                "memory_id": "not-a-valid-uuid-format",
                "confirm": True,
            },
        )

        # Should return an error about invalid UUID
        assert response.is_error, f"Expected error for invalid UUID, got: {response.text}"


@pytest.mark.e2e
class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_unicode_text_handling(
        self,
        mcp_client: MCPSSEClient,
        clean_workspace: str,
    ) -> None:
        """Test handling of unicode and special characters."""
        # Text with various unicode characters
        unicode_text = (
            "This text contains emoji: \U0001f4da\U0001f40d\U0001f680 "
            "and CJK characters: \u4e2d\u6587\u6d4b\u8bd5 "
            "and Cyrillic: \u041f\u0440\u0438\u0432\u0435\u0442 \u043c\u0438\u0440! "
            "and special chars: <>&\"' \u00e9\u00e8\u00ea \u00f1 \u00df "
            "and newlines:\nline2\nline3"
        )

        # Add memory with unicode
        add_response = mcp_client.call_tool(
            "add_memory",
            {"text": unicode_text},
        )
        add_response.assert_success("Failed to add memory with unicode text")

        # Search for unicode content
        search_response = mcp_client.call_tool(
            "search_memory",
            {"query": "emoji CJK Cyrillic", "limit": 5},
        )
        # Search should work even if no exact match
        assert search_response.raw is not None, "Search response should have raw data"

    def test_empty_database_search(
        self,
        mcp_client: MCPSSEClient,
        clean_workspace: str,
    ) -> None:
        """Test search on fresh empty workspace."""
        # The clean_workspace fixture creates a fresh workspace
        # Search should return gracefully with no results

        response = mcp_client.call_tool(
            "search_memory",
            {"query": "something that doesn't exist", "limit": 10},
        )

        # Should not error, just return empty or no matches
        response.assert_success("Search on empty workspace should not error")

    def test_empty_database_graph_status(
        self,
        mcp_client: MCPSSEClient,
        clean_workspace: str,
    ) -> None:
        """Test graph_status on empty database."""
        response = mcp_client.call_tool(
            "graph_status",
            {},
        )

        # Should return stats showing zero or minimal values
        response.assert_success("graph_status on empty workspace should not error")
        # Verify we got some response data
        assert response.text, "graph_status should return some text"

    def test_empty_database_get_stats(
        self,
        mcp_client: MCPSSEClient,
        clean_workspace: str,
    ) -> None:
        """Test get_stats on empty database."""
        response = mcp_client.call_tool(
            "get_stats",
            {},
        )

        response.assert_success("get_stats on empty workspace should not error")
        assert response.text, "get_stats should return statistics text"

    def test_special_characters_in_query(
        self,
        mcp_client: MCPSSEClient,
        clean_workspace: str,
    ) -> None:
        """Test search with special characters in query."""
        # Add a simple memory first
        mcp_client.call_tool(
            "add_memory",
            {"text": "Test document for special character search"},
        )

        # Search with special characters
        special_queries = [
            "test & search",
            "test | search",
            "test (search)",
            'test "search"',
            "test <search>",
            "test\ttab\nnewline",
        ]

        for query in special_queries:
            response = mcp_client.call_tool(
                "search_memory",
                {"query": query, "limit": 5},
            )
            # Should not crash, may return results or not
            assert response.raw is not None, f"Query '{query}' should not crash server"
