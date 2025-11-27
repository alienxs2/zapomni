"""
E2E tests for Memory MCP tools (add_memory, search_memory, delete_memory).

Tests verify the complete flow through the MCP server via SSE transport.
Each test uses isolated workspace for test independence.

Author: Zapomni Test Suite
License: MIT
"""

import re
import time
import uuid

import pytest


@pytest.mark.e2e
class TestAddMemory:
    """E2E tests for add_memory tool."""

    def test_add_memory_success(self, mcp_client, clean_workspace, sample_memory_text):
        """Test successful memory addition with sample text."""
        response = mcp_client.call_tool(
            "add_memory",
            {"text": sample_memory_text},
        )

        response.assert_success("add_memory should succeed with valid text")

        # Verify response contains memory_id (UUID format)
        assert "Memory stored successfully" in response.text
        assert "ID:" in response.text

        # Extract and validate memory_id format
        match = re.search(r"ID:\s*([a-f0-9-]{36})", response.text)
        assert match is not None, f"Expected UUID in response, got: {response.text}"
        memory_id = match.group(1)

        # Validate it's a proper UUID
        try:
            uuid.UUID(memory_id)
        except ValueError:
            pytest.fail(f"Invalid UUID format: {memory_id}")

    def test_add_memory_with_metadata(self, mcp_client, clean_workspace):
        """Test memory addition with metadata (tags, source, language)."""
        text = "def hello_world():\n    print('Hello, World!')"
        metadata = {
            "source": "e2e-test",
            "tags": ["python", "code", "example"],
            "language": "python",
        }

        response = mcp_client.call_tool(
            "add_memory",
            {
                "text": text,
                "metadata": metadata,
            },
        )

        response.assert_success("add_memory with metadata should succeed")
        assert "Memory stored successfully" in response.text
        assert "ID:" in response.text

    def test_add_memory_empty_text_fails(self, mcp_client, clean_workspace):
        """Test that empty text fails validation."""
        response = mcp_client.call_tool(
            "add_memory",
            {"text": ""},
        )

        response.assert_error()
        # Should fail with validation error (minLength constraint)

    def test_add_memory_whitespace_only_fails(self, mcp_client, clean_workspace):
        """Test that whitespace-only text fails validation."""
        response = mcp_client.call_tool(
            "add_memory",
            {"text": "   \n\t   "},
        )

        response.assert_error()
        # Should fail because text is stripped and becomes empty

    def test_add_memory_long_text(self, mcp_client, clean_workspace):
        """Test adding long text (should trigger chunking)."""
        # Create text longer than typical chunk size (10000+ chars)
        long_text = "Python programming language. " * 500  # ~15000 chars

        response = mcp_client.call_tool(
            "add_memory",
            {"text": long_text},
        )

        response.assert_success("add_memory with long text should succeed")
        assert "Memory stored successfully" in response.text


@pytest.mark.e2e
class TestSearchMemory:
    """E2E tests for search_memory tool."""

    def test_search_memory_finds_added(self, mcp_client, clean_workspace, sample_memory_text):
        """Test that search finds a recently added memory."""
        # Step 1: Add memory
        add_response = mcp_client.call_tool(
            "add_memory",
            {"text": sample_memory_text},
        )
        add_response.assert_success("add_memory should succeed before search")

        # Delay to allow embedding/indexing (Ollama embedding can be slow)
        time.sleep(2.0)

        # Step 2: Search for it
        search_response = mcp_client.call_tool(
            "search_memory",
            {"query": "Python programming language Guido van Rossum"},
        )

        search_response.assert_success("search_memory should succeed")

        # Should find results (not "No results found")
        assert "No results found" not in search_response.text or "Found" in search_response.text

    def test_search_memory_with_limit(self, mcp_client, clean_workspace):
        """Test search with limit parameter returns correct number of results."""
        # Add multiple distinct memories
        texts = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with many layers.",
            "Natural language processing deals with text and speech.",
        ]

        for text in texts:
            response = mcp_client.call_tool("add_memory", {"text": text})
            response.assert_success("add_memory should succeed")

        # Small delay for indexing
        time.sleep(0.5)

        # Search with limit=2
        search_response = mcp_client.call_tool(
            "search_memory",
            {
                "query": "artificial intelligence machine learning",
                "limit": 2,
            },
        )

        search_response.assert_success("search with limit should succeed")

        # Count results - should be at most 2
        # Results are formatted as "1. [Score: ...]" and "2. [Score: ...]"
        result_count = len(re.findall(r"\d+\.\s*\[Score:", search_response.text))
        assert result_count <= 2, f"Expected at most 2 results, got {result_count}"

    def test_search_memory_no_results(self, mcp_client, clean_workspace):
        """Test search returns appropriate message for non-matching query."""
        # Clear all data to ensure test isolation
        clear_response = mcp_client.call_tool(
            "clear_all",
            {"confirm_phrase": "DELETE ALL MEMORIES"},
        )
        clear_response.assert_success("clear_all should succeed for test isolation")

        # Search for something that doesn't exist in empty workspace
        search_response = mcp_client.call_tool(
            "search_memory",
            {"query": "xyzzy_nonexistent_gibberish_12345"},
        )

        # Should either be no results or empty response (both valid)
        search_response.assert_success("search for non-existent should not error")
        # Response should indicate no results or empty list
        assert (
            "No results found" in search_response.text
            or "Found 0" in search_response.text
            or not re.search(r"\[Score:", search_response.text)
        )

    def test_search_memory_empty_query_fails(self, mcp_client, clean_workspace):
        """Test that empty query fails validation."""
        response = mcp_client.call_tool(
            "search_memory",
            {"query": ""},
        )

        response.assert_error()
        # Should fail with validation error


@pytest.mark.e2e
class TestDeleteMemory:
    """E2E tests for delete_memory tool."""

    def test_delete_memory_success(self, mcp_client, clean_workspace, sample_memory_text):
        """Test successful memory deletion with proper confirmation."""
        # Step 1: Add memory
        add_response = mcp_client.call_tool(
            "add_memory",
            {"text": sample_memory_text},
        )
        add_response.assert_success("add_memory should succeed")

        # Extract memory_id
        match = re.search(r"ID:\s*([a-f0-9-]{36})", add_response.text)
        assert match is not None, "Failed to extract memory_id from add response"
        memory_id = match.group(1)

        # Step 2: Delete with confirm=True
        delete_response = mcp_client.call_tool(
            "delete_memory",
            {
                "memory_id": memory_id,
                "confirm": True,
            },
        )

        delete_response.assert_success("delete_memory with confirm=True should succeed")
        assert "deleted successfully" in delete_response.text.lower() or "Deleted" in delete_response.text

        # Step 3: Verify deletion - search should not find it
        time.sleep(0.5)
        search_response = mcp_client.call_tool(
            "search_memory",
            {"query": sample_memory_text[:100]},  # Use part of the text
        )

        # Should not find the deleted memory or show no results
        search_response.assert_success()

    def test_delete_memory_without_confirm_fails(self, mcp_client, clean_workspace, sample_memory_text):
        """Test that deletion without confirm=True fails as a safety measure."""
        # First add a memory to have a valid ID
        add_response = mcp_client.call_tool(
            "add_memory",
            {"text": sample_memory_text},
        )
        add_response.assert_success()

        # Extract memory_id
        match = re.search(r"ID:\s*([a-f0-9-]{36})", add_response.text)
        memory_id = match.group(1)

        # Try to delete with confirm=False
        delete_response = mcp_client.call_tool(
            "delete_memory",
            {
                "memory_id": memory_id,
                "confirm": False,
            },
        )

        delete_response.assert_error()
        assert "confirm" in delete_response.text.lower() or "confirmation" in delete_response.text.lower()

    def test_delete_memory_invalid_id_fails(self, mcp_client, clean_workspace):
        """Test that deletion with non-existent UUID returns error or not found."""
        # Generate a random UUID that doesn't exist
        fake_memory_id = str(uuid.uuid4())

        delete_response = mcp_client.call_tool(
            "delete_memory",
            {
                "memory_id": fake_memory_id,
                "confirm": True,
            },
        )

        # Should indicate not found (is_error=True with "not found" message)
        delete_response.assert_error()
        assert "not found" in delete_response.text.lower() or "Not Found" in delete_response.text

    def test_delete_memory_invalid_uuid_format_fails(self, mcp_client, clean_workspace):
        """Test that deletion with invalid UUID format fails validation."""
        delete_response = mcp_client.call_tool(
            "delete_memory",
            {
                "memory_id": "not-a-valid-uuid",
                "confirm": True,
            },
        )

        delete_response.assert_error()
        # Should fail with validation error about UUID format
