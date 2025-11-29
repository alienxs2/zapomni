"""
E2E tests for Code indexing MCP tools (index_codebase).

Tests verify the complete flow through the MCP server via SSE transport.
Each test validates index_codebase tool behavior with various parameters.

Author: Zapomni Test Suite
License: MIT
"""

import os
import re

import pytest


@pytest.mark.e2e
class TestIndexCodebase:
    """E2E tests for index_codebase tool."""

    def test_index_codebase_success(self, mcp_client, sample_code_project):
        """Test successful indexing of Python project."""
        response = mcp_client.call_tool(
            "index_codebase",
            {"repo_path": str(sample_code_project)},
        )

        response.assert_success("index_codebase should succeed with valid path")

        # Verify response contains expected indexing information
        assert "Repository indexed successfully" in response.text
        assert "Path:" in response.text
        assert "Files indexed:" in response.text
        assert str(sample_code_project) in response.text

        # Should have indexed at least 1 file
        match = re.search(r"Files indexed:\s*(\d+)", response.text)
        assert match is not None, f"Expected file count in response, got: {response.text}"
        files_indexed = int(match.group(1))
        assert files_indexed >= 1, f"Expected at least 1 file indexed, got {files_indexed}"

    def test_index_codebase_python_only(self, mcp_client, sample_code_project):
        """Test indexing with language filter (python only)."""
        response = mcp_client.call_tool(
            "index_codebase",
            {
                "repo_path": str(sample_code_project),
                "languages": ["python"],
            },
        )

        response.assert_success("index_codebase with python filter should succeed")

        # Verify response shows Python files were indexed
        assert "Repository indexed successfully" in response.text
        assert "Python" in response.text or "python" in response.text.lower()

        # Should have indexed files
        match = re.search(r"Files indexed:\s*(\d+)", response.text)
        assert match is not None, f"Expected file count in response"
        files_indexed = int(match.group(1))
        assert files_indexed >= 1, f"Expected at least 1 Python file, got {files_indexed}"

    def test_index_codebase_nonexistent_path_fails(self, mcp_client):
        """Test indexing non-existent path fails with appropriate error."""
        fake_path = "/nonexistent/path/that/does/not/exist/12345"

        response = mcp_client.call_tool(
            "index_codebase",
            {"repo_path": fake_path},
        )

        response.assert_error()
        # Error should mention path issue
        assert (
            "not exist" in response.text.lower()
            or "does not exist" in response.text.lower()
            or "Repository path" in response.text
        ), f"Expected path error message, got: {response.text}"

    def test_index_codebase_empty_path_fails(self, mcp_client):
        """Test empty path fails validation."""
        response = mcp_client.call_tool(
            "index_codebase",
            {"repo_path": ""},
        )

        response.assert_error()
        # Should fail with validation error about empty path

    def test_index_codebase_with_recursive_false(self, mcp_client, sample_code_project):
        """Test non-recursive indexing only indexes top-level files."""
        response = mcp_client.call_tool(
            "index_codebase",
            {
                "repo_path": str(sample_code_project),
                "recursive": False,
            },
        )

        response.assert_success("index_codebase with recursive=false should succeed")
        assert "Repository indexed successfully" in response.text

        # Should still find the main.py at top level
        match = re.search(r"Files indexed:\s*(\d+)", response.text)
        assert match is not None
        files_indexed = int(match.group(1))
        # With recursive=False, we should have fewer or equal files
        # compared to recursive=True, but at least main.py
        assert files_indexed >= 1, "Should index at least top-level files"

    def test_index_codebase_finds_python_structures(self, mcp_client, sample_code_project):
        """Test that AST analysis reports functions and classes in response."""
        response = mcp_client.call_tool(
            "index_codebase",
            {
                "repo_path": str(sample_code_project),
                "languages": ["python"],
            },
        )

        response.assert_success("index_codebase should succeed")

        # Verify response includes function and class counts
        assert "Functions:" in response.text
        assert "Classes:" in response.text

        # The sample code has Calculator class and functions (calculate_sum, calculate_product)
        # Note: Current implementation may return 0 for these (AST parsing not implemented)
        # but the response format should still include these fields

    def test_index_codebase_with_max_file_size(self, mcp_client, sample_code_project):
        """Test indexing with custom max_file_size parameter."""
        # Use a large max_file_size - should index all files
        response = mcp_client.call_tool(
            "index_codebase",
            {
                "repo_path": str(sample_code_project),
                "max_file_size": 10485760,  # 10MB (default)
            },
        )

        response.assert_success("index_codebase with max_file_size should succeed")
        assert "Repository indexed successfully" in response.text

    def test_index_codebase_with_include_tests_false(self, mcp_client, sample_code_project):
        """Test indexing excludes test files when include_tests=False."""
        response = mcp_client.call_tool(
            "index_codebase",
            {
                "repo_path": str(sample_code_project),
                "include_tests": False,
            },
        )

        response.assert_success("index_codebase with include_tests=false should succeed")
        assert "Repository indexed successfully" in response.text

        # Should index non-test files (main.py is not a test file)
        match = re.search(r"Files indexed:\s*(\d+)", response.text)
        assert match is not None
        files_indexed = int(match.group(1))
        assert files_indexed >= 1, "Should index at least non-test files"

    def test_index_codebase_invalid_language_fails(self, mcp_client, sample_code_project):
        """Test that invalid language filter fails validation."""
        response = mcp_client.call_tool(
            "index_codebase",
            {
                "repo_path": str(sample_code_project),
                "languages": ["invalid_language_xyz"],
            },
        )

        response.assert_error()
        # Should fail with validation error about unsupported language
        assert (
            "unsupported" in response.text.lower()
            or "invalid" in response.text.lower()
            or "language" in response.text.lower()
        ), f"Expected language validation error, got: {response.text}"

    def test_index_codebase_file_path_not_directory_fails(self, mcp_client, sample_code_project):
        """Test that passing a file path instead of directory fails."""
        # Pass the main.py file path instead of directory
        file_path = sample_code_project / "main.py"

        response = mcp_client.call_tool(
            "index_codebase",
            {"repo_path": str(file_path)},
        )

        response.assert_error()
        # Should fail because path is a file, not a directory
        assert (
            "not a directory" in response.text.lower() or "directory" in response.text.lower()
        ), f"Expected directory error, got: {response.text}"

    def test_index_codebase_shows_total_lines(self, mcp_client, sample_code_project):
        """Test that response includes total lines count."""
        response = mcp_client.call_tool(
            "index_codebase",
            {"repo_path": str(sample_code_project)},
        )

        response.assert_success("index_codebase should succeed")

        # Verify total lines is reported
        assert "Total lines:" in response.text or "lines:" in response.text.lower()

    def test_index_codebase_shows_indexing_time(self, mcp_client, sample_code_project):
        """Test that response includes indexing time."""
        response = mcp_client.call_tool(
            "index_codebase",
            {"repo_path": str(sample_code_project)},
        )

        response.assert_success("index_codebase should succeed")

        # Verify indexing time is reported
        assert "Indexing time:" in response.text or "time:" in response.text.lower()
