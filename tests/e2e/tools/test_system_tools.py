"""
E2E tests for System MCP tools (get_stats, set_model, prune_memory, clear_all).

Tests verify the complete flow through the MCP server via SSE transport.
System tools provide configuration, statistics, and maintenance operations.

IMPORTANT: Some tests are intentionally skipped or limited for safety:
- clear_all success test is NOT included (would destroy all data)
- prune_memory tests use dry_run=true only
- set_model tests use known models only

Author: Zapomni Test Suite
License: MIT
"""

import pytest


@pytest.mark.e2e
class TestGetStats:
    """E2E tests for get_stats tool."""

    def test_get_stats_returns_statistics(self, mcp_client):
        """Test get_stats returns valid statistics."""
        response = mcp_client.call_tool("get_stats", {})

        response.assert_success("get_stats should succeed")

        # Verify essential fields are present
        assert "Total Memories:" in response.text
        assert "Total Chunks:" in response.text

    def test_get_stats_format(self, mcp_client):
        """Test get_stats response format contains expected structure."""
        response = mcp_client.call_tool("get_stats", {})

        response.assert_success("get_stats should succeed")

        # Should have header and formatted fields
        assert "Memory System Statistics:" in response.text
        assert "Database Size:" in response.text
        assert "MB" in response.text  # Database size unit

    def test_get_stats_performance_metrics(self, mcp_client):
        """Test get_stats includes performance metrics when available."""
        response = mcp_client.call_tool("get_stats", {})

        response.assert_success("get_stats should succeed")

        # Check for optional performance metrics (may or may not be present)
        # At minimum, basic stats should be present
        text = response.text
        assert "Total Memories:" in text
        assert "Total Chunks:" in text
        assert "Average Chunks per Memory:" in text

        # SSE metrics may be present if session manager is available
        # (Optional - don't fail if not present)

    def test_get_stats_no_parameters_required(self, mcp_client):
        """Test get_stats works with empty parameters."""
        # Empty dict should work
        response = mcp_client.call_tool("get_stats", {})
        response.assert_success("get_stats should work with empty params")

        # Should return same stats regardless of empty params
        assert "Total Memories:" in response.text


@pytest.mark.e2e
class TestSetModel:
    """E2E tests for set_model tool."""

    def test_set_model_success(self, mcp_client):
        """Test setting existing Ollama model."""
        # Use qwen2.5:latest which is known to exist in the test environment
        response = mcp_client.call_tool(
            "set_model",
            {"model_name": "qwen2.5:latest"},
        )

        response.assert_success("set_model with valid model should succeed")

        # Response should confirm the change
        assert "qwen2.5:latest" in response.text
        # Should mention what the model is used for
        assert "entity" in response.text.lower() or "relationship" in response.text.lower()

    def test_set_model_shows_previous_model(self, mcp_client):
        """Test set_model shows previous and new model names."""
        # Set to a model (may be same as current, that's OK)
        response = mcp_client.call_tool(
            "set_model",
            {"model_name": "qwen2.5:latest"},
        )

        response.assert_success("set_model should succeed")

        # Response should show model change information
        text = response.text.lower()
        assert "model" in text
        # Should mention what changed (from X to Y)
        assert "to" in text or "changed" in text

    def test_set_model_empty_name_fails(self, mcp_client):
        """Test empty model name fails."""
        response = mcp_client.call_tool(
            "set_model",
            {"model_name": ""},
        )

        response.assert_error()
        # Should indicate the name cannot be empty
        assert "empty" in response.text.lower() or "error" in response.text.lower()

    def test_set_model_whitespace_name_fails(self, mcp_client):
        """Test whitespace-only model name fails."""
        response = mcp_client.call_tool(
            "set_model",
            {"model_name": "   "},
        )

        response.assert_error()
        # Should fail because name becomes empty after strip


@pytest.mark.e2e
class TestPruneMemory:
    """E2E tests for prune_memory tool."""

    def test_prune_memory_dry_run_default(self, mcp_client):
        """Test prune_memory in dry_run mode (default)."""
        # dry_run=true by default - safe to run
        response = mcp_client.call_tool(
            "prune_memory",
            {},
        )

        response.assert_success("prune_memory dry_run should succeed")

        # Should indicate this is a dry run / preview
        text = response.text.lower()
        assert "preview" in text or "dry run" in text or "dry_run" in text

    def test_prune_memory_dry_run_explicit(self, mcp_client):
        """Test prune_memory with explicit dry_run=true."""
        response = mcp_client.call_tool(
            "prune_memory",
            {"dry_run": True},
        )

        response.assert_success("prune_memory with dry_run=true should succeed")

        # Should show what would be deleted
        text = response.text
        assert "delete" in text.lower() or "prune" in text.lower()

    def test_prune_memory_with_strategy_stale_code(self, mcp_client):
        """Test prune_memory with stale_code strategy."""
        response = mcp_client.call_tool(
            "prune_memory",
            {
                "strategy": "stale_code",
                "dry_run": True,
            },
        )

        response.assert_success("prune_memory with stale_code strategy should succeed")

        # Response should mention the strategy
        assert "stale_code" in response.text.lower() or "stale" in response.text.lower()

    def test_prune_memory_with_strategy_orphaned_chunks(self, mcp_client):
        """Test prune_memory with orphaned_chunks strategy."""
        response = mcp_client.call_tool(
            "prune_memory",
            {
                "strategy": "orphaned_chunks",
                "dry_run": True,
            },
        )

        response.assert_success("prune_memory with orphaned_chunks strategy should succeed")

        # Response should mention the strategy
        assert "orphan" in response.text.lower() or "chunk" in response.text.lower()

    def test_prune_memory_with_strategy_orphaned_entities(self, mcp_client):
        """Test prune_memory with orphaned_entities strategy."""
        response = mcp_client.call_tool(
            "prune_memory",
            {
                "strategy": "orphaned_entities",
                "dry_run": True,
            },
        )

        response.assert_success("prune_memory with orphaned_entities strategy should succeed")

        # Response should mention entities
        assert "entity" in response.text.lower() or "entities" in response.text.lower()

    def test_prune_memory_with_strategy_all(self, mcp_client):
        """Test prune_memory with 'all' strategy (all strategies combined)."""
        response = mcp_client.call_tool(
            "prune_memory",
            {
                "strategy": "all",
                "dry_run": True,
            },
        )

        response.assert_success("prune_memory with 'all' strategy should succeed")

        # Response should summarize all strategies
        text = response.text.lower()
        assert "all" in text or ("memory" in text and "chunk" in text)

    def test_prune_memory_confirm_required_for_deletion(self, mcp_client):
        """Test actual prune requires confirm=True."""
        # dry_run=false without confirm should fail
        response = mcp_client.call_tool(
            "prune_memory",
            {
                "dry_run": False,
                "confirm": False,
            },
        )

        response.assert_error()

        # Should require confirmation
        text = response.text.lower()
        assert "confirm" in text or "confirmation" in text


@pytest.mark.e2e
class TestClearAll:
    """
    E2E tests for clear_all tool.

    IMPORTANT: We only test failure cases here.
    Testing successful clear_all would destroy all test data.
    """

    def test_clear_all_wrong_phrase_fails(self, mcp_client):
        """Test clear_all with wrong confirm_phrase fails."""
        # Wrong case - should fail
        response = mcp_client.call_tool(
            "clear_all",
            {"confirm_phrase": "delete all memories"},  # lowercase - wrong
        )

        response.assert_error()

        # Should indicate phrase is wrong
        text = response.text.lower()
        assert "confirm" in text or "phrase" in text or "invalid" in text

    def test_clear_all_partial_phrase_fails(self, mcp_client):
        """Test clear_all with partial phrase fails."""
        response = mcp_client.call_tool(
            "clear_all",
            {"confirm_phrase": "DELETE ALL"},  # incomplete
        )

        response.assert_error()

        # Should indicate phrase is wrong
        text = response.text
        assert "Error" in text or "error" in text.lower()

    def test_clear_all_empty_phrase_fails(self, mcp_client):
        """Test clear_all with empty phrase fails."""
        response = mcp_client.call_tool(
            "clear_all",
            {"confirm_phrase": ""},
        )

        response.assert_error()

        # Should indicate phrase is required or empty
        text = response.text.lower()
        assert "empty" in text or "required" in text or "confirm" in text

    def test_clear_all_missing_phrase_fails(self, mcp_client):
        """Test clear_all without confirm_phrase parameter fails."""
        response = mcp_client.call_tool(
            "clear_all",
            {},  # Missing required parameter
        )

        response.assert_error()

        # Should indicate missing required parameter
        text = response.text.lower()
        assert "required" in text or "confirm" in text or "missing" in text

    def test_clear_all_random_phrase_fails(self, mcp_client):
        """Test clear_all with random wrong phrase fails."""
        response = mcp_client.call_tool(
            "clear_all",
            {"confirm_phrase": "yes please delete everything now"},
        )

        response.assert_error()

        # Should indicate exact phrase required
        text = response.text
        # The error message should mention the required phrase
        assert "DELETE ALL MEMORIES" in text or "Invalid" in text or "error" in text.lower()

    # NOTE: We intentionally DO NOT test successful clear_all
    # def test_clear_all_success(self, mcp_client):
    #     """DO NOT RUN: Would destroy all test data."""
    #     pass
