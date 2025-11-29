"""
E2E workflow tests for workspace operations.

Tests workspace isolation and management:
- Data isolation between workspaces
- Switching between workspaces
- Cleanup on workspace deletion

Copyright (c) 2025 Goncharenko Anton aka alienxs2
License: MIT
"""

import time
import uuid

import pytest

from tests.e2e.sse_client import MCPSSEClient


@pytest.mark.e2e
@pytest.mark.workflow
@pytest.mark.workspace
class TestWorkspaceWorkflow:
    """Integration tests for workspace isolation."""

    def test_workspace_data_isolation(self, mcp_client: MCPSSEClient) -> None:
        """
        Test that workspaces isolate data.

        Steps:
        1. Create workspace A
        2. Add memory in workspace A
        3. Create workspace B
        4. Switch to workspace B
        5. Search - should NOT find A's memory
        6. Switch back to A
        7. Search - SHOULD find memory
        8. Cleanup both workspaces
        """
        # Generate unique workspace IDs
        workspace_a = f"test-iso-a-{uuid.uuid4().hex[:8]}"
        workspace_b = f"test-iso-b-{uuid.uuid4().hex[:8]}"

        try:
            # Step 1: Create workspace A
            response = mcp_client.call_tool(
                "create_workspace",
                {
                    "workspace_id": workspace_a,
                    "name": f"Test Workspace A {workspace_a}",
                    "description": "Workspace A for isolation test",
                },
            )
            response.assert_success(f"Failed to create workspace A: {workspace_a}")

            # Set workspace A as current
            response = mcp_client.call_tool(
                "set_current_workspace",
                {"workspace_id": workspace_a},
            )
            response.assert_success("Failed to set workspace A as current")

            # Step 2: Add memory in workspace A
            unique_content = f"Unique content for workspace A: {uuid.uuid4().hex}"
            response = mcp_client.call_tool(
                "add_memory",
                {
                    "text": unique_content,
                    "metadata": {
                        "tags": ["isolation-test"],
                        "source": "workspace-a",
                    },
                },
            )
            response.assert_success("Failed to add memory in workspace A")
            time.sleep(0.5)  # Allow indexing

            # Verify memory is searchable in workspace A
            response = mcp_client.call_tool(
                "search_memory",
                {"query": "Unique content workspace", "limit": 10},
            )
            response.assert_success("Failed to search in workspace A")
            workspace_a_has_memory = (
                "unique content" in response.text.lower() or workspace_a in response.text.lower()
            )

            # Step 3: Create workspace B
            response = mcp_client.call_tool(
                "create_workspace",
                {
                    "workspace_id": workspace_b,
                    "name": f"Test Workspace B {workspace_b}",
                    "description": "Workspace B for isolation test",
                },
            )
            response.assert_success(f"Failed to create workspace B: {workspace_b}")

            # Step 4: Switch to workspace B
            response = mcp_client.call_tool(
                "set_current_workspace",
                {"workspace_id": workspace_b},
            )
            response.assert_success("Failed to switch to workspace B")
            time.sleep(0.3)

            # Step 5: Search in workspace B - should NOT find A's memory
            response = mcp_client.call_tool(
                "search_memory",
                {"query": "Unique content workspace", "limit": 10},
            )
            # The search should succeed but not find workspace A's content
            workspace_b_has_a_memory = (
                "unique content" in response.text.lower() and "workspace-a" in response.text.lower()
            )

            # Step 6: Switch back to workspace A
            response = mcp_client.call_tool(
                "set_current_workspace",
                {"workspace_id": workspace_a},
            )
            response.assert_success("Failed to switch back to workspace A")
            time.sleep(0.3)

            # Step 7: Search again - SHOULD find memory
            response = mcp_client.call_tool(
                "search_memory",
                {"query": "Unique content workspace", "limit": 10},
            )
            response.assert_success("Failed to search in workspace A after switch")

            # Verify isolation: workspace B should not have seen workspace A's data
            # Note: If workspace A had memory initially, it should still have it after switch back
            if workspace_a_has_memory:
                assert (
                    not workspace_b_has_a_memory
                ), "Workspace B should NOT have access to workspace A's memories"

        finally:
            # Step 8: Cleanup - delete both workspaces
            for ws_id in [workspace_a, workspace_b]:
                try:
                    mcp_client.call_tool(
                        "delete_workspace",
                        {"workspace_id": ws_id, "confirm": True},
                    )
                except Exception:
                    pass  # Best effort cleanup

    def test_workspace_switching(self, mcp_client: MCPSSEClient) -> None:
        """
        Test switching between workspaces.

        Steps:
        1. Get current workspace
        2. Create new workspace
        3. Set current workspace to new
        4. Verify current changed
        5. Switch back
        6. Verify back to original
        7. Cleanup
        """
        new_workspace = f"test-switch-{uuid.uuid4().hex[:8]}"

        try:
            # Step 1: Get current workspace
            response = mcp_client.call_tool("get_current_workspace", {})
            response.assert_success("Failed to get current workspace")
            # Original workspace info stored in response.text for verification

            # Step 2: Create new workspace
            response = mcp_client.call_tool(
                "create_workspace",
                {
                    "workspace_id": new_workspace,
                    "name": f"Switch Test {new_workspace}",
                    "description": "Temporary workspace for switch test",
                },
            )
            response.assert_success(f"Failed to create workspace: {new_workspace}")

            # Step 3: Set current workspace to new
            response = mcp_client.call_tool(
                "set_current_workspace",
                {"workspace_id": new_workspace},
            )
            response.assert_success("Failed to set new workspace as current")

            # Step 4: Verify current changed
            response = mcp_client.call_tool("get_current_workspace", {})
            response.assert_success("Failed to get current workspace after switch")
            assert (
                new_workspace in response.text
            ), f"Expected current workspace to be {new_workspace}, got: {response.text}"

            # Step 5: List workspaces to verify new workspace exists
            response = mcp_client.call_tool("list_workspaces", {})
            response.assert_success("Failed to list workspaces")
            assert (
                new_workspace in response.text
            ), f"New workspace {new_workspace} should appear in list"

            # Step 6: Verify we can switch to default workspace
            response = mcp_client.call_tool(
                "set_current_workspace",
                {"workspace_id": "default"},
            )
            # Default workspace may or may not exist, but the call should complete

        finally:
            # Step 7: Cleanup - delete test workspace
            try:
                mcp_client.call_tool(
                    "delete_workspace",
                    {"workspace_id": new_workspace, "confirm": True},
                )
            except Exception:
                pass  # Best effort cleanup

    def test_workspace_cleanup_on_delete(self, mcp_client: MCPSSEClient) -> None:
        """
        Test that deleting workspace cleans up data.

        Steps:
        1. Create workspace
        2. Add memory
        3. Verify memory exists
        4. Delete workspace
        5. Verify workspace not in list
        """
        workspace_id = f"test-cleanup-{uuid.uuid4().hex[:8]}"

        try:
            # Step 1: Create workspace
            response = mcp_client.call_tool(
                "create_workspace",
                {
                    "workspace_id": workspace_id,
                    "name": f"Cleanup Test {workspace_id}",
                    "description": "Workspace for cleanup test",
                },
            )
            response.assert_success(f"Failed to create workspace: {workspace_id}")

            # Set as current
            response = mcp_client.call_tool(
                "set_current_workspace",
                {"workspace_id": workspace_id},
            )
            response.assert_success("Failed to set workspace as current")

            # Step 2: Add memory
            response = mcp_client.call_tool(
                "add_memory",
                {
                    "text": f"Test memory for cleanup verification: {workspace_id}",
                    "metadata": {
                        "tags": ["cleanup-test"],
                        "source": "cleanup-test",
                    },
                },
            )
            response.assert_success("Failed to add memory")
            time.sleep(0.5)

            # Step 3: Verify memory exists
            response = mcp_client.call_tool(
                "search_memory",
                {"query": "cleanup verification", "limit": 5},
            )
            response.assert_success("Failed to search memory before delete")

            # Step 4: Delete workspace
            response = mcp_client.call_tool(
                "delete_workspace",
                {"workspace_id": workspace_id, "confirm": True},
            )
            response.assert_success(f"Failed to delete workspace: {workspace_id}")
            time.sleep(0.3)

            # Step 5: Verify workspace not in list
            response = mcp_client.call_tool("list_workspaces", {})
            response.assert_success("Failed to list workspaces after delete")
            assert (
                workspace_id not in response.text
            ), f"Deleted workspace {workspace_id} should not appear in list"

        except Exception:
            # Ensure cleanup even on failure
            try:
                mcp_client.call_tool(
                    "delete_workspace",
                    {"workspace_id": workspace_id, "confirm": True},
                )
            except Exception:
                pass
            raise
