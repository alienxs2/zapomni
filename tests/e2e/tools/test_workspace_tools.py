"""
E2E tests for Workspace MCP tools.

Tests verify the complete flow through the MCP server via SSE transport for:
- create_workspace: Create new workspaces
- list_workspaces: List all workspaces
- set_current_workspace: Switch to a workspace
- get_current_workspace: Get current workspace info
- delete_workspace: Delete a workspace

Each test uses unique workspace IDs for test independence.

Author: Zapomni Test Suite
License: MIT
"""

import uuid

import pytest


@pytest.mark.e2e
@pytest.mark.workspace
class TestCreateWorkspace:
    """E2E tests for create_workspace tool."""

    def test_create_workspace_success(self, mcp_client, test_workspace_id):
        """Test successful workspace creation."""
        workspace_name = f"Test Workspace {test_workspace_id}"

        response = mcp_client.call_tool(
            "create_workspace",
            {
                "workspace_id": test_workspace_id,
                "name": workspace_name,
            },
        )

        response.assert_success("create_workspace should succeed with valid parameters")

        # Verify response contains success message and workspace info
        assert "created successfully" in response.text.lower()
        assert test_workspace_id in response.text
        assert workspace_name in response.text

        # Cleanup: delete the workspace
        mcp_client.call_tool(
            "delete_workspace",
            {
                "workspace_id": test_workspace_id,
                "confirm": True,
            },
        )
        # We don't assert cleanup success as workspace might already be deleted

    def test_create_workspace_with_description(self, mcp_client):
        """Test workspace creation with description."""
        workspace_id = f"test-desc-{uuid.uuid4().hex[:8]}"
        workspace_name = "Test Workspace With Description"
        description = "This is a test workspace for E2E testing with a detailed description."

        response = mcp_client.call_tool(
            "create_workspace",
            {
                "workspace_id": workspace_id,
                "name": workspace_name,
                "description": description,
            },
        )

        response.assert_success("create_workspace with description should succeed")

        # Verify response contains success message and all fields
        assert "created successfully" in response.text.lower()
        assert workspace_id in response.text
        assert workspace_name in response.text
        assert description in response.text

        # Cleanup
        mcp_client.call_tool(
            "delete_workspace",
            {"workspace_id": workspace_id, "confirm": True},
        )

    def test_create_workspace_invalid_id_fails(self, mcp_client):
        """Test invalid workspace_id format fails."""
        # Try creating with special characters not allowed in workspace_id
        invalid_ids = [
            "test@invalid",  # Contains @
            "test#invalid",  # Contains #
            "test invalid",  # Contains space
            "test.invalid",  # Contains .
        ]

        for invalid_id in invalid_ids:
            response = mcp_client.call_tool(
                "create_workspace",
                {
                    "workspace_id": invalid_id,
                    "name": "Test Workspace",
                },
            )

            response.assert_error(contains=None)  # Error message may vary

    def test_create_workspace_duplicate_fails(self, mcp_client):
        """Test creating duplicate workspace fails."""
        workspace_id = f"test-dup-{uuid.uuid4().hex[:8]}"
        workspace_name = "Duplicate Test Workspace"

        # Create workspace first time - should succeed
        first_response = mcp_client.call_tool(
            "create_workspace",
            {
                "workspace_id": workspace_id,
                "name": workspace_name,
            },
        )
        first_response.assert_success("First creation should succeed")

        # Try to create same workspace again - should fail
        second_response = mcp_client.call_tool(
            "create_workspace",
            {
                "workspace_id": workspace_id,
                "name": workspace_name,
            },
        )
        second_response.assert_error()

        # Cleanup
        mcp_client.call_tool(
            "delete_workspace",
            {"workspace_id": workspace_id, "confirm": True},
        )


@pytest.mark.e2e
@pytest.mark.workspace
class TestListWorkspaces:
    """E2E tests for list_workspaces tool."""

    def test_list_workspaces_returns_list(self, mcp_client):
        """Test list_workspaces returns a list of workspaces."""
        response = mcp_client.call_tool(
            "list_workspaces",
            {},
        )

        response.assert_success("list_workspaces should succeed")

        # Should contain workspace info (at least default workspace)
        assert (
            "workspace" in response.text.lower()
            or "default" in response.text.lower()
            or "Found" in response.text
        )

    def test_list_workspaces_includes_created(self, mcp_client, test_workspace_id):
        """Test newly created workspace appears in list."""
        workspace_name = f"Listed Workspace {test_workspace_id}"

        # Create a new workspace
        create_response = mcp_client.call_tool(
            "create_workspace",
            {
                "workspace_id": test_workspace_id,
                "name": workspace_name,
            },
        )
        create_response.assert_success("create_workspace should succeed")

        # List workspaces and verify our new workspace is included
        list_response = mcp_client.call_tool(
            "list_workspaces",
            {},
        )
        list_response.assert_success("list_workspaces should succeed")

        # Verify our workspace appears in the list
        assert (
            test_workspace_id in list_response.text
        ), f"Expected workspace {test_workspace_id} in list, got: {list_response.text}"

        # Cleanup
        mcp_client.call_tool(
            "delete_workspace",
            {"workspace_id": test_workspace_id, "confirm": True},
        )


@pytest.mark.e2e
@pytest.mark.workspace
class TestSetCurrentWorkspace:
    """E2E tests for set_current_workspace tool."""

    def test_set_current_workspace_success(self, mcp_client, test_workspace_id):
        """Test switching to workspace."""
        workspace_name = f"Switchable Workspace {test_workspace_id}"

        # Create workspace first
        create_response = mcp_client.call_tool(
            "create_workspace",
            {
                "workspace_id": test_workspace_id,
                "name": workspace_name,
            },
        )
        create_response.assert_success("create_workspace should succeed")

        # Switch to the workspace
        set_response = mcp_client.call_tool(
            "set_current_workspace",
            {"workspace_id": test_workspace_id},
        )

        set_response.assert_success("set_current_workspace should succeed")

        # Verify response contains confirmation
        assert test_workspace_id in set_response.text
        assert (
            "set to" in set_response.text.lower()
            or "current workspace" in set_response.text.lower()
        )

        # Cleanup
        mcp_client.call_tool(
            "delete_workspace",
            {"workspace_id": test_workspace_id, "confirm": True},
        )

    def test_set_current_workspace_nonexistent_fails(self, mcp_client):
        """Test switching to non-existent workspace fails."""
        nonexistent_id = f"nonexistent-{uuid.uuid4().hex[:8]}"

        response = mcp_client.call_tool(
            "set_current_workspace",
            {"workspace_id": nonexistent_id},
        )

        response.assert_error()
        assert "not found" in response.text.lower()


@pytest.mark.e2e
@pytest.mark.workspace
class TestGetCurrentWorkspace:
    """E2E tests for get_current_workspace tool."""

    def test_get_current_workspace_returns_info(self, mcp_client):
        """Test get_current_workspace returns valid info."""
        response = mcp_client.call_tool(
            "get_current_workspace",
            {},
        )

        response.assert_success("get_current_workspace should succeed")

        # Should contain workspace info with statistics
        assert "workspace" in response.text.lower()
        # Should include stats (memories, chunks, entities, relationships)
        assert "memories" in response.text.lower() or "statistics" in response.text.lower()

    @pytest.mark.xfail(
        reason="SSE sessions are stateless - workspace state may not persist between connections"
    )
    def test_get_current_workspace_after_switch(self, mcp_client, test_workspace_id):
        """Test current workspace changes after switch."""
        workspace_name = f"Current Test Workspace {test_workspace_id}"

        # Create workspace
        create_response = mcp_client.call_tool(
            "create_workspace",
            {
                "workspace_id": test_workspace_id,
                "name": workspace_name,
            },
        )
        create_response.assert_success("create_workspace should succeed")

        # Switch to the new workspace
        set_response = mcp_client.call_tool(
            "set_current_workspace",
            {"workspace_id": test_workspace_id},
        )
        set_response.assert_success("set_current_workspace should succeed")

        # Get current workspace and verify it's the one we switched to
        get_response = mcp_client.call_tool(
            "get_current_workspace",
            {},
        )

        get_response.assert_success("get_current_workspace should succeed")

        # Verify the current workspace is the one we switched to
        assert (
            test_workspace_id in get_response.text
        ), f"Expected current workspace to be {test_workspace_id}, got: {get_response.text}"

        # Cleanup
        mcp_client.call_tool(
            "delete_workspace",
            {"workspace_id": test_workspace_id, "confirm": True},
        )


@pytest.mark.e2e
@pytest.mark.workspace
class TestDeleteWorkspace:
    """E2E tests for delete_workspace tool."""

    def test_delete_workspace_success(self, mcp_client):
        """Test successful workspace deletion."""
        workspace_id = f"test-delete-{uuid.uuid4().hex[:8]}"
        workspace_name = "Deletable Workspace"

        # Step 1: Create workspace
        create_response = mcp_client.call_tool(
            "create_workspace",
            {
                "workspace_id": workspace_id,
                "name": workspace_name,
            },
        )
        create_response.assert_success("create_workspace should succeed")

        # Step 2: Delete with confirm=True
        delete_response = mcp_client.call_tool(
            "delete_workspace",
            {
                "workspace_id": workspace_id,
                "confirm": True,
            },
        )

        delete_response.assert_success("delete_workspace with confirm=True should succeed")
        assert (
            "deleted" in delete_response.text.lower()
        ), f"Expected deletion confirmation, got: {delete_response.text}"

        # Step 3: Verify not in list
        list_response = mcp_client.call_tool(
            "list_workspaces",
            {},
        )
        list_response.assert_success("list_workspaces should succeed")

        # Verify deleted workspace is not in the list
        assert (
            workspace_id not in list_response.text
        ), f"Deleted workspace {workspace_id} should not appear in list"

    def test_delete_workspace_without_confirm_fails(self, mcp_client, test_workspace_id):
        """Test deletion without confirm fails."""
        workspace_name = "No Confirm Delete Workspace"

        # Create workspace
        create_response = mcp_client.call_tool(
            "create_workspace",
            {
                "workspace_id": test_workspace_id,
                "name": workspace_name,
            },
        )
        create_response.assert_success("create_workspace should succeed")

        # Try to delete with confirm=False (or without confirm)
        delete_response = mcp_client.call_tool(
            "delete_workspace",
            {
                "workspace_id": test_workspace_id,
                "confirm": False,
            },
        )

        # Should return warning message (not is_error=True, but also not deleted)
        # The response should ask for confirmation
        assert (
            "confirm" in delete_response.text.lower()
            or "warning" in delete_response.text.lower()
            or "irreversible" in delete_response.text.lower()
        ), f"Expected confirmation warning, got: {delete_response.text}"

        # Verify workspace still exists
        list_response = mcp_client.call_tool(
            "list_workspaces",
            {},
        )
        assert (
            test_workspace_id in list_response.text
        ), "Workspace should still exist after failed delete"

        # Cleanup
        mcp_client.call_tool(
            "delete_workspace",
            {"workspace_id": test_workspace_id, "confirm": True},
        )

    def test_delete_workspace_nonexistent_fails(self, mcp_client):
        """Test deleting non-existent workspace fails."""
        nonexistent_id = f"nonexistent-del-{uuid.uuid4().hex[:8]}"

        response = mcp_client.call_tool(
            "delete_workspace",
            {
                "workspace_id": nonexistent_id,
                "confirm": True,
            },
        )

        # Should indicate workspace not found or was not found
        assert (
            "not found" in response.text.lower() or "was not found" in response.text.lower()
        ), f"Expected 'not found' message, got: {response.text}"

    def test_delete_default_workspace_fails(self, mcp_client):
        """Test deleting default workspace fails."""
        response = mcp_client.call_tool(
            "delete_workspace",
            {
                "workspace_id": "default",
                "confirm": True,
            },
        )

        response.assert_error()
        assert (
            "default" in response.text.lower() or "cannot delete" in response.text.lower()
        ), f"Expected error about default workspace, got: {response.text}"
