"""
E2E test fixtures for Zapomni MCP server.

Provides pytest fixtures for end-to-end testing of the MCP server:
- mcp_client: Session-scoped SSE client connected to the server
- mcp_server_url: Server URL from environment or default
- test_workspace_id: Unique workspace ID for test isolation
- clean_workspace: Auto-cleanup workspace fixture
- sample_memory_text: Test data for memory operations
- sample_code_project: Path to test code project

Usage:
    Run E2E tests with a running MCP server:
    $ python -m zapomni_mcp --host 127.0.0.1 --port 8000
    $ pytest tests/e2e/ -v

Copyright (c) 2025 Goncharenko Anton aka alienxs2
License: MIT
"""

import logging
import os
import uuid
from pathlib import Path
from typing import Generator, Optional

import pytest

from tests.e2e.sse_client import MCPSSEClient

# Configure logging for E2E tests
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("e2e.fixtures")


# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers for E2E tests."""
    config.addinivalue_line(
        "markers",
        "e2e: mark test as end-to-end test (requires running MCP server)",
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow (may take more than 5 seconds)",
    )
    config.addinivalue_line(
        "markers",
        "workspace: mark test as using workspace isolation",
    )
    config.addinivalue_line(
        "markers",
        "workflow: mark test as workflow integration test",
    )


# =============================================================================
# Server URL Fixture
# =============================================================================


@pytest.fixture(scope="session")
def mcp_server_url() -> str:
    """
    Get the MCP server URL from environment or use default.

    Environment variable: MCP_SERVER_URL
    Default: http://127.0.0.1:8000

    Returns:
        Server URL string
    """
    url = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000")
    logger.info(f"MCP server URL: {url}")
    return url


# =============================================================================
# Server Availability Check
# =============================================================================


@pytest.fixture(scope="session", autouse=True)
def skip_if_server_unavailable(mcp_server_url: str) -> Generator[None, None, None]:
    """
    Check if MCP server is available before running E2E tests.

    This fixture runs automatically for all E2E tests.
    If the server is not reachable, all tests in the session are skipped.

    Args:
        mcp_server_url: Server URL from fixture

    Yields:
        None if server is available

    Raises:
        pytest.skip: If server is not available
    """
    logger.info(f"Checking server availability at {mcp_server_url}")

    client = MCPSSEClient(mcp_server_url, timeout=10.0)
    try:
        # Try to check health endpoint first (quick check)
        health = client.health_check()
        logger.info(f"Server health: {health.get('status', 'unknown')}")
        logger.info(f"Server version: {health.get('version', 'unknown')}")
    except Exception as e:
        logger.warning(f"Server not available: {e}")
        pytest.skip(
            f"MCP server not available at {mcp_server_url}. "
            f"Start server with: python -m zapomni_mcp --host 127.0.0.1 --port 8000"
        )
    finally:
        client.close()

    yield


# =============================================================================
# MCP Client Fixture
# =============================================================================


@pytest.fixture(scope="session")
def mcp_client(mcp_server_url: str) -> Generator[MCPSSEClient, None, None]:
    """
    Create and connect an MCP SSE client for the test session.

    This client is shared across all E2E tests in the session.
    Connection is established once at session start and closed at session end.

    Args:
        mcp_server_url: Server URL from fixture

    Yields:
        Connected MCPSSEClient instance

    Raises:
        RuntimeError: If connection fails
    """
    logger.info(f"Creating MCP client for {mcp_server_url}")

    client = MCPSSEClient(mcp_server_url, timeout=30.0)

    try:
        session_id = client.connect()
        logger.info(f"MCP client connected, session_id: {session_id}")
    except Exception as e:
        logger.error(f"Failed to connect MCP client: {e}")
        client.close()
        raise RuntimeError(f"Failed to connect to MCP server: {e}") from e

    yield client

    # Cleanup
    logger.info("Closing MCP client")
    client.close()


# =============================================================================
# Workspace Fixtures
# =============================================================================


@pytest.fixture(scope="function")
def test_workspace_id() -> str:
    """
    Generate a unique workspace ID for each test.

    Format: test-{uuid[:8]}
    Example: test-a1b2c3d4

    Returns:
        Unique workspace ID string
    """
    workspace_id = f"test-{uuid.uuid4().hex[:8]}"
    logger.debug(f"Generated test workspace ID: {workspace_id}")
    return workspace_id


@pytest.fixture(scope="function")
def clean_workspace(
    mcp_client: MCPSSEClient,
    test_workspace_id: str,
) -> Generator[str, None, None]:
    """
    Create a clean workspace before test and delete it after.

    This fixture provides test isolation by:
    1. Creating a new workspace with unique ID before each test
    2. Setting it as the current workspace
    3. Deleting the workspace after the test completes

    Args:
        mcp_client: Connected MCP client
        test_workspace_id: Unique workspace ID

    Yields:
        Workspace ID string

    Note:
        Cleanup happens even if the test fails.
    """
    workspace_id = test_workspace_id
    workspace_name = f"Test Workspace {workspace_id}"

    logger.info(f"Creating clean workspace: {workspace_id}")

    # Create workspace
    try:
        response = mcp_client.call_tool(
            "create_workspace",
            {
                "workspace_id": workspace_id,
                "name": workspace_name,
                "description": "Temporary workspace for E2E testing",
            },
        )
        if response.is_error:
            logger.warning(f"Failed to create workspace: {response.text}")
        else:
            logger.debug(f"Workspace created: {workspace_id}")
    except Exception as e:
        logger.warning(f"Exception creating workspace: {e}")

    # Set as current workspace
    try:
        response = mcp_client.call_tool(
            "set_current_workspace",
            {"workspace_id": workspace_id},
        )
        if response.is_error:
            logger.warning(f"Failed to set workspace: {response.text}")
        else:
            logger.debug(f"Workspace set as current: {workspace_id}")
    except Exception as e:
        logger.warning(f"Exception setting workspace: {e}")

    yield workspace_id

    # Cleanup: Delete workspace
    logger.info(f"Cleaning up workspace: {workspace_id}")
    try:
        response = mcp_client.call_tool(
            "delete_workspace",
            {
                "workspace_id": workspace_id,
                "confirm": True,
            },
        )
        if response.is_error:
            logger.warning(f"Failed to delete workspace: {response.text}")
        else:
            logger.debug(f"Workspace deleted: {workspace_id}")
    except Exception as e:
        logger.warning(f"Exception deleting workspace: {e}")


# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def sample_memory_text() -> str:
    """
    Return sample text for testing add_memory and search_memory.

    This text contains keywords suitable for semantic search testing.

    Returns:
        Sample text string for memory operations
    """
    return (
        "Python is a high-level, general-purpose programming language. "
        "Its design philosophy emphasizes code readability with the use of "
        "significant indentation. Python is dynamically typed and garbage-collected. "
        "It supports multiple programming paradigms, including structured, "
        "object-oriented and functional programming. Python was conceived in the "
        "late 1980s by Guido van Rossum at Centrum Wiskunde & Informatica (CWI) "
        "in the Netherlands as a successor to the ABC programming language."
    )


@pytest.fixture(scope="session")
def sample_code_project() -> Path:
    """
    Return path to sample Python project for code indexing tests.

    The sample project is located at:
    tests/e2e/fixtures/sample_code/python_project

    Returns:
        Path object to the sample code project directory

    Raises:
        FileNotFoundError: If the sample project directory does not exist
    """
    project_path = Path(__file__).parent / "fixtures" / "sample_code" / "python_project"

    if not project_path.exists():
        raise FileNotFoundError(
            f"Sample code project not found at: {project_path}. "
            f"Please create the test fixture directory."
        )

    logger.debug(f"Sample code project path: {project_path}")
    return project_path


# =============================================================================
# Additional Helper Fixtures
# =============================================================================


@pytest.fixture(scope="function")
def unique_memory_id() -> str:
    """
    Generate a unique ID for memory operations.

    Useful when tests need to create memories with identifiable content.

    Returns:
        Unique ID string (UUID format)
    """
    return str(uuid.uuid4())


@pytest.fixture(scope="session")
def sample_graph_text() -> str:
    """
    Return sample text for testing build_graph and get_related.

    This text contains entities and relationships suitable for graph extraction.

    Returns:
        Sample text string for graph operations
    """
    return (
        "Albert Einstein was a German-born theoretical physicist. "
        "He developed the theory of relativity, one of the two pillars of modern physics. "
        "Einstein received the Nobel Prize in Physics in 1921 for his discovery "
        "of the law of the photoelectric effect. "
        "He worked at the Institute for Advanced Study in Princeton, New Jersey."
    )


@pytest.fixture(scope="session")
def e2e_test_dir() -> Path:
    """
    Return the path to the E2E tests directory.

    Returns:
        Path object to tests/e2e/
    """
    return Path(__file__).parent


# =============================================================================
# Markers for conditional skipping
# =============================================================================


@pytest.fixture(scope="session")
def redis_enabled() -> bool:
    """
    Check if Redis is enabled for semantic cache testing.

    Reads from environment variable REDIS_ENABLED.

    Returns:
        True if Redis is enabled, False otherwise
    """
    return os.getenv("REDIS_ENABLED", "true").lower() == "true"


@pytest.fixture(scope="session")
def semantic_cache_enabled() -> bool:
    """
    Check if semantic cache is enabled.

    Reads from environment variable ENABLE_SEMANTIC_CACHE.

    Returns:
        True if semantic cache is enabled, False otherwise
    """
    return os.getenv("ENABLE_SEMANTIC_CACHE", "true").lower() == "true"
