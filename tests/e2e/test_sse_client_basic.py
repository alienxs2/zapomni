#!/usr/bin/env python3
"""
Basic test script for SSE client.

Tests the SSE client functionality by:
1. Starting the Zapomni MCP server in background
2. Testing health_check() endpoint
3. Testing call_tool("get_stats", {})
4. Stopping the server

This script is for development/testing purposes only.
"""

import os
import subprocess
import sys
import time

# Add parent directory to path to import sse_client
sys.path.insert(0, os.path.dirname(__file__))

from sse_client import MCPSSEClient  # noqa: E402


def test_sse_client():
    """Test basic SSE client functionality."""

    print("=" * 60)
    print("SSE Client Basic Test")
    print("=" * 60)

    # Step 1: Start the server
    print("\n[1/5] Starting Zapomni MCP server in background...")
    server_process = subprocess.Popen(
        ["python3", "-m", "zapomni_mcp", "--transport", "sse"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd="/home/dev/zapomni",
    )

    # Wait for server to start
    print("      Waiting for server to start (5 seconds)...")
    time.sleep(5)

    try:
        # Check if server is still running
        if server_process.poll() is not None:
            stdout, stderr = server_process.communicate()
            print("\n[ERROR] Server failed to start!")
            print(f"STDOUT: {stdout.decode()}")
            print(f"STDERR: {stderr.decode()}")
            return False

        print("      Server started successfully!")

        # Step 2: Test health check
        print("\n[2/5] Testing health_check()...")
        client = MCPSSEClient("http://localhost:8000", timeout=10.0)

        try:
            health = client.health_check()
            print(f"      Status: {health.get('status')}")
            print(f"      Version: {health.get('version')}")
            print(f"      Transport: {health.get('transport')}")
            print(f"      Active connections: {health.get('active_connections')}")
            print(f"      Uptime: {health.get('uptime_seconds')} seconds")

            if health.get("status") != "healthy":
                print("\n[ERROR] Server is not healthy!")
                return False

            print("      Health check passed!")

        except Exception as e:
            print(f"\n[ERROR] Health check failed: {e}")
            return False

        # Step 3: Connect to SSE
        print("\n[3/5] Connecting to SSE endpoint...")
        try:
            session_id = client.connect()
            print(f"      Session ID: {session_id}")
            print("      Connected successfully!")
        except Exception as e:
            print(f"\n[ERROR] Connection failed: {e}")
            import traceback

            traceback.print_exc()
            return False

        # Step 4: Call get_stats tool
        print("\n[4/5] Testing call_tool('get_stats', {})...")
        try:
            response = client.call_tool("get_stats", {})

            print("      Response received!")
            print(f"      Is error: {response.is_error}")
            print(f"      Content items: {len(response.content)}")
            print(f"      Full text: {response.text}")
            print(f"      Raw response: {response.raw}")

            # Assert success
            response.assert_success("get_stats should not return error")
            print("      Success assertion passed!")

            # Check that response contains expected data
            text = response.text
            if "total_memories" not in text:
                print("\n[WARNING] Response doesn't contain 'total_memories'")
            else:
                print("      Response contains expected data!")

        except Exception as e:
            print(f"\n[ERROR] Tool call failed: {e}")
            import traceback

            traceback.print_exc()
            return False

        # Step 5: Cleanup
        print("\n[5/5] Cleaning up...")
        client.close()
        print("      Client closed!")

        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)

        return True

    finally:
        # Always stop the server
        print("\nStopping server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()
            server_process.wait()
        print("Server stopped!")


if __name__ == "__main__":
    success = test_sse_client()
    sys.exit(0 if success else 1)
