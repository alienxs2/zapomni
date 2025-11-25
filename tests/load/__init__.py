"""
Load tests for Zapomni MCP Server.

Uses Locust framework for load testing SSE transport endpoints.

Usage:
    # Start the server first
    zapomni-mcp --host 127.0.0.1 --port 8000

    # Run locust with web UI
    cd tests/load
    locust -f locustfile.py --host=http://127.0.0.1:8000

    # Run headless for CI/CD
    locust -f locustfile.py --host=http://127.0.0.1:8000 \
        --headless -u 50 -r 10 --run-time 5m

Performance targets (from specification):
    - 50+ concurrent connections
    - Response time < 500ms (P95)
    - No errors under normal load
    - 30 minute stability test

Author: Goncharenko Anton aka alienxs2
License: MIT
"""
