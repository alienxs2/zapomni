"""
Synchronous SSE client for Zapomni MCP E2E tests.

Provides a simple, synchronous interface for testing MCP tools via SSE transport.
Uses httpx for HTTP requests and implements the MCP JSON-RPC protocol.

This client uses a background thread to maintain the SSE connection and receive
responses asynchronously, while providing a synchronous API for ease of testing.

Author: Zapomni Test Suite
License: MIT
"""

import json
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx


@dataclass
class MCPResponse:
    """
    Structured response from MCP tool calls.

    Attributes:
        content: List of response content items (e.g., [{"type": "text", "text": "..."}])
        is_error: True if the tool returned an error
        raw: Raw JSON-RPC response from the server
    """

    content: List[Dict[str, Any]]
    is_error: bool
    raw: Dict[str, Any]

    @property
    def text(self) -> str:
        """
        Extract text from content items.

        Concatenates all text content from the response.
        Returns empty string if no text content found.

        Returns:
            Concatenated text from all content items
        """
        texts = []
        for item in self.content:
            if item.get("type") == "text" and "text" in item:
                texts.append(item["text"])
        return "\n".join(texts)

    def assert_success(self, msg: Optional[str] = None) -> None:
        """
        Assert that the response is not an error.

        Args:
            msg: Optional custom error message

        Raises:
            AssertionError: If is_error is True
        """
        if self.is_error:
            error_msg = msg or f"Expected success but got error: {self.text}"
            raise AssertionError(error_msg)

    def assert_error(self, contains: Optional[str] = None) -> None:
        """
        Assert that the response is an error.

        Args:
            contains: Optional substring that should be in the error text

        Raises:
            AssertionError: If is_error is False or text doesn't contain substring
        """
        if not self.is_error:
            raise AssertionError(f"Expected error but got success: {self.text}")

        if contains is not None:
            if contains not in self.text:
                raise AssertionError(f"Expected error to contain '{contains}' but got: {self.text}")


class MCPSSEClient:
    """
    Synchronous SSE client for MCP protocol communication.

    This client implements the SSE transport protocol for MCP:
    1. Connect to /sse endpoint to get a session_id
    2. Maintain SSE connection in background thread
    3. Send JSON-RPC tool calls to /messages/{session_id}
    4. Receive responses via SSE stream
    5. Parse responses into structured MCPResponse objects

    Example:
        >>> client = MCPSSEClient("http://localhost:8000")
        >>> session_id = client.connect()
        >>> response = client.call_tool("get_stats", {})
        >>> response.assert_success()
        >>> print(response.text)
        >>> client.close()

    Attributes:
        base_url: Base URL of the MCP server (e.g., "http://localhost:8000")
        timeout: Request timeout in seconds
        session_id: Current SSE session ID (set after connect())
    """

    def __init__(self, base_url: str, timeout: float = 30.0):
        """
        Initialize the SSE client.

        Args:
            base_url: Base URL of the MCP server (e.g., "http://localhost:8000")
            timeout: Request timeout in seconds (default: 30.0)
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session_id: Optional[str] = None
        self._endpoint_url: Optional[str] = None  # Full endpoint URL from server
        self._sse_client = httpx.Client(timeout=timeout)  # For SSE stream
        self._post_client = httpx.Client(timeout=timeout)  # For POST requests
        self._request_id = 0

        # Thread-safe queue for responses
        self._response_queue: queue.Queue = queue.Queue()
        self._sse_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._connected = threading.Event()

    def connect(self) -> str:
        """
        Establish SSE connection, initialize MCP session, and get session ID.

        Performs the full MCP initialization handshake:
        1. Server creates a new SSE session with unique session_id
        2. Server sends "endpoint" event with the session endpoint URL
        3. Client extracts session_id from the endpoint URL
        4. Client starts background thread to listen for responses
        5. Client sends "initialize" request with protocol version and capabilities
        6. Client sends "notifications/initialized" notification
        7. Session is now ready for tool calls

        The endpoint URL format is: /messages/?session_id={uuid}

        Returns:
            Session ID string (e.g., "abc123...")

        Raises:
            httpx.HTTPError: If connection fails
            ValueError: If session_id cannot be extracted from response
            TimeoutError: If connection times out
            RuntimeError: If initialization fails
        """
        # Start SSE listener thread
        self._sse_thread = threading.Thread(target=self._sse_listener, daemon=True)
        self._sse_thread.start()

        # Wait for connection to be established (with timeout)
        if not self._connected.wait(timeout=self.timeout):
            raise TimeoutError("SSE connection timeout")

        if not self.session_id:
            raise ValueError("Failed to extract session_id from SSE response")

        # Perform MCP initialization handshake
        self._initialize()

        return self.session_id

    def _initialize(self) -> None:
        """
        Perform MCP protocol initialization handshake.

        Sends initialize request followed by notifications/initialized.
        This must be called before any tool calls.

        Raises:
            RuntimeError: If initialization fails
            TimeoutError: If response timeout
        """
        # Step 1: Send initialize request
        self._request_id += 1
        init_request = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "roots": {"listChanged": True},
                    "sampling": {},
                },
                "clientInfo": {
                    "name": "zapomni-e2e-test",
                    "version": "1.0.0",
                },
            },
            "id": self._request_id,
        }

        url = f"{self.base_url}{self._endpoint_url}"
        response = self._post_client.post(
            url,
            json=init_request,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()

        # Wait for initialize response from SSE stream
        start_time = time.time()
        init_response = None
        while time.time() - start_time < self.timeout:
            try:
                response_data = self._response_queue.get(timeout=0.5)
                if response_data.get("id") == self._request_id:
                    init_response = response_data
                    break
                else:
                    # Put back if wrong ID
                    self._response_queue.put(response_data)
            except queue.Empty:
                continue

        if init_response is None:
            raise TimeoutError("Initialize response timeout")

        if "error" in init_response:
            raise RuntimeError(f"Initialize failed: {init_response['error']}")

        # Step 2: Send notifications/initialized notification (no response expected)
        init_notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        }

        response = self._post_client.post(
            url,
            json=init_notification,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> MCPResponse:
        """
        Call an MCP tool via JSON-RPC.

        Sends a JSON-RPC 2.0 request to /messages/{session_id} endpoint:
        {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "tool_name",
                "arguments": {...}
            },
            "id": 1
        }

        The response is received asynchronously via the SSE stream and
        matched by request ID.

        Args:
            name: Tool name (e.g., "get_stats", "add_memory")
            arguments: Tool arguments as a dictionary

        Returns:
            MCPResponse object with parsed response data

        Raises:
            RuntimeError: If not connected (call connect() first)
            httpx.HTTPError: If request fails
            TimeoutError: If response timeout
        """
        if not self.session_id:
            raise RuntimeError("Not connected. Call connect() first.")

        # Increment request ID
        self._request_id += 1
        request_id = self._request_id

        # Build JSON-RPC request
        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments,
            },
            "id": request_id,
        }

        # Send POST request to the endpoint URL provided by the server
        # This will be /messages/?session_id={uuid} or /messages?session_id={uuid}
        url = f"{self.base_url}{self._endpoint_url}"
        response = self._post_client.post(
            url,
            json=request,
            headers={"Content-Type": "application/json"},
        )

        # Check for HTTP errors
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            # Try to get error details from response body
            try:
                error_detail = response.json()
            except:
                error_detail = response.text
            raise RuntimeError(f"HTTP {response.status_code}: {error_detail}") from e

        # Wait for response from SSE stream
        start_time = time.time()
        while True:
            try:
                # Check for timeout
                if time.time() - start_time > self.timeout:
                    raise TimeoutError(f"Response timeout for request {request_id}")

                # Try to get response from queue (non-blocking with short timeout)
                response_data = self._response_queue.get(timeout=0.1)

                # Check if this is the response we're waiting for
                if response_data.get("id") == request_id:
                    return self._parse_response(response_data)
                else:
                    # Wrong response, put it back (shouldn't happen with serial requests)
                    self._response_queue.put(response_data)

            except queue.Empty:
                # No response yet, continue waiting
                continue

    def health_check(self) -> Dict[str, Any]:
        """
        Check server health status.

        Sends a GET request to /health endpoint which returns:
        {
            "status": "healthy",
            "version": "0.2.0",
            "transport": "sse",
            "active_connections": 0,
            "uptime_seconds": 123.45,
            "metrics": {...}
        }

        Returns:
            Health status dictionary

        Raises:
            httpx.HTTPError: If health check fails
        """
        url = f"{self.base_url}/health"
        response = self._post_client.get(url)
        response.raise_for_status()
        return response.json()

    def close(self) -> None:
        """
        Close the HTTP client and cleanup resources.

        This is safe to call multiple times (idempotent).
        """
        # Signal thread to stop
        self._stop_event.set()

        # Wait for thread to finish (with timeout)
        if self._sse_thread and self._sse_thread.is_alive():
            self._sse_thread.join(timeout=5.0)

        # Close HTTP clients
        if self._sse_client:
            self._sse_client.close()
        if self._post_client:
            self._post_client.close()

        self.session_id = None

    def _sse_listener(self) -> None:
        """
        Background thread that maintains SSE connection and receives responses.

        This thread runs continuously until stop_event is set, reading SSE
        events from the server and queuing responses for the main thread.
        """
        url = f"{self.base_url}/sse"

        try:
            with self._sse_client.stream("GET", url) as response:
                response.raise_for_status()

                # Read SSE events from the stream
                for line in response.iter_lines():
                    # Check if we should stop
                    if self._stop_event.is_set():
                        break

                    if not line:
                        continue

                    # Parse SSE event format
                    # event: endpoint
                    # data: /messages/{session_id}
                    if line.startswith("event:"):
                        event_type = line.split(":", 1)[1].strip()

                        if event_type == "endpoint":
                            # Next line should contain the endpoint data
                            continue

                    elif line.startswith("data:"):
                        data_str = line.split(":", 1)[1].strip()

                        # Check if this is the endpoint data
                        if data_str.startswith("/messages"):
                            # Extract full endpoint URL including query params
                            # Format: /messages/?session_id={uuid} or /messages?session_id={uuid}
                            if "?session_id=" in data_str:
                                # Store the full endpoint URL for POST requests
                                self._endpoint_url = data_str
                                # Also extract session ID for reference
                                session_id_part = data_str.split("?session_id=")[1]
                                # Remove any additional query parameters
                                if "&" in session_id_part:
                                    session_id_part = session_id_part.split("&")[0]
                                self.session_id = session_id_part
                                self._connected.set()
                            continue

                        # Try to parse as JSON-RPC response
                        try:
                            data = json.loads(data_str)

                            # Check if this is a JSON-RPC response or notification
                            if "jsonrpc" in data:
                                # If it's a response (has "id" field), queue it
                                if "id" in data:
                                    self._response_queue.put(data)
                                # If it's a notification (no "id" field), we can ignore it for now
                                # (e.g., heartbeat notifications)

                        except json.JSONDecodeError:
                            # Not JSON data, ignore
                            continue

        except Exception as e:
            # Log error but don't crash
            # In a real implementation, we'd use proper logging
            if not self._stop_event.is_set():
                print(f"SSE listener error: {e}", flush=True)

    def _parse_response(self, data: Dict[str, Any]) -> MCPResponse:
        """
        Parse JSON-RPC response into MCPResponse.

        Args:
            data: Raw JSON-RPC response dictionary

        Returns:
            MCPResponse object
        """
        # Check for JSON-RPC error
        if "error" in data:
            error = data["error"]
            return MCPResponse(
                content=[{"type": "text", "text": f"Error: {error.get('message', str(error))}"}],
                is_error=True,
                raw=data,
            )

        # Parse successful result
        result = data.get("result", {})
        content = result.get("content", [])

        # Check isError field from MCP CallToolResult (per MCP spec)
        is_error = result.get("isError", False)

        # Fallback: also check if content indicates an error (e.g., {"type": "text", "text": "Error: ..."})
        if not is_error and content:
            for item in content:
                if item.get("type") == "text":
                    text = item.get("text", "")
                    if text.startswith("Error:"):
                        is_error = True
                        break

        return MCPResponse(
            content=content,
            is_error=is_error,
            raw=data,
        )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


__all__ = ["MCPSSEClient", "MCPResponse"]
