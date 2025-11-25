# Requirements Document - SSE Transport for Concurrent Connections

**Spec Name:** sse-transport
**Status:** Requirements Phase
**Created:** 2025-11-25
**Dependencies:** Phase 3 MCP Tools (complete)

---

## 1. Executive Summary

### 1.1 Problem Statement

The current Zapomni MCP server uses stdio transport which has critical limitations:

1. **Single-threaded blocking**: Stdio blocks on read/write, processing only one request at a time
2. **Exclusive access**: Only one parent process can connect (e.g., Claude Desktop OR VS Code, not both)
3. **No parallel processing**: Heavy requests block all other clients

### 1.2 Solution Overview

Migrate to SSE (Server-Sent Events) over HTTP using Starlette/Uvicorn to enable:

- **Async processing**: Multiple incoming connections processed concurrently
- **Parallel MCP clients**: Multiple clients share single knowledge base simultaneously
- **Non-blocking I/O**: Heavy requests don't block other clients

### 1.3 Key Deliverables

| Deliverable | Description |
|------------|-------------|
| SSE Transport Layer | HTTP server with `/sse` and `/messages/{session_id}` endpoints |
| Dual Transport Support | CLI flag `--transport stdio|sse` (default: sse) |
| Async Entity Extraction | ThreadPoolExecutor wrapper for CPU-bound SpaCy operations |
| Connection Pool Scaling | FalkorDB pool default of 20 connections (configurable) |
| Session Management | UUID-based session tracking with automatic cleanup |

---

## 2. User Stories

### US-1: Multiple Client Connection

```
AS a developer using multiple AI tools
I WANT to connect both Claude Desktop and VS Code to Zapomni simultaneously
SO THAT I can use my knowledge base from any AI assistant without disconnecting others

GIVEN Zapomni server is running with SSE transport
WHEN I connect Claude Desktop and VS Code to the same server
THEN both clients can send requests and receive responses independently
AND neither client blocks the other
```

**Acceptance Criteria:**
- [ ] Server accepts multiple concurrent SSE connections
- [ ] Each connection gets unique session ID
- [ ] Disconnecting one client does not affect others
- [ ] Both clients can execute tools simultaneously

### US-2: Non-Blocking Heavy Operations

```
AS an AI assistant user
I WANT heavy operations (build_graph, search_memory) to not block other requests
SO THAT quick queries return fast even when large operations are in progress

GIVEN Client A is running a build_graph operation on a large document
WHEN Client B sends a quick get_stats request
THEN Client B receives response immediately (< 500ms)
AND Client A's operation continues uninterrupted
```

**Acceptance Criteria:**
- [ ] CPU-bound operations run in ThreadPoolExecutor
- [ ] Event loop remains responsive during heavy processing
- [ ] Response latency for simple operations < 500ms under load
- [ ] No request starvation occurs

### US-3: Backward Compatibility

```
AS a current Zapomni user
I WANT to continue using stdio transport if needed
SO THAT my existing Claude Desktop configuration works without changes

GIVEN I have existing Claude Desktop setup using stdio
WHEN I start Zapomni with --transport stdio flag
THEN server works exactly as before with stdio transport
AND no configuration changes are required
```

**Acceptance Criteria:**
- [ ] `--transport stdio` preserves existing behavior
- [ ] Default transport is SSE (new default)
- [ ] Stdio mode works without SSE dependencies
- [ ] All existing tools work identically in both modes

### US-4: Graceful Connection Handling

```
AS a system administrator
I WANT the server to handle client disconnections gracefully
SO THAT resources are properly cleaned up and the server remains stable

GIVEN a client is connected via SSE
WHEN the client disconnects unexpectedly
THEN the server detects the disconnection
AND cleans up session resources (memory streams, session state)
AND logs the disconnection event
```

**Acceptance Criteria:**
- [ ] Disconnection detected within 30 seconds
- [ ] Session resources freed on disconnect
- [ ] No memory leaks from abandoned connections
- [ ] Server remains stable after many connect/disconnect cycles

---

## 3. Functional Requirements

### FR-001: SSE Endpoint Implementation

**Description:** Implement GET `/sse` endpoint for establishing SSE connections.

**Requirements:**
- FR-001.1: Server MUST accept GET requests at `/sse` endpoint
- FR-001.2: Server MUST generate unique UUID session ID per connection
- FR-001.3: Server MUST send SSE event with message endpoint URL immediately after connection
- FR-001.4: Server MUST keep SSE connection open for server-to-client messages
- FR-001.5: Server MUST detect client disconnection via stream closure

**SSE Event Format:**
```
event: endpoint
data: /messages/{session_id}

event: message
data: {"jsonrpc":"2.0","id":1,"result":{...}}
```

### FR-002: Messages Endpoint Implementation

**Description:** Implement POST `/messages/{session_id}` endpoint for receiving JSON-RPC messages.

**Requirements:**
- FR-002.1: Server MUST accept POST requests at `/messages/{session_id}`
- FR-002.2: Server MUST validate session_id exists and is active
- FR-002.3: Server MUST parse JSON-RPC 2.0 message from request body
- FR-002.4: Server MUST forward message to appropriate session's MCP server
- FR-002.5: Server MUST return 202 Accepted for valid requests
- FR-002.6: Server MUST return 404 Not Found for invalid session IDs
- FR-002.7: Server MUST return 400 Bad Request for malformed JSON-RPC

### FR-003: Session Management

**Description:** Implement session lifecycle management for SSE connections.

**Requirements:**
- FR-003.1: Server MUST create session state on new SSE connection
- FR-003.2: Server MUST store session in memory dictionary: `sessions[session_id] = transport`
- FR-003.3: Server MUST clean up session on client disconnect
- FR-003.4: Server MUST remove session from dictionary on cleanup
- FR-003.5: Server MUST log session creation and destruction events
- FR-003.6: Server SHOULD implement connection heartbeat (30s interval)
- FR-003.7: Server SHOULD implement maximum connection lifetime (1 hour)

### FR-004: Transport Selection

**Description:** Implement CLI argument for transport protocol selection.

**Requirements:**
- FR-004.1: Server MUST accept `--transport` CLI argument with values `stdio` or `sse`
- FR-004.2: Default transport MUST be `sse`
- FR-004.3: Server MUST use `server.run()` for stdio transport
- FR-004.4: Server MUST use `server.run_sse()` for SSE transport
- FR-004.5: Transport selection MUST be logged at startup

**CLI Interface:**
```bash
# Start with SSE (default)
python -m zapomni_mcp

# Start with SSE explicitly
python -m zapomni_mcp --transport sse

# Start with stdio (backward compatible)
python -m zapomni_mcp --transport stdio
```

### FR-005: SSE Server Configuration

**Description:** Implement configuration options for SSE server.

**Requirements:**
- FR-005.1: Server MUST support `ZAPOMNI_SSE_HOST` environment variable (default: 127.0.0.1)
- FR-005.2: Server MUST support `ZAPOMNI_SSE_PORT` environment variable (default: 8000)
- FR-005.3: Server MUST support `ZAPOMNI_SSE_CORS_ORIGINS` environment variable (default: *)
- FR-005.4: Configuration MUST be validated at startup
- FR-005.5: Invalid configuration MUST result in clear error message and exit

### FR-006: CORS Configuration

**Description:** Implement CORS middleware for SSE transport.

**Requirements:**
- FR-006.1: Server MUST enable CORS for configured origins
- FR-006.2: Server MUST allow GET, POST, OPTIONS methods
- FR-006.3: Server MUST allow all headers (Access-Control-Allow-Headers: *)
- FR-006.4: Server MUST expose all headers (Access-Control-Expose-Headers: *)
- FR-006.5: Server MUST support credentials (Access-Control-Allow-Credentials: true)

### FR-007: Health Endpoint

**Description:** Implement health check endpoint for monitoring and service verification.

**Requirements:**
- FR-007.1: Server MUST implement GET `/health` endpoint
- FR-007.2: Response MUST include status field ("healthy" or "unhealthy")
- FR-007.3: Response MUST include active_sessions count
- FR-007.4: Response MUST include uptime_seconds
- FR-007.5: Response MUST include version field
- FR-007.6: Response MUST be JSON formatted
- FR-007.7: Endpoint MUST return 200 OK when server is healthy

**Response Format:**
```json
{
  "status": "healthy",
  "active_sessions": 5,
  "uptime_seconds": 3600,
  "version": "0.1.0"
}
```

### FR-008: Async Entity Extraction

**Description:** Wrap CPU-bound EntityExtractor operations for async compatibility.

**Requirements:**
- FR-008.1: EntityExtractor MUST provide async `extract_entities()` method
- FR-008.2: CPU-bound SpaCy processing MUST run in ThreadPoolExecutor
- FR-008.3: Executor MUST have configurable worker count (default: 5)
- FR-008.4: Original sync method MUST be renamed to `_extract_entities_sync()`
- FR-008.5: Call sites MUST be updated to use `await`

### FR-009: Database Pool Scaling

**Description:** Increase FalkorDB connection pool for concurrent access.

**Requirements:**
- FR-009.1: Default pool size MUST be 20
- FR-009.2: Pool size MUST be configurable via `FALKORDB_POOL_SIZE` environment variable
- FR-009.3: Maximum pool size MUST be 200
- FR-009.4: Pool exhaustion SHOULD be logged as warning
- FR-009.5: Retry logic SHOULD handle temporary pool exhaustion

---

## 4. Non-Functional Requirements

### NFR-001: Performance

| Metric | Requirement | Notes |
|--------|------------|-------|
| Connection establishment | < 100ms | From GET /sse to receiving endpoint event |
| Simple tool response (get_stats) | < 500ms | Under 50 concurrent clients |
| Complex tool response (build_graph) | No blocking | Other clients unaffected |
| Memory per connection | < 10MB | Per SSE connection overhead |
| Max concurrent connections | 100+ | Without degradation |

### NFR-002: Reliability

| Metric | Requirement |
|--------|------------|
| Uptime | 99.9% (< 8.7 hours downtime/year) |
| Connection recovery | Automatic reconnection guidance |
| Error handling | Graceful degradation, no crashes |
| Data integrity | No memory corruption under load |

### NFR-003: Security

| Concern | Requirement |
|---------|------------|
| DNS rebinding | TransportSecuritySettings enabled |
| CORS | Configurable origin validation |
| Session hijacking | Cryptographically secure session IDs |
| Local deployment | Listen on 127.0.0.1 by default |

### NFR-004: Maintainability

| Concern | Requirement |
|---------|------------|
| Code organization | Single-responsibility modules |
| Test coverage | 80%+ for new code |
| Documentation | Inline docstrings, README updates |
| Logging | Structured logs with correlation IDs |

### NFR-005: Scalability

| Dimension | Requirement |
|-----------|------------|
| Concurrent clients | 100+ simultaneous connections |
| Request throughput | 1000 requests/minute combined |
| Memory scaling | Linear with connection count |
| CPU utilization | Efficient async processing |

---

## 5. Acceptance Criteria

### AC-1: Basic SSE Functionality

- [ ] Server starts on configured port (default 8000)
- [ ] GET /sse returns SSE stream with endpoint event
- [ ] POST /messages/{session_id} processes JSON-RPC messages
- [ ] Responses sent via SSE stream
- [ ] Session cleaned up on disconnect

### AC-2: Concurrent Client Support

- [ ] Two clients connect simultaneously
- [ ] Both clients can list tools
- [ ] Both clients can execute tools in parallel
- [ ] Client A's heavy operation doesn't block Client B
- [ ] Disconnecting Client A doesn't affect Client B

### AC-3: Backward Compatibility

- [ ] `--transport stdio` works identically to current behavior
- [ ] Existing Claude Desktop config works with stdio flag
- [ ] All tools function correctly in both modes
- [ ] No breaking changes to tool interfaces

### AC-4: Performance Under Load

- [ ] 50 concurrent clients for 30 minutes: stable
- [ ] Memory usage: < 500MB total
- [ ] No connection timeouts
- [ ] No request failures (excluding rate limits)
- [ ] Event loop latency: < 100ms

### AC-5: Error Handling

- [ ] Invalid session ID returns 404
- [ ] Malformed JSON-RPC returns 400
- [ ] Database errors propagate correctly
- [ ] Tool exceptions don't crash server
- [ ] Graceful shutdown on SIGTERM

---

## 6. Constraints & Assumptions

### 6.1 Technical Constraints

| Constraint | Description |
|-----------|-------------|
| TC-1 | Must use MCP SDK's `SseServerTransport` class |
| TC-2 | Must maintain compatibility with MCP protocol |
| TC-3 | Must use Starlette (not FastAPI) for minimal overhead |
| TC-4 | Must use Uvicorn as ASGI server |
| TC-5 | SpaCy model must remain synchronous (no async alternative) |

### 6.2 Design Constraints

| Constraint | Description |
|-----------|-------------|
| DC-1 | Single MCP server instance shared across all sessions |
| DC-2 | Per-session SseServerTransport instances |
| DC-3 | Singleton pattern for MemoryProcessor |
| DC-4 | No changes to tool interfaces |

### 6.3 Assumptions

| Assumption | Description |
|-----------|-------------|
| A-1 | FalkorDB connection pool is thread-safe |
| A-2 | MCP SDK's SseServerTransport handles session properly |
| A-3 | Clients support SSE (EventSource API) |
| A-4 | Local deployment only (no remote access initially) |
| A-5 | Single-server deployment (no load balancing) |

---

## 7. Out of Scope

| Item | Reason | Future Phase |
|------|--------|--------------|
| OS-1: Authentication/Authorization | Explicitly out of scope - local deployment only, no auth required. Server binds to 127.0.0.1 by default. | N/A |
| OS-2: Rate limiting | Explicitly out of scope - local deployment with limited users doesn't require rate limiting. | N/A |
| OS-4: Distributed sessions (Redis) | Single-server deployment sufficient | Phase 5 |
| OS-5: HTTPS/TLS | Local deployment uses HTTP | Phase 5 |
| OS-6: Load balancing | Single-server deployment | Phase 5 |
| OS-7: Connection persistence (Redis) | In-memory sessions sufficient | Phase 5 |
| OS-8: Metrics/Prometheus | Basic logging sufficient | Phase 5 |

---

## 8. Dependencies

### 8.1 Internal Dependencies

| Dependency | Status | Notes |
|-----------|--------|-------|
| Phase 1-3 MCP Tools | Complete | All tools implemented |
| EntityExtractor | Complete | Needs async wrapper |
| FalkorDBClient | Complete | Needs pool scaling |
| MemoryProcessor | Complete | No changes needed |

### 8.2 External Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| uvicorn[standard] | >=0.30.0 | ASGI server |
| starlette | >=0.38.0 | ASGI framework |
| mcp | existing | SseServerTransport |
| anyio | existing | Async I/O |

---

## 9. Risks & Mitigations

### R-1: Event Loop Blocking (CRITICAL)

**Risk:** SpaCy NLP processing blocks asyncio event loop, preventing concurrent requests.

**Impact:** High - Defeats purpose of SSE transport.

**Mitigation:**
1. Wrap `extract_entities()` in `run_in_executor()` with ThreadPoolExecutor
2. Test with concurrent build_graph + get_stats requests
3. Monitor event loop lag metrics

### R-2: Memory Leaks from Long-Lived Connections

**Risk:** SSE connections hold resources indefinitely, causing memory growth.

**Impact:** Medium - Gradual degradation over time.

**Mitigation:**
1. Use `async with` context managers for automatic cleanup
2. Implement connection heartbeats (30s)
3. Set maximum connection lifetime (1 hour)
4. Monitor session count and memory usage

### R-3: Connection Pool Exhaustion

**Risk:** Many concurrent clients exhaust FalkorDB connections.

**Impact:** Medium - Requests fail with connection errors.

**Mitigation:**
1. Increase pool size from 10 to 50
2. Add pool exhaustion logging
3. Implement retry with exponential backoff
4. Document pool sizing guidelines

### R-4: Session Hijacking

**Risk:** Session IDs in URLs could be intercepted.

**Impact:** Low (local deployment).

**Mitigation:**
1. Use cryptographically secure session IDs (`secrets.token_urlsafe(32)`)
2. Enable DNS rebinding protection
3. Bind to 127.0.0.1 by default
4. Document security considerations for remote deployment

---

## 10. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Concurrent client support | 100+ | Load testing |
| Response latency (simple) | < 500ms P95 | Load testing |
| Memory per connection | < 10MB | Profiling |
| Session cleanup success | 100% | Automated testing |
| Backward compatibility | 100% | Integration tests |
| Test coverage | > 80% | Coverage reports |

---

**End of Requirements Document**
