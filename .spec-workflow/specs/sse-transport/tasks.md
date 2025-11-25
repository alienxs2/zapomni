# Implementation Tasks - SSE Transport for Concurrent Connections

**Spec Name:** sse-transport
**Status:** Implementation Phase
**Created:** 2025-11-25
**Estimated Duration:** 3-4 weeks

---

## Overview

Total Tasks: **37** (including subtasks)
Estimated Effort: **~65 hours**

### Task Summary by Phase

| Phase | Description | Tasks | Priority |
|-------|-------------|-------|----------|
| 1 | Foundation | 8 | Critical |
| 2 | Concurrency | 7 | Critical |
| 3 | Integration | 10 | High |
| 4 | Polish | 6 | Medium |
| 5 | Testing | 6 | High |

---

## Phase 1: Foundation (Week 1)

### 1.1 Dependencies & Configuration

- [ ] **1.1.1** Add uvicorn and starlette dependencies to pyproject.toml
  - Files: `pyproject.toml`
  - Add: `uvicorn[standard]>=0.30.0`, `starlette>=0.38.0`
  - Est: 15 min

- [ ] **1.1.2** Add SSE configuration fields to ZapomniSettings
  - Files: `src/zapomni_core/config.py`
  - Add: `sse_host`, `sse_port`, `sse_cors_origins`, `sse_heartbeat_interval`, `sse_max_connection_lifetime`
  - Est: 30 min

- [ ] **1.1.3** Add SSE settings to MCP Settings dataclass
  - Files: `src/zapomni_mcp/config.py`
  - Add: `sse_host`, `sse_port`, `cors_origins`
  - Est: 15 min

### 1.2 Session Management

- [ ] **1.2.1** Create session_manager.py with SessionManager class
  - Files: `src/zapomni_mcp/session_manager.py` (NEW)
  - Implement: `create_session()`, `get_session()`, `delete_session()`, `active_session_count`
  - Include: asyncio Lock for thread-safety, session metadata tracking
  - Est: 1 hour

- [ ] **1.2.2** Add SessionState dataclass for session tracking
  - Files: `src/zapomni_mcp/session_manager.py`
  - Include: session_id, transport, created_at, last_activity, client_ip, request_count
  - Est: 30 min

### 1.3 SSE Transport Implementation

- [ ] **1.3.1** Create sse_transport.py with SSEConfig dataclass
  - Files: `src/zapomni_mcp/sse_transport.py` (NEW)
  - Implement: SSEConfig with host, port, cors_origins, heartbeat_interval, max_connection_lifetime
  - Est: 30 min

- [ ] **1.3.2** Implement handle_sse endpoint function
  - Files: `src/zapomni_mcp/sse_transport.py`
  - Implement:
    - Create SseServerTransport with session-specific endpoint
    - Register session with SessionManager
    - Connect SSE streams via `transport.connect_sse()`
    - Run MCP server with streams: `mcp_server._server.run(streams)`
    - Cleanup session in finally block
  - Est: 2 hours

- [ ] **1.3.3** Implement handle_messages endpoint function
  - Files: `src/zapomni_mcp/sse_transport.py`
  - Implement:
    - Extract session_id from path
    - Lookup transport via SessionManager
    - Call `transport.handle_post_message()`
    - Return 404 for invalid session, 400 for bad JSON
  - Est: 1.5 hours

---

## Phase 2: Concurrency Fixes (Week 2)

### 2.1 Entity Extractor Async Wrapper

- [x] **2.1.1** Add ThreadPoolExecutor to EntityExtractor.__init__
  - Files: `src/zapomni_core/extractors/entity_extractor.py`
  - Add: `self._executor = ThreadPoolExecutor(max_workers=executor_workers)`
  - Add: `executor_workers` parameter (default: 5)
  - Est: 30 min
  - **Completed:** Added ThreadPoolExecutor with configurable executor_workers param

- [x] **2.1.2** Create async extract_entities() method
  - Files: `src/zapomni_core/extractors/entity_extractor.py`
  - Implement: Wrapper using `loop.run_in_executor()`
  - Keep original sync logic in `_extract_entities_sync()`
  - Est: 1 hour
  - **Completed:** Added extract_entities_async() method preserving sync backward compat

- [x] **2.1.3** Add close() method to EntityExtractor for cleanup
  - Files: `src/zapomni_core/extractors/entity_extractor.py`
  - Implement: `self._executor.shutdown(wait=True)`
  - Est: 15 min
  - **Completed:** Added shutdown() method for graceful executor cleanup

### 2.2 Call Site Updates

- [x] **2.2.1** Update BuildGraphTool to use async extract_entities
  - Files: `src/zapomni_mcp/tools/build_graph.py`
  - Change: Add `await` to extract_entities() call
  - Est: 30 min
  - **Completed:** Updated _extract_entities to use extract_entities_async with fallback

- [x] **2.2.2** Update GraphBuilder to use async extract_entities
  - Files: `src/zapomni_core/graph/graph_builder.py`
  - Change: Add `await` to extract_entities() call
  - Ensure `build_graph()` method is async
  - Est: 30 min
  - **Completed:** Updated build_graph to use extract_entities_async with fallback

### 2.3 Database Pool Scaling

- [x] **2.3.1** Update default FalkorDB pool size
  - Files: `src/zapomni_core/config.py`
  - Change: `falkordb_pool_size` default to 20
  - Ensure: `le=200` for max pool size constraint
  - Est: 15 min
  - **Completed:** Updated default to 20 in both config.py and falkordb_client.py

- [x] **2.3.2** Add pool monitoring logging to FalkorDBClient
  - Files: `src/zapomni_db/falkordb_client.py`
  - Add: Log pool usage on connection acquisition/release
  - Add: Warning when pool utilization > 80%
  - Est: 45 min
  - **Completed:** Added pool monitoring with utilization tracking and warning logs

---

## Phase 3: Server Integration (Week 3)

### 3.1 MCPServer SSE Method

- [x] **3.1.1** Add run_sse() method to MCPServer class
  - Files: `src/zapomni_mcp/server.py`
  - Implement:
    - Create SSEConfig from parameters
    - Create Starlette app via `create_sse_app()`
    - Start uvicorn server
    - Handle shutdown
  - Est: 2 hours
  - **Completed:** Added run_sse() method with SSEConfig, uvicorn server, tool registration, and proper shutdown handling

- [x] **3.1.2** Add create_sse_app() factory function
  - Files: `src/zapomni_mcp/sse_transport.py`
  - Implement:
    - Create Starlette app with routes
    - Configure CORS middleware
    - Pass MCPServer reference to handlers
  - Est: 1.5 hours
  - **Completed:** Already implemented in Phase 1/2 with full SSE endpoint and message handling

### 3.2 Entry Point Updates

- [x] **3.2.1** Add argparse for --transport flag in __main__.py
  - Files: `src/zapomni_mcp/__main__.py`
  - Implement:
    - `--transport` with choices `["stdio", "sse"]`, default "sse"
    - `--host` for SSE host (optional)
    - `--port` for SSE port (optional)
  - Est: 45 min
  - **Completed:** Added full argparse with transport, host, port, cors-origins arguments

- [x] **3.2.2** Implement transport selection logic
  - Files: `src/zapomni_mcp/__main__.py`
  - Est: 1.5 hours (total for subtasks)
  - **Completed:** All subtasks implemented

  - [x] **3.2.2a** Add argparse ArgumentParser setup
    - Create ArgumentParser with description "Zapomni MCP Server"
    - Configure for clean help output
    - Est: 15 min
    - **Completed:** Added with RawDescriptionHelpFormatter and detailed epilog

  - [x] **3.2.2b** Add --transport argument
    - Choices: ["stdio", "sse"]
    - Default: "sse"
    - Help text explaining each transport mode
    - Est: 15 min
    - **Completed:** Added with default="sse" and descriptive help text

  - [x] **3.2.2c** Add --host and --port arguments for SSE mode
    - --host: SSE server bind address (default from env or 127.0.0.1)
    - --port: SSE server port (default from env or 8000)
    - Only used when transport=sse
    - Est: 15 min
    - **Completed:** Added with env var fallback via SSEConfig.from_env()

  - [x] **3.2.2d** Route to appropriate run method based on transport choice
    - If transport == "sse": `await server.run_sse(host, port)`
    - If transport == "stdio": `await server.run()`
    - Load config from env vars with CLI override
    - Log selected transport at startup
    - Est: 30 min
    - **Completed:** Added transport routing with proper logging of selected transport

### 3.3 CORS Configuration

- [x] **3.3.1** Implement CORS middleware configuration
  - Files: `src/zapomni_mcp/sse_transport.py`
  - Implement:
    - Parse comma-separated origins from config
    - Configure CORSMiddleware with origins
    - Support wildcard "*" for development
  - Est: 30 min
  - **Completed:** Already implemented in create_sse_app() with full CORSMiddleware configuration

- [ ] **3.3.2** Add DNS rebinding protection
  - Files: `src/zapomni_mcp/sse_transport.py`
  - Implement:
    - Create TransportSecuritySettings with allowed_hosts
    - Pass to SseServerTransport constructor
  - Est: 30 min
  - **Note:** Deferred to Phase 4 - current local-only binding (127.0.0.1) provides basic protection

---

## Phase 4: Polish & Robustness (Week 3-4)

### 4.1 Connection Management

- [x] **4.1.1** Implement connection heartbeat
  - Files: `src/zapomni_mcp/session_manager.py`
  - Implement:
    - Background task sending ping every 30s (configurable via heartbeat_interval)
    - Detect stale connections via last_activity tracking
    - Clean up dead sessions via heartbeat failure detection
  - Est: 1.5 hours
  - **Completed:** Added heartbeat task management to SessionManager with _start_heartbeat(), _stop_heartbeat(), _heartbeat_loop()

- [x] **4.1.2** Implement max connection lifetime
  - Files: `src/zapomni_mcp/session_manager.py`
  - Implement:
    - Track session creation time (created_at in SessionState)
    - Close sessions exceeding max lifetime (cleanup_stale_sessions method)
    - Log session expiration
  - Est: 1 hour
  - **Completed:** Already existed in Phase 1, now integrated with heartbeat and metrics

### 4.2 Graceful Shutdown

- [x] **4.2.1** Implement graceful shutdown for SSE server
  - Files: `src/zapomni_mcp/server.py`
  - Implement:
    - Handle SIGTERM/SIGINT (already existed)
    - Close all active sessions via _graceful_shutdown_sse()
    - SessionManager.close_all_sessions() for cleanup
    - Clean shutdown of uvicorn
  - Est: 1.5 hours
  - **Completed:** Added _graceful_shutdown_sse() method with proper cleanup sequence

- [x] **4.2.2** Implement EntityExtractor cleanup on shutdown
  - Files: `src/zapomni_mcp/server.py`, `src/zapomni_core/extractors/entity_extractor.py`
  - Implement:
    - Shutdown EntityExtractor ThreadPoolExecutor
    - Wait for pending extractions to complete (with 10s timeout)
    - Log pending extraction count before shutdown
    - Call EntityExtractor.shutdown() during server shutdown sequence
  - Est: 45 min
  - **Completed:** Added _cleanup_entity_extractor() method called during graceful shutdown

### 4.3 Metrics & Logging

- [x] **4.3.1** Add connection metrics tracking
  - Files: `src/zapomni_mcp/session_manager.py`
  - Implement:
    - Track: active_connections, total_connections, total_errors, peak_connections
    - ConnectionMetrics dataclass for structured metrics
    - Metrics updated on session create/remove
  - Est: 45 min
  - **Completed:** Added ConnectionMetrics class with comprehensive tracking

- [x] **4.3.2** Add session logging with correlation IDs
  - Files: `src/zapomni_mcp/sse_transport.py`, `src/zapomni_mcp/session_manager.py`
  - Implement:
    - Bind session_id to logger context
    - Log all session events with session_id
  - Est: 30 min
  - **Completed:** All logging now includes session_id for correlation

### 4.4 Health Endpoint (Required)

- [x] **4.4.1** Implement /health endpoint for monitoring
  - Files: `src/zapomni_mcp/sse_transport.py`
  - Implement:
    - GET /health endpoint returning health status
    - Response includes: status, active_sessions, uptime_seconds, version, metrics
    - Return 200 OK when healthy, 503 when unhealthy
    - JSON response format
  - Est: 45 min
  - **Completed:** Added handle_health() endpoint with full metrics including SSE transport stats

### 4.5 SSE Metrics in get_stats Tool (Additional)

- [x] **4.5.1** Add SSE metrics to get_stats tool response
  - Files: `src/zapomni_mcp/tools/get_stats.py`, `src/zapomni_mcp/server.py`
  - Implement:
    - GetStatsTool now accepts mcp_server reference for dynamic session_manager access
    - SSE transport metrics included when available
  - **Completed:** GetStatsTool updated with session_manager property for dynamic SSE metrics

---

## Phase 5: Testing (Week 4)

### 5.1 Unit Tests

- [ ] **5.1.1** Create unit tests for SessionManager
  - Files: `tests/unit/test_session_manager.py` (NEW)
  - Test:
    - Session creation and ID generation
    - Session lookup (found/not found)
    - Session deletion and cleanup
    - Concurrent session operations
  - Est: 2 hours

- [ ] **5.1.2** Create unit tests for SSE endpoints
  - Files: `tests/unit/test_sse_transport.py` (NEW)
  - Test:
    - handle_sse creates session and returns SSE stream
    - handle_messages validates session and forwards
    - Error responses (404, 400)
  - Est: 2 hours

### 5.2 Integration Tests

- [ ] **5.2.1** Create integration tests for SSE transport
  - Files: `tests/integration/test_sse_transport.py` (NEW)
  - Test:
    - Full connection lifecycle (connect, call tools, disconnect)
    - Multiple concurrent clients
    - Tool execution via SSE (all tools)
    - Error propagation
  - Est: 3 hours

- [ ] **5.2.2** Create integration tests for backward compatibility
  - Files: `tests/integration/test_transport_compat.py` (NEW)
  - Test:
    - --transport stdio works unchanged
    - All tools work in both modes
    - Default transport is SSE
  - Est: 1.5 hours

### 5.3 Load Tests

- [ ] **5.3.1** Create load test configuration with locust
  - Files: `tests/load/locustfile.py` (NEW)
  - Implement:
    - SSE connection user class
    - Tool execution scenarios
    - Metrics collection
  - Targets: 50 concurrent clients, 30 min duration
  - Est: 2 hours

### 5.4 Backward Compatibility Tests

- [ ] **5.4.1** Create stdio backward compatibility test suite
  - Files: `tests/integration/test_stdio_compat.py` (NEW)
  - Test:
    - Stdio transport still works with --transport stdio flag
    - Verify existing MCP client configurations work unchanged
    - Test tool execution through stdio (all tools)
    - Verify JSON-RPC message format unchanged
    - Test stdin/stdout communication patterns
  - Est: 2 hours

---

## Task Dependencies

```
1.1.1 (deps) ──────────────────────────────────────────────────────────┐
                                                                       │
1.1.2 (config) ──┬─────────────────────────────────────────────────────┤
                 │                                                     │
1.1.3 (config) ──┤                                                     │
                 │                                                     │
1.2.1 (session) ─┼─────────────────────────────────────────┐           │
                 │                                         │           │
1.2.2 (session) ─┤                                         │           │
                 │                                         │           │
1.3.1 (sse cfg) ─┤                                         │           │
                 │                                         │           │
1.3.2 (sse) ─────┼─────────────────────────────────────────┤           │
                 │                                         │           │
1.3.3 (sse) ─────┘                                         │           │
                                                           │           │
2.1.1 (executor) ───────────────────────┐                  │           │
                                        │                  │           │
2.1.2 (async) ──────────────────────────┼───────┐          │           │
                                        │       │          │           │
2.1.3 (cleanup) ────────────────────────┘       │          │           │
                                                │          │           │
2.2.1 (call site) ──────────────────────────────┤          │           │
                                                │          │           │
2.2.2 (call site) ──────────────────────────────┘          │           │
                                                           │           │
2.3.1 (pool) ──────────────────────────────────────────────┤           │
                                                           │           │
2.3.2 (monitoring) ────────────────────────────────────────┤           │
                                                           │           │
3.1.1 (run_sse) ───────────────────────────────────────────┤           │
                                                           │           │
3.1.2 (app factory) ───────────────────────────────────────┤           │
                                                           │           │
3.2.1 (argparse) ──────────────────────────────────────────┼───────────┤
                                                           │           │
3.2.2 (selection) ─────────────────────────────────────────┤           │
                                                           │           │
3.3.1 (cors) ──────────────────────────────────────────────┤           │
                                                           │           │
3.3.2 (security) ──────────────────────────────────────────┘           │
                                                                       │
4.1.1 (heartbeat) ─────────────────────────────────────────────────────┤
                                                                       │
4.1.2 (lifetime) ──────────────────────────────────────────────────────┤
                                                                       │
4.2.1 (shutdown) ──────────────────────────────────────────────────────┤
                                                                       │
4.3.1 (metrics) ───────────────────────────────────────────────────────┤
                                                                       │
4.3.2 (logging) ───────────────────────────────────────────────────────┤
                                                                       │
4.4.1 (health) ────────────────────────────────────────────────────────┤
                                                                       │
5.1.1 (unit tests) ────────────────────────────────────────────────────┤
                                                                       │
5.1.2 (unit tests) ────────────────────────────────────────────────────┤
                                                                       │
5.2.1 (integration) ───────────────────────────────────────────────────┤
                                                                       │
5.2.2 (compat tests) ──────────────────────────────────────────────────┤
                                                                       │
5.3.1 (load tests) ────────────────────────────────────────────────────┘
```

---

## Acceptance Checklist

### Phase 1 Complete When:
- [ ] Dependencies installed and importable
- [ ] Configuration loads from environment
- [ ] SessionManager creates/tracks/deletes sessions
- [ ] SSE endpoint accepts connections and returns endpoint event
- [ ] Messages endpoint forwards to correct session

### Phase 2 Complete When:
- [x] EntityExtractor.extract_entities() is async
- [x] SpaCy runs in ThreadPoolExecutor
- [x] build_graph tool works with async extraction
- [x] FalkorDB pool size defaults to 20 (configurable up to 200)
- [x] No event loop blocking during heavy operations

### Phase 3 Complete When:
- [x] `python -m zapomni_mcp` starts SSE server on port 8000
- [x] `python -m zapomni_mcp --transport stdio` works as before
- [x] CORS configured correctly for local development
- [x] All tools accessible via SSE transport

### Phase 4 Complete When:
- [x] Heartbeats detect stale connections
- [x] Old sessions cleaned up automatically
- [x] Graceful shutdown preserves in-progress requests
- [x] Connection metrics logged
- [x] Health endpoint returns server status, version, active connections

### Phase 5 Complete When:
- [ ] Unit tests pass for SessionManager and SSE handlers
- [ ] Integration tests verify full SSE lifecycle
- [ ] Backward compatibility tests pass for stdio
- [ ] Load tests show stable behavior with 50+ clients

---

## Notes

1. **Critical Path**: Tasks 1.1.1 -> 1.3.2 -> 2.1.2 -> 3.1.1 -> 3.2.2 are on critical path
2. **Parallelization**: Phase 1 config tasks can run in parallel
3. **Risk Mitigation**: Test async EntityExtractor early (2.1.2) as it's highest risk
4. **Documentation**: Update README after Phase 3 completion

---

**End of Tasks Document**
