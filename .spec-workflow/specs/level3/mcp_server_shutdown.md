# MCPServer.shutdown() - Function Specification

**Level:** 3 | **Component:** MCPServer | **Module:** zapomni_mcp
**Author:** Goncharenko Anton aka alienxs2 | **Status:** Draft | **V:** 1.0

## Signature
```python
async def shutdown(self) -> None:
    """
    Gracefully shutdown MCP server and cleanup resources.
    
    Performs cleanup in order:
    1. Stop accepting new requests
    2. Wait for in-flight requests to complete (timeout: 5s)
    3. Close database connections
    4. Flush logs
    5. Mark server as stopped
    
    Raises:
        None (errors logged but not raised)
    """
```

## Purpose

Gracefully shutdown server ensuring:
- No data loss (in-flight requests complete)
- Clean resource cleanup (DB connections closed)
- Logs flushed (audit trail preserved)

## Edge Cases
1. Called twice → Second call is no-op (idempotent)
2. In-flight requests timeout → Logged warning, continue shutdown
3. DB connection already closed → No error
4. No active requests → Immediate shutdown
5. Shutdown during request → Request completes first

## Algorithm
```
1. Check if already stopped (return if yes)
2. Set stopping flag (reject new requests)
3. Wait for active requests to complete (max 5s)
4. Close database client
5. Close embedder client
6. Flush structured logs
7. Set stopped flag
```

## Tests (10)
1. test_shutdown_no_active_requests
2. test_shutdown_with_active_requests
3. test_shutdown_idempotent
4. test_shutdown_timeout_warning
5. test_shutdown_closes_db
6. test_shutdown_closes_embedder
7. test_shutdown_flushes_logs
8. test_shutdown_sets_stopped_flag
9. test_shutdown_rejects_new_requests
10. test_shutdown_error_handling
