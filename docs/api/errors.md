# API Error Reference

Complete error handling guide for Zapomni MCP tools including error codes, types, and troubleshooting.

## Error Architecture

All Zapomni errors follow a consistent structure:

```python
{
    "error": "ErrorClassName",
    "message": "Human-readable error message",
    "error_code": "ERR_XXX",
    "details": {
        "field": "value",
        "context": "additional info"
    },
    "correlation_id": "uuid-for-tracing",
    "is_transient": True or False  # Can be retried?
}
```

## Error Categories

### By Severity

| Category | Retryable | Action |
|----------|-----------|--------|
| **Input Validation** | No | Fix request, resubmit |
| **Transient Errors** | Yes | Retry with backoff |
| **Permanent Errors** | No | Fix condition, resubmit |
| **System Errors** | Varies | Check logs, diagnose |

### By Source

| Source | Examples | Handling |
|--------|----------|----------|
| **Validation** (VAL_*) | Missing fields, wrong type | User error - fix input |
| **Processing** (PROC_*) | Chunking, extraction | Processing issue - retry or fix |
| **Embedding** (EMB_*) | Ollama unavailable | Service issue - start service |
| **Search** (SEARCH_*) | Query failed | Retry with different query |
| **Database** (DB_*, CONN_*, QUERY_*) | Connection, query errors | Check services |

## Error Code Reference

### Validation Errors (VAL_*)

**Characteristic**: Non-transient, user input issues

| Code | Name | Message | Cause | Fix |
|------|------|---------|-------|-----|
| **VAL_001** | Missing Field | "Missing required field 'X'" | Required parameter not provided | Add required parameter |
| **VAL_002** | Wrong Type | "Invalid field type for 'X'" | Parameter has wrong type | Check parameter type |
| **VAL_003** | Out of Range | "Field value out of range" | Value exceeds limits | Use value within limits |
| **VAL_004** | Invalid Format | "Invalid field format" | Value format incorrect | Use correct format |

#### Examples

**VAL_001 - Missing text**:
```json
{
    "error": "ValidationError",
    "message": "Missing required field 'text'",
    "error_code": "VAL_001",
    "details": {"missing_field": "text"}
}
```

**VAL_003 - Text too long**:
```json
{
    "error": "ValidationError",
    "message": "Field value out of range (max 10,000,000 chars)",
    "error_code": "VAL_003",
    "details": {"field": "text", "max": 10000000, "actual": 10000001}
}
```

### Processing Errors (PROC_*)

**Characteristic**: Usually non-transient, content processing issues

| Code | Name | Message | Cause | Fix |
|------|------|---------|-------|-----|
| **PROC_001** | Chunking Failed | "Chunking failed" | Text couldn't be split | Check text format |
| **PROC_002** | Extraction Failed | "Text extraction failed" | Content issue | Use valid UTF-8 |
| **PROC_003** | Invalid Format | "Invalid document format" | Unsupported content | Convert to supported format |

#### Examples

**PROC_001 - Chunking error**:
```json
{
    "error": "ProcessingError",
    "message": "Chunking failed: text too fragmented",
    "error_code": "PROC_001",
    "details": {"text_length": 10000, "reason": "insufficient_context"}
}
```

### Embedding Errors (EMB_*)

**Characteristic**: Usually transient, service/network issues

| Code | Name | Message | Cause | Fix |
|------|------|---------|-------|-----|
| **EMB_001** | Connection Failed | "Ollama connection failed" | Service unreachable | Start Ollama |
| **EMB_002** | Timeout | "Embedding timeout" | Request too slow | Retry or check load |
| **EMB_003** | Invalid Dimensions | "Invalid embedding dimensions" | Wrong vector size | Check model |
| **EMB_004** | Model Not Found | "Model not found" | Required model missing | Pull model |

#### Examples

**EMB_001 - Ollama down**:
```json
{
    "error": "EmbeddingError",
    "message": "Ollama connection failed: refused",
    "error_code": "EMB_001",
    "details": {"service": "ollama", "host": "localhost", "port": 11434},
    "is_transient": true
}
```

**EMB_002 - Timeout**:
```json
{
    "error": "EmbeddingError",
    "message": "Embedding timeout after 30 seconds",
    "error_code": "EMB_002",
    "details": {"timeout_ms": 30000, "chunk_count": 50},
    "is_transient": true
}
```

### Search Errors (SEARCH_*)

**Characteristic**: Non-transient, search logic issues

| Code | Name | Message | Cause | Fix |
|------|------|---------|-------|-----|
| **SEARCH_001** | Search Failed | "Vector search failed" | Query couldn't execute | Retry with simpler query |
| **SEARCH_002** | BM25 Failed | "BM25 search failed" | Keyword search error | Try different keywords |
| **SEARCH_003** | Reranking Failed | "Reranking failed" | Result reranking error | Retry search |

#### Examples

**SEARCH_001 - Vector search failed**:
```json
{
    "error": "SearchError",
    "message": "Vector search failed: index corrupted",
    "error_code": "SEARCH_001",
    "details": {"index": "vector_hnsw", "reason": "index_corrupted"}
}
```

### Extraction Errors (EXTR_*)

**Characteristic**: Usually transient, LLM/NLP issues

| Code | Name | Message | Cause | Fix |
|------|------|---------|-------|-----|
| **EXTR_001** | Extraction Failed | "Entity extraction failed" | Couldn't extract entities | Retry or use simpler text |
| **EXTR_002** | Relationship Failed | "Relationship detection failed" | Couldn't find relationships | Retry |
| **EXTR_003** | Parsing Failed | "LLM response parsing failed" | Couldn't parse LLM output | Retry |

#### Examples

**EXTR_001 - Entity extraction failed**:
```json
{
    "error": "ExtractionError",
    "message": "Entity extraction failed: LLM error",
    "error_code": "EXTR_001",
    "details": {"service": "ollama_llm", "reason": "timeout"},
    "is_transient": true
}
```

### Database Errors (DB_*)

**Characteristic**: Non-transient, database issues

| Code | Name | Message | Cause | Fix |
|------|------|---------|-------|-----|
| **DB_001** | Generic Error | "Database error" | Generic DB issue | Check logs |
| **DB_002** | Query Error | "Query error" | Malformed query | File bug report |
| **DB_003** | Constraint Error | "Constraint violation" | Duplicate key, etc | Resolve constraint |

#### Examples

**DB_001 - Generic database error**:
```json
{
    "error": "DatabaseError",
    "message": "Database error: operation failed",
    "error_code": "DB_001",
    "details": {"operation": "write", "table": "memories"}
}
```

### Connection Errors (CONN_*)

**Characteristic**: Usually transient, network/service issues

| Code | Name | Message | Cause | Fix |
|------|------|---------|-------|-----|
| **CONN_001** | Refused | "Connection refused" | Service down | Start service |
| **CONN_002** | Timeout | "Connection timeout" | Network issue | Check network |
| **CONN_003** | Auth Failed | "Authentication failed" | Wrong credentials | Check credentials |
| **CONN_004** | Pool Exhausted | "Connection pool exhausted" | Too many connections | Reduce load |

#### Examples

**CONN_001 - FalkorDB down**:
```json
{
    "error": "ConnectionError",
    "message": "FalkorDB connection refused",
    "error_code": "CONN_001",
    "details": {"service": "falkordb", "host": "localhost", "port": 6379},
    "is_transient": true
}
```

**CONN_002 - Network timeout**:
```json
{
    "error": "ConnectionError",
    "message": "Connection timeout",
    "error_code": "CONN_002",
    "details": {"timeout_ms": 5000},
    "is_transient": true
}
```

### Query Errors (QUERY_*)

**Characteristic**: Conditional transience, depends on error type

| Code | Name | Message | Cause | Fix |
|------|------|---------|-------|-----|
| **QUERY_001** | Syntax Error | "Syntax error in query" | Malformed Cypher | File bug report |
| **QUERY_002** | Timeout | "Query timeout" | Query too slow | Retry or simplify |
| **QUERY_003** | Index Missing | "Index not found" | Index doesn't exist | Rebuild indices |
| **QUERY_004** | Constraint | "Constraint violation" | Duplicate key, etc | Check constraint |

#### Examples

**QUERY_002 - Timeout**:
```json
{
    "error": "QueryError",
    "message": "Query timeout after 60 seconds",
    "error_code": "QUERY_002",
    "details": {"timeout_ms": 60000, "query_type": "vector_search"},
    "is_transient": true
}
```

### Timeout Errors (TIMEOUT_*)

**Characteristic**: Transient, network/service delays

| Code | Name | Message | Cause | Fix |
|------|------|---------|-------|-----|
| **TIMEOUT_001** | Operation Timeout | "Operation timeout" | Operation took too long | Retry or check load |
| **TIMEOUT_002** | Network Timeout | "Network timeout" | Network delay | Check network |
| **TIMEOUT_003** | Query Timeout | "Query timeout" | Query execution slow | Retry or optimize |

## MCP Tool Error Responses

All tools return errors in MCP format:

```json
{
    "content": [
        {
            "type": "text",
            "text": "Error: error message"
        }
    ],
    "isError": true
}
```

### Parsing Error Responses

```python
def parse_error_response(response):
    if response["isError"]:
        error_text = response["content"][0]["text"]
        # Extract error code if available
        if "ERR_" in error_text:
            code = extract_error_code(error_text)
            return code, error_text
        return None, error_text
    return None, None

# Usage
code, msg = parse_error_response(result)
if code:
    handle_error_code(code)
else:
    handle_error_message(msg)
```

## Error Handling Patterns

### Pattern 1: Retry on Transient Errors

```python
import asyncio
from exponential_backoff import exponential_backoff

async def add_memory_with_retry(text, metadata=None, max_retries=3):
    for attempt in range(max_retries):
        try:
            result = await add_memory.execute({
                "text": text,
                "metadata": metadata or {}
            })

            if not result["isError"]:
                return result

            error_msg = result["content"][0]["text"]

            # Check if error is transient
            if is_transient_error(error_msg):
                wait_time = exponential_backoff(attempt)
                print(f"Transient error, retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                raise ValueError(f"Non-transient error: {error_msg}")

        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait_time = exponential_backoff(attempt)
            await asyncio.sleep(wait_time)

    raise RuntimeError(f"Failed after {max_retries} attempts")

def is_transient_error(msg):
    transient_indicators = [
        "Ollama", "connection", "timeout", "temporarily",
        "unavailable", "retry"
    ]
    return any(indicator in msg for indicator in transient_indicators)
```

### Pattern 2: Graceful Degradation

```python
async def search_with_fallback(query):
    try:
        result = await search_memory.execute({
            "query": query,
            "limit": 10
        })

        if result["isError"]:
            error_msg = result["content"][0]["text"]

            if "Ollama" in error_msg:
                return {
                    "status": "degraded",
                    "message": "Semantic search unavailable",
                    "results": []
                }
            elif "Database" in error_msg:
                return {
                    "status": "offline",
                    "message": "Memory system offline",
                    "results": []
                }
            else:
                raise ValueError(f"Search failed: {error_msg}")

        return {
            "status": "ok",
            "message": "Search successful",
            "results": result["content"][0]["text"]
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}",
            "results": []
        }
```

### Pattern 3: Detailed Error Logging

```python
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

async def add_memory_with_logging(text, metadata=None):
    correlation_id = generate_correlation_id()

    try:
        result = await add_memory.execute({
            "text": text,
            "metadata": metadata or {}
        })

        if result["isError"]:
            error_msg = result["content"][0]["text"]
            logger.error(
                "add_memory_failed",
                extra={
                    "correlation_id": correlation_id,
                    "error": error_msg,
                    "text_length": len(text),
                    "has_metadata": bool(metadata),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            raise ValueError(error_msg)

        logger.info(
            "add_memory_success",
            extra={
                "correlation_id": correlation_id,
                "text_length": len(text),
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        return result

    except Exception as e:
        logger.exception(
            "add_memory_exception",
            extra={
                "correlation_id": correlation_id,
                "error_type": type(e).__name__,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        raise
```

## Common Errors and Solutions

### Ollama Service Errors

**Symptom**: "Ollama connection failed", "Model not found"

**Common Causes**:
1. Ollama service not running
2. Required model not loaded
3. Network connectivity issue

**Solutions**:

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running, start it
ollama serve

# If model not loaded, pull it
ollama pull nomic-embed-text

# Check loaded models
ollama list

# Verify connectivity
curl -X POST http://localhost:11434/api/embeddings \
  -d '{"model":"nomic-embed-text","prompt":"test"}'
```

### FalkorDB Connection Errors

**Symptom**: "Connection refused", "Database temporarily unavailable"

**Common Causes**:
1. FalkorDB container not running
2. Container crashed
3. Network connectivity issue

**Solutions**:

```bash
# Check container status
docker-compose ps

# View logs
docker-compose logs falkordb

# Restart services
docker-compose down
docker-compose up -d

# Verify connectivity
redis-cli ping  # Should return PONG

# Check memory info
redis-cli INFO memory
```

### Text Validation Errors

**Symptom**: "VAL_003: Field value out of range", "VAL_004: Invalid format"

**Common Causes**:
1. Text > 10MB
2. Non-UTF8 encoding
3. Invalid metadata format

**Solutions**:

```python
# Split large text
if len(text) > 10_000_000:
    chunk_size = 5_000_000
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        await add_memory.execute({"text": chunk})

# Ensure UTF-8 encoding
if isinstance(text, bytes):
    text = text.decode('utf-8', errors='replace')

# Validate metadata
metadata = {k: v for k, v in metadata.items() if is_valid_value(v)}
```

### High Latency Errors

**Symptom**: Queries timing out, "QUERY_002: Query timeout"

**Common Causes**:
1. System under heavy load
2. Database fragmented
3. Too many memories

**Solutions**:

```bash
# Check system load
top
htop

# Check database stats
redis-cli INFO stats

# Reduce dataset size
# (archive or delete old memories)

# Optimize indices
# (rebuild or recreate)

# Restart services to clear memory
docker-compose restart
```

### No Results Found

**Symptom**: "No results found matching your query" (not an error, but unexpected)

**Common Causes**:
1. No memories stored yet
2. Query too specific
3. Filters too restrictive

**Solutions**:

```python
# Check if memories exist
stats = await get_stats.execute({})
if stats["content"] and "Total Memories: 0" in stats["content"][0]["text"]:
    print("No memories stored yet")

# Try broader query
await search_memory.execute({
    "query": "python"  # More general
})

# Remove restrictive filters
await search_memory.execute({
    "query": "information",
    "filters": {}  # No filters
})

# Check specific metadata
await search_memory.execute({
    "query": "something",
    "filters": {"tags": ["important"]}
})
```

## Debugging Tips

### 1. Enable Debug Logging

```python
import logging
import structlog

logging.basicConfig(level=logging.DEBUG)
structlog.configure(
    processors=[
        structlog.processors.KeyValueRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)
```

### 2. Inspect Error Details

```python
def inspect_error(result):
    if result["isError"]:
        error_text = result["content"][0]["text"]
        print(f"Error: {error_text}")

        # Try to extract error code
        import re
        match = re.search(r'(ERR_\d+|VAL_\d+|EMB_\d+|DB_\d+)', error_text)
        if match:
            print(f"Error Code: {match.group(1)}")

        # Check for suggestions
        if "Ollama" in error_text:
            print("Action: Start Ollama - ollama serve")
        elif "Database" in error_text or "connection" in error_text:
            print("Action: Restart FalkorDB - docker-compose up -d")
```

### 3. Test Service Connectivity

```bash
# Test Ollama
curl -s http://localhost:11434/api/tags | jq .

# Test FalkorDB
redis-cli ping

# Test embedding
curl -s -X POST http://localhost:11434/api/embeddings \
  -d '{"model":"nomic-embed-text","prompt":"test"}' | jq .

# Test full system
python -c "import asyncio; from zapomni_mcp.tools import *; asyncio.run(test_all())"
```

### 4. Check System Resources

```bash
# Monitor CPU/Memory
watch -n 1 'ps aux | grep -E "ollama|falkordb|python"'

# Check disk space
df -h

# Check container resources
docker stats
```

## Error Response Examples

### add_memory - Invalid Text

```json
{
    "content": [
        {
            "type": "text",
            "text": "Error: text cannot be empty or contain only whitespace"
        }
    ],
    "isError": true
}
```

### search_memory - Embedding Service Down

```json
{
    "content": [
        {
            "type": "text",
            "text": "Error: Failed to process search query. Please try again."
        }
    ],
    "isError": true
}
```

### get_stats - Database Connection Failed

```json
{
    "content": [
        {
            "type": "text",
            "text": "Error: Failed to retrieve statistics - Connection refused"
        }
    ],
    "isError": true
}
```

## Best Practices

### 1. Always Check isError

```python
# Good
if result["isError"]:
    handle_error(result)
else:
    handle_success(result)

# Bad
print(result["content"][0]["text"])  # Assumes success
```

### 2. Parse Error Messages Carefully

```python
# Good
error_text = result["content"][0]["text"]
if "Ollama" in error_text:
    log_to_service_health("ollama", "offline")
elif "Database" in error_text or "connection" in error_text:
    log_to_service_health("falkordb", "offline")

# Bad
print(error_text)  # Doesn't classify error
```

### 3. Implement Retry Logic

```python
# Good
for attempt in range(3):
    try:
        result = await operation()
        if not result["isError"]:
            return result
        if is_transient(result):
            await asyncio.sleep(2 ** attempt)
        else:
            raise ValueError(...)
    except transient_error:
        await asyncio.sleep(2 ** attempt)

# Bad
result = await operation()  # No retry
```

### 4. Provide Context in Logs

```python
# Good
logger.error("operation_failed",
    operation="add_memory",
    error=error_text,
    text_length=len(text),
    correlation_id=id
)

# Bad
logger.error(error_text)  # No context
```

## Related Documentation

- **[add_memory](./tools/add_memory.md)** - Add memory error codes
- **[search_memory](./tools/search_memory.md)** - Search error codes
- **[get_stats](./tools/get_stats.md)** - Stats error codes
- **[Schemas](./schemas.md)** - Request/response formats
- **[Development Guide](../development/error_handling.md)** - Error handling strategy

---

**Last Updated**: 2025-11-24
**Version**: 1.0
**Status**: Complete
