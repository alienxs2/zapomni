# Error Handling Strategy - Module Specification

**Level:** 1 (Module)
**Author:** Goncharenko Anton aka alienxs2
**Status:** Draft
**Version:** 1.0
**Created:** 2025-11-23

## Overview

### Purpose

Defines the comprehensive error handling strategy for Zapomni across all architectural layers (MCP, Core, Database). Establishes exception hierarchy, error propagation patterns, retry strategies, logging standards, and recovery mechanisms to ensure system resilience and debuggability.

### Scope

**Included:**
- Exception hierarchy (base exceptions, specific error types)
- Error propagation between layers (MCP → Core → DB)
- Retry strategies (exponential backoff, circuit breaker patterns)
- Logging strategy (structured logs, error context, correlation IDs)
- User-facing error messages (sanitized, helpful, actionable)
- Recovery mechanisms (graceful degradation, fallback behaviors)
- Error monitoring and alerting patterns
- Testing approach for error scenarios

**Not Included:**
- Specific business logic validation (handled in individual modules)
- Performance monitoring (separate observability strategy)
- Security incident response (separate security module)
- UI error display patterns (MCP clients handle this)

### Position in Architecture

Error handling is a cross-cutting concern that affects all three packages:
```
┌─────────────────────────────────────────┐
│         zapomni_mcp                     │
│  - MCP protocol error marshalling       │
│  - User-facing error messages           │
│  - Tool execution error handling        │
└─────────────┬───────────────────────────┘
              │
              ↓
┌─────────────────────────────────────────┐
│         zapomni_core                    │
│  - Business logic exceptions            │
│  - Retry coordination                   │
│  - Error context enrichment             │
└─────────────┬───────────────────────────┘
              │
              ↓
┌─────────────────────────────────────────┐
│         zapomni_db                      │
│  - Database-specific errors             │
│  - Connection failure handling          │
│  - Transaction rollback logic           │
└─────────────────────────────────────────┘
```

## Architecture

### High-Level Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                   Error Handling Flow                        │
└──────────────────────────────────────────────────────────────┘

MCP Client Request
       ↓
┌──────────────────┐
│  MCP Server      │
│  - Validate      │ ← ValidationError
│  - Execute       │ ← ToolExecutionError
│  - Format        │
└────────┬─────────┘
         │
         ↓
┌──────────────────┐
│  Core Logic      │
│  - Process       │ ← ProcessingError
│  - Retry if      │ ← EmbeddingError
│    transient     │ ← ExtractionError
└────────┬─────────┘
         │
         ↓
┌──────────────────┐
│  Database        │
│  - Query         │ ← DatabaseError
│  - Retry if      │ ← ConnectionError
│    transient     │ ← TransactionError
└──────────────────┘
         │
         ↓
    Log Error (structlog)
         │
         ↓
    Return MCP Response
         │
         ↓
    Client Receives Error Message
```

### Key Responsibilities

1. **Exception Hierarchy Management**
   - Define base exception classes for each layer
   - Ensure exception inheritance is logical
   - Provide error codes for programmatic handling

2. **Error Propagation Control**
   - Wrap lower-level exceptions with context
   - Preserve original stack traces
   - Maintain error correlation IDs across layers

3. **Retry Logic Coordination**
   - Determine which errors are transient vs. permanent
   - Implement exponential backoff with jitter
   - Circuit breaker for cascading failures
   - Max retry attempts configuration

4. **Structured Logging**
   - Log all errors with full context
   - Include correlation IDs for tracing
   - Separate user-facing messages from debug info
   - Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL

5. **User-Facing Error Messages**
   - Sanitize errors (no internal details leaked)
   - Provide actionable guidance
   - Include error codes for support
   - Clear, non-technical language

6. **Recovery and Degradation**
   - Graceful degradation when dependencies fail
   - Fallback behaviors (e.g., fallback embedder)
   - Resource cleanup on errors
   - State consistency guarantees

## Exception Hierarchy

### Base Exception Classes

```python
# zapomni_core/exceptions.py

class ZapomniError(Exception):
    """
    Base exception for all Zapomni errors.

    Attributes:
        message: Human-readable error message
        error_code: Programmatic error code (e.g., "ERR_001")
        details: Additional context (dict)
        correlation_id: UUID for tracing across layers
        original_exception: Wrapped exception (if any)

    Example:
        raise ZapomniError(
            message="Operation failed",
            error_code="ERR_UNKNOWN",
            details={"param": "value"},
            correlation_id=uuid.uuid4()
        )
    """

    def __init__(
        self,
        message: str,
        error_code: str = "ERR_UNKNOWN",
        details: Optional[dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        original_exception: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.original_exception = original_exception

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
            "correlation_id": self.correlation_id,
            "original_error": str(self.original_exception) if self.original_exception else None
        }
```

### MCP Layer Exceptions

```python
# zapomni_mcp/exceptions.py

from zapomni_core.exceptions import ZapomniError

class MCPError(ZapomniError):
    """Base exception for MCP protocol layer."""
    pass

class ValidationError(MCPError):
    """
    Raised when tool input validation fails.

    Error Codes:
        VAL_001: Missing required field
        VAL_002: Invalid field type
        VAL_003: Field value out of range
        VAL_004: Invalid field format
    """
    pass

class ToolExecutionError(MCPError):
    """
    Raised when MCP tool execution fails.

    Error Codes:
        TOOL_001: Tool not found
        TOOL_002: Tool initialization failed
        TOOL_003: Tool execution timeout
    """
    pass

class ProtocolError(MCPError):
    """
    Raised when MCP protocol violation occurs.

    Error Codes:
        PROTO_001: Invalid JSON-RPC format
        PROTO_002: Unsupported method
        PROTO_003: Invalid response format
    """
    pass
```

### Core Layer Exceptions

```python
# zapomni_core/exceptions.py

class CoreError(ZapomniError):
    """Base exception for core processing logic."""
    pass

class ProcessingError(CoreError):
    """
    Raised when document processing fails.

    Error Codes:
        PROC_001: Chunking failed
        PROC_002: Text extraction failed
        PROC_003: Invalid document format
    """
    pass

class EmbeddingError(CoreError):
    """
    Raised when embedding generation fails.

    Error Codes:
        EMB_001: Ollama connection failed
        EMB_002: Embedding timeout
        EMB_003: Invalid embedding dimensions
        EMB_004: Model not found

    Retry Policy: YES (transient errors like timeouts, connection failures)
    """
    is_transient = True  # Flag for retry logic
    pass

class ExtractionError(CoreError):
    """
    Raised when entity/relationship extraction fails.

    Error Codes:
        EXTR_001: Entity extraction failed
        EXTR_002: Relationship detection failed
        EXTR_003: LLM response parsing failed

    Retry Policy: YES (LLM errors often transient)
    """
    is_transient = True
    pass

class SearchError(CoreError):
    """
    Raised when search operation fails.

    Error Codes:
        SEARCH_001: Vector search failed
        SEARCH_002: BM25 search failed
        SEARCH_003: Reranking failed
        SEARCH_004: No results found (not an error, but info)
    """
    pass

class CacheError(CoreError):
    """
    Raised when cache operation fails.

    Error Codes:
        CACHE_001: Redis connection failed
        CACHE_002: Cache serialization failed
        CACHE_003: Cache miss (not fatal, degrade gracefully)

    Retry Policy: NO (fail fast, fallback to non-cached)
    """
    is_transient = False
    pass
```

### Database Layer Exceptions

```python
# zapomni_db/exceptions.py

from zapomni_core.exceptions import ZapomniError

class DatabaseError(ZapomniError):
    """Base exception for database operations."""
    pass

class ConnectionError(DatabaseError):
    """
    Raised when database connection fails.

    Error Codes:
        CONN_001: FalkorDB connection refused
        CONN_002: Connection timeout
        CONN_003: Authentication failed
        CONN_004: Connection pool exhausted

    Retry Policy: YES (exponential backoff, max 3 retries)
    """
    is_transient = True
    pass

class QueryError(DatabaseError):
    """
    Raised when database query fails.

    Error Codes:
        QUERY_001: Syntax error in Cypher query
        QUERY_002: Query timeout
        QUERY_003: Index not found
        QUERY_004: Constraint violation

    Retry Policy: CONDITIONAL (only for timeouts)
    """

    def __init__(self, *args, is_timeout: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_transient = is_timeout  # Only retry on timeout
    pass

class TransactionError(DatabaseError):
    """
    Raised when transaction management fails.

    Error Codes:
        TXN_001: Transaction rollback failed
        TXN_002: Deadlock detected
        TXN_003: Transaction timeout

    Retry Policy: YES (for deadlocks and timeouts)
    """
    is_transient = True
    pass

class SchemaError(DatabaseError):
    """
    Raised when schema operation fails.

    Error Codes:
        SCHEMA_001: Index creation failed
        SCHEMA_002: Constraint creation failed
        SCHEMA_003: Migration failed

    Retry Policy: NO (structural issues, not transient)
    """
    is_transient = False
    pass
```

## Error Propagation Patterns

### Layer-to-Layer Propagation

```python
# Pattern 1: Wrap lower-level exceptions with context

async def add_memory_mcp_tool(content: str) -> dict:
    """MCP layer wraps Core layer exceptions."""
    correlation_id = str(uuid.uuid4())

    try:
        # Call core layer
        result = await core.process_memory(content, correlation_id=correlation_id)
        return {"status": "success", "memory_id": result.memory_id}

    except ValidationError:
        # Validation errors pass through (already user-facing)
        raise

    except EmbeddingError as e:
        # Wrap with MCP-layer context
        logger.error(
            "embedding_failed_in_tool",
            correlation_id=correlation_id,
            content_length=len(content),
            error=str(e)
        )
        raise ToolExecutionError(
            message="Failed to generate embeddings for memory",
            error_code="TOOL_EMB_001",
            details={"content_preview": content[:50]},
            correlation_id=correlation_id,
            original_exception=e
        ) from e

    except DatabaseError as e:
        # Wrap database errors
        logger.error(
            "database_error_in_tool",
            correlation_id=correlation_id,
            error=str(e)
        )
        raise ToolExecutionError(
            message="Failed to store memory in database",
            error_code="TOOL_DB_001",
            correlation_id=correlation_id,
            original_exception=e
        ) from e

    except Exception as e:
        # Catch-all for unexpected errors
        logger.exception(
            "unexpected_error_in_tool",
            correlation_id=correlation_id,
            error=str(e)
        )
        raise ToolExecutionError(
            message="An unexpected error occurred",
            error_code="TOOL_UNKNOWN",
            correlation_id=correlation_id,
            original_exception=e
        ) from e
```

### Stack Trace Preservation

```python
# Always use "raise ... from original_exception"
# This preserves the full stack trace

try:
    result = risky_operation()
except SpecificError as e:
    raise HigherLevelError("Context message") from e
    # ✅ Stack trace preserved

# Never do this:
try:
    result = risky_operation()
except SpecificError as e:
    raise HigherLevelError("Context message")
    # ❌ Stack trace lost!
```

### Correlation ID Flow

```python
# Correlation ID flows through all layers

# 1. MCP layer generates correlation_id
correlation_id = str(uuid.uuid4())

# 2. Pass to core layer
await core.process(data, correlation_id=correlation_id)

# 3. Core layer passes to DB layer
await db.store(data, correlation_id=correlation_id)

# 4. All logs include correlation_id
logger.info("operation", correlation_id=correlation_id, status="success")

# 5. Errors include correlation_id
raise SomeError(
    message="Failed",
    correlation_id=correlation_id
)

# Result: Can trace entire request flow through logs
```

## Retry Strategies

### Retry Decision Matrix

```python
# zapomni_core/retry.py

from dataclasses import dataclass
from typing import Callable, Type
import asyncio
import random

@dataclass
class RetryPolicy:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True  # Add randomness to avoid thundering herd

    # Which exception types to retry
    retryable_exceptions: tuple[Type[Exception], ...] = (
        ConnectionError,
        EmbeddingError,
        ExtractionError,
        TransactionError,
    )

    # Exceptions that should never be retried
    non_retryable_exceptions: tuple[Type[Exception], ...] = (
        ValidationError,
        SchemaError,
    )

async def retry_with_backoff(
    func: Callable,
    policy: RetryPolicy,
    *args,
    **kwargs
) -> Any:
    """
    Execute function with exponential backoff retry.

    Algorithm:
        1. Try function
        2. If transient error → wait and retry
        3. Delay = base^attempt * initial_delay (+ jitter)
        4. Max out at max_delay
        5. Give up after max_attempts

    Example:
        result = await retry_with_backoff(
            ollama_client.embed,
            policy=RetryPolicy(max_attempts=3),
            texts=["hello"]
        )
    """
    last_exception = None

    for attempt in range(1, policy.max_attempts + 1):
        try:
            # Attempt the operation
            result = await func(*args, **kwargs)

            # Success!
            if attempt > 1:
                logger.info(
                    "retry_succeeded",
                    attempt=attempt,
                    function=func.__name__
                )
            return result

        except policy.non_retryable_exceptions as e:
            # Permanent error, don't retry
            logger.error(
                "non_retryable_error",
                error=str(e),
                function=func.__name__
            )
            raise

        except policy.retryable_exceptions as e:
            last_exception = e

            # Last attempt? Give up
            if attempt >= policy.max_attempts:
                logger.error(
                    "retry_exhausted",
                    attempts=attempt,
                    function=func.__name__,
                    error=str(e)
                )
                raise

            # Calculate backoff delay
            delay = min(
                policy.initial_delay_seconds * (policy.exponential_base ** (attempt - 1)),
                policy.max_delay_seconds
            )

            # Add jitter (randomness)
            if policy.jitter:
                delay *= (0.5 + random.random())  # 50-150% of calculated delay

            logger.warning(
                "retry_attempt",
                attempt=attempt,
                max_attempts=policy.max_attempts,
                delay_seconds=delay,
                error=str(e),
                function=func.__name__
            )

            # Wait before retry
            await asyncio.sleep(delay)

        except Exception as e:
            # Unexpected error type, don't retry
            logger.exception(
                "unexpected_error_no_retry",
                error=str(e),
                function=func.__name__
            )
            raise

    # Should never reach here, but just in case
    if last_exception:
        raise last_exception
```

### Circuit Breaker Pattern

```python
# zapomni_core/circuit_breaker.py

from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""

    failure_threshold: int = 5  # Open after N failures
    success_threshold: int = 2  # Close after N successes in half-open
    timeout_seconds: int = 60   # How long to stay open

class CircuitBreaker:
    """
    Circuit breaker to prevent cascading failures.

    States:
        CLOSED: Normal operation, requests pass through
        OPEN: Too many failures, reject all requests immediately
        HALF_OPEN: Testing recovery, allow limited requests

    Use Case: Protect against failing external services (Ollama, FalkorDB)

    Example:
        breaker = CircuitBreaker(config)

        try:
            await breaker.call(ollama_client.embed, texts=["hello"])
        except CircuitBreakerOpenError:
            # Service is down, use fallback
            result = fallback_embedder.embed(texts=["hello"])
    """

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.lock = asyncio.Lock()

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker."""

        async with self.lock:
            # Check if circuit should transition states
            self._check_state_transition()

            # OPEN state: reject immediately
            if self.state == CircuitState.OPEN:
                logger.warning(
                    "circuit_breaker_open",
                    function=func.__name__,
                    failures=self.failure_count
                )
                raise CircuitBreakerOpenError(
                    f"Circuit breaker open for {func.__name__}"
                )

        # CLOSED or HALF_OPEN: attempt the call
        try:
            result = await func(*args, **kwargs)

            # Success!
            await self._record_success()
            return result

        except Exception as e:
            # Failure
            await self._record_failure()
            raise

    def _check_state_transition(self):
        """Check if state should transition."""

        if self.state == CircuitState.OPEN:
            # Check if timeout expired → transition to HALF_OPEN
            if self.last_failure_time:
                elapsed = datetime.now() - self.last_failure_time
                if elapsed > timedelta(seconds=self.config.timeout_seconds):
                    logger.info("circuit_breaker_half_open")
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0

    async def _record_success(self):
        """Record successful call."""
        async with self.lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1

                # Enough successes? → CLOSED
                if self.success_count >= self.config.success_threshold:
                    logger.info("circuit_breaker_closed_recovered")
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0

            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success
                self.failure_count = 0

    async def _record_failure(self):
        """Record failed call."""
        async with self.lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()

            # Too many failures? → OPEN
            if self.failure_count >= self.config.failure_threshold:
                logger.error(
                    "circuit_breaker_opened",
                    failures=self.failure_count,
                    threshold=self.config.failure_threshold
                )
                self.state = CircuitState.OPEN
                self.success_count = 0

class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass
```

### Retry Usage Example

```python
# In OllamaEmbedder class

class OllamaEmbedder:
    def __init__(self, host: str, model: str):
        self.host = host
        self.model = model
        self.retry_policy = RetryPolicy(
            max_attempts=3,
            initial_delay_seconds=1.0,
            retryable_exceptions=(EmbeddingError, httpx.TimeoutException)
        )
        self.circuit_breaker = CircuitBreaker(
            CircuitBreakerConfig(failure_threshold=5, timeout_seconds=60)
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings with retry and circuit breaker."""

        try:
            # Use circuit breaker + retry
            result = await self.circuit_breaker.call(
                retry_with_backoff,
                self._embed_batch,
                self.retry_policy,
                texts=texts
            )
            return result

        except CircuitBreakerOpenError:
            # Ollama is down, use fallback
            logger.warning("ollama_circuit_open_using_fallback")
            return await fallback_embedder.embed(texts)

        except EmbeddingError as e:
            # Retries exhausted
            logger.error("embedding_failed_after_retries", error=str(e))
            raise

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Internal method that actually calls Ollama."""
        try:
            response = await self.client.post(
                f"{self.host}/api/embeddings",
                json={"model": self.model, "prompt": texts},
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()["embeddings"]

        except httpx.TimeoutError as e:
            raise EmbeddingError(
                message="Ollama embedding timeout",
                error_code="EMB_002",
                original_exception=e
            ) from e

        except httpx.ConnectError as e:
            raise EmbeddingError(
                message="Cannot connect to Ollama",
                error_code="EMB_001",
                original_exception=e
            ) from e
```

## Logging Strategy

### Structured Logging Setup

```python
# zapomni_mcp/logging.py

import sys
import structlog
from pythonjsonlogger import jsonlogger

def setup_logging(level: str = "INFO"):
    """
    Setup structured logging with JSON output to stderr.

    Why stderr? MCP protocol uses stdout for communication,
    so all logging MUST go to stderr.

    Log Format (JSON):
        {
            "event": "operation_name",
            "timestamp": "2025-11-23T10:30:45.123Z",
            "level": "info",
            "correlation_id": "uuid-here",
            "param1": "value1",
            "param2": "value2",
            "error": "error message if any"
        }
    """

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        wrapper_class=structlog.BoundLogger,
        cache_logger_on_first_use=True,
    )
```

### Logging Patterns

```python
import structlog

logger = structlog.get_logger()

# ✅ GOOD: Structured logging with context
logger.info(
    "memory_added",
    correlation_id=correlation_id,
    memory_id=memory_id,
    chunks=len(chunks),
    embedding_time_ms=123,
    storage_time_ms=45
)

# ✅ GOOD: Error logging with full context
logger.error(
    "embedding_failed",
    correlation_id=correlation_id,
    content_length=len(content),
    error_code="EMB_002",
    error_message=str(e),
    attempt=2,
    max_attempts=3
)

# ✅ GOOD: Warning for degraded operation
logger.warning(
    "cache_miss_fallback_to_computation",
    correlation_id=correlation_id,
    cache_key=key,
    computation_time_ms=234
)

# ❌ BAD: Unstructured logging (hard to parse)
logger.info(f"Added memory {memory_id} with {len(chunks)} chunks")

# ❌ BAD: Missing context
logger.error("Embedding failed")

# ❌ BAD: Sensitive data in logs
logger.info("user_password", password=user_password)  # NEVER!
```

### Log Levels

```python
# DEBUG: Detailed diagnostic information (off in production)
logger.debug(
    "embedding_request",
    correlation_id=correlation_id,
    texts_count=len(texts),
    model=self.model,
    request_payload=payload  # Verbose details
)

# INFO: Normal operations (default level)
logger.info(
    "memory_stored",
    correlation_id=correlation_id,
    memory_id=memory_id,
    duration_ms=duration
)

# WARNING: Degraded performance or fallback used
logger.warning(
    "ollama_slow_response",
    correlation_id=correlation_id,
    response_time_ms=5000,
    threshold_ms=1000
)

# ERROR: Operation failed, needs attention
logger.error(
    "database_connection_failed",
    correlation_id=correlation_id,
    error_code="CONN_001",
    retry_attempt=2
)

# CRITICAL: System failure, immediate action required
logger.critical(
    "falkordb_unreachable",
    correlation_id=correlation_id,
    downtime_seconds=300,
    impact="all_operations_failing"
)
```

### Error Context Enrichment

```python
# Add context at each layer

# DB Layer: Basic error info
try:
    result = graph.query(cypher)
except Exception as e:
    logger.error(
        "query_failed",
        correlation_id=correlation_id,
        query_preview=cypher[:100],
        error=str(e)
    )
    raise QueryError(...) from e

# Core Layer: Add business context
try:
    await db.store(memory)
except QueryError as e:
    logger.error(
        "memory_storage_failed",
        correlation_id=correlation_id,
        memory_id=memory.id,
        chunk_count=len(memory.chunks),
        error=str(e)
    )
    raise ProcessingError(...) from e

# MCP Layer: Add user context
try:
    await core.process(content)
except ProcessingError as e:
    logger.error(
        "tool_execution_failed",
        correlation_id=correlation_id,
        tool_name="add_memory",
        content_length=len(content),
        user_metadata=metadata,
        error=str(e)
    )
    raise ToolExecutionError(...) from e
```

## User-Facing Error Messages

### Error Message Guidelines

```python
# Guidelines:
# 1. Clear, non-technical language
# 2. Explain what happened
# 3. Explain why (if known)
# 4. Suggest action user can take
# 5. Include error code for support

# ✅ GOOD User-Facing Message
{
    "error": "Unable to connect to the embedding service. This is likely a temporary issue. Please try again in a moment. If the problem persists, check that Ollama is running (Error: EMB_001)",
    "error_code": "EMB_001",
    "suggested_action": "Verify Ollama is running: ollama list",
    "correlation_id": "uuid-here"
}

# ❌ BAD User-Facing Message
{
    "error": "httpx.ConnectError: [Errno 61] Connection refused",
    # ↑ Technical jargon, no guidance
}

# ❌ BAD User-Facing Message
{
    "error": "An error occurred",
    # ↑ Too vague, not actionable
}
```

### Error Message Sanitization

```python
# zapomni_mcp/error_messages.py

def sanitize_error_for_user(error: Exception, correlation_id: str) -> dict[str, Any]:
    """
    Convert exception to user-friendly error message.

    Removes:
        - Stack traces
        - Internal paths
        - Database connection strings
        - Sensitive configuration

    Adds:
        - Clear explanation
        - Suggested action
        - Error code
        - Correlation ID for support
    """

    # Map error types to user messages
    error_messages = {
        ValidationError: {
            "message": "The input provided is invalid. {details}",
            "action": "Please check your input and try again."
        },
        EmbeddingError: {
            "message": "Unable to generate embeddings. The embedding service may be temporarily unavailable.",
            "action": "Verify Ollama is running (ollama list) and try again."
        },
        ConnectionError: {
            "message": "Cannot connect to the database. This is usually a temporary issue.",
            "action": "Check that FalkorDB is running (docker ps) and retry."
        },
        ProcessingError: {
            "message": "Failed to process your request due to an internal error.",
            "action": "Please try again. If the issue persists, contact support with error code {error_code}."
        },
    }

    # Get error-specific template
    error_type = type(error)
    template = error_messages.get(error_type, {
        "message": "An unexpected error occurred.",
        "action": "Please try again or contact support."
    })

    # Extract error details (if safe to show)
    details = ""
    if isinstance(error, ValidationError):
        # Validation errors are safe to show
        details = error.message

    # Build user message
    user_message = template["message"].format(
        details=details,
        error_code=getattr(error, 'error_code', 'UNKNOWN')
    )

    return {
        "error": user_message,
        "error_code": getattr(error, 'error_code', 'UNKNOWN'),
        "suggested_action": template["action"],
        "correlation_id": correlation_id,
        # Internal details go to logs only, not to user
    }
```

### MCP Error Response Format

```python
# zapomni_mcp/tools/base.py

async def execute_tool_safe(
    tool: MCPTool,
    arguments: dict[str, Any]
) -> dict[str, Any]:
    """
    Execute MCP tool with comprehensive error handling.

    Returns MCP-compliant error response on failure.
    """
    correlation_id = str(uuid.uuid4())

    try:
        # Execute tool
        result = await tool.execute(arguments, correlation_id=correlation_id)

        # Success response
        return {
            "content": [
                {
                    "type": "text",
                    "text": result.to_user_message()
                }
            ],
            "isError": False
        }

    except Exception as e:
        # Log full error details
        logger.exception(
            "tool_execution_failed",
            correlation_id=correlation_id,
            tool_name=tool.name,
            arguments=arguments,
            error=str(e)
        )

        # Sanitize error for user
        user_error = sanitize_error_for_user(e, correlation_id)

        # MCP error response
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Error: {user_error['error']}\n\n"
                           f"Suggested Action: {user_error['suggested_action']}\n\n"
                           f"Error Code: {user_error['error_code']}\n"
                           f"Correlation ID: {user_error['correlation_id']}"
                }
            ],
            "isError": True
        }
```

## Recovery Mechanisms

### Graceful Degradation

```python
# Example: Embedding service degradation

async def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings with graceful degradation.

    Fallback chain:
        1. Try Ollama (preferred)
        2. Try sentence-transformers (local fallback)
        3. Return zero vectors (last resort)
    """

    # Try primary: Ollama
    try:
        embeddings = await ollama_embedder.embed(texts)
        return embeddings

    except CircuitBreakerOpenError:
        logger.warning("ollama_unavailable_using_fallback")
        # Circuit open, skip straight to fallback

    except EmbeddingError as e:
        logger.warning(
            "ollama_failed_using_fallback",
            error=str(e)
        )

    # Try fallback: sentence-transformers
    try:
        embeddings = sentence_transformer_embedder.embed(texts)
        logger.info("using_fallback_embedder", count=len(texts))
        return embeddings

    except Exception as e:
        logger.error(
            "fallback_embedder_failed",
            error=str(e)
        )

    # Last resort: zero vectors (allows system to continue)
    logger.critical(
        "all_embedders_failed_using_zero_vectors",
        count=len(texts)
    )
    return [[0.0] * 768 for _ in texts]  # 768-dim zero vectors
```

### Resource Cleanup

```python
# Pattern: Use context managers for cleanup

class FalkorDBClient:
    async def __aenter__(self):
        """Initialize resources."""
        self.connection = await self._connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup resources, even on error."""
        if self.connection:
            await self.connection.close()

        # Log if exiting due to error
        if exc_type:
            logger.error(
                "falkordb_client_exited_with_error",
                error_type=exc_type.__name__,
                error=str(exc_val)
            )

# Usage:
async def process_memory():
    async with FalkorDBClient() as db:
        result = await db.query(...)
        # Connection auto-closes, even if error occurs
```

### State Consistency

```python
# Pattern: Transactional operations with rollback

async def add_memory_transactional(memory: Memory) -> str:
    """
    Add memory with transactional consistency.

    If any step fails, roll back all changes.
    """

    transaction = await db.begin_transaction()

    try:
        # Step 1: Create Memory node
        memory_id = await transaction.create_memory(memory)

        # Step 2: Create Chunk nodes
        chunk_ids = await transaction.create_chunks(memory_id, memory.chunks)

        # Step 3: Create relationships
        await transaction.link_chunks_to_memory(memory_id, chunk_ids)

        # All succeeded, commit
        await transaction.commit()

        logger.info(
            "memory_added_successfully",
            memory_id=memory_id,
            chunks=len(chunk_ids)
        )

        return memory_id

    except Exception as e:
        # Any failure → rollback
        await transaction.rollback()

        logger.error(
            "memory_addition_failed_rolled_back",
            error=str(e)
        )

        raise ProcessingError(
            message="Failed to add memory, changes rolled back",
            error_code="PROC_TXN_001",
            original_exception=e
        ) from e
```

## Design Decisions

### Decision 1: Exception Hierarchy with `is_transient` Flag

**Context:** Need to determine which errors should trigger retries

**Options Considered:**
- **Option A:** Separate exception classes for transient vs permanent errors
  - Pros: Clear separation
  - Cons: Double the number of exception classes
- **Option B:** Use `is_transient` attribute on exceptions
  - Pros: Flexible, single class hierarchy
  - Cons: Need to check attribute
- **Option C:** Retry logic in catch blocks (no metadata)
  - Pros: Simple
  - Cons: Scattered retry logic, hard to maintain

**Chosen:** Option B - `is_transient` flag

**Rationale:**
- Keeps exception hierarchy clean
- Allows per-instance configuration (some QueryErrors transient, some not)
- Centralized retry logic can check flag
- Easy to adjust without changing class hierarchy

### Decision 2: Structured Logging to stderr (JSON)

**Context:** Need comprehensive logging without breaking MCP protocol

**Options Considered:**
- **Option A:** Log to stdout (human-readable)
  - Pros: Easy to read during development
  - Cons: Breaks MCP protocol (stdout reserved)
- **Option B:** Log to file
  - Pros: Doesn't interfere with stdout
  - Cons: File management complexity, harder to view in real-time
- **Option C:** Log to stderr (JSON structured)
  - Pros: Standard practice, machine-parseable, doesn't break MCP
  - Cons: Less human-readable

**Chosen:** Option C - stderr JSON structured logging

**Rationale:**
- MCP protocol requires stdout exclusively for protocol messages
- stderr is standard for logging in CLI tools
- JSON structured logs enable:
  - Easy parsing by log aggregators
  - Correlation ID tracing
  - Programmatic analysis
- Can still view with `jq` for human readability: `stderr | jq`

### Decision 3: Correlation IDs for Request Tracing

**Context:** Need to trace errors across multi-layer architecture

**Options Considered:**
- **Option A:** No correlation (just log errors individually)
  - Pros: Simple
  - Cons: Cannot trace request flow across layers
- **Option B:** Correlation IDs (UUIDs)
  - Pros: Full request tracing, debugging easier
  - Cons: Need to pass IDs through all layers
- **Option C:** Thread-local storage for IDs
  - Pros: No manual passing
  - Cons: Doesn't work with asyncio, complex

**Chosen:** Option B - Explicit correlation IDs

**Rationale:**
- Essential for debugging distributed/layered systems
- Standard practice in observability
- Works perfectly with async code (pass as parameter)
- Minimal overhead (just a UUID string)
- Enables powerful log analysis:
  ```bash
  grep "correlation_id=abc123" logs.json | jq
  ```

### Decision 4: Circuit Breaker for External Services

**Context:** Need to prevent cascading failures when Ollama/FalkorDB down

**Options Considered:**
- **Option A:** Just retry with backoff
  - Pros: Simple
  - Cons: Wastes resources retrying known-dead service
- **Option B:** Circuit breaker pattern
  - Pros: Fast-fail when service down, auto-recovery
  - Cons: More complex
- **Option C:** Health check before each request
  - Pros: Always knows service status
  - Cons: Double the requests, latency overhead

**Chosen:** Option B - Circuit breaker with retry

**Rationale:**
- Industry best practice for resilience
- Prevents resource waste (stop trying when clearly down)
- Auto-recovery (test again after timeout)
- Protects against thundering herd
- Works well with retry strategy (circuit breaker wraps retry logic)

### Decision 5: Sanitized User-Facing Error Messages

**Context:** Need to balance helpfulness with security/clarity

**Options Considered:**
- **Option A:** Show raw exceptions to user
  - Pros: Maximum debugging info
  - Cons: Leaks internals, confusing to users, security risk
- **Option B:** Generic "error occurred" message
  - Pros: Safe
  - Cons: Not helpful, frustrates users
- **Option C:** Sanitized, actionable messages
  - Pros: Helpful, safe, actionable
  - Cons: Requires message mapping

**Chosen:** Option C - Sanitized actionable messages

**Rationale:**
- Security: Don't leak internal paths, connection strings
- UX: Users need actionable guidance, not stack traces
- Supportability: Include error codes and correlation IDs
- Developer experience: Full details in logs (stderr)
- Best practice: Separation of internal vs external error representation

## Non-Functional Requirements

### Performance

**Latency:**
- Error handling overhead: < 1ms per error
- Retry backoff: Configurable (default 1s initial, max 30s)
- Circuit breaker state check: < 0.1ms

**Throughput:**
- Logging: Asynchronous, non-blocking (structlog handles this)
- Error serialization: < 0.5ms per exception

**Resource Usage:**
- Memory: O(1) per error (no unbounded error history)
- CPU: Minimal (exception creation, logging)

### Reliability

**Error Handling Coverage:**
- ✅ 100% of public APIs have error handling
- ✅ All database operations wrapped in try-except
- ✅ All external service calls have retry logic
- ✅ All async operations have cleanup (async with)

**Recovery Guarantees:**
- Transactional operations: ACID guarantees via FalkorDB
- Idempotency: Retry-safe operations (same request → same result)
- Graceful degradation: System continues with reduced functionality

**Monitoring:**
- All errors logged with correlation IDs
- Error rates trackable via log aggregation
- Circuit breaker state changes logged

### Security

**Error Message Sanitization:**
- ✅ No stack traces to users
- ✅ No internal paths exposed
- ✅ No connection strings in logs
- ✅ No sensitive data (passwords, tokens) logged

**Error Code Disclosure:**
- Error codes are safe to share (just identifiers)
- Correlation IDs are UUIDs (no security risk)

### Debuggability

**Tracing:**
- Correlation IDs flow through all layers
- Full stack traces preserved in exceptions
- Structured logs enable grep/jq analysis

**Context:**
- Every error logged with full context:
  - What operation failed
  - Input parameters (sanitized)
  - Layer where error occurred
  - Time and duration
  - Retry attempts

**Tools:**
- Log aggregation friendly (JSON format)
- Can trace entire request flow:
  ```bash
  grep "correlation_id=uuid" logs | jq -s 'sort_by(.timestamp)'
  ```

## Testing Strategy

### Unit Testing

**Exception Creation Tests:**
```python
def test_zapomni_error_initialization():
    """Test base exception initialization."""
    error = ZapomniError(
        message="Test error",
        error_code="TEST_001",
        details={"key": "value"}
    )

    assert error.message == "Test error"
    assert error.error_code == "TEST_001"
    assert error.details == {"key": "value"}
    assert error.correlation_id is not None  # Auto-generated

def test_zapomni_error_to_dict():
    """Test exception serialization."""
    error = ValidationError(
        message="Invalid input",
        error_code="VAL_001"
    )

    error_dict = error.to_dict()

    assert error_dict["error"] == "ValidationError"
    assert error_dict["message"] == "Invalid input"
    assert error_dict["error_code"] == "VAL_001"
```

**Retry Logic Tests:**
```python
@pytest.mark.asyncio
async def test_retry_success_on_second_attempt():
    """Test successful retry after one failure."""
    call_count = 0

    async def flaky_function():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ConnectionError("First attempt fails")
        return "success"

    result = await retry_with_backoff(
        flaky_function,
        RetryPolicy(max_attempts=3)
    )

    assert result == "success"
    assert call_count == 2

@pytest.mark.asyncio
async def test_retry_exhausted():
    """Test retry gives up after max attempts."""
    async def always_fails():
        raise ConnectionError("Always fails")

    with pytest.raises(ConnectionError):
        await retry_with_backoff(
            always_fails,
            RetryPolicy(max_attempts=3)
        )

@pytest.mark.asyncio
async def test_non_retryable_error_no_retry():
    """Test that non-retryable errors fail immediately."""
    call_count = 0

    async def validation_error():
        nonlocal call_count
        call_count += 1
        raise ValidationError("Invalid input")

    with pytest.raises(ValidationError):
        await retry_with_backoff(
            validation_error,
            RetryPolicy(max_attempts=3)
        )

    assert call_count == 1  # No retries
```

**Circuit Breaker Tests:**
```python
@pytest.mark.asyncio
async def test_circuit_breaker_opens_after_failures():
    """Test circuit opens after threshold failures."""
    breaker = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3))

    async def failing_function():
        raise Exception("Simulated failure")

    # First 3 failures
    for _ in range(3):
        with pytest.raises(Exception):
            await breaker.call(failing_function)

    # Circuit should be open now
    assert breaker.state == CircuitState.OPEN

    # Next call should fail fast
    with pytest.raises(CircuitBreakerOpenError):
        await breaker.call(failing_function)

@pytest.mark.asyncio
async def test_circuit_breaker_half_open_recovery():
    """Test circuit recovers via half-open state."""
    breaker = CircuitBreaker(CircuitBreakerConfig(
        failure_threshold=2,
        success_threshold=2,
        timeout_seconds=1
    ))

    async def flaky_function():
        if breaker.state == CircuitState.HALF_OPEN:
            return "success"  # Succeed in half-open
        raise Exception("Fail in closed")

    # Open the circuit
    for _ in range(2):
        with pytest.raises(Exception):
            await breaker.call(flaky_function)

    assert breaker.state == CircuitState.OPEN

    # Wait for timeout
    await asyncio.sleep(1.1)

    # Should transition to half-open and succeed
    result = await breaker.call(flaky_function)
    assert result == "success"
```

### Integration Testing

**End-to-End Error Flow Tests:**
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_error_propagation_mcp_to_db(mcp_server, db_client):
    """Test error propagates from DB to MCP with context."""

    # Simulate DB failure
    db_client.query = Mock(side_effect=QueryError(
        message="Connection lost",
        error_code="QUERY_002"
    ))

    # Call MCP tool
    response = await mcp_server.call_tool(
        "add_memory",
        {"content": "test"}
    )

    # Verify error response
    assert response["isError"] is True
    assert "database" in response["content"][0]["text"].lower()
    assert "QUERY_002" in response["content"][0]["text"]

    # Verify logging happened
    # (Check stderr logs contain correlation_id and error details)

@pytest.mark.integration
async def test_retry_with_real_ollama(ollama_client):
    """Test retry logic with real Ollama service."""
    # Stop Ollama temporarily
    subprocess.run(["killall", "ollama"])

    # Should fail and retry
    with pytest.raises(EmbeddingError):
        await ollama_client.embed(["test"])

    # Restart Ollama
    subprocess.run(["ollama", "serve"])
    await asyncio.sleep(2)  # Wait for startup

    # Should succeed after retries
    result = await ollama_client.embed(["test"])
    assert len(result) == 1
    assert len(result[0]) == 768
```

**Graceful Degradation Tests:**
```python
@pytest.mark.integration
async def test_fallback_embedder_when_ollama_down(embedding_service):
    """Test fallback to sentence-transformers when Ollama fails."""

    # Kill Ollama
    embedding_service.ollama_breaker._force_open()  # Force circuit open

    # Should use fallback
    embeddings = await embedding_service.embed(["hello", "world"])

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 768

    # Verify fallback was logged
    # (Check logs for "using_fallback_embedder")
```

## Future Considerations

### Observability Enhancements

**Planned:**
- Distributed tracing integration (OpenTelemetry)
- Metrics export (Prometheus format)
- Error dashboards (Grafana)
- Alert rules (error rate thresholds)

**Example:**
```python
# Future: OpenTelemetry tracing
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

async def add_memory(content: str):
    with tracer.start_as_current_span("add_memory") as span:
        span.set_attribute("content.length", len(content))

        try:
            result = await process(content)
            span.set_attribute("result.memory_id", result.id)
            return result
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR))
            raise
```

### Error Recovery Automation

**Planned:**
- Auto-restart failed background tasks
- Automatic cache warming after recovery
- Self-healing database connections
- Predictive failure detection

### Enhanced User Feedback

**Planned:**
- Progress notifications during retries
- Estimated recovery time
- Alternative action suggestions
- Error history for debugging

## References

### Internal Documents
- [product.md](../../steering/product.md) - Vision and features
- [tech.md](../../steering/tech.md) - Technology decisions
- [structure.md](../../steering/structure.md) - Project structure
- [zapomni_mcp_module.md](zapomni_mcp_module.md) - MCP layer
- [zapomni_core_module.md](zapomni_core_module.md) - Core layer
- [zapomni_db_module.md](zapomni_db_module.md) - Database layer

### External Resources
- Python Exception Hierarchy: https://docs.python.org/3/library/exceptions.html
- structlog Documentation: https://www.structlog.org/
- Circuit Breaker Pattern: https://martinfowler.com/bliki/CircuitBreaker.html
- Retry Strategies: https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
- OpenTelemetry: https://opentelemetry.io/docs/python/

### Best Practices
- Google SRE Book - Error Handling: https://sre.google/sre-book/
- 12-Factor App - Logs: https://12factor.net/logs
- Microsoft Azure - Retry Guidance: https://docs.microsoft.com/azure/architecture/best-practices/retry-service-specific

---

**Document Status:** Draft v1.0
**Created:** 2025-11-23
**Authors:** Goncharenko Anton aka alienxs2 + Claude Code
**Copyright:** Copyright (c) 2025 Goncharenko Anton aka alienxs2
**License:** MIT License
**Ready for Review:** Yes ✅
