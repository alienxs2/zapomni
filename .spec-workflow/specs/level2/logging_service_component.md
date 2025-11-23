# LoggingService - Component Specification

**Level:** 2 (Component)
**Module:** shared (cross-cutting)
**Author:** Goncharenko Anton aka alienxs2
**Status:** Draft
**Version:** 1.0
**Created:** 2025-11-23

## Overview

### Purpose

LoggingService is a centralized structured logging service built on structlog that provides consistent, machine-readable logging across all Zapomni modules. It configures global logging settings, creates module-specific loggers, and enriches log entries with contextual metadata (correlation IDs, timestamps, operation names, performance metrics).

### Responsibilities

1. Configure global structured logging infrastructure (JSON output to stderr)
2. Provide module/component-specific loggers with automatic context enrichment
3. Log operations with full context (correlation_id, module, operation, metadata)
4. Log errors with stack traces, error codes, and recovery context
5. Log performance metrics for operations (duration, resource usage)
6. Ensure thread-safety for concurrent logging across async operations
7. Prevent sensitive data leakage in logs (sanitize passwords, tokens, PII)

### Position in Module

LoggingService is the foundational component of the Error Handling Strategy, used by all other components:

```
┌────────────────────────────────────────┐
│     Error Handling Strategy            │
├────────────────────────────────────────┤
│  LoggingService (this)                 │
│    ↓                                   │
│  ExceptionHierarchy                    │
│    ↓                                   │
│  RetryStrategy                         │
│    ↓                                   │
│  CircuitBreaker                        │
│    ↓                                   │
│  ErrorSanitizer                        │
└────────────────────────────────────────┘
```

All components depend on LoggingService for consistent logging.

## Class Definition

### Class Diagram

```
┌─────────────────────────────────────────────┐
│          LoggingService                     │
├─────────────────────────────────────────────┤
│ - _configured: bool                         │
│ - _log_level: str                           │
│ - _loggers: dict[str, BoundLogger]          │
│ - _sensitive_keys: set[str]                 │
├─────────────────────────────────────────────┤
│ + configure_logging(level, format)          │
│ + get_logger(name) -> BoundLogger           │
│ + log_operation(operation, metadata)        │
│ + log_error(error, context)                 │
│ + log_performance(operation, duration)      │
│ - _sanitize_metadata(data) -> dict          │
│ - _setup_processors() -> list               │
└─────────────────────────────────────────────┘
```

### Full Class Signature

```python
import sys
from typing import Any, Optional
from dataclasses import dataclass
import structlog
from structlog.types import Processor
from pythonjsonlogger import jsonlogger


@dataclass
class LoggingConfig:
    """
    Configuration for LoggingService.

    Attributes:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Output format ("json" or "console" for dev)
        output_stream: Output destination (default: sys.stderr)
        sanitize_sensitive: Whether to sanitize sensitive data
        sensitive_keys: Set of keys to sanitize (e.g., "password", "api_key")

    Example:
        config = LoggingConfig(
            level="INFO",
            format="json",
            sensitive_keys={"password", "api_key", "token"}
        )
    """
    level: str = "INFO"
    format: str = "json"  # "json" or "console"
    output_stream: Any = sys.stderr
    sanitize_sensitive: bool = True
    sensitive_keys: set[str] = None

    def __post_init__(self):
        if self.sensitive_keys is None:
            self.sensitive_keys = {
                "password", "passwd", "pwd",
                "api_key", "apikey", "key",
                "token", "access_token", "refresh_token",
                "secret", "auth", "authorization",
                "credit_card", "ssn", "social_security"
            }


class LoggingService:
    """
    Centralized structured logging service using structlog.

    Provides consistent, context-enriched, machine-readable logging
    across all Zapomni modules. Outputs JSON logs to stderr (MCP-compatible).

    Features:
        - Structured JSON logging
        - Automatic context enrichment (timestamps, correlation IDs)
        - Module-specific loggers
        - Sensitive data sanitization
        - Thread-safe operation
        - Performance metric logging

    Attributes:
        _configured: Whether logging has been initialized
        _log_level: Current log level
        _config: Logging configuration
        _loggers: Cache of module-specific loggers
        _sensitive_keys: Set of keys to sanitize

    Example:
        # Setup logging once at startup
        LoggingService.configure_logging(level="INFO", format="json")

        # Get logger for a module
        logger = LoggingService.get_logger("zapomni.mcp.tools")

        # Log operations
        logger.info(
            "tool_executed",
            tool_name="add_memory",
            correlation_id="uuid-here",
            duration_ms=123
        )

        # Log errors
        try:
            risky_operation()
        except Exception as e:
            logger.error(
                "operation_failed",
                error=str(e),
                error_type=type(e).__name__,
                correlation_id="uuid-here"
            )
    """

    # Class-level state
    _configured: bool = False
    _log_level: str = "INFO"
    _config: Optional[LoggingConfig] = None
    _loggers: dict[str, structlog.BoundLogger] = {}
    _sensitive_keys: set[str] = set()

    @classmethod
    def configure_logging(
        cls,
        level: str = "INFO",
        format: str = "json",
        config: Optional[LoggingConfig] = None
    ) -> None:
        """
        Configure global structured logging infrastructure.

        Sets up structlog with JSON output to stderr, configures log level,
        and initializes processors for context enrichment.

        This should be called ONCE at application startup before any logging.

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            format: Output format ("json" or "console")
            config: Optional LoggingConfig for advanced configuration

        Raises:
            ValueError: If level is invalid
            RuntimeError: If called after logging already configured

        Example:
            # Simple usage
            LoggingService.configure_logging(level="DEBUG", format="json")

            # Advanced usage
            config = LoggingConfig(
                level="INFO",
                format="json",
                sensitive_keys={"password", "secret"}
            )
            LoggingService.configure_logging(config=config)
        """

    @classmethod
    def get_logger(cls, name: str) -> structlog.BoundLogger:
        """
        Get a module/component-specific logger.

        Returns a cached logger if already created, otherwise creates
        a new logger with the given name for context.

        The logger name typically follows Python module path convention:
        "zapomni.mcp.tools.add_memory"

        Args:
            name: Logger name (typically module path)

        Returns:
            BoundLogger instance with automatic context enrichment

        Raises:
            RuntimeError: If logging not configured yet

        Example:
            # In a module/class
            logger = LoggingService.get_logger(__name__)

            # Use the logger
            logger.info(
                "processing_started",
                correlation_id="uuid",
                input_size=1024
            )
        """

    @classmethod
    def log_operation(
        cls,
        operation: str,
        correlation_id: str,
        metadata: Optional[dict[str, Any]] = None,
        logger_name: str = "zapomni",
        level: str = "info"
    ) -> None:
        """
        Log an operation with full context.

        Convenience method for logging operations with standardized structure.
        Automatically sanitizes sensitive data in metadata.

        Args:
            operation: Operation name (e.g., "add_memory", "search_memory")
            correlation_id: UUID for tracing this operation
            metadata: Additional context (parameters, results, metrics)
            logger_name: Which logger to use (default: "zapomni")
            level: Log level (default: "info")

        Example:
            LoggingService.log_operation(
                operation="chunk_document",
                correlation_id="uuid-here",
                metadata={
                    "document_length": 5000,
                    "chunk_size": 512,
                    "chunks_created": 10,
                    "duration_ms": 45
                }
            )
        """

    @classmethod
    def log_error(
        cls,
        error: Exception,
        correlation_id: str,
        context: Optional[dict[str, Any]] = None,
        logger_name: str = "zapomni",
        include_stack_trace: bool = True
    ) -> None:
        """
        Log an error with full context and stack trace.

        Convenience method for consistent error logging. Automatically
        extracts error type, message, error code (if ZapomniError),
        and optionally includes stack trace.

        Args:
            error: Exception instance
            correlation_id: UUID for tracing
            context: Additional context about where/why error occurred
            logger_name: Which logger to use (default: "zapomni")
            include_stack_trace: Whether to include full stack trace

        Example:
            try:
                result = dangerous_operation()
            except DatabaseError as e:
                LoggingService.log_error(
                    error=e,
                    correlation_id="uuid-here",
                    context={
                        "operation": "insert_memory",
                        "memory_id": "mem_123",
                        "retry_attempt": 2
                    }
                )
                raise
        """

    @classmethod
    def log_performance(
        cls,
        operation: str,
        duration_ms: float,
        correlation_id: str,
        metadata: Optional[dict[str, Any]] = None,
        logger_name: str = "zapomni"
    ) -> None:
        """
        Log performance metrics for an operation.

        Records operation duration and optional resource usage metrics.
        Useful for identifying slow operations and optimization opportunities.

        Args:
            operation: Operation name
            duration_ms: Duration in milliseconds
            correlation_id: UUID for tracing
            metadata: Additional metrics (memory usage, CPU, etc.)
            logger_name: Which logger to use

        Example:
            import time

            start = time.perf_counter()
            result = embed_texts(texts)
            duration_ms = (time.perf_counter() - start) * 1000

            LoggingService.log_performance(
                operation="embed_texts",
                duration_ms=duration_ms,
                correlation_id="uuid-here",
                metadata={
                    "text_count": len(texts),
                    "embedding_dimensions": 768,
                    "model": "nomic-embed-text"
                }
            )
        """

    @classmethod
    def _sanitize_metadata(cls, data: dict[str, Any]) -> dict[str, Any]:
        """
        Sanitize sensitive data from metadata before logging.

        Replaces values for sensitive keys with "[REDACTED]".
        Recursively processes nested dictionaries and lists.

        Args:
            data: Metadata dictionary to sanitize

        Returns:
            Sanitized copy of metadata

        Example:
            metadata = {
                "username": "alice",
                "password": "secret123",
                "api_key": "sk-abc123"
            }

            sanitized = LoggingService._sanitize_metadata(metadata)
            # Returns:
            # {
            #     "username": "alice",
            #     "password": "[REDACTED]",
            #     "api_key": "[REDACTED]"
            # }
        """

    @classmethod
    def _setup_processors(cls) -> list[Processor]:
        """
        Setup structlog processors based on configuration.

        Returns:
            List of structlog processors for log processing pipeline

        Processors (in order):
            1. filter_by_level: Filter logs below configured level
            2. add_logger_name: Add logger name to context
            3. add_log_level: Add log level to context
            4. TimeStamper: Add ISO timestamp
            5. StackInfoRenderer: Render stack info if requested
            6. format_exc_info: Format exception info
            7. UnicodeDecoder: Ensure unicode handling
            8. JSONRenderer or ConsoleRenderer: Final output format
        """
```

## Dependencies

### Component Dependencies

**Internal (Zapomni):**
- None (LoggingService is foundational, no other Zapomni components)

**Usage by other components:**
- ExceptionHierarchy: Uses LoggingService to log exception creation/handling
- RetryStrategy: Uses LoggingService to log retry attempts
- CircuitBreaker: Uses LoggingService to log state changes
- All MCP tools: Use LoggingService for operation logging
- All Core processors: Use LoggingService for processing logs
- All DB operations: Use LoggingService for query/transaction logs

### External Libraries

```python
# Required dependencies
structlog==23.2.0  # Structured logging framework
python-json-logger==2.0.7  # JSON log formatting
```

**Dependency Rationale:**
- **structlog**: Industry-standard structured logging for Python
  - Context binding (automatic enrichment)
  - Processor pipeline (flexible log processing)
  - Thread-safe by design
  - Excellent async support
- **python-json-logger**: Formats logs as JSON
  - Machine-parseable output
  - Log aggregation friendly (Elasticsearch, Splunk, etc.)
  - Consistent structure

### Dependency Injection

LoggingService uses class methods (singleton pattern) - no dependency injection needed. Configuration passed via `configure_logging()` at startup.

## State Management

### Attributes

**Class-level attributes (shared across all instances):**

- `_configured: bool`
  - **Type:** bool
  - **Purpose:** Track whether logging has been initialized
  - **Lifetime:** Application lifetime
  - **Initial value:** False

- `_log_level: str`
  - **Type:** str
  - **Purpose:** Current log level
  - **Lifetime:** Application lifetime
  - **Initial value:** "INFO"

- `_config: Optional[LoggingConfig]`
  - **Type:** Optional[LoggingConfig]
  - **Purpose:** Logging configuration
  - **Lifetime:** Application lifetime
  - **Initial value:** None

- `_loggers: dict[str, BoundLogger]`
  - **Type:** dict mapping logger name to BoundLogger
  - **Purpose:** Cache of created loggers
  - **Lifetime:** Application lifetime
  - **Initial value:** {} (empty dict)

- `_sensitive_keys: set[str]`
  - **Type:** set[str]
  - **Purpose:** Keys to sanitize in metadata
  - **Lifetime:** Application lifetime
  - **Initial value:** Set from config or default

### State Transitions

```
Application Startup
    ↓
[configure_logging called]
    ↓
State: _configured = True
    ↓
[get_logger("module.name") called]
    ↓
State: _loggers["module.name"] = BoundLogger(...)
    ↓
[Logging operations]
    ↓
(State unchanged, logs written)
    ↓
Application Shutdown
```

### Thread Safety

**Thread-Safe:** YES

**Mechanisms:**
- structlog is thread-safe by design
- Class-level state modifications only during startup (before threading)
- Logger cache (`_loggers`) only read after configuration
- No mutable state during logging operations
- Each log call creates new context dict (no shared mutable state)

**Concurrent Usage:**
```python
# Safe: Multiple async tasks logging simultaneously
async def task1():
    logger = LoggingService.get_logger("task1")
    logger.info("task1_operation", data="value1")

async def task2():
    logger = LoggingService.get_logger("task2")
    logger.info("task2_operation", data="value2")

await asyncio.gather(task1(), task2())
# Both log safely, no interference
```

## Public Methods (Detailed)

### Method 1: `configure_logging`

**Signature:**
```python
@classmethod
def configure_logging(
    cls,
    level: str = "INFO",
    format: str = "json",
    config: Optional[LoggingConfig] = None
) -> None
```

**Purpose:**
Initialize global structured logging infrastructure. Sets up structlog with JSON output to stderr, configures log level, and initializes processors for automatic context enrichment.

**Parameters:**

- `level`: str
  - **Description:** Log level for filtering messages
  - **Constraints:**
    - Must be one of: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    - Case-insensitive
  - **Default:** "INFO"
  - **Example:** "DEBUG"

- `format`: str
  - **Description:** Output format for logs
  - **Constraints:**
    - Must be "json" or "console"
    - "json": Machine-readable JSON (production)
    - "console": Human-readable colored output (development)
  - **Default:** "json"
  - **Example:** "json"

- `config`: Optional[LoggingConfig]
  - **Description:** Advanced configuration object
  - **Constraints:**
    - If provided, overrides `level` and `format` parameters
    - Must be valid LoggingConfig instance
  - **Default:** None
  - **Example:**
    ```python
    LoggingConfig(
        level="DEBUG",
        format="json",
        sensitive_keys={"password", "api_key"}
    )
    ```

**Returns:**
- **Type:** None
- **Side effects:**
  - Configures structlog global state
  - Sets `cls._configured = True`
  - Sets `cls._log_level`
  - Sets `cls._config`
  - Initializes `cls._sensitive_keys`

**Raises:**
- `ValueError`: When level is not a valid log level
- `RuntimeError`: When called after logging already configured

**Preconditions:**
- Must be called before any logging operations
- Should be called exactly once at application startup

**Postconditions:**
- Logging is configured globally
- `cls._configured == True`
- structlog processors are initialized
- Future `get_logger()` calls will work

**Algorithm Outline:**
```
1. Check if already configured → raise RuntimeError
2. Validate level parameter → raise ValueError if invalid
3. Use config if provided, otherwise create from parameters
4. Setup structlog processors based on format
5. Configure structlog with processors and output stream
6. Set class-level state (_configured, _log_level, _config)
7. Initialize _sensitive_keys from config
```

**Edge Cases:**

1. **Called twice:**
   - **Behavior:** Raise RuntimeError("Logging already configured")
   - **Rationale:** Prevent accidental reconfiguration

2. **Invalid level:**
   - **Input:** `level="INVALID"`
   - **Behavior:** Raise ValueError("Invalid log level: INVALID. Must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL")

3. **Both config and parameters provided:**
   - **Input:** `configure_logging(level="DEBUG", config=LoggingConfig(level="INFO"))`
   - **Behavior:** config takes precedence, parameters ignored
   - **Note:** Log warning about parameter override

4. **Format not "json" or "console":**
   - **Input:** `format="xml"`
   - **Behavior:** Raise ValueError("Invalid format: xml. Must be 'json' or 'console'")

**Related Methods:**
- Calls: `_setup_processors()` (internal)
- Called by: Application startup code (main.py, server.py)

---

### Method 2: `get_logger`

**Signature:**
```python
@classmethod
def get_logger(cls, name: str) -> structlog.BoundLogger
```

**Purpose:**
Get or create a module/component-specific logger with automatic context enrichment. Returns cached logger if already exists.

**Parameters:**

- `name`: str
  - **Description:** Logger name, typically Python module path
  - **Constraints:**
    - Cannot be empty
    - Typically follows dotted notation: "zapomni.mcp.tools"
    - Maximum length: 200 characters
  - **Example:** "zapomni.core.chunker"

**Returns:**
- **Type:** `structlog.BoundLogger`
- **Fields:**
  - Bound logger with `name` context pre-set
  - All log calls will include `logger=name` in output

**Raises:**
- `RuntimeError`: If logging not configured yet (call `configure_logging()` first)
- `ValueError`: If name is empty or invalid

**Preconditions:**
- `configure_logging()` must have been called
- `cls._configured == True`

**Postconditions:**
- Logger is cached in `cls._loggers[name]`
- Returned logger is ready for use
- Multiple calls with same name return same logger instance

**Algorithm Outline:**
```
1. Check if logging configured → raise RuntimeError if not
2. Validate name parameter → raise ValueError if empty
3. Check if logger already cached in cls._loggers
4. If cached → return cached logger
5. If not cached:
   a. Create new BoundLogger with name context
   b. Cache it in cls._loggers[name]
   c. Return new logger
```

**Edge Cases:**

1. **Empty name:**
   - **Input:** `name=""`
   - **Behavior:** Raise ValueError("Logger name cannot be empty")

2. **Very long name:**
   - **Input:** `name="x" * 300`
   - **Behavior:** Raise ValueError("Logger name exceeds maximum length (200)")

3. **Logging not configured:**
   - **Behavior:** Raise RuntimeError("Logging not configured. Call configure_logging() first.")

4. **Name with special characters:**
   - **Input:** `name="module/name"`
   - **Behavior:** Accept (no restriction), but log warning about non-standard naming

**Related Methods:**
- Called by: All modules/components needing logging
- Uses: structlog.get_logger() internally

---

### Method 3: `log_operation`

**Signature:**
```python
@classmethod
def log_operation(
    cls,
    operation: str,
    correlation_id: str,
    metadata: Optional[dict[str, Any]] = None,
    logger_name: str = "zapomni",
    level: str = "info"
) -> None
```

**Purpose:**
Convenience method for logging operations with standardized structure. Automatically enriches with correlation ID, operation name, and sanitized metadata.

**Parameters:**

- `operation`: str
  - **Description:** Operation name (verb_noun convention)
  - **Constraints:**
    - Cannot be empty
    - Recommended format: "verb_noun" (e.g., "add_memory", "search_chunks")
  - **Example:** "embed_texts"

- `correlation_id`: str
  - **Description:** UUID for tracing this operation across layers
  - **Constraints:**
    - Should be valid UUID string
    - Cannot be empty
  - **Example:** "550e8400-e29b-41d4-a716-446655440000"

- `metadata`: Optional[dict[str, Any]]
  - **Description:** Additional context (parameters, metrics, results)
  - **Constraints:**
    - Must be JSON-serializable
    - Sensitive keys will be sanitized
  - **Default:** None
  - **Example:**
    ```python
    {
        "chunk_size": 512,
        "chunks_created": 10,
        "duration_ms": 45
    }
    ```

- `logger_name`: str
  - **Description:** Which logger to use
  - **Default:** "zapomni"
  - **Example:** "zapomni.mcp.tools"

- `level`: str
  - **Description:** Log level for this message
  - **Constraints:** Must be "debug", "info", "warning", "error", "critical"
  - **Default:** "info"
  - **Example:** "info"

**Returns:**
- **Type:** None
- **Side effects:** Writes log entry to stderr

**Raises:**
- `ValueError`: If operation or correlation_id is empty
- `RuntimeError`: If logging not configured
- `TypeError`: If metadata is not JSON-serializable

**Preconditions:**
- Logging must be configured
- correlation_id should be generated by caller

**Postconditions:**
- Log entry written to stderr
- Metadata sanitized (sensitive keys redacted)

**Algorithm Outline:**
```
1. Validate operation and correlation_id → raise ValueError if empty
2. Get logger for logger_name
3. Sanitize metadata (if provided)
4. Build log context:
   - operation=operation
   - correlation_id=correlation_id
   - metadata (sanitized)
5. Call logger.{level}(operation, **context)
```

**Edge Cases:**

1. **Empty operation:**
   - **Input:** `operation=""`
   - **Behavior:** Raise ValueError("operation cannot be empty")

2. **Empty correlation_id:**
   - **Input:** `correlation_id=""`
   - **Behavior:** Raise ValueError("correlation_id cannot be empty")

3. **Metadata with sensitive keys:**
   - **Input:** `metadata={"user": "alice", "password": "secret"}`
   - **Behavior:** Sanitize to `{"user": "alice", "password": "[REDACTED]"}`

4. **Non-serializable metadata:**
   - **Input:** `metadata={"func": lambda x: x}`
   - **Behavior:** Raise TypeError("metadata is not JSON-serializable")

5. **Invalid log level:**
   - **Input:** `level="INVALID"`
   - **Behavior:** Raise ValueError("Invalid log level: INVALID")

**Related Methods:**
- Calls: `get_logger()`, `_sanitize_metadata()`
- Called by: All modules performing operations

---

### Method 4: `log_error`

**Signature:**
```python
@classmethod
def log_error(
    cls,
    error: Exception,
    correlation_id: str,
    context: Optional[dict[str, Any]] = None,
    logger_name: str = "zapomni",
    include_stack_trace: bool = True
) -> None
```

**Purpose:**
Log an error with full context, error details, and optional stack trace. Automatically extracts error type, message, and error code (for ZapomniError subclasses).

**Parameters:**

- `error`: Exception
  - **Description:** The exception instance to log
  - **Constraints:**
    - Must be Exception instance
  - **Example:** `DatabaseError("Connection failed", error_code="CONN_001")`

- `correlation_id`: str
  - **Description:** UUID for tracing
  - **Constraints:** Cannot be empty
  - **Example:** "uuid-here"

- `context`: Optional[dict[str, Any]]
  - **Description:** Additional context about where/why error occurred
  - **Constraints:** Must be JSON-serializable
  - **Default:** None
  - **Example:**
    ```python
    {
        "operation": "insert_memory",
        "memory_id": "mem_123",
        "retry_attempt": 2
    }
    ```

- `logger_name`: str
  - **Description:** Which logger to use
  - **Default:** "zapomni"

- `include_stack_trace`: bool
  - **Description:** Whether to include full stack trace
  - **Default:** True
  - **Note:** Set False for expected errors (e.g., validation)

**Returns:**
- **Type:** None
- **Side effects:** Writes error log to stderr

**Raises:**
- `ValueError`: If correlation_id is empty
- `RuntimeError`: If logging not configured

**Preconditions:**
- Logging must be configured

**Postconditions:**
- Error logged with full details
- Stack trace included (if requested)

**Algorithm Outline:**
```
1. Validate correlation_id
2. Get logger for logger_name
3. Extract error details:
   - error_type = type(error).__name__
   - error_message = str(error)
   - error_code = getattr(error, 'error_code', None)
4. Build log context:
   - error_type
   - error_message
   - error_code (if available)
   - correlation_id
   - context (sanitized)
5. If include_stack_trace:
   - Call logger.exception() (includes stack trace)
6. Else:
   - Call logger.error() (no stack trace)
```

**Edge Cases:**

1. **ZapomniError with error_code:**
   - **Behavior:** Extract and include error_code in log

2. **Generic Exception:**
   - **Behavior:** Log error_type and message, no error_code

3. **Error with original_exception:**
   - **Behavior:** Log original exception details too (nested)

4. **Context with sensitive data:**
   - **Behavior:** Sanitize before logging

**Related Methods:**
- Calls: `get_logger()`, `_sanitize_metadata()`
- Called by: All error handling blocks

---

### Method 5: `log_performance`

**Signature:**
```python
@classmethod
def log_performance(
    cls,
    operation: str,
    duration_ms: float,
    correlation_id: str,
    metadata: Optional[dict[str, Any]] = None,
    logger_name: str = "zapomni"
) -> None
```

**Purpose:**
Log performance metrics for an operation. Used for identifying slow operations and optimization opportunities.

**Parameters:**

- `operation`: str
  - **Description:** Operation name
  - **Constraints:** Cannot be empty
  - **Example:** "embed_texts"

- `duration_ms`: float
  - **Description:** Operation duration in milliseconds
  - **Constraints:** Must be >= 0
  - **Example:** 123.45

- `correlation_id`: str
  - **Description:** UUID for tracing
  - **Constraints:** Cannot be empty

- `metadata`: Optional[dict[str, Any]]
  - **Description:** Additional metrics (memory, CPU, throughput)
  - **Default:** None
  - **Example:**
    ```python
    {
        "text_count": 100,
        "embedding_dimensions": 768,
        "model": "nomic-embed-text"
    }
    ```

- `logger_name`: str
  - **Description:** Which logger to use
  - **Default:** "zapomni"

**Returns:**
- **Type:** None
- **Side effects:** Writes performance log to stderr (info level)

**Raises:**
- `ValueError`: If operation/correlation_id empty or duration_ms < 0
- `RuntimeError`: If logging not configured

**Preconditions:**
- Logging configured
- duration_ms measured by caller

**Postconditions:**
- Performance metric logged

**Algorithm Outline:**
```
1. Validate parameters
2. Get logger
3. Sanitize metadata
4. Build context:
   - operation
   - duration_ms
   - correlation_id
   - metadata
5. Call logger.info("performance_metric", **context)
```

**Edge Cases:**

1. **Negative duration:**
   - **Input:** `duration_ms=-10`
   - **Behavior:** Raise ValueError("duration_ms cannot be negative")

2. **Very large duration:**
   - **Input:** `duration_ms=1000000` (16+ minutes)
   - **Behavior:** Accept, but log warning about unusually long operation

**Related Methods:**
- Calls: `get_logger()`, `_sanitize_metadata()`
- Called by: Performance-critical operations

## Error Handling

### Exceptions Defined

```python
# No custom exceptions defined
# LoggingService uses standard exceptions:
# - ValueError: Invalid parameters
# - RuntimeError: Logging not configured
# - TypeError: Non-serializable data
```

### Error Recovery

**Retry Strategy:** NO
- Logging failures should not retry (avoid infinite loops)
- If logging fails, fail fast and log to fallback (stderr print)

**Fallback Behavior:**
```python
try:
    logger.info("operation", data=data)
except Exception as e:
    # Fallback: print to stderr directly
    print(f"LOGGING FAILED: {e}", file=sys.stderr)
```

**Error Propagation:**
- Configuration errors (ValueError, RuntimeError) propagate to caller
- Logging errors during operations should not crash application

## Usage Examples

### Basic Usage

```python
# 1. Configure logging at startup
from zapomni_core.logging_service import LoggingService

LoggingService.configure_logging(level="INFO", format="json")

# 2. Get logger for a module
logger = LoggingService.get_logger("zapomni.mcp.tools")

# 3. Log operations
logger.info(
    "tool_executed",
    tool_name="add_memory",
    correlation_id="550e8400-e29b-41d4-a716-446655440000",
    duration_ms=123
)

# Output to stderr (JSON):
# {
#   "event": "tool_executed",
#   "timestamp": "2025-11-23T10:30:45.123Z",
#   "level": "info",
#   "logger": "zapomni.mcp.tools",
#   "tool_name": "add_memory",
#   "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
#   "duration_ms": 123
# }
```

### Advanced Usage

```python
import uuid
import time
from zapomni_core.logging_service import LoggingService
from zapomni_core.exceptions import DatabaseError

# Configure with advanced options
config = LoggingConfig(
    level="DEBUG",
    format="json",
    sensitive_keys={"password", "api_key", "secret"}
)
LoggingService.configure_logging(config=config)

# Get logger
logger = LoggingService.get_logger(__name__)

# Log operation with context
correlation_id = str(uuid.uuid4())

LoggingService.log_operation(
    operation="process_memory",
    correlation_id=correlation_id,
    metadata={
        "memory_id": "mem_123",
        "chunk_count": 10,
        "content_length": 5000
    }
)

# Log performance
start = time.perf_counter()
result = expensive_operation()
duration_ms = (time.perf_counter() - start) * 1000

LoggingService.log_performance(
    operation="expensive_operation",
    duration_ms=duration_ms,
    correlation_id=correlation_id,
    metadata={
        "result_size": len(result),
        "cache_hit": False
    }
)

# Log errors
try:
    db.query("MATCH (n) RETURN n")
except DatabaseError as e:
    LoggingService.log_error(
        error=e,
        correlation_id=correlation_id,
        context={
            "operation": "query_database",
            "query_type": "MATCH",
            "retry_attempt": 1
        }
    )
    raise
```

### Sensitive Data Sanitization

```python
# Automatic sanitization of sensitive keys
logger = LoggingService.get_logger(__name__)

user_data = {
    "username": "alice",
    "email": "alice@example.com",
    "password": "super_secret_password",
    "api_key": "sk-abc123def456"
}

logger.info(
    "user_created",
    correlation_id="uuid",
    user_data=user_data
)

# Logged output (sanitized):
# {
#   "event": "user_created",
#   "user_data": {
#     "username": "alice",
#     "email": "alice@example.com",
#     "password": "[REDACTED]",
#     "api_key": "[REDACTED]"
#   }
# }
```

## Testing Approach

### Unit Tests Required

1. **test_configure_logging_success()**
   - Test normal configuration
   - Verify structlog is configured
   - Verify class state updated

2. **test_configure_logging_invalid_level()**
   - Test with invalid log level
   - Expect ValueError

3. **test_configure_logging_already_configured()**
   - Test calling twice
   - Expect RuntimeError

4. **test_get_logger_success()**
   - Test getting logger
   - Verify logger cached
   - Verify repeated calls return same instance

5. **test_get_logger_not_configured()**
   - Test without configuration
   - Expect RuntimeError

6. **test_log_operation_success()**
   - Test logging operation
   - Verify output structure

7. **test_log_error_with_stack_trace()**
   - Test error logging
   - Verify stack trace included

8. **test_log_performance_metrics()**
   - Test performance logging
   - Verify duration recorded

9. **test_sanitize_metadata_sensitive_keys()**
   - Test sanitization
   - Verify sensitive keys redacted

10. **test_sanitize_metadata_nested_dicts()**
    - Test nested sanitization
    - Verify recursive processing

### Mocking Strategy

- Mock `structlog.configure()` to verify call
- Mock `sys.stderr` to capture output
- Mock processors to test pipeline

### Integration Tests

1. **test_end_to_end_logging_flow()**
   - Configure → Get Logger → Log → Verify output
   - Check JSON structure in stderr

2. **test_concurrent_logging()**
   - Multiple async tasks logging simultaneously
   - Verify no interference

## Performance Considerations

### Time Complexity

- `configure_logging()`: O(1) - one-time setup
- `get_logger()`: O(1) - dict lookup
- `log_operation()`: O(n) - where n = size of metadata
- `_sanitize_metadata()`: O(n) - recursive dict traversal

### Space Complexity

- Logger cache: O(m) - where m = number of unique loggers
- Each log entry: O(n) - where n = metadata size
- No unbounded memory growth

### Optimization Opportunities

- Logger caching prevents repeated structlog.get_logger() calls
- Lazy evaluation of log messages (only if level enabled)
- Potential: Async logging to avoid blocking I/O

## References

- **Module spec:** [error_handling_strategy.md](../level1/error_handling_strategy.md)
- **Related components:**
  - ExceptionHierarchy (uses LoggingService)
  - RetryStrategy (uses LoggingService)
  - CircuitBreaker (uses LoggingService)
- **External docs:**
  - structlog: https://www.structlog.org/
  - python-json-logger: https://github.com/madzak/python-json-logger

---

**Status:** Draft v1.0
**Created:** 2025-11-23
**Ready for Implementation:** Yes
