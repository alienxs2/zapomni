# InputValidator - Component Specification

**Level:** 2 (Component)
**Module:** zapomni_core
**Author:** Goncharenko Anton aka alienxs2
**Status:** Draft
**Version:** 1.0
**Created:** 2025-11-23

---

## Overview

### Purpose

InputValidator is the **centralized validation gateway** for all user-provided data entering the Zapomni system. It acts as the first line of defense against invalid, malformed, or malicious inputs, ensuring data integrity and security before any processing occurs.

This component validates text content, metadata dictionaries, search queries, and pagination parameters. It enforces strict constraints (encoding, size limits, structure) and provides clear, actionable error messages to help users fix invalid inputs.

### Responsibilities

1. **Text Validation**
   - Verify non-empty text after stripping whitespace
   - Enforce UTF-8 encoding
   - Check maximum size limits (10MB for memory text)
   - Detect and reject binary data or null bytes

2. **Metadata Validation**
   - Ensure metadata is a valid dictionary
   - Verify all values are JSON-serializable
   - Enforce maximum metadata size (1MB)
   - Prevent injection of reserved keys (`memory_id`, `timestamp`, `chunks`)

3. **Query Validation**
   - Validate search queries are non-empty
   - Enforce maximum query length (1000 characters)
   - Strip excessive whitespace

4. **Parameter Validation**
   - Validate pagination limits (1-100)
   - Validate numeric ranges for configurable parameters
   - Type checking for all inputs

5. **Input Sanitization**
   - Normalize Unicode (NFC normalization)
   - Strip leading/trailing whitespace
   - Remove null bytes and control characters
   - Prevent XSS attacks in text fields

### Position in Module

InputValidator is used by **all entry points** in zapomni_core:

```
┌──────────────────────────────────────────┐
│        zapomni_core Entry Points         │
├──────────────────────────────────────────┤
│  MemoryProcessor.add_memory()            │
│  MemoryProcessor.search_memory()         │
│  MemoryProcessor.build_knowledge_graph() │
└────────────┬─────────────────────────────┘
             │ validates all inputs
             ↓
┌──────────────────────────────────────────┐
│          InputValidator (this)           │
│  - validate_text()                       │
│  - validate_metadata()                   │
│  - validate_query()                      │
│  - validate_limit()                      │
│  - sanitize_input()                      │
└────────────┬─────────────────────────────┘
             │ raises ValidationError if invalid
             ↓
         Processing continues
```

**Key Relationships:**
- **MemoryProcessor → InputValidator:** Every public method validates inputs first
- **InputValidator → Exceptions:** Raises ValidationError with clear messages
- **InputValidator → pydantic:** Uses Pydantic models for structured validation

---

## Class Definition

### Class Diagram

```
┌─────────────────────────────────────┐
│         InputValidator              │
├─────────────────────────────────────┤
│ - MAX_TEXT_SIZE: int = 10_000_000   │
│ - MAX_METADATA_SIZE: int = 1_000_000│
│ - MAX_QUERY_LENGTH: int = 1000      │
│ - MAX_LIMIT: int = 100              │
│ - RESERVED_KEYS: Set[str]           │
├─────────────────────────────────────┤
│ + validate_text(text, max_size)     │
│ + validate_metadata(metadata)       │
│ + validate_query(query)             │
│ + validate_limit(limit, max_value)  │
│ + sanitize_input(text)              │
│ - _check_encoding(text)             │
│ - _is_json_serializable(obj)        │
│ - _remove_null_bytes(text)          │
└─────────────────────────────────────┘
```

### Full Class Signature

```python
from typing import Any, Dict, Optional, Set
from pydantic import BaseModel, Field, validator
import unicodedata

class InputValidator:
    """
    Centralized input validation for all Zapomni user inputs.

    Provides validation for text content, metadata, search queries,
    and pagination parameters. Enforces encoding, size, and structure
    constraints. Prevents injection attacks and malformed data.

    Class Attributes:
        MAX_TEXT_SIZE: Maximum text size in bytes (10MB)
        MAX_METADATA_SIZE: Maximum metadata JSON size (1MB)
        MAX_QUERY_LENGTH: Maximum query string length (1000 chars)
        MAX_LIMIT: Maximum pagination limit (100)
        RESERVED_KEYS: Metadata keys reserved by system

    Example:
        ```python
        validator = InputValidator()

        # Validate text input
        try:
            validator.validate_text("Python is great", max_size=10_000_000)
        except ValidationError as e:
            print(f"Invalid: {e}")

        # Validate metadata
        metadata = {"source": "user", "tags": ["python", "tutorial"]}
        validator.validate_metadata(metadata)

        # Sanitize user input
        clean_text = validator.sanitize_input("  Hello\x00World  \n")
        # Returns: "Hello World"
        ```
    """

    # Class constants
    MAX_TEXT_SIZE: int = 10_000_000  # 10 MB
    MAX_METADATA_SIZE: int = 1_000_000  # 1 MB (serialized JSON)
    MAX_QUERY_LENGTH: int = 1000
    MAX_LIMIT: int = 100
    RESERVED_KEYS: Set[str] = {"memory_id", "timestamp", "chunks", "embeddings"}

    def __init__(self) -> None:
        """
        Initialize InputValidator.

        No configuration needed - uses class constants.
        """
        pass

    def validate_text(
        self,
        text: str,
        max_size: Optional[int] = None
    ) -> str:
        """
        Validate text input for memory storage.

        Checks that text is:
        - Non-empty (after stripping whitespace)
        - Valid UTF-8 encoding
        - Within size limit
        - Free of null bytes and control characters

        Args:
            text: Text content to validate
            max_size: Optional custom max size (bytes).
                     Defaults to MAX_TEXT_SIZE (10MB)

        Returns:
            Sanitized text (whitespace normalized, null bytes removed)

        Raises:
            ValidationError: If text is invalid with specific reason:
                - "Text cannot be empty"
                - "Text must be a string, got {type}"
                - "Text exceeds maximum size (10,000,000 bytes)"
                - "Text contains invalid UTF-8 encoding"
                - "Text contains null bytes or control characters"

        Example:
            ```python
            validator = InputValidator()

            # Valid
            text = validator.validate_text("Python is great")

            # Invalid - empty
            validator.validate_text("")  # raises ValidationError

            # Invalid - too large
            huge = "x" * 20_000_000
            validator.validate_text(huge)  # raises ValidationError
            ```
        """

    def validate_metadata(
        self,
        metadata: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Validate metadata dictionary.

        Checks that metadata:
        - Is a dictionary (if provided)
        - Contains only JSON-serializable values
        - Does not use reserved keys
        - Serializes to < 1MB JSON

        Args:
            metadata: Optional metadata dictionary

        Returns:
            Validated metadata (unchanged if valid, None if None)

        Raises:
            ValidationError: If metadata is invalid:
                - "Metadata must be a dictionary, got {type}"
                - "Metadata contains reserved key: {key}"
                - "Metadata value for '{key}' is not JSON-serializable"
                - "Metadata exceeds maximum size (1,000,000 bytes)"

        Example:
            ```python
            validator = InputValidator()

            # Valid
            meta = {"source": "user", "tags": ["ai", "ml"]}
            validator.validate_metadata(meta)

            # Valid - None
            validator.validate_metadata(None)

            # Invalid - reserved key
            meta = {"memory_id": "123"}
            validator.validate_metadata(meta)  # raises ValidationError

            # Invalid - not serializable
            meta = {"func": lambda x: x}
            validator.validate_metadata(meta)  # raises ValidationError
            ```
        """

    def validate_query(
        self,
        query: str
    ) -> str:
        """
        Validate search query.

        Checks that query:
        - Is a string
        - Is non-empty after stripping
        - Does not exceed max length (1000 chars)

        Args:
            query: Search query string

        Returns:
            Sanitized query (whitespace normalized)

        Raises:
            ValidationError: If query is invalid:
                - "Query cannot be empty"
                - "Query must be a string, got {type}"
                - "Query exceeds maximum length (1000 characters)"

        Example:
            ```python
            validator = InputValidator()

            # Valid
            query = validator.validate_query("Python programming")

            # Invalid - empty
            validator.validate_query("")  # raises ValidationError

            # Invalid - too long
            long_query = "x" * 1001
            validator.validate_query(long_query)  # raises ValidationError
            ```
        """

    def validate_limit(
        self,
        limit: int,
        max_value: Optional[int] = None
    ) -> int:
        """
        Validate pagination limit.

        Checks that limit:
        - Is an integer
        - Is positive (>= 1)
        - Does not exceed max_value

        Args:
            limit: Number of results to return
            max_value: Optional max limit (defaults to MAX_LIMIT = 100)

        Returns:
            Validated limit (unchanged)

        Raises:
            ValidationError: If limit is invalid:
                - "Limit must be an integer, got {type}"
                - "Limit must be positive (>= 1)"
                - "Limit exceeds maximum (100)"

        Example:
            ```python
            validator = InputValidator()

            # Valid
            validator.validate_limit(10)  # returns 10

            # Invalid - zero
            validator.validate_limit(0)  # raises ValidationError

            # Invalid - too large
            validator.validate_limit(200)  # raises ValidationError

            # Custom max
            validator.validate_limit(50, max_value=50)  # OK
            ```
        """

    def sanitize_input(
        self,
        text: str
    ) -> str:
        """
        Sanitize text input (remove unsafe content).

        Performs:
        - Strip leading/trailing whitespace
        - Remove null bytes (\\x00)
        - Remove control characters (except \\n, \\t)
        - Normalize Unicode (NFC normalization)
        - Collapse multiple spaces to single space

        Args:
            text: Text to sanitize

        Returns:
            Sanitized text (safe for storage and processing)

        Example:
            ```python
            validator = InputValidator()

            # Remove null bytes
            clean = validator.sanitize_input("Hello\\x00World")
            # Returns: "HelloWorld"

            # Normalize whitespace
            clean = validator.sanitize_input("  Python   is    great  ")
            # Returns: "Python is great"

            # Normalize Unicode
            clean = validator.sanitize_input("café")  # é as e + combining accent
            # Returns: "café"  # é as single character
            ```
        """

    # Private helper methods

    def _check_encoding(
        self,
        text: str
    ) -> None:
        """
        Verify text is valid UTF-8.

        Args:
            text: Text to check

        Raises:
            ValidationError: If text contains invalid UTF-8 sequences
        """

    def _is_json_serializable(
        self,
        obj: Any
    ) -> bool:
        """
        Check if object can be JSON-serialized.

        Args:
            obj: Object to check

        Returns:
            True if serializable, False otherwise

        Implementation:
            Attempts json.dumps(), catches TypeError
        """

    def _remove_null_bytes(
        self,
        text: str
    ) -> str:
        """
        Remove null bytes and control characters from text.

        Args:
            text: Text to clean

        Returns:
            Text with null bytes removed

        Note:
            Preserves \\n (newline) and \\t (tab)
        """
```

---

## Dependencies

### Component Dependencies

**Internal (zapomni_core):**
- `exceptions.ValidationError` - Raised for all validation failures
- None (InputValidator has no dependencies on other core components)

### External Libraries

**Required:**
- `pydantic>=2.5.0` - For structured validation models (optional, for future Pydantic-based models)
- `typing` (stdlib) - Type hints

**Optional:**
- `structlog>=23.2.0` - Structured logging (for debug logs)

### Dependency Injection

InputValidator is stateless and requires no dependencies. All configuration is via class constants.

**Usage Pattern:**
```python
# Direct instantiation
validator = InputValidator()

# Or use as class methods (if made static in future)
InputValidator.validate_text("text")
```

---

## State Management

### Attributes

**Class Attributes (constants):**
- `MAX_TEXT_SIZE`: int = 10,000,000 - Maximum text size in bytes
- `MAX_METADATA_SIZE`: int = 1,000,000 - Maximum metadata JSON size
- `MAX_QUERY_LENGTH`: int = 1000 - Maximum query string length
- `MAX_LIMIT`: int = 100 - Maximum pagination limit
- `RESERVED_KEYS`: Set[str] - Set of reserved metadata keys

**Instance Attributes:**
- None (stateless class)

### State Transitions

InputValidator is **completely stateless**. Each method call is independent.

```
validate_text(text)
    ↓
[No state change]
    ↓
Return sanitized text OR raise ValidationError
```

### Thread Safety

**Thread-Safe:** Yes ✅

InputValidator has no mutable state, making it inherently thread-safe. Multiple threads can use the same instance concurrently without synchronization.

**Usage in Concurrent Code:**
```python
# Safe to share instance across async tasks
validator = InputValidator()

async def process_many(texts):
    tasks = [validate_and_process(text, validator) for text in texts]
    await asyncio.gather(*tasks)
```

---

## Public Methods (Detailed)

### Method 1: `validate_text`

**Signature:**
```python
def validate_text(self, text: str, max_size: Optional[int] = None) -> str
```

**Purpose:**
Validate text input for memory storage. Ensures text is non-empty, valid UTF-8, within size limits, and free of malicious content.

**Parameters:**

**`text`: str**
- Description: Text content to validate
- Constraints:
  - Must be a string (not bytes, int, etc.)
  - Must be non-empty after stripping whitespace
  - Must be valid UTF-8 encoding
  - Must not contain null bytes (`\x00`)
  - Must not exceed max_size bytes
- Example: `"Python is a programming language"`

**`max_size`: Optional[int]**
- Description: Custom maximum size in bytes
- Default: `InputValidator.MAX_TEXT_SIZE` (10,000,000)
- Constraints: Must be positive if provided
- Example: `1000` (limit to 1KB)

**Returns:**
- Type: `str`
- Content: Sanitized text (whitespace normalized, null bytes removed)
- Guarantees:
  - Non-empty
  - Valid UTF-8
  - Within size limit
  - Safe for storage

**Raises:**

**`ValidationError`**
- When: text is invalid
- Messages:
  - `"Text cannot be empty"` - Empty string after strip
  - `"Text must be a string, got {type}"` - Wrong type
  - `"Text exceeds maximum size (10,000,000 bytes)"` - Too large
  - `"Text contains invalid UTF-8 encoding"` - Encoding error
  - `"Text contains null bytes"` - Null byte detected

**Preconditions:**
- None (validates all inputs)

**Postconditions:**
- If no exception: text is valid and sanitized
- If exception: no side effects (stateless)

**Algorithm Outline:**
```
1. Type check (must be str)
2. Sanitize input (strip, remove null bytes)
3. Check non-empty
4. Check encoding (valid UTF-8)
5. Check size (encode to bytes, compare to max_size)
6. Return sanitized text
```

**Edge Cases:**

1. **Empty string:** `""` → `ValidationError("Text cannot be empty")`
2. **Whitespace-only:** `"   "` → `ValidationError("Text cannot be empty")`
3. **Null bytes:** `"Hello\x00World"` → Sanitized to `"HelloWorld"`
4. **Exactly max size:** `"x" * 10_000_000` → Valid (boundary case)
5. **One byte over:** `"x" * 10_000_001` → `ValidationError("exceeds maximum")`
6. **Non-string type:** `123` → `ValidationError("must be a string")`
7. **Invalid UTF-8:** `b"\xff\xfe".decode("utf-8", errors="replace")` → `ValidationError("invalid UTF-8")`

**Related Methods:**
- Calls: `sanitize_input()`, `_check_encoding()`, `_remove_null_bytes()`
- Called by: `MemoryProcessor.add_memory()`

---

### Method 2: `validate_metadata`

**Signature:**
```python
def validate_metadata(self, metadata: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]
```

**Purpose:**
Validate metadata dictionary. Ensures metadata is structured correctly, contains only JSON-serializable values, and doesn't use reserved keys.

**Parameters:**

**`metadata`: Optional[Dict[str, Any]]**
- Description: User-provided metadata
- Constraints:
  - Must be dict if provided (or None)
  - All values must be JSON-serializable (str, int, float, bool, list, dict, None)
  - Keys must not be in RESERVED_KEYS
  - Serialized JSON must be < 1MB
- Example: `{"source": "user", "tags": ["python", "ai"]}`

**Returns:**
- Type: `Optional[Dict[str, Any]]`
- Content: Validated metadata (unchanged if valid, None if None)

**Raises:**

**`ValidationError`**
- When: metadata is invalid
- Messages:
  - `"Metadata must be a dictionary, got {type}"` - Wrong type
  - `"Metadata contains reserved key: {key}"` - Reserved key used
  - `"Metadata value for '{key}' is not JSON-serializable"` - Non-serializable value
  - `"Metadata exceeds maximum size (1,000,000 bytes)"` - Too large

**Algorithm Outline:**
```
1. If metadata is None, return None
2. Type check (must be dict)
3. For each key:
   a. Check not in RESERVED_KEYS
   b. Check value is JSON-serializable (_is_json_serializable())
4. Serialize to JSON, check size < MAX_METADATA_SIZE
5. Return metadata unchanged
```

**Edge Cases:**

1. **None metadata:** `None` → Returns `None` (valid)
2. **Empty dict:** `{}` → Valid
3. **Reserved key:** `{"memory_id": "123"}` → `ValidationError("reserved key")`
4. **Non-serializable:** `{"func": lambda x: x}` → `ValidationError("not JSON-serializable")`
5. **Nested structure:** `{"user": {"name": "Alice", "tags": ["admin"]}}` → Valid
6. **Circular reference:** `d = {}; d["self"] = d` → `ValidationError("not JSON-serializable")`
7. **Exactly 1MB:** Metadata that serializes to 1,000,000 bytes → Valid
8. **Over 1MB:** → `ValidationError("exceeds maximum")`

**Related Methods:**
- Calls: `_is_json_serializable()`
- Called by: `MemoryProcessor.add_memory()`, `MemoryProcessor.search_memory()`

---

### Method 3: `validate_query`

**Signature:**
```python
def validate_query(self, query: str) -> str
```

**Purpose:**
Validate search query string. Ensures query is non-empty and within length limits.

**Parameters:**

**`query`: str**
- Description: Search query from user
- Constraints:
  - Must be a string
  - Must be non-empty after stripping
  - Must not exceed MAX_QUERY_LENGTH (1000 chars)
- Example: `"Python programming tutorials"`

**Returns:**
- Type: `str`
- Content: Sanitized query (whitespace normalized)

**Raises:**

**`ValidationError`**
- When: query is invalid
- Messages:
  - `"Query cannot be empty"` - Empty after strip
  - `"Query must be a string, got {type}"` - Wrong type
  - `"Query exceeds maximum length (1000 characters)"` - Too long

**Algorithm Outline:**
```
1. Type check (must be str)
2. Sanitize (strip whitespace)
3. Check non-empty
4. Check length <= MAX_QUERY_LENGTH
5. Return sanitized query
```

**Edge Cases:**

1. **Empty string:** `""` → `ValidationError("cannot be empty")`
2. **Whitespace-only:** `"   \n  "` → `ValidationError("cannot be empty")`
3. **Exactly 1000 chars:** `"x" * 1000` → Valid
4. **1001 chars:** `"x" * 1001` → `ValidationError("exceeds maximum")`
5. **Special characters:** `"What is $variable?"` → Valid (no sanitization of content)
6. **Unicode query:** `"Python программирование"` → Valid

**Related Methods:**
- Calls: `sanitize_input()`
- Called by: `MemoryProcessor.search_memory()`

---

### Method 4: `validate_limit`

**Signature:**
```python
def validate_limit(self, limit: int, max_value: Optional[int] = None) -> int
```

**Purpose:**
Validate pagination limit parameter. Ensures limit is a positive integer within bounds.

**Parameters:**

**`limit`: int**
- Description: Number of results to return
- Constraints:
  - Must be an integer
  - Must be >= 1
  - Must be <= max_value
- Example: `10`

**`max_value`: Optional[int]**
- Description: Custom maximum limit
- Default: `InputValidator.MAX_LIMIT` (100)
- Example: `50`

**Returns:**
- Type: `int`
- Content: Validated limit (unchanged)

**Raises:**

**`ValidationError`**
- When: limit is invalid
- Messages:
  - `"Limit must be an integer, got {type}"` - Wrong type
  - `"Limit must be positive (>= 1)"` - Zero or negative
  - `"Limit exceeds maximum (100)"` - Too large

**Algorithm Outline:**
```
1. Type check (must be int)
2. Check limit >= 1
3. Check limit <= (max_value or MAX_LIMIT)
4. Return limit
```

**Edge Cases:**

1. **Zero:** `0` → `ValidationError("must be positive")`
2. **Negative:** `-5` → `ValidationError("must be positive")`
3. **Exactly 1:** `1` → Valid (minimum)
4. **Exactly 100:** `100` → Valid (default maximum)
5. **101:** `101` → `ValidationError("exceeds maximum")`
6. **Custom max:** `validate_limit(50, max_value=50)` → Valid
7. **Float type:** `10.5` → `ValidationError("must be an integer")`

**Related Methods:**
- Called by: `MemoryProcessor.search_memory()`

---

### Method 5: `sanitize_input`

**Signature:**
```python
def sanitize_input(self, text: str) -> str
```

**Purpose:**
Sanitize text by removing unsafe characters and normalizing whitespace. Used internally by other validators.

**Parameters:**

**`text`: str**
- Description: Text to sanitize
- No constraints (handles all strings)

**Returns:**
- Type: `str`
- Content: Sanitized text
- Transformations:
  - Leading/trailing whitespace removed
  - Null bytes removed
  - Control characters removed (except `\n`, `\t`)
  - Multiple spaces collapsed to single space
  - Unicode normalized (NFC)

**Raises:**
- None (best-effort sanitization)

**Algorithm Outline:**
```
1. Strip leading/trailing whitespace
2. Normalize Unicode (NFC)
3. Remove null bytes (\x00)
4. Remove control characters (except \n, \t)
5. Collapse multiple spaces to single space
6. Return sanitized text
```

**Edge Cases:**

1. **Null bytes:** `"Hello\x00World"` → `"HelloWorld"`
2. **Control chars:** `"Hello\x01\x02World"` → `"HelloWorld"`
3. **Multiple spaces:** `"Python   is    great"` → `"Python is great"`
4. **Mixed whitespace:** `"  \n  Python  \t  "` → `"\n  Python  \t"` (preserves \n, \t)
5. **Unicode combining:** `"café"` (e + ´) → `"café"` (single é)
6. **Empty after sanitization:** `"\x00\x01"` → `""` (valid, but would fail validate_text)

**Related Methods:**
- Called by: `validate_text()`, `validate_query()`

---

## Error Handling

### Exceptions Defined

```python
# zapomni_core/exceptions.py

class ValidationError(ZapomniCoreError):
    """
    Raised when input validation fails.

    Attributes:
        message: User-friendly error message
        field: Optional field name that failed validation

    Example:
        raise ValidationError("Text cannot be empty", field="text")
    """
    def __init__(self, message: str, field: Optional[str] = None):
        self.message = message
        self.field = field
        super().__init__(message)
```

### Error Recovery

**Caller Responsibility:**
- InputValidator does NOT recover from errors
- Raises ValidationError immediately
- Caller must handle exception (e.g., return error to user)

**No Retry Logic:**
- Validation errors are permanent (bad input)
- Retrying with same input will fail again
- User must fix input and resubmit

**Error Propagation:**
- All ValidationErrors bubble up to MCP layer
- MCP layer formats error for user (see zapomni_mcp spec)

---

## Usage Examples

### Basic Usage

```python
from zapomni_core.validation import InputValidator
from zapomni_core.exceptions import ValidationError

validator = InputValidator()

# Validate text
try:
    text = validator.validate_text("Python is a programming language")
    print(f"Valid text: {text}")
except ValidationError as e:
    print(f"Invalid: {e.message}")

# Validate metadata
metadata = {"source": "user", "tags": ["python", "tutorial"]}
try:
    validator.validate_metadata(metadata)
    print("Metadata is valid")
except ValidationError as e:
    print(f"Invalid metadata: {e.message}")

# Validate query
query = validator.validate_query("Python tutorials")
print(f"Query: {query}")

# Validate limit
limit = validator.validate_limit(10)
print(f"Limit: {limit}")
```

### Advanced Usage (Integration with MemoryProcessor)

```python
from zapomni_core.processing import MemoryProcessor
from zapomni_core.validation import InputValidator
from zapomni_core.exceptions import ValidationError

class MemoryProcessor:
    def __init__(self, ...):
        self.validator = InputValidator()
        ...

    async def add_memory(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add memory with validation."""
        # Validate inputs FIRST
        try:
            text = self.validator.validate_text(text)
            metadata = self.validator.validate_metadata(metadata)
        except ValidationError as e:
            # Re-raise with context
            raise ValidationError(f"Invalid input for add_memory: {e.message}")

        # Proceed with processing
        chunks = self.chunker.chunk(text)
        embeddings = await self.embedder.embed([c.text for c in chunks])
        ...
        return memory_id

    async def search_memory(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search memory with validation."""
        # Validate all parameters
        query = self.validator.validate_query(query)
        limit = self.validator.validate_limit(limit)
        filters = self.validator.validate_metadata(filters)

        # Proceed with search
        ...
```

### Error Handling Pattern

```python
def safe_add_memory(processor, text, metadata):
    """Add memory with user-friendly error handling."""
    try:
        memory_id = await processor.add_memory(text, metadata)
        return {"success": True, "memory_id": memory_id}
    except ValidationError as e:
        # User-friendly error
        return {
            "success": False,
            "error": e.message,
            "field": e.field,
            "help": "Please check your input and try again"
        }
    except Exception as e:
        # Unexpected error
        logger.error("Unexpected error", error=str(e))
        return {
            "success": False,
            "error": "An unexpected error occurred"
        }
```

---

## Testing Approach

### Unit Tests Required

**Happy Path Tests:**
1. `test_validate_text_success()` - Valid text passes
2. `test_validate_metadata_success()` - Valid metadata passes
3. `test_validate_query_success()` - Valid query passes
4. `test_validate_limit_success()` - Valid limit passes
5. `test_sanitize_input_basic()` - Basic sanitization works

**Error Tests:**
6. `test_validate_text_empty_raises()` - Empty text fails
7. `test_validate_text_too_large_raises()` - Oversized text fails
8. `test_validate_text_invalid_utf8_raises()` - Bad encoding fails
9. `test_validate_text_wrong_type_raises()` - Non-string fails
10. `test_validate_metadata_reserved_key_raises()` - Reserved key fails
11. `test_validate_metadata_not_serializable_raises()` - Lambda fails
12. `test_validate_metadata_too_large_raises()` - Oversized metadata fails
13. `test_validate_query_empty_raises()` - Empty query fails
14. `test_validate_query_too_long_raises()` - Long query fails
15. `test_validate_limit_zero_raises()` - Zero limit fails
16. `test_validate_limit_negative_raises()` - Negative limit fails
17. `test_validate_limit_too_large_raises()` - Oversized limit fails

**Edge Case Tests:**
18. `test_sanitize_null_bytes()` - Null bytes removed
19. `test_sanitize_control_chars()` - Control chars removed
20. `test_sanitize_unicode_normalization()` - Unicode normalized
21. `test_sanitize_whitespace_collapse()` - Spaces collapsed
22. `test_validate_text_boundary_size()` - Exactly max size OK
23. `test_validate_metadata_none()` - None metadata OK
24. `test_validate_metadata_empty_dict()` - Empty dict OK
25. `test_validate_limit_custom_max()` - Custom max works

### Mocking Strategy

**No Mocking Needed:**
InputValidator has no external dependencies to mock. All tests use real instances.

**Test Fixtures:**
```python
# tests/conftest.py

import pytest
from zapomni_core.validation import InputValidator

@pytest.fixture
def validator():
    """Provide InputValidator instance."""
    return InputValidator()

@pytest.fixture
def valid_text():
    """Valid text sample."""
    return "Python is a programming language"

@pytest.fixture
def valid_metadata():
    """Valid metadata sample."""
    return {
        "source": "user",
        "tags": ["python", "programming"],
        "date": "2025-11-23"
    }
```

### Integration Tests

**With MemoryProcessor:**
```python
# tests/integration/test_validation_integration.py

@pytest.mark.asyncio
async def test_add_memory_validates_input(memory_processor):
    """Test that add_memory validates input via InputValidator."""
    # Invalid text should raise ValidationError
    with pytest.raises(ValidationError, match="cannot be empty"):
        await memory_processor.add_memory(text="", metadata={})

    # Invalid metadata should raise ValidationError
    with pytest.raises(ValidationError, match="reserved key"):
        await memory_processor.add_memory(
            text="Valid text",
            metadata={"memory_id": "123"}  # Reserved key
        )
```

---

## Performance Considerations

### Time Complexity

**validate_text:**
- Type check: O(1)
- Sanitization: O(n) where n = text length
- Encoding check: O(n)
- Size check: O(n) (encode to bytes)
- **Overall: O(n)**

**validate_metadata:**
- Type check: O(1)
- Iterate keys: O(k) where k = number of keys
- JSON serialization: O(m) where m = metadata structure size
- **Overall: O(k + m)**

**validate_query:**
- Type check: O(1)
- Sanitization: O(n) where n = query length
- **Overall: O(n)**

**validate_limit:**
- Type check: O(1)
- Range checks: O(1)
- **Overall: O(1)**

**sanitize_input:**
- Strip: O(n)
- Unicode normalization: O(n)
- Null byte removal: O(n)
- **Overall: O(n)**

### Space Complexity

**All methods: O(n)** where n = input size

- `validate_text`: Creates sanitized copy (worst case: same size)
- `validate_metadata`: JSON serialization creates string (worst case: ~2x size)
- `validate_query`: Creates sanitized copy
- `validate_limit`: O(1) (returns int)

### Optimization Opportunities

**Early Exit:**
- Check type BEFORE sanitization (fail fast for wrong types)
- Check size BEFORE encoding validation (fail fast for oversized input)

**Lazy Validation:**
- Don't sanitize if type check fails
- Don't check encoding if empty

**Caching (Future):**
- Could cache validation results for identical inputs
- Trade-off: Memory vs. CPU

**Current Performance (Estimated):**
- Small input (< 1KB): < 1ms
- Medium input (10KB): < 5ms
- Large input (1MB): < 50ms
- Maximum input (10MB): < 500ms

**Target Performance:**
- Validation overhead should be < 5% of total processing time
- For 1KB text: < 1ms validation, ~100ms total → 1% overhead ✅

---

## References

### Component Spec
- Parent module: [zapomni_core_module.md](../level1/zapomni_core_module.md)

### Related Components (Future)
- MemoryProcessor (uses InputValidator)
- Chunker (receives validated text)
- Embedder (receives validated chunks)

### External Docs
- Python Unicode Normalization: https://docs.python.org/3/library/unicodedata.html#unicodedata.normalize
- JSON Serialization: https://docs.python.org/3/library/json.html
- Pydantic Validation: https://docs.pydantic.dev/latest/concepts/validators/

---

## Appendix: Validation Rules Reference

### Text Validation Rules

| Rule | Constraint | Error Message |
|------|------------|---------------|
| Type | Must be `str` | "Text must be a string, got {type}" |
| Empty | Non-empty after strip | "Text cannot be empty" |
| Encoding | Valid UTF-8 | "Text contains invalid UTF-8 encoding" |
| Size | <= 10,000,000 bytes | "Text exceeds maximum size (10,000,000 bytes)" |
| Content | No null bytes | Sanitized (removed silently) |

### Metadata Validation Rules

| Rule | Constraint | Error Message |
|------|------------|---------------|
| Type | Must be `dict` or `None` | "Metadata must be a dictionary, got {type}" |
| Keys | Not in RESERVED_KEYS | "Metadata contains reserved key: {key}" |
| Values | JSON-serializable | "Metadata value for '{key}' is not JSON-serializable" |
| Size | Serialized < 1,000,000 bytes | "Metadata exceeds maximum size (1,000,000 bytes)" |

### Query Validation Rules

| Rule | Constraint | Error Message |
|------|------------|---------------|
| Type | Must be `str` | "Query must be a string, got {type}" |
| Empty | Non-empty after strip | "Query cannot be empty" |
| Length | <= 1000 characters | "Query exceeds maximum length (1000 characters)" |

### Limit Validation Rules

| Rule | Constraint | Error Message |
|------|------------|---------------|
| Type | Must be `int` | "Limit must be an integer, got {type}" |
| Positive | >= 1 | "Limit must be positive (>= 1)" |
| Maximum | <= 100 (or custom) | "Limit exceeds maximum (100)" |

### Sanitization Rules

| Operation | Applied To | Result |
|-----------|-----------|--------|
| Strip whitespace | All text | Leading/trailing removed |
| Remove null bytes | All text | `\x00` removed |
| Remove control chars | All text | `\x01-\x1F` removed (except `\n`, `\t`) |
| Normalize Unicode | All text | NFC normalization |
| Collapse spaces | All text | Multiple spaces → single space |

---

**Document Status:** Draft v1.0
**Created:** 2025-11-23
**Author:** Goncharenko Anton aka alienxs2
**Copyright:** Copyright (c) 2025 Goncharenko Anton aka alienxs2
**License:** MIT License

**Total Methods:** 5 public + 3 private
**Total Test Scenarios:** 25+
**Ready for Review:** Yes ✅
