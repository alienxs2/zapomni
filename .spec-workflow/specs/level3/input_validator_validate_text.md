# InputValidator.validate_text - Function Specification

**Level:** 3 (Function)
**Component:** InputValidator
**Module:** zapomni_core
**Author:** Goncharenko Anton aka alienxs2
**Status:** Draft
**Version:** 1.0
**Created:** 2025-11-23

---

## Function Signature

```python
def validate_text(
    self,
    text: str,
    max_size: Optional[int] = None
) -> str:
    """
    Validate text input for memory storage.

    Ensures text is non-empty, valid UTF-8, within size limits, and free
    of null bytes and control characters. This is the first line of defense
    against invalid or malicious text inputs entering the Zapomni system.

    The function performs sanitization (removes null bytes, normalizes whitespace,
    normalizes Unicode) and validation (checks type, emptiness, encoding, size).
    Returns sanitized text that is safe for storage and processing.

    Args:
        text: Text content to validate. Must be a string (not bytes, int, etc.).
              After sanitization, must be non-empty and valid UTF-8.
        max_size: Optional custom maximum size in bytes. If not provided,
                  defaults to InputValidator.MAX_TEXT_SIZE (10,000,000 bytes).
                  Must be positive if provided.

    Returns:
        Sanitized text (whitespace normalized, null bytes removed, Unicode
        normalized). Guaranteed to be:
        - Non-empty (after strip)
        - Valid UTF-8 encoding
        - Within max_size bytes
        - Free of null bytes and control characters (except \n, \t)

    Raises:
        ValidationError: If text is invalid with specific reason:
            - "Text must be a string, got {type}" - Wrong type
            - "Text cannot be empty" - Empty string after strip
            - "Text contains invalid UTF-8 encoding" - Encoding error
            - "Text exceeds maximum size ({max_size} bytes)" - Too large

    Example:
        ```python
        validator = InputValidator()

        # Valid text
        text = validator.validate_text("Python is great")
        # Returns: "Python is great"

        # Whitespace stripped
        text = validator.validate_text("  Python  ")
        # Returns: "Python"

        # Null bytes removed
        text = validator.validate_text("Hello\x00World")
        # Returns: "HelloWorld"

        # Invalid - empty
        validator.validate_text("")
        # Raises: ValidationError("Text cannot be empty")

        # Invalid - too large
        huge = "x" * 20_000_000
        validator.validate_text(huge)
        # Raises: ValidationError("Text exceeds maximum size (10,000,000 bytes)")

        # Custom max size
        text = validator.validate_text("Short text", max_size=1000)
        # Returns: "Short text"
        ```
    """
```

---

## Purpose & Context

### What It Does

`validate_text()` is the **primary text validation function** in the InputValidator component. It performs comprehensive validation and sanitization of user-provided text before it enters the Zapomni system.

**Key Operations:**
1. **Type Validation**: Ensures input is a string (not bytes, int, list, etc.)
2. **Sanitization**: Calls `sanitize_input()` to remove unsafe characters and normalize
3. **Emptiness Check**: Verifies text is non-empty after sanitization
4. **Encoding Validation**: Calls `_check_encoding()` to ensure valid UTF-8
5. **Size Validation**: Encodes to bytes and checks against max_size limit
6. **Return**: Returns sanitized, validated text safe for storage

### Why It Exists

**Security & Data Integrity**: User-provided text can contain:
- Malicious content (null bytes, control characters)
- Invalid encoding (non-UTF-8 sequences)
- Excessively large data (DoS potential)
- Empty or whitespace-only content (wastes resources)

This function prevents ALL of these issues before text reaches processing pipeline (chunking, embedding, storage).

**Centralized Validation**: Instead of each component (MemoryProcessor, Chunker, etc.) doing its own validation, ALL text goes through this single function, ensuring consistency.

### When To Use

**Called By:**
- `MemoryProcessor.add_memory(text, ...)` - Before chunking user text
- Any entry point accepting user-provided text content

**When NOT To Use:**
- For query validation → use `validate_query()` instead (different length limits)
- For metadata validation → use `validate_metadata()` instead (different structure)
- For internal system text (already validated) → skip validation

### When NOT To Use

- **System-generated text**: Text created internally (e.g., chunk IDs, timestamps) doesn't need validation
- **Already-validated text**: If text was validated earlier in the call chain, don't re-validate
- **Binary data**: This function expects text strings, not binary data (bytes)

---

## Parameters (Detailed)

### text: str

**Type:** `str`

**Purpose:** Text content to validate for memory storage. This is the user-provided content that will be chunked, embedded, and stored in the knowledge graph.

**Constraints:**
- **Type**: Must be a string (not `bytes`, `int`, `list`, `None`, etc.)
- **Non-empty**: After stripping whitespace, must have length > 0
- **Encoding**: Must be valid UTF-8 (no invalid byte sequences)
- **Size**: Encoded as UTF-8 bytes, must be <= max_size
- **Content**: Will be sanitized (null bytes removed automatically)

**Validation Logic:**
```python
# Step 1: Type check
if not isinstance(text, str):
    raise ValidationError(f"Text must be a string, got {type(text).__name__}")

# Step 2: Sanitize
text = self.sanitize_input(text)  # Remove null bytes, normalize

# Step 3: Check non-empty
if not text:
    raise ValidationError("Text cannot be empty")

# Step 4: Check encoding
self._check_encoding(text)  # Raises ValidationError if invalid UTF-8

# Step 5: Check size
text_bytes = text.encode('utf-8')
effective_max_size = max_size or self.MAX_TEXT_SIZE
if len(text_bytes) > effective_max_size:
    raise ValidationError(f"Text exceeds maximum size ({effective_max_size} bytes)")
```

**Examples:**

**Valid:**
- `"Python is a programming language"` → Returns: `"Python is a programming language"`
- `"  Python  "` → Returns: `"Python"` (whitespace stripped)
- `"Hello\nWorld"` → Returns: `"Hello\nWorld"` (newlines preserved)
- `"café"` (Unicode) → Returns: `"café"` (NFC normalized)
- `"x" * 10_000_000` → Returns: `"xxx..."` (exactly at max size, valid)

**Invalid (raises ValidationError):**
- `""` → `"Text cannot be empty"`
- `"   "` → `"Text cannot be empty"` (whitespace-only)
- `123` → `"Text must be a string, got int"`
- `None` → `"Text must be a string, got NoneType"`
- `["text"]` → `"Text must be a string, got list"`
- `b"bytes"` → `"Text must be a string, got bytes"`
- `"x" * 10_000_001` → `"Text exceeds maximum size (10,000,000 bytes)"`

**Special Cases:**
- `"Hello\x00World"` → Returns: `"HelloWorld"` (null byte removed during sanitization)
- `"Python\t\tis\tgreat"` → Returns: `"Python\t\tis\tgreat"` (tabs preserved)
- `"Line1\nLine2\n\nLine3"` → Returns: `"Line1\nLine2\n\nLine3"` (newlines preserved)

### max_size: Optional[int]

**Type:** `Optional[int]`

**Purpose:** Custom maximum size in bytes for this specific validation. Allows different size limits for different use cases without changing class constant.

**Default:** `None` → uses `InputValidator.MAX_TEXT_SIZE` (10,000,000 bytes = 10 MB)

**Constraints:**
- Must be `int` or `None`
- If provided, must be positive (> 0)
- Recommended: >= 1000 bytes (1 KB minimum for practical use)

**Usage:**
```python
# Use default (10 MB)
validator.validate_text(text)  # max_size defaults to 10,000,000

# Custom smaller limit (e.g., for short notes)
validator.validate_text(text, max_size=1_000)  # 1 KB limit

# Custom larger limit (if needed for special cases)
validator.validate_text(text, max_size=50_000_000)  # 50 MB limit
```

**Implementation:**
```python
effective_max_size = max_size if max_size is not None else self.MAX_TEXT_SIZE
```

**Examples:**

**Valid:**
- `None` → Uses default 10,000,000
- `1000` → 1 KB limit
- `100_000` → 100 KB limit
- `50_000_000` → 50 MB limit

**Invalid (if validation added):**
- `0` → Should raise ValidationError (zero size meaningless)
- `-1000` → Should raise ValidationError (negative size invalid)
- `"1000"` → Should raise ValidationError (wrong type)

**Note:** Current spec from parent component doesn't validate `max_size` parameter itself. Future enhancement could add:
```python
if max_size is not None:
    if not isinstance(max_size, int):
        raise ValidationError(f"max_size must be an integer, got {type(max_size).__name__}")
    if max_size <= 0:
        raise ValidationError("max_size must be positive (> 0)")
```

---

## Return Value

**Type:** `str`

**Content:** Sanitized and validated text

**Guarantees:**
- ✅ Non-empty (length > 0 after strip)
- ✅ Valid UTF-8 encoding
- ✅ Within size limit (encoded bytes <= max_size)
- ✅ Free of null bytes (`\x00`)
- ✅ Control characters removed (except `\n`, `\t`)
- ✅ Unicode normalized (NFC normalization)
- ✅ Leading/trailing whitespace removed
- ✅ Multiple spaces collapsed to single space

**Transformations Applied:**

| Original | Sanitized | Reason |
|----------|-----------|--------|
| `"  Python  "` | `"Python"` | Leading/trailing whitespace stripped |
| `"Hello\x00World"` | `"HelloWorld"` | Null byte removed |
| `"Python   is    great"` | `"Python is great"` | Multiple spaces collapsed |
| `"café"` (e + ´) | `"café"` (single é) | Unicode NFC normalization |
| `"Hello\x01World"` | `"HelloWorld"` | Control character removed |
| `"Hello\nWorld"` | `"Hello\nWorld"` | Newline preserved (allowed) |
| `"Hello\tWorld"` | `"Hello\tWorld"` | Tab preserved (allowed) |

**Examples:**

```python
validator = InputValidator()

# Basic case
result = validator.validate_text("Python is great")
assert result == "Python is great"

# Whitespace handling
result = validator.validate_text("  Python   is    great  ")
assert result == "Python is great"

# Null byte removal
result = validator.validate_text("Hello\x00World")
assert result == "HelloWorld"

# Unicode normalization
result = validator.validate_text("café")  # e + combining accent
assert result == "café"  # single character é

# Newlines and tabs preserved
result = validator.validate_text("Line1\nLine2\tTabbed")
assert result == "Line1\nLine2\tTabbed"
```

---

## Exceptions

### ValidationError

**When Raised:**
1. **Wrong Type**: Input is not a string
2. **Empty Text**: String is empty after sanitization
3. **Invalid Encoding**: Text contains invalid UTF-8 sequences
4. **Size Exceeded**: Encoded bytes exceed max_size limit

**Message Format:**

```python
# Type error
f"Text must be a string, got {type(text).__name__}"
# Examples:
# - "Text must be a string, got int"
# - "Text must be a string, got NoneType"
# - "Text must be a string, got bytes"

# Empty error
"Text cannot be empty"

# Encoding error
"Text contains invalid UTF-8 encoding"

# Size error
f"Text exceeds maximum size ({max_size} bytes)"
# Example: "Text exceeds maximum size (10,000,000 bytes)"
```

**Exception Hierarchy:**
```python
ZapomniCoreError (base)
    └─ ValidationError (this exception)
```

**Exception Attributes:**
```python
class ValidationError(ZapomniCoreError):
    message: str  # Error message
    field: Optional[str] = "text"  # Field that failed validation
```

**Usage Example:**
```python
from zapomni_core.exceptions import ValidationError

validator = InputValidator()

try:
    result = validator.validate_text("")
except ValidationError as e:
    print(f"Validation failed: {e.message}")
    # Output: "Validation failed: Text cannot be empty"
    print(f"Field: {e.field}")
    # Output: "Field: text"
```

**Error Recovery:**

**Caller Responsibility:**
- InputValidator does NOT retry or recover from errors
- Raises ValidationError immediately
- Caller must handle exception

**No Retry Logic:**
- Validation errors are permanent (bad input from user)
- Retrying with same input will fail again
- User must fix input and resubmit

**Error Propagation:**
```
User Input
    ↓
MemoryProcessor.add_memory()
    ↓ (calls)
InputValidator.validate_text()
    ↓ (raises)
ValidationError
    ↓ (propagates up)
MemoryProcessor (catches, re-raises with context)
    ↓ (propagates up)
MCP Tool (catches, formats as MCP error response)
    ↓
User sees friendly error message
```

---

## Algorithm (Pseudocode)

```
FUNCTION validate_text(text, max_size):
    # ===== STEP 1: TYPE CHECK =====
    IF text is not instance of str:
        type_name = get_type_name(text)
        RAISE ValidationError(f"Text must be a string, got {type_name}")

    # ===== STEP 2: SANITIZE INPUT =====
    # Removes null bytes, normalizes Unicode, strips whitespace,
    # removes control chars (except \n, \t), collapses spaces
    sanitized_text = sanitize_input(text)

    # ===== STEP 3: CHECK NON-EMPTY =====
    IF sanitized_text is empty string:
        RAISE ValidationError("Text cannot be empty")

    # ===== STEP 4: CHECK ENCODING =====
    # Verifies text is valid UTF-8
    TRY:
        _check_encoding(sanitized_text)
    CATCH EncodingError:
        RAISE ValidationError("Text contains invalid UTF-8 encoding")

    # ===== STEP 5: CHECK SIZE =====
    # Encode to bytes to get accurate size
    text_bytes = encode_to_utf8(sanitized_text)
    byte_count = length(text_bytes)

    # Determine effective max size
    IF max_size is None:
        effective_max_size = MAX_TEXT_SIZE  # Class constant (10,000,000)
    ELSE:
        effective_max_size = max_size

    # Check size limit
    IF byte_count > effective_max_size:
        RAISE ValidationError(f"Text exceeds maximum size ({effective_max_size} bytes)")

    # ===== STEP 6: RETURN SANITIZED TEXT =====
    RETURN sanitized_text

END FUNCTION
```

**Complexity Analysis:**
- **Time Complexity**: O(n) where n = length of text
  - Type check: O(1)
  - Sanitization: O(n) (processes every character)
  - Encoding check: O(n) (validates every byte)
  - Size check: O(n) (encodes to bytes)
- **Space Complexity**: O(n)
  - Creates sanitized copy: O(n)
  - Encodes to bytes: O(n)
  - Total: O(n)

**Performance Estimates:**
- Small input (< 1 KB): < 1 ms
- Medium input (10 KB): < 5 ms
- Large input (1 MB): < 50 ms
- Maximum input (10 MB): < 500 ms

---

## Preconditions

**Required:**
- ✅ InputValidator instance initialized (`__init__` called)
- ✅ No dependencies required (InputValidator is self-contained)

**Not Required:**
- Database connection (validation is local)
- External services (no network calls)
- Configuration files (uses class constants)

**State:**
- InputValidator is stateless → no internal state required
- Each call is independent → no setup needed between calls

---

## Postconditions

**On Success (no exception):**
- ✅ Returned text is guaranteed valid:
  - Non-empty
  - Valid UTF-8
  - Within size limit
  - Sanitized (safe for storage)
- ✅ No state changes (InputValidator is stateless)
- ✅ No side effects (no logging, no I/O)

**On Failure (ValidationError raised):**
- ✅ No state changes
- ✅ No partial results (all-or-nothing validation)
- ✅ Clear error message indicating reason for failure
- ✅ Caller can inspect exception for details

**Guarantees:**
- **Idempotent**: Calling with same input always produces same result
- **Pure Function**: No side effects, output depends only on input
- **Thread-Safe**: Can be called concurrently from multiple threads

---

## Edge Cases & Handling

### Edge Case 1: Empty String

**Scenario:** User passes empty string `""`

**Input:**
```python
validator.validate_text("")
```

**Expected Behavior:**
```python
raise ValidationError("Text cannot be empty")
```

**Reason:** Empty text has no content to store, would waste resources (chunking, embedding, storage)

**Test Scenario:**
```python
def test_validate_text_empty_string_raises():
    validator = InputValidator()
    with pytest.raises(ValidationError, match="Text cannot be empty"):
        validator.validate_text("")
```

---

### Edge Case 2: Whitespace-Only String

**Scenario:** User passes string containing only whitespace `"   \n  \t  "`

**Input:**
```python
validator.validate_text("   \n  \t  ")
```

**Expected Behavior:**
```python
# After sanitization: strip() removes all whitespace
# Result is empty string ""
raise ValidationError("Text cannot be empty")
```

**Reason:** Whitespace-only text has no meaningful content

**Test Scenario:**
```python
def test_validate_text_whitespace_only_raises():
    validator = InputValidator()
    with pytest.raises(ValidationError, match="Text cannot be empty"):
        validator.validate_text("   \n  \t  ")
```

---

### Edge Case 3: Non-String Type (int)

**Scenario:** User passes integer instead of string

**Input:**
```python
validator.validate_text(123)
```

**Expected Behavior:**
```python
raise ValidationError("Text must be a string, got int")
```

**Reason:** Type safety - function expects strings, not numbers

**Test Scenario:**
```python
def test_validate_text_int_type_raises():
    validator = InputValidator()
    with pytest.raises(ValidationError, match="Text must be a string, got int"):
        validator.validate_text(123)
```

---

### Edge Case 4: Non-String Type (None)

**Scenario:** User passes `None` (common mistake)

**Input:**
```python
validator.validate_text(None)
```

**Expected Behavior:**
```python
raise ValidationError("Text must be a string, got NoneType")
```

**Reason:** None is not valid text content

**Test Scenario:**
```python
def test_validate_text_none_type_raises():
    validator = InputValidator()
    with pytest.raises(ValidationError, match="Text must be a string, got NoneType"):
        validator.validate_text(None)
```

---

### Edge Case 5: Non-String Type (bytes)

**Scenario:** User passes bytes instead of string

**Input:**
```python
validator.validate_text(b"Python is great")
```

**Expected Behavior:**
```python
raise ValidationError("Text must be a string, got bytes")
```

**Reason:** Function expects decoded strings, not raw bytes

**Test Scenario:**
```python
def test_validate_text_bytes_type_raises():
    validator = InputValidator()
    with pytest.raises(ValidationError, match="Text must be a string, got bytes"):
        validator.validate_text(b"Python is great")
```

---

### Edge Case 6: Text with Null Bytes

**Scenario:** User input contains null bytes `\x00` (malicious or corrupted data)

**Input:**
```python
validator.validate_text("Hello\x00World\x00!")
```

**Expected Behavior:**
```python
# Sanitization removes null bytes
return "HelloWorld!"
```

**Reason:** Null bytes are unsafe for storage, removed during sanitization

**Test Scenario:**
```python
def test_validate_text_null_bytes_removed():
    validator = InputValidator()
    result = validator.validate_text("Hello\x00World\x00!")
    assert result == "HelloWorld!"
    assert "\x00" not in result
```

---

### Edge Case 7: Text Exactly at Maximum Size

**Scenario:** Text is exactly 10,000,000 bytes (boundary case)

**Input:**
```python
text = "x" * 10_000_000  # Exactly 10 MB (ASCII, 1 byte per char)
validator.validate_text(text)
```

**Expected Behavior:**
```python
# Exactly at limit → VALID
return "x" * 10_000_000
```

**Reason:** Boundary inclusive (<=, not <)

**Test Scenario:**
```python
def test_validate_text_exactly_max_size_valid():
    validator = InputValidator()
    text = "x" * 10_000_000
    result = validator.validate_text(text)
    assert len(result.encode('utf-8')) == 10_000_000
```

---

### Edge Case 8: Text One Byte Over Maximum Size

**Scenario:** Text is 10,000,001 bytes (just over limit)

**Input:**
```python
text = "x" * 10_000_001  # 10 MB + 1 byte
validator.validate_text(text)
```

**Expected Behavior:**
```python
raise ValidationError("Text exceeds maximum size (10,000,000 bytes)")
```

**Reason:** Exceeds limit by 1 byte

**Test Scenario:**
```python
def test_validate_text_one_byte_over_max_raises():
    validator = InputValidator()
    text = "x" * 10_000_001
    with pytest.raises(ValidationError, match="Text exceeds maximum size"):
        validator.validate_text(text)
```

---

### Edge Case 9: Unicode Text with Multibyte Characters

**Scenario:** Unicode text where character count ≠ byte count

**Input:**
```python
# "café" with combining accent = 5 characters but potentially 6 bytes
text = "café" * 2_500_000  # ~7.5 MB in bytes (3 bytes per café)
validator.validate_text(text)
```

**Expected Behavior:**
```python
# Check byte count, not character count
# If within 10 MB → VALID
return normalized_text
```

**Reason:** Size limit is in bytes, not characters

**Test Scenario:**
```python
def test_validate_text_unicode_multibyte_within_limit():
    validator = InputValidator()
    # Create text close to but under 10 MB
    text = "café" * 2_000_000  # ~6 MB
    result = validator.validate_text(text)
    assert len(result.encode('utf-8')) < 10_000_000
```

---

### Edge Case 10: Invalid UTF-8 Encoding

**Scenario:** Text contains invalid UTF-8 byte sequences (e.g., from corrupted file)

**Input:**
```python
# This example is tricky because Python strings are already UTF-8
# Invalid UTF-8 would typically occur when decoding bytes incorrectly
# Simulate by attempting to create invalid sequence
invalid_text = "\udcff"  # Surrogate character (invalid in UTF-8)
validator.validate_text(invalid_text)
```

**Expected Behavior:**
```python
raise ValidationError("Text contains invalid UTF-8 encoding")
```

**Reason:** Invalid encoding cannot be safely stored

**Test Scenario:**
```python
def test_validate_text_invalid_utf8_raises():
    validator = InputValidator()
    # Use surrogate or other invalid UTF-8 sequence
    invalid_text = "\udcff"  # Lone surrogate
    with pytest.raises(ValidationError, match="invalid UTF-8 encoding"):
        validator.validate_text(invalid_text)
```

**Note:** In practice, Python 3 strings are always valid Unicode. Invalid UTF-8 usually comes from:
- Incorrectly decoded bytes: `bytes_data.decode('utf-8', errors='replace')`
- Surrogate pairs: `\udcXX` characters
- Implementation of `_check_encoding()` should catch these cases

---

### Edge Case 11: Custom max_size Smaller Than Default

**Scenario:** Using custom max_size parameter (e.g., 1000 bytes)

**Input:**
```python
text = "x" * 500  # 500 bytes
validator.validate_text(text, max_size=1000)
```

**Expected Behavior:**
```python
# 500 bytes < 1000 bytes → VALID
return "x" * 500
```

**Test Scenario:**
```python
def test_validate_text_custom_max_size_within_limit():
    validator = InputValidator()
    text = "x" * 500
    result = validator.validate_text(text, max_size=1000)
    assert len(result) == 500
```

---

### Edge Case 12: Custom max_size Exceeded

**Scenario:** Text exceeds custom max_size

**Input:**
```python
text = "x" * 2000  # 2000 bytes
validator.validate_text(text, max_size=1000)
```

**Expected Behavior:**
```python
raise ValidationError("Text exceeds maximum size (1000 bytes)")
```

**Test Scenario:**
```python
def test_validate_text_custom_max_size_exceeded_raises():
    validator = InputValidator()
    text = "x" * 2000
    with pytest.raises(ValidationError, match="Text exceeds maximum size \\(1000 bytes\\)"):
        validator.validate_text(text, max_size=1000)
```

---

## Test Scenarios (Complete List)

### Happy Path Tests

#### 1. test_validate_text_basic_success
**Input:** Valid plain text
```python
def test_validate_text_basic_success(validator, valid_text):
    """Test validation of normal, valid text."""
    result = validator.validate_text("Python is a programming language")
    assert result == "Python is a programming language"
    assert isinstance(result, str)
```

#### 2. test_validate_text_with_whitespace_normalization
**Input:** Text with leading/trailing whitespace
```python
def test_validate_text_with_whitespace_normalization(validator):
    """Test whitespace is stripped and normalized."""
    result = validator.validate_text("  Python   is    great  ")
    assert result == "Python is great"  # Normalized
```

#### 3. test_validate_text_with_newlines_and_tabs
**Input:** Text with newlines and tabs (should be preserved)
```python
def test_validate_text_with_newlines_and_tabs(validator):
    """Test newlines and tabs are preserved during validation."""
    text = "Line1\nLine2\tTabbed"
    result = validator.validate_text(text)
    assert result == "Line1\nLine2\tTabbed"
    assert "\n" in result
    assert "\t" in result
```

#### 4. test_validate_text_unicode_content
**Input:** Unicode text with special characters
```python
def test_validate_text_unicode_content(validator):
    """Test validation of Unicode text (non-ASCII)."""
    text = "Python программирование café 日本語"
    result = validator.validate_text(text)
    assert result == "Python программирование café 日本語"
```

#### 5. test_validate_text_at_boundary_size
**Input:** Text exactly at maximum size (10 MB)
```python
def test_validate_text_at_boundary_size(validator):
    """Test text exactly at max size is valid (boundary case)."""
    text = "x" * 10_000_000  # Exactly 10 MB
    result = validator.validate_text(text)
    assert len(result.encode('utf-8')) == 10_000_000
```

#### 6. test_validate_text_custom_max_size_success
**Input:** Text within custom max_size
```python
def test_validate_text_custom_max_size_success(validator):
    """Test custom max_size parameter works correctly."""
    text = "Short text"
    result = validator.validate_text(text, max_size=1000)
    assert result == "Short text"
```

---

### Error Tests

#### 7. test_validate_text_empty_string_raises
**Input:** Empty string `""`
```python
def test_validate_text_empty_string_raises(validator):
    """Test empty string raises ValidationError."""
    with pytest.raises(ValidationError, match="Text cannot be empty"):
        validator.validate_text("")
```

#### 8. test_validate_text_whitespace_only_raises
**Input:** Whitespace-only string `"   \n  \t  "`
```python
def test_validate_text_whitespace_only_raises(validator):
    """Test whitespace-only text raises ValidationError."""
    with pytest.raises(ValidationError, match="Text cannot be empty"):
        validator.validate_text("   \n  \t  ")
```

#### 9. test_validate_text_wrong_type_int_raises
**Input:** Integer `123`
```python
def test_validate_text_wrong_type_int_raises(validator):
    """Test non-string type (int) raises ValidationError."""
    with pytest.raises(ValidationError, match="Text must be a string, got int"):
        validator.validate_text(123)
```

#### 10. test_validate_text_wrong_type_none_raises
**Input:** `None`
```python
def test_validate_text_wrong_type_none_raises(validator):
    """Test None raises ValidationError."""
    with pytest.raises(ValidationError, match="Text must be a string, got NoneType"):
        validator.validate_text(None)
```

#### 11. test_validate_text_wrong_type_bytes_raises
**Input:** Bytes `b"text"`
```python
def test_validate_text_wrong_type_bytes_raises(validator):
    """Test bytes type raises ValidationError."""
    with pytest.raises(ValidationError, match="Text must be a string, got bytes"):
        validator.validate_text(b"Python is great")
```

#### 12. test_validate_text_wrong_type_list_raises
**Input:** List `["text"]`
```python
def test_validate_text_wrong_type_list_raises(validator):
    """Test list type raises ValidationError."""
    with pytest.raises(ValidationError, match="Text must be a string, got list"):
        validator.validate_text(["Python", "is", "great"])
```

#### 13. test_validate_text_exceeds_default_max_size_raises
**Input:** Text larger than 10 MB
```python
def test_validate_text_exceeds_default_max_size_raises(validator):
    """Test text exceeding default max size raises ValidationError."""
    huge_text = "x" * 10_000_001  # 1 byte over
    with pytest.raises(ValidationError, match="Text exceeds maximum size \\(10,000,000 bytes\\)"):
        validator.validate_text(huge_text)
```

#### 14. test_validate_text_exceeds_custom_max_size_raises
**Input:** Text exceeding custom max_size
```python
def test_validate_text_exceeds_custom_max_size_raises(validator):
    """Test text exceeding custom max_size raises ValidationError."""
    text = "x" * 2000  # 2000 bytes
    with pytest.raises(ValidationError, match="Text exceeds maximum size \\(1000 bytes\\)"):
        validator.validate_text(text, max_size=1000)
```

#### 15. test_validate_text_invalid_utf8_raises
**Input:** Invalid UTF-8 sequence
```python
def test_validate_text_invalid_utf8_raises(validator):
    """Test invalid UTF-8 encoding raises ValidationError."""
    # Surrogate character (invalid in UTF-8)
    invalid_text = "\udcff"
    with pytest.raises(ValidationError, match="invalid UTF-8 encoding"):
        validator.validate_text(invalid_text)
```

---

### Sanitization Tests

#### 16. test_validate_text_null_bytes_removed
**Input:** Text with null bytes
```python
def test_validate_text_null_bytes_removed(validator):
    """Test null bytes are removed during sanitization."""
    text = "Hello\x00World\x00!"
    result = validator.validate_text(text)
    assert result == "HelloWorld!"
    assert "\x00" not in result
```

#### 17. test_validate_text_control_characters_removed
**Input:** Text with control characters
```python
def test_validate_text_control_characters_removed(validator):
    """Test control characters (except \\n, \\t) are removed."""
    text = "Hello\x01\x02World\x03!"
    result = validator.validate_text(text)
    assert result == "HelloWorld!"
    assert "\x01" not in result
```

#### 18. test_validate_text_unicode_normalization
**Input:** Unicode with combining characters
```python
def test_validate_text_unicode_normalization(validator):
    """Test Unicode is normalized (NFC)."""
    # "café" with e + combining accent (2 characters for é)
    text_decomposed = "cafe\u0301"  # e + combining acute accent
    result = validator.validate_text(text_decomposed)

    # Should be normalized to single character é
    expected = "café"  # Composed form (NFC)
    assert result == expected
```

#### 19. test_validate_text_multiple_spaces_collapsed
**Input:** Text with multiple consecutive spaces
```python
def test_validate_text_multiple_spaces_collapsed(validator):
    """Test multiple spaces collapsed to single space."""
    text = "Python    is     great"
    result = validator.validate_text(text)
    assert result == "Python is great"
```

---

### Integration Tests

#### 20. test_validate_text_calls_sanitize_input
**Input:** Any text
```python
def test_validate_text_calls_sanitize_input(validator, mocker):
    """Test validate_text calls sanitize_input internally."""
    mock_sanitize = mocker.spy(validator, 'sanitize_input')
    validator.validate_text("Test text")
    mock_sanitize.assert_called_once_with("Test text")
```

#### 21. test_validate_text_calls_check_encoding
**Input:** Any text
```python
def test_validate_text_calls_check_encoding(validator, mocker):
    """Test validate_text calls _check_encoding internally."""
    mock_check = mocker.spy(validator, '_check_encoding')
    validator.validate_text("Test text")
    mock_check.assert_called_once()
```

---

### Performance Tests

#### 22. test_validate_text_small_input_performance
**Input:** Small text (< 1 KB)
```python
def test_validate_text_small_input_performance(validator):
    """Test validation of small input is fast (< 10ms)."""
    import time
    text = "Python is great" * 50  # ~750 bytes

    start = time.time()
    validator.validate_text(text)
    elapsed = (time.time() - start) * 1000  # Convert to ms

    assert elapsed < 10  # Should be < 10ms
```

#### 23. test_validate_text_large_input_performance
**Input:** Large text (~1 MB)
```python
def test_validate_text_large_input_performance(validator):
    """Test validation of large input completes in reasonable time."""
    import time
    text = "x" * 1_000_000  # 1 MB

    start = time.time()
    validator.validate_text(text)
    elapsed = (time.time() - start) * 1000  # ms

    assert elapsed < 100  # Should be < 100ms for 1 MB
```

---

## Performance Requirements

### Latency

**Target Latency:**
- **Small input (< 1 KB)**: < 1 ms
- **Medium input (10 KB)**: < 5 ms
- **Large input (1 MB)**: < 50 ms
- **Maximum input (10 MB)**: < 500 ms

**Measurement Points:**
- Type check: O(1) - negligible
- Sanitization: O(n) - dominant factor
- Encoding check: O(n) - moderate
- Size check: O(n) - encode operation

### Throughput

**Concurrent Calls:**
- InputValidator is stateless and thread-safe
- Can handle concurrent calls from multiple threads
- Expected throughput: 1000+ validations/sec (small inputs)

**Bottlenecks:**
- String sanitization (iterates characters)
- Unicode normalization (NFC)
- UTF-8 encoding (for size check)

### Resource Usage

**Memory:**
- **Space Complexity**: O(n) where n = text length
- Creates sanitized copy: ~same size as input
- Encodes to bytes: ~same size as input
- **Peak Memory**: ~2x input size

**CPU:**
- **Time Complexity**: O(n)
- Single-threaded processing
- No parallelization opportunities (sequential validation steps)

**Optimization Opportunities:**
1. **Early Exit**: Type check BEFORE sanitization (fail fast)
2. **Lazy Encoding**: Only encode to bytes if size check needed
3. **Streaming Validation**: For very large inputs (future)

---

## Security Considerations

### Input Validation

**Protection Against:**
- ✅ **Type Confusion**: Strict type checking (`isinstance(text, str)`)
- ✅ **Null Byte Injection**: Removed during sanitization
- ✅ **Control Character Injection**: Removed (except `\n`, `\t`)
- ✅ **Encoding Attacks**: UTF-8 validation prevents invalid sequences
- ✅ **DoS via Large Input**: Size limit prevents memory exhaustion

### Data Sanitization

**Sanitization Steps:**
1. Strip leading/trailing whitespace
2. Remove null bytes (`\x00`)
3. Remove control characters (`\x01-\x1F` except `\n`, `\t`)
4. Normalize Unicode (NFC)
5. Collapse multiple spaces

**Safe for Storage:** After validation, text can be safely:
- Stored in database (no SQL injection via control chars)
- Embedded via API (no encoding issues)
- Displayed to users (no XSS via control chars)

### Error Messages

**No Sensitive Data Leakage:**
- Error messages reveal only:
  - Type of input (e.g., "got int")
  - Reason for failure (e.g., "too large")
  - Size limit (public constant)
- Error messages do NOT reveal:
  - Actual input content
  - Internal system details
  - Stack traces (handled at higher level)

**Example Safe Error:**
```python
"Text exceeds maximum size (10,000,000 bytes)"
# Does NOT include actual text or size
```

---

## Related Functions

### Calls

**`sanitize_input(text: str) -> str`**
- Purpose: Remove unsafe characters, normalize whitespace and Unicode
- When: Step 2 of validation, after type check
- Returns: Sanitized text

**`_check_encoding(text: str) -> None`**
- Purpose: Verify text is valid UTF-8
- When: Step 4 of validation, after emptiness check
- Raises: ValidationError if invalid encoding

### Called By

**`MemoryProcessor.add_memory(text: str, ...) -> str`**
- Purpose: Add user memory to knowledge graph
- When: First step before chunking
- Usage:
  ```python
  text = self.validator.validate_text(text)  # Validate first
  chunks = self.chunker.chunk(text)  # Then process
  ```

**Potential Future Callers:**
- `MemoryProcessor.update_memory(memory_id, text)` - Update existing memory
- `MemoryProcessor.replace_memory(memory_id, text)` - Replace memory content

### Related Methods in InputValidator

**Similar Validation Methods:**
- `validate_query(query: str) -> str` - Validates search queries (different max length)
- `validate_metadata(metadata: dict) -> dict` - Validates metadata structure
- `validate_limit(limit: int) -> int` - Validates pagination limits

**Helper Methods:**
- `sanitize_input(text: str) -> str` - Called by validate_text
- `_check_encoding(text: str) -> None` - Called by validate_text
- `_remove_null_bytes(text: str) -> str` - Called by sanitize_input

---

## Implementation Notes

### Libraries Used

**Standard Library Only:**
- `typing` - Type hints (Optional, str)
- `unicodedata` - Unicode normalization (NFC)
- Built-in `str` methods - strip(), encode(), isinstance()

**No External Dependencies:**
- InputValidator is self-contained
- No database, network, or filesystem operations

### Implementation Hints

**Type Checking:**
```python
if not isinstance(text, str):
    type_name = type(text).__name__
    raise ValidationError(f"Text must be a string, got {type_name}")
```

**Sanitization Call:**
```python
text = self.sanitize_input(text)
```

**Emptiness Check:**
```python
if not text:  # Empty string after sanitization
    raise ValidationError("Text cannot be empty")
```

**Encoding Validation:**
```python
self._check_encoding(text)  # Raises if invalid
```

**Size Validation:**
```python
text_bytes = text.encode('utf-8')
effective_max_size = max_size if max_size is not None else self.MAX_TEXT_SIZE

if len(text_bytes) > effective_max_size:
    raise ValidationError(f"Text exceeds maximum size ({effective_max_size} bytes)")
```

### Known Limitations

**Character vs. Byte Count:**
- Size limit is in BYTES, not characters
- Unicode characters can be 1-4 bytes
- User might expect character limit, gets byte limit

**Unicode Normalization:**
- NFC normalization may change visual appearance slightly
- User input: `"café"` (e + combining accent)
- Stored result: `"café"` (composed é)
- Visually identical but different byte representation

**Control Character Handling:**
- Only `\n` and `\t` preserved
- Other control characters (`\r`, `\v`, `\f`) removed
- May affect text with special formatting

### Future Enhancements

**Potential Improvements:**

1. **Streaming Validation:**
   - For very large inputs (>100 MB)
   - Validate in chunks instead of loading entire text
   - Requires async/generator approach

2. **Configurable Sanitization:**
   - Allow caller to specify which characters to preserve
   - Example: `validate_text(text, preserve=['\r', '\v'])`

3. **Detailed Error Context:**
   - Include position of invalid encoding
   - Example: `"Invalid UTF-8 at byte position 1234"`

4. **Performance Profiling:**
   - Add optional timing metrics
   - Return validation time with result
   - Help identify slow validations

5. **Batch Validation:**
   - Validate multiple texts in one call
   - Parallel processing for better throughput
   - Example: `validate_text_batch(texts: list[str])`

---

## References

### Component Spec
- [InputValidator Component](../level2/input_validator_component.md) - Parent component specification

### Module Spec
- [zapomni_core Module](../level1/zapomni_core_module.md) - Parent module specification

### Related Function Specs
- `sanitize_input()` - Called by this function
- `_check_encoding()` - Called by this function
- `validate_query()` - Similar validation for queries
- `validate_metadata()` - Similar validation for metadata

### External Documentation
- [Python Unicode HOWTO](https://docs.python.org/3/howto/unicode.html)
- [Unicode Normalization (NFC)](https://unicode.org/reports/tr15/)
- [UTF-8 Encoding](https://datatracker.ietf.org/doc/html/rfc3629)

---

## Appendix: Validation Examples

### Example 1: Basic Validation

```python
from zapomni_core.validation import InputValidator
from zapomni_core.exceptions import ValidationError

validator = InputValidator()

# Valid text
try:
    result = validator.validate_text("Python is a programming language")
    print(f"✅ Valid: {result}")
except ValidationError as e:
    print(f"❌ Error: {e.message}")

# Output: ✅ Valid: Python is a programming language
```

### Example 2: Whitespace Handling

```python
validator = InputValidator()

# Input with extra whitespace
text = "  Python   is    great  "
result = validator.validate_text(text)

print(f"Input:  '{text}'")
print(f"Output: '{result}'")

# Output:
# Input:  '  Python   is    great  '
# Output: 'Python is great'
```

### Example 3: Error Handling

```python
validator = InputValidator()

# Test various invalid inputs
invalid_inputs = [
    ("", "empty string"),
    ("   ", "whitespace only"),
    (123, "integer"),
    (None, "None"),
    (b"bytes", "bytes"),
    ("x" * 10_000_001, "too large"),
]

for input_val, description in invalid_inputs:
    try:
        validator.validate_text(input_val)
        print(f"❌ {description}: Should have raised ValidationError")
    except ValidationError as e:
        print(f"✅ {description}: {e.message}")

# Output:
# ✅ empty string: Text cannot be empty
# ✅ whitespace only: Text cannot be empty
# ✅ integer: Text must be a string, got int
# ✅ None: Text must be a string, got NoneType
# ✅ bytes: Text must be a string, got bytes
# ✅ too large: Text exceeds maximum size (10,000,000 bytes)
```

### Example 4: Sanitization in Action

```python
validator = InputValidator()

# Text with null bytes and control characters
dirty_text = "Hello\x00World\x01\x02!"
clean_text = validator.validate_text(dirty_text)

print(f"Input (repr):  {repr(dirty_text)}")
print(f"Output (repr): {repr(clean_text)}")

# Output:
# Input (repr):  'Hello\x00World\x01\x02!'
# Output (repr): 'HelloWorld!'
```

### Example 5: Custom Max Size

```python
validator = InputValidator()

# Use custom max size for short notes
short_text = "This is a short note"
result = validator.validate_text(short_text, max_size=1000)
print(f"✅ Within 1KB limit: {result}")

# Exceed custom max size
long_text = "x" * 2000
try:
    validator.validate_text(long_text, max_size=1000)
except ValidationError as e:
    print(f"❌ Exceeds 1KB limit: {e.message}")

# Output:
# ✅ Within 1KB limit: This is a short note
# ❌ Exceeds 1KB limit: Text exceeds maximum size (1000 bytes)
```

---

**Document Status:** Draft v1.0
**Created:** 2025-11-23
**Author:** Goncharenko Anton aka alienxs2
**Copyright:** Copyright (c) 2025 Goncharenko Anton aka alienxs2
**License:** MIT License

**Function Complexity:** Medium
**Edge Cases Identified:** 12
**Test Scenarios:** 23
**Ready for Implementation:** Yes ✅
