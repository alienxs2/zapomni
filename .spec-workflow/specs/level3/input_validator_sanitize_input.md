# InputValidator.sanitize_input() - Function Specification

**Level:** 3 | **Component:** InputValidator | **Module:** zapomni_core
**Author:** Goncharenko Anton aka alienxs2 | **Status:** Draft | **V:** 1.0

## Signature
```python
@staticmethod
def sanitize_input(text: str, max_length: int = 10_000_000) -> str:
    """
    Sanitize and validate text input.
    
    Args:
        text: Raw text input
        max_length: Maximum allowed length (default: 10MB)
        
    Returns:
        Sanitized text (stripped whitespace)
        
    Raises:
        ValidationError: If text empty after strip
        ValidationError: If text exceeds max_length
        ValidationError: If text contains invalid UTF-8
    """
```

## Purpose

Static method that sanitizes text input:
- Strips leading/trailing whitespace
- Validates non-empty after strip
- Validates length <= max_length
- Validates UTF-8 encoding

## Edge Cases
1. text = "" → ValidationError("Text cannot be empty")
2. text = "   " → ValidationError("Text cannot be empty") (after strip)
3. text = "a" * (max_length + 1) → ValidationError("Text exceeds maximum length")
4. text = "valid" → Returns "valid"
5. text = "  valid  " → Returns "valid" (stripped)
6. text with invalid UTF-8 bytes → ValidationError("Text must be valid UTF-8")

## Algorithm
```
1. Strip leading/trailing whitespace
2. Check non-empty after strip
3. Check length <= max_length
4. Attempt encode/decode UTF-8 to verify valid encoding
5. Return sanitized text
```

## Tests (10)
1. test_sanitize_empty_raises
2. test_sanitize_whitespace_raises
3. test_sanitize_too_long_raises
4. test_sanitize_invalid_utf8_raises
5. test_sanitize_valid_text_success
6. test_sanitize_strips_whitespace
7. test_sanitize_max_length_boundary
8. test_sanitize_single_char_success
9. test_sanitize_unicode_success
10. test_sanitize_newlines_preserved
