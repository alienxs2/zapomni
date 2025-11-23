# EntityExtractor.normalize_entity() - Function Specification

**Level:** 3 | **Component:** EntityExtractor | **Module:** zapomni_core
**Author:** Goncharenko Anton aka alienxs2 | **Status:** Draft | **V:** 1.0

## Signature
```python
def normalize_entity(self, entity_text: str, entity_type: str) -> str:
    """
    Normalize entity text for consistency.
    
    Args:
        entity_text: Raw entity text from extraction
        entity_type: Entity type (PERSON, ORG, GPE, etc.)
        
    Returns:
        Normalized entity text
        
    Example:
        >>> extractor.normalize_entity("  Python  ", "LANGUAGE")
        'Python'
        >>> extractor.normalize_entity("U.S.A.", "GPE")
        'USA'
    """
```

## Purpose

Normalize entity text for consistent storage and matching:
- Strip whitespace
- Remove extra spaces (multiple → single)
- Title case for PERSON entities
- Upper case for acronyms/abbreviations (GPE)
- Preserve case for other types

## Edge Cases
1. "  entity  " → "entity" (stripped)
2. "multiple   spaces" → "multiple spaces"
3. "john doe" (PERSON) → "John Doe" (title case)
4. "u.s.a." (GPE) → "USA" (acronym normalization)
5. "" → "" (empty preserved)
6. "123" → "123" (numbers preserved)

## Algorithm
```
1. Strip leading/trailing whitespace
2. Collapse multiple spaces to single
3. If entity_type == "PERSON", apply title case
4. If entity_type == "GPE" and looks like acronym, uppercase
5. Otherwise preserve case
6. Return normalized text
```

## Tests (10)
1. test_normalize_strips_whitespace
2. test_normalize_collapses_spaces
3. test_normalize_person_title_case
4. test_normalize_gpe_acronym_uppercase
5. test_normalize_empty_string
6. test_normalize_numbers_preserved
7. test_normalize_special_chars_preserved
8. test_normalize_unicode_preserved
9. test_normalize_org_case_preserved
10. test_normalize_idempotent
