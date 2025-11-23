# EntityExtractor.extract_entities() - Function Specification

**Level:** 3 (Function)
**Component:** EntityExtractor
**Module:** zapomni_core
**Author:** Goncharenko Anton aka alienxs2
**Status:** Draft
**Version:** 1.0
**Created:** 2025-11-23

---

## Function Signature

```python
def extract_entities(
    self,
    text: str
) -> List[Entity]:
    """
    Extract named entities from text using hybrid SpaCy + LLM approach.

    Identifies entities (people, organizations, technologies, concepts) from input
    text using a two-stage pipeline:
    1. Fast SpaCy NER extraction (80% recall, 60% precision)
    2. Optional LLM refinement via Ollama (90% precision, validates + enhances)

    Target metrics:
    - Precision: 80%+
    - Recall: 75%+
    - Latency: < 50ms (SpaCy only), ~1-2s (with LLM)

    Args:
        text: Input text to extract entities from
            - Min length: 10 characters
            - Max length: 100,000 characters
            - Must be valid UTF-8

    Returns:
        List[Entity]: Extracted entities with metadata
            - Sorted by confidence (highest first)
            - Deduplicated (merged similar entities)
            - Normalized names (e.g., "Python lang" → "Python")

    Raises:
        ValidationError: If text invalid (empty, too long, bad encoding)
        ExtractionError: If both SpaCy and LLM fail

    Example:
        >>> extractor = EntityExtractor(spacy_model=nlp)
        >>> text = "Python was created by Guido van Rossum in 1991"
        >>> entities = extractor.extract_entities(text)
        >>> [(e.name, e.type) for e in entities]
        [('Python', 'TECHNOLOGY'), ('Guido van Rossum', 'PERSON')]
    """
```

---

## Purpose & Context

Extracts structured entities from unstructured text for knowledge graph construction. Critical for building semantic relationships in Zapomni's memory system.

---

## Parameters (Detailed)

### text: str

- **Type:** UTF-8 string
- **Constraints:**
  - Min: 10 chars (too short for meaningful extraction)
  - Max: 100,000 chars (~20,000 tokens)
  - Non-empty after stripping

**Validation:**
```python
if not text or len(text.strip()) < 10:
    raise ValidationError("Text too short for entity extraction (min 10 chars)")
if len(text) > 100000:
    raise ValidationError("Text exceeds max length (100,000 chars)")
```

---

## Return Value

**Type:** `List[Entity]`

**Entity Structure:**
```python
@dataclass
class Entity:
    name: str  # Normalized entity name
    type: str  # PERSON, ORG, TECHNOLOGY, CONCEPT, etc.
    description: str  # Brief description
    confidence: float  # 0.0-1.0
    mentions: int  # Occurrence count
    source_span: Optional[str]  # Original text span
```

**Guarantees:**
- Sorted by confidence descending
- No duplicates (merged)
- All confidences >= 0.7 (confidence_threshold)

---

## Algorithm (Pseudocode)

```
FUNCTION extract_entities(self, text: str) -> List[Entity]:
    # Step 1: Validate input
    VALIDATE text is non-empty, valid UTF-8, within length limits

    # Step 2: SpaCy extraction
    spacy_entities = self._spacy_extract(text)
    # Returns: [Entity(name="Python", type="TECHNOLOGY", confidence=0.85), ...]

    # Step 3: LLM refinement (if enabled)
    IF self.enable_llm_refinement:
        refined_entities = self._llm_refine(text, spacy_entities)
    ELSE:
        refined_entities = spacy_entities

    # Step 4: Deduplication
    unique_entities = self._deduplicate_entities(refined_entities)

    # Step 5: Filter by confidence
    filtered_entities = [e FOR e IN unique_entities IF e.confidence >= self.confidence_threshold]

    # Step 6: Sort by confidence
    sorted_entities = SORT filtered_entities BY confidence DESC

    RETURN sorted_entities
END FUNCTION
```

---

## Edge Cases & Handling

### Edge Case 1: No Entities Found

**Scenario:** Text has no recognizable entities

**Input:** `"The weather is nice today."`

**Expected:** Empty list `[]`

**Test:**
```python
def test_extract_entities_none_found():
    extractor = EntityExtractor(nlp)
    entities = extractor.extract_entities("The weather is nice.")
    assert len(entities) == 0
```

---

### Edge Case 2: Same Entity Multiple Times

**Scenario:** Entity mentioned repeatedly

**Input:** `"Python is great. I love Python. Python rocks."`

**Processing:**
1. SpaCy finds 3 "Python" mentions
2. Deduplicate: merge into 1 Entity with mentions=3
3. Return single Entity

**Expected:**
```python
entities = extractor.extract_entities(text)
assert len(entities) == 1
assert entities[0].name == "Python"
assert entities[0].mentions == 3
```

---

### Edge Case 3: Low-Confidence Entities

**Scenario:** SpaCy returns entities with confidence < 0.7

**Processing:**
1. Extract entities
2. Filter: confidence < 0.7 removed
3. Return only high-confidence entities

---

### Edge Case 4: Very Long Text (50k+ chars)

**Scenario:** Large document

**Processing:**
1. SpaCy processes in chunks internally
2. May take 100-200ms
3. Returns all entities

---

### Edge Case 5: Non-English Text

**Scenario:** Text in Russian/Chinese/etc.

**Processing:**
1. SpaCy model is English-only (en_core_web_sm)
2. May extract few/no entities
3. Return sparse results (not an error)

**Note:** Phase 2 could support multilingual models

---

### Edge Case 6: Code Snippets

**Scenario:** Text is programming code

**Input:**
```python
"""
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
```

**Processing:**
1. SpaCy may identify "fibonacci" as entity
2. Type might be PRODUCT or CONCEPT
3. LLM refinement improves accuracy (Phase 2)

---

## Test Scenarios (Complete List)

### Happy Path Tests

**1. test_extract_entities_person**
- Input: "Guido van Rossum created Python"
- Expected: [Entity(name="Guido van Rossum", type="PERSON"), ...]

**2. test_extract_entities_organization**
- Input: "OpenAI developed GPT-4"
- Expected: [Entity(name="OpenAI", type="ORG"), ...]

**3. test_extract_entities_technology**
- Input: "Docker is a containerization platform"
- Expected: [Entity(name="Docker", type="TECHNOLOGY"), ...]

**4. test_extract_entities_multiple**
- Input: Text with multiple entity types
- Expected: List with all entities

**5. test_extract_entities_sorted_by_confidence**
- Expected: Entities sorted highest confidence first

---

### Edge Case Tests

**6. test_extract_entities_none_found**
- Edge case 1

**7. test_extract_entities_repeated_mentions**
- Edge case 2

**8. test_extract_entities_low_confidence_filtered**
- Edge case 3

**9. test_extract_entities_long_text**
- Edge case 4

**10. test_extract_entities_non_english**
- Edge case 5

**11. test_extract_entities_code_snippet**
- Edge case 6

---

### Validation Tests

**12. test_extract_entities_empty_text**
- Input: ""
- Expected: ValidationError

**13. test_extract_entities_text_too_short**
- Input: "Hi"
- Expected: ValidationError

**14. test_extract_entities_text_too_long**
- Input: "x" * 100001
- Expected: ValidationError

---

### Integration Tests

**15. test_extract_entities_with_llm_refinement**
- LLM enabled
- Verify improved accuracy

**16. test_extract_entities_deduplication**
- Verify duplicate removal

---

## Performance Requirements

- **SpaCy only:** < 50ms (P95)
- **With LLM:** ~1-2s (depends on Ollama)
- **Throughput:** 20-50 docs/sec (SpaCy), 1-2 docs/sec (LLM)

---

## References

### Component Spec
- [EntityExtractor Component](../level2/entity_extractor_component.md)

---

## Document Status

**Version:** 1.0
**Created:** 2025-11-23
**Author:** Goncharenko Anton aka alienxs2
**License:** MIT
**Status:** Draft

**Estimated Implementation:** 2-3 hours
**Lines of Code:** ~120 lines
**Test Coverage Target:** 90%+
**Test File:** `tests/unit/core/test_entity_extractor_extract_entities.py`
