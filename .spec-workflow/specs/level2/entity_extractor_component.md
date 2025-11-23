# EntityExtractor - Component Specification

**Level:** 2 (Component)
**Module:** zapomni_core
**Author:** Goncharenko Anton aka alienxs2
**Status:** Draft
**Version:** 1.0
**Created:** 2025-11-23

---

## Overview

### Purpose

EntityExtractor is a hybrid NER (Named Entity Recognition) component that extracts entities (people, organizations, concepts, technologies) from text using a two-stage approach: fast SpaCy-based extraction followed by optional LLM refinement via Ollama (Phase 2).

This component is responsible for identifying and normalizing entities that will become nodes in the knowledge graph, achieving the target precision of 80%+ and recall of 75%+ as specified in the parent module spec.

### Responsibilities

1. **Entity Extraction:** Identify entities from text using SpaCy NER
2. **Entity Normalization:** Standardize entity names (e.g., "Python lang" → "Python")
3. **LLM Refinement (Phase 2):** Validate and enhance entities using Ollama
4. **Relationship Detection (Phase 2):** Identify relationships between entities
5. **Confidence Scoring:** Assign confidence scores to extracted entities
6. **Deduplication:** Merge duplicate entities across text chunks

### Position in Module

EntityExtractor is a **core processing component** within zapomni_core, sitting between the embedding pipeline and knowledge graph construction:

```
MemoryProcessor
    ↓
TextChunker → OllamaEmbedder → EntityExtractor → GraphBuilder
                                      ↑
                                      │
                                OllamaClient (LLM refinement)
                                SpaCy (NER)
```

**Used by:**
- `MemoryProcessor.build_knowledge_graph()` - Main orchestration
- `GraphBuilder` - Consumes extracted entities to create graph nodes

**Dependencies:**
- SpaCy NER pipeline (en_core_web_sm)
- OllamaClient (for LLM refinement in Phase 2)

---

## Class Definition

### Class Diagram

```
┌─────────────────────────────────────────────┐
│          EntityExtractor                    │
├─────────────────────────────────────────────┤
│ - spacy_nlp: Language                       │
│ - ollama_client: Optional[OllamaClient]     │
│ - enable_llm_refinement: bool               │
│ - confidence_threshold: float               │
│ - entity_types: set[str]                    │
├─────────────────────────────────────────────┤
│ + extract_entities(text: str) -> List[Entity]              │
│ + extract_relationships(text: str, entities: List[Entity]) │
│   -> List[Relationship]                     │
│ + normalize_entity(entity: str, type: str) -> str          │
│ - _spacy_extract(text: str) -> List[Entity] │
│ - _llm_refine(text: str, entities: List[Entity])           │
│   -> List[Entity]                           │
│ - _detect_relationship_llm(text: str, ent1: Entity,        │
│   ent2: Entity) -> Optional[Relationship]   │
│ - _deduplicate_entities(entities: List[Entity])            │
│   -> List[Entity]                           │
└─────────────────────────────────────────────┘
```

### Full Class Signature

```python
from typing import Optional, List, Set
from dataclasses import dataclass
import spacy
from spacy.language import Language

@dataclass
class Entity:
    """
    Knowledge graph entity extracted from text.

    Attributes:
        name: Normalized entity name (e.g., "Python")
        type: Entity type (PERSON, ORG, TECHNOLOGY, CONCEPT, etc.)
        description: Brief description of the entity
        confidence: Extraction confidence score (0.0-1.0)
        mentions: Number of times entity appears in text
        source_span: Original text span where entity was found
    """
    name: str
    type: str
    description: str
    confidence: float
    mentions: int = 1
    source_span: Optional[str] = None

@dataclass
class Relationship:
    """
    Knowledge graph relationship between entities.

    Attributes:
        source_entity: Source entity name
        target_entity: Target entity name
        relationship_type: Type (CREATED_BY, USES, IS_A, PART_OF, etc.)
        confidence: Detection confidence (0.0-1.0)
        evidence: Text snippet supporting this relationship
    """
    source_entity: str
    target_entity: str
    relationship_type: str
    confidence: float
    evidence: str


class EntityExtractor:
    """
    Hybrid entity extraction component using SpaCy + LLM refinement.

    Implements two-stage entity extraction:
    1. Fast SpaCy NER pass (80% recall, ~60% precision)
    2. Optional LLM refinement (90% precision, validates + enhances)

    Achieves target metrics:
    - Entity extraction: 80%+ precision, 75%+ recall
    - Relationship detection: 70%+ precision, 65%+ recall (Phase 2)

    Performance:
    - SpaCy extraction: < 10ms per document
    - LLM refinement: ~1-2s per document (Phase 2)
    - Hybrid approach: 3x faster than LLM-only

    Attributes:
        spacy_nlp: Loaded SpaCy language model (en_core_web_sm)
        ollama_client: Optional Ollama client for LLM refinement (Phase 2)
        enable_llm_refinement: Whether to use LLM refinement (default: False)
        confidence_threshold: Minimum confidence to keep entity (default: 0.7)
        entity_types: Set of entity types to extract

    Example:
        ```python
        import spacy
        from zapomni_core.extraction import EntityExtractor

        # Phase 1: SpaCy only
        nlp = spacy.load("en_core_web_sm")
        extractor = EntityExtractor(spacy_model=nlp)

        entities = extractor.extract_entities(
            "Python was created by Guido van Rossum in 1991"
        )
        # Returns: [
        #   Entity(name="Python", type="TECHNOLOGY", confidence=0.85, ...),
        #   Entity(name="Guido van Rossum", type="PERSON", confidence=0.92, ...)
        # ]

        # Phase 2: With LLM refinement
        from zapomni_core.embeddings import OllamaClient
        ollama = OllamaClient(host="http://localhost:11434")
        extractor = EntityExtractor(
            spacy_model=nlp,
            ollama_client=ollama,
            enable_llm_refinement=True
        )

        entities = extractor.extract_entities(
            "Django is a web framework for Python"
        )
        # LLM enhances: Django (TECHNOLOGY), Python (TECHNOLOGY)
        # with better descriptions and confidence
        ```
    """

    # Supported entity types
    DEFAULT_ENTITY_TYPES: Set[str] = {
        "PERSON",       # People (Guido van Rossum)
        "ORG",          # Organizations (OpenAI, Google)
        "GPE",          # Geopolitical entities (USA, San Francisco)
        "TECHNOLOGY",   # Technologies (Python, Docker)
        "CONCEPT",      # Abstract concepts (machine learning)
        "PRODUCT",      # Products (Claude, GPT-4)
        "EVENT",        # Events (PyCon 2024)
        "DATE",         # Dates and times
    }

    def __init__(
        self,
        spacy_model: Language,
        ollama_client: Optional['OllamaClient'] = None,
        enable_llm_refinement: bool = False,
        confidence_threshold: float = 0.7,
        entity_types: Optional[Set[str]] = None
    ) -> None:
        """
        Initialize EntityExtractor with NER pipeline.

        Args:
            spacy_model: Loaded SpaCy language model (must have NER component)
            ollama_client: Optional Ollama client for LLM refinement (Phase 2)
            enable_llm_refinement: Enable LLM-based refinement (default: False)
                - Phase 1: False (SpaCy only)
                - Phase 2: True (hybrid SpaCy + LLM)
            confidence_threshold: Minimum confidence to keep entity (0.0-1.0)
                - Default: 0.7 (filters low-confidence entities)
                - Lower = more entities but lower precision
                - Higher = fewer entities but higher precision
            entity_types: Set of entity types to extract (default: all supported)

        Raises:
            ValueError: If spacy_model doesn't have NER component
            ValueError: If confidence_threshold not in [0.0, 1.0]
            ValueError: If enable_llm_refinement=True but ollama_client is None

        Example:
            ```python
            import spacy
            nlp = spacy.load("en_core_web_sm")

            # Phase 1: SpaCy only
            extractor = EntityExtractor(spacy_model=nlp)

            # Phase 2: With LLM refinement
            from zapomni_core.embeddings import OllamaClient
            ollama = OllamaClient()
            extractor = EntityExtractor(
                spacy_model=nlp,
                ollama_client=ollama,
                enable_llm_refinement=True
            )
            ```
        """

    def extract_entities(self, text: str) -> List[Entity]:
        """
        Extract entities from text using hybrid SpaCy + LLM approach.

        Workflow:
        1. SpaCy NER pass (fast, high recall)
        2. Normalize entity names
        3. Optional LLM refinement (if enabled, Phase 2)
        4. Deduplicate entities
        5. Filter by confidence threshold
        6. Return sorted by confidence (descending)

        Args:
            text: Input text to extract entities from (max 100,000 chars)

        Returns:
            List of Entity objects sorted by confidence (high to low)
            - Each entity has: name, type, description, confidence, mentions
            - Deduplicated (same name+type merged)
            - Filtered by confidence_threshold

        Raises:
            ValidationError: If text is empty or exceeds max length
            ExtractionError: If SpaCy processing fails
            LLMError: If LLM refinement enabled but fails (Phase 2)

        Performance:
            - SpaCy only: < 10ms per doc
            - With LLM refinement: 1-2s per doc (Phase 2)

        Example:
            ```python
            extractor = EntityExtractor(spacy_model=nlp)

            entities = extractor.extract_entities(
                "Python was created by Guido van Rossum at CWI in Amsterdam"
            )

            # Returns:
            # [
            #   Entity(
            #     name="Python",
            #     type="TECHNOLOGY",
            #     description="Programming language",
            #     confidence=0.85,
            #     mentions=1
            #   ),
            #   Entity(
            #     name="Guido van Rossum",
            #     type="PERSON",
            #     description="Creator of Python",
            #     confidence=0.92,
            #     mentions=1
            #   ),
            #   Entity(
            #     name="CWI",
            #     type="ORG",
            #     description="Research institute",
            #     confidence=0.78,
            #     mentions=1
            #   ),
            #   Entity(
            #     name="Amsterdam",
            #     type="GPE",
            #     description="City in Netherlands",
            #     confidence=0.88,
            #     mentions=1
            #   )
            # ]
            ```
        """

    def extract_relationships(
        self,
        text: str,
        entities: List[Entity]
    ) -> List[Relationship]:
        """
        Extract relationships between entities using LLM (Phase 2 only).

        This method is only available when enable_llm_refinement=True.
        Uses LLM to detect semantic relationships between entity pairs.

        Algorithm:
        1. Find entity co-occurrences in text (within ±100 tokens)
        2. For each entity pair:
           a. Extract surrounding context
           b. Query LLM: "What relationship exists between X and Y?"
           c. Parse LLM response (relationship type + confidence)
        3. Filter by confidence threshold (0.6 for relationships)
        4. Deduplicate (keep highest confidence per unique triple)
        5. Return sorted by confidence

        Args:
            text: Original text where entities were found
            entities: List of entities (from extract_entities)

        Returns:
            List of Relationship objects sorted by confidence
            - Each has: source_entity, target_entity, relationship_type,
              confidence, evidence (text snippet)
            - Only relationships with confidence > 0.6

        Raises:
            NotImplementedError: If enable_llm_refinement=False (Phase 1)
            ValidationError: If text empty or entities list empty
            LLMError: If LLM query fails

        Performance:
            - O(n²) where n = number of entities
            - Mitigation: Only check co-occurring entities (reduces to ~O(n))
            - Target: 70%+ precision, 65%+ recall

        Relationship Types:
            - CREATED_BY: X was created by Y (Python CREATED_BY Guido van Rossum)
            - USES: X uses Y (Django USES Python)
            - IS_A: X is a type of Y (Python IS_A programming language)
            - PART_OF: X is part of Y (NumPy PART_OF Python ecosystem)
            - LOCATED_IN: X located in Y (CWI LOCATED_IN Amsterdam)
            - WORKS_FOR: X works for Y (Guido WORKS_FOR Google)

        Example:
            ```python
            extractor = EntityExtractor(
                spacy_model=nlp,
                ollama_client=ollama,
                enable_llm_refinement=True  # Required for relationships
            )

            text = "Python was created by Guido van Rossum at CWI"
            entities = extractor.extract_entities(text)
            relationships = extractor.extract_relationships(text, entities)

            # Returns:
            # [
            #   Relationship(
            #     source_entity="Python",
            #     target_entity="Guido van Rossum",
            #     relationship_type="CREATED_BY",
            #     confidence=0.92,
            #     evidence="Python was created by Guido van Rossum"
            #   ),
            #   Relationship(
            #     source_entity="Guido van Rossum",
            #     target_entity="CWI",
            #     relationship_type="WORKS_FOR",
            #     confidence=0.85,
            #     evidence="Guido van Rossum at CWI"
            #   )
            # ]
            ```
        """

    def normalize_entity(self, entity: str, entity_type: str) -> str:
        """
        Normalize entity name for consistent graph representation.

        Normalization rules:
        1. Strip whitespace
        2. Title case for PERSON, ORG, GPE
        3. Uppercase for abbreviations (e.g., "usa" → "USA")
        4. Remove articles ("the Python" → "Python")
        5. Standardize common variations ("Python lang" → "Python")

        Args:
            entity: Raw entity string from NER
            entity_type: Entity type (affects normalization rules)

        Returns:
            Normalized entity name

        Example:
            ```python
            extractor = EntityExtractor(spacy_model=nlp)

            # Person names
            extractor.normalize_entity("guido van rossum", "PERSON")
            # Returns: "Guido van Rossum"

            # Organizations
            extractor.normalize_entity("  openai  ", "ORG")
            # Returns: "OpenAI"

            # Technologies (remove articles)
            extractor.normalize_entity("the Python programming language", "TECHNOLOGY")
            # Returns: "Python"

            # Abbreviations
            extractor.normalize_entity("usa", "GPE")
            # Returns: "USA"
            ```
        """
```

---

## Dependencies

### Component Dependencies

**Internal (zapomni_core):**
- `OllamaClient` (from `zapomni_core.embeddings`) - For LLM refinement (Phase 2)
  - Used by: `_llm_refine()`, `extract_relationships()`
  - Purpose: Query LLM for entity validation and relationship detection

**Data Models (zapomni_core.models):**
- `Entity` - Entity data class (defined above)
- `Relationship` - Relationship data class (defined above)

### External Libraries

**Required (Phase 1):**
- `spacy>=3.7.0` - NER engine
  - Purpose: Fast entity extraction, linguistic processing
  - Component: `en_core_web_sm` language model
  - Install: `python -m spacy download en_core_web_sm`

**Optional (Phase 2):**
- `httpx>=0.25.0` - Used by OllamaClient for LLM API calls
  - Purpose: LLM-based entity refinement and relationship detection
  - Already included in zapomni_core dependencies

**Utilities:**
- `dataclasses` (stdlib) - For Entity and Relationship models
- `typing` (stdlib) - Type annotations
- `re` (stdlib) - Text normalization patterns

### Dependency Injection

Dependencies are injected via constructor:

```python
# Phase 1: SpaCy only (no LLM dependency)
import spacy
nlp = spacy.load("en_core_web_sm")
extractor = EntityExtractor(spacy_model=nlp)

# Phase 2: With LLM refinement
from zapomni_core.embeddings import OllamaClient
ollama = OllamaClient(host="http://localhost:11434")
extractor = EntityExtractor(
    spacy_model=nlp,
    ollama_client=ollama,
    enable_llm_refinement=True
)
```

**Rationale:**
- Loose coupling: Can swap SpaCy models (sm → md → lg)
- Testability: Easy to mock OllamaClient
- Phase-based: LLM optional (Phase 1 works without it)

---

## State Management

### Attributes

**Immutable Configuration (set at init):**
- `spacy_nlp: Language` - Loaded SpaCy model
  - Lifetime: Entire extractor instance
  - Thread-safe: Yes (SpaCy models are thread-safe for reading)
- `ollama_client: Optional[OllamaClient]` - LLM client
  - Lifetime: Entire instance
  - Phase 1: None, Phase 2: OllamaClient instance
- `enable_llm_refinement: bool` - Feature flag
  - Lifetime: Entire instance
  - Default: False (Phase 1)
- `confidence_threshold: float` - Filter threshold (0.7)
  - Lifetime: Entire instance
  - Tunable parameter
- `entity_types: Set[str]` - Supported entity types
  - Lifetime: Entire instance
  - Default: All supported types

**No Mutable State:**
- EntityExtractor is **stateless** between method calls
- Each `extract_entities()` call is independent
- No caching (caching handled by upper layers if needed)

### State Transitions

EntityExtractor has no state machine - it's a **pure transformation component**:

```
Input (text)
    ↓
extract_entities() - Stateless transformation
    ↓
Output (List[Entity])
```

No state persists between calls.

### Thread Safety

**Thread-Safe:** ✅ Yes

**Reasoning:**
- No mutable shared state
- SpaCy Language model is thread-safe for inference
- OllamaClient uses httpx (thread-safe async client)

**Usage in Concurrent Context:**
```python
# Safe to use same instance across multiple threads/tasks
extractor = EntityExtractor(spacy_model=nlp)

async def process_batch(texts: List[str]) -> List[List[Entity]]:
    tasks = [extract_entities_async(text) for text in texts]
    return await asyncio.gather(*tasks)

async def extract_entities_async(text: str):
    # Safe: no shared mutable state
    return extractor.extract_entities(text)
```

---

## Public Methods (Detailed)

### Method 1: `extract_entities`

**Signature:**
```python
def extract_entities(self, text: str) -> List[Entity]
```

**Purpose:** Extract and normalize entities from text using hybrid approach

**Parameters:**
- `text`: str
  - Description: Input text to extract entities from
  - Constraints:
    - Must not be empty (after strip)
    - Maximum length: 100,000 characters (~100KB)
    - Must be valid UTF-8
  - Example: "Python was created by Guido van Rossum"

**Returns:**
- Type: `List[Entity]`
- Description: Extracted entities sorted by confidence (descending)
- Guarantees:
  - All entities have confidence >= confidence_threshold
  - Deduplicated (same name+type appears once)
  - Normalized names (consistent formatting)
  - Non-empty if entities found, empty list if none

**Raises:**
- `ValidationError`: When text is empty or exceeds max length
- `ExtractionError`: When SpaCy processing fails
- `LLMError`: When LLM refinement enabled but fails (Phase 2)

**Preconditions:**
- ✅ EntityExtractor initialized with valid SpaCy model
- ✅ If enable_llm_refinement=True, ollama_client must be set

**Postconditions:**
- ✅ All returned entities have confidence >= threshold
- ✅ Entity names are normalized
- ✅ No duplicate name+type pairs
- ✅ Sorted by confidence descending

**Algorithm Outline:**
```
1. Validate input (non-empty, max length)
2. SpaCy NER pass:
   a. Process text with spacy_nlp
   b. Extract entities from doc.ents
   c. Map SpaCy labels to our entity types
   d. Assign initial confidence (from SpaCy scores)
3. Normalize entity names (for each entity)
4. Optional LLM refinement (if enabled):
   a. Send entities + text to LLM
   b. LLM validates and enhances entities
   c. Merge LLM results with SpaCy results
5. Deduplicate entities:
   a. Group by (name, type)
   b. Merge mentions count
   c. Keep highest confidence
6. Filter by confidence_threshold
7. Sort by confidence descending
8. Return List[Entity]
```

**Edge Cases:**

1. **Empty text** → ValidationError
   ```python
   extractor.extract_entities("")
   # Raises: ValidationError("Text cannot be empty")
   ```

2. **Text too long** → ValidationError
   ```python
   extractor.extract_entities("x" * 100_001)
   # Raises: ValidationError("Text exceeds max length (100,000)")
   ```

3. **No entities found** → Empty list
   ```python
   extractor.extract_entities("Lorem ipsum dolor sit amet")
   # Returns: []
   ```

4. **All entities below threshold** → Empty list
   ```python
   extractor.confidence_threshold = 0.95  # Very high
   extractor.extract_entities("Maybe Python?")
   # Returns: [] (if confidence < 0.95)
   ```

5. **Duplicate entities** → Merged
   ```python
   text = "Python is great. I love Python. Python rocks!"
   entities = extractor.extract_entities(text)
   # Returns: [Entity(name="Python", mentions=3, ...)]
   ```

6. **LLM refinement fails** → Fallback to SpaCy results
   ```python
   # If ollama_client fails, log warning and return SpaCy-only entities
   # Does not raise error (graceful degradation)
   ```

**Related Methods:**
- Calls: `normalize_entity()`, `_spacy_extract()`, `_llm_refine()` (if enabled), `_deduplicate_entities()`
- Called by: `MemoryProcessor.build_knowledge_graph()`

---

### Method 2: `extract_relationships`

**Signature:**
```python
def extract_relationships(
    self,
    text: str,
    entities: List[Entity]
) -> List[Relationship]
```

**Purpose:** Detect relationships between entities using LLM (Phase 2)

**Parameters:**

- `text`: str
  - Description: Original text where entities were found
  - Constraints: Same as extract_entities
  - Example: "Python was created by Guido van Rossum"

- `entities`: List[Entity]
  - Description: Entities from extract_entities()
  - Constraints:
    - Must not be empty
    - Must be from same text
    - Each entity must have valid name and type
  - Example: [Entity(name="Python", ...), Entity(name="Guido van Rossum", ...)]

**Returns:**
- Type: `List[Relationship]`
- Description: Detected relationships sorted by confidence
- Structure: Each Relationship has source, target, type, confidence, evidence
- Guarantees:
  - All relationships have confidence >= 0.6 (relationship threshold)
  - Deduplicated (same source-target-type appears once)
  - Only relationships between provided entities

**Raises:**
- `NotImplementedError`: If enable_llm_refinement=False (Phase 1)
  - Message: "Relationship extraction requires LLM refinement (Phase 2)"
- `ValidationError`: If text empty or entities list empty
- `LLMError`: If LLM query fails

**Preconditions:**
- ✅ enable_llm_refinement=True (Phase 2 only)
- ✅ ollama_client is set and functional
- ✅ entities list is non-empty

**Postconditions:**
- ✅ All returned relationships have confidence >= 0.6
- ✅ All source/target entities exist in input entities list
- ✅ No duplicate (source, target, type) triples
- ✅ Sorted by confidence descending

**Algorithm Outline:**
```
1. Validate inputs (text, entities)
2. If not Phase 2: raise NotImplementedError
3. Find entity co-occurrences:
   a. Tokenize text
   b. For each entity pair:
      - Check if both appear within ±100 tokens
      - If yes, add to candidates
4. For each candidate pair (e1, e2):
   a. Extract context (surrounding ±50 tokens)
   b. Query LLM:
      Prompt: "In this context: '{context}'
               What is the relationship between {e1.name} and {e2.name}?
               Respond with: relationship_type, confidence, explanation"
   c. Parse LLM response
   d. Create Relationship object
5. Filter by confidence >= 0.6
6. Deduplicate (keep highest confidence per triple)
7. Sort by confidence descending
8. Return List[Relationship]
```

**Edge Cases:**

1. **Empty entities list** → ValidationError
   ```python
   extractor.extract_relationships(text, [])
   # Raises: ValidationError("Entities list cannot be empty")
   ```

2. **Phase 1 mode** → NotImplementedError
   ```python
   extractor = EntityExtractor(nlp, enable_llm_refinement=False)
   extractor.extract_relationships(text, entities)
   # Raises: NotImplementedError("Requires Phase 2")
   ```

3. **No co-occurring entities** → Empty list
   ```python
   # If no entities appear near each other in text
   # Returns: []
   ```

4. **LLM finds no relationships** → Empty list
   ```python
   # If LLM determines no semantic relationships exist
   # Returns: []
   ```

5. **All relationships below threshold** → Empty list
   ```python
   # If all detected relationships have confidence < 0.6
   # Returns: []
   ```

6. **LLM timeout** → LLMError
   ```python
   # If LLM query exceeds timeout (30s)
   # Raises: LLMError("LLM query timeout")
   ```

**Related Methods:**
- Calls: `_detect_relationship_llm()` (for each entity pair)
- Called by: `MemoryProcessor.build_knowledge_graph()`
- Requires: `extract_entities()` to be called first

---

### Method 3: `normalize_entity`

**Signature:**
```python
def normalize_entity(self, entity: str, entity_type: str) -> str
```

**Purpose:** Normalize entity name for consistent graph representation

**Parameters:**

- `entity`: str
  - Description: Raw entity string from NER
  - Constraints: Must not be empty after strip
  - Examples: "  python  ", "the Python programming language", "usa"

- `entity_type`: str
  - Description: Entity type (affects normalization rules)
  - Constraints: Must be in supported entity types
  - Examples: "PERSON", "ORG", "TECHNOLOGY", "GPE"

**Returns:**
- Type: `str`
- Description: Normalized entity name
- Guarantees:
  - No leading/trailing whitespace
  - Consistent case (based on entity_type)
  - Articles removed for TECHNOLOGY, CONCEPT
  - Abbreviations uppercased for GPE

**Raises:**
- `ValidationError`: If entity is empty after strip
- `ValidationError`: If entity_type not supported

**Algorithm Outline:**
```
1. Strip whitespace
2. If empty after strip: raise ValidationError
3. Apply type-specific rules:
   - PERSON: Title case ("john doe" → "John Doe")
   - ORG: Title case ("openai" → "OpenAI")
   - GPE: Uppercase if abbreviation ("usa" → "USA"), else title case
   - TECHNOLOGY: Remove articles ("the Python" → "Python"), title case
   - CONCEPT: Remove articles, lowercase
   - Other: Title case
4. Remove redundant words:
   - "Python programming language" → "Python"
   - "OpenAI company" → "OpenAI"
5. Return normalized string
```

**Edge Cases:**

1. **Empty entity** → ValidationError
   ```python
   extractor.normalize_entity("", "PERSON")
   # Raises: ValidationError("Entity cannot be empty")
   ```

2. **Whitespace only** → ValidationError
   ```python
   extractor.normalize_entity("   ", "ORG")
   # Raises: ValidationError("Entity cannot be empty")
   ```

3. **Unknown entity type** → ValidationError
   ```python
   extractor.normalize_entity("Python", "UNKNOWN_TYPE")
   # Raises: ValidationError("Unsupported entity type: UNKNOWN_TYPE")
   ```

4. **Abbreviation detection** → Uppercase
   ```python
   extractor.normalize_entity("usa", "GPE")
   # Returns: "USA"
   extractor.normalize_entity("san francisco", "GPE")
   # Returns: "San Francisco"
   ```

5. **Article removal** → Clean name
   ```python
   extractor.normalize_entity("the Python programming language", "TECHNOLOGY")
   # Returns: "Python"
   ```

6. **Multiple spaces** → Single space
   ```python
   extractor.normalize_entity("Guido  van   Rossum", "PERSON")
   # Returns: "Guido van Rossum"
   ```

**Related Methods:**
- Called by: `extract_entities()` (for each entity)
- Pure function: No side effects, deterministic

---

## Error Handling

### Exceptions Defined

```python
class ExtractionError(ZapomniCoreError):
    """Raised when entity extraction fails."""
    pass

class LLMError(ZapomniCoreError):
    """Raised when LLM refinement fails."""
    pass
```

### Error Recovery

**SpaCy Processing Failure:**
- Retry: No (SpaCy errors are usually unrecoverable)
- Fallback: Raise ExtractionError with informative message
- Logging: Log full stack trace at DEBUG level

**LLM Refinement Failure (Phase 2):**
- Retry: 3 attempts with exponential backoff (1s, 2s, 4s)
- Fallback: Return SpaCy-only results, log warning
- Graceful degradation: System continues with lower precision

**LLM Timeout:**
- Retry: No (timeout is terminal)
- Fallback: Log error, return SpaCy results
- Message: "LLM refinement timed out, using SpaCy-only results"

**Invalid Input:**
- Retry: No (user error)
- Behavior: Raise ValidationError immediately
- Message: Clear, actionable (e.g., "Text cannot be empty")

### Error Propagation

**Bubbles Up:**
- `ValidationError` - Input validation failures
- `ExtractionError` - Unrecoverable SpaCy errors

**Handled Internally:**
- `LLMError` - Logged, fallback to SpaCy results
- Network errors from OllamaClient - Logged, graceful degradation

---

## Usage Examples

### Basic Usage (Phase 1: SpaCy Only)

```python
import spacy
from zapomni_core.extraction import EntityExtractor

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize extractor
extractor = EntityExtractor(
    spacy_model=nlp,
    confidence_threshold=0.7  # Filter low-confidence entities
)

# Extract entities
text = """
Python is a high-level programming language created by Guido van Rossum.
It was first released in 1991. Python is maintained by the Python Software
Foundation and is widely used at companies like Google and Dropbox.
"""

entities = extractor.extract_entities(text)

# Print results
for entity in entities:
    print(f"{entity.name} ({entity.type}): {entity.confidence:.2f}")

# Output:
# Python (TECHNOLOGY): 0.85
# Guido van Rossum (PERSON): 0.92
# Python Software Foundation (ORG): 0.88
# Google (ORG): 0.91
# Dropbox (ORG): 0.87
```

### Advanced Usage (Phase 2: With LLM Refinement)

```python
import spacy
from zapomni_core.extraction import EntityExtractor
from zapomni_core.embeddings import OllamaClient

# Load dependencies
nlp = spacy.load("en_core_web_sm")
ollama = OllamaClient(host="http://localhost:11434", model="llama3.2:3b")

# Initialize with LLM refinement
extractor = EntityExtractor(
    spacy_model=nlp,
    ollama_client=ollama,
    enable_llm_refinement=True,  # Phase 2 feature
    confidence_threshold=0.75
)

text = """
Django is a web framework for Python. It was created by Adrian Holovaty
and Simon Willison at the Lawrence Journal-World newspaper. Django follows
the model-view-template (MVT) architectural pattern.
"""

# Extract entities
entities = extractor.extract_entities(text)
print(f"Found {len(entities)} entities")

# Extract relationships (Phase 2)
relationships = extractor.extract_relationships(text, entities)

# Print relationships
for rel in relationships:
    print(f"{rel.source_entity} --[{rel.relationship_type}]--> {rel.target_entity}")
    print(f"  Confidence: {rel.confidence:.2f}")
    print(f"  Evidence: {rel.evidence}")

# Output:
# Django --[USES]--> Python
#   Confidence: 0.94
#   Evidence: "Django is a web framework for Python"
# Django --[CREATED_BY]--> Adrian Holovaty
#   Confidence: 0.89
#   Evidence: "created by Adrian Holovaty and Simon Willison"
# Django --[CREATED_BY]--> Simon Willison
#   Confidence: 0.89
#   Evidence: "created by Adrian Holovaty and Simon Willison"
```

### Normalization Example

```python
extractor = EntityExtractor(spacy_model=nlp)

# Person names
print(extractor.normalize_entity("guido van rossum", "PERSON"))
# Output: "Guido van Rossum"

# Organizations
print(extractor.normalize_entity("  openai  ", "ORG"))
# Output: "OpenAI"

# Technologies (article removal)
print(extractor.normalize_entity("the Python programming language", "TECHNOLOGY"))
# Output: "Python"

# Geopolitical (abbreviations)
print(extractor.normalize_entity("usa", "GPE"))
# Output: "USA"

print(extractor.normalize_entity("san francisco", "GPE"))
# Output: "San Francisco"
```

### Error Handling Example

```python
try:
    entities = extractor.extract_entities("")
except ValidationError as e:
    print(f"Validation error: {e}")
    # Output: "Validation error: Text cannot be empty"

try:
    huge_text = "x" * 200_000
    entities = extractor.extract_entities(huge_text)
except ValidationError as e:
    print(f"Validation error: {e}")
    # Output: "Validation error: Text exceeds max length (100,000)"

# Phase 1 trying to extract relationships
extractor_phase1 = EntityExtractor(spacy_model=nlp, enable_llm_refinement=False)
try:
    rels = extractor_phase1.extract_relationships(text, entities)
except NotImplementedError as e:
    print(f"Not available: {e}")
    # Output: "Not available: Relationship extraction requires LLM refinement (Phase 2)"
```

---

## Testing Approach

### Unit Tests Required

**Test Categories:**

1. **Initialization Tests:**
   - `test_init_success()` - Normal initialization
   - `test_init_without_ner_component_raises()` - Invalid SpaCy model
   - `test_init_invalid_confidence_threshold_raises()` - Threshold out of range
   - `test_init_llm_refinement_without_client_raises()` - Phase 2 config error

2. **Entity Extraction Tests:**
   - `test_extract_entities_success()` - Happy path
   - `test_extract_entities_empty_text_raises()` - Validation
   - `test_extract_entities_too_long_raises()` - Max length
   - `test_extract_entities_no_entities_found()` - Empty result
   - `test_extract_entities_deduplication()` - Merge duplicates
   - `test_extract_entities_confidence_filtering()` - Threshold filtering
   - `test_extract_entities_sorted_by_confidence()` - Output ordering

3. **Normalization Tests:**
   - `test_normalize_entity_person()` - Title case
   - `test_normalize_entity_org()` - Organization formatting
   - `test_normalize_entity_technology_removes_articles()` - Article removal
   - `test_normalize_entity_gpe_abbreviation()` - USA → USA
   - `test_normalize_entity_empty_raises()` - Edge case
   - `test_normalize_entity_whitespace_only_raises()` - Edge case

4. **Relationship Extraction Tests (Phase 2):**
   - `test_extract_relationships_success()` - Happy path
   - `test_extract_relationships_phase1_raises()` - NotImplementedError
   - `test_extract_relationships_empty_entities_raises()` - Validation
   - `test_extract_relationships_no_cooccurrence()` - Empty result
   - `test_extract_relationships_confidence_filtering()` - Threshold
   - `test_extract_relationships_deduplication()` - Unique triples

5. **Error Handling Tests:**
   - `test_spacy_processing_error()` - ExtractionError
   - `test_llm_refinement_failure_fallback()` - Graceful degradation
   - `test_llm_timeout_fallback()` - Timeout handling

### Mocking Strategy

**Mock SpaCy:**
```python
@pytest.fixture
def mock_spacy_nlp(mocker):
    """Mock SpaCy Language model."""
    nlp = mocker.Mock(spec=Language)

    # Mock doc with entities
    doc = mocker.Mock()
    ent1 = mocker.Mock(text="Python", label_="PRODUCT")
    ent2 = mocker.Mock(text="Guido van Rossum", label_="PERSON")
    doc.ents = [ent1, ent2]

    nlp.return_value = doc
    return nlp
```

**Mock OllamaClient:**
```python
@pytest.fixture
def mock_ollama_client(mocker):
    """Mock Ollama client for LLM calls."""
    client = mocker.Mock(spec=OllamaClient)

    # Mock LLM response for entity refinement
    client.generate.return_value = {
        "response": "Entity: Python, Type: TECHNOLOGY, Confidence: 0.95"
    }

    return client
```

**Example Test:**
```python
def test_extract_entities_success(mock_spacy_nlp):
    """Test successful entity extraction with SpaCy."""
    extractor = EntityExtractor(spacy_model=mock_spacy_nlp)

    entities = extractor.extract_entities("Python was created by Guido")

    assert len(entities) == 2
    assert entities[0].name in ["Python", "Guido van Rossum"]
    assert all(e.confidence >= 0.7 for e in entities)
    mock_spacy_nlp.assert_called_once()
```

### Integration Tests

**Test with Real SpaCy:**
```python
@pytest.mark.integration
def test_extract_entities_real_spacy():
    """Test with actual SpaCy model (requires en_core_web_sm)."""
    nlp = spacy.load("en_core_web_sm")
    extractor = EntityExtractor(spacy_model=nlp)

    text = "Python was created by Guido van Rossum in 1991"
    entities = extractor.extract_entities(text)

    # Verify expected entities found
    entity_names = [e.name for e in entities]
    assert "Python" in entity_names
    assert "Guido van Rossum" in entity_names
```

**Test with Real Ollama (Phase 2):**
```python
@pytest.mark.integration
@pytest.mark.phase2
def test_extract_relationships_real_ollama():
    """Test relationship extraction with real Ollama (requires running service)."""
    nlp = spacy.load("en_core_web_sm")
    ollama = OllamaClient(host="http://localhost:11434")

    extractor = EntityExtractor(
        spacy_model=nlp,
        ollama_client=ollama,
        enable_llm_refinement=True
    )

    text = "Python was created by Guido van Rossum"
    entities = extractor.extract_entities(text)
    relationships = extractor.extract_relationships(text, entities)

    # Verify expected relationship
    assert any(r.relationship_type == "CREATED_BY" for r in relationships)
```

---

## Performance Considerations

### Time Complexity

**extract_entities():**
- SpaCy processing: O(n) where n = text length
- Normalization: O(m) where m = number of entities
- LLM refinement (Phase 2): O(m × L) where L = LLM latency (~1-2s)
- Deduplication: O(m log m) (sorting)
- **Overall:** O(n + m log m) for Phase 1, O(n + m × L) for Phase 2

**extract_relationships():**
- Co-occurrence detection: O(n + m²) worst case, O(n + m × k) typical (k = avg entities per window)
- LLM queries: O(pairs × L) where pairs ≈ m × k
- **Overall:** O(m × k × L) - dominated by LLM calls

**normalize_entity():**
- O(1) - constant time string operations

### Space Complexity

**extract_entities():**
- SpaCy doc: O(n) - stores full parsed document
- Entity list: O(m) - stores m entities
- **Overall:** O(n + m)

**extract_relationships():**
- Context windows: O(m × k × window_size)
- Relationships: O(r) where r = number of relationships
- **Overall:** O(m × k)

### Optimization Opportunities

**1. Batch Processing (Future Enhancement):**
```python
def extract_entities_batch(self, texts: List[str]) -> List[List[Entity]]:
    """Process multiple texts in batch for efficiency."""
    # Use spacy.pipe() for 3-5x speedup
    docs = self.spacy_nlp.pipe(texts)
    return [self._extract_from_doc(doc) for doc in docs]
```

**2. Caching Normalized Entities:**
```python
# Cache normalized forms to avoid repeated computation
self._normalization_cache: Dict[Tuple[str, str], str] = {}
```

**3. Early Stopping for Relationships:**
```python
# Stop after finding N high-confidence relationships
max_relationships = 50  # Limit to prevent O(m²) explosion
```

**4. Parallel LLM Queries (Phase 2):**
```python
# Query LLM for multiple entity pairs in parallel
async def _llm_refine_parallel(self, pairs):
    tasks = [self._query_llm(pair) for pair in pairs]
    return await asyncio.gather(*tasks)
```

### Performance Targets

| Operation | Target | Notes |
|-----------|--------|-------|
| extract_entities (SpaCy only) | < 10ms | Per document |
| extract_entities (with LLM) | 1-2s | Phase 2, dominated by LLM |
| extract_relationships | 2-5s | Phase 2, multiple LLM calls |
| normalize_entity | < 1ms | Per entity |

**Trade-offs:**
- **Precision vs. Speed:** LLM refinement adds 100-200x latency but improves precision by 30%
- **Recall vs. Precision:** Lower confidence_threshold → higher recall but lower precision
- **Scalability:** Relationship extraction O(m²) limits to ~100 entities per doc

---

## References

### Module Spec
- [zapomni_core_module.md](../level1/zapomni_core_module.md) - Parent module specification
  - Section: Entity & Relationship Extraction (lines 119-123, 636-656)
  - Data Models: Entity, Relationship (lines 400-433)

### Related Components
- `OllamaClient` (embeddings/ollama_client.md) - LLM client for refinement
- `MemoryProcessor` (processing/memory_processor.md) - Consumer of this component
- `GraphBuilder` (graph/graph_builder.md) - Uses extracted entities to build graph

### External Documentation
- SpaCy NER: https://spacy.io/usage/linguistic-features#named-entities
- SpaCy Entity Types: https://spacy.io/api/annotation#named-entities
- Ollama API: https://github.com/ollama/ollama/blob/main/docs/api.md

### Research References
- "LLMs for Information Extraction" - Hybrid NER approaches
- "Knowledge Graph Construction from Text" - Entity normalization best practices
- MTEB Benchmark: Entity extraction metrics (precision/recall targets)

---

## Appendix: Entity Type Mapping

### SpaCy → Our Types

| SpaCy Label | Our Type | Examples |
|-------------|----------|----------|
| PERSON | PERSON | "Guido van Rossum", "Tim Peters" |
| ORG | ORG | "Google", "Python Software Foundation" |
| GPE | GPE | "USA", "Amsterdam", "California" |
| PRODUCT | TECHNOLOGY | "Python", "Django", "Docker" |
| EVENT | EVENT | "PyCon 2024", "DjangoCon" |
| DATE | DATE | "1991", "December 2024" |
| LOC | GPE | "San Francisco Bay Area" |
| FAC | ORG | "CWI" (facilities treated as orgs) |

**Custom Types (Phase 2 LLM Enhancement):**
- TECHNOLOGY - Programming languages, frameworks, tools
- CONCEPT - Abstract ideas (e.g., "machine learning")

**Unmapped (Ignored):**
- CARDINAL, ORDINAL, QUANTITY - Numbers (not entities)
- MONEY, PERCENT - Measurements (not core entities)
- LANGUAGE - Usually not entities themselves

---

**Document Status:** Draft v1.0
**Created:** 2025-11-23
**Author:** Goncharenko Anton aka alienxs2
**Copyright:** Copyright (c) 2025 Goncharenko Anton aka alienxs2
**License:** MIT License

**Ready for Verification:** ✅ Yes
