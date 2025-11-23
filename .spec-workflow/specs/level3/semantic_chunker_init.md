# SemanticChunker.__init__() - Function Specification

**Level:** 3 (Function)
**Component:** SemanticChunker
**Module:** zapomni_core
**Author:** Goncharenko Anton aka alienxs2
**Status:** Draft
**Version:** 1.0
**Created:** 2025-11-23

---

## Function Signature

```python
def __init__(
    self,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    min_chunk_size: int = 100,
    separators: Optional[List[str]] = None
) -> None:
    """
    Initialize SemanticChunker with chunking configuration.

    Args:
        chunk_size: Target chunk size in tokens (default: 512)
            - Constraints: 100 <= chunk_size <= 2048
            - Typical: 256-512 for dense embeddings, 1024+ for sparse

        chunk_overlap: Number of overlapping tokens between chunks (default: 50)
            - Constraints: 0 <= chunk_overlap < chunk_size
            - Typical: 10-20% of chunk_size for context preservation

        min_chunk_size: Minimum chunk size in tokens (default: 100)
            - Constraints: 50 <= min_chunk_size <= chunk_size
            - Purpose: Prevent tiny chunks (low information content)

        separators: Custom sentence separators (default: None uses ["\n\n", "\n", ". ", "! ", "? "])
            - Constraints: List of strings, non-empty
            - Purpose: Define semantic boundaries

    Raises:
        ValueError: If chunk_size < 100 or > 2048
        ValueError: If chunk_overlap < 0 or >= chunk_size
        ValueError: If min_chunk_size < 50 or > chunk_size
        ValueError: If separators is empty list

    Example:
        >>> # Default configuration
        >>> chunker = SemanticChunker()

        >>> # Custom configuration
        >>> chunker = SemanticChunker(
        ...     chunk_size=256,
        ...     chunk_overlap=25,
        ...     min_chunk_size=50
        ... )
    """
```

---

## Purpose & Context

### What It Does

Initializes SemanticChunker with validated chunking parameters. Creates the RecursiveCharacterTextSplitter instance from LangChain with configured settings.

### Why It Exists

Proper initialization ensures:
- Chunk sizes are appropriate for embedding models (512 tokens ~= 768-dim optimal)
- Overlap preserves context across chunks
- Minimum size prevents low-quality micro-chunks
- Semantic boundaries respected (sentence/paragraph breaks)

### When To Use

Called once when creating a chunker instance at application startup or in tests.

---

## Edge Cases

1. **chunk_size = 99** → ValueError("chunk_size must be >= 100")
2. **chunk_size = 2049** → ValueError("chunk_size must be <= 2048")
3. **chunk_overlap = chunk_size** → ValueError("chunk_overlap must be < chunk_size")
4. **min_chunk_size > chunk_size** → ValueError("min_chunk_size must be <= chunk_size")
5. **separators = []** → ValueError("separators cannot be empty")
6. **chunk_overlap = 0** → Valid (no overlap, but allowed)

---

## Test Scenarios

1. **test_init_defaults_success** - Default params work
2. **test_init_custom_params_success** - Custom params accepted
3. **test_init_chunk_size_too_small_raises** - chunk_size < 100
4. **test_init_chunk_size_too_large_raises** - chunk_size > 2048
5. **test_init_chunk_overlap_negative_raises** - chunk_overlap < 0
6. **test_init_chunk_overlap_too_large_raises** - chunk_overlap >= chunk_size
7. **test_init_min_chunk_size_too_small_raises** - min_chunk_size < 50
8. **test_init_min_chunk_size_too_large_raises** - min_chunk_size > chunk_size
9. **test_init_empty_separators_raises** - separators = []
10. **test_init_creates_text_splitter** - self.splitter is not None

---

## Algorithm (Pseudocode)

```
1. Validate chunk_size: 100 <= value <= 2048
2. Validate chunk_overlap: 0 <= value < chunk_size
3. Validate min_chunk_size: 50 <= value <= chunk_size
4. Validate separators: not empty if provided
5. Store all parameters as instance attributes
6. Create default separators if None: ["\n\n", "\n", ". ", "! ", "? "]
7. Initialize RecursiveCharacterTextSplitter with params
8. Log initialization success
```

---

**References:**
- [semantic_chunker_component.md](../level2/semantic_chunker_component.md)
- LangChain RecursiveCharacterTextSplitter: https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/recursive_text_splitter

---

**License:** MIT
