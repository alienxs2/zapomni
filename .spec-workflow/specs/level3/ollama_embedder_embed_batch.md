# OllamaEmbedder.embed_batch() - Function Specification

**Level:** 3 (Function)
**Component:** OllamaEmbedder
**Module:** zapomni_core
**Author:** Goncharenko Anton aka alienxs2
**Status:** Draft
**Version:** 1.0
**Created:** 2025-11-23

---

## Function Signature

```python
async def embed_batch(
    self,
    texts: List[str],
    batch_size: int = 32
) -> List[List[float]]:
    """
    Generate embeddings for multiple texts in parallel batches.

    Efficiently processes large lists of texts by splitting into batches
    and processing batches concurrently via asyncio.gather(). This provides
    significant performance improvements over sequential embed_text() calls.

    Performance improvements:
    - Sequential: 100 texts × 100ms = 10,000ms (10 seconds)
    - Batch (32): 4 batches × 100ms = 400ms (0.4 seconds)
    - Speedup: 25x faster

    The method implements intelligent batching with retry logic per batch,
    fallback to sentence-transformers if Ollama fails, and comprehensive
    validation of all embeddings.

    Args:
        texts: List of text strings to embed (1 to 10,000 texts)
            - Each text: 1 to 8192 tokens (nomic-embed-text limit)
            - All texts validated before processing
            - Empty strings automatically filtered out

        batch_size: Number of texts to process concurrently (default: 32)
            - Recommended: 16-64 for optimal throughput
            - Higher values: Faster but more memory
            - Lower values: Slower but safer
            - Must be: 1 ≤ batch_size ≤ 128

    Returns:
        List[List[float]]: List of 768-dimensional embeddings, one per input text
            - Order preserved: embeddings[i] corresponds to texts[i]
            - Each embedding: 768 floats (nomic-embed-text dimension)
            - All embeddings validated for correct dimensionality

    Raises:
        ValidationError: If any text is invalid (empty, too long, non-UTF-8)
        EmbeddingError: If all batches fail (both Ollama and fallback)
        ValueError: If batch_size out of valid range (1-128)

    Example:
        >>> embedder = OllamaEmbedder()
        >>> texts = [
        ...     "Python is a programming language",
        ...     "Ollama runs AI models locally",
        ...     "Machine learning is fascinating"
        ... ]
        >>> embeddings = await embedder.embed_batch(texts, batch_size=32)
        >>> len(embeddings)
        3
        >>> len(embeddings[0])
        768
        >>> # All embeddings have same dimension
        >>> all(len(emb) == 768 for emb in embeddings)
        True

    Thread Safety:
        Async-safe. Multiple coroutines can call this concurrently.
        Internal batching ensures controlled concurrency.

    Performance:
        - 100 texts (batch_size=32): ~400ms total
        - 1000 texts (batch_size=32): ~3.5s total
        - Throughput: ~250-300 texts/second
    """
```

---

## Purpose & Context

### What It Does

The `embed_batch()` method efficiently generates embeddings for **multiple texts in parallel** by:

1. **Validating** all input texts (length, encoding, non-empty)
2. **Splitting** texts into batches of size `batch_size`
3. **Processing** batches concurrently using `asyncio.gather()`
4. **Retrying** failed batches with exponential backoff
5. **Falling back** to sentence-transformers if Ollama fails
6. **Validating** all output embeddings (768 dimensions)
7. **Preserving** input order in output

This method is **critical for performance** when processing many text chunks.

### Why It Exists

**Performance Requirement:**
- Processing 100 chunks sequentially: 10 seconds (too slow)
- Processing in batches of 32: 0.4 seconds (25x faster)
- Required for add_memory operation to meet < 2s target

**Efficiency:**
- HTTP connection pooling across batch
- Ollama can process multiple requests concurrently
- Async I/O prevents blocking

### When To Use

**Called By:**
- `MemoryProcessor.add_memory()` - Embed all chunks of a memory
- `MemoryProcessor.search_memory()` - Pre-compute embeddings for batch queries
- `SemanticChunker` - Re-embed merged chunks

**Use When:**
- You have 2+ texts to embed
- Performance matters (batch is 10-25x faster than sequential)
- Texts are independent (no ordering dependencies)

### When NOT To Use

**Don't use this if:**
- You have only 1 text → use `embed_text()` directly
- You need streaming results → batch waits for all
- Texts are arriving incrementally → buffer and batch when full

---

## Parameters (Detailed)

### texts: List[str]

**Type:** `List[str]`

**Purpose:**
List of text strings to generate embeddings for. Each text is independently embedded.

**Constraints:**

1. **List Length:**
   - Minimum: 1 text
   - Maximum: 10,000 texts (practical limit)
   - Recommended: 10-1000 texts per call

2. **Individual Text Constraints:**
   - Type: `str` (UTF-8 encoded)
   - Minimum length: 1 character (after stripping)
   - Maximum length: 8192 tokens (~32,000 characters for English)
   - Cannot be only whitespace
   - Must be valid UTF-8

3. **Memory Constraints:**
   - Total memory: ~10MB per 1000 texts (embeddings)
   - Input memory: ~1MB per 1000 texts (text storage)
   - Peak memory: ~2x during processing

**Validation Logic:**
```python
# Step 1: Validate list
if not isinstance(texts, list):
    raise ValidationError("texts must be a list")

if len(texts) == 0:
    raise ValidationError("texts list cannot be empty")

if len(texts) > 10000:
    raise ValidationError("texts list too large (max 10,000)")

# Step 2: Validate each text
validated_texts = []
for i, text in enumerate(texts):
    if not isinstance(text, str):
        raise ValidationError(f"texts[{i}] must be string, got {type(text)}")

    # Strip whitespace
    text_stripped = text.strip()

    if not text_stripped:
        raise ValidationError(f"texts[{i}] is empty or whitespace-only")

    if len(text_stripped) > 32000:  # ~8192 tokens
        raise ValidationError(f"texts[{i}] exceeds max length (32,000 chars)")

    # Check UTF-8 encoding
    try:
        text_stripped.encode('utf-8')
    except UnicodeEncodeError:
        raise ValidationError(f"texts[{i}] contains invalid UTF-8")

    validated_texts.append(text_stripped)

return validated_texts
```

**Examples:**

**Valid - Small Batch:**
```python
texts = [
    "Python is great",
    "Ollama runs locally",
    "Privacy matters"
]
embeddings = await embedder.embed_batch(texts)
# Returns: [[...768 dims...], [...768 dims...], [...768 dims...]]
```

**Valid - Large Batch:**
```python
texts = ["Text number " + str(i) for i in range(500)]
embeddings = await embedder.embed_batch(texts, batch_size=64)
# Returns: 500 embeddings, each 768 dimensions
```

**Valid - Mixed Lengths:**
```python
texts = [
    "Short",
    "This is a medium length sentence with more words.",
    "x" * 10000  # Very long text
]
embeddings = await embedder.embed_batch(texts)
# All valid, different lengths processed
```

**Invalid - Empty List:**
```python
texts = []
embeddings = await embedder.embed_batch(texts)
# Raises: ValidationError("texts list cannot be empty")
```

**Invalid - Contains Empty String:**
```python
texts = ["Valid text", "", "Another valid"]
embeddings = await embedder.embed_batch(texts)
# Raises: ValidationError("texts[1] is empty or whitespace-only")
```

**Invalid - Too Many Texts:**
```python
texts = ["Text"] * 10001
embeddings = await embedder.embed_batch(texts)
# Raises: ValidationError("texts list too large (max 10,000)")
```

**Invalid - Non-String Element:**
```python
texts = ["Valid", 123, "Valid"]
embeddings = await embedder.embed_batch(texts)
# Raises: ValidationError("texts[1] must be string, got <class 'int'>")
```

---

### batch_size: int

**Type:** `int`

**Purpose:**
Number of texts to process concurrently in each batch. Controls parallelism vs. memory tradeoff.

**Default:** `32` (recommended for most use cases)

**Constraints:**
- Minimum: 1 (sequential processing)
- Maximum: 128 (safety limit to prevent overwhelming Ollama)
- Recommended: 16-64

**Performance Considerations:**

| batch_size | Throughput | Memory | Ollama Load |
|------------|------------|--------|-------------|
| 1          | Slowest    | Lowest | Minimal     |
| 16         | Moderate   | Low    | Light       |
| 32         | Fast       | Medium | Moderate    |
| 64         | Fastest    | High   | Heavy       |
| 128        | Fastest*   | Highest| Max         |

\* Diminishing returns above 64

**Validation:**
```python
if not isinstance(batch_size, int):
    raise ValueError("batch_size must be int")

if batch_size < 1:
    raise ValueError("batch_size must be >= 1")

if batch_size > 128:
    raise ValueError("batch_size must be <= 128 (safety limit)")
```

**Examples:**

**Default (Recommended):**
```python
embeddings = await embedder.embed_batch(texts)
# Uses batch_size=32
```

**Conservative (Low Memory):**
```python
embeddings = await embedder.embed_batch(texts, batch_size=16)
# Slower but safer for resource-constrained environments
```

**Aggressive (High Throughput):**
```python
embeddings = await embedder.embed_batch(texts, batch_size=64)
# Faster for powerful machines with good Ollama performance
```

**Sequential (Debugging):**
```python
embeddings = await embedder.embed_batch(texts, batch_size=1)
# Processes one at a time (useful for debugging)
```

**Invalid - Zero:**
```python
embeddings = await embedder.embed_batch(texts, batch_size=0)
# Raises: ValueError("batch_size must be >= 1")
```

**Invalid - Too Large:**
```python
embeddings = await embedder.embed_batch(texts, batch_size=256)
# Raises: ValueError("batch_size must be <= 128 (safety limit)")
```

---

## Return Value

**Type:** `List[List[float]]`

**Purpose:**
List of embeddings, one per input text, in the same order as input.

**Structure:**
```python
[
    [0.12, -0.45, 0.78, ...],  # Embedding for texts[0] (768 floats)
    [0.34, -0.21, 0.56, ...],  # Embedding for texts[1] (768 floats)
    [0.89, -0.67, 0.23, ...],  # Embedding for texts[2] (768 floats)
    ...
]
```

**Guarantees:**
1. **Length:** `len(result) == len(texts)`
2. **Order:** `result[i]` is embedding for `texts[i]`
3. **Dimension:** Each `len(result[i]) == 768`
4. **Type:** Each `result[i][j]` is `float`
5. **Completeness:** No None values (all texts embedded or error raised)

**Example:**
```python
texts = ["Python", "Ollama", "AI"]
embeddings = await embedder.embed_batch(texts)

assert len(embeddings) == 3
assert len(embeddings[0]) == 768
assert all(isinstance(val, float) for val in embeddings[0])
assert embeddings[0] != embeddings[1]  # Different texts = different embeddings
```

---

## Exceptions

### ValidationError

**When Raised:**
- `texts` is empty list or not a list
- Any text in `texts` is empty, too long, or invalid UTF-8
- Any text is not a string type

**Message Formats:**
```python
"texts must be a list"
"texts list cannot be empty"
"texts list too large (max 10,000)"
"texts[{i}] must be string, got {type}"
"texts[{i}] is empty or whitespace-only"
"texts[{i}] exceeds max length (32,000 chars)"
"texts[{i}] contains invalid UTF-8"
```

**Recovery:** Fix input texts and retry

### ValueError

**When Raised:**
- `batch_size` is < 1 or > 128
- `batch_size` is not an integer

**Message Formats:**
```python
"batch_size must be int"
"batch_size must be >= 1"
"batch_size must be <= 128 (safety limit)"
```

**Recovery:** Use valid batch_size value

### EmbeddingError

**When Raised:**
- All batches failed to generate embeddings
- Both Ollama and sentence-transformers fallback failed
- Network errors persist after retries

**Message Formats:**
```python
"Failed to generate embeddings for batch after 3 retries"
"Ollama unavailable and fallback disabled"
"All embedding attempts failed"
```

**Recovery:** Check Ollama service, retry after delay, or enable fallback

---

## Algorithm (Pseudocode)

```
FUNCTION embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
    # Step 1: Validate inputs
    validated_texts = self._validate_texts(texts)
    self._validate_batch_size(batch_size)

    # Step 2: Split texts into batches
    batches = []
    FOR i IN RANGE(0, len(validated_texts), batch_size):
        batch = validated_texts[i:i+batch_size]
        batches.APPEND(batch)

    # Log batching info
    self._logger.info(
        "batch_embedding_started",
        total_texts=len(validated_texts),
        num_batches=len(batches),
        batch_size=batch_size
    )

    # Step 3: Process batches concurrently
    TRY:
        # Create list of async tasks (one per batch)
        tasks = [
            self._embed_single_batch(batch, batch_index)
            FOR batch_index, batch IN ENUMERATE(batches)
        ]

        # Execute all batches concurrently
        batch_results = AWAIT asyncio.gather(*tasks)

        # Step 4: Flatten results (list of lists -> single list)
        all_embeddings = []
        FOR batch_result IN batch_results:
            all_embeddings.EXTEND(batch_result)

        # Step 5: Validate all embeddings
        FOR i, embedding IN ENUMERATE(all_embeddings):
            IF len(embedding) != 768:
                RAISE EmbeddingError(f"Embedding {i} has wrong dimension: {len(embedding)}")

            IF NOT all(isinstance(val, float) FOR val IN embedding):
                RAISE EmbeddingError(f"Embedding {i} contains non-float values")

        # Step 6: Log success
        self._logger.info(
            "batch_embedding_completed",
            total_embeddings=len(all_embeddings)
        )

        RETURN all_embeddings

    CATCH Exception as e:
        self._logger.error(
            "batch_embedding_failed",
            error=str(e),
            exc_info=True
        )
        RAISE EmbeddingError(f"Batch embedding failed: {e}")

END FUNCTION


ASYNC FUNCTION _embed_single_batch(self, batch: List[str], batch_index: int) -> List[List[float]]:
    """Process a single batch with retry logic."""

    retries = 0
    max_retries = 3

    WHILE retries < max_retries:
        TRY:
            # Attempt to embed all texts in batch concurrently
            embedding_tasks = [
                self.embed_text(text)
                FOR text IN batch
            ]

            embeddings = AWAIT asyncio.gather(*embedding_tasks)
            RETURN embeddings

        CATCH (httpx.TimeoutError, httpx.ConnectError) as e:
            # Transient error - retry with backoff
            retries += 1
            IF retries < max_retries:
                backoff_seconds = 2 ** retries  # Exponential backoff
                self._logger.warning(
                    "batch_retry",
                    batch_index=batch_index,
                    retry=retries,
                    backoff=backoff_seconds
                )
                AWAIT asyncio.sleep(backoff_seconds)
            ELSE:
                # Max retries exceeded
                RAISE EmbeddingError(f"Batch {batch_index} failed after {max_retries} retries")

        CATCH Exception as unexpected:
            # Unexpected error - fail immediately
            self._logger.error(
                "batch_unexpected_error",
                batch_index=batch_index,
                error=str(unexpected),
                exc_info=True
            )
            RAISE
END FUNCTION
```

---

## Preconditions

✅ **Embedder Initialized:**
- `OllamaEmbedder.__init__()` called
- HTTP client initialized
- Ollama base URL configured

✅ **Ollama Available (or fallback enabled):**
- Ollama service running at configured URL
- OR `enable_fallback=True` for sentence-transformers

### Not Required

❌ **Ollama Model Pre-loaded:**
- Model lazy-loaded on first request

---

## Postconditions

### On Success

✅ **Embeddings Generated:**
- One embedding per input text
- All embeddings 768-dimensional
- Order preserved from input

✅ **All Validated:**
- Dimension checks passed
- Type checks passed
- Completeness verified

### On Error

❌ **No Partial Results:**
- Either all texts embedded or exception raised
- No incomplete embedding lists returned

---

## Edge Cases & Handling

### Edge Case 1: Single Text in List

**Scenario:** List has only one text

**Input:**
```python
texts = ["Single text to embed"]
embeddings = await embedder.embed_batch(texts)
```

**Processing:**
1. Validate: 1 text is valid (>= 1)
2. Create 1 batch of size 1
3. Process batch
4. Return list with 1 embedding

**Expected:**
```python
assert len(embeddings) == 1
assert len(embeddings[0]) == 768
```

**Test:**
```python
async def test_embed_batch_single_text():
    embedder = OllamaEmbedder()
    embeddings = await embedder.embed_batch(["Test"])
    assert len(embeddings) == 1
    assert len(embeddings[0]) == 768
```

---

### Edge Case 2: Texts Count Not Divisible By batch_size

**Scenario:** 100 texts, batch_size=32

**Processing:**
1. Create batches: [32, 32, 32, 4]
2. Last batch has only 4 texts (not 32)
3. All batches processed

**Expected:**
```python
texts = ["Text"] * 100
embeddings = await embedder.embed_batch(texts, batch_size=32)
assert len(embeddings) == 100  # All included
```

**Test:**
```python
async def test_embed_batch_uneven_division():
    embedder = OllamaEmbedder()
    texts = [f"Text {i}" for i in range(100)]
    embeddings = await embedder.embed_batch(texts, batch_size=32)
    assert len(embeddings) == 100
```

---

### Edge Case 3: One Batch Fails, Others Succeed

**Scenario:** Batch 2 of 4 fails transiently, retries succeed

**Processing:**
1. Process batches 1, 2, 3, 4 concurrently
2. Batch 2 times out (transient)
3. Batch 2 retries with backoff
4. Retry succeeds
5. All batches complete

**Expected:** All embeddings returned (no error)

**Test:**
```python
async def test_embed_batch_retry_success(mocker):
    embedder = OllamaEmbedder()

    # Mock to fail once, then succeed
    call_count = 0
    async def mock_embed(text):
        nonlocal call_count
        call_count += 1
        if call_count == 2:  # Fail on second call
            raise httpx.TimeoutError("Transient")
        return [0.1] * 768

    mocker.patch.object(embedder, 'embed_text', side_effect=mock_embed)

    texts = ["Text1", "Text2", "Text3"]
    embeddings = await embedder.embed_batch(texts)

    assert len(embeddings) == 3  # All succeeded after retry
```

---

### Edge Case 4: All Batches Fail

**Scenario:** Ollama completely down, all batches fail

**Processing:**
1. Each batch retries 3 times
2. All retries fail
3. Raise EmbeddingError

**Expected:**
```python
# Raises: EmbeddingError("Batch 0 failed after 3 retries")
```

**Test:**
```python
async def test_embed_batch_all_fail(mocker):
    embedder = OllamaEmbedder(enable_fallback=False)

    async def mock_fail(text):
        raise httpx.ConnectError("Ollama down")

    mocker.patch.object(embedder, 'embed_text', side_effect=mock_fail)

    texts = ["Text1", "Text2"]

    with pytest.raises(EmbeddingError, match="failed after 3 retries"):
        await embedder.embed_batch(texts)
```

---

### Edge Case 5: Very Large Batch (1000+ Texts)

**Scenario:** 5000 texts

**Processing:**
1. Split into ~156 batches (batch_size=32)
2. Process in parallel (asyncio.gather)
3. May take 15-20 seconds total
4. Memory usage: ~50MB

**Expected:** Completes successfully, all 5000 embeddings returned

**Test:**
```python
async def test_embed_batch_large():
    embedder = OllamaEmbedder()
    texts = [f"Document {i}" for i in range(5000)]

    embeddings = await embedder.embed_batch(texts, batch_size=32)

    assert len(embeddings) == 5000
    assert all(len(emb) == 768 for emb in embeddings)
```

---

### Edge Case 6: Mixed Text Lengths

**Scenario:** Some very short, some very long texts

**Processing:**
1. All texts validated individually
2. Different lengths processed equally
3. Embedding quality may vary (Ollama model behavior)

**Expected:** All embeddings generated, different dimensions

**Test:**
```python
async def test_embed_batch_mixed_lengths():
    embedder = OllamaEmbedder()
    texts = [
        "Hi",
        "This is a medium sentence.",
        "x" * 10000  # Very long
    ]

    embeddings = await embedder.embed_batch(texts)

    assert len(embeddings) == 3
    assert all(len(emb) == 768 for emb in embeddings)
    # Embeddings are different despite different input lengths
    assert embeddings[0] != embeddings[1] != embeddings[2]
```

---

## Test Scenarios (Complete List)

### Happy Path Tests

**1. test_embed_batch_small**
- Input: 3 texts, default batch_size
- Expected: 3 embeddings, 768 dims each

**2. test_embed_batch_medium**
- Input: 100 texts, batch_size=32
- Expected: 100 embeddings

**3. test_embed_batch_custom_batch_size**
- Input: 50 texts, batch_size=16
- Expected: 50 embeddings

**4. test_embed_batch_preserves_order**
- Input: Texts with unique identifiers
- Expected: Output order matches input

**5. test_embed_batch_different_lengths**
- Input: Mixed short/long texts
- Expected: All embedded correctly

---

### Edge Case Tests

**6. test_embed_batch_single_text**
- Edge case 1

**7. test_embed_batch_uneven_division**
- Edge case 2

**8. test_embed_batch_retry_success**
- Edge case 3

**9. test_embed_batch_all_fail**
- Edge case 4

**10. test_embed_batch_large**
- Edge case 5

**11. test_embed_batch_mixed_lengths**
- Edge case 6

---

### Validation Tests

**12. test_embed_batch_empty_list**
- Input: []
- Expected: ValidationError

**13. test_embed_batch_contains_empty_string**
- Input: ["Valid", ""]
- Expected: ValidationError

**14. test_embed_batch_non_string**
- Input: ["Valid", 123]
- Expected: ValidationError

**15. test_embed_batch_too_many_texts**
- Input: 10,001 texts
- Expected: ValidationError

**16. test_embed_batch_invalid_batch_size_zero**
- Input: batch_size=0
- Expected: ValueError

**17. test_embed_batch_invalid_batch_size_too_large**
- Input: batch_size=256
- Expected: ValueError

---

### Performance Tests

**18. test_embed_batch_faster_than_sequential**
- Verify: batch is significantly faster than N × embed_text()

**19. test_embed_batch_throughput**
- Measure: texts/second for different batch sizes

---

## Performance Requirements

### Latency Targets

- **100 texts:** < 500ms (batch_size=32)
- **1000 texts:** < 4s (batch_size=32)
- **Throughput:** 250-300 texts/second

### Speedup vs Sequential

- **Target:** 10-25x faster than sequential
- **Actual:** Depends on batch_size and Ollama performance

---

## Security Considerations

✅ **Input Validation:** All texts validated
✅ **No Injection:** Texts passed to Ollama API safely
✅ **Resource Limits:** batch_size capped at 128

---

## Related Functions

### Calls

**1. `self.embed_text(text)`**
- Purpose: Embed individual text
- When: For each text in batch

**2. `asyncio.gather(*tasks)`**
- Purpose: Run batches concurrently

**3. `self._logger.info/warning/error()`**
- Purpose: Log batch progress

### Called By

**1. `MemoryProcessor.add_memory()`**
- Purpose: Embed all chunks
- When: Adding new memory

---

## Implementation Notes

### Dependencies

- `asyncio` - Concurrency
- `httpx` - HTTP client
- `structlog` - Logging

### Known Limitations

**1. Memory Usage:**
- Large batches (1000+) use significant memory
- Mitigation: Use smaller batch_size

**2. Ollama Rate Limiting:**
- Too many concurrent requests may overwhelm Ollama
- Mitigation: Tune batch_size to Ollama capacity

---

## References

### Component Spec
- [OllamaEmbedder Component Specification](../level2/ollama_embedder_component.md)

### Related Functions
- `OllamaEmbedder.embed_text()` (Level 3)

---

## Document Status

**Version:** 1.0
**Created:** 2025-11-23
**Author:** Goncharenko Anton aka alienxs2
**License:** MIT License
**Status:** Draft

**Estimated Implementation:** 2-3 hours
**Lines of Code:** ~100 lines
**Test Coverage Target:** 95%+
**Test File:** `tests/unit/core/test_ollama_embedder_embed_batch.py`
