# OllamaEmbedder.embed_text - Function Specification

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
async def embed_text(
    self,
    text: str
) -> List[float]:
    """
    Generate 768-dimensional embedding for a single text using Ollama API.

    Transforms natural language text into a dense vector representation suitable
    for semantic similarity search. Uses Ollama's nomic-embed-text model locally
    (zero external API calls for privacy). Includes intelligent retry logic and
    fallback to sentence-transformers if Ollama is unavailable.

    This is the core method for single-text embedding generation. For batch
    processing of multiple texts, use embed_batch() which is more efficient.

    Args:
        text: Input text to generate embedding for. Must be non-empty, valid UTF-8,
              and contain at most 8192 tokens (approximately 32KB of text).
              Examples: "Python is a programming language", "User query about AI"

    Returns:
        List[float]: 768-dimensional embedding vector normalized to unit length.
        Each float represents a semantic feature dimension. Vector is compatible
        with cosine similarity calculations and FalkorDB vector index storage.

        Example return: [0.023, -0.145, 0.087, ..., 0.012] (768 floats total)

    Raises:
        ValidationError: When input text fails validation:
            - Text is empty string or whitespace-only
            - Text is not valid UTF-8 encoding
            - Text exceeds 8192 tokens (max for nomic-embed-text)

        EmbeddingError: When embedding generation fails:
            - Ollama API unreachable after max_retries attempts
            - Ollama model 'nomic-embed-text' not found (not pulled)
            - Ollama returns invalid embedding (wrong dimensions)
            - Both Ollama and fallback fail (if fallback enabled)

        TimeoutError: When request exceeds timeout duration:
            - Ollama API call takes longer than timeout (default: 30s)
            - Network latency issues causing delayed response

    Example:
        ```python
        embedder = OllamaEmbedder()

        # Generate embedding for user query
        query = "What are Python's key features?"
        embedding = await embedder.embed_text(query)

        # Verify dimensions
        assert len(embedding) == 768
        assert all(isinstance(x, float) for x in embedding)

        # Use for similarity search
        similar_chunks = await search_engine.find_similar(embedding)
        print(f"Found {len(similar_chunks)} similar chunks")
        ```
    """
```

---

## Purpose & Context

### What It Does

Generates a **768-dimensional semantic embedding vector** from input text by:

1. **Validating** input text (non-empty, UTF-8, token limit)
2. **Calling** Ollama API with nomic-embed-text model
3. **Retrying** on transient failures (up to 3 attempts with exponential backoff)
4. **Falling back** to sentence-transformers if Ollama unavailable
5. **Validating** embedding dimensions (must be exactly 768)
6. **Returning** normalized float vector for semantic search

The embedding captures semantic meaning - texts with similar meanings produce similar embeddings (high cosine similarity), enabling powerful semantic search capabilities.

### Why It Exists

**Business Reason:** Core functionality for Zapomni's semantic memory - transforms text into searchable vectors

**Technical Reason:** Required by MemoryProcessor to generate embeddings for text chunks before storing in FalkorDB

**Privacy Reason:** Uses local Ollama API (no cloud services) to preserve user privacy - all embeddings generated on-device

### When To Use

**Primary Use Cases:**
- Called by `embed_batch()` for each text in batch
- Called by `MemoryProcessor.add_memory()` for single memory items
- Called during semantic search query embedding
- Used in tests to verify embedding generation

**Workflow Position:**
```
User adds memory
    ↓
SemanticChunker.chunk_text()
    ↓
OllamaEmbedder.embed_text() ← YOU ARE HERE
    ↓
FalkorDBClient.store_chunk()
    ↓
Memory persisted
```

### When NOT To Use

**Use embed_batch() instead if:**
- Processing multiple texts (embed_batch is 10x+ faster for batches)
- Embedding list of chunks (batch is more efficient)

**Don't use if:**
- Text is pre-embedded (already have vector)
- Text is empty (validation will fail)
- Ollama unavailable AND fallback disabled (will raise error)

---

## Parameters (Detailed)

### text: str

**Type:** `str`

**Purpose:** Input text to transform into 768-dimensional embedding vector

**Constraints:**

1. **Must not be empty:**
   - `text.strip() != ""`
   - Whitespace-only strings treated as empty

2. **Must be valid UTF-8:**
   - Standard Python string encoding
   - No binary data, no invalid unicode sequences

3. **Maximum length: 8192 tokens**
   - Approximately 32,000 characters (average)
   - Token count computed using tiktoken (cl100k_base encoding)
   - Exceeding limit raises ValidationError

4. **Practical length: 100-5000 chars**
   - Optimal: 200-2000 chars (single paragraph to multiple paragraphs)
   - Too short (< 20 chars): May produce low-quality embeddings
   - Too long (> 8192 tokens): Will be rejected

**Validation Logic:**

```python
# Step 1: Check non-empty
if not text or not text.strip():
    raise ValidationError("Text cannot be empty or whitespace-only")

# Step 2: Check UTF-8 (Python strings are UTF-8 by default, but verify)
try:
    text.encode("utf-8")
except UnicodeEncodeError as e:
    raise ValidationError(f"Text must be valid UTF-8: {e}")

# Step 3: Count tokens
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
tokens = enc.encode(text)
if len(tokens) > 8192:
    raise ValidationError(
        f"Text exceeds max length: {len(tokens)} tokens > 8192 max. "
        f"Consider splitting into smaller chunks."
    )
```

**Examples:**

**Valid:**
- `"Python is a programming language"` (6 tokens, well within limit)
- `"User query: How do I implement async functions?"` (10 tokens)
- `"x" * 30000` (still < 8192 tokens after tokenization)
- Multi-paragraph text (up to ~32KB)

**Invalid:**
- `""` (empty) → `ValidationError("Text cannot be empty...")`
- `"   "` (whitespace only) → `ValidationError("Text cannot be empty...")`
- `"word " * 10000` (exceeds 8192 tokens) → `ValidationError("Text exceeds max length: 10000 tokens > 8192...")`
- Binary data `b"\x80\x81".decode("latin1")` (invalid UTF-8) → `ValidationError("Text must be valid UTF-8...")`

**Token Counting Details:**

```python
# Example token counts for common text lengths:
# "Python"                         → 1 token
# "Python is great"                → 3 tokens
# "Python is a programming lang."  → 6 tokens
# Average paragraph (500 chars)    → ~125 tokens
# Max length (8192 tokens)         → ~32,000 chars
```

---

## Return Value

**Type:** `List[float]`

**Description:** 768-dimensional embedding vector representing semantic meaning of input text

**Structure:**

```python
# Return value is a list of exactly 768 floats
embedding = [
    0.0234,    # Dimension 0
    -0.1456,   # Dimension 1
    0.0871,    # Dimension 2
    # ... 765 more dimensions ...
    0.0123     # Dimension 767
]
```

**Properties:**

1. **Length: Exactly 768**
   - Fixed dimensionality (required by nomic-embed-text model)
   - Compatible with FalkorDB vector index (configured for 768 dims)

2. **Type: All floats**
   - Each element is `float` type (Python `float` = 64-bit)
   - No NaN values allowed
   - No Inf/-Inf values allowed

3. **Normalized to unit length**
   - Vector magnitude (L2 norm) ≈ 1.0
   - Enables cosine similarity: `dot(v1, v2) = cos(theta)`
   - Formula: `sqrt(sum(x^2 for x in embedding)) ≈ 1.0`

4. **Value range: Typically [-1.0, 1.0]**
   - Most values between -0.5 and 0.5
   - Extreme values near -1 or 1 are rare but valid

**Validation:**

```python
def _validate_embedding(self, embedding: List[float]) -> None:
    """Validate embedding meets requirements."""

    # Check type
    if not isinstance(embedding, list):
        raise EmbeddingError(f"Expected list, got {type(embedding)}")

    # Check length
    if len(embedding) != 768:
        raise EmbeddingError(
            f"Invalid embedding dimensions: expected 768, got {len(embedding)}"
        )

    # Check all floats
    if not all(isinstance(x, float) for x in embedding):
        raise EmbeddingError("Embedding must contain only floats")

    # Check no NaN/Inf
    import math
    if any(math.isnan(x) or math.isinf(x) for x in embedding):
        raise EmbeddingError("Embedding contains NaN or Inf values")

    # Check normalization (optional, for verification)
    magnitude = math.sqrt(sum(x**2 for x in embedding))
    if not (0.9 <= magnitude <= 1.1):
        logger.warning("embedding_not_normalized", magnitude=magnitude)
```

**Success Case Example:**

```python
text = "Python is a programming language"
embedding = await embedder.embed_text(text)

# embedding = [0.023, -0.145, 0.087, ..., 0.012] (768 floats)
# len(embedding) == 768 ✅
# all(isinstance(x, float) for x in embedding) ✅
# sqrt(sum(x**2 for x in embedding)) ≈ 1.0 ✅
```

**Usage with Result:**

```python
# Store in FalkorDB
await falkor_client.store_chunk(
    chunk_id="chunk_123",
    text=text,
    embedding=embedding  # 768 floats used as vector
)

# Semantic search
query_embedding = await embedder.embed_text("Python tutorial")
# Cosine similarity = dot(query_embedding, embedding)
similarity = sum(a * b for a, b in zip(query_embedding, embedding))
# similarity ≈ 0.85 (high similarity, same topic)
```

---

## Exceptions

### ValidationError

**When Raised:**

1. **Empty text input:**
   ```python
   await embedder.embed_text("")
   # ValidationError: Text cannot be empty or whitespace-only
   ```

2. **Whitespace-only text:**
   ```python
   await embedder.embed_text("   \n\t  ")
   # ValidationError: Text cannot be empty or whitespace-only
   ```

3. **Text exceeds max length (8192 tokens):**
   ```python
   long_text = "word " * 10000  # ~10k tokens
   await embedder.embed_text(long_text)
   # ValidationError: Text exceeds max length: 10000 tokens > 8192 max
   ```

4. **Invalid UTF-8 encoding:**
   ```python
   invalid_text = b"\x80\x81".decode("latin1")  # Invalid UTF-8
   await embedder.embed_text(invalid_text)
   # ValidationError: Text must be valid UTF-8: ...
   ```

**Message Format:**
```python
raise ValidationError(f"Validation failed: {specific_reason}")
```

**Recovery Strategy:**
- Caller should fix input (remove whitespace, shorten text, fix encoding)
- Do NOT retry with same input (will fail again)
- Log error for debugging

**Example Handling:**
```python
try:
    embedding = await embedder.embed_text(user_input)
except ValidationError as e:
    logger.error("invalid_input", error=str(e), input_length=len(user_input))
    # Show user error message: "Text is too long, please shorten"
    return None
```

---

### EmbeddingError

**When Raised:**

1. **Ollama API unreachable (after 3 retries):**
   ```python
   # Scenario: Ollama service is down
   await embedder.embed_text("Python is great")
   # EmbeddingError: Failed to generate embedding: Connection refused (tried 3 times)
   ```

2. **Ollama model not found (404):**
   ```python
   # Scenario: User hasn't run `ollama pull nomic-embed-text`
   await embedder.embed_text("Python is great")
   # EmbeddingError: Model 'nomic-embed-text' not found. Run: ollama pull nomic-embed-text
   ```

3. **Invalid embedding returned by Ollama:**
   ```python
   # Scenario: Ollama bug returns wrong dimensions
   # Ollama returns 512 dims instead of 768
   # EmbeddingError: Invalid embedding dimensions: expected 768, got 512
   ```

4. **Both Ollama and fallback fail:**
   ```python
   # Scenario: Ollama down AND sentence-transformers fails
   embedder = OllamaEmbedder(enable_fallback=True)
   await embedder.embed_text("Python is great")
   # EmbeddingError: All embedding methods failed:
   #   Ollama: Connection refused
   #   Fallback: sentence-transformers not installed
   ```

**Message Format:**
```python
# Single failure
raise EmbeddingError(f"Failed to generate embedding: {specific_error}")

# Multiple failures
raise EmbeddingError(
    f"All embedding methods failed:\n"
    f"  Ollama: {ollama_error}\n"
    f"  Fallback: {fallback_error}"
)
```

**Recovery Strategy:**

**For transient errors (connection refused, timeout):**
- Caller should retry after delay (exponential backoff already done internally)
- Or queue for later processing
- Or skip and log error

**For persistent errors (model not found):**
- User must fix environment (run `ollama pull`)
- Do NOT retry until fixed

**Example Handling:**
```python
try:
    embedding = await embedder.embed_text(text)
except EmbeddingError as e:
    if "not found" in str(e).lower():
        # Permanent error - stop processing
        logger.critical("model_not_found", error=str(e))
        raise SystemExit("Please install Ollama model: ollama pull nomic-embed-text")
    else:
        # Transient error - retry later or skip
        logger.warning("embedding_failed_transient", error=str(e), text=text[:100])
        # Add to retry queue or skip this chunk
        return None
```

---

### TimeoutError

**When Raised:**

1. **Ollama API call exceeds timeout (default: 30s):**
   ```python
   # Scenario: Ollama is overloaded or network is slow
   embedder = OllamaEmbedder(timeout=10)  # 10s timeout
   await embedder.embed_text("very long text...")
   # TimeoutError: Embedding request timed out after 10 seconds
   ```

2. **Network latency issues:**
   ```python
   # Scenario: Ollama running on remote host with high latency
   embedder = OllamaEmbedder(base_url="http://192.168.1.100:11434", timeout=5)
   await embedder.embed_text("Python is great")
   # TimeoutError: Embedding request timed out after 5 seconds
   ```

**Message Format:**
```python
raise TimeoutError(f"Embedding request timed out after {timeout} seconds")
```

**Recovery Strategy:**

**For temporary overload:**
- Retry with longer timeout
- Reduce concurrent requests (lower batch_size)

**For persistent timeouts:**
- Increase timeout in configuration
- Check Ollama performance (CPU/GPU usage)
- Consider using faster model (if available)

**Example Handling:**
```python
try:
    embedding = await embedder.embed_text(text)
except TimeoutError as e:
    logger.warning("timeout_occurred", error=str(e), text_length=len(text))

    # Retry with 2x timeout
    embedder_slow = OllamaEmbedder(timeout=60)  # 60s
    try:
        embedding = await embedder_slow.embed_text(text)
    except TimeoutError:
        logger.error("timeout_persistent", text_length=len(text))
        # Give up or use fallback
        return None
```

---

## Algorithm (Pseudocode)

```
FUNCTION embed_text(text: str) -> List[float]:
    # ========================================
    # STEP 1: Input Validation
    # ========================================

    # 1.1: Check non-empty
    IF text is None OR text.strip() == "":
        RAISE ValidationError("Text cannot be empty or whitespace-only")

    # 1.2: Check UTF-8 encoding
    TRY:
        text.encode("utf-8")
    CATCH UnicodeEncodeError as e:
        RAISE ValidationError(f"Text must be valid UTF-8: {e}")

    # 1.3: Count tokens (tiktoken)
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)

    IF len(tokens) > 8192:
        RAISE ValidationError(
            f"Text exceeds max length: {len(tokens)} tokens > 8192 max"
        )

    # Log validation success
    logger.debug(
        "text_validated",
        text_length=len(text),
        token_count=len(tokens)
    )

    # ========================================
    # STEP 2: Try Ollama API (with retries)
    # ========================================

    ollama_error = None

    FOR retry_attempt in range(0, max_retries + 1):
        TRY:
            # Call Ollama API
            response = AWAIT http_client.post(
                url=f"{base_url}/api/embeddings",
                json={
                    "model": model_name,  # "nomic-embed-text"
                    "prompt": text
                },
                timeout=timeout  # Default: 30s
            )

            # Check response status
            IF response.status_code == 200:
                # Success - parse embedding
                data = response.json()
                embedding = data["embedding"]

                # Validate embedding
                CALL _validate_embedding(embedding)

                # Log success
                logger.info(
                    "embedding_generated",
                    model="ollama",
                    dimensions=len(embedding),
                    retry_attempt=retry_attempt
                )

                RETURN embedding  # SUCCESS EXIT

            ELIF response.status_code == 404:
                # Model not found - don't retry
                RAISE EmbeddingError(
                    f"Model '{model_name}' not found. "
                    f"Run: ollama pull {model_name}"
                )

            ELSE:
                # Other error - retry
                error_msg = response.text
                RAISE EmbeddingError(f"Ollama API error: {error_msg}")

        CATCH (ConnectionError, TimeoutError, EmbeddingError) as e:
            ollama_error = e

            # Log retry attempt
            logger.warning(
                "ollama_retry",
                error=str(e),
                retry_attempt=retry_attempt,
                max_retries=max_retries
            )

            # If not last attempt, wait and retry
            IF retry_attempt < max_retries:
                backoff_delay = 2 ** retry_attempt  # 1s, 2s, 4s
                AWAIT asyncio.sleep(backoff_delay)
                CONTINUE
            ELSE:
                # All retries exhausted
                BREAK

    # ========================================
    # STEP 3: Fallback to sentence-transformers
    # ========================================

    IF enable_fallback:
        logger.warning(
            "ollama_failed_using_fallback",
            ollama_error=str(ollama_error)
        )

        TRY:
            embedding = AWAIT _fallback_embed(text)

            # Validate embedding
            CALL _validate_embedding(embedding)

            # Log fallback success
            logger.info(
                "embedding_generated",
                model="sentence-transformers-fallback",
                dimensions=len(embedding)
            )

            RETURN embedding  # FALLBACK SUCCESS EXIT

        CATCH Exception as fallback_error:
            # Both Ollama and fallback failed
            logger.error(
                "all_embedding_methods_failed",
                ollama_error=str(ollama_error),
                fallback_error=str(fallback_error)
            )

            RAISE EmbeddingError(
                f"All embedding methods failed:\n"
                f"  Ollama: {ollama_error}\n"
                f"  Fallback: {fallback_error}"
            )

    # ========================================
    # STEP 4: No fallback - raise error
    # ========================================

    ELSE:
        logger.error(
            "ollama_failed_no_fallback",
            error=str(ollama_error)
        )

        RAISE EmbeddingError(
            f"Failed to generate embedding: {ollama_error}"
        )

END FUNCTION
```

---

## Preconditions

**Must be true BEFORE calling embed_text():**

1. **OllamaEmbedder instance initialized:**
   ```python
   embedder = OllamaEmbedder()  # __init__ completed successfully
   ```

2. **Dependencies available:**
   - `httpx` library installed
   - `tiktoken` library installed (for token counting)
   - `sentence-transformers` installed (if fallback enabled)

3. **Ollama service running (recommended but not required):**
   ```bash
   # Check Ollama is running
   curl http://localhost:11434/api/tags
   # Should return list of models
   ```

4. **Ollama model downloaded (recommended):**
   ```bash
   ollama pull nomic-embed-text
   ```

5. **Network connectivity (if Ollama remote):**
   - If `base_url` is remote host, network must be reachable
   - Firewall allows connections to Ollama port (default: 11434)

**Optional Preconditions:**

- Fallback enabled (`enable_fallback=True`) if Ollama may be unavailable
- Sufficient timeout configured for long texts or slow connections

---

## Postconditions

**Must be true AFTER successful embed_text() execution:**

1. **Embedding returned:**
   - Return value is `List[float]`
   - Length is exactly 768
   - All values are valid floats (no NaN/Inf)

2. **Embedding is normalized:**
   - Vector magnitude ≈ 1.0 (unit vector)
   - Suitable for cosine similarity calculations

3. **Logs generated:**
   - Success: `logger.info("embedding_generated", ...)`
   - Retries: `logger.warning("ollama_retry", ...)` (if retries occurred)
   - Fallback: `logger.warning("ollama_failed_using_fallback", ...)` (if fallback used)

4. **No side effects:**
   - Input text not modified
   - OllamaEmbedder state not changed (client reused)
   - No files created

5. **Deterministic (mostly):**
   - Same text → same embedding (within floating-point precision)
   - Minor variations possible due to Ollama model non-determinism

**If embed_text() raises exception:**

1. **Error logged:**
   - Appropriate log level (WARNING for retries, ERROR for failures)

2. **State unchanged:**
   - OllamaEmbedder instance still usable
   - Can retry same or different text

3. **Resources released:**
   - HTTP connections returned to pool
   - No memory leaks

---

## Edge Cases & Handling

### Edge Case 1: Empty Text

**Scenario:** User passes empty string `""`

**Input:**
```python
text = ""
```

**Expected Behavior:**
```python
raise ValidationError("Text cannot be empty or whitespace-only")
```

**Test Scenario:**
```python
@pytest.mark.asyncio
async def test_embed_text_empty_raises():
    """Empty text should raise ValidationError."""
    embedder = OllamaEmbedder()

    with pytest.raises(ValidationError, match="cannot be empty"):
        await embedder.embed_text("")
```

**Why This Edge Case:**
- Empty embeddings are meaningless
- Ollama API would reject anyway
- Fail fast at validation layer

---

### Edge Case 2: Whitespace-Only Text

**Scenario:** User passes text with only whitespace `"   \n\t  "`

**Input:**
```python
text = "   \n\t  "
```

**Expected Behavior:**
```python
raise ValidationError("Text cannot be empty or whitespace-only")
```

**Test Scenario:**
```python
@pytest.mark.asyncio
async def test_embed_text_whitespace_only_raises():
    """Whitespace-only text should raise ValidationError."""
    embedder = OllamaEmbedder()

    with pytest.raises(ValidationError, match="whitespace-only"):
        await embedder.embed_text("   \n\t  ")
```

**Why This Edge Case:**
- Whitespace has no semantic meaning
- Would produce near-zero embedding
- Better to reject explicitly

---

### Edge Case 3: Text Exceeds Max Length (8192 tokens)

**Scenario:** User passes very long text exceeding token limit

**Input:**
```python
text = "word " * 10000  # ~10,000 tokens
```

**Expected Behavior:**
```python
raise ValidationError(
    "Text exceeds max length: 10000 tokens > 8192 max. "
    "Consider splitting into smaller chunks."
)
```

**Test Scenario:**
```python
@pytest.mark.asyncio
async def test_embed_text_too_long_raises():
    """Text exceeding 8192 tokens should raise ValidationError."""
    embedder = OllamaEmbedder()

    # Create text with ~10k tokens
    long_text = "word " * 10000

    with pytest.raises(ValidationError, match="exceeds max length"):
        await embedder.embed_text(long_text)
```

**Why This Edge Case:**
- nomic-embed-text has hard limit of 8192 tokens
- Ollama would truncate or error
- Fail fast with clear error message

**User Action:**
- Split text into chunks using SemanticChunker
- Or manually split into smaller pieces

---

### Edge Case 4: Ollama Service Offline

**Scenario:** Ollama service is not running

**Setup:**
```bash
# Stop Ollama
sudo systemctl stop ollama
```

**Input:**
```python
text = "Python is great"
```

**Expected Behavior (fallback enabled):**
```python
# 1. Try Ollama 3 times → all fail
# 2. Log warning: "ollama_failed_using_fallback"
# 3. Use sentence-transformers fallback
# 4. Return 768-dim embedding (384 real + 384 zeros)
embedding = await embedder.embed_text(text)
assert len(embedding) == 768
```

**Expected Behavior (fallback disabled):**
```python
raise EmbeddingError(
    "Failed to generate embedding: Connection refused (tried 3 times)"
)
```

**Test Scenario:**
```python
@pytest.mark.asyncio
async def test_embed_text_ollama_offline_with_fallback(mocker):
    """Ollama offline should fallback to sentence-transformers."""
    # Mock Ollama connection error
    mocker.patch(
        "httpx.AsyncClient.post",
        side_effect=ConnectionError("Connection refused")
    )

    # Mock sentence-transformers fallback
    mock_st = mocker.MagicMock()
    mock_st.encode.return_value = [0.1] * 384
    mocker.patch(
        "sentence_transformers.SentenceTransformer",
        return_value=mock_st
    )

    embedder = OllamaEmbedder(enable_fallback=True, max_retries=1)
    embedding = await embedder.embed_text("test")

    assert len(embedding) == 768  # 384 + 384 padding
```

**Why This Edge Case:**
- Common scenario: Ollama not installed or stopped
- Fallback ensures system remains operational
- Logs help debug why fallback was used

---

### Edge Case 5: Ollama Model Not Downloaded

**Scenario:** User hasn't run `ollama pull nomic-embed-text`

**Input:**
```python
text = "Python is great"
```

**Expected Behavior:**
```python
# Ollama returns 404 Not Found
raise EmbeddingError(
    "Model 'nomic-embed-text' not found. "
    "Run: ollama pull nomic-embed-text"
)
```

**Test Scenario:**
```python
@pytest.mark.asyncio
async def test_embed_text_model_not_found_raises(mocker):
    """Missing Ollama model should raise EmbeddingError with helpful message."""
    # Mock 404 response
    mock_response = mocker.MagicMock()
    mock_response.status_code = 404
    mock_response.text = "model 'nomic-embed-text' not found"

    mocker.patch(
        "httpx.AsyncClient.post",
        return_value=mock_response
    )

    embedder = OllamaEmbedder()

    with pytest.raises(EmbeddingError, match="not found.*ollama pull"):
        await embedder.embed_text("test")
```

**Why This Edge Case:**
- Common first-time user mistake
- Error message provides exact fix
- Don't retry (won't help until model pulled)

---

### Edge Case 6: Ollama Returns Invalid Embedding (Wrong Dimensions)

**Scenario:** Ollama bug returns embedding with wrong dimensions

**Input:**
```python
text = "Python is great"
```

**Ollama Response (buggy):**
```json
{
  "embedding": [0.1, 0.2, ..., 0.5]  // Only 512 floats instead of 768
}
```

**Expected Behavior:**
```python
raise EmbeddingError(
    "Invalid embedding dimensions: expected 768, got 512"
)
```

**Test Scenario:**
```python
@pytest.mark.asyncio
async def test_embed_text_invalid_dimensions_raises(mocker):
    """Embedding with wrong dimensions should raise EmbeddingError."""
    # Mock response with 512 dims instead of 768
    mock_response = mocker.MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "embedding": [0.1] * 512  # Wrong!
    }

    mocker.patch(
        "httpx.AsyncClient.post",
        return_value=mock_response
    )

    embedder = OllamaEmbedder()

    with pytest.raises(EmbeddingError, match="expected 768, got 512"):
        await embedder.embed_text("test")
```

**Why This Edge Case:**
- Catches Ollama bugs or wrong model
- Prevents invalid data from entering system
- FalkorDB vector index expects exactly 768 dims

---

### Edge Case 7: Request Timeout

**Scenario:** Ollama request takes longer than timeout

**Input:**
```python
text = "Python is great"
embedder = OllamaEmbedder(timeout=1)  # 1 second timeout
```

**Ollama Response:**
```
# Takes 5 seconds to respond (too slow)
```

**Expected Behavior:**
```python
# After 1 second:
raise TimeoutError("Embedding request timed out after 1 seconds")
```

**Test Scenario:**
```python
@pytest.mark.asyncio
async def test_embed_text_timeout_raises(mocker):
    """Request exceeding timeout should raise TimeoutError."""
    # Mock slow response
    async def slow_post(*args, **kwargs):
        await asyncio.sleep(5)  # 5 seconds
        return mocker.MagicMock()

    mocker.patch(
        "httpx.AsyncClient.post",
        side_effect=slow_post
    )

    embedder = OllamaEmbedder(timeout=1)  # 1s timeout

    with pytest.raises(TimeoutError, match="timed out after 1"):
        await embedder.embed_text("test")
```

**Why This Edge Case:**
- Prevents hanging on slow Ollama
- User can adjust timeout if needed
- Timeout error is retriable (may succeed later)

---

### Edge Case 8: Non-UTF-8 Text

**Scenario:** User passes text with invalid UTF-8 encoding

**Input:**
```python
# Invalid UTF-8 bytes decoded as latin1
text = b"\x80\x81".decode("latin1")
```

**Expected Behavior:**
```python
raise ValidationError("Text must be valid UTF-8: ...")
```

**Test Scenario:**
```python
@pytest.mark.asyncio
async def test_embed_text_invalid_utf8_raises():
    """Non-UTF-8 text should raise ValidationError."""
    embedder = OllamaEmbedder()

    # Create invalid UTF-8 string
    invalid_text = b"\x80\x81".decode("latin1")

    with pytest.raises(ValidationError, match="valid UTF-8"):
        await embedder.embed_text(invalid_text)
```

**Why This Edge Case:**
- Ollama expects UTF-8
- Prevents encoding errors downstream
- Rare but possible with binary data

---

### Edge Case 9: Embedding Contains NaN or Inf

**Scenario:** Ollama returns embedding with invalid float values

**Input:**
```python
text = "Python is great"
```

**Ollama Response (buggy):**
```json
{
  "embedding": [0.1, NaN, 0.3, Infinity, ..., 0.5]
}
```

**Expected Behavior:**
```python
raise EmbeddingError("Embedding contains NaN or Inf values")
```

**Test Scenario:**
```python
@pytest.mark.asyncio
async def test_embed_text_nan_values_raises(mocker):
    """Embedding with NaN/Inf should raise EmbeddingError."""
    import math

    # Mock response with NaN
    mock_response = mocker.MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "embedding": [0.1, math.nan, 0.3] + [0.1] * 765
    }

    mocker.patch(
        "httpx.AsyncClient.post",
        return_value=mock_response
    )

    embedder = OllamaEmbedder()

    with pytest.raises(EmbeddingError, match="NaN or Inf"):
        await embedder.embed_text("test")
```

**Why This Edge Case:**
- NaN/Inf breaks vector operations
- FalkorDB can't handle invalid floats
- Catches numerical instability bugs

---

### Edge Case 10: Concurrent Calls (Thread Safety)

**Scenario:** Multiple asyncio tasks call embed_text() simultaneously

**Input:**
```python
texts = ["Python", "Ollama", "Embedding"] * 10
tasks = [embedder.embed_text(text) for text in texts]
results = await asyncio.gather(*tasks)
```

**Expected Behavior:**
```python
# All calls succeed independently
# Results are correct and in order
assert len(results) == 30
assert all(len(emb) == 768 for emb in results)
```

**Test Scenario:**
```python
@pytest.mark.asyncio
async def test_embed_text_concurrent_safe(mocker):
    """Concurrent embed_text calls should not interfere."""
    # Mock Ollama responses
    mock_response = mocker.MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "embedding": [0.1] * 768
    }

    mocker.patch(
        "httpx.AsyncClient.post",
        return_value=mock_response
    )

    embedder = OllamaEmbedder()

    # Launch 10 concurrent calls
    tasks = [embedder.embed_text(f"text {i}") for i in range(10)]
    results = await asyncio.gather(*tasks)

    assert len(results) == 10
    assert all(len(emb) == 768 for emb in results)
```

**Why This Edge Case:**
- Common use case: batch processing
- Verifies httpx.AsyncClient thread safety
- Ensures no race conditions

---

## Test Scenarios (Complete List)

### Happy Path Tests

**1. test_embed_text_success_normal_text**
- **Input:** `text = "Python is a programming language"`
- **Mock:** Ollama returns valid 768-dim embedding
- **Expected:** Returns `List[float]` with 768 elements, all floats
- **Assertions:**
  ```python
  assert len(embedding) == 768
  assert all(isinstance(x, float) for x in embedding)
  assert all(-1.0 <= x <= 1.0 for x in embedding)
  ```

**2. test_embed_text_success_long_text**
- **Input:** `text = "word " * 8000` (near max length, ~8000 tokens)
- **Mock:** Ollama returns valid embedding
- **Expected:** Success, embedding returned
- **Assertions:** Same as test 1

**3. test_embed_text_success_minimal_text**
- **Input:** `text = "x"` (single character)
- **Mock:** Ollama returns valid embedding
- **Expected:** Success (even though very short)
- **Assertions:** Same as test 1

---

### Error/Validation Tests

**4. test_embed_text_empty_text_raises**
- **Input:** `text = ""`
- **Expected:** `ValidationError("Text cannot be empty...")`
- **Mock:** None (fails before API call)

**5. test_embed_text_whitespace_only_raises**
- **Input:** `text = "   \n\t  "`
- **Expected:** `ValidationError("...whitespace-only")`

**6. test_embed_text_exceeds_max_length_raises**
- **Input:** `text = "word " * 10000` (>8192 tokens)
- **Expected:** `ValidationError("Text exceeds max length: 10000 tokens > 8192...")`

**7. test_embed_text_invalid_utf8_raises**
- **Input:** Invalid UTF-8 string
- **Expected:** `ValidationError("Text must be valid UTF-8...")`

---

### Retry & Fallback Tests

**8. test_embed_text_retry_on_connection_error**
- **Setup:** Mock Ollama raises `ConnectionError` twice, succeeds on 3rd try
- **Input:** Normal text
- **Expected:** 3 API calls made, success on 3rd, embedding returned
- **Assertions:**
  ```python
  assert mock_post.call_count == 3
  assert len(embedding) == 768
  ```

**9. test_embed_text_retry_exhausted_uses_fallback**
- **Setup:** Mock Ollama fails 3 times, fallback succeeds
- **Input:** Normal text
- **Expected:** 3 Ollama calls, then fallback, embedding returned
- **Assertions:**
  ```python
  assert ollama_mock.call_count == 3
  assert fallback_mock.called
  assert len(embedding) == 768
  ```

**10. test_embed_text_fallback_disabled_raises_on_failure**
- **Setup:** `enable_fallback=False`, Ollama fails
- **Input:** Normal text
- **Expected:** `EmbeddingError("Failed to generate embedding...")`

**11. test_embed_text_both_ollama_and_fallback_fail**
- **Setup:** Ollama fails, fallback also fails
- **Input:** Normal text
- **Expected:** `EmbeddingError("All embedding methods failed:\n  Ollama: ...\n  Fallback: ...")`

---

### Ollama API Error Tests

**12. test_embed_text_model_not_found_raises**
- **Setup:** Mock Ollama returns 404 status
- **Input:** Normal text
- **Expected:** `EmbeddingError("Model 'nomic-embed-text' not found. Run: ollama pull...")`

**13. test_embed_text_invalid_dimensions_raises**
- **Setup:** Mock Ollama returns embedding with 512 dims instead of 768
- **Input:** Normal text
- **Expected:** `EmbeddingError("Invalid embedding dimensions: expected 768, got 512")`

**14. test_embed_text_embedding_contains_nan_raises**
- **Setup:** Mock Ollama returns embedding with NaN value
- **Input:** Normal text
- **Expected:** `EmbeddingError("Embedding contains NaN or Inf values")`

**15. test_embed_text_timeout_raises**
- **Setup:** Mock Ollama takes 5 seconds, timeout=1s
- **Input:** Normal text
- **Expected:** `TimeoutError("Embedding request timed out after 1 seconds")`

---

### Performance & Concurrency Tests

**16. test_embed_text_performance_within_sla**
- **Setup:** Mock Ollama (instant response)
- **Input:** Normal text
- **Expected:** Execution time < 150ms (P95 target)
- **Assertions:**
  ```python
  start = time.time()
  embedding = await embedder.embed_text(text)
  duration_ms = (time.time() - start) * 1000
  assert duration_ms < 150
  ```

**17. test_embed_text_concurrent_calls_safe**
- **Setup:** Launch 10 concurrent embed_text() calls
- **Input:** 10 different texts
- **Expected:** All succeed, no interference
- **Assertions:**
  ```python
  results = await asyncio.gather(*tasks)
  assert len(results) == 10
  assert all(len(emb) == 768 for emb in results)
  ```

---

### Integration Tests (Require Real Ollama)

**18. test_embed_text_real_ollama_integration**
- **Setup:** Real Ollama running, model downloaded
- **Input:** Normal text
- **Expected:** Real embedding generated
- **Assertions:**
  ```python
  @pytest.mark.integration
  @pytest.mark.requires_ollama
  async def test():
      embedder = OllamaEmbedder()
      if not await embedder.health_check():
          pytest.skip("Ollama not available")

      embedding = await embedder.embed_text("Python is great")
      assert len(embedding) == 768
      assert all(isinstance(x, float) for x in embedding)
  ```

**19. test_embed_text_semantic_similarity**
- **Setup:** Real Ollama
- **Input:** Two similar texts, one dissimilar
- **Expected:** Similar texts have high cosine similarity
- **Assertions:**
  ```python
  emb1 = await embedder.embed_text("Python programming")
  emb2 = await embedder.embed_text("Python coding")
  emb3 = await embedder.embed_text("Banana recipe")

  similarity_12 = cosine_similarity(emb1, emb2)
  similarity_13 = cosine_similarity(emb1, emb3)

  assert similarity_12 > 0.8  # High similarity
  assert similarity_13 < 0.5  # Low similarity
  ```

**20. test_embed_text_deterministic**
- **Setup:** Real Ollama
- **Input:** Same text twice
- **Expected:** Identical embeddings (or very close)
- **Assertions:**
  ```python
  emb1 = await embedder.embed_text("Python is great")
  emb2 = await embedder.embed_text("Python is great")

  # Should be identical or within floating-point precision
  assert all(abs(a - b) < 1e-6 for a, b in zip(emb1, emb2))
  ```

---

## Performance Requirements

### Latency

**Single Text Embedding:**

| Text Length | Target Latency (P95) | Notes |
|-------------|---------------------|-------|
| < 100 chars | < 50ms | Very short text |
| 100-1000 chars | < 100ms | Normal paragraph |
| 1000-5000 chars | < 150ms | Long text |
| > 5000 chars | < 300ms | Very long (near limit) |

**Measurement:**
```python
import time

start = time.time()
embedding = await embedder.embed_text(text)
latency_ms = (time.time() - start) * 1000

assert latency_ms < 150  # P95 target
```

### Throughput

**Concurrent Requests:**
- Support: Up to 100 embed_text() calls/second
- Actual throughput depends on Ollama performance
- Batch processing (embed_batch) more efficient for multiple texts

### Resource Usage

**Memory:**
- Single embedding: ~3KB (768 floats * 4 bytes)
- httpx.AsyncClient overhead: ~1MB
- Total: < 10MB for typical usage

**CPU:**
- Minimal (mostly I/O-bound)
- CPU used by Ollama (on server side)

**Network:**
- Upload: ~few KB per request (text + JSON overhead)
- Download: ~3KB per response (embedding)

---

## Security Considerations

### Input Validation

**Protection Against:**

1. **Injection Attacks:**
   - Text is sent as JSON payload to Ollama
   - httpx handles JSON encoding (prevents injection)
   - No SQL/command injection risk (Ollama is local API)

2. **Resource Exhaustion:**
   - Max text length enforced (8192 tokens)
   - Prevents extremely long texts from consuming resources
   - Timeout prevents hanging on slow responses

3. **Invalid Encodings:**
   - UTF-8 validation prevents encoding attacks
   - Malformed unicode rejected early

**Validation Checklist:**
- ✅ Text length checked (prevents DoS)
- ✅ UTF-8 encoding verified
- ✅ Timeout enforced (prevents hanging)
- ✅ No sensitive data logged (only text length logged, not content)

### Data Protection

**Privacy:**
- All embeddings generated locally (Ollama on localhost or LAN)
- Zero data sent to external cloud services
- User text never leaves local environment

**Sensitive Data:**
- Text content is NOT logged (only metadata: length, token count)
- Embeddings are NOT considered sensitive (mathematical vectors)
- No PII in logs

**Error Messages:**
- Safe error messages (don't leak sensitive data)
- Don't log full text in error logs (only length/truncated)

**Example Safe Logging:**
```python
# ✅ SAFE - only metadata
logger.error("embedding_failed", text_length=len(text), error=str(e))

# ❌ UNSAFE - logs full text
logger.error("embedding_failed", text=text, error=str(e))
```

---

## Related Functions

### Calls (Dependencies)

**Internal Methods:**

1. **`_call_ollama(text: str, retry_count: int) -> List[float]`**
   - Purpose: Make actual HTTP request to Ollama API
   - Called by: `embed_text()` (with retry loop)
   - Raises: `EmbeddingError`, `TimeoutError`

2. **`_fallback_embed(text: str) -> List[float]`**
   - Purpose: Generate embedding using sentence-transformers fallback
   - Called by: `embed_text()` (if Ollama fails and fallback enabled)
   - Returns: 768-dim embedding (384 real + 384 zeros)

3. **`_validate_embedding(embedding: List[float]) -> None`**
   - Purpose: Validate embedding dimensions and values
   - Called by: `embed_text()` (after getting embedding)
   - Raises: `EmbeddingError` if invalid

**External Libraries:**

4. **`tiktoken.get_encoding("cl100k_base").encode(text)`**
   - Purpose: Count tokens in text
   - Used for: Max length validation (8192 token limit)

5. **`httpx.AsyncClient.post()`**
   - Purpose: Make async HTTP request to Ollama
   - Used for: API call to `/api/embeddings` endpoint

### Called By (Callers)

**Public Methods:**

1. **`embed_batch(texts: List[str], batch_size: int) -> List[List[float]]`**
   - Calls: `embed_text()` multiple times concurrently
   - Use case: Efficient batch processing

**External Components:**

2. **`MemoryProcessor.add_memory(content: str, metadata: dict)`**
   - Calls: `embed_text()` or `embed_batch()` for chunks
   - Use case: Generate embeddings during memory ingestion

3. **`SearchEngine.search(query: str)`**
   - Calls: `embed_text(query)` to embed search query
   - Use case: Semantic search

**Call Graph:**
```
User → MemoryProcessor.add_memory()
         ↓
       SemanticChunker.chunk_text()
         ↓
       OllamaEmbedder.embed_batch()  [high-level API]
         ↓
       OllamaEmbedder.embed_text()   [YOU ARE HERE]
         ↓
       _call_ollama() → Ollama API
         ↓ (if fails)
       _fallback_embed() → sentence-transformers
```

---

## Implementation Notes

### Libraries Used

**Primary:**
- `httpx>=0.25.0` - Async HTTP client for Ollama API calls
- `tiktoken>=0.5.0` - Token counting (OpenAI tokenizer, used by nomic-embed-text)
- `structlog>=23.2.0` - Structured logging

**Fallback:**
- `sentence-transformers>=2.2.0` - Fallback embedding model (all-MiniLM-L6-v2)

**Standard Library:**
- `asyncio` - Async/await support
- `math` - NaN/Inf checks, vector magnitude calculation

### Known Limitations

**1. Token Limit:**
- Hard limit: 8192 tokens (nomic-embed-text constraint)
- Longer texts must be chunked manually
- No automatic chunking in this method

**2. Determinism:**
- Embeddings are mostly deterministic (same text → same embedding)
- Minor floating-point variations possible
- Different Ollama versions may produce slightly different embeddings

**3. Performance:**
- Latency depends on Ollama performance (CPU/GPU)
- Network latency if Ollama on remote host
- No caching in this method (Phase 2 feature)

**4. Fallback Quality:**
- sentence-transformers produces 384-dim embeddings (padded to 768)
- Lower quality than nomic-embed-text (but better than nothing)
- Fallback embeddings NOT compatible with nomic-embed-text embeddings (different semantic spaces)

**5. Single Text Only:**
- This method processes one text at a time
- For batches, use `embed_batch()` (more efficient)

### Future Enhancements

**Phase 2 (Semantic Cache):**
- Cache embeddings by text hash (SHA256)
- LRU eviction, 24h TTL
- Target: 60%+ cache hit rate

**Phase 3 (Advanced):**
- Support alternative embedding models (Llama, BERT)
- Dynamic batch sizing based on Ollama performance
- GPU acceleration detection and optimization
- Streaming embeddings for very long texts

**Potential Optimizations:**
- Connection pooling (httpx already does this)
- Reduce retry delays for faster fallback
- Parallel Ollama instances for higher throughput

---

## References

### Component Spec
- [OllamaEmbedder Component Specification](/home/dev/zapomni/.spec-workflow/specs/level2/ollama_embedder_component.md) - Parent component

### Module Spec
- [zapomni_core Module Specification](/home/dev/zapomni/.spec-workflow/specs/level1/zapomni_core_module.md) - Parent module

### Related Function Specs
- (Future) `ollama_embedder_embed_batch.md` - Batch embedding method
- (Future) `semantic_chunker_chunk_text.md` - Produces chunks for embedding

### External Documentation
- **Ollama Embeddings API:** https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
- **nomic-embed-text Model:** https://huggingface.co/nomic-ai/nomic-embed-text-v1
- **tiktoken:** https://github.com/openai/tiktoken
- **httpx Async:** https://www.python-httpx.org/async/
- **sentence-transformers:** https://www.sbert.net/docs/pretrained_models.html

---

**Document Status:** Draft v1.0
**Created:** 2025-11-23
**Author:** Goncharenko Anton aka alienxs2
**Copyright:** Copyright (c) 2025 Goncharenko Anton aka alienxs2
**License:** MIT License

**Specification Completeness:**
- ✅ Function signature defined
- ✅ All parameters detailed (1 param: text)
- ✅ Return value fully specified
- ✅ All exceptions documented (3 types)
- ✅ Algorithm in pseudocode (complete)
- ✅ Edge cases identified (10 cases)
- ✅ Test scenarios defined (20 tests)
- ✅ Performance requirements specified
- ✅ Security considerations addressed

**Metrics:**
- Parameters: 1 (text: str)
- Return: List[float] (768 dimensions)
- Exceptions: 3 (ValidationError, EmbeddingError, TimeoutError)
- Edge Cases: 10
- Test Scenarios: 20
- Performance Targets: < 150ms P95

**Ready for Implementation:** ✅ YES
