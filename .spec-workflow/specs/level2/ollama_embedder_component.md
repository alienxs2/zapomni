# OllamaEmbedder - Component Specification

**Level:** 2 (Component)
**Module:** zapomni_core
**Author:** Goncharenko Anton aka alienxs2
**Status:** Draft
**Version:** 1.0
**Created:** 2025-11-23

---

## Overview

### Purpose

OllamaEmbedder is responsible for **generating semantic embeddings from text chunks using local Ollama API**. It provides the critical transformation from natural language text to dense vector representations (768-dimensional embeddings) that enable semantic search in Zapomni.

This component embodies Zapomni's **privacy-first philosophy** - all embeddings are generated locally via Ollama, with zero data sent to external cloud APIs. It includes intelligent fallback to sentence-transformers if Ollama is unavailable, ensuring system reliability.

### Responsibilities

1. **Embedding Generation:** Generate 768-dimensional embeddings via Ollama API (nomic-embed-text model)
2. **Batch Processing:** Efficiently process multiple texts in parallel (up to 32 concurrent)
3. **Error Handling:** Retry on transient failures (3 attempts with exponential backoff)
4. **Fallback Strategy:** Seamlessly fallback to sentence-transformers if Ollama unavailable
5. **Performance Optimization:** Async I/O for non-blocking API calls
6. **Dimension Validation:** Ensure embeddings have correct dimensionality (768)

### Position in Module

OllamaEmbedder sits in the middle of the core processing pipeline:

```
SemanticChunker
    ↓ produces List[Chunk]
OllamaEmbedder (THIS)  ← Receives chunks
    ↓ produces List[List[float]]
EntityExtractor
    ↓
FalkorDBClient
```

**Key Relationships:**
- **Used by:** MemoryProcessor (calls embed_text() and embed_batch())
- **Depends on:** Ollama API (httpx client), sentence-transformers (fallback)
- **Produces:** List[List[float]] - embeddings consumed by FalkorDBClient for vector storage

---

## Class Definition

### Class Diagram

```
┌─────────────────────────────────────┐
│        OllamaEmbedder               │
├─────────────────────────────────────┤
│ - base_url: str                     │
│ - model_name: str                   │
│ - dimensions: int                   │
│ - timeout: int                      │
│ - max_retries: int                  │
│ - client: httpx.AsyncClient         │
│ - fallback_model: SentenceTransformer│
├─────────────────────────────────────┤
│ + __init__(base_url, model_name)   │
│ + embed_text(text) -> List[float]  │
│ + embed_batch(texts) -> List[List[float]] │
│ + get_dimensions() -> int           │
│ + health_check() -> bool            │
│ - _call_ollama(text) -> List[float]│
│ - _fallback_embed(text) -> List[float] │
│ - _validate_embedding(emb) -> None │
└─────────────────────────────────────┘
```

### Full Class Signature

```python
from typing import List, Optional
import httpx
from sentence_transformers import SentenceTransformer
import structlog

logger = structlog.get_logger()


class OllamaEmbedder:
    """
    Local embedding generation via Ollama API with intelligent fallback.

    Uses Ollama's nomic-embed-text model (768 dimensions, 81.2% MTEB accuracy)
    for privacy-preserving local embedding generation. Falls back to
    sentence-transformers (all-MiniLM-L6-v2) if Ollama is unavailable.

    Attributes:
        base_url: Ollama API URL (e.g., "http://localhost:11434")
        model_name: Ollama embedding model (default: "nomic-embed-text")
        dimensions: Expected embedding dimensions (768 for nomic-embed-text)
        timeout: Request timeout in seconds (default: 30)
        max_retries: Maximum retry attempts for transient failures (default: 3)
        client: Async HTTP client for Ollama API calls
        fallback_model: sentence-transformers model (lazy loaded)

    Example:
        ```python
        embedder = OllamaEmbedder(
            base_url="http://localhost:11434",
            model_name="nomic-embed-text"
        )

        # Single text embedding
        embedding = await embedder.embed_text("Python is great")
        print(f"Dimensions: {len(embedding)}")  # 768

        # Batch embedding (efficient)
        texts = ["Python is great", "Ollama is local", "Privacy matters"]
        embeddings = await embedder.embed_batch(texts)
        print(f"Generated {len(embeddings)} embeddings")
        ```
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "nomic-embed-text",
        timeout: int = 30,
        max_retries: int = 3,
        enable_fallback: bool = True
    ) -> None:
        """
        Initialize OllamaEmbedder with connection parameters.

        Args:
            base_url: Ollama API URL (default: http://localhost:11434)
            model_name: Ollama embedding model name (default: nomic-embed-text)
            timeout: Request timeout in seconds (recommended: 10-60)
            max_retries: Max retry attempts for transient failures (recommended: 3-5)
            enable_fallback: Enable sentence-transformers fallback (default: True)

        Raises:
            ValueError: If base_url is invalid or timeout <= 0
            ConnectionError: If Ollama is unreachable and fallback disabled

        Example:
            ```python
            # Standard configuration (Phase 1)
            embedder = OllamaEmbedder()

            # Custom Ollama host
            embedder = OllamaEmbedder(base_url="http://192.168.1.100:11434")

            # Disable fallback (fail fast)
            embedder = OllamaEmbedder(enable_fallback=False)
            ```
        """

    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Algorithm:
        1. Validate input (non-empty, UTF-8)
        2. Call Ollama API with retry logic
        3. If Ollama fails after retries, fallback to sentence-transformers
        4. Validate embedding dimensions (must be 768)
        5. Return embedding vector

        Args:
            text: Input text to embed (max 8192 tokens for nomic-embed-text)

        Returns:
            List[float]: 768-dimensional embedding vector

        Raises:
            ValidationError: If text is empty or exceeds max length
            EmbeddingError: If both Ollama and fallback fail
            TimeoutError: If request exceeds timeout

        Performance Target:
            - Single embedding: < 150ms (P95)
            - Includes retry overhead if needed

        Example:
            ```python
            embedder = OllamaEmbedder()

            text = "Python is a programming language"
            embedding = await embedder.embed_text(text)

            assert len(embedding) == 768
            assert all(isinstance(x, float) for x in embedding)
            print(f"First 5 dims: {embedding[:5]}")
            ```
        """

    async def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.

        Processes texts in batches for optimal performance. Uses asyncio.gather
        for concurrent API calls (up to batch_size concurrent).

        Algorithm:
        1. Validate inputs (all non-empty, batch_size valid)
        2. Split texts into batches of size batch_size
        3. For each batch:
            a. Launch concurrent embed_text() calls (asyncio.gather)
            b. Collect results
        4. Flatten batches into single list
        5. Return embeddings

        Args:
            texts: List of input texts (max 1000 texts per call recommended)
            batch_size: Concurrent requests (default: 32, max recommended: 64)

        Returns:
            List[List[float]]: List of 768-dimensional embeddings (same order as inputs)

        Raises:
            ValidationError: If any text is empty or batch_size invalid
            EmbeddingError: If all texts in batch fail (partial failures logged, not raised)

        Performance Target:
            - 32 texts: < 1000ms (P95)
            - Throughput: ~200 texts/sec with batch_size=32

        Example:
            ```python
            embedder = OllamaEmbedder()

            texts = [
                "Python is great",
                "Ollama runs locally",
                "Privacy is important",
                "Semantic search is powerful"
            ]

            embeddings = await embedder.embed_batch(texts, batch_size=32)

            assert len(embeddings) == len(texts)
            assert all(len(emb) == 768 for emb in embeddings)
            print(f"Generated {len(embeddings)} embeddings")
            ```
        """

    def get_dimensions(self) -> int:
        """
        Return embedding dimensions for this model.

        Returns:
            int: Embedding dimensions (768 for nomic-embed-text)

        Example:
            ```python
            embedder = OllamaEmbedder()
            dims = embedder.get_dimensions()
            assert dims == 768
            ```
        """

    async def health_check(self) -> bool:
        """
        Check if Ollama API is reachable and model is available.

        Performs a lightweight test request to Ollama API to verify:
        - API is reachable
        - Model is downloaded and ready
        - Embeddings can be generated

        Returns:
            bool: True if Ollama healthy, False otherwise

        Performance Target:
            - Execution time: < 200ms

        Example:
            ```python
            embedder = OllamaEmbedder()

            if await embedder.health_check():
                print("Ollama is ready")
            else:
                print("Ollama unavailable, will use fallback")
            ```
        """

    async def _call_ollama(
        self,
        text: str,
        retry_count: int = 0
    ) -> List[float]:
        """
        Internal method: Call Ollama API with retry logic.

        Implements exponential backoff retry strategy:
        - Attempt 1: immediate
        - Attempt 2: wait 1s
        - Attempt 3: wait 2s
        - Attempt 4: wait 4s

        Args:
            text: Text to embed
            retry_count: Current retry attempt (internal)

        Returns:
            List[float]: 768-dimensional embedding

        Raises:
            EmbeddingError: If all retries exhausted
            TimeoutError: If request exceeds timeout

        Private method, not exposed in public API.
        """

    async def _fallback_embed(self, text: str) -> List[float]:
        """
        Internal method: Generate embedding using sentence-transformers fallback.

        Uses all-MiniLM-L6-v2 model (384 dimensions). Pads with zeros to match
        768 dimensions for compatibility with FalkorDB vector index.

        Args:
            text: Text to embed

        Returns:
            List[float]: 768-dimensional embedding (384 real + 384 zeros)

        Private method, not exposed in public API.
        """

    def _validate_embedding(self, embedding: List[float]) -> None:
        """
        Internal method: Validate embedding dimensions and values.

        Checks:
        - Embedding is list of floats
        - Length is exactly 768
        - No NaN or Inf values

        Args:
            embedding: Embedding to validate

        Raises:
            EmbeddingError: If validation fails

        Private method, not exposed in public API.
        """
```

---

## Dependencies

### Component Dependencies

**Internal:**
- None (standalone component)

**External Libraries:**
- `httpx>=0.25.0` - Async HTTP client for Ollama API
- `sentence-transformers>=2.2.0` - Fallback embedding model
- `structlog>=23.2.0` - Structured logging

### Dependency Injection

**Constructor Injection:**
```python
# Dependencies injected via constructor
embedder = OllamaEmbedder(
    base_url="http://localhost:11434",  # Configurable Ollama host
    model_name="nomic-embed-text",      # Configurable model
    enable_fallback=True                # Enable/disable fallback
)
```

**No runtime injection needed** - all dependencies are standard Python libraries

---

## State Management

### Attributes

**Configuration (immutable after __init__):**
- `base_url: str` - Ollama API URL, set at initialization, never changes
- `model_name: str` - Embedding model name, set at initialization
- `dimensions: int` - Expected embedding dims (768), constant
- `timeout: int` - Request timeout, set at initialization
- `max_retries: int` - Max retry attempts, set at initialization

**Runtime State:**
- `client: httpx.AsyncClient` - HTTP client, initialized in __init__, reused for all requests
- `fallback_model: Optional[SentenceTransformer]` - Lazy loaded on first fallback usage, cached

### State Transitions

```
Initial State (after __init__)
    ↓
[embed_text called] → Ollama API Request → Success
    ↓                       ↓
[embed_text called]    Retry (3x)
    ↓                       ↓
[embed_text called]    Fallback (if enabled)
    ↓
Steady State (client reused for all calls)
```

### Thread Safety

**Is this component thread-safe?** Yes (with caveats)

**Concurrency Model:**
- Safe for concurrent async calls (asyncio tasks)
- httpx.AsyncClient is thread-safe for async usage
- fallback_model lazy loading is NOT thread-safe (use asyncio.Lock if needed)
- Recommendation: Use within single asyncio event loop

**Synchronization:**
- No explicit locks required for normal async usage
- If lazy-loading fallback concurrently, add Lock:
  ```python
  self._fallback_lock = asyncio.Lock()
  async with self._fallback_lock:
      if self.fallback_model is None:
          self.fallback_model = SentenceTransformer(...)
  ```

---

## Public Methods (Detailed)

### Method 1: `embed_text`

**Signature:**
```python
async def embed_text(self, text: str) -> List[float]
```

**Purpose:** Generate a 768-dimensional embedding for a single text using Ollama API

**Parameters:**
- `text`: str
  - Description: Input text to generate embedding for
  - Constraints: Must be non-empty, valid UTF-8, max 8192 tokens (~32KB)
  - Example: "Python is a programming language created by Guido van Rossum"
  - Validation: Check `text.strip()` non-empty, count tokens with tiktoken

**Returns:**
- Type: `List[float]`
- Description: 768-dimensional embedding vector
- Constraints:
  - Length is exactly 768
  - All values are floats (no NaN, no Inf)
  - Normalized to unit length (cosine similarity compatible)
- Example: `[0.023, -0.145, 0.087, ..., 0.012]` (768 floats)

**Raises:**
- `ValidationError`: When text is empty, non-UTF-8, or exceeds max length
  - Example: `ValidationError("Text cannot be empty")`
- `EmbeddingError`: When both Ollama and fallback fail
  - Example: `EmbeddingError("Failed to generate embedding: Ollama model 'nomic-embed-text' not found")`
- `TimeoutError`: When request exceeds timeout (30s default)
  - Example: `TimeoutError("Embedding request timed out after 30 seconds")`

**Preconditions:**
- OllamaEmbedder must be initialized
- Ollama service should be running (or fallback enabled)
- Model should be downloaded (`ollama pull nomic-embed-text`)

**Postconditions:**
- If success: Returns valid 768-dimensional embedding
- If Ollama fails: Attempts fallback (if enabled)
- If all fail: Raises EmbeddingError with actionable message
- Logs all attempts (DEBUG: API calls, WARNING: retries, ERROR: failures)

**Algorithm Outline:**
```
1. Validate input:
   - Check text is non-empty (after strip)
   - Check UTF-8 encoding
   - Count tokens (tiktoken), ensure <= 8192

2. Try Ollama API:
   - Call _call_ollama(text)
   - Retry up to max_retries times with exponential backoff
   - If success: validate embedding dimensions, return

3. If Ollama fails (after retries):
   - Log warning: "Ollama failed, falling back to sentence-transformers"
   - Call _fallback_embed(text)
   - Validate embedding dimensions, return

4. If fallback disabled or fails:
   - Raise EmbeddingError with helpful message

5. Log success:
   - logger.info("embedding_generated", model="ollama", dimensions=768)
```

**Edge Cases:**

1. **Empty text:**
   - Input: `text = ""`
   - Behavior: Raise `ValidationError("Text cannot be empty")`

2. **Very long text (> 8192 tokens):**
   - Input: `text = "word " * 10000`
   - Behavior: Raise `ValidationError("Text exceeds max length (8192 tokens)")`

3. **Ollama service offline:**
   - Scenario: Ollama not running
   - Behavior: Retry 3x → fallback to sentence-transformers → return 768-dim embedding

4. **Ollama model not downloaded:**
   - Scenario: `ollama pull nomic-embed-text` not run
   - Behavior: API returns 404 → raise `EmbeddingError("Model 'nomic-embed-text' not found. Run: ollama pull nomic-embed-text")`

5. **Non-UTF-8 text:**
   - Input: `text = b"\x80\x81".decode("latin1")`
   - Behavior: Raise `ValidationError("Text must be valid UTF-8")`

6. **Ollama returns invalid embedding:**
   - Scenario: Ollama bug returns 512-dim embedding instead of 768
   - Behavior: Raise `EmbeddingError("Invalid embedding dimensions: expected 768, got 512")`

**Related Methods:**
- Calls: `_call_ollama()`, `_fallback_embed()`, `_validate_embedding()`
- Called by: `embed_batch()`, `MemoryProcessor.add_memory()`

### Method 2: `embed_batch`

**Signature:**
```python
async def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]
```

**Purpose:** Generate embeddings for multiple texts efficiently with concurrent processing

**Parameters:**
- `texts`: List[str]
  - Description: List of input texts to embed
  - Constraints: Each text must be non-empty, max 1000 texts per call recommended
  - Example: `["Python is great", "Ollama runs locally", "Privacy matters"]`
  - Validation: Check all texts non-empty, list length > 0

- `batch_size`: int
  - Description: Number of concurrent API requests
  - Constraints: Must be 1-64 (recommended: 32)
  - Default: 32
  - Example: 32 (optimal for most systems)
  - Validation: Check 1 <= batch_size <= 64

**Returns:**
- Type: `List[List[float]]`
- Description: List of 768-dimensional embeddings (same order as inputs)
- Constraints:
  - Length matches `len(texts)`
  - Each embedding is 768 floats
  - Order preserved (embeddings[i] corresponds to texts[i])
- Example: `[[0.1, 0.2, ...], [0.3, 0.4, ...], ...]`

**Raises:**
- `ValidationError`: When any text is empty or batch_size invalid
  - Example: `ValidationError("All texts must be non-empty")`
- `EmbeddingError`: When all texts fail to embed (partial failures logged but not raised)
  - Example: `EmbeddingError("Failed to embed batch: all texts failed")`

**Preconditions:**
- OllamaEmbedder initialized
- At least 1 text in list
- Ollama service running (or fallback enabled)

**Postconditions:**
- Returns embeddings for all texts (same order)
- Partial failures logged (WARNING level)
- If > 50% of batch fails, raises EmbeddingError

**Algorithm Outline:**
```
1. Validate inputs:
   - Check texts list is non-empty
   - Check all texts are non-empty strings
   - Check batch_size in valid range (1-64)

2. Split texts into batches:
   - batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]

3. For each batch:
   - tasks = [embed_text(text) for text in batch]
   - results = await asyncio.gather(*tasks, return_exceptions=True)
   - Separate successes and failures
   - Log failures (WARNING level)

4. Collect all embeddings:
   - Flatten batches into single list
   - Preserve order

5. Validate:
   - Check len(embeddings) == len(texts)
   - If failure_rate > 50%, raise EmbeddingError

6. Return embeddings
```

**Edge Cases:**

1. **Single text:**
   - Input: `texts = ["Python is great"]`
   - Behavior: Works normally, returns `List[List[float]]` with 1 embedding

2. **Empty texts list:**
   - Input: `texts = []`
   - Behavior: Raise `ValidationError("texts list cannot be empty")`

3. **Batch size larger than texts:**
   - Input: `texts = ["a", "b"], batch_size = 32`
   - Behavior: Works normally, processes all in one batch

4. **Partial failures (some texts fail):**
   - Scenario: 5/10 texts fail due to transient Ollama errors
   - Behavior: Log failures, return embeddings for successful 5, skip failed (or raise if > 50% failed)

5. **All texts fail:**
   - Scenario: Ollama crashes mid-batch
   - Behavior: Raise `EmbeddingError("Failed to embed batch: all 10 texts failed")`

6. **Very large batch (1000 texts):**
   - Input: `texts = ["text"] * 1000, batch_size = 32`
   - Behavior: Split into 32 batches, process sequentially (each batch concurrent)

**Related Methods:**
- Calls: `embed_text()` (multiple times concurrently)
- Called by: `MemoryProcessor.add_memory()` (for chunk batches)

### Method 3: `get_dimensions`

**Signature:**
```python
def get_dimensions(self) -> int
```

**Purpose:** Return embedding dimensions for the configured model

**Parameters:**
- None

**Returns:**
- Type: `int`
- Value: 768 (for nomic-embed-text)
- Constant (does not change after initialization)

**Raises:**
- None (pure function, always succeeds)

**Example:**
```python
embedder = OllamaEmbedder(model_name="nomic-embed-text")
dims = embedder.get_dimensions()
assert dims == 768
```

**Use Case:**
- Used by FalkorDBClient to configure vector index dimensions
- Used for validation of embedding results

**Related Methods:**
- Called by: `FalkorDBClient.__init__()`, `_validate_embedding()`

### Method 4: `health_check`

**Signature:**
```python
async def health_check(self) -> bool
```

**Purpose:** Verify Ollama API is reachable and model is available

**Parameters:**
- None

**Returns:**
- Type: `bool`
- True: Ollama is healthy, embeddings can be generated
- False: Ollama unavailable, will fallback to sentence-transformers

**Raises:**
- None (catches all exceptions internally, returns False)

**Algorithm:**
```
1. Try to generate embedding for test text:
   - Test text: "health check"
   - Call Ollama API directly (no retries)

2. If success (200 OK, valid embedding):
   - Return True

3. If failure (any exception):
   - Log debug message
   - Return False
```

**Performance Target:**
- Execution time: < 200ms

**Example:**
```python
embedder = OllamaEmbedder()

if await embedder.health_check():
    print("✅ Ollama is ready")
else:
    print("⚠️  Ollama unavailable, using fallback")
```

**Use Case:**
- Called during MemoryProcessor initialization to verify Ollama availability
- Used in monitoring/health endpoints

**Related Methods:**
- Calls: `_call_ollama()` (with no retries)
- Called by: `MemoryProcessor.__init__()`, health check endpoints

---

## Error Handling

### Exceptions Defined

```python
# zapomni_core/exceptions.py

class EmbeddingError(ZapomniCoreError):
    """
    Raised when embedding generation fails.

    Scenarios:
    - Ollama API unreachable (after retries)
    - Model not found (404)
    - Invalid embedding returned
    - Both Ollama and fallback fail
    """
    pass

class ValidationError(ZapomniCoreError):
    """
    Raised when input validation fails.

    Scenarios:
    - Empty text
    - Non-UTF-8 encoding
    - Text exceeds max length (8192 tokens)
    - Invalid batch_size
    """
    pass

class TimeoutError(ZapomniCoreError):
    """
    Raised when request exceeds timeout.

    Scenarios:
    - Ollama API call takes > 30s
    - Network latency issues
    """
    pass
```

### Error Recovery

**Retry Strategy:**
- **Transient Errors (ConnectionError, Timeout):** Retry 3x with exponential backoff
  - Attempt 1: immediate
  - Attempt 2: wait 1s
  - Attempt 3: wait 2s
  - Attempt 4: wait 4s

**Fallback Behavior:**
- If Ollama fails after retries → fallback to sentence-transformers (if enabled)
- If fallback disabled → raise EmbeddingError immediately
- If fallback also fails → raise EmbeddingError with both error messages

**Error Propagation:**
- All errors propagate to caller (MemoryProcessor)
- MemoryProcessor decides whether to fail entire operation or skip failed chunks
- Structured logging ensures errors are traceable

---

## Usage Examples

### Basic Usage

```python
import asyncio
from zapomni_core.embedding import OllamaEmbedder

async def main():
    # Initialize embedder
    embedder = OllamaEmbedder(
        base_url="http://localhost:11434",
        model_name="nomic-embed-text"
    )

    # Check if Ollama is available
    if await embedder.health_check():
        print("✅ Ollama is ready")
    else:
        print("⚠️  Ollama unavailable, will use fallback")

    # Generate single embedding
    text = "Python is a programming language"
    embedding = await embedder.embed_text(text)

    print(f"Generated embedding with {len(embedding)} dimensions")
    print(f"First 5 values: {embedding[:5]}")

asyncio.run(main())
```

### Advanced Usage (Batch Processing)

```python
import asyncio
from zapomni_core.embedding import OllamaEmbedder
from zapomni_core.chunking import SemanticChunker

async def process_document():
    # Initialize components
    chunker = SemanticChunker(chunk_size=512, chunk_overlap=50)
    embedder = OllamaEmbedder()

    # Chunk long document
    document = "Python is a programming language. " * 100
    chunks = chunker.chunk_text(document)

    # Extract chunk texts
    chunk_texts = [chunk.text for chunk in chunks]

    # Generate embeddings in batch (efficient)
    embeddings = await embedder.embed_batch(
        texts=chunk_texts,
        batch_size=32  # 32 concurrent requests
    )

    print(f"Processed {len(chunks)} chunks")
    print(f"Generated {len(embeddings)} embeddings")

    # Verify dimensions
    assert all(len(emb) == 768 for emb in embeddings)
    print("✅ All embeddings valid (768 dimensions)")

asyncio.run(process_document())
```

### Error Handling Example

```python
import asyncio
from zapomni_core.embedding import OllamaEmbedder
from zapomni_core.exceptions import EmbeddingError, ValidationError

async def robust_embedding():
    embedder = OllamaEmbedder(
        enable_fallback=True,  # Enable fallback for reliability
        max_retries=3
    )

    texts = ["Python is great", "Ollama runs locally", ""]

    for text in texts:
        try:
            embedding = await embedder.embed_text(text)
            print(f"✅ Embedded: {text[:30]}... → {len(embedding)} dims")

        except ValidationError as e:
            print(f"❌ Validation failed: {e}")
            # Skip invalid text

        except EmbeddingError as e:
            print(f"❌ Embedding failed: {e}")
            # Log error, maybe retry later or skip

        except TimeoutError as e:
            print(f"⏰ Timeout: {e}")
            # Increase timeout or skip

asyncio.run(robust_embedding())
```

---

## Testing Approach

### Unit Tests Required

**Happy Path Tests:**
1. `test_init_success()` - Normal initialization with default parameters
2. `test_init_custom_params()` - Initialization with custom base_url, model, timeout
3. `test_embed_text_success()` - Single text embedding (mock Ollama response)
4. `test_embed_batch_success()` - Batch embedding (mock responses)
5. `test_get_dimensions()` - Returns 768

**Error Tests:**
6. `test_init_invalid_base_url_raises()` - Invalid URL format
7. `test_init_invalid_timeout_raises()` - timeout <= 0
8. `test_embed_text_empty_raises()` - Empty text input
9. `test_embed_text_too_long_raises()` - Text > 8192 tokens
10. `test_embed_text_non_utf8_raises()` - Invalid encoding
11. `test_embed_batch_empty_list_raises()` - Empty texts list
12. `test_embed_batch_invalid_batch_size_raises()` - batch_size < 1 or > 64

**Retry & Fallback Tests:**
13. `test_embed_text_retry_on_timeout()` - Retries 3x on timeout, succeeds on 3rd
14. `test_embed_text_fallback_on_failure()` - Ollama fails → fallback succeeds
15. `test_embed_text_fails_if_no_fallback()` - Ollama fails, fallback disabled → EmbeddingError
16. `test_embed_batch_partial_failures()` - Some texts fail, logs warnings, returns successes

**Performance Tests:**
17. `test_embed_text_performance()` - Single embedding < 150ms (mock, no real API)
18. `test_embed_batch_performance()` - 32 texts in < 1000ms (mock)

**Integration Tests (require Ollama running):**
19. `test_embed_text_real_ollama()` - Real Ollama API call
20. `test_health_check_real_ollama()` - Real health check

### Mocking Strategy

**Mock Ollama API responses:**
```python
import pytest
from unittest.mock import AsyncMock, patch

@pytest.fixture
def mock_ollama_response():
    """Mock successful Ollama API response."""
    return {
        "embedding": [0.1] * 768  # 768-dimensional mock embedding
    }

@pytest.mark.asyncio
async def test_embed_text_success(mock_ollama_response):
    """Test embed_text with mocked Ollama API."""
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        # Mock successful API response
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = mock_ollama_response

        embedder = OllamaEmbedder()
        embedding = await embedder.embed_text("test text")

        assert len(embedding) == 768
        assert all(isinstance(x, float) for x in embedding)
        mock_post.assert_called_once()
```

**Mock sentence-transformers (fallback):**
```python
@pytest.fixture
def mock_sentence_transformer():
    """Mock sentence-transformers model."""
    mock_model = AsyncMock()
    mock_model.encode.return_value = [0.1] * 384  # 384-dim embedding
    return mock_model

@pytest.mark.asyncio
async def test_embed_text_fallback(mock_sentence_transformer):
    """Test fallback to sentence-transformers."""
    with patch("httpx.AsyncClient.post", side_effect=ConnectionError):
        with patch("sentence_transformers.SentenceTransformer", return_value=mock_sentence_transformer):
            embedder = OllamaEmbedder(enable_fallback=True)
            embedding = await embedder.embed_text("test text")

            # Fallback returns 384 real + 384 zeros = 768
            assert len(embedding) == 768
```

### Integration Tests

**Test with real Ollama (optional):**
```python
@pytest.mark.integration
@pytest.mark.requires_ollama
@pytest.mark.asyncio
async def test_embed_text_real_ollama():
    """Test with real Ollama API (requires Ollama running)."""
    embedder = OllamaEmbedder()

    # Check Ollama is available
    if not await embedder.health_check():
        pytest.skip("Ollama not available")

    # Generate real embedding
    embedding = await embedder.embed_text("Python is a programming language")

    assert len(embedding) == 768
    assert all(isinstance(x, float) for x in embedding)
    assert all(-1.0 <= x <= 1.0 for x in embedding)  # Normalized
```

---

## Performance Considerations

### Time Complexity

**embed_text():**
- **Ollama API call:** O(n) where n = text length (tokenization + inference)
- **Validation:** O(n) for UTF-8 check and token counting
- **Overall:** O(n) per text

**embed_batch():**
- **Sequential batches:** O(m * n) where m = num_batches, n = avg text length
- **Concurrent within batch:** Amortized O(n) with batch_size concurrent requests
- **Overall:** O(total_tokens) but parallelized

### Space Complexity

**Memory usage:**
- **Single embedding:** 768 floats * 4 bytes = 3KB
- **Batch of 32:** ~96KB
- **httpx.AsyncClient:** ~1MB overhead
- **sentence-transformers model (if loaded):** ~200MB

**Total:** < 250MB for typical usage

### Optimization Opportunities

**Current Optimizations:**
1. **Async I/O:** Non-blocking API calls (httpx.AsyncClient)
2. **Batch processing:** Up to 32 concurrent requests via asyncio.gather
3. **Client reuse:** Single httpx.AsyncClient instance reused
4. **Lazy fallback loading:** sentence-transformers loaded only if needed

**Future Optimizations (Phase 2+):**
1. **Semantic caching:** Cache embeddings (SHA256 hash of text → embedding)
   - Target: 60%+ cache hit rate
   - LRU eviction, 24h TTL
2. **Connection pooling:** Reuse HTTP connections (httpx already does this)
3. **GPU acceleration:** Use Ollama GPU support if available (CUDA)
4. **Dynamic batch sizing:** Adjust batch_size based on Ollama performance

---

## References

### Module Spec
- [zapomni_core_module.md](/home/dev/zapomni/.spec-workflow/specs/level1/zapomni_core_module.md) - Parent module specification

### Related Components
- [semantic_chunker_component.md](/home/dev/zapomni/.spec-workflow/specs/level2/semantic_chunker_component.md) - Produces chunks for embedding
- (Future) `entity_extractor_component.md` - Consumes embeddings for entity extraction

### External Documentation
- **Ollama API:** https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
- **nomic-embed-text:** https://huggingface.co/nomic-ai/nomic-embed-text-v1
- **sentence-transformers:** https://www.sbert.net/docs/pretrained_models.html
- **httpx:** https://www.python-httpx.org/async/

---

**Document Status:** Draft v1.0
**Created:** 2025-11-23
**Author:** Goncharenko Anton aka alienxs2
**Copyright:** Copyright (c) 2025 Goncharenko Anton aka alienxs2
**License:** MIT License

**Total Sections:** 11
**Total Methods:** 4 public + 3 private
**Total Test Scenarios:** 20
**Ready for Review:** Yes ✅
