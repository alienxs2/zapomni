# SemanticChunker.chunk_text - Function Specification

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
def chunk_text(self, text: str) -> List[Chunk]:
    """
    Split text into semantic chunks with configurable size and overlap.

    Intelligently splits input text at natural boundaries (paragraphs, sentences,
    words) while maintaining target chunk size. Creates overlapping chunks to
    preserve context at boundaries. Each chunk includes metadata for tracking
    position and provenance.

    This is the core chunking function used by MemoryProcessor to prepare text
    for embedding and storage in the knowledge graph.

    Args:
        text: Input text to chunk. Must be non-empty UTF-8 string with max
              length of 10,000,000 characters (~10MB). Can contain natural
              language, technical documentation, or any text content.

    Returns:
        List of Chunk objects, ordered by position in original text:
        - text (str): Chunk content
        - index (int): 0-based position in chunk sequence
        - start_char (int): Character offset in original text
        - end_char (int): End character offset
        - metadata (Optional[dict]): None in current implementation

        Guarantees:
        - Length >= 1 (at least one chunk for valid input)
        - Chunks ordered (chunk[i].index == i)
        - No gaps (chunk[i].end_char == chunk[i+1].start_char - overlap)
        - All chunks >= min_chunk_size tokens (except possibly last)
        - Overlap exists between adjacent chunks (chunk_overlap tokens)

    Raises:
        ValidationError: If text is empty, too large, or invalid UTF-8
        ChunkingError: If internal chunking operation fails

    Performance Target:
        - Small text (< 1KB): < 10ms
        - Medium text (< 100KB): < 50ms
        - Large text (< 1MB): < 200ms
        - Maximum text (10MB): < 2000ms

    Example:
        ```python
        chunker = SemanticChunker(chunk_size=512, chunk_overlap=50)

        text = \"\"\"
        Python is a high-level programming language.

        It was created by Guido van Rossum in 1991.
        Python emphasizes code readability and simplicity.
        \"\"\"

        chunks = chunker.chunk_text(text)

        print(f"Created {len(chunks)} chunks")
        for chunk in chunks:
            print(f"Chunk {chunk.index}: {chunk.text[:50]}...")
            print(f"  Position: chars {chunk.start_char}-{chunk.end_char}")
        ```
    """
```

---

## Purpose & Context

### What It Does

`chunk_text()` splits input text into semantically meaningful chunks optimized for embedding and retrieval. It:

1. **Validates input** - Ensures text is non-empty, valid UTF-8, and within size limits
2. **Splits at natural boundaries** - Uses LangChain RecursiveCharacterTextSplitter to split at paragraph breaks (\n\n), then line breaks (\n), then word boundaries, while respecting target chunk size
3. **Creates overlap** - Adjacent chunks share `chunk_overlap` tokens to preserve context at boundaries
4. **Generates metadata** - Tracks chunk index, character offsets for provenance
5. **Merges small chunks** - Chunks smaller than `min_chunk_size` are merged with neighbors to maintain quality
6. **Returns structured data** - List of Chunk objects ready for embedding

### Why It Exists

**Problem:** Naive fixed-size chunking (split every N characters) creates poor chunks:
- ❌ Splits mid-sentence, destroying semantic meaning
- ❌ Loses context at boundaries (retrieval misses relevant information)
- ❌ Creates inconsistent chunk quality (some tiny, some huge)

**Solution:** Semantic chunking:
- ✅ Respects natural text structure (paragraphs, sentences)
- ✅ Maintains context with overlap between chunks
- ✅ Produces consistent, high-quality chunks for embedding

**Impact:** Better chunking → Better embeddings → Better search results

### When To Use

**Primary Use Case:**
- Called by `MemoryProcessor.add_memory()` for ALL text inputs (user messages, documents, notes)

**Typical Flow:**
```
User adds memory via MCP tool
    ↓
add_memory MCP tool
    ↓
MemoryProcessor.add_memory(text="...")
    ↓
SemanticChunker.chunk_text(text) ← THIS FUNCTION
    ↓
OllamaEmbedder.embed(chunks)
    ↓
FalkorDBClient.store(chunks, embeddings)
```

### When NOT To Use

**Don't use for:**
- ❌ **Code files (Phase 1-2)** - Use as text in Phase 1, will use `chunk_code()` in Phase 3
- ❌ **Already chunked text** - If caller has pre-split text, use Chunk objects directly
- ❌ **Binary data** - Not designed for non-text content
- ❌ **Real-time streaming** - Requires complete text, not suitable for incremental chunking

---

## Parameters (Detailed)

### text: str

**Type:** `str`

**Purpose:** Input text to split into semantic chunks

**Constraints:**

1. **Non-empty (after strip):**
   - Must contain at least 1 non-whitespace character
   - Empty string `""` → ValidationError
   - Whitespace-only `"   \n\t  "` → ValidationError (stripped before check)

2. **Valid UTF-8 encoding:**
   - Must be valid Unicode string
   - Binary data or invalid encodings → ValidationError
   - Python `str` type guarantees UTF-8 by default

3. **Length limits:**
   - Minimum: 1 character (after strip)
   - Maximum: 10,000,000 characters (~10MB)
   - Exceeding max → ValidationError
   - **Rationale:** 10MB limit prevents memory exhaustion, reasonable for text documents

4. **Content:**
   - Can contain any Unicode characters (letters, numbers, punctuation, emojis, etc.)
   - Newlines (\n) and paragraph breaks (\n\n) are used as split boundaries
   - No restrictions on specific characters

**Validation (pseudocode):**
```python
# Step 1: Strip and check empty
stripped_text = text.strip()
if not stripped_text:
    raise ValidationError("Text cannot be empty")

# Step 2: Check length
if len(text) > 10_000_000:
    raise ValidationError(f"Text exceeds maximum length (10,000,000 chars): {len(text)}")

# Step 3: Verify UTF-8 (implicit in Python str type)
# If text is `str`, it's already valid UTF-8
# Binary data would be `bytes`, not `str`
if not isinstance(text, str):
    raise ValidationError("Text must be string, not bytes")
```

**Examples:**

**Valid inputs:**
```python
# Simple sentence
text = "Python is a programming language."

# Multiple paragraphs
text = """
Python is a high-level language.

It was created in 1991.
"""

# Technical documentation
text = "The RecursiveCharacterTextSplitter splits at \\n\\n, \\n, and spaces."

# Single character
text = "A"

# Maximum length (10MB)
text = "A" * 10_000_000  # Exactly at limit, accepted

# Unicode content
text = "Python支持Unicode字符 🐍"
```

**Invalid inputs:**
```python
# Empty string
text = ""  # → ValidationError("Text cannot be empty")

# Whitespace only
text = "   \n\n\t  "  # → ValidationError("Text cannot be empty") after strip

# Too large
text = "A" * 10_000_001  # → ValidationError("Text exceeds maximum length")

# Not a string (would cause type error before reaching function)
text = b"bytes data"  # TypeError: expected str, got bytes
text = 123  # TypeError: expected str, got int
```

**Edge Cases:**

1. **Single character:** `"A"` → Returns single Chunk with text="A"
2. **Text smaller than chunk_size:** Returns single Chunk (no splitting needed)
3. **Text exactly chunk_size:** Returns single Chunk
4. **Text = chunk_size + 1:** Returns 2 Chunks with overlap
5. **All whitespace between words:** `"A    B    C"` → Split at word boundaries
6. **No paragraph breaks:** Long continuous text → Split at sentence/word boundaries
7. **Many paragraph breaks:** `"A\n\n\n\nB"` → Split at paragraph boundaries (\n\n)
8. **Code as text (Phase 1):** Treated as plain text, no special handling

---

## Return Value

**Type:** `List[Chunk]`

**Structure:**

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class Chunk:
    text: str                    # Chunk content
    index: int                   # 0-based position in sequence
    start_char: int              # Character offset in original text
    end_char: int                # End character offset
    metadata: Optional[dict]     # None in current implementation
```

**Guarantees:**

1. **Non-empty list:**
   - Length >= 1 for all valid inputs
   - Even single character returns [Chunk(...)]

2. **Ordered chunks:**
   - `chunks[i].index == i` for all i
   - First chunk has index=0
   - Last chunk has index=len(chunks)-1

3. **Complete coverage:**
   - Union of all chunk texts covers entire input (accounting for overlap)
   - No "gaps" where text is skipped
   - `chunks[0].start_char == 0`
   - `chunks[-1].end_char == len(text)`

4. **Overlap between adjacent chunks:**
   - Last ~`chunk_overlap` tokens of chunk[i] == first ~`chunk_overlap` tokens of chunk[i+1]
   - Preserves context at boundaries
   - **Exception:** Last chunk may have no overlap if it's final text

5. **Chunk size compliance:**
   - All chunks >= `min_chunk_size` tokens (except possibly last chunk)
   - All chunks <= `chunk_size` tokens (approximately, LangChain may slightly exceed)
   - Small chunks merged to meet min_chunk_size

6. **Metadata:**
   - Currently always `None`
   - Reserved for future use (e.g., chunk quality score, semantic type)

**Success Case Example:**

```python
chunker = SemanticChunker(chunk_size=50, chunk_overlap=10, min_chunk_size=20)

text = """Python is a high-level programming language.

It was created by Guido van Rossum in 1991.
Python emphasizes code readability."""

chunks = chunker.chunk_text(text)

# Returns:
[
    Chunk(
        text="Python is a high-level programming language.",
        index=0,
        start_char=0,
        end_char=45,
        metadata=None
    ),
    Chunk(
        text="It was created by Guido van Rossum in 1991.\nPython emphasizes code readability.",
        index=1,
        start_char=47,  # After \n\n
        end_char=130,
        metadata=None
    )
]

# Note: Actual split points depend on token count, not character count
# This is illustrative
```

**Edge Case Returns:**

1. **Single character input:**
   ```python
   text = "A"
   chunks = chunker.chunk_text(text)
   # [Chunk(text="A", index=0, start_char=0, end_char=1, metadata=None)]
   ```

2. **Text smaller than chunk_size:**
   ```python
   text = "Short text"  # < 50 tokens
   chunks = chunker.chunk_text(text)
   # [Chunk(text="Short text", index=0, start_char=0, end_char=10, metadata=None)]
   ```

3. **Large text (multiple chunks):**
   ```python
   text = "A" * 1000  # Large text
   chunks = chunker.chunk_text(text)
   # [Chunk(...), Chunk(...), Chunk(...), ...]  # Multiple chunks with overlap
   assert len(chunks) >= 2
   assert chunks[0].index == 0
   assert chunks[-1].index == len(chunks) - 1
   ```

---

## Exceptions

### ValidationError

**When Raised:**

1. **Empty text (after strip):**
   ```python
   text = ""
   # → ValidationError("Text cannot be empty")
   ```

2. **Whitespace-only text:**
   ```python
   text = "   \n\n\t  "
   # → ValidationError("Text cannot be empty")
   ```

3. **Text exceeds maximum length:**
   ```python
   text = "A" * 10_000_001  # 10MB + 1 char
   # → ValidationError("Text exceeds maximum length (10,000,000 chars): 10000001")
   ```

4. **Invalid UTF-8 (shouldn't occur with Python str, but defensive check):**
   ```python
   # If somehow bytes passed instead of str:
   text = b"\xff\xfe"  # Invalid UTF-8 bytes
   # → ValidationError("Text must be valid UTF-8 string")
   ```

**Message Format:**
```python
f"Text cannot be empty"
f"Text exceeds maximum length (10,000,000 chars): {len(text)}"
f"Text must be valid UTF-8 string"
```

**Recovery Strategy:**
- **Caller should NOT retry** (input is invalid)
- **User action required:** Provide valid text or split large text into smaller documents
- **MCP tool response:** Return error to user with helpful message

**Example Handling:**
```python
try:
    chunks = chunker.chunk_text(text)
except ValidationError as e:
    logger.error(f"Validation failed: {e}")
    return {"error": str(e), "hint": "Ensure text is non-empty and < 10MB"}
```

---

### ChunkingError

**When Raised:**

1. **RecursiveCharacterTextSplitter internal error:**
   ```python
   # Rare case: LangChain splitter raises exception
   # Possible causes: corrupted state, unexpected input format
   # → ChunkingError("Failed to split text: <original exception>")
   ```

2. **Token counting failure:**
   ```python
   # tiktoken raises exception (extremely rare)
   # → ChunkingError("Failed to count tokens: <original exception>")
   ```

3. **Merge operation failure:**
   ```python
   # merge_small_chunks() raises exception
   # → ChunkingError("Failed to merge small chunks: <original exception>")
   ```

**Message Format:**
```python
f"Failed to split text: {original_exception}"
f"Failed to count tokens: {original_exception}"
f"Failed to merge small chunks: {original_exception}"
```

**Recovery Strategy:**
- **Caller MAY retry once** (could be transient error)
- **If retry fails:** Log error, return failure to user
- **Escalation:** Log stack trace for debugging (likely a bug)

**Example Handling:**
```python
try:
    chunks = chunker.chunk_text(text)
except ChunkingError as e:
    logger.error(f"Chunking failed: {e}", exc_info=True)
    # Retry once
    try:
        chunks = chunker.chunk_text(text)
    except ChunkingError:
        return {"error": "Chunking failed after retry. Please contact support."}
```

---

## Algorithm (Pseudocode)

```
FUNCTION chunk_text(text: str) -> List[Chunk]:
    """Split text into semantic chunks with overlap."""

    # ===== STEP 1: VALIDATE INPUT =====
    stripped_text = text.strip()

    IF stripped_text is empty:
        RAISE ValidationError("Text cannot be empty")

    IF len(text) > 10_000_000:
        RAISE ValidationError(f"Text exceeds maximum length: {len(text)}")

    IF text is not instance of str:
        RAISE ValidationError("Text must be valid UTF-8 string")

    LOG debug: f"Chunking text of length {len(text)} chars"

    # ===== STEP 2: SPLIT TEXT USING LANGCHAIN =====
    TRY:
        # RecursiveCharacterTextSplitter splits at:
        # 1. Paragraph boundaries (\n\n)
        # 2. Line boundaries (\n)
        # 3. Word boundaries (" ")
        # 4. Character boundaries (if needed)
        # Respects chunk_size and chunk_overlap (in tokens, not chars)

        text_chunks = self.splitter.split_text(text)
        # Returns: List[str] with raw chunk texts

        LOG debug: f"Initial split created {len(text_chunks)} chunks"

    CATCH exception as e:
        LOG error: f"RecursiveCharacterTextSplitter failed: {e}"
        RAISE ChunkingError(f"Failed to split text: {e}")

    # ===== STEP 3: CREATE CHUNK OBJECTS WITH METADATA =====
    chunks = []
    current_offset = 0

    FOR index, chunk_text in enumerate(text_chunks):
        # Find actual position in original text
        # (Accounting for overlap - chunks may repeat text)

        start_char = text.find(chunk_text, current_offset)

        IF start_char == -1:
            # Chunk not found - should never happen
            LOG error: f"Chunk {index} not found in original text"
            RAISE ChunkingError(f"Chunk positioning error at index {index}")

        end_char = start_char + len(chunk_text)

        chunk = Chunk(
            text=chunk_text,
            index=index,
            start_char=start_char,
            end_char=end_char,
            metadata=None  # Reserved for future use
        )

        chunks.append(chunk)

        # Update offset for next chunk
        # Move forward by (chunk_size - overlap) to account for overlap
        current_offset = start_char + len(chunk_text) - (self.chunk_overlap * 4)
        # Note: Multiply overlap by ~4 to convert tokens to chars (rough estimate)
        # Actual overlap is in tokens, this is approximation for char search

    LOG debug: f"Created {len(chunks)} Chunk objects"

    # ===== STEP 4: MERGE SMALL CHUNKS =====
    TRY:
        merged_chunks = self.merge_small_chunks(chunks)

        LOG debug: f"After merge: {len(merged_chunks)} chunks"

    CATCH exception as e:
        LOG error: f"merge_small_chunks failed: {e}"
        RAISE ChunkingError(f"Failed to merge small chunks: {e}")

    # ===== STEP 5: FINAL VALIDATION =====
    # Sanity checks (should never fail unless bug)
    IF len(merged_chunks) == 0:
        LOG error: "No chunks created from valid input"
        RAISE ChunkingError("Chunking produced empty result")

    # Verify chunk indices are sequential
    FOR i, chunk in enumerate(merged_chunks):
        IF chunk.index != i:
            LOG warning: f"Re-indexing chunk {chunk.index} to {i}"
            chunk.index = i

    # ===== STEP 6: RETURN RESULT =====
    LOG info: f"Successfully chunked text into {len(merged_chunks)} chunks"

    RETURN merged_chunks

END FUNCTION
```

---

## Preconditions

**Required state before calling `chunk_text()`:**

1. ✅ **SemanticChunker instance initialized**
   - `__init__()` must have been called successfully
   - `self.splitter` is not None (RecursiveCharacterTextSplitter initialized)
   - `self.tokenizer` is not None (tiktoken encoding initialized)

2. ✅ **Valid configuration parameters**
   - `self.chunk_size > 0`
   - `self.chunk_overlap < self.chunk_size`
   - `self.min_chunk_size > 0 and < self.chunk_size`

3. ✅ **No external dependencies required**
   - No database connection needed
   - No API calls required
   - Self-contained operation

**No preconditions on input `text`** - validation happens inside function.

---

## Postconditions

**State after successful execution:**

1. ✅ **No state changes**
   - SemanticChunker instance unchanged (stateless operation)
   - `self.chunk_size`, `self.chunk_overlap`, etc. remain same
   - Pure function behavior (same input → same output)

2. ✅ **Return value guarantees**
   - `len(result) >= 1` (at least one chunk)
   - All chunks have valid metadata (index, start_char, end_char)
   - Chunks are ordered (sorted by index)

3. ✅ **Text coverage guarantee**
   - Union of all chunk texts covers entire input (accounting for overlap)
   - No gaps, no missing text
   - `result[0].start_char == 0`
   - `result[-1].end_char <= len(text)` (approximately equal, within overlap tolerance)

4. ✅ **Chunk quality guarantee**
   - All chunks >= `min_chunk_size` tokens (except possibly last)
   - All chunks <= `chunk_size` tokens (approximately)
   - Adjacent chunks have ~`chunk_overlap` tokens in common

**State after failure (exception raised):**

1. ✅ **No partial results**
   - Exception raised before any chunks returned
   - No incomplete Chunk objects exposed

2. ✅ **No state corruption**
   - SemanticChunker instance still usable
   - Can retry with different input

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
raise ValidationError("Text cannot be empty")
```

**Rationale:** Cannot create meaningful chunks from empty input. Fail fast.

**Test Scenario:**
```python
def test_chunk_text_empty_raises():
    """Test that empty text raises ValidationError."""
    chunker = SemanticChunker()

    with pytest.raises(ValidationError, match="cannot be empty"):
        chunker.chunk_text("")
```

---

### Edge Case 2: Whitespace-Only Text

**Scenario:** Text contains only whitespace `"   \n\n\t  "`

**Input:**
```python
text = "   \n\n\t  "
```

**Expected Behavior:**
```python
# After strip: "" (empty)
raise ValidationError("Text cannot be empty")
```

**Rationale:** Whitespace-only text has no semantic content. Treat as empty.

**Test Scenario:**
```python
def test_chunk_text_whitespace_only_raises():
    """Test that whitespace-only text raises ValidationError."""
    chunker = SemanticChunker()

    with pytest.raises(ValidationError, match="cannot be empty"):
        chunker.chunk_text("   \n\n\t  ")
```

---

### Edge Case 3: Single Character

**Scenario:** Text is single character `"A"`

**Input:**
```python
text = "A"
```

**Expected Behavior:**
```python
# Returns single chunk
[Chunk(text="A", index=0, start_char=0, end_char=1, metadata=None)]
```

**Rationale:** Minimal valid input. Create one chunk.

**Test Scenario:**
```python
def test_chunk_text_single_char():
    """Test chunking single character."""
    chunker = SemanticChunker()

    chunks = chunker.chunk_text("A")

    assert len(chunks) == 1
    assert chunks[0].text == "A"
    assert chunks[0].index == 0
    assert chunks[0].start_char == 0
    assert chunks[0].end_char == 1
```

---

### Edge Case 4: Text Smaller Than chunk_size

**Scenario:** Text is 50 tokens, chunk_size is 512 tokens

**Input:**
```python
text = "This is a short paragraph with about 15 words in it for testing purposes."
chunker = SemanticChunker(chunk_size=512, chunk_overlap=50)
```

**Expected Behavior:**
```python
# No split needed - return single chunk
[Chunk(text="This is a short...", index=0, start_char=0, end_char=75, metadata=None)]
```

**Rationale:** Text fits in single chunk. No need to split.

**Test Scenario:**
```python
def test_chunk_text_below_chunk_size():
    """Test that small text returns single chunk."""
    chunker = SemanticChunker(chunk_size=512, chunk_overlap=50)

    text = "Short text with about 10 tokens."
    chunks = chunker.chunk_text(text)

    assert len(chunks) == 1
    assert chunks[0].text == text
```

---

### Edge Case 5: Text Exactly chunk_size

**Scenario:** Text is exactly 512 tokens (matches chunk_size)

**Input:**
```python
# Generate text that's exactly 512 tokens
text = "word " * 512  # Approximately 512 tokens
chunker = SemanticChunker(chunk_size=512, chunk_overlap=50)
```

**Expected Behavior:**
```python
# Single chunk (no split needed, fits exactly)
[Chunk(text="word word word...", index=0, start_char=0, end_char=2560, metadata=None)]
```

**Rationale:** Fits in single chunk without exceeding limit.

**Test Scenario:**
```python
def test_chunk_text_exactly_chunk_size():
    """Test text that's exactly chunk_size tokens."""
    chunker = SemanticChunker(chunk_size=100, chunk_overlap=10)

    # Create text with exactly 100 tokens
    text = "word " * 100
    chunks = chunker.chunk_text(text)

    assert len(chunks) == 1
```

---

### Edge Case 6: Text Slightly Larger Than chunk_size

**Scenario:** Text is 513 tokens (1 token over chunk_size)

**Input:**
```python
text = "word " * 513  # 513 tokens
chunker = SemanticChunker(chunk_size=512, chunk_overlap=50)
```

**Expected Behavior:**
```python
# Split into 2 chunks with overlap
[
    Chunk(text="word word... (512 tokens)", index=0, start_char=0, end_char=~2560),
    Chunk(text="...word word (remaining tokens + overlap)", index=1, start_char=~2310, end_char=2565)
]
```

**Rationale:** Exceeds single chunk, split with overlap.

**Test Scenario:**
```python
def test_chunk_text_over_chunk_size_by_one():
    """Test text slightly larger than chunk_size creates 2 chunks."""
    chunker = SemanticChunker(chunk_size=100, chunk_overlap=10)

    # Create text with 101 tokens
    text = "word " * 101
    chunks = chunker.chunk_text(text)

    assert len(chunks) == 2
    # Check overlap exists
    # Last ~10 words of chunk[0] should appear in chunk[1]
```

---

### Edge Case 7: Very Long Text (Maximum 10MB)

**Scenario:** Text is exactly 10,000,000 characters (maximum allowed)

**Input:**
```python
text = "A" * 10_000_000  # Exactly 10MB
chunker = SemanticChunker(chunk_size=512, chunk_overlap=50)
```

**Expected Behavior:**
```python
# Successfully chunk (allowed)
# Returns many chunks (~2000+ chunks for 10MB text)
# All chunks valid, ordered, with overlap
assert len(chunks) > 1000
assert chunks[0].start_char == 0
assert chunks[-1].end_char <= 10_000_000
```

**Rationale:** Within limit. Process normally (may be slow, but allowed).

**Test Scenario:**
```python
@pytest.mark.slow
def test_chunk_text_max_length():
    """Test chunking maximum allowed text (10MB)."""
    chunker = SemanticChunker(chunk_size=512, chunk_overlap=50)

    text = "A" * 10_000_000  # Exactly at limit
    chunks = chunker.chunk_text(text)

    assert len(chunks) >= 1
    assert chunks[0].start_char == 0
    # Should complete in < 2 seconds (performance target)
```

---

### Edge Case 8: Text Exceeds Maximum (10MB + 1 char)

**Scenario:** Text is 10,000,001 characters (1 char over limit)

**Input:**
```python
text = "A" * 10_000_001  # Over limit
```

**Expected Behavior:**
```python
raise ValidationError("Text exceeds maximum length (10,000,000 chars): 10000001")
```

**Rationale:** Prevent memory exhaustion. Enforce limit strictly.

**Test Scenario:**
```python
def test_chunk_text_too_large_raises():
    """Test that text over 10MB raises ValidationError."""
    chunker = SemanticChunker()

    text = "A" * 10_000_001  # 1 char over limit

    with pytest.raises(ValidationError, match="exceeds maximum length"):
        chunker.chunk_text(text)
```

---

### Edge Case 9: Text With No Natural Boundaries

**Scenario:** Long continuous text with no paragraphs, sentences, or spaces

**Input:**
```python
text = "A" * 1000  # No word boundaries, just continuous "AAA..."
chunker = SemanticChunker(chunk_size=100, chunk_overlap=10)
```

**Expected Behavior:**
```python
# RecursiveCharacterTextSplitter falls back to character-level splitting
# Creates chunks at character boundaries (every ~100 chars)
# Overlap still preserved
assert len(chunks) >= 2
assert all(len(c.text) <= 100 for c in chunks)  # Approximate (token-based)
```

**Rationale:** When no natural boundaries exist, split at character level.

**Test Scenario:**
```python
def test_chunk_text_no_boundaries():
    """Test chunking text with no natural boundaries."""
    chunker = SemanticChunker(chunk_size=100, chunk_overlap=10)

    text = "A" * 500  # No spaces, paragraphs, or sentences
    chunks = chunker.chunk_text(text)

    assert len(chunks) >= 2  # Should split somewhere
    # Verify overlap exists
    assert chunks[0].text[-10:] in chunks[1].text  # Last 10 chars of first chunk
```

---

### Edge Case 10: Text With Many Paragraph Breaks

**Scenario:** Text with excessive paragraph breaks `\n\n`

**Input:**
```python
text = "Paragraph 1\n\n\n\n\n\nParagraph 2\n\n\n\nParagraph 3"
chunker = SemanticChunker(chunk_size=512, chunk_overlap=50)
```

**Expected Behavior:**
```python
# RecursiveCharacterTextSplitter treats multiple \n\n as single boundary
# Splits at paragraph boundaries
# Returns 3 chunks (one per paragraph) if each is small enough
# Or merges small chunks if < min_chunk_size
```

**Rationale:** Multiple \n\n collapsed to single paragraph break.

**Test Scenario:**
```python
def test_chunk_text_multiple_paragraph_breaks():
    """Test text with excessive paragraph breaks."""
    chunker = SemanticChunker(chunk_size=100, chunk_overlap=10)

    text = "Para 1\n\n\n\n\n\nPara 2\n\n\n\nPara 3"
    chunks = chunker.chunk_text(text)

    # Should respect paragraph boundaries
    assert any("Para 1" in c.text for c in chunks)
    assert any("Para 2" in c.text for c in chunks)
```

---

## Test Scenarios (Complete List)

### Happy Path Tests

1. **test_chunk_text_success_simple**
   - Input: Simple sentence "Python is a programming language."
   - Expected: 1 chunk, correct metadata (index=0, start_char=0, end_char=35)

2. **test_chunk_text_success_multiple_paragraphs**
   - Input: 3 paragraphs separated by \n\n
   - Expected: Chunks respect paragraph boundaries, overlap exists

3. **test_chunk_text_success_large_document**
   - Input: 1MB text (realistic document)
   - Expected: Many chunks, all valid, ordered, complete coverage

4. **test_chunk_text_respects_chunk_size**
   - Input: Text that should create 5 chunks based on chunk_size=100
   - Expected: ~5 chunks, each ~100 tokens (approximately)

5. **test_chunk_text_has_overlap**
   - Input: Text creating 2 chunks
   - Expected: Last ~10% of chunk[0].text appears in chunk[1].text

6. **test_chunk_text_preserves_paragraph_boundaries**
   - Input: Text with clear paragraph breaks
   - Expected: Chunks split at \n\n, not mid-paragraph

7. **test_chunk_text_metadata_correct**
   - Input: Sample text
   - Expected: chunk.index sequential, start_char/end_char correct

---

### Error Tests (Edge Cases Above)

8. **test_chunk_text_empty_raises**
   - Edge case 1 above

9. **test_chunk_text_whitespace_only_raises**
   - Edge case 2 above

10. **test_chunk_text_too_large_raises**
    - Edge case 8 above

---

### Boundary Tests

11. **test_chunk_text_single_char**
    - Edge case 3 above

12. **test_chunk_text_below_chunk_size**
    - Edge case 4 above

13. **test_chunk_text_exactly_chunk_size**
    - Edge case 5 above

14. **test_chunk_text_over_chunk_size_by_one**
    - Edge case 6 above

15. **test_chunk_text_max_length**
    - Edge case 7 above

---

### Special Content Tests

16. **test_chunk_text_unicode_content**
    - Input: Text with emojis, Chinese characters, etc.
    - Expected: Handled correctly, no encoding errors

17. **test_chunk_text_code_as_text** (Phase 1)
    - Input: Python code (treated as text in Phase 1)
    - Expected: Chunked as text (no AST parsing)

18. **test_chunk_text_no_boundaries**
    - Edge case 9 above

19. **test_chunk_text_multiple_paragraph_breaks**
    - Edge case 10 above

---

### Performance Tests

20. **test_chunk_text_performance_small**
    - Input: < 1KB text
    - Expected: Complete in < 10ms

21. **test_chunk_text_performance_medium**
    - Input: < 100KB text
    - Expected: Complete in < 50ms

22. **test_chunk_text_performance_large**
    - Input: < 1MB text
    - Expected: Complete in < 200ms

---

### Integration Tests (With MemoryProcessor)

23. **test_chunk_text_used_by_memory_processor**
    - Verify MemoryProcessor correctly uses chunk_text()
    - Verify chunks are passed to embedder

---

## Performance Requirements

**Latency Targets:**

| Input Size | Target Latency | Maximum Allowed |
|------------|----------------|-----------------|
| < 1KB      | < 10ms         | 50ms            |
| < 10KB     | < 20ms         | 100ms           |
| < 100KB    | < 50ms         | 200ms           |
| < 1MB      | < 200ms        | 500ms           |
| 10MB (max) | < 2000ms       | 5000ms          |

**Throughput:**
- Sequential chunking: 50-100 documents/sec (< 10KB docs)
- Large documents (1MB): 5-10 documents/sec

**Resource Usage:**
- Memory: O(n) where n = input size (~2x input for metadata)
- CPU: Single-threaded (no parallelism in Phase 1)

**Benchmark Test:**
```python
@pytest.mark.benchmark
def test_chunk_text_benchmark(benchmark):
    """Benchmark chunking performance."""
    chunker = SemanticChunker(chunk_size=512, chunk_overlap=50)
    text = "word " * 1000  # ~5KB text

    result = benchmark(chunker.chunk_text, text)

    assert len(result) >= 1
    # benchmark will verify latency automatically
```

---

## Security Considerations

**Input Validation:**
- ✅ All inputs validated before use (empty check, length check, type check)
- ✅ No injection vulnerabilities (text is treated as data, not code)
- ✅ Safe error messages (no sensitive data leaked in exceptions)

**Data Protection:**
- ❓ **Sensitive data in text?** Depends on user input
  - If text contains PII, caller responsible for sanitization
  - SemanticChunker does NOT log text content (only metadata)
  - Chunks stored in FalkorDB (caller ensures encryption at rest)

**Logging:**
- ✅ Only log metadata (text length, chunk count)
- ❌ Never log actual text content (could contain secrets)

**Example Safe Logging:**
```python
logger.info(f"Chunked text of {len(text)} chars into {len(chunks)} chunks")
# ✅ Safe - no content logged

logger.debug(f"Chunk 0 text: {chunks[0].text}")
# ❌ UNSAFE - logs actual content (could be sensitive)
```

---

## Related Functions

**Calls (internal):**
- `_validate_input(text)` - Validates text before chunking
- `_count_tokens(text)` - Counts tokens using tiktoken
- `merge_small_chunks(chunks)` - Merges chunks < min_chunk_size

**Calls (external dependencies):**
- `RecursiveCharacterTextSplitter.split_text(text)` - LangChain chunking
- `tiktoken.Encoding.encode(text)` - Token counting

**Called By:**
- `MemoryProcessor.add_memory(text)` - Main use case (memory ingestion)
- User code (direct usage, though typically through MemoryProcessor)

**Related Methods (same class):**
- `chunk_code(code, language)` - AST-based chunking for code (Phase 3)
- `merge_small_chunks(chunks)` - Post-processing helper

---

## Implementation Notes

**Libraries Used:**
- `langchain>=0.1.0` - RecursiveCharacterTextSplitter
- `tiktoken>=0.5.0` - Token counting (cl100k_base encoding)

**Algorithm Choice:**
- **Why RecursiveCharacterTextSplitter?**
  - Industry standard for semantic chunking
  - Well-tested, handles edge cases
  - Respects natural boundaries (paragraph → sentence → word)
  - Token-aware (counts tokens, not just characters)

**Known Limitations:**
1. **Character offset approximation:** Overlap is in tokens, but char offsets calculated approximately
   - Impact: Minor (affects internal tracking, not chunk quality)
   - Mitigation: Use robust search to find chunk positions in original text

2. **Cannot handle binary data:** Only UTF-8 text supported
   - Impact: Code/images must be converted to text first
   - Mitigation: Phase 3 will add `chunk_code()` for AST-based code chunking

3. **No streaming support:** Requires complete text in memory
   - Impact: Cannot chunk infinite/real-time streams
   - Mitigation: Future feature (streaming chunking)

**Future Enhancements:**
1. **Adaptive chunk_size:** Automatically adjust based on content type
2. **Streaming chunking:** Process text incrementally
3. **Semantic boundary detection:** ML-based detection of topic shifts
4. **Multi-language support:** Language-specific chunking rules

---

## References

**Component Spec:**
- [semantic_chunker_component.md](../level2/semantic_chunker_component.md) - Parent component specification

**Module Spec:**
- [zapomni_core_module.md](../level1/zapomni_core_module.md) - Parent module

**Related Function Specs (Level 3):**
- `SemanticChunker.__init__` - Constructor (to be created)
- `SemanticChunker.merge_small_chunks` - Merging helper (to be created)

**External Documentation:**
- LangChain Text Splitters: https://python.langchain.com/docs/modules/data_connection/document_transformers/
- tiktoken: https://github.com/openai/tiktoken
- RecursiveCharacterTextSplitter source: https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/text_splitter.py

---

**Document Status:** Draft v1.0
**Created:** 2025-11-23
**Author:** Goncharenko Anton aka alienxs2
**Copyright:** Copyright (c) 2025 Goncharenko Anton aka alienxs2
**License:** MIT License

**Ready for Implementation:** ✅ Yes

**Test Coverage Target:** 23 test scenarios defined (exceeds minimum 10 required)

**Edge Cases Identified:** 10 edge cases (exceeds minimum 6 required)

**Next Steps:**
1. Review and approve this function spec
2. Implement `chunk_text()` method following this spec exactly
3. Write all 23 test scenarios defined above
4. Verify performance targets met
5. Create function specs for remaining SemanticChunker methods
