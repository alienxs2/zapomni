# SemanticChunker - Component Specification

**Level:** 2 (Component)
**Module:** zapomni_core
**Author:** Goncharenko Anton aka alienxs2
**Status:** Draft
**Version:** 1.0
**Created:** 2025-11-23

---

## Overview

### Purpose

SemanticChunker is responsible for **intelligently splitting text and code into semantically meaningful chunks** optimized for embedding and retrieval. Unlike naive fixed-size chunking, SemanticChunker respects natural boundaries (paragraphs, sentences, code blocks) to preserve context and improve search quality.

This component serves as the **first critical step** in the memory processing pipeline - the quality of chunking directly impacts embedding effectiveness and retrieval accuracy.

### Responsibilities

1. **Text Chunking:** Split natural language text at semantic boundaries using RecursiveCharacterTextSplitter
2. **Chunk Validation:** Ensure chunks meet size constraints (min/max tokens) and quality standards
3. **Overlap Management:** Create overlapping chunks (10-20%) to preserve context at boundaries
4. **Metadata Preservation:** Track chunk index, character offsets, and provenance
5. **Code Chunking (Phase 3):** AST-based chunking using tree-sitter for code files

### Position in Module

SemanticChunker sits at the beginning of the core processing pipeline:

```
MemoryProcessor
    ↓
SemanticChunker (THIS)  ← Receives raw text
    ↓ produces List[Chunk]
OllamaEmbedder
    ↓
EntityExtractor
    ↓
FalkorDBClient
```

**Key Relationships:**
- **Used by:** MemoryProcessor (calls chunk_text() for all text inputs)
- **Depends on:** LangChain RecursiveCharacterTextSplitter, tiktoken tokenizer
- **Produces:** List[Chunk] objects consumed by OllamaEmbedder

---

## Class Definition

### Class Diagram

```
┌─────────────────────────────────────┐
│        SemanticChunker              │
├─────────────────────────────────────┤
│ - chunk_size: int                   │
│ - chunk_overlap: int                │
│ - min_chunk_size: int               │
│ - splitter: RecursiveCharacterTextSplitter │
│ - tokenizer: tiktoken.Encoding      │
├─────────────────────────────────────┤
│ + __init__(chunk_size, overlap)     │
│ + chunk_text(text) -> List[Chunk]   │
│ + chunk_code(code, lang) -> List[Chunk] │ (Phase 3)
│ + merge_small_chunks(chunks) -> List[Chunk] │
│ - _validate_input(text) -> None     │
│ - _count_tokens(text) -> int        │
└─────────────────────────────────────┘
```

### Full Class Signature

```python
from typing import List, Optional
from dataclasses import dataclass
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken


@dataclass
class Chunk:
    """
    Represents a semantic chunk of text.

    Attributes:
        text: Chunk content
        index: Position in original document (0-based)
        start_char: Character offset in original text
        end_char: End character offset
        metadata: Optional chunk-specific metadata
    """
    text: str
    index: int
    start_char: int
    end_char: int
    metadata: Optional[dict] = None


class SemanticChunker:
    """
    Intelligent text chunking with semantic boundary detection.

    Uses LangChain's RecursiveCharacterTextSplitter to split text at natural
    boundaries (paragraphs, sentences, words) while maintaining target chunk size.
    Includes configurable overlap to preserve context at chunk boundaries.

    Attributes:
        chunk_size: Target chunk size in tokens (default: 512)
        chunk_overlap: Overlap between chunks in tokens (default: 50)
        min_chunk_size: Minimum chunk size in tokens (default: 100)
        splitter: LangChain text splitter instance
        tokenizer: tiktoken encoding for token counting

    Example:
        ```python
        chunker = SemanticChunker(chunk_size=512, chunk_overlap=50)

        text = "Python is a programming language. " * 100
        chunks = chunker.chunk_text(text)

        print(f"Created {len(chunks)} chunks")
        for chunk in chunks[:3]:
            print(f"Chunk {chunk.index}: {len(chunk.text)} chars")
        ```
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100
    ) -> None:
        """
        Initialize SemanticChunker with size parameters.

        Args:
            chunk_size: Target chunk size in tokens (recommended: 256-1024)
            chunk_overlap: Overlap between chunks in tokens (recommended: 10-20% of chunk_size)
            min_chunk_size: Minimum acceptable chunk size in tokens

        Raises:
            ValueError: If chunk_size <= 0 or chunk_overlap >= chunk_size
            ValueError: If min_chunk_size >= chunk_size

        Example:
            ```python
            # Standard configuration (Phase 1)
            chunker = SemanticChunker(chunk_size=512, chunk_overlap=50)

            # Smaller chunks for fine-grained search
            chunker = SemanticChunker(chunk_size=256, chunk_overlap=25)

            # Larger chunks for narrative text
            chunker = SemanticChunker(chunk_size=1024, chunk_overlap=100)
            ```
        """

    def chunk_text(self, text: str) -> List[Chunk]:
        """
        Split text into semantic chunks with overlap.

        Algorithm:
        1. Validate input (non-empty, UTF-8, max 10MB)
        2. Split using RecursiveCharacterTextSplitter (paragraph → sentence → word boundaries)
        3. Create Chunk objects with metadata (index, char offsets)
        4. Merge chunks smaller than min_chunk_size with neighbors
        5. Return List[Chunk]

        Args:
            text: Input text to chunk (max 10,000,000 characters)

        Returns:
            List of Chunk objects, each containing:
            - text: chunk content
            - index: 0-based position
            - start_char: character offset in original text
            - end_char: end character offset
            - metadata: None (reserved for future use)

        Raises:
            ValidationError: If text is empty or exceeds max length
            ValidationError: If text contains non-UTF-8 characters
            ChunkingError: If chunking fails due to internal error

        Performance Target:
            - Small text (< 1KB): < 10ms
            - Medium text (< 100KB): < 50ms
            - Large text (< 1MB): < 200ms

        Example:
            ```python
            chunker = SemanticChunker()

            text = \"\"\"
            Python is a high-level programming language.

            It was created by Guido van Rossum in 1991.
            Python emphasizes code readability.
            \"\"\"

            chunks = chunker.chunk_text(text)

            # Chunks respect paragraph boundaries
            assert len(chunks) >= 1
            assert chunks[0].index == 0
            assert chunks[0].start_char == 0
            ```
        """

    def chunk_code(
        self,
        code: str,
        language: str = "python"
    ) -> List[Chunk]:
        """
        Split code using AST-based chunking (Phase 3 feature).

        Uses tree-sitter to parse code into AST, then chunks at function/class boundaries.
        Preserves structural integrity (complete functions, no mid-block splits).

        Args:
            code: Source code to chunk
            language: Programming language (python, javascript, typescript, go, rust)

        Returns:
            List of Chunk objects representing code blocks (functions, classes, methods)

        Raises:
            ValidationError: If code is empty or language unsupported
            ChunkingError: If AST parsing fails (syntax errors)
            NotImplementedError: In Phase 1-2 (not implemented yet)

        Performance Target:
            - Small file (< 10KB): < 50ms
            - Medium file (< 100KB): < 200ms

        Example (Phase 3):
            ```python
            chunker = SemanticChunker()

            code = \"\"\"
            def greet(name):
                return f"Hello, {name}!"

            class Greeter:
                def greet(self, name):
                    return greet(name)
            \"\"\"

            chunks = chunker.chunk_code(code, language="python")

            # Should create chunks for function and class
            assert len(chunks) == 2
            assert "def greet" in chunks[0].text
            assert "class Greeter" in chunks[1].text
            ```
        """

    def merge_small_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Merge chunks smaller than min_chunk_size with adjacent chunks.

        Small chunks (< min_chunk_size tokens) reduce retrieval quality and increase
        overhead. This method merges them with neighbors to maintain quality standards.

        Algorithm:
        1. Iterate through chunks
        2. If chunk < min_chunk_size tokens:
           - Merge with next chunk if available
           - Or merge with previous chunk if last
        3. Update metadata (char offsets, index)
        4. Return merged list

        Args:
            chunks: List of chunks to process

        Returns:
            List of chunks where all chunks >= min_chunk_size (except possibly last)

        Raises:
            ValueError: If chunks is empty

        Performance Target:
            - Execution time: O(n) where n = len(chunks)

        Example:
            ```python
            chunker = SemanticChunker(min_chunk_size=100)

            # Create chunks with one too small
            chunks = [
                Chunk(text="A" * 200, index=0, start_char=0, end_char=200),
                Chunk(text="B" * 50, index=1, start_char=200, end_char=250),  # Too small!
                Chunk(text="C" * 200, index=2, start_char=250, end_char=450),
            ]

            merged = chunker.merge_small_chunks(chunks)

            # Small chunk merged with neighbor
            assert len(merged) == 2
            assert all(len(c.text) >= 100 for c in merged)
            ```
        """

    def _validate_input(self, text: str) -> None:
        """
        Validate text input before chunking (private helper).

        Args:
            text: Text to validate

        Raises:
            ValidationError: If validation fails
        """

    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken (private helper).

        Args:
            text: Text to tokenize

        Returns:
            Token count (int)
        """
```

---

## Dependencies

### Component Dependencies

**Internal (from zapomni_core):**
- `Chunk` dataclass (from zapomni_core.models)
- `ValidationError`, `ChunkingError` exceptions (from zapomni_core.exceptions)

**None required from other zapomni modules** - SemanticChunker is self-contained.

### External Libraries

**Required:**
- `langchain>=0.1.0` - For RecursiveCharacterTextSplitter
  - Purpose: Semantic-aware text splitting with configurable separators
  - Why: Industry standard, well-tested, respects natural boundaries

- `tiktoken>=0.5.0` - For token counting
  - Purpose: Count tokens accurately using OpenAI's tokenizer (cl100k_base)
  - Why: Standard tokenizer, matches Ollama's token counting

**Phase 3 (Code Chunking):**
- `tree-sitter>=0.20.0` - AST parsing for code
- `tree-sitter-python>=0.20.0` - Python grammar
- `tree-sitter-javascript>=0.20.0` - JavaScript/TypeScript grammar

### Dependency Injection

SemanticChunker does NOT use dependency injection. All dependencies are instantiated internally:

```python
def __init__(self, chunk_size: int = 512, ...):
    self.tokenizer = tiktoken.get_encoding("cl100k_base")
    self.splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=self._count_tokens,
        separators=["\n\n", "\n", " ", ""]  # Paragraph → Line → Word
    )
```

**Rationale:**
- Simple, no external configuration needed
- All behavior controlled by constructor parameters
- Easy to test (no mocks needed, pure functions)

---

## State Management

### Attributes

**Configuration (immutable after __init__):**
- `chunk_size: int` - Target chunk size in tokens (default: 512)
  - Lifetime: Set at initialization, never changes
  - Purpose: Controls RecursiveCharacterTextSplitter chunk size

- `chunk_overlap: int` - Overlap between chunks in tokens (default: 50)
  - Lifetime: Set at initialization, never changes
  - Purpose: Ensures context preservation at boundaries

- `min_chunk_size: int` - Minimum acceptable chunk size (default: 100)
  - Lifetime: Set at initialization, never changes
  - Purpose: Quality threshold for merge_small_chunks()

**Internal State (private):**
- `splitter: RecursiveCharacterTextSplitter` - LangChain splitter instance
  - Lifetime: Created in __init__, reused for all chunk_text() calls
  - Purpose: Actual chunking logic

- `tokenizer: tiktoken.Encoding` - Tokenizer for counting tokens
  - Lifetime: Created in __init__, reused for all _count_tokens() calls
  - Purpose: Accurate token counting

### State Transitions

SemanticChunker is **stateless** - it has no mutable state:

```
Initial State (after __init__)
    ↓
Ready State (can call chunk_text() repeatedly)
    ↓
[No state changes - pure function behavior]
```

Each call to `chunk_text()` is **independent** - no side effects, no shared state.

### Thread Safety

**Thread-Safe:** ✅ Yes

**Reasoning:**
- No mutable state (all attributes are final after __init__)
- LangChain RecursiveCharacterTextSplitter is stateless
- tiktoken tokenizer is thread-safe
- Pure functional behavior (input → output, no side effects)

**Concurrency Support:**
- Multiple threads can call `chunk_text()` simultaneously
- No need for locks or synchronization
- Safe to share single SemanticChunker instance across threads

---

## Public Methods (Detailed)

### Method 1: `__init__`

**Signature:**
```python
def __init__(
    self,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    min_chunk_size: int = 100
) -> None
```

**Purpose:** Initialize chunker with size configuration parameters

**Parameters:**

- `chunk_size: int` (default: 512)
  - Description: Target chunk size in tokens
  - Constraints:
    - Must be > 0
    - Must be > chunk_overlap
    - Recommended range: 256-1024 tokens
  - Example: 512 (default), 256 (fine-grained), 1024 (narrative text)

- `chunk_overlap: int` (default: 50)
  - Description: Overlap between adjacent chunks in tokens
  - Constraints:
    - Must be >= 0
    - Must be < chunk_size
    - Recommended: 10-20% of chunk_size (e.g., 50-100 for chunk_size=512)
  - Example: 50 (default, ~10%), 100 (~20% for chunk_size=512)

- `min_chunk_size: int` (default: 100)
  - Description: Minimum acceptable chunk size after splitting
  - Constraints:
    - Must be > 0
    - Must be < chunk_size
    - Recommended: 20-50% of chunk_size
  - Example: 100 (default), 128 (stricter quality)

**Returns:** None (constructor)

**Raises:**
- `ValueError`: If chunk_size <= 0
- `ValueError`: If chunk_overlap >= chunk_size
- `ValueError`: If min_chunk_size >= chunk_size
- `ValueError`: If min_chunk_size <= 0

**Preconditions:** None (can always instantiate)

**Postconditions:**
- SemanticChunker instance ready to use
- Internal splitter and tokenizer initialized

**Algorithm Outline:**
```
1. Validate parameters (chunk_size > 0, chunk_overlap < chunk_size, etc.)
2. Initialize tiktoken tokenizer (cl100k_base encoding)
3. Create RecursiveCharacterTextSplitter with:
   - chunk_size = chunk_size
   - chunk_overlap = chunk_overlap
   - length_function = self._count_tokens
   - separators = ["\n\n", "\n", " ", ""] (paragraph → line → word)
4. Store parameters as instance attributes
```

**Edge Cases:**

1. **chunk_size = 0** → ValueError("chunk_size must be positive")
2. **chunk_overlap >= chunk_size** → ValueError("overlap must be less than chunk_size")
3. **min_chunk_size >= chunk_size** → ValueError("min_chunk_size must be less than chunk_size")
4. **Very large chunk_size (> 10000)** → Warning logged, allowed (user choice)

**Related Methods:**
- Called by: MemoryProcessor.__init__() (creates SemanticChunker instance)
- Calls: tiktoken.get_encoding(), RecursiveCharacterTextSplitter()

---

### Method 2: `chunk_text`

**Signature:**
```python
def chunk_text(self, text: str) -> List[Chunk]
```

**Purpose:** Split text into semantic chunks with overlap and metadata

**Parameters:**

- `text: str`
  - Description: Input text to chunk
  - Constraints:
    - Must not be empty (after .strip())
    - Must be valid UTF-8
    - Max length: 10,000,000 characters (~10MB)
    - Min length: 1 character
  - Example: "Python is a programming language created by Guido van Rossum."

**Returns:**
- Type: `List[Chunk]`
- Guarantees:
  - Length >= 1 (at least one chunk for valid input)
  - Chunks are ordered (chunk[i].index == i)
  - No gaps in character coverage (chunk[i].end_char == chunk[i+1].start_char)
  - All chunks meet min_chunk_size (except possibly last chunk)
  - Overlaps exist between adjacent chunks (~10-20%)

**Raises:**
- `ValidationError`: If text is empty
- `ValidationError`: If text exceeds 10,000,000 characters
- `ValidationError`: If text contains non-UTF-8 characters
- `ChunkingError`: If RecursiveCharacterTextSplitter fails (internal error)

**Preconditions:**
- SemanticChunker must be initialized (__init__ called)

**Postconditions:**
- No state changes (pure function)
- Returned chunks are immutable
- Total text coverage: union of all chunks covers entire input (accounting for overlap)

**Algorithm Outline:**
```
1. Validate input:
   - Check text is not empty (after strip)
   - Check text length <= 10,000,000 chars
   - Verify UTF-8 encoding

2. Split text using RecursiveCharacterTextSplitter:
   - splitter.split_text(text) → List[str]
   - Splits at paragraph (\n\n), then line (\n), then word (" ")
   - Respects chunk_size and chunk_overlap

3. Create Chunk objects:
   - For each split text chunk (index i):
     - Calculate start_char (sum of previous chunk lengths - overlaps)
     - Calculate end_char (start_char + len(chunk))
     - Create Chunk(text, index=i, start_char, end_char, metadata=None)

4. Merge small chunks:
   - Call merge_small_chunks(chunks)
   - Merges chunks < min_chunk_size with neighbors

5. Return List[Chunk]
```

**Edge Cases:**

1. **Empty text `""`** → ValidationError("Text cannot be empty")
2. **Whitespace-only `"   "`** → ValidationError("Text cannot be empty") (after strip)
3. **Single character `"A"`** → Returns [Chunk(text="A", index=0, start_char=0, end_char=1)]
4. **Text < chunk_size** → Returns single chunk
5. **Text exactly chunk_size** → Returns single chunk (no split needed)
6. **Text > chunk_size** → Multiple chunks with overlap
7. **Text = 10,000,000 chars (max)** → Allowed, chunks normally
8. **Text = 10,000,001 chars** → ValidationError("Text exceeds maximum length")
9. **Non-UTF-8 bytes** → ValidationError("Text must be valid UTF-8")
10. **Code as text (Phase 1)** → Treated as plain text (no AST parsing)

**Related Methods:**
- Called by: MemoryProcessor.add_memory() (main use case)
- Calls: _validate_input(), splitter.split_text(), merge_small_chunks(), _count_tokens()

---

### Method 3: `chunk_code` (Phase 3)

**Signature:**
```python
def chunk_code(self, code: str, language: str = "python") -> List[Chunk]
```

**Purpose:** Split code using AST-based chunking (preserves functions, classes)

**Parameters:**

- `code: str`
  - Description: Source code to chunk
  - Constraints: Same as chunk_text (non-empty, UTF-8, max 10MB)

- `language: str` (default: "python")
  - Description: Programming language for AST parsing
  - Constraints: Must be in ["python", "javascript", "typescript", "go", "rust"]
  - Example: "python", "javascript"

**Returns:**
- Type: `List[Chunk]`
- Chunks represent complete code units (functions, classes, methods)

**Raises:**
- `ValidationError`: If code is empty or language unsupported
- `ChunkingError`: If AST parsing fails (syntax errors in code)
- `NotImplementedError`: In Phase 1-2 (feature not implemented yet)

**Algorithm Outline (Phase 3):**
```
1. Validate code (same as chunk_text)
2. Validate language is supported
3. Parse code into AST using tree-sitter
4. Extract top-level nodes (functions, classes, imports)
5. For each node:
   - Extract node text
   - Create Chunk with metadata (node type, name)
6. Merge small nodes (< min_chunk_size) with neighbors
7. Return List[Chunk]
```

**Edge Cases:**
1. **Syntax error in code** → ChunkingError("Failed to parse: SyntaxError at line X")
2. **Empty file** → ValidationError("Code cannot be empty")
3. **Only imports (no functions)** → Return chunks for imports
4. **Very large function (> chunk_size)** → Split function internally (fallback to text chunking)

**Related Methods:**
- Called by: MemoryProcessor.add_memory() (when content_type="code")
- Calls: tree-sitter parser, merge_small_chunks()

---

### Method 4: `merge_small_chunks`

**Signature:**
```python
def merge_small_chunks(self, chunks: List[Chunk]) -> List[Chunk]
```

**Purpose:** Merge chunks smaller than min_chunk_size to maintain quality

**Parameters:**

- `chunks: List[Chunk]`
  - Description: List of chunks to process
  - Constraints: Must not be empty

**Returns:**
- Type: `List[Chunk]`
- Guarantees:
  - All chunks >= min_chunk_size (except possibly last chunk if can't merge)
  - Chunk order preserved
  - Total text coverage unchanged (no data loss)

**Raises:**
- `ValueError`: If chunks is empty list

**Algorithm Outline:**
```
1. If chunks is empty → raise ValueError
2. Initialize result = []
3. For each chunk (index i):
   - Count tokens in chunk.text
   - If tokens >= min_chunk_size:
     - Append to result (keep as-is)
   - Else (chunk too small):
     - If next chunk exists:
       - Merge chunk with next chunk
       - Update metadata (start_char from current, end_char from next)
       - Skip next chunk (already merged)
     - Else (last chunk):
       - If previous chunk exists:
         - Merge chunk with previous chunk (update previous)
       - Else (only one chunk and it's small):
         - Keep it anyway (log warning)
4. Update chunk indices (0, 1, 2, ...)
5. Return result
```

**Edge Cases:**
1. **Empty list `[]`** → ValueError("chunks cannot be empty")
2. **Single chunk < min_chunk_size** → Return as-is with warning log
3. **All chunks < min_chunk_size** → Merge all into one chunk
4. **Alternating large/small chunks** → Small merged with next large

**Performance Target:**
- Time complexity: O(n) where n = len(chunks)
- Space complexity: O(n) for result list

**Related Methods:**
- Called by: chunk_text(), chunk_code()
- Calls: _count_tokens()

---

## Error Handling

### Exceptions Defined

```python
# From zapomni_core.exceptions

class ValidationError(ZapomniCoreError):
    """Input validation failed."""
    pass

class ChunkingError(ZapomniCoreError):
    """Text chunking failed due to internal error."""
    pass
```

### Error Recovery

**Validation Errors (no recovery, fail fast):**
- Empty text → Raise ValidationError immediately
- Text too large → Raise ValidationError immediately
- Non-UTF-8 → Raise ValidationError immediately

**Chunking Errors (retry not applicable):**
- If RecursiveCharacterTextSplitter raises exception → Wrap in ChunkingError
- Log error details to structlog
- Propagate to caller (MemoryProcessor)

**No Retry Logic:**
- Chunking is deterministic (same input → same output)
- Retry won't fix invalid input
- Caller decides how to handle errors

### Error Propagation

```
SemanticChunker.chunk_text()
    ↓ raises ValidationError/ChunkingError
MemoryProcessor.add_memory()
    ↓ catches, logs, re-raises
MCP Tool (add_memory)
    ↓ catches, formats as MCP error response
User
```

---

## Usage Examples

### Basic Usage

```python
from zapomni_core.chunking import SemanticChunker

# Initialize with defaults
chunker = SemanticChunker()

# Simple text chunking
text = """
Python is a high-level programming language.

It was created by Guido van Rossum and first released in 1991.
Python emphasizes code readability and simplicity.

The language supports multiple programming paradigms including
procedural, object-oriented, and functional programming.
"""

chunks = chunker.chunk_text(text)

print(f"Created {len(chunks)} chunks")
for chunk in chunks:
    print(f"Chunk {chunk.index}: {chunk.text[:50]}...")
    print(f"  Offset: {chunk.start_char}-{chunk.end_char}")
```

### Advanced Usage (Custom Configuration)

```python
# Fine-grained chunking for technical documentation
fine_chunker = SemanticChunker(
    chunk_size=256,      # Smaller chunks
    chunk_overlap=50,    # ~20% overlap
    min_chunk_size=50    # Allow smaller chunks
)

# Narrative text with larger chunks
narrative_chunker = SemanticChunker(
    chunk_size=1024,     # Larger chunks for context
    chunk_overlap=100,   # ~10% overlap
    min_chunk_size=200   # Higher quality threshold
)

# Process technical doc
tech_text = "..." # Technical documentation
tech_chunks = fine_chunker.chunk_text(tech_text)

# Process narrative
story_text = "..." # Novel chapter
story_chunks = narrative_chunker.chunk_text(story_text)
```

### Error Handling Example

```python
from zapomni_core.chunking import SemanticChunker
from zapomni_core.exceptions import ValidationError, ChunkingError

chunker = SemanticChunker()

try:
    # This will raise ValidationError
    chunks = chunker.chunk_text("")
except ValidationError as e:
    print(f"Validation failed: {e}")
    # Handle gracefully (ask user for valid input)

try:
    # This will raise ValidationError (too large)
    huge_text = "A" * 20_000_000  # 20MB
    chunks = chunker.chunk_text(huge_text)
except ValidationError as e:
    print(f"Text too large: {e}")
    # Suggest splitting into smaller documents

try:
    # Normal operation
    text = "Valid text content"
    chunks = chunker.chunk_text(text)
    print(f"Success: {len(chunks)} chunks created")
except ChunkingError as e:
    print(f"Chunking failed: {e}")
    # Log error, notify user
```

### Integration with MemoryProcessor

```python
from zapomni_core import MemoryProcessor
from zapomni_core.chunking import SemanticChunker
from zapomni_db import FalkorDBClient

# MemoryProcessor creates SemanticChunker internally
processor = MemoryProcessor(
    db=FalkorDBClient(),
    chunk_size=512,      # Passed to SemanticChunker
    chunk_overlap=50
)

# Chunker is used automatically in add_memory()
memory_id = await processor.add_memory(
    text="Python is a great language for AI development."
)

# Internally:
# 1. processor.chunker.chunk_text(text) → chunks
# 2. processor.embedder.embed(chunks) → embeddings
# 3. processor.db.store(chunks, embeddings) → memory_id
```

---

## Testing Approach

### Unit Tests Required

**Basic Functionality:**
1. `test_init_defaults()` - Default parameters work
2. `test_init_custom_params()` - Custom parameters accepted
3. `test_init_invalid_chunk_size_raises()` - chunk_size <= 0 raises ValueError
4. `test_init_invalid_overlap_raises()` - overlap >= chunk_size raises ValueError
5. `test_chunk_text_success()` - Normal text chunked correctly
6. `test_chunk_text_respects_chunk_size()` - Chunks are ~chunk_size tokens
7. `test_chunk_text_has_overlap()` - Adjacent chunks overlap
8. `test_chunk_text_preserves_boundaries()` - Paragraph boundaries respected
9. `test_chunk_text_metadata_correct()` - Chunk indices and offsets correct

**Edge Cases:**
10. `test_chunk_text_empty_raises()` - Empty text raises ValidationError
11. `test_chunk_text_whitespace_only_raises()` - "   " raises ValidationError
12. `test_chunk_text_single_char()` - Single char returns single chunk
13. `test_chunk_text_below_chunk_size()` - Small text returns single chunk
14. `test_chunk_text_exactly_chunk_size()` - Text == chunk_size handled correctly
15. `test_chunk_text_max_length()` - 10MB text accepted
16. `test_chunk_text_too_large_raises()` - 10MB+1 raises ValidationError
17. `test_chunk_text_non_utf8_raises()` - Binary data raises ValidationError

**Merge Small Chunks:**
18. `test_merge_small_chunks_no_small()` - All chunks large → no merge
19. `test_merge_small_chunks_one_small()` - One small chunk merged
20. `test_merge_small_chunks_all_small()` - All small → merge all
21. `test_merge_small_chunks_empty_raises()` - Empty list raises ValueError
22. `test_merge_small_chunks_preserves_order()` - Order maintained after merge

**Performance:**
23. `test_chunk_text_performance_small()` - < 1KB in < 10ms
24. `test_chunk_text_performance_medium()` - < 100KB in < 50ms

### Mocking Strategy

**No Mocking Needed** - SemanticChunker is self-contained:
- LangChain RecursiveCharacterTextSplitter is real (integration test)
- tiktoken tokenizer is real (integration test)
- No external API calls
- No database dependencies

**Test Data:**
- Use fixtures for sample texts (tech doc, narrative, code)
- Generate synthetic text (repeated patterns)
- Real-world samples (Wikipedia paragraphs, GitHub READMEs)

### Integration Tests

**With MemoryProcessor:**
- Test that chunker is used correctly in full pipeline
- Verify chunks are embedded and stored properly

**Real-World Texts:**
- Test with actual documents (PDFs converted to text)
- Verify chunk quality (no mid-sentence splits)

### Test Example

```python
# tests/unit/test_semantic_chunker.py

import pytest
from zapomni_core.chunking import SemanticChunker, Chunk
from zapomni_core.exceptions import ValidationError, ChunkingError


class TestSemanticChunkerInit:
    def test_init_defaults(self):
        """Test initialization with default parameters."""
        chunker = SemanticChunker()

        assert chunker.chunk_size == 512
        assert chunker.chunk_overlap == 50
        assert chunker.min_chunk_size == 100
        assert chunker.splitter is not None
        assert chunker.tokenizer is not None

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        chunker = SemanticChunker(
            chunk_size=256,
            chunk_overlap=25,
            min_chunk_size=50
        )

        assert chunker.chunk_size == 256
        assert chunker.chunk_overlap == 25
        assert chunker.min_chunk_size == 50

    def test_init_invalid_chunk_size_raises(self):
        """Test that chunk_size <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            SemanticChunker(chunk_size=0)

        with pytest.raises(ValueError, match="chunk_size must be positive"):
            SemanticChunker(chunk_size=-10)

    def test_init_invalid_overlap_raises(self):
        """Test that overlap >= chunk_size raises ValueError."""
        with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
            SemanticChunker(chunk_size=100, chunk_overlap=100)

        with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
            SemanticChunker(chunk_size=100, chunk_overlap=150)


class TestSemanticChunkerChunkText:
    def test_chunk_text_success(self):
        """Test basic chunking with valid input."""
        chunker = SemanticChunker(chunk_size=100, chunk_overlap=10)

        text = "Python is great. " * 50  # ~850 chars
        chunks = chunker.chunk_text(text)

        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)
        assert chunks[0].index == 0
        assert chunks[0].start_char == 0

    def test_chunk_text_empty_raises(self):
        """Test that empty text raises ValidationError."""
        chunker = SemanticChunker()

        with pytest.raises(ValidationError, match="cannot be empty"):
            chunker.chunk_text("")

    def test_chunk_text_whitespace_only_raises(self):
        """Test that whitespace-only text raises ValidationError."""
        chunker = SemanticChunker()

        with pytest.raises(ValidationError, match="cannot be empty"):
            chunker.chunk_text("   \n\n\t  ")

    def test_chunk_text_single_char(self):
        """Test chunking single character."""
        chunker = SemanticChunker()

        chunks = chunker.chunk_text("A")

        assert len(chunks) == 1
        assert chunks[0].text == "A"
        assert chunks[0].start_char == 0
        assert chunks[0].end_char == 1

    def test_chunk_text_respects_boundaries(self):
        """Test that paragraph boundaries are respected."""
        chunker = SemanticChunker(chunk_size=50, chunk_overlap=5)

        text = """Paragraph one.

Paragraph two.

Paragraph three."""

        chunks = chunker.chunk_text(text)

        # Should split at paragraph boundaries (\n\n)
        assert len(chunks) >= 2
        # First chunk should end at paragraph boundary
        assert chunks[0].text.strip().endswith("one.")

    @pytest.mark.performance
    def test_chunk_text_performance_small(self, benchmark):
        """Test performance with small input (< 1KB)."""
        chunker = SemanticChunker()
        text = "A" * 500  # 500 chars

        result = benchmark(chunker.chunk_text, text)

        assert len(result) >= 1
        # Benchmark will automatically verify < 10ms


class TestSemanticChunkerMergeSmallChunks:
    def test_merge_small_chunks_no_small(self):
        """Test that large chunks are not merged."""
        chunker = SemanticChunker(min_chunk_size=10)

        chunks = [
            Chunk(text="A" * 50, index=0, start_char=0, end_char=50),
            Chunk(text="B" * 50, index=1, start_char=50, end_char=100),
        ]

        merged = chunker.merge_small_chunks(chunks)

        assert len(merged) == 2  # No merge
        assert merged[0].text == chunks[0].text

    def test_merge_small_chunks_one_small(self):
        """Test that one small chunk is merged."""
        chunker = SemanticChunker(min_chunk_size=50)

        chunks = [
            Chunk(text="A" * 100, index=0, start_char=0, end_char=100),
            Chunk(text="B" * 10, index=1, start_char=100, end_char=110),  # Small!
            Chunk(text="C" * 100, index=2, start_char=110, end_char=210),
        ]

        merged = chunker.merge_small_chunks(chunks)

        assert len(merged) == 2  # Small chunk merged
        # Check that B was merged with C
        assert "B" in merged[1].text and "C" in merged[1].text

    def test_merge_small_chunks_empty_raises(self):
        """Test that empty list raises ValueError."""
        chunker = SemanticChunker()

        with pytest.raises(ValueError, match="cannot be empty"):
            chunker.merge_small_chunks([])
```

---

## Performance Considerations

### Time Complexity

**chunk_text():**
- Tokenization: O(n) where n = len(text)
- RecursiveCharacterTextSplitter: O(n)
- Chunk object creation: O(k) where k = number of chunks
- merge_small_chunks: O(k)
- **Total: O(n) linear with text length**

**merge_small_chunks():**
- Iterate chunks: O(k)
- Token counting per chunk: O(m) where m = avg chunk length
- **Total: O(k * m) ≈ O(n)**

**chunk_code() (Phase 3):**
- AST parsing: O(n log n) (tree-sitter)
- Node extraction: O(k)
- **Total: O(n log n)**

### Space Complexity

**chunk_text():**
- Input text: n bytes
- LangChain split_text() output: n bytes (reuses text slices)
- Chunk objects: O(k) where k = number of chunks (~100 bytes each)
- **Total: O(n) for text + O(k) for metadata ≈ O(n)**

**Memory Efficiency:**
- Chunks store references to original text (no duplication if using slices)
- Minimal metadata overhead (4 integers + 1 string per chunk)

### Optimization Opportunities

**Current (Phase 1):**
- Already optimized (LangChain RecursiveCharacterTextSplitter is efficient)
- Token counting cached in LangChain

**Future:**
1. **Parallel chunking** for very large documents (> 1MB)
   - Split into sections, chunk in parallel, merge results
   - Trade-off: More complex, boundary handling
2. **Streaming chunking** for infinite/large inputs
   - Chunk incrementally without loading entire text
   - Trade-off: Can't optimize global boundaries
3. **Adaptive chunk_size** based on content type
   - Technical docs → smaller chunks
   - Narrative text → larger chunks
   - Trade-off: Adds complexity, heuristics needed

---

## Performance Requirements

**Latency Targets:**

| Input Size | Target | Max Allowed |
|------------|--------|-------------|
| < 1KB | < 10ms | 50ms |
| < 10KB | < 20ms | 100ms |
| < 100KB | < 50ms | 200ms |
| < 1MB | < 200ms | 500ms |
| 10MB (max) | < 2000ms | 5000ms |

**Throughput:**
- Sequential chunking: 50-100 documents/sec (small docs < 10KB)
- Large docs (1MB): 5-10 documents/sec

**Resource Usage:**
- Memory: O(n) where n = input size (~2x input size including metadata)
- CPU: Single-threaded (no parallelism in Phase 1)

---

## References

**Module Spec:**
- [zapomni_core_module.md](../level1/zapomni_core_module.md) - Parent module specification

**Related Components:**
- OllamaEmbedder (next in pipeline, consumes Chunk objects)
- MemoryProcessor (orchestrates chunking + embedding + storage)

**External Documentation:**
- LangChain RecursiveCharacterTextSplitter: https://python.langchain.com/docs/modules/data_connection/document_transformers/
- tiktoken: https://github.com/openai/tiktoken
- tree-sitter (Phase 3): https://tree-sitter.github.io/tree-sitter/

---

**Document Status:** Draft v1.0
**Created:** 2025-11-23
**Author:** Goncharenko Anton aka alienxs2
**Copyright:** Copyright (c) 2025 Goncharenko Anton aka alienxs2
**License:** MIT License

**Ready for Review:** Yes ✅
**Next Steps:**
1. Review and approve this component spec
2. Create function-level specs for each public method (Level 3)
3. Implement SemanticChunker class
4. Write tests (24 unit tests defined above)
