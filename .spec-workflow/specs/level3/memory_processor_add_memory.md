# MemoryProcessor.add_memory() - Function Specification

**Level:** 3 (Function)
**Component:** MemoryProcessor
**Module:** zapomni_core
**Author:** Goncharenko Anton aka alienxs2
**Status:** Draft
**Version:** 1.0
**Created:** 2025-11-23

---

## Function Signature

```python
async def add_memory(
    self,
    text: str,
    metadata: Optional[Dict[str, Any]] = None
) -> MemoryResult:
    """
    Add memory to system with full 6-stage processing pipeline.

    This is the main entry point for adding new memories to the Zapomni knowledge graph.
    It orchestrates the complete end-to-end pipeline: validation → chunking → embedding
    generation → entity extraction (Phase 2) → storage → result return.

    The pipeline is transactional - if any stage fails, no data is stored in the database.
    All errors are caught, logged with context, and re-raised to the caller for handling.

    Args:
        text: Text content to remember (natural language, code, or structured data)
            - Constraints: Non-empty after strip, max 10,000,000 chars (~10MB)
            - Encoding: Must be valid UTF-8
            - Content: Treated as plain text in Phase 1 (no format-specific parsing)

        metadata: Optional metadata dictionary (tags, source, author, custom fields)
            - Structure: Flat dict with string keys, JSON-serializable values
            - Reserved keys: "memory_id", "timestamp", "chunks" (auto-populated)
            - Common fields:
                * "tags": list[str] - categorization tags
                * "source": str - origin identifier (e.g., "user", "api", "import")
                * "date": str - ISO 8601 date (e.g., "2025-11-23")
                * "author": str - creator identifier

    Returns:
        MemoryResult object containing:
            - id: str - UUID v4 identifying the stored memory
            - chunks_created: int - Number of chunks created from text
            - processing_time_ms: int - Total processing time in milliseconds

    Raises:
        ValidationError: If text empty/too large/non-UTF-8, or metadata has reserved keys
        ChunkingError: If semantic chunking fails (internal error, should not occur)
        EmbeddingError: If embedding generation fails (Ollama unavailable/offline)
        ExtractionError: If entity extraction fails (Phase 2, if enabled)
        StorageError: If FalkorDB storage fails (connection error, transaction failed)

    Performance Target:
        - Small input (< 1KB): < 100ms (P95)
        - Medium input (< 10KB): < 300ms (P95)
        - Large input (< 100KB): < 500ms (P95)
        - Maximum allowed: < 1000ms

    Example:
        ```python
        # Initialize processor
        processor = MemoryProcessor(
            db_client=FalkorDBClient(),
            chunker=SemanticChunker(),
            embedder=OllamaEmbedder()
        )

        # Simple add without metadata
        result = await processor.add_memory(
            text="Python is a high-level programming language created by Guido van Rossum."
        )
        print(f"Stored: {result.id}, Chunks: {result.chunks_created}, Time: {result.processing_time_ms}ms")

        # Add with metadata
        result = await processor.add_memory(
            text="Django is a Python web framework released in 2005.",
            metadata={
                "tags": ["python", "django", "web"],
                "source": "wikipedia",
                "date": "2025-11-23",
                "author": "user123"
            }
        )

        # Error handling
        try:
            result = await processor.add_memory("")
        except ValidationError as e:
            print(f"Invalid input: {e}")
        except EmbeddingError as e:
            print(f"Embedding failed (check Ollama): {e}")
        except StorageError as e:
            print(f"Storage failed (check DB): {e}")
        ```
    """
```

---

## Purpose & Context

### What It Does

Processes a new memory through the complete 6-stage pipeline:

1. **Validation Stage**: Validates text and metadata inputs against constraints
2. **Chunking Stage**: Splits text into semantic chunks using SemanticChunker
3. **Embedding Stage**: Generates vector embeddings for all chunks via OllamaEmbedder
4. **Extraction Stage**: Extracts entities and relationships (Phase 2, if enabled)
5. **Storage Stage**: Stores chunks, embeddings, metadata, and entities in FalkorDB (transactional)
6. **Return Stage**: Returns MemoryResult with ID, chunk count, and processing time

The function ensures atomicity - either all data is stored successfully, or nothing is stored (transaction rollback on error).

### Why It Exists

This is the **primary API** for adding memories to Zapomni. It provides:

- **Single Entry Point**: Unified interface for all memory addition (MCP tools, API, CLI all use this)
- **End-to-End Orchestration**: Coordinates all specialized components (chunker, embedder, extractor, DB)
- **Error Handling**: Centralizes error handling for the entire pipeline
- **Instrumentation**: Logs all stages, tracks timing, provides observability

### When To Use

Call this function whenever you need to add a new memory to the knowledge graph:

- **User Input**: When user provides text via MCP tool or API
- **Batch Import**: When importing documents or datasets
- **Programmatic Storage**: When other components need to store processed information

### When NOT To Use

Do NOT use this function for:

- **Updating Existing Memory**: Use `update_memory()` (Phase 2) instead
- **Searching Memories**: Use `search_memory()` instead
- **Retrieving Statistics**: Use `get_stats()` instead
- **Building Knowledge Graph**: Use `build_knowledge_graph()` (Phase 2) instead

---

## Parameters (Detailed)

### text: str

**Type:** `str`

**Purpose:** The content to be stored as a memory (natural language text, code, or structured data)

**Constraints:**
- Must not be empty after `strip()` (whitespace-only strings rejected)
- Must be valid UTF-8 encoding (non-UTF-8 bytes rejected)
- Maximum length: `config.max_text_length` (default: 10,000,000 characters ≈ 10MB)
- Minimum length: 1 character (after strip)
- No format restrictions: can be plain text, Markdown, code, JSON, etc.

**Validation Algorithm:**
```python
# Step 1: Strip whitespace
text_stripped = text.strip()

# Step 2: Check non-empty
if not text_stripped:
    raise ValidationError("Text cannot be empty or whitespace-only")

# Step 3: Check UTF-8 encoding
try:
    text.encode('utf-8')
except UnicodeEncodeError as e:
    raise ValidationError(f"Text must be valid UTF-8: {e}")

# Step 4: Check length
if len(text) > self.config.max_text_length:
    raise ValidationError(
        f"Text exceeds maximum length ({len(text)} > {self.config.max_text_length})"
    )
```

**Examples:**

- **Valid:**
  - `"Python is a programming language."` (normal text)
  - `"Short"` (minimum valid input)
  - `"x" * 10_000_000` (at max length)
  - `"def hello():\n    print('Hi')"` (code)
  - `'{"key": "value"}'` (JSON)
  - `"Привет мир 🌍"` (Unicode/emoji)

- **Invalid:**
  - `""` (empty string) → ValidationError: "Text cannot be empty"
  - `"   "` (whitespace only) → ValidationError: "Text cannot be empty"
  - `"x" * 10_000_001` (exceeds max) → ValidationError: "Text exceeds maximum length"
  - `b'\xff\xfe'` (non-UTF-8 bytes) → ValidationError: "Text must be valid UTF-8"

---

### metadata: Optional[Dict[str, Any]]

**Type:** `Optional[Dict[str, Any]]`

**Purpose:** Optional metadata to attach to the memory (tags, source, date, custom fields)

**Default:** `None` (treated as empty dict `{}` internally)

**Structure (when provided):**
```python
{
    # Common fields (recommended but optional)
    "tags": ["tag1", "tag2"],           # List of categorization tags
    "source": "user",                    # Origin identifier
    "date": "2025-11-23",               # ISO 8601 date
    "author": "username",                # Creator identifier

    # Custom fields (any JSON-serializable values)
    "custom_field": "any value",
    "priority": 5,
    "project_id": "proj-123"
}
```

**Constraints:**
- All keys must be strings
- All values must be JSON-serializable (str, int, float, bool, list, dict, None)
- Reserved keys (auto-populated by system):
  - `"memory_id"` - Will be overwritten with UUID
  - `"timestamp"` - Will be overwritten with current time
  - `"chunks"` - Will be populated with chunk count
- No maximum size limit on metadata dict (but keep reasonable, < 1KB recommended)

**Validation Algorithm:**
```python
# Step 1: Handle None
if metadata is None:
    metadata = {}

# Step 2: Check type
if not isinstance(metadata, dict):
    raise ValidationError(f"metadata must be dict, got {type(metadata)}")

# Step 3: Check keys are strings
for key in metadata.keys():
    if not isinstance(key, str):
        raise ValidationError(f"metadata keys must be strings, got {type(key)}")

# Step 4: Check for reserved keys
reserved_keys = {"memory_id", "timestamp", "chunks"}
for key in reserved_keys:
    if key in metadata:
        raise ValidationError(
            f"Reserved metadata key '{key}' will be overwritten. "
            f"Please use a different key name."
        )

# Step 5: Check JSON-serializable
import json
try:
    json.dumps(metadata)
except (TypeError, ValueError) as e:
    raise ValidationError(f"metadata must be JSON-serializable: {e}")
```

**Examples:**

- **Valid:**
  - `None` (default, becomes `{}`)
  - `{}` (empty dict, allowed)
  - `{"tags": ["python"]}` (minimal)
  - `{"tags": ["python", "web"], "source": "api", "date": "2025-11-23"}` (typical)
  - `{"custom": "value", "number": 42, "list": [1, 2, 3]}` (complex)

- **Invalid:**
  - `"not_a_dict"` → ValidationError: "metadata must be dict"
  - `{123: "value"}` → ValidationError: "metadata keys must be strings"
  - `{"memory_id": "x"}` → ValidationError: "Reserved metadata key 'memory_id'"
  - `{"timestamp": "x"}` → ValidationError: "Reserved metadata key 'timestamp'"
  - `{"func": lambda x: x}` → ValidationError: "metadata must be JSON-serializable"

---

## Return Value

**Type:** `MemoryResult`

**Structure:**
```python
@dataclass
class MemoryResult:
    """Result of memory addition operation."""
    id: str                    # UUID v4 identifying the stored memory
    chunks_created: int        # Number of chunks created from text
    processing_time_ms: int    # Total processing time in milliseconds
```

**Field Details:**

- **id: str**
  - Format: UUID v4 (e.g., `"550e8400-e29b-41d4-a716-446655440000"`)
  - Uniqueness: Guaranteed unique across all memories
  - Stability: Same ID for this memory forever (idempotent if text/metadata unchanged)
  - Generated: Via `uuid.uuid4()` in `_create_memory_id()` helper

- **chunks_created: int**
  - Range: 1 to N (at least 1 chunk always created)
  - Typical: 1-10 chunks for normal text (< 10KB)
  - Large texts: Can be 100+ chunks for very large documents
  - Purpose: Indicates granularity of storage (useful for debugging/monitoring)

- **processing_time_ms: int**
  - Unit: Milliseconds
  - Range: 10ms to 1000ms (target < 500ms for P95)
  - Measured: From function entry to return (includes all stages)
  - Purpose: Performance monitoring and SLA tracking

**Success Example:**
```python
MemoryResult(
    id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    chunks_created=3,
    processing_time_ms=156
)
```

**Interpretation:**
- Memory stored successfully with ID `a1b2c3d4...`
- Text was split into 3 semantic chunks
- Total processing took 156ms (within target)

---

## Exceptions

### ValidationError

**When Raised:**
- Text is empty or whitespace-only
- Text exceeds `max_text_length`
- Text has non-UTF-8 encoding
- Metadata is not a dict
- Metadata has non-string keys
- Metadata contains reserved keys (`memory_id`, `timestamp`, `chunks`)
- Metadata is not JSON-serializable

**Message Format:**
```python
f"Validation failed for {field_name}: {specific_reason}"
```

**Examples:**
```python
ValidationError("Text cannot be empty or whitespace-only")
ValidationError("Text exceeds maximum length (10000001 > 10000000)")
ValidationError("Text must be valid UTF-8: 'utf-8' codec can't encode...")
ValidationError("Reserved metadata key 'memory_id' will be overwritten")
ValidationError("metadata must be JSON-serializable: Object of type function is not...")
```

**Recovery:** No automatic recovery. Caller must fix input and retry.

---

### ChunkingError

**When Raised:**
- SemanticChunker fails to chunk text (internal error, rare)
- Chunking algorithm raises exception (e.g., out of memory for huge text)

**Message Format:**
```python
f"Chunking failed: {original_error_message}"
```

**Example:**
```python
ChunkingError("Chunking failed: Out of memory while processing 10MB text")
```

**Recovery:** Internal error, should not occur in normal operation. Log and escalate.

---

### EmbeddingError

**When Raised:**
- Ollama service is unavailable (connection refused, timeout)
- Ollama model not downloaded (e.g., `nomic-embed-text` not pulled)
- Ollama API returns error (e.g., invalid request)
- Embedding generation timeout (> 30 seconds per batch)

**Message Format:**
```python
f"Embedding generation failed: {original_error_message}"
```

**Examples:**
```python
EmbeddingError("Embedding generation failed: Connection refused to http://localhost:11434")
EmbeddingError("Embedding generation failed: Model 'nomic-embed-text' not found")
EmbeddingError("Embedding generation failed: Request timeout after 30s")
```

**Recovery:**
- **Automatic Retry**: OllamaEmbedder retries 3x with exponential backoff
- **Fallback**: If configured, fall back to sentence-transformers (local embedding)
- **Final Failure**: If all retries/fallbacks fail, raise EmbeddingError to caller

**User Action:**
- Check Ollama service is running: `systemctl status ollama` or `docker ps | grep ollama`
- Check model is downloaded: `ollama list | grep nomic-embed-text`
- Pull model if missing: `ollama pull nomic-embed-text`
- Check network connectivity to Ollama host

---

### ExtractionError (Phase 2 only)

**When Raised:**
- EntityExtractor fails to extract entities (Phase 2, if `config.enable_extraction=True`)
- SpaCy model not loaded
- LLM API error during relationship extraction

**Message Format:**
```python
f"Entity extraction failed: {original_error_message}"
```

**Example:**
```python
ExtractionError("Entity extraction failed: SpaCy model 'en_core_web_sm' not found")
```

**Recovery:**
- **Phase 1**: Not applicable (entity extraction not implemented)
- **Phase 2**: Log warning and continue WITHOUT entities (graceful degradation)
- **Rationale**: Entity extraction is enhancement, not critical for core functionality

---

### StorageError

**When Raised:**
- FalkorDB connection error (Redis/FalkorDB not running)
- Transaction rollback (e.g., constraint violation, out of memory)
- Query execution error (e.g., invalid Cypher syntax, internal DB error)
- Connection pool exhausted

**Message Format:**
```python
f"Storage failed: {original_error_message}"
```

**Examples:**
```python
StorageError("Storage failed: Connection refused to Redis at localhost:6379")
StorageError("Storage failed: Transaction rolled back due to constraint violation")
StorageError("Storage failed: FalkorDB out of memory")
```

**Recovery:**
- **Automatic Retry**: FalkorDBClient retries 3x with exponential backoff for transient errors
- **Transaction Rollback**: All changes rolled back (atomicity preserved)
- **Final Failure**: If all retries fail, raise StorageError to caller

**User Action:**
- Check FalkorDB/Redis service is running: `redis-cli ping` or `docker ps | grep falkordb`
- Check disk space: `df -h`
- Check memory: `free -h`
- Check connection limits: FalkorDB/Redis config

---

## Algorithm (Detailed Pseudocode)

```
FUNCTION add_memory(text: str, metadata: Optional[Dict]) -> MemoryResult:
    # ═══════════════════════════════════════════════════════════════
    # STAGE 0: INITIALIZATION
    # ═══════════════════════════════════════════════════════════════
    start_time = current_time_milliseconds()

    LOG(level=INFO, message=f"Starting add_memory: text_length={len(text)}")

    # ═══════════════════════════════════════════════════════════════
    # STAGE 1: VALIDATION
    # ═══════════════════════════════════════════════════════════════
    TRY:
        # 1.1: Validate text
        CALL self._validate_text(text)
        # Raises ValidationError if:
        #   - text is empty after strip
        #   - text exceeds max_text_length
        #   - text is not valid UTF-8

        # 1.2: Validate metadata
        CALL self._validate_metadata(metadata)
        # Raises ValidationError if:
        #   - metadata is not dict
        #   - metadata has non-string keys
        #   - metadata has reserved keys
        #   - metadata is not JSON-serializable

        # 1.3: Generate memory ID
        memory_id = CALL self._create_memory_id()
        # Returns: UUID v4 string

        # 1.4: Populate metadata
        IF metadata is None:
            metadata = {}
        metadata["memory_id"] = memory_id
        metadata["timestamp"] = current_timestamp_iso8601()

        LOG(level=INFO, message=f"Validation passed: memory_id={memory_id}")

    CATCH ValidationError as e:
        LOG(level=ERROR, message=f"Validation failed: {e}")
        RAISE e  # Propagate to caller

    # ═══════════════════════════════════════════════════════════════
    # STAGE 2: CHUNKING
    # ═══════════════════════════════════════════════════════════════
    TRY:
        LOG(level=INFO, message="Starting chunking stage")

        chunks = AWAIT self._process_chunks(text)
        # Returns: List[Chunk]
        # Chunk = dataclass(text: str, start_idx: int, end_idx: int, metadata: dict)

        chunks_count = len(chunks)
        metadata["chunks"] = chunks_count

        LOG(level=INFO, message=f"Chunking complete: {chunks_count} chunks created")

    CATCH ChunkingError as e:
        LOG(level=ERROR, message=f"Chunking failed: {e}")
        RAISE e  # Propagate to caller

    # ═══════════════════════════════════════════════════════════════
    # STAGE 3: EMBEDDING GENERATION
    # ═══════════════════════════════════════════════════════════════
    TRY:
        LOG(level=INFO, message="Starting embedding generation")

        embeddings = AWAIT self._generate_embeddings(chunks)
        # Returns: List[List[float]] (one embedding per chunk)
        # Each embedding is 768-dimensional vector (for nomic-embed-text)

        # Verify embeddings count matches chunks count
        ASSERT len(embeddings) == len(chunks), "Embeddings count mismatch"

        LOG(level=INFO, message=f"Embeddings generated: {len(embeddings)} vectors")

    CATCH EmbeddingError as e:
        LOG(level=ERROR, message=f"Embedding generation failed: {e}")
        RAISE e  # Propagate to caller

    # ═══════════════════════════════════════════════════════════════
    # STAGE 4: ENTITY EXTRACTION (Phase 2, Optional)
    # ═══════════════════════════════════════════════════════════════
    entities = None
    relationships = None

    IF self.config.enable_extraction AND self.extractor is not None:
        TRY:
            LOG(level=INFO, message="Starting entity extraction (Phase 2)")

            entities, relationships = AWAIT self._extract_entities(text, chunks)
            # Returns: (List[Entity], List[Relationship])

            LOG(level=INFO, message=f"Extraction complete: {len(entities)} entities, {len(relationships)} relationships")

        CATCH ExtractionError as e:
            # GRACEFUL DEGRADATION: Log warning, continue without entities
            LOG(level=WARNING, message=f"Entity extraction failed (continuing without entities): {e}")
            entities = None
            relationships = None
    ELSE:
        LOG(level=DEBUG, message="Entity extraction skipped (Phase 1 or disabled)")

    # ═══════════════════════════════════════════════════════════════
    # STAGE 5: STORAGE (TRANSACTIONAL)
    # ═══════════════════════════════════════════════════════════════
    TRY:
        LOG(level=INFO, message="Starting storage stage (transactional)")

        stored_id = AWAIT self._store_memory(
            memory_id=memory_id,
            chunks=chunks,
            embeddings=embeddings,
            metadata=metadata,
            entities=entities,           # Phase 2, None in Phase 1
            relationships=relationships   # Phase 2, None in Phase 1
        )
        # Returns: memory_id (same as input, for consistency)

        # Verify stored ID matches generated ID
        ASSERT stored_id == memory_id, "Storage ID mismatch"

        LOG(level=INFO, message=f"Storage complete: memory_id={memory_id}")

    CATCH StorageError as e:
        LOG(level=ERROR, message=f"Storage failed (transaction rolled back): {e}")
        RAISE e  # Propagate to caller

    # ═══════════════════════════════════════════════════════════════
    # STAGE 6: RETURN RESULT
    # ═══════════════════════════════════════════════════════════════
    end_time = current_time_milliseconds()
    processing_time_ms = end_time - start_time

    result = MemoryResult(
        id=memory_id,
        chunks_created=chunks_count,
        processing_time_ms=processing_time_ms
    )

    LOG(level=INFO, message=f"add_memory complete: {result}")

    RETURN result

END FUNCTION
```

---

## Preconditions

Before calling `add_memory()`, ensure:

1. **✅ MemoryProcessor Initialized**
   - `__init__()` called successfully
   - All required dependencies injected (db_client, chunker, embedder)
   - Config validated (if provided)

2. **✅ Dependencies Available**
   - `db_client` connected to FalkorDB/Redis
   - `chunker` initialized and ready
   - `embedder` can connect to Ollama (or fallback configured)
   - `extractor` loaded (if Phase 2 and `enable_extraction=True`)

3. **✅ External Services Running**
   - FalkorDB/Redis accessible at configured host:port
   - Ollama service running at configured host (default: `http://localhost:11434`)
   - Embedding model downloaded in Ollama (e.g., `nomic-embed-text`)

4. **✅ Resources Available**
   - Sufficient memory for chunking and embedding (depends on text size)
   - Database has available connections (connection pool not exhausted)
   - Disk space available for storage

**Verification:**
```python
# Minimal precondition check (recommended before first call)
async def verify_preconditions(processor: MemoryProcessor):
    # Check DB connection
    await processor.db_client.ping()  # Raises if DB unavailable

    # Check Ollama connection
    await processor.embedder.health_check()  # Raises if Ollama unavailable

    # Check model available
    models = await processor.embedder.list_models()
    assert "nomic-embed-text" in models, "Model not downloaded"
```

---

## Postconditions

After `add_memory()` returns successfully:

1. **✅ Memory Stored in Database**
   - Memory node created in FalkorDB with unique `memory_id`
   - All chunks stored as separate nodes linked to memory
   - Embeddings stored in vector index (searchable)
   - Metadata persisted (tags, source, timestamp, etc.)
   - Entities and relationships stored (if Phase 2)

2. **✅ Atomicity Guaranteed**
   - Either ALL data stored, or NOTHING stored (transaction)
   - No partial data in database on error

3. **✅ Searchable**
   - Memory immediately searchable via `search_memory()`
   - Vector similarity search functional
   - Metadata filters work

4. **✅ State Unchanged**
   - MemoryProcessor instance state not modified
   - No side effects on dependencies (chunker, embedder still usable)
   - Resources released (connections returned to pool)

5. **✅ Logged**
   - All stages logged with INFO level
   - Performance metrics captured (processing_time_ms)
   - Errors logged with ERROR level (if raised)

**Verification:**
```python
# Verify postcondition: memory retrievable
result = await processor.add_memory("Test memory")
memory_id = result.id

# Search for added memory
search_results = await processor.search_memory("Test memory", limit=1)
assert len(search_results) > 0
assert search_results[0].memory_id == memory_id
```

---

## Edge Cases & Handling

### Edge Case 1: Empty Text After Strip

**Scenario:** User passes text that is empty or whitespace-only

**Input:**
```python
text = ""           # Empty string
# OR
text = "   \n\t  "  # Whitespace only
```

**Expected Behavior:**
```python
raise ValidationError("Text cannot be empty or whitespace-only")
```

**Rationale:** Storing empty memories wastes resources and has no semantic value.

**Test Scenario:**
```python
def test_add_memory_empty_text_raises():
    processor = MemoryProcessor(...)

    with pytest.raises(ValidationError, match="empty or whitespace"):
        await processor.add_memory("")

    with pytest.raises(ValidationError, match="empty or whitespace"):
        await processor.add_memory("   \n\t  ")
```

---

### Edge Case 2: Text Exactly at Max Length

**Scenario:** Text is exactly `max_text_length` (10,000,000 chars)

**Input:**
```python
text = "x" * 10_000_000  # Exactly at limit
```

**Expected Behavior:**
- ✅ ACCEPTED (boundary is inclusive)
- Processes normally through all stages
- May create 1000+ chunks (large input)
- Processing time may approach 1000ms (max target)

**Rationale:** Limit is inclusive to allow maximum utilization.

**Test Scenario:**
```python
def test_add_memory_max_length_accepted():
    processor = MemoryProcessor(...)

    text = "x" * processor.config.max_text_length
    result = await processor.add_memory(text)

    assert result.id is not None
    assert result.chunks_created > 0
    assert result.processing_time_ms < 1000  # Within SLA
```

---

### Edge Case 3: Text Exceeds Max Length by 1

**Scenario:** Text is `max_text_length + 1` (10,000,001 chars)

**Input:**
```python
text = "x" * 10_000_001  # 1 char over limit
```

**Expected Behavior:**
```python
raise ValidationError("Text exceeds maximum length (10000001 > 10000000)")
```

**Rationale:** Strict enforcement of limits prevents OOM and performance degradation.

**Test Scenario:**
```python
def test_add_memory_exceeds_max_length_raises():
    processor = MemoryProcessor(...)

    text = "x" * (processor.config.max_text_length + 1)

    with pytest.raises(ValidationError, match="exceeds maximum length"):
        await processor.add_memory(text)
```

---

### Edge Case 4: Metadata with Reserved Key

**Scenario:** User provides metadata with reserved key `memory_id`, `timestamp`, or `chunks`

**Input:**
```python
metadata = {
    "memory_id": "user_provided_id",  # Reserved!
    "tags": ["python"]
}
```

**Expected Behavior:**
```python
raise ValidationError("Reserved metadata key 'memory_id' will be overwritten")
```

**Rationale:** System auto-populates these keys; user values would be silently overwritten, causing confusion.

**Test Scenario:**
```python
def test_add_memory_reserved_metadata_key_raises():
    processor = MemoryProcessor(...)

    # Test each reserved key
    for reserved_key in ["memory_id", "timestamp", "chunks"]:
        with pytest.raises(ValidationError, match=f"Reserved.*{reserved_key}"):
            await processor.add_memory(
                text="Test",
                metadata={reserved_key: "value"}
            )
```

---

### Edge Case 5: Metadata Not JSON-Serializable

**Scenario:** User provides metadata with non-JSON-serializable value (e.g., function, class instance)

**Input:**
```python
metadata = {
    "tags": ["python"],
    "callback": lambda x: x  # Not JSON-serializable!
}
```

**Expected Behavior:**
```python
raise ValidationError("metadata must be JSON-serializable: Object of type function is not...")
```

**Rationale:** FalkorDB stores metadata as JSON; non-serializable values cause storage errors.

**Test Scenario:**
```python
def test_add_memory_non_serializable_metadata_raises():
    processor = MemoryProcessor(...)

    with pytest.raises(ValidationError, match="JSON-serializable"):
        await processor.add_memory(
            text="Test",
            metadata={"func": lambda x: x}
        )
```

---

### Edge Case 6: Ollama Service Offline

**Scenario:** Ollama service is not running or unreachable during embedding generation

**Input:**
```python
# Ollama service stopped or network issue
text = "Python is great"
```

**Expected Behavior:**
```python
# After 3 retries with exponential backoff:
raise EmbeddingError("Embedding generation failed: Connection refused to http://localhost:11434")
```

**Fallback (if configured):**
- If `config.fallback_to_local_embedder=True`, use sentence-transformers locally
- Otherwise, fail after retries

**Rationale:** Ollama is external dependency; must handle unavailability gracefully.

**Test Scenario:**
```python
def test_add_memory_ollama_offline_raises(mocker):
    processor = MemoryProcessor(...)

    # Mock embedder to simulate connection error
    mocker.patch.object(
        processor.embedder,
        'embed',
        side_effect=EmbeddingError("Connection refused")
    )

    with pytest.raises(EmbeddingError, match="Connection refused"):
        await processor.add_memory("Test")
```

---

### Edge Case 7: FalkorDB Offline During Storage

**Scenario:** FalkorDB/Redis becomes unavailable during storage stage

**Input:**
```python
text = "Python is great"  # Valid input
# FalkorDB crashes or connection lost during storage
```

**Expected Behavior:**
```python
# After 3 retries:
raise StorageError("Storage failed: Connection refused to Redis at localhost:6379")
```

**Transaction Behavior:**
- If transaction started, it's rolled back automatically
- No partial data stored in database
- Atomicity preserved

**Rationale:** Database errors are transient (often); retry increases reliability.

**Test Scenario:**
```python
def test_add_memory_db_offline_raises(mocker):
    processor = MemoryProcessor(...)

    # Mock DB client to simulate connection error
    mocker.patch.object(
        processor.db_client,
        'execute_transaction',
        side_effect=StorageError("Connection refused")
    )

    with pytest.raises(StorageError, match="Connection refused"):
        await processor.add_memory("Test")
```

---

### Edge Case 8: Extremely Large Input (Near Max)

**Scenario:** User provides text close to max length (9MB+), creating 1000+ chunks

**Input:**
```python
text = "Python is great. " * 500_000  # ~9MB, 1000+ chunks
```

**Expected Behavior:**
- ✅ ACCEPTED (within limit)
- Chunking creates 1000+ chunks
- Embedding generation batched (32 chunks per request by default)
- Processing time approaches max target (~1000ms)
- Memory usage spikes (10+ GB for embeddings)

**Performance Impact:**
- Chunking: O(n) where n = text length (~200ms for 9MB)
- Embedding: O(k * e) where k = chunks, e = embedding time (~500ms for 1000 chunks)
- Storage: O(k) for chunk nodes (~200ms for 1000 chunks)
- **Total: ~900ms (within 1000ms SLA)**

**Rationale:** System must handle large inputs gracefully (spec allows up to 10MB).

**Test Scenario:**
```python
def test_add_memory_large_input_performance():
    processor = MemoryProcessor(...)

    # Generate ~9MB text
    large_text = "Python is great. " * 500_000

    result = await processor.add_memory(large_text)

    assert result.chunks_created > 1000  # Many chunks
    assert result.processing_time_ms < 1000  # Within SLA
```

---

### Edge Case 9: Non-UTF-8 Encoding

**Scenario:** User provides text with invalid UTF-8 bytes (e.g., binary data, corrupted file)

**Input:**
```python
# Simulated non-UTF-8 bytes
text = "Valid text" + chr(0xD800)  # Surrogate pair (invalid UTF-8)
```

**Expected Behavior:**
```python
raise ValidationError("Text must be valid UTF-8: 'utf-8' codec can't encode character...")
```

**Rationale:** System assumes UTF-8 throughout (chunker, embedder, DB); non-UTF-8 causes errors downstream.

**Test Scenario:**
```python
def test_add_memory_non_utf8_raises():
    processor = MemoryProcessor(...)

    # Create invalid UTF-8 string
    invalid_text = "Valid" + chr(0xD800)

    with pytest.raises(ValidationError, match="valid UTF-8"):
        await processor.add_memory(invalid_text)
```

---

### Edge Case 10: Concurrent Calls (Thread Safety)

**Scenario:** Multiple threads/coroutines call `add_memory()` simultaneously on same MemoryProcessor instance

**Input:**
```python
# 10 concurrent calls
tasks = [
    processor.add_memory(f"Memory {i}")
    for i in range(10)
]
results = await asyncio.gather(*tasks)
```

**Expected Behavior:**
- ✅ All 10 memories stored successfully
- Each gets unique `memory_id`
- No race conditions or data corruption
- MemoryProcessor instance remains consistent

**Rationale:** MemoryProcessor is stateless (no shared mutable state); safe for concurrent use.

**Concurrency Guarantees:**
- FalkorDBClient uses connection pooling (thread-safe)
- OllamaEmbedder is stateless (thread-safe)
- SemanticChunker is stateless (thread-safe)
- No shared mutable state in MemoryProcessor

**Test Scenario:**
```python
@pytest.mark.asyncio
async def test_add_memory_concurrent_calls():
    processor = MemoryProcessor(...)

    # 10 concurrent calls
    tasks = [processor.add_memory(f"Memory {i}") for i in range(10)]
    results = await asyncio.gather(*tasks)

    # All succeeded
    assert len(results) == 10

    # All unique IDs
    ids = [r.id for r in results]
    assert len(set(ids)) == 10

    # All stored
    for result in results:
        search_results = await processor.search_memory(f"Memory", limit=1)
        assert len(search_results) > 0
```

---

## Test Scenarios (Complete List)

### Happy Path Tests (5 scenarios)

#### 1. test_add_memory_success_minimal
**Purpose:** Verify basic memory addition with minimal input

**Input:**
- text: `"Python is a programming language."`
- metadata: `None`

**Expected:**
- Returns `MemoryResult` with valid `id` (UUID v4)
- `chunks_created >= 1`
- `processing_time_ms > 0` and `< 100` (small input)
- Memory searchable via `search_memory()`

---

#### 2. test_add_memory_success_with_metadata
**Purpose:** Verify memory addition with complete metadata

**Input:**
- text: `"Django is a Python web framework."`
- metadata: `{"tags": ["python", "django"], "source": "user", "date": "2025-11-23"}`

**Expected:**
- Returns `MemoryResult` with valid `id`
- Metadata stored correctly (verify via `search_memory()` result)
- System metadata populated: `memory_id`, `timestamp`, `chunks`

---

#### 3. test_add_memory_success_boundary_text_length
**Purpose:** Verify handling of text exactly at max length

**Input:**
- text: `"x" * config.max_text_length` (10,000,000 chars)
- metadata: `None`

**Expected:**
- Accepted (no ValidationError)
- `chunks_created > 1000` (many chunks)
- `processing_time_ms < 1000` (within SLA)

---

#### 4. test_add_memory_success_unicode_emoji
**Purpose:** Verify handling of Unicode and emoji characters

**Input:**
- text: `"Hello мир 🌍 Python"` (mixed English, Cyrillic, emoji)
- metadata: `None`

**Expected:**
- Accepted (valid UTF-8)
- Returns valid `MemoryResult`
- Searchable

---

#### 5. test_add_memory_success_code_content
**Purpose:** Verify handling of code snippets

**Input:**
- text: `"def hello():\n    print('Hi')"`
- metadata: `{"tags": ["code", "python"]}`

**Expected:**
- Accepted
- Returns valid `MemoryResult`
- Searchable

---

### Validation Error Tests (9 scenarios)

#### 6. test_add_memory_empty_text_raises
**Purpose:** Verify rejection of empty text

**Input:**
- text: `""`

**Expected:**
- Raises `ValidationError` with message `"Text cannot be empty"`

---

#### 7. test_add_memory_whitespace_only_raises
**Purpose:** Verify rejection of whitespace-only text

**Input:**
- text: `"   \n\t  "`

**Expected:**
- Raises `ValidationError` with message `"empty or whitespace"`

---

#### 8. test_add_memory_exceeds_max_length_raises
**Purpose:** Verify rejection of text exceeding max length

**Input:**
- text: `"x" * (config.max_text_length + 1)`

**Expected:**
- Raises `ValidationError` with message `"exceeds maximum length"`

---

#### 9. test_add_memory_non_utf8_raises
**Purpose:** Verify rejection of non-UTF-8 text

**Input:**
- text: `"Valid" + chr(0xD800)` (invalid UTF-8 surrogate)

**Expected:**
- Raises `ValidationError` with message `"valid UTF-8"`

---

#### 10. test_add_memory_metadata_not_dict_raises
**Purpose:** Verify rejection of non-dict metadata

**Input:**
- text: `"Test"`
- metadata: `"not_a_dict"` (string instead of dict)

**Expected:**
- Raises `ValidationError` with message `"metadata must be dict"`

---

#### 11. test_add_memory_metadata_non_string_keys_raises
**Purpose:** Verify rejection of metadata with non-string keys

**Input:**
- text: `"Test"`
- metadata: `{123: "value"}` (integer key)

**Expected:**
- Raises `ValidationError` with message `"keys must be strings"`

---

#### 12. test_add_memory_reserved_metadata_key_raises
**Purpose:** Verify rejection of reserved metadata keys

**Input:**
- text: `"Test"`
- metadata: `{"memory_id": "x"}` (reserved key)

**Expected:**
- Raises `ValidationError` with message `"Reserved metadata key 'memory_id'"`

---

#### 13. test_add_memory_non_serializable_metadata_raises
**Purpose:** Verify rejection of non-JSON-serializable metadata

**Input:**
- text: `"Test"`
- metadata: `{"func": lambda x: x}` (function)

**Expected:**
- Raises `ValidationError` with message `"JSON-serializable"`

---

#### 14. test_add_memory_metadata_reserved_timestamp_raises
**Purpose:** Verify rejection of metadata with 'timestamp' key

**Input:**
- text: `"Test"`
- metadata: `{"timestamp": "2025-11-23"}` (reserved)

**Expected:**
- Raises `ValidationError` with message `"Reserved metadata key 'timestamp'"`

---

### Pipeline Stage Error Tests (4 scenarios)

#### 15. test_add_memory_ollama_offline_raises
**Purpose:** Verify handling of Ollama service unavailable

**Setup:**
- Mock `embedder.embed()` to raise `EmbeddingError("Connection refused")`

**Input:**
- text: `"Test"`

**Expected:**
- Raises `EmbeddingError` after retries
- Message contains `"Connection refused"`
- Transaction not started (no DB writes)

---

#### 16. test_add_memory_db_offline_raises
**Purpose:** Verify handling of FalkorDB unavailable

**Setup:**
- Mock `db_client.execute_transaction()` to raise `StorageError("Connection refused")`

**Input:**
- text: `"Test"`

**Expected:**
- Raises `StorageError` after retries
- Message contains `"Connection refused"`
- Transaction rolled back (no partial data)

---

#### 17. test_add_memory_chunking_error_raises
**Purpose:** Verify handling of chunking failure (rare internal error)

**Setup:**
- Mock `chunker.chunk_text()` to raise `ChunkingError("Out of memory")`

**Input:**
- text: `"Test"`

**Expected:**
- Raises `ChunkingError`
- Pipeline aborted before embedding stage

---

#### 18. test_add_memory_extraction_error_graceful_degradation
**Purpose:** Verify graceful degradation when entity extraction fails (Phase 2)

**Setup:**
- `config.enable_extraction = True`
- Mock `extractor.extract_entities()` to raise `ExtractionError("SpaCy model not found")`

**Input:**
- text: `"Test"`

**Expected:**
- Does NOT raise exception (graceful degradation)
- Returns valid `MemoryResult`
- Memory stored WITHOUT entities
- Warning logged

---

### Performance Tests (3 scenarios)

#### 19. test_add_memory_performance_small_input
**Purpose:** Verify performance target for small input

**Input:**
- text: `"Python is great."` (< 1KB)

**Expected:**
- `processing_time_ms < 100` (P95 target)

---

#### 20. test_add_memory_performance_medium_input
**Purpose:** Verify performance target for medium input

**Input:**
- text: ~10KB (paragraphs of text)

**Expected:**
- `processing_time_ms < 300` (P95 target)

---

#### 21. test_add_memory_performance_large_input
**Purpose:** Verify performance target for large input

**Input:**
- text: ~100KB (long document)

**Expected:**
- `processing_time_ms < 500` (P95 target)

---

### Concurrency Tests (3 scenarios)

#### 22. test_add_memory_concurrent_calls_thread_safe
**Purpose:** Verify thread safety with concurrent calls

**Input:**
- 10 concurrent `add_memory()` calls with different texts

**Expected:**
- All 10 succeed
- All 10 unique `memory_id` values
- No race conditions or corruption

---

#### 23. test_add_memory_concurrent_same_text_unique_ids
**Purpose:** Verify unique IDs even for identical text

**Input:**
- 5 concurrent calls with SAME text: `"Python is great"`

**Expected:**
- All 5 succeed
- All 5 get DIFFERENT `memory_id` values
- All 5 searchable independently

---

#### 24. test_add_memory_concurrent_db_connection_pool
**Purpose:** Verify connection pool handles concurrent load

**Input:**
- 100 concurrent calls (exceeds typical pool size of 10)

**Expected:**
- All 100 succeed (may take longer due to queuing)
- No connection pool exhaustion errors

---

### Integration Tests (3 scenarios)

#### 25. test_add_memory_then_search_found
**Purpose:** Verify end-to-end flow: add → search → found

**Flow:**
1. Add memory: `"Python was created by Guido van Rossum"`
2. Search: `"Who created Python?"`
3. Verify search result contains added memory

---

#### 26. test_add_memory_with_tags_filter_search
**Purpose:** Verify metadata filtering in search

**Flow:**
1. Add memory with `metadata={"tags": ["python"]}`
2. Search with `filters={"tags": ["python"]}`
3. Verify result found

---

#### 27. test_add_memory_multiple_then_search_ranking
**Purpose:** Verify search ranking works correctly

**Flow:**
1. Add 3 memories about Python, Django, Flask
2. Search: `"Python web framework"`
3. Verify Django/Flask ranked higher than general Python memory

---

## Performance Requirements

### Latency Targets

| Input Size | Metric | Target (P50) | Target (P95) | Target (P99) | Max Allowed |
|------------|--------|--------------|--------------|--------------|-------------|
| < 1KB      | Processing time | 30ms | 50ms | 100ms | 200ms |
| < 10KB     | Processing time | 100ms | 150ms | 300ms | 500ms |
| < 100KB    | Processing time | 200ms | 300ms | 500ms | 1000ms |
| 10MB (max) | Processing time | 500ms | 700ms | 900ms | 1000ms |

### Throughput

- **Sequential**: 10 ops/sec (single thread, no concurrency)
- **Concurrent**: 50 ops/sec (with 10 concurrent workers)
- **Bottleneck**: Ollama embedding generation (~50ms per batch of 32 chunks)

### Resource Usage

- **Memory (Per Call)**:
  - Small input (< 1KB): ~10MB (chunks + embeddings + temp buffers)
  - Large input (10MB): ~500MB (1000+ chunks × 768-dim embeddings)

- **CPU**:
  - Chunking: Low (O(n) text processing)
  - Embedding: None (offloaded to Ollama)
  - Storage: Low (I/O bound)

- **Network**:
  - Embedding requests: 1-32 requests to Ollama (depending on chunk count)
  - Database writes: 1 transaction with k chunks (k = 1 to 1000+)

### Performance Breakdown (10KB input, ~5 chunks)

| Stage | Time (ms) | % of Total |
|-------|-----------|------------|
| Validation | 1 | 1% |
| Chunking | 20 | 13% |
| Embedding (Ollama) | 100 | 67% |
| Extraction (Phase 2) | - | 0% (disabled) |
| Storage (FalkorDB) | 30 | 20% |
| **Total** | **~150ms** | **100%** |

**Bottleneck:** Ollama embedding generation (67% of total time)

**Optimization:** Use semantic caching (Phase 2) to reduce embedding time by 60%+ (hit rate dependent)

---

## Security Considerations

### Input Validation

- ✅ **All inputs validated** before processing
- ✅ **No injection vulnerabilities**: Text stored as-is (no code execution)
- ✅ **Safe error messages**: No sensitive data leaked in exceptions

### Data Protection

- **Text Content**: Stored in plaintext in FalkorDB (not encrypted at rest in Phase 1)
- **Metadata**: Stored in plaintext (consider encryption for sensitive fields in production)
- **Embeddings**: Not reversible to original text (privacy-preserving)

### Rate Limiting (Not Implemented in Phase 1)

- **Phase 2**: Add rate limiting at MCP tool level (e.g., 100 calls/min per client)
- **Rationale**: Prevent abuse, DoS attacks

### Authentication (Not Implemented in Phase 1)

- **Phase 1**: No authentication (local use only)
- **Phase 2+**: Add API key authentication for remote access

---

## Related Functions

### Called By (Upstream)

- **AddMemoryTool.execute()** (zapomni_mcp module)
  - MCP tool that wraps `add_memory()` for MCP protocol
  - Handles MCP request parsing and response formatting

- **BatchImporter.import_documents()** (future)
  - Batch imports multiple documents
  - Calls `add_memory()` for each document

### Calls (Downstream)

- **self._validate_text(text: str) -> None** (private helper)
  - Validates text constraints (non-empty, UTF-8, length)
  - Raises: `ValidationError`

- **self._validate_metadata(metadata: Optional[Dict]) -> None** (private helper)
  - Validates metadata structure (dict, string keys, JSON-serializable, no reserved keys)
  - Raises: `ValidationError`

- **self._create_memory_id() -> str** (private helper)
  - Generates UUID v4 for memory
  - Returns: UUID string

- **self._process_chunks(text: str) -> List[Chunk]** (private helper)
  - Delegates to `chunker.chunk_text()`
  - Returns: List of Chunk objects
  - Raises: `ChunkingError`

- **self._generate_embeddings(chunks: List[Chunk]) -> List[List[float]]** (private helper)
  - Delegates to `embedder.embed()` (batched)
  - Checks cache first (if enabled, Phase 2)
  - Returns: List of embedding vectors
  - Raises: `EmbeddingError`

- **self._extract_entities(text: str, chunks: List[Chunk]) -> Tuple[List[Entity], List[Relationship]]** (private helper, Phase 2)
  - Delegates to `extractor.extract_entities()`
  - Returns: Entities and relationships
  - Raises: `ExtractionError`

- **self._store_memory(...) -> str** (private helper)
  - Delegates to `db_client.execute_transaction()`
  - Stores all data atomically
  - Returns: memory_id
  - Raises: `StorageError`

---

## Implementation Notes

### Libraries Used

- **uuid**: UUID v4 generation (`uuid.uuid4()`)
- **datetime**: Timestamp generation (`datetime.now(timezone.utc).isoformat()`)
- **structlog**: Structured logging (`logger.info()`, `logger.error()`)
- **json**: Metadata serialization check (`json.dumps()`)

### Known Limitations

1. **Max Text Size**: 10MB limit (configurable but not recommended to increase due to performance)
2. **UTF-8 Only**: No support for other encodings (binary data rejected)
3. **No Duplicate Detection**: Same text can be stored multiple times (each gets unique ID)
4. **No Versioning**: Cannot update existing memory (Phase 2 feature)
5. **No Async Batching**: Each call processes independently (Phase 2: add batch API)

### Future Enhancements (Phase 2+)

1. **Semantic Deduplication**: Detect and merge similar memories (via embedding similarity)
2. **Streaming Support**: Process very large texts (> 10MB) via streaming chunking
3. **Async Batching**: `add_memories(List[str])` for efficient batch processing
4. **Versioning**: Update existing memory while preserving history
5. **Encryption**: Encrypt sensitive fields at rest and in transit

---

## References

### Component Spec
- [memory_processor_component.md](/home/dev/zapomni/.spec-workflow/specs/level2/memory_processor_component.md) - Parent component specification

### Module Spec
- [zapomni_core_module.md](/home/dev/zapomni/.spec-workflow/specs/level1/zapomni_core_module.md) - Parent module specification

### Related Function Specs
- `search_memory()` - Search stored memories
- `get_stats()` - Retrieve system statistics
- `build_knowledge_graph()` - Build entity graph (Phase 2)

### Related Component Specs
- SemanticChunker - Text chunking service
- OllamaEmbedder - Embedding generation service
- FalkorDBClient - Database storage service

### External Documentation
- UUID v4: https://docs.python.org/3/library/uuid.html
- Python asyncio: https://docs.python.org/3/library/asyncio.html
- FalkorDB Documentation: https://docs.falkordb.com/
- Ollama API: https://github.com/ollama/ollama/blob/main/docs/api.md

---

**Document Status:** Draft v1.0
**Created:** 2025-11-23
**Author:** Goncharenko Anton aka alienxs2
**Copyright:** Copyright (c) 2025 Goncharenko Anton aka alienxs2
**License:** MIT License

**Ready for Review:** Yes ✅

**Next Steps:**
1. Review and approve this function specification
2. Implement `add_memory()` method in MemoryProcessor class
3. Write 27 unit tests defined in Test Scenarios section
4. Integration tests with real dependencies (Ollama, FalkorDB)
5. Performance benchmarking against latency targets
