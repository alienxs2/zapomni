# MemoryProcessor.__init__() - Function Specification

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
def __init__(
    self,
    db_client: FalkorDBClient,
    chunker: SemanticChunker,
    embedder: OllamaEmbedder,
    extractor: Optional[EntityExtractor] = None,
    cache: Optional[SemanticCache] = None,
    task_manager: Optional[TaskManager] = None,
    config: Optional[ProcessorConfig] = None
) -> None:
    """
    Initialize MemoryProcessor with all dependencies via dependency injection.

    This constructor implements the dependency injection pattern where all
    required components are provided externally, making the class testable
    and flexible. It validates all inputs and initializes the processor
    in a ready-to-use state.

    Args:
        db_client: FalkorDB client for storage and retrieval (REQUIRED)
            - Must be instantiated and ready to connect
            - Connection check performed on first operation (lazy connect)
            - Example: FalkorDBClient(host="localhost", port=6379)

        chunker: SemanticChunker for text chunking (REQUIRED)
            - Must be instantiated with valid configuration
            - Example: SemanticChunker(chunk_size=512, chunk_overlap=50)

        embedder: OllamaEmbedder for embedding generation (REQUIRED)
            - Must be instantiated with Ollama host configured
            - Example: OllamaEmbedder(host="http://localhost:11434", model="nomic-embed-text")

        extractor: EntityExtractor for entity/relationship extraction (OPTIONAL, Phase 2)
            - None disables entity extraction
            - If provided, must be instantiated with valid config
            - Example: EntityExtractor(spacy_model="en_core_web_sm")

        cache: SemanticCache for embedding caching (OPTIONAL, Phase 2)
            - None disables caching (all embeddings generated fresh)
            - If provided, must have Redis client configured
            - Example: SemanticCache(redis_client=RedisClient())

        task_manager: TaskManager for background tasks (OPTIONAL, Phase 2)
            - None disables background task support
            - Required for build_knowledge_graph() method
            - Example: TaskManager()

        config: ProcessorConfig with system configuration (OPTIONAL)
            - None uses default ProcessorConfig()
            - Must have valid values (max_text_length > 0, etc.)
            - Example: ProcessorConfig(enable_cache=True, max_text_length=5_000_000)

    Returns:
        None (constructor)

    Raises:
        ValueError: If db_client is None (required dependency)
        ValueError: If chunker is None (required dependency)
        ValueError: If embedder is None (required dependency)
        ValueError: If config.max_text_length <= 0 (invalid configuration)
        ValueError: If config.batch_size <= 0 (invalid configuration)
        ValueError: If config.search_mode not in ["vector", "bm25", "hybrid", "graph"]
        TypeError: If any dependency has wrong type (e.g., db_client is not FalkorDBClient)

    Example:
        >>> # Phase 1: Minimal configuration
        >>> processor = MemoryProcessor(
        ...     db_client=FalkorDBClient(host="localhost"),
        ...     chunker=SemanticChunker(),
        ...     embedder=OllamaEmbedder(host="http://localhost:11434")
        ... )

        >>> # Phase 2: Full configuration
        >>> processor = MemoryProcessor(
        ...     db_client=FalkorDBClient(host="localhost"),
        ...     chunker=SemanticChunker(chunk_size=512),
        ...     embedder=OllamaEmbedder(host="http://localhost:11434"),
        ...     extractor=EntityExtractor(spacy_model="en_core_web_sm"),
        ...     cache=SemanticCache(redis_client=RedisClient()),
        ...     task_manager=TaskManager(),
        ...     config=ProcessorConfig(enable_cache=True, enable_extraction=True)
        ... )
    """
```

---

## Purpose & Context

### What It Does

The `__init__()` method initializes a MemoryProcessor instance by:
1. Validating all required dependencies are provided (not None)
2. Type-checking all dependencies to ensure correct types
3. Validating configuration parameters if provided
4. Storing all dependencies as instance attributes
5. Creating a default configuration if none provided
6. Initializing a structured logger with component context
7. Logging successful initialization with configuration details

This method follows the **dependency injection pattern** where all components are provided externally rather than created internally, which provides:
- **Testability**: Easy to mock dependencies in unit tests
- **Flexibility**: Can swap implementations (e.g., different embedders)
- **Clarity**: All dependencies are explicit in the constructor
- **Best practice**: Industry-standard DI pattern for clean architecture

### Why It Exists

Initialization is critical for establishing the processor's operational readiness. Without proper initialization:
- Dependencies could be missing, leading to runtime errors later
- Configuration could be invalid, causing unexpected behavior
- The processor could be in an inconsistent state

By validating everything upfront in `__init__`, we follow the **fail-fast principle**: detect problems immediately at construction time rather than during runtime operations.

### When To Use

Called once when creating a MemoryProcessor instance:
- At application startup (by MCP server initialization code)
- In test fixtures (with mocked dependencies)
- In factory functions that construct processors with specific configs

### When NOT To Use

Never call `__init__` directly after construction. Instead:
- Create a new instance if different dependencies needed
- Python automatically calls `__init__` during object construction

---

## Parameters (Detailed)

### db_client: FalkorDBClient

**Type:** `FalkorDBClient` (required)

**Purpose:** Database client for storage, retrieval, and graph operations

**Constraints:**
- Must not be None (raises ValueError)
- Must be instance of FalkorDBClient (raises TypeError)
- Must have valid host and port configuration
- Connection is lazy (not checked in __init__, validated on first use)

**Validation:**
```python
if db_client is None:
    raise ValueError("db_client is required")
if not isinstance(db_client, FalkorDBClient):
    raise TypeError(f"db_client must be FalkorDBClient, got {type(db_client)}")
```

**Examples:**
- Valid: `FalkorDBClient(host="localhost", port=6379)`
- Valid: `FalkorDBClient(host="192.168.1.100", port=6379, db=1)`
- Invalid: `None` → ValueError
- Invalid: `"localhost"` → TypeError

---

### chunker: SemanticChunker

**Type:** `SemanticChunker` (required)

**Purpose:** Text chunking service for splitting content into semantic units

**Constraints:**
- Must not be None (raises ValueError)
- Must be instance of SemanticChunker (raises TypeError)
- Must have valid configuration (chunk_size > 0, etc.)

**Validation:**
```python
if chunker is None:
    raise ValueError("chunker is required")
if not isinstance(chunker, SemanticChunker):
    raise TypeError(f"chunker must be SemanticChunker, got {type(chunker)}")
```

**Examples:**
- Valid: `SemanticChunker(chunk_size=512, chunk_overlap=50)`
- Valid: `SemanticChunker()` (uses defaults)
- Invalid: `None` → ValueError
- Invalid: `123` → TypeError

---

### embedder: OllamaEmbedder

**Type:** `OllamaEmbedder` (required)

**Purpose:** Embedding generation service via Ollama API

**Constraints:**
- Must not be None (raises ValueError)
- Must be instance of OllamaEmbedder (raises TypeError)
- Must have valid host URL configured

**Validation:**
```python
if embedder is None:
    raise ValueError("embedder is required")
if not isinstance(embedder, OllamaEmbedder):
    raise TypeError(f"embedder must be OllamaEmbedder, got {type(embedder)}")
```

**Examples:**
- Valid: `OllamaEmbedder(host="http://localhost:11434", model="nomic-embed-text")`
- Valid: `OllamaEmbedder(host="http://10.0.0.5:11434")`
- Invalid: `None` → ValueError
- Invalid: `object()` → TypeError

---

### extractor: Optional[EntityExtractor]

**Type:** `Optional[EntityExtractor]` (optional, Phase 2)

**Purpose:** Entity and relationship extraction service

**Default:** `None` (extraction disabled)

**Constraints:**
- Can be None (no validation if None)
- If not None, must be instance of EntityExtractor (raises TypeError)
- If not None, must have valid SpaCy model configured

**Validation:**
```python
if extractor is not None and not isinstance(extractor, EntityExtractor):
    raise TypeError(f"extractor must be EntityExtractor, got {type(extractor)}")
```

**Examples:**
- Valid: `None` (extraction disabled)
- Valid: `EntityExtractor(spacy_model="en_core_web_sm")`
- Invalid: `"extractor"` → TypeError

---

### cache: Optional[SemanticCache]

**Type:** `Optional[SemanticCache]` (optional, Phase 2)

**Purpose:** Semantic embedding cache for performance

**Default:** `None` (caching disabled)

**Constraints:**
- Can be None (no caching)
- If not None, must be instance of SemanticCache (raises TypeError)

**Validation:**
```python
if cache is not None and not isinstance(cache, SemanticCache):
    raise TypeError(f"cache must be SemanticCache, got {type(cache)}")
```

**Examples:**
- Valid: `None` (caching disabled)
- Valid: `SemanticCache(redis_client=RedisClient())`
- Invalid: `{}` → TypeError

---

### task_manager: Optional[TaskManager]

**Type:** `Optional[TaskManager]` (optional, Phase 2)

**Purpose:** Background task manager for async operations

**Default:** `None` (background tasks disabled)

**Constraints:**
- Can be None (no background tasks)
- If not None, must be instance of TaskManager (raises TypeError)

**Validation:**
```python
if task_manager is not None and not isinstance(task_manager, TaskManager):
    raise TypeError(f"task_manager must be TaskManager, got {type(task_manager)}")
```

**Examples:**
- Valid: `None` (background tasks disabled)
- Valid: `TaskManager()`
- Valid: `TaskManager(max_workers=5)`
- Invalid: `[]` → TypeError

---

### config: Optional[ProcessorConfig]

**Type:** `Optional[ProcessorConfig]` (optional)

**Purpose:** System configuration and feature flags

**Default:** `None` (uses `ProcessorConfig()` defaults)

**Constraints:**
- Can be None (uses defaults)
- If provided, must be instance of ProcessorConfig (raises TypeError)
- If provided, max_text_length must be > 0 (raises ValueError)
- If provided, batch_size must be > 0 (raises ValueError)
- If provided, search_mode must be in ["vector", "bm25", "hybrid", "graph"] (raises ValueError)

**Validation:**
```python
if config is not None:
    if not isinstance(config, ProcessorConfig):
        raise TypeError(f"config must be ProcessorConfig, got {type(config)}")
    if config.max_text_length <= 0:
        raise ValueError("config.max_text_length must be positive")
    if config.batch_size <= 0:
        raise ValueError("config.batch_size must be positive")
    if config.search_mode not in ["vector", "bm25", "hybrid", "graph"]:
        raise ValueError(f"Invalid search_mode: {config.search_mode}")
```

**Examples:**
- Valid: `None` (uses defaults)
- Valid: `ProcessorConfig()`
- Valid: `ProcessorConfig(enable_cache=True, max_text_length=5_000_000)`
- Invalid: `ProcessorConfig(max_text_length=0)` → ValueError
- Invalid: `ProcessorConfig(search_mode="invalid")` → ValueError

---

## Return Value

**Type:** `None` (constructor)

**Purpose:** Constructors in Python return None implicitly

**Side Effects:**
- `self.db_client` set to db_client parameter
- `self.chunker` set to chunker parameter
- `self.embedder` set to embedder parameter
- `self.extractor` set to extractor parameter (or None)
- `self.cache` set to cache parameter (or None)
- `self.task_manager` set to task_manager parameter (or None)
- `self.config` set to config parameter or default ProcessorConfig()
- `self.logger` created via structlog.get_logger()
- Initialization logged with INFO level

---

## Exceptions

### ValueError: db_client is None

**When Raised:** db_client parameter is None

**Message Format:** `"db_client is required"`

**Example:**
```python
try:
    processor = MemoryProcessor(db_client=None, chunker=chunker, embedder=embedder)
except ValueError as e:
    print(e)  # "db_client is required"
```

**Recovery:** Provide valid FalkorDBClient instance

---

### ValueError: chunker is None

**When Raised:** chunker parameter is None

**Message Format:** `"chunker is required"`

**Recovery:** Provide valid SemanticChunker instance

---

### ValueError: embedder is None

**When Raised:** embedder parameter is None

**Message Format:** `"embedder is required"`

**Recovery:** Provide valid OllamaEmbedder instance

---

### ValueError: Invalid config.max_text_length

**When Raised:** config.max_text_length <= 0

**Message Format:** `"config.max_text_length must be positive"`

**Recovery:** Provide config with max_text_length > 0

---

### ValueError: Invalid config.batch_size

**When Raised:** config.batch_size <= 0

**Message Format:** `"config.batch_size must be positive"`

**Recovery:** Provide config with batch_size > 0

---

### ValueError: Invalid config.search_mode

**When Raised:** config.search_mode not in allowed modes

**Message Format:** `"Invalid search_mode: {mode}. Must be one of: vector, bm25, hybrid, graph"`

**Recovery:** Provide valid search_mode

---

### TypeError: Wrong dependency type

**When Raised:** Dependency has incorrect type (e.g., db_client is not FalkorDBClient)

**Message Format:** `"{param_name} must be {expected_type}, got {actual_type}"`

**Example:**
```python
try:
    processor = MemoryProcessor(db_client="localhost", chunker=chunker, embedder=embedder)
except TypeError as e:
    print(e)  # "db_client must be FalkorDBClient, got <class 'str'>"
```

**Recovery:** Provide correct type for dependency

---

## Algorithm (Pseudocode)

```
FUNCTION __init__(db_client, chunker, embedder, extractor, cache, task_manager, config):
    # Step 1: Validate required dependencies
    IF db_client is None:
        RAISE ValueError("db_client is required")

    IF chunker is None:
        RAISE ValueError("chunker is required")

    IF embedder is None:
        RAISE ValueError("embedder is required")

    # Step 2: Type-check required dependencies
    IF NOT isinstance(db_client, FalkorDBClient):
        RAISE TypeError(f"db_client must be FalkorDBClient, got {type(db_client)}")

    IF NOT isinstance(chunker, SemanticChunker):
        RAISE TypeError(f"chunker must be SemanticChunker, got {type(chunker)}")

    IF NOT isinstance(embedder, OllamaEmbedder):
        RAISE TypeError(f"embedder must be OllamaEmbedder, got {type(embedder)}")

    # Step 3: Type-check optional dependencies (if provided)
    IF extractor is not None AND NOT isinstance(extractor, EntityExtractor):
        RAISE TypeError(f"extractor must be EntityExtractor, got {type(extractor)}")

    IF cache is not None AND NOT isinstance(cache, SemanticCache):
        RAISE TypeError(f"cache must be SemanticCache, got {type(cache)}")

    IF task_manager is not None AND NOT isinstance(task_manager, TaskManager):
        RAISE TypeError(f"task_manager must be TaskManager, got {type(task_manager)}")

    # Step 4: Validate config (if provided)
    IF config is not None:
        IF NOT isinstance(config, ProcessorConfig):
            RAISE TypeError(f"config must be ProcessorConfig, got {type(config)}")

        IF config.max_text_length <= 0:
            RAISE ValueError("config.max_text_length must be positive")

        IF config.batch_size <= 0:
            RAISE ValueError("config.batch_size must be positive")

        IF config.search_mode NOT IN ["vector", "bm25", "hybrid", "graph"]:
            RAISE ValueError(f"Invalid search_mode: {config.search_mode}")

    # Step 5: Store all dependencies as instance attributes
    self.db_client = db_client
    self.chunker = chunker
    self.embedder = embedder
    self.extractor = extractor
    self.cache = cache
    self.task_manager = task_manager

    # Step 6: Create default config if not provided
    IF config is None:
        self.config = ProcessorConfig()  # Uses defaults
    ELSE:
        self.config = config

    # Step 7: Initialize structured logger
    self.logger = structlog.get_logger().bind(
        component="MemoryProcessor",
        config={
            "enable_cache": self.config.enable_cache,
            "enable_extraction": self.config.enable_extraction,
            "enable_graph": self.config.enable_graph,
            "max_text_length": self.config.max_text_length
        }
    )

    # Step 8: Log successful initialization
    self.logger.info(
        "MemoryProcessor initialized",
        has_extractor=extractor is not None,
        has_cache=cache is not None,
        has_task_manager=task_manager is not None
    )

    # Instance is now ready to use
    RETURN None
END FUNCTION
```

---

## Preconditions

- All required dependencies (db_client, chunker, embedder) must be instantiated
- Optional dependencies (if provided) must be instantiated
- Config (if provided) must have valid values

**Note:** Database connection is NOT required at initialization time (lazy connection).
Connection is established on first database operation (add_memory, search_memory, etc.).

---

## Postconditions

- ✅ All instance attributes set to valid values
- ✅ self.config is never None (defaults created if not provided)
- ✅ self.logger initialized with component context
- ✅ Instance ready to call add_memory(), search_memory(), get_stats()
- ✅ Initialization logged for observability

---

## Edge Cases & Handling

### Edge Case 1: db_client is None

**Scenario:** User passes None for required db_client

**Expected Behavior:**
```python
raise ValueError("db_client is required")
```

**Test Scenario:**
```python
def test_init_missing_db_client_raises():
    with pytest.raises(ValueError, match="db_client is required"):
        MemoryProcessor(
            db_client=None,
            chunker=SemanticChunker(),
            embedder=OllamaEmbedder()
        )
```

---

### Edge Case 2: chunker is None

**Scenario:** User passes None for required chunker

**Expected Behavior:**
```python
raise ValueError("chunker is required")
```

**Test Scenario:**
```python
def test_init_missing_chunker_raises():
    with pytest.raises(ValueError, match="chunker is required"):
        MemoryProcessor(
            db_client=FalkorDBClient(),
            chunker=None,
            embedder=OllamaEmbedder()
        )
```

---

### Edge Case 3: embedder is None

**Scenario:** User passes None for required embedder

**Expected Behavior:**
```python
raise ValueError("embedder is required")
```

**Test Scenario:**
```python
def test_init_missing_embedder_raises():
    with pytest.raises(ValueError, match="embedder is required"):
        MemoryProcessor(
            db_client=FalkorDBClient(),
            chunker=SemanticChunker(),
            embedder=None
        )
```

---

### Edge Case 4: Wrong type for db_client

**Scenario:** User passes string instead of FalkorDBClient

**Expected Behavior:**
```python
raise TypeError("db_client must be FalkorDBClient, got <class 'str'>")
```

**Test Scenario:**
```python
def test_init_wrong_type_db_client_raises():
    with pytest.raises(TypeError, match="db_client must be FalkorDBClient"):
        MemoryProcessor(
            db_client="localhost",
            chunker=SemanticChunker(),
            embedder=OllamaEmbedder()
        )
```

---

### Edge Case 5: config.max_text_length = 0

**Scenario:** User provides config with max_text_length = 0

**Expected Behavior:**
```python
raise ValueError("config.max_text_length must be positive")
```

**Test Scenario:**
```python
def test_init_invalid_max_text_length_raises():
    config = ProcessorConfig(max_text_length=0)
    with pytest.raises(ValueError, match="max_text_length must be positive"):
        MemoryProcessor(
            db_client=FalkorDBClient(),
            chunker=SemanticChunker(),
            embedder=OllamaEmbedder(),
            config=config
        )
```

---

### Edge Case 6: config.batch_size = 0

**Scenario:** User provides config with batch_size = 0

**Expected Behavior:**
```python
raise ValueError("config.batch_size must be positive")
```

**Test Scenario:**
```python
def test_init_invalid_batch_size_raises():
    config = ProcessorConfig(batch_size=0)
    with pytest.raises(ValueError, match="batch_size must be positive"):
        MemoryProcessor(
            db_client=FalkorDBClient(),
            chunker=SemanticChunker(),
            embedder=OllamaEmbedder(),
            config=config
        )
```

---

### Edge Case 7: config.search_mode = "invalid"

**Scenario:** User provides config with unsupported search_mode

**Expected Behavior:**
```python
raise ValueError("Invalid search_mode: invalid. Must be one of: vector, bm25, hybrid, graph")
```

**Test Scenario:**
```python
def test_init_invalid_search_mode_raises():
    config = ProcessorConfig(search_mode="invalid")
    with pytest.raises(ValueError, match="Invalid search_mode"):
        MemoryProcessor(
            db_client=FalkorDBClient(),
            chunker=SemanticChunker(),
            embedder=OllamaEmbedder(),
            config=config
        )
```

---

### Edge Case 8: All optional dependencies None

**Scenario:** User provides minimal Phase 1 configuration (no extractor, cache, task_manager)

**Expected Behavior:** Success, all optional attributes set to None

**Test Scenario:**
```python
def test_init_minimal_config_success():
    processor = MemoryProcessor(
        db_client=FalkorDBClient(),
        chunker=SemanticChunker(),
        embedder=OllamaEmbedder()
    )
    assert processor.extractor is None
    assert processor.cache is None
    assert processor.task_manager is None
    assert processor.config is not None  # Default config created
```

---

### Edge Case 9: Config is None (uses defaults)

**Scenario:** User doesn't provide config parameter

**Expected Behavior:** Default ProcessorConfig() created

**Test Scenario:**
```python
def test_init_no_config_uses_defaults():
    processor = MemoryProcessor(
        db_client=FalkorDBClient(),
        chunker=SemanticChunker(),
        embedder=OllamaEmbedder()
    )
    assert processor.config is not None
    assert processor.config.max_text_length == 10_000_000  # Default
    assert processor.config.batch_size == 32  # Default
    assert processor.config.search_mode == "vector"  # Default
```

---

### Edge Case 10: Wrong type for optional dependency

**Scenario:** User provides extractor as string instead of EntityExtractor

**Expected Behavior:**
```python
raise TypeError("extractor must be EntityExtractor, got <class 'str'>")
```

**Test Scenario:**
```python
def test_init_wrong_type_extractor_raises():
    with pytest.raises(TypeError, match="extractor must be EntityExtractor"):
        MemoryProcessor(
            db_client=FalkorDBClient(),
            chunker=SemanticChunker(),
            embedder=OllamaEmbedder(),
            extractor="extractor"
        )
```

---

## Test Scenarios (Complete List)

### Happy Path Tests

1. **test_init_minimal_phase1_success**
   - Input: db_client, chunker, embedder (all required)
   - Expected: Success, optional attributes None, default config created

2. **test_init_full_phase2_success**
   - Input: All dependencies including extractor, cache, task_manager
   - Expected: Success, all attributes set correctly

3. **test_init_custom_config_success**
   - Input: Required deps + custom ProcessorConfig
   - Expected: Success, config attributes match custom config

4. **test_init_logger_initialized**
   - Input: Valid dependencies
   - Expected: self.logger is not None, has component context

### Error Tests - Required Dependencies

5. **test_init_missing_db_client_raises**
   - Input: db_client=None
   - Expected: ValueError("db_client is required")

6. **test_init_missing_chunker_raises**
   - Input: chunker=None
   - Expected: ValueError("chunker is required")

7. **test_init_missing_embedder_raises**
   - Input: embedder=None
   - Expected: ValueError("embedder is required")

### Error Tests - Type Validation

8. **test_init_wrong_type_db_client_raises**
   - Input: db_client="localhost"
   - Expected: TypeError("db_client must be FalkorDBClient")

9. **test_init_wrong_type_chunker_raises**
   - Input: chunker=123
   - Expected: TypeError("chunker must be SemanticChunker")

10. **test_init_wrong_type_embedder_raises**
    - Input: embedder=object()
    - Expected: TypeError("embedder must be OllamaEmbedder")

11. **test_init_wrong_type_extractor_raises**
    - Input: extractor="extractor"
    - Expected: TypeError("extractor must be EntityExtractor")

12. **test_init_wrong_type_cache_raises**
    - Input: cache={}
    - Expected: TypeError("cache must be SemanticCache")

13. **test_init_wrong_type_task_manager_raises**
    - Input: task_manager=[]
    - Expected: TypeError("task_manager must be TaskManager")

14. **test_init_wrong_type_config_raises**
    - Input: config="config"
    - Expected: TypeError("config must be ProcessorConfig")

### Error Tests - Config Validation

15. **test_init_zero_max_text_length_raises**
    - Input: config=ProcessorConfig(max_text_length=0)
    - Expected: ValueError("max_text_length must be positive")

16. **test_init_negative_max_text_length_raises**
    - Input: config=ProcessorConfig(max_text_length=-1000)
    - Expected: ValueError("max_text_length must be positive")

17. **test_init_zero_batch_size_raises**
    - Input: config=ProcessorConfig(batch_size=0)
    - Expected: ValueError("batch_size must be positive")

18. **test_init_invalid_search_mode_raises**
    - Input: config=ProcessorConfig(search_mode="invalid")
    - Expected: ValueError("Invalid search_mode")

### Attribute Tests

19. **test_init_stores_all_dependencies**
    - Input: All dependencies provided
    - Expected: self.db_client, self.chunker, etc. all match input

20. **test_init_creates_default_config_when_none**
    - Input: config=None
    - Expected: self.config is ProcessorConfig with defaults

---

## Performance Requirements

**Execution Time:**
- Minimal config: < 5ms
- Full config: < 10ms (includes structlog logger initialization)

**Memory:**
- < 1KB overhead (just stores references to dependencies)

**CPU:**
- O(1) operations only (no expensive validation)

---

## Security Considerations

**Input Validation:**
- ✅ All dependencies validated (type and nullability)
- ✅ Config values validated (positive numbers, valid modes)
- ✅ No external input (dependencies provided by trusted code)

**Data Protection:**
- Dependencies may contain sensitive config (DB credentials, API keys)
- Not logged (only configuration values logged, not connection strings)

---

## Related Functions

**Calls:**
- `ProcessorConfig()` - Create default config if not provided
- `structlog.get_logger()` - Create logger instance
- `logger.bind()` - Bind context to logger
- `logger.info()` - Log initialization success

**Called By:**
- Application startup code (creates processor instance)
- Test fixtures (creates processor with mocked dependencies)
- Factory functions (e.g., `create_memory_processor()`)

---

## Implementation Notes

**Libraries Used:**
- `structlog` - Structured logging
- `typing` - Type hints (Optional, etc.)

**Design Patterns:**
- **Dependency Injection:** All dependencies provided externally
- **Fail-Fast:** Validate all inputs in constructor, not in methods
- **Default Parameters:** Optional dependencies default to None

**Thread Safety:**
- Constructor is not thread-safe (but only called once per instance)
- No shared mutable state (all attributes are final after __init__)

---

## References

**Component Spec:**
- [memory_processor_component.md](../level2/memory_processor_component.md) - Parent component

**Related Components:**
- [semantic_chunker_component.md](../level2/semantic_chunker_component.md) - Chunker dependency
- [ollama_embedder_component.md](../level2/ollama_embedder_component.md) - Embedder dependency
- [falkordb_client_component.md](../level2/falkordb_client_component.md) - DB client dependency

**External Documentation:**
- Dependency Injection: https://en.wikipedia.org/wiki/Dependency_injection
- Python Type Hints: https://docs.python.org/3/library/typing.html
- Structlog: https://www.structlog.org/

---

**Document Status:** Draft v1.0
**Created:** 2025-11-23
**Author:** Goncharenko Anton aka alienxs2
**License:** MIT
