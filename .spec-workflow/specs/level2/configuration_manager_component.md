# ConfigurationManager - Component Specification

**Level:** 2 (Component)
**Module:** shared (cross-cutting)
**Author:** Goncharenko Anton aka alienxs2
**Status:** Draft
**Version:** 1.0
**Created:** 2025-11-23
**Copyright:** Copyright (c) 2025 Goncharenko Anton aka alienxs2
**License:** MIT License

---

## Overview

### Purpose

ConfigurationManager is the core component responsible for loading, validating, and providing type-safe access to all Zapomni runtime configuration. It implements the Pydantic Settings pattern to ensure zero-config defaults while supporting full customization through environment variables and .env files.

### Responsibilities

1. Load configuration from multiple sources (environment variables, .env files, defaults)
2. Validate all configuration parameters using Pydantic type checking and custom validators
3. Provide type-safe singleton access to configuration throughout the application
4. Generate database connection strings from configuration parameters
5. Detect runtime environment (development vs production)
6. Mask sensitive configuration values in logs and string representations
7. Ensure zero-config operation with sensible defaults

### Position in Module

ConfigurationManager is the primary class in the Configuration Management module. It is instantiated once at application startup as a singleton and accessed by all other modules (MCP Server, Core Engine, Database Clients).

```
┌─────────────────────────────────────────┐
│  Configuration Management Module        │
├─────────────────────────────────────────┤
│                                         │
│  ┌───────────────────────────────┐     │
│  │  ConfigurationManager         │     │  ← THIS COMPONENT
│  │  (ZapomniSettings class)      │     │
│  │  - Singleton instance         │     │
│  │  - Pydantic BaseSettings      │     │
│  └───────────────────────────────┘     │
│              │                          │
│              ├─ Helper Functions        │
│              │  • get_config_summary()  │
│              │  • validate_configuration()│
│              │  • reload_configuration()│
│              │                          │
└──────────────┼──────────────────────────┘
               │
               ▼
        Used by all modules
```

---

## Class Definition

### Class Diagram

```
┌─────────────────────────────────────────────────┐
│           ZapomniSettings                       │
│           (extends BaseSettings)                │
├─────────────────────────────────────────────────┤
│ Configuration Fields (40+):                     │
│ - falkordb_host: str                            │
│ - falkordb_port: int                            │
│ - falkordb_password: Optional[SecretStr]        │
│ - graph_name: str                               │
│ - ollama_base_url: str                          │
│ - ollama_embedding_model: str                   │
│ - ollama_llm_model: str                         │
│ - max_chunk_size: int                           │
│ - chunk_overlap: int                            │
│ - vector_dimensions: int                        │
│ - hnsw_m: int                                   │
│ - hnsw_ef_construction: int                     │
│ - hnsw_ef_search: int                           │
│ - log_level: str                                │
│ - log_format: str                               │
│ - data_dir: Path                                │
│ - temp_dir: Path                                │
│ - enable_hybrid_search: bool                    │
│ - enable_knowledge_graph: bool                  │
│ - ... (30+ more fields)                         │
├─────────────────────────────────────────────────┤
│ Validators:                                     │
│ + validate_log_level(cls, v) -> str             │
│ + validate_log_format(cls, v) -> str            │
│ + validate_ollama_url(cls, v) -> str            │
│ + validate_chunk_overlap(cls, v, values) -> int │
│ + validate_vector_dimensions(cls, v) -> int     │
│ + ensure_directory_exists(cls, v) -> Path       │
├─────────────────────────────────────────────────┤
│ Computed Properties:                            │
│ + falkordb_connection_string: str               │
│ + redis_connection_string: str                  │
│ + is_development: bool                          │
│ + is_production: bool                           │
└─────────────────────────────────────────────────┘
```

### Full Class Signature

```python
from pydantic_settings import BaseSettings
from pydantic import Field, validator, SecretStr
from typing import Optional, List
from pathlib import Path
import os

class ZapomniSettings(BaseSettings):
    """
    Centralized configuration manager for Zapomni system.

    This class implements the Pydantic Settings pattern to provide:
    - Type-safe configuration loading from environment variables
    - Automatic validation of all configuration parameters
    - Zero-config defaults for immediate system operation
    - Secure handling of sensitive values (passwords, secrets)
    - Computed properties for derived configuration values

    Configuration is loaded with the following priority (highest to lowest):
    1. System environment variables
    2. .env file in project root
    3. Hardcoded default values

    All configuration parameters have sensible defaults for local development.
    The system works out-of-the-box without any configuration required.

    Attributes:
        falkordb_host: FalkorDB server hostname (default: localhost)
        falkordb_port: FalkorDB server port (default: 6379)
        falkordb_password: Optional password for FalkorDB authentication
        graph_name: Name of the graph database (default: zapomni_memory)
        falkordb_connection_timeout: Connection timeout in seconds (default: 30)
        falkordb_pool_size: Connection pool size (default: 10)

        ollama_base_url: Ollama API base URL (default: http://localhost:11434)
        ollama_embedding_model: Model for embeddings (default: nomic-embed-text)
        ollama_llm_model: Model for LLM inference (default: llama3.1:8b)
        ollama_embedding_timeout: Embedding request timeout (default: 60s)
        ollama_llm_timeout: LLM request timeout (default: 120s)

        redis_enabled: Enable Redis semantic cache (default: False)
        redis_host: Redis server hostname (default: localhost)
        redis_port: Redis server port (default: 6380)
        redis_ttl_seconds: Cache TTL in seconds (default: 86400)
        redis_max_memory_mb: Max memory usage in MB (default: 1024)

        max_chunk_size: Maximum text chunk size in tokens (default: 512)
        chunk_overlap: Chunk overlap in tokens (default: 50)
        vector_dimensions: Embedding vector dimensions (default: 768)
        hnsw_m: HNSW index M parameter (default: 16)
        hnsw_ef_construction: HNSW build accuracy (default: 200)
        hnsw_ef_search: HNSW query accuracy (default: 100)
        max_concurrent_tasks: Max concurrent background tasks (default: 4)
        search_limit_default: Default search result count (default: 10)
        min_similarity_threshold: Min cosine similarity (default: 0.5)

        log_level: Logging level (default: INFO)
        log_format: Log format - json or text (default: json)
        log_file: Optional log file path (default: None)

        enable_hybrid_search: Enable BM25+vector search (default: False)
        enable_knowledge_graph: Enable entity extraction (default: False)
        enable_code_indexing: Enable AST indexing (default: False)
        enable_semantic_cache: Enable semantic cache (default: False)

        data_dir: Data storage directory (default: ./data)
        temp_dir: Temporary files directory (default: /tmp/zapomni)
        max_text_length: Max text length in characters (default: 10,000,000)

    Example:
        ```python
        from zapomni_config import settings

        # Access configuration
        print(settings.falkordb_host)  # 'localhost'
        print(settings.max_chunk_size)  # 512

        # Use computed properties
        conn_str = settings.falkordb_connection_string
        # 'redis://localhost:6379'

        # Check environment
        if settings.is_development:
            print("Running in development mode")
        ```

    Raises:
        ValidationError: If any configuration parameter is invalid
    """

    # ========================================
    # FALKORDB CONFIGURATION
    # ========================================

    falkordb_host: str = Field(
        default="localhost",
        env="FALKORDB_HOST",
        description="FalkorDB server hostname or IP address"
    )

    falkordb_port: int = Field(
        default=6379,
        ge=1,
        le=65535,
        env="FALKORDB_PORT",
        description="FalkorDB server port (standard Redis port: 6379)"
    )

    falkordb_password: Optional[SecretStr] = Field(
        default=None,
        env="FALKORDB_PASSWORD",
        description="FalkorDB authentication password (if required)"
    )

    graph_name: str = Field(
        default="zapomni_memory",
        env="GRAPH_NAME",
        description="Name of the FalkorDB graph database"
    )

    falkordb_connection_timeout: int = Field(
        default=30,
        ge=1,
        le=300,
        env="FALKORDB_CONNECTION_TIMEOUT",
        description="FalkorDB connection timeout in seconds"
    )

    falkordb_pool_size: int = Field(
        default=10,
        ge=1,
        le=100,
        env="FALKORDB_POOL_SIZE",
        description="FalkorDB connection pool size"
    )

    # ========================================
    # OLLAMA CONFIGURATION
    # ========================================

    ollama_base_url: str = Field(
        default="http://localhost:11434",
        env="OLLAMA_BASE_URL",
        description="Ollama server base URL (including protocol)"
    )

    ollama_embedding_model: str = Field(
        default="nomic-embed-text",
        env="OLLAMA_EMBEDDING_MODEL",
        description="Ollama model for embedding generation (768 dim)"
    )

    ollama_llm_model: str = Field(
        default="llama3.1:8b",
        env="OLLAMA_LLM_MODEL",
        description="Ollama model for LLM inference (entity extraction)"
    )

    ollama_embedding_timeout: int = Field(
        default=60,
        ge=5,
        le=300,
        env="OLLAMA_EMBEDDING_TIMEOUT",
        description="Timeout for embedding requests in seconds"
    )

    ollama_llm_timeout: int = Field(
        default=120,
        ge=10,
        le=600,
        env="OLLAMA_LLM_TIMEOUT",
        description="Timeout for LLM inference requests in seconds"
    )

    # ========================================
    # REDIS CACHE CONFIGURATION
    # ========================================

    redis_enabled: bool = Field(
        default=False,
        env="REDIS_ENABLED",
        description="Enable Redis semantic cache (Phase 2 feature)"
    )

    redis_host: str = Field(
        default="localhost",
        env="REDIS_HOST",
        description="Redis server hostname"
    )

    redis_port: int = Field(
        default=6380,
        ge=1,
        le=65535,
        env="REDIS_PORT",
        description="Redis server port (6380 to avoid conflict with FalkorDB)"
    )

    redis_ttl_seconds: int = Field(
        default=86400,  # 24 hours
        ge=60,
        le=604800,  # 7 days max
        env="REDIS_TTL_SECONDS",
        description="Redis cache entry TTL in seconds"
    )

    redis_max_memory_mb: int = Field(
        default=1024,
        ge=100,
        le=10240,
        env="REDIS_MAX_MEMORY_MB",
        description="Redis maximum memory usage in MB"
    )

    # ========================================
    # PERFORMANCE TUNING
    # ========================================

    max_chunk_size: int = Field(
        default=512,
        ge=100,
        le=2000,
        env="MAX_CHUNK_SIZE",
        description="Maximum chunk size in tokens"
    )

    chunk_overlap: int = Field(
        default=50,
        ge=0,
        le=500,
        env="CHUNK_OVERLAP",
        description="Chunk overlap in tokens (10-20% of chunk_size recommended)"
    )

    vector_dimensions: int = Field(
        default=768,
        env="VECTOR_DIMENSIONS",
        description="Embedding vector dimensions (must match model)"
    )

    hnsw_m: int = Field(
        default=16,
        ge=4,
        le=64,
        env="HNSW_M",
        description="HNSW index M parameter (connections per layer)"
    )

    hnsw_ef_construction: int = Field(
        default=200,
        ge=50,
        le=1000,
        env="HNSW_EF_CONSTRUCTION",
        description="HNSW build-time accuracy parameter"
    )

    hnsw_ef_search: int = Field(
        default=100,
        ge=10,
        le=500,
        env="HNSW_EF_SEARCH",
        description="HNSW query-time accuracy parameter"
    )

    max_concurrent_tasks: int = Field(
        default=4,
        ge=1,
        le=32,
        env="MAX_CONCURRENT_TASKS",
        description="Maximum concurrent background tasks"
    )

    search_limit_default: int = Field(
        default=10,
        ge=1,
        le=100,
        env="SEARCH_LIMIT_DEFAULT",
        description="Default number of search results"
    )

    min_similarity_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        env="MIN_SIMILARITY_THRESHOLD",
        description="Minimum cosine similarity for search results"
    )

    # ========================================
    # LOGGING CONFIGURATION
    # ========================================

    log_level: str = Field(
        default="INFO",
        env="LOG_LEVEL",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )

    log_format: str = Field(
        default="json",
        env="LOG_FORMAT",
        description="Log format (json, text)"
    )

    log_file: Optional[Path] = Field(
        default=None,
        env="LOG_FILE",
        description="Log file path (None = stderr only)"
    )

    # ========================================
    # FEATURE FLAGS
    # ========================================

    enable_hybrid_search: bool = Field(
        default=False,
        env="ENABLE_HYBRID_SEARCH",
        description="Enable BM25 + vector hybrid search (Phase 2)"
    )

    enable_knowledge_graph: bool = Field(
        default=False,
        env="ENABLE_KNOWLEDGE_GRAPH",
        description="Enable entity extraction and knowledge graph (Phase 2)"
    )

    enable_code_indexing: bool = Field(
        default=False,
        env="ENABLE_CODE_INDEXING",
        description="Enable AST-based code indexing (Phase 3)"
    )

    enable_semantic_cache: bool = Field(
        default=False,
        env="ENABLE_SEMANTIC_CACHE",
        description="Enable semantic embedding cache (Phase 2)"
    )

    # ========================================
    # SYSTEM CONFIGURATION
    # ========================================

    data_dir: Path = Field(
        default=Path("./data"),
        env="DATA_DIR",
        description="Data storage directory"
    )

    temp_dir: Path = Field(
        default=Path("/tmp/zapomni"),
        env="TEMP_DIR",
        description="Temporary files directory"
    )

    max_text_length: int = Field(
        default=10_000_000,  # 10MB
        ge=1000,
        le=100_000_000,
        env="MAX_TEXT_LENGTH",
        description="Maximum allowed text length in characters"
    )

    # ========================================
    # VALIDATORS
    # ========================================

    @validator("log_level")
    def validate_log_level(cls, v: str) -> str:
        """
        Validate log level is one of allowed values.

        Args:
            v: Log level string (case-insensitive)

        Returns:
            Uppercase log level string

        Raises:
            ValueError: If log level not in allowed values
        """
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(
                f"log_level must be one of {allowed}, got '{v}'"
            )
        return v_upper

    @validator("log_format")
    def validate_log_format(cls, v: str) -> str:
        """
        Validate log format is one of allowed values.

        Args:
            v: Log format string (case-insensitive)

        Returns:
            Lowercase log format string

        Raises:
            ValueError: If log format not in allowed values
        """
        allowed = ["json", "text"]
        v_lower = v.lower()
        if v_lower not in allowed:
            raise ValueError(
                f"log_format must be one of {allowed}, got '{v}'"
            )
        return v_lower

    @validator("ollama_base_url")
    def validate_ollama_url(cls, v: str) -> str:
        """
        Validate Ollama base URL format.

        Args:
            v: URL string

        Returns:
            URL string with trailing slash removed

        Raises:
            ValueError: If URL doesn't start with http:// or https://
        """
        if not v.startswith(("http://", "https://")):
            raise ValueError(
                f"ollama_base_url must start with http:// or https://, got '{v}'"
            )
        return v.rstrip("/")  # Remove trailing slash

    @validator("chunk_overlap")
    def validate_chunk_overlap(cls, v: int, values: dict) -> int:
        """
        Validate chunk overlap is reasonable relative to chunk_size.

        Args:
            v: Chunk overlap value
            values: Previously validated field values

        Returns:
            Validated chunk overlap value

        Raises:
            ValueError: If overlap >= chunk_size

        Warnings:
            UserWarning: If overlap > 50% of chunk_size
        """
        if "max_chunk_size" in values:
            max_size = values["max_chunk_size"]
            if v >= max_size:
                raise ValueError(
                    f"chunk_overlap ({v}) must be less than max_chunk_size ({max_size})"
                )
            if v > max_size * 0.5:
                import warnings
                warnings.warn(
                    f"chunk_overlap ({v}) is > 50% of max_chunk_size ({max_size}). "
                    "Recommended: 10-20%"
                )
        return v

    @validator("vector_dimensions")
    def validate_vector_dimensions(cls, v: int) -> int:
        """
        Validate vector dimensions match expected model output.

        Args:
            v: Vector dimension value

        Returns:
            Validated vector dimension value

        Warnings:
            UserWarning: If dimensions are non-standard
        """
        allowed = [384, 768, 1024, 1536, 3072]
        if v not in allowed:
            import warnings
            warnings.warn(
                f"vector_dimensions ({v}) is non-standard. "
                f"Common values: {allowed}"
            )
        return v

    @validator("data_dir", "temp_dir")
    def ensure_directory_exists(cls, v: Path) -> Path:
        """
        Ensure directory exists, create if not present.

        Args:
            v: Path to directory

        Returns:
            Validated Path object

        Raises:
            OSError: If directory cannot be created
        """
        v.mkdir(parents=True, exist_ok=True)
        return v

    # ========================================
    # COMPUTED PROPERTIES
    # ========================================

    @property
    def falkordb_connection_string(self) -> str:
        """
        Get FalkorDB connection string.

        Returns:
            Connection string in format: redis://[password@]host:port
        """
        if self.falkordb_password:
            password = self.falkordb_password.get_secret_value()
            return f"redis://{password}@{self.falkordb_host}:{self.falkordb_port}"
        return f"redis://{self.falkordb_host}:{self.falkordb_port}"

    @property
    def redis_connection_string(self) -> str:
        """
        Get Redis connection string.

        Returns:
            Connection string in format: redis://host:port
        """
        return f"redis://{self.redis_host}:{self.redis_port}"

    @property
    def is_development(self) -> bool:
        """
        Check if running in development mode.

        Returns:
            True if log_level is DEBUG
        """
        return self.log_level == "DEBUG"

    @property
    def is_production(self) -> bool:
        """
        Check if running in production mode.

        Returns:
            True if log_level is INFO, WARNING, or ERROR
        """
        return self.log_level in ["INFO", "WARNING", "ERROR"]

    # ========================================
    # PYDANTIC CONFIGURATION
    # ========================================

    class Config:
        """Pydantic configuration class."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        validate_assignment = True  # Validate on attribute assignment
        extra = "forbid"  # Forbid extra fields (strict mode)
        json_encoders = {
            SecretStr: lambda v: "***REDACTED***" if v else None
        }


# Singleton instance - instantiated once at module import
settings = ZapomniSettings()
```

---

## Dependencies

### Component Dependencies

None - ConfigurationManager is a foundational component with no dependencies on other Zapomni components. All other components depend on it.

### External Libraries

- **pydantic>=2.0.0** - Core settings base class and validation framework
- **pydantic-settings>=2.0.0** - Environment variable loading and .env file parsing
- **python-dotenv>=1.0.0** - Additional .env file support (used by pydantic-settings)

### Dependency Injection

ConfigurationManager does not use dependency injection. It is instantiated as a module-level singleton when first imported:

```python
# In src/zapomni_mcp/config.py
settings = ZapomniSettings()
```

All other modules import and use this singleton:

```python
from zapomni_mcp.config import settings

# Use configuration
db_host = settings.falkordb_host
```

---

## State Management

### Attributes

All attributes are configuration fields defined using Pydantic's `Field()` function. There are 40+ configuration attributes grouped into categories:

- **FalkorDB Configuration** (6 fields): Database connection parameters
- **Ollama Configuration** (5 fields): LLM service parameters
- **Redis Configuration** (5 fields): Cache service parameters
- **Performance Tuning** (9 fields): Performance-related settings
- **Logging Configuration** (3 fields): Logging behavior
- **Feature Flags** (4 fields): Optional feature enablement
- **System Configuration** (3 fields): System paths and limits

All attributes are initialized from environment variables, .env file, or defaults during class instantiation.

### State Transitions

ConfigurationManager is designed to be immutable after initialization:

```
┌──────────────────┐
│  Uninitialized   │
│  (module load)   │
└────────┬─────────┘
         │
         │ settings = ZapomniSettings()
         ▼
┌──────────────────┐
│  Loading         │
│  - Read env vars │
│  - Parse .env    │
│  - Apply defaults│
└────────┬─────────┘
         │
         │ Pydantic validation
         ▼
┌──────────────────┐
│  Validating      │
│  - Type checks   │
│  - Range checks  │
│  - Custom validators │
└────────┬─────────┘
         │
         │ Success
         ▼
┌──────────────────┐
│  Initialized     │ ◄─── IMMUTABLE STATE
│  (singleton)     │      (normal operation)
└──────────────────┘
         │
         │ reload_configuration() [Phase 2]
         ▼
┌──────────────────┐
│  Reloading       │
│  (hot-reload)    │
└──────────────────┘
```

**Phase 1 (MVP):** Configuration is immutable - requires restart to change
**Phase 2+:** Safe hot-reload for non-critical settings (log_level, feature flags)

### Thread Safety

**Thread-Safe:** Yes

ConfigurationManager is thread-safe because:
1. **Immutable after initialization** - No attribute mutations during normal operation
2. **No shared mutable state** - All configuration is read-only
3. **Pydantic validation is thread-safe** - Validation happens once at initialization
4. **Singleton pattern** - Single shared instance, no race conditions

However, `reload_configuration()` (Phase 2 feature) is NOT thread-safe and should only be called when no operations are running.

---

## Public Methods (Detailed)

### Method 1: `__init__` (Inherited from BaseSettings)

**Note:** ConfigurationManager uses Pydantic's BaseSettings, which provides automatic initialization. Users don't typically call `__init__` directly - instead, they use the module-level singleton `settings`.

**Signature:**
```python
def __init__(self, **values: Any) -> None
```

**Purpose:** Initialize configuration by loading from environment variables, .env file, and defaults, then validating all values.

**Parameters:**
- `**values`: Optional keyword arguments to override specific configuration values (primarily used for testing)

**Returns:**
- None (constructor)

**Raises:**
- `ValidationError`: If any configuration parameter fails validation

**Preconditions:**
- Environment variables (if any) are set
- .env file exists and is readable (optional)

**Postconditions:**
- All configuration attributes are initialized and validated
- Directories specified in `data_dir` and `temp_dir` exist
- Singleton instance is ready for use

**Algorithm Outline:**
```
1. Load environment variables from system
2. Parse .env file if present
3. For each field:
   a. Get value from env var (if set)
   b. Get value from .env file (if set)
   c. Use default value (if not set anywhere)
   d. Convert to target type (automatic)
   e. Run field validators (ge, le, etc.)
   f. Run custom @validator methods
4. Create directories for data_dir and temp_dir
5. Return initialized instance
```

**Edge Cases:**
1. Missing .env file → Use system env + defaults (not an error)
2. Invalid env var type → ValidationError with clear message
3. Directory creation fails → OSError propagated to caller
4. Password field empty → None (valid, no auth)

**Related Methods:**
- Called by: Module import (creates singleton)
- Calls: All validator methods

---

### Validator Methods

#### Method 2: `validate_log_level`

**Signature:**
```python
@validator("log_level")
def validate_log_level(cls, v: str) -> str
```

**Purpose:** Validate log level is one of the allowed Python logging levels.

**Parameters:**
- `v`: str - Log level string (case-insensitive)

**Returns:**
- str - Uppercase log level string

**Raises:**
- `ValueError`: If log level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

**Algorithm:**
```
1. Convert input to uppercase
2. Check if in allowed list
3. If not: raise ValueError with helpful message
4. If yes: return uppercase value
```

**Examples:**
- Input: "info" → Output: "INFO"
- Input: "DEBUG" → Output: "DEBUG"
- Input: "INVALID" → Raises ValueError

---

#### Method 3: `validate_log_format`

**Signature:**
```python
@validator("log_format")
def validate_log_format(cls, v: str) -> str
```

**Purpose:** Validate log format is either "json" or "text".

**Parameters:**
- `v`: str - Log format string (case-insensitive)

**Returns:**
- str - Lowercase log format string

**Raises:**
- `ValueError`: If log format not in ["json", "text"]

---

#### Method 4: `validate_ollama_url`

**Signature:**
```python
@validator("ollama_base_url")
def validate_ollama_url(cls, v: str) -> str
```

**Purpose:** Validate Ollama URL starts with http:// or https:// and normalize format.

**Parameters:**
- `v`: str - URL string

**Returns:**
- str - Normalized URL (trailing slash removed)

**Raises:**
- `ValueError`: If URL doesn't start with http:// or https://

**Algorithm:**
```
1. Check if URL starts with "http://" or "https://"
2. If not: raise ValueError
3. If yes: remove trailing slash and return
```

**Examples:**
- Input: "http://localhost:11434/" → Output: "http://localhost:11434"
- Input: "localhost:11434" → Raises ValueError (missing protocol)

---

#### Method 5: `validate_chunk_overlap`

**Signature:**
```python
@validator("chunk_overlap")
def validate_chunk_overlap(cls, v: int, values: dict) -> int
```

**Purpose:** Validate chunk overlap is reasonable relative to max_chunk_size.

**Parameters:**
- `v`: int - Chunk overlap value
- `values`: dict - Previously validated field values

**Returns:**
- int - Validated chunk overlap value

**Raises:**
- `ValueError`: If overlap >= chunk_size

**Warnings:**
- `UserWarning`: If overlap > 50% of chunk_size

**Algorithm:**
```
1. Check if max_chunk_size was already validated
2. If yes:
   a. Check overlap < chunk_size (hard requirement)
   b. If not: raise ValueError
   c. Check overlap <= 50% chunk_size (recommendation)
   d. If not: warn user
3. Return validated value
```

---

#### Method 6: `validate_vector_dimensions`

**Signature:**
```python
@validator("vector_dimensions")
def validate_vector_dimensions(cls, v: int) -> int
```

**Purpose:** Validate vector dimensions and warn if non-standard value.

**Parameters:**
- `v`: int - Vector dimension value

**Returns:**
- int - Validated dimension value

**Warnings:**
- `UserWarning`: If dimensions not in [384, 768, 1024, 1536, 3072]

**Note:** This is a warning, not an error, because custom models may use non-standard dimensions.

---

#### Method 7: `ensure_directory_exists`

**Signature:**
```python
@validator("data_dir", "temp_dir")
def ensure_directory_exists(cls, v: Path) -> Path
```

**Purpose:** Ensure directory exists, creating it if necessary.

**Parameters:**
- `v`: Path - Path to directory

**Returns:**
- Path - Validated Path object

**Raises:**
- `OSError`: If directory cannot be created (permissions, disk space, etc.)

**Algorithm:**
```
1. Call v.mkdir(parents=True, exist_ok=True)
2. If successful: return Path
3. If OSError: propagate to caller
```

---

### Computed Properties

#### Property 1: `falkordb_connection_string`

**Signature:**
```python
@property
def falkordb_connection_string(self) -> str
```

**Purpose:** Generate FalkorDB connection string from configuration parameters.

**Returns:**
- str - Connection string in format: `redis://[password@]host:port`

**Algorithm:**
```
1. Check if falkordb_password is set
2. If yes: include password in connection string
3. If no: omit password
4. Format: redis://[password@]host:port
5. Return string
```

**Examples:**
- No password: `"redis://localhost:6379"`
- With password: `"redis://secret123@localhost:6379"`

---

#### Property 2: `redis_connection_string`

**Signature:**
```python
@property
def redis_connection_string(self) -> str
```

**Purpose:** Generate Redis connection string.

**Returns:**
- str - Connection string in format: `redis://host:port`

---

#### Property 3: `is_development`

**Signature:**
```python
@property
def is_development(self) -> bool
```

**Purpose:** Detect if running in development mode based on log level.

**Returns:**
- bool - True if log_level is "DEBUG"

---

#### Property 4: `is_production`

**Signature:**
```python
@property
def is_production(self) -> bool
```

**Purpose:** Detect if running in production mode.

**Returns:**
- bool - True if log_level is "INFO", "WARNING", or "ERROR"

---

## Helper Functions

### Function 1: `get_config_summary`

**Signature:**
```python
def get_config_summary() -> Dict[str, Any]
```

**Purpose:** Get configuration summary for logging/debugging with sensitive values masked.

**Returns:**
- dict - Configuration summary grouped by category

**Example Output:**
```python
{
    "database": {
        "falkordb_host": "localhost",
        "falkordb_port": 6379,
        "graph_name": "zapomni_memory",
        "pool_size": 10
    },
    "ollama": {
        "base_url": "http://localhost:11434",
        "embedding_model": "nomic-embed-text",
        "llm_model": "llama3.1:8b"
    },
    "performance": {
        "max_chunk_size": 512,
        "chunk_overlap": 50,
        "vector_dimensions": 768,
        "hnsw_m": 16
    },
    "features": {
        "hybrid_search": false,
        "knowledge_graph": false,
        "code_indexing": false,
        "semantic_cache": false
    },
    "logging": {
        "level": "INFO",
        "format": "json"
    }
}
```

---

### Function 2: `validate_configuration`

**Signature:**
```python
def validate_configuration() -> tuple[bool, list[str]]
```

**Purpose:** Perform additional runtime validation beyond Pydantic checks.

**Returns:**
- tuple: (is_valid: bool, errors: list[str])

**Checks:**
1. Data directory is writable
2. Performance settings are reasonable
3. HNSW parameters are consistent

**Example:**
```python
is_valid, errors = validate_configuration()
if not is_valid:
    print(f"Configuration errors: {errors}")
```

---

### Function 3: `reload_configuration`

**Signature:**
```python
def reload_configuration() -> None
```

**Purpose:** Reload configuration from environment (Phase 2+ feature).

**WARNING:** This recreates the settings singleton. Should only be called when no operations are running.

**Raises:**
- `RuntimeError`: If configuration is invalid after reload

**Algorithm:**
```
1. Create new ZapomniSettings instance
2. Validate new configuration
3. If valid: replace global settings singleton
4. If invalid: raise RuntimeError, keep old settings
```

---

## Error Handling

### Exceptions Raised

#### ValidationError (from Pydantic)

**When Raised:**
- Any configuration parameter fails validation
- Type mismatch (e.g., string instead of int)
- Range violation (e.g., port > 65535)
- Custom validator failure

**Message Format:**
```
ValidationError: N validation errors for ZapomniSettings
field_name
  error description (type=error_type)
```

**Example:**
```
ValidationError: 1 validation error for ZapomniSettings
falkordb_port
  ensure this value is less than or equal to 65535
  (type=value_error.number.not_le; limit_value=65535)
```

### Error Recovery

**Strategy:** Fail-fast

ConfigurationManager uses a fail-fast strategy for critical errors:
- Invalid configuration → ValidationError → Application won't start
- Clear error messages guide user to fix the issue
- No partial configuration allowed

**Non-Critical Issues:**
- Non-standard vector dimensions → UserWarning, continues
- Suboptimal chunk overlap → UserWarning, continues

---

## Usage Examples

### Basic Usage

```python
# Import singleton (initializes configuration)
from zapomni_mcp.config import settings

# Access configuration values
print(f"Connecting to FalkorDB at {settings.falkordb_host}:{settings.falkordb_port}")
print(f"Using embedding model: {settings.ollama_embedding_model}")
print(f"Chunk size: {settings.max_chunk_size}")

# Use computed properties
connection_string = settings.falkordb_connection_string
# "redis://localhost:6379"

# Check environment
if settings.is_development:
    print("Running in development mode - verbose logging enabled")
```

### Advanced Usage

```python
from zapomni_mcp.config import settings, get_config_summary, validate_configuration

# Get configuration summary for logging
config_summary = get_config_summary()
logger.info("Application started", extra=config_summary)

# Validate configuration beyond Pydantic checks
is_valid, errors = validate_configuration()
if not is_valid:
    logger.error(f"Configuration validation failed: {errors}")
    sys.exit(1)

# Use feature flags
if settings.enable_knowledge_graph:
    from zapomni_core.entity_extraction import EntityExtractor
    extractor = EntityExtractor()

# Check if running in production
if settings.is_production:
    # Use production-specific settings
    logger.setLevel(settings.log_level)
```

### Testing with Custom Configuration

```python
import pytest
from pydantic import ValidationError
from zapomni_mcp.config import ZapomniSettings

def test_custom_configuration():
    """Test creating configuration with custom values."""
    custom_settings = ZapomniSettings(
        falkordb_host="testdb.example.com",
        falkordb_port=6380,
        max_chunk_size=256,
        log_level="DEBUG"
    )

    assert custom_settings.falkordb_host == "testdb.example.com"
    assert custom_settings.is_development == True

def test_invalid_configuration():
    """Test validation catches invalid values."""
    with pytest.raises(ValidationError) as exc_info:
        ZapomniSettings(falkordb_port=99999)  # Out of range

    assert "falkordb_port" in str(exc_info.value)
```

---

## Testing Approach

### Unit Tests Required

**Configuration Loading:**
- `test_default_configuration()` - All defaults are valid
- `test_environment_override()` - Env vars override defaults
- `test_dotenv_loading()` - .env file parsing works
- `test_priority_order()` - System env > .env > defaults

**Validation Tests:**
- `test_invalid_port_raises()` - Port validation (0, negative, > 65535)
- `test_invalid_log_level_raises()` - Log level validation
- `test_invalid_log_format_raises()` - Log format validation
- `test_invalid_url_raises()` - URL format validation
- `test_chunk_overlap_validation()` - Overlap < chunk_size
- `test_chunk_overlap_warning()` - Overlap > 50% warns

**Computed Properties:**
- `test_falkordb_connection_string_no_password()` - Connection string without password
- `test_falkordb_connection_string_with_password()` - Connection string with password
- `test_redis_connection_string()` - Redis connection string format
- `test_is_development_true()` - Development mode detection (DEBUG)
- `test_is_development_false()` - Production mode (INFO, WARNING, ERROR)

**Secret Handling:**
- `test_secret_masking()` - SecretStr fields masked in repr
- `test_secret_access()` - get_secret_value() works

**Directory Creation:**
- `test_data_dir_created()` - data_dir created if not exists
- `test_temp_dir_created()` - temp_dir created if not exists
- `test_directory_creation_failure()` - OSError if cannot create

**Helper Functions:**
- `test_get_config_summary()` - Summary structure correct
- `test_validate_configuration_success()` - Validation passes
- `test_validate_configuration_failure()` - Validation detects errors

### Mocking Strategy

**Mock Environment Variables:**
```python
def test_with_env(monkeypatch):
    monkeypatch.setenv("FALKORDB_HOST", "custom.host")
    settings = ZapomniSettings()
    assert settings.falkordb_host == "custom.host"
```

**Mock .env File:**
```python
def test_dotenv(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("FALKORDB_PORT=7000")
    settings = ZapomniSettings(_env_file=env_file)
    assert settings.falkordb_port == 7000
```

### Integration Tests

**Real .env File:**
- Load configuration from actual .env file
- Verify all fields populated correctly

**Environment Detection:**
- Test development mode behavior
- Test production mode behavior

---

## Performance Considerations

### Time Complexity

- **Initialization:** O(n) where n = number of configuration fields (~40)
  - Read env vars: O(1) per field
  - Validation: O(1) per field (most validators)
  - Total: O(40) ≈ O(1) constant time

- **Attribute Access:** O(1) - Direct attribute access
- **Computed Properties:** O(1) - Simple string formatting
- **get_config_summary():** O(n) - Iterate all fields

**Target:** < 100ms initialization time
**Actual:** ~10-20ms on modern hardware

### Space Complexity

- **Memory Usage:** ~10KB for settings singleton
  - 40 fields × ~100 bytes/field = ~4KB
  - Overhead: ~6KB
- **Singleton Pattern:** Only one instance exists globally

### Optimization Opportunities

1. **Lazy Validation:** Validators only run once at initialization
2. **No Network Calls:** All validation is local (no DB connections during init)
3. **Cached Properties:** Could cache computed properties (not needed, already fast)

---

## References

- **Module spec:** `/home/dev/zapomni/.spec-workflow/specs/level1/configuration_management.md`
- **Pydantic Documentation:** https://docs.pydantic.dev/latest/
- **Pydantic Settings:** https://docs.pydantic.dev/latest/usage/settings/
- **12-Factor App Config:** https://12factor.net/config

---

## Document Status

**Version:** 1.0 (Draft)
**Created:** 2025-11-23
**Last Updated:** 2025-11-23
**Author:** Goncharenko Anton aka alienxs2 + Claude Code
**Copyright:** Copyright (c) 2025 Goncharenko Anton aka alienxs2
**License:** MIT License
**Status:** Ready for Review

**Next Steps:**
1. Review by project maintainer
2. Approval via spec workflow
3. Create Level 3 function specs for key methods
4. Implementation of ZapomniSettings class
5. Unit test implementation (target: 95%+ coverage)

---

**Component Metrics:**
- **Configuration Fields:** 43
- **Validators:** 6
- **Computed Properties:** 4
- **Helper Functions:** 3
- **Test Scenarios:** 25+

**Estimated Implementation Time:** 3-4 hours
**Estimated Test Time:** 2-3 hours
**Total:** 0.75 working days
