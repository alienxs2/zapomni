# Configuration Management - Module Specification

**Level:** 1 (Module)
**Author:** Goncharenko Anton aka alienxs2
**Status:** Draft
**Version:** 1.0
**Created:** 2025-11-23
**Copyright:** Copyright (c) 2025 Goncharenko Anton aka alienxs2
**License:** MIT License

---

## Overview

### Purpose

Configuration Management module provides a centralized, validated, and type-safe system for managing all runtime configuration of Zapomni. It handles environment variables, default values, validation, and provides a single source of truth for configuration throughout the application.

This module embodies Zapomni's "zero-config defaults" philosophy: the system should work out-of-the-box with sensible defaults while remaining fully customizable for advanced use cases.

### Scope

**Included:**
- Environment variable loading from `.env` files and system environment
- Configuration schema definition with types and constraints
- Validation of all configuration parameters (required fields, formats, ranges)
- Default value provision for all optional parameters
- Runtime configuration access (singleton pattern)
- Configuration hot-reload support (for non-breaking changes)
- Security validation (no secrets in code, sensitive data masking)

**Not Included:**
- Persistent configuration storage (this is runtime-only)
- User preferences or session state
- Application state management (handled by core modules)
- Secret management systems (e.g., Vault integration - future enhancement)

### Position in Architecture

Configuration Management is a foundational cross-cutting concern that sits at the **initialization layer** of Zapomni. It is loaded **before** any other modules and provides configuration to:
- MCP Server (`zapomni_mcp`)
- Core Processing Engine (`zapomni_core`)
- Database Clients (`zapomni_db`)

```
┌─────────────────────────────────────┐
│   System Startup                    │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│   Configuration Management          │ ◄── THIS MODULE
│   - Load .env                        │
│   - Validate all settings            │
│   - Provide settings singleton       │
└────────────┬────────────────────────┘
             │
             ├─────────┬─────────┬─────────┐
             ▼         ▼         ▼         ▼
       ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
       │  MCP   │ │  Core  │ │   DB   │ │ Utils  │
       │ Server │ │ Engine │ │ Client │ │        │
       └────────┘ └────────┘ └────────┘ └────────┘
```

---

## Architecture

### High-Level Diagram

```
┌─────────────────────────────────────────────────────────────┐
│  Configuration Sources (Priority Order)                     │
├─────────────────────────────────────────────────────────────┤
│  1. System Environment Variables (highest priority)         │
│  2. .env File (project root)                                │
│  3. Hardcoded Defaults (in code)                            │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│  Pydantic Settings Model (ZapomniSettings)                  │
│  - Type annotations with constraints                        │
│  - Validators for complex rules                             │
│  - Computed properties                                      │
│  - Sensitive field masking                                  │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│  Validation & Error Reporting                               │
│  - Required field checks                                    │
│  - Type validation                                          │
│  - Range and format validation                              │
│  - Connection string parsing                                │
│  - Clear error messages                                     │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│  Settings Singleton (settings)                              │
│  - Immutable configuration object                           │
│  - Type-safe access (IDE autocomplete)                      │
│  - Exported globally                                        │
│  - Hot-reload support (future)                              │
└─────────────────────────────────────────────────────────────┘
```

### Key Responsibilities

1. **Load Configuration from Multiple Sources**
   - Read environment variables from system
   - Parse `.env` file if present
   - Merge with priority (system env > .env > defaults)

2. **Validate All Configuration**
   - Type checking (str, int, bool, float)
   - Range validation (min/max values)
   - Format validation (URLs, file paths)
   - Required vs optional field enforcement
   - Cross-field validation (e.g., host + port consistency)

3. **Provide Zero-Config Defaults**
   - Sensible defaults for local development
   - localhost for all services
   - Standard ports (FalkorDB: 6379, Ollama: 11434)
   - Reasonable performance tuning (chunk_size: 512)

4. **Ensure Security**
   - No hardcoded secrets in source code
   - Sensitive values masked in logs
   - Clear guidance on what needs configuration
   - Validation prevents insecure configurations

5. **Enable Runtime Access**
   - Singleton pattern for global access
   - Type-safe property access
   - No mutation after initialization (immutable)

---

## Public API

### Interfaces

```python
from pydantic_settings import BaseSettings
from pydantic import Field, validator, SecretStr
from typing import Optional, List
from pathlib import Path
import os

class ZapomniSettings(BaseSettings):
    """
    Centralized configuration for Zapomni system.

    Configuration is loaded from:
    1. System environment variables (highest priority)
    2. .env file in project root
    3. Hardcoded defaults (lowest priority)

    All settings are validated on load. Invalid configuration
    will raise ValidationError with clear error messages.

    Example:
        >>> from zapomni_config import settings
        >>> print(settings.falkordb_host)
        'localhost'
        >>> print(settings.chunk_size)
        512
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
        """Validate log level is one of allowed values."""
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(
                f"log_level must be one of {allowed}, got '{v}'"
            )
        return v_upper

    @validator("log_format")
    def validate_log_format(cls, v: str) -> str:
        """Validate log format is one of allowed values."""
        allowed = ["json", "text"]
        v_lower = v.lower()
        if v_lower not in allowed:
            raise ValueError(
                f"log_format must be one of {allowed}, got '{v}'"
            )
        return v_lower

    @validator("ollama_base_url")
    def validate_ollama_url(cls, v: str) -> str:
        """Validate Ollama base URL format."""
        if not v.startswith(("http://", "https://")):
            raise ValueError(
                f"ollama_base_url must start with http:// or https://, got '{v}'"
            )
        return v.rstrip("/")  # Remove trailing slash

    @validator("chunk_overlap")
    def validate_chunk_overlap(cls, v: int, values: dict) -> int:
        """Validate chunk overlap is reasonable (< chunk_size)."""
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
        """Validate vector dimensions match expected model output."""
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
        """Ensure directory exists, create if not."""
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
            redis://[password@]host:port
        """
        if self.falkordb_password:
            password = self.falkordb_password.get_secret_value()
            return f"redis://{password}@{self.falkordb_host}:{self.falkordb_port}"
        return f"redis://{self.falkordb_host}:{self.falkordb_port}"

    @property
    def redis_connection_string(self) -> str:
        """Get Redis connection string."""
        return f"redis://{self.redis_host}:{self.redis_port}"

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.log_level == "DEBUG"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.log_level in ["INFO", "WARNING", "ERROR"]

    # ========================================
    # PYDANTIC CONFIGURATION
    # ========================================

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        validate_assignment = True  # Validate on attribute assignment
        extra = "forbid"  # Forbid extra fields (strict mode)
        json_encoders = {
            SecretStr: lambda v: "***REDACTED***" if v else None
        }


# Singleton instance
settings = ZapomniSettings()
```

### Helper Functions

```python
from typing import Dict, Any
import json

def get_config_summary() -> Dict[str, Any]:
    """
    Get configuration summary for logging/debugging.

    Masks sensitive values and returns clean dict.

    Returns:
        dict: Configuration summary with masked secrets
    """
    config = {
        "database": {
            "falkordb_host": settings.falkordb_host,
            "falkordb_port": settings.falkordb_port,
            "graph_name": settings.graph_name,
            "pool_size": settings.falkordb_pool_size,
        },
        "ollama": {
            "base_url": settings.ollama_base_url,
            "embedding_model": settings.ollama_embedding_model,
            "llm_model": settings.ollama_llm_model,
        },
        "performance": {
            "max_chunk_size": settings.max_chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "vector_dimensions": settings.vector_dimensions,
            "hnsw_m": settings.hnsw_m,
        },
        "features": {
            "hybrid_search": settings.enable_hybrid_search,
            "knowledge_graph": settings.enable_knowledge_graph,
            "code_indexing": settings.enable_code_indexing,
            "semantic_cache": settings.enable_semantic_cache,
        },
        "logging": {
            "level": settings.log_level,
            "format": settings.log_format,
        }
    }
    return config


def validate_configuration() -> tuple[bool, list[str]]:
    """
    Validate configuration and return status.

    Returns:
        tuple: (is_valid, list_of_errors)
    """
    errors = []

    # Check critical services can be reached
    # (actual connection testing done elsewhere)

    # Check file paths are writable
    try:
        test_file = settings.data_dir / ".test"
        test_file.touch()
        test_file.unlink()
    except Exception as e:
        errors.append(f"data_dir not writable: {e}")

    # Check performance settings are reasonable
    if settings.chunk_overlap >= settings.max_chunk_size:
        errors.append(
            f"chunk_overlap ({settings.chunk_overlap}) >= "
            f"max_chunk_size ({settings.max_chunk_size})"
        )

    # Check HNSW parameters
    if settings.hnsw_ef_search > settings.hnsw_ef_construction:
        import warnings
        warnings.warn(
            f"hnsw_ef_search ({settings.hnsw_ef_search}) > "
            f"hnsw_ef_construction ({settings.hnsw_ef_construction}). "
            "This may reduce search quality."
        )

    is_valid = len(errors) == 0
    return is_valid, errors


def reload_configuration() -> None:
    """
    Reload configuration from environment.

    WARNING: This recreates the settings singleton.
    Should only be called when no operations are running.

    Raises:
        RuntimeError: If configuration is invalid after reload
    """
    global settings
    settings = ZapomniSettings()

    is_valid, errors = validate_configuration()
    if not is_valid:
        raise RuntimeError(
            f"Configuration reload failed: {'; '.join(errors)}"
        )
```

---

## Dependencies

### External Dependencies

- `pydantic>=2.0.0` (purpose: Settings base class, validation)
- `pydantic-settings>=2.0.0` (purpose: Environment variable loading)
- `python-dotenv>=1.0.0` (purpose: .env file parsing)

### Internal Dependencies

None - Configuration Management is a foundational module with no internal dependencies. All other modules depend on it.

### Dependency Rationale

**Pydantic** was chosen for configuration management because:
1. Type-safe: Automatic type conversion and validation
2. IDE support: Full autocomplete and type checking
3. Validation: Rich validation rules with custom validators
4. Documentation: Self-documenting via Field descriptions
5. Industry standard: Widely used in FastAPI, Django Ninja, etc.

**Alternative Considered**: `dynaconf`
- Pros: More features (multiple environments, secrets)
- Cons: More complex, overkill for our needs
- Decision: Pydantic is simpler and sufficient

---

## Data Flow

### Input

**Environment Variables:**
```bash
# .env file or system environment
FALKORDB_HOST=localhost
FALKORDB_PORT=6379
OLLAMA_BASE_URL=http://localhost:11434
LOG_LEVEL=INFO
MAX_CHUNK_SIZE=512
```

**Format Requirements:**
- Strings: Plain text, no quotes needed
- Integers: Numeric only (e.g., `512`, not `"512"`)
- Booleans: `true`/`false`, `1`/`0`, `yes`/`no` (case-insensitive)
- Paths: Absolute or relative file paths
- URLs: Must include protocol (`http://` or `https://`)

### Processing

1. **Load from sources (priority order):**
   - System environment variables (highest)
   - `.env` file in project root
   - Hardcoded defaults (lowest)

2. **Type conversion:**
   - Pydantic automatically converts strings to target types
   - Example: `"512"` → `512` (int), `"true"` → `True` (bool)

3. **Validation:**
   - Check types match annotations
   - Run `@validator` functions
   - Check field constraints (ge, le, etc.)
   - Validate cross-field dependencies

4. **Error reporting:**
   - If validation fails, raise `ValidationError` with detailed message
   - Include field name, invalid value, expected format

### Output

**Settings Singleton:**
```python
from zapomni_config import settings

# Type-safe access
host = settings.falkordb_host  # str
port = settings.falkordb_port  # int
chunk_size = settings.max_chunk_size  # int

# Computed properties
conn_str = settings.falkordb_connection_string  # str
is_dev = settings.is_development  # bool
```

**Guarantees:**
- All required fields are present
- All values pass validation
- Types are correct (enforced by Pydantic)
- Secrets are masked in string representations
- Immutable after initialization (unless reload_configuration called)

---

## Design Decisions

### Decision 1: Use Pydantic Settings

**Context:** Need type-safe, validated configuration system

**Options Considered:**
- **Option A:** Plain Python dict + manual validation
  - Pros: Simple, no dependencies
  - Cons: No type safety, manual validation, error-prone
- **Option B:** ConfigParser (stdlib)
  - Pros: Standard library, INI format
  - Cons: No type validation, outdated format
- **Option C:** Pydantic Settings (chosen)
  - Pros: Type-safe, automatic validation, IDE support
  - Cons: External dependency

**Chosen:** Option C (Pydantic Settings)

**Rationale:**
- Type safety prevents configuration errors at startup
- Validation catches misconfigurations immediately
- IDE autocomplete improves developer experience
- Industry standard (used in FastAPI, etc.)
- Worth the dependency for quality and safety

---

### Decision 2: Environment Variables Over Config Files

**Context:** How should users provide configuration?

**Options Considered:**
- **Option A:** YAML/JSON config files
  - Pros: Hierarchical, comments, familiar
  - Cons: Another file to manage, parsing overhead
- **Option B:** Environment variables (chosen)
  - Pros: Standard, Docker-friendly, 12-factor app compliant
  - Cons: Flat namespace, no hierarchy
- **Option C:** Python config module
  - Pros: Pythonic, programmable
  - Cons: Security risk (code execution), less portable

**Chosen:** Option B (Environment Variables)

**Rationale:**
- **12-Factor App Compliance:** Environment variables are standard for configuration
- **Docker/Kubernetes Native:** Easy integration with container orchestration
- **Security:** No risk of config file leaks (secrets in env only)
- **Simplicity:** Clear, flat namespace
- **Flexibility:** Can still use .env file for development

---

### Decision 3: Zero-Config Defaults

**Context:** Should system require configuration or work out-of-box?

**Options Considered:**
- **Option A:** Require explicit configuration (no defaults)
  - Pros: Forces users to think about settings
  - Cons: High friction, frustrating for new users
- **Option B:** Defaults for everything (chosen)
  - Pros: Works immediately, great UX
  - Cons: Users might not understand defaults
- **Option C:** Required fields for critical settings
  - Pros: Balance between safety and UX
  - Cons: Still friction for basic usage

**Chosen:** Option B (Zero-Config Defaults)

**Rationale:**
- **Product Vision:** "Works out-of-the-box" is core to Zapomni philosophy
- **Developer Experience:** New users can start immediately
- **Sensible Defaults:** localhost, standard ports, reasonable performance settings
- **Override When Needed:** Power users can customize everything
- **Quick Start:** Reduces setup time from 30 minutes to 5 minutes

**Defaults Provided:**
- FalkorDB: `localhost:6379` (standard Redis port)
- Ollama: `http://localhost:11434` (standard Ollama port)
- Chunk size: `512` tokens (research-backed optimal)
- Log level: `INFO` (not too noisy, not too quiet)

---

### Decision 4: Immutable Configuration

**Context:** Should configuration be mutable at runtime?

**Options Considered:**
- **Option A:** Fully mutable (allow runtime changes)
  - Pros: Flexible, can adjust without restart
  - Cons: Unpredictable behavior, race conditions
- **Option B:** Immutable (chosen for MVP)
  - Pros: Predictable, thread-safe, simple
  - Cons: Requires restart to change config
- **Option C:** Hot-reload for safe changes only
  - Pros: Best of both worlds
  - Cons: Complex to implement safely

**Chosen:** Option B (Immutable) for MVP, Option C for Phase 2+

**Rationale:**
- **Predictability:** Configuration doesn't change mid-operation
- **Thread Safety:** No locks needed for config access
- **Simplicity:** Easier to reason about
- **Future:** Can add hot-reload for log_level, feature flags (safe changes)
- **Dangerous Changes:** Never hot-reload database connection, ports (restart required)

---

### Decision 5: Fail-Fast on Invalid Configuration

**Context:** What to do if configuration is invalid?

**Options Considered:**
- **Option A:** Fail-fast (raise exception on startup)
  - Pros: Catches errors immediately, clear feedback
  - Cons: Application won't start
- **Option B:** Use defaults for invalid values
  - Pros: Application starts anyway
  - Cons: Silent failures, unpredictable behavior
- **Option C:** Warn and continue (chosen for some cases)
  - Pros: Allows operation with degraded config
  - Cons: May hide real issues

**Chosen:** Option A (Fail-Fast) for critical settings, Option C for warnings

**Rationale:**
- **Critical Settings:** Database host/port, required paths → fail-fast
- **Performance Tuning:** Chunk size, HNSW params → warn if suboptimal
- **Feature Flags:** Missing flags → default to False (safe)
- **Clear Errors:** ValidationError provides exact issue and how to fix

---

## Non-Functional Requirements

### Performance

**Configuration Load Time:**
- **Target:** < 100ms to load and validate all settings
- **Measurement:** Time from `import settings` to object ready
- **Optimization:** Lazy validators, no network calls during init

**Access Performance:**
- **Target:** O(1) attribute access (no lookups)
- **Implementation:** Direct attribute access via Pydantic
- **Guarantee:** No performance overhead vs plain Python object

### Scalability

**Configuration Size:**
- **Limit:** < 100 configuration parameters (currently ~40)
- **Rationale:** Beyond 100 parameters, consider grouping or splitting
- **Current:** Well within limits, room for growth

**Environment Variable Parsing:**
- **Limit:** No limit (Pydantic handles efficiently)
- **Note:** .env files with 1000+ lines still parse instantly

### Security

**Secret Handling:**
- **No Hardcoded Secrets:** All secrets via environment variables only
- **Masking:** SecretStr type masks values in logs and repr
- **Validation:** Warn if password appears in non-secret fields
- **Best Practice:** Use system secrets managers (AWS Secrets Manager, Vault) in production

**Example:**
```python
# BAD: Secret in code
falkordb_password = "my_secret_password"

# GOOD: Secret in environment
FALKORDB_PASSWORD=my_secret_password  # .env file (gitignored)

# BEST: Secret from secrets manager
FALKORDB_PASSWORD=$(aws secretsmanager get-secret-value ...)
```

**Configuration Validation:**
- **No Injection Attacks:** Validate URLs, paths, prevent command injection
- **Port Range:** 1-65535 only
- **Path Traversal:** Reject paths with `..` in data_dir

### Reliability

**Validation Guarantees:**
- **Type Safety:** All types validated on load
- **Range Checks:** Numeric values within acceptable ranges
- **Format Validation:** URLs, paths, enums validated
- **Cross-Field:** Dependent fields validated together

**Error Handling:**
```python
try:
    from zapomni_config import settings
except ValidationError as e:
    # Detailed error with field name and issue
    print(f"Configuration error: {e}")
    print("\nCheck your .env file or environment variables")
    sys.exit(1)
```

**Recovery:**
- If .env file malformed → use system env + defaults
- If validation fails → clear error message, exit (fail-fast)
- No partial configuration (all-or-nothing)

### Usability

**Clear Error Messages:**
```
ValidationError: 1 validation error for ZapomniSettings
falkordb_port
  ensure this value is greater than or equal to 1
  (type=value_error.number.not_ge; limit_value=1)

Received: FALKORDB_PORT=0
Expected: Integer between 1 and 65535
```

**Helpful Warnings:**
```
UserWarning: chunk_overlap (300) is > 50% of max_chunk_size (512).
Recommended: 10-20% (51-102 tokens)
```

**Self-Documenting:**
- Every field has `description` (visible in IDE)
- Pydantic generates JSON schema automatically
- Can export documentation: `settings.schema_json(indent=2)`

---

## Testing Strategy

### Unit Testing

**Test Coverage:**
- Configuration loading from different sources
- Type conversion and validation
- Validator functions (all @validator methods)
- Computed properties
- Error messages for invalid values

**Example Tests:**
```python
def test_default_configuration():
    """Test all defaults are valid."""
    settings = ZapomniSettings()
    assert settings.falkordb_host == "localhost"
    assert settings.falkordb_port == 6379
    assert settings.max_chunk_size == 512


def test_environment_override(monkeypatch):
    """Test environment variable overrides defaults."""
    monkeypatch.setenv("FALKORDB_HOST", "custom.host")
    monkeypatch.setenv("FALKORDB_PORT", "7000")

    settings = ZapomniSettings()
    assert settings.falkordb_host == "custom.host"
    assert settings.falkordb_port == 7000


def test_invalid_port_raises_error():
    """Test invalid port raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        ZapomniSettings(falkordb_port=70000)  # Out of range

    assert "falkordb_port" in str(exc_info.value)


def test_chunk_overlap_validator():
    """Test chunk_overlap must be < max_chunk_size."""
    with pytest.raises(ValidationError):
        ZapomniSettings(
            max_chunk_size=100,
            chunk_overlap=100  # Equal, should fail
        )


def test_computed_connection_string():
    """Test connection string property."""
    settings = ZapomniSettings(
        falkordb_host="db.example.com",
        falkordb_port=6380
    )

    expected = "redis://db.example.com:6380"
    assert settings.falkordb_connection_string == expected


def test_secret_masking():
    """Test SecretStr fields are masked."""
    settings = ZapomniSettings(
        falkordb_password="secret123"
    )

    # Secret should be masked in string representation
    settings_str = str(settings)
    assert "secret123" not in settings_str
    assert "***REDACTED***" in settings_str
```

### Integration Testing

**Test Scenarios:**
- Load configuration from .env file
- Validate with real FalkorDB connection parameters
- Test configuration in Docker environment
- Verify hot-reload functionality (Phase 2)

**Example:**
```python
def test_load_from_env_file(tmp_path):
    """Test loading from .env file."""
    env_file = tmp_path / ".env"
    env_file.write_text("""
FALKORDB_HOST=testhost
FALKORDB_PORT=6380
LOG_LEVEL=DEBUG
    """)

    settings = ZapomniSettings(_env_file=env_file)
    assert settings.falkordb_host == "testhost"
    assert settings.falkordb_port == 6380
    assert settings.log_level == "DEBUG"
```

---

## Future Considerations

### Phase 2: Hot-Reload Support

Add safe hot-reload for non-critical configuration:
- Log level changes without restart
- Feature flag toggles
- Performance tuning adjustments

**Implementation:**
```python
def reload_safe_settings():
    """Reload only safe-to-change settings."""
    new_settings = ZapomniSettings()

    # Only reload these fields
    settings.log_level = new_settings.log_level
    settings.enable_hybrid_search = new_settings.enable_hybrid_search
    # ...
```

### Phase 3: Configuration Profiles

Support multiple environments via profiles:
- `ZAPOMNI_ENV=development` → use dev defaults
- `ZAPOMNI_ENV=production` → use prod defaults
- `ZAPOMNI_ENV=testing` → use test defaults

### Phase 4: Secrets Management Integration

Integrate with secrets managers:
- AWS Secrets Manager
- HashiCorp Vault
- Azure Key Vault
- Kubernetes Secrets

**Example:**
```python
falkordb_password = get_secret(
    "aws-secretsmanager://zapomni/falkordb-password"
)
```

### Phase 5: Configuration Validation Service

Add MCP tool to validate configuration:
```python
@server.call_tool()
async def validate_config() -> Dict[str, Any]:
    """Validate current configuration."""
    is_valid, errors = validate_configuration()
    return {
        "valid": is_valid,
        "errors": errors,
        "summary": get_config_summary()
    }
```

---

## References

### Product Alignment
- **product.md:** Zero-config philosophy (Quick Start section)
- **product.md:** Local-first design (no cloud dependencies)
- **product.md:** Developer experience (simple setup)

### Technical Stack
- **tech.md:** FalkorDB configuration (host, port, connection)
- **tech.md:** Ollama configuration (base URL, models)
- **tech.md:** Performance tuning (chunk size, HNSW parameters)

### Structure Conventions
- **structure.md:** Configuration in `src/zapomni_mcp/config.py`
- **structure.md:** Environment variables loaded at startup
- **structure.md:** Pydantic Settings pattern

### External Documentation
- [Pydantic Settings Documentation](https://docs.pydantic.dev/latest/usage/settings/)
- [12-Factor App: Config](https://12factor.net/config)
- [FalkorDB Configuration](https://docs.falkordb.com/)
- [Ollama Configuration](https://github.com/ollama/ollama/blob/main/docs/faq.md)

---

## Document Status

**Version:** 1.0 (Draft)
**Created:** 2025-11-23
**Last Updated:** 2025-11-23
**Authors:** Goncharenko Anton aka alienxs2 + Claude Code
**Copyright:** Copyright (c) 2025 Goncharenko Anton aka alienxs2
**License:** MIT License
**Status:** Ready for Review

**Next Steps:**
1. Review by project maintainer
2. Approval via spec workflow
3. Implementation of ZapomniSettings class
4. Unit test coverage (target: 90%+)
5. Integration with other modules

---

**Document Metrics:**
- **Lines:** ~1200
- **Configuration Parameters:** 40+
- **Validators:** 7
- **Computed Properties:** 4
- **Test Scenarios:** 15+

**Estimated Implementation Time:** 4-6 hours
**Estimated Test Time:** 2-3 hours
**Total:** 1 working day
