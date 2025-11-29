"""
Configuration Management for Zapomni.

Provides centralized, type-safe configuration loading using Pydantic Settings.
Supports environment variables, .env files, and sensible defaults for zero-config operation.

Copyright (c) 2025 Goncharenko Anton aka alienxs2
License: MIT
"""

import warnings
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings


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

    Example:
        ```python
        from zapomni_core.config import settings

        # Access configuration
        print(settings.falkordb_host)  # 'localhost'
        print(settings.max_chunk_size)  # 512

        # Use computed properties
        conn_str = settings.falkordb_connection_string
        # 'redis://localhost:6381'

        # Check environment
        if settings.is_development:
            print("Running in development mode")
        ```
    """

    # ========================================
    # FALKORDB CONFIGURATION
    # ========================================

    falkordb_host: str = Field(
        default="localhost", description="FalkorDB server hostname or IP address"
    )

    falkordb_port: int = Field(
        default=6381,
        ge=1,
        le=65535,
        description="FalkorDB server port (6381 to avoid conflict with Redis)",
    )

    falkordb_password: Optional[SecretStr] = Field(
        default=None, description="FalkorDB authentication password (if required)"
    )

    graph_name: str = Field(
        default="zapomni_memory", description="Name of the FalkorDB graph database"
    )

    falkordb_connection_timeout: int = Field(
        default=30, ge=1, le=300, description="FalkorDB connection timeout in seconds"
    )

    falkordb_pool_size: int = Field(
        default=20,
        ge=1,
        le=200,
        description="FalkorDB connection pool size (increased for SSE concurrency). "
        "Deprecated: Use falkordb_pool_max_size instead.",
    )

    # ========================================
    # DATABASE POOL CONFIGURATION
    # ========================================

    falkordb_pool_min_size: int = Field(
        default=5, ge=1, le=50, description="Minimum database connections to maintain in pool"
    )

    falkordb_pool_max_size: int = Field(
        default=20, ge=1, le=200, description="Maximum database connections allowed in pool"
    )

    falkordb_pool_timeout: float = Field(
        default=10.0,
        ge=1.0,
        le=60.0,
        description="Seconds to wait for available connection from pool",
    )

    falkordb_socket_timeout: float = Field(
        default=30.0, ge=5.0, le=120.0, description="Socket timeout for query execution in seconds"
    )

    falkordb_health_check_interval: int = Field(
        default=30,
        ge=10,
        le=300,
        description="Health check interval for pool connections in seconds",
    )

    # ========================================
    # RETRY CONFIGURATION
    # ========================================

    falkordb_max_retries: int = Field(
        default=3, ge=0, le=10, description="Maximum retry attempts for transient database errors"
    )

    falkordb_retry_initial_delay: float = Field(
        default=0.1, ge=0.01, le=1.0, description="Initial delay for retry backoff in seconds"
    )

    falkordb_retry_max_delay: float = Field(
        default=2.0, ge=0.1, le=30.0, description="Maximum delay for retry backoff in seconds"
    )

    # ========================================
    # OLLAMA CONFIGURATION
    # ========================================

    ollama_base_url: str = Field(
        default="http://localhost:11434", description="Ollama server base URL (including protocol)"
    )

    ollama_embedding_model: str = Field(
        default="nomic-embed-text", description="Ollama model for embedding generation (768 dim)"
    )

    ollama_llm_model: str = Field(
        default="llama3.1:8b", description="Ollama model for LLM inference (entity extraction)"
    )

    ollama_embedding_timeout: int = Field(
        default=60, ge=5, le=300, description="Timeout for embedding requests in seconds"
    )

    ollama_llm_timeout: int = Field(
        default=120, ge=10, le=600, description="Timeout for LLM inference requests in seconds"
    )

    # ========================================
    # REDIS CACHE CONFIGURATION
    # ========================================

    redis_enabled: bool = Field(
        default=False, description="Enable Redis semantic cache (Phase 2 feature)"
    )

    redis_host: str = Field(default="localhost", description="Redis server hostname")

    redis_port: int = Field(
        default=6380,
        ge=1,
        le=65535,
        description="Redis server port (6380 to avoid conflict with FalkorDB)",
    )

    redis_ttl_seconds: int = Field(
        default=86400,  # 24 hours
        ge=60,
        le=604800,  # 7 days max
        description="Redis cache entry TTL in seconds",
    )

    redis_max_memory_mb: int = Field(
        default=1024, ge=100, le=10240, description="Redis maximum memory usage in MB"
    )

    # ========================================
    # PERFORMANCE TUNING
    # ========================================

    max_chunk_size: int = Field(
        default=512, ge=100, le=2000, description="Maximum chunk size in tokens"
    )

    chunk_overlap: int = Field(
        default=50,
        ge=0,
        le=500,
        description="Chunk overlap in tokens (10-20% of chunk_size recommended)",
    )

    vector_dimensions: int = Field(
        default=768, description="Embedding vector dimensions (must match model)"
    )

    hnsw_m: int = Field(
        default=16, ge=4, le=64, description="HNSW index M parameter (connections per layer)"
    )

    hnsw_ef_construction: int = Field(
        default=200, ge=50, le=1000, description="HNSW build-time accuracy parameter"
    )

    hnsw_ef_search: int = Field(
        default=100, ge=10, le=500, description="HNSW query-time accuracy parameter"
    )

    max_concurrent_tasks: int = Field(
        default=4, ge=1, le=32, description="Maximum concurrent background tasks"
    )

    search_limit_default: int = Field(
        default=10, ge=1, le=100, description="Default number of search results"
    )

    min_similarity_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Minimum cosine similarity for search results"
    )

    # ========================================
    # LOGGING CONFIGURATION
    # ========================================

    log_level: str = Field(
        default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )

    log_format: str = Field(default="json", description="Log format (json, text)")

    log_file: Optional[Path] = Field(default=None, description="Log file path (None = stderr only)")

    # ========================================
    # FEATURE FLAGS
    # ========================================

    enable_hybrid_search: bool = Field(
        default=True, description="Enable BM25 + vector hybrid search"
    )

    enable_knowledge_graph: bool = Field(
        default=True, description="Enable entity extraction and knowledge graph"
    )

    enable_code_indexing: bool = Field(default=True, description="Enable AST-based code indexing")

    enable_semantic_cache: bool = Field(
        default=True, description="Enable semantic embedding cache (Redis + in-memory fallback)"
    )

    # ========================================
    # SSE TRANSPORT CONFIGURATION
    # ========================================

    sse_host: str = Field(
        default="127.0.0.1", description="SSE server bind address (127.0.0.1 for local only)"
    )

    sse_port: int = Field(default=8000, ge=1, le=65535, description="SSE server port")

    sse_cors_origins: str = Field(
        default="*", description="Comma-separated CORS origins (* for all)"
    )

    sse_heartbeat_interval: int = Field(
        default=30, ge=5, le=300, description="SSE heartbeat interval in seconds"
    )

    sse_max_connection_lifetime: int = Field(
        default=3600, ge=60, le=86400, description="Maximum SSE connection lifetime in seconds"
    )

    # ========================================
    # EXECUTOR CONFIGURATION
    # ========================================

    entity_extractor_workers: int = Field(
        default=5, ge=1, le=20, description="Thread pool workers for entity extraction"
    )

    # ========================================
    # SYSTEM CONFIGURATION
    # ========================================

    data_dir: Path = Field(default=Path("./data"), description="Data storage directory")

    temp_dir: Path = Field(default=Path("/tmp/zapomni"), description="Temporary files directory")

    max_text_length: int = Field(
        default=10_000_000,  # 10MB
        ge=1000,
        le=100_000_000,
        description="Maximum allowed text length in characters",
    )

    # ========================================
    # VALIDATORS
    # ========================================

    @field_validator("log_level")
    @classmethod
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
            raise ValueError(f"log_level must be one of {allowed}, got '{v}'")
        return v_upper

    @field_validator("log_format")
    @classmethod
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
            raise ValueError(f"log_format must be one of {allowed}, got '{v}'")
        return v_lower

    @field_validator("ollama_base_url")
    @classmethod
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
            raise ValueError(f"ollama_base_url must start with http:// or https://, got '{v}'")
        return v.rstrip("/")  # Remove trailing slash

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, v: int, info: Any) -> int:
        """
        Validate chunk overlap is reasonable relative to chunk_size.

        Args:
            v: Chunk overlap value
            info: Validation context with previously validated fields

        Returns:
            Validated chunk overlap value

        Raises:
            ValueError: If overlap >= chunk_size

        Warnings:
            UserWarning: If overlap > 50% of chunk_size
        """
        if info.data and "max_chunk_size" in info.data:
            max_size = info.data["max_chunk_size"]
            if v >= max_size:
                raise ValueError(
                    f"chunk_overlap ({v}) must be less than max_chunk_size ({max_size})"
                )
            if v > max_size * 0.5:
                warnings.warn(
                    f"chunk_overlap ({v}) is > 50% of max_chunk_size ({max_size}). "
                    "Recommended: 10-20%"
                )
        return v

    @field_validator("vector_dimensions")
    @classmethod
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
            warnings.warn(f"vector_dimensions ({v}) is non-standard. " f"Common values: {allowed}")
        return v

    @field_validator("data_dir", "temp_dir")
    @classmethod
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

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "validate_assignment": True,  # Validate on attribute assignment
        "extra": "forbid",  # Forbid extra fields (strict mode)
    }


# ============================================================
# HELPER FUNCTIONS
# ============================================================


def get_config_summary(settings: ZapomniSettings) -> Dict[str, Any]:
    """
    Get configuration summary for logging/debugging with sensitive values masked.

    Args:
        settings: ZapomniSettings instance

    Returns:
        Configuration summary grouped by category
    """
    return {
        "database": {
            "falkordb_host": settings.falkordb_host,
            "falkordb_port": settings.falkordb_port,
            "graph_name": settings.graph_name,
            "pool_size": settings.falkordb_pool_size,
            "pool_min_size": settings.falkordb_pool_min_size,
            "pool_max_size": settings.falkordb_pool_max_size,
            "pool_timeout": settings.falkordb_pool_timeout,
            "socket_timeout": settings.falkordb_socket_timeout,
            "health_check_interval": settings.falkordb_health_check_interval,
            "max_retries": settings.falkordb_max_retries,
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
        },
    }


def validate_configuration(settings: ZapomniSettings) -> tuple[bool, list[str]]:
    """
    Perform additional runtime validation beyond Pydantic checks.

    Args:
        settings: ZapomniSettings instance to validate

    Returns:
        Tuple of (is_valid, errors)
        - is_valid: True if configuration is valid
        - errors: List of error messages (empty if valid)
    """
    errors = []

    # Check data directory is writable
    try:
        test_file = settings.data_dir / ".write_test"
        test_file.write_text("test")
        test_file.unlink()
    except (OSError, PermissionError) as e:
        errors.append(f"Data directory not writable: {settings.data_dir} ({e})")

    # Check performance settings are reasonable
    if settings.hnsw_ef_search > settings.hnsw_ef_construction:
        errors.append(
            f"hnsw_ef_search ({settings.hnsw_ef_search}) should not exceed "
            f"hnsw_ef_construction ({settings.hnsw_ef_construction})"
        )

    return (len(errors) == 0, errors)


# Singleton instance - instantiated once at module import
settings = ZapomniSettings()
