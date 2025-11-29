"""
Connection pool and retry configuration for FalkorDB client.

Provides configuration classes for connection pooling and retry logic
with exponential backoff for transient errors.

Copyright (c) 2025 Goncharenko Anton aka alienxs2
License: MIT
"""

import os
from dataclasses import dataclass

from zapomni_db.exceptions import ValidationError


@dataclass
class PoolConfig:
    """
    Connection pool configuration for FalkorDB.

    Attributes:
        min_size: Minimum connections to maintain in pool (default: 5)
        max_size: Maximum connections allowed in pool (default: 20)
        timeout: Seconds to wait for available connection from pool (default: 10.0)
        socket_timeout: Socket timeout for query execution in seconds (default: 30.0)
        socket_connect_timeout: Timeout for initial connection in seconds (default: 5.0)
        health_check_interval: Seconds between connection health checks (default: 30)
    """

    min_size: int = 5
    max_size: int = 20
    timeout: float = 10.0
    socket_timeout: float = 30.0
    socket_connect_timeout: float = 5.0
    health_check_interval: int = 30

    def __post_init__(self) -> None:
        """Validate pool configuration after initialization."""
        if self.min_size < 1:
            raise ValidationError(f"min_size must be >= 1, got {self.min_size}")

        if self.max_size < self.min_size:
            raise ValidationError(
                f"max_size ({self.max_size}) must be >= min_size ({self.min_size})"
            )

        if self.max_size > 200:
            raise ValidationError(f"max_size must be <= 200, got {self.max_size}")

        if self.timeout <= 0:
            raise ValidationError(f"timeout must be > 0, got {self.timeout}")

        if self.socket_timeout <= 0:
            raise ValidationError(f"socket_timeout must be > 0, got {self.socket_timeout}")

        if self.socket_connect_timeout <= 0:
            raise ValidationError(
                f"socket_connect_timeout must be > 0, got {self.socket_connect_timeout}"
            )

        if self.health_check_interval < 10:
            raise ValidationError(
                f"health_check_interval must be >= 10, got {self.health_check_interval}"
            )

    @classmethod
    def from_env(cls) -> "PoolConfig":
        """
        Load pool configuration from environment variables.

        Environment variables (with FALKORDB_ prefix):
            FALKORDB_POOL_MIN_SIZE: Minimum pool size (default: 5)
            FALKORDB_POOL_MAX_SIZE: Maximum pool size (default: 20)
            FALKORDB_POOL_TIMEOUT: Pool acquisition timeout (default: 10.0)
            FALKORDB_SOCKET_TIMEOUT: Query execution timeout (default: 30.0)
            FALKORDB_HEALTH_CHECK_INTERVAL: Health check interval (default: 30)

        Returns:
            PoolConfig instance with values from environment
        """
        return cls(
            min_size=int(os.getenv("FALKORDB_POOL_MIN_SIZE", "5")),
            max_size=int(os.getenv("FALKORDB_POOL_MAX_SIZE", "20")),
            timeout=float(os.getenv("FALKORDB_POOL_TIMEOUT", "10.0")),
            socket_timeout=float(os.getenv("FALKORDB_SOCKET_TIMEOUT", "30.0")),
            health_check_interval=int(os.getenv("FALKORDB_HEALTH_CHECK_INTERVAL", "30")),
        )


@dataclass
class RetryConfig:
    """
    Retry configuration for transient database errors.

    Implements exponential backoff strategy:
    - Attempt 1: Execute immediately
    - Attempt 2: Wait initial_delay, then execute
    - Attempt 3: Wait initial_delay * exponential_base, then execute
    - etc.

    Attributes:
        max_retries: Maximum retry attempts (default: 3)
        initial_delay: First retry delay in seconds (default: 0.1)
        max_delay: Maximum delay cap in seconds (default: 2.0)
        exponential_base: Backoff multiplier (default: 2.0)
    """

    max_retries: int = 3
    initial_delay: float = 0.1
    max_delay: float = 2.0
    exponential_base: float = 2.0

    def __post_init__(self) -> None:
        """Validate retry configuration after initialization."""
        if self.max_retries < 0:
            raise ValidationError(f"max_retries must be >= 0, got {self.max_retries}")

        if self.max_retries > 10:
            raise ValidationError(f"max_retries must be <= 10, got {self.max_retries}")

        if self.initial_delay <= 0:
            raise ValidationError(f"initial_delay must be > 0, got {self.initial_delay}")

        if self.max_delay < self.initial_delay:
            raise ValidationError(
                f"max_delay ({self.max_delay}) must be >= initial_delay ({self.initial_delay})"
            )

        if self.exponential_base < 1.0:
            raise ValidationError(f"exponential_base must be >= 1.0, got {self.exponential_base}")

    @classmethod
    def from_env(cls) -> "RetryConfig":
        """
        Load retry configuration from environment variables.

        Environment variables (with FALKORDB_ prefix):
            FALKORDB_MAX_RETRIES: Maximum retry attempts (default: 3)
            FALKORDB_RETRY_INITIAL_DELAY: Initial delay in seconds (default: 0.1)
            FALKORDB_RETRY_MAX_DELAY: Maximum delay in seconds (default: 2.0)

        Returns:
            RetryConfig instance with values from environment
        """
        return cls(
            max_retries=int(os.getenv("FALKORDB_MAX_RETRIES", "3")),
            initial_delay=float(os.getenv("FALKORDB_RETRY_INITIAL_DELAY", "0.1")),
            max_delay=float(os.getenv("FALKORDB_RETRY_MAX_DELAY", "2.0")),
        )


__all__ = ["PoolConfig", "RetryConfig"]
