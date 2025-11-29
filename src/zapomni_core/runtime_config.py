"""
RuntimeConfig - Thread-safe singleton for runtime configuration hot-reload.

Provides centralized configuration storage that can be updated without server restart.
Used for runtime-configurable parameters like Ollama LLM model selection.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

import threading
from typing import Any, Dict, Optional

import structlog

logger = structlog.get_logger(__name__)


class RuntimeConfig:
    """
    Thread-safe singleton for runtime configuration.

    Allows hot-reload of configuration parameters without restarting the MCP server.
    All configuration changes are immediately visible to all components.

    Thread-Safety:
        Uses threading.Lock to ensure atomic updates and reads.
        Safe for concurrent access from multiple MCP sessions.

    Example:
        ```python
        # Get singleton instance
        config = RuntimeConfig.get_instance()

        # Read current model
        model = config.llm_model

        # Update model (hot-reload)
        config.set_llm_model("llama3:latest")
        ```
    """

    _instance: Optional["RuntimeConfig"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        """
        Initialize RuntimeConfig with default values.

        DO NOT call directly - use get_instance() instead.
        """
        if RuntimeConfig._instance is not None:
            raise RuntimeError("RuntimeConfig is a singleton. Use RuntimeConfig.get_instance()")

        # Runtime-configurable parameters
        self._llm_model = "qwen2.5:latest"  # Default Ollama LLM model

        # Lock for thread-safe updates
        self._config_lock = threading.Lock()

        logger.info("runtime_config_initialized", llm_model=self._llm_model)

    @classmethod
    def get_instance(cls) -> "RuntimeConfig":
        """
        Get singleton instance of RuntimeConfig.

        Thread-safe singleton creation using double-checked locking pattern.

        Returns:
            RuntimeConfig singleton instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = RuntimeConfig()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset singleton instance (for testing only).

        WARNING: Only use in test teardown. Never call in production code.
        """
        with cls._lock:
            cls._instance = None

    @property
    def llm_model(self) -> str:
        """
        Get current Ollama LLM model name.

        Thread-safe read operation.

        Returns:
            Model name (e.g., "qwen2.5:latest", "llama3:latest")
        """
        with self._config_lock:
            return self._llm_model

    def set_llm_model(self, model_name: str) -> None:
        """
        Set Ollama LLM model name (hot-reload).

        Changes take effect immediately for all new LLM requests.
        Existing OllamaLLMClient instances will use the new model.

        Thread-safe write operation.

        Args:
            model_name: Ollama model name (e.g., "llama3:latest")

        Example:
            >>> config = RuntimeConfig.get_instance()
            >>> config.set_llm_model("llama3:latest")
        """
        with self._config_lock:
            old_model = self._llm_model
            self._llm_model = model_name
            logger.info(
                "llm_model_changed",
                old_model=old_model,
                new_model=model_name,
            )

    def get_all_config(self) -> Dict[str, Any]:
        """
        Get all runtime configuration as dictionary.

        Thread-safe read operation.

        Returns:
            Dictionary with all configuration values
        """
        with self._config_lock:
            return {
                "llm_model": self._llm_model,
            }
