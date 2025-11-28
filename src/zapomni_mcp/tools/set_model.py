"""
set_model - Hot-reload Ollama LLM model without server restart.

Allows switching the Ollama LLM model at runtime by updating RuntimeConfig.
All existing and new OllamaLLMClient instances will immediately use the new model.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from typing import Any, Dict, List, Optional

import httpx
import structlog

from zapomni_core.config import settings
from zapomni_core.runtime_config import RuntimeConfig

logger = structlog.get_logger(__name__)


class SetModelTool:
    """
    MCP tool for hot-reloading Ollama LLM model configuration.

    Changes the Ollama model used for entity refinement and relationship extraction
    without restarting the MCP server. The new model takes effect immediately.

    Example:
        Input: {"model_name": "llama3:latest"}
        Result: LLM model switched from qwen2.5:latest to llama3:latest
    """

    name = "set_model"
    description = (
        "Hot-reload Ollama LLM model without restarting the MCP server. "
        "Changes take effect immediately for entity refinement and relationship extraction. "
        "Model must be available via 'ollama pull <model_name>' before use."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "model_name": {
                "type": "string",
                "description": (
                    "Ollama model name to use for LLM operations. "
                    "Examples: 'qwen2.5:latest', 'llama3:latest', 'mistral:latest'. "
                    "Model must be pulled via 'ollama pull <model_name>' before use."
                ),
            },
        },
        "required": ["model_name"],
    }

    def __init__(self, ollama_base_url: Optional[str] = None) -> None:
        """
        Initialize SetModelTool.

        Args:
            ollama_base_url: Ollama API URL. If None, uses settings.ollama_base_url.
        """
        self._ollama_base_url = ollama_base_url or settings.ollama_base_url
        logger.info(
            "set_model_tool_initialized",
            tool=self.name,
            ollama_url=self._ollama_base_url,
        )

    async def _get_available_models(self) -> List[str]:
        """
        Get list of available models from Ollama API.

        Calls Ollama's /api/tags endpoint to retrieve all pulled models.

        Returns:
            List of model names available in Ollama.
            Empty list if Ollama is unreachable or returns error.

        Example:
            >>> models = await tool._get_available_models()
            >>> print(models)
            ['qwen2.5:latest', 'llama3:latest', 'nomic-embed-text:latest']
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self._ollama_base_url}/api/tags")

                if response.status_code == 200:
                    data = response.json()
                    models = data.get("models", [])
                    # Extract model names from response
                    return [m.get("name", "") for m in models if m.get("name")]

                logger.warning(
                    "ollama_tags_request_failed",
                    status_code=response.status_code,
                )
                return []

        except httpx.ConnectError:
            logger.warning("ollama_not_reachable", url=self._ollama_base_url)
            return []
        except httpx.TimeoutException:
            logger.warning("ollama_tags_timeout", url=self._ollama_base_url)
            return []
        except Exception as e:
            logger.warning("ollama_tags_error", error=str(e))
            return []

    async def _validate_model_exists(self, model_name: str) -> tuple[bool, List[str]]:
        """
        Check if model exists in Ollama.

        Validates that the requested model is available in Ollama by:
        1. Getting list of available models via /api/tags
        2. Checking for exact match or partial match (e.g., "llama3" matches "llama3:latest")

        Args:
            model_name: Model name to validate (e.g., "llama3:latest", "qwen2.5")

        Returns:
            Tuple of (is_valid, available_models):
            - is_valid: True if model exists or Ollama is unreachable (fail-open)
            - available_models: List of available model names (for error messages)

        Note:
            If Ollama is unreachable, returns (True, []) to allow fail-open behavior.
            This prevents blocking users when Ollama is temporarily unavailable.
        """
        available_models = await self._get_available_models()

        # If can't reach Ollama, allow the change (fail-open)
        if not available_models:
            logger.info(
                "ollama_unreachable_allowing_model_change",
                model=model_name,
            )
            return True, []

        # Check for exact match
        if model_name in available_models:
            return True, available_models

        # Check for partial match (e.g., "llama3" matches "llama3:latest")
        # Also handles case where user specifies "llama3:latest" but Ollama has "llama3:8b"
        model_base = model_name.split(":")[0]
        for available in available_models:
            available_base = available.split(":")[0]
            if model_base == available_base:
                return True, available_models

        return False, available_models

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute set_model tool to change Ollama LLM model at runtime.

        Validates that the model exists in Ollama before changing.
        If Ollama is unreachable, allows the change with a warning (fail-open).

        Args:
            arguments: Tool arguments with 'model_name' field

        Returns:
            Result dict with confirmation message or error

        Example:
            >>> result = await tool.execute({"model_name": "llama3:latest"})
            >>> print(result["content"][0]["text"])
            "LLM model changed from qwen2.5:latest to llama3:latest"
        """
        model_name = arguments.get("model_name", "").strip()

        if not model_name:
            logger.warning("set_model_empty_name")
            return {
                "content": [
                    {"type": "text", "text": "Error: model_name cannot be empty"}
                ],
                "isError": True,
            }

        # Validate model exists in Ollama
        is_valid, available_models = await self._validate_model_exists(model_name)

        if not is_valid:
            logger.warning(
                "set_model_not_found",
                model=model_name,
                available=available_models[:10],  # Log first 10 models
            )

            # Format available models for error message
            models_list = ", ".join(available_models[:10])
            if len(available_models) > 10:
                models_list += f" ... and {len(available_models) - 10} more"

            error_text = (
                f"Error: Model '{model_name}' not found in Ollama.\n\n"
                f"Available models:\n{models_list}\n\n"
                f"To add this model, run:\n"
                f"  ollama pull {model_name}"
            )

            return {
                "content": [{"type": "text", "text": error_text}],
                "isError": True,
            }

        # Get RuntimeConfig singleton
        config = RuntimeConfig.get_instance()

        # Get current model before change
        old_model = config.llm_model

        # Update model (hot-reload)
        config.set_llm_model(model_name)

        logger.info(
            "llm_model_changed_via_tool",
            old_model=old_model,
            new_model=model_name,
        )

        # Build result message
        if available_models:
            result_text = (
                f"LLM model changed from {old_model} to {model_name}\n\n"
                f"The new model will be used for:\n"
                f"- Entity refinement (enhancing SpaCy NER results)\n"
                f"- Relationship extraction (detecting entity connections)"
            )
        else:
            # Ollama was unreachable - warn the user
            result_text = (
                f"LLM model changed from {old_model} to {model_name}\n\n"
                f"WARNING: Could not verify model existence (Ollama unreachable).\n"
                f"If the model doesn't exist, LLM operations will fail.\n\n"
                f"The new model will be used for:\n"
                f"- Entity refinement (enhancing SpaCy NER results)\n"
                f"- Relationship extraction (detecting entity connections)\n\n"
                f"Ensure model is available via 'ollama pull {model_name}' before use."
            )

        return {
            "content": [{"type": "text", "text": result_text}],
            "isError": False,
        }
