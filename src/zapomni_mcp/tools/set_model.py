"""
set_model - Hot-reload Ollama LLM model without server restart.

Allows switching the Ollama LLM model at runtime by updating RuntimeConfig.
All existing and new OllamaLLMClient instances will immediately use the new model.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from typing import Any, Dict

import structlog

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

    def __init__(self) -> None:
        """Initialize SetModelTool."""
        logger.info("set_model_tool_initialized", tool=self.name)

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute set_model tool to change Ollama LLM model at runtime.

        Args:
            arguments: Tool arguments with 'model_name' field

        Returns:
            Result dict with confirmation message

        Example:
            >>> result = await tool.execute({"model_name": "llama3:latest"})
            >>> print(result["result"])
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

        result_text = (
            f"LLM model changed from {old_model} to {model_name}\n\n"
            f"The new model will be used for:\n"
            f"- Entity refinement (enhancing SpaCy NER results)\n"
            f"- Relationship extraction (detecting entity connections)\n\n"
            f"Note: Ensure model is available via 'ollama pull {model_name}' before use."
        )

        return {
            "content": [{"type": "text", "text": result_text}],
            "isError": False,
        }
