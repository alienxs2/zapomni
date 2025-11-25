"""
set_model - Hot-reload Ollama LLM model without server restart.

Allows switching the Ollama LLM model at runtime by updating RuntimeConfig.
All existing and new OllamaLLMClient instances will immediately use the new model.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

import mcp.types as types
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

    NAME = "set_model"
    DESCRIPTION = (
        "Hot-reload Ollama LLM model without restarting the MCP server. "
        "Changes take effect immediately for entity refinement and relationship extraction. "
        "Model must be available via 'ollama pull <model_name>' before use."
    )

    @staticmethod
    def get_schema() -> types.Tool:
        """
        Get MCP tool schema for set_model.

        Returns:
            MCP Tool schema with input parameters
        """
        return types.Tool(
            name=SetModelTool.NAME,
            description=SetModelTool.DESCRIPTION,
            inputSchema={
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
            },
        )

    @staticmethod
    async def run(arguments: dict) -> list[types.TextContent]:
        """
        Execute set_model tool to change Ollama LLM model at runtime.

        Args:
            arguments: Tool arguments with 'model_name' field

        Returns:
            List with single TextContent containing confirmation message

        Example:
            >>> result = await SetModelTool.run({"model_name": "llama3:latest"})
            >>> print(result[0].text)
            "LLM model changed from qwen2.5:latest to llama3:latest"
        """
        model_name = arguments.get("model_name", "").strip()

        if not model_name:
            logger.warning("set_model_empty_name")
            return [
                types.TextContent(
                    type="text",
                    text="Error: model_name cannot be empty",
                )
            ]

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

        return [
            types.TextContent(
                type="text",
                text=f"LLM model changed from {old_model} to {model_name}\n\n"
                f"The new model will be used for:\n"
                f"- Entity refinement (enhancing SpaCy NER results)\n"
                f"- Relationship extraction (detecting entity connections)\n\n"
                f"Note: Ensure model is available via 'ollama pull {model_name}' before use.",
            )
        ]
