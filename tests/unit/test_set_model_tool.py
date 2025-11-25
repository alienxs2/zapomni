"""
Tests for SetModelTool.
"""

import pytest

from zapomni_core.runtime_config import RuntimeConfig
from zapomni_mcp.tools.set_model import SetModelTool


class TestSetModelTool:
    """Test set_model MCP tool."""

    def setup_method(self):
        """Reset RuntimeConfig before each test."""
        RuntimeConfig.reset_instance()

    def teardown_method(self):
        """Reset RuntimeConfig after each test."""
        RuntimeConfig.reset_instance()

    def test_get_schema(self):
        """Test that get_schema returns valid MCP tool schema."""
        schema = SetModelTool.get_schema()

        assert schema.name == "set_model"
        assert "hot-reload" in schema.description.lower()
        assert "model_name" in schema.inputSchema["properties"]
        assert schema.inputSchema["required"] == ["model_name"]

    @pytest.mark.asyncio
    async def test_run_changes_model(self):
        """Test that run() changes the LLM model in RuntimeConfig."""
        config = RuntimeConfig.get_instance()
        initial_model = config.llm_model

        # Change model via tool
        result = await SetModelTool.run({"model_name": "llama3:latest"})

        # Check model was changed
        assert config.llm_model == "llama3:latest"
        assert config.llm_model != initial_model

        # Check result message
        assert len(result) == 1
        assert "llama3:latest" in result[0].text

    @pytest.mark.asyncio
    async def test_run_empty_model_name(self):
        """Test that run() handles empty model name gracefully."""
        config = RuntimeConfig.get_instance()
        initial_model = config.llm_model

        # Try to set empty model name
        result = await SetModelTool.run({"model_name": ""})

        # Model should not change
        assert config.llm_model == initial_model

        # Should return error message
        assert len(result) == 1
        assert "Error" in result[0].text
        assert "empty" in result[0].text

    @pytest.mark.asyncio
    async def test_run_whitespace_model_name(self):
        """Test that run() handles whitespace-only model name."""
        config = RuntimeConfig.get_instance()
        initial_model = config.llm_model

        # Try to set whitespace model name
        result = await SetModelTool.run({"model_name": "   "})

        # Model should not change
        assert config.llm_model == initial_model

        # Should return error message
        assert len(result) == 1
        assert "Error" in result[0].text

    @pytest.mark.asyncio
    async def test_run_returns_confirmation_message(self):
        """Test that run() returns detailed confirmation message."""
        result = await SetModelTool.run({"model_name": "mistral:latest"})

        assert len(result) == 1
        message = result[0].text

        # Should mention both old and new model
        assert "qwen2.5:latest" in message  # Old default model
        assert "mistral:latest" in message  # New model

        # Should mention what the model is used for
        assert "entity" in message.lower() or "refinement" in message.lower()

    @pytest.mark.asyncio
    async def test_run_with_various_model_formats(self):
        """Test that run() accepts various Ollama model name formats."""
        test_models = [
            "llama3:latest",
            "qwen2.5:7b",
            "mistral:instruct",
            "deepseek-coder:6.7b",
            "codellama:13b-python",
        ]

        config = RuntimeConfig.get_instance()

        for model in test_models:
            result = await SetModelTool.run({"model_name": model})

            # Should succeed and update config
            assert config.llm_model == model
            assert len(result) == 1
            assert model in result[0].text


class TestSetModelToolIntegration:
    """Integration tests for SetModelTool with RuntimeConfig."""

    def setup_method(self):
        """Reset RuntimeConfig before each test."""
        RuntimeConfig.reset_instance()

    def teardown_method(self):
        """Reset RuntimeConfig after each test."""
        RuntimeConfig.reset_instance()

    @pytest.mark.asyncio
    async def test_model_change_persists_across_config_instances(self):
        """Test that model change via tool persists for all RuntimeConfig instances."""
        config1 = RuntimeConfig.get_instance()

        # Change model via tool
        await SetModelTool.run({"model_name": "llama3:latest"})

        # Get new config instance (should be same singleton)
        config2 = RuntimeConfig.get_instance()

        assert config1 is config2
        assert config2.llm_model == "llama3:latest"

    @pytest.mark.asyncio
    async def test_multiple_consecutive_model_changes(self):
        """Test multiple consecutive model changes via tool."""
        config = RuntimeConfig.get_instance()
        models = ["model1:latest", "model2:latest", "model3:latest"]

        for model in models:
            result = await SetModelTool.run({"model_name": model})

            assert config.llm_model == model
            assert len(result) == 1

        # Final model should be the last one
        assert config.llm_model == "model3:latest"
