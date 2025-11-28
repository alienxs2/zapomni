"""
Tests for SetModelTool.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from unittest.mock import AsyncMock, patch

import httpx
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

    def test_tool_schema(self):
        """Test that SetModelTool has valid MCP tool schema."""
        tool = SetModelTool()

        assert tool.name == "set_model"
        assert "hot-reload" in tool.description.lower()
        assert "model_name" in tool.input_schema["properties"]
        assert tool.input_schema["required"] == ["model_name"]

    def test_init_with_custom_ollama_url(self):
        """Test that SetModelTool accepts custom Ollama URL."""
        custom_url = "http://192.168.1.100:11434"
        tool = SetModelTool(ollama_base_url=custom_url)

        assert tool._ollama_base_url == custom_url

    @pytest.mark.asyncio
    async def test_execute_empty_model_name(self):
        """Test that execute() handles empty model name gracefully."""
        tool = SetModelTool()
        config = RuntimeConfig.get_instance()
        initial_model = config.llm_model

        # Try to set empty model name
        result = await tool.execute({"model_name": ""})

        # Model should not change
        assert config.llm_model == initial_model

        # Should return error (MCP format)
        assert result["isError"] is True
        assert len(result["content"]) == 1
        assert "empty" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_whitespace_model_name(self):
        """Test that execute() handles whitespace-only model name."""
        tool = SetModelTool()
        config = RuntimeConfig.get_instance()
        initial_model = config.llm_model

        # Try to set whitespace model name
        result = await tool.execute({"model_name": "   "})

        # Model should not change
        assert config.llm_model == initial_model

        # Should return error (MCP format)
        assert result["isError"] is True
        assert len(result["content"]) == 1


class TestSetModelToolValidation:
    """Test model validation in SetModelTool."""

    def setup_method(self):
        """Reset RuntimeConfig before each test."""
        RuntimeConfig.reset_instance()

    def teardown_method(self):
        """Reset RuntimeConfig after each test."""
        RuntimeConfig.reset_instance()

    @pytest.mark.asyncio
    async def test_get_available_models_success(self):
        """Test _get_available_models returns model list from Ollama."""
        tool = SetModelTool()

        mock_response = httpx.Response(
            200,
            json={
                "models": [
                    {"name": "llama3:latest", "size": 4000000000},
                    {"name": "qwen2.5:latest", "size": 3000000000},
                    {"name": "nomic-embed-text:latest", "size": 500000000},
                ]
            },
        )

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            models = await tool._get_available_models()

        assert models == ["llama3:latest", "qwen2.5:latest", "nomic-embed-text:latest"]

    @pytest.mark.asyncio
    async def test_get_available_models_ollama_unreachable(self):
        """Test _get_available_models returns empty list when Ollama is unreachable."""
        tool = SetModelTool()

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.ConnectError("Connection refused")
            models = await tool._get_available_models()

        assert models == []

    @pytest.mark.asyncio
    async def test_get_available_models_timeout(self):
        """Test _get_available_models handles timeout gracefully."""
        tool = SetModelTool()

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.TimeoutException("Request timed out")
            models = await tool._get_available_models()

        assert models == []

    @pytest.mark.asyncio
    async def test_get_available_models_error_response(self):
        """Test _get_available_models handles error response."""
        tool = SetModelTool()

        mock_response = httpx.Response(500, text="Internal Server Error")

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            models = await tool._get_available_models()

        assert models == []

    @pytest.mark.asyncio
    async def test_validate_model_exists_exact_match(self):
        """Test _validate_model_exists with exact model name match."""
        tool = SetModelTool()

        mock_response = httpx.Response(
            200,
            json={
                "models": [
                    {"name": "llama3:latest"},
                    {"name": "qwen2.5:latest"},
                ]
            },
        )

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            is_valid, available = await tool._validate_model_exists("llama3:latest")

        assert is_valid is True
        assert "llama3:latest" in available

    @pytest.mark.asyncio
    async def test_validate_model_exists_partial_match(self):
        """Test _validate_model_exists with partial model name match (base name)."""
        tool = SetModelTool()

        mock_response = httpx.Response(
            200,
            json={
                "models": [
                    {"name": "llama3:8b"},
                    {"name": "qwen2.5:latest"},
                ]
            },
        )

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            # User requests "llama3:latest" but only "llama3:8b" is available
            is_valid, available = await tool._validate_model_exists("llama3:latest")

        # Should match because base name "llama3" is the same
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_validate_model_not_found(self):
        """Test _validate_model_exists when model doesn't exist."""
        tool = SetModelTool()

        mock_response = httpx.Response(
            200,
            json={
                "models": [
                    {"name": "llama3:latest"},
                    {"name": "qwen2.5:latest"},
                ]
            },
        )

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            is_valid, available = await tool._validate_model_exists("nonexistent-model:latest")

        assert is_valid is False
        assert "llama3:latest" in available
        assert "qwen2.5:latest" in available

    @pytest.mark.asyncio
    async def test_validate_model_ollama_unreachable_fail_open(self):
        """Test _validate_model_exists allows change when Ollama is unreachable (fail-open)."""
        tool = SetModelTool()

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.ConnectError("Connection refused")
            is_valid, available = await tool._validate_model_exists("any-model:latest")

        # Should allow the change (fail-open behavior)
        assert is_valid is True
        assert available == []


class TestSetModelToolExecuteWithValidation:
    """Test execute() with model validation."""

    def setup_method(self):
        """Reset RuntimeConfig before each test."""
        RuntimeConfig.reset_instance()

    def teardown_method(self):
        """Reset RuntimeConfig after each test."""
        RuntimeConfig.reset_instance()

    @pytest.mark.asyncio
    async def test_execute_valid_model_success(self):
        """Test execute() changes model when model exists in Ollama."""
        tool = SetModelTool()
        config = RuntimeConfig.get_instance()
        initial_model = config.llm_model

        mock_response = httpx.Response(
            200,
            json={
                "models": [
                    {"name": "llama3:latest"},
                    {"name": "qwen2.5:latest"},
                ]
            },
        )

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            result = await tool.execute({"model_name": "llama3:latest"})

        # Check model was changed
        assert config.llm_model == "llama3:latest"
        assert config.llm_model != initial_model

        # Check result (MCP format)
        assert result["isError"] is False
        assert len(result["content"]) == 1
        assert "llama3:latest" in result["content"][0]["text"]
        # Should NOT have warning about unreachable Ollama
        assert "WARNING" not in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_invalid_model_rejected(self):
        """Test execute() rejects model that doesn't exist in Ollama."""
        tool = SetModelTool()
        config = RuntimeConfig.get_instance()
        initial_model = config.llm_model

        mock_response = httpx.Response(
            200,
            json={
                "models": [
                    {"name": "llama3:latest"},
                    {"name": "qwen2.5:latest"},
                ]
            },
        )

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            result = await tool.execute({"model_name": "nonexistent-model:latest"})

        # Model should NOT be changed
        assert config.llm_model == initial_model

        # Should return error
        assert result["isError"] is True
        message = result["content"][0]["text"]
        assert "nonexistent-model:latest" in message
        assert "not found" in message.lower()
        assert "Available models" in message
        assert "llama3:latest" in message
        assert "ollama pull" in message

    @pytest.mark.asyncio
    async def test_execute_ollama_unreachable_allows_change_with_warning(self):
        """Test execute() allows change but warns when Ollama is unreachable."""
        tool = SetModelTool()
        config = RuntimeConfig.get_instance()

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.ConnectError("Connection refused")
            result = await tool.execute({"model_name": "any-model:latest"})

        # Model should be changed (fail-open)
        assert config.llm_model == "any-model:latest"

        # Should succeed but with warning
        assert result["isError"] is False
        message = result["content"][0]["text"]
        assert "any-model:latest" in message
        assert "WARNING" in message
        assert "unreachable" in message.lower()

    @pytest.mark.asyncio
    async def test_execute_returns_confirmation_message(self):
        """Test that execute() returns detailed confirmation message."""
        tool = SetModelTool()

        mock_response = httpx.Response(
            200,
            json={
                "models": [
                    {"name": "mistral:latest"},
                ]
            },
        )

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            result = await tool.execute({"model_name": "mistral:latest"})

        assert result["isError"] is False
        message = result["content"][0]["text"]

        # Should mention both old and new model
        assert "qwen2.5:latest" in message  # Old default model
        assert "mistral:latest" in message  # New model

        # Should mention what the model is used for
        assert "entity" in message.lower() or "refinement" in message.lower()

    @pytest.mark.asyncio
    async def test_execute_with_various_model_formats(self):
        """Test that execute() accepts various Ollama model name formats when Ollama unreachable."""
        tool = SetModelTool()
        test_models = [
            "llama3:latest",
            "qwen2.5:7b",
            "mistral:instruct",
            "deepseek-coder:6.7b",
            "codellama:13b-python",
        ]

        config = RuntimeConfig.get_instance()

        # Test with Ollama unreachable (fail-open behavior)
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.ConnectError("Connection refused")

            for model in test_models:
                result = await tool.execute({"model_name": model})

                # Should succeed and update config (MCP format)
                assert config.llm_model == model
                assert result["isError"] is False
                assert model in result["content"][0]["text"]


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
        tool = SetModelTool()
        config1 = RuntimeConfig.get_instance()

        mock_response = httpx.Response(
            200,
            json={"models": [{"name": "llama3:latest"}]},
        )

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            await tool.execute({"model_name": "llama3:latest"})

        # Get new config instance (should be same singleton)
        config2 = RuntimeConfig.get_instance()

        assert config1 is config2
        assert config2.llm_model == "llama3:latest"

    @pytest.mark.asyncio
    async def test_multiple_consecutive_model_changes(self):
        """Test multiple consecutive model changes via tool."""
        tool = SetModelTool()
        config = RuntimeConfig.get_instance()
        models = ["model1:latest", "model2:latest", "model3:latest"]

        # Test with Ollama unreachable (fail-open behavior)
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.ConnectError("Connection refused")

            for model in models:
                result = await tool.execute({"model_name": model})

                assert config.llm_model == model
                assert result["isError"] is False

        # Final model should be the last one
        assert config.llm_model == "model3:latest"
