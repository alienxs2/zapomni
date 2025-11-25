"""
Tests for RuntimeConfig singleton.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from zapomni_core.runtime_config import RuntimeConfig


class TestRuntimeConfigSingleton:
    """Test singleton behavior of RuntimeConfig."""

    def setup_method(self):
        """Reset singleton before each test."""
        RuntimeConfig.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        RuntimeConfig.reset_instance()

    def test_get_instance_returns_same_instance(self):
        """Test that get_instance() always returns the same instance."""
        config1 = RuntimeConfig.get_instance()
        config2 = RuntimeConfig.get_instance()

        assert config1 is config2

    def test_direct_instantiation_raises_error(self):
        """Test that direct instantiation is prevented."""
        # First call via get_instance to create singleton
        RuntimeConfig.get_instance()

        # Second call via __init__ should raise error
        with pytest.raises(RuntimeError, match="singleton"):
            RuntimeConfig()

    def test_reset_instance_clears_singleton(self):
        """Test that reset_instance() allows new instance creation."""
        config1 = RuntimeConfig.get_instance()

        RuntimeConfig.reset_instance()

        config2 = RuntimeConfig.get_instance()

        # Should be different instances after reset
        assert config1 is not config2


class TestRuntimeConfigLLMModel:
    """Test LLM model configuration."""

    def setup_method(self):
        """Reset singleton before each test."""
        RuntimeConfig.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        RuntimeConfig.reset_instance()

    def test_default_llm_model(self):
        """Test default LLM model value."""
        config = RuntimeConfig.get_instance()

        assert config.llm_model == "qwen2.5:latest"

    def test_set_llm_model(self):
        """Test setting LLM model."""
        config = RuntimeConfig.get_instance()

        config.set_llm_model("llama3:latest")

        assert config.llm_model == "llama3:latest"

    def test_set_llm_model_persists_across_get_instance(self):
        """Test that model change persists across get_instance() calls."""
        config1 = RuntimeConfig.get_instance()
        config1.set_llm_model("mistral:latest")

        config2 = RuntimeConfig.get_instance()

        assert config2.llm_model == "mistral:latest"

    def test_get_all_config(self):
        """Test get_all_config returns all configuration."""
        config = RuntimeConfig.get_instance()
        config.set_llm_model("llama3:latest")

        all_config = config.get_all_config()

        assert all_config == {"llm_model": "llama3:latest"}


class TestRuntimeConfigThreadSafety:
    """Test thread-safety of RuntimeConfig."""

    def setup_method(self):
        """Reset singleton before each test."""
        RuntimeConfig.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        RuntimeConfig.reset_instance()

    def test_concurrent_get_instance(self):
        """Test that concurrent get_instance() calls return same instance."""
        instances = []

        def get_instance():
            instances.append(RuntimeConfig.get_instance())

        # Create multiple threads trying to get instance
        threads = [threading.Thread(target=get_instance) for _ in range(10)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # All instances should be the same
        assert len(instances) == 10
        assert all(instance is instances[0] for instance in instances)

    def test_concurrent_set_llm_model(self):
        """Test that concurrent model changes are thread-safe."""
        config = RuntimeConfig.get_instance()
        results = []

        def set_and_read_model(model_name):
            config.set_llm_model(model_name)
            # Small delay to increase chance of race condition
            import time

            time.sleep(0.001)
            results.append(config.llm_model)

        # Create threads with different model names
        models = [f"model{i}:latest" for i in range(10)]
        threads = [threading.Thread(target=set_and_read_model, args=(model,)) for model in models]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Final model should be one of the set models
        final_model = config.llm_model
        assert final_model in models

        # All read results should be valid model names
        assert all(result in models for result in results)

    def test_concurrent_reads_during_write(self):
        """Test that reads are safe during concurrent writes."""
        config = RuntimeConfig.get_instance()
        read_results = []

        def read_model():
            for _ in range(100):
                read_results.append(config.llm_model)

        def write_model(model_name):
            for _ in range(10):
                config.set_llm_model(model_name)

        # Start readers and writers concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            # 3 readers, 2 writers
            futures = [
                executor.submit(read_model),
                executor.submit(read_model),
                executor.submit(read_model),
                executor.submit(write_model, "model1:latest"),
                executor.submit(write_model, "model2:latest"),
            ]

            # Wait for all to complete
            for future in futures:
                future.result()

        # All reads should return valid model names (no corrupted data)
        valid_models = ["qwen2.5:latest", "model1:latest", "model2:latest"]
        assert all(result in valid_models for result in read_results)
        assert len(read_results) == 300  # 3 readers * 100 reads each
