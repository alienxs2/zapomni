"""
Unit test fixtures.

Isolates unit tests from environment variables (.env file)
to ensure tests verify actual default values.

Copyright (c) 2025 Goncharenko Anton aka alienxs2
License: MIT
"""

import os
import tempfile

import pytest


# Environment variables that affect ZapomniSettings defaults
CONFIG_ENV_VARS = [
    "FALKORDB_HOST",
    "FALKORDB_PORT",
    "GRAPH_NAME",
    "FALKORDB_CONNECTION_TIMEOUT",
    "FALKORDB_POOL_SIZE",
    "OLLAMA_BASE_URL",
    "OLLAMA_EMBEDDING_MODEL",
    "OLLAMA_LLM_MODEL",
    "OLLAMA_EMBEDDING_TIMEOUT",
    "OLLAMA_LLM_TIMEOUT",
    "MAX_CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "VECTOR_DIMENSIONS",
    "LOG_LEVEL",
    "LOG_FORMAT",
    "LOG_FILE",
    "ENABLE_HYBRID_SEARCH",
    "ENABLE_KNOWLEDGE_GRAPH",
    "ENABLE_CODE_INDEXING",
    "ENABLE_SEMANTIC_CACHE",
    "SIMILARITY_THRESHOLD",
    "LLM_PROVIDER",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
]


@pytest.fixture(autouse=True)
def clean_env_for_unit_tests(monkeypatch, tmp_path):
    """
    Remove all config-related environment variables and change working
    directory to avoid loading .env file.

    This ensures unit tests verify actual default values, not values
    from .env file or system environment.
    """
    # Remove env variables
    for var in CONFIG_ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    # Change to temp directory to avoid loading .env from project root
    original_dir = os.getcwd()
    os.chdir(tmp_path)
    yield
    os.chdir(original_dir)
