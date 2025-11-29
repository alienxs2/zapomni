"""
Unit tests for OllamaEmbedder component.

Comprehensive test suite following TDD principles:
- Happy path tests
- Error handling tests
- Retry & fallback tests
- Edge case tests
- Performance tests

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest

from zapomni_core.embeddings.ollama_embedder import OllamaEmbedder
from zapomni_core.exceptions import EmbeddingError, ValidationError

# === Fixtures ===


@pytest.fixture
def mock_ollama_response():
    """Mock successful Ollama API response."""
    return {"embedding": [0.1] * 768}


@pytest.fixture
def mock_httpx_client():
    """Mock httpx.AsyncClient for Ollama API calls."""
    client = AsyncMock(spec=httpx.AsyncClient)
    return client


@pytest.fixture
def mock_sentence_transformer():
    """Mock sentence-transformers model for fallback."""
    mock_model = MagicMock()
    # sentence-transformers returns numpy array
    import numpy as np

    mock_model.encode.return_value = np.array([0.1] * 384)
    return mock_model


# === Happy Path Tests ===


@pytest.mark.unit
def test_init_success():
    """Test successful initialization with default parameters."""
    embedder = OllamaEmbedder()

    assert embedder.base_url == "http://localhost:11434"
    assert embedder.model_name == "nomic-embed-text"
    assert embedder.timeout == 30
    assert embedder.max_retries == 3
    assert embedder.enable_fallback is True
    assert embedder.dimensions == 768


@pytest.mark.unit
def test_init_custom_params():
    """Test initialization with custom parameters."""
    embedder = OllamaEmbedder(
        base_url="http://192.168.1.100:11434",
        model_name="custom-model",
        timeout=60,
        max_retries=5,
        enable_fallback=False,
    )

    assert embedder.base_url == "http://192.168.1.100:11434"
    assert embedder.model_name == "custom-model"
    assert embedder.timeout == 60
    assert embedder.max_retries == 5
    assert embedder.enable_fallback is False


@pytest.mark.unit
def test_get_dimensions():
    """Test get_dimensions returns correct value."""
    embedder = OllamaEmbedder()
    assert embedder.get_dimensions() == 768


@pytest.mark.unit
@pytest.mark.asyncio
async def test_embed_text_success(mock_ollama_response, mock_httpx_client):
    """Test successful single text embedding."""
    # Setup mock response
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = Mock(return_value=mock_ollama_response)  # json() is sync
    mock_httpx_client.post = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        embedder = OllamaEmbedder()
        embedding = await embedder.embed_text("Python is great")

        assert isinstance(embedding, list)
        assert len(embedding) == 768
        assert all(isinstance(x, float) for x in embedding)
        mock_httpx_client.post.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_embed_batch_success(mock_ollama_response, mock_httpx_client):
    """Test successful batch embedding using batch API."""
    # Create mock responses for both batch API and single API
    mock_embedding = [0.1] * 768

    def create_response(url, **kwargs):
        response = AsyncMock()
        response.status_code = 200
        if "/api/embed" in str(url) and "/api/embeddings" not in str(url):
            # Batch API response
            input_texts = kwargs.get("json", {}).get("input", [])
            response.json = Mock(return_value={"embeddings": [mock_embedding] * len(input_texts)})
        else:
            # Single API response
            response.json = Mock(return_value=mock_ollama_response)
        return response

    mock_httpx_client.post = AsyncMock(side_effect=create_response)

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        embedder = OllamaEmbedder()
        texts = ["Python is great", "Ollama runs locally", "Privacy matters"]
        embeddings = await embedder.embed_batch(texts, batch_size=2)

        assert isinstance(embeddings, list)
        assert len(embeddings) == 3
        assert all(len(emb) == 768 for emb in embeddings)
        assert all(isinstance(emb, list) for emb in embeddings)
        # With batch API: 2 calls (batch[0:2] + batch[2:3])
        assert mock_httpx_client.post.call_count == 2


# === Error Tests ===


@pytest.mark.unit
def test_init_invalid_base_url_raises():
    """Test initialization with invalid base URL."""
    with pytest.raises(ValidationError, match="Invalid base_url"):
        OllamaEmbedder(base_url="not-a-url")


@pytest.mark.unit
def test_init_invalid_timeout_raises():
    """Test initialization with invalid timeout."""
    with pytest.raises(ValidationError, match="timeout must be positive"):
        OllamaEmbedder(timeout=0)

    with pytest.raises(ValidationError, match="timeout must be positive"):
        OllamaEmbedder(timeout=-5)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_embed_text_empty_raises():
    """Test embed_text with empty text raises ValidationError."""
    embedder = OllamaEmbedder()

    with pytest.raises(ValidationError, match="Text cannot be empty"):
        await embedder.embed_text("")

    with pytest.raises(ValidationError, match="Text cannot be empty"):
        await embedder.embed_text("   ")  # Whitespace only


@pytest.mark.unit
@pytest.mark.asyncio
async def test_embed_text_too_long_raises(mock_httpx_client):
    """Test embed_text with text exceeding max length."""
    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        embedder = OllamaEmbedder()
        # Create very long text (simulate > 8192 tokens)
        long_text = "word " * 10000

        with pytest.raises(ValidationError, match="exceeds max length"):
            await embedder.embed_text(long_text)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_embed_batch_empty_list_raises():
    """Test embed_batch with empty texts list."""
    embedder = OllamaEmbedder()

    with pytest.raises(ValidationError, match="texts list cannot be empty"):
        await embedder.embed_batch([])


@pytest.mark.unit
@pytest.mark.asyncio
async def test_embed_batch_invalid_batch_size_raises():
    """Test embed_batch with invalid batch_size."""
    embedder = OllamaEmbedder()
    texts = ["text1", "text2"]

    with pytest.raises(ValidationError, match="batch_size must be between 1 and 64"):
        await embedder.embed_batch(texts, batch_size=0)

    with pytest.raises(ValidationError, match="batch_size must be between 1 and 64"):
        await embedder.embed_batch(texts, batch_size=100)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_embed_text_invalid_dimensions_raises(mock_httpx_client):
    """Test embed_text raises error when Ollama returns wrong dimensions."""
    # Mock response with wrong dimensions (512 instead of 768)
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = Mock(return_value={"embedding": [0.1] * 512})  # json() is sync
    mock_httpx_client.post = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        embedder = OllamaEmbedder(enable_fallback=False)

        with pytest.raises(EmbeddingError, match="Invalid embedding dimensions"):
            await embedder.embed_text("test")


# === Retry & Fallback Tests ===


@pytest.mark.unit
@pytest.mark.asyncio
async def test_embed_text_retry_on_timeout(mock_ollama_response, mock_httpx_client):
    """Test retry logic on timeout - succeeds on 3rd attempt."""
    # First 2 calls timeout, 3rd succeeds
    mock_response_success = AsyncMock()
    mock_response_success.status_code = 200
    mock_response_success.json = Mock(return_value=mock_ollama_response)  # json() is sync

    mock_httpx_client.post = AsyncMock(
        side_effect=[
            httpx.TimeoutException("Timeout 1"),
            httpx.TimeoutException("Timeout 2"),
            mock_response_success,
        ]
    )

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        with patch("asyncio.sleep", new_callable=AsyncMock):  # Speed up test
            embedder = OllamaEmbedder()
            embedding = await embedder.embed_text("test")

            assert len(embedding) == 768
            assert mock_httpx_client.post.call_count == 3


@pytest.mark.unit
@pytest.mark.asyncio
async def test_embed_text_fallback_on_failure(mock_httpx_client, mock_sentence_transformer):
    """Test fallback to sentence-transformers when Ollama fails."""
    # Ollama always fails
    mock_httpx_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        with patch(
            "zapomni_core.embeddings.ollama_embedder.SentenceTransformer",
            return_value=mock_sentence_transformer,
        ):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                embedder = OllamaEmbedder(enable_fallback=True)
                embedding = await embedder.embed_text("test")

                # Fallback returns 384 real + 384 zeros = 768
                assert len(embedding) == 768
                assert mock_sentence_transformer.encode.called


@pytest.mark.unit
@pytest.mark.asyncio
async def test_embed_text_fails_if_no_fallback(mock_httpx_client):
    """Test raises EmbeddingError when Ollama fails and fallback disabled."""
    # Ollama fails
    mock_httpx_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        with patch("asyncio.sleep", new_callable=AsyncMock):
            embedder = OllamaEmbedder(enable_fallback=False)

            with pytest.raises(EmbeddingError, match="Failed to generate embedding"):
                await embedder.embed_text("test")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_embed_batch_partial_failures(mock_ollama_response, mock_httpx_client):
    """Test batch processing with some failures (< 50%)."""
    # First text succeeds, second fails, third succeeds
    mock_response_success = AsyncMock()
    mock_response_success.status_code = 200
    mock_response_success.json = Mock(return_value=mock_ollama_response)  # json() is sync

    mock_httpx_client.post = AsyncMock(
        side_effect=[
            mock_response_success,  # text1 succeeds
            httpx.ConnectError("Failed"),  # text2 fails (triggers retries)
            httpx.ConnectError("Failed"),
            httpx.ConnectError("Failed"),
            httpx.ConnectError("Failed"),  # All retries fail
            mock_response_success,  # text3 succeeds
        ]
    )

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        with patch("asyncio.sleep", new_callable=AsyncMock):
            embedder = OllamaEmbedder(enable_fallback=False)

            # With our implementation, ANY failure raises immediately
            with pytest.raises(EmbeddingError, match="Failed to embed text at index"):
                texts = ["text1", "text2", "text3"]
                await embedder.embed_batch(texts, batch_size=1)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_check_success(mock_ollama_response, mock_httpx_client):
    """Test health_check returns True when Ollama is available."""
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = Mock(return_value=mock_ollama_response)  # json() is sync
    mock_httpx_client.post = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        embedder = OllamaEmbedder()
        result = await embedder.health_check()

        assert result is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_check_failure(mock_httpx_client):
    """Test health_check returns False when Ollama is unavailable."""
    mock_httpx_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        embedder = OllamaEmbedder()
        result = await embedder.health_check()

        assert result is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_check_model_not_found(mock_httpx_client):
    """Test health_check returns False when model not found."""
    mock_response = AsyncMock()
    mock_response.status_code = 404
    mock_response.json = Mock(return_value={"error": "model not found"})  # json() is sync
    mock_httpx_client.post = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        embedder = OllamaEmbedder()
        result = await embedder.health_check()

        assert result is False


# === Edge Cases ===


@pytest.mark.unit
@pytest.mark.asyncio
async def test_embed_batch_single_text(mock_ollama_response, mock_httpx_client):
    """Test embed_batch with single text works correctly."""
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = Mock(return_value=mock_ollama_response)  # json() is sync
    mock_httpx_client.post = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        embedder = OllamaEmbedder()
        embeddings = await embedder.embed_batch(["single text"])

        assert len(embeddings) == 1
        assert len(embeddings[0]) == 768


@pytest.mark.unit
@pytest.mark.asyncio
async def test_embed_batch_size_larger_than_texts(mock_ollama_response, mock_httpx_client):
    """Test embed_batch when batch_size > number of texts using batch API."""
    mock_embedding = [0.1] * 768

    def create_response(url, **kwargs):
        response = AsyncMock()
        response.status_code = 200
        if "/api/embed" in str(url) and "/api/embeddings" not in str(url):
            # Batch API response
            input_texts = kwargs.get("json", {}).get("input", [])
            response.json = Mock(return_value={"embeddings": [mock_embedding] * len(input_texts)})
        else:
            # Single API response
            response.json = Mock(return_value=mock_ollama_response)
        return response

    mock_httpx_client.post = AsyncMock(side_effect=create_response)

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        embedder = OllamaEmbedder()
        texts = ["a", "b"]
        embeddings = await embedder.embed_batch(texts, batch_size=32)

        assert len(embeddings) == 2
        # With batch API: 1 call for all texts (batch_size > len(texts))
        assert mock_httpx_client.post.call_count == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_embed_text_special_characters(mock_ollama_response, mock_httpx_client):
    """Test embed_text handles special characters correctly."""
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = Mock(return_value=mock_ollama_response)  # json() is sync
    mock_httpx_client.post = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        embedder = OllamaEmbedder()
        # Test various special characters
        text = "Hello 世界! 🎉 \n\t Special: @#$%"
        embedding = await embedder.embed_text(text)

        assert len(embedding) == 768


@pytest.mark.unit
@pytest.mark.asyncio
async def test_validate_embedding_with_nan_raises():
    """Test _validate_embedding raises error for NaN values."""
    embedder = OllamaEmbedder()
    import math

    invalid_embedding = [0.1] * 767 + [math.nan]

    with pytest.raises(EmbeddingError, match="contains NaN or Inf"):
        embedder._validate_embedding(invalid_embedding)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_validate_embedding_with_inf_raises():
    """Test _validate_embedding raises error for Inf values."""
    embedder = OllamaEmbedder()
    import math

    invalid_embedding = [0.1] * 767 + [math.inf]

    with pytest.raises(EmbeddingError, match="contains NaN or Inf"):
        embedder._validate_embedding(invalid_embedding)


# === Performance Tests (Mocked) ===


@pytest.mark.unit
@pytest.mark.asyncio
async def test_embed_text_performance(mock_ollama_response, mock_httpx_client):
    """Test single embedding performance target (< 150ms)."""
    import time

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = Mock(return_value=mock_ollama_response)  # json() is sync
    mock_httpx_client.post = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        embedder = OllamaEmbedder()

        start = time.time()
        await embedder.embed_text("Python is great")
        elapsed = (time.time() - start) * 1000  # Convert to ms

        # Mocked should be very fast
        assert elapsed < 150


@pytest.mark.unit
@pytest.mark.asyncio
async def test_embed_batch_performance(mock_ollama_response, mock_httpx_client):
    """Test batch embedding performance target (32 texts < 1000ms)."""
    import time

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = Mock(return_value=mock_ollama_response)  # json() is sync
    mock_httpx_client.post = AsyncMock(return_value=mock_response)

    with patch("httpx.AsyncClient", return_value=mock_httpx_client):
        embedder = OllamaEmbedder()
        texts = [f"text {i}" for i in range(32)]

        start = time.time()
        await embedder.embed_batch(texts, batch_size=32)
        elapsed = (time.time() - start) * 1000

        # Mocked should be very fast
        assert elapsed < 1000


# === Integration Tests (Optional - Require Ollama) ===


@pytest.mark.integration
@pytest.mark.requires_ollama
@pytest.mark.asyncio
async def test_embed_text_real_ollama():
    """
    Test with real Ollama API (requires Ollama running).

    This test is marked as integration and requires_ollama,
    so it will be skipped in normal unit test runs.
    """
    embedder = OllamaEmbedder()

    # Check Ollama is available
    if not await embedder.health_check():
        pytest.skip("Ollama not available")

    # Generate real embedding
    embedding = await embedder.embed_text("Python is a programming language")

    assert len(embedding) == 768
    assert all(isinstance(x, float) for x in embedding)
    # Embeddings should be finite floats - actual range varies by model
    assert all(isinstance(x, float) and not (x != x) for x in embedding)  # Check no NaN


@pytest.mark.integration
@pytest.mark.requires_ollama
@pytest.mark.asyncio
async def test_health_check_real_ollama():
    """Test health check with real Ollama API."""
    embedder = OllamaEmbedder()
    result = await embedder.health_check()

    # This will be True if Ollama is running, False otherwise
    # We just check it returns a boolean
    assert isinstance(result, bool)


# === Cleanup Tests ===


@pytest.mark.unit
@pytest.mark.asyncio
async def test_client_cleanup():
    """Test that httpx client is properly cleaned up."""
    embedder = OllamaEmbedder()

    # Client should be created
    assert embedder.client is not None

    # Cleanup
    await embedder.client.aclose()

    # Verify closed (in real implementation)
    assert embedder.client.is_closed
