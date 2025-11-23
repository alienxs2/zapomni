"""
Unit tests for EmbeddingCache.

Tests Redis-backed embedding cache with:
- get/set operations
- Cache key generation with text normalization
- TTL support
- Hit/miss statistics
- Mock Redis client (no external dependencies)

Copyright (c) 2025 Goncharenko Anton aka alienxs2
License: MIT
"""

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from zapomni_core.embeddings.embedding_cache import EmbeddingCache
from zapomni_core.exceptions import ValidationError


# ============================================================
# FIXTURES
# ============================================================


@pytest.fixture
def mock_redis_client():
    """Create a mock Redis client."""
    client = AsyncMock()
    return client


@pytest.fixture
def embedding_cache(mock_redis_client):
    """Create an EmbeddingCache instance with mock Redis."""
    cache = EmbeddingCache(
        redis_client=mock_redis_client,
        ttl_seconds=3600,
    )
    return cache


@pytest.fixture
def sample_embedding():
    """Create a sample embedding vector (768-dimensional)."""
    return [0.1] * 768


@pytest.fixture
def sample_text():
    """Create a sample text to embed."""
    return "This is a sample text for embedding"


# ============================================================
# CACHE KEY GENERATION TESTS
# ============================================================


@pytest.mark.unit
def test_cache_key_generation_normalization(embedding_cache, sample_text):
    """Test that cache keys normalize text (lowercase, whitespace, etc)."""
    text1 = "Hello World"
    text2 = "hello world"
    text3 = "HELLO   WORLD"

    key1 = embedding_cache._generate_cache_key(text1)
    key2 = embedding_cache._generate_cache_key(text2)
    key3 = embedding_cache._generate_cache_key(text3)

    # All should generate same key due to normalization
    assert key1 == key2 == key3
    # Key should be a string (hash-based)
    assert isinstance(key1, str)
    assert len(key1) > 0


@pytest.mark.unit
def test_cache_key_generation_different_texts(embedding_cache):
    """Test that different texts generate different cache keys."""
    text1 = "Text A"
    text2 = "Text B"

    key1 = embedding_cache._generate_cache_key(text1)
    key2 = embedding_cache._generate_cache_key(text2)

    assert key1 != key2


@pytest.mark.unit
def test_cache_key_generation_with_special_chars(embedding_cache):
    """Test cache key generation with special characters."""
    text = "Hello! @#$% World & (stuff)"

    key = embedding_cache._generate_cache_key(text)

    # Should handle special characters gracefully
    assert isinstance(key, str)
    assert len(key) > 0


@pytest.mark.unit
def test_cache_key_generation_whitespace_handling(embedding_cache):
    """Test that multiple whitespace types are normalized."""
    text1 = "A B C"
    text2 = "A  B  C"  # Multiple spaces
    text3 = "A\tB\tC"  # Tabs
    text4 = "A\nB\nC"  # Newlines

    key1 = embedding_cache._generate_cache_key(text1)
    key2 = embedding_cache._generate_cache_key(text2)
    key3 = embedding_cache._generate_cache_key(text3)
    key4 = embedding_cache._generate_cache_key(text4)

    # All should normalize to same key
    assert key1 == key2 == key3 == key4


# ============================================================
# GET/SET OPERATIONS TESTS
# ============================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_set_embedding(embedding_cache, mock_redis_client, sample_text, sample_embedding):
    """Test setting an embedding in cache."""
    mock_redis_client.set.return_value = True

    result = await embedding_cache.set(sample_text, sample_embedding)

    # Should call Redis set with proper TTL
    mock_redis_client.set.assert_called_once()
    # Verify it was called with a key and value
    call_args = mock_redis_client.set.call_args
    assert call_args is not None
    assert call_args.kwargs.get("ex") == 3600  # TTL should be 3600


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_embedding_hit(embedding_cache, mock_redis_client, sample_text, sample_embedding):
    """Test getting an embedding that exists in cache (cache hit)."""
    import json

    # Mock successful Redis get
    mock_redis_client.get.return_value = json.dumps(sample_embedding)

    result = await embedding_cache.get(sample_text)

    # Should return embedding
    assert result == sample_embedding
    # Should increment hit count
    assert embedding_cache.stats["hits"] == 1
    assert embedding_cache.stats["misses"] == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_embedding_miss(embedding_cache, mock_redis_client, sample_text):
    """Test getting an embedding that doesn't exist in cache (cache miss)."""
    # Mock Redis miss (returns None)
    mock_redis_client.get.return_value = None

    result = await embedding_cache.get(sample_text)

    # Should return None
    assert result is None
    # Should increment miss count
    assert embedding_cache.stats["hits"] == 0
    assert embedding_cache.stats["misses"] == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_set_then_get(embedding_cache, mock_redis_client, sample_text, sample_embedding):
    """Test setting and then getting an embedding."""
    import json

    # Setup mocks
    mock_redis_client.set.return_value = True
    mock_redis_client.get.return_value = json.dumps(sample_embedding)

    # Set
    await embedding_cache.set(sample_text, sample_embedding)

    # Get
    result = await embedding_cache.get(sample_text)

    assert result == sample_embedding
    assert embedding_cache.stats["hits"] == 1
    assert embedding_cache.stats["misses"] == 0


# ============================================================
# VALIDATION TESTS
# ============================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_with_empty_text_raises_error(embedding_cache):
    """Test that getting empty text raises ValidationError."""
    with pytest.raises(ValidationError):
        await embedding_cache.get("")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_with_none_text_raises_error(embedding_cache):
    """Test that getting None text raises ValidationError."""
    with pytest.raises(ValidationError):
        await embedding_cache.get(None)  # type: ignore


@pytest.mark.unit
@pytest.mark.asyncio
async def test_set_with_empty_text_raises_error(embedding_cache, sample_embedding):
    """Test that setting empty text raises ValidationError."""
    with pytest.raises(ValidationError):
        await embedding_cache.set("", sample_embedding)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_set_with_invalid_embedding_dimension(embedding_cache, sample_text):
    """Test that setting embedding with wrong dimensions raises error."""
    invalid_embedding = [0.1] * 384  # Wrong dimension (should be 768)

    with pytest.raises(ValidationError):
        await embedding_cache.set(sample_text, invalid_embedding)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_set_with_empty_embedding_list(embedding_cache, sample_text):
    """Test that setting empty embedding raises ValidationError."""
    with pytest.raises(ValidationError):
        await embedding_cache.set(sample_text, [])


@pytest.mark.unit
@pytest.mark.asyncio
async def test_set_with_non_list_embedding(embedding_cache, sample_text):
    """Test that setting non-list embedding raises ValidationError."""
    with pytest.raises(ValidationError):
        await embedding_cache.set(sample_text, "not a list")  # type: ignore


# ============================================================
# STATISTICS TESTS
# ============================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_statistics_initialization(embedding_cache):
    """Test that statistics are initialized correctly."""
    assert embedding_cache.stats["hits"] == 0
    assert embedding_cache.stats["misses"] == 0
    assert embedding_cache.stats["total_requests"] == 0
    assert embedding_cache.stats["hit_rate"] == 0.0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_statistics_hit_rate_calculation(embedding_cache, mock_redis_client):
    """Test that hit rate is calculated correctly."""
    import json

    embedding = [0.1] * 768
    mock_redis_client.get.return_value = json.dumps(embedding)

    # Simulate 3 hits and 2 misses
    await embedding_cache.get("text1")  # hit
    mock_redis_client.get.return_value = None
    await embedding_cache.get("text2")  # miss
    await embedding_cache.get("text3")  # miss
    mock_redis_client.get.return_value = json.dumps(embedding)
    await embedding_cache.get("text4")  # hit
    await embedding_cache.get("text5")  # hit

    # hit_rate = 3 / 5 = 0.6
    assert embedding_cache.stats["hits"] == 3
    assert embedding_cache.stats["misses"] == 2
    assert embedding_cache.stats["total_requests"] == 5
    assert abs(embedding_cache.stats["hit_rate"] - 0.6) < 0.01


@pytest.mark.unit
@pytest.mark.asyncio
async def test_statistics_all_misses(embedding_cache, mock_redis_client):
    """Test statistics when all requests are misses."""
    mock_redis_client.get.return_value = None

    await embedding_cache.get("text1")
    await embedding_cache.get("text2")
    await embedding_cache.get("text3")

    assert embedding_cache.stats["hits"] == 0
    assert embedding_cache.stats["misses"] == 3
    assert embedding_cache.stats["total_requests"] == 3
    assert embedding_cache.stats["hit_rate"] == 0.0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_statistics_all_hits(embedding_cache, mock_redis_client):
    """Test statistics when all requests are hits."""
    import json

    embedding = [0.1] * 768
    mock_redis_client.get.return_value = json.dumps(embedding)

    await embedding_cache.get("text1")
    await embedding_cache.get("text2")
    await embedding_cache.get("text3")

    assert embedding_cache.stats["hits"] == 3
    assert embedding_cache.stats["misses"] == 0
    assert embedding_cache.stats["total_requests"] == 3
    assert embedding_cache.stats["hit_rate"] == 1.0


@pytest.mark.unit
def test_get_statistics(embedding_cache):
    """Test getting statistics snapshot."""
    embedding_cache.stats["hits"] = 100
    embedding_cache.stats["misses"] = 25
    embedding_cache.stats["total_requests"] = 125
    embedding_cache.stats["hit_rate"] = 0.8

    stats = embedding_cache.get_statistics()

    assert stats["hits"] == 100
    assert stats["misses"] == 25
    assert stats["total_requests"] == 125
    assert stats["hit_rate"] == 0.8


# ============================================================
# TTL TESTS
# ============================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_custom_ttl(mock_redis_client, sample_text, sample_embedding):
    """Test setting custom TTL."""
    custom_ttl = 7200  # 2 hours
    cache = EmbeddingCache(redis_client=mock_redis_client, ttl_seconds=custom_ttl)

    mock_redis_client.set.return_value = True

    await cache.set(sample_text, sample_embedding)

    # Verify TTL was passed to Redis
    call_args = mock_redis_client.set.call_args
    assert call_args.kwargs.get("ex") == custom_ttl


@pytest.mark.unit
@pytest.mark.asyncio
async def test_default_ttl(mock_redis_client, sample_text, sample_embedding):
    """Test that default TTL is 1 hour."""
    cache = EmbeddingCache(redis_client=mock_redis_client)

    mock_redis_client.set.return_value = True

    await cache.set(sample_text, sample_embedding)

    # Verify default TTL (1 hour = 3600 seconds)
    call_args = mock_redis_client.set.call_args
    assert call_args.kwargs.get("ex") == 3600


# ============================================================
# ERROR HANDLING TESTS
# ============================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_redis_connection_error_on_get(embedding_cache, mock_redis_client):
    """Test handling Redis connection errors on get."""
    mock_redis_client.get.side_effect = Exception("Redis connection failed")

    with pytest.raises(Exception):
        await embedding_cache.get("some text")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_redis_connection_error_on_set(embedding_cache, mock_redis_client, sample_embedding):
    """Test handling Redis connection errors on set."""
    mock_redis_client.set.side_effect = Exception("Redis connection failed")

    with pytest.raises(Exception):
        await embedding_cache.set("some text", sample_embedding)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_invalid_json_in_cache(embedding_cache, mock_redis_client):
    """Test handling invalid JSON from Redis."""
    mock_redis_client.get.return_value = "not valid json"

    with pytest.raises(Exception):
        await embedding_cache.get("some text")


# ============================================================
# EDGE CASES
# ============================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_with_whitespace_only_text(embedding_cache):
    """Test that whitespace-only text is rejected."""
    with pytest.raises(ValidationError):
        await embedding_cache.get("   ")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_set_with_whitespace_only_text(embedding_cache, sample_embedding):
    """Test that whitespace-only text is rejected."""
    with pytest.raises(ValidationError):
        await embedding_cache.set("   ", sample_embedding)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_very_long_text(embedding_cache, mock_redis_client, sample_embedding):
    """Test handling very long text."""
    long_text = "a" * 10000

    mock_redis_client.set.return_value = True

    # Should handle long text without issues
    await embedding_cache.set(long_text, sample_embedding)

    mock_redis_client.set.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_embedding_with_nan_values(embedding_cache, sample_text):
    """Test handling embeddings with NaN values."""
    import math

    embedding_with_nan = [0.1] * 767 + [math.nan]

    with pytest.raises(ValidationError):
        await embedding_cache.set(sample_text, embedding_with_nan)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_embedding_with_inf_values(embedding_cache, sample_text):
    """Test handling embeddings with infinity values."""
    import math

    embedding_with_inf = [0.1] * 767 + [math.inf]

    with pytest.raises(ValidationError):
        await embedding_cache.set(sample_text, embedding_with_inf)


# ============================================================
# INTEGRATION SCENARIOS
# ============================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_multiple_texts_independence(embedding_cache, mock_redis_client, sample_embedding):
    """Test that cache entries for different texts are independent."""
    import json

    text1 = "Text A"
    text2 = "Text B"
    embedding1 = [0.1] * 768
    embedding2 = [0.2] * 768

    # Setup mock to return different embeddings for different keys
    def mock_get_side_effect(key):
        if "text a" in key.lower():  # Key is based on normalized text
            return json.dumps(embedding1)
        elif "text b" in key.lower():
            return json.dumps(embedding2)
        return None

    mock_redis_client.get.side_effect = mock_get_side_effect
    mock_redis_client.set.return_value = True

    # Set both
    await embedding_cache.set(text1, embedding1)
    await embedding_cache.set(text2, embedding2)

    # Get both
    result1 = await embedding_cache.get(text1)
    result2 = await embedding_cache.get(text2)

    assert result1 == embedding1
    assert result2 == embedding2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cache_hit_rate_target_60_percent(embedding_cache, mock_redis_client, sample_embedding):
    """Test that cache can achieve 60%+ hit rate."""
    import json

    # Simulate realistic usage: some texts hit, some miss
    hit_texts = ["python", "django", "flask"]  # These will hit
    miss_texts = ["react", "vue", "svelte"]   # These will miss

    def mock_get_side_effect(key):
        # Return embedding for hit texts, None for miss texts
        for text in hit_texts:
            if text in key.lower():
                return json.dumps(sample_embedding)
        return None

    mock_redis_client.get.side_effect = mock_get_side_effect
    mock_redis_client.set.return_value = True

    # Access: 3 hits + 3 misses = 50% (edge case)
    for text in hit_texts:
        await embedding_cache.get(text)
    for text in miss_texts:
        await embedding_cache.get(text)

    # Access hit texts again: 3 more hits
    for text in hit_texts:
        await embedding_cache.get(text)

    # Total: 6 hits, 3 misses = 66.7% hit rate (exceeds 60% target)
    hit_rate = embedding_cache.stats["hit_rate"]
    assert hit_rate >= 0.60, f"Expected hit rate >= 60%, got {hit_rate * 100}%"
