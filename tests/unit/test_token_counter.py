"""
Unit tests for TokenCounter utility.

Covers token counting with tiktoken, fallback behavior, batch estimation,
and edge cases as specified in the requirements.
"""

from __future__ import annotations

from typing import List

import pytest

from zapomni_core.utils.token_counter import TokenCounter


class TestTokenCounterInit:
    """Tests for TokenCounter initialization."""

    def test_init_default_encoding(self) -> None:
        """Default encoding should be cl100k_base."""
        counter = TokenCounter()
        assert counter.encoding_name == "cl100k_base"

    def test_init_custom_encoding(self) -> None:
        """Custom encoding should be accepted and stored."""
        counter = TokenCounter(encoding="p50k_base")
        assert counter.encoding_name == "p50k_base"

    def test_init_caches_encoding(self) -> None:
        """Encoding should be cached after initialization (or fallback mode enabled)."""
        counter = TokenCounter()
        # Either encoding is cached, or fallback mode is enabled
        assert counter._encoding is not None or counter._fallback_mode is True

    def test_init_with_unavailable_encoding_fallback(self) -> None:
        """Should fallback gracefully if encoding unavailable."""
        # This tests the fallback path - even if tiktoken is available,
        # requesting an invalid encoding should fallback
        counter = TokenCounter(encoding="invalid_encoding")
        # Should not raise, but have fallback mode enabled
        assert counter._fallback_mode is True


class TestTokenCounterCountTokens:
    """Tests for TokenCounter.count_tokens single text counting."""

    def test_count_tokens_normal_text(self) -> None:
        """Counting tokens in normal text should return positive integer."""
        counter = TokenCounter()
        text = "The quick brown fox jumps over the lazy dog"
        token_count = counter.count_tokens(text)

        assert isinstance(token_count, int)
        assert token_count > 0

    def test_count_tokens_empty_string(self) -> None:
        """Empty string should return 0 tokens."""
        counter = TokenCounter()
        token_count = counter.count_tokens("")
        assert token_count == 0

    def test_count_tokens_whitespace_only(self) -> None:
        """Whitespace-only string should return minimal tokens."""
        counter = TokenCounter()
        token_count = counter.count_tokens("   \n\t  ")
        assert token_count == 0

    def test_count_tokens_single_word(self) -> None:
        """Single word should count as at least 1 token."""
        counter = TokenCounter()
        token_count = counter.count_tokens("word")
        assert token_count >= 1

    def test_count_tokens_long_text(self) -> None:
        """Long text should count as more tokens than short text."""
        counter = TokenCounter()
        short_text = "hello"
        long_text = "hello world. This is a longer text with more words and content."

        short_count = counter.count_tokens(short_text)
        long_count = counter.count_tokens(long_text)

        assert long_count > short_count

    def test_count_tokens_unicode(self) -> None:
        """Unicode text should be handled correctly."""
        counter = TokenCounter()
        unicode_text = "Hello 世界 مرحبا мир 🌍"
        token_count = counter.count_tokens(unicode_text)

        assert isinstance(token_count, int)
        assert token_count > 0

    def test_count_tokens_multiline(self) -> None:
        """Multiline text should be counted correctly."""
        counter = TokenCounter()
        text = """This is line 1.
This is line 2.
This is line 3."""
        token_count = counter.count_tokens(text)

        assert isinstance(token_count, int)
        assert token_count > 0

    def test_count_tokens_punctuation(self) -> None:
        """Text with punctuation should be handled correctly."""
        counter = TokenCounter()
        text = "Hello, world! How are you? I'm fine, thanks."
        token_count = counter.count_tokens(text)

        assert isinstance(token_count, int)
        assert token_count > 0

    def test_count_tokens_code_snippet(self) -> None:
        """Code snippet should be tokenized correctly."""
        counter = TokenCounter()
        code = """def hello_world():
    print("Hello, World!")
    return True"""
        token_count = counter.count_tokens(code)

        assert isinstance(token_count, int)
        assert token_count > 0

    def test_count_tokens_json(self) -> None:
        """JSON should be tokenized correctly."""
        counter = TokenCounter()
        json_text = '{"name": "John", "age": 30, "city": "New York"}'
        token_count = counter.count_tokens(json_text)

        assert isinstance(token_count, int)
        assert token_count > 0


class TestTokenCounterEstimateTokens:
    """Tests for TokenCounter.estimate_tokens batch estimation."""

    def test_estimate_tokens_single_text(self) -> None:
        """Estimating single text should return list with one count."""
        counter = TokenCounter()
        texts = ["The quick brown fox jumps over the lazy dog"]
        estimates = counter.estimate_tokens(texts)

        assert isinstance(estimates, list)
        assert len(estimates) == 1
        assert isinstance(estimates[0], int)
        assert estimates[0] > 0

    def test_estimate_tokens_multiple_texts(self) -> None:
        """Estimating multiple texts should return list matching length."""
        counter = TokenCounter()
        texts = ["First text", "Second text is longer than first", "Third"]
        estimates = counter.estimate_tokens(texts)

        assert len(estimates) == 3
        assert all(isinstance(count, int) for count in estimates)
        assert all(count >= 0 for count in estimates)

    def test_estimate_tokens_mixed_lengths(self) -> None:
        """Mixed-length texts should maintain relative ordering."""
        counter = TokenCounter()
        texts = ["a", "This is a longer text with more content", "mid-length text here"]
        estimates = counter.estimate_tokens(texts)

        assert len(estimates) == 3
        assert estimates[0] < estimates[1]  # short < long
        assert estimates[2] < estimates[1]  # mid < long

    def test_estimate_tokens_empty_list(self) -> None:
        """Empty text list should return empty list."""
        counter = TokenCounter()
        estimates = counter.estimate_tokens([])

        assert estimates == []

    def test_estimate_tokens_with_empty_strings(self) -> None:
        """List with empty strings should return zeros."""
        counter = TokenCounter()
        texts = ["hello", "", "world"]
        estimates = counter.estimate_tokens(texts)

        assert len(estimates) == 3
        assert estimates[1] == 0

    def test_estimate_tokens_consistency(self) -> None:
        """Estimate should match count_tokens for individual items."""
        counter = TokenCounter()
        texts = ["First text", "Second text", "Third"]
        estimates = counter.estimate_tokens(texts)

        for text, estimate in zip(texts, estimates):
            count = counter.count_tokens(text)
            assert count == estimate

    def test_estimate_tokens_large_batch(self) -> None:
        """Large batch should be processed without error."""
        counter = TokenCounter()
        texts = [f"Text {i}" for i in range(100)]
        estimates = counter.estimate_tokens(texts)

        assert len(estimates) == 100
        assert all(isinstance(count, int) for count in estimates)

    def test_estimate_tokens_unicode_batch(self) -> None:
        """Batch with unicode should work correctly."""
        counter = TokenCounter()
        texts = ["Hello world", "Hello 世界", "Hello مرحبا", "Hello мир"]
        estimates = counter.estimate_tokens(texts)

        assert len(estimates) == 4
        assert all(count > 0 for count in estimates)


class TestTokenCounterFallback:
    """Tests for fallback behavior when tiktoken unavailable."""

    def test_fallback_mode_count_tokens(self) -> None:
        """Fallback mode should still count tokens (via whitespace split)."""
        counter = TokenCounter()
        # Force fallback mode
        counter._fallback_mode = True
        counter._encoding = None

        text = "The quick brown fox"
        token_count = counter.count_tokens(text)

        # Fallback uses whitespace split, should still work
        assert isinstance(token_count, int)
        assert token_count >= 1

    def test_fallback_mode_estimate_tokens(self) -> None:
        """Fallback mode should work for batch estimation."""
        counter = TokenCounter()
        counter._fallback_mode = True
        counter._encoding = None

        texts = ["Hello world", "Good morning", ""]
        estimates = counter.estimate_tokens(texts)

        assert len(estimates) == 3
        assert estimates[0] > 0
        assert estimates[1] > 0
        assert estimates[2] == 0

    def test_fallback_empty_string(self) -> None:
        """Fallback should handle empty strings."""
        counter = TokenCounter()
        counter._fallback_mode = True

        token_count = counter.count_tokens("")
        assert token_count == 0

    def test_fallback_whitespace_only(self) -> None:
        """Fallback should handle whitespace-only strings."""
        counter = TokenCounter()
        counter._fallback_mode = True

        token_count = counter.count_tokens("   \n\t  ")
        assert token_count == 0


class TestTokenCounterCaching:
    """Tests for encoding caching."""

    def test_encoding_cached_on_init(self) -> None:
        """Encoding should be cached to avoid repeated initialization."""
        counter = TokenCounter()
        encoding1 = counter._encoding
        encoding2 = counter._encoding

        # Should be the same object (cached)
        assert encoding1 is encoding2

    def test_multiple_counters_independent(self) -> None:
        """Multiple TokenCounter instances should be independent."""
        counter1 = TokenCounter()
        counter2 = TokenCounter(encoding="p50k_base")

        # Different configurations should exist
        assert counter1.encoding_name != counter2.encoding_name


class TestTokenCounterEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_very_long_text(self) -> None:
        """Very long text should be handled correctly."""
        counter = TokenCounter()
        # Create 10KB of text
        text = "word " * 10000
        token_count = counter.count_tokens(text)

        assert isinstance(token_count, int)
        assert token_count > 0

    def test_special_characters(self) -> None:
        """Special characters should be handled."""
        counter = TokenCounter()
        text = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        token_count = counter.count_tokens(text)

        assert isinstance(token_count, int)

    def test_newlines_and_tabs(self) -> None:
        """Newlines and tabs should be handled."""
        counter = TokenCounter()
        text = "line1\nline2\n\tindented\n\nspaced"
        token_count = counter.count_tokens(text)

        assert isinstance(token_count, int)
        assert token_count > 0

    def test_repeated_words(self) -> None:
        """Repeated words might get optimized in tokenization."""
        counter = TokenCounter()
        text = "word " * 50
        token_count = counter.count_tokens(text)

        assert isinstance(token_count, int)
        assert token_count > 0

    def test_mixed_case(self) -> None:
        """Mixed case should be handled."""
        counter = TokenCounter()
        text = "HeLLo WoRLd ThIs Is MiXeD CaSe"
        token_count = counter.count_tokens(text)

        assert isinstance(token_count, int)
        assert token_count > 0


class TestTokenCounterIntegration:
    """Integration tests for TokenCounter."""

    def test_count_and_estimate_consistency(self) -> None:
        """count_tokens and estimate_tokens should be consistent."""
        counter = TokenCounter()
        texts = ["The quick brown fox", "Python is great", "Machine learning is powerful"]

        # Using estimate_tokens
        estimates = counter.estimate_tokens(texts)

        # Using count_tokens individually
        individual_counts = [counter.count_tokens(text) for text in texts]

        assert estimates == individual_counts

    def test_practical_chunking_scenario(self) -> None:
        """Practical scenario: counting tokens for chunk validation."""
        counter = TokenCounter()

        # Simulate chunks of different sizes
        chunks = [
            "First chunk of text",
            "Second chunk with more content and longer text",
            "Third short chunk",
        ]

        estimates = counter.estimate_tokens(chunks)

        # Verify we can identify which chunks need merging (e.g., if min size is 50 tokens)
        min_chunk_size = 50
        small_chunks = [text for text, count in zip(chunks, estimates) if count < min_chunk_size]

        # At least some chunks should be small enough to potentially merge
        assert len(small_chunks) > 0

    def test_encoding_switching(self) -> None:
        """TokenCounter should support different encodings."""
        counter_cl = TokenCounter(encoding="cl100k_base")
        text = "Hello world"

        count_cl = counter_cl.count_tokens(text)
        assert isinstance(count_cl, int)
        assert count_cl > 0

    def test_fallback_graceful_degradation(self) -> None:
        """If tiktoken fails, should fallback gracefully."""
        counter = TokenCounter()

        # Simulate tiktoken failure by forcing fallback
        counter._fallback_mode = True
        counter._encoding = None

        text = "Test text for fallback mode"
        count = counter.count_tokens(text)

        # Should still work, albeit with whitespace-based approximation
        assert count > 0
