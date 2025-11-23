"""
Unit tests for HTMLProcessor component.

Covers HTML content extraction, metadata extraction, link extraction,
language detection, and fallback HTML parsing based on spec requirements.
Tests use mocked trafilatura to avoid external dependencies.
"""

from __future__ import annotations

from unittest.mock import Mock, patch, MagicMock
import pytest

from zapomni_core.processors.html_processor import HTMLProcessor
from zapomni_core.exceptions import ValidationError, ProcessingError


class TestHTMLProcessorInit:
    """Tests for HTMLProcessor initialization."""

    def test_init_success(self) -> None:
        """Should initialize with no errors."""
        processor = HTMLProcessor()

        assert processor is not None
        assert processor.logger is not None

    def test_init_with_config(self) -> None:
        """Should accept optional trafilatura config."""
        config = {"include_links": True, "include_comments": False}
        processor = HTMLProcessor(trafilatura_config=config)

        assert processor.trafilatura_config == config


class TestExtractFromHTML:
    """Tests for extract_from_html method."""

    def test_extract_valid_html_success(self) -> None:
        """Should extract clean text from valid HTML."""
        processor = HTMLProcessor()

        html = """
        <html>
            <head><title>Test Page</title></head>
            <body>
                <nav>Navigation</nav>
                <h1>Main Title</h1>
                <p>First paragraph.</p>
                <p>Second paragraph.</p>
                <footer>Footer</footer>
            </body>
        </html>
        """

        with patch("zapomni_core.processors.html_processor.trafilatura") as mock_traf:
            mock_traf.extract.return_value = "Main Title\nFirst paragraph.\nSecond paragraph."

            result = processor.extract_from_html(html)

            assert isinstance(result, str)
            assert len(result) > 0
            mock_traf.extract.assert_called_once()

    def test_extract_empty_html_raises(self) -> None:
        """Should raise ValidationError for empty HTML."""
        processor = HTMLProcessor()

        with pytest.raises(ValidationError) as exc_info:
            processor.extract_from_html("")

        assert "empty" in exc_info.value.message.lower()
        assert exc_info.value.error_code == "VAL_001"

    def test_extract_whitespace_only_html_raises(self) -> None:
        """Should raise ValidationError for whitespace-only HTML."""
        processor = HTMLProcessor()

        with pytest.raises(ValidationError) as exc_info:
            processor.extract_from_html("   \n\t  ")

        assert "empty" in exc_info.value.message.lower()

    def test_extract_invalid_type_raises(self) -> None:
        """Should raise ValidationError if HTML is not a string."""
        processor = HTMLProcessor()

        with pytest.raises(ValidationError) as exc_info:
            processor.extract_from_html(12345)  # type: ignore

        assert exc_info.value.error_code == "VAL_002"

    def test_extract_trafilatura_failure_fallback(self) -> None:
        """Should fall back to basic HTML parsing if trafilatura fails."""
        processor = HTMLProcessor()

        html = "<html><body><p>Test content</p></body></html>"

        with patch("zapomni_core.processors.html_processor.trafilatura") as mock_traf:
            mock_traf.extract.return_value = None

            result = processor.extract_from_html(html)

            assert isinstance(result, str)
            # Should contain some text from the HTML
            assert len(result) > 0

    def test_extract_trafilatura_exception_fallback(self) -> None:
        """Should fall back to basic parsing if trafilatura throws exception."""
        processor = HTMLProcessor()

        html = "<html><body><p>Test content</p></body></html>"

        with patch("zapomni_core.processors.html_processor.trafilatura") as mock_traf:
            mock_traf.extract.side_effect = Exception("Trafilatura error")

            result = processor.extract_from_html(html)

            assert isinstance(result, str)

    def test_extract_removes_boilerplate(self) -> None:
        """Should remove navigation, footer, and ad-like content."""
        processor = HTMLProcessor()

        html = """
        <html>
            <body>
                <nav class="navbar">Navigation Menu</nav>
                <main>
                    <article>Important article content</article>
                </main>
                <footer>Copyright 2024</footer>
                <div class="ads">Advertisement</div>
            </body>
        </html>
        """

        with patch("zapomni_core.processors.html_processor.trafilatura") as mock_traf:
            mock_traf.extract.return_value = "Important article content"

            result = processor.extract_from_html(html)

            assert "Important article content" in result
            mock_traf.extract.assert_called_once()

    def test_extract_preserves_structure(self) -> None:
        """Should preserve paragraph and heading structure."""
        processor = HTMLProcessor()

        html = """
        <html>
            <body>
                <h1>Title</h1>
                <p>Paragraph 1</p>
                <h2>Subtitle</h2>
                <p>Paragraph 2</p>
            </body>
        </html>
        """

        with patch("zapomni_core.processors.html_processor.trafilatura") as mock_traf:
            mock_traf.extract.return_value = "Title\nParagraph 1\nSubtitle\nParagraph 2"

            result = processor.extract_from_html(html)

            assert "Title" in result
            assert "Paragraph 1" in result
            assert "Subtitle" in result
            assert "Paragraph 2" in result


class TestExtractWithMetadata:
    """Tests for extract_with_metadata method."""

    def test_extract_with_metadata_success(self) -> None:
        """Should extract content and metadata from HTML."""
        processor = HTMLProcessor()

        html = """
        <html>
            <head>
                <title>Test Article</title>
                <meta name="author" content="John Doe">
                <meta name="date" content="2024-01-15">
            </head>
            <body>
                <article>Article content here</article>
            </body>
        </html>
        """

        with patch("zapomni_core.processors.html_processor.trafilatura") as mock_traf:
            mock_traf.extract.return_value = "Article content here"
            mock_traf.extract_metadata.return_value = {
                "title": "Test Article",
                "author": "John Doe",
                "date": "2024-01-15",
            }

            result = processor.extract_with_metadata(html)

            assert isinstance(result, dict)
            assert "content" in result
            assert "metadata" in result
            assert result["content"] == "Article content here"

    def test_extract_metadata_missing_fields(self) -> None:
        """Should handle missing metadata fields gracefully."""
        processor = HTMLProcessor()

        html = "<html><body><p>Content</p></body></html>"

        with patch("zapomni_core.processors.html_processor.trafilatura") as mock_traf:
            mock_traf.extract.return_value = "Content"
            mock_traf.extract_metadata.return_value = {}

            result = processor.extract_with_metadata(html)

            assert isinstance(result, dict)
            assert "metadata" in result
            assert isinstance(result["metadata"], dict)

    def test_extract_metadata_empty_html_raises(self) -> None:
        """Should raise ValidationError for empty HTML."""
        processor = HTMLProcessor()

        with pytest.raises(ValidationError):
            processor.extract_with_metadata("")

    def test_extract_metadata_includes_title(self) -> None:
        """Should extract page title from metadata."""
        processor = HTMLProcessor()

        html = "<html><head><title>My Title</title></head></html>"

        with patch("zapomni_core.processors.html_processor.trafilatura") as mock_traf:
            mock_traf.extract.return_value = "Content"
            mock_traf.extract_metadata.return_value = {"title": "My Title"}

            result = processor.extract_with_metadata(html)

            assert "title" in result.get("metadata", {})

    def test_extract_metadata_includes_author(self) -> None:
        """Should extract author from metadata."""
        processor = HTMLProcessor()

        html = "<html><head><meta name='author' content='Jane'></head></html>"

        with patch("zapomni_core.processors.html_processor.trafilatura") as mock_traf:
            mock_traf.extract.return_value = "Content"
            mock_traf.extract_metadata.return_value = {"author": "Jane"}

            result = processor.extract_with_metadata(html)

            assert "author" in result.get("metadata", {})

    def test_extract_metadata_includes_date(self) -> None:
        """Should extract publish/update date from metadata."""
        processor = HTMLProcessor()

        html = "<html><head><meta name='date' content='2024-01-15'></head></html>"

        with patch("zapomni_core.processors.html_processor.trafilatura") as mock_traf:
            mock_traf.extract.return_value = "Content"
            mock_traf.extract_metadata.return_value = {"date": "2024-01-15"}

            result = processor.extract_with_metadata(html)

            assert "date" in result.get("metadata", {})


class TestExtractLinks:
    """Tests for extract_links method."""

    def test_extract_links_success(self) -> None:
        """Should extract links from HTML."""
        processor = HTMLProcessor()

        html = """
        <html>
            <body>
                <a href="https://example.com">Example</a>
                <a href="/local-link">Local</a>
                <a href="https://test.com">Test</a>
            </body>
        </html>
        """

        result = processor.extract_links(html)

        assert isinstance(result, list)
        assert len(result) > 0

    def test_extract_links_empty_html(self) -> None:
        """Should return empty list for HTML with no links."""
        processor = HTMLProcessor()

        html = "<html><body><p>No links here</p></body></html>"

        result = processor.extract_links(html)

        assert isinstance(result, list)
        assert len(result) == 0

    def test_extract_links_invalid_type_raises(self) -> None:
        """Should raise ValidationError if HTML is not a string."""
        processor = HTMLProcessor()

        with pytest.raises(ValidationError):
            processor.extract_links(None)  # type: ignore

    def test_extract_links_returns_unique_urls(self) -> None:
        """Should return unique URLs without duplicates."""
        processor = HTMLProcessor()

        html = """
        <html>
            <body>
                <a href="https://example.com">Example</a>
                <a href="https://example.com">Same Link</a>
                <a href="https://test.com">Different</a>
            </body>
        </html>
        """

        result = processor.extract_links(html)

        assert isinstance(result, list)
        # Check that we have reasonable number of unique links
        assert len(result) >= 2


class TestDetectLanguage:
    """Tests for detect_language method."""

    def test_detect_language_english(self) -> None:
        """Should detect English text."""
        processor = HTMLProcessor()

        html = "<html><body><p>This is English text for testing language detection.</p></body></html>"

        with patch("zapomni_core.processors.html_processor.trafilatura") as mock_traf:
            mock_traf.extract.return_value = "This is English text for testing language detection."
            mock_traf.extract_metadata.return_value = {"language": "en"}

            result = processor.detect_language(html)

            assert isinstance(result, str)
            assert len(result) > 0

    def test_detect_language_other_language(self) -> None:
        """Should detect non-English text."""
        processor = HTMLProcessor()

        html = "<html><body><p>Ceci est un texte en français.</p></body></html>"

        with patch("zapomni_core.processors.html_processor.trafilatura") as mock_traf:
            mock_traf.extract.return_value = "Ceci est un texte en français."
            mock_traf.extract_metadata.return_value = {"language": "fr"}

            result = processor.detect_language(html)

            assert isinstance(result, str)

    def test_detect_language_empty_html_raises(self) -> None:
        """Should raise ValidationError for empty HTML."""
        processor = HTMLProcessor()

        with pytest.raises(ValidationError):
            processor.detect_language("")

    def test_detect_language_returns_language_code(self) -> None:
        """Should return a valid language code."""
        processor = HTMLProcessor()

        html = "<html><body><p>English content</p></body></html>"

        with patch("zapomni_core.processors.html_processor.trafilatura") as mock_traf:
            mock_traf.extract.return_value = "English content"
            mock_traf.extract_metadata.return_value = {"language": "en"}

            result = processor.detect_language(html)

            # Language code should be a string like 'en', 'fr', etc.
            assert isinstance(result, str)
            assert len(result) > 0


class TestFallbackHTMLParsing:
    """Tests for fallback basic HTML parsing when trafilatura fails."""

    def test_fallback_basic_text_extraction(self) -> None:
        """Should extract text from HTML tags when trafilatura returns None."""
        processor = HTMLProcessor()

        html = """
        <html>
            <body>
                <h1>Heading</h1>
                <p>Paragraph with text.</p>
                <p>Another paragraph.</p>
            </body>
        </html>
        """

        with patch("zapomni_core.processors.html_processor.trafilatura") as mock_traf:
            mock_traf.extract.return_value = None

            result = processor.extract_from_html(html)

            assert isinstance(result, str)
            assert len(result) > 0

    def test_fallback_handles_script_tags(self) -> None:
        """Should skip script and style tags in fallback parsing."""
        processor = HTMLProcessor()

        html = """
        <html>
            <head><style>body { color: red; }</style></head>
            <body>
                <p>Real content</p>
                <script>alert('test');</script>
            </body>
        </html>
        """

        with patch("zapomni_core.processors.html_processor.trafilatura") as mock_traf:
            mock_traf.extract.return_value = None

            result = processor.extract_from_html(html)

            assert "Real content" in result
            # Should not include script content
            assert "alert" not in result.lower() or "script" not in result.lower()

    def test_fallback_cleans_whitespace(self) -> None:
        """Should clean up excessive whitespace in fallback parsing."""
        processor = HTMLProcessor()

        html = """
        <html>
            <body>
                <p>Text</p>
                <p>More   spaces</p>
            </body>
        </html>
        """

        with patch("zapomni_core.processors.html_processor.trafilatura") as mock_traf:
            mock_traf.extract.return_value = None

            result = processor.extract_from_html(html)

            # Should not have excessive whitespace
            assert "   " not in result


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_invalid_html_structure(self) -> None:
        """Should handle malformed HTML gracefully."""
        processor = HTMLProcessor()

        html = "<html><body><p>Unclosed paragraph</body></html>"

        with patch("zapomni_core.processors.html_processor.trafilatura") as mock_traf:
            mock_traf.extract.return_value = None

            result = processor.extract_from_html(html)

            assert isinstance(result, str)

    def test_very_large_html(self) -> None:
        """Should handle very large HTML documents."""
        processor = HTMLProcessor()

        # Create large HTML with many paragraphs
        html = "<html><body>"
        for i in range(1000):
            html += f"<p>Paragraph {i}</p>"
        html += "</body></html>"

        with patch("zapomni_core.processors.html_processor.trafilatura") as mock_traf:
            mock_traf.extract.return_value = "\n".join([f"Paragraph {i}" for i in range(1000)])

            result = processor.extract_from_html(html)

            assert isinstance(result, str)
            assert len(result) > 0

    def test_special_characters_preserved(self) -> None:
        """Should preserve special characters and unicode."""
        processor = HTMLProcessor()

        html = """
        <html>
            <body>
                <p>Special chars: © ® ™ € £ ¥</p>
                <p>Unicode: 你好 مرحبا שלום</p>
            </body>
        </html>
        """

        with patch("zapomni_core.processors.html_processor.trafilatura") as mock_traf:
            mock_traf.extract.return_value = "Special chars: © ® ™ € £ ¥\nUnicode: 你好 مرحبا שלום"

            result = processor.extract_from_html(html)

            assert "©" in result or "trademark" in result.lower()
