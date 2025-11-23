"""
HTMLProcessor - Extracts clean text from HTML using trafilatura.

Provides high-level HTML processing with content extraction, metadata
extraction, link extraction, language detection, and fallback parsing.

Pipeline: HTML → clean text + metadata (title, author, date, links, language)

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from typing import Dict, Any, List, Optional
import structlog
from html.parser import HTMLParser
from urllib.parse import urljoin
import re

try:
    import trafilatura
except ImportError:
    trafilatura = None  # type: ignore

from zapomni_core.exceptions import ValidationError, ProcessingError


class _HTMLTextExtractor(HTMLParser):
    """
    Basic HTML parser for fallback text extraction.

    Used when trafilatura fails or returns None.
    Strips HTML tags while preserving paragraph structure.
    """

    def __init__(self) -> None:
        """Initialize the HTML text extractor."""
        super().__init__()
        self.text_parts: List[str] = []
        self.in_script = False
        self.in_style = False

    def handle_starttag(self, tag: str, attrs: List[tuple]) -> None:
        """Handle opening HTML tags."""
        if tag in ("script", "style"):
            if tag == "script":
                self.in_script = True
            else:
                self.in_style = True
        elif tag in ("p", "div", "h1", "h2", "h3", "h4", "h5", "h6"):
            # Add newline before block elements if text exists
            if self.text_parts and self.text_parts[-1].strip():
                self.text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        """Handle closing HTML tags."""
        if tag == "script":
            self.in_script = False
        elif tag == "style":
            self.in_style = False
        elif tag in ("p", "div", "h1", "h2", "h3", "h4", "h5", "h6"):
            # Add newline after block elements
            if self.text_parts and self.text_parts[-1].strip():
                self.text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        """Handle text content between tags."""
        if not self.in_script and not self.in_style:
            text = data.strip()
            if text:
                self.text_parts.append(text)
                self.text_parts.append(" ")

    def get_text(self) -> str:
        """Get extracted text with cleaned whitespace."""
        text = "".join(self.text_parts)
        # Clean up excessive whitespace
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n\s+", "\n", text)
        return text.strip()


class _LinkExtractor(HTMLParser):
    """
    HTML parser for extracting links from HTML.

    Extracts href attributes from anchor tags.
    """

    def __init__(self, base_url: Optional[str] = None) -> None:
        """Initialize the link extractor."""
        super().__init__()
        self.links: List[str] = []
        self.base_url = base_url

    def handle_starttag(self, tag: str, attrs: List[tuple]) -> None:
        """Handle opening HTML tags."""
        if tag == "a":
            for attr, value in attrs:
                if attr == "href" and value:
                    # Convert relative URLs to absolute if base_url provided
                    if self.base_url and not value.startswith(("http://", "https://", "ftp://")):
                        value = urljoin(self.base_url, value)
                    self.links.append(value)

    def get_links(self) -> List[str]:
        """Get extracted unique links."""
        return list(dict.fromkeys(self.links))  # Remove duplicates while preserving order


class HTMLProcessor:
    """
    Extract clean text and metadata from HTML documents.

    Provides:
    - Main content extraction (via trafilatura)
    - Metadata extraction (title, author, date)
    - Link extraction
    - Language detection
    - Fallback basic HTML parsing

    Example:
        ```python
        processor = HTMLProcessor()

        # Extract main content
        text = processor.extract_from_html(html)

        # Extract with metadata
        result = processor.extract_with_metadata(html)
        print(f"Title: {result['metadata'].get('title')}")
        print(f"Content: {result['content']}")

        # Extract links
        links = processor.extract_links(html)

        # Detect language
        lang = processor.detect_language(html)
        ```
    """

    def __init__(self, trafilatura_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize HTMLProcessor.

        Args:
            trafilatura_config: Optional configuration dict for trafilatura
                               (e.g., {"include_links": True, "include_comments": False})

        Example:
            ```python
            config = {
                "include_links": True,
                "include_comments": False,
                "favor_precision": True
            }
            processor = HTMLProcessor(trafilatura_config=config)
            ```
        """
        self.trafilatura_config = trafilatura_config or {}
        self.logger = structlog.get_logger(__name__)

        self.logger.info(
            "html_processor_initialized",
            trafilatura_available=trafilatura is not None,
            config_keys=list(self.trafilatura_config.keys()) if self.trafilatura_config else []
        )

    def extract_from_html(self, html: str) -> str:
        """
        Extract clean main content from HTML.

        Uses trafilatura for intelligent extraction, falls back to basic
        HTML parsing if trafilatura fails.

        Args:
            html: HTML string to process

        Returns:
            Extracted clean text (main content only)

        Raises:
            ValidationError: If HTML is not a string or is empty

        Example:
            ```python
            processor = HTMLProcessor()
            html = '''<html><body>
                <nav>Menu</nav>
                <article>Main content</article>
                <footer>Footer</footer>
            </body></html>'''

            text = processor.extract_from_html(html)
            # Returns: "Main content"
            ```
        """
        # VALIDATION
        if not isinstance(html, str):
            raise ValidationError(
                message=f"html must be a string, got {type(html).__name__}",
                error_code="VAL_002",
                details={"expected_type": "str", "actual_type": type(html).__name__}
            )

        if not html or not html.strip():
            raise ValidationError(
                message="html cannot be empty or whitespace-only",
                error_code="VAL_001",
                details={"html_length": len(html)}
            )

        self.logger.debug(
            "extract_from_html_started",
            html_length=len(html),
            has_trafilatura=trafilatura is not None
        )

        # TRY TRAFILATURA EXTRACTION
        if trafilatura:
            try:
                extracted = trafilatura.extract(
                    html,
                    **self.trafilatura_config
                )

                if extracted:
                    self.logger.info(
                        "trafilatura_extraction_success",
                        content_length=len(extracted)
                    )
                    return extracted

                self.logger.debug("trafilatura_returned_none")

            except Exception as e:
                self.logger.warning(
                    "trafilatura_extraction_failed",
                    error=str(e),
                    error_type=type(e).__name__
                )

        # FALLBACK: BASIC HTML PARSING
        self.logger.debug("using_fallback_html_parsing")

        try:
            extractor = _HTMLTextExtractor()
            extractor.feed(html)
            text = extractor.get_text()

            if text:
                self.logger.info(
                    "fallback_extraction_success",
                    content_length=len(text)
                )
                return text

            raise ProcessingError(
                message="Could not extract any text from HTML using fallback parser",
                error_code="PROC_002",
                details={"html_length": len(html)}
            )

        except ProcessingError:
            raise
        except Exception as e:
            raise ProcessingError(
                message=f"Fallback HTML parsing failed: {e}",
                error_code="PROC_002",
                original_exception=e
            )

    def extract_with_metadata(self, html: str) -> Dict[str, Any]:
        """
        Extract content and metadata from HTML.

        Returns dict with:
        - content: Main extracted text
        - metadata: Dict with title, author, date (if available)

        Args:
            html: HTML string to process

        Returns:
            Dict with 'content' and 'metadata' keys

        Raises:
            ValidationError: If HTML is not a string or is empty

        Example:
            ```python
            processor = HTMLProcessor()
            result = processor.extract_with_metadata(html)

            print(result["content"])  # Main text
            print(result["metadata"]["title"])  # Page title
            print(result["metadata"]["author"])  # Article author
            print(result["metadata"]["date"])  # Publication date
            ```
        """
        # VALIDATION
        if not isinstance(html, str):
            raise ValidationError(
                message=f"html must be a string, got {type(html).__name__}",
                error_code="VAL_002",
                details={"expected_type": "str", "actual_type": type(html).__name__}
            )

        if not html or not html.strip():
            raise ValidationError(
                message="html cannot be empty or whitespace-only",
                error_code="VAL_001",
                details={"html_length": len(html)}
            )

        self.logger.debug("extract_with_metadata_started")

        # EXTRACT CONTENT
        content = self.extract_from_html(html)

        # EXTRACT METADATA
        metadata: Dict[str, Any] = {}

        if trafilatura:
            try:
                meta = trafilatura.extract_metadata(html)
                if meta:
                    metadata = {
                        "title": meta.get("title"),
                        "author": meta.get("author"),
                        "date": meta.get("date"),
                    }
                    # Remove None values
                    metadata = {k: v for k, v in metadata.items() if v is not None}

                    self.logger.debug(
                        "metadata_extracted",
                        metadata_keys=list(metadata.keys())
                    )

            except Exception as e:
                self.logger.warning(
                    "metadata_extraction_failed",
                    error=str(e),
                    error_type=type(e).__name__
                )

        self.logger.info(
            "extract_with_metadata_complete",
            content_length=len(content),
            metadata_keys=list(metadata.keys())
        )

        return {
            "content": content,
            "metadata": metadata
        }

    def extract_links(self, html: str) -> List[str]:
        """
        Extract all links from HTML.

        Returns list of unique URLs found in href attributes.

        Args:
            html: HTML string to process

        Returns:
            List of unique URLs (order-preserved)

        Raises:
            ValidationError: If HTML is not a string

        Example:
            ```python
            processor = HTMLProcessor()
            links = processor.extract_links(html)
            # Returns: ["https://example.com", "/local-link", ...]
            ```
        """
        # VALIDATION
        if not isinstance(html, str):
            raise ValidationError(
                message=f"html must be a string, got {type(html).__name__}",
                error_code="VAL_002",
                details={"expected_type": "str", "actual_type": type(html).__name__}
            )

        self.logger.debug("extract_links_started")

        try:
            extractor = _LinkExtractor()
            extractor.feed(html)
            links = extractor.get_links()

            self.logger.info(
                "links_extracted",
                num_links=len(links)
            )

            return links

        except Exception as e:
            self.logger.error(
                "link_extraction_failed",
                error=str(e),
                error_type=type(e).__name__
            )
            raise ProcessingError(
                message=f"Failed to extract links from HTML: {e}",
                error_code="PROC_003",
                original_exception=e
            )

    def detect_language(self, html: str) -> str:
        """
        Detect language of HTML content.

        Args:
            html: HTML string to process

        Returns:
            Language code (e.g., 'en', 'fr', 'de')

        Raises:
            ValidationError: If HTML is not a string or is empty
            ProcessingError: If language detection fails

        Example:
            ```python
            processor = HTMLProcessor()
            lang = processor.detect_language(html)
            print(f"Language: {lang}")  # Output: "Language: en"
            ```
        """
        # VALIDATION
        if not isinstance(html, str):
            raise ValidationError(
                message=f"html must be a string, got {type(html).__name__}",
                error_code="VAL_002",
                details={"expected_type": "str", "actual_type": type(html).__name__}
            )

        if not html or not html.strip():
            raise ValidationError(
                message="html cannot be empty or whitespace-only",
                error_code="VAL_001",
                details={"html_length": len(html)}
            )

        self.logger.debug("detect_language_started")

        try:
            # First extract content
            content = self.extract_from_html(html)

            # Try trafilatura language detection
            if trafilatura:
                try:
                    meta = trafilatura.extract_metadata(html)
                    if meta and meta.get("language"):
                        lang = meta["language"]
                        self.logger.info(
                            "language_detected",
                            language=lang
                        )
                        return lang
                except Exception as e:
                    self.logger.debug(
                        "trafilatura_language_detection_failed",
                        error=str(e)
                    )

            # Fallback: Simple language detection based on content
            # Try to detect common language patterns
            lang = self._simple_language_detect(content)

            self.logger.info(
                "language_detected",
                language=lang,
                method="fallback"
            )

            return lang

        except ValidationError:
            raise
        except Exception as e:
            raise ProcessingError(
                message=f"Failed to detect language: {e}",
                error_code="PROC_004",
                original_exception=e
            )

    def _simple_language_detect(self, text: str) -> str:
        """
        Simple fallback language detection.

        Uses character ranges and common patterns to guess language.

        Args:
            text: Text to analyze

        Returns:
            Language code (defaults to 'en')
        """
        if not text:
            return "en"

        # Check for common non-Latin scripts
        if re.search(r"[\u4e00-\u9fff]", text):  # Chinese
            return "zh"
        if re.search(r"[\u3040-\u309f\u30a0-\u30ff]", text):  # Japanese
            return "ja"
        if re.search(r"[\uac00-\ud7af]", text):  # Korean
            return "ko"
        if re.search(r"[\u0600-\u06ff]", text):  # Arabic
            return "ar"
        if re.search(r"[\u0400-\u04ff]", text):  # Cyrillic (Russian, etc.)
            return "ru"
        if re.search(r"[\u0590-\u05ff]", text):  # Hebrew
            return "he"

        # Default to English for Latin script
        return "en"
