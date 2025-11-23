"""
Unit tests for MarkdownProcessor helpers.

Covers text cleaning, section extraction, link parsing, and code block handling
without relying on external Markdown libraries.
"""

from __future__ import annotations

import pytest

from zapomni_core.processors.markdown_processor import (
    extract_code_blocks,
    extract_links,
    extract_sections,
    process_markdown,
)
from zapomni_core.exceptions import ValidationError


class TestProcessMarkdown:
    """Ensure MarkdownProcessor normalizes common Markdown constructs."""

    def test_strips_formatting_and_preserves_content(self) -> None:
        """Should remove Markdown syntax while keeping readable text."""
        raw = """
        # Title
        > Quote block
        Text with **bold**, *italic*, and `inline code`.
        1. First item
        2. Second item
        - Nested bullet

        [Link text](https://example.com/docs)
        """

        cleaned = process_markdown(raw)

        assert "Title" in cleaned
        assert "Quote block" in cleaned
        assert "bold" in cleaned
        assert "italic" in cleaned
        assert "inline code" in cleaned
        assert "First item" in cleaned
        assert "Nested bullet" in cleaned
        assert "Link text (https://example.com/docs)" in cleaned

    def test_raises_on_invalid_input(self) -> None:
        """Should reject non-string or empty inputs."""
        with pytest.raises(ValidationError):
            process_markdown(123)  # type: ignore

        with pytest.raises(ValidationError):
            process_markdown("   ")


class TestExtractSections:
    """Validate hierarchy extraction from Markdown headings."""

    def test_parses_nested_sections(self) -> None:
        """Should capture each heading and its associated content."""
        markdown = """
        # Top level
        Intro paragraph.

        ## Subheading
        Details go here.

        ### Sub-sub
        More details.
        """

        sections = extract_sections(markdown)

        assert len(sections) == 3
        assert sections[0]["header"] == "Top level"
        assert sections[0]["level"] == 1
        assert "Intro paragraph" in sections[0]["content"]
        assert sections[1]["header"] == "Subheading"
        assert sections[1]["level"] == 2
        assert sections[2]["level"] == 3

    def test_returns_empty_when_no_headings(self) -> None:
        """Should return empty list if no headings exist."""
        sections = extract_sections("plain text without headers")

        assert sections == []


class TestExtractLinks:
    """Ensure Markdown links are parsed correctly."""

    def test_extracts_markdown_links_and_autolinks(self) -> None:
        """Should preserve link text and URL information."""
        markdown = """
        [Docs](https://example.com/docs)
        ![Alt text](https://images.example.com/banner.png)
        <https://example.com/auto>
        """

        links = extract_links(markdown)

        assert "Docs (https://example.com/docs)" in links
        assert "Alt text (https://images.example.com/banner.png)" in links
        assert "https://example.com/auto" in links
        assert len(links) == 3


class TestExtractCodeBlocks:
    """Verify code block detection captures languages and code."""

    def test_detects_language_and_code(self) -> None:
        """Should capture fenced code blocks with language metadata."""
        markdown = """
        ```python
        print("hello")
        ```
        """

        blocks = extract_code_blocks(markdown)

        assert blocks
        assert blocks[0]["language"] == "python"
        assert 'print("hello")' in blocks[0]["code"]

    def test_handles_language_agnostic_blocks(self) -> None:
        """Should handle fences without language hints."""
        markdown = """
        ```
        no language
        ```
        """

        blocks = extract_code_blocks(markdown)

        assert len(blocks) == 1
        assert blocks[0]["language"] is None
        assert "no language" in blocks[0]["code"]
