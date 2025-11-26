"""
MarkdownProcessor - Lightweight Markdown normalization utilities.

Provides helpers to clean Markdown syntax, extract structured sections,
collect links, and keep fenced code blocks intact without pulling in heavy
parsing dependencies.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from zapomni_core.exceptions import ProcessingError, ValidationError
from zapomni_core.utils import get_logger

logger = get_logger(__name__)


CODE_BLOCK_PATTERN = re.compile(
    r"(?m)^\s*(?P<fence>`{3,}|~{3,})[ \t]*(?P<lang>[^\n]*)\n"
    r"(?P<code>.*?)(?:\n\s*(?P=fence)[ \t]*$)",
    re.DOTALL | re.MULTILINE,
)
HEADER_PATTERN = re.compile(r"(?m)^\s*(?P<hashes>#{1,6})\s*(?P<header>.+?)(?:\s+#*\s*)?$")
IMAGE_LINK_PATTERN = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
AUTOLINK_PATTERN = re.compile(r"<(https?://[^>\s]+)>")


def _ensure_text_type(text: object, context: str, allow_empty: bool = True) -> str:
    if not isinstance(text, str):
        raise ValidationError(
            message=f"{context} must be a string",
            error_code="VAL_002",
        )
    if not allow_empty and not text.strip():
        raise ValidationError(
            message=f"{context} cannot be empty or whitespace",
            error_code="VAL_001",
        )
    return text


def _strip_code_blocks(text: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        code_text = match.group("code").rstrip()
        return f"\n{code_text}\n"

    return CODE_BLOCK_PATTERN.sub(_replace, text)


def _strip_headings(text: str) -> str:
    return re.sub(r"(?m)^#{1,6}\s*", "", text)


def _strip_blockquotes(text: str) -> str:
    return re.sub(r"(?m)^>\s*", "", text)


def _strip_list_markers(text: str) -> str:
    return re.sub(r"(?m)^\s*([-+*]|\d+\.)\s+", "", text)


def _strip_inline_formatting(text: str) -> str:
    text = re.sub(r"(?s)(\*\*|__)(.+?)\1", r"\2", text)
    text = re.sub(r"(?s)(\*|_)(.+?)\1", r"\2", text)
    text = re.sub(r"(?s)~~(.+?)~~", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return text


def _normalize_links(text: str) -> str:
    def _format(label: str, url: str) -> str:
        clean_label = label.strip()
        clean_url = url.strip()
        if clean_label:
            return f"{clean_label} ({clean_url})"
        return clean_url

    text = IMAGE_LINK_PATTERN.sub(lambda m: _format(m.group(1), m.group(2)), text)
    text = MARKDOWN_LINK_PATTERN.sub(lambda m: _format(m.group(1), m.group(2)), text)
    text = AUTOLINK_PATTERN.sub(lambda m: m.group(1).strip(), text)
    return text


def _cleanup_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def process_markdown(text: str) -> str:
    """
    Clean Markdown syntax while preserving readable structure.

    Args:
        text: Raw Markdown string to normalize.

    Returns:
        Cleaned text without Markdown decorations.
    """
    _text = _ensure_text_type(text, "text", allow_empty=False)

    try:
        normalized = _text.replace("\r\n", "\n")
        normalized = _strip_code_blocks(normalized)
        normalized = _strip_headings(normalized)
        normalized = _strip_blockquotes(normalized)
        normalized = _strip_list_markers(normalized)
        normalized = _strip_inline_formatting(normalized)
        normalized = _normalize_links(normalized)
        normalized = _cleanup_whitespace(normalized)
        return normalized.strip()
    except Exception as exc:
        logger.error("markdown_processing_failed", error=str(exc))
        raise ProcessingError(
            message="Failed to process Markdown text",
            original_exception=exc,
        )


def extract_sections(text: str) -> List[Dict[str, str]]:
    """
    Extract headers with optional content sections from Markdown.

    Args:
        text: Markdown document text.

    Returns:
        Ordered list of sections with header level and content.
    """
    source = _ensure_text_type(text, "text", allow_empty=True)
    matches = list(HEADER_PATTERN.finditer(source))
    sections: List[Dict[str, str]] = []

    for index, match in enumerate(matches):
        header_raw = match.group("header") or ""
        header_text = header_raw.rstrip("#").strip()
        if not header_text:
            continue

        level = len(match.group("hashes") or "")
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(source)
        content = source[start:end].strip()

        sections.append(
            {
                "header": header_text,
                "level": level,
                "content": content,
            }
        )

    return sections


def extract_links(text: str) -> List[str]:
    """
    Extract Markdown links (inline, images, autolinks).

    Args:
        text: Markdown string to scan for links.

    Returns:
        List of string representations containing label and URL.
    """
    source = _ensure_text_type(text, "text", allow_empty=True)
    seen: set[str] = set()
    results: List[str] = []

    for matcher, handler in (
        (IMAGE_LINK_PATTERN, lambda m: (m.group(1) or "image", m.group(2))),
        (MARKDOWN_LINK_PATTERN, lambda m: (m.group(1), m.group(2))),
    ):
        for match in matcher.finditer(source):
            label, url = handler(match)
            entry = f"{label.strip()} ({url.strip()})" if label else url.strip()
            if entry not in seen:
                seen.add(entry)
                results.append(entry)

    for match in AUTOLINK_PATTERN.finditer(source):
        url = match.group(1).strip()
        if url and url not in seen:
            seen.add(url)
            results.append(url)

    return results


def extract_code_blocks(text: str) -> List[Dict[str, Optional[str]]]:
    """
    Find fenced code blocks in the document.

    Args:
        text: Markdown content containing code fences.

    Returns:
        List of dictionaries with language and code payload.
    """
    source = _ensure_text_type(text, "text", allow_empty=True)
    blocks: List[Dict[str, Optional[str]]] = []

    for match in CODE_BLOCK_PATTERN.finditer(source):
        language = match.group("lang").strip()
        code = match.group("code").rstrip()
        blocks.append(
            {
                "language": language or None,
                "code": code,
            }
        )

    return blocks


class MarkdownProcessor:
    """
    Facade for Markdown parsing helpers.
    """

    def __init__(self) -> None:
        self.logger = get_logger(__name__)

    def process(self, text: str) -> str:
        return process_markdown(text)

    def extract_sections(self, text: str) -> List[Dict[str, str]]:
        return extract_sections(text)

    def extract_links(self, text: str) -> List[str]:
        return extract_links(text)

    def extract_code_blocks(self, text: str) -> List[Dict[str, Optional[str]]]:
        return extract_code_blocks(text)
