"""
Unit tests for the PDFProcessor component.

Cover text extraction, metadata enrichment, and error handling scenarios.
"""

from __future__ import annotations

from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from zapomni_core.exceptions import ProcessingError, ValidationError
from zapomni_core.processors.pdf_processor import PDFProcessor


def _create_pdf_path(tmp_path: Path, name: str = "document.pdf") -> Path:
    path = tmp_path / name
    path.write_bytes(b"%PDF-1.4\n")
    return path


def _build_page(text: str) -> MagicMock:
    page = MagicMock()
    page.get_text.return_value = text
    return page


def _mock_document(
    page_texts: List[str],
    metadata: dict | None = None,
    is_encrypted: bool = False,
) -> MagicMock:
    doc = MagicMock()
    doc.metadata = metadata or {}
    doc.page_count = len(page_texts)
    doc.is_encrypted = is_encrypted
    doc.needs_password = is_encrypted
    doc.load_page.side_effect = [_build_page(text) for text in page_texts]
    return doc


def _attach_context_manager(
    mock_open: MagicMock, document: MagicMock
) -> None:
    context = MagicMock()
    context.__enter__.return_value = document
    context.__exit__.return_value = False
    mock_open.return_value = context


class TestPDFProcessor:
    """Coverage for both extraction helpers."""

    def setup_method(self) -> None:
        self.processor = PDFProcessor()

    @patch("zapomni_core.processors.pdf_processor.fitz.open")
    def test_process_pdf_extracts_text_from_pages(
        self, mock_open: MagicMock, tmp_path: Path
    ) -> None:
        """Should extract and clean text from each page."""
        pdf_path = _create_pdf_path(tmp_path)
        page_texts = ["  Hello\nWorld  ", "\nSecond   page\n"]
        document = _mock_document(page_texts)
        _attach_context_manager(mock_open, document)

        result = self.processor.process_pdf(str(pdf_path))

        assert result == "Hello\nWorld\nSecond page"
        mock_open.assert_called_once_with(str(pdf_path))

    @patch("zapomni_core.processors.pdf_processor.fitz.open")
    def test_process_pdf_with_metadata_returns_text_and_data(
        self, mock_open: MagicMock, tmp_path: Path
    ) -> None:
        """Should return both the cleaned text and expected metadata."""
        pdf_path = _create_pdf_path(tmp_path)
        metadata_dict = {
            "title": "Report",
            "author": "Analyst",
            "creationDate": "D:20250101000000",
            "modDate": "D:20250102000000",
        }
        document = _mock_document(["Page content"], metadata=metadata_dict)
        _attach_context_manager(mock_open, document)

        result = self.processor.process_pdf_with_metadata(str(pdf_path))

        assert result["text"] == "Page content"
        assert result["metadata"]["title"] == "Report"
        assert result["metadata"]["author"] == "Analyst"
        assert result["metadata"]["date"] == "D:20250101000000"
        assert result["metadata"]["pages"] == 1

    @patch("zapomni_core.processors.pdf_processor.fitz.open")
    def test_process_pdf_encrypted_raises_processing_error(
        self, mock_open: MagicMock, tmp_path: Path
    ) -> None:
        """Should surface a ProcessingError for encrypted PDFs."""
        pdf_path = _create_pdf_path(tmp_path)
        document = _mock_document(["Secret"], is_encrypted=True)
        _attach_context_manager(mock_open, document)

        with pytest.raises(ProcessingError) as exc_info:
            self.processor.process_pdf(str(pdf_path))

        assert "encrypted" in exc_info.value.message.lower()
        mock_open.assert_called_once()

    def test_process_pdf_file_not_found_raises_validation_error(self) -> None:
        """Should ensure the file exists before attempting to open."""
        missing = Path("/tmp/non-existent.pdf")
        processor = PDFProcessor()

        with pytest.raises(ValidationError) as exc_info:
            processor.process_pdf(str(missing))

        assert "exist" in exc_info.value.message.lower()

    @patch("zapomni_core.processors.pdf_processor.fitz.open")
    def test_process_pdf_corrupted_raises_processing_error(
        self, mock_open: MagicMock, tmp_path: Path
    ) -> None:
        """Should wrap low-level errors while opening the document."""
        pdf_path = _create_pdf_path(tmp_path)
        mock_open.side_effect = RuntimeError("file is corrupt")

        with pytest.raises(ProcessingError) as exc_info:
            self.processor.process_pdf(str(pdf_path))

        assert "corrupt" in exc_info.value.message.lower()

    @patch("zapomni_core.processors.pdf_processor.fitz.open")
    def test_process_pdf_empty_document_returns_empty_text(
        self, mock_open: MagicMock, tmp_path: Path
    ) -> None:
        """Should handle zero-page PDFs without errors."""
        pdf_path = _create_pdf_path(tmp_path)
        document = _mock_document([])
        _attach_context_manager(mock_open, document)

        result = self.processor.process_pdf_with_metadata(str(pdf_path))

        assert result["text"] == ""
        assert result["metadata"]["pages"] == 0
