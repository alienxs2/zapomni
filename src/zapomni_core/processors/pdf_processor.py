"""
PDFProcessor - Extracts text and metadata from PDF files.

Handles validation, encrypted documents, corrupted streams, and per-page logging.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import fitz  # type: ignore[import-untyped]

from zapomni_core.exceptions import ProcessingError, ValidationError
from zapomni_core.utils import get_logger

Metadata = Dict[str, Any]


class PDFProcessor:
    """
    Extracts text from PDF documents using PyMuPDF.

    Supports streaming page-by-page extraction, metadata enrichment, and
    error signals for invalid or corrupted files.
    """

    def __init__(self) -> None:
        self.logger = get_logger(__name__)

    def process_pdf(self, file_path: str) -> str:
        """
        Extract cleaned text from the specified PDF.
        """
        text, _ = self._process(file_path)
        return text

    def process_pdf_with_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract cleaned text and metadata from the specified PDF.
        """
        text, metadata = self._process(file_path)
        return {"text": text, "metadata": metadata}

    def _process(self, file_path: str) -> Tuple[str, Metadata]:
        path = Path(file_path)
        self.logger.debug("pdf_validation_started", file=str(path))
        self._validate_path(path)

        try:
            with fitz.open(str(path)) as document:
                self.logger.info(
                    "pdf_opened",
                    file=str(path),
                    page_count=document.page_count,
                )

                self._ensure_not_encrypted(document, path)

                text = self._extract_text(document, path)
                metadata = self._extract_metadata(document)

                return text, metadata
        except ProcessingError:
            raise
        except Exception as exc:
            message = str(exc)
            if "password" in message.lower() or "encrypted" in message.lower():
                raise ProcessingError(
                    message="Encrypted PDF requires a password",
                    details={"file_path": str(path)},
                    original_exception=exc,
                )

            self.logger.error(
                "pdf_processing_failed",
                file=str(path),
                error=message,
            )
            raise ProcessingError(
                message=f"Failed to read PDF: {message}",
                details={"file_path": str(path)},
                original_exception=exc,
            )

    def _validate_path(self, path: Path) -> None:
        if not path.exists():
            raise ValidationError(
                message=f"PDF file does not exist: {path}",
                error_code="VAL_004",
                details={"file_path": str(path)},
            )

        if path.suffix.lower() != ".pdf":
            raise ValidationError(
                message="File must have .pdf extension",
                error_code="VAL_003",
                details={"file_path": str(path)},
            )

    def _ensure_not_encrypted(self, document: fitz.Document, path: Path) -> None:
        if getattr(document, "is_encrypted", False) or getattr(document, "needs_password", False):
            raise ProcessingError(
                message="Encrypted PDF requires a password",
                error_code="PROC_002",
                details={"file_path": str(path)},
            )

    def _extract_text(self, document: fitz.Document, path: Path) -> str:
        page_count = document.page_count
        extracted_texts: List[str] = []

        for index in range(page_count):
            self.logger.info(
                "pdf_page_extraction_started",
                page_number=index + 1,
                total_pages=page_count,
                file=str(path),
            )

            page = document.load_page(index)
            raw_text = page.get_text("text") or ""
            cleaned = self._clean_text(raw_text)

            if cleaned:
                extracted_texts.append(cleaned)

        return "\n".join(extracted_texts).strip()

    def _clean_text(self, text: str) -> str:
        normalized_lines: List[str] = []

        for line in text.splitlines():
            stripped = line.strip()

            if not stripped:
                continue

            normalized = " ".join(stripped.split())
            if normalized:
                normalized_lines.append(normalized)

        return "\n".join(normalized_lines)

    def _extract_metadata(self, document: fitz.Document) -> Metadata:
        raw_metadata = getattr(document, "metadata", {}) or {}
        title = raw_metadata.get("title") or raw_metadata.get("Title")
        author = raw_metadata.get("author") or raw_metadata.get("Author")
        date = (
            raw_metadata.get("creationDate")
            or raw_metadata.get("CreationDate")
            or raw_metadata.get("modDate")
            or raw_metadata.get("ModDate")
        )

        return {
            "title": title,
            "author": author,
            "date": date,
            "pages": document.page_count,
        }
