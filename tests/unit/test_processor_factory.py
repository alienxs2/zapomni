"""
Unit tests for the ProcessorFactory component.

Tests automatic processor selection based on file type detection.
"""

from __future__ import annotations

import pytest

from zapomni_core import code_processor
from zapomni_core.exceptions import ValidationError
from zapomni_core.processors import ProcessorFactory
from zapomni_core.processors.docx_processor import DOCXProcessor
from zapomni_core.processors.html_processor import HTMLProcessor
from zapomni_core.processors.markdown_processor import MarkdownProcessor
from zapomni_core.processors.pdf_processor import PDFProcessor
from zapomni_core.processors.text_processor import TextProcessor


class TestProcessorFactory:
    """Test suite for ProcessorFactory file type detection and routing."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.factory = ProcessorFactory()

    def test_get_processor_returns_pdf_processor_for_pdf_files(self) -> None:
        """Should return PDFProcessor instance for .pdf files."""
        processor = self.factory.get_processor("document.pdf")
        assert isinstance(processor, PDFProcessor)

    def test_get_processor_returns_docx_processor_for_docx_files(self) -> None:
        """Should return DOCXProcessor instance for .docx files."""
        processor = self.factory.get_processor("report.docx")
        assert isinstance(processor, DOCXProcessor)

    def test_get_processor_returns_docx_processor_for_doc_files(self) -> None:
        """Should return DOCXProcessor instance for .doc files."""
        processor = self.factory.get_processor("legacy.doc")
        assert isinstance(processor, DOCXProcessor)

    def test_get_processor_returns_markdown_processor_for_md_files(self) -> None:
        """Should return MarkdownProcessor instance for .md files."""
        processor = self.factory.get_processor("readme.md")
        assert isinstance(processor, MarkdownProcessor)

    def test_get_processor_returns_markdown_processor_for_markdown_extension(
        self,
    ) -> None:
        """Should return MarkdownProcessor instance for .markdown files."""
        processor = self.factory.get_processor("documentation.markdown")
        assert isinstance(processor, MarkdownProcessor)

    def test_get_processor_returns_code_processor_module_for_py_files(self) -> None:
        """Should return code_processor module for .py files."""
        processor = self.factory.get_processor("script.py")
        # code_processor is a module, not a class instance
        assert processor == code_processor

    def test_get_processor_returns_html_processor_for_html_files(self) -> None:
        """Should return HTMLProcessor instance for .html files."""
        processor = self.factory.get_processor("page.html")
        assert isinstance(processor, HTMLProcessor)

    def test_get_processor_returns_html_processor_for_htm_files(self) -> None:
        """Should return HTMLProcessor instance for .htm files."""
        processor = self.factory.get_processor("index.htm")
        assert isinstance(processor, HTMLProcessor)

    def test_get_processor_returns_text_processor_for_txt_files(self) -> None:
        """Should return TextProcessor instance for .txt files."""
        processor = self.factory.get_processor("notes.txt")
        assert isinstance(processor, TextProcessor)

    def test_get_processor_returns_text_processor_for_unknown_types(self) -> None:
        """Should fall back to TextProcessor for unknown file types."""
        processor = self.factory.get_processor("data.xyz")
        assert isinstance(processor, TextProcessor)

    def test_get_processor_handles_uppercase_extensions(self) -> None:
        """Should handle uppercase file extensions correctly."""
        processor = self.factory.get_processor("DOCUMENT.PDF")
        assert isinstance(processor, PDFProcessor)

    def test_get_processor_handles_mixed_case_extensions(self) -> None:
        """Should handle mixed-case file extensions correctly."""
        processor = self.factory.get_processor("Report.DocX")
        assert isinstance(processor, DOCXProcessor)

    def test_get_processor_raises_validation_error_for_empty_path(self) -> None:
        """Should raise ValidationError when file_path is empty."""
        with pytest.raises(ValidationError) as exc_info:
            self.factory.get_processor("")

        assert exc_info.value.error_code == "VAL_001"
        assert "empty" in exc_info.value.message.lower()

    def test_get_processor_raises_validation_error_for_none_path(self) -> None:
        """Should raise ValidationError when file_path is None."""
        with pytest.raises(ValidationError) as exc_info:
            self.factory.get_processor(None)  # type: ignore

        assert exc_info.value.error_code == "VAL_001"

    def test_detect_file_type_extracts_extension_correctly(self) -> None:
        """Should extract file extension in lowercase with dot."""
        extension = self.factory.detect_file_type("document.pdf")
        assert extension == ".pdf"

    def test_detect_file_type_handles_uppercase(self) -> None:
        """Should convert uppercase extensions to lowercase."""
        extension = self.factory.detect_file_type("FILE.PDF")
        assert extension == ".pdf"

    def test_detect_file_type_handles_no_extension(self) -> None:
        """Should return empty string for files without extension."""
        extension = self.factory.detect_file_type("README")
        assert extension == ""

    def test_detect_file_type_handles_path_with_directories(self) -> None:
        """Should extract extension from full file path."""
        extension = self.factory.detect_file_type("/path/to/document.pdf")
        assert extension == ".pdf"

    def test_detect_file_type_raises_validation_error_for_empty_path(self) -> None:
        """Should raise ValidationError for empty file path."""
        with pytest.raises(ValidationError) as exc_info:
            self.factory.detect_file_type("")

        assert exc_info.value.error_code == "VAL_001"
