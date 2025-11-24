"""
ProcessorFactory - Automatically selects the correct processor based on file type.

Supports PDF, DOCX, Markdown, Python code, HTML, and plain text documents.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from zapomni_core import code_processor
from zapomni_core.exceptions import ValidationError
from zapomni_core.processors.docx_processor import DOCXProcessor
from zapomni_core.processors.html_processor import HTMLProcessor
from zapomni_core.processors.markdown_processor import MarkdownProcessor
from zapomni_core.processors.pdf_processor import PDFProcessor
from zapomni_core.processors.text_processor import TextProcessor
from zapomni_core.utils import get_logger


class ProcessorFactory:
    """
    Factory for creating appropriate document processors based on file type.

    The factory inspects the file extension and returns an appropriate processor
    instance. For Python source files it returns the ``code_processor`` module
    itself (because it exposes functions instead of a class). For unknown
    extensions it falls back to :class:`TextProcessor`.

    Typical usage::

        factory = ProcessorFactory()
        processor = factory.get_processor("document.pdf")
        text = processor.process_pdf("document.pdf")
    """

    def __init__(self) -> None:
        """Initialize the factory and configure a structured logger."""
        self.logger = get_logger(__name__)

    def detect_file_type(self, file_path: str) -> str:
        """
        Detect file type from the given path.

        Args:
            file_path: Path to the file whose type should be detected.

        Returns:
            The file extension in lowercase, including the leading dot
            (for example ``".pdf"``). If the path has no extension,
            an empty string is returned.

        Raises:
            ValidationError: If ``file_path`` is ``None`` or an empty string.
        """
        # Basic validation: reject None / empty / whitespace-only values
        if file_path is None or (isinstance(file_path, str) and file_path.strip() == ""):
            raise ValidationError(
                message="File path cannot be empty",
                error_code="VAL_001",
                details={"file_path": file_path},
            )

        # Normalise to string in case a Path-like object is provided
        file_path_str = str(file_path)
        path = Path(file_path_str)

        extension = path.suffix.lower()
        self.logger.debug(
            "file_type_detected",
            file_path=file_path_str,
            extension=extension,
        )
        return extension

    def get_processor(self, file_path: str) -> Any:
        """
        Return an appropriate processor for the provided file path.

        The decision is based solely on the file extension. The mapping is:

        - ``.pdf``           → :class:`PDFProcessor`
        - ``.docx`` / ``.doc`` → :class:`DOCXProcessor`
        - ``.md`` / ``.markdown`` → :class:`MarkdownProcessor`
        - ``.py``            → :mod:`zapomni_core.code_processor` module
        - ``.html`` / ``.htm`` → :class:`HTMLProcessor`
        - anything else (including ``.txt`` and files without extension)
          → :class:`TextProcessor`

        Args:
            file_path: Path to the file to be processed.

        Returns:
            An instance of the selected processor, or the
            :mod:`zapomni_core.code_processor` module for Python files.

        Raises:
            ValidationError: If ``file_path`` is ``None`` or an empty string.
        """
        # This call performs validation and logs the detected extension.
        extension = self.detect_file_type(file_path)
        file_path_str = str(file_path)

        self.logger.debug(
            "processor_selection_started",
            file_path=file_path_str,
            extension=extension,
        )

        # Dispatch based on extension
        if extension == ".pdf":
            processor: Any = PDFProcessor()
        elif extension in (".docx", ".doc"):
            processor = DOCXProcessor()
        elif extension in (".md", ".markdown"):
            processor = MarkdownProcessor()
        elif extension == ".py":
            # Special case: return the module so callers can use its functions.
            processor = code_processor
        elif extension in (".html", ".htm"):
            processor = HTMLProcessor()
        else:
            # Fallback for .txt and any unknown extension.
            processor = TextProcessor()

        processor_name = getattr(processor, "__name__", processor.__class__.__name__)
        self.logger.info(
            "processor_selected",
            file_path=file_path_str,
            extension=extension,
            processor_type=processor_name,
        )

        return processor
