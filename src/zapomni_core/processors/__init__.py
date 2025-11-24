"""
Processors package - Text processing pipeline orchestration.

Provides high-level text processing components that coordinate chunking,
embedding, and storage operations.

Main Components:
    TextProcessor: Orchestrates text → chunks → embeddings → storage pipeline
    HTMLProcessor: Extracts clean text from HTML using trafilatura

Example:
    ```python
    from zapomni_core.processors import TextProcessor, HTMLProcessor

    # Process plain text
    processor = TextProcessor()
    memory_id = await processor.add_text(
        text="Python is a programming language.",
        metadata={"source": "test"}
    )

    # Extract from HTML
    html_processor = HTMLProcessor()
    text = html_processor.extract_from_html(html)
    result = html_processor.extract_with_metadata(html)
    ```
"""

from zapomni_core.processors.text_processor import TextProcessor
from zapomni_core.processors.html_processor import HTMLProcessor
from zapomni_core.processors.pdf_processor import PDFProcessor
from zapomni_core.processors.markdown_processor import MarkdownProcessor
from zapomni_core.processors.code_processor import (
    extract_classes,
    extract_functions,
    extract_imports,
    parse_python,
    validate_syntax,
)
from zapomni_core.processors.docx_processor import DOCXProcessor
from zapomni_core.processors.processor_factory import ProcessorFactory

__all__ = [
    "TextProcessor",
    "HTMLProcessor",
    "PDFProcessor",
    "MarkdownProcessor",
    "DOCXProcessor",
    "ProcessorFactory",
    "extract_classes",
    "extract_functions",
    "extract_imports",
    "parse_python",
    "validate_syntax",
]
