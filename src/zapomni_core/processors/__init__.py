"""
Processors package - Text processing pipeline orchestration.

Provides high-level text processing components that coordinate chunking,
embedding, and storage operations.

Main Components:
    TextProcessor: Orchestrates text → chunks → embeddings → storage pipeline

Example:
    ```python
    from zapomni_core.processors import TextProcessor

    processor = TextProcessor()
    memory_id = await processor.add_text(
        text="Python is a programming language.",
        metadata={"source": "test"}
    )
    ```
"""

from zapomni_core.processors.text_processor import TextProcessor

__all__ = ["TextProcessor"]
