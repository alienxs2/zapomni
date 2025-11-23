"""
TextProcessor - Orchestrates text processing pipeline.

Coordinates SemanticChunker, OllamaEmbedder, and FalkorDBClient to transform
raw text into stored memories with semantic embeddings.

Pipeline: text → chunks → embeddings → storage → memory_id

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from typing import Dict, Any, List, Optional
import structlog

from zapomni_core.chunking.semantic_chunker import SemanticChunker
from zapomni_core.embeddings.ollama_embedder import OllamaEmbedder
from zapomni_db.falkordb_client import FalkorDBClient
from zapomni_core.exceptions import (
    ValidationError,
    ProcessingError,
    EmbeddingError,
    DatabaseError,
)
from zapomni_db.models import Memory, Chunk


class TextProcessor:
    """
    Orchestrates text processing pipeline (chunking → embedding → storage).

    Coordinates three main components:
    1. SemanticChunker: Splits text into semantic chunks
    2. OllamaEmbedder: Generates embeddings for chunks
    3. FalkorDBClient: Stores memories in graph database

    Example:
        ```python
        processor = TextProcessor()

        text = "Python is a high-level programming language..."
        metadata = {"source": "wikipedia", "author": "community"}

        memory_id = await processor.add_text(text, metadata)
        print(f"Stored memory: {memory_id}")
        ```
    """

    def __init__(
        self,
        chunker: Optional[SemanticChunker] = None,
        embedder: Optional[OllamaEmbedder] = None,
        db_client: Optional[FalkorDBClient] = None,
    ) -> None:
        """
        Initialize TextProcessor with dependencies.

        Args:
            chunker: SemanticChunker instance (default: new instance with defaults)
            embedder: OllamaEmbedder instance (default: new instance with defaults)
            db_client: FalkorDBClient instance (default: new instance with defaults)

        Example:
            ```python
            # Use defaults
            processor = TextProcessor()

            # Or provide custom instances
            custom_chunker = SemanticChunker(chunk_size=256)
            custom_embedder = OllamaEmbedder(model_name="custom-model")
            custom_db = FalkorDBClient(host="custom-host")

            processor = TextProcessor(
                chunker=custom_chunker,
                embedder=custom_embedder,
                db_client=custom_db
            )
            ```
        """
        # Initialize dependencies (lazy initialization pattern)
        self.chunker = chunker or SemanticChunker()
        self.embedder = embedder or OllamaEmbedder()
        self.db_client = db_client or FalkorDBClient()

        # Get logger
        self.logger = structlog.get_logger(__name__)

        self.logger.info(
            "text_processor_initialized",
            chunker_type=type(self.chunker).__name__,
            embedder_type=type(self.embedder).__name__,
            db_type=type(self.db_client).__name__,
        )

    async def add_text(
        self,
        text: str,
        metadata: Dict[str, Any]
    ) -> str:
        """
        Process text and store as memory in database.

        Pipeline stages:
        1. Validate input (text and metadata)
        2. Chunk text using SemanticChunker
        3. Generate embeddings using OllamaEmbedder
        4. Create Memory object
        5. Store in FalkorDB using FalkorDBClient
        6. Return memory_id

        Args:
            text: Input text to process (non-empty string)
            metadata: Metadata dict (source, author, tags, etc.)

        Returns:
            memory_id: UUID string identifying the stored memory

        Raises:
            ValidationError: If input validation fails
            ProcessingError: If chunking fails or pipeline validation fails
            EmbeddingError: If embedding generation fails
            DatabaseError: If storage fails

        Example:
            ```python
            processor = TextProcessor()

            text = \"\"\"
            Python is a high-level programming language.
            It was created by Guido van Rossum in 1991.
            Python emphasizes code readability.
            \"\"\"

            metadata = {
                "source": "wikipedia",
                "author": "community",
                "category": "programming",
                "tags": ["python", "programming", "language"]
            }

            try:
                memory_id = await processor.add_text(text, metadata)
                print(f"Success! Memory ID: {memory_id}")
            except ValidationError as e:
                print(f"Validation failed: {e}")
            except ProcessingError as e:
                print(f"Processing failed: {e}")
            except EmbeddingError as e:
                print(f"Embedding failed: {e}")
            except DatabaseError as e:
                print(f"Storage failed: {e}")
            ```
        """
        # STEP 1: INPUT VALIDATION
        self.logger.debug(
            "add_text_started",
            text_length=len(text) if isinstance(text, str) else 0,
            metadata_keys=list(metadata.keys()) if isinstance(metadata, dict) else []
        )

        # Validate text
        if not isinstance(text, str):
            raise ValidationError(
                message=f"text must be a string, got {type(text).__name__}",
                error_code="VAL_002",
                details={"expected_type": "str", "actual_type": type(text).__name__}
            )

        if not text or not text.strip():
            raise ValidationError(
                message="text cannot be empty or whitespace-only",
                error_code="VAL_001",
                details={"text_length": len(text)}
            )

        # Validate metadata
        if not isinstance(metadata, dict):
            raise ValidationError(
                message=f"metadata must be a dict, got {type(metadata).__name__}",
                error_code="VAL_002",
                details={"expected_type": "dict", "actual_type": type(metadata).__name__}
            )

        # STEP 2: CHUNK TEXT
        self.logger.info(
            "chunking_text",
            text_length=len(text)
        )

        try:
            chunks: List[Chunk] = self.chunker.chunk_text(text)
        except ProcessingError:
            # Re-raise ProcessingError as-is
            self.logger.error(
                "chunking_failed",
                text_length=len(text)
            )
            raise
        except Exception as e:
            # Wrap unexpected errors
            self.logger.error(
                "chunking_unexpected_error",
                error=str(e),
                error_type=type(e).__name__
            )
            raise ProcessingError(
                message=f"Unexpected error during chunking: {e}",
                error_code="PROC_001",
                original_exception=e
            )

        self.logger.info(
            "chunking_complete",
            num_chunks=len(chunks)
        )

        # STEP 3: GENERATE EMBEDDINGS
        self.logger.info(
            "generating_embeddings",
            num_chunks=len(chunks)
        )

        # Extract chunk texts for embedding
        chunk_texts = [chunk.text for chunk in chunks]

        try:
            embeddings: List[List[float]] = await self.embedder.embed_batch(chunk_texts)
        except EmbeddingError:
            # Re-raise EmbeddingError as-is
            self.logger.error(
                "embedding_failed",
                num_chunks=len(chunks)
            )
            raise
        except Exception as e:
            # Wrap unexpected errors
            self.logger.error(
                "embedding_unexpected_error",
                error=str(e),
                error_type=type(e).__name__
            )
            raise EmbeddingError(
                message=f"Unexpected error during embedding: {e}",
                error_code="EMB_001",
                original_exception=e
            )

        self.logger.info(
            "embedding_complete",
            num_embeddings=len(embeddings)
        )

        # STEP 4: VALIDATE PIPELINE RESULTS
        # Ensure chunks and embeddings count match
        if len(chunks) != len(embeddings):
            raise ProcessingError(
                message=f"Chunks/embeddings count mismatch: {len(chunks)} chunks, {len(embeddings)} embeddings",
                error_code="PROC_001",
                details={
                    "num_chunks": len(chunks),
                    "num_embeddings": len(embeddings)
                }
            )

        # Validate embedding dimensions (must be 768)
        for i, embedding in enumerate(embeddings):
            if len(embedding) != 768:
                raise ProcessingError(
                    message=f"Invalid embedding dimension at index {i}: expected 768, got {len(embedding)}",
                    error_code="PROC_001",
                    details={
                        "index": i,
                        "expected_dimension": 768,
                        "actual_dimension": len(embedding)
                    }
                )

        # STEP 5: CREATE MEMORY OBJECT
        memory = Memory(
            text=text,
            chunks=chunks,
            embeddings=embeddings,
            metadata=metadata
        )

        # STEP 6: STORE IN DATABASE
        self.logger.info(
            "storing_memory",
            num_chunks=len(chunks),
            metadata_keys=list(metadata.keys())
        )

        try:
            memory_id = await self.db_client.add_memory(memory)
        except DatabaseError:
            # Re-raise DatabaseError as-is
            self.logger.error(
                "storage_failed",
                num_chunks=len(chunks)
            )
            raise
        except Exception as e:
            # Wrap unexpected errors
            self.logger.error(
                "storage_unexpected_error",
                error=str(e),
                error_type=type(e).__name__
            )
            raise DatabaseError(
                message=f"Unexpected error during storage: {e}",
                error_code="DB_001",
                original_exception=e
            )

        # STEP 7: SUCCESS
        self.logger.info(
            "text_processed_successfully",
            memory_id=memory_id,
            num_chunks=len(chunks),
            text_length=len(text)
        )

        return memory_id
