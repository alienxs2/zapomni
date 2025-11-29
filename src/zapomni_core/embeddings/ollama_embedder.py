"""
OllamaEmbedder - Local embedding generation via Ollama API.

Provides privacy-first embedding generation using local Ollama with intelligent
fallback to sentence-transformers. Includes retry logic, batch processing,
and comprehensive error handling.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

import asyncio
import math
from typing import List, Optional
from urllib.parse import urlparse

import httpx
import structlog
from sentence_transformers import SentenceTransformer

from zapomni_core.exceptions import EmbeddingError, TimeoutError, ValidationError

logger = structlog.get_logger()


class OllamaEmbedder:
    """
    Local embedding generation via Ollama API with intelligent fallback.

    Uses Ollama's nomic-embed-text model (768 dimensions, 81.2% MTEB accuracy)
    for privacy-preserving local embedding generation. Falls back to
    sentence-transformers (all-MiniLM-L6-v2) if Ollama is unavailable.

    Attributes:
        base_url: Ollama API URL (e.g., "http://localhost:11434")
        model_name: Ollama embedding model (default: "nomic-embed-text")
        dimensions: Expected embedding dimensions (768 for nomic-embed-text)
        timeout: Request timeout in seconds (default: 30)
        max_retries: Maximum retry attempts for transient failures (default: 3)
        client: Async HTTP client for Ollama API calls
        fallback_model: sentence-transformers model (lazy loaded)

    Example:
        ```python
        embedder = OllamaEmbedder(
            base_url="http://localhost:11434",
            model_name="nomic-embed-text"
        )

        # Single text embedding
        embedding = await embedder.embed_text("Python is great")
        print(f"Dimensions: {len(embedding)}")  # 768

        # Batch embedding (efficient)
        texts = ["Python is great", "Ollama is local", "Privacy matters"]
        embeddings = await embedder.embed_batch(texts)
        print(f"Generated {len(embeddings)} embeddings")
        ```
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "nomic-embed-text",
        timeout: int = 30,
        max_retries: int = 3,
        enable_fallback: bool = True,
    ) -> None:
        """
        Initialize OllamaEmbedder with connection parameters.

        Args:
            base_url: Ollama API URL (default: http://localhost:11434)
            model_name: Ollama embedding model name (default: nomic-embed-text)
            timeout: Request timeout in seconds (recommended: 10-60)
            max_retries: Max retry attempts for transient failures (recommended: 3-5)
            enable_fallback: Enable sentence-transformers fallback (default: True)

        Raises:
            ValidationError: If base_url is invalid or timeout <= 0

        Example:
            ```python
            # Standard configuration (Phase 1)
            embedder = OllamaEmbedder()

            # Custom Ollama host
            embedder = OllamaEmbedder(base_url="http://192.168.1.100:11434")

            # Disable fallback (fail fast)
            embedder = OllamaEmbedder(enable_fallback=False)
            ```
        """
        # Validate base_url
        try:
            parsed = urlparse(base_url)
            if not parsed.scheme or not parsed.netloc:
                raise ValidationError(
                    message=f"Invalid base_url format: {base_url}",
                    error_code="VAL_004",
                    details={"base_url": base_url},
                )
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(
                message=f"Invalid base_url: {base_url}",
                error_code="VAL_004",
                details={"base_url": base_url, "error": str(e)},
            )

        # Validate timeout
        if timeout <= 0:
            raise ValidationError(
                message=f"timeout must be positive, got {timeout}",
                error_code="VAL_003",
                details={"timeout": timeout},
            )

        self.base_url = base_url
        self.model_name = model_name
        self.dimensions = 768  # nomic-embed-text dimensions
        self.timeout = timeout
        self.max_retries = max_retries
        self.enable_fallback = enable_fallback

        # Initialize HTTP client with connection pooling
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
        )

        # Lazy-loaded fallback model
        self.fallback_model: Optional[SentenceTransformer] = None

        logger.info(
            "ollama_embedder_initialized",
            base_url=base_url,
            model=model_name,
            timeout=timeout,
            fallback_enabled=enable_fallback,
        )

    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Algorithm:
        1. Validate input (non-empty, UTF-8)
        2. Call Ollama API with retry logic
        3. If Ollama fails after retries, fallback to sentence-transformers
        4. Validate embedding dimensions (must be 768)
        5. Return embedding vector

        Args:
            text: Input text to embed (max 8192 tokens for nomic-embed-text)

        Returns:
            List[float]: 768-dimensional embedding vector

        Raises:
            ValidationError: If text is empty or exceeds max length
            EmbeddingError: If both Ollama and fallback fail
            TimeoutError: If request exceeds timeout

        Performance Target:
            - Single embedding: < 150ms (P95)
            - Includes retry overhead if needed

        Example:
            ```python
            embedder = OllamaEmbedder()

            text = "Python is a programming language"
            embedding = await embedder.embed_text(text)

            assert len(embedding) == 768
            assert all(isinstance(x, float) for x in embedding)
            print(f"First 5 dims: {embedding[:5]}")
            ```
        """
        # Validate input
        if not text or not text.strip():
            raise ValidationError(
                message="Text cannot be empty",
                error_code="VAL_001",
                details={"text_length": len(text)},
            )

        # Check text length (rough token estimate: 1 token ≈ 4 chars)
        # nomic-embed-text supports up to 8192 tokens
        estimated_tokens = len(text) // 4
        if estimated_tokens > 8192:
            raise ValidationError(
                message=f"Text exceeds max length (8192 tokens, got ~{estimated_tokens})",
                error_code="VAL_003",
                details={"estimated_tokens": estimated_tokens, "max_tokens": 8192},
            )

        # Try Ollama API with retries
        try:
            embedding = await self._call_ollama(text)
            self._validate_embedding(embedding)
            logger.debug(
                "embedding_generated",
                source="ollama",
                text_length=len(text),
                dimensions=len(embedding),
            )
            return embedding

        except Exception as e:
            logger.warning(
                "ollama_embedding_failed",
                error=str(e),
                fallback_enabled=self.enable_fallback,
            )

            # Try fallback if enabled
            if self.enable_fallback:
                try:
                    embedding = await self._fallback_embed(text)
                    self._validate_embedding(embedding)
                    logger.info(
                        "embedding_generated",
                        source="fallback",
                        text_length=len(text),
                        dimensions=len(embedding),
                    )
                    return embedding
                except Exception as fallback_error:
                    raise EmbeddingError(
                        message=f"Both Ollama and fallback failed: {str(e)}, {str(fallback_error)}",
                        error_code="EMB_001",
                        details={
                            "ollama_error": str(e),
                            "fallback_error": str(fallback_error),
                        },
                        original_exception=e,
                    )
            else:
                raise EmbeddingError(
                    message=f"Failed to generate embedding: {str(e)}",
                    error_code="EMB_001",
                    details={"error": str(e)},
                    original_exception=e,
                )

    async def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently using Ollama batch API.

        Uses Ollama's /api/embed endpoint which accepts multiple texts in a single
        HTTP request, significantly reducing latency compared to individual calls.

        Algorithm:
        1. Validate inputs (all non-empty, batch_size valid)
        2. Split texts into batches of size batch_size
        3. For each batch: call /api/embed with all texts
        4. Flatten batches into single list
        5. Return embeddings

        Args:
            texts: List of input texts (max 1000 texts per call recommended)
            batch_size: Texts per batch request (default: 32, max recommended: 64)

        Returns:
            List[List[float]]: List of 768-dimensional embeddings (same order as inputs)

        Raises:
            ValidationError: If any text is empty or batch_size invalid
            EmbeddingError: If batch embedding fails

        Performance Target:
            - 32 texts: < 500ms (P95) with batch API
            - Throughput: ~500+ texts/sec with batch_size=32

        Example:
            ```python
            embedder = OllamaEmbedder()

            texts = [
                "Python is great",
                "Ollama runs locally",
                "Privacy is important",
                "Semantic search is powerful"
            ]

            embeddings = await embedder.embed_batch(texts, batch_size=32)

            assert len(embeddings) == len(texts)
            assert all(len(emb) == 768 for emb in embeddings)
            print(f"Generated {len(embeddings)} embeddings")
            ```
        """
        # Validate inputs
        if not texts:
            raise ValidationError(
                message="texts list cannot be empty",
                error_code="VAL_001",
                details={"texts_count": 0},
            )

        if batch_size < 1 or batch_size > 64:
            raise ValidationError(
                message=f"batch_size must be between 1 and 64, got {batch_size}",
                error_code="VAL_003",
                details={"batch_size": batch_size},
            )

        # Check all texts are non-empty (basic check)
        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise ValidationError(
                    message=f"Text at index {i} is empty",
                    error_code="VAL_001",
                    details={"index": i},
                )

        # Process texts in batches using batch API
        embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            try:
                # Use batch API for efficiency
                batch_embeddings = await self._call_ollama_batch(batch)
                embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.warning(
                    "batch_api_failed_fallback_to_individual",
                    batch_index=i,
                    batch_size=len(batch),
                    error=str(e),
                )
                # Fallback to individual calls if batch API fails
                for j, text in enumerate(batch):
                    try:
                        embedding = await self.embed_text(text)
                        embeddings.append(embedding)
                    except Exception as inner_e:
                        raise EmbeddingError(
                            message=f"Failed to embed text at index {i + j}: {str(inner_e)}",
                            error_code="EMB_001",
                            details={"index": i + j, "error": str(inner_e)},
                            original_exception=inner_e,
                        )

        logger.info(
            "batch_embeddings_generated",
            total=len(texts),
            batch_size=batch_size,
            api="batch",
        )

        return embeddings

    async def _call_ollama_batch(self, texts: List[str], retry_count: int = 0) -> List[List[float]]:
        """
        Internal method: Call Ollama batch API /api/embed with retry logic.

        Uses the newer /api/embed endpoint which accepts multiple texts in one request.
        This is significantly faster than making individual /api/embeddings calls.

        Args:
            texts: List of texts to embed (max ~100 recommended per call)
            retry_count: Current retry attempt (internal)

        Returns:
            List[List[float]]: List of 768-dimensional embeddings

        Raises:
            EmbeddingError: If all retries exhausted or API returns error
            TimeoutError: If request exceeds timeout

        Private method, not exposed in public API.
        """
        try:
            response = await self.client.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model_name, "input": texts},
            )

            if response.status_code == 200:
                data = response.json()
                if "embeddings" not in data:
                    raise EmbeddingError(
                        message="Invalid Ollama batch response: missing 'embeddings' field",
                        error_code="EMB_001",
                        details={"response_keys": list(data.keys())},
                    )

                embeddings = data["embeddings"]

                # Validate all embeddings
                for i, emb in enumerate(embeddings):
                    if len(emb) != self.dimensions:
                        raise EmbeddingError(
                            message=f"Invalid embedding dimensions at index {i}: "
                            f"expected {self.dimensions}, got {len(emb)}",
                            error_code="EMB_003",
                            details={"index": i, "expected": self.dimensions, "got": len(emb)},
                        )

                return embeddings

            elif response.status_code == 404:
                # Model not found or batch API not supported - will fallback
                raise EmbeddingError(
                    message=f"Model '{self.model_name}' not found or batch API not supported. "
                    f"Run: ollama pull {self.model_name}",
                    error_code="EMB_004",
                    details={"model": self.model_name, "status_code": 404},
                )

            else:
                raise EmbeddingError(
                    message=f"Ollama batch API error: {response.status_code}",
                    error_code="EMB_001",
                    details={"status_code": response.status_code, "response": response.text[:500]},
                )

        except httpx.TimeoutException as e:
            if retry_count < self.max_retries:
                wait_time = 2**retry_count
                logger.warning(
                    "ollama_batch_timeout_retry",
                    retry=retry_count + 1,
                    max_retries=self.max_retries,
                    wait_time=wait_time,
                    batch_size=len(texts),
                )
                await asyncio.sleep(wait_time)
                return await self._call_ollama_batch(texts, retry_count + 1)
            else:
                raise TimeoutError(
                    message=f"Ollama batch request timed out after {self.max_retries} retries",
                    error_code="TIMEOUT_002",
                    details={"retries": retry_count, "batch_size": len(texts)},
                    original_exception=e,
                )

        except (httpx.ConnectError, httpx.NetworkError) as e:
            if retry_count < self.max_retries:
                wait_time = 2**retry_count
                logger.warning(
                    "ollama_batch_connection_retry",
                    retry=retry_count + 1,
                    max_retries=self.max_retries,
                    wait_time=wait_time,
                )
                await asyncio.sleep(wait_time)
                return await self._call_ollama_batch(texts, retry_count + 1)
            else:
                raise EmbeddingError(
                    message=f"Ollama batch connection failed after {self.max_retries} retries: "
                    f"{str(e)}",
                    error_code="EMB_001",
                    details={"retries": retry_count, "error": str(e)},
                    original_exception=e,
                )

        except EmbeddingError:
            raise

        except Exception as e:
            raise EmbeddingError(
                message=f"Unexpected error calling Ollama batch API: {str(e)}",
                error_code="EMB_001",
                details={"error": str(e), "batch_size": len(texts)},
                original_exception=e,
            )

    def get_dimensions(self) -> int:
        """
        Return embedding dimensions for this model.

        Returns:
            int: Embedding dimensions (768 for nomic-embed-text)

        Example:
            ```python
            embedder = OllamaEmbedder()
            dims = embedder.get_dimensions()
            assert dims == 768
            ```
        """
        return self.dimensions

    async def health_check(self) -> bool:
        """
        Check if Ollama API is reachable and model is available.

        Performs a lightweight test request to Ollama API to verify:
        - API is reachable
        - Model is downloaded and ready
        - Embeddings can be generated

        Returns:
            bool: True if Ollama healthy, False otherwise

        Performance Target:
            - Execution time: < 200ms

        Example:
            ```python
            embedder = OllamaEmbedder()

            if await embedder.health_check():
                print("Ollama is ready")
            else:
                print("Ollama unavailable, will use fallback")
            ```
        """
        try:
            # Try to generate a simple embedding
            test_text = "health check"
            response = await self.client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model_name, "prompt": test_text},
                timeout=5.0,  # Short timeout for health check
            )

            if response.status_code == 200:
                data = response.json()
                if "embedding" in data and len(data["embedding"]) == self.dimensions:
                    logger.debug("ollama_health_check_passed")
                    return True

            logger.debug("ollama_health_check_failed", status_code=response.status_code)
            return False

        except Exception as e:
            logger.debug("ollama_health_check_failed", error=str(e))
            return False

    async def _call_ollama(self, text: str, retry_count: int = 0) -> List[float]:
        """
        Internal method: Call Ollama API with retry logic.

        Implements exponential backoff retry strategy:
        - Attempt 1: immediate
        - Attempt 2: wait 1s
        - Attempt 3: wait 2s
        - Attempt 4: wait 4s

        Args:
            text: Text to embed
            retry_count: Current retry attempt (internal)

        Returns:
            List[float]: 768-dimensional embedding

        Raises:
            EmbeddingError: If all retries exhausted
            TimeoutError: If request exceeds timeout

        Private method, not exposed in public API.
        """
        try:
            response = await self.client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model_name, "prompt": text},
            )

            if response.status_code == 200:
                data = response.json()
                if "embedding" not in data:
                    raise EmbeddingError(
                        message="Invalid Ollama response: missing 'embedding' field",
                        error_code="EMB_001",
                        details={"response": data},
                    )
                return data["embedding"]

            elif response.status_code == 404:
                raise EmbeddingError(
                    message=f"Model '{self.model_name}' not found. "
                    f"Run: ollama pull {self.model_name}",
                    error_code="EMB_004",
                    details={"model": self.model_name},
                )

            else:
                raise EmbeddingError(
                    message=f"Ollama API error: {response.status_code}",
                    error_code="EMB_001",
                    details={"status_code": response.status_code, "response": response.text},
                )

        except httpx.TimeoutException as e:
            if retry_count < self.max_retries:
                wait_time = 2**retry_count  # Exponential backoff: 1s, 2s, 4s
                logger.warning(
                    "ollama_timeout_retry",
                    retry=retry_count + 1,
                    max_retries=self.max_retries,
                    wait_time=wait_time,
                )
                await asyncio.sleep(wait_time)
                return await self._call_ollama(text, retry_count + 1)
            else:
                raise TimeoutError(
                    message=f"Ollama request timed out after {self.max_retries} retries",
                    error_code="TIMEOUT_002",
                    details={"retries": retry_count},
                    original_exception=e,
                )

        except (httpx.ConnectError, httpx.NetworkError) as e:
            if retry_count < self.max_retries:
                wait_time = 2**retry_count
                logger.warning(
                    "ollama_connection_retry",
                    retry=retry_count + 1,
                    max_retries=self.max_retries,
                    wait_time=wait_time,
                )
                await asyncio.sleep(wait_time)
                return await self._call_ollama(text, retry_count + 1)
            else:
                raise EmbeddingError(
                    message=f"Ollama connection failed after {self.max_retries} retries: {str(e)}",
                    error_code="EMB_001",
                    details={"retries": retry_count, "error": str(e)},
                    original_exception=e,
                )

        except EmbeddingError:
            # Re-raise EmbeddingError as-is
            raise

        except Exception as e:
            raise EmbeddingError(
                message=f"Unexpected error calling Ollama: {str(e)}",
                error_code="EMB_001",
                details={"error": str(e)},
                original_exception=e,
            )

    async def _fallback_embed(self, text: str) -> List[float]:
        """
        Internal method: Generate embedding using sentence-transformers fallback.

        Uses all-MiniLM-L6-v2 model (384 dimensions). Pads with zeros to match
        768 dimensions for compatibility with FalkorDB vector index.

        Args:
            text: Text to embed

        Returns:
            List[float]: 768-dimensional embedding (384 real + 384 zeros)

        Private method, not exposed in public API.
        """
        # Lazy load fallback model
        if self.fallback_model is None:
            logger.info("loading_fallback_model", model="all-MiniLM-L6-v2")
            self.fallback_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Generate embedding (returns numpy array)
        embedding_array = self.fallback_model.encode(text)

        # Convert to list
        embedding = embedding_array.tolist()

        # Pad to 768 dimensions (all-MiniLM-L6-v2 is 384-dim)
        if len(embedding) < self.dimensions:
            padding = [0.0] * (self.dimensions - len(embedding))
            embedding.extend(padding)

        return embedding

    def _validate_embedding(self, embedding: List[float]) -> None:
        """
        Internal method: Validate embedding dimensions and values.

        Checks:
        - Embedding is list of floats
        - Length is exactly 768
        - No NaN or Inf values

        Args:
            embedding: Embedding to validate

        Raises:
            EmbeddingError: If validation fails

        Private method, not exposed in public API.
        """
        # Check type
        if not isinstance(embedding, list):
            raise EmbeddingError(
                message=f"Embedding must be list, got {type(embedding)}",
                error_code="EMB_003",
                details={"type": str(type(embedding))},
            )

        # Check dimensions
        if len(embedding) != self.dimensions:
            raise EmbeddingError(
                message=f"Invalid embedding dimensions: expected {self.dimensions}, "
                f"got {len(embedding)}",
                error_code="EMB_003",
                details={"expected": self.dimensions, "got": len(embedding)},
            )

        # Check for NaN or Inf
        for i, val in enumerate(embedding):
            if not isinstance(val, (int, float)):
                raise EmbeddingError(
                    message=f"Embedding value at index {i} is not a number: {type(val)}",
                    error_code="EMB_003",
                    details={"index": i, "type": str(type(val))},
                )
            if math.isnan(val) or math.isinf(val):
                raise EmbeddingError(
                    message=f"Embedding contains NaN or Inf at index {i}",
                    error_code="EMB_003",
                    details={"index": i, "value": str(val)},
                )

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup client."""
        await self.client.aclose()
