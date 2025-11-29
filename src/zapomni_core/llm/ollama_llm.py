"""
OllamaLLMClient - Local LLM inference via Ollama API.

Provides LLM capabilities for:
- Entity refinement (enhancing SpaCy NER results)
- Relationship extraction (detecting connections between entities)

Uses Ollama's /api/generate endpoint for text generation.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx
import structlog

from zapomni_core.exceptions import ExtractionError, TimeoutError, ValidationError
from zapomni_core.runtime_config import RuntimeConfig

logger = structlog.get_logger()


# ============================================================================
# Prompts for Entity Refinement
# ============================================================================

ENTITY_REFINEMENT_PROMPT = """You are an expert entity extraction system. \
Given a text and a list of entities extracted by SpaCy NER, your task is to:

1. VALIDATE each entity - confirm it's a real entity, not noise
2. ENHANCE entity names - expand abbreviations, fix typos, use canonical names
3. ADD descriptions - brief 5-10 word description for each entity
4. ASSIGN confidence - score from 0.0 to 1.0 based on certainty

INPUT TEXT:
{text}

SPACY ENTITIES:
{entities_json}

Return ONLY valid JSON array with refined entities. Each entity must have:
- name: string (canonical/full name)
- type: string (PERSON, ORG, GPE, TECHNOLOGY, CONCEPT, PRODUCT, EVENT, DATE)
- description: string (brief description)
- confidence: number (0.0-1.0)

Example output:
[
  {{"name": "Guido van Rossum", "type": "PERSON", \
"description": "Creator of Python programming language", "confidence": 0.95}},
  {{"name": "Python", "type": "TECHNOLOGY", \
"description": "High-level programming language", "confidence": 0.98}}
]

IMPORTANT: Return ONLY the JSON array, no other text or explanation.

REFINED ENTITIES:"""


RELATIONSHIP_EXTRACTION_PROMPT = """You are an expert relationship extraction system. \
Given a text and a list of entities, identify relationships between them.

INPUT TEXT:
{text}

ENTITIES:
{entities_json}

Identify relationships between these entities. Use these relationship types:
- CREATED: Person/Org created something (Guido CREATED Python)
- WORKS_FOR: Person works for Organization
- LOCATED_IN: Entity is located in a place
- PART_OF: Entity is part of another
- USES: Entity uses another entity
- RELATED_TO: General relationship

Return ONLY valid JSON array with relationships. Each relationship must have:
- source: string (source entity name, must match an entity from the list)
- target: string (target entity name, must match an entity from the list)
- type: string (relationship type from the list above)
- confidence: number (0.0-1.0)
- evidence: string (short quote from text supporting this relationship)

Example output:
[
  {{"source": "Guido van Rossum", "target": "Python", "type": "CREATED", \
"confidence": 0.95, "evidence": "Guido created Python"}}
]

IMPORTANT:
- Return ONLY the JSON array, no other text
- Only include relationships where BOTH entities exist in the entity list
- If no relationships found, return empty array []

RELATIONSHIPS:"""


class OllamaLLMClient:
    """
    Local LLM client via Ollama API for entity refinement and relationship extraction.

    Uses Ollama's /api/generate endpoint for text generation with configurable
    model, timeout, and retry settings.

    Attributes:
        base_url: Ollama API URL (e.g., "http://localhost:11434")
        timeout: Request timeout in seconds (default: 120)
        max_retries: Maximum retry attempts (default: 2)
        temperature: Generation temperature (default: 0.1 for deterministic output)

    Note:
        Model selection is managed via RuntimeConfig singleton for hot-reload support.
        Use RuntimeConfig.get_instance().set_llm_model() to change the model at runtime.

    Example:
        ```python
        # Model is managed via RuntimeConfig (hot-reload support)
        from zapomni_core.runtime_config import RuntimeConfig

        # Optional: Set model explicitly (updates RuntimeConfig)
        client = OllamaLLMClient(
            base_url="http://localhost:11434",
            model_name="qwen2.5:latest"
        )

        # Or use default RuntimeConfig model
        client = OllamaLLMClient(base_url="http://localhost:11434")

        # Change model at runtime (affects all clients)
        RuntimeConfig.get_instance().set_llm_model("llama3:latest")

        # Refine entities
        refined = await client.refine_entities(text, spacy_entities)

        # Extract relationships
        relationships = await client.extract_relationships(text, entities)
        ```
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 2,
        temperature: float = 0.1,
        keep_alive: str = "24h",
    ) -> None:
        """
        Initialize OllamaLLMClient.

        Args:
            base_url: Ollama API URL (default: http://localhost:11434)
            model_name: Optional override for Ollama LLM model. If provided, updates RuntimeConfig.
                        If None, uses current RuntimeConfig value (default: None)
            timeout: Request timeout in seconds (default: 120)
            max_retries: Max retry attempts for transient failures (default: 2)
            temperature: Generation temperature, lower = more deterministic (default: 0.1)
            keep_alive: How long to keep model in memory (default: 24h).
                        Prevents model reload lag. Use "0" to unload immediately.

        Raises:
            ValidationError: If base_url is invalid or timeout <= 0
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

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self.keep_alive = keep_alive

        # Get RuntimeConfig instance
        self._runtime_config = RuntimeConfig.get_instance()

        # If model_name provided, update RuntimeConfig
        if model_name is not None:
            self._runtime_config.set_llm_model(model_name)

        # Initialize HTTP client
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )

        logger.info(
            "ollama_llm_client_initialized",
            base_url=base_url,
            model=self._runtime_config.llm_model,
            timeout=timeout,
            temperature=temperature,
        )

    async def generate(self, prompt: str, retry_count: int = 0) -> str:
        """
        Generate text using Ollama LLM.

        Args:
            prompt: Input prompt for generation
            retry_count: Current retry attempt (internal)

        Returns:
            Generated text response

        Raises:
            ExtractionError: If generation fails after retries
            TimeoutError: If request exceeds timeout
        """
        # Get current model from RuntimeConfig (supports hot-reload)
        model_name = self._runtime_config.llm_model

        try:
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": self.keep_alive,  # Keep model in memory
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": 2048,  # Max tokens to generate
                    },
                },
            )

            if response.status_code == 200:
                data = response.json()
                if "response" not in data:
                    raise ExtractionError(
                        message="Invalid Ollama response: missing 'response' field",
                        error_code="EXTR_003",
                        details={"response": data},
                    )
                return str(data["response"])

            elif response.status_code == 404:
                raise ExtractionError(
                    message=f"Model '{model_name}' not found. Run: ollama pull {model_name}",
                    error_code="EXTR_003",
                    details={"model": model_name},
                )

            else:
                raise ExtractionError(
                    message=f"Ollama API error: {response.status_code}",
                    error_code="EXTR_003",
                    details={"status_code": response.status_code, "response": response.text},
                )

        except httpx.TimeoutException as e:
            if retry_count < self.max_retries:
                wait_time = 2**retry_count
                logger.warning(
                    "ollama_llm_timeout_retry",
                    retry=retry_count + 1,
                    max_retries=self.max_retries,
                    wait_time=wait_time,
                )
                await asyncio.sleep(wait_time)
                return await self.generate(prompt, retry_count + 1)
            else:
                raise TimeoutError(
                    message=f"Ollama LLM request timed out after {self.max_retries} retries",
                    error_code="TIMEOUT_002",
                    details={"retries": retry_count},
                    original_exception=e,
                )

        except (httpx.ConnectError, httpx.NetworkError) as e:
            if retry_count < self.max_retries:
                wait_time = 2**retry_count
                logger.warning(
                    "ollama_llm_connection_retry",
                    retry=retry_count + 1,
                    max_retries=self.max_retries,
                    wait_time=wait_time,
                )
                await asyncio.sleep(wait_time)
                return await self.generate(prompt, retry_count + 1)
            else:
                raise ExtractionError(
                    message=f"Ollama connection failed after {self.max_retries} retries: {str(e)}",
                    error_code="EXTR_003",
                    details={"retries": retry_count, "error": str(e)},
                    original_exception=e,
                )

        except ExtractionError:
            raise

        except Exception as e:
            raise ExtractionError(
                message=f"Unexpected error calling Ollama LLM: {str(e)}",
                error_code="EXTR_003",
                details={"error": str(e)},
                original_exception=e,
            )

    async def refine_entities(
        self,
        text: str,
        entities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Refine entities extracted by SpaCy using LLM.

        Takes SpaCy NER entities and enhances them with:
        - Full/canonical names
        - Brief descriptions
        - Confidence scores
        - Validation (removes false positives)

        Args:
            text: Original text (max 10,000 chars for context)
            entities: List of entity dicts from SpaCy with 'name' and 'type'

        Returns:
            List of refined entity dicts with name, type, description, confidence

        Raises:
            ExtractionError: If LLM call or JSON parsing fails
        """
        if not entities:
            return []

        # Truncate text for context (keep first 10k chars)
        context_text = text[:10000] if len(text) > 10000 else text

        # Format entities for prompt
        entities_json = json.dumps(
            [{"name": e.get("name", ""), "type": e.get("type", "")} for e in entities], indent=2
        )

        # Build prompt
        prompt = ENTITY_REFINEMENT_PROMPT.format(
            text=context_text,
            entities_json=entities_json,
        )

        logger.debug(
            "refining_entities",
            num_entities=len(entities),
            text_length=len(context_text),
        )

        # Call LLM
        response = await self.generate(prompt)

        # Parse JSON response
        try:
            refined = self._parse_json_response(response)

            # Validate structure
            validated = []
            for entity in refined:
                if isinstance(entity, dict) and "name" in entity and "type" in entity:
                    validated.append(
                        {
                            "name": str(entity.get("name", "")),
                            "type": str(entity.get("type", "")),
                            "description": str(entity.get("description", "")),
                            "confidence": float(entity.get("confidence", 0.9)),
                        }
                    )

            logger.info(
                "entities_refined",
                input_count=len(entities),
                output_count=len(validated),
            )

            return validated

        except Exception as e:
            logger.warning(
                "entity_refinement_parse_failed",
                error=str(e),
                response_preview=response[:200],
            )
            # Fallback: return original entities with default confidence
            return [
                {
                    "name": e.get("name", ""),
                    "type": e.get("type", ""),
                    "description": "",
                    "confidence": 0.85,
                }
                for e in entities
            ]

    async def extract_relationships(
        self,
        text: str,
        entities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Extract relationships between entities using LLM.

        Analyzes text to find connections between entities such as:
        - CREATED: Person/Org created something
        - WORKS_FOR: Person works for Organization
        - LOCATED_IN: Entity is located in a place
        - PART_OF: Entity is part of another
        - USES: Entity uses another entity
        - RELATED_TO: General relationship

        Args:
            text: Original text (max 10,000 chars for context)
            entities: List of entity dicts with 'name' and 'type'

        Returns:
            List of relationship dicts with source, target, type, confidence, evidence

        Raises:
            ExtractionError: If LLM call or JSON parsing fails
        """
        if not entities or len(entities) < 2:
            return []  # Need at least 2 entities for relationships

        # Truncate text for context
        context_text = text[:10000] if len(text) > 10000 else text

        # Format entities for prompt
        entities_json = json.dumps(
            [{"name": e.get("name", ""), "type": e.get("type", "")} for e in entities], indent=2
        )

        # Build prompt
        prompt = RELATIONSHIP_EXTRACTION_PROMPT.format(
            text=context_text,
            entities_json=entities_json,
        )

        logger.debug(
            "extracting_relationships",
            num_entities=len(entities),
            text_length=len(context_text),
        )

        # Call LLM
        response = await self.generate(prompt)

        # Parse JSON response
        try:
            relationships = self._parse_json_response(response)

            # Build set of valid entity names for validation
            valid_names = {e.get("name", "").lower() for e in entities}

            # Validate structure and entity references
            validated = []
            for rel in relationships:
                if not isinstance(rel, dict):
                    continue

                source = str(rel.get("source", ""))
                target = str(rel.get("target", ""))
                rel_type = str(rel.get("type", "RELATED_TO"))

                # Check both entities exist
                if source.lower() in valid_names and target.lower() in valid_names:
                    validated.append(
                        {
                            "source": source,
                            "target": target,
                            "type": rel_type,
                            "confidence": float(rel.get("confidence", 0.8)),
                            "evidence": str(rel.get("evidence", ""))[:200],  # Limit evidence length
                        }
                    )

            logger.info(
                "relationships_extracted",
                num_entities=len(entities),
                num_relationships=len(validated),
            )

            return validated

        except Exception as e:
            logger.warning(
                "relationship_extraction_parse_failed",
                error=str(e),
                response_preview=response[:200],
            )
            return []

    def _parse_json_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse JSON array from LLM response.

        Handles common LLM output quirks:
        - Extra text before/after JSON
        - Markdown code blocks
        - Escaped characters

        Args:
            response: Raw LLM response text

        Returns:
            Parsed list of dictionaries

        Raises:
            ValueError: If JSON parsing fails
        """
        # Clean response
        text = response.strip()

        # Remove markdown code blocks if present
        if "```json" in text:
            text = re.sub(r"```json\s*", "", text)
            text = re.sub(r"```\s*$", "", text)
        elif "```" in text:
            text = re.sub(r"```\s*", "", text)

        # Find JSON array in response
        # Look for [ ... ] pattern
        match = re.search(r"\[[\s\S]*\]", text)
        if match:
            text = match.group(0)

        # Parse JSON
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
            else:
                return [result] if isinstance(result, dict) else []
        except json.JSONDecodeError as e:
            logger.debug("json_parse_failed", error=str(e), text_preview=text[:100])
            raise ValueError(f"Failed to parse JSON: {str(e)}")

    async def health_check(self) -> bool:
        """
        Check if Ollama LLM is available.

        Returns:
            True if Ollama is healthy and model is available
        """
        # Get current model from RuntimeConfig
        model_name = self._runtime_config.llm_model

        try:
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": "Hi",
                    "stream": False,
                    "keep_alive": "5m",  # Short keep_alive for health check
                    "options": {"num_predict": 1},
                },
                timeout=10.0,
            )

            if response.status_code == 200:
                logger.debug("ollama_llm_health_check_passed")
                return True

            logger.debug("ollama_llm_health_check_failed", status_code=response.status_code)
            return False

        except Exception as e:
            logger.debug("ollama_llm_health_check_failed", error=str(e))
            return False

    async def __aenter__(self) -> "OllamaLLMClient":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        """Async context manager exit - cleanup client."""
        await self.client.aclose()
