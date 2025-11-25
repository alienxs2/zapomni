"""
GraphBuilder - Knowledge graph construction component for Zapomni Phase 2.

Builds knowledge graphs from extracted entities and relationships:
1. Accepts extracted entities from EntityExtractor
2. Creates entity nodes in FalkorDB knowledge graph
3. Phase 2 stub: Relationship detection and linking

Implements async patterns for integration with MemoryProcessor.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from __future__ import annotations

import uuid
import asyncio
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timezone

import structlog

from zapomni_core.extractors.entity_extractor import Entity as ExtractedEntity, EntityExtractor
from zapomni_db import FalkorDBClient
from zapomni_db.models import Entity as DBEntity
from zapomni_core.exceptions import (
    ValidationError,
    ExtractionError,
    DatabaseError,
    ProcessingError,
)

logger = structlog.get_logger(__name__)


# ============================================================================
# GraphNode Data Models
# ============================================================================

class GraphNode:
    """
    Represents a node in the knowledge graph.

    Attributes:
        entity_id: Unique identifier for the entity in the graph
        entity_name: Name of the entity (e.g., "Python")
        entity_type: Type of entity (PERSON, ORG, TECHNOLOGY, CONCEPT, etc.)
        description: Brief description or context
        confidence: Confidence score of the extraction (0.0-1.0)
        mentions: Number of times entity appeared in source text
        created_at: Timestamp when node was added to graph
    """

    def __init__(
        self,
        entity_id: str,
        entity_name: str,
        entity_type: str,
        description: str = "",
        confidence: float = 0.85,
        mentions: int = 1,
        created_at: Optional[datetime] = None,
    ):
        self.entity_id = entity_id
        self.entity_name = entity_name
        self.entity_type = entity_type
        self.description = description
        self.confidence = confidence
        self.mentions = mentions
        self.created_at = created_at or datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert GraphNode to dictionary for serialization."""
        return {
            "entity_id": self.entity_id,
            "entity_name": self.entity_name,
            "entity_type": self.entity_type,
            "description": self.description,
            "confidence": self.confidence,
            "mentions": self.mentions,
            "created_at": self.created_at.isoformat(),
        }


class GraphRelationship:
    """
    Represents a relationship between two entities in the graph.

    Attributes:
        relationship_id: Unique identifier for the relationship
        source_entity_id: Source entity UUID
        target_entity_id: Target entity UUID
        relationship_type: Type of relationship (USES, CREATES, RELATED_TO, etc.)
        confidence: Confidence of the relationship detection (0.0-1.0)
        evidence: Text snippet or context supporting the relationship
        created_at: Timestamp when relationship was added
    """

    def __init__(
        self,
        source_entity_id: str,
        target_entity_id: str,
        relationship_type: str,
        confidence: float = 0.7,
        evidence: str = "",
        relationship_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
    ):
        self.relationship_id = relationship_id or str(uuid.uuid4())
        self.source_entity_id = source_entity_id
        self.target_entity_id = target_entity_id
        self.relationship_type = relationship_type
        self.confidence = confidence
        self.evidence = evidence
        self.created_at = created_at or datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert GraphRelationship to dictionary for serialization."""
        return {
            "relationship_id": self.relationship_id,
            "source_entity_id": self.source_entity_id,
            "target_entity_id": self.target_entity_id,
            "relationship_type": self.relationship_type,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "created_at": self.created_at.isoformat(),
        }


# ============================================================================
# GraphBuilder Class
# ============================================================================

class GraphBuilder:
    """
    Knowledge graph builder for Zapomni Phase 2.

    Coordinates graph construction from extracted entities:
    1. Validates entity extraction results
    2. Creates entity nodes in FalkorDB
    3. Phase 2: Adds relationship edges between entities

    Features:
        - Async pattern for integration with MemoryProcessor
        - Batch operations for performance
        - Deduplication of entities
        - Entity linking (merge duplicate entities)
        - Relationship detection stub (Phase 2)

    Attributes:
        entity_extractor: EntityExtractor for entity extraction
        db_client: FalkorDBClient for graph storage
        _entity_map: Cache of entity name -> entity_id for linking
        _batch_size: Batch size for bulk operations
    """

    DEFAULT_BATCH_SIZE = 32
    DEFAULT_RELATIONSHIP_CONFIDENCE = 0.7

    def __init__(
        self,
        entity_extractor: EntityExtractor,
        db_client: FalkorDBClient,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        """
        Initialize GraphBuilder with dependencies.

        Args:
            entity_extractor: EntityExtractor instance for entity extraction
            db_client: FalkorDBClient instance for database operations
            batch_size: Batch size for bulk operations (default: 32)

        Raises:
            ValueError: If entity_extractor or db_client is None
            ValueError: If batch_size < 1
        """
        if entity_extractor is None:
            raise ValueError("entity_extractor cannot be None")

        if db_client is None:
            raise ValueError("db_client cannot be None")

        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        self.entity_extractor = entity_extractor
        self.db_client = db_client
        self._batch_size = batch_size
        self._entity_map: Dict[str, str] = {}  # entity_name -> entity_id
        self._logger = logger.bind(component="graph_builder")

        self._logger.info(
            "graph_builder_initialized",
            batch_size=batch_size,
        )

    async def build_graph(
        self,
        memories: List[Dict[str, Any]],
        text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build knowledge graph from memories.

        Workflow:
        1. Extract entities from memories
        2. Add entity nodes to graph (deduplication)
        3. Phase 2: Detect relationships between entities
        4. Phase 2: Add relationship edges to graph

        Args:
            memories: List of memory dictionaries (or chunks) to extract from
            text: Optional complete text for entity extraction (if not in memories)

        Returns:
            Dict with statistics:
                {
                    "entities_created": int,
                    "entities_merged": int,
                    "relationships_created": int,
                    "total_nodes": int,
                    "total_edges": int,
                    "extraction_time_ms": float,
                }

        Raises:
            ValidationError: If memories list is empty or invalid
            ExtractionError: If entity extraction fails
            DatabaseError: If graph creation fails
        """
        # Validate inputs
        if not memories:
            raise ValidationError(
                message="Memories list cannot be empty",
                error_code="VAL_001",
                details={"memories_count": len(memories)},
            )

        if len(memories) > 1000:
            raise ValidationError(
                message="Memories list exceeds max size (1000 items)",
                error_code="VAL_002",
                details={"memories_count": len(memories)},
            )

        # Prepare text for extraction
        if text is None:
            # Build text from memories
            if isinstance(memories[0], dict) and "text" in memories[0]:
                text = " ".join([m.get("text", "") for m in memories if "text" in m])
            else:
                # Assume memories are strings
                text = " ".join([str(m) for m in memories])

        if not text or not text.strip():
            raise ValidationError(
                message="No text available for entity extraction",
                error_code="VAL_001",
                details={"text_length": len(text) if text else 0},
            )

        try:
            # STEP 1: Extract entities using async method to avoid blocking event loop
            self._logger.debug("extracting_entities", text_length=len(text))

            # Use async extraction if available (SSE transport), fallback to sync
            if hasattr(self.entity_extractor, 'extract_entities_async'):
                extracted_entities = await self.entity_extractor.extract_entities_async(text)
            else:
                # Fallback for older EntityExtractor instances without async support
                extracted_entities = self.entity_extractor.extract_entities(text)

            self._logger.info(
                "entities_extracted",
                num_entities=len(extracted_entities),
                text_length=len(text),
            )

            if not extracted_entities:
                self._logger.warning("no_entities_extracted")
                return {
                    "entities_created": 0,
                    "entities_merged": 0,
                    "relationships_created": 0,
                    "total_nodes": 0,
                    "total_edges": 0,
                    "extraction_time_ms": 0,
                }

            # STEP 2: Add entity nodes to graph
            created_count, merged_count = await self.add_entity_nodes(extracted_entities)

            # STEP 3: Phase 2 stub - relationship detection
            relationships_created = 0
            try:
                relationships_created = await self.add_relationships(
                    extracted_entities, text
                )
            except NotImplementedError:
                self._logger.debug("relationships_phase2_stub")

            # STEP 4: Return statistics
            stats = {
                "entities_created": created_count,
                "entities_merged": merged_count,
                "relationships_created": relationships_created,
                "total_nodes": created_count + merged_count,
                "total_edges": relationships_created,
                "extraction_time_ms": 0,  # Would measure in real implementation
            }

            self._logger.info("graph_building_complete", **stats)
            return stats

        except ValidationError:
            raise
        except Exception as e:
            self._logger.error("graph_building_failed", error=str(e))
            raise ProcessingError(
                message=f"Graph building failed: {str(e)}",
                error_code="GRAPH_001",
                details={"error": str(e), "memories_count": len(memories)},
                original_exception=e,
            )

    async def add_entity_nodes(self, entities: List[ExtractedEntity]) -> Tuple[int, int]:
        """
        Add entity nodes to FalkorDB graph.

        Deduplication strategy:
        - Check if entity already exists in graph (by name)
        - If exists: merge mentions, update confidence if higher
        - If new: create new node

        Args:
            entities: List of ExtractedEntity objects from extraction

        Returns:
            Tuple of (created_count, merged_count)

        Raises:
            ValidationError: If entities list is empty
            DatabaseError: If database operations fail
        """
        if not entities:
            raise ValidationError(
                message="Entities list cannot be empty",
                error_code="VAL_001",
            )

        if len(entities) > 10000:
            raise ValidationError(
                message="Entities list exceeds max size (10000 items)",
                error_code="VAL_002",
                details={"entities_count": len(entities)},
            )

        created_count = 0
        merged_count = 0
        errors = []

        self._logger.debug(
            "adding_entity_nodes",
            num_entities=len(entities),
        )

        # Process entities in batches
        for i in range(0, len(entities), self._batch_size):
            batch = entities[i : i + self._batch_size]

            # Add entities to graph
            for entity in batch:
                try:
                    # Validate entity
                    if not entity.name or not entity.name.strip():
                        self._logger.warning(
                            "skipping_empty_entity",
                            entity_type=entity.type,
                        )
                        continue

                    if not entity.type or not entity.type.strip():
                        self._logger.warning(
                            "skipping_entity_no_type",
                            entity_name=entity.name,
                        )
                        continue

                    # Check if entity exists in cache
                    entity_key = f"{entity.name}:{entity.type}".lower()

                    if entity_key in self._entity_map:
                        # Entity already exists - merge
                        merged_count += 1
                        self._logger.debug(
                            "entity_merged",
                            entity_name=entity.name,
                            entity_type=entity.type,
                        )
                    else:
                        # New entity - add to graph
                        # Convert ExtractedEntity to DBEntity (Pydantic model)
                        db_entity = DBEntity(
                            name=entity.name,
                            type=entity.type,
                            description=entity.description or "",
                            confidence=entity.confidence,
                        )

                        entity_id = await self.db_client.add_entity(db_entity)

                        # Store in cache
                        self._entity_map[entity_key] = entity_id
                        created_count += 1

                        self._logger.debug(
                            "entity_created",
                            entity_id=entity_id,
                            entity_name=entity.name,
                            entity_type=entity.type,
                        )

                except Exception as e:
                    self._logger.warning(
                        "entity_add_failed",
                        entity_name=entity.name,
                        error=str(e),
                    )
                    errors.append(str(e))
                    continue

            # Small delay to avoid overwhelming database
            if i + self._batch_size < len(entities):
                await asyncio.sleep(0.01)

        if errors:
            self._logger.warning(
                "entity_add_partial_failure",
                num_errors=len(errors),
                total_entities=len(entities),
            )

        self._logger.info(
            "entity_nodes_added",
            created=created_count,
            merged=merged_count,
            total=created_count + merged_count,
        )

        return created_count, merged_count

    async def add_relationships(
        self,
        entities: List[ExtractedEntity],
        text: str,
    ) -> int:
        """
        Add relationship edges between entities using LLM extraction.

        Relationship types:
        - CREATED: Person/Org created something
        - WORKS_FOR: Person works for Organization
        - LOCATED_IN: Entity is located in a place
        - PART_OF: Entity is part of another
        - USES: Entity uses another entity
        - RELATED_TO: General association

        Args:
            entities: List of Entity objects
            text: Original text for relationship context

        Returns:
            Number of relationships created

        Raises:
            NotImplementedError: If EntityExtractor doesn't have LLM enabled
            ValidationError: If inputs are invalid
            DatabaseError: If relationship creation fails
        """
        if not entities:
            raise ValidationError(
                message="Entities list cannot be empty",
                error_code="VAL_001",
            )

        if not text or not text.strip():
            raise ValidationError(
                message="Text cannot be empty",
                error_code="VAL_001",
            )

        if len(entities) < 2:
            self._logger.debug("not_enough_entities_for_relationships", count=len(entities))
            return 0

        # Check if EntityExtractor has LLM enabled
        if not self.entity_extractor.enable_llm_refinement:
            self._logger.debug(
                "llm_not_enabled_for_relationships",
                status="skipping",
            )
            return 0

        try:
            # Extract relationships using EntityExtractor's LLM
            relationships = self.entity_extractor.extract_relationships(text, entities)

            if not relationships:
                self._logger.debug("no_relationships_found")
                return 0

            # Add relationships to graph
            created_count = 0
            for rel in relationships:
                try:
                    # Find entity IDs from cache
                    source_key = f"{rel.source_entity}:{self._find_entity_type(rel.source_entity, entities)}".lower()
                    target_key = f"{rel.target_entity}:{self._find_entity_type(rel.target_entity, entities)}".lower()

                    source_id = self._entity_map.get(source_key)
                    target_id = self._entity_map.get(target_key)

                    if not source_id or not target_id:
                        self._logger.debug(
                            "relationship_entity_not_found",
                            source=rel.source_entity,
                            target=rel.target_entity,
                        )
                        continue

                    # Add relationship to database
                    await self.db_client.add_relationship(
                        from_entity_id=source_id,
                        to_entity_id=target_id,
                        relationship_type=rel.relationship_type,
                        properties={
                            "confidence": rel.confidence,
                            "evidence": rel.evidence[:200] if rel.evidence else "",
                        },
                    )
                    created_count += 1

                    self._logger.debug(
                        "relationship_created",
                        source=rel.source_entity,
                        target=rel.target_entity,
                        type=rel.relationship_type,
                    )

                except Exception as e:
                    self._logger.warning(
                        "relationship_add_failed",
                        source=rel.source_entity,
                        target=rel.target_entity,
                        error=str(e),
                    )
                    continue

            self._logger.info(
                "relationships_added",
                extracted=len(relationships),
                created=created_count,
            )

            return created_count

        except NotImplementedError:
            # LLM not available
            self._logger.debug("llm_not_available_for_relationships")
            return 0
        except Exception as e:
            self._logger.warning(
                "relationship_extraction_failed",
                error=str(e),
            )
            return 0

    def _find_entity_type(self, entity_name: str, entities: List[ExtractedEntity]) -> str:
        """Find entity type by name from entities list."""
        for e in entities:
            if e.name.lower() == entity_name.lower():
                return e.type
        return "UNKNOWN"

    def get_graph_stats(self) -> Dict[str, Any]:
        """
        Get current graph building statistics.

        Returns:
            Dict with entity cache stats and batching info

        Raises:
            Exception: If stats retrieval fails
        """
        stats = {
            "entities_in_cache": len(self._entity_map),
            "batch_size": self._batch_size,
            "cache_entries": list(self._entity_map.keys())[:10],  # First 10
        }

        return stats

    def clear_cache(self) -> None:
        """Clear the entity cache (used after batch processing)."""
        self._entity_map.clear()
        self._logger.info("entity_cache_cleared")
