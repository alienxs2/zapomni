"""
Unit tests for GraphBuilder component.

Test Coverage:
- __init__: 5 tests
- build_graph: 8 tests
- add_entity_nodes: 6 tests
- add_relationships: 3 tests
- get_graph_stats: 2 tests
- clear_cache: 1 test
- Error handling and edge cases: 6+ tests

Total: 31+ comprehensive tests
"""

import asyncio
import uuid
from datetime import datetime, timezone
from typing import List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from zapomni_core.exceptions import (
    DatabaseError,
    ExtractionError,
    ProcessingError,
    ValidationError,
)
from zapomni_core.extractors.entity_extractor import Entity, EntityExtractor
from zapomni_core.graph.graph_builder import (
    GraphBuilder,
    GraphNode,
    GraphRelationship,
)
from zapomni_db import FalkorDBClient
from zapomni_db.models import Entity as DBEntity

# ============================================================================
# __init__ TESTS (5 tests)
# ============================================================================


class TestGraphBuilderInit:
    """Test GraphBuilder initialization."""

    def test_init_with_valid_params(self):
        """Test initialization with valid dependencies."""
        extractor = Mock(spec=EntityExtractor)
        db_client = Mock(spec=FalkorDBClient)

        builder = GraphBuilder(extractor, db_client)

        assert builder.entity_extractor is extractor
        assert builder.db_client is db_client
        assert builder._batch_size == 32

    def test_init_with_custom_batch_size(self):
        """Test initialization with custom batch size."""
        extractor = Mock(spec=EntityExtractor)
        db_client = Mock(spec=FalkorDBClient)

        builder = GraphBuilder(extractor, db_client, batch_size=16)

        assert builder._batch_size == 16

    def test_init_none_extractor_raises(self):
        """Test that None entity_extractor raises ValueError."""
        db_client = Mock(spec=FalkorDBClient)

        with pytest.raises(ValueError, match="entity_extractor"):
            GraphBuilder(None, db_client)

    def test_init_none_db_client_raises(self):
        """Test that None db_client raises ValueError."""
        extractor = Mock(spec=EntityExtractor)

        with pytest.raises(ValueError, match="db_client"):
            GraphBuilder(extractor, None)

    def test_init_invalid_batch_size_raises(self):
        """Test that batch_size < 1 raises ValueError."""
        extractor = Mock(spec=EntityExtractor)
        db_client = Mock(spec=FalkorDBClient)

        with pytest.raises(ValueError, match="batch_size"):
            GraphBuilder(extractor, db_client, batch_size=0)


# ============================================================================
# build_graph TESTS (8 tests)
# ============================================================================


class TestGraphBuilderBuildGraph:
    """Test GraphBuilder.build_graph method."""

    @pytest.fixture
    def setup(self):
        """Setup mock dependencies."""
        extractor = AsyncMock(spec=EntityExtractor)
        db_client = AsyncMock(spec=FalkorDBClient)
        builder = GraphBuilder(extractor, db_client)
        return builder, extractor, db_client

    @pytest.mark.asyncio
    async def test_build_graph_empty_memories_raises(self, setup):
        """Test that empty memories list raises ValidationError."""
        builder, _, _ = setup

        with pytest.raises(ValidationError, match="empty"):
            await builder.build_graph([])

    @pytest.mark.asyncio
    async def test_build_graph_too_many_memories_raises(self, setup):
        """Test that > 1000 memories raises ValidationError."""
        builder, _, _ = setup

        memories = [{"text": f"memory {i}"} for i in range(1001)]

        with pytest.raises(ValidationError, match="max size"):
            await builder.build_graph(memories)

    @pytest.mark.asyncio
    async def test_build_graph_no_text_raises(self, setup):
        """Test that no text available raises ValidationError."""
        builder, _, _ = setup

        memories = [{"text": ""}]  # Empty text field

        with pytest.raises(ValidationError, match="No text"):
            await builder.build_graph(memories)

    @pytest.mark.asyncio
    async def test_build_graph_no_entities_extracted(self, setup):
        """Test build_graph when no entities are extracted."""
        builder, extractor, db_client = setup

        # Mock extractor to return no entities
        extractor.extract_entities.return_value = []

        memories = [{"text": "Some text with no entities"}]

        result = await builder.build_graph(memories)

        assert result["entities_created"] == 0
        assert result["entities_merged"] == 0
        assert result["relationships_created"] == 0

    @pytest.mark.asyncio
    async def test_build_graph_success_with_entities(self, setup):
        """Test successful graph building with extracted entities."""
        builder, extractor, db_client = setup

        # Mock entities
        entities = [
            Entity(
                name="Python", type="TECHNOLOGY", description="Programming language", confidence=0.9
            ),
            Entity(name="Google", type="ORG", description="Tech company", confidence=0.95),
        ]
        extractor.extract_entities.return_value = entities

        # Mock db_client to return entity IDs
        db_client.add_entity.side_effect = [
            str(uuid.uuid4()),
            str(uuid.uuid4()),
        ]

        memories = [{"text": "Python is created by Google"}]

        result = await builder.build_graph(memories)

        assert result["entities_created"] == 2
        assert result["entities_merged"] == 0
        assert result["relationships_created"] == 0
        assert result["total_nodes"] == 2

    @pytest.mark.asyncio
    async def test_build_graph_with_string_memories(self, setup):
        """Test build_graph with string memories instead of dicts."""
        builder, extractor, db_client = setup

        entities = [
            Entity(name="Python", type="TECHNOLOGY", description="", confidence=0.85),
        ]
        extractor.extract_entities.return_value = entities

        db_client.add_entity.return_value = str(uuid.uuid4())

        memories = ["memory1", "memory2"]

        result = await builder.build_graph(memories)

        assert result["entities_created"] == 1
        assert extractor.extract_entities.called

    @pytest.mark.asyncio
    async def test_build_graph_extraction_error(self, setup):
        """Test that ExtractionError is propagated."""
        builder, extractor, db_client = setup

        extractor.extract_entities.side_effect = ExtractionError(
            message="Extraction failed",
            error_code="EXTR_001",
        )

        memories = [{"text": "Some text"}]

        with pytest.raises(ProcessingError):
            await builder.build_graph(memories)

    @pytest.mark.asyncio
    async def test_build_graph_with_explicit_text(self, setup):
        """Test build_graph with explicit text parameter."""
        builder, extractor, db_client = setup

        entities = [
            Entity(name="Python", type="TECHNOLOGY", description="", confidence=0.85),
        ]
        extractor.extract_entities.return_value = entities
        db_client.add_entity.return_value = str(uuid.uuid4())

        memories = [{"text": "ignored"}]
        explicit_text = "This text should be used"

        await builder.build_graph(memories, text=explicit_text)

        # Verify explicit text was used
        extractor.extract_entities.assert_called_with(explicit_text)


# ============================================================================
# add_entity_nodes TESTS (6 tests)
# ============================================================================


class TestGraphBuilderAddEntityNodes:
    """Test GraphBuilder.add_entity_nodes method."""

    @pytest.fixture
    def setup(self):
        """Setup mock dependencies."""
        extractor = AsyncMock(spec=EntityExtractor)
        db_client = AsyncMock(spec=FalkorDBClient)
        builder = GraphBuilder(extractor, db_client)
        return builder, extractor, db_client

    @pytest.mark.asyncio
    async def test_add_entity_nodes_empty_list_raises(self, setup):
        """Test that empty entities list raises ValidationError."""
        builder, _, _ = setup

        with pytest.raises(ValidationError, match="empty"):
            await builder.add_entity_nodes([])

    @pytest.mark.asyncio
    async def test_add_entity_nodes_too_many_entities_raises(self, setup):
        """Test that > 10000 entities raises ValidationError."""
        builder, _, _ = setup

        entities = [
            Entity(name=f"entity{i}", type="TECHNOLOGY", description="", confidence=0.85)
            for i in range(10001)
        ]

        with pytest.raises(ValidationError, match="max size"):
            await builder.add_entity_nodes(entities)

    @pytest.mark.asyncio
    async def test_add_entity_nodes_single_entity(self, setup):
        """Test adding a single entity."""
        builder, _, db_client = setup

        entity = Entity(
            name="Python", type="TECHNOLOGY", description="Programming language", confidence=0.9
        )
        db_client.add_entity.return_value = "entity-1"

        created, merged = await builder.add_entity_nodes([entity])

        assert created == 1
        assert merged == 0
        assert db_client.add_entity.called

    @pytest.mark.asyncio
    async def test_add_entity_nodes_deduplication(self, setup):
        """Test that duplicate entities are deduplicated."""
        builder, _, db_client = setup

        entities = [
            Entity(name="Python", type="TECHNOLOGY", description="Language", confidence=0.9),
            Entity(name="Python", type="TECHNOLOGY", description="Lang", confidence=0.85),
        ]
        db_client.add_entity.return_value = "entity-1"

        created, merged = await builder.add_entity_nodes(entities)

        assert created == 1  # First Python
        assert merged == 1  # Second Python (duplicate)

    @pytest.mark.asyncio
    async def test_add_entity_nodes_skip_empty_name(self, setup):
        """Test that entities with empty names are skipped."""
        builder, _, db_client = setup

        entities = [
            Entity(name="", type="TECHNOLOGY", description="", confidence=0.85),
            Entity(name="Python", type="TECHNOLOGY", description="", confidence=0.85),
        ]
        db_client.add_entity.return_value = "entity-1"

        created, merged = await builder.add_entity_nodes(entities)

        assert created == 1  # Only Python
        assert merged == 0

    @pytest.mark.asyncio
    async def test_add_entity_nodes_batch_processing(self, setup):
        """Test batch processing of entities."""
        builder, _, db_client = setup

        # Create entities exceeding batch size
        entities = [
            Entity(name=f"entity{i}", type="TECHNOLOGY", description="", confidence=0.85)
            for i in range(50)
        ]
        db_client.add_entity.return_value = str(uuid.uuid4())

        created, merged = await builder.add_entity_nodes(entities)

        assert created == 50
        assert merged == 0
        assert db_client.add_entity.call_count == 50

    @pytest.mark.asyncio
    async def test_add_entity_nodes_partial_failure(self, setup):
        """Test partial failure in entity addition."""
        builder, _, db_client = setup

        entities = [
            Entity(name="Python", type="TECHNOLOGY", description="", confidence=0.85),
            Entity(name="Google", type="ORG", description="", confidence=0.85),
        ]

        # First succeeds, second fails
        db_client.add_entity.side_effect = [
            "entity-1",
            DatabaseError("Connection error"),
        ]

        created, merged = await builder.add_entity_nodes(entities)

        assert created == 1  # Only Python succeeded
        assert merged == 0


# ============================================================================
# add_relationships TESTS (3 tests)
# ============================================================================


class TestGraphBuilderAddRelationships:
    """Test GraphBuilder.add_relationships method."""

    @pytest.fixture
    def setup(self):
        """Setup mock dependencies."""
        extractor = AsyncMock(spec=EntityExtractor)
        db_client = AsyncMock(spec=FalkorDBClient)
        builder = GraphBuilder(extractor, db_client)
        return builder, extractor, db_client

    @pytest.mark.asyncio
    async def test_add_relationships_not_implemented(self, setup):
        """Test that add_relationships raises NotImplementedError (Phase 2)."""
        builder, _, _ = setup

        entities = [
            Entity(name="Python", type="TECHNOLOGY", description="", confidence=0.85),
            Entity(name="Google", type="ORG", description="", confidence=0.85),
        ]

        with pytest.raises(NotImplementedError, match="Phase 2"):
            await builder.add_relationships(entities, "Some text")

    @pytest.mark.asyncio
    async def test_add_relationships_empty_entities_raises(self, setup):
        """Test that empty entities list raises ValidationError."""
        builder, _, _ = setup

        with pytest.raises(ValidationError):
            await builder.add_relationships([], "Some text")

    @pytest.mark.asyncio
    async def test_add_relationships_empty_text_raises(self, setup):
        """Test that empty text raises ValidationError."""
        builder, _, _ = setup

        entities = [
            Entity(name="Python", type="TECHNOLOGY", description="", confidence=0.85),
        ]

        with pytest.raises(ValidationError):
            await builder.add_relationships(entities, "")


# ============================================================================
# get_graph_stats TESTS (2 tests)
# ============================================================================


class TestGraphBuilderGetGraphStats:
    """Test GraphBuilder.get_graph_stats method."""

    def test_get_graph_stats_empty_cache(self):
        """Test get_graph_stats with empty cache."""
        extractor = Mock(spec=EntityExtractor)
        db_client = Mock(spec=FalkorDBClient)
        builder = GraphBuilder(extractor, db_client)

        stats = builder.get_graph_stats()

        assert stats["entities_in_cache"] == 0
        assert stats["batch_size"] == 32

    def test_get_graph_stats_with_cached_entities(self):
        """Test get_graph_stats with cached entities."""
        extractor = Mock(spec=EntityExtractor)
        db_client = Mock(spec=FalkorDBClient)
        builder = GraphBuilder(extractor, db_client)

        # Manually add entities to cache
        builder._entity_map["python:technology"] = "entity-1"
        builder._entity_map["google:org"] = "entity-2"

        stats = builder.get_graph_stats()

        assert stats["entities_in_cache"] == 2


# ============================================================================
# clear_cache TESTS (1 test)
# ============================================================================


class TestGraphBuilderClearCache:
    """Test GraphBuilder.clear_cache method."""

    def test_clear_cache(self):
        """Test clearing entity cache."""
        extractor = Mock(spec=EntityExtractor)
        db_client = Mock(spec=FalkorDBClient)
        builder = GraphBuilder(extractor, db_client)

        # Add entities to cache
        builder._entity_map["python:technology"] = "entity-1"
        builder._entity_map["google:org"] = "entity-2"

        assert len(builder._entity_map) == 2

        builder.clear_cache()

        assert len(builder._entity_map) == 0


# ============================================================================
# GraphNode and GraphRelationship TESTS (4 tests)
# ============================================================================


class TestGraphNode:
    """Test GraphNode data model."""

    def test_graph_node_creation(self):
        """Test creating a GraphNode."""
        node = GraphNode(
            entity_id="node-1",
            entity_name="Python",
            entity_type="TECHNOLOGY",
            description="Programming language",
            confidence=0.9,
            mentions=5,
        )

        assert node.entity_id == "node-1"
        assert node.entity_name == "Python"
        assert node.entity_type == "TECHNOLOGY"
        assert node.confidence == 0.9
        assert node.mentions == 5

    def test_graph_node_to_dict(self):
        """Test converting GraphNode to dictionary."""
        node = GraphNode(
            entity_id="node-1",
            entity_name="Python",
            entity_type="TECHNOLOGY",
            description="Language",
            confidence=0.9,
        )

        node_dict = node.to_dict()

        assert node_dict["entity_id"] == "node-1"
        assert node_dict["entity_name"] == "Python"
        assert "created_at" in node_dict


class TestGraphRelationship:
    """Test GraphRelationship data model."""

    def test_graph_relationship_creation(self):
        """Test creating a GraphRelationship."""
        rel = GraphRelationship(
            source_entity_id="entity-1",
            target_entity_id="entity-2",
            relationship_type="USES",
            confidence=0.85,
            evidence="Python uses libraries",
        )

        assert rel.source_entity_id == "entity-1"
        assert rel.target_entity_id == "entity-2"
        assert rel.relationship_type == "USES"
        assert rel.confidence == 0.85

    def test_graph_relationship_to_dict(self):
        """Test converting GraphRelationship to dictionary."""
        rel = GraphRelationship(
            source_entity_id="entity-1",
            target_entity_id="entity-2",
            relationship_type="USES",
            confidence=0.85,
        )

        rel_dict = rel.to_dict()

        assert rel_dict["source_entity_id"] == "entity-1"
        assert rel_dict["target_entity_id"] == "entity-2"
        assert "created_at" in rel_dict


# ============================================================================
# Integration Tests (3 tests)
# ============================================================================


class TestGraphBuilderIntegration:
    """Integration tests for GraphBuilder."""

    @pytest.mark.asyncio
    async def test_full_pipeline_with_mocks(self):
        """Test complete pipeline: build_graph -> add_entity_nodes."""
        extractor = AsyncMock(spec=EntityExtractor)
        db_client = AsyncMock(spec=FalkorDBClient)
        builder = GraphBuilder(extractor, db_client)

        # Mock data
        entities = [
            Entity(name="Python", type="TECHNOLOGY", description="Language", confidence=0.9),
            Entity(name="Django", type="TECHNOLOGY", description="Framework", confidence=0.88),
            Entity(name="Google", type="ORG", description="Company", confidence=0.95),
        ]
        extractor.extract_entities.return_value = entities
        db_client.add_entity.side_effect = [
            "ent-1",
            "ent-2",
            "ent-3",
        ]

        memories = [{"text": "Python framework Django used by Google"}]

        result = await builder.build_graph(memories)

        assert result["entities_created"] == 3
        assert result["total_nodes"] == 3
        assert db_client.add_entity.call_count == 3

    @pytest.mark.asyncio
    async def test_cache_persistence_across_calls(self):
        """Test that entity cache persists across calls."""
        extractor = AsyncMock(spec=EntityExtractor)
        db_client = AsyncMock(spec=FalkorDBClient)
        builder = GraphBuilder(extractor, db_client)

        # First batch of entities
        entities1 = [
            Entity(name="Python", type="TECHNOLOGY", description="", confidence=0.85),
        ]
        extractor.extract_entities.return_value = entities1
        db_client.add_entity.return_value = "ent-1"

        await builder.add_entity_nodes(entities1)

        # Second batch with duplicate
        entities2 = [
            Entity(name="Python", type="TECHNOLOGY", description="", confidence=0.90),
            Entity(name="Java", type="TECHNOLOGY", description="", confidence=0.85),
        ]

        db_client.add_entity.return_value = "ent-2"

        created, merged = await builder.add_entity_nodes(entities2)

        assert merged == 1  # Python was seen before
        assert created == 1  # Java is new

    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """Test recovery from partial failures."""
        extractor = AsyncMock(spec=EntityExtractor)
        db_client = AsyncMock(spec=FalkorDBClient)
        builder = GraphBuilder(extractor, db_client)

        entities = [
            Entity(name="Python", type="TECHNOLOGY", description="", confidence=0.85),
            Entity(name="Go", type="TECHNOLOGY", description="", confidence=0.85),
            Entity(name="Rust", type="TECHNOLOGY", description="", confidence=0.85),
        ]

        # Middle entity fails
        db_client.add_entity.side_effect = [
            "ent-1",
            DatabaseError("Connection lost"),
            "ent-3",
        ]

        created, merged = await builder.add_entity_nodes(entities)

        # Should still add 2 entities despite one failure
        assert created == 2
        assert merged == 0
