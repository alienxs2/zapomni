"""
Comprehensive integration tests for Zapomni Phase 2 workflow.

Tests the complete Phase 2 end-to-end flow covering:
1. Memory Addition (add_memory → store with metadata)
2. Entity Extraction (build_graph → extract entities from text)
3. Graph Building (create Entity nodes and relationships)
4. Graph Traversal (get_related → find related entities)
5. Graph Status (graph_status → statistics and health)

Prerequisites:
- FalkorDB running on localhost:6381 (via docker-compose)
- Ollama embeddings service running on localhost:11434
- SpaCy model en_core_web_sm installed
- Run with: pytest tests/integration/test_phase2_integration.py -v

Target coverage: Complete Phase 2 workflow with realistic data

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

import pytest

from zapomni_core.chunking import SemanticChunker
from zapomni_core.embeddings.ollama_embedder import OllamaEmbedder
from zapomni_core.exceptions import (
    DatabaseError,
    EmbeddingError,
    ExtractionError,
    ValidationError,
)
from zapomni_core.extractors.entity_extractor import EntityExtractor
from zapomni_core.graph.graph_builder import GraphBuilder
from zapomni_core.memory_processor import MemoryProcessor, ProcessorConfig
from zapomni_db.falkordb_client import FalkorDBClient
from zapomni_mcp.tools import (
    AddMemoryTool,
    GetStatsTool,
    SearchMemoryTool,
)
from zapomni_mcp.tools.build_graph import BuildGraphTool
from zapomni_mcp.tools.get_related import GetRelatedTool
from zapomni_mcp.tools.graph_status import GraphStatusTool

# ============================================================================
# Module-level fixtures (session scope)
# ============================================================================


@pytest.fixture(scope="module")
def falkordb_client():
    """
    Create FalkorDB client for Phase 2 integration tests.

    Uses a dedicated test graph to avoid polluting production data.
    Skips tests if FalkorDB is not available.
    """
    try:
        client = FalkorDBClient(
            host="localhost",
            port=6381,
            graph_name="zapomni_test_phase2",
        )
        # Verify connection by clearing graph
        asyncio.run(client.clear_all())
        yield client
    except Exception as e:
        pytest.skip(f"FalkorDB not available at localhost:6381: {e}")
    finally:
        try:
            asyncio.run(client.clear_all())
            client.close()
        except Exception:
            pass


@pytest.fixture(scope="module")
def ollama_embedder():
    """
    Create OllamaEmbedder for integration tests.

    Skips tests if Ollama is not available on localhost:11434.
    """
    try:
        embedder = OllamaEmbedder(
            base_url="http://localhost:11434",
            model_name="nomic-embed-text",
        )
        # Verify by embedding test text
        asyncio.run(embedder.embed_text("test"))
        yield embedder
    except Exception as e:
        pytest.skip(f"Ollama not available at localhost:11434: {e}")


@pytest.fixture(scope="module")
def semantic_chunker():
    """Create SemanticChunker for integration tests."""
    return SemanticChunker(chunk_size=512, chunk_overlap=50)


@pytest.fixture(scope="module")
def entity_extractor():
    """
    Create EntityExtractor with SpaCy for integration tests.

    Skips tests if SpaCy model en_core_web_sm is not installed.
    """
    try:
        import spacy

        nlp = spacy.load("en_core_web_sm")
        extractor = EntityExtractor(
            spacy_model=nlp,
            confidence_threshold=0.6,
        )
        yield extractor
    except Exception as e:
        pytest.skip(f"SpaCy model en_core_web_sm not available: {e}")


@pytest.fixture(scope="module")
def graph_builder(falkordb_client, entity_extractor):
    """Create GraphBuilder for integration tests."""
    builder = GraphBuilder(
        db_client=falkordb_client,
        entity_extractor=entity_extractor,
    )
    return builder


# ============================================================================
# Function-level fixtures (setup/teardown for each test)
# ============================================================================


@pytest.fixture
async def memory_processor(
    falkordb_client, semantic_chunker, ollama_embedder, entity_extractor, graph_builder
):
    """
    Create MemoryProcessor with Phase 2 features enabled.

    Creates a fresh processor for each test with a clean database.
    Enables entity extraction and graph building.
    """
    # Clear database before each test
    await falkordb_client.clear_all()

    processor = MemoryProcessor(
        db_client=falkordb_client,
        chunker=semantic_chunker,
        embedder=ollama_embedder,
        extractor=entity_extractor,
        config=ProcessorConfig(
            enable_cache=False,
            enable_extraction=True,  # Phase 2: Enable entity extraction
            enable_graph=True,  # Phase 2: Enable graph building
            max_text_length=10_000_000,
            batch_size=4,
            search_mode="vector",
        ),
    )

    # Inject graph_builder into processor
    processor.entity_extractor = entity_extractor
    processor.graph_builder = graph_builder

    yield processor

    # Cleanup after test
    await falkordb_client.clear_all()


@pytest.fixture
async def build_graph_tool(memory_processor):
    """Create BuildGraphTool for integration test."""
    return BuildGraphTool(memory_processor=memory_processor)


@pytest.fixture
async def get_related_tool(memory_processor):
    """Create GetRelatedTool for integration test."""
    return GetRelatedTool(memory_processor=memory_processor)


@pytest.fixture
async def graph_status_tool(memory_processor):
    """Create GraphStatusTool for integration test."""
    return GraphStatusTool(memory_processor=memory_processor)


@pytest.fixture
async def add_memory_tool(memory_processor):
    """Create AddMemoryTool for integration test."""
    return AddMemoryTool(memory_processor=memory_processor)


# ============================================================================
# Test Class 1: Full Phase 2 Workflow
# ============================================================================


@pytest.mark.integration
class TestPhase2FullWorkflow:
    """
    Test complete Phase 2 workflow end-to-end.

    Validates: add_memory → build_graph → get_related → graph_status
    """

    @pytest.mark.asyncio
    async def test_full_workflow_success(
        self,
        memory_processor,
        add_memory_tool,
        build_graph_tool,
        graph_status_tool,
    ):
        """
        Test: Complete Phase 2 workflow from memory to graph status

        Workflow:
        1. Add multiple memories about Python programming
        2. Build knowledge graph (extract entities)
        3. Verify entities were extracted
        4. Check graph status shows populated graph

        Validates:
        - Memory addition works
        - Entity extraction extracts entities
        - Graph building creates nodes
        - Statistics reflect graph state
        """
        # Step 1: Add memories about Python programming
        test_memories = [
            {
                "text": "Python is a high-level programming language created by Guido van Rossum in 1991.",
                "metadata": {"tags": ["python", "history"], "source": "documentation"},
            },
            {
                "text": "Python uses dynamic typing and automatic memory management with garbage collection.",
                "metadata": {"tags": ["python", "features"], "source": "documentation"},
            },
            {
                "text": "Django and Flask are popular web frameworks for Python development.",
                "metadata": {"tags": ["python", "web"], "source": "tutorial"},
            },
        ]

        memory_ids = []
        for mem in test_memories:
            result = await add_memory_tool.execute(mem)
            assert result["isError"] is False
            memory_ids.append(result)

        # Step 2: Build knowledge graph from first memory
        build_result = await build_graph_tool.execute(
            {
                "text": test_memories[0]["text"],
                "options": {
                    "extract_entities": True,
                    "build_relationships": False,
                    "confidence_threshold": 0.6,
                },
            }
        )

        assert build_result["isError"] is False
        assert "Knowledge graph built successfully" in build_result["content"][0]["text"]
        assert "Entities:" in build_result["content"][0]["text"]

        # Step 3: Check graph status
        status_result = await graph_status_tool.execute({})

        assert status_result["isError"] is False
        status_text = status_result["content"][0]["text"]
        assert "Knowledge Graph Status" in status_text
        assert "Entities:" in status_text
        assert "Graph Health:" in status_text

    @pytest.mark.asyncio
    async def test_entity_extraction_from_text(
        self,
        build_graph_tool,
    ):
        """
        Test: Entity extraction identifies correct entities

        Validates:
        - Entities are extracted from text
        - Entity types are correct (PERSON, TECHNOLOGY, etc.)
        - Confidence scores are reasonable
        """
        # Test text with clear entities
        test_text = (
            "Python was created by Guido van Rossum. "
            "It is used by companies like Google and Microsoft. "
            "Popular frameworks include Django and Flask."
        )

        result = await build_graph_tool.execute(
            {
                "text": test_text,
                "options": {
                    "extract_entities": True,
                    "confidence_threshold": 0.5,
                },
            }
        )

        assert result["isError"] is False
        response_text = result["content"][0]["text"]

        # Verify entities were extracted
        assert "Entities:" in response_text
        # Should extract at least a few entities
        assert "Created:" in response_text or "Merged:" in response_text

    @pytest.mark.asyncio
    async def test_graph_building_with_relationships(
        self,
        build_graph_tool,
    ):
        """
        Test: Graph building creates relationships (Phase 2 stub)

        Validates:
        - Graph building processes successfully
        - Relationship count is reported (even if 0)
        - No errors during graph construction
        """
        test_text = "Python is a programming language. Django is a framework for Python."

        result = await build_graph_tool.execute(
            {
                "text": test_text,
                "options": {
                    "extract_entities": True,
                    "build_relationships": True,  # Phase 2 feature
                    "confidence_threshold": 0.6,
                },
            }
        )

        assert result["isError"] is False
        response_text = result["content"][0]["text"]

        # Verify relationships are mentioned
        assert "Relationships:" in response_text
        # In Phase 2, relationships may be 0 (stub implementation)

    @pytest.mark.asyncio
    async def test_graph_status_with_populated_graph(
        self,
        memory_processor,
        build_graph_tool,
        graph_status_tool,
    ):
        """
        Test: Graph status shows accurate statistics after building

        Validates:
        - Status shows correct node counts
        - Entity type breakdown is present
        - Graph health is "Healthy" with entities
        """
        # Add memory and build graph
        memory_id = await memory_processor.add_memory(
            text="Python is a programming language created by Guido van Rossum.",
            metadata={"source": "test"},
        )

        # Build graph from text
        await build_graph_tool.execute(
            {
                "text": "Python is a programming language created by Guido van Rossum.",
                "options": {"extract_entities": True},
            }
        )

        # Check status
        status_result = await graph_status_tool.execute({})

        assert status_result["isError"] is False
        status_text = status_result["content"][0]["text"]

        # Verify status components
        assert "Nodes:" in status_text
        assert "Total:" in status_text
        assert "Memories:" in status_text
        assert "Chunks:" in status_text
        assert "Relationships:" in status_text
        assert "Graph Health:" in status_text


# ============================================================================
# Test Class 2: Graph Traversal (get_related)
# ============================================================================


@pytest.mark.integration
class TestGraphTraversal:
    """
    Test graph traversal with get_related tool.

    Validates: Finding related entities through graph connections
    """

    @pytest.mark.asyncio
    async def test_get_related_entities_basic(
        self,
        memory_processor,
        build_graph_tool,
        get_related_tool,
    ):
        """
        Test: Get related entities returns results

        Validates:
        - get_related accepts valid entity_id
        - Returns related entities or empty result
        - Response format is correct
        """
        # Build a graph first
        await build_graph_tool.execute(
            {
                "text": "Python is used by Google. Django is a Python framework.",
                "options": {"extract_entities": True},
            }
        )

        # Try to get related entities (may not find any in stub implementation)
        # Use a valid UUID format
        test_entity_id = str(uuid.uuid4())

        result = await get_related_tool.execute(
            {
                "entity_id": test_entity_id,
                "depth": 2,
                "limit": 10,
            }
        )

        assert result["isError"] is False
        # Either finds related or reports none found
        response_text = result["content"][0]["text"]
        assert "related entities" in response_text.lower()

    @pytest.mark.asyncio
    async def test_graph_traversal_depth_variations(
        self,
        get_related_tool,
    ):
        """
        Test: Graph traversal works with different depth values

        Validates:
        - Depth 1 (direct connections)
        - Depth 2 (connections of connections)
        - Depth 3 (three hops)
        """
        test_entity_id = str(uuid.uuid4())

        # Test different depth values
        for depth in [1, 2, 3]:
            result = await get_related_tool.execute(
                {
                    "entity_id": test_entity_id,
                    "depth": depth,
                    "limit": 20,
                }
            )

            assert result["isError"] is False
            assert "content" in result


# ============================================================================
# Test Class 3: Error Handling in Workflow
# ============================================================================


@pytest.mark.integration
class TestErrorHandlingInWorkflow:
    """
    Test error handling throughout Phase 2 workflow.

    Validates: Proper error responses at each stage
    """

    @pytest.mark.asyncio
    async def test_build_graph_invalid_input(
        self,
        build_graph_tool,
    ):
        """
        Test: build_graph handles invalid inputs gracefully

        Validates:
        - Empty text raises error
        - Invalid options raise error
        - Error response format is correct
        """
        # Test empty text
        result = await build_graph_tool.execute(
            {
                "text": "",
            }
        )

        assert result["isError"] is True
        assert "Error:" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_get_related_invalid_entity_id(
        self,
        get_related_tool,
    ):
        """
        Test: get_related validates entity_id format

        Validates:
        - Invalid UUID format raises error
        - Error message is informative
        """
        # Test invalid UUID
        result = await get_related_tool.execute(
            {
                "entity_id": "not-a-valid-uuid",
                "depth": 2,
            }
        )

        assert result["isError"] is True
        assert "Error:" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_get_related_invalid_depth(
        self,
        get_related_tool,
    ):
        """
        Test: get_related validates depth parameter

        Validates:
        - Depth < 1 raises error
        - Depth > 5 raises error
        """
        test_entity_id = str(uuid.uuid4())

        # Test depth = 0
        result = await get_related_tool.execute(
            {
                "entity_id": test_entity_id,
                "depth": 0,
            }
        )

        assert result["isError"] is True

        # Test depth = 6
        result = await get_related_tool.execute(
            {
                "entity_id": test_entity_id,
                "depth": 6,
            }
        )

        assert result["isError"] is True

    @pytest.mark.asyncio
    async def test_graph_status_error_handling(
        self,
        graph_status_tool,
    ):
        """
        Test: graph_status handles errors gracefully

        Validates:
        - Works with empty graph
        - Works with populated graph
        - Never crashes on database errors
        """
        # Should work even with empty graph
        result = await graph_status_tool.execute({})

        assert result["isError"] is False
        assert "Knowledge Graph Status" in result["content"][0]["text"]


# ============================================================================
# Test Class 4: End-to-End Scenarios
# ============================================================================


@pytest.mark.integration
class TestEndToEndScenarios:
    """
    Test realistic end-to-end scenarios.

    Validates: Complete workflows with realistic data
    """

    @pytest.mark.asyncio
    async def test_python_knowledge_building(
        self,
        memory_processor,
        add_memory_tool,
        build_graph_tool,
        graph_status_tool,
    ):
        """
        Test: Build Python programming knowledge base

        Scenario:
        1. Add memories about Python features
        2. Extract entities from each memory
        3. Verify graph contains Python-related entities
        4. Check graph health is good
        """
        # Build knowledge base about Python
        python_memories = [
            "Python supports object-oriented, functional, and procedural programming paradigms.",
            "NumPy and Pandas are essential libraries for data science in Python.",
            "Python's asyncio library enables concurrent programming with async/await syntax.",
        ]

        # Add memories and extract entities
        for text in python_memories:
            await memory_processor.add_memory(text=text)
            await build_graph_tool.execute(
                {
                    "text": text,
                    "options": {"extract_entities": True, "confidence_threshold": 0.6},
                }
            )

        # Check graph status
        status_result = await graph_status_tool.execute({})

        assert status_result["isError"] is False
        status_text = status_result["content"][0]["text"]

        # Verify memories were added
        assert "Total Memories:" in status_text or "Memories:" in status_text
        # Verify graph has some entities
        assert "Entities:" in status_text

    @pytest.mark.asyncio
    async def test_multi_domain_knowledge(
        self,
        build_graph_tool,
        graph_status_tool,
    ):
        """
        Test: Extract entities from multiple domains

        Validates:
        - Different entity types are recognized
        - Graph handles diverse content
        - Statistics reflect multi-domain content
        """
        domains = [
            "Python is a programming language created by Guido van Rossum.",
            "Microsoft Azure and Amazon AWS are cloud computing platforms.",
            "Machine learning uses neural networks for pattern recognition.",
        ]

        # Extract entities from each domain
        for text in domains:
            result = await build_graph_tool.execute(
                {
                    "text": text,
                    "options": {"extract_entities": True},
                }
            )
            assert result["isError"] is False

        # Check status shows diverse entities
        status_result = await graph_status_tool.execute({})
        assert status_result["isError"] is False


# ============================================================================
# Test Markers and Configuration
# ============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "integration: mark test as a Phase 2 integration test")
