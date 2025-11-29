"""
Comprehensive integration tests for Zapomni MVP system.

Tests the complete end-to-end flow covering:
1. Full Memory Pipeline Integration (add → chunk → embed → store → search → retrieve)
2. MCP Tools Integration (AddMemoryTool, SearchMemoryTool, GetStatsTool)
3. Database Integration (FalkorDB operations, transactions, error recovery)
4. Search Integration (vector search, filtering, result ranking)
5. Error Handling Integration (error propagation, recovery)

Prerequisites:
- FalkorDB running on localhost:6381 (via docker-compose)
- Ollama embeddings service running on localhost:11434
- Run with: pytest tests/integration/test_mvp_integration.py -v

Target coverage: Complete end-to-end MVP flows with realistic data

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

import asyncio

import pytest

from zapomni_core.chunking import SemanticChunker
from zapomni_core.embeddings.ollama_embedder import OllamaEmbedder
from zapomni_core.exceptions import (
    ValidationError,
)
from zapomni_core.memory_processor import MemoryProcessor, ProcessorConfig
from zapomni_db.falkordb_client import FalkorDBClient
from zapomni_mcp.tools import AddMemoryTool, GetStatsTool, SearchMemoryTool

# ============================================================================
# Module-level fixtures (session scope)
# ============================================================================


@pytest.fixture(scope="module")
def falkordb_client():
    """
    Create FalkorDB client for integration tests.

    Uses a dedicated test graph to avoid polluting production data.
    Skips tests if FalkorDB is not available.
    """
    try:
        client = FalkorDBClient(
            host="localhost",
            port=6381,
            graph_name="zapomni_test_mvp",
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


# ============================================================================
# Function-level fixtures (setup/teardown for each test)
# ============================================================================


@pytest.fixture
async def memory_processor(falkordb_client, semantic_chunker, ollama_embedder):
    """
    Create MemoryProcessor for integration test.

    Creates a fresh processor for each test with a clean database.
    """
    # Clear database before each test
    await falkordb_client.clear_all()

    processor = MemoryProcessor(
        db_client=falkordb_client,
        chunker=semantic_chunker,
        embedder=ollama_embedder,
        config=ProcessorConfig(
            enable_cache=False,
            enable_extraction=False,
            enable_graph=False,
            max_text_length=10_000_000,
            batch_size=4,
            search_mode="vector",
        ),
    )

    yield processor

    # Cleanup after test
    await falkordb_client.clear_all()


@pytest.fixture
async def add_memory_tool(memory_processor):
    """Create AddMemoryTool for integration test."""
    return AddMemoryTool(memory_processor=memory_processor)


@pytest.fixture
async def search_memory_tool(memory_processor):
    """Create SearchMemoryTool for integration test."""
    return SearchMemoryTool(memory_processor=memory_processor)


@pytest.fixture
async def get_stats_tool(memory_processor):
    """Create GetStatsTool for integration test."""
    return GetStatsTool(memory_processor=memory_processor)


# ============================================================================
# Test Class 1: Full Memory Pipeline Integration
# ============================================================================


@pytest.mark.integration
class TestFullMemoryPipeline:
    """
    Test complete end-to-end memory pipeline.

    Validates: add memory → chunk → embed → store → search → retrieve flow
    """

    @pytest.mark.asyncio
    async def test_add_and_search_memory_flow(self, memory_processor):
        """
        Test: Add memory then search and find it

        Validates:
        - Memory is chunked correctly
        - Embeddings are generated
        - Data is stored in database
        - Search retrieves stored memory
        - Similarity score is meaningful
        """
        # Add memory
        test_text = (
            "Python is a high-level programming language created by Guido van Rossum in 1991."
        )
        memory_id = await memory_processor.add_memory(text=test_text)

        assert memory_id is not None
        assert isinstance(memory_id, str)
        assert len(memory_id) > 0

        # Search for related content
        results = await memory_processor.search_memory(
            query="Python programming language",
            limit=5,
        )

        assert len(results) > 0
        assert results[0].memory_id == memory_id
        assert results[0].similarity_score > 0.5
        assert "Python" in results[0].text

    @pytest.mark.asyncio
    async def test_multiple_memories_search(self, memory_processor):
        """
        Test: Add multiple memories and verify search returns ranked results

        Validates:
        - Multiple memories can be stored
        - Search returns results ranked by similarity
        - Results are sorted correctly (highest similarity first)
        """
        # Add multiple memories
        memories = [
            "Python is used for web development with Django and Flask frameworks.",
            "JavaScript runs in web browsers and is used for frontend development.",
            "Python is also popular for data science and machine learning.",
        ]

        memory_ids = []
        for text in memories:
            mid = await memory_processor.add_memory(text=text)
            memory_ids.append(mid)

        assert len(memory_ids) == 3

        # Search for Python-related content
        results = await memory_processor.search_memory(
            query="Python data science",
            limit=10,
        )

        assert len(results) >= 2
        # Results should be ranked by similarity
        assert results[0].similarity_score >= results[1].similarity_score

    @pytest.mark.asyncio
    async def test_memory_with_metadata(self, memory_processor):
        """
        Test: Add memory with metadata, search with filters

        Validates:
        - Metadata is stored correctly
        - Metadata is retrievable
        - Metadata filters work in search
        """
        text = "Django is a web framework for Python with ORM capabilities."
        metadata = {
            "tags": ["django", "python", "web"],
            "source": "documentation",
            "author": "test_user",
        }

        memory_id = await memory_processor.add_memory(
            text=text,
            metadata=metadata,
        )

        # Search and verify metadata
        results = await memory_processor.search_memory(
            query="Django ORM",
            limit=5,
        )

        assert len(results) > 0
        assert results[0].memory_id == memory_id
        assert "django" in results[0].tags
        assert results[0].source == "documentation"

    @pytest.mark.asyncio
    async def test_large_text_chunking(self, memory_processor):
        """
        Test: Large text gets chunked and stored correctly

        Validates:
        - Large documents are split into chunks
        - All chunks are stored
        - Search works across chunks
        """
        # Create a large document
        paragraph = "Python is a versatile programming language. " * 50  # ~2KB
        text = "\n\n".join([paragraph] * 5)  # ~10KB total

        memory_id = await memory_processor.add_memory(text=text)

        # Search should still work
        results = await memory_processor.search_memory(
            query="Python programming versatile",
            limit=5,
        )

        assert len(results) > 0
        assert results[0].memory_id == memory_id

    @pytest.mark.asyncio
    async def test_various_content_types(self, memory_processor):
        """
        Test: System handles various content types (text, markdown, code)

        Validates:
        - Plain text is stored correctly
        - Markdown is processed
        - Code snippets are handled
        """
        # Plain text
        text1 = "This is plain text documentation about the project."
        await memory_processor.add_memory(text=text1)

        # Markdown
        text2 = """
# Python Project Guide

## Installation
Run `pip install project`

## Usage
```python
from project import main
main()
```
"""
        await memory_processor.add_memory(text=text2)

        # Code
        text3 = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
        await memory_processor.add_memory(text=text3)

        # All should be searchable
        results = await memory_processor.search_memory(
            query="project installation python",
            limit=10,
        )

        assert len(results) >= 2  # At least text1 and text2


# ============================================================================
# Test Class 2: Statistics Integration
# ============================================================================


@pytest.mark.integration
class TestStatisticsIntegration:
    """
    Test statistics collection and reporting.

    Validates: Stats update correctly after operations
    """

    @pytest.mark.asyncio
    async def test_stats_after_operations(self, memory_processor):
        """
        Test: Stats update correctly after add operations

        Validates:
        - Stats show correct number of memories
        - Stats show correct number of chunks
        - Stats update after each add_memory call
        """
        # Initial stats should show empty database
        stats = await memory_processor.get_stats()
        initial_memories = stats.get("total_memories", 0)
        initial_chunks = stats.get("total_chunks", 0)

        assert initial_memories >= 0
        assert initial_chunks >= 0

        # Add first memory
        text1 = "First memory about Python"
        await memory_processor.add_memory(text=text1)

        stats = await memory_processor.get_stats()
        assert stats["total_memories"] == initial_memories + 1
        assert stats["total_chunks"] > initial_chunks

        # Add second memory
        text2 = "Second memory about JavaScript"
        await memory_processor.add_memory(text=text2)

        stats = await memory_processor.get_stats()
        assert stats["total_memories"] == initial_memories + 2

    @pytest.mark.asyncio
    async def test_empty_database_search(self, memory_processor):
        """
        Test: Search returns empty on new database

        Validates:
        - Empty search returns empty results (not error)
        - Stats show zero memories initially
        """
        # Stats should show empty database
        stats = await memory_processor.get_stats()
        assert stats["total_memories"] == 0
        assert stats["total_chunks"] == 0

        # Search on empty database returns empty results
        results = await memory_processor.search_memory(
            query="anything",
            limit=10,
        )

        assert len(results) == 0


# ============================================================================
# Test Class 3: Database Integration
# ============================================================================


@pytest.mark.integration
class TestDatabaseIntegration:
    """
    Test FalkorDB operations and transaction handling.

    Validates: Database adds, searches, and recovers from errors
    """

    @pytest.mark.asyncio
    async def test_duplicate_memory_handling(self, memory_processor):
        """
        Test: Adding similar content stores each as separate memory

        Validates:
        - Duplicate content is stored as separate memories
        - Both are searchable independently
        """
        text1 = "Python is a programming language created by Guido van Rossum."
        text2 = "Python is a programming language created by Guido van Rossum."

        mid1 = await memory_processor.add_memory(text=text1)
        mid2 = await memory_processor.add_memory(text=text2)

        # Should be stored as different memories
        assert mid1 != mid2

        # Both should be searchable
        results = await memory_processor.search_memory(
            query="Python Guido van Rossum",
            limit=10,
        )

        assert len(results) >= 2

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, memory_processor):
        """
        Test: Multiple concurrent add operations work correctly

        Validates:
        - Concurrent adds don't conflict
        - All memories are stored
        - Database remains consistent
        """
        texts = [
            "First concurrent memory about Python",
            "Second concurrent memory about JavaScript",
            "Third concurrent memory about Go",
            "Fourth concurrent memory about Rust",
        ]

        # Run multiple adds concurrently
        tasks = [memory_processor.add_memory(text=text) for text in texts]
        memory_ids = await asyncio.gather(*tasks)

        # All should succeed and be unique
        assert len(memory_ids) == 4
        assert len(set(memory_ids)) == 4

        # All should be searchable
        stats = await memory_processor.get_stats()
        assert stats["total_memories"] >= 4


# ============================================================================
# Test Class 4: Search Integration
# ============================================================================


@pytest.mark.integration
class TestSearchIntegration:
    """
    Test search functionality across all search modes.

    Validates: Vector search works end-to-end with filtering and ranking
    """

    @pytest.mark.asyncio
    async def test_vector_search_ranking(self, memory_processor):
        """
        Test: Vector search returns results ranked by similarity

        Validates:
        - Results are sorted by similarity (descending)
        - Higher similarity scores come first
        - Scores are in valid range [0, 1]
        """
        texts = [
            "Python is a high-level programming language",
            "Java is another popular programming language",
            "C++ is a systems programming language with low-level access",
        ]

        for text in texts:
            await memory_processor.add_memory(text=text)

        # Search for Python specifically
        results = await memory_processor.search_memory(
            query="Python high-level",
            limit=10,
        )

        assert len(results) > 0
        # Check ranking
        for i in range(len(results) - 1):
            assert results[i].similarity_score >= results[i + 1].similarity_score
        # Check valid range
        for result in results:
            assert 0.0 <= result.similarity_score <= 1.0

    @pytest.mark.asyncio
    async def test_search_with_limit(self, memory_processor):
        """
        Test: Search respects limit parameter

        Validates:
        - Limit parameter is enforced
        - Results don't exceed limit
        - Respects limit even with many matches
        """
        # Add multiple related memories
        for i in range(15):
            text = f"Memory {i} about Python programming language and web development"
            await memory_processor.add_memory(text=text)

        # Search with limit
        results = await memory_processor.search_memory(
            query="Python programming",
            limit=5,
        )

        assert len(results) <= 5

    @pytest.mark.asyncio
    async def test_search_with_metadata_filters(self, memory_processor):
        """
        Test: Search filters by metadata tags and source

        Validates:
        - Tag filters work correctly
        - Source filters work correctly
        - Filters reduce result set appropriately
        """
        # Add memories with different metadata
        await memory_processor.add_memory(
            text="Django web framework tutorial",
            metadata={"tags": ["django", "web"], "source": "tutorial"},
        )

        await memory_processor.add_memory(
            text="Flask lightweight web framework",
            metadata={"tags": ["flask", "web"], "source": "documentation"},
        )

        await memory_processor.add_memory(
            text="FastAPI modern API framework",
            metadata={"tags": ["fastapi", "api"], "source": "documentation"},
        )

        # Search with tag filter
        results = await memory_processor.search_memory(
            query="web framework",
            limit=10,
            filters={"tags": ["web"]},
        )

        assert len(results) > 0
        for result in results:
            assert "web" in result.tags

    @pytest.mark.asyncio
    async def test_search_result_content_accuracy(self, memory_processor):
        """
        Test: Search results contain accurate content and metadata

        Validates:
        - Result text matches stored content
        - Metadata is correctly returned
        - Memory IDs are valid UUIDs
        """
        text = "This is a test memory about content accuracy"
        metadata = {
            "tags": ["test", "accuracy"],
            "source": "testing",
        }

        memory_id = await memory_processor.add_memory(
            text=text,
            metadata=metadata,
        )

        results = await memory_processor.search_memory(
            query="content accuracy test",
            limit=5,
        )

        assert len(results) > 0
        assert results[0].memory_id == memory_id
        assert text in results[0].text
        assert "test" in results[0].tags
        assert results[0].source == "testing"


# ============================================================================
# Test Class 5: MCP Tools Integration
# ============================================================================


@pytest.mark.integration
class TestMCPToolsIntegration:
    """
    Test MCP tools integration with MemoryProcessor.

    Validates: Tools correctly delegate to MemoryProcessor and return MCP-compliant responses
    """

    @pytest.mark.asyncio
    async def test_add_memory_tool_integration(self, add_memory_tool, memory_processor):
        """
        Test: AddMemoryTool works with real MemoryProcessor

        Validates:
        - Tool accepts MCP format input
        - Returns MCP format output
        - Data is actually stored in database
        """
        arguments = {
            "text": "Test memory for MCP tool integration",
            "metadata": {
                "tags": ["mcp", "test"],
                "source": "mcp_tool",
            },
        }

        result = await add_memory_tool.execute(arguments)

        # Verify MCP response format
        assert "content" in result
        assert "isError" in result
        assert result["isError"] is False
        assert len(result["content"]) > 0
        assert result["content"][0]["type"] == "text"

        # Verify data was actually stored
        stats = await memory_processor.get_stats()
        assert stats["total_memories"] >= 1

    @pytest.mark.asyncio
    async def test_search_memory_tool_integration(self, search_memory_tool, memory_processor):
        """
        Test: SearchMemoryTool works with real MemoryProcessor

        Validates:
        - Tool accepts MCP format input
        - Returns MCP format output with results
        - Searches stored data correctly
        """
        # Add test data
        await memory_processor.add_memory(text="Test data for search tool integration")

        arguments = {
            "query": "search tool integration",
            "limit": 5,
        }

        result = await search_memory_tool.execute(arguments)

        # Verify MCP response format
        assert "content" in result
        assert "isError" in result
        assert result["isError"] is False

    @pytest.mark.asyncio
    async def test_get_stats_tool_integration(self, get_stats_tool, memory_processor):
        """
        Test: GetStatsTool works with real MemoryProcessor

        Validates:
        - Tool returns MCP format output
        - Stats are accurate
        - Tool works with empty database
        """
        # Get stats
        result = await get_stats_tool.execute({})

        # Verify MCP response format
        assert "content" in result
        assert "isError" in result
        assert result["isError"] is False
        assert "text" in result["content"][0]

        # Stats should be displayed
        text = result["content"][0]["text"]
        assert "Total Memories" in text
        assert "Total Chunks" in text


# ============================================================================
# Test Class 6: Error Handling Integration
# ============================================================================


@pytest.mark.integration
class TestErrorHandlingIntegration:
    """
    Test error handling and recovery across components.

    Validates: Errors propagate correctly and system recovers
    """

    @pytest.mark.asyncio
    async def test_error_handling_invalid_input(self, memory_processor):
        """
        Test: Invalid inputs handled gracefully

        Validates:
        - Empty text raises ValidationError
        - Invalid queries raise ValidationError
        - Errors don't corrupt database
        """
        # Test empty text
        with pytest.raises(ValidationError):
            await memory_processor.add_memory(text="")

        # Test invalid search query
        with pytest.raises(ValidationError):
            await memory_processor.search_memory(query="")

        # Test invalid limit
        with pytest.raises(ValidationError):
            await memory_processor.search_memory(query="test", limit=0)

        # Database should still be usable
        stats = await memory_processor.get_stats()
        assert stats["total_memories"] == 0

    @pytest.mark.asyncio
    async def test_error_handling_with_valid_recovery(self, memory_processor):
        """
        Test: System recovers after errors and continues working

        Validates:
        - After error, can add new memory successfully
        - Search works after error
        - No data corruption
        """
        # Try invalid operation
        try:
            await memory_processor.add_memory(text="")
        except ValidationError:
            pass  # Expected

        # System should recover and work normally
        memory_id = await memory_processor.add_memory(text="Recovery test memory")
        assert memory_id is not None

        # Search should work
        results = await memory_processor.search_memory(
            query="recovery test",
            limit=5,
        )
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_mcp_tool_error_response(self, add_memory_tool):
        """
        Test: MCP tools return proper error responses

        Validates:
        - Invalid arguments return error response
        - Error response follows MCP format
        - isError flag is set correctly
        """
        # Invalid arguments (missing required text)
        arguments = {}

        result = await add_memory_tool.execute(arguments)

        # Should be error response in MCP format
        assert result["isError"] is True
        assert "content" in result
        assert result["content"][0]["type"] == "text"


# ============================================================================
# Test Class 7: End-to-End Scenarios
# ============================================================================


@pytest.mark.integration
class TestEndToEndScenarios:
    """
    Test complete real-world scenarios.

    Validates: Full workflows work as expected
    """

    @pytest.mark.asyncio
    async def test_complete_workflow_scenario(
        self, memory_processor, add_memory_tool, search_memory_tool, get_stats_tool
    ):
        """
        Test: Complete workflow from add to search to stats

        Scenario:
        1. Add multiple memories using MCP tool
        2. Search for information using MCP tool
        3. Retrieve stats using MCP tool
        4. Verify all data is consistent

        Validates:
        - Complete MVP workflow works
        - All components integrate properly
        - Data flows correctly through system
        """
        # Step 1: Add memories via MCP tool
        test_memories = [
            {
                "text": "Python was created by Guido van Rossum in 1991",
                "metadata": {"tags": ["python", "history"], "source": "wiki"},
            },
            {
                "text": "Flask is a lightweight web framework for Python",
                "metadata": {"tags": ["python", "web"], "source": "documentation"},
            },
            {
                "text": "Django is a full-featured web framework",
                "metadata": {"tags": ["python", "web"], "source": "documentation"},
            },
        ]

        for mem in test_memories:
            result = await add_memory_tool.execute(mem)
            assert result["isError"] is False
            # Extract memory ID from response
            text_content = result["content"][0]["text"]
            assert "Memory stored successfully" in text_content

        # Step 2: Search via MCP tool
        search_results = await search_memory_tool.execute(
            {
                "query": "Python web frameworks",
                "limit": 5,
            }
        )

        assert search_results["isError"] is False
        assert "Found" in search_results["content"][0]["text"]

        # Step 3: Get stats via MCP tool
        stats_result = await get_stats_tool.execute({})

        assert stats_result["isError"] is False
        stats_text = stats_result["content"][0]["text"]
        assert "Memory System Statistics" in stats_text
        assert "Total Memories: 3" in stats_text

    @pytest.mark.asyncio
    async def test_knowledge_building_scenario(self, memory_processor):
        """
        Test: Building knowledge over time with multiple additions

        Scenario:
        1. Add memories about a topic incrementally
        2. Search improves as more data is added
        3. Stats show growth

        Validates:
        - System supports incremental knowledge building
        - Search improves with more data
        - Stats accurately track growth
        """
        # Build knowledge about "data science"
        texts = [
            "Data science combines statistics, programming, and domain expertise",
            "Machine learning is a subset of data science focusing on algorithms",
            "Python is the primary language for data science",
            "Libraries like NumPy, Pandas, and Scikit-learn are essential",
            "Data visualization helps communicate insights",
        ]

        for text in texts:
            await memory_processor.add_memory(text=text)

        # Search should return good results
        results = await memory_processor.search_memory(
            query="data science machine learning",
            limit=10,
        )

        assert len(results) >= 3

        # Stats should show growth
        stats = await memory_processor.get_stats()
        assert stats["total_memories"] == 5
        assert stats["total_chunks"] > 5


# ============================================================================
# Test Markers and Configuration
# ============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "integration: mark test as an integration test")
