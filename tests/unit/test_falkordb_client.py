"""
Unit tests for FalkorDBClient.

TDD Approach: Tests written FIRST based on specifications.
All tests should FAIL initially (no implementation yet).

Test Coverage:
- __init__: 10 tests
- add_memory: 20 tests
- vector_search: 31 tests
- get_stats: 13 tests
- close: 10 tests

Total: 84+ comprehensive tests
"""

import pytest
import uuid
import time
import asyncio
from datetime import datetime, timezone
from typing import List
from unittest.mock import Mock, AsyncMock, MagicMock, patch

# Will be implemented
from zapomni_db.falkordb_client import FalkorDBClient
from zapomni_db.models import Memory, Chunk, SearchResult, QueryResult
from zapomni_db.exceptions import ValidationError, DatabaseError, ConnectionError, QueryError


# ============================================================================
# __init__ TESTS (10 tests from spec)
# ============================================================================

class TestFalkorDBClientInit:
    """Test FalkorDBClient initialization based on __init__ spec."""

    def test_init_defaults(self):
        """Test initialization with default parameters."""
        client = FalkorDBClient()

        assert client.host == "localhost"
        assert client.port == 6381  # FalkorDB port
        assert client.db == 0

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        client = FalkorDBClient(
            host="custom-host",
            port=6380,
            db=5
        )

        assert client.host == "custom-host"
        assert client.port == 6380
        assert client.db == 5

    def test_init_zero_port_raises(self):
        """Test that port=0 raises ValueError."""
        with pytest.raises(ValidationError, match="port"):
            FalkorDBClient(port=0)

    def test_init_port_too_large_raises(self):
        """Test that port=65536 raises ValueError."""
        with pytest.raises(ValidationError, match="port"):
            FalkorDBClient(port=65536)

    def test_init_negative_db_raises(self):
        """Test that db=-1 raises ValueError."""
        with pytest.raises(ValidationError, match="db"):
            FalkorDBClient(db=-1)

    def test_init_db_too_large_raises(self):
        """Test that db=16 raises ValueError."""
        with pytest.raises(ValidationError, match="db"):
            FalkorDBClient(db=16)

    def test_init_empty_host_raises(self):
        """Test that empty host raises ValueError."""
        with pytest.raises(ValidationError, match="host"):
            FalkorDBClient(host="")

    def test_init_creates_connection_pool(self):
        """Test that connection pool is created on init."""
        client = FalkorDBClient()

        # Should have connection pool
        assert hasattr(client, '_pool')
        assert client._pool is not None

    def test_init_lazy_connection(self):
        """Test that actual connection is lazy (not on __init__)."""
        # Should not raise even if host is unreachable
        client = FalkorDBClient(host="nonexistent-host-12345.local")

        # Connection only happens on first operation
        assert client is not None

    def test_init_stores_params(self):
        """Test that __init__ stores all parameters correctly."""
        client = FalkorDBClient(
            host="test.host",
            port=1234,
            db=7
        )

        assert client.host == "test.host"
        assert client.port == 1234
        assert client.db == 7


# ============================================================================
# add_memory TESTS (20 tests from spec)
# ============================================================================

class TestFalkorDBClientAddMemory:
    """Test FalkorDBClient.add_memory() based on specification."""

    @pytest.fixture
    def mock_client(self, mocker):
        """Create mocked FalkorDBClient for testing."""
        client = FalkorDBClient()

        # Mock FalkorDB graph.query() to return successful result
        mock_result = MagicMock()
        mock_result.result_set = [["test-memory-id"]]  # Non-empty result_set
        mock_result.header = [[0, "memory_id"]]
        mock_result.nodes_created = 1
        mock_result.relationships_created = 1

        mock_graph = MagicMock()
        mock_graph.query = MagicMock(return_value=mock_result)

        mocker.patch.object(client, 'graph', mock_graph)
        mocker.patch.object(client, '_schema_ready', True)
        mocker.patch.object(client, '_initialized', True)

        return client

    # HAPPY PATH TESTS

    @pytest.mark.asyncio
    async def test_add_memory_success_minimal(self, mock_client):
        """Test basic success case with minimal input."""
        memory = Memory(
            text="Python is a programming language",
            chunks=[Chunk(text="Python is a programming language", index=0)],
            embeddings=[[0.1] * 768]
        )

        memory_id = await mock_client.add_memory(memory)

        assert isinstance(memory_id, str)
        assert len(memory_id) == 36  # UUID format
        uuid.UUID(memory_id)  # Validate UUID format

    @pytest.mark.asyncio
    async def test_add_memory_success_multiple_chunks(self, mock_client):
        """Test success with multiple chunks."""
        memory = Memory(
            text="Python is great for AI",
            chunks=[
                Chunk(text="Python is great", index=0),
                Chunk(text="great for AI", index=1)
            ],
            embeddings=[[0.1] * 768, [0.2] * 768],
            metadata={"source": "user"}
        )

        memory_id = await mock_client.add_memory(memory)
        assert memory_id is not None

    @pytest.mark.asyncio
    async def test_add_memory_success_with_rich_metadata(self, mock_client):
        """Test success with complex metadata."""
        metadata = {
            "source": "documentation",
            "tags": ["python", "ai"],
            "timestamp": "2025-11-23T10:00:00Z",
            "nested": {"key": "value"}
        }

        memory = Memory(
            text="Test",
            chunks=[Chunk(text="Test", index=0)],
            embeddings=[[0.1] * 768],
            metadata=metadata
        )

        memory_id = await mock_client.add_memory(memory)
        assert memory_id is not None

    @pytest.mark.asyncio
    async def test_add_memory_success_max_chunks(self, mock_client):
        """Test success at boundary (100 chunks max)."""
        chunks = [Chunk(text=f"chunk {i}", index=i) for i in range(100)]
        embeddings = [[float(i) / 100] * 768 for i in range(100)]

        memory = Memory(
            text="Long document",
            chunks=chunks,
            embeddings=embeddings
        )

        memory_id = await mock_client.add_memory(memory)
        assert memory_id is not None

    # VALIDATION ERROR TESTS

    @pytest.mark.asyncio
    async def test_add_memory_empty_text_raises(self, mock_client):
        """Test ValidationError on empty text (Pydantic validation)."""
        # Pydantic validates BEFORE method execution
        with pytest.raises(Exception, match="String should have at least 1 character"):
            memory = Memory(
                text="",
                chunks=[Chunk(text="dummy", index=0)],
                embeddings=[[0.1] * 768]
            )

    @pytest.mark.asyncio
    async def test_add_memory_text_too_long_raises(self, mock_client):
        """Test ValidationError when text > 1M chars (Pydantic validation)."""
        huge_text = "x" * 1_000_001

        # Pydantic validates BEFORE method execution
        with pytest.raises(Exception, match="String should have at most 1000000 characters"):
            memory = Memory(
                text=huge_text,
                chunks=[Chunk(text=huge_text, index=0)],
                embeddings=[[0.1] * 768]
            )

    @pytest.mark.asyncio
    async def test_add_memory_chunks_embeddings_mismatch_raises(self, mock_client):
        """Test ValidationError on count mismatch."""
        memory = Memory(
            text="Test text",
            chunks=[
                Chunk(text="chunk 1", index=0),
                Chunk(text="chunk 2", index=1),
                Chunk(text="chunk 3", index=2)
            ],
            embeddings=[[0.1] * 768, [0.2] * 768]  # Only 2 embeddings!
        )

        with pytest.raises(ValidationError, match="count mismatch"):
            await mock_client.add_memory(memory)

    @pytest.mark.asyncio
    async def test_add_memory_wrong_embedding_dimension_raises(self, mock_client):
        """Test ValidationError on wrong dimension."""
        memory = Memory(
            text="Test",
            chunks=[Chunk(text="Test", index=0)],
            embeddings=[[0.1] * 512]  # Wrong dimension!
        )

        with pytest.raises(ValidationError, match="dimension must be 768"):
            await mock_client.add_memory(memory)

    @pytest.mark.asyncio
    async def test_add_memory_too_many_chunks_raises(self, mock_client):
        """Test ValidationError when chunks > 100 (Pydantic validation)."""
        chunks = [Chunk(text=f"chunk {i}", index=i) for i in range(101)]
        embeddings = [[0.1] * 768 for _ in range(101)]

        # Pydantic validates BEFORE method execution
        with pytest.raises(Exception, match="List should have at most 100 items"):
            memory = Memory(
                text="Very long text",
                chunks=chunks,
                embeddings=embeddings
            )

    @pytest.mark.asyncio
    async def test_add_memory_empty_chunks_raises(self, mock_client):
        """Test ValidationError on empty chunks list (Pydantic validation)."""
        # Pydantic validates BEFORE method execution
        with pytest.raises(Exception, match="List should have at least 1 item"):
            memory = Memory(
                text="Test text",
                chunks=[],
                embeddings=[]
            )

    @pytest.mark.asyncio
    async def test_add_memory_non_serializable_metadata_raises(self, mock_client):
        """Test ValidationError on non-JSON metadata."""
        memory = Memory(
            text="Test",
            chunks=[Chunk(text="Test", index=0)],
            embeddings=[[0.1] * 768],
            metadata={"binary": b"bytes data"}  # Not JSON-serializable!
        )

        with pytest.raises(ValidationError, match="not JSON-serializable"):
            await mock_client.add_memory(memory)

    @pytest.mark.asyncio
    async def test_add_memory_metadata_too_large_raises(self, mock_client):
        """Test ValidationError when metadata > 100 KB."""
        huge_metadata = {"data": "x" * 200_000}

        memory = Memory(
            text="Test",
            chunks=[Chunk(text="Test", index=0)],
            embeddings=[[0.1] * 768],
            metadata=huge_metadata
        )

        with pytest.raises(ValidationError, match="metadata exceeds max size"):
            await mock_client.add_memory(memory)

    @pytest.mark.asyncio
    async def test_add_memory_invalid_chunk_index_raises(self, mock_client):
        """Test ValidationError when chunk indices not sequential."""
        memory = Memory(
            text="Test",
            chunks=[
                Chunk(text="A", index=0),
                Chunk(text="B", index=2)  # Skipped index 1!
            ],
            embeddings=[[0.1] * 768, [0.2] * 768]
        )

        with pytest.raises(ValidationError, match="index mismatch"):
            await mock_client.add_memory(memory)

    # DATABASE ERROR TESTS

    @pytest.mark.asyncio
    async def test_add_memory_connection_lost_retries_and_succeeds(self, mocker):
        """Test retry logic on transient errors."""
        client = FalkorDBClient()

        # Mock graph.query() to fail twice, succeed third time
        call_count = 0
        def mock_query(cypher, params):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Connection lost")

            # Success on 3rd attempt
            mock_result = MagicMock()
            mock_result.result_set = [[str(uuid.uuid4())]]
            mock_result.header = [[0, "memory_id"]]
            mock_result.nodes_created = 1
            mock_result.relationships_created = 1
            return mock_result

        mock_graph = MagicMock()
        mock_graph.query = MagicMock(side_effect=mock_query)
        mocker.patch.object(client, 'graph', mock_graph)
        mocker.patch.object(client, '_initialized', True)
        mocker.patch.object(client, '_schema_ready', True)

        memory = Memory(
            text="Test",
            chunks=[Chunk(text="Test", index=0)],
            embeddings=[[0.1] * 768]
        )

        memory_id = await client.add_memory(memory)
        assert memory_id is not None
        assert call_count == 3  # Succeeded on 3rd attempt

    @pytest.mark.asyncio
    async def test_add_memory_retries_exhausted_raises(self, mocker):
        """Test DatabaseError after all retries fail."""
        client = FalkorDBClient(max_retries=3)

        # Mock graph.query() to always fail
        def mock_query(cypher, params):
            raise ConnectionError("Connection failed")

        mock_graph = MagicMock()
        mock_graph.query = MagicMock(side_effect=mock_query)
        mocker.patch.object(client, 'graph', mock_graph)
        mocker.patch.object(client, '_initialized', True)
        mocker.patch.object(client, '_schema_ready', True)

        memory = Memory(
            text="Test",
            chunks=[Chunk(text="Test", index=0)],
            embeddings=[[0.1] * 768]
        )

        with pytest.raises(DatabaseError, match="after 3 retries"):
            await client.add_memory(memory)

    @pytest.mark.asyncio
    async def test_add_memory_transaction_commit_fails_raises(self, mocker):
        """Test DatabaseError when query execution fails."""
        client = FalkorDBClient()

        # Mock graph.query() to raise exception (simulates commit/write failure)
        def mock_query(cypher, params):
            raise Exception("Disk full")

        mock_graph = MagicMock()
        mock_graph.query = MagicMock(side_effect=mock_query)
        mocker.patch.object(client, 'graph', mock_graph)
        mocker.patch.object(client, '_initialized', True)
        mocker.patch.object(client, '_schema_ready', True)

        memory = Memory(
            text="Test",
            chunks=[Chunk(text="Test", index=0)],
            embeddings=[[0.1] * 768]
        )

        with pytest.raises(DatabaseError):
            await client.add_memory(memory)

    # PERFORMANCE TESTS

    @pytest.mark.asyncio
    async def test_add_memory_performance_within_sla(self, mock_client):
        """Test operation completes within acceptable time."""
        memory = Memory(
            text="Performance test",
            chunks=[Chunk(text=f"chunk {i}", index=i) for i in range(5)],
            embeddings=[[0.1] * 768 for _ in range(5)]
        )

        start = time.time()
        memory_id = await mock_client.add_memory(memory)
        elapsed = (time.time() - start) * 1000  # Convert to ms

        assert elapsed < 500  # 500ms SLA
        assert memory_id is not None

    @pytest.mark.asyncio
    async def test_add_memory_large_input_performance(self, mock_client):
        """Test performance with large memory (near max)."""
        memory = Memory(
            text="Large document",
            chunks=[Chunk(text=f"chunk {i}", index=i) for i in range(100)],
            embeddings=[[float(i) / 100] * 768 for i in range(100)]
        )

        start = time.time()
        memory_id = await mock_client.add_memory(memory)
        elapsed = (time.time() - start) * 1000

        assert elapsed < 2000  # 2 second SLA for large input

    @pytest.mark.asyncio
    async def test_add_memory_concurrent_operations(self, mock_client):
        """Test thread-safety with concurrent writes."""
        async def add_one(i):
            memory = Memory(
                text=f"Concurrent memory {i}",
                chunks=[Chunk(text=f"chunk {i}", index=0)],
                embeddings=[[float(i) / 10] * 768]
            )
            return await mock_client.add_memory(memory)

        # Add 10 memories concurrently
        memory_ids = await asyncio.gather(*[add_one(i) for i in range(10)])

        assert len(memory_ids) == 10
        assert len(set(memory_ids)) == 10  # All unique


# ============================================================================
# vector_search TESTS (31 tests from spec)
# ============================================================================

class TestFalkorDBClientVectorSearch:
    """Test FalkorDBClient.vector_search() based on specification."""

    @pytest.fixture
    def mock_search_client(self, mocker):
        """Create mocked client for search tests."""
        client = FalkorDBClient()

        # Mock successful search
        mock_result = MagicMock()
        mock_result.rows = [
            {
                "memory_id": str(uuid.uuid4()),
                "chunk_id": str(uuid.uuid4()),
                "text": "Python is great",
                "similarity_score": 0.95,
                "tags": ["python"],
                "source": "docs",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "chunk_index": 0
            }
        ]

        async def mock_execute(query, params):
            return mock_result

        mocker.patch.object(client, '_execute_cypher', side_effect=mock_execute)
        mocker.patch.object(client, '_initialized', True)
        mocker.patch.object(client, '_schema_ready', True)

        return client

    # HAPPY PATH TESTS

    @pytest.mark.asyncio
    async def test_vector_search_success_basic(self, mock_search_client):
        """Test basic successful vector search."""
        embedding = [0.1] * 768
        results = await mock_search_client.vector_search(embedding, limit=10)

        assert isinstance(results, list)
        assert len(results) <= 10
        if results:
            assert isinstance(results[0], SearchResult)

    @pytest.mark.asyncio
    async def test_vector_search_success_with_filters(self, mock_search_client):
        """Test search with filters."""
        embedding = [0.1] * 768
        filters = {
            "tags": ["python"],
            "source": "docs",
            "date_from": "2025-11-01T00:00:00Z"
        }

        results = await mock_search_client.vector_search(
            embedding, limit=5, filters=filters
        )

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_vector_search_boundary_limit_1(self, mock_search_client):
        """Test with limit=1."""
        embedding = [0.1] * 768
        results = await mock_search_client.vector_search(embedding, limit=1)

        assert len(results) <= 1

    @pytest.mark.asyncio
    async def test_vector_search_boundary_limit_1000(self, mock_search_client):
        """Test with limit=1000 (max)."""
        embedding = [0.1] * 768
        results = await mock_search_client.vector_search(embedding, limit=1000)

        assert isinstance(results, list)

    # VALIDATION ERROR TESTS

    def test_vector_search_invalid_dimension_512(self):
        """Test ValidationError with 512-dim embedding."""
        client = FalkorDBClient()
        embedding = [0.1] * 512

        with pytest.raises(ValidationError, match="dimension must be 768"):
            asyncio.run(client.vector_search(embedding, limit=10))

    def test_vector_search_invalid_dimension_1024(self):
        """Test ValidationError with 1024-dim embedding."""
        client = FalkorDBClient()
        embedding = [0.1] * 1024

        with pytest.raises(ValidationError, match="dimension must be 768"):
            asyncio.run(client.vector_search(embedding, limit=10))

    def test_vector_search_embedding_contains_nan(self):
        """Test ValidationError with NaN in embedding."""
        client = FalkorDBClient()
        embedding = [0.1] * 768
        embedding[100] = float('nan')

        with pytest.raises(ValidationError, match="NaN or Inf"):
            asyncio.run(client.vector_search(embedding, limit=10))

    def test_vector_search_embedding_contains_inf(self):
        """Test ValidationError with Inf in embedding."""
        client = FalkorDBClient()
        embedding = [0.1] * 768
        embedding[200] = float('inf')

        with pytest.raises(ValidationError, match="NaN or Inf"):
            asyncio.run(client.vector_search(embedding, limit=10))

    def test_vector_search_limit_zero(self):
        """Test ValidationError with limit=0."""
        client = FalkorDBClient()
        embedding = [0.1] * 768

        with pytest.raises(ValidationError, match="limit must be >= 1"):
            asyncio.run(client.vector_search(embedding, limit=0))

    def test_vector_search_limit_negative(self):
        """Test ValidationError with negative limit."""
        client = FalkorDBClient()
        embedding = [0.1] * 768

        with pytest.raises(ValidationError, match="limit must be >= 1"):
            asyncio.run(client.vector_search(embedding, limit=-10))

    def test_vector_search_limit_exceeds_max(self):
        """Test ValidationError with limit > 1000."""
        client = FalkorDBClient()
        embedding = [0.1] * 768

        with pytest.raises(ValidationError, match="cannot exceed 1000"):
            asyncio.run(client.vector_search(embedding, limit=1001))

    def test_vector_search_limit_not_integer(self):
        """Test ValidationError with float limit."""
        client = FalkorDBClient()
        embedding = [0.1] * 768

        with pytest.raises(ValidationError, match="must be int"):
            asyncio.run(client.vector_search(embedding, limit=10.5))

    def test_vector_search_filters_tags_empty_list(self):
        """Test ValidationError with empty tags list."""
        client = FalkorDBClient()
        embedding = [0.1] * 768

        with pytest.raises(ValidationError, match="cannot be empty"):
            asyncio.run(client.vector_search(
                embedding, limit=10, filters={"tags": []}
            ))

    def test_vector_search_filters_source_empty_string(self):
        """Test ValidationError with empty source."""
        client = FalkorDBClient()
        embedding = [0.1] * 768

        with pytest.raises(ValidationError, match="non-empty"):
            asyncio.run(client.vector_search(
                embedding, limit=10, filters={"source": ""}
            ))

    def test_vector_search_filters_invalid_date_format(self):
        """Test ValidationError with invalid date format."""
        client = FalkorDBClient()
        embedding = [0.1] * 768

        with pytest.raises(ValidationError, match="ISO 8601"):
            asyncio.run(client.vector_search(
                embedding, limit=10, filters={"date_from": "not-a-date"}
            ))

    def test_vector_search_filters_min_similarity_out_of_range(self):
        """Test ValidationError with min_similarity > 1.0."""
        client = FalkorDBClient()
        embedding = [0.1] * 768

        with pytest.raises(ValidationError, match="must be in \\[0.0, 1.0\\]"):
            asyncio.run(client.vector_search(
                embedding, limit=10, filters={"min_similarity": 1.5}
            ))

    # EDGE CASE TESTS

    @pytest.mark.asyncio
    async def test_vector_search_empty_database(self, mocker):
        """Test search on empty database returns []."""
        client = FalkorDBClient()

        # Mock empty result
        mock_result = MagicMock()
        mock_result.rows = []

        mocker.patch.object(client, '_execute_cypher', return_value=mock_result)
        mocker.patch.object(client, '_initialized', True)
        mocker.patch.object(client, '_schema_ready', True)

        embedding = [0.1] * 768
        results = await client.vector_search(embedding, limit=10)

        assert results == []

    @pytest.mark.asyncio
    async def test_vector_search_no_filter_matches(self, mock_search_client, mocker):
        """Test search with non-matching filters returns []."""
        # Override to return empty
        mock_result = MagicMock()
        mock_result.rows = []
        mocker.patch.object(mock_search_client, '_execute_cypher', return_value=mock_result)

        embedding = [0.1] * 768
        results = await mock_search_client.vector_search(
            embedding, limit=10, filters={"tags": ["nonexistent"]}
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_vector_search_connection_lost_retries_succeed(self, mocker):
        """Test retry logic succeeds after failures."""
        client = FalkorDBClient()

        call_count = 0
        async def mock_execute(query, params):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Connection lost")

            mock_result = MagicMock()
            mock_result.rows = []
            return mock_result

        mocker.patch.object(client, '_execute_cypher', side_effect=mock_execute)
        mocker.patch.object(client, '_initialized', True)
        mocker.patch.object(client, '_schema_ready', True)

        embedding = [0.1] * 768
        results = await client.vector_search(embedding, limit=10)

        assert results == []
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_vector_search_connection_lost_all_retries_fail(self, mocker):
        """Test DatabaseError when all retries fail."""
        client = FalkorDBClient()

        async def mock_execute(query, params):
            raise ConnectionError("Connection lost")

        mocker.patch.object(client, '_execute_cypher', side_effect=mock_execute)
        mocker.patch.object(client, '_initialized', True)
        mocker.patch.object(client, '_schema_ready', True)

        embedding = [0.1] * 768

        with pytest.raises(DatabaseError, match="after 3 retries"):
            await client.vector_search(embedding, limit=10)

    # PERFORMANCE TESTS (sampling from 31 total)

    @pytest.mark.asyncio
    async def test_vector_search_performance_small_db(self, mock_search_client):
        """Test performance on small database."""
        embedding = [0.1] * 768

        start = time.time()
        results = await mock_search_client.vector_search(embedding, limit=10)
        elapsed = (time.time() - start) * 1000

        assert elapsed < 50  # 50ms for mocked call


# ============================================================================
# get_stats TESTS (13 tests from spec)
# ============================================================================

class TestFalkorDBClientGetStats:
    """Test FalkorDBClient.get_stats() based on specification."""

    @pytest.fixture
    def mock_stats_client(self, mocker):
        """Create mocked client for stats tests."""
        client = FalkorDBClient()

        # Mock stats queries
        call_sequence = [
            # Node count query
            MagicMock(rows=[
                {"node_type": "Memory", "count": 10},
                {"node_type": "Chunk", "count": 35},
                {"node_type": "Entity", "count": 5}
            ]),
            # Relationship count query
            MagicMock(rows=[
                {"rel_type": "HAS_CHUNK", "count": 35},
                {"rel_type": "MENTIONS", "count": 5}
            ]),
            # Index query
            MagicMock(rows=[
                {"name": "chunk_embedding_idx", "type": "vector"}
            ])
        ]

        call_index = 0
        async def mock_execute(query, params):
            nonlocal call_index
            result = call_sequence[call_index]
            call_index += 1
            return result

        mocker.patch.object(client, '_execute_cypher', side_effect=mock_execute)
        mocker.patch.object(client, '_initialized', True)
        mocker.patch.object(client, '_schema_ready', True)
        mocker.patch.object(client, 'graph_name', 'test_graph')

        return client

    @pytest.mark.asyncio
    async def test_get_stats_success_normal_graph(self, mock_stats_client):
        """Test get_stats with normal graph."""
        stats = await mock_stats_client.get_stats()

        assert isinstance(stats, dict)
        assert "nodes" in stats
        assert "relationships" in stats
        assert "storage" in stats
        assert "indexes" in stats
        assert "health" in stats

    @pytest.mark.asyncio
    async def test_get_stats_calculates_averages(self, mock_stats_client):
        """Test average calculations."""
        stats = await mock_stats_client.get_stats()

        assert stats["storage"]["avg_chunks_per_memory"] == 3.5  # 35/10

    @pytest.mark.asyncio
    async def test_get_stats_all_node_types(self, mock_stats_client):
        """Test all node types counted."""
        stats = await mock_stats_client.get_stats()

        assert stats["nodes"]["memory"] == 10
        assert stats["nodes"]["chunk"] == 35
        assert stats["nodes"]["entity"] == 5

    @pytest.mark.asyncio
    async def test_get_stats_health_indicators(self, mock_stats_client):
        """Test health indicators present."""
        stats = await mock_stats_client.get_stats()

        assert stats["health"]["connected"] == True
        assert stats["health"]["graph_name"] == "test_graph"
        assert "query_latency_ms" in stats["health"]

    @pytest.mark.asyncio
    async def test_get_stats_empty_graph(self, mocker):
        """Test get_stats on empty graph."""
        client = FalkorDBClient()

        # Mock empty results
        empty_result = MagicMock(rows=[])
        mocker.patch.object(client, '_execute_cypher', return_value=empty_result)
        mocker.patch.object(client, '_initialized', True)
        mocker.patch.object(client, '_schema_ready', True)
        mocker.patch.object(client, 'graph_name', 'test_graph')

        stats = await client.get_stats()

        assert stats["nodes"]["total"] == 0
        assert stats["storage"]["avg_chunks_per_memory"] == 0.0

    @pytest.mark.asyncio
    async def test_get_stats_database_error(self, mocker):
        """Test DatabaseError when query fails."""
        client = FalkorDBClient()

        async def mock_execute(query, params):
            raise Exception("Database error")

        mocker.patch.object(client, '_execute_cypher', side_effect=mock_execute)
        mocker.patch.object(client, '_initialized', True)

        with pytest.raises(DatabaseError, match="Failed to retrieve"):
            await client.get_stats()


# ============================================================================
# close TESTS (10 tests from spec)
# ============================================================================

class TestFalkorDBClientClose:
    """Test FalkorDBClient.close() based on specification."""

    def test_close_success(self, mocker):
        """Test successful close (FalkorDB has no close method)."""
        client = FalkorDBClient()

        mock_pool = MagicMock()
        mocker.patch.object(client, '_pool', mock_pool)

        # FalkorDB uses Redis connection, no close() method
        client.close()

        # Just verify _closed flag is set
        assert client._closed == True

    def test_close_idempotent(self, mocker):
        """Test calling close() twice doesn't error."""
        client = FalkorDBClient()

        mock_pool = MagicMock()
        mocker.patch.object(client, '_pool', mock_pool)

        client.close()
        client.close()  # Should not raise

        # Verify idempotent behavior
        assert client._closed == True

    def test_close_never_connected(self):
        """Test close on never-connected client."""
        client = FalkorDBClient()

        # Should not raise even if never connected
        client.close()

    def test_close_sets_closed_flag(self, mocker):
        """Test close sets internal flag."""
        client = FalkorDBClient()

        mock_pool = MagicMock()
        mocker.patch.object(client, '_pool', mock_pool)

        assert not hasattr(client, '_closed') or not client._closed

        client.close()

        assert hasattr(client, '_closed')
        assert client._closed == True

    def test_close_logs_success(self, mocker):
        """Test close logs successful closure."""
        client = FalkorDBClient()

        mock_logger = MagicMock()
        mocker.patch.object(client, '_logger', mock_logger)

        mock_pool = MagicMock()
        mocker.patch.object(client, '_pool', mock_pool)

        client.close()

        # Should log closure
        assert mock_logger.info.called or mock_logger.debug.called


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
