"""
Unit tests for GraphExporter component.

Tests cover:
- Initialization and validation
- GraphML export with XML structure validation
- Cytoscape JSON export with schema validation
- Neo4j Cypher export with statement validation
- Simple JSON export
- Error handling (database errors, file write errors)
- Empty graph handling
- Output path validation
- Node type filtering
"""

import json
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from zapomni_core.exceptions import (
    DatabaseError,
    ProcessingError,
    ValidationError,
)
from zapomni_core.graph.graph_exporter import (
    ExportResult,
    GraphData,
    GraphExporter,
)
from zapomni_db import FalkorDBClient
from zapomni_db.models import QueryResult

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_db_client():
    """Create mock FalkorDBClient."""
    client = Mock(spec=FalkorDBClient)
    client.graph_query = AsyncMock()
    return client


@pytest.fixture
def graph_exporter(mock_db_client):
    """Create GraphExporter with mock client."""
    return GraphExporter(db_client=mock_db_client)


@pytest.fixture
def sample_graph_data():
    """Sample graph data for testing."""
    return GraphData(
        nodes=[
            {
                "id": "node-1",
                "labels": ["Entity"],
                "properties": {
                    "name": "Python",
                    "type": "TECHNOLOGY",
                    "confidence": 0.95,
                },
            },
            {
                "id": "node-2",
                "labels": ["Entity"],
                "properties": {
                    "name": "Guido van Rossum",
                    "type": "PERSON",
                    "confidence": 0.90,
                },
            },
            {
                "id": "node-3",
                "labels": ["Memory"],
                "properties": {
                    "text": "Python programming language",
                    "created_at": "2025-01-01T00:00:00Z",
                },
            },
        ],
        edges=[
            {
                "id": "edge-1",
                "source": "node-2",
                "target": "node-1",
                "type": "CREATED",
                "properties": {"strength": 0.95, "context": "created Python"},
            },
            {
                "id": "edge-2",
                "source": "node-3",
                "target": "node-1",
                "type": "MENTIONS",
                "properties": {"strength": 0.88},
            },
        ],
        metadata={
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "node_count": 3,
            "edge_count": 2,
            "node_types_filter": [],
        },
    )


@pytest.fixture
def empty_graph_data():
    """Empty graph data for testing."""
    return GraphData(
        nodes=[],
        edges=[],
        metadata={
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "node_count": 0,
            "edge_count": 0,
            "node_types_filter": [],
        },
    )


@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# Test Initialization
# ============================================================================


class TestGraphExporterInit:
    """Test GraphExporter initialization."""

    def test_init_success(self, mock_db_client):
        """Test successful initialization."""
        exporter = GraphExporter(db_client=mock_db_client)

        assert exporter.db_client is mock_db_client
        assert exporter.logger is not None

    def test_init_fails_with_none_client(self):
        """Test initialization fails with None client."""
        with pytest.raises(ValueError, match="db_client cannot be None"):
            GraphExporter(db_client=None)

    def test_init_fails_with_invalid_type(self):
        """Test initialization fails with invalid type."""
        with pytest.raises(TypeError, match="db_client must be FalkorDBClient"):
            GraphExporter(db_client="invalid")


# ============================================================================
# Test GraphML Export
# ============================================================================


class TestExportGraphML:
    """Test export_graphml method."""

    @pytest.mark.asyncio
    async def test_export_graphml_success(
        self, graph_exporter, mock_db_client, sample_graph_data, temp_dir
    ):
        """Test successful GraphML export."""
        # Mock database response
        mock_db_client.graph_query.side_effect = [
            # First call: nodes
            QueryResult(
                rows=[
                    {
                        "n": Mock(
                            properties=sample_graph_data.nodes[0]["properties"],
                            labels=sample_graph_data.nodes[0]["labels"],
                        )
                    },
                    {
                        "n": Mock(
                            properties=sample_graph_data.nodes[1]["properties"],
                            labels=sample_graph_data.nodes[1]["labels"],
                        )
                    },
                ],
                row_count=2,
                execution_time_ms=10,
            ),
            # Second call: edges
            QueryResult(
                rows=[
                    {
                        "r": Mock(
                            properties=sample_graph_data.edges[0]["properties"],
                            relation=sample_graph_data.edges[0]["type"],
                        ),
                        "source_id": sample_graph_data.edges[0]["source"],
                        "target_id": sample_graph_data.edges[0]["target"],
                    },
                ],
                row_count=1,
                execution_time_ms=5,
            ),
        ]

        # Export to GraphML
        output_path = temp_dir / "test.graphml"
        result = await graph_exporter.export_graphml(
            output_path=str(output_path), options={"pretty_print": True}
        )

        # Verify result
        assert isinstance(result, ExportResult)
        assert result.format == "graphml"
        assert result.output_path == str(output_path)
        assert result.nodes_count == 2
        assert result.edges_count == 1
        assert result.file_size_bytes > 0
        assert result.export_time_ms > 0

        # Verify file exists
        assert output_path.exists()

        # Parse and validate XML structure
        tree = ET.parse(output_path)
        root = tree.getroot()

        # Check root tag contains graphml
        assert "graphml" in root.tag
        # Namespace is embedded in tag, e.g. {http://graphml.graphdrawing.org/xmlns}graphml

        # Check for graph element (namespace-aware search)
        graph = root.find(".//{http://graphml.graphdrawing.org/xmlns}graph") or root.find(
            ".//graph"
        )
        assert graph is not None

        # Check for nodes
        nodes = graph.findall(".//{http://graphml.graphdrawing.org/xmlns}node") or graph.findall(
            ".//node"
        )
        assert len(nodes) >= 1

        # Check for edges
        edges = graph.findall(".//{http://graphml.graphdrawing.org/xmlns}edge") or graph.findall(
            ".//edge"
        )
        assert len(edges) >= 1

    @pytest.mark.asyncio
    async def test_export_graphml_invalid_extension(self, graph_exporter, temp_dir):
        """Test GraphML export with invalid extension."""
        output_path = temp_dir / "test.json"

        with pytest.raises(ValidationError, match="must have .graphml extension"):
            await graph_exporter.export_graphml(output_path=str(output_path))

    @pytest.mark.asyncio
    async def test_export_graphml_invalid_directory(self, graph_exporter):
        """Test GraphML export with non-existent directory."""
        output_path = "/nonexistent/directory/test.graphml"

        with pytest.raises(ValidationError, match="Parent directory does not exist"):
            await graph_exporter.export_graphml(output_path=output_path)

    @pytest.mark.asyncio
    async def test_export_graphml_empty_graph(
        self, graph_exporter, mock_db_client, empty_graph_data, temp_dir
    ):
        """Test GraphML export with empty graph."""
        # Mock empty database response
        mock_db_client.graph_query.side_effect = [
            QueryResult(rows=[], row_count=0, execution_time_ms=1),
            QueryResult(rows=[], row_count=0, execution_time_ms=1),
        ]

        output_path = temp_dir / "empty.graphml"
        result = await graph_exporter.export_graphml(output_path=str(output_path))

        assert result.nodes_count == 0
        assert result.edges_count == 0
        assert output_path.exists()

    @pytest.mark.asyncio
    async def test_export_graphml_database_error(self, graph_exporter, mock_db_client, temp_dir):
        """Test GraphML export with database error."""
        mock_db_client.graph_query.side_effect = DatabaseError("Connection failed")

        output_path = temp_dir / "test.graphml"

        with pytest.raises(DatabaseError, match="Failed to fetch graph data"):
            await graph_exporter.export_graphml(output_path=str(output_path))


# ============================================================================
# Test Cytoscape Export
# ============================================================================


class TestExportCytoscape:
    """Test export_cytoscape method."""

    @pytest.mark.asyncio
    async def test_export_cytoscape_success(
        self, graph_exporter, mock_db_client, sample_graph_data, temp_dir
    ):
        """Test successful Cytoscape export."""
        # Mock database response
        mock_db_client.graph_query.side_effect = [
            QueryResult(
                rows=[
                    {
                        "n": Mock(
                            properties=sample_graph_data.nodes[0]["properties"],
                            labels=sample_graph_data.nodes[0]["labels"],
                        )
                    },
                ],
                row_count=1,
                execution_time_ms=5,
            ),
            QueryResult(
                rows=[
                    {
                        "r": Mock(
                            properties=sample_graph_data.edges[0]["properties"],
                            relation=sample_graph_data.edges[0]["type"],
                        ),
                        "source_id": sample_graph_data.edges[0]["source"],
                        "target_id": sample_graph_data.edges[0]["target"],
                    },
                ],
                row_count=1,
                execution_time_ms=5,
            ),
        ]

        output_path = temp_dir / "test.json"
        result = await graph_exporter.export_cytoscape(
            output_path=str(output_path),
            options={"pretty_print": True, "include_style": True},
        )

        # Verify result
        assert result.format == "cytoscape"
        assert result.nodes_count == 1
        assert result.edges_count == 1

        # Verify file exists and is valid JSON
        assert output_path.exists()
        with output_path.open("r") as f:
            data = json.load(f)

        # Validate Cytoscape JSON structure
        assert "elements" in data
        assert "nodes" in data["elements"]
        assert "edges" in data["elements"]
        assert "style" in data
        assert "metadata" in data

        # Check nodes structure
        assert len(data["elements"]["nodes"]) >= 1
        node = data["elements"]["nodes"][0]
        assert "data" in node
        assert "id" in node["data"]
        assert "label" in node["data"]

    @pytest.mark.asyncio
    async def test_export_cytoscape_without_style(self, graph_exporter, mock_db_client, temp_dir):
        """Test Cytoscape export without style hints."""
        mock_db_client.graph_query.side_effect = [
            QueryResult(rows=[], row_count=0, execution_time_ms=1),
            QueryResult(rows=[], row_count=0, execution_time_ms=1),
        ]

        output_path = temp_dir / "nostyle.json"
        result = await graph_exporter.export_cytoscape(
            output_path=str(output_path), options={"include_style": False}
        )

        with output_path.open("r") as f:
            data = json.load(f)

        assert "style" not in data or len(data["style"]) == 0

    @pytest.mark.asyncio
    async def test_export_cytoscape_invalid_extension(self, graph_exporter, temp_dir):
        """Test Cytoscape export with invalid extension."""
        output_path = temp_dir / "test.graphml"

        with pytest.raises(ValidationError, match="must have .json extension"):
            await graph_exporter.export_cytoscape(output_path=str(output_path))


# ============================================================================
# Test Neo4j Export
# ============================================================================


class TestExportNeo4j:
    """Test export_neo4j method."""

    @pytest.mark.asyncio
    async def test_export_neo4j_success(
        self, graph_exporter, mock_db_client, sample_graph_data, temp_dir
    ):
        """Test successful Neo4j Cypher export."""
        mock_db_client.graph_query.side_effect = [
            QueryResult(
                rows=[
                    {
                        "n": Mock(
                            properties=sample_graph_data.nodes[0]["properties"],
                            labels=sample_graph_data.nodes[0]["labels"],
                        )
                    },
                ],
                row_count=1,
                execution_time_ms=5,
            ),
            QueryResult(
                rows=[
                    {
                        "r": Mock(
                            properties=sample_graph_data.edges[0]["properties"],
                            relation=sample_graph_data.edges[0]["type"],
                        ),
                        "source_id": sample_graph_data.edges[0]["source"],
                        "target_id": sample_graph_data.edges[0]["target"],
                    },
                ],
                row_count=1,
                execution_time_ms=5,
            ),
        ]

        output_path = temp_dir / "test.cypher"
        result = await graph_exporter.export_neo4j(
            output_path=str(output_path), options={"batch_size": 500}
        )

        # Verify result
        assert result.format == "neo4j"
        assert result.nodes_count == 1
        assert result.edges_count == 1

        # Verify file exists
        assert output_path.exists()

        # Read and validate Cypher content
        content = output_path.read_text()
        assert "CREATE" in content
        assert "MATCH" in content

        # Check for header comments
        assert "Zapomni Knowledge Graph Export" in content
        assert "Exported:" in content

    @pytest.mark.asyncio
    async def test_export_neo4j_invalid_extension(self, graph_exporter, temp_dir):
        """Test Neo4j export with invalid extension."""
        output_path = temp_dir / "test.json"

        with pytest.raises(ValidationError, match="must have .cypher extension"):
            await graph_exporter.export_neo4j(output_path=str(output_path))

    @pytest.mark.asyncio
    async def test_export_neo4j_with_relationships(
        self, graph_exporter, mock_db_client, sample_graph_data, temp_dir
    ):
        """Test Neo4j export includes relationships."""
        mock_db_client.graph_query.side_effect = [
            QueryResult(
                rows=[
                    {
                        "n": Mock(
                            properties={"id": "n1", "name": "Test"},
                            labels=["Entity"],
                        )
                    },
                ],
                row_count=1,
                execution_time_ms=5,
            ),
            QueryResult(
                rows=[
                    {
                        "r": Mock(properties={"strength": 0.9}, relation="USES"),
                        "source_id": "n1",
                        "target_id": "n2",
                    },
                ],
                row_count=1,
                execution_time_ms=5,
            ),
        ]

        output_path = temp_dir / "relationships.cypher"
        result = await graph_exporter.export_neo4j(output_path=str(output_path))

        content = output_path.read_text()

        # Verify relationship creation
        assert "USES" in content
        assert "strength" in content


# ============================================================================
# Test JSON Export
# ============================================================================


class TestExportJSON:
    """Test export_json method."""

    @pytest.mark.asyncio
    async def test_export_json_success(
        self, graph_exporter, mock_db_client, sample_graph_data, temp_dir
    ):
        """Test successful JSON export."""
        mock_db_client.graph_query.side_effect = [
            QueryResult(
                rows=[
                    {
                        "n": Mock(
                            properties=sample_graph_data.nodes[0]["properties"],
                            labels=sample_graph_data.nodes[0]["labels"],
                        )
                    },
                ],
                row_count=1,
                execution_time_ms=5,
            ),
            QueryResult(
                rows=[
                    {
                        "r": Mock(
                            properties=sample_graph_data.edges[0]["properties"],
                            relation=sample_graph_data.edges[0]["type"],
                        ),
                        "source_id": sample_graph_data.edges[0]["source"],
                        "target_id": sample_graph_data.edges[0]["target"],
                    },
                ],
                row_count=1,
                execution_time_ms=5,
            ),
        ]

        output_path = temp_dir / "test.json"
        result = await graph_exporter.export_json(
            output_path=str(output_path), options={"pretty_print": True}
        )

        # Verify result
        assert result.format == "json"
        assert result.nodes_count == 1
        assert result.edges_count == 1

        # Verify file exists and is valid JSON
        assert output_path.exists()
        with output_path.open("r") as f:
            data = json.load(f)

        # Validate JSON structure
        assert "nodes" in data
        assert "edges" in data
        assert "metadata" in data

        # Check nodes structure
        assert len(data["nodes"]) >= 1
        node = data["nodes"][0]
        assert "id" in node
        assert "labels" in node
        assert "properties" in node

    @pytest.mark.asyncio
    async def test_export_json_compact(self, graph_exporter, mock_db_client, temp_dir):
        """Test JSON export with compact format."""
        mock_db_client.graph_query.side_effect = [
            QueryResult(rows=[], row_count=0, execution_time_ms=1),
            QueryResult(rows=[], row_count=0, execution_time_ms=1),
        ]

        output_path = temp_dir / "compact.json"
        result = await graph_exporter.export_json(
            output_path=str(output_path), options={"pretty_print": False}
        )

        # Read content and verify it's compact (no extra whitespace)
        content = output_path.read_text()
        assert "\n  " not in content or content.count("\n") < 5

    @pytest.mark.asyncio
    async def test_export_json_invalid_extension(self, graph_exporter, temp_dir):
        """Test JSON export with invalid extension."""
        output_path = temp_dir / "test.cypher"

        with pytest.raises(ValidationError, match="must have .json extension"):
            await graph_exporter.export_json(output_path=str(output_path))


# ============================================================================
# Test Error Handling
# ============================================================================


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_empty_output_path(self, graph_exporter):
        """Test export with empty output path."""
        with pytest.raises(ValidationError, match="output_path cannot be empty"):
            await graph_exporter.export_graphml(output_path="")

    @pytest.mark.asyncio
    async def test_database_connection_error(self, graph_exporter, mock_db_client, temp_dir):
        """Test export with database connection error."""
        mock_db_client.graph_query.side_effect = DatabaseError("Connection lost")

        output_path = temp_dir / "test.json"

        with pytest.raises(DatabaseError):
            await graph_exporter.export_json(output_path=str(output_path))

    @pytest.mark.asyncio
    async def test_file_write_error(self, graph_exporter, mock_db_client):
        """Test export with file write error (read-only directory)."""
        mock_db_client.graph_query.side_effect = [
            QueryResult(rows=[], row_count=0, execution_time_ms=1),
            QueryResult(rows=[], row_count=0, execution_time_ms=1),
        ]

        # Try to write to root (should fail)
        output_path = "/test.json"

        with pytest.raises((ValidationError, ProcessingError, PermissionError)):
            await graph_exporter.export_json(output_path=output_path)


# ============================================================================
# Test Node Type Filtering
# ============================================================================


class TestNodeTypeFiltering:
    """Test node type filtering."""

    @pytest.mark.asyncio
    async def test_filter_by_node_types(self, graph_exporter, mock_db_client, temp_dir):
        """Test export with node type filtering."""
        mock_db_client.graph_query.side_effect = [
            QueryResult(
                rows=[
                    {
                        "n": Mock(
                            properties={"id": "e1", "name": "Python", "type": "TECHNOLOGY"},
                            labels=["Entity"],
                        )
                    },
                ],
                row_count=1,
                execution_time_ms=5,
            ),
            QueryResult(rows=[], row_count=0, execution_time_ms=1),
        ]

        output_path = temp_dir / "filtered.json"
        result = await graph_exporter.export_json(
            output_path=str(output_path), options={"node_types": ["Entity"]}
        )

        # Verify filtering was applied
        assert result.nodes_count == 1

        # Check query was called with filter
        calls = mock_db_client.graph_query.call_args_list
        assert len(calls) == 2
        # First call should have WHERE clause with node type filter
        first_query = calls[0][0][0]
        assert "WHERE" in first_query or "n:Entity" in first_query


# ============================================================================
# Test Helper Methods
# ============================================================================


class TestHelperMethods:
    """Test private helper methods."""

    def test_validate_output_path_success(self, graph_exporter, temp_dir):
        """Test output path validation success."""
        output_path = temp_dir / "test.json"
        validated = graph_exporter._validate_output_path(str(output_path), ".json")

        assert validated.suffix == ".json"
        assert validated.parent.exists()

    def test_validate_output_path_wrong_extension(self, graph_exporter, temp_dir):
        """Test output path validation with wrong extension."""
        output_path = temp_dir / "test.xml"

        with pytest.raises(ValidationError, match="must have .json extension"):
            graph_exporter._validate_output_path(str(output_path), ".json")

    def test_node_to_dict(self, graph_exporter):
        """Test node conversion to dict."""
        mock_node = Mock()
        mock_node.properties = {"id": "n1", "name": "Test"}
        mock_node.labels = ["Entity"]

        result = graph_exporter._node_to_dict(mock_node)

        assert result["id"] == "n1"
        assert result["labels"] == ["Entity"]
        assert result["properties"]["name"] == "Test"

    def test_edge_to_dict(self, graph_exporter):
        """Test edge conversion to dict."""
        mock_edge = Mock()
        mock_edge.properties = {"id": "e1", "strength": 0.9}
        mock_edge.relation = "USES"

        result = graph_exporter._edge_to_dict(mock_edge, "n1", "n2")

        assert result["id"] == "e1"
        assert result["source"] == "n1"
        assert result["target"] == "n2"
        assert result["type"] == "USES"
        assert result["properties"]["strength"] == 0.9
