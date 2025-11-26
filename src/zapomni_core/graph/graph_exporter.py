"""
GraphExporter - Export knowledge graphs to various formats.

Supports export to:
- GraphML (XML) for Gephi/yEd/Cytoscape
- Cytoscape JSON for Cytoscape.js web visualization
- Neo4j Cypher statements for Neo4j import
- Simple JSON for backup and custom processing

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from __future__ import annotations

import json
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from zapomni_core.exceptions import (
    DatabaseError,
    ProcessingError,
    ValidationError,
)
from zapomni_db import FalkorDBClient

logger = structlog.get_logger(__name__)


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class ExportResult:
    """
    Result of graph export operation.

    Attributes:
        format: Export format used (graphml, cytoscape, neo4j, json)
        output_path: Absolute path to exported file
        nodes_count: Number of nodes exported
        edges_count: Number of edges exported
        file_size_bytes: File size in bytes
        export_time_ms: Export duration in milliseconds
    """

    format: str
    output_path: str
    nodes_count: int
    edges_count: int
    file_size_bytes: int
    export_time_ms: float


@dataclass
class GraphData:
    """
    Internal representation of graph data.

    Attributes:
        nodes: List of node dicts with id, labels, properties
        edges: List of edge dicts with id, source, target, type, properties
        metadata: Export metadata (timestamp, counts, filters)
    """

    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    metadata: Dict[str, Any]


# ============================================================================
# GraphExporter Class
# ============================================================================


class GraphExporter:
    """
    Export knowledge graphs to various formats for visualization and analysis.

    This class handles fetching graph data from FalkorDB and converting
    it to different export formats for visualization tools and integration.

    Supported formats:
    1. GraphML (XML) - for Gephi, yEd, NetworkX
    2. Cytoscape JSON - for Cytoscape.js, Cytoscape Desktop
    3. Neo4j Cypher - for Neo4j import
    4. Simple JSON - for backup and custom processing

    Attributes:
        db_client: FalkorDBClient for graph queries
        logger: Structured logger for operations

    Example:
        >>> from zapomni_db import FalkorDBClient
        >>> from zapomni_core.graph import GraphExporter
        >>>
        >>> db = FalkorDBClient(host="localhost", port=6381)
        >>> exporter = GraphExporter(db_client=db)
        >>>
        >>> # Export to GraphML
        >>> result = await exporter.export_graphml(
        ...     output_path="/tmp/graph.graphml",
        ...     options={"pretty_print": True}
        ... )
        >>> print(f"Exported {result.nodes_count} nodes, {result.edges_count} edges")
    """

    def __init__(self, db_client: FalkorDBClient) -> None:
        """
        Initialize GraphExporter.

        Args:
            db_client: FalkorDB client for graph queries

        Raises:
            ValueError: If db_client is None
            TypeError: If db_client is not FalkorDBClient instance
        """
        if db_client is None:
            raise ValueError("db_client cannot be None")

        if not isinstance(db_client, FalkorDBClient):
            raise TypeError(f"db_client must be FalkorDBClient, got {type(db_client).__name__}")

        self.db_client = db_client
        self.logger = logger.bind(component="graph_exporter")

        self.logger.info("graph_exporter_initialized")

    async def export_graphml(
        self, output_path: str, options: Optional[Dict[str, Any]] = None
    ) -> ExportResult:
        """
        Export graph to GraphML (XML) format.

        GraphML is supported by: Gephi, yEd, Cytoscape, Neo4j Desktop, NetworkX.

        Args:
            output_path: Absolute path to output file (.graphml extension)
            options: Optional export options:
                - pretty_print (bool): Pretty-print XML (default: True)
                - include_metadata (bool): Include node metadata (default: True)
                - node_types (list): Filter by node types (default: all)

        Returns:
            ExportResult with export statistics

        Raises:
            ValidationError: If output_path invalid or wrong extension
            DatabaseError: If graph query fails
            IOError: If file write fails

        Example:
            >>> result = await exporter.export_graphml(
            ...     output_path="/tmp/graph.graphml",
            ...     options={"pretty_print": True}
            ... )
            >>> print(f"Exported {result.nodes_count} nodes")
        """
        start_time = datetime.now(timezone.utc)

        # Validate output path
        path_obj = self._validate_output_path(output_path, ".graphml")

        # Parse options
        options = options or {}
        pretty_print = options.get("pretty_print", True)
        include_metadata = options.get("include_metadata", True)
        node_types = options.get("node_types", None)

        try:
            # Fetch full graph
            self.logger.debug("fetching_graph_for_graphml")
            graph_data = await self._fetch_full_graph(node_types=node_types)

            # Build GraphML XML
            xml_content = self._build_graphml(
                graph_data, pretty_print=pretty_print, include_metadata=include_metadata
            )

            # Write to file
            path_obj.write_text(xml_content, encoding="utf-8")

            # Calculate statistics
            file_size = path_obj.stat().st_size
            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            self.logger.info(
                "graphml_export_complete",
                nodes=len(graph_data.nodes),
                edges=len(graph_data.edges),
                file_size_kb=round(file_size / 1024, 2),
            )

            return ExportResult(
                format="graphml",
                output_path=str(path_obj),
                nodes_count=len(graph_data.nodes),
                edges_count=len(graph_data.edges),
                file_size_bytes=file_size,
                export_time_ms=round(elapsed, 2),
            )

        except (ValidationError, DatabaseError):
            raise
        except Exception as e:
            raise ProcessingError(
                message=f"GraphML export failed: {str(e)}",
                error_code="PROC_001",
                original_exception=e,
            )

    async def export_cytoscape(
        self, output_path: str, options: Optional[Dict[str, Any]] = None
    ) -> ExportResult:
        """
        Export graph to Cytoscape JSON format.

        Compatible with Cytoscape.js for web visualization and Cytoscape Desktop.

        Args:
            output_path: Absolute path to output file (.json extension)
            options: Optional export options:
                - pretty_print (bool): Pretty-print JSON (default: True)
                - include_metadata (bool): Include metadata (default: True)
                - include_style (bool): Include style hints (default: True)
                - node_types (list): Filter by node types (default: all)

        Returns:
            ExportResult with export statistics

        Raises:
            ValidationError: If output_path invalid or wrong extension
            DatabaseError: If graph query fails
            IOError: If file write fails

        Example:
            >>> result = await exporter.export_cytoscape(
            ...     output_path="/tmp/graph.json",
            ...     options={"include_style": True}
            ... )
        """
        start_time = datetime.now(timezone.utc)

        # Validate output path
        path_obj = self._validate_output_path(output_path, ".json")

        # Parse options
        options = options or {}
        pretty_print = options.get("pretty_print", True)
        include_metadata = options.get("include_metadata", True)
        include_style = options.get("include_style", True)
        node_types = options.get("node_types", None)

        try:
            # Fetch full graph
            self.logger.debug("fetching_graph_for_cytoscape")
            graph_data = await self._fetch_full_graph(node_types=node_types)

            # Build Cytoscape JSON
            cytoscape_data = self._build_cytoscape(
                graph_data,
                include_metadata=include_metadata,
                include_style=include_style,
            )

            # Write to file
            indent = 2 if pretty_print else None
            with path_obj.open("w", encoding="utf-8") as f:
                json.dump(cytoscape_data, f, indent=indent, ensure_ascii=False)

            # Calculate statistics
            file_size = path_obj.stat().st_size
            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            self.logger.info(
                "cytoscape_export_complete",
                nodes=len(graph_data.nodes),
                edges=len(graph_data.edges),
                file_size_kb=round(file_size / 1024, 2),
            )

            return ExportResult(
                format="cytoscape",
                output_path=str(path_obj),
                nodes_count=len(graph_data.nodes),
                edges_count=len(graph_data.edges),
                file_size_bytes=file_size,
                export_time_ms=round(elapsed, 2),
            )

        except (ValidationError, DatabaseError):
            raise
        except Exception as e:
            raise ProcessingError(
                message=f"Cytoscape export failed: {str(e)}",
                error_code="PROC_001",
                original_exception=e,
            )

    async def export_neo4j(
        self, output_path: str, options: Optional[Dict[str, Any]] = None
    ) -> ExportResult:
        """
        Export graph to Neo4j Cypher statements.

        Creates Cypher CREATE statements for Neo4j import.

        Args:
            output_path: Absolute path to output file (.cypher extension)
            options: Optional export options:
                - batch_size (int): Nodes per transaction (default: 1000)
                - include_metadata (bool): Include metadata (default: True)
                - node_types (list): Filter by node types (default: all)

        Returns:
            ExportResult with export statistics

        Raises:
            ValidationError: If output_path invalid or wrong extension
            DatabaseError: If graph query fails
            IOError: If file write fails

        Example:
            >>> result = await exporter.export_neo4j(
            ...     output_path="/tmp/graph.cypher",
            ...     options={"batch_size": 500}
            ... )
        """
        start_time = datetime.now(timezone.utc)

        # Validate output path
        path_obj = self._validate_output_path(output_path, ".cypher")

        # Parse options
        options = options or {}
        batch_size = options.get("batch_size", 1000)
        include_metadata = options.get("include_metadata", True)
        node_types = options.get("node_types", None)

        try:
            # Fetch full graph
            self.logger.debug("fetching_graph_for_neo4j")
            graph_data = await self._fetch_full_graph(node_types=node_types)

            # Build Neo4j Cypher
            cypher_content = self._build_neo4j(
                graph_data, batch_size=batch_size, include_metadata=include_metadata
            )

            # Write to file
            path_obj.write_text(cypher_content, encoding="utf-8")

            # Calculate statistics
            file_size = path_obj.stat().st_size
            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            self.logger.info(
                "neo4j_export_complete",
                nodes=len(graph_data.nodes),
                edges=len(graph_data.edges),
                file_size_kb=round(file_size / 1024, 2),
            )

            return ExportResult(
                format="neo4j",
                output_path=str(path_obj),
                nodes_count=len(graph_data.nodes),
                edges_count=len(graph_data.edges),
                file_size_bytes=file_size,
                export_time_ms=round(elapsed, 2),
            )

        except (ValidationError, DatabaseError):
            raise
        except Exception as e:
            raise ProcessingError(
                message=f"Neo4j export failed: {str(e)}",
                error_code="PROC_001",
                original_exception=e,
            )

    async def export_json(
        self, output_path: str, options: Optional[Dict[str, Any]] = None
    ) -> ExportResult:
        """
        Export graph to simple JSON format.

        Simple backup format with full graph data in JSON structure.

        Args:
            output_path: Absolute path to output file (.json extension)
            options: Optional export options:
                - pretty_print (bool): Pretty-print JSON (default: True)
                - include_metadata (bool): Include metadata (default: True)
                - node_types (list): Filter by node types (default: all)

        Returns:
            ExportResult with export statistics

        Raises:
            ValidationError: If output_path invalid or wrong extension
            DatabaseError: If graph query fails
            IOError: If file write fails

        Example:
            >>> result = await exporter.export_json(
            ...     output_path="/tmp/graph.json"
            ... )
        """
        start_time = datetime.now(timezone.utc)

        # Validate output path
        path_obj = self._validate_output_path(output_path, ".json")

        # Parse options
        options = options or {}
        pretty_print = options.get("pretty_print", True)
        include_metadata = options.get("include_metadata", True)
        node_types = options.get("node_types", None)

        try:
            # Fetch full graph
            self.logger.debug("fetching_graph_for_json")
            graph_data = await self._fetch_full_graph(node_types=node_types)

            # Build simple JSON
            json_data = self._build_json(graph_data, include_metadata=include_metadata)

            # Write to file
            indent = 2 if pretty_print else None
            with path_obj.open("w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=indent, ensure_ascii=False)

            # Calculate statistics
            file_size = path_obj.stat().st_size
            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            self.logger.info(
                "json_export_complete",
                nodes=len(graph_data.nodes),
                edges=len(graph_data.edges),
                file_size_kb=round(file_size / 1024, 2),
            )

            return ExportResult(
                format="json",
                output_path=str(path_obj),
                nodes_count=len(graph_data.nodes),
                edges_count=len(graph_data.edges),
                file_size_bytes=file_size,
                export_time_ms=round(elapsed, 2),
            )

        except (ValidationError, DatabaseError):
            raise
        except Exception as e:
            raise ProcessingError(
                message=f"JSON export failed: {str(e)}",
                error_code="PROC_001",
                original_exception=e,
            )

    # ========================================================================
    # Private Helper Methods
    # ========================================================================

    async def _fetch_full_graph(self, node_types: Optional[List[str]] = None) -> GraphData:
        """
        Fetch complete graph data from FalkorDB.

        Args:
            node_types: Optional list of node types to filter (e.g., ["Entity", "Memory"])

        Returns:
            GraphData with nodes, edges, metadata

        Raises:
            DatabaseError: If graph query fails
        """
        self.logger.debug("fetching_full_graph", node_types=node_types)

        try:
            # Build node query with optional type filter
            if node_types:
                # Filter by node labels (types)
                node_filter = " OR ".join([f"n:{t}" for t in node_types])
                node_query = f"MATCH (n) WHERE {node_filter} RETURN n"
            else:
                # Get all nodes
                node_query = "MATCH (n) RETURN n"

            # Fetch nodes
            node_result = await self.db_client.graph_query(node_query)

            nodes = []
            for row in node_result.rows:
                node = row["n"]
                nodes.append(self._node_to_dict(node))

            # Build edge query with optional type filter
            if node_types:
                edge_query = f"""
                MATCH (a)-[r]->(b)
                WHERE ({node_filter.replace('n:', 'a:')}) AND ({node_filter.replace('n:', 'b:')})
                RETURN r, a.id AS source_id, b.id AS target_id
                """
            else:
                edge_query = "MATCH (a)-[r]->(b) RETURN r, a.id AS source_id, b.id AS target_id"

            # Fetch edges
            edge_result = await self.db_client.graph_query(edge_query)

            edges = []
            for row in edge_result.rows:
                edge = self._edge_to_dict(row["r"], row["source_id"], row["target_id"])
                edges.append(edge)

            metadata = {
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "node_count": len(nodes),
                "edge_count": len(edges),
                "node_types_filter": node_types or [],
            }

            self.logger.info("graph_fetched", nodes=len(nodes), edges=len(edges))

            return GraphData(nodes=nodes, edges=edges, metadata=metadata)

        except Exception as e:
            raise DatabaseError(f"Failed to fetch graph data: {str(e)}", original_exception=e)

    def _node_to_dict(self, node: Any) -> Dict[str, Any]:
        """
        Convert FalkorDB node to dictionary.

        Args:
            node: FalkorDB node object

        Returns:
            Dict with id, labels, properties
        """
        # FalkorDB node structure: node.id, node.labels, node.properties
        return {
            "id": node.properties.get("id", str(uuid.uuid4())),
            "labels": list(node.labels) if hasattr(node, "labels") else [],
            "properties": dict(node.properties) if hasattr(node, "properties") else {},
        }

    def _edge_to_dict(self, edge: Any, source_id: str, target_id: str) -> Dict[str, Any]:
        """
        Convert FalkorDB edge to dictionary.

        Args:
            edge: FalkorDB edge object
            source_id: Source node ID
            target_id: Target node ID

        Returns:
            Dict with id, source, target, type, properties
        """
        # FalkorDB edge structure: edge.id, edge.relation, edge.properties
        return {
            "id": (
                edge.properties.get("id", str(uuid.uuid4()))
                if hasattr(edge, "properties")
                else str(uuid.uuid4())
            ),
            "source": source_id,
            "target": target_id,
            "type": edge.relation if hasattr(edge, "relation") else "RELATED",
            "properties": dict(edge.properties) if hasattr(edge, "properties") else {},
        }

    def _validate_output_path(self, path: str, expected_ext: str) -> Path:
        """
        Validate output path and extension.

        Args:
            path: Output path string
            expected_ext: Expected file extension (e.g., ".graphml")

        Returns:
            Validated Path object

        Raises:
            ValidationError: If path is invalid, wrong extension, or not writable
        """
        if not path:
            raise ValidationError(message="output_path cannot be empty", error_code="VAL_001")

        path_obj = Path(path).expanduser().resolve()

        # Check extension
        if path_obj.suffix.lower() != expected_ext:
            raise ValidationError(
                message=f"output_path must have {expected_ext} extension, got {path_obj.suffix}",
                error_code="VAL_001",
                details={"path": path, "expected_ext": expected_ext},
            )

        # Check parent directory exists
        if not path_obj.parent.exists():
            raise ValidationError(
                message=f"Parent directory does not exist: {path_obj.parent}",
                error_code="VAL_001",
                details={"path": path},
            )

        # Check parent is directory
        if not path_obj.parent.is_dir():
            raise ValidationError(
                message=f"Parent path is not a directory: {path_obj.parent}",
                error_code="VAL_001",
            )

        return path_obj

    # ========================================================================
    # Format Builders
    # ========================================================================

    def _build_graphml(
        self, graph_data: GraphData, pretty_print: bool = True, include_metadata: bool = True
    ) -> str:
        """
        Build GraphML XML from graph data.

        Args:
            graph_data: Graph data to export
            pretty_print: Whether to pretty-print XML
            include_metadata: Whether to include metadata

        Returns:
            GraphML XML string
        """
        # Create root element
        root = ET.Element("graphml")
        root.set("xmlns", "http://graphml.graphdrawing.org/xmlns")

        # Define attribute keys
        self._add_graphml_key(root, "name", "node", "string")
        self._add_graphml_key(root, "type", "node", "string")
        self._add_graphml_key(root, "confidence", "node", "double")
        self._add_graphml_key(root, "label", "edge", "string")
        self._add_graphml_key(root, "strength", "edge", "double")

        # Create graph element
        graph = ET.SubElement(root, "graph")
        graph.set("id", "zapomni_knowledge_graph")
        graph.set("edgedefault", "directed")

        # Add nodes
        for node in graph_data.nodes:
            node_elem = ET.SubElement(graph, "node")
            node_elem.set("id", str(node["id"]))

            # Add properties as data elements
            props = node["properties"]
            if "name" in props:
                self._add_graphml_data(node_elem, "name", str(props["name"]))
            if "type" in props:
                self._add_graphml_data(node_elem, "type", str(props["type"]))
            if "confidence" in props:
                self._add_graphml_data(node_elem, "confidence", str(props["confidence"]))

        # Add edges
        for i, edge in enumerate(graph_data.edges):
            edge_elem = ET.SubElement(graph, "edge")
            edge_elem.set("id", str(edge.get("id", f"edge_{i}")))
            edge_elem.set("source", str(edge["source"]))
            edge_elem.set("target", str(edge["target"]))

            # Add edge properties
            self._add_graphml_data(edge_elem, "label", str(edge.get("type", "")))
            if "strength" in edge.get("properties", {}):
                self._add_graphml_data(edge_elem, "strength", str(edge["properties"]["strength"]))

        # Convert to string
        tree = ET.ElementTree(root)
        if pretty_print:
            ET.indent(tree, space="  ")

        # Return XML as string
        return ET.tostring(root, encoding="unicode", xml_declaration=True)

    def _add_graphml_key(
        self, parent: ET.Element, key_id: str, for_type: str, attr_type: str
    ) -> None:
        """Add GraphML key definition."""
        key = ET.SubElement(parent, "key")
        key.set("id", key_id)
        key.set("for", for_type)
        key.set("attr.name", key_id)
        key.set("attr.type", attr_type)

    def _add_graphml_data(self, parent: ET.Element, key: str, value: str) -> None:
        """Add GraphML data element."""
        data = ET.SubElement(parent, "data")
        data.set("key", key)
        data.text = value

    def _build_cytoscape(
        self,
        graph_data: GraphData,
        include_metadata: bool = True,
        include_style: bool = True,
    ) -> Dict[str, Any]:
        """
        Build Cytoscape JSON from graph data.

        Args:
            graph_data: Graph data to export
            include_metadata: Whether to include metadata
            include_style: Whether to include style hints

        Returns:
            Cytoscape JSON dict
        """
        elements: Dict[str, List[Dict[str, Any]]] = {"nodes": [], "edges": []}

        # Convert nodes
        for node in graph_data.nodes:
            props = node["properties"]
            node_data = {
                "id": str(node["id"]),
                "label": props.get("name", node["id"]),
            }
            # Add all properties
            node_data.update(props)

            elements["nodes"].append({"data": node_data})

        # Convert edges
        for edge in graph_data.edges:
            edge_data = {
                "id": str(edge.get("id", str(uuid.uuid4()))),
                "source": str(edge["source"]),
                "target": str(edge["target"]),
                "label": edge.get("type", ""),
            }
            # Add edge properties
            edge_data.update(edge.get("properties", {}))

            elements["edges"].append({"data": edge_data})

        result: Dict[str, Any] = {"elements": elements}

        # Add metadata
        if include_metadata:
            result["metadata"] = graph_data.metadata

        # Add basic style
        if include_style:
            result["style"] = self._get_cytoscape_style()

        return result

    def _get_cytoscape_style(self) -> List[Dict[str, Any]]:
        """Get default Cytoscape style hints."""
        return [
            {
                "selector": "node",
                "style": {
                    "background-color": "#0074D9",
                    "label": "data(label)",
                    "text-valign": "center",
                    "color": "#fff",
                },
            },
            {
                "selector": "edge",
                "style": {
                    "width": 2,
                    "line-color": "#ccc",
                    "target-arrow-color": "#ccc",
                    "target-arrow-shape": "triangle",
                    "curve-style": "bezier",
                    "label": "data(label)",
                },
            },
            {
                "selector": "node[type='PERSON']",
                "style": {"background-color": "#FF4136"},
            },
            {
                "selector": "node[type='ORG']",
                "style": {"background-color": "#2ECC40"},
            },
            {
                "selector": "node[type='TECHNOLOGY']",
                "style": {"background-color": "#B10DC9"},
            },
        ]

    def _build_neo4j(
        self, graph_data: GraphData, batch_size: int = 1000, include_metadata: bool = True
    ) -> str:
        """
        Build Neo4j Cypher statements from graph data.

        Args:
            graph_data: Graph data to export
            batch_size: Nodes per transaction batch
            include_metadata: Whether to include metadata comments

        Returns:
            Cypher statements as string
        """
        lines = []

        # Header comments
        if include_metadata:
            lines.extend(
                [
                    "// Zapomni Knowledge Graph Export",
                    f"// Exported: {graph_data.metadata['exported_at']}",
                    f"// Nodes: {len(graph_data.nodes)}, Edges: {len(graph_data.edges)}",
                    "",
                ]
            )

        # Create nodes
        lines.append("// Create nodes")
        for node in graph_data.nodes:
            label = node["labels"][0] if node["labels"] else "Node"
            props = node["properties"].copy()
            props["id"] = node["id"]

            # Format properties
            props_str = ", ".join([f"{k}: {json.dumps(v)}" for k, v in props.items()])

            lines.append(f"CREATE (:{label} {{{props_str}}});")

        lines.append("")

        # Create relationships
        lines.append("// Create relationships")
        for edge in graph_data.edges:
            rel_type = edge.get("type", "RELATED_TO")
            props = edge.get("properties", {})

            # Format properties
            props_str = ""
            if props:
                props_str = (
                    " {" + ", ".join([f"{k}: {json.dumps(v)}" for k, v in props.items()]) + "}"
                )

            lines.append(
                f"MATCH (a {{id: {json.dumps(edge['source'])}}}), "
                f"(b {{id: {json.dumps(edge['target'])}}})"
            )
            lines.append(f"CREATE (a)-[:{rel_type}{props_str}]->(b);")
            lines.append("")

        return "\n".join(lines)

    def _build_json(self, graph_data: GraphData, include_metadata: bool = True) -> Dict[str, Any]:
        """
        Build simple JSON from graph data.

        Args:
            graph_data: Graph data to export
            include_metadata: Whether to include metadata

        Returns:
            JSON dict with nodes and edges
        """
        result: Dict[str, Any] = {
            "nodes": graph_data.nodes,
            "edges": graph_data.edges,
        }

        if include_metadata:
            result["metadata"] = graph_data.metadata

        return result
