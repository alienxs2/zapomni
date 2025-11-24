# Architecture Plan: Phase 2 & Phase 3 Implementation

**Document Version:** 1.0
**Created:** 2025-11-24
**Author:** Senior Developer / Lead Architect
**Project:** Zapomni - Local-First MCP Memory Server

---

## Executive Summary

This comprehensive architecture plan details the implementation of **Phase 2 (Enhanced Search)** and **Phase 3 (Code Intelligence)** for the Zapomni project.

### Phase Overview
- **Phase 1 (MVP):** ✅ Complete - Basic memory storage and vector search operational
- **Phase 2 (Enhanced Search):** ~80% core ready → Need 3 MCP tool wrappers
- **Phase 3 (Code Intelligence):** ~70% core ready → Need 1 core component + 4 MCP tool wrappers

### Critical Path
The **graph_exporter.py** component is the critical path item that blocks Phase 3. It must be implemented before the `export_graph` tool can be created.

### Task Distribution
- **3 Simple tools** (complexity: 100-150 lines) → Haiku agent
- **4 Medium tools** (complexity: 150-300 lines) → Haiku agent
- **1 Complex component** (complexity: 400+ lines) → Sonnet agent (graph_exporter.py)

### Estimated Total LOC
- **New Code:** ~2,150 lines
- **Tests:** ~1,800 lines
- **Total:** ~3,950 lines

---

## Table of Contents

1. [Phase 2: Enhanced Search](#phase-2-enhanced-search)
2. [Phase 3: Code Intelligence](#phase-3-code-intelligence)
3. [Critical Component: Graph Exporter](#critical-component-graph-exporter)
4. [Implementation Order & Dependencies](#implementation-order--dependencies)
5. [File Structure](#file-structure)
6. [Testing Strategy](#testing-strategy)
7. [Code Quality Standards](#code-quality-standards)
8. [Integration Guidelines](#integration-guidelines)
9. [Detailed API Specifications](#detailed-api-specifications)
10. [Next Steps](#next-steps)

---

## Phase 2: Enhanced Search

Phase 2 adds knowledge graph capabilities to the existing vector search system, enabling entity extraction, relationship detection, and graph traversal.

### 2.1 Tool: build_graph

**Purpose:** Extract entities from text and build knowledge graph structures.

**Complexity:** Medium (200 lines)

**Core Dependencies:**
- `zapomni_core.extractors.entity_extractor.EntityExtractor`
- `zapomni_core.graph.graph_builder.GraphBuilder`
- `zapomni_db.falkordb_client.FalkorDBClient`

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "text": {
      "type": "string",
      "description": "Text to extract entities from and build graph",
      "minLength": 1,
      "maxLength": 100000
    },
    "options": {
      "type": "object",
      "description": "Optional processing options",
      "properties": {
        "extract_entities": {
          "type": "boolean",
          "description": "Enable entity extraction (default: true)",
          "default": true
        },
        "build_relationships": {
          "type": "boolean",
          "description": "Enable relationship detection (Phase 2 LLM, default: false)",
          "default": false
        },
        "confidence_threshold": {
          "type": "number",
          "description": "Minimum confidence for entities (0.0-1.0, default: 0.7)",
          "minimum": 0.0,
          "maximum": 1.0,
          "default": 0.7
        }
      },
      "additionalProperties": false
    }
  },
  "required": ["text"],
  "additionalProperties": false
}
```

**Response Format:**
```json
{
  "content": [{
    "type": "text",
    "text": "Knowledge graph built successfully.\nEntities: 15\nRelationships: 0\nProcessing time: 1.2s"
  }],
  "isError": false
}
```

**Response Data Structure (internal):**
```python
{
    "entities_count": int,        # Number of entities extracted
    "relationships_count": int,   # Number of relationships created (0 in Phase 1)
    "entities_created": int,      # New entities added to graph
    "entities_merged": int,       # Existing entities updated
    "processing_time_ms": float,  # Total processing time
    "confidence_avg": float       # Average entity confidence score
}
```

**Implementation Notes:**
- Initialize `EntityExtractor` with SpaCy model (en_core_web_sm)
- Initialize `GraphBuilder` with extractor and db_client
- Call `graph_builder.build_graph()` with text
- Handle `NotImplementedError` for relationship detection (Phase 2 stub)
- Format statistics for user-friendly display

**Error Handling:**
- `ValidationError`: Text empty, too long, or invalid options
- `ExtractionError`: Entity extraction failed
- `DatabaseError`: Graph storage failed
- `ProcessingError`: Generic processing error

**Estimated Lines:** ~200

---

### 2.2 Tool: get_related

**Purpose:** Find entities related to a given entity via graph traversal.

**Complexity:** Simple (150 lines)

**Core Dependencies:**
- `zapomni_db.falkordb_client.FalkorDBClient.get_related_entities()`

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "entity_id": {
      "type": "string",
      "description": "UUID of the entity to start traversal from",
      "pattern": "^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    },
    "depth": {
      "type": "integer",
      "description": "Traversal depth (number of hops, 1-5)",
      "minimum": 1,
      "maximum": 5,
      "default": 1
    },
    "limit": {
      "type": "integer",
      "description": "Maximum number of related entities to return",
      "minimum": 1,
      "maximum": 100,
      "default": 20
    }
  },
  "required": ["entity_id"],
  "additionalProperties": false
}
```

**Response Format:**
```json
{
  "content": [{
    "type": "text",
    "text": "Found 5 related entities:\n\n1. Python (TECHNOLOGY) - Distance: 1, Strength: 0.95\n   Description: Programming language\n\n2. asyncio (TECHNOLOGY) - Distance: 2, Strength: 0.87\n   Description: Async framework\n..."
  }],
  "isError": false
}
```

**Response Data Structure (internal):**
```python
{
    "related_entities": List[{
        "entity_id": str,
        "name": str,
        "type": str,
        "description": str,
        "confidence": float,
        "distance": int,           # Number of hops from source
        "relationship_strength": float  # Avg strength of path
    }],
    "count": int,
    "source_entity_id": str
}
```

**Implementation Notes:**
- Validate entity_id as UUID format
- Validate depth and limit ranges
- Call `db_client.get_related_entities(entity_id, depth, limit)`
- Format results with entity type, description, and relationship info
- Sort by relationship strength (descending)

**Error Handling:**
- `ValidationError`: Invalid UUID, depth out of range, limit out of range
- `DatabaseError`: Graph query failed
- Handle case when entity_id not found (return empty list)

**Estimated Lines:** ~150

---

### 2.3 Tool: graph_status

**Purpose:** Display knowledge graph statistics and health metrics.

**Complexity:** Simple (120 lines)

**Core Dependencies:**
- `zapomni_db.falkordb_client.FalkorDBClient.get_stats()` (extended with graph metrics)

**Input Schema:**
```json
{
  "type": "object",
  "properties": {},
  "required": [],
  "additionalProperties": false
}
```

**Response Format:**
```json
{
  "content": [{
    "type": "text",
    "text": "Knowledge Graph Status:\n\nNodes:\n- Total: 156\n- Memories: 42\n- Chunks: 98\n- Entities: 15\n- Documents: 1\n\nRelationships:\n- Total: 140\n- HAS_CHUNK: 98\n- MENTIONS: 42\n- RELATED_TO: 0\n\nEntity Types:\n- TECHNOLOGY: 8\n- PERSON: 4\n- ORG: 3\n\nGraph Health: Healthy"
  }],
  "isError": false
}
```

**Response Data Structure (internal):**
```python
{
    "nodes": {
        "total": int,
        "memory": int,
        "chunk": int,
        "entity": int,
        "document": int
    },
    "relationships": {
        "total": int,
        "has_chunk": int,
        "mentions": int,
        "related_to": int
    },
    "entity_types": {
        "TECHNOLOGY": int,
        "PERSON": int,
        "ORG": int,
        "GPE": int,
        "CONCEPT": int,
        "PRODUCT": int,
        "EVENT": int
    },
    "health": str  # "Healthy", "Warning", "Critical"
}
```

**Implementation Notes:**
- Call `db_client.get_stats()` to get base statistics
- Execute additional Cypher queries for entity type breakdown:
  ```cypher
  MATCH (e:Entity) RETURN e.type AS type, count(e) AS count
  ```
- Calculate health status based on:
  - Nodes > 0: "Healthy"
  - Only memories/chunks, no entities: "Warning"
  - No nodes: "Critical"
- Format as hierarchical display

**Error Handling:**
- `DatabaseError`: Stats retrieval failed
- Handle empty graph gracefully

**Estimated Lines:** ~120

---

### 2.4 Integration: Phase 2

**Files to Modify:**

1. **src/zapomni_mcp/server.py**
   - Import new tools: `BuildGraphTool`, `GetRelatedTool`, `GraphStatusTool`
   - Update `register_all_tools()` to include Phase 2 tools
   - Add 3 new tool registrations

2. **src/zapomni_mcp/tools/__init__.py**
   - Add imports for new tools
   - Update `__all__` export list

3. **src/zapomni_core/memory_processor.py**
   - Enable feature flags in default `ProcessorConfig`:
     ```python
     enable_extraction: bool = True  # Changed from False
     enable_graph: bool = True       # Changed from False
     ```
   - Update docstring to reflect Phase 2 capabilities

4. **pyproject.toml** (if needed)
   - Ensure SpaCy model specified in dependencies
   - Add `en-core-web-sm` if not present

**Feature Flags Configuration:**
```python
# In memory_processor.py initialization or config
ProcessorConfig(
    enable_cache=False,        # Phase 2 feature (optional)
    enable_extraction=True,    # ENABLE for Phase 2
    enable_graph=True,         # ENABLE for Phase 2
    max_text_length=10_000_000,
    batch_size=32,
    search_mode="vector"
)
```

---

## Phase 3: Code Intelligence

Phase 3 adds code repository indexing, graph export capabilities, and memory management tools.

### 3.1 Critical Component: graph_exporter.py

**⚠️ CRITICAL PATH - MUST BE IMPLEMENTED FIRST**

See [Section 3: Critical Component: Graph Exporter](#critical-component-graph-exporter) for full specification.

---

### 3.2 Tool: export_graph

**Purpose:** Export knowledge graph to various formats for visualization and analysis.

**Complexity:** Medium (300 lines)

**Core Dependencies:**
- `zapomni_core.graph.graph_exporter.GraphExporter` ← **DEPENDS ON 3.1**
- `zapomni_db.falkordb_client.FalkorDBClient`

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "format": {
      "type": "string",
      "description": "Export format",
      "enum": ["graphml", "cytoscape", "neo4j", "json"],
      "default": "json"
    },
    "output_path": {
      "type": "string",
      "description": "Absolute path to output file",
      "minLength": 1
    },
    "options": {
      "type": "object",
      "description": "Format-specific options",
      "properties": {
        "include_metadata": {
          "type": "boolean",
          "description": "Include node/edge metadata (default: true)",
          "default": true
        },
        "pretty_print": {
          "type": "boolean",
          "description": "Pretty-print JSON/XML (default: true)",
          "default": true
        },
        "node_types": {
          "type": "array",
          "description": "Filter by node types (empty = all)",
          "items": {"type": "string"},
          "default": []
        }
      }
    }
  },
  "required": ["format", "output_path"],
  "additionalProperties": false
}
```

**Response Format:**
```json
{
  "content": [{
    "type": "text",
    "text": "Graph exported successfully.\nFormat: GraphML\nPath: /home/user/graph.graphml\nNodes: 156\nEdges: 140\nSize: 24.5 KB\nExport time: 0.8s"
  }],
  "isError": false
}
```

**Response Data Structure (internal):**
```python
{
    "format": str,               # Export format used
    "output_path": str,          # Absolute path to exported file
    "nodes_count": int,          # Number of nodes exported
    "edges_count": int,          # Number of edges exported
    "file_size_bytes": int,      # File size in bytes
    "export_time_ms": float      # Export duration
}
```

**Implementation Notes:**
- Initialize `GraphExporter` with db_client
- Validate output_path is absolute and writable
- Route to appropriate export method based on format:
  - `graphml` → `exporter.export_graphml(output_path, options)`
  - `cytoscape` → `exporter.export_cytoscape(output_path, options)`
  - `neo4j` → `exporter.export_neo4j(output_path, options)`
  - `json` → `exporter.export_json(output_path, options)`
- Get file stats after export (size, timestamp)
- Format user-friendly response with metrics

**Error Handling:**
- `ValidationError`: Invalid format, path not absolute, path not writable
- `DatabaseError`: Graph query failed
- `IOError`: File write failed
- `ProcessingError`: Export failed

**Estimated Lines:** ~300

---

### 3.3 Tool: index_codebase

**Purpose:** Index code repository with AST analysis for code search.

**Complexity:** Medium (250 lines)

**Core Dependencies:**
- `zapomni_core.code.repository_indexer.CodeRepositoryIndexer`
- `zapomni_core.memory_processor.MemoryProcessor` (for storing indexed code)

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "repo_path": {
      "type": "string",
      "description": "Absolute path to code repository",
      "minLength": 1
    },
    "languages": {
      "type": "array",
      "description": "Programming languages to index (empty = all)",
      "items": {
        "type": "string",
        "enum": ["python", "javascript", "typescript", "java", "go", "rust", "cpp", "c"]
      },
      "default": []
    },
    "options": {
      "type": "object",
      "description": "Indexing options",
      "properties": {
        "recursive": {
          "type": "boolean",
          "description": "Recursively index subdirectories (default: true)",
          "default": true
        },
        "max_file_size": {
          "type": "integer",
          "description": "Max file size in bytes (default: 10MB)",
          "minimum": 1024,
          "maximum": 100000000,
          "default": 10485760
        },
        "include_tests": {
          "type": "boolean",
          "description": "Include test files (default: false)",
          "default": false
        }
      }
    }
  },
  "required": ["repo_path"],
  "additionalProperties": false
}
```

**Response Format:**
```json
{
  "content": [{
    "type": "text",
    "text": "Repository indexed successfully.\nPath: /home/user/project\nFiles indexed: 156\nFunctions: 342\nClasses: 89\nLanguages: Python (120), JavaScript (36)\nTotal lines: 15,432\nIndexing time: 3.2s"
  }],
  "isError": false
}
```

**Response Data Structure (internal):**
```python
{
    "repo_path": str,
    "files_indexed": int,
    "functions_extracted": int,
    "classes_extracted": int,
    "languages": Dict[str, int],  # language -> file count
    "total_lines": int,
    "indexing_time_ms": float,
    "memories_created": int       # Number of memory nodes created
}
```

**Implementation Notes:**
- Initialize `CodeRepositoryIndexer` with config
- Validate repo_path exists and is directory
- Map language names to file extensions:
  ```python
  LANGUAGE_EXTENSIONS = {
      "python": [".py"],
      "javascript": [".js", ".jsx"],
      "typescript": [".ts", ".tsx"],
      "java": [".java"],
      "go": [".go"],
      "rust": [".rs"],
      "cpp": [".cpp", ".cc", ".cxx"],
      "c": [".c", ".h"]
  }
  ```
- Call `indexer.index_repository(repo_path)` to scan files
- For each code file:
  - Extract AST structure (functions, classes)
  - Create memory for each function/class with code + metadata
  - Store via `memory_processor.add_memory()`
- Aggregate statistics and format response

**Error Handling:**
- `ValidationError`: Invalid path, path not directory, unsupported language
- `ProcessingError`: Repository indexing failed, AST parsing failed
- `DatabaseError`: Memory storage failed

**Estimated Lines:** ~250

---

### 3.4 Tool: delete_memory

**Purpose:** Delete a specific memory by ID (with confirmation).

**Complexity:** Simple (150 lines)

**Core Dependencies:**
- `zapomni_db.falkordb_client.FalkorDBClient.delete_memory()`

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "memory_id": {
      "type": "string",
      "description": "UUID of memory to delete",
      "pattern": "^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    },
    "confirm": {
      "type": "boolean",
      "description": "Explicit confirmation required (must be true)",
      "enum": [true]
    }
  },
  "required": ["memory_id", "confirm"],
  "additionalProperties": false
}
```

**Response Format:**
```json
{
  "content": [{
    "type": "text",
    "text": "Memory deleted successfully.\nMemory ID: abc-123-def\nChunks deleted: 5"
  }],
  "isError": false
}
```

**Response Data Structure (internal):**
```python
{
    "deleted": bool,
    "memory_id": str,
    "chunks_deleted": int
}
```

**Implementation Notes:**
- Validate memory_id as UUID format
- **SAFETY**: Require `confirm=true` explicitly
- Call `db_client.delete_memory(memory_id)`
- Return success or "not found" message
- Log deletion with memory_id and timestamp

**Error Handling:**
- `ValidationError`: Invalid UUID, confirm not true
- `DatabaseError`: Delete operation failed
- Handle not found gracefully (return "Memory not found")

**Safety Features:**
- Require explicit `confirm=true` parameter
- Log all deletions with timestamp
- Return confirmation message with deleted count

**Estimated Lines:** ~150

---

### 3.5 Tool: clear_all

**Purpose:** Clear all memories from the system (with strict confirmation).

**Complexity:** Simple (140 lines)

**Core Dependencies:**
- `zapomni_db.falkordb_client.FalkorDBClient.clear_all()`

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "confirm_phrase": {
      "type": "string",
      "description": "Must match exactly: 'DELETE ALL MEMORIES'",
      "pattern": "^DELETE ALL MEMORIES$"
    }
  },
  "required": ["confirm_phrase"],
  "additionalProperties": false
}
```

**Response Format:**
```json
{
  "content": [{
    "type": "text",
    "text": "All memories cleared.\nDeleted:\n- Memories: 42\n- Chunks: 98\n- Entities: 15\n- Total nodes: 156\n- Total edges: 140"
  }],
  "isError": false
}
```

**Response Data Structure (internal):**
```python
{
    "cleared": bool,
    "nodes_deleted": int,
    "memories_deleted": int,
    "chunks_deleted": int,
    "entities_deleted": int,
    "edges_deleted": int
}
```

**Implementation Notes:**
- **SAFETY**: Require exact phrase match: `"DELETE ALL MEMORIES"`
- Get stats before deletion for response
- Call `db_client.clear_all()`
- Log deletion with timestamp and node counts
- Return detailed breakdown of deleted items

**Error Handling:**
- `ValidationError`: Phrase mismatch (reject with helpful error)
- `DatabaseError`: Clear operation failed

**Safety Features:**
- Require exact phrase match (case-sensitive)
- Get stats before deletion to show what was deleted
- Log all clear operations with timestamp
- Return detailed confirmation message

**Estimated Lines:** ~140

---

### 3.6 Integration: Phase 3

**Files to Modify:**

1. **src/zapomni_mcp/server.py**
   - Import new tools: `ExportGraphTool`, `IndexCodebaseTool`, `DeleteMemoryTool`, `ClearAllTool`
   - Update `register_all_tools()` to include Phase 3 tools
   - Add 4 new tool registrations

2. **src/zapomni_mcp/tools/__init__.py**
   - Add imports for new tools
   - Update `__all__` export list

3. **src/zapomni_core/graph/__init__.py**
   - Add `GraphExporter` import and export
   - Update `__all__` list

---

## Critical Component: Graph Exporter

**Location:** `src/zapomni_core/graph/graph_exporter.py`

**Complexity:** Complex (400-450 lines)

**⚠️ This component BLOCKS the export_graph tool - implement FIRST**

### 3.1.1 Purpose

Export knowledge graph from FalkorDB to various formats for:
- Visualization in tools (Gephi, Cytoscape, Neo4j Browser)
- Backup and data portability
- Integration with other systems
- Analysis and debugging

### 3.1.2 Supported Formats

#### 1. GraphML (XML Format)
- **Use Case:** Import into Gephi, yEd, NetworkX
- **Standard:** W3C GraphML specification
- **Features:**
  - Node attributes (id, labels, properties)
  - Edge attributes (id, type, properties)
  - DTD compliance
  - Schema validation

**Example Output:**
```xml
<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <key id="name" for="node" attr.name="name" attr.type="string"/>
  <key id="type" for="node" attr.name="type" attr.type="string"/>
  <graph id="zapomni_graph" edgedefault="directed">
    <node id="entity-123">
      <data key="name">Python</data>
      <data key="type">TECHNOLOGY</data>
    </node>
    <edge id="e1" source="entity-123" target="entity-456" label="USES"/>
  </graph>
</graphml>
```

#### 2. Cytoscape JSON
- **Use Case:** Import into Cytoscape.js, Cytoscape Desktop
- **Features:**
  - Elements array (nodes + edges)
  - Data properties
  - Position hints
  - Style metadata

**Example Output:**
```json
{
  "elements": {
    "nodes": [
      {
        "data": {
          "id": "entity-123",
          "label": "Python",
          "type": "TECHNOLOGY",
          "confidence": 0.95
        },
        "position": {"x": 100, "y": 200}
      }
    ],
    "edges": [
      {
        "data": {
          "id": "e1",
          "source": "entity-123",
          "target": "entity-456",
          "label": "USES",
          "strength": 0.87
        }
      }
    ]
  },
  "style": [
    {"selector": "node[type='TECHNOLOGY']", "style": {"background-color": "#0074D9"}}
  ]
}
```

#### 3. Neo4j Cypher
- **Use Case:** Import into Neo4j, Memgraph
- **Features:**
  - CREATE statements for nodes
  - CREATE statements for relationships
  - Parameterized queries
  - Transaction batching

**Example Output:**
```cypher
// Nodes
CREATE (n1:Entity {id: 'entity-123', name: 'Python', type: 'TECHNOLOGY', confidence: 0.95});
CREATE (n2:Entity {id: 'entity-456', name: 'asyncio', type: 'TECHNOLOGY', confidence: 0.87});

// Relationships
MATCH (a:Entity {id: 'entity-123'}), (b:Entity {id: 'entity-456'})
CREATE (a)-[:USES {strength: 0.87, context: 'Python uses asyncio'}]->(b);
```

#### 4. Simple JSON
- **Use Case:** Custom processing, debugging, backup
- **Features:**
  - Simple {nodes, edges} structure
  - Full property preservation
  - Easy to parse
  - Compact format

**Example Output:**
```json
{
  "nodes": [
    {
      "id": "entity-123",
      "labels": ["Entity"],
      "properties": {
        "name": "Python",
        "type": "TECHNOLOGY",
        "confidence": 0.95,
        "created_at": "2025-11-24T18:00:00Z"
      }
    }
  ],
  "edges": [
    {
      "id": "e1",
      "source": "entity-123",
      "target": "entity-456",
      "type": "USES",
      "properties": {
        "strength": 0.87,
        "context": "Python uses asyncio"
      }
    }
  ],
  "metadata": {
    "exported_at": "2025-11-24T18:00:00Z",
    "node_count": 156,
    "edge_count": 140,
    "format_version": "1.0"
  }
}
```

### 3.1.3 Class Design

```python
"""
GraphExporter - Export knowledge graphs to various formats.

Supports export to:
- GraphML (XML) for Gephi/yEd
- Cytoscape JSON for Cytoscape.js
- Neo4j Cypher statements
- Simple JSON for custom processing

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass

import structlog

from zapomni_db import FalkorDBClient
from zapomni_core.exceptions import (
    ValidationError,
    DatabaseError,
    ProcessingError,
)


logger = structlog.get_logger(__name__)


@dataclass
class ExportResult:
    """Result of graph export operation."""
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
        metadata: Export metadata
    """
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class GraphExporter:
    """
    Export knowledge graphs to various formats.

    This class handles fetching graph data from FalkorDB and converting
    it to different export formats for visualization and integration.

    Attributes:
        db_client: FalkorDBClient for graph queries
        logger: Structured logger for operations

    Example:
        ```python
        from zapomni_db import FalkorDBClient
        from zapomni_core.graph import GraphExporter

        db = FalkorDBClient(host="localhost", port=6381)
        exporter = GraphExporter(db_client=db)

        # Export to GraphML
        result = await exporter.export_graphml(
            output_path="/home/user/graph.graphml",
            options={"pretty_print": True}
        )
        print(f"Exported {result.nodes_count} nodes, {result.edges_count} edges")
        ```
    """

    def __init__(self, db_client: FalkorDBClient) -> None:
        """
        Initialize GraphExporter.

        Args:
            db_client: FalkorDB client for graph queries

        Raises:
            ValueError: If db_client is None
            TypeError: If db_client is not FalkorDBClient
        """
        if db_client is None:
            raise ValueError("db_client cannot be None")

        if not isinstance(db_client, FalkorDBClient):
            raise TypeError(
                f"db_client must be FalkorDBClient, got {type(db_client).__name__}"
            )

        self.db_client = db_client
        self.logger = logger.bind(component="graph_exporter")

        self.logger.info("graph_exporter_initialized")

    async def export_graphml(
        self,
        output_path: str,
        options: Optional[Dict[str, Any]] = None
    ) -> ExportResult:
        """
        Export graph to GraphML (XML) format.

        Args:
            output_path: Absolute path to output file (.graphml)
            options: Optional export options:
                - pretty_print (bool): Pretty-print XML (default: True)
                - include_metadata (bool): Include node metadata (default: True)
                - node_types (list): Filter by node types (default: all)

        Returns:
            ExportResult with export statistics

        Raises:
            ValidationError: If output_path invalid
            DatabaseError: If graph query fails
            IOError: If file write fails
        """
        # Implementation details in following methods
        pass

    async def export_cytoscape(
        self,
        output_path: str,
        options: Optional[Dict[str, Any]] = None
    ) -> ExportResult:
        """
        Export graph to Cytoscape JSON format.

        Args:
            output_path: Absolute path to output file (.json)
            options: Optional export options:
                - pretty_print (bool): Pretty-print JSON (default: True)
                - include_metadata (bool): Include metadata (default: True)
                - include_style (bool): Include style hints (default: True)

        Returns:
            ExportResult with export statistics

        Raises:
            ValidationError: If output_path invalid
            DatabaseError: If graph query fails
            IOError: If file write fails
        """
        pass

    async def export_neo4j(
        self,
        output_path: str,
        options: Optional[Dict[str, Any]] = None
    ) -> ExportResult:
        """
        Export graph to Neo4j Cypher statements.

        Args:
            output_path: Absolute path to output file (.cypher)
            options: Optional export options:
                - batch_size (int): Nodes per transaction (default: 1000)
                - include_metadata (bool): Include metadata (default: True)

        Returns:
            ExportResult with export statistics

        Raises:
            ValidationError: If output_path invalid
            DatabaseError: If graph query fails
            IOError: If file write fails
        """
        pass

    async def export_json(
        self,
        output_path: str,
        options: Optional[Dict[str, Any]] = None
    ) -> ExportResult:
        """
        Export graph to simple JSON format.

        Args:
            output_path: Absolute path to output file (.json)
            options: Optional export options:
                - pretty_print (bool): Pretty-print JSON (default: True)
                - include_metadata (bool): Include metadata (default: True)

        Returns:
            ExportResult with export statistics

        Raises:
            ValidationError: If output_path invalid
            DatabaseError: If graph query fails
            IOError: If file write fails
        """
        pass

    async def _fetch_full_graph(
        self,
        node_types: Optional[List[str]] = None
    ) -> GraphData:
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

        # Build Cypher query for nodes
        if node_types:
            # Filter by node types
            node_filter = " OR ".join([f"n:{t}" for t in node_types])
            node_query = f"MATCH (n) WHERE {node_filter} RETURN n"
        else:
            # Get all nodes
            node_query = "MATCH (n) RETURN n"

        # Fetch nodes
        node_result = await self.db_client.graph_query(node_query)

        nodes = []
        for row in node_result.rows:
            node = row['n']
            # Convert FalkorDB node to dict
            nodes.append(self._node_to_dict(node))

        # Fetch edges
        if node_types:
            edge_query = f"""
            MATCH (a)-[r]->(b)
            WHERE ({node_filter.replace('n:', 'a:')}) AND ({node_filter.replace('n:', 'b:')})
            RETURN r, a.id AS source_id, b.id AS target_id
            """
        else:
            edge_query = "MATCH (a)-[r]->(b) RETURN r, a.id AS source_id, b.id AS target_id"

        edge_result = await self.db_client.graph_query(edge_query)

        edges = []
        for row in edge_result.rows:
            edge = self._edge_to_dict(row['r'], row['source_id'], row['target_id'])
            edges.append(edge)

        metadata = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "node_count": len(nodes),
            "edge_count": len(edges),
            "node_types_filter": node_types or []
        }

        self.logger.info(
            "graph_fetched",
            nodes=len(nodes),
            edges=len(edges)
        )

        return GraphData(nodes=nodes, edges=edges, metadata=metadata)

    def _node_to_dict(self, node: Any) -> Dict[str, Any]:
        """Convert FalkorDB node to dictionary."""
        # FalkorDB node structure: node.id, node.labels, node.properties
        return {
            "id": node.properties.get("id", str(uuid.uuid4())),
            "labels": list(node.labels),
            "properties": dict(node.properties)
        }

    def _edge_to_dict(self, edge: Any, source_id: str, target_id: str) -> Dict[str, Any]:
        """Convert FalkorDB edge to dictionary."""
        # FalkorDB edge structure: edge.id, edge.relation, edge.properties
        return {
            "id": edge.properties.get("id", str(uuid.uuid4())),
            "source": source_id,
            "target": target_id,
            "type": edge.relation,
            "properties": dict(edge.properties)
        }

    def _validate_output_path(self, path: str, expected_ext: str) -> Path:
        """
        Validate output path.

        Args:
            path: Output path string
            expected_ext: Expected file extension (e.g., ".graphml")

        Returns:
            Validated Path object

        Raises:
            ValidationError: If path is invalid
        """
        if not path:
            raise ValidationError(
                message="output_path cannot be empty",
                error_code="VAL_001"
            )

        path_obj = Path(path).expanduser().resolve()

        # Check extension
        if path_obj.suffix.lower() != expected_ext:
            raise ValidationError(
                message=f"output_path must have {expected_ext} extension",
                error_code="VAL_001",
                details={"path": path, "expected_ext": expected_ext}
            )

        # Check parent directory exists
        if not path_obj.parent.exists():
            raise ValidationError(
                message=f"Parent directory does not exist: {path_obj.parent}",
                error_code="VAL_001",
                details={"path": path}
            )

        # Check writable
        if not path_obj.parent.is_dir():
            raise ValidationError(
                message=f"Parent path is not a directory: {path_obj.parent}",
                error_code="VAL_001"
            )

        return path_obj

    def _get_file_size(self, path: Path) -> int:
        """Get file size in bytes."""
        return path.stat().st_size if path.exists() else 0


# Additional helper methods for each format would be implemented here
# _build_graphml(), _build_cytoscape(), _build_neo4j(), etc.
```

### 3.1.4 Implementation Details

**GraphML Builder:**
```python
def _build_graphml(self, graph_data: GraphData, pretty: bool = True) -> str:
    """Build GraphML XML from graph data."""
    xml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
        '  <!-- Node attribute keys -->',
        '  <key id="name" for="node" attr.name="name" attr.type="string"/>',
        '  <key id="type" for="node" attr.name="type" attr.type="string"/>',
        '  <key id="confidence" for="node" attr.name="confidence" attr.type="double"/>',
        '  <!-- Edge attribute keys -->',
        '  <key id="strength" for="edge" attr.name="strength" attr.type="double"/>',
        '  <graph id="zapomni_graph" edgedefault="directed">',
    ]

    # Add nodes
    for node in graph_data.nodes:
        xml_lines.append(f'    <node id="{node["id"]}">')
        for key, value in node["properties"].items():
            xml_lines.append(f'      <data key="{key}">{value}</data>')
        xml_lines.append('    </node>')

    # Add edges
    for edge in graph_data.edges:
        xml_lines.append(
            f'    <edge id="{edge["id"]}" source="{edge["source"]}" '
            f'target="{edge["target"]}" label="{edge["type"]}">'
        )
        for key, value in edge["properties"].items():
            xml_lines.append(f'      <data key="{key}">{value}</data>')
        xml_lines.append('    </edge>')

    xml_lines.extend(['  </graph>', '</graphml>'])

    return '\n'.join(xml_lines)
```

**Cytoscape Builder:**
```python
def _build_cytoscape(self, graph_data: GraphData) -> Dict[str, Any]:
    """Build Cytoscape JSON from graph data."""
    elements = {
        "nodes": [],
        "edges": []
    }

    # Convert nodes
    for node in graph_data.nodes:
        elements["nodes"].append({
            "data": {
                "id": node["id"],
                "label": node["properties"].get("name", node["id"]),
                **node["properties"]
            }
        })

    # Convert edges
    for edge in graph_data.edges:
        elements["edges"].append({
            "data": {
                "id": edge["id"],
                "source": edge["source"],
                "target": edge["target"],
                "label": edge["type"],
                **edge["properties"]
            }
        })

    return {
        "elements": elements,
        "style": self._get_cytoscape_style(),
        "metadata": graph_data.metadata
    }
```

**Neo4j Builder:**
```python
def _build_neo4j(self, graph_data: GraphData, batch_size: int = 1000) -> str:
    """Build Neo4j Cypher statements from graph data."""
    lines = [
        "// Zapomni Knowledge Graph Export",
        f"// Exported: {graph_data.metadata['exported_at']}",
        f"// Nodes: {len(graph_data.nodes)}, Edges: {len(graph_data.edges)}",
        "",
        "// Create nodes",
    ]

    # Group nodes by label for efficient batching
    nodes_by_label = {}
    for node in graph_data.nodes:
        label = node["labels"][0] if node["labels"] else "Node"
        if label not in nodes_by_label:
            nodes_by_label[label] = []
        nodes_by_label[label].append(node)

    # Create nodes with UNWIND for batching
    for label, nodes in nodes_by_label.items():
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i+batch_size]
            lines.append(f"UNWIND $nodes_{label}_{i} AS node")
            lines.append(f"CREATE (n:{label})")
            lines.append("SET n = node.properties;")
            lines.append("")

    # Create relationships
    lines.append("// Create relationships")
    for edge in graph_data.edges:
        lines.append(
            f"MATCH (a {{id: '{edge['source']}'}}), (b {{id: '{edge['target']}'}})\n"
            f"CREATE (a)-[:{edge['type']} {{{self._format_props(edge['properties'])}}}]->(b);"
        )

    return '\n'.join(lines)
```

### 3.1.5 Error Handling Strategy

```python
try:
    graph_data = await self._fetch_full_graph(node_types)
except DatabaseError as e:
    raise DatabaseError(f"Failed to fetch graph data: {e}")

try:
    xml_content = self._build_graphml(graph_data, pretty_print)
except Exception as e:
    raise ProcessingError(f"Failed to build GraphML: {e}")

try:
    output_path.write_text(xml_content, encoding='utf-8')
except IOError as e:
    raise IOError(f"Failed to write file: {e}")
```

### 3.1.6 Testing Requirements

**Unit Tests:**
```python
# tests/unit/test_graph_exporter.py
- test_init_success()
- test_init_fails_with_none_client()
- test_fetch_full_graph_all_nodes()
- test_fetch_full_graph_filtered()
- test_build_graphml()
- test_build_cytoscape()
- test_build_neo4j()
- test_build_json()
- test_validate_output_path()
- test_export_graphml_success()
- test_export_cytoscape_success()
- test_export_neo4j_success()
- test_export_json_success()
- test_export_graphml_invalid_path()
- test_export_database_error()
- test_export_write_error()
```

**Estimated Lines:** ~450 (including helpers and formatters)

---

## Implementation Order & Dependencies

### Dependency Graph

```
Phase 2 Tools (Parallel):
├── build_graph.py      ✓ Independent
├── get_related.py      ✓ Independent
└── graph_status.py     ✓ Independent

Phase 3 (Sequential for critical path):
1. graph_exporter.py    ← MUST BE FIRST (blocks export_graph)
2. export_graph.py      ← Depends on #1

Phase 3 Tools (Parallel):
├── index_codebase.py   ✓ Independent
├── delete_memory.py    ✓ Independent
└── clear_all.py        ✓ Independent
```

### Critical Path

**BLOCKER:** `graph_exporter.py` must be implemented before `export_graph.py` tool

**Optimal Implementation Order:**

1. **Week 1: Phase 2 Tools (Parallel)**
   - Agent 1: `build_graph.py` (200 lines, 2 days)
   - Agent 2: `get_related.py` (150 lines, 1 day)
   - Agent 3: `graph_status.py` (120 lines, 1 day)
   - Integration: Phase 2 tools + tests (1 day)

2. **Week 2: Critical Path + Phase 3 Parallel**
   - **Sonnet Agent**: `graph_exporter.py` (450 lines, 3 days) ← CRITICAL
   - Haiku Agent 1: `index_codebase.py` (250 lines, 2 days)
   - Haiku Agent 2: `delete_memory.py` + `clear_all.py` (290 lines, 2 days)

3. **Week 3: Export Tool + Integration**
   - Agent 1: `export_graph.py` (300 lines, 2 days) - AFTER graph_exporter.py
   - Integration: Phase 3 tools + tests (2 days)
   - Documentation + final testing (1 day)

### Agent Assignment Strategy

**Sonnet Agent (Complex):**
- `graph_exporter.py` - Complex graph traversal, multiple formats, error handling

**Haiku Agents (Simple/Medium):**
- All MCP tools (simple delegation pattern, well-established)
- Tests (follow existing patterns)

---

## File Structure

### New Files to Create

```
src/
├── zapomni_core/
│   └── graph/
│       └── graph_exporter.py                    # NEW - Phase 3 Core (450 lines)
│
└── zapomni_mcp/
    └── tools/
        ├── build_graph.py                       # NEW - Phase 2 Tool (200 lines)
        ├── get_related.py                       # NEW - Phase 2 Tool (150 lines)
        ├── graph_status.py                      # NEW - Phase 2 Tool (120 lines)
        ├── export_graph.py                      # NEW - Phase 3 Tool (300 lines)
        ├── index_codebase.py                    # NEW - Phase 3 Tool (250 lines)
        ├── delete_memory.py                     # NEW - Phase 3 Tool (150 lines)
        └── clear_all.py                         # NEW - Phase 3 Tool (140 lines)

tests/
├── unit/
│   ├── test_graph_exporter.py                  # NEW - Core tests (300 lines)
│   ├── test_build_graph_tool.py                # NEW - Tool tests (200 lines)
│   ├── test_get_related_tool.py                # NEW - Tool tests (150 lines)
│   ├── test_graph_status_tool.py               # NEW - Tool tests (120 lines)
│   ├── test_export_graph_tool.py               # NEW - Tool tests (250 lines)
│   ├── test_index_codebase_tool.py             # NEW - Tool tests (200 lines)
│   ├── test_delete_memory_tool.py              # NEW - Tool tests (150 lines)
│   └── test_clear_all_tool.py                  # NEW - Tool tests (130 lines)
│
└── integration/
    ├── test_phase2_integration.py              # NEW - Phase 2 flows (400 lines)
    └── test_phase3_integration.py              # NEW - Phase 3 flows (500 lines)
```

### Files to Modify

```
src/
├── zapomni_core/
│   ├── __init__.py                             # MODIFY - Add exports
│   ├── memory_processor.py                     # MODIFY - Enable flags (10 lines)
│   └── graph/
│       └── __init__.py                         # MODIFY - Add GraphExporter
│
└── zapomni_mcp/
    ├── server.py                               # MODIFY - Register tools (40 lines)
    └── tools/
        └── __init__.py                         # MODIFY - Add imports (15 lines)

pyproject.toml                                  # MODIFY - Add SpaCy model (if needed)
```

---

## Testing Strategy

### Unit Test Coverage Target: > 85%

Each tool and component must have comprehensive unit tests covering:

1. **Initialization Tests**
   - Valid initialization
   - Invalid parameter types
   - None parameter handling

2. **Success Path Tests**
   - Minimal valid input
   - Full input with all options
   - Edge cases (empty results, max limits)

3. **Validation Tests**
   - Missing required parameters
   - Invalid parameter types
   - Invalid parameter values
   - Boundary conditions

4. **Error Handling Tests**
   - Database errors
   - Processing errors
   - Validation errors
   - Unexpected errors

5. **Response Formatting Tests**
   - Success response structure
   - Error response structure
   - MCP protocol compliance

### Unit Test Pattern (from existing tools)

```python
"""
Unit tests for {ToolName} MCP tool.

Tests cover:
- Tool initialization and validation
- Successful execution
- Input validation and error handling
- Error response formatting
- Integration with core dependencies
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from pydantic import ValidationError

from zapomni_mcp.tools.{tool_module} import {ToolClass}
from zapomni_core.exceptions import (
    ValidationError as CoreValidationError,
    DatabaseError,
    ProcessingError,
)


class Test{ToolName}Init:
    """Test {ToolName} initialization."""

    def test_init_success(self):
        """Test successful initialization."""
        # Setup
        mock_dependency = Mock()

        # Execute
        tool = {ToolClass}(dependency=mock_dependency)

        # Verify
        assert tool.name == "{tool_name}"
        assert tool.description is not None
        assert tool.input_schema is not None

    def test_init_fails_with_invalid_dependency(self):
        """Test initialization fails with invalid dependency."""
        with pytest.raises(TypeError):
            {ToolClass}(dependency="invalid")


class Test{ToolName}Execute:
    """Test {ToolName}.execute() method."""

    @pytest.fixture
    def mock_dependency(self):
        """Create mock dependency."""
        dependency = Mock()
        dependency.method = AsyncMock(return_value="result")
        return dependency

    @pytest.fixture
    def tool(self, mock_dependency):
        """Create tool with mock dependency."""
        return {ToolClass}(dependency=mock_dependency)

    @pytest.mark.asyncio
    async def test_execute_success_minimal(self, tool, mock_dependency):
        """Test successful execution with minimal input."""
        # Setup
        arguments = {"required_param": "value"}

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is False
        assert "content" in result
        mock_dependency.method.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_validation_error(self, tool):
        """Test execution with validation error."""
        # Setup
        arguments = {}  # Missing required param

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "Error" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_execute_database_error(self, tool, mock_dependency):
        """Test execution with database error."""
        # Setup
        arguments = {"required_param": "value"}
        mock_dependency.method.side_effect = DatabaseError("DB error")

        # Execute
        result = await tool.execute(arguments)

        # Verify
        assert result["isError"] is True
        assert "database" in result["content"][0]["text"].lower()
```

### Integration Test Strategy

**Phase 2 Integration Test (`test_phase2_integration.py`):**
```python
@pytest.mark.asyncio
async def test_phase2_full_workflow(memory_processor):
    """Test complete Phase 2 workflow: add → build_graph → get_related → graph_status."""

    # Step 1: Add memory with entities
    memory_id = await memory_processor.add_memory(
        text="Python is a programming language created by Guido van Rossum. "
             "It uses asyncio for async programming.",
        metadata={"source": "test"}
    )

    # Step 2: Build knowledge graph
    build_tool = BuildGraphTool(memory_processor=memory_processor)
    build_result = await build_tool.execute({
        "text": "Python is a programming language created by Guido van Rossum. "
                "It uses asyncio for async programming.",
        "options": {"extract_entities": True}
    })
    assert build_result["isError"] is False

    # Step 3: Get entities (manually query to find IDs)
    # ... query graph for entity IDs ...

    # Step 4: Get related entities
    related_tool = GetRelatedTool(db_client=memory_processor.db_client)
    related_result = await related_tool.execute({
        "entity_id": python_entity_id,
        "depth": 2,
        "limit": 10
    })
    assert related_result["isError"] is False

    # Step 5: Check graph status
    status_tool = GraphStatusTool(db_client=memory_processor.db_client)
    status_result = await status_tool.execute({})
    assert status_result["isError"] is False
    assert "Entities:" in status_result["content"][0]["text"]
```

**Phase 3 Integration Test (`test_phase3_integration.py`):**
```python
@pytest.mark.asyncio
async def test_phase3_full_workflow(memory_processor, tmp_path):
    """Test complete Phase 3 workflow: index_codebase → export_graph → delete_memory."""

    # Step 1: Index code repository
    index_tool = IndexCodebaseTool(memory_processor=memory_processor)
    index_result = await index_tool.execute({
        "repo_path": "/path/to/test/repo",
        "languages": ["python"],
        "options": {"recursive": True}
    })
    assert index_result["isError"] is False

    # Step 2: Export graph
    export_tool = ExportGraphTool(db_client=memory_processor.db_client)
    export_path = tmp_path / "graph.json"
    export_result = await export_tool.execute({
        "format": "json",
        "output_path": str(export_path)
    })
    assert export_result["isError"] is False
    assert export_path.exists()

    # Step 3: Delete specific memory
    delete_tool = DeleteMemoryTool(db_client=memory_processor.db_client)
    delete_result = await delete_tool.execute({
        "memory_id": some_memory_id,
        "confirm": True
    })
    assert delete_result["isError"] is False

    # Step 4: Clear all (with confirmation)
    clear_tool = ClearAllTool(db_client=memory_processor.db_client)
    clear_result = await clear_tool.execute({
        "confirm_phrase": "DELETE ALL MEMORIES"
    })
    assert clear_result["isError"] is False
```

### Test Coverage Breakdown

| Component | Unit Tests | Integration Tests | Total Lines |
|-----------|-----------|------------------|-------------|
| graph_exporter.py | 300 | - | 300 |
| build_graph.py | 200 | ✓ | 200 |
| get_related.py | 150 | ✓ | 150 |
| graph_status.py | 120 | ✓ | 120 |
| export_graph.py | 250 | ✓ | 250 |
| index_codebase.py | 200 | ✓ | 200 |
| delete_memory.py | 150 | ✓ | 150 |
| clear_all.py | 130 | ✓ | 130 |
| Phase 2 Integration | - | 400 | 400 |
| Phase 3 Integration | - | 500 | 500 |
| **Total** | **1,500** | **900** | **2,400** |

---

## Code Quality Standards

All code must adhere to the following standards:

### 1. Formatting
- **Tool:** Black (already configured in project)
- **Line Length:** 100 characters
- **Imports:** Sorted with isort
- **Run before commit:**
  ```bash
  black src/ tests/
  isort src/ tests/
  ```

### 2. Type Hints
- **Tool:** mypy strict mode
- **Coverage:** 100% for function signatures
- **Examples:**
  ```python
  async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
  def _validate_input(self, text: str) -> None:
  ```

### 3. Docstrings
- **Style:** Google style (follow existing patterns)
- **Required for:**
  - All public classes
  - All public methods
  - All module-level functions
- **Example:**
  ```python
  def export_graphml(self, output_path: str, options: Optional[Dict[str, Any]] = None) -> ExportResult:
      """
      Export graph to GraphML (XML) format.

      Args:
          output_path: Absolute path to output file (.graphml)
          options: Optional export options:
              - pretty_print (bool): Pretty-print XML (default: True)
              - include_metadata (bool): Include node metadata (default: True)

      Returns:
          ExportResult with export statistics

      Raises:
          ValidationError: If output_path invalid
          DatabaseError: If graph query fails
          IOError: If file write fails

      Example:
          >>> exporter = GraphExporter(db_client=db)
          >>> result = await exporter.export_graphml("/tmp/graph.graphml")
          >>> print(f"Exported {result.nodes_count} nodes")
      """
  ```

### 4. Error Handling
- **Use custom exceptions** from `zapomni_core.exceptions`
- **Error hierarchy:**
  ```
  ZapomniError (base)
  ├── ValidationError (input validation)
  ├── ProcessingError (processing failures)
  ├── DatabaseError (DB operations)
  ├── ExtractionError (entity extraction)
  └── SearchError (search operations)
  ```
- **Error response format (MCP):**
  ```python
  return {
      "content": [{
          "type": "text",
          "text": f"Error: {user_friendly_message}"
      }],
      "isError": True
  }
  ```

### 5. Logging
- **Library:** structlog (already configured)
- **Log levels:**
  - `DEBUG`: Detailed diagnostic information
  - `INFO`: General informational messages
  - `WARNING`: Warning messages (non-fatal issues)
  - `ERROR`: Error messages (operation failures)
- **Example:**
  ```python
  logger.info(
      "graph_exported",
      format="graphml",
      nodes=156,
      edges=140,
      file_size_kb=24.5
  )
  ```

### 6. Testing
- **Framework:** pytest + pytest-asyncio
- **Coverage:** > 85% for all new code
- **Run tests:**
  ```bash
  pytest tests/unit/test_new_tool.py -v
  pytest tests/integration/ -v --cov=src
  ```

### 7. Validation
- **Library:** Pydantic (for input schemas)
- **Pattern:**
  ```python
  class ToolRequest(BaseModel):
      param: str = Field(..., min_length=1)
      optional: int = Field(default=10, ge=1, le=100)

      model_config = ConfigDict(extra="forbid")
  ```

---

## Integration Guidelines

### Phase 2 Integration Checklist

**Before Implementation:**
- [ ] Verify EntityExtractor works with SpaCy model
- [ ] Verify GraphBuilder API is stable
- [ ] Verify FalkorDB.get_related_entities() works
- [ ] Create feature branch: `feature/phase2-enhanced-search`

**During Implementation:**
1. **build_graph.py:**
   - [ ] Implement tool class
   - [ ] Add Pydantic request/response models
   - [ ] Add unit tests (200 lines)
   - [ ] Test with real EntityExtractor
   - [ ] Update `tools/__init__.py`

2. **get_related.py:**
   - [ ] Implement tool class
   - [ ] Add Pydantic models
   - [ ] Add unit tests (150 lines)
   - [ ] Test with FalkorDB
   - [ ] Update `tools/__init__.py`

3. **graph_status.py:**
   - [ ] Implement tool class
   - [ ] Add Pydantic models
   - [ ] Add unit tests (120 lines)
   - [ ] Test with FalkorDB stats
   - [ ] Update `tools/__init__.py`

4. **Integration:**
   - [ ] Update `server.py` - register 3 tools
   - [ ] Enable feature flags in `memory_processor.py`
   - [ ] Update `zapomni_core/__init__.py` exports
   - [ ] Create integration test (`test_phase2_integration.py`)
   - [ ] Run full test suite: `pytest tests/ -v`
   - [ ] Verify MCP protocol compliance

**After Implementation:**
- [ ] Code review
- [ ] Test with real data
- [ ] Update CHANGELOG.md
- [ ] Merge to main

### Phase 3 Integration Checklist

**Before Implementation:**
- [ ] **CRITICAL:** Implement `graph_exporter.py` FIRST
- [ ] Verify CodeRepositoryIndexer API
- [ ] Verify delete/clear APIs work
- [ ] Create feature branch: `feature/phase3-code-intelligence`

**During Implementation (Sequential):**
1. **graph_exporter.py (CRITICAL PATH):**
   - [ ] Implement GraphExporter class
   - [ ] Implement `_fetch_full_graph()`
   - [ ] Implement `export_graphml()`
   - [ ] Implement `export_cytoscape()`
   - [ ] Implement `export_neo4j()`
   - [ ] Implement `export_json()`
   - [ ] Add unit tests (300 lines)
   - [ ] Test each export format
   - [ ] Update `graph/__init__.py`
   - [ ] **CHECKPOINT:** Verify exporter works before continuing

2. **export_graph.py (AFTER graph_exporter.py):**
   - [ ] Implement tool class
   - [ ] Add Pydantic models
   - [ ] Add unit tests (250 lines)
   - [ ] Test all formats
   - [ ] Update `tools/__init__.py`

3. **Parallel Tools:**
   - [ ] `index_codebase.py` (250 lines)
   - [ ] `delete_memory.py` (150 lines)
   - [ ] `clear_all.py` (140 lines)
   - [ ] Unit tests for each
   - [ ] Update `tools/__init__.py`

4. **Integration:**
   - [ ] Update `server.py` - register 4 tools
   - [ ] Update `zapomni_core/__init__.py` exports
   - [ ] Create integration test (`test_phase3_integration.py`)
   - [ ] Test export workflow end-to-end
   - [ ] Test code indexing workflow
   - [ ] Test delete/clear safety features
   - [ ] Run full test suite: `pytest tests/ -v`

**After Implementation:**
- [ ] Code review (focus on safety for delete/clear)
- [ ] Security review (delete/clear confirmation)
- [ ] Test with real repositories
- [ ] Update CHANGELOG.md
- [ ] Update documentation
- [ ] Merge to main

---

## Detailed API Specifications

### Phase 2 Tool APIs

#### BuildGraphTool
```python
class BuildGraphTool:
    """Build knowledge graph from text."""

    name = "build_graph"
    description = "Extract entities and build knowledge graph from text"

    input_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "minLength": 1, "maxLength": 100000},
            "options": {
                "type": "object",
                "properties": {
                    "extract_entities": {"type": "boolean", "default": True},
                    "build_relationships": {"type": "boolean", "default": False},
                    "confidence_threshold": {"type": "number", "default": 0.7, "minimum": 0, "maximum": 1}
                }
            }
        },
        "required": ["text"]
    }

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute build_graph tool.

        Flow:
        1. Validate text input
        2. Initialize EntityExtractor with SpaCy
        3. Initialize GraphBuilder with extractor + db_client
        4. Call graph_builder.build_graph(text)
        5. Format statistics for response

        Returns MCP-formatted response with entity/relationship counts.
        """
```

#### GetRelatedTool
```python
class GetRelatedTool:
    """Get entities related to a given entity."""

    name = "get_related"
    description = "Find entities related to a given entity via graph traversal"

    input_schema = {
        "type": "object",
        "properties": {
            "entity_id": {"type": "string", "pattern": "^[0-9a-f-]{36}$"},
            "depth": {"type": "integer", "minimum": 1, "maximum": 5, "default": 1},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20}
        },
        "required": ["entity_id"]
    }

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute get_related tool.

        Flow:
        1. Validate entity_id as UUID
        2. Validate depth and limit ranges
        3. Call db_client.get_related_entities(entity_id, depth, limit)
        4. Format entities with relationship info

        Returns MCP-formatted response with related entities list.
        """
```

#### GraphStatusTool
```python
class GraphStatusTool:
    """Display graph statistics."""

    name = "graph_status"
    description = "View knowledge graph statistics and health metrics"

    input_schema = {
        "type": "object",
        "properties": {},
        "required": []
    }

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute graph_status tool.

        Flow:
        1. Call db_client.get_stats() for base stats
        2. Execute Cypher query for entity type breakdown
        3. Calculate health status
        4. Format hierarchical display

        Returns MCP-formatted response with graph status.
        """
```

### Phase 3 Core API

#### GraphExporter
```python
class GraphExporter:
    """Export knowledge graphs to various formats."""

    async def export_graphml(
        self,
        output_path: str,
        options: Optional[Dict[str, Any]] = None
    ) -> ExportResult:
        """
        Export to GraphML (XML) format.

        Flow:
        1. Validate output_path (.graphml extension)
        2. Fetch graph data with _fetch_full_graph()
        3. Build GraphML XML with _build_graphml()
        4. Write to file
        5. Return ExportResult with stats
        """

    async def export_cytoscape(
        self,
        output_path: str,
        options: Optional[Dict[str, Any]] = None
    ) -> ExportResult:
        """
        Export to Cytoscape JSON format.

        Flow:
        1. Validate output_path (.json extension)
        2. Fetch graph data
        3. Build Cytoscape JSON with _build_cytoscape()
        4. Write to file
        5. Return ExportResult with stats
        """

    async def export_neo4j(
        self,
        output_path: str,
        options: Optional[Dict[str, Any]] = None
    ) -> ExportResult:
        """
        Export to Neo4j Cypher statements.

        Flow:
        1. Validate output_path (.cypher extension)
        2. Fetch graph data
        3. Build Cypher statements with _build_neo4j()
        4. Write to file
        5. Return ExportResult with stats
        """

    async def export_json(
        self,
        output_path: str,
        options: Optional[Dict[str, Any]] = None
    ) -> ExportResult:
        """
        Export to simple JSON format.

        Flow:
        1. Validate output_path (.json extension)
        2. Fetch graph data
        3. Build simple JSON with _build_json()
        4. Write to file
        5. Return ExportResult with stats
        """

    async def _fetch_full_graph(
        self,
        node_types: Optional[List[str]] = None
    ) -> GraphData:
        """
        Fetch complete graph from FalkorDB.

        Queries:
        1. Nodes: MATCH (n) RETURN n (with optional type filter)
        2. Edges: MATCH (a)-[r]->(b) RETURN r, a.id, b.id

        Returns GraphData with nodes, edges, metadata.
        """
```

### Phase 3 Tool APIs

#### ExportGraphTool
```python
class ExportGraphTool:
    """Export knowledge graph to various formats."""

    name = "export_graph"
    description = "Export knowledge graph for visualization and analysis"

    input_schema = {
        "type": "object",
        "properties": {
            "format": {"type": "string", "enum": ["graphml", "cytoscape", "neo4j", "json"]},
            "output_path": {"type": "string", "minLength": 1},
            "options": {
                "type": "object",
                "properties": {
                    "include_metadata": {"type": "boolean", "default": True},
                    "pretty_print": {"type": "boolean", "default": True},
                    "node_types": {"type": "array", "items": {"type": "string"}}
                }
            }
        },
        "required": ["format", "output_path"]
    }

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute export_graph tool.

        Flow:
        1. Validate format and output_path
        2. Initialize GraphExporter(db_client)
        3. Route to appropriate export method
        4. Get file stats (size, etc.)
        5. Format response with metrics

        Returns MCP-formatted response with export stats.
        """
```

#### IndexCodebaseTool
```python
class IndexCodebaseTool:
    """Index code repository with AST analysis."""

    name = "index_codebase"
    description = "Index code repository for semantic code search"

    input_schema = {
        "type": "object",
        "properties": {
            "repo_path": {"type": "string", "minLength": 1},
            "languages": {"type": "array", "items": {"type": "string"}},
            "options": {
                "type": "object",
                "properties": {
                    "recursive": {"type": "boolean", "default": True},
                    "max_file_size": {"type": "integer", "default": 10485760},
                    "include_tests": {"type": "boolean", "default": False}
                }
            }
        },
        "required": ["repo_path"]
    }

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute index_codebase tool.

        Flow:
        1. Validate repo_path exists
        2. Initialize CodeRepositoryIndexer
        3. Call indexer.index_repository(repo_path)
        4. For each code file:
           - Extract functions/classes
           - Create memory nodes via memory_processor
        5. Aggregate statistics

        Returns MCP-formatted response with indexing stats.
        """
```

#### DeleteMemoryTool
```python
class DeleteMemoryTool:
    """Delete a specific memory by ID."""

    name = "delete_memory"
    description = "Delete a memory and its associated chunks"

    input_schema = {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "pattern": "^[0-9a-f-]{36}$"},
            "confirm": {"type": "boolean", "enum": [True]}
        },
        "required": ["memory_id", "confirm"]
    }

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute delete_memory tool.

        Flow:
        1. Validate memory_id as UUID
        2. SAFETY: Require confirm=true
        3. Call db_client.delete_memory(memory_id)
        4. Log deletion with timestamp
        5. Return success or "not found" message

        Returns MCP-formatted response with deletion status.
        """
```

#### ClearAllTool
```python
class ClearAllTool:
    """Clear all memories (with strict confirmation)."""

    name = "clear_all"
    description = "Clear all memories from the system (DESTRUCTIVE)"

    input_schema = {
        "type": "object",
        "properties": {
            "confirm_phrase": {"type": "string", "pattern": "^DELETE ALL MEMORIES$"}
        },
        "required": ["confirm_phrase"]
    }

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute clear_all tool.

        Flow:
        1. SAFETY: Require exact phrase match "DELETE ALL MEMORIES"
        2. Get stats before deletion
        3. Call db_client.clear_all()
        4. Log deletion with timestamp and counts
        5. Return detailed breakdown

        Returns MCP-formatted response with deletion stats.
        """
```

---

## Next Steps for Implementation

### Step 1: Architecture Review Meeting
- Review this plan with team
- Clarify any ambiguities
- Adjust timelines if needed
- Assign agents to tasks

### Step 2: Environment Preparation
- Ensure FalkorDB running (localhost:6381)
- Ensure Ollama running (localhost:11434)
- Ensure SpaCy model installed: `python -m spacy download en_core_web_sm`
- Create feature branches:
  - `feature/phase2-enhanced-search`
  - `feature/phase3-code-intelligence`

### Step 3: Phase 2 Implementation (Week 1)
- **Day 1-2:** Haiku Agent 1 implements `build_graph.py` + tests
- **Day 2:** Haiku Agent 2 implements `get_related.py` + tests
- **Day 2:** Haiku Agent 3 implements `graph_status.py` + tests
- **Day 3:** Integration - register tools, enable flags, integration tests
- **Day 4:** Code review, fixes, merge to main

### Step 4: Phase 3 Critical Path (Week 2)
- **Day 1-3:** Sonnet Agent implements `graph_exporter.py` + tests (CRITICAL)
  - Day 1: Core structure, _fetch_full_graph(), export_json()
  - Day 2: export_graphml(), export_cytoscape()
  - Day 3: export_neo4j(), tests, validation
- **Day 1-2:** Haiku Agent 1 implements `index_codebase.py` + tests (parallel)
- **Day 1-2:** Haiku Agent 2 implements `delete_memory.py` + `clear_all.py` + tests (parallel)

### Step 5: Phase 3 Export Tool (Week 3)
- **Day 1-2:** Haiku Agent implements `export_graph.py` + tests (AFTER graph_exporter.py)
- **Day 3:** Integration - register tools, integration tests
- **Day 4:** Code review, security review (delete/clear)
- **Day 5:** Documentation, final testing, merge to main

### Step 6: Final Testing & Documentation
- Run full test suite: `pytest tests/ -v --cov=src`
- Manual testing with real data
- Update CHANGELOG.md
- Update README.md with Phase 2/3 features
- Create release notes

---

## Summary

This architecture plan provides a comprehensive blueprint for implementing Phase 2 (Enhanced Search) and Phase 3 (Code Intelligence) for the Zapomni project.

**Key Highlights:**
- ✅ **Phase 2:** 3 MCP tools (470 lines) - straightforward delegation to existing core APIs
- ⚠️ **Phase 3 Critical Path:** `graph_exporter.py` (450 lines) - complex component that MUST be implemented first
- ✅ **Phase 3 Tools:** 4 MCP tools (840 lines) - include safety features for destructive operations
- 📊 **Total New Code:** ~2,150 lines + ~2,400 test lines = ~4,550 lines total
- ⏱️ **Timeline:** 3 weeks with parallel execution
- 🎯 **Quality:** > 85% test coverage, type hints, comprehensive error handling

**Critical Success Factors:**
1. Implement `graph_exporter.py` FIRST (blocks `export_graph.py`)
2. Enable Phase 2 feature flags in `memory_processor.py`
3. Add safety confirmations for delete/clear operations
4. Test each export format thoroughly
5. Follow existing code patterns for consistency

This plan is ready to be used by implementing agents. Each tool specification includes enough detail to implement without ambiguity, while maintaining consistency with the existing codebase architecture.
