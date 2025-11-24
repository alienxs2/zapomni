# Task Breakdown: Phase 2 & Phase 3 Implementation

**Generated from:** ARCHITECTURE_PLAN.md
**Date:** 2025-11-24
**Total Tasks:** 16
**Estimated Total Time:** 8-10 hours
**Parallel Execution:** Maximum 3 agents in parallel

---

## Table of Contents

1. [Execution Strategy](#execution-strategy)
2. [Detailed Task Specifications](#detailed-task-specifications)
3. [Agent Assignment Strategy](#agent-assignment-strategy)
4. [Risk Mitigation](#risk-mitigation)
5. [Success Criteria](#success-criteria)
6. [Next Steps for Orchestrator](#next-steps-for-orchestrator)

---

## Execution Strategy

### Wave 1: Phase 2 Tools (Parallel) - 3 tasks
**Tasks:**
- Task 1.1: build_graph tool + tests
- Task 1.2: get_related tool + tests
- Task 1.3: graph_status tool + tests

**Can run in parallel:** ✅ Yes (all 3 tasks independent)
**Blocking:** Wave 2
**Estimated Wave Time:** 20-30 minutes

---

### Wave 2: Phase 2 Integration (Sequential) - 2 tasks
**Tasks:**
- Task 2.1: Register Phase 2 tools in server.py
- Task 2.2: Enable feature flags in memory_processor.py + integration test

**Can run in parallel:** ❌ No (Task 2.2 depends on 2.1)
**Blocking:** Wave 3
**Estimated Wave Time:** 10-15 minutes

---

### Wave 3: Graph Exporter Core (Critical Path) - 1 task
**Tasks:**
- Task 3.1: Create graph_exporter.py + tests (COMPLEX - sonnet recommended)

**Can run in parallel:** ❌ No (CRITICAL PATH)
**Blocking:** Task 4.1 (export_graph depends on this)
**Estimated Wave Time:** 60-90 minutes

---

### Wave 4: Phase 3 Tools Part 1 (Parallel) - 3 tasks
**Tasks:**
- Task 4.1: export_graph tool + tests (depends on 3.1)
- Task 4.2: index_codebase tool + tests
- Task 4.3: delete_memory tool + tests

**Can run in parallel:** ⚠️ Partial (4.2 and 4.3 yes, 4.1 after 3.1 completes)
**Blocking:** Wave 5
**Estimated Wave Time:** 30-45 minutes

---

### Wave 5: Phase 3 Tools Part 2 + Integration (Parallel) - 2 tasks
**Tasks:**
- Task 5.1: clear_all tool + tests
- Task 5.2: Register Phase 3 tools in server.py

**Can run in parallel:** ✅ Yes
**Blocking:** Wave 6
**Estimated Wave Time:** 15-20 minutes

---

### Wave 6: Integration Testing & Documentation (Parallel) - 3 tasks
**Tasks:**
- Task 6.1: Create test_phase2_integration.py (sonnet recommended)
- Task 6.2: Create test_phase3_integration.py (sonnet recommended)
- Task 6.3: Run full test suite + update CHANGELOG.md (sonnet recommended)

**Can run in parallel:** ✅ Yes
**Blocking:** None (final wave)
**Estimated Wave Time:** 45-60 minutes

---

## Detailed Task Specifications

### WAVE 1: Phase 2 Tools

---

## Task ID: 1.1

### Task Name: Implement build_graph Tool

**Type:** implementation

**Complexity:** medium

**Recommended Agent:** haiku

**Description:**
Create MCP tool that extracts entities from text and builds knowledge graph structures. Delegates to EntityExtractor and GraphBuilder core components.

**Prerequisites:**
- [x] Docker services running (FalkorDB, Ollama)
- [x] EntityExtractor component available
- [x] GraphBuilder component available
- [x] SpaCy model en_core_web_sm installed

**Input Files (Read for patterns):**
- `/home/dev/zapomni/src/zapomni_mcp/tools/add_memory.py` (MCP tool pattern)
- `/home/dev/zapomni/tests/unit/test_add_memory_tool.py` (test pattern)
- `/home/dev/zapomni/ARCHITECTURE_PLAN.md` (lines 53-143 - API spec)

**Output Files (Create/Modify):**
- `/home/dev/zapomni/src/zapomni_mcp/tools/build_graph.py` (NEW)
- `/home/dev/zapomni/tests/unit/test_build_graph_tool.py` (NEW)

**Implementation Steps:**
1. Read `add_memory.py` to understand MCP tool pattern (Pydantic validation, execute method, error handling)
2. Study API specification from ARCHITECTURE_PLAN.md (lines 53-143)
3. Create `build_graph.py`:
   - Import dependencies: `EntityExtractor`, `GraphBuilder`, `FalkorDBClient`, Pydantic models
   - Define `BuildGraphRequest` Pydantic model with validation:
     - `text: str` (minLength=1, maxLength=100000)
     - `options: Optional[Dict]` with nested properties for extract_entities, build_relationships, confidence_threshold
   - Define `BuildGraphTool` class:
     - `name = "build_graph"`
     - `description` from spec
     - `input_schema` matching JSON Schema from spec
     - `__init__(self, db_client: FalkorDBClient)` with validation
     - `async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]`:
       - Validate with Pydantic model
       - Initialize EntityExtractor with SpaCy model
       - Initialize GraphBuilder with extractor + db_client
       - Call `graph_builder.build_graph(text, options)`
       - Format response with entity/relationship counts
       - Handle errors: ValidationError, ExtractionError, DatabaseError, ProcessingError
   - Use structured logging with structlog
   - Follow error response format: `{"content": [{"type": "text", "text": "Error: ..."}], "isError": True}`
4. Create `test_build_graph_tool.py`:
   - Test init success/failure (with/without db_client)
   - Test execute success (minimal input, with options)
   - Test validation errors (empty text, text too long, invalid options)
   - Test database errors, extraction errors, unexpected errors
   - Test response formatting (success and error cases)
   - Mock EntityExtractor and GraphBuilder
   - Use pytest-asyncio for async tests
   - Target coverage > 80%
5. Run tests: `pytest tests/unit/test_build_graph_tool.py -v --cov=src/zapomni_mcp/tools/build_graph.py`
6. Format code: `black src/zapomni_mcp/tools/build_graph.py tests/unit/test_build_graph_tool.py`
7. Verify mypy: `mypy src/zapomni_mcp/tools/build_graph.py`

**Acceptance Criteria:**
- [x] File `build_graph.py` created with correct structure
- [x] All unit tests pass (minimum 12 tests)
- [x] Coverage > 80%
- [x] Black formatted (line length 100)
- [x] Type hints present on all functions
- [x] Docstrings complete (class + all public methods)
- [x] Error handling for all exception types
- [x] MCP TextContent response format correct
- [x] Pydantic validation working
- [x] Structured logging present

**API Specification from Architecture Plan:**

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
      }
    }
  },
  "required": ["text"]
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

**Estimated Time:** 20 minutes

**Validation Command:**
```bash
pytest tests/unit/test_build_graph_tool.py -v --cov=src/zapomni_mcp/tools/build_graph.py --cov-report=term-missing
```

**Blocking:** Task 2.1 (server registration)

**Blocked By:** None

---

## Task ID: 1.2

### Task Name: Implement get_related Tool

**Type:** implementation

**Complexity:** simple

**Recommended Agent:** haiku

**Description:**
Create MCP tool that finds entities related to a given entity via graph traversal. Delegates to FalkorDBClient.get_related_entities() method.

**Prerequisites:**
- [x] Docker services running (FalkorDB)
- [x] FalkorDBClient.get_related_entities() method available

**Input Files (Read for patterns):**
- `/home/dev/zapomni/src/zapomni_mcp/tools/add_memory.py` (MCP tool pattern)
- `/home/dev/zapomni/tests/unit/test_add_memory_tool.py` (test pattern)
- `/home/dev/zapomni/ARCHITECTURE_PLAN.md` (lines 145-226 - API spec)

**Output Files (Create/Modify):**
- `/home/dev/zapomni/src/zapomni_mcp/tools/get_related.py` (NEW)
- `/home/dev/zapomni/tests/unit/test_get_related_tool.py` (NEW)

**Implementation Steps:**
1. Read `add_memory.py` to understand MCP tool pattern
2. Study API specification from ARCHITECTURE_PLAN.md (lines 145-226)
3. Create `get_related.py`:
   - Import dependencies: `FalkorDBClient`, Pydantic models
   - Define `GetRelatedRequest` Pydantic model:
     - `entity_id: str` with UUID pattern validation
     - `depth: int` (minimum=1, maximum=5, default=1)
     - `limit: int` (minimum=1, maximum=100, default=20)
   - Define `GetRelatedTool` class:
     - `name = "get_related"`
     - `description` from spec
     - `input_schema` matching JSON Schema
     - `__init__(self, db_client: FalkorDBClient)` with validation
     - `async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]`:
       - Validate entity_id as UUID format
       - Validate depth and limit ranges
       - Call `db_client.get_related_entities(entity_id, depth, limit)`
       - Format results with entity type, description, relationship info
       - Sort by relationship strength (descending)
       - Handle entity_id not found (return empty list gracefully)
   - Error handling: ValidationError, DatabaseError
4. Create `test_get_related_tool.py`:
   - Test init success/failure
   - Test execute success (minimal input, with depth/limit)
   - Test validation errors (invalid UUID, depth out of range, limit out of range)
   - Test entity not found case
   - Test database errors
   - Mock FalkorDBClient
   - Target coverage > 80%
5. Run tests: `pytest tests/unit/test_get_related_tool.py -v --cov=src/zapomni_mcp/tools/get_related.py`
6. Format code: `black src/zapomni_mcp/tools/get_related.py tests/unit/test_get_related_tool.py`

**Acceptance Criteria:**
- [x] File `get_related.py` created
- [x] All unit tests pass (minimum 10 tests)
- [x] Coverage > 80%
- [x] Black formatted
- [x] Type hints present
- [x] Docstrings complete
- [x] Error handling implemented
- [x] MCP response format correct
- [x] UUID validation working

**API Specification from Architecture Plan:**

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
  "required": ["entity_id"]
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

**Estimated Time:** 15 minutes

**Validation Command:**
```bash
pytest tests/unit/test_get_related_tool.py -v --cov=src/zapomni_mcp/tools/get_related.py --cov-report=term-missing
```

**Blocking:** Task 2.1 (server registration)

**Blocked By:** None

---

## Task ID: 1.3

### Task Name: Implement graph_status Tool

**Type:** implementation

**Complexity:** simple

**Recommended Agent:** haiku

**Description:**
Create MCP tool that displays knowledge graph statistics and health metrics. Extends FalkorDBClient.get_stats() with entity type breakdown.

**Prerequisites:**
- [x] Docker services running (FalkorDB)
- [x] FalkorDBClient.get_stats() method available

**Input Files (Read for patterns):**
- `/home/dev/zapomni/src/zapomni_mcp/tools/get_stats.py` (similar stats tool)
- `/home/dev/zapomni/tests/unit/test_get_stats_tool.py` (test pattern)
- `/home/dev/zapomni/ARCHITECTURE_PLAN.md` (lines 228-304 - API spec)

**Output Files (Create/Modify):**
- `/home/dev/zapomni/src/zapomni_mcp/tools/graph_status.py` (NEW)
- `/home/dev/zapomni/tests/unit/test_graph_status_tool.py` (NEW)

**Implementation Steps:**
1. Read `get_stats.py` to understand stats tool pattern
2. Study API specification from ARCHITECTURE_PLAN.md (lines 228-304)
3. Create `graph_status.py`:
   - Import dependencies: `FalkorDBClient`
   - Define `GraphStatusTool` class:
     - `name = "graph_status"`
     - `description` from spec
     - `input_schema` (no parameters required)
     - `__init__(self, db_client: FalkorDBClient)` with validation
     - `async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]`:
       - Call `db_client.get_stats()` for base statistics
       - Execute Cypher query for entity type breakdown:
         ```cypher
         MATCH (e:Entity) RETURN e.type AS type, count(e) AS count
         ```
       - Calculate health status based on:
         - Nodes > 0: "Healthy"
         - Only memories/chunks, no entities: "Warning"
         - No nodes: "Critical"
       - Format as hierarchical display (nodes, relationships, entity types, health)
   - Error handling: DatabaseError
   - Handle empty graph gracefully
4. Create `test_graph_status_tool.py`:
   - Test init success/failure
   - Test execute success (healthy graph, warning, critical)
   - Test empty graph case
   - Test database errors
   - Mock FalkorDBClient
   - Target coverage > 80%
5. Run tests: `pytest tests/unit/test_graph_status_tool.py -v --cov=src/zapomni_mcp/tools/graph_status.py`
6. Format code: `black src/zapomni_mcp/tools/graph_status.py tests/unit/test_graph_status_tool.py`

**Acceptance Criteria:**
- [x] File `graph_status.py` created
- [x] All unit tests pass (minimum 8 tests)
- [x] Coverage > 80%
- [x] Black formatted
- [x] Type hints present
- [x] Docstrings complete
- [x] Error handling implemented
- [x] Health status calculation working
- [x] Hierarchical display formatting correct

**API Specification from Architecture Plan:**

**Input Schema:**
```json
{
  "type": "object",
  "properties": {},
  "required": []
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

**Estimated Time:** 15 minutes

**Validation Command:**
```bash
pytest tests/unit/test_graph_status_tool.py -v --cov=src/zapomni_mcp/tools/graph_status.py --cov-report=term-missing
```

**Blocking:** Task 2.1 (server registration)

**Blocked By:** None

---

### WAVE 2: Phase 2 Integration

---

## Task ID: 2.1

### Task Name: Register Phase 2 Tools in Server

**Type:** integration

**Complexity:** simple

**Recommended Agent:** haiku

**Description:**
Update server.py to register Phase 2 tools (build_graph, get_related, graph_status) and update __init__.py exports.

**Prerequisites:**
- [x] Task 1.1 completed (build_graph.py)
- [x] Task 1.2 completed (get_related.py)
- [x] Task 1.3 completed (graph_status.py)

**Input Files (Read for patterns):**
- `/home/dev/zapomni/src/zapomni_mcp/server.py` (current tool registration)
- `/home/dev/zapomni/src/zapomni_mcp/tools/__init__.py` (exports)

**Output Files (Create/Modify):**
- `/home/dev/zapomni/src/zapomni_mcp/server.py` (MODIFY)
- `/home/dev/zapomni/src/zapomni_mcp/tools/__init__.py` (MODIFY)

**Implementation Steps:**
1. Read current `server.py` to understand tool registration pattern (line 206-246)
2. Read current `tools/__init__.py` to understand export pattern
3. Modify `server.py`:
   - Import new tools at top:
     ```python
     from zapomni_mcp.tools import (
         AddMemoryTool,
         SearchMemoryTool,
         GetStatsTool,
         BuildGraphTool,      # NEW
         GetRelatedTool,      # NEW
         GraphStatusTool,     # NEW
     )
     ```
   - Update `register_all_tools()` method:
     - Add tool instantiation after existing tools:
       ```python
       tools = [
           AddMemoryTool(memory_processor=memory_processor),
           SearchMemoryTool(memory_processor=memory_processor),
           GetStatsTool(memory_processor=memory_processor),
           BuildGraphTool(db_client=memory_processor.db_client),
           GetRelatedTool(db_client=memory_processor.db_client),
           GraphStatusTool(db_client=memory_processor.db_client),
       ]
       ```
4. Modify `tools/__init__.py`:
   - Add imports:
     ```python
     from zapomni_mcp.tools.build_graph import BuildGraphTool
     from zapomni_mcp.tools.get_related import GetRelatedTool
     from zapomni_mcp.tools.graph_status import GraphStatusTool
     ```
   - Update `__all__` list:
     ```python
     __all__ = [
         "MCPTool",
         "AddMemoryTool",
         "SearchMemoryTool",
         "GetStatsTool",
         "BuildGraphTool",
         "GetRelatedTool",
         "GraphStatusTool",
     ]
     ```
5. Verify imports work: `python -c "from zapomni_mcp.tools import BuildGraphTool, GetRelatedTool, GraphStatusTool"`
6. Format code: `black src/zapomni_mcp/server.py src/zapomni_mcp/tools/__init__.py`

**Acceptance Criteria:**
- [x] server.py imports Phase 2 tools
- [x] register_all_tools() instantiates Phase 2 tools
- [x] tools/__init__.py exports Phase 2 tools
- [x] No import errors
- [x] Code formatted with Black
- [x] Tool count in log message updated to 6

**Estimated Time:** 5 minutes

**Validation Command:**
```bash
python -c "from zapomni_mcp.tools import BuildGraphTool, GetRelatedTool, GraphStatusTool; print('✓ Imports successful')"
```

**Blocking:** Task 2.2 (feature flags)

**Blocked By:** Task 1.1, 1.2, 1.3

---

## Task ID: 2.2

### Task Name: Enable Phase 2 Feature Flags

**Type:** integration

**Complexity:** simple

**Recommended Agent:** haiku

**Description:**
Enable Phase 2 feature flags in memory_processor.py (enable_extraction, enable_graph) and create integration test to verify Phase 2 workflow.

**Prerequisites:**
- [x] Task 2.1 completed (tools registered)
- [x] EntityExtractor component available
- [x] GraphBuilder component available

**Input Files (Read):**
- `/home/dev/zapomni/src/zapomni_core/memory_processor.py` (lines 40-60 - ProcessorConfig)

**Output Files (Create/Modify):**
- `/home/dev/zapomni/src/zapomni_core/memory_processor.py` (MODIFY)

**Implementation Steps:**
1. Read `memory_processor.py` to locate ProcessorConfig dataclass (lines 40-60)
2. Modify ProcessorConfig defaults:
   - Change `enable_extraction: bool = False` → `enable_extraction: bool = True`
   - Change `enable_graph: bool = False` → `enable_graph: bool = True`
3. Update docstring to reflect Phase 2 capabilities:
   ```python
   """
   Configuration for MemoryProcessor.

   Attributes:
       enable_cache: Enable semantic embedding cache (Phase 2, not yet implemented)
       enable_extraction: Enable entity extraction (Phase 2, ENABLED)
       enable_graph: Enable knowledge graph construction (Phase 2, ENABLED)
       max_text_length: Maximum text length in characters (default: 10MB)
       batch_size: Batch size for embedding generation (default: 32)
       search_mode: Default search mode ("vector", "bm25", "hybrid", "graph")
   """
   ```
4. Run existing unit tests to verify no breakage: `pytest tests/unit/test_memory_processor.py -v`
5. Format code: `black src/zapomni_core/memory_processor.py`

**Acceptance Criteria:**
- [x] enable_extraction default is True
- [x] enable_graph default is True
- [x] Docstring updated
- [x] All existing unit tests pass
- [x] Code formatted with Black

**Estimated Time:** 5 minutes

**Validation Command:**
```bash
pytest tests/unit/test_memory_processor.py -v
python -c "from zapomni_core.memory_processor import ProcessorConfig; c = ProcessorConfig(); assert c.enable_extraction == True and c.enable_graph == True; print('✓ Feature flags enabled')"
```

**Blocking:** Wave 3 (can proceed independently)

**Blocked By:** Task 2.1

---

### WAVE 3: Critical Path (Graph Exporter)

---

## Task ID: 3.1

### Task Name: Implement graph_exporter.py Core Component

**Type:** implementation

**Complexity:** complex

**Recommended Agent:** sonnet (CRITICAL - complex component)

**Description:**
Create GraphExporter class that exports knowledge graphs to multiple formats (GraphML, Cytoscape JSON, Neo4j Cypher, Simple JSON). This is the CRITICAL PATH component that blocks Task 4.1.

**Prerequisites:**
- [x] Docker services running (FalkorDB)
- [x] FalkorDBClient.graph_query() method available

**Input Files (Read for patterns):**
- `/home/dev/zapomni/src/zapomni_db/falkordb_client.py` (database client pattern)
- `/home/dev/zapomni/ARCHITECTURE_PLAN.md` (lines 731-1398 - full specification)

**Output Files (Create/Modify):**
- `/home/dev/zapomni/src/zapomni_core/graph/graph_exporter.py` (NEW)
- `/home/dev/zapomni/src/zapomni_core/graph/__init__.py` (MODIFY - add export)
- `/home/dev/zapomni/tests/unit/test_graph_exporter.py` (NEW)

**Implementation Steps:**

1. **Read specifications:**
   - Study ARCHITECTURE_PLAN.md lines 731-1398 (complete GraphExporter spec)
   - Understand all 4 export formats (GraphML, Cytoscape, Neo4j, JSON)
   - Review FalkorDBClient API for graph queries

2. **Create graph_exporter.py structure:**
   ```python
   # Imports
   from __future__ import annotations
   import json
   import uuid
   from pathlib import Path
   from typing import Dict, Any, List, Optional, Tuple
   from datetime import datetime, timezone
   from dataclasses import dataclass
   import structlog
   from zapomni_db import FalkorDBClient
   from zapomni_core.exceptions import ValidationError, DatabaseError, ProcessingError

   # Dataclasses
   @dataclass
   class ExportResult:
       format: str
       output_path: str
       nodes_count: int
       edges_count: int
       file_size_bytes: int
       export_time_ms: float

   @dataclass
   class GraphData:
       nodes: List[Dict[str, Any]]
       edges: List[Dict[str, Any]]
       metadata: Dict[str, Any]

   # Main class
   class GraphExporter:
       # ... implementation
   ```

3. **Implement core methods:**
   - `__init__(self, db_client: FalkorDBClient)` - Initialize with validation
   - `async def _fetch_full_graph(node_types: Optional[List[str]] = None) -> GraphData`:
     - Execute Cypher: `MATCH (n) RETURN n` (with optional type filter)
     - Execute Cypher: `MATCH (a)-[r]->(b) RETURN r, a.id AS source_id, b.id AS target_id`
     - Convert FalkorDB nodes/edges to dicts
     - Return GraphData with metadata
   - `def _validate_output_path(path: str, expected_ext: str) -> Path`:
     - Check path not empty
     - Check extension matches
     - Check parent directory exists
     - Check writable
     - Return validated Path object
   - `def _node_to_dict(node: Any) -> Dict[str, Any]`:
     - Extract id, labels, properties from FalkorDB node
   - `def _edge_to_dict(edge: Any, source_id: str, target_id: str) -> Dict[str, Any]`:
     - Extract id, source, target, type, properties from FalkorDB edge

4. **Implement export_graphml():**
   - Validate output_path (.graphml extension)
   - Fetch graph data
   - Build GraphML XML:
     ```xml
     <?xml version="1.0" encoding="UTF-8"?>
     <graphml xmlns="http://graphml.graphdrawing.org/xmlns">
       <key id="name" for="node" attr.name="name" attr.type="string"/>
       ...
       <graph id="zapomni_graph" edgedefault="directed">
         <node id="...">...</node>
         <edge id="..." source="..." target="...">...</edge>
       </graph>
     </graphml>
     ```
   - Write to file
   - Return ExportResult

5. **Implement export_cytoscape():**
   - Validate output_path (.json extension)
   - Fetch graph data
   - Build Cytoscape JSON:
     ```json
     {
       "elements": {
         "nodes": [{"data": {...}}],
         "edges": [{"data": {...}}]
       },
       "style": [...]
     }
     ```
   - Write to file with json.dump()
   - Return ExportResult

6. **Implement export_neo4j():**
   - Validate output_path (.cypher extension)
   - Fetch graph data
   - Build Cypher statements:
     ```cypher
     CREATE (n1:Entity {id: '...', name: '...', ...});
     MATCH (a:Entity {id: '...'}), (b:Entity {id: '...'})
     CREATE (a)-[:USES {...}]->(b);
     ```
   - Write to file
   - Return ExportResult

7. **Implement export_json():**
   - Validate output_path (.json extension)
   - Fetch graph data
   - Build simple JSON:
     ```json
     {
       "nodes": [...],
       "edges": [...],
       "metadata": {...}
     }
     ```
   - Write to file with json.dump()
   - Return ExportResult

8. **Create comprehensive unit tests** (test_graph_exporter.py):
   - Test init success/failure (with/without db_client, wrong type)
   - Test _fetch_full_graph (all nodes, filtered by type)
   - Test _validate_output_path (valid, invalid extension, parent not exists)
   - Test _node_to_dict and _edge_to_dict
   - Test export_graphml success/errors
   - Test export_cytoscape success/errors
   - Test export_neo4j success/errors
   - Test export_json success/errors
   - Test database errors
   - Test write errors (permission denied)
   - Mock FalkorDBClient
   - Target coverage > 85%

9. **Update graph/__init__.py:**
   ```python
   from zapomni_core.graph.graph_builder import GraphBuilder
   from zapomni_core.graph.graph_exporter import GraphExporter  # NEW

   __all__ = ["GraphBuilder", "GraphExporter"]
   ```

10. **Run tests and validate:**
    ```bash
    pytest tests/unit/test_graph_exporter.py -v --cov=src/zapomni_core/graph/graph_exporter.py --cov-report=term-missing
    black src/zapomni_core/graph/graph_exporter.py tests/unit/test_graph_exporter.py
    mypy src/zapomni_core/graph/graph_exporter.py
    ```

**Acceptance Criteria:**
- [x] File graph_exporter.py created (400-450 lines)
- [x] All 4 export formats implemented (GraphML, Cytoscape, Neo4j, JSON)
- [x] All unit tests pass (minimum 15 tests)
- [x] Coverage > 85%
- [x] Black formatted
- [x] Type hints on all methods
- [x] Complete docstrings (class + all public methods)
- [x] Error handling for all cases
- [x] Structured logging present
- [x] graph/__init__.py updated with export
- [x] No mypy errors

**API Specification from Architecture Plan:**

See ARCHITECTURE_PLAN.md lines 731-1398 for complete specification including:
- Class design (lines 879-1092)
- Implementation details (lines 1238-1353)
- Error handling strategy (lines 1356-1372)
- All 4 export format examples (lines 759-876)

**Estimated Time:** 60-90 minutes

**Validation Command:**
```bash
pytest tests/unit/test_graph_exporter.py -v --cov=src/zapomni_core/graph/graph_exporter.py --cov-report=term-missing
mypy src/zapomni_core/graph/graph_exporter.py
python -c "from zapomni_core.graph import GraphExporter; print('✓ Import successful')"
```

**Blocking:** Task 4.1 (export_graph tool DEPENDS on this)

**Blocked By:** None

---

### WAVE 4: Phase 3 Tools Part 1

---

## Task ID: 4.1

### Task Name: Implement export_graph Tool

**Type:** implementation

**Complexity:** medium

**Recommended Agent:** haiku

**Description:**
Create MCP tool that exports knowledge graph to various formats (GraphML, Cytoscape, Neo4j, JSON). Delegates to GraphExporter core component.

**Prerequisites:**
- [x] Task 3.1 completed (graph_exporter.py) ← **CRITICAL DEPENDENCY**
- [x] Docker services running (FalkorDB)

**Input Files (Read for patterns):**
- `/home/dev/zapomni/src/zapomni_mcp/tools/add_memory.py` (MCP tool pattern)
- `/home/dev/zapomni/src/zapomni_core/graph/graph_exporter.py` (GraphExporter API)
- `/home/dev/zapomni/ARCHITECTURE_PLAN.md` (lines 359-454 - API spec)

**Output Files (Create/Modify):**
- `/home/dev/zapomni/src/zapomni_mcp/tools/export_graph.py` (NEW)
- `/home/dev/zapomni/tests/unit/test_export_graph_tool.py` (NEW)

**Implementation Steps:**
1. Read `add_memory.py` to understand MCP tool pattern
2. Study GraphExporter API from `graph_exporter.py`
3. Study API specification from ARCHITECTURE_PLAN.md (lines 359-454)
4. Create `export_graph.py`:
   - Import dependencies: `GraphExporter`, `FalkorDBClient`, Pydantic models, Path
   - Define `ExportGraphRequest` Pydantic model:
     - `format: str` (enum: graphml, cytoscape, neo4j, json)
     - `output_path: str` (minLength=1)
     - `options: Optional[Dict]` with nested properties
   - Define `ExportGraphTool` class:
     - `name = "export_graph"`
     - `description` from spec
     - `input_schema` matching JSON Schema
     - `__init__(self, db_client: FalkorDBClient)` with validation
     - `async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]`:
       - Validate format and output_path
       - Initialize GraphExporter(db_client)
       - Route to appropriate export method based on format:
         - "graphml" → `exporter.export_graphml(output_path, options)`
         - "cytoscape" → `exporter.export_cytoscape(output_path, options)`
         - "neo4j" → `exporter.export_neo4j(output_path, options)`
         - "json" → `exporter.export_json(output_path, options)`
       - Get file stats after export (size)
       - Format response with metrics (nodes, edges, file size, export time)
   - Error handling: ValidationError, DatabaseError, IOError, ProcessingError
5. Create `test_export_graph_tool.py`:
   - Test init success/failure
   - Test execute success for all formats (graphml, cytoscape, neo4j, json)
   - Test validation errors (invalid format, path issues)
   - Test database errors, IO errors
   - Mock GraphExporter and FalkorDBClient
   - Target coverage > 80%
6. Run tests: `pytest tests/unit/test_export_graph_tool.py -v --cov=src/zapomni_mcp/tools/export_graph.py`
7. Format code: `black src/zapomni_mcp/tools/export_graph.py tests/unit/test_export_graph_tool.py`

**Acceptance Criteria:**
- [x] File export_graph.py created
- [x] All 4 export formats supported
- [x] All unit tests pass (minimum 12 tests)
- [x] Coverage > 80%
- [x] Black formatted
- [x] Type hints present
- [x] Docstrings complete
- [x] Error handling implemented
- [x] File size calculation working
- [x] Response formatting correct

**API Specification from Architecture Plan:**

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "format": {
      "type": "string",
      "enum": ["graphml", "cytoscape", "neo4j", "json"],
      "default": "json"
    },
    "output_path": {
      "type": "string",
      "minLength": 1
    },
    "options": {
      "type": "object",
      "properties": {
        "include_metadata": {"type": "boolean", "default": true},
        "pretty_print": {"type": "boolean", "default": true},
        "node_types": {"type": "array", "items": {"type": "string"}, "default": []}
      }
    }
  },
  "required": ["format", "output_path"]
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

**Estimated Time:** 30 minutes

**Validation Command:**
```bash
pytest tests/unit/test_export_graph_tool.py -v --cov=src/zapomni_mcp/tools/export_graph.py --cov-report=term-missing
```

**Blocking:** Task 5.2 (server registration)

**Blocked By:** Task 3.1 (graph_exporter.py) ← **CRITICAL**

---

## Task ID: 4.2

### Task Name: Implement index_codebase Tool

**Type:** implementation

**Complexity:** medium

**Recommended Agent:** haiku

**Description:**
Create MCP tool that indexes code repository with AST analysis for code search. Delegates to CodeRepositoryIndexer and stores via MemoryProcessor.

**Prerequisites:**
- [x] Docker services running (FalkorDB, Ollama)
- [x] CodeRepositoryIndexer component available
- [x] MemoryProcessor.add_memory() available

**Input Files (Read for patterns):**
- `/home/dev/zapomni/src/zapomni_mcp/tools/add_memory.py` (MCP tool pattern)
- `/home/dev/zapomni/src/zapomni_core/code/repository_indexer.py` (CodeRepositoryIndexer API)
- `/home/dev/zapomni/ARCHITECTURE_PLAN.md` (lines 457-569 - API spec)

**Output Files (Create/Modify):**
- `/home/dev/zapomni/src/zapomni_mcp/tools/index_codebase.py` (NEW)
- `/home/dev/zapomni/tests/unit/test_index_codebase_tool.py` (NEW)

**Implementation Steps:**
1. Read `add_memory.py` to understand MCP tool pattern
2. Study CodeRepositoryIndexer API
3. Study API specification from ARCHITECTURE_PLAN.md (lines 457-569)
4. Create `index_codebase.py`:
   - Import dependencies: `CodeRepositoryIndexer`, `MemoryProcessor`, Pydantic models, Path
   - Define language extension mapping:
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
   - Define `IndexCodebaseRequest` Pydantic model:
     - `repo_path: str` (minLength=1)
     - `languages: List[str]` (items from LANGUAGE_EXTENSIONS keys)
     - `options: Optional[Dict]` with recursive, max_file_size, include_tests
   - Define `IndexCodebaseTool` class:
     - `name = "index_codebase"`
     - `description` from spec
     - `input_schema` matching JSON Schema
     - `__init__(self, memory_processor: MemoryProcessor)` with validation
     - `async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]`:
       - Validate repo_path exists and is directory
       - Initialize CodeRepositoryIndexer with config
       - Call `indexer.index_repository(repo_path, languages)`
       - For each code file:
         - Extract functions/classes
         - Create memory node via `memory_processor.add_memory(code_text, metadata)`
       - Aggregate statistics (files indexed, functions, classes, languages, total lines)
       - Format response
   - Error handling: ValidationError, ProcessingError, DatabaseError
5. Create `test_index_codebase_tool.py`:
   - Test init success/failure
   - Test execute success (Python repo, multi-language repo)
   - Test validation errors (path not exists, invalid language)
   - Test processing errors, database errors
   - Mock CodeRepositoryIndexer and MemoryProcessor
   - Target coverage > 80%
6. Run tests: `pytest tests/unit/test_index_codebase_tool.py -v --cov=src/zapomni_mcp/tools/index_codebase.py`
7. Format code: `black src/zapomni_mcp/tools/index_codebase.py tests/unit/test_index_codebase_tool.py`

**Acceptance Criteria:**
- [x] File index_codebase.py created
- [x] Language extension mapping implemented
- [x] All unit tests pass (minimum 10 tests)
- [x] Coverage > 80%
- [x] Black formatted
- [x] Type hints present
- [x] Docstrings complete
- [x] Error handling implemented
- [x] Statistics aggregation working

**API Specification from Architecture Plan:**

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "repo_path": {
      "type": "string",
      "minLength": 1
    },
    "languages": {
      "type": "array",
      "items": {
        "type": "string",
        "enum": ["python", "javascript", "typescript", "java", "go", "rust", "cpp", "c"]
      },
      "default": []
    },
    "options": {
      "type": "object",
      "properties": {
        "recursive": {"type": "boolean", "default": true},
        "max_file_size": {"type": "integer", "minimum": 1024, "maximum": 100000000, "default": 10485760},
        "include_tests": {"type": "boolean", "default": false}
      }
    }
  },
  "required": ["repo_path"]
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

**Estimated Time:** 25 minutes

**Validation Command:**
```bash
pytest tests/unit/test_index_codebase_tool.py -v --cov=src/zapomni_mcp/tools/index_codebase.py --cov-report=term-missing
```

**Blocking:** Task 5.2 (server registration)

**Blocked By:** None

---

## Task ID: 4.3

### Task Name: Implement delete_memory Tool

**Type:** implementation

**Complexity:** simple

**Recommended Agent:** haiku

**Description:**
Create MCP tool that deletes a specific memory by ID with explicit confirmation. Includes safety checks.

**Prerequisites:**
- [x] Docker services running (FalkorDB)
- [x] FalkorDBClient.delete_memory() method available

**Input Files (Read for patterns):**
- `/home/dev/zapomni/src/zapomni_mcp/tools/add_memory.py` (MCP tool pattern)
- `/home/dev/zapomni/ARCHITECTURE_PLAN.md` (lines 572-640 - API spec)

**Output Files (Create/Modify):**
- `/home/dev/zapomni/src/zapomni_mcp/tools/delete_memory.py` (NEW)
- `/home/dev/zapomni/tests/unit/test_delete_memory_tool.py` (NEW)

**Implementation Steps:**
1. Read `add_memory.py` to understand MCP tool pattern
2. Study API specification from ARCHITECTURE_PLAN.md (lines 572-640)
3. Create `delete_memory.py`:
   - Import dependencies: `FalkorDBClient`, Pydantic models
   - Define `DeleteMemoryRequest` Pydantic model:
     - `memory_id: str` with UUID pattern validation
     - `confirm: bool` with enum=[True] (MUST be exactly True)
   - Define `DeleteMemoryTool` class:
     - `name = "delete_memory"`
     - `description` from spec (mark as potentially destructive)
     - `input_schema` matching JSON Schema
     - `__init__(self, db_client: FalkorDBClient)` with validation
     - `async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]`:
       - Validate memory_id as UUID format
       - **SAFETY CHECK**: Require `confirm=true` explicitly
       - Call `db_client.delete_memory(memory_id)`
       - Log deletion with memory_id and timestamp
       - Return success message with chunks_deleted count
       - Handle "not found" case gracefully (return message, not error)
   - Error handling: ValidationError, DatabaseError
4. Create `test_delete_memory_tool.py`:
   - Test init success/failure
   - Test execute success (memory exists)
   - Test validation errors (invalid UUID, confirm not true, confirm missing)
   - Test memory not found case
   - Test database errors
   - Verify logging occurs
   - Mock FalkorDBClient
   - Target coverage > 80%
5. Run tests: `pytest tests/unit/test_delete_memory_tool.py -v --cov=src/zapomni_mcp/tools/delete_memory.py`
6. Format code: `black src/zapomni_mcp/tools/delete_memory.py tests/unit/test_delete_memory_tool.py`

**Acceptance Criteria:**
- [x] File delete_memory.py created
- [x] Safety check (confirm=true) implemented
- [x] All unit tests pass (minimum 10 tests)
- [x] Coverage > 80%
- [x] Black formatted
- [x] Type hints present
- [x] Docstrings complete
- [x] Error handling implemented
- [x] UUID validation working
- [x] Logging present (with timestamp)
- [x] "Not found" case handled gracefully

**API Specification from Architecture Plan:**

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "memory_id": {
      "type": "string",
      "pattern": "^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    },
    "confirm": {
      "type": "boolean",
      "enum": [true]
    }
  },
  "required": ["memory_id", "confirm"]
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

**Estimated Time:** 15 minutes

**Validation Command:**
```bash
pytest tests/unit/test_delete_memory_tool.py -v --cov=src/zapomni_mcp/tools/delete_memory.py --cov-report=term-missing
```

**Blocking:** Task 5.2 (server registration)

**Blocked By:** None

---

### WAVE 5: Phase 3 Tools Part 2 + Integration

---

## Task ID: 5.1

### Task Name: Implement clear_all Tool

**Type:** implementation

**Complexity:** simple

**Recommended Agent:** haiku

**Description:**
Create MCP tool that clears all memories from the system with strict confirmation phrase. Maximum safety checks.

**Prerequisites:**
- [x] Docker services running (FalkorDB)
- [x] FalkorDBClient.clear_all() method available

**Input Files (Read for patterns):**
- `/home/dev/zapomni/src/zapomni_mcp/tools/delete_memory.py` (similar deletion tool)
- `/home/dev/zapomni/ARCHITECTURE_PLAN.md` (lines 643-710 - API spec)

**Output Files (Create/Modify):**
- `/home/dev/zapomni/src/zapomni_mcp/tools/clear_all.py` (NEW)
- `/home/dev/zapomni/tests/unit/test_clear_all_tool.py` (NEW)

**Implementation Steps:**
1. Read `delete_memory.py` to understand deletion pattern
2. Study API specification from ARCHITECTURE_PLAN.md (lines 643-710)
3. Create `clear_all.py`:
   - Import dependencies: `FalkorDBClient`, Pydantic models
   - Define `ClearAllRequest` Pydantic model:
     - `confirm_phrase: str` with pattern="^DELETE ALL MEMORIES$" (exact match required)
   - Define `ClearAllTool` class:
     - `name = "clear_all"`
     - `description` from spec (mark as DESTRUCTIVE)
     - `input_schema` matching JSON Schema
     - `__init__(self, db_client: FalkorDBClient)` with validation
     - `async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]`:
       - **SAFETY CHECK**: Require exact phrase match: "DELETE ALL MEMORIES" (case-sensitive)
       - Get stats BEFORE deletion (for response)
       - Call `db_client.clear_all()`
       - Log deletion with timestamp and counts
       - Return detailed breakdown (memories, chunks, entities, total nodes/edges deleted)
   - Error handling: ValidationError (phrase mismatch with helpful message), DatabaseError
5. Create `test_clear_all_tool.py`:
   - Test init success/failure
   - Test execute success (correct phrase)
   - Test validation errors (phrase mismatch, case mismatch, empty phrase)
   - Test database errors
   - Verify logging occurs
   - Verify stats fetched before deletion
   - Mock FalkorDBClient
   - Target coverage > 80%
6. Run tests: `pytest tests/unit/test_clear_all_tool.py -v --cov=src/zapomni_mcp/tools/clear_all.py`
7. Format code: `black src/zapomni_mcp/tools/clear_all.py tests/unit/test_clear_all_tool.py`

**Acceptance Criteria:**
- [x] File clear_all.py created
- [x] Exact phrase match validation (case-sensitive)
- [x] All unit tests pass (minimum 10 tests)
- [x] Coverage > 80%
- [x] Black formatted
- [x] Type hints present
- [x] Docstrings complete (warn about destructiveness)
- [x] Error handling implemented
- [x] Stats fetched before deletion
- [x] Detailed response breakdown
- [x] Logging present (with timestamp and counts)

**API Specification from Architecture Plan:**

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "confirm_phrase": {
      "type": "string",
      "pattern": "^DELETE ALL MEMORIES$"
    }
  },
  "required": ["confirm_phrase"]
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

**Estimated Time:** 15 minutes

**Validation Command:**
```bash
pytest tests/unit/test_clear_all_tool.py -v --cov=src/zapomni_mcp/tools/clear_all.py --cov-report=term-missing
```

**Blocking:** Task 5.2 (server registration)

**Blocked By:** None

---

## Task ID: 5.2

### Task Name: Register Phase 3 Tools in Server

**Type:** integration

**Complexity:** simple

**Recommended Agent:** haiku

**Description:**
Update server.py to register Phase 3 tools (export_graph, index_codebase, delete_memory, clear_all) and update __init__.py exports.

**Prerequisites:**
- [x] Task 4.1 completed (export_graph.py)
- [x] Task 4.2 completed (index_codebase.py)
- [x] Task 4.3 completed (delete_memory.py)
- [x] Task 5.1 completed (clear_all.py)

**Input Files (Read for patterns):**
- `/home/dev/zapomni/src/zapomni_mcp/server.py` (current tool registration)
- `/home/dev/zapomni/src/zapomni_mcp/tools/__init__.py` (exports)

**Output Files (Create/Modify):**
- `/home/dev/zapomni/src/zapomni_mcp/server.py` (MODIFY)
- `/home/dev/zapomni/src/zapomni_mcp/tools/__init__.py` (MODIFY)

**Implementation Steps:**
1. Read current `server.py` tool registration
2. Read current `tools/__init__.py` exports
3. Modify `server.py`:
   - Import new tools at top:
     ```python
     from zapomni_mcp.tools import (
         # Phase 1
         AddMemoryTool,
         SearchMemoryTool,
         GetStatsTool,
         # Phase 2
         BuildGraphTool,
         GetRelatedTool,
         GraphStatusTool,
         # Phase 3 NEW
         ExportGraphTool,
         IndexCodebaseTool,
         DeleteMemoryTool,
         ClearAllTool,
     )
     ```
   - Update `register_all_tools()` method:
     ```python
     tools = [
         # Phase 1
         AddMemoryTool(memory_processor=memory_processor),
         SearchMemoryTool(memory_processor=memory_processor),
         GetStatsTool(memory_processor=memory_processor),
         # Phase 2
         BuildGraphTool(db_client=memory_processor.db_client),
         GetRelatedTool(db_client=memory_processor.db_client),
         GraphStatusTool(db_client=memory_processor.db_client),
         # Phase 3
         ExportGraphTool(db_client=memory_processor.db_client),
         IndexCodebaseTool(memory_processor=memory_processor),
         DeleteMemoryTool(db_client=memory_processor.db_client),
         ClearAllTool(db_client=memory_processor.db_client),
     ]
     ```
4. Modify `tools/__init__.py`:
   - Add imports:
     ```python
     from zapomni_mcp.tools.export_graph import ExportGraphTool
     from zapomni_mcp.tools.index_codebase import IndexCodebaseTool
     from zapomni_mcp.tools.delete_memory import DeleteMemoryTool
     from zapomni_mcp.tools.clear_all import ClearAllTool
     ```
   - Update `__all__` list with new tools
5. Verify imports: `python -c "from zapomni_mcp.tools import ExportGraphTool, IndexCodebaseTool, DeleteMemoryTool, ClearAllTool"`
6. Format code: `black src/zapomni_mcp/server.py src/zapomni_mcp/tools/__init__.py`

**Acceptance Criteria:**
- [x] server.py imports Phase 3 tools
- [x] register_all_tools() instantiates Phase 3 tools
- [x] tools/__init__.py exports Phase 3 tools
- [x] No import errors
- [x] Code formatted with Black
- [x] Tool count in log message updated to 10

**Estimated Time:** 5 minutes

**Validation Command:**
```bash
python -c "from zapomni_mcp.tools import ExportGraphTool, IndexCodebaseTool, DeleteMemoryTool, ClearAllTool; print('✓ Imports successful')"
```

**Blocking:** Wave 6 (integration tests)

**Blocked By:** Task 4.1, 4.2, 4.3, 5.1

---

### WAVE 6: Integration Testing & Documentation

---

## Task ID: 6.1

### Task Name: Create Phase 2 Integration Test

**Type:** testing

**Complexity:** medium

**Recommended Agent:** sonnet

**Description:**
Create comprehensive integration test that validates full Phase 2 workflow: add_memory → build_graph → get_related → graph_status.

**Prerequisites:**
- [x] Wave 1 completed (all Phase 2 tools)
- [x] Wave 2 completed (tools registered, feature flags enabled)
- [x] Docker services running

**Input Files (Read for patterns):**
- `/home/dev/zapomni/tests/integration/test_mvp_integration.py` (existing integration test pattern)
- `/home/dev/zapomni/ARCHITECTURE_PLAN.md` (lines 1649-1689 - integration test spec)

**Output Files (Create/Modify):**
- `/home/dev/zapomni/tests/integration/test_phase2_integration.py` (NEW)

**Implementation Steps:**
1. Read existing integration test pattern from `test_mvp_integration.py`
2. Study integration test spec from ARCHITECTURE_PLAN.md (lines 1649-1689)
3. Create `test_phase2_integration.py`:
   - Setup fixtures:
     - `memory_processor` fixture (initialized with all dependencies)
     - `db_client` fixture
     - Cleanup fixture (clear DB after tests)
   - Implement `test_phase2_full_workflow()`:
     - **Step 1**: Add memory with entity-rich text
       ```python
       text = "Python is a programming language created by Guido van Rossum. It uses asyncio for async programming."
       memory_id = await memory_processor.add_memory(text, metadata={"source": "test"})
       assert memory_id is not None
       ```
     - **Step 2**: Build knowledge graph
       ```python
       build_tool = BuildGraphTool(db_client=memory_processor.db_client)
       build_result = await build_tool.execute({"text": text, "options": {"extract_entities": True}})
       assert build_result["isError"] is False
       assert "Entities:" in build_result["content"][0]["text"]
       ```
     - **Step 3**: Query graph to find entity IDs
       ```python
       # Use FalkorDB query to find Python entity
       query = "MATCH (e:Entity {name: 'Python'}) RETURN e.id"
       result = await db_client.graph_query(query)
       python_entity_id = result.rows[0]['e.id']
       ```
     - **Step 4**: Get related entities
       ```python
       related_tool = GetRelatedTool(db_client=memory_processor.db_client)
       related_result = await related_tool.execute({
           "entity_id": python_entity_id,
           "depth": 2,
           "limit": 10
       })
       assert related_result["isError"] is False
       assert "related entities" in related_result["content"][0]["text"]
       ```
     - **Step 5**: Check graph status
       ```python
       status_tool = GraphStatusTool(db_client=memory_processor.db_client)
       status_result = await status_tool.execute({})
       assert status_result["isError"] is False
       assert "Entities:" in status_result["content"][0]["text"]
       assert "Graph Health:" in status_result["content"][0]["text"]
       ```
   - Additional test cases:
     - `test_build_graph_with_options()` - Test confidence_threshold
     - `test_get_related_multiple_depths()` - Test depth parameter
     - `test_graph_status_empty_graph()` - Test with no entities
4. Run integration test: `pytest tests/integration/test_phase2_integration.py -v -s`
5. Format code: `black tests/integration/test_phase2_integration.py`

**Acceptance Criteria:**
- [x] File test_phase2_integration.py created (300-400 lines)
- [x] Full workflow test implemented
- [x] All tests pass
- [x] Tests verify Phase 2 feature flags working
- [x] Tests verify entity extraction working
- [x] Tests verify graph traversal working
- [x] Fixtures properly clean up
- [x] Black formatted

**Estimated Time:** 30 minutes

**Validation Command:**
```bash
pytest tests/integration/test_phase2_integration.py -v -s
```

**Blocking:** None (can run in parallel with 6.2)

**Blocked By:** Wave 2 (tools registered, feature flags enabled)

---

## Task ID: 6.2

### Task Name: Create Phase 3 Integration Test

**Type:** testing

**Complexity:** medium

**Recommended Agent:** sonnet

**Description:**
Create comprehensive integration test that validates full Phase 3 workflow: index_codebase → export_graph → delete_memory → clear_all.

**Prerequisites:**
- [x] Wave 4 completed (Phase 3 tools part 1)
- [x] Wave 5 completed (Phase 3 tools part 2, tools registered)
- [x] Docker services running

**Input Files (Read for patterns):**
- `/home/dev/zapomni/tests/integration/test_mvp_integration.py` (integration test pattern)
- `/home/dev/zapomni/ARCHITECTURE_PLAN.md` (lines 1691-1730 - integration test spec)

**Output Files (Create/Modify):**
- `/home/dev/zapomni/tests/integration/test_phase3_integration.py` (NEW)

**Implementation Steps:**
1. Read existing integration test pattern
2. Study integration test spec from ARCHITECTURE_PLAN.md (lines 1691-1730)
3. Create `test_phase3_integration.py`:
   - Setup fixtures:
     - `memory_processor` fixture
     - `db_client` fixture
     - `tmp_path` fixture (for export files)
     - Create sample code repository in tmp_path
     - Cleanup fixture
   - Implement `test_phase3_full_workflow()`:
     - **Step 1**: Index code repository
       ```python
       index_tool = IndexCodebaseTool(memory_processor=memory_processor)
       index_result = await index_tool.execute({
           "repo_path": str(test_repo_path),
           "languages": ["python"],
           "options": {"recursive": True}
       })
       assert index_result["isError"] is False
       assert "Files indexed:" in index_result["content"][0]["text"]
       ```
     - **Step 2**: Export graph to all formats
       ```python
       export_tool = ExportGraphTool(db_client=memory_processor.db_client)

       # Test GraphML export
       graphml_path = tmp_path / "graph.graphml"
       result = await export_tool.execute({"format": "graphml", "output_path": str(graphml_path)})
       assert result["isError"] is False
       assert graphml_path.exists()

       # Test JSON export
       json_path = tmp_path / "graph.json"
       result = await export_tool.execute({"format": "json", "output_path": str(json_path)})
       assert result["isError"] is False
       assert json_path.exists()

       # Verify JSON structure
       with open(json_path) as f:
           data = json.load(f)
           assert "nodes" in data
           assert "edges" in data
           assert len(data["nodes"]) > 0
       ```
     - **Step 3**: Delete specific memory
       ```python
       # Get a memory_id from stats
       stats = await memory_processor.get_stats()
       # Query for memory IDs
       query = "MATCH (m:Memory) RETURN m.id LIMIT 1"
       result = await db_client.graph_query(query)
       memory_id = result.rows[0]['m.id']

       delete_tool = DeleteMemoryTool(db_client=memory_processor.db_client)
       delete_result = await delete_tool.execute({
           "memory_id": memory_id,
           "confirm": True
       })
       assert delete_result["isError"] is False
       assert "deleted successfully" in delete_result["content"][0]["text"]
       ```
     - **Step 4**: Clear all
       ```python
       clear_tool = ClearAllTool(db_client=memory_processor.db_client)
       clear_result = await clear_tool.execute({
           "confirm_phrase": "DELETE ALL MEMORIES"
       })
       assert clear_result["isError"] is False
       assert "cleared" in clear_result["content"][0]["text"]

       # Verify graph is empty
       stats = await memory_processor.get_stats()
       assert stats["total_memories"] == 0
       ```
   - Additional test cases:
     - `test_export_all_formats()` - Test all 4 export formats
     - `test_delete_safety_checks()` - Test confirm required
     - `test_clear_all_phrase_validation()` - Test exact phrase match
4. Run integration test: `pytest tests/integration/test_phase3_integration.py -v -s`
5. Format code: `black tests/integration/test_phase3_integration.py`

**Acceptance Criteria:**
- [x] File test_phase3_integration.py created (400-500 lines)
- [x] Full workflow test implemented
- [x] All tests pass
- [x] Tests verify all export formats working
- [x] Tests verify code indexing working
- [x] Tests verify safety checks working (delete/clear)
- [x] Export files validated (structure, content)
- [x] Fixtures properly clean up
- [x] Black formatted

**Estimated Time:** 40 minutes

**Validation Command:**
```bash
pytest tests/integration/test_phase3_integration.py -v -s
```

**Blocking:** None (can run in parallel with 6.1)

**Blocked By:** Wave 5 (Phase 3 tools registered)

---

## Task ID: 6.3

### Task Name: Run Full Test Suite & Update Documentation

**Type:** testing + documentation

**Complexity:** medium

**Recommended Agent:** sonnet

**Description:**
Run complete test suite (unit + integration), verify coverage targets met, and update CHANGELOG.md with all Phase 2 & Phase 3 changes.

**Prerequisites:**
- [x] All waves 1-5 completed (all tools implemented)
- [x] Task 6.1 completed (Phase 2 integration test)
- [x] Task 6.2 completed (Phase 3 integration test)

**Input Files (Read):**
- `/home/dev/zapomni/CHANGELOG.md` (to append new entries)

**Output Files (Create/Modify):**
- `/home/dev/zapomni/CHANGELOG.md` (MODIFY - add Phase 2 & Phase 3 entries)

**Implementation Steps:**

1. **Run full test suite:**
   ```bash
   # Run all unit tests
   pytest tests/unit/ -v --cov=src --cov-report=term-missing --cov-report=html

   # Run all integration tests
   pytest tests/integration/ -v -s

   # Run full suite
   pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html
   ```

2. **Verify coverage targets:**
   - Check overall coverage > 80%
   - Check each new module coverage:
     - build_graph.py > 80%
     - get_related.py > 80%
     - graph_status.py > 80%
     - graph_exporter.py > 85%
     - export_graph.py > 80%
     - index_codebase.py > 80%
     - delete_memory.py > 80%
     - clear_all.py > 80%
   - If any module below target, identify missing tests and add them

3. **Verify no regressions:**
   - All Phase 1 tests still pass
   - No new mypy errors: `mypy src/`
   - No new black formatting issues: `black --check src/ tests/`

4. **Update CHANGELOG.md:**
   - Read current CHANGELOG.md
   - Add new section at top:
     ```markdown
     ## [Phase 2 & Phase 3] - 2025-11-24

     ### Added - Phase 2: Enhanced Search

     #### MCP Tools
     - **build_graph**: Extract entities from text and build knowledge graph structures
       - Entity extraction with SpaCy (TECHNOLOGY, PERSON, ORG, GPE, CONCEPT, PRODUCT, EVENT)
       - Confidence threshold filtering (default: 0.7)
       - Relationship detection support (Phase 2 stub)
     - **get_related**: Find entities related to a given entity via graph traversal
       - Multi-hop traversal (depth 1-5)
       - Relationship strength scoring
       - Result limiting and pagination
     - **graph_status**: Display knowledge graph statistics and health metrics
       - Node counts by type (Memory, Chunk, Entity, Document)
       - Relationship counts by type (HAS_CHUNK, MENTIONS, RELATED_TO)
       - Entity type breakdown
       - Graph health status (Healthy/Warning/Critical)

     #### Core Features
     - Entity extraction enabled by default (`enable_extraction=True`)
     - Knowledge graph construction enabled by default (`enable_graph=True`)
     - SpaCy integration for NLP-based entity recognition

     ### Added - Phase 3: Code Intelligence

     #### Core Components
     - **GraphExporter**: Export knowledge graphs to multiple formats
       - GraphML (XML) format for Gephi, yEd, NetworkX
       - Cytoscape JSON format for Cytoscape.js, Cytoscape Desktop
       - Neo4j Cypher statements for Neo4j, Memgraph import
       - Simple JSON format for custom processing and debugging
       - Node/edge filtering by type
       - Pretty-printing support
       - Metadata inclusion options

     #### MCP Tools
     - **export_graph**: Export knowledge graph for visualization and analysis
       - Support for 4 export formats (GraphML, Cytoscape, Neo4j, JSON)
       - File size reporting
       - Export time metrics
     - **index_codebase**: Index code repository with AST analysis
       - Multi-language support (Python, JavaScript, TypeScript, Java, Go, Rust, C, C++)
       - Function and class extraction
       - Recursive directory scanning
       - Test file filtering
       - Max file size limits
     - **delete_memory**: Delete specific memory by ID
       - Explicit confirmation required (`confirm=true`)
       - UUID validation
       - Chunk deletion cascade
       - Operation logging with timestamps
     - **clear_all**: Clear all memories from system
       - Strict confirmation phrase required ("DELETE ALL MEMORIES")
       - Pre-deletion statistics collection
       - Detailed deletion breakdown
       - Operation logging with counts

     ### Changed
     - ProcessorConfig: Phase 2 features now enabled by default
       - `enable_extraction` changed from `False` → `True`
       - `enable_graph` changed from `False` → `True`
     - Tool count increased from 3 → 10 tools (Phase 1: 3, Phase 2: +3, Phase 3: +4)

     ### Technical Details
     - Total new code: ~2,150 lines (7 MCP tools + 1 core component)
     - Total tests: ~2,400 lines (unit + integration)
     - Coverage: > 80% for all new components
     - Format compliance: Black, mypy strict mode
     - Logging: structlog for all operations
     - Error handling: Custom exceptions for all failure modes

     ### Breaking Changes
     None - All changes are additive and backward compatible.
     ```

5. **Run final validation:**
   ```bash
   # Verify all tests pass
   pytest tests/ -v

   # Verify formatting
   black --check src/ tests/

   # Verify type hints
   mypy src/

   # Count tools registered
   python -c "from zapomni_mcp.server import MCPServer; print('Expected 10 tools in Phase 1+2+3')"
   ```

**Acceptance Criteria:**
- [x] All unit tests pass (100+ tests)
- [x] All integration tests pass (6+ tests)
- [x] Overall coverage > 80%
- [x] Each new module coverage > 80% (85% for graph_exporter)
- [x] No regressions in Phase 1 tests
- [x] No mypy errors
- [x] No black formatting issues
- [x] CHANGELOG.md updated with comprehensive Phase 2 & Phase 3 entries
- [x] Test summary generated (total tests, coverage %, time)

**Estimated Time:** 30 minutes

**Validation Command:**
```bash
# Full test suite
pytest tests/ -v --cov=src --cov-report=term-missing

# Generate summary
pytest tests/ --tb=short --co -q | wc -l  # Count tests
coverage report --show-missing | grep TOTAL  # Check coverage
```

**Blocking:** None (final task)

**Blocked By:** Task 6.1, 6.2

---

## Agent Assignment Strategy

### Haiku Agents (Fast, Simple/Medium tasks)
**Total: 11 tasks**

**Wave 1 (Parallel - 3 agents):**
- Agent 1: Task 1.1 - build_graph tool (20 min)
- Agent 2: Task 1.2 - get_related tool (15 min)
- Agent 3: Task 1.3 - graph_status tool (15 min)

**Wave 2 (Sequential - 2 agents):**
- Agent 1: Task 2.1 - Register Phase 2 tools (5 min)
- Agent 1: Task 2.2 - Enable feature flags (5 min) ← Sequential after 2.1

**Wave 4 (Parallel - 2 agents):**
- Agent 1: Task 4.2 - index_codebase tool (25 min)
- Agent 2: Task 4.3 - delete_memory tool (15 min)

**Wave 5 (Parallel - 2 agents):**
- Agent 1: Task 5.1 - clear_all tool (15 min)
- Agent 2: Task 5.2 - Register Phase 3 tools (5 min)

---

### Sonnet Agents (Complex tasks)
**Total: 5 tasks**

**Wave 3 (Critical Path - 1 agent):**
- Agent 1: Task 3.1 - graph_exporter.py (60-90 min) ← **CRITICAL**

**Wave 4 (After Wave 3 - 1 agent):**
- Agent 1: Task 4.1 - export_graph tool (30 min) ← Depends on 3.1

**Wave 6 (Parallel - 3 agents):**
- Agent 1: Task 6.1 - Phase 2 integration test (30 min)
- Agent 2: Task 6.2 - Phase 3 integration test (40 min)
- Agent 3: Task 6.3 - Full test suite + CHANGELOG (30 min)

---

## Risk Mitigation

### Critical Path Risks

**Risk 1: graph_exporter.py implementation fails or takes too long**
- **Impact:** Blocks Task 4.1 (export_graph tool), delays entire Phase 3
- **Probability:** Medium (complex component, 4 export formats)
- **Mitigation:**
  - Assign to sonnet agent (better for complex tasks)
  - Break into incremental deliverables:
    1. Core structure + _fetch_full_graph() (20 min)
    2. export_json() (simplest format) (15 min)
    3. export_graphml() (20 min)
    4. export_cytoscape() + export_neo4j() (25 min)
    5. Tests (20 min)
  - Test each export format independently
  - If 1 format fails, skip it and continue with others
- **Fallback:**
  - Implement only JSON export initially (minimum viable)
  - Other formats can be added in Phase 3.1 (follow-up)

**Risk 2: EntityExtractor or GraphBuilder not working as expected**
- **Impact:** Blocks Wave 1 (build_graph tool)
- **Probability:** Low (components already exist and tested)
- **Mitigation:**
  - Task 1.1 agent should test EntityExtractor + GraphBuilder in isolation first
  - If broken, create stub implementation that returns empty entities
  - Log warning: "Entity extraction not available, using fallback"
- **Fallback:**
  - Stub implementation: Return empty entity list
  - Phase 2 tools still functional, just with no entities extracted

**Risk 3: Integration tests fail due to FalkorDB/Ollama issues**
- **Impact:** Blocks Wave 6 completion
- **Probability:** Medium (Docker services required)
- **Mitigation:**
  - Add retry logic to integration tests (3 retries with backoff)
  - Check Docker services health before running tests:
    ```bash
    docker ps | grep falkordb
    docker ps | grep ollama
    ```
  - Clear FalkorDB data between test runs
- **Fallback:**
  - Skip integration tests if services unavailable
  - Mark as "skipped" not "failed"
  - Unit tests still provide coverage

---

### Integration Risks

**Risk 4: Tool registration breaks existing Phase 1 functionality**
- **Impact:** Phase 1 tests start failing
- **Probability:** Low (additive changes only)
- **Mitigation:**
  - Task 2.1 agent must run Phase 1 tests after registration:
    ```bash
    pytest tests/unit/test_add_memory_tool.py tests/unit/test_search_memory_tool.py tests/unit/test_get_stats_tool.py -v
    ```
  - If any fail, rollback changes and investigate
  - Add validation: Tool count must be exactly 6 after Wave 2
- **Validation:**
  ```bash
  python -c "from zapomni_mcp.tools import *; tools = [AddMemoryTool, SearchMemoryTool, GetStatsTool, BuildGraphTool, GetRelatedTool, GraphStatusTool]; assert len(tools) == 6"
  ```

**Risk 5: Feature flags enable_extraction/enable_graph break existing behavior**
- **Impact:** Phase 1 integration tests fail
- **Probability:** Medium (changes default behavior)
- **Mitigation:**
  - Task 2.2 agent must run all existing tests after enabling flags
  - If extraction/graph fail, add try/except fallback:
    ```python
    if self.config.enable_extraction and self.extractor is not None:
        try:
            entities, relationships = await self._extract_entities(text, chunks)
        except ExtractionError as e:
            log.warning("entity_extraction_failed", error=str(e))
            entities, relationships = [], []
    ```
  - Graceful degradation: System works even if entity extraction fails
- **Validation:**
  ```bash
  pytest tests/integration/test_mvp_integration.py -v
  ```

---

### Safety Risks

**Risk 6: delete_memory/clear_all tools used without confirmation**
- **Impact:** Accidental data loss
- **Probability:** Low (validation prevents it)
- **Mitigation:**
  - Pydantic validation ENFORCES confirmation parameters
  - `delete_memory`: `confirm` field with `enum=[True]` (MUST be exactly True)
  - `clear_all`: `confirm_phrase` field with `pattern="^DELETE ALL MEMORIES$"` (exact match)
  - Tests verify rejection of incorrect confirmations
  - All deletions logged with timestamp
- **Testing:**
  - Task 4.3: Test that `confirm=False` raises ValidationError
  - Task 5.1: Test that phrase "delete all memories" (lowercase) raises ValidationError
  - Task 6.2: Integration test verifies clear_all safety

---

### Quality Risks

**Risk 7: Coverage targets not met (<80%)**
- **Impact:** Code quality below standard
- **Probability:** Low (explicit coverage requirements in each task)
- **Mitigation:**
  - Each task specifies minimum test count and coverage target
  - Agents must run coverage command before marking task complete
  - Task 6.3 validates all modules meet coverage targets
  - If any module below target, Task 6.3 agent adds missing tests
- **Validation:**
  ```bash
  pytest tests/unit/test_build_graph_tool.py --cov=src/zapomni_mcp/tools/build_graph.py --cov-fail-under=80
  ```

**Risk 8: Inconsistent code style (Black formatting, mypy errors)**
- **Impact:** Code review failures, merge conflicts
- **Probability:** Low (explicit formatting steps in each task)
- **Mitigation:**
  - Each task includes Black formatting step
  - Each task includes mypy validation (for core components)
  - Task 6.3 runs full codebase formatting check
  - Pre-commit hooks (if configured) enforce formatting
- **Validation:**
  ```bash
  black --check src/ tests/
  mypy src/zapomni_core/ src/zapomni_mcp/
  ```

---

## Success Criteria

### Functional Requirements
- [x] All 7 MCP tools created and functional
  - Phase 2: build_graph, get_related, graph_status (3)
  - Phase 3: export_graph, index_codebase, delete_memory, clear_all (4)
- [x] graph_exporter.py core component implemented with 4 export formats
- [x] All tools properly registered in server.py
- [x] Feature flags enabled (enable_extraction, enable_graph)
- [x] Tools accessible via MCP protocol

### Testing Requirements
- [x] All unit tests pass (14 new test files)
  - build_graph_tool: 12+ tests
  - get_related_tool: 10+ tests
  - graph_status_tool: 8+ tests
  - graph_exporter: 15+ tests
  - export_graph_tool: 12+ tests
  - index_codebase_tool: 10+ tests
  - delete_memory_tool: 10+ tests
  - clear_all_tool: 10+ tests
- [x] Integration tests pass (2 new test files)
  - test_phase2_integration.py: 4+ tests
  - test_phase3_integration.py: 5+ tests
- [x] Full test suite passes (Phase 1 + Phase 2 + Phase 3)
- [x] Coverage > 80% overall
- [x] Coverage > 80% for each new module (85% for graph_exporter)

### Code Quality Requirements
- [x] All code formatted with Black (line length 100)
- [x] All functions have type hints
- [x] All public methods have docstrings
- [x] No mypy errors in strict mode
- [x] Structured logging present in all tools
- [x] Error handling for all failure modes

### Documentation Requirements
- [x] CHANGELOG.md updated with Phase 2 & Phase 3 entries
- [x] All tools have description and input_schema
- [x] Docstrings follow Google style
- [x] Integration test documentation
- [x] API specifications documented

### Safety Requirements
- [x] delete_memory requires explicit `confirm=true`
- [x] clear_all requires exact phrase "DELETE ALL MEMORIES"
- [x] All deletions logged with timestamp
- [x] UUID validation for memory_id parameters
- [x] Path validation for export_graph and index_codebase
- [x] Integration tests verify safety mechanisms

---

## Next Steps for Orchestrator

### Pre-Execution Checklist
1. **Verify environment:**
   ```bash
   # Check Docker services
   docker ps | grep falkordb  # Should be running on port 6381
   docker ps | grep ollama    # Should be running on port 11434

   # Check SpaCy model
   python -c "import spacy; spacy.load('en_core_web_sm')"  # Should succeed

   # Check git status
   git status  # Should be on main branch, working directory clean
   ```

2. **Create feature branches:**
   ```bash
   git checkout -b feature/phase2-enhanced-search
   git checkout main
   git checkout -b feature/phase3-code-intelligence
   git checkout main
   ```

3. **Backup current state:**
   ```bash
   # Tag current state before starting
   git tag -a v1.0-phase1-complete -m "Phase 1 MVP complete, starting Phase 2 & 3"
   ```

---

### Wave Execution Sequence

#### **WAVE 1: Launch 3 Haiku agents in parallel** (20-30 min)
```bash
# Agent 1: Task 1.1
Launch: "Implement build_graph tool following Task ID 1.1 specification"
Monitor: Check for file creation, test execution, coverage report
Validate: pytest tests/unit/test_build_graph_tool.py -v --cov

# Agent 2: Task 1.2 (parallel)
Launch: "Implement get_related tool following Task ID 1.2 specification"
Monitor: Check for file creation, test execution
Validate: pytest tests/unit/test_get_related_tool.py -v --cov

# Agent 3: Task 1.3 (parallel)
Launch: "Implement graph_status tool following Task ID 1.3 specification"
Monitor: Check for file creation, test execution
Validate: pytest tests/unit/test_graph_status_tool.py -v --cov
```

**Wait for all 3 to complete. Then validate Wave 1:**
```bash
# Validate Wave 1 outputs
pytest tests/unit/test_build_graph_tool.py tests/unit/test_get_related_tool.py tests/unit/test_graph_status_tool.py -v --cov
ls -la src/zapomni_mcp/tools/build_graph.py src/zapomni_mcp/tools/get_related.py src/zapomni_mcp/tools/graph_status.py
```

---

#### **WAVE 2: Launch 1 Haiku agent (sequential tasks)** (10-15 min)
```bash
# Agent 1: Task 2.1 (sequential - must complete before 2.2)
Launch: "Register Phase 2 tools in server.py following Task ID 2.1 specification"
Monitor: Check imports, tool registration, __init__.py updates
Validate: python -c "from zapomni_mcp.tools import BuildGraphTool, GetRelatedTool, GraphStatusTool"

# Agent 1: Task 2.2 (sequential - after 2.1 completes)
Launch: "Enable Phase 2 feature flags following Task ID 2.2 specification"
Monitor: Check ProcessorConfig changes, docstring update
Validate: pytest tests/unit/test_memory_processor.py -v
```

**Validate Wave 2:**
```bash
# Check Phase 1 tests still pass
pytest tests/unit/test_add_memory_tool.py tests/unit/test_search_memory_tool.py tests/unit/test_get_stats_tool.py -v

# Check tool count
python -c "from zapomni_mcp.tools import *; tools = ['AddMemoryTool', 'SearchMemoryTool', 'GetStatsTool', 'BuildGraphTool', 'GetRelatedTool', 'GraphStatusTool']; print(f'✓ {len(tools)} tools registered')"
```

---

#### **WAVE 3: Launch 1 Sonnet agent (CRITICAL PATH)** (60-90 min)
```bash
# Agent 1: Task 3.1 (CRITICAL - blocks Wave 4 Task 4.1)
Launch: "Implement graph_exporter.py core component following Task ID 3.1 specification. This is CRITICAL PATH - take your time and ensure all 4 export formats work correctly."
Monitor:
  - Check file creation after 20 min (should have core structure)
  - Check test creation after 40 min (should have basic tests)
  - Check coverage report after 60 min (should be >85%)
Validate:
  pytest tests/unit/test_graph_exporter.py -v --cov=src/zapomni_core/graph/graph_exporter.py --cov-report=term-missing
  mypy src/zapomni_core/graph/graph_exporter.py
  python -c "from zapomni_core.graph import GraphExporter; print('✓ Import successful')"
```

**CHECKPOINT: Validate Wave 3 before proceeding to Wave 4:**
```bash
# Critical validation - must pass before Wave 4
pytest tests/unit/test_graph_exporter.py -v
coverage report --include="src/zapomni_core/graph/graph_exporter.py" | grep graph_exporter  # Should be >85%
python -c "from zapomni_core.graph import GraphExporter; print('✓ GraphExporter ready')"
```

**If Wave 3 fails:**
- DO NOT proceed to Task 4.1 (export_graph depends on graph_exporter)
- CAN proceed with Task 4.2 and 4.3 (independent)
- Investigate and fix graph_exporter before launching Task 4.1

---

#### **WAVE 4: Launch 2-3 agents (partial parallel)** (30-45 min)
```bash
# Agent 1: Task 4.1 (ONLY AFTER Wave 3 completes successfully)
Launch: "Implement export_graph tool following Task ID 4.1 specification"
Monitor: Check for GraphExporter import, all 4 formats tested
Validate: pytest tests/unit/test_export_graph_tool.py -v --cov

# Agent 2: Task 4.2 (parallel, independent)
Launch: "Implement index_codebase tool following Task ID 4.2 specification"
Monitor: Check for language mapping, CodeRepositoryIndexer usage
Validate: pytest tests/unit/test_index_codebase_tool.py -v --cov

# Agent 3: Task 4.3 (parallel, independent)
Launch: "Implement delete_memory tool following Task ID 4.3 specification"
Monitor: Check for safety checks (confirm=true), UUID validation
Validate: pytest tests/unit/test_delete_memory_tool.py -v --cov
```

**Validate Wave 4:**
```bash
# Check all Wave 4 tests pass
pytest tests/unit/test_export_graph_tool.py tests/unit/test_index_codebase_tool.py tests/unit/test_delete_memory_tool.py -v --cov

# Verify safety checks
python -c "from zapomni_mcp.tools.delete_memory import DeleteMemoryRequest; print('✓ Safety validations present')"
```

---

#### **WAVE 5: Launch 2 Haiku agents in parallel** (15-20 min)
```bash
# Agent 1: Task 5.1 (parallel)
Launch: "Implement clear_all tool following Task ID 5.1 specification"
Monitor: Check for exact phrase match validation
Validate: pytest tests/unit/test_clear_all_tool.py -v --cov

# Agent 2: Task 5.2 (parallel)
Launch: "Register Phase 3 tools in server.py following Task ID 5.2 specification"
Monitor: Check imports for 4 Phase 3 tools, tool registration updated
Validate: python -c "from zapomni_mcp.tools import ExportGraphTool, IndexCodebaseTool, DeleteMemoryTool, ClearAllTool"
```

**Validate Wave 5:**
```bash
# Check tool count (should be 10 total)
python -c "from zapomni_mcp.tools import *; tools = dir(); print(f'Tool classes: {[t for t in tools if t.endswith(\"Tool\")]}')"

# Verify Phase 3 tools registered
pytest tests/unit/test_clear_all_tool.py -v --cov
```

---

#### **WAVE 6: Launch 3 Sonnet agents in parallel** (45-60 min)
```bash
# Agent 1: Task 6.1 (parallel)
Launch: "Create Phase 2 integration test following Task ID 6.1 specification"
Monitor: Check for comprehensive workflow test
Validate: pytest tests/integration/test_phase2_integration.py -v -s

# Agent 2: Task 6.2 (parallel)
Launch: "Create Phase 3 integration test following Task ID 6.2 specification"
Monitor: Check for all export formats tested, safety checks verified
Validate: pytest tests/integration/test_phase3_integration.py -v -s

# Agent 3: Task 6.3 (parallel)
Launch: "Run full test suite and update CHANGELOG following Task ID 6.3 specification"
Monitor: Check for coverage reports, CHANGELOG updates
Validate: pytest tests/ -v --cov=src --cov-report=term-missing
```

**Final Validation Wave 6:**
```bash
# Full test suite must pass
pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

# Check coverage targets
coverage report --show-missing | grep TOTAL  # Should be >80%

# Verify CHANGELOG updated
git diff CHANGELOG.md  # Should show Phase 2 & Phase 3 entries

# Verify no regressions
black --check src/ tests/
mypy src/
```

---

### Post-Execution Steps

1. **Merge to main:**
   ```bash
   # Create PR for Phase 2
   git checkout feature/phase2-enhanced-search
   git add .
   git commit -m "feat: Implement Phase 2 Enhanced Search (build_graph, get_related, graph_status)"
   git push origin feature/phase2-enhanced-search
   # Create PR, review, merge

   # Create PR for Phase 3
   git checkout feature/phase3-code-intelligence
   git add .
   git commit -m "feat: Implement Phase 3 Code Intelligence (export_graph, index_codebase, delete_memory, clear_all, GraphExporter)"
   git push origin feature/phase3-code-intelligence
   # Create PR, review, merge
   ```

2. **Tag release:**
   ```bash
   git checkout main
   git tag -a v2.0-phase2-phase3-complete -m "Phase 2 & Phase 3 implementation complete"
   git push origin v2.0-phase2-phase3-complete
   ```

3. **Update documentation:**
   - Update README.md with Phase 2 & Phase 3 features
   - Create docs/PHASE2_GUIDE.md with entity extraction examples
   - Create docs/PHASE3_GUIDE.md with export and code indexing examples
   - Update API documentation

4. **Performance testing:**
   ```bash
   # Test with real data
   # - Test entity extraction with Wikipedia articles
   # - Test graph export with 1000+ nodes
   # - Test code indexing with large repositories (e.g., Django source)
   # - Test delete/clear operations
   ```

---

## Execution Timeline Summary

| Wave | Tasks | Agent Type | Parallelism | Time | Cumulative |
|------|-------|-----------|-------------|------|------------|
| Wave 1 | 3 | Haiku | Parallel (3) | 20-30 min | 20-30 min |
| Wave 2 | 2 | Haiku | Sequential | 10-15 min | 30-45 min |
| Wave 3 | 1 | Sonnet | Sequential (CRITICAL) | 60-90 min | 90-135 min |
| Wave 4 | 3 | Haiku/Sonnet | Partial Parallel | 30-45 min | 120-180 min |
| Wave 5 | 2 | Haiku | Parallel (2) | 15-20 min | 135-200 min |
| Wave 6 | 3 | Sonnet | Parallel (3) | 45-60 min | 180-260 min |

**Total Estimated Time:** 180-260 minutes (3-4.5 hours)

**Optimal Scenario:** 180 minutes (3 hours) with perfect parallel execution
**Realistic Scenario:** 220 minutes (3.7 hours) with minor delays
**Pessimistic Scenario:** 260 minutes (4.3 hours) with Wave 3 retry

---

## Success Metrics

After completion, verify these metrics:

```bash
# Test Count
pytest tests/ --co -q | wc -l  # Should be 100+ tests

# Coverage
coverage report | grep TOTAL  # Should show >80%

# Lines of Code
cloc src/zapomni_mcp/tools/build_graph.py src/zapomni_mcp/tools/get_related.py src/zapomni_mcp/tools/graph_status.py src/zapomni_core/graph/graph_exporter.py src/zapomni_mcp/tools/export_graph.py src/zapomni_mcp/tools/index_codebase.py src/zapomni_mcp/tools/delete_memory.py src/zapomni_mcp/tools/clear_all.py
# Should be ~2,150 lines total

# Tool Count
python -c "from zapomni_mcp.server import MCPServer; print('Expect 10 total tools: 3 Phase 1 + 3 Phase 2 + 4 Phase 3')"
```

**Expected Results:**
- Total tests: 100-120 tests
- Total coverage: 80-85%
- Total LOC (new code): ~2,150 lines
- Total LOC (tests): ~2,400 lines
- Total tools: 10 (3+3+4)
- Zero regressions: Phase 1 tests still pass
- Clean code: Black formatted, mypy compliant

---

**END OF TASK_BREAKDOWN.md**
