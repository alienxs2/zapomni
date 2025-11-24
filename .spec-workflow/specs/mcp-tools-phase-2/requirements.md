# Requirements: Phase 2 MCP Tools

**Spec Name:** mcp-tools-phase-2
**Status:** Requirements Phase
**Created:** 2025-11-24
**Dependencies:** Phase 1 MVP (add_memory, search_memory, get_stats)

---

## 1. Executive Summary

Phase 2 extends Zapomni's MCP interface to expose existing knowledge graph functionality. Three new tools (`build_graph`, `get_related`, `graph_status`) will enable entity extraction, graph construction, and relationship traversal—all built on already-implemented core and database layers.

**Key Points:**
- **What:** Expose 3 new MCP tools for knowledge graph operations
- **Why:** Enable AI assistants to build and query knowledge graphs from memories
- **How:** Wrap existing `EntityExtractor`, `GraphBuilder`, and `FalkorDBClient` implementations
- **Scope:** MCP tool wrappers only—no changes to core/db layers
- **Success:** Tools follow Phase 1 patterns, pass integration tests, documented

---

## 2. Background & Context

### 2.1 Current State (Phase 1 MVP)

**Existing MCP Tools:**
- `add_memory`: Stores text with semantic chunking + embeddings
- `search_memory`: Vector similarity search on chunks
- `get_stats`: Retrieves memory system statistics

**Existing Core Functionality (Not Exposed):**
- `EntityExtractor` (src/zapomni_core/extractors/entity_extractor.py)
  - Hybrid SpaCy + LLM entity extraction
  - Normalizes entities, 80%+ precision target
  - Methods: `extract_entities()`, `normalize_entity()`

- `GraphBuilder` (src/zapomni_core/graph/graph_builder.py)
  - Builds knowledge graphs from extracted entities
  - Methods: `build_graph()`, `add_entity_nodes()`, `add_relationships()`
  - Returns statistics: entities created/merged, relationships added

- `FalkorDBClient` (src/zapomni_db/falkordb_client.py)
  - Graph database operations
  - Methods: `add_entity()`, `add_relationship()`, `get_related_entities()`, `get_stats()`
  - Supports Cypher queries, entity traversal

**Current MemoryProcessor Config:**
```python
@dataclass
class ProcessorConfig:
    enable_cache: bool = False
    enable_extraction: bool = False  # Phase 2 feature flag
    enable_graph: bool = False       # Phase 2 feature flag
```

### 2.2 Problem Statement

**User Need:**
AI assistants need to build knowledge graphs from stored memories to discover relationships, traverse entity networks, and answer complex queries that require graph reasoning.

**Current Gap:**
While the core functionality exists, it's not accessible via MCP. Users cannot:
1. Trigger entity extraction and graph building
2. Query related entities via graph traversal
3. View graph construction statistics

**Impact:**
Without these tools, Zapomni remains a "flat" semantic search system. Knowledge graph capabilities (entity relationships, concept networks, contextual understanding) are implemented but unusable.

---

## 3. User Stories

### 3.1 build_graph Tool

**US-1: Extract Entities from Memories**
```
AS an AI assistant
I WANT to extract entities from stored memories and build a knowledge graph
SO THAT I can discover relationships and perform graph-based reasoning

GIVEN I have stored memories containing entities (people, organizations, concepts)
WHEN I call build_graph with memory_id(s) or "all"
THEN the system extracts entities, creates nodes, detects relationships
AND returns statistics (entities created/merged, relationships detected)
```

**Acceptance Criteria:**
- [ ] Tool accepts `memory_ids` parameter (List[str] or "all")
- [ ] Tool accepts `mode` parameter ("entities_only", "relationships_only", "full")
- [ ] Tool validates memory_ids are valid UUIDs
- [ ] Tool enables `enable_extraction=True` and `enable_graph=True` in ProcessorConfig
- [ ] Tool delegates to `GraphBuilder.build_graph()`
- [ ] Tool returns statistics: `entities_created`, `entities_merged`, `relationships_created`
- [ ] Tool handles errors gracefully (missing memories, extraction failures)

**EARS Criteria:**
- **Event:** User invokes `build_graph` tool
- **Action:** System extracts entities, builds graph, returns stats
- **Response:** Success response with statistics
- **Success:** Graph construction completes, statistics returned

---

### 3.2 get_related Tool

**US-2: Find Related Entities via Graph Traversal**
```
AS an AI assistant
I WANT to find entities related to a given entity via graph relationships
SO THAT I can explore concept networks and discover contextual connections

GIVEN I have a knowledge graph with entities and relationships
WHEN I call get_related with entity_name="Python" and depth=2
THEN the system traverses the graph 2 hops deep
AND returns related entities sorted by relationship strength
```

**Acceptance Criteria:**
- [ ] Tool accepts `entity_name` parameter (string, entity to query)
- [ ] Tool accepts `depth` parameter (int, 1-5 hops, default: 1)
- [ ] Tool accepts `limit` parameter (int, max results, default: 20, max: 100)
- [ ] Tool validates entity exists in graph
- [ ] Tool delegates to `FalkorDBClient.get_related_entities()`
- [ ] Tool returns list of related entities with types, descriptions, relationship strengths
- [ ] Tool handles edge cases (entity not found, no relationships, isolated node)

**EARS Criteria:**
- **Event:** User invokes `get_related` tool
- **Action:** System traverses graph N hops from entity
- **Response:** List of related entities with metadata
- **Success:** Related entities returned, sorted by relevance

---

### 3.3 graph_status Tool

**US-3: View Knowledge Graph Statistics**
```
AS an AI assistant
I WANT to view knowledge graph construction status and statistics
SO THAT I can understand graph coverage and health

GIVEN I have built a knowledge graph from memories
WHEN I call graph_status
THEN the system returns comprehensive graph statistics
AND includes entity counts, relationship counts, coverage metrics
```

**Acceptance Criteria:**
- [ ] Tool accepts no parameters (reads entire graph)
- [ ] Tool delegates to `FalkorDBClient.get_stats()` for graph-specific stats
- [ ] Tool returns: `total_entities`, `total_relationships`, `entity_types` (breakdown), `relationship_types` (breakdown)
- [ ] Tool returns: `graph_coverage` (% of memories with entities), `avg_entities_per_memory`, `avg_relationships_per_entity`
- [ ] Tool returns: `most_connected_entities` (top 10 by relationship count)
- [ ] Tool formats response as human-readable text with sections

**EARS Criteria:**
- **Event:** User invokes `graph_status` tool
- **Action:** System queries graph statistics
- **Response:** Formatted statistics report
- **Success:** Statistics returned with entity/relationship breakdowns

---

## 4. Functional Requirements

### 4.1 build_graph Tool

**FR-1.1: Input Validation**
- MUST validate `memory_ids` are valid UUID format
- MUST validate `mode` is one of: "entities_only", "relationships_only", "full"
- MUST return ValidationError for invalid inputs

**FR-1.2: Configuration**
- MUST enable `ProcessorConfig.enable_extraction=True` when building graph
- MUST enable `ProcessorConfig.enable_graph=True` when building graph
- MUST initialize EntityExtractor with SpaCy model (en_core_web_sm)
- MUST initialize GraphBuilder with EntityExtractor and FalkorDBClient

**FR-1.3: Entity Extraction**
- MUST extract entities from specified memories
- MUST deduplicate entities (merge by name + type)
- MUST normalize entity names via `EntityExtractor.normalize_entity()`
- MUST handle extraction errors gracefully (log warnings, continue with partial results)

**FR-1.4: Graph Construction**
- MUST create entity nodes in FalkorDB via `GraphBuilder.add_entity_nodes()`
- MUST detect relationships between entities (if mode="full" or "relationships_only")
- MUST store relationship edges in FalkorDB via `GraphBuilder.add_relationships()`
- MUST return statistics: entities_created, entities_merged, relationships_created

**FR-1.5: Error Handling**
- MUST handle missing memories (return error if memory_id not found)
- MUST handle empty entity extraction (return success with 0 entities)
- MUST handle database errors (return DatabaseError with retry logic)

---

### 4.2 get_related Tool

**FR-2.1: Input Validation**
- MUST validate `entity_name` is non-empty string
- MUST validate `depth` is integer in range [1, 5]
- MUST validate `limit` is integer in range [1, 100]
- MUST return ValidationError for invalid inputs

**FR-2.2: Entity Lookup**
- MUST search for entity by name in FalkorDB
- MUST handle case-insensitive entity matching
- MUST return error if entity not found in graph

**FR-2.3: Graph Traversal**
- MUST delegate to `FalkorDBClient.get_related_entities(entity_id, depth, limit)`
- MUST traverse graph N hops deep (configurable via depth parameter)
- MUST collect all reachable entities within depth limit
- MUST calculate relationship strength (avg strength of edges in path)

**FR-2.4: Result Formatting**
- MUST return list of Entity objects (name, type, description, confidence)
- MUST sort results by relationship strength (descending)
- MUST limit results to `limit` parameter
- MUST format response as human-readable text with entity details

**FR-2.5: Error Handling**
- MUST handle entity not found (return empty list with warning message)
- MUST handle isolated entities (no relationships) gracefully
- MUST handle database errors (return DatabaseError)

---

### 4.3 graph_status Tool

**FR-3.1: Input Validation**
- MUST accept no parameters (or empty dict)
- MUST return error if unexpected parameters provided

**FR-3.2: Statistics Collection**
- MUST delegate to `FalkorDBClient.get_stats()` for base statistics
- MUST query entity count: `MATCH (e:Entity) RETURN count(e)`
- MUST query relationship count: `MATCH ()-[r]->() RETURN count(r)`
- MUST query entity type breakdown: `MATCH (e:Entity) RETURN e.type, count(e)`
- MUST query relationship type breakdown: `MATCH ()-[r]->() RETURN type(r), count(r)`

**FR-3.3: Derived Metrics**
- MUST calculate `graph_coverage` = (memories with entities / total memories) * 100
- MUST calculate `avg_entities_per_memory` = total entities / memories with entities
- MUST calculate `avg_relationships_per_entity` = total relationships / total entities

**FR-3.4: Top Entities Query**
- MUST find most connected entities (top 10 by relationship count)
- MUST return entity name, type, and connection count

**FR-3.5: Response Formatting**
- MUST format as human-readable text report with sections:
  - **Graph Overview:** Total entities, relationships, coverage
  - **Entity Types:** Breakdown by type (PERSON, ORG, TECHNOLOGY, etc.)
  - **Relationship Types:** Breakdown by type (MENTIONS, USES, RELATED_TO, etc.)
  - **Top Entities:** Most connected entities
  - **Health Metrics:** Avg entities per memory, avg relationships per entity

**FR-3.6: Error Handling**
- MUST handle empty graph (return stats with 0 entities/relationships)
- MUST handle database errors (return DatabaseError)

---

## 5. Non-Functional Requirements

### 5.1 Performance

**NFR-1.1: build_graph Performance**
- Target: < 600ms per document for entity extraction
- Target: < 50ms per entity for graph insertion
- Target: < 10 minutes for 1,000 documents (background processing)

**NFR-1.2: get_related Performance**
- Target: < 200ms for depth=1 queries
- Target: < 500ms for depth=2 queries
- Target: < 1s for depth=5 queries (max)

**NFR-1.3: graph_status Performance**
- Target: < 100ms for statistics retrieval (cached aggregates)
- Target: No expensive computations in request path

### 5.2 Reliability

**NFR-2.1: Error Recovery**
- MUST retry database operations up to 3 times with exponential backoff
- MUST provide graceful degradation (partial results on extraction failures)
- MUST log all errors with structured logging (correlation IDs)

**NFR-2.2: Data Integrity**
- MUST ensure atomic graph construction (transaction-based)
- MUST deduplicate entities to prevent duplicate nodes
- MUST validate entity relationships before insertion

### 5.3 Maintainability

**NFR-3.1: Code Quality**
- MUST follow existing MCP tool patterns (AddMemoryTool, SearchMemoryTool)
- MUST use Pydantic models for input validation
- MUST include docstrings with examples
- MUST achieve 80%+ test coverage

**NFR-3.2: Consistency**
- MUST use same error handling patterns as Phase 1 tools
- MUST use same logging patterns as Phase 1 tools
- MUST use same response format as Phase 1 tools (MCP protocol)

### 5.4 Usability

**NFR-4.1: Error Messages**
- MUST provide clear, actionable error messages
- MUST include parameter validation errors with examples
- MUST suggest fixes for common errors (e.g., "entity not found, try searching first")

**NFR-4.2: Documentation**
- MUST include usage examples for each tool
- MUST document input schemas with descriptions
- MUST provide troubleshooting guide for common issues

---

## 6. Constraints & Assumptions

### 6.1 Technical Constraints

**TC-1:** Must use existing `EntityExtractor`, `GraphBuilder`, and `FalkorDBClient` implementations
- No modifications to core/db layers allowed
- Must work within current APIs

**TC-2:** Must follow MCP protocol specification
- Input schema: JSON Schema format
- Output format: MCP response with content array
- Error handling: isError flag + error messages

**TC-3:** Must integrate with existing `MemoryProcessor`
- Use `ProcessorConfig` for feature flags
- Delegate to `MemoryProcessor` methods where possible

**TC-4:** Must use SpaCy for entity extraction
- Requires `en_core_web_sm` model installed
- Cannot change to different NER library

### 6.2 Design Constraints

**DC-1:** Must maintain consistency with Phase 1 tools
- Same file structure: `src/zapomni_mcp/tools/<tool_name>.py`
- Same class structure: `<ToolName>Tool` class with `execute()` method
- Same validation patterns: Pydantic models for requests

**DC-2:** Must not modify database schema
- Use existing FalkorDB schema
- Use existing Entity/Relationship node labels

### 6.3 Assumptions

**A-1:** SpaCy model (`en_core_web_sm`) is installed
- Installation handled by deployment
- Tool initialization fails gracefully if model missing

**A-2:** FalkorDB is running and accessible
- Connection handled by `FalkorDBClient`
- Errors propagated to tool layer

**A-3:** Memories already exist in database
- Tools assume memories were added via `add_memory` tool
- Empty memory set handled gracefully

**A-4:** LLM refinement is disabled (Phase 2 stub)
- Only SpaCy-based entity extraction
- LLM relationship detection not implemented yet

---

## 7. Success Criteria

### 7.1 Acceptance Criteria

**AC-1: Functional Completeness**
- [ ] All 3 tools implemented: build_graph, get_related, graph_status
- [ ] All user stories fulfilled with acceptance criteria met
- [ ] All functional requirements implemented

**AC-2: Quality Metrics**
- [ ] Entity extraction precision: 80%+ (on test corpus)
- [ ] Entity extraction recall: 75%+ (on test corpus)
- [ ] Test coverage: 80%+ for all tools
- [ ] Zero critical bugs in code review

**AC-3: Integration Tests**
- [ ] End-to-end test: add_memory → build_graph → get_related → graph_status
- [ ] Test with real memories (Wikipedia articles, code snippets)
- [ ] Test error scenarios (invalid inputs, missing entities, database errors)

**AC-4: Documentation**
- [ ] README updated with Phase 2 tools
- [ ] API documentation generated (docstrings → docs)
- [ ] Usage examples provided for each tool
- [ ] Troubleshooting guide added

### 7.2 Performance Benchmarks

**PB-1: build_graph**
- [ ] 100 short documents (< 1KB each): < 60s total
- [ ] 10 medium documents (< 10KB each): < 30s total
- [ ] 1 large document (< 100KB): < 10s total

**PB-2: get_related**
- [ ] Depth=1 query: < 200ms (P95)
- [ ] Depth=2 query: < 500ms (P95)
- [ ] Depth=5 query: < 1s (P95)

**PB-3: graph_status**
- [ ] Statistics retrieval: < 100ms (P95)

### 7.3 Definition of Done

- [ ] All code merged to main branch
- [ ] All tests passing (unit + integration)
- [ ] Code reviewed and approved
- [ ] Documentation complete and reviewed
- [ ] Performance benchmarks met
- [ ] No known critical or high-priority bugs
- [ ] Tools registered in `MCPServer.register_all_tools()` (Phase 2 conditional)

---

## 8. Out of Scope

**OS-1:** LLM-based relationship detection
- Current: Stub returns empty relationships
- Future: Phase 2.1 or Phase 3

**OS-2:** Graph visualization
- No UI/rendering of knowledge graphs
- Future: Separate visualization tool

**OS-3:** Graph query language
- No custom query DSL (use Cypher via existing `graph_query` method)
- Future: Phase 3

**OS-4:** Entity disambiguation
- No coreference resolution or entity linking
- Current: Relies on EntityExtractor normalization only

**OS-5:** Incremental graph updates
- No real-time entity extraction on memory addition
- Current: Explicit `build_graph` call required

**OS-6:** Graph analytics
- No centrality metrics, community detection, or PageRank
- Future: Phase 3 advanced analytics

---

## 9. Dependencies

### 9.1 Internal Dependencies

**ID-1:** Phase 1 MVP complete
- Requires: `add_memory`, `search_memory`, `get_stats` tools working
- Requires: `MemoryProcessor`, `FalkorDBClient`, `SemanticChunker` tested

**ID-2:** EntityExtractor implementation
- File: `src/zapomni_core/extractors/entity_extractor.py`
- Status: Implemented (tests passing)

**ID-3:** GraphBuilder implementation
- File: `src/zapomni_core/graph/graph_builder.py`
- Status: Implemented (tests passing)

**ID-4:** FalkorDBClient graph methods
- Methods: `add_entity()`, `add_relationship()`, `get_related_entities()`
- Status: Implemented (tests passing)

### 9.2 External Dependencies

**ED-1:** SpaCy model (en_core_web_sm)
- Version: 3.0+
- Installation: `python -m spacy download en_core_web_sm`

**ED-2:** FalkorDB
- Version: 4.0+
- Status: Required for graph storage

**ED-3:** Python packages
- pydantic: Input validation
- structlog: Structured logging
- mcp: MCP protocol SDK

---

## 10. Risks & Mitigations

### 10.1 Technical Risks

**R-1: Entity Extraction Quality**
- **Risk:** SpaCy-only extraction may have low precision (false positives)
- **Impact:** Graph polluted with incorrect entities
- **Mitigation:** Set confidence threshold (0.7), filter low-confidence entities
- **Mitigation:** Include entity normalization to reduce duplicates

**R-2: Graph Construction Performance**
- **Risk:** Large document sets (10K+ documents) may take hours to process
- **Impact:** User experience degradation, tool timeouts
- **Mitigation:** Implement batch processing with progress tracking
- **Mitigation:** Add `limit` parameter to build_graph (process subset)

**R-3: Database Connection Failures**
- **Risk:** FalkorDB unavailable during graph operations
- **Impact:** Tools fail with DatabaseError
- **Mitigation:** Retry logic with exponential backoff (3 attempts)
- **Mitigation:** Graceful error messages suggesting retry

### 10.2 Usability Risks

**R-4: Complex Error Messages**
- **Risk:** Users confused by internal error details
- **Impact:** Poor user experience, support burden
- **Mitigation:** Sanitize error messages (hide stack traces)
- **Mitigation:** Provide actionable suggestions ("Try X instead")

**R-5: Inconsistent Entity Naming**
- **Risk:** Same entity appears with different names ("Python" vs "python programming language")
- **Impact:** Duplicate nodes, fragmented graph
- **Mitigation:** EntityExtractor normalization removes suffixes, standardizes casing
- **Mitigation:** Document entity normalization rules

### 10.3 Schedule Risks

**R-6: Scope Creep**
- **Risk:** Requests for additional graph features (analytics, visualization)
- **Impact:** Delayed Phase 2 delivery
- **Mitigation:** Strict adherence to spec, defer to Phase 3
- **Mitigation:** Clear "Out of Scope" section in requirements

---

## 11. Appendices

### 11.1 Glossary

- **Entity:** A named concept, person, organization, or technology extracted from text
- **Relationship:** A semantic connection between two entities (e.g., "Python USES Django")
- **Graph Traversal:** Walking the graph along relationship edges to find related entities
- **Knowledge Graph:** A graph database representing entities and their relationships
- **MCP Tool:** A callable function exposed via Model Context Protocol
- **Entity Extraction:** NER (Named Entity Recognition) process to identify entities in text
- **Graph Coverage:** Percentage of memories that have extracted entities

### 11.2 References

- MCP Specification: https://modelcontextprotocol.io/
- SpaCy NER: https://spacy.io/usage/linguistic-features#named-entities
- FalkorDB Cypher: https://docs.falkordb.com/cypher.html
- Phase 1 MCP Tools: `src/zapomni_mcp/tools/`

### 11.3 Example Usage

**Example 1: Build Graph from All Memories**
```json
{
  "tool": "build_graph",
  "arguments": {
    "memory_ids": "all",
    "mode": "full"
  }
}
```

**Response:**
```json
{
  "content": [
    {
      "type": "text",
      "text": "Knowledge graph built successfully.\n\nEntities Created: 127\nEntities Merged: 43\nRelationships Created: 0 (Phase 2 stub)\n\nTotal Nodes: 170\nTotal Edges: 0"
    }
  ],
  "isError": false
}
```

**Example 2: Find Related Entities**
```json
{
  "tool": "get_related",
  "arguments": {
    "entity_name": "Python",
    "depth": 2,
    "limit": 10
  }
}
```

**Response:**
```json
{
  "content": [
    {
      "type": "text",
      "text": "Found 10 entities related to 'Python':\n\n1. Django (TECHNOLOGY)\n   Type: Framework\n   Relationship: USES\n   Strength: 0.95\n\n2. Guido van Rossum (PERSON)\n   Type: Creator\n   Relationship: CREATED_BY\n   Strength: 0.90\n\n..."
    }
  ],
  "isError": false
}
```

**Example 3: View Graph Status**
```json
{
  "tool": "graph_status",
  "arguments": {}
}
```

**Response:**
```json
{
  "content": [
    {
      "type": "text",
      "text": "Knowledge Graph Statistics:\n\n=== Graph Overview ===\nTotal Entities: 170\nTotal Relationships: 0\nGraph Coverage: 85.3% (123/144 memories)\n\n=== Entity Types ===\nTECHNOLOGY: 67 (39%)\nPERSON: 45 (26%)\nORG: 32 (19%)\nCONCEPT: 26 (15%)\n\n=== Relationship Types ===\n(No relationships yet - Phase 2 stub)\n\n=== Top Entities ===\n1. Python (TECHNOLOGY) - 0 connections\n2. OpenAI (ORG) - 0 connections\n...\n\n=== Health Metrics ===\nAvg Entities per Memory: 1.38\nAvg Relationships per Entity: 0.00"
    }
  ],
  "isError": false
}
```

---

**End of Requirements Document**
