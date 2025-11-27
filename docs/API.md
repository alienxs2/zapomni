# Zapomni MCP API Reference

## Table of Contents

- [Overview](#overview)
- [Memory Operations](#memory-operations)
- [Graph Operations](#graph-operations)
- [Code Intelligence](#code-intelligence)
- [System Management](#system-management)
- [Workspace Management](#workspace-management)
- [Error Handling](#error-handling)

## Overview

Zapomni exposes 17 MCP (Model Context Protocol) tools for AI agents. All tools follow a consistent interface:

- **name**: Unique tool identifier
- **description**: Human-readable purpose
- **input_schema**: JSON Schema for input validation
- **execute()**: Async method that performs the operation

**Tool Status**: All 17 tools registered and available

## Memory Operations

### add_memory

Add a memory (text or code) to the knowledge graph.

**Name**: `add_memory`

**Description**: Stores new information in memory with semantic chunking and embeddings. The memory will be processed, chunked, embedded, and stored for later retrieval.

**Input Schema**:
```json
{
  "type": "object",
  "properties": {
    "text": {
      "type": "string",
      "description": "Text content to remember. Can be natural language, code, documentation, or any UTF-8 text. Maximum 10MB.",
      "minLength": 1,
      "maxLength": 10000000
    },
    "metadata": {
      "type": "object",
      "description": "Optional metadata to attach to this memory. Useful for filtering and organization.",
      "properties": {
        "source": {
          "type": "string",
          "description": "Source of the memory (e.g., 'user', 'api', 'file')"
        },
        "tags": {
          "type": "array",
          "items": {"type": "string"},
          "description": "Tags for categorization"
        },
        "timestamp": {
          "type": "string",
          "format": "date-time",
          "description": "Optional timestamp (ISO 8601 format)"
        },
        "language": {
          "type": "string",
          "description": "Programming language if text is code (e.g., 'python')"
        }
      }
    },
    "workspace_id": {
      "type": "string",
      "description": "Optional workspace ID. If not specified, uses the current session workspace."
    }
  },
  "required": ["text"]
}
```

**Output**:
```json
{
  "status": "success",
  "memory_id": "uuid-string",
  "chunks_created": 5,
  "text_preview": "First 100 characters...",
  "error": null
}
```

**Example**:
```python
# Add simple text memory
result = await add_memory(
    text="Python is a high-level programming language.",
    metadata={
        "source": "user",
        "tags": ["programming", "python"]
    }
)
# Returns: memory_id, chunks_created

# Add code memory
result = await add_memory(
    text="""
    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)
    """,
    metadata={
        "language": "python",
        "source": "file",
        "tags": ["algorithm", "recursion"]
    }
)
```

**Processing Pipeline**:
1. Text → ProcessorFactory (detects type)
2. Processor extracts content
3. SemanticChunker splits into chunks (512 tokens, 50 overlap)
4. OllamaEmbedder generates embeddings (768-dim)
5. EntityExtractor extracts entities (optional)
6. FalkorDB stores memory + chunks + entities

**Error Cases**:
- `ValidationError`: Invalid input (e.g., text too long)
- `EmbeddingError`: Failed to generate embeddings
- `DatabaseError`: Failed to store in database

---

### search_memory

Search your personal memory graph for information using semantic similarity.

**Name**: `search_memory`

**Description**: Performs semantic search to find relevant memories based on meaning, not just keyword matching. Returns ranked results with similarity scores.

**Input Schema**:
```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "Natural language search query (e.g., 'information about Python', 'my notes on machine learning')",
      "minLength": 1,
      "maxLength": 1000
    },
    "limit": {
      "type": "integer",
      "description": "Maximum number of results to return (default: 10, max: 100)",
      "default": 10,
      "minimum": 1,
      "maximum": 100
    },
    "filters": {
      "type": "object",
      "description": "Optional metadata filters to narrow results",
      "properties": {
        "tags": {
          "type": "array",
          "items": {"type": "string"},
          "description": "Only return memories with these tags"
        },
        "source": {
          "type": "string",
          "description": "Only return memories from this source"
        },
        "date_from": {
          "type": "string",
          "format": "date",
          "description": "Only memories created after this date (YYYY-MM-DD)"
        },
        "date_to": {
          "type": "string",
          "format": "date",
          "description": "Only memories created before this date (YYYY-MM-DD)"
        }
      }
    },
    "workspace_id": {
      "type": "string",
      "description": "Optional workspace ID override"
    }
  },
  "required": ["query"]
}
```

**Output**:
```json
{
  "results": [
    {
      "memory_id": "uuid",
      "chunk_id": "uuid",
      "text": "Relevant text content...",
      "similarity_score": 0.89,
      "metadata": {
        "source": "user",
        "tags": ["python"]
      },
      "created_at": "2025-11-26T10:00:00Z"
    }
  ],
  "total_results": 10,
  "search_time_ms": 45.2
}
```

**Example**:
```python
# Basic semantic search
results = await search_memory(
    query="What did I learn about Python?",
    limit=5
)

# Search with filters
results = await search_memory(
    query="machine learning algorithms",
    limit=10,
    filters={
        "tags": ["ml", "algorithms"],
        "date_from": "2025-01-01"
    }
)
```

**Search Strategies** (configured via feature flags):
- **Vector Search**: Semantic similarity using HNSW index
- **BM25 Search**: Keyword-based ranking (TF-IDF)
- **Hybrid Search**: Combines vector + BM25 with RRF fusion
- **Reranking**: Cross-encoder reranking for better relevance

---

### get_stats

Get statistics about the memory system.

**Name**: `get_stats`

**Description**: Returns database statistics including total memories, chunks, entities, database size, and performance metrics.

**Input Schema**: No parameters required.

**Output**:
```json
{
  "total_memories": 150,
  "total_chunks": 750,
  "total_entities": 250,
  "total_relationships": 500,
  "database_size_mb": 125.4,
  "avg_chunks_per_memory": 5.0,
  "vector_index_size_mb": 45.2,
  "workspaces": 3,
  "current_workspace": "default"
}
```

**Example**:
```python
stats = await get_stats()
print(f"Total memories: {stats['total_memories']}")
print(f"Database size: {stats['database_size_mb']} MB")
```

---

### delete_memory

Delete a specific memory by ID.

**Name**: `delete_memory`

**Description**: Deletes a memory and all associated chunks, embeddings, and relationships. Requires explicit confirmation for safety.

**Input Schema**:
```json
{
  "type": "object",
  "properties": {
    "memory_id": {
      "type": "string",
      "format": "uuid",
      "description": "UUID of the memory to delete"
    },
    "confirm": {
      "type": "boolean",
      "description": "REQUIRED: Must be true to confirm deletion"
    },
    "workspace_id": {
      "type": "string",
      "description": "Optional workspace ID"
    }
  },
  "required": ["memory_id", "confirm"]
}
```

**Output**:
```json
{
  "status": "success",
  "memory_id": "uuid",
  "chunks_deleted": 5,
  "entities_deleted": 3,
  "relationships_deleted": 10
}
```

**Safety Features**:
- Requires `confirm=true` to proceed
- Cascading delete of all related nodes
- Transaction-based for atomicity

---

## Graph Operations

### build_graph

Build a knowledge graph from text by extracting entities and creating graph structures.

**Name**: `build_graph`

**Description**: Uses EntityExtractor for entity recognition and GraphBuilder for graph construction.

**Input Schema**:
```json
{
  "type": "object",
  "properties": {
    "text": {
      "type": "string",
      "description": "Text to extract entities from and build graph. Maximum 100,000 characters.",
      "minLength": 1,
      "maxLength": 100000
    },
    "options": {
      "type": "object",
      "properties": {
        "extract_entities": {
          "type": "boolean",
          "description": "Enable entity extraction (default: true)",
          "default": true
        },
        "build_relationships": {
          "type": "boolean",
          "description": "Enable relationship detection - Phase 2 LLM (default: false)",
          "default": false
        },
        "confidence_threshold": {
          "type": "number",
          "description": "Minimum confidence for entities (0.0-1.0, default: 0.7)",
          "default": 0.7,
          "minimum": 0.0,
          "maximum": 1.0
        }
      }
    }
  },
  "required": ["text"]
}
```

**Output**:
```json
{
  "status": "success",
  "entities_count": 15,
  "relationships_count": 8,
  "entities_created": 10,
  "entities_merged": 5,
  "processing_time_ms": 1250.5,
  "confidence_avg": 0.85
}
```

**Example**:
```python
# Basic entity extraction
result = await build_graph(
    text="Claude is an AI assistant created by Anthropic. It uses constitutional AI.",
    options={
        "extract_entities": True,
        "confidence_threshold": 0.7
    }
)
# Entities: Claude (PRODUCT), Anthropic (ORG), AI (CONCEPT)

# With relationship detection
result = await build_graph(
    text="Python was created by Guido van Rossum. It's widely used in data science.",
    options={
        "extract_entities": True,
        "build_relationships": True,
        "confidence_threshold": 0.8
    }
)
# Entities + Relationships: CREATED_BY, USED_IN
```

**Entity Types**:
- PERSON, ORG, GPE (geopolitical entity)
- PRODUCT, WORK_OF_ART, LAW
- DATE, TIME, MONEY, QUANTITY
- NORP (nationalities/religions)
- FAC (facilities), LOC (locations)

---

### get_related

Find entities related to a given entity through graph traversal.

**Name**: `get_related`

**Description**: Performs breadth-first search up to 5 hops away. Returns related entities sorted by relationship strength.

**Input Schema**:
```json
{
  "type": "object",
  "properties": {
    "entity_id": {
      "type": "string",
      "format": "uuid",
      "description": "UUID of the entity to find related entities for"
    },
    "depth": {
      "type": "integer",
      "description": "Maximum traversal depth in hops (default: 2, range: 1-5)",
      "default": 2,
      "minimum": 1,
      "maximum": 5
    },
    "limit": {
      "type": "integer",
      "description": "Maximum number of related entities (default: 20, max: 50)",
      "default": 20,
      "minimum": 1,
      "maximum": 50
    },
    "relationship_types": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Optional filter by relationship types (e.g., ['MENTIONS', 'RELATED_TO'])"
    }
  },
  "required": ["entity_id"]
}
```

**Output**:
```json
{
  "entity_id": "uuid",
  "related_entities": [
    {
      "entity_id": "uuid",
      "name": "Anthropic",
      "type": "ORG",
      "relationship_type": "CREATED_BY",
      "distance": 1,
      "strength": 0.95
    }
  ],
  "total_found": 15
}
```

---

### graph_status

Get statistics about the knowledge graph.

**Name**: `graph_status`

**Description**: Returns node counts, relationship counts, entity types, and overall health metrics.

**Input Schema**: No parameters required.

**Output**:
```json
{
  "total_entities": 450,
  "total_relationships": 1200,
  "entity_types": {
    "PERSON": 120,
    "ORG": 80,
    "PRODUCT": 50,
    "CONCEPT": 200
  },
  "relationship_types": {
    "MENTIONS": 800,
    "RELATED_TO": 300,
    "CREATED_BY": 100
  },
  "graph_density": 0.0118,
  "avg_connections_per_entity": 5.3,
  "largest_component_size": 380
}
```

---

### export_graph

Export the knowledge graph to various formats for visualization and analysis.

**Name**: `export_graph`

**Description**: Supports GraphML (Gephi/yEd), Cytoscape JSON (web), Neo4j Cypher, and simple JSON.

**Input Schema**:
```json
{
  "type": "object",
  "properties": {
    "format": {
      "type": "string",
      "enum": ["graphml", "cytoscape", "neo4j", "json"],
      "description": "Export format"
    },
    "output_path": {
      "type": "string",
      "description": "Absolute path to output file. Extension should match format."
    },
    "options": {
      "type": "object",
      "properties": {
        "node_types": {
          "type": "array",
          "items": {"type": "string"},
          "description": "Filter by node types (e.g., ['Entity', 'Memory'])"
        },
        "include_metadata": {
          "type": "boolean",
          "default": true,
          "description": "Include export metadata"
        },
        "pretty_print": {
          "type": "boolean",
          "default": true,
          "description": "Pretty-print output"
        },
        "batch_size": {
          "type": "integer",
          "default": 1000,
          "description": "Batch size for Neo4j export"
        }
      }
    }
  },
  "required": ["format", "output_path"]
}
```

**Output**:
```json
{
  "status": "success",
  "output_path": "/path/to/graph.graphml",
  "nodes_exported": 450,
  "relationships_exported": 1200,
  "file_size_mb": 2.5
}
```

**Supported Formats**:
- **GraphML**: For Gephi, yEd visualization
- **Cytoscape JSON**: For web-based Cytoscape.js
- **Neo4j Cypher**: Import statements for Neo4j
- **JSON**: Simple JSON format

---

## Code Intelligence

### index_codebase

Index entire code repositories with AST-based analysis.

**Name**: `index_codebase`

**Description**: Performs AST parsing, function extraction, call graph analysis, and class hierarchy building.

**Input Schema**:
```json
{
  "type": "object",
  "properties": {
    "path": {
      "type": "string",
      "description": "Absolute path to repository or directory"
    },
    "options": {
      "type": "object",
      "properties": {
        "languages": {
          "type": "array",
          "items": {"type": "string"},
          "description": "Languages to index (e.g., ['python', 'javascript'])"
        },
        "exclude_patterns": {
          "type": "array",
          "items": {"type": "string"},
          "description": "Patterns to exclude (e.g., ['node_modules', '__pycache__'])"
        },
        "extract_functions": {
          "type": "boolean",
          "default": true,
          "description": "Extract function signatures"
        },
        "build_call_graph": {
          "type": "boolean",
          "default": true,
          "description": "Build call graph"
        },
        "analyze_classes": {
          "type": "boolean",
          "default": true,
          "description": "Analyze class hierarchies"
        }
      }
    },
    "workspace_id": {
      "type": "string",
      "description": "Optional workspace ID"
    }
  },
  "required": ["path"]
}
```

**Output**:
```json
{
  "status": "success",
  "files_indexed": 250,
  "functions_extracted": 1200,
  "classes_analyzed": 80,
  "call_graph_edges": 3500,
  "total_loc": 45000,
  "processing_time_ms": 15000
}
```

**Supported Languages**:
- Python, JavaScript, TypeScript
- Go, Rust (experimental)

**Features**:
- AST-based code chunking
- Function signature extraction
- Call graph analysis
- Class hierarchy mapping
- Semantic code search

---

## System Management

### clear_all

Clear ALL data from the knowledge graph.

**Name**: `clear_all`

**Description**: DESTRUCTIVE operation that deletes all stored memories. Requires exact confirmation phrase for safety.

**Input Schema**:
```json
{
  "type": "object",
  "properties": {
    "confirm_phrase": {
      "type": "string",
      "description": "REQUIRED: Must be EXACTLY 'DELETE ALL MEMORIES' (case-sensitive)"
    },
    "workspace_id": {
      "type": "string",
      "description": "Optional workspace ID. Clears only that workspace."
    }
  },
  "required": ["confirm_phrase"]
}
```

**Output**:
```json
{
  "status": "success",
  "memories_deleted": 150,
  "chunks_deleted": 750,
  "entities_deleted": 450,
  "relationships_deleted": 1200
}
```

**Safety Features**:
- Requires EXACT phrase "DELETE ALL MEMORIES"
- Case-sensitive check
- Irreversible operation
- Optional workspace-scoped deletion

---

### prune_memory

Prune stale or orphaned nodes from the knowledge graph.

**Name**: `prune_memory`

**Description**: Removes stale code memories, orphaned chunks, and orphaned entities. Defaults to dry_run=true (preview only).

**Input Schema**:
```json
{
  "type": "object",
  "properties": {
    "strategy": {
      "type": "string",
      "enum": ["stale_code", "orphaned_chunks", "orphaned_entities", "all"],
      "default": "stale_code",
      "description": "What to prune"
    },
    "dry_run": {
      "type": "boolean",
      "default": true,
      "description": "Preview mode - show what would be deleted. Set false for actual deletion."
    },
    "confirm": {
      "type": "boolean",
      "default": false,
      "description": "REQUIRED for deletion: Must be true with dry_run=false"
    },
    "workspace_id": {
      "type": "string",
      "description": "Workspace to prune (default: current workspace)"
    }
  }
}
```

**Output**:
```json
{
  "status": "success",
  "strategy": "stale_code",
  "dry_run": true,
  "items_to_delete": [
    {
      "id": "uuid",
      "type": "Memory",
      "reason": "Stale code memory (file no longer exists)",
      "metadata": {"file_path": "/path/to/deleted/file.py"}
    }
  ],
  "total_items": 15,
  "estimated_space_freed_mb": 5.2
}
```

**Pruning Strategies**:
- **stale_code**: Delete code memories for non-existent files
- **orphaned_chunks**: Delete chunks without parent memories
- **orphaned_entities**: Delete entities without mentions
- **all**: Run all strategies

---

### set_model

Hot-reload Ollama LLM model without restarting the MCP server.

**Name**: `set_model`

**Description**: Changes take effect immediately for entity refinement and relationship extraction. Model must be available via 'ollama pull <model_name>' before use.

**Input Schema**:
```json
{
  "type": "object",
  "properties": {
    "model_name": {
      "type": "string",
      "description": "Ollama model name (e.g., 'qwen2.5:latest', 'llama3:latest'). Model must be pulled first."
    }
  },
  "required": ["model_name"]
}
```

**Output**:
```json
{
  "status": "success",
  "previous_model": "llama3.1:8b",
  "new_model": "qwen2.5:latest",
  "message": "Model updated successfully. Changes take effect immediately."
}
```

**Example**:
```python
# Switch to qwen2.5 for better entity extraction
result = await set_model(model_name="qwen2.5:latest")

# Verify model is available first
# ollama pull qwen2.5:latest
```

---

## Workspace Management

### create_workspace

Create a new workspace for data isolation.

**Name**: `create_workspace`

**Description**: Workspaces allow you to organize memories into separate contexts.

**Input Schema**:
```json
{
  "type": "object",
  "properties": {
    "workspace_id": {
      "type": "string",
      "pattern": "^[a-zA-Z0-9_-]+$",
      "minLength": 1,
      "maxLength": 64,
      "description": "Unique identifier for the workspace"
    },
    "name": {
      "type": "string",
      "minLength": 1,
      "maxLength": 255,
      "description": "Human-readable name for the workspace"
    },
    "description": {
      "type": "string",
      "maxLength": 1000,
      "description": "Optional description of the workspace"
    }
  },
  "required": ["workspace_id", "name"]
}
```

**Output**:
```json
{
  "status": "success",
  "workspace_id": "project-alpha",
  "name": "Project Alpha",
  "created_at": "2025-11-26T10:00:00Z"
}
```

---

### list_workspaces

List all available workspaces.

**Name**: `list_workspaces`

**Description**: Shows workspace ID, name, description, and creation date.

**Input Schema**: No parameters required.

**Output**:
```json
{
  "workspaces": [
    {
      "workspace_id": "default",
      "name": "Default Workspace",
      "description": "Default workspace for general use",
      "created_at": "2025-11-01T00:00:00Z",
      "memory_count": 150,
      "chunk_count": 750
    },
    {
      "workspace_id": "project-alpha",
      "name": "Project Alpha",
      "description": "Workspace for Project Alpha",
      "created_at": "2025-11-26T10:00:00Z",
      "memory_count": 25,
      "chunk_count": 120
    }
  ],
  "total_workspaces": 2
}
```

---

### set_current_workspace

Set the current workspace for this session.

**Name**: `set_current_workspace`

**Description**: All subsequent memory operations will use this workspace.

**Input Schema**:
```json
{
  "type": "object",
  "properties": {
    "workspace_id": {
      "type": "string",
      "pattern": "^[a-zA-Z0-9_-]+$",
      "description": "ID of the workspace to switch to"
    }
  },
  "required": ["workspace_id"]
}
```

**Output**:
```json
{
  "status": "success",
  "previous_workspace": "default",
  "current_workspace": "project-alpha",
  "workspace_name": "Project Alpha"
}
```

---

### get_current_workspace

Get the current workspace for this session.

**Name**: `get_current_workspace`

**Description**: Shows the workspace ID and name, plus statistics about memories.

**Input Schema**: No parameters required.

**Output**:
```json
{
  "workspace_id": "project-alpha",
  "name": "Project Alpha",
  "description": "Workspace for Project Alpha",
  "memory_count": 25,
  "chunk_count": 120,
  "entity_count": 50
}
```

---

### delete_workspace

Delete a workspace and ALL its data.

**Name**: `delete_workspace`

**Description**: WARNING: This is irreversible. All memories, chunks, and entities in the workspace will be permanently deleted.

**Input Schema**:
```json
{
  "type": "object",
  "properties": {
    "workspace_id": {
      "type": "string",
      "pattern": "^[a-zA-Z0-9_-]+$",
      "description": "ID of the workspace to delete"
    },
    "confirm": {
      "type": "boolean",
      "default": false,
      "description": "Must be true to confirm deletion"
    }
  },
  "required": ["workspace_id"]
}
```

**Output**:
```json
{
  "status": "success",
  "workspace_id": "project-alpha",
  "memories_deleted": 25,
  "chunks_deleted": 120,
  "entities_deleted": 50
}
```

**Safety Features**:
- Requires `confirm=true` to proceed
- Cannot delete "default" workspace
- Prevents accidental deletion

---

## Error Handling

All tools return errors in a consistent format:

```json
{
  "error": {
    "type": "ValidationError",
    "message": "Text exceeds maximum length of 10MB",
    "details": {
      "actual_length": 15000000,
      "max_length": 10000000
    }
  },
  "isError": true
}
```

### Error Types

| Error Type | Description | Common Causes |
|------------|-------------|---------------|
| `ValidationError` | Invalid input parameters | Text too long, invalid format |
| `EmbeddingError` | Failed to generate embeddings | Ollama not running, model not found |
| `DatabaseError` | Database operation failed | Connection lost, query timeout |
| `SearchError` | Search operation failed | Invalid query, index corrupted |
| `WorkspaceError` | Workspace operation failed | Workspace not found, already exists |
| `ProcessingError` | Failed to process input | Unsupported format, corrupted file |

### Error Recovery

Tools implement automatic retry with exponential backoff for transient errors:
- Max retries: 3
- Initial delay: 0.1s
- Max delay: 2.0s

---

## Best Practices

### Memory Organization

```python
# Use meaningful metadata
await add_memory(
    text="...",
    metadata={
        "source": "documentation",
        "tags": ["python", "async", "tutorial"],
        "timestamp": "2025-11-26T10:00:00Z",
        "language": "python"
    }
)

# Use workspaces for project isolation
await create_workspace(
    workspace_id="project-alpha",
    name="Project Alpha"
)
await set_current_workspace(workspace_id="project-alpha")
```

### Search Optimization

```python
# Use specific queries
await search_memory(query="async/await syntax in Python")  # Good
await search_memory(query="python")  # Too broad

# Use filters to narrow results
await search_memory(
    query="machine learning",
    filters={
        "tags": ["ml", "tutorial"],
        "date_from": "2025-01-01"
    }
)
```

### Graph Building

```python
# Extract entities with appropriate threshold
await build_graph(
    text="...",
    options={
        "confidence_threshold": 0.8,  # Higher = fewer but more accurate
        "extract_entities": True,
        "build_relationships": False  # Expensive, use sparingly
    }
)
```

---

## Tool Registration Status

| Tool | Registered | Available via MCP |
|------|-----------|------------------|
| add_memory | ✅ | ✅ |
| search_memory | ✅ | ✅ |
| get_stats | ✅ | ✅ |
| delete_memory | ✅ | ✅ |
| build_graph | ✅ | ✅ |
| get_related | ✅ | ✅ |
| graph_status | ✅ | ✅ |
| export_graph | ✅ | ✅ |
| index_codebase | ✅ | ✅ |
| clear_all | ✅ | ✅ |
| prune_memory | ✅ | ✅ |
| set_model | ✅ | ✅ |
| create_workspace | ✅ | ✅ |
| list_workspaces | ✅ | ✅ |
| set_current_workspace | ✅ | ✅ |
| get_current_workspace | ✅ | ✅ |
| delete_workspace | ✅ | ✅ |

**Total**: All 17 tools registered and available

---

## Related Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)**: System architecture and design
- **[CONFIGURATION.md](CONFIGURATION.md)**: Configuration options
- **[CLI.md](CLI.md)**: Command-line tools
- **[DEVELOPMENT.md](DEVELOPMENT.md)**: Development and testing

---

**Document Version**: 1.1
**Last Updated**: 2025-11-27
**Based On**: T0.1 MCP Tools Audit Report
