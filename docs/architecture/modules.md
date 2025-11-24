# Zapomni Module Structure

## Core Modules

### 1. Core Module (`src/core/`)

**Responsibility:** Foundation layer for memory system operations

**Components:**
- **FalkorDB Client** - Direct database operations and connection management
- **Embedding Generator** - Vector embedding creation (nomic-embed-text)
- **Entity Extractor** - NLP-based entity identification from text
- **Relationship Detector** - Connection discovery between entities
- **Vector Indexer** - Embedding storage and retrieval
- **Graph Builder** - Knowledge graph construction and updates

**Key Responsibilities:**
- Connect to local FalkorDB instance
- Generate semantic embeddings
- Extract named entities from text
- Detect relationships between concepts
- Index content in vector store
- Build and maintain knowledge graph

**Files:**
```
src/core/
├── __init__.py
├── database.py          # FalkorDB operations
├── embeddings.py        # Embedding generation
├── extraction.py        # Entity/relationship extraction
├── indexing.py          # Vector and graph indexing
└── models.py            # Data models (Entity, Relationship, Document)
```

### 2. MCP Module (`src/mcp/`)

**Responsibility:** Model Context Protocol server and tool interface

**Components:**
- **MCP Server** - Protocol handling and tool registration
- **Tool Definitions** - Tool interface specifications
- **Message Handler** - Request/response routing
- **Protocol Handler** - stdio communication management

**Key Responsibilities:**
- Register MCP tools with the protocol
- Handle incoming requests from Claude CLI
- Route requests to appropriate handlers
- Format responses according to MCP spec
- Manage tool discovery and metadata

**Files:**
```
src/mcp/
├── __init__.py
├── server.py            # MCP server implementation
├── tools.py             # Tool definitions and handlers
└── protocol.py          # Protocol handling
```

### 3. Memory Module (`src/memory/`)

**Responsibility:** High-level memory operations and semantic reasoning

**Components:**
- **Memory Store** - Unified memory interface
- **Semantic Search** - Intelligent search using graph and vectors
- **Semantic Indexer** - Content organization and discovery
- **Context Manager** - Maintaining relevant context
- **Reasoning Engine** - Drawing insights and connections

**Key Responsibilities:**
- Provide unified memory interface
- Implement semantic search algorithms
- Organize content for retrieval
- Maintain conversation context
- Extract insights from knowledge graph
- Identify relevant connections

**Files:**
```
src/memory/
├── __init__.py
├── store.py             # Memory store interface
├── search.py            # Semantic search implementation
├── context.py           # Context management
├── reasoning.py         # Insight extraction
└── models.py            # Memory-specific models
```

### 4. Agents Module (`src/agents/`)

**Responsibility:** Task-specific agent implementations

**Components:**
- **Index Agent** - Handles content indexing and updates
- **Search Agent** - Executes search queries
- **Extraction Agent** - Performs entity/relationship extraction
- **Reasoning Agent** - Generates insights and analysis

**Key Responsibilities:**
- Execute specific memory operations
- Coordinate with core and memory modules
- Provide specialized processing
- Handle complex workflows

**Files:**
```
src/agents/
├── __init__.py
├── base_agent.py        # Base agent class
├── index_agent.py       # Indexing operations
├── search_agent.py      # Search operations
├── extraction_agent.py  # Entity extraction
└── reasoning_agent.py   # Insight generation
```

### 5. Utils Module (`src/utils/`)

**Responsibility:** Cross-cutting utilities and helpers

**Components:**
- **Logging** - Structured logging
- **Configuration** - Application config management
- **Error Handling** - Custom exceptions
- **Data Serialization** - JSON/model conversions
- **Performance** - Metrics and profiling

**Key Responsibilities:**
- Provide logging infrastructure
- Manage configuration
- Handle errors consistently
- Serialize/deserialize data
- Track performance metrics

**Files:**
```
src/utils/
├── __init__.py
├── logging.py           # Logging setup
├── config.py            # Configuration management
├── exceptions.py        # Custom exceptions
├── serialization.py     # Data conversion
└── metrics.py           # Performance tracking
```

## Module Dependencies

```
┌─────────────┐
│   MCP CLI   │
└──────┬──────┘
       │ stdio
┌──────▼──────────────────────────────┐
│  MCP Module (src/mcp)               │
│  ├─ Server                          │
│  ├─ Tool Handlers                   │
│  └─ Protocol                        │
└──────┬──────────────────────────────┘
       │
┌──────▼──────────────────────────────┐
│  Memory Module (src/memory)         │
│  ├─ Store                           │
│  ├─ Search                          │
│  ├─ Reasoning                       │
│  └─ Context                         │
└──────┬──────────────────────────────┘
       │
┌──────▼──────────────────────────────┐
│  Agents Module (src/agents)         │
│  ├─ Index Agent                     │
│  ├─ Search Agent                    │
│  ├─ Extraction Agent                │
│  └─ Reasoning Agent                 │
└──────┬──────────────────────────────┘
       │
┌──────▼──────────────────────────────┐
│  Core Module (src/core)             │
│  ├─ Database                        │
│  ├─ Embeddings                      │
│  ├─ Entity Extraction               │
│  ├─ Relationship Detection          │
│  ├─ Indexing                        │
│  └─ Graph Building                  │
└──────┬──────────────────────────────┘
       │
┌──────▼──────────────────────────────┐
│  External Services                  │
│  ├─ FalkorDB                        │
│  ├─ Ollama (Embeddings)             │
│  └─ Local File System               │
└─────────────────────────────────────┘
```

## Data Models

### Entity
- `id`: Unique identifier
- `name`: Entity name
- `type`: Entity type (person, concept, code, etc.)
- `attributes`: Key-value metadata
- `created_at`: Creation timestamp
- `updated_at`: Last update timestamp

### Relationship
- `source_id`: Source entity ID
- `target_id`: Target entity ID
- `type`: Relationship type (relates_to, implements, references, etc.)
- `weight`: Relationship strength (0-1)
- `metadata`: Additional context

### Document
- `id`: Unique identifier
- `content`: Raw text/code content
- `type`: Document type (text, code, message, etc.)
- `entities`: Related entities
- `embeddings`: Vector representation
- `metadata`: Custom metadata
- `created_at`: Creation timestamp

### Memory
- `id`: Unique identifier
- `content`: Memory content
- `entities`: Extracted entities
- `relationships`: Extracted relationships
- `embedding`: Vector representation
- `context`: Surrounding context
- `relevance_score`: Relevance metric

## Configuration

Key configuration points:
- Database connection parameters
- Embedding model selection
- Search algorithm tuning
- Context window size
- Performance optimization flags

See `src/utils/config.py` for implementation details.

## Testing Strategy

### Unit Tests
- Test individual components in isolation
- Mock external dependencies (FalkorDB, Ollama)
- Verify data transformations
- Test error conditions

### Integration Tests
- Test module interactions
- Verify end-to-end workflows
- Test with real FalkorDB instance
- Validate MCP protocol compliance

### Performance Tests
- Benchmark search operations
- Monitor memory usage
- Track embedding generation time
- Measure graph traversal performance

Target: 90%+ code coverage with focus on critical paths.
