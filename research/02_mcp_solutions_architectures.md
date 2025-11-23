# MCP Solutions & Architectures Research

## Executive Summary

Based on comprehensive analysis of Cognee MCP, Claude Context, FalkorDB MCP implementations, and the Model Context Protocol specification, the key findings are:

1. **MCP is a standardized JSON-RPC 2.0 protocol** over stdio transport that enables AI models to access external tools, resources, and prompts through a unified interface
2. **Cognee provides a complete memory system** with knowledge graph generation (cognify), vector search, and graph database integration supporting both Neo4j and NetworkX
3. **Claude Context demonstrates efficient semantic code search** using hybrid BM25 + vector embeddings with Milvus, achieving 40% token reduction
4. **FalkorDB offers native graph database + MCP integration** with Graphiti framework for agentic memory, providing 496x faster P99 latency and 6x better memory efficiency
5. **For Zapomni**: FalkorDB + Ollama stack is validated as viable; we need to implement core MCP tools (add_memory, search, build_graph) using stdio transport and Python SDK

## 1. Cognee MCP Deep Dive

### Repository Analysis

- **URL**: https://github.com/topoteretes/cognee
- **Stars/Activity**: Active development, official MCP server implementation
- **Language**: Python 92.8%, with TypeScript frontend dashboard
- **License**: Apache-2.0
- **Architecture Type**: Modular pipeline-based memory engine

### Architecture

Cognee implements a sophisticated ECL (Extract, Cognify, Load) pipeline architecture:

**Core Components**:
```
┌─────────────────────────────────────────────────┐
│             Cognee Memory Engine                │
├─────────────────────────────────────────────────┤
│  1. Data Ingestion Layer                        │
│     - Text, documents, code repositories        │
│     - Configurable data pipelines               │
│                                                  │
│  2. Cognify Pipeline                            │
│     - Document classification                   │
│     - Text chunking (semantic segments)         │
│     - Entity extraction (LLM-powered)           │
│     - Relationship detection                    │
│     - Graph construction + embeddings           │
│     - Hierarchical summarization                │
│                                                  │
│  3. Storage Layer                               │
│     ├─ Vector DB (LanceDB/Qdrant/PGVector)     │
│     ├─ Graph DB (Neo4j/NetworkX)               │
│     └─ Metadata Store                           │
│                                                  │
│  4. Query Layer                                 │
│     - GRAPH_COMPLETION: LLM + graph context    │
│     - RAG_COMPLETION: Traditional RAG          │
│     - CHUNKS: Raw text retrieval               │
│     - SUMMARIES: Pre-computed hierarchies      │
│     - CODE: Syntax-aware search                │
│     - INSIGHTS: Graph-derived analytics        │
│                                                  │
│  5. MCP Server Interface                        │
│     - Stdio/SSE/HTTP transports                │
│     - JSON-RPC 2.0 protocol                    │
│     - Tool exposure to MCP clients             │
└─────────────────────────────────────────────────┘
```

**Pipeline Task System**:
- Tasks are independent business logic units
- Tasks chain together to form pipelines
- Supports custom user-defined tasks
- Async background processing with status tracking

### Key Features & Functions

**1. cognify(data, graph_model_file=None, graph_model_name=None, custom_prompt=None)**
- **Purpose**: Transform raw data into structured knowledge graph
- **Process**:
  - Document classification and permission validation
  - Semantic text chunking
  - LLM-powered entity extraction
  - Relationship discovery between entities
  - Graph construction with embeddings
  - Content summarization (hierarchical)
- **Requirements**: LLM_API_KEY (mandatory for processing)
- **Returns**: Background task ID (async processing due to time constraints)
- **Custom Schema**: Supports custom graph models via importlib

**2. search(search_query, search_type)**
- **Purpose**: Query the knowledge graph for insights
- **Search Types**:
  - `GRAPH_COMPLETION`: Natural language Q&A with full graph context + LLM reasoning (slowest, most intelligent)
  - `RAG_COMPLETION`: Traditional RAG using document chunks without graph (medium speed)
  - `CHUNKS`: Raw text segments via vector similarity (fastest, no LLM)
  - `SUMMARIES`: Pre-generated hierarchical summaries (fast, pre-computed)
  - `CODE`: Code-specific with syntax understanding (medium speed)
  - `CYPHER`: Direct graph database queries (advanced users)
  - `FEELING_LUCKY`: Auto-selects best search type intelligently
- **Returns**: Format varies by search type (conversational AI, text chunks, code structures, or raw graph results)

**3. add(data)**
- **Purpose**: Ingest new data into memory system
- **Supports**: Text, documents, file paths, any text-extractable content
- **Process**: Queues data for later cognify processing
- **Multi-modal**: Works with natural language, structured data (CSV/JSON), code repos, academic papers

**4. codify(repo_path)**
- **Purpose**: Generate code-specific knowledge graph from software repository
- **Process**: Analyzes code structure, maps relationships, builds code graph
- **Use Cases**: Finding functions, classes, implementation patterns
- **Returns**: Background task ID for status tracking

**5. Additional Tools**:
- `list_data(dataset_id=None)`: List all datasets and data items with IDs
- `delete(data_id, dataset_id, mode="soft")`: Remove specific data (soft/hard deletion)
- `prune()`: Complete reset of knowledge graph (irreversible)
- `cognify_status()`: Check cognify pipeline progress
- `codify_status()`: Check codify pipeline progress

### Code Structure

```
cognee/
├── cognee/                  # Core Python library
│   ├── api/                # Public API functions
│   │   ├── cognify.py     # Main cognify implementation
│   │   ├── search.py      # Search functionality
│   │   └── add.py         # Data ingestion
│   ├── tasks/             # Pipeline task implementations
│   │   ├── chunking/      # Text segmentation
│   │   ├── extraction/    # Entity extraction
│   │   ├── graph/         # Graph construction
│   │   └── summarization/ # Content summarization
│   ├── infrastructure/    # Database adapters
│   │   ├── databases/
│   │   │   ├── vector/    # Vector DB implementations
│   │   │   └── graph/     # Graph DB implementations
│   │   └── llm/           # LLM provider integrations
│   └── shared/            # Utilities and common code
├── cognee-mcp/            # MCP Server implementation
│   ├── src/
│   │   ├── server.py      # Main MCP server entry point
│   │   ├── client.py      # CogneeClient wrapper
│   │   └── tools/         # MCP tool definitions
│   ├── pyproject.toml     # Python package config
│   └── README.md          # MCP setup instructions
├── cognee-frontend/       # React dashboard (optional)
├── cognee-starter-kit/    # Template projects
├── examples/              # Usage demonstrations
├── deployment/            # Docker configs
└── notebooks/             # Jupyter tutorials
```

### MCP Integration Points

**Transport Protocols Supported**:
1. **Stdio** (default): Standard input/output, ideal for local CLI usage
2. **SSE** (Server-Sent Events): Real-time streaming for web deployments
3. **HTTP**: Streamable HTTP for web-based integrations

**Server Launch Commands**:
```bash
# Stdio (default)
python src/server.py

# SSE transport
python src/server.py --transport sse

# HTTP transport
python src/server.py --transport http --host 127.0.0.1 --port 8000 --path /mcp
```

**Claude Desktop Integration**:
```json
{
  "mcpServers": {
    "cognee": {
      "command": "python",
      "args": ["/path/to/cognee-mcp/src/server.py"],
      "env": {
        "LLM_API_KEY": "your-api-key",
        "GRAPH_DATABASE_PROVIDER": "neo4j",
        "VECTOR_DB_PROVIDER": "qdrant"
      }
    }
  }
}
```

**MCP Protocol Implementation**:
- Uses `@modelcontextprotocol/sdk` (Python version)
- Implements JSON-RPC 2.0 over chosen transport
- Exposes tools via standard MCP tool interface
- Supports background task execution with status endpoints

### Technology Stack Used

**Vector Databases**:
- LanceDB (default, embedded)
- Qdrant (scalable vector search)
- PGVector (PostgreSQL extension)
- Weaviate (cloud-native vector database)

**Graph Databases**:
- NetworkX (in-memory, development)
- Neo4j (production-grade, persistent)

**LLM Providers**:
- OpenAI (default)
- Multiple provider support via LLM_PROVIDER env variable
- Rate limiting support (LLM_RATE_LIMIT_ENABLED, LLM_RATE_LIMIT_REQUESTS)

**Pipeline Technologies**:
- Python 3.10-3.13
- Poetry/uv/pip for dependency management
- Async processing for long-running operations
- Background task queue with status tracking

**Frontend (Optional)**:
- React dashboard for visualization
- WebSocket for real-time updates
- REST API for dashboard communication

### Strengths for Zapomni

1. **Complete reference implementation** - Full working MCP server with all features
2. **Modular architecture** - Easy to adapt components for FalkorDB
3. **Multiple search modes** - Can implement progressive feature rollout
4. **Background processing pattern** - Handles long-running tasks elegantly
5. **Flexible database support** - Already supports swapping databases
6. **Local-first design** - Defaults to local storage, aligns with Zapomni goals
7. **Python ecosystem** - Matches our planned tech stack
8. **Active development** - Regular updates and community support
9. **Apache 2.0 license** - Permissive for adaptation
10. **Status tracking** - Built-in job monitoring for user feedback

### What We Can Reuse

**Architecture Patterns**:
- ECL pipeline approach (Extract, Cognify, Load)
- Task-based modular design for pipelines
- Background job processing with status endpoints
- Multi-transport MCP server setup

**Code Components**:
```python
# 1. MCP Server Structure (cognee-mcp/src/server.py pattern)
from mcp.server import Server
from mcp.server.stdio import stdio_server

server = Server("zapomni-memory")

@server.call_tool()
async def add_memory(text: str) -> dict:
    # Implementation
    pass

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)
```

**Database Abstraction Pattern**:
```python
# Abstract database interface Cognee uses
class DatabaseProvider:
    async def add_data_points(self, data_points: List):
        raise NotImplementedError

    async def search(self, query_vector, top_k=10):
        raise NotImplementedError

class FalkorDBProvider(DatabaseProvider):
    # Zapomni implementation for FalkorDB
    pass
```

**Search Type Enumeration**:
```python
from enum import Enum

class SearchType(Enum):
    GRAPH_COMPLETION = "graph_completion"
    RAG_COMPLETION = "rag_completion"
    CHUNKS = "chunks"
    SUMMARIES = "summaries"
```

**Environment Configuration Pattern**:
```python
import os
from dotenv import load_dotenv

load_dotenv()

config = {
    "llm_provider": os.getenv("LLM_PROVIDER", "ollama"),
    "llm_model": os.getenv("LLM_MODEL", "llama3"),
    "graph_db_host": os.getenv("FALKORDB_HOST", "localhost"),
    "graph_db_port": int(os.getenv("FALKORDB_PORT", 6379)),
}
```

## 2. Claude Context Analysis

### Repository Analysis

- **URL**: https://github.com/zilliztech/claude-context
- **Developer**: Zilliz Tech (creators of Milvus vector database)
- **Stars/Activity**: Active community, enterprise-backed
- **Language**: TypeScript (monorepo with pnpm workspaces), Node.js >= 20.0.0, < 24.0.0
- **License**: Open source (license not specified in search results)
- **Purpose**: Semantic code search MCP plugin for AI coding agents

### Architecture

Claude Context implements a **hybrid search architecture** optimized for large codebases:

```
┌──────────────────────────────────────────────────┐
│         Claude Context Architecture              │
├──────────────────────────────────────────────────┤
│  1. Indexing Pipeline                            │
│     ├─ Code Scanner (recursive directory walk)   │
│     ├─ File Filter (.gitignore + custom rules)  │
│     ├─ Embedding Generator (OpenAI/Voyage)       │
│     └─ Index Builder (Milvus vector DB)         │
│                                                   │
│  2. Hybrid Search Engine                         │
│     ├─ BM25 Keyword Search (lexical matching)   │
│     ├─ Dense Vector Search (semantic similarity)│
│     └─ Fusion Scoring (combined relevance)      │
│                                                   │
│  3. MCP Server Layer                             │
│     ├─ Stdio Transport (JSON-RPC 2.0)           │
│     ├─ Tool Exposure (4 main tools)             │
│     └─ Status Tracking (indexing progress)      │
│                                                   │
│  4. Vector Database (Milvus/Zilliz Cloud)        │
│     ├─ Code Embeddings Storage                   │
│     ├─ Metadata (file paths, line numbers)      │
│     └─ Similarity Search                         │
│                                                   │
│  5. Client Integration                           │
│     └─ Claude Code, Cursor, Cline compatible    │
└──────────────────────────────────────────────────┘
```

**Key Design Principles**:
- **Semantic search** over entire codebase (millions of lines)
- **Cost optimization**: Only relevant code in context (not entire directories)
- **Privacy**: Code never leaves machine (when using local Milvus)
- **Universal compatibility**: Works with any MCP client

### Core Capabilities

**1. index_codebase**
- **Purpose**: Index a directory for hybrid search (BM25 + dense vector)
- **Process**:
  - Recursively scans codebase
  - Respects .gitignore and exclusion patterns
  - Generates embeddings via OpenAI/Voyage API
  - Stores in Milvus with metadata (file paths, line numbers)
- **Configuration**: Inclusion/exclusion rules for file types
- **Progress Tracking**: Percentage-based status updates

**2. search_code**
- **Purpose**: Natural language queries using hybrid search
- **Algorithm**:
  - BM25 for keyword/lexical matching
  - Dense vectors for semantic similarity
  - Fusion scoring for combined relevance
- **Input**: Natural language query (e.g., "Find functions that handle user authentication")
- **Output**: Ranked code snippets with metadata
- **Performance**: ~40% token reduction vs loading full directories

**3. clear_index**
- **Purpose**: Remove indexed data for specific codebases
- **Use Case**: Re-indexing after major changes or cleanup

**4. get_indexing_status**
- **Purpose**: Shows progress percentage for active indexing
- **Output**: Completion status and statistics

### MCP Integration Details

**Transport**: Stdio (standard input/output via JSON-RPC 2.0)

**Installation & Configuration**:
```bash
# Command-line setup
claude mcp add claude-context \
  -e OPENAI_API_KEY=sk-your-openai-api-key \
  -e MILVUS_TOKEN=your-zilliz-cloud-api-key \
  -- npx @zilliz/claude-context-mcp@latest
```

**MCP Config (Claude Desktop)**:
```json
{
  "mcpServers": {
    "claude-context": {
      "command": "npx",
      "args": ["@zilliz/claude-context-mcp@latest"],
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "MILVUS_TOKEN": "db_...",
        "EMBEDDING_MODEL": "text-embedding-3-small"
      }
    }
  }
}
```

### Technology Stack

**Backend**:
- **Runtime**: Node.js >= 20.0.0, < 24.0.0
- **Language**: TypeScript
- **Package Manager**: pnpm (monorepo workspaces)
- **MCP SDK**: @modelcontextprotocol/sdk

**Vector Database**:
- **Primary**: Milvus (via Zilliz Cloud)
- **Local Option**: Milvus standalone (Docker)
- **Connection**: Redis protocol (port 6379)

**Embedding Providers**:
- **Default**: OpenAI text-embedding-3-small
- **Alternatives**: text-embedding-3-large, VoyageAI, custom models
- **API**: OpenAI Embeddings API

**Search Technologies**:
- **BM25**: Keyword-based lexical search
- **Vector Similarity**: Cosine similarity for semantic search
- **Hybrid Fusion**: Weighted combination of both scores

### Code Structure

```
claude-context/
├── packages/
│   ├── mcp/                    # Main MCP server package
│   │   ├── src/
│   │   │   ├── server.ts       # MCP server implementation
│   │   │   ├── tools/          # Tool definitions
│   │   │   │   ├── index-codebase.ts
│   │   │   │   ├── search-code.ts
│   │   │   │   ├── clear-index.ts
│   │   │   │   └── get-status.ts
│   │   │   └── transport/      # Stdio transport
│   │   └── package.json
│   │
│   ├── core/                   # Core indexing/search logic
│   │   ├── src/
│   │   │   ├── indexer/        # Code scanning & embedding
│   │   │   ├── searcher/       # Hybrid search implementation
│   │   │   ├── embedding/      # Embedding providers
│   │   │   │   ├── openai-embedding.ts
│   │   │   │   └── voyage-embedding.ts
│   │   │   └── vector-db/      # Milvus integration
│   │   └── package.json
│   │
│   └── python/                 # Python bindings (optional)
│
├── evaluation/                 # Retrieval quality benchmarks
├── examples/                   # Usage examples
│   └── basic-usage/
├── docs/                       # Documentation
├── pnpm-workspace.yaml         # Monorepo config
├── package.json
└── README.md
```

### Performance Metrics

**Token Efficiency**:
- **40% token reduction** while maintaining retrieval quality
- Cost savings in production (fewer API tokens)
- Faster response times (smaller context windows)

**Search Speed**:
- Hybrid search balances accuracy and speed
- Vector similarity: Fast approximate nearest neighbor search
- BM25: Efficient inverted index lookups

**Scalability**:
- Handles millions of lines of code
- Incremental indexing support
- Horizontal scaling via Milvus clustering

### Strengths for Zapomni

1. **Hybrid search architecture** - BM25 + vector is proven effective
2. **TypeScript reference** - Clean, modern codebase to study
3. **Production-ready** - Used by real developers daily
4. **Token efficiency** - 40% reduction is significant cost saving
5. **Privacy-conscious** - Local deployment option
6. **Milvus integration** - Similar to FalkorDB (both support Redis protocol)
7. **Simple tool set** - Just 4 tools cover core use cases
8. **Progress tracking** - Good UX for long-running indexing
9. **Flexible embedding** - Supports multiple providers

### What We Can Adapt for Zapomni

**Hybrid Search Concept**:
```python
# Zapomni can implement similar hybrid approach
async def search_memory(query: str, top_k: int = 10):
    # 1. Vector search component (FalkorDB vector capabilities)
    vector_results = await falkordb.vector_search(
        embedding=embed(query),
        limit=top_k
    )

    # 2. Graph traversal component (FalkorDB Cypher)
    graph_results = await falkordb.query(
        f"MATCH (n) WHERE n.text CONTAINS '{query}' RETURN n"
    )

    # 3. Fusion scoring
    return merge_and_rank(vector_results, graph_results)
```

**Indexing Progress Pattern**:
```python
# Status tracking for long-running operations
class IndexingStatus:
    def __init__(self):
        self.total_files = 0
        self.processed_files = 0
        self.status = "idle"  # idle, running, completed, error

    @property
    def progress_percentage(self):
        if self.total_files == 0:
            return 0
        return (self.processed_files / self.total_files) * 100
```

**File Filtering Logic**:
```python
import pathspec

def should_index_file(file_path: str, gitignore_spec) -> bool:
    # Respect .gitignore patterns
    if gitignore_spec.match_file(file_path):
        return False

    # Custom exclusions (node_modules, build artifacts, etc.)
    exclusions = [
        'node_modules', 'dist', 'build', '.git',
        '*.pyc', '*.log', '*.tmp'
    ]

    for pattern in exclusions:
        if pathspec.match_file(pattern, file_path):
            return False

    return True
```

**Transport Setup Pattern** (TypeScript reference):
```typescript
// For Zapomni Python equivalent
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { Server } from "@modelcontextprotocol/sdk/server/index.js";

const server = new Server({
  name: "claude-context",
  version: "1.0.0"
});

const transport = new StdioServerTransport();
await server.connect(transport);
```

### Differences from Cognee

| Aspect | Claude Context | Cognee | Zapomni Plan |
|--------|----------------|--------|--------------|
| **Focus** | Code search only | General knowledge graphs | General memory + code |
| **Language** | TypeScript | Python | Python |
| **Graph DB** | None (pure vector) | Neo4j/NetworkX | FalkorDB (vector + graph) |
| **Search** | Hybrid BM25 + vector | Graph + vector + RAG | Graph + vector |
| **Embedding** | Required (OpenAI) | Required (OpenAI) | Local (Ollama) |
| **Privacy** | Local option | Local by default | Local only |
| **Complexity** | Simple (4 tools) | Complex (many tools) | Medium (6-8 tools) |

## 3. MCP Protocol Understanding

### Core Concepts

The Model Context Protocol (MCP) is an **open-source standard** for connecting AI assistants to systems where data lives. Announced by Anthropic in November 2024, it solves the "M×N problem" - instead of building M×N integrations between M LLMs and N tools, MCP provides a single standard protocol.

**Key Principles**:
1. **Universal Compatibility**: Any language that can read/write text can implement MCP
2. **Process Isolation**: Natural security boundaries without complex authentication
3. **Debugging Transparency**: Every message is visible and reproducible
4. **Standardized Interface**: JSON-RPC 2.0 over various transports

### Architecture Components

```
┌─────────────────────────────────────────────────┐
│              MCP Architecture                   │
├─────────────────────────────────────────────────┤
│                                                  │
│  ┌──────────────┐         ┌─────────────────┐  │
│  │  MCP Client  │ ◄─────► │   MCP Server    │  │
│  │ (Claude, AI) │  JSON   │ (Your Tool/DB)  │  │
│  └──────────────┘  -RPC   └─────────────────┘  │
│         │            2.0           │            │
│         │                          │            │
│         ▼                          ▼            │
│  ┌──────────────┐         ┌─────────────────┐  │
│  │   Sampling   │         │  Tools          │  │
│  │   Roots      │         │  Resources      │  │
│  └──────────────┘         │  Prompts        │  │
│                            └─────────────────┘  │
│                                                  │
│         Transport Layer (stdio/SSE/HTTP)        │
│                                                  │
└─────────────────────────────────────────────────┘
```

**Role Definitions**:
- **MCP Host**: The AI application (Claude Desktop, Cursor, Cline)
- **MCP Client**: Bridge within host that connects to servers
- **MCP Server**: Your custom tool that provides data/capabilities
- **Data Sources**: APIs, databases, files, or services behind the server

### Three Core Primitives

#### 1. Tools (Server → Client)

**Definition**: Executable functions that perform actions or computations

**Use Cases**:
- Querying a database
- Searching the web
- Sending emails
- Processing files
- Mathematical calculations

**Implementation Pattern**:
```python
from mcp.server import Server
from pydantic import Field

server = Server("my-server")

@server.call_tool()
async def search_memory(
    query: str = Field(description="Search query"),
    limit: int = Field(default=10, description="Max results")
) -> dict:
    """Search the knowledge graph for relevant information."""
    results = await db.search(query, limit=limit)
    return {
        "content": [{
            "type": "text",
            "text": json.dumps(results, indent=2)
        }]
    }
```

**Characteristics**:
- AI model decides when to call tools
- User typically approves tool calls (security)
- Tools can have side effects (mutations)
- Input validation via Pydantic or Zod schemas

#### 2. Resources (Server → Client)

**Definition**: Data entities that servers expose for clients to read

**Use Cases**:
- Configuration files
- User profiles
- Database records
- Static content (greetings, documentation)
- Dynamic data (API responses)

**Implementation Pattern**:
```python
@server.list_resources()
async def list_resources() -> list[Resource]:
    """List available memory resources."""
    return [
        Resource(
            uri="memory://recent",
            name="Recent Memories",
            description="Last 100 memory entries",
            mimeType="application/json"
        ),
        Resource(
            uri="memory://graph",
            name="Knowledge Graph",
            description="Full graph structure",
            mimeType="application/json"
        )
    ]

@server.read_resource()
async def read_resource(uri: str) -> str:
    """Read resource content by URI."""
    if uri == "memory://recent":
        memories = await db.get_recent(100)
        return json.dumps(memories)
    elif uri == "memory://graph":
        graph = await db.export_graph()
        return json.dumps(graph)
    raise ValueError(f"Unknown resource: {uri}")
```

**Characteristics**:
- Application consumes resources (not direct user invocation)
- Can be static or dynamic
- URI-based addressing
- Read-only (no mutations)

#### 3. Prompts (Server → Client)

**Definition**: Reusable templates that guide AI interactions

**Use Cases**:
- Question templates
- Instruction patterns
- Conversation starters
- Context injection

**Implementation Pattern**:
```python
@server.list_prompts()
async def list_prompts() -> list[Prompt]:
    """List available prompt templates."""
    return [
        Prompt(
            name="search_memory",
            description="Search knowledge graph with context",
            arguments=[
                PromptArgument(
                    name="topic",
                    description="Topic to search for",
                    required=True
                )
            ]
        )
    ]

@server.get_prompt()
async def get_prompt(name: str, arguments: dict) -> GetPromptResult:
    """Get prompt template with filled arguments."""
    if name == "search_memory":
        topic = arguments["topic"]
        return GetPromptResult(
            messages=[
                PromptMessage(
                    role="user",
                    content=f"Search the knowledge graph for information about: {topic}\n"
                            f"Include related concepts and relationships."
                )
            ]
        )
```

**Characteristics**:
- User invokes prompts explicitly
- Support dynamic payloads (argument substitution)
- Autocomplete in MCP clients
- Guide conversation structure

### Client-Side Primitives

**Sampling** (Client capability):
- Allows server to request LLM completions
- Server can ask AI to generate text
- Enables recursive AI reasoning patterns

**Roots** (Client capability):
- Client exposes filesystem roots to server
- Server can understand context boundaries
- Security: Limited to specific directories

### Transport Mechanisms

#### Stdio Transport

**How it works**:
```
Client Process                 Server Process
     │                              │
     │  stdin  (client → server)   │
     ├──────────────────────────────►
     │                              │
     │  stdout (server → client)   │
     ◄──────────────────────────────┤
     │                              │
     │  stderr (logs, diagnostics)  │
     ◄──────────────────────────────┤
```

**Message Format**:
- Newline-delimited JSON-RPC 2.0 messages
- Each message must not contain embedded newlines
- Bidirectional communication

**Example Message**:
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "search_memory",
    "arguments": {
      "query": "machine learning concepts"
    }
  }
}
```

**Advantages**:
- Simple: No network configuration needed
- Secure: OS-level process isolation
- Debuggable: Messages easily logged and replayed
- Universal: Works on all platforms

**Disadvantages**:
- Local only (no remote servers)
- Single client per server process
- No multiplexing

#### SSE (Server-Sent Events) Transport

**How it works**:
- HTTP-based server-to-client streaming
- Client sends requests via HTTP POST
- Server streams responses via SSE

**Use Cases**:
- Web-based integrations
- Real-time updates
- Cloud deployments

#### HTTP Transport

**How it works**:
- Streamable HTTP requests/responses
- Supports long-polling
- Can be proxied and load-balanced

**Use Cases**:
- Production web services
- Multi-client scenarios
- RESTful integrations

### Creating MCP Server - Step-by-Step

#### Step 1: Choose SDK and Initialize Project

**Python**:
```bash
# Install MCP SDK
pip install mcp

# Create project structure
mkdir zapomni-mcp
cd zapomni-mcp
touch server.py
```

**TypeScript**:
```bash
npm init
npm install @modelcontextprotocol/sdk zod
touch server.ts
```

#### Step 2: Create Server Instance

**Python**:
```python
from mcp.server import Server
from mcp.server.stdio import stdio_server
import asyncio

# Create server
server = Server("zapomni-memory")

async def main():
    # Run with stdio transport
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
```

**TypeScript**:
```typescript
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";

const server = new Server({
  name: "zapomni-memory",
  version: "1.0.0"
}, {
  capabilities: {
    tools: {}
  }
});

const transport = new StdioServerTransport();
await server.connect(transport);
```

#### Step 3: Implement Tools

**Python Example**:
```python
from pydantic import Field
import json

@server.call_tool()
async def add_memory(
    text: str = Field(description="Text to remember"),
    tags: list[str] = Field(default=[], description="Optional tags")
) -> dict:
    """Add new information to memory."""
    # Store in FalkorDB
    memory_id = await db.add_memory(text, tags)

    return {
        "content": [{
            "type": "text",
            "text": f"Memory added with ID: {memory_id}"
        }]
    }

@server.call_tool()
async def search_memory(
    query: str = Field(description="Search query"),
    limit: int = Field(default=10)
) -> dict:
    """Search memories by query."""
    results = await db.search(query, limit)

    return {
        "content": [{
            "type": "text",
            "text": json.dumps(results, indent=2)
        }]
    }
```

#### Step 4: Add Resources (Optional)

```python
from mcp.types import Resource

@server.list_resources()
async def list_resources() -> list[Resource]:
    return [
        Resource(
            uri="zapomni://stats",
            name="Memory Statistics",
            mimeType="application/json"
        )
    ]

@server.read_resource()
async def read_resource(uri: str) -> str:
    if uri == "zapomni://stats":
        stats = await db.get_statistics()
        return json.dumps(stats)
```

#### Step 5: Configure Client Integration

**Claude Desktop** (`claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "zapomni": {
      "command": "python",
      "args": ["/path/to/zapomni-mcp/server.py"],
      "env": {
        "FALKORDB_HOST": "localhost",
        "FALKORDB_PORT": "6379",
        "OLLAMA_HOST": "http://localhost:11434"
      }
    }
  }
}
```

**Cursor IDE**:
```json
{
  "mcpServers": {
    "zapomni": {
      "command": "python",
      "args": ["/path/to/server.py"]
    }
  }
}
```

### Standard Functions for Memory MCP Server

Based on Cognee, Claude Context, and MCP best practices:

**Core Functions**:
1. `add_memory(text, metadata)` - Ingest new information
2. `search_memory(query, filters)` - Query stored knowledge
3. `build_graph(data)` - Construct knowledge graph
4. `get_related(entity_id)` - Find related entities
5. `delete_memory(memory_id)` - Remove specific memory
6. `clear_all()` - Reset database
7. `get_stats()` - Database statistics
8. `export_graph()` - Export full graph structure

**Optional Advanced Functions**:
9. `summarize(topic)` - Generate summaries
10. `find_connections(entity1, entity2)` - Path finding
11. `cluster_memories(algorithm)` - Group similar memories
12. `temporal_search(query, time_range)` - Time-based filtering

### Best Practices

1. **Input Validation**: Always use Pydantic (Python) or Zod (TypeScript) for schema validation
2. **Error Handling**: Wrap external calls in try/catch with meaningful errors
3. **Async/Await**: Use async functions for I/O operations (database, APIs)
4. **Structured Responses**: Return consistent JSON with `type` and `text` fields
5. **Security**: Never expose sensitive data without user approval
6. **Documentation**: Use docstrings and descriptions for AI discoverability
7. **Logging**: Write logs to stderr (stdout reserved for MCP messages)
8. **Environment Variables**: Store configuration in `.env` files
9. **Graceful Degradation**: Handle missing dependencies or services elegantly
10. **Status Feedback**: Provide progress updates for long-running operations

## 4. Comparison Table

| Feature | Cognee MCP | Claude Context | FalkorDB MCP | Zapomni Plan |
|---------|------------|----------------|--------------|--------------|
| **Primary Language** | Python | TypeScript | TypeScript | Python |
| **Vector Database** | LanceDB/Qdrant/PGVector/Weaviate | Milvus (Zilliz Cloud) | N/A (integrated) | FalkorDB (built-in) |
| **Graph Database** | Neo4j/NetworkX | None (pure vector) | FalkorDB | FalkorDB |
| **LLM Provider** | OpenAI (+ others) | OpenAI/Voyage | Configurable | Ollama (local) |
| **Local Support** | Yes (default) | Yes (optional) | Yes | Yes (required) |
| **MCP Transport** | Stdio/SSE/HTTP | Stdio | HTTP | Stdio |
| **License** | Apache-2.0 | Open source | MIT | TBD |
| **Main Use Case** | General knowledge graphs | Code search only | Graph RAG | General memory + code |
| **Search Types** | 6+ types (Graph, RAG, Chunks, Code, Summaries, Cypher) | Hybrid (BM25 + vector) | Cypher queries | Graph + Vector hybrid |
| **Entity Extraction** | Yes (LLM-powered) | No | Optional (Graphiti) | Yes (Ollama-powered) |
| **Relationship Mapping** | Yes (automatic) | No | Yes (graph-native) | Yes (automatic) |
| **Background Processing** | Yes (async tasks) | Yes (indexing) | N/A | Yes (planned) |
| **Status Tracking** | Yes (cognify_status, codify_status) | Yes (get_indexing_status) | No | Yes (planned) |
| **Code Indexing** | Yes (codify tool) | Yes (primary feature) | No | Yes (planned) |
| **Multi-tenant** | Via datasets | Via codebase separation | Yes (isolated graphs) | No (single user) |
| **Performance Metrics** | Not specified | 40% token reduction | 496x faster P99, 6x memory efficiency | TBD |
| **Dashboard UI** | Yes (React frontend) | No | No | Optional (future) |
| **Tool Count** | 10+ tools | 4 tools | 4 endpoints | 6-8 tools (planned) |
| **Embedding Required** | Yes | Yes | Yes (for vector features) | Yes (local via Ollama) |
| **Privacy Level** | High (local default) | High (local option) | High (self-hosted) | Maximum (local only) |
| **Cloud Option** | Yes (API mode) | Yes (Zilliz Cloud) | Yes (hosted FalkorDB) | No |
| **Docker Support** | Yes | Yes | Yes | Yes (planned) |
| **Custom Schema** | Yes (graph_model_file) | No | Yes (Cypher) | Yes (planned) |
| **Rate Limiting** | Yes (configurable) | N/A | N/A | Optional |
| **Production Ready** | Yes | Yes | Yes | In development |
| **Documentation** | Extensive | Good | Good | In progress |
| **Community Activity** | Active | Active | Growing | N/A (new) |

### Key Insights from Comparison

1. **Cognee is most feature-complete** for general knowledge graphs with 10+ tools and multiple search types
2. **Claude Context is specialized** for code search with proven 40% token efficiency
3. **FalkorDB MCP is performance-focused** with 496x faster queries and native graph support
4. **Zapomni occupies middle ground**: General memory + code, local-only, Python-based

### Strategic Advantages of Our Stack

**FalkorDB + Ollama Combination**:
1. **Unified Database**: FalkorDB handles both vector and graph in one system (unlike Cognee's split architecture)
2. **Local LLM**: Ollama eliminates API costs and privacy concerns (unlike Cognee/Claude Context)
3. **Performance**: FalkorDB's proven 496x speed advantage over alternatives
4. **Memory Efficiency**: 6x better than competitors
5. **Simplicity**: Fewer moving parts than Cognee (no separate vector + graph DBs)
6. **Redis Protocol**: Standard interface, well-documented, battle-tested

## 5. Recommendations for Zapomni

### Core MCP Functions Needed

Based on analysis of Cognee, Claude Context, and MCP best practices, Zapomni should implement these prioritized functions:

#### Phase 1: MVP (Essential Functions)

**1. add_memory(text: str, metadata: dict = None) -> dict**
- **Purpose**: Ingest new information into memory
- **Process**:
  - Generate embeddings via Ollama
  - Store in FalkorDB with vector + graph
  - Extract entities (optional in MVP)
- **Return**: Memory ID and status
- **Inspiration**: Cognee's `add()` + Claude Context's `index_codebase`

**2. search_memory(query: str, limit: int = 10, filters: dict = None) -> dict**
- **Purpose**: Query stored memories
- **Process**:
  - Hybrid search (vector similarity + graph traversal)
  - Rank results by relevance
- **Return**: Ranked memory entries with metadata
- **Inspiration**: Cognee's `search()` + Claude Context's `search_code`

**3. get_stats() -> dict**
- **Purpose**: Database statistics and health check
- **Return**: Total memories, entities, relationships, storage size
- **Inspiration**: Resource pattern from MCP spec

#### Phase 2: Knowledge Graph (Core Intelligence)

**4. build_graph(memory_id: str = None) -> dict**
- **Purpose**: Generate knowledge graph from memories
- **Process**:
  - Entity extraction via Ollama
  - Relationship detection
  - Graph construction in FalkorDB
- **Return**: Task ID for async processing
- **Inspiration**: Cognee's `cognify()`

**5. get_related(entity: str, depth: int = 1) -> dict**
- **Purpose**: Find related entities and memories
- **Process**: Graph traversal via Cypher queries
- **Return**: Related entities with relationship types
- **Inspiration**: FalkorDB's native graph capabilities

**6. graph_status(task_id: str = None) -> dict**
- **Purpose**: Check graph building progress
- **Return**: Status, percentage, errors
- **Inspiration**: Cognee's `cognify_status()`

#### Phase 3: Advanced Features

**7. index_codebase(path: str, exclusions: list = None) -> dict**
- **Purpose**: Index code repositories
- **Process**:
  - Scan files (respect .gitignore)
  - Extract code structure
  - Build code graph
- **Return**: Task ID for async processing
- **Inspiration**: Cognee's `codify()` + Claude Context's `index_codebase`

**8. delete_memory(memory_id: str, mode: str = "soft") -> dict**
- **Purpose**: Remove specific memory
- **Process**: Soft (mark deleted) or hard (full removal)
- **Return**: Deletion confirmation
- **Inspiration**: Cognee's `delete()`

**9. clear_all() -> dict**
- **Purpose**: Reset entire database (with confirmation)
- **Return**: Success status
- **Inspiration**: Cognee's `prune()`

**10. export_graph(format: str = "json") -> dict**
- **Purpose**: Export knowledge graph
- **Formats**: JSON, GraphML, Cypher
- **Return**: Graph data structure
- **Inspiration**: MCP Resource pattern

### Architecture Proposal

```
┌─────────────────────────────────────────────────────────────┐
│                  Zapomni MCP Architecture                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │             MCP Server Layer                        │    │
│  │  - Stdio Transport (JSON-RPC 2.0)                  │    │
│  │  - Tool Definitions (10 functions)                 │    │
│  │  - Input Validation (Pydantic)                     │    │
│  │  - Error Handling & Logging                        │    │
│  └────────────────────────────────────────────────────┘    │
│                         │                                    │
│                         ▼                                    │
│  ┌────────────────────────────────────────────────────┐    │
│  │           Memory Processing Engine                  │    │
│  │                                                      │    │
│  │  ┌────────────────────────────────────────────┐   │    │
│  │  │  Embedding Generator (Ollama)              │   │    │
│  │  │  - Model: nomic-embed-text / mxbai-embed   │   │    │
│  │  │  - Local HTTP API (port 11434)             │   │    │
│  │  └────────────────────────────────────────────┘   │    │
│  │                                                      │    │
│  │  ┌────────────────────────────────────────────┐   │    │
│  │  │  Entity Extraction (Ollama LLM)            │   │    │
│  │  │  - Model: llama3 / mistral / mixtral       │   │    │
│  │  │  - Prompt-based extraction                 │   │    │
│  │  └────────────────────────────────────────────┘   │    │
│  │                                                      │    │
│  │  ┌────────────────────────────────────────────┐   │    │
│  │  │  Graph Builder                              │   │    │
│  │  │  - Relationship detection                   │   │    │
│  │  │  - Cypher query generation                  │   │    │
│  │  └────────────────────────────────────────────┘   │    │
│  │                                                      │    │
│  └────────────────────────────────────────────────────┘    │
│                         │                                    │
│                         ▼                                    │
│  ┌────────────────────────────────────────────────────┐    │
│  │          FalkorDB Storage Layer                     │    │
│  │                                                      │    │
│  │  ┌─────────────────────────────────────────┐       │    │
│  │  │  Vector Index                            │       │    │
│  │  │  - Embeddings storage                    │       │    │
│  │  │  - Similarity search                     │       │    │
│  │  │  - HNSW/Flat index                       │       │    │
│  │  └─────────────────────────────────────────┘       │    │
│  │                                                      │    │
│  │  ┌─────────────────────────────────────────┐       │    │
│  │  │  Property Graph                          │       │    │
│  │  │  - Entity nodes (name, type, properties) │       │    │
│  │  │  - Relationship edges                    │       │    │
│  │  │  - Cypher query engine                   │       │    │
│  │  └─────────────────────────────────────────┘       │    │
│  │                                                      │    │
│  │  ┌─────────────────────────────────────────┐       │    │
│  │  │  Metadata Store                          │       │    │
│  │  │  - Memory timestamps                     │       │    │
│  │  │  - Tags and categories                   │       │    │
│  │  │  - Source information                    │       │    │
│  │  └─────────────────────────────────────────┘       │    │
│  │                                                      │    │
│  └────────────────────────────────────────────────────┘    │
│                         │                                    │
│                         ▼                                    │
│  ┌────────────────────────────────────────────────────┐    │
│  │           Background Task Manager                   │    │
│  │  - Async job queue                                  │    │
│  │  - Status tracking (in-progress, completed, failed) │    │
│  │  - Progress percentage updates                      │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Directory Structure Proposal

```
zapomni/
├── zapomni-mcp/                    # MCP server package
│   ├── src/
│   │   ├── server.py               # Main MCP server entry point
│   │   ├── tools/                  # MCP tool implementations
│   │   │   ├── __init__.py
│   │   │   ├── memory.py           # add_memory, search_memory
│   │   │   ├── graph.py            # build_graph, get_related
│   │   │   ├── code.py             # index_codebase
│   │   │   ├── management.py       # delete, clear, export
│   │   │   └── status.py           # get_stats, graph_status
│   │   ├── resources/              # MCP resources (optional)
│   │   │   └── stats.py
│   │   └── config.py               # Configuration management
│   ├── tests/                      # Unit and integration tests
│   ├── pyproject.toml              # Python package config
│   ├── .env.example                # Environment template
│   └── README.md                   # Setup instructions
│
├── zapomni-core/                   # Core processing engine
│   ├── src/
│   │   ├── embeddings/             # Ollama embedding integration
│   │   │   ├── __init__.py
│   │   │   ├── ollama_client.py   # Ollama API client
│   │   │   └── cache.py           # Embedding cache
│   │   ├── extraction/             # Entity extraction
│   │   │   ├── __init__.py
│   │   │   ├── entity_extractor.py # LLM-based extraction
│   │   │   └── prompts.py         # Extraction prompts
│   │   ├── graph/                  # Graph operations
│   │   │   ├── __init__.py
│   │   │   ├── builder.py         # Graph construction
│   │   │   └── queries.py         # Common Cypher queries
│   │   ├── search/                 # Hybrid search
│   │   │   ├── __init__.py
│   │   │   ├── vector_search.py   # Vector similarity
│   │   │   ├── graph_search.py    # Graph traversal
│   │   │   └── hybrid.py          # Fusion ranking
│   │   └── tasks/                  # Background tasks
│   │       ├── __init__.py
│   │       ├── task_manager.py    # Job queue
│   │       └── status_tracker.py  # Progress tracking
│   └── tests/
│
├── zapomni-db/                     # FalkorDB integration
│   ├── src/
│   │   ├── __init__.py
│   │   ├── connection.py           # FalkorDB client
│   │   ├── vector_ops.py           # Vector operations
│   │   ├── graph_ops.py            # Graph operations
│   │   ├── schema.py               # Data models
│   │   └── migrations/             # Schema migrations
│   └── tests/
│
├── scripts/                        # Utility scripts
│   ├── setup_falkordb.sh          # FalkorDB Docker setup
│   ├── test_ollama.py             # Ollama connectivity test
│   └── benchmark.py               # Performance testing
│
├── docs/                           # Documentation
│   ├── architecture.md
│   ├── api_reference.md
│   ├── setup_guide.md
│   └── examples/
│
├── research/                       # Research notes
│   ├── 01_tech_stack.md
│   ├── 02_mcp_solutions_architectures.md
│   └── benchmarks/
│
├── docker/                         # Docker configurations
│   ├── docker-compose.yml         # Multi-container setup
│   ├── Dockerfile.mcp             # MCP server container
│   └── Dockerfile.falkordb        # FalkorDB container
│
├── .env.example                    # Global environment template
├── .gitignore
├── pyproject.toml                  # Monorepo config
└── README.md                       # Project overview
```

### Implementation Priority

#### Phase 1: Foundation (Week 1-2)

**Objectives**:
- Basic MCP server running
- FalkorDB connection established
- Simple memory add/search working

**Tasks**:
1. Set up project structure (directories, pyproject.toml)
2. Install dependencies (mcp SDK, falkordb-py, ollama)
3. Create basic MCP server with stdio transport
4. Implement FalkorDB connection module
5. Test Ollama embeddings generation
6. Implement `add_memory()` tool (basic version)
7. Implement `search_memory()` tool (vector only)
8. Implement `get_stats()` tool
9. Test with Claude Desktop / Cursor
10. Write basic documentation

**Deliverables**:
- Working MCP server with 3 tools
- FalkorDB storing memories with embeddings
- Basic search functionality
- Setup guide for users

#### Phase 2: Knowledge Graph (Week 3-4)

**Objectives**:
- Entity extraction working
- Graph relationships created
- Hybrid search implemented

**Tasks**:
1. Design entity extraction prompts for Ollama
2. Implement entity extractor using llama3
3. Create graph builder module
4. Define Cypher schema for entities/relationships
5. Implement `build_graph()` tool with async processing
6. Implement background task manager
7. Implement `graph_status()` tool
8. Implement `get_related()` tool
9. Enhance `search_memory()` with graph context
10. Performance testing and optimization

**Deliverables**:
- Knowledge graph generation working
- Hybrid search (vector + graph)
- Background processing with status
- Performance benchmarks

#### Phase 3: Advanced Features (Week 5-6)

**Objectives**:
- Code indexing functional
- Management tools complete
- Production-ready quality

**Tasks**:
1. Implement file scanner with .gitignore support
2. Create code-specific entity extraction
3. Implement `index_codebase()` tool
4. Implement `delete_memory()` tool
5. Implement `clear_all()` tool with confirmation
6. Implement `export_graph()` tool
7. Add comprehensive error handling
8. Add logging and diagnostics
9. Create Docker setup (docker-compose)
10. Write comprehensive documentation
11. Create example notebooks/scripts
12. Security review and hardening

**Deliverables**:
- Full feature set (10 tools)
- Production-ready code quality
- Docker deployment
- Complete documentation
- Example projects

#### Phase 4: Polish & Optimization (Week 7-8)

**Objectives**:
- Performance optimization
- User experience improvements
- Community readiness

**Tasks**:
1. Benchmark against Cognee/Claude Context
2. Optimize FalkorDB queries
3. Implement embedding caching
4. Add configuration file support (.zapomnirc)
5. Create CLI tool (optional)
6. Add telemetry (optional, privacy-respecting)
7. Write troubleshooting guide
8. Create video tutorials
9. Set up GitHub repository
10. Prepare for public release

**Deliverables**:
- Optimized performance
- Excellent documentation
- Public GitHub repository
- Community-ready project

### Technical Decisions Summary

**Why Python over TypeScript?**
- Ollama has excellent Python SDK
- FalkorDB Python client is mature
- MCP Python SDK is official and well-documented
- Easier for ML/AI integrations
- Matches Cognee's approach (proven successful)

**Why Stdio Transport?**
- Simplest to implement and debug
- Secure by default (process isolation)
- Supported by all major MCP clients
- No network configuration needed
- Can add HTTP later if needed

**Why FalkorDB over Neo4j + Separate Vector DB?**
- Unified storage (simpler architecture)
- 496x faster performance
- 6x better memory efficiency
- Redis protocol (familiar, well-documented)
- Built-in vector support (no separate DB needed)
- Lower operational complexity

**Why Ollama over OpenAI?**
- 100% local (no API costs, no privacy concerns)
- Supports embedding models (nomic-embed-text)
- Supports LLMs for entity extraction (llama3, mistral)
- Easy to install and run
- Good Python SDK
- Aligns with "local-first" philosophy

**Why Async Background Processing?**
- Graph building takes time (can't block MCP calls)
- Better user experience (immediate response)
- Matches Cognee's proven pattern
- Enables progress tracking
- Scales better for large datasets

## 6. Code Examples & Patterns

### Example 1: Basic MCP Server Setup

```python
# zapomni-mcp/src/server.py

from mcp.server import Server
from mcp.server.stdio import stdio_server
from pydantic import Field
import asyncio
import json
import logging
import sys

# Configure logging (stderr only, stdout reserved for MCP)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# Create server instance
server = Server("zapomni-memory")

# Import core modules
from zapomni_core.embeddings import OllamaEmbeddings
from zapomni_db.connection import FalkorDBClient

# Initialize dependencies
db = FalkorDBClient()
embedder = OllamaEmbeddings()

@server.call_tool()
async def add_memory(
    text: str = Field(description="Text to remember"),
    metadata: dict = Field(default={}, description="Optional metadata (tags, source, etc.)")
) -> dict:
    """Add new information to memory."""
    try:
        logger.info(f"Adding memory: {text[:50]}...")

        # Generate embedding
        embedding = await embedder.embed(text)

        # Store in FalkorDB
        memory_id = await db.add_memory(
            text=text,
            embedding=embedding,
            metadata=metadata
        )

        logger.info(f"Memory added with ID: {memory_id}")

        return {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "status": "success",
                    "memory_id": memory_id,
                    "text_preview": text[:100]
                }, indent=2)
            }]
        }
    except Exception as e:
        logger.error(f"Error adding memory: {e}")
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "status": "error",
                    "message": str(e)
                })
            }],
            "isError": True
        }

@server.call_tool()
async def search_memory(
    query: str = Field(description="Search query"),
    limit: int = Field(default=10, description="Maximum results to return"),
    min_score: float = Field(default=0.5, description="Minimum similarity score (0-1)")
) -> dict:
    """Search memories by semantic similarity."""
    try:
        logger.info(f"Searching: {query}")

        # Generate query embedding
        query_embedding = await embedder.embed(query)

        # Vector search in FalkorDB
        results = await db.search_similar(
            embedding=query_embedding,
            limit=limit,
            min_score=min_score
        )

        logger.info(f"Found {len(results)} results")

        return {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "status": "success",
                    "count": len(results),
                    "results": results
                }, indent=2)
            }]
        }
    except Exception as e:
        logger.error(f"Search error: {e}")
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "status": "error",
                    "message": str(e)
                })
            }],
            "isError": True
        }

@server.call_tool()
async def get_stats() -> dict:
    """Get database statistics."""
    try:
        stats = await db.get_statistics()

        return {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "status": "success",
                    "statistics": stats
                }, indent=2)
            }]
        }
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "status": "error",
                    "message": str(e)
                })
            }],
            "isError": True
        }

async def main():
    """Main server entry point."""
    logger.info("Starting Zapomni MCP Server...")

    # Initialize database connection
    await db.connect()
    logger.info("Connected to FalkorDB")

    # Run server with stdio transport
    async with stdio_server() as (read_stream, write_stream):
        logger.info("MCP server running on stdio")
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
```

### Example 2: FalkorDB Client Implementation

```python
# zapomni-db/src/connection.py

import os
from typing import List, Dict, Any, Optional
from falkordb import FalkorDB
import json
import logging

logger = logging.getLogger(__name__)

class FalkorDBClient:
    """FalkorDB client for Zapomni memory storage."""

    def __init__(
        self,
        host: str = None,
        port: int = None,
        graph_name: str = "zapomni_memory"
    ):
        self.host = host or os.getenv("FALKORDB_HOST", "localhost")
        self.port = port or int(os.getenv("FALKORDB_PORT", 6379))
        self.graph_name = graph_name
        self.client = None
        self.graph = None

    async def connect(self):
        """Establish connection to FalkorDB."""
        try:
            self.client = FalkorDB(host=self.host, port=self.port)
            self.graph = self.client.select_graph(self.graph_name)

            # Initialize schema
            await self._initialize_schema()

            logger.info(f"Connected to FalkorDB at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"FalkorDB connection failed: {e}")
            raise

    async def _initialize_schema(self):
        """Create indexes and constraints."""
        try:
            # Create vector index for embeddings
            self.graph.query("""
                CREATE VECTOR INDEX FOR (m:Memory) ON (m.embedding)
                OPTIONS {dimension: 768, similarityFunction: 'cosine'}
            """)

            # Create index on memory ID
            self.graph.query("""
                CREATE INDEX FOR (m:Memory) ON (m.id)
            """)

            logger.info("Schema initialized")
        except Exception as e:
            # Index may already exist
            logger.warning(f"Schema initialization: {e}")

    async def add_memory(
        self,
        text: str,
        embedding: List[float],
        metadata: Dict[str, Any] = None
    ) -> str:
        """Add memory to graph database."""
        import uuid

        memory_id = str(uuid.uuid4())
        metadata = metadata or {}

        # Prepare Cypher query
        query = """
            CREATE (m:Memory {
                id: $id,
                text: $text,
                embedding: $embedding,
                tags: $tags,
                source: $source,
                timestamp: timestamp()
            })
            RETURN m.id as id
        """

        params = {
            "id": memory_id,
            "text": text,
            "embedding": embedding,
            "tags": metadata.get("tags", []),
            "source": metadata.get("source", "user")
        }

        result = self.graph.query(query, params)

        logger.info(f"Memory created: {memory_id}")
        return memory_id

    async def search_similar(
        self,
        embedding: List[float],
        limit: int = 10,
        min_score: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Search for similar memories using vector similarity."""

        query = """
            CALL db.idx.vector.queryNodes('Memory', 'embedding', $limit, $embedding)
            YIELD node, score
            WHERE score >= $min_score
            RETURN
                node.id as id,
                node.text as text,
                node.tags as tags,
                node.source as source,
                node.timestamp as timestamp,
                score
            ORDER BY score DESC
        """

        params = {
            "embedding": embedding,
            "limit": limit,
            "min_score": min_score
        }

        result = self.graph.query(query, params)

        memories = []
        for record in result.result_set:
            memories.append({
                "id": record[0],
                "text": record[1],
                "tags": record[2],
                "source": record[3],
                "timestamp": record[4],
                "similarity_score": record[5]
            })

        return memories

    async def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""

        # Count memories
        count_query = "MATCH (m:Memory) RETURN count(m) as count"
        result = self.graph.query(count_query)
        memory_count = result.result_set[0][0] if result.result_set else 0

        # Count entities (for future graph features)
        entity_query = "MATCH (e:Entity) RETURN count(e) as count"
        result = self.graph.query(entity_query)
        entity_count = result.result_set[0][0] if result.result_set else 0

        # Count relationships
        rel_query = "MATCH ()-[r]->() RETURN count(r) as count"
        result = self.graph.query(rel_query)
        relationship_count = result.result_set[0][0] if result.result_set else 0

        return {
            "total_memories": memory_count,
            "total_entities": entity_count,
            "total_relationships": relationship_count,
            "graph_name": self.graph_name,
            "database_host": self.host
        }

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a specific memory."""
        query = "MATCH (m:Memory {id: $id}) DELETE m"
        self.graph.query(query, {"id": memory_id})
        logger.info(f"Memory deleted: {memory_id}")
        return True

    async def clear_all(self) -> bool:
        """Clear entire database (use with caution!)."""
        query = "MATCH (n) DETACH DELETE n"
        self.graph.query(query)
        logger.warning("All data cleared from database")
        return True
```

### Example 3: Ollama Embeddings Client

```python
# zapomni-core/src/embeddings/ollama_client.py

import os
import httpx
from typing import List
import logging

logger = logging.getLogger(__name__)

class OllamaEmbeddings:
    """Ollama embeddings client for local embedding generation."""

    def __init__(
        self,
        model: str = None,
        base_url: str = None
    ):
        self.model = model or os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        self.base_url = base_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.client = httpx.AsyncClient(timeout=30.0)
        logger.info(f"Initialized Ollama embeddings with model: {self.model}")

    async def embed(self, text: str) -> List[float]:
        """Generate embedding for text."""
        try:
            response = await self.client.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text
                }
            )
            response.raise_for_status()

            data = response.json()
            embedding = data.get("embedding")

            if not embedding:
                raise ValueError("No embedding in response")

            logger.debug(f"Generated embedding of dimension: {len(embedding)}")
            return embedding

        except httpx.HTTPError as e:
            logger.error(f"Ollama HTTP error: {e}")
            raise
        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            raise

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = []
        for text in texts:
            embedding = await self.embed(text)
            embeddings.append(embedding)
        return embeddings

    async def test_connection(self) -> bool:
        """Test Ollama connection and model availability."""
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()

            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]

            if self.model in model_names:
                logger.info(f"Model {self.model} is available")
                return True
            else:
                logger.warning(f"Model {self.model} not found. Available: {model_names}")
                return False

        except Exception as e:
            logger.error(f"Ollama connection test failed: {e}")
            return False
```

### Example 4: Entity Extraction with Ollama LLM

```python
# zapomni-core/src/extraction/entity_extractor.py

import os
import httpx
import json
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class EntityExtractor:
    """Extract entities and relationships using Ollama LLM."""

    def __init__(
        self,
        model: str = None,
        base_url: str = None
    ):
        self.model = model or os.getenv("OLLAMA_LLM_MODEL", "llama3")
        self.base_url = base_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.client = httpx.AsyncClient(timeout=60.0)

    async def extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract entities and relationships from text."""

        prompt = f"""Extract entities and relationships from the following text.
Return ONLY valid JSON, no other text.

Format:
{{
  "entities": [
    {{"name": "entity name", "type": "Person|Organization|Location|Concept|Event", "properties": {{}}}},
  ],
  "relationships": [
    {{"from": "entity1", "to": "entity2", "type": "RELATION_TYPE"}},
  ]
}}

Text:
{text}

JSON:"""

        try:
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json"
                }
            )
            response.raise_for_status()

            data = response.json()
            response_text = data.get("response", "")

            # Parse JSON response
            extracted = json.loads(response_text)

            logger.info(f"Extracted {len(extracted.get('entities', []))} entities, "
                       f"{len(extracted.get('relationships', []))} relationships")

            return extracted

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            return {"entities": [], "relationships": []}
        except Exception as e:
            logger.error(f"Entity extraction error: {e}")
            return {"entities": [], "relationships": []}
```

### Example 5: Background Task Manager

```python
# zapomni-core/src/tasks/task_manager.py

import asyncio
import uuid
from typing import Dict, Any, Callable, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class Task:
    def __init__(self, task_id: str, name: str, func: Callable, *args, **kwargs):
        self.task_id = task_id
        self.name = name
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.status = TaskStatus.PENDING
        self.progress = 0.0
        self.result = None
        self.error = None
        self.created_at = None
        self.started_at = None
        self.completed_at = None

class TaskManager:
    """Manage background asynchronous tasks."""

    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self._lock = asyncio.Lock()

    def create_task(
        self,
        name: str,
        func: Callable,
        *args,
        **kwargs
    ) -> str:
        """Create and queue a new task."""
        task_id = str(uuid.uuid4())

        task = Task(task_id, name, func, *args, **kwargs)
        task.created_at = asyncio.get_event_loop().time()

        self.tasks[task_id] = task

        # Start task in background
        asyncio.create_task(self._run_task(task_id))

        logger.info(f"Task created: {task_id} - {name}")
        return task_id

    async def _run_task(self, task_id: str):
        """Execute task in background."""
        async with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                return

            task.status = TaskStatus.RUNNING
            task.started_at = asyncio.get_event_loop().time()

        try:
            logger.info(f"Running task: {task_id}")

            # Execute task function
            if asyncio.iscoroutinefunction(task.func):
                result = await task.func(*task.args, **task.kwargs)
            else:
                result = task.func(*task.args, **task.kwargs)

            async with self._lock:
                task.status = TaskStatus.COMPLETED
                task.result = result
                task.progress = 100.0
                task.completed_at = asyncio.get_event_loop().time()

            logger.info(f"Task completed: {task_id}")

        except Exception as e:
            logger.error(f"Task failed: {task_id} - {e}")

            async with self._lock:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.completed_at = asyncio.get_event_loop().time()

    def get_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status."""
        task = self.tasks.get(task_id)
        if not task:
            return None

        return {
            "task_id": task.task_id,
            "name": task.name,
            "status": task.status.value,
            "progress": task.progress,
            "error": task.error,
            "result": task.result,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at
        }

    def list_tasks(self) -> List[Dict[str, Any]]:
        """List all tasks."""
        return [self.get_status(task_id) for task_id in self.tasks.keys()]
```

### Example 6: Configuration Management

```python
# zapomni-mcp/src/config.py

import os
from typing import Any, Dict
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class DatabaseConfig:
    host: str = os.getenv("FALKORDB_HOST", "localhost")
    port: int = int(os.getenv("FALKORDB_PORT", 6379))
    graph_name: str = os.getenv("FALKORDB_GRAPH", "zapomni_memory")
    password: str = os.getenv("FALKORDB_PASSWORD", "")

@dataclass
class OllamaConfig:
    host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    llm_model: str = os.getenv("OLLAMA_LLM_MODEL", "llama3")
    timeout: int = int(os.getenv("OLLAMA_TIMEOUT", 60))

@dataclass
class ServerConfig:
    name: str = "zapomni-memory"
    version: str = "0.1.0"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

class Config:
    """Global configuration."""

    def __init__(self):
        self.database = DatabaseConfig()
        self.ollama = OllamaConfig()
        self.server = ServerConfig()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "database": self.database.__dict__,
            "ollama": self.ollama.__dict__,
            "server": self.server.__dict__
        }

# Global config instance
config = Config()
```

## 7. Resources

### Official Documentation

1. **Model Context Protocol**:
   - Official Site: https://modelcontextprotocol.io/
   - GitHub: https://github.com/modelcontextprotocol
   - Spec: https://docs.anthropic.com/en/docs/mcp
   - Courses: https://anthropic.skilljar.com/introduction-to-model-context-protocol

2. **Cognee MCP**:
   - GitHub: https://github.com/topoteretes/cognee
   - MCP Server: https://github.com/topoteretes/cognee/tree/main/cognee-mcp
   - Documentation: https://deepwiki.com/topoteretes/cognee/10.3-mcp-server-configuration
   - Playbook: https://playbooks.com/mcp/topoteretes-cognee-mcp

3. **Claude Context**:
   - GitHub: https://github.com/zilliztech/claude-context
   - README: https://github.com/zilliztech/claude-context/blob/master/README.md
   - Examples: https://github.com/zilliztech/claude-context/tree/master/examples

4. **FalkorDB**:
   - Main Site: https://www.falkordb.com/
   - GitHub: https://github.com/FalkorDB/FalkorDB
   - MCP Server: https://github.com/FalkorDB/FalkorDB-MCPServer
   - Docs: https://docs.falkordb.com/
   - Graphiti Integration: https://docs.falkordb.com/agentic-memory/graphiti-mcp-server.html

5. **Ollama**:
   - Official Site: https://ollama.ai/
   - GitHub: https://github.com/ollama/ollama
   - Python SDK: https://github.com/ollama/ollama-python
   - Model Library: https://ollama.ai/library

### Community Resources

6. **Awesome MCP Servers**:
   - https://github.com/punkpeye/awesome-mcp-servers
   - https://github.com/wong2/awesome-mcp-servers

7. **Tutorials & Guides**:
   - FreeCodeCamp TypeScript Tutorial: https://www.freecodecamp.org/news/how-to-build-a-custom-mcp-server-with-typescript-a-handbook-for-developers/
   - MCPcat Guide: https://mcpcat.io/guides/building-mcp-server-typescript/
   - Composio Guide: https://composio.dev/blog/mcp-server-step-by-step-guide-to-building-from-scrtch
   - Microsoft MCP for Beginners: https://github.com/microsoft/mcp-for-beginners

8. **Example Repositories**:
   - Official MCP Servers: https://github.com/modelcontextprotocol/servers
   - GitHub MCP Server: https://github.com/github/github-mcp-server
   - Simple MCP Example: https://github.com/alejandro-ao/mcp-server-example
   - Quick MCP Example: https://github.com/ALucek/quick-mcp-example

### Technical References

9. **JSON-RPC & Transport**:
   - Transports Guide: https://modelcontextprotocol.io/docs/concepts/transports
   - JSON-RPC in MCP: https://mcpcat.io/guides/understanding-json-rpc-protocol-mcp/
   - STDIO Transport: https://mcp-framework.com/docs/Transports/stdio-transport/

10. **Knowledge Graphs & RAG**:
    - GraphRAG Overview: https://www.falkordb.com/blog/mcp-integration-falkordb-graphrag/
    - Graphiti Framework: https://blog.getzep.com/graphiti-knowledge-graphs-falkordb-support/
    - Knowledge Graphs for LLMs: https://www.anthropic.com/news/model-context-protocol

### Python Libraries

11. **Key Dependencies**:
    - `mcp` - Official MCP SDK for Python
    - `falkordb` - FalkorDB Python client (Redis-compatible)
    - `httpx` - Async HTTP client for Ollama
    - `pydantic` - Data validation and settings
    - `python-dotenv` - Environment variable management

### Docker & Deployment

12. **Container Images**:
    - FalkorDB: `docker pull falkordb/falkordb:latest`
    - Ollama: `docker pull ollama/ollama:latest`

### Blog Posts & Articles

13. **Medium & Dev.to**:
    - MCP Deep Dive: https://medium.com/@amanatulla1606/anthropics-model-context-protocol-mcp-a-deep-dive-for-developers-1d3db39c9fdc
    - Building MCP Servers: https://dev.to/shadid12/how-to-build-mcp-servers-with-typescript-sdk-1c28
    - MCP for Any Language: https://dev.to/yigit-konur/building-mcp-servers-for-any-language-including-kotlin-ruby-rust-java-go-typescript--2ofi

14. **Industry Coverage**:
    - InfoQ: https://www.infoq.com/news/2024/12/anthropic-model-context-protocol/
    - Weights & Biases: https://wandb.ai/onlineinference/mcp/reports/The-Model-Context-Protocol-MCP-by-Anthropic-Origins-functionality-and-impact--VmlldzoxMTY5NDI4MQ

---

**Research Completed**: 2025-11-22
**Next Steps**: Implement Phase 1 (Foundation) - Basic MCP server with add_memory, search_memory, get_stats tools
