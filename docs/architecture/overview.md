# Zapomni Architecture Overview

## Project Vision

**"AI agents with perfect memory—private, intelligent, truly theirs"**

Zapomni is a local-first MCP memory system that gives AI agents a "second brain" for:
- Remembering everything they've learned, forever
- Understanding relationships between concepts, not just similarities
- Complete privacy (never sends data to external servers)
- Zero recurring costs
- Full offline capability

## Core Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **Database** | FalkorDB | Unified vector+graph, 496x faster P99, 6x memory efficient |
| **LLM Runtime** | Ollama | 100% local, easy installation, excellent API |
| **Embeddings** | nomic-embed-text | 81.2% accuracy, 2048 token context, multilingual |
| **LLM Reasoning** | Llama 3.1 / DeepSeek-R1 / Qwen2.5 | State-of-art local reasoning |
| **Language** | Python 3.10+ | MCP SDK native, ML ecosystem, matches research |
| **Protocol** | MCP (stdio) | Official Anthropic standard, simple, secure |

## System Architecture

```
User (Claude CLI, Cursor, Cline)
    ↓ MCP stdio protocol
Zapomni Server (Python)
    ├─ MCP Tool Interface
    ├─ Agent Management
    └─ Memory System
        ├─ Vector Search
        ├─ Knowledge Graph
        ├─ Semantic Reasoning
        └─ Code Analysis
            ↓
FalkorDB (Local)
    ├─ Vector Embeddings
    ├─ Entity Relationships
    └─ Knowledge Graph
```

## Key Characteristics

### 1. Spec-Driven Development
- Three-level specifications: Module → Component → Function
- Each level has tests and implementation guidance
- Clear accountability and progress tracking

### 2. Knowledge Graph Intelligence
- Entity extraction from natural language
- Relationship detection between concepts
- Code-aware AST-based indexing
- Call graph analysis for dependencies

### 3. Hybrid Search
- Vector similarity search
- Graph traversal for relationships
- Combined ranking for relevance
- Context-aware completion

### 4. MCP Protocol Integration
- Seamless integration with Claude CLI
- Works with Cursor and Cline
- stdio-based communication (secure, no network)
- Tool discovery and dynamic registration

## Value Proposition

### Problems Solved
1. **Privacy Concerns** - Cloud dependencies create data exposure risks
2. **High Costs** - Recurring API fees for embedding and graph services
3. **Performance Issues** - Latency and memory inefficiency at scale
4. **Limited Intelligence** - Basic similarity search without relationships
5. **Vendor Lock-in** - Proprietary solutions limit flexibility

### Our Solution
- ✅ 100% local execution (zero API costs, complete privacy)
- ✅ FalkorDB unified database (496x faster P99, 6x memory efficient)
- ✅ Knowledge graph intelligence (entities, relationships, context)
- ✅ Code-aware analysis (AST-based indexing, call graphs)
- ✅ MCP-native protocol (seamless AI agent integration)

## Project Scope

**MVP Deliverables (3 Core Tools):**
1. **Memory Tool** - Add/retrieve semantic knowledge
2. **Search Tool** - Query with vector and graph intelligence
3. **Reasoning Tool** - Extract insights and connections

**Implementation Phases:**
- Phase 1: Foundation (database, core indexing, MCP server)
- Phase 2: Intelligence (knowledge graph, entity extraction)
- Phase 3: Integration (CLI tools, dashboard, examples)

## Quality Gates

Before MVP release:
1. ✅ Module/Component specifications complete
2. ✅ 90%+ test coverage achieved
3. ✅ All critical path functions tested
4. ✅ Integration tests passing
5. ✅ Documentation complete
6. ✅ Example implementations validated

## Next Steps

See the following documentation for more details:
- **modules.md** - Detailed module structure and responsibilities
- **data-flow.md** - Data flow and interaction patterns
- **../installation/quickstart.md** - Getting started guide
- **../usage/** - Feature documentation and examples
