# Zapomni MCP API Documentation

Complete API reference for Zapomni MCP tools, schemas, and error handling.

## Overview

Zapomni exposes a set of MCP (Model Context Protocol) tools that allow AI agents to store and retrieve information from a local knowledge graph. All tools use structured input validation, semantic processing, and consistent error handling.

## Core Principles

- **Semantic Awareness**: Tools use embeddings and vector similarity for intelligent search
- **Type Safety**: All inputs validated using Pydantic schemas
- **Error Transparency**: Clear error codes and messages for debugging
- **Performance**: Optimized for low-latency queries with caching support
- **Privacy**: All processing stays on your local machine

## Tool Categories

### Phase 1: MVP Tools (Current)

Core memory management tools available now:

- **[add_memory](./tools/add_memory.md)** - Store text with automatic chunking and embeddings
  - Input: Text content + optional metadata
  - Output: Memory ID, chunk count, preview
  - Use: Adding facts, documents, code snippets to memory

- **[search_memory](./tools/search_memory.md)** - Semantic search across memories
  - Input: Natural language query + optional filters
  - Output: Ranked results with similarity scores
  - Use: Retrieving relevant information from memory

- **[get_stats](./tools/get_stats.md)** - View system statistics
  - Input: None (no parameters)
  - Output: Memory count, chunk count, database size, metrics
  - Use: Monitoring memory system health

### Phase 2: Enhanced Search (Future)

- `build_graph` - Extract entities and relationships from memories
- `get_related` - Find related entities via knowledge graph traversal
- `graph_status` - View knowledge graph statistics

### Phase 3: Code Intelligence (Future)

- `index_codebase` - Index code repository with AST analysis
- `delete_memory` - Delete specific memory by ID
- `clear_all` - Clear all memories (with confirmation)
- `export_graph` - Export knowledge graph in various formats

## API Reference

### Tool Structure

Each MCP tool follows a consistent structure:

```
Tool Name (tool_name)
├── Description
├── Parameters
│   ├── name: Type
│   ├── default: value
│   └── validation rules
├── Response (success)
├── Response (error)
└── Error Codes
```

### Request Format

All tool invocations use the MCP protocol format:

```json
{
  "method": "tools/call",
  "params": {
    "name": "tool_name",
    "arguments": {
      "param1": "value1",
      "param2": 42
    }
  }
}
```

### Response Format

Successful responses:

```json
{
  "content": [
    {
      "type": "text",
      "text": "Result text here..."
    }
  ],
  "isError": false
}
```

Error responses:

```json
{
  "content": [
    {
      "type": "text",
      "text": "Error: Description of what went wrong"
    }
  ],
  "isError": true
}
```

## Data Models

See [schemas.md](./schemas.md) for detailed JSON schemas and type definitions:

- **Memory** - A stored piece of information with chunks and embeddings
- **Chunk** - A semantic piece of text with embeddings
- **SearchResult** - A single search result with relevance score
- **SearchResultItem** - Extended search result with metadata

## Error Handling

Comprehensive error handling with error codes for programmatic handling:

See [errors.md](./errors.md) for:

- Error code reference (VAL_*, EMB_*, DB_*, etc.)
- Error types and their characteristics
- Transient vs permanent errors
- Retry strategies
- Common errors and solutions

## Usage Examples

### Python (Using MCP Client)

```python
from mcp import create_client

async with create_client("zapomni") as client:
    # Add memory
    result = await client.call("add_memory", {
        "text": "Python was created by Guido van Rossum in 1991",
        "metadata": {"source": "learning"}
    })

    # Search memory
    results = await client.call("search_memory", {
        "query": "Python creator",
        "limit": 5
    })

    # Get stats
    stats = await client.call("get_stats", {})
```

### cURL (If HTTP transport available)

```bash
# Add memory
curl -X POST http://localhost:5000/mcp/call \
  -H "Content-Type: application/json" \
  -d '{
    "method": "tools/call",
    "params": {
      "name": "add_memory",
      "arguments": {
        "text": "Important information",
        "metadata": {"source": "api"}
      }
    }
  }'
```

## Configuration

MCP server configuration example for Claude:

```json
{
  "mcpServers": {
    "zapomni": {
      "command": "python",
      "args": ["-m", "zapomni_mcp.server"],
      "env": {
        "FALKORDB_HOST": "localhost",
        "FALKORDB_PORT": "6379",
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "OLLAMA_EMBEDDING_MODEL": "nomic-embed-text"
      }
    }
  }
}
```

## Performance Targets (MVP)

- **add_memory latency**: < 2s (P95) - includes chunking, embedding
- **search_memory latency**: < 500ms (P95) - vector similarity search
- **get_stats latency**: < 100ms (P95)
- **Embedding cache hit rate**: 60-68% (Phase 2)
- **Max concurrent requests**: 4 (configurable)

## Limits

- **Max text per memory**: 10 MB (10,000,000 characters)
- **Max query length**: 1,000 characters
- **Max search results**: 100 results
- **Max metadata size**: No limit (but serialized with memory)
- **Vector dimensions**: 768 (nomic-embed-text)

## Tools Reference

| Tool | Parameters | Returns | Error Codes |
|------|-----------|---------|-------------|
| **add_memory** | text, metadata | memory_id, chunks_created | VAL_*, EMB_*, DB_* |
| **search_memory** | query, limit, filters | results array | VAL_*, EMB_*, SEARCH_* |
| **get_stats** | (none) | stats object | DB_* |

## Related Documentation

- **[add_memory](./tools/add_memory.md)** - Detailed parameter and response documentation
- **[search_memory](./tools/search_memory.md)** - Query syntax, filters, ranking explanation
- **[get_stats](./tools/get_stats.md)** - Statistics fields and metrics
- **[schemas.md](./schemas.md)** - JSON schemas and type definitions
- **[errors.md](./errors.md)** - Error codes and troubleshooting
- **[../architecture/](../architecture/)** - System architecture documentation
- **[../development/](../development/)** - Development guide and examples

## Quick Links

### Getting Started

1. [Installation Guide](../installation/)
2. [Configuration](../development/configuration.md)
3. [Hello World Example](../examples/)

### Deep Dive

- [Architecture Overview](../architecture/README.md)
- [Data Flow](../architecture/data_flow.md)
- [Error Handling Strategy](../development/error_handling.md)

### Troubleshooting

- [Common Issues](./errors.md#common-errors-and-solutions)
- [Logging and Debugging](../development/debugging.md)
- [Performance Tuning](../development/performance.md)

## Version Information

- **API Version**: 1.0 (MVP)
- **MCP Protocol**: 2024.11
- **Python Version**: 3.10+
- **Status**: Alpha Development

## Support

For issues, questions, or contributions:

- **GitHub Issues**: [github.com/alienxs2/zapomni/issues](https://github.com/alienxs2/zapomni/issues)
- **GitHub Discussions**: [github.com/alienxs2/zapomni/discussions](https://github.com/alienxs2/zapomni/discussions)
- **Documentation**: [docs/README.md](../README.md)

---

**Last Updated**: 2025-11-24
**License**: MIT
**Author**: Goncharenko Anton aka alienxs2
