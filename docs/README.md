# Zapomni Documentation

Welcome to the Zapomni documentation! This directory contains all public documentation for the Zapomni local-first MCP memory system.

## Documentation Structure

### Getting Started

**[Installation & Quickstart](installation/quickstart.md)**
- Prerequisites and system requirements
- Installation instructions for all platforms
- MCP client configuration
- Verification and first run

### Usage & Features

**[Basic Operations Guide](usage/basic-operations.md)**
- Adding and searching memories
- Understanding search types
- Viewing statistics
- Common usage patterns

### Architecture & Design

**[Architecture Overview](architecture/overview.md)**
- System vision and goals
- Technology stack
- Value proposition
- Quality gates

**[Module Structure](architecture/modules.md)**
- Core modules and responsibilities
- Data models
- Module dependencies
- Testing strategy

**[Data Flow & Interactions](architecture/data-flow.md)**
- Memory indexing flow
- Semantic search patterns
- Entity lifecycle
- Performance optimizations
- Integration points

### Examples

**[Basic Usage Example](examples/basic-usage.py)**
Python script demonstrating:
- Simple memory addition
- Batch operations
- Semantic search
- Entity relationships
- Contextual search
- Metadata filtering

**[MCP Configuration](examples/mcp-client-config.json)**
- Configuration templates for Claude, Cursor, Cline
- Environment variable reference
- Setup steps
- Troubleshooting guide

## Quick Links

### For Users
- Start with [Installation](installation/quickstart.md)
- Learn [Basic Operations](usage/basic-operations.md)
- Run [Examples](examples/)

### For Developers
- Read [Architecture Overview](architecture/overview.md)
- Explore [Module Structure](architecture/modules.md)
- Understand [Data Flow](architecture/data-flow.md)
- Check [../development](../development/) for development setup

### For Contributors
- See root [CONTRIBUTING.md](../CONTRIBUTING.md) (coming soon)
- Review [development setup guide](../development/) (coming soon)
- Check [code style guide](../development/) (coming soon)

## Key Concepts

### Local-First Architecture
Zapomni runs entirely on your machine with zero cloud dependencies. All data, processing, and models stay local for complete privacy and control.

### Knowledge Graph Intelligence
Beyond simple vector similarity, Zapomni understands relationships between concepts through a knowledge graph, enabling more intelligent search and reasoning.

### MCP Protocol
Zapomni integrates with Claude, Cursor, Cline, and other tools via the Model Context Protocol (MCP) for seamless AI agent integration.

### Hybrid Search
Combining vector similarity, keyword matching, and graph traversal for optimal search results across different query types.

## Feature Overview

### Current (MVP)
- [x] Local-first memory system
- [x] Vector embedding storage
- [x] Semantic search
- [x] MCP server with tools
- [x] Redis caching
- [x] Entity extraction

### Upcoming Phases
- [ ] Advanced knowledge graph features
- [ ] Code repository indexing
- [ ] Multi-language support
- [ ] Performance optimizations
- [ ] Extended visualization tools

## Common Tasks

### Add information to memory
See [Basic Operations - Adding Memories](usage/basic-operations.md#adding-memories)

### Search for stored information
See [Basic Operations - Searching](usage/basic-operations.md#searching-memories)

### Configure for your IDE
See [MCP Configuration](examples/mcp-client-config.json)

### Understand system architecture
See [Architecture Overview](architecture/overview.md)

## Troubleshooting

### Connection Issues
Check [MCP Configuration - Troubleshooting](examples/mcp-client-config.json#troubleshooting)

### Performance Problems
See [Data Flow - Performance Optimization](architecture/data-flow.md#performance-optimization-strategies)

### Installation Problems
Refer to [Installation Guide](installation/quickstart.md)

## Support & Community

- **GitHub Issues**: Report bugs and request features
- **GitHub Discussions**: Ask questions and share ideas
- **Contributing**: Help improve Zapomni!

## Documentation Standards

All documentation follows these principles:
- **Clear**: Written for both users and developers
- **Complete**: Covers all aspects of each topic
- **Practical**: Includes examples and real use cases
- **Up-to-date**: Synchronized with code changes
- **Accessible**: Beginner-friendly with links to details

## File Organization

```
docs/
├── README.md                    # This file
├── installation/
│   └── quickstart.md           # Getting started guide
├── usage/
│   └── basic-operations.md     # How to use features
├── architecture/
│   ├── overview.md             # System vision & design
│   ├── modules.md              # Module structure
│   └── data-flow.md            # Data flow patterns
└── examples/
    ├── basic-usage.py          # Python examples
    └── mcp-client-config.json  # Configuration templates
```

## Building Documentation Locally

To view this documentation with a documentation server:

```bash
# Using Python's built-in server
cd docs
python -m http.server 8000

# Then visit http://localhost:8000
```

## Contributing to Documentation

Found a typo? Want to improve documentation? Great!

1. Edit the relevant .md file
2. Test your changes locally
3. Submit a pull request

See [Contributing Guide](../CONTRIBUTING.md) for details.

## License

This documentation is part of Zapomni and is licensed under the MIT License. See [LICENSE](../LICENSE) for details.

---

**Last Updated:** 2025-11-24
**Documentation Version:** 1.0
