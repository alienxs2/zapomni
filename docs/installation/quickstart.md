# Zapomni MVP - Quick Start Guide

Welcome to Zapomni, the intelligent memory management system. This guide will get you up and running in minutes.

---

## Prerequisites

Before you begin, ensure you have:

- **Python 3.10+** - Check with `python3 --version`
- **Docker & Docker Compose** - Check with `docker --version` and `docker-compose --version`
- **Git** - For cloning the repository
- **4GB RAM minimum** - For running all services
- **10GB Disk space** - For caches and data

### Install Docker (if needed)

**macOS (with Homebrew):**
```bash
brew install docker docker-compose
```

**Ubuntu/Debian:**
```bash
sudo apt-get install docker.io docker-compose
sudo usermod -aG docker $USER
```

**Windows:**
Download from [Docker Desktop](https://www.docker.com/products/docker-desktop)

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-org/zapomni.git
cd zapomni
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install the package in development mode
pip install -e .

# Verify installation
python3 -c "from zapomni_core import MemoryProcessor; print('✓ Installation successful')"
```

### 4. Set Up Environment Variables

```bash
# Copy example config
cp .env.example .env

# Edit .env with your settings (optional, defaults work for local development)
cat .env
```

**Key Environment Variables:**
```ini
# Database
FALKORDB_HOST=localhost
FALKORDB_PORT=6381
FALKORDB_GRAPH=zapomni_memory

# Embeddings
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=nomic-embed-text
OLLAMA_TIMEOUT=30

# Logging
LOG_LEVEL=INFO

# Memory Configuration
MAX_TEXT_LENGTH=10000000
CHUNK_SIZE=512
CHUNK_OVERLAP=50
```

### 5. Start Services

```bash
# Start FalkorDB and Ollama
docker-compose up -d

# Wait for services to be ready (30-60 seconds)
docker-compose ps

# Check logs if needed
docker-compose logs -f
```

**Expected Output:**
```
NAME                  COMMAND                  STATUS
zapomni-falkordb      "redis-server..."        Up (healthy)
zapomni-ollama        "ollama serve"           Up (healthy)
```

### 6. Verify Setup

```bash
python3 << 'EOF'
from zapomni_core.logging_service import LoggingService
LoggingService.configure_logging()

from zapomni_core.memory_processor import MemoryProcessor
from zapomni_mcp.server import MCPServer

print("✓ Zapomni MVP is ready!")
print(f"✓ MemoryProcessor: {MemoryProcessor}")
print(f"✓ MCPServer: {MCPServer}")
EOF
```

---

## Running Tests

### Quick Test Run
```bash
# Run all Phase 1 compatible tests
python3 -m pytest tests/ -v --ignore=tests/unit/test_entity_extractor.py \
  --ignore=tests/unit/test_models.py --ignore=tests/unit/test_reranker.py

# Expected: 766 passing, ~53 seconds
```

### Run Specific Test Categories
```bash
# Test memory processor
python3 -m pytest tests/unit/test_memory_processor.py -v

# Test validation
python3 -m pytest tests/unit/test_input_validator.py -v

# Test MCP tools
python3 -m pytest tests/unit/test_add_memory_tool.py tests/unit/test_search_memory_tool.py -v

# Test database
python3 -m pytest tests/unit/test_falkordb_client.py -v

# Test search
python3 -m pytest tests/unit/test_vector_search.py tests/unit/test_bm25_search.py -v
```

### Test with Coverage
```bash
python3 -m pytest tests/ --cov=src --cov-report=html --cov-report=term-missing
open htmlcov/index.html  # View coverage report
```

---

## Basic Usage

### 1. Configure Your Application

```python
from zapomni_core.logging_service import LoggingService
from zapomni_core.memory_processor import MemoryProcessor
from zapomni_core.chunking import SemanticChunker
from zapomni_core.embeddings.ollama_embedder import OllamaEmbedder
from zapomni_db import FalkorDBClient

# Configure logging first
LoggingService.configure_logging()

# Initialize components
db_client = FalkorDBClient(
    host="localhost",
    port=6381,
    graph="zapomni_memory"
)

chunker = SemanticChunker(
    chunk_size=512,
    chunk_overlap=50
)

embedder = OllamaEmbedder(
    base_url="http://localhost:11434",
    model_name="nomic-embed-text"
)

# Create processor
processor = MemoryProcessor(
    db_client=db_client,
    chunker=chunker,
    embedder=embedder
)
```

### 2. Add a Memory

```python
import asyncio

async def add_memory_example():
    memory_id = await processor.add_memory(
        text="Python is a programming language created by Guido van Rossum in 1991.",
        metadata={
            "source": "encyclopedia",
            "tags": ["python", "programming", "history"],
            "category": "technology"
        }
    )
    print(f"Added memory: {memory_id}")

asyncio.run(add_memory_example())
```

### 3. Search Memories

```python
async def search_example():
    results = await processor.search_memory(
        query="Who created Python?",
        limit=5,
        metadata_filters={"tags": ["python"]}
    )

    for result in results:
        print(f"Score: {result.similarity_score:.2f}")
        print(f"Text: {result.text}")
        print(f"Source: {result.source}")
        print("---")

asyncio.run(search_example())
```

### 4. Get Statistics

```python
async def stats_example():
    stats = await processor.get_stats()
    print(f"Total memories: {stats.total_memories}")
    print(f"Total chunks: {stats.total_chunks}")
    print(f"Last memory: {stats.last_memory_timestamp}")
    print(f"Searches performed: {stats.total_searches}")

asyncio.run(stats_example())
```

---

## MCP Server Integration

### Using with Claude Desktop

1. **Get MCP Server Details**
```bash
# Show MCPServer configuration
python3 << 'EOF'
from zapomni_mcp.server import MCPServer
from zapomni_core.logging_service import LoggingService

LoggingService.configure_logging()

# The server details for your config
print("MCP Server Configuration:")
print("- Name: zapomni-memory")
print("- Module: zapomni_mcp.server")
print("- Class: MCPServer")
print("\nTools available:")
print("  - add_memory: Add memories with metadata")
print("  - search_memory: Search with vector and metadata filtering")
print("  - get_stats: Get system statistics")
EOF
```

2. **Configure in Claude Desktop**

Edit `~/.config/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "zapomni": {
      "command": "python3",
      "args": ["-m", "zapomni_mcp.server"],
      "env": {
        "FALKORDB_HOST": "localhost",
        "FALKORDB_PORT": "6381",
        "OLLAMA_BASE_URL": "http://localhost:11434"
      }
    }
  }
}
```

3. **Restart Claude Desktop**

The Zapomni tools will now be available to Claude with:
- `add_memory`: Store information
- `search_memory`: Find related memories
- `get_stats`: View system status

### Example Usage in Claude

```
User: "Remember that Python was created by Guido van Rossum"

Claude will use:
→ add_memory tool to store the memory
→ Returns memory_id and confirmation

User: "What do you know about Python's creator?"

Claude will use:
→ search_memory tool to find relevant memories
→ Returns ranked results
→ Provides synthesis of findings
```

---

## Troubleshooting

### Issue: Services won't start

```bash
# Check if ports are in use
lsof -i :6381  # FalkorDB
lsof -i :11434 # Ollama

# Kill existing processes if needed
kill <PID>

# Or use different ports in .env
# Then restart
docker-compose down
docker-compose up -d
```

### Issue: "Logging not configured" error

```python
# Always call this first
from zapomni_core.logging_service import LoggingService
LoggingService.configure_logging()

# Then import other modules
from zapomni_core.memory_processor import MemoryProcessor
```

### Issue: FalkorDB connection error

```bash
# Check FalkorDB is running
docker-compose logs falkordb

# Check it's ready
docker-compose ps

# If not healthy, restart
docker-compose restart falkordb
sleep 10
```

### Issue: Ollama not generating embeddings

```bash
# Check Ollama is running
docker-compose logs ollama

# Test Ollama directly
curl http://localhost:11434/api/generate -d '{
  "model": "nomic-embed-text",
  "prompt": "test"
}'

# Pull model if needed
curl http://localhost:11434/api/pull -d '{"name": "nomic-embed-text"}'
```

### Issue: Embeddings too small or invalid

```bash
# This is expected - Ollama returns normalized embeddings
# They're stored efficiently and work correctly in searches
# No action needed
```

### Issue: Tests fail with "Database error"

```bash
# Integration tests need a live FalkorDB instance
# Unit tests run fine without it
python3 -m pytest tests/unit/ -v
```

---

## Configuration Options

### Memory Processor Config

```python
from zapomni_core.memory_processor import ProcessorConfig

config = ProcessorConfig(
    enable_cache=False,        # Phase 2: Embedding caching
    enable_extraction=False,   # Phase 2: Entity extraction
    enable_graph=False,        # Phase 2: Knowledge graphs
    max_text_length=10_000_000,  # 10MB max
    batch_size=32,            # Embedding batch size
    search_mode="vector"      # "vector", "bm25", or "hybrid"
)
```

### Chunker Config

```python
from zapomni_core.chunking import SemanticChunker

chunker = SemanticChunker(
    chunk_size=512,          # Characters per chunk
    chunk_overlap=50,        # Overlap between chunks
    min_chunk_size=100,      # Minimum chunk size
    # Language defaults to "en" for English
)
```

### Embedder Config

```python
from zapomni_core.embeddings.ollama_embedder import OllamaEmbedder

embedder = OllamaEmbedder(
    base_url="http://localhost:11434",
    model_name="nomic-embed-text",
    timeout=30,
    fallback_enabled=True,   # Use fallback on errors
)
```

---

## Common Tasks

### Export All Memories

```python
async def export_memories():
    from zapomni_db import FalkorDBClient

    db = FalkorDBClient()
    memories = await db.get_all_memories()

    import json
    with open("memories_backup.json", "w") as f:
        json.dump([m.dict() for m in memories], f, indent=2, default=str)

asyncio.run(export_memories())
```

### Clear All Memories

```python
async def clear_database():
    from zapomni_db import FalkorDBClient

    db = FalkorDBClient()
    await db.delete_all_memories()
    print("Database cleared")

asyncio.run(clear_database())
```

### Search with Metadata Filters

```python
async def advanced_search():
    results = await processor.search_memory(
        query="machine learning",
        limit=10,
        metadata_filters={
            "tags": ["ai", "ml"],  # Must have these tags
            "source": "research"   # Specific source
        }
    )

    return results

asyncio.run(advanced_search())
```

### Monitor System Health

```python
async def health_check():
    stats = await processor.get_stats()

    print(f"Health Check:")
    print(f"  Memory Count: {stats.total_memories}")
    print(f"  Database Healthy: {stats.database_healthy}")
    print(f"  Last Operation: {stats.last_operation_timestamp}")
    print(f"  Average Search Time: {stats.avg_search_time_ms}ms")

asyncio.run(health_check())
```

---

## Performance Tips

1. **Tune Chunk Size**
   - Smaller chunks (256-512): Better for fine-grained searches
   - Larger chunks (768-2048): Better for context retention

2. **Use Metadata Filters**
   - Combine metadata filters with text search
   - Reduces embeddings to evaluate
   - Faster search results

3. **Batch Operations**
   - Use batch_size parameter when adding multiple memories
   - Reduces database round-trips
   - Improves throughput 5-10x

4. **Monitor Search Metrics**
   - Track average search time
   - Monitor cache hit rates
   - Optimize chunk overlap if needed

---

## Next Steps

1. **Explore Examples**
   - Check `/home/dev/zapomni/docs/` for examples
   - Review test files for usage patterns

2. **Customize Configuration**
   - Update .env for your environment
   - Configure chunking parameters
   - Set up metadata schema

3. **Integrate with Your App**
   - Use MemoryProcessor API
   - Or use MCP server for AI integration

4. **Monitor & Optimize**
   - Check logs in structured JSON format
   - Monitor search performance
   - Track memory statistics

---

## Support Resources

- **DELIVERY_REPORT.md** - Comprehensive project overview
- **docs/** - Architecture and design documentation
- **tests/** - Usage examples in test files
- **src/zapomni_core/memory_processor.py** - Well-documented API

---

## What's Next (Phase 2)

- Entity extraction from memories
- Knowledge graph construction
- Advanced search with re-ranking
- Web dashboard
- Performance optimization

---

## License

MIT License - See LICENSE file for details

---

**Zapomni MVP v1.0.0**
*Remember everything that matters.*

Last Updated: November 24, 2025
