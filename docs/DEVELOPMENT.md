# Zapomni Development Guide

## Table of Contents

- [Overview](#overview)
- [Development Setup](#development-setup)
- [Testing](#testing)
- [Code Quality Tools](#code-quality-tools)
- [Adding New MCP Tools](#adding-new-mcp-tools)
- [CI/CD](#cicd)
- [Project Structure](#project-structure)
- [Contributing Guidelines](#contributing-guidelines)

## Overview

This guide covers development workflow, testing, and contribution guidelines for Zapomni.

**Tech Stack**:
- **Language**: Python 3.10+
- **Database**: FalkorDB (Redis module)
- **Cache**: Redis
- **Embeddings**: Ollama (local LLM)
- **Protocol**: MCP (Model Context Protocol)
- **Testing**: pytest, pytest-asyncio, pytest-cov
- **Type Checking**: mypy (strict mode)
- **Linting**: ruff
- **Formatting**: black

---

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Docker and Docker Compose
- Git
- Ollama (for embeddings)

### 1. Clone Repository

```bash
git clone https://github.com/your-org/zapomni.git
cd zapomni
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Using conda
conda create -n zapomni python=3.10
conda activate zapomni
```

### 3. Install Dependencies

```bash
# Install in development mode
pip install -e .

# Or using pip directly
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-asyncio pytest-cov pytest-mock
pip install mypy ruff black
```

### 4. Start Services

```bash
# Start FalkorDB and Redis
docker-compose up -d

# Verify services running
docker ps
# Should show: falkordb (6381), redis (6380)
```

### 5. Configure Environment

```bash
# Copy example configuration
cp .env.example .env

# Edit configuration
nano .env
```

**Minimal development .env**:
```bash
FALKORDB_HOST=localhost
FALKORDB_PORT=6381
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_LLM_MODEL=llama3.1:8b

# Disable advanced features for faster development
ENABLE_HYBRID_SEARCH=false
ENABLE_KNOWLEDGE_GRAPH=false
ENABLE_CODE_INDEXING=false
ENABLE_SEMANTIC_CACHE=false

# Verbose logging
LOG_LEVEL=DEBUG
LOG_FORMAT=console
```

### 6. Install Ollama Models

```bash
# Install embedding model
ollama pull nomic-embed-text

# Install LLM model (optional, for entity refinement)
ollama pull llama3.1:8b
```

### 7. Initialize Database

```bash
# Run schema initialization (automatic on first start)
python -m zapomni_mcp

# Or manually
python -c "
from zapomni_db import FalkorDBClient
from zapomni_db.schema_manager import SchemaManager
import asyncio

async def init():
    client = FalkorDBClient()
    schema = SchemaManager(client)
    await schema.initialize_schema()
    
asyncio.run(init())
"
```

### 8. Verify Setup

```bash
# Run basic tests
pytest tests/unit/test_config.py -v

# Check MCP server starts
python -m zapomni_mcp
# Should output: MCP server started successfully
```

---

## Testing

Zapomni has a comprehensive test suite with **1,858 tests** (6 skipped) covering unit, integration, and performance testing. Test suite runs in ~35 seconds.

### Test Structure

```
tests/
├── unit/                       # ~1,700 tests (92%)
│   ├── test_*_tool.py         # MCP tools
│   ├── test_*_processor.py    # Core processors
│   ├── test_*_search.py       # Search algorithms
│   ├── test_falkordb*.py      # Database layer
│   └── ...
├── integration/               # ~155 tests (8%)
│   ├── test_*_integration.py
│   ├── test_workspace_isolation.py
│   └── test_garbage_collector_integration.py
└── e2e/                       # 0 tests (planned)
```

### Running Tests

**Run all tests**:
```bash
pytest
# ~1858 tests, takes ~35 seconds
```

**Run specific test category**:
```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Specific module
pytest tests/unit/test_add_memory_tool.py

# Specific test function
pytest tests/unit/test_add_memory_tool.py::test_add_memory_success
```

**Run with coverage**:
```bash
# Full coverage report
pytest --cov=src --cov-report=html

# Coverage for specific module
pytest --cov=src/zapomni_mcp/tools --cov-report=term

# Open HTML report
open htmlcov/index.html
```

**Run in parallel** (faster):
```bash
# Install pytest-xdist
pip install pytest-xdist

# Run tests in parallel
pytest -n auto
# Uses all CPU cores
```

**Verbose output**:
```bash
# Show full output
pytest -v

# Show print statements
pytest -s

# Both
pytest -vs
```

### Test Coverage by Layer

| Layer | Coverage | Status |
|-------|----------|--------|
| MCP Tools | 74% | Good (some tools at 27-48%) |
| Core Processors | 89% | Excellent |
| Search | 86% | Excellent |
| Chunking | 88% | Excellent |
| Embeddings | 89% | Excellent |
| Code Analysis | 72% | Good |
| DB Layer | 71% | Good |

### Writing Tests

**Unit Test Example**:
```python
import pytest
from zapomni_mcp.tools.add_memory import AddMemoryTool
from zapomni_core.memory_processor import MemoryProcessor

@pytest.mark.asyncio
async def test_add_memory_success(mock_memory_processor):
    """Test successful memory addition."""
    tool = AddMemoryTool(mock_memory_processor)
    
    result = await tool.execute({
        "text": "Test memory content",
        "metadata": {"source": "test"}
    })
    
    assert result["status"] == "success"
    assert "memory_id" in result
    assert result["chunks_created"] > 0
```

**Integration Test Example**:
```python
import pytest
from zapomni_db import FalkorDBClient
from zapomni_core.memory_processor import MemoryProcessor

@pytest.mark.asyncio
async def test_add_and_search_integration():
    """Test full add + search workflow."""
    db_client = FalkorDBClient()
    processor = MemoryProcessor(db_client)
    
    # Add memory
    memory_id = await processor.add_memory(
        text="Python is a programming language",
        metadata={"source": "test"}
    )
    
    # Search for it
    results = await processor.search_memory(
        query="Python programming",
        limit=5
    )
    
    assert len(results) > 0
    assert results[0]["memory_id"] == memory_id
```

### Test Fixtures

Common fixtures in `tests/conftest.py`:

```python
@pytest.fixture
def mock_db_client():
    """Mock FalkorDB client."""
    return Mock(spec=FalkorDBClient)

@pytest.fixture
def mock_embedder():
    """Mock Ollama embedder."""
    embedder = Mock()
    embedder.embed_text.return_value = [0.1] * 768
    return embedder

@pytest.fixture
async def test_workspace(db_client):
    """Create test workspace."""
    workspace_id = "test-workspace"
    await db_client.create_workspace(workspace_id, "Test Workspace")
    yield workspace_id
    await db_client.delete_workspace(workspace_id, confirm=True)
```

---

## Code Quality Tools

### Linting (Ruff)

```bash
# Check code style
ruff check src/ tests/

# Auto-fix issues
ruff check --fix src/ tests/

# Configuration in pyproject.toml
[tool.ruff]
line-length = 100
select = ["E", "F", "I", "N", "W"]
ignore = ["E501"]  # Line too long (handled by black)
```

### Formatting (Black)

```bash
# Format code
black src/ tests/

# Check without modifying
black --check src/ tests/

# Configuration in pyproject.toml
[tool.black]
line-length = 100
target-version = ['py310']
```

### Type Checking (mypy)

```bash
# Type check
mypy src/

# Strict mode
mypy --strict src/

# Configuration in pyproject.toml
[tool.mypy]
python_version = "3.10"
strict = true
warn_unused_ignores = true
warn_redundant_casts = true
```

### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

**`.pre-commit-config.yaml`**:
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.9.1
    hooks:
      - id: black
  
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.0
    hooks:
      - id: ruff
        args: [--fix]
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.1
    hooks:
      - id: mypy
        additional_dependencies: [pydantic, structlog]
```

---

## Adding New MCP Tools

### Step-by-Step Guide

#### 1. Create Tool File

Create `src/zapomni_mcp/tools/your_tool.py`:

```python
"""
YourTool MCP Tool - Implementation.

Description of what your tool does.

Author: Your Name
License: MIT
"""

from typing import Any, Dict
import structlog
from pydantic import BaseModel, ConfigDict

from zapomni_core.memory_processor import MemoryProcessor
from zapomni_core.exceptions import ValidationError

logger = structlog.get_logger(__name__)


class YourToolRequest(BaseModel):
    """Pydantic model for validating input."""
    
    model_config = ConfigDict(extra="forbid")
    
    param1: str
    param2: int = 10  # Optional with default


class YourTool:
    """
    MCP tool for doing something useful.
    
    Attributes:
        name: Tool identifier ("your_tool")
        description: Human-readable description
        input_schema: JSON Schema for input validation
        memory_processor: MemoryProcessor instance
        logger: Structured logger
    """
    
    name = "your_tool"
    description = "Does something useful with data."
    input_schema = {
        "type": "object",
        "properties": {
            "param1": {
                "type": "string",
                "description": "First parameter"
            },
            "param2": {
                "type": "integer",
                "description": "Second parameter",
                "default": 10
            }
        },
        "required": ["param1"]
    }
    
    def __init__(self, memory_processor: MemoryProcessor):
        """Initialize tool with memory processor."""
        self.memory_processor = memory_processor
        self.logger = logger.bind(tool=self.name)
    
    async def execute(self, arguments: Dict[str, Any]) -> list[Dict[str, Any]]:
        """
        Execute the tool.
        
        Args:
            arguments: Tool input arguments
            
        Returns:
            MCP-formatted response: list[Dict[str, Any]]
            
        Raises:
            ValidationError: Invalid input
        """
        try:
            # 1. Validate input
            request = YourToolRequest(**arguments)
            
            self.logger.info("executing", param1=request.param1)
            
            # 2. Delegate to Core layer
            result = await self.memory_processor.your_method(
                param1=request.param1,
                param2=request.param2
            )
            
            # 3. Format response
            return [{
                "type": "text",
                "text": f"Success: {result}"
            }]
            
        except Exception as e:
            self.logger.error("execution_failed", error=str(e))
            return [{
                "type": "text",
                "text": f"Error: {str(e)}",
                "isError": True
            }]
```

#### 2. Register Tool

Add to `src/zapomni_mcp/tools/__init__.py`:

```python
from .your_tool import YourTool

__all__ = [
    # Existing tools...
    "YourTool",
]
```

#### 3. Add to MCP Server

Update `src/zapomni_mcp/server.py`:

```python
from zapomni_mcp.tools import YourTool

# In MCPServer.__init__():
self.tools = [
    # Existing tools...
    YourTool(memory_processor),
]
```

#### 4. Write Tests

Create `tests/unit/test_your_tool.py`:

```python
import pytest
from zapomni_mcp.tools.your_tool import YourTool
from unittest.mock import Mock

@pytest.fixture
def mock_memory_processor():
    processor = Mock()
    processor.your_method.return_value = "result"
    return processor

@pytest.fixture
def tool(mock_memory_processor):
    return YourTool(mock_memory_processor)

@pytest.mark.asyncio
async def test_your_tool_success(tool):
    """Test successful execution."""
    result = await tool.execute({
        "param1": "test",
        "param2": 20
    })
    
    assert len(result) == 1
    assert result[0]["type"] == "text"
    assert "Success" in result[0]["text"]

@pytest.mark.asyncio
async def test_your_tool_validation_error(tool):
    """Test validation error."""
    result = await tool.execute({})  # Missing required param1
    
    assert result[0]["isError"] is True
```

#### 5. Run Tests

```bash
pytest tests/unit/test_your_tool.py -v
```

#### 6. Update Documentation

Add tool to `docs/API.md`:

```markdown
### your_tool

Does something useful with data.

**Input Schema**: ...
**Output**: ...
**Example**: ...
```

---

## CI/CD

### GitHub Actions Workflow

**`.github/workflows/ci.yml`**:

```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      falkordb:
        image: falkordb/falkordb:latest
        ports:
          - 6381:6379
      
      redis:
        image: redis:7-alpine
        ports:
          - 6380:6379
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e .
          pip install pytest pytest-cov
      
      - name: Run tests
        run: |
          pytest --cov=src --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
  
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install tools
        run: |
          pip install ruff black mypy
      
      - name: Ruff
        run: ruff check src/ tests/
      
      - name: Black
        run: black --check src/ tests/
      
      - name: mypy
        run: mypy src/
```

### Docker Build

```yaml
name: Docker

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build image
        run: docker build -t zapomni:${{ github.ref_name }} .
      
      - name: Push to registry
        run: docker push zapomni:${{ github.ref_name }}
```

---

## Project Structure

```
zapomni/
├── src/
│   ├── zapomni_mcp/         # MCP layer
│   │   ├── server.py
│   │   ├── tools/
│   │   │   ├── __init__.py
│   │   │   ├── add_memory.py
│   │   │   └── ...
│   │   └── ...
│   ├── zapomni_core/        # Core business logic
│   │   ├── memory_processor.py
│   │   ├── processors/
│   │   ├── search/
│   │   ├── embeddings/
│   │   └── ...
│   ├── zapomni_db/          # Database layer
│   │   ├── falkordb_client.py
│   │   ├── models.py
│   │   └── ...
│   └── zapomni_cli/         # CLI utilities
│       ├── __main__.py
│       └── install_hooks.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── conftest.py
├── docs/
│   ├── ARCHITECTURE.md
│   ├── API.md
│   ├── CONFIGURATION.md
│   ├── CLI.md
│   └── DEVELOPMENT.md
├── docker-compose.yml
├── .env.example
├── pyproject.toml
└── README.md
```

---

## Contributing Guidelines

### Workflow

1. **Fork and clone** repository
2. **Create feature branch**: `git checkout -b feature/your-feature`
3. **Make changes** with tests
4. **Run tests**: `pytest`
5. **Run linters**: `ruff check && black . && mypy src/`
6. **Commit**: `git commit -am "Add feature"`
7. **Push**: `git push origin feature/your-feature`
8. **Open Pull Request**

### Commit Messages

Follow conventional commits:

```
feat: Add new search algorithm
fix: Resolve connection pool leak
docs: Update API documentation
test: Add integration tests for workspace isolation
refactor: Simplify entity extraction logic
perf: Optimize HNSW index build
```

### Pull Request Checklist

- [ ] Tests pass (`pytest`)
- [ ] Linters pass (`ruff check`, `black --check`, `mypy`)
- [ ] Coverage maintained or improved
- [ ] Documentation updated
- [ ] Changelog updated (if applicable)
- [ ] No breaking changes (or clearly documented)

### Code Review Process

1. Automated checks run on PR
2. Maintainer reviews code
3. Address feedback
4. Approved PR merged to main

---

## Performance Profiling

### Using pytest-benchmark

```bash
# Install
pip install pytest-benchmark

# Run benchmarks
pytest tests/benchmarks/ --benchmark-only
```

**Example benchmark**:
```python
def test_embedding_performance(benchmark):
    """Benchmark embedding generation."""
    embedder = OllamaEmbedder()
    
    result = benchmark(
        embedder.embed_text,
        "Sample text for embedding"
    )
    
    assert len(result) == 768
```

### Using cProfile

```bash
# Profile MCP server startup
python -m cProfile -o profile.stats -m zapomni_mcp

# Analyze with snakeviz
pip install snakeviz
snakeviz profile.stats
```

---

## Debugging

### Debug Logging

```bash
# Enable debug logs
export LOG_LEVEL=DEBUG

# Run with verbose output
python -m zapomni_mcp
```

### pdb Debugging

```python
# Add breakpoint
import pdb; pdb.set_trace()

# Or using Python 3.7+
breakpoint()
```

### Async Debugging

```python
import asyncio

# Enable debug mode
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
loop = asyncio.new_event_loop()
loop.set_debug(True)
```

---

## Related Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)**: System architecture
- **[API.md](API.md)**: MCP tools reference
- **[CONFIGURATION.md](CONFIGURATION.md)**: Configuration options
- **[CLI.md](CLI.md)**: Command-line tools

---

## Resources

### Official Documentation
- [MCP Protocol Specification](https://modelcontextprotocol.io/)
- [FalkorDB Documentation](https://docs.falkordb.com/)
- [Ollama Documentation](https://ollama.ai/docs)

### Community
- GitHub Issues: Report bugs and request features
- Discussions: Ask questions and share ideas

---

**Document Version**: 1.1
**Last Updated**: 2025-11-27
**Contributors**: Goncharenko Anton (alienxs2)
