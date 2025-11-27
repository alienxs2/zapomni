# Contributing to Zapomni

Thank you for your interest in contributing to Zapomni! We welcome contributions of all kinds - from bug reports and documentation improvements to feature implementations and code refactoring.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Development Setup](#development-setup)
- [Branching Strategy](#branching-strategy)
- [Commit Message Conventions](#commit-message-conventions)
- [Pull Request Process](#pull-request-process)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing Requirements](#testing-requirements)
- [Review Process](#review-process)
- [Developer Certificate of Origin](#developer-certificate-of-origin)

## Code of Conduct

This project adheres to the Contributor Covenant Code of Conduct. By participating, you are expected to uphold this code. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for details.

## Development Setup

For detailed setup instructions, see [DEVELOPMENT.md](docs/DEVELOPMENT.md).

### Prerequisites

- **Python 3.10+**
- **Git** and **GitHub** account
- **Docker Desktop** (for running FalkorDB and Redis locally)
- **Ollama** (for local embeddings and LLM inference)

### Clone and Setup

```bash
# Fork the repository on GitHub

# Clone your fork
git clone https://github.com/YOUR_USERNAME/zapomni.git
cd zapomni

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# OR
.venv\Scripts\activate  # Windows

# Install dependencies (including dev tools)
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Start Services

```bash
# Copy environment template
cp .env.example .env

# Start FalkorDB and Redis containers
docker-compose up -d

# Verify services are running
docker-compose ps
```

### Verify Setup

```bash
# Run tests to verify installation
pytest tests/unit -v

# Run code quality checks
pre-commit run --all-files
```

### Understanding the Codebase

Before diving into development, familiarize yourself with the project structure:

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System design, data flow, and component relationships
- **[API.md](docs/API.md)** - Complete MCP tools reference and API documentation
- **[CONFIGURATION.md](docs/CONFIGURATION.md)** - Configuration options and environment variables
- **[CLI.md](docs/CLI.md)** - Command-line interface reference

## Branching Strategy

We use a simple branching strategy for Zapomni:

### Main Branches

- **`main`** - Production-ready code. Protected branch, requires PR reviews.
- **`develop`** - Development branch for integrating features. Staging area for the next release.

### Feature Branches

Create feature branches from `develop` with descriptive names:

```bash
# For new features
git checkout -b feature/add-graph-query-tool
git checkout -b feature/improve-search-performance

# For bug fixes
git checkout -b bugfix/fix-embedding-cache-issue
git checkout -b bugfix/handle-malformed-json

# For documentation
git checkout -b docs/add-api-reference
git checkout -b docs/update-quickstart
```

### Branch Naming Convention

- `feature/*` - New features or functionality
- `bugfix/*` - Bug fixes
- `docs/*` - Documentation improvements
- `refactor/*` - Code refactoring without behavior changes
- `test/*` - Test additions or improvements
- `perf/*` - Performance improvements
- `ci/*` - CI/CD pipeline changes

## Commit Message Conventions

We follow the **Conventional Commits** specification. This provides clear, semantic commit history and enables automatic changelog generation.

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Type

Must be one of:

- **feat:** A new feature
- **fix:** A bug fix
- **docs:** Documentation only changes
- **style:** Changes that don't affect code meaning (formatting, missing semicolons, etc.)
- **refactor:** Code change that neither fixes a bug nor adds a feature
- **perf:** Code change that improves performance
- **test:** Adding missing tests or correcting existing tests
- **chore:** Changes to build process, dependencies, or tooling
- **ci:** Changes to CI/CD configuration

### Scope

Optional. The scope specifies what part of the codebase is affected:

- `core` - Core business logic
- `embeddings` - Embedding generation
- `search` - Search algorithms
- `db` - Database layer (FalkorDB)
- `cache` - Redis caching
- `mcp` - MCP server implementation
- `tests` - Test infrastructure
- `docs` - Documentation

### Subject

- Use imperative, present tense: "add feature" not "added feature"
- Don't capitalize first letter
- No period (.) at the end
- Limit to 50 characters

### Body

Optional. Provide additional context:

- Explain **what** and **why**, not how
- Wrap at 72 characters
- Separate from subject with blank line

### Footer

Optional. Reference related issues:

```
Fixes #123
Closes #456
Refs #789
```

### Examples

```
feat(search): add BM25 keyword search support

Implement hybrid search combining vector similarity with BM25
keyword search using rank-bm25 library. Results are combined
using reciprocal rank fusion (RRF).

Fixes #42
Closes #15
```

```
fix(cache): clear cache on memory update

Previously, updating a memory didn't invalidate related cache
entries, causing stale results in subsequent searches.

Refs #89
```

```
docs: add entity extraction guide

Document the entity extraction pipeline including SpaCy NER
and LLM-based extraction strategies.
```

## Pull Request Process

### Before Creating a PR

1. **Create an Issue First** - Please open a GitHub issue describing the problem or feature
2. **Discuss Your Approach** - Get feedback on your proposed solution before investing time
3. **Work on Your Feature** - Implement your changes on a feature branch

### Creating a PR

1. **Update from `develop`**
   ```bash
   git fetch origin
   git rebase origin/develop
   ```

2. **Push Your Branch**
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create Pull Request**
   - Use GitHub web interface or `gh pr create`
   - Link related issues: "Fixes #123" or "Closes #456"
   - Provide clear description of changes
   - Ensure all CI checks pass

4. **PR Description Template**
   ```markdown
   ## Description
   Brief description of changes

   ## Related Issue(s)
   Fixes #123
   Closes #456

   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Breaking change
   - [ ] Documentation update

   ## Testing
   - [ ] Unit tests added/updated
   - [ ] Integration tests added/updated
   - [ ] Manual testing completed

   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Documentation updated
   - [ ] Tests pass locally
   - [ ] No new warnings generated
   - [ ] Coverage maintained/improved
   ```

### PR Requirements

- **Tests Required** - Must include tests for new functionality
- **Coverage** - Maintain or improve code coverage (80%+ target)
- **CI Checks** - All GitHub Actions must pass
- **Reviews** - At least one maintainer approval required
- **Updated Docs** - Update relevant documentation and docstrings
- **No Conflicts** - Resolve any merge conflicts with target branch

## Code Style Guidelines

Zapomni enforces consistent code style through automated tools. All code must comply with these standards before merging.

### Tools Used

All enforced by `.pre-commit-config.yaml` and CI pipeline:

1. **Black** - Code formatting
2. **isort** - Import sorting
3. **Flake8** - Linting
4. **MyPy** - Type checking (strict mode)

### Running Code Quality Checks

```bash
# Format code automatically
black src/ tests/
isort src/ tests/

# Check types
mypy src/

# Lint code
flake8 src/ tests/

# Run all checks (what CI runs)
pre-commit run --all-files
```

### Code Style Rules

#### Formatting

- **Line Length:** 100 characters (Black)
- **Indentation:** 4 spaces
- **String Quotes:** Double quotes for code
- **Imports:** Alphabetical, grouped (isort with Black profile)

```python
# Good
from typing import Optional

import numpy as np
from pydantic import BaseModel

from zapomni_core.embeddings import EmbeddingGenerator
from zapomni_db.falkordb import FalkorDBClient


class MemoryStore(BaseModel):
    """Manages memory persistence."""

    name: str
    vector_dim: int = 768

    def add_memory(self, content: str) -> None:
        """Add memory with automatic chunking."""
        pass
```

#### Type Hints

All functions must have complete type hints (MyPy strict mode):

```python
# Good
def search_memory(
    query: str,
    top_k: int = 5,
    min_score: float = 0.5,
) -> list[dict[str, Any]]:
    """Search memories by semantic similarity.

    Args:
        query: Search query text
        top_k: Number of results to return
        min_score: Minimum similarity score threshold

    Returns:
        List of matching memories with scores
    """
    pass


# Bad - missing return type
def search_memory(query: str, top_k: int = 5):
    pass
```

#### Docstrings

Use Google-style docstrings:

```python
def add_memory(
    self,
    content: str,
    metadata: Optional[dict[str, Any]] = None,
) -> str:
    """Store new memory with automatic chunking and embedding.

    This method chunks the content, generates embeddings using Ollama,
    and stores both in FalkorDB with optional metadata.

    Args:
        content: Text content to store
        metadata: Optional metadata dict (e.g., source, timestamp)

    Returns:
        Memory ID (UUID string) for reference

    Raises:
        ValueError: If content is empty
        ConnectionError: If database connection fails

    Example:
        >>> memory_id = store.add_memory("Python created in 1991")
        >>> print(memory_id)
        550e8400-e29b-41d4-a716-446655440000
    """
    if not content.strip():
        raise ValueError("Content cannot be empty")
    # implementation
    return memory_id
```

#### Naming Conventions

```python
# Classes: PascalCase
class MemoryStore:
    pass

# Functions/methods: snake_case
def add_memory():
    pass

# Constants: UPPER_SNAKE_CASE
MAX_CHUNK_SIZE = 512
EMBEDDING_DIMENSION = 768

# Private: leading underscore
def _internal_helper():
    pass

# Protected: single leading underscore (convention, not enforcement)
def _process_chunk():
    pass
```

## Testing Requirements

All contributions must include appropriate tests with **80%+ code coverage**. Zapomni has a comprehensive test suite with 2019+ tests covering unit, integration, and end-to-end scenarios.

### Test Structure

```
tests/
├── unit/              # 70% of tests (fast, isolated)
├── integration/       # 25% of tests (with services)
└── e2e/              # 5% of tests (full workflows)
```

### Running Tests

For comprehensive testing guidelines, see [DEVELOPMENT.md](docs/DEVELOPMENT.md).

```bash
# All tests
pytest

# Specific test file
pytest tests/unit/test_embeddings.py

# Specific test function
pytest tests/unit/test_embeddings.py::test_embedding_generation

# With coverage
pytest --cov=src --cov-report=html
open htmlcov/index.html

# Fast unit tests only
pytest tests/unit -v
```

### Writing Tests

Use `pytest` with these conventions:

```python
# tests/unit/test_embeddings.py
from unittest.mock import Mock, patch

import pytest

from zapomni_core.embeddings import EmbeddingGenerator


class TestEmbeddingGenerator:
    """Tests for EmbeddingGenerator class."""

    @pytest.fixture
    def generator(self):
        """Create EmbeddingGenerator instance."""
        return EmbeddingGenerator(model="nomic-embed-text")

    def test_embedding_dimension(self, generator):
        """Verify embeddings have correct dimension."""
        embedding = generator.embed("test text")
        assert len(embedding) == 768

    def test_embedding_not_zero(self, generator):
        """Verify embeddings are not all zeros."""
        embedding = generator.embed("test text")
        assert not all(v == 0 for v in embedding)

    def test_embedding_consistency(self, generator):
        """Verify same text produces same embedding."""
        text = "consistent text"
        emb1 = generator.embed(text)
        emb2 = generator.embed(text)
        assert emb1 == emb2

    @patch("zapomni_core.embeddings.ollama.embeddings")
    def test_ollama_called_correctly(self, mock_ollama, generator):
        """Verify Ollama is called with correct parameters."""
        mock_ollama.return_value = [0.1] * 768

        generator.embed("test")

        mock_ollama.assert_called_once_with(
            model="nomic-embed-text",
            prompt="test",
        )
```

### Test Naming

- **Files:** `test_<module>.py` or `<module>_test.py`
- **Classes:** `Test<Class>` or `<Class>Tests`
- **Functions:** `test_<functionality>`

### Fixtures and Mocks

```python
# Use pytest fixtures for reusable setup
@pytest.fixture
def mock_db():
    """Mock FalkorDB client."""
    return Mock(spec=FalkorDBClient)

@pytest.fixture
def memory_store(mock_db):
    """Create MemoryStore with mocked database."""
    return MemoryStore(db=mock_db)

# Use Mock for external dependencies
@patch("zapomni_db.falkordb.FalkorDBClient")
def test_with_mock(mock_falkordb_class):
    """Test with mocked external dependency."""
    mock_instance = Mock()
    mock_falkordb_class.return_value = mock_instance
    # test code
```

### Coverage Requirements

```bash
# View coverage report
pytest --cov=src --cov-report=html
open htmlcov/index.html

# Check specific modules
pytest --cov=src/zapomni_core --cov-report=term-missing
```

## Review Process

### Code Review Standards

1. **Functionality** - Does it work correctly? Do tests pass?
2. **Design** - Is the architecture sound? Does it fit the codebase?
3. **Code Quality** - Does it follow style guidelines? Is it readable?
4. **Testing** - Are tests comprehensive? Coverage adequate?
5. **Documentation** - Are changes documented? Are docstrings clear?
6. **Performance** - Are there performance implications?

### What Reviewers Look For

- [ ] Tests are added/updated
- [ ] Coverage is maintained (80%+)
- [ ] Code follows style guidelines
- [ ] Type hints are complete
- [ ] Docstrings are clear and complete
- [ ] No performance regressions
- [ ] Documentation is updated
- [ ] Commit messages are clear

### Responding to Reviews

1. **Questions** - Ask for clarification if you don't understand feedback
2. **Disagreements** - Discuss respectfully, provide reasoning
3. **Changes** - Make requested changes and push to same branch
4. **Follow-ups** - Comment when you've addressed feedback

## Developer Certificate of Origin

By contributing to Zapomni, you certify that:

1. The contribution was created in whole or in part by you
2. You have the right to submit the contribution under the MIT license
3. The contribution does not violate any known intellectual property claims

We do not require a Contributor License Agreement (CLA). Instead, we use the Developer Certificate of Origin (DCO) - a lightweight alternative. By committing to this repository, you assert that your contributions are original and that you have the rights to contribute them.

No additional action is required - your signed commits indicate acceptance of the DCO.

## Getting Help

- **Questions?** Open a [GitHub Discussion](https://github.com/alienxs2/zapomni/discussions)
- **Found a bug?** [Create an Issue](https://github.com/alienxs2/zapomni/issues)
- **Need guidance?** Comment on a related issue or discussion

## Recognition

Contributors are recognized in:

1. **CHANGELOG.md** - Listed in release notes for features/fixes
2. **Repository** - Git history preserves your contribution
3. **GitHub** - Your profile shows contribution activity

Thank you for contributing to Zapomni!
