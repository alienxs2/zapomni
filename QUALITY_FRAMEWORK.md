# Quality Assurance Framework

**Project:** Zapomni
**Author:** Goncharenko Anton aka alienxs2
**License:** MIT
**Date:** 2025-11-23

Related: [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) | [SPEC_METHODOLOGY.md](SPEC_METHODOLOGY.md)

---

## 📋 Overview

Quality Framework определяет строгие стандарты качества для Zapomni:
- 6 Quality Gates (checkpoints)
- Test-Driven Development (90%+ coverage)
- Automated quality checks (pre-commit, CI/CD)
- Code review standards
- Performance benchmarking
- Security validation

### Quality Philosophy

**"Quality is not negotiable"**

Принципы:
1. **Prevention over Detection** - найти проблемы ДО кода (в specs)
2. **Automation First** - автоматизировать все что можно
3. **Measurable Standards** - чёткие метрики, не субъективность
4. **No Shortcuts** - нельзя skip quality gates
5. **Continuous Improvement** - учимся на каждом цикле

---

## 🚦 Quality Gates (6 Major Checkpoints)

### Quality Gate 1: Module-Level Specs Approved

**Trigger:** После завершения Phase 1 (Module specs)

**Criteria:**
- ✅ All 7 module specs created
- ✅ Verification cycle completed (5 agents → synthesis → reconciliation)
- ✅ Zero CRITICAL issues in final reports
- ✅ Maximum 3 WARNING issues (non-blocking)
- ✅ 100% alignment with steering documents
- ✅ User approval obtained

**Metrics:**
- Documents consistency score: 100%
- Coverage: All features from product.md represented
- Technical feasibility: Confirmed by all verification agents

**Exit Criteria:**
- User says "Module specs approved" or "approve Phase 1"
- All criteria ✅ green

**If Failed:**
- Additional refinement cycle
- User review and decision
- Maximum 3 iteration cycles before escalation

**Deliverables:**
- 7 verified module specs in `.spec-workflow/specs/level1/`
- Verification reports in `verification_reports/level1/`
- User approval documented

---

### Quality Gate 2: Component-Level Specs Approved

**Trigger:** После завершения Phase 2 (Component specs)

**Criteria:**
- ✅ All 15-20 component specs created
- ✅ Each component traceable to parent module spec
- ✅ Public API fully defined (all methods with signatures)
- ✅ Dependency graph validated (acyclic, no circular deps)
- ✅ Verification passed (synthesis approved)
- ✅ User approval obtained

**Metrics:**
- API coverage: 100% of public methods documented
- Dependency graph: Valid (no cycles detected)
- Consistency score: 100% (cross-component)

**Exit Criteria:**
- All component APIs match module interfaces
- No blocking issues
- User approval

**If Failed:**
- Identify root cause (conflicting requirements, unclear module spec)
- Refinement or escalate to user

**Deliverables:**
- 15-20 component specs in `.spec-workflow/specs/level2/`
- Verified dependency graph diagram
- API definition index

---

### Quality Gate 3: Function-Level Specs Approved

**Trigger:** После завершения Phase 3 (Function specs)

**Criteria:**
- ✅ All 40-50 function specs created
- ✅ Every public function documented
- ✅ Edge cases enumerated (minimum 3 per function)
- ✅ Test scenarios defined (minimum 5 per function)
- ✅ Algorithm pseudocode provided for complex functions
- ✅ Verification passed
- ✅ User approval obtained

**Metrics:**
- Functions documented: 100% of public functions
- Edge cases per function: Average >= 3
- Test scenarios defined: Total >= 200
- Spec completeness score: >= 95%

**Exit Criteria:**
- Can write tests directly from specs (no ambiguity)
- Can write code directly from specs
- User approval

**If Failed:**
- Incomplete specs → identify missing functions
- Insufficient edge cases → expand analysis
- User clarification if requirements unclear

**Deliverables:**
- 40-50 function specs in `.spec-workflow/specs/level3/`
- Test scenario catalog (>= 200 scenarios)
- Implementation readiness checklist

---

### Quality Gate 4: Tests Complete (RED Phase)

**Trigger:** После завершения Phase 4 (Test Development)

**Criteria:**
- ✅ All tests written based on function specs
- ✅ All tests FAIL (RED) - code not implemented yet
- ✅ Test coverage target: 100% of functions have tests
- ✅ Test naming convention followed: `test_{function}_{scenario}_{expected}`
- ✅ Code review of tests passed
- ✅ User approval obtained

**Metrics:**
- Tests written: >= 200 test cases
- Test-to-function ratio: >= 5:1 (5 tests per function average)
- Test status: 100% FAIL (expected before code)
- Test quality score: >= 90% (review assessment)

**Exit Criteria:**
- Every function spec has corresponding tests
- Every edge case has test
- All tests fail appropriately (not broken, just no code)
- User approval

**If Failed:**
- Missing tests → write missing tests
- Tests don't match specs → align with specs
- Tests passing (shouldn't be) → investigate why

**Deliverables:**
- Complete test suite in `tests/unit/`, `tests/integration/`, `tests/e2e/`
- Test coverage report (functions covered, not line coverage yet)
- Test review report

---

### Quality Gate 5: Implementation Complete (GREEN Phase)

**Trigger:** После завершения Phases 5-7 (Implementation)

**Criteria:**
- ✅ All features merged to main
- ✅ All tests GREEN (100% passing)
- ✅ Test coverage actual: >= 90% (line coverage)
- ✅ Type coverage (mypy): 100% strict mode
- ✅ Linting errors: 0 (black, isort, pylint)
- ✅ Code review passed for all PRs
- ✅ Integration tests passing
- ✅ E2E tests passing
- ✅ Performance benchmarks met (SLAs from tech.md)
- ✅ User approval obtained

**Metrics:**
- Test coverage: >= 90% overall
  - Unit: >= 95%
  - Integration: >= 85%
  - E2E: >= 70%
- Tests passing: 100% (all green)
- Type coverage: 100% (mypy strict)
- Linting: 0 errors
- Performance: All SLAs met
  - add_memory: < 500ms
  - search_memory: < 200ms
  - Memory usage: < 2GB for 100K memories

**Exit Criteria:**
- Code quality tools all green
- All tests green
- Performance validated
- User approval after testing

**If Failed:**
- Tests failing → fix code
- Coverage < 90% → write more tests
- Performance issues → optimize
- Type/linting errors → fix

**Deliverables:**
- Production-ready code in `src/`
- All tests passing
- Coverage report >= 90%
- Performance benchmark results
- Code review approvals for all PRs

---

### Quality Gate 6: MVP Ready for Production

**Trigger:** После завершения Phases 8-9 (Integration, Documentation)

**Criteria:**
- ✅ All features integrated and tested
- ✅ Documentation complete (README, API docs, user guide)
- ✅ Installation tested on clean environment (30-min target met)
- ✅ MCP integration tested (Claude CLI works)
- ✅ Docker compose up works
- ✅ User acceptance testing passed
- ✅ CHANGELOG created
- ✅ GitHub release tagged (v0.1.0-mvp)
- ✅ User final approval → **MVP COMPLETE** 🎉

**Metrics:**
- Installation time: <= 30 minutes (tested)
- MCP tools working: 3/3 (add_memory, search_memory, get_stats)
- Documentation completeness: 100%
- User satisfaction: High (subjective)

**Exit Criteria:**
- Can install from scratch in 30 min
- Can use all 3 MCP tools successfully
- Documentation clear and complete
- User says "MVP ready for production use"

**If Failed:**
- Installation issues → fix setup scripts, improve docs
- MCP integration broken → fix integration
- Documentation unclear → improve docs
- User concerns → address and re-test

**Deliverables:**
- GitHub release v0.1.0-mvp
- Complete documentation site
- Installation guide tested
- User acceptance sign-off

---

## 🧪 Test-Driven Development (TDD)

### TDD Workflow

**RED → GREEN → REFACTOR cycle:**

```
1. RED Phase:
   - Write test FIRST (based on function spec)
   - Run test → MUST FAIL (no code yet)
   - Verify test fails for right reason

2. GREEN Phase:
   - Write MINIMAL code to make test pass
   - Run test → should turn GREEN
   - Don't refactor yet

3. REFACTOR Phase:
   - Improve code quality
   - Tests MUST stay GREEN during refactoring
   - Run tests after each change
```

### Coverage Targets

**Overall: >= 90% (hard requirement)**

Breakdown:
- **Unit tests: >= 95%**
  - Every function tested
  - Every branch tested
  - Every edge case tested

- **Integration tests: >= 85%**
  - Module interactions tested
  - Database integration tested
  - Ollama integration tested

- **E2E tests: >= 70%**
  - Full user workflows
  - MCP protocol end-to-end
  - Error scenarios

### Test Structure

**Naming Convention:**
```python
def test_{function_name}_{scenario}_{expected_result}():
    """Test {function} when {scenario} should {expected}."""
```

**Examples:**
```python
def test_add_memory_empty_text_raises_validation_error():
    """Test add_memory when text is empty should raise ValidationError."""

def test_search_memory_success_returns_results():
    """Test search_memory with valid query should return results."""

def test_get_stats_no_data_returns_zero_counts():
    """Test get_stats when database empty should return zero counts."""
```

**File Organization:**
```
tests/
├── unit/
│   ├── test_add_memory_tool.py
│   ├── test_search_service.py
│   └── test_db_client.py
├── integration/
│   ├── test_mcp_to_core.py
│   ├── test_core_to_db.py
│   └── test_full_pipeline.py
└── e2e/
    ├── test_add_and_search_workflow.py
    └── test_mcp_protocol.py
```

### Test Quality Standards

**Good Test Characteristics:**
1. **Isolated** - No shared state between tests
2. **Repeatable** - Same result every run
3. **Fast** - Unit tests < 100ms each
4. **Clear** - Test name explains what it tests
5. **Focused** - One assertion per test (when possible)

**Test Anti-Patterns (avoid):**
```python
# ❌ BAD - Vague name
def test_function():
    ...

# ✅ GOOD - Clear name
def test_add_memory_success_simple_text():
    ...

# ❌ BAD - Multiple unrelated assertions
def test_everything():
    assert add_memory(...) == ...
    assert search_memory(...) == ...
    assert get_stats() == ...

# ✅ GOOD - One logical assertion per test
def test_add_memory_returns_valid_id():
    result = add_memory("text")
    assert is_valid_uuid(result.id)
```

### Mocking Strategy

**Mock External Dependencies:**
- FalkorDB (use in-memory mock for unit tests)
- Ollama (mock API responses)
- File I/O (use temporary directories)

**Don't Mock Internal Code:**
- Test real implementation of internal modules
- Only mock at system boundaries

**Example:**
```python
from unittest.mock import Mock, patch

def test_add_memory_ollama_offline_raises(mocker):
    """Test add_memory when Ollama unavailable should raise EmbeddingError."""
    # Mock Ollama client to simulate offline
    mocker.patch('ollama.Client.embeddings', side_effect=ConnectionError)

    tool = AddMemoryTool(ollama_client=OllamaClient())

    with pytest.raises(EmbeddingError, match="Ollama unavailable"):
        tool.execute({"text": "test"})
```

---

## 🤖 Automated Quality Checks

### Pre-Commit Hooks

**Configuration:** `.pre-commit-config.yaml`

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.1.1
    hooks:
      - id: black
        args: [--line-length=100]

  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: [--profile=black, --line-length=100]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
        args: [--strict, --ignore-missing-imports]

  - repo: local
    hooks:
      - id: pytest-quick
        name: pytest unit tests
        entry: pytest
        language: system
        args: [--maxfail=1, --tb=short, tests/unit/]
        pass_filenames: false
```

**What Happens on Commit:**
1. **Black** - Auto-formats code (100 char line length)
2. **isort** - Sorts imports alphabetically
3. **mypy** - Type checks (strict mode)
4. **pytest** - Runs unit tests

**If ANY check fails → commit BLOCKED**

### CI/CD Pipeline (GitHub Actions)

**Configuration:** `.github/workflows/ci.yml`

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"

      - name: Run black check
        run: black --check src/ tests/

      - name: Run isort check
        run: isort --check-only src/ tests/

      - name: Run mypy
        run: mypy src/

      - name: Run pytest with coverage
        run: |
          pytest --cov=src --cov-report=xml --cov-report=html --cov-report=term

      - name: Check coverage threshold
        run: |
          coverage report --fail-under=90

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml

  integration:
    runs-on: ubuntu-latest
    services:
      falkordb:
        image: falkordb/falkordb:latest
        ports:
          - 6379:6379
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -e ".[dev]"
      - name: Run integration tests
        run: pytest tests/integration/

  e2e:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install Ollama
        run: curl -fsSL https://ollama.com/install.sh | sh
      - name: Pull Ollama model
        run: ollama pull nomic-embed-text
      - name: Start services
        run: docker-compose up -d
      - name: Install dependencies
        run: pip install -e ".[dev]"
      - name: Run E2E tests
        run: pytest tests/e2e/
```

**Pipeline Stages:**
1. **Test** - Unit tests, coverage check
2. **Integration** - Integration tests with FalkorDB
3. **E2E** - Full workflow tests with Ollama

**Branch Protection Rules (GitHub):**
- ✅ Require PR reviews (1 approver minimum)
- ✅ Require status checks to pass:
  - test (unit + coverage >= 90%)
  - integration
  - e2e
- ✅ Require branches to be up to date before merge
- ✅ Do not allow force push
- ✅ Require linear history (no merge commits, rebase only)

**Result:** Cannot merge to main if ANY check fails

---

## 🔍 Code Review Standards

### Review Checklist

**1. Code Quality**

**Style & Formatting:**
- [ ] Black formatted (100 chars)
- [ ] isort applied
- [ ] Consistent naming conventions (snake_case for functions/variables)
- [ ] No commented-out code
- [ ] No debug print statements

**Readability:**
- [ ] Functions < 50 lines (ideally < 30)
- [ ] Clear variable names (no `x`, `tmp`, `data`)
- [ ] No magic numbers/strings (use constants)
- [ ] DRY principle (no code duplication)
- [ ] Complexity reasonable (cyclomatic complexity < 10)

**Documentation:**
- [ ] Docstrings present (Google style)
- [ ] Complex logic has explanatory comments
- [ ] Type hints 100% (all params, returns)
- [ ] Public API documented

---

**2. Testing**

**Coverage:**
- [ ] Overall coverage >= 90%
- [ ] All new functions have tests
- [ ] All edge cases from specs tested
- [ ] Happy path + error paths tested

**Quality:**
- [ ] Test names descriptive (`test_function_scenario_expected`)
- [ ] Tests isolated (no shared state, no order dependency)
- [ ] Mocking used appropriately (external deps only)
- [ ] Assertions clear (not `assert result` but `assert result.success is True`)

---

**3. Security**

**Input Validation:**
- [ ] All user inputs validated before use
- [ ] No SQL injection vectors (using parameterized queries or ORM)
- [ ] No command injection (avoid shell=True, validate inputs)
- [ ] No path traversal (validate file paths)
- [ ] No XSS (if web UI, sanitize outputs)

**Error Handling:**
- [ ] Exceptions caught appropriately (not bare `except:`)
- [ ] No sensitive data in error messages (passwords, tokens)
- [ ] Proper logging (no secrets logged)
- [ ] Resources cleaned up (connections closed, files closed)

**Data Protection:**
- [ ] Sensitive data encrypted if stored
- [ ] No hardcoded secrets (use environment variables)
- [ ] API keys not in code (use config)

---

**4. Alignment with Specs**

**Spec Compliance:**
- [ ] All function signatures match specs exactly
- [ ] All parameters match spec types
- [ ] All return types match specs
- [ ] All exceptions from specs raised correctly

**Completeness:**
- [ ] All required functions implemented (nothing missing)
- [ ] No extra functions (scope creep)
- [ ] All edge cases from specs handled in code

---

**5. Performance**

**Efficiency:**
- [ ] No obvious performance issues (N^2 loops where N could be large)
- [ ] Database queries optimized (no N+1 queries)
- [ ] Resource usage reasonable (no memory leaks)
- [ ] Caching used where appropriate

**Benchmarks:**
- [ ] Performance meets SLAs from tech.md
  - add_memory: < 500ms
  - search_memory: < 200ms
  - Memory: < 2GB for 100K items

---

### Review Process

**1. Automated Review:**
- Pre-commit hooks already ran (black, mypy, etc)
- CI/CD pipeline ran (tests, coverage, integration)

**2. Agent Code Review:**
- Code review agent analyzes PR
- Generates review report (APPROVE / REQUEST_CHANGES / REJECT)

**3. Human Review (if critical):**
- User reviews agent's assessment
- User makes final decision

**4. Addressing Feedback:**
- If REQUEST_CHANGES: agent or user fixes issues
- Re-run automated checks
- Re-review

**5. Merge:**
- All checks green ✅
- Review approved ✅
- Squash and merge to main
- Delete feature branch

---

## 📊 Performance Benchmarking

### SLAs (Service Level Agreements)

From tech.md:

**Latency Targets:**
- **add_memory:** < 500ms (P95) for 2KB text
- **search_memory:** < 200ms (P95) for 10K memories in DB
- **get_stats:** < 50ms (P95)

**Throughput Targets:**
- **add_memory:** >= 10 requests/second
- **search_memory:** >= 50 requests/second

**Resource Targets:**
- **Memory usage:** < 2GB for 100,000 memories stored
- **Disk usage:** < 10GB for 100,000 memories + indices
- **Startup time:** < 5 seconds (MCP server ready)

### Benchmarking Process

**Setup:**
```python
# tests/performance/bench_add_memory.py
import pytest
import time

def bench_add_memory_latency(benchmark):
    """Benchmark add_memory latency."""
    tool = AddMemoryTool(...)

    result = benchmark(
        tool.execute,
        {"text": "Sample text " * 100}  # ~2KB
    )

    # Assert SLA
    assert benchmark.stats['mean'] < 0.5  # 500ms
```

**Running Benchmarks:**
```bash
pytest tests/performance/ --benchmark-only
```

**Output:**
```
---------------------------------- benchmark: add_memory ----------------------------------
Name                         Min      Max     Mean    StdDev   Median     P95
add_memory_latency       234.5ms  512.3ms  342.1ms   45.2ms  338.7ms  467.3ms  ✅ PASS

SLA: < 500ms P95 → 467.3ms ✅
```

**If SLA Failed:**
1. Profile code (cProfile, line_profiler)
2. Identify bottlenecks
3. Optimize
4. Re-benchmark

---

## 🔐 Security Validation

### Security Checklist

**1. Input Validation**
- [ ] All inputs validated (type, format, range)
- [ ] SQL injection prevented (ORM or parameterized queries)
- [ ] Command injection prevented (no shell=True, validate paths)
- [ ] Path traversal prevented (validate file paths, use safe joins)

**2. Authentication & Authorization**
- [ ] MCP protocol: stdio mode (local only, no network auth needed)
- [ ] File access: restricted to zapomni data directory
- [ ] No elevation of privileges

**3. Data Protection**
- [ ] Sensitive data encrypted at rest (if applicable)
- [ ] No secrets in logs or error messages
- [ ] Environment variables for configuration (not hardcoded)

**4. Dependencies**
- [ ] All dependencies from trusted sources (PyPI official)
- [ ] Dependency versions pinned (no vulnerabilities)
- [ ] Regular dependency updates (security patches)

**5. Error Handling**
- [ ] No stack traces to users (log internally, generic message to user)
- [ ] Resources cleaned up on errors (connections closed)
- [ ] No denial of service vectors (rate limiting if needed)

### Security Review Process

**Automated:**
- Bandit (security linter for Python)
- Safety (dependency vulnerability scanner)

**Manual:**
- Code review includes security checklist
- Penetration testing (if applicable for production)

---

## 📈 Quality Metrics Dashboard

### Tracked Metrics

**Code Quality:**
- Lines of code (LOC)
- Test-to-code ratio
- Cyclomatic complexity (average, max)
- Technical debt ratio

**Test Coverage:**
- Overall: current / target (90%)
- Unit: current / target (95%)
- Integration: current / target (85%)
- E2E: current / target (70%)

**Test Health:**
- Tests passing: count / total (%)
- Test execution time: average, total
- Flaky tests: count (should be 0)

**Performance:**
- add_memory latency: P50, P95, P99
- search_memory latency: P50, P95, P99
- Memory usage: current / target

**Security:**
- Known vulnerabilities: count (should be 0)
- Security linting issues: count (should be 0)

### Visualization (Dashboard)

```
┌─────────────────── Code Quality ────────────────────┐
│ Coverage:  █████████░ 92% (target: 90% ✅)          │
│ Type Hint: ██████████ 100% ✅                        │
│ Linting:   0 errors ✅                               │
│ Complexity: Avg 4.2, Max 8 ✅ (target: < 10)        │
└─────────────────────────────────────────────────────┘

┌─────────────────── Test Health ─────────────────────┐
│ Passing:   145/145 (100%) ✅                         │
│ Execution: 23.4s (fast ✅)                           │
│ Flaky:     0 ✅                                      │
└─────────────────────────────────────────────────────┘

┌─────────────────── Performance ─────────────────────┐
│ add_memory:    P95 467ms  ✅ (SLA: < 500ms)         │
│ search_memory: P95 178ms  ✅ (SLA: < 200ms)         │
│ Memory usage:  1.8GB      ✅ (target: < 2GB)        │
└─────────────────────────────────────────────────────┘

┌─────────────────── Security ────────────────────────┐
│ Vulnerabilities: 0 ✅                                │
│ Security Lint:   0 issues ✅                         │
│ Last Scan:       2 hours ago                        │
└─────────────────────────────────────────────────────┘
```

---

## 📚 Best Practices

### Writing Testable Code

**1. Single Responsibility Principle**
```python
# ❌ BAD - Function does too much
def add_and_search(text):
    add_memory(text)
    return search_memory(text)

# ✅ GOOD - Separate functions
def add_memory(text):
    ...

def search_memory(query):
    ...
```

**2. Dependency Injection**
```python
# ❌ BAD - Hard to test (creates DB inside)
class AddMemoryTool:
    def __init__(self):
        self.db = FalkorDBClient()  # Hard to mock

# ✅ GOOD - Injectable dependency
class AddMemoryTool:
    def __init__(self, db_client: DatabaseClient):
        self.db = db_client  # Easy to mock in tests
```

**3. Avoid Side Effects**
```python
# ❌ BAD - Modifies global state
global_config = {}

def set_config(key, value):
    global_config[key] = value

# ✅ GOOD - Returns new state, no side effects
def create_config(base_config, key, value):
    new_config = base_config.copy()
    new_config[key] = value
    return new_config
```

---

## 📖 References

- [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) - Overall plan
- [SPEC_METHODOLOGY.md](SPEC_METHODOLOGY.md) - Spec creation
- [structure.md](.spec-workflow/steering/structure.md) - Project structure
- [tech.md](.spec-workflow/steering/tech.md) - Performance SLAs

---

**Author:** Goncharenko Anton aka alienxs2
**License:** MIT
**GitHub:** https://github.com/alienxs2/zapomni

*This document defines the quality assurance framework for Zapomni development.*
