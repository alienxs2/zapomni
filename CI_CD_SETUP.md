# GitHub Actions CI/CD Setup for Zapomni

## Overview

Comprehensive CI/CD pipelines have been configured for Zapomni using GitHub Actions. These workflows automate testing, linting, building, and packaging.

## Directory Structure

```
.github/
├── workflows/          # GitHub Actions workflow files
│   ├── tests.yml      # Testing pipeline (unit + integration)
│   ├── lint.yml       # Code quality checks
│   └── build.yml      # Package building and PyPI publishing
├── ISSUE_TEMPLATE/    # GitHub issue templates
│   ├── bug_report.md  # Bug report template
│   └── feature_request.md  # Feature request template
└── PULL_REQUEST_TEMPLATE.md  # Pull request template
```

## Workflows

### 1. Tests Workflow (`.github/workflows/tests.yml`)

**Purpose:** Automated testing with multiple Python versions and services

**Triggers:**
- Push to `main` and `develop` branches
- Pull requests to `main` and `develop` branches

**Configuration:**
- **Python versions:** 3.10, 3.11, 3.12 (matrix strategy)
- **Runner:** Ubuntu Latest
- **Services:** FalkorDB (port 6381), Redis (port 6380)

**Steps:**
1. Checkout code
2. Setup Python with pip caching
3. Install dependencies: `pip install -e ".[dev]"`
4. Wait for services (health checks via redis-cli)
5. Run unit tests: `pytest tests/unit/ -v --cov=src --cov-report=xml`
6. Run integration tests: `pytest tests/integration/ -v --cov=src --cov-append --cov-report=xml`
7. Upload coverage to Codecov

**Coverage:**
- XML format reports generated and combined
- Uploaded to Codecov (non-blocking failure)
- Tracks both unit and integration test coverage

### 2. Lint Workflow (`.github/workflows/lint.yml`)

**Purpose:** Code quality and style enforcement

**Triggers:**
- Push to `main` and `develop` branches
- Pull requests to `main` and `develop` branches

**Configuration:**
- **Python version:** 3.10
- **Runner:** Ubuntu Latest
- **Directories checked:** `src/` and `tests/`

**Tools & Checks:**
1. **Black** - Code formatter
   - Command: `black --check src/ tests/`
   - Configuration: 100 char line length (from `pyproject.toml`)

2. **isort** - Import sorter
   - Command: `isort --check-only src/ tests/`
   - Configuration: black profile, 100 char line length

3. **flake8** - Style guide enforcement
   - Command: `flake8 src/ tests/ --max-line-length=100 --extend-ignore=E203,W503`
   - Ignored rules:
     - E203: Whitespace before ':'
     - W503: Line break before binary operator

4. **mypy** - Static type checker
   - Command: `mypy src/`
   - Configuration: Strict mode from `pyproject.toml`

**Behavior:**
- All checks run sequentially
- All checks must pass (fail-fast on first violation)
- No continue-on-error flags set

### 3. Build Workflow (`.github/workflows/build.yml`)

**Purpose:** Package building, verification, and distribution

**Triggers:**
- Push to `main` branch
- Push to version tags matching `v*`
- Pull requests to `main` branch

**Configuration:**
- **Python version:** 3.10
- **Runner:** Ubuntu Latest

#### Build Job

**Steps:**
1. Checkout code
2. Setup Python with pip caching
3. Install build tools: `build`, `wheel`, `setuptools`
4. Build wheel: `python -m build`
5. Verify artifacts:
   - Check `dist/` directory exists
   - Verify `.whl` file created
   - Fail if no wheel found
6. Test installation:
   - Install from built wheel
   - Verify `zapomni_mcp` module importable
7. Test server startup:
   - Import `zapomni_mcp.server`
   - Verify `main` function is callable
8. Upload artifacts:
   - Artifact name: `dist-{python-version}`
   - Retention: 90 days

#### Publish-to-PyPI Job

**Conditions:**
- Only runs after build job succeeds
- Only triggers on version tags (`refs/tags/v*`)
- Requires `PYPI_API_TOKEN` GitHub secret

**Steps:**
1. Download build artifacts
2. Publish to PyPI:
   - Uses `pypa/gh-action-pypi-publish`
   - Skips if version already exists
3. Create GitHub Release:
   - Attaches all built wheels
   - Detects pre-release from tag (alpha/beta)

## Templates

### Bug Report Template (`.github/ISSUE_TEMPLATE/bug_report.md`)

Structured template for bug reports with sections:
- Description
- Steps to Reproduce
- Expected vs Actual Behavior
- Environment (OS, Python version, Zapomni version)
- Configuration
- Logs & Error Messages
- Minimal Reproducible Example
- Additional Context
- Pre-submission Checklist

### Feature Request Template (`.github/ISSUE_TEMPLATE/feature_request.md`)

Structured template for feature requests with sections:
- Summary
- Problem Statement
- Proposed Solution
- Alternative Solutions
- Benefits (with checkboxes for categories)
- Use Cases
- Implementation Details
- Backwards Compatibility Assessment
- Additional Context
- Pre-submission Checklist

### Pull Request Template (`.github/PULL_REQUEST_TEMPLATE.md`)

Comprehensive PR template with sections:
- Description
- Type of Change (radio buttons)
- Related Issues
- Motivation and Context
- Testing Details
- Breaking Changes
- Backwards Compatibility
- Performance Impact
- Documentation Status
- Checklist (code style, tests, etc.)
- Screenshots (for UI changes)
- Reviewer Notes
- CI/CD Status

## Setup Instructions

### 1. GitHub Secrets (Optional)

For PyPI publishing, add this secret to your GitHub repository:

**Setting:** `Settings` → `Secrets and variables` → `Actions`

- **Name:** `PYPI_API_TOKEN`
- **Value:** Your PyPI API token

The `GITHUB_TOKEN` is provided automatically by GitHub Actions.

### 2. Branch Protection (Recommended)

Configure branch protection on `main` branch:

1. Go to `Settings` → `Branches` → `Branch protection rules`
2. Add rule for `main` branch
3. Enable:
   - Require status checks to pass before merging
   - Require branches to be up to date before merging
   - Require code reviews before merging

### 3. Codecov Integration (Optional)

For coverage tracking on Codecov:

1. Visit https://codecov.io
2. Connect your GitHub account
3. Enable the repository
4. No token needed for public repos

## Configuration Alignment

### pyproject.toml Integration

All workflows respect project configuration from `pyproject.toml`:

**Black:**
- Line length: 100
- Target version: py310

**isort:**
- Profile: black
- Line length: 100
- Multi-line: 3 (vertical hanging indent)

**mypy:**
- Python version: 3.10
- Strict mode: enabled
- Disallowed untyped defs: true

**pytest:**
- Test paths: `tests/`
- Markers: unit, integration, e2e, requires_ollama
- Coverage source: `src/`

**flake8:**
- Max line length: 100
- Ignored rules: E203, W503

### Service Ports

Matches `docker-compose.yml`:
- **FalkorDB:** 6381
- **Redis:** 6380
- **Ollama:** 11434 (not used in CI/CD)

### Python Versions

Aligns with `pyproject.toml`:
- **Minimum:** 3.10
- **Tested versions:** 3.10, 3.11, 3.12

## Workflow Execution Details

### Test Workflow Execution Time

Estimated times (per Python version):
- Python setup: ~30 seconds
- Dependencies install: ~60 seconds
- Service startup: ~30 seconds
- Unit tests: ~30 seconds (varies)
- Integration tests: ~60 seconds (varies)
- Coverage upload: ~10 seconds
- **Total per version:** ~3 minutes
- **Total (3 versions):** ~9 minutes

### Lint Workflow Execution Time

- Python setup: ~30 seconds
- Dependencies install: ~60 seconds
- Black check: ~10 seconds
- isort check: ~10 seconds
- flake8 check: ~10 seconds
- mypy check: ~20 seconds
- **Total:** ~2 minutes

### Build Workflow Execution Time

- Python setup: ~30 seconds
- Build tools install: ~30 seconds
- Wheel build: ~30 seconds
- Artifact upload: ~10 seconds
- PyPI publish (on tags): ~10 seconds
- Release creation: ~5 seconds
- **Total:** ~2 minutes

## Troubleshooting

### Tests Fail: Service Connection Issues

**Problem:** Tests timeout waiting for FalkorDB/Redis

**Solutions:**
1. Services may need more startup time on slow runners
2. Check service logs in GitHub Actions UI
3. Increase timeout in workflow (adjust `timeout 30`)

### Build Fails: Wheel Not Created

**Problem:** `python -m build` fails or doesn't create wheel

**Solutions:**
1. Verify `pyproject.toml` is valid: `pip install build && python -m build --help`
2. Check for compilation errors in source
3. Ensure setuptools, wheel, build are installed

### PyPI Publish Fails: Authentication

**Problem:** "Unauthorized" error when publishing to PyPI

**Solutions:**
1. Verify `PYPI_API_TOKEN` secret is set correctly
2. Check token is valid and not expired on PyPI
3. Verify token has upload permissions

### Type Checking Fails: Module Not Found

**Problem:** mypy can't find imported modules

**Solutions:**
1. Check `mypy.overrides` in `pyproject.toml` for untyped modules
2. Ensure all dev dependencies are installed
3. Add `[tool.mypy] ignore_missing_imports = true` for problematic packages

## Best Practices

1. **Keep Workflows Updated:**
   - Review quarterly for new GitHub Actions versions
   - Update action versions (checkout@v4, setup-python@v4, etc.)

2. **Monitor CI/CD Performance:**
   - Watch execution times
   - Cache optimization opportunities
   - Consider parallel job execution if needed

3. **Handle Flaky Tests:**
   - Use `pytest-xdist` for parallel execution if needed
   - Add retries for integration tests if appropriate
   - Document flaky tests with markers

4. **Code Quality:**
   - Pre-commit hooks match CI/CD checks
   - Run locally before pushing: `black`, `isort`, `flake8`, `mypy`, `pytest`
   - Use `.pre-commit-config.yaml` for consistency

5. **Release Process:**
   - Use semantic versioning (v0.1.0, v1.0.0, etc.)
   - Tag releases with: `git tag -a v0.1.0 -m "Release 0.1.0"`
   - Push tags: `git push origin v0.1.0`

## Local Development

To replicate CI/CD checks locally:

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Start services
docker-compose up -d

# Run all checks
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/
pytest tests/unit/ -v
pytest tests/integration/ -v

# Stop services
docker-compose down
```

## CI/CD Status Badge

Add to README.md for workflow status display:

```markdown
[![Tests](https://github.com/alienxs2/zapomni/actions/workflows/tests.yml/badge.svg)](https://github.com/alienxs2/zapomni/actions/workflows/tests.yml)
[![Lint](https://github.com/alienxs2/zapomni/actions/workflows/lint.yml/badge.svg)](https://github.com/alienxs2/zapomni/actions/workflows/lint.yml)
[![Build](https://github.com/alienxs2/zapomni/actions/workflows/build.yml/badge.svg)](https://github.com/alienxs2/zapomni/actions/workflows/build.yml)
```

## Maintenance

### Updating Python Versions

To add or remove Python versions from testing:

1. Edit `.github/workflows/tests.yml`
2. Modify the `strategy.matrix.python-version` list
3. Example: `["3.10", "3.11", "3.12", "3.13"]`

### Updating Dependencies

If dev dependencies change:

1. Update `pyproject.toml`
2. Run `pip install -e ".[dev]"` locally
3. Update any workflow steps that reference specific packages
4. Commit changes

### Monitoring Coverage

Track coverage trends:

1. Link Codecov to repository
2. Configure coverage thresholds in Codecov dashboard
3. Review coverage reports in pull requests
4. Address coverage gaps in code reviews

## References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Act - Run GitHub Actions locally](https://github.com/nektos/act)
- [pytest Documentation](https://docs.pytest.org/)
- [Black Code Formatter](https://black.readthedocs.io/)
- [mypy Documentation](https://mypy.readthedocs.io/)
- [flake8 Documentation](https://flake8.pycqa.org/)
