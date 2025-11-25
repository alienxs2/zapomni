# Git Integration - Design

## Architecture

### Components
1. **CLI Module** (`src/zapomni_cli/`) - New module for CLI commands
2. **Hook Installer** - Install/uninstall Git hooks
3. **Hook Scripts** - Bash/Python scripts in `.git/hooks/`
4. **Delta Indexer** - Wrapper around existing `index_codebase` tool

### Flow Diagram
```
User runs: zapomni install-hooks
    ↓
CLI detects .git/ directory
    ↓
Generates hook scripts (post-checkout, post-merge, post-commit)
    ↓
Copies scripts to .git/hooks/ (makes executable)
    ↓
===== Later: Git Operation =====
    ↓
Git operation completes (commit/checkout/merge)
    ↓
Hook script executes
    ↓
Extract changed files from git diff
    ↓
Call zapomni indexer with file list
    ↓
Delta indexing updates only changed files
```

## Implementation Details

### 1. CLI Structure
```
src/zapomni_cli/
  __init__.py
  __main__.py        # Entry point: python -m zapomni_cli
  commands/
    __init__.py
    install_hooks.py # Install hooks command
    uninstall_hooks.py # Uninstall hooks command
```

Add to `pyproject.toml`:
```toml
[project.scripts]
zapomni = "zapomni_cli.__main__:main"
```

### 2. Hook Scripts
**Template**: `.git/hooks/post-commit`
```bash
#!/bin/bash
# Zapomni auto-indexing hook
# Get changed files
FILES=$(git diff-tree --no-commit-id --name-only -r HEAD)
# Call indexer in background
python -m zapomni_cli index-delta --files "$FILES" >> .git/zapomni-hooks.log 2>&1 &
```

**post-checkout**:
```bash
git diff --name-only $1 $2
```

**post-merge**:
```bash
git diff --name-only ORIG_HEAD HEAD
```

### 3. Delta Indexing API
Extend `index_codebase` tool or create new `index_delta` command:
```python
# CLI: zapomni index-delta --files file1.py file2.py
def index_delta(file_paths: List[str]) -> None:
    """Index only specified files using delta indexing."""
    # 1. Mark files as stale
    # 2. Re-index each file
    # 3. Mark fresh
    # 4. Prune deleted files
```

### 4. Hook Management
```python
class GitHookManager:
    def install_hooks(self, hooks: List[str]) -> None:
        """Install Git hooks for zapomni indexing."""
        # Detect .git/
        # Generate scripts from templates
        # Write to .git/hooks/
        # Make executable (chmod +x)
        # Backup existing hooks

    def uninstall_hooks(self) -> None:
        """Remove zapomni hooks."""
        # Remove hook scripts
        # Restore backups if exist
```

## Data Model
No new database schema. Reuses existing:
- Memory nodes (code files)
- `is_stale` property for delta tracking
- File path metadata

## Error Handling
- **Hook script errors**: Log to `.git/zapomni-hooks.log`, exit 0 (don't block Git)
- **Service unavailable**: Skip indexing, log warning
- **Invalid files**: Skip individual files, continue batch

## Configuration
Optional `.zapomnirc.toml` in repo root:
```toml
[git_hooks]
enabled = true
hooks = ["post-commit", "post-merge", "post-checkout"]
background = true  # Index in background
log_file = ".git/zapomni-hooks.log"

[git_hooks.patterns]
include = ["*.py", "*.js", "*.ts"]
exclude = ["tests/*", "*.test.py"]
```

## Performance Optimizations
- Background execution (don't block Git)
- Batch file processing
- Skip unchanged files (hash comparison)
- Async indexing pipeline

## Security Considerations
- Validate file paths (no path traversal)
- Sanitize git output
- Run with user permissions only
- No sensitive data in logs
