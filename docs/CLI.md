# Zapomni CLI Documentation

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Git Hooks Integration](#git-hooks-integration)
- [Available Commands](#available-commands)
- [Hook Scripts](#hook-scripts)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## Overview

The Zapomni CLI (`zapomni_cli`) provides command-line utilities for managing Zapomni operations outside of the MCP protocol. The primary feature is **Git hooks integration** for automatic code re-indexing.

**Key Features**:
- Install Git hooks for automatic re-indexing
- Post-commit, post-merge, post-checkout hooks
- Backup existing hooks
- Standalone tool (no dependencies on other Zapomni layers)

**Location**: `src/zapomni_cli/` (~100 lines of code)

## Installation

### Via pip (if installed as package)

```bash
# After installing zapomni
pip install zapomni

# CLI is available as 'zapomni' command
zapomni --help
```

### Via python -m (development)

```bash
# From project root
python -m zapomni_cli --help
```

### Check Installation

```bash
zapomni --version
# Output: zapomni 1.0.0
```

---

## Git Hooks Integration

Git hooks enable automatic re-indexing of code files when Git operations occur. This keeps your code memory in sync with your repository.

### What are Git Hooks?

Git hooks are scripts that run automatically on specific Git events:
- **post-commit**: After each commit
- **post-merge**: After pulling/merging changes
- **post-checkout**: After switching branches

### Why Use Git Hooks?

Without hooks:
```bash
# Manual re-indexing after changes
git commit -am "Updated parser.py"
# MCP client must manually call index_codebase
```

With hooks:
```bash
# Automatic re-indexing
git commit -am "Updated parser.py"
# Hook detects changed files → automatically triggers re-indexing
```

**Benefits**:
- Always up-to-date code search
- No manual intervention required
- Transparent to workflow
- Background processing (non-blocking)

---

## Available Commands

### install-hooks

Install Git hooks for automatic re-indexing.

**Usage**:
```bash
zapomni install-hooks [--repo-path PATH]
```

**Options**:
- `--repo-path PATH`: Path to Git repository (default: current directory)
- `--help`: Show help message

**Example**:
```bash
# Install in current directory
cd /path/to/your/project
zapomni install-hooks

# Install in specific directory
zapomni install-hooks --repo-path /path/to/project
```

**Output**:
```
✅ Installed post-commit
✅ Installed post-merge
✅ Installed post-checkout

🎉 Successfully installed 3 Git hooks!
📂 Repository: /path/to/project

Git operations (commit, merge, checkout) will now trigger
automatic re-indexing of changed files.
```

### --version

Display Zapomni CLI version.

```bash
zapomni --version
# Output: zapomni 1.0.0
```

### --help

Show help message and available commands.

```bash
zapomni --help
```

**Output**:
```
usage: zapomni [-h] [--version] {install-hooks} ...

Zapomni MCP Server CLI

positional arguments:
  {install-hooks}  Available commands
    install-hooks  Install Git hooks for automatic re-indexing

optional arguments:
  -h, --help       show this help message and exit
  --version        show program's version number and exit
```

---

## Hook Scripts

### post-commit

Triggers re-indexing after each commit.

**Location**: `src/zapomni_cli/hooks/post-commit`

**What it does**:
1. Detects files changed in the latest commit
2. Logs changes to `.git/zapomni-hooks.log`
3. (TODO) Calls `index_codebase` MCP tool with changed files

**Script**:
```bash
#!/bin/sh
# Zapomni Git Hook - post-commit
# Triggers re-indexing of files changed in the latest commit

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null)
if [ -z "$REPO_ROOT" ]; then
    exit 0
fi

# Get changed files in HEAD commit
CHANGED_FILES=$(git diff-tree --no-commit-id --name-only -r HEAD 2>/dev/null)

if [ -z "$CHANGED_FILES" ]; then
    exit 0
fi

# Log hook execution
LOG_FILE="$REPO_ROOT/.git/zapomni-hooks.log"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] post-commit: Detected changes" >> "$LOG_FILE"

# Log changed files
echo "$CHANGED_FILES" | while read file; do
    echo "  - $file" >> "$LOG_FILE"
done

# TODO: Call index_codebase MCP tool
exit 0
```

**Example Log** (`.git/zapomni-hooks.log`):
```
[2025-11-26 10:30:45] post-commit: Detected changes in commit abc123
  - src/parser.py
  - src/tokenizer.py
  - tests/test_parser.py
```

### post-merge

Triggers re-indexing after pulling or merging changes.

**Location**: `src/zapomni_cli/hooks/post-merge`

**What it does**:
1. Detects files changed in the merge
2. Logs merge event
3. (TODO) Calls `index_codebase` with changed files

**Use Cases**:
- After `git pull`
- After `git merge`
- Team collaboration (sync with remote changes)

### post-checkout

Triggers re-indexing after switching branches.

**Location**: `src/zapomni_cli/hooks/post-checkout`

**What it does**:
1. Detects branch switch
2. Identifies files changed between branches
3. (TODO) Re-indexes changed files

**Use Cases**:
- Switching between feature branches
- Checking out different versions
- Ensures code memory matches current branch

---

## Configuration

### Hook Log File

Hooks log activity to `.git/zapomni-hooks.log`:

**Location**: `<repo>/.git/zapomni-hooks.log`

**Format**:
```
[TIMESTAMP] hook-name: Event description
  - changed-file-1.py
  - changed-file-2.py
```

**Viewing Logs**:
```bash
# View hook activity
cat .git/zapomni-hooks.log

# Tail logs in real-time
tail -f .git/zapomni-hooks.log
```

### Disabling Hooks

Temporarily disable hooks without uninstalling:

```bash
# Rename hook to disable
mv .git/hooks/post-commit .git/hooks/post-commit.disabled

# Re-enable
mv .git/hooks/post-commit.disabled .git/hooks/post-commit
```

### Uninstalling Hooks

Remove hooks manually:

```bash
# Remove all Zapomni hooks
rm .git/hooks/post-commit
rm .git/hooks/post-merge
rm .git/hooks/post-checkout

# Restore backups if they exist
mv .git/hooks/post-commit.backup .git/hooks/post-commit
```

---

## Troubleshooting

### Issue: "Not a Git repository"

**Error**:
```
❌ Error: /path/to/dir is not a Git repository
```

**Solution**:
```bash
# Initialize Git repository first
git init

# Or specify correct path
zapomni install-hooks --repo-path /path/to/git/repo
```

### Issue: "Permission denied" when executing hooks

**Error**:
```
.git/hooks/post-commit: Permission denied
```

**Solution**:
```bash
# Make hooks executable
chmod +x .git/hooks/post-commit
chmod +x .git/hooks/post-merge
chmod +x .git/hooks/post-checkout

# Or reinstall
zapomni install-hooks
```

### Issue: Hooks not triggering

**Check hook installation**:
```bash
# Verify hooks exist
ls -la .git/hooks/post-*

# Should show:
# -rwxr-xr-x  post-commit
# -rwxr-xr-x  post-merge
# -rwxr-xr-x  post-checkout
```

**Check hook execution**:
```bash
# Make a test commit
git commit --allow-empty -m "Test hook"

# Check log file
cat .git/zapomni-hooks.log
# Should see new entry
```

### Issue: Existing hooks being overwritten

**Zapomni backs up existing hooks**:
```
📦 Backed up existing post-commit to post-commit.backup
✅ Installed post-commit
```

**Restore backup**:
```bash
mv .git/hooks/post-commit.backup .git/hooks/post-commit
```

**Merge hooks manually**:
```bash
# If you have existing hooks, merge them:
cat .git/hooks/post-commit.backup >> .git/hooks/post-commit
```

### Issue: Hooks running but no re-indexing

**Current Status**: Hooks log changes but don't trigger re-indexing (TODO in code).

**Workaround**:
```bash
# Manual re-indexing via MCP client
# (using Claude, Cursor, or Cline)
index_codebase(path="/path/to/repo")
```

**Future**: Hooks will automatically call `index_codebase` MCP tool.

---

## Development

### Hook Development

Hooks are simple shell scripts in `src/zapomni_cli/hooks/`:

**Structure**:
```bash
#!/bin/sh
# Hook description

# 1. Detect repository
REPO_ROOT=$(git rev-parse --show-toplevel)

# 2. Get changed files
CHANGED_FILES=$(git diff ...)

# 3. Log activity
echo "[...] Hook triggered" >> "$REPO_ROOT/.git/zapomni-hooks.log"

# 4. Trigger re-indexing (TODO)
# Call MCP tool: index_codebase(files=CHANGED_FILES)

exit 0
```

### Testing Hooks

```bash
# Test hook manually
cd /path/to/repo
.git/hooks/post-commit

# Check logs
cat .git/zapomni-hooks.log
```

### Adding New Hooks

1. Create hook script in `src/zapomni_cli/hooks/`:
```bash
touch src/zapomni_cli/hooks/pre-push
chmod +x src/zapomni_cli/hooks/pre-push
```

2. Add to `install_hooks.py`:
```python
HOOKS_TO_INSTALL = [
    "post-commit",
    "post-merge",
    "post-checkout",
    "pre-push",  # New hook
]
```

3. Test installation:
```bash
zapomni install-hooks --repo-path /tmp/test-repo
```

---

## Integration with MCP

### Current Architecture

```
Git Operation (commit/merge/checkout)
  ↓
Git Hook (post-commit/post-merge/post-checkout)
  ↓
Log changed files to .git/zapomni-hooks.log
  ↓
TODO: Call index_codebase MCP tool
```

### Future Integration

```
Git Operation
  ↓
Git Hook
  ↓
Detect changed files
  ↓
Call MCP tool: index_codebase(files=[...])
  ↓
RepositoryIndexer processes files
  ↓
Updated code memory in FalkorDB
```

**Implementation Plan**:
1. Add MCP client to hook script
2. Call `index_codebase` with file list
3. Handle async processing (background job)
4. Report status in log file

---

## Examples

### Basic Workflow

```bash
# 1. Install hooks
cd ~/projects/my-app
zapomni install-hooks

# 2. Make code changes
vim src/parser.py

# 3. Commit (hook triggers automatically)
git add src/parser.py
git commit -m "Improved parser performance"
# → Hook logs: [2025-11-26 10:30:45] post-commit: Detected changes
#              - src/parser.py

# 4. Check hook activity
cat .git/zapomni-hooks.log
```

### Team Setup

```bash
# Setup script for team onboarding
#!/bin/bash

# Clone repo
git clone https://github.com/team/project.git
cd project

# Install Zapomni hooks
zapomni install-hooks

# Start Zapomni MCP server
docker-compose up -d
python -m zapomni_mcp

# Configure MCP client (Claude, Cursor, Cline)
# ...
```

### CI/CD Integration

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Install Zapomni CLI
        run: pip install zapomni
      
      - name: Install Git hooks
        run: zapomni install-hooks
      
      - name: Run tests
        run: pytest
```

---

## Related Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)**: System architecture (CLI layer)
- **[API.md](API.md)**: MCP tools (index_codebase)
- **[DEVELOPMENT.md](DEVELOPMENT.md)**: Development setup
- **[CONFIGURATION.md](CONFIGURATION.md)**: Configuration options

---

## Changelog

- **2025-11-26**: Initial CLI documentation
- Added install-hooks command
- 3 hooks: post-commit, post-merge, post-checkout
- TODO: Complete MCP integration for automatic re-indexing

---

**Document Version**: 1.0
**Last Updated**: 2025-11-26
**Status**: Hooks log activity but don't trigger re-indexing yet (TODO)
