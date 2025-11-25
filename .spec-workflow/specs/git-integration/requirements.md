# Git Integration - Requirements

## Overview
Automatic code indexing via Git hooks for delta-based updates. Track only changed files after Git operations.

## User Requirements
- **CLI Command**: `zapomni install-hooks` to set up Git hooks
- **Git Hooks**: post-checkout, post-merge, post-commit
- **Delta Indexing**: Only index files changed in Git operation
- **No Watchdog**: Git hooks only (no filesystem watching)

## Functional Requirements

### FR-1: Hook Installation
- CLI command `zapomni install-hooks` installs hooks to `.git/hooks/`
- Detect if in Git repository
- Create hook scripts: `post-checkout`, `post-merge`, `post-commit`
- Make hooks executable
- Preserve existing hooks (backup or chain)

### FR-2: Hook Execution
- Hooks trigger after Git operations complete
- Extract list of changed files from Git
- Call indexing for changed files only
- Handle errors gracefully (don't block Git operations)

### FR-3: Delta Indexing
- Reuse existing delta indexing in `index_codebase` tool
- Mark stale → index changed files → mark fresh
- Support file additions, modifications, deletions
- Update existing memories for modified files

### FR-4: Configuration
- Optional: `.zapomnirc` or `pyproject.toml` config
- File patterns to include/exclude
- Hook enable/disable per hook type

## Non-Functional Requirements

### NFR-1: Performance
- Hooks complete in <2s for typical commits (1-10 files)
- Background indexing option for large changesets

### NFR-2: Safety
- Hooks never fail Git operations
- Log errors to `.git/zapomni-hooks.log`
- Graceful degradation if Zapomni service unavailable

### NFR-3: Compatibility
- Support Git 2.20+
- Work with existing Git workflows
- Compatible with other Git hooks (e.g., pre-commit)

## Out of Scope
- Filesystem watchdog/inotify
- Real-time indexing
- Non-Git VCS systems
- GUI for hook management
