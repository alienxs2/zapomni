# Git Integration - Tasks

## 1. CLI Infrastructure
- [ ] 1.1 Create `src/zapomni_cli/` module structure
- [ ] 1.2 Implement `__main__.py` with argparse/click
- [ ] 1.3 Add `zapomni` script entry point to `pyproject.toml`
- [ ] 1.4 Add CLI dependencies (click/typer if needed)

## 2. Hook Installation Command
- [ ] 2.1 Implement `GitHookManager` class
- [ ] 2.2 Implement `detect_git_repository()` function
- [ ] 2.3 Create hook script templates (bash/python hybrid)
- [ ] 2.4 Implement `install_hooks()` - write scripts to `.git/hooks/`
- [ ] 2.5 Add `chmod +x` for hook scripts
- [ ] 2.6 Implement hook backup mechanism

## 3. Hook Scripts
- [ ] 3.1 Create `post-commit` hook template
- [ ] 3.2 Create `post-merge` hook template
- [ ] 3.3 Create `post-checkout` hook template
- [ ] 3.4 Add error handling (log to `.git/zapomni-hooks.log`)
- [ ] 3.5 Extract changed files using `git diff`

## 4. Delta Indexing Command
- [ ] 4.1 Create `index-delta` CLI subcommand
- [ ] 4.2 Accept file paths as arguments
- [ ] 4.3 Initialize MCP components (db, embedder, processor)
- [ ] 4.4 Call delta indexing for specified files only
- [ ] 4.5 Handle file deletions (mark stale, prune)

## 5. Testing
- [ ] 5.1 Unit tests for `GitHookManager`
- [ ] 5.2 Integration test: install hooks → commit → verify indexing
- [ ] 5.3 Test hook error handling (service down)
- [ ] 5.4 Test uninstall/cleanup

## 6. Documentation
- [ ] 6.1 Update README with `zapomni install-hooks` usage
- [ ] 6.2 Add troubleshooting guide for hooks
- [ ] 6.3 Document `.zapomnirc.toml` configuration
