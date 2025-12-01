# Issue #27: Bi-temporal Model

**Status**: Planning Complete
**Milestone**: v0.8.0 - Knowledge Graph 2.0
**Estimate**: 8 days
**Created**: 2025-12-01

## Overview

Bi-temporal model transforms Zapomni from a "current code snapshot" into a "time machine for code", enabling AI agents to analyze change history and perform time-travel queries.

## Two Temporal Dimensions

| Dimension | Field | Description |
|-----------|-------|-------------|
| **Valid Time** | `valid_from`, `valid_to` | When the fact was true in reality (git commit date) |
| **Transaction Time** | `created_at`, `transaction_to` | When we recorded the fact in database |

## Key Features

1. **Time-travel debugging** - Query code state at any point in time
2. **Historical queries** - "What changed in the last week?"
3. **Version tracking** - Full history of each entity
4. **Git integration** - Extract valid_from from commit dates

## New MCP Tools

| Tool | Purpose |
|------|---------|
| `get_timeline` | History of changes for a specific entity |
| `get_changes` | Changes in codebase within a time range |
| `get_snapshot` | State of codebase at a specific point in time |

## Documents

| Document | Description |
|----------|-------------|
| [requirements.md](./requirements.md) | Detailed requirements |
| [design.md](./design.md) | Technical design |
| [tasks.md](./tasks.md) | Implementation tasks |

## Related Issues

- Parent: #27 - Bi-temporal model
- Phase 1: Schema & Models
- Phase 2: Database Layer
- Phase 3: Git Integration
- Phase 4: MCP Tools
- Phase 5: Documentation & Release
