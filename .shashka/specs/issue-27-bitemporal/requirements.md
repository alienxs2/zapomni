# Issue #27: Bi-temporal Model - Requirements

## Functional Requirements

### FR-1: Temporal Fields on Memory Node

The system MUST store the following temporal fields on Memory nodes:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `valid_from` | ISO 8601 timestamp | Yes | When the entity appeared in reality |
| `valid_to` | ISO 8601 timestamp | No | When the entity ceased to exist (NULL = current) |
| `transaction_to` | ISO 8601 timestamp | No | When the record became inactive in DB (NULL = current) |
| `version` | Integer | Yes | Version number (1, 2, 3...) |
| `previous_version_id` | UUID | No | Reference to previous version |
| `is_current` | Boolean | Yes | Flag for current version (optimization) |

### FR-2: Point-in-Time Queries

The system MUST support the following query types:

1. **Current State Query** - Get current version of entities
2. **Valid Time Query** - Get state as it was in reality at time T
3. **Transaction Time Query** - Get state as recorded at time T
4. **Bi-temporal Query** - Combine both time dimensions

### FR-3: MCP Tools

The system MUST provide the following MCP tools:

#### FR-3.1: get_timeline
- Input: entity_id OR file_path OR qualified_name
- Output: List of all versions with timestamps and changes
- Ordering: By valid_from DESC

#### FR-3.2: get_changes
- Input: since (datetime), until (optional), change_type, path_pattern
- Output: List of changes (created/modified/deleted)
- Filtering: By change type and path glob pattern

#### FR-3.3: get_snapshot (optional)
- Input: as_of (datetime), time_type (valid/transaction/both)
- Output: State of codebase at specified time

### FR-4: Git Integration

The system SHOULD extract valid_from from git commit dates:
- Use `git log -1 --format="%ai" -- <file>` for commit date
- Fallback to file mtime if git unavailable
- Fallback to current timestamp if mtime unavailable

### FR-5: Version Management

When a file is re-indexed with changes:
1. Close the previous version (set valid_to, transaction_to)
2. Create new version with incremented version number
3. Link to previous version via previous_version_id
4. Set is_current = true only on new version

### FR-6: Migration

The system MUST migrate existing data:
- Set valid_from = created_at for existing records
- Set valid_to = NULL, transaction_to = NULL
- Set version = 1, is_current = true
- Migration MUST be idempotent

## Non-Functional Requirements

### NFR-1: Performance

- Current state queries MUST NOT degrade by more than 10%
- is_current index MUST be used for optimization
- Point-in-time queries MAY be slower (acceptable 2-3x)

### NFR-2: Storage

- Version history increases storage proportionally
- Consider periodic compaction for old versions (future)

### NFR-3: Backwards Compatibility

- Existing API methods MUST continue to work
- New temporal parameters SHOULD be optional
- Default behavior = current state (is_current = true)

## Acceptance Criteria

- [ ] All Memory nodes have temporal fields
- [ ] Point-in-time queries return correct results
- [ ] get_timeline tool shows version history
- [ ] get_changes tool shows changes in time range
- [ ] Git integration extracts commit dates
- [ ] Migration script works on existing data
- [ ] All tests pass (target: 100+ new tests)
- [ ] Performance benchmarks meet NFR-1
