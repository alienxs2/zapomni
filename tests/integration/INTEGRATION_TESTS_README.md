# Zapomni MVP Integration Tests

Comprehensive integration test suite for the Zapomni MVP system with complete end-to-end flow validation.

## Overview

This integration test suite validates the complete MVP system including:

1. **Full Memory Pipeline Integration** - Add → Chunk → Embed → Store → Search → Retrieve flow
2. **MCP Tools Integration** - AddMemoryTool, SearchMemoryTool, GetStatsTool with MemoryProcessor
3. **Database Integration** - FalkorDB operations, transactions, and error recovery
4. **Search Integration** - Vector search, filtering, result ranking
5. **Error Handling Integration** - Error propagation and recovery

## Test File

**Location:** `/home/dev/zapomni/tests/integration/test_mvp_integration.py`

## Test Statistics

- **Total Tests:** 21
- **Test Classes:** 7
- **Test Coverage Areas:**
  - Full Memory Pipeline: 5 tests
  - Statistics Integration: 2 tests
  - Database Integration: 2 tests
  - Search Integration: 4 tests
  - MCP Tools Integration: 3 tests
  - Error Handling Integration: 3 tests
  - End-to-End Scenarios: 2 tests

## Test Classes

### 1. TestFullMemoryPipeline (5 tests)
Validates complete end-to-end memory pipeline operations.

- `test_add_and_search_memory_flow` - Add memory then search and find it
  - Validates memory chunking, embedding, storage, and retrieval
  - Checks similarity scoring

- `test_multiple_memories_search` - Add multiple memories and verify search returns ranked results
  - Validates multiple memory storage
  - Checks result ranking by similarity

- `test_memory_with_metadata` - Add memory with metadata, search with filters
  - Validates metadata storage and retrieval
  - Checks metadata-based filtering

- `test_large_text_chunking` - Large text gets chunked and stored correctly
  - Validates chunking of large documents (~10KB+)
  - Checks cross-chunk search

- `test_various_content_types` - System handles various content types (text, markdown, code)
  - Tests plain text, markdown, and code handling
  - Verifies all types are searchable

### 2. TestStatisticsIntegration (2 tests)
Validates statistics collection and reporting.

- `test_stats_after_operations` - Stats update correctly after add operations
  - Validates correct memory count tracking
  - Validates chunk count tracking
  - Checks stats update after each operation

- `test_empty_database_search` - Search returns empty on new database
  - Validates empty search returns empty results (not errors)
  - Checks initial stats show zero

### 3. TestDatabaseIntegration (2 tests)
Validates FalkorDB operations and transaction handling.

- `test_duplicate_memory_handling` - Adding similar content stores as separate memories
  - Validates duplicate content handling
  - Checks both are independently searchable

- `test_concurrent_operations` - Multiple concurrent operations don't conflict
  - Validates concurrent add operations
  - Checks database consistency
  - Validates no data loss

### 4. TestSearchIntegration (4 tests)
Validates search functionality and result accuracy.

- `test_vector_search_ranking` - Results ranked by similarity
  - Validates similarity-based ranking
  - Checks score validity (0-1 range)
  - Validates descending order

- `test_search_with_limit` - Search respects limit parameter
  - Validates limit enforcement
  - Tests with high result count

- `test_search_with_metadata_filters` - Search filters by metadata
  - Validates tag filtering
  - Validates source filtering
  - Checks filter combinations

- `test_search_result_content_accuracy` - Results contain accurate content
  - Validates result text accuracy
  - Checks metadata in results
  - Validates UUID format

### 5. TestMCPToolsIntegration (3 tests)
Validates MCP tools work with real MemoryProcessor.

- `test_add_memory_tool_integration` - AddMemoryTool works with MemoryProcessor
  - Validates MCP format input/output
  - Checks data is actually stored
  - Validates response format

- `test_search_memory_tool_integration` - SearchMemoryTool works with MemoryProcessor
  - Validates MCP format responses
  - Checks search correctness

- `test_get_stats_tool_integration` - GetStatsTool works with MemoryProcessor
  - Validates MCP format output
  - Checks stat accuracy
  - Tests with empty database

### 6. TestErrorHandlingIntegration (3 tests)
Validates error handling and recovery.

- `test_error_handling_invalid_input` - Invalid inputs handled gracefully
  - Tests empty text validation
  - Tests invalid query validation
  - Tests invalid limit validation
  - Checks database integrity after errors

- `test_error_handling_with_valid_recovery` - System recovers after errors
  - Tests recovery after validation error
  - Validates normal operations work after error
  - Checks no data corruption

- `test_mcp_tool_error_response` - MCP tools return proper error responses
  - Validates error response format
  - Checks isError flag
  - Tests MCP compliance

### 7. TestEndToEndScenarios (2 tests)
Real-world scenario testing.

- `test_complete_workflow_scenario` - Complete workflow from add to search to stats
  - Tests multi-step workflow
  - Validates all components integrate
  - Checks data consistency across tools

- `test_knowledge_building_scenario` - Building knowledge over time
  - Tests incremental knowledge building
  - Validates search improvement with data
  - Checks stats track growth

## Prerequisites

### Services Required
1. **FalkorDB** running on `localhost:6381`
   ```bash
   docker run -d --name falkordb -p 6381:6381 falkordb/falkordb:latest
   ```

2. **Ollama** with embeddings model on `localhost:11434`
   ```bash
   ollama serve
   ollama pull nomic-embed-text
   ```

### Python Dependencies
All dependencies installed via `pyproject.toml`:
- pytest >= 9.0
- pytest-asyncio
- structlog
- pydantic
- falkordb
- requests (for Ollama)

## Running Tests

### Run All Integration Tests
```bash
python3 -m pytest tests/integration/test_mvp_integration.py -v
```

### Run Specific Test Class
```bash
python3 -m pytest tests/integration/test_mvp_integration.py::TestFullMemoryPipeline -v
```

### Run Single Test
```bash
python3 -m pytest tests/integration/test_mvp_integration.py::TestFullMemoryPipeline::test_add_and_search_memory_flow -v
```

### Run with Detailed Output
```bash
python3 -m pytest tests/integration/test_mvp_integration.py -vv --tb=short
```

### Run with Coverage Report
```bash
python3 -m pytest tests/integration/test_mvp_integration.py --cov=zapomni_core --cov=zapomni_db --cov=zapomni_mcp --cov-report=html
```

## Test Execution Flow

### Fixture Setup (per test)
1. **Module-level fixtures** (once per session):
   - `falkordb_client` - Creates FalkorDB connection to dedicated test graph
   - `ollama_embedder` - Creates OllamaEmbedder instance
   - `semantic_chunker` - Creates SemanticChunker instance

2. **Function-level fixtures** (per test):
   - `memory_processor` - Fresh processor with clean database
   - `add_memory_tool` - AddMemoryTool instance
   - `search_memory_tool` - SearchMemoryTool instance
   - `get_stats_tool` - GetStatsTool instance

3. **Cleanup**:
   - Database cleared after each test
   - All connections closed properly

## Expected Test Results

When all services are running correctly:
- All 21 tests should PASS
- Total execution time: ~2-5 minutes (depends on Ollama performance)
- No data loss or corruption
- Clean database state after each test

## Test Results Analysis

### Test Report
Run the full test suite and generate a report:

```bash
python3 -m pytest tests/integration/test_mvp_integration.py \
  -v \
  --tb=short \
  --junit-xml=test-results.xml \
  --html=test-report.html
```

### Key Metrics
- **Success Rate:** Should be 100% with all prerequisites
- **Average Test Duration:** 5-30 seconds (depends on test)
- **Database Operations:** Each test performs complete add/search/stats cycle
- **Error Handling:** 3 tests verify error scenarios don't corrupt data

## Known Issues & Limitations

### FalkorDB datetime() Function
- **Issue:** FalkorDB may not support `datetime()` function in some versions
- **Workaround:** Store timestamps as ISO strings instead
- **Status:** Being addressed in schema_manager

### Ollama Event Loop
- **Issue:** Event loop closed errors in some scenarios
- **Workaround:** System automatically falls back to CPU embeddings
- **Status:** Handled gracefully with fallback

## Architecture & Design

### Test Fixtures Design
- **Session-level fixtures**: Expensive resources (DB connections, embedders)
- **Function-level fixtures**: Per-test setup/cleanup for isolation
- **Async fixtures**: Using pytest-asyncio for async test support

### Error Handling Strategy
- Each test validates successful operations AND error scenarios
- Database integrity checked after error tests
- MCP protocol compliance verified for all tool tests

### Data Isolation
- Each test uses a dedicated FalkorDB graph instance
- Graph cleared before and after each test
- No cross-test data leakage

## Coverage Goals

This integration test suite provides MVP validation for:

| Component | Coverage | Tests |
|-----------|----------|-------|
| Memory Pipeline | End-to-end | 5 |
| Vector Search | Full | 4 |
| MCP Tools | Integration | 3 |
| Error Handling | Recovery | 3 |
| Database | Transactions & Concurrency | 2 |
| Statistics | Collection & Reporting | 2 |
| Scenarios | Real-world workflows | 2 |
| **Total** | **MVP Critical Paths** | **21** |

## Debugging Failed Tests

### Check Services Running
```bash
# FalkorDB
redis-cli -p 6381 ping

# Ollama
curl http://localhost:11434/api/tags
```

### Enable Debug Logging
```bash
python3 -m pytest tests/integration/test_mvp_integration.py -vv --log-cli-level=DEBUG
```

### Inspect Database State
```bash
# Connect to FalkorDB test graph
redis-cli -p 6381
> GRAPH.QUERY zapomni_test_mvp "MATCH (n) RETURN COUNT(n)"
```

## Maintenance & Updates

### Adding New Tests
1. Add test method to appropriate test class
2. Include docstring explaining what's validated
3. Use existing fixtures (no new setup code needed)
4. Mark with `@pytest.mark.asyncio` for async tests
5. Include assertions with clear failure messages

### Updating Test Data
- Modify test text in test methods
- Update expected values in assertions
- Consider impact on multiple tests

### Database Schema Changes
- Update schema in schema_manager.py
- Tests will auto-detect and use new schema
- No fixture changes needed unless schema is incompatible

## Contact & Support

For issues with integration tests:
1. Check prerequisites are running
2. Review test output and logs
3. Check FalkorDB and Ollama service status
4. Verify network connectivity

## Author
Goncharenko Anton aka alienxs2

## License
MIT
