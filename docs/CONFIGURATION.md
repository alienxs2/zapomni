# Zapomni Configuration Guide

## Table of Contents

- [Overview](#overview)
- [Essential Settings](#essential-settings)
- [Connection Pool Configuration](#connection-pool-configuration)
- [Retry Configuration](#retry-configuration)
- [Redis Cache Configuration](#redis-cache-configuration)
- [Performance Tuning](#performance-tuning)
- [Logging Configuration](#logging-configuration)
- [Feature Flags](#feature-flags)
- [SSE Transport Configuration](#sse-transport-configuration)
- [System Configuration](#system-configuration)
- [Configuration Files](#configuration-files)

## Overview

Zapomni uses environment variables for configuration, managed via Pydantic Settings. Configuration can be provided through:

1. Environment variables
2. `.env` file (recommended for development)
3. Default values in code

**Total Configuration Variables**: 43 (in `.env.example`) + 7 SSE variables = 50 total

**Configuration Loading Order** (highest priority first):
1. Environment variables
2. `.env` file
3. Default values in `src/zapomni_core/config.py`

## Essential Settings

Core database and embedding service configuration.

### FalkorDB Connection

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `FALKORDB_HOST` | string | `localhost` | FalkorDB host address |
| `FALKORDB_PORT` | integer | `6381` | FalkorDB port (external port in docker-compose) |
| `FALKORDB_PASSWORD` | string | (empty) | FalkorDB password (optional, use SecretStr) |
| `GRAPH_NAME` | string | `zapomni_memory` | Graph database name |
| `FALKORDB_CONNECTION_TIMEOUT` | integer | `30` | Connection timeout in seconds |

**Example**:
```bash
FALKORDB_HOST=localhost
FALKORDB_PORT=6381
FALKORDB_PASSWORD=
GRAPH_NAME=zapomni_memory
FALKORDB_CONNECTION_TIMEOUT=30
```

**Docker Compose Mapping**:
```yaml
ports:
  - "6381:6379"  # External 6381 → Internal 6379
```

### Ollama Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `OLLAMA_BASE_URL` | string | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_EMBEDDING_MODEL` | string | `nomic-embed-text` | Embedding model (768 dimensions) |
| `OLLAMA_LLM_MODEL` | string | `llama3.1:8b` | LLM model for entity refinement |
| `OLLAMA_EMBEDDING_TIMEOUT` | integer | `60` | Embedding API timeout (seconds) |
| `OLLAMA_LLM_TIMEOUT` | integer | `120` | LLM API timeout (seconds) |

**Example**:
```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_LLM_MODEL=llama3.1:8b
OLLAMA_EMBEDDING_TIMEOUT=60
OLLAMA_LLM_TIMEOUT=120
```

**Supported Embedding Models**:
- `nomic-embed-text` (768 dimensions) - **Recommended**
- `mxbai-embed-large` (1024 dimensions)
- Custom models (update `VECTOR_DIMENSIONS` accordingly)

**Supported LLM Models**:
- `llama3.1:8b` - Good balance
- `qwen2.5:latest` - Better entity extraction
- `mistral:latest` - Fast inference

**Note**: Models must be pulled first:
```bash
ollama pull nomic-embed-text
ollama pull llama3.1:8b
```

---

## Connection Pool Configuration

Async connection pooling for concurrent SSE connections and high throughput.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `FALKORDB_POOL_SIZE` | integer | `20` | Maximum connections in pool |
| `FALKORDB_POOL_MIN_SIZE` | integer | `5` | Minimum idle connections |
| `FALKORDB_POOL_MAX_SIZE` | integer | `20` | Hard limit on connections |
| `FALKORDB_POOL_TIMEOUT` | float | `10.0` | Timeout to acquire connection (seconds) |
| `FALKORDB_SOCKET_TIMEOUT` | float | `30.0` | Socket read/write timeout (seconds) |
| `FALKORDB_HEALTH_CHECK_INTERVAL` | integer | `30` | Health check frequency (seconds) |

**Example**:
```bash
FALKORDB_POOL_SIZE=20
FALKORDB_POOL_MIN_SIZE=5
FALKORDB_POOL_MAX_SIZE=20
FALKORDB_POOL_TIMEOUT=10.0
FALKORDB_SOCKET_TIMEOUT=30.0
FALKORDB_HEALTH_CHECK_INTERVAL=30
```

**Tuning Guidelines**:
- **Low traffic** (1-5 concurrent clients): `POOL_SIZE=10`
- **Medium traffic** (5-20 concurrent clients): `POOL_SIZE=20` (default)
- **High traffic** (20-50 concurrent clients): `POOL_SIZE=50`
- **Keep** `POOL_MIN_SIZE` at ~25% of `POOL_SIZE` for warm connections

**Performance Impact**:
- Larger pools: Higher memory usage, better concurrency
- Smaller pools: Lower memory, may block on high load
- Health checks: Automatic recovery from connection failures

---

## Retry Configuration

Exponential backoff for transient database errors.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `FALKORDB_MAX_RETRIES` | integer | `3` | Maximum retry attempts |
| `FALKORDB_RETRY_INITIAL_DELAY` | float | `0.1` | Initial retry delay (seconds) |
| `FALKORDB_RETRY_MAX_DELAY` | float | `2.0` | Maximum retry delay (seconds) |

**Example**:
```bash
FALKORDB_MAX_RETRIES=3
FALKORDB_RETRY_INITIAL_DELAY=0.1
FALKORDB_RETRY_MAX_DELAY=2.0
```

**Retry Pattern** (exponential backoff):
- 1st retry: 0.1s delay
- 2nd retry: 0.2s delay
- 3rd retry: 0.4s delay

**Retriable Errors**:
- Connection timeouts
- Temporary network failures
- Database lock contention

**Non-retriable Errors**:
- Authentication failures
- Query syntax errors
- Constraint violations

---

## Redis Cache Configuration

Optional semantic caching to reduce redundant embedding generation.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `REDIS_ENABLED` | boolean | `false` | Enable Redis caching (**disabled by default**) |
| `REDIS_HOST` | string | `localhost` | Redis host address |
| `REDIS_PORT` | integer | `6380` | Redis port (external port in docker-compose) |
| `REDIS_TTL_SECONDS` | integer | `86400` | Cache TTL (24 hours) |
| `REDIS_MAX_MEMORY_MB` | integer | `1024` | Max memory for Redis (1GB) |

**Example**:
```bash
REDIS_ENABLED=false
REDIS_HOST=localhost
REDIS_PORT=6380
REDIS_TTL_SECONDS=86400
REDIS_MAX_MEMORY_MB=1024
```

**Enabling Redis Cache**:
```bash
# 1. Start Redis via docker-compose
docker-compose up -d redis

# 2. Enable in .env
REDIS_ENABLED=true

# 3. Restart Zapomni MCP server
```

**Performance Benefits**:
- **Cache Hit**: ~5ms (vs ~500ms for Ollama API call)
- **Reduced API calls**: 60-80% hit rate for repeated queries
- **Lower latency**: Faster search response times

**Memory Estimation**:
- Each embedding: ~3KB (768 floats)
- 1GB cache: ~350,000 cached embeddings

---

## Performance Tuning

Parameters for chunking, indexing, and search performance.

### Chunking Parameters

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MAX_CHUNK_SIZE` | integer | `512` | Maximum tokens per chunk |
| `CHUNK_OVERLAP` | integer | `50` | Token overlap between chunks |

**Example**:
```bash
MAX_CHUNK_SIZE=512
CHUNK_OVERLAP=50
```

**Tuning Guidelines**:
- **Smaller chunks** (256-384 tokens): Better precision, more chunks
- **Larger chunks** (512-768 tokens): Better context, fewer chunks
- **Overlap**: 10-20% of chunk size for context preservation

### Vector Index Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `VECTOR_DIMENSIONS` | integer | `768` | Embedding vector dimensions |
| `HNSW_M` | integer | `16` | HNSW connections per node |
| `HNSW_EF_CONSTRUCTION` | integer | `200` | Build quality parameter |
| `HNSW_EF_SEARCH` | integer | `100` | Search quality parameter |

**Example**:
```bash
VECTOR_DIMENSIONS=768
HNSW_M=16
HNSW_EF_CONSTRUCTION=200
HNSW_EF_SEARCH=100
```

**HNSW Parameter Guide**:
- `HNSW_M`: More connections = better recall, larger index
  - Low (8-12): Faster build, lower recall
  - Medium (16-24): Balanced (recommended)
  - High (32-64): Slower build, higher recall
  
- `HNSW_EF_CONSTRUCTION`: Higher = better index quality
  - Low (100): Fast build, lower quality
  - Medium (200): Balanced (recommended)
  - High (400): Slow build, best quality
  
- `HNSW_EF_SEARCH`: Higher = better search recall
  - Low (50): Fast search, lower recall
  - Medium (100): Balanced (recommended)
  - High (200): Slower search, higher recall

**Performance vs Quality Trade-offs**:
```
Fast Search (EF_SEARCH=50):    ~5ms per query,  ~85% recall
Balanced (EF_SEARCH=100):     ~10ms per query,  ~95% recall
High Quality (EF_SEARCH=200): ~20ms per query, ~99% recall
```

### Concurrency and Limits

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MAX_CONCURRENT_TASKS` | integer | `4` | Max parallel tasks |
| `SEARCH_LIMIT_DEFAULT` | integer | `10` | Default search result limit |
| `MIN_SIMILARITY_THRESHOLD` | float | `0.5` | Minimum similarity score (0.0-1.0) |
| `MAX_TEXT_LENGTH` | integer | `10000000` | Max text size (10MB) |

**Example**:
```bash
MAX_CONCURRENT_TASKS=4
SEARCH_LIMIT_DEFAULT=10
MIN_SIMILARITY_THRESHOLD=0.5
MAX_TEXT_LENGTH=10000000
```

**Tuning Guidelines**:
- `MAX_CONCURRENT_TASKS`: Set to CPU cores for CPU-bound tasks
- `MIN_SIMILARITY_THRESHOLD`: 
  - 0.5-0.6: Broad search (more results, lower relevance)
  - 0.7-0.8: Balanced (recommended)
  - 0.9+: Strict (very similar results only)

---

## Logging Configuration

Structured logging with JSON output.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `LOG_LEVEL` | string | `INFO` | Logging level |
| `LOG_FORMAT` | string | `json` | Log output format |
| `LOG_FILE` | string | (empty) | Optional log file path |

**Example**:
```bash
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE=
```

**Log Levels** (from most to least verbose):
- `DEBUG`: Detailed debugging info
- `INFO`: General information (default)
- `WARNING`: Warning messages
- `ERROR`: Error messages only
- `CRITICAL`: Critical errors only

**Log Formats**:
- `json`: Structured JSON logs (recommended for production)
- `console`: Human-readable console logs (development)

**Log File Output**:
```bash
# Enable file logging
LOG_FILE=/var/log/zapomni/server.log

# Logs written to both stdout and file
```

---

## Feature Flags

Control advanced features. **All enabled by default** (except semantic cache which requires Redis).

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENABLE_HYBRID_SEARCH` | boolean | `true` | Enable hybrid vector + BM25 search |
| `ENABLE_KNOWLEDGE_GRAPH` | boolean | `true` | Enable entity extraction and graph building |
| `ENABLE_CODE_INDEXING` | boolean | `true` | Enable AST-based code indexing |
| `ENABLE_SEMANTIC_CACHE` | boolean | `false` | Enable Redis embedding cache (requires Redis) |

**Default** (all features enabled):
```bash
ENABLE_HYBRID_SEARCH=true
ENABLE_KNOWLEDGE_GRAPH=true
ENABLE_CODE_INDEXING=true
ENABLE_SEMANTIC_CACHE=false  # Requires Redis
```

**To disable features**, set to `false`:

```bash
# Disable all advanced features
ENABLE_HYBRID_SEARCH=false
ENABLE_KNOWLEDGE_GRAPH=false
ENABLE_CODE_INDEXING=false
ENABLE_SEMANTIC_CACHE=false
```

### Feature Impact

| Feature | Performance Impact | Use Case |
|---------|-------------------|----------|
| Hybrid Search | +20-30ms per query | Better search relevance |
| Knowledge Graph | +500ms per add_memory | Entity extraction and relationships |
| Code Indexing | +2-5s per file | AST analysis, call graphs |
| Semantic Cache | -90% latency (on hit) | Repeated queries |

**Recommended Configurations**:

```bash
# Minimal (fast, basic features)
ENABLE_HYBRID_SEARCH=false
ENABLE_KNOWLEDGE_GRAPH=false
ENABLE_CODE_INDEXING=false
ENABLE_SEMANTIC_CACHE=false

# Balanced (good performance + features)
ENABLE_HYBRID_SEARCH=true
ENABLE_KNOWLEDGE_GRAPH=false
ENABLE_CODE_INDEXING=false
ENABLE_SEMANTIC_CACHE=true

# Full features (slower but most capable)
ENABLE_HYBRID_SEARCH=true
ENABLE_KNOWLEDGE_GRAPH=true
ENABLE_CODE_INDEXING=true
ENABLE_SEMANTIC_CACHE=true
```

---

## SSE Transport Configuration

Server-Sent Events transport for web-based MCP clients. **Not in `.env.example`** but configurable via environment variables.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ZAPOMNI_SSE_HOST` | string | `127.0.0.1` | SSE server host (localhost only for security) |
| `ZAPOMNI_SSE_PORT` | integer | `8000` | SSE server port |
| `ZAPOMNI_SSE_CORS_ORIGINS` | string | `*` | CORS allowed origins (comma-separated) |
| `ZAPOMNI_SSE_HEARTBEAT_INTERVAL` | integer | `30` | Heartbeat interval (seconds) |
| `ZAPOMNI_SSE_MAX_CONNECTION_LIFETIME` | integer | `3600` | Max connection lifetime (seconds, 1 hour) |
| `ZAPOMNI_SSE_ALLOWED_HOSTS` | string | (auto) | Allowed Host headers (comma-separated) |
| `ZAPOMNI_SSE_DNS_REBINDING_PROTECTION` | boolean | `true` | Enable DNS rebinding protection |

**Example**:
```bash
ZAPOMNI_SSE_HOST=127.0.0.1
ZAPOMNI_SSE_PORT=8000
ZAPOMNI_SSE_CORS_ORIGINS=*
ZAPOMNI_SSE_HEARTBEAT_INTERVAL=30
ZAPOMNI_SSE_MAX_CONNECTION_LIFETIME=3600
ZAPOMNI_SSE_ALLOWED_HOSTS=localhost,127.0.0.1
ZAPOMNI_SSE_DNS_REBINDING_PROTECTION=true
```

**Security Considerations**:
- `SSE_HOST=127.0.0.1`: Localhost only by default
- `CORS_ORIGINS=*`: Allow all origins (⚠️ change in production)
- `DNS_REBINDING_PROTECTION=true`: Blocks suspicious Host headers
- `ALLOWED_HOSTS`: Whitelist specific hosts

**Production Configuration**:
```bash
ZAPOMNI_SSE_HOST=0.0.0.0  # Listen on all interfaces
ZAPOMNI_SSE_CORS_ORIGINS=https://myapp.com,https://dashboard.myapp.com
ZAPOMNI_SSE_ALLOWED_HOSTS=myapp.com,dashboard.myapp.com
ZAPOMNI_SSE_DNS_REBINDING_PROTECTION=true
```

---

## System Configuration

File system and resource limits.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DATA_DIR` | string | `./data` | Data directory for temporary files |
| `TEMP_DIR` | string | `/tmp/zapomni` | Temporary processing directory |
| `MAX_TEXT_LENGTH` | integer | `10000000` | Maximum text size (10MB) |

**Example**:
```bash
DATA_DIR=./data
TEMP_DIR=/tmp/zapomni
MAX_TEXT_LENGTH=10000000
```

**Directory Structure**:
```
./data/
├── cache/          # Temporary caches
├── exports/        # Graph exports
└── uploads/        # Uploaded files
```

---

## Configuration Files

### .env.example

Reference configuration file with all 41 variables and documentation.

**Location**: `/home/dev/zapomni/.env.example`

**Usage**:
```bash
# Copy to .env and customize
cp .env.example .env

# Edit with your values
nano .env
```

### config.py

Pydantic Settings class with type validation and defaults.

**Location**: `src/zapomni_core/config.py`

**Key Features**:
- Environment variable parsing
- Type validation (Pydantic)
- Default values
- SecretStr for passwords
- Lazy singleton pattern

**Example**:
```python
from zapomni_core.config import get_settings

settings = get_settings()
print(settings.falkordb_host)  # localhost
print(settings.ollama_base_url)  # http://localhost:11434
```

### docker-compose.yml

Docker services configuration.

**Location**: `/home/dev/zapomni/docker-compose.yml`

**Services**:
```yaml
services:
  falkordb:
    image: falkordb/falkordb:latest
    ports:
      - "6381:6379"  # External:Internal
    environment:
      - FALKORDB_PASSWORD=${FALKORDB_PASSWORD:-}
    
  redis:
    image: redis:7-alpine
    ports:
      - "6380:6379"  # External:Internal
    command: redis-server --maxmemory 2gb --maxmemory-policy allkeys-lru
```

**Port Mappings**:
- FalkorDB: External `6381` → Internal `6379`
- Redis: External `6380` → Internal `6379`

---

## Configuration Examples

### Development Setup

```bash
# .env for local development
FALKORDB_HOST=localhost
FALKORDB_PORT=6381
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_LLM_MODEL=llama3.1:8b

# Disable advanced features for speed
ENABLE_HYBRID_SEARCH=false
ENABLE_KNOWLEDGE_GRAPH=false
ENABLE_CODE_INDEXING=false
ENABLE_SEMANTIC_CACHE=false

# Verbose logging
LOG_LEVEL=DEBUG
LOG_FORMAT=console
```

### Production Setup

```bash
# .env for production
FALKORDB_HOST=falkordb.internal
FALKORDB_PORT=6379
FALKORDB_PASSWORD=secure_password_here
FALKORDB_POOL_SIZE=50  # Higher for production load

OLLAMA_BASE_URL=http://ollama.internal:11434
REDIS_ENABLED=true
REDIS_HOST=redis.internal
REDIS_PORT=6379

# Enable all features
ENABLE_HYBRID_SEARCH=true
ENABLE_KNOWLEDGE_GRAPH=true
ENABLE_CODE_INDEXING=true
ENABLE_SEMANTIC_CACHE=true

# Production logging
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE=/var/log/zapomni/server.log

# SSE for web clients
ZAPOMNI_SSE_HOST=0.0.0.0
ZAPOMNI_SSE_PORT=8000
ZAPOMNI_SSE_CORS_ORIGINS=https://app.example.com
```

### High-Performance Setup

```bash
# Optimized for throughput
FALKORDB_POOL_SIZE=100
FALKORDB_POOL_MIN_SIZE=20
MAX_CONCURRENT_TASKS=16

# Large cache
REDIS_ENABLED=true
REDIS_MAX_MEMORY_MB=4096

# Faster HNSW (trade recall for speed)
HNSW_M=12
HNSW_EF_CONSTRUCTION=150
HNSW_EF_SEARCH=50

# Enable caching features
ENABLE_SEMANTIC_CACHE=true
ENABLE_HYBRID_SEARCH=false  # Faster than hybrid
```

---

## Configuration Validation

Zapomni validates configuration on startup:

```python
# Automatic validation via Pydantic
class ZapomniSettings(BaseSettings):
    falkordb_port: int = Field(ge=1, le=65535)  # Valid port range
    chunk_overlap: int = Field(ge=0, le=512)    # Overlap < chunk size
    min_similarity_threshold: float = Field(ge=0.0, le=1.0)  # 0-1 range
```

**Validation Errors**:
```
ValidationError: 
  falkordb_port: ensure this value is less than or equal to 65535
  hnsw_m: ensure this value is greater than or equal to 4
```

---

## Environment Variable Reference

Quick reference of all 48 configuration variables:

**Essential** (5):
- FALKORDB_HOST, FALKORDB_PORT, FALKORDB_PASSWORD, GRAPH_NAME, FALKORDB_CONNECTION_TIMEOUT

**Ollama** (5):
- OLLAMA_BASE_URL, OLLAMA_EMBEDDING_MODEL, OLLAMA_LLM_MODEL, OLLAMA_EMBEDDING_TIMEOUT, OLLAMA_LLM_TIMEOUT

**Connection Pool** (6):
- FALKORDB_POOL_SIZE, FALKORDB_POOL_MIN_SIZE, FALKORDB_POOL_MAX_SIZE, FALKORDB_POOL_TIMEOUT, FALKORDB_SOCKET_TIMEOUT, FALKORDB_HEALTH_CHECK_INTERVAL

**Retry** (3):
- FALKORDB_MAX_RETRIES, FALKORDB_RETRY_INITIAL_DELAY, FALKORDB_RETRY_MAX_DELAY

**Redis** (5):
- REDIS_ENABLED, REDIS_HOST, REDIS_PORT, REDIS_TTL_SECONDS, REDIS_MAX_MEMORY_MB

**Performance** (9):
- MAX_CHUNK_SIZE, CHUNK_OVERLAP, VECTOR_DIMENSIONS, HNSW_M, HNSW_EF_CONSTRUCTION, HNSW_EF_SEARCH, MAX_CONCURRENT_TASKS, SEARCH_LIMIT_DEFAULT, MIN_SIMILARITY_THRESHOLD

**Logging** (3):
- LOG_LEVEL, LOG_FORMAT, LOG_FILE

**Feature Flags** (4):
- ENABLE_HYBRID_SEARCH, ENABLE_KNOWLEDGE_GRAPH, ENABLE_CODE_INDEXING, ENABLE_SEMANTIC_CACHE

**SSE** (7):
- ZAPOMNI_SSE_HOST, ZAPOMNI_SSE_PORT, ZAPOMNI_SSE_CORS_ORIGINS, ZAPOMNI_SSE_HEARTBEAT_INTERVAL, ZAPOMNI_SSE_MAX_CONNECTION_LIFETIME, ZAPOMNI_SSE_ALLOWED_HOSTS, ZAPOMNI_SSE_DNS_REBINDING_PROTECTION

**System** (3):
- DATA_DIR, TEMP_DIR, MAX_TEXT_LENGTH

---

## Troubleshooting

### Common Configuration Issues

**Issue**: "Connection refused" to FalkorDB
```bash
# Check port mapping
docker ps | grep falkordb
# Should show: 0.0.0.0:6381->6379/tcp

# Verify .env
FALKORDB_PORT=6381  # Use external port 6381, not internal 6379
```

**Issue**: "Model not found" from Ollama
```bash
# Pull model first
ollama pull nomic-embed-text
ollama pull llama3.1:8b

# Verify Ollama running
curl http://localhost:11434/api/tags
```

**Issue**: Feature flags not working
```bash
# Feature flags are enabled by default (true)
# If disabled, check your .env file:
ENABLE_HYBRID_SEARCH=true  # Should be true

# Restart Zapomni after changing .env
```

**Issue**: Redis cache not working
```bash
# 1. Enable Redis
REDIS_ENABLED=true

# 2. Start Redis
docker-compose up -d redis

# 3. Verify connection
redis-cli -p 6380 PING  # Should return PONG
```

---

## Related Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)**: System architecture
- **[API.md](API.md)**: MCP tools reference
- **[CLI.md](CLI.md)**: Command-line tools
- **[DEVELOPMENT.md](DEVELOPMENT.md)**: Development setup

---

**Document Version**: 1.0
**Last Updated**: 2025-11-26
**Based On**: T0.4 Configuration Audit Report
