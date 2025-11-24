# Zapomni Data Flow & Interaction Patterns

## Core Data Flow Patterns

### 1. Memory Indexing Flow

```
User Input (via MCP Tool)
    ↓
MCP Server (receives "add_to_memory" call)
    ↓
Memory Module (processes input)
    ├─ Splits content into chunks
    ├─ Extracts entities
    └─ Detects relationships
    ↓
Core Module (processes extracted data)
    ├─ Generates embeddings (Ollama)
    ├─ Creates entity records
    └─ Establishes relationships
    ↓
FalkorDB (stores everything)
    ├─ Vector embeddings
    ├─ Entity nodes
    ├─ Relationship edges
    └─ Metadata
    ↓
Response to User
    └─ Confirmation + statistics
```

### 2. Semantic Search Flow

```
User Query (via MCP Tool)
    ↓
MCP Server (receives "search_memory" call)
    ↓
Memory Module (prepares search)
    ├─ Processes query text
    ├─ Generates query embedding
    └─ Identifies search intent
    ↓
Core Module (executes search)
    ├─ Vector similarity search
    └─ Graph pattern matching
    ↓
FalkorDB (returns results)
    ├─ Vector matches (similarity)
    ├─ Graph paths (relationships)
    └─ Scored results (ranked)
    ↓
Memory Module (ranks & combines)
    ├─ Merges vector and graph results
    ├─ Re-ranks by relevance
    └─ Adds context
    ↓
Response to User
    └─ Ranked search results with explanations
```

### 3. Reasoning Flow

```
User Request (via MCP Tool)
    ↓
MCP Server (receives "get_insights" call)
    ↓
Reasoning Agent (processes request)
    ├─ Identifies relevant entities
    ├─ Traverses knowledge graph
    └─ Extracts connection patterns
    ↓
FalkorDB (queries graph)
    ├─ Finds entity relationships
    ├─ Traces multi-hop connections
    └─ Returns path information
    ↓
Reasoning Engine (generates insights)
    ├─ Analyzes relationship patterns
    ├─ Identifies clusters
    ├─ Generates explanations
    └─ Ranks insights by relevance
    ↓
Response to User
    └─ Insights + reasoning + evidence
```

## Entity Lifecycle

### Creation
```
Raw Text Input
    ↓
Entity Extraction
    (Named Entity Recognition)
    ↓
Entity Normalization
    (deduplication, canonicalization)
    ↓
Create Entity Record
    ├─ Name
    ├─ Type
    ├─ Attributes
    └─ Source reference
    ↓
FalkorDB Storage
    ├─ Node creation
    ├─ Index building
    └─ Embedding storage
```

### Updates
```
New Content Mentioning Entity
    ↓
Extract & Verify Entity Match
    ↓
Update Existing Record
    ├─ Merge attributes
    ├─ Track confidence
    └─ Update source references
    ↓
Re-embed if needed
    ↓
FalkorDB Update
    ├─ Modify node properties
    └─ Update indices
```

### Relationships
```
Entity Pair Detection
    ├─ From same document
    ├─ Extracted by LLM
    └─ Graph inference
    ↓
Relationship Creation
    ├─ Type identification
    ├─ Bidirectional edge
    └─ Weight/confidence
    ↓
FalkorDB Storage
    ├─ Edge creation
    ├─ Property storage
    └─ Index updates
```

## Search Interaction Pattern

### Vector Search
```
Query Text
    ↓
Generate Embedding
    (Ollama + nomic-embed-text)
    ↓
Vector Similarity Search
    (FalkorDB cosine similarity)
    ↓
Return Top-K Results
    ├─ With similarity scores
    └─ With original content
```

### Graph Search
```
Query Analysis
    ├─ Identify key entities
    ├─ Determine search intent
    └─ Plan graph traversal
    ↓
Graph Pattern Matching
    (FalkorDB Cypher queries)
    ↓
Relationship Traversal
    ├─ 1-hop connections
    ├─ 2-hop connections
    └─ Cluster detection
    ↓
Return Connected Entities
    ├─ With path information
    └─ With relationship details
```

### Hybrid Search
```
Execute Vector Search
    ├─ Get top-K vector matches
    └─ Store relevance scores
    ↓
Execute Graph Search
    ├─ Get relationship matches
    └─ Calculate graph scores
    ↓
Combine Results
    ├─ Merge result sets
    ├─ Weighted re-ranking
    │  (vector_weight * vector_score)
    │  + (graph_weight * graph_score)
    └─ Remove duplicates
    ↓
Return Ranked Results
    └─ With evidence from both methods
```

## Context Management Flow

### Conversation Context
```
New User Message
    ↓
Extract Context Entities
    ├─ Current topic
    ├─ Mentioned concepts
    └─ Related history
    ↓
Query Memory for Context
    ├─ Previous interactions
    ├─ Related memories
    └─ Entity history
    ↓
Build Context Window
    ├─ Recent history
    ├─ Relevant memories
    └─ Entity relationships
    ↓
Include in Response
    └─ Enrich user-facing output
```

## Error Handling & Resilience

### Database Connection
```
Operation Request
    ↓
Check Connection
    ├─ If connected → proceed
    └─ If disconnected → reconnect
    ↓
Retry Logic
    ├─ Exponential backoff
    ├─ Max 3 retries
    └─ Timeout: 30s
    ↓
Fallback
    ├─ If still failed
    └─ Return error with context
```

### Embedding Generation
```
Request Embedding
    ↓
Call Ollama
    ├─ If success → return vector
    ├─ If timeout → retry with smaller chunk
    └─ If failed → use fallback
    ↓
Fallback Options
    ├─ Use cached embeddings
    ├─ Use approximate vector
    └─ Continue without embedding
```

## Performance Optimization Strategies

### 1. Caching
```
Query Received
    ↓
Check Cache
    ├─ If hit → return cached result
    │   (TTL: 5 minutes)
    ├─ If miss → proceed to search
    └─ Store result for future
```

### 2. Batch Operations
```
Multiple Additions
    ↓
Collect into Batch
    (max 100 items or 5 seconds)
    ↓
Single Database Transaction
    ├─ All embeddings at once
    ├─ Single graph update
    └─ Atomic commit
```

### 3. Lazy Loading
```
Search Results
    ↓
Load Essential Fields
    ├─ ID, score, summary
    └─ Defer full content
    ↓
On Access
    └─ Load full details
```

## Integration Points

### With Claude CLI
```
User Query in Claude
    ↓
MCP Tool Invocation
    └─ stdio connection to Zapomni
    ↓
Zapomni Processes
    ↓
Response Streamed Back
    └─ Integrated into Claude conversation
```

### With External Tools
```
Code Analysis Request
    ↓
AST Extraction (in Zapomni)
    ├─ Parse source code
    ├─ Extract structure
    └─ Build call graphs
    ↓
Store in Knowledge Graph
    ├─ Functions as nodes
    ├─ Calls as edges
    └─ Signatures as properties
```

## State Management

### Memory State
- In-Memory Cache (recent queries, popular results)
- FalkorDB (persistent storage)
- Indices (optimized search structures)

### Session State
- Current user context
- Recent search queries
- Active entities
- Conversation history

### Lifecycle
- Initialize on startup (load indices)
- Maintain during operation (cache updates)
- Persist on shutdown (flush to disk)

## Data Consistency

### Eventual Consistency
- Vector embeddings may lag behind text content
- Graph relationships built asynchronously
- Search results improve over time

### Strong Consistency
- Entity existence
- Relationship existence
- Direct entity attribute reads

### Handling Conflicts
- Last-write-wins for attributes
- Union strategy for relationships
- Manual resolution for conflicts via re-indexing

## Monitoring & Observability

### Logged Events
- Document indexing (content length, entities found)
- Search queries (query text, result count, latency)
- Graph operations (traversal depth, nodes visited)
- Errors and retries

### Metrics
- Indexing throughput (docs/minute)
- Search latency (p50, p95, p99)
- Graph traversal depth
- Cache hit rate
- Memory usage
- Embedding generation time

### Debugging
- Request tracing (end-to-end)
- Component logs (per module)
- Query explain (why results ranked this way)
- Performance profiles (bottleneck identification)
