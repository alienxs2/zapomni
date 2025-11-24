# Basic Operations Guide

## Adding Memories

### Simple Memory Addition

```python
# Via MCP Tool (in Claude/Cursor)
User: Remember that Python was created by Guido van Rossum in 1991

# Claude uses the add_memory tool
Result: ✓ Memory stored with ID: 550e8400-e29b-41d4-a716-446655440000
```

### What Happens Behind the Scenes

1. **Text Processing**
   - Content is split into chunks (default: 512 tokens)
   - Chunks overlap by 50 tokens for continuity
   - Whitespace normalized

2. **Embedding Generation**
   - Each chunk is embedded using nomic-embed-text
   - 768-dimensional vectors created
   - Embeddings cached for fast retrieval

3. **Entity Extraction**
   - Named entities identified (people, places, concepts)
   - Entity relationships detected
   - Context preserved

4. **Storage**
   - Embeddings stored in FalkorDB vector index
   - Entities created as graph nodes
   - Relationships stored as edges
   - Metadata indexed for fast lookup

### Batch Operations

For large amounts of content, use batch operations:

```python
# Efficient for multiple documents
memories = [
    "Python created by Guido van Rossum in 1991",
    "JavaScript created by Brendan Eich in 1995",
    "Rust introduced by Mozilla in 2010"
]

# Add all at once (single transaction)
for memory in memories:
    add_memory(memory)
```

## Searching Memories

### Simple Semantic Search

```
User: Who created Python?

# Claude uses search_memory tool
Result: Python was created by Guido van Rossum in 1991
        (Found in memory added earlier)
```

### Search Types

#### 1. Vector Similarity Search
- Best for: Finding conceptually related content
- How it works: Converts query to embedding, finds similar vectors
- Speed: Fast (< 100ms)
- Accuracy: Good for broad concept matching

```
Query: "programming languages"
Results:
  - Python creation story
  - JavaScript history
  - Rust introduction
```

#### 2. Graph Relationship Search
- Best for: Finding connected entities
- How it works: Traverses knowledge graph relationships
- Speed: Medium (100-500ms)
- Accuracy: Excellent for specific relationships

```
Query: "Who created what?"
Results:
  - Guido van Rossum → created → Python
  - Brendan Eich → created → JavaScript
  - Mozilla → introduced → Rust
```

#### 3. Hybrid Search
- Best for: Complex queries requiring both similarity and relationships
- How it works: Combines vector and graph results
- Speed: Medium (100-500ms)
- Accuracy: Best overall

```
Query: "Tell me about Python's origins"
Results (combined):
  - Vector: [Python history, creator info, language features]
  - Graph: [Guido van Rossum connections, influenced by...]
```

### Advanced Search Operators

```
# Exact phrase search
search("\"Guido van Rossum\"")

# Entity type search
search("Python", entity_type="programming_language")

# Relationship search
search("created", relationship_type="created_by")

# Temporal search
search("Python", after_date="1990-01-01")
```

## Viewing Statistics

### Memory Statistics

```
User: How much do you remember?

Result:
  Total Memories: 125
  Total Chunks: 487
  Total Size: 2.3 MB
  Unique Entities: 156
  Relationships: 289
  Search Latency (P95): 234ms
  Cache Hit Rate: 64%
```

### Detailed Statistics

```
Memory Distribution:
  - Text: 78% (95 items)
  - Code: 15% (18 items)
  - Chat: 7% (12 items)

Entity Types:
  - Person: 34
  - Concept: 45
  - Tool: 32
  - Code: 45

Recent Searches:
  - "Python" (5 times)
  - "memory system" (3 times)
  - "embedding" (2 times)
```

## Common Patterns

### Learning Conversation

```
User: I want to teach you about distributed systems