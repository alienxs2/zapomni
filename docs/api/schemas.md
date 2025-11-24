# API Schemas and Data Models

Complete JSON schemas and type definitions for Zapomni MCP API requests and responses.

## Request Schemas

### add_memory Request Schema

**Tool**: `add_memory`

```json
{
  "type": "object",
  "properties": {
    "text": {
      "type": "string",
      "description": "Text content to remember (1-10MB)",
      "minLength": 1,
      "maxLength": 10000000
    },
    "metadata": {
      "type": "object",
      "description": "Optional metadata for organization and filtering",
      "properties": {
        "source": {
          "type": "string",
          "description": "Source identifier (e.g., 'github:user/repo')"
        },
        "tags": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "description": "Tags for categorization"
        },
        "timestamp": {
          "type": "string",
          "format": "date-time",
          "description": "ISO 8601 timestamp"
        },
        "language": {
          "type": "string",
          "description": "Programming language if code (e.g., 'python')"
        }
      },
      "additionalProperties": true
    }
  },
  "required": ["text"],
  "additionalProperties": false
}
```

### search_memory Request Schema

**Tool**: `search_memory`

```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "Natural language search query",
      "minLength": 1,
      "maxLength": 1000
    },
    "limit": {
      "type": "integer",
      "description": "Maximum results to return (1-100)",
      "minimum": 1,
      "maximum": 100,
      "default": 10
    },
    "filters": {
      "type": "object",
      "description": "Optional metadata filters",
      "properties": {
        "tags": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "description": "Match memories with all specified tags"
        },
        "source": {
          "type": "string",
          "description": "Match specific source"
        },
        "date_from": {
          "type": "string",
          "format": "date",
          "description": "Match from date (YYYY-MM-DD)"
        },
        "date_to": {
          "type": "string",
          "format": "date",
          "description": "Match to date (YYYY-MM-DD)"
        }
      },
      "additionalProperties": false
    }
  },
  "required": ["query"],
  "additionalProperties": false
}
```

### get_stats Request Schema

**Tool**: `get_stats`

```json
{
  "type": "object",
  "properties": {},
  "required": [],
  "additionalProperties": false
}
```

No parameters required.

## Response Schemas

### Success Response Schema

**All tools**

```json
{
  "type": "object",
  "properties": {
    "content": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "type": {
            "type": "string",
            "enum": ["text", "resource", "image"]
          },
          "text": {
            "type": "string"
          }
        },
        "required": ["type", "text"]
      },
      "minItems": 1
    },
    "isError": {
      "type": "boolean",
      "enum": [false]
    }
  },
  "required": ["content", "isError"]
}
```

### Error Response Schema

**All tools**

```json
{
  "type": "object",
  "properties": {
    "content": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "type": {
            "type": "string",
            "enum": ["text"]
          },
          "text": {
            "type": "string",
            "pattern": "^Error: "
          }
        },
        "required": ["type", "text"]
      },
      "minItems": 1
    },
    "isError": {
      "type": "boolean",
      "enum": [true]
    }
  },
  "required": ["content", "isError"]
}
```

## Data Models

### Memory

Complete memory object stored in database.

```typescript
interface Memory {
  // Required
  id: string;                      // UUID v4
  text: string;                    // Original text (1 - 1MB)
  chunks: Chunk[];                 // Semantic chunks
  embeddings: number[][];          // Vector embeddings (768d)
  created_at: string;              // ISO 8601 timestamp

  // Optional
  metadata?: {
    source?: string;               // Memory source
    tags?: string[];               // Category tags
    language?: string;             // Language/programming language
    [key: string]: any;            // Custom fields
  };

  // Graph (Phase 2)
  entities?: Entity[];             // Extracted entities
  relationships?: Relationship[];  // Extracted relationships
}
```

### Chunk

Individual text chunk with embedding.

```typescript
interface Chunk {
  // Required
  text: string;                    // Chunk text (1 - 10k chars)
  index: number;                   // Position in memory (0-indexed)
  embedding: number[];             // Vector embedding (768d)

  // Optional
  start_char?: number;             // Start position in original
  end_char?: number;               // End position in original
  metadata?: {
    [key: string]: any;
  };
}
```

### SearchResult

Single search result from query.

```typescript
interface SearchResult {
  // Required
  memory_id: string;               // UUID of matching memory
  text: string;                    // Chunk text (up to 200 chars)
  similarity_score: number;        // Cosine similarity (0-1)

  // Optional
  tags?: string[];                 // Tags from memory
  source?: string;                 // Source from memory
  timestamp?: string;              // ISO 8601 when created
  chunk_index?: number;            // Which chunk in memory

  // Legacy
  chunk_id?: string;               // For backward compatibility
  relevance_score?: number;        // Alias for similarity_score
}
```

### SearchResultItem

Search result as used in search_memory response.

```python
@dataclass
class SearchResultItem:
    memory_id: str                          # UUID
    text: str                               # Chunk text preview
    similarity_score: float                 # 0.0 to 1.0
    tags: List[str]                        # Memory tags
    source: str                            # Memory source
    timestamp: datetime                    # Creation time
    highlight: Optional[str] = None        # Highlighted excerpt
```

### Entity (Phase 2)

Knowledge graph entity node.

```typescript
interface Entity {
  // Required
  name: string;                    // Entity name (1-1000 chars)
  type: string;                    // Entity type (person, org, concept, etc.)

  // Optional
  description?: string;            // Entity description
  confidence?: number;             // Extraction confidence (0-1)
  properties?: {
    [key: string]: any;
  };
}
```

### Relationship (Phase 2)

Knowledge graph relationship edge.

```typescript
interface Relationship {
  // Required
  from_entity_id: string;         // Source entity UUID
  to_entity_id: string;           // Target entity UUID
  relationship_type: string;       // Relationship type (created_by, part_of, etc.)

  // Optional
  strength?: number;               // Relationship strength (0-1)
  confidence?: number;             // Extraction confidence (0-1)
  context?: string;                // Relationship context
}
```

## TypeScript Type Definitions

```typescript
// Basic types
type UUID = string & { readonly __brand: 'UUID' };
type EmbeddingVector = number[] & { readonly __brand: 'EmbeddingVector' };

// Metadata type
type Metadata = {
  source?: string;
  tags?: string[];
  timestamp?: string;
  language?: string;
  [key: string]: any;
};

// Filter type
type SearchFilter = {
  tags?: string[];
  source?: string;
  date_from?: string;  // YYYY-MM-DD
  date_to?: string;    // YYYY-MM-DD
};

// Tool response
type ToolResponse<T> = {
  content: Array<{ type: 'text'; text: string }>;
  isError: false;
} | {
  content: Array<{ type: 'text'; text: string }>;
  isError: true;
};

// Tool execution result
type ToolResult =
  | { success: true; data: any }
  | { success: false; error: string; code: string };
```

## Python Type Definitions

```python
from typing import TypeVar, Generic, Optional, List, Dict, Any
from datetime import datetime
from dataclasses import dataclass
from pydantic import BaseModel, Field

# UUID type
UUID = str  # v4 format

# Embedding vector
EmbeddingVector = List[float]  # 768 dimensions

# Metadata type
Metadata = Dict[str, Any]

# Filter type
SearchFilter = {
    "tags": Optional[List[str]],
    "source": Optional[str],
    "date_from": Optional[str],  # YYYY-MM-DD
    "date_to": Optional[str],    # YYYY-MM-DD
}

@dataclass
class Memory:
    id: UUID
    text: str
    chunks: List['Chunk']
    embeddings: List[EmbeddingVector]
    created_at: datetime
    metadata: Optional[Metadata] = None

@dataclass
class Chunk:
    text: str
    index: int
    embedding: EmbeddingVector
    start_char: Optional[int] = None
    end_char: Optional[int] = None

@dataclass
class SearchResult:
    memory_id: UUID
    text: str
    similarity_score: float
    tags: Optional[List[str]] = None
    source: Optional[str] = None
    timestamp: Optional[datetime] = None
```

## Example Requests

### add_memory - Minimal

```json
{
  "text": "Python was created by Guido van Rossum"
}
```

### add_memory - Complete

```json
{
  "text": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
  "metadata": {
    "source": "tutorial",
    "tags": ["python", "algorithms"],
    "language": "python",
    "timestamp": "2025-11-24T10:30:00Z",
    "difficulty": "intermediate",
    "author": "John Doe"
  }
}
```

### search_memory - Minimal

```json
{
  "query": "Python programming"
}
```

### search_memory - Complete

```json
{
  "query": "How do decorators work?",
  "limit": 5,
  "filters": {
    "tags": ["python", "patterns"],
    "source": "documentation",
    "date_from": "2025-01-01",
    "date_to": "2025-12-31"
  }
}
```

### get_stats - Request

```json
{
}
```

## Example Responses

### add_memory - Success

```json
{
  "content": [
    {
      "type": "text",
      "text": "Memory stored successfully.\nID: 550e8400-e29b-41d4-a716-446655440000\nPreview: Python was created by Guido van Rossum in 1991..."
    }
  ],
  "isError": false
}
```

### add_memory - Error

```json
{
  "content": [
    {
      "type": "text",
      "text": "Error: text cannot be empty or contain only whitespace"
    }
  ],
  "isError": true
}
```

### search_memory - Success

```json
{
  "content": [
    {
      "type": "text",
      "text": "Found 2 results:\n\n1. [Score: 0.94]\nPython is a high-level programming language...\n\n2. [Score: 0.87]\nGuido van Rossum created Python in 1991..."
    }
  ],
  "isError": false
}
```

### search_memory - No Results

```json
{
  "content": [
    {
      "type": "text",
      "text": "No results found matching your query."
    }
  ],
  "isError": false
}
```

### search_memory - Error

```json
{
  "content": [
    {
      "type": "text",
      "text": "Error: Invalid input - query: query cannot be empty or contain only whitespace"
    }
  ],
  "isError": true
}
```

### get_stats - Success

```json
{
  "content": [
    {
      "type": "text",
      "text": "Memory System Statistics:\nTotal Memories: 42\nTotal Chunks: 156\nDatabase Size: 12.45 MB\nAverage Chunks per Memory: 3.7\nCache Hit Rate: 65.3%\nAvg Query Latency: 245 ms"
    }
  ],
  "isError": false
}
```

### get_stats - Error

```json
{
  "content": [
    {
      "type": "text",
      "text": "Error: Failed to retrieve statistics - Connection refused"
    }
  ],
  "isError": true
}
```

## Schema Validation

### add_memory Input Validation

```python
from pydantic import BaseModel, StringConstraints, ConfigDict
from typing_extensions import Annotated

class AddMemoryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: Annotated[str, StringConstraints(min_length=1, max_length=10_000_000)]
    metadata: Dict[str, Any] = {}

# Validation examples
valid = AddMemoryRequest(text="Hello world")  # ✓ Valid
valid = AddMemoryRequest(text="Test", metadata={"tags": ["a"]})  # ✓ Valid

try:
    invalid = AddMemoryRequest(text="")  # ✗ ValidationError
except ValidationError as e:
    print(e.errors())  # [{"type": "string_too_short", ...}]

try:
    invalid = AddMemoryRequest(text="x" * 10_000_001)  # ✗ Too long
except ValidationError as e:
    print(e.errors())  # [{"type": "string_too_long", ...}]
```

### search_memory Input Validation

```python
class SearchMemoryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: Annotated[str, StringConstraints(min_length=1, max_length=1000)]
    limit: int = 10
    filters: Optional[Dict[str, Any]] = None

# Validation examples
valid = SearchMemoryRequest(query="Python")  # ✓ Valid
valid = SearchMemoryRequest(query="Python", limit=5)  # ✓ Valid
valid = SearchMemoryRequest(
    query="Python",
    filters={"tags": ["programming"]}
)  # ✓ Valid

try:
    invalid = SearchMemoryRequest(query="")  # ✗ Too short
except ValidationError as e:
    print(e.errors())

try:
    invalid = SearchMemoryRequest(query="test", limit=200)  # ✗ Out of range
except ValidationError as e:
    print(e.errors())
```

## OpenAPI/Swagger Definition

### Partial OpenAPI 3.0 Definition

```yaml
openapi: 3.0.0
info:
  title: Zapomni MCP API
  version: 1.0.0
  description: Local-first memory MCP server

servers:
  - url: http://localhost:5000
    description: Local MCP server (via stdio transport)

paths:
  /mcp/tools/call:
    post:
      summary: Call MCP tool
      requestBody:
        required: true
        content:
          application/json:
            schema:
              oneOf:
                - $ref: '#/components/schemas/AddMemoryRequest'
                - $ref: '#/components/schemas/SearchMemoryRequest'
                - $ref: '#/components/schemas/GetStatsRequest'
      responses:
        '200':
          description: Successful response
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ToolResponse'
        '400':
          description: Invalid request
        '500':
          description: Server error

components:
  schemas:
    AddMemoryRequest:
      type: object
      required:
        - text
      properties:
        text:
          type: string
          minLength: 1
          maxLength: 10000000
        metadata:
          type: object

    SearchMemoryRequest:
      type: object
      required:
        - query
      properties:
        query:
          type: string
          minLength: 1
          maxLength: 1000
        limit:
          type: integer
          minimum: 1
          maximum: 100
          default: 10
        filters:
          type: object

    GetStatsRequest:
      type: object

    ToolResponse:
      oneOf:
        - $ref: '#/components/schemas/SuccessResponse'
        - $ref: '#/components/schemas/ErrorResponse'

    SuccessResponse:
      type: object
      required:
        - content
        - isError
      properties:
        content:
          type: array
          items:
            $ref: '#/components/schemas/ContentBlock'
        isError:
          const: false

    ErrorResponse:
      type: object
      required:
        - content
        - isError
      properties:
        content:
          type: array
          items:
            $ref: '#/components/schemas/ContentBlock'
        isError:
          const: true

    ContentBlock:
      type: object
      required:
        - type
        - text
      properties:
        type:
          enum: [text, resource, image]
        text:
          type: string
```

## Validation Rules Summary

| Field | Type | Min | Max | Format | Required |
|-------|------|-----|-----|--------|----------|
| **add_memory.text** | string | 1 | 10MB | UTF-8 | Yes |
| **add_memory.metadata** | object | - | - | JSON | No |
| **search_memory.query** | string | 1 | 1000 | UTF-8 | Yes |
| **search_memory.limit** | integer | 1 | 100 | - | No |
| **search_memory.filters** | object | - | - | JSON | No |

## Size Limits

| Entity | Limit | Notes |
|--------|-------|-------|
| **Memory text** | 10 MB | Enforced in schema |
| **Query text** | 1 KB | Enforced in schema |
| **Metadata** | Unlimited | Serialized with memory |
| **Embedding vector** | 768 dimensions | Fixed by model |
| **Search results** | 100 max | Enforced in limit parameter |
| **Tag list** | Unlimited | But included in metadata |
| **Source string** | Unlimited | But included in metadata |

## Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| Schema validation | < 1ms | Local validation |
| Request parsing | < 5ms | JSON parsing |
| Response serialization | < 5ms | JSON encoding |

## Related Documentation

- **[add_memory Tool](./tools/add_memory.md)** - Parameter details and examples
- **[search_memory Tool](./tools/search_memory.md)** - Parameter details and examples
- **[get_stats Tool](./tools/get_stats.md)** - Parameter details and examples
- **[Error Reference](./errors.md)** - Error codes and handling

---

**Last Updated**: 2025-11-24
**Version**: 1.0
**Status**: Complete
