# SearchMemoryTool.execute() - Function Specification

**Level:** 3 (Function)
**Component:** SearchMemoryTool
**Module:** zapomni_mcp
**Author:** Goncharenko Anton aka alienxs2
**Status:** Draft
**Version:** 1.0
**Created:** 2025-11-23

**Parent Spec:** [SearchMemoryTool Component](../level2/search_memory_tool_component.md)

---

## Function Signature

```python
async def execute(
    self,
    arguments: dict[str, Any]
) -> dict[str, Any]:
    """
    Execute search_memory tool with provided arguments.

    This is the main entry point called by the MCP server when a client
    invokes the search_memory tool. It orchestrates the complete workflow:
    validate → delegate → format → return.

    The function implements a pure delegation pattern: all search intelligence
    (embeddings, ranking, filtering) is handled by zapomni_core. This function
    focuses solely on MCP protocol adaptation.

    Args:
        arguments: Dictionary from MCP client with keys:
            - query (str, required): Natural language search query
            - limit (int, optional): Maximum results to return (default: 10, max: 100)
            - filters (dict, optional): Metadata filters to narrow results

    Returns:
        MCP-formatted response dictionary with structure:
        {
            "content": [
                {
                    "type": "text",
                    "text": "Found N results:\n1. [Score: X.XX] text..."
                }
            ],
            "isError": False  # or True if error occurred
        }

    Raises:
        No exceptions raised - all errors caught and returned as MCP error responses.
        Internally catches:
            - ValidationError: Input validation failures
            - SearchError: Core search operation failures
            - CoreError: General core processing errors
            - Exception: Unexpected errors

    Performance Target:
        - Validation: < 5ms
        - Core delegation: < 200ms (P50), < 500ms (P95)
        - Formatting: < 10ms
        - Total: < 220ms (P50), < 520ms (P95)

    Example:
        ```python
        tool = SearchMemoryTool(core=core_engine)

        # Basic search
        result = await tool.execute({
            "query": "What is Python?"
        })

        # Advanced search with filters
        result = await tool.execute({
            "query": "machine learning tutorials",
            "limit": 20,
            "filters": {
                "tags": ["ML", "tutorial"],
                "date_from": "2025-01-01"
            }
        })

        if result["isError"]:
            print(f"Error: {result['content'][0]['text']}")
        else:
            print(result["content"][0]["text"])
            # Output:
            # Found 3 results:
            #
            # 1. [Score: 0.92] Python is a programming language...
            # 2. [Score: 0.88] Python was created by Guido...
            # 3. [Score: 0.85] Python is widely used for...
        ```
    """
```

---

## Purpose & Context

### What It Does

The `execute()` method is the **sole entry point** for the search_memory MCP tool. It performs a 4-step workflow:

1. **Validate** - Parse and validate arguments using Pydantic (SearchRequest model)
2. **Delegate** - Call core.search_memory() with validated request
3. **Format** - Transform SearchResponse into human-readable MCP text
4. **Return** - Return MCP-compliant response (never raises exceptions)

The function is a **thin adapter** - all search logic (vector similarity, ranking, filtering) is delegated to the core layer. This maintains clean separation of concerns:
- MCP layer: Protocol adaptation
- Core layer: Business logic and intelligence

### Why It Exists

**Required by MCP Protocol:**
All MCP tools must implement an `execute(arguments: dict) -> dict` method. This is the contract between the MCP server and tool implementations.

**Separation of Concerns:**
By isolating MCP protocol handling from search logic, we achieve:
- Independent testing of protocol vs. business logic
- Easy swapping of core implementations
- Clear error boundaries and handling

### When To Use

**Automatic Invocation:**
This function is called automatically by the MCP server when:
1. Claude Desktop sends a `tools/call` request with name="search_memory"
2. MCP server routes the request to SearchMemoryTool
3. Server calls `tool.execute(arguments)`

**Never Called Directly:**
User code should NOT call this method directly. It's part of the MCP infrastructure.

### When NOT To Use

**Direct Core Access:**
If you need search functionality within your Python code (not via MCP), use the core layer directly:

```python
# Don't do this
from zapomni_mcp.tools.search_memory import SearchMemoryTool
tool = SearchMemoryTool(core)
result = await tool.execute({"query": "..."})  # Wrong approach

# Do this instead
from zapomni_core import MemoryProcessor
core = MemoryProcessor(db=db)
response = await core.search_memory(SearchRequest(query="..."))  # Direct
```

---

## Parameters (Detailed)

### arguments: dict[str, Any]

**Type:** `dict[str, Any]`

**Purpose:**
Contains all tool arguments passed from the MCP client. The dictionary structure is defined by the tool's `input_schema` (JSON Schema) but arrives as a raw dict that must be validated.

**Structure:**
```python
{
    "query": str,              # Required
    "limit": int,              # Optional (default: 10)
    "filters": {               # Optional
        "tags": list[str],     # Optional
        "source": str,         # Optional
        "date_from": str,      # Optional (YYYY-MM-DD)
        "date_to": str         # Optional (YYYY-MM-DD)
    }
}
```

**Validation Flow:**
```
Raw arguments dict
    ↓
_validate_input()
    ↓
Pydantic SearchRequest model
    ↓
Validated, immutable request object
```

---

#### arguments["query"]: str (Required)

**Type:** `str`

**Purpose:**
Natural language search query that will be embedded and compared against stored memories using vector similarity.

**Constraints:**
- Must be present (required field)
- Must be string type
- Minimum length: 1 character (after stripping whitespace)
- Maximum length: 1,000 characters
- Must be valid UTF-8 text
- Cannot be whitespace-only

**Validation Rules:**
```python
# Pydantic model (in SearchRequest)
query: str = Field(
    ...,  # Required
    min_length=1,
    max_length=1000,
    description="Natural language search query"
)

# Additional runtime validation
@field_validator('query')
def validate_query_not_whitespace(cls, v):
    if not v.strip():
        raise ValueError("query cannot be whitespace-only")
    return v.strip()
```

**Examples:**

**Valid:**
- `"What is Python?"` - Simple question
- `"information about machine learning algorithms"` - Descriptive query
- `"my notes on Docker containers from last week"` - Contextual query
- `"Python" * 100` - Long but under 1000 chars
- `"тест"` - Unicode/non-English (UTF-8)

**Invalid:**
- `""` → ValidationError: "query: ensure this value has at least 1 character"
- `"   "` → ValidationError: "query: cannot be whitespace-only"
- `"x" * 1001` → ValidationError: "query: ensure this value has at most 1000 characters"
- `123` → ValidationError: "query: str type expected"
- `None` → ValidationError: "query: field required"

**Security Considerations:**
- Query text is NOT sanitized for SQL injection (FalkorDB uses parameterized queries)
- Query text is NOT sanitized for XSS (returned as plain text, not HTML)
- Query text IS logged (first 50 chars only to avoid log flooding)
- No PII validation required (user controls their own data)

---

#### arguments["limit"]: int (Optional)

**Type:** `int`

**Purpose:**
Maximum number of search results to return. Controls both:
1. Database query limit (reduces DB load)
2. Response size (reduces network/parsing overhead)

**Default:** 10

**Constraints:**
- Must be integer type (if provided)
- Minimum value: 1
- Maximum value: 100
- Default: 10 (when not provided or None)

**Validation Rules:**
```python
# Pydantic model
limit: int = Field(
    default=10,
    ge=1,  # Greater than or equal to 1
    le=100,  # Less than or equal to 100
    description="Maximum number of results"
)
```

**Rationale for Max Limit (100):**
- MCP response size: ~20KB for 100 results (manageable)
- User attention span: >50 results rarely useful
- Core performance: Vector search optimized for top-100
- Claude context window: 100 results fit comfortably

**Examples:**

**Valid:**
- `1` - Minimum, returns single best match
- `10` - Default, good for most queries
- `50` - Large result set
- `100` - Maximum allowed
- `None` or omitted → defaults to 10

**Invalid:**
- `0` → ValidationError: "limit: ensure this value is greater than or equal to 1"
- `-5` → ValidationError: "limit: ensure this value is greater than or equal to 1"
- `101` → ValidationError: "limit: ensure this value is less than or equal to 100"
- `200` → ValidationError: "limit: ensure this value is less than or equal to 100"
- `"10"` → ValidationError: "limit: value is not a valid integer"
- `10.5` → ValidationError: "limit: value is not a valid integer"

**Performance Impact:**
```
limit=10:  ~100ms (P50), ~200ms (P95)
limit=50:  ~150ms (P50), ~300ms (P95)
limit=100: ~200ms (P50), ~500ms (P95)
```

---

#### arguments["filters"]: dict[str, Any] (Optional)

**Type:** `Optional[dict[str, Any]]`

**Purpose:**
Optional metadata filters to narrow search results. Filters are applied AFTER vector similarity ranking (hybrid filtering approach).

**Default:** `None` (no filtering, return all matched results)

**Structure (when provided):**
```python
{
    "tags": list[str],      # Only memories with ALL these tags
    "source": str,          # Only memories from this source
    "date_from": str,       # Only memories created >= this date (YYYY-MM-DD)
    "date_to": str          # Only memories created <= this date (YYYY-MM-DD)
}
```

**Validation Rules:**
```python
# Pydantic model
class SearchFilters(BaseModel):
    tags: Optional[list[str]] = Field(
        default=None,
        description="Filter by tags (AND logic)"
    )
    source: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Filter by source identifier"
    )
    date_from: Optional[date] = Field(
        default=None,
        description="Filter memories created on or after this date"
    )
    date_to: Optional[date] = Field(
        default=None,
        description="Filter memories created on or before this date"
    )

    @model_validator(mode='after')
    def validate_date_range(self):
        if self.date_from and self.date_to:
            if self.date_from > self.date_to:
                raise ValueError("date_from must be <= date_to")
        return self
```

**Additional Constraints:**
- No extra keys allowed (Pydantic raises error for unknown fields)
- All fields optional (filters can be partially specified)
- Empty dict `{}` treated as "no filters" (same as None)

**Examples:**

**Valid:**
```python
# Single filter
{"tags": ["python"]}

# Multiple filters
{
    "tags": ["python", "tutorial"],
    "source": "documentation",
    "date_from": "2025-01-01"
}

# Date range
{
    "date_from": "2025-01-01",
    "date_to": "2025-12-31"
}

# Empty (no filtering)
None
{}
```

**Invalid:**
```python
# Unknown key
{"unknown_key": "value"}
→ ValidationError: "filters: extra fields not permitted"

# Wrong type for tags
{"tags": "python"}  # Should be list
→ ValidationError: "filters.tags: value is not a valid list"

# Invalid date format
{"date_from": "2025/01/01"}  # Should be YYYY-MM-DD
→ ValidationError: "filters.date_from: invalid date format, expected YYYY-MM-DD"

# Invalid date range
{"date_from": "2025-12-31", "date_to": "2025-01-01"}
→ ValidationError: "filters: date_from must be <= date_to"

# Tags not strings
{"tags": [123, 456]}
→ ValidationError: "filters.tags: value is not a valid string"
```

**Filter Semantics:**

**Tags (AND Logic):**
```python
{"tags": ["python", "tutorial"]}
→ Returns ONLY memories that have BOTH tags
→ If memory has ["python"] only → excluded
→ If memory has ["python", "tutorial", "advanced"] → included
```

**Source (Exact Match):**
```python
{"source": "documentation"}
→ Returns ONLY memories where metadata.source == "documentation"
→ Case-sensitive match
→ No partial matching
```

**Date Range (Inclusive):**
```python
{"date_from": "2025-01-01", "date_to": "2025-01-31"}
→ Returns memories with created_at >= 2025-01-01 00:00:00
→ AND created_at <= 2025-01-31 23:59:59
→ Inclusive on both ends
```

**Filter Application Order:**
```
1. Vector similarity search (returns top N candidates)
2. Apply filters (removes non-matching candidates)
3. Return remaining results (may be fewer than limit)
```

**Empty Results After Filtering:**
If filters exclude all candidates:
```python
# Vector search found 10 results
# But filters excluded all of them
→ Response: "No results found matching your query."
→ count=0, results=[]
```

---

## Return Value

### Type: dict[str, Any]

**Structure:** MCP-compliant response format

**Success Response:**
```python
{
    "content": [
        {
            "type": "text",
            "text": "Found N results:\n\n1. [Score: 0.XX] text...\n\n2. ..."
        }
    ],
    "isError": False
}
```

**Error Response:**
```python
{
    "content": [
        {
            "type": "text",
            "text": "Error: <user-friendly error message>"
        }
    ],
    "isError": True
}
```

**Response Fields:**

#### content: list[dict[str, str]]

**Type:** `list` of content blocks (MCP standard)

**Structure:**
```python
[
    {
        "type": "text",  # Always "text" for this tool
        "text": str      # The actual response message
    }
]
```

**Always contains exactly 1 content block** (search results are consolidated into single text block)

#### isError: bool

**Type:** `bool`

**Values:**
- `False` - Operation succeeded (results found or no results, but no error)
- `True` - Operation failed (validation error, search error, etc.)

**Important:** Even when `count=0` (no results found), `isError=False` because "no results" is a valid outcome, not an error.

---

### Response Text Format (Success)

#### Case 1: Results Found

**Format:**
```
Found {count} results:

1. [Score: {score:.2f}] [Tags: tag1, tag2]
{text_preview}

2. [Score: {score:.2f}]
{text_preview}

3. ...
```

**Example:**
```
Found 3 results:

1. [Score: 0.92] [Tags: python, programming]
Python is a high-level, interpreted programming language known for its simplicity and readability...

2. [Score: 0.88] [Tags: python, history]
Python was created by Guido van Rossum and first released in 1991. The language emphasizes code...

3. [Score: 0.85] [Tags: python, web]
Python is widely used for web development, data science, automation, and machine learning. Popular...
```

**Formatting Rules:**
- Results numbered starting from 1
- Similarity score formatted to 2 decimal places
- Tags shown if present in metadata (comma-separated)
- Text preview: first 200 characters
- If text > 200 chars: append "..."
- Blank line between results

**Text Truncation:**
```python
text_preview = result.text[:200]
if len(result.text) > 200:
    text_preview += "..."
```

---

#### Case 2: No Results Found

**Format:**
```
No results found matching your query.
```

**When This Occurs:**
- Vector search returned no matches (score too low)
- Filters excluded all candidates
- Database empty (no memories stored yet)

**Example Response:**
```python
{
    "content": [
        {
            "type": "text",
            "text": "No results found matching your query."
        }
    ],
    "isError": False  # NOT an error, just no matches
}
```

---

### Response Text Format (Error)

**Format:**
```
Error: {error_category}: {specific_message}
```

**Error Categories:**

#### Validation Errors

**Format:**
```
Error: Invalid input - {field}: {validation_message}
```

**Examples:**
```
Error: Invalid input - query: ensure this value has at least 1 character
Error: Invalid input - limit: ensure this value is less than or equal to 100
Error: Invalid input - filters.tags: value is not a valid list
Error: Invalid input - filters.date_from: invalid date format, expected YYYY-MM-DD
```

**Multiple Validation Errors:**
```
Error: Invalid input - query: field required; limit: ensure this value is greater than or equal to 1
```

---

#### Search Errors

**Format:**
```
Error: Search failed: {error_message}
```

**Examples:**
```
Error: Search failed: Vector index not found
Error: Search failed: Embedding generation failed
Error: Search failed: Database connection timeout
```

---

#### Core Errors

**Format:**
```
Error: Processing error: {error_message}
```

**Examples:**
```
Error: Processing error: Memory database not initialized
Error: Processing error: Insufficient resources
```

---

#### Unexpected Errors

**Format:**
```
Error: An unexpected error occurred. Please check logs for details.
```

**When Used:**
- Any exception not caught by specific handlers
- Protects against leaking sensitive information
- Full traceback logged to stderr (not returned to user)

**Security Note:**
Never include in error messages:
- File paths
- Database credentials
- Internal variable names
- Stack traces
- System information

---

## Exceptions

### No Direct Exceptions Raised

**Important:** This function **never raises exceptions** to the caller. All errors are caught internally and returned as MCP error responses.

**Rationale:**
- MCP protocol expects dict responses, not exceptions
- MCP JSON-RPC layer handles errors differently
- Tool should be resilient and always return valid response

---

### Internal Exception Handling

The function catches and formats these exception types:

#### ValidationError (from Pydantic)

**When Caught:**
```python
try:
    request = SearchRequest(**arguments)
except ValidationError as e:
    # Caught here
```

**Source:**
- Pydantic validation during SearchRequest instantiation
- Field-level validators (e.g., query length, limit range)
- Model-level validators (e.g., date range validation)

**Error Structure:**
```python
# Pydantic ValidationError provides:
e.errors() → [
    {
        "loc": ("query",),  # Field path
        "msg": "ensure this value has at least 1 character",
        "type": "value_error.any_str.min_length"
    }
]
```

**Response Generation:**
```python
# Extract field-level errors
error_msgs = []
for err in e.errors():
    field = ".".join(str(loc) for loc in err["loc"])
    message = err["msg"]
    error_msgs.append(f"{field}: {message}")

error_text = f"Invalid input - {'; '.join(error_msgs)}"

return {
    "content": [{"type": "text", "text": f"Error: {error_text}"}],
    "isError": True
}
```

**Logging:**
```python
self.logger.warning(
    "validation_error",
    errors=e.errors(),
    arguments=arguments
)
```

---

#### SearchError (from zapomni_core.exceptions)

**When Caught:**
```python
try:
    response = await self.core.search_memory(request)
except SearchError as e:
    # Caught here
```

**Source:**
- Core layer search failures
- Vector index issues
- Embedding generation failures
- Query processing errors

**Common Scenarios:**
```python
# Vector index not built
SearchError("Vector index not found - run add_memory first")

# Embedding service down
SearchError("Embedding generation failed: Ollama connection refused")

# Invalid vector dimensions
SearchError("Vector dimension mismatch: expected 384, got 768")
```

**Response Generation:**
```python
error_text = f"Search failed: {str(e)}"

return {
    "content": [{"type": "text", "text": f"Error: {error_text}"}],
    "isError": True
}
```

**Logging:**
```python
self.logger.error(
    "search_error",
    error=str(e),
    query=request.query[:50],
    limit=request.limit
)
```

---

#### CoreError (from zapomni_core.exceptions)

**When Caught:**
```python
try:
    response = await self.core.search_memory(request)
except CoreError as e:
    # Caught here
```

**Source:**
- General core processing errors
- Database connection failures
- Resource unavailability
- Configuration errors

**Common Scenarios:**
```python
# Database connection
CoreError("FalkorDB connection failed: connection refused")

# Resource limits
CoreError("Memory limit exceeded")

# Initialization
CoreError("Core engine not initialized")
```

**Response Generation:**
```python
error_text = f"Processing error: {str(e)}"

return {
    "content": [{"type": "text", "text": f"Error: {error_text}"}],
    "isError": True
}
```

**Logging:**
```python
self.logger.error(
    "core_error",
    error=str(e),
    error_type=type(e).__name__
)
```

---

#### Exception (Generic)

**When Caught:**
```python
try:
    # ... all operations ...
except Exception as e:
    # Caught here (last resort)
```

**Source:**
- Unexpected errors
- Bugs in code
- System-level failures
- Unhandled edge cases

**Response Generation:**
```python
error_text = "An unexpected error occurred. Please check logs for details."

return {
    "content": [{"type": "text", "text": f"Error: {error_text}"}],
    "isError": True
}
```

**Logging:**
```python
self.logger.exception(
    "unexpected_error",
    error=str(e),
    error_type=type(e).__name__,
    arguments=arguments
    # Full traceback logged automatically by .exception()
)
```

**Security:**
Generic message prevents leaking:
- Internal paths
- Variable names
- Stack traces
- System details

User must check logs for details (logs only accessible to system admin, not MCP client).

---

## Algorithm (Detailed Pseudocode)

```
FUNCTION execute(arguments: dict[str, Any]) -> dict[str, Any]:

    # === STEP 0: Initialize Logging ===
    LOG DEBUG "search_memory_execute_start"
        WITH arguments=<redacted_if_sensitive>

    TRY:
        # === STEP 1: Validate Input ===
        # Duration: ~5ms
        # Failure rate: ~5% (user input errors)

        START_TIMER validation_timer

        TRY:
            request = SearchRequest(**arguments)
            # Pydantic validates:
            # - query: non-empty, max 1000 chars
            # - limit: 1-100 range, default 10
            # - filters: valid structure if provided

        CATCH ValidationError as e:
            # Validation failed - user input error
            RAISE ValidationError(e)  # Re-raise to outer handler

        STOP_TIMER validation_timer

        LOG INFO "search_request_validated"
            WITH query_length=len(request.query)
            WITH limit=request.limit
            WITH has_filters=(request.filters is not None)
            WITH validation_time_ms=validation_timer.elapsed

        # === STEP 2: Delegate to Core Engine ===
        # Duration: ~200ms (P50), ~500ms (P95)
        # Failure rate: ~1% (system errors)

        START_TIMER search_timer

        TRY:
            response = AWAIT self.core.search_memory(request)
            # Core performs:
            # 1. Embed query using Ollama
            # 2. Vector similarity search in FalkorDB
            # 3. Apply metadata filters
            # 4. Rank and return top results

        CATCH SearchError as e:
            # Search operation failed
            RAISE SearchError(e)  # Re-raise to outer handler

        CATCH CoreError as e:
            # General core processing error
            RAISE CoreError(e)  # Re-raise to outer handler

        STOP_TIMER search_timer

        LOG INFO "search_completed"
            WITH result_count=response.count
            WITH query_preview=request.query[:50]
            WITH search_time_ms=search_timer.elapsed
            WITH has_results=(response.count > 0)

        # === STEP 3: Format Response ===
        # Duration: ~10ms
        # Failure rate: ~0% (pure formatting, no external deps)

        START_TIMER format_timer

        formatted_response = _format_response(response)
        # Builds human-readable text from SearchResponse
        # Includes:
        # - Result count
        # - Numbered list
        # - Similarity scores
        # - Text previews (200 chars)
        # - Tags if present

        STOP_TIMER format_timer

        LOG DEBUG "response_formatted"
            WITH format_time_ms=format_timer.elapsed
            WITH response_length=len(formatted_response["content"][0]["text"])

        # === STEP 4: Return Success ===

        RETURN formatted_response
        # Structure:
        # {
        #     "content": [{"type": "text", "text": "Found N results..."}],
        #     "isError": False
        # }

    # === ERROR HANDLING ===

    CATCH ValidationError as e:
        # User input validation failed

        LOG WARNING "validation_error"
            WITH errors=e.errors()
            WITH arguments=arguments

        error_response = _format_error(e)
        # Generates user-friendly message:
        # "Error: Invalid input - query: field required"

        RETURN error_response
        # Structure:
        # {
        #     "content": [{"type": "text", "text": "Error: ..."}],
        #     "isError": True
        # }

    CATCH SearchError as e:
        # Search operation failed (core layer)

        LOG ERROR "search_error"
            WITH error=str(e)
            WITH query=arguments.get("query", "")[:50]

        error_response = _format_error(e)
        # Generates:
        # "Error: Search failed: <specific message>"

        RETURN error_response

    CATCH CoreError as e:
        # General core processing error

        LOG ERROR "core_error"
            WITH error=str(e)
            WITH error_type=type(e).__name__

        error_response = _format_error(e)
        # Generates:
        # "Error: Processing error: <specific message>"

        RETURN error_response

    CATCH Exception as e:
        # Unexpected error - anything not caught above

        LOG EXCEPTION "unexpected_error"
            WITH error=str(e)
            WITH error_type=type(e).__name__
            WITH arguments=arguments
            # Full traceback logged

        error_response = _format_error(e)
        # Generates generic message:
        # "Error: An unexpected error occurred. Please check logs for details."

        RETURN error_response

END FUNCTION


# === HELPER FUNCTION: _format_response ===

FUNCTION _format_response(response: SearchResponse) -> dict[str, Any]:

    # Case 1: No results
    IF response.count == 0:
        RETURN {
            "content": [
                {
                    "type": "text",
                    "text": "No results found matching your query."
                }
            ],
            "isError": False
        }

    # Case 2: Results found
    lines = ["Found " + str(response.count) + " results:\n"]

    FOR i, result IN enumerate(response.results, start=1):
        # Format score
        score_str = f"{result.similarity_score:.2f}"

        # Format text preview (first 200 chars)
        text_preview = result.text[:200]
        IF len(result.text) > 200:
            text_preview = text_preview + "..."

        # Format tags (if present)
        tags_str = ""
        IF "tags" IN result.metadata AND result.metadata["tags"]:
            tag_list = ", ".join(result.metadata["tags"])
            tags_str = f" [Tags: {tag_list}]"

        # Build result line
        result_line = f"\n{i}. [Score: {score_str}]{tags_str}\n{text_preview}\n"
        lines.append(result_line)

    # Combine all lines
    formatted_text = "".join(lines)

    RETURN {
        "content": [
            {
                "type": "text",
                "text": formatted_text
            }
        ],
        "isError": False
    }

END FUNCTION


# === HELPER FUNCTION: _format_error ===

FUNCTION _format_error(error: Exception) -> dict[str, Any]:

    # Determine error message based on type

    IF isinstance(error, ValidationError):
        # Extract field-level errors from Pydantic
        error_msgs = []
        FOR err IN error.errors():
            field_path = ".".join(str(loc) FOR loc IN err["loc"])
            message = err["msg"]
            error_msgs.append(f"{field_path}: {message}")

        error_text = "Invalid input - " + "; ".join(error_msgs)

    ELSE IF isinstance(error, SearchError):
        error_text = f"Search failed: {str(error)}"

    ELSE IF isinstance(error, CoreError):
        error_text = f"Processing error: {str(error)}"

    ELSE:
        # Generic/unexpected error
        error_text = "An unexpected error occurred. Please check logs for details."

    RETURN {
        "content": [
            {
                "type": "text",
                "text": f"Error: {error_text}"
            }
        ],
        "isError": True
    }

END FUNCTION
```

---

## Preconditions

### System Preconditions

1. **SearchMemoryTool Initialized**
   - `__init__()` has been called successfully
   - `self.core` references valid MemoryEngine instance
   - `self.logger` configured for structured logging

2. **Core Engine Ready**
   - Core engine initialized and operational
   - Database connection established
   - Embedding service available (Ollama)

3. **MCP Server Context**
   - Function called within MCP server event loop
   - Async context available (not called from sync code)

### Input Preconditions

1. **arguments Parameter**
   - `arguments` is a dict (guaranteed by MCP server)
   - `arguments` is not None
   - `arguments` may be empty `{}` (will fail validation)

2. **No Schema Pre-validation**
   - MCP server does NOT validate against input_schema
   - All validation happens inside execute()
   - Arguments may contain unknown keys (Pydantic will reject)

### State Preconditions

1. **No Request State**
   - Function is stateless
   - No previous call state affects current call
   - Each invocation independent

2. **Database State**
   - Database MAY be empty (no memories stored) → valid, returns 0 results
   - Database MAY have no vector index → SearchError
   - Database MAY be unavailable → CoreError

---

## Postconditions

### Guaranteed Postconditions

1. **Always Returns dict**
   - Return type is always `dict[str, Any]`
   - Never returns None
   - Never raises exceptions to caller

2. **MCP-Compliant Response**
   - Response has "content" key (list of content blocks)
   - Response has "isError" key (bool)
   - Content blocks have "type" and "text" keys

3. **Logging Complete**
   - Start event logged
   - Success OR error event logged
   - All structured data captured

### Success Postconditions

When `isError=False`:

1. **Valid Search Performed**
   - Query was validated
   - Core search executed
   - Results (or no results) returned

2. **Response Format**
   - Text is human-readable
   - Results numbered and formatted
   - Scores displayed to 2 decimals

3. **No State Changes**
   - Tool instance unchanged
   - No cache updates
   - No side effects (except logs)

### Error Postconditions

When `isError=True`:

1. **Error Captured**
   - Error type determined
   - Error message formatted
   - Full details logged

2. **Safe Error Message**
   - No sensitive data leaked
   - User-friendly message
   - Actionable (when possible)

3. **Graceful Degradation**
   - System remains operational
   - No resource leaks
   - Ready for next request

---

## Edge Cases & Handling

### Edge Case 1: Empty Query String

**Scenario:**
```python
arguments = {"query": ""}
```

**Expected Behavior:**
```python
# ValidationError raised by Pydantic
# Caught in execute()
# Formatted as error response
```

**Response:**
```python
{
    "content": [
        {
            "type": "text",
            "text": "Error: Invalid input - query: ensure this value has at least 1 character"
        }
    ],
    "isError": True
}
```

**Logging:**
```python
self.logger.warning(
    "validation_error",
    errors=[{"loc": ("query",), "msg": "ensure this value has at least 1 character"}],
    arguments={"query": ""}
)
```

**Test Scenario:**
```python
async def test_execute_empty_query_raises_validation_error():
    """Edge Case 1: Empty query string"""
    tool = SearchMemoryTool(core=mock_core)

    result = await tool.execute({"query": ""})

    assert result["isError"] is True
    assert "Invalid input" in result["content"][0]["text"]
    assert "query" in result["content"][0]["text"]
    assert "at least 1 character" in result["content"][0]["text"]
```

---

### Edge Case 2: Query Exceeds Maximum Length

**Scenario:**
```python
arguments = {"query": "x" * 1001}  # 1001 characters
```

**Expected Behavior:**
```python
# ValidationError: query too long
# Max length is 1000 characters
```

**Response:**
```python
{
    "content": [
        {
            "type": "text",
            "text": "Error: Invalid input - query: ensure this value has at most 1000 characters"
        }
    ],
    "isError": True
}
```

**Logging:**
```python
self.logger.warning(
    "validation_error",
    errors=[{"loc": ("query",), "msg": "ensure this value has at most 1000 characters"}],
    arguments={"query": "x" * 1001}  # Logged, may be truncated in logs
)
```

**Test Scenario:**
```python
async def test_execute_query_too_long_raises_validation_error():
    """Edge Case 2: Query exceeds 1000 char limit"""
    tool = SearchMemoryTool(core=mock_core)

    huge_query = "x" * 1001
    result = await tool.execute({"query": huge_query})

    assert result["isError"] is True
    assert "Invalid input" in result["content"][0]["text"]
    assert "at most 1000 characters" in result["content"][0]["text"]
```

---

### Edge Case 3: Whitespace-Only Query

**Scenario:**
```python
arguments = {"query": "   "}  # Only spaces
```

**Expected Behavior:**
```python
# Field validator detects whitespace-only
# ValidationError raised
```

**Response:**
```python
{
    "content": [
        {
            "type": "text",
            "text": "Error: Invalid input - query: cannot be whitespace-only"
        }
    ],
    "isError": True
}
```

**Validation Logic:**
```python
@field_validator('query')
def validate_query_not_whitespace(cls, v):
    if not v.strip():
        raise ValueError("query cannot be whitespace-only")
    return v.strip()  # Return stripped version
```

**Test Scenario:**
```python
async def test_execute_whitespace_query_raises_validation_error():
    """Edge Case 3: Whitespace-only query"""
    tool = SearchMemoryTool(core=mock_core)

    result = await tool.execute({"query": "   "})

    assert result["isError"] is True
    assert "whitespace-only" in result["content"][0]["text"]
```

---

### Edge Case 4: Limit Zero

**Scenario:**
```python
arguments = {"query": "test", "limit": 0}
```

**Expected Behavior:**
```python
# ValidationError: limit must be >= 1
```

**Response:**
```python
{
    "content": [
        {
            "type": "text",
            "text": "Error: Invalid input - limit: ensure this value is greater than or equal to 1"
        }
    ],
    "isError": True
}
```

**Test Scenario:**
```python
async def test_execute_limit_zero_raises_validation_error():
    """Edge Case 4: Limit zero"""
    tool = SearchMemoryTool(core=mock_core)

    result = await tool.execute({"query": "test", "limit": 0})

    assert result["isError"] is True
    assert "greater than or equal to 1" in result["content"][0]["text"]
```

---

### Edge Case 5: Limit Exceeds Maximum

**Scenario:**
```python
arguments = {"query": "test", "limit": 200}
```

**Expected Behavior:**
```python
# ValidationError: limit must be <= 100
```

**Response:**
```python
{
    "content": [
        {
            "type": "text",
            "text": "Error: Invalid input - limit: ensure this value is less than or equal to 100"
        }
    ],
    "isError": True
}
```

**Test Scenario:**
```python
async def test_execute_limit_exceeds_max_raises_validation_error():
    """Edge Case 5: Limit exceeds 100"""
    tool = SearchMemoryTool(core=mock_core)

    result = await tool.execute({"query": "test", "limit": 200})

    assert result["isError"] is True
    assert "less than or equal to 100" in result["content"][0]["text"]
```

---

### Edge Case 6: Negative Limit

**Scenario:**
```python
arguments = {"query": "test", "limit": -5}
```

**Expected Behavior:**
```python
# ValidationError: limit must be >= 1
```

**Response:**
```python
{
    "content": [
        {
            "type": "text",
            "text": "Error: Invalid input - limit: ensure this value is greater than or equal to 1"
        }
    ],
    "isError": True
}
```

**Test Scenario:**
```python
async def test_execute_negative_limit_raises_validation_error():
    """Edge Case 6: Negative limit"""
    tool = SearchMemoryTool(core=mock_core)

    result = await tool.execute({"query": "test", "limit": -5})

    assert result["isError"] is True
    assert "greater than or equal to 1" in result["content"][0]["text"]
```

---

### Edge Case 7: Invalid Filter Keys

**Scenario:**
```python
arguments = {
    "query": "test",
    "filters": {"unknown_key": "value"}
}
```

**Expected Behavior:**
```python
# Pydantic rejects extra fields
# ValidationError: extra fields not permitted
```

**Response:**
```python
{
    "content": [
        {
            "type": "text",
            "text": "Error: Invalid input - filters: extra fields not permitted"
        }
    ],
    "isError": True
}
```

**Test Scenario:**
```python
async def test_execute_invalid_filter_keys_raises_validation_error():
    """Edge Case 7: Unknown filter keys"""
    tool = SearchMemoryTool(core=mock_core)

    result = await tool.execute({
        "query": "test",
        "filters": {"invalid_key": "value"}
    })

    assert result["isError"] is True
    assert "extra fields not permitted" in result["content"][0]["text"]
```

---

### Edge Case 8: Malformed Date Filter

**Scenario:**
```python
arguments = {
    "query": "test",
    "filters": {"date_from": "2025/01/01"}  # Wrong format
}
```

**Expected Behavior:**
```python
# Pydantic date parsing fails
# ValidationError: invalid date format
```

**Response:**
```python
{
    "content": [
        {
            "type": "text",
            "text": "Error: Invalid input - filters.date_from: invalid date format, expected YYYY-MM-DD"
        }
    ],
    "isError": True
}
```

**Test Scenario:**
```python
async def test_execute_malformed_date_filter_raises_validation_error():
    """Edge Case 8: Malformed date format"""
    tool = SearchMemoryTool(core=mock_core)

    result = await tool.execute({
        "query": "test",
        "filters": {"date_from": "2025/01/01"}
    })

    assert result["isError"] is True
    assert "invalid date format" in result["content"][0]["text"]
```

---

### Edge Case 9: Invalid Date Range

**Scenario:**
```python
arguments = {
    "query": "test",
    "filters": {
        "date_from": "2025-12-31",
        "date_to": "2025-01-01"  # Before date_from
    }
}
```

**Expected Behavior:**
```python
# Model validator catches invalid range
# ValidationError: date_from must be <= date_to
```

**Response:**
```python
{
    "content": [
        {
            "type": "text",
            "text": "Error: Invalid input - filters: date_from must be <= date_to"
        }
    ],
    "isError": True
}
```

**Test Scenario:**
```python
async def test_execute_invalid_date_range_raises_validation_error():
    """Edge Case 9: date_from > date_to"""
    tool = SearchMemoryTool(core=mock_core)

    result = await tool.execute({
        "query": "test",
        "filters": {
            "date_from": "2025-12-31",
            "date_to": "2025-01-01"
        }
    })

    assert result["isError"] is True
    assert "date_from must be <=" in result["content"][0]["text"]
```

---

### Edge Case 10: No Results Found

**Scenario:**
```python
# Valid query, but no matches in database
arguments = {"query": "nonexistent topic xyz123"}
```

**Expected Behavior:**
```python
# Core returns SearchResponse with count=0
# NOT an error, just no results
```

**Response:**
```python
{
    "content": [
        {
            "type": "text",
            "text": "No results found matching your query."
        }
    ],
    "isError": False  # NOT an error!
}
```

**Test Scenario:**
```python
async def test_execute_no_results_found_returns_empty_message():
    """Edge Case 10: No results (not an error)"""
    mock_core = MagicMock()
    mock_core.search_memory = AsyncMock(
        return_value=SearchResponse(count=0, results=[])
    )

    tool = SearchMemoryTool(core=mock_core)
    result = await tool.execute({"query": "nonexistent"})

    assert result["isError"] is False  # Success, just no results
    assert "No results found" in result["content"][0]["text"]
```

---

### Edge Case 11: Database Connection Failure

**Scenario:**
```python
# Valid query, but database unavailable
arguments = {"query": "test"}
# Core raises CoreError due to DB connection failure
```

**Expected Behavior:**
```python
# CoreError caught
# Formatted as processing error
```

**Response:**
```python
{
    "content": [
        {
            "type": "text",
            "text": "Error: Processing error: Database connection failed"
        }
    ],
    "isError": True
}
```

**Test Scenario:**
```python
async def test_execute_database_failure_returns_processing_error():
    """Edge Case 11: Database connection fails"""
    mock_core = MagicMock()
    mock_core.search_memory = AsyncMock(
        side_effect=CoreError("Database connection failed")
    )

    tool = SearchMemoryTool(core=mock_core)
    result = await tool.execute({"query": "test"})

    assert result["isError"] is True
    assert "Processing error" in result["content"][0]["text"]
    assert "Database connection" in result["content"][0]["text"]
```

---

### Edge Case 12: Embedding Service Unavailable

**Scenario:**
```python
# Valid query, but Ollama (embedding service) down
arguments = {"query": "test"}
# Core raises SearchError
```

**Expected Behavior:**
```python
# SearchError caught
# Formatted as search failure
```

**Response:**
```python
{
    "content": [
        {
            "type": "text",
            "text": "Error: Search failed: Embedding generation failed - Ollama connection refused"
        }
    ],
    "isError": True
}
```

**Test Scenario:**
```python
async def test_execute_embedding_service_down_returns_search_error():
    """Edge Case 12: Embedding service unavailable"""
    mock_core = MagicMock()
    mock_core.search_memory = AsyncMock(
        side_effect=SearchError("Embedding generation failed - Ollama connection refused")
    )

    tool = SearchMemoryTool(core=mock_core)
    result = await tool.execute({"query": "test"})

    assert result["isError"] is True
    assert "Search failed" in result["content"][0]["text"]
    assert "Embedding" in result["content"][0]["text"]
```

---

### Edge Case 13: Filters Exclude All Results

**Scenario:**
```python
# Query matches results, but filters exclude all
arguments = {
    "query": "Python",
    "filters": {"tags": ["nonexistent_tag"]}
}
# Core returns 0 results after filtering
```

**Expected Behavior:**
```python
# SearchResponse with count=0
# NOT an error
```

**Response:**
```python
{
    "content": [
        {
            "type": "text",
            "text": "No results found matching your query."
        }
    ],
    "isError": False
}
```

**Test Scenario:**
```python
async def test_execute_filters_exclude_all_results():
    """Edge Case 13: Filters exclude all matches"""
    mock_core = MagicMock()
    mock_core.search_memory = AsyncMock(
        return_value=SearchResponse(count=0, results=[])
    )

    tool = SearchMemoryTool(core=mock_core)
    result = await tool.execute({
        "query": "Python",
        "filters": {"tags": ["nonexistent"]}
    })

    assert result["isError"] is False
    assert "No results found" in result["content"][0]["text"]
```

---

## Test Scenarios (Complete List)

### Happy Path Tests (5)

#### 1. test_execute_success_minimal_arguments
**Purpose:** Verify basic search with only query parameter

**Input:**
```python
{"query": "What is Python?"}
```

**Mocked Core Response:**
```python
SearchResponse(
    count=3,
    results=[
        SearchResult(
            memory_id="uuid-1",
            text="Python is a programming language",
            similarity_score=0.92,
            metadata={"tags": ["python"]}
        ),
        SearchResult(
            memory_id="uuid-2",
            text="Python was created by Guido",
            similarity_score=0.88,
            metadata={}
        ),
        SearchResult(
            memory_id="uuid-3",
            text="Python is used for web development",
            similarity_score=0.85,
            metadata={"tags": ["python", "web"]}
        )
    ]
)
```

**Expected Response:**
```python
{
    "content": [
        {
            "type": "text",
            "text": "Found 3 results:\n\n1. [Score: 0.92] [Tags: python]\nPython is a programming language\n\n2. [Score: 0.88]\nPython was created by Guido\n\n3. [Score: 0.85] [Tags: python, web]\nPython is used for web development\n"
        }
    ],
    "isError": False
}
```

**Assertions:**
- `result["isError"]` is `False`
- `"Found 3 results"` in text
- All 3 results numbered correctly
- Scores formatted to 2 decimals
- Tags displayed when present
- core.search_memory called once with correct request

---

#### 2. test_execute_success_with_custom_limit
**Purpose:** Verify custom limit parameter works

**Input:**
```python
{"query": "Python", "limit": 5}
```

**Mocked Core Response:**
```python
SearchResponse(count=5, results=[...])  # 5 results
```

**Assertions:**
- core.search_memory called with limit=5
- Response shows "Found 5 results"
- isError=False

---

#### 3. test_execute_success_with_filters
**Purpose:** Verify filters passed to core correctly

**Input:**
```python
{
    "query": "Python",
    "filters": {
        "tags": ["tutorial"],
        "source": "documentation",
        "date_from": "2025-01-01"
    }
}
```

**Mocked Core Response:**
```python
SearchResponse(count=2, results=[...])
```

**Assertions:**
- core.search_memory called with filters in request
- request.filters.tags == ["tutorial"]
- request.filters.source == "documentation"
- request.filters.date_from == date(2025, 1, 1)
- isError=False

---

#### 4. test_execute_success_boundary_limit_1
**Purpose:** Verify minimum limit (1) works

**Input:**
```python
{"query": "test", "limit": 1}
```

**Mocked Core Response:**
```python
SearchResponse(count=1, results=[...])  # Single result
```

**Assertions:**
- "Found 1 results:" in text (grammatical note: spec says "results" plural)
- Only 1 result displayed
- isError=False

---

#### 5. test_execute_success_boundary_limit_100
**Purpose:** Verify maximum limit (100) works

**Input:**
```python
{"query": "test", "limit": 100}
```

**Mocked Core Response:**
```python
SearchResponse(count=100, results=[...])  # 100 results
```

**Assertions:**
- core.search_memory called with limit=100
- Response shows "Found 100 results"
- isError=False

---

### Validation Error Tests (10)

#### 6. test_execute_empty_query_raises_validation_error
**Purpose:** Edge Case 1 - Empty query string

See Edge Case 1 above for full details.

---

#### 7. test_execute_query_too_long_raises_validation_error
**Purpose:** Edge Case 2 - Query exceeds 1000 chars

See Edge Case 2 above for full details.

---

#### 8. test_execute_whitespace_query_raises_validation_error
**Purpose:** Edge Case 3 - Whitespace-only query

See Edge Case 3 above for full details.

---

#### 9. test_execute_limit_zero_raises_validation_error
**Purpose:** Edge Case 4 - Limit zero

See Edge Case 4 above for full details.

---

#### 10. test_execute_limit_exceeds_max_raises_validation_error
**Purpose:** Edge Case 5 - Limit > 100

See Edge Case 5 above for full details.

---

#### 11. test_execute_negative_limit_raises_validation_error
**Purpose:** Edge Case 6 - Negative limit

See Edge Case 6 above for full details.

---

#### 12. test_execute_invalid_filter_keys_raises_validation_error
**Purpose:** Edge Case 7 - Unknown filter keys

See Edge Case 7 above for full details.

---

#### 13. test_execute_malformed_date_filter_raises_validation_error
**Purpose:** Edge Case 8 - Invalid date format

See Edge Case 8 above for full details.

---

#### 14. test_execute_invalid_date_range_raises_validation_error
**Purpose:** Edge Case 9 - date_from > date_to

See Edge Case 9 above for full details.

---

#### 15. test_execute_missing_query_field_raises_validation_error
**Purpose:** Missing required query field

**Input:**
```python
{}  # Empty arguments
```

**Expected Response:**
```python
{
    "content": [
        {
            "type": "text",
            "text": "Error: Invalid input - query: field required"
        }
    ],
    "isError": True
}
```

**Assertions:**
- isError=True
- "field required" in text
- "query" in text

---

### Search/Core Error Tests (5)

#### 16. test_execute_no_results_found_returns_empty_message
**Purpose:** Edge Case 10 - No results (not error)

See Edge Case 10 above for full details.

---

#### 17. test_execute_database_failure_returns_processing_error
**Purpose:** Edge Case 11 - Database connection fails

See Edge Case 11 above for full details.

---

#### 18. test_execute_embedding_service_down_returns_search_error
**Purpose:** Edge Case 12 - Embedding service unavailable

See Edge Case 12 above for full details.

---

#### 19. test_execute_filters_exclude_all_results
**Purpose:** Edge Case 13 - Filters exclude all matches

See Edge Case 13 above for full details.

---

#### 20. test_execute_unexpected_exception_returns_generic_error
**Purpose:** Unexpected exception handling

**Input:**
```python
{"query": "test"}
# Core raises RuntimeError (not expected exception)
```

**Mocked Behavior:**
```python
mock_core.search_memory.side_effect = RuntimeError("Unexpected crash")
```

**Expected Response:**
```python
{
    "content": [
        {
            "type": "text",
            "text": "Error: An unexpected error occurred. Please check logs for details."
        }
    ],
    "isError": True
}
```

**Assertions:**
- isError=True
- Generic error message (no specifics)
- logger.exception called (full traceback logged)

---

### Response Formatting Tests (5)

#### 21. test_format_response_text_truncation
**Purpose:** Verify text preview truncated to 200 chars

**Input to _format_response:**
```python
SearchResponse(
    count=1,
    results=[
        SearchResult(
            memory_id="uuid",
            text="x" * 500,  # 500 characters
            similarity_score=0.90,
            metadata={}
        )
    ]
)
```

**Expected Output:**
```python
# Text preview should be 200 chars + "..."
"Found 1 results:\n\n1. [Score: 0.90]\n" + ("x" * 200) + "...\n"
```

**Assertions:**
- Text truncated to 200 chars
- "..." appended
- No tags shown (metadata empty)

---

#### 22. test_format_response_includes_tags_when_present
**Purpose:** Verify tags displayed

**Input:**
```python
SearchResponse(
    count=1,
    results=[
        SearchResult(
            memory_id="uuid",
            text="Short text",
            similarity_score=0.95,
            metadata={"tags": ["python", "tutorial", "beginner"]}
        )
    ]
)
```

**Expected Output:**
```python
"Found 1 results:\n\n1. [Score: 0.95] [Tags: python, tutorial, beginner]\nShort text\n"
```

**Assertions:**
- Tags displayed comma-separated
- Tags in correct order

---

#### 23. test_format_response_no_tags_when_absent
**Purpose:** Verify no tags shown when metadata empty

**Input:**
```python
SearchResponse(
    count=1,
    results=[
        SearchResult(
            memory_id="uuid",
            text="Text",
            similarity_score=0.90,
            metadata={}  # No tags
        )
    ]
)
```

**Expected Output:**
```python
"Found 1 results:\n\n1. [Score: 0.90]\nText\n"
# No [Tags: ...] section
```

**Assertions:**
- "[Tags:" not in output
- Score still displayed

---

#### 24. test_format_response_score_formatting
**Purpose:** Verify similarity score formatted to 2 decimals

**Input:**
```python
SearchResponse(
    count=3,
    results=[
        SearchResult(text="a", similarity_score=0.923456, ...),  # Should be 0.92
        SearchResult(text="b", similarity_score=0.885, ...),     # Should be 0.88 (not 0.89)
        SearchResult(text="c", similarity_score=1.0, ...)        # Should be 1.00
    ]
)
```

**Assertions:**
- First score: "[Score: 0.92]"
- Second score: "[Score: 0.88]" (not 0.89, verify truncation not rounding)
- Third score: "[Score: 1.00]" (2 decimals even for .0)

Note: Verify that Python's f-string formatting `{score:.2f}` is used (which does rounding, not truncation).

---

#### 25. test_format_response_multiple_results_numbered
**Purpose:** Verify results numbered 1, 2, 3, ...

**Input:**
```python
SearchResponse(
    count=5,
    results=[...5 results...]
)
```

**Assertions:**
- First result starts with "1. "
- Second result starts with "2. "
- Fifth result starts with "5. "
- Numbering sequential

---

### Integration Tests (3)

#### 26. test_execute_full_workflow_integration
**Purpose:** End-to-end test with real Pydantic validation (not mocked)

**Setup:**
- Use real SearchRequest model (not mocked)
- Mock only core.search_memory()

**Input:**
```python
{
    "query": "Python programming",
    "limit": 10,
    "filters": {"tags": ["tutorial"]}
}
```

**Assertions:**
- Pydantic validates successfully
- SearchRequest created with correct fields
- Core called with validated request
- Response formatted correctly
- No mocking of validation logic

---

#### 27. test_execute_logging_workflow
**Purpose:** Verify all log events emitted

**Input:**
```python
{"query": "test"}
```

**Assertions:**
- logger.debug called with "search_memory_execute_start"
- logger.info called with "search_request_validated"
- logger.info called with "search_completed"
- Log context includes query_length, limit, result_count

---

#### 28. test_execute_performance_within_target
**Purpose:** Verify performance target met

**Input:**
```python
{"query": "test"}
# Mock core.search_memory to return quickly
```

**Assertions:**
- Total execution time < 250ms (allowing buffer over 220ms P50 target)
- Validation < 10ms
- Formatting < 20ms

---

## Performance Requirements

### Latency Targets

**Total Latency (P50):** < 220ms
**Total Latency (P95):** < 520ms
**Total Latency (P99):** < 1000ms

**Breakdown (P50):**
- Input validation: < 5ms
- Core delegation: < 200ms (depends on core, outside control)
- Response formatting: < 10ms
- Overhead (logging, etc.): < 5ms

**Breakdown (P95):**
- Input validation: < 10ms (complex filters)
- Core delegation: < 500ms (large result sets)
- Response formatting: < 20ms (100 results)
- Overhead: < 10ms

**Timeout:**
No explicit timeout enforced in execute() - MCP server may timeout requests at protocol level (typically 30-60 seconds).

---

### Throughput Targets

**Sequential Requests:** N/A (MCP stdio is single-threaded)
**Concurrent Requests:** Not applicable (stdio transport)

**Future (HTTP Transport):**
If MCP adds HTTP transport, this function is thread-safe (stateless) and could handle:
- 100 req/sec (with core layer scaling)
- Limited only by core performance

---

### Resource Usage

**Memory:**
- Instance overhead: ~500 bytes (core ref, logger)
- Per-request: ~1-5 KB (arguments, request/response models)
- Response formatting: O(results * avg_text_length)
  - Worst case: 100 results * 200 chars = 20 KB
- Total: < 30 KB per request

**CPU:**
- Validation: O(1) for simple fields, O(n) for filters (n = filter keys, small)
- Formatting: O(m) where m = number of results (max 100)
- Negligible compared to core embedding/search operations

**I/O:**
- Logging: ~1-2 KB per request (structured logs)
- No direct I/O (delegated to core)

---

### Optimization Opportunities

**Current Performance:** Acceptable for Phase 1

**Potential Optimizations (Future):**

1. **Schema Caching**
   - Cache Pydantic JSON schema generation
   - Saves ~1ms per request
   - Trade-off: Adds state (violates stateless design)

2. **Response Streaming**
   - Stream large result sets (>50 results)
   - Reduces memory footprint
   - Requires MCP protocol changes

3. **Lazy Formatting**
   - Format results on-demand
   - Useful if MCP supports pagination
   - Current: format all results immediately

**Decision:** Keep current implementation. Optimize only if profiling shows tool layer is bottleneck (unlikely - core layer dominates latency).

---

## Security Considerations

### Input Validation

**All inputs validated before use:**
- Pydantic enforces type constraints
- Length limits prevent DoS (query max 1000 chars)
- No arbitrary code execution (query is data, not code)

**Injection Prevention:**
- SQL injection: N/A (FalkorDB uses parameterized queries in core)
- NoSQL injection: Filters validated by Pydantic structure
- Command injection: No shell commands executed

---

### Data Protection

**Sensitive Data in Arguments:**
- Query may contain PII (user's search text)
- Logged with truncation (first 50 chars only)
- Not sanitized (user controls their own data)

**Sensitive Data in Responses:**
- Results may contain PII (user's memories)
- Returned as-is (user requested their data)
- No redaction applied

**Data Not Logged:**
- Full query text (only first 50 chars)
- Full results (only count logged)
- Filter values (only presence logged)

---

### Error Message Safety

**Safe Error Messages:**
- Never leak file paths
- Never leak database connection strings
- Never leak stack traces to user
- Never leak internal variable names

**Detailed Errors in Logs Only:**
- Full traceback: stderr logs
- Exception details: stderr logs
- System information: stderr logs

**User sees only:**
- High-level error category
- Actionable message (when possible)
- Generic fallback for unexpected errors

---

### Logging Security

**Structured Logging:**
- All logs JSON-formatted
- Sensitive fields redacted
- Query text truncated

**Log Access:**
- Logs written to stderr
- Not accessible to MCP client
- Only system admin can read logs

**PII in Logs:**
- Query preview (50 chars) - acceptable
- No full memory text logged
- No user identifiers logged (MCP protocol doesn't provide them)

---

## Related Functions

### Functions This Calls

#### 1. `_validate_input(arguments: dict) -> SearchRequest`

**Purpose:** Validate and parse arguments into SearchRequest model

**Called from:** execute() step 1

**Relation:**
- execute() delegates all validation to this helper
- If validation fails, ValidationError bubbles up to execute() error handler

---

#### 2. `core.search_memory(request: SearchRequest) -> SearchResponse`

**Purpose:** Perform actual search operation

**Called from:** execute() step 2

**Relation:**
- execute() is pure delegation - all search logic in core
- If search fails, SearchError/CoreError bubbles up to execute() error handler

---

#### 3. `_format_response(response: SearchResponse) -> dict`

**Purpose:** Format SearchResponse into MCP text response

**Called from:** execute() step 3

**Relation:**
- execute() delegates all formatting to this helper
- Pure function (no side effects, no errors)

---

#### 4. `_format_error(error: Exception) -> dict`

**Purpose:** Format exception into MCP error response

**Called from:** execute() error handlers

**Relation:**
- execute() delegates all error formatting to this helper
- Pure function (no side effects)

---

### Functions That Call This

#### 1. `MCPServer._handle_tool_call(tool_name: str, arguments: dict)`

**Purpose:** MCP server routes tool calls to appropriate tool

**Call Pattern:**
```python
if tool_name == "search_memory":
    result = await search_tool.execute(arguments)
    return result
```

**Relation:**
- Server orchestrates MCP protocol
- execute() just implements tool logic

---

#### 2. User Code (Testing)

**Purpose:** Direct invocation for testing

**Call Pattern:**
```python
tool = SearchMemoryTool(core=mock_core)
result = await tool.execute({"query": "test"})
assert result["isError"] is False
```

**Relation:**
- Tests call execute() directly
- Bypasses MCP server infrastructure

---

## Implementation Notes

### Libraries Used

**Pydantic (>=2.0):**
- Input validation via SearchRequest model
- JSON schema generation for MCP input_schema
- Field validators for custom constraints

**structlog:**
- Structured logging with context binding
- JSON output for observability
- Automatic log formatting

**typing (stdlib):**
- Type hints for function signature
- dict[str, Any] for MCP protocol types
- Protocol for MemoryEngine interface

---

### Known Limitations

**Text Preview Truncation:**
- First 200 characters only
- No intelligent sentence breaking
- May cut mid-word
- Future: Use NLP for sentence boundaries

**No Pagination:**
- All results returned at once
- Limited to 100 results max
- No cursor-based pagination
- Future: Add MCP pagination support

**No Result Caching:**
- Identical queries re-execute
- No deduplication
- Core layer may cache, but tool doesn't
- Future: Add tool-level caching

**Single Content Block:**
- All results in one text block
- No structured data format
- MCP supports multiple blocks, but not used
- Future: Return structured JSON results

---

### Future Enhancements

**Phase 2 (BM25 Hybrid Search):**
- Add search_type parameter ("vector" | "bm25" | "hybrid")
- Pass through to core layer
- Update input_schema

**Phase 3 (Advanced Filtering):**
- Add regex support for text filters
- Add numeric range filters (if metadata includes numbers)
- Add geolocation filters (if memories have location)

**Phase 4 (Result Ranking):**
- Add re-ranking parameter
- Add diversity parameter (reduce similar results)
- Expose core ranking options

---

## References

### Component Spec
- [SearchMemoryTool Component](../level2/search_memory_tool_component.md)
  - Section: "Public Methods > execute()" - High-level overview
  - Section: "Error Handling" - Exception strategy
  - Section: "Testing Approach" - Test categories

### Module Spec
- [zapomni_mcp Module](../level1/zapomni_mcp_module.md)
  - Section: "Public API > MCPTool Protocol" - Interface contract
  - Section: "Data Models > ToolResponse" - Response format

### Cross-Module Specs
- [Cross-Module Interfaces](../level1/cross_module_interfaces.md)
  - Section: "MCP → Core: MemoryEngine Protocol" - Core interface
  - Section: "Error Handling Strategy" - Exception propagation

### External Documentation
- [MCP Specification - Tools](https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/tools/)
  - Tool execution contract
  - Response format requirements

- [Pydantic Documentation - Validation](https://docs.pydantic.dev/latest/concepts/validators/)
  - Field validators
  - Model validators
  - Error handling

- [structlog Documentation](https://www.structlog.org/en/stable/logging-best-practices.html)
  - Structured logging patterns
  - Context binding

---

## Document Status

**Version:** 1.0
**Created:** 2025-11-23
**Author:** Goncharenko Anton aka alienxs2
**Status:** Draft - Ready for Verification

**Verification Checklist:**
- ✅ Function signature complete with full docstring
- ✅ All parameters detailed (type, constraints, examples)
- ✅ All return values documented (success + error cases)
- ✅ All exceptions documented (internal catching)
- ✅ Algorithm in detailed pseudocode
- ✅ Preconditions specified
- ✅ Postconditions specified
- ✅ Edge cases identified (13 cases)
- ✅ Test scenarios defined (28 tests)
- ✅ Performance targets specified
- ✅ Security considerations documented
- ✅ Examples provided (usage + test)

**Next Steps:**
1. Multi-agent verification
2. Reconciliation with parent component spec
3. Approval
4. Ready for implementation

**Estimated Implementation Time:** 4-6 hours
**Estimated Test Writing Time:** 6-8 hours (28 test scenarios)
**Total Lines of Code (Estimated):** ~300 lines (function + helpers + tests)

---

**Copyright:** Copyright (c) 2025 Goncharenko Anton aka alienxs2
**License:** MIT License
**Document Length:** ~2100 lines
**Estimated Reading Time:** 60-75 minutes
**Target Audience:** Developer implementing SearchMemoryTool.execute() method
