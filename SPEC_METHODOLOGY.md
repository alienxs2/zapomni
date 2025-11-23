# Specification Methodology - Three-Level Cascade

**Project:** Zapomni
**Author:** Goncharenko Anton aka alienxs2
**License:** MIT
**Date:** 2025-11-22

Related: [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md)

---

## 📋 Overview

Этот документ описывает детальную методологию 3-уровневого spec-driven подхода к разработке Zapomni. Спецификации создаются ДО написания кода, обеспечивая максимальную согласованность и качество.

### Why Spec-Driven?

**Проблемы code-first подхода:**
- ❌ Несогласованность между модулями
- ❌ Недокументированные assumptions
- ❌ Сложная координация параллельных разработчиков/агентов
- ❌ Обнаружение проблем поздно (во время интеграции)

**Преимущества spec-first:**
- ✅ Ранее обнаружение противоречий
- ✅ Чёткие contracts между компонентами
- ✅ Параллельная разработка без конфликтов
- ✅ Тесты пишутся на основе specs (TDD)
- ✅ Documentation by design

### The Three-Level Cascade

```
Level 1: MODULE SPECS (5-7 docs)
   ↓ Breakdown
Level 2: COMPONENT SPECS (15-20 docs)
   ↓ Breakdown
Level 3: FUNCTION SPECS (40-50 docs)
   ↓ Implementation
TESTS (200+ test cases)
   ↓ Code
WORKING SOFTWARE
```

Каждый уровень проходит строгий verification process перед переходом к следующему.

---

## 🏗️ Level 1: Module-Level Specifications

### Purpose

Определить high-level архитектуру системы на уровне модулей.

### Deliverables (7 documents)

1. **zapomni_mcp_module.md**
   - MCP protocol adapter
   - Exposes tools через MCP stdio

2. **zapomni_core_module.md**
   - Core processing engine
   - Chunking, embeddings, entity extraction

3. **zapomni_db_module.md**
   - Database clients
   - FalkorDB integration

4. **cross_module_interfaces.md**
   - How modules communicate
   - Interface definitions
   - Data contracts

5. **data_flow_architecture.md**
   - How data flows through system
   - Processing pipelines
   - State management

6. **error_handling_strategy.md**
   - Error propagation
   - Recovery strategies
   - Logging approach

7. **configuration_management.md**
   - Configuration structure
   - Environment variables
   - Runtime settings

### Document Template

```markdown
# [Module Name] - Module Specification

**Level:** 1 (Module)
**Author:** Goncharenko Anton aka alienxs2
**Status:** [Draft | Verified | Approved]
**Version:** 1.0

## Overview

### Purpose
[What this module does, why it exists]

### Scope
[What's included, what's NOT included]

### Position in Architecture
[How this module fits into overall system]

## Architecture

### High-Level Diagram
```
[ASCII or Mermaid diagram]
```

### Key Responsibilities
1. [Responsibility 1]
2. [Responsibility 2]
3. [Responsibility 3]

## Public API

### Interfaces

```python
class [ModuleName]:
    """[Purpose of this class]"""

    def __init__(self, config: Config) -> None:
        """Initialize module."""

    def public_method_1(self, param: Type) -> ReturnType:
        """[What this method does]"""
```

### Data Models

```python
@dataclass
class [ModelName]:
    """[Purpose of this data model]"""
    field1: str
    field2: int
    field3: Optional[dict]
```

## Dependencies

### External Dependencies
- library_name==version (purpose: why we need it)

### Internal Dependencies
- other_module (for: specific functionality)

### Dependency Rationale
[Why these specific dependencies]

## Data Flow

### Input
- What data enters this module
- Format and structure
- Validation requirements

### Processing
- Key transformations
- Business logic
- Side effects

### Output
- What data leaves this module
- Format and structure
- Guarantees provided

## Design Decisions

### Decision 1: [Title]
**Context:** [Why we needed to decide]
**Options Considered:**
- Option A: [description] (pros/cons)
- Option B: [description] (pros/cons)
**Chosen:** Option X
**Rationale:** [Why this choice]

### Decision 2: [Title]
[Same format]

## Non-Functional Requirements

### Performance
- Latency: [target]
- Throughput: [target]
- Resource usage: [limits]

### Scalability
- How module scales
- Bottlenecks
- Mitigation strategies

### Security
- Authentication/authorization
- Input validation
- Data protection

### Reliability
- Error handling approach
- Recovery strategies
- Fail-safe mechanisms

## Testing Strategy

### Unit Testing
- What to test at unit level
- Mocking strategy

### Integration Testing
- Integration points to test
- Test environment requirements

## Future Considerations

- Potential enhancements
- Known limitations
- Evolution path

## References

- product.md: [relevant sections]
- tech.md: [relevant sections]
- Related specs: [links]
```

### Verification Criteria

**Internal Consistency:**
- ✅ No contradictions within document
- ✅ All references valid
- ✅ Diagrams match text descriptions

**External Consistency:**
- ✅ Aligns with product.md vision
- ✅ Uses tech stack from tech.md
- ✅ Follows structure.md conventions

**Completeness:**
- ✅ All sections filled
- ✅ API fully defined
- ✅ Dependencies listed
- ✅ Design decisions documented

**Technical Feasibility:**
- ✅ Can be implemented with chosen stack
- ✅ Performance targets realistic
- ✅ No architectural impossibilities

---

## 🔧 Level 2: Component-Level Specifications

### Purpose

Break down каждый module на components (classes, services) с детальными API.

### How Many Components?

Rule of thumb: **3-4 components per module**

Example for zapomni_mcp (4 components):
1. MCP Server (main server class)
2. Tool Registry (manages tools)
3. AddMemory Tool (specific tool)
4. SearchMemory Tool (specific tool)
5. GetStats Tool (specific tool)

Total: ~15-20 component specs для 3 modules

### Document Template

```markdown
# [Component Name] - Component Specification

**Level:** 2 (Component)
**Module:** [Parent module name]
**Author:** Goncharenko Anton aka alienxs2
**Status:** [Draft | Verified | Approved]
**Version:** 1.0

## Overview

### Purpose
[What this component does]

### Responsibilities
1. [Responsibility 1]
2. [Responsibility 2]

### Position in Module
[How component fits within its module]

## Class Definition

### Class Diagram

```
┌─────────────────────────────┐
│     [ClassName]             │
├─────────────────────────────┤
│ - attribute1: Type          │
│ - attribute2: Type          │
├─────────────────────────────┤
│ + public_method1()          │
│ + public_method2()          │
│ - private_method()          │
└─────────────────────────────┘
```

### Full Class Signature

```python
class [ClassName]:
    """
    [Detailed class docstring]

    Attributes:
        attribute1: [description]
        attribute2: [description]

    Example:
        ```python
        instance = ClassName(param1="value")
        result = instance.public_method1()
        ```
    """

    # Class attributes
    CLASS_CONSTANT: str = "value"

    def __init__(
        self,
        param1: str,
        param2: Optional[int] = None
    ) -> None:
        """
        Initialize [ClassName].

        Args:
            param1: [description, constraints]
            param2: [description, default behavior]

        Raises:
            ValidationError: If param1 invalid
        """

    def public_method1(
        self,
        arg1: str,
        arg2: dict[str, Any]
    ) -> Result:
        """
        [What this method does]

        Args:
            arg1: [description]
            arg2: [description]

        Returns:
            Result object containing [fields]

        Raises:
            SpecificError: When [condition]
        """

    def public_method2(self) -> None:
        """[Brief description]"""
```

## Dependencies

### Component Dependencies
- OtherComponent (for: specific functionality)
- ThirdComponent (for: data access)

### External Libraries
- library_name (for: specific use)

### Dependency Injection
[How dependencies are provided]

## State Management

### Attributes
- `attribute1`: [type, purpose, lifetime]
- `attribute2`: [type, purpose, lifecycle]

### State Transitions
```
Initial State
    ↓
[Action] → State A
    ↓
[Action] → State B
```

### Thread Safety
- Is this component thread-safe? [Yes/No]
- If not: [concurrency constraints]
- If yes: [synchronization mechanisms]

## Public Methods (Detailed)

### Method 1: `public_method1`

**Signature:**
```python
def public_method1(self, arg1: str, arg2: dict) -> Result
```

**Purpose:** [Detailed description]

**Parameters:**
- `arg1`: str
  - Description: [what it represents]
  - Constraints: [validation rules]
  - Example: "example_value"

- `arg2`: dict[str, Any]
  - Description: [structure]
  - Required keys: ["key1", "key2"]
  - Optional keys: ["key3"]
  - Example: {"key1": "value", "key2": 123}

**Returns:**
- Type: `Result`
- Fields:
  - `success`: bool - operation status
  - `data`: Any - result data
  - `error`: Optional[str] - error message

**Raises:**
- `ValidationError`: When arg1 is empty or invalid format
- `ProcessingError`: When arg2 missing required keys

**Preconditions:**
- Component must be initialized
- Dependencies available

**Postconditions:**
- State updated to reflect operation
- Result logged

**Algorithm Outline:**
```
1. Validate inputs (arg1, arg2)
2. Process arg1 through dependency
3. Transform arg2 based on result
4. Update internal state
5. Return Result object
```

**Edge Cases:**
1. Empty arg1 → ValidationError
2. arg2 missing keys → ValidationError with specific message
3. Dependency unavailable → ProcessingError, retry logic
4. Result too large → truncate, log warning

**Related Methods:**
- Calls: `private_helper_method()`
- Called by: `higher_level_method()`

### Method 2: [Same detailed format]

## Error Handling

### Exceptions Defined
```python
class ComponentSpecificError(Exception):
    """Raised when [condition]"""
```

### Error Recovery
- Retry strategy: [when and how]
- Fallback behavior: [if can't recover]
- Error propagation: [what exceptions bubble up]

## Usage Examples

### Basic Usage
```python
# Initialize component
component = ClassName(param1="value")

# Basic operation
result = component.public_method1(
    arg1="input",
    arg2={"key": "value"}
)

if result.success:
    print(f"Success: {result.data}")
else:
    print(f"Error: {result.error}")
```

### Advanced Usage
```python
# Complex workflow
component = ClassName(param1="value")

try:
    result = component.public_method1(...)
    processed = component.public_method2(result.data)
except ComponentSpecificError as e:
    # Handle specific error
    logger.error(f"Failed: {e}")
```

## Testing Approach

### Unit Tests Required
- `test_init_success()` - normal initialization
- `test_init_invalid_params()` - validation
- `test_method1_success()` - happy path
- `test_method1_invalid_input()` - error cases
- [List all test scenarios]

### Mocking Strategy
- Mock `DependencyComponent`
- Mock external library calls
- Mock I/O operations

### Integration Tests
- Test with real dependencies
- Test component interactions

## Performance Considerations

### Time Complexity
- Method1: O(n) where n = input size
- Method2: O(1) constant time

### Space Complexity
- Memory usage: [estimate]
- Caching strategy: [if applicable]

### Optimization Opportunities
- [Potential optimizations]
- [Trade-offs]

## References

- Module spec: zapomni_[module]_module.md
- Related components: [links]
- External docs: [if any]
```

### Verification Criteria

**API Completeness:**
- ✅ All public methods have signatures
- ✅ All parameters typed
- ✅ All returns typed
- ✅ All exceptions documented

**Consistency:**
- ✅ Matches module spec interfaces
- ✅ Dependencies available in module
- ✅ State transitions logical

**Implementation Readiness:**
- ✅ Enough detail to write code
- ✅ Edge cases identified
- ✅ Test scenarios defined

---

## 📝 Level 3: Function-Level Specifications

### Purpose

Define EVERY public function/method с максимальной детализацией для прямой реализации и TDD.

### How Many Functions?

Rule of thumb: **2-3 public methods per component**

Example:
- 20 components × 2.5 methods average = **~50 function specs**

### Document Template

```markdown
# [Function Name] - Function Specification

**Level:** 3 (Function)
**Component:** [Parent component]
**Module:** [Parent module]
**Author:** Goncharenko Anton aka alienxs2
**Status:** [Draft | Verified | Approved]
**Version:** 1.0

## Function Signature

```python
def function_name(
    self,
    param1: str,
    param2: int,
    param3: Optional[dict[str, Any]] = None
) -> Result[Data]:
    """
    [One-line summary]

    [Detailed description of what this function does,
    why it exists, and when to use it.]

    Args:
        param1: [Detailed description, constraints, examples]
        param2: [Detailed description, valid ranges]
        param3: [Detailed description, structure if dict]

    Returns:
        Result object containing:
        - success: bool
        - data: Data object with [fields]
        - processing_time_ms: int

    Raises:
        ValidationError: When param1 is empty or invalid
        ProcessingError: When operation cannot complete
        DatabaseError: When DB connection fails

    Example:
        ```python
        obj = ClassName()
        result = obj.function_name(
            param1="valid_input",
            param2=42,
            param3={"key": "value"}
        )

        if result.success:
            print(result.data.field1)
        ```
    """
```

## Purpose & Context

### What It Does
[Detailed explanation of functionality]

### Why It Exists
[Business/technical reason for this function]

### When To Use
[Scenarios where this function is called]

### When NOT To Use
[Scenarios where alternative should be used]

## Parameters (Detailed)

### param1: str

**Type:** `str`

**Purpose:** [What this parameter represents]

**Constraints:**
- Must not be empty
- Must be valid UTF-8
- Maximum length: 10,000 characters
- Pattern: `^[a-zA-Z0-9_]+$` (if applicable)

**Validation:**
```python
if not param1:
    raise ValidationError("param1 cannot be empty")
if len(param1) > 10000:
    raise ValidationError("param1 exceeds max length")
```

**Examples:**
- Valid: `"user_input_123"`
- Invalid: `""` (empty)
- Invalid: `"x" * 10001` (too long)

### param2: int

**Type:** `int`

**Purpose:** [What this represents]

**Constraints:**
- Must be positive: `> 0`
- Valid range: `1 to 10000`

**Validation:**
```python
if param2 <= 0:
    raise ValidationError("param2 must be positive")
if param2 > 10000:
    raise ValidationError("param2 exceeds maximum")
```

**Examples:**
- Valid: `42`
- Invalid: `0`, `-5`, `10001`

### param3: Optional[dict[str, Any]]

**Type:** `Optional[dict[str, Any]]`

**Purpose:** [Optional metadata or configuration]

**Default:** `None` (empty dict behavior)

**Structure (when provided):**
```python
{
    "key1": str,  # Required if dict provided
    "key2": int,  # Optional
    "key3": list[str]  # Optional
}
```

**Validation:**
```python
if param3 is not None:
    if "key1" not in param3:
        raise ValidationError("param3 missing required key 'key1'")
    if not isinstance(param3["key1"], str):
        raise ValidationError("param3['key1'] must be string")
```

**Examples:**
- Valid: `None`
- Valid: `{"key1": "value"}`
- Valid: `{"key1": "value", "key2": 123}`
- Invalid: `{}` (missing required key1)
- Invalid: `{"key1": 123}` (wrong type)

## Return Value

**Type:** `Result[Data]`

**Structure:**
```python
@dataclass
class Result:
    success: bool
    data: Optional[Data]
    error: Optional[str]
    processing_time_ms: int

@dataclass
class Data:
    field1: str
    field2: int
    field3: list[str]
```

**Success Case:**
```python
Result(
    success=True,
    data=Data(field1="result", field2=42, field3=["a", "b"]),
    error=None,
    processing_time_ms=156
)
```

**Error Case:**
```python
Result(
    success=False,
    data=None,
    error="Specific error message",
    processing_time_ms=12
)
```

## Exceptions

### ValidationError

**When Raised:**
- param1 is empty or too long
- param2 is out of range
- param3 has invalid structure

**Message Format:**
```python
f"Validation failed for {param_name}: {specific_reason}"
```

**Example:**
```python
raise ValidationError("Validation failed for param1: exceeds max length")
```

### ProcessingError

**When Raised:**
- Operation cannot complete due to business logic
- Dependency returned error
- Resource temporarily unavailable

**Recovery:** Caller should retry (if transient) or escalate

### DatabaseError

**When Raised:**
- DB connection failed
- Query execution failed
- Transaction rollback

**Recovery:** Caller should retry with exponential backoff

## Algorithm (Pseudocode)

```
FUNCTION function_name(param1, param2, param3):
    # Step 1: Validate inputs
    VALIDATE param1 is not empty and within length
    VALIDATE param2 is positive and in range
    IF param3 provided:
        VALIDATE param3 structure

    # Step 2: Initialize
    start_time = current_time()
    result_data = empty Data object

    # Step 3: Process param1
    TRY:
        processed_param1 = preprocess(param1)
        result_data.field1 = processed_param1
    CATCH ProcessingError:
        RETURN Result(success=False, error="Processing failed")

    # Step 4: Apply param2
    result_data.field2 = param2 * calculation()

    # Step 5: Handle param3 (optional)
    IF param3:
        result_data.field3 = extract_list_from(param3)
    ELSE:
        result_data.field3 = default_list()

    # Step 6: Persist (if applicable)
    TRY:
        database.save(result_data)
    CATCH DatabaseError:
        RETURN Result(success=False, error="DB error")

    # Step 7: Return success
    processing_time = current_time() - start_time
    RETURN Result(
        success=True,
        data=result_data,
        error=None,
        processing_time_ms=processing_time
    )
END FUNCTION
```

## Preconditions

- ✅ Object must be initialized (`__init__` called)
- ✅ Dependencies injected and available
- ✅ Database connection established (if applicable)

## Postconditions

- ✅ If success=True: data is valid and complete
- ✅ If success=False: error message is informative
- ✅ State updated consistently
- ✅ Resources released properly

## Edge Cases & Handling

### Edge Case 1: Empty param1

**Scenario:** User passes empty string `""`

**Expected Behavior:**
```python
raise ValidationError("param1 cannot be empty")
```

**Test Scenario:**
```python
def test_function_name_empty_param1():
    obj = ClassName()
    with pytest.raises(ValidationError, match="param1 cannot be empty"):
        obj.function_name(param1="", param2=1)
```

### Edge Case 2: param1 Extremely Long

**Scenario:** param1 is 1 million characters

**Expected Behavior:**
```python
raise ValidationError("param1 exceeds max length")
```

**Test Scenario:**
```python
def test_function_name_param1_too_long():
    obj = ClassName()
    huge_input = "x" * 1_000_000
    with pytest.raises(ValidationError, match="exceeds max length"):
        obj.function_name(param1=huge_input, param2=1)
```

### Edge Case 3: param2 Zero

**Scenario:** param2 = 0

**Expected Behavior:**
```python
raise ValidationError("param2 must be positive")
```

**Test Scenario:**
```python
def test_function_name_param2_zero():
    obj = ClassName()
    with pytest.raises(ValidationError, match="must be positive"):
        obj.function_name(param1="valid", param2=0)
```

### Edge Case 4: param3 Missing Required Key

**Scenario:** param3 = `{"wrong_key": "value"}`

**Expected Behavior:**
```python
raise ValidationError("param3 missing required key 'key1'")
```

**Test Scenario:**
```python
def test_function_name_param3_missing_key():
    obj = ClassName()
    with pytest.raises(ValidationError, match="missing required key"):
        obj.function_name(param1="valid", param2=1, param3={"wrong": "value"})
```

### Edge Case 5: Database Connection Fails

**Scenario:** DB unavailable during save

**Expected Behavior:**
```python
Result(success=False, error="Database connection failed", ...)
```

**Test Scenario:**
```python
def test_function_name_db_failure(mocker):
    obj = ClassName()
    mocker.patch.object(obj.db, 'save', side_effect=DatabaseError)

    result = obj.function_name(param1="valid", param2=1)

    assert result.success is False
    assert "Database" in result.error
```

### Edge Case 6: Processing Takes Very Long

**Scenario:** Operation exceeds timeout

**Expected Behavior:**
```python
raise TimeoutError("Operation exceeded timeout")
```

**Test Scenario:**
```python
def test_function_name_timeout(mocker):
    obj = ClassName(timeout=1)  # 1 second timeout
    mocker.patch('time.sleep', lambda x: None)  # Don't actually sleep
    mocker.patch.object(obj, '_process', side_effect=lambda: time.sleep(2))

    with pytest.raises(TimeoutError):
        obj.function_name(param1="valid", param2=1)
```

## Test Scenarios (Complete List)

### Happy Path Tests

1. **test_function_name_success_minimal**
   - Input: Valid param1, param2, param3=None
   - Expected: success=True, data populated correctly

2. **test_function_name_success_with_param3**
   - Input: Valid param1, param2, param3 with all fields
   - Expected: success=True, param3 data integrated

3. **test_function_name_success_boundary_values**
   - Input: param2=1 (minimum), param2=10000 (maximum)
   - Expected: success=True for both

### Error Tests

4. **test_function_name_empty_param1_raises**
   - Edge case 1 above

5. **test_function_name_param1_too_long_raises**
   - Edge case 2 above

6. **test_function_name_param2_zero_raises**
   - Edge case 3 above

7. **test_function_name_param2_negative_raises**
   - Input: param2=-5
   - Expected: ValidationError

8. **test_function_name_param2_too_large_raises**
   - Input: param2=10001
   - Expected: ValidationError

9. **test_function_name_param3_missing_key_raises**
   - Edge case 4 above

10. **test_function_name_param3_wrong_type_raises**
    - Input: param3={"key1": 123}  # Should be string
    - Expected: ValidationError

### Integration/Dependency Tests

11. **test_function_name_db_failure_returns_error**
    - Edge case 5 above

12. **test_function_name_dependency_unavailable**
    - Mock dependency to raise error
    - Expected: ProcessingError

13. **test_function_name_timeout_raises**
    - Edge case 6 above

### Performance Tests

14. **test_function_name_performance_within_sla**
    - Input: Normal size param1
    - Expected: processing_time_ms < 100ms

15. **test_function_name_large_input_performance**
    - Input: param1 near max length
    - Expected: Still completes within acceptable time

## Performance Requirements

**Latency:**
- Normal input (< 1KB): < 50ms
- Large input (< 10KB): < 200ms
- Maximum allowed: 500ms

**Throughput:**
- Concurrent calls: Support up to 100/sec

**Resource Usage:**
- Memory: O(n) where n = input size
- CPU: O(n) for processing

## Security Considerations

**Input Validation:**
- ✅ All inputs validated before use
- ✅ No injection vulnerabilities
- ✅ Safe error messages (no sensitive data leaked)

**Data Protection:**
- Sensitive data in param3? [Yes/No]
- If yes: encryption required? Logging restrictions?

## Related Functions

**Calls:**
- `preprocess(param1: str) -> str`
- `calculation() -> int`
- `extract_list_from(data: dict) -> list[str]`

**Called By:**
- `higher_level_workflow()` in WorkflowComponent
- `batch_process()` for multiple items

## Implementation Notes

**Libraries Used:**
- `library_name` for specific processing

**Known Limitations:**
- Cannot handle binary data in param1 (UTF-8 only)
- param3 structure limited to JSON-serializable types

**Future Enhancements:**
- Support streaming for very large param1
- Async version for better concurrency

## References

- Component spec: [link]
- Module spec: [link]
- Related function specs: [links]
```

### Verification Criteria

**Completeness:**
- ✅ Every parameter fully specified (type, constraints, examples)
- ✅ Every edge case identified (minimum 3)
- ✅ Every test scenario defined (minimum 5)
- ✅ Algorithm in pseudocode

**Testability:**
- ✅ Can write tests directly from spec
- ✅ Expected behaviors clear
- ✅ Edge cases have test scenarios

**Implementation Readiness:**
- ✅ Developer can code function from spec alone
- ✅ No ambiguity in requirements
- ✅ All dependencies identified

---

## 🔍 Multi-Agent Verification Process

### Overview

Каждый level проходит строгий verification process с 5 overlapping agents для максимальной надёжности.

### Verification Matrix (Level 1 Example)

**Documents:**
1. zapomni_mcp_module.md
2. zapomni_core_module.md
3. zapomni_db_module.md
4. cross_module_interfaces.md
5. data_flow_architecture.md
6. error_handling_strategy.md
7. configuration_management.md

**Agent Assignment:**
- **Agent 1:** Docs [1, 2, 3]
- **Agent 2:** Docs [3, 4, 5]
- **Agent 3:** Docs [5, 6, 7]
- **Agent 4:** Docs [2, 4, 6]
- **Agent 5:** Docs [1, 3, 5, 7]

**Coverage Analysis:**
| Doc | Verified By | Count |
|-----|-------------|-------|
| 1 | Agents 1, 5 | 2x |
| 2 | Agents 1, 4 | 2x |
| 3 | Agents 1, 2, 5 | **3x** ⭐ |
| 4 | Agents 2, 4 | 2x |
| 5 | Agents 2, 3, 5 | **3x** ⭐ |
| 6 | Agents 3, 4 | 2x |
| 7 | Agents 3, 5 | 2x |

Критичные документы (3, 5) проверяются 3 агентами для максимальной надёжности.

### Verification Checklist

**Each agent checks:**

1. **Internal Consistency** (within single document)
   - ✅ No contradictions
   - ✅ All cross-references valid
   - ✅ Diagrams match text
   - ✅ Examples executable

2. **Cross-Document Consistency** (between assigned docs)
   - ✅ API contracts match
     - If doc A exports interface X, doc B imports X correctly
   - ✅ Data models aligned
     - Same structure defined identically across docs
   - ✅ Dependencies correct
     - If A depends on B, B provides it
   - ✅ No circular dependencies

3. **Steering Alignment**
   - ✅ Matches product.md (vision, features)
   - ✅ Uses tech.md stack (FalkorDB, Ollama, Python)
   - ✅ Follows structure.md conventions (naming, organization)

4. **Technical Feasibility**
   - ✅ Can be implemented with chosen tech stack
   - ✅ Performance targets realistic
   - ✅ No architectural impossibilities

5. **Completeness**
   - ✅ All features from product.md covered
   - ✅ All edge cases identified
   - ✅ Error handling specified

### Verification Report Template

```markdown
# Verification Report - Agent [N]

**Documents Verified:** [1, 2, 3]
**Date:** 2025-11-22
**Verifier:** Agent [N]

## ✅ APPROVED ASPECTS

- Document 1:
  - Well-defined API
  - Clear responsibilities
  - Good separation of concerns

- Documents 1 & 2:
  - Interface contracts match perfectly
  - Data models consistent

## ⚠️ WARNINGS (Non-Critical)

### Warning 1: Naming Inconsistency
- **Location:** Doc1:Section3, Doc2:Section2
- **Issue:** Document 1 calls it "MemoryStore", Document 2 calls it "MemoryStorage"
- **Suggestion:** Standardize on "MemoryStore"
- **Priority:** Low

### Warning 2: Missing Example
- **Location:** Doc2:API_Definition
- **Issue:** No usage example provided for complex API
- **Suggestion:** Add example in Section 5
- **Priority:** Medium

## ❌ CRITICAL ISSUES (Blocking)

### Issue 1: Interface Mismatch
- **Location:** Doc1:Section2 <-> Doc2:Section3
- **Type:** Contradiction
- **Description:**
  - Doc1 defines `process(data: str) -> Result`
  - Doc2 expects `process(data: dict) -> Response`
  - Types incompatible!
- **Impact:** Cannot compile, integration will fail
- **Recommendation:**
  - Change Doc1 to accept `dict` OR
  - Change Doc2 to send `str`
  - Prefer dict for flexibility
- **Priority:** Critical

### Issue 2: Missing Dependency
- **Location:** Doc3:Dependencies
- **Type:** Missing
- **Description:**
  - Doc3 uses FalkorDBClient but doesn't list it as dependency
  - Doc zapomni_db_module.md provides it but not referenced
- **Impact:** Unclear dependency graph
- **Recommendation:** Add cross-reference
- **Priority:** High

## CROSS-DOCUMENT FINDINGS

### Finding 1: Circular Dependency
- **Documents:** Doc1 → Doc2 → Doc3 → Doc1
- **Issue:** Circular import will cause initialization problems
- **Fix:** Introduce abstraction layer or dependency injection

### Finding 2: Duplicate Responsibility
- **Documents:** Doc1 and Doc2
- **Issue:** Both modules handle "validation"
- **Fix:** Centralize in one module or clarify different validation types

## METRICS

- Documents analyzed: 3
- Issues found: Critical 2, Warnings 2
- Consistency score: 85%
- Feasibility: ✅ Confirmed

## FINAL VERDICT

- [ ] APPROVE (zero critical issues)
- [ ] APPROVE WITH WARNINGS (< 3 warnings, zero critical)
- [X] REJECT (>= 1 critical issue)

**Justification:** 2 critical issues found that will block implementation. Must be fixed before proceeding.

## RECOMMENDATIONS

1. Fix Issue 1 (interface mismatch) - highest priority
2. Fix Issue 2 (missing dependency reference)
3. Address warnings (naming, examples) - nice to have
4. Consider refactoring to eliminate circular dependency
```

### Synthesis Process

After 5 agents complete verification:

**Synthesis Agent:**
1. Reads all 5 verification reports
2. Identifies patterns:
   - If 2+ agents found same issue → **confirmed issue**
   - If only 1 agent → **possible false positive**, but investigate
3. Aggregates all issues
4. Prioritizes:
   - CRITICAL issues (block progress)
   - WARNINGS (address if time permits)
   - APPROVED aspects (good to know)
5. Creates synthesis report

**Synthesis Report Template:**

```markdown
# Synthesis Report - Level 1 Verification

**Date:** 2025-11-22
**Input:** 5 agent verification reports
**Documents:** 7 module-level specs

## SUMMARY

- Total agents: 5
- Total issues found: 8 (3 critical, 5 warnings)
- Confirmed issues: 5 (found by 2+ agents)
- Unique issues: 3 (found by 1 agent)

## CONFIRMED CRITICAL ISSUES (2+ agents)

### Issue 1: Interface Mismatch (Found by Agents 1, 2, 5)
- **Consistency:** 3 agents independently identified
- **Severity:** Critical
- **Description:** [consolidated from reports]
- **Recommended Fix:** [consensus approach]

### Issue 2: Circular Dependency (Found by Agents 2, 4)
- **Consistency:** 2 agents identified
- **Severity:** Critical
- **Description:** [consolidated]
- **Recommended Fix:** [consensus]

## UNIQUE CRITICAL ISSUES (1 agent)

### Issue 3: Missing Error Handling (Agent 3 only)
- **Needs Validation:** Only 1 agent found this
- **Possible False Positive:** Or other agents missed it
- **Action:** Reconciliation agent should investigate

## WARNINGS (Non-Blocking)

[Consolidated list of warnings from all agents]

## APPROVED ASPECTS

[What all agents agreed is good]

## DECISION

- [ ] APPROVE (zero critical)
- [ ] APPROVE WITH WARNINGS
- [X] REJECT - Must fix 2 confirmed critical issues

## NEXT STEPS

1. Reconciliation agent validates unique issues
2. Refinement agent fixes confirmed issues
3. Re-verification of changed docs
```

### Reconciliation Process

**Reconciliation Agent:**
1. Takes synthesis report
2. Reads steering documents (product.md, tech.md, structure.md)
3. For each issue:
   - Does recommended fix align with steering vision?
   - Is it technically sound per tech.md?
   - Does it follow structure.md conventions?
4. Validates or adjusts recommendations
5. Creates reconciliation report

**Reconciliation Report:**

```markdown
# Reconciliation Report - Level 1

**Date:** 2025-11-22
**Input:** Synthesis report + Steering documents

## STEERING ALIGNMENT CHECK

### Issue 1: Interface Mismatch
**Synthesis Recommendation:** Change to `dict` type

**Steering Check:**
- ✅ product.md: No specific requirement
- ✅ tech.md: Python encourages dict for flexibility
- ✅ structure.md: Typing conventions support dict[str, Any]

**Validation:** ✅ APPROVED
**Adjusted Recommendation:** Use `dict[str, Any]` with TypedDict for structure

### Issue 2: Circular Dependency
**Synthesis Recommendation:** Introduce abstraction layer

**Steering Check:**
- ✅ tech.md: Supports dependency injection
- ✅ structure.md: Recommends abstract base classes
- ⚠️ product.md: Keep it simple (don't over-engineer)

**Validation:** ⚠️ APPROVED with caution
**Adjusted Recommendation:** Use lightweight protocol (typing.Protocol), not full DI framework

## TECHNICAL FEASIBILITY

All recommended fixes are feasible with Python 3.10+ and chosen stack.

## FINAL RECOMMENDATIONS

1. **Issue 1:** Change interface to `dict[str, Any]` ✅
2. **Issue 2:** Use typing.Protocol for abstraction ✅
3. **All Warnings:** Address during refinement ✅

## DECISION

Proceed to refinement phase with approved recommendations.
```

### Refinement Process

**Refinement Agent:**
1. Takes reconciliation report
2. Edits specs to fix issues
3. Tracks changes
4. Prepares for re-verification

**Refinement Actions:**

```markdown
# Refinement Actions - Level 1

**Date:** 2025-11-22
**Input:** Reconciliation report

## CHANGES MADE

### Changed Document 1 (zapomni_mcp_module.md)

**Section:** API Definition
**Before:**
```python
def process(self, data: str) -> Result
```

**After:**
```python
def process(self, data: dict[str, Any]) -> Result
```

**Rationale:** Fix interface mismatch with Document 2

### Changed Document 3 (zapomni_db_module.md)

**Section:** Dependencies
**Before:**
[No FalkorDBClient mentioned]

**After:**
```
### Internal Dependencies
- FalkorDBClient (from zapomni_db.clients)
```

**Rationale:** Make dependency explicit

### Changed Documents 1, 2, 3 (Circular Dependency Fix)

**Added to each:**
```python
from typing import Protocol

class DataProcessor(Protocol):
    def process(self, data: dict[str, Any]) -> Result: ...
```

**Rationale:** Break circular dependency with Protocol

## WARNINGS ADDRESSED

- Fixed naming inconsistency: "MemoryStore" everywhere
- Added usage examples to Doc2

## FILES MODIFIED

- zapomni_mcp_module.md (lines 45-47, 102-105)
- zapomni_core_module.md (lines 23-25, 67-70)
- zapomni_db_module.md (lines 89-93)

## RE-VERIFICATION NEEDED

- Documents 1, 2, 3 (modified)
- Documents 4, 5, 6, 7 (unchanged, skip re-verification)

## STATUS

Refinement complete. Ready for re-verification cycle.
```

### Re-Verification (If Needed)

- Only verify CHANGED documents
- Use subset of agents (e.g., Agents 1, 2 who verified those docs originally)
- If changes are minor and low-risk → skip re-verification (user decision)

### Iteration Limit

**Maximum 3 cycles:**
- Cycle 1: Initial verification → usually finds issues
- Cycle 2: Re-verification after fixes → should be much cleaner
- Cycle 3: Final check → rare

**If still issues after 3 cycles:**
→ Escalate to user for manual review/decision

---

## 🎯 Workflow Integration

### Spec Creation Flow

```
1. USER REQUEST
   ↓
2. LOAD SPEC-WORKFLOW GUIDE
   ↓
3. CREATE STEERING DOCS (if not exist)
   - product.md
   - tech.md
   - structure.md
   ↓
4. LEVEL 1: MODULE SPECS
   - Create 7 module specs
   - Multi-agent verification
   - Fix issues, re-verify
   - Approval checkpoint
   ↓
5. LEVEL 2: COMPONENT SPECS
   - Create ~20 component specs
   - Multi-agent verification
   - Fix issues, re-verify
   - Approval checkpoint
   ↓
6. LEVEL 3: FUNCTION SPECS
   - Create ~50 function specs
   - Multi-agent verification
   - Fix issues, re-verify
   - Approval checkpoint
   ↓
7. IMPLEMENTATION READY
   - All specs verified
   - Tests can be written from specs
   - Code can be implemented
```

### Agent Coordination

**Roles:**
1. **Creation Agent:** Writes initial spec drafts
2. **Verification Agents (5):** Check specs for issues
3. **Synthesis Agent:** Aggregates verification findings
4. **Reconciliation Agent:** Validates against steering docs
5. **Refinement Agent:** Fixes identified issues
6. **User:** Final approval at each level

**Communication Protocol:**
- All agents read same steering documents
- Verification reports use standardized template
- Clear handoff points between agents
- Audit trail maintained

### Quality Gates

**Level 1 Gate:**
- ✅ All 7 module specs created
- ✅ Multi-agent verification passed
- ✅ Zero critical issues
- ✅ Warnings < 5 total
- ✅ User approval obtained

**Level 2 Gate:**
- ✅ All ~20 component specs created
- ✅ Multi-agent verification passed
- ✅ Zero critical issues
- ✅ All components map to modules
- ✅ User approval obtained

**Level 3 Gate:**
- ✅ All ~50 function specs created
- ✅ Multi-agent verification passed
- ✅ Zero critical issues
- ✅ All functions map to components
- ✅ Test scenarios complete (5+ per function)
- ✅ User approval obtained

---

## 📚 Best Practices

### Writing Good Specs

1. **Be Specific, Not Ambiguous**
   - ❌ "Validate the input"
   - ✅ "Check param1 is non-empty string, max 10,000 chars, UTF-8"

2. **Include Examples**
   - Every parameter: valid + invalid examples
   - Every function: usage example
   - Every error: example error message

3. **Think Edge Cases First**
   - Empty inputs
   - Maximum size inputs
   - Null/None values
   - Concurrent access
   - Failures of dependencies

4. **Define "Why", Not Just "What"**
   - Explain design decisions
   - Document trade-offs
   - Note future considerations

5. **Use Consistent Terminology**
   - Create glossary in steering docs
   - Use same terms throughout all specs
   - Define abbreviations

6. **Make It Testable**
   - Every requirement → test scenario
   - Every edge case → test case
   - Clear success/failure criteria

### Common Pitfalls

**Pitfall 1: Under-specification**
- Issue: "Returns a result"
- Fix: "Returns Result object with success: bool, data: Data, error: Optional[str]"

**Pitfall 2: Implicit Assumptions**
- Issue: Assuming DB is always available
- Fix: Explicitly state preconditions and error handling

**Pitfall 3: Inconsistent Terminology**
- Issue: "Memory", "MemoryItem", "MemoryData" used interchangeably
- Fix: Pick ONE term, use everywhere, define in glossary

**Pitfall 4: Missing Edge Cases**
- Issue: Only happy path specified
- Fix: Minimum 3 edge cases per function

**Pitfall 5: Vague Performance Requirements**
- Issue: "Should be fast"
- Fix: "Latency < 50ms for inputs < 1KB"

**Pitfall 6: No Error Recovery**
- Issue: "Raises DatabaseError when DB fails"
- Fix: "Raises DatabaseError; caller should retry with exponential backoff up to 3 times"

**Pitfall 7: Copy-Paste Without Adaptation**
- Issue: Using template placeholders verbatim
- Fix: Replace ALL [brackets] with actual content

**Pitfall 8: Circular References**
- Issue: Module A depends on B, B depends on A
- Fix: Introduce abstraction (Protocol, ABC) or reorganize

### Documentation Standards

**File Naming:**
- Module level: `{module_name}_module.md`
- Component level: `{component_name}_component.md`
- Function level: `{function_name}_function.md`

**Markdown Structure:**
- Use H1 (#) for document title only
- Use H2 (##) for major sections
- Use H3 (###) for subsections
- Use code fences with language tags

**Version Control:**
- Every spec has version number
- Track changes in git
- Tag major milestones (L1_verified, L2_verified, L3_verified)

**Cross-References:**
- Use relative links between specs
- Link to steering documents
- Maintain bidirectional links (parent ↔ child)

---

## 🔬 Advanced Topics

### Handling Changes

**During Spec Phase:**
1. Update spec document
2. Trigger re-verification (only changed docs)
3. Update dependent specs if needed
4. Re-obtain approval

**After Implementation Started:**
1. Assess impact (what code affected?)
2. Update spec + code together
3. Update tests
4. Full regression testing

### Parallel Development

**Multiple Agents Working Simultaneously:**

1. **L1:** Agent creates module specs sequentially
2. **L2:** After L1 verified, spawn 3 agents:
   - Agent A: Components for Module 1
   - Agent B: Components for Module 2
   - Agent C: Components for Module 3
3. **L3:** After L2 verified, spawn 5 agents:
   - Each agent handles subset of functions
   - Ensures no overlapping components

**Coordination:**
- Shared steering documents (read-only)
- Separate spec files per agent (no conflicts)
- Merge verification at end

### Spec-to-Code Mapping

**Traceability Matrix:**

| Spec Level | Code Artifact | Test Artifact |
|------------|---------------|---------------|
| Module | Python package | Integration tests |
| Component | Python class | Component tests |
| Function | Python method | Unit tests |

**Example:**
```
zapomni_mcp_module.md
  ↓
zapomni/mcp/__init__.py (package)
  ↓
tests/integration/test_mcp_module.py

AddMemoryTool_component.md
  ↓
zapomni/mcp/tools/add_memory.py (class AddMemoryTool)
  ↓
tests/component/test_add_memory_tool.py

AddMemoryTool.execute_function.md
  ↓
AddMemoryTool.execute() method
  ↓
tests/unit/test_add_memory_execute.py
```

### Metrics & Quality

**Spec Quality Metrics:**
- Completeness: % sections filled vs. template
- Consistency: Issues found per verification
- Testability: Test scenarios per requirement
- Clarity: Words like "maybe", "probably" (should be zero)

**Target Metrics:**
- Completeness: 100% (all sections filled)
- Critical issues: 0 after final verification
- Warnings: < 3 per document
- Test coverage: 5+ scenarios per function
- Edge cases: 3+ per function

**Tracking:**
```markdown
# Spec Metrics - Zapomni Project

**Date:** 2025-11-22

## Level 1 (Modules)
- Documents: 7
- Completeness: 100%
- Critical issues: 0
- Warnings: 2
- Verification cycles: 2

## Level 2 (Components)
- Documents: 18
- Completeness: 98% (2 missing examples)
- Critical issues: 0
- Warnings: 5
- Verification cycles: 1

## Level 3 (Functions)
- Documents: 47
- Completeness: 100%
- Critical issues: 0
- Warnings: 1
- Test scenarios: 267 (avg 5.7 per function)
- Edge cases: 156 (avg 3.3 per function)
- Verification cycles: 1

## Overall
- Total specs: 72
- Total verification agents used: 15
- Total verification hours: 28
- Ready for implementation: ✅ YES
```

---

## 🛠️ Tools & Automation

### Spec Templates

**Template Generator:**
```bash
# Generate module spec template
./tools/generate_spec.py --level 1 --name zapomni_mcp

# Generate component spec template
./tools/generate_spec.py --level 2 --module mcp --name AddMemoryTool

# Generate function spec template
./tools/generate_spec.py --level 3 --component AddMemoryTool --name execute
```

### Validation Scripts

**Spec Validator:**
```bash
# Validate single spec
./tools/validate_spec.py specs/level1/zapomni_mcp_module.md

# Validate all specs at level
./tools/validate_spec.py --level 1

# Check cross-references
./tools/validate_spec.py --check-refs
```

**Checks:**
- All template sections present
- No [placeholder] text remaining
- Valid Python syntax in code blocks
- Links resolve correctly
- Consistent terminology

### Verification Automation

**Agent Orchestration:**
```bash
# Run verification with 5 agents
./tools/verify_specs.py --level 1 --agents 5

# Generate synthesis report
./tools/synthesize_verification.py --reports reports/*.md

# Run reconciliation
./tools/reconcile.py --synthesis synthesis.md --steering steering/
```

### Metrics Dashboard

**Generate Metrics:**
```bash
# Calculate spec metrics
./tools/metrics.py --level 1

# Generate quality report
./tools/quality_report.py --all-levels

# Export to HTML dashboard
./tools/dashboard.py --output metrics.html
```

---

## 📖 Examples

### Example: Module Spec (Abbreviated)

```markdown
# Zapomni MCP Module - Module Specification

**Level:** 1 (Module)
**Author:** Goncharenko Anton aka alienxs2
**Status:** Verified
**Version:** 1.0

## Overview

### Purpose
Provides MCP protocol adapter that exposes Zapomni memory system as MCP tools via stdio transport.

### Scope
**Included:**
- MCP server implementation (stdio)
- Tool definitions (add_memory, search_memory, get_stats)
- Request/response handling
- Error marshalling

**Not Included:**
- Memory processing logic (in zapomni_core)
- Database operations (in zapomni_db)
- Embedding generation (in zapomni_core)

### Position in Architecture
Entry point for Claude Desktop and other MCP clients. Acts as adapter between MCP protocol and Zapomni core.

## Architecture

### High-Level Diagram
```
┌─────────────┐
│ MCP Client  │
│ (Claude)    │
└──────┬──────┘
       │ stdio
       ↓
┌─────────────┐
│ MCP Server  │
│ (this)      │
└──────┬──────┘
       │
       ↓
┌─────────────┐
│ Zapomni Core│
└─────────────┘
```

### Key Responsibilities
1. MCP protocol compliance (stdio transport)
2. Tool registration and routing
3. Input validation and sanitization
4. Response formatting (MCP schema)
5. Error handling and reporting

## Public API

### Interfaces

```python
from typing import Protocol, Any

class MCPTool(Protocol):
    """Interface that all MCP tools must implement."""

    name: str
    description: str
    input_schema: dict[str, Any]

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute tool with provided arguments."""
        ...

class MCPServer:
    """Main MCP server implementing stdio transport."""

    def __init__(self, core_engine: ZapomniCore) -> None:
        """Initialize server with core engine."""

    def register_tool(self, tool: MCPTool) -> None:
        """Register a tool with the server."""

    async def run(self) -> None:
        """Start server and process requests from stdin."""
```

### Data Models

```python
@dataclass
class ToolRequest:
    """Incoming MCP tool request."""
    method: str  # "tools/call"
    params: dict[str, Any]

@dataclass
class ToolResponse:
    """Outgoing MCP tool response."""
    content: list[dict[str, Any]]  # MCP content blocks
    isError: bool
```

[... rest of module spec ...]
```

### Example: Component Spec (Abbreviated)

```markdown
# AddMemoryTool - Component Specification

**Level:** 2 (Component)
**Module:** zapomni_mcp
**Author:** Goncharenko Anton aka alienxs2
**Status:** Verified
**Version:** 1.0

## Overview

### Purpose
Implements MCP tool for adding memories to Zapomni knowledge graph.

### Responsibilities
1. Validate add_memory tool inputs
2. Call ZapomniCore.add_memory()
3. Format results as MCP response
4. Handle errors gracefully

## Class Definition

### Full Class Signature

```python
class AddMemoryTool:
    """
    MCP tool for adding memories to knowledge graph.

    Attributes:
        name: Tool name ("add_memory")
        description: Human-readable description
        input_schema: JSON schema for tool inputs
        core: ZapomniCore instance

    Example:
        ```python
        tool = AddMemoryTool(core=core_engine)
        result = await tool.execute({
            "content": "Python is a programming language",
            "metadata": {"source": "user"}
        })
        ```
    """

    name: str = "add_memory"
    description: str = "Add a memory to the knowledge graph"

    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "Memory content to store"
            },
            "metadata": {
                "type": "object",
                "description": "Optional metadata"
            }
        },
        "required": ["content"]
    }

    def __init__(self, core: ZapomniCore) -> None:
        """Initialize tool with core engine."""
        self.core = core

    async def execute(
        self,
        arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Execute add_memory tool.

        Args:
            arguments: Tool arguments matching input_schema

        Returns:
            MCP-formatted response with results

        Raises:
            ValidationError: If arguments invalid
            ProcessingError: If core processing fails
        """

[... rest of component spec ...]
```

### Example: Function Spec (Abbreviated)

```markdown
# AddMemoryTool.execute - Function Specification

**Level:** 3 (Function)
**Component:** AddMemoryTool
**Module:** zapomni_mcp
**Author:** Goncharenko Anton aka alienxs2
**Status:** Verified
**Version:** 1.0

## Function Signature

```python
async def execute(
    self,
    arguments: dict[str, Any]
) -> dict[str, Any]:
    """
    Execute add_memory tool with provided arguments.

    Validates inputs, calls core engine to process memory,
    and returns MCP-formatted response.

    Args:
        arguments: Dictionary with keys:
            - content (str, required): Memory text to store
            - metadata (dict, optional): Additional metadata

    Returns:
        Dictionary in MCP response format:
        {
            "content": [
                {
                    "type": "text",
                    "text": "Success message with memory_id"
                }
            ],
            "isError": false
        }

    Raises:
        ValidationError: If content missing or empty
        ProcessingError: If core engine fails

    Example:
        ```python
        tool = AddMemoryTool(core)
        result = await tool.execute({
            "content": "Python is great",
            "metadata": {"source": "chat"}
        })
        # Returns: {"content": [...], "isError": false}
        ```
    """
```

## Purpose & Context

### What It Does
Validates input arguments, extracts content and optional metadata, calls ZapomniCore to process and store the memory, formats the result as MCP response.

### Why It Exists
Required by MCP protocol - all tools must implement execute() method. This is the entry point when Claude calls add_memory tool.

### When To Use
Called automatically by MCP server when it receives add_memory tool request from client.

## Parameters (Detailed)

### arguments: dict[str, Any]

**Type:** `dict[str, Any]`

**Purpose:** Contains tool arguments from MCP client

**Structure:**
```python
{
    "content": str,  # Required
    "metadata": dict[str, Any]  # Optional
}
```

**Constraints:**
- Must have "content" key
- content must be non-empty string
- content max length: 100,000 chars
- metadata must be JSON-serializable if provided

**Validation:**
```python
if "content" not in arguments:
    raise ValidationError("Missing required field: content")

content = arguments["content"]
if not isinstance(content, str):
    raise ValidationError("content must be string")

if not content.strip():
    raise ValidationError("content cannot be empty")

if len(content) > 100_000:
    raise ValidationError("content exceeds max length (100,000)")
```

**Examples:**
- Valid: `{"content": "Python is great"}`
- Valid: `{"content": "...", "metadata": {"source": "user"}}`
- Invalid: `{}` (missing content)
- Invalid: `{"content": ""}` (empty)
- Invalid: `{"content": 123}` (wrong type)

[... rest of function spec with all edge cases, tests, etc. ...]
```

---

## 🎓 Training & Onboarding

### For New Agents

**Step 1: Read Steering Documents**
- product.md - understand vision
- tech.md - understand tech stack
- structure.md - understand conventions

**Step 2: Study This Methodology**
- Read entire SPEC_METHODOLOGY.md
- Understand 3-level cascade
- Review verification process

**Step 3: Review Examples**
- Read example specs in /specs/examples/
- Note level of detail required
- Understand edge case handling

**Step 4: Practice**
- Create draft spec from template
- Run through verification checklist yourself
- Compare with examples

**Step 5: Collaboration**
- Work with experienced agent first
- Get feedback on first few specs
- Gradually take on full documents

### Quick Reference Card

```
┌─────────────────────────────────────────┐
│   SPEC METHODOLOGY QUICK REFERENCE      │
├─────────────────────────────────────────┤
│ LEVELS:                                 │
│   L1: Modules (7 docs)                  │
│   L2: Components (15-20 docs)           │
│   L3: Functions (40-50 docs)            │
│                                         │
│ VERIFICATION:                           │
│   - 5 overlapping agents                │
│   - Synthesis → Reconciliation          │
│   - Max 3 iteration cycles              │
│                                         │
│ QUALITY GATES:                          │
│   - 0 critical issues                   │
│   - < 5 warnings                        │
│   - 100% completeness                   │
│   - User approval required              │
│                                         │
│ BEST PRACTICES:                         │
│   - Be specific (no ambiguity)          │
│   - Include examples (every param)      │
│   - Think edge cases (min 3)            │
│   - Define "why" (not just "what")      │
│   - Test scenarios (min 5)              │
│                                         │
│ COMMON PITFALLS:                        │
│   ❌ Under-specification                │
│   ❌ Implicit assumptions               │
│   ❌ Inconsistent terminology           │
│   ❌ Missing edge cases                 │
│   ❌ Vague performance reqs             │
│                                         │
│ TEMPLATES:                              │
│   ./tools/generate_spec.py --level N    │
│                                         │
│ VALIDATION:                             │
│   ./tools/validate_spec.py <file>       │
└─────────────────────────────────────────┘
```

---

## 📚 References

### Internal Documents
- [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) - Main development plan
- [product.md](.spec-workflow/steering/product.md) - Product vision
- [tech.md](.spec-workflow/steering/tech.md) - Technical decisions
- [structure.md](.spec-workflow/steering/structure.md) - Project structure

### External Resources
- MCP Specification: https://spec.modelcontextprotocol.io/
- Python Type Hints (PEP 484): https://peps.python.org/pep-0484/
- FalkorDB Documentation: https://docs.falkordb.com/
- Pytest Documentation: https://docs.pytest.org/

### Related Methodologies
- TDD (Test-Driven Development)
- BDD (Behavior-Driven Development)
- Design by Contract
- Specification by Example

---

## 🔄 Changelog

### Version 1.0 (2025-11-22)
- Initial methodology document
- Defined 3-level cascade (modules, components, functions)
- Specified multi-agent verification process
- Created comprehensive templates for all levels
- Added examples and best practices
- Documented quality gates and metrics

---

**Author:** Goncharenko Anton aka alienxs2
**License:** MIT
**GitHub:** https://github.com/alienxs2/zapomni

*This document defines the specification methodology for Zapomni development.*
