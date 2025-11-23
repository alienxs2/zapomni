# TaskManager - Component Specification

**Level:** 2 (Component)
**Module:** zapomni_core
**Author:** Goncharenko Anton aka alienxs2
**Status:** Draft
**Version:** 1.0
**Created:** 2025-11-23

---

## Overview

### Purpose

TaskManager provides asynchronous background task management for long-running operations in Zapomni Core. It handles the execution, status tracking, and lifecycle management of tasks such as knowledge graph construction (`build_knowledge_graph`) and repository codification (`codify_repository`).

This component enables non-blocking execution of CPU-intensive operations by managing them as async tasks with proper state tracking, allowing users to submit tasks, monitor progress, and cancel operations when needed.

### Responsibilities

1. **Task Submission:** Accept async coroutines and schedule them for execution
2. **Task Execution:** Run tasks in the background using asyncio event loop
3. **Status Tracking:** Maintain task state (PENDING, RUNNING, COMPLETED, FAILED, CANCELLED)
4. **Progress Reporting:** Provide real-time status updates and completion percentage
5. **Task Cancellation:** Support cancelling running tasks gracefully
6. **Error Handling:** Capture and report task failures with detailed error messages
7. **Task Cleanup:** Remove completed/failed tasks after configurable TTL

### Position in Module

TaskManager is used by MemoryProcessor for background operations:

```
┌──────────────────────┐
│  MemoryProcessor     │
│  (Public API)        │
└──────────┬───────────┘
           │ delegates long-running ops
           ↓
┌──────────────────────┐
│   TaskManager        │  ← THIS COMPONENT
│  (Background Tasks)  │
└──────────┬───────────┘
           │ executes
           ↓
┌──────────────────────┐
│  EntityExtractor,    │
│  GraphBuilder,       │
│  CodeParser          │
└──────────────────────┘
```

**Key Relationships:**
- **MemoryProcessor → TaskManager:** Submits `build_knowledge_graph()` as async task
- **MCP Tools → TaskManager:** Query status via `get_task_status()`, cancel via `cancel_task()`
- **TaskManager → asyncio:** Uses asyncio.Task and asyncio.Queue for execution

---

## Class Definition

### Class Diagram

```
┌─────────────────────────────────────┐
│         TaskManager                 │
├─────────────────────────────────────┤
│ - tasks: dict[str, TaskState]       │
│ - queue: asyncio.Queue              │
│ - max_concurrent: int               │
│ - task_ttl: int                     │
│ - running_tasks: set[str]           │
├─────────────────────────────────────┤
│ + __init__(max_concurrent, ttl)     │
│ + submit_task(task_id, coro)        │
│ + get_task_status(task_id)          │
│ + cancel_task(task_id)              │
│ + get_all_tasks()                   │
│ - _cleanup_old_tasks()              │
│ - _execute_task(task_id, coro)      │
└─────────────────────────────────────┘
```

### Full Class Signature

```python
from typing import Dict, Any, Optional, Callable, Coroutine, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import asyncio

class TaskState(str, Enum):
    """Task execution states."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class TaskInfo:
    """
    Task metadata and status.

    Attributes:
        task_id: Unique identifier for task
        state: Current execution state
        progress: Completion percentage (0-100)
        created_at: Timestamp when task was submitted
        started_at: Timestamp when task started execution (None if pending)
        completed_at: Timestamp when task finished (None if not finished)
        result: Task result data (None if not completed)
        error: Error message (None if no error)
        metadata: Additional task-specific metadata
    """
    task_id: str
    state: TaskState
    progress: int
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

class TaskManager:
    """
    Manages asynchronous background tasks for long-running operations.

    Provides task submission, status tracking, cancellation, and lifecycle
    management using asyncio. Designed for operations like knowledge graph
    construction that would block the MCP server if run synchronously.

    Attributes:
        tasks: Dictionary mapping task_id to TaskInfo
        queue: Async queue for pending tasks
        max_concurrent: Maximum number of concurrent tasks
        task_ttl: Time-to-live for completed tasks (seconds)
        running_tasks: Set of currently running task IDs

    Example:
        ```python
        from zapomni_core.tasks import TaskManager
        import asyncio

        # Initialize manager
        manager = TaskManager(max_concurrent=5, task_ttl=3600)

        # Submit a task
        async def build_graph():
            await asyncio.sleep(10)  # Simulate work
            return {"entities": 100, "relationships": 50}

        task_id = await manager.submit_task(
            task_id="graph_build_001",
            coro=build_graph()
        )

        # Check status
        status = await manager.get_task_status(task_id)
        print(f"State: {status.state}, Progress: {status.progress}%")

        # Wait for completion
        while status.state == TaskState.RUNNING:
            await asyncio.sleep(1)
            status = await manager.get_task_status(task_id)

        # Get result
        if status.state == TaskState.COMPLETED:
            print(f"Result: {status.result}")
        ```

    Thread Safety:
        TaskManager is designed for single-threaded asyncio use. All methods
        must be called from the same event loop. Not thread-safe.
    """

    def __init__(
        self,
        max_concurrent: int = 5,
        task_ttl: int = 3600
    ) -> None:
        """
        Initialize TaskManager.

        Args:
            max_concurrent: Maximum concurrent tasks (default: 5)
            task_ttl: Time-to-live for completed tasks in seconds (default: 3600 = 1 hour)

        Raises:
            ValueError: If max_concurrent < 1 or task_ttl < 0
        """

    async def submit_task(
        self,
        task_id: str,
        coro: Coroutine,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Submit an async task for background execution.

        Creates a TaskInfo entry with PENDING state, schedules the coroutine
        for execution, and returns the task_id immediately for tracking.

        Args:
            task_id: Unique identifier for task (e.g., "build_graph_001")
            coro: Async coroutine to execute
            metadata: Optional metadata (task_type, description, etc.)

        Returns:
            task_id: The same task_id provided (for convenience)

        Raises:
            ValueError: If task_id already exists or is empty
            TaskError: If task queue is full (max_concurrent limit reached)

        Performance:
            - Execution time: < 10ms (just creates entry, doesn't run task)
            - Non-blocking: Returns immediately
        """

    async def get_task_status(
        self,
        task_id: str
    ) -> TaskInfo:
        """
        Get current status of a task.

        Returns complete TaskInfo with state, progress, timestamps, result/error.

        Args:
            task_id: Task identifier to query

        Returns:
            TaskInfo object with current status

        Raises:
            TaskNotFoundError: If task_id doesn't exist

        Performance:
            - Execution time: < 5ms (dict lookup)
        """

    async def cancel_task(
        self,
        task_id: str
    ) -> bool:
        """
        Cancel a running or pending task.

        If task is PENDING, removes from queue and marks CANCELLED.
        If task is RUNNING, sends cancellation signal to asyncio.Task.
        If task is already finished (COMPLETED/FAILED/CANCELLED), no-op.

        Args:
            task_id: Task identifier to cancel

        Returns:
            True if task was cancelled, False if already finished

        Raises:
            TaskNotFoundError: If task_id doesn't exist

        Performance:
            - Execution time: < 20ms
            - Graceful: Allows task to handle CancelledError

        Note:
            Task must handle asyncio.CancelledError to clean up resources.
            TaskManager waits up to 5 seconds for graceful shutdown.
        """

    async def get_all_tasks(
        self,
        state_filter: Optional[TaskState] = None
    ) -> List[TaskInfo]:
        """
        Get list of all tasks, optionally filtered by state.

        Args:
            state_filter: Optional state to filter by (e.g., TaskState.RUNNING)

        Returns:
            List of TaskInfo objects, sorted by created_at (descending)

        Performance:
            - Execution time: < 10ms for 100 tasks
        """

    async def _cleanup_old_tasks(self) -> int:
        """
        Remove completed/failed tasks older than task_ttl.

        Internal method called periodically (every 5 minutes).

        Returns:
            Number of tasks removed

        Performance:
            - Execution time: < 50ms for 1000 tasks
        """

    async def _execute_task(
        self,
        task_id: str,
        coro: Coroutine
    ) -> None:
        """
        Execute a task coroutine and update its status.

        Internal method that wraps the coroutine with state management:
        1. Update state to RUNNING
        2. Execute coroutine
        3. Update state to COMPLETED (on success) or FAILED (on error)
        4. Store result or error message

        Args:
            task_id: Task identifier
            coro: Coroutine to execute

        Handles:
            - asyncio.CancelledError: Sets state to CANCELLED
            - Exception: Sets state to FAILED, captures error message
        """
```

---

## Dependencies

### Component Dependencies

**Internal (zapomni_core):**
- None - TaskManager is standalone

### External Libraries

**Core:**
- `asyncio` (standard library) - For async task execution and queues
- `dataclasses` (standard library) - For TaskInfo data model
- `datetime` (standard library) - For timestamps
- `enum` (standard library) - For TaskState enum

**Logging:**
- `structlog>=23.2.0` - Structured logging for task lifecycle events

### Dependency Injection

TaskManager has no external dependencies to inject. It's a self-contained component instantiated by MemoryProcessor:

```python
from zapomni_core.tasks import TaskManager

class MemoryProcessor:
    def __init__(self, ...):
        self.task_manager = TaskManager(
            max_concurrent=5,
            task_ttl=3600
        )
```

---

## State Management

### Attributes

**tasks: dict[str, TaskInfo]**
- **Type:** Dictionary mapping task_id (str) to TaskInfo
- **Purpose:** Central state store for all task metadata
- **Lifetime:** Entries removed after task_ttl seconds (for finished tasks)
- **Concurrency:** Modified only within event loop (no locking needed)

**queue: asyncio.Queue**
- **Type:** asyncio.Queue[tuple[str, Coroutine]]
- **Purpose:** Pending tasks waiting for execution slot
- **Lifetime:** Persistent throughout TaskManager lifetime
- **Concurrency:** Thread-safe (asyncio guarantees)

**max_concurrent: int**
- **Type:** int (constant after init)
- **Purpose:** Maximum parallel tasks allowed
- **Default:** 5
- **Constraints:** Must be >= 1

**task_ttl: int**
- **Type:** int (seconds)
- **Purpose:** Time-to-live for completed tasks before cleanup
- **Default:** 3600 (1 hour)
- **Constraints:** Must be >= 0

**running_tasks: set[str]**
- **Type:** Set of task_id strings
- **Purpose:** Track currently executing tasks for concurrency control
- **Lifetime:** Entries added when task starts, removed when finishes
- **Max size:** Bounded by max_concurrent

### State Transitions

```
         submit_task()
              ↓
      ┌─────────────┐
      │   PENDING   │ ──cancel_task()──→ CANCELLED
      └──────┬──────┘
             │ (execution slot available)
             ↓
      ┌─────────────┐
      │   RUNNING   │ ──cancel_task()──→ CANCELLED
      └──────┬──────┘
             │
        ┌────┴────┐
        │         │
    (success) (error)
        │         │
        ↓         ↓
   ┌─────────┐ ┌─────────┐
   │COMPLETED│ │ FAILED  │
   └─────────┘ └─────────┘
        │         │
        └────┬────┘
             │ (after task_ttl)
             ↓
      [Task Removed]
```

**State Invariants:**
- PENDING → RUNNING: Only when running_tasks.size < max_concurrent
- RUNNING → COMPLETED/FAILED/CANCELLED: Only when task finishes
- COMPLETED/FAILED/CANCELLED: Terminal states (no further transitions)
- Progress: 0% (PENDING) → 0-99% (RUNNING) → 100% (COMPLETED/FAILED/CANCELLED)

### Thread Safety

**Is TaskManager thread-safe?** No

**Concurrency Constraints:**
- All methods must be called from the same asyncio event loop
- Not safe for use with threading.Thread
- Safe for concurrent asyncio tasks within same event loop

**Rationale:**
- TaskManager uses asyncio primitives (Queue, Task) which are event-loop-bound
- No need for thread safety in single-threaded async context
- Adding locks would hurt performance without benefit

---

## Public Methods (Detailed)

### Method 1: `submit_task`

**Signature:**
```python
async def submit_task(
    self,
    task_id: str,
    coro: Coroutine,
    metadata: Optional[Dict[str, Any]] = None
) -> str
```

**Purpose:** Submit an async coroutine for background execution without blocking caller

**Parameters:**

- `task_id`: str
  - Description: Unique identifier for task (used for status queries and cancellation)
  - Constraints: Must be non-empty, unique (not already in tasks dict), max 256 chars
  - Example: "build_graph_20251123_143022"

- `coro`: Coroutine
  - Description: Async coroutine to execute (e.g., `build_knowledge_graph()`)
  - Constraints: Must be awaitable, should handle CancelledError for graceful shutdown
  - Example: `entity_extractor.extract_all(memory_ids=["id1", "id2"])`

- `metadata`: Optional[dict[str, Any]]
  - Description: Optional task metadata (task_type, description, user_id, etc.)
  - Structure:
    ```python
    {
        "task_type": "build_graph",  # Optional
        "description": "Build knowledge graph for 1000 documents",  # Optional
        "total_items": 1000  # Optional, for progress calculation
    }
    ```
  - Default: None (empty dict)

**Returns:**
- Type: `str`
- Value: The same `task_id` provided (for convenience in chaining)

**Raises:**
- `ValueError`: When task_id is empty, too long, or already exists
- `TaskError`: When task queue is full (running_tasks.size >= max_concurrent and queue is full)

**Preconditions:**
- TaskManager must be initialized
- task_id must not exist in tasks dict
- Event loop must be running

**Postconditions:**
- TaskInfo created with state=PENDING, created_at=now()
- Task added to tasks dict
- Coroutine scheduled for execution (or queued if at max_concurrent limit)
- Returns immediately without blocking

**Algorithm Outline:**
```
FUNCTION submit_task(task_id, coro, metadata):
    # Step 1: Validate task_id
    IF task_id is empty OR task_id in tasks:
        RAISE ValueError("Invalid or duplicate task_id")

    # Step 2: Create TaskInfo
    task_info = TaskInfo(
        task_id=task_id,
        state=PENDING,
        progress=0,
        created_at=now(),
        metadata=metadata or {}
    )

    # Step 3: Add to tasks dict
    tasks[task_id] = task_info

    # Step 4: Schedule execution
    IF running_tasks.size < max_concurrent:
        # Execute immediately
        asyncio.create_task(_execute_task(task_id, coro))
    ELSE:
        # Queue for later
        IF queue.full():
            RAISE TaskError("Task queue is full")
        queue.put_nowait((task_id, coro))

    # Step 5: Return task_id
    RETURN task_id
END FUNCTION
```

**Edge Cases:**

1. **Duplicate task_id**: ValueError with message "Task with ID 'xyz' already exists"
2. **Empty task_id**: ValueError with message "task_id cannot be empty"
3. **Queue full**: TaskError with message "Task queue is full. Please try again later or cancel existing tasks."
4. **Coroutine already awaited**: Will fail during execution with error "cannot reuse already awaited coroutine"

**Related Methods:**
- Calls: `_execute_task()` (internal)
- Called by: `MemoryProcessor.build_knowledge_graph()`, `MemoryProcessor.codify_repository()`

---

### Method 2: `get_task_status`

**Signature:**
```python
async def get_task_status(self, task_id: str) -> TaskInfo
```

**Purpose:** Retrieve current status and metadata of a task

**Parameters:**

- `task_id`: str
  - Description: Task identifier to query
  - Constraints: Must exist in tasks dict
  - Example: "build_graph_20251123_143022"

**Returns:**
- Type: `TaskInfo`
- Fields:
  - `task_id`: str - Task identifier
  - `state`: TaskState - Current state (PENDING, RUNNING, COMPLETED, FAILED, CANCELLED)
  - `progress`: int - Completion percentage (0-100)
  - `created_at`: datetime - When task was submitted
  - `started_at`: Optional[datetime] - When task started execution
  - `completed_at`: Optional[datetime] - When task finished
  - `result`: Optional[Any] - Task result (if COMPLETED)
  - `error`: Optional[str] - Error message (if FAILED)
  - `metadata`: dict - Original metadata

**Raises:**
- `TaskNotFoundError`: When task_id doesn't exist in tasks dict

**Preconditions:**
- task_id must exist in tasks dict

**Postconditions:**
- No state mutation (read-only operation)
- Returns complete snapshot of task state

**Algorithm Outline:**
```
FUNCTION get_task_status(task_id):
    # Step 1: Check existence
    IF task_id NOT in tasks:
        RAISE TaskNotFoundError(f"Task '{task_id}' not found")

    # Step 2: Return task info
    RETURN tasks[task_id]
END FUNCTION
```

**Edge Cases:**

1. **Task doesn't exist**: TaskNotFoundError
2. **Task removed by cleanup**: TaskNotFoundError (cleanup removed old tasks)

**Related Methods:**
- Called by: MCP tools (get_graph_status), web UI status polling

---

### Method 3: `cancel_task`

**Signature:**
```python
async def cancel_task(self, task_id: str) -> bool
```

**Purpose:** Cancel a running or pending task gracefully

**Parameters:**

- `task_id`: str
  - Description: Task identifier to cancel
  - Constraints: Must exist in tasks dict
  - Example: "build_graph_20251123_143022"

**Returns:**
- Type: `bool`
- Value:
  - `True`: Task was successfully cancelled (was PENDING or RUNNING)
  - `False`: Task was already finished (COMPLETED, FAILED, or CANCELLED)

**Raises:**
- `TaskNotFoundError`: When task_id doesn't exist

**Preconditions:**
- task_id must exist in tasks dict

**Postconditions:**
- If task was PENDING: state → CANCELLED, removed from queue
- If task was RUNNING: asyncio.Task.cancel() called, state → CANCELLED after task handles CancelledError
- If task was already finished: No state change

**Algorithm Outline:**
```
FUNCTION cancel_task(task_id):
    # Step 1: Check existence
    IF task_id NOT in tasks:
        RAISE TaskNotFoundError(f"Task '{task_id}' not found")

    task_info = tasks[task_id]

    # Step 2: Check if already finished
    IF task_info.state in [COMPLETED, FAILED, CANCELLED]:
        RETURN False

    # Step 3: Cancel based on state
    IF task_info.state == PENDING:
        # Remove from queue
        queue.remove((task_id, coro))  # Note: requires custom queue implementation
        task_info.state = CANCELLED
        task_info.completed_at = now()
        RETURN True

    IF task_info.state == RUNNING:
        # Cancel asyncio.Task
        asyncio_task = running_tasks_map[task_id]
        asyncio_task.cancel()
        # State will be updated to CANCELLED by _execute_task when CancelledError caught
        RETURN True
END FUNCTION
```

**Edge Cases:**

1. **Task doesn't exist**: TaskNotFoundError
2. **Task already completed**: Returns False, no state change
3. **Task doesn't handle CancelledError**: Task may leave resources uncleaned (documented in TaskManager usage guide)

**Related Methods:**
- Calls: `asyncio.Task.cancel()` (stdlib)
- Called by: MCP tools, user-initiated cancellations

---

### Method 4: `get_all_tasks`

**Signature:**
```python
async def get_all_tasks(
    self,
    state_filter: Optional[TaskState] = None
) -> List[TaskInfo]
```

**Purpose:** Retrieve all tasks, optionally filtered by state

**Parameters:**

- `state_filter`: Optional[TaskState]
  - Description: Filter tasks by state (e.g., TaskState.RUNNING)
  - Constraints: Must be valid TaskState enum value
  - Default: None (return all tasks)
  - Examples:
    - `TaskState.RUNNING` - only running tasks
    - `TaskState.COMPLETED` - only completed tasks
    - `None` - all tasks

**Returns:**
- Type: `List[TaskInfo]`
- Value: List of TaskInfo objects, sorted by `created_at` descending (newest first)
- Empty list if no tasks match filter

**Raises:**
- No exceptions raised (returns empty list if no tasks)

**Preconditions:**
- None (safe to call anytime)

**Postconditions:**
- No state mutation (read-only)
- Results sorted by created_at (descending)

**Algorithm Outline:**
```
FUNCTION get_all_tasks(state_filter):
    # Step 1: Get all tasks
    all_tasks = tasks.values()

    # Step 2: Apply filter if provided
    IF state_filter is not None:
        filtered_tasks = [t for t in all_tasks if t.state == state_filter]
    ELSE:
        filtered_tasks = all_tasks

    # Step 3: Sort by created_at (descending)
    sorted_tasks = sorted(filtered_tasks, key=lambda t: t.created_at, reverse=True)

    # Step 4: Return list
    RETURN sorted_tasks
END FUNCTION
```

**Edge Cases:**

1. **No tasks exist**: Returns empty list
2. **No tasks match filter**: Returns empty list
3. **Invalid state_filter**: Python will raise TypeError (not caught by TaskManager)

**Related Methods:**
- Called by: Dashboard UI (show all tasks), MCP tools (list running tasks)

---

## Error Handling

### Exceptions Defined

```python
# zapomni_core/tasks/exceptions.py

class TaskError(Exception):
    """Base exception for TaskManager errors."""
    pass

class TaskNotFoundError(TaskError):
    """Raised when task_id doesn't exist."""
    pass

class TaskQueueFullError(TaskError):
    """Raised when task queue is at capacity."""
    pass
```

### Error Recovery

**Retry Strategy:**
- TaskManager does NOT automatically retry failed tasks
- Caller (e.g., MemoryProcessor) can detect FAILED state and resubmit if desired
- Recommendation: Exponential backoff (1s, 2s, 4s) for transient failures

**Fallback Behavior:**
- If task fails, state → FAILED, error message stored in TaskInfo.error
- Caller retrieves error via get_task_status() and handles accordingly
- No automatic fallback (TaskManager is low-level component)

**Error Propagation:**
- Exceptions raised inside task coroutines are caught by _execute_task()
- Exception message stored in TaskInfo.error
- TaskState set to FAILED
- Exception does NOT propagate to TaskManager (prevents event loop crash)

---

## Usage Examples

### Basic Usage

```python
from zapomni_core.tasks import TaskManager, TaskState
import asyncio

# Initialize TaskManager
manager = TaskManager(max_concurrent=5, task_ttl=3600)

# Define a long-running task
async def process_documents(doc_ids: list[str]):
    total = len(doc_ids)
    for i, doc_id in enumerate(doc_ids):
        # Simulate processing
        await asyncio.sleep(1)
        # Update progress (note: requires progress callback mechanism)
        print(f"Processed {i+1}/{total} documents")
    return {"processed": total, "success": True}

# Submit task
task_id = await manager.submit_task(
    task_id="process_001",
    coro=process_documents(["doc1", "doc2", "doc3"]),
    metadata={"task_type": "document_processing", "total_items": 3}
)

print(f"Task submitted: {task_id}")

# Poll for completion
while True:
    status = await manager.get_task_status(task_id)
    print(f"State: {status.state}, Progress: {status.progress}%")

    if status.state in [TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED]:
        break

    await asyncio.sleep(1)

# Check result
if status.state == TaskState.COMPLETED:
    print(f"Success! Result: {status.result}")
elif status.state == TaskState.FAILED:
    print(f"Failed: {status.error}")
```

### Advanced Usage: Cancellation

```python
from zapomni_core.tasks import TaskManager, TaskState
import asyncio

manager = TaskManager()

# Submit a long task
async def long_task():
    try:
        for i in range(100):
            await asyncio.sleep(1)
            print(f"Step {i+1}/100")
    except asyncio.CancelledError:
        print("Task was cancelled, cleaning up...")
        # Cleanup resources here
        raise  # Re-raise to let TaskManager know task was cancelled

task_id = await manager.submit_task("long_001", long_task())

# Let it run for 5 seconds
await asyncio.sleep(5)

# Cancel the task
was_cancelled = await manager.cancel_task(task_id)
print(f"Cancellation successful: {was_cancelled}")

# Verify state
status = await manager.get_task_status(task_id)
assert status.state == TaskState.CANCELLED
```

### Advanced Usage: Progress Tracking

```python
from zapomni_core.tasks import TaskManager, TaskState
import asyncio

manager = TaskManager()

# Task with progress updates
async def build_graph_with_progress(manager, task_id, total_docs):
    """
    Example task that updates its own progress via manager.

    Note: TaskManager doesn't currently support progress callbacks,
    so task must update TaskInfo.progress directly (requires access to manager).
    """
    for i in range(total_docs):
        # Process document
        await asyncio.sleep(0.5)

        # Update progress
        progress = int((i + 1) / total_docs * 100)
        manager.tasks[task_id].progress = progress

    return {"entities_extracted": total_docs * 10}

task_id = "graph_build_001"
await manager.submit_task(
    task_id=task_id,
    coro=build_graph_with_progress(manager, task_id, total_docs=20),
    metadata={"task_type": "build_graph"}
)

# Monitor progress in real-time
while True:
    status = await manager.get_task_status(task_id)
    print(f"Progress: {status.progress}%")

    if status.state != TaskState.RUNNING:
        break

    await asyncio.sleep(2)
```

---

## Testing Approach

### Unit Tests Required

**Initialization:**
- `test_init_success()` - Normal initialization with default params
- `test_init_custom_params()` - Custom max_concurrent and task_ttl
- `test_init_invalid_max_concurrent()` - ValueError for max_concurrent < 1
- `test_init_invalid_ttl()` - ValueError for task_ttl < 0

**Task Submission:**
- `test_submit_task_success()` - Happy path, task created with PENDING state
- `test_submit_task_duplicate_id()` - ValueError for duplicate task_id
- `test_submit_task_empty_id()` - ValueError for empty task_id
- `test_submit_task_with_metadata()` - Metadata stored correctly
- `test_submit_task_queue_full()` - TaskQueueFullError when at capacity

**Status Queries:**
- `test_get_task_status_success()` - Returns correct TaskInfo
- `test_get_task_status_not_found()` - TaskNotFoundError for invalid task_id
- `test_get_all_tasks_empty()` - Returns empty list when no tasks
- `test_get_all_tasks_with_filter()` - Filters by state correctly
- `test_get_all_tasks_sorted()` - Results sorted by created_at descending

**Cancellation:**
- `test_cancel_pending_task()` - PENDING task cancelled successfully
- `test_cancel_running_task()` - RUNNING task cancelled gracefully
- `test_cancel_completed_task()` - Returns False for already completed task
- `test_cancel_not_found()` - TaskNotFoundError for invalid task_id

**Task Execution:**
- `test_execute_task_success()` - Task runs and state → COMPLETED
- `test_execute_task_failure()` - Exception caught, state → FAILED
- `test_execute_task_cancelled()` - CancelledError caught, state → CANCELLED

**Cleanup:**
- `test_cleanup_old_tasks()` - Removes tasks older than task_ttl
- `test_cleanup_preserves_recent()` - Doesn't remove recent tasks

### Mocking Strategy

**Mock asyncio components:**
- Mock `asyncio.create_task()` to avoid actual task execution in unit tests
- Mock `asyncio.Queue` for queue behavior testing
- Mock `asyncio.sleep()` to speed up time-based tests

**Mock coroutines:**
```python
# Example mock coroutine
async def mock_success_coro():
    await asyncio.sleep(0)  # Yield control
    return {"status": "success"}

async def mock_failure_coro():
    await asyncio.sleep(0)
    raise ValueError("Simulated error")
```

**Fixtures:**
```python
@pytest.fixture
def task_manager():
    """Create TaskManager instance for testing."""
    return TaskManager(max_concurrent=3, task_ttl=60)

@pytest.fixture
async def submitted_task(task_manager):
    """Submit a test task and return task_id."""
    async def dummy_task():
        await asyncio.sleep(0.1)
        return {"result": "done"}

    task_id = await task_manager.submit_task("test_001", dummy_task())
    return task_id
```

### Integration Tests

**Test with real async tasks:**
```python
@pytest.mark.asyncio
async def test_full_task_lifecycle():
    """Test complete task lifecycle: submit → run → complete."""
    manager = TaskManager()

    async def real_task():
        await asyncio.sleep(0.5)
        return {"data": "processed"}

    # Submit
    task_id = await manager.submit_task("real_001", real_task())

    # Initial state
    status = await manager.get_task_status(task_id)
    assert status.state in [TaskState.PENDING, TaskState.RUNNING]

    # Wait for completion
    for _ in range(10):
        status = await manager.get_task_status(task_id)
        if status.state == TaskState.COMPLETED:
            break
        await asyncio.sleep(0.1)

    # Verify completion
    assert status.state == TaskState.COMPLETED
    assert status.result == {"data": "processed"}
    assert status.progress == 100
```

**Test concurrency limits:**
```python
@pytest.mark.asyncio
async def test_max_concurrent_enforcement():
    """Test that max_concurrent limit is respected."""
    manager = TaskManager(max_concurrent=2)

    async def slow_task():
        await asyncio.sleep(1)

    # Submit 5 tasks
    task_ids = []
    for i in range(5):
        task_id = await manager.submit_task(f"task_{i}", slow_task())
        task_ids.append(task_id)

    # Check running count
    running = [t for t in await manager.get_all_tasks() if t.state == TaskState.RUNNING]
    assert len(running) <= 2  # Respects max_concurrent
```

---

## Performance Considerations

### Time Complexity

**submit_task:**
- O(1) - Dict insertion, queue enqueue

**get_task_status:**
- O(1) - Dict lookup

**cancel_task:**
- O(1) - Dict lookup, asyncio.Task.cancel()

**get_all_tasks:**
- O(n log n) - Sorting all tasks by created_at
- O(n) - If filtering by state

**_cleanup_old_tasks:**
- O(n) - Iterate all tasks, filter by age

### Space Complexity

**Memory Usage:**
- O(n) where n = number of tasks in tasks dict
- Each TaskInfo: ~500 bytes (rough estimate)
- 1000 tasks = ~500 KB
- Bounded by task_ttl (old tasks auto-removed)

**Queue Size:**
- O(max_concurrent) - Queue size limited by concurrency

### Optimization Opportunities

**Index by State:**
```python
# Current: O(n) to filter by state
tasks_by_state: dict[TaskState, set[str]] = {
    TaskState.RUNNING: {"task_1", "task_3"},
    TaskState.PENDING: {"task_2"}
}
# get_all_tasks(state_filter=RUNNING) becomes O(k) where k = tasks in that state
```

**Lazy Cleanup:**
- Current: Cleanup runs every 5 minutes (background task)
- Optimization: Cleanup on-demand when tasks dict exceeds threshold (e.g., 1000 tasks)

**Trade-offs:**
- More memory for indexes vs. faster queries
- Current implementation favors simplicity (no indexes)

---

## References

### Component Spec
- Module spec: [zapomni_core_module.md](/home/dev/zapomni/.spec-workflow/specs/level1/zapomni_core_module.md)

### Related Components
- MemoryProcessor (uses TaskManager)
- EntityExtractor (executed as task)
- GraphBuilder (executed as task)

### External Documentation
- Python asyncio: https://docs.python.org/3/library/asyncio.html
- asyncio.Task: https://docs.python.org/3/library/asyncio-task.html
- asyncio.Queue: https://docs.python.org/3/library/asyncio-queue.html

---

**Document Status:** Draft v1.0
**Created:** 2025-11-23
**Author:** Goncharenko Anton aka alienxs2
**Copyright:** Copyright (c) 2025 Goncharenko Anton aka alienxs2
**License:** MIT License

**Total Sections:** 12
**Total Methods:** 6 (4 public, 2 private)
**Total Test Scenarios:** 25+
**Ready for Review:** Yes
