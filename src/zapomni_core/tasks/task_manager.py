"""
TaskManager - Async background task management for Zapomni Core.

Provides asynchronous background task management for long-running operations
like knowledge graph construction and repository codification. Handles task
submission, status tracking, cancellation, and lifecycle management using asyncio.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

import structlog

from .exceptions import TaskError, TaskNotFoundError, TaskQueueFullError

logger = structlog.get_logger(__name__)


class TaskState(str, Enum):
    """
    Task execution states.

    Attributes:
        PENDING: Task queued, waiting for execution slot
        RUNNING: Task currently executing
        COMPLETED: Task finished successfully
        FAILED: Task failed with error
        CANCELLED: Task was cancelled by user
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskInfo:
    """
    Task metadata and status.

    Stores complete information about a task including its current state,
    execution timeline, result/error, and optional metadata.

    Attributes:
        task_id: Unique identifier for task
        state: Current execution state (TaskState enum)
        progress: Completion percentage (0-100)
        created_at: Timestamp when task was submitted
        started_at: Timestamp when task started execution (None if not started)
        completed_at: Timestamp when task finished (None if not finished)
        result: Task result data (None if not completed)
        error: Error message (None if no error)
        metadata: Additional task-specific metadata (e.g., task_type, description)
    """

    task_id: str
    state: TaskState
    progress: int
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskManager:
    """
    Manages asynchronous background tasks for long-running operations.

    Provides task submission, status tracking, cancellation, and lifecycle
    management using asyncio. Designed for operations like knowledge graph
    construction that would block the server if run synchronously.

    Attributes:
        tasks: Dictionary mapping task_id to TaskInfo
        _queue: Internal async queue for pending tasks
        _max_concurrent: Maximum number of concurrent tasks
        _task_ttl: Time-to-live for completed tasks (seconds)
        _running_tasks: Set of currently running task IDs
        _asyncio_tasks: Mapping of task_id to asyncio.Task objects

    Example:
        ```python
        from zapomni_core.tasks import TaskManager, TaskState

        # Initialize manager
        manager = TaskManager(max_concurrent=5, task_ttl=3600)

        # Submit a task
        async def build_graph():
            await asyncio.sleep(10)
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
        task_ttl: int = 3600,
    ) -> None:
        """
        Initialize TaskManager.

        Args:
            max_concurrent: Maximum concurrent tasks (default: 5)
                Must be >= 1
            task_ttl: Time-to-live for completed tasks in seconds (default: 3600)
                Must be >= 0. Completed/failed tasks older than TTL are removed.

        Raises:
            ValueError: If max_concurrent < 1 or task_ttl < 0
        """
        if max_concurrent < 1:
            raise ValueError(f"max_concurrent must be >= 1, got {max_concurrent}")
        if task_ttl < 0:
            raise ValueError(f"task_ttl must be >= 0, got {task_ttl}")

        self._max_concurrent = max_concurrent
        self._task_ttl = task_ttl
        self.tasks: Dict[str, TaskInfo] = {}
        self._running_tasks: Set[str] = set()
        self._asyncio_tasks: Dict[str, asyncio.Task] = {}
        # Queue can hold up to max_concurrent pending tasks
        self._queue: asyncio.Queue[tuple[str, Coroutine]] = asyncio.Queue(maxsize=max_concurrent)
        self._cleanup_task: Optional[asyncio.Task] = None
        self._worker_tasks: Set[asyncio.Task] = set()

        logger.info(
            "TaskManager initialized",
            max_concurrent=max_concurrent,
            task_ttl=task_ttl,
        )

    async def submit_task(
        self,
        task_id: str,
        coro: Coroutine,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Submit an async task for background execution.

        Creates a TaskInfo entry with PENDING state, schedules the coroutine
        for execution, and returns the task_id immediately for tracking.

        Args:
            task_id: Unique identifier for task (e.g., "build_graph_001")
                Must be non-empty and unique (not already in tasks dict)
                Max 256 characters
            coro: Async coroutine to execute
                Must be awaitable, should handle CancelledError for graceful shutdown
            metadata: Optional metadata (task_type, description, etc.)

        Returns:
            str: The same task_id provided (for convenience in chaining)

        Raises:
            ValueError: If task_id is empty, too long, or already exists
            TaskQueueFullError: If task queue is full (running_tasks.size >= max_concurrent)

        Preconditions:
            - TaskManager must be initialized
            - task_id must not exist in tasks dict
            - Event loop must be running

        Postconditions:
            - TaskInfo created with state=PENDING, created_at=now()
            - Task added to tasks dict
            - Coroutine scheduled for execution or queued
            - Returns immediately without blocking
        """
        # Step 1: Validate task_id
        if not task_id:
            raise ValueError("task_id cannot be empty")
        if len(task_id) > 256:
            raise ValueError(f"task_id must be <= 256 chars, got {len(task_id)}")
        if task_id in self.tasks:
            raise ValueError(f"Task with ID '{task_id}' already exists")

        # Step 2: Create TaskInfo with PENDING state
        now = datetime.now(timezone.utc)
        task_info = TaskInfo(
            task_id=task_id,
            state=TaskState.PENDING,
            progress=0,
            created_at=now,
            metadata=metadata or {},
        )

        # Step 3: Add to tasks dict
        self.tasks[task_id] = task_info

        # Step 4: Schedule execution
        try:
            if len(self._running_tasks) < self._max_concurrent:
                # Execute immediately
                await self._execute_task_wrapper(task_id, coro)
            else:
                # Queue for later (will be picked up by worker)
                try:
                    self._queue.put_nowait((task_id, coro))
                    # Ensure worker is running to process the queue
                    self._ensure_worker_running()
                except asyncio.QueueFull:
                    # Clean up the task entry since it won't be queued
                    del self.tasks[task_id]
                    raise TaskQueueFullError(
                        "Task queue is full. Please try again later or cancel existing tasks."
                    )
        except TaskQueueFullError:
            raise
        except Exception as e:
            logger.error(
                "Failed to submit task",
                task_id=task_id,
                error=str(e),
            )
            # Clean up on submission failure
            if task_id in self.tasks:
                del self.tasks[task_id]
            raise

        logger.info(
            "Task submitted",
            task_id=task_id,
            metadata=metadata,
        )

        return task_id

    async def get_task_status(self, task_id: str) -> TaskInfo:
        """
        Get current status of a task.

        Returns complete TaskInfo with state, progress, timestamps, result/error.

        Args:
            task_id: Task identifier to query

        Returns:
            TaskInfo: Object with current status

        Raises:
            TaskNotFoundError: If task_id doesn't exist

        Preconditions:
            - task_id must exist in tasks dict

        Postconditions:
            - No state mutation (read-only operation)
            - Returns complete snapshot of task state
        """
        if task_id not in self.tasks:
            raise TaskNotFoundError(f"Task '{task_id}' not found")

        return self.tasks[task_id]

    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running or pending task.

        If task is PENDING, removes from queue and marks CANCELLED.
        If task is RUNNING, sends cancellation signal to asyncio.Task.
        If task is already finished, returns False (no-op).

        Args:
            task_id: Task identifier to cancel

        Returns:
            bool: True if task was cancelled, False if already finished

        Raises:
            TaskNotFoundError: If task_id doesn't exist

        Preconditions:
            - task_id must exist in tasks dict

        Postconditions:
            - If task was PENDING: state → CANCELLED, removed from queue
            - If task was RUNNING: asyncio.Task.cancel() called, state → CANCELLED after task handles CancelledError
            - If task was already finished: No state change

        Note:
            Task must handle asyncio.CancelledError to clean up resources.
            TaskManager waits up to 5 seconds for graceful shutdown.
        """
        if task_id not in self.tasks:
            raise TaskNotFoundError(f"Task '{task_id}' not found")

        task_info = self.tasks[task_id]

        # If already finished, return False (no-op)
        if task_info.state in [
            TaskState.COMPLETED,
            TaskState.FAILED,
            TaskState.CANCELLED,
        ]:
            return False

        # If running, cancel the asyncio task
        if task_info.state == TaskState.RUNNING:
            if task_id in self._asyncio_tasks:
                asyncio_task = self._asyncio_tasks[task_id]
                asyncio_task.cancel()

                # Wait up to 5 seconds for graceful shutdown
                try:
                    await asyncio.wait_for(asyncio_task, timeout=5.0)
                except asyncio.CancelledError:
                    pass
                except asyncio.TimeoutError:
                    logger.warning(
                        "Task cancellation timeout",
                        task_id=task_id,
                    )

            logger.info("Task cancelled", task_id=task_id)
            return True

        # If pending, mark as cancelled and try to remove from queue
        # (Note: We can't easily remove from queue, so we mark as cancelled
        # and skip execution when dequeued)
        task_info.state = TaskState.CANCELLED
        task_info.completed_at = datetime.now(timezone.utc)

        logger.info("Pending task cancelled", task_id=task_id)
        return True

    async def get_all_tasks(
        self,
        state_filter: Optional[TaskState] = None,
    ) -> List[TaskInfo]:
        """
        Get list of all tasks, optionally filtered by state.

        Args:
            state_filter: Optional state to filter by (e.g., TaskState.RUNNING)
                Default: None (return all tasks)

        Returns:
            List[TaskInfo]: List of TaskInfo objects, sorted by created_at (descending)
                Empty list if no tasks match filter

        Preconditions:
            - None (safe to call anytime)

        Postconditions:
            - No state mutation (read-only)
            - Results sorted by created_at (descending)
        """
        # Get all tasks
        all_tasks = list(self.tasks.values())

        # Apply filter if provided
        if state_filter is not None:
            filtered_tasks = [t for t in all_tasks if t.state == state_filter]
        else:
            filtered_tasks = all_tasks

        # Sort by created_at (descending - newest first)
        sorted_tasks = sorted(
            filtered_tasks,
            key=lambda t: t.created_at,
            reverse=True,
        )

        return sorted_tasks

    async def shutdown(self) -> None:
        """
        Shutdown the TaskManager gracefully.

        Cancels all pending and running tasks, waits for worker tasks to finish.
        Should be called before application shutdown.

        Postconditions:
            - All pending tasks cancelled
            - All running tasks cancelled (with graceful CancelledError handling)
            - Worker tasks finished
            - Cleanup task cancelled
        """
        logger.info("TaskManager shutting down")

        # Cancel all pending and running tasks
        task_ids = list(self.tasks.keys())
        for task_id in task_ids:
            try:
                await self.cancel_task(task_id)
            except TaskNotFoundError:
                pass

        # Cancel cleanup task if running
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Wait for worker tasks to finish
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)

        logger.info("TaskManager shutdown complete")

    # ========================================================================
    # Private Methods
    # ========================================================================

    def _ensure_worker_running(self) -> None:
        """
        Ensure a worker task is running.

        Non-async method that spawns a worker if needed.
        """
        # Check if worker is already running
        if self._worker_tasks:
            # Check if any worker is still alive
            for task in list(self._worker_tasks):
                if task.done():
                    self._worker_tasks.discard(task)
            if self._worker_tasks:
                return  # Worker still running

        # Spawn a new worker
        try:
            worker_task = asyncio.create_task(self._worker_loop())
            self._worker_tasks.add(worker_task)
            worker_task.add_done_callback(self._worker_tasks.discard)
        except RuntimeError:
            # No event loop running, this is fine - will create when needed
            pass

    async def _execute_task_wrapper(
        self,
        task_id: str,
        coro: Coroutine,
    ) -> None:
        """
        Wrapper that executes a task and manages its lifecycle.

        Creates an asyncio.Task for the coroutine and tracks it.

        Args:
            task_id: Task identifier
            coro: Coroutine to execute
        """
        # Create asyncio task to track it
        asyncio_task = asyncio.create_task(self._execute_task(task_id, coro))
        self._asyncio_tasks[task_id] = asyncio_task

        # Ensure worker is running to handle queued tasks
        self._ensure_worker_running()

    async def _execute_task(
        self,
        task_id: str,
        coro: Coroutine,
    ) -> None:
        """
        Execute a task coroutine and update its status.

        Internal method that wraps the coroutine with state management:
        1. Update state to RUNNING, set started_at
        2. Execute coroutine
        3. Update state to COMPLETED, set result (on success)
        4. Update state to FAILED, set error (on exception)
        5. Handle asyncio.CancelledError for cancellation

        Args:
            task_id: Task identifier
            coro: Coroutine to execute

        Handles:
            - asyncio.CancelledError: Sets state to CANCELLED
            - Exception: Sets state to FAILED, captures error message
        """
        if task_id not in self.tasks:
            logger.error("Task not found during execution", task_id=task_id)
            return

        task_info = self.tasks[task_id]

        try:
            # Step 1: Mark as running
            task_info.state = TaskState.RUNNING
            task_info.started_at = datetime.now(timezone.utc)
            self._running_tasks.add(task_id)

            logger.info("Task execution started", task_id=task_id)

            # Step 2: Execute coroutine
            result = await coro

            # Step 3: Mark as completed
            task_info.state = TaskState.COMPLETED
            task_info.result = result
            task_info.progress = 100
            task_info.completed_at = datetime.now(timezone.utc)

            logger.info(
                "Task execution completed",
                task_id=task_id,
                result=result,
            )

        except asyncio.CancelledError:
            # Task was cancelled
            task_info.state = TaskState.CANCELLED
            task_info.completed_at = datetime.now(timezone.utc)
            logger.info("Task execution cancelled", task_id=task_id)

        except Exception as e:
            # Task failed with error
            task_info.state = TaskState.FAILED
            task_info.error = str(e)
            task_info.completed_at = datetime.now(timezone.utc)

            logger.error(
                "Task execution failed",
                task_id=task_id,
                error=str(e),
                exc_info=True,
            )

        finally:
            # Clean up running task tracking
            self._running_tasks.discard(task_id)
            self._asyncio_tasks.pop(task_id, None)

    async def _worker_loop(self) -> None:
        """
        Worker loop that processes queued tasks.

        Continuously dequeues pending tasks and executes them when slots
        become available (i.e., when running_tasks.size < max_concurrent).

        Runs until cancelled.
        """
        try:
            while True:
                # Wait for a slot to become available
                while len(self._running_tasks) >= self._max_concurrent:
                    await asyncio.sleep(0.1)

                # Get next queued task (with timeout to allow cancellation)
                try:
                    task_id, coro = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=1.0,
                    )
                except asyncio.TimeoutError:
                    # Queue empty, try again
                    continue

                # Check if task was cancelled while queued
                if task_id in self.tasks:
                    task_info = self.tasks[task_id]
                    if task_info.state == TaskState.CANCELLED:
                        logger.info(
                            "Skipping cancelled queued task",
                            task_id=task_id,
                        )
                        continue

                    # Execute the task
                    await self._execute_task_wrapper(task_id, coro)
                else:
                    logger.warning(
                        "Queued task no longer in tasks dict",
                        task_id=task_id,
                    )

        except asyncio.CancelledError:
            logger.info("Worker loop cancelled")

    async def _cleanup_old_tasks(self) -> int:
        """
        Remove completed/failed tasks older than task_ttl.

        Internal method called periodically (every 5 minutes).

        Returns:
            int: Number of tasks removed

        Performance:
            - Execution time: < 50ms for 1000 tasks
        """
        now = datetime.now(timezone.utc)
        removed_count = 0
        task_ids_to_remove = []

        # Find tasks older than TTL
        for task_id, task_info in self.tasks.items():
            if task_info.state in [TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED]:
                if task_info.completed_at:
                    age_seconds = (now - task_info.completed_at).total_seconds()
                    if age_seconds > self._task_ttl:
                        task_ids_to_remove.append(task_id)

        # Remove old tasks
        for task_id in task_ids_to_remove:
            del self.tasks[task_id]
            removed_count += 1

        if removed_count > 0:
            logger.info(
                "Cleaned up old tasks",
                removed_count=removed_count,
                task_ttl=self._task_ttl,
            )

        return removed_count
