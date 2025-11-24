"""
TaskManager-specific exceptions.

Defines error types for task management operations.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from zapomni_core.exceptions import ZapomniError


class TaskError(ZapomniError):
    """
    Base exception for TaskManager errors.

    Error Codes:
        TASK_001: Generic task error
        TASK_002: Task queue full
        TASK_003: Task not found

    Attributes:
        message: Human-readable error message
        error_code: Programmatic error code
        details: Additional context
        correlation_id: UUID for tracing
    """

    def __init__(self, message: str, error_code: str = "TASK_001", **kwargs):
        super().__init__(message=message, error_code=error_code, **kwargs)
        self.is_transient = False


class TaskNotFoundError(TaskError):
    """
    Raised when a task_id doesn't exist in the task manager.

    Error Code: TASK_003
    Not transient (invalid task ID should not be retried).
    """

    def __init__(self, message: str, **kwargs):
        super().__init__(message=message, error_code="TASK_003", **kwargs)
        self.is_transient = False


class TaskQueueFullError(TaskError):
    """
    Raised when task queue is at capacity.

    Error Code: TASK_002
    Transient (can retry after task completion).
    """

    def __init__(self, message: str, **kwargs):
        super().__init__(message=message, error_code="TASK_002", **kwargs)
        self.is_transient = True
