"""
TaskManager - Async background task management for Zapomni Core.

Provides task submission, status tracking, cancellation, and lifecycle management
using asyncio for long-running operations like knowledge graph construction.

Author: Goncharenko Anton aka alienxs2
License: MIT
"""

from .exceptions import TaskError, TaskNotFoundError, TaskQueueFullError
from .task_manager import TaskInfo, TaskManager, TaskState

__all__ = [
    "TaskManager",
    "TaskState",
    "TaskInfo",
    "TaskError",
    "TaskNotFoundError",
    "TaskQueueFullError",
]
