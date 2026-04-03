"""Task-specific pipeline — CUSTOMIZE these files when forking."""

from .pipeline import TaskPipeline
from .config import TaskConfig

__all__ = ["TaskPipeline", "TaskConfig"]
