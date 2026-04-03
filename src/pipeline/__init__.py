"""Custom pipeline module — CUSTOMIZE this when forking.

Defines how raw samples are processed into the standardized format.
Called by ``core.pipeline.run_pipeline()``.
"""

from .pipeline import TaskPipeline
from .config import TaskConfig

__all__ = ["TaskPipeline", "TaskConfig"]
