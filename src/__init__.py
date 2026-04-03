"""Task-specific modules — CUSTOMIZE these when forking.

Two sub-modules:
    src.download  — custom download logic
    src.pipeline  — custom processing logic
"""

from .pipeline import TaskPipeline, TaskConfig
from .download import create_downloader, TaskDownloader

__all__ = [
    "TaskPipeline",
    "TaskConfig",
    "create_downloader",
    "TaskDownloader",
]
