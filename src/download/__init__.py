"""Custom download module — CUSTOMIZE this when forking.

Defines how raw data is fetched from your dataset source.
Called by ``core.download.run_download()``.
"""

from .downloader import create_downloader, TaskDownloader

__all__ = ["create_downloader", "TaskDownloader"]
