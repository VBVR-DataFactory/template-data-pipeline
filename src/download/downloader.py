"""
Your download implementation.

CUSTOMIZE THIS FILE to define how your dataset is downloaded.
The ``create_downloader`` factory is called by ``core.download.run_download()``.
"""

from pathlib import Path
from typing import Iterator, Optional

from core.download import HuggingFaceDownloader


class TaskDownloader:
    """
    VideoThinkBench downloader — fetches data from HuggingFace.

    REPLACE THIS with your own download logic when forking.
    """

    def __init__(self, config):
        self.config = config
        self.hf_downloader = HuggingFaceDownloader(
            repo_id=config.hf_repo,
            split=config.split,
            raw_dir=Path("raw"),
        )

    def download(self, limit: Optional[int] = None) -> Iterator[dict]:
        """Download dataset and yield raw samples."""
        yield from self.hf_downloader.download(limit=limit)


def create_downloader(config) -> TaskDownloader:
    """Factory called by ``core.download.run_download()``.

    Args:
        config: A pipeline config instance (typically your TaskConfig).

    Returns:
        A downloader instance with a ``.download(limit=...)`` method.
    """
    return TaskDownloader(config)
