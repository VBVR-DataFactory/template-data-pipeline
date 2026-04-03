"""Download module — handles fetching data from external sources."""

from typing import Iterator, Optional
from datasets import load_dataset


class HuggingFaceDownloader:
    """Download datasets from HuggingFace Hub.

    Wraps ``datasets.load_dataset`` with convenience helpers for streaming
    and limiting the number of samples.
    """

    def __init__(self, repo_id: str, split: str = "test"):
        self.repo_id = repo_id
        self.split = split

    def download(self, limit: Optional[int] = None) -> Iterator[dict]:
        """Download dataset and yield raw samples.

        Args:
            limit: Maximum number of samples to yield (None = all).

        Yields:
            Raw sample dicts straight from HuggingFace.
        """
        print(f"Downloading {self.repo_id} (split: {self.split})...")
        dataset = load_dataset(self.repo_id, split=self.split)

        if limit is not None:
            dataset = dataset.select(range(min(limit, len(dataset))))

        print(f"Streaming {len(dataset)} samples...")

        for item in dataset:
            yield item
