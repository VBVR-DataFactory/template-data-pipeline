"""Download module — handles fetching data from external sources."""

from pathlib import Path
from typing import Iterator, Optional
from datasets import load_dataset


class HuggingFaceDownloader:
    """Download datasets from HuggingFace Hub into the ``raw/`` directory.

    Wraps ``datasets.load_dataset`` with convenience helpers for streaming
    and limiting the number of samples.
    """

    def __init__(self, repo_id: str, split: str = "test", raw_dir: Path = Path("raw")):
        self.repo_id = repo_id
        self.split = split
        self.raw_dir = Path(raw_dir)

    def download(self, limit: Optional[int] = None) -> Iterator[dict]:
        """Download dataset and yield raw samples.

        Raw data is cached under ``self.raw_dir``.

        Args:
            limit: Maximum number of samples to yield (None = all).

        Yields:
            Raw sample dicts straight from HuggingFace.
        """
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {self.repo_id} (split: {self.split}) → {self.raw_dir}/")
        dataset = load_dataset(
            self.repo_id,
            split=self.split,
            cache_dir=str(self.raw_dir / ".cache"),
        )

        if limit is not None:
            dataset = dataset.select(range(min(limit, len(dataset))))

        print(f"Streaming {len(dataset)} samples...")

        for item in dataset:
            yield item
