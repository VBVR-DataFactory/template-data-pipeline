"""Core download module — fetch raw data from external sources.

Provides generic download utilities (HuggingFace, S3). The actual
dataset-specific download logic lives in ``src.download``; this module
always delegates to it via :func:`run_download`.
"""

from pathlib import Path
from typing import Iterator, Optional

import boto3
from datasets import load_dataset


# ============================================================================
#  HuggingFace downloader
# ============================================================================

class HuggingFaceDownloader:
    """Download datasets from HuggingFace Hub into the ``raw/`` directory.

    Wraps ``datasets.load_dataset`` with convenience helpers for streaming
    and limiting the number of samples.
    """

    def __init__(self, repo_id: str, split: Optional[str] = None, raw_dir: Path = Path("raw")):
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
        cache_dir = str(self.raw_dir / ".cache")

        if self.split is None:
            print(f"Downloading {self.repo_id} (split: auto) → {self.raw_dir}/")
            dataset_obj = load_dataset(self.repo_id, cache_dir=cache_dir)
            if hasattr(dataset_obj, "keys"):
                # DatasetDict-like: choose a reasonable default split.
                candidates = ("train", "test", "validation")
                chosen = next((c for c in candidates if c in dataset_obj), None)
                if chosen is None:
                    keys = list(dataset_obj.keys())
                    chosen = keys[0] if keys else None
                if chosen is None:
                    raise ValueError(f"No splits found for dataset: {self.repo_id}")
                self.split = chosen
                print(f"Using split: {self.split}")
                dataset = dataset_obj[self.split]
            else:
                dataset = dataset_obj
        else:
            print(f"Downloading {self.repo_id} (split: {self.split}) → {self.raw_dir}/")
            dataset = load_dataset(
                self.repo_id,
                split=self.split,
                cache_dir=cache_dir,
            )

        if limit is not None:
            dataset = dataset.select(range(min(limit, len(dataset))))

        print(f"Streaming {len(dataset)} samples...")

        for item in dataset:
            yield item


# ============================================================================
#  S3 upload / download utilities
# ============================================================================

def upload_directory_to_s3(
    local_dir: Path, bucket_name: str, s3_prefix: str = ""
) -> tuple:
    """Upload entire directory to S3, preserving structure.

    Args:
        local_dir: Local directory to upload.
        bucket_name: S3 bucket name.
        s3_prefix: Prefix for S3 keys (e.g. ``'datasets/'``).

    Returns:
        ``(uploaded, failed)`` counts.
    """
    s3_client = boto3.client("s3")

    files = [f for f in Path(local_dir).rglob("*") if f.is_file()]
    print(f"Found {len(files)} files to upload...")

    uploaded = 0
    failed = 0

    content_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".json": "application/json",
        ".txt": "text/plain",
        ".mp4": "video/mp4",
    }

    for file_path in files:
        relative_path = file_path.relative_to(local_dir)
        s3_key = f"{s3_prefix}{relative_path}".replace("\\", "/")
        ct = content_types.get(file_path.suffix.lower(), "application/octet-stream")

        try:
            s3_client.upload_file(
                str(file_path), bucket_name, s3_key, ExtraArgs={"ContentType": ct}
            )
            uploaded += 1
            if uploaded % 10 == 0:
                print(f"Uploaded {uploaded}/{len(files)} files...")
        except Exception:
            failed += 1
            print(f"Failed to upload: {file_path}")

    print(f"\n✓ Upload complete: {uploaded} successful, {failed} failed")
    return uploaded, failed


def download_from_s3(
    bucket_name: str, s3_prefix: str, local_dir: Path
) -> int:
    """Download dataset from S3 to local directory.

    Args:
        bucket_name: S3 bucket name.
        s3_prefix: S3 prefix to download from.
        local_dir: Local directory to save files.

    Returns:
        Number of files downloaded.
    """
    s3_client = boto3.client("s3")
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading from s3://{bucket_name}/{s3_prefix}...")

    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix)

    files = []
    for page in pages:
        if "Contents" in page:
            files.extend(page["Contents"])

    print(f"Found {len(files)} files to download...")

    downloaded = 0
    for obj in files:
        s3_key = obj["Key"]
        relative_path = s3_key.replace(s3_prefix, "", 1).lstrip("/")
        local_path = local_dir / relative_path
        local_path.parent.mkdir(parents=True, exist_ok=True)

        s3_client.download_file(bucket_name, s3_key, str(local_path))
        downloaded += 1

        if downloaded % 10 == 0:
            print(f"Downloaded {downloaded}/{len(files)} files...")

    print(f"\n✓ Download complete: {downloaded} files")
    return downloaded


# ============================================================================
#  Orchestration — delegates to src.download
# ============================================================================

def run_download(config) -> Iterator[dict]:
    """Standard download entry point.

    Imports and calls the custom downloader defined in ``src.download``.

    Args:
        config: A :class:`PipelineConfig` (or subclass) instance.

    Yields:
        Raw sample dicts from the custom downloader.
    """
    from src.download import create_downloader

    downloader = create_downloader(config)
    yield from downloader.download(limit=config.num_samples)
