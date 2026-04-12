"""CoreCognition downloader for Video-MCP style MCQA samples."""

import csv
import io
import os
import re
import zipfile
from pathlib import Path
from typing import Any, Iterator, Optional

from huggingface_hub import hf_hub_download


_IMAGE_PLACEHOLDER_RE = re.compile(
    r"^<image-placeholder:\s*([^>]+)>\s*", flags=re.IGNORECASE
)
_VIDEO_PLACEHOLDER_RE = re.compile(r"<video-placeholder:", flags=re.IGNORECASE)
_CHOICES_KV_RE = re.compile(
    r"""(?P<k>['"]?[A-Z]['"]?)\s*:\s*(?P<v>nan|None|['"].*?['"])""",
    flags=re.IGNORECASE,
)


def _split_single_image(images: str | None) -> str | None:
    if images is None:
        return None
    parts = [p.strip() for p in str(images).split(";") if p.strip()]
    if len(parts) != 1:
        return None
    return parts[0]


def _strip_image_placeholder(question: str) -> tuple[str | None, str]:
    q = str(question or "")
    match = _IMAGE_PLACEHOLDER_RE.match(q)
    if match is None:
        return None, q.strip()
    return match.group(1).strip(), q[match.end() :].strip()


def _parse_choices(raw: Any) -> dict[str, str]:
    text = str(raw or "").strip()
    parsed: dict[str, str] = {}
    for match in _CHOICES_KV_RE.finditer(text):
        key = match.group("k").strip().strip("'").strip('"').upper()
        value = match.group("v").strip()
        if not key or value.lower() in {"nan", "none"}:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key in {"A", "B", "C", "D"}:
            parsed[key] = value
    return parsed


class TaskDownloader:
    """Downloads and filters CoreCognition MCQA single-image examples."""

    def __init__(self, config):
        self.config = config
        self.raw_dir = Path("raw")
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def _download_zip(self) -> Path:
        token = os.environ.get("HF_TOKEN") or None
        local_dir = self.raw_dir / "corecognition"
        local_dir.mkdir(parents=True, exist_ok=True)
        zip_path = hf_hub_download(
            repo_id=self.config.hf_repo,
            repo_type="dataset",
            filename=self.config.hf_zip_filename,
            token=token,
            local_dir=str(local_dir),
        )
        return Path(zip_path)

    def download(self, limit: Optional[int] = None) -> Iterator[dict]:
        """Yield filtered rows with image bytes for rendering."""
        zip_path = self._download_zip()
        csv_name = "CoreCognition_20250622/CoreCognition.csv"
        yielded = 0

        with zipfile.ZipFile(zip_path) as archive:
            available = set(archive.namelist())
            rows = archive.read(csv_name).decode("utf-8", errors="replace")
            reader = csv.DictReader(io.StringIO(rows))

            for row in reader:
                if str(row.get("type", "")).strip().upper() != "MC":
                    continue
                if str(row.get("videos", "")).strip():
                    continue
                if _VIDEO_PLACEHOLDER_RE.search(str(row.get("question", ""))):
                    continue

                img_col = _split_single_image(row.get("images"))
                img_q, question = _strip_image_placeholder(row.get("question", ""))
                image_name = img_col or img_q
                if not image_name:
                    continue

                choices = _parse_choices(row.get("choices"))
                answer = str(row.get("answer", "")).strip().upper()
                if not choices or answer not in {"A", "B", "C", "D"} or answer not in choices:
                    continue

                media_path = f"CoreCognition_20250622/media/{image_name}"
                if media_path not in available:
                    continue

                yield {
                    "dataset": "CoreCognition",
                    "source_id": str(row.get("id", "")).strip(),
                    "question": question,
                    "choices": choices,
                    "answer": answer,
                    "image_filename": Path(image_name).name,
                    "image_bytes": archive.read(media_path),
                }
                yielded += 1
                if limit is not None and yielded >= limit:
                    break


def create_downloader(config) -> TaskDownloader:
    """Factory called by ``core.download.run_download()``."""
    return TaskDownloader(config)
