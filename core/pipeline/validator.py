"""Validator utilities for checking data consistency with standardized format.

Standardized format::

    data/questions/{domain}_task/{task_id}/
    ├── first_frame.png          (required)
    ├── final_frame.png          (optional)
    ├── prompt.txt               (required)
    ├── first_video.mp4          (optional)
    ├── last_video.mp4           (optional)
    └── ground_truth.mp4         (optional)
"""

from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image


def validate_task_data(
    first_frame: Image.Image,
    prompt: str,
    final_frame: Optional[Image.Image] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """Validate task data matches standardized format.

    Args:
        first_frame: Initial state image (required).
        prompt: Natural language instruction (required).
        final_frame: Final state image (optional).
        metadata: Task metadata dict (optional).

    Returns:
        ``True`` if valid.
    """
    if first_frame is None:
        return False

    if not prompt or not prompt.strip():
        return False

    return True


def validate_task_directory(task_dir: Path) -> bool:
    """Validate task directory structure.

    Args:
        task_dir: Path to task directory.

    Returns:
        ``True`` if the required files are present.
    """
    if not task_dir.exists() or not task_dir.is_dir():
        return False

    if not (task_dir / "first_frame.png").exists():
        return False

    if not (task_dir / "prompt.txt").exists():
        return False

    return True
