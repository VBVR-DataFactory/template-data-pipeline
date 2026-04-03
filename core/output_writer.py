"""Output writer — writes TaskSamples to the standardized folder structure."""

import json
import shutil
from pathlib import Path
from typing import List

from .schemas import TaskSample


class OutputWriter:
    """Writes task samples to the standardized directory layout.

    Produces the same on-disk structure as template-data-generator::

        data/questions/{domain}_task/{task_id}/
        ├── first_frame.png
        ├── final_frame.png
        ├── prompt.txt
        ├── first_video.mp4
        ├── last_video.mp4
        ├── ground_truth.mp4
        └── metadata.json
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_sample(self, sample: TaskSample) -> Path:
        """Write a single sample to disk."""
        task_dir = self.output_dir / f"{sample.domain}_task" / sample.task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        # Write images
        if sample.first_image is not None:
            sample.first_image.save(task_dir / "first_frame.png")

        if sample.final_image is not None:
            sample.final_image.save(task_dir / "final_frame.png")

        # Write prompt
        (task_dir / "prompt.txt").write_text(sample.prompt)

        # Write videos if provided
        for video_path, filename in [
            (sample.first_video, "first_video.mp4"),
            (sample.last_video, "last_video.mp4"),
            (sample.ground_truth_video, "ground_truth.mp4"),
        ]:
            if video_path is not None and Path(video_path).exists():
                video_src = Path(video_path)
                shutil.copy(video_src, task_dir / filename)

        # Write metadata
        if sample.metadata is not None:
            (task_dir / "metadata.json").write_text(
                json.dumps(sample.metadata, ensure_ascii=False, indent=2)
            )

        return task_dir

    def write_dataset(self, samples: List[TaskSample]) -> Path:
        """Write all samples to disk."""
        for sample in samples:
            self.write_sample(sample)
        return self.output_dir
