"""
Your pipeline implementation.

CUSTOMIZE THIS FILE to define how your dataset is downloaded and processed.
Subclasses BasePipeline and implements download() + process_sample().
"""

from typing import Iterator, Optional

from core import BasePipeline, HuggingFaceDownloader, SampleProcessor, TaskSample
from .config import TaskConfig
from . import transforms


class TaskPipeline(BasePipeline):
    """
    VideoThinkBench pipeline — downloads from HuggingFace and converts
    to the standardized format.

    REPLACE THIS with your own dataset pipeline when forking.
    """

    def __init__(self, config: TaskConfig):
        super().__init__(config)
        self.task_config = config
        self.downloader = HuggingFaceDownloader(
            repo_id=config.hf_repo,
            split=config.split,
        )

    # ── 1) Download ───────────────────────────────────────────────────────

    def download(self) -> Iterator[dict]:
        """Download VideoThinkBench from HuggingFace."""
        yield from self.downloader.download(limit=self.config.num_samples)

    # ── 2) Process ────────────────────────────────────────────────────────

    def process_sample(self, raw_sample: dict, idx: int) -> Optional[TaskSample]:
        """Transform one VideoThinkBench sample to standardized format."""
        domain = transforms.extract_domain(raw_sample, default=self.task_config.domain)
        task_id = f"vtb_{self.task_config.split}_{idx:05d}"

        videos = transforms.extract_videos(raw_sample)

        return SampleProcessor.build_sample(
            task_id=task_id,
            domain=domain,
            first_image=transforms.extract_first_image(raw_sample),
            prompt=transforms.extract_prompt(raw_sample),
            final_image=transforms.extract_final_image(raw_sample),
            first_video=videos.get("first_video"),
            last_video=videos.get("last_video"),
            ground_truth_video=videos.get("ground_truth_video"),
        )
