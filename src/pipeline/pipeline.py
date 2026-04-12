"""Video-MCP style pipeline implementation on top of core template orchestration."""

from typing import Iterator, Optional

from core.pipeline import BasePipeline, SampleProcessor, TaskSample
from core.download import run_download
from .config import TaskConfig
from . import transforms


class TaskPipeline(BasePipeline):
    """
    CoreCognition MCQA → Video-MCP style task conversion.
    """

    def __init__(self, config: TaskConfig):
        super().__init__(config)
        self.task_config = config

    # ── 1) Download ───────────────────────────────────────────────────────

    def download(self) -> Iterator[dict]:
        """Download dataset via core.download → src.download."""
        yield from run_download(self.task_config)

    # ── 2) Process ────────────────────────────────────────────────────────

    def process_sample(self, raw_sample: dict, idx: int) -> Optional[TaskSample]:
        """Render first/final frames and optional ground-truth clip."""
        domain = transforms.extract_domain(raw_sample, default=self.task_config.domain)
        global_idx = int(getattr(self.task_config, "start_index", 0)) + idx
        task_id = f"{domain}_{global_idx:08d}"
        source_image = transforms.extract_source_image(raw_sample)
        if source_image is None:
            return None

        question = str(raw_sample.get("question", "")).strip()
        choices = raw_sample.get("choices", {})
        answer = str(raw_sample.get("answer", "")).strip().upper()
        if not question or not choices or answer not in {"A", "B", "C", "D"}:
            return None

        first_frame = transforms.render_video_mcp_frame(
            source_image=source_image,
            question=question,
            choices=choices,
            lit_choice=None,
            lit_progress=0.0,
            lit_style=self.task_config.lit_style,
            width=self.task_config.width,
            height=self.task_config.height,
        )
        final_frame = transforms.render_video_mcp_frame(
            source_image=source_image,
            question=question,
            choices=choices,
            lit_choice=answer,
            lit_progress=1.0,
            lit_style=self.task_config.lit_style,
            width=self.task_config.width,
            height=self.task_config.height,
        )
        ground_truth_video = transforms.render_ground_truth_video(
            source_image=source_image,
            question=question,
            choices=choices,
            answer=answer,
            task_id=task_id,
            output_dir=self.task_config.output_dir.parent / "generated_videos",
            fps=self.task_config.fps,
            num_frames=self.task_config.num_frames,
            width=self.task_config.width,
            height=self.task_config.height,
            lit_style=self.task_config.lit_style,
        )

        return SampleProcessor.build_sample(
            task_id=task_id,
            domain=domain,
            first_image=first_frame,
            prompt=transforms.format_prompt(raw_sample),
            final_image=final_frame,
            ground_truth_video=ground_truth_video,
            metadata={
                "dataset": raw_sample.get("dataset", "CoreCognition"),
                "source_id": raw_sample.get("source_id"),
                "choices": choices,
                "answer": answer,
                "image_filename": raw_sample.get("image_filename"),
                "video_spec": {
                    "fps": self.task_config.fps,
                    "num_frames": self.task_config.num_frames,
                    "width": self.task_config.width,
                    "height": self.task_config.height,
                    "lit_style": self.task_config.lit_style,
                },
            },
        )
