"""Abstract base pipeline — download then process."""

from abc import ABC, abstractmethod
from typing import Iterator, List, Optional

from ..schemas import PipelineConfig, TaskSample
from .output_writer import OutputWriter


class BasePipeline(ABC):
    """Base class for dataset pipelines.

    Subclass this and implement :meth:`download` and :meth:`process_sample`.
    The :meth:`run` method orchestrates the full workflow:
    download → process → write.

    Example::

        class MyPipeline(BasePipeline):
            def download(self):
                for item in load_dataset(...):
                    yield item

            def process_sample(self, raw, idx):
                return SampleProcessor.build_sample(...)

        pipeline = MyPipeline(config)
        samples = pipeline.run()   # downloads, processes, writes to disk
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    # ── 1) Download ───────────────────────────────────────────────────────

    @abstractmethod
    def download(self) -> Iterator[dict]:
        """Download / stream raw samples from the source.

        Yields:
            Raw sample dicts (structure depends on the source dataset).
        """

    # ── 2) Process ────────────────────────────────────────────────────────

    @abstractmethod
    def process_sample(self, raw_sample: dict, idx: int) -> Optional[TaskSample]:
        """Transform one raw sample into a standardized :class:`TaskSample`.

        Return ``None`` to skip invalid or unusable samples.

        Args:
            raw_sample: A single dict yielded by :meth:`download`.
            idx: Zero-based index of the sample.

        Returns:
            A :class:`TaskSample`, or ``None`` to skip.
        """

    # ── Orchestration (don't override) ────────────────────────────────────

    def run(self) -> List[TaskSample]:
        """Execute the full pipeline: download → process → write.

        Returns:
            List of successfully processed :class:`TaskSample` objects.
        """
        writer = OutputWriter(self.config.output_dir)
        samples: List[TaskSample] = []
        processed = 0

        for idx, raw in enumerate(self.download()):
            sample = self.process_sample(raw, idx)
            if sample is None:
                print(f"  Skipped sample {idx}")
                continue

            writer.write_sample(sample)
            samples.append(sample)
            processed += 1

            if processed % 10 == 0:
                print(f"  Processed {processed} samples...")

        print(
            f"Done! Processed {processed} samples "
            f"-> {self.config.output_dir}/{self.config.domain}_task/"
        )
        return samples
