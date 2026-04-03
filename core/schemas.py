"""Pydantic schemas for pipeline data."""

from typing import Optional, Any, Dict
from pathlib import Path
from pydantic import BaseModel, Field


class PipelineConfig(BaseModel):
    """Pipeline configuration — base class for task-specific configs."""

    num_samples: Optional[int] = None
    domain: str = "dataset"
    output_dir: Path = Path("data/questions")
    split: str = "test"


class TaskSample(BaseModel):
    """A single task sample in the standardized format.

    Mirrors the TaskPair schema from template-data-generator so both
    generators and pipelines produce the same on-disk layout.
    """

    task_id: str
    domain: str
    prompt: str
    first_image: Any  # PIL Image
    final_image: Optional[Any] = None  # PIL Image
    first_video: Optional[str] = None  # Path to first segment video
    last_video: Optional[str] = None  # Path to last segment video
    ground_truth_video: Optional[str] = None  # Path to full video
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        arbitrary_types_allowed = True
