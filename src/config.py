"""
Your pipeline configuration.

CUSTOMIZE THIS FILE to define your dataset-specific settings.
Inherits common settings from core.PipelineConfig.
"""

from pydantic import Field
from core import PipelineConfig


class TaskConfig(PipelineConfig):
    """
    Your dataset-specific configuration.

    CUSTOMIZE THIS CLASS for the dataset you are converting.

    Inherited from PipelineConfig:
        - num_samples: Optional[int]  # Max samples (None = all)
        - domain: str                 # Task domain name
        - output_dir: Path            # Where to save outputs
        - split: str                  # Dataset split (default: "test")
    """

    # ======================================================================
    #  OVERRIDE DEFAULTS
    # ======================================================================

    domain: str = Field(default="videothinkbench")

    # ======================================================================
    #  DATASET-SPECIFIC SETTINGS
    # ======================================================================

    hf_repo: str = Field(
        default="video-think-bench/VideoThinkBench",
        description="HuggingFace dataset repository ID",
    )
