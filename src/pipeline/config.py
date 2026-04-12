"""Pipeline configuration for the Video-MCP style CoreCognition task."""

from typing import Literal

from pydantic import Field

from core.pipeline import PipelineConfig


class TaskConfig(PipelineConfig):
    """Dataset + rendering settings for Video-MCP style generation."""

    domain: str = Field(default="corecognition")

    hf_repo: str = Field(
        default="williamium/CoreCognition",
        description="HuggingFace dataset repository ID",
    )
    hf_zip_filename: str = Field(
        default="CoreCognition_20250622.zip",
        description="Complete dataset ZIP filename hosted in the HF repo",
    )

    # Video-MCP defaults (Wan2.2-compatible profile)
    fps: int = Field(default=16, ge=1)
    num_frames: int = Field(default=81, ge=5)
    width: int = Field(default=832, ge=64)
    height: int = Field(default=480, ge=64)
    lit_style: Literal["darken", "red_border"] = Field(default="darken")
