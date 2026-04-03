"""Core pipeline utilities — KEEP as-is when forking.

Two sub-modules:
    core.download  — fetch raw data (HuggingFace, S3, etc.)
    core.pipeline  — process raw data into standardized format
"""

from .schemas import TaskSample, PipelineConfig

# Download module
from .download import HuggingFaceDownloader, upload_directory_to_s3, download_from_s3

# Pipeline module
from .pipeline import (
    BasePipeline,
    SampleProcessor,
    OutputWriter,
    convert_to_pil_image,
    numpy_to_pil,
    load_from_path,
    validate_task_data,
    validate_task_directory,
)

__all__ = [
    # Schemas
    "TaskSample",
    "PipelineConfig",
    # Download
    "HuggingFaceDownloader",
    "upload_directory_to_s3",
    "download_from_s3",
    # Pipeline
    "BasePipeline",
    "SampleProcessor",
    "OutputWriter",
    "convert_to_pil_image",
    "numpy_to_pil",
    "load_from_path",
    "validate_task_data",
    "validate_task_directory",
]
