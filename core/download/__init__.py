"""Download module — fetch raw data from external sources."""

from .downloader import HuggingFaceDownloader
from .s3 import upload_directory_to_s3, download_from_s3

__all__ = [
    "HuggingFaceDownloader",
    "upload_directory_to_s3",
    "download_from_s3",
]
