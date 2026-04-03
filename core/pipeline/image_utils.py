"""Image utilities for handling various image formats."""

from pathlib import Path
from typing import Union, Optional
import numpy as np
from PIL import Image


def convert_to_pil_image(
    image_input: Union[Image.Image, np.ndarray, str, Path, None],
    mode: str = "RGB",
) -> Optional[Image.Image]:
    """Convert various image formats to PIL Image.

    Args:
        image_input: PIL Image, numpy array, or file path.
        mode: Target colour mode (default: RGB).

    Returns:
        PIL Image in *mode*, or ``None`` if conversion fails.
    """
    if image_input is None:
        return None

    if isinstance(image_input, Image.Image):
        return image_input.convert(mode) if image_input.mode != mode else image_input

    if isinstance(image_input, np.ndarray):
        return numpy_to_pil(image_input, mode)

    if isinstance(image_input, (str, Path)):
        return load_from_path(Path(image_input), mode)

    return None


def numpy_to_pil(arr: np.ndarray, mode: str = "RGB") -> Optional[Image.Image]:
    """Convert numpy array to PIL Image."""
    if arr.dtype in [np.float32, np.float64]:
        if arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    if arr.ndim == 2:
        img = Image.fromarray(arr, mode="L")
    elif arr.ndim == 3:
        if arr.shape[2] == 3:
            img = Image.fromarray(arr, mode="RGB")
        elif arr.shape[2] == 4:
            img = Image.fromarray(arr, mode="RGBA")
        else:
            return None
    else:
        return None

    return img.convert(mode) if img.mode != mode else img


def load_from_path(path: Path, mode: str = "RGB") -> Optional[Image.Image]:
    """Load image from file path."""
    if not path.exists():
        return None
    img = Image.open(path)
    return img.convert(mode) if img.mode != mode else img
