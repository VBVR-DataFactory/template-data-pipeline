"""Process module — transforms raw samples into the standardized format."""

from typing import Optional, Dict, Any

from .image_utils import convert_to_pil_image
from ..schemas import TaskSample


class SampleProcessor:
    """Utilities for building TaskSample objects from raw data.

    Subclass or use directly to map source dataset fields to the
    standardized output format.
    """

    @staticmethod
    def build_sample(
        task_id: str,
        domain: str,
        first_image: Any,
        prompt: str,
        final_image: Any = None,
        first_video: Optional[str] = None,
        last_video: Optional[str] = None,
        ground_truth_video: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[TaskSample]:
        """Build a validated TaskSample from raw inputs.

        Converts images via ``convert_to_pil_image`` and returns ``None``
        if required fields are missing or invalid.
        """
        first_pil = convert_to_pil_image(first_image)
        if first_pil is None:
            return None

        if not prompt or not prompt.strip():
            return None

        final_pil = convert_to_pil_image(final_image) if final_image is not None else None

        return TaskSample(
            task_id=task_id,
            domain=domain,
            prompt=prompt.strip(),
            first_image=first_pil,
            final_image=final_pil,
            first_video=first_video,
            last_video=last_video,
            ground_truth_video=ground_truth_video,
            metadata=metadata,
        )
