"""
Your dataset transforms.

CUSTOMIZE THIS FILE to map source dataset fields to the standardized format.

Each transform function takes a raw sample dict from HuggingFace and returns
the value for one standardized field. Modify these to match your source
dataset's schema.
"""


def extract_first_image(raw_sample: dict):
    """Extract the first frame / initial state image.

    Args:
        raw_sample: Raw sample dict from the source dataset.

    Returns:
        PIL Image, numpy array, file path, or ``None``.
    """
    return raw_sample.get("image")


def extract_final_image(raw_sample: dict):
    """Extract the final frame / goal state image.

    Returns:
        PIL Image, numpy array, file path, or ``None``.
    """
    return raw_sample.get("target_image")


def extract_prompt(raw_sample: dict) -> str:
    """Extract the natural-language prompt / question.

    Returns:
        Prompt string.
    """
    return (
        raw_sample.get("question")
        or raw_sample.get("prompt")
        or "Solve this visual reasoning task."
    )


def extract_domain(raw_sample: dict, default: str = "videothinkbench") -> str:
    """Extract the domain / task-type label.

    Returns:
        Domain string used in the output path ``{domain}_task/``.
    """
    return raw_sample.get("task_type") or raw_sample.get("category") or default


def extract_videos(raw_sample: dict) -> dict:
    """Extract optional video paths / bytes.

    Returns:
        Dict with keys ``first_video``, ``last_video``, ``ground_truth_video``
        (values may be ``None``).
    """
    return {
        "first_video": raw_sample.get("first_video"),
        "last_video": raw_sample.get("last_video"),
        "ground_truth_video": raw_sample.get("ground_truth_video"),
    }
