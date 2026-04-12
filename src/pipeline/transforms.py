"""Video-MCP style transforms and rendering helpers."""

import subprocess
import tempfile
import textwrap
from io import BytesIO
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

CHOICE_ORDER = ("A", "B", "C", "D")


def extract_source_image(raw_sample: dict) -> Optional[Image.Image]:
    """Decode source image bytes from the downloader."""
    image_bytes = raw_sample.get("image_bytes")
    if not image_bytes:
        return None
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def extract_domain(raw_sample: dict, default: str = "corecognition") -> str:
    return default


def format_prompt(raw_sample: dict) -> str:
    """Match video-mcp prompt style: question + options + answer."""
    question = str(raw_sample.get("question", "")).strip()
    choices = raw_sample.get("choices", {})
    answer = str(raw_sample.get("answer", "")).strip().upper()

    lines = [question, ""]
    for key in CHOICE_ORDER:
        if key in choices:
            lines.append(f"{key}: {choices[key]}")
    lines.append("")
    lines.append(f"Answer: {answer}")
    return "\n".join(lines) + "\n"


def _fit_into_box(image: Image.Image, box_w: int, box_h: int) -> Image.Image:
    scale = min(box_w / image.width, box_h / image.height)
    new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def render_video_mcp_frame(
    *,
    source_image: Image.Image,
    question: str,
    choices: dict[str, str],
    lit_choice: Optional[str],
    lit_progress: float,
    lit_style: str,
    width: int,
    height: int,
) -> Image.Image:
    """Render one frame with two-column prompt panel and choice highlights."""
    frame = Image.new("RGBA", (width, height), (245, 245, 245, 255))
    draw = ImageDraw.Draw(frame)
    font = ImageFont.load_default()

    pad = 20
    left_w = int(width * 0.55)
    right_x = left_w + pad

    # Left image panel
    draw.rectangle((pad, pad, left_w - pad, height - pad), outline=(170, 170, 170), width=2)
    fitted = _fit_into_box(source_image, left_w - (pad * 3), height - (pad * 3))
    image_x = pad + ((left_w - (pad * 2) - fitted.width) // 2)
    image_y = pad + ((height - (pad * 2) - fitted.height) // 2)
    frame.paste(fitted, (image_x, image_y))

    # Right prompt panel
    draw.rounded_rectangle(
        (right_x, pad, width - pad, height - pad),
        radius=12,
        fill=(255, 255, 255, 255),
        outline=(180, 180, 180, 255),
        width=2,
    )
    text_x = right_x + 14
    text_y = pad + 14
    wrapped = textwrap.fill(question, width=36)
    draw.multiline_text((text_x, text_y), wrapped, fill=(20, 20, 20), font=font, spacing=4)
    text_y += 84
    for key in CHOICE_ORDER:
        if key in choices:
            draw.text(
                (text_x, text_y),
                f"{key}: {choices[key]}",
                fill=(20, 20, 20),
                font=font,
            )
            text_y += 22

    # Corner answer boxes
    corner_boxes = {
        "A": (10, 10, 90, 70),
        "B": (width - 90, 10, width - 10, 70),
        "C": (10, height - 70, 90, height - 10),
        "D": (width - 90, height - 70, width - 10, height - 10),
    }
    for key, box in corner_boxes.items():
        draw.rounded_rectangle(box, radius=8, fill=(255, 255, 255, 220), outline=(60, 60, 60), width=2)
        draw.text((box[0] + 34, box[1] + 24), key, fill=(0, 0, 0), font=font)

    if lit_choice in corner_boxes and lit_progress > 0:
        box = corner_boxes[lit_choice]
        progress = max(0.0, min(1.0, lit_progress))
        overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
        o = ImageDraw.Draw(overlay)
        if lit_style == "red_border":
            border = max(2, int(12 * progress))
            o.rounded_rectangle(box, radius=8, outline=(220, 20, 20, 255), width=border)
        else:
            alpha = int(160 * progress)
            o.rounded_rectangle(box, radius=8, fill=(0, 0, 0, alpha))
        frame = Image.alpha_composite(frame, overlay)

    return frame.convert("RGB")


def render_ground_truth_video(
    *,
    source_image: Image.Image,
    question: str,
    choices: dict[str, str],
    answer: str,
    task_id: str,
    output_dir: Path,
    fps: int,
    num_frames: int,
    width: int,
    height: int,
    lit_style: str,
) -> Optional[str]:
    """Render progressive highlighted frames and compile into mp4 using ffmpeg."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{task_id}.mp4"

    try:
        with tempfile.TemporaryDirectory() as tmp:
            frames_dir = Path(tmp)
            for idx in range(num_frames):
                if idx == 0:
                    lit_choice = None
                    progress = 0.0
                else:
                    lit_choice = answer
                    progress = idx / max(1, num_frames - 1)
                frame = render_video_mcp_frame(
                    source_image=source_image,
                    question=question,
                    choices=choices,
                    lit_choice=lit_choice,
                    lit_progress=progress,
                    lit_style=lit_style,
                    width=width,
                    height=height,
                )
                frame.save(frames_dir / f"frame_{idx:04d}.png", format="PNG")

            cmd = [
                "ffmpeg",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(frames_dir / "frame_%04d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-s",
                f"{width}x{height}",
                "-loglevel",
                "error",
                str(output_path),
            ]
            subprocess.run(cmd, check=True)
        return str(output_path)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
