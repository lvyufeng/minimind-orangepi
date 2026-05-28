from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class VLMPreprocessConfig:
    image_size: int = 256
    image_token_len: int = 64
    image_placeholder: str = "<image>"
    image_special_token: str = "<|image_pad|>"


def expand_image_placeholders(prompt: str, config: VLMPreprocessConfig = VLMPreprocessConfig()) -> str:
    count = prompt.count(config.image_placeholder)
    if count == 0:
        raise ValueError(f"prompt must contain {config.image_placeholder}")
    replacement = config.image_special_token * config.image_token_len
    return prompt.replace(config.image_placeholder, replacement)


def count_image_special_tokens(prompt: str, config: VLMPreprocessConfig = VLMPreprocessConfig()) -> int:
    return prompt.count(config.image_special_token)


def validate_image_placeholders(prompt: str, image_count: int = 1, config: VLMPreprocessConfig = VLMPreprocessConfig()) -> None:
    expected = image_count * config.image_token_len
    actual = count_image_special_tokens(prompt, config)
    if actual != expected:
        raise ValueError(f"expected {expected} {config.image_special_token} tokens, found {actual}")


def preprocess_image(path: Path, config: VLMPreprocessConfig = VLMPreprocessConfig()) -> list[float]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("MiniMind-V image preprocessing requires Pillow") from exc

    image = Image.open(path).convert("RGB")
    image = image.resize((config.image_size, config.image_size))
    pixels = list(image.getdata())
    output: list[float] = []
    output_extend = output.extend
    for r, g, b in pixels:
        output_extend(((r / 255.0 - 0.5) / 0.5, (g / 255.0 - 0.5) / 0.5, (b / 255.0 - 0.5) / 0.5))
    return output


def image_tensor_shape(config: VLMPreprocessConfig = VLMPreprocessConfig()) -> tuple[int, int, int]:
    return (3, config.image_size, config.image_size)


def list_images(image_dir: Path) -> Sequence[Path]:
    suffixes = {".png", ".jpg", ".jpeg", ".bmp"}
    return tuple(sorted(path for path in image_dir.iterdir() if path.suffix.lower() in suffixes))
