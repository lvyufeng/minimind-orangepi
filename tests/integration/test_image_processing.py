from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "python"))

from minimind_orangepi.image_processing import (
    VLMPreprocessConfig,
    count_image_special_tokens,
    expand_image_placeholders,
    image_tensor_shape,
    validate_image_placeholders,
)


def test_expand_and_validate_placeholders() -> None:
    config = VLMPreprocessConfig()
    prompt = expand_image_placeholders("<image>\ndescribe it", config)
    assert count_image_special_tokens(prompt, config) == 64
    validate_image_placeholders(prompt, 1, config)
    with pytest.raises(ValueError):
        validate_image_placeholders(prompt, 2, config)
    assert image_tensor_shape(config) == (3, 256, 256)


def test_run_vlm_preprocess_cli(tmp_path: Path) -> None:
    pillow = pytest.importorskip("PIL.Image")
    root = ROOT
    image_path = tmp_path / "image.png"
    image = pillow.new("RGB", (8, 8), color=(255, 0, 0))
    image.save(image_path)

    result = subprocess.run(
        [
            "python3",
            str(root / "src" / "python" / "tools" / "run_vlm.py"),
            "--image",
            str(image_path),
            "--prompt",
            "<image>\nhello",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "prompt_image_tokens=64" in result.stdout
    assert "image_shape=(3, 256, 256)" in result.stdout
    assert "pixel_values=196608" in result.stdout
