#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "python"))

from minimind_orangepi.image_processing import (
    VLMPreprocessConfig,
    expand_image_placeholders,
    image_tensor_shape,
    preprocess_image,
    validate_image_placeholders,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MiniMind-V image inputs for the local runtime")
    parser.add_argument("--model", default="models/minimind-v", help="Reserved for the real VLM checkpoint path")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", default="<image>\n请描述这张图中的主要物体和场景。")
    args = parser.parse_args()

    config = VLMPreprocessConfig()
    processed_prompt = expand_image_placeholders(args.prompt, config)
    validate_image_placeholders(processed_prompt, 1, config)
    pixels = preprocess_image(args.image, config)

    print(f"prompt_image_tokens={config.image_token_len}")
    print(f"image_shape={image_tensor_shape(config)}")
    print(f"pixel_values={len(pixels)}")
    print(processed_prompt[: min(len(processed_prompt), 120)])


if __name__ == "__main__":
    main()
