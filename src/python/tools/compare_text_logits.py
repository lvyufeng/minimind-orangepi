#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def parse_tokens(output: str, label: str) -> list[int]:
    for line in output.splitlines():
        if line.startswith(label + ":"):
            return [int(part) for part in line.split(":", 1)[1].split()]
    raise ValueError(f"missing {label} in runtime output")


def run_runtime(executable: Path, model: Path | None, prompt: str, max_new_tokens: int) -> tuple[list[int], list[int], str]:
    cmd = [str(executable), "--prompt", prompt, "--max-new-tokens", str(max_new_tokens)]
    if model is not None:
        cmd.extend(["--model", str(model)])
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return (
        parse_tokens(result.stdout, "prompt_tokens"),
        parse_tokens(result.stdout, "generated_tokens"),
        result.stdout,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-check MiniMind runtime generation and prepare logits comparison inputs")
    parser.add_argument("--runtime-model", type=Path, default=None)
    parser.add_argument("--reference-model", type=Path, default=None, help="Reserved for PyTorch/HF logits comparison")
    parser.add_argument("--executable", type=Path, default=ROOT / "build" / "minimind_generate")
    parser.add_argument("--prompt", default="hello")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--expect-generated", type=int, default=None)
    args = parser.parse_args()

    prompt_tokens, generated_tokens, raw = run_runtime(
        args.executable,
        args.runtime_model,
        args.prompt,
        args.max_new_tokens,
    )
    if not prompt_tokens:
        print("runtime produced no prompt tokens", file=sys.stderr)
        return 1
    if args.max_new_tokens > 0 and not generated_tokens:
        print("runtime produced no generated tokens", file=sys.stderr)
        return 1
    if args.expect_generated is not None and len(generated_tokens) != args.expect_generated:
        print(
            f"expected {args.expect_generated} generated tokens, got {len(generated_tokens)}",
            file=sys.stderr,
        )
        return 1

    print(raw, end="")
    if args.reference_model is not None:
        print("reference_model comparison is not implemented yet; runtime smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
