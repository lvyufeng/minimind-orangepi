#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "python"))

from minimind_orangepi.session import TextSession


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MiniMind text generation on the local runtime")
    parser.add_argument("--model", type=Path, default=None, help="Runtime model directory exported by export_minimind_runtime.py")
    parser.add_argument("--prompt", default="MiniMind")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--executable", type=Path, default=None)
    args = parser.parse_args()

    session = TextSession(args.executable, args.model)
    print(session.generate(args.prompt, args.max_new_tokens), end="")


if __name__ == "__main__":
    main()
