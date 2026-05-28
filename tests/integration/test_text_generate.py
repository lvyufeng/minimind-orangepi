from __future__ import annotations

import subprocess
from pathlib import Path


def test_run_text_cli() -> None:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            "python3",
            str(root / "src" / "python" / "tools" / "run_text.py"),
            "--prompt",
            "hi",
            "--max-new-tokens",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "generated_tokens:" in result.stdout


def test_minimind_generate_accepts_explicit_tokens() -> None:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            str(root / "build" / "minimind_generate"),
            "--tokens",
            "1,2",
            "--max-new-tokens",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "prompt_tokens: 1 2" in result.stdout
    assert "generated_tokens:" in result.stdout
