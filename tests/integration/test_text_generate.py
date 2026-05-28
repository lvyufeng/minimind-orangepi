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
