from __future__ import annotations

import subprocess
from pathlib import Path


def test_compare_text_logits_runtime_smoke() -> None:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            "python3",
            str(root / "src" / "python" / "tools" / "compare_text_logits.py"),
            "--prompt",
            "hello",
            "--max-new-tokens",
            "2",
            "--expect-generated",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "prompt_tokens:" in result.stdout
    assert "generated_tokens:" in result.stdout
