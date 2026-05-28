from __future__ import annotations

import subprocess
from pathlib import Path


class TextSession:
    def __init__(self, executable: Path | None = None) -> None:
        root = Path(__file__).resolve().parents[3]
        self.executable = executable or root / "build" / "minimind_generate"

    def generate(self, prompt: str, max_new_tokens: int = 8) -> str:
        result = subprocess.run(
            [
                str(self.executable),
                "--prompt",
                prompt,
                "--max-new-tokens",
                str(max_new_tokens),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout
