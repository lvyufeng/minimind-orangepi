from __future__ import annotations

import subprocess
from pathlib import Path


class TextSession:
    def __init__(self, executable: Path | None = None, model: Path | None = None) -> None:
        root = Path(__file__).resolve().parents[3]
        self.executable = executable or root / "build" / "minimind_generate"
        self.model = model

    def _encode_with_tokenizer_json(self, prompt: str) -> list[int] | None:
        if self.model is None:
            return None
        tokenizer_path = self.model / "tokenizer.json"
        if not tokenizer_path.exists():
            return None
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RuntimeError("tokenizer.json requires the tokenizers package") from exc
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        return tokenizer.encode(prompt).ids

    def generate(self, prompt: str, max_new_tokens: int = 8) -> str:
        cmd = [str(self.executable), "--max-new-tokens", str(max_new_tokens)]
        if self.model is not None:
            cmd.extend(["--model", str(self.model)])
        token_ids = self._encode_with_tokenizer_json(prompt)
        if token_ids is None:
            cmd.extend(["--prompt", prompt])
        else:
            cmd.extend(["--tokens", ",".join(str(token) for token in token_ids)])
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout
