from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GenerationResult:
    raw_output: str
    prompt_tokens: list[int]
    generated_tokens: list[int]
    generated_text: str | None = None

    def format(self) -> str:
        if self.generated_text is None:
            return self.raw_output
        return f"{self.raw_output}generated_text: {self.generated_text}\n"


def chat_prompt(prompt: str, open_thinking: bool = False) -> str:
    if open_thinking:
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def parse_token_line(output: str, label: str) -> list[int]:
    for line in output.splitlines():
        if line.startswith(label + ":"):
            return [int(part) for part in line.split(":", 1)[1].split()]
    raise ValueError(f"missing {label} in runtime output")


class TextSession:
    def __init__(self, executable: Path | None = None, model: Path | None = None) -> None:
        root = Path(__file__).resolve().parents[3]
        self.executable = executable or root / "build" / "minimind_generate"
        self.model = model

    def _load_tokenizer(self):
        if self.model is None:
            return None
        tokenizer_path = self.model / "tokenizer.json"
        if not tokenizer_path.exists():
            return None
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RuntimeError("tokenizer.json requires the tokenizers package") from exc
        return Tokenizer.from_file(str(tokenizer_path))

    def generate_result(
        self,
        prompt: str,
        max_new_tokens: int = 8,
        raw_prompt: bool = False,
        open_thinking: bool = False,
    ) -> GenerationResult:
        cmd = [str(self.executable), "--max-new-tokens", str(max_new_tokens)]
        if self.model is not None:
            cmd.extend(["--model", str(self.model)])
        tokenizer = self._load_tokenizer()
        if tokenizer is None:
            cmd.extend(["--prompt", prompt])
        else:
            formatted_prompt = prompt if raw_prompt else chat_prompt(prompt, open_thinking)
            token_ids = tokenizer.encode(formatted_prompt).ids
            cmd.extend(["--tokens", ",".join(str(token) for token in token_ids)])
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        prompt_tokens = parse_token_line(result.stdout, "prompt_tokens")
        generated_tokens = parse_token_line(result.stdout, "generated_tokens")
        generated_text = tokenizer.decode(generated_tokens) if tokenizer is not None else None
        return GenerationResult(
            raw_output=result.stdout,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            generated_text=generated_text,
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 8,
        raw_prompt: bool = False,
        open_thinking: bool = False,
    ) -> str:
        return self.generate_result(prompt, max_new_tokens, raw_prompt, open_thinking).format()
