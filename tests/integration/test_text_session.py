from __future__ import annotations

from minimind_orangepi.session import GenerationResult, chat_prompt, parse_token_line


def test_parse_token_line() -> None:
    output = "prompt_tokens: 1 2\ngenerated_tokens: 3 4\n"
    assert parse_token_line(output, "prompt_tokens") == [1, 2]
    assert parse_token_line(output, "generated_tokens") == [3, 4]


def test_generation_result_formats_decoded_text() -> None:
    result = GenerationResult(
        raw_output="prompt_tokens: 1\ngenerated_tokens: 2\n",
        prompt_tokens=[1],
        generated_tokens=[2],
        generated_text="hello",
    )
    assert result.format().endswith("generated_text: hello\n")


def test_chat_prompt_matches_minimind_markers() -> None:
    assert chat_prompt("你好") == "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n"
    assert chat_prompt("你好", open_thinking=True).endswith("assistant\n<think>\n")
