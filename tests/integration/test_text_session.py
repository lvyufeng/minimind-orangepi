from __future__ import annotations

from pathlib import Path

import pytest

from minimind_orangepi.session import (
    GenerationResult,
    TextSession,
    chat_prompt,
    chat_prompt_from_history,
    parse_stream_token_line,
    parse_token_line,
    validate_runtime_model_dir,
)


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


def test_chat_prompt_from_history_matches_single_turn_prompt() -> None:
    assert chat_prompt_from_history([("你好", "")]) == chat_prompt("你好")


def test_chat_prompt_from_history_includes_previous_turns() -> None:
    prompt = chat_prompt_from_history([("你好", "您好"), ("你是谁", "")])
    assert prompt == (
        "<|im_start|>user\n你好<|im_end|>\n"
        "<|im_start|>assistant\n您好<|im_end|>\n"
        "<|im_start|>user\n你是谁<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def test_chat_prompt_from_history_applies_thinking_to_final_turn_only() -> None:
    prompt = chat_prompt_from_history([("你好", "您好"), ("继续", "")], open_thinking=True)
    assert prompt.count("<think>") == 1
    assert prompt.endswith("<|im_start|>assistant\n<think>\n")


def test_chat_prompt_from_history_rejects_empty_history() -> None:
    with pytest.raises(ValueError, match="history cannot be empty"):
        chat_prompt_from_history([])


def test_validate_runtime_model_dir_allows_toy_fallback() -> None:
    validate_runtime_model_dir(None)


def test_validate_runtime_model_dir_rejects_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        validate_runtime_model_dir(tmp_path / "missing")


def test_validate_runtime_model_dir_rejects_file(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.write_text("not a directory")
    with pytest.raises(NotADirectoryError):
        validate_runtime_model_dir(model)


def test_validate_runtime_model_dir_rejects_missing_runtime_files(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    with pytest.raises(ValueError, match="minimind_runtime_config.txt"):
        validate_runtime_model_dir(model)

    (model / "minimind_runtime_config.txt").write_text("hidden_size=4\n")
    with pytest.raises(ValueError, match="weights.bin"):
        validate_runtime_model_dir(model)


def test_validate_runtime_model_dir_accepts_runtime_dir(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "minimind_runtime_config.txt").write_text("hidden_size=4\n")
    (model / "weights.bin").write_bytes(b"MMRTW001")
    validate_runtime_model_dir(model)


def test_parse_stream_token_line() -> None:
    assert parse_stream_token_line("token: 42") == 42
    with pytest.raises(ValueError):
        parse_stream_token_line("generated_tokens: 42")
    with pytest.raises(ValueError):
        parse_stream_token_line("token: 1 2")


def test_text_session_stream_matches_blocking_toy_generation() -> None:
    root = Path(__file__).resolve().parents[2]
    session = TextSession(root / "build" / "minimind_generate", None)
    streamed = list(session.stream_generate_result("hi", 3))
    blocking = session.generate_result("hi", 3)
    assert streamed
    assert streamed[-1].generated_tokens == blocking.generated_tokens
    assert streamed[-1].prompt_tokens == blocking.prompt_tokens


def test_text_session_streams_chat_history_with_toy_generation() -> None:
    root = Path(__file__).resolve().parents[2]
    session = TextSession(root / "build" / "minimind_generate", None)
    history = [("hi", "hello"), ("who are you", "")]
    streamed = list(session.stream_generate_chat_result(history, 3))
    assert streamed
    assert streamed[-1].generated_tokens
    assert streamed[-1].prompt_tokens
