#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "python"))

from minimind_orangepi.session import TextSession, validate_runtime_model_dir


def model_mode(model: Path | None) -> str:
    if model is None:
        return "model: toy fallback (no --model supplied)"
    return f"model: {model}"


def format_details(result, mode: str) -> str:
    prompt_tokens = " ".join(str(token) for token in result.prompt_tokens)
    generated_tokens = " ".join(str(token) for token in result.generated_tokens)
    return f"{mode}\nprompt_tokens: {prompt_tokens}\ngenerated_tokens: {generated_tokens}"


def output_text(result) -> str:
    if result.generated_text is not None:
        return result.generated_text
    if result.generated_tokens:
        return "generated_tokens: " + " ".join(str(token) for token in result.generated_tokens)
    return result.raw_output


def generate_text(
    session: TextSession,
    mode: str,
    prompt: str,
    max_new_tokens: int,
    raw_prompt: bool,
    open_thinking: bool,
):
    if not prompt.strip():
        yield "Please enter a prompt.", mode
        return
    try:
        yielded = False
        for result in session.stream_generate_result(prompt, int(max_new_tokens), raw_prompt, open_thinking):
            yielded = True
            yield output_text(result), format_details(result, mode)
        if not yielded:
            yield "", mode
    except subprocess.CalledProcessError as exc:
        error = exc.stderr.strip() or exc.output.strip() or str(exc)
        yield f"Runtime error:\n{error}", mode
    except Exception as exc:
        yield f"Error: {exc}", mode


def build_demo(
    session: TextSession,
    mode: str,
    default_max_new_tokens: int,
    default_raw_prompt: bool,
    default_open_thinking: bool,
):
    import gradio as gr

    with gr.Blocks(title="MiniMind Text LLM Demo") as demo:
        gr.Markdown(
            "# MiniMind Text LLM Demo\n"
            "Text-only MiniMind runtime demo. MiniMind-V and MiniMind-O inference are not supported here."
        )
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", value="你好", lines=5)
                max_new_tokens = gr.Slider(
                    label="Max new tokens",
                    minimum=1,
                    maximum=256,
                    step=1,
                    value=default_max_new_tokens,
                )
                raw_prompt = gr.Checkbox(label="Raw prompt", value=default_raw_prompt)
                open_thinking = gr.Checkbox(label="Open thinking marker", value=default_open_thinking)
                submit = gr.Button("Generate", variant="primary")
            with gr.Column():
                output = gr.Textbox(label="Generated text / runtime output", lines=12)
                details = gr.Textbox(label="Runtime details", value=mode, lines=6)

        submit.click(
            fn=lambda prompt_value, token_value, raw_value, thinking_value: generate_text(
                session,
                mode,
                prompt_value,
                token_value,
                raw_value,
                thinking_value,
            ),
            inputs=[prompt, max_new_tokens, raw_prompt, open_thinking],
            outputs=[output, details],
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a text-only MiniMind Gradio demo")
    parser.add_argument("--model", type=Path, default=None, help="Exported MiniMind text runtime model directory")
    parser.add_argument("--executable", type=Path, default=None, help="Path to build/minimind_generate")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--raw-prompt", action="store_true")
    parser.add_argument("--open-thinking", action="store_true")
    args = parser.parse_args()

    validate_runtime_model_dir(args.model)
    session = TextSession(args.executable, args.model)
    demo = build_demo(
        session,
        model_mode(args.model),
        args.max_new_tokens,
        args.raw_prompt,
        args.open_thinking,
    )
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
