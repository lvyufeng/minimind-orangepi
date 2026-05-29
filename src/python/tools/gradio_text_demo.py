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


def chat_content_to_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        for key in ("text", "content", "value"):
            if key in content:
                return chat_content_to_text(content[key])
        return ""
    if isinstance(content, (list, tuple)):
        return "".join(chat_content_to_text(item) for item in content)
    return str(content)


def message_field(message, name: str, default=""):
    if isinstance(message, dict):
        return message.get(name, default)
    return getattr(message, name, default)


def history_to_messages(history: list[tuple[str, str | None]]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for user_text, assistant_text in history:
        messages.append({"role": "user", "content": user_text})
        if assistant_text is not None:
            messages.append({"role": "assistant", "content": assistant_text})
    return messages


def messages_to_history(messages: list[dict[str, str]] | None) -> list[tuple[str, str | None]]:
    history: list[tuple[str, str | None]] = []
    if not messages:
        return history
    pending_user: str | None = None
    for message in messages:
        role = message_field(message, "role")
        content = chat_content_to_text(message_field(message, "content", ""))
        if role == "user":
            if pending_user is not None:
                history.append((pending_user, ""))
            pending_user = content
        elif role == "assistant" and pending_user is not None:
            history.append((pending_user, content))
            pending_user = None
    if pending_user is not None:
        history.append((pending_user, ""))
    return history


def generate_chat(
    session: TextSession,
    mode: str,
    message: str,
    messages: list[dict[str, str]] | None,
    max_new_tokens: int,
    raw_prompt: bool,
    open_thinking: bool,
):
    history = messages_to_history(messages)
    message_text = chat_content_to_text(message)
    if not message_text.strip():
        yield history_to_messages(history), "", "Please enter a prompt."
        return
    active_history = history + [(message_text, "")]
    if raw_prompt:
        stream = session.stream_generate_result(message_text, int(max_new_tokens), True, open_thinking)
    else:
        stream = session.stream_generate_chat_result(active_history, int(max_new_tokens), open_thinking)
    yield history_to_messages(active_history), "", mode
    try:
        yielded = False
        for result in stream:
            yielded = True
            active_history[-1] = (message_text, output_text(result))
            yield history_to_messages(active_history), "", format_details(result, mode)
        if not yielded:
            yield history_to_messages(active_history), "", mode
    except subprocess.CalledProcessError as exc:
        error = exc.stderr.strip() or exc.output.strip() or str(exc)
        active_history[-1] = (message_text, f"Runtime error:\n{error}")
        yield history_to_messages(active_history), "", mode
    except Exception as exc:
        active_history[-1] = (message_text, f"Error: {exc}")
        yield history_to_messages(active_history), "", mode


def clear_chat(mode: str):
    return [], "", mode


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
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Chat", height=480)
                prompt = gr.Textbox(label="Message", value="你好", lines=3)
                with gr.Row():
                    submit = gr.Button("Send", variant="primary")
                    clear = gr.Button("Clear")
            with gr.Column(scale=1):
                max_new_tokens = gr.Slider(
                    label="Max new tokens",
                    minimum=1,
                    maximum=256,
                    step=1,
                    value=default_max_new_tokens,
                )
                raw_prompt = gr.Checkbox(
                    label="Raw prompt (single-turn debug)",
                    value=default_raw_prompt,
                )
                open_thinking = gr.Checkbox(label="Open thinking marker", value=default_open_thinking)
                details = gr.Textbox(label="Runtime details", value=mode, lines=10)

        def generate_chat_for_ui(message_value, history_value, token_value, raw_value, thinking_value):
            yield from generate_chat(
                session,
                mode,
                message_value,
                history_value or [],
                token_value,
                raw_value,
                thinking_value,
            )

        submit.click(
            fn=generate_chat_for_ui,
            inputs=[prompt, chatbot, max_new_tokens, raw_prompt, open_thinking],
            outputs=[chatbot, prompt, details],
        )
        prompt.submit(
            fn=generate_chat_for_ui,
            inputs=[prompt, chatbot, max_new_tokens, raw_prompt, open_thinking],
            outputs=[chatbot, prompt, details],
        )
        clear.click(
            fn=lambda: clear_chat(mode),
            inputs=None,
            outputs=[chatbot, prompt, details],
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
