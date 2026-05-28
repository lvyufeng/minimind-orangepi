#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inspect_minimind_checkpoint import MiniMindConfig, expected_shapes, load_config, validate_config


TENSOR_ORDER_PREFIX = (
    "model.embed_tokens.weight",
    "model.norm.weight",
    "lm_head.weight",
)


@dataclass(frozen=True)
class RawTensor:
    shape: tuple[int, ...]
    values: list[float]


def fp16_to_float(value: int) -> float:
    sign = -1.0 if value & 0x8000 else 1.0
    exponent = (value >> 10) & 0x1F
    fraction = value & 0x3FF
    if exponent == 0:
        return sign * math.ldexp(fraction / 1024.0, -14)
    if exponent == 0x1F:
        return sign * (math.inf if fraction == 0 else math.nan)
    return sign * math.ldexp(1.0 + fraction / 1024.0, exponent - 15)


def bf16_to_float(value: int) -> float:
    return struct.unpack("<f", struct.pack("<I", value << 16))[0]


def decode_raw_tensor(dtype: str, shape: tuple[int, ...], payload: bytes) -> RawTensor:
    count = math.prod(shape)
    if dtype == "F32":
        values = list(struct.unpack(f"<{count}f", payload))
    elif dtype == "F16":
        raw = struct.unpack(f"<{count}H", payload)
        values = [fp16_to_float(value) for value in raw]
    elif dtype == "BF16":
        raw = struct.unpack(f"<{count}H", payload)
        values = [bf16_to_float(value) for value in raw]
    else:
        raise RuntimeError(f"unsupported safetensors dtype: {dtype}")
    return RawTensor(shape=shape, values=values)


def load_safetensors_fallback(path: Path) -> dict[str, RawTensor]:
    data = path.read_bytes()
    header_size = struct.unpack("<Q", data[:8])[0]
    header_end = 8 + header_size
    header = json.loads(data[8:header_end])
    tensors: dict[str, RawTensor] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        shape = tuple(int(value) for value in meta["shape"])
        payload = data[header_end + start : header_end + end]
        tensors[name] = decode_raw_tensor(meta["dtype"], shape, payload)
    return tensors


def load_state_dict(model_dir: Path) -> dict[str, Any]:
    safetensors_files = sorted(model_dir.glob("*.safetensors"))
    if safetensors_files:
        state: dict[str, Any] = {}
        try:
            from safetensors.torch import load_file
        except ImportError:
            for path in safetensors_files:
                state.update(load_safetensors_fallback(path))
        else:
            for path in safetensors_files:
                state.update(load_file(path, device="cpu"))
        return state

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch checkpoint export requires torch to be installed") from exc

    for name in ("pytorch_model.bin", "model.pth"):
        path = model_dir / name
        if path.exists():
            loaded = torch.load(path, map_location="cpu")
            break
    else:
        pth_files = sorted(model_dir.glob("*.pth"))
        if not pth_files:
            raise FileNotFoundError(f"no safetensors, pytorch_model.bin, or .pth checkpoint found in {model_dir}")
        loaded = torch.load(pth_files[0], map_location="cpu")

    if isinstance(loaded, dict) and "model" in loaded and isinstance(loaded["model"], dict):
        loaded = loaded["model"]
    if isinstance(loaded, dict) and "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
        loaded = loaded["state_dict"]
    if not isinstance(loaded, dict):
        raise RuntimeError("unsupported checkpoint format")
    return loaded


def tensor_order(config: MiniMindConfig) -> list[str]:
    names = list(TENSOR_ORDER_PREFIX)
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}."
        names.extend(
            [
                prefix + "input_layernorm.weight",
                prefix + "post_attention_layernorm.weight",
                prefix + "self_attn.q_norm.weight",
                prefix + "self_attn.k_norm.weight",
                prefix + "self_attn.q_proj.weight",
                prefix + "self_attn.k_proj.weight",
                prefix + "self_attn.v_proj.weight",
                prefix + "self_attn.o_proj.weight",
                prefix + "mlp.gate_proj.weight",
                prefix + "mlp.up_proj.weight",
                prefix + "mlp.down_proj.weight",
            ]
        )
    return names


def write_string(handle, value: str) -> None:
    data = value.encode("utf-8")
    handle.write(struct.pack("<Q", len(data)))
    handle.write(data)


def tensor_to_float_list(tensor: Any) -> list[float]:
    if isinstance(tensor, RawTensor):
        return tensor.values
    if hasattr(tensor, "detach"):
        tensor = tensor.detach().cpu().float().contiguous()
        return tensor.reshape(-1).tolist()
    if hasattr(tensor, "astype"):
        tensor = tensor.astype("float32")
        return tensor.reshape(-1).tolist()
    return [float(value) for value in tensor]


def shape_of(tensor: Any) -> tuple[int, ...]:
    if isinstance(tensor, RawTensor):
        return tensor.shape
    return tuple(int(value) for value in tensor.shape)


def write_runtime_config(config: MiniMindConfig, output_dir: Path) -> None:
    lines = [
        f"hidden_size={config.hidden_size}",
        f"num_hidden_layers={config.num_hidden_layers}",
        f"use_moe={int(config.use_moe)}",
        f"vocab_size={config.vocab_size}",
        f"num_attention_heads={config.num_attention_heads}",
        f"num_key_value_heads={config.num_key_value_heads}",
        f"head_dim={config.head_dim}",
        f"intermediate_size={config.intermediate_size}",
        f"max_position_embeddings={config.max_position_embeddings}",
        f"tie_word_embeddings={int(config.tie_word_embeddings)}",
        f"moe_intermediate_size={config.moe_intermediate_size}",
    ]
    (output_dir / "minimind_runtime_config.txt").write_text("\n".join(lines) + "\n")


def export_weights(model_dir: Path, output_dir: Path) -> None:
    config = load_config(model_dir)
    config_errors = validate_config(config)
    if config_errors:
        raise RuntimeError("invalid config: " + config_errors[0])
    if config.use_moe:
        raise RuntimeError("runtime export currently supports dense MiniMind checkpoints only")

    state = load_state_dict(model_dir)
    expected = expected_shapes(config)
    if "lm_head.weight" not in state and config.tie_word_embeddings:
        state["lm_head.weight"] = state["model.embed_tokens.weight"]

    errors: list[str] = []
    for name in tensor_order(config):
        if name not in state:
            errors.append(f"missing tensor: {name}")
            continue
        expected_shape, _required = expected[name]
        actual_shape = shape_of(state[name])
        if actual_shape != expected_shape:
            errors.append(f"shape mismatch for {name}: expected {expected_shape}, got {actual_shape}")
    if errors:
        raise RuntimeError("\n".join(errors[:20]))

    output_dir.mkdir(parents=True, exist_ok=True)
    write_runtime_config(config, output_dir)
    with (output_dir / "weights.bin").open("wb") as handle:
        handle.write(b"MMRTW001")
        for name in tensor_order(config):
            values = tensor_to_float_list(state[name])
            write_string(handle, name)
            handle.write(struct.pack("<Q", len(values)))
            handle.write(struct.pack(f"<{len(values)}f", *values))

    tokenizer = model_dir / "tokenizer.json"
    if tokenizer.exists():
        shutil.copy2(tokenizer, output_dir / "tokenizer.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export dense MiniMind weights into the local runtime format")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_weights(args.model, args.output)
    print(f"exported runtime weights to {args.output}")


if __name__ == "__main__":
    main()
