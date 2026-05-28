#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MiniMindConfig:
    hidden_size: int = 768
    num_hidden_layers: int = 8
    use_moe: bool = False
    vocab_size: int = 6400
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int = 96
    intermediate_size: int = 2432
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    tie_word_embeddings: bool = True
    hidden_act: str = "silu"
    num_experts: int = 4
    num_experts_per_tok: int = 1
    moe_intermediate_size: int = 2432


def default_intermediate_size(hidden_size: int) -> int:
    return math.ceil(hidden_size * math.pi / 64) * 64


def load_config(model_dir: Path) -> MiniMindConfig:
    config_path = model_dir / "config.json"
    data: dict[str, Any] = {}
    if config_path.exists():
        data = json.loads(config_path.read_text())

    hidden_size = int(data.get("hidden_size", 768))
    num_attention_heads = int(data.get("num_attention_heads", 8))
    intermediate_size = int(data.get("intermediate_size", default_intermediate_size(hidden_size)))
    use_moe = bool(data.get("use_moe", data.get("num_experts", 0) not in (0, None)))

    return MiniMindConfig(
        hidden_size=hidden_size,
        num_hidden_layers=int(data.get("num_hidden_layers", data.get("n_layers", 8))),
        use_moe=use_moe,
        vocab_size=int(data.get("vocab_size", 6400)),
        num_attention_heads=num_attention_heads,
        num_key_value_heads=int(data.get("num_key_value_heads", 4)),
        head_dim=int(data.get("head_dim", hidden_size // num_attention_heads)),
        intermediate_size=intermediate_size,
        max_position_embeddings=int(data.get("max_position_embeddings", 32768)),
        rms_norm_eps=float(data.get("rms_norm_eps", 1e-6)),
        rope_theta=float(data.get("rope_theta", 1_000_000.0)),
        tie_word_embeddings=bool(data.get("tie_word_embeddings", True)),
        hidden_act=str(data.get("hidden_act", "silu")),
        num_experts=int(data.get("num_experts", 4)),
        num_experts_per_tok=int(data.get("num_experts_per_tok", 1)),
        moe_intermediate_size=int(data.get("moe_intermediate_size", data.get("intermediate_size", intermediate_size))),
    )


def validate_config(config: MiniMindConfig) -> list[str]:
    errors: list[str] = []
    positive_fields = [
        "hidden_size",
        "num_hidden_layers",
        "vocab_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "intermediate_size",
        "max_position_embeddings",
    ]
    for field in positive_fields:
        if getattr(config, field) <= 0:
            errors.append(f"{field} must be positive")
    if config.hidden_size % config.num_attention_heads != 0:
        errors.append("hidden_size must be divisible by num_attention_heads")
    if config.head_dim != config.hidden_size // config.num_attention_heads:
        errors.append("head_dim must equal hidden_size / num_attention_heads")
    if config.num_attention_heads % config.num_key_value_heads != 0:
        errors.append("num_attention_heads must be divisible by num_key_value_heads")
    if config.hidden_act != "silu":
        errors.append("only silu hidden_act is currently supported")
    if config.rms_norm_eps <= 0:
        errors.append("rms_norm_eps must be positive")
    if config.rope_theta <= 0:
        errors.append("rope_theta must be positive")
    if config.use_moe:
        if config.num_experts <= 0:
            errors.append("num_experts must be positive")
        if config.num_experts_per_tok != 1:
            errors.append("only top-1 MoE routing is currently supported")
        if config.moe_intermediate_size <= 0:
            errors.append("moe_intermediate_size must be positive")
    return errors


def expected_shapes(config: MiniMindConfig) -> dict[str, tuple[tuple[int, ...], bool]]:
    expected: dict[str, tuple[tuple[int, ...], bool]] = {
        "model.embed_tokens.weight": ((config.vocab_size, config.hidden_size), True),
        "model.norm.weight": ((config.hidden_size,), True),
        "lm_head.weight": ((config.vocab_size, config.hidden_size), not config.tie_word_embeddings),
    }
    q_out = config.num_attention_heads * config.head_dim
    kv_out = config.num_key_value_heads * config.head_dim
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}."
        expected[prefix + "input_layernorm.weight"] = ((config.hidden_size,), True)
        expected[prefix + "post_attention_layernorm.weight"] = ((config.hidden_size,), True)
        expected[prefix + "self_attn.q_proj.weight"] = ((q_out, config.hidden_size), True)
        expected[prefix + "self_attn.k_proj.weight"] = ((kv_out, config.hidden_size), True)
        expected[prefix + "self_attn.v_proj.weight"] = ((kv_out, config.hidden_size), True)
        expected[prefix + "self_attn.o_proj.weight"] = ((config.hidden_size, q_out), True)
        expected[prefix + "self_attn.q_norm.weight"] = ((config.head_dim,), True)
        expected[prefix + "self_attn.k_norm.weight"] = ((config.head_dim,), True)
        if config.use_moe:
            expected[prefix + "mlp.gate.weight"] = ((config.num_experts, config.hidden_size), True)
            for expert in range(config.num_experts):
                expert_prefix = prefix + f"mlp.experts.{expert}."
                expected[expert_prefix + "gate_proj.weight"] = ((config.moe_intermediate_size, config.hidden_size), True)
                expected[expert_prefix + "up_proj.weight"] = ((config.moe_intermediate_size, config.hidden_size), True)
                expected[expert_prefix + "down_proj.weight"] = ((config.hidden_size, config.moe_intermediate_size), True)
        else:
            expected[prefix + "mlp.gate_proj.weight"] = ((config.intermediate_size, config.hidden_size), True)
            expected[prefix + "mlp.up_proj.weight"] = ((config.intermediate_size, config.hidden_size), True)
            expected[prefix + "mlp.down_proj.weight"] = ((config.hidden_size, config.intermediate_size), True)
    return expected


def safetensors_metadata(path: Path) -> dict[str, tuple[int, ...]]:
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))
    shapes: dict[str, tuple[int, ...]] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        shapes[name] = tuple(int(value) for value in meta["shape"])
    return shapes


def pytorch_zip_metadata(path: Path) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch checkpoint inspection requires torch to be installed") from exc
    state = torch.load(path, map_location="meta")
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise RuntimeError("unsupported PyTorch checkpoint format")
    for name, tensor in state.items():
        if hasattr(tensor, "shape"):
            shapes[name] = tuple(int(value) for value in tensor.shape)
    return shapes


def load_weight_metadata(model_dir: Path) -> dict[str, tuple[int, ...]]:
    safetensors_files = sorted(model_dir.glob("*.safetensors"))
    if safetensors_files:
        shapes: dict[str, tuple[int, ...]] = {}
        for path in safetensors_files:
            shapes.update(safetensors_metadata(path))
        return shapes

    for name in ("pytorch_model.bin", "model.pth"):
        path = model_dir / name
        if path.exists():
            return pytorch_zip_metadata(path)

    pth_files = sorted(model_dir.glob("*.pth"))
    if pth_files:
        return pytorch_zip_metadata(pth_files[0])

    raise FileNotFoundError(f"no safetensors, pytorch_model.bin, or .pth checkpoint found in {model_dir}")


def validate_weights(actual: dict[str, tuple[int, ...]], expected: dict[str, tuple[tuple[int, ...], bool]]) -> list[str]:
    errors: list[str] = []
    for name, (shape, required) in expected.items():
        if name not in actual:
            if required:
                errors.append(f"missing tensor: {name}")
            continue
        if actual[name] != shape:
            errors.append(f"shape mismatch for {name}: expected {shape}, got {actual[name]}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect and validate a MiniMind checkpoint directory")
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--skip-weights", action="store_true")
    args = parser.parse_args()

    config = load_config(args.model_dir)
    config_errors = validate_config(config)
    if config_errors:
        for error in config_errors:
            print(f"config error: {error}", file=sys.stderr)
        return 1

    print(f"config: hidden={config.hidden_size} layers={config.num_hidden_layers} vocab={config.vocab_size} moe={config.use_moe}")

    if args.skip_weights:
        return 0

    actual = load_weight_metadata(args.model_dir)
    errors = validate_weights(actual, expected_shapes(config))
    if errors:
        for error in errors[:50]:
            print(f"weight error: {error}", file=sys.stderr)
        if len(errors) > 50:
            print(f"... {len(errors) - 50} more errors", file=sys.stderr)
        return 1

    print(f"weights: {len(actual)} tensors validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
