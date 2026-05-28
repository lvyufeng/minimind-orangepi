from __future__ import annotations

import json
import struct
import subprocess
from pathlib import Path


def write_fake_safetensors(path: Path, tensors: dict[str, tuple[list[int], list[float]]]) -> None:
    offset = 0
    header = {}
    payload = bytearray()
    for name, (shape, values) in tensors.items():
        data = struct.pack(f"<{len(values)}f", *values)
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [offset, offset + len(data)]}
        payload.extend(data)
        offset += len(data)
    header_bytes = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + payload)


def test_export_minimind_runtime_from_safetensors(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    model_dir = tmp_path / "model"
    out_dir = tmp_path / "runtime"
    model_dir.mkdir()
    config = {
        "hidden_size": 4,
        "num_hidden_layers": 1,
        "vocab_size": 8,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "intermediate_size": 8,
        "tie_word_embeddings": False,
        "use_moe": False,
    }
    (model_dir / "config.json").write_text(json.dumps(config))

    def values(size: int, value: float) -> list[float]:
        return [value] * size

    tensors = {
        "model.embed_tokens.weight": ([8, 4], values(32, 0.1)),
        "model.norm.weight": ([4], values(4, 1.0)),
        "lm_head.weight": ([8, 4], values(32, 0.2)),
        "model.layers.0.input_layernorm.weight": ([4], values(4, 1.0)),
        "model.layers.0.post_attention_layernorm.weight": ([4], values(4, 1.0)),
        "model.layers.0.self_attn.q_norm.weight": ([2], values(2, 1.0)),
        "model.layers.0.self_attn.k_norm.weight": ([2], values(2, 1.0)),
        "model.layers.0.self_attn.q_proj.weight": ([4, 4], values(16, 0.01)),
        "model.layers.0.self_attn.k_proj.weight": ([2, 4], values(8, 0.01)),
        "model.layers.0.self_attn.v_proj.weight": ([2, 4], values(8, 0.01)),
        "model.layers.0.self_attn.o_proj.weight": ([4, 4], values(16, 0.01)),
        "model.layers.0.mlp.gate_proj.weight": ([8, 4], values(32, 0.01)),
        "model.layers.0.mlp.up_proj.weight": ([8, 4], values(32, 0.01)),
        "model.layers.0.mlp.down_proj.weight": ([4, 8], values(32, 0.01)),
    }
    write_fake_safetensors(model_dir / "model.safetensors", tensors)

    result = subprocess.run(
        [
            "python3",
            str(root / "src" / "python" / "tools" / "export_minimind_runtime.py"),
            "--model",
            str(model_dir),
            "--output",
            str(out_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "exported runtime weights" in result.stdout
    assert (out_dir / "minimind_runtime_config.txt").exists()
    assert (out_dir / "weights.bin").read_bytes().startswith(b"MMRTW001")
