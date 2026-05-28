from __future__ import annotations

import json
import subprocess
from pathlib import Path


def test_inspect_minimind_config_only(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "vocab_size": 16,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "intermediate_size": 16,
                "tie_word_embeddings": True,
            }
        )
    )

    result = subprocess.run(
        [
            "python3",
            str(root / "src" / "python" / "tools" / "inspect_minimind_checkpoint.py"),
            str(model_dir),
            "--skip-weights",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "hidden=8" in result.stdout
    assert "moe=False" in result.stdout
