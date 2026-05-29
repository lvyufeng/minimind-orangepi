# MiniMind Orange Pi

MiniMind runtime experiments for Orange Pi / Ascend 310B.

Current status: the text MiniMind LLM path is the only end-to-end runnable path. MiniMind-V and MiniMind-O code exists as early scaffolding/tests, but full image/audio multimodal inference is not supported yet.

## What works now

- Export a MiniMind text checkpoint into the local runtime format.
- Run greedy text generation from C++ or the Python wrapper.
- Use Ascend 310B/CANN for the main decode path when the toolkit is available.
- Use Cube-accelerated matvec for model projections and lm_head argmax.
- Use a resident Ascend attention KV cache for decode.
- Run unit tests and decode benchmarks.

## What is not ready yet

- MiniMind-V end-to-end image-text generation.
- MiniMind-O audio/image/text omni generation.
- Quantized checkpoints.
- Fully fused device-resident decoder layers.
- Stable long-context 20 TPS decode. Short decode is close, but longer decode still drops because the current runtime still has host/device activation round trips between layer operations.

## Requirements

- Linux on Orange Pi / Ascend 310B or a CANN-capable development environment.
- CMake and a C++17 compiler.
- CANN toolkit, typically under one of:
  - `/usr/local/Ascend/cann/latest`
  - `/usr/local/Ascend/cann`
  - `/usr/local/Ascend/cann-8.5.0`
- Python 3 for checkpoint export and wrapper scripts.
- Optional Python packages for checkpoint export depending on checkpoint format:
  - `torch`
  - `safetensors`

Model weights are not committed to this repository.

## Environment

```bash
source scripts/set_env.sh
```

This sets CANN-related paths when available and adds `src/python` to `PYTHONPATH`.

If CANN is installed somewhere else, set `ASCEND_TOOLKIT_ROOT` first:

```bash
export ASCEND_TOOLKIT_ROOT=/path/to/Ascend/cann
source scripts/set_env.sh
```

## Build

```bash
./scripts/build.sh
```

The build script creates `build/` and enables Ascend support when CANN libraries are found.

## Export a MiniMind text checkpoint

Put or download the original MiniMind text checkpoint under a local ignored directory, for example:

```text
models/minimind/
```

Then export it into the local runtime format:

```bash
python3 src/python/tools/export_minimind_runtime.py \
  --model models/minimind \
  --output models/minimind-runtime
```

The output runtime directory contains:

```text
models/minimind-runtime/minimind_runtime_config.txt
models/minimind-runtime/weights.bin
models/minimind-runtime/tokenizer.json   # if present in source checkpoint
```

## Run text generation

C++ tool:

```bash
build/minimind_generate --model models/minimind-runtime --prompt "你好" --max-new-tokens 32
```

Python wrapper:

```bash
python3 src/python/tools/run_text.py \
  --model models/minimind-runtime \
  --prompt "你好" \
  --max-new-tokens 32
```

By default, the Python wrapper applies the MiniMind chat markers. Use `--raw-prompt` to pass the prompt directly.

## Gradio text demo

The Gradio demo is text-only and streams generated tokens into the output box. It does not support MiniMind-V image-text inference or MiniMind-O omni/audio inference.

Install the optional demo dependency:

```bash
python3 -m pip install -r requirements-demo.txt
```

Smoke test without real weights; this intentionally uses the native toy fallback:

```bash
python3 src/python/tools/gradio_text_demo.py --max-new-tokens 2
```

Run with an exported MiniMind text runtime model:

```bash
python3 src/python/tools/gradio_text_demo.py \
  --model models/minimind-runtime \
  --max-new-tokens 32
```

If the runtime directory includes `tokenizer.json`, install `tokenizers` to show decoded text. If `--model` is provided, the demo validates that the directory contains `minimind_runtime_config.txt` and `weights.bin` instead of silently falling back to the toy model.

The native CLI also has an opt-in streaming mode:

```bash
build/minimind_generate \
  --model models/minimind-runtime \
  --prompt "你好" \
  --max-new-tokens 32 \
  --stream
```

Without `--stream`, the CLI keeps the stable non-streaming output format for scripts.

## Benchmark text decode

```bash
build/bench_decode --model models/minimind-runtime --tokens 16 --warmup 1
```

The benchmark reports prefill and decode separately:

```text
prompt_tokens=5
generated_tokens=16
prefill_tokens=5
prefill_seconds=...
prefill_tokens_per_second=...
decode_steps=15
decode_seconds=...
decode_tokens_per_second=...
```

`decode_steps` is `generated_tokens - 1` because the first generated token is produced by the final prefill forward pass. Decode TPS is therefore computed from actual decode forwards only.

Recent Ascend 310B results from this work session:

```text
prompt=hello, prompt_tokens=5, generated_tokens=16:
prefill_tokens_per_second ~= 21.7
decode_tokens_per_second  ~= 17.9

--input-tokens 1, generated_tokens=16:
prefill_tokens_per_second ~= 23.1
decode_tokens_per_second  ~= 19.2
```

So prefill is already above 20 TPS in this benchmark. Decode is close to 20 TPS for short contexts but not yet stable above 20 TPS across longer outputs.

## Tests

Focused tests:

```bash
build/test_decoder_layer
build/test_language_model
build/test_ascend_custom_ops
```

Common full build/test loop:

```bash
./scripts/build.sh
build/test_smoke
build/test_tensor
build/test_minimind_config
build/test_rope
build/test_weights
build/test_decoder_layer
build/test_language_model
build/test_runtime_loader
build/test_ascend_custom_ops
```

## Current performance work

The text path currently includes several Ascend/Cube optimizations:

- fused QKV and gate/up weight layout where supported by the runtime loader;
- transposed cached Cube weights;
- reusable Cube matvec scratch buffers;
- device-side lm_head argmax;
- resident Ascend attention KV cache;
- disabled standalone host-vector custom ops in the decode hot path when they would add extra H2D/D2H transfers;
- separate prefill/decode timing in `bench_decode`.

The next major step for stable 20+ TPS decode is to reduce per-layer host/device activation round trips, likely by introducing chained device-resident matvecs or a fused decode-layer custom operator.
