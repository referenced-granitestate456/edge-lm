---
license: mit
base_model:
  - google/gemma-4-E4B-it
base_model_relation: quantized
library_name: mlx
pipeline_tag: image-text-to-text
tags:
  - mlx
  - gemma
  - gemma-4
  - edge
  - on-device
  - apple-silicon
  - quantization
  - gptq
  - aqlm
  - ple
language:
  - en
  - multilingual
---

# TheStageAI/gemma-4-E4B-it

A compressed, edge-ready variant of Google's **Gemma 4 E4B (instruction-tuned)**, packaged for
[MLX](https://github.com/ml-explore/mlx) on Apple Silicon Macs and iPhones. The checkpoint fits in
**~2.6 GB** — small enough to download quickly and stay within mobile memory budgets — while
preserving the capabilities that matter most for on-device assistants: general world knowledge,
instruction following, and tool use.

- **Run it with:** [`TheStageAI/edge-lm`](https://github.com/TheStageAI/edge-lm)
- **Base model:** [`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it)
- **Sibling release:** [`TheStageAI/gemma-4-E2B-it`](https://huggingface.co/TheStageAI/gemma-4-E2B-it)
- **Write-up:** *7× size reduction for Gemma 4 Edge models — Compressing PLE architectures.*

## Why this exists

Gemma 4 E4B is a "4B" model by *effective* parameter count, but the dense checkpoint is closer to
**8B** parameters once Per-Layer Embeddings (PLE) are counted — and in BF16 the PLE table dominates the
footprint. On mobile hardware, three things block deployment: download size, runtime memory footprint
(iOS enforces a ~3 GB per-app budget), and generation speed. We compress the model along its natural
structure to address all three at once.

## How it was compressed

- **Transformer blocks** — GPTQ with Quantization Error Propagation (QEP) and range clipping, emitted
  as flat, MLX-compatible per-group weight-only tensors.
- **PLE tables** — an AQLM-style vector-quantization codec with sensitivity-weighted (Fisher-style)
  assignments, decompressed on the fly with a single batched gather across all layers.
- **Token embeddings / LM head** — flat per-group scalar quantization matched to the same runtime contract.
- **Bit-width schedule** — chosen per module by Riemannian Constrained Optimization (RCO) under an exact
  byte budget; the release checkpoint is re-quantized from the dense model in one consistent GPTQ/QEP pass.

## Operating points

This repo ships two release operating points, selected via the `size` argument:

| `size` | Trade-off | Compression |
|---|---|---|
| `l` | More quality, larger artifact | 4.64× |
| `m` | Smaller headline target (**default**) | **5.60×** |

It also includes optional 4-bit vision and audio towers for image understanding and audio transcription.

## Usage

```bash
git clone https://github.com/TheStageAI/edge-lm.git
pip install -e edge-lm
```

```python
from edge_lm import load
from mlx_vlm import stream_generate

model, tokenizer = load("TheStageAI/gemma-4-E4B-it", size="l")  # use "m" for the smaller target

prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Explain gravity in one sentence."}],
    tokenize=False, add_generation_prompt=True,
)
for chunk in stream_generate(model, tokenizer, prompt, max_tokens=128):
    print(chunk.text, end="", flush=True)
```

Vision and audio (loads the optional towers):

```python
model, tokenizer = load("TheStageAI/gemma-4-E4B-it", include_vision=True)   # image understanding
model, tokenizer = load("TheStageAI/gemma-4-E4B-it", include_audio=True)    # audio transcription
```

Only the files needed for the requested size are downloaded.

## Benchmarks

Every model — ours and the GGUF baselines — is dequantized to a standard BF16 checkpoint and served
through vLLM, so the backend is equalized. We report **MMLU-Pro** (general knowledge), **IFEval**
(instruction following), and **τ²-Bench / Tau2** (multi-step tool use). For Tau2 the Gemma checkpoint
acts as the agent while a fixed `Qwen3-235B-A22B-2507` simulates the user.

| Model | Compression | MMLU-Pro | IFEval | Tau2 |
|---|---|---|---|---|
| BF16 (reference) | 1.00× | 70.49 | 81.33 | 37.19 |
| **Ours L** | 4.64× | 67.41 | 81.52 | **33.25** |
| **Ours M** | **5.60×** | 63.54 | **80.78** | 29.04 |
| Unsloth Q3-K-S | 3.90× | **63.66** | 77.08 | 30.47 |
| Unsloth UD-Q2-K-XL | 4.01× | 58.69 | 79.67 | 22.91 |

Bold marks the best result among the compressed checkpoints in each column.

## Files

| File | Contents |
|---|---|
| `config.json` | Shared model config (architecture) |
| `model_{s,m,l}.safetensors` | Quantized decoder weights per operating point (quantization map in metadata) |
| `ple_{s,m,l}.safetensors` | Compact AQLM PLE codes + codebooks |
| `vision_tower.safetensors` | Optional 4-bit vision tower |
| `audio_tower.safetensors` | Optional 4-bit audio tower |
| `tokenizer.json`, `tokenizer_config.json` | Tokenizer |

## License

Released under the [MIT License](https://github.com/TheStageAI/edge-lm/blob/main/LICENSE),
© 2025 thestage.ai labs. As a derivative of Google's Gemma 4, the weights are additionally subject
to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).

## Citation

If you use these checkpoints, please cite the Gemma 4 release and the methods we build on
(GPTQ, QEP, AQLM, RCO) — see the references in the [edge-lm](https://github.com/TheStageAI/edge-lm)
write-up.
