"""Load an exported MLX model from a directory or HuggingFace repo.

Currently ships Gemma 4 (TheStage compressed variant). Expects:

    model_dir/
        config.json              — shared model config (architecture)
        model_{size}.safetensors — quantized decoder weights (s/m/l);
                                    per-size quantization map in its metadata
        ple_{size}.safetensors   — PLE codes + codebooks (compact, per-size;
                                    falls back to a shared ple.safetensors)
        vision_tower.safetensors — optional, shared
        audio_tower.safetensors  — optional, shared
        tokenizer.json
        tokenizer_config.json

Legacy single-size checkpoints (model.safetensors + quantization in
config.json) are still supported as a fallback.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import safetensors.numpy
from mlx.utils import tree_map

from edge_lm.models.gemma.embeddings import _load_ple_compact


# ---------------------------------------------------------------------------
# Prefill logits trimming (drop all-but-last hidden states before lm_head)
# ---------------------------------------------------------------------------

def set_prefill_logits_to_keep(model, n: int = 1):
    """Keep only the last `n` positions' logits (set n=0 to restore all).

    Generation only consumes the final position's logits, so during prefill
    the full [B, S, vocab] tensor is wasted work + memory (~0.5 GB at S=1024,
    vocab=262144). Enabling this trims the text model's hidden states to the
    last token before the lm_head projection. Safe to leave on: single-token
    decode steps are unaffected; loglikelihood / logit-comparison paths that
    never call this keep all positions (default 0).
    """
    _install_keep_last_logits()
    model.language_model.model._num_logits_to_keep = int(n)


_keep_last_logits_installed = False


def _install_keep_last_logits():
    """Class-level wrap of Gemma4TextModel.__call__ that trims returned hidden
    states to the last `_num_logits_to_keep` positions when that attribute is
    set on the instance.

    Class-level (not instance) because Python dispatches ``obj()`` via the type,
    so an instance ``__call__`` attribute is ignored. The trim runs at the very
    end of the forward — after KV-cache writes and any ``hidden_sink`` capture —
    so caches, speculative-decoding hidden capture, and mlx-vlm's chunked prefill
    are all unaffected; only the tensor fed to ``embed_tokens.as_linear`` shrinks.
    Instances without ``_num_logits_to_keep`` (ref models, scoring paths) are
    untouched by the guard, so wrapping the shared class is side-effect free."""
    global _keep_last_logits_installed
    if _keep_last_logits_installed:
        return
    from mlx_vlm.models.gemma4.language import Gemma4TextModel

    orig_call = Gemma4TextModel.__call__

    def _call_keep_last(self, *args, **kwargs):
        h = orig_call(self, *args, **kwargs)
        keep = int(getattr(self, "_num_logits_to_keep", 0) or 0)
        if keep and isinstance(h, mx.array) and h.ndim >= 2 and h.shape[-2] > keep:
            h = h[..., -keep:, :]
        return h

    Gemma4TextModel.__call__ = _call_keep_last
    _keep_last_logits_installed = True


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "TheStageAI/gemma-4-E2B-it"
DEFAULT_SIZE = "m"


def _resolve_model_files(
    model_id: str | Path,
    size: Optional[str],
    include_vision: bool,
    include_audio: bool,
) -> tuple[Path, str, str]:
    """Resolve the model directory, decoder filename and PLE filename for the size.

    For HuggingFace repos, only the files actually needed are downloaded
    (config, the size-specific decoder, the size-specific PLE, tokenizer, and any
    requested towers) — not the whole repo. The PLE may be per-size
    (ple_{size}.safetensors) or, for legacy checkpoints, a single shared
    ple.safetensors. Returns (model_dir, model_filename, ple_filename).
    """
    model_filename = f"model_{size}.safetensors" if size else "model.safetensors"
    ple_filename = f"ple_{size}.safetensors" if size else "ple.safetensors"
    local = Path(model_id)
    if local.is_dir() and (local / "config.json").exists():
        if not (local / model_filename).exists() and (local / "model.safetensors").exists():
            print(f"{model_filename} not found, falling back to model.safetensors")
            model_filename = "model.safetensors"
        if not (local / ple_filename).exists() and (local / "ple.safetensors").exists():
            ple_filename = "ple.safetensors"
        return local, model_filename, ple_filename

    from huggingface_hub import hf_hub_download

    def fetch(fname: str, required: bool = True) -> Optional[str]:
        try:
            return hf_hub_download(str(model_id), fname)
        except Exception:
            if required:
                raise
            return None

    config_path = fetch("config.json")
    model_dir = Path(config_path).parent

    if size and fetch(model_filename, required=False) is None:
        print(f"{model_filename} not found in repo, falling back to model.safetensors")
        model_filename = "model.safetensors"
    fetch(model_filename)

    if size and fetch(ple_filename, required=False) is None:
        ple_filename = "ple.safetensors"
    fetch(ple_filename, required=False)
    for tf in ("tokenizer.json", "tokenizer_config.json"):
        fetch(tf, required=False)
    if include_vision:
        fetch("vision_tower.safetensors", required=False)
    if include_audio:
        fetch("audio_tower.safetensors", required=False)

    return model_dir, model_filename, ple_filename


def _read_quantization(model_path: Path, config: dict) -> dict:
    """Read the per-size quantization map from the model file's metadata,
    falling back to config.json for legacy single-size checkpoints."""
    try:
        with safetensors.numpy.safe_open(str(model_path), framework="numpy") as f:
            meta = f.metadata() or {}
        if "quantization" in meta:
            return json.loads(meta["quantization"])
    except Exception:
        pass
    return config.get("quantization", {})


def load(
    model_id: str | Path = DEFAULT_MODEL,
    size: Optional[str] = DEFAULT_SIZE,
    lazy: bool = False,
    include_vision: bool = False,
    include_audio: bool = False,
) -> tuple:
    """Load model and tokenizer from a local directory or HuggingFace repo.

    Args:
        model_id: local path or HuggingFace repo id (default: TheStageAI/gemma-4-E2B-it)
        size: checkpoint size tag s/m/l (default "m"); loads model_{size}.safetensors.
            Pass None to load the legacy model.safetensors. Falls back to
            model.safetensors if the size-specific file is absent.
        include_vision: load vision tower (from vision_tower.safetensors)
        include_audio: load audio tower (from audio_tower.safetensors)

    Returns (model, tokenizer).
    """
    model_dir, model_filename, ple_filename = _resolve_model_files(model_id, size, include_vision, include_audio)

    with open(model_dir / "config.json") as f:
        config = json.load(f)

    from mlx_vlm.models.gemma4.config import ModelConfig, TextConfig, VisionConfig, AudioConfig
    from mlx_vlm.models.gemma4.gemma4 import Model

    model_config = ModelConfig.from_dict(config)
    for attr, cls in (("text_config", TextConfig), ("vision_config", VisionConfig),
                      ("audio_config", AudioConfig)):
        sub = getattr(model_config, attr)
        if isinstance(sub, dict):
            setattr(model_config, attr, cls.from_dict(sub))

    model_path = model_dir / model_filename
    weights = mx.load(str(model_path))

    # Load optional vision/audio tower weights
    vision_path = model_dir / "vision_tower.safetensors"
    audio_path = model_dir / "audio_tower.safetensors"

    if include_vision and vision_path.exists():
        print("Loading vision tower...")
        weights.update(mx.load(str(vision_path)))
    if include_audio and audio_path.exists():
        print("Loading audio tower...")
        weights.update(mx.load(str(audio_path)))

    # Disable audio/vision if not loaded
    if not include_audio or not any("audio_tower" in k for k in weights):
        model_config.audio_config = None

    model = Model(model_config)

    quantization = _read_quantization(model_path, config)
    if quantization:
        def class_predicate(path, module):
            if not hasattr(module, "to_quantized"):
                return False
            if path in quantization:
                return quantization[path]
            # Vision/audio towers carry their own quantization. Infer the group
            # size from the scales shape so any group size works (E2B uses 64,
            # E4B uses 32) without hardcoding or extra metadata.
            if any(x in path for x in ("vision_tower", "audio_tower", "embed_vision", "embed_audio")):
                skey = f"{path}.scales"
                if skey in weights:
                    group_size = module.weight.shape[-1] // weights[skey].shape[-1]
                    return {"bits": 4, "group_size": group_size}
                return False
            return f"{path}.scales" in weights

        nn.quantize(
            model,
            group_size=quantization.get("group_size", 64),
            bits=quantization.get("bits", 4),
            class_predicate=class_predicate,
        )

    # Replace embed_tokens_per_layer with the compact AQLM embedding. It is a
    # drop-in for the stock nn.Embedding (same __call__ contract), so the stock
    # Gemma4TextModel.get_per_layer_inputs path drives it with no forward patch.
    ple_path = model_dir / ple_filename
    if ple_path.exists():
        model.language_model.model.embed_tokens_per_layer = _load_ple_compact(ple_path)

    model.load_weights(list(weights.items()), strict=False)

    # Drop unused vision tower to free RAM
    if not include_vision and hasattr(model, "vision_tower"):
        del model.vision_tower
        del model.embed_vision

    # Cast any leftover float32 params to bfloat16
    model.update(tree_map(
        lambda p: p.astype(mx.bfloat16) if p.dtype == mx.float32 else p,
        model.parameters(),
    ))

    if not lazy:
        mx.eval(model.parameters())
    model.eval()

    # Build an mlx-vlm-ready tokenizer so mlx_vlm.stream_generate works directly:
    # load_tokenizer wraps the HF tokenizer with a streaming detokenizer, and
    # StoppingCriteria (from the checkpoint's eos_token_id = [1, 106] = <eos>,
    # <turn|>) is what stream_generate checks to stop generation.
    from mlx_vlm.tokenizer_utils import load_tokenizer
    from mlx_vlm.utils import StoppingCriteria

    tokenizer = load_tokenizer(model_dir)

    if not getattr(tokenizer, "chat_template", None):
        # Some checkpoints ship no chat_template; borrow gemma's. apply_chat_template
        # is forwarded to the inner tokenizer, so set it there (not on the wrapper).
        from transformers import AutoTokenizer
        hf_id = config.get("_name_or_path", "google/gemma-4-E2B-it")
        try:
            tokenizer._tokenizer.chat_template = AutoTokenizer.from_pretrained(hf_id).chat_template
        except Exception:
            pass

    tokenizer.stopping_criteria = StoppingCriteria(
        config.get("eos_token_id") or [1, 106], tokenizer,
    )

    return model, tokenizer
