"""Compact AQLM Per-Layer Embedding (PLE) for Gemma 4.

Gemma 4's per-layer embeddings, materialized, are ~4.7 GB. Here they are stored
as AQLM codes + codebooks (~296 MB) and decompressed on the fly with a single
batched gather, so the full table never has to live in memory.

`AQLMPLEEmbedding` is a drop-in replacement for the `nn.Embedding` at
`embed_tokens_per_layer`: its `__call__(input_ids)` returns
`[*input_ids.shape, num_layers * ple_dim]`, exactly the shape the stock
`Gemma4TextModel.get_per_layer_inputs` consumes — no forward patching needed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import safetensors.numpy


# ---------------------------------------------------------------------------
# Bit unpacking
# ---------------------------------------------------------------------------

def _packed_index_bytes(numel: int, bit_width: int) -> int:
    return int(math.ceil(numel * bit_width / 8))


def _unpack_uint_indices(packed: np.ndarray, shape: tuple[int, ...], bit_width: int) -> np.ndarray:
    count = int(np.prod(shape))
    packed = packed.astype(np.uint8).ravel()[:_packed_index_bytes(count, bit_width)]

    if bit_width == 8:
        return packed[:count].astype(np.uint8).reshape(shape)
    if bit_width == 4:
        low = (packed & 0x0F).astype(np.uint8)
        high = ((packed >> 4) & 0x0F).astype(np.uint8)
        return np.stack([low, high], axis=-1).ravel()[:count].reshape(shape)

    lanes = np.arange(bit_width, dtype=np.uint64)
    out = np.empty(count, dtype=np.int64)
    chunk = max(8, (1_048_576 // 8) * 8)
    for start in range(0, count, chunk):
        stop = min(start + chunk, count)
        n = stop - start
        bs = (start * bit_width) // 8
        be = bs + _packed_index_bytes(n, bit_width)
        bits = np.unpackbits(packed[bs:be], bitorder="little")[:n * bit_width]
        out[start:stop] = (bits.reshape(n, bit_width).astype(np.uint64) << lanes).sum(axis=1).astype(np.int64)
    return out.astype(np.uint8).reshape(shape)


# ---------------------------------------------------------------------------
# Compact PLE Embedding (296 MB instead of 4.7 GB)
# ---------------------------------------------------------------------------

class AQLMPLEEmbedding(nn.Module):
    """PLE stored as AQLM codes + codebooks (296 MB vs 4.7 GB materialized).

    Drop-in replacement for the ``nn.Embedding`` at ``embed_tokens_per_layer``:
    ``__call__(input_ids)`` returns ``[*input_ids.shape, num_layers * ple_dim]``,
    exactly the shape the stock ``Gemma4TextModel.get_per_layer_inputs`` consumes
    (it applies ``embed_tokens_per_layer_scale`` afterwards). No forward patching
    is needed — the stock per-layer path drives this module unchanged.
    """

    def __init__(self, codes, codebooks, num_layers, num_groups, group_size, ple_dim, embed_scale):
        super().__init__()
        self.codes = codes            # [L, G, V] uint8
        self.codebooks = codebooks    # [L, G, K, D] float16
        self.num_layers = num_layers
        self.num_groups = num_groups
        self.group_size = group_size
        self.ple_dim = ple_dim
        self._embed_scale = embed_scale

        # Pre-computed index arrays for vectorized gathers
        LG = num_layers * num_groups
        self._codes_flat = codes.reshape(LG, -1)         # [L*G, V]
        self._codebooks_flat = codebooks.reshape(LG, -1, group_size)  # [L*G, K, D]
        self._lg_idx = mx.arange(LG)                     # [L*G]
        self._l_idx = mx.arange(num_layers).reshape(num_layers, 1, 1)
        self._g_idx = mx.arange(num_groups).reshape(1, num_groups, 1)

    def lookup_all_single_token(self, token_id: mx.array) -> mx.array:
        """All layers for 1 token via single flat gather. Returns [1, L*G*D]."""
        all_codes = self._codes_flat[:, token_id[0]]              # [L*G]
        gathered = self._codebooks_flat[self._lg_idx, all_codes]  # [L*G, D]
        return gathered.reshape(1, -1)

    def lookup_all_batched(self, flat_ids: mx.array) -> mx.array:
        """All layers for N tokens via batched gather. Returns [N, L*G*D]."""
        N = flat_ids.shape[0]
        # codes[:, :, flat_ids] → [L, G, N]
        batch_codes = self.codes[:, :, flat_ids]
        # codebooks[l, g, batch_codes[l, g, n]] → [L, G, N, D]
        gathered = self.codebooks[self._l_idx, self._g_idx, batch_codes]
        # → [N, L, G, D] → [N, L*G*D]
        return gathered.transpose(2, 0, 1, 3).reshape(N, -1)

    def __call__(self, input_ids: mx.array) -> mx.array:
        """Full lookup: picks optimal strategy based on token count."""
        input_shape = input_ids.shape
        flat_ids = input_ids.reshape(-1)
        N = flat_ids.shape[0]

        if N == 1:
            out = self.lookup_all_single_token(flat_ids)
        else:
            out = self.lookup_all_batched(flat_ids)

        return out.reshape(*input_shape, self.num_layers * self.ple_dim)


def _load_ple_compact(ple_path: Path) -> AQLMPLEEmbedding:
    # Load codebooks via mlx (supports bfloat16), codes via numpy
    ple_data = mx.load(str(ple_path))
    codebooks_mx = ple_data["codebooks"]

    with safetensors.numpy.safe_open(str(ple_path), framework="numpy") as f:
        meta = f.metadata()
        codes_packed = f.get_tensor("codes_packed")

    stages = int(meta["stages"])
    codes_np = _unpack_uint_indices(codes_packed, tuple(json.loads(meta["codes_shape"])), int(meta["index_storage_bits"]))
    if stages == 1:
        codes_np = codes_np[:, :, :, 0]
        codebooks_mx = codebooks_mx[:, :, 0, :, :]

    codes = mx.array(codes_np.astype(np.uint8))
    codebooks = codebooks_mx

    print(f"PLE compact: codes {codes.shape} ({codes.nbytes / 1e6:.0f} MB) + "
          f"codebooks {codebooks.shape} ({codebooks.nbytes / 1e6:.1f} MB)")

    return AQLMPLEEmbedding(
        codes=codes, codebooks=codebooks,
        num_layers=int(meta["num_layers"]), num_groups=int(meta["num_groups"]),
        group_size=int(meta["group_size"]), ple_dim=int(meta["ple_dim"]),
        embed_scale=float(meta["embed_scale"]),
    )
