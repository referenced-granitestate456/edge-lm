"""Benchmark prefill latency and decode throughput for MLX gemma4 models.

Methodology (follows benchmark_llm.py):
    TTFT  = min time to generate 1 token (prefill + 1 decode step)
    Full  = min time to generate N tokens (prefill + N decode steps)
    TPS   = (N - 1) / (Full - TTFT)

Usage:
    python benchmarks/performance.py --model mlx_model --input-tokens 1024 --output-tokens 1024
    python benchmarks/performance.py --model mlx_model --input-tokens 1024 --output-tokens 1024 --compare-ref --compare-ref-4bit
"""

import argparse
import timeit

import mlx.core as mx
import mlx.nn as nn

from edge_lm.models.load import load, set_prefill_logits_to_keep


def make_input_ids(tokenizer, n_tokens: int) -> mx.array:
    text = "The quick brown fox jumps over the lazy dog. " * (n_tokens // 8 + 1)
    ids = tokenizer.encode(text)[:n_tokens]
    return mx.array([ids], dtype=mx.int32)


def _prefill(model, input_ids, cache, step_size):
    """Run prefill, optionally in `step_size`-token chunks.

    Chunking lowers peak memory: feeding the prompt in pieces (sharing the KV
    cache) means MLX evaluates fewer positions per `mx.eval`, so it doesn't
    hold the whole prompt's activation working set live at once. Mirrors
    mlx-lm's `prefill_step_size`. `step_size` <= 0 (or prompt shorter than it)
    runs a single forward.
    """
    seq_len = input_ids.shape[1]
    if not step_size or step_size <= 0 or seq_len <= step_size:
        out = model(input_ids, cache=cache)
        return out.logits if hasattr(out, "logits") else out

    logits = None
    for i in range(0, seq_len, step_size):
        out = model(input_ids[:, i:i + step_size], cache=cache)
        logits = out.logits if hasattr(out, "logits") else out
        # Force this chunk (including its KV-cache writes) to execute before
        # building the next chunk's graph, so activations are freed in between.
        kv = [c.keys for c in cache if getattr(c, "keys", None) is not None]
        kv += [c.values for c in cache if getattr(c, "values", None) is not None]
        mx.eval(logits, *kv)
    return logits


def generate_n_tokens(model, input_ids, n_tokens: int, prefill_step_size=None):
    cache = model.language_model.make_cache()
    # IMPORTANT: extract the logits ARRAY before mx.eval. The model returns a
    # LanguageModelOutput object; mx.eval on the object is a no-op (not an
    # mx.array), so the prior ordering never forced the lazy graph to execute --
    # timeit then measured graph CONSTRUCTION, not GPU execution, inflating TPS
    # ~5x and faking TTFT/peak RAM (mx.synchronize alone doesn't help: it only
    # waits on already-submitted work).
    logits = _prefill(model, input_ids, cache, prefill_step_size)
    mx.eval(logits)
    for _ in range(n_tokens - 1):
        next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        logits = model(next_token, cache=cache)
        if hasattr(logits, "logits"):
            logits = logits.logits
        mx.eval(logits)


def measure_peak_memory(model, input_ids, output_tokens, prefill_step_size=None):
    """Run one full generation; return MLX peak GPU bytes.

    MLX peak = mx.get_peak_memory(): bytes from MLX's own Metal allocator only
               (weights + activations + KV that MLX tracks), reset per call.
    """
    mx.reset_peak_memory()
    generate_n_tokens(model, input_ids, output_tokens, prefill_step_size)
    return mx.get_peak_memory()


def bench_model(model, input_ids, output_tokens, repeat, warmup, label="",
                prefill_step_size=None):
    if label:
        print(f"\n--- {label} ---")

    for _ in range(warmup):
        generate_n_tokens(model, input_ids, 1, prefill_step_size)

    def sync_and_generate(n):
        def run():
            generate_n_tokens(model, input_ids, n, prefill_step_size)
        return run

    print("  Measuring TTFT...")
    ttft_runs = timeit.repeat(
        sync_and_generate(1),
        number=1, repeat=repeat,
        setup="import mlx.core as mx; mx.eval(mx.zeros(1))",
    )
    ttft = min(ttft_runs)

    print(f"  Measuring decode ({output_tokens} tokens)...")
    full_runs = timeit.repeat(
        sync_and_generate(output_tokens),
        number=1, repeat=repeat,
        setup="import mlx.core as mx; mx.eval(mx.zeros(1))",
    )
    full = min(full_runs)

    print("  Measuring peak memory...")
    peak_mem = measure_peak_memory(model, input_ids, output_tokens, prefill_step_size)

    tps = (output_tokens - 1) / (full - ttft)

    return {"ttft": ttft, "full": full, "tps": tps, "peak_mem": peak_mem}


def load_ref_model(hf_model: str, quantize_4bit: bool = False,
                   group_size: int = 64, mode: str = "affine"):
    from mlx_vlm import load as load_vlm
    model, processor = load_vlm(hf_model)

    if quantize_4bit:
        predicate = getattr(model, "quant_predicate", None)
        if predicate:
            nn.quantize(model, group_size=group_size, bits=4, mode=mode,
                        class_predicate=predicate)
        else:
            nn.quantize(model, group_size=group_size, bits=4, mode=mode)
        mx.eval(model.parameters())

    return model, processor.tokenizer


def print_results(results: list[dict]):
    width = 82
    print(f"\n{'=' * width}")
    header = (f"{'Model':<30s} {'TTFT (ms)':>10s} {'Full (ms)':>10s} "
              f"{'TPS':>10s} {'MLX mem':>12s}")
    print(header)
    print("-" * width)
    for r in results:
        mem_str = f"{r['peak_mem'] / 1e6:.0f} MB"
        print(f"{r['label']:<30s} {r['ttft']*1000:>10.1f} {r['full']*1000:>10.1f} "
              f"{r['tps']:>10.1f} {mem_str:>12s}")
    print(f"{'=' * width}")
    print("MLX mem = mx.get_peak_memory() (MLX Metal allocator only).")


def main():
    parser = argparse.ArgumentParser(description="Benchmark MLX gemma4 models")
    parser.add_argument("--model", type=str, default="TheStageAI/gemma-4-E2B-it")
    parser.add_argument("--input-tokens", type=int, default=128)
    parser.add_argument("--output-tokens", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--compare-ref", action="store_true", help="Compare with bf16 reference model")
    parser.add_argument("--compare-ref-4bit", action="store_true", help="Compare with 4-bit quantized reference")
    parser.add_argument("--ref-4bit-mode", type=str, default="affine",
                        help="Quantization mode for the 4-bit reference (affine, mxfp4, nvfp4)")
    parser.add_argument("--ref-4bit-group-size", type=int, default=64,
                        help="Group size for the 4-bit reference (nvfp4 requires 16, mxfp4 requires 32)")
    parser.add_argument("--hf-model", type=str, default="google/gemma-4-E2B-it")
    parser.add_argument("--prefill-step-size", type=int, default=256,
                        help="Process the prompt in chunks of this many tokens "
                             "(lowers prefill peak memory; <=0 disables chunking)")
    parser.add_argument("--cache-limit-mb", type=int, default=None,
                        help="Cap MLX's freed-buffer cache pool (MB). Lower = "
                             "return freed buffers to the OS sooner.")
    args = parser.parse_args()

    if args.cache_limit_mb is not None:
        prev = mx.set_cache_limit(args.cache_limit_mb * 1024 * 1024)
        print(f"MLX cache limit: {prev/1e6:.0f} MB -> {args.cache_limit_mb} MB")

    print(f"Input tokens:  {args.input_tokens}")
    print(f"Output tokens: {args.output_tokens}")
    print(f"Repeats:       {args.repeat}")
    print(f"Prefill step:  {args.prefill_step_size}")

    all_results = []

    # Our model
    print(f"\nLoading {args.model}...")
    model, tokenizer = load(args.model)
    set_prefill_logits_to_keep(model, 1)  # generation only needs last-token logits
    input_ids = make_input_ids(tokenizer, args.input_tokens)
    actual_len = input_ids.shape[1]

    r = bench_model(model, input_ids, args.output_tokens, args.repeat, args.warmup,
                    label="TheStage", prefill_step_size=args.prefill_step_size)
    r["label"] = f"TheStage ({args.model})"
    all_results.append(r)
    del model

    # Reference bf16
    if args.compare_ref:
        print(f"\nLoading reference bf16 {args.hf_model}...")
        ref_model, ref_tok = load_ref_model(args.hf_model, quantize_4bit=False)
        ref_ids = make_input_ids(ref_tok, args.input_tokens)

        r = bench_model(ref_model, ref_ids, args.output_tokens, args.repeat, args.warmup,
                        label="Ref bf16", prefill_step_size=args.prefill_step_size)
        r["label"] = "Ref bf16 (mlx-vlm)"
        all_results.append(r)
        del ref_model

    # Reference 4-bit
    if args.compare_ref_4bit:
        print(f"\nLoading reference 4-bit {args.hf_model} "
              f"(mode={args.ref_4bit_mode}, group_size={args.ref_4bit_group_size})...")
        ref4_model, ref4_tok = load_ref_model(
            args.hf_model, quantize_4bit=True,
            group_size=args.ref_4bit_group_size, mode=args.ref_4bit_mode)
        ref4_ids = make_input_ids(ref4_tok, args.input_tokens)

        r = bench_model(ref4_model, ref4_ids, args.output_tokens, args.repeat, args.warmup,
                        label="Ref 4-bit", prefill_step_size=args.prefill_step_size)
        r["label"] = f"Ref 4-bit {args.ref_4bit_mode} gs{args.ref_4bit_group_size}"
        all_results.append(r)
        del ref4_model

    print(f"\nInput tokens: {actual_len}")
    print_results(all_results)


if __name__ == "__main__":
    main()
