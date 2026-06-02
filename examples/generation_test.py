"""Test generation quality: compare exported model vs reference bf16 mlx-vlm model.

Usage:
    python examples/generation_test.py --prompts "What is 2+2?" "Write a haiku"
    python examples/generation_test.py --compare-ref --prompts "What is 2+2?"
    python examples/generation_test.py --use-ref --prompts "What is 2+2?"
"""

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np

from edge_lm.models.load import load
from mlx_vlm import stream_generate


def generate_greedy(model, tokenizer, prompt: str, max_tokens: int = 100) -> str:
    ids = [int(x) for x in tokenizer.encode(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
    )]
    # Pass input_ids directly to bypass prepare_inputs (text-only); stream_generate
    # handles chunked prefill, greedy decoding (temp=0), and EOS stopping.
    text = ""
    for result in stream_generate(
        model, tokenizer, "", input_ids=mx.array([ids], dtype=mx.int32),
        max_tokens=max_tokens,
    ):
        text += result.text
    return text


def compare_logits(model, ref_model, tokenizer, prompt: str):
    """Compare last-position logits between two models."""
    ids = [int(x) for x in tokenizer.encode(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
    )]
    ids_mx = mx.array([ids], dtype=mx.int32)

    our = model(ids_mx)
    ref = ref_model(ids_mx)
    mx.eval(our, ref)

    our_l = our.logits[0, -1, :] if hasattr(our, "logits") else our[0, -1, :]
    ref_l = ref.logits[0, -1, :] if hasattr(ref, "logits") else ref[0, -1, :]
    mx.eval(our_l, ref_l)

    our_top = int(mx.argmax(our_l).item())
    ref_top = int(mx.argmax(ref_l).item())

    our_l_f32 = our_l.astype(mx.float32)
    ref_l_f32 = ref_l.astype(mx.float32)
    diff = mx.abs(our_l_f32 - ref_l_f32)
    mx.eval(diff)

    return {
        "our_top": (our_top, tokenizer.decode([our_top]), float(our_l[our_top])),
        "ref_top": (ref_top, tokenizer.decode([ref_top]), float(ref_l[ref_top])),
        "match": our_top == ref_top,
        "mean_logit_diff": float(mx.mean(diff)),
        "max_logit_diff": float(mx.max(diff)),
    }


def main():
    parser = argparse.ArgumentParser(description="Test generation vs reference")
    parser.add_argument("--model", type=str, default="TheStageAI/gemma-4-E2B-it")
    parser.add_argument("--size", type=str, default="m", help="Checkpoint size tag (s/m/l)")
    parser.add_argument("--use-ref", action="store_true", help="Use original HF model instead of exported")
    parser.add_argument("--ref-model", type=str, default="google/gemma-4-E2B-it")
    parser.add_argument("--no-ref", action="store_true", default=True, help="Skip reference model comparison")
    parser.add_argument("--compare-ref", action="store_true", help="Also load reference model for comparison")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument(
        "--prompts", nargs="+",
        default=[
            "What is 2+2?",
            "Explain quantum computing in one sentence.",
            "Напиши хайку о луне",
            "Write a Python function to compute fibonacci numbers.",
        ],
    )
    args = parser.parse_args()

    if args.use_ref:
        print(f"Loading original model {args.ref_model}...")
        from mlx_vlm import load as load_vlm
        model, processor = load_vlm(args.ref_model)
        tokenizer = processor.tokenizer
        args.no_ref = True
    else:
        print(f"Loading exported model from {args.model} (size={args.size})...")
        model, tokenizer = load(args.model, size=args.size)

    # Load reference
    ref_model = None
    if args.compare_ref:
        args.no_ref = False
    if not args.no_ref:
        print(f"Loading reference model {args.ref_model}...")
        from mlx_vlm import load as load_ref
        ref_model, ref_processor = load_ref(args.ref_model)
        if not getattr(tokenizer, "chat_template", None):
            tokenizer.chat_template = ref_processor.tokenizer.chat_template

    # --- Logit comparison ---
    if ref_model is not None:
        print(f"\n{'=' * 60}")
        print("LOGIT COMPARISON")
        print(f"{'=' * 60}")

        for prompt in args.prompts:
            info = compare_logits(model, ref_model, tokenizer, prompt)
            top_match = "✓" if info["match"] else "✗"
            print(f"\n  Prompt: {prompt[:50]}")
            print(f"  Ref  top: {info['ref_top'][1]:>20s} (logit {info['ref_top'][2]:6.2f})")
            print(f"  Ours top: {info['our_top'][1]:>20s} (logit {info['our_top'][2]:6.2f}) {top_match}")
            print(f"  Mean logit diff: {info['mean_logit_diff']:.3f}, max: {info['max_logit_diff']:.3f}")

    # --- Generation ---
    print(f"\n{'=' * 60}")
    print("GENERATION" + (" COMPARISON" if ref_model else ""))
    print(f"{'=' * 60}")

    for prompt in args.prompts:
        our_text = generate_greedy(model, tokenizer, prompt, args.max_tokens)
        print(f"\n  Q: {prompt}")
        if ref_model is not None:
            ref_text = generate_greedy(ref_model, tokenizer, prompt, args.max_tokens)
            print(f"  Ref: {ref_text[:200]}")
        print(f"  Our: {our_text[:200]}")

    print(f"\n{'=' * 60}")
    print("Done.")


if __name__ == "__main__":
    main()
