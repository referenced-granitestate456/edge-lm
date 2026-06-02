"""Test audio capabilities: transcribe or describe audio.

Usage:
    python examples/test_audio.py --audio recording.wav
    python examples/test_audio.py --audio recording.wav --prompt "Transcribe this speech"
    python examples/test_audio.py --use-ref --audio recording.wav
"""

import argparse
import sys
import warnings

import mlx.core as mx
from mlx_vlm.utils import prepare_inputs

warnings.filterwarnings("ignore", message="At least one mel filter")

from edge_lm.models.load import load


def generate_with_audio(model, processor, tokenizer, audio_path, prompt, max_tokens=500):
    """Run audio through the model and generate response."""
    messages = [{"role": "user", "content": [
        {"type": "audio", "audio": audio_path},
        {"type": "text", "text": prompt},
    ]}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = prepare_inputs(processor, audio=[audio_path], prompts=formatted)

    input_ids = inputs["input_ids"]
    kwargs = {}
    for key in ("input_features", "input_features_mask", "audio_features", "audio_mask",
                "feature_attention_mask", "audio_feature_lengths"):
        if key in inputs and inputs[key] is not None:
            kwargs[key] = inputs[key]

    cache = model.language_model.make_cache()
    logits = model(input_ids, cache=cache, **kwargs)
    mx.eval(logits)
    if hasattr(logits, "logits"):
        logits = logits.logits

    for _ in range(max_tokens):
        next_token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        if next_token in (1, 106):
            break
        sys.stdout.write(tokenizer.decode([next_token]))
        sys.stdout.flush()
        logits = model(mx.array([[next_token]], dtype=mx.int32), cache=cache)
        mx.eval(logits)
        if hasattr(logits, "logits"):
            logits = logits.logits
    print()


def main():
    parser = argparse.ArgumentParser(description="Test audio with Gemma 4")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument("--prompt", type=str, default="Transcribe this speech.")
    parser.add_argument("--model", type=str, default="TheStageAI/gemma-4-E2B-it")
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--use-ref", action="store_true", help="Use original HF model")
    args = parser.parse_args()

    if args.use_ref:
        print(f"Loading reference model google/gemma-4-E2B-it...")
        from mlx_vlm import load as load_vlm
        model, processor = load_vlm("google/gemma-4-E2B-it")
        tokenizer = processor.tokenizer
    else:
        print(f"Loading {args.model} with audio...")
        model, tokenizer = load(args.model, include_audio=True)
        from mlx_vlm import load as load_vlm
        _, processor = load_vlm("google/gemma-4-E2B-it")

    print(f"\nAudio: {args.audio}")
    print(f"Prompt: {args.prompt}")
    print(f"Response: ", end="")
    generate_with_audio(model, processor, tokenizer, args.audio, args.prompt, args.max_tokens)


if __name__ == "__main__":
    main()
