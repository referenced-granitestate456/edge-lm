"""Test vision capabilities: describe an image.

Usage:
    python examples/test_vision.py --image photo.jpg
    python examples/test_vision.py --image photo.jpg --prompt "What objects are in this image?"
    python examples/test_vision.py --use-ref --image photo.jpg
"""

import argparse
import sys
import warnings

import mlx.core as mx
from mlx_vlm.utils import prepare_inputs

from edge_lm.models.load import load

warnings.filterwarnings("ignore", message="At least one mel filter")


def generate_with_image(model, processor, tokenizer, image_path, prompt, max_tokens=500):
    """Run image through the model and generate response."""
    from PIL import Image
    image = Image.open(image_path).convert("RGB")

    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": prompt},
    ]}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = prepare_inputs(processor, images=[image], prompts=formatted)

    input_ids = inputs["input_ids"]
    kwargs = {}
    if "pixel_values" in inputs and inputs["pixel_values"] is not None:
        kwargs["pixel_values"] = inputs["pixel_values"]

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
    parser = argparse.ArgumentParser(description="Test vision with Gemma 4")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--prompt", type=str, default="Describe this image in detail.")
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
        print(f"Loading {args.model} with vision...")
        model, tokenizer = load(args.model, include_vision=True)
        from mlx_vlm import load as load_vlm
        _, processor = load_vlm("google/gemma-4-E2B-it")

    print(f"\nImage: {args.image}")
    print(f"Prompt: {args.prompt}")
    print(f"Response: ", end="")
    generate_with_image(model, processor, tokenizer, args.image, args.prompt, args.max_tokens)


if __name__ == "__main__":
    main()
