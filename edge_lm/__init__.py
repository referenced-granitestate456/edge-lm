"""edge-lm — tiny LLMs on Apple Silicon via MLX.

Quickstart:

    from edge_lm import load
    model, tokenizer = load()  # TheStageAI/gemma-4-E2B-it by default
"""

from edge_lm.models.load import load

__version__ = "0.1.0"
__all__ = ["load"]
