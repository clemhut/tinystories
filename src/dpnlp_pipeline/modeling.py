from __future__ import annotations

from src.dpnlp_pipeline.config import ModelSpec
from src.transformer.transformer import Transformer


def build_model(vocab_size: int, model_spec: ModelSpec, max_seq_len: int) -> Transformer:
    return Transformer(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=model_spec.d_model,
        n_layers=model_spec.n_layers,
        d_ff=model_spec.d_ff,
        n_heads=model_spec.n_heads,
    )
