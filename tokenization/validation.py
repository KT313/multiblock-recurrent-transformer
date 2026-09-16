# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Cheap tokenizer/model contract checks, with no device tensor reads or training forwards."""
from __future__ import annotations

from typing import Any



def check_model_vocabulary(config: Any, contract: dict[str, Any] | None, model: Any = None) -> None:
    if contract is not None and config.vocab_size != contract["vocab_size"]:
        raise ValueError(f"tokenizer {contract['profile']} has {contract['vocab_size']} usable IDs but model vocab_size={config.vocab_size}; physical padding does not make those IDs valid")
    padded = config.padded_vocab_size
    if padded is None or padded < config.vocab_size:
        raise ValueError("padded_vocab_size must contain the real vocabulary")
    if model is None:
        return
    if model.config.vocab_size != config.vocab_size or model.config.padded_vocab_size != padded:
        raise ValueError("actual model and supplied config disagree on vocabulary dimensions")
    embedding, head = model.transformer.wte, model.lm_head
    expected = (padded, config.n_embd)
    if (tuple(embedding.weight.shape) != expected or embedding.num_embeddings != padded
            or embedding.embedding_dim != config.n_embd):
        raise ValueError(f"embedding shape/attributes disagree with expected {expected}: {tuple(embedding.weight.shape)}")
    if tuple(head.weight.shape) != expected or head.out_features != padded or head.in_features != config.n_embd:
        raise ValueError(f"LM head shape/attributes disagree with expected {expected}: {tuple(head.weight.shape)}")
    if head.bias is not None and tuple(head.bias.shape) != (padded,):
        raise ValueError("LM head bias does not match padded vocabulary")
    if (embedding.weight is head.weight) != config.tie_embeddings:
        raise ValueError("embedding/head parameter tying disagrees with tie_embeddings")
    if (not config.tie_embeddings and not embedding.weight.is_meta and not head.weight.is_meta
            and embedding.weight.untyped_storage().data_ptr() == head.weight.untyped_storage().data_ptr()):
        raise ValueError("untied embedding/head parameters unexpectedly share storage")
