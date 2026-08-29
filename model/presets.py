# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""Named architecture presets. `crow-300m-final` is the thesis model; `tiny` is the ~240K-parameter test shrink."""

from typing import Any

CROW_300M_FINAL = dict(
    name="crow-300m-final",
    block_size=2048,
    vocab_size=32000,
    padding_multiple=2048,
    tie_embeddings=True,
    num_attention_heads=16,
    n_embd=1024,
    intermediate_size=4096,
    norm_eps=1e-6,
    qk_bias=True,
    n_layers_in_prelude=2,
    n_layers_in_recurrent_block=[4, 4, 4],
    n_layers_in_coda=2,
    mean_recurrence=12,
    mean_backprop_depth=8,
)

TINY = dict(
    CROW_300M_FINAL,
    name="tiny",
    block_size=256,
    vocab_size=512,
    padding_multiple=512,
    num_attention_heads=4,
    n_embd=64,
    intermediate_size=128,
    n_layers_in_prelude=2,
    n_layers_in_recurrent_block=[1, 1],
    n_layers_in_coda=1,
    mean_recurrence=2,
    mean_backprop_depth=2,
)

PRESETS: dict[str, dict[str, Any]] = {"crow-300m-final": CROW_300M_FINAL, "tiny": TINY}
