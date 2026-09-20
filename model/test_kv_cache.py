# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Cache storage reuse and attention parity across allocation boundaries."""

import pytest
import torch

from model.generation import KVCache
from model.layers.attention import CausalSelfAttention, precompute_freqs_cis
from model.test_config import tiny_config


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_cache_reuses_storage_and_preserves_prefix_on_growth(dtype: torch.dtype) -> None:
    cache = KVCache()
    initial_key, initial_value = cache.key, cache.value
    assert initial_key is None and initial_value is None
    # Strided inputs, multiple batch rows, exact boundaries, and an append spanning several allocation chunks.
    source = torch.arange(2 * 900 * 3 * 4, dtype=torch.float32).reshape(2, 900, 3, 4).to(dtype)
    keys, values = source[..., ::2], -source[..., ::2]
    end = 0
    previous_capacity = 0
    previous_pointers: tuple[int, int] | None = None
    for length, capacity in [(250, 256), (6, 256), (1, 512), (255, 512), (1, 768), (387, 1024)]:
        start, end = end, end + length
        actual_k, actual_v = cache.append_and_get(keys[:, start:end], values[:, start:end])
        torch.testing.assert_close(actual_k, keys[:, :end], rtol=0, atol=0)
        torch.testing.assert_close(actual_v, values[:, :end], rtol=0, atol=0)
        for populated, actual in ((cache.key, actual_k), (cache.value, actual_v)):
            assert populated is not None
            assert populated.data_ptr() == actual.data_ptr()
            torch.testing.assert_close(populated, actual, rtol=0, atol=0)
        pointers = (actual_k.data_ptr(), actual_v.data_ptr())
        if previous_capacity == capacity:
            assert pointers == previous_pointers
        elif previous_pointers is not None:
            assert all(current != previous for current, previous in zip(pointers, previous_pointers))
        for actual in (actual_k, actual_v):
            assert actual.shape[1] == end
            assert actual.untyped_storage().nbytes() == 2 * capacity * 3 * 2 * actual.element_size()
        previous_capacity, previous_pointers = capacity, pointers


@torch.inference_mode()
def test_attention_cache_matches_full_prefix_across_growth() -> None:
    torch.manual_seed(17)
    config = tiny_config(n_embd=32, model_max_sequence_length=530)
    layer = CausalSelfAttention(config).eval()
    x = torch.randn(2, 530, config.n_embd)
    rotary = precompute_freqs_cis(config.head_size, 530, config.rope_settings.rope_base)
    expected = layer(x, rotary)
    cache = KVCache()
    end = 0
    for length in (250, 6, 1, 260, 13):
        start, end = end, end + length
        mask = torch.arange(end)[None, :] <= torch.arange(start, end)[:, None]
        actual = layer(x[:, start:end], rotary[:, start:end], mask, cache)
        torch.testing.assert_close(actual, expected[:, start:end], atol=3e-5, rtol=2e-5)


@pytest.mark.parametrize("change", ["batch", "heads", "dtype", "length"])
@torch.inference_mode()
def test_cache_rejects_incompatible_append_without_changing_prefix(change: str) -> None:
    cache = KVCache()
    key = torch.ones(2, 3, 4, 8)
    before_k, before_v = cache.append_and_get(key, key)
    incoming = key
    value = key
    if change == "batch":
        incoming = value = key[:1]
    elif change == "heads":
        incoming = value = key[:, :, :1]
    elif change == "dtype":
        incoming = value = key.double()
    else:
        value = key[:, :1]
    with pytest.raises(ValueError, match="cache K/V"):
        cache.append_and_get(incoming, value)
    for populated, before in ((cache.key, before_k), (cache.value, before_v)):
        assert populated is not None
        assert populated.data_ptr() == before.data_ptr()
        torch.testing.assert_close(populated, before, rtol=0, atol=0)
    torch.testing.assert_close(before_k, key)
    torch.testing.assert_close(before_v, key)
