# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Tests for `model.model.RecurrentGPT`: gradient flow through every layer, step handling, the recurrence
sampler, loss masking, gradient checkpointing and the seeded golden forward (numerics regression guard).
"""

from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import Tensor

from model import build_model
from model.test_config import TINY_ARCHITECTURE, tiny_config
import model.model as model_module
from model.blocks import recurrence
from model.blocks.recurrence import sample_recurrence_steps
from model.blocks.sandwich import SandwichBlock
from model.layers.attention import precompute_freqs_cis
from model.model import RecurrentGPT, TransformerModules

GOLDEN_PATH = Path(__file__).with_name("golden_tiny_forward.pt")
VOCAB = 512


def ids(batch: int = 2, seq: int = 32, seed: int = 1) -> torch.Tensor:
    return torch.randint(0, VOCAB, (batch, seq), generator=torch.Generator().manual_seed(seed))


def core_block(model: RecurrentGPT, idx: int) -> torch.nn.ModuleList:
    block = model.transformer.core_blocks[idx]
    assert isinstance(block, torch.nn.ModuleList)
    return block


def per_block(values: int | list[int]) -> list[int]:
    """
    `RecurrentConfig` normalizes per-block fields to lists; narrow the declared union for the type checkers.
    """

    assert isinstance(values, list)
    return values


def seeded_tiny(seed: int = 0, **kwargs: Any) -> RecurrentGPT:
    torch.manual_seed(seed)
    return build_model(TINY_ARCHITECTURE, **kwargs)


# --- structure -------------------------------------------------------------------------------------------------------


def test_structure_follows_config(tiny_model: RecurrentGPT) -> None:
    cfg = tiny_model.config
    t = tiny_model.transformer
    assert len(t.prelude) == cfg.n_layers_in_prelude == 2
    assert [len(core_block(tiny_model, i)) for i in range(2)] == cfg.n_layers_in_recurrent_block == [1, 1]
    assert len(t.adapters) == len(t.ln_fs) == 2
    assert len(t.coda) == cfg.n_layers_in_coda == 1
    assert t.wte.weight.shape == (cfg.padded_vocab_size, cfg.n_embd)
    assert tiny_model.lm_head.weight is t.wte.weight  # tied
    assert tiny_model.freqs_cis.shape == (1, cfg.block_size, 1, cfg.head_size // 2, 2)
    assert tiny_model.emb_scale == pytest.approx(cfg.n_embd**0.5)
    assert sum(p.numel() for p in tiny_model.parameters()) == 256_256


def test_state_dict_keys_are_pinned(tiny_model: RecurrentGPT) -> None:
    """
    Module and parameter names of the tiny model (checkpoint / HF export compatibility): recorded from the thesis
    code, any change here breaks every saved checkpoint.
    """

    expected = [
        "freqs_cis",
        "transformer.wte.weight",
        "transformer.prelude.0.norm_1.weight",
        "transformer.prelude.0.attn.qk_bias",
        "transformer.prelude.0.attn.Wqkv.weight",
        "transformer.prelude.0.attn.proj.weight",
        "transformer.prelude.0.norm_2.weight",
        "transformer.prelude.0.mlp.fc.weight",
        "transformer.prelude.0.mlp.proj.weight",
        "transformer.prelude.0.norm_3.weight",
        "transformer.prelude.0.norm_4.weight",
        "transformer.prelude.1.norm_1.weight",
        "transformer.prelude.1.attn.qk_bias",
        "transformer.prelude.1.attn.Wqkv.weight",
        "transformer.prelude.1.attn.proj.weight",
        "transformer.prelude.1.norm_2.weight",
        "transformer.prelude.1.mlp.fc.weight",
        "transformer.prelude.1.mlp.proj.weight",
        "transformer.prelude.1.norm_3.weight",
        "transformer.prelude.1.norm_4.weight",
        "transformer.adapters.0.weight",
        "transformer.adapters.1.weight",
        "transformer.core_blocks.0.0.norm_1.weight",
        "transformer.core_blocks.0.0.attn.qk_bias",
        "transformer.core_blocks.0.0.attn.Wqkv.weight",
        "transformer.core_blocks.0.0.attn.proj.weight",
        "transformer.core_blocks.0.0.norm_2.weight",
        "transformer.core_blocks.0.0.mlp.fc.weight",
        "transformer.core_blocks.0.0.mlp.proj.weight",
        "transformer.core_blocks.0.0.norm_3.weight",
        "transformer.core_blocks.0.0.norm_4.weight",
        "transformer.core_blocks.1.0.norm_1.weight",
        "transformer.core_blocks.1.0.attn.qk_bias",
        "transformer.core_blocks.1.0.attn.Wqkv.weight",
        "transformer.core_blocks.1.0.attn.proj.weight",
        "transformer.core_blocks.1.0.norm_2.weight",
        "transformer.core_blocks.1.0.mlp.fc.weight",
        "transformer.core_blocks.1.0.mlp.proj.weight",
        "transformer.core_blocks.1.0.norm_3.weight",
        "transformer.core_blocks.1.0.norm_4.weight",
        "transformer.coda.0.norm_1.weight",
        "transformer.coda.0.attn.qk_bias",
        "transformer.coda.0.attn.Wqkv.weight",
        "transformer.coda.0.attn.proj.weight",
        "transformer.coda.0.norm_2.weight",
        "transformer.coda.0.mlp.fc.weight",
        "transformer.coda.0.mlp.proj.weight",
        "transformer.coda.0.norm_3.weight",
        "transformer.coda.0.norm_4.weight",
        "transformer.ln_fs.0.weight",
        "transformer.ln_fs.0.bias",
        "transformer.ln_fs.1.weight",
        "transformer.ln_fs.1.bias",
        "transformer.ln_final.weight",
        "transformer.ln_final.bias",
        "lm_head.weight",
    ]
    assert list(tiny_model.state_dict()) == expected


def test_build_model_kwargs_routing() -> None:
    m = seeded_tiny(ignore_index=-1, gradient_checkpointing=True, n_layers_in_coda=3)
    assert m.ignore_index == -1 and m.gradient_checkpointing is True
    assert len(m.transformer.coda) == 3
    cfg = tiny_config()
    with pytest.raises(ValueError, match="overrides"):
        build_model(cfg, n_embd=32)
    assert isinstance(build_model(cfg), RecurrentGPT)
    assert build_model(cfg).config is cfg


def test_transformer_module_dict_and_buffer() -> None:
    m = seeded_tiny()
    assert isinstance(m.transformer, TransformerModules)
    assert set(m.transformer.keys()) == {"wte", "prelude", "adapters", "core_blocks", "coda", "ln_fs", "ln_final"}
    assert "freqs_cis" in dict(m.named_buffers())  # persistent buffer -> part of the state dict
    assert "freqs_cis" in m.state_dict()


def test_precompute_freqs_cis_method_matches_function() -> None:
    m = seeded_tiny()
    cfg = m.config
    expected = precompute_freqs_cis(cfg.head_size, cfg.block_size, cfg.rope_settings.rope_base)
    assert torch.equal(m._precompute_freqs_cis(), expected)
    assert torch.equal(m.freqs_cis, expected)


def test_reset_parameters_reinitializes_embedding_and_norms_only() -> None:
    m = seeded_tiny()
    first = m.transformer.prelude[0]
    assert isinstance(first, SandwichBlock)
    block_weight = first.attn.Wqkv.weight.clone()
    with torch.no_grad():
        m.transformer.wte.weight.zero_()
        m.transformer.ln_final.weight.fill_(3.0)
        m.transformer.ln_final.bias.fill_(3.0)
        for ln in m.transformer.ln_fs:
            assert isinstance(ln, torch.nn.LayerNorm)
            ln.weight.fill_(2.0)
    m.reset_parameters()
    assert m.transformer.wte.weight.std().item() == pytest.approx(m.config.init.table["std"], rel=0.1)
    assert torch.equal(m.transformer.ln_final.weight, torch.ones(64))
    assert torch.equal(m.transformer.ln_final.bias, torch.zeros(64))
    for ln in m.transformer.ln_fs:
        assert isinstance(ln, torch.nn.LayerNorm)
        assert torch.equal(ln.weight, torch.ones(64))
    assert torch.equal(first.attn.Wqkv.weight, block_weight)
    assert m.lm_head.weight is m.transformer.wte.weight  # tie survives the reset


def test_iterate_forward_matches_manual_loop(tiny_model: RecurrentGPT) -> None:
    freqs = tiny_model.freqs_cis[:, :6]
    x = torch.randn(1, 6, 64)
    block = core_block(tiny_model, 1)
    torch.manual_seed(4)
    got = tiny_model.run_core_block(x, freqs, None, (2, 1), 1)
    torch.manual_seed(4)
    x_base = tiny_model.transformer.ln_fs[1](x)
    latent = torch.randn_like(x)
    for _ in range(3):
        latent = recurrence.core_block_forward(latent, x_base, freqs, None, tiny_model.transformer.adapters[1], block)
    torch.testing.assert_close(got, latent)
    assert got.requires_grad
    no_grad = tiny_model.run_core_block(x, freqs, None, (2, 0), 1)
    assert not no_grad.requires_grad  # all steps under no_grad


# --- train forward/backward --------------------------------------------------------------------------------------------


def test_train_forward_backward_every_layer_gets_gradient(tiny_model: RecurrentGPT) -> None:
    x = ids()
    out = tiny_model(x, labels=x, return_logits=True)
    assert out["logits"].shape == (2, 32, VOCAB)
    assert out["logits"].dtype == torch.float32
    assert out["loss"].ndim == 0 and torch.isfinite(out["loss"])
    assert torch.equal(out["log_ppl"], out["loss"].detach())
    out["loss"].backward()

    t = tiny_model.transformer
    groups = {
        **{f"prelude.{i}": layer for i, layer in enumerate(t.prelude)},
        **{f"core_blocks.{i}": block for i, block in enumerate(t.core_blocks)},
        **{f"adapters.{i}": a for i, a in enumerate(t.adapters)},
        **{f"ln_fs.{i}": n for i, n in enumerate(t.ln_fs)},
        **{f"coda.{i}": layer for i, layer in enumerate(t.coda)},
        "ln_final": t.ln_final,
        "wte": t.wte,
    }
    for name, module in groups.items():
        for pname, p in module.named_parameters():
            assert p.grad is not None, f"{name}.{pname} has no grad"
            assert p.grad.abs().sum() > 0, f"{name}.{pname} has an all-zero grad"


def test_prelude_layers_chain(tiny_model: RecurrentGPT) -> None:
    """
    Each prelude layer feeds the next: zeroing layer 0's contribution changes what layer 1 sees.
    """

    x = ids()
    seen_in: list[Tensor] = []
    seen_out: list[Tensor] = []

    def hook(_m: torch.nn.Module, inputs: tuple[Tensor, ...], output: Tensor) -> None:
        seen_in.append(inputs[0].clone())
        seen_out.append(output.clone())

    handles = [layer.register_forward_hook(hook) for layer in tiny_model.transformer.prelude]
    tiny_model(x, num_steps=(1, 1))
    for h in handles:
        h.remove()
    assert len(seen_in) == 2
    torch.testing.assert_close(seen_in[0], tiny_model.transformer.wte(x) * tiny_model.emb_scale)
    assert torch.equal(seen_in[1], seen_out[0])  # layer 1 consumes layer 0's output, not the embedding
    assert not torch.allclose(seen_in[0], seen_in[1])


def test_forward_matches_hand_composed_pipeline(tiny_model: RecurrentGPT) -> None:
    """
    Spell out the architecture (CLAUDE.md "RecurrentGPT forward") with explicit steps and check the logits: scaled
    embedding -> chained prelude -> per block: iterate (norm, random latent, n + k iterations) then residual onto the
    block input, which becomes the next block's input -> coda -> ln_final -> tied head -> fp32 logits.
    """

    steps = [(2, 1), (1, 2)]
    x = ids(1, 10)
    torch.manual_seed(9)
    got = tiny_model(x, return_logits=True, num_steps=steps)["logits"]

    t = tiny_model.transformer
    freqs = tiny_model.freqs_cis[:, :10]
    torch.manual_seed(9)
    h = t.wte(x) * tiny_model.config.n_embd**0.5
    for layer in t.prelude:
        h = layer(h, freqs, None)
    for idx, (n, k) in enumerate(steps):
        x_base = t.ln_fs[idx](h)
        latent = torch.randn_like(h)
        for _ in range(n + k):
            latent = t.adapters[idx](torch.cat([latent, x_base], dim=-1))
            for layer in core_block(tiny_model, idx):
                latent = layer(latent, freqs, None)
        h = latent + h
    for layer in t.coda:
        h = layer(h, freqs, None)
    expected = (t.ln_final(h) @ t.wte.weight.T).float()
    torch.testing.assert_close(got, expected)


def test_untied_embeddings_give_a_separate_head() -> None:
    m = seeded_tiny(tie_embeddings=False)
    assert m.lm_head.weight is not m.transformer.wte.weight
    assert m.lm_head.weight.shape == m.transformer.wte.weight.shape
    assert not torch.equal(m.lm_head.weight, m.transformer.wte.weight)
    assert m.lm_head.weight.std().item() == pytest.approx(m.config.init.table["std"], rel=0.1)
    assert sum(p.numel() for p in m.parameters()) == 256_256 + 512 * 64


def test_no_labels_gives_zero_loss_and_no_logits(tiny_model: RecurrentGPT) -> None:
    out = tiny_model(ids())
    assert out["logits"] is None
    assert out["loss"].item() == 0.0 and out["log_ppl"].item() == 0.0


# --- eval mode and explicit steps -------------------------------------------------------------------------------------


def test_eval_sampler_returns_mean_recurrence_and_zero_grad_steps(tiny_model: RecurrentGPT) -> None:
    tiny_model.eval()
    for block_idx, mean in enumerate(per_block(tiny_model.config.mean_recurrence)):
        n, k = tiny_model.sample_block_depths(block_idx)
        assert n.dtype == torch.long and k.dtype == torch.long
        assert (n.item(), k.item()) == (mean, 0)


def test_eval_forward_is_deterministic_under_a_seed_and_equals_explicit_mean_steps(
    tiny_model: RecurrentGPT, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Eval mode draws no recurrence depth, but the latent init still consumes the global RNG, so determinism holds
    under a fixed seed (not across two un-seeded calls).
    """

    tiny_model.eval()
    x = ids()
    torch.manual_seed(5)
    a = tiny_model(x, return_logits=True)["logits"]
    torch.manual_seed(5)
    b = tiny_model(x, return_logits=True)["logits"]
    assert torch.equal(a, b)
    # Explicit (mean_recurrence, 0) steps compute the same thing, but the sampler's own `torch.rand((1,))` draw (one per
    # block, kept for bit-identity with the thesis code) is skipped, so the latent init differs under the same seed...
    explicit = [(m, 0) for m in per_block(tiny_model.config.mean_recurrence)]
    torch.manual_seed(5)
    c = tiny_model(x, return_logits=True, num_steps=explicit)["logits"]
    assert not torch.equal(a, c)
    # ... and is bit-identical once that draw is replayed (per block: latent init first, then the sampler's rand).
    orig = recurrence.initialize_state  # the function `model.model` calls

    def replay(t: Tensor) -> Tensor:
        latent = orig(t)
        torch.rand((1,))
        return latent

    monkeypatch.setattr(model_module, "initialize_state", replay)
    torch.manual_seed(5)
    d = tiny_model(x, return_logits=True, num_steps=explicit)["logits"]
    assert torch.equal(a, d)


def test_num_steps_pair_broadcast_pair_equals_per_block_list(tiny_model: RecurrentGPT) -> None:
    x = ids()
    torch.manual_seed(3)
    a = tiny_model(x, return_logits=True, num_steps=(1, 2))["logits"]
    torch.manual_seed(3)
    b = tiny_model(x, return_logits=True, num_steps=[(1, 2), (1, 2)])["logits"]
    torch.manual_seed(3)
    c = tiny_model(x, return_logits=True, num_steps=[torch.tensor([1, 2]), torch.tensor([1, 2])])["logits"]
    torch.manual_seed(3)
    d = tiny_model(x, return_logits=True, num_steps=torch.tensor([1, 2]))["logits"]
    assert torch.equal(a, b) and torch.equal(a, c) and torch.equal(a, d)


def test_scalar_steps_mean_no_grad_only(tiny_model: RecurrentGPT) -> None:
    x = ids()
    torch.manual_seed(3)
    a = tiny_model(x, return_logits=True, num_steps=3)["logits"]
    torch.manual_seed(3)
    b = tiny_model(x, return_logits=True, num_steps=(3, 0))["logits"]
    assert torch.equal(a, b)


def test_num_steps_pair_list_length_mismatch_raises(tiny_model: RecurrentGPT) -> None:
    with pytest.raises(ValueError, match="num_steps has 3 entries but there are 2 blocks"):
        tiny_model(ids(), num_steps=[(1, 1), (1, 1), (1, 1)])


def test_per_block_depths_actually_differ(tiny_model: RecurrentGPT) -> None:
    x = ids()
    torch.manual_seed(3)
    a = tiny_model(x, return_logits=True, num_steps=[(1, 1), (1, 1)])["logits"]
    torch.manual_seed(3)
    b = tiny_model(x, return_logits=True, num_steps=[(1, 1), (4, 1)])["logits"]
    assert not torch.allclose(a, b)


@pytest.mark.parametrize(("n", "k"), [(2, 1), (1, 2), (0, 3), (3, 0)])
def test_first_n_iterations_run_without_grad_and_last_k_with_grad(
    tiny_model: RecurrentGPT, monkeypatch: pytest.MonkeyPatch, n: int, k: int
) -> None:
    """
    (n, k) is not symmetric: exactly the first n core-block applications happen under no_grad.
    """

    grad_modes: list[bool] = []
    orig = recurrence.core_block_forward

    def spy(*args: Any, **kwargs: Any) -> Tensor:
        grad_modes.append(torch.is_grad_enabled())
        return orig(*args, **kwargs)

    monkeypatch.setattr(recurrence, "core_block_forward", spy)
    tiny_model.run_core_block(torch.randn(1, 4, 64), tiny_model.freqs_cis[:, :4], None, (n, k), 0)
    assert grad_modes == [False] * n + [True] * k


def test_no_grad_steps_cut_gradient_when_k_is_zero(tiny_model: RecurrentGPT) -> None:
    x = ids()
    tiny_model(x, labels=x, num_steps=(2, 0))["loss"].backward()
    for block in tiny_model.transformer.core_blocks:
        for p in block.parameters():
            assert p.grad is None or p.grad.abs().sum() == 0
    for p in tiny_model.transformer.prelude.parameters():  # residual path still carries gradient
        assert p.grad is not None and p.grad.abs().sum() > 0


def test_position_ids_select_rope_rows(tiny_model: RecurrentGPT) -> None:
    """
    `position_ids` index the RoPE table along the sequence axis: a shifted contiguous range gives the same logits
    (relative positions unchanged), a permuted one does not. S=6 differs from every other table axis (1, 256, 8, 2).
    """

    tiny_model.eval()
    x = ids(1, 6)
    torch.manual_seed(1)
    a = tiny_model(x, return_logits=True)["logits"]
    torch.manual_seed(1)
    b = tiny_model(x, position_ids=torch.arange(6), return_logits=True)["logits"]
    assert torch.equal(a, b)
    torch.manual_seed(1)
    c = tiny_model(x, position_ids=torch.arange(100, 106), return_logits=True)["logits"]
    torch.testing.assert_close(a, c, atol=1e-4, rtol=1e-4)
    torch.manual_seed(1)
    d = tiny_model(x, position_ids=torch.tensor([0, 1, 2, 30, 31, 32]), return_logits=True)["logits"]
    assert not torch.allclose(a, d)


# --- prepare_attention_inputs -----------------------------------------------------------------------------------------


def freqs_table(sequence_length: int = 8, head_size: int = 4) -> Tensor:
    return precompute_freqs_cis(head_size, sequence_length + 4, 50_000.0)


def test_prepare_attention_inputs_without_arguments_is_the_plain_causal_path() -> None:
    """
    The training path: the first S rows and no mask at all, so `attention_sdpa` keeps `is_causal=True`.
    """

    table = freqs_table()
    x = ids(2, 6)
    rotary, mask = model_module.prepare_attention_inputs(table, x)
    assert mask is None
    assert torch.equal(rotary, table[:, :6]) and rotary.shape[0] == 1


def test_prepare_attention_inputs_gathers_rope_rows_per_sequence() -> None:
    table = freqs_table()
    x = ids(2, 3)
    one_d, _ = model_module.prepare_attention_inputs(table, x, position_ids=torch.tensor([2, 0, 1]))
    assert torch.equal(one_d, table.index_select(1, torch.tensor([2, 0, 1])))
    positions = torch.tensor([[0, 0, 1], [0, 1, 2]])  # the left-padded rows `transformers` builds
    two_d, _ = model_module.prepare_attention_inputs(table, x, position_ids=positions)
    assert two_d.shape == (2, 3, 1, table.shape[3], 2)
    for row in range(2):
        assert torch.equal(two_d[row], table[0].index_select(0, positions[row]))


def test_prepare_attention_inputs_rejects_shapes_it_cannot_use() -> None:
    table = freqs_table()
    x = ids(2, 3)
    with pytest.raises(ValueError, match="position_ids must be 1-D"):
        model_module.prepare_attention_inputs(table, x, position_ids=torch.zeros(2, 3, 1, dtype=torch.long))
    with pytest.raises(ValueError, match=r"attention_mask must be \(B, S\)"):
        model_module.prepare_attention_inputs(table, x, attention_mask=torch.ones(2, 3, 3, dtype=torch.long))
    with pytest.raises(ValueError, match=r"attention_mask must be \(B, S\)"):
        model_module.prepare_attention_inputs(table, x, attention_mask=torch.ones(2, 4, dtype=torch.long))


def test_prepare_attention_inputs_builds_a_causal_padding_bool_mask() -> None:
    """
    The mask carries the causal triangle itself (sdpa's math kernel rejects a mask next to `is_causal=True`),
    and no query row is ever fully masked: an all-masked row would make softmax NaN and the NaN would spread.
    """

    table = freqs_table()
    x = ids(2, 4)
    ints = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
    _rotary, mask = model_module.prepare_attention_inputs(table, x, attention_mask=ints)
    assert mask is not None and mask.dtype == torch.bool and mask.shape == (2, 1, 4, 4)
    assert torch.equal(mask[1, 0], torch.ones(4, 4, dtype=torch.bool).tril()), "an all-ones mask is plain causality"
    expected_padded = torch.tensor(
        [[True, False, False, False],  # a pad query keeps only itself
         [False, True, False, False],
         [False, False, True, False],
         [False, False, True, True]],
    )
    assert torch.equal(mask[0, 0], expected_padded)
    assert bool(mask.any(dim=-1).all()), "no row attends to nothing"
    _rotary, from_bools = model_module.prepare_attention_inputs(table, x, attention_mask=ints.bool())
    assert from_bools is not None and torch.equal(from_bools, mask), "1/0 ints and bools mean the same"


def test_a_padding_mask_hides_the_pad_tokens_from_the_real_ones(
    tiny_model: RecurrentGPT, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    End to end through the model: with the mask, the real positions of a left-padded row give the same logits as
    the row on its own. The latent state is the only randomness left in eval mode; it is zeroed so both runs match.
    """

    monkeypatch.setattr(model_module, "initialize_state", torch.zeros_like)
    tiny_model.eval()
    real = torch.tensor([[5, 7, 11]])
    padded = torch.tensor([[0, 0, 5, 7, 11]])
    mask = torch.tensor([[0, 0, 1, 1, 1]])
    positions = torch.tensor([[0, 0, 0, 1, 2]])
    alone = tiny_model(real, return_logits=True)["logits"]
    with_pads = tiny_model(padded, attention_mask=mask, position_ids=positions, return_logits=True)["logits"]
    assert with_pads is not None and alone is not None
    torch.testing.assert_close(with_pads[:, 2:], alone, atol=1e-5, rtol=1e-5)
    unmasked = tiny_model(padded, return_logits=True)["logits"]
    assert unmasked is not None and not torch.allclose(unmasked[:, 2:], alone, atol=1e-3)


# --- gradient checkpointing -------------------------------------------------------------------------------------------


def test_gradient_checkpointing_matches_plain_path(monkeypatch: pytest.MonkeyPatch) -> None:
    plain = seeded_tiny()
    ckpt = seeded_tiny(gradient_checkpointing=True)
    calls: list[int] = []
    orig_checkpoint = recurrence._checkpoint

    def counting_checkpoint(*args: Any, **kwargs: Any) -> Tensor:
        calls.append(1)
        return cast(Tensor, orig_checkpoint(*args, **kwargs))

    monkeypatch.setattr(recurrence, "_checkpoint", counting_checkpoint)
    x = ids()
    torch.manual_seed(11)
    out_a = plain(x, labels=x, return_logits=True)
    torch.manual_seed(11)
    out_b = ckpt(x, labels=x, return_logits=True)
    # step 0 sampled k >= 1 per block: the checkpoint wrapper ran once per backprop iteration
    expected_calls = sum(int(ckpt.sample_block_depths(i)[1].item()) for i in range(2))
    assert expected_calls >= 2 and len(calls) == expected_calls
    assert torch.equal(out_a["logits"], out_b["logits"])
    out_a["loss"].backward()
    out_b["loss"].backward()
    for (na, pa), (nb, pb) in zip(plain.named_parameters(), ckpt.named_parameters()):
        assert na == nb
        torch.testing.assert_close(pa.grad, pb.grad, atol=1e-6, rtol=1e-5, msg=na)


# --- sampler (the scheme itself is tested in blocks/test_recurrence.py) ----------------------------------------------


@pytest.mark.parametrize("training", [True, False])
def test_randomized_iteration_sampler_binds_step_mode_and_config(training: bool) -> None:
    m = seeded_tiny(mean_recurrence=[12, 6], mean_backprop_depth=[8, 3]).train(training)
    m.step = 7
    for block_idx, (mean, depth) in enumerate(zip([12, 6], [8, 3])):
        n, k = m.sample_block_depths(block_idx)
        ref_n, ref_k = sample_recurrence_steps(mean, depth, step=7, training=training)
        assert (n.item(), k.item()) == (ref_n.item(), ref_k.item())
    m.step = 8
    n8, _ = m.sample_block_depths(0)
    assert torch.equal(n8, sample_recurrence_steps(12, 8, step=8, training=training)[0])


def test_train_forward_deterministic_under_seed_and_step(tiny_model: RecurrentGPT) -> None:
    x = ids()
    tiny_model.step = 7
    torch.manual_seed(1)
    a = tiny_model(x, labels=x, return_logits=True)
    torch.manual_seed(1)
    b = tiny_model(x, labels=x, return_logits=True)
    assert torch.equal(a["logits"], b["logits"]) and torch.equal(a["loss"], b["loss"])


# --- loss -----------------------------------------------------------------------------------------------------------


def test_loss_ignores_ignore_index_labels(tiny_model: RecurrentGPT) -> None:
    x = ids()
    labels = x.clone()
    labels[0, :10] = -100
    labels[1, 20:] = -100
    torch.manual_seed(2)
    out = tiny_model(x, labels=labels, return_logits=True, num_steps=(1, 1))
    logits = out["logits"]
    keep = labels != -100
    manual = torch.nn.functional.cross_entropy(logits[keep], labels[keep])
    torch.testing.assert_close(out["loss"], manual)
    assert not torch.allclose(out["loss"], torch.nn.functional.cross_entropy(logits.view(-1, VOCAB), x.view(-1)))


def test_out_of_range_labels_are_masked_too(tiny_model: RecurrentGPT) -> None:
    x = ids()
    labels = x.clone()
    labels[0, :10] = VOCAB + 5
    ref = x.clone()
    ref[0, :10] = -100
    torch.manual_seed(2)
    a = tiny_model(x, labels=labels, num_steps=(1, 1))["loss"]
    torch.manual_seed(2)
    b = tiny_model(x, labels=ref, num_steps=(1, 1))["loss"]
    assert torch.equal(a, b)


def test_custom_ignore_index() -> None:
    m = seeded_tiny(ignore_index=-1)
    x = ids()
    labels = x.clone()
    labels[:, :16] = -1
    torch.manual_seed(2)
    out = m(x, labels=labels, return_logits=True, num_steps=(1, 1))
    manual = torch.nn.functional.cross_entropy(out["logits"][:, 16:].reshape(-1, VOCAB), x[:, 16:].reshape(-1))
    torch.testing.assert_close(out["loss"], manual)


# --- golden forward ------------------------------------------------------------------------------------------------------


def golden_forward() -> dict[str, torch.Tensor]:
    """
    `tiny` built with seed 0, input ids from generator seed 1, global seed 123 before a train-mode forward at
    step 0 (sampled recurrence depths + random latent init).
    """

    model = seeded_tiny(0)
    x = ids(2, 32, seed=1)
    torch.manual_seed(123)
    out = model(x, labels=x, return_logits=True)
    return {"logits": out["logits"].detach().clone(), "loss": out["loss"].detach().clone()}


def record_golden() -> Path:
    """
    Re-record `golden_tiny_forward.pt`. ONLY do this in a commit whose purpose is a numerics change:
    `uv run python -c "from model.test_model import record_golden; record_golden()"`.
    """

    torch.save(golden_forward(), GOLDEN_PATH)
    return GOLDEN_PATH


def test_golden_tiny_forward() -> None:
    """
    Numerics regression guard: seeded `tiny` forward must reproduce the committed logits/loss (atol 1e-5).

    The golden file `model/golden_tiny_forward.pt` may only be re-recorded (via `record_golden()`) in a commit whose
    explicit purpose is a numerics change; any other failure here is a regression in the model code.
    """

    assert GOLDEN_PATH.exists(), "golden file missing; record it with record_golden() in a numerics commit"
    golden = torch.load(GOLDEN_PATH, weights_only=True)
    got = golden_forward()
    assert got["logits"].shape == golden["logits"].shape == (2, 32, VOCAB)
    assert torch.isfinite(golden["loss"])  # labels == inputs (tied embeddings), so the loss sits well below ln(512)
    torch.testing.assert_close(got["logits"], golden["logits"], atol=1e-5, rtol=0)
    torch.testing.assert_close(got["loss"], golden["loss"], atol=1e-5, rtol=0)


@pytest.mark.gpu
def test_compile_smoke() -> None:
    model = seeded_tiny().cuda()
    compiled = torch.compile(model)
    x = ids().cuda()
    torch.manual_seed(1)
    out = compiled(x, labels=x, return_logits=True)
    assert out["logits"].shape == (2, 32, VOCAB)
    out["loss"].backward()
