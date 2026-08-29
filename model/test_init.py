# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for `model.init`: the takase std table, orthogonality of `trunc_orthogonal_`, zero biases."""

from math import sqrt

import pytest
import torch

from model.init import Init, Linear, init_glu, init_qkv, trunc_orthogonal_, wrapped_trunc_ortho

DIM, HEAD, LAYERS = 1024, 64, 50
STD = sqrt(2 / (5 * DIM))


@pytest.fixture
def init() -> Init:
    torch.manual_seed(0)
    return Init(DIM, HEAD, LAYERS)


def test_table_values(init: Init) -> None:
    assert init.table["std"] == pytest.approx(STD)
    assert init.table["out_proj"] == pytest.approx(STD / sqrt(2 * LAYERS))
    assert init.table["embedding"] == pytest.approx(STD)
    assert init.embedding_scale == pytest.approx(sqrt(DIM))
    assert init.logit_scale == 1.0
    assert init._std("out_attn") == init._std("out_proj") == init.table["out_proj"]
    assert init._std("head") == init._std("in_proj") == init._std("glu") == STD
    assert "takase" in repr(init)


def test_depth_only_scales_the_output_projections() -> None:
    shallow, deep = Init(DIM, HEAD, 2), Init(DIM, HEAD, 200)
    assert shallow._std("out_proj") == pytest.approx(deep._std("out_proj") * 10)
    for name in ("std", "embedding", "head", "qkv", "glu", "in_proj"):
        assert shallow._std(name) == deep._std(name) == STD


@pytest.mark.parametrize(("rows", "cols"), [(256, 256), (512, 256), (256, 512)])
def test_trunc_orthogonal_is_orthogonal(rows: int, cols: int) -> None:
    torch.manual_seed(0)
    w = trunc_orthogonal_(torch.empty(rows, cols))
    if rows >= cols:
        torch.testing.assert_close(w.T @ w, torch.eye(cols), atol=1e-4, rtol=0)
    else:
        torch.testing.assert_close(w @ w.T, torch.eye(rows), atol=1e-4, rtol=0)


def test_trunc_orthogonal_gain_and_wrapped_std() -> None:
    torch.manual_seed(0)
    w = trunc_orthogonal_(torch.empty(256, 256), gain=3.0)
    torch.testing.assert_close(w.T @ w, 9.0 * torch.eye(256), atol=1e-3, rtol=0)
    w = torch.empty(1024, 512)
    wrapped_trunc_ortho(w, std=0.05)  # in place: the gain is std * sqrt(max(rows, cols))
    assert w.std().item() == pytest.approx(0.05, rel=0.05)


@pytest.mark.parametrize(
    ("name", "shape", "expected_std"),
    [
        ("qkv", (3 * DIM, DIM), STD),
        ("glu", (2 * 4 * DIM, DIM), STD),
        ("head", (2048, DIM), STD),
        ("embedding", (2048, DIM), STD),
        ("in_proj", (DIM, 2 * DIM), STD),
        ("out_attn", (DIM, DIM), STD / sqrt(2 * LAYERS)),
        ("out_proj", (DIM, 4 * DIM), STD / sqrt(2 * LAYERS)),
    ],
)
def test_fn_std_on_large_tensor(init: Init, name: str, shape: tuple[int, int], expected_std: float) -> None:
    w = torch.empty(*shape)
    init.fn(name)(w)
    assert w.std().item() == pytest.approx(expected_std, rel=0.05)
    assert abs(w.mean().item()) < expected_std * 0.05


def test_qkv_init_is_per_component_orthogonal(init: Init) -> None:
    w = torch.empty(3 * DIM, DIM)
    init.fn("qkv")(w)
    q, k, v = torch.split(w, DIM, dim=0)
    for part in (q, k, v):
        torch.testing.assert_close(part @ part.T / (STD**2 * DIM), torch.eye(DIM), atol=1e-3, rtol=0)
    assert not torch.allclose(q, k)


def test_glu_init_is_per_half_orthogonal() -> None:
    torch.manual_seed(0)
    w = torch.empty(2 * 256, 256)
    init_glu(w, 0.1, 0.1)
    g, h = w.chunk(2, dim=0)
    for part in (g, h):
        torch.testing.assert_close(part.T @ part / (0.01 * 256), torch.eye(256), atol=1e-3, rtol=0)
    assert not torch.allclose(g, h)


def test_init_qkv_explicit_stds() -> None:
    torch.manual_seed(0)
    dim, hd = 256, 32
    w = torch.empty(3 * dim, dim)
    init_qkv(w, qk_std=0.01, v_std=0.1, dim=dim, head_dim=hd)
    q, k, v = torch.split(w, dim, dim=0)
    assert q.std().item() == pytest.approx(0.01, rel=0.1)
    assert k.std().item() == pytest.approx(0.01, rel=0.1)
    assert v.std().item() == pytest.approx(0.1, rel=0.1)


def test_normalization_fn_is_ones(init: Init) -> None:
    w = torch.zeros(8)
    init.fn("normalization")(w)
    assert torch.equal(w, torch.ones(8))


def test_apply_zeroes_bias_and_sets_weight(init: Init) -> None:
    ln = torch.nn.LayerNorm(16)
    with torch.no_grad():
        ln.weight.fill_(2.0)
        ln.bias.fill_(2.0)
    init.apply(ln, "normalization")
    assert torch.equal(ln.weight, torch.ones(16))
    assert torch.equal(ln.bias, torch.zeros(16))

    emb = torch.nn.Embedding(2048, DIM)
    init.apply(emb, "embedding")
    assert emb.weight.std().item() == pytest.approx(STD, rel=0.05)


def test_linear_uses_init_method_and_zero_bias(init: Init) -> None:
    torch.manual_seed(0)
    lin = Linear(DIM, DIM, bias=True, init_method=init.fn("out_attn"))
    assert lin.weight.std().item() == pytest.approx(STD / sqrt(2 * LAYERS), rel=0.05)
    assert torch.equal(lin.bias, torch.zeros(DIM))
    with torch.no_grad():
        lin.bias.fill_(1.0)
    lin.reset_parameters()
    assert torch.equal(lin.bias, torch.zeros(DIM))


def test_init_is_seed_deterministic(init: Init) -> None:
    torch.manual_seed(7)
    a = torch.empty(256, 256)
    init.fn("head")(a)
    torch.manual_seed(7)
    b = torch.empty(256, 256)
    init.fn("head")(b)
    assert torch.equal(a, b)
