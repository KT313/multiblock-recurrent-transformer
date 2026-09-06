# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
The packed path on CUDA: the FlexAttention kernel under a `BlockMask`, that mask through the non-reentrant
checkpoint, and the compiled forward over it. `training/data/test_packing_parity.py` runs on the CPU, where
`document_attention_mask` returns a dense bool mask and sdpa does the math, so it covers none of these three.
"""

from pathlib import Path
from typing import Callable

import pytest
import torch
from torch._dynamo.utils import counters

import model.model as model_module
from model import build_model
from model.layers.attention import document_attention_mask
from model.model import RecurrentGPT
from model.test_config import TINY_ARCHITECTURE
from training.data.collate import Sample, pad_and_shift
from training.data.packing import pack_samples
from training.data.tokenizer import Tokenizer

pytestmark = pytest.mark.gpu

TRAINING_MAX_SEQUENCE_LENGTH = 256
PACK_LENGTH = 64
LENGTHS = (12, 7, 20, 5)  # tokens per document, bos and eos included
OTHER_LENGTHS = (9, 15, 6, 14)  # a different document layout of the same pack (same sum)
STEPS = (1, 2)  # fixed depths, so no sampling happens

Grads = dict[str, torch.Tensor]


@pytest.fixture(autouse=True)
def zero_latent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_module, "initialize_state", torch.zeros_like)  # the latent draw is per token


def documents(tokenizer: Tokenizer, lengths: tuple[int, ...], seed: int = 0) -> list[Sample]:
    generator = torch.Generator().manual_seed(seed)
    samples: list[Sample] = []
    for index, length in enumerate(lengths):
        body = torch.randint(3, tokenizer.vocab_size, (length - 2,), generator=generator)
        ids = torch.cat([torch.tensor([tokenizer.bos_id]), body, torch.tensor([tokenizer.eos_id])])
        samples.append((ids, ids.clone(), f"doc{index}"))
    return samples


def packed_inputs(tokenizer: Tokenizer, lengths: tuple[int, ...]) -> dict[str, torch.Tensor]:
    """
    The documents of `lengths` packed into one `PACK_LENGTH` row on the GPU, with the `BlockMask` built outside
    the forward as the training step does it.
    """

    pack = pack_samples(documents(tokenizer, lengths), PACK_LENGTH, tokenizer)
    document_ids = pack.document_ids.cuda()
    return {
        "input_ids": pack.input_ids.cuda(),
        "labels": pack.labels.cuda(),
        "position_ids": pack.position_ids.cuda(),
        "attention_mask": document_attention_mask(document_ids),
    }


def tiny_cuda_model(**kwargs: object) -> RecurrentGPT:
    torch.manual_seed(0)
    return build_model(TINY_ARCHITECTURE, **kwargs).cuda().train()


def loss_and_grads(
    model: Callable[..., dict[str, torch.Tensor]], parameters: RecurrentGPT, **inputs: object
) -> tuple[torch.Tensor, Grads]:
    """
    One forward and backward of `model` (the module or its compiled wrapper) at the fixed depths; the gradients are
    read from `parameters`, the module that owns them, and cleared afterwards.
    """

    parameters.zero_grad(set_to_none=True)
    out = model(**inputs, num_steps=STEPS)
    out["loss"].backward()  # type: ignore[no-untyped-call]  # Tensor.backward is unannotated in torch
    grads = {n: p.grad.detach().clone() for n, p in parameters.named_parameters() if p.grad is not None}
    return out["loss"].detach(), grads


def assert_same_grads(actual: Grads, expected: Grads, *, atol: float, rtol: float) -> None:
    assert set(actual) == set(expected)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], atol=atol, rtol=rtol, msg=name)


def test_flex_attention_backward_matches_dense_on_cuda(tiny_tokenizer_dir: Path) -> None:
    tokenizer = Tokenizer(tiny_tokenizer_dir)
    padded = pad_and_shift(documents(tokenizer, LENGTHS), tokenizer, TRAINING_MAX_SEQUENCE_LENGTH)
    model = tiny_cuda_model()
    padded_loss, padded_grads = loss_and_grads(
        model, model, input_ids=padded.input_ids.cuda(), labels=padded.labels.cuda()
    )
    packed_loss, packed_grads = loss_and_grads(model, model, **packed_inputs(tokenizer, LENGTHS))

    torch.testing.assert_close(packed_loss, padded_loss, atol=1e-6, rtol=1e-6)
    assert_same_grads(packed_grads, padded_grads, atol=1e-6, rtol=1e-5)


def test_block_mask_through_non_reentrant_checkpoint_on_cuda(tiny_tokenizer_dir: Path) -> None:
    inputs = packed_inputs(Tokenizer(tiny_tokenizer_dir), LENGTHS)
    model = tiny_cuda_model()
    plain_loss, plain_grads = loss_and_grads(model, model, **inputs)
    model.gradient_checkpointing = True
    ckpt_loss, ckpt_grads = loss_and_grads(model, model, **inputs)

    assert torch.equal(ckpt_loss, plain_loss)
    assert_same_grads(ckpt_grads, plain_grads, atol=0.0, rtol=0.0)


def test_compiled_forward_with_packing_matches_eager_on_cuda(tiny_tokenizer_dir: Path) -> None:
    torch._dynamo.reset()
    tokenizer = Tokenizer(tiny_tokenizer_dir)
    inputs = packed_inputs(tokenizer, LENGTHS)
    other_inputs = packed_inputs(tokenizer, OTHER_LENGTHS)
    model = tiny_cuda_model()
    eager_loss, eager_grads = loss_and_grads(model, model, **inputs)
    compiled = torch.compile(model, dynamic=True)
    compiled_loss, compiled_grads = loss_and_grads(compiled, model, **inputs)

    torch.testing.assert_close(compiled_loss, eager_loss, atol=1e-4, rtol=1e-4)
    assert_same_grads(compiled_grads, eager_grads, atol=1e-4, rtol=1e-4)

    # A new document layout changes the mask's contents, not its shape: the compiled graphs must be reused. Counted
    # as graphs, not with the `fail_on_recompile` stance: that stance raises on every tensor-carrying frame without a
    # cache entry, which the non-recursively disabled `run_core_blocks` is on every call.
    graphs = counters["stats"]["unique_graphs"]
    other_loss, _ = loss_and_grads(compiled, model, **other_inputs)
    assert torch.isfinite(other_loss)
    assert counters["stats"]["unique_graphs"] == graphs
