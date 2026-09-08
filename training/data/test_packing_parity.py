# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Parity of sequence packing against padding on the tiny model: four documents, once as a padded 4-row batch, once
packed into a single row (document mask, per-document positions), must give the same loss and the same gradient
for every parameter. The only test of the packed path's gradients (`test_packing.py` covers the layout).
"""

import pytest
import torch

import model.model as model_module
from model import build_model
from model.layers.attention import document_attention_mask
from model.test_config import TINY_ARCHITECTURE
from training.data.collate import Sample, pad_and_shift
from training.data.packing import pack_samples
from training.data.tokenizer import Tokenizer

TRAINING_MAX_SEQUENCE_LENGTH = 256
LENGTHS = (12, 7, 20, 5)  # tokens per document, bos and eos included


def _documents(tokenizer: Tokenizer, seed: int = 0) -> list[Sample]:
    generator = torch.Generator().manual_seed(seed)
    samples: list[Sample] = []
    for index, length in enumerate(LENGTHS):
        body = torch.randint(3, tokenizer.vocab_size, (length - 2,), generator=generator)
        ids = torch.cat([torch.tensor([tokenizer.bos_id]), body, torch.tensor([tokenizer.eos_id])])
        samples.append((ids, ids.clone(), f"doc{index}"))
    return samples


@pytest.mark.parametrize("pack_length", [sum(LENGTHS) - len(LENGTHS), 64, TRAINING_MAX_SEQUENCE_LENGTH])
def test_padded_and_packed_batches_give_the_same_gradients(
    tokenizer: Tokenizer, monkeypatch: pytest.MonkeyPatch, pack_length: int
) -> None:
    monkeypatch.setattr(model_module, "initialize_state", torch.zeros_like)  # the latent draw is per token
    samples = _documents(tokenizer)
    steps = (1, 2)

    def loss_and_grads(**inputs: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        torch.manual_seed(0)
        model = build_model(TINY_ARCHITECTURE).train()
        out = model(**inputs, num_steps=steps)
        out["loss"].backward()
        return out["loss"].detach(), {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}

    padded = pad_and_shift(samples, tokenizer, TRAINING_MAX_SEQUENCE_LENGTH)
    padded_loss, padded_grads = loss_and_grads(input_ids=padded.input_ids, labels=padded.labels)

    pack = pack_samples(samples, pack_length, tokenizer)
    assert pack.input_ids.shape == (1, pack_length) and len(pack.data_ids) == 4
    packed_loss, packed_grads = loss_and_grads(
        input_ids=pack.input_ids,
        labels=pack.labels,
        position_ids=pack.position_ids,
        attention_mask=document_attention_mask(pack.document_ids),
    )

    # the same supervised positions in both layouts, so the same mean loss
    assert int((padded.labels != -100).sum()) == int((pack.labels != -100).sum()) == sum(LENGTHS) - len(LENGTHS)
    torch.testing.assert_close(packed_loss, padded_loss, atol=1e-6, rtol=1e-6)
    assert set(packed_grads) == set(padded_grads)
    for name in padded_grads:
        torch.testing.assert_close(packed_grads[name], padded_grads[name], atol=1e-6, rtol=1e-5, msg=name)
