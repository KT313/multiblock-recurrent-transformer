# Vendored from pytorch/ao (torchao)

The 8-bit optimizer-state tensor of torchao, copied so that `ELLISAdam8bit` (`training/optim/__init__.py`) can keep
its Adam moments block-wise quantised without depending on the torchao package (whose wheels lag behind new torch
and Python releases). BSD 3-Clause, see `LICENSE`; cite with `CITATION.cff`.

Source: https://github.com/pytorch/ao, commit `6792133` (main, 2026-09-08), directory `torchao/optim/`.

| file here | origin | changes |
|---|---|---|
| `subclass_8bit.py` | `torchao/optim/subclass_8bit.py` | imports point at this package; the DTensor / distributed-checkpoint ops (`view`, the c10d all-gather and wait ops, `is_pinned`, `slice`) are left out, nothing here shards the state |
| `quant_utils.py` | `torchao/optim/quant_utils.py` | only the four 8-bit functions (`create_dynamic_map`, `scale_tensor`, `quantize_8bit_with_qmap`, `dequant_with_qmap`); the 4-bit and stochastic-rounding helpers are left out |
| `base.py` | `torchao/utils.py` | `_implements` and `_dispatch__torch_dispatch__` as the methods of a small base class, standing in for `TorchAOBaseTensor` |
| `LICENSE`, `CITATION.cff` | repository root | verbatim |

The quantisation is the one of "8-bit Optimizers via Block-wise Quantization" (Dettmers et al., 2021): the tensor is
cut into blocks of 256 values, each block is scaled by its absmax and every value is rounded to the nearest entry of a
256-entry dynamic (exponent + fraction) code map, signed for the first moment and unsigned for the second. The
subclass implements only `copy_` (quantise into it), `lerp` (dequantise, then lerp) and `_to_copy` (device moves);
every other op raises, so the optimizer dequantises into fp32 temporaries, updates those and copies back.

Excluded from ruff, mypy and basedpyright (pyproject.toml): the files stay as close to upstream as possible so a later
sync is a diff, not a rewrite.
