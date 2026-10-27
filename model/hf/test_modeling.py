# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Tests for `model.hf.modeling`: recurrence-step parsing, config conversion and the trust_remote_code export round trip.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

import model.model as model_module
from model.layers.norms import RMSNorm
from model import build_model
from model.config import RecurrentConfig, RoPESettings
from model.test_config import TINY_ARCHITECTURE, tiny_config
from model.hf.modeling import (
    RecurrentGPTConfig,
    RecurrentGPTForCausalLM,
    export_sources,
    export_to_hf,
    flat_module_name,
    flatten_relative_imports,
    mask_padded_vocabulary,
    parse_recurrence_steps,
    resolve_special_token_ids,
)


def ids(batch: int = 2, seq: int = 16) -> torch.Tensor:
    return torch.randint(0, 512, (batch, seq), generator=torch.Generator().manual_seed(1))


@pytest.mark.parametrize(
    ("text", "num_blocks", "expected"),
    [
        ("12", 3, (12, 0)),
        (" 7 ", 1, (7, 0)),
        ("4,12,4", 3, [(4, 0), (12, 0), (4, 0)]),
        ("4, 4 ,4", 3, [(4, 0), (4, 0), (4, 0)]),
        ("", 3, None),
        ("   ", 3, None),
    ],
)
def test_parse_recurrence_steps(text: str, num_blocks: int, expected: object) -> None:
    assert parse_recurrence_steps(text, num_blocks) == expected


def test_parse_recurrence_steps_length_mismatch() -> None:
    with pytest.raises(ValueError, match="got 2 recurrence values but the model has 3"):
        parse_recurrence_steps("4,4", 3)


# --- flat source export ------------------------------------------------------------------------------------------------


def fake_package(root: Path) -> Path:
    """
    `a.py`, `pkg/{__init__,b,c}.py`, `pkg/sub/d.py`, a test file and a `__pycache__` entry.
    """

    pkg = root / "package"
    for rel, text in {
        "__init__.py": "from .a import A\n",
        "a.py": "from .pkg.c import C\n\nA = 1\n",
        "pkg/__init__.py": "from .b import B\n",
        "pkg/b.py": (
            "from typing import TYPE_CHECKING\n\nfrom ..a import A\nfrom .c import (\n    C,\n)\n"
            "from .sub.d import D\n\nif TYPE_CHECKING:\n    from ..a import A as A2\n\nB = 2\n"
        ),
        "pkg/c.py": "import os\n\nC = 3\n",
        "pkg/sub/d.py": "D = 4\n",
        "pkg/test_b.py": "from .b import B\n",
        "__pycache__/a.cpython-311.py": "",
    }.items():
        (pkg / rel).parent.mkdir(parents=True, exist_ok=True)
        (pkg / rel).write_text(text)
    return pkg


def test_flat_module_name() -> None:
    assert flat_module_name(Path("config.py")) == "config"
    assert flat_module_name(Path("layers/norms.py")) == "layers_norms"
    assert flat_module_name(Path("hf/modeling.py")) == "hf_modeling"


def test_flatten_relative_imports_rewrites_only_import_lines(tmp_path: Path) -> None:
    pkg = fake_package(tmp_path)
    flat = flatten_relative_imports((pkg / "pkg" / "b.py").read_text(), Path("pkg/b.py"), pkg)
    assert flat == (
        "from typing import TYPE_CHECKING\n\nfrom .a import A\nfrom .pkg_c import (\n    C,\n)\n"
        "from .pkg_sub_d import D\n\nif TYPE_CHECKING:\n    from .a import A as A2\n\nB = 2\n"
    )
    assert flatten_relative_imports((pkg / "a.py").read_text(), Path("a.py"), pkg) == "from .pkg_c import C\n\nA = 1\n"
    # same-directory imports of top-level modules are already flat
    assert flatten_relative_imports("from .a import A\n", Path("e.py"), pkg) == "from .a import A\n"
    # nothing but relative-import lines is touched
    text = "import os\nfrom os import path\nx = 'from .a import A'\n# from .a import A\n"
    assert flatten_relative_imports(text, Path("e.py"), pkg) == text


@pytest.mark.parametrize(
    ("module", "line", "message"),
    [
        ("e.py", "from .pkg import B", "does not name a module file"),
        ("e.py", "from . import a", "cannot be flattened"),
        ("pkg/b.py", "from ...a import A", "cannot be flattened"),
        ("e.py", "from .missing import X", "does not name a module file"),
    ],
)
def test_flatten_relative_imports_rejects_package_imports(tmp_path: Path, module: str, line: str, message: str) -> None:
    pkg = fake_package(tmp_path)
    with pytest.raises(ValueError, match=message):
        flatten_relative_imports(line + "\n", Path(module), pkg)


def test_export_sources_flattens_the_tree(tmp_path: Path) -> None:
    pkg = fake_package(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    written = export_sources(pkg, out)
    assert sorted(p.name for p in written) == ["a.py", "pkg_b.py", "pkg_c.py", "pkg_sub_d.py"]
    assert sorted(p.name for p in out.iterdir()) == ["a.py", "pkg_b.py", "pkg_c.py", "pkg_sub_d.py"]
    assert "from .pkg_sub_d import D" in (out / "pkg_b.py").read_text()
    assert (out / "pkg_c.py").read_text() == "import os\n\nC = 3\n"


def test_export_sources_rejects_flat_name_clash(tmp_path: Path) -> None:
    pkg = fake_package(tmp_path)
    (pkg / "pkg_c.py").write_text("X = 1\n")
    (tmp_path / "out").mkdir()
    with pytest.raises(ValueError, match="both flatten to pkg_c.py"):
        export_sources(pkg, tmp_path / "out")


@pytest.mark.parametrize('orthogonal', [True, False])
def test_config_round_trip(orthogonal: bool) -> None:
    cfg = tiny_config(rope_settings=RoPESettings(rope_base=12_345), mean_recurrence=[3, 5], init_orthogonal=orthogonal)
    hf_cfg = RecurrentGPTConfig.from_recurrent_config(cfg)
    assert hf_cfg.model_type == "recurrent_gpt"
    assert hf_cfg.rope_base == 12_345
    assert hf_cfg.hidden_size == cfg.n_embd
    assert hf_cfg.num_hidden_layers == cfg.effective_expected_depth
    assert hf_cfg.tie_word_embeddings is True
    back = hf_cfg.to_recurrent_config()
    assert back.to_dict() == cfg.to_dict()
    assert back.head_size == cfg.head_size and back.max_backprop_layers == cfg.max_backprop_layers


def test_hf_config_from_a_recurrent_config_dict_keeps_the_rope_base() -> None:
    """
    `RecurrentConfig.to_dict()` emits the nested `rope_settings`, not `rope_base`; it used to fall through to
    `PretrainedConfig` as an opaque attribute while `rope_base` stayed at its default.
    """

    cfg = tiny_config(rope_settings=RoPESettings(rope_base=777))
    hf_cfg = RecurrentGPTConfig(**cfg.to_dict())
    assert hf_cfg.rope_base == 777
    assert hf_cfg.to_recurrent_config().to_dict() == cfg.to_dict()


def test_hf_config_rejects_a_rope_base_disagreeing_with_rope_settings() -> None:
    with pytest.raises(ValueError, match="disagree"):
        RecurrentGPTConfig(rope_base=1, rope_settings={"rope_base": 2})


def test_hf_config_defaults_are_the_dataclass_defaults() -> None:
    """
    Keys missing from a config.json fall back to `RecurrentConfig()` (the export has no access to config/).
    """

    hf_cfg = RecurrentGPTConfig()
    defaults = RecurrentConfig()
    assert hf_cfg.n_layers_in_recurrent_block == defaults.n_layers_in_recurrent_block == [4]
    assert hf_cfg.num_hidden_layers == defaults.effective_expected_depth
    assert hf_cfg.n_embd == defaults.n_embd == 1024 and hf_cfg.vocab_size == 32000
    assert hf_cfg.rope_base == 50_000
    assert hf_cfg.to_recurrent_config() == defaults


@pytest.mark.parametrize('scaling', ['none', 'inverse_sqrt_depth'])
def test_hf_config_survives_json_round_trip(tmp_path: Path, scaling: str) -> None:
    hf_cfg = RecurrentGPTConfig.from_recurrent_config(tiny_config(residual_scaling=scaling))
    hf_cfg.save_pretrained(tmp_path)
    loaded = RecurrentGPTConfig.from_pretrained(tmp_path)
    assert loaded.to_recurrent_config() == hf_cfg.to_recurrent_config()
    assert loaded.to_recurrent_config().residual_scale == hf_cfg.to_recurrent_config().residual_scale


def tiny_hf_model() -> RecurrentGPTForCausalLM:
    torch.manual_seed(0)
    return RecurrentGPTForCausalLM(RecurrentGPTConfig.from_recurrent_config(tiny_config()))


def test_wrapper_forward_matches_inner_model_in_eval() -> None:
    hf_model = tiny_hf_model().train(False)
    assert hf_model.num_recurrent_blocks == 2
    assert hf_model.get_output_embeddings().weight is hf_model.get_input_embeddings().weight
    x = ids()
    torch.manual_seed(1)
    out = hf_model(x, labels=x)
    torch.manual_seed(1)
    ref = hf_model.model(x, return_logits=True, num_steps=[(2, 0), (2, 0)])
    assert torch.equal(out.logits, ref["logits"])
    torch.manual_seed(1)
    tup = hf_model(x, return_dict=False)
    assert isinstance(tup, tuple) and len(tup) == 1 and torch.equal(tup[0], ref["logits"])


def test_wrapper_loss_is_the_next_token_loss_shifted_internally() -> None:
    """
    The HF contract: `model(x, labels=x).loss` predicts token t+1 from token t. The INNER model takes
    pre-shifted labels (the trainer's collate shifts), so its own loss on the same call is a different number.
    """

    hf_model = tiny_hf_model().train(False)
    x = ids(2, 8)
    torch.manual_seed(1)
    out = hf_model(x, labels=x)
    logits = out.logits
    by_hand = torch.nn.functional.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1), ignore_index=-100
    )
    assert torch.allclose(out.loss, by_hand, atol=0, rtol=0)
    unshifted = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), x.reshape(-1))
    assert not torch.allclose(out.loss, unshifted), "the unshifted loss is what the inner model computes"
    # -100 positions are ignored, and a label column of only -100 leaves the loss to the remaining columns
    masked = x.clone()
    masked[:, 1:4] = -100
    torch.manual_seed(1)
    loss_masked = hf_model(x, labels=masked).loss
    kept = torch.nn.functional.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.shape[-1]), masked[:, 1:].reshape(-1), ignore_index=-100
    )
    assert torch.allclose(loss_masked, kept, atol=0, rtol=0)


def test_wrapper_in_train_mode_uses_the_sampler_and_returns_loss_tuple() -> None:
    hf_model = tiny_hf_model().train(True)
    x = ids()
    torch.manual_seed(1)
    out = hf_model(x, labels=x, return_dict=False)
    assert isinstance(out, tuple) and len(out) == 2
    torch.manual_seed(1)
    ref = hf_model.model(x, return_logits=True)  # num_steps=None -> sampled at the step the wrapper wrote
    assert torch.equal(out[1], ref["logits"])
    assert torch.equal(out[0], hf_model.model.loss(out[1][:, :-1].contiguous(), x[:, 1:].contiguous()))
    # ... which is not the eval path
    torch.manual_seed(1)
    eval_ref = hf_model.model(x, return_logits=True, num_steps=[(2, 0), (2, 0)])["logits"]
    assert not torch.equal(out[1], eval_ref)


def test_training_forwards_count_the_samplers_steps() -> None:
    """
    Nothing outside sets the inner model's `step` in a HF Trainer or PEFT run, so the wrapper counts its own
    sampled training forwards; otherwise every forward would draw the depths of step 0 forever.
    """

    hf_model = tiny_hf_model().train(True)
    x = ids()
    assert hf_model._training_forwards == 0
    torch.manual_seed(1)
    first = hf_model(x).logits
    assert hf_model.model.step == 0 and hf_model._training_forwards == 1
    torch.manual_seed(1)
    second = hf_model(x).logits  # the value written for the backward stays until the next forward
    assert hf_model.model.step == 1 and hf_model._training_forwards == 2
    assert not torch.equal(first, second), "the same seed and input, so only the drawn depths can differ"

    # the forward at counter value s is the native forward at `step = s`
    native = tiny_hf_model().train(True).model
    for step, wrapped in enumerate((first, second)):
        native.step = step
        torch.manual_seed(1)
        assert torch.equal(native(x, return_logits=True)["logits"], wrapped)

    # an explicit `num_steps` (and eval mode) leaves the counter alone: only sampled forwards are sampler steps
    hf_model(x, num_steps=[(2, 0), (2, 0)])
    hf_model.train(False)
    hf_model(x)
    assert hf_model._training_forwards == 2


def test_env_recurrence_steps_override(monkeypatch: pytest.MonkeyPatch) -> None:
    hf_model = tiny_hf_model().train(False)
    x = ids()
    monkeypatch.setenv("EVAL_RECURRENCE_STEPS", "1,4")
    torch.manual_seed(1)
    out = hf_model(x).logits
    torch.manual_seed(1)
    ref = hf_model.model(x, return_logits=True, num_steps=[(1, 0), (4, 0)])["logits"]
    assert torch.equal(out, ref)
    monkeypatch.setenv("EVAL_RECURRENCE_STEPS", "1,2,3")
    with pytest.raises(ValueError, match="recurrence values"):
        hf_model(x)


def test_env_recurrence_steps_is_ignored_in_training_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    The env var fixes the eval depths with zero backprop iterations: honouring it in training mode would train
    the recurrence without any gradient reaching it.
    """

    monkeypatch.setenv("EVAL_RECURRENCE_STEPS", "1,1")
    hf_model = tiny_hf_model().train(True)
    x = ids()
    torch.manual_seed(1)
    out = hf_model(x).logits
    torch.manual_seed(1)
    sampled = hf_model.model(x, return_logits=True)["logits"]  # num_steps=None -> sampled at the wrapper's step
    assert torch.equal(out, sampled)
    torch.manual_seed(1)
    fixed = hf_model.model(x, return_logits=True, num_steps=[(1, 0), (1, 0)])["logits"]
    assert not torch.equal(out, fixed), "the env var is an eval knob"
    hf_model.train(False)  # and it is honoured again in eval mode
    torch.manual_seed(1)
    assert torch.equal(hf_model(x).logits, fixed)


def test_hf_config_broadcasts_the_per_block_int_shorthand() -> None:
    """
    `mean_recurrence: 12` for every block is the documented shorthand `RecurrentConfig` broadcasts; the wrapper's
    config used to raise a TypeError on it (it zipped over an int).
    """

    hf_cfg = RecurrentGPTConfig(n_layers_in_recurrent_block=[2, 2, 2], mean_recurrence=3, mean_backprop_depth=2)
    assert hf_cfg.mean_recurrence == [3, 3, 3] and hf_cfg.mean_backprop_depth == [2, 2, 2]
    assert hf_cfg.num_hidden_layers == hf_cfg.to_recurrent_config().effective_expected_depth
    single = RecurrentGPTConfig(n_layers_in_recurrent_block=4, mean_recurrence=12, mean_backprop_depth=8)
    assert single.n_layers_in_recurrent_block == [4] and single.mean_recurrence == [12] == single.to_recurrent_config().mean_recurrence
    with pytest.raises(ValueError, match="mean_recurrence has 2 entries but there are 3"):
        RecurrentGPTConfig(n_layers_in_recurrent_block=[2, 2, 2], mean_recurrence=[3, 3])


def test_mask_padded_vocabulary_helper() -> None:
    logits = torch.zeros(2, 3, 8)
    assert mask_padded_vocabulary(logits, 8, 8) is logits, "no padding: the logits are handed on untouched"
    masked = mask_padded_vocabulary(logits, 5, 8)
    assert torch.isinf(masked[..., 5:]).all() and (masked[..., 5:] < 0).all()
    assert torch.equal(masked[..., :5], logits[..., :5]) and torch.equal(logits, torch.zeros(2, 3, 8)), "not in place"


def test_padded_vocabulary_columns_are_masked_and_never_generated() -> None:
    """
    The embedding table is padded to `padding_multiple` and those columns are trained on no target, so the
    wrapper hides them: `generate(do_sample=True)` can only draw ids the tokenizer knows.
    """

    cfg = tiny_config(vocab_size=500, padding_multiple=512)
    assert (cfg.vocab_size, cfg.padded_vocab_size) == (500, 512)
    torch.manual_seed(0)
    hf_model = RecurrentGPTForCausalLM(RecurrentGPTConfig.from_recurrent_config(cfg)).train(False)
    x = ids() % 500
    logits = hf_model(x).logits
    assert logits.shape[-1] == 512
    assert torch.isinf(logits[..., 500:]).all() and (logits[..., 500:] < 0).all()
    assert torch.isfinite(logits[..., :500]).all()
    torch.manual_seed(1)
    # transformers' `GenerativePreTrainedModel` protocol lists attributes PreTrainedModel only sets dynamically.
    generate = hf_model.generate  # pyright: ignore[reportAttributeAccessIssue]
    generated = generate(x[:, :4], max_new_tokens=6, do_sample=True)
    assert isinstance(generated, torch.Tensor) and (generated < 500).all()
    assert torch.isfinite(tiny_hf_model().train(False)(x).logits).all(), "an unpadded table is not masked"


def test_a_label_in_the_padding_columns_is_ignored_instead_of_infinite() -> None:
    """
    A label in `[vocab_size, padded_vocab_size)` used to be a valid target for the loss while its logit column is
    -inf here, which gave an infinite loss through the wrapper and a finite one natively. Both mask at
    `vocab_size` now, so the label is ignored on either side.
    """

    cfg = tiny_config(vocab_size=500, padding_multiple=512)
    torch.manual_seed(0)
    hf_model = RecurrentGPTForCausalLM(RecurrentGPTConfig.from_recurrent_config(cfg)).train(False)
    x = ids() % 500
    labels = x.clone()
    labels[0, 5] = 505  # a padding row: inside the table, outside the vocabulary
    ignored = x.clone()
    ignored[0, 5] = -100

    torch.manual_seed(1)  # the latent state is drawn per forward
    loss = hf_model(x, labels=labels).loss
    assert torch.isfinite(loss)
    torch.manual_seed(1)
    assert torch.equal(loss, hf_model(x, labels=ignored).loss)
    inner = hf_model.model
    torch.manual_seed(1)
    logits = inner(x, return_logits=True, num_steps=[(2, 0), (2, 0)])["logits"]
    assert logits is not None
    shifted = logits[:, :-1, :].contiguous()
    native = inner.loss(shifted, labels[:, 1:].contiguous())
    assert torch.equal(native, inner.loss(shifted, ignored[:, 1:].contiguous()))


@pytest.fixture
def zero_latent(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Zero the random latent state: with it, an eval-mode forward at a fixed recurrence depth is deterministic and
    two differently shaped batches of the same tokens can be compared.
    """

    monkeypatch.setattr(model_module, "initialize_state", torch.zeros_like)


PADDED = torch.tensor([[0, 0, 5, 7, 11], [2, 3, 5, 7, 13]])
PADDING_MASK = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])


def test_left_padded_batch_gives_the_logits_of_the_single_prompts(zero_latent: None) -> None:
    """
    The wrapper's own path (`prepare_inputs_for_generation` -> `forward`): a left-padded batch must attend to no
    pad and count no pad as a position, so every row's real positions have the logits of that prompt run alone.
    """

    hf_model = tiny_hf_model().train(False)
    prepared = hf_model.prepare_inputs_for_generation(PADDED, attention_mask=PADDING_MASK)
    batched = hf_model(**prepared).logits
    short = hf_model(torch.tensor([[5, 7, 11]])).logits
    long = hf_model(torch.tensor([[2, 3, 5, 7, 13]])).logits
    torch.testing.assert_close(batched[0, 2:], short[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(batched[1], long[0], atol=1e-5, rtol=1e-5)
    without_mask = hf_model(PADDED).logits  # what the wrapper used to do: the pads are attended to
    assert not torch.allclose(without_mask[0, 2:], short[0], atol=1e-3)


def test_masks_and_positions_of_both_shapes_reach_the_model(zero_latent: None) -> None:
    hf_model = tiny_hf_model().train(False)
    x = ids(2, 6)
    plain = hf_model(x).logits
    ones = torch.ones(2, 6, dtype=torch.long)
    from_ints = hf_model(x, attention_mask=ones).logits
    assert torch.equal(from_ints, hf_model(x, attention_mask=ones.bool()).logits), "1/0 ints and bools agree"
    torch.testing.assert_close(from_ints, plain, atol=1e-5, rtol=1e-5)  # an all-ones mask is the plain causal run
    assert torch.equal(hf_model(x, position_ids=torch.arange(6)).logits, plain)
    torch.testing.assert_close(
        hf_model(x, position_ids=torch.arange(6).expand(2, 6)).logits, plain, atol=1e-6, rtol=1e-6
    )


def test_generate_batches_left_padded_prompts_like_single_prompts(zero_latent: None) -> None:
    hf_model = tiny_hf_model().train(False)
    hf_model.generation_config.pad_token_id = 0
    # transformers' `GenerativePreTrainedModel` protocol lists attributes PreTrainedModel only sets dynamically.
    generate = hf_model.generate  # pyright: ignore[reportAttributeAccessIssue]
    batched = generate(PADDED, attention_mask=PADDING_MASK, max_new_tokens=3, do_sample=False)
    short = generate(torch.tensor([[5, 7, 11]]), max_new_tokens=3, do_sample=False)
    long = generate(torch.tensor([[2, 3, 5, 7, 13]]), max_new_tokens=3, do_sample=False)
    assert torch.equal(batched[0, 5:], short[0, 3:])
    assert torch.equal(batched[1, 5:], long[0, 5:])


def load_exported(out_dir: Path) -> RecurrentGPTForCausalLM:
    """
    The dynamically loaded class is a copy of `RecurrentGPTForCausalLM` with an identical interface.
    """

    loaded = cast(RecurrentGPTForCausalLM, AutoModelForCausalLM.from_pretrained(out_dir, trust_remote_code=True))
    loaded.train(False)
    return loaded


def test_init_weights_is_a_noop() -> None:
    hf_model = tiny_hf_model()
    before = hf_model.model.transformer.wte.weight.clone()
    hf_model._init_weights(hf_model.model.transformer.wte)
    hf_model._init_weights(hf_model.model.lm_head)
    assert torch.equal(hf_model.model.transformer.wte.weight, before)


def test_embedding_accessors() -> None:
    hf_model = tiny_hf_model()
    assert hf_model.get_input_embeddings() is hf_model.model.transformer.wte
    assert hf_model.get_output_embeddings() is hf_model.model.lm_head
    new_wte = torch.nn.Embedding(512, 64)
    hf_model.set_input_embeddings(new_wte)
    assert hf_model.model.transformer.wte is new_wte
    new_head = torch.nn.Linear(64, 512, bias=False)
    hf_model.set_output_embeddings(new_head)
    assert hf_model.model.lm_head is new_head
    assert hf_model.get_output_embeddings() is new_head


@pytest.mark.parametrize("tied", [True, False])
@pytest.mark.parametrize(("new_num_tokens", "pad_to_multiple_of"), [(500, None), (520, None), (512, None), (None, 1024)])
def test_vocabulary_resize_rejected_without_mutation(
    tied: bool, new_num_tokens: int | None, pad_to_multiple_of: int | None,
) -> None:
    hf_model = RecurrentGPTForCausalLM(RecurrentGPTConfig.from_recurrent_config(tiny_config(tie_embeddings=tied)))
    embedding = hf_model.get_input_embeddings()
    head = hf_model.get_output_embeddings()
    parameters = dict(hf_model.named_parameters())
    weights = {name: parameter.detach().clone() for name, parameter in parameters.items()}
    hf_config = hf_model.config.to_dict()
    native_config = hf_model.model.config.to_dict()
    rng = torch.get_rng_state().clone()

    with pytest.raises(NotImplementedError, match="Vocabulary resizing is not supported.*before constructing"):
        hf_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of, mean_resizing=False)

    assert hf_model.get_input_embeddings() is embedding
    assert hf_model.get_output_embeddings() is head
    assert hf_model.config.to_dict() == hf_config
    assert hf_model.model.config.to_dict() == native_config
    assert torch.equal(torch.get_rng_state(), rng)
    for name, parameter in hf_model.named_parameters():
        assert parameter is parameters[name]
        assert torch.equal(parameter, weights[name])


def test_vocabulary_resize_without_arguments_remains_an_embedding_lookup() -> None:
    hf_model = tiny_hf_model()
    assert hf_model.resize_token_embeddings() is hf_model.get_input_embeddings()


def test_prepare_inputs_for_generation_forwards_the_mask_and_the_row_positions() -> None:
    hf_model = tiny_hf_model()
    x = ids(1, 4)
    assert hf_model.prepare_inputs_for_generation(x, past_key_values=None) == {"input_ids": x}
    mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])  # a left-padded batch, as `generate` builds it
    prepared = hf_model.prepare_inputs_for_generation(ids(2, 4), attention_mask=mask, past_key_values=None)
    assert list(prepared) == ["input_ids", "attention_mask", "position_ids"]
    assert prepared["attention_mask"] is mask
    assert torch.equal(prepared["position_ids"], torch.tensor([[0, 0, 0, 1], [0, 1, 2, 3]]))
    given = torch.zeros(2, 4, dtype=torch.long)
    assert hf_model.prepare_inputs_for_generation(x, attention_mask=mask, position_ids=given)["position_ids"] is given


def test_export_with_tokenizer_and_nested_dir(tmp_path: Path, tiny_tokenizer_dir: Path) -> None:
    torch.manual_seed(0)
    model = build_model(TINY_ARCHITECTURE, use_custom_kernels=False)
    out_dir = export_to_hf(model, model.config, tmp_path / "a" / "b", tokenizer_dir=tiny_tokenizer_dir)
    assert out_dir == tmp_path / "a" / "b"
    tok = AutoTokenizer.from_pretrained(out_dir)
    ref = AutoTokenizer.from_pretrained(tiny_tokenizer_dir)
    assert tok("hello world")["input_ids"] == ref("hello world")["input_ids"]
    loaded = load_exported(out_dir)
    for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
        assert getattr(loaded.config, name) == getattr(loaded.generation_config, name) == getattr(tok, name)
    assert loaded.generation_config.pad_token_id == 0
    assert_stops_on_token(loaded, 2)


class ForceToken(LogitsProcessor):
    def __init__(self, token_id: int) -> None:
        self.token_id = token_id

    # transformers annotates scores as FloatTensor, but generation passes an ordinary Tensor.
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:  # pyright: ignore[reportIncompatibleMethodOverride]
        scores.fill_(float("-inf"))
        scores[:, self.token_id] = 0
        return scores


def assert_stops_on_token(model: RecurrentGPTForCausalLM, token_id: int) -> None:
    prompt = ids(1, 4)
    generated = model.generate(
        prompt, attention_mask=torch.ones_like(prompt), max_new_tokens=5, do_sample=False,
        logits_processor=LogitsProcessorList([ForceToken(token_id)]),
    )
    assert generated.shape == (1, 5)
    assert generated[0, -1].item() == token_id


@pytest.mark.parametrize("eos", [0, [0, 2]])
def test_explicit_export_with_optional_ids_absent(tmp_path: Path, eos: int | list[int]) -> None:
    model = build_model(TINY_ARCHITECTURE, use_custom_kernels=False)
    out_dir = export_to_hf(model, model.config, tmp_path / "export", eos_token_id=eos)
    loaded = load_exported(out_dir)
    for filename in ("config.json", "generation_config.json"):
        metadata = json.loads((out_dir / filename).read_text())
        assert metadata["eos_token_id"] == eos
        assert metadata.get("bos_token_id") is None and metadata.get("pad_token_id") is None
    for token_id in eos if isinstance(eos, list) else [eos]:
        assert_stops_on_token(loaded, token_id)


def test_tokenizer_instance_explicit_supplement_does_not_mutate_source(
    tmp_path: Path, tiny_tokenizer_dir: Path,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(tiny_tokenizer_dir)
    tokenizer.bos_token = None
    tokenizer.pad_token = None
    model = build_model(TINY_ARCHITECTURE, use_custom_kernels=False)
    out_dir = export_to_hf(model, model.config, tmp_path / "export", tokenizer=tokenizer, bos_token_id=1, eos_token_id=[2])
    saved = AutoTokenizer.from_pretrained(out_dir)
    assert saved.bos_token_id == 1 and saved.pad_token_id is None and saved.eos_token_id == 2
    assert tokenizer.bos_token_id is None and tokenizer.pad_token_id is None
    assert len(saved) == len(tokenizer)


@pytest.mark.parametrize("field", ["bos_token_id", "eos_token_id", "pad_token_id"])
@pytest.mark.parametrize("invalid", [True, False, 1.5, "2", -1, 511, [], [True], [2, -1]])
def test_invalid_metadata_fails_before_output_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, invalid: object,
) -> None:
    # 511 is a padded-only row: the tiny model still has 512 physical rows.
    model = build_model(TINY_ARCHITECTURE, use_custom_kernels=False, vocab_size=511)
    out_dir = tmp_path / "export"
    out_dir.mkdir()
    sentinel = out_dir / "config.json"
    sentinel.write_text("existing export")

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("invalid metadata reached weight access")

    monkeypatch.setattr(model, "state_dict", forbidden)
    kwargs = {"eos_token_id": 2, field: invalid}
    with pytest.raises(ValueError, match=field):
        export_to_hf(model, model.config, out_dir, **cast(dict[str, Any], kwargs))
    assert list(out_dir.iterdir()) == [sentinel] and sentinel.read_text() == "existing export"


def test_missing_and_conflicting_metadata_fail_before_directory_creation(
    tmp_path: Path, tiny_tokenizer_dir: Path,
) -> None:
    model = build_model(TINY_ARCHITECTURE, use_custom_kernels=False)
    out_dir = tmp_path / "export"
    with pytest.raises(ValueError, match="allow_missing_generation_metadata=True"):
        export_to_hf(model, model.config, out_dir)
    with pytest.raises(ValueError, match="eos_token_id conflicts"):
        export_to_hf(model, model.config, out_dir, tiny_tokenizer_dir, eos_token_id=3)
    tokenizer = AutoTokenizer.from_pretrained(tiny_tokenizer_dir)
    with pytest.raises(ValueError, match="either tokenizer_dir or tokenizer"):
        export_to_hf(model, model.config, out_dir, tiny_tokenizer_dir, tokenizer=tokenizer)
    assert not out_dir.exists()
    out_dir = export_to_hf(model, model.config, out_dir, allow_missing_generation_metadata=True)
    loaded = load_exported(out_dir)
    assert loaded.config.eos_token_id is None and loaded.generation_config.eos_token_id is None


def test_tokenizer_metadata_is_validated_after_loading(tiny_tokenizer_dir: Path) -> None:
    tokenizer = AutoTokenizer.from_pretrained(tiny_tokenizer_dir)
    with pytest.raises(ValueError, match="eos_token_id"):
        resolve_special_token_ids(2, tokenizer)
    tokenizer.bos_token = None
    with pytest.raises(ValueError, match="unknown to the tokenizer"):
        resolve_special_token_ids(1024, tokenizer, bos_token_id=1000)


@pytest.mark.parametrize("eos", [3, None])
def test_tokenizer_free_export_refuses_existing_tokenizer_without_mutation(
    tmp_path: Path, tiny_tokenizer_dir: Path, monkeypatch: pytest.MonkeyPatch, eos: int | None,
) -> None:
    model = build_model(TINY_ARCHITECTURE, use_custom_kernels=False)
    out_dir = export_to_hf(model, model.config, tmp_path / "export", tiny_tokenizer_dir)
    assert AutoTokenizer.from_pretrained(out_dir).eos_token_id == 2
    before = {path.relative_to(out_dir): path.read_bytes() for path in out_dir.rglob("*") if path.is_file()}

    with monkeypatch.context() as patch:
        def forbidden(*args: object, **kwargs: object) -> None:
            pytest.fail("ambiguous tokenizer reuse reached weight access")

        patch.setattr(model, "state_dict", forbidden)
        with pytest.raises(ValueError, match="fresh directory or supply a tokenizer"):
            export_to_hf(model, model.config, out_dir, eos_token_id=eos, allow_missing_generation_metadata=eos is None)

    after = {path.relative_to(out_dir): path.read_bytes() for path in out_dir.rglob("*") if path.is_file()}
    assert after == before
    # A clean explicit-only/model-only destination remains supported with the requested metadata.
    fresh = export_to_hf(
        model, model.config, tmp_path / "fresh", eos_token_id=eos, allow_missing_generation_metadata=eos is None,
    )
    loaded = load_exported(fresh)
    assert loaded.config.eos_token_id == loaded.generation_config.eos_token_id == eos
    # Supplying the tokenizer also makes reuse unambiguous.
    export_to_hf(model, model.config, out_dir, tiny_tokenizer_dir)
    assert AutoTokenizer.from_pretrained(out_dir).eos_token_id == load_exported(out_dir).generation_config.eos_token_id == 2


@pytest.mark.parametrize("artifact", ["tokenizer_config.json", "tokenizer.json", "vocab.txt", "spiece.model"])
def test_tokenizer_free_export_refuses_partial_tokenizer_artifacts(tmp_path: Path, artifact: str) -> None:
    model = build_model(TINY_ARCHITECTURE, use_custom_kernels=False)
    path = tmp_path / artifact
    path.write_bytes(b"existing tokenizer artifact")
    with pytest.raises(ValueError, match="fresh directory or supply a tokenizer"):
        export_to_hf(model, model.config, tmp_path, eos_token_id=3)
    assert list(tmp_path.iterdir()) == [path]
    assert path.read_bytes() == b"existing tokenizer artifact"


def test_saved_config_revalidates_special_token_ids(tmp_path: Path) -> None:
    config = RecurrentGPTConfig.from_recurrent_config(tiny_config(vocab_size=511), eos_token_id=2)
    config.save_pretrained(tmp_path)
    path = tmp_path / "config.json"
    saved = json.loads(path.read_text())
    saved["eos_token_id"] = 511
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="eos_token_id"):
        RecurrentGPTConfig.from_pretrained(tmp_path)


@pytest.mark.parametrize('scaling', ['none', 'inverse_sqrt_depth'])
def test_export_and_reload_with_trust_remote_code(tmp_path: Path, scaling: str) -> None:
    torch.manual_seed(0)
    model = build_model(TINY_ARCHITECTURE, use_custom_kernels=False, residual_scaling=scaling)
    out_dir = export_to_hf(model, model.config, tmp_path / "export", allow_missing_generation_metadata=True)
    names = {p.name for p in out_dir.iterdir()}
    assert {"config.json", "model.safetensors", "hf_modeling.py", "model.py", "config.py", "layers_norms.py"} <= names
    assert not any(n.startswith("test_") for n in names)
    assert "__init__.py" not in names

    cfg = AutoConfig.from_pretrained(out_dir, trust_remote_code=True)
    assert cfg.model_type == "recurrent_gpt"
    loaded = load_exported(out_dir)
    assert loaded.model.config.residual_scaling == scaling
    assert loaded.model.config.residual_scale == model.config.residual_scale
    # In-process, transformers resolves the registered class from `model.hf.modeling` (see the standalone test for the copied
    # sources); the weights nevertheless come from the exported safetensors.
    assert isinstance(loaded, RecurrentGPTForCausalLM)
    model.eval()
    x = ids()
    torch.manual_seed(1)
    ref = model(x, return_logits=True, num_steps=[(2, 0), (2, 0)])["logits"]
    torch.manual_seed(1)
    got = loaded(x).logits
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=0)
    assert loaded.model.lm_head.weight.data_ptr() == loaded.model.transformer.wte.weight.data_ptr()
    assert loaded.get_output_embeddings() is loaded.model.lm_head
    assert loaded.get_input_embeddings() is loaded.model.transformer.wte
    assert json.loads((out_dir / "config.json").read_text())["auto_map"]["AutoModelForCausalLM"] == (
        "hf_modeling.RecurrentGPTForCausalLM"
    )


def test_generate_runs(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = build_model(TINY_ARCHITECTURE, use_custom_kernels=False)
    out_dir = export_to_hf(model, model.config, tmp_path / "export", allow_missing_generation_metadata=True)
    loaded = load_exported(out_dir)
    prompt = ids(1, 8)
    torch.manual_seed(1)
    # transformers' `GenerativePreTrainedModel` protocol lists attributes PreTrainedModel only sets dynamically.
    gen = loaded.generate(prompt, max_new_tokens=4, do_sample=False)
    assert isinstance(gen, torch.Tensor)
    assert gen.shape == (1, 12)
    assert torch.equal(gen[:, :8], prompt)
    assert (gen[:, 8:] < 512).all()


def test_exported_folder_loads_standalone_without_the_repo(tmp_path: Path) -> None:
    """
    The copied sources must work without `model` importable: load in a subprocess whose cwd is the temp dir and
    whose only `sys.path` entries are the interpreter's own, then compare logits with the un-exported model.
    """

    torch.manual_seed(0)
    model = build_model(TINY_ARCHITECTURE, use_custom_kernels=False)
    out_dir = export_to_hf(model, model.config, tmp_path / "export", bos_token_id=1, eos_token_id=2, pad_token_id=0)
    model.eval()
    x = ids()
    torch.manual_seed(1)
    ref = model(x, return_logits=True, num_steps=[(2, 0), (2, 0)])["logits"]
    torch.save({"x": x, "ref": ref}, tmp_path / "ref.pt")
    script = f"""
import importlib.util, sys
assert importlib.util.find_spec("model") is None, "repo package importable; test would not be standalone"
import torch
from transformers import AutoModelForCausalLM
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
loaded = AutoModelForCausalLM.from_pretrained({str(out_dir)!r}, trust_remote_code=True).train(False)
assert type(loaded).__module__.startswith("transformers_modules"), type(loaded).__module__
assert loaded.execution_policy().precision is None
with loaded.execution_policy().autocast("cpu"):
    assert not torch.is_autocast_enabled("cpu")
data = torch.load({str(tmp_path / "ref.pt")!r}, weights_only=True)
torch.manual_seed(1)
got = loaded(data["x"]).logits
torch.testing.assert_close(got, data["ref"], atol=1e-5, rtol=0)
class ForceEOS(LogitsProcessor):
    def __call__(self, input_ids, scores):
        scores.fill_(float("-inf"))
        scores[:, 2] = 0
        return scores
assert loaded.config.eos_token_id == loaded.generation_config.eos_token_id == 2
prompt = data["x"][:1, :4]
output = loaded.generate(prompt, max_new_tokens=5, do_sample=False, logits_processor=LogitsProcessorList([ForceEOS()]))
assert output.shape == (1, 5) and output[0, -1].item() == 2
print("STANDALONE_OK")
"""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)}
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_PROGRESS_BARS="1")
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert "STANDALONE_OK" in result.stdout


def test_hf_config_carries_bf16_residual_stream(tmp_path: Path) -> None:
    hf_cfg = RecurrentGPTConfig.from_recurrent_config(tiny_config(bf16_residual_stream="core"))
    assert hf_cfg.bf16_residual_stream == "core"
    hf_cfg.save_pretrained(tmp_path)
    loaded = RecurrentGPTConfig.from_pretrained(tmp_path)
    assert loaded.to_recurrent_config().bf16_residual_stream == "core"
    model = RecurrentGPTForCausalLM(loaded)
    core_norm = cast(RMSNorm, model.model.get_submodule("transformer.core_blocks.0.0.norm_1"))
    prelude_norm = cast(RMSNorm, model.model.get_submodule("transformer.prelude.0.norm_1"))
    assert core_norm.autocast_output is True and prelude_norm.autocast_output is False


@pytest.mark.parametrize("precision", [None, "32", "bf16-mixed"])
def test_export_execution_metadata_is_optional_and_nonarchitectural(tmp_path: Path, precision: str | None) -> None:
    from model.execution import ExecutionPolicy

    model = build_model(TINY_ARCHITECTURE, use_custom_kernels=False)
    policy = None if precision is None else ExecutionPolicy(precision)
    out_dir = export_to_hf(model, model.config, tmp_path / "export", execution_policy=policy, allow_missing_generation_metadata=True)
    loaded = load_exported(out_dir)
    assert loaded.config.execution_precision == precision
    assert loaded.model.config == model.config
    assert loaded.execution_policy() == ExecutionPolicy(precision)
    assert (out_dir / "execution.py").is_file()
    assert "execution_precision" not in loaded.model.config.to_dict()
    with loaded.execution_policy().autocast("cpu"):
        assert torch.is_autocast_enabled("cpu") == (precision == "bf16-mixed")


def test_execution_metadata_preserves_positional_rope_configuration() -> None:
    config = RecurrentGPTConfig(None, {"rope_base": 1234}, execution_precision="bf16-mixed")
    assert config.rope_base == 1234 and config.execution_precision == "bf16-mixed"


@pytest.mark.parametrize("values,field", [
    ({"tie_embeddings": "false"}, "tie_embeddings"),
    ({"qk_bias": 1}, "qk_bias"),
    ({"norm_eps": float("nan")}, "norm_eps"),
    ({"rope_base": True}, "rope_base"),
    ({"rope_base": "50000"}, "rope_base"),
    ({"rope_base": float("inf")}, "rope_base"),
    ({"rope_base": True, "rope_settings": {"rope_base": 1}}, "rope_base"),
    ({"rope_settings": {"rope_base": float("nan")}}, "rope_base"),
])
def test_hf_config_rejects_invalid_native_values(values: dict[str, Any], field: str) -> None:
    with pytest.raises(ValueError, match=field):
        RecurrentGPTConfig(**values)


def test_hf_config_preserves_fractional_rope_base_and_rejects_close_conflicts() -> None:
    config = RecurrentGPTConfig(rope_base=12.5)
    assert config.to_recurrent_config().rope_settings.rope_base == 12.5
    with pytest.raises(ValueError, match="disagree"):
        RecurrentGPTConfig(rope_base=12.5, rope_settings={"rope_base": 12.6})
    config.rope_base = True
    with pytest.raises(ValueError, match="rope_base"):
        config.to_recurrent_config()


@pytest.mark.parametrize("text", ["0", "-1", "4,0", "1.5", "true", "nan", "1,2,3"])
def test_hf_text_depth_rejects_invalid_specifications(text: str) -> None:
    with pytest.raises(ValueError):
        parse_recurrence_steps(text, 2)


@pytest.mark.parametrize("steps", [True, 1.5, torch.tensor([1, 2, 3]), [(1, 0), (False, 0)]])
def test_hf_invalid_depth_precedes_recurrence(monkeypatch: pytest.MonkeyPatch, steps: Any) -> None:
    model = RecurrentGPTForCausalLM(RecurrentGPTConfig.from_recurrent_config(tiny_config()))
    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("invalid explicit depth reached recurrence")
    monkeypatch.setattr(model.model, "run_core_blocks", forbidden)
    with pytest.raises(ValueError, match="num_steps"):
        model(ids(1, 2), num_steps=steps)


def test_export_keeps_the_trainable_initial_state(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = build_model(TINY_ARCHITECTURE, use_custom_kernels=False, use_trainable_initial_state=True).eval()
    with torch.no_grad():  # a state unlike a fresh init, so a reinitialised load would show
        for state in model.transformer.initial_states:
            state.mul_(3)
    out_dir = export_to_hf(model, model.config, tmp_path / "export", eos_token_id=2)
    loaded = load_exported(out_dir).train(False)
    assert loaded.config.use_trainable_initial_state is True
    for exported, original in zip(loaded.model.transformer.initial_states, model.transformer.initial_states):
        assert torch.equal(exported, original)
    x = ids()
    with torch.no_grad():
        expected = model(x, return_logits=True)["logits"]
        actual = loaded(x).logits
    assert expected is not None
    torch.testing.assert_close(actual, expected)
