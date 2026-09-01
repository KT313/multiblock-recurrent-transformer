# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Tests for `model.hf.modeling`: recurrence-step parsing, config conversion and the trust_remote_code export round trip."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import model.model as model_module
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
    """`a.py`, `pkg/{__init__,b,c}.py`, `pkg/sub/d.py`, a test file and a `__pycache__` entry."""
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


def test_config_round_trip() -> None:
    cfg = tiny_config(rope_settings=RoPESettings(rope_base=12_345), mean_recurrence=[3, 5])
    hf_cfg = RecurrentGPTConfig.from_recurrent_config(cfg)
    assert hf_cfg.model_type == "recurrent_gpt"
    assert hf_cfg.rope_base == 12_345
    assert hf_cfg.hidden_size == cfg.n_embd
    assert hf_cfg.num_hidden_layers == cfg.effective_expected_depth
    assert hf_cfg.tie_word_embeddings is True
    back = hf_cfg.to_recurrent_config()
    expected = cfg.to_dict()
    expected["name"] = ""  # the architecture label is not an HF field
    assert back.to_dict() == expected
    assert back.head_size == cfg.head_size and back.n_layer == cfg.n_layer


def test_hf_config_defaults_are_the_dataclass_defaults() -> None:
    """Keys missing from a config.json fall back to `RecurrentConfig()` (the export has no access to config/)."""
    hf_cfg = RecurrentGPTConfig()
    defaults = RecurrentConfig()
    assert hf_cfg.n_layers_in_recurrent_block == defaults.n_layers_in_recurrent_block == [4]
    assert hf_cfg.num_hidden_layers == defaults.effective_expected_depth
    assert hf_cfg.n_embd == defaults.n_embd == 1024 and hf_cfg.vocab_size == 32000
    assert hf_cfg.rope_base == 50_000
    assert hf_cfg.to_recurrent_config() == defaults


def test_hf_config_survives_json_round_trip(tmp_path: Path) -> None:
    hf_cfg = RecurrentGPTConfig.from_recurrent_config(tiny_config())
    hf_cfg.save_pretrained(tmp_path)
    loaded = RecurrentGPTConfig.from_pretrained(tmp_path)
    assert loaded.to_recurrent_config() == hf_cfg.to_recurrent_config()


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
    ref = hf_model.model(x, return_logits=True, num_steps_pair=[(2, 0), (2, 0)])
    assert torch.equal(out.logits, ref["logits"])
    torch.manual_seed(1)
    tup = hf_model(x, return_dict=False)
    assert isinstance(tup, tuple) and len(tup) == 1 and torch.equal(tup[0], ref["logits"])


def test_wrapper_loss_is_the_next_token_loss_shifted_internally() -> None:
    """The HF contract: `model(x, labels=x).loss` predicts token t+1 from token t. The INNER model takes
    pre-shifted labels (the trainer's collate shifts), so its own loss on the same call is a different number."""
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
    hf_model.model.step = 3
    torch.manual_seed(1)
    out = hf_model(x, labels=x, return_dict=False)
    assert isinstance(out, tuple) and len(out) == 2
    torch.manual_seed(1)
    ref = hf_model.model(x, return_logits=True)  # num_steps_pair=None -> sampled at step 3
    assert torch.equal(out[1], ref["logits"])
    assert torch.equal(out[0], hf_model.model.loss(out[1][:, :-1].contiguous(), x[:, 1:].contiguous()))
    # ... which is not the eval path
    torch.manual_seed(1)
    eval_ref = hf_model.model(x, return_logits=True, num_steps_pair=[(2, 0), (2, 0)])["logits"]
    assert not torch.equal(out[1], eval_ref)


def test_env_recurrence_steps_override(monkeypatch: pytest.MonkeyPatch) -> None:
    hf_model = tiny_hf_model().train(False)
    x = ids()
    monkeypatch.setenv("EVAL_RECURRENCE_STEPS", "1,4")
    torch.manual_seed(1)
    out = hf_model(x).logits
    torch.manual_seed(1)
    ref = hf_model.model(x, return_logits=True, num_steps_pair=[(1, 0), (4, 0)])["logits"]
    assert torch.equal(out, ref)
    monkeypatch.setenv("EVAL_RECURRENCE_STEPS", "1,2,3")
    with pytest.raises(ValueError, match="recurrence values"):
        hf_model(x)


def test_env_recurrence_steps_is_ignored_in_training_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """The env var fixes the eval depths with zero backprop iterations: honouring it in training mode would train
    the recurrence without any gradient reaching it."""
    monkeypatch.setenv("EVAL_RECURRENCE_STEPS", "1,1")
    hf_model = tiny_hf_model().train(True)
    hf_model.model.step = 3
    x = ids()
    torch.manual_seed(1)
    out = hf_model(x).logits
    torch.manual_seed(1)
    sampled = hf_model.model(x, return_logits=True)["logits"]  # num_steps_pair=None -> sampled at step 3
    assert torch.equal(out, sampled)
    torch.manual_seed(1)
    fixed = hf_model.model(x, return_logits=True, num_steps_pair=[(1, 0), (1, 0)])["logits"]
    assert not torch.equal(out, fixed), "the env var is an eval knob"
    hf_model.train(False)  # and it is honoured again in eval mode
    torch.manual_seed(1)
    assert torch.equal(hf_model(x).logits, fixed)


def test_hf_config_broadcasts_the_per_block_int_shorthand() -> None:
    """`mean_recurrence: 12` for every block is the documented shorthand `RecurrentConfig` broadcasts; the wrapper's
    config used to raise a TypeError on it (it zipped over an int)."""
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
    """The embedding table is padded to `padding_multiple` and those columns are trained on no target, so the
    wrapper hides them: `generate(do_sample=True)` can only draw ids the tokenizer knows."""
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


@pytest.fixture
def zero_latent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero the random latent state: with it, an eval-mode forward at a fixed recurrence depth is deterministic and
    two differently shaped batches of the same tokens can be compared."""
    monkeypatch.setattr(model_module, "initialize_state", torch.zeros_like)


PADDED = torch.tensor([[0, 0, 5, 7, 11], [2, 3, 5, 7, 13]])
PADDING_MASK = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])


def test_left_padded_batch_gives_the_logits_of_the_single_prompts(zero_latent: None) -> None:
    """The wrapper's own path (`prepare_inputs_for_generation` -> `forward`): a left-padded batch must attend to no
    pad and count no pad as a position, so every row's real positions have the logits of that prompt run alone."""
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
    """The dynamically loaded class is a copy of `RecurrentGPTForCausalLM` with an identical interface."""
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
    model = build_model(TINY_ARCHITECTURE)
    out_dir = export_to_hf(model, model.config, tmp_path / "a" / "b", tokenizer_dir=tiny_tokenizer_dir)
    assert out_dir == tmp_path / "a" / "b"
    tok = AutoTokenizer.from_pretrained(out_dir)
    ref = AutoTokenizer.from_pretrained(tiny_tokenizer_dir)
    assert tok("hello world")["input_ids"] == ref("hello world")["input_ids"]


def test_export_and_reload_with_trust_remote_code(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = build_model(TINY_ARCHITECTURE)
    out_dir = export_to_hf(model, model.config, tmp_path / "export")
    names = {p.name for p in out_dir.iterdir()}
    assert {"config.json", "model.safetensors", "hf_modeling.py", "model.py", "config.py", "layers_norms.py"} <= names
    assert not any(n.startswith("test_") for n in names)
    assert "__init__.py" not in names

    cfg = AutoConfig.from_pretrained(out_dir, trust_remote_code=True)
    assert cfg.model_type == "recurrent_gpt"
    loaded = load_exported(out_dir)
    # In-process, transformers resolves the registered class from `model.hf.modeling` (see the standalone test for the copied
    # sources); the weights nevertheless come from the exported safetensors.
    assert isinstance(loaded, RecurrentGPTForCausalLM)
    model.eval()
    x = ids()
    torch.manual_seed(1)
    ref = model(x, return_logits=True, num_steps_pair=[(2, 0), (2, 0)])["logits"]
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
    model = build_model(TINY_ARCHITECTURE)
    out_dir = export_to_hf(model, model.config, tmp_path / "export")
    loaded = load_exported(out_dir)
    prompt = ids(1, 8)
    torch.manual_seed(1)
    # transformers' `GenerativePreTrainedModel` protocol lists attributes PreTrainedModel only sets dynamically.
    gen = loaded.generate(prompt, max_new_tokens=4, do_sample=False)  # type: ignore[misc]
    assert isinstance(gen, torch.Tensor)
    assert gen.shape == (1, 12)
    assert torch.equal(gen[:, :8], prompt)
    assert (gen[:, 8:] < 512).all()


def test_exported_folder_loads_standalone_without_the_repo(tmp_path: Path) -> None:
    """The copied sources must work without `model` importable: load in a subprocess whose cwd is the temp dir and
    whose only `sys.path` entries are the interpreter's own, then compare logits with the un-exported model."""
    torch.manual_seed(0)
    model = build_model(TINY_ARCHITECTURE)
    out_dir = export_to_hf(model, model.config, tmp_path / "export")
    model.eval()
    x = ids()
    torch.manual_seed(1)
    ref = model(x, return_logits=True, num_steps_pair=[(2, 0), (2, 0)])["logits"]
    torch.save({"x": x, "ref": ref}, tmp_path / "ref.pt")
    script = f"""
import importlib.util, sys
assert importlib.util.find_spec("model") is None, "repo package importable; test would not be standalone"
import torch
from transformers import AutoModelForCausalLM
loaded = AutoModelForCausalLM.from_pretrained({str(out_dir)!r}, trust_remote_code=True).train(False)
assert type(loaded).__module__.startswith("transformers_modules"), type(loaded).__module__
data = torch.load({str(tmp_path / "ref.pt")!r}, weights_only=True)
torch.manual_seed(1)
got = loaded(data["x"]).logits
torch.testing.assert_close(got, data["ref"], atol=1e-5, rtol=0)
print("STANDALONE_OK")
"""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)}
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_PROGRESS_BARS="1")
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert "STANDALONE_OK" in result.stdout
