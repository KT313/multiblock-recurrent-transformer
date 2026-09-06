# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
Tests for `model.config`: per-block broadcasting, derived sizes, the shipped architecture YAMLs and the JSON
round trip. `tiny_config` / `TINY_ARCHITECTURE` are the shared helpers of the other `model/test_*.py` files.
"""

from pathlib import Path
from typing import Any

import pytest

from model.config import RecurrentConfig, RoPESettings, broadcast_per_block, find_multiple

REPO_ROOT = Path(__file__).resolve().parent.parent
ARCHITECTURE_DIR = REPO_ROOT / "config" / "model_architecture"
TINY_ARCHITECTURE = ARCHITECTURE_DIR / "tiny.yaml"
CROW_ARCHITECTURE = ARCHITECTURE_DIR / "crow_300m_final.yaml"


def tiny_config(**overrides: Any) -> RecurrentConfig:
    """
    `config/model_architecture/tiny.yaml` with overrides applied (the test-suite model).
    """

    return RecurrentConfig.from_yaml(TINY_ARCHITECTURE, **overrides)


tiny = tiny_config


@pytest.mark.parametrize(("n", "k", "expected"), [(512, 512, 512), (500, 128, 512), (1, 8, 8), (32000, 2048, 32768)])
def test_find_multiple(n: int, k: int, expected: object) -> None:
    assert find_multiple(n, k) == expected


def test_int_fields_broadcast_to_single_block_lists() -> None:
    cfg = tiny(n_layers_in_recurrent_block=4, mean_recurrence=12, mean_backprop_depth=8)
    assert cfg.n_layers_in_recurrent_block == [4]
    assert cfg.mean_recurrence == [12]
    assert cfg.mean_backprop_depth == [8]


def test_int_fields_broadcast_across_blocks() -> None:
    cfg = tiny(n_layers_in_recurrent_block=[4, 4, 4], mean_recurrence=12, mean_backprop_depth=8)
    assert cfg.mean_recurrence == [12, 12, 12]
    assert cfg.mean_backprop_depth == [8, 8, 8]


def test_single_element_list_is_broadcast_like_an_int() -> None:
    cfg = tiny(n_layers_in_recurrent_block=[1, 2], mean_recurrence=[5], mean_backprop_depth=[3])
    assert cfg.mean_recurrence == [5, 5]
    assert cfg.mean_backprop_depth == [3, 3]


def test_per_block_lists_are_kept() -> None:
    cfg = tiny(n_layers_in_recurrent_block=[1, 2, 3], mean_recurrence=[4, 5, 6], mean_backprop_depth=[1, 2, 3])
    assert cfg.mean_recurrence == [4, 5, 6]
    assert cfg.mean_backprop_depth == [1, 2, 3]


@pytest.mark.parametrize("field", ["mean_recurrence", "mean_backprop_depth"])
def test_length_mismatch_raises(field: str) -> None:
    per_block: dict[str, Any] = {"mean_recurrence": 1, "mean_backprop_depth": 1, field: [1, 2]}
    with pytest.raises(ValueError, match=f"{field} has 2 entries but there are 3"):
        tiny(n_layers_in_recurrent_block=[1, 1, 1], **per_block)


def test_broadcast_helper_directly() -> None:
    assert broadcast_per_block("f", 3, 1) == [3]
    assert broadcast_per_block("f", 3, 3) == [3, 3, 3]
    assert broadcast_per_block("f", [3], 2) == [3, 3]
    assert broadcast_per_block("f", [1, 2], 2) == [1, 2]
    with pytest.raises(ValueError, match="f has 3 entries but there are 2"):
        broadcast_per_block("f", [1, 2, 3], 2)


def test_rope_settings_default() -> None:
    assert RoPESettings().rope_base == 50_000
    assert RoPESettings(rope_base=7) != RoPESettings()


def test_padded_vocab_size_derived_from_padding_multiple() -> None:
    cfg = tiny(vocab_size=500, padding_multiple=128)
    assert cfg.padded_vocab_size == 512
    assert cfg.vocab_size == 500


def test_explicit_padded_vocab_size_must_hold_the_vocabulary() -> None:
    cfg = tiny(vocab_size=700, padded_vocab_size=768)
    assert (cfg.padded_vocab_size, cfg.vocab_size) == (768, 700)
    with pytest.raises(ValueError, match="padded_vocab_size 768 is smaller than vocab_size 1000"):
        tiny(vocab_size=1000, padded_vocab_size=768)  # used to truncate the vocabulary silently


def test_head_size_and_intermediate_size() -> None:
    cfg = tiny(n_embd=64, num_attention_heads=4, intermediate_size=None)
    assert cfg.head_size == 16
    assert cfg.intermediate_size == 256


def test_n_embd_not_divisible_by_heads_raises() -> None:
    with pytest.raises(ValueError, match="divisible"):
        tiny(n_embd=65, num_attention_heads=4)


def test_depth_arithmetic() -> None:
    cfg = tiny(
        n_layers_in_prelude=2,
        n_layers_in_coda=1,
        n_layers_in_recurrent_block=[1, 2],
        mean_recurrence=[3, 4],
        mean_backprop_depth=[2, 1],
    )
    assert cfg.effective_expected_depth == 2 + 1 + (1 * 3 + 2 * 4)
    assert cfg.mean_backprop_layers == 1 * 2 + 2 * 1
    assert cfg.init.num_layers == cfg.effective_expected_depth


# --- the shipped architecture YAMLs ----------------------------------------------------------------------------------


def test_crow_architecture_yaml() -> None:
    cfg = RecurrentConfig.from_yaml(CROW_ARCHITECTURE)
    assert cfg.n_layers_in_recurrent_block == [4, 4, 4]
    assert cfg.mean_recurrence == [12, 12, 12] and cfg.mean_backprop_depth == [8, 8, 8]
    assert cfg.effective_expected_depth == 2 + 2 + 3 * 4 * 12
    assert cfg.mean_backprop_layers == 3 * 4 * 8
    assert cfg.padded_vocab_size == 32768
    assert cfg.head_size == 64
    assert cfg.intermediate_size == 4096 and cfg.model_max_sequence_length == 2048 and cfg.vocab_size == 32000
    assert isinstance(cfg.norm_eps, float) and cfg.norm_eps == 1e-6  # YAML floats need a dot: 1e-6 would be a str
    assert cfg.rope_settings == RoPESettings(rope_base=50_000)
    assert cfg.qk_bias is True and cfg.tie_embeddings is True


def test_tiny_architecture_yaml() -> None:
    cfg = tiny()
    assert (cfg.model_max_sequence_length, cfg.n_embd, cfg.intermediate_size, cfg.num_attention_heads) == (256, 64, 128, 4)
    assert (cfg.vocab_size, cfg.padded_vocab_size, cfg.head_size) == (512, 512, 16)
    assert cfg.n_layers_in_prelude == 2 and cfg.n_layers_in_coda == 1
    assert cfg.n_layers_in_recurrent_block == [1, 1]
    assert cfg.mean_recurrence == [2, 2] and cfg.mean_backprop_depth == [2, 2]
    assert isinstance(cfg.norm_eps, float) and cfg.norm_eps == 1e-6


def test_architecture_yamls_list_every_tunable_field() -> None:
    """
    Both files list the same keys: every dataclass field except the fixed single-value ones.
    """

    import yaml

    fixed = {
        "attn_impl",
        "init_strategy",
        "init_orthogonal",
        "activation_checkpoint_impl",
        "injection_type",
        "state_init",
        "sampling_scheme",
    }
    expected = {f.name for f in RecurrentConfig.__dataclass_fields__.values()} - fixed
    for path in (TINY_ARCHITECTURE, CROW_ARCHITECTURE):
        with open(path, encoding="utf-8") as fp:
            keys = set(yaml.safe_load(fp))
        assert keys == expected, path
    assert {"rope_settings", "n_layers_in_recurrent_block", "mean_recurrence", "mean_backprop_depth"} <= expected


def test_from_yaml_applies_overrides() -> None:
    cfg = tiny(n_embd=32, num_attention_heads=2, mean_recurrence=[7, 9])
    assert cfg.n_embd == 32
    assert cfg.head_size == 16
    assert cfg.mean_recurrence == [7, 9]
    assert tiny().n_embd == 64


def test_from_yaml_does_not_share_state_between_loads() -> None:
    a = tiny(rope_settings={"rope_base": 1})
    b = tiny(n_layers_in_recurrent_block=3, mean_recurrence=99, mean_backprop_depth=9)
    c = tiny()
    assert a.rope_settings.rope_base == 1 and b.rope_settings.rope_base == 50_000
    assert b.n_layers_in_recurrent_block == [3] and b.mean_recurrence == [99] and b.mean_backprop_depth == [9]
    assert c == tiny() and c.n_layers_in_recurrent_block == [1, 1] and c.mean_recurrence == [2, 2]
    assert c.rope_settings is not tiny().rope_settings
    assert c.mean_recurrence is not tiny().mean_recurrence


def test_from_yaml_unknown_key_raises(tmp_path: Path) -> None:
    path = tmp_path / "arch.yaml"
    path.write_text(TINY_ARCHITECTURE.read_text() + "\nn_heads: 4\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"arch\.yaml: unknown RecurrentConfig key\(s\) \['n_heads'\]"):
        RecurrentConfig.from_yaml(path)


def test_from_yaml_requires_a_mapping(tmp_path: Path) -> None:
    path = tmp_path / "arch.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected a mapping"):
        RecurrentConfig.from_yaml(path)


def test_from_yaml_accepts_str_path_and_nested_rope_settings(tmp_path: Path) -> None:
    path = tmp_path / "arch.yaml"
    path.write_text("n_embd: 32\nnum_attention_heads: 2\nrope_settings:\n  rope_base: 123\n", encoding="utf-8")
    cfg = RecurrentConfig.from_yaml(str(path))
    assert cfg.n_embd == 32 and cfg.rope_settings == RoPESettings(rope_base=123)
    assert cfg.n_layers_in_recurrent_block == [4]  # dataclass defaults fill the rest


def test_from_yaml_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        RecurrentConfig.from_yaml(ARCHITECTURE_DIR / "does-not-exist.yaml")


def test_json_round_trip(tmp_path: Path) -> None:
    cfg = tiny(rope_settings=RoPESettings(rope_base=10_000), mean_recurrence=[3, 5])
    path = tmp_path / "config.json"
    cfg.to_json(path)
    loaded = RecurrentConfig.from_json(path)
    assert loaded == cfg
    assert isinstance(loaded.rope_settings, RoPESettings)
    assert loaded.rope_settings.rope_base == 10_000
    assert loaded.head_size == cfg.head_size
    assert loaded.effective_expected_depth == cfg.effective_expected_depth


def test_from_json_overrides(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    tiny().to_json(path)
    assert RecurrentConfig.from_json(path, n_layers_in_coda=5).n_layers_in_coda == 5


def test_rope_settings_accepts_dict() -> None:
    assert tiny(rope_settings={"rope_base": 123}).rope_settings == RoPESettings(rope_base=123)


def test_to_dict_contains_only_dataclass_fields() -> None:
    d = tiny().to_dict()
    assert "init" not in d and "head_size" not in d and "mean_backprop_layers" not in d
    assert d["rope_settings"] == {"rope_base": 50_000}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("attn_impl", "flash"),
        ("init_strategy", "normal"),
        ("init_orthogonal", False),
        ("activation_checkpoint_impl", "per-block"),
        ("injection_type", "add"),
        ("state_init", "zero"),
        ("sampling_scheme", "uniform"),
    ],
)
def test_invalid_single_value_fields_rejected(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=f"{field}="):
        tiny(**{field: value})


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"n_embd": 66, "num_attention_heads": 6}, "must be even"),
        ({"n_layers_in_recurrent_block": [1, 0]}, "n_layers_in_recurrent_block must be >= 1"),
        ({"mean_backprop_depth": 0}, "mean_backprop_depth must be >= 1"),
        ({"mean_recurrence": 0, "mean_backprop_depth": 0}, "mean_backprop_depth must be >= 1"),
        ({"mean_recurrence": [4, 8], "mean_backprop_depth": [8, 8], "n_layers_in_recurrent_block": [1, 1]}, "mean_recurrence \\(4\\) must be >= mean_backprop_depth"),
        ({"n_layers_in_prelude": -1}, "must be >= 0"),
    ],
)
def test_degenerate_recurrence_values_are_rejected_at_config_time(overrides: dict[str, object], match: str) -> None:
    """
    These used to pass validation and fail (or silently train without gradient) at the first forward.
    """

    with pytest.raises(ValueError, match=match):
        tiny(**overrides)


@pytest.mark.parametrize("value", ["none", "core", "all"])
def test_bf16_residual_stream_values(value: str, tmp_path: Path) -> None:
    cfg = tiny_config(bf16_residual_stream=value)
    assert cfg.bf16_residual_stream == value
    cfg.to_json(tmp_path / "cfg.json")
    assert RecurrentConfig.from_json(tmp_path / "cfg.json").bf16_residual_stream == value


def test_bf16_residual_stream_rejects_other_values() -> None:
    with pytest.raises(ValueError, match="bf16_residual_stream='everywhere'"):
        tiny_config(bf16_residual_stream="everywhere")


def test_bf16_residual_stream_default_is_off_in_every_architecture_yaml() -> None:
    for path in (TINY_ARCHITECTURE, CROW_ARCHITECTURE):
        assert RecurrentConfig.from_yaml(path).bf16_residual_stream == "none", path
