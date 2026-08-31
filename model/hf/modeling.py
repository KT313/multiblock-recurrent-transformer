# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""HuggingFace `transformers` wrapper and export for `RecurrentGPT`.

`export_to_hf` writes safetensors + config.json and copies this package's modules flat next to them, so the folder
loads anywhere with `AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)`. transformers' dynamic-module
loader (`dynamic_module_utils.get_relative_imports`) only follows `from .name import` lines between files of ONE
directory and an `auto_map` entry is `module.Class` with exactly one dot, so sub-packages cannot be exported as they
are: every module `a/b.py` is written as top-level `a_b.py` and its relative imports are rewritten by
`flatten_relative_imports` (import lines only, see there). The `auto_map` names this module's flat name.
"""

import os
import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PretrainedConfig, PreTrainedModel
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..config import RecurrentConfig, RoPESettings
from ..blocks.recurrence import NumSteps, StepsPair, StepsSpec
from ..model import RecurrentGPT

# The `RecurrentConfig` fields stored in config.json (all of them except `name` and the nested `rope_settings`).
_MODEL_FIELDS = (
    "block_size",
    "n_embd",
    "intermediate_size",
    "num_attention_heads",
    "vocab_size",
    "padding_multiple",
    "padded_vocab_size",
    "tie_embeddings",
    "attn_impl",
    "norm_eps",
    "qk_bias",
    "init_strategy",
    "init_orthogonal",
    "activation_checkpoint_impl",
    "injection_type",
    "n_layers_in_prelude",
    "n_layers_in_coda",
    "n_layers_in_recurrent_block",
    "state_init",
    "sampling_scheme",
    "mean_recurrence",
    "mean_backprop_depth",
)


def parse_recurrence_steps(steps_str: str, num_blocks: int) -> StepsPair | list[StepsSpec] | None:
    """Parse "12" into (12, 0) for all blocks, or "4,12,4" into one (n, 0) pair per block."""
    steps_str = steps_str.strip()
    if not steps_str:
        return None
    if "," not in steps_str:
        return (int(steps_str), 0)

    per_block: list[StepsSpec] = []
    for value in steps_str.split(","):
        per_block.append((int(value.strip()), 0))
    if len(per_block) != num_blocks:
        raise ValueError(f"got {len(per_block)} recurrence values but the model has {num_blocks} recurrent blocks")
    return per_block


class RecurrentGPTConfig(PretrainedConfig):  # type: ignore[no-untyped-call]  # transformers' __init_subclass__ is untyped
    """`RecurrentConfig` fields as a `PretrainedConfig` (RoPE settings flattened to `rope_base`)."""

    model_type = "recurrent_gpt"

    def __init__(self, rope_base: int = 50_000, **kwargs: Any) -> None:
        # Defaults only fill keys missing from a config.json (export_to_hf writes every field); the exported folder
        # is standalone and has no access to config/model_architecture/, so the dataclass defaults are used.
        defaults = RecurrentConfig()
        for name in _MODEL_FIELDS:
            setattr(self, name, kwargs.pop(name, getattr(defaults, name)))
        self.rope_base = rope_base

        # Standard HF attribute names, derived from ours (`num_hidden_layers` = the expected unrolled depth).
        self.hidden_size = self.n_embd
        recurrent_depth = 0
        for n_layers, mean_recurrence in zip(self.n_layers_in_recurrent_block, self.mean_recurrence):
            recurrent_depth += n_layers * mean_recurrence
        self.num_hidden_layers = self.n_layers_in_prelude + self.n_layers_in_coda + recurrent_depth

        kwargs.setdefault("tie_word_embeddings", self.tie_embeddings)
        super().__init__(**kwargs)

    @classmethod
    def from_recurrent_config(cls, config: RecurrentConfig, **kwargs: Any) -> "RecurrentGPTConfig":
        field_values: dict[str, Any] = {}
        for name in _MODEL_FIELDS:
            field_values[name] = getattr(config, name)
        return cls(rope_base=config.rope_settings.rope_base, **field_values, **kwargs)

    def to_recurrent_config(self) -> RecurrentConfig:
        field_values: dict[str, Any] = {}
        for name in _MODEL_FIELDS:
            field_values[name] = getattr(self, name)
        return RecurrentConfig(rope_settings=RoPESettings(rope_base=int(self.rope_base)), **field_values)


class RecurrentGPTForCausalLM(PreTrainedModel, GenerationMixin):  # type: ignore[no-untyped-call]  # see above
    """Wraps `RecurrentGPT` for lm-eval / generation; no KV cache, the full sequence is recomputed per step."""

    config_class = RecurrentGPTConfig
    base_model_prefix = "model"
    _no_split_modules = ["SandwichBlock"]
    _supports_cache_class = False
    _tied_weights_keys = {"model.lm_head.weight": "model.transformer.wte.weight"}

    def __init__(self, config: RecurrentGPTConfig) -> None:
        super().__init__(config)
        self.model: RecurrentGPT = RecurrentGPT(config.to_recurrent_config())
        self.num_recurrent_blocks = len(self.model.transformer.core_blocks)
        self.post_init()  # type: ignore[no-untyped-call]  # untyped in transformers

    def _init_weights(self, module: torch.nn.Module) -> None:
        """Weights are initialized by `RecurrentGPT` itself."""

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_dict: bool | None = None,
        num_steps_pair: NumSteps = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, ...] | CausalLMOutputWithPast:
        """`num_steps_pair` as in `RecurrentGPT.forward`; if None, `EVAL_RECURRENCE_STEPS` ("12" or "4,12,4") is used,
        else in eval mode the config's `mean_recurrence` per block, else (training) the sampler."""
        if return_dict is None:
            return_dict = self.config.return_dict

        if num_steps_pair is None:
            env_steps = os.environ.get("EVAL_RECURRENCE_STEPS", "").strip()
            if env_steps:
                num_steps_pair = parse_recurrence_steps(env_steps, self.num_recurrent_blocks)
            elif not self.training:
                per_block: list[StepsSpec] = []
                for mean_recurrence in self.config.mean_recurrence:
                    per_block.append((mean_recurrence, 0))
                num_steps_pair = per_block

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            return_logits=True,
            num_steps_pair=num_steps_pair,
        )
        logits = outputs["logits"]
        loss = None
        if labels is not None:
            loss = outputs["loss"]

        if return_dict:
            return CausalLMOutputWithPast(loss=loss, logits=logits)
        if loss is None:
            return (logits,)
        return (loss, logits)

    def prepare_inputs_for_generation(self, input_ids: torch.Tensor, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Attention is always causal over the full (unpadded) sequence; the HF padding mask is not forwarded."""
        return {"input_ids": input_ids}

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.model.transformer.wte

    def set_input_embeddings(self, value: torch.nn.Module) -> None:
        # transformers types the argument as Module; the contract (resize_token_embeddings) passes an Embedding.
        self.model.transformer.wte = value  # type: ignore[assignment]

    def get_output_embeddings(self) -> torch.nn.Module:
        return self.model.lm_head

    def set_output_embeddings(self, new_embeddings: torch.nn.Linear) -> None:
        self.model.lm_head = new_embeddings  # type: ignore[assignment]  # RecurrentGPT declares its own Linear subclass


AutoConfig.register("recurrent_gpt", RecurrentGPTConfig)
AutoModelForCausalLM.register(RecurrentGPTConfig, RecurrentGPTForCausalLM)


# --- export ----------------------------------------------------------------------------------------------------------

_PACKAGE_DIR = Path(__file__).resolve().parents[1]  # the `model/` package (this file is `model/hf/modeling.py`)
# One `from .x import` / `from ..x.y import` line: leading whitespace, the dots, the dotted module name.
_RELATIVE_IMPORT = re.compile(r"^(?P<indent>[ \t]*)from[ \t]+(?P<dots>\.+)(?P<name>[\w.]*)[ \t]+import\b", re.MULTILINE)


def flat_module_name(module: Path) -> str:
    """Top-level name of a package module in the export folder: `layers/norms.py` -> `layers_norms`."""
    return "_".join(module.with_suffix("").parts)


def flatten_relative_imports(source: str, module: Path, package_dir: Path) -> str:
    """Rewrite the package-relative imports of `module` (its path relative to `package_dir`) for the flat export.

    Only `from .x import` lines change: `from .layers.norms import X` -> `from .layers_norms import X`, `from ..config
    import Y` -> `from .config import Y`. An import must name the defining module file; one that resolves to a package
    (`from .layers import X`, served by an `__init__.py`, or `from . import x`) raises, because `__init__.py` files
    are not exported.
    """
    # Directory of `module` inside the package, e.g. ("hf",) for hf/modeling.py or () for a top-level module.
    module_package = module.parent.parts

    def rewrite(match: re.Match[str]) -> str:
        line = match[0].strip()
        num_dots = len(match["dots"])
        levels_up = num_dots - 1  # `.` = same package, `..` = one package up, ...
        if not match["name"] or levels_up > len(module_package):
            # `from . import x` names no module; more dots than packages would leave the package.
            raise ValueError(f"{module}: {line!r} cannot be flattened (import from the defining module)")

        base_package = module_package[: len(module_package) - levels_up]
        target_module = (*base_package, *match["name"].split("."))
        if not package_dir.joinpath(*target_module).with_suffix(".py").is_file():
            raise ValueError(f"{module}: {line!r} does not name a module file (import from the defining module)")
        flat_name = "_".join(target_module)
        return f"{match['indent']}from .{flat_name} import"

    return _RELATIVE_IMPORT.sub(rewrite, source)


def export_sources(package_dir: Path, out_dir: Path) -> list[Path]:
    """Write every module of the package (no `__init__.py`, tests or `__pycache__`) flat into `out_dir`."""
    written: list[Path] = []
    for source in sorted(package_dir.rglob("*.py")):
        module = source.relative_to(package_dir)
        if "__pycache__" in module.parts or module.name == "__init__.py" or module.name.startswith("test_"):
            continue
        target = out_dir / f"{flat_module_name(module)}.py"
        if target in written:
            raise ValueError(f"{module} and another module both flatten to {target.name}")
        text = flatten_relative_imports(source.read_text(encoding="utf-8"), module, package_dir)
        target.write_text(text, encoding="utf-8")
        written.append(target)
    return written


def export_to_hf(
    model: RecurrentGPT, config: RecurrentConfig, out_dir: str | Path, tokenizer_dir: str | Path | None = None
) -> Path:
    """Write `model` as a self-contained `trust_remote_code` folder (safetensors, config.json, model sources)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    this_module = flat_module_name(Path(__file__).resolve().relative_to(_PACKAGE_DIR))
    hf_config = RecurrentGPTConfig.from_recurrent_config(config)
    hf_config.auto_map = {
        "AutoConfig": f"{this_module}.RecurrentGPTConfig",
        "AutoModelForCausalLM": f"{this_module}.RecurrentGPTForCausalLM",
    }

    # Build the wrapper without allocating weights, then hand it `model`'s tensors (prefixed with `model.`, the
    # wrapper's attribute name).
    with torch.device("meta"):
        hf_model = RecurrentGPTForCausalLM(hf_config)
    state_dict: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        state_dict[f"model.{name}"] = tensor.detach().cpu()
    hf_model.load_state_dict(state_dict, assign=True)
    hf_model.save_pretrained(out_dir, safe_serialization=True)
    export_sources(_PACKAGE_DIR, out_dir)

    if tokenizer_dir is not None:
        AutoTokenizer.from_pretrained(tokenizer_dir).save_pretrained(out_dir)
    return out_dir
