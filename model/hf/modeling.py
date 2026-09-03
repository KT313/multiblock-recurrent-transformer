# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""HuggingFace `transformers` wrapper and export for `RecurrentGPT`.

`export_to_hf` writes safetensors + config.json and copies this package's modules flat next to them, so the folder
loads with `AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)`. Flat because transformers' dynamic
module loader only follows `from .name import` lines within one directory and an `auto_map` entry is `module.Class`
with exactly one dot: every module `a/b.py` is written as `a_b.py` and `flatten_relative_imports` rewrites its import
lines. The `auto_map` names this module's flat name.
"""

import os
import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PretrainedConfig, PreTrainedModel
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..config import RecurrentConfig, RoPESettings, broadcast_per_block
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


def mask_padded_vocabulary(logits: torch.Tensor, vocab_size: int, padded_vocab_size: int) -> torch.Tensor:
    """`logits` with the embedding table's padding columns (`vocab_size:`) set to -inf; unchanged when the table is
    not padded. Those columns are trained on no target, and `generate(do_sample=True)` could otherwise draw an id the
    tokenizer cannot decode. Only the HF wrapper masks: the training loss must see the model's own logits (the golden
    tests check those numbers)."""
    if padded_vocab_size <= vocab_size:
        return logits
    masked = logits.clone()  # a clone, not an in-place write: `logits` is part of the autograd graph
    masked[..., vocab_size:] = float("-inf")
    return masked


class RecurrentGPTConfig(PretrainedConfig):  # type: ignore[no-untyped-call]  # transformers' __init_subclass__ is untyped
    """`RecurrentConfig` fields as a `PretrainedConfig` (RoPE settings flattened to `rope_base`)."""

    model_type = "recurrent_gpt"

    def __init__(self, rope_base: int = 50_000, **kwargs: Any) -> None:
        # Defaults fill keys missing from a config.json (export_to_hf writes every field); the exported folder has no
        # access to config/model_architecture/, so the dataclass defaults apply.
        defaults = RecurrentConfig()
        values: dict[str, Any] = {}
        for name in _MODEL_FIELDS:
            values[name] = kwargs.pop(name, getattr(defaults, name))
        # Per-block fields take the int shorthand of `RecurrentConfig` (`mean_recurrence: 12` = every block) and are
        # broadcast the way `RecurrentConfig.__post_init__` does; everything below reads one entry per core block.
        layers_per_block = values["n_layers_in_recurrent_block"]
        values["n_layers_in_recurrent_block"] = [layers_per_block] if isinstance(layers_per_block, int) else list(layers_per_block)
        num_blocks = len(values["n_layers_in_recurrent_block"])
        for name in ("mean_recurrence", "mean_backprop_depth"):
            values[name] = broadcast_per_block(name, values[name], num_blocks)
        for name, value in values.items():
            setattr(self, name, value)
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
        num_steps: NumSteps = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, ...] | CausalLMOutputWithPast:
        """`num_steps` as in `RecurrentGPT.forward`. When None: in eval mode `EVAL_RECURRENCE_STEPS` ("12" or
        "4,12,4") if set, else the config's `mean_recurrence` per block; in training mode always the sampler (the env
        var sets zero backprop iterations, which would silently train the recurrence without gradient).

        Logits over the padding columns of the embedding table are -inf, so sampling only produces decodable ids. The
        inner model is untouched: masking there would change the training numerics.

        `labels` follow the HuggingFace contract and are shifted here: `model(x, labels=x).loss` is the next-token
        loss `CE(logits[t], x[t + 1])`, positions labelled -100 ignored. The inner `RecurrentGPT` expects pre-shifted
        labels, so it gets `labels=None`. `attention_mask` `(B, S)` (1 = keep) and `position_ids` (1-D or `(B, S)`)
        are forwarded to the inner model."""
        if return_dict is None:
            return_dict = self.config.return_dict

        if num_steps is None and not self.training:
            env_steps = os.environ.get("EVAL_RECURRENCE_STEPS", "").strip()
            if env_steps:
                num_steps = parse_recurrence_steps(env_steps, self.num_recurrent_blocks)
            else:
                per_block: list[StepsSpec] = []
                for mean_recurrence in self.config.mean_recurrence:
                    per_block.append((mean_recurrence, 0))
                num_steps = per_block

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=None,  # the inner loss is unshifted; the HF contract shifts, below
            return_logits=True,
            num_steps=num_steps,
        )
        logits = outputs["logits"]
        assert logits is not None  # `return_logits=True`
        logits = mask_padded_vocabulary(logits, int(self.config.vocab_size), int(self.config.padded_vocab_size))
        loss: torch.Tensor | None = None
        if labels is not None:
            # `contiguous()`: both slices are views, and the inner `loss` flattens them with `view`.
            loss = self.model.loss(logits[:, :-1, :].contiguous(), labels[:, 1:].contiguous())

        if return_dict:
            # transformers annotates the field as FloatTensor; ours is a plain float32 Tensor (there is no such subclass)
            return CausalLMOutputWithPast(loss=loss, logits=logits)  # type: ignore[arg-type]
        if loss is None:
            return (logits,)
        return (loss, logits)

    def prepare_inputs_for_generation(self, input_ids: torch.Tensor, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """The whole sequence every step (no KV cache), plus the padding mask and the positions.

        `generate` left-pads prompts of different lengths; without the mask the model would attend to the pads and
        count them as positions. With it every row's positions are `cumsum(mask) - 1`, clamped at 0, so the first real
        token of every row sits at position 0. Everything else `generate` passes (a cache, embeddings) is dropped.
        `*args` / `**kwargs`: transformers' signature changes between versions and `generate` only passes keywords."""
        attention_mask: torch.Tensor | None = kwargs.get("attention_mask")
        position_ids: torch.Tensor | None = kwargs.get("position_ids")
        model_inputs: dict[str, Any] = {"input_ids": input_ids}
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask
            if position_ids is None:
                position_ids = (attention_mask.to(torch.long).cumsum(-1) - 1).clamp(min=0)
        if position_ids is not None:
            model_inputs["position_ids"] = position_ids
        return model_inputs

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
    import Y` -> `from .config import Y`. An import that resolves to a package (`from .layers import X` or
    `from . import x`) raises: `__init__.py` files are not exported, so imports must name the defining module."""
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

    # Build the wrapper without allocating weights, then hand it `model`'s tensors under the wrapper's `model.` prefix.
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
