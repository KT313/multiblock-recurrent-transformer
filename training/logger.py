# Ported from seal-rg/recurrent-pretraining (Apache-2.0), commit 3055b7f; modified by Tobias Kerner 2025-2026.
"""Thin wandb wrapper (offline by default) and the gradient/parameter/MFU metric helpers that were logged."""

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, cast

import torch
from torch.nn import Module
from torch.optim import Optimizer

from model import RecurrentGPT
from training.checkpoint import unwrap_compiled

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run


class Logger:
    """wandb run wrapper; every method is a no-op when `enabled=False` (wandb is then never imported)."""

    def __init__(
        self, project: str, run_name: str, out_dir: str | Path, offline: bool = True, enabled: bool = True
    ) -> None:
        self.enabled = enabled
        self.run: Optional[Run] = None
        if not enabled:
            return
        import wandb

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        self.run = wandb.init(
            project=project, name=run_name, dir=str(out_dir), mode="offline" if offline else "online"
        )

    def log(self, metrics: dict[str, Any], step: int) -> None:
        if self.run is None:
            return
        self.run.log({k: _to_scalar(v) for k, v in metrics.items()}, step=step)

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        if self.run is None:
            return
        self.run.config.update(params, allow_val_change=True)  # type: ignore[no-untyped-call]  # wandb Config.update is unannotated

    def log_summary(self, values: dict[str, Any]) -> None:
        if self.run is None:
            return
        for k, v in values.items():
            self.run.summary[k] = _to_scalar(v)

    def finish(self) -> None:
        if self.run is None:
            return
        self.run.finish()
        self.run = None


def _to_scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return value.item()
    return value


def num_parameters(model: Module, only_trainable: bool = False) -> int:
    """Total number of parameters (tied weights counted once, as `parameters()` deduplicates them)."""
    param_list = list(model.parameters())
    if only_trainable:
        param_list = [p for p in param_list if p.requires_grad]
    return sum(p.numel() for p in param_list)


def describe_parameters(model: Module) -> str:
    """The parameter-count line printed at the start of a run: total parameters, parameters inside the recurrent core
    blocks and the count of the unrolled model at the mean recurrence (`total - recurrent + recurrent * mean of
    mean_recurrence`). Accepts the compiled wrapper too (it is unwrapped)."""
    plain_model = cast(RecurrentGPT, unwrap_compiled(model))
    total_parameters = num_parameters(plain_model)
    core_blocks = cast(Iterable[Module], plain_model.transformer.core_blocks)
    recurrent_parameters = sum(p.numel() for block in core_blocks for p in block.parameters())
    mean_recurrence = cast(list[int], plain_model.config.mean_recurrence)  # a list after RecurrentConfig.__post_init__
    mean_of_means = sum(mean_recurrence) / len(mean_recurrence)
    unrolled_parameters = int(total_parameters - recurrent_parameters + recurrent_parameters * mean_of_means)
    return (
        f"Model: {total_parameters:,} parameters, {recurrent_parameters:,} in recurrent blocks, unfolds to "
        f"{unrolled_parameters:,} at mean recurrence."
    )


def _reverse_engineer_adam_effective_lr(
    param: torch.Tensor, param_state: dict[str, torch.Tensor], group: dict[str, Any]
) -> torch.Tensor:
    """Recompute Adam's per-element effective LR (ignoring bias correction and the scheduled LR)."""
    grad = param.grad
    assert grad is not None, "effective LR needs a gradient"
    exp_avg = param_state["exp_avg"].float()
    denom = param_state["exp_avg_sq"].float().sqrt().add_(group["eps"])
    return torch.where(
        grad.float().abs() > group["eps"],
        exp_avg / denom / grad.float(),
        exp_avg / denom / group["eps"],
    )


def _qkv_dims(model: Module) -> Optional[tuple[int, int, int]]:
    """(n_embd, query width, key/value width) for slicing fused qkv gradients; None for non-transformer models."""
    config = getattr(model, "config", None)
    if config is None or not all(hasattr(config, a) for a in ("n_embd", "head_size", "num_attention_heads")):
        return None
    return config.n_embd, config.n_embd, config.head_size * config.num_attention_heads  # no GQA: kv width == q width


@torch.no_grad()
def track_gradient_metrics(model: Module, optimizer: Optimizer) -> dict[str, torch.Tensor]:
    """Gradient norms, Adam second-moment RMS, effective LRs and parameter norms. Call after `optimizer.step()`
    and before `zero_grad()`."""
    metrics: dict[str, torch.Tensor] = {}
    dims = _qkv_dims(model)
    transformer = getattr(model, "transformer", None)
    wte_module: Optional[Module] = getattr(transformer, "wte", None)
    wte_weight: Optional[torch.Tensor] = getattr(wte_module, "weight", None)

    # Specific gradient norms
    qkv_layer_counter, mlp_layer_counter = 0, 0
    qkv_params: list[torch.Tensor] = []
    proj_params: list[torch.Tensor] = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            if "qkv" in name and "weight" in name:
                qkv_params.append(param)
                if (~torch.isfinite(param.grad)).sum() == 0:
                    if dims is not None and param.grad.numel() % dims[0] == 0:
                        q_grad = param.grad.view(-1, dims[0])[: dims[1], :]
                        metrics[f"query_grad_{qkv_layer_counter}"] = q_grad.norm()
                else:
                    metrics[f"query_grad_{qkv_layer_counter}"] = torch.as_tensor(float("NaN"))
                qkv_layer_counter += 1
            if "mlp" in name and "proj" in name and "weight" in name:
                proj_params.append(param)
                if (~torch.isfinite(param.grad)).sum() == 0:
                    metrics[f"ffn2_grad_{mlp_layer_counter}"] = param.grad.norm()
                else:
                    metrics[f"ffn2_grad_{mlp_layer_counter}"] = torch.as_tensor(float("NaN"))
                mlp_layer_counter += 1

    # 2nd moment quality and effective learning rates
    total_rms: torch.Tensor | float = 0.0
    num_params_with_grad = 0
    qkv_layer_counter, mlp_layer_counter = 0, 0
    params_with_finite_grad = []
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None and (~torch.isfinite(param.grad)).sum() == 0:
                params_with_finite_grad.append(param)
                if param in optimizer.state and "exp_avg_sq" in optimizer.state[param]:
                    exp_avg_sq = optimizer.state[param]["exp_avg_sq"]
                    if exp_avg_sq.shape == param.grad.shape:
                        rms = (
                            param.grad.float().pow(2).div_(exp_avg_sq.float().clamp_(min=group["eps"] ** 2)).mean().sqrt()
                        )
                        total_rms += rms
                        num_params_with_grad += 1
                        if wte_weight is not None and param is wte_weight:
                            metrics["embed_RMS"] = rms

                        if any(param is p for p in qkv_params):  # identity check, `in` would compare values
                            qkv_lr = _reverse_engineer_adam_effective_lr(param, optimizer.state[param], group)
                            if dims is not None and qkv_lr.numel() % dims[0] == 0:
                                H, dim_q, dim_kv = dims
                                qkv_lr = qkv_lr.view(-1, H)
                                metrics[f"q_effective_lr_{qkv_layer_counter}"] = qkv_lr[:dim_q, :].mean()
                                metrics[f"k_effective_lr_{qkv_layer_counter}"] = qkv_lr[dim_q : dim_q + dim_kv, :].mean()
                                metrics[f"v_effective_lr_{qkv_layer_counter}"] = qkv_lr[dim_q + dim_kv :, :].mean()
                            qkv_layer_counter += 1

                        if any(param is p for p in proj_params):
                            proj_lr = _reverse_engineer_adam_effective_lr(param, optimizer.state[param], group)
                            metrics[f"ffn2_effective_lr_{mlp_layer_counter}"] = proj_lr.mean()
                            mlp_layer_counter += 1

    if num_params_with_grad > 0:
        metrics["avg_RMS"] = torch.as_tensor(total_rms / num_params_with_grad)  # already a Tensor after one add

    if len(params_with_finite_grad) > 0:
        metrics["local_l1_grad_norm"] = torch.mean(
            torch.stack([torch.norm(p.grad.detach(), 1.0) for p in params_with_finite_grad])
        )

    # Parameter norms
    metrics["l2_param_norm"] = torch.norm(torch.stack([torch.norm(p.detach()) for p in model.parameters()]))
    metrics["l1_param_norm"] = torch.mean(torch.stack([torch.norm(p.detach(), 1.0) for p in model.parameters()]))

    core_blocks = getattr(transformer, "core_blocks", None)
    if core_blocks is not None:
        for block_idx, core_block in enumerate(core_blocks):
            metrics[f"core_block_{block_idx}_l2_param_norm"] = torch.norm(
                torch.stack([torch.norm(p.detach()) for p in core_block.parameters()])
            )
    if wte_module is not None and wte_weight is not None:
        metrics["word_embed_l2_param_norm"] = torch.norm(
            torch.stack([torch.norm(p.detach()) for p in wte_module.parameters()])
        )
        metrics["model_l2_param_norm"] = torch.norm(
            torch.stack([torch.norm(p.detach()) for n, p in model.named_parameters() if "wte" not in n])
        )
    return metrics
