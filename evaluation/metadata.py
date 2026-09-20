# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Versioned, allowlisted benchmark settings; unknown values stay explicitly unavailable.

HFLM's configured batch size is not a trace of actual batch lengths. Automatic detection can keep
only its last scoring schedule, and generation/rolling scoring can use an unreported local value.
No optional package is imported, and no model/cache paths or environment values are recorded here.
"""

import inspect
import sys
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import torch

from evaluation.session import InferenceSession

METADATA_VERSION = 1


def _attribute(obj: Any, name: str) -> dict[str, Any]:
    if not hasattr(obj, name):
        return {"value": None, "source": "unavailable"}
    value = getattr(obj, name)
    if isinstance(value, torch.dtype):
        value = str(value)
    return {"value": value, "source": f"hflm.{name}"}


def _cache_setting(results: Mapping[str, Any], evaluator: Any, name: str, *, enabled_only: bool = False) -> dict[str, Any]:
    config = results.get("config", {})
    if isinstance(config, Mapping) and name in config:
        value, source = config[name], "harness_result.config"
    else:
        parameter = inspect.signature(evaluator).parameters.get(name)
        if parameter is None or parameter.default is inspect.Parameter.empty:
            return {"value": None, "source": "unavailable"}
        value, source = parameter.default, "simple_evaluate.signature_default"
    # A response-cache path is not useful execution metadata and may contain sensitive account details.
    return {"value": value is not None if enabled_only else value, "source": source}


def dependency_versions(*, custom_kernels: bool) -> dict[str, Any]:
    """Distribution versions for used packages, without loading optional dependencies for reporting."""
    versions: dict[str, Any] = {}
    for distribution, module in (("torch", "torch"), ("transformers", "transformers"), ("lm_eval", "lm_eval"), ("triton", "triton")):
        applicable = distribution != "triton" or custom_kernels
        if not applicable or module not in sys.modules:
            versions[distribution] = {"version": None, "status": "not_applicable" if not applicable else "not_loaded"}
            continue
        try:
            versions[distribution] = {"version": version(distribution), "status": "installed_distribution"}
        except PackageNotFoundError:
            versions[distribution] = {"version": None, "status": "distribution_unavailable"}
    return versions


def benchmark_execution_metadata(
    session: InferenceSession, language_model: Any, wrapper: Any, results: Mapping[str, Any],
    evaluator: Any, *, batch_size: int | str, context_cap: int, custom_kernels: bool,
) -> dict[str, Any]:
    """Read constructed harness settings after evaluation, retaining automatic detection when exposed.

    A source of ``unavailable`` is distinct from an observed None (e.g. disabled mixed precision).
    Generation config is a default, not evidence that a task used it: HFLM can override generate kwargs.
    """
    autocast_enabled = torch.is_autocast_enabled(session.device.type)
    config = results.get("config", {})
    reported_batches = config.get("batch_sizes") if isinstance(config, Mapping) else None
    return {
        "batching": {
            "requested": batch_size,
            "configured": _attribute(language_model, "batch_size"),
            "automatic_schedule": _attribute(language_model, "batch_sizes"),
            "automatic_max_batch_size": _attribute(language_model, "max_batch_size"),
            "harness_reported_batch_sizes": reported_batches,
            "actual_batch_lengths": None,
            "observation_scope": "retained_harness_state_only; automatic selections may be incomplete",
        },
        "precision": {
            "requested_policy": session.execution_policy.precision,
            "session_autocast_enabled": autocast_enabled,
            "session_autocast_dtype": str(torch.get_autocast_dtype(session.device.type)) if autocast_enabled else None,
            "hflm_mixed_precision_dtype": _attribute(language_model, "mixed_precision_dtype"),
            "hflm_softmax_dtype": _attribute(language_model, "softmax_dtype"),
            "parameter_dtypes": sorted({str(parameter.dtype) for parameter in session.model.parameters()}),
            "device_type": session.device.type,
        },
        "custom_kernels": {"enabled": custom_kernels, "selection_source": "model.config.use_custom_kernels"},
        "context": {"requested_cap": context_cap, "effective_cap": _attribute(language_model, "max_length")},
        "scoring_cache": {
            "logits_cache": _attribute(language_model, "logits_cache"),
            "response_cache_enabled": _cache_setting(results, evaluator, "use_cache", enabled_only=True),
            "cache_requests": _cache_setting(results, evaluator, "cache_requests"),
            "rewrite_requests_cache": _cache_setting(results, evaluator, "rewrite_requests_cache"),
            "delete_requests_cache": _cache_setting(results, evaluator, "delete_requests_cache"),
            "wrapper_forward_use_cache_default": False,
        },
        "generation": {
            "wrapper_generation_config_use_cache": getattr(wrapper.generation_config, "use_cache", None),
            "hflm_max_gen_toks": _attribute(language_model, "max_gen_toks"),
            "effective_per_request_use_cache": {"value": None, "source": "unavailable"},
            "policy_when_use_cache_true": "persistent_per_token_core_latents; separate_kv_per_recurrence_occurrence",
            "policy_when_use_cache_false": "legacy_full_prefix_latent_resampling; no_kv_cache",
            "observation_scope": "wrapper_defaults_only; harness_or_task_generation_overrides_are_not_observed",
        },
    }
