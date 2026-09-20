# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Bounded post-update representation/state diagnostics; definitions in docs/representation_metrics.md.

No hooks on compiled training graphs, backward, head projection, or data-loader reads. A rank-zero probe uses
one document from the last training pack. Private generators and restored eval/RNG contexts isolate training.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast

import torch
from torch import Tensor
from torch.utils.hooks import RemovableHandle

from evaluation.mode import evaluation_mode
from model import RecurrentGPT
from model.blocks.recurrence import adapter_base_projection, core_block_forward
from model.blocks.sandwich import SandwichBlock
from model.model import TransformerModules, prepare_attention_inputs
from training.backend.base import Backend
from training.data.packing import PackedBatch
from training.settings import normalize_log_correlations

PROBE_VERSION = 3  # detailed adapter/attention/MLP correlations are selected by log_correlations
PROBE_SEED = 233
MAX_PROBE_TOKENS = 2048
MAX_PROBE_DEPTH = 8
MAX_SIMILARITY_TOKENS = 64  # at most 2016 distinct pairs, evaluated without a quadratic matrix
STATE_PERTURBATION = 0.01


def _pair_mean(vectors: Tensor) -> tuple[Tensor, Tensor]:
    """Mean dot product over distinct unit-vector pairs; degenerate rows excluded and counted separately."""
    norms = vectors.norm(dim=-1)
    normalized = vectors[norms > 0] / norms[norms > 0, None]
    count = normalized.shape[0]
    pairs = torch.tensor(count*(count-1)//2)
    if count < 2:
        return torch.tensor(float('nan')), pairs
    # ||sum(u)||² - sum(||u||²) is the ordered-pair sum with self-pairs removed.
    mean = (normalized.sum(0).square().sum()-normalized.square().sum())/(count*(count-1))
    return mean.clamp(-1, 1), pairs


@torch.no_grad()
def representation_metrics(
    hidden: Tensor, valid_mask: Tensor | None = None, *, token_ids: Tensor | None = None,
    max_similarity_tokens: int = MAX_SIMILARITY_TOKENS,
) -> dict[str, Tensor]:
    """FP32 summaries for ONE document shaped [token, hidden]; never pool packed documents.

    RMS/dispersion use all valid positions. Similarities use at most 64 evenly spaced valid positions, averaging
    all distinct pairs via vector sums. Correlation centers each token across hidden features first. Padding is
    excluded by valid_mask; repeated token IDs are retained and their sampled pair fraction reported when supplied.
    Undefined metrics are NaN, with valid-pair/degeneracy counts; nonfinite activations are explicitly reported.
    """
    if hidden.ndim != 2 or hidden.shape[1] < 1 or max_similarity_tokens < 2:
        raise ValueError('representation_metrics expects one [token, hidden] document and at least two sample slots')
    if valid_mask is None:
        valid_mask = torch.ones(hidden.shape[0], device=hidden.device, dtype=torch.bool)
    if valid_mask.shape != hidden.shape[:1] or valid_mask.dtype != torch.bool:
        raise ValueError('valid_mask must be a boolean vector over token positions')
    if token_ids is not None and token_ids.shape != hidden.shape[:1]:
        raise ValueError('token_ids must match the document token positions')
    values = hidden.detach()[valid_mask.to(hidden.device)].float().cpu()
    count = values.shape[0]
    result: dict[str, Tensor] = {'tokens': torch.tensor(count)}
    nan = torch.tensor(float('nan'))
    sampled = min(count, max_similarity_tokens)
    result['sampled_tokens'] = torch.tensor(sampled)
    result['nonfinite_fraction'] = (~torch.isfinite(values).all(-1)).float().mean() if count else nan
    if not count or not torch.isfinite(values).all():
        result.update(dict.fromkeys(('correlation', 'cosine', 'dispersion', 'rms', 'zero_vector_fraction',
                                     'zero_variance_fraction', 'same_token_pair_fraction'), nan))
        result.update(cosine_pairs=torch.tensor(0), correlation_pairs=torch.tensor(0))
        return result
    energy = values.square().sum()
    result['rms'] = values.square().mean().sqrt()
    result['dispersion'] = (values-values.mean(0)).square().sum() / energy if energy > 0 else nan
    centered = values-values.mean(-1, keepdim=True)
    result['zero_vector_fraction'] = (values.norm(dim=-1) == 0).float().mean()
    result['zero_variance_fraction'] = (centered.norm(dim=-1) == 0).float().mean()
    positions = torch.linspace(0, count-1, sampled).round().long()
    result['cosine'], result['cosine_pairs'] = _pair_mean(values[positions])
    result['correlation'], result['correlation_pairs'] = _pair_mean(centered[positions])
    result['same_token_pair_fraction'] = nan
    if token_ids is not None and sampled > 1:
        ids = token_ids.detach().cpu()[valid_mask.cpu()][positions]
        _, frequencies = torch.unique(ids, return_counts=True)
        result['same_token_pair_fraction'] = (frequencies*(frequencies-1)).sum().float()/(sampled*(sampled-1))
    return result


@torch.no_grad()
def sampled_correlation_metrics(hidden: Tensor, *, max_tokens: int = MAX_SIMILARITY_TOKENS) -> dict[str, Tensor]:
    """Correlation for one unpadded [token, hidden] document using the existing evenly spaced sample.

    Gather at most 64 positions BEFORE copying to CPU: per-iteration logging must not transfer full activations.
    Counts/fractions here describe the sample; representation_metrics separately reports full-document statistics.
    """
    if hidden.ndim != 2 or hidden.shape[1] < 1 or max_tokens < 2:
        raise ValueError('sampled_correlation_metrics expects one [token, hidden] document and at least two sample slots')
    count = min(hidden.shape[0], max_tokens)
    positions = torch.linspace(0, hidden.shape[0]-1, count).round().long() if count else torch.empty(0, dtype=torch.long)
    values = hidden.detach().index_select(0, positions.to(hidden.device)).float().cpu()
    nan = torch.tensor(float('nan'))
    metrics = {'correlation': nan, 'correlation_pairs': torch.tensor(0), 'sampled_tokens': torch.tensor(count),
               'sampled_nonfinite_fraction': (~torch.isfinite(values).all(-1)).float().mean() if count else nan,
               'sampled_zero_variance_fraction': nan}
    if count and torch.isfinite(values).all():
        centered = values-values.mean(-1, keepdim=True)
        metrics['sampled_zero_variance_fraction'] = (centered.norm(dim=-1) == 0).float().mean()
        metrics['correlation'], metrics['correlation_pairs'] = _pair_mean(centered)
    return metrics


@torch.no_grad()
def state_sensitivity_metrics(state: Tensor, alternative: Tensor, output: Tensor, changed_output: Tensor) -> dict[str, Tensor]:
    """Relative NEXT-output response / actual relative input-state perturbation (not prediction usefulness).

    Evaluate the two outputs with the same weights, injected input, positions and mask. Ratios with zero norm
    or zero effective perturbation are undefined, not evidence of state ignoring. All reductions are FP32.
    """
    if state.shape != alternative.shape or output.shape != changed_output.shape:
        raise ValueError('state/output pairs must have matching shapes')
    state, alternative, output, changed_output = (value.detach().float() for value in (state, alternative, output, changed_output))
    scalars = torch.stack((state.norm(), output.norm(), (alternative-state).norm(), (changed_output-output).norm())).cpu()
    state_norm, output_norm, input_delta, output_delta = scalars.unbind()
    nan = torch.tensor(float('nan'))
    input_relative = input_delta/state_norm if state_norm > 0 else nan
    output_relative = output_delta/output_norm if output_norm > 0 else nan
    response = output_relative/input_relative if input_relative > 0 else nan
    return {'state_sensitivity': response, 'state_perturbation': input_relative, 'output_response': output_relative}


def select_probe_document(batch: PackedBatch, max_tokens: int) -> Tensor | None:
    """Longest supervised document in this rank's actual pack, prefix-capped; padding never becomes a document.

    Rank-zero DDP composition fields cover ALL ranks, so use the local document IDs/labels, not data_tokens or
    padding_tokens. Prompt input tokens are retained even when their targets are ignored.
    """
    if max_tokens < 2:
        return None
    ids, labels, documents = (tensor.detach().cpu() for tensor in (batch.input_ids, batch.labels, batch.document_ids))
    if ids.shape[0] != 1:
        raise ValueError('a packed training microbatch must contain one row')
    _, lengths = torch.unique_consecutive(documents[0], return_counts=True)
    best_start, best_length, start = 0, 0, 0
    for length in lengths.tolist():
        if length > best_length and (labels[0, start:start+length] >= 0).any():
            best_start, best_length = start, length
        start += length
    if best_length < 2:
        return None
    return ids[:, best_start:best_start+min(best_length, max_tokens)].clone()


def _perturb_state(state: Tensor, generator: torch.Generator) -> Tensor:
    value = state.float()
    rms = value.square().mean(-1, keepdim=True).sqrt()
    noise = torch.randn(value.shape, dtype=torch.float32, device=value.device, generator=generator)
    noise /= noise.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-12)
    alternative = value+STATE_PERTURBATION*rms*noise
    alternative *= rms/alternative.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-12)
    return alternative.to(state.dtype)


def _watch_input(module: torch.nn.Module, prefix: str, metrics: dict[str, Tensor], handles: list[RemovableHandle]) -> None:
    def observe(module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
        stats = sampled_correlation_metrics(cast(Tensor, inputs[0])[0])
        metrics.update({f'{prefix}/{key}': value for key, value in stats.items()})
    handles.append(module.register_forward_pre_hook(observe))


def _watch_output(module: torch.nn.Module, prefix: str, metrics: dict[str, Tensor], handles: list[RemovableHandle]) -> None:
    def observe(module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        stats = sampled_correlation_metrics(cast(Tensor, output)[0])
        metrics.update({f'{prefix}/{key}': value for key, value in stats.items()})
    handles.append(module.register_forward_hook(observe))


@contextmanager
def _observe_iteration(
    layers: torch.nn.ModuleList, core: int, iteration: int, targets: frozenset[str], metrics: dict[str, Tensor],
) -> Iterator[None]:
    """Observe only selected boundaries during one normal eager probe call; remove all hooks on every exit."""
    handles: list[RemovableHandle] = []
    try:
        if 'adapter' in targets:
            _watch_input(layers[0], f'adapter/core_{core}/iter_{iteration}/merged_output', metrics, handles)
        if targets.intersection(('attention', 'mlp')):
            for layer_index, raw_layer in enumerate(layers):
                layer = cast(SandwichBlock, raw_layer)
                for target in ('attention', 'mlp'):
                    if target not in targets:
                        continue
                    module = layer.attn if target == 'attention' else layer.mlp
                    prefix = f'{target}/core_{core}/iter_{iteration}/layer_{layer_index}'
                    _watch_input(module, f'{prefix}/input', metrics, handles)
                    _watch_output(module, f'{prefix}/output', metrics, handles)
        yield
    finally:
        for handle in handles:
            handle.remove()


@torch._dynamo.disable(recursive=True)  # type: ignore[no-untyped-call, untyped-decorator]  # torch stub gap
@torch.no_grad()
def track_recurrence_metrics(
    model: RecurrentGPT, backend: Backend, batch: PackedBatch, *, correlations: str | None = "",
) -> dict[str, Tensor]:
    """Rank-zero, post-update probe; unchanged parameters, gradients, RNG, mode and sampling context.

    Uses the real modules/core operation, configured precision/residual dtype and native causal SDPA. Recurrence
    is deterministic min(mean, 8) per core; it does not call the training depth sampler. One extra final core
    application per core measures perturbation response. No vocabulary projection or diagnostic backward.
    """
    if not backend.is_main or not isinstance(getattr(model, 'transformer', None), TransformerModules):
        return {}
    targets = frozenset(filter(None, normalize_log_correlations(correlations).split(',')))
    tokens = select_probe_document(batch, min(MAX_PROBE_TOKENS, model.config.model_max_sequence_length))
    metrics = {'recurrence_probe/available': torch.tensor(int(tokens is not None))}
    if tokens is None:
        metrics.update({key: torch.tensor(float('nan')) for key in ('token_correlation', 'token_dispersion', 'state_sensitivity')})
        return metrics
    device = backend.device
    tokens = tokens.to(device)
    generator = torch.Generator(device=device).manual_seed(PROBE_SEED)
    perturbation_generator = torch.Generator(device=device).manual_seed(PROBE_SEED+1)
    means = model.config.mean_recurrence
    assert isinstance(means, list)
    metrics.update({'recurrence_probe/version': torch.tensor(PROBE_VERSION), 'recurrence_probe/tokens': torch.tensor(tokens.numel()),
                    'recurrence_probe/residual_scale': torch.tensor(model.config.residual_scale),
                    'recurrence_probe/seed': torch.tensor(PROBE_SEED), 'recurrence_probe/rank': torch.tensor(backend.rank),
                    'recurrence_probe/step': torch.tensor(model.step+1)})
    def record(name: str, hidden: Tensor) -> dict[str, Tensor]:
        stats = representation_metrics(hidden[0], token_ids=tokens[0])
        metrics.update({f'representation/{name}/{key}': value for key, value in stats.items()})
        return stats
    # Private generators do not alter global generators, even on devices other than this rank's device.
    devices = [device.index or 0] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices), evaluation_mode(model), backend.autocast():
        rotary, mask = prepare_attention_inputs(model.freqs_cis, tokens)
        modules = model.transformer
        x = modules.wte(tokens)*model.emb_scale
        for layer in modules.prelude:
            x = layer(x, rotary, mask)
        record('prelude', x)
        responses = []
        for index, raw_layers in enumerate(modules.core_blocks):
            layers = cast(torch.nn.ModuleList, raw_layers)
            e = modules.ln_fs[index](x)
            adapter = modules.adapters[index]
            base = adapter_base_projection(e, adapter)
            s = torch.randn(x.shape, device=device, dtype=x.dtype, generator=generator)
            if model.core_bf16_stream and torch.is_autocast_enabled(device.type):
                s = s.to(torch.get_autocast_dtype(device.type))
            depth = min(means[index], MAX_PROBE_DEPTH)
            metrics[f'recurrence/core_{index}/depth'] = torch.tensor(depth)
            block_input_stats = sampled_correlation_metrics(e[0]) if 'adapter' in targets else None
            for iteration in range(depth):
                incoming = s
                prefix = f'adapter/core_{index}/iter_{iteration+1}'
                state_stats = sampled_correlation_metrics(incoming[0]) if block_input_stats is not None else None
                # Observe actual submodule boundaries, not reconstructed/differently rounded projections.
                # Hooks are gone before the perturbation call and before compiled training resumes.
                with _observe_iteration(layers, index, iteration+1, targets, metrics):
                    s = core_block_forward(incoming, e, rotary, mask, adapter, layers, base)
                if block_input_stats is not None and state_stats is not None:
                    if f'{prefix}/merged_output/correlation' not in metrics:
                        raise RuntimeError('No adapter output observed during the diagnostic core iteration')
                    stages = {'block_input': block_input_stats, 'state_input': state_stats,
                              'core_output': sampled_correlation_metrics(s[0])}
                    metrics.update({f'{prefix}/{stage}/{key}': value
                                    for stage, stats in stages.items() for key, value in stats.items()})
                if iteration == depth-1:
                    alternative = _perturb_state(incoming, perturbation_generator)
                    changed = core_block_forward(alternative, e, rotary, mask, adapter, layers, base)
                    response = state_sensitivity_metrics(incoming, alternative, s, changed)
                    metrics.update({f'recurrence/core_{index}/{key}': value for key, value in response.items()})
                    responses.append(response['state_sensitivity'])
            record(f'core_{index}', s)
            x = x+s
        for layer in modules.coda:
            x = layer(x, rotary, mask)
        final = record('pre_head', modules.ln_final(x))
        metrics.update(token_correlation=final['correlation'], token_dispersion=final['dispersion'],
                       state_sensitivity=torch.stack(responses).min())
    return metrics
