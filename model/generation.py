# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Ephemeral inference state: independent per-token/core latents and per-occurrence rotated K/V.

No state is registered on the model or serialized in a checkpoint. A session belongs to one model revision,
batch, precision and fixed recurrence schedule. Its private normal generators assign noise one token at a time,
so prefill and incremental execution use the same noise regardless of how the prefix is chunked. A model with
`use_trainable_initial_state` draws no latents here: every token starts from the model's learned state.
"""

from __future__ import annotations

from typing import TypeAlias

import torch
from torch import Tensor

Slot: TypeAlias = tuple[str, int, int, int]


class KVCache:
    """One inference attention occurrence, with capacity grown in 256-token chunks.

    K is already rotated, V is the original projected value. Public key/value tensors and returned tensors are
    views of the populated prefix; unused capacity is never passed to attention. Treat these views as read-only.
    """

    _ALLOCATION_CHUNK = 256

    def __init__(self) -> None:
        self._key_buffer: Tensor | None = None
        self._value_buffer: Tensor | None = None
        self._length = 0

    @property
    def key(self) -> Tensor | None:
        return None if self._key_buffer is None else self._key_buffer[:, :self._length]

    @property
    def value(self) -> Tensor | None:
        return None if self._value_buffer is None else self._value_buffer[:, :self._length]

    def append_and_get(self, key: Tensor, value: Tensor) -> tuple[Tensor, Tensor]:
        """Append new (B, S, heads, head_dim) K/V and return the entire populated prefix without copying it."""
        if key.ndim != 4 or value.ndim != 4 or key.shape[:2] != value.shape[:2]:
            raise ValueError("cache K/V must be four-dimensional with matching batch and sequence lengths")
        for incoming, stored in ((key, self._key_buffer), (value, self._value_buffer)):
            if stored is not None and (
                incoming.shape[:1] + incoming.shape[2:] != stored.shape[:1] + stored.shape[2:]
                or incoming.dtype != stored.dtype or incoming.device != stored.device
            ):
                raise ValueError("cache K/V batch, head dimensions, dtype and device must remain unchanged")

        used = self._length
        needed = used + key.shape[1]
        if self._key_buffer is None or needed > self._key_buffer.shape[1]:
            capacity = ((needed + self._ALLOCATION_CHUNK - 1) // self._ALLOCATION_CHUNK) * self._ALLOCATION_CHUNK
            key_buffer = key.new_empty((key.shape[0], capacity, *key.shape[2:]))
            value_buffer = value.new_empty((value.shape[0], capacity, *value.shape[2:]))
            if self._key_buffer is not None:
                assert self._value_buffer is not None
                key_buffer[:, :used].copy_(self._key_buffer[:, :used])
                value_buffer[:, :used].copy_(self._value_buffer[:, :used])
            self._key_buffer, self._value_buffer = key_buffer, value_buffer

        assert self._value_buffer is not None
        self._key_buffer[:, used:needed].copy_(key)
        self._value_buffer[:, used:needed].copy_(value)
        self._length = needed
        populated_key, populated_value = self.key, self.value
        assert populated_key is not None and populated_value is not None
        return populated_key, populated_value


class GenerationState:
    """Persistent-token latent policy, usable with KV caching or as a full-prefix reference.

    Supply the same seed to independent sessions to compare cached and full-prefix forwards. With seed=None,
    draw one CPU seed from the caller's torch RNG (sampling helpers already isolate that RNG). Noise uses private
    generators, independent of token selection and of the order cores are evaluated. Padding columns get noise too;
    changing batch membership/width can change the noise of real tokens, as with legacy sampling.
    """

    is_compileable = False  # transformers' generation auto-compile must not compile this Python session

    def __init__(self, *, seed: int | None = None) -> None:
        self._seed = int(torch.randint(0, 2**31 - 1, ()).item()) if seed is None else seed
        self.slots: dict[Slot, KVCache] = {}
        self.latents: dict[int, Tensor] = {}
        self._generators: dict[int, torch.Generator] = {}
        self._signature: tuple[object, ...] | None = None
        self._steps: tuple[int, ...] | None = None
        self._ids: Tensor | None = None
        self._mask: Tensor | None = None
        self._positions: Tensor | None = None
        self._length = 0
        self._pending: tuple[Tensor, Tensor, Tensor] | None = None
        self._failed = False
        self._cached: bool | None = None

    @property
    def seed(self) -> int:
        return self._seed

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._length

    def invalidate(self) -> None:
        """An exception leaves no partially updated cache available for reuse."""
        self.slots.clear()
        self.latents.clear()
        self._generators.clear()
        self._pending = None
        self._ids = self._mask = self._positions = None
        self._signature = None
        self._length = 0
        self._failed = True

    def validate_prefix(self, input_ids: Tensor, positions: Tensor | None = None) -> None:
        """HF passes the full sequence before slicing; reject edited/reordered cached prefix rows."""
        if self._failed:
            raise ValueError("generation state was invalidated; start a new generation")
        if self._ids is not None and (
            input_ids.shape[0] != self._ids.shape[0]
            or input_ids.shape[1] <= self._length
            or not torch.equal(input_ids[:, :self._length], self._ids)
        ):
            raise ValueError("generation prefix or batch changed; start a new generation")

        if positions is not None and self._positions is not None:
            prefix_positions = positions[..., :self._length]
            if prefix_positions.dim() == 1:
                prefix_positions = prefix_positions.unsqueeze(0).expand(input_ids.shape[0], -1)
            elif prefix_positions.dim() == 2 and prefix_positions.shape[0] == 1:
                prefix_positions = prefix_positions.expand(input_ids.shape[0], -1)
            if not torch.equal(prefix_positions, self._positions):
                raise ValueError("generation positions changed; start a new generation")

    def begin(
        self, model: torch.nn.Module, input_ids: Tensor, mask: Tensor, positions: Tensor,
        steps: tuple[int, ...], *, use_cache: bool, max_positions: int,
    ) -> int:
        """Validate a forward before any K/V mutation and return its first query storage column.

        Cached input_ids are new tokens only; mask always describes the entire prefix. Reference input_ids and
        positions are the entire prefix. Explicit nondefault positions remain supported and validated.
        """
        # validate mode, model identity and incoming prefix before any mutation
        self._validate_generation_mode(model, use_cache)
        signature = self._build_model_signature(model, input_ids)
        self._validate_model_signature(signature, steps)
        start = self._validate_generation_inputs(input_ids, mask, positions, use_cache, max_positions)
        all_ids, all_positions = self._assemble_pending_prefix(input_ids, positions, use_cache)

        # retain the validated prefix until finish publishes it
        self._signature, self._steps, self._cached = signature, steps, use_cache
        self._pending = all_ids.clone(), mask.clone(), all_positions.clone()
        return start

    def _validate_generation_mode(self, model: torch.nn.Module, use_cache: bool) -> None:
        if self._failed or self._pending is not None:
            raise ValueError("generation state is invalid or has an unfinished forward; start a new generation")
        if model.training or torch.is_grad_enabled():
            raise ValueError("generation state requires eval mode and no_grad/inference_mode")
        if self._cached is not None and self._cached != use_cache:
            raise ValueError("cannot switch cache mode within a generation")

    def _build_model_signature(self, model: torch.nn.Module, input_ids: Tensor) -> tuple[object, ...]:
        parameters, buffers = tuple(model.parameters()), tuple(model.buffers())
        if any(tensor.is_inference() for tensor in (*parameters, *buffers)):
            raise ValueError(
                "generation cache needs versioned model tensors; construct/load the model outside torch.inference_mode "
                "(torch.no_grad is fine)"
            )
        signature: tuple[object, ...] = (
            id(model), getattr(model, "step", None), torch.get_float32_matmul_precision(),
            tuple((id(t), t.data_ptr(), t._version, t.dtype, t.device) for t in parameters),
            tuple((id(t), t.data_ptr(), t._version, t.dtype, t.device) for t in buffers),
            torch.is_autocast_enabled(input_ids.device.type),
            torch.get_autocast_dtype(input_ids.device.type),
        )
        return signature

    def _validate_model_signature(self, signature: tuple[object, ...], steps: tuple[int, ...]) -> None:
        if self._signature is not None and self._signature != signature:
            raise ValueError("model weights, device or precision changed; start a new generation")
        if self._steps is not None and self._steps != steps:
            raise ValueError("recurrence schedule changed; start a new generation")

    def _validate_generation_inputs(
        self, input_ids: Tensor, mask: Tensor, positions: Tensor, use_cache: bool, max_positions: int,
    ) -> int:
        batch, query_length = input_ids.shape
        start = self._length if use_cache else 0
        total = start + query_length
        if query_length < 1 or mask.shape != (batch, total) or positions.shape != (batch, query_length):
            raise ValueError("generation needs new-token ids/positions and a full-prefix (B, S) attention mask")
        if self._ids is not None and (batch != self._ids.shape[0] or total <= self._length):
            raise ValueError("generation must append tokens to the same batch")
        if positions.device != input_ids.device or mask.device != input_ids.device:
            raise ValueError("generation ids, mask and positions must share a device")
        if bool(((positions < 0) | (positions >= max_positions)).any()):
            raise ValueError("generation positions exceed the model's RoPE table")
        if self._mask is not None and not torch.equal(mask[:, :self._length], self._mask):
            raise ValueError("generation padding mask changed; start a new generation")
        return start

    def _assemble_pending_prefix(self, input_ids: Tensor, positions: Tensor, use_cache: bool) -> tuple[Tensor, Tensor]:
        if use_cache:
            all_ids = input_ids if self._ids is None else torch.cat((self._ids, input_ids), dim=1)
            all_positions = positions if self._positions is None else torch.cat((self._positions, positions), dim=1)
        else:
            self.validate_prefix(input_ids)
            if self._positions is not None and not torch.equal(positions[:, :self._length], self._positions):
                raise ValueError("generation positions changed; start a new generation")
            all_ids, all_positions = input_ids, positions
        return all_ids, all_positions

    def finish(self) -> None:
        assert self._pending is not None
        self._ids, self._mask, self._positions = self._pending
        self._length = self._ids.shape[1]
        self._pending = None

    def latent(self, core: int, x: Tensor, start: int) -> Tensor:
        """Retain each token's independent initial normal state, once per core and generation."""
        total = start + x.shape[1]
        old = self.latents.get(core)
        old_length = 0 if old is None else old.shape[1]
        if old is not None and (old.dtype != x.dtype or old.device != x.device or old.shape[0] != x.shape[0]):
            raise ValueError("latent dtype, device or batch changed; start a new generation")
        if core not in self._generators:
            self._generators[core] = torch.Generator(device=x.device).manual_seed((self.seed + 2**24 * core) % (2**63 - 1))
        if total > old_length:
            # Draw each column separately so chunk length never changes the mapping of RNG values to tokens.
            columns = [
                torch.randn((x.shape[0], 1, x.shape[2]), device=x.device, dtype=x.dtype, generator=self._generators[core])
                for _ in range(total - old_length)
            ]
            if old is not None:
                columns.insert(0, old)
            old = torch.cat(columns, dim=1)
            self.latents[core] = old
        assert old is not None
        return old[:, start:total]

    def slot(self, key: Slot) -> KVCache:
        if key not in self.slots:
            self.slots[key] = KVCache()
        return self.slots[key]

    def reorder_cache(self, beam_idx: Tensor) -> None:
        raise ValueError("beam/reordered generation is unsupported; use_cache=False selects legacy generation")
