# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Ensure local HFLM generation never synchronizes unequal decoding workloads."""
from typing import Any


def create_local_harness_class(base: Any) -> Any:
    class LocalHFLM(base):  # type: ignore[misc]
        inference_input_tokens_with_padding: int = 0
        inference_generated_tokens_with_padding: int = 0

        def _model_call(self, inps: Any, *args: Any, **kwargs: Any) -> Any:
            self.inference_input_tokens_with_padding += int(inps.numel())
            return super()._model_call(inps, *args, **kwargs)

        def _model_generate(self, *args: Any, **kwargs: Any) -> Any:
            kwargs["synced_gpus"] = False
            context = args[0] if args else kwargs["context"]
            self.inference_input_tokens_with_padding += int(context.numel())
            output = super()._model_generate(*args, **kwargs)
            self.inference_generated_tokens_with_padding += int(output.shape[0] * (output.shape[1] - context.shape[1]))
            return output
    return LocalHFLM
