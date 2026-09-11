"""Strict custom-kernel loading and actionable errors; native execution is explicit."""
import hashlib
import importlib.util
from functools import wraps
from typing import Callable, ParamSpec, TypeVar

import torch
from torch import Tensor

MLPKernel = Callable[[Tensor, torch.nn.Module, torch.nn.Module, torch.nn.Module], Tensor]
HeadKernel = Callable[[Tensor, torch.nn.Module, Tensor, float, int], Tensor]
RoPEKernel = Callable[[Tensor | None, Tensor, Tensor, int], tuple[Tensor, Tensor, Tensor]]
DISABLE_HINT = 'Set use_custom_kernels: false in the run/model config to use native PyTorch operations.'
P = ParamSpec('P')
R = TypeVar('R')


class CustomKernelError(RuntimeError):
    """A requested custom kernel could not be loaded or executed."""


def kernel_namespace(module: str, operation: str) -> str:
    """Distinct stable registrations for repository and independently loaded HF export modules."""
    return f"mbrt_{operation}_{hashlib.sha256(module.encode()).hexdigest()[:16]}"


def kernels_available() -> bool:
    # This feature probe does not initialize CUDA. An enabled-but-unavailable kernel is an error, not fallback.
    return (
        torch.version.cuda is not None and importlib.util.find_spec('triton') is not None
        and all(hasattr(torch.library, name) for name in ('custom_op', 'triton_op', 'wrap_triton'))
    )


def require_supported(supported: bool, operation: str, requirements: str) -> None:
    if not supported:
        raise CustomKernelError(f'{operation} custom kernel cannot handle these inputs. {requirements} {DISABLE_HINT}')


def kernel_errors(operation: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Preserve the original error as the cause; never retry with a native implementation.

    Only use for loading and opaque operation bodies. Compiled public entry points use their own try/except:
    wrapping different kernels in this shared code object would make them compete for Dynamo's cache limit.
    """
    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        @wraps(function)
        def run(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                return function(*args, **kwargs)
            except CustomKernelError:
                raise
            except Exception as error:
                raise CustomKernelError(f'{operation} custom kernel failed: {error}. {DISABLE_HINT}') from error
        return run
    return decorate


def _require_runtime() -> None:
    if not kernels_available():
        raise CustomKernelError(
            'Custom kernels require a CUDA-enabled PyTorch build, Triton, and torch.library custom_op/triton_op/wrap_triton. '
            + DISABLE_HINT
        )


@kernel_errors('MLP loading')
def load_mlp() -> MLPKernel:
    _require_runtime()
    from .mlp import mlp_projection
    return mlp_projection


@kernel_errors('LM-head loading')
def load_head() -> HeadKernel:
    _require_runtime()
    if not hasattr(torch.ops.aten.addmm, 'dtype_out'):
        raise CustomKernelError('LM-head custom kernel requires PyTorch addmm with FP32 output accumulation. ' + DISABLE_HINT)
    from .lm_head import fused_linear_cross_entropy
    return fused_linear_cross_entropy


@kernel_errors('RoPE loading')
def load_rope() -> RoPEKernel:
    _require_runtime()
    from .rope import qkv_bias_rope
    return qkv_bias_rope
