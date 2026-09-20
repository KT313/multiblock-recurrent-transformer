# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
#
# Vendored from torchao/utils.py (pytorch/ao 6792133): `_implements` and `_dispatch__torch_dispatch__` as the methods
# of a minimal base class. Upstream's `TorchAOBaseTensor` carries far more (torch-function tables, layout registries,
# generic flatten/unflatten); the 8-bit state tensor needs only the aten-op table below.
import functools

import torch


class TorchAOBaseTensor(torch.Tensor):
    """
    A tensor subclass whose aten ops are the functions registered with `implements`; any other op raises.
    """

    @classmethod
    def implements(cls, aten_ops):
        """Decorator for implementing aten ops like `torch.ops.aten.linear.default` for
        tensor subclass, the implemented functions are called in ``__torch_dispatch__`` callback
        for ``torch.Tensor`` subclasses
        """
        if not hasattr(cls, "_ATEN_OP_TABLE"):
            cls._ATEN_OP_TABLE = {}
        if cls not in cls._ATEN_OP_TABLE:
            cls._ATEN_OP_TABLE[cls] = {}
        if not isinstance(aten_ops, (list, tuple)):
            aten_ops = [aten_ops]

        def decorator(func):
            for op in aten_ops:

                @functools.wraps(func)
                def wrapper(f, types, args, kwargs, _func=func):
                    return _func(f, types, args, kwargs)

                cls._ATEN_OP_TABLE[cls][op] = wrapper
            return func

        return decorator

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        if (
            hasattr(cls, "_ATEN_OP_TABLE")
            and cls in cls._ATEN_OP_TABLE
            and func in cls._ATEN_OP_TABLE[cls]
        ):
            return cls._ATEN_OP_TABLE[cls][func](func, types, args, kwargs)

        arg_types = tuple(type(arg) for arg in args)
        kwarg_types = {k: type(arg) for k, arg in kwargs.items()}
        raise NotImplementedError(
            f"{cls.__name__} dispatch: attempting to run unimplemented operator/function: {func=}, {types=}, {arg_types=}, {kwarg_types=}"
        )

    __torch_function__ = torch._C._disabled_torch_function_impl
