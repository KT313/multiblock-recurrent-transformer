# Copyright (c) Meta Platforms, Inc. and affiliates. BSD 3-Clause, see LICENSE in this directory.
"""
torchao's 8-bit optimizer-state tensor, vendored: see README.md in this directory for the source commit and the trims.
"""

from .subclass_8bit import OptimState8bit

__all__ = ["OptimState8bit"]
