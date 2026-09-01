# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Training data pipeline: parquet streaming, on-the-fly tokenization, collation, the per-source train loaders and
the per-stage validation loaders.

Deliberately NO re-exports: `dataset_resolver` is framework-neutral, and a convenience import of the torch
modules here would load torch for everyone importing it (enforced by
`test_dataset_resolver.py::test_framework_neutral_modules_do_not_load_torch`). Import the submodules directly.
"""
