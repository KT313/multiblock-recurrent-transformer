# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Read-only readiness of managed tokenizer folders, without importing a tokenizer at planning time."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data_preparation.lib.storage.manifest import Manifest

REQUIRED_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")


@dataclass(frozen=True)
class TokenizerAssessment:
    manifest: Manifest | None
    problem: str | None = None

    @property
    def ready(self) -> bool:
        return self.problem is None


def tokenizer_files_problem(directory: Path) -> str | None:
    """The supported saved-directory contract; optional special-token maps are not required."""
    missing = [name for name in REQUIRED_TOKENIZER_FILES if not (directory / name).is_file()]
    return f"no {', '.join(missing)} in {directory}" if missing else None


def assess_tokenizer_folder(directory: Path, expected_hash: str, *, validate_payload: bool = False) -> TokenizerAssessment:
    """Check metadata/files cheaply; preparation can additionally load-check once at its stage boundary.

    No validation cache: generation IDs attest managed publication, not manual payload edits. Status checks
    existence only, and deliberately does not deserialize potentially large tokenizer vocabularies.
    """
    from data_preparation.lib.storage.manifest import Manifest

    manifest = Manifest.load(directory)
    if manifest is None:
        return TokenizerAssessment(None, f"missing or unreadable tokenizer manifest in {directory}")
    if manifest.stage != "tokenizer":
        return TokenizerAssessment(manifest, f"manifest stage {manifest.stage!r} is not 'tokenizer' in {directory}")
    if not manifest.is_current(expected_hash):
        return TokenizerAssessment(manifest, f"stale tokenizer manifest in {directory}")
    if not manifest.generation_complete:
        return TokenizerAssessment(manifest, f"incomplete tokenizer generation in {directory}")
    problem = tokenizer_files_problem(directory)
    if problem is None and validate_payload:
        from data_preparation.lib.stages.tokenizer_loader import SavedTokenizer

        try:
            SavedTokenizer(directory)
        except Exception as error:  # the Rust tokenizer parser also raises plain Exception
            problem = f"unusable tokenizer in {directory}: {type(error).__name__}: {error}"
    return TokenizerAssessment(manifest, problem)
