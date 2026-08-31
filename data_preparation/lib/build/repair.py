# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""The repair step of ``prepare()``: one pass over every source folder, one report, one confirmation.

Before anything is downloaded or built, :func:`repair_broken_and_stale_folders` looks at the ``raw/`` and
``processed/`` folder of every source the config uses and decides what has to go:

* **raw** (downloaded, expensive): a *stale* folder (its manifest hash differs from :meth:`DatasetConfig.raw_hash`,
  so the source identity or the tokenizer changed) or an *outdated* one (stored with a smaller ``max_seq_length``
  than the config asks for now) is **deleted and downloaded again — after the user confirmed**. A folder with a
  *broken* shard (missing, unreadable, wrong row count) is truncated to its good prefix with
  :func:`truncate_raw_to_good_prefix` (the next download resumes there); when no prefix can be kept it is queued for
  the same confirmed deletion. Shards without a manifest are an error: nothing says where those rows came from, and
  guessing would either delete data or resume from the wrong offset.
* **processed** (derived, cheap): deleted without confirmation when stale, broken, without a manifest, built from raw
  shards that no longer exist (its ``extra["input_shards"]`` is not a prefix of the raw shard list — e.g. after a
  truncation), or when its raw folder is being deleted. A leftover ``processed/<name>.tmp`` of an interrupted
  all-at-once build is removed too.

Nothing is touched until every folder was inspected; the queued raw deletions are then confirmed **once** with one
list, and only then is anything deleted or truncated. ``dry_run=True`` (``prepare.py status``) records what would be
done and touches nothing. A refused or impossible confirmation raises :class:`ConfirmationRequired` with the same
list — ``prepare.py`` prints it and exits 2; ``train.py``'s auto-prepare never prompts and never deletes raw.
"""

from __future__ import annotations

import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from data_preparation.dataset_config import DatasetConfig
from data_preparation.layout import DatasetLayout
from data_preparation.lib.log import get_logger
from data_preparation.lib.stages.download import truncate_raw_to_good_prefix
from data_preparation.lib.storage.manifest import Manifest, has_shards, shard_problem

log = get_logger(__name__)

FolderKind = Literal["raw", "processed"]
RepairVerb = Literal["delete", "truncate", "would_delete", "would_truncate"]
ShardList = list[list[Any]]  # ``[[shard name, rows], ...]`` — the shape of ``processed`` manifests' ``extra["input_shards"]``
Confirm = Callable[[str], bool]

CONFIRMATION_HEADER = "The following raw folders will be deleted and downloaded again:"
CONFIRMATION_QUESTION = "Continue? [y/N] "
YES_ANSWERS = ("y", "yes")


class RepairError(RuntimeError):
    """The repair step cannot decide safely (e.g. raw shards without a manifest) or was not allowed to proceed."""


class ConfirmationRequired(RepairError):
    """Raw folders have to be deleted but the user did not confirm — no terminal to ask on, or the answer was not
    yes. Nothing was changed. ``message`` is the confirmation prompt (the list of folders and why), ``report`` says
    what would have been done."""

    def __init__(self, report: RepairReport, message: str, *, interactive: bool) -> None:
        self.report = report
        self.message = message
        self.interactive = interactive
        hint = "not confirmed, nothing was changed" if interactive else "no terminal to ask on; rerun with --yes (prepare.py) to confirm, nothing was changed"
        super().__init__(f"{message.rstrip()}\n{hint}")


@dataclass(frozen=True)
class RepairAction:
    """One thing the repair step did (``delete`` / ``truncate``) or would do (``would_*``) to one folder."""

    source: str
    folder: Path
    kind: FolderKind
    action: RepairVerb
    reason: str

    def describe(self) -> str:
        verb = self.action.replace("_", " ")
        return f"{verb} {self.kind} {self.folder} ({self.source}): {self.reason}"


@dataclass
class RepairReport:
    """Every action of one repair pass, in inspection order (raw before processed, source by source)."""

    actions: list[RepairAction] = field(default_factory=list)

    def describe(self) -> str:
        """One line per action; ``"nothing to repair"`` when there is none."""
        if not self.actions:
            return "nothing to repair"
        return "\n".join(action.describe() for action in self.actions)

    def raw_deleted(self) -> list[RepairAction]:
        """Raw folders that were deleted (performed, not planned)."""
        return [action for action in self.actions if action.kind == "raw" and action.action == "delete"]

    def processed_deleted(self) -> list[RepairAction]:
        """Processed folders that were deleted (performed, not planned)."""
        return [action for action in self.actions if action.kind == "processed" and action.action == "delete"]

    def raw_deletions_planned(self) -> list[RepairAction]:
        """Raw folders queued for deletion (before they are confirmed, or in a dry run)."""
        return [action for action in self.actions if action.kind == "raw" and action.action in ("delete", "would_delete")]

    def as_planned(self) -> RepairReport:
        """The same actions with every verb in its ``would_*`` form (nothing was performed)."""
        return RepairReport([replace(action, action=_planned_verb(action.action)) for action in self.actions])


def _planned_verb(verb: RepairVerb) -> RepairVerb:
    if verb == "delete":
        return "would_delete"
    if verb == "truncate":
        return "would_truncate"
    return verb


# --- the step ------------------------------------------------------------------------------------------------------------


def repair_broken_and_stale_folders(
    config: DatasetConfig,
    layout: DatasetLayout,
    *,
    assume_yes: bool,
    dry_run: bool = False,
    confirm: Confirm | None = None,
) -> RepairReport:
    """Inspect the raw and processed folder of every source in ``config``, confirm the raw deletions once, perform
    everything (see the module docstring) and return what was done.

    ``assume_yes`` skips the prompt; otherwise ``confirm(message)`` decides when given, else the question is put on
    stdin when it is a terminal. Without a terminal, or on an answer other than yes, :class:`ConfirmationRequired`
    is raised and nothing is changed. ``dry_run`` inspects only and reports ``would_*`` actions without raising.
    Raises :class:`RepairError` for a raw folder that holds shards but no manifest.
    """
    planned = RepairReport()
    for name in config.sources:
        raw_shards = inspect_raw_folder(config, name, layout.raw_dir(name), planned)
        inspect_processed_folder(config, name, layout.processed_dir(name), raw_shards, planned)
        inspect_leftover_temporary_folder(name, layout.processed_dir(name), planned)
    if dry_run:
        report = planned.as_planned()
        log.info("repair (dry run):\n%s", report.describe())
        return report
    queued = planned.raw_deletions_planned()
    if queued:
        confirm_raw_deletions(queued, planned, assume_yes=assume_yes, confirm=confirm)
    perform_repairs(planned)
    return planned


# --- inspection (read-only) ----------------------------------------------------------------------------------------------


def inspect_raw_folder(config: DatasetConfig, name: str, folder: Path, report: RepairReport) -> ShardList | None:
    """Plan what happens to the raw folder of ``name`` and return the shards it will hold afterwards as
    ``[[name, rows], ...]`` (empty when there is no folder), or None when the folder is queued for deletion."""
    manifest = Manifest.load(folder)
    if manifest is None:
        if has_shards(folder):
            raise RepairError(f"{name}: {folder} holds shards but no manifest; refusing to guess where the rows came from — delete the directory to download the source again")
        return []
    if not manifest.is_current(config.raw_hash(name)):
        _plan(report, name, folder, "raw", "delete", "stale: source identity or tokenizer changed")
        return None
    if manifest.is_outdated(config.max_seq_length):
        _plan(report, name, folder, "raw", "delete", f"outdated: max_seq_length {manifest.truncated_at_tokens} -> {config.max_seq_length}")
        return None
    good, problem = _good_prefix_length(folder, manifest)
    if problem is None:
        return _shard_list(manifest)
    kept = manifest.shards[:good]
    if good == 0 or kept[-1].offset is None:
        _plan(report, name, folder, "raw", "delete", f"broken: {problem}")
        return None
    dropped = [shard.name for shard in manifest.shards[good:]]
    _plan(report, name, folder, "raw", "truncate", f"broken: {problem}; dropping {len(dropped)} shard(s) {dropped[0]}..{dropped[-1]}, keeping {good}")
    return [[shard.name, shard.rows] for shard in kept]


def inspect_processed_folder(config: DatasetConfig, name: str, folder: Path, raw_shards: ShardList | None, report: RepairReport) -> None:
    """Plan what happens to the processed folder of ``name`` given the raw shards it will be able to build from
    (None: the raw folder is being deleted). Every problem is a deletion; derived data needs no confirmation."""
    manifest = Manifest.load(folder)
    if manifest is None:
        if has_shards(folder):
            _plan(report, name, folder, "processed", "delete", "no manifest")
        return
    if raw_shards is None:
        _plan(report, name, folder, "processed", "delete", "built from a raw folder that is being deleted")
        return
    if not manifest.is_current(config.processed_hash(name)):
        _plan(report, name, folder, "processed", "delete", "stale: processing settings, max_seq_length or the source changed")
        return
    _, problem = _good_prefix_length(folder, manifest)
    if problem is not None:
        _plan(report, name, folder, "processed", "delete", f"broken: {problem}")
        return
    covered: ShardList = manifest.extra.get("input_shards", [])
    if raw_shards[: len(covered)] != covered:
        _plan(report, name, folder, "processed", "delete", "built from raw shards that no longer exist")


def inspect_leftover_temporary_folder(name: str, processed_dir: Path, report: RepairReport) -> None:
    """Plan the removal of ``processed/<name>.tmp`` left behind by an interrupted all-at-once build."""
    temporary = processed_dir.with_name(processed_dir.name + ".tmp")
    if temporary.exists():
        _plan(report, name, temporary, "processed", "delete", "leftover of an interrupted all-at-once build")


def _good_prefix_length(folder: Path, manifest: Manifest) -> tuple[int, str | None]:
    """How many leading shards of ``manifest`` verify against ``folder``, and the first problem (None if all do)."""
    for index, shard in enumerate(manifest.shards):
        problem = shard_problem(folder, shard)
        if problem is not None:
            return index, problem
    return len(manifest.shards), None


def _shard_list(manifest: Manifest) -> ShardList:
    return [[shard.name, shard.rows] for shard in manifest.shards]


def _plan(report: RepairReport, source: str, folder: Path, kind: FolderKind, action: RepairVerb, reason: str) -> None:
    report.actions.append(RepairAction(source=source, folder=folder, kind=kind, action=action, reason=reason))


# --- confirmation ----------------------------------------------------------------------------------------------------------


def confirmation_message(queued: list[RepairAction]) -> str:
    """The one prompt for every queued raw deletion: header, ``  <name>: <reason>`` per folder, the question."""
    lines = [CONFIRMATION_HEADER, *(f"  {action.source}: {action.reason}" for action in queued), CONFIRMATION_QUESTION]
    return "\n".join(lines)


def confirm_raw_deletions(queued: list[RepairAction], planned: RepairReport, *, assume_yes: bool, confirm: Confirm | None) -> None:
    """Ask once for all ``queued`` raw deletions; return when they may proceed, raise :class:`ConfirmationRequired`
    (carrying the planned report) otherwise. ``assume_yes`` answers without asking, ``confirm`` replaces the
    terminal prompt, and without either the question is put on stdin only when it is a terminal."""
    if assume_yes:
        log.warning("deleting %d raw folder(s) without asking (assume_yes): %s", len(queued), ", ".join(action.source for action in queued))
        return
    message = confirmation_message(queued)
    if confirm is not None:
        answered_yes = confirm(message)
    elif sys.stdin is not None and sys.stdin.isatty():
        answered_yes = input(message).strip().lower() in YES_ANSWERS
    else:
        raise ConfirmationRequired(planned.as_planned(), message, interactive=False)
    if not answered_yes:
        raise ConfirmationRequired(planned.as_planned(), message, interactive=True)


# --- performing ------------------------------------------------------------------------------------------------------------


def perform_repairs(report: RepairReport) -> None:
    """Carry out every planned action of ``report`` in order: processed folders first (so a crash never leaves
    derived data next to a raw folder it no longer matches), then raw truncations and deletions."""
    processed = [action for action in report.actions if action.kind == "processed"]
    raw = [action for action in report.actions if action.kind == "raw"]
    for action in processed + raw:
        if action.action == "delete":
            log.warning("%s: deleting %s (%s)", action.source, action.folder, action.reason)
            shutil.rmtree(action.folder)
        elif action.action == "truncate":
            log.warning("%s: truncating %s (%s)", action.source, action.folder, action.reason)
            _truncate_raw(action)
        else:
            raise RepairError(f"{action.source}: cannot perform a planned-only action {action.action!r} on {action.folder}")


def _truncate_raw(action: RepairAction) -> None:
    manifest = Manifest.load(action.folder)
    if manifest is None or not truncate_raw_to_good_prefix(action.folder, manifest):
        raise RepairError(f"{action.source}: {action.folder} changed while repairing; could not truncate to its good prefix")


__all__ = [
    "CONFIRMATION_HEADER",
    "CONFIRMATION_QUESTION",
    "ConfirmationRequired",
    "RepairAction",
    "RepairError",
    "RepairReport",
    "confirm_raw_deletions",
    "confirmation_message",
    "inspect_leftover_temporary_folder",
    "inspect_processed_folder",
    "inspect_raw_folder",
    "perform_repairs",
    "repair_broken_and_stale_folders",
]
