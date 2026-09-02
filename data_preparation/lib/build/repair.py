# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""The repair step of ``prepare()``: one pass over every source folder, one report, one confirmation.

Before anything is downloaded or built, :func:`repair_broken_and_stale_folders` looks at the ``raw/`` and
``processed/`` folder of every source the config uses and decides what has to go:

* **raw** (downloaded, expensive): a *stale* folder (its manifest hash differs from :meth:`DatasetConfig.raw_hash`,
  so the source identity or the tokenizer changed) or an *outdated* one (stored with a smaller ``max_seq_length``
  than the config asks for now) is **deleted and downloaded again — after the user confirmed**. A folder with a
  *broken* shard (missing, unreadable, wrong row count) is truncated to its good prefix with
  :meth:`RawFolder.truncate_to_good_prefix` (the next download resumes there, with the offset and the reject
  counters the last kept shard recorded) — unconfirmed when only the broken shard itself is dropped, but when
  healthy shards after the broken one would be discarded too, the truncation joins the same one confirmation as
  the deletions (they are downloaded rows lost for a repair, re-downloaded next run); when no prefix can be kept
  the folder is queued for the confirmed deletion. Shards without a manifest are an error: nothing says where
  those rows came from, and guessing would either delete data or resume from the wrong offset.
* **processed** (derived, cheap): judged by the shared verdict (``lib/build/assessment.py``), which attaches the
  cheapest repair — and this step performs exactly that repair, never more. A rebuild is a deletion without
  confirmation: stale, broken, without a manifest, unlisted stray shards, built from raw shards that no longer
  exist (its ``extra["input_shards"]`` is not a prefix of the raw shard list — e.g. after a truncation), or its
  raw folder is being deleted. The one exception is the crash leftover of an interrupted per-shard build — a single
  unlisted file that is exactly the next shard the resumed build writes: it is left alone (the build overwrites
  it). The rename-aside swap of an all-at-once build (``lib/stages/build.py:_swap_into_place``) can be interrupted
  too: a **complete** ``processed/<name>.tmp`` (its own manifest is current and every shard verifies) next to a
  *missing* processed folder is the swap's data — it is renamed into place instead of deleted; an incomplete
  ``.tmp`` is removed as the leftover of an interrupted build, and a leftover ``processed/<name>.old`` (the folder
  the swap already replaced) is removed without asking.

Nothing is touched until every folder was inspected; the queued raw deletions and healthy-shard-dropping
truncations are then confirmed **once** with one list, and only then is anything deleted or truncated.
``dry_run=True`` (``prepare.py status``) records what would be done and touches nothing. A refused or impossible
confirmation raises :class:`ConfirmationRequired` with the same list and **nothing** is changed — not even the
unconfirmed repairs; ``prepare.py`` prints it and exits 2; ``train.py``'s auto-prepare never prompts and never
deletes or truncates raw.
"""

from __future__ import annotations

import shutil
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from data_preparation.dataset_config import DatasetConfig
from data_preparation.layout import DatasetLayout
from data_preparation.lib.build.assessment import ShardList, assess_processed_folder
from data_preparation.lib.log import get_logger
from data_preparation.lib.ui.dashboard import suspended
from data_preparation.lib.storage.manifest import shard_list, Manifest, has_shards, shard_problem
from data_preparation.lib.storage.raw_folder import RawFolder

log = get_logger(__name__)

FolderKind = Literal["raw", "processed"]
RepairVerb = Literal["delete", "truncate", "swap", "would_delete", "would_truncate", "would_swap"]
Confirm = Callable[[str], bool]

CONFIRMATION_HEADER = "The following raw folders will be deleted or truncated, the dropped rows downloaded again:"
CONFIRMATION_QUESTION = "Continue? [y/N] "
YES_ANSWERS = ("y", "yes")


class RepairError(RuntimeError):
    """The repair step cannot decide safely (e.g. raw shards without a manifest) or was not allowed to proceed."""


class ConfirmationRequired(RepairError):
    """Raw folders have to be deleted or truncated past healthy shards but the user did not confirm — no terminal
    to ask on, or the answer was not yes. Nothing was changed. ``message`` is the confirmation prompt (the list of
    folders and why), ``report`` says what would have been done."""

    def __init__(self, report: RepairReport, message: str, *, interactive: bool) -> None:
        self.report = report
        self.message = message
        self.interactive = interactive
        hint = "not confirmed, nothing was changed" if interactive else "no terminal to ask on; rerun with --yes (prepare.py) to confirm, nothing was changed"
        super().__init__(f"{message.rstrip()}\n{hint}")


@dataclass(frozen=True)
class RepairAction:
    """One thing the repair step did (``delete`` / ``truncate`` / ``swap``) or would do (``would_*``) to one folder."""

    source: str
    folder: Path
    kind: FolderKind
    action: RepairVerb
    reason: str
    needs_confirmation: bool = False  # a truncation dropping healthy shards after the broken one; deletions of raw always ask

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

    def raw_confirmations_planned(self) -> list[RepairAction]:
        """The raw actions the one confirmation covers: every queued deletion, and every truncation that would drop
        healthy shards after the broken one (a tail-only truncation repairs without asking)."""
        return [
            action
            for action in self.actions
            if action.kind == "raw" and (action.action in ("delete", "would_delete") or action.needs_confirmation)
        ]

    def as_planned(self) -> RepairReport:
        """The same actions with every verb in its ``would_*`` form (nothing was performed)."""
        return RepairReport([replace(action, action=_planned_verb(action.action)) for action in self.actions])


def _planned_verb(verb: RepairVerb) -> RepairVerb:
    if verb == "delete":
        return "would_delete"
    if verb == "truncate":
        return "would_truncate"
    if verb == "swap":
        return "would_swap"
    return verb


# --- the step ------------------------------------------------------------------------------------------------------------


def repair_broken_and_stale_folders(
    config: DatasetConfig,
    layout: DatasetLayout,
    *,
    assume_yes: bool,
    dry_run: bool = False,
    confirm: Confirm | None = None,
    sources: Iterable[str] | None = None,
) -> RepairReport:
    """Inspect the raw and processed folder of every source in ``config`` (or of ``sources``), confirm the raw
    deletions and healthy-shard-dropping truncations once, perform everything (see the module docstring) and
    return what was done.

    ``assume_yes`` skips the prompt; otherwise ``confirm(message)`` decides when given, else the question is put on
    stdin when it is a terminal. Without a terminal, or on an answer other than yes, :class:`ConfirmationRequired`
    is raised and nothing is changed. ``dry_run`` inspects only and reports ``would_*`` actions without raising.
    Raises :class:`RepairError` for a raw folder that holds shards but no manifest.
    """
    planned = RepairReport()
    for name in config.sources if sources is None else sources:
        raw_shards = inspect_raw_folder(config, name, layout.raw_dir(name), planned)
        inspect_processed_folder(config, name, layout.processed_dir(name), raw_shards, planned)
        inspect_swap_leftovers(config, name, layout.processed_dir(name), raw_shards, planned)
    if dry_run:
        report = planned.as_planned()
        log.info("repair (dry run):\n%s", report.describe())
        return report
    queued = planned.raw_confirmations_planned()
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
        return shard_list(manifest.shards)
    kept = manifest.shards[:good]
    if good == 0 or kept[-1].offset is None:
        _plan(report, name, folder, "raw", "delete", f"broken: {problem}")
        return None
    dropped = [shard.name for shard in manifest.shards[good:]]
    healthy = len(dropped) - 1  # everything after the broken shard itself verified fine (or was never reached)
    reason = f"broken: {problem}; dropping {len(dropped)} shard(s) {dropped[0]}..{dropped[-1]}, keeping {good}"
    if healthy > 0:
        reason += f" — {healthy} healthy shard(s) after the broken one are discarded and re-downloaded next run"
    _plan(report, name, folder, "raw", "truncate", reason, needs_confirmation=healthy > 0)
    return shard_list(kept)


def inspect_processed_folder(config: DatasetConfig, name: str, folder: Path, raw_shards: ShardList | None, report: RepairReport) -> None:
    """Plan what happens to the processed folder of ``name`` given the raw shards it will be able to build from
    (None: the raw folder is being deleted): the shared verdict of
    :func:`~data_preparation.lib.build.assessment.assess_processed_folder` decides, and this step performs exactly
    the cheapest repair the verdict attaches — a ``rebuild`` is a deletion (derived data needs no confirmation, the
    build writes the folder again), everything else is left alone. In particular the single crash leftover of an
    interrupted per-shard build (an unlisted file that is exactly the next shard the resumed build writes) is no
    longer treated as corruption: the build overwrites it, so deleting the whole folder would redo the entire
    cleaning for one file."""
    assessment = assess_processed_folder(config, name, folder, raw_shards)
    if assessment.repair == "rebuild":
        _plan(report, name, folder, "processed", "delete", assessment.reason)
    elif assessment.problem == "crash_leftover":
        log.info("%s: leaving %s alone (%s)", name, folder, assessment.reason)


def inspect_swap_leftovers(config: DatasetConfig, name: str, processed_dir: Path, raw_shards: ShardList | None, report: RepairReport) -> None:
    """Plan the cleanup after an interrupted rename-aside swap of an all-at-once build
    (``lib/stages/build.py:_swap_into_place``): a **complete** ``processed/<name>.tmp`` (the shared verdict on the
    folder itself is ``ok`` — current manifest, every shard verifies) next to a missing processed folder is the
    swap's data and is renamed into place; any other leftover ``.tmp`` is removed as an interrupted build's; a
    leftover ``processed/<name>.old`` (the folder a swap already replaced) is removed without asking."""
    temporary = processed_dir.with_name(processed_dir.name + ".tmp")
    if temporary.exists():
        if not processed_dir.exists() and assess_processed_folder(config, name, temporary, raw_shards).verdict == "ok":
            _plan(report, name, temporary, "processed", "swap", "complete build of an interrupted swap; renaming it into place")
        else:
            _plan(report, name, temporary, "processed", "delete", "leftover of an interrupted all-at-once build")
    old = processed_dir.with_name(processed_dir.name + ".old")
    if old.exists():
        _plan(report, name, old, "processed", "delete", "leftover of a completed folder swap")


def _good_prefix_length(folder: Path, manifest: Manifest) -> tuple[int, str | None]:
    """How many leading shards of ``manifest`` verify against ``folder``, and the first problem (None if all do)."""
    for index, shard in enumerate(manifest.shards):
        problem = shard_problem(folder, shard)
        if problem is not None:
            return index, problem
    return len(manifest.shards), None


def _plan(report: RepairReport, source: str, folder: Path, kind: FolderKind, action: RepairVerb, reason: str, *, needs_confirmation: bool = False) -> None:
    report.actions.append(RepairAction(source=source, folder=folder, kind=kind, action=action, reason=reason, needs_confirmation=needs_confirmation))


# --- confirmation ----------------------------------------------------------------------------------------------------------


def confirmation_message(queued: list[RepairAction]) -> str:
    """The one prompt for every queued raw deletion and healthy-shard-dropping truncation: header,
    ``  <name>: <reason>`` per folder, the question."""
    lines = [CONFIRMATION_HEADER, *(f"  {action.source}: {action.reason}" for action in queued), CONFIRMATION_QUESTION]
    return "\n".join(lines)


def confirm_raw_deletions(queued: list[RepairAction], planned: RepairReport, *, assume_yes: bool, confirm: Confirm | None) -> None:
    """Ask once for all ``queued`` raw deletions and truncations; return when they may proceed, raise
    :class:`ConfirmationRequired` (carrying the planned report) otherwise. ``assume_yes`` answers without asking,
    ``confirm`` replaces the terminal prompt, and without either the question is put on stdin only when it is a
    terminal."""
    if assume_yes:
        log.warning("repairing %d raw folder(s) without asking (assume_yes): %s", len(queued), ", ".join(action.source for action in queued))
        return
    message = confirmation_message(queued)
    if confirm is not None:
        answered_yes = confirm(message)
    elif sys.stdin is not None and sys.stdin.isatty():
        with suspended():  # the live dashboard is cleared while the question is on the terminal
            answered_yes = input(message).strip().lower() in YES_ANSWERS
    else:
        raise ConfirmationRequired(planned.as_planned(), message, interactive=False)
    if not answered_yes:
        raise ConfirmationRequired(planned.as_planned(), message, interactive=True)


# --- performing ------------------------------------------------------------------------------------------------------------


def perform_repairs(report: RepairReport) -> None:
    """Carry out every planned action of ``report`` in order: processed folders first (deletions and the swap of a
    complete ``.tmp`` into place — so a crash never leaves derived data next to a raw folder it no longer matches),
    then raw truncations and deletions."""
    processed = [action for action in report.actions if action.kind == "processed"]
    raw = [action for action in report.actions if action.kind == "raw"]
    for action in processed + raw:
        if action.action == "delete":
            log.warning("%s: deleting %s (%s)", action.source, action.folder, action.reason)
            shutil.rmtree(action.folder)
        elif action.action == "truncate":
            log.warning("%s: truncating %s (%s)", action.source, action.folder, action.reason)
            _truncate_raw(action)
        elif action.action == "swap":
            log.warning("%s: renaming %s into place (%s)", action.source, action.folder, action.reason)
            action.folder.rename(action.folder.with_name(action.folder.name.removesuffix(".tmp")))
        else:
            raise RepairError(f"{action.source}: cannot perform a planned-only action {action.action!r} on {action.folder}")


def _truncate_raw(action: RepairAction) -> None:
    manifest = Manifest.load(action.folder)
    if manifest is None or not RawFolder(action.folder, manifest).truncate_to_good_prefix():
        raise RepairError(f"{action.source}: {action.folder} changed while repairing; could not truncate to its good prefix")
