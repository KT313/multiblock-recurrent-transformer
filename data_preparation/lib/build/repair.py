# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
The repair step of prepare(): one pass over every source folder, one report, one confirmation.

Before anything is downloaded or built, :func:`repair_broken_and_stale_folders` inspects the raw/ and
processed/ folder of every source the config uses and decides what has to go:

* raw (downloaded, expensive): a *stale* folder (manifest hash differs from :meth:`DatasetConfig.raw_hash`; the
  reason lists the changed fields) or an *outdated* one (stored with a smaller dataset_max_sequence_length than the
  config asks for) is deleted and downloaded again, after the user confirmed. A *tokenizer_changed* folder (its
  token counts were made with another tokenizer or token_count than the config's, the rows are the same) is
  *adopted* after the user confirmed: its manifest is re-labelled with the config's tokenizer and token_count
  (the change is logged under `tokenizer_changes` in the manifest) and the rows are kept, at the cost that their
  stored token counts and the truncation of the pretrain texts do not match the new tokenizer. A folder with a
  *broken* shard (missing, unreadable, wrong row count) is truncated to its good prefix
  (:func:`good_prefix_length`, :meth:`RawFolder.truncate_to`); the next download resumes there. Dropping only the
  broken tail needs no confirmation; dropping healthy shards after it joins the one confirmation, and when no
  prefix can be kept the folder is queued for deletion. Shards without a manifest are an error: nothing says where
  those rows came from. A manifest that cannot be parsed next to shards is reported and left alone (the rows may
  have been expensive; the user fixes or deletes the folder by hand).
* processed (derived, cheap): the shared verdict (lib/build/assessment.py) attaches the cheapest repair and
  this step performs exactly that. A rebuild is a deletion; a *stale* folder (the config's processed fingerprint
  changed; the reason lists the changed fields) and a manifest that cannot be parsed join the one confirmation,
  every other rebuild (broken or stray shards, no manifest, raw shards gone or being deleted) goes without asking.
  A crash leftover (one unlisted file that is exactly the next shard the resumed build writes) is left alone.
  Leftovers of an interrupted rename-aside swap (lib/stages/build.py:_swap_into_place): a complete
  processed/<name>.tmp next to a missing processed folder is renamed into place, an incomplete .tmp and a
  processed/<name>.old are removed without asking.

Nothing is touched until every folder was inspected; the queued raw deletions and adoptions, healthy-shard-dropping
truncations and stale / unparsable-manifest processed deletions are then confirmed once with one list. dry_run=True
(prepare.py status) records what would be done and touches nothing. A refused or impossible confirmation raises
:class:`ConfirmationRequired` with the same list and nothing is changed, not even the unconfirmed repairs;
prepare.py prints it and exits 2; train.py's auto-prepare never prompts, so it refuses a processed rebuild as well
as a raw deletion and prints the `--yes` command.

Raw folders are keyed by source name and shared by every dataset config, so a config that gives a name another
identity (revision, text field, split) sees the other config's folder as stale. Its raw manifest records the
file name of the config it was downloaded under (`Manifest.dataset_config`); a queued deletion of a folder written
under another config is *foreign* and needs allow_foreign_raw (prepare.py --allow_foreign_raw) before the
confirmation is even asked, `--yes` alone does not delete it.
"""

from __future__ import annotations

import shutil
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from data_preparation.dataset_config import DatasetConfig
from data_preparation.layout import DatasetLayout
from data_preparation.lib.build.assessment import ShardList, assess_processed_folder
from data_preparation.lib.log import get_logger
from data_preparation.lib.ui.dashboard import suspended
from data_preparation.lib.storage.manifest import shard_list, Manifest, has_shards
from data_preparation.lib.stages.download import RawInspection, inspect_raw, token_measure
from data_preparation.lib.storage.raw_folder import RawFolder, good_prefix_length

log = get_logger(__name__)

FolderKind = Literal["raw", "processed"]
RepairVerb = Literal["delete", "truncate", "adopt", "swap", "leave"]
Confirm = Callable[[str], bool]

CONFIRMATION_HEADER = (
    "The following folders will be deleted, truncated or re-labelled (raw: the dropped rows are downloaded again, "
    "an adopted folder keeps its rows under the new tokenizer; processed: rebuilt from raw):"
)
CONFIRMATION_QUESTION = "Continue? [y/N] "
YES_ANSWERS = ("y", "yes")
FOREIGN_HEADER = "The following raw folders were downloaded under another dataset config and would be deleted:"
FOREIGN_HINT = "rerun with --allow_foreign_raw (prepare.py) to let this config delete them, nothing was changed"


class RepairError(RuntimeError):
    """
    The repair step cannot decide safely (e.g. raw shards without a manifest) or was not allowed to proceed.
    """


class ConfirmationRequired(RepairError):
    """
    Raw folders have to be deleted or truncated past healthy shards but the user did not confirm: no terminal
    to ask on, or the answer was not yes, or (hint given) a raw folder of another dataset config is queued
    without allow_foreign_raw. Nothing was changed. message is the confirmation prompt (the list of folders and
    why), report says what would have been done.
    """

    def __init__(self, report: RepairReport, message: str, *, interactive: bool, hint: str | None = None) -> None:
        self.report = report
        self.message = message
        self.interactive = interactive
        if hint is None:
            hint = "not confirmed, nothing was changed" if interactive else "no terminal to ask on; rerun with --yes (prepare.py) to confirm, nothing was changed"
        super().__init__(f"{message.rstrip()}\n{hint}")


@dataclass(frozen=True)
class RepairAction:
    """
    One thing the repair step does (delete / truncate / adopt / swap, or leave: a raw folder it refuses to touch)
    to one folder; whether it was done is the report's :attr:`RepairReport.performed`.
    """

    source: str
    folder: Path
    kind: FolderKind
    action: RepairVerb
    reason: str
    needs_confirmation: bool = False  # joins the one confirmation (healthy-shard-dropping truncation, adoption, stale or unparsable processed manifest); raw deletions always ask
    keep_shards: int | None = None  # truncations: the good prefix the inspection found (`good_prefix_length`)
    foreign_config: str | None = None  # raw deletions: the dataset config the folder was downloaded under, when it is another one
    adopt: dict[str, Any] | None = None  # adoptions: the manifest's new token_count / tokenizer / tokenizer_hash (`token_measure`)

    def describe(self) -> str:
        return f"{self.action} {self.kind} {self.folder} ({self.source}): {self.reason}"


@dataclass
class RepairReport:
    """
    Every action of one repair pass, in inspection order (raw before processed, source by source), and whether
    they were carried out (performed; False for a dry run and for the report a refused confirmation carries).
    """

    actions: list[RepairAction] = field(default_factory=list)
    performed: bool = False

    def describe(self) -> str:
        """
        One line per action (would ... while not performed); "nothing to repair" when there is none.
        """

        if not self.actions:
            return "nothing to repair"
        prefix = "" if self.performed else "would "
        return "\n".join(prefix + action.describe() for action in self.actions)

    def confirmations_planned(self) -> list[RepairAction]:
        """
        The actions the one confirmation covers: every queued raw deletion and adoption, every truncation that
        would drop healthy shards after the broken one (a tail-only truncation repairs without asking), and the
        deletion of a stale processed folder or of one whose manifest cannot be parsed.
        """

        return [action for action in self.actions if (action.kind == "raw" and action.action == "delete") or action.needs_confirmation]

    def foreign_deletions_planned(self) -> list[RepairAction]:
        """
        The queued raw deletions of folders another dataset config downloaded (see the module docstring).
        """

        return [action for action in self.actions if action.foreign_config is not None]


# --- the step ------------------------------------------------------------------------------------------------------------


def repair_broken_and_stale_folders(
    config: DatasetConfig,
    layout: DatasetLayout,
    *,
    assume_yes: bool,
    dry_run: bool = False,
    confirm: Confirm | None = None,
    sources: Iterable[str] | None = None,
    config_name: str | None = None,
    allow_foreign_raw: bool = False,
) -> RepairReport:
    """
    Inspect the raw and processed folder of every source in config (or of sources), confirm the raw
    deletions and healthy-shard-dropping truncations once, perform everything (see the module docstring) and
    return what was done.

    assume_yes skips the prompt; otherwise confirm(message) decides when given, else the question is put on
    stdin when it is a terminal. Without a terminal, or on an answer other than yes, :class:`ConfirmationRequired`
    is raised and nothing is changed. config_name (the dataset config's file name) tells a raw folder another
    config downloaded: its deletion raises :class:`ConfirmationRequired` naming allow_foreign_raw unless that is
    given, before any prompt. dry_run inspects only and returns the plan (performed False) without raising.
    Raises :class:`RepairError` for a raw folder that holds shards but no manifest.
    """

    planned = RepairReport()
    for name in config.sources if sources is None else sources:
        inspect_source(config, name, layout, planned, config_name=config_name)
    if dry_run:
        return planned
    foreign = planned.foreign_deletions_planned()
    if foreign and not allow_foreign_raw:
        lines = [FOREIGN_HEADER, *(f"  {action.source}: {action.reason}" for action in foreign)]
        raise ConfirmationRequired(planned, "\n".join(lines), interactive=False, hint=FOREIGN_HINT)
    queued = planned.confirmations_planned()
    if queued:
        confirm_repairs(queued, planned, assume_yes=assume_yes, confirm=confirm)
    perform_repairs(planned)
    return planned


# --- inspection (read-only) ----------------------------------------------------------------------------------------------


def inspect_source(config: DatasetConfig, name: str, layout: DatasetLayout, report: RepairReport, *, config_name: str | None = None) -> None:
    """
    Plan the repairs of one source: the raw folder, then the processed folder and the swap leftovers against
    the raw shards that remain. A raw manifest nobody can parse is listed as left alone and ends the inspection:
    the processed folder is not judged against raw shards nobody knows.
    """

    inspection = inspect_raw(config, name, layout)
    if inspection.state == "unreadable":
        _plan(report, name, layout.raw_dir(name), "raw", "leave", inspection.reason)
        return
    raw_shards = inspect_raw_folder(config, name, layout, inspection, report, config_name=config_name)
    inspect_processed_folder(config, name, layout.processed_dir(name), raw_shards, report)
    inspect_swap_leftovers(config, name, layout.processed_dir(name), raw_shards, report)


def inspect_raw_folder(
    config: DatasetConfig, name: str, layout: DatasetLayout, inspection: RawInspection, report: RepairReport, *, config_name: str | None = None
) -> ShardList | None:
    """
    Plan what happens to the raw folder of name (inspection is its :func:`inspect_raw` state) and return the
    shards it will hold afterwards as [[name, rows], ...] (empty when there is no folder), or None when the
    folder is queued for deletion. A stale or outdated folder whose manifest names another dataset config than
    config_name (both known) is queued as foreign. A tokenizer_changed folder is queued for adoption (it asks)
    and then checked for broken shards like a current one, so one run heals both.
    """

    folder = layout.raw_dir(name)
    manifest = inspection.manifest
    if manifest is None:
        if has_shards(folder):
            raise RepairError(f"{name}: {folder} holds shards but no manifest; refusing to guess where the rows came from; delete the directory to download the source again")
        return []
    if inspection.state == "tokenizer_changed":
        _plan(report, name, folder, "raw", "adopt", inspection.reason, needs_confirmation=True, adopt=token_measure(config))
    elif inspection.state != "current":
        reason, foreign = inspection.reason, None
        if config_name is not None and manifest.dataset_config not in (None, config_name):
            foreign = manifest.dataset_config
            reason += f"; downloaded under dataset config {foreign}, deleting it needs --allow_foreign_raw"
        _plan(report, name, folder, "raw", "delete", reason, foreign_config=foreign)
        return None
    good, problem = good_prefix_length(folder, manifest)
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
        reason += f"; {healthy} healthy shard(s) after the broken one are discarded and re-downloaded next run"
    _plan(report, name, folder, "raw", "truncate", reason, needs_confirmation=healthy > 0, keep_shards=good)
    return shard_list(kept)


def inspect_processed_folder(config: DatasetConfig, name: str, folder: Path, raw_shards: ShardList | None, report: RepairReport) -> None:
    """
    Plan what happens to the processed folder of name given the raw shards it will be able to build from
    (None: the raw folder is being deleted). The shared verdict of
    :func:`~data_preparation.lib.build.assessment.assess_processed_folder` decides; this step performs exactly the
    cheapest repair it attaches: a rebuild is a deletion, everything else is left alone. Two deletions ask: a
    stale folder, because a fingerprint change that invalidates data is the user's to confirm (the reason lists the
    changed fields; every stale verdict asks, also another stage's manifest or an older column set, both odd
    enough to be worth a look), and a manifest that cannot be parsed, corruption worth a look first. Broken or
    stray shards, a missing manifest, raw shards that are gone or being deleted are repaired without asking. A
    crash leftover (the next shard the resumed build writes) is not corruption: the build overwrites it.
    """

    assessment = assess_processed_folder(config, name, folder, raw_shards)
    if assessment.repair == "rebuild":
        asks = assessment.problem in ("unreadable_manifest", "stale")
        _plan(report, name, folder, "processed", "delete", assessment.reason, needs_confirmation=asks)
    elif assessment.problem == "crash_leftover":
        log.info("%s: leaving %s alone (%s)", name, folder, assessment.reason)


def inspect_swap_leftovers(config: DatasetConfig, name: str, processed_dir: Path, raw_shards: ShardList | None, report: RepairReport) -> None:
    """
    Plan the cleanup after an interrupted rename-aside swap of an all-at-once build
    (lib/stages/build.py:_swap_into_place): a complete processed/<name>.tmp (verdict ok: current
    manifest, every shard verifies) next to a missing processed folder is the swap's data and is renamed into
    place; any other leftover .tmp is removed as an interrupted build's; a leftover processed/<name>.old
    (the folder a swap already replaced) is removed without asking.
    """

    temporary = processed_dir.with_name(processed_dir.name + ".tmp")
    if temporary.exists():
        if not processed_dir.exists() and assess_processed_folder(config, name, temporary, raw_shards).problem == "none":
            _plan(report, name, temporary, "processed", "swap", "complete build of an interrupted swap; renaming it into place")
        else:
            _plan(report, name, temporary, "processed", "delete", "leftover of an interrupted all-at-once build")
    old = processed_dir.with_name(processed_dir.name + ".old")
    if old.exists():
        _plan(report, name, old, "processed", "delete", "leftover of a completed folder swap")


def _plan(
    report: RepairReport, source: str, folder: Path, kind: FolderKind, action: RepairVerb, reason: str, *,
    needs_confirmation: bool = False, keep_shards: int | None = None, foreign_config: str | None = None, adopt: dict[str, Any] | None = None,
) -> None:  # fmt: skip
    report.actions.append(
        RepairAction(
            source=source, folder=folder, kind=kind, action=action, reason=reason, needs_confirmation=needs_confirmation,
            keep_shards=keep_shards, foreign_config=foreign_config, adopt=adopt,
        )
    )


# --- confirmation ----------------------------------------------------------------------------------------------------------


def confirmation_message(queued: list[RepairAction]) -> str:
    """
    The one prompt for every queued raw deletion and adoption, healthy-shard-dropping truncation and stale /
    unparsable processed deletion: header, <name>: <reason> per folder, the question.
    """

    lines = [CONFIRMATION_HEADER, *(f"  {action.source}: {action.reason}" for action in queued), CONFIRMATION_QUESTION]
    return "\n".join(lines)


def confirm_repairs(queued: list[RepairAction], planned: RepairReport, *, assume_yes: bool, confirm: Confirm | None) -> None:
    """
    Ask once for all queued repairs (:meth:`RepairReport.confirmations_planned`); return when they may proceed, raise
    :class:`ConfirmationRequired` (carrying the planned report) otherwise. assume_yes answers without asking,
    confirm replaces the terminal prompt, and without either the question is put on stdin only when it is a
    terminal.
    """

    if assume_yes:
        log.warning("repairing %d folder(s) without asking (assume_yes): %s", len(queued), ", ".join(action.source for action in queued))
        return
    message = confirmation_message(queued)
    if confirm is not None:
        answered_yes = confirm(message)
    elif sys.stdin is not None and sys.stdin.isatty():
        with suspended():  # the live dashboard is cleared while the question is on the terminal
            answered_yes = input(message).strip().lower() in YES_ANSWERS
    else:
        raise ConfirmationRequired(planned, message, interactive=False)
    if not answered_yes:
        raise ConfirmationRequired(planned, message, interactive=True)


# --- performing ------------------------------------------------------------------------------------------------------------


def perform_repairs(report: RepairReport) -> None:
    """
    Carry out every planned action of report in order: processed folders first (deletions and the swap of a
    complete .tmp into place, so a crash never leaves derived data next to a raw folder it no longer matches),
    then raw adoptions, truncations and deletions; the report is marked performed.
    """

    processed = [action for action in report.actions if action.kind == "processed"]
    raw = [action for action in report.actions if action.kind == "raw"]
    for action in processed + raw:
        if action.action == "leave":
            log.warning("%s: leaving %s alone (%s)", action.source, action.folder, action.reason)
        elif action.action == "delete":
            log.warning("%s: deleting %s (%s)", action.source, action.folder, action.reason)
            shutil.rmtree(action.folder)
        elif action.action == "truncate":
            log.warning("%s: truncating %s (%s)", action.source, action.folder, action.reason)
            _truncate_raw(action)
        elif action.action == "adopt":
            log.warning("%s: keeping %s under the new tokenizer (%s)", action.source, action.folder, action.reason)
            _adopt_raw(action)
        else:
            log.warning("%s: renaming %s into place (%s)", action.source, action.folder, action.reason)
            action.folder.rename(action.folder.with_name(action.folder.name.removesuffix(".tmp")))
    report.performed = True


def _truncate_raw(action: RepairAction) -> None:
    manifest = Manifest.load(action.folder)
    if manifest is None or action.keep_shards is None:
        raise RepairError(f"{action.source}: {action.folder} has no manifest to truncate")
    RawFolder(action.folder, manifest).truncate_to(action.keep_shards)


def _adopt_raw(action: RepairAction) -> None:
    """
    Re-label the raw manifest with the config's token_count / tokenizer / tokenizer_hash (action.adopt) and log
    the switch under extra["tokenizer_changes"] with the row count it happened at: the rows up to there carry
    counts made under the old tokenizer, later downloads count with the new one.
    """

    manifest = Manifest.load(action.folder)
    if manifest is None or action.adopt is None:
        raise RepairError(f"{action.source}: {action.folder} has no manifest to adopt")
    before = {key: getattr(manifest, key) for key in action.adopt}
    manifest.extra.setdefault("tokenizer_changes", []).append({"from": before, "to": dict(action.adopt), "at_rows": manifest.rows()})
    for key, value in action.adopt.items():
        setattr(manifest, key, value)
    manifest.save(action.folder)
