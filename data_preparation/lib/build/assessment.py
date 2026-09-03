# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""
One health verdict per processed/<source> folder, with the cheapest repair that heals it attached.

The repair step, the planner, the build's own resume and the training-side verifier all judge processed folders;
:func:`assess_processed_folder` is the one place that knows what a processed folder can be, so they cannot
disagree. Every verdict carries the cheapest repair: rebuild (delete the folder, the build writes it again;
derived data, no confirmation) or nothing.

crash_leftover is the one anomaly whose repair is *nothing*: a build that crashes between publishing a shard
and saving the manifest leaves exactly one unlisted file, data-{len(manifest.shards):05d}.parquet, the name the
resumed build publishes next. While raw shards are still uncovered, the resume rewrites that file and the folder
heals itself. The same stray on a folder that covers every raw shard is a rebuild: no build would overwrite it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from data_preparation.dataset_config import DatasetConfig
from data_preparation.layout import processed_columns
from data_preparation.lib.storage.manifest import Manifest, has_shards, shard_problem
from data_preparation.lib.storage.parquet import list_parquet_files, shard_name

ShardList = list[list[Any]]  # [[shard name, rows], ...], the shape of a processed manifest's input_shards

ProcessedProblem = Literal[
    "none",  # healthy: current manifest, every listed shard verifies, nothing unlisted, raw prefix intact
    "absent",  # no manifest and no shards: nothing built yet
    "crash_leftover",  # one stray file == the next shard the resumed build writes; the build overwrites it (M1)
    "unreadable_manifest",  # MANIFEST.json exists but cannot be parsed
    "no_manifest",  # shard files without a manifest
    "raw_deleted",  # the raw folder the shards were built from is being deleted
    "stale",  # manifest hash != the config's processed hash, another stage's manifest, or other columns
    "broken_shard",  # a listed shard is missing, unreadable, or has the wrong row count
    "stray_shards",  # unlisted shard file(s) no resumed build would overwrite
    "raw_changed",  # input_shards is no longer a prefix of the raw shard list
    "behind_raw",  # current, but raw shards remain to be built (a pending build, not a repair)
]

CheapestRepair = Literal["nothing", "rebuild"]

# the problems a rebuild heals; the others are healthy, not built yet, pending, or healed by the resumed build
_REBUILD: frozenset[ProcessedProblem] = frozenset(
    {"unreadable_manifest", "no_manifest", "raw_deleted", "stale", "broken_shard", "stray_shards", "raw_changed"}
)


@dataclass(frozen=True)
class ProcessedAssessment:
    """
    The health of one processed folder: the problem, one reason line (the tail of processed <reason> in the
    status table), and the manifest when it was readable. :attr:`repair` derives from the problem.
    """

    problem: ProcessedProblem
    reason: str
    manifest: Manifest | None

    @property
    def repair(self) -> CheapestRepair:
        """
        The cheapest repair that heals the folder: rebuild = delete it and build again (derived data, no
        confirmation); nothing = healthy, not built yet, pending, or the resumed build heals it by itself.
        """

        return "rebuild" if self.problem in _REBUILD else "nothing"


def next_shard_to_write(manifest: Manifest) -> str:
    """
    The file name the next :meth:`~data_preparation.lib.stages.build.ProcessedOutput.publish` call writes:
    shards are named by their index, so it is data-{len(manifest.shards):05d}.parquet. This is the one stray a
    crash between publishing a shard and saving the manifest leaves behind.
    """

    return shard_name(len(manifest.shards))


def assess_processed_folder(
    config: DatasetConfig, name: str, folder: Path, raw_shards: ShardList | None, *, check_files: bool = True
) -> ProcessedAssessment:
    """
    The verdict on processed/<name> at folder, given the raw shards it will be able to build from as
    [[name, rows], ...] (None: the raw folder is being deleted).

    check_files=True (the repair step) also verifies the listed shard files (parquet footers) and looks for
    unlisted ones. check_files=False (the planner, the build) reads the manifest only, so broken_shard,
    stray_shards and crash_leftover are never reported: broken files are the repair step's business.
    """

    try:
        manifest = Manifest.load(folder)
    except RuntimeError:
        return ProcessedAssessment("unreadable_manifest", "unreadable manifest", None)
    if manifest is None:
        if has_shards(folder):
            return ProcessedAssessment("no_manifest", "no manifest", None)
        return ProcessedAssessment("absent", "missing", None)
    if raw_shards is None:
        return ProcessedAssessment("raw_deleted", "built from a raw folder that is being deleted", manifest)
    if manifest.stage != "processed":
        return ProcessedAssessment("stale", f"stale: a {manifest.stage} manifest where a processed one belongs", manifest)
    if not manifest.is_current(config.processed_hash(name)):
        return ProcessedAssessment("stale", "stale: processing settings, max_seq_length or the source changed", manifest)
    if manifest.columns != list(processed_columns(config.sources[name].kind)):
        return ProcessedAssessment("stale", "stale: the shards predate the current columns", manifest)
    covered = manifest.input_shards
    if check_files:
        for shard in manifest.shards:
            problem = shard_problem(folder, shard)
            if problem is not None:
                return ProcessedAssessment("broken_shard", f"broken: {problem}", manifest)
        listed = {shard.name for shard in manifest.shards}
        unlisted = sorted(path.name for path in list_parquet_files(folder) if path.name not in listed)
        if unlisted == [next_shard_to_write(manifest)] and _more_raw_follows(covered, raw_shards):
            reason = f"crash leftover {unlisted[0]}: the next shard the resumed build writes; the build overwrites it"
            return ProcessedAssessment("crash_leftover", reason, manifest)
        if unlisted:
            return ProcessedAssessment("stray_shards", f"unlisted shard(s): {', '.join(unlisted)}", manifest)
    if raw_shards[: len(covered)] != covered:
        return ProcessedAssessment("raw_changed", "built from raw shards that no longer exist", manifest)
    if len(covered) < len(raw_shards):
        return ProcessedAssessment("behind_raw", "behind raw", manifest)
    return ProcessedAssessment("none", "ok", manifest)


def _more_raw_follows(covered: ShardList, raw_shards: ShardList) -> bool:
    """
    Whether the covered raw shards are a *proper* prefix of the raw shard list: a resumed build has work left,
    so its next publish overwrites the crash leftover. A folder that covers everything gets no further publish.
    """

    return len(covered) < len(raw_shards) and raw_shards[: len(covered)] == covered
