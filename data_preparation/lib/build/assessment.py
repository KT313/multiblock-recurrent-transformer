# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""One health verdict per ``processed/<source>`` folder, with the cheapest repair that heals it attached.

Several code paths judge a processed folder — the repair step (``lib/build/repair.py``), the planner
(``lib/build/planner.py``, behind ``status`` / ``prepare --dry_run`` / ``prepare``), the build's own resume
(``lib/stages/build.py``) and the training-side verifier. They used to judge for themselves and drifted (an
unreadable manifest crashed ``status`` while ``prepare`` healed it; a crash leftover made the repair step delete a
folder the resumed build would have healed). :func:`assess_processed_folder` is now the one place that knows what a
processed folder can be, and every verdict carries the cheapest repair: ``rebuild`` (delete the folder, the build
writes it again — derived data, no confirmation) or ``nothing``.

The one verdict whose cheapest repair is *nothing* despite a visible anomaly deserves its name: a build that
crashes between publishing a shard file and saving the manifest leaves exactly one unlisted file —
``data-{len(manifest.shards):05d}.parquet``, the very name the resumed build publishes next
(:meth:`~data_preparation.lib.stages.build.ProcessedOutput.publish` names shards by their index and
``publish_shard`` replaces atomically). While raw shards are still uncovered, the resume rewrites that file from
the same rows and the folder heals itself; deleting the whole folder for it (what the repair step used to do)
redoes hours of cleaning for one file. The same stray on a folder that covers every raw shard is *not* resumable —
no build would overwrite it, the training resolver would refuse the folder — so it stays a rebuild.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from data_preparation.dataset_config import DatasetConfig
from data_preparation.lib.storage.manifest import Manifest, has_shards, shard_problem
from data_preparation.lib.storage.parquet import list_parquet_files, shard_name

ShardList = list[list[Any]]  # ``[[shard name, rows], ...]`` — the shape of ``processed`` manifests' ``extra["input_shards"]``

ProcessedProblem = Literal[
    "none",  # healthy: current manifest, every listed shard verifies, nothing unlisted, raw prefix intact
    "absent",  # no manifest and no shards: nothing built yet
    "crash_leftover",  # one stray file == the next shard the resumed build writes; the build overwrites it (M1)
    "unreadable_manifest",  # MANIFEST.json exists but cannot be parsed
    "no_manifest",  # shard files without a manifest
    "raw_deleted",  # the raw folder the shards were built from is being deleted
    "stale",  # manifest hash != the config's processed hash
    "broken_shard",  # a listed shard is missing, unreadable, or has the wrong row count
    "stray_shards",  # unlisted shard file(s) no resumed build would overwrite
    "raw_changed",  # ``extra["input_shards"]`` is no longer a prefix of the raw shard list
]

Verdict = Literal["ok", "missing", "resumable", "stale", "broken"]
CheapestRepair = Literal["nothing", "rebuild"]

_VERDICTS: dict[ProcessedProblem, Verdict] = {
    "none": "ok",
    "absent": "missing",
    "crash_leftover": "resumable",
    "unreadable_manifest": "broken",
    "no_manifest": "broken",
    "raw_deleted": "broken",
    "stale": "stale",
    "broken_shard": "broken",
    "stray_shards": "broken",
    "raw_changed": "broken",
}


@dataclass(frozen=True)
class ProcessedAssessment:
    """The health of one processed folder: the fine-grained problem (for consumers with extra knowledge of their
    own), one reason line, and the manifest when it was readable. :attr:`verdict` and :attr:`repair` derive from
    the problem, so a verdict can never disagree with the repair attached to it."""

    problem: ProcessedProblem
    reason: str
    manifest: Manifest | None

    @property
    def verdict(self) -> Verdict:
        return _VERDICTS[self.problem]

    @property
    def repair(self) -> CheapestRepair:
        """The cheapest repair that heals the folder: ``rebuild`` = delete it and build again (derived data, no
        confirmation); ``nothing`` = healthy, not built yet, or the resumed build heals it by itself."""
        return "rebuild" if self.verdict in ("stale", "broken") else "nothing"


def next_shard_to_write(manifest: Manifest) -> str:
    """The file name the next :meth:`~data_preparation.lib.stages.build.ProcessedOutput.publish` call writes:
    shards are named by their index, so it is ``data-{len(manifest.shards):05d}.parquet`` — the one stray a crash
    between publishing a shard and saving the manifest leaves behind."""
    return shard_name(len(manifest.shards))


def assess_processed_folder(
    config: DatasetConfig, name: str, folder: Path, raw_shards: ShardList | None, *, check_files: bool = True
) -> ProcessedAssessment:
    """The verdict on ``processed/<name>`` at ``folder``, given the raw shards it will be able to build from as
    ``[[name, rows], ...]`` (None: the raw folder is being deleted).

    ``check_files=True`` (the repair step) also verifies the listed shard files (parquet footers) and looks for
    unlisted ones; ``check_files=False`` (the planner) reads the manifest only, so ``broken_shard``, ``stray_shards``
    and ``crash_leftover`` are never reported — broken files are the repair step's business.
    """
    try:
        manifest = Manifest.load(folder)
    except RuntimeError:
        return ProcessedAssessment("unreadable_manifest", "unreadable manifest", None)
    if manifest is None:
        if has_shards(folder):
            return ProcessedAssessment("no_manifest", "no manifest", None)
        return ProcessedAssessment("absent", "nothing built yet", None)
    if raw_shards is None:
        return ProcessedAssessment("raw_deleted", "built from a raw folder that is being deleted", manifest)
    if not manifest.is_current(config.processed_hash(name)):
        return ProcessedAssessment("stale", "stale: processing settings, max_seq_length or the source changed", manifest)
    covered: ShardList = manifest.extra.get("input_shards", [])
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
    return ProcessedAssessment("none", "ok", manifest)


def _more_raw_follows(covered: ShardList, raw_shards: ShardList) -> bool:
    """Whether the covered raw shards are a *proper* prefix of the raw shard list — a resumed build has work left,
    so its next publish overwrites the crash leftover. A folder that covers everything gets no further publish."""
    return len(covered) < len(raw_shards) and raw_shards[: len(covered)] == covered


__all__ = [
    "CheapestRepair",
    "ProcessedAssessment",
    "ProcessedProblem",
    "ShardList",
    "Verdict",
    "assess_processed_folder",
    "next_shard_to_write",
]
