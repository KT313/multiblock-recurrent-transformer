# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Restore, stage, and publish the processed output of a source-local build."""
from __future__ import annotations

import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa

from data_preparation.lib.dataset_config import DatasetConfig
from data_preparation.lib.layout import processed_columns
from data_preparation.lib.build.assessment import ProcessedAssessment, assess_processed_folder
from data_preparation.lib.log import get_logger
from data_preparation.lib.stages.download import new_manifest
from data_preparation.lib.stages.exact_dedup import stored_hashes
from data_preparation.lib.storage.manifest import Manifest, ShardInfo, shard_list
from data_preparation.lib.storage.ownership import BuildWorkspace, guarded_path
from data_preparation.lib.storage.parquet import publish_shard, shard_name

log = get_logger("data_preparation.lib.stages.build")
Row = dict[str, Any]


@dataclass
class ProcessedOutput:
    """
    The processed folder of one build: its manifest, the directory shards are published into (the final folder,
    or the owned temporary slot of an all-at-once build) and whether the manifest was created by this call (a new
    manifest is saved even when nothing is appended, so an empty source still counts as built).
    """

    manifest: Manifest
    directory: Path
    is_new: bool

    @classmethod
    def resume(cls, config: DatasetConfig, name: str, source_hash: str, processed_dir: Path, assessment: ProcessedAssessment) -> ProcessedOutput:
        """
        The stored manifest if new raw shards can be appended to it (the shared verdict says built or behind
        raw: current hash, expected columns, covered shards a prefix of the raw shards); otherwise a fresh one, and
        the folder is deleted first, so no shard of the previous build survives unlisted. A manifest that cannot be
        parsed never gets here (:func:`build_source` refuses it before choosing a path).
        """

        if assessment.problem in ("none", "behind_raw") and assessment.manifest is not None:
            return cls(assessment.manifest, processed_dir, is_new=False)
        if assessment.problem != "absent":
            log.warning("%s: processed %s, rebuilding everything", name, assessment.reason)
        if processed_dir.exists():
            log.info("%s: removing %s before the rebuild", name, processed_dir)
            guarded_path(processed_dir.parent.parent, processed_dir)
            shutil.rmtree(processed_dir)
        return cls(_fresh_manifest(config, name, source_hash), processed_dir, is_new=True)

    def covered(self) -> int:
        """
        Raw shards the manifest already covers.
        """

        return len(self.manifest.input_shards)

    def stored_hashes(self) -> Iterator[int]:
        """
        The exact-dedup keys of every processed row on disk, in manifest order (refills the dedup filter).
        """

        return stored_hashes(self.directory / shard.name for shard in self.manifest.shards)

    def publish(self, rows: list[Row], shard_size: int) -> None:
        """
        Append rows as shard(s) of at most shard_size rows, each recorded in the manifest.
        """

        if self.manifest.generation_complete or self.manifest.generation_id is None:
            self.manifest.begin_generation(self.directory)
        for start in range(0, len(rows), shard_size):
            chunk = rows[start : start + shard_size]
            path = publish_shard(pa.Table.from_pylist(chunk), self.directory / shard_name(len(self.manifest.shards)))
            self.manifest.add_shard(path.name, len(chunk), sum(int(row["tokens"]) for row in chunk))

    def save(self, covered: list[list[Any]]) -> None:
        self.manifest.input_shards = list(covered)
        self.manifest.save(self.directory)
        self.is_new = False


def _fresh_manifest(config: DatasetConfig, name: str, source_hash: str) -> Manifest:
    source = config.sources[name]
    processing = config.source_processing(name)
    manifest = new_manifest(config, name, source_hash, "processed", tokens=True, hash_payload=config.processed_hash_payload(name))
    stats: dict[str, Any] = {
        "input_rows": 0,
        "dedup": {"mode": processing.dedup.mode, "duplicates_removed": 0},
    }
    if source.kind == "pretrain":
        stats["length_filter"] = {"input_samples": 0, "removed_too_short": 0, "removed_invalid": 0, "output_samples": 0}
        stats["quality_filter"] = {"enabled": processing.quality_filter, "filtered_count": 0, "rejection_reasons": {}}
        stats["decontamination"] = {
            "enabled": processing.decontamination.enabled, "contaminated_count": 0, "contaminated_by_benchmark": {},
        }  # fmt: skip
    else:
        stats["inverted"] = 0  # rows replaced by their input inversion (`source.input_inversions` share, seeded per row)
        stats["removed_empty"] = 0  # rows without instruction or output after stripping
        stats["removed_too_long"] = 0  # rows over `dataset_max_sequence_length` tokens (a safety net; the download already drops them)
    manifest.columns = list(processed_columns(source.kind))
    manifest.shuffled = config.shuffle_of(name)
    manifest.shuffle_seed = source.seed
    manifest.stats = stats
    return manifest


def prepare_source_output(
    config: DatasetConfig, name: str, source_hash: str, raw: Manifest, processed_dir: Path, *, all_at_once: bool,
) -> Manifest | tuple[ProcessedOutput, list[ShardInfo]]:
    """Return completed output immediately, or restore the output and pending shards to process."""
    assessment = assess_processed_folder(config, name, processed_dir, shard_list(raw.shards), check_files=False)
    if assessment.problem == "unreadable_manifest":  # only the confirmed repair step may delete it
        raise RuntimeError(
            f"{name}: {processed_dir / 'MANIFEST.json'} cannot be parsed; the repair step deletes the folder after "
            "confirmation (`prepare` asks, `--yes` answers), or fix or delete it by hand"
        )

    if all_at_once:
        BuildWorkspace(processed_dir).check_start()
        if assessment.problem == "none" and assessment.manifest is not None:
            if not assessment.manifest.generation_complete:
                assessment.manifest.complete_generation(processed_dir)
            return assessment.manifest  # built from exactly the current raw shards
        config.check_all_at_once_rows(name, raw.rows(), at_build=True)  # check actual rows before allocating the filter/pool
        output = ProcessedOutput(_fresh_manifest(config, name, source_hash), BuildWorkspace(processed_dir).path("temporary"), is_new=True)
        pending = list(raw.shards)
    else:
        output = ProcessedOutput.resume(config, name, source_hash, processed_dir, assessment)
        pending = raw.shards[output.covered() :]
        if not pending:
            if output.is_new:
                output.save([])  # an exhausted raw folder with zero shards still gets its processed manifest
            if output.manifest.generation_id is None or not output.manifest.generation_complete:
                output.manifest.complete_generation(output.directory)
            return output.manifest

    return output, pending
