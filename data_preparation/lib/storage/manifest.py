# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""``MANIFEST.json`` beside a shard directory: what the directory contains and which config built it.

Every stage directory (``dataset/sources/<source>/raw/``, ``dataset/processed/<source>/``, tokenizers) carries one
manifest. ``source_hash`` is the stage's key from the config that produced it (:meth:`DatasetConfig.raw_hash` for
``raw/`` — loader identity plus token settings, so processing changes never invalidate downloads —,
:meth:`DatasetConfig.processed_hash` for ``processed/``, ``tokenizer_hash`` for tokenizers); a manifest whose hash
differs from the current config is stale: a processed folder is rebuilt, a raw folder is an error until the repair
step deletes it after confirmation (raw is never re-downloaded silently). A raw manifest also records
``truncated_at_tokens`` (the ``max_seq_length`` its texts were cut at): raising the cap above it makes the folder
*outdated* (:meth:`Manifest.is_outdated`), lowering it never does. Verification is cheap (parquet metadata only).
"""

from __future__ import annotations

import contextlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from data_preparation.lib.log import get_logger
from data_preparation.lib.storage.atomic import write_atomically

log = get_logger(__name__)

MANIFEST_NAME = "MANIFEST.json"
Stage = Literal["raw", "processed", "tokenizer"]
STAGES: tuple[str, ...] = ("raw", "processed", "tokenizer")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class ShardInfo:
    name: str
    rows: int
    tokens: int | None = None
    offset: int | None = None  # raw shards: the loader offset (source rows consumed) right after this shard's last row
    # Raw shards: the folder's reject totals right after this shard's last row, so a truncation to the good prefix
    # restores them along with `offset` (`lib/storage/raw_folder.py`). None on processed shards and on raw shards
    # written before these fields existed (a truncation then restarts the counters at 0, with a log line).
    skipped_malformed: int | None = None
    dropped_too_long: int | None = None


@dataclass
class Manifest:
    source: str
    source_hash: str
    stage: str
    rows_fetched: int = 0
    shards: list[ShardInfo] = field(default_factory=list)
    token_count: str | None = None
    tokenizer: str | None = None
    # raw manifests: texts were truncated to this many tokens at download time (the config's `max_seq_length` then);
    # None for processed / tokenizer manifests and for raw folders downloaded before truncation existed
    truncated_at_tokens: int | None = None
    versions: dict[str, str] = field(default_factory=dict)
    created: str = field(default_factory=_utc_now_iso)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ValueError(f"unknown manifest stage {self.stage!r}; expected one of {STAGES}")

    # --- derived -----------------------------------------------------------------------------------------------------

    def rows(self) -> int:
        return sum(shard.rows for shard in self.shards)

    def tokens(self) -> int | None:
        """Total measured tokens, or None if any shard has no token count."""
        total = 0
        for shard in self.shards:
            if shard.tokens is None:
                return None
            total += shard.tokens
        return total

    def is_current(self, source_hash: str) -> bool:
        return self.source_hash == source_hash

    def is_outdated(self, max_seq_length: int) -> bool:
        """Whether a raw folder was stored with a smaller cap than the config asks for now.

        Cap rule: rows are at most ``truncated_at_tokens`` long, so raising ``max_seq_length`` above it outdates the
        folder (its texts are missing tokens the config now wants; it is re-downloaded after confirmation), while
        lowering it never does (the build clamps stored counts, training truncates at ``block_size`` anyway). A
        manifest without ``truncated_at_tokens`` is never outdated by this rule."""
        return self.truncated_at_tokens is not None and max_seq_length > self.truncated_at_tokens

    def add_shard(
        self,
        name: str,
        rows: int,
        tokens: int | None = None,
        offset: int | None = None,
        skipped_malformed: int | None = None,
        dropped_too_long: int | None = None,
    ) -> None:
        """Record a shard; an existing entry with the same name is replaced. Shards are kept sorted by name."""
        self.shards = [shard for shard in self.shards if shard.name != name]
        self.shards.append(
            ShardInfo(
                name=name, rows=rows, tokens=tokens, offset=offset,
                skipped_malformed=skipped_malformed, dropped_too_long=dropped_too_long,
            )
        )
        self.shards.sort(key=lambda shard: shard.name)

    # --- (de)serialisation -------------------------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Manifest:
        """Build from a JSON dict; unknown keys (from newer versions) are ignored."""
        kwargs = _known_fields_only(payload, cls)
        raw_shards: list[dict[str, Any]] = kwargs.get("shards", [])
        kwargs["shards"] = [ShardInfo(**_known_fields_only(shard, ShardInfo)) for shard in raw_shards]
        return cls(**kwargs)

    def save(self, directory: Path) -> Path:
        """Write ``directory/MANIFEST.json`` atomically (:func:`write_atomically`) and return its path."""
        path = directory / MANIFEST_NAME
        with write_atomically(path) as tmp:
            tmp.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        log.debug("wrote %s (%d shards, %d rows)", path, len(self.shards), self.rows())
        return path

    @classmethod
    def load(cls, directory: Path) -> Manifest | None:
        """The manifest in ``directory``, or None if absent. An unparsable manifest is only ignored (with a warning)
        when the directory holds no shards; next to shards it is an error — treating it as absent would make the
        next build start from shard 0 and delete data that may have been expensive to download."""
        path = directory / MANIFEST_NAME
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text())
            if not isinstance(payload, dict):
                raise TypeError("manifest is not a JSON object")
            return cls.from_dict(payload)
        except (ValueError, TypeError) as err:  # json errors are ValueErrors; bad fields raise TypeError/ValueError
            if has_shards(directory):
                raise RuntimeError(f"unreadable manifest {path} next to shards ({err}); fix it or delete the directory") from err
            log.warning("ignoring unparsable manifest %s: %s", path, err)
            return None


def has_shards(directory: Path) -> bool:
    """Whether ``directory`` holds any ``data-*.parquet`` shard."""
    return any(directory.glob("data-*.parquet"))


def _known_fields_only(payload: dict[str, Any], dataclass_type: type) -> dict[str, Any]:
    """``payload`` restricted to the field names of ``dataclass_type``."""
    known = {f.name for f in fields(dataclass_type)}
    return {key: value for key, value in payload.items() if key in known}


# --- shard helpers ---------------------------------------------------------------------------------------------------


def shard_rows(path: Path) -> int:
    """Row count of a parquet file from its footer metadata (no data read)."""
    return pq.read_metadata(path).num_rows


def shard_tokens(path: Path) -> int:
    """Sum of a shard's ``tokens`` column (one column read)."""
    column = pq.read_table(path, columns=["tokens"]).column("tokens")
    total = pc.sum(column).as_py()
    return 0 if total is None else int(total)


def verify_shards(directory: Path, manifest: Manifest) -> list[str]:
    """Problems between ``manifest`` and the files in ``directory``: missing shards, row-count mismatches."""
    problems: list[str] = []
    for shard in manifest.shards:
        problem = shard_problem(directory, shard)
        if problem is not None:
            problems.append(problem)
    return problems


def shard_problem(directory: Path, shard: ShardInfo) -> str | None:
    """Why ``shard`` does not match its file in ``directory`` (missing, unreadable, row count), or None."""
    path = directory / shard.name
    if not path.is_file():
        return f"missing shard {shard.name}"
    try:
        rows = shard_rows(path)
    except (OSError, pa.ArrowException) as err:
        return f"unreadable shard {shard.name}: {err}"
    if rows != shard.rows:
        return f"shard {shard.name}: manifest says {shard.rows} rows, file has {rows}"
    return None


def library_versions() -> dict[str, str]:
    """Python, pyarrow, datasets/transformers (if installed) and the repo git sha (if available)."""
    versions = {"python": platform.python_version(), "pyarrow": pa.__version__}
    for package in ("datasets", "transformers"):
        with contextlib.suppress(importlib_metadata.PackageNotFoundError):
            versions[package] = importlib_metadata.version(package)
    sha = _git_sha()
    if sha is not None:
        versions["git"] = sha
    return versions


def _git_sha() -> str | None:
    """HEAD commit of this repo, or None when git is unavailable or the tree is not a checkout."""
    repo_root = Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


__all__ = [
    "MANIFEST_NAME",
    "Manifest",
    "ShardInfo",
    "Stage",
    "library_versions",
    "has_shards",
    "shard_problem",
    "shard_rows",
    "shard_tokens",
    "verify_shards",
]
