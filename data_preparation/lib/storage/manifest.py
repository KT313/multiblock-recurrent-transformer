# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""``MANIFEST.json`` beside a shard directory: what a source/mixture directory contains and which config built it.

Every stage directory (``dataset/sources/<source>/{raw,filtered,processed}/``, mixtures, holdouts, tokenizers) carries
one manifest. ``source_hash`` is :meth:`DatasetConfig.source_hash` of the config that produced it; a manifest whose hash
differs from the current config is stale and its stage is rebuilt. Verification is cheap (parquet metadata only).
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
import pyarrow.parquet as pq

from data_preparation.lib.log import get_logger

log = get_logger(__name__)

MANIFEST_NAME = "MANIFEST.json"
Stage = Literal["raw", "filtered", "processed", "mixture", "holdout", "tokenizer"]
STAGES: tuple[str, ...] = ("raw", "filtered", "processed", "mixture", "holdout", "tokenizer")


@dataclass
class ShardInfo:
    name: str
    rows: int
    tokens: int | None = None


@dataclass
class Manifest:
    source: str
    source_hash: str
    stage: str
    rows_fetched: int = 0
    shards: list[ShardInfo] = field(default_factory=list)
    token_count: str | None = None
    tokenizer: str | None = None
    versions: dict[str, str] = field(default_factory=dict)
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ValueError(f"unknown manifest stage {self.stage!r}; expected one of {STAGES}")

    # --- derived -----------------------------------------------------------------------------------------------------

    def rows(self) -> int:
        return sum(s.rows for s in self.shards)

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

    def add_shard(self, name: str, rows: int, tokens: int | None = None) -> None:
        """Record a shard; an existing entry with the same name is replaced."""
        self.shards = [s for s in self.shards if s.name != name]
        self.shards.append(ShardInfo(name=name, rows=rows, tokens=tokens))
        self.shards.sort(key=lambda s: s.name)

    # --- (de)serialisation -------------------------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Manifest:
        """Build from a JSON dict; unknown keys (from newer versions) are ignored."""
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in payload.items() if k in known}
        shard_keys = {f.name for f in fields(ShardInfo)}
        kwargs["shards"] = [
            ShardInfo(**{k: v for k, v in s.items() if k in shard_keys}) for s in kwargs.get("shards", [])
        ]
        return cls(**kwargs)

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / MANIFEST_NAME
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        tmp.replace(path)
        log.debug("wrote %s (%d shards, %d rows)", path, len(self.shards), self.rows())
        return path

    @classmethod
    def load(cls, directory: Path) -> Manifest | None:
        """The manifest in ``directory``, or None if absent or unparsable (logged as a warning)."""
        path = directory / MANIFEST_NAME
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text())
            if not isinstance(payload, dict):
                raise TypeError("manifest is not a JSON object")
            return cls.from_dict(payload)
        except (ValueError, TypeError) as err:  # json errors are ValueErrors; bad fields raise TypeError/ValueError
            log.warning("ignoring unparsable manifest %s: %s", path, err)
            return None


# --- shard helpers ---------------------------------------------------------------------------------------------------


def shard_rows(path: Path) -> int:
    """Row count of a parquet file from its footer metadata (no data read)."""
    return pq.read_metadata(path).num_rows


def verify_shards(directory: Path, manifest: Manifest) -> list[str]:
    """Problems between ``manifest`` and the files in ``directory``: missing shards, row-count mismatches."""
    problems: list[str] = []
    for shard in manifest.shards:
        path = directory / shard.name
        if not path.is_file():
            problems.append(f"missing shard {shard.name}")
            continue
        try:
            rows = shard_rows(path)
        except (OSError, pa.ArrowException) as err:
            problems.append(f"unreadable shard {shard.name}: {err}")
            continue
        if rows != shard.rows:
            problems.append(f"shard {shard.name}: manifest says {shard.rows} rows, file has {rows}")
    return problems


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
    "shard_rows",
    "verify_shards",
]
