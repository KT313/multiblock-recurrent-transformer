# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Owned, recoverable build slots outside the namespace of final source identities."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from data_preparation.identifiers import validate_identifier
from data_preparation.lib.storage.atomic import write_atomically
from data_preparation.lib.storage.manifest import Manifest

Role = Literal["temporary", "backup"]
OWNER_NAME = "BUILD_OWNER.json"


class OwnershipError(RuntimeError):
    """A path cannot be proved safe to mutate; its contents must be preserved."""


def guarded_path(root: Path, target: Path) -> Path:
    """Allow a root alias, but reject symlinks in every child component before mutation."""
    root_absolute = root.absolute()
    target_absolute = target.absolute()
    try:
        relative = target_absolute.relative_to(root_absolute)
    except ValueError as error:
        raise OwnershipError(f"{target}: outside protected dataset root {root}") from error
    if not relative.parts or ".." in relative.parts:
        raise OwnershipError(f"{target}: refusing mutation of the dataset root or a traversal path")
    current = root_absolute
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise OwnershipError(f"{target}: unexpected child symlink {current}; preserving contents")
    if not target_absolute.resolve().is_relative_to(root_absolute.resolve()):
        raise OwnershipError(f"{target}: resolves outside protected dataset root {root}")
    return target


@dataclass(frozen=True)
class BuildWorkspace:
    """One final directory and two owned slots; metadata records each build's identity."""

    final: Path

    def __post_init__(self) -> None:
        validate_identifier(self.final.name, field="source name")
        if self.final.parent.name not in ("processed", "tokenizers"):
            raise OwnershipError(f"{self.final}: expected a final processed/tokenizers directory")

    @property
    def root(self) -> Path:
        return self.final.parent.parent

    def path(self, role: Role) -> Path:
        return self.root / ".build-work" / self.final.parent.name / self.final.name / role

    def legacy_path(self, role: Role) -> Path:
        return self.final.with_name(self.final.name + (".tmp" if role == "temporary" else ".old"))

    def check(self, path: Path) -> None:
        guarded_path(self.root, path)

    def evidence(self, path: Path, role: Role) -> str:
        """Validate explicit metadata, or a legacy processed manifest's source identity."""
        self.check(path)
        if path not in (self.path(role), self.legacy_path(role)):
            raise OwnershipError(f"{path}: not the {role} slot for {self.final}")
        marker = path / OWNER_NAME
        self.check(marker)
        if marker.exists():
            return self._read_owner(path, role)
        if path == self.legacy_path(role) and self.final.parent.name == "processed":
            self.check(path / "MANIFEST.json")
            try:
                manifest = Manifest.load(path)
            except (RuntimeError, OSError, ValueError, TypeError) as error:
                raise OwnershipError(f"{path}: unreadable legacy ownership; preserving contents") from error
            if manifest is not None and manifest.source == self.final.name and manifest.stage == "processed":
                return "legacy"
        raise OwnershipError(f"{path}: no ownership evidence for {self.final.name!r}; preserving ambiguous artifact")

    def _read_owner(self, path: Path, role: Role) -> str:
        marker = path / OWNER_NAME
        self.check(marker)
        try:
            payload = json.loads(marker.read_text())
            if (
                payload["version"] != 1 or payload["source"] != self.final.name
                or payload["final"] != str(self.final.resolve()) or payload["role"] != role
            ):
                raise ValueError("owner fields do not match")
            build_id = payload["build_id"]
            if not isinstance(build_id, str):
                raise ValueError("build_id is not a string")
            UUID(build_id)
            return build_id
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise OwnershipError(f"{path}: malformed or conflicting build ownership; preserving contents") from error

    def verify_replacement(self, backup: Path, replacement: Path) -> None:
        """A new backup may only be discarded for the generation that parked it aside."""
        build_id = self.evidence(backup, "backup")
        if build_id == "legacy":
            return  # legacy manifests prove source identity, but did not record transactions
        if self._read_owner(replacement, "temporary") != build_id:
            raise OwnershipError(f"{backup}: replacement belongs to another build; preserving both generations")

    def check_start(self) -> None:
        """Refuse unresolved swaps before building; an owned incomplete temporary can restart."""
        self.check(self.final)
        for role in ("temporary", "backup"):
            legacy = self.legacy_path(role)
            self.check(legacy)
            if legacy.exists():
                raise OwnershipError(f"{legacy}: unresolved legacy artifact; run repair before building")
        backup = self.path("backup")
        self.check(backup)
        if backup.exists():
            raise OwnershipError(f"{backup}: unresolved backup; run repair before building")
        temporary = self.path("temporary")
        self.check(temporary)
        if temporary.exists():
            self.evidence(temporary, "temporary")

    def mark(self, path: Path, role: Role, build_id: str) -> None:
        self.check(path / OWNER_NAME)
        self.check(path / (OWNER_NAME + ".tmp"))
        with write_atomically(path / OWNER_NAME) as temporary:
            temporary.write_text(json.dumps({
                "version": 1, "source": self.final.name, "final": str(self.final.resolve()),
                "role": role, "build_id": build_id,
            }) + "\n")

    def create_temporary(self) -> Path:
        path = self.path("temporary")
        self.check(path)
        if path.exists():
            raise OwnershipError(f"{path}: existing build artifact; run repair before starting another build")
        path.mkdir(parents=True)
        self.mark(path, "temporary", str(uuid4()))
        return path

    def remove(self, path: Path, role: Role) -> None:
        self.evidence(path, role)
        shutil.rmtree(path)

    def publish(self, temporary: Path) -> None:
        build_id = self.evidence(temporary, "temporary")
        self.check(self.final)
        backup = self.path("backup")
        self.check(backup)
        for role in ("temporary", "backup"):
            legacy = self.legacy_path(role)
            if legacy.exists() or legacy.is_symlink():
                raise OwnershipError(f"{legacy}: unresolved legacy artifact; run repair before publication")
        if backup.exists():
            raise OwnershipError(f"{backup}: unresolved backup; run repair before publication")
        if self.final.exists():
            self.mark(self.final, "backup", build_id)
            backup.parent.mkdir(parents=True, exist_ok=True)
            self.final.rename(backup)
        self.final.parent.mkdir(parents=True, exist_ok=True)
        temporary.rename(self.final)
        if backup.exists():
            self.remove(backup, "backup")
