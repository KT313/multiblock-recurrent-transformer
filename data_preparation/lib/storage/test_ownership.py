# (c) 2025-2026 Tobias Kerner. Apache-2.0.
"""Ownership checks must preserve every rejected artifact's contents."""

import json
from pathlib import Path

import pytest

from data_preparation.lib.storage.manifest import Manifest
from data_preparation.lib.storage.ownership import BuildWorkspace, OWNER_NAME, OwnershipError, guarded_path


def test_owned_partial_build_can_be_removed(tmp_path: Path) -> None:
    workspace = BuildWorkspace(tmp_path / "processed" / "Books")
    partial = workspace.create_temporary()
    (partial / "partial").write_text("incomplete")
    assert workspace.evidence(partial, "temporary")
    workspace.remove(partial, "temporary")
    assert not partial.exists()


@pytest.mark.parametrize("problem", ["missing", "malformed", "wrong_source", "wrong_role", "wrong_final", "wrong_build_id"])
def test_invalid_owner_preserves_partial_build(tmp_path: Path, problem: str) -> None:
    workspace = BuildWorkspace(tmp_path / "processed" / "a")
    artifact = workspace.create_temporary()
    (artifact / "sentinel").write_bytes(b"keep")
    marker = artifact / OWNER_NAME
    owner = json.loads(marker.read_text())
    if problem == "missing":
        marker.unlink()
    elif problem == "malformed":
        marker.write_text("[]")
    else:
        owner[{"wrong_source": "source", "wrong_role": "role", "wrong_final": "final", "wrong_build_id": "build_id"}[problem]] = "wrong"
        marker.write_text(json.dumps(owner))
    with pytest.raises(OwnershipError):
        workspace.remove(artifact, "temporary")
    assert (artifact / "sentinel").read_bytes() == b"keep"


def test_legacy_manifest_identifies_source_and_stage(tmp_path: Path) -> None:
    workspace = BuildWorkspace(tmp_path / "processed" / "a")
    artifact = workspace.legacy_path("backup")
    artifact.mkdir(parents=True)
    (artifact / "sentinel").write_bytes(b"keep")
    for source, stage in (("a.old", "processed"), ("b", "processed"), ("a", "raw")):
        Manifest(source=source, stage=stage, source_hash="old").save(artifact)
        with pytest.raises(OwnershipError):
            workspace.remove(artifact, "backup")
        assert (artifact / "sentinel").read_bytes() == b"keep"
    Manifest(source="a", stage="processed", source_hash="old").save(artifact)
    workspace.remove(artifact, "backup")
    assert not artifact.exists()


def test_root_alias_allowed_but_child_symlink_rejected(tmp_path: Path) -> None:
    root = tmp_path / "real"
    root.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    workspace = BuildWorkspace(alias / "processed" / "a")
    temporary = workspace.create_temporary()
    workspace.remove(temporary, "temporary")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "sentinel").write_bytes(b"keep")
    temporary.symlink_to(outside, target_is_directory=True)
    with pytest.raises(OwnershipError, match="symlink"):
        workspace.remove(temporary, "temporary")
    assert (outside / "sentinel").read_bytes() == b"keep"
    with pytest.raises(OwnershipError, match="outside|traversal"):
        guarded_path(root, root / ".." / "outside")
