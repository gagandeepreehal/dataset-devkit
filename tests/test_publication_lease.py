from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest

import dataset_devkit.publication as publication
from conftest import FeatureFactory
from dataset_devkit.config import GlobalConfig
from dataset_devkit.export import export_dataset
from dataset_devkit.publication import StagingLease, hash_regular_files_fd, publish_staging
from dataset_devkit.validation import DatasetValidationError, finalize_dataset
from test_export_dataset import _evidence


def _finalized_lease(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    feature_factory: FeatureFactory,
) -> tuple[StagingLease, str]:
    lease = StagingLease.create(tmp_path / "output", ".dataset.staging-")
    export_dataset(
        lease.root,
        _evidence(tmp_path / "input", config_factory, feature_factory),
        lease=lease,
    )
    report = finalize_dataset(lease.root, official_smoke=False, lease=lease)
    assert report.content_hash is not None
    return lease, report.content_hash


def test_staging_lease_cleanup_removes_nested_owned_tree(tmp_path: Path) -> None:
    lease = StagingLease.create(tmp_path / "output", ".dataset.staging-")
    nested = lease.root / "one/two"
    nested.mkdir(parents=True)
    (nested / "payload").write_bytes(b"owned")

    try:
        assert lease.cleanup()
        assert not lease.root.exists()
    finally:
        lease.close()


def test_staging_lease_cleanup_never_removes_same_name_replacement(
    tmp_path: Path,
) -> None:
    lease = StagingLease.create(tmp_path / "output", ".dataset.staging-")
    owned_file = lease.root / "owned"
    owned_file.write_bytes(b"owned")
    displaced = lease.root.with_name(f"{lease.root.name}.displaced")
    lease.root.rename(displaced)
    lease.root.mkdir()
    sentinel = lease.root / "keep"
    sentinel.write_bytes(b"unrelated")

    try:
        assert not lease.cleanup()
        assert sentinel.read_bytes() == b"unrelated"
        assert tuple(displaced.iterdir()) == ()
    finally:
        lease.close()


def test_displaced_validated_staging_replacement_is_never_published(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    feature_factory: FeatureFactory,
) -> None:
    lease, content_hash = _finalized_lease(tmp_path, config_factory, feature_factory)
    displaced = lease.root.with_name(f"{lease.root.name}.displaced")
    replacement = lease.root
    final = lease.parent / "v1.0-trainval"
    try:
        lease.root.rename(displaced)
        replacement.mkdir()
        sentinel = replacement / "unrelated"
        sentinel.write_text("must survive", encoding="utf-8")

        with pytest.raises(ValueError, match="leased root|entry"):
            publish_staging(lease, final, expected_content_hash=content_hash)

        assert not final.exists()
        assert sentinel.read_text(encoding="utf-8") == "must survive"
        assert displaced.is_dir()
    finally:
        lease.close()


def test_same_inode_same_size_rewrite_after_validation_blocks_publication(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    feature_factory: FeatureFactory,
) -> None:
    lease, content_hash = _finalized_lease(tmp_path, config_factory, feature_factory)
    final = lease.parent / "v1.0-trainval"
    target = lease.root / "mz_extensions/tags.json"
    original = target.read_bytes()
    changed = bytes([original[0] ^ 1]) + original[1:]
    assert len(changed) == len(original)
    try:
        descriptor = os.open(target, os.O_WRONLY)
        try:
            assert os.write(descriptor, changed) == len(changed)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

        with pytest.raises(DatasetValidationError, match="authorized publication manifest"):
            publish_staging(lease, final, expected_content_hash=content_hash)

        assert not final.exists()
    finally:
        lease.cleanup()
        lease.close()


def test_hash_walker_rejects_same_size_rewrite_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = root / "payload.bin"
    target.write_bytes(b"a" * (2 * 1024 * 1024))
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    original_read = os.read
    rewritten = False

    def racing_read(descriptor: int, size: int) -> bytes:
        nonlocal rewritten
        chunk = original_read(descriptor, size)
        if chunk and not rewritten:
            rewritten = True
            writer = os.open(target, os.O_WRONLY)
            try:
                os.pwrite(writer, b"b", 0)
                os.fsync(writer)
            finally:
                os.close(writer)
        return chunk

    monkeypatch.setattr("dataset_devkit.publication.os.read", racing_read)
    try:
        with pytest.raises(ValueError, match="changed while hashing"):
            hash_regular_files_fd(root_fd)
    finally:
        os.close(root_fd)


def test_hash_snapshot_rejects_rewrite_of_earlier_file_while_later_file_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    earlier = root / "a.bin"
    later = root / "b.bin"
    earlier.write_bytes(b"a" * (2 * 1024 * 1024))
    later.write_bytes(b"b" * (2 * 1024 * 1024))
    earlier_identity = (earlier.stat().st_dev, earlier.stat().st_ino)
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    original_read = os.read
    rewritten = False

    def racing_read(descriptor: int, size: int) -> bytes:
        nonlocal rewritten
        chunk = original_read(descriptor, size)
        current = os.fstat(descriptor)
        if chunk and not rewritten and (current.st_dev, current.st_ino) != earlier_identity:
            rewritten = True
            writer = os.open(earlier, os.O_WRONLY)
            try:
                os.pwrite(writer, b"z", 0)
                os.fsync(writer)
            finally:
                os.close(writer)
        return chunk

    monkeypatch.setattr("dataset_devkit.publication.os.read", racing_read)
    try:
        with pytest.raises(ValueError, match="changed while hashing: a.bin"):
            hash_regular_files_fd(root_fd)
        assert rewritten
    finally:
        os.close(root_fd)


def test_post_rename_rewrite_is_rejected_and_final_name_is_rolled_back(
    tmp_path: Path,
    config_factory: Callable[[], GlobalConfig],
    feature_factory: FeatureFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease, content_hash = _finalized_lease(tmp_path, config_factory, feature_factory)
    final = lease.parent / "v1.0-trainval"
    original_rename = publication._rename_exclusive

    def rewrite_after_rename(
        parent_fd: int, source: str, destination: str
    ) -> None:
        original_rename(parent_fd, source, destination)
        if destination != final.name:
            return
        target = final / "mz_extensions/tags.json"
        original = target.read_bytes()
        changed = bytes([original[0] ^ 1]) + original[1:]
        assert len(changed) == len(original)
        descriptor = os.open(target, os.O_WRONLY)
        try:
            assert os.write(descriptor, changed) == len(changed)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    monkeypatch.setattr(publication, "_rename_exclusive", rewrite_after_rename)
    try:
        with pytest.raises(DatasetValidationError, match="authorized publication manifest"):
            publish_staging(lease, final, expected_content_hash=content_hash)

        assert not final.exists()
        quarantined = list(lease.parent.glob(f".{final.name}.rejected-*"))
        assert len(quarantined) == 1
        assert (quarantined[0].stat().st_dev, quarantined[0].stat().st_ino) == (
            lease.root_identity
        )
    finally:
        lease.close()
